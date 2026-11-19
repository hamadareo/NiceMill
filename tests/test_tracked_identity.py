"""Tracking engine (PFB3x3) identity: the teacher maps tracked people (P-labels) to students."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline.tracked_identity import TrackedIdentityOutput, TrackedIdentityRecord
from pfb_app.services.review import (TRACK_REDO, ApprovalError, BusyError, approve,
                                     confirm_identity, edit_stage_output, is_stale_chain,
                                     regenerate_from, save_tracked_identity)
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

STUDENTS = (("Pa", "A子"), ("Pb", "B太"), ("Pc", "C美"), ("Pd", "D郎"))
FOUR = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": "Pd"}


@pytest.fixture
def repo(tmp_path):
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), source="model"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def submission(repo, engine="pfb3x3", cameras=("cam1",), require_approval=False):
    aid = repo.create_assignment("課題", Kind.video, "紙タワーを作る課題", "ルーブリック",
                                 "standard", require_approval)
    meta = {"cameras": {c: f"/x/{c}.mp4" for c in cameras}, "start_sec": 0.0, "end_sec": None,
            "auto_transcribe": False}
    if engine:
        meta["engine"] = engine
    sid = repo.create_submission(aid, "G1", meta)
    subjects = {pid: repo.add_subject(sid, pid, name, "外見", {}) for pid, name in STUDENTS}
    return sid, subjects


def track(repo, sid, camera="cam1", labels=("P1", "P2", "P3", "P4"), defaults=None):
    prep = repo.latest_run(sid, Stage.prepare, camera) or save(
        repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    return save(repo, sid, Stage.track, {
        "people": len(labels), "labels": list(labels), "sheet": "/x/sheet.jpg",
        "default_names": defaults or {}, "default_source": "rule" if defaults else None},
        camera=camera, upstream=[prep.id])


def queued(repo, sid):
    return [(r["stage"], r["camera"], r["model_key"]) for r in repo.conn.execute(
        "SELECT * FROM jobs WHERE submission_id=? AND status='queued' ORDER BY id", (sid,))]


def message(e) -> str:
    return "\n".join(str(err.get("ctx", {}).get("error", err["msg"])) for err in e.value.errors())


# --- the schema ---

def test_record_allows_outside_and_needs_a_p_label():
    rec = TrackedIdentityRecord(participant_id=None, label="P12", display_name=None)
    assert rec.note == ""
    with pytest.raises(ValidationError):
        TrackedIdentityRecord(participant_id="Pa", label="A1", display_name="A子")
    with pytest.raises(ValidationError):
        TrackedIdentityOutput(participants=[], labels=[], track_run_id=1, extra=1)


# --- save_tracked_identity ---

def test_valid_mapping_is_a_teacher_identity_run_tied_to_the_track(repo):
    sid, _ = submission(repo)
    t = track(repo, sid)
    repo.set_identity_confirmed(sid, True)
    run = save_tracked_identity(repo, sid, "cam1", {"P3": "Pc", "P1": "Pb", "P2": "Pa", "P4": "Pd"})
    assert (run.stage, run.camera, run.source, run.upstream_run_ids) == (
        Stage.identity, "cam1", "teacher", [t.id])
    assert run.output["track_run_id"] == t.id and run.output["labels"] == ["P1", "P2", "P3", "P4"]
    assert [(p["label"], p["participant_id"], p["display_name"]) for p in run.output["participants"]] == [
        ("P1", "Pb", "B太"), ("P2", "Pa", "A子"), ("P3", "Pc", "C美"), ("P4", "Pd", "D郎")]
    assert run.warnings == []
    assert not repo.get_submission(sid)["identity_confirmed"]   # must be confirmed again
    assert repo.latest_run(sid, Stage.identity, "cam1").id == run.id


def test_duplicate_student_is_rejected(repo):
    sid, _ = submission(repo)
    track(repo, sid)
    with pytest.raises(ValidationError) as e:
        save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pa", "P3": "Pc", "P4": "Pd"})
    assert "同じ参加者が複数の人物に割り当てられています: A子" in message(e)
    assert repo.latest_run(sid, Stage.identity, "cam1") is None


def test_unknown_and_missing_labels_are_rejected(repo):
    sid, _ = submission(repo)
    track(repo, sid)
    with pytest.raises(ValidationError) as e:
        save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P9": "Pd"})
    assert "人物の追跡結果にない番号です: P9" in message(e)
    assert "人物の割り当てがありません: P4" in message(e)
    with pytest.raises(ValidationError) as e:
        save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": "Pc", "X4": "Pd"})
    assert repo.latest_run(sid, Stage.identity, "cam1") is None


def test_participant_must_belong_to_the_submission(repo):
    sid, _ = submission(repo)
    other, _ = submission(repo)
    repo.add_subject(other, "Pz", "Z男", "外見", {})
    track(repo, sid)
    with pytest.raises(ValidationError) as e:
        save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": "Pz"})
    assert "このグループに登録されていない参加者です: Pz" in message(e)


def test_more_people_than_students_maps_the_rest_to_outside(repo):
    sid, _ = submission(repo)
    track(repo, sid, labels=("P1", "P2", "P3", "P4", "P5", "P6"))
    run = save_tracked_identity(repo, sid, "cam1", {**FOUR, "P5": None, "P6": None})
    outside = [p for p in run.output["participants"] if p["participant_id"] is None]
    assert [(p["label"], p["display_name"]) for p in outside] == [("P5", None), ("P6", None)]
    assert run.warnings == []


def test_fewer_people_than_students_is_allowed_with_a_warning(repo):
    sid, _ = submission(repo)
    track(repo, sid, labels=("P1", "P2"))
    run = save_tracked_identity(repo, sid, "cam1", {"P1": "Pb", "P2": None})
    assert run.status == "ok"
    assert [p["participant_id"] for p in run.output["participants"]] == ["Pb", None]
    assert len(run.warnings) == 1 and "A子、C美、D郎" in run.warnings[0]
    assert "根拠不足" in run.warnings[0]


def test_each_camera_is_mapped_on_its_own(repo):
    sid, _ = submission(repo, cameras=("cam1", "cam2"))
    track(repo, sid, "cam1")
    track(repo, sid, "cam2", labels=("P1", "P2", "P3"))
    save_tracked_identity(repo, sid, "cam1", FOUR)
    run2 = save_tracked_identity(repo, sid, "cam2", {"P1": "Pa", "P2": "Pd", "P3": None})
    assert run2.camera == "cam2"   # the same student on both cameras is fine
    assert repo.latest_run(sid, Stage.identity, "cam1").output["participants"][0]["participant_id"] == "Pa"


def test_save_refusals(repo):
    sid, _ = submission(repo)
    with pytest.raises(ValidationError) as e:   # no track run yet
        save_tracked_identity(repo, sid, "cam1", FOUR)
    assert "人物の追跡結果がありません" in message(e)
    track(repo, sid)
    with pytest.raises(ValidationError):
        save_tracked_identity(repo, sid, "cam9", FOUR)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    with pytest.raises(BusyError):
        save_tracked_identity(repo, sid, "cam1", FOUR)
    paper, _ = submission(repo, engine=None)
    track(repo, paper)
    with pytest.raises(ValidationError) as e:
        save_tracked_identity(repo, paper, "cam1", FOUR)
    assert "トラッキング方式ではありません" in message(e)


# --- confirmation, staleness and approval ---

def full_chain(repo, sid, subjects):
    """track -> mapping -> confirm -> behavior -> contribution -> pfb for A子 (cam1)."""
    t = track(repo, sid)
    ident = save_tracked_identity(repo, sid, "cam1", FOUR)
    confirm_identity(repo, sid)
    prep = repo.latest_run(sid, Stage.prepare, "cam1")
    ev = {"video_time": "00:05", "participant": "Pa", "observed_behavior": "紙を折る",
          "utterance": "", "speaker": "unknown", "evidence_source": "video",
          "confidence": "medium", "event_id": "cam1-w0000-1", "camera": "cam1",
          "task_elapsed": "00:05"}
    beh = save(repo, sid, Stage.behavior, {"events": [ev], "frames": {}}, camera="cam1",
               upstream=[ident.id, prep.id, t.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [{
        "ref_id": "cam1-w0000-1", "contribution_interpretation": "解釈",
        "is_positive_contribution": True, "rubric_criteria": []}]}, camera="cam1",
        upstream=[beh.id])
    pfb = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "紙を丁寧に折っていました。",
                                      "evidence_refs": ["cam1-w0000-1"]},
               subject_id=subjects["Pa"], upstream=[con.id])
    return t, beh, pfb


def test_mapping_edit_after_behavior_makes_it_stale_and_blocks_approval(repo):
    sid, subjects = submission(repo)
    _, beh, _ = full_chain(repo, sid, subjects)
    # the behavior job confirm queued has run (the chain above stands for its result)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    approve(repo, subjects["Pa"])   # fine before the edit
    save_tracked_identity(repo, sid, "cam1", {"P1": "Pb", "P2": "Pa", "P3": "Pc", "P4": "Pd"})
    assert is_stale_chain(repo, beh)
    with pytest.raises(ApprovalError, match="確認されていません"):
        approve(repo, subjects["Pa"])
    assert confirm_identity(repo, sid) != []   # behavior re-queued for the new mapping
    assert ("behavior", "cam1", "pfb3x3:standard") in queued(repo, sid)
    with pytest.raises(ApprovalError, match="古くなっています"):
        approve(repo, subjects["Pa"])


def test_confirm_refuses_a_mapping_of_an_older_track_run(repo):
    sid, _ = submission(repo)
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    track(repo, sid)   # tracking was re-run
    with pytest.raises(ApprovalError, match="最新の人物の追跡結果"):
        confirm_identity(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    assert confirm_identity(repo, sid) != []
    assert repo.get_submission(sid)["identity_confirmed"]


@pytest.mark.parametrize("stage, camera, status", [
    (Stage.track, "cam2", "queued"), (Stage.track, "cam2", "running"),
    (Stage.track, "cam2", "waiting_gpu"), (Stage.prepare, "cam2", "queued"),
    (Stage.prepare, None, "queued")])
def test_confirm_waits_for_tracking_still_to_come(repo, stage, camera, status):
    sid, _ = submission(repo, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        track(repo, sid, camera=cam)
        save_tracked_identity(repo, sid, cam, FOUR)
    jid = repo.enqueue(sid, stage, "track" if stage == Stage.track else "cpu", camera=camera)
    repo.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, jid))
    with pytest.raises(ApprovalError) as e:
        confirm_identity(repo, sid)
    assert str(e.value) == ("人物の追跡が実行中または待機中です（cam1）。完了後に確認してください。"
                            if camera is None else
                            "人物の追跡が実行中または待機中です（cam2）。完了後に確認してください。")
    assert not repo.get_submission(sid)["identity_confirmed"]
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (jid,))
    assert len(confirm_identity(repo, sid)) == 2


def test_paper_engine_confirm_ignores_prepare_jobs(repo):
    sid, _ = submission(repo, engine=None)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.identity, {"participants": []}, camera="cam1")
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    confirm_identity(repo, sid)   # unchanged: only identity jobs block it
    assert repo.get_submission(sid)["identity_confirmed"]


def test_confirm_without_a_mapping_says_so(repo):
    sid, _ = submission(repo)
    track(repo, sid)
    with pytest.raises(ApprovalError, match="対応づけがありません"):
        confirm_identity(repo, sid)


# --- regenerate ---

def test_regenerate_identity_is_refused_for_the_tracking_engine(repo):
    sid, _ = submission(repo, cameras=("cam1", "cam2"))
    with pytest.raises(ValidationError) as e:
        regenerate_from(repo, sid, Stage.identity, camera="cam1")
    assert TRACK_REDO in message(e)
    assert queued(repo, sid) == []
    regenerate_from(repo, sid, Stage.track)
    assert queued(repo, sid) == [("track", "cam1", "track"), ("track", "cam2", "track")]


def test_regenerate_track_is_refused_for_the_paper_engine(repo):
    sid, _ = submission(repo, engine=None)
    with pytest.raises(ValidationError):
        regenerate_from(repo, sid, Stage.track)
    assert regenerate_from(repo, sid, Stage.identity, camera="cam1") != []   # unchanged


# --- edit_stage_output on a mapping run ---

def test_edit_stage_output_validates_a_mapping(repo):
    sid, _ = submission(repo)
    track(repo, sid)
    run = save_tracked_identity(repo, sid, "cam1", FOUR)
    edited = dict(run.output)
    edited["participants"] = [dict(p) for p in run.output["participants"]]
    edited["participants"][3]["participant_id"] = None
    new = edit_stage_output(repo, run.id, edited)
    assert new.upstream_run_ids == run.upstream_run_ids and new.source == "teacher"
    assert new.output["participants"][3]["display_name"] is None
    assert "D郎" in new.warnings[0]
    edited["participants"][3]["participant_id"] = "Pa"
    with pytest.raises(ValidationError) as e:
        edit_stage_output(repo, new.id, edited)
    assert "同じ参加者" in message(e)
    track(repo, sid)   # an edit of a mapping of an older track run is refused
    with pytest.raises(ValidationError) as e:
        edit_stage_output(repo, new.id, new.output)
    assert "人物の追跡がやり直されています" in message(e)


def test_behavior_edit_accepts_a_mapping_with_outside_people(repo):
    sid, subjects = submission(repo)
    t = track(repo, sid, labels=("P1", "P2", "P3", "P4", "P5"))
    ident = save_tracked_identity(repo, sid, "cam1", {**FOUR, "P5": None})
    prep = repo.latest_run(sid, Stage.prepare, "cam1")
    ev = {"video_time": "00:05", "participant": "Pb", "observed_behavior": "紙を渡す",
          "utterance": "", "speaker": "unknown", "evidence_source": "video",
          "confidence": "medium", "event_id": "cam1-w0000-1", "camera": "cam1",
          "task_elapsed": "00:05"}
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1",
               upstream=[ident.id, prep.id, t.id])
    assert edit_stage_output(repo, beh.id, {"events": [ev]}).status == "ok"
    with pytest.raises(ValidationError):
        edit_stage_output(repo, beh.id, {"events": [{**ev, "participant": "P5"}]})
