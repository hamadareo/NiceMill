"""Review services: edit, regenerate, identity confirmation, approval and subject status."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun, SubjectStatus
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.services.review import (ApprovalError, BusyError, approve, confirm_identity,
                                     edit_stage_output, regenerate_from, subject_status,
                                     unapprove, write_manual_pfb)
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), source="model",
         status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "失敗",
        warnings=[], duration_sec=0.0, created_at=""))


IDENTITY = {"participants": [{
    "participant_id": "Pa", "appearance": "外見", "initial_position": "左",
    "verification_points": ["開始時点"], "identity_confidence": "high", "verified": True,
    "notes": ""}]}


def event(camera="cam1", n=1, participant="Pa"):
    return {"video_time": "00:05", "participant": participant, "observed_behavior": "紙を折る",
            "utterance": "", "speaker": "unknown", "evidence_source": "video",
            "confidence": "high", "event_id": f"{camera}-{n:04d}", "camera": camera,
            "task_elapsed": "00:00"}


def judgment(ref_id, positive=True):
    return {"ref_id": ref_id, "contribution_interpretation": "解釈",
            "is_positive_contribution": positive, "rubric_criteria": []}


PFB = {"pfb_status": "generated", "pfb": "紙を折って形を整えていました。",
       "evidence_refs": ["cam1-0001"]}
INSUFFICIENT = {"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []}


def video(repo, require_approval=False, model_key="standard", pfb=PFB):
    """One camera, one subject, a full run chain: prepare->identity->behavior->contribution->pfb."""
    aid = repo.create_assignment("課題", Kind.video, "塔を作る課題", "ルーブリック", model_key,
                                 require_approval)
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": "/x/cam1.mp4"},
                                             "start_sec": None, "end_sec": None,
                                             "auto_transcribe": False})
    subj = repo.add_subject(sid, "Pa", "A子", "縞模様", {"cam1": "左"})
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    ident = save(repo, sid, Stage.identity, IDENTITY, camera="cam1", upstream=[prep.id])
    beh = save(repo, sid, Stage.behavior, {"events": [event()]}, camera="cam1",
               upstream=[ident.id, prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [judgment("cam1-0001")]},
               camera="cam1", upstream=[beh.id])
    run = save(repo, sid, Stage.pfb, pfb, subject_id=subj, upstream=[con.id])
    repo.set_identity_confirmed(sid, True)
    return dict(aid=aid, sid=sid, subj=subj, prep=prep, ident=ident, beh=beh, con=con, pfb=run)


def document(repo):
    aid = repo.create_assignment("課題", Kind.document, "防災を調べる課題", "ルーブリック",
                                 "standard", False)
    sid = repo.create_submission(aid, "A", {"source_path": "/x/r.pdf",
                                            "original_filename": "r.pdf"})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x/r.pdf"})
    item = {"page": 1, "quote": "q", "observed_content": "c", "evidence_source": "text",
            "confidence": "high", "item_id": "p1-001", "quote_verified": True}
    ev = save(repo, sid, Stage.evidence, {"items": [item]}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [judgment("p1-001")]},
               upstream=[ev.id])
    return dict(sid=sid, subj=subj, ev=ev, con=con, item=item)


def queued(repo, sid):
    return [(r["stage"], r["camera"], r["subject_id"], r["model_key"]) for r in
            repo.conn.execute("SELECT * FROM jobs WHERE submission_id=? AND status='queued'"
                              " ORDER BY id", (sid,))]


# --- edit ---

def test_edit_behavior_makes_contribution_stale(repo):
    v = video(repo)
    assert not repo.is_stale(v["con"]) and not repo.is_stale(v["pfb"])
    new = edit_stage_output(repo, v["beh"].id, {"events": [event(n=1), event(n=2)]})
    assert new.source == "teacher" and new.version == 2 and new.status == "ok"
    assert new.upstream_run_ids == v["beh"].upstream_run_ids
    assert repo.latest_run(v["sid"], Stage.behavior, "cam1").id == new.id
    assert repo.is_stale(v["con"])
    audit = repo.latest_audit("edit_stage_output", f"run:{new.id}")
    assert audit["detail"]["base_run_id"] == v["beh"].id


def test_edit_rejected_while_job_running(repo):
    v = video(repo)
    repo.enqueue(v["sid"], Stage.behavior, "standard", camera="cam1")
    repo.claim_next_job({"standard"})
    with pytest.raises(BusyError):
        edit_stage_output(repo, v["beh"].id, {"events": []})
    assert repo.latest_run(v["sid"], Stage.behavior, "cam1").id == v["beh"].id


def test_regenerate_rejected_while_job_running(repo):
    v = video(repo)
    repo.enqueue(v["sid"], Stage.pfb, "standard", subject_id=v["subj"])
    with pytest.raises(BusyError):
        regenerate_from(repo, v["sid"], Stage.behavior, camera="cam1")


def test_edit_rejects_schema_violation(repo):
    v = video(repo)
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["beh"].id, {"events": [{"video_time": "00:05"}]})
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["con"].id, {"judgments": [], "extra": 1})
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["pfb"].id, {"pfb_status": "generated", "pfb": None,
                                              "evidence_refs": []})
    assert len(repo.run_history(v["sid"], Stage.behavior, "cam1")) == 1


def test_edit_rejects_unsupported_stage(repo):
    v = video(repo)
    with pytest.raises(ValidationError) as e:
        edit_stage_output(repo, v["prep"].id, {"frames": []})
    assert "編集" in str(e.value)


def test_edit_contribution_rejects_unknown_ref_id(repo):
    v = video(repo)
    with pytest.raises(ValidationError) as e:
        edit_stage_output(repo, v["con"].id, {"judgments": [judgment("cam1-0099")]})
    assert "cam1-0099" in str(e.value)
    with pytest.raises(ValidationError):  # duplicate
        edit_stage_output(repo, v["con"].id,
                          {"judgments": [judgment("cam1-0001"), judgment("cam1-0001")]})
    ok = edit_stage_output(repo, v["con"].id, {"judgments": [judgment("cam1-0001", False)]})
    assert ok.output["judgments"][0]["is_positive_contribution"] is False
    assert ok.upstream_run_ids == [v["beh"].id]


def test_edit_behavior_rejects_unknown_participant_duplicate_id_and_wrong_camera(repo):
    v = video(repo)
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["beh"].id, {"events": [event(participant="Pz")]})
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["beh"].id, {"events": [event(n=1), event(n=1)]})
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["beh"].id, {"events": [event(camera="cam2")]})


def test_edit_document_evidence_and_contribution(repo):
    d = document(repo)
    dup = dict(d["item"])
    with pytest.raises(ValidationError):
        edit_stage_output(repo, d["ev"].id, {"items": [d["item"], dup]})
    new_item = dict(d["item"], item_id="p1-002")
    ev = edit_stage_output(repo, d["ev"].id, {"items": [d["item"], new_item]})
    assert repo.is_stale(d["con"])
    with pytest.raises(ValidationError):
        edit_stage_output(repo, d["con"].id, {"judgments": [judgment("p9-001")]})
    con = save(repo, d["sid"], Stage.contribution, {"judgments": []}, upstream=[ev.id])
    edit_stage_output(repo, con.id, {"judgments": [judgment("p1-002")]})


def test_edit_pfb_checks_evidence_refs_and_keeps_upstream(repo):
    v = video(repo)
    with pytest.raises(ValidationError):
        edit_stage_output(repo, v["pfb"].id, {**PFB, "evidence_refs": ["cam1-0042"]})
    new = edit_stage_output(repo, v["pfb"].id, {**PFB, "pfb": "手順を整えていました。"})
    assert new.source == "teacher" and new.upstream_run_ids == v["pfb"].upstream_run_ids
    assert not repo.is_stale(new)


# --- identity confirmation reset ---

def test_teacher_identity_edit_resets_confirmation(repo):
    v = video(repo)
    assert repo.get_submission(v["sid"])["identity_confirmed"]
    edit_stage_output(repo, v["ident"].id, IDENTITY)
    assert not repo.get_submission(v["sid"])["identity_confirmed"]


def test_model_identity_run_resets_confirmation(repo):
    v = video(repo)
    save(repo, v["sid"], Stage.identity, IDENTITY, camera="cam1", upstream=[v["prep"].id])
    assert not repo.get_submission(v["sid"])["identity_confirmed"]


def test_other_stage_runs_keep_confirmation(repo):
    v = video(repo)
    edit_stage_output(repo, v["beh"].id, {"events": [event()]})
    assert repo.get_submission(v["sid"])["identity_confirmed"]


# --- regenerate ---

def test_regenerate_enqueues_with_assignment_model_key(repo):
    v = video(repo, model_key="high")
    assert regenerate_from(repo, v["sid"], Stage.behavior, camera="cam1") != []
    assert regenerate_from.__name__
    assert queued(repo, v["sid"]) == [("behavior", "cam1", None, "high")]


def test_regenerate_prepare_uses_cpu_and_fans_out_cameras(repo):
    v = video(repo)
    ids = regenerate_from(repo, v["sid"], Stage.prepare)
    assert len(ids) == 1
    assert queued(repo, v["sid"]) == [("prepare", "cam1", None, "cpu")]


def test_regenerate_pfb_for_all_participant_subjects(repo):
    v = video(repo)
    repo.add_subject(v["sid"], None, "名前のみ", None, {})
    regenerate_from(repo, v["sid"], Stage.pfb)
    assert queued(repo, v["sid"]) == [("pfb", None, v["subj"], "standard")]


def test_regenerate_rejects_stage_of_other_kind(repo):
    d = document(repo)
    with pytest.raises(ValidationError):
        regenerate_from(repo, d["sid"], Stage.behavior)


def test_regenerate_identity_is_idempotent_with_enqueue_once(repo):
    v = video(repo)
    first = regenerate_from(repo, v["sid"], Stage.identity, camera="cam1")
    assert len(first) == 1
    assert enqueue_once(repo, v["sid"], Stage.identity, "cam1") is None


# --- confirm_identity ---

def test_confirm_identity_enqueues_behavior_when_gated(repo):
    v = video(repo, require_approval=True)
    repo.set_identity_confirmed(v["sid"], False)
    # behavior is up to date, so only an out-of-date or missing behavior is enqueued
    edit_stage_output(repo, v["ident"].id, IDENTITY)
    ids = confirm_identity(repo, v["sid"])
    assert len(ids) == 1
    assert repo.get_submission(v["sid"])["identity_confirmed"]
    assert queued(repo, v["sid"]) == [("behavior", "cam1", None, "standard")]


def test_confirm_identity_without_gate_enqueues_nothing(repo):
    v = video(repo, require_approval=False)
    repo.set_identity_confirmed(v["sid"], False)
    assert confirm_identity(repo, v["sid"]) == []
    assert repo.get_submission(v["sid"])["identity_confirmed"]
    assert queued(repo, v["sid"]) == []


def test_confirm_identity_blocked_while_identity_job_active(repo):
    v = video(repo)
    repo.set_identity_confirmed(v["sid"], False)
    repo.enqueue(v["sid"], Stage.identity, "standard", camera="cam1")
    with pytest.raises(ApprovalError):
        confirm_identity(repo, v["sid"])
    assert not repo.get_submission(v["sid"])["identity_confirmed"]


def test_confirm_identity_blocked_when_a_camera_has_no_ok_identity(repo):
    v = video(repo)
    sub = repo.get_submission(v["sid"])
    repo.update_submission_meta(v["sid"], {**sub["meta"], "cameras": {
        "cam1": "/x/cam1.mp4", "cam2": "/x/cam2.mp4"}})
    repo.set_identity_confirmed(v["sid"], False)
    with pytest.raises(ApprovalError) as e:
        confirm_identity(repo, v["sid"])
    assert "cam2" in str(e.value)
    save(repo, v["sid"], Stage.identity, None, camera="cam2", status="failed")
    with pytest.raises(ApprovalError):
        confirm_identity(repo, v["sid"])


# --- approve ---

def test_approve_blocked_until_identity_confirmed(repo):
    v = video(repo)
    repo.set_identity_confirmed(v["sid"], False)
    with pytest.raises(ApprovalError):
        approve(repo, v["subj"])
    repo.set_identity_confirmed(v["sid"], True)
    approve(repo, v["subj"])
    subject = repo.list_subjects(v["sid"])[0]
    assert subject["approval_status"] == "approved" and subject["approved_run_id"] == v["pfb"].id


def test_approve_blocked_when_pfb_stale(repo):
    v = video(repo)
    edit_stage_output(repo, v["beh"].id, {"events": [event()]})
    with pytest.raises(ApprovalError):
        approve(repo, v["subj"])


def test_approve_blocked_for_failed_and_insufficient_pfb(repo):
    v = video(repo, pfb=INSUFFICIENT)
    with pytest.raises(ApprovalError):
        approve(repo, v["subj"])
    save(repo, v["sid"], Stage.pfb, None, subject_id=v["subj"], upstream=[v["con"].id],
         status="failed")
    with pytest.raises(ApprovalError):
        approve(repo, v["subj"])


def test_approve_blocked_without_pfb_run(repo):
    v = video(repo)
    other = repo.add_subject(v["sid"], "Pb", "B子", None, {})
    with pytest.raises(ApprovalError):
        approve(repo, other)


def test_manual_pfb_can_be_approved_when_insufficient(repo):
    v = video(repo, pfb=INSUFFICIENT)
    assert subject_status(repo, v["subj"]) == SubjectStatus.insufficient
    run = write_manual_pfb(repo, v["subj"], "道具を分けて渡していました。")
    assert run.source == "teacher" and run.warnings == ["手書き"]
    assert run.output["pfb_status"] == "generated" and run.output["evidence_refs"] == []
    assert run.upstream_run_ids == [v["con"].id]
    approve(repo, v["subj"])
    assert subject_status(repo, v["subj"]) == SubjectStatus.approved


def test_manual_pfb_rejects_empty_text(repo):
    v = video(repo)
    with pytest.raises(ValidationError):
        write_manual_pfb(repo, v["subj"], "  ")


def test_manual_pfb_rejected_while_job_running(repo):
    v = video(repo)
    repo.enqueue(v["sid"], Stage.pfb, "standard", subject_id=v["subj"])
    with pytest.raises(BusyError):
        write_manual_pfb(repo, v["subj"], "本文")


def test_approved_subject_returns_to_review_when_newer_pfb_exists(repo):
    v = video(repo)
    approve(repo, v["subj"])
    assert subject_status(repo, v["subj"]) == SubjectStatus.approved
    save(repo, v["sid"], Stage.pfb, PFB, subject_id=v["subj"], upstream=[v["con"].id])
    assert subject_status(repo, v["subj"]) == SubjectStatus.review


def test_unapprove_clears_approval(repo):
    v = video(repo)
    approve(repo, v["subj"])
    unapprove(repo, v["subj"])
    subject = repo.list_subjects(v["sid"])[0]
    assert subject["approval_status"] == "none" and subject["approved_run_id"] is None
    assert subject_status(repo, v["subj"]) == SubjectStatus.review


# --- status ---

def test_subject_status_transitions(repo):
    aid = repo.create_assignment("課題", Kind.document, "課題文", "ルーブリック", "standard", False)
    sid = repo.create_submission(aid, "A", {"source_path": "/x", "original_filename": "r.pdf"})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    assert subject_status(repo, subj) == SubjectStatus.pending
    job = repo.enqueue(sid, Stage.prepare, "cpu")
    assert subject_status(repo, subj) == SubjectStatus.processing
    repo.claim_next_job({"cpu"})
    assert subject_status(repo, subj) == SubjectStatus.processing
    repo.finish_job(job, "done")
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "x"})
    ev = save(repo, sid, Stage.evidence, {"items": []}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": []}, upstream=[ev.id])
    assert subject_status(repo, subj) == SubjectStatus.pending  # nothing running, no PFB yet
    pfb = save(repo, sid, Stage.pfb, PFB, subject_id=subj, upstream=[con.id])
    assert subject_status(repo, subj) == SubjectStatus.review
    approve(repo, subj)
    assert subject_status(repo, subj) == SubjectStatus.approved
    unapprove(repo, subj)
    save(repo, sid, Stage.pfb, INSUFFICIENT, subject_id=subj, upstream=[con.id])
    assert subject_status(repo, subj) == SubjectStatus.insufficient
    save(repo, sid, Stage.pfb, None, subject_id=subj, upstream=[con.id], status="failed")
    assert subject_status(repo, subj) == SubjectStatus.failed
    assert pfb.id


def test_subject_status_waiting_gpu_and_stale_review(repo):
    v = video(repo)
    jid = repo.enqueue(v["sid"], Stage.pfb, "standard", subject_id=v["subj"])
    repo.mark_waiting_gpu("standard", "VRAM不足")
    assert subject_status(repo, v["subj"]) == SubjectStatus.waiting_gpu
    repo.claim_next_job({"standard"})
    assert subject_status(repo, v["subj"]) == SubjectStatus.processing
    repo.finish_job(jid, "done")
    edit_stage_output(repo, v["beh"].id, {"events": [event()]})
    assert subject_status(repo, v["subj"]) == SubjectStatus.review  # stale, awaiting regeneration


def test_subject_status_ignores_another_persons_text_job(repo):
    """Only the person's own text job, or an earlier step, makes them 解析中; another person's
    text being written (or a names job) leaves a ready text 確認待ち."""
    v = video(repo)
    other = repo.add_subject(v["sid"], "Pb", "B子", None, {})
    repo.enqueue(v["sid"], Stage.pfb, "standard", subject_id=other)
    repo.enqueue(v["sid"], Stage.names, "cpu", camera="cam1")
    assert subject_status(repo, v["subj"]) == SubjectStatus.review
    assert subject_status(repo, other) == SubjectStatus.processing
    repo.enqueue(v["sid"], Stage.behavior, "standard", camera="cam1")
    assert subject_status(repo, v["subj"]) == SubjectStatus.processing


def test_subject_status_failed_upstream_without_pfb(repo):
    v = video(repo)
    other = repo.add_subject(v["sid"], "Pb", "B子", None, {})
    save(repo, v["sid"], Stage.contribution, None, camera="cam1", upstream=[v["beh"].id],
         status="failed")
    assert subject_status(repo, other) == SubjectStatus.failed
