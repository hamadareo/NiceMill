"""Tracking-engine (PFB3x3) submissions in the runner: model keys, follow-ups, the track stage."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fakes.engine_env import install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.engine import Pfb3x3Engine
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import runner, tracking
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import (PFB3X3_PREFIX, TRACK_KEY, enqueue_followups, execute_job,
                                     model_key_for, submission_engine)
from pfb_app.pipeline.tracking import GpuUnavailable, place_positions, rule_positions
from pfb_app.services.review import confirm_identity
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fake):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=fake.engine_dir, engine_python=fake.engine_python)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def make_video(path: Path, seconds: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        writer.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def tracked_submission(repo, tmp_path, cameras=("cam1", "cam2"), require_approval=True,
                       auto_transcribe=False, model_key="standard", engine="pfb3x3",
                       positions=None):
    aid = repo.create_assignment("課題", Kind.video, "紙タワーを作る課題", "ルーブリック",
                                 model_key, require_approval)
    paths = {c: str(make_video(tmp_path / f"{c}.mp4")) for c in cameras}
    meta = {"cameras": paths, "start_sec": 1.0, "end_sec": None,
            "auto_transcribe": auto_transcribe}
    if engine is not None:
        meta["engine"] = engine
    sid = repo.create_submission(aid, "G1", meta)
    positions = positions or {}
    for pid, name in (("Pa", "A子"), ("Pb", "B太"), ("Pc", "C美"), ("Pd", "D郎")):
        repo.add_subject(sid, pid, name, "外見", positions.get(pid, {}))
    return aid, sid


def save(repo, sid, stage, output, camera=None, source="model", upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def make_ctx(settings, repo, yield_check=lambda: False, device="cuda"):
    return StageContext(settings, repo, FakeBackend(["{}"]), yield_check,
                        engine=Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=2),
                        device=device)


def jobs(repo, sid):
    rows = repo.conn.execute(
        "SELECT stage, camera, model_key, status FROM jobs WHERE submission_id=? ORDER BY id",
        (sid,)).fetchall()
    return [tuple(r) for r in rows]


def queued(repo, sid):
    return [(s, c, k) for s, c, k, st in jobs(repo, sid) if st == "queued"]


def claim_run(repo, ctx, keys):
    job = repo.claim_next_job(keys)
    assert job is not None
    execute_job(job, ctx)
    return repo.get_job(job.id)


def teacher_mapping(repo, sid, camera):
    """What Task 4's save_tracked_identity stores (camera-scoped, source teacher)."""
    track = repo.latest_run(sid, Stage.track, camera)
    return save(repo, sid, Stage.identity, {
        "participants": [{"participant_id": "Pa", "label": "P1", "display_name": "A子", "note": ""}],
        "labels": track.output["labels"], "track_run_id": track.id},
        camera=camera, source="teacher", upstream=[track.id])


# --- model keys ---

def test_submission_engine_defaults_to_paper(repo, tmp_path):
    _, paper = tracked_submission(repo, tmp_path, cameras=("cam1",), engine=None)
    _, tracked = tracked_submission(repo, tmp_path, cameras=("cam1",))
    assert submission_engine(repo.get_submission(paper)) == "paper"
    assert submission_engine(repo.get_submission(tracked)) == "pfb3x3"


def test_model_keys_for_tracking_engine(repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), model_key="high")
    _, paper = tracked_submission(repo, tmp_path, cameras=("cam1",), model_key="high",
                                  engine=None)
    assert TRACK_KEY == "track" and PFB3X3_PREFIX == "pfb3x3:"
    assert model_key_for(repo, sid, Stage.track) == "track"
    assert model_key_for(repo, sid, Stage.behavior) == "pfb3x3:high"
    assert model_key_for(repo, sid, Stage.prepare) == "cpu"
    assert model_key_for(repo, sid, Stage.transcribe) == "whisper"
    # contribution and pfb ask the session's server too (no second model load)
    assert model_key_for(repo, sid, Stage.contribution) == "pfb3x3:high"
    assert model_key_for(repo, sid, Stage.pfb) == "pfb3x3:high"
    # the paper engine is unchanged
    assert model_key_for(repo, paper, Stage.behavior) == "high"
    assert model_key_for(repo, paper, Stage.contribution) == "high"
    assert model_key_for(repo, paper, Stage.pfb) == "high"
    assert model_key_for(repo, paper, Stage.identity) == "high"


# --- the stage sequence ---

def test_two_camera_sequence_until_behavior(settings, repo, tmp_path, fake, monkeypatch):
    _, sid = tracked_submission(repo, tmp_path, auto_transcribe=True)
    monkeypatch.setattr(runner, "transcribe", lambda *a: pytest.fail("no pfb_app whisper"))
    ctx = make_ctx(settings, repo)
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam2")

    assert claim_run(repo, ctx, {"cpu"}).status == JobStatus.done
    assert queued(repo, sid) == [("prepare", "cam2", "cpu"), ("track", "cam1", "track")]
    assert claim_run(repo, ctx, {"cpu"}).status == JobStatus.done
    # no pfb_app whisper transcript for the tracking engine: track transcribes per camera
    assert queued(repo, sid) == [("track", "cam1", "track"), ("track", "cam2", "track")]

    for cam in ("cam1", "cam2"):
        assert claim_run(repo, ctx, {"track"}).status == JobStatus.done
        run = repo.latest_run(sid, Stage.track, cam)
        prepare = repo.latest_run(sid, Stage.prepare, cam)
        assert run.status == "ok" and run.upstream_run_ids == [prepare.id]
        out = run.output
        assert out["people"] == 3 and out["labels"] == ["P1", "P2", "P3"]
        assert Path(out["sheet"]).is_file()
        stats = dict(out["track_stats"])
        assert stats.pop("tracker")["used"] == "engine"
        assert stats == {"P1": {"frames": 100}, "P2": {"frames": 90}, "P3": {"frames": 80}}
        assert out["tracker"] == {"used": "engine", "fallback": "not a TID dataset video"}
        assert out["default_names"] == {} and out["default_source"] is None
        assert out["device"] == "cuda" and out["video_id"] == f"s{sid}_{cam}"
        assert out["transcript"] is True and out["transcript_segments"] == 2
        assert out["sheet_frame"] == 152 and out["sheet_frame_rule"] == "auto_start"
    # automatic people: each camera gets an auto mapping and, once both have one, behavior is
    # queued with no coach step; no identity job exists
    assert [repo.latest_run(sid, Stage.identity, c).source for c in ("cam1", "cam2")] == [
        "auto", "auto"]
    assert repo.get_submission(sid)["identity_confirmed"]
    assert queued(repo, sid) == [("behavior", "cam1", "pfb3x3:standard"),
                                 ("behavior", "cam2", "pfb3x3:standard")]
    assert all(stage not in ("identity", "transcribe") for stage, *_ in jobs(repo, sid))
    track_calls = [c for c in fake.calls() if c["argv"][0] == "track"]
    assert [c["argv"][c["argv"].index("--device") + 1] for c in track_calls] == ["cuda", "cuda"]
    # auto_transcribe: the engine also transcribes each camera's own audio after tracking
    per_video = [(c["argv"][0], c["argv"][c["argv"].index("--video-id") + 1])
                 for c in fake.calls() if c["argv"][0] in ("track", "transcribe")]
    assert per_video == [("track", f"s{sid}_cam1"), ("transcribe", f"s{sid}_cam1"),
                         ("track", f"s{sid}_cam2"), ("transcribe", f"s{sid}_cam2")]
    sub_dir = files.submission_dir(settings, repo.get_submission(sid)["assignment_id"], sid)
    entry = yaml.safe_load(
        (sub_dir / "pfb3x3" / "videos" / f"s{sid}_cam1.yaml").read_text(encoding="utf-8"))
    assert entry["activity"] == "紙タワーを作る課題" and entry["transcript"] is True


@pytest.mark.parametrize("require_approval", [True, False])
def test_behavior_enqueued_only_after_confirm(settings, repo, tmp_path, require_approval):
    _, sid = tracked_submission(repo, tmp_path, require_approval=require_approval)
    ctx = make_ctx(settings, repo)
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        repo.enqueue(sid, Stage.track, "track", camera=cam)
        claim_run(repo, ctx, {"track"})
    # automatic people: behavior is queued once both cameras have an auto mapping
    assert queued(repo, sid) == [("behavior", "cam1", "pfb3x3:standard"),
                                 ("behavior", "cam2", "pfb3x3:standard")]
    repo.conn.execute("DELETE FROM jobs WHERE submission_id=? AND stage='behavior'", (sid,))
    for cam in ("cam1", "cam2"):
        teacher_mapping(repo, sid, cam)
    assert queued(repo, sid) == []  # saving the mapping is not a confirmation
    ids = confirm_identity(repo, sid)
    assert len(ids) == 2
    assert queued(repo, sid) == [("behavior", "cam1", "pfb3x3:standard"),
                                 ("behavior", "cam2", "pfb3x3:standard")]


def test_behavior_and_contribution_follow_ups(repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    job = repo.get_job(jid)
    enqueue_followups(repo, job)
    assert queued(repo, sid)[-1] == ("contribution", "cam1", "pfb3x3:standard")


def test_track_done_has_no_follow_up(repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    job = repo.get_job(repo.enqueue(sid, Stage.track, "track", camera="cam1"))
    assert enqueue_followups(repo, job) == []


def test_paper_engine_prepare_still_goes_to_identity(repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, engine=None)
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        job = repo.get_job(repo.enqueue(sid, Stage.prepare, "cpu", camera=cam))
        enqueue_followups(repo, job)
    assert [(s, c, k) for s, c, k in queued(repo, sid) if s != "prepare"] == [
        ("identity", "cam1", "standard"), ("identity", "cam2", "standard")]


# --- the track stage ---

def _run_track(settings, repo, tmp_path, cam="cam1"):
    _, sid = tracked_submission(repo, tmp_path, cameras=(cam,))
    save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    repo.enqueue(sid, Stage.track, "track", camera=cam)
    claim_run(repo, make_ctx(settings, repo), {"track"})
    return repo.latest_run(sid, Stage.track, cam)


def test_track_records_quality(settings, repo, tmp_path, fake):
    run = _run_track(settings, repo, tmp_path)
    assert set(run.output["quality"]) == {"P1", "P2", "P3"}
    assert run.output["quality"]["P1"]["coverage"] == 1.0
    assert run.warnings == []


def test_track_records_the_tracker(settings, repo, tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_TRACKER", "trackid3x3")
    run = _run_track(settings, repo, tmp_path)
    assert run.output["tracker"] == {"used": "trackid3x3", "fallback": None}
    assert tracking.tracker_text(run.output) == "TrackID3x3 の追跡（すでにある結果）"


def test_track_without_a_reported_tracker(settings, repo, tmp_path, fake, monkeypatch):
    """An engine before 58f8341 (no TID data, no `tracker`): the run is as before."""
    monkeypatch.setenv("FAKE_ENGINE_TRACKER", "none")
    run = _run_track(settings, repo, tmp_path)
    assert run.status == "ok" and run.output["tracker"] is None
    assert "tracker" not in run.output["track_stats"]
    assert tracking.tracker_text(run.output) is None


def test_tracker_text():
    assert tracking.tracker_text({"tracker": {"used": "engine", "fallback": None}}) == "標準の追跡"
    assert tracking.tracker_text({"tracker": {"used": "engine",
                                              "fallback": "no TID run"}}) == "標準の追跡"
    for out in ({}, {"tracker": None}, {"tracker": {"used": "other"}}, {"tracker": "engine"},
                None):
        assert tracking.tracker_text(out) is None


def test_track_quality_failure_is_a_warning(settings, repo, tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "quality")
    run = _run_track(settings, repo, tmp_path)
    assert run.status == "ok" and run.output["quality"] is None
    assert "人の写り方を調べられなかったため、映っている長さで選びました" in run.warnings
    assert run.output["labels"] == ["P1", "P2", "P3"]


def test_rule_defaults_from_initial_positions(settings, repo, tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_PEOPLE", "4")
    positions = {"Pa": {"cam1": "左前"}, "Pb": {"cam1": "左奥"}, "Pc": {"cam1": "right_front"},
                 "Pd": {"cam1": "右奥", "cam2": "右奥"}}
    _, sid = tracked_submission(repo, tmp_path, positions=positions)
    ctx = make_ctx(settings, repo)
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        repo.enqueue(sid, Stage.track, "track", camera=cam)
        claim_run(repo, ctx, {"track"})
    cam1 = repo.latest_run(sid, Stage.track, "cam1").output
    assert cam1["default_source"] == "rule"
    # the fake rule numbers the positions in order; the values are participant ids
    assert cam1["default_names"] == {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": "Pd"}
    rule_calls = [c["argv"] for c in fake.calls() if c["argv"][0] == "rule-names"]
    assert len(rule_calls) == 1  # cam2 lacks positions: no rule call, no defaults
    # the sheet the teacher sees and the rule use the same (auto-start) frame
    sheet_calls = [c["argv"] for c in fake.calls() if c["argv"][0] == "sheet"]
    assert "--auto-start" in rule_calls[0] and all("--auto-start" in a for a in sheet_calls)
    assert json.loads(rule_calls[0][rule_calls[0].index("--positions") + 1]) == {
        "left_front": "Pa", "left_back": "Pb", "right_front": "Pc", "right_back": "Pd"}
    cam2 = repo.latest_run(sid, Stage.track, "cam2").output
    assert cam2["default_names"] == {} and cam2["default_source"] is None


def test_rule_positions_needs_four_distinct_known_positions(repo, tmp_path):
    subjects = [{"participant_id": p, "initial_positions": {"cam1": pos}}
                for p, pos in (("Pa", "左前"), ("Pb", "左後ろ"), ("Pc", "右前"), ("Pd", "右後"))]
    assert rule_positions(subjects, "cam1") == {
        "left_front": "Pa", "left_back": "Pb", "right_front": "Pc", "right_back": "Pd"}
    assert rule_positions(subjects[:3], "cam1") is None
    dup = subjects[:3] + [{"participant_id": "Pd", "initial_positions": {"cam1": "左前"}}]
    assert rule_positions(dup, "cam1") is None
    vague = subjects[:3] + [{"participant_id": "Pd", "initial_positions": {"cam1": "窓際"}}]
    assert rule_positions(vague, "cam1") is None
    named_only = subjects + [{"participant_id": None, "initial_positions": {}}]
    assert rule_positions(named_only, "cam1") is not None  # no participant id: not a student


def test_rule_positions_accepts_grid_keys_and_new_aliases(repo, tmp_path):
    subjects = [{"participant_id": p, "initial_positions": {"cam1": pos}}
                for p, pos in (("Pa", "左前"), ("Pb", "中央前"), ("Pc", "右前"),
                               ("Pd", "真ん中の中段"))]
    assert rule_positions(subjects, "cam1") == {
        "left_front": "Pa", "center_front": "Pb", "right_front": "Pc", "center_middle": "Pd"}
    english = [{"participant_id": "Pe", "initial_positions": {"cam1": "Right_Back"}}]
    assert rule_positions(subjects[:3] + english, "cam1")["right_back"] == "Pe"
    # a column without a row (or a row without a column) is not a place
    vague = subjects[:3] + [{"participant_id": "Pd", "initial_positions": {"cam1": "中央"}}]
    assert rule_positions(vague, "cam1") is None
    both = subjects[:3] + [{"participant_id": "Pd", "initial_positions": {"cam1": "左右前"}}]
    assert rule_positions(both, "cam1") is None
    other = subjects[:3] + [{"participant_id": "Pd",
                             "initial_positions": {"cam1": "右から2番目の前"}}]
    assert rule_positions(other, "cam1") is None


def test_place_positions_six_students():
    places = ("左前", "中央前", "右前", "左奥", "真ん中奥", "右後ろ")
    subjects = [{"participant_id": f"P{c}", "initial_positions": {"cam1": pos}}
                for c, pos in zip("abcdef", places)]
    assert rule_positions(subjects, "cam1") is None   # the school rule needs exactly four
    assert place_positions(subjects, "cam1") == {
        "left_front": "Pa", "center_front": "Pb", "right_front": "Pc", "left_back": "Pd",
        "center_back": "Pe", "right_back": "Pf"}
    # a student without a place on this camera is left out; an unknown or repeated one stops it
    absent = subjects + [{"participant_id": "Pg", "initial_positions": {"cam2": "左前"}}]
    assert len(place_positions(absent, "cam1")) == 6
    unknown = subjects + [{"participant_id": "Pg", "initial_positions": {"cam1": "窓際"}}]
    assert place_positions(unknown, "cam1") is None
    dup = subjects + [{"participant_id": "Pg", "initial_positions": {"cam1": "左手前"}}]
    assert place_positions(dup, "cam1") is None
    assert place_positions([], "cam1") is None
    assert place_positions(subjects[:2], "cam1") == {"left_front": "Pa", "center_front": "Pb"}


def test_run_track_passes_sports_domain(settings, repo, tmp_path, fake):
    seen = []

    class RecordingEngine(Pfb3x3Engine):
        def track(self, *args, **kwargs):
            seen.append(kwargs)
            return super().track(*args, **kwargs)

    aid = repo.create_assignment("試合", Kind.video, "練習試合", "", "standard", True,
                                 domain="sports", sport="soccer")
    meta = {"cameras": {"cam1": str(make_video(tmp_path / "c.mp4"))}, "start_sec": 0.0,
            "engine": "pfb3x3", "target": "vehicle"}
    sid = repo.create_submission(aid, "G1", meta)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    ctx = StageContext(settings, repo, FakeBackend(["{}"]), lambda: False,
                       engine=RecordingEngine(settings, tick_sec=0.05, term_grace_sec=2),
                       device="cuda")
    claim_run(repo, ctx, {"track"})
    assert seen[-1]["domain"] == "sports" and seen[-1]["sport"] == "soccer"
    assert seen[-1]["target"] == "vehicle"
    entry = yaml.safe_load((files.submission_dir(settings, aid, sid) / "pfb3x3" / "videos"
                            / f"s{sid}_cam1.yaml").read_text(encoding="utf-8"))
    assert (entry["domain"], entry["sport"], entry["target"]) == ("sports", "soccer", "vehicle")
    # a school work stays the engine's generic domain, a person target
    _, school = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, school, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(school, Stage.track, "track", camera="cam1")
    claim_run(repo, ctx, {"track"})
    assert (seen[-1]["domain"], seen[-1]["sport"], seen[-1]["target"]) == (
        "generic", None, "person")


def test_run_track_bad_target_fails_plainly(settings, repo, tmp_path, fake):
    aid = repo.create_assignment("試合", Kind.video, "練習試合", "", "standard", True,
                                 domain="sports", sport="soccer")
    meta = {"cameras": {"cam1": str(make_video(tmp_path / "c.mp4"))}, "start_sec": 0.0,
            "engine": "pfb3x3", "target": "boat"}
    sid = repo.create_submission(aid, "G1", meta)
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.failed and "追跡する対象" in run.error
    assert "track" not in fake.commands()


def test_track_needs_prepare(settings, repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    job = claim_run(repo, make_ctx(settings, repo), {"track"})
    assert job.status == JobStatus.failed
    assert "前処理" in repo.latest_run(sid, Stage.track, "cam1").error


def test_track_engine_error_is_retryable_with_stderr_tail(settings, repo, tmp_path,
                                                          monkeypatch):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    job = claim_run(repo, make_ctx(settings, repo), {"track"})
    assert job.status == JobStatus.failed_retryable
    assert job.error.startswith("EngineError: ")
    assert "fake engine track: failure 1" in job.error
    assert repo.latest_run(sid, Stage.track, "cam1") is None


def test_track_yield_requeues(settings, repo, tmp_path, monkeypatch):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "30")
    ctx = make_ctx(settings, repo, yield_check=lambda: True)
    job = repo.claim_next_job({"track"})
    with pytest.raises(YieldRequested):
        execute_job(job, ctx)
    assert repo.get_job(jid).status == JobStatus.queued


def test_track_needs_gpu_requeues_with_reason(settings, repo, tmp_path, monkeypatch):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    job = repo.claim_next_job({"track"})
    with pytest.raises(GpuUnavailable) as err:
        execute_job(job, make_ctx(settings, repo))
    assert "GPU" in err.value.reason
    assert repo.get_job(jid).status == JobStatus.queued


def test_track_heartbeats_while_the_engine_runs(settings, repo, tmp_path, monkeypatch):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "0.5")
    beats = []
    real = Repo.heartbeat
    monkeypatch.setattr(Repo, "heartbeat", lambda self, job_id: beats.append(job_id) or
                        real(self, job_id))
    job = claim_run(repo, make_ctx(settings, repo, device="cpu"), {"track"})
    assert job.status == JobStatus.done
    assert beats and set(beats) == {jid}
    assert repo.latest_run(sid, Stage.track, "cam1").output["device"] == "cpu"


# --- transcripts for the engine ---

def _entry(settings, repo, sid, cam="cam1") -> dict:
    sub_dir = files.submission_dir(settings, repo.get_submission(sid)["assignment_id"], sid)
    return yaml.safe_load((sub_dir / "pfb3x3" / "videos" / f"s{sid}_{cam}.yaml")
                          .read_text(encoding="utf-8"))


def _track_once(settings, repo, sid, cam="cam1"):
    save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    repo.enqueue(sid, Stage.track, "track", camera=cam)
    job = claim_run(repo, make_ctx(settings, repo), {"track"})
    return job, repo.latest_run(sid, Stage.track, cam)


def test_no_transcript_wanted_means_no_engine_transcribe(settings, repo, tmp_path, fake):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.done
    assert "transcribe" not in fake.commands()
    assert run.output["transcript"] is False and _entry(settings, repo, sid)["transcript"] is False


def _transcript_file(settings, repo, sid, cam="cam1") -> Path:
    sub_dir = files.submission_dir(settings, repo.get_submission(sid)["assignment_id"], sid)
    return sub_dir / "pfb3x3" / "tracks" / f"s{sid}_{cam}" / "transcript.json"


def _paste(repo, sid):
    save(repo, sid, Stage.transcribe, {"segments": [
        {"start": 5.0, "end": 9.0, "text": "はい"}, {"start": 9.0, "end": 12.0, "text": " "},
        {"start": 12.0, "end": math.inf, "text": "どうぞ"}]}, source="teacher")


def test_a_pasted_transcript_is_the_engines_cam1_transcript(settings, repo, tmp_path, fake):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    _paste(repo, sid)
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.done and "transcribe" not in fake.commands()   # no whisper
    assert json.loads(_transcript_file(settings, repo, sid).read_text(encoding="utf-8")) == [
        {"start": 5.0, "end": 9.0, "text": "はい"},
        {"start": 12.0, "end": 86400.0, "text": "どうぞ"}]   # empty text dropped, open end
    assert run.output["transcript"] is True and run.output["transcript_segments"] == 2
    assert run.output["transcript_source"] == "pasted"
    assert _entry(settings, repo, sid)["transcript"] is True


@pytest.mark.parametrize("auto, cam2", [(False, None), (True, "whisper")])
def test_other_cameras_transcribe_only_with_auto_transcribe(settings, repo, tmp_path, fake,
                                                            auto, cam2):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1", "cam2"), auto_transcribe=auto)
    _paste(repo, sid)
    _, run1 = _track_once(settings, repo, sid, "cam1")
    _, run2 = _track_once(settings, repo, sid, "cam2")
    assert run1.output["transcript_source"] == "pasted"
    assert run2.output["transcript_source"] == cam2
    assert _entry(settings, repo, sid, "cam2")["transcript"] is (cam2 is not None)
    assert fake.commands().count("transcribe") == (1 if auto else 0)


def test_transcribe_failure_keeps_tracking_without_transcript(settings, repo, tmp_path, fake,
                                                              monkeypatch):
    """A video without audio: the track run is ok, behavior later runs on the video alone."""
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), auto_transcribe=True)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "transcribe")
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.done and run.status == "ok"
    assert run.output["transcript"] is False and run.output["transcript_segments"] == 0
    assert _entry(settings, repo, sid)["transcript"] is False
    assert any("文字起こし" in w for w in run.warnings)


def test_transcribe_whisper_python_is_pfb_apps_own(settings, repo, tmp_path, fake):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), auto_transcribe=True)
    _track_once(settings, repo, sid)
    argv = next(c["argv"] for c in fake.calls() if c["argv"][0] == "transcribe")
    assert "extern" not in argv[argv.index("--whisper-python") + 1]


def test_no_rule_defaults_unless_four_people_are_tracked(settings, repo, tmp_path, fake):
    positions = {"Pa": {"cam1": "左前"}, "Pb": {"cam1": "左奥"}, "Pc": {"cam1": "右前"},
                 "Pd": {"cam1": "右奥"}}
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), positions=positions)
    _, run = _track_once(settings, repo, sid)   # the fake tracker finds 3 people
    assert run.output["default_names"] == {} and run.output["default_source"] is None
    assert "rule-names" not in fake.commands()


def test_sheet_falls_back_to_the_default_frame(settings, repo, tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_PEOPLE", "4")
    monkeypatch.setenv("FAKE_ENGINE_SHEET_NO_AUTO", "1")
    positions = {"Pa": {"cam1": "左前"}, "Pb": {"cam1": "左奥"}, "Pc": {"cam1": "右前"},
                 "Pd": {"cam1": "右奥"}}
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), positions=positions)
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.done
    assert run.output["sheet_frame"] == 120 and run.output["sheet_frame_rule"] == "start"
    assert any("既定の画面" in w for w in run.warnings)
    rule = next(c["argv"] for c in fake.calls() if c["argv"][0] == "rule-names")
    assert "--auto-start" not in rule   # the same frame as the sheet


def test_sheet_uses_the_fullest_frame_when_nobody_is_there_at_the_start(
        settings, repo, tmp_path, fake, monkeypatch):
    """2026-11-15 free-video test: an intro / late start (people first boxed at frame 187 or
    1578) failed the whole track job with 「sheet: no boxes within 5.0s of frame 1」."""
    monkeypatch.setenv("FAKE_ENGINE_SHEET_NO_AUTO", "1")
    monkeypatch.setenv("FAKE_ENGINE_SHEET_NO_DEFAULT", "1")
    monkeypatch.setenv("FAKE_ENGINE_BOXES_FRAME", "187")
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.done
    argv = next(c["argv"] for c in fake.calls() if c["argv"][0] == "sheet" and "--frame" in c["argv"])
    assert argv[argv.index("--frame") + 1] == "187"
    assert run.output["sheet_frame"] == 187
    assert any("人が最も多く" in w for w in run.warnings)
    assert not any("既定の画面" in w for w in run.warnings)


def test_fullest_frame_of_a_boxes_file(tmp_path):
    from pfb_app.pipeline.tracking import _fullest_frame
    assert _fullest_frame(None) is None and _fullest_frame(tmp_path / "missing.txt") is None
    empty = tmp_path / "boxes.txt"
    empty.write_text("", encoding="utf-8")
    assert _fullest_frame(empty) is None
    busy = tmp_path / "busy.txt"
    busy.write_text("5,1,0,0,1,1,0.9\n9,1,0,0,1,1,0.9\n9,2,0,0,1,1,0.9\n12,1,0,0,1,1,0.9\n"
                    "12,2,0,0,1,1,0.9\n", encoding="utf-8")
    assert _fullest_frame(busy) == 9   # the earliest of the fullest frames


def test_track_with_nobody_in_the_video_stops_plainly(settings, repo, tmp_path, fake, monkeypatch):
    """The engine's sheet cannot make a sheet of nobody (KeyError: 1 on a waterfall clip): the
    job says so in plain words, with no engine message, and does not call it."""
    monkeypatch.setenv("FAKE_ENGINE_PEOPLE", "0")
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    job, run = _track_once(settings, repo, sid)
    assert job.status == JobStatus.failed
    assert "動画に人が見つかりませんでした" in run.error and "エンジン" not in run.error
    assert "sheet" not in fake.commands()


def test_engine_transcribe_only_heartbeats(settings, repo, tmp_path, monkeypatch):
    """whisper runs on the CPU: a busy GPU must not abort it, but the job keeps beating."""
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",), auto_transcribe=True)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "0.5")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP_CMD", "transcribe")
    pids = tmp_path / "transcribe.pids"   # written by the fake once transcribe is running
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids))
    beats = []
    real = Repo.heartbeat
    monkeypatch.setattr(Repo, "heartbeat", lambda self, job_id: beats.append(job_id) or
                        real(self, job_id))
    # the GPU gets busy once transcribe runs: asking then would abort the job
    ctx = make_ctx(settings, repo, yield_check=pids.exists)
    job = claim_run(repo, ctx, {"track"})
    assert job.status == JobStatus.done and pids.exists()
    assert repo.latest_run(sid, Stage.track, "cam1").output["transcript"] is True
    assert jid in beats


def _confirmed_mapping(settings, repo, sid, cam="cam1"):
    _track_once(settings, repo, sid, cam)
    mapping = teacher_mapping(repo, sid, cam)
    repo.set_identity_confirmed(sid, True)
    return mapping


def test_new_track_run_unconfirms_identity(settings, repo, tmp_path):
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    _confirmed_mapping(settings, repo, sid)
    assert repo.get_submission(sid)["identity_confirmed"]
    repo.enqueue(sid, Stage.track, "track", camera="cam1")   # e.g. track regenerated
    claim_run(repo, make_ctx(settings, repo), {"track"})
    # the teacher's mapping is void; the new track run gets its own automatic mapping, which
    # is confirmed again (the confirmation now refers to it, not to the teacher's)
    track = repo.latest_run(sid, Stage.track, "cam1")
    identity = repo.latest_run(sid, Stage.identity, "cam1")
    assert identity.source == "auto" and identity.output["track_run_id"] == track.id
    assert repo.get_submission(sid)["identity_confirmed"]
    repo.save_stage_run(track.model_copy(update={"id": None}))   # a track run with no follow-up
    assert not repo.get_submission(sid)["identity_confirmed"]


def test_behavior_refuses_a_mapping_of_an_older_track_run(settings, repo, tmp_path):
    from pfb_app.engine import ServerHandle
    from pfb_app.pipeline import tracking
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    _confirmed_mapping(settings, repo, sid)
    old_track = repo.latest_run(sid, Stage.track, "cam1")
    new_track = save(repo, sid, Stage.track, dict(old_track.output), camera="cam1",
                     upstream=old_track.upstream_run_ids)
    repo.set_identity_confirmed(sid, True)   # even if it were confirmed again by mistake
    repo.cancel_pending_jobs(sid, Stage.behavior, "cam1")   # the automatic people's one
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    job = repo.claim_next_job({"pfb3x3:standard"})
    ctx = make_ctx(settings, repo)
    ctx.server = ServerHandle(pid=1, port=8200, model="qwen3vl-8b",
                              base_url="http://localhost:8200/v1")
    run = tracking.run_behavior(ctx, job)
    assert run.status == "failed" and "最新の人物の追跡結果" in run.error
    assert new_track.id != old_track.id and jid == job.id


def test_run_track_gives_the_engine_the_profile_focus(settings, repo, tmp_path, fake):
    """A meeting work: the engine's activity line is the work's line plus the profile's
    English focus (its existing generic `activity` hook); a general work's stays the line."""
    aid, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    repo.conn.execute("UPDATE assignments SET profile='meeting', task_text='週の定例会'"
                      " WHERE id=?", (aid,))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    claim_run(repo, make_ctx(settings, repo), {"track"})
    entry = yaml.safe_load((files.submission_dir(settings, aid, sid) / "pfb3x3" / "videos"
                            / f"s{sid}_cam1.yaml").read_text(encoding="utf-8"))
    assert entry["domain"] == "generic"
    assert entry["activity"].startswith("週の定例会\nKind of activity: a meeting")


def _engine_with_frames(fake):
    """The pinned engine has `frames` (Pfb3x3Engine.has_frames_command reads its source)."""
    src = fake.engine_dir / "src" / "pfb3x3"
    src.mkdir(parents=True, exist_ok=True)
    (src / "engine.py").write_text("def cmd_frames(a):\n    pass\n", encoding="utf-8")


def test_track_caches_behaviors_frames_beside_it(settings, repo, tmp_path, fake):
    _engine_with_frames(fake)
    run = _run_track(settings, repo, tmp_path)
    assert run.status == "ok"
    calls = [c["argv"] for c in fake.calls()]
    frames = next(a for a in calls if a[0] == "frames")
    root = files.submission_dir(settings, repo.get_submission(run.submission_id)["assignment_id"],
                                run.submission_id) / "pfb3x3"
    assert frames[frames.index("--frames") + 1] == str(root / "frames")
    assert frames[frames.index("--condition") + 1] == settings.pfb3x3_condition
    assert (root / "frames" / f"s{run.submission_id}_cam1" / "f000001_w1920.jpg").exists()


def test_no_frame_cache_ahead_on_the_cpu_or_with_an_older_engine(settings, repo, tmp_path,
                                                                 fake, monkeypatch):
    run = _run_track(settings, repo, tmp_path)   # the fake engine dir has no cmd_frames
    assert run.status == "ok" and "frames" not in fake.commands()
    _engine_with_frames(fake)
    _, sid = tracked_submission(repo, tmp_path, cameras=("cam1",))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    claim_run(repo, make_ctx(settings, repo, device="cpu"), {"track"})
    assert "frames" not in fake.commands()


def test_a_failing_frame_cache_never_fails_the_track(settings, repo, tmp_path, fake,
                                                     monkeypatch):
    _engine_with_frames(fake)
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "frames")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    run = _run_track(settings, repo, tmp_path)
    assert run.status == "ok" and "frames" in fake.commands()
