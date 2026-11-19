"""execute_job / enqueue_followups: stage dispatch, job state and the follow-up rules."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pdfhelp import make_pdf as _make_pdf
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import runner
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import (SessionMustEnd, enqueue_followups, enqueue_once,
                                     execute_job, is_cuda_oom)
from pfb_app.preprocess.transcript import Segment
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

QUOTE = "We measured the distance to the shelter on a map"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def make_pdf(path: Path) -> Path:
    return _make_pdf(path, [QUOTE + "."])


def make_video(path: Path, seconds: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        writer.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def save(repo, sid, stage, output, camera=None, subject_id=None, source="model", status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=[], prompt_hash=None, model_id=None, output=output,
        status=status, error=None if status == "ok" else "x", warnings=[], duration_sec=0.0,
        created_at=""))


def responder(req):
    """Answer every stage with a minimal valid JSON, chosen by the requested schema."""
    title = req.json_schema.get("title")
    if title == "EvidenceOutput":
        return json.dumps({"items": [{"page": 1, "quote": QUOTE, "observed_content": "内容",
                                      "evidence_source": "text", "confidence": "high"}]})
    if title == "ContributionOutput":
        ids = [line.split(": ", 1)[1] for line in req.user_text.splitlines()
               if line.startswith("ref_id: ")]
        return json.dumps({"judgments": [
            {"ref_id": i, "contribution_interpretation": "解釈", "is_positive_contribution": True,
             "rubric_criteria": []} for i in ids]}, ensure_ascii=False)
    if title == "PfbOutput":
        ids = [line.split(": ", 1)[1] for line in req.user_text.splitlines()
               if line.startswith("ref_id: ")]
        return json.dumps({"pfb_status": "generated", "pfb": "調べたことが課題を進めていました。",
                           "evidence_refs": ids[:1]}, ensure_ascii=False)
    if title == "IdentityOutput":
        return json.dumps({"participants": [
            {"participant_id": "Pa", "appearance": "外見", "initial_position": "左",
             "verification_points": ["開始時点"], "identity_confidence": "high",
             "verified": True, "notes": ""}]}, ensure_ascii=False)
    raise AssertionError(f"unexpected schema {title}")


def make_ctx(settings, repo, backend=None, yield_check=lambda: False):
    return StageContext(settings, repo, backend or FakeBackend(responder), yield_check)


def doc_submission(repo, tmp_path, require_approval=False):
    aid = repo.create_assignment("課題", Kind.document, "防災を調べる課題", "ルーブリック",
                                 "standard", require_approval)
    src = make_pdf(tmp_path / "report.pdf")
    sid = repo.create_submission(aid, "A", {"source_path": str(src),
                                            "original_filename": "report.pdf"})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    return aid, sid, subj


def video_submission(repo, tmp_path, cameras=("cam1",), require_approval=False,
                     auto_transcribe=False, model_key="standard"):
    aid = repo.create_assignment("課題", Kind.video, "塔を作る課題", "ルーブリック", model_key,
                                 require_approval)
    paths = {c: str(make_video(tmp_path / f"{c}.mp4")) for c in cameras}
    sid = repo.create_submission(aid, "G1", {"cameras": paths, "start_sec": 1.0,
                                             "end_sec": None, "auto_transcribe": auto_transcribe})
    s_a = repo.add_subject(sid, "Pa", "A子", "縞模様", {c: "左" for c in cameras})
    s_n = repo.add_subject(sid, None, "名前のみ", None, {})
    return aid, sid, s_a, s_n


def run_job(repo, ctx, sid, stage, model_key, camera=None, subject_id=None):
    repo.enqueue(sid, stage, model_key, camera=camera, subject_id=subject_id)
    job = repo.claim_next_job({model_key})
    assert job is not None and job.stage == stage
    execute_job(job, ctx)
    return repo.get_job(job.id)


def jobs(repo, sid):
    rows = repo.conn.execute(
        "SELECT stage, camera, subject_id, model_key, status FROM jobs WHERE submission_id=?"
        " ORDER BY id", (sid,)).fetchall()
    return [tuple(r) for r in rows]


def queued(repo, sid):
    return [(s, c, subj, k) for s, c, subj, k, st in jobs(repo, sid) if st == "queued"]


# --- the tests named in the brief ---

def test_document_job_chain_runs_to_pfb(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    ctx = make_ctx(settings, repo)
    repo.enqueue(sid, Stage.prepare, "cpu")
    for keys in ({"cpu"}, {"standard"}, {"standard"}, {"standard"}):
        job = repo.claim_next_job(keys)
        execute_job(job, ctx)
        assert repo.get_job(job.id).status == JobStatus.done
    assert repo.claim_next_job({"cpu", "standard", "high", "whisper"}) is None
    assert [(s, k) for s, _, _, k, _ in jobs(repo, sid)] == [
        ("prepare", "cpu"), ("evidence", "standard"), ("contribution", "standard"),
        ("pfb", "standard")]
    prepare = repo.latest_run(sid, Stage.prepare)
    assert prepare.status == "ok" and prepare.camera is None
    assert set(prepare.output) == {"pages", "pdf_path"}
    assert QUOTE in prepare.output["pages"][0]["text"]
    assert Path(prepare.output["pdf_path"]).is_file()
    pfb = repo.latest_run(sid, Stage.pfb, subject_id=subj)
    assert pfb.status == "ok" and pfb.output["pfb_status"] == "generated"


def test_identity_approval_gate_blocks_behavior(settings, repo, tmp_path):
    aid, sid, s_a, _ = video_submission(repo, tmp_path, require_approval=True)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    ctx = make_ctx(settings, repo)
    job = run_job(repo, ctx, sid, Stage.identity, "standard", camera="cam1")
    assert job.status == JobStatus.done
    assert repo.latest_run(sid, Stage.identity, "cam1").status == "ok"
    assert queued(repo, sid) == []  # behavior waits for the teacher's confirmation

    # a new identity run (e.g. regeneration) voids the confirmation: behavior waits again
    repo.set_identity_confirmed(sid, True)
    job = run_job(repo, ctx, sid, Stage.identity, "standard", camera="cam1")
    assert not repo.get_submission(sid)["identity_confirmed"]
    assert queued(repo, sid) == []

    # once the teacher has confirmed, the follow-up rule lets behavior through
    repo.set_identity_confirmed(sid, True)
    assert len(enqueue_followups(repo, job)) == 1
    assert queued(repo, sid) == [("behavior", "cam1", None, "standard")]


def test_identity_without_approval_enqueues_behavior(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, require_approval=False, model_key="high")
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    run_job(repo, make_ctx(settings, repo), sid, Stage.identity, "high", camera="cam1")
    assert queued(repo, sid) == [("behavior", "cam1", None, "high")]


def _behavior_ready(repo, sid, camera):
    save(repo, sid, Stage.behavior, {"events": [{
        "video_time": "00:05", "participant": "Pa", "observed_behavior": "紙を折る",
        "utterance": "", "speaker": "unknown", "evidence_source": "video", "confidence": "high",
        "event_id": f"{camera}-0001", "camera": camera, "task_elapsed": "00:00"}]},
        camera=camera)


def test_pfb_enqueued_only_after_all_cameras(settings, repo, tmp_path):
    aid, sid, s_a, s_n = video_submission(repo, tmp_path, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        _behavior_ready(repo, sid, cam)
    ctx = make_ctx(settings, repo)
    run_job(repo, ctx, sid, Stage.contribution, "standard", camera="cam1")
    assert queued(repo, sid) == []
    run_job(repo, ctx, sid, Stage.contribution, "standard", camera="cam2")
    # one pfb per subject that has a participant id (the video pfb needs one)
    assert queued(repo, sid) == [("pfb", None, s_a, "standard")]


def test_pfb_not_enqueued_while_other_camera_contribution_is_stale(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        _behavior_ready(repo, sid, cam)
    ctx = make_ctx(settings, repo)
    run_job(repo, ctx, sid, Stage.contribution, "standard", camera="cam2")
    repo.conn.execute("UPDATE jobs SET status='done'")
    _behavior_ready(repo, sid, "cam2")  # cam2 behavior edited -> its contribution is stale
    run_job(repo, ctx, sid, Stage.contribution, "standard", camera="cam1")
    assert queued(repo, sid) == []


def test_yield_requeues_job_and_reraises(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    save(repo, sid, Stage.prepare, {"pages": [{"number": 1, "text": QUOTE, "image_path": None}],
                                    "pdf_path": "x.pdf"})
    repo.enqueue(sid, Stage.evidence, "standard")
    job = repo.claim_next_job({"standard"})
    with pytest.raises(YieldRequested):
        execute_job(job, make_ctx(settings, repo, yield_check=lambda: True))
    assert repo.get_job(job.id).status == JobStatus.queued
    assert repo.latest_run(sid, Stage.evidence) is None
    assert queued(repo, sid) == [("evidence", None, None, "standard")]


class _Raising:
    model_id = "boom"

    def __init__(self, exc):
        self.exc = exc

    def generate(self, reqs):
        raise self.exc

    def close(self):
        pass


def _evidence_job(repo, sid):
    save(repo, sid, Stage.prepare, {"pages": [{"number": 1, "text": QUOTE, "image_path": None}],
                                    "pdf_path": "x.pdf"})
    repo.enqueue(sid, Stage.evidence, "standard")
    return repo.claim_next_job({"standard"})


def test_oom_marks_retryable_and_raises_session_must_end(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    job = _evidence_job(repo, sid)
    # the one automatic retry is services.auto_retry's (test_auto_retry); off here
    ctx = make_ctx(settings.model_copy(update={"auto_retry": False}), repo,
                   backend=_Raising(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
    with pytest.raises(SessionMustEnd):
        execute_job(job, ctx)
    done = repo.get_job(job.id)
    assert done.status == JobStatus.failed_retryable and "CUDA out of memory" in done.error
    assert queued(repo, sid) == []


def test_other_exception_marks_retryable_without_ending_session(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    job = _evidence_job(repo, sid)
    execute_job(job, make_ctx(settings, repo, backend=_Raising(ValueError("壊れた応答"))))
    done = repo.get_job(job.id)
    assert done.status == JobStatus.failed_retryable and "壊れた応答" in done.error
    assert queued(repo, sid) == []


def test_failed_stage_run_marks_job_failed_without_followups(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    job = _evidence_job(repo, sid)
    execute_job(job, make_ctx(settings, repo, backend=FakeBackend(lambda r: "not json")))
    done = repo.get_job(job.id)
    assert done.status == JobStatus.failed and done.error
    assert repo.latest_run(sid, Stage.evidence).status == "failed"
    assert queued(repo, sid) == []


def test_is_cuda_oom_uses_torch_class_when_loaded(monkeypatch):
    class FakeOOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeOOM)))
    assert is_cuda_oom(FakeOOM("anything"))
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    wrapped = RuntimeError("generation failed")
    wrapped.__cause__ = FakeOOM("x")
    assert is_cuda_oom(wrapped)
    assert not is_cuda_oom(RuntimeError("other"))


def test_finished_job_not_overwritten_after_recovery(settings, repo, tmp_path):
    """A slow job that recover_stale_jobs already flipped stays failed_retryable."""
    aid, sid, subj = doc_submission(repo, tmp_path)
    job = _evidence_job(repo, sid)
    repo.conn.execute("UPDATE jobs SET status='failed_retryable', error='heartbeat timeout'"
                      " WHERE id=?", (job.id,))
    execute_job(job, make_ctx(settings, repo))
    assert repo.get_job(job.id).status == JobStatus.failed_retryable
    assert queued(repo, sid) == []  # not marked done -> no follow-ups


# --- prepare (CPU) ---

def test_video_prepare_one_run_per_camera_then_identity(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, cameras=("cam1", "cam2"))
    job = run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.prepare, "cpu")
    assert job.status == JobStatus.done
    for cam in ("cam1", "cam2"):
        run = repo.latest_run(sid, Stage.prepare, cam)
        assert run.status == "ok"
        frames = run.output["frames"]
        assert [f["video_sec"] for f in frames] == [1.0, 6.0, 11.0]
        assert all(Path(f["path"]).parent == files.frames_dir(settings, aid, sid, cam)
                   for f in frames)
        assert all(Path(f["path"]).is_file() for f in frames)
    assert queued(repo, sid) == [("identity", "cam1", None, "standard"),
                                 ("identity", "cam2", None, "standard")]


def test_video_prepare_with_auto_transcribe_enqueues_whisper(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, auto_transcribe=True)
    run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.prepare, "cpu")
    assert queued(repo, sid) == [("transcribe", None, None, "whisper")]


def test_teacher_transcript_skips_transcribe(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, auto_transcribe=True)
    save(repo, sid, Stage.transcribe, {"segments": [{"start": 0.0, "end": 9.0, "text": "はい"}]},
         source="teacher")
    run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.prepare, "cpu")
    assert queued(repo, sid) == [("identity", "cam1", None, "standard")]


def test_video_prepare_error_saves_failed_run_and_fails_job(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path)
    repo.update_submission_meta(sid, {"cameras": {"cam1": str(tmp_path / "missing.mp4")},
                                      "start_sec": None, "end_sec": None,
                                      "auto_transcribe": False})
    job = run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.prepare, "cpu")
    assert job.status == JobStatus.failed and "動画ファイルが見つかりません" in job.error
    run = repo.latest_run(sid, Stage.prepare, "cam1")
    assert run.status == "failed" and "動画ファイルが見つかりません" in run.error
    assert queued(repo, sid) == []


def test_document_prepare_error_saves_failed_run_and_fails_job(settings, repo, tmp_path):
    aid, sid, subj = doc_submission(repo, tmp_path)
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    repo.update_submission_meta(sid, {"source_path": str(bad), "original_filename": "notes.txt"})
    job = run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.prepare, "cpu")
    assert job.status == JobStatus.failed and "対応していないファイル形式" in job.error
    run = repo.latest_run(sid, Stage.prepare)
    assert run.status == "failed" and run.camera is None
    assert queued(repo, sid) == []


# --- transcribe ---

def test_transcribe_saves_run_then_enqueues_identity(settings, repo, tmp_path, monkeypatch):
    aid, sid, *_ = video_submission(repo, tmp_path, cameras=("cam1", "cam2"), auto_transcribe=True)
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    seen = []
    monkeypatch.setattr(runner, "transcribe", lambda path, s: seen.append(path) or [
        Segment(0.0, 4.0, "こんにちは")])
    job = run_job(repo, make_ctx(settings, repo, backend=None), sid, Stage.transcribe, "whisper")
    assert job.status == JobStatus.done
    assert seen == [str(tmp_path / "cam1.mp4")]
    run = repo.latest_run(sid, Stage.transcribe)
    assert run.source == "model" and run.output == {
        "segments": [{"start": 0.0, "end": 4.0, "text": "こんにちは"}]}
    assert queued(repo, sid) == [("identity", "cam1", None, "standard"),
                                 ("identity", "cam2", None, "standard")]


def test_transcribe_checks_yield_before_loading_whisper(settings, repo, tmp_path, monkeypatch):
    aid, sid, *_ = video_submission(repo, tmp_path, auto_transcribe=True)
    monkeypatch.setattr(runner, "transcribe", lambda *a: pytest.fail("must not transcribe"))
    repo.enqueue(sid, Stage.transcribe, "whisper")
    job = repo.claim_next_job({"whisper"})
    with pytest.raises(YieldRequested):
        execute_job(job, make_ctx(settings, repo, backend=None, yield_check=lambda: True))
    assert repo.get_job(job.id).status == JobStatus.queued


# --- idempotent follow-ups ---

def test_followups_are_idempotent(settings, repo, tmp_path):
    aid, sid, *_ = video_submission(repo, tmp_path, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    repo.enqueue(sid, Stage.prepare, "cpu")
    job = repo.claim_next_job({"cpu"})
    enqueue_followups(repo, job)
    enqueue_followups(repo, job)
    assert queued(repo, sid) == [("identity", "cam1", None, "standard"),
                                 ("identity", "cam2", None, "standard")]
    repo.finish_job(job.id, JobStatus.done)
    first = repo.claim_next_job({"standard"})  # running counts as active too
    assert enqueue_once(repo, sid, Stage.identity, camera="cam1") is None
    repo.finish_job(first.id, JobStatus.done)
    assert enqueue_once(repo, sid, Stage.identity, camera="cam1") is not None


def test_pfb_waits_for_registered_camera_without_prepare(settings, repo, tmp_path):
    """cam2 is registered but not prepared yet (per-camera prepare): no pfb from cam1 alone."""
    aid, sid, *_ = video_submission(repo, tmp_path, cameras=("cam1", "cam2"))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    _behavior_ready(repo, sid, "cam1")
    run_job(repo, make_ctx(settings, repo), sid, Stage.contribution, "standard", camera="cam1")
    assert queued(repo, sid) == []


# --- the names stage (roster cues after track) ---

@pytest.fixture
def nrepo(tmp_path):
    """A repo whose settings name an engine dir with a gate file enabling every cue."""
    s = Settings(data_dir=tmp_path / "ndata", engine_dir=tmp_path / "engine",
                 prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    write_gate(s, {c: {"enabled": True, "threshold": None, "reason": "ok"}
                   for c in ("number", "team", "position", "grid", "batting", "appearance",
                             "place")})
    conn = connect(files.db_path(s))
    init_schema(conn)
    return Repo(conn, s)


def write_gate(settings, cues, model="qwen3vl-32b"):
    f = settings.engine_dir / "configs" / "cue_gate.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({model: cues}), encoding="utf-8")   # JSON is YAML


def tracked_work(repo, tmp_path, *, domain="school", sport=None, roster=None, model_key="high",
                 cameras=("cam1", "cam2")):
    aid = repo.create_assignment("試合", Kind.video, "練習試合", "", model_key, True,
                                 domain=domain, sport=sport)
    sid = repo.create_submission(aid, "G1", {
        "cameras": {c: str(tmp_path / f"{c}.mp4") for c in cameras}, "engine": "pfb3x3"})
    for n, attrs in enumerate(roster or [], 1):
        places = {c: attrs["place"] for c in cameras} if attrs.get("place") else {}
        repo.add_subject(sid, f"r{n}", attrs.get("name") or f"選手{n}", attrs.get("appearance"),
                         places, attrs)
    return sid


def track_job(repo, sid, camera):
    save(repo, sid, Stage.track, {"labels": ["P1"], "video_id": f"s{sid}_{camera}"},
         camera=camera)
    return SimpleNamespace(submission_id=sid, stage=Stage.track, camera=camera, subject_id=None)


def queued_jobs(repo, sid):
    return [(s, c, k) for s, c, k in repo.conn.execute(
        "SELECT stage, camera, model_key FROM jobs WHERE submission_id=? AND status='queued'"
        " ORDER BY id", (sid,)).fetchall()]


def test_names_model_key_is_the_pfb3x3_session(nrepo, tmp_path):
    sid = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")
    assert runner.model_key_for(nrepo, sid, Stage.names) == "pfb3x3:high"


def test_track_followup_enqueues_names_for_sports(nrepo, tmp_path):
    sid = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")   # no roster needed
    assert runner.needs_names(nrepo, nrepo.get_submission(sid))
    new = enqueue_followups(nrepo, track_job(nrepo, sid, "cam1"))
    assert len(new) == 1   # the team cue and the numbers that name people: on the server
    assert queued_jobs(nrepo, sid) == [("names", "cam1", "pfb3x3:high")]
    assert enqueue_followups(nrepo, track_job(nrepo, sid, "cam1")) == []   # once per camera


def test_track_followup_none_for_school_without_roster(nrepo, tmp_path):
    sid = tracked_work(nrepo, tmp_path, roster=[{"name": "A子"}, {"name": "B太"}])
    nrepo.add_subject(sid, "Px", "C美", "赤い服", {"cam1": "左前"})   # typed, not a roster
    assert not runner.needs_names(nrepo, nrepo.get_submission(sid))
    assert enqueue_followups(nrepo, track_job(nrepo, sid, "cam1")) == []
    assert queued_jobs(nrepo, sid) == []


@pytest.mark.parametrize("role,value,key", [
    ("number", "7", "pfb3x3:high"), ("grid", "1", "pfb3x3:high"),
    ("batting", "1", "pfb3x3:high"), ("appearance", "赤い服", "pfb3x3:high"),
    ("place", "左前", "names:cpu")])
def test_track_followup_enqueues_names_for_a_school_roster_cue(nrepo, tmp_path, role, value,
                                                                key):
    sid = tracked_work(nrepo, tmp_path, roster=[{"name": "A子", role: value}])
    assert runner.needs_names(nrepo, nrepo.get_submission(sid))
    enqueue_followups(nrepo, track_job(nrepo, sid, "cam2"))
    assert queued_jobs(nrepo, sid) == [("names", "cam2", key)]


def test_track_followup_none_when_no_cue_can_run(nrepo, tmp_path):
    # a position column without a sport: needs_names holds, but the position cue has no input
    sid = tracked_work(nrepo, tmp_path, roster=[{"name": "A子", "position": "FW"}])
    assert runner.needs_names(nrepo, nrepo.get_submission(sid))
    assert enqueue_followups(nrepo, track_job(nrepo, sid, "cam1")) == []


def test_sports_names_without_number_names_need_no_server(nrepo, tmp_path):
    nrepo.settings = nrepo.settings.model_copy(update={"number_names": False})
    sid = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")
    enqueue_followups(nrepo, track_job(nrepo, sid, "cam1"))
    assert queued_jobs(nrepo, sid) == [("names", "cam1", "names:cpu")]   # only the team cue


def test_track_followup_reads_numbers_for_names_even_with_every_cue_off(nrepo, tmp_path):
    write_gate(nrepo.settings, {}, model="other-model")   # no section for this model
    sid = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")
    enqueue_followups(nrepo, track_job(nrepo, sid, "cam1"))
    enqueue_followups(nrepo, track_job(nrepo, sid, "cam2"))
    # both cameras keep their names job (the automatic confirmation does not cancel them)
    assert [j for j in queued_jobs(nrepo, sid) if j[0] == "names"] == [
        ("names", "cam1", "pfb3x3:high"), ("names", "cam2", "pfb3x3:high")]


def test_track_followup_none_when_the_gate_disables_every_cue(nrepo, tmp_path):
    nrepo.settings = nrepo.settings.model_copy(update={"number_names": False})
    write_gate(nrepo.settings, {"team": {"enabled": False, "reason": "precision 0.80"},
                                "number": {"enabled": True, "reason": "ok"}})
    sport = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")
    assert enqueue_followups(nrepo, track_job(nrepo, sport, "cam1")) == []   # no vLLM for nothing
    numbered = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer",
                            roster=[{"name": "山田", "number": "10"}])
    enqueue_followups(nrepo, track_job(nrepo, numbered, "cam1"))
    assert queued_jobs(nrepo, numbered) == [("names", "cam1", "pfb3x3:high")]
    write_gate(nrepo.settings, {}, model="other-model")   # no section for this model
    enqueue_followups(nrepo, track_job(nrepo, numbered, "cam2"))
    # no names job for cam2 (the automatic confirmation of both cameras also cancels cam1's)
    assert [j for j in queued_jobs(nrepo, numbered) if j[0] == "names"] == []


def test_names_followup_starts_nothing(nrepo, tmp_path):
    sid = tracked_work(nrepo, tmp_path, domain="sports", sport="soccer")
    job = SimpleNamespace(submission_id=sid, stage=Stage.names, camera="cam1", subject_id=None)
    assert enqueue_followups(nrepo, job) == []


def test_request_names_after_roster_change(nrepo, tmp_path):
    sid = tracked_work(nrepo, tmp_path, roster=[{"name": "A子"}],
                       cameras=("cam1", "cam2", "cam3"))
    save(nrepo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(nrepo, sid, Stage.prepare, {"frames": []}, camera="cam2")
    save(nrepo, sid, Stage.prepare, {"frames": []}, camera="cam3")
    track_job(nrepo, sid, "cam1")
    track_job(nrepo, sid, "cam2")
    save(nrepo, sid, Stage.track, None, camera="cam3", status="failed")
    assert runner.request_names(nrepo, sid) == []   # no roster cue: nothing to do
    # the coach gives a roster with numbers (Task 7's apply_roster replaces the subjects)
    nrepo.delete_subjects(sid)
    nrepo.add_subject(sid, "r1", "山田", None, {}, {"name": "山田", "number": "10"})
    ids = runner.request_names(nrepo, sid)
    assert len(ids) == 2
    assert queued_jobs(nrepo, sid) == [("names", "cam1", "pfb3x3:high"),
                                       ("names", "cam2", "pfb3x3:high")]
    assert runner.request_names(nrepo, sid) == []   # already queued
    # a running names job reads the old subjects: a fresh one is queued behind it
    nrepo.conn.execute("UPDATE jobs SET status='running' WHERE camera='cam1'")
    assert len(runner.request_names(nrepo, sid)) == 1
    assert queued_jobs(nrepo, sid) == [("names", "cam2", "pfb3x3:high"),
                                       ("names", "cam1", "pfb3x3:high")]
    # a queued job under the wrong key (the cues changed) is replaced
    nrepo.conn.execute("UPDATE jobs SET model_key='names:cpu' WHERE camera='cam2'")
    assert len(runner.request_names(nrepo, sid)) == 1
    assert sorted(queued_jobs(nrepo, sid)) == [("names", "cam1", "pfb3x3:high"),
                                               ("names", "cam2", "pfb3x3:high")]


def test_names_job_on_a_document_is_refused(settings, repo, tmp_path):
    _, sid, _ = doc_submission(repo, tmp_path)
    repo.enqueue(sid, Stage.names, "pfb3x3:standard")
    job = repo.claim_next_job({"pfb3x3:standard"})
    execute_job(job, make_ctx(settings, repo))
    assert repo.get_job(job.id).status == JobStatus.failed_retryable
