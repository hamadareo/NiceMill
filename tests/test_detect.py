"""The `detect` stage (content check of a video): probe per camera, then scene with a server.

A Python FakeEngine stands in for Pfb3x3Engine.probe / scene (no subprocess, no GPU)."""
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Job, JobStatus, Kind, Stage
from pfb_app.engine import (EngineError, EngineNeedsGpu, EngineTimeout, EngineYield, ProbeResult,
                            SceneResult, ServerHandle)
from pfb_app.pipeline import detect
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import DETECT_KEY, enqueue_followups, execute_job, model_key_for
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

SERVER = ServerHandle(pid=1, port=8200, model="qwen3vl-8b", base_url="http://localhost:8200/v1")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", session_idle_sec=0)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


class FakeEngine:
    """probe / scene as the engine client returns them; records every call."""

    engine_commit = "0123456789abcdef0123"

    def __init__(self, scene_error: BaseException | None = None,
                 probe_error: BaseException | None = None, persons=14):
        self.calls: list[tuple] = []
        self.timeouts: list[float | None] = []
        self.scene_error, self.probe_error, self.persons = scene_error, probe_error, persons

    def probe(self, sub_dir, video_id, video_path, *, start_s=0.0, end_s=None, samples=8,
              device="cpu", on_tick=None, timeout_sec=None):
        self.calls.append(("probe", video_id, Path(video_path).name, start_s, end_s, samples,
                           device))
        self.timeouts.append(timeout_sec)
        if on_tick is not None:
            assert on_tick() is False   # CPU work: heartbeat only, never a GPU yield
        if self.probe_error is not None:
            raise self.probe_error
        frame = Path(sub_dir) / "pfb3x3" / "tracks" / f"{video_id}__probe" / "f0.jpg"
        return ProbeResult(duration_s=1500.0 if video_id.endswith("cam1") else 1490.0,
                           persons={"median": 12, "p90": self.persons, "max": 15},
                           vehicles={"median": 0, "p90": 0, "max": 0},
                           camera_motion={"median_shift_frac": 0.01, "moving": False},
                           frames=[frame], fps=30.0, width=1920, height=1080,
                           n_frames=45000, samples=samples)

    def scene(self, sub_dir, video_id, *, base_url, model, on_tick=None, timeout_sec=None):
        self.calls.append(("scene", video_id, base_url, model))
        self.timeouts.append(timeout_sec)
        if self.scene_error is not None:
            raise self.scene_error
        return SceneResult(scene="sport", sport="soccer", sport_text="soccer match",
                           numbers_visible=True, model=model)


def make_submission(repo, tmp_path, cameras=("cam1", "cam2"), meta=None):
    aid = repo.create_assignment("試合", Kind.video, "", "", "standard", True)
    paths = {}
    for c in cameras:
        p = tmp_path / f"{c}.mp4"
        p.write_bytes(b"not decoded")
        paths[c] = str(p)
    sid = repo.create_submission(aid, "G1", {"cameras": paths, "start_sec": 0.0, "end_sec": None,
                                             "engine": "pfb3x3", **(meta or {})})
    return sid


def claim(repo, sid) -> Job:
    jid = repo.enqueue(sid, Stage.detect, model_key_for(repo, sid, Stage.detect))
    job = repo.claim_next_job({DETECT_KEY})
    assert job is not None and job.id == jid
    return job


def ctx_for(settings, repo, engine, server=None, yield_check=lambda: False):
    return StageContext(settings, repo, None, yield_check, engine=engine, server=server)


def test_detect_model_key_and_settings(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path)
    assert DETECT_KEY == "detect"
    assert model_key_for(repo, sid, Stage.detect) == DETECT_KEY
    assert settings.detect_samples == 8 and settings.detect_model_key == "standard"
    assert settings.detect_timeout_sec == 600


def test_run_detect_with_server(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path, meta={"start_sec": 30.0, "end_sec": 900.0})
    job = claim(repo, sid)
    engine = FakeEngine()
    run = detect.run_detect(ctx_for(settings, repo, engine, server=SERVER), job)
    assert run.status == "ok" and run.camera is None and run.stage == Stage.detect
    out = run.output
    assert set(out["cameras"]) == {"cam1", "cam2"}
    cam1 = out["cameras"]["cam1"]
    assert cam1["persons"]["p90"] == 14 and cam1["duration_s"] == 1500.0
    assert cam1["camera_motion"] == {"median_shift_frac": 0.01, "moving": False}
    assert isinstance(cam1["frames"][0], str)
    assert out["scene"] == {"scene": "sport", "sport": "soccer", "sport_text": "soccer match",
                            "numbers_visible": True, "model": "qwen3vl-8b"}
    assert out["scene_skipped"] is None
    assert out["duration_s"] == 1500.0
    probes = [c for c in engine.calls if c[0] == "probe"]
    assert probes == [("probe", f"s{sid}_cam1", "cam1.mp4", 30.0, 900.0, 8, "cpu"),
                      ("probe", f"s{sid}_cam2", "cam2.mp4", 30.0, 900.0, 8, "cpu")]
    # the scene verdict once, on camera 1, through the session's server
    assert engine.calls[-1] == ("scene", f"s{sid}_cam1", SERVER.base_url, SERVER.model)
    assert run.warnings == []
    assert run.model_id.startswith("pfb3x3:qwen3vl-8b")
    assert engine.timeouts == [600, 600, 600]   # every check call is limited


def test_run_detect_samples_from_settings(settings, repo, tmp_path):
    settings = settings.model_copy(update={"detect_samples": 5})
    sid = make_submission(repo, tmp_path, cameras=("cam1",))
    engine = FakeEngine()
    detect.run_detect(ctx_for(settings, repo, engine), claim(repo, sid))
    assert engine.calls[0][5] == 5


def test_run_detect_cpu_only_marks_gpu_busy(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path)
    engine = FakeEngine()
    run = detect.run_detect(ctx_for(settings, repo, engine, server=None), claim(repo, sid))
    assert run.status == "ok"
    assert run.output["scene"] is None and run.output["scene_skipped"] == "gpu_busy"
    assert set(run.output["cameras"]) == {"cam1", "cam2"}
    assert [c[0] for c in engine.calls] == ["probe", "probe"]   # no scene without a server
    assert run.warnings == []


def test_run_detect_scene_error_is_warning(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path)
    engine = FakeEngine(scene_error=EngineError("エンジン scene が失敗しました (exit 1): boom",
                                                "Traceback secret"))
    run = detect.run_detect(ctx_for(settings, repo, engine, server=SERVER), claim(repo, sid))
    assert run.status == "ok"
    assert run.output["scene"] is None and run.output["scene_skipped"] == "failed"
    assert run.output["cameras"]["cam1"]["persons"]["p90"] == 14
    assert len(run.warnings) == 1
    assert "boom" not in run.warnings[0] and "scene" not in run.warnings[0]


@pytest.mark.parametrize("error", [EngineYield("yield"), EngineNeedsGpu("exit 3"),
                                   EngineTimeout("エンジン scene がタイムアウトしました (600 秒)")])
def test_run_detect_scene_gpu_trouble_is_gpu_busy(settings, repo, tmp_path, error):
    sid = make_submission(repo, tmp_path, cameras=("cam1",))
    engine = FakeEngine(scene_error=error)
    run = detect.run_detect(ctx_for(settings, repo, engine, server=SERVER), claim(repo, sid))
    assert run.status == "ok"
    assert run.output["scene"] is None and run.output["scene_skipped"] == "gpu_busy"
    assert run.warnings == []   # no verdict: the card asks the coach to pick


def test_scene_tick_asks_the_session_to_yield(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path, cameras=("cam1",))
    asked = []

    class TickingEngine(FakeEngine):
        def scene(self, sub_dir, video_id, *, base_url, model, on_tick=None, **kw):
            asked.append(on_tick())
            return super().scene(sub_dir, video_id, base_url=base_url, model=model, **kw)

    detect.run_detect(ctx_for(settings, repo, TickingEngine(), server=SERVER,
                              yield_check=lambda: True), claim(repo, sid))
    assert asked == [True]


def test_probe_tick_stops_the_idle_server_without_aborting_the_probe(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path, cameras=("cam1",))
    released, ticks = [], []

    class TickingEngine(FakeEngine):
        def probe(self, *a, on_tick=None, **kw):
            ticks.append(on_tick())   # the GPU is wanted while the probe runs
            ticks.append(on_tick())
            return super().probe(*a, **kw)

    ctx = ctx_for(settings, repo, None, server=SERVER, yield_check=lambda: True)
    engine = TickingEngine()
    ctx.engine = engine

    def release():
        released.append(ctx.server)
        ctx.server = None

    ctx.release_server = release
    run = detect.run_detect(ctx, claim(repo, sid))
    assert ticks == [False, False]          # the probe goes on
    assert released == [SERVER]             # the server is stopped once
    assert run.status == "ok" and run.output["scene_skipped"] == "gpu_busy"
    assert [c[0] for c in engine.calls] == ["probe"]


@pytest.mark.parametrize("error", [EngineError("エンジン probe が失敗しました (exit 1): ffprobe",
                                               "Traceback secret"),
                                   EngineNeedsGpu("exit 3"),
                                   EngineTimeout("エンジン probe がタイムアウトしました (600 秒)")])
def test_probe_failure_fails_run(settings, repo, tmp_path, error):
    sid = make_submission(repo, tmp_path)
    job = claim(repo, sid)
    engine = FakeEngine(probe_error=error)
    ctx = ctx_for(settings, repo, engine, server=SERVER)
    execute_job(job, ctx)
    run = repo.latest_run(sid, Stage.detect)
    assert run.status == "failed"
    assert run.error == "内容を確かめられませんでした。どんな場面の動画か、下から選んでください。"
    assert repo.get_job(job.id).status == JobStatus.failed   # not retried: the card asks instead
    assert [c[0] for c in engine.calls] == ["probe"]   # no scene after a failed probe


def test_detect_without_cameras_fails_plainly(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path, cameras=())
    run = detect.run_detect(ctx_for(settings, repo, FakeEngine()), claim(repo, sid))
    assert run.status == "failed" and run.error == "カメラの動画が登録されていません。"


def test_detect_has_no_followups(settings, repo, tmp_path):
    sid = make_submission(repo, tmp_path)
    job = claim(repo, sid)
    execute_job(job, ctx_for(settings, repo, FakeEngine(), server=SERVER))
    assert repo.get_job(job.id).status == JobStatus.done
    assert enqueue_followups(repo, job) == []
    rows = repo.conn.execute("SELECT stage FROM jobs").fetchall()
    assert [r[0] for r in rows] == ["detect"]   # prepare waits for the coach's confirm


def test_detect_of_a_document_records_its_kind_without_the_engine(settings, repo, tmp_path):
    aid = repo.create_assignment("作文", Kind.document, "", "", "standard", False)
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF-1.4")
    sid = repo.create_submission(aid, "G1", {"source_path": str(src)})
    job = claim(repo, sid)
    engine = FakeEngine()
    execute_job(job, ctx_for(settings, repo, engine))
    assert repo.get_job(job.id).status == JobStatus.done
    run = repo.latest_run(sid, Stage.detect)
    assert run.status == "ok" and run.output == {"kind": "document"}
    assert engine.calls == []


def test_run_detect_keeps_the_activity_verdict(settings, repo, tmp_path):
    """An engine with `scene --activity`: the kind of activity is stored with the verdict."""
    class WithActivity(FakeEngine):
        def scene(self, sub_dir, video_id, *, base_url, model, on_tick=None, timeout_sec=None):
            super().scene(sub_dir, video_id, base_url=base_url, model=model)
            return SceneResult(scene="group_work", sport=None, sport_text=None,
                               numbers_visible=False, model=model, activity="meeting",
                               activity_text="Meeting")

    sid = make_submission(repo, tmp_path)
    run = detect.run_detect(ctx_for(settings, repo, WithActivity(), server=SERVER),
                            claim(repo, sid))
    assert run.output["scene"]["activity"] == "meeting"
    assert run.output["scene"]["activity_text"] == "Meeting"
