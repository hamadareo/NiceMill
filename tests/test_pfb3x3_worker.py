"""Worker sessions for the tracking engine: `track` and `pfb3x3:<key>`. FakeEngine only, no GPU."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fakes.engine_env import alive, fake_translator, install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.engine import LOG_PREFIX, Pfb3x3Engine, ProbeResult, SceneResult
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker.main import worker_loop
from pfb_app.worker.session import (DETECT_CPU_SESSION, EXIT_IDLE, EXIT_LOAD_FAILED,
                                    EXIT_NEEDS_GPU, EXIT_SERVER_DOWN, EXIT_YIELD, TRACK_CPU_SESSION,
                                    exit_on_signals, run_session)

POLL = 7
TOTAL_GB = 97.9


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fake):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=fake.engine_dir, engine_python=fake.engine_python,
                    gpu_poll_sec=POLL, session_idle_sec=0)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


class Guard(GpuGuard):
    """The real GpuGuard on a scripted nvidia-smi; `yield_when()` decides should_yield."""

    def __init__(self, settings, free_gb=90.0, yield_when=lambda: False):
        self.free_gb = free_gb
        self.yield_when = yield_when
        self.can_start_calls: list[float] = []
        self.yield_calls = 0
        super().__init__(settings, query=lambda: GpuSnapshot(TOTAL_GB, TOTAL_GB - self.free_gb))

    def can_start(self, budget_gb):
        self.can_start_calls.append(budget_gb)
        return super().can_start(budget_gb)

    def should_yield(self):
        self.yield_calls += 1
        return self.yield_when()


class RecordingEngine(Pfb3x3Engine):
    """Fast ticks; remembers the servers it started and every stop_stale_server call."""

    def __init__(self, settings):
        super().__init__(settings, tick_sec=0.05, term_grace_sec=2)
        self.servers = []
        self.stale_stops = 0

    def start_server(self, model, util, *, on_tick=None, **kw):
        self.start_kwargs = kw
        handle = super().start_server(model, util, on_tick=on_tick, **kw)
        self.servers.append(handle)
        return handle

    def stop_stale_server(self):
        self.stale_stops += 1
        super().stop_stale_server()

    def translator(self, handle):   # no HTTP to the fake "vllm"
        self.translated_for = handle
        return fake_translator()

    def chat_backend(self, handle):   # contribution / pfb on the session's server
        self.chat_for = handle
        return fake_translator()


def submission(repo, tmp_path, cameras=("cam1",), model_key="standard"):
    aid = repo.create_assignment("課題", Kind.video, "紙タワーを作る課題", "ルーブリック",
                                 model_key, True)
    paths = {}
    for c in cameras:
        p = tmp_path / f"{c}.mp4"
        p.write_bytes(b"not decoded by the fake engine")
        paths[c] = str(p)
    sid = repo.create_submission(aid, "G1", {"cameras": paths, "start_sec": 0.0,
                                             "end_sec": None, "engine": "pfb3x3"})
    repo.add_subject(sid, "Pa", "A子", "外見", {})
    return aid, sid


def save(repo, sid, stage, output, camera=None, source="model", upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def ready_for_behavior(repo, sid, camera="cam1"):
    """prepare + track + the teacher's confirmed mapping for one camera."""
    prepare = save(repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    track = save(repo, sid, Stage.track, {"people": 1, "labels": ["P1"], "video_id": f"s{sid}_{camera}",
                                          "sheet": "x", "track_stats": {}, "default_names": {},
                                          "default_source": None, "device": "cuda"},
                 camera=camera, upstream=[prepare.id])
    identity = save(repo, sid, Stage.identity, {
        "participants": [{"participant_id": "Pa", "label": "P1", "display_name": "A子",
                          "note": ""}], "labels": ["P1"], "track_run_id": track.id},
        camera=camera, source="teacher", upstream=[track.id])
    repo.set_identity_confirmed(sid, True)
    return prepare, track, identity


def run(settings, key, guard, engine, **kw):
    return run_session(None, key, settings=settings, guard=guard,
                       engine_factory=lambda s: engine, sleep=lambda s: None,
                       heartbeat_sec=kw.pop("heartbeat_sec", 0.05), **kw)


def runtime_pid_file(settings) -> Path:
    return settings.data_dir / "pfb3x3_runtime" / "serve.pid"


# --- worker: session order, budgets, waiting_gpu ---

def test_session_order_includes_track_and_pfb3x3_keys(settings, repo, tmp_path):
    _, sid = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:high", camera="cam1")
    repo.enqueue(sid, Stage.contribution, "standard", camera="cam1")
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    repo.enqueue(sid, Stage.transcribe, "whisper")
    guard = Guard(settings, free_gb=1.0)
    worker_loop(settings, guard=guard, spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True)
    # whisper, track, the paper models, then the pfb3x3 sessions with their own budgets
    assert guard.can_start_calls == [6, 4, 24, 34, 60]
    rows = repo.conn.execute("SELECT status FROM jobs").fetchall()
    assert {r[0] for r in rows} == {"waiting_gpu"}


def test_pfb3x3_waits_for_budget_plus_margin(settings, repo, tmp_path):
    _, sid = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, spawned = Guard(settings, free_gb=41.9), []
    worker_loop(settings, guard=guard, spawn=lambda key: spawned.append(key) or 0,
                sleep=lambda s: None, once=True)
    job = repo.get_job(jid)
    assert spawned == [] and job.status == JobStatus.waiting_gpu
    assert "必要 42GB" in job.error and "予算34" in job.error
    guard.free_gb = 42.0
    worker_loop(settings, guard=guard, spawn=lambda key: spawned.append(key) or 0,
                sleep=lambda s: None, once=True)
    assert spawned == ["pfb3x3:standard"]


def test_track_cpu_fallback_is_off_by_default(settings, repo, tmp_path):
    assert settings.track_cpu_fallback is False
    _, sid = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    now = [0.0]

    class Stop(Exception):
        pass

    def sleep(sec):
        now[0] += 600
        if now[0] > 7200:
            raise Stop()

    with pytest.raises(Stop):
        worker_loop(settings, guard=Guard(settings, free_gb=5.0),
                    spawn=lambda key: pytest.fail("must wait for the GPU"), sleep=sleep,
                    clock=lambda: now[0])
    assert repo.get_job(jid).status == JobStatus.waiting_gpu


def test_track_falls_back_to_cpu_after_20_minutes(settings, repo, tmp_path):
    settings = settings.model_copy(update={"track_cpu_fallback": True})
    _, sid = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    now = [0.0]
    spawned = []

    class Stop(Exception):
        pass

    def sleep(sec):
        now[0] += 600  # each pass: ten minutes later
        if len(spawned) or now[0] > 3600:
            raise Stop()

    def spawn(key):  # what the CPU session does: claim the track job and finish it
        spawned.append(key)
        job = repo.claim_next_job({"track"})
        repo.finish_job(job.id, JobStatus.done)
        return EXIT_IDLE

    guard = Guard(settings, free_gb=5.0)  # 4 GB budget + 8 GB margin never fits
    with pytest.raises(Stop):
        worker_loop(settings, guard=guard, spawn=spawn, sleep=sleep, clock=lambda: now[0])
    # t=0, 600: waiting_gpu; t=1200: 20 min without a start -> the CPU track session
    assert spawned == [TRACK_CPU_SESSION]
    assert guard.can_start_calls == [4, 4, 4]
    assert repo.get_job(jid).status == JobStatus.done


@pytest.mark.parametrize("stage,key,expected", [
    (Stage.behavior, "pfb3x3:standard", ["pfb3x3:standard"]),
    (Stage.track, "track", ["track"]),           # its tracker may still hold the GPU
    (Stage.detect, "detect", ["detect"]),        # its server may still hold the GPU
    (Stage.contribution, "standard", []),        # a paper session never touches the engine
])
def test_crashed_session_cleans_up_engine_leftovers(settings, repo, tmp_path, stage, key,
                                                     expected):
    _, sid = submission(repo, tmp_path)
    repo.enqueue(sid, stage, key, camera="cam1")
    cleanups = []
    worker_loop(settings, guard=Guard(settings), spawn=lambda k: -9,
                sleep=lambda s: None, once=True, engine_cleanup=cleanups.append)
    assert cleanups == expected


def test_default_cleanup_with_nothing_left_is_harmless(settings, repo, tmp_path, fake):
    _, sid = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    worker_loop(settings, guard=Guard(settings), spawn=lambda k: -9,
                sleep=lambda s: None, once=True)
    assert fake.commands() == []   # no serve.pid, no call record: nothing is run or signalled


# --- pfb3x3 session ---

def test_pfb3x3_session_runs_behavior_and_stops_server(settings, repo, tmp_path, fake):
    _, sid = submission(repo, tmp_path)
    prepare, track, identity = ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    # a server left behind by a crashed session of ours
    runtime = settings.data_dir / "pfb3x3_runtime"
    runtime.mkdir(parents=True)
    leftover = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "vllm"],
                                start_new_session=True)
    (runtime / "serve.pid").write_text(f"{leftover.pid}\n", encoding="utf-8")
    engine = RecordingEngine(settings)

    code = run(settings, "pfb3x3:standard", Guard(settings), engine)

    assert code == EXIT_IDLE
    assert leftover.wait(timeout=10) is not None  # stopped before the new server started
    cmds = fake.commands()
    assert cmds[:2] == ["stop", "serve"] and cmds[-1] == "stop"
    serve = next(c for c in fake.calls() if c["argv"][0] == "serve")
    assert serve["argv"][1:] == ["qwen3vl-8b", "8200"]
    assert serve["env"]["PFB_GPU_UTIL"] == "0.347"  # 34 GB of 97.9 GB
    behavior = next(c["argv"] for c in fake.calls() if c["argv"][0] == "behavior")
    assert behavior[behavior.index("--base-url") + 1] == "http://localhost:8200/v1"
    assert behavior[behavior.index("--model") + 1] == "qwen3vl-8b"
    assert behavior[behavior.index("--condition") + 1] == "clip_som"
    assert repo.get_job(jid).status == JobStatus.done
    run_ = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run_.status == "ok" and run_.source == "model"
    assert engine.translated_for == engine.servers[0]   # translated by the session's server
    assert [e["observed_behavior"] for e in run_.output["events"]] == ["紙を折る"]
    assert run_.upstream_run_ids == [identity.id, prepare.id, track.id]
    assert run_.model_id == "pfb3x3:qwen3vl-8b"
    # contribution and pfb follow in the same session, on the same server
    rows = repo.conn.execute("SELECT stage, model_key, status FROM jobs ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("behavior", "pfb3x3:standard", "done"),
                                        ("contribution", "pfb3x3:standard", "done"),
                                        ("pfb", "pfb3x3:standard", "done")]
    assert engine.chat_for == engine.servers[0]
    assert repo.latest_run(sid, Stage.contribution, "cam1").model_id == "fake-translator"
    # the server is stopped at the end of the session; nothing of ours is left
    assert len(engine.servers) == 1 and not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()


def test_a_pending_content_check_ends_a_pfb3x3_session_between_jobs(settings, repo, tmp_path,
                                                                   fake):
    _, sid = submission(repo, tmp_path, cameras=("cam1", "cam2"))
    ready_for_behavior(repo, sid, "cam1")
    ready_for_behavior(repo, sid, "cam2")
    first = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    second = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam2")
    (tmp_path / "other").mkdir()
    _, other = submission(repo, tmp_path / "other")          # another work's content check
    check = repo.enqueue(other, Stage.detect, "detect")
    engine = RecordingEngine(settings)

    code = run(settings, "pfb3x3:standard", Guard(settings), engine)

    assert code == EXIT_YIELD
    assert repo.get_job(first).status == JobStatus.done
    assert repo.get_job(second).status == JobStatus.queued   # left for the next session
    assert repo.get_job(check).status == JobStatus.queued
    assert len(engine.servers) == 1 and not alive(engine.servers[0].pid)


def test_a_pending_content_check_ends_a_track_session_between_jobs(settings, repo, tmp_path,
                                                                  fake):
    _, sid = submission(repo, tmp_path, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    jobs = [repo.enqueue(sid, Stage.track, "track", camera=c) for c in ("cam1", "cam2")]
    repo.enqueue(sid, Stage.detect, "detect")

    code = run(settings, "track", Guard(settings), RecordingEngine(settings))

    assert code == EXIT_YIELD
    assert [repo.get_job(j).status for j in jobs] == [JobStatus.done, JobStatus.queued]


def test_pfb3x3_needs_gpu_at_start_marks_waiting(settings, repo, tmp_path, fake, monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    code = run(settings, "pfb3x3:standard", Guard(settings), RecordingEngine(settings))
    assert code == EXIT_NEEDS_GPU
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu and "GPU" in job.error
    assert "behavior" not in fake.commands()
    assert not runtime_pid_file(settings).exists()


def test_pfb3x3_yield_mid_run_requeues_and_stops_server(settings, repo, tmp_path, fake,
                                                        monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    pids = tmp_path / "behavior.pids"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids))
    engine = RecordingEngine(settings)
    guard = Guard(settings, yield_when=pids.exists)  # GPU gets busy once behavior is running

    code = run(settings, "pfb3x3:standard", guard, engine)

    assert code == EXIT_YIELD
    job = repo.get_job(jid)
    assert job.status == JobStatus.queued
    assert repo.latest_run(sid, Stage.behavior, "cam1") is None
    for pid in pids.read_text().split():
        assert not alive(int(pid))  # the behavior subprocess and its child are gone
    assert not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()
    assert fake.commands()[-1] == "stop"


def test_pfb3x3_engine_crash_is_retryable_and_leaves_no_server(settings, repo, tmp_path, fake,
                                                               monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "behavior")
    engine = RecordingEngine(settings)

    code = run(settings, "pfb3x3:standard", Guard(settings), engine)

    assert code == EXIT_IDLE
    job = repo.get_job(jid)
    assert job.status == JobStatus.failed_retryable
    assert "fake engine behavior: failure 1" in job.error  # the engine's stderr tail
    assert not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()
    assert list((settings.data_dir / "pfb3x3_runtime").glob("*.pid")) == []


def test_pfb3x3_server_start_failure_counts_as_load_failure(settings, repo, tmp_path,
                                                            monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    code = run(settings, "pfb3x3:standard", Guard(settings), RecordingEngine(settings))
    assert code == EXIT_LOAD_FAILED
    entry = repo.latest_audit("model_load_failed", "pfb3x3:standard")
    assert "fake engine serve: failure 1" in entry["detail"]["error"]
    assert repo.get_job(jid).status == JobStatus.queued
    assert not runtime_pid_file(settings).exists()


def test_pfb3x3_idle_session_yields_and_stops_server(settings, repo, tmp_path):
    _, sid = submission(repo, tmp_path)
    s = settings.model_copy(update={"session_idle_sec": 60})
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    ready_for_behavior(repo, sid)
    engine = RecordingEngine(s)

    def yield_when():
        # free while the job runs; busy afterwards, while the session idles
        return repo.get_job(jid).status == JobStatus.done

    code = run(s, "pfb3x3:standard", Guard(s, yield_when=yield_when), engine)
    assert code == EXIT_YIELD and repo.get_job(jid).status == JobStatus.done
    assert not alive(engine.servers[0].pid)
    assert not runtime_pid_file(s).exists()


# --- track session ---

def test_track_session_runs_on_the_gpu(settings, repo, tmp_path, fake):
    _, sid = submission(repo, tmp_path)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    code = run(settings, "track", Guard(settings), RecordingEngine(settings))
    assert code == EXIT_IDLE
    assert repo.get_job(jid).status == JobStatus.done
    assert repo.latest_run(sid, Stage.track, "cam1").output["device"] == "cuda"
    assert "serve" not in fake.commands()


def test_cpu_track_session_never_polls_the_gpu_and_heartbeats(settings, repo, tmp_path,
                                                              monkeypatch):
    _, sid = submission(repo, tmp_path)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "0.5")
    beats = []
    real = Repo.heartbeat
    monkeypatch.setattr(Repo, "heartbeat",
                        lambda self, job_id: beats.append(job_id) or real(self, job_id))
    guard = Guard(settings, yield_when=lambda: True)  # would yield at once if it were asked
    code = run(settings, TRACK_CPU_SESSION, guard, RecordingEngine(settings))
    assert code == EXIT_IDLE and guard.yield_calls == 0
    assert repo.get_job(jid).status == JobStatus.done
    assert repo.latest_run(sid, Stage.track, "cam1").output["device"] == "cpu"
    assert jid in beats


def test_track_session_needs_gpu_marks_waiting(settings, repo, tmp_path, monkeypatch):
    _, sid = submission(repo, tmp_path)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    code = run(settings, "track", Guard(settings), RecordingEngine(settings))
    assert code == EXIT_NEEDS_GPU
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu and "GPU" in job.error


def test_pfb3x3_behavior_heartbeats_while_the_engine_runs(settings, repo, tmp_path,
                                                          monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "0.5")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP_CMD", "behavior")
    beats = []
    real = Repo.heartbeat
    monkeypatch.setattr(Repo, "heartbeat",
                        lambda self, job_id: beats.append(job_id) or real(self, job_id))
    # the heartbeat thread would wait an hour: the beats come from the engine's on_tick
    code = run(settings, "pfb3x3:standard", Guard(settings), RecordingEngine(settings),
               heartbeat_sec=3600)
    assert code == EXIT_IDLE and repo.get_job(jid).status == JobStatus.done
    assert jid in beats


# --- fix wave: leftovers, signals, a dead server, unknown keys ---

def _leftover_server(settings):
    runtime = settings.data_dir / "pfb3x3_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "vllm"],
                            start_new_session=True)
    (runtime / "serve.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc


def test_worker_start_stops_a_server_left_by_a_previous_worker(settings, repo, fake):
    leftover = _leftover_server(settings)   # e.g. scripts/stop.sh during a pfb3x3 session
    worker_loop(settings, guard=Guard(settings), spawn=lambda k: pytest.fail("no jobs"),
                sleep=lambda s: None, once=True)
    assert leftover.wait(timeout=10) is not None
    assert not runtime_pid_file(settings).exists()
    assert fake.commands() == ["stop"]   # through the engine's stop_server.sh only


def test_a_signal_during_behavior_stops_the_engine_and_the_server(settings, repo, tmp_path,
                                                                  fake, monkeypatch):
    """exit_on_signals turns SIGTERM/SIGHUP into SystemExit; the session's finally blocks
    then stop the engine command group and the vLLM server (simulated here by the
    SystemExit arriving during a tick)."""
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    pids = tmp_path / "behavior.pids"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids))
    engine = RecordingEngine(settings)

    def hup():
        if pids.exists():
            raise SystemExit(128 + signal.SIGHUP)
        return False

    with pytest.raises(SystemExit):
        run(settings, "pfb3x3:standard", Guard(settings, yield_when=hup), engine)
    for pid in pids.read_text().split():
        assert not alive(int(pid))
    assert not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()


def test_exit_on_signals_raises_system_exit():
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for sig in old:
            exit_on_signals()   # one session each: only its first signal ends it
            with pytest.raises(SystemExit) as e:
                os.kill(os.getpid(), sig)
                time.sleep(1)
            assert e.value.code == 128 + sig
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


@pytest.mark.parametrize("how", ["engine_says", "server_died"])
def test_a_dead_server_requeues_the_job_and_ends_the_session(settings, repo, tmp_path, fake,
                                                             monkeypatch, how):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    first = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "behavior")
    engine = RecordingEngine(settings)
    if how == "engine_says":
        monkeypatch.setenv("FAKE_ENGINE_ERROR", "server unavailable (connection refused); "
                                                "finished windows are saved, rerun to continue")
    else:
        monkeypatch.setattr(engine, "server_alive", lambda handle: False)

    code = run(settings, "pfb3x3:standard", Guard(settings), engine)

    assert code == EXIT_SERVER_DOWN
    assert repo.get_job(first).status == JobStatus.queued   # not failed_retryable
    assert fake.commands().count("behavior") == 1           # the session ended right away
    assert not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()


def test_an_engine_failure_names_its_log_in_the_job_error(settings, repo, tmp_path, fake,
                                                          monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "behavior")
    run(settings, "pfb3x3:standard", Guard(settings), RecordingEngine(settings))
    error = repo.get_job(jid).error
    assert error.splitlines()[0].startswith("EngineError: エンジン behavior が失敗しました")
    assert error.splitlines()[1].startswith(LOG_PREFIX) and error.splitlines()[1].endswith(
        "behavior.err")


def test_unknown_pfb3x3_key_waits_with_a_reason(settings, repo, tmp_path):
    _, sid = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:huge", camera="cam1")
    worker_loop(settings, guard=Guard(settings), spawn=lambda k: pytest.fail("no session"),
                sleep=lambda s: None, once=True)
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu
    assert "設定にないモデル（pfb3x3:huge）" in job.error and "pfb3x3_budget_gb" in job.error


def test_a_server_that_keeps_dying_fails_the_job_after_three_requeues(settings, repo, tmp_path,
                                                                     fake, monkeypatch):
    _, sid = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "behavior")
    monkeypatch.setenv("FAKE_ENGINE_ERROR", "server unavailable (connection refused)")

    def session():
        engine = RecordingEngine(settings)
        assert run(settings, "pfb3x3:standard", Guard(settings), engine) == EXIT_SERVER_DOWN
        assert not alive(engine.servers[0].pid)
        return repo.get_job(jid)

    assert [session().status for _ in range(2)] == [JobStatus.queued, JobStatus.queued]
    job = session()   # the third stop in a row
    assert job.status == JobStatus.failed_retryable
    assert job.error.splitlines()[0] == ("解析サーバーが繰り返し停止しました。"
                                         "詳細画面から再実行してください。")
    # the teacher re-runs it: the count starts again
    assert repo.retry_job(jid)
    assert session().status == JobStatus.queued


# --- detect (content check): GPU session with the server, or CPU-only at once ---

class DetectEngine(RecordingEngine):
    """probe / scene answered in Python (the shared fake engine has no content check); the
    server is the fake engine's real serve.sh / stop_server.sh."""

    def __init__(self, settings):
        super().__init__(settings)
        self.checks: list[tuple] = []

    def probe(self, sub_dir, video_id, video_path, *, start_s=0.0, end_s=None, samples=8,
              device="cpu", on_tick=None, timeout_sec=None):
        self.checks.append(("probe", video_id, device))
        assert on_tick() is False   # a probe is never aborted for the GPU
        return ProbeResult(duration_s=60.0, persons={"median": 4, "p90": 4, "max": 5},
                           vehicles={"median": 0, "p90": 0, "max": 0},
                           camera_motion={"median_shift_frac": 0.0, "moving": False}, frames=[])

    def scene(self, sub_dir, video_id, *, base_url, model, on_tick=None, timeout_sec=None):
        self.checks.append(("scene", video_id, base_url, model))
        return SceneResult(scene="group_work", sport=None, sport_text=None,
                           numbers_visible=False, model=model)


def detect_job(repo, tmp_path):
    _, sid = submission(repo, tmp_path)
    return sid, repo.enqueue(sid, Stage.detect, "detect")


def test_session_order_puts_detect_first_with_the_detect_model_budget(settings, repo, tmp_path):
    sid, _ = detect_job(repo, tmp_path)
    repo.enqueue(sid, Stage.transcribe, "whisper")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, spawned = Guard(settings, free_gb=90.0), []
    worker_loop(settings, guard=guard, spawn=lambda key: spawned.append(key) or 0,
                sleep=lambda s: None, once=True)
    assert spawned == ["detect"] and guard.can_start_calls == [34]   # pfb3x3 standard's budget
    high = settings.model_copy(update={"detect_model_key": "high"})
    guard = Guard(high, free_gb=90.0)
    worker_loop(high, guard=guard, spawn=lambda key: 0, sleep=lambda s: None, once=True)
    assert guard.can_start_calls[0] == 60


def test_gpu_pass_detect_falls_back_to_cpu_immediately(settings, repo, tmp_path, monkeypatch):
    sid, jid = detect_job(repo, tmp_path)
    marked, slept, spawned = [], [], []
    real = Repo.mark_waiting_gpu
    monkeypatch.setattr(Repo, "mark_waiting_gpu",
                        lambda self, key, reason: marked.append(key) or real(self, key, reason))

    def spawn(key):   # what the CPU session does: claim the detect job and finish it
        spawned.append(key)
        job = repo.claim_next_job({"detect"})
        repo.finish_job(job.id, JobStatus.done)
        return EXIT_IDLE

    guard = Guard(settings, free_gb=1.0)   # 34 GB + 8 GB margin never fits
    worker_loop(settings, guard=guard, spawn=spawn, sleep=slept.append, once=True)
    assert spawned == [DETECT_CPU_SESSION]
    assert "detect" not in marked and slept == []   # no waiting_gpu, no waiting time
    assert repo.get_job(jid).status == JobStatus.done


def test_gpu_pass_detect_without_a_configured_budget_runs_on_the_cpu(settings, repo, tmp_path):
    settings = settings.model_copy(update={"detect_model_key": "nope"})
    detect_job(repo, tmp_path)
    spawned = []
    worker_loop(settings, guard=Guard(settings), spawn=lambda key: spawned.append(key) or 0,
                sleep=lambda s: None, once=True)
    assert spawned == [DETECT_CPU_SESSION]


def test_detect_session_stops_server_on_exit(settings, repo, tmp_path, fake):
    sid, jid = detect_job(repo, tmp_path)
    other = repo.enqueue(sid, Stage.track, "track", camera="cam1")
    engine = DetectEngine(settings)

    code = run(settings, "detect", Guard(settings), engine)

    assert code == EXIT_IDLE
    assert repo.get_job(jid).status == JobStatus.done
    assert repo.get_job(other).status == JobStatus.queued   # only detect jobs are claimed
    serve = next(c for c in fake.calls() if c["argv"][0] == "serve")
    assert serve["argv"][1:] == ["qwen3vl-8b", "8200"]
    assert serve["env"]["PFB_GPU_UTIL"] == "0.347"   # 34 GB of 97.9 GB
    assert engine.start_kwargs == {"timeout_sec": 600}   # detect_timeout_sec
    assert engine.checks == [("probe", f"s{sid}_cam1", "cpu"),
                             ("scene", f"s{sid}_cam1", "http://localhost:8200/v1", "qwen3vl-8b")]
    out = repo.latest_run(sid, Stage.detect).output
    assert out["scene"]["scene"] == "group_work" and out["scene_skipped"] is None
    assert fake.commands()[-1] == "stop"
    assert len(engine.servers) == 1 and not alive(engine.servers[0].pid)
    assert not runtime_pid_file(settings).exists()


@pytest.mark.parametrize("exit_code", ["3", "1"])   # EngineNeedsGpu / EngineError at start
def test_detect_session_start_failure_runs_cpu_only(settings, repo, tmp_path, fake, monkeypatch,
                                                    exit_code):
    sid, jid = detect_job(repo, tmp_path)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", exit_code)
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    engine = DetectEngine(settings)
    code = run(settings, "detect", Guard(settings), engine)
    assert code == EXIT_IDLE
    assert repo.get_job(jid).status == JobStatus.done
    out = repo.latest_run(sid, Stage.detect).output
    assert out["scene"] is None and out["scene_skipped"] == "gpu_busy"
    assert [c[0] for c in engine.checks] == ["probe"]
    assert repo.latest_audit("model_load_failed", "detect") is None   # not a load failure
    assert not runtime_pid_file(settings).exists()


def test_detect_cpu_session_has_no_server(settings, repo, tmp_path, fake):
    sid, jid = detect_job(repo, tmp_path)
    guard = Guard(settings, yield_when=lambda: True)   # would yield at once if it were asked
    engine = DetectEngine(settings)
    code = run(settings, DETECT_CPU_SESSION, guard, engine)
    assert code == EXIT_IDLE and guard.yield_calls == 0
    assert "serve" not in fake.commands() and engine.servers == []
    assert repo.get_job(jid).status == JobStatus.done
    assert repo.latest_run(sid, Stage.detect).output["scene_skipped"] == "gpu_busy"


def test_detect_session_gives_the_gpu_back_between_jobs(settings, repo, tmp_path, fake):
    sid1, j1 = detect_job(repo, tmp_path)
    sid2, j2 = detect_job(repo, tmp_path)
    engine = DetectEngine(settings)

    def yield_when():   # the GPU gets busy once the first check is done
        return repo.get_job(j1).status == JobStatus.done

    code = run(settings, "detect", Guard(settings, yield_when=yield_when), engine)
    assert code == EXIT_IDLE
    assert repo.get_job(j1).status == JobStatus.done and repo.get_job(j2).status == JobStatus.done
    assert repo.latest_run(sid1, Stage.detect).output["scene_skipped"] is None
    # the server is stopped as soon as the GPU is needed; the next check runs CPU-only
    assert repo.latest_run(sid2, Stage.detect).output["scene_skipped"] == "gpu_busy"
    assert [c[0] for c in engine.checks] == ["probe", "scene", "probe"]
    assert not alive(engine.servers[0].pid) and not runtime_pid_file(settings).exists()


def test_detect_probe_stops_the_server_when_the_gpu_is_wanted(settings, repo, tmp_path, fake):
    sid, jid = detect_job(repo, tmp_path)
    engine = DetectEngine(settings)

    def yield_when():   # the GPU is wanted once the check's probe is running
        return bool(engine.checks)

    code = run(settings, "detect", Guard(settings, yield_when=yield_when), engine)
    assert code == EXIT_IDLE and repo.get_job(jid).status == JobStatus.done
    out = repo.latest_run(sid, Stage.detect).output
    assert out["scene"] is None and out["scene_skipped"] == "gpu_busy"
    assert [c[0] for c in engine.checks] == ["probe"]   # the probe went on, no scene
    assert not alive(engine.servers[0].pid) and not runtime_pid_file(settings).exists()
