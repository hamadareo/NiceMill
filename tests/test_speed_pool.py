"""Speed lane (2026-10-24): a pfb3x3 session's jobs side by side (worker.session._pool_job_loop),
the claim rules behind it (Repo.claim_next_job skip_stages / after_upstream), one engine call
record per concurrent command, the shared server's budget when the GPU has room, the engine's
GPU reserve as a setting and the prepare frames read by several readers. No GPU, no HTTP."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from fakes.engine_env import install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage
from pfb_app.engine import Pfb3x3Engine
from pfb_app.engine.pfb3x3 import engine_reserve_mib
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
from pfb_app.preprocess.frames import extract_frames
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import main as worker_main
from pfb_app.worker import session as session_mod
from pfb_app.worker.session import EXIT_IDLE, EXIT_YIELD, run_session
from pfb_app.worker.shared_server import SharedServer

KEY = "pfb3x3:standard"
TOTAL_GB = 97.9


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fake):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=fake.engine_dir, engine_python=fake.engine_python,
                    gpu_poll_sec=7, session_idle_sec=0)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


class Guard(GpuGuard):
    def __init__(self, settings, free=90.0):
        self.free = free
        super().__init__(settings, query=lambda: GpuSnapshot(TOTAL_GB, TOTAL_GB - self.free))

    def should_yield(self):
        return False


def submission(repo, tmp_path, subjects=("Pa",)):
    aid = repo.create_assignment("活動", Kind.video, "試合", "", "standard", True)
    p = tmp_path / f"cam1_{aid}.mp4"
    p.write_bytes(b"x")
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": str(p)}, "start_sec": 0.0,
                                             "end_sec": None, "engine": "pfb3x3"})
    ids = [repo.add_subject(sid, pid, f"{pid}さん", "外見", {}) for pid in subjects]
    return sid, ids


def fake_server(settings):
    from pfb_app.engine import ServerHandle
    runtime = settings.data_dir / "pfb3x3_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "vllm"],
                            start_new_session=True)
    (runtime / "serve.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    return ServerHandle(pid=proc.pid, port=settings.pfb3x3_port, model="qwen3vl-8b",
                        base_url=f"http://localhost:{settings.pfb3x3_port}/v1")


class Engine(Pfb3x3Engine):
    def __init__(self, settings):
        super().__init__(settings, tick_sec=0.05, term_grace_sec=2)

    def chat_backend(self, handle):
        from fakes.engine_env import fake_translator
        return fake_translator()


def session(settings, handle, sleep=lambda s: None):
    try:
        return run_session(None, KEY, settings=settings, guard=Guard(settings),
                           engine_factory=Engine, sleep=sleep, heartbeat_sec=0.05, server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)


class Recorder:
    """execute_job stand-in: records start/end per job and how many ran at once."""

    def __init__(self, act=None):
        self.lock = threading.Lock()
        self.running: set[int] = set()
        self.events: list[tuple[str, int, str]] = []
        self.most = 0
        self.act = act or {}

    def __call__(self, job, ctx):
        with self.lock:
            self.running.add(job.id)
            self.most = max(self.most, len(self.running))
            self.events.append(("start", job.id, job.stage.value))
        try:
            fn = self.act.get(job.stage)
            if fn is not None:
                fn(job, ctx)
            ctx.repo.finish_job(job.id, JobStatus.done)
        finally:
            with self.lock:
                self.running.discard(job.id)
                self.events.append(("end", job.id, job.stage.value))

    def overlapped(self, a: int, b: int) -> bool:
        """Did job b start while job a was running?"""
        start_a = self.events.index(("start", a, self._stage(a)))
        end_a = self.events.index(("end", a, self._stage(a)))
        start_b = self.events.index(("start", b, self._stage(b)))
        return start_a < start_b < end_a

    def _stage(self, jid):
        return next(s for _k, j, s in self.events if j == jid)


# --- side by side ---

def test_names_runs_beside_behavior_and_contribution_starts_as_behavior_ends(
        settings, repo, tmp_path, monkeypatch):
    sid, _ = submission(repo, tmp_path)
    behavior = repo.enqueue(sid, Stage.behavior, KEY, camera="cam1")
    names = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    both = threading.Barrier(2, timeout=10)
    contribution_started = threading.Event()
    later: list[int] = []

    def on_behavior(job, ctx):
        both.wait()   # only passes while names runs too
        later.append(ctx.repo.enqueue(sid, Stage.contribution, KEY, camera="cam1"))

    def on_names(job, ctx):
        both.wait()
        assert contribution_started.wait(10), "contribution waited for names"

    def on_contribution(job, ctx):
        contribution_started.set()

    rec = Recorder({Stage.behavior: on_behavior, Stage.names: on_names,
                    Stage.contribution: on_contribution})
    monkeypatch.setattr(session_mod, "execute_job", rec)
    assert session(settings, fake_server(settings)) == EXIT_IDLE
    assert all(repo.get_job(j).status == JobStatus.done for j in (behavior, names, *later))
    assert rec.overlapped(names, later[0])


def test_engine_commands_are_capped_and_text_jobs_go_past_them(settings, repo, tmp_path,
                                                               monkeypatch):
    settings = settings.model_copy(update={"served_engine_jobs": 1})
    a, _ = submission(repo, tmp_path)
    b, (pb,) = submission(repo, tmp_path, subjects=("Pb",))
    first = repo.enqueue(a, Stage.behavior, KEY, camera="cam1")
    second = repo.enqueue(a, Stage.names, KEY, camera="cam1")
    other = repo.enqueue(b, Stage.pfb, KEY, subject_id=pb)   # another group's text-only job
    pfb_ran = threading.Event()

    def on_behavior(job, ctx):
        assert pfb_ran.wait(10), "the pfb job behind the capped names job did not run"

    rec = Recorder({Stage.behavior: on_behavior, Stage.pfb: lambda j, c: pfb_ran.set()})
    monkeypatch.setattr(session_mod, "execute_job", rec)
    assert session(settings, fake_server(settings)) == EXIT_IDLE
    assert all(repo.get_job(j).status == JobStatus.done for j in (first, second, other))
    assert rec.overlapped(first, other) and not rec.overlapped(first, second)


def test_a_job_never_runs_beside_the_job_of_its_group_it_builds_on(settings, repo, tmp_path,
                                                                   monkeypatch):
    a, (pa,) = submission(repo, tmp_path)
    b, (pb,) = submission(repo, tmp_path, subjects=("Pb",))
    contribution = repo.enqueue(a, Stage.contribution, KEY, camera="cam1")
    own_pfb = repo.enqueue(a, Stage.pfb, KEY, subject_id=pa)
    other_pfb = repo.enqueue(b, Stage.pfb, KEY, subject_id=pb)
    other_ran = threading.Event()

    def on_contribution(job, ctx):
        assert other_ran.wait(10)
        time.sleep(0.2)   # room for a wrong claim of its own pfb

    rec = Recorder({Stage.contribution: on_contribution,
                    Stage.pfb: lambda j, c: other_ran.set() if j.id == other_pfb else None})
    monkeypatch.setattr(session_mod, "execute_job", rec)
    assert session(settings, fake_server(settings)) == EXIT_IDLE
    assert rec.overlapped(contribution, other_pfb)
    assert not rec.overlapped(contribution, own_pfb)
    assert repo.get_job(own_pfb).status == JobStatus.done


def test_a_pending_detect_job_stops_new_claims_until_the_running_end(settings, repo, tmp_path,
                                                                    monkeypatch):
    a, _ = submission(repo, tmp_path)
    first = repo.enqueue(a, Stage.behavior, KEY, camera="cam1")
    queued: list[int] = []

    def on_behavior(job, ctx):   # a new video arrives while behavior runs
        ctx.repo.enqueue(a, Stage.detect, "detect")
        queued.append(ctx.repo.enqueue(a, Stage.contribution, KEY, camera="cam1"))

    rec = Recorder({Stage.behavior: on_behavior})
    monkeypatch.setattr(session_mod, "execute_job", rec)
    assert session(settings, fake_server(settings)) == EXIT_YIELD
    assert repo.get_job(first).status == JobStatus.done
    assert repo.get_job(queued[0]).status == JobStatus.queued   # left for after the check


def test_one_job_at_a_time_when_set_so(settings, repo, tmp_path, monkeypatch):
    settings = settings.model_copy(update={"served_parallel_jobs": 1})
    a, _ = submission(repo, tmp_path)
    jobs = [repo.enqueue(a, Stage.behavior, KEY, camera="cam1"),
            repo.enqueue(a, Stage.names, KEY, camera="cam1")]
    rec = Recorder()
    monkeypatch.setattr(session_mod, "execute_job", rec)
    assert session(settings, fake_server(settings)) == EXIT_IDLE
    assert rec.most == 1 and all(repo.get_job(j).status == JobStatus.done for j in jobs)


def test_a_crash_in_one_job_ends_the_session_after_the_others(settings, repo, tmp_path,
                                                              monkeypatch):
    a, _ = submission(repo, tmp_path)
    crash = repo.enqueue(a, Stage.behavior, KEY, camera="cam1")
    fine = repo.enqueue(a, Stage.names, KEY, camera="cam1")
    both = threading.Barrier(2, timeout=10)

    def on_behavior(job, ctx):
        both.wait()
        raise RuntimeError("boom")

    def on_names(job, ctx):
        both.wait()
        time.sleep(0.2)

    rec = Recorder({Stage.behavior: on_behavior, Stage.names: on_names})
    monkeypatch.setattr(session_mod, "execute_job", rec)
    with pytest.raises(RuntimeError, match="boom"):
        session(settings, fake_server(settings))
    assert repo.get_job(fine).status == JobStatus.done
    assert repo.get_job(crash).status == JobStatus.running   # the worker recovers it


# --- the claim ---

def test_claim_skips_stages_and_waits_for_upstream_jobs(repo, tmp_path):
    a, (pa,) = submission(repo, tmp_path)
    behavior = repo.enqueue(a, Stage.behavior, KEY, camera="cam1")
    contribution = repo.enqueue(a, Stage.contribution, KEY, camera="cam1")
    pfb = repo.enqueue(a, Stage.pfb, KEY, subject_id=pa)
    names = repo.enqueue(a, Stage.names, KEY, camera="cam1")
    assert repo.claim_next_job({KEY}, after_upstream=True).id == behavior
    # behavior runs: its contribution and pfb wait, names does not
    assert repo.claim_next_job({KEY}, after_upstream=True).id == names
    assert repo.claim_next_job({KEY}, after_upstream=True) is None
    repo.finish_job(behavior, JobStatus.done)
    assert repo.claim_next_job({KEY}, after_upstream=True).id == contribution
    assert repo.claim_next_job({KEY}, after_upstream=True) is None
    repo.finish_job(contribution, JobStatus.done)
    assert repo.claim_next_job({KEY}, after_upstream=True).id == pfb
    # without the flag the oldest job is claimed, as before
    b, _ = submission(repo, tmp_path)
    repo.enqueue(b, Stage.behavior, KEY, camera="cam1")
    later = repo.enqueue(b, Stage.contribution, KEY, camera="cam1")
    assert repo.claim_next_job({KEY}, skip_stages={Stage.behavior, Stage.names}).id == later


# --- engine call records ---

def test_concurrent_engine_calls_each_have_a_record_and_are_all_stopped(tmp_path, monkeypatch):
    from test_engine_client import _alive
    fake = install_fake_engine(tmp_path, monkeypatch)
    settings = Settings(data_dir=tmp_path / "data", engine_dir=fake.engine_dir,
                        engine_python=fake.engine_python)
    engine = Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=2)
    sub_dirs = []
    for n in (1, 2):
        d = tmp_path / "data" / "assignments" / "1" / "submissions" / str(n)
        d.mkdir(parents=True)
        sub_dirs.append(d)
    script = tmp_path / "child.py"
    script.write_text(
        "import sys, threading\nfrom pathlib import Path\n"
        "from pfb_app.config import Settings\nfrom pfb_app.engine import Pfb3x3Engine\n"
        "s = Settings(data_dir=Path(sys.argv[1]), engine_dir=Path(sys.argv[2]),"
        " engine_python=Path(sys.argv[3]))\n"
        "e = Pfb3x3Engine(s, tick_sec=0.05)\n"
        "ts = [threading.Thread(target=e.track, args=(Path(d), f's{i}_cam1', Path('/v.mp4'),"
        " 0, None, None, None)) for i, d in enumerate(sys.argv[4:], 1)]\n"
        "[t.start() for t in ts]\n[t.join() for t in ts]\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    child = subprocess.Popen([sys.executable, str(script), str(settings.data_dir),
                              str(settings.engine_dir), str(settings.engine_python),
                              *map(str, sub_dirs)])
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (
                engine.call_record.exists() and len(engine.extra_call_records()) == 1):
            time.sleep(0.05)
        records = [engine.call_record, *engine.extra_call_records()]
        assert all(r.exists() for r in records) and len(records) == 2
        leaders = [int(r.read_text().split()[0]) for r in records]
        assert len(set(leaders)) == 2 and all(_alive(p) for p in leaders)
        child.kill()   # the session dies hard
        child.wait(timeout=10)
        assert engine.stop_stale_call() is True
        assert not any(_alive(p) for p in leaders)
        assert not engine.call_record.exists() and engine.extra_call_records() == []
    finally:
        if child.poll() is None:
            child.kill()


# --- the shared server's budget ---

def test_the_shared_server_takes_the_free_gpu_up_to_its_max(settings):
    roomy = settings.model_copy(update={"pfb3x3_max_budget_gb": {"standard": 80},
                                        "track_budget_gb": 4, "start_margin_gb": 8})
    shared = SharedServer(roomy, Guard(roomy, free=90.0), Engine(roomy))
    assert shared.start_budget() == 78          # 90 free - 4 track - 8 margin, under 80
    shared.guard.free = 95.0
    assert shared.start_budget() == 80          # capped
    shared.guard.free = 40.0
    assert shared.start_budget() == 34          # never below the budget
    shared.extra_gb = lambda: 6.0               # a linecall running beside it
    shared.guard.free = 90.0
    assert shared.start_budget() == 72
    plain = SharedServer(settings, Guard(settings, free=90.0), Engine(settings))
    assert plain.start_budget() == 34           # no max: the budget, as before


def test_the_engine_reserve_is_a_setting(settings, tmp_path):
    assert engine_reserve_mib(settings) == 32768
    small = settings.model_copy(update={"engine_gpu_reserve_gb": 4, "linecall_budget_gb": 1})
    assert engine_reserve_mib(small) == 4096
    assert engine_reserve_mib(settings.model_copy(update={"engine_gpu_reserve_gb": 1})) == 4096
    env = Pfb3x3Engine(small)._env(None)
    assert env["PFB_GPU_RESERVE_MIB"] == "4096"
    # linecall then waits for its own need: 4 + 6 + 1 GB, not the default 39 GB
    assert worker_main._budget_gb(small, "linecall") + small.start_margin_gb == pytest.approx(11)
    assert worker_main._budget_gb(settings, "linecall") + settings.start_margin_gb == \
        pytest.approx(39)


# --- prepare frames ---

@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    rng = np.random.default_rng(0)
    for i in range(600):   # 60 s at 10 fps, every frame different
        img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        cv2.putText(img, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(img)
    writer.release()
    return path


def test_several_readers_save_the_same_frames_as_one(video, tmp_path):
    one = extract_frames(video, tmp_path / "one", 0, 60, 1, (160, 120), workers=1)
    many = extract_frames(video, tmp_path / "many", 0, 60, 1, (160, 120), workers=4)
    assert [f.video_sec for f in one] == [f.video_sec for f in many] and len(one) == 60
    assert [f.path.name for f in one] == [f.path.name for f in many]
    assert all(a.path.read_bytes() == b.path.read_bytes() for a, b in zip(one, many))


# --- warm before the content check ---

def test_the_shared_server_warms_up_while_a_video_uploads(settings, repo, monkeypatch):
    from pfb_app.services import chunked_upload
    shared = SharedServer(settings, Guard(settings), Engine(settings))
    assert not shared.prewarm_wanted(repo)
    token = chunked_upload.issue_token(settings)
    up = chunked_upload.create(settings, token, "game.MOV", 1000)
    chunked_upload.append(settings, token, up.id, 0, b"x" * 10)
    assert chunked_upload.video_arriving(settings)
    assert shared.prewarm_wanted(repo)
    later = chunked_upload.time.time() + chunked_upload.ARRIVING_SEC + 1
    assert not chunked_upload.video_arriving(settings, now=later)   # stalled: no warm-up
    other = settings.model_copy(update={"detect_model_key": "high"})
    assert not SharedServer(other, Guard(other), Engine(other)).prewarm_wanted(repo)
