"""Release-2 fix wave: a linecall job runs OUTSIDE the worker's one-session loop, in a
detached, supervised child (worker.linecall_runner), so the other works go on meanwhile.
Fake children and the fake engine only: no GPU, no image is opened."""
from pathlib import Path

import pytest

from fakes.engine_env import install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage
from pfb_app.engine import Pfb3x3Engine
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
from pfb_app.pipeline.runner import LINECALL_KEY
from pfb_app.services import auto_retry
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import linecall_runner
from pfb_app.worker import main as worker_main
from pfb_app.worker.session import EXIT_IDLE, EXIT_NEEDS_GPU
from pfb_app.worker.shared_server import SharedServer
from test_shared_server import Clock, spawner
from test_shared_server import Engine as SharedEngine

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
    yield Repo(conn, settings)
    conn.close()


class Guard(GpuGuard):
    def __init__(self, settings, free_gb=90.0):
        self.free_gb = free_gb
        self.can_start_calls: list[float] = []
        super().__init__(settings, query=lambda: GpuSnapshot(TOTAL_GB, TOTAL_GB - self.free_gb))

    def can_start(self, budget_gb):
        self.can_start_calls.append(budget_gb)
        return super().can_start(budget_gb)

    def should_yield(self):
        return False


class Child:
    """A stand-in for the spawn child: alive until `end(code)`."""

    def __init__(self):
        self.exitcode = None
        self.pid = 4242
        self.terminated = False

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):
        pass

    def end(self, code):
        self.exitcode = code

    def terminate(self):
        self.terminated = True
        self.exitcode = -15

    def kill(self):
        self.exitcode = -9


class Starter:
    def __init__(self):
        self.children: list[Child] = []

    def __call__(self):
        child = Child()
        self.children.append(child)
        return child


def make_runner(settings, clock=None, cleaned=None):
    starter = Starter()
    runner = linecall_runner.LinecallRunner(
        settings, None, start=starter, clock=clock or Clock(),
        stop_stale=(lambda: cleaned.append("linecall")) if cleaned is not None else None)
    return runner, starter


def work(repo, tmp_path, name, domain="sports"):
    aid = repo.create_assignment(name, Kind.video, "3x3の試合", "", "standard", False,
                                 domain=domain, sport="basketball" if domain == "sports" else None)
    p = tmp_path / f"{name}.mp4"
    p.write_bytes(b"x")
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": str(p)}, "start_sec": 0.0,
                                            "end_sec": None, "engine": "pfb3x3"})
    return sid


def test_other_works_run_while_a_linecall_runs(settings, repo, tmp_path):
    lc = work(repo, tmp_path, "試合A")
    other = work(repo, tmp_path, "試合B")
    lc_job = repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.enqueue(other, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings), []
    runner, starter = make_runner(settings)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                          linecall=runner)
    assert len(starter.children) == 1 and runner.busy          # started, not waited for
    assert calls == [("track", None)]                           # in the same pass
    # the child claims its job (as the linecall session does) and keeps running
    assert repo.claim_next_job({LINECALL_KEY}).id == lc_job
    for stage in (Stage.behavior, Stage.pfb):
        repo.enqueue(other, stage, "pfb3x3:standard", camera="cam1" if stage == Stage.behavior
                     else None)
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              linecall=runner)
    assert calls[1:] == [("pfb3x3:standard", None), ("pfb3x3:standard", None)]
    assert len(starter.children) == 1                          # one linecall at a time
    assert repo.get_job(lc_job).status == JobStatus.running   # untouched by the others


def test_only_one_linecall_at_a_time(settings, repo, tmp_path):
    a, b = work(repo, tmp_path, "試合A"), work(repo, tmp_path, "試合B")
    repo.enqueue(a, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.enqueue(b, Stage.linecall, LINECALL_KEY, camera="cam1")
    guard = Guard(settings)
    runner, starter = make_runner(settings)
    for _ in range(3):
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                              linecall=runner)
    assert len(starter.children) == 1
    assert linecall_runner.MAX_RUNNING == 1


def test_a_running_linecall_counts_as_a_gpu_consumer(settings, repo, tmp_path):
    lc, other = work(repo, tmp_path, "試合A"), work(repo, tmp_path, "試合B")
    repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    guard = Guard(settings)
    runner, _ = make_runner(settings)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    repo.enqueue(other, Stage.track, "track", camera="cam1")
    guard.can_start_calls.clear()
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    assert guard.can_start_calls == [settings.track_budget_gb + linecall_runner.RUN_GB]


def test_the_linecall_start_needs_the_engines_free_memory(settings, repo, tmp_path):
    lc = work(repo, tmp_path, "試合A")
    jid = repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    guard = Guard(settings, free_gb=38.5)
    runner, starter = make_runner(settings)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    assert starter.children == [] and repo.get_job(jid).status == JobStatus.waiting_gpu


def test_the_shared_server_counts_a_running_linecall(settings, repo, tmp_path):
    guard = Guard(settings)
    shared = SharedServer(settings, guard, SharedEngine(settings), clock=Clock())
    runner, starter = make_runner(settings)
    lc = work(repo, tmp_path, "試合A")
    repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    try:
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                              shared=shared, linecall=runner)
        assert runner.busy
        guard.can_start_calls.clear()
        shared.request()
        assert guard.can_start_calls == [shared.budget + linecall_runner.RUN_GB]
    finally:
        shared.close()


def test_a_crashed_linecall_recovers_only_its_job_and_is_retried(settings, repo, tmp_path):
    lc, other = work(repo, tmp_path, "試合A"), work(repo, tmp_path, "試合B")
    lc_job = repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    track_job = repo.enqueue(other, Stage.track, "track", camera="cam1")
    cleaned = []
    runner, starter = make_runner(settings, cleaned=cleaned)
    worker_main._gpu_pass(repo, settings, Guard(settings), spawner(repo, [], finish=False),
                          lambda s: None, {}, linecall=runner)
    assert repo.claim_next_job({LINECALL_KEY}).id == lc_job
    assert repo.claim_next_job({"track"}).id == track_job      # another session's, running
    starter.children[0].end(1)                                 # the child crashed
    worker_main._linecall_poll(repo, settings, runner)
    assert cleaned == ["linecall"] and not runner.busy
    assert repo.get_job(track_job).status == JobStatus.running   # not ours: left alone
    assert repo.get_job(lc_job).status == JobStatus.queued       # recovered, retried once
    assert auto_retry.retried_before(repo, lc_job)


def test_a_needs_gpu_exit_waits_before_the_next_start(settings, repo, tmp_path):
    lc = work(repo, tmp_path, "試合A")
    repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    clock = Clock()
    runner, starter = make_runner(settings, clock=clock)
    guard = Guard(settings)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    starter.children[0].end(EXIT_NEEDS_GPU)
    worker_main._linecall_poll(repo, settings, runner)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    assert len(starter.children) == 1                          # not respawned at once
    clock.t += settings.gpu_poll_sec + 1
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    assert len(starter.children) == 2


def test_another_sessions_crash_leaves_the_linecall_job_running(settings, repo, tmp_path):
    lc, other = work(repo, tmp_path, "試合A"), work(repo, tmp_path, "試合B")
    lc_job = repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    runner, _ = make_runner(settings)
    guard = Guard(settings)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, []), lambda s: None, {},
                          linecall=runner)
    assert repo.claim_next_job({LINECALL_KEY}).id == lc_job
    repo.enqueue(other, Stage.track, "track", camera="cam1")

    def crash(key, server=None):
        repo.claim_next_job({key})
        return 1
    worker_main._gpu_pass(repo, settings, guard, crash, lambda s: None, {},
                          engine_cleanup=lambda key: None, linecall=runner)
    assert repo.get_job(lc_job).status == JobStatus.running


def test_the_worker_stops_the_linecall_child_when_it_ends(settings):
    runner, starter = make_runner(settings, cleaned=[])
    runner.start()
    runner.stop()
    assert starter.children[0].terminated and not runner.busy


def test_the_linecall_engine_call_has_its_own_record(settings):
    engine = Pfb3x3Engine(settings)
    assert engine.linecall_call_record != engine.call_record
    assert engine.linecall_call_record.parent == engine.runtime_dir
    # a session starting beside the linecall never stops the linecall's engine command
    engine.runtime_dir.mkdir(parents=True, exist_ok=True)
    engine.linecall_call_record.write_text("999999 1\n", encoding="utf-8")
    engine.stop_stale_call()
    assert engine.linecall_call_record.exists()
    engine.stop_stale_linecall()                                # not ours: removed, no signal
    assert not engine.linecall_call_record.exists()


def test_a_linecall_session_does_not_end_for_a_content_check(settings, repo, tmp_path):
    from pfb_app.worker import session
    sid = work(repo, tmp_path, "試合A")
    repo.enqueue(sid, Stage.detect, "detect")
    assert not session._detect_pending(repo, LINECALL_KEY)
    assert session._detect_pending(repo, "track")


def test_without_a_runner_linecall_stays_a_session(settings, repo, tmp_path):
    """--once (no runner): the old one-session path, as before."""
    lc = work(repo, tmp_path, "試合A")
    repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    calls = []
    worker_main._gpu_pass(repo, settings, Guard(settings), spawner(repo, calls), lambda s: None,
                          {})
    assert calls == [("linecall", None)]


def test_idle_exit_needs_no_cooldown(settings, repo, tmp_path):
    clock = Clock()
    runner, starter = make_runner(settings, clock=clock)
    runner.start()
    starter.children[0].end(EXIT_IDLE)
    assert runner.poll() == EXIT_IDLE and runner.ready()


# --- LOW: quiet alerts, retry kept, pre-warm not held up --------------------------------------

def test_a_stopped_linecall_is_retried_once_but_never_alerted(settings, repo, tmp_path):
    from pfb_app.ops import alerts
    assert Stage.linecall in auto_retry.QUIET_STAGES
    sid = work(repo, tmp_path, "試合A")
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.claim_next_job({LINECALL_KEY})
    repo.finish_job(jid, JobStatus.failed_retryable, "heartbeat timeout")
    assert auto_retry.after_failure(repo, jid, settings)         # retried once
    repo.claim_next_job({LINECALL_KEY})
    repo.finish_job(jid, JobStatus.failed_retryable, "heartbeat timeout")
    assert not auto_retry.after_failure(repo, jid, settings)
    assert not [a for a in alerts.recent(settings) if a["key"] == f"job-stopped:{jid}"]


def test_a_waiting_linecall_does_not_hold_the_prewarm_up(settings, repo, tmp_path):
    guard = Guard(settings)
    shared = SharedServer(settings, guard, SharedEngine(settings), clock=Clock())
    lc = work(repo, tmp_path, "試合A")
    repo.enqueue(lc, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.mark_waiting_gpu(LINECALL_KEY, "空き不足")
    other = work(repo, tmp_path, "試合B", domain="school")
    repo.enqueue(other, Stage.track, "track", camera="cam1")
    assert shared.prewarm_wanted(repo)
    repo.mark_waiting_gpu("track", "空き不足")
    assert not shared.prewarm_wanted(repo)                     # a step of a work still does


def test_the_session_crash_sweep_never_breaks_the_pass(settings, repo, tmp_path, monkeypatch):
    sid = work(repo, tmp_path, "試合A")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")

    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(auto_retry, "sweep", boom)
    worker_main._gpu_pass(repo, settings, Guard(settings), lambda key: 1, lambda s: None, {},
                          engine_cleanup=lambda key: None)   # logged, not raised
