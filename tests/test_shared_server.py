"""One shared, warm vLLM server for the standard model (worker.shared_server), the routing of
the tracking engine's contribution / pfb to it, concurrent requests and parallel pfb jobs.
FakeEngine (fake serve.sh / stop_server.sh) only: no GPU, no HTTP."""
import dataclasses
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from fakes.engine_env import alive, fake_translator, install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.engine import Pfb3x3Engine, ServerHandle
from pfb_app.inference.backend import ChatRequest, RawResponse
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
from pfb_app.inference.structured import YieldRequested, call_structured_batch
from pfb_app.domain.models import ContributionOutput
from pfb_app.pipeline import runner
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import main as worker_main
from pfb_app.worker import session as session_mod
from pfb_app.worker.main import worker_loop
from pfb_app.worker.session import (EXIT_IDLE, EXIT_SERVER_DOWN, EXIT_YIELD, IDLE_POLL_SEC,
                                    run_session)
from pfb_app.worker.shared_server import SharedServer

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
    def __init__(self, settings, free_gb=90.0):
        self.free_gb = free_gb
        self.wanted = False
        self.can_start_calls: list[float] = []
        super().__init__(settings, query=lambda: GpuSnapshot(TOTAL_GB, TOTAL_GB - self.free_gb))

    def can_start(self, budget_gb):
        self.can_start_calls.append(budget_gb)
        return super().can_start(budget_gb)

    def should_yield(self):
        return self.wanted


class Engine(Pfb3x3Engine):
    bound = ["127.0.0.1"]   # what `ss` reports for the server's port

    def __init__(self, settings):
        super().__init__(settings, tick_sec=0.05, term_grace_sec=2)
        self.stale_stops = 0

    def listen_addresses(self, port):
        return self.bound

    def stop_stale_server(self):
        self.stale_stops += 1
        super().stop_stale_server()

    def translator(self, handle):
        return fake_translator()

    def chat_backend(self, handle):
        self.chat_for = handle
        return fake_translator()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def submission(repo, tmp_path, model_key="standard", engine="pfb3x3", subjects=("Pa",)):
    aid = repo.create_assignment("課題", Kind.video, "紙タワー", "ルーブリック", model_key, True)
    p = tmp_path / f"cam1_{model_key}_{engine}.mp4"
    p.write_bytes(b"x")
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": str(p)}, "start_sec": 0.0,
                                             "end_sec": None, "engine": engine})
    ids = [repo.add_subject(sid, pid, f"{pid}さん", "外見", {}) for pid in subjects]
    return sid, ids


def save(repo, sid, stage, output, camera=None, source="model", upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def ready_for_behavior(repo, sid, camera="cam1"):
    prepare = save(repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    track = save(repo, sid, Stage.track, {"people": 1, "labels": ["P1"],
                                          "video_id": f"s{sid}_{camera}", "sheet": "x",
                                          "track_stats": {}, "default_names": {},
                                          "default_source": None, "device": "cuda"},
                 camera=camera, upstream=[prepare.id])
    save(repo, sid, Stage.identity, {
        "participants": [{"participant_id": "Pa", "label": "P1", "display_name": "Paさん",
                          "note": ""}], "labels": ["P1"], "track_run_id": track.id},
        camera=camera, source="teacher", upstream=[track.id])
    repo.set_identity_confirmed(sid, True)


def fake_vllm(settings) -> ServerHandle:
    """A running 'vllm' of ours, recorded in serve.pid as serve.sh would."""
    runtime = settings.data_dir / "pfb3x3_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "vllm"],
                            start_new_session=True)
    (runtime / "serve.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    return ServerHandle(pid=proc.pid, port=settings.pfb3x3_port, model="qwen3vl-8b",
                        base_url=f"http://localhost:{settings.pfb3x3_port}/v1")


def shared_server(settings, guard, clock=None, **kw) -> SharedServer:
    return SharedServer(settings, guard, Engine(settings), clock=clock or Clock(), **kw)


def ready(shared: SharedServer) -> ServerHandle:
    shared.wait_started(30)
    shared.poll()
    assert shared.handle is not None
    return shared.handle


# --- routing ---

def test_tracked_contribution_and_pfb_run_under_the_pfb3x3_key(repo, tmp_path):
    sid, (subj,) = submission(repo, tmp_path)
    assert runner.model_key_for(repo, sid, Stage.contribution, "cam1") == "pfb3x3:standard"
    assert runner.model_key_for(repo, sid, Stage.pfb) == "pfb3x3:standard"
    high, _ = submission(repo, tmp_path, model_key="high")
    assert runner.model_key_for(repo, high, Stage.pfb) == "pfb3x3:high"
    paper, _ = submission(repo, tmp_path, engine="paper")
    assert runner.model_key_for(repo, paper, Stage.contribution, "cam1") == "standard"
    assert runner.model_key_for(repo, paper, Stage.pfb) == "standard"


# --- structured retries in rounds ---

class Recording:
    model_id = "rec"

    def __init__(self, answers):
        self.answers = answers   # {user_text prefix: [answer per attempt]}
        self.batches: list[int] = []
        self.seen: dict[str, int] = {}

    def generate(self, reqs):
        self.batches.append(len(reqs))
        out = []
        for r in reqs:
            key = r.user_text.split("\n")[0]
            n = self.seen.get(key, 0)
            self.seen[key] = n + 1
            out.append(RawResponse(text=self.answers[key][min(n, len(self.answers[key]) - 1)],
                                   prompt_tokens=None, completion_tokens=None, duration_sec=0.0))
        return out

    def close(self):
        pass


def test_batch_retries_go_out_in_rounds(repo):
    good = json.dumps({"judgments": []})
    backend = Recording({"a": [good], "b": ["bad", good], "c": ["bad", "bad", good]})
    reqs = [ChatRequest("sys", t, [], ContributionOutput.model_json_schema(), 100)
            for t in ("a", "b", "c")]
    results = call_structured_batch(backend, repo, reqs, ContributionOutput, max_retries=2,
                                    yield_check=lambda: False)
    assert backend.batches == [3, 2, 1]   # one batch per round, not one call per retry
    assert all(r.parsed is not None for r in results)
    assert [len(r.raw_responses) for r in results] == [1, 2, 3]
    assert len(results[2].errors) == 2


# --- sessions on the shared server ---

def test_a_session_on_the_shared_server_neither_starts_nor_stops_it(settings, repo, tmp_path,
                                                                     fake):
    settings = settings.model_copy(update={"session_idle_sec": 300})   # not waited for here
    repo.settings = settings
    sid, _ = submission(repo, tmp_path)
    ready_for_behavior(repo, sid)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    handle = fake_vllm(settings)
    engine = Engine(settings)

    def no_sleep(s):
        raise AssertionError("a shared-server session does not idle")

    code = run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                       engine_factory=lambda s: engine, sleep=no_sleep, heartbeat_sec=0.05,
                       server=dataclasses.asdict(handle))

    assert code == EXIT_IDLE
    assert "serve" not in fake.commands() and "stop" not in fake.commands()
    behavior = next(c["argv"] for c in fake.calls() if c["argv"][0] == "behavior")
    assert behavior[behavior.index("--base-url") + 1] == handle.base_url
    rows = repo.conn.execute("SELECT stage, status FROM jobs ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("behavior", "done"), ("contribution", "done"),
                                        ("pfb", "done")]
    assert engine.chat_for == handle
    assert alive(handle.pid)   # still warm
    Pfb3x3Engine(settings).stop_server(handle)


def test_a_session_given_a_dead_shared_server_ends_without_claiming(settings, repo, tmp_path):
    sid, _ = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    dead = ServerHandle(pid=2 ** 22 + 7, port=8200, model="qwen3vl-8b",
                        base_url="http://localhost:8200/v1")
    code = run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                       engine_factory=Engine, sleep=lambda s: None, heartbeat_sec=0.05,
                       server=dead)
    assert code == EXIT_SERVER_DOWN and repo.get_job(jid).status == JobStatus.queued


def test_track_session_does_not_idle(settings, repo, fake):
    settings = settings.model_copy(update={"session_idle_sec": 300})

    def no_sleep(s):
        raise AssertionError("the track session ends at once when its queue is empty")

    code = run_session(None, "track", settings=settings, guard=Guard(settings),
                       engine_factory=Engine, sleep=no_sleep, heartbeat_sec=0.05)
    assert code == EXIT_IDLE


# --- parallel pfb jobs ---

def test_pfb_jobs_run_side_by_side_each_with_its_own_connection(settings, repo, tmp_path,
                                                                monkeypatch):
    sid, subjects = submission(repo, tmp_path, subjects=("Pa", "Pb", "Pc"))
    contribution = repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")
    pfbs = [repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=s) for s in subjects]
    handle = fake_vllm(settings)
    barrier = threading.Barrier(3, timeout=10)
    seen = []

    def fake_execute(job, ctx):
        seen.append((job.stage, threading.current_thread().name, id(ctx.repo)))
        if job.stage == Stage.pfb:
            barrier.wait()   # only passes when all three run at once
        ctx.repo.finish_job(job.id, JobStatus.done)

    monkeypatch.setattr(session_mod, "execute_job", fake_execute)
    try:
        code = run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                           engine_factory=Engine, sleep=lambda s: None, heartbeat_sec=0.05,
                           server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)
    assert code == EXIT_IDLE
    assert all(repo.get_job(j).status == JobStatus.done for j in [contribution, *pfbs])
    assert seen[0][0] == Stage.contribution
    pfb_runs = [s for s in seen if s[0] == Stage.pfb]
    assert len(pfb_runs) == 3 and len({s[2] for s in pfb_runs}) == 3   # own Repo each
    assert len({s[1] for s in pfb_runs}) == 3                           # own thread each


def test_a_yield_in_one_parallel_pfb_job_ends_the_session_after_all(settings, repo, tmp_path,
                                                                    monkeypatch):
    sid, subjects = submission(repo, tmp_path, subjects=("Pa", "Pb"))
    first, second = (repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=s)
                     for s in subjects)
    handle = fake_vllm(settings)

    def fake_execute(job, ctx):
        if job.id == first:
            ctx.repo.finish_job(job.id, JobStatus.queued)
            raise YieldRequested()
        ctx.repo.finish_job(job.id, JobStatus.done)

    monkeypatch.setattr(session_mod, "execute_job", fake_execute)
    try:
        code = run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                           engine_factory=Engine, sleep=lambda s: None, heartbeat_sec=0.05,
                           server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)
    assert code == EXIT_YIELD
    assert repo.get_job(first).status == JobStatus.queued
    assert repo.get_job(second).status == JobStatus.done


# --- the worker's shared server ---

def spawner(repo, calls, finish=True):
    """`spawn` stub: records (session, server) and finishes the claimed key's jobs."""
    def spawn(key, server=None):
        calls.append((key, server))
        claim = {"detect:cpu": "detect"}.get(key, key)
        if finish:
            while (job := repo.claim_next_job({claim})) is not None:
                repo.finish_job(job.id, JobStatus.done)
        return EXIT_IDLE
    return spawn


def test_a_confirmed_work_prewarms_the_server_while_track_runs(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("track", None)]          # track did not wait for the server
        assert shared.starting or shared.handle is not None
        handle = ready(shared)
        serve = next(c for c in fake.calls() if c["argv"][0] == "serve")
        assert serve["argv"][1:] == ["qwen3vl-8b", "8200"]
        assert 34 in guard.can_start_calls
        # behavior queued after track: it gets the warm server, no budget check of its own
        repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
        guard.can_start_calls.clear()
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls[-1] == ("pfb3x3:standard", dataclasses.asdict(handle))
        assert guard.can_start_calls == []
        assert fake.commands().count("serve") == 1
    finally:
        shared.close()
    assert not alive(handle.pid)
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


def test_no_prewarm_without_room_or_for_another_model(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard = Guard(settings, free_gb=30.0)   # 34 + 8 GB do not fit
    shared = shared_server(settings, guard)
    shared.maintain(repo)
    assert not shared.holds_gpu()
    high, _ = submission(repo, tmp_path, model_key="high")
    repo.conn.execute("DELETE FROM jobs")
    repo.enqueue(high, Stage.track, "track", camera="cam1")
    guard.free_gb = 90.0
    shared.maintain(repo)
    assert not shared.holds_gpu() and "serve" not in fake.commands()


def test_the_server_stays_warm_then_stops_after_the_idle_time(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, calls, clock = Guard(settings), [], Clock()
    shared = shared_server(settings, guard, clock=clock, warm_idle_s=600)
    try:
        assert shared.request() == ("starting", "")
        handle = ready(shared)
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("pfb3x3:standard", dataclasses.asdict(handle))]
        clock.t += 599
        shared.maintain(repo)
        assert shared.handle == handle and alive(handle.pid)
        # a second burst reuses it
        repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls[-1] == ("pfb3x3:standard", dataclasses.asdict(handle))
        clock.t += 601
        shared.maintain(repo)
        assert shared.handle is None and not alive(handle.pid)
        assert fake.commands().count("serve") == 1
    finally:
        shared.close()


def test_the_warm_server_yields_to_other_gpu_users(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard = Guard(settings)
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        guard.wanted = True   # someone else needs the GPU; the job is still queued
        shared.maintain(repo)
        assert shared.handle is None and not alive(handle.pid)
    finally:
        shared.close()


def test_the_watcher_yields_while_another_session_runs(settings, repo, tmp_path, fake):
    guard = Guard(settings)
    shared = shared_server(settings, guard, watch_sec=0.05)
    shared.open()
    try:
        shared.request()
        handle = ready(shared)
        with shared.lease():   # a session on it: the watcher leaves it alone
            guard.wanted = True
            threading.Event().wait(0.3)
            assert alive(handle.pid)
        for _ in range(200):
            if not alive(handle.pid):
                break
            threading.Event().wait(0.05)
        assert shared.handle is None and not alive(handle.pid)
    finally:
        shared.close()


def test_a_session_that_yields_for_the_gpu_stops_the_server(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, slept = Guard(settings), []
    shared = shared_server(settings, guard)

    def yielding(key, server=None):
        guard.wanted = True
        return EXIT_YIELD

    try:
        shared.request()
        handle = ready(shared)
        worker_main._gpu_pass(repo, settings, guard, yielding, slept.append, {}, shared=shared)
        assert shared.handle is None and not alive(handle.pid)
        assert slept == [POLL]
    finally:
        shared.close()


def test_the_content_check_waits_for_the_starting_server(settings, repo, tmp_path, fake,
                                                         monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "1.5")
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.detect, "detect")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, calls, slept = Guard(settings), [], []
    shared = shared_server(settings, guard)
    try:
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), slept.append, {},
                              shared=shared)
        assert calls == [] and slept == [IDLE_POLL_SEC]   # nothing else starts meanwhile
        handle = ready(shared)
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), slept.append, {},
                              shared=shared)
        assert calls == [("detect", dataclasses.asdict(handle))]
        assert alive(handle.pid)   # left warm for the work that follows
    finally:
        shared.close()


def test_the_content_check_runs_on_the_cpu_without_room(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.detect, "detect")
    guard, calls = Guard(settings, free_gb=10.0), []
    shared = shared_server(settings, guard)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                          shared=shared)
    assert calls == [("detect:cpu", None)] and "serve" not in fake.commands()


def test_another_models_server_stops_the_shared_one_first(settings, repo, tmp_path, fake):
    high, _ = submission(repo, tmp_path, model_key="high")
    repo.enqueue(high, Stage.behavior, "pfb3x3:high", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        seen_alive = []

        def spawn(key, server=None):
            seen_alive.append(alive(handle.pid))
            return spawner(repo, calls)(key, server)

        worker_main._gpu_pass(repo, settings, guard, spawn, lambda s: None, {}, shared=shared)
        assert calls == [("pfb3x3:high", None)] and seen_alive == [False]
    finally:
        shared.close()


def test_an_idle_warm_server_makes_room_for_a_session_that_does_not_fit(settings, repo,
                                                                        tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        def can_start(budget):     # the warm server's 34 GB leave 10 GB: too little for track
            guard.can_start_calls.append(budget)
            free = 10.0 if shared.handle is not None else 44.0
            return (budget + 8 <= free, "空き不足")

        guard.can_start = can_start
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("track", None)] and not alive(handle.pid)
    finally:
        shared.close()


def test_a_small_session_gets_room_when_stopping_the_server_frees_enough(settings, repo,
                                                                         tmp_path, fake):
    # track needs 4 + 8 (margin) GB. With the warm server 7 GB are free (between the yield
    # threshold and the margin); stopping it frees 34: free >= budget - 34 + margin holds
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        guard._query = lambda: GpuSnapshot(
            TOTAL_GB, TOTAL_GB - (7.0 if shared.handle is not None else 41.0))
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("track", None)] and not alive(handle.pid)
    finally:
        shared.close()


def test_a_failed_start_counts_as_a_load_failure(settings, repo, tmp_path, fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    sid, _ = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, calls, slept, failures = Guard(settings), [], [], {}
    shared = shared_server(settings, guard)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), slept.append, failures,
                          shared=shared)
    assert calls == [] and slept == [IDLE_POLL_SEC]   # starting: polled again soon
    shared.wait_started(30)
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), slept.append, failures,
                          shared=shared)
    assert calls == [] and failures == {"pfb3x3:standard": 1}
    assert repo.latest_audit("model_load_failed", "pfb3x3:standard") is not None
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu   # not retried before gpu_poll_sec
    assert slept[-1] == POLL
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


def test_the_worker_stops_the_shared_server_when_it_ends(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    shared.request()
    handle = ready(shared)
    worker_loop(settings, guard=guard, spawn=spawner(repo, calls), sleep=lambda s: None,
                once=True, shared=shared, on_start=lambda s: None)
    assert calls == [("pfb3x3:standard", dataclasses.asdict(handle))]
    assert not alive(handle.pid)
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


def test_main_keeps_a_shared_server_only_without_once(tmp_path, monkeypatch):
    seen, linecalls, videos_ = [], [], []

    def fake_loop(settings, *, guard, spawn, once, shared, linecall, videos=None):
        seen.append((once, shared))
        linecalls.append(linecall)
        videos_.append(videos)

    monkeypatch.setattr(worker_main, "worker_loop", fake_loop)
    monkeypatch.setattr(worker_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(worker_main, "exit_on_signals", lambda: None)
    # a temp data_dir: main takes <data_dir>/worker.lock and worker.running (ops lane)
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    worker_main.main(["--settings", str(cfg)])
    worker_main.main(["--settings", str(cfg), "--once"])
    assert isinstance(seen[0][1], SharedServer) and seen[0][0] is False
    assert seen[1] == (True, None)
    # linecall runs beside the sessions only without --once (its child would outlive it)
    from pfb_app.worker.linecall_runner import LinecallRunner
    assert isinstance(linecalls[0], LinecallRunner) and linecalls[1] is None
    # 解説動画 are made by a thread only without --once (with it: at the end of the pass)
    assert isinstance(videos_[0], worker_main.ExplainerRunner) and videos_[1] is None


# --- review round 1 ---

def test_while_the_server_starts_other_sessions_count_its_budget(settings, repo, tmp_path, fake,
                                                                 monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "1.5")
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings, free_gb=44.0), []   # 34 + 8 fits, 34 + 4 + 8 does not
    shared = shared_server(settings, guard)
    at_spawn = []

    def spawn(key, server=None):
        at_spawn.append(shared.holds_gpu())
        return spawner(repo, calls)(key, server)

    try:
        worker_main._gpu_pass(repo, settings, guard, spawn, lambda s: None, {}, shared=shared)
        assert 34 in guard.can_start_calls           # pre-warm began
        assert 4 + 34 in guard.can_start_calls       # track counted the starting server
        # our own track goes first: the pre-warm was cancelled, never both at once
        assert calls == [("track", None)] and at_spawn == [False]
    finally:
        shared.close()


def test_a_server_listening_beyond_this_machine_is_refused(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    shared.engine.bound = ["0.0.0.0"]
    try:
        assert shared.request() == ("starting", "")
        shared.wait_started(30)
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert shared.disabled and "0.0.0.0" in shared.disabled
        assert shared.handle is None
        assert calls == [("pfb3x3:standard", None)]   # the old way: its own server
        assert fake.commands().count("serve") == 1 and "stop" in fake.commands()
        assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()
    finally:
        shared.close()


def test_bind_check_and_ss_parsing():
    from pfb_app.engine.pfb3x3 import parse_listen_addresses
    from pfb_app.worker.shared_server import bind_problem
    out = ("LISTEN 0 2048 0.0.0.0:8200 0.0.0.0:*\n"
           "LISTEN 0 2048 127.0.0.1:8201 0.0.0.0:*\nLISTEN 0 50 [::1]:8202 [::]:*\n")
    assert parse_listen_addresses(out) == ["0.0.0.0", "127.0.0.1", "[::1]"]
    assert bind_problem(["127.0.0.1"]) is None and bind_problem(["[::1]"]) is None
    assert "0.0.0.0" in bind_problem(["127.0.0.1", "0.0.0.0"])
    assert bind_problem([]) and bind_problem(None)


def test_the_engine_is_asked_to_listen_on_localhost(settings, fake):
    engine = Engine(settings)
    handle = engine.start_server("qwen3vl-8b", 0.347)
    try:
        serve = next(c for c in fake.calls() if c["argv"][0] == "serve")
        assert serve["env"]["PFB_SERVE_HOST"] == "127.0.0.1"
    finally:
        engine.stop_server(handle)


class Stub:
    """A stand-in for vLLM's chat endpoint on 127.0.0.1; `answer(body) -> (status, obj)`."""

    def __init__(self, answer):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        bodies = self.bodies = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                status, obj = answer(body)
                data = (obj if isinstance(obj, str) else json.dumps(obj)).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


EVENT_IDS = ("cam1-0001-01", "cam1-0001-02")


def judgments(ids):
    return [{"ref_id": i, "contribution_interpretation": "解釈",
             "is_positive_contribution": True, "rubric_criteria": []} for i in ids]


def behavior_and_contribution(repo, sid, contribution=True):
    """prepare + behavior (one event each for Pa and Pb) [+ contribution] of cam1."""
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    events = [{"video_time": "0:05", "participant": pid, "observed_behavior": "パスを出す",
               "utterance": "", "speaker": "unknown", "evidence_source": "video",
               "confidence": "high", "event_id": eid, "camera": "cam1",
               "task_elapsed": "10:05"} for pid, eid in zip(("Pa", "Pb"), EVENT_IDS)]
    behavior = save(repo, sid, Stage.behavior, {"events": events}, camera="cam1")
    if contribution:
        save(repo, sid, Stage.contribution, {"judgments": judgments(EVENT_IDS)},
             camera="cam1", upstream=[behavior.id])


def served_session(settings, port):
    """A pfb3x3:standard session on a 'server' at `port` (the real chat backend)."""
    handle = fake_vllm(settings)
    handle = dataclasses.replace(handle, port=port, base_url=f"http://127.0.0.1:{port}/v1")

    class Real(Pfb3x3Engine):   # the real chat backend; nothing else of the engine is used
        def __init__(self, s):
            super().__init__(s, tick_sec=0.05, term_grace_sec=2)

    try:
        return run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                           engine_factory=Real, sleep=lambda s: None, heartbeat_sec=0.05,
                           server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)


@pytest.mark.parametrize("server", ["dead", "oom"])
def test_a_failing_server_requeues_contribution_and_ends_the_session(settings, repo, tmp_path,
                                                                     server):
    sid, _ = submission(repo, tmp_path)
    behavior_and_contribution(repo, sid, contribution=False)
    jid = repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")
    later = repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=1)
    stub = None if server == "dead" else Stub(
        lambda body: (500, {"error": "CUDA out of memory. Tried to allocate 2.00 GiB"}))
    try:
        code = served_session(settings, free_port() if stub is None else stub.port)
    finally:
        if stub is not None:
            stub.close()
    assert code == EXIT_SERVER_DOWN
    assert repo.get_job(jid).status == JobStatus.queued        # not failed_retryable
    assert repo.get_job(later).status == JobStatus.queued      # never tried on a broken server
    assert repo.latest_audit("engine_server_down", f"job:{jid}") is not None
    if stub is not None:
        assert len(stub.bodies) == 1                           # no retry against it


def test_a_dead_server_during_parallel_pfb_requeues_them_all(settings, repo, tmp_path):
    sid, subjects = submission(repo, tmp_path, subjects=("Pa", "Pb"))
    behavior_and_contribution(repo, sid)
    jobs = [repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=s) for s in subjects]
    code = served_session(settings, free_port())
    assert code == EXIT_SERVER_DOWN
    assert [repo.get_job(j).status for j in jobs] == [JobStatus.queued, JobStatus.queued]


def test_the_server_down_message_is_japanese_after_the_limit(settings, repo, tmp_path):
    sid, _ = submission(repo, tmp_path)
    behavior_and_contribution(repo, sid, contribution=False)
    jid = repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")
    for _ in range(3):
        served_session(settings, free_port())
    job = repo.get_job(jid)
    assert job.status == JobStatus.failed_retryable
    assert job.error.splitlines()[0].startswith("解析サーバー")
    assert "解析サーバーに接続できませんでした。しばらくしてから再実行してください。" in job.error


def test_contribution_states_the_schema_in_the_text(settings, repo, tmp_path):
    sid, _ = submission(repo, tmp_path, subjects=())   # no pfb follows
    behavior_and_contribution(repo, sid, contribution=False)
    repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")

    def answer(body):
        if "response_format" in body:
            return 400, {"error": "response_format not supported"}
        return 200, {"choices": [{"message": {"content": json.dumps(
            {"judgments": judgments(EVENT_IDS)})}}]}

    stub = Stub(answer)
    try:
        code = served_session(settings, stub.port)
    finally:
        stub.close()
    assert code == EXIT_IDLE
    texts = [b["messages"][1]["content"] for b in stub.bodies]
    assert len(texts) >= 2 and all("次のJSONスキーマに従うJSONのみを出力してください" in t
                                   and '"judgments"' in t for t in texts)
    assert "response_format" not in stub.bodies[-1]   # the retry kept the schema in the text
    assert repo.latest_run(sid, Stage.contribution, "cam1").status == "ok"


def test_translation_keeps_its_plain_requests(settings):
    handle = ServerHandle(pid=1, port=8200, model="qwen3vl-8b",
                          base_url="http://localhost:8200/v1")
    engine = Pfb3x3Engine(settings)
    assert engine.translator(handle)._schema_hint is False
    assert engine.translator(handle)._strict is False
    assert engine.chat_backend(handle)._schema_hint is True


def test_a_failed_call_is_retried_without_an_error_note(repo):
    class Flaky(Recording):
        def generate(self, reqs):
            self.batches.append(len(reqs))
            self.texts = getattr(self, "texts", []) + [r.user_text for r in reqs]
            first = len(self.batches) == 1
            return [RawResponse(text="" if first else json.dumps({"judgments": []}),
                                prompt_tokens=None, completion_tokens=None, duration_sec=0.0,
                                error="HTTP 400: bad" if first else None) for _ in reqs]

    backend = Flaky({})
    req = ChatRequest("sys", "元の依頼", [], ContributionOutput.model_json_schema(), 100)
    (result,) = call_structured_batch(backend, repo, [req], ContributionOutput, max_retries=2,
                                      yield_check=lambda: False)
    assert result.parsed is not None
    assert backend.texts == ["元の依頼", "元の依頼"]   # the retry is the same request


def test_a_session_that_found_the_server_failing_stops_it(settings, repo, tmp_path, fake):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    guard = Guard(settings)
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        worker_main._gpu_pass(repo, settings, guard, lambda key, server=None: EXIT_SERVER_DOWN,
                              lambda s: None, {}, shared=shared)
        assert shared.handle is None and not alive(handle.pid)
    finally:
        shared.close()


def test_the_watcher_ends_an_idle_warm_server_while_another_session_runs(settings, repo,
                                                                         tmp_path, fake):
    clock = Clock()
    guard = Guard(settings)
    shared = shared_server(settings, guard, clock=clock, warm_idle_s=600, watch_sec=0.05)
    shared.open(files.db_path(settings))
    try:
        shared.request()
        handle = ready(shared)
        clock.t += 601   # nothing queued for it
        for _ in range(200):
            if not alive(handle.pid):
                break
            threading.Event().wait(0.05)
        assert shared.handle is None and not alive(handle.pid)
    finally:
        shared.close()


@pytest.mark.parametrize("key,stage", [("standard", Stage.evidence), ("whisper", Stage.transcribe)])
def test_an_in_process_model_never_runs_beside_the_warm_server(settings, repo, tmp_path, fake,
                                                                key, stage):
    sid, _ = submission(repo, tmp_path, engine="paper")
    repo.enqueue(sid, stage, key)
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        seen = []

        def spawn(k, server=None):
            seen.append(alive(handle.pid))
            return spawner(repo, calls)(k, server)

        worker_main._gpu_pass(repo, settings, guard, spawn, lambda s: None, {}, shared=shared)
        assert calls == [(key, None)] and seen == [False]
    finally:
        shared.close()


def test_a_start_failure_seen_by_request_is_still_reported(settings, repo, tmp_path, fake,
                                                           monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    guard = Guard(settings)
    shared = shared_server(settings, guard)
    shared.request()
    shared.wait_started(30)
    assert shared.request()[0] == "no"            # polled here first
    failure = shared.maintain(repo)               # ... and reported here all the same
    assert failure is not None and failure.kind == "error"
    assert shared.maintain(repo) is None          # once


def test_no_prewarm_after_a_start_error_or_while_a_job_waits_for_the_gpu(settings, repo,
                                                                          tmp_path, fake,
                                                                          monkeypatch):
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    clock = Clock()
    guard = Guard(settings)
    shared = shared_server(settings, guard, clock=clock)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "serve")
    shared.maintain(repo)
    shared.wait_started(30)
    assert shared.maintain(repo).kind == "error"
    clock.t += 10 * POLL                          # long after the cool-down
    monkeypatch.delenv("FAKE_ENGINE_EXIT")
    shared.maintain(repo)
    assert not shared.holds_gpu()                 # no pre-warm after an error
    other, _ = submission(repo, tmp_path, model_key="high")
    fresh = shared_server(settings, guard, clock=clock)
    repo.enqueue(other, Stage.behavior, "pfb3x3:high", camera="cam1")
    repo.mark_waiting_gpu("pfb3x3:high", "空き不足")
    fresh.maintain(repo)
    assert not fresh.holds_gpu()                  # others wait for the GPU: no pre-warm


def test_the_warm_server_is_not_stopped_for_a_session_that_would_not_fit_anyway(
        settings, repo, tmp_path, fake):
    high, _ = submission(repo, tmp_path, model_key="track_only")
    repo.enqueue(high, Stage.track, "track", camera="cam1")
    guard, calls = Guard(settings), []
    shared = shared_server(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        guard.can_start = lambda budget: (False, "空き不足")   # even without the server
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [] and alive(handle.pid)   # no stop/start loop
    finally:
        shared.close()


def test_close_always_ends_with_the_pid_verified_stop(settings, fake):
    guard = Guard(settings)
    shared = shared_server(settings, guard)
    shared.request()
    ready(shared)

    def interrupted(reason):
        raise SystemExit(129)   # a second signal while stopping

    shared.stop = interrupted
    with pytest.raises(SystemExit):
        shared.close()
    assert shared.engine.stale_stops >= 1
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


def test_a_crashed_cpu_check_leaves_the_starting_shared_server_alone(settings, repo, tmp_path,
                                                                     fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "2")
    sid, _ = submission(repo, tmp_path)
    repo.enqueue(sid, Stage.detect, "detect")
    guard = Guard(settings)
    settings_slow = settings.model_copy(update={"detect_timeout_sec": 0})   # do not wait
    shared = shared_server(settings_slow, guard)
    try:
        shared.request()
        worker_main._gpu_pass(repo, settings_slow, guard, lambda key, server=None: -9,
                              lambda s: None, {}, shared=shared)   # detect:cpu crashed
        handle = ready(shared)
        assert alive(handle.pid)
    finally:
        shared.close()


def test_extra_pfb_jobs_never_skip_ahead_and_skip_deleted_works(settings, repo, tmp_path,
                                                                monkeypatch):
    sid, subjects = submission(repo, tmp_path, subjects=("Pa", "Pb", "Pc"))
    first = repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=subjects[0])
    other, other_subjects = submission(repo, tmp_path, subjects=("Qa",))
    gone = repo.enqueue(other, Stage.pfb, "pfb3x3:standard", subject_id=other_subjects[0])
    behavior = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    late = repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=subjects[1])
    meta = repo.get_submission(other)["meta"]
    repo.conn.execute("UPDATE submissions SET meta=? WHERE id=?",
                      (json.dumps({**meta, "delete_requested": True}), other))
    handle = fake_vllm(settings)
    order = []

    def fake_execute(job, ctx):
        order.append(job.id)
        ctx.repo.finish_job(job.id, JobStatus.done)

    monkeypatch.setattr(session_mod, "execute_job", fake_execute)
    try:
        run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                    engine_factory=Engine, sleep=lambda s: None, heartbeat_sec=0.05,
                    server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)
    assert gone not in order
    assert repo.conn.execute("SELECT COUNT(*) FROM jobs WHERE id=?", (gone,)).fetchone()[0] == 0
    assert order.index(behavior) < order.index(late)   # the later pfb waited its turn
    assert order[0] == first


def test_a_hook_after_pfb_runs_on_the_jobs_own_connection(settings, repo, tmp_path,
                                                          monkeypatch):
    """What runs after a pfb job (e.g. automatic approval queued by enqueue_followups) runs
    in the job's thread with the job's own Repo, never the session's connection."""
    sid, subjects = submission(repo, tmp_path, subjects=("Pa", "Pb"))
    behavior_and_contribution(repo, sid)
    jobs = [repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=s) for s in subjects]
    seen = []
    real = runner.enqueue_followups

    def hook(r, job):
        seen.append((job.id, r is not repo, threading.current_thread().name))
        r.audit("after_pfb_hook", f"job:{job.id}", {})   # writes through that connection
        return real(r, job)

    monkeypatch.setattr(runner, "enqueue_followups", hook)
    handle = fake_vllm(settings)
    try:
        code = run_session(None, "pfb3x3:standard", settings=settings, guard=Guard(settings),
                           engine_factory=Engine, sleep=lambda s: None, heartbeat_sec=0.05,
                           server=handle)
    finally:
        Pfb3x3Engine(settings).stop_server(handle)
    assert code == EXIT_IDLE
    # the two feedback jobs, then the group's tips job queued after the last of them
    assert sorted(j for j, _, _ in seen)[:2] == jobs and len(seen) == 3
    assert all(own for _, own, _ in seen) and all(t.startswith("pfb-job") for _, _, t in seen)
    assert all(repo.latest_audit("after_pfb_hook", f"job:{j}") for j in jobs)


def test_retrying_an_old_in_process_job_moves_it_to_the_pfb3x3_session(repo, tmp_path):
    sid, (subj,) = submission(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.pfb, "standard", subject_id=subj)   # an older release's key
    job = repo.claim_next_job({"standard"})
    repo.finish_job(job.id, JobStatus.failed_retryable, "x")
    assert repo.retry_job(jid)
    assert repo.get_job(jid).model_key == "pfb3x3:standard"
    paper, (ps,) = submission(repo, tmp_path, engine="paper")
    pj = repo.enqueue(paper, Stage.pfb, "standard", subject_id=ps)
    repo.finish_job(repo.claim_next_job({"standard"}).id, JobStatus.failed_retryable, "x")
    assert repo.retry_job(pj) and repo.get_job(pj).model_key == "standard"


def test_eta_counts_parallel_pfb_and_prefers_served_durations(repo, tmp_path):
    from pfb_app.ui.wizard.analysis import PARALLEL_PFB, weight_counts
    assert PARALLEL_PFB == session_mod.PARALLEL_JOBS
    assert weight_counts({"pfb": 13, "track": 2}, "pfb3x3") == {"pfb": 2, "track": 2}
    assert weight_counts({"pfb": 13}, "paper") == {"pfb": 13}
    sid, _ = submission(repo, tmp_path)

    def run(stage, sec, model_id):
        repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=stage, camera="cam1", subject_id=None,
            version=0, source="model", upstream_run_ids=[], prompt_hash=None,
            model_id=model_id, output={}, status="ok", error=None, warnings=[],
            duration_sec=sec, created_at=""))

    run(Stage.contribution, 400.0, "Qwen/Qwen3-VL-8B-Instruct")   # in-process, older
    run(Stage.track, 300.0, None)
    assert repo._stage_durations_all("pfb3x3", 50)["contribution"] == [400.0]
    run(Stage.contribution, 30.0, "served:qwen3vl-8b")
    run(Stage.contribution, 20.0, "served:qwen3vl-8b")
    d = repo._stage_durations_all("pfb3x3", 50)
    assert d["contribution"] == [20.0, 30.0] and d["track"] == [300.0]   # newest first


def test_a_cached_plain_answer_is_not_reused_for_a_request_with_the_schema_text(repo):
    from pfb_app.inference.served import ServedChatBackend
    from pfb_app.inference.structured import request_hash
    good = json.dumps({"judgments": []})
    stub = Stub(lambda body: (200, {"choices": [{"message": {"content": good}}]}))
    try:
        url = f"http://127.0.0.1:{stub.port}/v1"
        r = ChatRequest("sys", "t", [], ContributionOutput.model_json_schema(), 100)
        plain = ServedChatBackend(url, "m")
        hint = ServedChatBackend(url, "m", schema_hint=True)
        assert plain.model_id == hint.model_id
        # translation's plain key is what it always was, so its call_log answers still hit
        assert request_hash(plain.model_id, r, 0, plain.cache_variant) == request_hash(
            plain.model_id, r, 0)
        for backend in (plain, hint, session_mod._ServerGuard(
                ServedChatBackend(url, "m", schema_hint=True, strict_server=True))):
            results = call_structured_batch(backend, repo, [r], ContributionOutput,
                                            max_retries=0, yield_check=lambda: False)
            assert results[0].parsed is not None
    finally:
        stub.close()
    # plain asked; with the schema text asked again (no reuse); the guarded one: a cache hit
    assert len(stub.bodies) == 2
    assert "次のJSONスキーマ" not in stub.bodies[0]["messages"][1]["content"]
    assert "次のJSONスキーマ" in stub.bodies[1]["messages"][1]["content"]


def slow_stub(marker, slow_times, answer_text, delay=1.5):
    """Answers at once, except the first `slow_times` requests whose user text contains
    `marker`: those answer only after `delay` s (beyond the client timeout)."""
    import time
    seen, lock = {}, threading.Lock()

    def answer(body):
        text = body["messages"][1]["content"]
        with lock:
            n = seen[marker] = seen.get(marker, 0) + (marker in text)
        if marker in text and n <= slow_times:
            time.sleep(delay)
        return 200, {"choices": [{"message": {"content": answer_text}}]}

    return Stub(answer)


@pytest.mark.parametrize("slow_times", [1, 2])
def test_a_timeout_is_asked_once_more_before_it_ends_the_call(slow_times):
    from pfb_app.inference.served import ServedChatBackend, ServerFailed
    stub = slow_stub("r3", slow_times, json.dumps({"judgments": []}))
    try:
        backend = ServedChatBackend(f"http://127.0.0.1:{stub.port}/v1", "m", timeout_sec=0.5,
                                    strict_server=True)
        reqs = [ChatRequest("sys", f"r{i}", [], {"type": "object"}, 10) for i in range(8)]
        if slow_times == 1:
            out = backend.generate(reqs)
            assert [o.error for o in out] == [None] * 8 and all(o.text for o in out)
        else:
            with pytest.raises(ServerFailed):
                backend.generate(reqs)
    finally:
        stub.close()
    assert len(stub.bodies) == 8 + 1   # only the timed-out request was asked again


def eight_subjects_ready_for_pfb(repo, tmp_path):
    names = tuple(f"Q{i}" for i in range(8))
    sid, subjects = submission(repo, tmp_path, subjects=names)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    ids = [f"cam1-0001-{i + 1:02d}" for i in range(8)]
    events = [{"video_time": "0:05", "participant": pid, "observed_behavior": f"行動{i}番",
               "utterance": "", "speaker": "unknown", "evidence_source": "video",
               "confidence": "high", "event_id": eid, "camera": "cam1",
               "task_elapsed": "10:05"} for i, (pid, eid) in enumerate(zip(names, ids))]
    behavior = save(repo, sid, Stage.behavior, {"events": events}, camera="cam1")
    save(repo, sid, Stage.contribution, {"judgments": judgments(ids)}, camera="cam1",
         upstream=[behavior.id])
    return sid, subjects


@pytest.mark.parametrize("slow_times", [1, 2])
def test_one_pfb_timeout_among_eight_does_not_end_the_session(settings, repo, tmp_path,
                                                              monkeypatch, slow_times):
    from pfb_app.engine import pfb3x3 as engine_mod
    monkeypatch.setattr(engine_mod, "SERVED_TIMEOUT_SEC", 0.5)
    sid, subjects = eight_subjects_ready_for_pfb(repo, tmp_path)
    jobs = [repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=s) for s in subjects]
    stub = slow_stub("行動3番", slow_times, json.dumps(
        {"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []}))
    try:
        code = served_session(settings, stub.port)
    finally:
        stub.close()
    statuses = [repo.get_job(j).status for j in jobs]
    if slow_times == 1:   # asked once more: the session goes on, all 8 done
        assert code == EXIT_IDLE
        assert statuses == [JobStatus.done] * 8
        assert all(repo.latest_run(sid, Stage.pfb, subject_id=s).status == "ok"
                   for s in subjects)
    else:                 # timed out twice: the server is down, as before
        assert code == EXIT_SERVER_DOWN
        assert repo.get_job(jobs[3]).status == JobStatus.queued
        assert repo.latest_audit("engine_server_down", f"job:{jobs[3]}") is not None
