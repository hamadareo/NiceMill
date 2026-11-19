"""Worker loop and inference session lifecycle. The GPU is always a stub here."""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from pdfhelp import make_pdf
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.gpu import GpuQueryError
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import main as worker_main
from pfb_app.worker.main import spawn_session, worker_loop
from pfb_app.worker.session import EXIT_IDLE, EXIT_OOM, EXIT_YIELD, run_session

QUOTE = "We measured the distance to the shelter on a map"
POLL = 7


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    gpu_poll_sec=POLL, session_idle_sec=0)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


class StubGuard:
    """GpuGuard stand-in: fixed can_start answer, scripted should_yield answers."""

    def __init__(self, can=True, reason="空き 10.0GB < 必要 32GB（予算24+余白8）", yields=()):
        self.can, self.reason = can, reason
        self.yields = list(yields)
        self.can_start_calls: list[float] = []
        self.yield_calls = 0

    def can_start(self, budget_gb):
        self.can_start_calls.append(budget_gb)
        return (True, "") if self.can else (False, self.reason)

    def should_yield(self):
        self.yield_calls += 1
        return self.yields.pop(0) if self.yields else False


def responder(req):
    title = req.json_schema.get("title")
    ids = [line.split(": ", 1)[1] for line in req.user_text.splitlines()
           if line.startswith("ref_id: ")]
    if title == "EvidenceOutput":
        return json.dumps({"items": [{"page": 1, "quote": QUOTE, "observed_content": "内容",
                                      "evidence_source": "text", "confidence": "high"}]})
    if title == "ContributionOutput":
        return json.dumps({"judgments": [
            {"ref_id": i, "contribution_interpretation": "解釈", "is_positive_contribution": True,
             "rubric_criteria": []} for i in ids]}, ensure_ascii=False)
    if title == "PfbOutput":
        return json.dumps({"pfb_status": "generated", "pfb": "調べたことが課題を進めていました。",
                           "evidence_refs": ids[:1]}, ensure_ascii=False)
    raise AssertionError(title)


def doc_submission(repo, tmp_path, name="A", prepared=False):
    aid = repo.create_assignment("課題", Kind.document, "防災を調べる課題", "ルーブリック",
                                 "standard", False)
    src = tmp_path / f"{name}.pdf"
    make_pdf(src, [QUOTE + "."])
    sid = repo.create_submission(aid, name, {"source_path": str(src),
                                             "original_filename": src.name})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    if prepared:
        repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=Stage.prepare, camera=None, subject_id=None,
            version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
            output={"pages": [{"number": 1, "text": QUOTE, "image_path": None}],
                    "pdf_path": str(src)},
            status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
    return sid, subj


def in_process_spawn(settings, guard, backend, calls):
    """`spawn` stub: run the session in this process with a fake backend."""
    def spawn(key):
        calls.append(key)
        return run_session(None, key, settings=settings, guard=guard,
                           backend_factory=lambda s, k: backend, sleep=lambda s: None)
    return spawn


def statuses(repo):
    return [(r["stage"], r["status"]) for r in
            repo.conn.execute("SELECT stage, status FROM jobs ORDER BY id")]


# --- the tests named in the brief ---

def test_document_job_chain_runs_to_pfb(settings, repo, tmp_path):
    sid, subj = doc_submission(repo, tmp_path)
    repo.enqueue(sid, Stage.prepare, "cpu")
    guard, calls, sleeps = StubGuard(), [], []
    worker_loop(settings, guard=guard,
                spawn=in_process_spawn(settings, guard, FakeBackend(responder), calls),
                sleep=sleeps.append, once=True)
    assert calls == ["standard"]
    assert guard.can_start_calls == [24]
    assert statuses(repo) == [("prepare", "done"), ("evidence", "done"),
                              ("contribution", "done"), ("pfb", "done")]
    pfb = repo.latest_run(sid, Stage.pfb, subject_id=subj)
    assert pfb.status == "ok" and pfb.output["pfb_status"] == "generated"


def test_waiting_gpu_when_insufficient(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    guard, sleeps = StubGuard(can=False), []
    worker_loop(settings, guard=guard, spawn=lambda key: pytest.fail("must not spawn"),
                sleep=sleeps.append, once=True)
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu
    assert job.error == guard.reason
    assert sleeps == [POLL]


def test_gpu_query_error_counts_as_insufficient(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")

    class Broken(StubGuard):
        def can_start(self, budget_gb):
            raise GpuQueryError("nvidia-smiの実行に失敗しました")

    sleeps = []
    worker_loop(settings, guard=Broken(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=sleeps.append, once=True)
    job = repo.get_job(jid)
    assert job.status == JobStatus.waiting_gpu and "nvidia-smi" in job.error
    assert sleeps == [POLL]


def test_yield_requeues_and_session_exits_10(settings, repo, tmp_path):
    sid1, _ = doc_submission(repo, tmp_path, "A", prepared=True)
    sid2, _ = doc_submission(repo, tmp_path, "B", prepared=True)
    first = repo.enqueue(sid1, Stage.evidence, "standard")
    second = repo.enqueue(sid2, Stage.evidence, "standard")
    guard = StubGuard(yields=[False, True])  # GPU gets busy before the 2nd inference call
    backend = FakeBackend(responder)
    code = run_session(None, "standard", settings=settings, guard=guard,
                       backend_factory=lambda s, k: backend, sleep=lambda s: None)
    assert code == EXIT_YIELD == 10
    assert len(backend.calls) == 1
    assert repo.get_job(first).status == JobStatus.done
    assert repo.get_job(second).status == JobStatus.queued
    assert all(st != "running" for _, st in statuses(repo))
    assert ("contribution", "queued") in statuses(repo)


def test_oom_marks_retryable_and_exits_11(settings, repo, tmp_path):
    sid1, _ = doc_submission(repo, tmp_path, "A", prepared=True)
    sid2, _ = doc_submission(repo, tmp_path, "B", prepared=True)
    first = repo.enqueue(sid1, Stage.evidence, "standard")
    second = repo.enqueue(sid2, Stage.evidence, "standard")

    class OomBackend:
        model_id = "fake"

        def generate(self, reqs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        def close(self):
            pass

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: OomBackend(), sleep=lambda s: None)
    assert code == EXIT_OOM == 11
    job = repo.get_job(first)
    # recorded as a stop for the GPU, then put back once by services.auto_retry (the worker's
    # gpu_poll_sec wait before the next session is its backoff)
    assert job.status == JobStatus.queued
    assert "CUDA out of memory" in repo.latest_audit("auto_retry", f"job:{first}")["detail"]["error"]
    assert repo.get_job(second).status == JobStatus.queued  # untouched, session ended


def test_oom_while_loading_model_exits_11(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")

    def factory(s, k):
        raise RuntimeError("CUDA out of memory while loading weights")

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=factory, sleep=lambda s: None)
    assert code == EXIT_OOM
    assert repo.get_job(jid).status == JobStatus.queued  # never claimed


@pytest.mark.parametrize("code", [10, 11, 1, -9])
def test_no_immediate_respawn_after_yield(settings, repo, tmp_path, code):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    repo.enqueue(sid, Stage.evidence, "standard")
    calls, sleeps = [], []
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: calls.append(key) or code,
                sleep=sleeps.append, once=True)
    assert calls == ["standard"]
    assert sleeps == [POLL]


def test_crashed_session_leaves_no_running_job(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")

    def crashing_spawn(key):
        repo.claim_next_job({key})  # the child claimed the job, then died
        return 1

    worker_loop(settings, guard=StubGuard(), spawn=crashing_spawn, sleep=lambda s: None,
                once=True)
    job = repo.get_job(jid)
    # not left running: recovered (heartbeat timeout), then put back once (services.auto_retry)
    assert job.status == JobStatus.queued
    assert repo.latest_audit("auto_retry", f"job:{jid}")["detail"]["error"] == "heartbeat timeout"
    worker_loop(settings, guard=StubGuard(), spawn=crashing_spawn, sleep=lambda s: None,
                once=True)   # it dies again: now it stays stopped
    job = repo.get_job(jid)
    assert job.status == JobStatus.failed_retryable and job.error


# --- more lifecycle checks ---

def test_session_exits_0_when_queue_stays_empty(settings, repo):
    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: FakeBackend(responder),
                       sleep=lambda s: None)
    assert code == EXIT_IDLE == 0


def test_idle_session_waits_then_exits(settings, repo):
    settings = settings.model_copy(update={"session_idle_sec": 12})
    now = [0.0]
    slept = []

    def sleep(sec):
        slept.append(sec)
        now[0] += sec

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: FakeBackend(responder), sleep=sleep,
                       clock=lambda: now[0])
    assert code == 0 and sum(slept) >= 12


def test_idle_session_yields_when_gpu_gets_busy(settings, repo):
    settings = settings.model_copy(update={"session_idle_sec": 300})
    code = run_session(None, "standard", settings=settings, guard=StubGuard(yields=[True]),
                       backend_factory=lambda s, k: FakeBackend(responder),
                       sleep=lambda s: None)
    assert code == EXIT_YIELD


def test_idle_session_yields_as_soon_as_a_content_check_is_pending(settings, repo):
    settings = settings.model_copy(update={"session_idle_sec": 300})
    aid = repo.create_assignment("試合", Kind.video, "", "", "standard", False)
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": "/x.mp4"}})
    now, slept = [0.0], []

    def sleep(sec):              # a content check is queued during the first idle poll
        slept.append(sec)
        now[0] += sec
        if len(slept) == 1:
            repo.enqueue(sid, Stage.detect, "detect")

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: FakeBackend(responder), sleep=sleep,
                       clock=lambda: now[0])
    assert code == EXIT_YIELD
    assert len(slept) == 1 and sum(slept) < 300                # never the whole idle wait


def test_session_gpu_query_error_yields(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")

    class Broken(StubGuard):
        def should_yield(self):
            raise GpuQueryError("nvidia-smiの実行に失敗しました")

    code = run_session(None, "standard", settings=settings, guard=Broken(),
                       backend_factory=lambda s, k: FakeBackend(responder),
                       sleep=lambda s: None)
    assert code == EXIT_YIELD
    assert repo.get_job(jid).status == JobStatus.queued


def test_whisper_session_builds_no_llm_backend(settings, repo, tmp_path, monkeypatch):
    from pfb_app.pipeline import runner
    from pfb_app.preprocess.transcript import Segment
    aid = repo.create_assignment("課題", Kind.video, "課題文", "ルーブリック", "standard", False)
    sid = repo.create_submission(aid, "G", {"cameras": {"cam1": "/v/cam1.mp4"},
                                            "start_sec": None, "end_sec": None,
                                            "auto_transcribe": True})
    jid = repo.enqueue(sid, Stage.transcribe, "whisper")
    monkeypatch.setattr(runner, "transcribe", lambda p, s: [Segment(0.0, 1.0, "はい")])
    code = run_session(None, "whisper", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: pytest.fail("no LLM for whisper"),
                       sleep=lambda s: None)
    assert code == 0 and repo.get_job(jid).status == JobStatus.done


def test_heartbeat_while_job_runs(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    beats = []
    inner = FakeBackend(responder)

    class Slow:
        model_id = "fake"

        def generate(self, reqs):
            if repo.get_job(jid).status != JobStatus.running:  # later stages: no waiting
                return inner.generate(reqs)
            claimed = repo.get_job(jid).heartbeat_at
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                hb = repo.get_job(jid).heartbeat_at
                if hb != claimed:
                    beats.append(hb)
                    break
                time.sleep(0.02)
            return inner.generate(reqs)

        def close(self):
            pass

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: Slow(), sleep=lambda s: None,
                       heartbeat_sec=0.05)
    assert code == 0 and beats, "heartbeat_at was not refreshed during the job"
    assert not [t for t in threading.enumerate() if t.name == "pfb-heartbeat"]


def test_worker_recovers_stale_jobs_on_start(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    repo.claim_next_job({"standard"})
    repo.conn.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00+00:00'")
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("nothing queued"),
                sleep=lambda s: None, once=True)
    assert repo.get_job(jid).status == JobStatus.failed_retryable


def test_gpu_keys_are_checked_whisper_first(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    repo.enqueue(sid, Stage.evidence, "high")
    repo.enqueue(sid, Stage.transcribe, "whisper")
    repo.enqueue(sid, Stage.contribution, "standard")
    guard, sleeps = StubGuard(can=False), []
    worker_loop(settings, guard=guard, spawn=lambda key: pytest.fail("must not spawn"),
                sleep=sleeps.append, once=True)
    assert guard.can_start_calls == [6, 24, 52]
    assert {st for _, st in statuses(repo)} == {"waiting_gpu"}
    assert sleeps == [POLL]


def test_first_startable_key_is_spawned(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    repo.enqueue(sid, Stage.evidence, "high")
    repo.enqueue(sid, Stage.contribution, "standard")
    calls = []
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: calls.append(key) or 0,
                sleep=lambda s: None, once=True)
    assert calls == ["standard"]


# --- process plumbing ---

def _exit_with(code):
    sys.exit(code)


@pytest.mark.parametrize("code", [0, 10, 11])
def test_spawn_session_returns_child_exit_code(code):
    assert spawn_session("x", None, target=_exit_with, args=(code,)) == code


def test_worker_modules_do_not_import_torch():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, pfb_app.worker.main, pfb_app.worker.session, pfb_app.pipeline.runner;"
         " print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_cli_parses_arguments(monkeypatch, tmp_path):
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"data_dir: {tmp_path / 'data'}\ngpu_poll_sec: 3\n", encoding="utf-8")
    seen = {}

    def fake_loop(settings, *, guard, spawn, once, shared=None, linecall=None, videos=None):
        seen.update(settings=settings, once=once, spawn=spawn, shared=shared, linecall=linecall,
                    videos=videos)

    monkeypatch.setattr(worker_main, "worker_loop", fake_loop)
    worker_main.main(["--settings", str(cfg), "--once"])
    assert seen["once"] is True and seen["settings"].gpu_poll_sec == 3
    assert seen["spawn"].keywords == {"settings_path": str(cfg)}
    assert seen["shared"] is None   # --once: no warm server outlives the pass
    assert seen["videos"] is None   # --once: the 解説動画 are made at the end of the pass


# --- fix round 1 ---

class _StopLoop(Exception):
    pass


def _stop_after(n, on_sleep=None):
    """sleep stub: run on_sleep(i) on each call, stop the loop on call n."""
    calls = []

    def sleep(sec):
        calls.append(sec)
        if on_sleep:
            on_sleep(len(calls))
        if len(calls) >= n:
            raise _StopLoop()
    return sleep, calls


def test_stale_job_recovered_on_later_iteration(settings, repo, tmp_path):
    """Worker restarted quickly: the orphan's heartbeat is fresh at startup, stale later."""
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    repo.claim_next_job({"standard"})  # orphaned running job, fresh heartbeat

    def age(i):
        if i == 1:  # after the first pass: still running (fresh at startup), now goes stale
            assert repo.get_job(jid).status == JobStatus.running
            repo.conn.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00+00:00'")

    sleep, _ = _stop_after(2, age)
    with pytest.raises(_StopLoop):
        worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("nothing queued"),
                    sleep=sleep)
    job = repo.get_job(jid)
    assert job.status == JobStatus.failed_retryable and job.error == "heartbeat timeout"


def test_load_failures_stop_spawning_after_three(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    calls = []

    def failing_spawn(key):  # what the child does when the weights cannot be loaded
        calls.append(key)
        repo.audit("model_load_failed", key, {"error": "OSError: 重みが見つかりません"})
        return 12

    sleep, slept = _stop_after(5)
    with pytest.raises(_StopLoop):
        worker_loop(settings, guard=StubGuard(), spawn=failing_spawn, sleep=sleep)
    assert calls == ["standard"] * 3  # then no more loads onto the shared GPU
    assert slept[:3] == [POLL] * 3  # never respawned immediately
    job = repo.get_job(jid)
    assert job.status == JobStatus.failed_retryable
    assert job.error == "モデルを読み込めませんでした: OSError: 重みが見つかりません"

    # new jobs for the key -> spawning resumes
    repo.enqueue(sid, Stage.evidence, "standard")
    with pytest.raises(_StopLoop):
        worker_loop(settings, guard=StubGuard(), spawn=failing_spawn, sleep=_stop_after(1)[0])
    assert len(calls) == 4


def test_load_failure_count_resets_after_successful_session(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    codes = [12, 12, 10, 12, 12]

    def spawn(key):
        repo.audit("model_load_failed", key, {"error": "x"})
        return codes.pop(0)

    sleep, _ = _stop_after(5)
    with pytest.raises(_StopLoop):
        worker_loop(settings, guard=StubGuard(), spawn=spawn, sleep=sleep)
    assert repo.get_job(jid).status == JobStatus.queued


def test_session_load_failure_exits_12_and_records_reason(settings, repo, tmp_path):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")

    def factory(s, k):
        raise OSError("重みが見つかりません")

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=factory, sleep=lambda s: None)
    assert code == 12
    assert repo.latest_audit("model_load_failed", "standard")["detail"] == {
        "error": "OSError: 重みが見つかりません"}
    assert repo.get_job(jid).status == JobStatus.queued


def test_heartbeat_thread_survives_errors(settings, repo, tmp_path, monkeypatch):
    sid, _ = doc_submission(repo, tmp_path, prepared=True)
    jid = repo.enqueue(sid, Stage.evidence, "standard")
    real = Repo.heartbeat
    attempts = []

    def flaky(self, job_id):
        attempts.append(job_id)
        if len(attempts) == 1:
            raise RuntimeError("database is locked")
        real(self, job_id)

    monkeypatch.setattr(Repo, "heartbeat", flaky)
    inner = FakeBackend(responder)

    class Slow:
        model_id = "fake"

        def generate(self, reqs):
            deadline = time.monotonic() + 5
            while repo.get_job(jid).status == JobStatus.running and len(attempts) < 3 \
                    and time.monotonic() < deadline:
                time.sleep(0.02)
            return inner.generate(reqs)

        def close(self):
            pass

    code = run_session(None, "standard", settings=settings, guard=StubGuard(),
                       backend_factory=lambda s, k: Slow(), sleep=lambda s: None,
                       heartbeat_sec=0.02)
    assert code == 0 and len(attempts) >= 3  # kept beating after the first error


def test_the_worker_shows_it_is_alive_while_it_runs(settings, tmp_path):
    assert files.worker_beat_at(settings) is None
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    first = files.worker_beat_at(settings)
    assert first is not None
    path = tmp_path / "beat"
    with worker_main.WorkerBeat(path, interval=0.05):
        assert path.is_file()
        start = path.stat().st_mtime_ns
        deadline = time.monotonic() + 5
        while path.stat().st_mtime_ns == start and time.monotonic() < deadline:
            time.sleep(0.02)
        assert path.stat().st_mtime_ns > start             # touched again by the thread


def test_detect_session_comes_first_with_the_detect_model_budget(settings):
    order = worker_main._session_order(settings)
    assert order[0] == "detect" and order.count("detect") == 1
    assert worker_main._budget_gb(settings, "detect") == settings.pfb3x3_budget_gb["standard"]
    high = settings.model_copy(update={"detect_model_key": "high"})
    assert worker_main._budget_gb(high, "detect") == high.pfb3x3_budget_gb["high"]
