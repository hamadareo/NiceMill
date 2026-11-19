"""Inference session: runs in a child process; VRAM is returned by ending the process.

Session kinds (by model key):
- standard / high / whisper: the paper engine (a backend in this process, or whisper per call);
- track / track:cpu: the tracking engine's person tracking, as a subprocess on the GPU or,
  with track_cpu_fallback, after the GPU stayed full for track_cpu_after_sec, on the CPU (no
  GPU polling there);
- pfb3x3:<key>: the tracking engine's vLLM server (a subprocess at budget/total utilisation)
  plus engine subprocesses per job; the server is stopped when the session ends. It claims
  every job of its key: `behavior`, `names` (the roster cues after track), `contribution` and
  `pfb` (text-only requests to the same server, concurrent), so they all follow the same GPU
  budget and yield rules. Its jobs run side by side (_pool_job_loop): up to
  settings.served_parallel_jobs at once, of which at most settings.served_engine_jobs are
  engine commands (behavior / names), so a work's names run beside its behavior and its
  contribution starts as soon as its behavior has ended. Given the worker's shared server
  (worker.shared_server, `server=`) it uses that one instead, never stops it and ends as soon
  as its queue is empty (the worker keeps the server warm);
- linecall: 判定支援's linecall jobs, as track: one engine subprocess per job on the GPU, no
  server, no CPU fallback (exit 3 -> waiting_gpu);
- names:cpu: names jobs whose wanted cues need no server (team colours / initial places):
  engine subprocesses on the CPU, no server and no GPU polling;
- detect / detect:cpu: the content check (probe on the CPU per camera, then the scene verdict
  through the engine's server for detect_model_key when the session has one). `detect` starts
  that server and runs CPU-only if it cannot; the worker spawns `detect:cpu` (no server) at once
  when the GPU has no room. Both end as soon as no detect job is left. Given the shared
  server, `detect` uses it and leaves it running.
"""
import dataclasses
import logging
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings, load_settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.engine import (EngineError, EngineNeedsGpu, EngineServerDown, EngineYield,
                            Pfb3x3Engine, ServerHandle, engine_model_for)
from pfb_app.inference.backend import InferenceBackend
from pfb_app.inference.gpu import GpuGuard, GpuQueryError, query_nvidia_smi
from pfb_app.inference.served import ServerFailed
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import (DETECT_KEY, LINECALL_KEY, NAMES_CPU_KEY, PFB3X3_PREFIX,
                                     TRACK_KEY, WHISPER_KEY, SessionMustEnd, execute_job,
                                     is_cuda_oom)
from pfb_app.pipeline.tracking import GpuUnavailable
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ops import offline

log = logging.getLogger(__name__)

EXIT_IDLE = 0
EXIT_YIELD = 10
EXIT_OOM = 11
EXIT_LOAD_FAILED = 12
EXIT_NEEDS_GPU = 13  # the engine found too little free GPU memory; jobs are waiting_gpu
EXIT_SERVER_DOWN = 14  # the session's vLLM server failed (dead, unreachable, 5xx); jobs requeued
SERVER_CONNECT_MESSAGE = "解析サーバーに接続できませんでした。しばらくしてから再実行してください。"
LOAD_FAILED_ACTION = "model_load_failed"
TRACK_CPU_SESSION = f"{TRACK_KEY}:cpu"  # session name only: it claims TRACK_KEY jobs
DETECT_CPU_SESSION = f"{DETECT_KEY}:cpu"  # session name only: it claims DETECT_KEY jobs
# session name -> the key it claims (names:cpu is itself a job key: no GPU, no server)
_CPU_SESSIONS = {TRACK_CPU_SESSION: TRACK_KEY, DETECT_CPU_SESSION: DETECT_KEY,
                 NAMES_CPU_KEY: NAMES_CPU_KEY}
_ENGINE_SESSIONS = (TRACK_KEY, TRACK_CPU_SESSION, DETECT_KEY, DETECT_CPU_SESSION, NAMES_CPU_KEY,
                    LINECALL_KEY)

IDLE_POLL_SEC = 5
HEARTBEAT_SEC = 30
# Stages whose jobs a served session runs side by side: each pfb job is one subject's own
# request to the server, independent of the others (no follow-up stage is queued).
PARALLEL_STAGES = frozenset({Stage.pfb, Stage.plain})
PARALLEL_JOBS = 8
# Stages whose jobs run an engine command against the server (frame decoding on the CPU /
# NVDEC plus many concurrent VLM requests of their own): settings.served_engine_jobs at once.
ENGINE_STAGES = frozenset({Stage.behavior, Stage.names})
POOL_POLL_SEC = 1.0   # a pool with running jobs looks for newly queued ones this often


def setup_logging(settings: Settings | None = None, name: str = "worker") -> None:
    """INFO to stdout and, given the settings, to the rotating <log_dir>/<name>.log
    (ops.logs: worker.log for the worker, session.log for its sessions)."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s")
    if settings is not None:
        from pfb_app.ops import logs   # noqa: PLC0415 - keeps this module's imports light
        logs.setup(settings, name, console=False)


def _gpu_backend(settings: Settings, model_key: str) -> InferenceBackend:
    from pfb_app.inference.factory import make_backend
    return make_backend(settings, model_key, query_nvidia_smi().total_gb)


class _Heartbeat:
    """Refresh heartbeat_at of the current job every `interval` seconds from a thread.

    The thread uses its own connection: a sqlite3 connection must not be shared
    across threads, and long batched inference calls block the main thread.
    """

    def __init__(self, db_path: Path, interval: float):
        self._db_path = db_path
        self._interval = interval
        self._stop = threading.Event()
        self.job_id: int | None = None
        self.more_ids: tuple[int, ...] = ()   # jobs run alongside job_id (parallel pfb)
        self._thread = threading.Thread(target=self._run, name="pfb-heartbeat", daemon=True)

    def _run(self) -> None:
        conn = connect(self._db_path)
        try:
            repo = Repo(conn)
            while not self._stop.wait(self._interval):
                ids = [] if self.job_id is None else [self.job_id]
                for job_id in [*ids, *self.more_ids]:
                    try:
                        repo.heartbeat(job_id)
                    except Exception:  # e.g. database locked: keep beating on the next tick
                        log.exception("ハートビートの更新に失敗しました job=%s", job_id)
        finally:
            conn.close()

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_session(settings_path: str | None, model_key: str, *, settings: Settings | None = None,
                guard: GpuGuard | None = None,
                backend_factory: Callable[[Settings, str], InferenceBackend] | None = None,
                engine_factory: Callable[[Settings], Pfb3x3Engine] | None = None,
                sleep: Callable[[float], None] = time.sleep,
                clock: Callable[[], float] = time.monotonic,
                heartbeat_sec: float = HEARTBEAT_SEC,
                server: ServerHandle | dict | None = None) -> int:
    """Process jobs of `model_key` until idle (0), asked to yield (10) or out of memory (11).

    `settings_path` None means defaults. `server`: the worker's shared, already running vLLM
    server (worker.shared_server) for a `detect` or `pfb3x3:<key>` session: the session uses
    it and neither starts nor stops a server of its own. The other keyword arguments are for
    tests. Nothing GPU-related is set up before the backend is built.
    """
    if isinstance(server, dict):
        server = ServerHandle(**server)
    if model_key in _ENGINE_SESSIONS or model_key.startswith(PFB3X3_PREFIX):
        return _run_engine_session(settings_path, model_key, settings=settings, guard=guard,
                                   engine_factory=engine_factory or Pfb3x3Engine, sleep=sleep,
                                   clock=clock, heartbeat_sec=heartbeat_sec, shared=server)
    if settings is None:
        settings = load_settings(Path(settings_path) if settings_path else None)
    guard = guard or GpuGuard(settings)

    def yield_check() -> bool:
        try:
            return guard.should_yield()
        except GpuQueryError as e:  # cannot see the GPU -> assume others need it
            log.warning("GPUの状態を確認できないため譲ります: %s", e)
            return True

    db_path = files.db_path(settings)
    conn = connect(db_path)
    try:
        init_schema(conn)
        repo = Repo(conn, settings)
        backend = None
        if model_key != WHISPER_KEY:  # whisper is loaded per call by preprocess.transcribe
            log.info("モデルを読み込みます model_key=%s", model_key)
            try:
                backend = (backend_factory or _gpu_backend)(settings, model_key)
            except Exception as e:
                if is_cuda_oom(e):
                    log.error("モデル読み込み中にCUDAのメモリ不足 (OOM): %s", e)
                    return EXIT_OOM
                log.exception("モデルを読み込めませんでした model_key=%s", model_key)
                repo.audit(LOAD_FAILED_ACTION, model_key, {"error": f"{type(e).__name__}: {e}"})
                return EXIT_LOAD_FAILED
        ctx = StageContext(settings, repo, backend, yield_check)
        with _Heartbeat(db_path, heartbeat_sec) as hb:
            return _job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock)
    finally:
        conn.close()


def _run_engine_session(settings_path, session_key, *, settings, guard, engine_factory, sleep,
                        clock, heartbeat_sec, shared=None) -> int:
    """A tracking-engine session: `track`, `track:cpu`, `linecall`, `detect`, `detect:cpu`,
    `names:cpu` or `pfb3x3:<key>`."""
    if settings is None:
        settings = load_settings(Path(settings_path) if settings_path else None)
    guard = guard or GpuGuard(settings)
    on_cpu = session_key in _CPU_SESSIONS
    model_key = _CPU_SESSIONS.get(session_key, session_key)

    def yield_check() -> bool:
        if on_cpu:  # not on the GPU: nothing to give back
            return False
        try:
            return guard.should_yield()
        except GpuQueryError as e:
            log.warning("GPUの状態を確認できないため譲ります: %s", e)
            return True

    db_path = files.db_path(settings)
    conn = connect(db_path)
    try:
        init_schema(conn)
        repo = Repo(conn, settings)
        engine = engine_factory(settings)
        try:   # an engine command a killed session left running (e.g. the tracker on the GPU)
            # linecall has a record of its own: it runs beside the other sessions
            (engine.stop_stale_linecall if model_key == LINECALL_KEY else engine.stop_stale_call)()
        except Exception:
            log.exception("残っていたエンジンのプロセスを停止できませんでした")
        if model_key in (TRACK_KEY, NAMES_CPU_KEY, LINECALL_KEY):   # no server: ctx.server None
            ctx = StageContext(settings, repo, None, yield_check, engine=engine,
                               device="cpu" if on_cpu else "cuda")
            log.info("エンジンのセッションを開始します model_key=%s device=%s", model_key,
                     ctx.device)
            # idle_sec 0: nothing is kept loaded between engine calls, so waiting for more
            # jobs here would only hold back the next session (e.g. behavior after track)
            with _Heartbeat(db_path, heartbeat_sec) as hb:
                return _job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock,
                                 idle_sec=0)
        if model_key == DETECT_KEY:
            return _detect_session(settings, repo, db_path, guard, engine, on_cpu, sleep, clock,
                                   heartbeat_sec, None if on_cpu else shared)
        return _pfb3x3_session(settings, repo, db_path, model_key, guard, yield_check, engine,
                               sleep, clock, heartbeat_sec, shared)
    finally:
        conn.close()


def _pfb3x3_session(settings, repo, db_path, model_key, guard, yield_check, engine, sleep, clock,
                    heartbeat_sec, shared=None) -> int:
    """behavior, names, contribution and pfb of the tracking engine through one vLLM server:
    the worker's shared one (`shared`: used as it is, never stopped here; the session ends as
    soon as its queue is empty, the worker keeps the server warm), or one started for this
    session and stopped when it ends (session_idle_sec)."""
    if shared is not None:
        if not engine.server_alive(shared):
            log.warning("共有のPFB3x3サーバーが動いていないため、セッションを終了します pid=%s",
                        shared.pid)
            return EXIT_SERVER_DOWN
        log.info("共有のPFB3x3サーバーを使います model=%s pid=%s", shared.model, shared.pid)
        return _served_job_loop(settings, repo, db_path, model_key, yield_check, engine, shared,
                                sleep, clock, heartbeat_sec, idle_sec=0)
    base_key = model_key[len(PFB3X3_PREFIX):]
    try:
        model = engine_model_for(base_key)
        budget = settings.pfb3x3_budget_gb[base_key]
        util = guard.utilization_for(budget)
        engine.stop_stale_server()  # a leftover server of ours would make serve.sh refuse
        log.info("PFB3x3のサーバーを起動します model=%s 予算=%gGB 使用率=%g", model, budget, util)
        server = engine.start_server(model, util, on_tick=yield_check)
    except EngineNeedsGpu as e:
        reason = _first_line(e)
        n = repo.mark_waiting_gpu(model_key, reason)
        log.info("GPU待ち model_key=%s jobs=%d: %s", model_key, n, reason)
        return EXIT_NEEDS_GPU
    except EngineYield:
        log.info("サーバー起動中にGPUの空きが少なくなったため譲ります model_key=%s", model_key)
        return EXIT_YIELD
    except GpuQueryError as e:
        log.warning("GPUの状態を確認できないため譲ります: %s", e)
        return EXIT_YIELD
    except (EngineError, KeyError, ValueError) as e:
        log.exception("PFB3x3のサーバーを起動できませんでした model_key=%s", model_key)
        repo.audit(LOAD_FAILED_ACTION, model_key, {"error": f"{type(e).__name__}: {e}"})
        return EXIT_LOAD_FAILED
    try:
        return _served_job_loop(settings, repo, db_path, model_key, yield_check, engine, server,
                                sleep, clock, heartbeat_sec)
    finally:
        _stop_server(engine, server)


def _served_job_loop(settings, repo, db_path, model_key, yield_check, engine, server, sleep,
                     clock, heartbeat_sec, idle_sec: float | None = None) -> int:
    """The pfb3x3 session's jobs against `server`; contribution and pfb ask it through a
    text-only chat backend (concurrent requests), and pfb jobs run PARALLEL_JOBS at a time."""
    ctx = StageContext(settings, repo, _ServerGuard(engine.chat_backend(server)), yield_check,
                       engine=engine, server=server)
    with _Heartbeat(db_path, heartbeat_sec) as hb:
        if settings.served_parallel_jobs <= 1:   # one job at a time (pfb still in parallel)
            return _job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock,
                             idle_sec=idle_sec, parallel=PARALLEL_STAGES)
        return _pool_job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock,
                              idle_sec=idle_sec, max_jobs=settings.served_parallel_jobs,
                              max_engine_jobs=max(1, settings.served_engine_jobs))


class _ServerGuard:
    """The served session's chat backend: a failure of the server itself (ServerFailed:
    unreachable, timed out, 5xx such as vLLM out of CUDA memory) is EngineServerDown, so the
    job is requeued (up to runner.SERVER_DOWN_LIMIT) and the session ends, instead of every
    queued contribution / pfb failing against a broken server."""

    def __init__(self, inner):
        self._inner = inner
        self.model_id = inner.model_id
        self.cache_variant = getattr(inner, "cache_variant", None)

    def generate(self, reqs):
        try:
            return self._inner.generate(reqs)
        except ServerFailed as e:
            raise EngineServerDown(f"{SERVER_CONNECT_MESSAGE}\n{e}") from e

    def close(self) -> None:
        self._inner.close()


def _detect_session(settings, repo, db_path, guard, engine, on_cpu, sleep, clock,
                    heartbeat_sec, shared=None) -> int:
    """`detect`: the engine's server for engine_model_for(detect_model_key) at budget/total,
    stopped in finally. If it cannot start (no room, an engine error, a yield while starting),
    the claimed jobs run CPU-only. `detect:cpu`: ctx.server is None from the start.

    Either way the session claims only DETECT_KEY jobs and ends as soon as none is left (the
    worker's CPU jobs wait while a session runs). It gives the GPU back like the pfb3x3 session:
    a scene call is stopped when should_yield says so, and after each job the server is stopped
    when the GPU is needed (or the server died). The check itself never waits for the GPU: the
    job finishes without a scene verdict and the remaining jobs run CPU-only.

    `shared`: the worker's running server is used instead of starting one, and is left
    running at the end (the GPU yield still stops it, as it would stop our own)."""
    if shared is not None and not engine.server_alive(shared):
        log.warning("共有のPFB3x3サーバーが動いていないため、内容確認はCPUだけで行います")
        shared = None
        on_cpu = True
    own = shared is None
    server = shared if shared is not None else (
        None if on_cpu else _start_detect_server(settings, guard, engine))

    def gpu_wanted() -> bool:   # the yield check, only while our server holds GPU memory
        if ctx.server is None:
            return False
        try:
            return guard.should_yield()
        except GpuQueryError as e:
            log.warning("GPUの状態を確認できないため譲ります: %s", e)
            return True

    def release_server() -> None:
        handle = ctx.server
        if handle is None:
            return
        log.info("GPUを譲るためPFB3x3のサーバーを停止します。残りの内容確認はCPUだけで行います")
        _stop_server(engine, handle)
        ctx.server = None   # only now: a signal during the stop still reaches the finally

    ctx = StageContext(settings, repo, None, gpu_wanted, engine=engine, server=server,
                       release_server=release_server)

    def after_job() -> None:
        handle = ctx.server
        if handle is not None and (gpu_wanted() or not engine.server_alive(handle)):
            release_server()

    log.info("動画の内容確認のセッションを開始します server=%s", "あり" if server else "なし（CPUのみ）")
    try:
        with _Heartbeat(db_path, heartbeat_sec) as hb:
            return _job_loop(repo, ctx, DETECT_KEY, hb, gpu_wanted, sleep, clock, idle_sec=0,
                             after_job=after_job)
    finally:
        if own and ctx.server is not None:
            _stop_server(engine, ctx.server)


def _start_detect_server(settings, guard, engine):
    """The scene verdict's server, or None (the session then runs CPU-only)."""
    def yield_check() -> bool:
        try:
            return guard.should_yield()
        except GpuQueryError as e:
            log.warning("GPUの状態を確認できないため譲ります: %s", e)
            return True

    try:
        model = engine_model_for(settings.detect_model_key)
        budget = settings.pfb3x3_budget_gb[settings.detect_model_key]
        util = guard.utilization_for(budget)
        engine.stop_stale_server()  # a leftover server of ours would make serve.sh refuse
        log.info("内容確認用にPFB3x3のサーバーを起動します model=%s 予算=%gGB 使用率=%g",
                 model, budget, util)
        return engine.start_server(model, util, on_tick=yield_check,
                                   timeout_sec=settings.detect_timeout_sec)
    except (EngineNeedsGpu, EngineYield, GpuQueryError) as e:
        log.info("GPUに空きがないため、内容確認はCPUだけで行います: %s", _first_line(e))
    except Exception:   # an engine error, an unknown model key: still check on the CPU
        log.exception("内容確認用のサーバーを起動できませんでした。CPUだけで行います")
    return None


def _stop_server(engine: Pfb3x3Engine, server) -> None:
    """Stop our server; if that fails, stop whatever serve.pid of ours names."""
    try:
        engine.stop_server(server)
        log.info("PFB3x3のサーバーを停止しました pid=%s", server.pid)
        return
    except Exception:
        log.exception("PFB3x3のサーバーを停止できませんでした pid=%s", server.pid)
    try:
        engine.stop_stale_server()
    except Exception:
        log.exception("PFB3x3の残ったサーバーを停止できませんでした")


def _first_line(e: BaseException) -> str:
    lines = [line.strip() for line in str(e).splitlines() if line.strip()]
    return lines[0] if lines else type(e).__name__


def _detect_pending(repo, model_key: str) -> bool:
    """A detect job is pending and this is not the detect session: the content check never
    waits behind another work's session, busy or idle (the worker starts detect first)."""
    if model_key in (DETECT_KEY, LINECALL_KEY) or DETECT_KEY not in repo.pending_model_keys():
        return False   # linecall runs beside the other sessions: it never holds detect up
    log.info("動画の内容確認を先に行うため、セッションを終了します model_key=%s", model_key)
    return True


def _job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock, *,
              idle_sec: float | None = None, after_job: Callable[[], None] | None = None,
              parallel: frozenset = frozenset()) -> int:
    """Claim and run `model_key` jobs until idle for idle_sec (default session_idle_sec),
    asked to yield, or out of memory. after_job runs after each job that did not end it.
    Every session but detect also yields, after a job and at each idle poll, while a detect
    job is pending. A claimed job of a stage in `parallel` is run together with up to
    PARALLEL_JOBS - 1 more queued jobs of that stage and key (_run_parallel)."""
    idle_limit = ctx.settings.session_idle_sec if idle_sec is None else idle_sec
    idle_since = clock()
    while True:
        job = repo.claim_next_job({model_key})
        if job is None:
            if _detect_pending(repo, model_key):
                return EXIT_YIELD
            idle = clock() - idle_since
            if idle >= idle_limit:
                log.info("キューが空のためセッションを終了します model_key=%s", model_key)
                return EXIT_IDLE
            if yield_check():
                log.info("GPUの空きが少ないため、待機中のセッションを終了します")
                return EXIT_YIELD
            sleep(min(IDLE_POLL_SEC, idle_limit - idle))
            continue
        hb.job_id = job.id
        try:
            if job.stage in parallel:
                _run_parallel(repo, ctx, model_key, job, hb)
            else:
                execute_job(job, ctx)
        except GpuUnavailable as e:  # tracking engine exit 3: the job is already requeued
            n = repo.mark_waiting_gpu(model_key, e.reason)
            log.info("GPU待ち model_key=%s jobs=%d: %s", model_key, n, e.reason)
            return EXIT_NEEDS_GPU
        except EngineServerDown as e:   # the job is requeued; the next session starts anew
            log.warning("PFB3x3のサーバーが止まったため、セッションを終了します"
                        " model_key=%s: %s", model_key, e)
            return EXIT_SERVER_DOWN
        except YieldRequested:
            log.info("GPUの空きが少ないため譲ります（セッション終了） model_key=%s", model_key)
            return EXIT_YIELD
        except SessionMustEnd:
            return EXIT_OOM
        finally:
            hb.job_id = None
            hb.more_ids = ()
        if after_job is not None:
            after_job()
        if _detect_pending(repo, model_key):
            return EXIT_YIELD
        idle_since = clock()


_END_ORDER = (SessionMustEnd, GpuUnavailable, EngineServerDown, YieldRequested)


def _pool_job_loop(repo, ctx, model_key, hb, yield_check, sleep, clock, *,
                   idle_sec: float | None = None, max_jobs: int = PARALLEL_JOBS,
                   max_engine_jobs: int = 2) -> int:
    """The served session's jobs side by side: queued jobs of `model_key` are claimed (the
    oldest first) while fewer than `max_jobs` run, an ENGINE_STAGES job only while fewer than
    `max_engine_jobs` of those run (the claim then takes the oldest job of another stage). Each
    runs in its own thread with its own database connection (execute_job records its outcome).
    Jobs queued while others run (a contribution after its behavior, the pfb after it) are
    claimed as soon as they are queued.

    Ends as _job_loop does: idle for idle_sec (default session_idle_sec), asked to yield while
    idle, a pending detect job, or a job that must end the session (OOM, needs-GPU, a dead
    server, a yield). A pending detect ends it BETWEEN jobs, as _job_loop does: at most one
    claimed job still starts beside it and running jobs are never preempted; an idle pool with
    a waiting check hands over at once. From then on no job is claimed and the running ones
    end first; the exit
    code follows the first such exception in _END_ORDER, and any other exception (a crash) is
    re-raised, as _job_loop lets it through. A signal (SystemExit) in this thread asks the
    running jobs' engine commands to stop (their next tick yields) before it propagates."""
    idle_limit = ctx.settings.session_idle_sec if idle_sec is None else idle_sec
    idle_since = clock()
    db_path = files.db_path(ctx.settings)
    cancel = threading.Event()
    job_ctx = dataclasses.replace(ctx, yield_check=lambda: cancel.is_set() or ctx.yield_check())
    running: dict = {}   # future -> job
    errors: list[BaseException] = []
    detect_first = False

    def one(job):
        conn = connect(db_path)
        try:
            job_repo = Repo(conn, ctx.settings)
            execute_job(job, dataclasses.replace(job_ctx, repo=job_repo))
            return None
        except Exception as e:   # recorded by execute_job; decides how the session ends
            return e
        finally:
            conn.close()

    pool = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="pfb-job")
    try:
        while True:
            finished = [f for f in running if f.done()]
            for f in finished:
                running.pop(f)
                if (e := f.result()) is not None:
                    errors.append(e)
            if finished:
                idle_since = clock()
            if not errors and not detect_first and finished and _detect_pending(repo, model_key):
                detect_first = True   # the content check first: no new job from here on
            ending = bool(errors) or detect_first
            claimed_here = False   # jobs claimed in this pass (the check below stops a second)
            while not ending and len(running) < max_jobs:
                if claimed_here and _detect_pending(repo, model_key):
                    detect_first = True   # one job starts beside a waiting check, never two
                    break
                engines = sum(1 for j in running.values() if j.stage in ENGINE_STAGES)
                job = repo.claim_next_job(
                    {model_key}, skip_stages=ENGINE_STAGES if engines >= max_engine_jobs else None,
                    after_upstream=True)
                if job is None:
                    break
                if _drop_if_deleting(repo, job):
                    continue
                claimed_here = True
                if running:
                    log.info("%s を並行して実行します job=%s（実行中 %d件）", job.stage.value,
                             job.id, len(running))
                running[pool.submit(one, job)] = job
            hb.job_id = None
            hb.more_ids = tuple(j.id for j in running.values())
            if running:
                wait(list(running), timeout=POOL_POLL_SEC, return_when=FIRST_COMPLETED)
                continue
            if ending:
                break
            if _detect_pending(repo, model_key):
                detect_first = True   # nothing running: let the worker run the check first
                break
            idle = clock() - idle_since
            if idle >= idle_limit:
                log.info("キューが空のためセッションを終了します model_key=%s", model_key)
                return EXIT_IDLE
            if yield_check():
                log.info("GPUの空きが少ないため、待機中のセッションを終了します")
                return EXIT_YIELD
            sleep(min(IDLE_POLL_SEC, idle_limit - idle))
    except BaseException:
        cancel.set()   # the running engine commands stop at their next tick
        raise
    finally:
        pool.shutdown(wait=True)
        hb.more_ids = ()
    for kind in _END_ORDER:
        for e in errors:
            if isinstance(e, kind):
                return _end_code(repo, model_key, e)
    if errors:
        raise errors[0]
    return EXIT_YIELD   # a detect job is pending: the worker starts it first


def _end_code(repo, model_key, e: BaseException) -> int:
    """The session's exit code for an exception that ends it (as _job_loop maps them)."""
    if isinstance(e, GpuUnavailable):
        n = repo.mark_waiting_gpu(model_key, e.reason)
        log.info("GPU待ち model_key=%s jobs=%d: %s", model_key, n, e.reason)
        return EXIT_NEEDS_GPU
    if isinstance(e, EngineServerDown):
        log.warning("PFB3x3のサーバーが止まったため、セッションを終了します model_key=%s: %s",
                    model_key, e)
        return EXIT_SERVER_DOWN
    if isinstance(e, SessionMustEnd):
        return EXIT_OOM
    log.info("GPUの空きが少ないため譲ります（セッション終了） model_key=%s", model_key)
    return EXIT_YIELD


def _run_parallel(repo, ctx, model_key, first, hb) -> None:
    """Run `first` and up to PARALLEL_JOBS - 1 more claimed jobs of its stage at once, each in
    its own thread with its own database connection (sqlite connections stay in their
    thread). Every job records its own outcome (execute_job); once all have ended, the
    exception that must end the session is re-raised (OOM first, then needs-GPU, a dead
    server, a yield)."""
    jobs = [first]
    while len(jobs) < PARALLEL_JOBS:
        job = repo.claim_next_job({model_key}, stage=first.stage)   # never skips ahead
        if job is None:
            break
        if not _drop_if_deleting(repo, job):
            jobs.append(job)
    hb.more_ids = tuple(j.id for j in jobs[1:])
    if len(jobs) == 1:
        execute_job(first, ctx)
        return
    log.info("%s を%d件並行して実行します jobs=%s", first.stage.value, len(jobs),
             [j.id for j in jobs])
    db_path = files.db_path(ctx.settings)

    def one(job):
        conn = connect(db_path)
        try:
            job_repo = Repo(conn, ctx.settings)
            if job is not first and _drop_if_deleting(job_repo, job):   # asked meanwhile
                return None
            execute_job(job, dataclasses.replace(ctx, repo=job_repo))
            return None
        except Exception as e:   # recorded by execute_job; decides how the session ends
            return e
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="pfb-job") as pool:
        errors = [e for e in pool.map(one, jobs) if e is not None]
    for kind in _END_ORDER:
        for e in errors:
            if isinstance(e, kind):
                raise e
    if errors:
        raise errors[0]


def _drop_if_deleting(repo, job) -> bool:
    """A claimed job of a work that is to be deleted is not run: its row is removed (the
    deletion then goes ahead without waiting for it). True when it was dropped."""
    from pfb_app.services import deletion   # noqa: PLC0415 - services import the pipeline
    try:
        requested = deletion.is_requested(repo.get_submission(job.submission_id)["meta"])
    except KeyError:   # the work is gone already
        requested = True
    if not requested:
        return False
    repo.finish_job(job.id, JobStatus.queued)
    repo.cancel_pending_jobs(job.submission_id, job.stage, job.camera, job.subject_id)
    log.info("削除を頼まれた作業のジョブは実行しません job=%s", job.id)
    return True


def exit_on_signals() -> None:
    """SIGTERM / SIGHUP (scripts/stop.sh ends the tmux session) raise SystemExit, so the
    session's finally blocks run: the engine command's process group and the PFB3x3 vLLM
    server are stopped instead of being orphaned (they run in their own process groups and
    would not get the signal). Only the first signal ends the session: a later one (the
    linecall child's PR_SET_PDEATHSIG and its getppid watch both fire, or a stop sends SIGTERM
    and the tmux end SIGHUP) is logged and ignored, so it never cuts those finally blocks
    short and orphans the engine command."""
    ending = []

    def _exit(signum, _frame):
        name = signal.Signals(signum).name
        if ending:
            log.warning("シグナル %s を受けましたが、終了の処理中のため続けます", name)
            return
        ending.append(signum)
        log.warning("シグナル %s を受けたため終了します", name)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit)


def session_entry(settings_path: str | None, model_key: str, server: dict | None = None) -> None:
    """multiprocessing target: the exit code is the process exit status. `server`: the
    worker's shared server (ServerHandle fields) for a detect / pfb3x3 session."""
    try:
        log_settings = load_settings(Path(settings_path) if settings_path else None)
    except Exception:   # noqa: BLE001 - run_session reports a bad settings file itself
        log_settings = None
    setup_logging(log_settings, "session")
    offline.apply()   # as the worker: model libraries offline, no usage reports
    exit_on_signals()
    sys.exit(run_session(settings_path, model_key, server=server))
