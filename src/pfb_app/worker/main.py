"""Resident worker: CPU jobs in this process, GPU jobs in one child session at a time.

Usage: uv run python -m pfb_app.worker.main [--settings PATH] [--once]

This process never touches CUDA. It only reads nvidia-smi through GpuGuard and never
stops or signals any other process; the one exception is pfb_app's own PFB3x3 engine: its
vLLM server (serve.pid in <data_dir>/pfb3x3_runtime/, stopped by the engine's stop_server.sh)
and its recorded engine command group, after a pfb3x3/track/linecall/detect session died without
stopping them, and once at worker start (left by a previous worker). Both are identity-checked.

Without --once the worker also keeps the shared model's server warm itself
(worker.shared_server: started through serve.sh, stopped through stop_server.sh, both
pid-verified), and hands it to the detect / pfb3x3 sessions of that model, and runs 判定支援's
linecall jobs beside the one session, in a supervised child of their own
(worker.linecall_runner), so an hour-long linecall never holds the other works up.
"""
import argparse
import functools
import logging
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings, load_settings
from pfb_app.domain.models import JobStatus
from pfb_app.engine import Pfb3x3Engine
from pfb_app.inference.gpu import GpuGuard, GpuQueryError
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import (CPU_KEY, DETECT_KEY, LINECALL_KEY, NAMES_CPU_KEY,
                                     PFB3X3_PREFIX, TRACK_KEY, WHISPER_KEY, SessionMustEnd,
                                     execute_job, pfb3x3_key)
from pfb_app.ops import alerts, disk, offline
from pfb_app.ops.supervise import AlreadyRunning, RunningMarker, WorkerLock
from pfb_app.services import auto_retry, chunked_upload, deletion, erasure, explainer, retention
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker.session import (DETECT_CPU_SESSION, EXIT_IDLE, EXIT_LOAD_FAILED,
                                    EXIT_NEEDS_GPU, EXIT_OOM, EXIT_SERVER_DOWN, EXIT_YIELD,
                                    LOAD_FAILED_ACTION,
                                    TRACK_CPU_SESSION, exit_on_signals, session_entry,
                                    setup_logging)
from pfb_app.worker import linecall_runner
from pfb_app.worker.linecall_runner import LinecallRunner
from pfb_app.worker.shared_server import SharedServer, StartFailure

log = logging.getLogger(__name__)

IDLE_POLL_SEC = 5
DISK_HOLD_SEC = 30     # while the disk is too full: how often to look again
EXIT_ALREADY_RUNNING = 75   # another worker holds <data_dir>/worker.lock (systemd: no restart)
MAX_LOAD_FAILURES = 3  # consecutive model-load failures before giving up on a key
BEAT_SEC = 30          # how often the worker touches <data_dir>/worker.beat
UPLOAD_SWEEP_SEC = 3600   # how often stale and unclaimed uploads are swept (chunked_upload)
# the free GPU memory the engine's linecall needs to start (pfb3x3.engine.gpu_device:
# PFB_GPU_RESERVE_MIB (default 32768) + LINECALL_NEED_MIB 6144 = 38912 MiB; GpuGuard counts GB
# of 1024 MiB); with settings.engine_gpu_reserve_gb: linecall_engine_free_gb
LINECALL_NEED_MIB = 6144
LINECALL_ENGINE_FREE_GB = (32768 + LINECALL_NEED_MIB) / 1024


def linecall_engine_free_gb(settings: Settings) -> float:
    """The free GB the engine's linecall checks for with this install's reserve."""
    from pfb_app.engine.pfb3x3 import engine_reserve_mib   # noqa: PLC0415
    return (engine_reserve_mib(settings) + LINECALL_NEED_MIB) / 1024


class WorkerBeat:
    """Touch the worker's beat file now and then every `interval` seconds from a thread, for as
    long as the worker runs: also while it waits for a session or for the GPU, so the UI can
    tell a busy worker from one that is not running at all."""

    def __init__(self, path: Path, interval: float = BEAT_SEC):
        self.path = path
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pfb-worker-beat", daemon=True)

    def _touch(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()
        except OSError:
            log.exception("ワーカーの稼働記録を更新できませんでした: %s", self.path)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._touch()

    def __enter__(self) -> "WorkerBeat":
        self._touch()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


EXPLAINER_SWEEP_SEC = 60   # how often the explainer thread tidies the videos (explainer.sweep)


def _explainer_child(settings: Settings, job_id: int) -> None:
    """multiprocessing target: make one 解説動画 (services.explainer.run_job). SIGTERM (the
    worker stopping, or a hung render) leaves through the finally blocks: no part file stays."""
    import signal   # noqa: PLC0415
    import sys   # noqa: PLC0415
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    setup_logging(settings, "session")   # as the sessions: session.log
    conn = connect(files.db_path(settings))
    try:
        repo = Repo(conn, settings)
        job = repo.explainer_job(job_id)
        if job is not None and job["status"] == "running":
            explainer.run_job(repo, settings, job)
    finally:
        conn.close()


class ExplainerRunner:
    """解説動画 (services.explainer) beside the sessions: a thread with its own connection that
    makes one queued video at a time, each in a "spawn" child of its own (CPU only: OpenCV
    decoding, PyAV encoding with explainer_video.THREADS threads), and watches it: a child
    whose job has not beaten for `hung_sec` is stopped (terminate, then kill: our own child
    object, never a pid looked up) and its job fails with explainer.HUNG, so it can be made
    again; a child that died leaves its job failed. Every EXPLAINER_SWEEP_SEC it also tidies
    the videos (explainer.sweep), fails jobs nobody beats any more and starts the playable
    proxy a group needs (explainer.ensure_sources: one at a time, HDR tone-mapped). `stop` stops a video
    being made (not kept; the job says 途中で止まりました)."""

    def __init__(self, settings: Settings, poll: float = IDLE_POLL_SEC,
                 hung_sec: float | None = None, launch: Callable | None = None):
        self.settings = settings
        self._poll = poll
        self._hung = explainer.HUNG_SEC if hung_sec is None else hung_sec
        self._launch = launch or self._spawn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pfb-explainer", daemon=True)
        self._swept: float | None = None

    def _spawn(self, job: dict):
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(target=_explainer_child, args=(self.settings, job["id"]),
                           name=f"pfb-explainer-{job['id']}")
        proc.start()
        return proc

    def start(self) -> "ExplainerRunner":
        self._thread.start()
        return self

    def _tidy(self, repo: Repo) -> None:
        now = time.monotonic()
        if self._swept is not None and now - self._swept < EXPLAINER_SWEEP_SEC:
            return
        self._swept = now
        explainer.fail_stale(repo, self._hung)
        explainer.sweep(repo, self.settings)
        explainer.ensure_sources(repo, self.settings)   # playable proxies (HDR tone-mapped)

    def _run(self) -> None:
        conn = connect(files.db_path(self.settings))
        try:
            repo = Repo(conn, self.settings)
            while not self._stop.is_set():
                try:
                    self._tidy(repo)
                    job = repo.explainer_claim()
                    if job is not None:
                        self._supervise(repo, job)
                        continue
                except Exception:   # noqa: BLE001 - the thread keeps running
                    log.exception("解説動画の処理に失敗しました")
                self._stop.wait(self._poll)
        finally:
            conn.close()

    def _age(self, repo: Repo, job_id: int) -> float | None:
        from datetime import datetime, timezone   # noqa: PLC0415
        row = repo.explainer_job(job_id)
        if row is None or row["status"] != "running":
            return None
        try:
            beat = datetime.fromisoformat(row["heartbeat_at"] or row["started_at"])
        except (TypeError, ValueError):
            return None
        if beat.tzinfo is None:   # the repo writes UTC stamps; a bare one is read as UTC
            beat = beat.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - beat).total_seconds()

    def _end(self, proc) -> None:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)

    def _supervise(self, repo: Repo, job: dict) -> None:
        try:
            proc = self._launch(job)
        except Exception:   # noqa: BLE001 - no child started: the job fails now, not at HUNG_SEC
            repo.explainer_finish(job["id"], "failed", explainer.FAILED)
            raise
        try:
            self._watch(repo, job, proc)
        except BaseException:
            # an error in the watching itself (e.g. a locked database) must not leave the child
            # making a video beside the next one: stop it, then fail its job
            if proc.is_alive():
                self._end(proc)
            repo.explainer_finish(job["id"], "failed", explainer.FAILED)
            raise

    def _watch(self, repo: Repo, job: dict, proc) -> None:
        while proc.is_alive():
            proc.join(min(2.0, max(self._hung / 4, 0.05)))
            if self._stop.is_set():
                self._end(proc)
                repo.explainer_finish(job["id"], "failed", explainer.STOPPED)
                return
            age = self._age(repo, job["id"])
            if age is not None and age > self._hung:
                log.error("解説動画が%d秒進まないため止めます job=%s", int(age), job["id"])
                self._end(proc)
                repo.explainer_finish(job["id"], "failed", explainer.HUNG)
                return
        if repo.explainer_finish(job["id"], "failed", explainer.FAILED):   # it died
            log.error("解説動画の処理が途中で終わりました job=%s", job["id"])

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()           # the child being watched is stopped by _supervise
        if self._thread.is_alive():
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


def spawn_session(key: str, settings_path: str | None = None, *,
                  target: Callable = session_entry, args: tuple | None = None,
                  server: dict | None = None) -> int:
    """Run an inference session in a fresh "spawn" child, wait for it, return its exit code.
    `server`: the shared server's handle fields for a detect / pfb3x3 session."""
    ctx = multiprocessing.get_context("spawn")
    if args is None:
        args = (settings_path, key) if server is None else (settings_path, key, server)
    proc = ctx.Process(target=target, args=args, name=f"pfb-session-{key}")
    proc.start()
    proc.join()
    return proc.exitcode


def _budget_gb(settings: Settings, key: str) -> float:
    """The key's GPU budget; KeyError for a model the settings do not configure."""
    if key == NAMES_CPU_KEY:   # engine subprocesses on the CPU only
        return 0.0
    if key == DETECT_KEY:   # the scene verdict's served model
        return settings.pfb3x3_budget_gb[settings.detect_model_key]
    if key == WHISPER_KEY:
        return settings.whisper_budget_gb
    if key == TRACK_KEY:
        return settings.track_budget_gb
    if key == LINECALL_KEY:   # never less than the engine's own check plus 1 GB
        return max(settings.linecall_budget_gb,
                   linecall_engine_free_gb(settings) + 1 - settings.start_margin_gb)
    if key.startswith(PFB3X3_PREFIX):
        return settings.pfb3x3_budget_gb[key[len(PFB3X3_PREFIX):]]
    return settings.models[key].budget_gb


def _session_order(settings: Settings) -> list[str]:
    """detect (the coach is waiting for the content check), whisper, track, linecall (the
    coach pressed 判定材料を作る), names:cpu (no GPU), the paper models, then the pfb3x3
    sessions of configured models. One session at a time."""
    return [DETECT_KEY, WHISPER_KEY, TRACK_KEY, LINECALL_KEY, NAMES_CPU_KEY, *settings.models,
            *(pfb3x3_key(k) for k in settings.models if k in settings.pfb3x3_budget_gb)]


def unknown_key_reason(key: str) -> str:
    if key.startswith(PFB3X3_PREFIX):
        return (f"設定にないモデル（{key}）のため実行できません。設定の models と "
                "pfb3x3_budget_gb に、このモデルを加えてください。")
    return f"設定にないモデル（{key}）のため実行できません。設定の models を確認してください。"


def startup_cleanup(settings: Settings) -> None:
    """At worker start: stop what a previous worker's sessions may have left running (e.g.
    after scripts/stop.sh): the recorded engine command group and our vLLM server
    (<data_dir>/pfb3x3_runtime/serve.pid). Both are identity-checked before any signal; with
    nothing recorded nothing is run. A linecall child of a killed worker that still runs
    (linecall_runner.live_session) keeps its engine command: it stops it itself as it ends."""
    engine = Pfb3x3Engine(settings)
    orphan = linecall_runner.live_session(settings)
    if orphan is not None:   # it stops its own engine command on its way out (child_main)
        log.warning("前のワーカーの判定材料のセッション（pid %s）がまだ動いているため、"
                    "その処理は止めずに終わるのを待ちます", orphan)
    for what, stop in (("エンジンのプロセス", engine.stop_stale_call),
                       ("判定材料のプロセス",
                        engine.stop_stale_linecall if orphan is None else (lambda: None)),
                       ("PFB3x3のサーバー", engine.stop_stale_server)):
        try:
            stop()
        except Exception:
            log.exception("起動時に、残っていた%sを停止できませんでした", what)


def _cleanup_engine(settings: Settings, key: str, *, server: bool = True) -> None:
    """After a track / linecall / names:cpu / pfb3x3 session died: stop the engine command it left running (its
    recorded process group, e.g. the tracker on the GPU) and, for pfb3x3 / detect, its vLLM
    server (serve.pid in our runtime dir) unless `server` is False (the session had no server of
    its own: detect:cpu, or one on the worker's shared server, which the worker stops itself).
    Both are verified as ours before anything is signalled."""
    engine = Pfb3x3Engine(settings)
    if key == LINECALL_KEY:   # its own call record (it may run beside the other sessions)
        engine.stop_stale_linecall()
        return
    engine.stop_stale_call()
    if server and (key.startswith(PFB3X3_PREFIX) or key == DETECT_KEY):
        engine.stop_stale_server()


def _run_cpu_jobs(repo: Repo, settings: Settings) -> None:
    ctx = StageContext(settings, repo, None, lambda: False)
    while (job := repo.claim_next_job({CPU_KEY})) is not None:
        try:
            execute_job(job, ctx)
        except SessionMustEnd:  # not expected on the CPU; the job is already recorded
            log.error("CPUジョブでCUDAのメモリ不足が報告されました job=%s", job.id)


def _on_shared_failure(repo: Repo, shared: SharedServer, failure: StartFailure,
                       load_failures: dict[str, int]) -> None:
    """A start of the shared server ended without a server: no room -> its jobs wait for the
    GPU; an error counts as a load failure of its session key (as a session's own start)."""
    key = shared.session_key
    if failure.kind == "needs_gpu":
        n = repo.mark_waiting_gpu(key, failure.reason)
        log.info("GPU待ち model_key=%s jobs=%d: %s", key, n, failure.reason)
    elif failure.kind == "error":
        repo.audit(LOAD_FAILED_ACTION, key, {"error": failure.reason})
        _on_load_failure(repo, key, load_failures)


def _in_process(settings: Settings, key: str) -> bool:
    """The session loads a model into its own process (the paper engine / whisper)."""
    return key in settings.models or key == WHISPER_KEY


def _own_server(settings: Settings, key: str, shared: SharedServer) -> bool:
    """The session starts a vLLM server of its own (in the one runtime dir / port)."""
    return not shared.serves(key) and (
        key.startswith(PFB3X3_PREFIX) or (key == DETECT_KEY
                                          and settings.detect_model_key in settings.pfb3x3_budget_gb))


def _linecall_poll(repo: Repo, settings: Settings, linecall: LinecallRunner | None,
                   engine_cleanup: Callable[[str], None] | None = None) -> None:
    """Take the exit of a linecall child that has ended. A crash recovers only its own
    running job (as a heartbeat timeout: auto_retry may requeue it once) and stops the engine
    command it left (pid-verified). Never alerts: 判定支援 is a side stage."""
    if linecall is None:
        return
    code = linecall.poll()
    if code is None:
        return
    if code in (EXIT_IDLE, EXIT_YIELD, EXIT_NEEDS_GPU, EXIT_OOM, EXIT_LOAD_FAILED):
        log.info("判定材料のセッションが終了しました code=%s", code)
        return
    n = repo.recover_stale_jobs(0, model_key=LINECALL_KEY)
    log.error("判定材料のセッションが異常終了しました code=%s 回収したジョブ=%d", code, n)
    try:
        auto_retry.sweep(repo, settings)
    except Exception:   # noqa: BLE001 - the worker keeps running
        log.exception("止まったジョブの自動の再試行に失敗しました")
    try:
        if engine_cleanup is not None:
            engine_cleanup(LINECALL_KEY)
        else:
            linecall.cleanup()
    except Exception:   # noqa: BLE001
        log.exception("判定材料の残ったプロセスを停止できませんでした")


def _gpu_pass(repo: Repo, settings: Settings, guard, spawn, sleep,
              load_failures: dict[str, int], waiting_since: dict[str, float] | None = None,
              clock: Callable[[], float] = time.monotonic,
              engine_cleanup: Callable[[str], None] | None = None,
              shared: SharedServer | None = None,
              linecall: LinecallRunner | None = None) -> bool:
    """Start at most one session. Returns True if anything was pending.

    `waiting_since` remembers since when the track key could not start; after
    track_cpu_after_sec the track jobs run in a CPU session instead. The detect key never
    waits for room on the GPU: without it, its jobs run in a CPU-only session at once.

    `shared` (worker.shared_server): the keys it serves get its handle instead of starting a
    server of their own; while it starts, the content check waits for it (as it waited for
    its own server) up to detect_timeout_sec, other keys may run meanwhile.

    `linecall` (worker.linecall_runner): linecall jobs start in its child, without waiting
    for it, and the pass goes on to the other keys; while it runs, its RUN_GB is counted on
    top of every other GPU start. None (--once): linecall is a session like the others."""
    waiting_since = {} if waiting_since is None else waiting_since
    if linecall is not None and shared is not None:
        shared.extra_gb = linecall.gpu_gb   # a running linecall is a GPU consumer
    if shared is not None:
        failure = shared.maintain(repo)
        if failure is not None:
            _on_shared_failure(repo, shared, failure, load_failures)
        if shared.disabled:   # not bound to this machine only: every session as before
            shared = None
    pending = repo.pending_model_keys()
    order = _session_order(settings)
    unknown = pending - set(order) - {CPU_KEY}
    for key in sorted(unknown):   # shown on the status page instead of queueing silently
        if repo.count_jobs(key, JobStatus.queued):   # newly queued ones only (no log spam)
            n = repo.mark_waiting_gpu(key, unknown_key_reason(key))
            log.warning("設定にないmodel_keyのジョブは実行できません model_key=%s jobs=%d", key, n)
    keys = [k for k in order if k in pending]
    if linecall is not None and LINECALL_KEY in keys:
        keys.remove(LINECALL_KEY)
        if linecall.ready():   # first: it starts without taking the session's turn
            keys.insert(0, LINECALL_KEY)
        # else one at a time: the running child claims the queued ones itself
    if TRACK_KEY not in keys:
        waiting_since.pop(TRACK_KEY, None)
    if not keys:
        return False
    waiting_for_server = False
    for key in keys:
        use_shared = False
        if shared is not None and shared.serves(key):
            state, why = shared.request()
            if state == "ready":
                use_shared = True
            elif key == DETECT_KEY and state == "starting" \
                    and shared.starting_for() < settings.detect_timeout_sec:
                log.info("内容確認のためのサーバーの起動を待ちます")
                sleep(IDLE_POLL_SEC)   # nothing else starts meanwhile: the coach is waiting
                return True
            elif state == "starting" and key != DETECT_KEY:
                waiting_for_server = True
                continue
            elif key != DETECT_KEY:
                n = repo.mark_waiting_gpu(key, why)
                log.info("GPU待ち model_key=%s jobs=%d: %s", key, n, why)
                continue
            budget = None
            ok, reason = use_shared, why or "サーバーの起動が間に合いません"
        else:
            try:
                budget = _budget_gb(settings, key)
            except KeyError:
                if key != DETECT_KEY:
                    raise
                budget = None   # no served model for the scene verdict: detector only
            # while the shared server starts, vLLM has not taken its memory yet: count it
            extra = shared.held_gb if (shared is not None and shared.starting
                                      and budget is not None and key != NAMES_CPU_KEY) else 0.0
            if linecall is not None and budget is not None and key != NAMES_CPU_KEY:
                extra += linecall.gpu_gb()   # a running linecall: its pose model
            ok, reason = _can_start(guard, settings, key,
                                    None if budget is None else budget + extra)
            if not ok and shared is not None and key != NAMES_CPU_KEY and budget is not None \
                    and shared.holds_gpu() and not shared.wanted(repo):
                # no queued job needs it now: stop it, but only when that makes room
                freed = 0.0 if shared.starting else shared.held_gb
                if _can_start(guard, settings, key, budget - freed)[0]:
                    shared.stop(f"{key} のためにGPUを空けます")
                    ok, reason = _can_start(guard, settings, key, budget)
        session = key
        if ok and linecall is not None and key == LINECALL_KEY:
            linecall.start()   # beside the one session: the pass goes on to the other keys
            continue
        if ok:
            waiting_since.pop(key, None)
        elif key == DETECT_KEY:   # the content check never waits for the GPU
            session = DETECT_CPU_SESSION
            log.info("GPUに空きがないため、動画の内容確認をCPUだけで行います: %s", reason)
        elif key == TRACK_KEY and settings.track_cpu_fallback and (
                clock() - waiting_since.setdefault(key, clock()) >= settings.track_cpu_after_sec):
            waiting_since.pop(key)
            session = TRACK_CPU_SESSION
            log.info("GPUが%d秒以上空かないため、人物の追跡をCPUで実行します: %s",
                     settings.track_cpu_after_sec, reason)
        else:
            n = repo.mark_waiting_gpu(key, reason)
            log.info("GPU待ち model_key=%s jobs=%d: %s", key, n, reason)
            continue
        if use_shared:
            with shared.lease() as server:
                if server is None:   # stopped meanwhile (the GPU was wanted)
                    return True
                log.info("推論セッションを起動します model_key=%s 共有サーバー pid=%s", session,
                         server["pid"])
                code = spawn(session, server=server)
        else:
            if shared is not None and session != DETECT_CPU_SESSION \
                    and _own_server(settings, key, shared):
                shared.stop(f"{key} は別のサーバーを使うため")   # one serve.pid / port
            elif shared is not None and _in_process(settings, key):
                shared.stop(f"{key} はモデルを読み込むため")   # never both on the GPU
            log.info("推論セッションを起動します model_key=%s 予算=%sGB", session,
                     "-" if budget is None or session == DETECT_CPU_SESSION else f"{budget:g}")
            code = spawn(session)
        crashed = code not in (EXIT_IDLE, EXIT_YIELD, EXIT_LOAD_FAILED, EXIT_OOM, EXIT_NEEDS_GPU,
                               EXIT_SERVER_DOWN)
        if use_shared:
            shared.after_session(code, broken=crashed or code == EXIT_SERVER_DOWN,
                                 quiet_codes=(EXIT_IDLE, EXIT_LOAD_FAILED, EXIT_NEEDS_GPU))
        if code == EXIT_LOAD_FAILED:
            _on_load_failure(repo, key, load_failures)
        else:
            load_failures.pop(key, None)
        if code == EXIT_IDLE:
            log.info("セッション終了（キューが空） model_key=%s", key)
            return True
        if code == EXIT_YIELD:
            log.info("セッション終了（GPUを譲歩） model_key=%s", key)
        elif code == EXIT_SERVER_DOWN:
            log.warning("セッション終了（解析サーバーの異常） model_key=%s", key)
        elif code == EXIT_LOAD_FAILED:
            pass  # logged by _on_load_failure
        elif code == EXIT_OOM:
            log.warning("セッション終了（CUDAのメモリ不足） model_key=%s", key)
        elif code == EXIT_NEEDS_GPU:
            log.info("セッション終了（GPUの空きが足りないため待機） model_key=%s", key)
        else:
            # the session has exited: its running job is orphaned (a linecall child's is not)
            n = repo.recover_stale_jobs(0, exclude_key=None if linecall is None else LINECALL_KEY)
            log.error("セッションが異常終了しました model_key=%s code=%s 回収したジョブ=%d",
                      key, code, n)
            try:
                retried = auto_retry.sweep(repo, settings)   # started after the sleep below
            except Exception:   # noqa: BLE001 - the worker keeps running
                log.exception("止まったジョブの自動の再試行に失敗しました")
                retried = []
            if n and not retried and key != LINECALL_KEY:   # 判定支援: a side stage, quiet
                alerts.send(settings, f"session-crashed:{key}", "解析の処理が異常終了しました",
                            f"{key} の処理が異常終了しました（終了コード {code}）。"
                            "管理用の画面の「処理状況」で、止まった作業を確かめてください。",
                            "error")
            if key in (TRACK_KEY, LINECALL_KEY, DETECT_KEY, NAMES_CPU_KEY) \
                    or key.startswith(PFB3X3_PREFIX):
                own = shared is None or not (use_shared or session == DETECT_CPU_SESSION)
                try:
                    if engine_cleanup is not None:
                        engine_cleanup(key)
                    else:   # never the shared server (stopped above when it was used)
                        _cleanup_engine(settings, key, server=own)
                except Exception:
                    log.exception("PFB3x3の残ったプロセスを停止できませんでした")
        if code == EXIT_YIELD and shared is not None and _ended_for_detect(repo, guard):
            return True   # it made way for a content check, not for another GPU user
        sleep(settings.gpu_poll_sec)  # never respawn immediately after yield/OOM/crash
        return True
    if waiting_for_server:   # its jobs start as soon as the shared server is ready
        sleep(IDLE_POLL_SEC)
        return True
    log.info("%s秒後にGPUを再確認します", settings.gpu_poll_sec)
    sleep(settings.gpu_poll_sec)
    return True


def _ended_for_detect(repo: Repo, guard) -> bool:
    """A content check is queued and the GPU is not wanted by others."""
    if DETECT_KEY not in repo.pending_model_keys():
        return False
    try:
        return not guard.should_yield()
    except GpuQueryError:
        return False


def _can_start(guard, settings: Settings, key: str, budget: float | None) -> tuple[bool, str]:
    try:   # names:cpu needs no GPU: it never waits for one
        return ((True, "") if key == NAMES_CPU_KEY
                else guard.can_start(budget) if budget is not None
                else (False, f"設定にないモデル（{settings.detect_model_key}）です"))
    except GpuQueryError as e:
        return False, f"GPUの状態を確認できません: {e}"


def _on_load_failure(repo: Repo, key: str, load_failures: dict[str, int]) -> None:
    """Count consecutive load failures. After MAX_LOAD_FAILURES, fail the key's pending jobs
    so the weights are not loaded onto the shared GPU again until new jobs arrive."""
    load_failures[key] = load_failures.get(key, 0) + 1
    entry = repo.latest_audit(LOAD_FAILED_ACTION, key)
    reason = entry["detail"].get("error", "不明なエラー") if entry else "不明なエラー"
    log.error("モデルの読み込みに失敗しました model_key=%s (%d/%d回連続): %s",
              key, load_failures[key], MAX_LOAD_FAILURES, reason)
    if load_failures[key] >= MAX_LOAD_FAILURES:
        n = repo.fail_pending_jobs(key, f"モデルを読み込めませんでした: {reason}")
        load_failures.pop(key)
        log.error("model_key=%s の待機中ジョブ%d件を再試行可能な失敗にしました。"
                  "新しいジョブが来るまでこのモデルは起動しません", key, n)
        if repo.settings is not None:
            alerts.send(repo.settings, f"load-failed:{key}", "モデルを読み込めませんでした",
                        f"{key} の読み込みが{MAX_LOAD_FAILURES}回続けて失敗したため、"
                        f"待っていたジョブ{n}件を止めました。サーバーの状態を確かめてください。",
                        "error")


def worker_loop(settings: Settings, *, guard: GpuGuard, spawn: Callable[[str], int],
                sleep: Callable[[float], None] = time.sleep, once: bool = False,
                clock: Callable[[], float] = time.monotonic,
                engine_cleanup: Callable[[str], None] | None = None,
                on_start: Callable[[Settings], None] = startup_cleanup,
                shared: SharedServer | None = None,
                linecall: LinecallRunner | None = None,
                disk_check: Callable[[Settings], disk.DiskState] = disk.worker_state,
                videos: ExplainerRunner | None = None) -> None:
    """`shared`: the warm server for the shared model (worker.shared_server); None: every
    detect / pfb3x3 session starts and stops a server of its own (also with --once).
    `linecall`: runs linecall jobs beside the one session (worker.linecall_runner); None
    (--once): linecall is a session like the others.

    Each pass also re-queues once the steps that stopped for a passing reason
    (services.auto_retry), and starts nothing new while the disk is below disk_refuse_gb
    (`disk_check`, ops.disk; open uploads' promised bytes count as used): deletions still
    finish, so deleting works makes room. Hourly it sweeps uploads given up on and finished
    ones nobody claimed for 48 h (services.chunked_upload.sweep), and applies 「動画を自動で消す」
    (services.retention.sweep).

    解説動画 (services.explainer): a video left half made by a stopped worker has failed;
    `videos` (the resident worker's ExplainerRunner) makes the queued ones beside the
    sessions; None (--once): they are made at the end of the pass."""
    conn = connect(files.db_path(settings))
    try:
        init_schema(conn)
        repo = Repo(conn, settings)
        log.info("ワーカー開始 (worker started)")
        on_start(settings)
        n = explainer.recover(repo)
        if n:
            log.warning("途中で止まった解説動画を失敗にしました: %d", n)
        if videos is not None:
            videos.start()
        try:
            n = deletion.sweep_orphan_dirs(repo, settings)
            if n:
                log.warning("行のない作業フォルダを削除しました: %d", n)
        except Exception:   # noqa: BLE001 - the worker starts anyway
            log.exception("行のない作業フォルダの確認に失敗しました")
        load_failures: dict[str, int] = {}
        waiting_since: dict[str, float] = {}
        if shared is not None:
            shared.open(files.db_path(settings))
        holding = False
        swept: dict = {"t": None}
        retained: dict = {"t": None}
        with WorkerBeat(files.worker_beat_path(settings)):
            while True:
                # every pass: a job orphaned by a killed worker/child must not stay running
                n = repo.recover_stale_jobs(settings.stale_job_sec)
                if n:
                    log.warning("更新が止まっていたジョブを回収しました: %d", n)
                _linecall_poll(repo, settings, linecall, engine_cleanup)
                try:
                    auto_retry.sweep(repo, settings)
                except Exception:   # noqa: BLE001 - the worker keeps running
                    log.exception("止まったジョブの自動の再試行に失敗しました")
                # works whose running step has ended (also after a crash): delete them now
                try:
                    deletion.finish_requested_deletions(repo, settings)
                except Exception:   # noqa: BLE001 - the worker keeps running
                    log.exception("削除を頼まれた作業の確認に失敗しました")
                try:   # 「この人のデータを消す」 that stopped between its rows and its files
                    erasure.finish_unfinished(repo, settings)
                except Exception:   # noqa: BLE001 - the worker keeps running
                    log.exception("途中で止まった人のデータの消去の確認に失敗しました")
                _sweep_uploads(settings, clock, swept)
                _retention_pass(repo, settings, clock, retained)
                holding = _disk_pass(settings, disk_check, holding)
                if holding:
                    if once:
                        return
                    sleep(DISK_HOLD_SEC)
                    continue
                _run_cpu_jobs(repo, settings)
                busy = _gpu_pass(repo, settings, guard, spawn, sleep, load_failures,
                                 waiting_since, clock, engine_cleanup, shared, linecall)
                if once:
                    explainer.fail_stale(repo)
                    explainer.ensure_sources(repo, settings)
                    explainer.run_pending(repo, settings)
                    return
                if not busy:
                    sleep(IDLE_POLL_SEC)
    finally:
        try:
            if videos is not None:
                videos.stop()
        except Exception:   # noqa: BLE001 - the rest still stops
            log.exception("解説動画の処理を停止できませんでした")
        try:
            if linecall is not None:
                linecall.stop()
        except Exception:   # noqa: BLE001 - the shared server is still stopped
            log.exception("判定材料のセッションを停止できませんでした")
        try:
            if shared is not None:
                shared.close()
        finally:
            conn.close()


def _sweep_uploads(settings: Settings, clock: Callable[[], float], last: dict) -> None:
    """At most every UPLOAD_SWEEP_SEC: drop uploads given up on and finished files nobody
    claimed (their reservations and bytes go with them). Never stops the worker."""
    now = clock()
    if last.get("t") is not None and now - last["t"] < UPLOAD_SWEEP_SEC:
        return
    last["t"] = now
    try:
        n = chunked_upload.sweep(settings)
        if n:
            log.info("使われなかった送信ファイルを片づけました: %d", n)
    except Exception:   # noqa: BLE001 - the worker keeps running
        log.exception("送信ファイルの片づけに失敗しました")


def _retention_pass(repo: Repo, settings: Settings, clock: Callable[[], float],
                    last: dict) -> None:
    """At most every retention.SWEEP_SEC: 「動画を自動で消す」 (services.retention.sweep).
    Never stops the worker."""
    now = clock()
    if last.get("t") is not None and now - last["t"] < retention.SWEEP_SEC:
        return
    last["t"] = now
    try:
        done = retention.sweep(repo, settings)
        if done["cleared"] or done["deleted"]:
            log.info("設定どおりに消しました: 動画 %d件、作業 %d件", len(done["cleared"]),
                     len(done["deleted"]))
    except Exception:   # noqa: BLE001 - the worker keeps running
        log.exception("「動画を自動で消す」の処理に失敗しました")


def _disk_pass(settings: Settings, disk_check, holding: bool) -> bool:
    """True while nothing new may start (the disk is below disk_refuse_gb); warns and alerts
    once per change (alerts repeat at most once per alert_repeat_sec)."""
    try:
        state = disk_check(settings)
    except Exception:   # noqa: BLE001 - an unreadable disk state never stops the worker
        log.exception("保存場所の空きを確かめられませんでした")
        return False
    if state.level == "refuse":
        if not holding:
            log.error("保存場所の空きが%.1fGBしかないため、新しい処理を始めません（%sGB未満）",
                      state.free_gb, settings.disk_refuse_gb)
        alerts.send(settings, "disk-refuse", "保存場所の空きが足りません",
                    f"残り約{state.free_gb:.0f}GBのため、新しい解析と取り込みを止めています。"
                    "使い終わった作業を削除してください。", "error")
        return True
    if holding:
        log.info("保存場所の空きが戻りました（%.1fGB）。処理を再開します", state.free_gb)
        alerts.clear(settings, "disk-refuse")
    if state.level == "warn":
        alerts.send(settings, "disk-warn", "保存場所の空きが少なくなっています",
                    f"残り約{state.free_gb:.0f}GBです。使い終わった作業の削除をおすすめします。")
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pfb_app.worker.main",
                                     description="PFB worker (GPU-aware job runner)")
    parser.add_argument("--settings", help="settings YAML (default: built-in defaults)")
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    args = parser.parse_args(argv)
    offline.apply()   # before any model library is loaded; the sessions inherit it
    settings = load_settings(Path(args.settings) if args.settings else None)
    setup_logging(settings, "worker")
    exit_on_signals()   # stop.sh: leave through finally, not by being killed
    try:
        lock = WorkerLock(settings.data_dir / "worker.lock").acquire()
    except AlreadyRunning as e:
        log.error("別のワーカーが動いているため終了します: %s", e)
        raise SystemExit(EXIT_ALREADY_RUNNING) from None
    try:
        with RunningMarker(settings.data_dir / "worker.running") as marker:
            if marker.unclean:   # the previous worker did not leave through its finally
                alerts.send(settings, "worker-restarted", "ワーカーを再起動しました",
                            "前回のワーカーが正しく終了していませんでした（停電・強制終了など）。"
                            "止まった作業がないか「処理状況」で確かめてください。")
            guard = GpuGuard(settings)
            # --once ends the worker after one pass: a warm server would not outlive it
            shared = None if args.once else SharedServer.for_settings(settings, guard)
            # --once waits for every session; otherwise linecall runs beside them
            linecall = None if args.once else LinecallRunner(settings, args.settings)
            # --once makes the queued 解説動画 at the end of its pass instead
            videos = None if args.once else ExplainerRunner(settings)
            worker_loop(settings, guard=guard,
                        spawn=functools.partial(spawn_session, settings_path=args.settings),
                        once=args.once, shared=shared, linecall=linecall, videos=videos)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
