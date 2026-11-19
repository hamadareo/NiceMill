"""The worker's one warm vLLM server for the shared model (SHARED_MODEL_KEY, "standard").

One instance of the engine's scripts/serve.sh (qwen3vl-8b, OpenAI-compatible) answers every
VLM request of the works on that model: the content check's scene verdict, the engine's
behavior and names cues (--base-url), contribution, pfb and the translation of the engine's
English (pfb_app.inference.served). The worker owns it, not a session:

- it is started once per burst, in a background thread (another session, e.g. track, may run
  meanwhile): when a job that needs it is queued (`detect`, `pfb3x3:standard`), or ahead of
  time while a confirmed work of the model is still before behavior (prepare / transcribe /
  track / identity / names queued or running: pre-warm). Either way only when
  GpuGuard.can_start allows the model's budget (pfb3x3_budget_gb), as a session would. No
  pre-warm while any job waits for the GPU (but 判定支援's linecall, a side stage), or after a start that failed with an error (until
  a job starts it again and it comes up). While it starts, vLLM has not taken its memory
  yet: the worker then counts its budget on top of any other GPU session's (main._gpu_pass);
- it must listen on this machine only: right after it is ready its listening sockets are
  read (`ss`); bound to anything but 127.0.0.1 / ::1 (the engine's serve.sh before its
  --host fix binds 0.0.0.0), it is stopped, a warning is logged and the shared server is
  disabled for this worker (every session then starts its own server, as before);
- sessions get its handle (session.run_session(server=...)) and neither start nor stop it;
- it stays warm while such jobs are queued and WARM_IDLE_S after the last one (checked by
  the worker on every pass and by the watcher, also while another session runs);
- it gives the GPU back as a session's server did: while no session uses it, as soon as
  GpuGuard.should_yield (the worker on every pass, the watcher every WATCH_SEC); after a
  session that used it ended with a yield or OOM while the GPU is wanted; after a session that
  found it failing (EXIT_SERVER_DOWN) or crashed; before a session that needs a server of its
  own (pfb3x3:high, a detect model of another key: one serve.pid and one port per data dir) or
  loads a model in-process (standard / high / whisper); and when another session would fit
  without it while no queued job needs it;
- before a live run (engine.live: the page writes a release request) it is stopped unless a
  session is using it, the request is answered, and no server starts (for a job or a
  pre-warm) for live.RELEASE_HOLD_SEC; the worker checks on every pass and the watcher on
  every RELEASE_POLL_SEC;
- it is stopped when the worker ends, through the engine's pid-verified stop path
  (stop_server.sh checks serve.pid / serve.ident), and that path is run once more at the end
  in any case; worker start-up still stops any server a previous worker left
  (main.startup_cleanup).
"""
import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings
from pfb_app.engine import (EngineNeedsGpu, EngineYield, Pfb3x3Engine, ServerHandle,
                            engine_model_for)
from pfb_app.engine import live
from pfb_app.inference.gpu import GpuQueryError
from pfb_app.pipeline.runner import DETECT_KEY, PFB3X3_PREFIX
from pfb_app.storage.db import SIDE_STAGES, Repo, connect

log = logging.getLogger(__name__)

SHARED_MODEL_KEY = "standard"
WARM_IDLE_S = 600   # the server stays this long after the last job that needed it
WATCH_SEC = 30      # the watcher's checks while no session uses the server
RELEASE_POLL_SEC = 1.0   # the watcher looks for a live run's release request this often
# stages of a confirmed tracking-engine work that come before behavior (pre-warm)
PREWARM_STAGES = ("prepare", "transcribe", "track", "identity", "names")
LOCAL_ADDRESSES = ("127.0.0.1", "::1", "[::1]", "localhost")


@dataclass(frozen=True)
class StartFailure:
    """A start that ended without a server: kind "needs_gpu" (serve.sh exit 3), "yield" (the
    GPU was wanted while it started, or could not be read), "not_local" (it listened beyond
    this machine; stopped) or "error"."""
    kind: str
    reason: str


def _first_line(e: BaseException) -> str:
    lines = [line.strip() for line in str(e).splitlines() if line.strip()]
    return lines[0] if lines else type(e).__name__


def bind_problem(addresses: list[str] | None) -> str | None:
    """None when every listening address is loopback; else why the server is refused."""
    if addresses is None:
        return "待ち受けアドレスを確認できません（ss）"
    if not addresses:
        return "待ち受けているソケットが見つかりません"
    outside = [a for a in addresses if a not in LOCAL_ADDRESSES]
    if outside:
        return f"このマシンの外からも接続できるアドレスで待ち受けています: {', '.join(outside)}"
    return None


def configured(settings: Settings) -> bool:
    """The settings have a shared model with an engine budget (else the worker runs none)."""
    return SHARED_MODEL_KEY in settings.models and SHARED_MODEL_KEY in settings.pfb3x3_budget_gb


class SharedServer:
    def __init__(self, settings: Settings, guard, engine: Pfb3x3Engine | None = None, *,
                 model_key: str = SHARED_MODEL_KEY, clock: Callable[[], float] = time.monotonic,
                 warm_idle_s: float = WARM_IDLE_S, watch_sec: float = WATCH_SEC):
        self.settings = settings
        self.guard = guard
        self.engine = engine or Pfb3x3Engine(settings)
        self.model_key = model_key
        self.session_key = f"{PFB3X3_PREFIX}{model_key}"
        self.model = engine_model_for(model_key)
        self.budget = settings.pfb3x3_budget_gb[model_key]
        # more than the budget when the GPU has room (settings.pfb3x3_max_budget_gb)
        self.max_budget = settings.pfb3x3_max_budget_gb.get(model_key)
        self.held_gb = self.budget   # what the server takes (or will take while it starts)
        self.warm_idle_s = warm_idle_s
        self.watch_sec = watch_sec
        self._clock = clock
        self._lock = threading.RLock()
        self.handle: ServerHandle | None = None
        self.in_use = False
        self.disabled: str | None = None   # why the shared server is not used (not local)
        self._starter: threading.Thread | None = None
        self._started_at: float | None = None
        self._outcome: tuple[ServerHandle | None, StartFailure | None] | None = None
        self._cancel = threading.Event()
        self._last_wanted = clock()
        self._failed_at: float | None = None
        self._failure: StartFailure | None = None
        self._unreported: StartFailure | None = None   # kept until maintain() reports it
        self._watch_stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self.starts = 0   # servers started (for logs and tests)
        # GPU memory of consumers running beside it that the GPU may not show yet (a
        # linecall child: worker.linecall_runner), counted on top of its budget at a start
        self.extra_gb: Callable[[], float] = lambda: 0.0

    @classmethod
    def for_settings(cls, settings: Settings, guard) -> "SharedServer | None":
        """None when the settings have no shared model (or no engine budget for it)."""
        return cls(settings, guard) if configured(settings) else None

    # --- what needs it ---

    def serves(self, key: str) -> bool:
        """Whether jobs of session key `key` are answered by this server."""
        return key == self.session_key or (key == DETECT_KEY
                                           and self.settings.detect_model_key == self.model_key)

    def wanted(self, repo: Repo) -> bool:
        """A queued / waiting job needs the server now."""
        return any(self.serves(k) for k in repo.pending_model_keys())

    def prewarm_wanted(self, repo: Repo) -> bool:
        """A confirmed tracking-engine work of this model is queued or running before behavior,
        or (the content check's model) a video upload is still coming in
        (chunked_upload.video_arriving), and no job of any key waits for the GPU (they come
        first; a waiting linecall, a side stage, does not hold the work's analysis up)."""
        if repo.conn.execute("SELECT 1 FROM jobs WHERE status='waiting_gpu'"
                             f" AND stage NOT IN ({','.join('?' * len(SIDE_STAGES))}) LIMIT 1",
                             SIDE_STAGES).fetchone():
            return False
        row = repo.conn.execute(
            "SELECT 1 FROM jobs j JOIN submissions s ON j.submission_id = s.id"
            " JOIN assignments a ON s.assignment_id = a.id"
            " WHERE j.status IN ('queued', 'running')"
            f" AND j.stage IN ({','.join('?' * len(PREWARM_STAGES))})"
            " AND json_extract(s.meta, '$.engine') = 'pfb3x3' AND a.model_key = ? LIMIT 1",
            (*PREWARM_STAGES, self.model_key)).fetchone()
        return row is not None or self._upload_arriving()

    def _upload_arriving(self) -> bool:
        if self.settings.detect_model_key != self.model_key:
            return False
        from pfb_app.services import chunked_upload   # noqa: PLC0415 - services import us
        try:
            return chunked_upload.video_arriving(self.settings)
        except OSError:
            return False

    # --- state ---

    @property
    def starting(self) -> bool:
        return self._starter is not None

    def starting_for(self) -> float:
        return 0.0 if self._started_at is None else self._clock() - self._started_at

    def holds_gpu(self) -> bool:
        return self.handle is not None or self.starting

    def _cooling(self) -> bool:
        """A start failed less than gpu_poll_sec ago: not tried again yet."""
        return self._failed_at is not None \
            and self._clock() - self._failed_at < self.settings.gpu_poll_sec

    def _gpu_wanted(self) -> bool:
        try:
            return self.guard.should_yield()
        except GpuQueryError as e:   # cannot see the GPU -> assume others need it
            log.warning("GPUの状態を確認できないため譲ります: %s", e)
            return True

    # --- start / stop ---

    def start(self) -> None:
        """Start the server in a background thread (no-op while it runs or starts, or when
        the shared server is disabled)."""
        with self._lock:
            if self.disabled or self.handle is not None or self._starter is not None:
                return
            self._cancel.clear()
            self.held_gb = self.start_budget()
            self._started_at = self._clock()
            self._starter = threading.Thread(target=self._start, name="pfb-shared-server-start",
                                             daemon=True)
            self._starter.start()

    def _start(self) -> None:
        def tick() -> bool:   # every engine tick while serve.sh waits for the model
            return self._cancel.is_set() or self._gpu_wanted()

        try:
            util = self.guard.utilization_for(self.held_gb)
            self.engine.stop_stale_server()   # a leftover server of ours would make serve.sh refuse
            log.info("共有のPFB3x3サーバーを起動します model=%s 予算=%gGB 使用率=%g", self.model,
                     self.held_gb, util)
            handle = self.engine.start_server(self.model, util, on_tick=tick)
            problem = bind_problem(self.engine.listen_addresses(handle.port))
            if problem is None:
                outcome = (handle, None)
            else:
                log.warning("共有のPFB3x3サーバーを使いません（%s）。停止します pid=%s", problem,
                            handle.pid)
                self._stop_handle(handle)
                outcome = (None, StartFailure("not_local", problem))
        except EngineNeedsGpu as e:
            outcome = (None, StartFailure("needs_gpu", _first_line(e)))
        except (EngineYield, GpuQueryError) as e:
            outcome = (None, StartFailure("yield", _first_line(e)))
        except Exception as e:   # noqa: BLE001 - reported to the worker as a load failure
            log.exception("共有のPFB3x3サーバーを起動できませんでした")
            outcome = (None, StartFailure("error", f"{type(e).__name__}: {_first_line(e)}"))
        self._outcome = outcome   # read only after this thread is joined (no lock: stop()
        # joins this thread while holding the lock)

    def wait_started(self, timeout: float | None = None) -> None:
        """Wait for a running start to end (tests, shutdown)."""
        starter = self._starter
        if starter is not None:
            starter.join(timeout)

    def poll(self) -> None:
        """Take the result of a start that has ended (a failure is kept for maintain() to
        report); forget a server whose process is gone."""
        with self._lock:
            starter = self._starter
            if starter is not None and not starter.is_alive():
                starter.join()
                handle, failure = self._outcome or (None, StartFailure("error", "no outcome"))
                self._starter, self._outcome = None, None
                if handle is not None:
                    self.handle = handle
                    self.starts += 1
                    self._failed_at, self._failure = None, None
                    log.info("共有のPFB3x3サーバーの準備ができました pid=%s 起動%.0f秒", handle.pid,
                             self.starting_for())
                else:
                    self._failed_at, self._failure = self._clock(), failure
                    self._unreported = failure
                    if failure.kind == "not_local":
                        self.disabled = failure.reason
                        log.warning("共有のPFB3x3サーバーを無効にします。各セッションが自分の"
                                    "サーバーを使います: %s", failure.reason)
                    else:
                        log.info("共有のPFB3x3サーバーは起動しませんでした（%s）: %s", failure.kind,
                                 failure.reason)
            if self.handle is not None and not self.engine.server_alive(self.handle):
                log.warning("共有のPFB3x3サーバーが止まっていました pid=%s", self.handle.pid)
                self.handle = None

    def _stop_handle(self, handle: ServerHandle) -> None:
        try:
            self.engine.stop_server(handle)
            return
        except Exception:   # noqa: BLE001
            log.exception("共有のPFB3x3サーバーを停止できませんでした pid=%s", handle.pid)
        try:
            self.engine.stop_stale_server()
        except Exception:   # noqa: BLE001
            log.exception("PFB3x3の残ったサーバーを停止できませんでした")

    def stop(self, reason: str) -> None:
        """Stop a starting or running server (pid-verified); a no-op without one."""
        with self._lock:
            starter = self._starter
            if starter is not None:
                log.info("共有のPFB3x3サーバーの起動を取りやめます（%s）", reason)
                self._cancel.set()
                starter.join()   # the start's own tick ends serve.sh and cleans up after it
                self.poll()
                if self._unreported is not None and self._unreported.kind == "yield":
                    self._unreported = None   # our own cancel, not a failure to report
            handle, self.handle = self.handle, None
            if handle is None:
                return
            log.info("共有のPFB3x3サーバーを停止します（%s） pid=%s", reason, handle.pid)
            self._stop_handle(handle)

    # --- a live run's release request (engine.live) ---

    def _release_for_live(self) -> bool:
        """A live run asked for the GPU less than RELEASE_HOLD_SEC ago: stop a server no
        session is using and answer the request. True while the request holds."""
        try:
            req = live.release_request(self.settings)
        except OSError:
            return False
        if req is None:
            return False
        with self._lock:
            if self.holds_gpu() and not self.in_use:
                self.stop("ライブ判定のためにGPUを空けます")
            if not self.holds_gpu():
                try:
                    if not live.release_acked(self.settings, req["id"]):
                        live.ack_release(self.settings, req["id"])
                except OSError:
                    log.exception("ライブ判定への返事を書けませんでした")
        return True

    # --- the worker's calls ---

    def _idle_check(self, repo: Repo) -> None:
        """Refresh the last time a job needed the server; stop it WARM_IDLE_S after that."""
        wanted = self.wanted(repo) or self.prewarm_wanted(repo)
        if wanted:
            self._last_wanted = self._clock()
        elif self.holds_gpu() and self._clock() - self._last_wanted >= self.warm_idle_s:
            self.stop(f"{self.warm_idle_s:g}秒使われなかったため")

    def maintain(self, repo: Repo) -> StartFailure | None:
        """Once per worker pass, while no session runs: take a finished start, give the GPU
        back when it is wanted, stop the server WARM_IDLE_S after the last job that needed it,
        and pre-warm. Returns a start failure not reported yet (once)."""
        with self._lock:
            self.poll()
            failure, self._unreported = self._unreported, None
            if self.disabled:
                return failure
            if self.holds_gpu() and self._gpu_wanted():
                self.stop("GPUの空きが少ないため譲ります")
            if self._release_for_live():   # a live run starts: no pre-warm meanwhile
                return failure
            self._idle_check(repo)
            if not self.holds_gpu() and not self.wanted(repo) and self.prewarm_wanted(repo) \
                    and not self._cooling() \
                    and not (self._failure is not None and self._failure.kind == "error"):
                ok, reason = self._can_start()
                if ok:
                    log.info("確定した作品の解析に備えて、共有のPFB3x3サーバーを先に起動します")
                    self.start()
                else:
                    log.info("共有のPFB3x3サーバーの先行起動は見送ります: %s", reason)
        return failure

    def start_budget(self) -> float:
        """The GB the next start gives the server: its budget, or, with a max_budget above it,
        as much of the free GPU memory as leaves track_budget_gb + start_margin_gb (and what
        a running linecall needs) beside it, at most max_budget and never below the budget.
        The budget alone when the GPU cannot be read."""
        cap = self.max_budget
        if not cap or cap <= self.budget:
            return self.budget
        try:
            free = self.guard.free_now_gb()
        except (AttributeError, GpuQueryError):
            return self.budget
        room = free - self.extra_gb() - self.settings.track_budget_gb - self.settings.start_margin_gb
        return float(max(self.budget, min(cap, math.floor(room))))

    def _can_start(self) -> tuple[bool, str]:
        try:
            return self.guard.can_start(self.budget + self.extra_gb())
        except GpuQueryError as e:
            return False, f"GPUの状態を確認できません: {e}"

    def request(self) -> tuple[str, str]:
        """For a job that needs the server now: ("ready", ""), ("starting", "") (also when this
        call started it), or ("no", reason): no room on the GPU, or a start failed less than
        gpu_poll_sec ago."""
        with self._lock:
            self.poll()
            if self.handle is not None:
                return "ready", ""
            if self.starting:
                return "starting", ""
            if self._release_for_live():
                return "no", live.RELEASE_REASON
            if self._cooling():
                return "no", self._failure.reason if self._failure else "起動に失敗しました"
            ok, reason = self._can_start()
            if not ok:
                return "no", reason
            self.start()
            return "starting", ""

    @contextmanager
    def lease(self):
        """The handle as a dict for one session (None: no server any more); the watcher leaves
        the server alone while it is leased."""
        with self._lock:
            handle = self.handle
            self.in_use = handle is not None
        try:
            yield None if handle is None else asdict(handle)
        finally:
            with self._lock:
                self.in_use = False
                self._last_wanted = self._clock()

    def after_session(self, code: int, *, broken: bool, quiet_codes: tuple[int, ...]) -> None:
        """After a session that used the server: a crash or a failing server (`broken`) stops
        it; a yield / OOM stops it while the GPU is wanted; a dead server is forgotten."""
        if broken:
            self.stop("サーバーまたはセッションに異常があったため")
        elif code not in quiet_codes and self.handle is not None and self._gpu_wanted():
            self.stop("GPUの空きが少ないため譲ります")
        else:
            self.poll()

    # --- watcher ---

    def open(self, db_path: Path | None = None) -> "SharedServer":
        """Start the watcher; with `db_path` it also enforces WARM_IDLE_S while another session
        runs (its own connection)."""
        self._watch_stop.clear()
        self._watcher = threading.Thread(target=self._watch, args=(db_path,),
                                         name="pfb-shared-server-watch", daemon=True)
        self._watcher.start()
        return self

    def _watch(self, db_path: Path | None) -> None:
        conn = connect(db_path) if db_path is not None else None
        repo = Repo(conn, self.settings) if conn is not None else None
        tick = min(RELEASE_POLL_SEC, self.watch_sec)
        last = self._clock_now()
        try:
            while not self._watch_stop.wait(tick):
                try:
                    self._release_for_live()   # a live run is about to start: answer soon
                    if self._clock_now() - last < self.watch_sec:
                        continue
                    last = self._clock_now()
                    with self._lock:   # held while stopping: no session gets a stopping server
                        if self.handle is None or self.in_use:
                            continue
                        if self._gpu_wanted():
                            self.stop("GPUの空きが少ないため譲ります")
                        elif repo is not None:
                            self._idle_check(repo)
                except Exception:   # noqa: BLE001 - the watcher keeps watching
                    log.exception("共有のPFB3x3サーバーの確認に失敗しました")
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _clock_now() -> float:
        return time.monotonic()   # the watcher's own period (real time, not the test clock)

    def close(self) -> None:
        """Stop the watcher and the server (worker exit); whatever happens meanwhile (e.g. a
        second signal), the engine's pid-verified stop of serve.pid runs at the end."""
        try:
            self._watch_stop.set()
            if self._watcher is not None and self._watcher is not threading.current_thread():
                self._watcher.join(timeout=self.watch_sec + 5)
            self.stop("ワーカーを終了するため")
        finally:
            try:
                self.engine.stop_stale_server()
            except Exception:   # noqa: BLE001
                log.exception("PFB3x3の残ったサーバーを停止できませんでした")
