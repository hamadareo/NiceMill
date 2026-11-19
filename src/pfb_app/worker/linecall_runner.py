"""判定支援's linecall jobs beside the worker's one session (release-2 ruling).

A linecall job runs for about an hour with a small GPU need (the pose model, about 3 GB; the
engine checks for main.LINECALL_ENGINE_FREE_GB free only when it starts). Run in the worker's
one-session loop it held every other work up for that hour, so it runs outside it, the way the
live run does: in a detached child of its own that the worker supervises instead of waiting
for. The child is an ordinary `linecall` session (worker.session: the same job loop, so job
status, heartbeats, a work's delete request and the automatic retry work as before).

- At most MAX_RUNNING (1) at a time; after an exit that was not idle (no room on the GPU, a
  yield, a crash) the next start waits gpu_poll_sec, as a session's respawn does.
- The worker starts it only when the engine's start would fit (main._gpu_pass, the key's
  budget), and while it runs counts RUN_GB on top of every other GPU start (sessions and the
  shared server's) as a running GPU consumer. The shared server's yield rules are unchanged.
- Its engine command records itself in the engine's linecall_call_record, so a session started
  beside it never stops it as a leftover (and it never stops theirs).
- The worker polls it every pass (main._linecall_poll): a crash recovers only its own running
  job (heartbeat timeout, so auto_retry may requeue it once) and stops its engine command
  (pid-verified). When the worker ends, the child is stopped (SIGTERM: the session's finally
  stops the engine command; SIGKILL after STOP_WAIT_SEC).
- A worker that dies without that stop (SIGKILL, OOM killer) must not leave a child claiming
  jobs: the child (child_main) asks the kernel for SIGTERM when its parent dies
  (PR_SET_PDEATHSIG, Linux) and also watches getppid() every PARENT_POLL_SEC (ParentWatch, the
  portable fallback); either ends it through its SIGTERM path, so its session's finally stops
  the engine command. It records itself (pid + start time) in SESSION_RECORD; a new worker
  starts no child while that record, or linecall_call.pgid, names a live process it did not
  start (LinecallRunner.foreign: never signalled, only waited for), and its startup cleanup
  leaves such an orphan's engine command to the orphan.
"""
import ctypes
import ctypes.util
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings, load_settings
from pfb_app.engine import Pfb3x3Engine
from pfb_app.engine.pfb3x3 import (_cmdline, _is_our_group, _pid_alive, _remove_if,
                                   _start_time, _write_atomic)
from pfb_app.pipeline.runner import LINECALL_KEY
from pfb_app.worker.session import EXIT_IDLE, exit_on_signals, session_entry

log = logging.getLogger(__name__)

MAX_RUNNING = 1
# the engine's LINECALL_NEED_MIB (pfb3x3.engine.gpu_device): what the run takes once started
RUN_GB = 6144 / 1024
STOP_WAIT_SEC = 30.0
PARENT_POLL_SEC = 2.0
SESSION_RECORD = "linecall_session.pid"   # in the engine's runtime dir, beside linecall_call.pgid
_PR_SET_PDEATHSIG = 1
# the tail of a multiprocessing spawn child's command line (spawn.get_command_line)
_SPAWN_MARK = "--multiprocessing-fork"


def session_record(settings: Settings) -> Path:
    """`<pid> <start time>` of the running linecall child (written by the child itself)."""
    return Pfb3x3Engine(settings).runtime_dir / SESSION_RECORD


def live_session(settings: Settings) -> int | None:
    """The pid of the linecall child SESSION_RECORD names while it still runs: the same start
    time and a spawn child's command line (on Linux; without /proc only a live pid). Else
    None (no record, a torn one, or a pid that is now another process)."""
    try:
        fields = session_record(settings).read_text(encoding="utf-8").split()
        pid = int(fields[0])
        start = fields[1] if len(fields) > 1 else None
    except (FileNotFoundError, ValueError, IndexError):
        return None
    if not Path("/proc/self/stat").exists():
        return pid if start is None and _pid_alive(pid) else None
    if start is None or _start_time(pid) != start or not _pid_alive(pid):
        return None
    return pid if _SPAWN_MARK in _cmdline(pid) else None


def _set_pdeathsig(sig: int) -> bool:
    """Linux: deliver `sig` to this process when its parent (thread) dies. False elsewhere or
    when prctl is unavailable (ParentWatch then covers it)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def _terminate_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)   # exit_on_signals: SystemExit, the finally blocks run


class ParentWatch(threading.Thread):
    """Calls on_gone() once getppid() is no longer parent_pid (the worker died and the child
    was re-parented), polling every poll_sec. Daemon thread."""

    def __init__(self, parent_pid: int, poll_sec: float = PARENT_POLL_SEC, *,
                 on_gone: Callable[[], None] = _terminate_self,
                 getppid: Callable[[], int] = os.getppid):
        super().__init__(name="pfb-linecall-parent-watch", daemon=True)
        self._parent = parent_pid
        self._poll = poll_sec
        self._on_gone = on_gone
        self._getppid = getppid
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self._poll):
            if self._getppid() != self._parent:
                log.warning("ワーカーが終了していたため、判定材料のセッションを終了します")
                self._on_gone()
                return

    def stop(self) -> None:
        self._halt.set()


def child_main(settings_path: str | None, parent_pid: int, *,
               target: Callable[[str | None, str], object] = session_entry,
               pdeathsig: bool = True, poll_sec: float = PARENT_POLL_SEC) -> None:
    """The linecall child: ends with its worker (see the module doc), records itself in
    SESSION_RECORD while it runs, and runs the linecall session (`target`). The kernel's
    parent-death signal follows the THREAD that spawned the child: LinecallRunner.start runs
    on the worker's main thread (worker_loop), which lives as long as the worker."""
    exit_on_signals()   # first: a SIGTERM from here on ends through the finally blocks
    if pdeathsig:
        _set_pdeathsig(signal.SIGTERM)
    if os.getppid() != parent_pid:   # the worker died before (or while) we got here
        log.warning("ワーカーがすでに終了していたため、判定材料のセッションを始めません")
        return
    watch = ParentWatch(parent_pid, poll_sec)
    watch.start()
    record = stamp = None
    try:
        settings = load_settings(Path(settings_path) if settings_path else None)
        record = session_record(settings)
        stamp = f"{os.getpid()} {_start_time(os.getpid()) or ''}".strip() + "\n"
        _write_atomic(record, stamp)
    except Exception:   # noqa: BLE001 - the session reports a bad settings file itself
        record = None
    try:
        target(settings_path, LINECALL_KEY)
    finally:
        watch.stop()
        if record is not None:
            _remove_if(record, stamp)


def _spawn(settings_path: str | None):
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=child_main, args=(settings_path, os.getpid()),
                       name="pfb-linecall", daemon=True)
    proc.start()
    return proc


class LinecallRunner:
    def __init__(self, settings: Settings, settings_path: str | None, *,
                 start: Callable[[], object] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 stop_stale: Callable[[], object] | None = None):
        self.settings = settings
        self._start = start or (lambda: _spawn(settings_path))
        self._clock = clock
        self._stop_stale = stop_stale or Pfb3x3Engine(settings).stop_stale_linecall
        self._call_record = Pfb3x3Engine(settings).linecall_call_record
        self._foreign_seen: int | None = None
        self.proc = None
        self.started_at: float | None = None
        self._cool_until: float | None = None

    @property
    def busy(self) -> bool:
        """A child is running (it may have ended: poll() takes its exit)."""
        return self.proc is not None

    def gpu_gb(self) -> float:
        """What a running linecall takes on the GPU (counted on top of other starts)."""
        return RUN_GB if self.busy else 0.0

    def ready(self) -> bool:
        """No child runs, none ended for a passing reason less than gpu_poll_sec ago, and no
        linecall this runner did not start still runs (foreign)."""
        if self.proc is not None or (self._cool_until is not None
                                     and self._clock() < self._cool_until):
            return False
        other = self.foreign()
        if other != self._foreign_seen:
            self._foreign_seen = other
            if other is not None:
                log.warning("前のワーカーの判定材料（pid %s）がまだ動いているため、"
                            "終わるまで次を始めません", other)
        return other is None

    def foreign(self) -> int | None:
        """A linecall child or engine linecall command this runner did not start that still
        runs (an orphan of a killed worker, or another worker's): its pid / pgid, else None.
        Only read, never signalled."""
        pid = live_session(self.settings)
        if pid is not None and pid != getattr(self.proc, "pid", None):
            return pid
        if self.proc is not None:   # the recorded engine command is our own child's
            return None
        try:
            fields = self._call_record.read_text(encoding="utf-8").split()
            pgid, start = int(fields[0]), (fields[1] if len(fields) > 1 else None)
        except (FileNotFoundError, ValueError, IndexError):
            return None
        return pgid if _is_our_group(pgid, start) else None

    def start(self) -> None:
        if self.proc is not None:
            return
        self.proc = self._start()
        self.started_at = self._clock()
        log.info("判定材料のセッションを別に起動します pid=%s", getattr(self.proc, "pid", None))

    def poll(self) -> int | None:
        """The exit code of a child that has ended (once), else None."""
        proc = self.proc
        if proc is None or proc.is_alive():
            return None
        proc.join()
        code = proc.exitcode
        self.proc = None
        self._cool_until = None if code == EXIT_IDLE else \
            self._clock() + self.settings.gpu_poll_sec
        return code

    def cleanup(self) -> None:
        """Stop the engine command a dead child left running (pid-verified)."""
        try:
            self._stop_stale()
        except Exception:   # noqa: BLE001
            log.exception("判定材料の残ったプロセスを停止できませんでした")

    def stop(self) -> None:
        """Worker exit: stop the child (SIGTERM, then SIGKILL) and its engine command."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.is_alive():
                log.info("判定材料のセッションを停止します pid=%s", getattr(proc, "pid", None))
                proc.terminate()
                proc.join(STOP_WAIT_SEC)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
        finally:
            self.cleanup()
