"""The engine's `live` line-call demo: one long-running run per app.

`python -m pfb3x3.engine live ...` plays a video at 1x (or reads a stream) and writes its
JSON-lines stream `<data_dir>/live/pfb3x3/tracks/<id>/live/<session>/stream.jsonl` until it is
stopped. Its registry, tracks and logs live under `<data_dir>/live/pfb3x3/` (not under a
submission). The run sits outside the worker's one-call-at-a-time `call_record`.

- `<data_dir>/live/live.json` (the pidfile) holds {pid, start_time, boot_id, session, stream,
  video_id, started_at}. A run counts as alive only while that pid is still the engine
  process we started (same boot, its /proc start time and cmdline, `_is_our_group`; `_ours`):
  a stale pidfile, also one whose pid now belongs to someone else, reads as stopped and is
  never signalled. start() and stop() hold the flock `<data_dir>/live/live.lock`, so two
  starts never run two engines.
- The run has its own process group (start_new_session); stop = SIGTERM to the group (the
  engine forwards it and the stream ends with its `end` record), SIGKILL after stop_wait_sec
  (20 s: after the engine's own LIVE_STOP_WAIT_S of 10 s), and then also SIGKILL of an
  orphaned live_run.py child (collected before that SIGKILL: parent the engine, cmdline
  `live_run.py ... --parent-pid <engine pid>`, started after it; killed by pidfd).
- A reaper thread waits for the run and records its exit code in `<data_dir>/live/
  live.exit.json` (0 ok, 3 needs GPU, other error); without it (another app process started
  the run) the engine's last JSON line in logs/live.out decides. stop() writes
  `live.stopped.json`: a run the user stopped is "stopped", whatever its exit code.

- Before a run starts, the worker is asked to release its warm shared server (the vLLM
  server would leave the live run too little GPU): `request_gpu_release` writes
  `<data_dir>/live/release_gpu.json`; the worker's SharedServer stops the server (unless a
  session is using it), answers in `release_gpu.ack` and starts no server for
  RELEASE_HOLD_SEC. start() waits up to release_wait_sec for that answer, and only while a
  worker is running (its beat file is fresh).

Outputs are 判定材料 (judgement material), never a ruling.
"""
import fcntl
import json
import logging
import os
import secrets
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pfb_app.config import Settings
from pfb_app.engine.pfb3x3 import (
    EXIT_NEEDS_GPU, EngineError, Pfb3x3Engine, _cmdline, _group_alive, _is_our_group,
    _killpg, _last_json, _pid_alive, _start_time, _write_atomic,
)

log = logging.getLogger(__name__)

DEMO_VIDEO_ID = "linecall_demo"
DEMO_SPORT = "basketball"
STOP_WAIT_SEC = 20.0   # > the engine's LIVE_STOP_WAIT_S (10 s) before it kills its child
EXEC_GRACE_SEC = 5.0   # an empty cmdline counts as ours only this long after the start
_PF_KTHREAD = 0x00200000
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
# the engine's NeedsGpu texts (gpu_device), for a run whose exit code was not seen
_NEEDS_GPU_TEXTS = ("MiB free", "no GPU visible")

# the release of the worker's warm shared server before a live run
RELEASE_FILE = "release_gpu.json"
RELEASE_ACK = "release_gpu.ack"
RELEASE_HOLD_SEC = 120.0     # the worker starts no shared server this long after a request
RELEASE_WAIT_SEC = 20.0      # start() waits this long for the worker's answer
WORKER_ALIVE_SEC = 90.0      # a worker beat older than this: no worker to wait for
RELEASE_REASON = "ライブ判定のためにGPUを空けています"

Outcome = Literal["ok", "needs_gpu", "error", "stopped"]

# the reaper thread of each run started by this process, by pid (any controller can join it)
_REAPERS: dict[int, threading.Thread] = {}
_REAPERS_LOCK = threading.Lock()
# the stop_in_background() thread, by live dir: every page session sees a stop under way
_STOPPERS: dict[Path, threading.Thread] = {}


def _release_path(settings: Settings) -> Path:
    return settings.data_dir / "live" / RELEASE_FILE


def request_gpu_release(settings: Settings, now: float | None = None) -> str:
    """Ask the worker to release its warm shared server; returns the request's id."""
    rid = secrets.token_hex(8)
    _write_atomic(_release_path(settings), json.dumps(
        {"id": rid, "at": time.time() if now is None else now}))
    return rid


def release_request(settings: Settings, now: float | None = None) -> dict | None:
    """The release request of the last RELEASE_HOLD_SEC ({"id", "at"}), else None."""
    rec = _read_json(_release_path(settings))
    try:
        age = (time.time() if now is None else now) - float(rec["at"])
        rid = str(rec["id"])
    except (TypeError, KeyError, ValueError):
        return None
    return {"id": rid, "at": float(rec["at"])} if -60 <= age <= RELEASE_HOLD_SEC else None


def ack_release(settings: Settings, rid: str) -> None:
    """The worker's answer: nothing of its own holds the GPU for the request `rid`."""
    _write_atomic(settings.data_dir / "live" / RELEASE_ACK, str(rid))


def release_acked(settings: Settings, rid: str) -> bool:
    try:
        return (settings.data_dir / "live" / RELEASE_ACK).read_text(encoding="utf-8") == rid
    except OSError:
        return False


def _worker_running(settings: Settings, now: float | None = None) -> bool:
    try:
        beat = (settings.data_dir / "worker.beat").stat().st_mtime   # storage.files
    except OSError:
        return False
    return (time.time() if now is None else now) - beat <= WORKER_ALIVE_SEC


class LiveBusy(RuntimeError):
    """A live run is already going; stop it first."""


@dataclass(frozen=True)
class LiveSession:
    video_id: str
    session: str
    stream: Path
    pid: int
    started_at: str   # UTC, ISO 8601


class LiveController:
    def __init__(self, settings: Settings, engine: Pfb3x3Engine, *,
                 stop_wait_sec: float = STOP_WAIT_SEC,
                 release_wait_sec: float = RELEASE_WAIT_SEC):
        self._s = settings
        self._engine = engine
        self._stop_wait = stop_wait_sec
        self._release_wait = release_wait_sec

    # --- paths ---

    @property
    def dir(self) -> Path:
        return self._s.data_dir / "live"

    @property
    def root(self) -> Path:
        """The engine data root of the live runs (videos/, tracks/, logs/)."""
        return self.dir / "pfb3x3"

    @property
    def pidfile(self) -> Path:
        return self.dir / "live.json"

    @property
    def exit_record(self) -> Path:
        return self.dir / "live.exit.json"

    @property
    def stopped_record(self) -> Path:
        return self.dir / "live.stopped.json"

    @contextmanager
    def _locked(self):
        """The flock live.lock (exclusive): one start/stop at a time, across threads and app
        processes."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "live.lock", "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # --- control ---

    def start(self, video_path: Path, *, video_id: str = DEMO_VIDEO_ID,
              tid_video: str | None = None) -> LiveSession:
        """Write the registry entry (sports, basketball) for `video_path` and start the live
        run on the GPU; returns at once (exit 3 shows later as last_outcome() "needs_gpu").
        LiveBusy while a run is alive; FileNotFoundError without the video."""
        with self._locked():
            return self._start(video_path, video_id, tid_video)

    def _start(self, video_path: Path, video_id: str, tid_video: str | None) -> LiveSession:
        if self.status() is not None:
            raise LiveBusy("ライブ判定はすでに実行中です")
        _join_reapers()   # the last run's exit record is written before this run's is cleared
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"動画がありません: {video_path}")
        if not self._s.engine_dir.is_dir():
            raise EngineError(f"エンジンがセットアップされていません: {self._s.engine_dir} "
                              "(scripts/setup_pfb3x3_engine.sh)")
        root, engine = self.root, self._engine
        self._release_gpu()
        engine._write_entry(root, video_id, video_path, 0.0, None, None, None, False,
                            domain="sports", sport=DEMO_SPORT)
        session = self._new_session(video_id)
        argv = [str(engine.python), "-m", "pfb3x3.engine", "live", "--video-id", video_id,
                "--registry", str(root / "videos"), "--tracks", str(root / "tracks"),
                "--session", session, "--device", "cuda",
                "--tracker-python", str(engine.tracker_python),
                "--tid-python", str(engine.tid_python), "--weights", str(engine.weights)]
        if tid_video:
            argv += ["--tid-video", tid_video]
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.exit_record.unlink(missing_ok=True)
        self.stopped_record.unlink(missing_ok=True)
        with open(logs / "live.out", "w", encoding="utf-8") as out_f, \
                open(logs / "live.err", "w", encoding="utf-8") as err_f:
            try:
                proc = subprocess.Popen(argv, cwd=self._s.engine_dir, env=engine._env(root),
                                        stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            except OSError as e:
                raise EngineError(f"エンジンを起動できません ({argv[0]}): {e}") from e
        sess = LiveSession(video_id=video_id, session=session,
                           stream=root / "tracks" / video_id / "live" / session / "stream.jsonl",
                           pid=proc.pid,
                           started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        record = {"pid": proc.pid, "start_time": _start_time(proc.pid), "boot_id": _boot_id(),
                  "session": session,
                  "stream": str(sess.stream), "video_id": video_id,
                  "started_at": sess.started_at}
        _write_atomic(self.pidfile, json.dumps(record, ensure_ascii=False))
        reaper = threading.Thread(target=self._reap, args=(proc, record),
                                  name=f"live-reaper-{proc.pid}", daemon=True)
        with _REAPERS_LOCK:
            _REAPERS[proc.pid] = reaper
        reaper.start()
        log.info("ライブ判定を開始しました pid=%s session=%s", proc.pid, session)
        return sess

    def _release_gpu(self) -> None:
        """Ask the worker to release its warm shared server and, while a worker is running,
        wait up to release_wait_sec for its answer (the run starts either way). Settings
        without a shared server (SharedServer.for_settings None): nothing to ask or wait for."""
        from pfb_app.worker import shared_server   # noqa: PLC0415 - it imports this module
        if not shared_server.configured(self._s):
            return
        try:
            rid = request_gpu_release(self._s)
        except OSError:
            log.exception("共有サーバーの停止を頼めませんでした")
            return
        if not _worker_running(self._s):
            return
        deadline = time.monotonic() + self._release_wait
        while time.monotonic() < deadline:
            if release_acked(self._s, rid):
                log.info("ワーカーが共有サーバーを止めました（ライブ判定のため）")
                return
            time.sleep(0.25)
        log.info("ワーカーの返事がないまま、ライブ判定を始めます")

    def status(self) -> LiveSession | None:
        """The running live run, or None (no pidfile, a torn one, or a pid that is no longer
        the engine process we started)."""
        rec = _read_json(self.pidfile)
        try:
            pid = int(rec["pid"])
            sess = LiveSession(video_id=str(rec["video_id"]), session=str(rec["session"]),
                               stream=Path(rec["stream"]), pid=pid,
                               started_at=str(rec.get("started_at") or ""))
            start = rec.get("start_time")
        except (TypeError, KeyError, ValueError):
            return None
        boot = _boot_id()
        if boot is None or rec.get("boot_id") != boot:   # a pidfile of an earlier boot
            return None
        if not _ours(pid, None if start is None else str(start)):
            return None
        return sess

    def stop(self, expected_session: str | None = None) -> bool:
        """Stop the running live run: SIGTERM to its group, SIGKILL after stop_wait_sec.
        False (nothing signalled) when no run of ours is alive, or when `expected_session`
        is given and the running run is another one (a 「停止」 pressed for a run that has
        ended meanwhile never stops the run started after it)."""
        with self._locked():
            return self._stop(expected_session)

    def _stop(self, expected_session: str | None = None) -> bool:
        sess = self.status()
        if sess is None:
            self.pidfile.unlink(missing_ok=True)   # stale or absent: nothing to signal
            return False
        if expected_session is not None and sess.session != expected_session:
            log.info("停止の対象 %s は終わっています（実行中は %s）: 何もしません",
                     expected_session, sess.session)
            return False
        with _REAPERS_LOCK:
            reaper = _REAPERS.get(sess.pid)
        if reaper is not None and not reaper.is_alive():   # reaped already: pid may be reused
            return False
        log.info("ライブ判定を停止します pid=%s", sess.pid)
        start = _read_json(self.pidfile).get("start_time")
        _write_atomic(self.stopped_record, json.dumps({"session": sess.session, "pid": sess.pid}))
        _killpg(sess.pid, signal.SIGTERM)
        deadline = time.monotonic() + self._stop_wait
        while time.monotonic() < deadline and _group_alive(sess.pid):
            time.sleep(0.05)
        if _group_alive(sess.pid):
            log.warning("ライブ判定のプロセスグループ %s が SIGTERM 後も残っているため SIGKILL します",
                        sess.pid)
            # its live_run.py child, found while the engine still holds its pid (ppid)
            orphans = _orphans(sess.pid, None if start is None else str(start))
            _killpg(sess.pid, signal.SIGKILL)
            _kill_orphans(orphans)
        if reaper is not None:
            reaper.join(timeout=5)
        self._remove_pidfile(sess.pid)
        return True

    def stop_in_background(self, expected_session: str | None = None) -> threading.Thread:
        """stop(expected_session) in a daemon thread, for a page that must keep drawing while
        the engine stops (up to stop_wait_sec, more when it has to be killed). One stopper per
        live dir: while one is under way it is returned again. See stopping()."""
        key = self.dir.resolve()
        with _REAPERS_LOCK:
            thread = _STOPPERS.get(key)
            if thread is not None and thread.is_alive():
                return thread
            thread = threading.Thread(target=self._stop_quietly, args=(expected_session,),
                                      name="live-stopper", daemon=True)
            _STOPPERS[key] = thread
            thread.start()
        return thread

    def stopping(self) -> bool:
        """A stop_in_background() of this live dir is under way (in this app process)."""
        with _REAPERS_LOCK:
            thread = _STOPPERS.get(self.dir.resolve())
        return thread is not None and thread.is_alive()

    def _stop_quietly(self, expected_session: str | None) -> None:
        try:
            self.stop(expected_session)
        except Exception:   # noqa: BLE001 - a thread has no caller to raise to
            log.exception("ライブ判定を停止できませんでした")

    def last_outcome(self) -> Outcome | None:
        """How the last live run ended: "stopped" (by stop(), whatever its exit code), "ok",
        "needs_gpu" (exit 3) or "error"; None while it runs or when there was none."""
        if self.status() is not None:
            return None
        if self.stopped_record.exists():
            return "stopped"
        rec = _read_json(self.exit_record)
        if isinstance(rec.get("returncode"), int):
            return _outcome(rec["returncode"])
        summary = _last_json(self.root / "logs" / "live.out")
        if summary is None:
            return None
        error = summary.get("error")
        if error is None:
            return "ok"
        return "needs_gpu" if any(t in str(error) for t in _NEEDS_GPU_TEXTS) else "error"

    # --- internals ---

    def _new_session(self, video_id: str) -> str:
        """UTC %Y%m%dT%H%M%SZ (the engine's default), with _2, _3 .. when that session dir
        exists already (the engine refuses an existing session) or names the last run (it
        may have ended before its dir was made: a stop of that session must never match the
        next run)."""
        base = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        live = self.root / "tracks" / video_id / "live"
        last = {_read_json(r).get("session") for r in (self.exit_record, self.stopped_record)}
        session, k = base, 1
        while (live / session).exists() or session in last:
            k += 1
            session = f"{base}_{k}"
        return session

    def _reap(self, proc: subprocess.Popen, record: dict) -> None:
        rc = proc.wait()
        try:
            _write_atomic(self.exit_record, json.dumps({**record, "returncode": rc}))
            self._remove_pidfile(proc.pid)
        except OSError:
            log.exception("ライブ判定の終了を記録できませんでした")
        finally:
            with _REAPERS_LOCK:
                if _REAPERS.get(proc.pid) is threading.current_thread():
                    del _REAPERS[proc.pid]
        log.info("ライブ判定が終了しました pid=%s exit=%s", proc.pid, rc)

    def _remove_pidfile(self, pid: int) -> None:
        if _read_json(self.pidfile).get("pid") == pid:
            self.pidfile.unlink(missing_ok=True)


def _ours(pid: int, start: str | None) -> bool:
    """_is_our_group, also during our process's exec()s: Popen returns once the exec has
    begun, before the kernel has filled /proc/<pid>/cmdline, and a wrapper script execs once
    more. So while the leader with the recorded start time is alive (no zombie), an empty
    cmdline counts as ours too. The cmdline is read once, before the liveness check."""
    if start is None:
        return False
    cmd = _cmdline(pid)
    if _start_time(pid) == start and _pid_alive(pid):
        if "pfb3x3.engine" in cmd:
            return True
        if any(cmd) or _is_kthread(pid):
            return False
        age = _proc_age_s(pid)
        return age is not None and age <= EXEC_GRACE_SEC
    return _is_our_group(pid, start)


def _stat_fields(pid: int) -> list[str]:
    """/proc/<pid>/stat after the command name: fields[0] is field 3 (state)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return []


def _is_kthread(pid: int) -> bool:
    f = _stat_fields(pid)
    try:
        return bool(int(f[6]) & _PF_KTHREAD)   # field 9: flags
    except (IndexError, ValueError):
        return False


def _proc_age_s(pid: int) -> float | None:
    """Seconds since `pid` started (/proc/uptime − its start time); None when unknown."""
    try:
        start_ticks = int(_stat_fields(pid)[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def _boot_id() -> str | None:
    try:
        return _BOOT_ID.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _orphans(engine_pid: int, engine_start: str | None) -> list[tuple[int, str]]:
    """(pid, start time) of the engine run's live_run.py children: parent (stat field 4) the
    engine, cmdline with a `live_run.py` script and `--parent-pid <engine_pid>`, started no
    earlier than the engine. Called while the engine is still alive, so its pid is its own."""
    if engine_start is None:
        return []
    found = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(d.name)
        except ValueError:
            continue
        fields = _stat_fields(pid)
        if len(fields) < 2 or fields[1] != str(engine_pid):
            continue
        cmd = _cmdline(pid)
        if not any(Path(a).name == "live_run.py" for a in cmd):
            continue
        try:
            parent = cmd[cmd.index("--parent-pid") + 1]
        except (ValueError, IndexError):
            continue
        st = _start_time(pid)
        if parent == str(engine_pid) and st is not None and int(st) >= int(engine_start):
            found.append((pid, st))
    return found


def _kill_orphans(orphans: list[tuple[int, str]]) -> None:
    """SIGKILL each (pid, start time) collected by _orphans, once the engine was killed and
    so could not stop it. A process that cannot be signalled (another user's: an OSError) is
    logged, never raised."""
    for pid, st in orphans:
        log.warning("停止したライブ判定の子プロセス %s が残っているため SIGKILL します", pid)
        try:
            _sigkill_if(pid, st)
        except OSError as e:
            log.warning("ライブ判定の子プロセス %s を停止できません: %s", pid, e)


def _sigkill_if(pid: int, start: str) -> None:
    """SIGKILL `pid` only while it is the process with start time `start`: through a pidfd
    when the platform has one (the fd pins the process, so the check and the signal are about
    the same process), else a check right before os.kill."""
    pidfd_open = getattr(os, "pidfd_open", None)
    send = getattr(signal, "pidfd_send_signal", None)
    try:
        if pidfd_open is not None and send is not None:
            fd = pidfd_open(pid)
            try:
                if _start_time(pid) == start:
                    send(fd, signal.SIGKILL)
            finally:
                os.close(fd)
        elif _start_time(pid) == start:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass   # gone already


def _join_reapers(timeout: float = 5.0) -> None:
    """Wait for the reapers of runs that have ended (only one run is ever alive, and the
    caller has just found none)."""
    with _REAPERS_LOCK:
        reapers = list(_REAPERS.values())
    for r in reapers:
        r.join(timeout=timeout)


def _outcome(returncode: int) -> Outcome:
    """0 ok, 3 needs GPU, anything else an error (a user stop is "stopped" before this)."""
    if returncode == 0:
        return "ok"
    return "needs_gpu" if returncode == EXIT_NEEDS_GPU else "error"


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}
