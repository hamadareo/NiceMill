"""Process guards for the worker under tmux or systemd.

- WorkerLock: one worker per data_dir. An exclusive flock on <data_dir>/worker.lock, held for
  the worker's life and released by the kernel when the process ends however it ends, so a
  second worker (tmux and systemd both started, or a double start) exits at once instead of
  running GPU sessions beside the first.
- RunningMarker: <data_dir>/worker.running exists while the worker runs and is removed when
  it leaves through its finally (also on SIGTERM / SIGHUP: worker.session.exit_on_signals).
  Finding it at start means the previous worker ended without cleanup (power cut, SIGKILL,
  OOM killer): `unclean` is then True and the worker tells the operator.
- lock_held / db_open_by: read-only probes for the install preflight and restore.
"""
import fcntl
import os
from pathlib import Path


class AlreadyRunning(Exception):
    pass


def lock_held(path: Path) -> bool:
    """True when a process holds the worker lock at `path`. A read-only probe: it never
    creates, truncates or writes the file (the install and restore checks run against live
    data)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class WorkerLock:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> "WorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                holder = os.pread(fd, 32, 0).decode("ascii", "replace").strip()
            except OSError:
                holder = ""
            os.close(fd)
            raise AlreadyRunning(f"{self.path}（pid {holder or '?'}）") from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "WorkerLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


class RunningMarker:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.unclean = False

    def __enter__(self) -> "RunningMarker":
        self.unclean = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()}\n", encoding="ascii")
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def db_open_by(db: Path) -> list[int]:
    """Pids of other processes (that this user may inspect) with the database, or its -wal /
    -shm file, open: /proc/*/fd links. [] without /proc (macOS) or without the file. A
    read-only scan: the live app's UI and worker keep app.db open while they run."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    try:
        target = str(Path(db).resolve(strict=True))
    except OSError:
        return []
    names = {target, target + "-wal", target + "-shm"}
    me, found = os.getpid(), []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            fds = os.listdir(entry / "fd")
        except OSError:   # another user's process, or gone meanwhile
            continue
        for fd in fds:
            try:
                if os.readlink(entry / "fd" / fd) in names:
                    found.append(int(entry.name))
                    break
            except OSError:
                continue
    return sorted(found)
