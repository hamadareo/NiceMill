"""On-disk layout: data/app.db, data/worker.beat,
data/assignments/<aid>/submissions/<sid>/{frames/<camera>,pages}."""
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.config import Settings


def _ensure(path: Path, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def db_path(settings: Settings) -> Path:
    """The SQLite database file (data_dir is created if missing)."""
    return _ensure(settings.data_dir, True) / "app.db"


def worker_beat_path(settings: Settings) -> Path:
    """Touched by the running worker every few seconds (worker.main); its age tells the UI
    whether anyone is there to take queued work."""
    return settings.data_dir / "worker.beat"


def worker_beat_at(settings: Settings) -> datetime | None:
    """When the worker last showed it was alive (UTC), or None if it never did."""
    try:
        return datetime.fromtimestamp(worker_beat_path(settings).stat().st_mtime, timezone.utc)
    except OSError:
        return None


def assignment_dir(settings: Settings, aid: int, create: bool = True) -> Path:
    return _ensure(settings.data_dir / "assignments" / str(aid), create)


def submission_dir(settings: Settings, aid: int, sid: int, create: bool = True) -> Path:
    return _ensure(assignment_dir(settings, aid, create=False) / "submissions" / str(sid), create)


def frames_dir(settings: Settings, aid: int, sid: int, camera: str, create: bool = True) -> Path:
    return _ensure(submission_dir(settings, aid, sid, create=False) / "frames" / camera, create)


def pages_dir(settings: Settings, aid: int, sid: int, create: bool = True) -> Path:
    return _ensure(submission_dir(settings, aid, sid, create=False) / "pages", create)
