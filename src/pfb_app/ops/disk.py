"""Free space where data_dir lives: warn early, refuse new uploads before the disk fills.

- `disk_state(settings)`: free / total GB and a level: "ok", "warn" (below disk_warn_gb) or
  "refuse" (below disk_refuse_gb). Every screen shows the warning (ui/app.py); the worker
  starts no new step while the level is "refuse" (worker.main), so frames and engine outputs
  never fill the disk, and the SQLite database always has room to write.
- `ensure_room(settings, incoming_bytes)`: raises DiskLow (a VideoError, so the upload
  screens show its plain sentence) when saving that much would leave less than
  disk_refuse_gb free. The bytes still to come of open resumable uploads
  (services.chunked_upload.reserved_bytes) count as used: the intake, the proxies and the
  worker (`worker_state`) never take the room an upload was promised.
"""
import shutil
from dataclasses import dataclass
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.preprocess.frames import VideoError

GB = 1024 ** 3

WARN = ("保存場所の空きが少なくなっています（残り約{free}GB）。"
        "使い終わった作業を削除すると、空きが増えます。")
REFUSE = ("保存場所の空きが足りないため、新しい動画や書類を受け付けていません（残り約{free}GB）。"
          "使い終わった作業を削除してから、もう一度入れてください。")
TOO_BIG = ("保存場所の空きが足りないため、このファイルを受け付けられません"
           "（ファイル約{need}GB、残り約{free}GB）。"
           "使い終わった作業を削除してから、もう一度入れてください。")
HOLD = "保存場所の空きが足りないため、新しい解析を始めずに待っています。"


class DiskLow(VideoError):
    """Not enough free space for a new upload (the message is a plain sentence)."""


@dataclass(frozen=True)
class DiskState:
    path: Path
    free_gb: float
    total_gb: float
    level: str          # "ok" | "warn" | "refuse" | "unknown"

    @property
    def message(self) -> str:
        """The plain sentence for the screen ("" when there is enough room)."""
        if self.level == "refuse":
            return REFUSE.format(free=_gb(self.free_gb))
        if self.level == "warn":
            return WARN.format(free=_gb(self.free_gb))
        return ""


def _gb(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def _existing(path: Path) -> Path:
    """`path`, or its nearest existing parent (data_dir may not exist yet)."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _usage(path: Path):
    return shutil.disk_usage(_existing(path))


def disk_state(settings: Settings, usage=None) -> DiskState:
    """The free space of data_dir's file system. `usage`: shutil.disk_usage (tests)."""
    usage = usage or _usage
    try:
        u = usage(settings.data_dir)
    except OSError:
        return DiskState(settings.data_dir, 0.0, 0.0, "unknown")
    free, total = u.free / GB, u.total / GB
    level = ("refuse" if free < settings.disk_refuse_gb
             else "warn" if free < settings.disk_warn_gb else "ok")
    return DiskState(settings.data_dir, free, total, level)


def reserved_bytes(settings: Settings) -> int:
    """The bytes still to come of every open resumable upload (0 when unreadable)."""
    from pfb_app.services import chunked_upload   # noqa: PLC0415 - no import cycle at load
    try:
        return chunked_upload.reserved_bytes(settings)
    except OSError:
        return 0


def ensure_room(settings: Settings, incoming_bytes: int = 0, usage=None,
                reserved: int | None = None, margin_bytes: int = 0) -> None:
    """DiskLow when `incoming_bytes` more (plus `margin_bytes`) would leave less than
    disk_refuse_gb free, counting `reserved` (default: the open uploads' bytes still to come)
    as used. An unreadable file system is not refused (the save itself then reports what is
    wrong)."""
    usage = usage or _usage
    try:
        free = usage(settings.data_dir).free
    except OSError:
        return
    if reserved is None:
        reserved = reserved_bytes(settings)
    left = free - max(0, int(reserved))
    incoming = max(0, int(incoming_bytes))
    if left - incoming - max(0, int(margin_bytes)) >= settings.disk_refuse_gb * GB:
        return
    shown = max(0, left)
    if incoming and left >= settings.disk_refuse_gb * GB:
        raise DiskLow(TOO_BIG.format(need=_gb(incoming / GB), free=_gb(shown / GB)))
    raise DiskLow(REFUSE.format(free=_gb(shown / GB)))


def worker_state(settings: Settings, usage=None) -> DiskState:
    """disk_state for the worker: also "refuse" when the open uploads' promised bytes would
    take the free space below disk_refuse_gb (ensure_room), so no new step eats into it."""
    state = disk_state(settings, usage)
    if state.level in ("ok", "warn"):
        try:
            ensure_room(settings, 0, usage=usage or _usage)
        except DiskLow:
            return DiskState(state.path, state.free_gb, state.total_gb, "refuse")
    return state


def upload_size(fileobj) -> int:
    """The size of an upload (Streamlit's UploadedFile has .size; else seek to the end)."""
    size = getattr(fileobj, "size", None)
    if isinstance(size, int):
        return size
    try:
        pos = fileobj.tell()
        fileobj.seek(0, 2)
        end = fileobj.tell()
        fileobj.seek(pos)
        return max(0, end - pos)
    except (AttributeError, OSError, ValueError):
        return 0
