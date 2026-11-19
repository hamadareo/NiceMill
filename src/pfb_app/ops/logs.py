"""Rotating log files beside the console output (which tmux or the journal keeps).

Each process kind writes its own file in log_dir (default <data_dir>/logs), so no two
processes ever rotate the same file: worker.log (the resident worker), session.log (its
inference sessions: one at a time), ui.log (pfb_app's own messages in the Streamlit
process), backup.log (the nightly backup). Each file rotates at log_max_mb and keeps
log_keep old files (worker.log.1 …).
"""
import logging
import logging.handlers
import sys
from pathlib import Path

from pfb_app.config import Settings

FORMAT = "%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s"
_MARK = "_pfb_log_file"   # set on handlers this module added (setup is idempotent)


def log_dir(settings: Settings) -> Path:
    return settings.log_dir if settings.log_dir is not None else settings.data_dir / "logs"


def log_path(settings: Settings, name: str) -> Path:
    return log_dir(settings) / f"{name}.log"


def file_handler(settings: Settings, name: str) -> logging.Handler:
    path = log_path(settings, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max(1, settings.log_max_mb) * 1024 * 1024,
        backupCount=max(1, settings.log_keep), encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    setattr(handler, _MARK, (name, str(path)))
    return handler


def setup(settings: Settings | None, name: str, *, console: bool = True,
          logger: logging.Logger | None = None) -> Path | None:
    """INFO logging to stdout (when `console`) and to <log_dir>/<name>.log. Calling it again
    for the same file adds nothing; for another file of the same name (other settings) the
    old handler is replaced, so a process never holds more than one file per name. Without
    settings (or when the folder cannot be made) only the console is set up. Returns the
    file's path, or None."""
    root = logger or logging.getLogger()
    root.setLevel(logging.INFO)
    if console and not any(isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.FileHandler) for h in root.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(stream)
    if settings is None:
        return None
    path = log_path(settings, name)
    for h in list(root.handlers):
        mark = getattr(h, _MARK, None)
        if mark == (name, str(path)):
            return path
        if mark is not None and mark[0] == name:   # the settings changed: one file per name
            root.removeHandler(h)
            h.close()
    try:
        root.addHandler(file_handler(settings, name))
    except OSError:
        root.exception("ログファイルを開けませんでした: %s", path)
        return None
    return path


def setup_ui(settings: Settings) -> Path | None:
    """pfb_app's own loggers in the Streamlit process → ui.log (Streamlit's root logger and
    console stay as Streamlit set them up)."""
    return setup(settings, "ui", console=False, logger=logging.getLogger("pfb_app"))


def recent_lines(settings: Settings, name: str, levels=("ERROR", "CRITICAL"),
                 limit: int = 20, max_bytes: int = 2 * 1024 * 1024) -> list[str]:
    """The last `limit` lines of <name>.log at these levels (read from the file's tail)."""
    path = log_path(settings, name)
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    wanted = tuple(f" {lv} " for lv in levels)
    lines = [line for line in text.splitlines() if any(w in line for w in wanted)]
    return lines[-limit:]
