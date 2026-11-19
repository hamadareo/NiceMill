"""Low-disk guard (ops.disk) and rotating log files (ops.logs). Temp dirs only."""
import io
import logging
from collections import namedtuple

import pytest

from pfb_app.config import Settings
from pfb_app.ops import disk, logs
from pfb_app.services import intake
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.storage import files
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN

Usage = namedtuple("Usage", "total used free")
GB = disk.GB


def fake(free_gb, total_gb=1000):
    return lambda _path: Usage(total_gb * GB, (total_gb - free_gb) * GB, free_gb * GB)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                    disk_warn_gb=50, disk_refuse_gb=15)


def plain(text):
    assert not [w for w in (*SCHOOL_WORDS, *FORBIDDEN) if w in text], text


@pytest.mark.parametrize("free,level", [(500, "ok"), (49.9, "warn"), (14.9, "refuse")])
def test_disk_levels(settings, free, level):
    state = disk.disk_state(settings, usage=fake(free))
    assert state.level == level
    assert (state.message == "") == (level == "ok")
    if level != "ok":
        plain(state.message)
        assert "空き" in state.message and "削除" in state.message


def test_disk_state_works_before_data_dir_exists(settings):
    assert not settings.data_dir.exists()
    state = disk.disk_state(settings)   # the real file system of the temp dir's parent
    assert state.total_gb > 0 and state.level in ("ok", "warn", "refuse")


def test_unreadable_disk_is_unknown_not_refused(settings):
    def broken(_path):
        raise OSError("gone")
    assert disk.disk_state(settings, usage=broken).level == "unknown"
    disk.ensure_room(settings, 10 * GB, usage=broken)   # no exception


def test_ensure_room(settings):
    disk.ensure_room(settings, 4 * GB, usage=fake(20))          # 16 GB left: fine
    with pytest.raises(disk.DiskLow) as too_big:
        disk.ensure_room(settings, 6 * GB, usage=fake(20))      # 14 GB left: refused
    assert "このファイル" in str(too_big.value)
    with pytest.raises(disk.DiskLow) as full:
        disk.ensure_room(settings, 0, usage=fake(10))
    assert "受け付けていません" in str(full.value)
    for e in (too_big.value, full.value):
        plain(str(e))


def test_disk_low_is_a_video_error_the_upload_screens_show():
    from pfb_app.preprocess.frames import VideoError
    assert issubclass(disk.DiskLow, VideoError)


def test_save_uploaded_video_refuses_before_writing(settings, monkeypatch):
    monkeypatch.setattr(disk, "_usage", fake(16))
    data = io.BytesIO(b"x" * 1024)
    data.size = 2 * GB   # what Streamlit's UploadedFile reports
    with pytest.raises(disk.DiskLow):
        intake.save_uploaded_video(settings, "試合.mp4", data)
    assert not list((settings.data_dir / "uploads").glob("*"))
    monkeypatch.setattr(disk, "_usage", fake(500))
    path = intake.save_uploaded_video(settings, "試合.mp4", io.BytesIO(b"abc"))
    assert path.read_bytes() == b"abc"


def test_register_documents_refuses_every_file_when_full(settings, monkeypatch):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("作業", "document", "説明", "観点", "standard", False)
    monkeypatch.setattr(disk, "_usage", fake(1))
    report = intake.register_documents(repo, settings, aid, [("a.pdf", b"%PDF"), ("b.pdf", b"x")],
                                       None)
    assert [n for n, _ in report.errors] == ["a.pdf", "b.pdf"]
    assert not report.created and not repo.list_submissions(aid)


def test_upload_size():
    assert disk.upload_size(io.BytesIO(b"12345")) == 5
    f = io.BytesIO(b"12345")
    f.read(2)
    assert disk.upload_size(f) == 3 and f.tell() == 2


# --- log files ------------------------------------------------------------------------------

@pytest.fixture
def logger():
    lg = logging.getLogger("pfb_test_ops_logs")
    lg.propagate = False
    yield lg
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def test_log_file_is_written_and_setup_is_idempotent(settings, logger):
    path = logs.setup(settings, "worker", console=False, logger=logger)
    assert path == settings.data_dir / "logs" / "worker.log"
    logs.setup(settings, "worker", console=False, logger=logger)
    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1
    other = Settings(data_dir=settings.data_dir.parent / "other")
    logs.setup(other, "worker", console=False, logger=logger)   # replaced, not added
    assert [h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)] == [
        str(other.data_dir / "logs" / "worker.log")]
    logs.setup(settings, "worker", console=False, logger=logger)
    logger.error("止まりました job=1")
    logger.info("ふつうの記録")
    for h in logger.handlers:
        h.flush()
    text = path.read_text(encoding="utf-8")
    assert "止まりました job=1" in text and "ふつうの記録" in text
    assert logs.recent_lines(settings, "worker") == [
        line for line in text.splitlines() if " ERROR " in line]


def test_log_files_rotate(tmp_path, logger):
    settings = Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs", log_max_mb=1,
                        log_keep=2)
    logs.setup(settings, "session", console=False, logger=logger)
    line = "x" * 1000
    for _ in range(3000):   # ~3 MB through a 1 MB file
        logger.info(line)
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == ["session.log", "session.log.1", "session.log.2"]
    assert all(p.stat().st_size <= 1024 * 1024 + 2000 for p in (tmp_path / "logs").iterdir())


def test_without_settings_only_the_console(logger):
    assert logs.setup(None, "worker", logger=logger) is None
    assert all(not isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_worker_setup_logging_adds_the_file(settings, monkeypatch):
    from pfb_app.worker import session
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        session.setup_logging(settings, "worker")
        added = [h for h in root.handlers if h not in before]
        assert any(getattr(h, "baseFilename", "").endswith("worker.log") for h in added)
    finally:
        for h in [h for h in root.handlers if h not in before]:
            root.removeHandler(h)
            h.close()
