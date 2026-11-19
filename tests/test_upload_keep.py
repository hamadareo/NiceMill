"""Release-2 fix wave: a finished resumable upload survives a failed registration, uploads are
admitted with headroom, open uploads' reservations count for every disk check, and the
worker sweeps finished files nobody claimed. Synthetic files only."""
import time
from collections import namedtuple
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.ops import disk
from pfb_app.preprocess.frames import VideoError
from pfb_app.services import chunked_upload as cu
from pfb_app.services import content_check
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.wizard import big_upload, step_drop

GB = disk.GB
Usage = namedtuple("Usage", "total used free")


def fake(free_bytes):
    return lambda _path: Usage(1000 * GB, 1000 * GB - free_bytes, free_bytes)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                    disk_warn_gb=50, disk_refuse_gb=15, max_upload_mb=40960)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    yield Repo(conn, settings)
    conn.close()


def finished(settings, data=b"not a video", name="試合.mp4"):
    token = cu.issue_token(settings)
    up = cu.create(settings, token, name, len(data))
    cu.append(settings, token, up.id, 0, data)
    return big_upload.finished(settings, [{"id": up.id, "token": token}])[0], token, up


# --- HIGH 1: never delete a finished upload when the registration fails -----------------------

def test_a_failed_registration_keeps_the_upload_claimable(settings):
    pick, token, up = finished(settings)
    path = cu.status(settings, token, up.id).path

    def register(paths):
        assert paths == [path]
        raise VideoError("動画として読み込めません")
    assert step_drop.drop_videos(settings, [pick], False, register) is None
    assert path.is_file()                                       # never unlinked
    again = big_upload.finished(settings, [{"id": up.id, "token": token}])
    assert [p.id for p in again] == [up.id]                     # still a pick: press again
    assert step_drop.drop_videos(settings, again, False, lambda paths: "ok") == "ok"
    assert path.is_file()
    with pytest.raises(cu.UnknownUpload):                       # claimed after success only
        cu.status(settings, token, up.id)


def test_a_real_registration_failure_keeps_the_upload(settings, repo):
    """The content-check registration of a file that is not a video: the work is not
    created, the upload stays (register_video_group's cleanup skips unclaimed uploads)."""
    pick, token, up = finished(settings)
    path = cu.status(settings, token, up.id).path
    created = step_drop.drop_videos(
        settings, [pick], False,
        lambda paths: content_check.start_work_from_videos(repo, settings, paths, "standard"))
    assert created is None and repo.list_assignments() == []
    assert path.is_file() and cu.status(settings, token, up.id).complete


def test_a_disk_refusal_at_registration_keeps_the_upload(settings, repo, monkeypatch):
    pick, token, up = finished(settings)
    path = cu.status(settings, token, up.id).path
    monkeypatch.setattr(disk, "_usage", fake(10 * GB))          # below the refuse level
    created = step_drop.drop_videos(
        settings, [pick], False,
        lambda paths: content_check.start_work_from_videos(repo, settings, paths, "standard"))
    assert created is None and path.is_file()
    assert cu.status(settings, token, up.id).complete


def test_the_kept_upload_note_is_plain():
    assert "残して" in step_drop.UPLOAD_KEPT and "もう一度" in step_drop.UPLOAD_KEPT
    for word in ("claim", "upload", "ID", "id"):
        assert word not in step_drop.UPLOAD_KEPT


def test_a_streamlit_copy_is_still_removed_on_failure(settings):
    """A copy the drop zone made itself (Streamlit's uploader) is ours to remove."""
    import io

    class F(io.BytesIO):
        name = "a.mp4"
    def register(paths):
        raise VideoError("だめ")
    saved_dir = settings.data_dir / "uploads"
    assert step_drop.drop_videos(settings, [F(b"abc")], False, register) is None
    assert not [p for p in saved_dir.glob("*") if p.is_file()]


# --- HIGH 1: admitted with headroom ------------------------------------------------------------

def test_an_upload_is_admitted_only_with_a_margin(settings, monkeypatch):
    t = cu.issue_token(settings)
    size = 3 * GB
    # exactly enough for the file above the level, but no margin: refused
    monkeypatch.setattr(cu, "free_bytes", lambda _p: int(15 * GB) + size)
    with pytest.raises(cu.UploadRefused) as e:
        cu.create(settings, t, "a.mp4", size)
    assert "ファイル約3.0GB" in str(e.value)                       # the file's size, not + margin
    monkeypatch.setattr(cu, "free_bytes",
                        lambda _p: int(15 * GB) + size + cu.ADMIT_MARGIN_BYTES)
    cu.create(settings, t, "a.mp4", size)


# --- MED 9: reservations count everywhere --------------------------------------------------------

def test_open_upload_reservations_count_in_ensure_room(settings, monkeypatch):
    monkeypatch.setattr(cu, "free_bytes", lambda _p: 100 * GB)
    t = cu.issue_token(settings)
    cu.create(settings, t, "a.mp4", 30 * GB)
    assert disk.reserved_bytes(settings) == 30 * GB
    disk.ensure_room(settings, 0, usage=fake(50 * GB))          # 50 - 30 = 20 >= 15
    with pytest.raises(disk.DiskLow):
        disk.ensure_room(settings, 0, usage=fake(40 * GB))      # 40 - 30 = 10 < 15
    with pytest.raises(disk.DiskLow):
        disk.ensure_room(settings, 6 * GB, usage=fake(50 * GB))
    disk.ensure_room(settings, 0, usage=fake(40 * GB), reserved=0)


def test_reserved_bytes_without_uploads_makes_no_folder(settings):
    assert disk.reserved_bytes(settings) == 0
    assert not (settings.data_dir / "uploads").exists()


def test_the_worker_holds_for_reserved_space(settings, monkeypatch):
    monkeypatch.setattr(cu, "free_bytes", lambda _p: 100 * GB)
    t = cu.issue_token(settings)
    cu.create(settings, t, "a.mp4", 30 * GB)
    assert disk.worker_state(settings, usage=fake(100 * GB)).level == "ok"
    state = disk.worker_state(settings, usage=fake(40 * GB))
    assert state.level == "refuse" and state.free_gb == pytest.approx(40)


def test_the_worker_default_disk_check_counts_reservations():
    import inspect

    from pfb_app.worker import main
    assert inspect.signature(main.worker_loop).parameters["disk_check"].default \
        is disk.worker_state


# --- MED 9: the worker sweeps unclaimed uploads ------------------------------------------------

def test_the_worker_sweeps_unclaimed_uploads(settings, monkeypatch):
    from pfb_app.worker import main
    calls = []
    monkeypatch.setattr(cu, "sweep", lambda s, **kw: calls.append(s) or 0)
    clock = iter([0.0, 10.0, main.UPLOAD_SWEEP_SEC + 1]).__next__
    last = {"t": None}
    for _ in range(3):
        main._sweep_uploads(settings, clock, last)
    assert len(calls) == 2                                      # first pass, then hourly


def test_the_upload_sweep_never_breaks_the_worker(settings, monkeypatch):
    from pfb_app.worker import main

    def boom(s, **kw):
        raise OSError("gone")
    monkeypatch.setattr(cu, "sweep", boom)
    main._sweep_uploads(settings, lambda: 0.0, {"t": None})     # logged, not raised


def test_the_sweep_keeps_a_finished_file_a_work_uses(settings, repo):
    """A registration that succeeded but whose claim was lost (a crash in between): after
    KEEP_SEC the sweep forgets the upload but keeps the file the work uses."""
    pick, token, up = finished(settings)
    path = cu.status(settings, token, up.id).path
    aid = repo.create_assignment("作業", "video", "", "", "standard", True)
    repo.create_submission(aid, "前半", {"cameras": {"cam1": str(path)}})
    other, otoken, oup = finished(settings, name="b.mp4")
    opath = cu.status(settings, otoken, oup.id).path
    later = time.time() + cu.KEEP_SEC + 10
    assert cu.sweep(settings, now=later) == 2
    assert path.is_file() and not opath.exists()
    with pytest.raises(cu.UnknownUpload):
        cu.status(settings, token, up.id)
