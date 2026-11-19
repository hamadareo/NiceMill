"""A work's own uploads are decided on ("no other work uses this file") and removed under the
same write lock: nothing can register a video that uses the file between the check and the
unlink (rel5 finding, fixed in rel7). Synthetic data only."""
import sqlite3
from pathlib import Path

from pfb_app.services import deletion
from pfb_app.storage import files
from test_delete_work import full_work, outside, video_assignment   # noqa: F401 (fixture)
from test_wizard_ui_basic import env                                  # noqa: F401 (fixture)


def lock_is_held(settings) -> bool:
    """A second connection cannot begin a write transaction right now."""
    other = sqlite3.connect(files.db_path(settings), timeout=0, isolation_level=None)
    try:
        other.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        return True
    other.execute("ROLLBACK")
    return False


def test_an_upload_is_removed_while_the_write_lock_is_held(env, outside, monkeypatch):
    settings, repo = env
    aid, _, ups = full_work(settings, repo, outside)
    seen = {}
    real = Path.unlink

    def watching(self, *a, **kw):
        if self == ups["own"]:
            seen["locked"] = lock_is_held(settings)
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", watching)
    result = deletion.delete_work(repo, settings, aid)
    assert seen == {"locked": True}
    assert not ups["own"].exists()
    assert result.uploads_removed >= 1


def test_a_failed_deletion_removes_no_upload(env, outside, monkeypatch):
    """The files go only after the rows: an error before the end leaves everything."""
    settings, repo = env
    aid, _, ups = full_work(settings, repo, outside)

    def broken(*a, **kw):
        raise RuntimeError("audit failed")
    monkeypatch.setattr(repo, "audit", broken)
    try:
        deletion.delete_work(repo, settings, aid)
    except RuntimeError:
        pass
    assert ups["own"].read_bytes() == b"own"
    assert any(a["id"] == aid for a in repo.list_assignments())


def test_the_work_folder_is_removed_after_the_rows_are_committed(env, outside, monkeypatch):
    settings, repo = env
    aid, _, _ = full_work(settings, repo, outside)
    seen = {}
    real = deletion._remove_work_folder

    def watching(settings_, aid_):
        seen["locked"] = lock_is_held(settings)
        seen["row"] = repo.conn.execute("SELECT COUNT(*) FROM assignments WHERE id=?",
                                        (aid_,)).fetchone()[0]
        return real(settings_, aid_)

    monkeypatch.setattr(deletion, "_remove_work_folder", watching)
    deletion.delete_work(repo, settings, aid)
    assert seen == {"locked": False, "row": 0}      # not held up, and committed
