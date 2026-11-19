"""「データの扱い」: the additive columns and table, their migration on a live-like database,
and the Repo methods (synthetic rows only)."""
import sqlite3

import pytest

from pfb_app.storage.db import Repo, connect, init_schema

NEW_COLUMNS = {"assignments": {"consent", "consent_note", "media_removed_at"},
               "subjects": {"erased_at"}}


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def _work(repo: Repo):
    aid = repo.create_assignment("練習試合", "video", "試合", "", "standard", False)
    sid = repo.create_submission(aid, "前半", {})
    return aid, sid, repo.add_subject(sid, "P1", "人1", None, {})


def test_a_new_database_has_the_columns_and_the_settings_table(repo):
    assert "app_settings" in _tables(repo.conn)
    for table, cols in NEW_COLUMNS.items():
        assert cols <= _columns(repo.conn, table)


def test_migration_of_an_older_database_keeps_every_row(tmp_path):
    """The deployed database (feat/growth) has none of this lane's columns or the table: the
    start adds them, keeps every row, and a second start changes nothing."""
    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    conn = connect(db)
    init_schema(conn)
    repo = Repo(conn)
    aid, sid, subject = _work(repo)
    repo.set_pdf_downloaded(aid)
    conn.close()
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE app_settings")
    for table, cols in NEW_COLUMNS.items():
        for col in cols:
            raw.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
    raw.commit()
    raw.close()
    conn = connect(db)
    init_schema(conn)
    init_schema(conn)
    assert "app_settings" in _tables(conn)
    for table, cols in NEW_COLUMNS.items():
        assert cols <= _columns(conn, table)
    repo = Repo(conn)
    work = repo.get_assignment(aid)
    assert work["title"] == "練習試合" and work["pdf_downloaded_at"]
    assert (work["consent"], work["consent_note"], work["media_removed_at"]) == (None, None, None)
    assert repo.get_subject(subject)["erased_at"] is None
    assert repo.get_setting("retention") is None
    assert len(list(db.parent.glob("app.db.pre-migrate-*"))) == 1
    conn.close()


def test_consent_is_stored_and_cleared(repo):
    aid, _, _ = _work(repo)
    repo.set_consent(aid, "unconfirmed", "保護者に確認中")
    work = repo.get_assignment(aid)
    assert (work["consent"], work["consent_note"]) == ("unconfirmed", "保護者に確認中")
    repo.set_consent(aid, None, "")
    work = repo.get_assignment(aid)
    assert (work["consent"], work["consent_note"]) == (None, None)
    with pytest.raises(KeyError):
        repo.set_consent(aid + 99, "policy", "")


def test_settings_are_upserted_and_audited(repo):
    assert repo.get_setting("retention") is None
    repo.set_setting("retention", {"days": 30, "all": False})
    repo.set_setting("retention", {"days": 90, "all": True})
    assert repo.get_setting("retention") == {"days": 90, "all": True}
    entry = repo.latest_audit("set_setting", "retention")
    assert entry["detail"] == {"days": 90, "all": True}


def test_a_broken_setting_reads_as_missing(repo):
    repo.conn.execute("INSERT INTO app_settings (key, value, updated_at) VALUES"
                      " ('retention', 'not json', 'x')")
    assert repo.get_setting("retention") is None
