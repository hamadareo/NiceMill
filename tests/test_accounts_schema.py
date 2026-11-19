"""Accounts: the additive tables and columns, their migration on a pre-accounts database (the
live rel3 one), org 1 everywhere, and the actor in the audit log (synthetic rows only)."""
import sqlite3

import pytest

from pfb_app.storage.db import Repo, connect, init_schema

NEW_TABLES = {"orgs", "users", "auth_sessions", "invites", "login_failures"}
NEW_COLUMNS = {"assignments": {"org_id"}, "growth_people": {"org_id"},
               "audit_log": {"org_id", "actor_id", "actor"}}


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def test_a_new_database_has_the_tables_columns_and_org_1(repo):
    assert NEW_TABLES <= _tables(repo.conn)
    for table, cols in NEW_COLUMNS.items():
        assert cols <= _columns(repo.conn, table)
    for table in ("users", "auth_sessions", "invites"):
        assert "org_id" in _columns(repo.conn, table)
    assert [tuple(r) for r in repo.conn.execute("SELECT id FROM orgs")] == [(1,)]
    aid = repo.create_assignment("練習", "video", "試合", "", "standard", False)
    assert repo.conn.execute("SELECT org_id FROM assignments WHERE id=?", (aid,)).fetchone()[0] == 1


def test_migration_of_a_pre_accounts_database_keeps_rows_and_reads_org_1(tmp_path):
    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    conn = connect(db)
    init_schema(conn)
    repo = Repo(conn)
    aid = repo.create_assignment("練習試合", "video", "試合", "", "standard", False)
    pid = repo.create_growth_person("人A")
    repo.audit("rename", f"assignment:{aid}", {"title": "練習試合"})
    conn.close()
    raw = sqlite3.connect(db)
    for table in NEW_TABLES:
        raw.execute(f"DROP TABLE {table}")
    for table, cols in NEW_COLUMNS.items():
        for col in cols:
            raw.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
    raw.commit()
    raw.close()
    conn = connect(db)
    init_schema(conn)
    init_schema(conn)   # a second start changes nothing
    assert NEW_TABLES <= _tables(conn)
    for table, cols in NEW_COLUMNS.items():
        assert cols <= _columns(conn, table)
    assert tuple(conn.execute("SELECT org_id, title FROM assignments WHERE id=?",
                              (aid,)).fetchone()) == (1, "練習試合")
    assert conn.execute("SELECT org_id FROM growth_people WHERE id=?", (pid,)).fetchone()[0] == 1
    row = conn.execute("SELECT org_id, actor_id, actor FROM audit_log").fetchone()
    assert tuple(row) == (1, None, None)
    assert [tuple(r) for r in conn.execute("SELECT id FROM orgs")] == [(1,)]
    assert len(list(db.parent.glob("app.db.pre-migrate-*"))) == 1
    conn.close()


def test_the_actor_is_written_with_every_audit_entry(repo):
    repo.audit("plain", None, {})
    repo.actor = (7, "yamada")
    repo.audit("rename", "assignment:1", {"title": "x"})
    rows = [tuple(r) for r in repo.conn.execute(
        "SELECT action, actor_id, actor FROM audit_log ORDER BY id")]
    assert rows == [("plain", None, None), ("rename", 7, "yamada")]
