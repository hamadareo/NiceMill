"""成長の記録: the additive tables and column, their migration on a live-like database, and
the Repo methods (synthetic rows only)."""
import sqlite3

import pytest

from pfb_app.storage.db import Repo, connect, init_schema

GROWTH_TABLES = ("growth_people", "growth_links", "growth_declined")


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def _people(repo: Repo, n=2):
    aid = repo.create_assignment("練習試合", "video", "試合", "", "standard", False)
    sid = repo.create_submission(aid, "前半", {})
    return aid, sid, [repo.add_subject(sid, f"P{k}", f"人{k}", None, {}) for k in range(n)]


def test_a_new_database_has_the_growth_tables_and_column(repo):
    assert set(GROWTH_TABLES) <= _tables(repo.conn)
    assert "growth_group" in _columns(repo.conn, "assignments")


def test_migration_of_an_older_database_keeps_every_row(tmp_path):
    """The deployed database (rel2 1dece72) has none of the growth tables or the column: the
    start adds them, keeps every row, copies the file once before the column is added, and a
    second start changes nothing."""
    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    conn = connect(db)
    init_schema(conn)
    repo = Repo(conn)
    aid, sid, subjects = _people(repo)
    conn.close()
    raw = sqlite3.connect(db)     # as the deployed schema: without anything of this lane
    for table in GROWTH_TABLES:
        raw.execute(f"DROP TABLE {table}")
    raw.execute("ALTER TABLE assignments DROP COLUMN growth_group")
    raw.commit()
    raw.close()
    conn = connect(db)
    init_schema(conn)
    assert set(GROWTH_TABLES) <= _tables(conn)
    assert "growth_group" in _columns(conn, "assignments")
    repo = Repo(conn)
    assert repo.get_assignment(aid)["title"] == "練習試合"
    assert repo.get_assignment(aid)["growth_group"] is None
    assert [s["id"] for s in repo.list_subjects(sid)] == subjects
    copies = list(db.parent.glob("app.db.pre-migrate-*"))
    assert len(copies) == 1
    old = sqlite3.connect(copies[0])
    assert "growth_group" not in _columns(old, "assignments")
    old.close()
    init_schema(conn)
    assert len(list(db.parent.glob("app.db.pre-migrate-*"))) == 1
    conn.close()


def test_link_unlink_and_the_record_left_empty(repo):
    _, _, (a, b) = _people(repo)
    pid = repo.create_growth_person("人0")
    repo.link_growth(a, pid, "manual")
    repo.link_growth(b, pid, "suggested")
    assert repo.growth_person(pid)["name"] == "人0"
    assert {(r["subject_id"], r["how"]) for r in repo.growth_links(pid)} == {
        (a, "manual"), (b, "suggested")}
    assert repo.growth_person_of(a) == pid
    repo.unlink_growth(a)
    assert repo.growth_person_of(a) is None
    repo.unlink_growth(b)                       # the last link: the record goes too
    assert repo.growth_person(pid) is None


def test_link_moves_a_subject_to_another_record(repo):
    _, _, (a, _) = _people(repo)
    one, two = repo.create_growth_person("一"), repo.create_growth_person("二")
    repo.link_growth(a, one, "manual")
    repo.link_growth(a, two, "manual")
    assert repo.growth_person_of(a) == two
    assert repo.growth_person(one) is None      # left empty: removed


def test_declines_are_one_pair_in_either_order(repo):
    _, _, (a, b) = _people(repo)
    repo.decline_growth(b, a)
    repo.decline_growth(a, b)
    assert repo.growth_declined() == {(min(a, b), max(a, b))}


def test_deleting_a_work_drops_its_links_and_declines(repo):
    aid, _, (a, b) = _people(repo)
    pid = repo.create_growth_person("人0")
    repo.link_growth(a, pid, "manual")
    repo.decline_growth(a, b)
    repo.conn.execute("DELETE FROM assignments WHERE id=?", (aid,))
    assert repo.growth_links(pid) == []
    assert repo.growth_declined() == set()
    assert [p["id"] for p in repo.growth_people()] == []   # empty records are not listed


def test_rename_and_the_group_setting(repo):
    aid, _, (a, _) = _people(repo)
    pid = repo.create_growth_person("#5")
    repo.link_growth(a, pid, "manual")
    repo.rename_growth_person(pid, "山田")
    assert repo.growth_person(pid)["name"] == "山田"
    repo.set_growth_group(aid, "U12 A")
    assert repo.get_assignment(aid)["growth_group"] == "U12 A"
    repo.set_growth_group(aid, "  ")
    assert repo.get_assignment(aid)["growth_group"] is None


def test_deleting_a_work_from_home_removes_records_left_empty(tmp_path):
    """Review MED6: the person's name does not stay behind once their last work is gone."""
    from pfb_app.config import Settings
    from pfb_app.services import deletion
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    conn = connect(tmp_path / "app2.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid, _, (a, _) = _people(repo)
    keep_aid, _, (k, _) = _people(repo)
    gone = repo.create_growth_person("人0")
    repo.link_growth(a, gone, "manual")
    kept = repo.create_growth_person("人0")
    repo.link_growth(k, kept, "manual")
    deletion.delete_work(repo, settings, aid)
    assert repo.growth_person(gone) is None and repo.growth_person(kept) is not None
    repo.delete_assignment(keep_aid)
    assert repo.growth_person(kept) is None


def test_the_delete_confirm_says_the_links_go_too():
    from pfb_app.ui.wizard import home
    assert "成長の記録のつながり" in home.WHAT_GOES
