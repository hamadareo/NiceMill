"""解説動画: the additive explainer_jobs table, its migration on an older database and the Repo
methods (synthetic rows only)."""
import sqlite3

import pytest

from pfb_app.storage.db import Repo, connect, init_schema


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def _group(repo: Repo):
    aid = repo.create_assignment("練習試合", "video", "試合", "", "standard", False)
    sid = repo.create_submission(aid, "前半", {})
    return aid, sid, repo.add_subject(sid, "Pa", "山田花子", None, {})


def test_migration_of_an_older_database_adds_the_table_and_keeps_every_row(tmp_path):
    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    conn = connect(db)
    init_schema(conn)
    aid, sid, subject = _group(Repo(conn))
    conn.close()
    raw = sqlite3.connect(db)         # as production (29fe29d): no explainer_jobs
    raw.execute("DROP TABLE explainer_jobs")
    raw.commit()
    raw.close()
    conn = connect(db)
    init_schema(conn)
    assert "explainer_jobs" in _tables(conn)
    repo = Repo(conn)
    assert repo.get_assignment(aid)["title"] == "練習試合"
    assert [s["id"] for s in repo.list_subjects(sid)] == [subject]
    job, new = repo.explainer_enqueue(sid, "person", subject, "s1", None, None)
    init_schema(conn)                 # a second start changes nothing
    assert repo.explainer_job(job)["status"] == "queued"
    conn.close()


def test_one_active_job_per_key_and_the_queue(repo):
    _, sid, subject = _group(repo)
    a, new_a = repo.explainer_enqueue(sid, "person", subject, "st", 7, "coach")
    b, new_b = repo.explainer_enqueue(sid, "person", subject, "st", 7, "coach")
    t, new_t = repo.explainer_enqueue(sid, "tracking", None, "tt", 7, "coach")
    assert (new_a, new_b, new_t) == (True, False, True) and a == b and t != a
    row = repo.explainer_job(a)
    assert (row["submission_id"], row["subject_id"], row["kind"], row["stamp"],
            row["session_id"], row["actor"]) == (sid, subject, "person", "st", 7, "coach")
    claimed = repo.explainer_claim()
    assert claimed["id"] == a and claimed["status"] == "running" and claimed["started_at"]
    repo.explainer_progress(a, 0.5, "場面 1 / 2")
    assert (repo.explainer_job(a)["progress"], repo.explainer_job(a)["note"]) == (0.5, "場面 1 / 2")
    assert repo.explainer_finish(a, "done")
    assert not repo.explainer_finish(a, "failed", "x")      # an ended job stays as it ended
    assert repo.explainer_job(a)["finished_at"]
    again, new = repo.explainer_enqueue(sid, "person", subject, "st2", 7, "coach")
    assert new and again != a
    assert repo.explainer_latest(sid, "person", subject)["id"] == again
    assert repo.explainer_latest(sid, "tracking", None)["id"] == t


def test_cancel_for_groups_and_fail_left_running(repo):
    _, sid, subject = _group(repo)
    a, _ = repo.explainer_enqueue(sid, "person", subject, "st", None, None)
    b, _ = repo.explainer_enqueue(sid, "tracking", None, "tt", None, None)
    repo.explainer_claim()
    assert repo.explainer_fail_running("止まりました") == 1
    assert repo.explainer_job(a)["status"] == "failed"
    assert repo.explainer_job(a)["error"] == "止まりました"
    assert repo.explainer_cancel([sid], "消しました") == 1
    assert repo.explainer_job(b)["status"] == "cancelled"
    assert repo.explainer_claim() is None


def test_rows_go_with_the_group(repo):
    aid, sid, subject = _group(repo)
    a, _ = repo.explainer_enqueue(sid, "person", subject, "st", None, None)
    repo.delete_assignment(aid)
    assert repo.explainer_job(a) is None
