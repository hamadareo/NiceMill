"""Shared by the 「データの扱い」 tests: make a synthetic work look handed out at a given time."""
from datetime import datetime, timedelta


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


def hand_out(repo, aid: int, received: datetime) -> None:
    """Everything of the work happened before `received` (groups, steps and results two hours,
    approvals one hour before), and its handout PDF was received then."""
    before, approved = iso(received - timedelta(hours=2)), iso(received - timedelta(hours=1))
    groups = "(SELECT id FROM submissions WHERE assignment_id=?)"
    repo.conn.execute("UPDATE submissions SET created_at=? WHERE assignment_id=?", (before, aid))
    repo.conn.execute(f"UPDATE stage_runs SET created_at=? WHERE submission_id IN {groups}",
                      (before, aid))
    repo.conn.execute(f"UPDATE jobs SET created_at=? WHERE submission_id IN {groups}",
                      (before, aid))
    repo.conn.execute("UPDATE subjects SET approved_at=? WHERE approved_at IS NOT NULL AND"
                      f" submission_id IN {groups}", (approved, aid))
    repo.conn.execute("UPDATE assignments SET pdf_downloaded_at=? WHERE id=?",
                      (iso(received), aid))
