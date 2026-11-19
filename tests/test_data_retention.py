"""「動画を自動で消す」: the organisation-wide setting, when a work counts as finished, and the
card's line (synthetic rows only; `now` is fixed)."""
from datetime import datetime, timedelta, timezone

import pytest

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.services import retention
from pfb_app.storage.db import Repo, connect, init_schema
from data_helpers import hand_out, iso
from test_wizard_ui_people_wait import save

NOW = datetime(2026, 10, 31, 3, 0, tzinfo=timezone.utc)
THEN = datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def finished_work(repo, *, received: datetime, approved: datetime | None = None,
                  kind="video", title="練習試合"):
    """A work with one approved person whose handout PDF was received at `received`."""
    aid = repo.create_assignment(title, kind, "試合", "", "standard", False)
    sid = repo.create_submission(aid, "前半", {})
    subj = repo.add_subject(sid, "P1", "人1", None, {})
    run = save(repo, sid, Stage.pfb, {"pfb": "よかった。"}, subject_id=subj)
    repo.set_approval(subj, run.id)
    hand_out(repo, aid, received)
    if approved is not None:
        repo.conn.execute("UPDATE subjects SET approved_at=? WHERE id=?", (iso(approved), subj))
    return aid, sid, subj


def test_the_setting_is_off_until_saved_and_checked(repo):
    assert retention.load(repo) == retention.Policy(0, False)
    assert not retention.load(repo).on
    assert retention.save(repo, 90, False) == retention.Policy(90, False)
    assert retention.load(repo).on
    assert retention.save(repo, 30, True) == retention.Policy(30, True)
    with pytest.raises(retention.RetentionError):
        retention.save(repo, 45, False)
    assert retention.save(repo, 0, True) == retention.Policy(0, False)   # off: nothing to all
    repo.set_setting(retention.KEY, {"days": "x"})
    assert retention.load(repo) == retention.Policy(0, False)


def test_finished_means_received_after_the_last_approval(repo):
    received = NOW - timedelta(days=40)
    aid, _, subj = finished_work(repo, received=received)
    assert retention.finished_at(repo, aid) == received
    repo.conn.execute("UPDATE subjects SET approved_at=? WHERE id=?",
                      (iso(received + timedelta(minutes=5)), subj))
    assert retention.finished_at(repo, aid) is None      # approved again: not handed out yet
    other = repo.create_assignment("未完", "video", "", "", "standard", False)
    assert retention.finished_at(repo, other) is None    # never received


def test_a_coach_label_press_is_activity(repo):
    """Final review LOW 8: a 判定支援 label pressed after the handout was received means the
    work is still in use: not finished, so nothing is removed."""
    received = NOW - timedelta(days=40)
    aid, sid, _ = finished_work(repo, received=received)
    repo.set_linecall_label(sid, "cam1", "P1-arc-1", "correct", verdict="clear")
    repo.conn.execute("UPDATE linecall_labels SET labeled_at=?",
                      (iso(received - timedelta(minutes=5)),))
    assert retention.finished_at(repo, aid) == received      # pressed before: finished
    repo.conn.execute("UPDATE linecall_labels SET labeled_at=?",
                      (iso(received + timedelta(minutes=5)),))
    assert retention.finished_at(repo, aid) is None          # pressed after: in use


def test_card_lines(repo):
    received = datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)
    aid, _, _ = finished_work(repo, received=received)
    doc, _, _ = finished_work(repo, received=received, kind="document", title="報告書")
    waiting = repo.create_assignment("途中", "video", "", "", "standard", False)
    off = retention.Policy(0, False)
    assert retention.card_line(repo, repo.get_assignment(aid), off, THEN) is None
    media = retention.Policy(30, False)
    assert retention.card_line(repo, repo.get_assignment(aid), media, THEN) == \
        "動画は10月31日に消えます"
    assert retention.card_line(repo, repo.get_assignment(doc), media, THEN) == \
        "取り込んだ書類は10月31日に消えます"
    assert retention.card_line(repo, repo.get_assignment(waiting), media, THEN) == \
        "PDFを受け取ってから30日後に、動画が消えます"
    everything = retention.Policy(90, True)
    assert retention.card_line(repo, repo.get_assignment(aid), everything, THEN) == \
        "この作業は12月30日に消えます"
    assert retention.card_line(repo, repo.get_assignment(waiting), everything, THEN) == \
        "PDFを受け取ってから90日後に、この作業が消えます"
    assert retention.card_line(repo, repo.get_assignment(aid), media, NOW) == \
        "動画は、まもなく消えます"      # the next hourly pass
    late = retention.Policy(180, False)
    assert retention.card_line(repo, repo.get_assignment(aid), late, THEN) == \
        "動画は2027年3月30日に消えます"
    repo.conn.execute("UPDATE assignments SET media_removed_at=? WHERE id=?",
                      (iso(THEN - timedelta(days=1)), aid))
    for policy in (off, media):
        assert retention.card_line(repo, repo.get_assignment(aid), policy, THEN) == \
            "動画は10月19日に消しました。文章とPDFは残っています"


def test_due_works_skip_active_jobs_and_pending_deletions(repo):
    old = NOW - timedelta(days=31)
    a, _, _ = finished_work(repo, received=old, title="A")
    b, sid_b, _ = finished_work(repo, received=old, title="B")
    c, _, _ = finished_work(repo, received=NOW - timedelta(days=29), title="C")
    d, sid_d, _ = finished_work(repo, received=old, title="D")
    repo.enqueue(sid_b, Stage.pfb, "cpu")
    repo.conn.execute("UPDATE jobs SET created_at=?", (iso(old - timedelta(hours=3)),))
    repo.conn.execute("UPDATE submissions SET meta=json_set(meta, '$.delete_requested',"
                      " json('true')) WHERE id=?", (sid_d,))
    policy = retention.Policy(30, False)
    assert retention.due(repo, policy, NOW) == [a]
    assert retention.preview(repo, policy, NOW) == 2   # A, and B once its job ends; D is being deleted
    repo.conn.execute("UPDATE jobs SET status=?", (JobStatus.done.value,))
    assert retention.due(repo, policy, NOW) == [a, b]
    repo.conn.execute("UPDATE assignments SET media_removed_at=? WHERE id=?", (iso(NOW), a))
    assert retention.due(repo, policy, NOW) == [b]
    assert retention.due(repo, retention.Policy(30, True), NOW) == [a, b]   # すべて消す
    assert retention.due(repo, retention.Policy(0, False), NOW) == []
    assert c not in retention.due(repo, policy, NOW + timedelta(hours=23))
    assert c in retention.due(repo, policy, NOW + timedelta(days=2))


# --- anything after the handout re-opens the work (review of 2026-10-23, HIGH 2 and 3) ---

def _later(received):
    return iso(received + timedelta(minutes=5))


def test_a_group_added_after_the_handout_reopens_the_work(repo):
    received = NOW - timedelta(days=40)
    aid, _, _ = finished_work(repo, received=received)
    sid = repo.create_submission(aid, "後半", {"check": "pending"})   # no job, no run yet
    assert retention.finished_at(repo, aid) is None
    assert retention.due(repo, retention.Policy(30, False), NOW) == []
    repo.conn.execute("UPDATE submissions SET created_at=? WHERE id=?",
                      (iso(received - timedelta(hours=1)), sid))
    assert retention.finished_at(repo, aid) == received


def test_an_approval_taken_back_after_the_handout_reopens_the_work(repo):
    from pfb_app.services import review
    received = NOW - timedelta(days=40)
    aid, _, subj = finished_work(repo, received=received)
    review.unapprove(repo, subj)          # clears approved_at; only its audit entry is left
    assert retention.finished_at(repo, aid) is None
    assert retention.due(repo, retention.Policy(30, True), NOW) == []


def test_a_new_result_or_step_after_the_handout_reopens_the_work(repo):
    received = NOW - timedelta(days=40)
    aid, sid, subj = finished_work(repo, received=received)
    run = save(repo, sid, Stage.pfb, {"pfb": "書き直し"}, subject_id=subj)   # not approved yet
    assert retention.finished_at(repo, aid) is None
    repo.conn.execute("UPDATE stage_runs SET created_at=? WHERE id=?",
                      (iso(received - timedelta(hours=1)), run.id))
    assert retention.finished_at(repo, aid) == received
    repo.enqueue(sid, Stage.behavior, "cpu", camera="cam1")              # a re-analysis queued
    assert retention.finished_at(repo, aid) is None
