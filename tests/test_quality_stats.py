"""「この解析の確かさ」: this installation's measured numbers (services.quality_stats).

Only synthetic data: invented names and texts, no files."""
from datetime import datetime, timedelta, timezone

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage, StageRun
from pfb_app.services import quality_stats, retry, review, short_text
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def pfb(repo, sid, subject_id, source="model", warnings=(), status="generated"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subject_id,
        version=0, source=source, upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"pfb_status": status, "pfb": "よく声をかけていました。", "evidence_refs": []},
        status="ok", error=None, warnings=list(warnings), duration_sec=1.0, created_at=""))


def run(repo, sid, stage, camera="cam1"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source="model", upstream_run_ids=[], prompt_hash=None, model_id=None, output={},
        status="ok", error=None, warnings=[], duration_sec=1.0, created_at=""))


def group(repo, people=("Pa", "Pb", "Pc", "Pd", "Pe")):
    aid = repo.create_assignment("見本の練習", "video", "パスをつなぐ", "", "standard", False)
    sid = repo.create_submission(aid, "A班", {})
    ids = [repo.add_subject(sid, p, f"人{n}", None, {}) for n, p in enumerate(people, 1)]
    return aid, sid, ids


def set_time(repo, table, row_id, column, when):
    repo.conn.execute(f"UPDATE {table} SET {column}=? WHERE id=?",
                      (when.isoformat(timespec="microseconds"), row_id))


def job(repo, sid, status, stage=Stage.track, attempts=1, camera="cam1"):
    job_id = repo.enqueue(sid, stage, "track", camera=camera)
    repo.conn.execute("UPDATE jobs SET status=?, attempts=? WHERE id=?",
                      (status.value, attempts, job_id))
    return job_id


def drafted(repo, sid, ids, minutes):
    """Every person gets a first draft `minutes` after T0 (the group is registered at T0)."""
    set_time(repo, "submissions", sid, "created_at", T0)
    for person in ids:
        set_time(repo, "stage_runs", pfb(repo, sid, person).id, "created_at",
                 T0 + timedelta(minutes=minutes))


def test_nothing_yet_gives_zeros_and_no_shares(repo):
    stats = quality_stats.measure(repo)
    assert stats.works == 0 and stats.people == 0 and stats.approved_total == 0
    assert stats.share(quality_stats.AS_IS) is None and stats.auto_share is None
    assert stats.turnaround_median_sec is None and stats.first_try_share is None
    assert stats.since is None and stats.few


def test_approved_texts_are_split_by_who_wrote_them_and_auto_is_its_own_kind(repo):
    _, sid, (a, b, c, d, e) = group(repo)
    repo.set_approval(a, pfb(repo, sid, a).id)
    repo.set_approval(b, pfb(repo, sid, b).id, auto=True)                   # 自動で承認
    pfb(repo, sid, c)
    repo.set_approval(c, pfb(repo, sid, c, source="teacher").id)            # 直して承認
    repo.set_approval(d, pfb(repo, sid, d, source="teacher",
                             warnings=[review.MANUAL_WARNING]).id)          # 自分で書いた
    repo.set_approval(e, pfb(repo, sid, e, source="teacher",
                             warnings=[short_text.MARK]).id)                # 短い文
    stats = quality_stats.measure(repo)
    assert stats.approved == {"as_is": 1, "edited": 1, "rewritten": 1, "short": 1, "auto": 1}
    assert stats.approved_total == 5 and stats.people == 5 and stats.works == 1
    assert stats.share(quality_stats.AS_IS) == pytest.approx(0.2)    # a person's own approval
    assert stats.auto == 1 and stats.auto_share == pytest.approx(0.2)


def test_only_current_approvals_of_evaluated_people_count(repo):
    _, sid, (a, b, c, d, _e) = group(repo)
    repo.set_approval(a, pfb(repo, sid, a).id)
    repo.set_approval(a, None)                       # 承認を取り消した
    repo.set_approval(b, pfb(repo, sid, b).id)
    pfb(repo, sid, b)                                # made again after the approval: not current
    repo.set_approval(c, pfb(repo, sid, c).id)
    repo.conn.execute("UPDATE subjects SET excluded=1 WHERE id=?", (c,))   # この人は評価しない
    repo.set_approval(d, pfb(repo, sid, d).id)
    stats = quality_stats.measure(repo)
    assert stats.approved_total == 1 and stats.approved["as_is"] == 1
    assert stats.people == 4


def test_turnaround_runs_from_the_confirmed_check_to_the_last_first_draft(repo):
    _, sid, (a, b) = group(repo, ("Pa", "Pb"))
    set_time(repo, "submissions", sid, "created_at", T0)
    repo.audit("confirm_check", f"submission:{sid}", {})
    entry = repo.conn.execute("SELECT MAX(id) FROM audit_log").fetchone()[0]
    set_time(repo, "audit_log", entry, "ts", T0 + timedelta(minutes=5))
    set_time(repo, "stage_runs", pfb(repo, sid, a).id, "created_at", T0 + timedelta(minutes=30))
    set_time(repo, "stage_runs", pfb(repo, sid, b, status="insufficient").id, "created_at",
             T0 + timedelta(minutes=35))           # a thin first draft is still a first draft
    again = pfb(repo, sid, a)                       # made again much later: not the first draft
    set_time(repo, "stage_runs", again.id, "created_at", T0 + timedelta(hours=5))
    stats = quality_stats.measure(repo)
    assert stats.turnaround_sec == (30 * 60.0,)


def test_turnaround_starts_at_registration_without_a_check_and_takes_the_median(repo):
    for minutes in (10, 20, 60):
        _, sid, ids = group(repo, ("Pa",))
        drafted(repo, sid, ids, minutes)
    assert quality_stats.measure(repo).turnaround_median_sec == 20 * 60.0


def test_turnaround_counts_only_finished_groups(repo):
    _, done_sid, done_ids = group(repo, ("Pa",))
    drafted(repo, done_sid, done_ids, 10)
    _, half_sid, (a, _b) = group(repo, ("Pa", "Pb"))
    drafted(repo, half_sid, [a], 1)                  # Pb has no draft yet
    _, busy_sid, busy_ids = group(repo, ("Pa",))
    drafted(repo, busy_sid, busy_ids, 2)
    job(repo, busy_sid, JobStatus.queued, stage=Stage.pfb)        # still writing
    _, stuck_sid, stuck_ids = group(repo, ("Pa",))
    drafted(repo, stuck_sid, stuck_ids, 3)
    job(repo, stuck_sid, JobStatus.failed, stage=Stage.behavior)  # stopped, not retried
    assert quality_stats.measure(repo).turnaround_sec == (10 * 60.0,)


def test_stopped_steps_are_the_open_failures_of_the_kept_groups(repo):
    _, sid, _ = group(repo)
    job(repo, sid, JobStatus.failed, stage=Stage.behavior)
    job(repo, sid, JobStatus.failed_retryable, stage=Stage.track, camera="cam2")
    assert quality_stats.measure(repo).jobs_stopped == 2
    run(repo, sid, Stage.track, camera="cam2")     # cam2 went through after all
    assert quality_stats.measure(repo).jobs_stopped == 1


def test_first_try_means_done_and_never_put_back_by_a_retry(repo):
    aid, sid, _ = group(repo)
    job(repo, sid, JobStatus.done, attempts=3)            # yields and requeues raise attempts
    job(repo, sid, JobStatus.done, stage=Stage.prepare, attempts=2)
    retried = job(repo, sid, JobStatus.done, stage=Stage.behavior)
    repo.audit(retry.ACTION, f"assignment:{aid}", {"job_ids": [retried]})
    stats = quality_stats.measure(repo)
    assert (stats.jobs_done, stats.jobs_first_try) == (3, 2)
    assert stats.first_try_share == pytest.approx(2 / 3)


def test_line_call_steps_are_neither_first_tries_nor_stopped(repo):
    """判定支援 (linecall) is a side stage, outside the work's analysis: counted nowhere."""
    _, sid, _ = group(repo)
    job(repo, sid, JobStatus.done)
    job(repo, sid, JobStatus.done, stage=Stage.linecall)
    job(repo, sid, JobStatus.failed, stage=Stage.linecall, camera="cam2")
    stats = quality_stats.measure(repo)
    assert (stats.jobs_done, stats.jobs_first_try, stats.jobs_stopped) == (1, 1, 0)


def test_retries_of_deleted_works_are_not_counted(repo):
    aid, _, _ = group(repo)
    repo.audit(retry.ACTION, f"assignment:{aid}", {"job_ids": []})
    repo.audit(retry.ACTION, f"assignment:{aid + 100}", {"job_ids": []})   # deleted since
    assert quality_stats.measure(repo).retries == 1


def test_the_oldest_kept_work_dates_the_numbers_in_local_time(repo):
    aid, _, _ = group(repo)
    created = datetime(2026, 9, 30, 23, 30, tzinfo=timezone.utc)
    set_time(repo, "assignments", aid, "created_at", created)
    group(repo)
    assert quality_stats.measure(repo).since == created.astimezone().date().isoformat()


def test_a_step_put_back_on_the_status_page_is_not_a_first_try(repo):
    _, sid, _ = group(repo)
    fresh = job(repo, sid, JobStatus.done)
    again = job(repo, sid, JobStatus.done, stage=Stage.behavior)
    repo.audit(quality_stats.RETRY_JOB, f"job:{again}", {"job_ids": [again]})   # 「再試行」
    stats = quality_stats.measure(repo)
    assert (stats.jobs_done, stats.jobs_first_try) == (2, 1) and fresh


def test_a_step_done_after_an_earlier_failure_of_the_same_step_is_not_a_first_try(repo):
    _, sid, (a, *_rest) = group(repo)
    job(repo, sid, JobStatus.failed, stage=Stage.behavior)               # 作り直し・選び直し …
    job(repo, sid, JobStatus.done, stage=Stage.behavior)                 # … then it went through
    job(repo, sid, JobStatus.failed_retryable, stage=Stage.track, camera="cam2")
    job(repo, sid, JobStatus.done, stage=Stage.track, camera="cam1")     # another camera: fresh
    pfb_failed = repo.enqueue(sid, Stage.pfb, "standard", subject_id=a)
    repo.conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (pfb_failed,))
    pfb_done = repo.enqueue(sid, Stage.pfb, "standard", subject_id=a + 1)   # another person
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (pfb_done,))
    stats = quality_stats.measure(repo)
    assert (stats.jobs_done, stats.jobs_first_try) == (3, 2)


def test_works_waiting_to_be_deleted_are_left_out(repo, monkeypatch):
    from pfb_app.services import deletion   # noqa: PLC0415
    gone, sid, (a, *_rest) = group(repo)
    repo.set_approval(a, pfb(repo, sid, a).id)
    job(repo, sid, JobStatus.done)
    kept, _, _ = group(repo, ("Pa",))
    monkeypatch.setattr(deletion, "pending_deletions", lambda r: {gone})
    stats = quality_stats.measure(repo)
    assert stats.works == 1 and stats.people == 1 and stats.approved_total == 0
    assert stats.jobs_done == 0 and kept
