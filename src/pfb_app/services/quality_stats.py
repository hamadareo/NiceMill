"""「この解析の確かさ」: what this installation has measured about itself, from its own records.

Aggregates only (counts, shares, a median): never a name, a text or a file. Everything is
read from the works still kept in this server's database, so a deleted work drops out of the
numbers (the page says so).

- **approvals**: the *current* approvals of evaluated people only (`review.subject_status`
  says approved: the approved run is the latest feedback run and not outdated; a video's
  excluded people are not evaluated). Each is counted once, by who approved and who wrote
  it: approved automatically (`approved_auto`, 「自動で承認」), else a person approved the
  model's text unchanged, the model's text edited (a teacher run saved by 「直す」), a text
  written by hand (the 手書き mark of `review.write_manual_pfb`) or the short text (the 短い文
  mark of `services.short_text`).
- **turnaround**: per *finished* group (every evaluated person has a first draft, nothing
  still runs that a coach would wait for, no stopped step left), from the start of its
  analysis (the content check's confirmation when there is one, else its registration) to
  its last first draft (the first model feedback run of each person); the median.
- **analysis steps**: a done step is a first try when it was never put back (not in the
  `job_ids` of a 「もう一度ためす」 or a 管理用の画面 「再試行」 entry) and no earlier job of the
  same step (group, stage, camera, person) failed (作り直し, 選び直し and もう一度解析する
  enqueue new jobs). `attempts` also grows on yields and requeues, so it is not used. The
  steps stopped now are `retry.open_failures` of each kept group.
- **retries**: presses of 「もう一度ためす」 on works still kept.

Works waiting to be deleted (`deletion.pending_deletions`) are left out everywhere.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from statistics import median

from pfb_app.domain.models import JobStatus, Kind, Stage, SubjectStatus
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import deletion, retry, review, short_text
from pfb_app.storage.db import SIDE_STAGES, Repo

AS_IS, EDITED, REWRITTEN, SHORT, AUTO = "as_is", "edited", "rewritten", "short", "auto"
KINDS = (AS_IS, EDITED, REWRITTEN, SHORT, AUTO)   # a person's approvals first, then the worker's
FEW = 20    # below this many approved people the page says the numbers are rough
RETRY_JOB = "retry_job"   # audit action of 管理用の画面 処理状況's 「再試行」 ({"job_ids": [id]})
_FAILED = (JobStatus.failed.value, JobStatus.failed_retryable.value)


@dataclass(frozen=True)
class QualityStats:
    works: int                       # works (assignments) kept in the database
    people: int                      # evaluated people in them
    approved: dict[str, int]         # {kind in KINDS: people approved that way now}
    turnaround_sec: tuple[float, ...]   # one per finished group, oldest group first
    jobs_done: int                   # analysis steps that finished
    jobs_first_try: int              # of those, never put back by a retry
    jobs_stopped: int                # analysis steps stopped now (open failures)
    retries: int                     # presses of 「もう一度ためす」 on kept works
    since: str | None                # the oldest kept work's local date (YYYY-MM-DD), None: none

    @property
    def approved_total(self) -> int:
        return sum(self.approved.values())

    @property
    def auto(self) -> int:
        return self.approved.get(AUTO, 0)

    def share(self, kind: str) -> float | None:
        """Share of the approved people approved as `kind` (None: nobody approved)."""
        total = self.approved_total
        return self.approved.get(kind, 0) / total if total else None

    @property
    def auto_share(self) -> float | None:
        return self.share(AUTO)

    @property
    def turnaround_median_sec(self) -> float | None:
        return median(self.turnaround_sec) if self.turnaround_sec else None

    @property
    def first_try_share(self) -> float | None:
        return self.jobs_first_try / self.jobs_done if self.jobs_done else None

    @property
    def few(self) -> bool:
        """Too few approvals for more than a rough picture."""
        return self.approved_total < FEW


def _kind(subject: dict, source: str, warnings: list[str]) -> str:
    if subject.get("approved_auto"):
        return AUTO
    if source == "model":
        return AS_IS
    if short_text.MARK in warnings:
        return SHORT
    if review.MANUAL_WARNING in warnings:
        return REWRITTEN
    return EDITED


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _first_drafts(repo: Repo, sid: int) -> dict[int, str]:
    """{subject id: created_at of its first model feedback run} of one group."""
    rows = repo.conn.execute(
        "SELECT subject_id, created_at FROM stage_runs WHERE id IN (SELECT MIN(id) FROM"
        " stage_runs WHERE submission_id = ? AND stage = ? AND source = 'model' AND status = 'ok'"
        " AND subject_id IS NOT NULL GROUP BY subject_id)", (sid, Stage.pfb.value)).fetchall()
    return {subject_id: created for subject_id, created in rows}


def _confirmed(repo: Repo) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for target, ts in repo.conn.execute(
            "SELECT target, ts FROM audit_log WHERE action = 'confirm_check' ORDER BY id"):
        out.setdefault(target, []).append(ts)
    return out


def _turnaround(sub: dict, evaluated: list[dict], drafts: dict[int, str],
                confirmed: dict[str, list[str]]) -> float | None:
    if not evaluated or any(s["id"] not in drafts for s in evaluated):
        return None
    end = max(_time(drafts[s["id"]]) for s in evaluated)
    starts = [_time(ts) for ts in confirmed.get(f"submission:{sub['id']}", [])
              if _time(ts) <= end]
    seconds = (end - (max(starts) if starts else _time(sub["created_at"]))).total_seconds()
    return seconds if seconds > 0 else None


def _retried_job_ids(repo: Repo) -> set[int]:
    ids: set[int] = set()
    for (detail,) in repo.conn.execute("SELECT detail FROM audit_log WHERE action IN (?, ?)",
                                       (retry.ACTION, RETRY_JOB)):
        ids.update(json.loads(detail or "{}").get("job_ids") or [])
    return ids


def _done_jobs(repo: Repo, sids: list[int]) -> list[tuple[int, bool]]:
    """[(job id, an earlier job of the same step failed)] of the groups' done jobs; side
    stages (判定支援) are not the work's analysis and are left out, as in jobs_stopped."""
    if not sids:
        return []
    marks = ",".join("?" * len(sids))
    side = ",".join("?" * len(SIDE_STAGES))
    return [(job_id, bool(failed_before)) for job_id, failed_before in repo.conn.execute(
        "SELECT j.id, EXISTS (SELECT 1 FROM jobs e WHERE e.id < j.id"
        " AND e.submission_id = j.submission_id AND e.stage = j.stage"
        " AND e.camera IS j.camera AND e.subject_id IS j.subject_id"
        f" AND e.status IN (?, ?)) FROM jobs j WHERE j.status = ? AND j.submission_id IN ({marks})"
        f" AND j.stage NOT IN ({side})",
        (*_FAILED, JobStatus.done.value, *sids, *SIDE_STAGES))]


def measure(repo: Repo) -> QualityStats:
    """This installation's numbers, now."""
    approved = dict.fromkeys(KINDS, 0)
    people = stopped = 0
    turnarounds: list[float] = []
    confirmed = _confirmed(repo)
    pending = deletion.pending_deletions(repo)
    assignments = [a for a in repo.list_assignments() if a["id"] not in pending]
    sids: list[int] = []
    for assignment in assignments:
        video = assignment["kind"] == Kind.video.value
        for sub in repo.list_submissions(assignment["id"]):
            sid = sub["id"]
            sids.append(sid)
            rows = repo.submission_job_rows(sid)
            failures = retry.open_failures(repo, sid, rows)
            stopped += len(failures)
            evaluated = [s for s in repo.list_subjects(sid) if not video or is_evaluated(s)]
            people += len(evaluated)
            for subject in evaluated:
                if review.subject_status(repo, subject["id"]) != SubjectStatus.approved:
                    continue
                run = repo.get_run(subject["approved_run_id"])
                approved[_kind(subject, run.source, run.warnings)] += 1
            if not failures and not repo.has_blocking_jobs(sid):
                seconds = _turnaround(sub, evaluated, _first_drafts(repo, sid), confirmed)
                if seconds is not None:
                    turnarounds.append(seconds)
    retried = _retried_job_ids(repo)
    done = _done_jobs(repo, sids)
    kept = {f"assignment:{a['id']}" for a in assignments}
    retries = sum(1 for (target,) in repo.conn.execute(
        "SELECT target FROM audit_log WHERE action = ?", (retry.ACTION,)) if target in kept)
    oldest = min((a["created_at"] for a in assignments), default=None)
    return QualityStats(
        works=len(assignments), people=people, approved=approved,
        turnaround_sec=tuple(turnarounds), jobs_done=len(done),
        jobs_first_try=sum(1 for job_id, failed_before in done
                           if not failed_before and job_id not in retried),
        jobs_stopped=stopped,
        retries=retries,
        since=_time(oldest).astimezone().date().isoformat() if oldest else None)


def change_marker(repo: Repo) -> tuple:
    """Moves whenever a number above may move: the newest audit entry (approvals, deletions,
    retries), the newest stage run (drafts, edits) and the finished jobs. A cache keyed by it
    is never older than the records."""
    return tuple(repo.conn.execute(
        "SELECT (SELECT MAX(id) FROM audit_log), (SELECT MAX(id) FROM stage_runs),"
        " (SELECT COUNT(*) FROM jobs WHERE status IN (?, ?, ?))",
        (JobStatus.done.value, *_FAILED)).fetchone())
