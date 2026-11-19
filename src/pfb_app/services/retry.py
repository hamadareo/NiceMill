"""「もう一度ためす」: put a work's stopped analysis back in the queue, from the wizard or a home card.

- `open_failures(repo, sid)`: the failed jobs that still stop a group (the progress derivation
  reads the same list): the newest job of their stage key failed, no later job redoes it (a
  retry, or a regeneration from that stage or an earlier one of the same camera), and no ok
  run of that key was saved after the job (e.g. a feedback written by hand). A failed names
  job (fewer name proposals) or content check (a card without a verdict) never stops a work.
- `retry_work(repo, aid)` re-queues every open failure of every group of the work through
  `Repo.retry_job` and writes one audit entry `retry_work` (target `assignment:<id>`, detail:
  the job ids). `CannotRetry` (a plain sentence) when nothing can be retried (RETRYING when
  a job of the work is already waiting or running, e.g. after a second press) or the work is
  about to be deleted (checked inside the write transaction).
- `times_retried(repo, aid, job_ids)`: how many of those audit entries retried any of these
  jobs. After GIVE_UP_AFTER failed retries the wizard stops offering the button and says to
  tell the administrator (ui/wizard/progress.py). retry_work itself has no such guard: a stale
  tab or another group's button can still re-queue a given-up job (acceptable: it only tries
  once more). The worker's own automatic try (services.auto_retry, audit `auto_retry`) runs
  before the coach ever sees the stop and is not counted here: the coach keeps both tries.
"""
import json

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.services import deletion
from pfb_app.storage.db import NON_BLOCKING_STAGES, Repo

ACTION = "retry_work"
GIVE_UP_AFTER = 2
NOTHING = "いまは、もう一度ためせるものがありません。画面を新しくして、様子を確かめてください。"
DELETING = "この作業は削除を待っているため、もう一度ためせません。"
RETRYING = "もう一度ためしています。解析が進むと、画面に出ます。"   # pressed again meanwhile

_FAILED = {JobStatus.failed.value, JobStatus.failed_retryable.value}
_ACTIVE = (JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value)
_ORDER = {s.value: n for n, s in enumerate(Stage)}   # pipeline order


class CannotRetry(Exception):
    """Nothing to put back in the queue (the message is a plain sentence for the screen)."""


def _job_key(row: dict) -> tuple[int, str]:
    return _ORDER[row["stage"]], row["camera"] or ""


def _redoes(later: dict, failed: dict) -> bool:
    """True if the later job leads to the failed stage running again: the same stage, or an
    earlier stage of the pipeline (its follow-ups reach the failed stage), for the same
    camera when both name one; for a per-person stage, the same person."""
    if _ORDER[later["stage"]] > _ORDER[failed["stage"]]:
        return False
    if (later["camera"] is not None and failed["camera"] is not None
            and later["camera"] != failed["camera"]):
        return False
    return not (later["stage"] == failed["stage"] and failed["subject_id"] is not None
                and later["subject_id"] != failed["subject_id"])


def open_failures(repo: Repo, sid: int, rows: list[dict] | None = None) -> list[dict]:
    """The group's failed jobs that still stop it, in pipeline order (`rows`: its job rows,
    when already read)."""
    rows = repo.submission_job_rows(sid) if rows is None else rows
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[(row["stage"], row["camera"], row["subject_id"])] = row
    out = []
    for row in latest.values():
        if row["status"] not in _FAILED or row["stage"] in NON_BLOCKING_STAGES \
                or row["stage"] == Stage.detect.value:
            continue
        if any(r["id"] > row["id"] and _redoes(r, row) for r in rows):
            continue
        run = repo.latest_run(sid, Stage(row["stage"]), row["camera"], row["subject_id"])
        if run is not None and run.status == "ok" and run.created_at > row["created_at"]:
            continue
        out.append(row)
    return sorted(out, key=_job_key)


def failed_jobs(repo: Repo, aid: int) -> list[dict]:
    """Every open failure of the work's groups."""
    return [row for sub in repo.list_submissions(aid) for row in open_failures(repo, sub["id"])]


def retryable(repo: Repo, aid: int) -> bool:
    return bool(failed_jobs(repo, aid))


def _running(repo: Repo, aid: int) -> bool:
    """A job of the work that holds it up is waiting or running (e.g. the earlier press)."""
    marks = ",".join("?" * len(_ACTIVE))
    skip = ",".join("?" * len(NON_BLOCKING_STAGES))
    return repo.conn.execute(
        f"SELECT 1 FROM jobs j JOIN submissions s ON s.id=j.submission_id"
        f" WHERE s.assignment_id=? AND j.status IN ({marks}) AND j.stage NOT IN ({skip}) LIMIT 1",
        (aid, *_ACTIVE, *NON_BLOCKING_STAGES)).fetchone() is not None


def retry_work(repo: Repo, aid: int) -> list[int]:
    """Re-queue the work's open failures; returns the job ids put back in the queue.

    The deletion check runs inside the write transaction (BEGIN IMMEDIATE), so it is
    serialised with delete_work: a deletion requested from another tab is never followed by
    a re-queue. There is no give-up guard here: GIVE_UP_AFTER only hides the button."""
    deleting = running = False
    with repo._tx():
        if deletion.is_pending(repo, aid):
            deleting, ids = True, []
        else:
            ids = [row["id"] for row in failed_jobs(repo, aid) if repo.retry_job(row["id"])]
            if ids:
                repo.audit(ACTION, f"assignment:{aid}", {"job_ids": ids})
            else:
                running = _running(repo, aid)
    if deleting:
        raise CannotRetry(DELETING)
    if not ids:
        raise CannotRetry(RETRYING if running else NOTHING)
    return ids


def times_retried(repo: Repo, aid: int, job_ids) -> int:
    """How many `retry_work` entries of the work retried any of these jobs."""
    wanted = set(job_ids)
    if not wanted:
        return 0
    rows = repo.conn.execute("SELECT detail FROM audit_log WHERE action=? AND target=?",
                             (ACTION, f"assignment:{aid}")).fetchall()
    return sum(1 for (detail,) in rows
               if wanted & set(json.loads(detail).get("job_ids") or []))
