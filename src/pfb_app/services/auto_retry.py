"""One automatic re-queue of a step that stopped for a passing reason, before the coach sees
「止まりました」.

Passing reasons (`is_transient`): the GPU ran out of memory, the worker or its session died
while the step ran (heartbeat timeout), the engine was killed (exit -9 / 137, e.g. by the
system when memory ran out), a dropped connection, a busy database. Not passing: a step that
saved a failed result (`failed`), a model that could not be loaded or a server that kept
dying (those already retried in place: worker.main MAX_LOAD_FAILURES, runner
SERVER_DOWN_LIMIT), anything else. The content check and the names step (NO_RETRY_STAGES)
never hold a work up, so they are neither retried automatically nor alerted; 判定支援's
linecall (a side stage) is retried once like any step, but never alerted (QUIET_STAGES).

`retry_once(repo, job_id)` puts the job back in the queue at most once per job (audit entry
`auto_retry`, target `job:<id>`) and never for a work that is to be deleted. The backoff is
the worker's own: a step that stopped for the GPU or a crash ends its session, and the worker
waits gpu_poll_sec before it starts the next one.

The coach's 「もう一度ためす」 (services.retry) is separate: its give-up rule (GIVE_UP_AFTER)
counts only the coach's own `retry_work` entries, so the automatic try never uses up one of
the coach's tries, and a job the coach re-queued is not tried again automatically.

`sweep(repo)` (each worker pass) catches the jobs that stopped outside a running step (stale
recovery after a crash) within the last SWEEP_WINDOW_SEC; older failures are left as they are
(a first deploy never re-runs old work). `after_failure` is the step's own hook (runner):
retry once, or tell the operator (ops.alerts) that the step stopped.
"""
import logging
from datetime import datetime, timedelta, timezone

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

ACTION = "auto_retry"
SWEEP_WINDOW_SEC = 6 * 3600
TRANSIENT = ("heartbeat timeout", "CUDA out of memory", "OutOfMemoryError",
             "CUDA error: out of memory", "(exit -9)", "(exit 137)", "ConnectionRefusedError",
             "ConnectionResetError", "RemoteDisconnected", "Connection refused",
             "Connection reset by peer", "database is locked")
# the content check never holds a work up (its card shows without a verdict) and a names
# job only feeds proposals: neither is retried automatically nor alerted
NO_RETRY_STAGES = (Stage.detect, Stage.names)
# never alerted to the operator: those, and the side stages (判定支援's linecall, the plain
# versions of a text): the coach can ask again
QUIET_STAGES = (*NO_RETRY_STAGES, Stage.linecall, Stage.plain, Stage.tips)
# retried in place already (and given up on): never once more
NOT_TRANSIENT = ("解析サーバーが繰り返し停止しました", "モデルを読み込めませんでした",
                 "設定にないモデル")


def is_transient(error: str | None) -> bool:
    text = error or ""
    return not any(m in text for m in NOT_TRANSIENT) and any(m in text for m in TRANSIENT)


def _target(job_id: int) -> str:
    return f"job:{job_id}"


def retried_before(repo: Repo, job_id: int) -> bool:
    return repo.conn.execute("SELECT 1 FROM audit_log WHERE action=? AND target=? LIMIT 1",
                             (ACTION, _target(job_id))).fetchone() is not None


def _deleting(repo: Repo, submission_id: int) -> bool:
    from pfb_app.services import deletion   # noqa: PLC0415 - deletion imports the pipeline
    try:
        sub = repo.get_submission(submission_id)
    except KeyError:
        return True
    return deletion.is_requested(sub["meta"]) or deletion.is_pending(repo, sub["assignment_id"])


def retry_once(repo: Repo, job_id: int, settings: Settings | None = None) -> bool:
    """Re-queue a failed_retryable job whose error is passing, once per job. True when it was
    put back in the queue."""
    settings = settings or repo.settings
    if settings is not None and not settings.auto_retry:
        return False
    with repo._tx(join=repo.conn.in_transaction):
        try:
            job = repo.get_job(job_id)
        except KeyError:
            return False
        if job.stage in NO_RETRY_STAGES or job.status != JobStatus.failed_retryable \
                or not is_transient(job.error) \
                or retried_before(repo, job_id) or _deleting(repo, job.submission_id):
            return False
        reason = (job.error or "").strip().splitlines()[0][:200] if job.error else ""
        if not repo.retry_job(job_id):   # the same step is already waiting or running
            return False
        repo.audit(ACTION, _target(job_id), {"stage": job.stage.value, "error": reason})
    log.warning("一時的な理由で止まったため、自動でもう一度ためします job=%s: %s", job_id, reason)
    return True


def after_failure(repo: Repo, job_id: int, settings: Settings | None = None) -> bool:
    """A step just ended failed / failed_retryable: retry it once if the reason passes,
    otherwise send the operator alert. Returns True when it was re-queued. Never raises."""
    settings = settings or repo.settings
    try:
        if retry_once(repo, job_id, settings):
            return True
        job = repo.get_job(job_id)
        if settings is not None and job.stage not in QUIET_STAGES:
            from pfb_app.ops import alerts   # noqa: PLC0415
            again = "（自動でもう一度ためした後）" if retried_before(repo, job_id) else ""
            alerts.send(settings, f"job-stopped:{job_id}", "解析が止まりました",
                        f"ジョブ {job_id}（段階: {job.stage.value}）が止まりました{again}。"
                        "管理用の画面の「処理状況」で理由を確かめてください。", "error")
    except Exception:   # noqa: BLE001 - the job's own outcome is already recorded
        log.exception("止まったジョブの後処理に失敗しました job=%s", job_id)
    return False


def sweep(repo: Repo, settings: Settings | None = None,
          window_sec: float = SWEEP_WINDOW_SEC) -> list[int]:
    """Retry once every passing failure of the last `window_sec` not tried automatically yet
    (judged by the job's last heartbeat). Returns the re-queued job ids."""
    since = (datetime.now(timezone.utc) - timedelta(seconds=window_sec)).isoformat(
        timespec="microseconds")
    rows = repo.conn.execute(
        "SELECT id, error FROM jobs j WHERE status=? AND heartbeat_at >= ? AND NOT EXISTS"
        " (SELECT 1 FROM audit_log a WHERE a.action=? AND a.target = 'job:' || j.id)"
        " ORDER BY id", (JobStatus.failed_retryable.value, since, ACTION)).fetchall()
    out = []
    for job_id, error in rows:
        if is_transient(error):
            try:
                if retry_once(repo, job_id, settings):
                    out.append(job_id)
            except Exception:   # noqa: BLE001 - the worker keeps running
                log.exception("自動でもう一度ためせませんでした job=%s", job_id)
    return out
