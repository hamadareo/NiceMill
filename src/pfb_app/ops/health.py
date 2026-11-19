"""One snapshot of the server's health for the admin status page (ui/pages/ops_panel.py):
the worker, the queue, the disk, the last backup, recent stops and alerts, log files and
versions. Read-only: it never changes a row or a file. GPU memory stays on the status page's
own GPU section (nvidia-smi)."""
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.ops import alerts, backup, disk, logs
from pfb_app.services import auto_retry
from pfb_app.storage import files
from pfb_app.storage.db import Repo

ALIVE_SEC = 120     # the worker touches its beat every 30 s (worker.main BEAT_SEC)
RECENT_FAILURES = 10
_APP_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Failure:
    job_id: int
    stage: str
    status: str
    error: str
    at: str | None
    auto_retried: bool


@dataclass
class Health:
    checked_at: datetime
    worker_alive: bool
    worker_beat_age_sec: float | None
    worker_pid: str | None
    worker_unclean: bool        # worker.running exists while no worker beats
    queue: dict[str, int]       # job status -> count (queued, running, waiting_gpu, failed…)
    disk: disk.DiskState
    backup_last: dict | None
    backup_count: int
    backup_newest: str | None
    backup_bytes: int = 0
    failures: list[Failure] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    alert_configured: bool = False
    log_dir: Path | None = None
    log_errors: list[str] = field(default_factory=list)
    app_commit: str | None = None
    engine_commit: str | None = None
    untraced_calls: int = 0

    @property
    def backup_age_hours(self) -> float | None:
        if not self.backup_last or not self.backup_last.get("at"):
            return None
        try:
            at = datetime.fromisoformat(self.backup_last["at"])
        except ValueError:
            return None
        return (self.checked_at - at).total_seconds() / 3600


def _app_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_APP_ROOT,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _engine_commit(settings: Settings) -> str | None:
    try:
        return (settings.engine_dir / "ENGINE_COMMIT").read_text(encoding="ascii").strip()[:12]
    except OSError:
        return None


def _worker_pid(settings: Settings) -> str | None:
    try:
        return (settings.data_dir / "worker.lock").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def _failures(repo: Repo, limit: int) -> list[Failure]:
    rows = repo.conn.execute(
        "SELECT j.id, j.stage, j.status, j.error, j.heartbeat_at,"
        " EXISTS (SELECT 1 FROM audit_log a WHERE a.action=? AND a.target='job:' || j.id)"
        " FROM jobs j WHERE j.status IN ('failed', 'failed_retryable')"
        " ORDER BY j.id DESC LIMIT ?", (auto_retry.ACTION, limit)).fetchall()
    out = []
    for job_id, stage, status, error, at, retried in rows:
        first = next((line.strip() for line in (error or "").splitlines() if line.strip()), "")
        out.append(Failure(job_id, stage, status, first[:200], at, bool(retried)))
    return out


def snapshot(repo: Repo, settings: Settings, *, now: datetime | None = None,
             disk_check=disk.disk_state) -> Health:
    from pfb_app.services import deletion   # noqa: PLC0415 - deletion imports the pipeline
    now = now or datetime.now(timezone.utc)
    beat = files.worker_beat_at(settings)
    age = None if beat is None else max(0.0, (now - beat).total_seconds())
    alive = age is not None and age < ALIVE_SEC
    queue = {status: n for status, n in repo.conn.execute(
        "SELECT status, COUNT(*) FROM jobs GROUP BY status")}
    backups = backup.list_backups(settings)
    errors = []
    for name in ("worker", "session", "backup"):
        errors += logs.recent_lines(settings, name, limit=5)
    return Health(
        checked_at=now, worker_alive=alive, worker_beat_age_sec=age,
        worker_pid=_worker_pid(settings) if alive else None,
        worker_unclean=not alive and (settings.data_dir / "worker.running").exists(),
        queue=queue, disk=disk_check(settings), backup_last=backup.last_result(settings),
        backup_count=len(backups), backup_newest=backups[-1].name if backups else None,
        backup_bytes=backup.backup_bytes(settings),
        failures=_failures(repo, RECENT_FAILURES), alerts=alerts.recent(settings),
        alert_configured=alerts.webhook_url(settings) is not None
        if settings.alert_webhook_file is not None else False,
        log_dir=logs.log_dir(settings), log_errors=sorted(errors)[-10:],
        app_commit=_app_commit(), engine_commit=_engine_commit(settings),
        untraced_calls=deletion.untraced_calls(repo))


def service_hint() -> str:
    """How the app is kept running here: systemd (INVOCATION_ID is set by systemd) or not."""
    return "systemd" if os.environ.get("INVOCATION_ID") else "tmux / 手動"
