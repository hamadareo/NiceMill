"""「運用の状態」 on the admin status page (処理状況) and the low-disk banner on every screen.
Synthetic rows only; temp data dirs."""
from datetime import datetime, timezone

import pytest
from streamlit.testing.v1 import AppTest

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.ops import alerts, backup, health
from pfb_app.services import auto_retry
from pfb_app.storage import files
from test_neutral_wording import SCHOOL_WORDS
from test_ui import UI, env, run_page  # noqa: F401 (env)
from test_wizard_ui_basic import FORBIDDEN, screen_text


def stopped_job(repo, error="ValueError: 読めません"):
    aid = repo.create_assignment("作業", "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "G", {"cameras": {}})
    jid = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    repo.claim_next_job({"cpu"})
    repo.finish_job(jid, JobStatus.failed_retryable, error)
    return jid


def test_snapshot(env):
    settings, repo = env
    jid = stopped_job(repo, "heartbeat timeout")
    auto_retry.retry_once(repo, jid)
    repo.claim_next_job({"cpu"})
    repo.finish_job(jid, JobStatus.failed_retryable, "heartbeat timeout")
    repo.enqueue(repo.list_submissions(repo.list_assignments()[0]["id"])[0]["id"],
                 Stage.transcribe, "whisper")
    backup.run_backup(settings)
    alerts.send(settings, "disk-warn", "空きが少ない")
    files.worker_beat_path(settings).touch()
    snap = health.snapshot(repo, settings)
    assert snap.worker_alive and snap.worker_beat_age_sec < 5
    assert snap.queue == {"failed_retryable": 1, "queued": 1}
    assert snap.backup_count == 1 and snap.backup_last["ok"]
    assert 0 <= snap.backup_age_hours < 0.1
    f, = snap.failures
    assert (f.job_id, f.stage, f.error, f.auto_retried) == (jid, "prepare", "heartbeat timeout",
                                                            True)
    assert [a["key"] for a in snap.alerts] == ["disk-warn"]
    assert snap.disk.level in ("ok", "warn", "refuse")


def test_snapshot_of_a_stopped_worker(env):
    settings, repo = env
    snap = health.snapshot(repo, settings, now=datetime.now(timezone.utc))
    assert not snap.worker_alive and snap.worker_beat_age_sec is None
    (settings.data_dir / "worker.running").write_text("1")
    assert health.snapshot(repo, settings).worker_unclean


def test_status_page_shows_the_section(env):
    settings, repo = env
    stopped_job(repo)
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    assert [s.value for s in at.subheader][0] == "運用の状態"
    labels = [m.label for m in at.metric]
    assert labels[:4] == ["ワーカー", "待ち／実行中", "保存場所の空き", "最後のバックアップ"]
    assert at.metric[0].value == "止まっています"
    assert any("ワーカーが動いていません" in e.value for e in at.error)
    assert any(i.value == "バックアップはまだありません。" for i in at.info)
    text = screen_text(at)
    assert "最近止まったもの（1件）" in text
    assert not [w for w in SCHOOL_WORDS if w in text]


def test_status_page_with_a_failed_backup_and_a_live_worker(env, tmp_path):
    settings, _ = env
    bk = tmp_path / "bk"
    with pytest.raises(backup.BackupError):   # no database there: a failed run in bk
        backup.run_backup(settings.model_copy(update={"data_dir": tmp_path / "none",
                                                      "backup_dir": bk}))
    files.worker_beat_path(settings).touch()
    at = AppTest.from_function(_render_panel, args=(str(settings.data_dir), str(bk)))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "動いています"
    assert at.metric[3].value == "失敗"
    assert any("前回のバックアップに失敗しました" in e.value for e in at.error)


def _render_panel(data_dir, backup_dir):
    from pathlib import Path

    from pfb_app.config import Settings
    from pfb_app.storage import files as f
    from pfb_app.storage.db import Repo, connect, init_schema
    from pfb_app.ui import ops_panel
    s = Settings(data_dir=Path(data_dir), backup_dir=Path(backup_dir))
    conn = connect(f.db_path(s))
    init_schema(conn)
    ops_panel.render(Repo(conn, s), s)


def _banner(level, data_dir):
    from pathlib import Path

    from pfb_app.config import Settings
    from pfb_app.ops import disk
    from pfb_app.ui import ops_panel
    s = Settings(data_dir=Path(data_dir))
    ops_panel.app_banner(s, disk_check=lambda _s: disk.DiskState(s.data_dir, 9.0, 100.0, level))


@pytest.mark.parametrize("level,kind", [("warn", "warning"), ("refuse", "error"), ("ok", None)])
def test_low_disk_banner(level, kind, tmp_path):
    at = AppTest.from_function(_banner, args=(level, str(tmp_path / "data")))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    shown = [e.value for e in at.warning] + [e.value for e in at.error]
    if kind is None:
        assert shown == []
    else:
        msg, = getattr(at, kind)
        assert "保存場所の空き" in msg.value and "約9.0GB" in msg.value
        assert not [w for w in (*SCHOOL_WORDS, *FORBIDDEN) if w in msg.value]


def test_the_home_screen_still_renders_with_the_banner(env):
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    settings, _ = env
    assert (settings.data_dir / "logs" / "ui.log").is_file()
