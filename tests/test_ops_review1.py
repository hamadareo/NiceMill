"""Ops review round 1 (review-ops-findings.md F1–F9 and the privacy note). Temp dirs, fake
systemctl / tmux on PATH, local sockets only: nothing touches a real unit or the live app."""
import logging
import os
import socket
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.ops import alerts, backup, disk, health, systemd
from pfb_app.ops.supervise import WorkerLock
from pfb_app.services import deletion, intake
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import main as worker_main
from test_ops_backup import closed_port
from test_worker import StubGuard

ROOT = Path(__file__).resolve().parents[1]
GB = disk.GB


def config(tmp_path, data_dir, **extra) -> Path:
    cfg = tmp_path / "pfb.yaml"
    lines = [f"data_dir: {data_dir}"] + [f"{k}: {v}" for k, v in extra.items()]
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def fake_bin(tmp_path, enabled=False, active=False) -> tuple[Path, Path]:
    """systemctl / tmux stand-ins that record their calls (and answer is-enabled/is-active)."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls.log"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        f'echo "systemctl $*" >> "{calls}"\n'
        'case "$*" in\n'
        f'  *is-enabled*) exit {0 if enabled else 1} ;;\n'
        f'  *is-active*) exit {0 if active else 3} ;;\n'
        "esac\nexit 0\n")
    (bin_dir / "tmux").write_text(f'#!/bin/sh\necho "tmux $*" >> "{calls}"\nexit 1\n')
    (bin_dir / "loginctl").write_text("#!/bin/sh\necho yes\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir, calls


def install(tmp_path, *args, bin_dir=None):
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PFB_CONFIG")}
    env.update(HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "home" / ".config"))
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(["bash", str(ROOT / "scripts" / "install_user_services.sh"),
                           "--python", sys.executable, *args],
                          capture_output=True, text=True, env=env, timeout=120)


def unit_dir(tmp_path) -> Path:
    return tmp_path / "home" / ".config" / "systemd" / "user"


# --- F1: --enable never starts a second worker or UI on data that is in use ------------------

def test_enable_refuses_a_fresh_worker_beat(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "worker.beat").touch()
    bin_dir, calls = fake_bin(tmp_path)
    res = install(tmp_path, "--enable", "--config", str(config(tmp_path, data)),
                  "--host", "127.0.0.1", "--port", str(closed_port()), bin_dir=bin_dir)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "ワーカーが動いています" in res.stderr
    assert not unit_dir(tmp_path).exists()            # nothing written
    assert "enable" not in (calls.read_text() if calls.exists() else "")


def test_enable_refuses_a_held_worker_lock(tmp_path):
    data = tmp_path / "data"
    held = WorkerLock(data / "worker.lock").acquire()
    try:
        bin_dir, calls = fake_bin(tmp_path)
        res = install(tmp_path, "--enable", "--config", str(config(tmp_path, data)),
                      "--host", "127.0.0.1", "--port", str(closed_port()), bin_dir=bin_dir)
    finally:
        held.release()
    assert res.returncode == 1 and "worker.lock" in res.stderr
    assert not unit_dir(tmp_path).exists()


def test_enable_refuses_a_listening_ui_port(tmp_path):
    data = tmp_path / "data"
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        bin_dir, _ = fake_bin(tmp_path)
        res = install(tmp_path, "--enable", "--config", str(config(tmp_path, data)),
                      "--host", "127.0.0.1", "--port", str(port), bin_dir=bin_dir)
    assert res.returncode == 1 and f"127.0.0.1:{port}" in res.stderr
    assert not unit_dir(tmp_path).exists()


def test_enable_goes_ahead_on_quiet_data(tmp_path):
    data = tmp_path / "data"
    bin_dir, calls = fake_bin(tmp_path)
    res = install(tmp_path, "--enable", "--prefix", "pfb-t", "--config",
                  str(config(tmp_path, data)), "--host", "127.0.0.1",
                  "--port", str(closed_port()), bin_dir=bin_dir)
    assert res.returncode == 0, res.stdout + res.stderr
    log = calls.read_text()
    assert "systemctl --user daemon-reload" in log
    assert "enable --now pfb-t-worker.service pfb-t-ui.service pfb-t-backup.timer" in log
    assert (unit_dir(tmp_path) / "pfb-t-worker.service").is_file()


def test_another_prefix_needs_its_own_settings(tmp_path):
    res = install(tmp_path, "--prefix", "pfb-t", "--unit-dir", str(tmp_path / "u"))
    assert res.returncode == 2 and "--config" in res.stderr
    assert not (tmp_path / "u").exists()


def test_preflight_lists_every_problem(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    s.data_dir.mkdir()
    (s.data_dir / "worker.beat").touch()
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        problems = systemd.preflight(s, "127.0.0.1", srv.getsockname()[1])
    assert len(problems) == 2
    assert systemd.preflight(s, "127.0.0.1", closed_port(), beat_fresh_sec=0) == []
    assert not (s.data_dir / "worker.lock").exists()   # the probe never creates it


# --- F2: never retarget units written for another checkout -----------------------------------

def test_units_of_another_checkout_are_not_overwritten(tmp_path):
    out = tmp_path / "u"
    out.mkdir()
    other = systemd.render(tmp_path / "other-checkout", prefix="pfb")
    (out / "pfb-worker.service").write_text(other["pfb-worker.service"])
    res = install(tmp_path, "--unit-dir", str(out))
    assert res.returncode == 1 and "other-checkout" in res.stderr
    assert (out / "pfb-worker.service").read_text() == other["pfb-worker.service"]
    res = install(tmp_path, "--unit-dir", str(out), "--force")
    assert res.returncode == 0, res.stderr
    assert f"for {ROOT}\n" in (out / "pfb-worker.service").read_text()
    assert install(tmp_path, "--unit-dir", str(out)).returncode == 0   # same checkout: fine


def test_existing_target(tmp_path):
    assert systemd.existing_target(tmp_path, "pfb") is None
    (tmp_path / "pfb-ui.service").write_text(systemd.render(Path("/srv/a b"))["pfb-ui.service"])
    assert systemd.existing_target(tmp_path, "pfb") == Path("/srv/a b")


# --- F3: restore refuses while a worker holds the lock or the UI answers ---------------------

@pytest.fixture
def bsettings(tmp_path):
    return Settings(data_dir=tmp_path / "data", host="127.0.0.1", port=closed_port())


def made_backup(s):
    conn = connect(files.db_path(s))
    init_schema(conn)
    Repo(conn, s).create_assignment("作業", "video", "説明", "", "standard", False)
    conn.close()
    return backup.run_backup(s)


def test_restore_refuses_a_held_lock_even_with_an_old_beat(bsettings):
    path = made_backup(bsettings)
    held = WorkerLock(bsettings.data_dir / "worker.lock").acquire()
    try:
        assert backup.worker_running(bsettings)
        with pytest.raises(backup.RestoreError, match="ワーカー"):
            backup.restore(bsettings, path.name)
    finally:
        held.release()
    assert not backup.worker_running(bsettings)
    backup.restore(bsettings, path.name)


def test_restore_refuses_while_the_ui_answers(bsettings):
    path = made_backup(bsettings)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        s = bsettings.model_copy(update={"port": srv.getsockname()[1]})
        with pytest.raises(backup.RestoreError, match="画面"):
            backup.restore(s, path.name)
        assert backup.restore(s, path.name, force=True).name == path.name


def test_live_ui_checks_only_for_the_live_data_dir(bsettings):
    assert backup.ui_running(bsettings, live_checks=lambda: True) is None   # a temp data dir
    live = Settings(host="127.0.0.1", port=closed_port())
    assert backup.ui_running(live, live_checks=lambda: True)
    assert backup.ui_running(live, live_checks=lambda: False) is None


# --- F4: the UI keeps trying at boot; back-off instead of a start limit ----------------------

def test_units_back_off_and_never_give_up_the_ui(tmp_path):
    units = systemd.render(tmp_path)
    ui, worker = units["pfb-ui.service"], units["pfb-worker.service"]
    for text in (ui, worker):
        assert "StartLimitIntervalSec=0" in text
        assert "RestartSteps=" in text and "RestartMaxDelaySec=" in text
        assert "network-online.target" not in text   # absent in a --user manager


# --- F5 / tests: a backup while commits land between its steps -------------------------------

def test_backup_while_another_connection_keeps_committing(bsettings):
    conn = connect(files.db_path(bsettings))
    init_schema(conn)
    repo = Repo(conn, bsettings)
    for i in range(3000):
        repo.audit("filler", None, {"i": i, "pad": "x" * 200})
    stop = threading.Event()

    def writer():
        w = connect(files.db_path(bsettings))
        while not stop.is_set():
            w.execute("INSERT INTO audit_log (ts, action, target, detail) VALUES"
                      " ('t','w',NULL,'{}')")
        w.close()
    t = threading.Thread(target=writer)
    t.start()
    try:
        made = [backup.run_backup(bsettings) for _ in range(3)]
    finally:
        stop.set()
        t.join()
    for path in made:
        assert backup.verify(path).ok
    assert backup.PAGES_PER_STEP == -1


def test_online_copy_with_steps_and_commits_between_them(tmp_path):
    src = tmp_path / "src.db"
    c = sqlite3.connect(src)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t (x TEXT)")
    c.executemany("INSERT INTO t VALUES (?)", [("x" * 500,)] * 2000)
    c.commit()
    commits = []

    def progress(status, remaining, total):
        if len(commits) < 5:
            c.execute("INSERT INTO t VALUES ('late')")
            c.commit()
            commits.append(remaining)
    backup.online_copy(src, tmp_path / "copy.db", pages=8, progress=progress)
    copy = sqlite3.connect(tmp_path / "copy.db")
    assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert copy.execute("SELECT COUNT(*) FROM t").fetchone()[0] >= 2000
    assert commits


# --- F6: copies beside app.db are rotated; the backup folder size is shown --------------------

def test_restore_and_migration_copies_keep_the_newest(bsettings):
    """pre-migrate copies: the newest 3; before-restore copies are never removed (round 2)."""
    from datetime import datetime, timedelta
    made_backup(bsettings)
    names = [(datetime.now() - timedelta(hours=h)).strftime("%Y%m%d-%H%M%S")
             for h in (6, 5, 4, 3, 2, 1)]
    for n in names:
        (bsettings.data_dir / f"app.db.pre-migrate-{n}").write_bytes(b"x")
        (bsettings.data_dir / f"app.db.before-restore-{n}").write_bytes(b"x")
    backup.run_backup(bsettings)
    left = sorted(p.name for p in bsettings.data_dir.glob("app.db.pre-migrate-*"))
    assert left == [f"app.db.pre-migrate-{n}" for n in names[3:]]
    assert len(list(bsettings.data_dir.glob("app.db.before-restore-*"))) == 6


def test_health_reports_the_backup_folder_size(bsettings):
    made_backup(bsettings)
    conn = connect(files.db_path(bsettings))
    snap = health.snapshot(Repo(conn, bsettings), bsettings)
    assert snap.backup_bytes > 0


# --- F7: alert texts for exit 75 and the backup unit -----------------------------------------

def test_exit_75_has_its_own_text():
    title, text = alerts.unit_failed_text("pfb-worker.service",
                                          {"ActiveState": "failed", "ExecMainStatus": "75"})
    assert "別のワーカー" in text and "再起動をやめました" not in text


def test_the_backup_unit_alerts_once_itself():
    units = systemd.render(Path("/srv/app"))
    assert "OnFailure" not in units["pfb-backup.service"]
    assert "OnFailure=pfb-alert@%n.service" in units["pfb-worker.service"]


# --- F8: start.sh / stop.sh also look at enabled units ---------------------------------------

def run_script(name, tmp_path, enabled, active=False):
    bin_dir, calls = fake_bin(tmp_path, enabled=enabled, active=active)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PFB_CONFIG")}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    res = subprocess.run(["bash", str(ROOT / "scripts" / name)], capture_output=True,
                         text=True, env=env, timeout=60)
    return res, (calls.read_text() if calls.exists() else "")


def test_start_sh_leaves_enabled_units_to_systemd(tmp_path):
    res, calls = run_script("start.sh", tmp_path, enabled=True)
    assert res.returncode == 0 and "systemd" in res.stdout
    assert "tmux new-session" not in calls and "tmux has-session" not in calls


def test_stop_sh_mentions_enabled_units(tmp_path):
    res, _ = run_script("stop.sh", tmp_path, enabled=True)
    assert "systemctl --user" in res.stdout and "disable" in res.stdout


# --- F9 ----------------------------------------------------------------------------------------

def test_concurrent_alerts_are_all_recorded(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    threads = [threading.Thread(target=alerts.send, args=(s, f"k{i}", "題")) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(alerts.recent(s, limit=100)) == 30


def test_a_readable_webhook_file_is_warned_about(tmp_path, caplog):
    hook = tmp_path / "hook"
    hook.write_text("https://example.invalid/hook/SECRET", encoding="utf-8")
    hook.chmod(0o644)
    s = Settings(data_dir=tmp_path / "data", alert_webhook_file=hook)
    with caplog.at_level(logging.WARNING):
        assert alerts.webhook_url(s)
    assert "600" in caplog.text and "SECRET" not in caplog.text
    caplog.clear()
    hook.chmod(0o600)
    with caplog.at_level(logging.WARNING):
        alerts.webhook_url(s)
    assert caplog.text == ""


def test_inbox_registration_is_refused_when_the_disk_is_full(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    video = tmp_path / "inbox" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not read")
    conn = connect(files.db_path(s))
    init_schema(conn)
    repo = Repo(conn, s)
    aid = repo.create_assignment("作業", "video", "説明", "", "standard", False)
    monkeypatch.setattr(disk, "_usage", lambda _p: type("U", (), {"free": 1 * GB,
                                                                   "total": 100 * GB})())
    with pytest.raises(disk.DiskLow):
        intake.register_video_group(repo, s, aid, "G", {"cam1": video}, None, None, None,
                                    False, [])
    assert repo.list_submissions(aid) == [] and video.exists()


def test_owner_record_retries_a_busy_database(tmp_path, caplog):
    s = Settings(data_dir=tmp_path / "data")
    conn = connect(files.db_path(s))
    init_schema(conn)
    repo = Repo(conn, s)
    aid = repo.create_assignment("作業", "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "G", {})
    real, tries = repo._insert_owner, []

    def flaky(h):
        tries.append(h)
        if len(tries) < 3:
            raise sqlite3.OperationalError("database is locked")
        real(h)
    repo._insert_owner = flaky
    repo.call_owner = sid
    repo.cache_put("h", "m", "{}", 1, 1, 0.1, None)
    assert len(tries) == 3
    assert repo.conn.execute("SELECT submission_id FROM call_log_owner").fetchone()[0] == sid
    repo._insert_owner = lambda h: (_ for _ in ()).throw(sqlite3.OperationalError("locked"))
    with caplog.at_level(logging.ERROR):
        repo.cache_put("h2", "m", "{}", 1, 1, 0.1, None)
    assert "h2" in caplog.text


# --- tests asked for: deletions finish while new work is held for the disk -------------------

def test_deletions_finish_during_a_disk_hold(tmp_path):
    s = Settings(data_dir=tmp_path / "data", gpu_poll_sec=0, session_idle_sec=0)
    conn = connect(files.db_path(s))
    init_schema(conn)
    repo = Repo(conn, s)
    aid = repo.create_assignment("消す作業", "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "G", {"cameras": {}})
    jid = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    repo.claim_next_job({"cpu"})
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, s, aid, stop_analysis=True)
    repo.finish_job(jid, JobStatus.done)             # its step ends meanwhile
    other = repo.create_assignment("残る作業", "video", "説明", "", "standard", False)
    osid = repo.create_submission(other, "G", {"cameras": {}})
    ojob = repo.enqueue(osid, Stage.prepare, "cpu", camera="cam1")
    full = lambda _s: disk.DiskState(_s.data_dir, 3.0, 1000.0, "refuse")   # noqa: E731
    worker_main.worker_loop(s, guard=StubGuard(), spawn=lambda k, **kw: 0, sleep=lambda x: None,
                            once=True, on_start=lambda _s: None, disk_check=full)
    assert [a["id"] for a in repo.list_assignments()] == [other]   # the deletion finished
    assert repo.get_job(ojob).status == JobStatus.queued            # nothing new started


# --- privacy note: deleted works stay in the backups for a while -----------------------------

def test_the_guide_and_the_panel_say_backups_keep_deleted_works():
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "バックアップ" in guide and "14日" in guide and "削除" in guide
    from pfb_app.ui import ops_panel
    assert "14日" in ops_panel.backup_privacy_note(Settings())
    assert "3日" in ops_panel.backup_privacy_note(Settings(backup_keep=3))
