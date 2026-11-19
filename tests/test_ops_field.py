"""Field readiness (ops lane): weekly backup retention, the free-space floor of a backup, the
restore rehearsal, the status watch (ops.guard) and its line on the status page, the guard
units. Synthetic data, temp dirs."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from pfb_app.config import Settings
from pfb_app.ops import alerts, backup, guard
from pfb_app.ops.disk import DiskState
from pfb_app.ops import systemd
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_ops_backup import add_work, closed_port
from test_ui import env, run_page  # noqa: F401 (env)

NOW = datetime(2026, 11, 14, 12, 0, 0)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "app" / "data", backup_keep=3, backup_keep_weekly=2,
                    host="127.0.0.1", port=closed_port())


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    yield Repo(conn, settings)
    conn.close()


def fake_backup(settings, name):
    d = backup.backup_root(settings) / name
    d.mkdir(parents=True)
    (d / backup.MANIFEST).write_text("{}")
    return d


def days_ago(n, hour=3):
    return (NOW - timedelta(days=n)).replace(hour=hour).strftime("app-%Y%m%d-%H0000")


# --- weekly retention -------------------------------------------------------------------------

def test_rotation_keeps_daily_then_weekly(settings):
    # nightly for 40 days: keep=3 days, then 2 weekly buckets (ages 3..9 and 10..16 days)
    names = [days_ago(n) for n in range(40, -1, -1)]
    for n in names:
        fake_backup(settings, n)
    removed = backup.rotate(settings, now=NOW)
    left = [p.name for p in backup.list_backups(settings)]
    ages = sorted(int((NOW - datetime.strptime(n[4:12], "%Y%m%d")).days) for n in left)
    # today, 1, 2 days ago are the daily ones; 3 and 10 days ago are the newest of each week bucket
    assert ages[:3] == [0, 1, 2] and len(left) == 5, ages
    assert 2 <= ages[3] <= 9 and 10 <= ages[4] <= 16
    assert max(ages) <= backup.max_age_days(settings) == 3 + 14
    assert len(removed) == 41 - 5


def test_weekly_zero_is_the_plain_count(settings):
    s = settings.model_copy(update={"backup_keep_weekly": 0})
    for n in range(10, -1, -1):
        fake_backup(s, days_ago(n))
    backup.rotate(s, now=NOW)
    assert len(backup.list_backups(s)) == 3
    assert backup.max_age_days(s) == 3


def test_the_newest_backup_always_stays(settings):
    fake_backup(settings, days_ago(90))
    backup.rotate(settings, now=NOW)
    assert len(backup.list_backups(settings)) == 1


# --- the free-space floor ---------------------------------------------------------------------

def test_a_backup_is_refused_below_the_floor(settings, repo):
    add_work(repo)
    low = lambda path: SimpleNamespace(free=settings.disk_refuse_gb * 1024 ** 3 + 1000)  # noqa: E731
    with pytest.raises(backup.BackupError, match="空きが足りない"):
        backup.run_backup(settings, now=NOW, usage=low)
    assert backup.list_backups(settings) == []
    last = backup.last_result(settings)
    assert last["ok"] is False and "空きが足りない" in last["error"]
    assert [a["key"] for a in alerts.recent(settings)] == ["backup-failed"]
    # nothing half-written is left behind
    assert not [p for p in backup.backup_root(settings).iterdir() if p.name.startswith(".tmp-app")]
    plenty = lambda path: SimpleNamespace(free=500 * 1024 ** 3)  # noqa: E731
    assert backup.run_backup(settings, now=NOW, usage=plenty).is_dir()


# --- restore rehearsal ------------------------------------------------------------------------

def test_rehearsal_restores_into_scratch_and_compares(settings, repo):
    add_work(repo)
    made = backup.run_backup(settings, now=NOW)
    add_work(repo, "作業B")   # the live database moves on after the backup
    res = backup.rehearse(settings)
    assert res.ok and res.name == made.name and res.problems == []
    assert res.tables["assignments"] == 1
    assert res.live_diff["assignments"] == 1 and res.live_diff["submissions"] == 1
    root = backup.backup_root(settings)
    assert not [p for p in root.iterdir() if p.name.startswith(".tmp-rehearse")]
    rec = backup.last_rehearsal(settings)
    assert rec["ok"] and rec["name"] == made.name
    # the live database is untouched
    conn = sqlite3.connect(files.db_path(settings))
    assert conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 2
    conn.close()


def test_rehearsal_of_a_damaged_backup_fails(settings, repo):
    add_work(repo)
    made = backup.run_backup(settings, now=NOW)
    conn = sqlite3.connect(made / "app.db")
    conn.execute("DELETE FROM subjects")
    conn.commit()
    conn.close()
    res = backup.rehearse(settings, made.name)
    assert not res.ok and any("一致しません" in p for p in res.problems)
    assert backup.last_rehearsal(settings)["ok"] is False


def test_rehearsal_without_backups(settings):
    with pytest.raises(backup.RestoreError):
        backup.rehearse(settings)


def test_the_status_files_are_not_part_of_the_manifest(settings, repo):
    add_work(repo)
    guard.check(settings, listening=lambda h, p: True)
    made = backup.run_backup(settings, now=NOW)
    import gzip
    text = gzip.open(made / backup.FILE_LIST, "rt", encoding="utf-8").read()
    assert "ops_status" not in text


# --- the status watch -------------------------------------------------------------------------

UTC_NOW = datetime(2026, 11, 14, 3, 0, 0, tzinfo=timezone.utc)


def disk_ok(s):
    return DiskState(s.data_dir, 80.0, 1000.0, "ok")


def touch_beat(settings, ago_sec, now=UTC_NOW):
    p = files.worker_beat_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    import os
    t = now.timestamp() - ago_sec
    os.utime(p, (t, t))


def test_all_good(settings, repo):
    add_work(repo)
    backup.run_backup(settings, now=NOW)
    backup.rehearse(settings)
    now = datetime.now(timezone.utc)
    # the backup's own timestamps are real time: judge them, and the worker beat, against it
    touch_beat(settings, 20, now=now)
    st = guard.collect(settings, now, listening=lambda h, p: True, disk_check=disk_ok)
    assert st["level"] == "ok", st
    assert [i["key"] for i in st["items"]] == ["worker", "ui", "disk", "backup", "rehearsal"]


def test_problems_are_named_in_plain_words(settings, repo):
    touch_beat(settings, 4000)
    st = guard.collect(settings, UTC_NOW, listening=lambda h, p: False,
                       disk_check=lambda s: DiskState(s.data_dir, 9.0, 1000.0, "refuse"))
    by = {i["key"]: i for i in st["items"]}
    assert by["worker"]["level"] == "error" and "止まっています" in by["worker"]["text"]
    assert by["ui"]["level"] == "error"
    assert by["disk"]["level"] == "error" and "受け付けていません" in by["disk"]["text"]
    assert by["backup"]["level"] == "warn"   # never ran
    assert st["level"] == "error" and "対応が必要" in st["summary"]
    text = guard.render_text(st)
    assert "[要対応]" in text and "まとめ:" in text


def test_a_worker_that_is_restarting_is_only_a_warning(settings):
    touch_beat(settings, 300)
    st = guard.collect(settings, UTC_NOW, listening=lambda h, p: True, disk_check=disk_ok)
    assert {i["key"]: i["level"] for i in st["items"]}["worker"] == "warn"


def test_check_writes_files_and_raises_then_clears_alerts(settings):
    touch_beat(settings, 4000)
    st = guard.check(settings, UTC_NOW, listening=lambda h, p: True,
                     disk_check=lambda s: DiskState(s.data_dir, 20.0, 1000.0, "warn"))
    js, txt = guard.status_paths(settings)
    assert json.loads(js.read_text(encoding="utf-8"))["level"] == st["level"] == "error"
    assert "ワーカーが止まっています" in txt.read_text(encoding="utf-8")
    keys = {a["key"] for a in alerts.recent(settings)}
    assert keys == {"guard-worker", "disk-warn"}
    # recovered: the keys are forgotten, so the next problem alerts at once
    touch_beat(settings, 5)
    guard.check(settings, UTC_NOW, listening=lambda h, p: True, disk_check=disk_ok)
    assert {a["key"] for a in alerts.recent(settings)} == set()


def test_read_status_and_its_age(settings):
    assert guard.read_status(settings) is None
    guard.check(settings, listening=lambda h, p: True, disk_check=disk_ok)
    status, age = guard.read_status(settings)
    assert status["items"] and 0 <= age < 5


def test_cli_report_exit_codes(settings, tmp_path, capsys):
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"data_dir: {settings.data_dir}\nhost: 127.0.0.1\nport: {settings.port}\n")
    assert guard.main(["--settings", str(cfg), "report"]) == 2   # no worker, no screens
    out = capsys.readouterr().out
    assert "[要対応]" in out
    assert guard.main(["--settings", str(cfg), "check"]) == 0    # never fails a unit
    assert guard.status_paths(settings)[1].is_file()


def test_the_status_page_shows_the_watch(env):
    settings, repo = env
    at = run_page("status")
    assert not at.exception and not any("見張り" in c.value for c in at.caption)   # no watch yet
    guard.check(settings, listening=lambda h, p: True, disk_check=disk_ok)
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value.startswith("見張り（") and "確認" in c.value for c in at.caption)
    # a watch that stopped updating is called out
    import os
    js = guard.status_paths(settings)[0]
    old = js.stat().st_mtime - 3 * 3600
    os.utime(js, (old, old))
    at = run_page("status")
    assert any("見張りの確認が" in w.value and "pfb-guard.timer" in w.value for w in at.warning)


def test_guard_units(tmp_path):
    units = systemd.render(tmp_path / "app", config=tmp_path / "s.yaml", home=tmp_path)
    svc, timer = units["pfb-guard.service"], units["pfb-guard.timer"]
    assert "-m pfb_app.ops.guard" in svc
    assert "Type=oneshot" in svc and f"--settings {tmp_path}/s.yaml check" in svc
    assert "OnBootSec=3min" in timer and "OnUnitActiveSec=5min" in timer
    assert "Unit=pfb-guard.service" in timer and "WantedBy=timers.target" in timer
    other = systemd.render(tmp_path / "app", prefix="x", guard_every="1min", home=tmp_path)
    assert "OnUnitActiveSec=1min" in other["x-guard.timer"]


# --- scripts/ops/install_services.sh and check_ops.sh (fake systemctl / tmux / loginctl) -------

import os
import socket
import subprocess
import sys
from pathlib import Path

from test_ops_review1 import fake_bin, unit_dir

ROOT = Path(__file__).resolve().parents[1]


def run_ops(tmp_path, script, *args, bin_dir, wait="4"):
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PFB_CONFIG")}
    env.update(HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "home" / ".config"),
               PFB_PYTHON=sys.executable, PFB_OPS_WAIT_SEC=wait, PFB_OPS_POLL_SEC="1",
               PATH=f"{bin_dir}:{env['PATH']}")
    res = subprocess.run(["bash", str(ROOT / "scripts" / "ops" / script), *args],
                         capture_output=True, text=True, env=env, timeout=120)
    res.all = res.stdout + res.stderr   # --apply sends stderr through the log's tee
    return res


def ops_config(tmp_path, port):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(f"data_dir: {data}\nhost: 127.0.0.1\nport: {port}\n"
                   f"backup_dir: {tmp_path / 'backups'}\n")
    return cfg, data


def changed_systemd(calls: Path) -> bool:
    log = calls.read_text() if calls.exists() else ""
    return any(w in log for w in ("enable --now", "daemon-reload", " disable"))


def test_plan_and_check_change_nothing(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    bin_dir, calls = fake_bin(tmp_path)
    for cmd in ("check", "install", "uninstall"):
        res = run_ops(tmp_path, "install_services.sh", cmd, "--config", str(cfg), bin_dir=bin_dir)
        assert res.returncode == 0, res.all
    assert "--apply" in res.stdout
    assert not unit_dir(tmp_path).exists() and not changed_systemd(calls)
    assert not (tmp_path / "home" / "pfb_deploy_logs").exists()   # no log: nothing was applied


def test_install_refuses_without_linger(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    bin_dir, calls = fake_bin(tmp_path)
    (bin_dir / "loginctl").write_text("#!/bin/sh\necho no\n")
    res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                  bin_dir=bin_dir)
    assert res.returncode == 1 and "enable-linger" in res.all
    assert not unit_dir(tmp_path).exists() and not changed_systemd(calls)


def test_install_refuses_while_tmux_runs_unless_asked(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    bin_dir, calls = fake_bin(tmp_path)
    (bin_dir / "tmux").write_text(f'#!/bin/sh\necho "tmux $*" >> "{calls}"\nexit 0\n')
    res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                  bin_dir=bin_dir)
    assert res.returncode == 1 and "--stop-tmux" in res.all
    assert not unit_dir(tmp_path).exists() and not changed_systemd(calls)


def test_install_refuses_while_a_job_runs(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    settings = Settings(data_dir=data)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    from test_ops_panel import stopped_job
    stopped_job(repo)
    conn.execute("UPDATE jobs SET status='running'")
    conn.commit()
    bin_dir, calls = fake_bin(tmp_path)
    res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                  bin_dir=bin_dir)
    assert res.returncode == 1 and "解析が 1 件動いています" in res.all
    assert not changed_systemd(calls)


def test_install_stops_without_changes_when_the_screens_already_answer(tmp_path):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        cfg, data = ops_config(tmp_path, srv.getsockname()[1])
        bin_dir, calls = fake_bin(tmp_path)
        res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                      bin_dir=bin_dir)
    assert res.returncode == 1 and "何も変更せずに止めました" in res.all
    assert not unit_dir(tmp_path).exists() and not changed_systemd(calls)


def test_install_rolls_back_when_nothing_comes_up(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    bin_dir, calls = fake_bin(tmp_path, enabled=False, active=False)   # nothing ever active
    res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                  bin_dir=bin_dir)
    assert res.returncode == 1 and "元に戻します" in res.all, res.all
    log = calls.read_text()
    assert "enable --now pfb-worker.service pfb-ui.service pfb-backup.timer pfb-guard.timer" in log
    assert not (unit_dir(tmp_path) / "pfb-worker.service").exists()   # removed again
    assert "disable --now" in log


def test_install_success_path(tmp_path):
    port = closed_port()
    cfg, data = ops_config(tmp_path, port)
    bin_dir, calls = fake_bin(tmp_path)
    # a fake `systemctl enable --now` that brings the "worker" (a beat) and the "screens" (a
    # listener) up in the background, and answers is-active from a marker file
    helper = tmp_path / "bring_up.py"
    helper.write_text(
        "import socket, time, pathlib, sys\n"
        "beat = pathlib.Path(sys.argv[1]); s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', int(sys.argv[2]))); s.listen()\n"
        "for _ in range(80):\n    beat.touch(); time.sleep(0.25)\n")
    marker = tmp_path / "enabled"
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        f'echo "systemctl $*" >> "{calls}"\n'
        'case "$*" in *"enable --now"*)\n'
        f'  touch "{marker}"\n'
        f'  nohup "{sys.executable}" "{helper}" "{data / "worker.beat"}" {port} >/dev/null 2>&1 &\n'
        ";; esac\n"
        'case "$*" in *is-enabled*) exit 1 ;; *is-active*)\n'
        f'  [ -e "{marker}" ] && exit 0 || exit 3 ;; esac\nexit 0\n')
    res = run_ops(tmp_path, "install_services.sh", "install", "--apply", "--config", str(cfg),
                  bin_dir=bin_dir, wait="30")
    assert res.returncode == 0, res.all
    assert "OK: ワーカーと画面が systemd で動いています" in res.all
    assert (unit_dir(tmp_path) / "pfb-worker.service").is_file()
    assert (tmp_path / "home" / "pfb_deploy_logs").is_dir()   # the apply was logged


def test_check_ops_reports_levels_and_never_writes(tmp_path):
    cfg, data = ops_config(tmp_path, closed_port())
    bin_dir, calls = fake_bin(tmp_path)
    res = run_ops(tmp_path, "check_ops.sh", "--config", str(cfg), bin_dir=bin_dir)
    assert res.returncode == 2, res.stdout   # nothing runs
    assert "[要対応]" in res.stdout and "結果:" in res.stdout
    assert not (data / "ops_status.txt").exists()   # report writes nothing
    assert not changed_systemd(calls)
