"""Ops review round 2 (R1–R6). Temp dirs, a temp HOME (so the "default" data_dir is a temp
path too), fake systemctl / tmux on PATH: never the live app."""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.ops import backup, supervise, systemd
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_ops_backup import closed_port
from test_ops_review1 import config, fake_bin, unit_dir

ROOT = Path(__file__).resolve().parents[1]
linux_only = pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc")


def install(tmp_path, *args, bin_dir=None, cwd=None):
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PFB_CONFIG")}
    env.update(HOME=str(tmp_path / "home"), XDG_CONFIG_HOME=str(tmp_path / "home" / ".config"))
    if bin_dir is not None:
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return subprocess.run(["bash", str(ROOT / "scripts" / "install_user_services.sh"),
                           "--python", sys.executable, *args], capture_output=True, text=True,
                          env=env, timeout=120, cwd=cwd)


def holder(path: Path) -> subprocess.Popen:
    """Another process that keeps `path` open (as a UI or worker keeps app.db open)."""
    proc = subprocess.Popen([sys.executable, "-c",
                             "import sys,time; f=open(sys.argv[1],'rb'); print('ok',flush=True);"
                             " time.sleep(60)", str(path)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ok"
    return proc


def make_db(data_dir: Path) -> Path:
    s = Settings(data_dir=data_dir)
    conn = connect(files.db_path(s))
    init_schema(conn)
    Repo(conn, s).create_assignment("作業", "video", "説明", "", "standard", False)
    conn.close()
    return data_dir / "app.db"


# --- R1: open handles on app.db; tmux for the default data dir, any prefix --------------------

@linux_only
def test_db_open_by_finds_another_process(tmp_path):
    db = make_db(tmp_path / "data")
    assert supervise.db_open_by(db) == []
    with open(db, "rb"):            # this process itself never counts
        assert supervise.db_open_by(db) == []
    proc = holder(db)
    try:
        assert supervise.db_open_by(db) == [proc.pid]
    finally:
        proc.kill()
        proc.wait()
    assert supervise.db_open_by(tmp_path / "missing.db") == []


@linux_only
def test_preflight_and_restore_refuse_an_open_database(tmp_path):
    s = Settings(data_dir=tmp_path / "data", host="127.0.0.1", port=closed_port())
    db = make_db(s.data_dir)
    path = backup.run_backup(s)
    proc = holder(db)
    try:
        problems = systemd.preflight(s, s.host, s.port)
        assert any(str(proc.pid) in p and "app.db" in p for p in problems)
        with pytest.raises(backup.RestoreError, match=str(proc.pid)):
            backup.restore(s, path.name)
    finally:
        proc.kill()
        proc.wait()
    assert systemd.preflight(s, s.host, s.port) == []
    backup.restore(s, path.name)


def test_tmux_sessions_block_the_default_data_for_any_prefix(tmp_path):
    home_data = tmp_path / "home" / "pfb_app" / "data"
    bin_dir, _ = fake_bin(tmp_path)
    (bin_dir / "tmux").write_text('#!/bin/sh\ncase "$*" in *has-session*pfb-worker*) exit 0;; esac\n'
                                  "exit 1\n")
    # prefix pfb on the default data (no --config): refused for the tmux worker
    res = install(tmp_path, "--enable", "--host", "127.0.0.1", "--port", str(closed_port()),
                  bin_dir=bin_dir)
    assert res.returncode == 1 and "pfb-worker" in res.stderr, res.stderr
    # prefix pfb with its own data: the live tmux sessions do not matter
    other = config(tmp_path, tmp_path / "data")
    res = install(tmp_path, "--enable", "--config", str(other), "--host", "127.0.0.1",
                  "--port", str(closed_port()), bin_dir=bin_dir)
    assert res.returncode == 0, res.stderr
    assert not home_data.exists()


def test_preflight_live_checks_only_for_the_default_data(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data")
    monkeypatch.setattr(systemd, "_default_data_dir", lambda: (tmp_path / "data").resolve())
    assert systemd.preflight(s, "127.0.0.1", closed_port(), live_checks=lambda: ["pfb-ui"])
    monkeypatch.setattr(systemd, "_default_data_dir", lambda: tmp_path / "elsewhere")
    assert systemd.preflight(s, "127.0.0.1", closed_port(), live_checks=lambda: ["pfb-ui"]) == []


# --- R2: config paths, the default data for another prefix, unreadable locks ------------------

def test_a_relative_config_is_read_from_the_callers_folder(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    config(work, tmp_path / "data")       # work/pfb.yaml
    out = tmp_path / "u"
    res = install(tmp_path, "--prefix", "pfb-t", "--config", "pfb.yaml", "--unit-dir", str(out),
                  cwd=work)
    assert res.returncode == 0, res.stderr
    assert f"--settings {work / 'pfb.yaml'}" in (out / "pfb-t-worker.service").read_text()


def test_a_missing_config_is_refused_plainly(tmp_path):
    res = install(tmp_path, "--prefix", "pfb-t", "--config", "nope.yaml",
                  "--unit-dir", str(tmp_path / "u"), cwd=tmp_path)
    assert res.returncode == 2 and "設定ファイルが見つかりません" in res.stderr
    assert not (tmp_path / "u").exists()


def test_another_prefix_may_not_point_at_the_default_data(tmp_path):
    cfg = config(tmp_path, "~/pfb_app/data")   # the default, through the temp HOME
    res = install(tmp_path, "--prefix", "pfb-t", "--config", str(cfg),
                  "--unit-dir", str(tmp_path / "u"))
    assert res.returncode == 2 and "本番" in res.stderr
    assert not (tmp_path / "u").exists()


def test_an_unreadable_lock_is_reported_and_refused(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    s = Settings(data_dir=tmp_path / "data", host="127.0.0.1", port=closed_port())
    path = backup.run_backup(s) if make_db(s.data_dir) else None
    lock = s.data_dir / "worker.lock"
    lock.write_text("1")
    lock.chmod(0)
    try:
        assert any("確認できません" in p for p in systemd.preflight(s, s.host, s.port))
        with pytest.raises(backup.RestoreError, match="確認できません"):
            backup.restore(s, path.name)
    finally:
        lock.chmod(0o644)


# --- R3 / R4: before-restore copies stay; time-based retention -------------------------------

def stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S")


def test_before_restore_copies_are_never_removed_and_old_ones_age_out(tmp_path):
    s = Settings(data_dir=tmp_path / "data", backup_keep=14, backup_keep_weekly=0)
    make_db(s.data_dir)
    now = datetime.now()
    for days in (40, 20, 13, 1):
        (s.data_dir / f"app.db.pre-migrate-{stamp(now - timedelta(days=days))}").write_bytes(b"x")
        (s.data_dir / f"app.db.before-restore-{stamp(now - timedelta(days=days))}").write_bytes(b"x")
    root = backup.backup_root(s)
    old = [backup.run_backup(s, now=now - timedelta(days=d)) for d in (30, 15, 2)]
    newest = backup.run_backup(s, now=now)
    names = [p.name for p in backup.list_backups(s)]
    assert names == [old[2].name, newest.name]               # 30 and 15 days: gone
    assert len(list(s.data_dir.glob("app.db.before-restore-*"))) == 4   # all kept
    left = sorted(p.name for p in s.data_dir.glob("app.db.pre-migrate-*"))
    assert left == sorted(f"app.db.pre-migrate-{stamp(now - timedelta(days=d))}" for d in (13, 1))
    assert root.is_dir()


def test_the_newest_backup_is_kept_whatever_its_age(tmp_path):
    s = Settings(data_dir=tmp_path / "data", backup_keep=1)
    make_db(s.data_dir)
    only = backup.run_backup(s, now=datetime.now() - timedelta(days=30))
    assert [p.name for p in backup.list_backups(s)] == [only.name]


def test_the_privacy_note_is_accurate():
    from pfb_app.ui import ops_panel
    note = ops_panel.backup_privacy_note(Settings())
    assert "14日" in note and "管理者が消すまで残ります" in note
    assert "記録が始まる前の作業のモデルの回答は、整理の手順を実行するまで残ります" in note
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "管理者が消すまで残ります" in guide
    assert "記録が始まる前の作業のモデルの回答は、整理の手順を実行するまで残ります" in guide


# --- R5: --uninstall checks whose units they are ----------------------------------------------

def test_uninstall_leaves_another_checkouts_units(tmp_path):
    out = tmp_path / "u"
    out.mkdir()
    for name, text in systemd.render(tmp_path / "other-checkout").items():
        (out / name).write_text(text)
    res = install(tmp_path, "--unit-dir", str(out), "--uninstall")
    assert res.returncode == 1 and "other-checkout" in res.stderr
    assert len(list(out.iterdir())) == 7
    res = install(tmp_path, "--unit-dir", str(out), "--uninstall", "--force")
    assert res.returncode == 0 and list(out.iterdir()) == []


# --- R6: folders compared resolved ------------------------------------------------------------

def test_a_symlinked_app_folder_is_the_same_checkout(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    (tmp_path / "pfb-worker.service").write_text(systemd.render(link)["pfb-worker.service"])
    assert systemd.same_target(tmp_path, "pfb", real)
    assert not systemd.same_target(tmp_path, "pfb", tmp_path / "other")
    assert systemd.same_target(tmp_path / "empty", "pfb", real)   # nothing written yet


def test_live_ui_checks_compare_resolved_data_dirs(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setattr(backup, "_default_data_dir", lambda: real.resolve())
    s = Settings(data_dir=tmp_path / "link", host="127.0.0.1", port=closed_port())
    assert backup.ui_running(s, live_checks=lambda: True)
