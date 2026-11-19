"""Backups of data/app.db (ops.backup): online copy + manifest, verify, rotate, restore,
the pre-migration copy and the CLI. Temp dirs and synthetic rows only."""
import gzip
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.ops import alerts, backup
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

ROOT = Path(__file__).resolve().parents[1]


def closed_port() -> int:
    """A local port nobody listens on (restore refuses while the UI's port answers)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "app" / "data", backup_keep=3, host="127.0.0.1",
                    port=closed_port())


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    yield Repo(conn, settings)
    conn.close()


def add_work(repo, title="作業A"):
    aid = repo.create_assignment(title, "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "グループ1", {"cameras": {}})
    repo.add_subject(sid, None, "Aさん", None, {})
    return aid, sid


def data_file(settings, rel, data=b"x"):
    path = settings.data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


TODAY = datetime.now().date()   # backups older than backup_keep days age out


def at(hour, minute=0, second=0):
    return datetime(TODAY.year, TODAY.month, TODAY.day, hour, minute, second)


def day(dt=None) -> str:
    return (dt or at(0)).strftime("%Y%m%d")


def test_backup_is_complete_verified_and_beside_data_dir(settings, repo):
    add_work(repo)
    data_file(settings, "assignments/1/submissions/1/frames/cam1/000001.jpg", b"jpg")
    data_file(settings, "uploads/ab12cd34_clip.mp4", b"video")
    (settings.data_dir / "worker.beat").touch()
    (settings.data_dir / "logs").mkdir()
    (settings.data_dir / "logs" / "worker.log").write_text("x")
    path = backup.run_backup(settings, now=at(3, 30))
    assert path == settings.data_dir.parent / "backups" / f"app-{day()}-033000"
    assert sorted(p.name for p in path.iterdir()) == ["app.db", "data_files.tsv.gz",
                                                      "manifest.json"]
    m = backup.read_manifest(path)
    assert m["db"]["integrity"] == "ok"
    assert m["db"]["tables"]["assignments"] == 1 and m["db"]["tables"]["subjects"] == 1
    assert m["data"]["files"] == 2 and m["data"]["bytes"] == 8
    assert set(m["data"]["by_top"]) == {"assignments", "uploads"}
    with gzip.open(path / "data_files.tsv.gz", "rt", encoding="utf-8") as f:
        listed = [line.split("\t")[0] for line in f.read().splitlines()[1:]]
    assert listed == ["assignments/1/submissions/1/frames/cam1/000001.jpg",
                      "uploads/ab12cd34_clip.mp4"]
    assert backup.verify(path).ok
    # the copy is one self-contained file (no WAL beside it) and opens on its own
    conn = sqlite3.connect(path / "app.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert conn.execute("SELECT title FROM assignments").fetchone()[0] == "作業A"
    conn.close()
    last = backup.last_result(settings)
    assert last["ok"] and last["name"] == path.name
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".tmp-")]


def test_backup_sees_committed_rows_only_while_a_writer_is_open(settings, repo):
    add_work(repo, "保存済み")
    writer = connect(files.db_path(settings))
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO audit_log (ts, action, target, detail) VALUES ('t','a',NULL,'{}')")
    try:
        path = backup.run_backup(settings, now=at(1))
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    assert backup.read_manifest(path)["db"]["tables"]["audit_log"] == 0
    assert backup.verify(path).ok


def test_verify_finds_a_changed_copy(settings, repo):
    add_work(repo)
    path = backup.run_backup(settings, now=at(2))
    conn = sqlite3.connect(path / "app.db")
    conn.execute("DELETE FROM subjects")
    conn.commit()
    conn.close()
    result = backup.verify(path)
    assert not result.ok
    assert any("一致しません" in p for p in result.problems)


def test_rotation_keeps_the_newest(settings, repo):
    add_work(repo)
    made = [backup.run_backup(settings, now=at(1, m)).name for m in range(5)]
    assert [p.name for p in backup.list_backups(settings)] == made[-3:]
    # someone else's folder in backup_dir is never touched
    other = backup.backup_root(settings) / "keep-me"
    other.mkdir()
    backup.run_backup(settings, now=at(2))
    assert other.is_dir() and len(backup.list_backups(settings)) == 3


def test_same_second_gets_a_new_name(settings, repo):
    add_work(repo)
    a = backup.run_backup(settings, now=at(4))
    b = backup.run_backup(settings, now=at(4))
    assert a.name == f"app-{day()}-040000" and b.name == f"app-{day()}-040000-1"


def test_failure_is_recorded_and_alerted(settings, caplog):
    with pytest.raises(backup.BackupError):
        backup.run_backup(settings, now=at(3))   # no database yet
    last = backup.last_result(settings)
    assert last["ok"] is False and "データベースがありません" in last["error"]
    assert any(a["key"] == "backup-failed" for a in alerts.recent(settings))


def test_backup_dir_inside_data_dir_is_refused(tmp_path):
    s = Settings(data_dir=tmp_path / "data", backup_dir=tmp_path / "data" / "bk")
    conn = connect(files.db_path(s))
    init_schema(conn)
    conn.close()
    with pytest.raises(backup.BackupError, match="data_dir の中"):
        backup.run_backup(s)


def test_restore_round_trip(settings, repo):
    aid, _ = add_work(repo, "戻したい作業")
    data_file(settings, "uploads/ab12cd34_clip.mp4", b"video")
    path = backup.run_backup(settings, now=at(3))
    # afterwards: the work is renamed, another added, and the upload is gone
    repo.rename_assignment(aid, "変わった名前")
    add_work(repo, "後から足した作業")
    (settings.data_dir / "uploads" / "ab12cd34_clip.mp4").unlink()
    repo.conn.close()
    result = backup.restore(settings, path.name, now=at(9))
    assert result.previous == settings.data_dir / f"app.db.before-restore-{day()}-090000"
    assert result.missing_files == 1 and result.missing_sample == ["uploads/ab12cd34_clip.mp4"]
    conn = connect(files.db_path(settings))
    assert [r[0] for r in conn.execute("SELECT title FROM assignments")] == ["戻したい作業"]
    conn.close()
    prev = sqlite3.connect(result.previous)
    assert {r[0] for r in prev.execute("SELECT title FROM assignments")} == {
        "変わった名前", "後から足した作業"}
    prev.close()
    # the restored database works with the app as before
    conn = connect(files.db_path(settings))
    init_schema(conn)
    assert Repo(conn, settings).list_assignments()[0]["title"] == "戻したい作業"
    conn.close()


def test_restore_refuses_while_the_worker_runs(settings, repo):
    add_work(repo)
    path = backup.run_backup(settings, now=at(3))
    files.worker_beat_path(settings).touch()
    with pytest.raises(backup.RestoreError, match="ワーカーが動いています"):
        backup.restore(settings, path.name)
    assert backup.restore(settings, path.name, force=True).name == path.name


def test_restore_refuses_a_bad_backup(settings, repo):
    add_work(repo)
    path = backup.run_backup(settings, now=at(3))
    (path / "app.db").write_bytes(b"not a database")
    with pytest.raises(backup.RestoreError, match="確認に失敗"):
        backup.restore(settings, path.name)
    with pytest.raises(backup.RestoreError, match="見つかりません"):
        backup.restore(settings, "app-19990101-000000")


def test_pre_migration_copy_before_adding_a_column(tmp_path):
    db = tmp_path / "data" / "app.db"
    db.parent.mkdir()
    conn = connect(db)
    init_schema(conn)
    assert not list(db.parent.glob("app.db.pre-migrate-*"))   # a new database: no copy
    conn.close()
    # an older database without the newest column
    raw = sqlite3.connect(db)
    raw.execute("ALTER TABLE subjects DROP COLUMN approved_auto")
    raw.commit()
    raw.close()
    conn = connect(db)
    init_schema(conn)
    copies = list(db.parent.glob("app.db.pre-migrate-*"))
    assert len(copies) == 1
    old = sqlite3.connect(copies[0])
    assert "approved_auto" not in {r[1] for r in old.execute("PRAGMA table_info(subjects)")}
    old.close()
    assert "approved_auto" in {r[1] for r in conn.execute("PRAGMA table_info(subjects)")}
    init_schema(conn)   # nothing missing any more: no second copy
    assert len(list(db.parent.glob("app.db.pre-migrate-*"))) == 1
    conn.close()


def test_backup_skips_its_own_copies(settings, repo):
    add_work(repo)
    data_file(settings, "app.db.before-restore-20261001-000000")
    data_file(settings, "app.db.pre-migrate-20261001-000000")
    data_file(settings, "alerts.json", b"{}")
    data_file(settings, "alerts.json.lock", b"")
    path = backup.run_backup(settings, now=at(5))
    assert backup.read_manifest(path)["data"]["files"] == 0


def run_cli(settings, *args, env_extra=None):
    cfg = settings.data_dir.parent / "pfb.yaml"
    cfg.write_text(json.dumps({"data_dir": str(settings.data_dir), "backup_keep": 3,
                               "host": settings.host, "port": settings.port}),
                   encoding="utf-8")
    env = {**os.environ, "PFB_CONFIG": str(cfg), **(env_extra or {})}
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run([str(ROOT / "scripts" / "backup.sh"), *args], capture_output=True,
                          text=True, env={**env, "PFB_PYTHON": sys.executable}, timeout=120)


def test_scripts_run_list_verify_restore(settings, repo):
    add_work(repo)
    repo.conn.close()
    made = run_cli(settings, "run")
    assert made.returncode == 0, made.stderr
    name = Path(made.stdout.strip()).name
    listed = run_cli(settings, "list")
    assert listed.returncode == 0 and listed.stdout.startswith(name)
    checked = run_cli(settings, "verify")
    assert checked.returncode == 0 and checked.stdout.strip() == f"{name}: ok"
    assert run_cli(settings, "verify", "app-19990101-000000").returncode == 1
    restored = subprocess.run(
        [str(ROOT / "scripts" / "restore.sh"), name], capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=120,
        env={**{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
             "PFB_CONFIG": str(settings.data_dir.parent / "pfb.yaml"),
             "PFB_PYTHON": sys.executable})
    assert restored.returncode == 0, restored.stderr
    assert f"restored {name}" in restored.stdout
    assert (settings.data_dir / "logs" / "backup.log").is_file()


# --- review rel2: temp files stay out of the manifest; organisation/ is backed up -------------

def test_the_manifest_leaves_temp_files_out(settings, repo):
    add_work(repo)
    keep = ["assignments/1/submissions/1/frames/cam1/000001.jpg", "proxies/0123abcd.json",
            "models/yolo11l.pt"]
    temp = ["assignments/1/out.json.tmp", "assignments/1/.p1.4242.tmp.jpg",
            "pfb3x3_runtime/engine.yaml.tmp", "models/yolo11l.pt.tmp.4242",
            "proxies/0123abcd.part.mp4", "uploads/.partial/u1.part",
            "uploads/.partial/u1.json", "uploads/clip.mp4.0123456789abcdef.tmp",
            "organisation/.logo.png.4242.tmp", "assignments/1/.tmp-staging/a.jpg"]
    for rel in keep + temp:
        data_file(settings, rel)
    path = backup.run_backup(settings, now=at(4))
    with gzip.open(path / "data_files.tsv.gz", "rt", encoding="utf-8") as f:
        listed = [line.split("\t")[0] for line in f.read().splitlines()[1:]]
    assert sorted(listed) == sorted(keep)
    assert backup.read_manifest(path)["data"]["files"] == len(keep)


def test_organisation_is_backed_up_verified_and_restored(settings, repo):
    add_work(repo)
    data_file(settings, "organisation/organisation.json", b'{"name": "A"}')
    data_file(settings, "organisation/logo.png", b"png")
    data_file(settings, "organisation/.logo.png.4242.tmp", b"half")
    path = backup.run_backup(settings, now=at(3))
    assert sorted(p.name for p in (path / "organisation").iterdir()) == [
        "logo.png", "organisation.json"]
    assert backup.verify(path).ok
    # afterwards the settings change and a logo is added
    data_file(settings, "organisation/organisation.json", b'{"name": "B"}')
    data_file(settings, "organisation/logo.svg", b"<svg/>")
    repo.conn.close()
    backup.restore(settings, path.name, now=at(9))
    org = settings.data_dir / "organisation"
    assert sorted(p.name for p in org.iterdir()) == ["logo.png", "organisation.json"]
    assert (org / "organisation.json").read_bytes() == b'{"name": "A"}'
    before = settings.data_dir / f"organisation.before-restore-{day()}-090000"
    assert (before / "organisation.json").read_bytes() == b'{"name": "B"}'
    assert (before / "logo.svg").is_file()


def test_a_changed_organisation_copy_fails_verify(settings, repo):
    add_work(repo)
    data_file(settings, "organisation/organisation.json", b'{"name": "A"}')
    path = backup.run_backup(settings, now=at(3))
    (path / "organisation" / "organisation.json").write_bytes(b'{"name": "X"}')
    assert any("団体の設定" in p for p in backup.verify(path).problems)


def test_a_backup_without_organisation_leaves_the_current_one(settings, repo):
    add_work(repo)
    path = backup.run_backup(settings, now=at(3))           # no organisation/ yet
    data_file(settings, "organisation/organisation.json", b'{"name": "A"}')
    repo.conn.close()
    backup.restore(settings, path.name, now=at(9))
    assert (settings.data_dir / "organisation" / "organisation.json").is_file()
