"""Backups of data/app.db (SQLite's online backup) with a data-dir manifest, verified,
rotated, and restorable.

A backup is a folder <backup_dir>/app-YYYYmmdd-HHMMSS/ with
- app.db: a consistent copy taken through the online backup API while the app runs (WAL
  included; the copy is a single self-contained file, journal_mode=DELETE);
- manifest.json: when, the app's commit, the copy's size / sha256 / integrity_check / row
  counts per table, and a summary of data_dir (files and bytes per top-level folder);
- data_files.tsv.gz: every file in data_dir (path, size, mtime, f/l), so a restore can say
  which videos, frames or engine outputs the restored database refers to that are gone;
  files being written (temp names: *.tmp, *.tmp.*, *.part, *.part.*, a .tmp-… folder, the
  uploads' .partial/) are left out;
- organisation/: a copy of data_dir/organisation/ (団体の設定: name, accent, logo; small, made
  by hand, so it cannot come back from anywhere else), its files' sha256 in the manifest.
Videos and frames are not copied (they are large and come from the inbox / the phone).

It is written into a hidden .tmp-… folder, verified (sha256, integrity_check, row counts)
and only then renamed into place, so a listed backup is always a complete, verified one.
After each run the oldest backups beyond backup_keep are removed, except the newest one of each
of the backup_keep_weekly weeks after those (only folders named like ours that hold a
manifest; max_age_days() is the longest a backup lives), and <backup_dir>/last_result.json records the outcome for the
status page; a failure also sends an operator alert (ops.alerts).

Restore (`restore`): refuses while a worker is running (it holds worker.lock, or its beat is
fresh) or a UI may use the data (its host:port answers; for the live data_dir also the pfb-ui
unit or tmux session) or another process has app.db open (/proc/*/fd), unless forced, verifies the backup, keeps the current database as app.db.before-restore-<time>
(online backup again), then copies the backup into data/app.db through the backup API (so
the WAL files stay consistent), checks it and reports the manifest's files that are missing
now. A backup with organisation/ also puts that back (the current folder is kept as
organisation.before-restore-<time>); one without it leaves the current folder alone. A
restored older database never deletes newer work folders: the worker's orphan sweep
only removes folders the database itself recorded as deleted (services.deletion).

Pre-migration copy (`pre_migration_copy`): storage.db.init_schema calls it once before it
adds a missing column to an existing database (app.db.pre-migrate-<time> beside it).

A backup is refused while the free space where backup_dir lives, after this backup, would be
below disk_refuse_gb (the same floor that keeps the database writable): the disk never fills
because of a backup.

Restore rehearsal (`rehearse`): puts a backup back into a scratch folder (never the live data),
checks integrity_check, the row counts against the manifest and the organisation files, says how
far the live database is from it, and records the outcome (last_rehearsal.json) for check_ops.sh.

CLI (scripts/backup.sh, scripts/restore.sh, the pfb-backup timer):
  python -m pfb_app.ops.backup [--settings PATH] run | list | verify [NAME] | rehearse [NAME]
                                                  | restore NAME [--force]
"""
import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfb_app.config import Settings, load_settings
from pfb_app.storage import files

log = logging.getLogger(__name__)

FORMAT = 1
NAME_RE = re.compile(r"^app-\d{8}-\d{6}(-\d+)?$")
DB_NAME = "app.db"
MANIFEST = "manifest.json"
FILE_LIST = "data_files.tsv.gz"
LAST_RESULT = "last_result.json"
LAST_REHEARSAL = "last_rehearsal.json"
GB = 1024 ** 3
BACKUP_SLACK_BYTES = 64 * 1024 * 1024   # the file list, the manifest, the organisation copy
# online backup in one step: a step-wise copy restarts whenever another connection commits
# between steps (WAL writers), so under load it might never finish; one step holds a read
# snapshot instead (writers carry on in the WAL meanwhile)
PAGES_PER_STEP = -1
WORKER_FRESH_SEC = 120         # a beat younger than this: the worker is running
COPIES_KEEP = 3                # app.db.pre-migrate-* kept beside app.db (and none older than backup_keep days)
# data_dir entries that are not data: the live database files, beats, our own state and logs
_SKIP_TOP = {"app.db", "app.db-wal", "app.db-shm", "worker.beat", "worker.lock",
             "worker.running", "alerts.json", "alerts.json.lock", "logs",
             "ops_status.json", "ops_status.txt"}
BEFORE_RESTORE = "app.db.before-restore-"   # kept until an operator removes it
PRE_MIGRATE = "app.db.pre-migrate-"         # rotated (rotate_copies)
ORG = "organisation"                        # services.organisation.org_dir, copied
ORG_BEFORE_RESTORE = "organisation.before-restore-"   # kept like BEFORE_RESTORE
_SKIP_PREFIX = (BEFORE_RESTORE, PRE_MIGRATE, ORG_BEFORE_RESTORE)
# names writers use while a file is incomplete (renamed or removed when done): x.json.tmp,
# .x.<pid>.tmp, .x.<pid>.tmp.jpg, x.yaml.tmp, x.pt.tmp.<pid>, <key>.part.mp4, <uid>.part
_TEMP_PARTS = {"tmp", "part"}
_TEMP_DIRS = {".partial"}                   # services.chunked_upload: uploads in progress


def _is_temp(name: str) -> bool:
    return name in _TEMP_DIRS or name.startswith(".tmp-") \
        or bool(_TEMP_PARTS & set(name.split(".")[1:]))


class BackupError(Exception):
    pass


class RestoreError(Exception):
    pass


@dataclass
class VerifyResult:
    name: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class RestoreResult:
    name: str
    previous: Path | None          # the database kept from before the restore
    missing_files: int
    missing_sample: list[str]


def backup_root(settings: Settings) -> Path:
    if settings.backup_dir is not None:
        return settings.backup_dir
    return settings.data_dir.parent / "backups"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _app_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        " ORDER BY name")]
    return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}


def integrity(conn: sqlite3.Connection) -> str:
    rows = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    return "ok" if rows == ["ok"] else "; ".join(rows[:5])


def online_copy(src_path: Path, dest_path: Path, pages: int | None = None,
                progress=None) -> None:
    """A consistent single-file copy of a live database (the online backup API)."""
    src = sqlite3.connect(str(src_path), timeout=30)
    try:
        dst = sqlite3.connect(str(dest_path))
        try:
            src.backup(dst, pages=PAGES_PER_STEP if pages is None else pages,
                       progress=progress, sleep=0.01)
            dst.execute("PRAGMA journal_mode=DELETE")
        finally:
            dst.close()
    finally:
        src.close()


def _data_entries(data_dir: Path):
    """(relative path, size, mtime, kind) for every file and link in data_dir, not following
    links, without the live database, beats, logs and our own copies."""
    if not data_dir.is_dir():
        return
    for root, dirs, names in os.walk(data_dir):
        rel_root = Path(root).relative_to(data_dir)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in _SKIP_TOP and not d.startswith(_SKIP_PREFIX)]
        dirs[:] = [d for d in dirs if not _is_temp(d)]
        links = [d for d in dirs if (Path(root) / d).is_symlink()]
        dirs[:] = sorted(d for d in dirs if d not in links)   # one fixed order on every filesystem
        for name in sorted([*names, *links]):
            rel = rel_root / name
            if rel_root == Path(".") and (name in _SKIP_TOP or name.startswith(_SKIP_PREFIX)):
                continue
            if _is_temp(name):
                continue
            try:
                st = (Path(root) / name).lstat()
            except OSError:
                continue
            kind = "l" if (Path(root) / name).is_symlink() else "f"
            yield rel.as_posix(), st.st_size, int(st.st_mtime), kind


def _org_files(folder: Path) -> list[Path]:
    """The regular files directly in an organisation folder (no links, no temp files)."""
    try:
        return sorted(p for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                      and not _is_temp(p.name))
    except OSError:
        return []


def _copy_org(settings: Settings, dest: Path) -> dict[str, str] | None:
    """Copy data_dir/organisation/ into `dest`/organisation; {file name: sha256}, or None
    when there is nothing to copy."""
    found = _org_files(settings.data_dir / ORG)
    if not found:
        return None
    (dest / ORG).mkdir()
    out = {}
    for src in found:
        shutil.copy2(src, dest / ORG / src.name)
        out[src.name] = _sha256(dest / ORG / src.name)
    return out


def _restore_org(settings: Settings, path: Path, stamp: str) -> Path | None:
    """Put the backup's organisation/ back (written beside, then swapped in); the current
    folder is kept as organisation.before-restore-<stamp>, which is returned."""
    want = read_manifest(path).get(ORG)
    if not want:
        return None
    live = settings.data_dir / ORG
    new = settings.data_dir / f".tmp-{ORG}-{stamp}"
    shutil.rmtree(new, ignore_errors=True)
    shutil.copytree(path / ORG, new)
    previous = None
    if live.exists():
        previous = settings.data_dir / f"{ORG_BEFORE_RESTORE}{stamp}"
        os.rename(live, previous)
    os.rename(new, live)
    return previous


def _new_name(root: Path, now: datetime) -> str:
    base = now.strftime("app-%Y%m%d-%H%M%S")
    name, n = base, 0
    while (root / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


def list_backups(settings: Settings) -> list[Path]:
    """Complete backups (named like ours, with a manifest), oldest first."""
    root = backup_root(settings)
    try:
        dirs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                and NAME_RE.match(p.name) and (p / MANIFEST).is_file()]
    except OSError:
        return []
    return sorted(dirs, key=lambda p: p.name)


def read_manifest(path: Path) -> dict:
    return json.loads((path / MANIFEST).read_text(encoding="utf-8"))


def verify(path: Path) -> VerifyResult:
    """sha256, integrity_check and row counts of a backup against its manifest."""
    result = VerifyResult(path.name)
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError) as e:
        result.problems.append(f"manifest を読めません: {type(e).__name__}")
        return result
    db = path / DB_NAME
    if not db.is_file():
        result.problems.append("app.db がありません")
        return result
    want = manifest.get("db", {})
    if _sha256(db) != want.get("sha256"):
        result.problems.append("app.db の内容が manifest と一致しません")
    try:
        conn = _ro(db)
        try:
            check = integrity(conn)
            counts = table_counts(conn)
        finally:
            conn.close()
    except sqlite3.Error as e:
        result.problems.append(f"app.db を開けません: {e}")
        return result
    if check != "ok":
        result.problems.append(f"integrity_check: {check}")
    if counts != want.get("tables"):
        result.problems.append("行数が manifest と一致しません")
    org = manifest.get(ORG) or {}
    copied = {p.name: _sha256(p) for p in _org_files(path / ORG)}
    if copied != org:
        result.problems.append("団体の設定のコピーが manifest と一致しません")
    return result


def _write_last(settings: Settings, record: dict) -> None:
    root = backup_root(settings)
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f".{LAST_RESULT}.tmp"
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, root / LAST_RESULT)
    except OSError:
        log.exception("バックアップの結果を記録できませんでした")


def last_result(settings: Settings) -> dict | None:
    """The newest run's outcome: {"at", "ok", "name", "error", …} (None: never ran)."""
    try:
        return json.loads((backup_root(settings) / LAST_RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stamp(name: str, prefix: str) -> datetime | None:
    """The time in a name like app-20261023-033000(-1) or app.db.pre-migrate-20261023-033000."""
    try:
        return datetime.strptime(name[len(prefix):len(prefix) + 15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _too_old(name: str, prefix: str, cutoff: datetime) -> bool:
    at = _stamp(name, prefix)
    return at is not None and at < cutoff


def rotate_copies(settings: Settings, keep: int = COPIES_KEEP,
                  now: datetime | None = None) -> list[str]:
    """The app.db.pre-migrate-* copies beside the database: the newest `keep` are kept, and
    none older than backup_keep days. app.db.before-restore-* copies are never removed
    automatically (an operator decides). Returns the removed names."""
    cutoff = (now or datetime.now()) - timedelta(days=max(1, settings.backup_keep))
    prefix = PRE_MIGRATE
    try:
        copies = sorted(p for p in settings.data_dir.iterdir()
                        if p.name.startswith(prefix) and p.is_file() and not p.is_symlink())
    except OSError:
        return []
    removed = []
    for i, path in enumerate(copies):
        if i < len(copies) - keep or _too_old(path.name, prefix, cutoff):
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                log.exception("古いコピーを削除できませんでした: %s", path)
    return removed


def backup_bytes(settings: Settings) -> int:
    """The size of the backup folder (our backups only)."""
    total = 0
    for path in list_backups(settings):
        for f in path.rglob("*"):
            try:
                total += f.stat().st_size if f.is_file() else 0
            except OSError:
                pass
    return total


def max_age_days(settings: Settings) -> int:
    """The longest a backup is kept: backup_keep days, then backup_keep_weekly more weeks."""
    return max(1, settings.backup_keep) + 7 * max(0, settings.backup_keep_weekly)


def keepers(settings: Settings, backups: list[Path], now: datetime) -> set[str]:
    """Names of the backups rotation keeps: the newest backup_keep that are not older than
    backup_keep days, plus the newest backup of each week bucket after that (bucket b: ages
    backup_keep + 7b … +7 days, for b < backup_keep_weekly), and always the newest one."""
    keep = max(1, settings.backup_keep)
    keep_days = timedelta(days=keep)
    keep_names = {p.name for p in backups[-keep:]
                  if not _too_old(p.name, "app-", now - keep_days)}
    newest: dict[int, Path] = {}
    for path in backups:   # oldest first, so the last one of a bucket wins
        at = _stamp(path.name, "app-")
        if at is None or now - at < keep_days:
            continue
        bucket = int((now - at - keep_days).days // 7)
        if bucket < max(0, settings.backup_keep_weekly):
            newest[bucket] = path
    keep_names |= {p.name for p in newest.values()}
    if backups:
        keep_names.add(backups[-1].name)
    return keep_names


def rotate(settings: Settings, keep: int | None = None, now: datetime | None = None) -> list[str]:
    """Remove the backups rotation does not keep (see keepers: backup_keep daily ones, the
    weekly ones after them; the newest always stays), leftover .tmp- folders older than an
    hour and old pre-migrate copies (rotate_copies). `keep` overrides backup_keep (tests).
    Returns the removed names."""
    if keep is not None:
        settings = settings.model_copy(update={"backup_keep": keep})
    now = now or datetime.now()
    removed = []
    backups = list_backups(settings)
    wanted = keepers(settings, backups, now)
    for path in backups:
        if path.name not in wanted:
            shutil.rmtree(path)
            removed.append(path.name)
    root = backup_root(settings)
    try:
        for p in root.iterdir():
            if p.name.startswith(".tmp-app-") and p.is_dir() and not p.is_symlink() \
                    and time.time() - p.stat().st_mtime > 3600:
                shutil.rmtree(p)
                removed.append(p.name)
    except OSError:
        pass
    return removed + rotate_copies(settings, now=now)


def check_room(settings: Settings, usage=None) -> None:
    """BackupError when this backup would leave less than disk_refuse_gb free where
    backup_dir lives (the database with its WAL, plus a margin, is what it needs)."""
    from pfb_app.ops import disk   # noqa: PLC0415
    usage = usage or disk._usage
    db = settings.data_dir / DB_NAME
    need = BACKUP_SLACK_BYTES
    for suffix in ("", "-wal"):
        try:
            need += (settings.data_dir / (DB_NAME + suffix)).stat().st_size
        except OSError:
            pass
    root = backup_root(settings)
    try:
        free = usage(root).free
    except OSError:
        return   # an unreadable file system: the copy itself reports what is wrong
    if free - need < settings.disk_refuse_gb * GB:
        raise BackupError(f"保存場所の空きが足りないため、バックアップを作りませんでした"
                          f"（残り約{free / GB:.1f}GB、必要約{need / GB:.2f}GB、"
                          f"空きの下限 {settings.disk_refuse_gb:g}GB: {root}）")


def run_backup(settings: Settings, now: datetime | None = None, usage=None) -> Path:
    """Take, verify and rotate one backup; returns its folder. BackupError on failure (the
    outcome is recorded in last_result.json either way)."""
    started = time.monotonic()
    now = now or datetime.now()
    root = backup_root(settings)
    db = settings.data_dir / DB_NAME
    name = None
    try:
        if not db.is_file():
            raise BackupError(f"データベースがありません: {db}")
        if root.resolve().is_relative_to(settings.data_dir.resolve()):
            raise BackupError("backup_dir が data_dir の中にあります（外にしてください）")
        root.mkdir(parents=True, exist_ok=True)
        check_room(settings, usage)
        name = _new_name(root, now)
        tmp = root / f".tmp-{name}"
        tmp.mkdir()
        try:
            copy = tmp / DB_NAME
            online_copy(db, copy)
            conn = _ro(copy)
            try:
                check, counts = integrity(conn), table_counts(conn)
            finally:
                conn.close()
            if check != "ok":
                raise BackupError(f"コピーの integrity_check が ok ではありません: {check}")
            by_top: dict[str, list[int]] = {}
            n_files = n_bytes = 0
            with gzip.open(tmp / FILE_LIST, "wt", encoding="utf-8") as out:
                out.write("path\tsize\tmtime\tkind\n")
                for rel, size, mtime, kind in _data_entries(settings.data_dir):
                    out.write(f"{rel}\t{size}\t{mtime}\t{kind}\n")
                    top = rel.split("/", 1)[0] if "/" in rel else "."
                    agg = by_top.setdefault(top, [0, 0])
                    agg[0] += 1
                    agg[1] += size
                    n_files += 1
                    n_bytes += size
            org = _copy_org(settings, tmp)
            manifest = {
                "format": FORMAT, "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(), "app_commit": _app_commit(),
                "source": str(db),
                "db": {"file": DB_NAME, "size": copy.stat().st_size, "sha256": _sha256(copy),
                       "integrity": check, "tables": counts},
                "data": {"dir": str(settings.data_dir), "files": n_files, "bytes": n_bytes,
                         "by_top": by_top, "list": FILE_LIST},
                "duration_sec": round(time.monotonic() - started, 2),
            }
            if org is not None:
                manifest[ORG] = org
            (tmp / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
            checked = verify(tmp)
            if not checked.ok:
                raise BackupError("確認に失敗しました: " + " / ".join(checked.problems))
            final = root / name
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        removed = rotate(settings)
        _write_last(settings, {"at": manifest["created_at"], "ok": True, "name": name,
                               "db_size": manifest["db"]["size"], "files": n_files,
                               "removed": removed, "duration_sec": manifest["duration_sec"]})
        log.info("バックアップを作りました: %s（古いもの %d 件を削除）", final, len(removed))
        from pfb_app.ops import alerts   # noqa: PLC0415
        alerts.clear(settings, "backup-failed")
        return final
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        _write_last(settings, {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "ok": False, "name": name, "error": message})
        log.error("バックアップに失敗しました: %s", message)
        from pfb_app.ops import alerts   # noqa: PLC0415
        alerts.send(settings, "backup-failed", "バックアップに失敗しました",
                    "夜間のバックアップを作れませんでした。サーバーの記録（backup.log）を"
                    "確かめてください。", "error")
        if isinstance(e, BackupError):
            raise
        raise BackupError(message) from e


def find(settings: Settings, name: str) -> Path:
    """A backup by folder name (or path); RestoreError when there is none."""
    path = Path(name)
    if not path.is_absolute():
        path = backup_root(settings) / name
    if not (path / MANIFEST).is_file() or not (path / DB_NAME).is_file():
        raise RestoreError(f"バックアップが見つかりません: {name}")
    return path


def _missing(settings: Settings, path: Path, sample: int = 5) -> tuple[int, list[str]]:
    try:
        with gzip.open(path / FILE_LIST, "rt", encoding="utf-8") as f:
            next(f, None)
            missing = [line.split("\t", 1)[0] for line in f
                       if line.strip() and not os.path.lexists(
                           settings.data_dir / line.split("\t", 1)[0])]
    except (OSError, EOFError):
        return 0, []
    return len(missing), missing[:sample]


def worker_running(settings: Settings, fresh_sec: float = WORKER_FRESH_SEC) -> bool:
    """A worker holds <data_dir>/worker.lock (read-only probe), or its beat is fresh (a
    worker of an older release that takes no lock)."""
    from pfb_app.ops.supervise import lock_held   # noqa: PLC0415
    if lock_held(settings.data_dir / "worker.lock"):
        return True
    beat = files.worker_beat_at(settings)
    return beat is not None and (datetime.now(timezone.utc) - beat).total_seconds() < fresh_sec


def _default_data_dir() -> Path:
    """The live app's data folder (the built-in default), resolved."""
    return Settings().data_dir.resolve()


def _live_ui_checks() -> bool:
    """The live app's UI: the pfb-ui unit is active or the tmux session pfb-ui exists."""
    for cmd in (["systemctl", "--user", "is-active", "--quiet", "pfb-ui.service"],
                ["tmux", "has-session", "-t", "pfb-ui"]):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def ui_running(settings: Settings, live_checks=_live_ui_checks) -> str | None:
    """Why a UI may be using this data (None: none seen): the settings' host:port answers,
    or, for the default (live) data_dir, the pfb-ui unit or tmux session exists."""
    from pfb_app.ops.systemd import _listening   # noqa: PLC0415
    if _listening(settings.host, settings.port):
        return f"画面（{settings.host}:{settings.port}）が動いています"
    if settings.data_dir.resolve() == _default_data_dir() and live_checks():
        return "画面（pfb-ui）が動いています"
    return None


def restore(settings: Settings, name: str, *, force: bool = False,
            now: datetime | None = None) -> RestoreResult:
    """Put a verified backup back as data/app.db (see the module doc). Stop the UI and the
    worker first (systemctl --user stop pfb-ui pfb-worker, or scripts/stop.sh)."""
    path = find(settings, name)
    checked = verify(path)
    if not checked.ok:
        raise RestoreError("このバックアップは確認に失敗しました: " + " / ".join(checked.problems))
    stop = "（systemctl --user stop pfb-worker pfb-ui、または scripts/stop.sh）"
    if not force:
        try:
            running = worker_running(settings)
        except OSError as e:
            raise RestoreError(f"{settings.data_dir / 'worker.lock'} を確認できません"
                               f"（{type(e).__name__}）。ワーカーが止まっているか確かめて、"
                               "--force を付けてください。") from None
        if running:
            raise RestoreError(f"ワーカーが動いています。先に止めてから戻してください{stop}。"
                               f"止めた直後は、最後の応答から約{WORKER_FRESH_SEC // 60}分"
                               "たつまで動いているものとして扱います。少し待ってから、もう一度実行してください。")
        from pfb_app.ops.supervise import db_open_by   # noqa: PLC0415
        pids = db_open_by(settings.data_dir / DB_NAME)
        if pids:
            raise RestoreError(f"データベース（{settings.data_dir / DB_NAME}）をほかのプロセス"
                               f"（pid {', '.join(map(str, pids))}）が開いています。"
                               f"先に止めてから戻してください{stop}。")
    ui = None if force else ui_running(settings)
    if ui:
        raise RestoreError(f"{ui}。先に止めてから戻してください{stop}。")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = settings.data_dir / DB_NAME
    previous = None
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    if db.is_file():
        previous = settings.data_dir / f"app.db.before-restore-{stamp}"
        online_copy(db, previous)
    src = _ro(path / DB_NAME)
    try:
        dst = sqlite3.connect(str(db), timeout=30)
        try:
            src.backup(dst)
            check, counts = integrity(dst), table_counts(dst)
        finally:
            dst.close()
    finally:
        src.close()
    want = read_manifest(path)["db"]["tables"]
    if check != "ok" or counts != want:
        raise RestoreError("戻したデータベースの確認に失敗しました"
                           + (f"（前のものは {previous} にあります）" if previous else ""))
    org_previous = _restore_org(settings, path, stamp)
    n, sample = _missing(settings, path)
    log.info("バックアップ %s を戻しました（前のもの: %s、団体の設定の前のもの: %s、"
             "見つからないファイル: %d）", path.name, previous, org_previous, n)
    return RestoreResult(path.name, previous, n, sample)


@dataclass
class RehearsalResult:
    name: str
    problems: list[str] = field(default_factory=list)
    tables: dict[str, int] = field(default_factory=dict)       # row counts of the restored copy
    live_diff: dict[str, int] = field(default_factory=dict)    # table -> live rows minus backup rows
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems


def last_rehearsal(settings: Settings) -> dict | None:
    """The newest rehearsal's outcome: {"at", "ok", "name", "problems", …} (None: never)."""
    try:
        return json.loads((backup_root(settings) / LAST_REHEARSAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rehearse(settings: Settings, name: str | None = None) -> RehearsalResult:
    """Restore a backup (default: the newest) into a scratch folder and check it the way a real
    restore would be checked: verify (sha256, integrity_check, row counts, organisation files),
    then the same online-backup copy restore() makes, integrity_check and row counts of that
    copy against the manifest, and how far the live database is from it. Never touches the
    live data; the scratch folder is removed. RestoreError when there is no backup."""
    started = time.monotonic()
    if name is None:
        backups = list_backups(settings)
        if not backups:
            raise RestoreError("バックアップがありません")
        path = backups[-1]
    else:
        path = find(settings, name)
    result = RehearsalResult(path.name)
    root = backup_root(settings)
    scratch = root / f".tmp-rehearse-{os.getpid()}-{int(time.time())}"
    try:
        scratch.mkdir(parents=True)
        result.problems += verify(path).problems
        restored = scratch / DB_NAME
        src = _ro(path / DB_NAME)
        try:
            dst = sqlite3.connect(str(restored))
            try:
                src.backup(dst)
                check, counts = integrity(dst), table_counts(dst)
            finally:
                dst.close()
        finally:
            src.close()
        want = read_manifest(path)["db"]["tables"]
        result.tables = counts
        if check != "ok":
            result.problems.append(f"戻したコピーの integrity_check: {check}")
        if counts != want:
            result.problems.append("戻したコピーの行数が manifest と一致しません")
        live = settings.data_dir / DB_NAME
        if live.is_file():
            conn = _ro(live)
            try:
                live_counts = table_counts(conn)
            finally:
                conn.close()
            result.live_diff = {t: live_counts.get(t, 0) - counts.get(t, 0)
                                for t in sorted(set(live_counts) | set(counts))
                                if live_counts.get(t, 0) != counts.get(t, 0)}
    except (OSError, sqlite3.Error, ValueError, KeyError) as e:
        result.problems.append(f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    result.seconds = round(time.monotonic() - started, 2)
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = root / f".{LAST_REHEARSAL}.tmp"
        tmp.write_text(json.dumps(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ok": result.ok,
             "name": result.name, "problems": result.problems, "tables": result.tables,
             "live_diff": result.live_diff, "seconds": result.seconds}, ensure_ascii=False),
            encoding="utf-8")
        os.replace(tmp, root / LAST_REHEARSAL)
    except OSError:
        log.exception("復元の試験の結果を記録できませんでした")
    return result


def pre_migration_copy(db_file: Path, now: datetime | None = None) -> Path | None:
    """Before a schema change of an existing database: an online copy beside it,
    app.db.pre-migrate-<time> (None when it could not be made; the change goes ahead)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = db_file.with_name(f"{db_file.name}.pre-migrate-{stamp}")
    if dest.exists():
        return dest
    try:
        online_copy(db_file, dest)
    except (OSError, sqlite3.Error):
        log.exception("変更前のデータベースのコピーを作れませんでした")
        return None
    log.warning("データベースの列を追加する前に、コピーを作りました: %s", dest)
    return dest


def _print_list(settings: Settings) -> None:
    for path in list_backups(settings):
        try:
            m = read_manifest(path)
            print(f"{path.name}\t{m['created_at']}\tdb {m['db']['size']} B\t"
                  f"files {m['data']['files']}")
        except (OSError, ValueError, KeyError):
            print(f"{path.name}\t(manifest unreadable)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pfb_app.ops.backup")
    parser.add_argument("--settings", help="settings YAML (default: PFB_CONFIG or built-ins)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="take, verify and rotate one backup")
    sub.add_parser("list", help="list the complete backups")
    v = sub.add_parser("verify", help="verify one backup (default: the newest)")
    v.add_argument("name", nargs="?")
    t = sub.add_parser("rehearse", help="restore a backup into a scratch folder and check it "
                                        "(default: the newest); the live data is not touched")
    t.add_argument("name", nargs="?")
    r = sub.add_parser("restore", help="put a backup back as data/app.db")
    r.add_argument("name")
    r.add_argument("--force", action="store_true", help="even if the worker seems to run")
    args = parser.parse_args(argv)
    path = args.settings or os.environ.get("PFB_CONFIG")
    settings = load_settings(Path(path) if path else None)
    from pfb_app.ops import logs   # noqa: PLC0415
    logs.setup(settings, "backup")
    try:
        if args.cmd == "run":
            print(run_backup(settings))
        elif args.cmd == "list":
            _print_list(settings)
        elif args.cmd == "verify":
            backups = list_backups(settings)
            target = find(settings, args.name) if args.name else (backups[-1] if backups
                                                                  else None)
            if target is None:
                print("no backups", file=sys.stderr)
                return 1
            result = verify(target)
            print(f"{result.name}: " + ("ok" if result.ok else "; ".join(result.problems)))
            return 0 if result.ok else 1
        elif args.cmd == "rehearse":
            rh = rehearse(settings, args.name)
            print(f"{rh.name}: " + ("ok" if rh.ok else "; ".join(rh.problems)))
            print("rows in the restored copy: " + ", ".join(f"{t}={n}" for t, n in
                                                           rh.tables.items() if n))
            print("live database differs from it in: "
                  + (", ".join(f"{t} {d:+d}" for t, d in rh.live_diff.items()) or "nothing"))
            return 0 if rh.ok else 1
        else:
            res = restore(settings, args.name, force=args.force)
            print(f"restored {res.name}; previous database: {res.previous}; "
                  f"files listed in the backup but missing now: {res.missing_files}")
            for p in res.missing_sample:
                print(f"  missing: {p}")
    except (BackupError, RestoreError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
