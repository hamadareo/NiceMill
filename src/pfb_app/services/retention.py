"""「動画を自動で消す」: the organisation-wide retention setting, and the worker's hourly pass.

Setting (app_settings key "retention"): {"days": 0 | 30 | 90 | 180, "all": bool}; anything
else reads as off. Counted from when the work is finished: its handout PDF was received
(`pdf_downloaded_at`) and no approval is newer than that receipt (the home screen's 完了,
without the derivation). A work never handed out is never deleted.

- days only: the work's videos (documents: the document copies) and the heavy intermediate
  files go (`clear_media`); approved texts, the saved PDFs (services.kept) and 成長の記録 stay.
- "all" (「すべて消す」): the whole work goes (services.deletion.delete_work).

The worker calls `sweep` (at most every SWEEP_SEC, at most MAX_PER_PASS works). A work with a
queued, running or GPU-waiting step, or one being deleted, is never touched; see
`clear_media` for the order of things and the path checks.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.services import kept
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

KEY = "retention"
DAYS = (30, 90, 180)
SWEEP_SEC = 3600
MAX_PER_PASS = 3   # each saves its PDFs first (WeasyPrint, in the worker)
DONE = "retention_media_done"   # audit action: the files of a cleared work are gone
FAILED = "retention_failed"     # audit action: the PDFs of a due work could not be saved
FAIL_PAUSE = timedelta(hours=24)   # such a work is tried again after this (others go first)
_ACTIVE = ("queued", "running", "waiting_gpu")


class RetentionError(ValueError):
    """A setting that cannot be saved (plain Japanese message)."""


@dataclass(frozen=True)
class Policy:
    days: int = 0          # 0: off
    all: bool = False      # 「すべて消す」

    @property
    def on(self) -> bool:
        return self.days in DAYS


def load(repo: Repo) -> Policy:
    value = repo.get_setting(KEY)
    if not isinstance(value, dict) or value.get("days") not in DAYS:
        return Policy()
    return Policy(int(value["days"]), value.get("all") is True)


def save(repo: Repo, days: int, everything: bool) -> Policy:
    """Store the setting (0: off; 「すべて消す」 only means something when on)."""
    if days not in (0, *DAYS):
        raise RetentionError("日数は一覧から選んでください。")
    policy = Policy(days, bool(everything) and days in DAYS)
    repo.set_setting(KEY, {"days": policy.days, "all": policy.all})
    return policy


def _parse(stamp: str | None) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(stamp))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


# the work's last activity: an approval, a new group, a step queued or a result saved, an
# approval taken back (review.unapprove clears approved_at, its audit entry stays), or a
# 判定支援 label pressed (linecall_labels.labeled_at; final review LOW 8). Every
# stamp is storage.db._now (UTC, microseconds), so they compare as text; '' for none.
_LAST_ACTIVITY = (
    "MAX(COALESCE((SELECT MAX(s.approved_at) FROM subjects s JOIN submissions b"
    "  ON s.submission_id = b.id WHERE b.assignment_id = a.id), ''),"
    " COALESCE((SELECT MAX(b.created_at) FROM submissions b WHERE b.assignment_id = a.id), ''),"
    " COALESCE((SELECT MAX(r.created_at) FROM stage_runs r JOIN submissions b"
    "  ON r.submission_id = b.id WHERE b.assignment_id = a.id), ''),"
    " COALESCE((SELECT MAX(j.created_at) FROM jobs j JOIN submissions b"
    "  ON j.submission_id = b.id WHERE b.assignment_id = a.id), ''),"
    " COALESCE((SELECT MAX(l.ts) FROM audit_log l WHERE l.action = 'unapprove' AND l.target IN"
    "  (SELECT 'subject:' || s.id FROM subjects s JOIN submissions b ON s.submission_id = b.id"
    "   WHERE b.assignment_id = a.id)), ''),"
    " COALESCE((SELECT MAX(c.labeled_at) FROM linecall_labels c JOIN submissions b"
    "  ON c.submission_id = b.id WHERE b.assignment_id = a.id), ''))")
_FINISHED = f"SELECT a.id, a.pdf_downloaded_at, {_LAST_ACTIVITY} FROM assignments a"


def _finished(received: str | None, last_activity: str | None) -> datetime | None:
    """Finished: the handout was received, and nothing happened to the work after that (no
    approval given or taken back, no new group, no step queued, no result saved)."""
    if not received or (last_activity and last_activity > received):
        return None
    return _parse(received)


def finished_at(repo: Repo, aid: int) -> datetime | None:
    """When the work was finished (its handout received after its last activity), or None."""
    row = repo.conn.execute(_FINISHED + " WHERE a.id=?", (aid,)).fetchone()
    return None if row is None else _finished(row[1], row[2])


def _date(d: datetime, now: datetime) -> str:
    local, today = d.astimezone(), now.astimezone()
    year = f"{local.year}年" if local.year != today.year else ""
    return f"{year}{local.month}月{local.day}日"


def media_word(work: dict) -> str:
    return "取り込んだ書類" if work.get("kind") == "document" else "動画"


def _outside_videos(repo: Repo, settings: Settings, aid: int) -> bool:
    """The work uses a video that is not one of the app's uploads (an inbox original, which is
    never deleted)."""
    from pfb_app.services import deletion   # noqa: PLC0415
    for sub in repo.list_submissions(aid):
        for path in ((sub.get("meta") or {}).get("cameras") or {}).values():
            if deletion._own_entry(settings, Path(path)) is None:
                return True
    return False


def card_line(repo: Repo, work: dict, policy: Policy, now: datetime | None = None,
              settings: Settings | None = None) -> str | None:
    """What the card says about the automatic deletion, or None (nothing set, nothing done).
    `settings`: to say that inbox originals stay (they are never deleted)."""
    now = now or datetime.now(timezone.utc)
    removed = _parse(work.get("media_removed_at"))
    word = media_word(work)
    if removed is not None and not (policy.on and policy.all):
        return (f"{word}は{_date(removed, now)}に消しました。文章とPDFは残っています"
                + ("（受け取りフォルダの元の動画は残ります）" if settings is not None
                   and _outside_videos(repo, settings, work["id"]) else ""))
    if not policy.on:
        return None
    if not policy.all and work.get("pdf_downloaded_at") and not _has_approved(repo, work["id"]):
        # handed out once, nobody approved now: nothing would be kept, so the videos stay
        return f"承認した文章がないため、{word}は自動では消しません"
    if work["id"] in _failed_since(repo, now - FAIL_PAUSE):
        return f"{word}を自動で消せませんでした。続くときは、管理者に知らせてください"
    done = finished_at(repo, work["id"])
    if done is None:
        what = "この作業が" if policy.all else f"{word}が"
        return f"PDFを受け取ってから{policy.days}日後に、{what}消えます"
    due = done + timedelta(days=policy.days)
    what = "この作業" if policy.all else word
    if due <= now:
        return f"{what}は、まもなく消えます"
    return f"{what}は{_date(due, now)}に消えます"


def _pending_deletions(repo: Repo) -> set[int]:
    from pfb_app.services import deletion   # noqa: PLC0415 - deletion imports nothing of ours
    return deletion.pending_deletions(repo)


def _active_works(repo: Repo) -> set[int]:
    return {r[0] for r in repo.conn.execute(
        "SELECT DISTINCT b.assignment_id FROM jobs j JOIN submissions b"
        f" ON j.submission_id = b.id WHERE j.status IN ({','.join('?' * len(_ACTIVE))})",
        _ACTIVE)}


def _has_approved(repo: Repo, aid: int) -> bool:
    return repo.conn.execute(
        "SELECT 1 FROM subjects s JOIN submissions b ON s.submission_id = b.id WHERE"
        " b.assignment_id=? AND s.approval_status='approved' LIMIT 1", (aid,)).fetchone() \
        is not None


def _due_all(repo: Repo, policy: Policy, now: datetime) -> list[int]:
    """Works the setting reaches now (active jobs not considered), oldest id first."""
    if not policy.on:
        return []
    out = []
    for aid, received, last, removed in repo.conn.execute(
            _FINISHED.replace("FROM assignments a", ", a.media_removed_at FROM assignments a")
            + " WHERE a.pdf_downloaded_at IS NOT NULL ORDER BY a.id"):
        done = _finished(received, last)
        if done is None or done + timedelta(days=policy.days) > now:
            continue
        if removed and not policy.all:
            continue
        if not policy.all and not _has_approved(repo, aid):
            continue   # nothing to hand out any more: its videos stay (never a stuck pass)
        out.append(aid)
    pending = _pending_deletions(repo)
    paused = _failed_since(repo, now - FAIL_PAUSE)
    return [a for a in out if a not in pending and a not in paused]


def _failed_since(repo: Repo, since: datetime) -> set[int]:
    """Works whose PDFs could not be saved after `since` (they wait FAIL_PAUSE)."""
    stamp = since.isoformat(timespec="microseconds")
    out = set()
    for (target,) in repo.conn.execute(
            "SELECT target FROM audit_log WHERE action=? AND ts > ?", (FAILED, stamp)):
        tail = str(target).split(":", 1)[-1]
        if tail.isascii() and tail.isdigit():
            out.add(int(tail))
    return out


def due(repo: Repo, policy: Policy, now: datetime | None = None) -> list[int]:
    """The works the next pass handles: reached by the setting, no active step, not being
    deleted."""
    now = now or datetime.now(timezone.utc)
    active = _active_works(repo)
    return [a for a in _due_all(repo, policy, now) if a not in active]


def preview_works(repo: Repo, policy: Policy, now: datetime | None = None
                  ) -> list[tuple[int, str]]:
    """[(id, title)] of the works the setting reaches now, as `preview` counts them."""
    ids = _due_all(repo, policy, now or datetime.now(timezone.utc))
    titles = {r[0]: r[1] for r in repo.conn.execute("SELECT id, title FROM assignments")}
    return [(a, titles.get(a, "")) for a in ids]


def preview(repo: Repo, policy: Policy, now: datetime | None = None) -> int:
    """How many works the setting reaches now (a work still analysed counts: it goes once its
    steps end); the settings area shows it before the setting is saved."""
    return len(_due_all(repo, policy, now or datetime.now(timezone.utc)))


# --- the worker's pass ---------------------------------------------------------------------

# folders of heavy intermediate files, wherever they are inside a work's folder
HEAVY_DIRS = frozenset({"frames", "pages", "people", "handout", "tracks", "cache"})
# files that are pictures, videos, documents or arrays, anywhere inside a work's folder
MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".aac",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    ".pdf", ".docx", ".doc", ".npy", ".npz", ".pt"})
REMOVED = "retention_media"   # audit action: the work's videos are to go (the flag is set)


@dataclass(frozen=True)
class Cleared:
    assignment_id: int
    files: int       # files removed inside the work's folder
    bytes: int
    uploads: int     # the work's own uploads removed


def _within(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def _size(path: Path) -> int:
    try:
        return path.lstat().st_size
    except OSError:
        return 0


def _tree(path: Path) -> tuple[int, int]:
    """(files, bytes) under a real folder, links counted as themselves, never followed."""
    n = size = 0
    for dirpath, _dirs, names in os.walk(path, followlinks=False):
        for name in names:
            n += 1
            size += _size(Path(dirpath) / name)
    return n, size


def _sweep_folder(settings: Settings, aid: int) -> tuple[int, int]:
    """Remove HEAVY_DIRS and MEDIA_SUFFIXES files inside data/assignments/<aid>/ (its `kept`
    folder excepted). The work folder must be a real folder inside data_dir/assignments; no
    link is followed (a link loses only itself, and only when its name says it is heavy)."""
    root = files.assignment_dir(settings, aid, create=False)
    try:
        data_root = settings.data_dir.resolve()
        parent = (settings.data_dir / "assignments").resolve()
        if not _within(parent, data_root) or root.is_symlink() or not root.is_dir():
            if root.is_symlink():
                log.warning("作業フォルダがリンクのため、中の動画は消しません: %s", root)
            return 0, 0
        real = root.resolve()
        if real.parent != parent:
            return 0, 0
    except OSError:
        return 0, 0
    keep = real / kept.FOLDER
    removed = size = 0
    for dirpath, dirs, names in os.walk(real, followlinks=False):
        here = Path(dirpath)
        if here == keep or not _within(here.resolve(), parent) and here != real:
            dirs[:] = []
            continue
        if here == real:
            dirs[:] = [d for d in dirs if d != kept.FOLDER and not d.startswith(f"{kept.FOLDER}.")]
        for name in list(dirs):
            path = here / name
            if name not in HEAVY_DIRS:
                continue
            dirs.remove(name)
            try:
                if path.is_symlink():
                    path.unlink()
                    removed, size = removed + 1, size + _size(path)
                elif path.is_dir() and _within(path.resolve(), real):
                    n, b = _tree(path)
                    shutil.rmtree(path)
                    removed, size = removed + n, size + b
            except OSError:
                log.exception("途中のファイルを消せませんでした: %s", path)
        for name in names:
            path = here / name
            if path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            try:
                b = _size(path)
                path.unlink()   # a link loses only itself
                removed, size = removed + 1, size + b
            except OSError:
                log.exception("途中のファイルを消せませんでした: %s", path)
    return removed, size


def _other_metas(repo: Repo, aid: int) -> list[dict]:
    """The groups of the other works whose videos are still in use (not cleared)."""
    import json   # noqa: PLC0415
    return [json.loads(r[0] or "{}") for r in repo.conn.execute(
        "SELECT b.meta FROM submissions b JOIN assignments a ON a.id = b.assignment_id"
        " WHERE b.assignment_id <> ? AND a.media_removed_at IS NULL", (aid,))]


def _finish(repo: Repo, settings: Settings, aid: int) -> Cleared:
    """Remove the files of a work whose flag is set: its own uploads no other work in use
    needs, their proxies, its 解説動画 (services.explainer: jobs cancelled first), the heavy
    files of its folder; then audit DONE."""
    from pfb_app.services import deletion, explainer, proxy   # noqa: PLC0415
    explainer.drop_work(repo, settings, aid)   # 解説動画: its jobs stop, its videos go first
    subs = repo.list_submissions(aid)
    others = _other_metas(repo, aid)
    ours = deletion._own_uploads(settings, (s["meta"] for s in subs))
    ours -= deletion._own_uploads(settings, others)
    videos = deletion._videos(subs) - deletion._videos({"meta": m} for m in others)
    proxy.remove_for(settings, videos)
    uploads = 0
    for entry in sorted(ours):
        try:
            if entry.is_symlink() or entry.is_file():
                entry.unlink()   # a link loses only itself; a hardlink only this name
                uploads += 1
        except OSError:
            log.exception("取り込んだ動画を消せませんでした: %s", entry)
    n, size = _sweep_folder(settings, aid)
    repo.audit(DONE, f"assignment:{aid}", {"files": n, "bytes": size, "uploads": uploads})
    log.info("動画と途中のファイルを消しました assignment=%s files=%d bytes=%d uploads=%d",
             aid, n, size, uploads)
    return Cleared(aid, n, size, uploads)


def _busy(repo: Repo, aid: int) -> bool:
    return aid in _active_works(repo) or aid in _pending_deletions(repo)


def clear_media(repo: Repo, settings: Settings, aid: int, now: datetime | None = None
                ) -> Cleared | None:
    """Save the work's PDFs, then (in one write transaction that checks again that no step
    is queued, running or waiting, and the work is not being deleted) set media_removed_at,
    so nothing is queued for it any more; then remove the files (`_finish`). None when the
    work was left alone (busy, already cleared, or its PDFs could not be saved)."""
    now = now or datetime.now(timezone.utc)
    work = repo.get_assignment(aid)
    if work.get("media_removed_at") or _busy(repo, aid):
        return None
    try:
        kept.save(repo, aid)
        for (sid,) in repo.conn.execute(   # erased while the PDFs were made: never kept
                "SELECT s.id FROM subjects s JOIN submissions b ON s.submission_id = b.id"
                " WHERE b.assignment_id=? AND s.erased_at IS NOT NULL", (aid,)).fetchall():
            kept.drop_person(settings, aid, sid)
    except Exception:   # noqa: BLE001 - never remove what could not be kept
        log.exception("PDFを保存できなかったため、動画は消しません assignment=%s", aid)
        repo.audit(FAILED, f"assignment:{aid}", {})
        return None
    with repo._tx():
        # an approval changed while the PDFs were made: they are no longer what was handed
        # out, so the videos stay for the next pass
        if _busy(repo, aid) or kept.current(repo, aid) is None \
                or finished_at(repo, aid) is None:   # re-opened meanwhile (review R28)
            return None
        changed = repo.conn.execute(
            "UPDATE assignments SET media_removed_at=? WHERE id=? AND media_removed_at IS NULL",
            (now.isoformat(timespec="microseconds"), aid)).rowcount
        if not changed:
            return None
        repo.audit(REMOVED, f"assignment:{aid}", {})
    return _finish(repo, settings, aid)


def _unfinished(repo: Repo) -> list[int]:
    """Cleared works whose files were not all removed yet (a stop half way)."""
    return [r[0] for r in repo.conn.execute(
        "SELECT id FROM assignments a WHERE media_removed_at IS NOT NULL AND NOT EXISTS"
        " (SELECT 1 FROM audit_log l WHERE l.action=? AND l.target='assignment:' || a.id)"
        " ORDER BY id", (DONE,))]


def sweep(repo: Repo, settings: Settings, now: datetime | None = None) -> dict[str, list[int]]:
    """The worker's pass: finish clearings that stopped half way, then handle at most
    MAX_PER_PASS works the setting reaches. Returns {"cleared": ids, "deleted": ids}."""
    from pfb_app.services import deletion   # noqa: PLC0415
    now = now or datetime.now(timezone.utc)
    out: dict[str, list[int]] = {"cleared": [], "deleted": []}
    for aid in _unfinished(repo):
        if not _busy(repo, aid):
            _finish(repo, settings, aid)
    policy = load(repo)
    tries = 0
    for aid in due(repo, policy, now):
        if len(out["cleared"]) + len(out["deleted"]) >= MAX_PER_PASS \
                or tries >= 2 * MAX_PER_PASS:   # a work that fails never holds the others up
            break
        tries += 1
        if policy.all:
            try:
                deletion.delete_work(repo, settings, aid)
            except (KeyError, deletion.WorkActive, deletion.StillRunning):
                continue
            repo.audit("retention_delete", f"assignment:{aid}", {"days": policy.days})
            log.info("設定どおり作業を消しました assignment=%s days=%d", aid, policy.days)
            out["deleted"].append(aid)
        elif clear_media(repo, settings, aid, now) is not None:
            out["cleared"].append(aid)
    return out
