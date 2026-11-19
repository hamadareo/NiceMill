"""Deleting a work (an assignment) from the home screen.

What goes: the work's rows (groups, people, results, steps of the analysis; the PDF download
record is a column of the work itself), its folder data/assignments/<aid>/ (frames, pages,
document copies, the tracking engine's outputs) and the browser uploads in data/uploads/ that
only this work uses.

What never goes: anything outside data_dir, or inside the inbox folder (wherever that is). A
video picked from the inbox stays; a symlink or hardlink in the app's folders loses only the
app's own name for the file (the link), never the file it points at; a work folder that is
itself a symlink loses only the link. Only direct children of data/uploads are uploads.

The model's answers (call_log) go too: every row a group's job wrote or reused is recorded in
call_log_owner (storage.db, Repo.call_owner); deleting the work removes those records with
its groups, then every call_log row of theirs that no remaining group uses (an identical
request of another work keeps the shared row). Rows written before call_log_owner existed
have no owner and cannot be traced to a work (call_log is keyed by request_hash, a sha256
over the model, the prompt text and the image contents); they stay until
`purge_untraced_calls` removes them (an operator step: it only costs a fresh model call when
an old step is redone).

Stopping the analysis (「解析を止めて削除」): queued and GPU-waiting steps are cancelled. A step
that is already running is never stopped from here (no process is signalled from the UI):
the work's groups get meta["delete_requested"] = true (and an audit entry), so nothing new is
queued for them (pipeline.runner.enqueue_once / request_names), and the worker finishes the
deletion as soon as no step of the work runs any more (finish_requested_deletions, after
each job and on every pass).

The check, the cancellation, the mark and the row deletion run in one write transaction, so
the worker cannot claim one of the work's steps in between (claim_next_job takes the same
write lock). The work's own uploads are removed in that transaction too, after its last
write: whether another work still uses a file is decided and acted on under one lock, so
no group registered meanwhile loses its video. The work's folder (no other work can use
it) and the proxies (a video another work registers later is simply proxied again) go
after the commit.
"""
import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

RUNNING = "いまの処理が終わると削除されます。"
REQUESTED = "delete_requested"   # submissions meta key: the work is to be deleted
ACTIVE = "この作業はまだ解析中です。「解析を止めて削除」で削除できます。"

_PENDING = (JobStatus.queued.value, JobStatus.waiting_gpu.value)
_RUNNING = JobStatus.running.value


class WorkActive(Exception):
    """The work still has analysis waiting or running and the caller did not ask to stop it."""

    def __init__(self, message: str = ACTIVE):
        super().__init__(message)


class StillRunning(Exception):
    """A step of the work is running: it is left to finish, the work is marked and the worker
    deletes it once nothing of it runs. `cancelled`: how many waiting steps were cancelled."""

    def __init__(self, cancelled: int, message: str = RUNNING):
        super().__init__(message)
        self.cancelled = cancelled


@dataclass(frozen=True)
class DeletePlan:
    """What deleting the work would remove, for the confirm area."""
    assignment_id: int
    title: str
    groups: int
    pending: int    # analysis steps queued or waiting for the GPU
    running: int    # analysis steps running now

    @property
    def active(self) -> bool:
        return bool(self.pending or self.running)


@dataclass(frozen=True)
class DeleteResult:
    title: str
    jobs_cancelled: int
    uploads_removed: int


def _counts(repo: Repo, aid: int) -> tuple[int, int, int, int, int]:
    """(groups, people, results, pending, running) of the work, in one statement."""
    row = repo.conn.execute(
        "SELECT (SELECT COUNT(*) FROM submissions WHERE assignment_id=:a),"
        " (SELECT COUNT(*) FROM subjects s JOIN submissions u ON s.submission_id=u.id"
        "  WHERE u.assignment_id=:a),"
        " (SELECT COUNT(*) FROM stage_runs r JOIN submissions u ON r.submission_id=u.id"
        "  WHERE u.assignment_id=:a),"
        " (SELECT COUNT(*) FROM jobs j JOIN submissions u ON j.submission_id=u.id"
        "  WHERE u.assignment_id=:a AND j.status IN (:q, :w)),"
        " (SELECT COUNT(*) FROM jobs j JOIN submissions u ON j.submission_id=u.id"
        "  WHERE u.assignment_id=:a AND j.status=:r)",
        {"a": aid, "q": _PENDING[0], "w": _PENDING[1], "r": _RUNNING}).fetchone()
    return tuple(int(v) for v in row)


def deletion_plan(repo: Repo, aid: int) -> DeletePlan:
    """KeyError for an unknown (e.g. already deleted) work."""
    title = repo.get_assignment(aid)["title"]
    groups, _, _, pending, running = _counts(repo, aid)
    return DeletePlan(aid, title, groups, pending, running)


def _within(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def _own_entry(settings: Settings, path: Path) -> Path | None:
    """The directory entry `path` names, if it is a direct child of data_dir/uploads (that
    folder inside data_dir) and not inside the inbox folder; else None. The last component is
    not followed, so for a symlink this is the link itself."""
    try:
        data_root = settings.data_dir.resolve()
        uploads = (settings.data_dir / "uploads").resolve()
        inbox = settings.inbox_dir.resolve()
        entry = Path(path).parent.resolve() / Path(path).name
    except (OSError, RuntimeError):
        return None
    if Path(path).name in ("", ".", "..") or not _within(uploads, data_root):
        return None
    if entry.parent != uploads or entry == inbox or entry.is_relative_to(inbox):
        return None
    return entry


def _referenced_paths(meta: dict) -> list[str]:
    paths = list((meta.get("cameras") or {}).values())
    if meta.get("source_path"):
        paths.append(meta["source_path"])
    return [p for p in paths if isinstance(p, str) and p]


def _own_uploads(settings: Settings, metas) -> set[Path]:
    """The entries in data_dir/uploads that these submissions' meta point at."""
    return {e for meta in metas for p in _referenced_paths(meta)
            if (e := _own_entry(settings, Path(p))) is not None}


def _remove_work_folder(settings: Settings, aid: int) -> None:
    """data/assignments/<aid>: a symlink loses only the link; a real folder is removed with
    everything in it (rmtree never follows the symlinks inside, it removes the links)."""
    adir = files.assignment_dir(settings, aid, create=False)
    try:
        assignments_root = (settings.data_dir / "assignments").resolve()
        if not _within(assignments_root, settings.data_dir.resolve()):
            log.warning("作業フォルダの親がデータフォルダの外にあるため、残します: %s", adir)
            return
        if adir.is_symlink():
            adir.unlink()
        elif adir.is_dir() and _within(adir.resolve(), assignments_root):
            shutil.rmtree(adir)
    except OSError:
        log.exception("作業フォルダを削除できませんでした: %s", adir)


_CHUNK = 500   # bound parameters per statement


def _owned_calls(repo: Repo, aid: int) -> list[str]:
    """The call_log keys the work's groups made or reused."""
    return [r[0] for r in repo.conn.execute(
        "SELECT DISTINCT o.request_hash FROM call_log_owner o JOIN submissions s"
        " ON s.id = o.submission_id WHERE s.assignment_id=?", (aid,))]


def _drop_unowned_calls(repo: Repo, hashes: list[str]) -> int:
    """Delete these call_log rows unless a remaining group still uses them."""
    removed = 0
    for i in range(0, len(hashes), _CHUNK):
        part = hashes[i:i + _CHUNK]
        removed += repo.conn.execute(
            f"DELETE FROM call_log WHERE request_hash IN ({','.join('?' * len(part))})"
            " AND NOT EXISTS (SELECT 1 FROM call_log_owner o"
            " WHERE o.request_hash = call_log.request_hash)", part).rowcount
    return removed


def untraced_calls(repo: Repo) -> int:
    """call_log rows with no owner (written before call_log_owner existed, or outside a job)."""
    return repo.conn.execute(
        "SELECT COUNT(*) FROM call_log c WHERE NOT EXISTS"
        " (SELECT 1 FROM call_log_owner o WHERE o.request_hash = c.request_hash)").fetchone()[0]


def purge_untraced_calls(repo: Repo) -> int:
    """Delete every call_log row with no owner (an operator step; audited)."""
    with repo._tx():
        n = repo.conn.execute(
            "DELETE FROM call_log WHERE NOT EXISTS (SELECT 1 FROM call_log_owner o"
            " WHERE o.request_hash = call_log.request_hash)").rowcount
        repo.audit("purge_untraced_calls", None, {"removed": n})
    return n


def delete_work(repo: Repo, settings: Settings, aid: int, *,
                stop_analysis: bool = False) -> DeleteResult:
    """Delete the work, its folder and its own uploads.

    KeyError: unknown work. WorkActive: analysis is waiting or running and `stop_analysis` is
    False (nothing changed). StillRunning: `stop_analysis` cancelled the waiting steps but a
    step is running (the work is kept; try again once it ends)."""
    blocked = None
    removed = 0
    with repo._tx():
        title = repo.get_assignment(aid)["title"]
        groups, people, results, pending, running = _counts(repo, aid)
        cancelled = 0
        if (pending or running) and not stop_analysis:
            blocked = WorkActive()
        else:
            if pending:
                cancelled = repo.conn.execute(
                    "DELETE FROM jobs WHERE status IN (?, ?) AND submission_id IN"
                    " (SELECT id FROM submissions WHERE assignment_id=?)",
                    (*_PENDING, aid)).rowcount
            if running:
                repo.conn.execute(
                    f"UPDATE submissions SET meta = json_set(meta, '$.{REQUESTED}', json('true'))"
                    " WHERE assignment_id=?", (aid,))
                repo.audit("delete_requested", f"assignment:{aid}",
                           {"jobs_cancelled": cancelled, "running": running})
                blocked = StillRunning(cancelled)
            else:
                ours = _own_uploads(settings, (s["meta"] for s in repo.list_submissions(aid)))
                hashes = _owned_calls(repo, aid)
                videos = _videos(repo.list_submissions(aid))   # their 1080p proxies go too
                repo.conn.execute("DELETE FROM assignments WHERE id=?", (aid,))
                repo.prune_growth_people()   # 成長の記録 left without any work
                calls = _drop_unowned_calls(repo, hashes)   # owners went with the groups
                videos -= _videos({"meta": json.loads(r[0])} for r in repo.conn.execute(
                    "SELECT meta FROM submissions"))
                if ours:   # an upload another work still uses is kept
                    ours -= _own_uploads(settings, (json.loads(r[0]) for r in repo.conn.execute(
                        "SELECT meta FROM submissions")))
                # with the row deletion: the record sweep_orphan_dirs needs to finish a
                # deletion that stopped before its folder was removed
                repo.audit("delete_work", f"assignment:{aid}",
                           {"groups": groups, "people": people, "results": results,
                            "jobs_cancelled": cancelled, "uploads_removed": len(ours),
                            "calls_removed": calls})
                # the uploads go under the same write lock that decided no other work uses
                # them (the check and the unlink are one step: nothing can register a video
                # that uses the file in between); last, so that an error above leaves them
                for entry in sorted(ours):
                    try:
                        if entry.is_symlink() or entry.is_file():
                            entry.unlink()   # a link loses only itself; a hardlink only this name
                            removed += 1
                    except OSError:
                        log.exception("取り込んだファイルを削除できませんでした: %s", entry)
    if blocked is not None:     # raised after the block, so the cancellation is committed
        raise blocked
    _remove_work_folder(settings, aid)
    from pfb_app.services import proxy   # noqa: PLC0415 (one self-contained hunk)
    proxy.remove_for(settings, videos)   # a video another work still uses keeps its proxy
    proxy.maybe_sweep(settings)
    return DeleteResult(title, cancelled, removed)


def _videos(submissions) -> set[str]:
    """The camera videos the submissions use (their 1080p proxies: services.proxy)."""
    return {str(p) for s in submissions
            for p in ((s.get("meta") or {}).get("cameras") or {}).values()}


def is_requested(meta: dict | None) -> bool:
    """The group's work is to be deleted once its running step ends: queue nothing new."""
    return bool((meta or {}).get(REQUESTED))


_REQUESTED_WORKS = (
    f"SELECT DISTINCT assignment_id FROM submissions WHERE json_extract(meta, '$.{REQUESTED}')"
    " OR 'assignment:' || assignment_id IN"
    " (SELECT target FROM audit_log WHERE action='delete_requested')")


def pending_deletions(repo: Repo) -> set[int]:
    """Works waiting for their running step to end before they are deleted (one statement).
    The audit entry also counts, so a group meta rewritten meanwhile cannot lose the request."""
    return {r[0] for r in repo.conn.execute(_REQUESTED_WORKS)}


def is_pending(repo: Repo, aid: int) -> bool:
    """This work waits for its running step to end before it is deleted (one statement)."""
    return repo.conn.execute(f"SELECT 1 FROM ({_REQUESTED_WORKS}) w WHERE w.assignment_id=?"
                             " LIMIT 1", (aid,)).fetchone() is not None


def finish_requested_deletions(repo: Repo, settings: Settings) -> list[int]:
    """Delete every requested work none of whose steps is running (the worker calls this after
    each job and on every pass). Returns the deleted work ids."""
    rows = repo.conn.execute(
        f"SELECT assignment_id FROM ({_REQUESTED_WORKS}) w WHERE NOT EXISTS"
        " (SELECT 1 FROM jobs j JOIN submissions u ON j.submission_id = u.id"
        "  WHERE u.assignment_id = w.assignment_id AND j.status=?)", (_RUNNING,)).fetchall()
    done = []
    for (aid,) in rows:
        try:
            delete_work(repo, settings, aid, stop_analysis=True)
        except (KeyError, StillRunning):   # gone meanwhile / a step started meanwhile
            continue
        except Exception:   # noqa: BLE001 - a failed deletion must never stop the worker
            log.exception("削除を頼まれた作業を削除できませんでした assignment=%s", aid)
            continue
        log.info("解析の終了を待っていた作業を削除しました assignment=%s", aid)
        done.append(aid)
    return done


def _deleted_ids(repo: Repo) -> set[int]:
    """Work ids the audit log says were deleted, or asked to be."""
    out = set()
    for (target,) in repo.conn.execute(
            "SELECT target FROM audit_log WHERE action IN ('delete_work', 'delete_requested')"
            " AND target LIKE 'assignment:%'"):
        tail = target.split(":", 1)[1]
        if tail.isascii() and tail.isdigit():
            out.add(int(tail))
    return out


def sweep_orphan_dirs(repo: Repo, settings: Settings) -> int:
    """At worker start: remove data/assignments/<n> folders left by a deletion that stopped
    half way (its row is gone, its folder is not), with the same checks as a deletion.
    Returns how many.

    Only folders this database is known to have deleted go: the name is ASCII digits, the id
    is at most the highest work id this database ever gave (sqlite_sequence) and the audit
    log records its deletion (delete_work or delete_requested). Nothing is swept when
    data/assignments is a symlink, when the database has no sequence row for works or no
    work at all (a fresh or emptied database next to old folders), so a restored older
    backup or a new database never removes folders it does not know."""
    root = settings.data_dir / "assignments"
    if root.is_symlink():
        log.warning("作業フォルダの親がリンクのため、片付けません: %s", root)
        return 0
    try:
        names = [p.name for p in root.iterdir() if p.name.isascii() and p.name.isdigit()]
    except OSError:
        return 0
    if not names:
        return 0
    try:
        row = repo.conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name='assignments'").fetchone()
    except sqlite3.OperationalError:   # no AUTOINCREMENT row was ever written
        return 0
    known = {r[0] for r in repo.conn.execute("SELECT id FROM assignments")}
    if row is None or not known:
        return 0
    highest, deleted = int(row[0]), _deleted_ids(repo)
    removed = 0
    for name in sorted(names, key=int):
        aid = int(name)
        if aid in known or aid > highest or aid not in deleted:
            continue
        _remove_work_folder(settings, aid)
        if not (root / name).exists() and not (root / name).is_symlink():
            removed += 1
            log.info("行のない作業フォルダを削除しました: %s", root / name)
    return removed
