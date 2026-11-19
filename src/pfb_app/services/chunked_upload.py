"""Resumable uploads in chunks for big phone videos (phone-first lane, audit #4).

Streamlit's own uploader holds the whole file in the server's memory and starts over when the
connection drops. These uploads are written to disk chunk by chunk instead, and a dropped one
goes on from the last byte the server has (the browser asks for the offset and sends the rest).

Layout (all under data_dir/uploads, beside the uploader's copies):
- `.partial/<id>.part`: the bytes so far; `.partial/<id>.json`: name, size, owner, times.
- a finished upload moves (same file system: a rename) to `uploads/<8 hex>_<safe name><ext>`,
  named as services.intake.save_uploaded_video names a copy, and waits there until the drop zone
  claims it (`claim`), which it does only after the registration succeeded: a failed
  registration never removes a finished upload (`unclaimed_paths` tells the intake's cleanup
  which files to keep), so the coach can press again without sending it again.

Who may upload: the people the screens let add work (services.auth: 管理者, 評価する人). A token signed with a per-install secret (`data_dir/.upload_secret`, 0600) is handed to
each browser session by the page (`issue_token`); it expires after TOKEN_TTL_SEC. The upload
belongs to the token that created it (its owner key): another token cannot see, extend, cancel
or claim it. The signature also makes a token survive an app restart, so an upload can go on
after one. Ids are 32 lowercase hex characters and never become paths otherwise.

Limits: the file types the intake accepts (videos; documents up to DOC_MAX_MB), the size
`settings.max_upload_mb`, CHUNK_MAX per request, at most MAX_OPEN_PER_OWNER unfinished uploads
per token and MAX_OPEN_TOTAL in all. Disk: an upload starts only when the bytes still to come
of every open upload plus this one, plus ADMIT_MARGIN_BYTES, leave at least the refuse level
free (`ensure_room`, the ops lane's level), so the registration after it still has room; each
chunk checks the level again. The bytes still to come of open uploads (`reserved_bytes`) also
count for every other disk check (ops.disk.ensure_room: the worker, the intake, proxies). Creates are serialised (one lock), so
concurrent ones count each other. Ownership is checked before any per-upload lock file is
made. An unfinished upload without progress for STALE_SEC no longer counts toward the caps or
the reservation. `sweep` removes such stale uploads, finished files nobody claimed for
KEEP_SEC (the worker runs it hourly; a file some work already uses only loses its upload
record), `.part` files without an upload (older than STALE_SEC) and lock files without an
upload. A finished file is named by its upload id, and `done` is written before the rename,
so a crash in between leaves nothing the sweep cannot find.
"""
import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from pfb_app.config import Settings

PARTIAL = ".partial"
SECRET_FILE = ".upload_secret"
TOKEN_TTL_SEC = 24 * 3600
KEEP_SEC = 48 * 3600                  # a finished file nobody claimed
STALE_SEC = 3 * 3600                  # an unfinished upload without progress: given up
ARRIVING_SEC = 120                    # video_arriving: bytes received this recently
CHUNK_MAX = 16 * 1024 * 1024          # bytes per request (the browser sends 8 MiB)
GB = 1024 ** 3
# the free space kept after every open upload is complete: ops' disk_refuse_gb
# (`ensure_room` is pfb_app.ops.disk.ensure_room); this is only that setting's default
REFUSE_GB = 15.0
# kept free beyond the refuse level when an upload is admitted: the registration after it
# checks the refuse level again, and other writers (logs, frames) go on meanwhile
ADMIT_MARGIN_BYTES = 2 * GB
MAX_OPEN_PER_OWNER = 8                # unfinished uploads per token
MAX_OPEN_TOTAL = 32                   # unfinished uploads of the whole server
DOC_MAX_MB = 200
_ID = re.compile(r"^[0-9a-f]{32}$")
# the intake's types (services.intake): kept here as plain tuples so this module has no
# import of the pipeline (the upload endpoint loads it in the web server process)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm")
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".doc")


class UploadError(Exception):
    """A refusal with a plain Japanese message (shown in the browser as it is)."""


class BadToken(UploadError):
    pass


class UnknownUpload(UploadError):
    pass


class UploadRefused(UploadError):
    pass


class OffsetMismatch(UploadError):
    def __init__(self, offset: int):
        super().__init__("続きの位置が合いません。続きから送り直します。")
        self.offset = offset


BAD_TOKEN = "この画面を開き直してから、もう一度送ってください。"
UNKNOWN = "この送信は見つかりませんでした。もう一度ファイルを選んでください。"
WRONG_TYPE = "送れない種類のファイルです。動画（MP4・MOVなど）か、PDF・Wordの書類を選んでください。"
EMPTY = "空のファイルは送れません。"
TOO_BIG = "ファイルが大きすぎます（1ファイル{mb} MBまで）。"
TOO_MANY = "送り終わっていないファイルが多すぎます。送り終わるか、取り消してから選んでください。"
PAST_END = "ファイルの大きさを超えて送られました。もう一度ファイルを選んでください。"
CHUNK_TOO_BIG = "一度に送る量が多すぎます。"
NEGATIVE_OFFSET = "送る位置が正しくありません。もう一度ファイルを選んでください。"
NOT_DONE = "まだ送り終わっていません。"


@dataclass
class Upload:
    id: str
    name: str          # the browser's file name (shown only)
    size: int
    offset: int
    complete: bool
    path: Path | None  # the finished file in data/uploads, once complete


# --- tokens -----------------------------------------------------------------------------------

def secret_path(settings: Settings) -> Path:
    return settings.data_dir / SECRET_FILE


def _secret(settings: Settings) -> bytes:
    """The upload key, made once, private and whole (services.auth.install_secret)."""
    from pfb_app.services.auth import install_secret   # noqa: PLC0415
    return install_secret(secret_path(settings))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign(settings: Settings, payload: str) -> str:
    return _b64(hmac.new(_secret(settings), payload.encode(), hashlib.sha256).digest())


def issue_token(settings: Settings, now: float | None = None,
                session_id: int | None = None) -> str:
    """A new token for one browser session: `<issued>.<random>.<signature>`, or with the
    signed-in person's auth session `<issued>.<random>.<session id>.<signature>` (the endpoint
    accepts only those, while that session is valid: docs/accounts.md)."""
    payload = f"{int(now if now is not None else time.time())}.{secrets.token_hex(16)}"
    if session_id is not None:
        payload += f".{int(session_id)}"
    return f"{payload}.{_sign(settings, payload)}"


def _verified(settings: Settings, token: str | None, now: float | None) -> list[str] | None:
    if not isinstance(token, str) or not token.isascii():   # never a crash on odd bytes
        return None
    parts = token.split(".")
    if len(parts) not in (3, 4) or not parts[0].isdigit() \
            or not re.fullmatch(r"[0-9a-f]{32}", parts[1]) \
            or (len(parts) == 4 and not parts[2].isdigit()):
        return None
    payload = ".".join(parts[:-1])
    if not hmac.compare_digest(_sign(settings, payload).encode(), parts[-1].encode()):
        return None
    age = (now if now is not None else time.time()) - int(parts[0])
    if age < -60 or age > TOKEN_TTL_SEC:
        return None
    return parts


def check_token(settings: Settings, token: str | None, now: float | None = None) -> str | None:
    """The token's owner key when it is ours and not expired, else None."""
    parts = _verified(settings, token, now)
    return None if parts is None else hashlib.sha256(parts[1].encode()).hexdigest()[:32]


def token_session(settings: Settings, token: str | None, now: float | None = None) -> int | None:
    """The auth session a valid token was issued to (None: none, or not a valid token)."""
    parts = _verified(settings, token, now)
    return int(parts[2]) if parts is not None and len(parts) == 4 else None


def _owner(settings: Settings, token: str) -> str:
    owner = check_token(settings, token)
    if owner is None:
        raise BadToken(BAD_TOKEN)
    return owner


# --- files ------------------------------------------------------------------------------------

def uploads_dir(settings: Settings) -> Path:
    return settings.data_dir / "uploads"


def _partial_dir(settings: Settings) -> Path:
    d = uploads_dir(settings) / PARTIAL
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(settings: Settings, uid: str) -> Path:
    if not isinstance(uid, str) or not _ID.match(uid):
        raise UnknownUpload(UNKNOWN)
    return _partial_dir(settings) / f"{uid}.json"


def _part_path(settings: Settings, uid: str) -> Path:
    return _partial_dir(settings) / f"{uid}.part"


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def ensure_room(settings: Settings, incoming_bytes: int = 0, usage=None, reserved: int = 0,
                margin_bytes: int = 0) -> None:
    """pfb_app.ops.disk.ensure_room (the disk_refuse_gb level and its plain sentences), as
    UploadRefused. `usage`: shutil.disk_usage for tests; by default `free_bytes`. `reserved`:
    the open uploads' bytes still to come that the caller counts (0: a chunk of its own)."""
    from pfb_app.ops import disk   # noqa: PLC0415 - loaded only when an upload is checked

    def _free(path):
        return usage(path) if usage is not None else SimpleNamespace(free=free_bytes(path))

    try:
        disk.ensure_room(settings, incoming_bytes, usage=lambda path: _free(disk._existing(path)),
                         reserved=reserved, margin_bytes=margin_bytes)
    except disk.DiskLow as e:
        raise UploadRefused(str(e)) from None


def _metas(settings: Settings, make: bool = True):
    """(meta file, meta) of every upload record; none (and no folder made) when `make` is
    False and there is no upload folder yet."""
    d = uploads_dir(settings) / PARTIAL
    if not make and not d.is_dir():
        return
    for meta_file in (_partial_dir(settings) if make else d).glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _ID.match(str(meta.get("id", ""))):
            yield meta_file, meta


def _open_uploads(settings: Settings, now: float | None = None, make: bool = True) -> list[dict]:
    """Every unfinished upload's meta, with its bytes so far (`have`)."""
    out = []
    for _, meta in _metas(settings, make):
        if meta.get("done"):
            continue
        if (time.time() if now is None else now) - float(meta.get("updated", 0)) > STALE_SEC:
            continue          # given up on: the sweep removes it
        try:
            meta["have"] = _part_path(settings, meta["id"]).stat().st_size
        except OSError:
            meta["have"] = 0
        out.append(meta)
    return out


def video_arriving(settings: Settings, now: float | None = None,
                   within_sec: float = ARRIVING_SEC) -> bool:
    """A video upload is still coming in: unfinished, with bytes received in the last
    `within_sec` seconds (the worker warms the shared server up for its content check)."""
    now = time.time() if now is None else now
    return any(m.get("ext") in VIDEO_EXTENSIONS and now - float(m.get("updated", 0)) <= within_sec
               for m in _open_uploads(settings, now, make=False))


def reserved_bytes(settings: Settings, now: float | None = None) -> int:
    """The bytes still to come of every open upload (stale ones no longer count)."""
    return sum(max(0, int(m.get("size", 0)) - m["have"])
               for m in _open_uploads(settings, now, make=False))


def unclaimed_paths(settings: Settings) -> set[Path]:
    """The finished files still waiting for their claim (resolved): a registration that
    failed must leave them, they are the coach's uploads, not copies of its own."""
    out = set()
    for _, meta in _metas(settings, make=False):
        if meta.get("done"):
            try:
                out.add(Path(meta["done"]).resolve())
            except OSError:
                continue
    return out


@contextmanager
def _create_lock(settings: Settings):
    with open(_partial_dir(settings) / ".create.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _safe_name(filename: str) -> tuple[str, str]:
    """(stem, ext) as intake.save_uploaded_video makes them."""
    base = filename.replace("\\", "/").split("/")[-1].strip()
    ext = Path(base).suffix.lower()
    stem = re.sub(r"[^\w.-]+", "_", Path(base).stem, flags=re.UNICODE).strip("._") or "video"
    return stem[:80], ext


def _read(settings: Settings, uid: str) -> dict:
    try:
        return json.loads(_meta_path(settings, uid).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        raise UnknownUpload(UNKNOWN) from None


def _write(settings: Settings, meta: dict) -> None:
    path = _meta_path(settings, meta["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _locked(settings: Settings, uid: str):
    """One request at a time per upload (a resent chunk may race the original)."""
    lock = _partial_dir(settings) / f"{_meta_path(settings, uid).stem}.lock"
    with open(lock, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _mine(settings: Settings, token: str, uid: str) -> dict:
    owner = _owner(settings, token)
    meta = _read(settings, uid)
    if not hmac.compare_digest(meta.get("owner", ""), owner):
        raise UnknownUpload(UNKNOWN)
    return meta


def _upload(settings: Settings, meta: dict) -> Upload:
    done = meta.get("done")
    if done:
        offset = meta["size"]
    else:
        try:
            offset = _part_path(settings, meta["id"]).stat().st_size
        except FileNotFoundError:
            offset = 0
    return Upload(meta["id"], meta["name"], meta["size"], offset, bool(done),
                  Path(done) if done else None)


def limit_bytes(settings: Settings, filename: str) -> int:
    ext = _safe_name(filename)[1]
    mb = DOC_MAX_MB if ext in DOCUMENT_EXTENSIONS else settings.max_upload_mb
    return int(mb) * 1024 * 1024


def create(settings: Settings, token: str, filename: str, size: int) -> Upload:
    owner = _owner(settings, token)
    stem, ext = _safe_name(str(filename or ""))
    if ext not in VIDEO_EXTENSIONS + DOCUMENT_EXTENSIONS:
        raise UploadRefused(WRONG_TYPE)
    size = int(size)
    if size <= 0:
        raise UploadRefused(EMPTY)
    limit = limit_bytes(settings, filename)
    if size > limit:
        raise UploadRefused(TOO_BIG.format(mb=limit // (1024 * 1024)))
    with _create_lock(settings):   # concurrent creates see each other's reservation
        open_ = _open_uploads(settings)
        if len(open_) >= MAX_OPEN_TOTAL \
                or sum(1 for m in open_ if m.get("owner") == owner) >= MAX_OPEN_PER_OWNER:
            raise UploadRefused(TOO_MANY)
        pending = sum(max(0, int(m.get("size", 0)) - m["have"]) for m in open_)
        ensure_room(settings, size, reserved=pending, margin_bytes=ADMIT_MARGIN_BYTES)
        uid = uuid.uuid4().hex
        now = time.time()
        meta = {"id": uid, "owner": owner, "name": str(filename)[:200], "stem": stem,
                "ext": ext, "size": size, "created": now, "updated": now, "done": None}
        _part_path(settings, uid).touch()
        _write(settings, meta)
    return _upload(settings, meta)


def status(settings: Settings, token: str, uid: str) -> Upload:
    return _upload(settings, _mine(settings, token, uid))


def append(settings: Settings, token: str, uid: str, offset: int, data: bytes) -> int:
    """Write `data` at `offset`, which must be the bytes the server has (OffsetMismatch
    otherwise, carrying them). Returns the new offset; the last chunk finishes the file."""
    if len(data) > CHUNK_MAX:
        raise UploadRefused(CHUNK_TOO_BIG)
    if int(offset) < 0:
        raise UploadRefused(NEGATIVE_OFFSET)
    _mine(settings, token, uid)            # before any lock file is made for this id
    ensure_room(settings, len(data))       # the refuse level, again for every chunk
    with _locked(settings, uid):
        meta = _mine(settings, token, uid)
        up = _upload(settings, meta)
        if up.complete or int(offset) != up.offset:
            raise OffsetMismatch(up.offset)
        if up.offset + len(data) > up.size:
            raise UploadRefused(PAST_END)
        part = _part_path(settings, uid)
        with open(part, "ab") as f:
            f.write(data)
        new = up.offset + len(data)
        meta["updated"] = time.time()
        if new == up.size:
            dest = uploads_dir(settings) / f"{uid}_{meta['stem']}{meta['ext']}"
            meta["done"] = str(dest)
            _write(settings, meta)      # first: a crash before the rename stays findable
            os.replace(part, dest)
        _write(settings, meta)
        return new


def cancel(settings: Settings, token: str, uid: str) -> None:
    """Drop an upload (its part, or its finished file not claimed yet)."""
    _mine(settings, token, uid)
    with _locked(settings, uid):
        meta = _mine(settings, token, uid)
        _remove(settings, meta)


def finished_path(settings: Settings, token: str, uid: str) -> Path:
    """A finished file's path in data/uploads, for the registration; it stays this upload's
    (and is never removed by a failed registration) until `claim`."""
    meta = _mine(settings, token, uid)
    if not meta.get("done"):
        raise UploadRefused(NOT_DONE)
    path = Path(meta["done"])
    if not path.is_file():
        raise UnknownUpload(UNKNOWN)
    return path


def claim(settings: Settings, token: str, uid: str) -> Path:
    """After the registration succeeded: the finished file becomes the work's (once); its
    path in data/uploads."""
    _mine(settings, token, uid)
    with _locked(settings, uid):
        meta = _mine(settings, token, uid)
        if not meta.get("done"):
            raise UploadRefused(NOT_DONE)
        path = Path(meta["done"])
        if not path.is_file():
            _meta_path(settings, uid).unlink(missing_ok=True)
            raise UnknownUpload(UNKNOWN)
        _meta_path(settings, uid).unlink(missing_ok=True)
        (_partial_dir(settings) / f"{uid}.lock").unlink(missing_ok=True)
        return path


def _remove(settings: Settings, meta: dict) -> None:
    uid = meta["id"]
    _part_path(settings, uid).unlink(missing_ok=True)
    if meta.get("done"):
        done = Path(meta["done"])
        if done.parent.resolve() == uploads_dir(settings).resolve():
            done.unlink(missing_ok=True)
    _meta_path(settings, uid).unlink(missing_ok=True)
    (_partial_dir(settings) / f"{uid}.lock").unlink(missing_ok=True)


def touch(settings: Settings, uid: str, when: float) -> None:
    """Set an upload's last activity (tests and the sweep)."""
    meta = _read(settings, uid)
    meta["updated"] = when
    _write(settings, meta)


def _used_by_works(settings: Settings) -> set[Path] | None:
    """Every video path a work's group uses (resolved); None when the database cannot be
    read (then no finished file is removed)."""
    from pfb_app.storage import files                    # noqa: PLC0415
    from pfb_app.storage.db import Repo, connect, init_schema   # noqa: PLC0415
    if not files.db_path(settings).exists():
        return set()   # no works yet
    try:
        conn = connect(files.db_path(settings))
        try:
            init_schema(conn)
            repo = Repo(conn, settings)
            out = set()
            for work in repo.list_assignments():
                for sub in repo.list_submissions(work["id"]):
                    for p in (sub["meta"].get("cameras") or {}).values():
                        try:
                            out.add(Path(str(p)).resolve())
                        except OSError:
                            continue
            return out
        finally:
            conn.close()
    except Exception:   # noqa: BLE001 - unknown: keep every finished file
        return None


def sweep(settings: Settings, now: float | None = None) -> int:
    """Remove unfinished uploads without progress for STALE_SEC, finished files nobody claimed
    for KEEP_SEC, and leftovers without an upload (parts, locks). Returns how many uploads.

    A finished file some work already uses (its registration succeeded but the claim was
    lost, e.g. a crash in between) keeps its file: only its upload record goes."""
    now = time.time() if now is None else now
    removed = 0
    used, looked = None, False
    for _, meta in list(_metas(settings)):
        age = now - float(meta.get("updated", 0))
        if age <= (KEEP_SEC if meta.get("done") else STALE_SEC):
            continue
        if meta.get("done"):
            if not looked:
                used, looked = _used_by_works(settings), True
            if used is None:
                continue
            try:
                in_use = Path(meta["done"]).resolve() in used
            except OSError:
                in_use = True
            if in_use:
                meta = {**meta, "done": None}   # drop the record, keep the work's video
        _remove(settings, meta)
        removed += 1
    for part in _partial_dir(settings).glob("*.part"):   # a part whose upload is gone
        try:
            old = now - part.stat().st_mtime > STALE_SEC
        except OSError:
            continue
        if _ID.match(part.stem) and old and not (part.parent / f"{part.stem}.json").exists():
            part.unlink(missing_ok=True)
    for lock in _partial_dir(settings).glob("*.lock"):   # without an upload (older versions)
        if _ID.match(lock.stem) and not (lock.parent / f"{lock.stem}.json").exists():
            lock.unlink(missing_ok=True)
    return removed
