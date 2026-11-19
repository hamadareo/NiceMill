"""Accounts (audit rank 7): local users with argon2 password hashes, three roles, signed
expiring sessions, rate-limited sign-in, the first administrator's one-time setup code, and
one-time invitation and password-reset codes. docs/accounts.md has the rulings.

- Roles: `admin` 管理者 (everything), `evaluator` 評価する人 (the wizard end to end),
  `viewer` 見るだけの人 (looks only). `can(role, what)` with what = "edit" | "admin".
- One organisation per install: every row is org 1 (`ORG`).
- A session is a row in auth_sessions plus a cookie `v1.<id>.<expires>.<random>.<sig>`: the
  signature is HMAC-SHA256 with the install's key (data_dir/.auth_secret, 0600), the row keeps
  SHA-256 of <random> only. Sign-out, a password change or reset, a role change and switching a
  person off revoke rows, so open tabs lose their access on their next run.
- Failed sign-ins and code tries are rows in login_failures; too many in WINDOW_SEC for a key
  and that key waits (TooManyTries) — even the right password.
- No password, code, cookie or key is ever logged or written to audit_log.
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from pfb_app.config import Settings
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

ROLES = ("admin", "evaluator", "viewer")
ROLE_LABELS = {"admin": "管理者", "evaluator": "評価する人", "viewer": "見るだけの人"}
_CAPS = {"edit": {"admin", "evaluator"}, "admin": {"admin"}}
ORG = 1

SECRET_FILE = ".auth_secret"
SETUP_FILE = ".setup_code"
COOKIE = "pfb_auth"      # + "_<port>": other apps on the same address keep their own
COOKIE_VERSION = "v1"

PASSWORD_MIN, PASSWORD_MAX = 10, 128
NAME_MAX = 40
_LOGIN = re.compile(r"[a-z0-9._@-]{3,32}")

WINDOW_SEC = 15 * 60
LOGIN_LIMIT = 5          # tries per login name (and current-password tries per person)
IP_LIMIT = 20            # tries per source (address, Tailscale login or browser) in WINDOW_SEC
CODE_LIMIT = 20          # code tries per source
CODE_PREFIX_LIMIT = 10   # code tries per code prefix (its first 4 characters)
SETUP_LIMIT = 10         # setup-code tries per source
SETUP_ALL_LIMIT = 100    # setup-code tries of everyone (the CLI is never limited)
SETUP_CODE_HOURS = 24    # a setup code older than this is refused; the CLI makes a new one
KEEP_FAILURES_SEC = 24 * 3600
# bumped by every change that ends sessions in this process: ui.access drops its short cache
GENERATION = [0]

INVITE_DAYS = 7
RESET_HOURS = 24
_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"   # no 0/O, 1/I
CODE_LEN = 12

HASHER = PasswordHasher()   # argon2id, RFC 9106 low-memory defaults

LOGIN_REFUSED = "ログイン名かパスワードがちがいます。"
TOO_MANY = "何度もまちがえたため、しばらく入れません。15分ほどたってから、もう一度ためしてください。"
CODE_REFUSED = "このコードは使えません。"
LAST_ADMIN = "管理者が一人もいなくなるため、変えられません。先にほかの人を管理者にしてください。"


class AuthError(Exception):
    """Plain Japanese for the screen."""


class BadPassword(AuthError):
    pass


class BadLogin(AuthError):
    pass


class BadName(AuthError):
    pass


class LoginRefused(AuthError):
    pass


class TooManyTries(AuthError):
    pass


class CodeRefused(AuthError):
    pass


class LastAdmin(AuthError):
    pass


def can(role: str | None, what: str) -> bool:
    return role in _CAPS.get(what, ())


def cookie_name(port: int | None) -> str:
    """The cookie of this app on this port (other apps may share the address)."""
    return f"{COOKIE}_{int(port or 0)}"


def _same(a: str | None, b: str | None) -> bool:
    """Constant-time equality of two ASCII strings; anything else is simply unequal."""
    if not isinstance(a, str) or not isinstance(b, str) or not a.isascii() or not b.isascii():
        return False
    return hmac.compare_digest(a.encode(), b.encode())


def _digits(text: str) -> bool:
    return text.isascii() and text.isdigit()


@dataclass(frozen=True)
class User:
    id: int
    login: str
    display_name: str
    role: str
    active: bool = True
    org_id: int = ORG
    last_login_at: str | None = None

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    @property
    def label(self) -> str:
        return f"{self.display_name}（{self.role_label}）"


@dataclass(frozen=True)
class Session:
    id: int
    user: User
    expires: float


@dataclass(frozen=True)
class Code:
    """An open invitation (user_id None) or password reset (of user_id)."""
    id: int
    role: str
    display_name: str
    user_id: int | None


# --- time -------------------------------------------------------------------------------------

def _now(now: float | None) -> float:
    return time.time() if now is None else now


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="microseconds")


def _epoch(iso: str | None) -> float | None:
    return None if not iso else datetime.fromisoformat(iso).timestamp()


# --- secrets on disk --------------------------------------------------------------------------

def _create_private(path: Path, data: bytes) -> None:
    """Write `data` to `path` once, private (0600) and whole: written to a file of our own and
    then linked into place, so a concurrent creator never reads a half-written file; the first
    link wins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        tmp.unlink(missing_ok=True)


def install_secret(path: Path) -> bytes:
    """The install's key at `path` (32 random bytes), made once."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    _create_private(path, secrets.token_bytes(32))
    return path.read_bytes()


def _key(settings: Settings) -> bytes:
    return install_secret(settings.data_dir / SECRET_FILE)


# --- names and passwords ----------------------------------------------------------------------

def normalize_login(login: str) -> str:
    value = unicodedata.normalize("NFKC", login or "").strip().lower()
    if not _LOGIN.fullmatch(value):
        raise BadLogin("ログイン名は、半角の英字・数字・「. _ - @」で3〜32文字にしてください。")
    return value


def _name(display_name: str) -> str:
    value = " ".join((display_name or "").split())
    if not value:
        raise BadName("名前を入れてください。")
    if len(value) > NAME_MAX:
        raise BadName(f"名前は{NAME_MAX}文字までにしてください。")
    return value


def check_password(password: str, login: str = "") -> None:
    if len(password or "") < PASSWORD_MIN:
        raise BadPassword(f"パスワードは{PASSWORD_MIN}文字以上にしてください。")
    if len(password) > PASSWORD_MAX:
        raise BadPassword(f"パスワードは{PASSWORD_MAX}文字までにしてください。")
    if login and password.strip().lower() == login.strip().lower():
        raise BadPassword("ログイン名と同じパスワードは使えません。")


def hash_password(password: str) -> str:
    return HASHER.hash(password)


_DUMMY: list[str] = []


def _verify(stored: str | None, password: str) -> bool:
    """argon2 verify; an unknown name verifies against a dummy hash (the same time)."""
    if stored is None:
        if not _DUMMY:
            _DUMMY.append(HASHER.hash(secrets.token_hex(16)))
        stored = _DUMMY[0]
        try:
            HASHER.verify(stored, password)
        except (VerificationError, InvalidHashError):
            pass
        return False
    try:
        return HASHER.verify(stored, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# --- users ------------------------------------------------------------------------------------

def _user(row) -> User:
    return User(id=row["id"], login=row["login"], display_name=row["display_name"],
                role=row["role"], active=bool(row["active"]), org_id=row["org_id"],
                last_login_at=row["last_login_at"])


def user_count(repo: Repo) -> int:
    return repo.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def get_user(repo: Repo, user_id: int) -> User | None:
    row = repo.conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return None if row is None else _user(row)


def find_user(repo: Repo, login: str) -> User | None:
    row = repo.conn.execute("SELECT * FROM users WHERE login=?",
                            ((login or "").strip().lower(),)).fetchone()
    return None if row is None else _user(row)


def users(repo: Repo) -> list[User]:
    return [_user(r) for r in repo.conn.execute(
        "SELECT * FROM users ORDER BY active DESC, display_name, id")]


def _role(role: str) -> str:
    if role not in ROLES:
        raise AuthError("役割を選んでください。")
    return role


def _insert_user(repo: Repo, login: str, display_name: str, role: str, password: str,
                 now: float | None) -> int:
    login, name, role = normalize_login(login), _name(display_name), _role(role)
    check_password(password, login)
    if find_user(repo, login) is not None:
        raise BadLogin("このログイン名は使われています。ほかの名前にしてください。")
    stamp = _iso(_now(now))
    cur = repo.conn.execute(
        "INSERT INTO users (org_id, login, display_name, role, password_hash, active,"
        " created_at, password_changed_at) VALUES (?,?,?,?,?,1,?,?)",
        (ORG, login, name, role, hash_password(password), stamp, stamp))
    uid = cur.lastrowid
    repo.audit("user_created", f"user:{uid}", {"role": role})
    return uid


def create_user(repo: Repo, login: str, display_name: str, role: str, password: str,
                now: float | None = None) -> User:
    with repo._tx(join=True):
        uid = _insert_user(repo, login, display_name, role, password, now)
    return get_user(repo, uid)


def _active_admins(repo: Repo) -> int:
    return repo.conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]


def _keeps_an_admin(repo: Repo, user: User, role: str, active: bool) -> None:
    if user.role == "admin" and user.active and (role != "admin" or not active) \
            and _active_admins(repo) <= 1:
        raise LastAdmin(LAST_ADMIN)


def _existing(repo: Repo, user_id: int) -> User:
    user = get_user(repo, user_id)
    if user is None:
        raise AuthError("この人は見つかりません。")
    return user


def set_role(repo: Repo, user_id: int, role: str) -> None:
    """Another role; the person's sessions end (they sign in again)."""
    _role(role)
    with repo._tx(join=True):
        user = _existing(repo, user_id)
        if user.role == role:
            return
        _keeps_an_admin(repo, user, role, user.active)
        repo.conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
        _revoke_all(repo, user_id)
        repo.audit("role_changed", f"user:{user_id}", {"from": user.role, "to": role})


def set_active(repo: Repo, user_id: int, active: bool) -> None:
    """「使えなくする」 / 「使えるようにする」; switching off ends the person's sessions."""
    with repo._tx(join=True):
        user = _existing(repo, user_id)
        if user.active == active:
            return
        _keeps_an_admin(repo, user, user.role, active)
        repo.conn.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))
        GENERATION[0] += 1
        if not active:
            _revoke_all(repo, user_id)
        repo.audit("user_enabled" if active else "user_disabled", f"user:{user_id}", {})


def _set_password(repo: Repo, user: User, password: str, action: str) -> None:
    check_password(password, user.login)
    repo.conn.execute("UPDATE users SET password_hash=?, password_changed_at=? WHERE id=?",
                      (hash_password(password), _iso(time.time()), user.id))
    _revoke_all(repo, user.id)
    _cancel_resets(repo, user.id)
    repo.audit(action, f"user:{user.id}", {})


def change_password(repo: Repo, user_id: int, old: str, new: str) -> None:
    """The person's own change: the old password first. Every session of theirs ends (the
    screen starts a fresh one for the browser that asked)."""
    now = time.time()
    key = f"pw:{int(user_id)}"
    rows = _attempt(repo, [(key, LOGIN_LIMIT)], now)
    row = repo.conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None or not _verify(row[0], old or ""):
        raise LoginRefused("いまのパスワードがちがいます。")
    _forget(repo, rows, key)
    with repo._tx(join=True):
        _set_password(repo, _existing(repo, user_id), new, "password_changed")


def reset_password(repo: Repo, user_id: int, new: str) -> None:
    """Set a password without the old one (the operator's CLI, a reset code)."""
    with repo._tx(join=True):
        _set_password(repo, _existing(repo, user_id), new, "password_reset")


# --- rate limits ------------------------------------------------------------------------------

def _count(repo: Repo, key: str, now: float) -> int:
    return repo.conn.execute("SELECT COUNT(*) FROM login_failures WHERE key=? AND ts>?",
                             (key, now - WINDOW_SEC)).fetchone()[0]


def _attempt(repo: Repo, keys: list[tuple[str, int]], now: float) -> list[int]:
    """Record one try under every key first, in one write transaction, and only then count:
    parallel tries cannot pass a limit. TooManyTries when a key has more than its limit in
    WINDOW_SEC (the refused try counts too). The rows of a try that succeeds are removed
    (`_forget`); a failed one's stay as its failures."""
    with repo._tx(join=True):
        repo.conn.execute("DELETE FROM login_failures WHERE ts<?", (now - KEEP_FAILURES_SEC,))
        rows = [repo.conn.execute("INSERT INTO login_failures (key, ts) VALUES (?,?)",
                                  (key, now)).lastrowid for key, _ in keys]
        over = any(_count(repo, key, now) > limit for key, limit in keys)
    if over:
        raise TooManyTries(TOO_MANY)
    return rows


def _forget(repo: Repo, rows: list[int], *keys: str) -> None:
    with repo._tx(join=True):
        repo.conn.executemany("DELETE FROM login_failures WHERE rowid=?", [(r,) for r in rows])
        repo.conn.executemany("DELETE FROM login_failures WHERE key=?", [(k,) for k in keys])


# --- signing in -------------------------------------------------------------------------------

def login(repo: Repo, login_name: str, password: str, ip: str | None = None,
          now: float | None = None) -> User:
    """The person, or LoginRefused (the same words whether the name exists or not), or
    TooManyTries while the name or the source waits. `ip`: the source key (an address, the
    Tailscale login behind the local proxy, else the browser session: ui.auth_ui)."""
    now = _now(now)
    name = unicodedata.normalize("NFKC", login_name or "").strip().lower()[:64]
    key = f"login:{name}"
    keys = [(key, LOGIN_LIMIT)] + ([(f"ip:{ip}", IP_LIMIT)] if ip else [])
    rows = _attempt(repo, keys, now)
    row = repo.conn.execute("SELECT * FROM users WHERE login=?", (name,)).fetchone()
    ok = _verify(row["password_hash"] if row is not None else None, password or "")
    if not ok or not row["active"]:
        if _count(repo, key, now) == LOGIN_LIMIT:
            prev, repo.actor = repo.actor, None
            try:
                repo.audit("login_locked", f"user:{row['id']}" if row is not None else None,
                           {})
            finally:
                repo.actor = prev
        raise LoginRefused(LOGIN_REFUSED)
    user = _user(row)
    _forget(repo, rows, key)
    with repo._tx(join=True):
        repo.conn.execute("UPDATE users SET last_login_at=? WHERE id=?", (_iso(now), user.id))
        if HASHER.check_needs_rehash(row["password_hash"]):
            repo.conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                              (hash_password(password), user.id))
    return get_user(repo, user.id)


# --- sessions ---------------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sign(settings: Settings, payload: str) -> str:
    return hmac.new(_key(settings), payload.encode(), hashlib.sha256).hexdigest()


def start_session(repo: Repo, settings: Settings, user: User, now: float | None = None) -> str:
    """A new session for `user`; the cookie value (the only copy of its random part)."""
    now = _now(now)
    expires = int(now + settings.session_days * 86400)
    rand = secrets.token_urlsafe(24)
    with repo._tx(join=True):
        cur = repo.conn.execute(
            "INSERT INTO auth_sessions (org_id, user_id, secret_hash, created_at, expires_at)"
            " VALUES (?,?,?,?,?)", (ORG, user.id, _sha(rand), _iso(now), _iso(expires)))
        sid = cur.lastrowid
        prev, repo.actor = repo.actor, (user.id, user.login)
        try:
            repo.audit("login", f"user:{user.id}", {})
        finally:
            repo.actor = prev
    payload = f"{COOKIE_VERSION}.{sid}.{expires}.{rand}"
    return f"{payload}.{_sign(settings, payload)}"


def _session_row(repo: Repo, sid: int, now: float):
    row = repo.conn.execute(
        "SELECT s.secret_hash, s.expires_at, s.revoked_at, u.* FROM auth_sessions s"
        " JOIN users u ON u.id = s.user_id WHERE s.id=?", (sid,)).fetchone()
    if row is None or row["revoked_at"] or not row["active"] \
            or (_epoch(row["expires_at"]) or 0) <= now:
        return None
    return row


def check_cookie(repo: Repo, settings: Settings, cookie: str | None,
                 now: float | None = None) -> Session | None:
    """The session the cookie stands for, or None (forged, expired, revoked, switched off)."""
    now = _now(now)
    if not isinstance(cookie, str) or not cookie.isascii():
        return None
    parts = cookie.split(".")
    if len(parts) != 5 or parts[0] != COOKIE_VERSION or not _digits(parts[1]) \
            or not _digits(parts[2]):
        return None
    payload = ".".join(parts[:4])
    if not _same(_sign(settings, payload), parts[4]):
        return None
    if int(parts[2]) <= now:
        return None
    row = _session_row(repo, int(parts[1]), now)
    if row is None or not _same(row["secret_hash"], _sha(parts[3])):
        return None
    return Session(id=int(parts[1]), user=_user(row), expires=float(parts[2]))


def session_user(repo: Repo, session_id: int | None, now: float | None = None) -> User | None:
    """The person of a still valid session, by its id (the upload endpoint's tokens carry it;
    they are signed with their own key)."""
    if session_id is None:
        return None
    row = _session_row(repo, int(session_id), _now(now))
    return None if row is None else _user(row)


def revoke_session(repo: Repo, session_id: int) -> None:
    with repo._tx(join=True):
        row = repo.conn.execute("SELECT user_id FROM auth_sessions WHERE id=? AND revoked_at"
                                " IS NULL", (session_id,)).fetchone()
        if row is None:
            return
        repo.conn.execute("UPDATE auth_sessions SET revoked_at=? WHERE id=?",
                          (_iso(time.time()), session_id))
        GENERATION[0] += 1
        repo.audit("logout", f"user:{row[0]}", {})


def _revoke_all(repo: Repo, user_id: int) -> None:
    repo.conn.execute("UPDATE auth_sessions SET revoked_at=? WHERE user_id=? AND revoked_at"
                      " IS NULL", (_iso(time.time()), user_id))
    GENERATION[0] += 1


def revoke_everyone(repo: Repo) -> int:
    """End every session (after restoring an old backup: its sessions come back with it)."""
    with repo._tx(join=True):
        n = repo.conn.execute("UPDATE auth_sessions SET revoked_at=? WHERE revoked_at IS NULL",
                              (_iso(time.time()),)).rowcount
        repo.audit("sign_out_all", None, {"sessions": n})
    GENERATION[0] += 1
    return n


# --- codes ------------------------------------------------------------------------------------

def make_code() -> str:
    raw = "".join(secrets.choice(_ALPHABET) for _ in range(CODE_LEN))
    return "-".join(raw[i:i + 4] for i in range(0, CODE_LEN, 4))


def normalize_code(code: str) -> str:
    """The code's 12 letters, or "" when it cannot be one (other characters, another length)."""
    value = re.sub(r"[\s\-ー－]", "", unicodedata.normalize("NFKC", code or "")).upper()
    return value if re.fullmatch(f"[{_ALPHABET}]{{{CODE_LEN}}}", value) else ""


def _code_hash(code: str) -> str:
    return _sha(normalize_code(code))


def setup_code_path(settings: Settings) -> Path:
    return settings.data_dir / SETUP_FILE


def _fresh(path: Path) -> bool:
    return time.time() - path.stat().st_mtime <= SETUP_CODE_HOURS * 3600


def setup_code(settings: Settings, create: bool = True, renew: bool = False) -> str | None:
    """The one-time code of the first administrator (data_dir/.setup_code, 0600), made on the
    first ask. It is refused after SETUP_CODE_HOURS (None here); `renew` (the operator's CLI,
    `uv run pfb-admin setup-code`) then makes a new one. It is never logged."""
    path = setup_code_path(settings)
    if path.exists():
        if _fresh(path):
            return path.read_text(encoding="utf-8").strip()
        if not renew:
            return None
        path.unlink(missing_ok=True)
    elif not create:
        return None
    _create_private(path, make_code().encode())
    return path.read_text(encoding="utf-8").strip()


def _end_setup(settings: Settings) -> None:
    setup_code_path(settings).unlink(missing_ok=True)


def setup_admin(repo: Repo, settings: Settings, code: str, login_name: str, display_name: str,
                password: str, ip: str | None = None, now: float | None = None) -> User:
    """The first administrator, from the setup screen: only while nobody exists, only with a
    fresh setup code (SETUP_LIMIT tries per source, SETUP_ALL_LIMIT of everyone, per
    WINDOW_SEC; the CLI's create-admin is never held by them)."""
    now = _now(now)
    keys = [("setup:*", SETUP_ALL_LIMIT)] + ([(f"setup:{ip}", SETUP_LIMIT)] if ip else [])
    rows = _attempt(repo, keys, now)
    expected = setup_code(settings, create=False)
    given = normalize_code(code)
    if user_count(repo) or expected is None or not given \
            or not _same(normalize_code(expected), given):
        raise CodeRefused(CODE_REFUSED)
    try:
        user = create_first_admin(repo, settings, login_name, display_name, password, now=now)
    except AuthError:
        _forget(repo, rows)     # a bad name or password is not a wrong code
        raise
    _forget(repo, rows)
    return user


def create_first_admin(repo: Repo, settings: Settings, login_name: str, display_name: str,
                       password: str, now: float | None = None) -> User:
    """The very first administrator (the setup screen): refused once anybody exists (checked
    in the same transaction). Ends the setup."""
    return _create_admin(repo, settings, login_name, display_name, password, now, first=True)


def create_admin(repo: Repo, settings: Settings, login_name: str, display_name: str,
                 password: str, now: float | None = None) -> User:
    """An administrator made by the operator on the server (`pfb-admin create-admin`), also
    when people exist already. Ends the setup."""
    return _create_admin(repo, settings, login_name, display_name, password, now, first=False)


def _create_admin(repo: Repo, settings: Settings, login_name: str, display_name: str,
                  password: str, now: float | None, first: bool) -> User:
    with repo._tx(join=True):
        nobody = user_count(repo) == 0
        if first and not nobody:
            raise CodeRefused(CODE_REFUSED)
        uid = _insert_user(repo, login_name, display_name, "admin", password, now)
        if nobody:
            prev, repo.actor = repo.actor, (uid, normalize_login(login_name))
            try:
                repo.audit("setup_done", f"user:{uid}", {})
            finally:
                repo.actor = prev
    _end_setup(settings)
    return get_user(repo, uid)


def _new_code(repo: Repo, role: str, display_name: str, user_id: int | None,
              by: int | None, seconds: float, now: float | None) -> str:
    now = _now(now)
    code = make_code()
    repo.conn.execute(
        "INSERT INTO invites (org_id, code_hash, role, display_name, user_id, created_by,"
        " created_at, expires_at) VALUES (?,?,?,?,?,?,?,?)",
        (ORG, _code_hash(code), role, display_name, user_id, by, _iso(now), _iso(now + seconds)))
    return code


def create_invite(repo: Repo, role: str, display_name: str = "", by: int | None = None,
                  now: float | None = None) -> str:
    """A one-time code for one new account with `role` (INVITE_DAYS); shown once."""
    _role(role)
    name = " ".join((display_name or "").split())[:NAME_MAX]
    with repo._tx(join=True):
        code = _new_code(repo, role, name, None, by, INVITE_DAYS * 86400, now)
        repo.audit("invite_created", None, {"role": role})
    return code


def create_reset(repo: Repo, user_id: int, by: int | None = None,
                 now: float | None = None) -> str:
    """A one-time code with which `user_id` sets a new password (RESET_HOURS)."""
    with repo._tx(join=True):
        user = _existing(repo, user_id)
        _cancel_resets(repo, user.id)          # only the newest reset code works
        code = _new_code(repo, user.role, user.display_name, user.id, by, RESET_HOURS * 3600,
                         now)
        repo.audit("reset_created", f"user:{user_id}", {})
    return code


def _cancel_resets(repo: Repo, user_id: int) -> None:
    repo.conn.execute("UPDATE invites SET cancelled_at=? WHERE user_id=? AND used_at IS NULL"
                      " AND cancelled_at IS NULL", (_iso(time.time()), user_id))


def _open_code(repo: Repo, code: str, now: float):
    row = repo.conn.execute("SELECT * FROM invites WHERE code_hash=?",
                            (_code_hash(code),)).fetchone()
    if row is None or row["used_at"] or row["cancelled_at"] \
            or (_epoch(row["expires_at"]) or 0) <= now:
        return None
    if row["user_id"] is not None and not (get_user(repo, row["user_id"]) or User(
            0, "", "", "", active=False)).active:
        return None
    return row


def peek_code(repo: Repo, code: str, ip: str | None = None, now: float | None = None) -> Code:
    """What an entered code is (not used yet), or CodeRefused, or TooManyTries: tries are
    counted per source (`ip`) and per code prefix, never in one bucket for everyone."""
    now = _now(now)
    value = normalize_code(code)
    prefix = value[:4] or re.sub(r"[^A-Z0-9]", "", (code or "").upper())[:4].ljust(4, "-")
    keys = [(f"code:{prefix}", CODE_PREFIX_LIMIT)] + ([(f"code:src:{ip}", CODE_LIMIT)] if ip
                                                        else [])
    rows = _attempt(repo, keys, now)
    row = _open_code(repo, value, now) if value else None
    if row is None:
        raise CodeRefused(CODE_REFUSED)
    _forget(repo, rows)
    return Code(row["id"], row["role"], row["display_name"], row["user_id"])


def _use(repo: Repo, code: str, now: float, reset: bool):
    row = _open_code(repo, normalize_code(code), now) if normalize_code(code) else None
    if row is None or (row["user_id"] is not None) != reset:
        raise CodeRefused(CODE_REFUSED)
    used = repo.conn.execute("UPDATE invites SET used_at=? WHERE id=? AND used_at IS NULL",
                             (_iso(now), row["id"])).rowcount
    if not used:
        raise CodeRefused(CODE_REFUSED)
    return row


def redeem_invite(repo: Repo, code: str, login_name: str, display_name: str, password: str,
                  ip: str | None = None, now: float | None = None) -> User:
    now = _now(now)
    peek_code(repo, code, ip=ip, now=now)
    with repo._tx(join=True):
        row = _use(repo, code, now, reset=False)
        uid = _insert_user(repo, login_name, display_name, row["role"], password, now)
    return get_user(repo, uid)


def redeem_reset(repo: Repo, code: str, password: str, ip: str | None = None,
                 now: float | None = None) -> User:
    now = _now(now)
    peek_code(repo, code, ip=ip, now=now)
    with repo._tx(join=True):
        row = _use(repo, code, now, reset=True)
        user = _existing(repo, row["user_id"])
        _set_password(repo, user, password, "password_reset")
    return get_user(repo, user.id)


def open_invites(repo: Repo, now: float | None = None) -> list[dict]:
    """Codes not used, cancelled or expired: id, role, display_name, user_id, expires_at."""
    now = _now(now)
    rows = repo.conn.execute(
        "SELECT id, role, display_name, user_id, expires_at FROM invites WHERE used_at IS NULL"
        " AND cancelled_at IS NULL ORDER BY id").fetchall()
    return [dict(r) for r in rows if (_epoch(r["expires_at"]) or 0) > now]


def cancel_invite(repo: Repo, invite_id: int) -> None:
    with repo._tx(join=True):
        if repo.conn.execute("UPDATE invites SET cancelled_at=? WHERE id=? AND used_at IS NULL"
                             " AND cancelled_at IS NULL",
                             (_iso(time.time()), invite_id)).rowcount:
            repo.audit("invite_cancelled", f"invite:{invite_id}", {})
