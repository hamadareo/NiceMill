"""Who is signed in on this browser session, and what they may do: one-line checks for the
screens (services.auth decides).

Every check re-validates the session held by this browser session (`TOKEN`, through
ui.auth_ui.resolve), never a person merely kept in session_state: widget callbacks run before
ui/app.py's gate and fragment reruns run without it, so a revoked, demoted, switched-off or
expired session must fail here too. The answer is cached per token for CACHE_SEC (and
dropped at once when this process ends a session: services.auth.GENERATION).

- `can_edit()`: 管理者 and 評価する人 (new works, uploads, approvals, edits…).
- `is_admin()`: 管理者 only (削除, 団体の設定, データの扱い, 管理用の画面, 利用する人).
- In a page run without a valid editing session, ui.common.get_repo hands out a read-only
  connection (`read_only_guard`); `refused(error)` tells ui/app.py to say VIEW_ONLY.
- `refuse_unless_edit()`: for callbacks whose effect is not a database write (the live demo).
- `require_page(need)`: the first line of every file in ui/pages/. ui/app.py's navigation
  offers each page only to its roles, but Streamlit can run a page file without app.py: with a
  `pages/` folder next to the main script, a fresh server process runs its first script run
  in the old pages-folder mode (until st.navigation is first called), and an address such as
  /users then runs pages/users.py directly, gate-less. ui/serve.py turns that mode off
  (`no_pages_folder_mode`), and this check stops such a run anyway.
"""
import sqlite3
import threading
import time

import streamlit as st

from pfb_app.services import auth

TOKEN = "_pfb_auth_token"        # this browser session's sign-in (the cookie's value)
USER = "_pfb_auth_user"          # never trusted (the session is re-checked from TOKEN)
SESSION = "_pfb_auth_session"
VIEW_ONLY = "見るだけの人は、この操作はできません。"
NOT_HERE = "このページは開けません。ホームから開いてください。"
CACHE_SEC = 1.0
_cache: dict[str, tuple[float, int, auth.Session | None]] = {}
_lock = threading.Lock()


def in_page_run() -> bool:
    """True inside a page run, a callback or a fragment rerun (not a bare thread)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx   # noqa: PLC0415
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:   # noqa: BLE001
        return False


def current_session() -> auth.Session | None:
    """This browser session's valid session (re-checked; cached CACHE_SEC per token)."""
    if not in_page_run():
        return None
    try:
        token = st.session_state.get(TOKEN)
    except Exception:   # noqa: BLE001
        return None
    now = time.monotonic()
    if isinstance(token, str):
        with _lock:
            hit = _cache.get(token)
        if hit is not None and hit[0] > now and hit[1] == auth.GENERATION[0]:
            return hit[2]
    from pfb_app.ui import auth_ui          # noqa: PLC0415 - auth_ui imports this module
    from pfb_app.ui.common import get_settings   # noqa: PLC0415
    gen = auth.GENERATION[0]
    session = auth_ui.resolve(get_settings())
    token = st.session_state.get(TOKEN)
    if isinstance(token, str):
        with _lock:
            for key in [k for k, v in _cache.items() if v[0] <= now]:
                _cache.pop(key, None)
            _cache[token] = (now + CACHE_SEC, gen, session)
    return session


def current_user() -> auth.User | None:
    session = current_session()
    return session.user if session is not None else None


def role() -> str | None:
    user = current_user()
    return user.role if user is not None else None


def can_edit() -> bool:
    return auth.can(role(), "edit")


def is_admin() -> bool:
    return auth.can(role(), "admin")


def is_viewer() -> bool:
    return role() == "viewer"


def session_id() -> int | None:
    session = current_session()
    return session.id if session is not None and session.id > 0 else None


def actor() -> tuple[int, str] | None:
    user = current_user()
    return (user.id, user.login) if user is not None else None


def refuse_unless_edit() -> bool:
    """True (and VIEW_ONLY said on the next run) when this session may not change anything."""
    if can_edit():
        return False
    from pfb_app.ui.common import flash   # noqa: PLC0415
    flash(VIEW_ONLY, kind="warning")
    return True


def require_page(need: str | None = None) -> None:
    """Stop this page file's run (st.stop, with NOT_HERE) unless this browser session has a
    valid sign-in that may open it: `need` None (any role), "edit" or "admin"."""
    user = current_user()
    if user is not None and (need is None or auth.can(user.role, need)):
        return
    st.info(NOT_HERE)
    st.stop()


def no_pages_folder_mode() -> None:
    """Never run the old pages-folder mode (see require_page): Streamlit decides it once per
    process from whether `pages/` sits next to the main script, unless already decided."""
    from streamlit.runtime.pages_manager import PagesManager   # noqa: PLC0415
    PagesManager.uses_pages_directory = False


_WRITES = {getattr(sqlite3, name) for name in (
    "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER",
    "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE", "SQLITE_ATTACH",
    "SQLITE_DETACH", "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE") if hasattr(sqlite3, name)}
# pragmas a reading screen may ask: about a table or an index (its name is the argument), or
# a value read without setting one; nothing that changes the database
_NAMED_PRAGMAS = {"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo",
                  "foreign_key_list"}
_VALUE_PRAGMAS = {"table_list", "database_list", "data_version", "page_count", "page_size",
                  "freelist_count", "user_version", "schema_version", "compile_options",
                  "journal_mode", "foreign_keys"}


def _authorizer(action, arg1, arg2, *_):
    if action in _WRITES:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA:
        name = (arg1 or "").lower()
        ok = name in _NAMED_PRAGMAS or (name in _VALUE_PRAGMAS and arg2 is None)
        return sqlite3.SQLITE_OK if ok else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def read_only_guard(conn: sqlite3.Connection) -> None:
    """From now on this connection reads only: any write raises sqlite3.DatabaseError."""
    conn.set_authorizer(_authorizer)


def refused(error: BaseException) -> bool:
    """The error is a write refused by read_only_guard."""
    return isinstance(error, sqlite3.DatabaseError) and "not authorized" in str(error)
