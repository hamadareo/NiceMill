"""The sign-in gate of ui/app.py and the person's own account box (services.auth decides).

`gate(settings)` runs first on every page run. With a valid session (this browser session's
token, else the browser's `pfb_auth` cookie as it was sent when the page connected) it sets
the person for the run (ui.access) and returns them. Otherwise it shows, through a
navigation of its own (so no page of the app is reachable), either
- 「はじめに、管理者を登録します」 while nobody is registered (the one-time setup code from
  `uv run pfb-admin setup-code`; the UI log says where to get it, never the code), or
- 「ログイン」, with 「招待コード・再設定コードをお持ちの方」 below it,
and stops the run.

With the setting `auth_enabled: false` (open mode, ruling 2026-11-13: 「ログインなしで使える
ように」) `resolve` answers an open administrator instead — no sign-in screen is ever shown
and the account box is one quiet caption. The tables, screens and CLI below stay untouched,
so setting the flag back to true returns the gate exactly as it was. The open session is
served only to a browser that arrives directly from a private / tailnet / loopback address
without the marks of a public proxy (ui/open_guard.py, 2026-11-14); any other browser meets the
sign-in gate as with the flag on.

The cookie (`pfb_auth_<port>`) is set by the server, HttpOnly (ui/auth_api): the page gets
a one-time ticket in a hidden element (`st.html` with its script allowed, the notify.py
pattern) and exchanges it at /pfb-auth/cookie; a stale cookie is cleared through
/pfb-auth/clear. Sign-in tries are counted per source (`source_key`: the address, else this
browser session; never a header a client can choose, such as Tailscale-User-Login).
"""
import json
import logging
import secrets
import time
from html import escape

import streamlit as st

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui import access, auth_api, keyed, open_guard
from pfb_app.ui.common import flash

log = logging.getLogger(__name__)

TOKEN = access.TOKEN             # this browser session's cookie value (server side only)
SENT = "_pfb_auth_sent"          # the token a ticket was last handed out for
CLEARED = "_pfb_auth_cleared"    # a stale browser cookie was asked to be cleared
BROWSER = "_pfb_auth_browser"    # a random key of this browser session (rate limits)
CODE = "_pfb_auth_code"          # (code, auth.Code) checked on the code form
APP = "ナイスミル"
SETUP_TITLE = "はじめに、管理者を登録します"
SETUP_GUIDE = ("このアプリには、まだ誰も登録されていません。最初の管理者を登録すると、"
               "ほかの人を招待できるようになります。")
SETUP_CODE_HELP = ("確認コードは、このアプリを入れた人に聞いてください"
                   "（サーバーで uv run pfb-admin setup-code と入力すると表示されます）。")
SETUP_GO = "管理者を登録して始める"
LOGIN_TITLE = "ログイン"
LOGIN_GO = "ログイン"
CODES = "招待コード・再設定コードをお持ちの方"
CODE_CHECK = "コードを確かめる"
CODE_JOIN = "登録して始める"
CODE_RESET = "パスワードを設定して始める"
MISMATCH = "2回のパスワードがちがいます。"
PASSWORD_HELP = f"{auth.PASSWORD_MIN}文字以上。長めの文や、ほかで使っていないものがおすすめです。"
LOGIN_HELP = "半角の英字・数字・「. _ - @」で3〜32文字（大文字と小文字は区別しません）。"
ACCOUNT = "アカウント"
LOGOUT = "ログアウト"
PW_TITLE = "パスワードを変える"
PW_GO = "パスワードを変える"
PW_DONE = "パスワードを変えました。ほかの端末では、もう一度ログインしてください。"
OPEN_NAME = "だれでも"      # the open mode's person (settings auth_enabled: false)
OPEN_NOTE = "ログインなしで使えています。"
OPEN_USER = auth.User(id=0, login="open", display_name=OPEN_NAME, role="admin")
OPEN_SESSION = auth.Session(id=0, user=OPEN_USER, expires=4e9)   # never expires; id 0 is not
# a real session row: access.session_id() reads it as None, and audit rows say (0, "open")
_HINTED: list[bool] = []   # the "nobody registered yet" line goes to the log once per process


def _repo(settings: Settings, schema: bool = True) -> Repo:
    """A connection of the gate's own (never read-only: a viewer's sign-out writes)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(files.db_path(settings))
    if schema:
        init_schema(conn)
    return Repo(conn, settings)


def _browser_cookie() -> str | None:
    try:
        return st.context.cookies.get(auth.cookie_name(auth_api.port()))
    except Exception:   # noqa: BLE001 - no browser (a bare script run)
        return None


def _ip() -> str | None:
    try:
        return st.context.ip_address
    except Exception:   # noqa: BLE001
        return None


def source_key() -> str:
    """Whom sign-in tries are counted for: the address; without one (the local proxy, or a
    connection to 127.0.0.1) this browser session. The Tailscale-User-Login header is never
    used (ruling 2026-10-24): other users of the server reach 127.0.0.1 directly and can send
    any value. A browser key can be renewed by opening a new session, so the per-name lock
    and the 60-bit codes are the real protection (docs/accounts.md)."""
    ip = _ip()
    if ip:
        return ip
    key = st.session_state.get(BROWSER)
    if not isinstance(key, str):
        key = st.session_state[BROWSER] = secrets.token_hex(8)
    return f"browser:{key}"


def resolve(settings: Settings) -> auth.Session | None:
    """This browser session's valid session, or None (a bad cookie is just no session).
    Open mode (auth_enabled: false) answers the open administrator to a browser that reaches
    the app directly from a private / tailnet / loopback address (ui.open_guard); any other
    browser (a public tunnel, an unknown peer) is resolved as with the gate on."""
    if not settings.auth_enabled and open_guard.streamlit_allows():
        return OPEN_SESSION
    if not files.db_path(settings).exists():
        return None
    repo = _repo(settings, schema=False)
    try:
        token = st.session_state.get(TOKEN)
        session = auth.check_cookie(repo, settings, token) if token else None
        if session is None:
            st.session_state.pop(TOKEN, None)
            cookie = _browser_cookie()
            session = auth.check_cookie(repo, settings, cookie) if cookie else None
            if session is not None:
                st.session_state[TOKEN] = cookie
        return session
    finally:
        repo.conn.close()


def _fetch_html(path: str, ticket: str | None) -> str:
    """A hidden element whose script asks the server to set (ticket) or clear the cookie."""
    headers = json.dumps({"X-PFB-Ticket": ticket} if ticket else {})
    js = ("(function(){try{fetch(%s,{method:'POST',credentials:'same-origin',headers:%s})"
          ".catch(function(){});}catch(e){}})();") % (json.dumps(path), headers)
    return ("<style>.st-key-pfb_auth_cookie { display: none; }</style>"
            f'<span hidden data-pfb-auth="{escape(path)}"></span><script>{js}</script>')


def _emit(path: str, ticket: str | None = None) -> None:
    with keyed.container(key="pfb_auth_cookie"):
        st.html(_fetch_html(path, ticket), unsafe_allow_javascript=True)


def _start(repo: Repo, settings: Settings, user: auth.User) -> None:
    st.session_state[TOKEN] = auth.start_session(repo, settings, user)
    for key in (CODE, CLEARED):
        st.session_state.pop(key, None)


def gate(settings: Settings) -> auth.User:
    """The signed-in person; otherwise the setup or sign-in screen, and the run stops."""
    st.session_state.pop(access.USER, None)      # old states: never trusted
    st.session_state.pop(access.SESSION, None)
    session = access.current_session()
    if session is None:
        st.navigation([st.Page(lambda: _signed_out(settings), title=LOGIN_TITLE,
                               url_path="login", default=True)], position="hidden").run()
        st.stop()
    token = st.session_state.get(TOKEN)
    if auth_api.MOUNTED and token and _browser_cookie() != token \
            and st.session_state.get(SENT) != token:
        st.session_state[SENT] = token
        _emit(auth_api.COOKIE_PATH,
              auth_api.ticket_for(token, max(1, int(session.expires - time.time()))))
    return session.user


# --- the signed-out screens -----------------------------------------------------------------

def _signed_out(settings: Settings) -> None:
    from pfb_app.ui.wizard import theme   # noqa: PLC0415
    theme.inject_theme()
    if auth_api.MOUNTED and _browser_cookie() and not st.session_state.get(CLEARED):
        st.session_state[CLEARED] = True
        _emit(auth_api.CLEAR_PATH)   # a stale sign-in in the browser: cleared
    repo = _repo(settings)
    try:
        st.html(theme.brand_html())
        if auth.user_count(repo) == 0:
            _setup(repo, settings)
        else:
            _login(repo, settings)
            _codes(repo, settings)
    finally:
        repo.conn.close()


def _hint_once(settings: Settings) -> None:
    if not _HINTED:
        _HINTED.append(True)
        log.warning("No administrator is registered yet: the screens show the setup screen. "
                    "Run `uv run pfb-admin setup-code` on this server (same PFB_CONFIG) to see "
                    "the one-time setup code.")


def _setup(repo: Repo, settings: Settings) -> None:
    auth.setup_code(settings)          # made now, so the CLI shows the same one
    _hint_once(settings)
    from pfb_app.ui.wizard import theme   # noqa: PLC0415
    with theme.step_card(SETUP_TITLE, SETUP_GUIDE, key="setup"):
        with st.form("pfb_setup", border=False):
            code = st.text_input("確認コード", key="pfb_setup_code", help=SETUP_CODE_HELP,
                                 placeholder="XXXX-XXXX-XXXX")
            st.caption(SETUP_CODE_HELP)
            name = st.text_input("名前（画面に出ます）", key="pfb_setup_name",
                                 max_chars=auth.NAME_MAX)
            login = st.text_input("ログイン名", key="pfb_setup_login", help=LOGIN_HELP)
            pw = st.text_input("パスワード", type="password", key="pfb_setup_pw",
                               help=PASSWORD_HELP)
            pw2 = st.text_input("パスワード（もう一度）", type="password", key="pfb_setup_pw2")
            go = st.form_submit_button(SETUP_GO, type="primary", key="pfb_setup_go")
    if not go:
        return
    if pw != pw2:
        st.error(MISMATCH)
        return
    try:
        user = auth.setup_admin(repo, settings, code, login, name, pw, ip=source_key())
    except auth.AuthError as e:
        st.error(str(e))
        return
    _start(repo, settings, user)
    st.rerun()


def _login(repo: Repo, settings: Settings) -> None:
    from pfb_app.ui.wizard import theme   # noqa: PLC0415
    with theme.step_card(LOGIN_TITLE, "ログイン名とパスワードを入れてください。", key="login"):
        with st.form("pfb_login", border=False):
            login = st.text_input("ログイン名", key="pfb_login_name", autocomplete="username")
            pw = st.text_input("パスワード", type="password", key="pfb_login_pw",
                               autocomplete="current-password")
            go = st.form_submit_button(LOGIN_GO, type="primary", key="pfb_login_go")
        st.caption("パスワードを忘れたときは、管理者に再設定コードを作ってもらってください。")
    if not go:
        return
    try:
        user = auth.login(repo, login, pw, ip=source_key())
    except auth.AuthError as e:
        st.error(str(e))
        return
    _start(repo, settings, user)
    st.rerun()


def _codes(repo: Repo, settings: Settings) -> None:
    held = st.session_state.get(CODE)
    with st.expander(CODES, expanded=held is not None):
        if held is None:
            with st.form("pfb_code_form", border=False):
                code = st.text_input("コード", key="pfb_code", placeholder="XXXX-XXXX-XXXX")
                check = st.form_submit_button(CODE_CHECK, key="pfb_code_check")
            if check:
                try:
                    found = auth.peek_code(repo, code, ip=source_key())
                except auth.AuthError as e:
                    st.error(str(e))
                else:
                    st.session_state[CODE] = (code, found)
                    st.rerun()
            return
        code, found = held
        if found.user_id is None:
            _join(repo, settings, code, found)
        else:
            _reset(repo, settings, code)
        if st.button("ほかのコードを入れる", key="pfb_code_back", type="tertiary"):
            st.session_state.pop(CODE, None)
            st.rerun()


def _join(repo: Repo, settings: Settings, code: str, found: auth.Code) -> None:
    st.caption(f"「{auth.ROLE_LABELS.get(found.role, found.role)}」として登録します。")
    with st.form("pfb_code_join", border=False):
        name = st.text_input("名前（画面に出ます）", value=found.display_name,
                             key="pfb_code_name", max_chars=auth.NAME_MAX)
        login = st.text_input("ログイン名", key="pfb_code_login", help=LOGIN_HELP)
        pw = st.text_input("パスワード", type="password", key="pfb_code_pw", help=PASSWORD_HELP)
        pw2 = st.text_input("パスワード（もう一度）", type="password", key="pfb_code_pw2")
        go = st.form_submit_button(CODE_JOIN, key="pfb_code_go")
    if not go:
        return
    if pw != pw2:
        st.error(MISMATCH)
        return
    try:
        user = auth.redeem_invite(repo, code, login, name, pw, ip=source_key())
    except auth.AuthError as e:
        st.error(str(e))
        return
    _start(repo, settings, user)
    st.rerun()


def _reset(repo: Repo, settings: Settings, code: str) -> None:
    st.caption("新しいパスワードを決めてください。")
    with st.form("pfb_code_reset", border=False):
        pw = st.text_input("新しいパスワード", type="password", key="pfb_code_pw",
                           help=PASSWORD_HELP)
        pw2 = st.text_input("新しいパスワード（もう一度）", type="password", key="pfb_code_pw2")
        go = st.form_submit_button(CODE_RESET, key="pfb_code_go")
    if not go:
        return
    if pw != pw2:
        st.error(MISMATCH)
        return
    try:
        user = auth.redeem_reset(repo, code, pw, ip=source_key())
    except auth.AuthError as e:
        st.error(str(e))
        return
    _start(repo, settings, user)
    st.rerun()


# --- the signed-in person's own box (the sidebar) ---------------------------------------------

def account_box(settings: Settings) -> None:
    """「名前（役割）」, 「パスワードを変える」 and 「ログアウト」. Open mode is one
    caption: there is no password to change and nothing to sign out of."""
    user = access.current_user()
    if user is None:
        return
    if user is OPEN_USER:
        st.caption(OPEN_NOTE)
        return
    st.html(f'<p class="pfb-sidenav__who">{escape(user.label)}</p>')
    with st.popover(ACCOUNT, icon=":material/person:", width="stretch"):
        with st.form("pfb_pw_form", border=False, enter_to_submit=False):
            st.caption(PW_TITLE)
            old = st.text_input("いまのパスワード", type="password", key="pfb_pw_old")
            new = st.text_input("新しいパスワード", type="password", key="pfb_pw_new",
                                help=PASSWORD_HELP)
            new2 = st.text_input("新しいパスワード（もう一度）", type="password", key="pfb_pw_new2")
            go = st.form_submit_button(PW_GO, key="pfb_pw_go")
        if go:
            _change_password(settings, user, old, new, new2)
        if st.button(LOGOUT, key="pfb_logout", icon=":material/logout:", width="stretch"):
            _logout(settings, user)


def _change_password(settings: Settings, user: auth.User, old: str, new: str,
                     new2: str) -> None:
    if new != new2:
        st.error(MISMATCH)
        return
    repo = _repo(settings)
    repo.actor = (user.id, user.login)
    try:
        auth.change_password(repo, user.id, old, new)
        fresh = auth.get_user(repo, user.id)
        st.session_state[TOKEN] = auth.start_session(repo, settings, fresh)
    except auth.AuthError as e:
        st.error(str(e))
        return
    finally:
        repo.conn.close()
    flash(PW_DONE)
    st.rerun()


def _logout(settings: Settings, user: auth.User) -> None:
    repo = _repo(settings)
    repo.actor = (user.id, user.login)
    try:
        sid = access.session_id()
        if sid is not None:
            auth.revoke_session(repo, sid)
    finally:
        repo.conn.close()
    for key in (TOKEN, access.USER, access.SESSION, CODE):
        st.session_state.pop(key, None)
    st.rerun()
