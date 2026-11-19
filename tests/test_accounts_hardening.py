"""Security review of feat/accounts (2026-10-24): regressions for each finding.

HIGH  callbacks and fragment reruns run before (or without) ui/app.py's gate: every check
      re-validates the stored session (ui.access), and a page run without a valid editing
      session gets a read-only connection.
MED   the cookie is set by the server (HttpOnly) through /pfb-auth; its name carries the port.
MED   non-ASCII cookies, codes and upload tokens are refused, never crash.
MED   rate limits: attempts are recorded before they are checked; behind the local proxy the
      Tailscale identity (else the browser session) is the key; codes per prefix; the CLI is
      never limited by the web's setup lockout.
LOW   resets, current-password tries, the first admin, the setup code's age, データの扱い,
      viewer pragmas, sign-out-all.
Synthetic people and works only."""
import os
import sqlite3
import time

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from streamlit.testing.v1 import AppTest

from pfb_app import admin_cli
from pfb_app.services import auth
from pfb_app.services import chunked_upload as cu
from pfb_app.ui import access
from test_accounts_ui import PW, WORK, env, signed_in  # noqa: F401 (env)
from test_linecall_pages import lc  # noqa: F401 (fixture)
from test_wizard_ui_basic import UI, ok, screen_text

pytestmark = pytest.mark.real_auth


def script_app(source: str, token: str) -> AppTest:
    at = AppTest.from_string(source, default_timeout=60)
    at.session_state[access.TOKEN] = token
    return at.run()


WRITES = '''
import streamlit as st
from pfb_app.ui.common import get_repo

def write():
    get_repo().audit("probe", "x", {})

st.button("callback", key="cb", on_click=write)

@st.fragment
def part():
    if st.button("in fragment", key="fr"):
        write()
part()
'''


def probes(repo) -> int:
    return repo.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='probe'").fetchone()[0]


# --- HIGH: callbacks and fragments ------------------------------------------------------------

@pytest.mark.parametrize("change", ["revoke", "demote", "disable"])
def test_a_callback_after_the_session_ended_writes_nothing(env, change):
    settings, repo = env
    signed_in(env)
    user, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(script_app(WRITES, cookie))
    ok(at.button(key="cb").click().run())
    assert probes(repo) == 1                                  # allowed while signed in
    if change == "revoke":
        auth.revoke_session(repo, auth.check_cookie(repo, settings, cookie).id)
    elif change == "demote":
        auth.set_role(repo, user.id, "viewer")
    else:
        auth.set_active(repo, user.id, False)
    at = at.button(key="cb").click().run()
    assert probes(repo) == 1


def test_a_fragment_rerun_after_the_session_ended_writes_nothing(env):
    settings, repo = env
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(script_app(WRITES, cookie))
    ok(at.button(key="fr").click().run())
    assert probes(repo) == 1
    auth.revoke_session(repo, auth.check_cookie(repo, settings, cookie).id)
    at.button(key="fr").click().run()
    assert probes(repo) == 1


def test_an_expired_session_writes_nothing(env, monkeypatch):
    settings, repo = env
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(script_app(WRITES, cookie))
    real = time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + settings.session_days * 86400 + 5)
    monkeypatch.setattr(access, "CACHE_SEC", 0)
    at.button(key="cb").click().run()
    assert probes(repo) == 0


def test_the_session_user_in_session_state_alone_is_not_trusted(env):
    _, repo = env
    boss, _ = signed_in(env)
    at = AppTest.from_string(WRITES, default_timeout=60)
    at.session_state[access.USER] = boss          # no valid token: nobody
    at = at.run()
    at.button(key="cb").click().run()
    assert probes(repo) == 0
    assert access.VIEW_ONLY  # the refusal words exist


def test_the_review_page_toggle_after_demotion_is_refused(env):
    settings, repo = env
    signed_in(env)
    # 確認・承認 is a 管理者 page (pages/review.py's require_page): a second 管理者, demoted
    user, cookie = signed_in(env, role="admin", login="kanri2", name="管理 二郎")
    at = AppTest.from_file(str(UI / "pages" / "review.py"), default_timeout=60)
    at.session_state[access.TOKEN] = cookie
    at.session_state["current_assignment"] = 1
    at = ok(at.run())
    auth.set_role(repo, user.id, "viewer")
    at.toggle(key="auto-approve-1").set_value(True).run()
    assert not repo.get_assignment(1)["auto_approve"]


def test_the_live_demo_does_not_start_after_demotion(lc):
    from test_linecall_pages import LIVE_PAGE, controller
    repo = lc.repo
    auth.create_user(repo, "kanri", "管理 一郎", "admin", PW)
    user = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    cookie = auth.start_session(repo, lc.settings, user)
    at = AppTest.from_file(str(LIVE_PAGE), default_timeout=60)
    at.session_state[access.TOKEN] = cookie
    at = ok(at.run())
    auth.set_role(repo, user.id, "viewer")
    at.button(key="lc_live_start").click().run()
    assert controller(lc.settings).status() is None


# --- MED: the cookie is the server's ----------------------------------------------------------

def auth_client(settings):
    from pfb_app.ui import auth_api
    app = Starlette(routes=auth_api.routes(settings))
    return TestClient(app, base_url="http://pfb.test")


def test_the_cookie_name_carries_the_port():
    assert auth.cookie_name(8501) == "pfb_auth_8501"
    assert auth.cookie_name(8611) != auth.cookie_name(8501)


def test_a_ticket_becomes_an_httponly_cookie_once(env):
    settings, repo = env
    from pfb_app.ui import auth_api
    _, cookie = signed_in(env)
    ticket = auth_api.ticket_for(cookie, max_age=3600)
    with auth_client(settings) as c:
        r = c.post(auth_api.COOKIE_PATH, headers={"X-PFB-Ticket": ticket})
        assert r.status_code == 204
        set_cookie = r.headers["set-cookie"]
        assert set_cookie.startswith(f"{auth.cookie_name(auth_api.port())}=")
        assert cookie in set_cookie
        for part in ("HttpOnly", "SameSite=Strict", "Path=/", "Max-Age=3600"):
            assert part.lower() in set_cookie.lower(), part
        again = c.post(auth_api.COOKIE_PATH, headers={"X-PFB-Ticket": ticket})
        assert again.status_code == 403 and "set-cookie" not in again.headers
        other = c.post(auth_api.COOKIE_PATH, headers={"X-PFB-Ticket": ticket,
                                                       "Origin": "https://evil.example"})
        assert other.status_code == 403


def test_an_old_ticket_is_refused_and_sign_out_clears_the_cookie(env, monkeypatch):
    settings, _ = env
    from pfb_app.ui import auth_api
    _, cookie = signed_in(env)
    ticket = auth_api.ticket_for(cookie, max_age=3600)
    real = time.monotonic
    monkeypatch.setattr(auth_api.time, "monotonic", lambda: real() + auth_api.TICKET_SEC + 1)
    with auth_client(settings) as c:
        assert c.post(auth_api.COOKIE_PATH, headers={"X-PFB-Ticket": ticket}).status_code == 403
        r = c.post(auth_api.CLEAR_PATH)
        assert r.status_code == 204 and "max-age=0" in r.headers["set-cookie"].lower()


def test_after_login_the_page_gets_a_ticket_never_the_cookie(env, monkeypatch):
    from pfb_app.ui import auth_api, auth_ui
    monkeypatch.setattr(auth_api, "MOUNTED", True)
    signed_in(env)
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
    at.text_input(key="pfb_login_name").input("kanri")
    at.text_input(key="pfb_login_pw").input(PW)
    at = ok(at.button(key="pfb_login_go").click().run())
    page = "".join(el.proto.body for el in at.get("html"))
    token = at.session_state[auth_ui.TOKEN]
    assert auth_api.COOKIE_PATH in page and token not in page and "document.cookie" not in page


# --- MED: non-ASCII ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["v1.1.9999999999.abc.é", "v1.².9999999999.a.b",
                                 "v1.1.9999999999.ａｂｃ.def", "日本語"])
def test_non_ascii_cookies_count_for_nothing(env, bad):
    settings, repo = env
    signed_in(env)
    assert auth.check_cookie(repo, settings, bad) is None


def test_a_non_ascii_browser_cookie_is_cleared_not_a_crash(env, monkeypatch):
    from pfb_app.ui import auth_api, auth_ui
    monkeypatch.setattr(auth_api, "MOUNTED", True)
    signed_in(env)
    monkeypatch.setattr(auth_ui, "_browser_cookie", lambda: "v1.1.9999999999.ü.é")
    at = ok(AppTest.from_file(str(UI / "app.py"), default_timeout=60).run())
    assert "ログイン" in screen_text(at) and WORK not in screen_text(at)
    assert any(auth_api_path() in el.proto.body for el in at.get("html"))


def auth_api_path():
    from pfb_app.ui import auth_api
    return auth_api.CLEAR_PATH


def test_non_ascii_setup_codes_are_refused_not_a_crash(env):
    settings, repo = env
    auth.setup_code(settings)
    with pytest.raises(auth.CodeRefused):
        auth.setup_admin(repo, settings, "あいうえ-かきくけ-さしすせ", "kanri", "管理", PW)
    with pytest.raises(auth.CodeRefused):
        auth.peek_code(repo, "ＡＢＣＤ-ÉÉÉÉ-²²²²")


@pytest.mark.parametrize("bad", ["²²²²²²²²²².0123456789abcdef0123456789abcdef.é",
                                 "1.0123456789abcdef0123456789abcdef.².x", "ü.ü.ü"])
def test_non_ascii_upload_tokens_are_refused(env, bad):
    settings, _ = env
    assert cu.check_token(settings, bad) is None and cu.token_session(settings, bad) is None


def test_streamlit_hides_error_details():
    text = (UI.parents[2] / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'showErrorDetails = "none"' in text


# --- MED: rate limits -------------------------------------------------------------------------

def test_attempts_are_recorded_before_they_are_checked(env):
    """Parallel sign-ins cannot pass the limit: each one is counted before its verify."""
    _, repo = env
    signed_in(env)
    seen = []
    real = auth._verify

    def slow(stored, password):
        seen.append(repo.conn.execute("SELECT COUNT(*) FROM login_failures WHERE key=?",
                                      ("login:kanri",)).fetchone()[0])
        return real(stored, password)
    auth._verify = slow
    try:
        with pytest.raises(auth.LoginRefused):
            auth.login(repo, "kanri", "wrong password")
    finally:
        auth._verify = real
    assert seen == [1]


def test_behind_the_proxy_the_tailscale_header_is_ignored(env, monkeypatch):
    """Ruling 2026-10-24 (release 4): other users of the server reach 127.0.0.1 directly and
    can send any Tailscale-User-Login, so it never names the source: without an address the
    key is this browser session (the per-name lock and the 60-bit codes are the protection)."""
    from types import SimpleNamespace

    from pfb_app.ui import auth_ui
    monkeypatch.setattr(auth_ui.st, "session_state", {})   # one browser session

    def sent(login, ip=None):
        monkeypatch.setattr(auth_ui.st, "context", SimpleNamespace(
            ip_address=ip, headers={"Tailscale-User-Login": login} if login else {}))
        return auth_ui.source_key()
    key = sent("a@example.com")
    assert key.startswith("browser:") and "example" not in key
    assert sent("b@example.com") == key                # another header: the same key
    assert sent(None) == key
    assert sent("a@example.com", ip="100.64.0.9") == "100.64.0.9"


def test_codes_are_limited_per_prefix_not_for_everyone(env):
    _, repo = env
    boss, _ = signed_in(env)
    good = auth.create_invite(repo, "viewer", "", by=boss.id)
    for i in range(auth.CODE_LIMIT + 5):
        with pytest.raises((auth.CodeRefused, auth.TooManyTries)):
            auth.peek_code(repo, "AAAA-BBBB-CC%02d" % i, ip=f"100.64.0.{i}")
    assert auth.peek_code(repo, good, ip="100.64.1.1")       # others still get in


def test_the_cli_is_not_held_by_the_web_setup_lockout(env, monkeypatch, capsys):
    settings, repo = env
    auth.setup_code(settings)
    for _ in range(auth.SETUP_LIMIT + 1):
        with pytest.raises((auth.CodeRefused, auth.TooManyTries)):
            auth.setup_admin(repo, settings, "AAAA-BBBB-CCCC", "kanri", "管理", PW,
                             ip="100.64.0.9")
    import io
    import sys
    import yaml
    cfg = settings.data_dir.parent / "pfb.yaml"
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    monkeypatch.setattr(sys, "stdin", io.StringIO(PW + "\n"))
    assert admin_cli.main(["create-admin", "--login", "kanri", "--password-stdin"]) == 0
    assert auth.find_user(repo, "kanri").role == "admin" and yaml


# --- LOW ----------------------------------------------------------------------------------------

def test_a_new_password_cancels_open_reset_codes(env):
    _, repo = env
    boss, _ = signed_in(env)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    code = auth.create_reset(repo, helper.id, by=boss.id)
    auth.change_password(repo, helper.id, PW, "another long secret")
    with pytest.raises(auth.CodeRefused):
        auth.peek_code(repo, code)


def test_a_new_reset_code_cancels_the_older_one(env):
    _, repo = env
    boss, _ = signed_in(env)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    first = auth.create_reset(repo, helper.id, by=boss.id)
    second = auth.create_reset(repo, helper.id, by=boss.id)
    with pytest.raises(auth.CodeRefused):
        auth.peek_code(repo, first)
    assert auth.peek_code(repo, second).user_id == helper.id


def test_current_password_tries_are_limited(env):
    _, repo = env
    user, _ = signed_in(env)
    for _ in range(auth.LOGIN_LIMIT):
        with pytest.raises(auth.LoginRefused):
            auth.change_password(repo, user.id, "not the password", "another long secret")
    with pytest.raises(auth.TooManyTries):
        auth.change_password(repo, user.id, PW, "another long secret")


def test_setup_is_only_for_the_very_first_admin(env):
    settings, repo = env
    code = auth.setup_code(settings)
    auth.create_user(repo, "someone", "誰か", "viewer", PW)   # appeared meanwhile
    with pytest.raises(auth.CodeRefused):
        auth.setup_admin(repo, settings, code, "kanri", "管理", PW)
    with pytest.raises(auth.AuthError):
        auth.create_first_admin(repo, settings, "kanri", "管理", PW)


def test_the_setup_code_expires_and_the_cli_makes_a_new_one(env):
    settings, repo = env
    code = auth.setup_code(settings)
    path = settings.data_dir / auth.SETUP_FILE
    old = time.time() - auth.SETUP_CODE_HOURS * 3600 - 60
    os.utime(path, (old, old))
    with pytest.raises(auth.CodeRefused):
        auth.setup_admin(repo, settings, code, "kanri", "管理", PW)
    assert auth.setup_code(settings, renew=True) != code
    assert auth.setup_admin(repo, settings, auth.setup_code(settings), "kanri", "管理", PW)


def test_the_data_page_is_for_admins_only(env):
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = AppTest.from_file(str(UI / "pages" / "data.py"), default_timeout=60)
    at.session_state[access.TOKEN] = cookie
    at = ok(at.run())
    # release 4: the page file itself stops first (access.require_page), before data_view's
    # own 「管理者だけ」 check
    assert access.NOT_HERE in screen_text(at)


def test_a_viewer_s_connection_refuses_writing_pragmas(tmp_path):
    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE t (a)")
    access.read_only_guard(conn)
    assert conn.execute("PRAGMA table_info(t)").fetchall()
    for pragma in ("PRAGMA journal_mode=DELETE", "PRAGMA user_version=5",
                   "PRAGMA writable_schema=ON"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(pragma)


def test_sign_out_all_after_restoring_a_backup(env, monkeypatch, capsys):
    settings, repo = env
    _, cookie = signed_in(env)
    monkeypatch.setenv("PFB_CONFIG", str(settings.data_dir.parent / "pfb.yaml"))
    assert admin_cli.main(["sign-out-all"]) == 0
    assert auth.check_cookie(repo, settings, cookie) is None
