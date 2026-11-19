"""services.auth: passwords, users and roles, signed sessions, rate limits, the first
administrator's setup code, invitations and password resets (synthetic people only)."""
import os
import stat

import pytest

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

PW = "correct horse battery"
T0 = 1_800_000_000.0


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    settings.data_dir.mkdir(parents=True)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return settings, Repo(conn, settings)


def admin(repo, login="kanri", name="管理 一郎"):
    return auth.create_user(repo, login, name, "admin", PW)


def audits(repo, action):
    return [tuple(r) for r in repo.conn.execute(
        "SELECT target, actor_id, detail FROM audit_log WHERE action=? ORDER BY id", (action,))]


# --- roles -----------------------------------------------------------------------------------

def test_roles_and_what_each_may_do():
    assert auth.ROLE_LABELS == {"admin": "管理者", "evaluator": "評価する人",
                                "viewer": "見るだけの人"}
    assert auth.can("admin", "edit") and auth.can("admin", "admin")
    assert auth.can("evaluator", "edit") and not auth.can("evaluator", "admin")
    assert not auth.can("viewer", "edit") and not auth.can("viewer", "admin")
    assert not auth.can("nobody", "edit") and not auth.can(None, "edit")


# --- passwords and users ---------------------------------------------------------------------

def test_password_rules():
    for bad in ("short", "x" * 129, "kanri12345"):
        with pytest.raises(auth.BadPassword):
            auth.check_password(bad, "kanri12345")
    auth.check_password("ten chars!", "kanri")


def test_login_names_are_plain_and_case_insensitive(env):
    _, repo = env
    for bad in ("", "ab", "a b c", "名前です", "x" * 33):
        with pytest.raises(auth.BadLogin):
            auth.normalize_login(bad)
    assert auth.normalize_login("  Yamada.T ") == "yamada.t"
    admin(repo, "Yamada")
    with pytest.raises(auth.BadLogin):
        auth.create_user(repo, "yamada", "別の人", "viewer", PW)


def test_a_user_is_stored_with_an_argon2_hash_never_the_password(env):
    _, repo = env
    user = admin(repo)
    row = repo.conn.execute("SELECT * FROM users WHERE id=?", (user.id,)).fetchone()
    assert row["password_hash"].startswith("$argon2id$") and PW not in row["password_hash"]
    assert (row["org_id"], row["role"], row["active"]) == (1, "admin", 1)
    assert user.label == "管理 一郎（管理者）"
    assert PW not in str(list(repo.conn.execute("SELECT * FROM audit_log")))
    assert audits(repo, "user_created")[0][0] == f"user:{user.id}"


def test_a_display_name_is_needed(env):
    _, repo = env
    with pytest.raises(auth.BadName):
        auth.create_user(repo, "someone", "  ", "viewer", PW)


# --- signing in ------------------------------------------------------------------------------

def test_login_with_the_right_password_and_a_wrong_one(env):
    _, repo = env
    user = admin(repo, "Kanri")
    assert auth.login(repo, "KANRI", PW, now=T0).id == user.id
    with pytest.raises(auth.LoginRefused) as wrong:
        auth.login(repo, "kanri", "wrong password", now=T0)
    with pytest.raises(auth.LoginRefused) as unknown:
        auth.login(repo, "nobody", PW, now=T0)
    assert str(wrong.value) == str(unknown.value)   # no hint whether the name exists
    assert auth.get_user(repo, user.id).last_login_at


def test_a_switched_off_user_cannot_sign_in(env):
    _, repo = env
    admin(repo)
    viewer = auth.create_user(repo, "miru", "見る 人", "viewer", PW)
    auth.set_active(repo, viewer.id, False)
    with pytest.raises(auth.LoginRefused):
        auth.login(repo, "miru", PW, now=T0)


def test_five_wrong_passwords_lock_the_name_for_15_minutes(env):
    _, repo = env
    admin(repo)
    for _ in range(auth.LOGIN_LIMIT):
        with pytest.raises(auth.LoginRefused):
            auth.login(repo, "kanri", "wrong password", now=T0)
    with pytest.raises(auth.TooManyTries):
        auth.login(repo, "kanri", PW, now=T0 + 60)       # even the right one waits
    assert auth.login(repo, "kanri", PW, now=T0 + auth.WINDOW_SEC + 1)
    assert len(audits(repo, "login_locked")) == 1


def test_an_address_is_limited_across_names(env):
    _, repo = env
    admin(repo)
    for i in range(auth.IP_LIMIT):
        with pytest.raises(auth.LoginRefused):
            auth.login(repo, f"guess{i}", "wrong password", ip="100.64.0.9", now=T0)
    with pytest.raises(auth.TooManyTries):
        auth.login(repo, "kanri", PW, ip="100.64.0.9", now=T0)
    assert auth.login(repo, "kanri", PW, ip="100.64.0.10", now=T0)


def test_a_success_clears_the_name_s_failures(env):
    _, repo = env
    admin(repo)
    for _ in range(auth.LOGIN_LIMIT - 1):
        with pytest.raises(auth.LoginRefused):
            auth.login(repo, "kanri", "wrong password", now=T0)
    auth.login(repo, "kanri", PW, now=T0)
    with pytest.raises(auth.LoginRefused):
        auth.login(repo, "kanri", "wrong password", now=T0)   # counts from zero again


# --- sessions --------------------------------------------------------------------------------

def test_a_session_cookie_round_trip(env):
    settings, repo = env
    user = admin(repo)
    cookie = auth.start_session(repo, settings, user, now=T0)
    session = auth.check_cookie(repo, settings, cookie, now=T0 + 10)
    assert session.user.id == user.id and session.id
    stored = repo.conn.execute("SELECT secret_hash FROM auth_sessions").fetchone()[0]
    assert cookie.split(".")[3] not in stored and len(stored) == 64
    assert stat.S_IMODE(os.stat(settings.data_dir / auth.SECRET_FILE).st_mode) == 0o600


def test_forged_expired_and_foreign_cookies_count_for_nothing(env, tmp_path):
    settings, repo = env
    user = admin(repo)
    cookie = auth.start_session(repo, settings, user, now=T0)
    v, sid, exp, rand, sig = cookie.split(".")
    for bad in ("", "garbage", f"{v}.{sid}.{int(exp) + 999}.{rand}.{sig}",
                f"{v}.{int(sid) + 1}.{exp}.{rand}.{sig}", f"{v}.{sid}.{exp}.{rand}x.{sig}",
                cookie + "x", None):
        assert auth.check_cookie(repo, settings, bad, now=T0) is None, bad
    assert auth.check_cookie(repo, settings, cookie,
                             now=T0 + settings.session_days * 86400 + 1) is None
    other = Settings(data_dir=tmp_path / "other")   # another install's key
    other.data_dir.mkdir()
    assert auth.check_cookie(repo, other, cookie, now=T0) is None


def test_sign_out_password_change_role_change_and_switch_off_end_sessions(env):
    settings, repo = env
    boss = admin(repo)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)

    def fresh():
        return auth.start_session(repo, settings, helper, now=T0)

    one = fresh()
    auth.revoke_session(repo, auth.check_cookie(repo, settings, one, now=T0).id)
    assert auth.check_cookie(repo, settings, one, now=T0) is None
    a, b = fresh(), fresh()
    auth.change_password(repo, helper.id, PW, "another long secret")
    assert auth.check_cookie(repo, settings, a, now=T0) is None
    assert auth.check_cookie(repo, settings, b, now=T0) is None
    with pytest.raises(auth.LoginRefused):
        auth.change_password(repo, helper.id, "not the password", "yet another secret")
    c = fresh()
    auth.set_role(repo, helper.id, "viewer")
    assert auth.check_cookie(repo, settings, c, now=T0) is None
    d = fresh()
    auth.set_active(repo, helper.id, False)
    assert auth.check_cookie(repo, settings, d, now=T0) is None
    assert auth.check_cookie(repo, settings, auth.start_session(repo, settings, boss, now=T0),
                             now=T0) is not None


def test_session_user_by_id_for_the_upload_endpoint(env):
    settings, repo = env
    user = admin(repo)
    cookie = auth.start_session(repo, settings, user, now=T0)
    sid = auth.check_cookie(repo, settings, cookie, now=T0).id
    assert auth.session_user(repo, sid, now=T0).id == user.id
    assert auth.session_user(repo, sid + 99, now=T0) is None
    assert auth.session_user(repo, sid, now=T0 + settings.session_days * 86400 + 1) is None
    auth.revoke_session(repo, sid)
    assert auth.session_user(repo, sid, now=T0) is None


def test_the_last_admin_stays_an_admin(env):
    _, repo = env
    boss = admin(repo)
    with pytest.raises(auth.LastAdmin):
        auth.set_role(repo, boss.id, "evaluator")
    with pytest.raises(auth.LastAdmin):
        auth.set_active(repo, boss.id, False)
    second = auth.create_user(repo, "kanri2", "管理 二郎", "admin", PW)
    auth.set_role(repo, boss.id, "evaluator")
    assert auth.get_user(repo, boss.id).role == "evaluator"
    with pytest.raises(auth.LastAdmin):
        auth.set_active(repo, second.id, False)


def test_role_changes_are_audited_with_the_actor(env):
    _, repo = env
    boss = admin(repo)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    repo.actor = (boss.id, boss.login)
    auth.set_role(repo, helper.id, "viewer")
    target, actor_id, detail = audits(repo, "role_changed")[0]
    assert (target, actor_id) == (f"user:{helper.id}", boss.id) and "viewer" in detail


# --- the first administrator ------------------------------------------------------------------

def test_the_setup_code_is_private_stable_and_made_once(env):
    settings, repo = env
    assert auth.setup_code(settings, create=False) is None
    code = auth.setup_code(settings)
    assert len(auth.normalize_code(code)) == 12 and code.count("-") == 2
    assert auth.setup_code(settings) == code
    path = settings.data_dir / auth.SETUP_FILE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_setup_with_the_code_makes_the_first_admin_and_removes_the_code(env):
    settings, repo = env
    code = auth.setup_code(settings)
    with pytest.raises(auth.CodeRefused):
        auth.setup_admin(repo, settings, "AAAA-BBBB-CCCC", "kanri", "管理 一郎", PW)
    assert auth.user_count(repo) == 0
    user = auth.setup_admin(repo, settings, code.lower().replace("-", " "), "kanri",
                            "管理 一郎", PW)
    assert user.role == "admin" and auth.user_count(repo) == 1
    assert not (settings.data_dir / auth.SETUP_FILE).exists()
    assert audits(repo, "setup_done")[0][1] == user.id
    with pytest.raises(auth.CodeRefused):   # once someone exists, setup is over
        auth.setup_admin(repo, settings, auth.setup_code(settings), "x2x", "二人目", PW)


def test_setup_tries_are_limited(env):
    settings, repo = env
    code = auth.setup_code(settings)
    for _ in range(auth.SETUP_LIMIT):
        with pytest.raises(auth.CodeRefused):
            auth.setup_admin(repo, settings, "AAAA-BBBB-CCCC", "kanri", "管理", PW,
                             ip="100.64.0.9", now=T0)
    with pytest.raises(auth.TooManyTries):
        auth.setup_admin(repo, settings, code, "kanri", "管理", PW, ip="100.64.0.9", now=T0)


def test_a_cli_made_admin_also_ends_the_setup(env):
    settings, repo = env
    auth.setup_code(settings)
    auth.create_first_admin(repo, settings, "kanri", "管理 一郎", PW)
    assert not (settings.data_dir / auth.SETUP_FILE).exists()


# --- invitations and resets -------------------------------------------------------------------

def test_an_invitation_makes_one_account_with_its_role(env):
    _, repo = env
    boss = admin(repo)
    repo.actor = (boss.id, boss.login)
    code = auth.create_invite(repo, "evaluator", "佐藤 花子", by=boss.id, now=T0)
    stored = repo.conn.execute("SELECT code_hash FROM invites").fetchone()[0]
    assert auth.normalize_code(code) not in stored
    found = auth.peek_code(repo, code, now=T0)
    assert (found.role, found.display_name, found.user_id) == ("evaluator", "佐藤 花子", None)
    repo.actor = None
    user = auth.redeem_invite(repo, code, "sato", "佐藤 花子", PW, now=T0)
    assert user.role == "evaluator"
    with pytest.raises(auth.CodeRefused):
        auth.redeem_invite(repo, code, "sato2", "佐藤 二", PW, now=T0)   # once only
    assert "evaluator" in audits(repo, "invite_created")[0][2]
    assert code not in str(list(repo.conn.execute("SELECT * FROM audit_log")))


def test_expired_and_cancelled_codes_say_only_that_they_cannot_be_used(env):
    _, repo = env
    boss = admin(repo)
    old = auth.create_invite(repo, "viewer", "", by=boss.id, now=T0)
    with pytest.raises(auth.CodeRefused):
        auth.peek_code(repo, old, now=T0 + auth.INVITE_DAYS * 86400 + 1)
    gone = auth.create_invite(repo, "viewer", "消す 人", by=boss.id, now=T0)
    [invite] = [i for i in auth.open_invites(repo, now=T0) if i["display_name"] == "消す 人"]
    auth.cancel_invite(repo, invite["id"])
    assert all(i["id"] != invite["id"] for i in auth.open_invites(repo, now=T0))
    with pytest.raises(auth.CodeRefused) as e:
        auth.peek_code(repo, gone, now=T0)
    assert str(e.value) == auth.CODE_REFUSED


def test_code_tries_are_limited_per_address(env):
    _, repo = env
    admin(repo)
    for _ in range(auth.CODE_LIMIT):
        with pytest.raises(auth.CodeRefused):
            auth.peek_code(repo, auth.make_code(), ip="100.64.0.9", now=T0)
    with pytest.raises(auth.TooManyTries):
        auth.peek_code(repo, auth.make_code(), ip="100.64.0.9", now=T0)


def test_a_reset_code_sets_a_new_password_and_ends_sessions(env):
    settings, repo = env
    boss = admin(repo)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    cookie = auth.start_session(repo, settings, helper, now=T0)
    code = auth.create_reset(repo, helper.id, by=boss.id, now=T0)
    assert auth.peek_code(repo, code, now=T0).user_id == helper.id
    auth.redeem_reset(repo, code, "a brand new secret", now=T0)
    assert auth.check_cookie(repo, settings, cookie, now=T0) is None
    assert auth.login(repo, "tasuke", "a brand new secret", now=T0).id == helper.id
    with pytest.raises(auth.CodeRefused):
        auth.redeem_reset(repo, code, "once more secret", now=T0)
    with pytest.raises(auth.CodeRefused):   # an invitation is not a reset, and back
        auth.redeem_reset(repo, auth.create_invite(repo, "viewer", "", by=boss.id, now=T0),
                          "a brand new secret", now=T0)
