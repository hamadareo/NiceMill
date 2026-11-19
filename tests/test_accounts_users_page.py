"""「利用する人」 (pages/users.py, 管理者 only): invitations, roles, switching off, reset codes,
cancelling codes (AppTest; synthetic people only)."""
import pytest
from streamlit.testing.v1 import AppTest

from pfb_app.services import auth
from pfb_app.ui import access
from test_accounts_ui import PW, env, everything  # noqa: F401 (env)
from test_wizard_ui_basic import UI, ok, screen_text

pytestmark = pytest.mark.real_auth


def page(user: auth.User) -> AppTest:
    """The page as `user` signed in (a real session: the page re-checks it)."""
    at = AppTest.from_file(str(UI / "pages" / "users.py"), default_timeout=60)
    at.session_state[access.TOKEN] = SESSIONS[user.id]
    return at.run()


SESSIONS: dict[int, str] = {}


def people(env):
    _, repo = env
    settings, _ = env
    boss = auth.create_user(repo, "kanri", "管理 一郎", "admin", PW)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    for user in (boss, helper):
        SESSIONS[user.id] = auth.start_session(repo, settings, user)
    return boss, helper


def test_only_an_admin_gets_the_page(env):
    _, helper = people(env)
    at = ok(page(helper))
    assert access.VIEW_ONLY not in screen_text(at)
    # release 4: pages/users.py stops first (access.require_page), before users_view's own check
    assert access.NOT_HERE in screen_text(at)
    assert "tasuke" not in everything(at)


def test_the_page_lists_people_with_roles_and_one_primary(env):
    boss, helper = people(env)
    at = ok(page(boss))
    text = everything(at)
    assert "管理 一郎" in text and "手伝う 人" in text and "評価する人" in text
    assert "あなたです" in text
    assert [b.label for b in at.button if b.proto.type == "primary"] == ["招待コードを作る"]
    assert "argon2" not in text


def test_an_invitation_code_is_shown_once_and_recorded_with_the_actor(env):
    _, repo = env
    boss, _ = people(env)
    at = page(boss)
    at.radio(key="pfb_invite_role").set_value("viewer")
    at.text_input(key="pfb_invite_name").input("見る 人")
    at = ok(at.button(key="pfb_invite_go").click().run())
    [code_el] = at.code
    code = code_el.value
    assert auth.peek_code(repo, code).role == "viewer"
    assert "使われていないコード（1件）" in screen_text(at)
    row = repo.conn.execute("SELECT actor_id FROM audit_log WHERE action='invite_created'"
                            ).fetchone()
    assert row[0] == boss.id
    at = ok(at.run())
    assert not at.code                    # shown once only


def test_a_role_change_and_switching_off_end_the_person_s_sign_in(env):
    settings, repo = env
    boss, helper = people(env)
    cookie = auth.start_session(repo, settings, helper)
    at = page(boss)
    at = at.selectbox(key=f"pfb_user_role_{helper.id}").set_value("viewer").run()
    at = ok(at.button(key=f"pfb_user_role_go_{helper.id}").click().run())
    assert auth.get_user(repo, helper.id).role == "viewer"
    assert auth.check_cookie(repo, settings, cookie) is None
    at = ok(at.button(key=f"pfb_user_off_{helper.id}").click().run())
    assert not auth.get_user(repo, helper.id).active
    at = ok(at.button(key=f"pfb_user_on_{helper.id}").click().run())
    assert auth.get_user(repo, helper.id).active


def test_a_reset_code_is_made_and_can_be_cancelled(env):
    _, repo = env
    boss, helper = people(env)
    at = ok(page(boss).button(key=f"pfb_user_reset_{helper.id}").click().run())
    code = at.code[0].value
    assert auth.peek_code(repo, code).user_id == helper.id
    [open_code] = auth.open_invites(repo)
    at = ok(at.button(key=f"pfb_code_cancel_{open_code['id']}").click().run())
    assert auth.open_invites(repo) == []
    with pytest.raises(auth.CodeRefused):
        auth.peek_code(repo, code)
