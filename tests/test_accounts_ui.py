"""The sign-in gate of ui/app.py (AppTest, the real gate): the setup screen while nobody is
registered (never the data), sign-in and sign-out, the cookie, codes, and what each role
sees. Synthetic people and works only."""
import logging

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_wizard_ui_basic import UI, html, ok, screen_text

pytestmark = pytest.mark.real_auth
PW = "correct horse battery"
WORK = "練習試合カードの題"


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3")
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir)}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    repo.create_assignment(WORK, "document", "調べてまとめる", "観点A", "standard", False)
    return settings, repo


def app(token: str | None = None, cookie: str | None = None, monkeypatch=None,
        session: dict | None = None) -> AppTest:
    if monkeypatch is not None:
        from pfb_app.ui import auth_ui
        monkeypatch.setattr(auth_ui, "_browser_cookie", lambda: cookie)
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    if token:
        from pfb_app.ui import auth_ui
        at.session_state[auth_ui.TOKEN] = token
    for key, value in (session or {}).items():
        at.session_state[key] = value
    return at.run()


def signed_in(env, role="admin", login="kanri", name="管理 一郎"):
    settings, repo = env
    user = auth.create_user(repo, login, name, role, PW)
    return user, auth.start_session(repo, settings, user)


def fill(at: AppTest, values: dict) -> None:
    for key, value in values.items():
        at.text_input(key=key).input(value)


def press(at: AppTest, key: str) -> AppTest:
    return at.button(key=key).click().run()


def everything(at: AppTest) -> str:
    return screen_text(at) + html(at) + "".join(p.proto.label for p in at.get("page_link"))


# --- nobody registered yet ------------------------------------------------------------------

def test_with_nobody_registered_the_setup_screen_shows_and_never_the_data(env, caplog):
    settings, _ = env
    with caplog.at_level(logging.INFO):
        at = ok(app())
    text = everything(at)
    assert "はじめに、管理者を登録します" in text
    assert WORK not in text and "ホーム" not in [b.label for b in at.button]
    code = auth.setup_code(settings, create=False)
    assert code and code not in caplog.text and code not in text
    assert "pfb-admin setup-code" in caplog.text
    assert [b.label for b in at.button if b.proto.type == "primary"] == ["管理者を登録して始める"]


def test_a_wrong_setup_code_is_refused(env):
    settings, repo = env
    at = app()
    fill(at, {"pfb_setup_code": "AAAA-BBBB-CCCC", "pfb_setup_name": "管理 一郎",
              "pfb_setup_login": "kanri", "pfb_setup_pw": PW, "pfb_setup_pw2": PW})
    at = ok(press(at, "pfb_setup_go"))
    assert auth.CODE_REFUSED in screen_text(at) and auth.user_count(repo) == 0
    assert WORK not in everything(at)


def test_the_setup_code_makes_the_first_admin_and_opens_the_app(env, monkeypatch):
    from pfb_app.ui import auth_api
    monkeypatch.setattr(auth_api, "MOUNTED", True)
    settings, repo = env
    at = app()
    code = auth.setup_code(settings, create=False)
    fill(at, {"pfb_setup_code": code.lower(), "pfb_setup_name": "管理 一郎",
              "pfb_setup_login": "kanri", "pfb_setup_pw": PW, "pfb_setup_pw2": PW})
    at = ok(press(at, "pfb_setup_go"))
    assert WORK in everything(at)
    assert auth.find_user(repo, "kanri").role == "admin"
    assert auth_api.COOKIE_PATH in html(at)     # the server sets the browser's cookie
    assert "管理 一郎（管理者）" in everything(at)


def test_two_different_passwords_are_refused_on_setup(env):
    settings, repo = env
    at = app()
    fill(at, {"pfb_setup_code": auth.setup_code(settings), "pfb_setup_name": "管理",
              "pfb_setup_login": "kanri", "pfb_setup_pw": PW, "pfb_setup_pw2": PW + "x"})
    at = ok(press(at, "pfb_setup_go"))
    assert "2回のパスワードがちがいます" in screen_text(at) and auth.user_count(repo) == 0


# --- signing in and out --------------------------------------------------------------------

def test_the_login_screen_refuses_a_wrong_password_and_lets_the_right_one_in(env):
    signed_in(env)
    at = app()
    assert "ログイン" in screen_text(at) and WORK not in everything(at)
    fill(at, {"pfb_login_name": "kanri", "pfb_login_pw": "wrong password"})
    at = ok(press(at, "pfb_login_go"))
    assert auth.LOGIN_REFUSED in screen_text(at) and WORK not in everything(at)
    fill(at, {"pfb_login_name": "kanri", "pfb_login_pw": PW})
    at = ok(press(at, "pfb_login_go"))
    assert WORK in everything(at)


def test_too_many_tries_say_to_wait(env):
    signed_in(env)
    at = app()
    for _ in range(auth.LOGIN_LIMIT + 1):
        fill(at, {"pfb_login_name": "kanri", "pfb_login_pw": "wrong password"})
        at = press(at, "pfb_login_go")
    assert auth.TOO_MANY in screen_text(ok(at))


def test_a_valid_browser_cookie_signs_in_and_a_revoked_one_does_not(env, monkeypatch):
    from pfb_app.ui import auth_api
    monkeypatch.setattr(auth_api, "MOUNTED", True)
    settings, repo = env
    _, cookie = signed_in(env)
    at = ok(app(cookie=cookie, monkeypatch=monkeypatch))
    assert WORK in everything(at) and auth_api.COOKIE_PATH not in html(at)   # already kept
    auth.revoke_session(repo, auth.check_cookie(repo, settings, cookie).id)
    at = ok(app(cookie=cookie, monkeypatch=monkeypatch))
    assert WORK not in everything(at)
    assert auth_api.CLEAR_PATH in html(at)       # the stale cookie is cleared


def test_sign_out_ends_the_session(env):
    settings, repo = env
    _, cookie = signed_in(env)
    at = ok(app(token=cookie))
    assert WORK in everything(at)
    at = ok(press(at, "pfb_logout"))
    assert WORK not in everything(at) and "ログイン" in screen_text(at)
    assert auth.check_cookie(repo, settings, cookie) is None


def test_a_password_change_keeps_this_browser_and_ends_the_others(env):
    settings, repo = env
    user, cookie = signed_in(env)
    other = auth.start_session(repo, settings, user)
    at = ok(app(token=cookie))
    fill(at, {"pfb_pw_old": PW, "pfb_pw_new": "another long secret",
              "pfb_pw_new2": "another long secret"})
    at = ok(press(at, "pfb_pw_go"))
    assert WORK in everything(at)
    assert auth.check_cookie(repo, settings, other) is None
    assert auth.check_cookie(repo, settings, cookie) is None   # replaced by a fresh one
    assert auth.login(repo, "kanri", "another long secret")


def test_a_session_revoked_meanwhile_ends_the_open_tab(env):
    settings, repo = env
    user, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    signed_in(env)
    at = ok(app(token=cookie))
    assert WORK in everything(at)
    auth.set_role(repo, user.id, "viewer")
    at = ok(at.run())
    assert WORK not in everything(at) and "ログイン" in screen_text(at)


# --- codes ----------------------------------------------------------------------------------

def test_an_invitation_code_registers_a_new_person(env):
    settings, repo = env
    boss, _ = signed_in(env)
    code = auth.create_invite(repo, "evaluator", "佐藤 花子", by=boss.id)
    at = app()
    fill(at, {"pfb_code": code})
    at = ok(press(at, "pfb_code_check"))
    assert at.text_input(key="pfb_code_name").value == "佐藤 花子"
    fill(at, {"pfb_code_login": "sato", "pfb_code_pw": PW, "pfb_code_pw2": PW})
    at = ok(press(at, "pfb_code_go"))
    assert WORK in everything(at) and "佐藤 花子（評価する人）" in everything(at)
    assert auth.find_user(repo, "sato").role == "evaluator"


def test_a_reset_code_sets_a_new_password(env):
    settings, repo = env
    boss, _ = signed_in(env)
    helper = auth.create_user(repo, "tasuke", "手伝う 人", "evaluator", PW)
    code = auth.create_reset(repo, helper.id, by=boss.id)
    at = app()
    fill(at, {"pfb_code": code})
    at = ok(press(at, "pfb_code_check"))
    fill(at, {"pfb_code_pw": "a brand new secret", "pfb_code_pw2": "a brand new secret"})
    at = ok(press(at, "pfb_code_go"))
    assert WORK in everything(at)
    assert auth.login(repo, "tasuke", "a brand new secret").id == helper.id


def test_a_wrong_code_says_only_that_it_cannot_be_used(env):
    signed_in(env)
    at = app()
    fill(at, {"pfb_code": "AAAA-BBBB-CCCC"})
    at = ok(press(at, "pfb_code_check"))
    assert auth.CODE_REFUSED in screen_text(at)


# --- what each role sees --------------------------------------------------------------------

def links(at: AppTest) -> list[str]:
    return [p.proto.label for p in at.get("page_link")]


def test_an_admin_sees_the_admin_section(env):
    _, cookie = signed_in(env)
    at = ok(app(token=cookie))
    assert {"データの扱い", "利用する人", "処理状況"} <= set(links(at))


def test_an_evaluator_has_no_admin_section(env):
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(app(token=cookie))
    assert not {"データの扱い", "利用する人", "処理状況", "活動"} & set(links(at))
    assert "管理用の画面（くわしい操作）" not in html(at)
    assert "成長の記録" in links(at)


def test_a_viewer_sees_works_but_nothing_to_change(env):
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = ok(app(token=cookie))
    text = everything(at)
    assert WORK in text and "見る 人（見るだけの人）" in text
    labels = [b.label for b in at.button]
    assert "新しく始める" not in labels and "見る" in labels
    assert "成長の記録" not in links(at) and "データの扱い" not in links(at)


def test_a_viewer_s_write_is_refused_with_plain_words(env, monkeypatch):
    from pfb_app.ui.common import get_repo
    from pfb_app.ui.wizard import router

    def writes():
        get_repo().rename_assignment(1, "変えた名前")
    monkeypatch.setattr(router, "render", writes)
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = ok(app(token=cookie))
    assert "見るだけの人は、この操作はできません。" in screen_text(at)
    assert env[1].get_assignment(1)["title"] == WORK


def test_the_actor_is_recorded_from_the_signed_in_person(env, monkeypatch):
    from pfb_app.ui.common import get_repo
    from pfb_app.ui.wizard import router

    def writes():
        get_repo().audit("rename", "assignment:1", {})
    monkeypatch.setattr(router, "render", writes)
    user, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    ok(app(token=cookie))
    row = env[1].conn.execute("SELECT actor_id, actor FROM audit_log WHERE action='rename'"
                              ).fetchone()
    assert tuple(row) == (user.id, "tasuke")


def test_a_viewer_opens_a_work_read_only(env):
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = ok(app(token=cookie, session={"wizard": {"assignment_id": 1, "step": None,
                                                  "submission_id": None}}))
    text = everything(at)
    assert "見るだけの人として開いています" in text and WORK in text
    labels = [b.label for b in at.button]
    assert not {"承認する", "これで決定して次へ", "登録して解析を始める"} & set(labels)


def test_a_viewer_cannot_start_a_new_work_through_the_address(env):
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    from pfb_app.ui import auth_ui
    at.session_state[auth_ui.TOKEN] = cookie
    at.query_params["work"] = "new"
    at = ok(at.run())
    assert "入れて確かめる" not in [b.label for b in at.button]
    assert WORK in everything(at)


def test_only_an_admin_may_delete_a_work(env):
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(app(token=cookie))
    labels = [b.label for b in at.button]
    assert "この名前にする" in labels and "削除" not in labels


def test_a_viewer_reads_the_approved_texts_and_may_take_the_pdf(env):
    from test_neutral_wording import doc_work
    signed_in(env)
    aid = doc_work(env[1])
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = ok(app(token=cookie, session={"wizard": {"assignment_id": aid, "step": None,
                                                  "submission_id": None}}))
    text = everything(at)
    assert "山田太郎さん" in text and "鈴木一郎さん（" in text
    assert [b.label for b in at.download_button] == ["参加者に渡すPDFを受け取る"]
    assert env[1].get_assignment(aid)["pdf_downloaded_at"] is None


def test_the_drop_zone_s_upload_token_carries_the_sign_in(env, monkeypatch):
    from pfb_app.services import chunked_upload as cu
    from pfb_app.ui import upload_api
    from pfb_app.ui.wizard import big_upload
    settings, repo = env
    monkeypatch.setattr(upload_api, "MOUNTED", True)
    # the uploader component is registered once per process: register it in this test's
    # runtime and forget it afterwards (test_big_upload_ui registers it in its own)
    monkeypatch.setattr(big_upload, "_component", None)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    at = ok(app(token=cookie, session={"wizard": {"assignment_id": None, "step": 1,
                                                  "submission_id": None}}))
    token = at.session_state[big_upload.TOKEN]
    sid = auth.check_cookie(repo, settings, cookie).id
    assert cu.token_session(settings, token) == sid
    assert upload_api.may_upload(settings, sid)
    auth.revoke_session(repo, sid)
    assert not upload_api.may_upload(settings, sid)
