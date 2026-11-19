"""Release 4 (accounts + dashboard): 試合・練習のまとめ (/dashboard) follows the roles of the
accounts lane. It is registered only for 管理者 and 評価する人 (a 見るだけの人 opening the address
gets Streamlit's page-not-found and the home screen), the page itself refuses a run without
a valid editing session, and no clip file is cut for a revoked session. Synthetic people and
works only (pdf_synthetic's generated videos)."""
import pytest
from streamlit.testing.v1 import AppTest

from pdf_synthetic import video_group, work
from pfb_app.services import auth
from pfb_app.storage import files
from test_accounts_ui import env, signed_in  # noqa: F401 (env)
from test_neutral_wording import everything
from test_wizard_ui_basic import UI, ok, screen_text


def screen_lines(at: AppTest) -> list[str]:
    return [line for line in screen_text(at).splitlines() if line.strip()]

pytestmark = pytest.mark.real_auth
PAGE = "pages/dashboard.py"
JUMP = "この場面から見る"
CSV_PEOPLE = "人ごとの表をCSVで受け取る"


def video_work(env) -> int:
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=40)
    return aid


def as_person(cookie: str, path: str = "app.py") -> AppTest:
    from pfb_app.ui import auth_ui
    at = AppTest.from_file(str(UI / path), default_timeout=90)
    at.session_state[auth_ui.TOKEN] = cookie
    return at


def registered(at: AppTest) -> set[str]:
    return {str(info.get("url_pathname")) for info in at._registered_pages.values()}


def dashboard_hash(at: AppTest) -> str:
    return next(h for h, info in at._registered_pages.items()
                if info.get("url_pathname") == "dashboard")


@pytest.mark.parametrize("role", ["admin", "evaluator"])
def test_admins_and_evaluators_open_the_dashboard(env, role):
    aid = video_work(env)
    signed_in(env)
    _, cookie = signed_in(env, role=role, login="tasuke2", name="手伝う 人")
    at = ok(as_person(cookie).run())
    assert "dashboard" in registered(at)
    at.query_params["work"] = str(aid)
    at = ok(at.switch_page(PAGE).run())
    text = everything(at)
    assert "試合・練習のまとめ" in text
    assert CSV_PEOPLE in [b.label for b in at.download_button]


def as_the_server_entry(monkeypatch):
    """What ui/serve.py makes of every run: never Streamlit's pages-folder mode (AppTest
    resets that choice before each run, so it is made again after the reset)."""
    from streamlit.runtime.pages_manager import PagesManager

    from pfb_app.ui import access
    real = PagesManager.__init__

    def init(self, *a, **k):
        access.no_pages_folder_mode()
        real(self, *a, **k)
    monkeypatch.setattr(PagesManager, "__init__", init)


def test_a_viewer_opening_the_dashboard_address_gets_the_home_screen(env, monkeypatch):
    as_the_server_entry(monkeypatch)
    aid = video_work(env)
    _, boss = signed_in(env)
    # the page hash a browser would ask for at /dashboard (st.Page hashes its url_path)
    known = dashboard_hash(ok(as_person(boss).run()))
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = ok(as_person(cookie).run())
    assert "dashboard" not in registered(at)
    with pytest.raises(ValueError):
        at.switch_page(PAGE)                          # not a page of this viewer's app
    at.query_params["work"] = str(aid)
    at._page_hash = known                             # as typing /dashboard?work=N
    at = ok(at.run())
    text = everything(at)
    assert "試合・練習のまとめ" not in text and CSV_PEOPLE not in text
    # page not found: the default page instead, i.e. the viewer's own read-only view of the
    # work (?work=N is also the wizard's address of the open work)
    assert "見るだけの人として開いています" in text
    assert [b.label for b in at.download_button] == ["選手に渡すPDFを受け取る"]


def test_the_dashboard_page_refuses_a_viewer_run_directly(env):
    from pfb_app.ui import access
    aid = video_work(env)
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = as_person(cookie, PAGE)
    at.query_params["work"] = str(aid)
    at = ok(at.run())
    text = everything(at)
    assert access.NOT_HERE in text
    assert CSV_PEOPLE not in text and not at.download_button and not at.dataframe


ADMIN_PAGES = ["assignments", "submissions", "status", "review", "export", "data", "users"]
EDIT_PAGES = ["growth", "dashboard", "ask", "linecall_live", "linecall_review"]
ANY_PAGES = ["guide", "sample", "quality"]


def test_every_page_file_is_listed_here():
    files = {p.stem for p in (UI / "pages").glob("*.py") if p.stem != "__init__"}
    assert files == set(ADMIN_PAGES + EDIT_PAGES + ANY_PAGES)


@pytest.mark.parametrize("name", ADMIN_PAGES + EDIT_PAGES + ANY_PAGES)
def test_a_page_file_run_without_a_sign_in_shows_nothing(env, name):
    from pfb_app.ui import access
    video_work(env)
    signed_in(env)
    at = ok(AppTest.from_file(str(UI / "pages" / f"{name}.py"), default_timeout=90).run())
    assert screen_lines(at) == [access.NOT_HERE]
    assert not at.dataframe and not at.download_button and not at.button


@pytest.mark.parametrize("name", ADMIN_PAGES + EDIT_PAGES + ANY_PAGES)
def test_a_page_file_follows_the_viewer_s_role(env, name):
    from pfb_app.ui import access
    video_work(env)
    signed_in(env)
    _, cookie = signed_in(env, role="viewer", login="miru", name="見る 人")
    at = as_person(cookie, f"pages/{name}.py").run()
    if name in ANY_PAGES:   # (their links to other pages need app.py's navigation)
        assert access.NOT_HERE not in screen_text(at)
    else:
        assert screen_lines(ok(at)) == [access.NOT_HERE]


@pytest.mark.parametrize("name", ["users", "growth", "dashboard"])
def test_a_fresh_server_s_first_run_cannot_skip_the_gate(env, name):
    """A new server process starts in Streamlit's pages-folder mode until st.navigation is
    first called: the first run asking for /users ran pages/users.py without app.py. The
    page's own check stops it (and ui/serve.py turns the mode off)."""
    from pfb_app.ui import access
    aid = video_work(env)
    _, boss = signed_in(env)
    known = next(h for h, info in ok(as_person(boss).run())._registered_pages.items()
                 if info.get("url_pathname") == name)
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=90)   # a fresh process' state
    at.query_params["work"] = str(aid)
    at._page_hash = known
    at = ok(at.run())
    text = everything(at)
    assert access.NOT_HERE in text or "ログイン" in text
    assert not at.dataframe and not at.download_button and "練習試合" not in text


def test_the_server_entry_turns_the_pages_folder_mode_off(monkeypatch):
    from streamlit.runtime.pages_manager import PagesManager

    from pfb_app.ui import access
    monkeypatch.setattr(PagesManager, "uses_pages_directory", True)
    access.no_pages_folder_mode()
    assert PagesManager.uses_pages_directory is False
    source = (UI / "serve.py").read_text(encoding="utf-8")
    assert source.index("access.no_pages_folder_mode()") < source.index("st.App(")


def test_a_revoked_session_cuts_no_clip(env, monkeypatch):
    from pfb_app.services import moment_clip
    settings, repo = env
    aid = video_work(env)
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    cut = []
    real = moment_clip.clip
    monkeypatch.setattr(moment_clip, "clip", lambda *a, **k: cut.append(a) or real(*a, **k))
    at = as_person(cookie, PAGE)
    at.query_params["work"] = str(aid)
    at = ok(at.run())
    jump = next(b for b in at.button if b.label == JUMP)
    auth.revoke_session(repo, auth.check_cookie(repo, settings, cookie).id)
    at = ok(jump.click().run())
    assert cut == []
    assert not at.get("video")
    clips = moment_clip.clips_dir(settings, aid, repo.list_submissions(aid)[0]["id"])
    assert not clips.exists() or not any(clips.iterdir())


def test_a_session_revoked_between_the_page_and_the_player_cuts_no_clip(env, monkeypatch):
    """The player checks again right before cutting (a revocation within the same run)."""
    from pfb_app.services import moment_clip
    from pfb_app.ui.wizard import dashboard
    settings, repo = env
    aid = video_work(env)
    signed_in(env)
    _, cookie = signed_in(env, role="evaluator", login="tasuke", name="手伝う 人")
    cut = []
    monkeypatch.setattr(moment_clip, "clip", lambda *a, **k: cut.append(a))
    at = as_person(cookie, PAGE)
    at.query_params["work"] = str(aid)
    at = ok(at.run())
    real_source = moment_clip.source

    def revoked_meanwhile(*a, **k):
        from pfb_app.storage.db import Repo, connect
        mine = Repo(connect(files.db_path(settings)), settings)   # the page's thread
        auth.revoke_session(mine, auth.check_cookie(mine, settings, cookie).id)
        return real_source(*a, **k)
    monkeypatch.setattr(moment_clip, "source", revoked_meanwhile)
    at = ok(next(b for b in at.button if b.label == JUMP).click().run())
    assert cut == []
    assert dashboard.NOT_ALLOWED in everything(at)


def test_the_entry_link_is_offered_only_to_editors(env):
    aid = video_work(env)
    signed_in(env)
    _, viewer = signed_in(env, role="viewer", login="miru", name="見る 人")

    def entry(aid):
        from pfb_app.ui.common import get_repo
        from pfb_app.ui.wizard import dashboard
        dashboard.entry_link(get_repo().get_assignment(aid))
    from pfb_app.ui import auth_ui
    at = AppTest.from_function(entry, args=(aid,), default_timeout=60)
    at.session_state[auth_ui.TOKEN] = viewer
    at = ok(at.run())
    assert not at.get("page_link")
