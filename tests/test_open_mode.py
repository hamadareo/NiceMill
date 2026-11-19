"""ログインなしで使えるモード (open mode, ruling 2026-11-13).

With `auth_enabled: false` every browser opens the app as an administrator of role without
a sign-in (ui.auth_ui.OPEN_SESSION), uploads need only the page's own token, the sidebar
says so quietly, and the admin pages are reachable. Setting the flag back to true returns
the sign-in gate exactly as it was (the rel4 gate tests cover that side). Synthetic people
and works only."""
import os
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.services import chunked_upload as cu
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui import auth_ui, open_guard, upload_api
from test_wizard_ui_basic import UI, html, ok, screen_text

pytestmark = pytest.mark.real_auth      # the real gate: no synthetic admin bypass

PW = "correct horse battery"
WORK = "オープンモードの練習カード"


@pytest.fixture
def open_env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3", auth_enabled=False)
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir),
                                   "auth_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    repo.create_assignment(WORK, "document", "調べてまとめる", "観点A", "standard", False)
    auth.create_user(repo, "kanri", "管理 一郎", "admin", PW)   # its password is never asked
    return settings, repo


def open_app(monkeypatch, cookie: str | None = None) -> AppTest:
    # a browser on the tailnet, directly (the rule itself: tests/test_open_guard.py)
    monkeypatch.setattr(open_guard, "streamlit_view",
                        lambda: ("127.0.0.1", [("host", "127.0.0.1:8501")]))
    monkeypatch.setattr(auth_ui, "_browser_cookie", lambda: cookie)
    return AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()


def everything(at: AppTest) -> str:
    return screen_text(at) + html(at) + "".join(p.proto.label for p in at.get("page_link"))


def test_the_app_opens_without_a_sign_in(open_env, monkeypatch):
    at = ok(open_app(monkeypatch))
    text = everything(at)
    assert WORK in text                                  # the app itself, not a gate
    assert "はじめに、管理者を登録します" not in text
    assert [b for b in at.button if b.key == "pfb_login_go"] == []


def test_the_sidebar_says_it_is_open_and_the_admin_pages_are_offered(open_env, monkeypatch):
    at = ok(open_app(monkeypatch))
    assert auth_ui.OPEN_NOTE in screen_text(at)
    labels = [p.proto.label for p in at.get("page_link")]
    assert "データの扱い" in labels and "利用する人" in labels


def test_an_open_upload_token_is_enough(open_env):
    settings, _ = open_env
    token = cu.issue_token(settings)                     # no session to carry
    assert cu.check_token(settings, token) is not None
    assert cu.token_session(settings, token) is None
    assert upload_api.may_upload(settings, cu.token_session(settings, token))
    closed = settings.model_copy(update={"auth_enabled": True})
    assert not upload_api.may_upload(closed, None)       # the gate's rule is untouched


def test_the_flag_back_on_returns_the_gate(open_env, monkeypatch):
    at = ok(open_app(monkeypatch))
    assert WORK in everything(at)
    cfg = Path(os.environ["PFB_CONFIG"])
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["auth_enabled"] = True
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    at = ok(open_app(monkeypatch))
    assert "ログイン" in screen_text(at) and WORK not in everything(at)
    assert auth_ui.OPEN_NOTE not in screen_text(at)
