"""`uv run pfb-admin ...` (pfb_app.admin_cli): the operator's way to the first administrator
and out of a lock-out (synthetic people only)."""
import io
import sys

import pytest
import yaml

from pfb_app import admin_cli
from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

PW = "correct horse battery"


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir)}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    return settings


def repo_of(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def run(args, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    return admin_cli.main(args)


def test_setup_code_prints_the_code_the_setup_screen_wants(env, capsys):
    assert run(["setup-code"]) == 0
    out = capsys.readouterr().out
    code = auth.setup_code(env, create=False)
    assert code and code in out


def test_create_admin_without_a_login_shows_the_setup_code_too(env, capsys):
    assert run(["create-admin"]) == 0
    assert auth.setup_code(env, create=False) in capsys.readouterr().out


def test_create_admin_with_a_login_makes_the_admin_and_ends_the_setup(env, capsys,
                                                                     monkeypatch):
    run(["setup-code"])
    assert run(["create-admin", "--login", "kanri", "--name", "管理 一郎",
                "--password-stdin"], stdin=PW + "\n", monkeypatch=monkeypatch) == 0
    repo = repo_of(env)
    user = auth.find_user(repo, "kanri")
    assert user.role == "admin" and auth.login(repo, "kanri", PW)
    assert auth.setup_code(env, create=False) is None
    assert PW not in capsys.readouterr().out
    assert run(["setup-code"]) == 1          # nobody needs the code any more


def test_a_bad_password_is_refused_with_plain_words(env, capsys, monkeypatch):
    assert run(["create-admin", "--login", "kanri", "--password-stdin"], stdin="short\n",
               monkeypatch=monkeypatch) == 2
    assert "10文字以上" in capsys.readouterr().err
    assert auth.user_count(repo_of(env)) == 0


def test_reset_password_ends_sessions_and_users_lists_people(env, capsys, monkeypatch):
    repo = repo_of(env)
    user = auth.create_user(repo, "kanri", "管理 一郎", "admin", PW)
    cookie = auth.start_session(repo, env, user)
    assert run(["reset-password", "--login", "kanri", "--password-stdin"],
               stdin="a new long secret\n", monkeypatch=monkeypatch) == 0
    assert auth.check_cookie(repo, env, cookie) is None
    assert auth.login(repo, "kanri", "a new long secret")
    assert run(["reset-password", "--login", "nobody", "--password-stdin"],
               stdin="a new long secret\n", monkeypatch=monkeypatch) == 2
    capsys.readouterr()
    assert run(["users"]) == 0
    out = capsys.readouterr().out
    assert "kanri" in out and "管理者" in out and "argon2" not in out
