"""The operator's account commands on the server (docs/accounts.md):

    uv run pfb-admin setup-code                     # the one-time code of the setup screen
    uv run pfb-admin create-admin                   # the same, while nobody is registered
    uv run pfb-admin create-admin --login NAME [--name 表示名] [--password-stdin]
    uv run pfb-admin reset-password --login NAME [--password-stdin]
    uv run pfb-admin users
    uv run pfb-admin sign-out-all                   # e.g. after restoring an old backup

Settings: --settings PATH, else the YAML named by PFB_CONFIG (as the screens read it), else the
defaults. Passwords are asked twice without echo (or read as one line from stdin) and never
printed. Exit codes: 0 done, 1 nothing to do, 2 refused.
"""
import argparse
import getpass
import os
import sys
from pathlib import Path

from pfb_app.config import Settings, load_settings
from pfb_app.services import auth
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


def _settings(path: str | None) -> Settings:
    path = path or os.environ.get("PFB_CONFIG")
    return load_settings(Path(path)) if path else Settings()


def _repo(settings: Settings) -> Repo:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def _password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("新しいパスワード: ")
    if getpass.getpass("もう一度: ") != first:
        raise auth.BadPassword("2回のパスワードがちがいます。")
    return first


def _setup_code(settings: Settings, repo: Repo) -> int:
    if auth.user_count(repo):
        print("もう利用する人が登録されています。確認コードはいりません。"
              "パスワードを忘れたときは reset-password を使ってください。")
        return 1
    code = auth.setup_code(settings, renew=True)     # an expired code is replaced
    print(f"最初の管理者を登録する確認コード（一度だけ、{auth.SETUP_CODE_HOURS}時間使えます）:")
    print(f"  {code}")
    print("アプリを開くと出る「はじめに、管理者を登録します」の画面に入れてください。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pfb-admin", description="利用する人の管理（サーバーで）")
    parser.add_argument("--settings", help="settings YAML (default: PFB_CONFIG, else defaults)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup-code", help="最初の管理者を登録する確認コードを表示する")
    create = sub.add_parser("create-admin", help="管理者を作る（--login なし: 確認コードを表示）")
    create.add_argument("--login")
    create.add_argument("--name")
    create.add_argument("--password-stdin", action="store_true")
    reset = sub.add_parser("reset-password", help="パスワードを設定し直す（ログイン中の画面は終わる）")
    reset.add_argument("--login", required=True)
    reset.add_argument("--password-stdin", action="store_true")
    sub.add_parser("users", help="利用する人の一覧")
    sub.add_parser("sign-out-all", help="すべての端末のログインを終える（古いバックアップを戻したあとなど）")
    args = parser.parse_args(argv)
    settings = _settings(args.settings)
    repo = _repo(settings)
    try:
        if args.command == "setup-code" or (args.command == "create-admin" and not args.login):
            return _setup_code(settings, repo)
        if args.command == "create-admin":
            user = auth.create_admin(repo, settings, args.login, args.name or args.login,
                                     _password(args.password_stdin))
            print(f"管理者「{user.display_name}」（ログイン名 {user.login}）を作りました。")
            return 0
        if args.command == "reset-password":
            user = auth.find_user(repo, args.login)
            if user is None:
                print("このログイン名の人はいません。", file=sys.stderr)
                return 2
            auth.reset_password(repo, user.id, _password(args.password_stdin))
            print(f"「{user.display_name}」のパスワードを設定し直しました。ログイン中の画面は終わりました。")
            return 0
        if args.command == "sign-out-all":
            n = auth.revoke_everyone(repo)
            print(f"{n}件のログインを終えました。みなさんにもう一度ログインしてもらってください。")
            return 0
        for user in auth.users(repo):
            state = "使える" if user.active else "止めています"
            print(f"{user.id}\t{user.login}\t{user.display_name}\t{user.role_label}\t{state}\t"
                  f"{user.last_login_at or '-'}")
        return 0
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2
    finally:
        repo.conn.close()


if __name__ == "__main__":
    sys.exit(main())
