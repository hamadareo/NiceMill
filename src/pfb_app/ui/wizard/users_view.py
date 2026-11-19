"""「利用する人」 (管理者 only, in 「管理用の画面」): everyone who may sign in, their roles, and
the one-time codes (services.auth).

- 招待する: a role and an optional name → a code shown once (INVITE_DAYS days). The primary.
- Each person: 名前, ログイン名, 役割, 状態, 最後にログインした日; 「変える」 holds 「役割を変える」,
  「パスワードの再設定コードを作る」 (RESET_HOURS hours) and 「使えなくする」 / 「使えるようにする」.
  A change of role or switching off ends that person's sign-ins. Your own row has no 「変える」
  (another 管理者 changes it), and the last 管理者 stays one (services.auth.LastAdmin).
- 使われていないコード: each open code with 「取り消す」.
"""
from datetime import datetime
from html import escape

import streamlit as st

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import flash, show_flash
from pfb_app.ui.wizard import theme
from pfb_app.ui.wizard.home import md_escape

SHOWN = "_pfb_users_code"   # (what, code, until) shown once after it was made
TITLE = "利用する人"
GUIDE = "このアプリにログインできる人と、その役割です。"
ROLE_HELP = ("管理者：すべての操作と、この画面・データの扱い・削除。　評価する人：作業を作って、"
             "確認・承認してPDFを渡す。　見るだけの人：承認した文章とPDFを見るだけ。")
INVITE = "招待する"
INVITE_GUIDE = (f"役割を選んで招待コードを作ります。コードは一度だけ表示され、{auth.INVITE_DAYS}日間"
                "使えます。招待した人に伝えると、ログインの画面の「招待コード・再設定コードをお持ちの方」"
                "から登録できます。")
INVITE_GO = "招待コードを作る"
ONLY_ADMIN = "この画面は管理者だけが使えます。"
SHOW_ONCE = "このコードは今だけ表示されます。いま控えて、本人だけに伝えてください。"


def _when(iso: str | None) -> str:
    if not iso:
        return "まだありません"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return "-"


def render(repo: Repo, settings: Settings) -> None:
    theme.inject_theme()
    if not access.is_admin():
        st.warning(ONLY_ADMIN)
        return
    show_flash()
    st.title(TITLE)
    st.caption(GUIDE)
    _shown_code()
    _invite(repo)
    _people(repo, access.current_user())
    _open_codes(repo)


def _shown_code() -> None:
    held = st.session_state.pop(SHOWN, None)
    if not held:
        return
    what, code, until = held
    with keyed.container(border=True, key="pfb_users_code"):
        st.success(md_escape(what))
        st.code(code, language=None)
        st.caption(f"{SHOW_ONCE}（{until}まで使えます）")


def _me_id() -> int | None:
    me = access.current_user()
    return me.id if me is not None else None


def _invite(repo: Repo) -> None:
    with theme.step_card(INVITE, INVITE_GUIDE, key="users_invite"):
        with st.form("pfb_invite", border=False):
            role = st.radio("役割", list(auth.ROLES), index=1, horizontal=True,
                            format_func=auth.ROLE_LABELS.get, key="pfb_invite_role",
                            help=ROLE_HELP)
            name = st.text_input("名前（入れなくてもかまいません）", key="pfb_invite_name",
                                 max_chars=auth.NAME_MAX)
            go = st.form_submit_button(INVITE_GO, type="primary", key="pfb_invite_go")
    if go:
        code = auth.create_invite(repo, role, name, by=_me_id())
        label = auth.ROLE_LABELS[role] + (f"・{name.strip()}さん" if name.strip() else "")
        until = _when(auth.open_invites(repo)[-1]["expires_at"])
        st.session_state[SHOWN] = (f"招待コードを作りました（{label}）。", code, until)
        st.rerun()


def _people(repo: Repo, me: auth.User | None) -> None:
    people = auth.users(repo)
    st.subheader(f"登録している人（{len(people)}人）")
    for user in people:
        with keyed.container(border=True, key=f"pfb_user_{user.id}"):
            main, action = st.columns([4, 1], vertical_alignment="center")
            state = "使える" if user.active else "使えなくしています"
            with main:
                st.html(f'<div class="pfb-workcard"><h3 class="pfb-workcard__title">'
                        f'{escape(user.display_name)}</h3><span class="pfb-workcard__meta">'
                        f'ログイン名 {escape(user.login)}・{escape(user.role_label)}・'
                        f'{escape(state)}</span></div><p class="pfb-workcard__reason">'
                        f'最後にログインした日：{escape(_when(user.last_login_at))}</p>')
            with action:
                if me is not None and user.id == me.id:
                    st.caption("あなたです")
                else:
                    _change(repo, user)


def _guarded(action) -> None:
    """Run a change; a refusal (the last 管理者, say) is said in its own words."""
    try:
        action()
    except auth.AuthError as e:
        flash(str(e), kind="warning")
    st.rerun()


def _change(repo: Repo, user: auth.User) -> None:
    with st.popover("変える", icon=":material/edit:", width="stretch"):
        role = st.selectbox("役割", list(auth.ROLES), index=auth.ROLES.index(user.role),
                            format_func=auth.ROLE_LABELS.get, key=f"pfb_user_role_{user.id}")
        if st.button("役割を変える", key=f"pfb_user_role_go_{user.id}", width="stretch",
                     disabled=role == user.role):
            def change():
                auth.set_role(repo, user.id, role)
                flash(f"{md_escape(user.display_name)}さんを「{auth.ROLE_LABELS[role]}」にしました。"
                      "もう一度ログインしてもらってください。", raw=True)
            _guarded(change)
        if user.active and st.button("パスワードの再設定コードを作る",
                                     key=f"pfb_user_reset_{user.id}", width="stretch"):
            code = auth.create_reset(repo, user.id, by=_me_id())
            until = _when(next(i["expires_at"] for i in reversed(auth.open_invites(repo))
                               if i["user_id"] == user.id))
            st.session_state[SHOWN] = (f"{user.display_name}さんの再設定コードを作りました。",
                                       code, until)
            st.rerun()
        if user.active:
            if st.button("使えなくする", key=f"pfb_user_off_{user.id}", type="tertiary",
                         icon=":material/block:", width="stretch"):
                def off():
                    auth.set_active(repo, user.id, False)
                    flash(f"{md_escape(user.display_name)}さんを使えなくしました。", raw=True)
                _guarded(off)
        elif st.button("使えるようにする", key=f"pfb_user_on_{user.id}", width="stretch"):
            def on():
                auth.set_active(repo, user.id, True)
                flash(f"{md_escape(user.display_name)}さんを使えるようにしました。", raw=True)
            _guarded(on)


def _open_codes(repo: Repo) -> None:
    codes = auth.open_invites(repo)
    if not codes:
        return
    names = {u.id: u.display_name for u in auth.users(repo)}
    st.subheader(f"使われていないコード（{len(codes)}件）")
    for c in codes:
        if c["user_id"] is not None:
            what = f"再設定コード・{names.get(c['user_id'], '')}さん"
        else:
            what = "招待コード・" + auth.ROLE_LABELS.get(c["role"], c["role"]) \
                + (f"・{c['display_name']}さん" if c["display_name"] else "")
        text, action = st.columns([4, 1], vertical_alignment="center")
        with text:
            st.html(f'<p class="pfb-workcard__reason">{escape(what)}'
                    f'（{escape(_when(c["expires_at"]))}まで）</p>')
        with action:
            if st.button("取り消す", key=f"pfb_code_cancel_{c['id']}", width="stretch"):
                def cancel(invite_id=c["id"]):
                    auth.cancel_invite(repo, invite_id)
                    flash("コードを取り消しました。")
                _guarded(cancel)
