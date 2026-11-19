"""Streamlit entry point: streamlit run src/pfb_app/ui/app.py

The default page is the step-by-step wizard (its home screen lists every work); the earlier
pages stay under a quiet 「管理用の画面（くわしい操作）」 section (the sidebar starts collapsed) for tables, retries and edits.
The 「判定支援」 section after it holds the line-call pages of sports works (試合後の判定材料, ライブ判定);
it is left out of the menu while there is no sports work and no live demo video (the pages
stay reachable).

The sidebar menu is drawn here rather than by st.navigation: its 「ホーム」 is a button, so it
always leads back to the home screen, also from inside a work (a link to the page already
shown would keep the open work).

The app is called ナイスミル (theme.APP_NAME): the tab title, the tab icon and the logo in
the sidebar (st.logo) are set here, before the sign-in gate, so every screen carries them.

Accounts (ui/auth_ui, services.auth): nothing is shown before the gate has a signed-in person
(the setup screen while nobody is registered, else the sign-in). The pages offered follow the
role: the 「管理用の画面」 section (with 「データの扱い」 and 「利用する人」) only to 管理者; 成長の記録,
試合・練習のまとめ (/dashboard), 見本 and 判定支援 not to 見るだけの人 (この解析の確かさ stays reachable from the guide). A write a
見るだけの人's run still attempts is refused by their read-only connection and said in plain words.
"""
import sqlite3

import streamlit as st

from pfb_app.ui import access, auth_ui, keyed, linecall_view, ops_panel
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import router, theme

st.set_page_config(page_title=theme.APP_NAME, page_icon=str(theme.BRAND_DIR / theme.MARK_PNG),
                   layout="wide", initial_sidebar_state="collapsed")
# the wordmark at the top of the navy sidebar; the mark alone while the sidebar is closed
st.logo(str(theme.BRAND_DIR / theme.WORDMARK_ONDARK), size="large",
        icon_image=str(theme.BRAND_DIR / theme.MARK))
auth_ui.gate(get_settings())   # nobody signed in: the setup or sign-in screen, and the run stops
admin, edit = access.is_admin(), access.can_edit()
home = st.Page(router.render, title="ホーム", icon=":material/home:", url_path="home",
               default=True)
details = [
    st.Page(router.DETAIL_PAGES["assignments"], title="活動", url_path="assignments"),
    st.Page(router.DETAIL_PAGES["submissions"], title="ファイルの登録", url_path="submissions"),
    st.Page(router.DETAIL_PAGES["status"], title="処理状況", url_path="status"),
    st.Page(router.DETAIL_PAGES["review"], title="確認・承認", url_path="review"),
    st.Page(router.DETAIL_PAGES["export"], title="PDF出力", url_path="export"),
]
quality = st.Page("pages/quality.py", title="この解析の確かさ", url_path="quality")
details += [quality,
            st.Page("pages/data.py", title="データの扱い", url_path="data"),
            st.Page("pages/users.py", title="利用する人", url_path="users")]
guide = st.Page("pages/guide.py", title="はじめての方へ", url_path="guide")
sample = st.Page("pages/sample.py", title="見本の作業", url_path="sample")
growth = st.Page("pages/growth.py", title="成長の記録", url_path="growth")
dashboard = st.Page("pages/dashboard.py", title="まとめ", url_path="dashboard")   # ?work=<id>
ask = st.Page("pages/ask.py", title="ききたいことを聞く", url_path="ask")           # ?work=<id>
ADMIN_SECTION = "管理用の画面（くわしい操作）"
# 判定支援 (line calls of sports works), after the admin section
LINECALL_SECTION = "判定支援"
linecall = [
    st.Page(router.LINECALL_PAGES["linecall"], title="試合後の判定材料", url_path="linecall"),
    st.Page(router.LINECALL_PAGES["live"], title="ライブ判定", url_path="live"),
]
# 試合・練習のまとめ (/dashboard?work=<id>) is registered for 管理者 and 評価する人 only: for a
# 見るだけの人 the address is not a page of this app (Streamlit's page-not-found)
sections = {"": [home, guide, sample, growth, dashboard, ask] if edit else [home, guide, sample]}
if admin:
    sections[ADMIN_SECTION] = details
else:
    sections[""].append(quality)        # linked from the guide; not listed in the sidebar
if edit:
    sections[LINECALL_SECTION] = linecall
page = st.navigation(sections, position="hidden")
theme.inject_theme()
ops_panel.app_banner(get_settings())   # ops: ui.log, and the low-disk warning on every screen
with st.sidebar:
    here = page.url_path == home.url_path
    with keyed.container(key="pfb_sidenav_here" if here else "pfb_sidenav"):
        if st.button("ホーム", key="pfb_side_home", icon=":material/home:", type="tertiary",
                     width="stretch"):
            router.go_home()
            if not here:
                st.switch_page(home)
            st.rerun()
        st.page_link(guide, icon=":material/help:")
        if edit:
            st.page_link(growth, icon=":material/timeline:")
            st.page_link(ask, icon=":material/forum:")
        if admin:
            st.html(f'<p class="pfb-sidenav__section">{ADMIN_SECTION}</p>')
            for detail in details:
                st.page_link(detail)
        if edit and linecall_view.section_shown(get_repo(), get_settings()):   # none without sports
            st.html(f'<p class="pfb-sidenav__section">{LINECALL_SECTION}</p>')
            for entry in linecall:
                st.page_link(entry)
    auth_ui.account_box(get_settings())
try:
    page.run()
except sqlite3.DatabaseError as e:
    if not access.refused(e):          # only a 見るだけの人's refused write is said in words
        raise
    st.warning(access.VIEW_ONLY)
