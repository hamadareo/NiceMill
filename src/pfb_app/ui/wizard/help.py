"""In-app help: one short contextual tip under each wizard step, and 「はじめての方へ」.

The tip is one or two sentences about what matters most on the step shown (the step's own
「この画面の使い方」 stays the full how-to), followed by a quiet link to 「はじめての方へ」; the
PDF step also links 「この解析の確かさ」. The router draws it below every screen
(`step_footer`), so the step files themselves stay as they are.
"""
from html import escape
from pathlib import Path

import streamlit as st

from pfb_app.ui import keyed
from pfb_app.ui.wizard import theme

GUIDE_PAGE = "pages/guide.py"
QUALITY_PAGE = "pages/quality.py"
SAMPLE_PAGE = "pages/sample.py"
GUIDE_TITLE = "はじめての方へ"
QUALITY_TITLE = "この解析の確かさ"
SAMPLE_TITLE = "見本の作業"
USER_GUIDE = Path(__file__).resolve().parents[4] / "docs" / "user_guide.md"

SHOOTING = (   # 撮り方のこつ: what makes the analysis work best
    "スマートフォンは横向きにして、三脚や台に置いて動かさずに撮ります。",
    "コートや部屋など、動く場所の全体が入るように撮ります。",
    "画質は 1080p・30fps で十分です。4K はファイルがとても大きくなります。",
    "長い動画は、はじめの20分を解析します。解析する部分は、始める前に変えられます。",
)

# key -> the tip shown under that step (general words; theme.L turns them into sports words)
TIPS = {
    "drop": "動画の撮り方のこつ：" + SHOOTING[0] + "動く場所の全体が入るように撮ると、人の見分けが確かになります。",
    "check": "結果は推定です。違うところだけ直せば、あとは自動で進みます。",
    "assignment": "ここに書いた内容と観点に沿って、フィードバックの文章が作られます。",
    "files": "同じ場面を別のカメラで撮った動画は、同じグループにまとめて入れます。",
    "people": "この段階はとばしてもかまいません。名前は「確認して承認」でもつけられます。",
    "wait": "画面を閉じても解析は続きます。終わると、ホームの札が「あなたの番」に変わります。",
    "review": "文章は、時刻つきの場面をもとに作られています。根拠の場面と合っているかを見てから承認します。",
    "pdf": "参加者に渡す前に、PDFを開いて名前と文章、写真・図を確かめてください。",
}

CSS = """<style>
.pfb-tip {
  max-width: 46rem; margin: 0.75rem auto 0; padding: 0.6rem 0.9rem; box-sizing: border-box;
  border-left: 3px solid var(--pfb-board); background: var(--pfb-mist);
  border-radius: 0 8px 8px 0; color: var(--pfb-ink); font-size: 0.9375rem; line-height: 1.7;
}
.pfb-tip__label { font-weight: 700; color: var(--pfb-board); margin-right: 0.5rem; }
.st-key-pfb_help_links { max-width: 46rem; margin: 0 auto; }
</style>"""


def _home_page() -> "st.Page":
    """The home page as app.py registers it (a page is matched by its url_path)."""
    from pfb_app.ui.wizard import router   # noqa: PLC0415 (router imports the step screens)
    return st.Page(router.render, title="ホーム", icon=":material/home:", url_path="home",
                   default=True)


def go_home() -> None:
    """From a page of its own (はじめての方へ, 見本の作業, この解析の確かさ) to the home screen."""
    from pfb_app.ui.wizard import router   # noqa: PLC0415
    router.go_home()
    st.switch_page(_home_page())


def start_new() -> None:
    """「新しく始める」 from a page of its own: the drop zone of a new work."""
    from pfb_app.ui.wizard import router   # noqa: PLC0415
    router.start_new()
    st.session_state[router._SHOWN] = None   # not "the work left through a link": no going home
    st.switch_page(_home_page())


def tip_html(key: str) -> str:
    text = TIPS.get(key)
    if not text:
        return ""
    return (f'<p class="pfb-tip"><span class="pfb-tip__label">ひとこと</span>'
            f'{escape(theme.L(text))}</p>')


def step_footer(key: str) -> None:
    """Under a step screen: its tip, then 「はじめての方へ」 (and on the PDF step,
    「この解析の確かさ」)."""
    tip = tip_html(key)
    if tip:
        st.html(CSS + tip)
    with keyed.container(horizontal=True, key="pfb_help_links"):
        st.page_link(GUIDE_PAGE, label=GUIDE_TITLE, icon=":material/help:")
        if key == "pdf":
            st.page_link(QUALITY_PAGE, label=QUALITY_TITLE, icon=":material/fact_check:")


def guide_sections() -> list[tuple[str, str]]:
    """docs/user_guide.md as [(heading, body)] by its "## " headings ([] when the file is not
    there, e.g. an installed package without the docs)."""
    try:
        text = USER_GUIDE.read_text(encoding="utf-8")
    except OSError:
        return []
    sections, title, body = [], None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title is not None:
                sections.append((title, "\n".join(body).strip()))
            title, body = line[3:].strip(), []
        elif title is not None:
            body.append(line)
    if title is not None:
        sections.append((title, "\n".join(body).strip()))
    return sections
