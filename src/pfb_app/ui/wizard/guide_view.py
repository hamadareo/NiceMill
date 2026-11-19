"""「はじめての方へ」: the whole flow in four lines, the sample work, how to film, short answers,
and the user guide (docs/user_guide.md) section by section.

One primary: 「見本を見てみる」 while the sample is shown, else 「新しく始める」.
"""
from html import escape

import streamlit as st

from pfb_app.services import quality_stats
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import show_flash
from pfb_app.ui.wizard import help as guide
from pfb_app.ui.wizard import quality_view, sample_view, theme

LEAD = ("このアプリは、動画や書類から、一人一人の「よかったところ」を伝える文章の下書きを作ります。"
        "あなたが確かめて承認したものだけを、PDFで渡せます。")
FLOW = (
    ("動画か書類を入れる", "中身は自動で確かめます。違うところだけ直します。"),
    ("解析を待つ", "自動で進みます。画面を閉じても続きます。"),
    ("確かめて承認する", "文章と根拠の場面を見て、よければ承認します。"),
    ("PDFで渡す", "承認した人の分だけが入ります。"),
)
SAMPLE_GUIDE = "架空の選手とイラストで作った見本で、はじめから終わりまでの流れを先に見られます。"
SAMPLE_HIDDEN = "見本の作業は消してあります。もう一度出すと、ホームからも開けます。"
SEE = "見本を見てみる"
RESTORE = "見本をもう一度出す"
START = "新しく始める"
LOCAL = ("いいえ。解析はこのサーバーの中だけで行い、動画や文章をインターネット上のサービスには送りません。")
ACCURACY = ("自動の文章は下書きです。ふだんは、あなたが確かめて承認するまで、だれにも渡りません。"
            "作業で「確認せずに自動で承認する」をオンにしたときは、文章は確かめずに自動で承認されます。"
            "その人の配布物には「自動で作り、そのまま渡しています」と書きます。"
            "これまでに確かめたことと、まだ確かめていないことは「この解析の確かさ」にまとめてあります。")
NAMES = ("動画では、スポーツの作業なら、背番号（車なら車の番号）が読み取れた人に"
         "「#10」のような番号の名前が自動でつきます。読み取れない人は「選手1」のような名前のままです。"
         "名簿から選んだ名前や、自分でつけた名前は変わりません。"
         "顔などの見た目から名前を推定することは、まだ十分に確かではないためしていません。"
         "書類では、名前をファイル名から読み取ります。")
UNDO = ("「確認して承認」でその人を開き、「直す」で書き直して承認し直せます。"
        "PDFは、承認を変えたあとにもう一度受け取ってください。")
TIME_UNKNOWN = "動画の長さと人数によります。数十分かかることがあります。待っている間に画面を閉じてもかまいません。"

CSS = """<style>
.pfb-guide { max-width: 46rem; margin: 0 auto; font-family: var(--pfb-font-ui); color: var(--pfb-ink); }
.pfb-guide__title { margin: 0.5rem 0 0.25rem; font-size: 1.75rem; font-weight: 700; line-height: 1.4; }
.pfb-guide__lead { margin: 0 0 1.5rem; font-size: 1.0625rem; line-height: 1.85; max-width: 36em; }
.pfb-guide__h { margin: 1.75rem 0 0.6rem; font-size: 1.1875rem; font-weight: 700; }
.pfb-flow { list-style: none; margin: 0; padding: 0; display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; }
.pfb-flow li { margin: 0; position: relative; padding-top: 2.75rem; }
.pfb-flow li:not(:last-child)::after { content: ""; position: absolute; top: 1.1rem;
  left: 2.75rem; right: -0.75rem; height: 2px; background: var(--pfb-board); }
.pfb-flow__n { position: absolute; top: 0; left: 0; width: 2.25rem; height: 2.25rem;
  border-radius: 50%; background: var(--pfb-board); color: #FFFFFF; text-align: center;
  font-family: var(--pfb-font-hand); font-weight: 600; font-size: 1.125rem; line-height: 2.25rem; }
.pfb-flow__what { display: block; font-weight: 700; }
.pfb-flow__how { display: block; margin-top: 0.2rem; color: var(--pfb-muted); font-size: 0.9375rem;
  line-height: 1.65; }
.pfb-checks { list-style: none; margin: 0; padding: 0; }
.pfb-checks li { margin: 0 0 0.4rem; padding-left: 1.75rem; position: relative; line-height: 1.7; }
.pfb-checks li::before { content: "✓"; position: absolute; left: 0.25rem; top: 0;
  color: var(--pfb-board); font-weight: 700; }
@media (max-width: 40rem) {
  .pfb-flow { grid-template-columns: 1fr; gap: 0.75rem; }
  .pfb-flow li { padding: 0 0 0 3rem; min-height: 2.25rem; }
  .pfb-flow li:not(:last-child)::after { top: 2.5rem; bottom: -0.6rem; left: 1.05rem;
    right: auto; width: 2px; height: auto; }
}
</style>"""


def flow_html() -> str:
    items = "".join(f'<li><span class="pfb-flow__n" aria-hidden="true">{n}</span>'
                    f'<span class="pfb-flow__what">{escape(what)}</span>'
                    f'<span class="pfb-flow__how">{escape(how)}</span></li>'
                    for n, (what, how) in enumerate(FLOW, start=1))
    return f'<ol class="pfb-flow" aria-label="使う流れ">{items}</ol>'


def shooting_html() -> str:
    items = "".join(f"<li>{escape(line)}</li>" for line in guide.SHOOTING)
    return (f'<h2 class="pfb-guide__h">動画の撮り方のこつ</h2>'
            f'<ul class="pfb-checks" aria-label="動画の撮り方のこつ">{items}</ul>')


def time_answer(stats: quality_stats.QualityStats) -> str:
    seconds = stats.turnaround_median_sec
    if seconds is None:
        return TIME_UNKNOWN
    minutes = max(1, round(seconds / 60))
    return (f"このサーバーでは、解析を始めてから下書きがそろうまで、これまでの中央値で約{minutes}分でした"
            f"（{len(stats.turnaround_sec)}グループの記録から）。動画の長さと人数で変わります。"
            "待っている間に画面を閉じてもかまいません。")


def _sample_card(repo: Repo) -> bool:
    """The sample work's card; True when its 「見本を見てみる」 is the screen's primary."""
    shown = sample_view.shown(repo)
    with theme.step_card(guide.SAMPLE_TITLE, SAMPLE_GUIDE if shown else SAMPLE_HIDDEN,
                         key="guide_sample"):
        if shown:
            if st.button(SEE, key="pfb_guide_sample", type="primary",
                         icon=":material/visibility:"):
                st.switch_page(guide.SAMPLE_PAGE)
        elif access.can_edit() and st.button(RESTORE, key="pfb_guide_restore", icon=":material/undo:"):
            sample_view.set_shown(repo, True)
            st.rerun()
    return shown


def render(repo: Repo) -> None:
    theme.inject_theme()
    theme.set_domain("school")
    if st.button("ホームへ", key="pfb_guide_home", icon=":material/home:"):
        guide.go_home()
    show_flash()
    st.html(CSS + f'<div class="pfb-guide"><h1 class="pfb-guide__title">{guide.GUIDE_TITLE}</h1>'
            f'<p class="pfb-guide__lead">{escape(LEAD)}</p>{flow_html()}</div>')
    sample_primary = _sample_card(repo)
    st.html(CSS + f'<div class="pfb-guide">{shooting_html()}'
            '<h2 class="pfb-guide__h">よくある質問</h2></div>')
    stats = quality_view.measured(repo)
    with keyed.container(key="pfb_guide_faq"):
        for question, answer in (("動画や文章は、外に送られますか？", LOCAL),
                                 ("どのくらい時間がかかりますか？", time_answer(stats)),
                                 ("自動の文章は、どのくらい確かですか？", ACCURACY),
                                 ("動画の人の名前は、自動でつきますか？", NAMES),
                                 ("承認したあとで直したいときは？", UNDO)):
            with st.expander(question):
                st.markdown(answer)
                if answer == ACCURACY:
                    st.page_link(guide.QUALITY_PAGE, label=guide.QUALITY_TITLE,
                                 icon=":material/fact_check:")
    sections = guide.guide_sections()
    if sections:
        st.html(CSS + '<div class="pfb-guide"><h2 class="pfb-guide__h">くわしい使い方</h2></div>')
        with keyed.container(key="pfb_guide_manual"):
            for title, body in sections:
                with st.expander(title):
                    st.markdown(body)
    if not sample_primary and access.can_edit():
        with keyed.container(key="pfb_guide_start"):
            if st.button(START, key="pfb_guide_new", type="primary", icon=":material/add:"):
                guide.start_new()
