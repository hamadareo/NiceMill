"""「データの扱い」 (pages/data.py, in the admin section): where the data lives, the setting
「動画を自動で消す」 (services.retention) and 「この人のデータを消す」 (services.erasure).

Every sentence about where data goes is true of the code: the analysis models run on this
machine (inference.served talks only to this machine; the models load offline), the screens
send no usage statistics and load no fonts from outside (.streamlit/config.toml), and the
only thing ever sent out is the optional operator notice (ops.alerts), which carries no
names, texts, file names or pictures (tests/test_data_truth.py pins this). The page's one
primary button is 「この設定にする」.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from pfb_app.export.sheets import local_time
from pfb_app.ops import backup, logs
from pfb_app.services import erasure, retention
from pfb_app.storage.db import Repo
from pfb_app.ui import access
from pfb_app.ui.common import flash, show_flash
from pfb_app.ui.wizard import theme

TITLE = "データの扱い"
GUIDE = ("このアプリのデータは、すべてこのサーバーの中にあります。外のサービスには送りません。"
         "ここで、動画を自動で消す設定と、一人分のデータを消す操作ができます。")
WHERE = "データのある場所"
RETENTION = "動画を自動で消す"
SAVE = "この設定にする"
ALL = "すべて消す（文章・PDF・成長の記録も）"
ERASE = erasure.TITLE
DAY_OPTIONS = (0, *retention.DAYS)
HELP = [
    "「動画を自動で消す」は、PDFを受け取ってから決めた日数がたった作業の動画と、解析の途中の"
    "ファイル（画像・追跡の記録・判定支援の判定材料など）を消します。承認した文章、PDF、成長の記録は残ります。",
    "「すべて消す」をオンにすると、日数がたった作業を、文章やPDFも含めて作業ごと消します。",
    "消す処理は1時間ごとに行います。解析中の作業は、解析が終わるまで消しません。",
    "ホームの作業のカードに、いつ何が消えるかが出ます。",
    "「この人のデータを消す」は、本人や保護者から消してほしいと頼まれたときに使います。",
]
CSS = """<style>
.pfb-data p, .pfb-data li { color: var(--pfb-ink); }
.pfb-data__muted { color: var(--pfb-muted); font-size: 0.875rem; }
</style>"""


def _days_label(days: int) -> str:
    return "消さない" if days == 0 else f"PDFを受け取ってから{days}日後"


def where_lines(settings) -> list[str]:
    """The plain statements of where the data is (the truth test reads them)."""
    alert = settings.alert_webhook_file is not None
    last = backup.last_result(settings)
    if not last:
        when = "バックアップはまだ作られていません"
    else:
        when = (f"最後のバックアップ：{local_time(last.get('at'))}"
                + ("" if last.get("ok") else "（失敗しました）"))
    return [
        f"作業の動画のコピー・書類・解析の途中のファイル・データベース（文章・承認・成長の記録）は、"
        f"このサーバーのデータフォルダ（{settings.data_dir}）にあります。",
        "解析は、このサーバーの中のモデルで行います。動画・書類・文章を外のサービスに送ることは"
        "ありません。解析に使う道具（人を見つける道具・書き起こし・文章を作るモデル）は、"
        "外につながない設定で動かし、使い方の記録も送りません。",
        "画面はブラウザーからこのサーバーに直接つないで見ます。利用状況の統計は送らず、"
        "文字の形（フォント）もこのサーバーから読みます。",
        f"受け取りフォルダ（{settings.inbox_dir}）に置いた元の動画は、アプリが消すことはありません。",
        "運用のお知らせ（止まった処理や空きの不足を管理者に知らせるもの）は、設定したときだけ"
        "送ります。名前・文章・ファイル名・映像は入りません。いまは"
        + ("設定されています。" if alert else "設定されていません。"),
        f"データベースのコピー（バックアップ）は、新しいものから{settings.backup_keep}回分を残"
        + (f"し、そのあとは週に1回分ずつ{settings.backup_keep_weekly}週分を残します"
           if settings.backup_keep_weekly > 0 else "します")
        + f"（標準では毎晩作ります。場所：{backup.backup_root(settings)}。動画は入りません）。"
        f"消したデータも、バックアップが入れ替わるまで残ります。{when}。",
        f"処理の記録（ログ）は {logs.log_dir(settings)} にあり、古いものから入れ替わります。",
        "受け取ったPDFは、保存したパソコンやスマートフォンに残ります。アプリからは消せません。",
    ]


def _where(settings) -> None:
    st.subheader(WHERE)
    st.html('<div class="pfb-data"><ul>' + "".join(
        f"<li>{escape(line)}</li>" for line in where_lines(settings)) + "</ul></div>")


SHOWN = 15   # titles listed before 「ほかn件」


def timing_text(n: int, everything: bool) -> str:
    """When the chosen setting removes what it reaches now (MAX_PER_PASS works an hour)."""
    what = "作業" if everything else "作業の動画"
    if not n:
        return "この設定で、いますぐ消えるものはありません。"
    per = retention.MAX_PER_PASS
    if n <= per:
        return f"この設定にすると、次の1時間のうちに{n}件の{what}が消えます。"
    return (f"この設定にすると、{n}件の{what}が、1時間に{per}件ずつ、"
            f"約{-(-n // per)}時間かけて消えます。")


def _retention(repo: Repo) -> None:
    st.subheader(RETENTION)
    current = retention.load(repo)
    st.caption(f"いまの設定：{_days_label(current.days)}"
               + ("（すべて消す）" if current.all else ""))
    days = st.radio("いつ消しますか", DAY_OPTIONS, index=DAY_OPTIONS.index(current.days),
                    format_func=_days_label, horizontal=True, key="pfb_data_days")
    everything = st.checkbox(ALL, value=current.all, disabled=days == 0, key="pfb_data_all")
    chosen = retention.Policy(days, everything and days != 0)
    sure = True
    if chosen.on and chosen != current:
        works = retention.preview_works(repo, chosen)
        st.caption(timing_text(len(works), chosen.all))
        if works:   # whole works or videos go for good: name them and ask first
            more = len(works) - SHOWN
            st.html('<div class="pfb-delete"><p class="pfb-delete__title">消えるもの</p>'
                    + "".join(f"<p>「{escape(t)}」" + ("" if chosen.all else "の動画") + "</p>"
                              for _, t in works[:SHOWN])
                    + (f"<p>ほか{more}件</p>" if more > 0 else "")
                    + '<p class="pfb-delete__undo">消したものは元に戻せません。</p></div>')
            sure = st.checkbox("これらが消えることを確認しました",
                               key=f"pfb_data_sure_{chosen.days}_{int(chosen.all)}")
    if st.button(SAVE, key="pfb_data_save", type="primary",
                 disabled=chosen == current or not sure):
        retention.save(repo, chosen.days, chosen.all)
        flash("設定を保存しました。")
        st.rerun()


def _erase(repo: Repo, settings) -> None:
    st.subheader(ERASE)
    st.caption("本人や保護者から消してほしいと頼まれたときに使います。同じ人として成長の記録で"
               "つないだ作業の分も、まとめて消えます。")
    if erasure.unfinished(repo):
        st.info(erasure.STOPPED_NOTE)
    people = erasure.people(repo)
    if not people:
        st.caption("消せる人はいません。")
        return
    word = st.text_input("名前や作業名でしぼりこむ", key="pfb_data_find").strip()
    shown = [p for p in people if not word or word in p["name"] or word in p["title"]]
    if not shown:
        st.caption("見つかりませんでした。")
        return
    ids = [p["subject_id"] for p in shown]
    labels = {p["subject_id"]: erasure.label(p) for p in shown}
    chosen = st.selectbox("消す人", ids, index=None, format_func=labels.get,
                          placeholder="選んでください", key="pfb_data_person")
    if chosen is None:
        return
    try:
        plan = erasure.plan(repo, chosen)
    except KeyError:
        st.caption("この人はもういません。")
        return
    st.html('<div class="pfb-delete"><p class="pfb-delete__title">消えるもの</p>'
            + "".join(f"<p>{escape(line)}</p>" for line in erasure.what_goes(plan))
            + '<p class="pfb-delete__title">残るもの</p>'
            + "".join(f"<p>{escape(line)}</p>" for line in
                      (*erasure.keeps(settings), *erasure.engine_note(plan)))
            + "".join(f"<p>{escape(line)}</p>" for line in
                      (lambda found: found + [erasure.MENTIONS] if found else [])(
                          erasure.mentions(repo, plan)))
            + '<p class="pfb-delete__undo">この操作は元に戻せません。</p></div>')
    if plan.busy:
        st.info(erasure.BUSY)
    sure = st.checkbox("消してよいことを確認しました", key=f"pfb_data_sure_{chosen}",
                       disabled=plan.busy)
    if st.button(ERASE, key=f"pfb_data_erase_{chosen}", icon=":material/delete:",
                 disabled=plan.busy or not sure):
        try:
            done = erasure.erase(repo, settings, chosen)
        except erasure.Busy as e:
            flash(str(e), kind="warning")
        except KeyError:
            flash("この人はもういません。", kind="warning")
        else:
            flash(f"{len(done.items)}件の作業から、この人のデータを消しました。"
                  + "".join(erasure.engine_note(done)))
        st.session_state.pop("pfb_data_person", None)
        st.rerun()


ONLY_ADMIN = "この画面は管理者だけが使えます。"


def render(repo: Repo, settings) -> None:
    theme.inject_theme()
    if not access.is_admin():       # 管理者 only (the navigation offers it to no one else)
        st.warning(ONLY_ADMIN)
        return
    theme.set_domain("school")      # the general words: the page is about every work
    st.html(CSS)
    st.title(TITLE)
    show_flash()
    st.caption(GUIDE)
    _where(settings)
    _retention(repo)
    _erase(repo, settings)
    theme.help_expander(HELP)
