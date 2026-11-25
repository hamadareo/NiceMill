"""試合・活動のまとめ (pages/dashboard.py?work=<id>): one video group's numbers and pictures,
from services.dashboard (only what the analysis recorded; nothing new is estimated).

One filter row (グループ, 見る人) scopes everything below it: the stat tiles, 人ごとの表 (the
numbers of every chart, sortable), the charts (minutes per person, good moments per category
and person, per team or for the whole group), 時間の流れ with the moment list, sports' 位置
(court heatmaps and the estimated distance with the court geometry; else the image-plane
heatmap of one camera, no distance) or the other profiles' 場面ごとの様子 (who did what kind of
good moment when), 見どころ (the moments the texts cite first), one person's view, the two CSV
downloads (the person table is the one primary) and 「この数字について」.

「この場面から見る」 cuts a short excerpt (services.moment_clip) and plays it under that row from
the moment; only one player is open. One approved person's view also offers their 解説動画
(ui/wizard/explainer_view.person_panel). A work whose videos were removed has no player.

Accounts (release 4): ui/app.py registers the page for 管理者 and 評価する人 only; pages/dashboard.py
also stops a run without a valid editing session (access.require_page: a 見るだけの人, signed
out, revoked), the entry link is only drawn for them, and the player checks again right
before a clip file is cut (NOT_ALLOWED). Its data comes through ui.common.get_repo (read-only
without such a session).
Charts are export.charts / export.dashboard_charts SVG shown as images with their words.
"""
from __future__ import annotations

import logging
import re
from html import escape, unescape

import streamlit as st

from pfb_app.domain import profiles
from pfb_app.domain.models import Kind
from pfb_app.export import charts, labels
from pfb_app.export import dashboard_charts as dc
from pfb_app.export.charts import Bar, Mark
from pfb_app.services import dashboard as dash
from pfb_app.services import moment_clip
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import get_settings
from pfb_app.ui.wizard import svg, theme

log = logging.getLogger(__name__)

PAGE = "pages/dashboard.py"
TITLE = "活動のまとめ"            # localised: 試合・練習のまとめ, 会議のまとめ, ...
LINK = "活動のまとめを見る"
GUIDE = ("解析で記録されたことだけから、人ごとの時間やよかった場面をまとめています。"
         "記録がないところは、ないと書いています。")
NO_WORK = "作業が選ばれていません。作業の「確認して承認」か「PDFを受け取る」の画面から開いてください。"
NOT_FOUND = "この作業は見つかりませんでした（削除された可能性があります）。"
VIDEO_ONLY = "このまとめは、動画の作業で見られます。"
NO_GROUP = "まだ動画が登録されていません。"
NOBODY = "まだ評価する参加者がいません。人を選ぶと、ここにまとめが出ます。"
GROUP = "グループ"
WHO = "見る人"
EVERYONE = "全員"
JUMP = "この場面から見る"
CLOSE = "動画を閉じる"
CSV_PEOPLE = "人ごとの表をCSVで受け取る"
CSV_MOMENTS = "場面の一覧をCSVで受け取る"
CSV_TEAMS = "チームごとの表をCSVで受け取る"     # sports
CSV_GROUPS = "グループの合計をCSVで受け取る"    # the other profiles
NOT_ALLOWED = "このまとめは、管理者と評価する人が見られます。ログインし直してから開いてください。"
TWIN = "表で見る"
PAGE_SIZE = 30
WAITING_CHIP = "承認前"
# 「あとから訂正」 (board.live): the check state of the newest text, in words without 承認
LIVE_HELP = ("数字は、PDFの文と同じ記録から出しています（AIが作成した文。訂正した人は訂正した文）。"
             "誤りの報告を確認中の人の文は出していません。")
LIVE_NO_TEXT = "文章は出していません（{status}）。"
LIVE_NONE = "文章はまだありません。"
LIVE_ABOUT = ("よかった場面：よかった点として判断された場面です。いまの文（AIが作成した文。"
              "訂正した人は訂正した文）のもとになった場面です。")
LIVE_CHIP_ABOUT = ("見どころと場面の一覧で「誤りの報告あり・確認中」とあるものは、"
                   "誤りの報告を確認中の文のもとになった場面です。")
REMOVED = ("この作業の動画は消してあります。映っていた時間と位置の記録は動画といっしょに消えたため、"
           "残っている記録（よかった場面と文章）だけを出しています。場面の動画は見られません。")
RETRACK = "人の追跡をやり直しているため、映っていた時間と位置は出していません。"
NO_TIME = "映っていた時間の記録がありません。"
NO_MOMENTS = "よかった場面の記録がまだありません。"
CLIP_FAILED = "この場面の動画を用意できませんでした。"
PLAY_KEY = "_pfb_dash_play"      # (submission id, row key, camera, seconds in the video)
BOARD_KEY = "_pfb_dash_board"    # (aid, sid, stamp, Board)
SCENES = "場面ごとの様子"
SCENES_NOTE = "時間を区切って、その間に誰がどんな場面でよかったかを並べています（よかった場面として記録されたものだけ）。"
HIGHLIGHTS = "見どころ"
HIGHLIGHTS_NOTE = "フィードバックの文章が取り上げた場面から、一人2つまでを時間の順に並べています。"
ABOUT = "この数字について"
HELP = [
    "上の「グループ」と「見る人」で、下のすべての表と図の範囲が変わります。",
    "「この場面から見る」を押すと、その場面の少し前からの短い動画がその下に出ます。",
    "「人ごとの表をCSVで受け取る」で、表計算ソフトで開ける表を受け取れます。",
    "数字は、承認した文章と同じ記録から出しています。承認前の人は、いまの下書きのもとになった場面です。",
]
CSS = """<style>
.pfb-dash-meta { margin: 0 0 0.5rem; color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-dash-chart img { max-width: 100%; height: auto; }
.pfb-dash-row { margin: 0; padding: 0.375rem 0; border-top: 1px solid var(--pfb-line);
  color: var(--pfb-ink); }
.pfb-dash-row__when { font-weight: 700; margin-right: 0.5rem; }
.pfb-dash-row__who { margin-right: 0.5rem; }
.pfb-dash-row__what { display: block; color: var(--pfb-ink); }
.pfb-dash-row img.glyph-img { vertical-align: -2px; margin-right: 0.25rem; }
.pfb-dash-caption { margin: 0.5rem 0 0.25rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-dash-row__chip { display: inline-block; margin-right: 0.5rem; padding: 0 0.375rem;
  border: 1px solid var(--pfb-line); border-radius: 999px; font-size: 0.8125rem;
  color: var(--pfb-muted); }
.pfb-dash-note { margin: 0.25rem 0; color: var(--pfb-muted); font-size: 0.875rem; }
</style>"""


def title_of(work: dict) -> str:
    return _localized(TITLE, work)


def _wording(work: dict) -> str:
    return profiles.of_work(dict(work))


def _localized(text: str, work: dict) -> str:
    from pfb_app.domain import vocab   # noqa: PLC0415
    return vocab.localize(text, _wording(work))


def entry_link(work: dict) -> None:
    """The quiet link to this page (steps 5 and 6 of a video work; 管理者 and 評価する人 only:
    for anyone else the page is not registered)."""
    if work.get("kind") != Kind.video.value or not access.can_edit():
        return
    st.page_link(PAGE, label=_localized(LINK, work), icon=":material/monitoring:",
                 query_params={"work": str(work["id"])})


def _number(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _board(repo: Repo, aid: int, sid: int) -> dash.Board:
    key = dash.stamp(repo, aid)
    held = st.session_state.get(BOARD_KEY)
    if held and held[:3] == (aid, sid, key):
        return held[3]
    board = dash.build(repo, aid, sid)
    st.session_state[BOARD_KEY] = (aid, sid, key, board)
    return board


_DESC = re.compile(r"<desc>(.*?)</desc>", re.S)


def _img(drawn: str, alt: str) -> None:
    """The chart as an image; its alt text is `alt` plus the chart's own values in words."""
    if drawn:
        found = _DESC.search(drawn)
        words = unescape(found.group(1)) if found else ""
        st.html(f'<div class="pfb-dash-chart">{svg.img(drawn, f"{alt}。{words}" if words else alt)}'
                '</div>')


def _twin(rows: list[dict]) -> None:
    """The chart's table twin: the same numbers as a table, folded."""
    if rows:
        with st.expander(TWIN):
            st.dataframe(rows, hide_index=True, width="stretch")


def _head(text: str) -> None:
    st.html(f'<h3 class="pfb-subhead">{escape(theme.L(text))}</h3>')


def _caption(text: str) -> None:
    """The visible name of the chart under it (its alt text says the same)."""
    st.html(f'<p class="pfb-dash-caption">{escape(text)}</p>')


def _note(text: str) -> None:
    st.html(f'<p class="pfb-dash-note">{escape(text)}</p>')


def _minutes_text(seconds: float | None) -> str:
    return "記録なし" if seconds is None else f"{seconds / 60:.1f}分"


# --- render ---------------------------------------------------------------------------------

def render(repo: Repo) -> None:
    theme.inject_theme()
    st.html(CSS)
    aid = _number(st.query_params.get("work"))
    if aid is None:
        theme.set_domain("school")
        st.info(NO_WORK)
        return
    try:
        work = repo.get_assignment(aid)
    except KeyError:
        theme.set_domain("school")
        st.info(NOT_FOUND)
        return
    theme.set_domain(_wording(work))
    if work.get("kind") != Kind.video.value:
        st.info(theme.L(VIDEO_ONLY))
        return
    found = dash.groups(repo, aid)
    if not found:
        st.info(theme.L(NO_GROUP))
        return
    with theme.step_card(TITLE, GUIDE, key="dash"):
        st.html(f'<p class="pfb-dash-meta">{escape(work["title"])}</p>')
        sid, who = _filters(repo, aid, found)
        try:
            board = _board(repo, aid, sid)
        except KeyError:
            st.info(theme.L(NOT_FOUND))
            return
        if not board.people:
            st.info(theme.L(NOBODY))
            return
        person = board.person(who) if who is not None else None
        _notes(board)
        _tiles(board, person)
        _table(board, person)
        _downloads(board)
    if person is None:
        _charts(board)
    else:
        _person(board, person)
        if person.approved and not board.media_removed:
            from pfb_app.ui.wizard import explainer_view   # noqa: PLC0415 - 解説動画
            _head(explainer_view.TITLE)
            _note(explainer_view.GUIDE)
            explainer_view.person_panel(repo, board.submission_id, person.subject_id,
                                        person.name, work["title"])
    _timeline(repo, board, person)
    if board.sports:
        _places(board, person)
    else:
        _scenes(repo, board, person)
    if person is None:
        _highlights(repo, board)
    _about(board)
    theme.help_expander(HELP[:-1] + [LIVE_HELP] if board.live else HELP)


def _filters(repo: Repo, aid: int, found: list[tuple[int, str]]) -> tuple[int, int | None]:
    """The one filter row: グループ (with several groups) and 見る人."""
    cols = st.columns(2 if len(found) > 1 else 1)
    labels_ = dict(found)
    if len(found) > 1:
        with cols[0]:
            sid = st.selectbox(theme.L(GROUP), [g for g, _ in found],
                               format_func=lambda g: labels_[g], key=f"pfb_dash_group_{aid}")
    else:
        sid = found[0][0]
    try:
        board = _board(repo, aid, sid)
        names = {p.subject_id: p.name for p in board.people}
    except KeyError:
        names = {}
    with cols[-1]:
        who = st.selectbox(theme.L(WHO), [0, *names], format_func=lambda i: names.get(i, EVERYONE),
                           key=f"pfb_dash_who_{aid}_{sid}")
    return sid, (who or None)


def _notes(board: dash.Board) -> None:
    if board.media_removed:
        st.info(theme.L(REMOVED), icon=":material/info:")
    elif board.retracking:
        st.info(theme.L(RETRACK), icon=":material/info:")


def _tiles(board: dash.Board, person: dash.Person | None) -> None:
    a, b, c = st.columns(3)
    a.metric("解析した長さ", "記録なし" if not board.duration_s else
             f"{board.duration_s / 60:.1f}分")
    if person is None:
        b.metric(theme.L("評価した参加者"), f"{len(board.people)}人")
        c.metric("よかった場面", f"{board.total}回")
    else:
        b.metric("映っていた時間", _minutes_text(person.on_screen_s))
        c.metric("よかった場面", f"{person.total}回")


def _rows(board: dash.Board, people: list[dash.Person]) -> list[dict]:
    rows = []
    for p in people:
        row = {"名前": p.name}
        if board.sports:
            row["チーム"] = p.team or "—"
        row["状態"] = p.status
        row["映っていた時間（分）"] = dash.minutes(p.on_screen_s)
        row["よかった場面（回）"] = p.total
        for cat in board.categories:
            row[cat.label] = p.counts.get(cat.key, 0)
        if board.untyped:
            row["種類がわからない"] = p.untyped
        if board.court:
            row["動いた距離（推定・m）"] = dash.distance_shown(p.distance_m)
        rows.append(row)
    return rows


def _table(board: dash.Board, person: dash.Person | None) -> None:
    _head("人ごとの表")
    people = board.people if person is None else [person]
    st.dataframe(_rows(board, people), hide_index=True, width="stretch")
    _note("「—」や空欄は記録がないところです。表の見出しを押すと並べかえられます。")
    if any(len(p.on_screen_by_camera) > 1 for p in people):
        _note(dash.SEVERAL_CAMERAS + "（カメラごとの時間はCSVにあります）。")


def _downloads(board: dash.Board) -> None:
    name = f"{board.work['title']}_{board.group_label}"
    with keyed.container(horizontal=True, key="pfb_dash_downloads"):
        st.download_button(theme.L(CSV_PEOPLE), dash.people_csv(board),
                           file_name=f"{name}_人ごと.csv",
                           mime="text/csv", type="primary", icon=":material/download:",
                           key="pfb_dash_csv_people")
        st.download_button(theme.L(CSV_MOMENTS), dash.moments_csv(board),
                           file_name=f"{name}_場面.csv",
                           mime="text/csv", icon=":material/download:",
                           key="pfb_dash_csv_moments")
        teams = CSV_TEAMS if board.sports else CSV_GROUPS
        st.download_button(theme.L(teams), dash.teams_csv(board), file_name=f"{name}_合計.csv",
                           mime="text/csv", icon=":material/download:",
                           key="pfb_dash_csv_teams")


def _charts(board: dash.Board) -> None:
    timed = sorted((p for p in board.people if p.on_screen_s),
                   key=lambda p: (-p.on_screen_s, p.name))
    left, right = st.columns(2)
    with left:
        _head("映っていた時間")
        drawn = charts.bar_chart([Bar(p.name, round(p.on_screen_s / 60, 1)) for p in timed],
                                 title="映っていた時間", unit="分", width=320, row_h=22,
                                 thick=10, value_text=lambda v: f"{v:.1f}分")
        if drawn:
            _img(drawn, "映っていた時間（分）")
            _twin([{"名前": p.name, "映っていた時間（分）": dash.minutes(p.on_screen_s)}
                   for p in timed])
        else:
            _note(theme.L(RETRACK) if board.retracking else NO_TIME)
    with right:
        _head("よかった場面の数")
        counted = sorted((p for p in board.people if p.total), key=lambda p: (-p.total, p.name))
        drawn = charts.bar_chart([Bar(p.name, p.total) for p in counted], title="よかった場面の数",
                                 unit="回", width=320, row_h=22, thick=10)
        if drawn:
            _img(drawn, "よかった場面の数")
            _twin([{"名前": p.name, "よかった場面（回）": p.total} for p in counted])
        else:
            _note(NO_MOMENTS)
    _head("種類ごとのよかった場面")
    names = [p.name for p in board.people]
    rows = [(cat, [p.counts.get(cat.key, 0) for p in board.people]) for cat in board.categories]
    drawn = dc.count_grid(names, rows, title="種類ごとのよかった場面（人ごと）")
    if drawn:
        _caption("種類ごとのよかった場面（人ごと）")
        _img(drawn, "種類ごとのよかった場面（人ごと）。数字は回数です")
        _twin([{"種類": cat.label, **{p.name: p.counts.get(cat.key, 0) for p in board.people}}
               for cat in board.categories])
    teams = dash.team_totals(board)
    if board.sports and len(teams) > 1:
        drawn = dc.count_grid([t.name for t in teams],
                              [(cat, [t.counts.get(cat.key, 0) for t in teams])
                               for cat in board.categories],
                              title="種類ごとのよかった場面（チームごと）")
        if drawn:
            _caption("種類ごとのよかった場面（チームごと）")
            _img(drawn, "種類ごとのよかった場面（チームごと）。数字は回数です")
            _twin([{"種類": cat.label, **{t.name: t.counts.get(cat.key, 0) for t in teams}}
                   for cat in board.categories])
        _note("チームごとの数は、評価した選手の分だけを足しています（相手の選手や、評価しない人は入りません）。")
    if board.untyped:
        _note(f"種類がわからない場面が{board.untyped}件あります（種類ごとの数には入れていません）。")
    if drawn:
        _note("数字は回数です。色が濃いほど多く記録されています。")


def _person(board: dash.Board, p: dash.Person) -> None:
    _head(f"{p.name}さん")
    if p.team:
        _note(p.team)
    counts = [(cat, p.counts[cat.key]) for cat in board.categories if p.counts.get(cat.key)]
    if counts:
        _caption("どんな場面でよかったか")
        drawn = charts.bar_chart([Bar(c.label, n, c) for c, n in counts],
                                 title="どんな場面でよかったか", unit="回", width=360)
        _img(drawn, "どんな場面でよかったか")
        _twin([{"種類": c.label, "回数": n} for c, n in counts])
    elif not p.total:
        _note(NO_MOMENTS)
    if p.text:
        st.html(f'<blockquote class="pfb-growth-quote">{escape(p.text)}</blockquote>')
    elif board.live:       # 「あとから訂正」: the check state, never 承認
        _note(LIVE_NO_TEXT.format(status=p.status) if p.status else LIVE_NONE)
    else:
        _note(f"文章はまだ承認されていません（{p.status}）。")


def _marks(moments) -> list[Mark]:
    return [Mark(m.t_s, m.category) for m in moments if m.t_s is not None]


def _timeline(repo: Repo, board: dash.Board, person: dash.Person | None) -> None:
    _head("時間の流れ")
    moments = [m for m in board.moments if person is None or m.subject_id == person.subject_id]
    if board.duration_s:
        spans = person.spans if person is not None else []
        drawn = charts.timeline(board.duration_s, spans, _marks(moments), tick=board.tick,
                                title="時間の流れ", width=640)
        _img(drawn, "時間の流れ。" + "、".join(
            f"{m.when} {m.name} {m.category.label if m.category else 'よかった場面'}"
            for m in moments[:40]))
        _note("目もりは" + (theme.L(board.words.time_anchor) if board.clock_offset_s is not None
                         else "開始から") + "の時間です。")
    if not moments:
        _note(NO_MOMENTS)
        return
    with st.expander(f"場面の一覧（{len(moments)}件）", expanded=person is not None):
        pages = (len(moments) + PAGE_SIZE - 1) // PAGE_SIZE
        page = 1
        if pages > 1:
            page = st.selectbox(
                "ページ", list(range(1, pages + 1)),
                format_func=lambda n: f"{n}（{(n - 1) * PAGE_SIZE + 1}〜"
                                      f"{min(n * PAGE_SIZE, len(moments))}件め）",
                key=f"pfb_dash_page_{board.submission_id}_{person.subject_id if person else 0}")
        first = (page - 1) * PAGE_SIZE
        for k, m in enumerate(moments[first:first + PAGE_SIZE], start=first):
            _moment_row(repo, board, m, f"list_{k}")


def _glyph(category) -> str:
    return svg.img(charts.glyph(category, 12), category.label if category else "よかった場面",
                   cls="glyph-img")


def _moment_row(repo: Repo, board: dash.Board, m: dash.Moment, key: str) -> None:
    kind = m.category.label if m.category else "種類がわからない場面"
    word = (m.state if not m.approved else "") if board.live else ("" if m.approved else WAITING_CHIP)
    chip = f'<span class="pfb-dash-row__chip">{escape(word)}</span>' if word else ""
    st.html(f'<p class="pfb-dash-row"><span class="pfb-dash-row__when">{escape(m.when)}</span>{chip}'
            f'<span class="pfb-dash-row__who">{escape(m.name)}</span>{_glyph(m.category)}'
            f'{escape(kind)}<span class="pfb-dash-row__what">{escape(m.what)}</span></p>')
    _jump(repo, board, m.camera, m.t_s, key)


def _jump(repo: Repo, board: dash.Board, camera: str | None, t_s: float | None,
          key: str) -> None:
    """「この場面から見る」 under a row, and the player when it is the one pressed."""
    if board.media_removed or t_s is None or not camera or camera not in board.videos:
        return
    at = board.start_s + t_s
    here = (board.submission_id, key, camera, at)
    if st.session_state.get(PLAY_KEY) == here:
        _player(repo, board, camera, at)
        if st.button(CLOSE, key=f"pfb_dash_close_{key}", type="tertiary",
                     icon=":material/close:"):
            st.session_state.pop(PLAY_KEY, None)
            st.rerun()
    elif st.button(JUMP, key=f"pfb_dash_jump_{key}", icon=":material/play_circle:"):
        st.session_state[PLAY_KEY] = here
        st.rerun()


def _player(repo: Repo, board: dash.Board, camera: str, at: float) -> None:
    settings = get_settings()
    src, why = moment_clip.source(settings, board.videos.get(camera),
                                  board.frame_sizes.get(camera))
    if src is None:
        _note(why)
        return
    if not access.can_edit():          # checked again right before a file is written
        _note(NOT_ALLOWED)
        return
    try:
        clip, offset = moment_clip.clip(
            moment_clip.clips_dir(settings, board.work["id"], board.submission_id),
            camera, src, at, settings=settings)
        # inside the try: a clip removed meanwhile (another press's _prune) is a plain message
        st.video(str(clip), start_time=int(offset), format="video/mp4")
    except moment_clip.ClipRefused as e:
        _note(str(e))
    except Exception:   # noqa: BLE001 - a clip is extra: the page stays
        log.exception("the clip of %s at %.1f s could not be made or played", camera, at)
        _note(CLIP_FAILED)


def _places(board: dash.Board, person: dash.Person | None) -> None:
    _head("位置")
    people = board.people if person is None else [person]
    if board.court:
        groups = [(person.name, [person])] if person is not None else dash.team_members(board)
        cols = st.columns(max(1, min(len(groups), 2)))
        for k, (name, members) in enumerate(groups):
            pts = [pt for p in members for pt in p.court_points]
            drawn = charts.court(pts, title=f"{name}のいた所", width=300) if pts else ""
            with cols[k % len(cols)]:
                _caption(f"{name}のいた所")
                if drawn:
                    _img(drawn, f"{name}のいた所（コートの上、1 mごと）")
                    _twin(dash.court_cells(pts))
                else:
                    _note(f"{name}：コートの上の位置の記録がありません。")
        _distance_notes(people)
        return
    shown = _frame_camera(board, people)
    if shown is None:
        _note(theme.L(RETRACK) if board.retracking else
              "位置の記録がありません。" if not board.media_removed else
              "位置の記録は動画といっしょに消してあります。")
        return
    size = board.frame_sizes.get(shown)
    _note(f"画面の中の位置です（{labels.camera_word(shown)}から見た位置）。コートの位置がわからないため、"
          "動いた距離は出していません。")
    groups = [(person.name, [person])] if person is not None else dash.team_members(board)
    cols = st.columns(max(1, min(len(groups), 2)))
    for k, (name, members) in enumerate(groups):
        pts = [pt for p in members for pt in p.frame_points.get(shown, [])]
        drawn = dc.frame_heatmap(pts, size, title=f"{name}：{labels.camera_word(shown)}から見た位置") \
            if size else ""
        with cols[k % len(cols)]:
            _caption(f"{name}：{labels.camera_word(shown)}から見た位置")
            if drawn:
                _img(drawn, f"{name}：{labels.camera_word(shown)}から見た位置（画面の中、コートの上ではありません）")
                _twin(dash.frame_cells(pts, size))
            else:
                _note(f"{name}：この画面の中の位置の記録がありません。")


def _frame_camera(board: dash.Board, people: list[dash.Person]) -> str | None:
    """The camera whose picture the heatmap shows: picked when there are several, else the
    one with the most samples."""
    totals = {c: sum(len(p.frame_points.get(c, [])) for p in people) for c in board.cameras}
    cams = [c for c in board.cameras if totals.get(c) and c in board.frame_sizes]
    if not cams:
        return None
    if len(cams) == 1:
        return cams[0]
    best = max(cams, key=lambda c: totals[c])
    return st.radio("カメラ", cams, index=cams.index(best), format_func=labels.camera_word,
                    horizontal=True, key=f"pfb_dash_cam_{board.submission_id}")


def _distance_notes(people: list[dash.Person]) -> None:
    with_d = [p for p in people if p.distance_m is not None]
    if not with_d:
        _note("動いた距離は出せません（コートの上の記録が30秒より短いか、ありません）。")
        return
    for p in with_d:
        shown = dash.distance_shown(p.distance_m)
        if shown is None:
            continue
        err = (f"位置の誤差 約±{round(p.position_error_cm)} cm" if p.position_error_cm
               else "位置の誤差の見積もりはありません")
        _note(f"{p.name}：動いた距離 約{shown} m（推定。{labels.camera_word(p.court_camera)}の映像から、"
              f"{err}）")


def _scenes(repo: Repo, board: dash.Board, person: dash.Person | None) -> None:
    _head(SCENES)
    _note(SCENES_NOTE)
    found = dash.scenes(board)
    if person is not None:
        found = [s for s in found if any(name == person.name for name, _ in s.who)]
    if not found:
        _note(NO_MOMENTS)
        return
    for k, s in enumerate(found):
        who = "、".join(
            f"{escape(name)}（" + "・".join(
                f"{_glyph(c)}{escape(c.label if c else '種類がわからない')} {n}回" for c, n in cats)
            + "）" for name, cats in s.who)
        st.html(f'<p class="pfb-dash-row"><span class="pfb-dash-row__when">{escape(s.label)}'
                f'</span><span class="pfb-dash-row__what">{who}</span></p>')
        first = next((m for m in board.moments if m.t_s == s.first_t), None)
        if first is not None:
            _jump(repo, board, first.camera, first.t_s, f"scene_{k}")


def _highlights(repo: Repo, board: dash.Board) -> None:
    found = dash.highlights(board)
    if not found:
        return
    _head(HIGHLIGHTS)
    _note(HIGHLIGHTS_NOTE)
    for k, m in enumerate(found):
        _moment_row(repo, board, m, f"hl_{k}")


def _about(board: dash.Board) -> None:
    lines = [
        "映っていた時間：人の枠が記録された時間です（1.5秒より短いすき間はつなげています）。"
        "画面に映っていた時間で、出ていた時間とはかぎりません。カメラが何台あっても、重なる時間は1回だけ数えます。",
        LIVE_ABOUT if board.live else
        "よかった場面：よかった点として判断された場面です。承認前の参加者は、いまの下書きのもとになった場面です。",
        "評価した参加者だけを数えています。評価しない人、データを消した人は出てきません。",
    ]
    if board.sports and board.court:
        lines.append("動いた距離：コートの上の位置から出した推定です。1台のカメラの映像だけから出し、"
                     "カメラどうしを足すことはありません。コートの上の記録が30秒より短い人には出しません。")
    elif board.sports:
        lines.append("位置：コートの位置がわからない動画のため、画面の中の位置だけを出しています。"
                     "カメラが動くと、同じ場所でも画面の中の位置は変わります。")
    if board.several_cameras:
        lines.append(dash.SEVERAL_CAMERAS + "。カメラごとの時間は人ごとの表のCSVにあります。")
    lines.append(LIVE_CHIP_ABOUT if board.live else
                 "見どころと場面の一覧で「承認前」とあるものは、まだ承認していない文章のもとになった場面です。")
    if board.untyped:
        lines.append(f"種類がわからない場面（{board.untyped}件）は、種類ごとの数に入れていません。")
    if board.media_removed:
        lines.append("動画を消した作業では、映っていた時間・位置・場面の動画は残っていません。")
    with st.expander(ABOUT):
        st.markdown("\n".join(f"- {theme.L(line)}" for line in lines))
