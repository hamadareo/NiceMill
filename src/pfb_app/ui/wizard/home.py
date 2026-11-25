"""The home screen: every work with where it stands and the one thing to do next.

The works come in three groups (UX round 2, #7): 「あなたの番」 (the coach's turn, and stopped
works), 「解析中」 (with the toggle 「終わったら音で知らせる」), and 「終わった作業（n件）」, folded.
While a work's analysis runs, its card shows a thin bar with the step it is at and its
percentage (「したことの書き出し・63%」) and a chip word; under it (also before the analysis starts,
while it waits) its place in the queue (「前に2件の作業があります」) and the time left (「あと約15分
（目安）」, services.eta), or 「目安はまだ出せません」 while there are too few earlier analyses; no bar
once it is done. Below 40rem the cards drop their small stepper.
「削除」 and 「名前を変える」 sit in each card's 「その他」. The screen keeps to one primary button,
like the step screens: 「続ける」 on the first work that waits for the coach, or 「新しく始める」
when nothing waits. Under the reason, a card shows its 「データの扱い」 lines: the consent record
(「同意：…」, set in 「その他」→「同意の記録」) and what 「動画を自動で消す」 removes when. The tab title, the chime and the refresh while works are analysed come
from notify.
"""
import re
from html import escape

import streamlit as st

from pfb_app.domain import vocab
from pfb_app.services import assignments, consent, deletion, eta, retention, retry
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import flash, get_settings, show_flash
from pfb_app.ui.wizard import (bars, done_band, find, notify, progress_panel, router, theme,
                               waiting_panel)
from pfb_app.ui.wizard.analysis import RUNNING, STOPPED, WAITING_GPU
from pfb_app.ui.wizard.progress import RETRY, WorkProgress, all_work

DELETING = "_pfb_deleting"   # the assignment id whose card shows the delete confirm area
RENAMED = "_pfb_renamed_"    # + id: renamed; the name field starts again from the saved name
SAVED_CONSENT = "_pfb_consent_saved_"   # + id: the consent fields start again from the record
_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")   # ":" too: no :emoji:


def md_escape(text: str) -> str:
    """User text inside a markdown message (the flash): every markdown character escaped."""
    return _MARKDOWN.sub(r"\\\1", text)


WHAT_GOES = "動画・書類・解析結果・PDF の記録、成長の記録のつながり"
PENDING = "いまの処理が終わると削除されます"   # a card whose work waits to be deleted

TURN, RUNNING, FINISHED = "あなたの番", "解析中", "終わった作業"
VIEW = "見る"   # a 見るだけの人's way into a work (ui/wizard/view_only)
NOTHING_YET = "まだ作業がありません"
NOTHING_YET_GUIDE = "作業ができると、ここに並びます。見るだけの人として開いています。"
GROWTH, GROWTH_PAGE = "成長の記録", "pages/growth.py"   # one person across works
HELP = [
    "「新しく始める」で動画か書類を入れると、中身を確かめて、一人一人へのフィードバックの文を作ります。"
    "できた文はすぐPDFで受け取れます。違っていたところは、あとから直せます（直した記録が残ります）。",
    "確かめて承認した文だけをPDFに入れたいときは、作業の「活動を決める」の「作業の設定」で"
    "「承認して出す」を選びます（前からある作業は、これまでどおり承認して出します）。",
    "作業（活動・試合）は「あなたの番」「解析中」「終わった作業」に分けて並びます。"
    "「続ける」で、いまやることの画面が開きます。",
    "「終わったら音で知らせる」をオンにすると、解析が終わったときに短い音で知らせます。"
    "この画面のタブの名前でも、いまの様子がわかります。",
    "解析を待つ作業には、残りの時間の目安と、前に待っている作業の数が出ます。"
    "実績が少ないうちは「目安はまだ出せません」と出ます。",
    "解析が終わって確認できる作業があると、画面の上に帯が出ます。",
    "作業が2つ以上あると「作業をさがす」で、名前・人の名前・競技・日付から絞り込めます。",
    "「確認待ちの人をまとめて見る」で、文章を読んで「読んで確認した」にチェックした人を、"
    "まとめて承認できます（承認の前に、もう一度確かめる画面が出ます）。",
    "作業の名前は、カードの「その他」→「名前を変える」で直せます。",
    "いらなくなった作業は、カードの「その他」→「削除」で消せます。消す前に、確認の欄が開きます。",
    "人の同意をどう得たかは、カードの「その他」→「同意の記録」に残せます（任意）。",
    "動画を自動で消す設定と一人分のデータを消す操作は、管理用の画面の「データの扱い」にあります。"
    "設定すると、カードにいつ何が消えるかが出ます。",
    "「成長の記録」で、同じ人のいくつかの作業での記録をまとめて見られます。",
    "表や再試行などの細かい操作は、管理用の画面からいつでも使えます。",
]
# Home only: the small stepper goes on phones (the reason line and the bar stay), the group
# headings and the time left.
CSS = """<style>
@media (max-width: 40rem) { .pfb-mini { display: none; } }
.pfb-homegroup {
  margin: 1.25rem 0 0; padding: 0; font-size: 1.125rem; font-weight: 700;
  color: var(--pfb-ink); font-family: var(--pfb-font-ui);
}
.pfb-homegroup__count { font-weight: 500; color: var(--pfb-muted); }
.pfb-workcard__eta { margin: 0.25rem 0 0; color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-workcard__data { margin: 0.25rem 0 0; color: var(--pfb-muted); font-size: 0.875rem; }
</style>"""


def _start_button(primary: bool) -> None:
    if not access.can_edit():          # 見るだけの人 start nothing
        return
    if st.button("新しく始める", key="pfb_new", type="primary" if primary else "secondary",
                 icon=":material/add:"):
        router.start_new()
        st.rerun()


def _card_bar(work: WorkProgress) -> str:
    """The thin bar 「したことの書き出し・63%」 with its chip word while the analysis has started and is
    not finished (none before it starts, once it is done, or for a work without groups)."""
    a = work.analysis
    if a is None or a.finished:
        return ""
    if a.fraction <= 0 and not any(s.status in (RUNNING, WAITING_GPU, STOPPED) for s in a.stages):
        return ""
    current = a.current
    name = vocab.localize(current.name if current is not None else a.review.name, work.wording)
    return bars.card_bar_html(progress_panel.held_percent(work.assignment_id, a), name,
                              progress_panel.status_of(a))


def time_left_text(seconds: float) -> str:
    """「あと約15分（目安）」 (rounded as the waiting screen's 「残り約15分（目安）」)."""
    return eta.time_left_text(seconds)


def _time_left(repo: Repo, work: WorkProgress) -> str:
    """Under the card's bar while the analysis moves on by itself: the place in the queue
    (「前に2件の作業があります」), the time left (「あと約15分（目安）」, with the waiting when it
    is known), or, without enough earlier analyses, that no estimate can be given yet."""
    a = work.analysis
    if a is None or not a.active:
        return ""
    est = eta.estimate(repo, work.assignment_id)
    return "".join(f'<p class="pfb-workcard__eta">{escape(line)}</p>'
                   for line in eta.lines(est))


def data_lines(repo: Repo, work: WorkProgress, policy: retention.Policy) -> list[str]:
    """The card's 「データの扱い」 lines: the consent record and what the automatic deletion
    removes when (none when nothing is recorded or set)."""
    try:
        row = repo.get_assignment(work.assignment_id)
    except KeyError:
        return []
    return [line for line in (consent.card_text(row),
                              retention.card_line(repo, row, policy, settings=repo.settings))
            if line]


def _data_html(repo: Repo, work: WorkProgress, policy: retention.Policy | None) -> str:
    lines = data_lines(repo, work, policy or retention.load(repo))
    return "".join(f'<p class="pfb-workcard__data">{escape(line)}</p>' for line in lines)


def _work_card(repo: Repo, work: WorkProgress, primary: bool, pending: bool,
               policy: retention.Policy | None = None) -> None:
    """`pending`: the work is to be deleted once its running step ends (no 「その他」 then).
    `policy`: the retention setting, read once per screen."""
    with keyed.container(border=True, key=f"pfb_work_{work.assignment_id}"):
        main, action = st.columns([5, 1], vertical_alignment="center")
        with main:
            bar = _card_bar(work)
            st.html(f'<div class="pfb-workcard"><h3 class="pfb-workcard__title">'
                    f'{escape(work.title)}</h3>'
                    f'<span class="pfb-workcard__meta">{escape(router.work_meta(work))}</span>'
                    f'{router.work_chip(work)}</div>'
                    f'<p class="pfb-workcard__reason">{escape(vocab.localize(work.reason, work.wording))}</p>'
                    + bar + _time_left(repo, work)
                    + _data_html(repo, work, policy))
            st.html(router.mini_stepper_html(work))     # hidden below 40rem (CSS)
        with action:
            _card_actions(repo, work, primary, pending)
        if not pending and st.session_state.get(DELETING) == work.assignment_id:
            _delete_area(repo, work.assignment_id)


def _card_actions(repo: Repo, work: WorkProgress, primary: bool, pending: bool) -> None:
    """The card's right side (below it on a phone): 「続ける」, what a stopped work offers,
    then 「その他」."""
    edit = access.can_edit()
    if st.button("続ける" if edit else VIEW, key=f"pfb_open_{work.assignment_id}",
                 type="primary" if primary else "secondary",
                 width="stretch", disabled=pending):
        router.open_work(work.assignment_id)
        st.rerun()
    if work.stopped and edit:
        _retry_button(repo, work, pending)
    if pending:
        st.html(f'<p class="pfb-delete__pending">{escape(theme.L(PENDING))}</p>')
    elif edit:
        _more(repo, work)


def _more(repo: Repo, work: WorkProgress) -> None:
    """「その他」: 「名前を変える」 (a name field and 「この名前にする」), then the quiet 「削除」,
    which opens the confirm area under the card."""
    aid = work.assignment_id
    if st.session_state.pop(f"{RENAMED}{aid}", None) is not None:
        # just renamed: the field starts again from the saved (cleaned) name; dropping its
        # state before it exists avoids Streamlit's "value and Session State" warning
        st.session_state.pop(f"pfb_rename_{aid}", None)
    with st.popover(theme.L("その他"), icon=":material/more_horiz:", width="stretch"):
        name = st.text_input(theme.L("名前を変える"), value=work.title, key=f"pfb_rename_{aid}",
                             max_chars=assignments.MAX_TITLE)
        if st.button(theme.L("この名前にする"), key=f"pfb_rename_go_{aid}",
                     icon=":material/edit:", width="stretch"):
            _rename(repo, aid, name)
        _consent_form(repo, aid)
        if access.is_admin() and st.button(theme.L("削除"), key=f"pfb_del_{aid}", type="tertiary",
                     icon=":material/delete:", width="stretch"):
            st.session_state[DELETING] = aid
            st.rerun()


def _consent_keys(aid: int) -> tuple[str, str]:
    return f"pfb_consent_{aid}", f"pfb_consent_note_{aid}"


def _consent_form(repo: Repo, aid: int) -> None:
    """「同意の記録」 in 「その他」: the choice, a note and 「同意の記録を保存」."""
    try:
        row = repo.get_assignment(aid)
    except KeyError:
        return
    choice_key, note_key = _consent_keys(aid)
    if st.session_state.pop(f"{SAVED_CONSENT}{aid}", None) is not None:
        # just saved: the fields start again from the stored (cleaned) record
        st.session_state.pop(choice_key, None)
        st.session_state.pop(note_key, None)
    stored = row.get("consent") if row.get("consent") in consent.CHOICES else consent.NONE
    options = consent.options()
    choice = st.selectbox(theme.L(consent.TITLE), options, index=options.index(stored),
                          format_func=consent.label, key=choice_key)
    note = st.text_input(theme.L("メモ（入れなくてもかまいません）"),
                         value=row.get("consent_note") or "", key=note_key,
                         max_chars=consent.MAX_NOTE, disabled=not choice)
    if st.button(theme.L(consent.SAVE), key=f"pfb_consent_go_{aid}", icon=":material/verified_user:",
                 width="stretch"):
        try:
            consent.record(repo, aid, choice, note)
        except consent.ConsentError as e:
            flash(str(e), kind="warning")
        except KeyError:
            flash("この作業はすでに削除されています。")
        else:
            st.session_state[f"{SAVED_CONSENT}{aid}"] = True
            flash(consent.SAVED)
        st.rerun()


def _consent_editing(repo: Repo, aid: int) -> bool:
    """The consent fields hold something else than the stored record."""
    choice_key, note_key = _consent_keys(aid)
    if choice_key not in st.session_state and note_key not in st.session_state:
        return False
    try:
        row = repo.get_assignment(aid)
    except KeyError:
        return False
    stored = row.get("consent") if row.get("consent") in consent.CHOICES else consent.NONE
    return (st.session_state.get(choice_key, stored) != stored
            or st.session_state.get(note_key, row.get("consent_note") or "")
            != (row.get("consent_note") or ""))


def _rename(repo: Repo, aid: int, name: str) -> None:
    try:
        saved = assignments.rename_work(repo, aid, name)
    except ValueError:
        flash(assignments.NO_TITLE, kind="warning")
    except KeyError:
        flash("この作業はすでに削除されています。")
    else:
        st.session_state[f"{RENAMED}{aid}"] = saved
        flash(f"名前を「{md_escape(saved)}」に変えました。", raw=True)   # the name as typed
    st.rerun()


def _retry_button(repo: Repo, work: WorkProgress, pending: bool = False) -> None:
    """A stopped work's way on, on its card (lane L1, UX round 2): 「もう一度ためす」 when its stop
    can be retried (services.retry; secondary: home keeps one primary), then the quiet link
    「くわしく見る（管理用の画面）」. After 2 failed retries the card's reason says to tell the
    administrator and only the link is left."""
    aid = work.assignment_id
    if any(s.action == "retry" for s in work.subs) and st.button(
            theme.L(RETRY), key=f"pfb_retry_{aid}", type="secondary", width="stretch",
            icon=":material/refresh:", disabled=pending):
        try:
            retry.retry_work(repo, aid)
        except retry.CannotRetry as e:
            flash(str(e), kind="warning")
        else:
            flash(f"「{md_escape(work.title)}」をもう一度ためしています。", raw=True)   # the title as typed
        st.rerun()
    router.detail_link("status", router.DETAILS)


def _close_delete() -> None:
    st.session_state.pop(DELETING, None)


def _delete_area(repo: Repo, aid: int) -> None:
    """Under the card: what goes, that it cannot be undone, a confirm box, then the red
    「削除する」 (「解析を止めて削除」 while analysis is still waiting or running) and 「やめる」."""
    try:
        plan = deletion.deletion_plan(repo, aid)
    except KeyError:    # deleted meanwhile (another tab)
        _close_delete()
        return
    # Keyed by what is shown: a press on 「削除する」 never becomes 「解析を止めて削除」 when
    # analysis started meanwhile (the rerun shows the other button, unpressed and unticked).
    mode = "stop" if plan.active else "go"
    with keyed.container(key=f"pfb_delete_area_{aid}"):
        notes = ["削除されるもの：" + WHAT_GOES,
                 "もとの場所にある動画や書類のファイルは、そのまま残ります。"]
        if plan.active:
            notes.append("この作業はまだ解析が続いています。削除すると、待っている解析を止めます。")
        st.html(f'<div class="pfb-delete"><p class="pfb-delete__title">'
                f'「{escape(plan.title)}」を削除しますか？</p>'
                + "".join(f"<p>{escape(theme.L(n))}</p>" for n in notes)
                + f'<p class="pfb-delete__undo">{escape(theme.L("この操作は元に戻せません。"))}</p>'
                "</div>")
        sure = st.checkbox(theme.L("削除してよいことを確認しました"),
                           key=f"pfb_delete_ok_{mode}_{aid}")
        go, back, _ = st.columns([2, 1, 3], vertical_alignment="center")
        with go:
            label = "解析を止めて削除" if plan.active else "削除する"
            pressed = st.button(theme.L(label), key=f"pfb_delete_{mode}_{aid}", disabled=not sure,
                                icon=":material/delete:", width="stretch")
        with back:
            if st.button(theme.L("やめる"), key=f"pfb_delete_cancel_{aid}", width="stretch"):
                _close_delete()
                st.rerun()
    if pressed:
        _delete(repo, aid, stop=plan.active)


def _delete(repo: Repo, aid: int, stop: bool) -> None:
    try:
        result = deletion.delete_work(repo, get_settings(), aid, stop_analysis=stop)
    except KeyError:
        flash("この作業はすでに削除されています。")
    except deletion.WorkActive as e:      # analysis started meanwhile: ask again
        flash(str(e), kind="warning")
        st.rerun()
    except deletion.StillRunning as e:   # the worker deletes it when the step ends
        flash(("待っていた解析は止めました。" if e.cancelled else "") + str(e), kind="warning")
    else:
        flash(f"「{md_escape(result.title)}」を削除しました。", raw=True)   # as typed
        if st.session_state.get("current_assignment") == aid:
            st.session_state.pop("current_assignment", None)   # the detail pages' choice
    _close_delete()
    st.rerun()


def render(repo: Repo) -> None:
    works = all_work(repo)
    show_flash()
    if not works and not access.can_edit():
        with theme.step_card(NOTHING_YET, NOTHING_YET_GUIDE, key="home"):
            st.page_link("pages/guide.py", label="はじめての方へ", icon=":material/help:")
        notify.for_home(works)
        return
    if not works:
        with theme.step_card("はじめての作業を始めましょう",
                             "動画か書類を入れると、一人一人へのフィードバックの文が"
                             "できます。", key="home"):
            _start_button(primary=True)
        from pfb_app.ui.wizard import sample_view    # noqa: PLC0415 (onboarding lane)
        sample_view.home_card(repo)
        theme.help_expander(HELP)
        st.page_link("pages/guide.py", label="はじめての方へ", icon=":material/help:")
        notify.for_home(works)
        return
    pending = deletion.pending_deletions(repo)
    policy = retention.load(repo)
    shown = find.apply(repo, works)    # 「作業をさがす」: the works that pass what is chosen
    first_turn = next((w.assignment_id for w in shown
                       if w.your_turn and w.assignment_id not in pending), None)
    turn, running, finished = groups(shown)
    if not access.can_edit():          # 見るだけの人: 「見る」 on the first card is the primary
        first_turn = next((w.assignment_id for w in (*turn, *running, *finished)
                           if w.assignment_id not in pending), None)
    st.html(CSS)
    done_band.for_home(repo, works)    # 「〇〇の解析が終わりました」 at the very top
    head, start = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.html('<div class="pfb-homehead"><h1 class="pfb-homehead__title">ホーム</h1>'
                '<p class="pfb-homehead__guide">あなたの番の作業が上に並びます。'
                '「続ける」で、いまやることの画面が開きます。</p></div>')
    with start:
        _start_button(primary=first_turn is None)
    if access.can_edit():
        st.page_link(GROWTH_PAGE, label=GROWTH, icon=":material/timeline:")
    find.draw(repo, works, shown)
    waiting_panel.render(repo, shown, pending)

    def cards(group: list[WorkProgress]) -> None:
        for work in group:
            _work_card(repo, work, primary=work.assignment_id == first_turn,
                       pending=work.assignment_id in pending, policy=policy)

    if turn:
        _heading(TURN, len(turn))
        cards(turn)
    if running:
        _heading(RUNNING, len(running))
        notify.sound_toggle()
        cards(running)
    if finished:
        with st.expander(f"{FINISHED}（{len(finished)}件）", expanded=False):
            cards(finished)
    theme.help_expander(HELP)
    notify.for_home(works, held=_busy(works, repo))


def _busy(works: list[WorkProgress], repo: Repo | None = None) -> bool:
    """The coach is deleting (the confirm area is open), renaming (a name field holds
    something else than the work's name) or writing a consent record: home must not rerun by
    itself meanwhile."""
    if st.session_state.get(DELETING) is not None:
        return True
    if any(st.session_state.get(f"pfb_rename_{w.assignment_id}", w.title) != w.title
           for w in works):
        return True
    return repo is not None and any(_consent_editing(repo, w.assignment_id) for w in works)


def groups(works: list[WorkProgress]) -> tuple[list, list, list]:
    """(あなたの番: the coach's turn or stopped, 解析中: the rest still going, 終わった作業),
    each in `all_work`'s order."""
    turn = [w for w in works if not w.complete and (w.your_turn or w.stopped)]
    running = [w for w in works if not w.complete and not (w.your_turn or w.stopped)]
    return turn, running, [w for w in works if w.complete]


def _heading(name: str, count: int) -> None:
    st.html(f'<h2 class="pfb-homegroup">{escape(name)}'
            f'<span class="pfb-homegroup__count">（{count}件）</span></h2>')
