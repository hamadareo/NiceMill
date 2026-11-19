"""Step 4 (documents: 3), 解析を待つ: how far the work's analysis has come, in plain words.

First the progress panel (progress_panel.draw): the overall bar with its percentage and the
rough remaining time (目安), the stages with a chip each (the running one with its own bar and
note), the review line, a compact bar per group (or per camera), and, when work waits with
nobody to do it, a note to call the administrator. Then per group (or document): its status
and reason; a group that has stopped says why and links to the admin pages (「くわしく見る」),
and when the derivation names an action for it (a failed job: 「もう一度ためす」, which puts
every failed job of the work back in the queue through services.retry; nobody to evaluate:
「映っている人を自動で選び直す」; no behavior recorded: 「もう一度解析する」), that action is
offered as the screen's only primary button; for videos one line per student. The title reads
「解析しています」 while a blocking job of the work is active, else 「解析が止まりました」 when a
group has stopped. While the work waits on this step there is nothing to press: the secondary
「ホームへ戻る（解析は続きます）」 (「ホームへ戻る」 when it has stopped) takes the coach home; on a
later work's step 4, 「続ける」 opens the step where it is the coach's turn.

The board redraws itself every 10 seconds (live.watch), from one snapshot of the work per
tick. When the work moves on (a group's step or status changes, or the percentage passes a
multiple of 10) the page reruns; when someone has become ready to check, the screen follows
the derived step, which is 確認して承認.
"""
from html import escape

import streamlit as st

from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import usable
from pfb_app.services import auto_people, retry, review
from pfb_app.storage.db import Repo
from pfb_app.ui.common import flash
from pfb_app.ui.wizard import live, progress_panel, router, theme
from pfb_app.ui.wizard.progress import RERUN, RESELECT, RETRY, SubProgress, View

_FAILED = {"failed", "failed_retryable"}

HELP = [
    "解析は自動で進みます。この画面は10秒ごとに新しくなり、確認できる人が出ると次の段階へ進みます。",
    "うまくいかないとき：「止まっています」と出たグループは、理由を読んで、出ているボタン"
    "（「もう一度ためす」など）を押します。何度ためしても止まるときや、長い間まったく進まないときは、"
    "管理者に知らせてください。くわしい様子は管理用の画面で見られます。",
    "待っている間に画面を閉じても、解析は続きます。ホームからいつでも戻れます。",
]


def student_states(repo: Repo, sub: dict) -> list[tuple[str, str, str]]:
    """[(name, state word, chip kind)] of a video group's students."""
    jobs = live.latest_jobs(repo, sub["id"])
    out = []
    for subj in repo.list_subjects(sub["id"]):
        if not is_evaluated(subj):
            continue
        job = jobs.get((Stage.pfb.value, None, subj["id"]))
        status = job["status"] if job else None
        run = repo.latest_run(sub["id"], Stage.pfb, subject_id=subj["id"])
        fresh = usable(run) and not review.is_stale_chain(repo, run)
        if status == "running":
            state = ("文章を書いています", "processing")
        elif status == "waiting_gpu":
            state = ("順番を待っています", "waiting_gpu")
        elif status == "queued":
            state = ("文章の順番を待っています", "processing")
        elif fresh and subj["approval_status"] == "approved" \
                and subj["approved_run_id"] == run.id:
            state = ("承認済み", "approved")
        elif fresh and (run.output or {}).get("pfb_status") == "generated":
            state = ("確認できます", "your_turn")
        elif fresh:
            state = ("場面が少ない", "insufficient")
        elif (run is not None and run.status == "failed") or status in _FAILED:
            state = ("作れませんでした", "stopped")
        else:
            state = ("前の処理を待っています", "insufficient")
        out.append((subj["display_name"], *state))
    return out


def _chip(kind: str, text: str) -> str:
    return theme.status_chip(kind, text)


def _group_html(p: SubProgress, students: str = "") -> str:
    """A group's line (name, chip, reason), then its students when given."""
    return (f'<section class="pfb-live" aria-label="{escape(p.label)}">'
            f'<div class="pfb-group"><span class="pfb-group__name">{escape(p.label)}</span>'
            f'{router.sub_chip(p.status)}<span class="pfb-group__reason">{escape(theme.L(p.reason))}'
            f'</span></div>{students}</section>')


def _students_html(repo: Repo, sub: dict, kind: Kind) -> str:
    if kind != Kind.video:
        return ""
    rows = "".join(
        f'<li class="pfb-student"><span class="pfb-student__name">{escape(name)}</span>'
        f'{_chip(chip, word)}</li>' for name, word, chip in student_states(repo, sub))
    if not rows:
        return ""
    return f'<ul class="pfb-students" aria-label="{escape(theme.L("参加者ごとの様子"))}">{rows}</ul>'


def _signature(aid: int):
    def signature(view: View) -> tuple:
        progress = view.progress
        review_step = progress.number("review")
        return (router.default_step(progress),
                tuple((s.submission_id, s.step, s.status) for s in progress.subs),
                review_step if progress.reachable(review_step) else None,
                progress_panel.bucket(progress))
    return signature


def _moved(old: tuple, new: tuple) -> None:
    """Someone became ready to check: open 確認して承認 (or, if it cannot be opened, follow
    the derived step)."""
    before = {sid for sid, step, _ in old[1] if step in ("review", "pdf")}
    if any(step in ("review", "pdf") and sid not in before for sid, step, _ in new[1]):
        router.go(new[2])
        router.state()["submission_id"] = None     # the step changes: no group kept


def _board(aid: int, settings):
    def draw(view: View) -> None:
        repo, progress = view.repo, view.progress
        kind = Kind(progress.kind)
        subs = {s["id"]: s for s in repo.list_submissions(aid)}
        progress_panel.draw(view, settings)
        for p in progress.subs:
            students = _students_html(repo, subs[p.submission_id], kind)
            if p.status == "stopped" and p.action not in ("reselect", "rerun"):
                # (a reselect or a rerun is the way out by itself)
                st.html(_group_html(p))
                router.detail_link("status", router.DETAILS)
                if students:
                    st.html(students)
            else:
                st.html(_group_html(p, students))
    return draw


_ACTIONS = {"retry": RETRY, "reselect": RESELECT, "rerun": RERUN}
RESELECTING = "映っている人を選び直しています"
RESELECTED = "映っている人を選び直しました。解析を続けます。"
RERUN_STARTED = "もう一度解析を始めました。"
CANNOT_RESELECT = "このグループでは選び直せません。"
CANNOT_RERUN = "いまはもう一度解析できません。"
RETRY_STARTED = "もう一度ためしています。解析が進むと、この画面に出ます。"
HOME_WHILE_RUNNING = "ホームへ戻る（解析は続きます）"
HOME_STOPPED = "ホームへ戻る"
STOPPED_TITLE = "解析が止まりました"


def retry_button(repo: Repo, aid: int, key: str, primary: bool, disabled: bool = False) -> None:
    """「もう一度ためす」 for a stopped work (every failed job of it, services.retry): a plain
    message when nothing can be retried, else a flash and a rerun. Also used on step 3."""
    if not st.button(theme.L(RETRY), key=key, type="primary" if primary else "secondary",
                     width="stretch", disabled=disabled, icon=":material/refresh:"):
        return
    try:
        retry.retry_work(repo, aid)
    except retry.CannotRetry as e:
        # pressed again while the retry runs: kind news, not an error
        (st.info if str(e) == retry.RETRYING else st.error)(theme.L(str(e)))
        return
    flash(RETRY_STARTED)
    router.go(None)            # follow the work to where it moves on
    st.rerun()


def _act(screen: router.Screen, p: SubProgress) -> None:
    """Run a stopped group's action, then rerun the page (a plain message when busy)."""
    if screen.pending:      # the work is being deleted: nothing new starts
        return
    try:
        if p.action == "reselect":
            with st.spinner(theme.L(RESELECTING)):
                auto_people.reselect(screen.repo, screen.settings, p.submission_id)
            done = RESELECTED
        else:
            auto_people.rerun_behavior(screen.repo, p.submission_id)
            done = RERUN_STARTED
    except review.BusyError:
        st.error(theme.L(auto_people.BUSY if p.action == "reselect" else auto_people.BUSY_RERUN))
        return
    except review.ValidationError:
        st.error(theme.L(CANNOT_RESELECT if p.action == "reselect" else CANNOT_RERUN))
        return
    flash(done)
    st.rerun()


def _action_buttons(screen: router.Screen, groups: list[SubProgress]) -> None:
    """One button per stopped group with an action (one 「もう一度ためす」 for the whole work,
    first); the first is the screen's primary."""
    retrying = [p for p in groups if p.action == "retry"]
    if retrying:
        retry_button(screen.repo, screen.progress.assignment_id, "pfb_wait_retry", primary=True,
                     disabled=screen.pending)
    groups = [p for p in groups if p.action != "retry"]
    several = len(screen.progress.subs) > 1
    for n, p in enumerate(groups, start=len(retrying[:1])):
        if several:
            st.caption(p.label)
        if st.button(theme.L(_ACTIONS[p.action]), key=f"pfb_wait_{p.action}_{p.submission_id}",
                     type="primary" if n == 0 else "secondary", width="stretch",
                     disabled=screen.pending):
            _act(screen, p)


def render(screen: router.Screen) -> None:
    progress = screen.progress
    aid = progress.assignment_id
    waiting = progress.number("wait") == screen.derived
    actions = [p for p in progress.subs if p.status == "stopped" and p.action in _ACTIONS]
    running = any(screen.snap.has_blocking_jobs(p.submission_id) for p in progress.subs)
    halted = not running and any(p.status == "stopped" for p in progress.subs)
    if not waiting:
        title = "解析の様子"
    else:
        title = STOPPED_TITLE if halted else "解析しています"
    if waiting and halted and any(p.action == "retry" for p in actions):
        guidance = "「もう一度ためす」を押すと、止まったところから解析をやり直します。"
    elif waiting and halted:
        guidance = "止まった理由を読んで、直してください。"
    elif waiting:
        guidance = "解析が終わった人から確かめられるようになり、この画面も自動で次へ進みます。"
    else:
        guidance = "グループごとの解析の様子です。"
    with theme.step_card(title, guidance):
        live.watch("wait", screen.view, _signature(aid), _board(aid, screen.settings), _moved)
        _action_buttons(screen, actions)
        if waiting:     # nothing to press here: the analysis goes on without the coach
            if st.button(theme.L(HOME_STOPPED if halted else HOME_WHILE_RUNNING),
                         key="pfb_wait_next", icon=":material/home:", width="stretch"):
                router.go_home()
                st.rerun()
        elif st.button(theme.L("続ける"), key="pfb_wait_next",
                       type="secondary" if actions else "primary", disabled=screen.pending,
                       help=theme.L("あなたの番の段階へ進みます")):
            router.go(None)
            st.rerun()
    theme.help_expander(HELP)
