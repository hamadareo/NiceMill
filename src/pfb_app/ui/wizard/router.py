"""The wizard's router: which work is open and which of its steps is shown.

`st.session_state["wizard"] = {"assignment_id": int | None, "step": int | None,
"submission_id": int | None}`.

- No work open: the home screen (or the drop zone of a new work, `step == 1`).
- A work open: the step is `step` when the teacher picked one on the stepper and it can be
  opened; otherwise it is derived from the DB on every run (`turn_step`, else `current`), so
  it moves on by itself as the work progresses and never drifts from the data.
- The open work is mirrored in the URL (`?work=<id>`, or `?work=new`), so a browser reload,
  which starts a new session, reopens the same work at its derived step. A link in the
  sidebar drops the query string; the router takes that as going home.
"""
from dataclasses import dataclass
from html import escape

import streamlit as st

from pfb_app.config import Settings
from pfb_app.domain.vocab import words
from pfb_app.services import deletion
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import KIND_LABELS, flash, get_repo, get_settings, show_flash
from pfb_app.ui.wizard import bars, progress_panel, theme
from pfb_app.ui.wizard.progress import (SubProgress, View, WorkProgress, held_submissions,
                                        snapshot, work_progress)

STATE = "wizard"
QUERY = "work"
NEW = "new"
_SHOWN = "_pfb_wizard_url"    # what the router last wrote to ?work= (None: nothing yet)
ADDING = "_pfb_adding"        # assignment id whose step 2 shows the registration form again
REVIEW_STAMP = "_pfb_review_stamp"   # (subject id, name) just approved: step 5 stamps it once
REVIEW_ACTION = "_pfb_review_action"  # (action, subject id, run id) a step-5 button accepted
REVIEW_BULK = "_pfb_review_bulk"      # ({subject id: run id} offered, [(id, why) left]): 全員まとめて承認
REVIEW_BULK_DONE = "_pfb_review_bulk_done"   # what that press did: shown once on step 5
DROP_DOCS = "_pfb_drop_docs"  # [(name, bytes)] dropped on the drop zone, until the work exists

# The quiet detail pages (「管理用の画面（くわしい操作）」), by path relative to ui/app.py.
DETAIL_PAGES = {
    "assignments": "pages/assignments.py",
    "submissions": "pages/submissions.py",
    "status": "pages/status.py",
    "review": "pages/review.py",
    "export": "pages/export.py",
}
# The 判定支援 pages (sports works' line calls), by path relative to ui/app.py.
LINECALL_PAGES = {"linecall": "pages/linecall_review.py", "live": "pages/linecall_live.py"}


@dataclass
class Screen:
    """What a step screen gets: the repo (for writes), the settings, the work's progress, the
    step shown (1-based) and the page run's view of the work (`view.repo`: the one snapshot
    every reader of this run shares)."""
    repo: Repo
    settings: Settings
    progress: WorkProgress
    step: int
    view: View
    pending: bool = False   # the work is deleted once its running step ends: nothing to press

    @property
    def snap(self) -> Repo:
        """The page run's read-only snapshot of the work."""
        return self.view.repo

    @property
    def key(self) -> str:
        return self.progress.keys[self.step - 1]

    @property
    def label(self) -> str:
        return self.progress.steps[self.step - 1]

    @property
    def derived(self) -> int:
        return default_step(self.progress)


# --- state ----------------------------------------------------------------------------------

def state() -> dict:
    s = st.session_state.get(STATE)
    if not isinstance(s, dict):
        s = {}
    for k in ("assignment_id", "step", "submission_id"):
        s.setdefault(k, None)
    st.session_state[STATE] = s
    return s


def open_work(assignment_id: int | None, step: int | None = None,
              submission_id: int | None = None) -> None:
    """Open a work (None: the home screen). `step` None: the step is derived. Screen-local
    modes of the work left behind (step 2's 「もう1件追加」 form) are dropped, and so are
    documents dropped for a new work unless that new work is what opens."""
    st.session_state.pop(ADDING, None)
    if not (assignment_id is None and step == 1):
        st.session_state.pop(DROP_DOCS, None)
    st.session_state[STATE] = {"assignment_id": assignment_id, "step": step,
                               "submission_id": submission_id}


def go_home() -> None:
    st.session_state.pop(DROP_DOCS, None)
    open_work(None)


def start_new() -> None:
    """Step 1 of a new work: the drop zone, empty."""
    st.session_state.pop(DROP_DOCS, None)
    open_work(None, 1)


def go(step: int | None) -> None:
    """Show `step` of the open work; None: follow the derived step."""
    state()["step"] = step


def default_step(progress: WorkProgress) -> int:
    """The step a work opens at: where it is the teacher's turn, else the earliest step."""
    return progress.turn_step or progress.current


def _url_value(s: dict) -> str | None:
    if s["assignment_id"] is not None:
        return str(s["assignment_id"])
    return NEW if s["step"] == 1 else None


def _sync_url() -> None:
    """Reconcile the session with ?work=: a reload brings the URL only, a sidebar link
    brings no query string; otherwise the session is mirrored into the URL."""
    s = state()
    url = st.query_params.get(QUERY)
    shown = st.session_state.get(_SHOWN)
    if url != shown:
        if url is None:                   # left through a link: home
            go_home()
        elif url == NEW:
            if shown is not None and shown.isdigit():
                go_home()   # back from a work to the 「新しく始める」 form: no second copy
            else:
                start_new()
        elif url.isdigit():               # a reload, or the browser's back and forward
            if s["assignment_id"] != int(url):
                open_work(int(url))
        else:
            go_home()
    value = _url_value(state())
    if value is None:
        st.query_params.pop(QUERY, None)
    else:
        st.query_params[QUERY] = value
    st.session_state[_SHOWN] = value


# --- shared pieces --------------------------------------------------------------------------

DETAILS = "くわしく見る（管理用の画面）"   # the quiet link of a stopped group or work


def detail_link(page: str, label: str = "管理用の画面で開く", query: dict | None = None) -> None:
    """A quiet link to one of the old pages (the open work is preselected there; `query`
    can preselect more, e.g. {"subject": id} on 確認・承認). Only 管理者 have those pages."""
    if not access.is_admin():
        return
    st.page_link(DETAIL_PAGES[page], label=label, icon=":material/open_in_new:",
                 query_params=query)


def work_meta(progress: WorkProgress) -> str:
    """「動画・2グループ・参加者8人」(sports: 選手)"""
    parts = [KIND_LABELS[progress.kind]]
    groups = len(progress.subs)
    if progress.kind == "video":
        if groups:
            parts.append(f"{groups}グループ")
        if progress.total:
            parts.append(f"{words(progress.wording).people}{progress.total}人")
    elif groups:
        parts.append(f"{groups}件")
    return "・".join(parts)


def work_chip(progress: WorkProgress) -> str:
    """The work's status as a chip: a word, a mark and a colour."""
    if progress.complete:
        return theme.status_chip("done", "完了")
    if progress.stopped:
        return theme.status_chip("stopped", "止まっています")
    if progress.your_turn:
        return theme.status_chip("your_turn", "あなたの番")
    if any(s.status == "processing" for s in progress.subs):
        return theme.status_chip("processing", "解析中")
    return theme.status_chip("waiting_gpu", "順番待ち")


SUB_CHIP_TEXT = {"your_turn": "あなたの番", "processing": "解析中", "waiting_gpu": "順番待ち",
                 "stopped": "止まっています", "done": "完了"}


def sub_chip(status: str) -> str:
    return theme.status_chip(status, SUB_CHIP_TEXT[status])


def group_list(subs: list[SubProgress]) -> None:
    """A short list of groups (or documents): name, status in words, and why."""
    if not subs:
        return
    items = "".join(
        f'<li class="pfb-group"><span class="pfb-group__name">{escape(s.label)}</span>'
        f'{sub_chip(s.status)}<span class="pfb-group__reason">{escape(theme.L(s.reason))}</span></li>'
        for s in subs)
    st.html(f'<ul class="pfb-groups" aria-label="{escape(theme.L("グループごとの様子"))}">{items}</ul>')


def shown_group(screen: Screen, step_key: str) -> SubProgress:
    """The group a per-group step shows: the one picked with 「このグループへ」, else the
    first on this step whose turn it is, else the first on this step, else the first."""
    subs = screen.progress.subs
    picked = state().get("submission_id")
    for s in subs:
        if s.submission_id == picked:
            return s
    return next((s for s in subs if s.step == step_key and s.status == "your_turn"),
                next((s for s in subs if s.step == step_key), subs[0]))


def group_picker(subs: list[SubProgress], shown: SubProgress) -> None:
    """With several groups: one row each (name, status, why) and 「このグループへ」."""
    if len(subs) < 2:
        return
    for s in subs:
        text, action = st.columns([3, 1], vertical_alignment="center")
        with text:
            st.html(f'<div class="pfb-group"><span class="pfb-group__name">{escape(s.label)}'
                    f'</span>{sub_chip(s.status)}<span class="pfb-group__reason">'
                    f'{escape(theme.L(s.reason))}</span></div>')
        with action:
            here = s.submission_id == shown.submission_id
            if st.button("表示中" if here else "このグループへ", key=f"pfb_group_{s.submission_id}",
                         disabled=here, width="stretch"):
                state()["submission_id"] = s.submission_id
                st.rerun()


def mini_stepper_html(progress: WorkProgress) -> str:
    """The small stepper of a home card (0-based for theme.stepper_html)."""
    current = default_step(progress) - 1
    done = {d - 1 for d in progress.done}
    return f'<div class="pfb-mini">{theme.stepper_html(progress.steps, current, done)}</div>'


# --- the stepper you can click --------------------------------------------------------------

def _step_state(n: int, shown: int, progress: WorkProgress) -> tuple[list[str], str, str]:
    classes = ["pfb-step"]
    done, current = n in progress.done, n == shown
    if done:
        classes.append("pfb-step--done")
    if current:
        classes.append("pfb-step--current")
    if current:
        state_text = "いまここ"
    elif done:
        state_text = "済み"
    elif progress.reachable(n):
        state_text = "開けます"
    else:
        state_text = "これから"
    return classes, ("✓" if done else str(n)), state_text


def _nav_progress(progress: WorkProgress) -> str:
    """Next to the work's current step: the analysis's percentage while it runs by itself,
    「あなたの番」 while it waits for the coach, else nothing."""
    a = progress.analysis
    if a is None:
        return ""
    if a.active:
        return bars.nav_percent_html(progress_panel.held_percent(progress.assignment_id, a))
    return bars.nav_turn_html() if a.waiting_for_you else ""


def compact_nav_html(progress: WorkProgress, shown: int) -> str:
    """The phone stepper: 「5/6 確認して承認」 and a thin bar (shown only below 40rem)."""
    count = len(progress.steps)
    label = escape(progress.steps[shown - 1])
    return (f'<div class="pfb-navline"><span class="pfb-navline__text">{shown}/{count} {label}'
            f'</span>{_nav_progress(progress) if shown == progress.current else ""}</div>'
            f'<div class="pfb-navbar" role="progressbar" aria-label="作業の段階" aria-valuemin="1" '
            f'aria-valuemax="{count}" aria-valuenow="{shown}" '
            f'aria-valuetext="{shown}/{count} {label}">'
            f'<div class="pfb-navbar__fill" style="width:{shown * 100 // count}%"></div></div>')


def _go_step(progress: WorkProgress, n: int) -> None:
    go(None if n == default_step(progress) else n)
    state()["submission_id"] = None   # another step: its own first group
    st.rerun()


def step_nav(progress: WorkProgress, shown: int) -> None:
    """The stepper across the top; every step that can be opened is a button. Below 40rem
    the CSS swaps it for a compact line, a thin bar and 「ほかの段階」 (same steps)."""
    count = len(progress.steps)
    with keyed.container(key="pfb_stepcompact"):
        st.html(compact_nav_html(progress, shown))
        with st.popover("ほかの段階", width="stretch"):
            for n, label in enumerate(progress.steps, start=1):
                _, _, state_text = _step_state(n, shown, progress)
                if st.button(f"{n}. {label}（{state_text}）", key=f"pfb_navm_{n}",
                             type="tertiary", width="stretch",
                             disabled=not progress.reachable(n) or n == shown):
                    _go_step(progress, n)
    with keyed.container(key="pfb_stepnav"):
        cols = st.columns(count, gap=None)
        for n, (col, label) in enumerate(zip(cols, progress.steps), start=1):
            classes, mark, state_text = _step_state(n, shown, progress)
            last = " pfb-navstep--last" if n == count else ""
            with col:
                st.html(f'<div class="pfb-navstep {" ".join(classes)}{last}" aria-hidden="true">'
                        f'<span class="pfb-step__mark">{mark}</span></div>')
                clicked = st.button(label, key=f"pfb_nav_{n}", type="tertiary",
                                    disabled=not progress.reachable(n) or n == shown,
                                    help=f"{n}. {label}（{state_text}）",
                                    width="stretch")
                st.html(f'<div class="pfb-navstate{" pfb-navstate--current" if n == shown else ""}">'
                        f'{state_text}{_nav_progress(progress) if n == progress.current else ""}'
                        f'</div>')
            if clicked:
                _go_step(progress, n)


def _header(progress: WorkProgress) -> None:
    left, right = st.columns([1, 5], vertical_alignment="center")
    with left:
        if st.button("ホームへ", key="pfb_home", icon=":material/home:"):
            go_home()
            st.rerun()
    with right:
        st.html(f'<div class="pfb-workhead"><span class="pfb-workhead__title">'
                f'{escape(progress.title)}</span>'
                f'<span class="pfb-workhead__meta">{escape(work_meta(progress))}</span>'
                f'{work_chip(progress)}</div>')


def _new_header() -> None:
    if st.button("ホームへ", key="pfb_home", icon=":material/home:"):
        go_home()
        st.rerun()


# --- screens --------------------------------------------------------------------------------

def _screens():
    from pfb_app.ui.wizard import (step_assignment, step_files, step_pdf,  # noqa: PLC0415
                                   step_people, step_review, step_wait)
    return {"assignment": step_assignment.render, "files": step_files.render,
            "people": step_people.render, "wait": step_wait.render,
            "review": step_review.render, "pdf": step_pdf.render}


def _screen_of(screen: Screen):
    """The screen function of the step shown. Step 1 of a video work with a group still held
    for its content check is that group's check card; step 5 of a 「あとから訂正」 work is
    「文を読む・直す」 (step_correct); otherwise the step's own screen."""
    if screen.key == "assignment" and screen.progress.kind == "video" \
            and held_submissions(screen.snap, screen.progress.assignment_id):
        from pfb_app.ui.wizard import step_check   # noqa: PLC0415
        return step_check.render
    if screen.key == "review" and screen.progress.live:
        from pfb_app.ui.wizard import step_correct   # noqa: PLC0415
        return step_correct.render
    return _screens()[screen.key]


def render() -> None:
    """The wizard page: the home screen, or one step of the open work."""
    theme.inject_theme()
    theme.set_domain("school")      # a work's own domain is set below, once it is read
    repo, settings = get_repo(), get_settings()
    _sync_url()
    s = state()
    if s["assignment_id"] is None and s["step"] == 1 and not access.can_edit():
        go_home()                       # 見るだけの人 start no new work
        st.query_params.pop(QUERY, None)
        st.session_state[_SHOWN] = None
        s = state()
    if s["assignment_id"] is None:
        if s["step"] == 1:
            from pfb_app.ui.wizard import step_drop         # noqa: PLC0415
            _new_header()
            step_drop.render_new(repo, settings)
            _help_footer("drop")
        else:
            from pfb_app.ui.wizard import home              # noqa: PLC0415
            home.render(repo)
        return
    try:
        snap = snapshot(repo, s["assignment_id"])      # one read of the work for this run
        progress = work_progress(snap, s["assignment_id"])
    except KeyError:
        flash("この作業は見つかりませんでした（削除された可能性があります）。")
        go_home()
        st.query_params.pop(QUERY, None)
        st.session_state[_SHOWN] = None
        st.rerun()
    st.session_state["current_assignment"] = progress.assignment_id   # for the detail pages
    step = s["step"]
    if step is None or not progress.reachable(step):
        step = default_step(progress)
    theme.set_domain(progress.wording)   # the work's profile words
    from pfb_app.ui.wizard import done_band, notify    # noqa: PLC0415
    notify.for_work(progress)                         # the tab title, and the chime
    done_band.for_work(repo, progress)                # 「〇〇の解析が終わりました」 (other works)
    if not access.can_edit():                         # 見るだけの人: the results, read only
        from pfb_app.ui.wizard import view_only        # noqa: PLC0415
        _header(progress)
        view_only.render(snap, progress)
        return
    _header(progress)
    step_nav(progress, step)
    show_flash()
    pending = deletion.is_pending(repo, progress.assignment_id)
    if pending:
        st.info(theme.L(deletion.RUNNING))
    screen = Screen(repo, settings, progress, step, View(snap, progress), pending)
    if screen.key != "review":
        st.session_state.pop(REVIEW_STAMP, None)   # an approval stamp is never kept for later
        st.session_state.pop(REVIEW_ACTION, None)  # nor a step-5 press it never acted on
        st.session_state.pop(REVIEW_BULK, None)
        st.session_state.pop(REVIEW_BULK_DONE, None)
    show = _screen_of(screen)
    show(screen)
    _help_footer("check" if show.__module__.endswith("step_check") else screen.key,
                 progress.live)


def _help_footer(key: str, live: bool = False) -> None:
    """The step's one-line tip and the link to 「はじめての方へ」 (ui/wizard/help)."""
    from pfb_app.ui.wizard import help as guide          # noqa: PLC0415
    guide.step_footer(key, live)
