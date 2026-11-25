"""Step 5's 「次にためしてみること」: each person's practice tips and the group's, to read, change
and confirm (services.tips).

- Under a person's feedback (to check, or approved): a section, open while the person has tips.
  The box holds the tips, one a line (three at most); the coach may change, add to or empty it.
  The model's tips, left as they are, are confirmed by the approval itself (not when their
  wording has a fault, nor when the text was written again, nor after the coach's deletion:
  the screen says so, and 「ヒントを確定する」 is the coach's own look). A box the coach changed
  is confirmed by 「承認する」 too (`confirm_shown`), when its wording is clean;
  「ヒントを確定する」 confirms after a change; 「ヒントを渡さない」 empties the
  box and confirms, so no PDF carries tips for this person. Only confirmed tips reach the PDFs.
  A person without tips gets the section closed, with an empty box to write in.
- Under the list of people: the group's tips, the same, confirmed by the coach with its own
  button (they are made from several people's texts, so no single approval carries them); the
  screen says that texts not yet approved are material too. Without tips (the job never ran) or
  with outdated ones, 「チームのヒントを作る」 / 「チームのヒントを作り直す」 asks the worker.
- Faults of the wording (a name, a grade, a number, a time, an order instead of a suggestion) are
  shown in plain words under the box as the coach types; they advise and never refuse.
- Every press is bound to the tips shown (their run id): a newer set is refused with a plain
  message, and the screen keeps this step.
- A 「あとから訂正」 work (`live`, step 5 「文を読む・直す」): no approval confirms the tips; the
  model's tips that an approval would have confirmed go into the PDF as they are
  (services.tips.auto_ok / team_auto_ok), the screen says whether they do, and the same
  buttons change or drop them later. No word of approval is shown.
"""
from typing import Callable

import streamlit as st

from pfb_app.domain.models import Stage
from pfb_app.pipeline import checks
from pfb_app.services import review, tips
from pfb_app.services.review import BusyError
from pfb_app.services.tips import Tips, TipsError
from pfb_app.ui import keyed
from pfb_app.ui.common import flash
from pfb_app.ui.wizard import theme

CONFIRM = "ヒントを確定する"
DROP = "ヒントを渡さない"
TEAM_CONFIRM = "このヒントで確定する"
TEAM_DROP = "チームのヒントは渡さない"
PERSON_NOTE = ("この参加者の文章の根拠になった場面から作った案です。確かめて、直してもかまいません。"
               "1行に1つ、{n}つまで書けます。")
TEAM_NOTE = ("何人かの文章に共通することから作った案です。確かめて、直してもかまいません。"
             "1行に1つ、{n}つまで書けます。")
EMPTY_NOTE = "この参加者のヒントは作られませんでした。渡したいときは、ここに書けます。"
TEAM_EMPTY_NOTE = "ヒントは作られませんでした。渡したいときは、ここに書けます。"
DONE = "確定しました。このヒントがPDFに載ります。"
DROPPED = "確定しました。ヒントはPDFに載りません。"
# 「あとから訂正」 works: the same presses, worded by what the PDF does (never 渡す)
DROP_LIVE = "ヒントをPDFに入れない"
TEAM_DROP_LIVE = "チームのヒントはPDFに入れない"
EMPTY_NOTE_LIVE = "この参加者のヒントは作られませんでした。入れたいときは、ここに書けます。"
TEAM_EMPTY_NOTE_LIVE = "ヒントは作られませんでした。入れたいときは、ここに書けます。"
PENDING_REVIEW = "よければ「承認する」を押してください。承認と一緒に確定します。"
PENDING_APPROVED = "まだ確定していません。PDFには載りません。「ヒントを確定する」を押してください。"
TEAM_PENDING = "まだ確定していません。PDFには載りません。「このヒントで確定する」を押してください。"
HAS_FAULT = "気になる点があるため、自動では確定しません。確かめてから押してください。"
WORDING_TAIL = "（このままでも確定できます）"
TEXT_CHANGED = ("文章を書き直したため、ヒントは自動では確定しません。"
                "確かめて「ヒントを確定する」を押してください。")
TEAM_MATERIAL = "まだ承認していない人の文章も材料です。"
TEAM_MAKE = "チームのヒントを作る"
TEAM_REMAKE = "チームのヒントを作り直す"
TEAM_QUEUED = "チームのヒントを作っています。できると、ここに出ます。"
TEAM_ASKED = "チームのヒントを作りはじめました。"
OUTDATED = "解析が更新されたため、ヒントは古くなりました。この人だけ作り直すと、新しいヒントができます。"
TEAM_OUTDATED = "解析が更新されたため、チームのヒントは古くなりました。"
# 「あとから訂正」 (live): what goes into the PDF without a press, in words without 承認
LIVE_AUTO = "AIが作った案です。このままPDFに載ります。違っていれば、直して「ヒントを確定する」を押してください。"
LIVE_FAULT = "気になる点があるため、このままではPDFに載りません。確かめて「ヒントを確定する」を押してください。"
LIVE_TEXT_CHANGED = "文を直したため、このヒントはPDFに載りません。確かめて「ヒントを確定する」を押してください。"
LIVE_REPORTED = "誤りの報告を確認中のため、このヒントはPDFに載りません。"
LIVE_HELD = "このままではPDFに載りません。確かめて「ヒントを確定する」を押してください。"
LIVE_TEAM_AUTO = "AIが作った案です。このままPDFに載ります。違っていれば、直して「このヒントで確定する」を押してください。"
LIVE_TEAM_HELD = "このままではPDFに載りません。確かめて「このヒントで確定する」を押してください。"
_BOX = "pfb_tips_box_{}_{}"       # (person id or "team", the run shown)


def _key(who, now: Tips) -> str:
    return _BOX.format(who, now.run_id or 0)


def _lines(now: Tips) -> str:
    return "\n".join(now.items)


def _box(label: str, who, now: Tips) -> str:
    return st.text_area(theme.L(label), value=_lines(now), key=_key(who, now), height=110,
                        label_visibility="collapsed")


def _faults(repo, submission_id: int, text: str) -> None:
    for line in tips.wording(repo, submission_id, checks.split_tips(text)):
        st.warning(theme.L(line + WORDING_TAIL))


def _refuse(error: Exception) -> None:
    st.error(theme.L(error.args[0] if error.args else tips.NEWER))


def _saved(pin: Callable[[], None], message: str) -> None:
    pin()
    flash(message)
    st.rerun()


def _state_line(now: Tips, approved: bool, faulty: bool, pending: str) -> None:
    if now.confirmed:
        st.caption(theme.L(DONE if now.items else DROPPED))
    elif faulty:
        st.caption(theme.L(HAS_FAULT))
    elif now.text_changed:
        st.caption(theme.L(TEXT_CHANGED))
    else:
        st.caption(theme.L(PENDING_APPROVED if approved else pending))


def _live_line(repo, sid: int, subject_id: int, now: Tips, faulty: bool) -> None:
    """A 「あとから訂正」 person's tips: whether the PDF carries them as they are."""
    from pfb_app.services import corrections   # noqa: PLC0415
    if now.confirmed:
        st.caption(theme.L(DONE if now.items else DROPPED))
    elif tips.auto_ok(repo, sid, subject_id, now):
        st.caption(theme.L(LIVE_AUTO))
    elif corrections.open_report(repo, subject_id) is not None:
        st.caption(theme.L(LIVE_REPORTED))
    elif faulty:
        st.caption(theme.L(LIVE_FAULT))
    elif now.text_changed:
        st.caption(theme.L(LIVE_TEXT_CHANGED))
    else:
        st.caption(theme.L(LIVE_HELD))


def person(screen, student, pin: Callable[[], None], *, live: bool = False) -> None:
    """The section under a person's feedback (state review or approved; `live`: step 5 of a
    「あとから訂正」 work)."""
    repo, sid = screen.snap, student.subject["submission_id"]
    now = tips.view(repo, sid, student.id)
    if now.stale:
        st.html(f'<h3 class="pfb-subhead">{tips.PERSON_TITLE}</h3>')
        st.caption(theme.L(OUTDATED))
        return
    approved = student.state == "approved"
    with keyed.expander(theme.L(tips.PERSON_TITLE), key=f"pfb_tips_open_{student.id}",
                        expanded=bool(now.items), icon=":material/lightbulb:"):
        st.caption(theme.L(PERSON_NOTE.format(n=tips.MAX_PERSON) if now.items else
                           EMPTY_NOTE_LIVE if live else EMPTY_NOTE))
        text = _box(tips.PERSON_TITLE, student.id, now)
        shown = checks.split_tips(text)
        faulty = bool(tips.wording(repo, sid, shown))
        _faults(repo, sid, text)
        if (now.items or now.confirmed or shown) and live:
            _live_line(repo, sid, student.id, now, faulty and not now.confirmed)
        elif now.items or now.confirmed or shown:
            _state_line(now, approved, faulty and not now.confirmed, PENDING_REVIEW)
        with keyed.container(horizontal=True, key=f"pfb_tips_actions_{student.id}"):
            confirm = st.button(theme.L(CONFIRM), key=f"pfb_tips_confirm_{student.id}_{now.run_id}",
                                disabled=not (shown or now.items) or screen.pending)
            drop = st.button(theme.L(DROP_LIVE if live else DROP),
                             key=f"pfb_tips_drop_{student.id}_{now.run_id}",
                             type="tertiary", disabled=not now.items or screen.pending)
    if confirm or drop:
        try:
            tips.confirm(screen.repo, sid, student.id, text if confirm else "", now.run_id)
        except (TipsError, BusyError) as e:
            _refuse(e)
            return
        _saved(pin, DONE if confirm and shown else DROPPED)


def confirm_shown(screen, student) -> bool:
    """Before 「承認する」: a box the coach changed is confirmed, when its wording is clean (a
    changed box with a fault is left for 「ヒントを確定する」); a box left as it is is the
    approval's own business (`tips.confirm_on_approval`). False: refused (more lines than
    allowed, a newer set), said on screen; the approval must not go on."""
    repo, sid = screen.repo, student.subject["submission_id"]
    now = tips.view(repo, sid, student.id)
    key = _key(student.id, now)
    if now.stale or key not in st.session_state:
        return True            # not on screen: the approval confirms the model's own tips
    text = st.session_state[key]
    lines = checks.split_tips(text)
    if lines == now.items:
        return True            # as it was made (or as confirmed before): nothing for the coach to save
    faulty = bool(lines) and len(lines) <= tips.MAX_PERSON and bool(tips.wording(repo, sid, lines))
    try:
        # a fault in the coach's change: kept as their words, theirs to confirm on purpose
        # (the model's own tips are not confirmed in its place)
        tips.confirm(repo, sid, student.id, text, now.run_id, draft=faulty)
    except (TipsError, BusyError) as e:
        _refuse(e)
        return False
    return True


def team(screen, unit, pin: Callable[[], None], *, live: bool = False) -> None:
    """The group's tips under the list of people: shown once someone's text can be checked
    (`live`: step 5 of a 「あとから訂正」 work)."""
    if len(unit.submission_ids) != 1 or not any(
            s.state in ("review", "approved") for s in unit.students):
        return
    repo, sid = screen.snap, unit.submission_ids[0]
    now = tips.view(repo, sid, None)
    domain = repo.get_assignment(repo.get_submission(sid)["assignment_id"]).get("domain")
    title = tips.team_title(domain)
    ask = None
    if now.stale or now.run is None:
        if tips.team_queued(repo, sid):
            ask = TEAM_QUEUED
        elif tips.can_ask_team(repo, sid):
            ask = TEAM_REMAKE if now.stale else TEAM_MAKE
    if now.stale:
        st.html(f'<h3 class="pfb-subhead">{title}</h3>')
        st.caption(theme.L(TEAM_OUTDATED))
        _ask(screen, sid, ask, pin)
        return
    with keyed.expander(title, key=f"pfb_tips_team_open_{sid}", expanded=bool(now.items),
                        icon=":material/groups:"):
        st.caption(theme.L(TEAM_NOTE.format(n=tips.MAX_TEAM) if now.items else
                           TEAM_EMPTY_NOTE_LIVE if live else TEAM_EMPTY_NOTE))
        text = _box(title, f"team{sid}", now)
        shown = checks.split_tips(text)
        _faults(repo, sid, text)
        if now.items and not live and any(s.state == "review" for s in unit.students):
            st.caption(theme.L(TEAM_MATERIAL))
        _ask(screen, sid, ask, pin)
        if (now.items or now.confirmed) and live:
            st.caption(theme.L((DONE if now.items else DROPPED) if now.confirmed else
                               LIVE_TEAM_AUTO if tips.team_auto_ok(repo, sid, now) else
                               LIVE_TEAM_HELD))
        elif now.items or now.confirmed:
            _state_line(now, False, False, TEAM_PENDING)
        with keyed.container(horizontal=True, key=f"pfb_tips_team_actions_{sid}"):
            confirm = st.button(theme.L(TEAM_CONFIRM), key=f"pfb_tips_team_confirm_{sid}_{now.run_id}",
                                disabled=not shown or screen.pending)
            drop = st.button(theme.L(TEAM_DROP_LIVE if live else TEAM_DROP),
                             key=f"pfb_tips_team_drop_{sid}_{now.run_id}",
                             type="tertiary", disabled=not now.items or screen.pending)
    if confirm or drop:
        try:
            tips.confirm(screen.repo, sid, None, text if confirm else "", now.run_id)
        except (TipsError, BusyError) as e:
            _refuse(e)
            return
        _saved(pin, DONE if confirm and shown else DROPPED)


def _ask(screen, sid: int, ask: str | None, pin: Callable[[], None]) -> None:
    """The group's tips are being made (a line), or can be asked for (a button)."""
    if ask is None:
        return
    if ask == TEAM_QUEUED:
        st.caption(theme.L(TEAM_QUEUED))
        return
    if st.button(theme.L(ask), key=f"pfb_tips_team_ask_{sid}", disabled=screen.pending):
        tips.queue_team(screen.repo, sid)
        _saved(pin, TEAM_ASKED)
