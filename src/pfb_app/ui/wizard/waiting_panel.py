"""「確認待ちの人」: on the home screen, who waits for a look in each work, and 「読んで確認した人を
まとめて承認」.

Opened with 「確認待ちの人をまとめて見る」 (shown only to a person who can edit, and only while a
work has a group at the step 「確認して承認」). For each work the people whose text waits are
listed with their text. A person whose text needs no second thought (services.pending_review:
the bar of step 5's 「全員まとめて承認」) has the box 「読んで確認した」, empty at first: the box is
the coach's own word that the text was read. Everyone else is listed with the plain reason and
is approved one by one in the work (「開く」).

Approving goes in two presses, never one:

1. 「選んだ{n}人を承認する」 does not approve: it keeps what the screen showed (each text's
   run id) and shows who and which text will be approved, with the way to undo it.
2. 「承認する」 approves them with services.review.bulk_approve (the audit entries, the run-id
   check and the gates are the service's), or 「やめる」 drops the choice. A text that changed
   in between, or no longer passes the bar, is not approved and is said so in plain words.

Afterwards the screen says how many were approved, who was left and why, and how to undo an
approval (services.pending_review.UNDO). The press's callback re-checks the sign-in
(ui.access) before it writes: a viewer's or an expired session approves nothing.

「あとから訂正」 works have nothing to approve. Their counterpart is 「誤りの報告がある人」: the people
whose text someone reported wrong (services.pending_review.reported), each with the reason, the
text and 「開く」 (the work, where step 5 「文を読む・直す」 corrects it or closes the report).
Nothing is done in bulk there.
"""
from html import escape

import streamlit as st

from pfb_app.services import pending_review
from pfb_app.ui import access, keyed
from pfb_app.ui.common import flash, get_repo
from pfb_app.ui.wizard import router
from pfb_app.ui.wizard.progress import WorkProgress

OPEN_KEY = "_pfb_wait_open"
ASK = "_pfb_wait_ask"          # the choice waiting for the second press: [(subject, run, name, work, text)]
REPORT = "_pfb_wait_report"    # (approved count, [(name, work, why)]) shown once
PICK = "pfb_wait_pick_{sid}"

OPEN = "確認待ちの人をまとめて見る（{n}件の作業）"
CLOSE = "とじる"
INTRO = ("文章を読んで、問題がなければ「読んで確認した」にチェックを入れてください。"
         "チェックした人だけを、まとめて承認できます。"
         "気になる点のある人は、作業を開いて一人ずつ確かめます。")
READ = "読んで確認した"
ASK_BUTTON = "選んだ{n}人を承認する"
NONE_PICKED = "チェックした人がいません。読んだ人にチェックを入れてください。"
NOBODY = "いま、確認を待っている人はいません。"
CONFIRM_HEAD = "次の{n}人の文章を承認します。よろしいですか。"
CONFIRM_NOTE = ("承認すると、その人の文章はPDFに入れられます。"
                "次にためしてみること（ヒント）も、気になる点がなければ文章と一緒に確定します。"
                "承認は、あとから一人ずつ直したり取り消したりできます。")
GO, CANCEL = "承認する", "やめる"
DONE = "{n}人を承認しました。"
DONE_NONE = "承認した人はいませんでした。"
LEFT = "承認しなかった人（作業を開いて、一人ずつ確かめてください）"
OPEN_WORK = "開く"
EDIT_ONLY = "見るだけの人は、承認できません。"
REPORTS = "誤りの報告がある人（{n}人）"           # 「あとから訂正」 works
REPORTS_INTRO = ("「この文は間違い」と報告された文です。作業を開いて「文を読む・直す」で、"
                 "文を直すか、確かめて報告を閉じてください。")
REPORT_WHY = "報告の理由：{reason}"

CSS = """<style>
.pfb-wait__work { margin: 0.75rem 0 0.25rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-wait__who { margin: 0.5rem 0 0; font-weight: 700; color: var(--pfb-ink); }
.pfb-wait__group { font-weight: 500; color: var(--pfb-muted); }
.pfb-wait__text { margin: 0.125rem 0 0.25rem; padding: 0.5rem 0.75rem; white-space: pre-wrap;
  overflow-wrap: anywhere; border-left: 3px solid var(--pfb-line); color: var(--pfb-ink); }
.pfb-wait__why { margin: 0.125rem 0 0.25rem; color: var(--pfb-muted); font-size: 0.9375rem; }
.pfb-wait__done { margin: 0.25rem 0; color: var(--pfb-ink); font-weight: 700; }
.pfb-wait__undo { margin: 0.5rem 0 0; color: var(--pfb-muted); font-size: 0.875rem; }
</style>"""


def candidates(works: list[WorkProgress], pending: set[int]) -> list[int]:
    """The works with a group at 「確認して承認」 where it is the coach's turn (a cheap read of
    what the home screen already holds), in the home screen's order. 「あとから訂正」 works
    have nothing to approve: they are never offered."""
    return [w.assignment_id for w in works
            if not w.complete and w.assignment_id not in pending and not w.live
            and any(s.step == "review" and s.status == "your_turn" for s in w.subs)]


def _toggle() -> None:
    st.session_state[OPEN_KEY] = not st.session_state.get(OPEN_KEY, False)
    st.session_state.pop(ASK, None)


def _ask(offered: dict[int, tuple[int | None, str, str, str]]) -> None:
    """First press: keep who is picked (from what the screen offered) for the second."""
    if access.refuse_unless_edit():
        return
    picked = [(sid, *offered[sid]) for sid in offered
              if st.session_state.get(PICK.format(sid=sid))]
    if picked:
        st.session_state[ASK] = picked
    else:
        flash(NONE_PICKED, kind="warning")


def _cancel() -> None:
    st.session_state.pop(ASK, None)


def _go() -> None:
    """Second press: approve what the first press kept, through the review service."""
    if access.refuse_unless_edit():
        st.session_state.pop(ASK, None)
        return
    asked = st.session_state.pop(ASK, None) or []
    if not asked:
        return
    repo = get_repo()
    try:
        result = pending_review.approve(repo, {sid: run for sid, run, *_ in asked})
    finally:
        repo.conn.close()
    names = {sid: (name, work) for sid, _run, name, work, _text in asked}
    for sid in result.approved:
        st.session_state.pop(PICK.format(sid=sid), None)
    st.session_state[REPORT] = (len(result.approved),
                                [(*names.get(sid, ("", "")), why) for sid, why in result.skipped])


def _report() -> None:
    done = st.session_state.pop(REPORT, None)
    if not done:
        return
    n, left = done
    rows = "".join(f"<li>{escape(name)}さん（{escape(work)}）: {escape(why)}</li>"
                   for name, work, why in left)
    tail = f"<p class=\"pfb-wait__why\">{escape(LEFT)}</p><ul>{rows}</ul>" if left else ""
    st.html(f'<div role="status"><p class="pfb-wait__done">'
            f'{escape(DONE.format(n=n) if n else DONE_NONE)}</p>{tail}'
            f'<p class="pfb-wait__undo">{escape(pending_review.UNDO)}</p></div>')


def _confirm() -> None:
    asked = st.session_state.get(ASK) or []
    rows = "".join(f"<li>{escape(work)}: {escape(name)}さん</li>" for _s, _r, name, work, _t in asked)
    st.html(f'<p class="pfb-wait__done">{escape(CONFIRM_HEAD.format(n=len(asked)))}</p>'
            f'<ul>{rows}</ul><p class="pfb-wait__undo">{escape(CONFIRM_NOTE)}</p>')
    go, cancel, _ = st.columns([1, 1, 4])
    with go:
        st.button(GO, key="pfb_wait_go", on_click=_go)
    with cancel:
        st.button(CANCEL, key="pfb_wait_cancel", on_click=_cancel, type="tertiary")


def _list(repo, ids: list[int]) -> None:
    lists = pending_review.waiting(repo, ids)
    if not lists:
        st.caption(NOBODY)
        return
    st.caption(INTRO)
    offered: dict[int, tuple[int | None, str, str, str]] = {}
    for work in lists:
        with keyed.container(key=f"pfb_wait_work_{work.assignment_id}", border=True):
            head, open_ = st.columns([5, 1], vertical_alignment="center")
            with head:
                st.html(f'<p class="pfb-wait__work">{escape(work.title)}</p>')
            with open_:
                if st.button(OPEN_WORK, key=f"pfb_wait_open_{work.assignment_id}",
                             width="stretch"):
                    router.open_work(work.assignment_id)
                    st.rerun()
            for person in work.people:
                group = f'<span class="pfb-wait__group">（{escape(person.group)}）</span>'
                st.html(f'<p class="pfb-wait__who">{escape(person.name)}さん {group}</p>')
                if person.text:
                    st.html(f'<p class="pfb-wait__text">{escape(person.text)}</p>')
                if person.ready:
                    st.checkbox(READ, key=PICK.format(sid=person.subject_id))
                    offered[person.subject_id] = (person.run_id, person.name, work.title,
                                                  person.text)
                else:
                    st.html(f'<p class="pfb-wait__why">{escape(person.why or "")}</p>')
    if offered:
        n = sum(1 for sid in offered if st.session_state.get(PICK.format(sid=sid)))
        st.button(ASK_BUTTON.format(n=n), key="pfb_wait_ask", on_click=_ask, args=(offered,),
                  disabled=n == 0)
    # a box of someone who is no longer offered must not stay ticked for a later text
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith("pfb_wait_pick_")]:
        if int(key.rsplit("_", 1)[1]) not in offered:
            st.session_state.pop(key, None)


def _reports(repo, works: list[WorkProgress], pending: set[int]) -> None:
    """「誤りの報告がある人」 of the 「あとから訂正」 works (nothing when there is none)."""
    live = {w.assignment_id: w for w in works if w.live and w.assignment_id not in pending}
    found = pending_review.reported(repo, list(live))
    if not found:
        return
    n = sum(len(w.people) for w in found)
    st.html(CSS)
    with keyed.expander(REPORTS.format(n=n), key="pfb_wait_reports", icon=":material/report:"):
        st.caption(REPORTS_INTRO)
        for work in found:
            head, open_ = st.columns([5, 1], vertical_alignment="center")
            with head:
                st.html(f'<p class="pfb-wait__work">{escape(work.title)}</p>')
            with open_:
                if st.button(OPEN_WORK, key=f"pfb_wait_open_live_{work.assignment_id}",
                             width="stretch"):
                    # 「文を読む・直す」 (step 5 of a video work, 4 of a documents one)
                    router.open_work(work.assignment_id,
                                     live[work.assignment_id].number("review"))
                    st.rerun()
            for person in work.people:
                group = f'<span class="pfb-wait__group">（{escape(person.group)}）</span>'
                why = REPORT_WHY.format(reason=person.reason) + (
                    f"（{person.note}）" if person.note else "")
                st.html(f'<p class="pfb-wait__who">{escape(person.name)}さん {group}</p>'
                        f'<p class="pfb-wait__why">{escape(why)}</p>'
                        + (f'<p class="pfb-wait__text">{escape(person.text)}</p>' if person.text
                           else ""))


def render(repo, works: list[WorkProgress], pending: set[int]) -> None:
    """On the home screen, under the works' groups' heading area: the entry, and the list
    while it is open. Nothing for a person who cannot edit or when nobody is at the check.
    「あとから訂正」 works: 「誤りの報告がある人」 (`_reports`)."""
    if not access.can_edit():
        return
    _reports(repo, works, pending)
    ids = candidates(works, pending)
    if not ids:                      # everyone is done (the last press's report is still said)
        st.session_state.pop(OPEN_KEY, None)
        st.session_state.pop(ASK, None)
    if ids or st.session_state.get(REPORT):
        st.html(CSS)
        _report()
    if not ids:
        return
    opened = bool(st.session_state.get(OPEN_KEY, False))
    st.button(CLOSE if opened else OPEN.format(n=len(ids)), key="pfb_wait_toggle",
              on_click=_toggle, icon=":material/checklist:", type="secondary")
    if not opened:
        return
    if st.session_state.get(ASK):
        _confirm()
        return
    _list(repo, ids)
