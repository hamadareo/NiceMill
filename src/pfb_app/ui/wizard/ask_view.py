"""ききたいことを聞く (pages/ask.py, optional ?work=<id>): the coach asks one video work a
question in Japanese; the answer comes from the records only (services.ask), with the scenes it
stands on.

Reading only: nothing is written to the database. The questions and answers of this
browser session are kept in the session's memory (st.session_state) so the page can show them,
at most HISTORY_MAX of them, and are gone when the session ends or with 「履歴を消す」; nothing
about a question is saved anywhere else. A person taken out of the evaluation or whose data
was removed is not in the record, so never in an answer.

The model is services.ask.make_backend's local served model (`backend_for`; None when none is
set up: the answer then says so). Accounts: ui/app.py registers the page for 管理者 and
評価する人 only, pages/ask.py stops any other run (access.require_page), and the entry link is
only drawn for them.
"""
from __future__ import annotations

import logging
from html import escape

import streamlit as st

from pfb_app.domain import profiles
from pfb_app.domain.models import Kind
from pfb_app.inference.backend import InferenceBackend
from pfb_app.services import ask, dashboard as dash
from pfb_app.storage.db import Repo
from pfb_app.ui import access
from pfb_app.ui.common import get_settings
from pfb_app.ui.wizard import theme

log = logging.getLogger(__name__)

PAGE = "pages/ask.py"
TITLE = "ききたいことを聞く"
LINK = "記録にききたいことを聞く"
GUIDE = ("試合や活動の記録について、ふだんの言葉で聞けます。答えは、解析で記録されたことだけから作り、"
         "もとになった場面を並べます。")
NO_WORK = "聞ける作業がまだありません。動画の作業を解析すると、ここで聞けます。"
NOBODY = "この作業には、まだ評価する人がいません。人を選ぶと、ここで聞けます。"
NOT_FOUND = "この作業は見つかりませんでした（削除された可能性があります）。"
WORK = "作業"
GROUP = "グループ"
QUESTION = "ききたいこと"
PLACEHOLDER = "例：声をかけていたのは誰？"
ASK = "たずねる"
EXAMPLES = "こんなふうに聞けます"
CLEAR = "履歴を消す"
SPINNER = "記録を調べています…"
BASIS = "もとになった場面"
RANGE = "調べた内容"
KEPT = "聞いたことと答えは、ブラウザーのこのタブを閉じるか開き直すと消えます。どこにも保存しません。"
HISTORY_MAX = 20
HISTORY_KEY = "_pfb_ask_history"     # [(work id, group id, stamp, question, Answer)], newest first
WANTED_KEY = "_pfb_ask_wanted"       # the ?work= this session last followed
RECORD_KEY = "_pfb_ask_record"       # (aid, sid, stamp, Record)
HELP = [
    "「ききたいこと」に、ふだんの言葉で書いて「たずねる」を押します。",
    "答えられるのは、人ごとの場面と回数、場面の時間、話し声、承認した文章についてです。"
    "「声をかけていたのは誰？」「Aさんの場面を教えて」「前半のパスは何回？」のように聞けます。",
    "数は、記録をコンピューターが数えた数です。答えには、もとになった場面の時間と人が並びます。",
    "記録にないことや、人どうしを比べて決めること、これからの助言は、答えられません。"
    "そのときは「記録からはわかりません」と出ます。",
    "「前半」「後半」は、解析した動画の前の半分、後ろの半分のことです。",
    "うまく答えられないときは、人の名前や場面の種類を入れて、聞き方を変えてみてください。",
]
CSS = """<style>
.pfb-ask-q { margin: 0.75rem 0 0.25rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-ask-a { margin: 0 0 0.25rem; color: var(--pfb-ink); white-space: pre-wrap; }
.pfb-ask-note { margin: 0.125rem 0; color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-ask-row { margin: 0; padding: 0.375rem 0; border-top: 1px solid var(--pfb-line);
  color: var(--pfb-ink); }
.pfb-ask-row__when { font-weight: 700; margin-right: 0.5rem; }
.pfb-ask-row__who { margin-right: 0.5rem; }
.pfb-ask-row__kind { margin-right: 0.5rem; padding: 0 0.375rem; border: 1px solid var(--pfb-line);
  border-radius: 999px; font-size: 0.8125rem; color: var(--pfb-muted); }
.pfb-ask-row__what { display: block; color: var(--pfb-ink); }
</style>"""


def backend_for(settings) -> InferenceBackend | None:
    """The model a question goes to (tests replace this with a fake one)."""
    return ask.make_backend(settings)


def entry_link(work: dict) -> None:
    """The quiet link from step 6 of a video work to this page, at that work (管理者 and
    評価する人 only: for anyone else the page is not registered)."""
    if work.get("kind") != Kind.video.value or not access.can_edit():
        return
    st.page_link(PAGE, label=theme.L(LINK), icon=":material/forum:",
                 query_params={"work": str(work["id"])})


def _number(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _works(repo: Repo) -> dict[int, str]:
    """The video works that have at least one group, newest first: id -> title."""
    out: dict[int, str] = {}
    for w in sorted(repo.list_assignments(), key=lambda w: -w["id"]):
        if w.get("kind") == Kind.video.value and dash.groups(repo, w["id"]):
            out[w["id"]] = w["title"]
    return out


def _record(repo: Repo, aid: int, sid: int) -> ask.Record:
    key = dash.stamp(repo, aid)
    held = st.session_state.get(RECORD_KEY)
    if held and held[:3] == (aid, sid, key):
        return held[3]
    record = ask.load(repo, aid, sid)
    st.session_state[RECORD_KEY] = (aid, sid, key, record)
    return record


def _stamp(repo: Repo, aid: int) -> str | None:
    try:
        return dash.stamp(repo, aid)
    except Exception:   # noqa: BLE001 - a work that cannot be read has no history to keep
        return None


def _history(repo: Repo) -> list:
    """This session's answers, without those whose work changed since (a person taken out,
    a name, an approval, new runs: dash.stamp): an old answer never outlives the record it
    was made from."""
    held = st.session_state.setdefault(HISTORY_KEY, [])
    now: dict[int, str | None] = {}
    kept = [h for h in held if h[2] == now.setdefault(h[0], _stamp(repo, h[0]))]
    if len(kept) != len(held):
        held[:] = kept
    return held


def _remember(repo: Repo, aid: int, sid: int, question: str, answer: ask.Answer) -> None:
    held = _history(repo)
    held.insert(0, (aid, sid, _stamp(repo, aid), question, answer))
    del held[HISTORY_MAX:]


def _row(e: ask.Evidence) -> str:
    who = f'<span class="pfb-ask-row__who">{escape(e.who)}</span>' if e.who else ""
    kind = (f'<span class="pfb-ask-row__kind">{escape(e.kind_label)}</span>'
            if e.kind_label else "")
    cam = f" {escape(e.camera)}" if e.camera else ""
    if e.kind == "speech":
        kind = '<span class="pfb-ask-row__kind">話し声</span>'
    return (f'<p class="pfb-ask-row"><span class="pfb-ask-row__when">{escape(e.when)}{cam}</span>'
            f'{who}{kind}<span class="pfb-ask-row__what">{escape(e.text)}</span></p>')


def _show(answer: ask.Answer, question: str) -> None:
    st.html(f'<p class="pfb-ask-q">{escape(question)}</p>')
    st.html(f'<p class="pfb-ask-a">{escape(answer.text)}</p>')
    if answer.scope and answer.status == "answered":
        st.html(f'<p class="pfb-ask-note">{RANGE}：{escape(answer.scope)}</p>')
    for note in answer.notes:
        st.html(f'<p class="pfb-ask-note">{escape(note)}</p>')
    if answer.evidence:
        st.html(f'<p class="pfb-ask-note"><strong>{BASIS}</strong></p>'
                + "".join(_row(e) for e in answer.evidence))


def render(repo: Repo) -> None:
    theme.inject_theme()
    st.html(CSS)
    works = _works(repo)
    if not works:
        theme.set_domain("school")
        with theme.step_card(TITLE, GUIDE, key="ask"):
            st.info(NO_WORK)
        return
    wanted = _number(st.query_params.get("work"))
    ids = list(works)
    if wanted in works and st.session_state.get(WANTED_KEY) != wanted:
        st.session_state["pfb_ask_work"] = wanted        # a link to this work: open it
        st.session_state[WANTED_KEY] = wanted
    if st.session_state.get("pfb_ask_work") not in works:
        st.session_state.pop("pfb_ask_work", None)
    with theme.step_card(TITLE, GUIDE, key="ask"):
        aid = st.selectbox(WORK, ids, format_func=works.get, key="pfb_ask_work")
        try:
            work = repo.get_assignment(aid)
            found = dash.groups(repo, aid)
        except KeyError:
            st.info(NOT_FOUND)
            return
        theme.set_domain(profiles.of_work(dict(work)))
        labels_ = dict(found)
        sid = found[0][0]
        if len(found) > 1:
            sid = st.selectbox(theme.L(GROUP), list(labels_), format_func=labels_.get,
                               key=f"pfb_ask_group_{aid}")
        try:
            record = _record(repo, aid, sid)
        except KeyError:
            st.info(NOT_FOUND)
            return
        if not record.people:
            st.info(theme.L(NOBODY))
            return
        st.caption(KEPT)
        _form(repo, aid, sid, record)
    mine = [h for h in _history(repo) if h[:2] == (aid, sid)]
    if mine:
        with theme.step_card("これまでに聞いたこと", "新しい順に並んでいます。", key="ask_history"):
            for _, _, _, question, answer in mine:
                _show(answer, question)
                st.divider()
            if st.button(CLEAR, key="pfb_ask_clear", type="tertiary"):
                st.session_state[HISTORY_KEY] = []
                st.rerun()
    theme.help_expander(HELP)


def _form(repo: Repo, aid: int, sid: int, record: ask.Record) -> None:
    pending = st.session_state.pop("_pfb_ask_pending", None)
    with st.form(f"pfb_ask_form_{aid}_{sid}", clear_on_submit=True, border=False):
        text = st.text_input(QUESTION, max_chars=ask.MAX_QUESTION_CHARS, placeholder=PLACEHOLDER,
                             key=f"pfb_ask_text_{aid}_{sid}")
        sent = st.form_submit_button(ASK, type="primary", icon=":material/send:")
    examples = ask.examples(record)
    if examples:
        st.caption(EXAMPLES)
        cols = st.columns(len(examples))
        for k, (col, example) in enumerate(zip(cols, examples)):
            with col:
                if st.button(example, key=f"pfb_ask_example_{k}", width="stretch"):
                    st.session_state["_pfb_ask_pending"] = example
                    st.rerun()
    question = pending if pending else (text if sent else None)
    if question is None:
        return
    with st.spinner(SPINNER):
        answer = ask.ask(backend_for(get_settings()), record, question)
    _remember(repo, aid, sid, question, answer)
    st.rerun()
