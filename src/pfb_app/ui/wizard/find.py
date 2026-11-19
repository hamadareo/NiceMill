"""「作業をさがす」: the row above the home screen's works that narrows them by a word (a work's
name, a person's name, a sport, a day such as 11/14), a sport and a period.

It appears from `FIND_FROM` works on (with one work there is nothing to narrow). The choices
live in the session (the widgets' keys), so the home screen's own refresh while analyses run
keeps them. 「条件を消す」 puts everything back. What passes is decided by services.search; the
home screen draws only those works, in its usual groups. The tab title, the chime and the
refresh watch every work, whatever is shown.
"""
import streamlit as st

from pfb_app.services import search
from pfb_app.storage.db import Repo
from pfb_app.ui.wizard.progress import WorkProgress

FIND_FROM = 2
TEXT, SPORT, PERIOD = "pfb_find_text", "pfb_find_sport", "pfb_find_period"
LABEL = "作業をさがす"
PLACEHOLDER = "作業の名前・人の名前・競技・日付（例: 11/14）"
SPORT_LABEL, PERIOD_LABEL = "競技", "期間"
CLEAR = "条件を消す"
NOTHING = "条件に合う作業はありません。"
FOUND = "{n}件の作業が見つかりました（全部で{total}件）。"
HELP = ("作業の名前、人の名前、競技、日付（11/14、11月14日、2026-11-14）で絞り込めます。"
        "空白で区切ると、すべてに合う作業だけが残ります。")


def _clear() -> None:
    st.session_state[TEXT] = ""
    st.session_state[SPORT] = ""
    st.session_state[PERIOD] = search.ALL


def query() -> search.Query:
    """The coach's choices as they are in the session."""
    sport = st.session_state.get(SPORT)
    period = st.session_state.get(PERIOD)
    return search.Query(str(st.session_state.get(TEXT) or ""), sport if sport else None,
                        period if period in search.PERIODS else search.ALL)


def _sport_options(facts) -> dict[str, str]:
    """{choice: label}; "" is any sport (a widget's choice is never None)."""
    options: dict[str, str] = {"": search.ANY_SPORT_LABEL}
    options.update(dict(search.sports_present(facts)))
    if search.has_non_sport(facts) and len(options) > 1:
        options[search.NO_SPORT] = search.NO_SPORT_LABEL
    return options


def apply(repo: Repo, works: list[WorkProgress]) -> list[WorkProgress]:
    """The works that pass the choices in the session (all of them below FIND_FROM works or
    when nothing is chosen). Read before the row is drawn: a widget's value is in the session
    from the start of the run."""
    if len(works) < FIND_FROM:
        return works
    q = query()
    if not q.active:
        return works
    keep = search.filter_ids(search.load(repo), q)
    return [w for w in works if w.assignment_id in keep]


def draw(repo: Repo, works: list[WorkProgress], shown: list[WorkProgress]) -> None:
    """The row (the word, the sport, the period) and, while something is chosen, how many
    works passed with 「条件を消す」."""
    if len(works) < FIND_FROM:
        return
    options = _sport_options(search.load(repo).values())
    if st.session_state.get(SPORT, "") not in options:     # that sport's last work is gone
        st.session_state.pop(SPORT, None)
    text_col, sport_col, period_col = st.columns([3, 2, 2], vertical_alignment="bottom")
    with text_col:
        st.text_input(LABEL, key=TEXT, placeholder=PLACEHOLDER, help=HELP,
                      icon=":material/search:")
    with sport_col:
        if len(options) > 1:
            st.selectbox(SPORT_LABEL, list(options), key=SPORT,
                         format_func=lambda k: options[k])
    with period_col:
        st.selectbox(PERIOD_LABEL, list(search.PERIODS), key=PERIOD,
                     format_func=lambda k: search.PERIODS[k])
    if not query().active:
        return
    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        st.caption(FOUND.format(n=len(shown), total=len(works)) if shown else NOTHING)
    with right:
        st.button(CLEAR, key="pfb_find_clear", on_click=_clear, type="tertiary",
                  icon=":material/close:")
