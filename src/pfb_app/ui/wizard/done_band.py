"""「〇〇の解析が終わりました。確認できます」: a band at the top of the screen when a work's analysis
has finished and it is the coach's turn to look at the texts.

It shows when this browser session opens or reloads a screen and a work is waiting for the
coach's look (no browser notification API; the chime and the tab title of ui/wizard/notify
stay as they are):

- a work that was being analysed when this session last saw it and now waits for the coach;
- a work seen for the first time in this session that finished within `RECENT_SEC` (the time
  of its newest first draft): the coach who comes back later still sees it, an old one is not
  announced again.

A band stays until the coach presses 「閉じる」 or opens that work (its screens, any step), or
the work is no longer waiting. It is remembered per browser session only; nothing is written
to the database. A work analysed again shows a new band (the finish time is the marker the
dismissal is kept against).

Each band has 「確認する」 (opens the work at the step where it is the coach's turn) and
「閉じる」, both quiet buttons: the screen's one primary button stays where it was. At most
`MAX_SHOWN` bands are drawn; the rest are counted in a line.
"""
import time
from datetime import datetime, timezone
from html import escape
from typing import Callable, Iterable

import streamlit as st

from pfb_app.storage.db import Repo
from pfb_app.ui import keyed
from pfb_app.ui.wizard import notify, router
from pfb_app.ui.wizard.progress import WorkProgress, work_progress

STATE = "_pfb_done_band"        # {"seen": {id: state}, "shown": {id: marker}, "dismissed": {id: marker},
                                #  "titles": {id: title}, "checked": monotonic seconds of the last look}
RECENT_SEC = 12 * 3600
MAX_SHOWN = 3
CHECK_EVERY = 30         # seconds: how often a work screen looks at the other works again
TEXT = "「{title}」の解析が終わりました。確認できます"
OPEN, CLOSE = "確認する", "閉じる"
MORE = "ほかにも、確認できる作業が{n}件あります。ホームで見られます。"

CSS = """<style>
.st-key-pfb_doneband { margin: 2rem 0 0.75rem; }
.pfb-doneband {
  margin: 0; padding: 0.625rem 0.875rem; border-left: 0.3125rem solid var(--pfb-pen);
  background: var(--pfb-mist); color: var(--pfb-ink); font-weight: 700;
  font-family: var(--pfb-font-ui); border-radius: 0.25rem;
}
.pfb-doneband__more { margin: 0.25rem 0 0; color: var(--pfb-muted); font-size: 0.875rem; }
</style>"""


def _state() -> dict:
    s = st.session_state.get(STATE)
    if not isinstance(s, dict):
        s = {}
    for key in ("seen", "shown", "dismissed", "titles"):
        if not isinstance(s.get(key), dict):
            s[key] = {}
    st.session_state[STATE] = s
    return s


def at_review(work: WorkProgress) -> bool:
    """The coach's turn at the step where the finished texts are checked."""
    return (notify.work_state(work) == notify.READY and work.turn_step is not None
            and work.keys[work.turn_step - 1] == "review")


def finished_at(repo: Repo, assignment_id: int) -> str:
    """The time of the work's newest first draft (the marker of its band); '' if none."""
    row = repo.conn.execute(
        "SELECT MAX(r.created_at) FROM stage_runs r JOIN submissions s ON r.submission_id = s.id"
        " WHERE s.assignment_id = ? AND r.stage = 'pfb' AND r.source = 'model'"
        " AND r.status = 'ok'", (assignment_id,)).fetchone()
    return row[0] or ""


def _recent(marker: str, now: datetime) -> bool:
    try:
        return (now - datetime.fromisoformat(marker)).total_seconds() <= RECENT_SEC
    except ValueError:
        return False


def announce(state: dict, works: Iterable[WorkProgress], marker_of: Callable[[int], str],
             now: datetime) -> list[int]:
    """Update the session's memory (`state`: seen, shown, dismissed) with the works as they are
    now, and return the ids to announce: shown, not dismissed, still waiting for the coach.
    `marker_of(id)` is the work's finish time, asked only for a work worth a band."""
    seen, shown, dismissed = state["seen"], state["shown"], state["dismissed"]
    ids = set()
    for work in works:
        aid = work.assignment_id
        ids.add(aid)
        ready = at_review(work)
        before = seen.get(aid)
        if ready and aid not in shown:
            if before == notify.RUNNING:
                shown[aid] = marker_of(aid)
            elif before is None and _recent(marker_of(aid), now):
                shown[aid] = marker_of(aid)
        if not ready:
            shown.pop(aid, None)     # analysed again, or checked: a new finish brings a new band
        seen[aid] = notify.work_state(work)
    for aid in [a for a in shown if a not in ids]:
        shown.pop(aid, None)         # the work is gone
    return [aid for aid, marker in shown.items() if dismissed.get(aid) != marker]


def dismiss(aid: int) -> None:
    state = _state()
    if aid in state["shown"]:
        state["dismissed"][aid] = state["shown"][aid]


def opened(aid: int) -> None:
    """The coach opened the work: its band is done."""
    dismiss(aid)


def _open(aid: int) -> None:
    dismiss(aid)
    router.open_work(aid)


def _monotonic() -> float:
    return time.monotonic()


def _band(titles: dict[int, str], ids: list[int], skip: int | None) -> None:
    ids = [a for a in ids if a != skip and a in titles]
    if not ids:
        return
    st.html(CSS)
    with keyed.container(key="pfb_doneband"):
        for aid in ids[:MAX_SHOWN]:
            text, open_, close = st.columns([6, 1, 1], vertical_alignment="center")
            with text:
                st.html(f'<p class="pfb-doneband" role="status">'
                        f'{escape(TEXT.format(title=titles[aid]))}</p>')
            with open_:
                st.button(OPEN, key=f"pfb_band_open_{aid}", on_click=_open, args=(aid,),
                          width="stretch")
            with close:
                st.button(CLOSE, key=f"pfb_band_close_{aid}", on_click=dismiss, args=(aid,),
                          type="tertiary", width="stretch")
        if len(ids) > MAX_SHOWN:
            st.html(f'<p class="pfb-doneband__more">{escape(MORE.format(n=len(ids) - MAX_SHOWN))}</p>')


def _look(state: dict, works: Iterable[WorkProgress], repo: Repo) -> list[int]:
    """Announce from the works as they are now; remember their titles and when it was done."""
    works = list(works)
    ids = announce(state, works, lambda a: finished_at(repo, a), datetime.now(timezone.utc))
    state["titles"] = {w.assignment_id: w.title for w in works}
    state["checked"] = _monotonic()
    return ids


def for_home(repo: Repo, works: list[WorkProgress]) -> None:
    """At the top of the home screen (it holds every work already)."""
    state = _state()
    ids = _look(state, works, repo)
    _band(state["titles"], ids, None)


def for_work(repo: Repo, progress: WorkProgress) -> None:
    """At the top of a work's screens: the other works that can be checked now. The open work
    itself needs no band (it is in front of the coach: it counts as seen).

    The other works are read at most every `CHECK_EVERY` seconds per browser session (the home
    screen's look counts too): a work screen is drawn again and again (every press, every
    refresh while an analysis runs), and reading every work each time would slow it down as
    works pile up. In between, the band shows what the last look found, less what was closed."""
    state = _state()
    checked = state.get("checked")
    if isinstance(checked, (int, float)) and 0 <= _monotonic() - checked < CHECK_EVERY:
        ids = [aid for aid, marker in state["shown"].items()
               if state["dismissed"].get(aid) != marker]
    else:
        try:
            # the open work is the page run's own progress (its snapshot is not built again)
            works = [progress if a["id"] == progress.assignment_id else work_progress(repo, a["id"])
                     for a in sorted(repo.list_assignments(), key=lambda a: -a["id"])]
        except Exception:   # noqa: BLE001 - a band is never worth a broken screen
            opened(progress.assignment_id)
            return
        ids = _look(state, works, repo)
    opened(progress.assignment_id)     # also when it finished while the coach was looking at it
    _band(state["titles"], ids, progress.assignment_id)
