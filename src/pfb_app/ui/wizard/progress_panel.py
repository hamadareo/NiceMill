"""The progress panel of the waiting screens (step 4, and step 3 while it tracks).

From the tick's own view (its snapshot and its progress; nothing is read again):
- the overall bar 「全体の進み具合 63%」 with its status chip and, when known, 「残り約12分（目安）」
  and the note on how that time is guessed; the shown percentage never goes back (`hold`),
  except once when a redo starts;
- the stages in order, each with its chip, the active one with its own bar and note, and the
  review line 「確認と承認」 (承認 n / 全 m) last;
- a compact bar per group when the work has several groups, else per camera when its one
  group has several cameras;
- the note that the worker seems to have stopped, when work waits with nobody to do it.

While the analysis waits for the coach the bars say 「あなたの番」 and no percentage.
"""
import math
from html import escape

import streamlit as st

from pfb_app.config import Settings
from pfb_app.services import eta
from pfb_app.ui.wizard import bars, live, theme
from pfb_app.ui.wizard.analysis import (MOVING, QUEUED, REVIEW_NAME, RUNNING, Analysis,
                                        hold)
from pfb_app.ui.wizard.progress import View, WorkProgress, camera_label

HOLD = "_pfb_hold_"   # + assignment id: (the fraction shown, whether it was a redo)


def _percent(fraction: float) -> int:
    return math.floor(round(fraction * 100, 6))


def stage_rows(a: Analysis) -> list[bars.StageRow]:
    """The stages, then the review line; a stage whose progress is not known says 進んでいます."""
    rows = [bars.StageRow(theme.L(s.name), s.status, s.percent, s.known,
                          theme.L(s.note if s.known else (s.note or MOVING)))
            for s in a.stages]
    r = a.review
    rows.append(bars.StageRow(theme.L(REVIEW_NAME), r.status, r.percent, True, theme.L(r.note)))
    return rows


def held_percent(assignment_id: int, a: Analysis) -> int:
    """The percentage to show: never lower than the one shown before in this session, except
    once when a redo starts (analysis.hold over the stored (fraction, redo))."""
    key = f"{HOLD}{assignment_id}"
    previous = st.session_state.get(key)
    fraction = hold(tuple(previous) if previous else None, a)
    st.session_state[key] = (fraction, a.redo)
    return _percent(fraction)


def bucket(progress: WorkProgress) -> int:
    """The analysis's percentage in tens (-1 without one), for the waiting screens' page
    signatures: a page rerun at least every 10 points keeps the stepper's percentage current."""
    return progress.analysis.percent // 10 if progress.analysis is not None else -1


def status_of(a: Analysis) -> str:
    """The bar's status: 「あなたの番」 while waiting for the coach; 待機中 while the step it
    stands at is only queued; else the analysis's own."""
    if a.waiting_for_you:
        return bars.YOUR_TURN
    status = a.status
    current = a.current
    if status == RUNNING and current is not None and current.status == QUEUED:
        return QUEUED
    return status


def _caption(a: Analysis) -> str:
    current = a.current
    return theme.L(current.name if current is not None else REVIEW_NAME)


def _compact(view: View) -> str:
    subs = [s for s in view.progress.subs if s.analysis is not None]
    if len(subs) > 1:
        rows = [bars.compact_bar_html(s.label, s.analysis.percent, _caption(s.analysis),
                                      status_of(s.analysis)) for s in subs]
        what = "グループごとの進み具合"
    elif subs and len(subs[0].analysis.cameras) > 1:
        a = subs[0].analysis
        status = status_of(a)
        rows = [bars.compact_bar_html(camera_label(cam), _percent(f), _caption(a),
                                      "done" if f >= 1 else status)
                for cam, f in a.cameras.items()]
        what = "カメラごとの進み具合"
    else:
        return ""
    return (f'<div class="pfb-compacts" role="group" aria-label="{escape(theme.L(what))}">'
            f'{"".join(rows)}</div>')


def draw(view: View, settings: Settings) -> None:
    progress = view.progress
    a = progress.analysis
    ids = [s.submission_id for s in progress.subs]
    if a is not None:
        est = eta.estimate(view.repo, progress.assignment_id) if a.active else None
        seconds = est.seconds if est is not None else None
        remaining = theme.L(live.remaining_text(seconds)) if seconds is not None else None
        st.html(bars.overall_bar_html(held_percent(progress.assignment_id, a), remaining,
                                      status_of(a)))
        if remaining:
            st.caption(theme.L(live.ESTIMATE_NOTE))
        elif est is not None and not est.idle:
            st.caption(theme.L(eta.EMPTY))
        for line in eta.queue_lines(est) if est is not None else ():
            st.caption(theme.L(line))
        st.html(bars.stage_list_html(stage_rows(a)) + _compact(view))
    live.show_worker_warning(view.repo, settings, ids)
