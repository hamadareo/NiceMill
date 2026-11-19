"""The live parts of the waiting screens (steps 3 and 4).

- `watch`: a fragment that redraws itself every `WATCH_SEC` seconds (st.fragment, not a
  full-page loop) and reruns the whole page once the work has moved on, so the stepper,
  the header and the step follow the database. Each tick reads the work once (one
  progress.View); the first draw uses the page run's own view.
- `worker_stopped`: queued work but no sign of the worker for `WORKER_SILENT_SEC`.
- `remaining_jobs` / `estimate_sec` / `estimate_text`: a rough remaining time (目安): the
  jobs still to come per method and stage (the active ones and the follow-ups they will
  start) times the median time of that stage's finished runs of the same method; nothing
  when one of them has no history.
- `remaining_sec` / `remaining_text`: the time left of the whole analysis for the progress
  panel (「残り約12分（目安）」): through the automatic people choice, less what the running
  steps have already done.
"""
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import streamlit as st

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.services import eta
from pfb_app.storage import files
from pfb_app.storage.db import Repo
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard.progress import View

WATCH_SEC = 10
WORKER_SILENT_SEC = 300
WORKER_STOPPED = "解析を担当する仕組みが止まっているようです。管理者に連絡してください。"
# under the (disabled) primary of a waiting screen: nothing to press, it moves on by itself
NOTHING_TO_PRESS = "いまは押すものはありません。終わると自動で次へ進みます。"
ESTIMATE_NOTE = ("これまでの解析にかかった時間からの目安です。"
                 "順番待ちの時間は含みません。")

ACTIVE = {JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value}
FRONT_STAGES = eta.FRONT_STAGES
_BASELINE = "_pfb_watch_"


# --- the job rows -----------------------------------------------------------------------------

def latest_jobs(repo: Repo, submission_id: int) -> dict[tuple, dict]:
    """The newest job row of each (stage, camera, subject_id) of a submission."""
    latest: dict[tuple, dict] = {}
    for row in repo.submission_job_rows(submission_id):
        latest[(row["stage"], row["camera"], row["subject_id"])] = row
    return latest


def active_jobs(repo: Repo, submission_id: int) -> list[dict]:
    return [r for r in latest_jobs(repo, submission_id).values() if r["status"] in ACTIVE]


# --- is anyone there to do the work? ----------------------------------------------------------

def worker_stopped(settings: Settings, has_active: bool, now: datetime | None = None) -> bool:
    """True when work waits but the worker has not shown it is alive for a while (or ever)."""
    if not has_active:
        return False
    beat = files.worker_beat_at(settings)
    now = now or datetime.now(timezone.utc)
    return beat is None or (now - beat).total_seconds() > WORKER_SILENT_SEC


# --- remaining time ---------------------------------------------------------------------------

def remaining_jobs(repo: Repo, assignment_id: int, *, front_only: bool = False,
                   through: bool = False) -> Counter:
    """Jobs still to come per (method, stage): every active job, plus the follow-ups it will
    start up to the next point where the teacher has to act (services.eta.remaining_slots,
    counted). The method is "pfb3x3", "paper" or "document". `front_only`: only the stages up
    to the people mapping. `through`: the tracking method goes on past the people mapping
    (the people are chosen automatically, so its behavior follows the tracking without the
    coach)."""
    return eta.remaining_jobs(repo, assignment_id, front_only=front_only, through=through)


def estimate_sec(repo: Repo, counts: Counter) -> float | None:
    """Σ jobs × the median time of the stage's finished runs of the same method; None
    without jobs, or when a stage that is still to come has never finished before."""
    if not counts:
        return None
    total = 0.0
    for (engine, stage), n in counts.items():
        durations = repo.stage_durations(Stage(stage), engine)
        if not durations:
            return None
        total += n * statistics.median(durations)
    return total


def remaining_sec(repo: Repo, assignment_id: int) -> float | None:
    """The time left of the whole analysis (services.eta.remaining_sec): the jobs still to
    come priced by the length of the video where earlier runs allow it, else by the median
    of the stage's earlier runs, less what running steps with their own progress have done;
    at least 0. None without an estimate."""
    return eta.remaining_sec(repo, assignment_id)


_rounded = eta.rounded   # 「約15分」: rounded up, to 5 minutes from 10, to 10 from an hour


def estimate_text(seconds: float, label: str = "残り時間の目安") -> str:
    """「残り時間の目安: 約15分」 (rounded as `_rounded`); `label` names what is estimated
    (the check card: 「解析にかかる時間の目安」)."""
    return f"{label}: {_rounded(seconds)}"


def remaining_text(seconds: float) -> str:
    """「残り約12分（目安）」, 「残り約1時間10分（目安）」 (rounded as `estimate_text`)."""
    return f"残り{_rounded(seconds)}（目安）"


def show_worker_warning(repo: Repo, settings: Settings, submission_ids: list[int]) -> None:
    if worker_stopped(settings, any(repo.has_active_jobs(s) for s in submission_ids)):
        st.warning(WORKER_STOPPED)


# --- the 10-second watch ----------------------------------------------------------------------

def tick(repo: Repo | None, assignment_id: int, name: str, signature: Callable[[View], tuple],
         draw: Callable[[View], None],
         on_change: Callable[[tuple, tuple], None] | None = None,
         view: View | None = None) -> None:
    """One redraw from one snapshot of the work (`view`, else read now from `repo`): when the
    signature differs from the page run's, `on_change(old, new)` and a rerun of the page.
    A work or group deleted meanwhile (KeyError on a fresh read) reruns the page too, which
    then says so (or no longer shows the group)."""
    fresh = view is None
    try:
        if view is None:
            view = View.load(repo, assignment_id)
        now = signature(view)
        base = st.session_state.get(_BASELINE + name)
        if base is not None and now != base:
            st.session_state.pop(_BASELINE + name, None)
            if on_change is not None:
                on_change(base, now)
            st.rerun()                # the whole page: the work has moved on
        draw(view)
    except KeyError:
        if not fresh:                 # the page run's own view: not a deletion meanwhile
            raise
        st.session_state.pop(_BASELINE + name, None)
        st.rerun()


@st.fragment(run_every=WATCH_SEC)
def _watch(name: str, assignment_id: int, signature: Callable[[View], tuple],
           draw: Callable[[View], None], on_change: Callable[[tuple, tuple], None] | None,
           first: list) -> None:
    if first:                         # drawn with the page: the page run's view
        tick(None, assignment_id, name, signature, draw, on_change, view=first.pop())
        return
    repo = get_repo()                 # a fragment rerun may run in another thread
    try:
        tick(repo, assignment_id, name, signature, draw, on_change)
    finally:
        repo.conn.close()


def watch(name: str, view: View, signature: Callable[[View], tuple],
          draw: Callable[[View], None],
          on_change: Callable[[tuple, tuple], None] | None = None) -> None:
    """Draw `draw(view)` now (from the page run's view: its snapshot and the router's
    progress) and again every WATCH_SEC seconds from one fresh snapshot per tick; when
    `signature` differs from what it was on this page run, call `on_change(old, new)` and
    rerun the page."""
    st.session_state[_BASELINE + name] = signature(view)
    _watch(name, view.assignment_id, signature, draw, on_change, [view])
