"""判定支援 / ライブ判定: start and stop the engine's live line-call demo and follow it.

- The source is the settings demo (`linecall_demo_video`; `--tid-video` for a TrackID video
  or its 1920 proxy) or a camera of a sports work's video group (pfb_app.engine.live: one
  run per app, outside the worker). The settings, the controller and the sources are read
  once per page run (the repo is closed again) and handed to the fragment.
- A fragment redraws every LIVE_TICK_SEC with no DB or settings work: the run's status,
  a stop under way, the stream.jsonl tail (byte offset, into a LiveState kept in the
  session), the last outcome and whether the chosen video exists. It draws the court
  mini-map (an SVG data URI in st.image, since st.html drops inline SVG), the latency, the
  share of frames the engine dropped (with the drops note once any was: 「混んでいるため…」
  only when other work kept the machine busy, inference.gpu.machine_busy, asked at most every
  LIVE_LOAD_CHECK_SEC while frames are dropped and remembered for the run) and the alerts,
  newest first. While a run goes, the source shows the running video.
- 「停止」 runs stop() in a thread (it can take 20 s) for the session it was drawn for: the
  page says 「停止しています…」 meanwhile and keeps drawing, and a press that arrives after
  that run ended never stops the next one.
- How the last run ended: a user stop or the video's end is 停止中; exit 3 is the
  needs-GPU copy; any other failure a plain message, never the engine's text. Those two
  banners are shown for LIVE_OUTCOME_SHOWN_SEC after the run's end only.

What it shows is 判定材料 (judgement material), never a ruling.
"""
import logging
import time
from html import escape

import streamlit as st

from pfb_app.domain.vocab import words
from pfb_app.engine import EngineError, Pfb3x3Engine
from pfb_app.engine import live
from pfb_app.engine.live import LiveBusy, LiveController
from pfb_app.inference import gpu
from pfb_app.ui import access
from pfb_app.ui import linecall_view as lv
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import theme
from pfb_app.ui.access import require_page

require_page("edit")   # first: also when this file runs without ui/app.py's gate

log = logging.getLogger(__name__)

FOLLOW = "_lc_live_follow"   # {"stream", "tail": StreamTail, "state": LiveState, "busy",
#                              "busy_at"}
MESSAGE = "_lc_live_message"   # what the last start press could not do (plain copy)
SOURCE_KEY = "lc_live_source"

W = words("sports")


def controller(settings) -> LiveController:
    return LiveController(settings, Pfb3x3Engine(settings), stop_wait_sec=live.STOP_WAIT_SEC)


def load_sources(settings) -> list[lv.LiveSource]:
    repo = get_repo()
    try:
        return lv.live_sources(repo, settings)
    finally:
        repo.conn.close()


def follow(stream=None) -> dict | None:
    """The stream this session follows; a new `stream` (a run started here or elsewhere)
    starts over with an empty state. New records are folded in on every call."""
    f = st.session_state.get(FOLLOW)
    if stream is not None and (f is None or f["stream"] != str(stream)):
        f = {"stream": str(stream), "tail": lv.StreamTail(stream), "state": lv.LiveState(),
             "busy": False, "busy_at": None}
        st.session_state[FOLLOW] = f
    if f is not None:
        f["state"].apply(f["tail"].read_new())
    return f


def start(ctl: LiveController, source: lv.LiveSource) -> None:
    """The 「デモを開始」 callback: starts the run (it returns at once)."""
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    st.session_state.pop(MESSAGE, None)
    try:
        sess = ctl.start(source.path, video_id=source.video_id, tid_video=source.tid_video)
    except LiveBusy:      # started meanwhile (another session): the page follows that run
        return
    except FileNotFoundError:
        st.session_state[MESSAGE] = lv.LIVE_VIDEO_MISSING
        return
    except (EngineError, OSError, ValueError):
        log.exception("ライブ判定を開始できませんでした")
        st.session_state[MESSAGE] = lv.LIVE_START_FAILED
        return
    st.session_state.pop(FOLLOW, None)
    follow(sess.stream)


def busy_now(f: dict, sess) -> bool:
    """Whether other work kept the machine busy while this run dropped frames: asked (gpu.
    machine_busy, the live run itself excluded) at most every LIVE_LOAD_CHECK_SEC while the run
    goes and has dropped; once seen, remembered for the run."""
    state = f["state"]
    if not f.get("busy") and sess is not None and state.dropped > 0:
        now = time.monotonic()
        if f.get("busy_at") is None or now - f["busy_at"] >= lv.LIVE_LOAD_CHECK_SEC:
            f["busy_at"] = now
            f["busy"] = bool(gpu.machine_busy(sess.pid))
    return bool(f.get("busy"))


def stop(ctl: LiveController, session: str) -> None:
    """The 「停止」 callback: stops the run it was drawn for, in the background."""
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    st.session_state.pop(MESSAGE, None)
    ctl.stop_in_background(session)


def recent_failure(ctl: LiveController) -> str | None:
    """"needs_gpu" or "error" when the last run ended so within LIVE_OUTCOME_SHOWN_SEC."""
    outcome = ctl.last_outcome()
    if outcome not in ("needs_gpu", "error"):
        return None
    for record in (ctl.exit_record, ctl.root / "logs" / "live.out"):
        try:
            age = time.time() - record.stat().st_mtime
        except OSError:
            continue
        return outcome if age <= lv.LIVE_OUTCOME_SHOWN_SEC else None
    return None


def alerts_html(alerts: list[dict]) -> str:
    rows = []
    for a in alerts[:lv.LIVE_ALERTS_SHOWN]:
        chip = theme.status_chip("your_turn", lv.NEEDS_CHECK) + " " if a.get("needs_check") else ""
        rows.append(f'<li class="pfb-lc-alert">{chip}{escape(lv.alert_text(a))}</li>')
    return ('<ul class="pfb-lc-alerts" style="list-style:none;padding:0;margin:0">'
            + "".join(rows) + "</ul>")


@st.fragment(run_every=lv.LIVE_TICK_SEC)
def live_panel(ctl: LiveController, sources: list[lv.LiveSource]) -> None:
    stopping = ctl.stopping()
    sess = ctl.status()
    f = follow(None if sess is None else sess.stream)
    running = sess is not None and not stopping
    state = f["state"] if f is not None else None

    by_key = {s.key: s for s in sources}
    if SOURCE_KEY in st.session_state and st.session_state[SOURCE_KEY] not in by_key:
        del st.session_state[SOURCE_KEY]   # a camera that is gone: back to the demo
    shown = None
    if sess is not None:   # the radio shows the running video
        shown = next((s for s in sources if s.video_id == sess.video_id), None)
        if shown is not None:
            st.session_state[SOURCE_KEY] = shown.key
    key = st.radio(lv.LIVE_SOURCE_LABEL, list(by_key), key=SOURCE_KEY,
                   format_func=lambda k: by_key[k].label, disabled=running or stopping,
                   help=f"デモ動画か、{W.work}のカメラの動画を選べます。")
    if sess is not None:
        st.caption(lv.LIVE_RUNNING_VIDEO.format(label=shown.label) if shown is not None
                   else lv.LIVE_RUNNING_ELSEWHERE)
    source = by_key[key]

    if running:
        st.button(lv.LIVE_STOP, key="lc_live_stop", icon=":material/stop:", on_click=stop,
                  args=(ctl, sess.session))
    else:
        missing = source.path is None or not source.path.is_file()
        if source.path is None:
            st.info(lv.LIVE_NO_DEMO)
        elif missing:
            st.info(lv.LIVE_VIDEO_MISSING)
        st.button(lv.LIVE_START, key="lc_live_start", type="primary",
                  icon=":material/play_arrow:", disabled=missing or stopping,
                  on_click=start, args=(ctl, source))

    if stopping:
        st.status(lv.LIVE_STOPPING, state="running")
    elif running and not (state is not None and state.ended):
        st.html(theme.status_chip("processing", lv.LIVE_RUNNING))
    else:
        st.html(theme.status_chip("stopped", lv.LIVE_STOPPED))
    if not running and not stopping:
        failure = recent_failure(ctl)
        if failure == "needs_gpu":
            st.warning(lv.LIVE_NO_GPU)
        elif failure == "error":
            st.error(lv.LIVE_FAILED)
    if st.session_state.get(MESSAGE):
        st.warning(st.session_state[MESSAGE])

    if state is None:
        return
    left, right = st.columns([3, 2])
    with left:
        alt = lv.minimap_alt(state.players, state.flagged)
        st.image(lv.minimap_data_uri(state.players, state.flagged), caption=alt, alt=alt,
                 width="stretch")
    with right:
        latency = lv.latency_text(state.latency_s)
        if latency:
            st.markdown(f"**{latency}**")
        drops, drops_note = lv.drop_texts(state, busy_now(f, sess))
        if drops:
            st.markdown(drops)
        if drops_note:   # a dropped frame may hide a step on the line: say so
            st.info(drops_note)
        if state.alerts:
            st.html(alerts_html(state.alerts))
        else:
            st.caption(lv.LIVE_NO_ALERTS)


settings = get_settings()
st.title("ライブ判定")
st.caption(lv.NOTE)
st.caption(lv.LIVE_SIDELINE_NOTE)
live_panel(controller(settings), load_sources(settings))
