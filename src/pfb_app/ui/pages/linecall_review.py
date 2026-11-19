"""判定支援 / 試合後の判定材料: the engine's near-line episodes of one camera of a sports work,
with the evidence photos, the clip and the coach's 「正しい」「違う」「判定不能」.

- Pick a sports work, a group and a camera (the first of each by default). A work that is
  to be deleted stays listed until it is gone, but nothing new is built for it.
- 「判定材料を作る」 (the one primary button) queues the camera's linecall job
  (services.linecall.request). It needs the camera's people found first (a usable track
  run). While the job waits or runs the page says 「判定材料を作っています」 and warns when
  the worker seems stopped (the work's progress snapshot leaves linecall jobs out, so the
  page does its own check); every WATCH_SEC a fragment rereads both and reruns the page
  when either changed (not on the page run itself: that tick uses the page's own read). A failed job shows only
  the fixed failure copy, never the job's error, and still lists an earlier run's episodes.
  Without the pose weights (missing, or not the pinned file) the button is off and the page
  says 「判定材料を作る準備ができていません」 instead (the engine is never asked to fetch them).
- The episodes come from episodes.json (missing or torn: 「判定材料はまだありません」),
  PAGE_SIZE at a time. Their crop, overlay and clip paths are relative to the camera's
  linecall/ dir; only pictures and mp4 files inside it are read (once, as bytes), and a
  missing or unreadable one is said in words. A clip is read only when its 「動画で見る」 is
  opened.
- A 踏んだ at a line toward the camera (linecall_view.toward_camera, from geometry.json; also
  when the geometry is unknown) is shown as 要確認 with the reason, never as a plain 踏んだ;
  a sideline verdict says the sideline is not checked against ground truth yet.
- A press stores the label with the engine's verdict (services.linecall.save_label). When the
  episodes were rebuilt meanwhile and that verdict changed, nothing is stored and the page
  says the material is new, in that episode's box above its buttons (and as a toast). A
  label given for an earlier verdict is not shown as current.

What it shows is 判定材料 (judgement material), never a ruling.
"""
import logging
from html import escape

import streamlit as st

from pfb_app.domain.models import Stage
from pfb_app.domain.vocab import words
from pfb_app.engine.pfb3x3 import Pfb3x3Engine
from pfb_app.pipeline.common import usable
from pfb_app.services import linecall
from pfb_app.ui import access, keyed
from pfb_app.ui import linecall_view as lv
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import theme
from pfb_app.ui.wizard.live import WORKER_STOPPED, worker_stopped
from pfb_app.ui.wizard.progress import camera_label
from pfb_app.ui.access import require_page

require_page("edit")   # first: also when this file runs without ui/app.py's gate

log = logging.getLogger(__name__)

MESSAGE = "_lc_review_message"   # (episode id, or BUILD_TARGET, plain copy) of the last press
BUILD_TARGET = None              # ... a message for the build button
OFFSET = "_lc_review_offset"     # {"<submission>/<camera>": first episode shown}
WORK_KEY, GROUP_KEY, CAMERA_KEY = "lc_work", "lc_group", "lc_camera"
WATCH_SEC = 5.0

W = words("sports")


def sports_groups(repo) -> list[tuple[dict, list[dict]]]:
    """[(work, [group, ...])]: the sports video works with a group that has cameras."""
    out = []
    for work in repo.list_assignments():
        if work.get("domain") != "sports" or work.get("kind") != "video":
            continue
        groups = [sub for sub in repo.list_submissions(work["id"])
                  if isinstance((sub.get("meta") or {}).get("cameras"), dict)
                  and sub["meta"]["cameras"]]
        if groups:
            out.append((work, groups))
    return out


def pick(label: str, options: list, key: str, fmt) -> object:
    """A selectbox whose stored choice is dropped when it is no longer offered."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.selectbox(label, options, key=key, format_func=fmt)


def say(target, text: str) -> None:
    st.session_state[MESSAGE] = (target, text)


def build(sid: int, cam: str) -> None:
    """「判定材料を作る」: queue the camera's linecall job."""
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    st.session_state.pop(MESSAGE, None)
    repo = get_repo()
    try:
        queued = linecall.request(repo, sid, cam)
        if queued is None and linecall.job_state(repo, sid, cam) != "active":
            # the work is to be deleted, its videos are gone, or a person is being erased
            say(BUILD_TARGET, lv.BLOCKED_TEXTS[linecall.blocked(repo, sid) or "deleting"])
    except (KeyError, ValueError):   # the group or camera is gone: the rerun shows it
        log.exception("判定材料の依頼ができませんでした submission=%s camera=%s", sid, cam)
    finally:
        repo.conn.close()


def press(sid: int, cam: str, episode_id: str, label: str, shown_verdict: str) -> None:
    """A coach button: the label for the verdict the page showed."""
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    st.session_state.pop(MESSAGE, None)
    settings, repo = get_settings(), get_repo()
    try:
        linecall.save_label(settings, repo, sid, cam, episode_id, label, shown_verdict)
    except (KeyError, ValueError):   # the episodes were rebuilt meanwhile (or are gone)
        say(episode_id, lv.STALE_VERDICT)
    finally:
        repo.conn.close()


def turn(key: str, offset: int) -> None:
    st.session_state.setdefault(OFFSET, {})[key] = max(0, offset)


@st.fragment(run_every=WATCH_SEC)
def watch(sid: int, cam: str, shown: tuple, first: list) -> None:
    """While the job waits or runs: rerun the page once lv.watch_signature changed. The
    first call is the page run's own (`first` is emptied): it reads nothing."""
    if first:
        first.pop()
        return
    settings, repo = get_settings(), get_repo()
    try:
        now = lv.watch_signature(settings, repo, sid, cam)
    except KeyError:   # the group is gone
        now = None
    finally:
        repo.conn.close()
    if now != shown:
        st.rerun()


def message_for(message, target) -> None:
    if message is not None and message[0] == target:
        st.warning(message[1])


def episode(ep: dict, base, records: dict, sid: int, cam: str, message, geom=None,
            points: dict | None = None) -> None:
    eid = str(ep["id"])
    verdict = str(ep.get("verdict"))   # the engine's: labels are stored for it
    shown = lv.shown_verdict(ep, geom, points)  # 要確認 for a 踏んだ toward the camera
    row = lv.episode_row(ep, geom, points)
    kind = lv.VERDICT_CHIPS[shown]
    with st.container(border=True):
        st.caption(f"{row['時間']}　{row['選手']}　{row['線']}")
        st.html(f'<p class="pfb-lc-verdict">{escape(lv.VERDICT_PREFIX)} '
                f'{theme.status_chip(kind, row["判定材料"])}</p>')
        for line in (*lv.episode_notes(ep, shown), lv.distance_text(ep), lv.reason_text(ep)):
            if line:
                st.caption(line)
        photos = [(lv.evidence_file(base, ep.get(k), "photo"), caption)
                  for k, caption in (("crop", lv.CROP_CAPTION), ("overlay", lv.OVERLAY_CAPTION))]
        if all(data is None for data, _ in photos):
            st.caption(lv.NO_PHOTO)
        else:
            for col, (data, caption) in zip(st.columns(len(photos)), photos):
                with col:
                    if data is None:
                        st.caption(lv.NO_PHOTO)
                    else:
                        st.image(data, caption=caption, width="stretch")
        if lv.evidence_path(base, ep.get("clip"), "clip") is None:
            st.caption(lv.NO_CLIP)
        else:   # read only when opened (a page of clips stays light)
            box = keyed.expander(lv.WATCH_CLIP, key=f"lc_clip_{eid}", on_change="rerun")
            if box.open:
                with box:
                    clip = lv.evidence_file(base, ep.get("clip"), "clip")
                    if clip is None:
                        st.caption(lv.NO_CLIP)
                    else:
                        st.video(clip, format="video/mp4")
        message_for(message, eid)
        cols = st.columns(len(lv.LABEL_BUTTONS))
        for col, (label, text) in zip(cols, lv.LABEL_BUTTONS):
            with col:
                st.button(text, key=f"lc_{eid}_{label}", width="stretch", on_click=press,
                          args=(sid, cam, eid, label, verdict))
        # a label given for another verdict (the episodes were rebuilt) is not current
        shown = lv.coach_label_text(lv._label_of(records.get(eid), verdict))
        if shown:
            st.markdown(f"**{shown}**")


def pager(key: str, start: int, shown: int, total: int, where: str = "") -> None:
    """「前の20件」「次の20件」 (keys lc_prev / lc_next, plus `where`)."""
    back, more = start > 0, start + shown < total
    if not (back or more):
        return
    left, right = st.columns(2)
    with left:
        if back:
            st.button(lv.PREV_PAGE, key=f"lc_prev{where}", icon=":material/arrow_back:",
                      on_click=turn, args=(key, start - lv.PAGE_SIZE))
    with right:
        if more:
            st.button(lv.NEXT_PAGE, key=f"lc_next{where}", icon=":material/arrow_forward:",
                      on_click=turn, args=(key, start + lv.PAGE_SIZE))


def camera_panel(settings, repo, work: dict, sub: dict, cam: str, message) -> None:
    sid = sub["id"]
    state, stopped = lv.watch_signature(settings, repo, sid, cam)
    tracked = usable(repo.latest_run(sid, Stage.track, cam))
    why = linecall.blocked(repo, sid)   # deleting / media_removed / erasing
    ready = Pfb3x3Engine(settings).pose_ready()
    if state == "active":
        st.html(theme.status_chip("processing", lv.BUILDING))
        if stopped:
            st.warning(WORKER_STOPPED)
        watch(sid, cam, (state, stopped), [True])
    st.button(lv.BUILD, key="lc_build", type="primary", icon=":material/fact_check:",
              disabled=state == "active" or not tracked or why is not None or not ready,
              on_click=build, args=(sid, cam))
    message_for(message, BUILD_TARGET)
    if why is not None and not (message and message[0] == BUILD_TARGET):
        st.caption(lv.BLOCKED_TEXTS[why])
    elif not tracked and state != "active":
        st.caption(lv.NEEDS_TRACK)
    if not ready:
        st.warning(lv.NOT_READY)
    elif state == "failed":
        st.error(lv.BUILD_FAILED)

    path = linecall.episodes_path(settings, repo, sid, cam)
    episodes = lv.listed(lv.load_episodes(path) or [])
    if not episodes:
        if state != "failed":
            st.info(lv.NO_EPISODES)
        return
    records = repo.linecall_label_records(sid, cam)
    st.markdown(lv.agreement(episodes, records))

    key = f"{sid}/{cam}"
    start, shown = lv.page_slice(episodes, st.session_state.get(OFFSET, {}).get(key, 0))
    if len(episodes) > lv.PAGE_SIZE:
        st.caption(lv.page_range_text(start, len(shown), len(episodes)))
    pager(key, start, len(shown), len(episodes), "_top")
    geom = lv.load_geometry(path.parent)   # where the camera is: lines toward it
    points = lv.court_points(path.parent, lv.stepped_frames(shown)) if geom is not None else {}
    for ep in shown:
        episode(ep, path.parent, records, sid, cam, message, geom, points)
    pager(key, start, len(shown), len(episodes))


def main() -> None:
    st.title(lv.REVIEW_TITLE)
    st.caption(lv.NOTE)
    st.caption(lv.REVIEW_GUIDE)
    message = st.session_state.pop(MESSAGE, None)   # shown once, where it belongs
    if message is not None:
        st.toast(message[1])
    settings, repo = get_settings(), get_repo()
    try:
        works = sports_groups(repo)
        if not works:
            st.info(lv.NO_SPORTS_WORK)
            return
        by_work = {w["id"]: (w, groups) for w, groups in works}
        aid = pick(W.work, list(by_work), WORK_KEY, lambda i: by_work[i][0]["title"])
        groups = {g["id"]: g for g in by_work[aid][1]}
        sid = pick(lv.PICK_GROUP, list(groups), GROUP_KEY, lambda i: groups[i]["label"])
        cams = list(groups[sid]["meta"]["cameras"])
        cam = pick(lv.PICK_CAMERA, cams, CAMERA_KEY, camera_label)
        camera_panel(settings, repo, by_work[aid][0], groups[sid], cam, message)
    finally:
        repo.conn.close()


main()
