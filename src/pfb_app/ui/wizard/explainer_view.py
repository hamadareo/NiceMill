"""解説動画 on the screens (services.explainer makes them in the worker).

`person_panel`: one person's video — 「解説動画を作る」 (secondary: the screen's one primary
stays its download), then a progress line that refreshes itself (an st.fragment every
POLL_SEC; the page reruns once the video is there), then the player, 「解説動画を受け取る」 and
「作り直す」. A video whose stamp is not the current one is never shown (「内容が変わったため…」);
a failed one says why in the job's own words. `tracking_panel`: the group's 「追跡の様子」, the
same way. `section` is step 6's part (people of the work, groups); the dashboard's one-person
view calls `person_panel`.

管理者 and 評価する人 only (ui.access.can_edit; anyone else sees nothing). The request is made on a
fresh connection (ui.common.get_repo: read-only without an editing session) with this browser
session's auth session id, which the worker checks again before it starts and before it
publishes. No keyed containers here (only widget keys).
"""
from __future__ import annotations

import logging
from html import escape

import streamlit as st

from pfb_app.domain.models import Kind
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import explainer
from pfb_app.storage.db import Repo
from pfb_app.ui import access
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import theme

log = logging.getLogger(__name__)

TITLE = "解説動画"
GUIDE = ("よかった場面ごとに、追跡した枠、AIに見せた範囲と画像、場面の説明をつけた短い動画です。"
         "スマートフォンでも見られます。")
NOTE = "AIの注目の可視化は研究中です。点線と右の画像は、AIに渡した範囲と画像です。"
WHO = "だれの動画か"
GROUP = "グループ"
MAKE = "解説動画を作る"
REMAKE = "作り直す"
AGAIN = "もう一度作る"
DOWNLOAD = "解説動画を受け取る"
TRACK_TITLE = "追跡の様子"
TRACK_GUIDE = "解析した動画の始めから最大2分、評価した人の枠と名前をつけた動画です（ほかの人は灰色の枠だけです）。"
TRACK_MAKE = "追跡の様子の動画を作る"
TRACK_DOWNLOAD = "追跡の様子の動画を受け取る"
QUEUED = "順番を待っています。作り終わると、ここに動画が出ます。"
RUNNING = "作っています（{p}%）{note}"
STALE = "前に作った動画のあとで、文章や名前、追跡の記録が変わりました。作り直してください。"
NOBODY = "承認した人がいると、解説動画を作れます。"
LIVE_NOBODY = "PDFに入る文のある人がいると、解説動画を作れます。"   # 「あとから訂正」
REMOVED = "この作業の動画は消してあるため、解説動画は作れません。"
POLL_SEC = 3


def _write_request(sid: int, subject_id: int | None) -> None:
    """Queue the video (a fresh connection; refused words shown on the next run)."""
    from pfb_app.ui.common import flash   # noqa: PLC0415
    if access.refuse_unless_edit():
        return
    repo = get_repo()
    try:
        user = access.current_user()
        explainer.request(repo, sid, subject_id, session_id=access.session_id(),
                          actor=user.login if user else None)
    except explainer.Refused as e:
        flash(str(e), kind="warning")
    except Exception as e:   # noqa: BLE001 - the screen stays
        if access.refused(e):
            flash(access.VIEW_ONLY, kind="warning")
        else:
            log.exception("解説動画を頼めませんでした submission=%s subject=%s", sid, subject_id)
            flash(explainer.FAILED, kind="warning")
    finally:
        repo.conn.close()


def _file_name(*parts: str) -> str:
    """`<work>_<who>_<what>.mp4` without characters file systems refuse."""
    from pfb_app.export.labels import _UNSAFE   # noqa: PLC0415
    return _UNSAFE.sub("_", "_".join(p for p in parts if p)) + ".mp4"


def _note(text: str) -> None:
    st.html(f'<p class="pfb-dash-note">{escape(text)}</p>')


def _size(n: int | None) -> str:
    return "" if not n else f"{n / 1024 ** 2:.0f} MB" if n >= 1024 ** 2 else "1 MB未満"


@st.fragment(run_every=POLL_SEC)
def _progress(sid: int, subject_id: int | None) -> None:
    """The progress line (a video being made, or its video being prepared), redrawn by itself;
    the page reruns once that has ended. Re-checks the session on every redraw (ui.access): a
    revoked, demoted or expired one sees nothing more."""
    if not access.can_edit():
        return
    repo = get_repo()
    try:
        shown = explainer.status(repo, get_settings(), sid, subject_id)
    except KeyError:
        shown = explainer.Status("none")
    finally:
        repo.conn.close()
    if shown.state not in ("queued", "running", "preparing"):
        st.rerun()
    if shown.state == "preparing":
        st.progress(min(max(int(round((shown.progress or 0.0) * 100)), 0), 100),
                    text=shown.reason or "")
    elif shown.state == "queued":
        st.info(QUEUED, icon=":material/schedule:")
    else:
        pct = int(round((shown.progress or 0.0) * 100))
        st.progress(min(max(pct, 0), 100),
                    text=RUNNING.format(p=pct, note=f"・{shown.note}" if shown.note else ""))


def _panel(repo: Repo, sid: int, subject_id: int | None, *, make: str, download: str,
           file_name: str, key: str) -> None:
    settings = get_settings()
    try:
        shown = explainer.status(repo, settings, sid, subject_id)
    except KeyError:
        return
    if shown.state == "unavailable":
        _note(shown.reason or "")
        return
    if shown.state in ("queued", "running"):
        _progress(sid, subject_id)
        return
    if shown.state == "preparing":         # its video is being made playable first
        _progress(sid, subject_id)
        st.button(theme.L(make), key=f"pfb_explainer_make_{key}", icon=":material/movie:",
                  disabled=True)
        return
    if shown.state == "done" and shown.path is not None:
        try:
            st.video(str(shown.path), format="video/mp4")
        except Exception:   # noqa: BLE001 - removed meanwhile (a cap, an erasure)
            log.exception("解説動画を開けませんでした: %s", shown.path)
            _note(explainer.FAILED)
            return
        seconds = f"{shown.seconds:.0f}秒" if shown.seconds else ""
        _note("・".join(s for s in (seconds, _size(shown.size)) if s))
        path = shown.path
        st.download_button(theme.L(download), lambda: path.read_bytes(), file_name=file_name,
                           mime="video/mp4", icon=":material/download:",
                           key=f"pfb_explainer_dl_{key}")
        if st.button(REMAKE, key=f"pfb_explainer_remake_{key}", type="tertiary",
                     icon=":material/refresh:"):
            _write_request(sid, subject_id)
            st.rerun()
        return
    if shown.state == "stale":
        _note(STALE)
    elif shown.state == "failed":
        st.warning(shown.reason or explainer.FAILED, icon=":material/info:")
    label = AGAIN if shown.state == "failed" else make
    if st.button(theme.L(label), key=f"pfb_explainer_make_{key}", icon=":material/movie:"):
        _write_request(sid, subject_id)
        st.rerun()


def person_panel(repo: Repo, sid: int, subject_id: int, name: str, title: str) -> None:
    """One person's 解説動画 (nothing for anyone but 管理者 and 評価する人)."""
    if not access.can_edit():
        return
    _panel(repo, sid, subject_id, make=MAKE, download=DOWNLOAD,
           file_name=_file_name(title, f"{name}さん", "解説動画"), key=f"{sid}_{subject_id}")
    _note(NOTE)


def tracking_panel(repo: Repo, sid: int, title: str, group: str) -> None:
    if not access.can_edit():
        return
    _panel(repo, sid, None, make=TRACK_MAKE, download=TRACK_DOWNLOAD,
           file_name=_file_name(title, group, "追跡の様子"), key=f"{sid}_tracking")


def _head(text: str) -> None:
    st.html(f'<h3 class="pfb-subhead">{escape(theme.L(text))}</h3>')


def section(repo: Repo, work: dict) -> None:
    """Step 6's 解説動画: pick an approved person (「あとから訂正」: one whose text is in use), their
    panel; then the group's 追跡の様子."""
    if work.get("kind") != Kind.video.value or not access.can_edit():
        return
    _head(TITLE)
    _note(GUIDE)
    if work.get("media_removed_at"):
        _note(REMOVED)
        return
    from pfb_app.services import review   # noqa: PLC0415
    from pfb_app.domain.models import SubjectStatus   # noqa: PLC0415
    groups = repo.list_submissions(work["id"])
    several = len(groups) > 1
    people = []
    live = work.get("review_mode") == "correct_later"
    from pfb_app.services import corrections   # noqa: PLC0415
    for g in groups:
        for s in repo.list_subjects(g["id"]):
            if not is_evaluated(s):
                continue
            # 「あとから訂正」: a text in use (not reported wrong); the panel says if not in the PDF
            if (corrections.usable_run(repo, s, work) is not None if live
                    else review.subject_status(repo, s["id"]) == SubjectStatus.approved):
                label = f"{s['display_name']}さん" + (f"（{g['label']}）" if several else "")
                people.append((s["id"], g["id"], s["display_name"], label))
    if not people:
        _note(LIVE_NOBODY if live else NOBODY)
    else:
        labels_ = {p[0]: p[3] for p in people}
        picked = st.selectbox(theme.L(WHO), list(labels_), format_func=labels_.get,
                              key=f"pfb_explainer_who_{work['id']}")
        sid, name = next((p[1], p[2]) for p in people if p[0] == picked)
        person_panel(repo, sid, picked, name, work["title"])
    _head(TRACK_TITLE)
    _note(TRACK_GUIDE)
    if several:
        names = {g["id"]: g["label"] for g in groups}
        gid = st.selectbox(theme.L(GROUP), list(names), format_func=names.get,
                           key=f"pfb_explainer_group_{work['id']}")
    else:
        gid = groups[0]["id"] if groups else None
    if gid is not None:
        tracking_panel(repo, gid, work["title"], next(g["label"] for g in groups
                                                      if g["id"] == gid))
