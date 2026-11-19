"""Step 1 of a new work, the drop zone: 「評価したいファイルを入れてください」.

The coach drops files (or, only when the server's receiving folder has videos, picks them
there, the newest already picked) and presses the one primary 「入れて確かめる」. Nothing else
is asked here: 解析の細かさ starts as 標準 and is changed on the check card's 「内容を変える」.
Under the uploader, on a phone only, a tip to film in 1080p・30fps; a file the browser refuses
(over the size limit) is explained in Japanese before anything is sent (common.upload_refused_style).
The kind comes from the file extensions (the intake rules):

- videos (one per camera, at most 4): a new video work is created and its first group is held
  for the content check (`start_work_from_videos`); the work opens at its derived step, which
  is step 1, the check card (step_check);
- documents: the files are kept as (name, bytes) in `st.session_state[DROP_DOCS]` and the
  existing step-1 form follows with the kind fixed (step_assignment.render_new); the files are
  registered right after the work is created. Documents get no content check.

Documents and videos together are refused (「書類と動画は別々に入れてください」).
`pick_files` / `drop_videos` are shared with step 2, where a later video group of a checked
work goes through the same drop and then its own check card.
"""
import logging
from pathlib import Path

import streamlit as st

from pfb_app.config import Settings
from pfb_app.preprocess.frames import VideoError
from pfb_app.services import chunked_upload as cu
from pfb_app.services.content_check import (MAX_CAMERAS, TOO_MANY_CAMERAS,
                                            start_work_from_videos, used_video_paths)
from pfb_app.services.intake import (DOCUMENT_EXTENSIONS, VIDEO_EXTENSIONS, list_inbox_videos,
                                     save_uploaded_video)
from pfb_app.storage.db import Repo
from pfb_app.ui.common import (UNEXPECTED, IntakeFailed, flash, show_intake_error,
                               upload_note, upload_refused_style,
                               upload_watch_script)
from pfb_app.ui.wizard import big_upload, router, theme

log = logging.getLogger(__name__)

DROP_DOCS = router.DROP_DOCS      # [(name, bytes)] of dropped documents, until the work exists
KEYS = ("drop_source", "drop_files", "drop_inbox")
TITLE = "評価したいファイルを入れてください"
GUIDE = "動画か書類を入れると、種類と中身を確かめてから評価の準備を始めます。"
PRIMARY = "入れて確かめる"
MIXED = "書類と動画は別々に入れてください"
SOURCES = {"upload": "この端末から送る", "inbox": "受け取り用フォルダの動画から選ぶ"}
INBOX_FOUND = "受け取り用フォルダに新しい動画があります（{n}件）。"
INBOX_ALL_USED = "受け取り用フォルダの動画は、どれもすでに使われています。"
USED_MARK = "（使用済み）"
CAMERAS_NOTE = ("同じ場面を複数のカメラで撮った動画は、まとめて入れます（4つまで）。"
                "別の時間の動画（後半など）は、あとで「もう1件追加」から入れます。"
                "書類はいくつでも入れられます。")
INBOX_NOTE = "選んだ動画はコピーせず、そのまま使います。"
VIDEO_DROPPED = "動画を入れました。中身を確かめています。"
# after a failed registration of a file sent with the resumable uploader: it is kept
UPLOAD_KEPT = ("送ったファイルは消さずに残してあります。直してから、もう一度"
               "「入れて確かめる」を押すと、送り直さずに使えます。")
PHONE_TIP = ("スマホの4K動画は大きく、送るのに時間がかかります。"
             "カメラの設定を「1080p・30fps」にすると、速く送れて解析も速くなります。")
# with the resumable uploader: the limit it enforces (services.chunked_upload), and the tip
BIG_NOTE = "動画（MP4・MOVなど）は1ファイル最大 {mb} MB、書類（PDF・Word）も送れます。"
PHONE_TIP_BIG = ("スマホの4K動画は大きく、送るのに時間がかかります。送っている間は画面を開いたままに"
                 "してください。カメラの設定を「1080p・30fps」にすると、速く送れて解析も速くなります。")
# the tip is for phones only (narrow screens); a wide screen never shows it
_PHONE_TIP_STYLE = ("<style>.pfb-phone-tip { display: none; margin: 0.25rem 0 0.5rem;"
                    " color: var(--pfb-muted); font-size: 0.875rem; }"
                    " @media (max-width: 40rem) { .pfb-phone-tip { display: block; } }</style>")

HELP = [
    "動画（グループ活動や、試合や練習などの録画）か、書類（レポートや企画書など）を入れます。"
    "種類はファイルから自動で決まります。",
    "動画は、入れるとすぐに中身（どんな場面か、映っている人は何人くらいか）を自動で確かめます。"
    "結果を見て、よければそのまま始めます。解析の細かさも、そこで変えられます。",
    "書類は、入れたあとに活動の名前と内容を書きます。",
    "うまくいかないとき：入れられなかったときは、出てきた理由を読みます。ファイルの種類と大きさ"
    "（入れる欄の下に書いてあります）を確かめて、もう一度入れてください。",
]

def kind_of(name: str) -> str | None:
    """"video" or "document" from the file extension (the intake rules); None otherwise."""
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return None


def _name(pick) -> str:
    return pick.name


def default_model(settings: Settings) -> str:
    """解析の細かさ of a work started here: 標準 (changed on the check card)."""
    return "standard" if "standard" in settings.models else next(iter(settings.models))


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def _newest(paths: list[Path]) -> Path:
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    return max(paths, key=mtime)


def pick_files(settings: Settings, key: str, *, documents: bool,
               used: set[Path] | frozenset = frozenset()) -> tuple[str, list]:
    """The uploader (videos, plus documents when `documents`); only when the server's
    receiving folder has videos, a source radio and its multiselect (videos only). `used`:
    videos some work already uses (content_check.used_video_paths); they are not counted as
    new, are marked 「（使用済み）」 and never picked to start with; the newest other video is.
    Returns (source, picks)."""
    inbox = list_inbox_videos(settings)
    fresh = [p for p in inbox if _resolved(p) not in used]
    source = "upload"
    if inbox:
        st.caption(INBOX_FOUND.format(n=len(fresh)) if fresh else INBOX_ALL_USED)
        source = st.radio(theme.L("ファイルの選び方"), list(SOURCES), format_func=SOURCES.get,
                          key=f"{key}_source", horizontal=True)
    if source == "inbox":
        st.caption(INBOX_NOTE)
        pick_key = f"{key}_inbox"
        chosen = st.session_state.get(pick_key)
        if chosen is None or any(p not in inbox for p in chosen):
            st.session_state[pick_key] = [_newest(fresh)] if fresh else []
        picks = st.multiselect(theme.L("動画（カメラごとに1つ、4つまで）"), inbox,
                               format_func=lambda p: _name(p) + (
                                   "" if _resolved(p) not in used else USED_MARK),
                               key=pick_key, placeholder="選んでください")
        return source, list(picks)
    if big_upload.available():   # the resumable uploader (ui/serve.py): big phone videos
        picks = big_upload.uploader(
            settings, key, list(VIDEO_EXTENSIONS + (DOCUMENT_EXTENSIONS if documents else ())),
            max_videos=MAX_CAMERAS, videos=VIDEO_EXTENSIONS)
        st.caption(BIG_NOTE.format(mb=settings.max_upload_mb))
        st.html(f'{_PHONE_TIP_STYLE}<p class="pfb-phone-tip">{PHONE_TIP_BIG}</p>')
        return "upload", picks
    types = [e.lstrip(".") for e in VIDEO_EXTENSIONS + (DOCUMENT_EXTENSIONS if documents else ())]
    label = "動画または書類" if documents else "動画（カメラごとに1つ、4つまで）"
    picks = st.file_uploader(theme.L(label), type=types, accept_multiple_files=True,
                             key=f"{key}_files")
    what = ("動画（MP4・MOVなど）か書類（PDFかWord）のファイル" if documents
            else "MP4・MOVなどの動画ファイル")
    st.caption(upload_note(what))
    st.html(f'{_PHONE_TIP_STYLE}{upload_refused_style()}'
            f'<p class="pfb-phone-tip">{PHONE_TIP}</p>')
    st.html(upload_watch_script(), unsafe_allow_javascript=True)
    return "upload", list(picks or [])


def save_videos(settings: Settings, picks: list, from_inbox: bool, saved: list[Path]
                ) -> list[Path]:
    """The videos' paths: inbox paths as they are, uploads copied to data/uploads (each copy
    is appended to `saved` as soon as it exists, so a caller can remove them on failure).
    A resumable upload is already in data/uploads: its path is used as it is, and it stays
    the coach's upload (never in `saved`) until drop_videos claims it after success."""
    if from_inbox:
        return [Path(p) for p in picks]
    paths = []
    for f in picks:
        if isinstance(f, big_upload.BigPick):   # no copy; claimed only after the registration
            paths.append(f.path(settings))
            continue
        path = save_uploaded_video(settings, f.name, f)
        saved.append(path)
        paths.append(path)
    return paths


def drop_videos(settings: Settings, picks: list, from_inbox: bool, register):
    """Save the videos and call `register(paths)`. Returns what `register` returned, or None
    after showing why it could not. On failure the copies made here are removed again, but a
    finished resumable upload is kept (and stays a pick), so pressing again needs no resend;
    after success it is claimed (the work's from now on)."""
    if len(picks) > MAX_CAMERAS:
        st.error(TOO_MANY_CAMERAS)
        return None
    saved: list[Path] = []
    big = [] if from_inbox else [f for f in picks if isinstance(f, big_upload.BigPick)]
    try:
        result = register(save_videos(settings, picks, from_inbox, saved))
    except (VideoError, ValueError, cu.UploadError) as e:
        error = IntakeFailed(str(e))
    except Exception as e:  # noqa: BLE001 - nothing was registered; keep the page alive
        error = IntakeFailed(UNEXPECTED, f"{type(e).__name__}: {e}")
    else:
        for f in big:
            try:
                f.claim(settings)
            except cu.UploadError:   # the work has the file; the sweep keeps a used one
                log.warning("登録した動画の送信記録を閉じられませんでした id=%s", f.id)
        return result
    for path in saved:      # copies made for a registration that did not happen
        path.unlink(missing_ok=True)
    show_intake_error(error)
    if big:
        st.info(UPLOAD_KEPT)
    return None


def _drop(repo: Repo, settings: Settings, source: str, picks: list) -> None:
    names = [Path(str(getattr(p, "name", p))).name for p in picks]
    unknown = [n for n in names if kind_of(n) is None]
    if unknown:
        st.error(f"入れられない種類のファイルです: {'、'.join(unknown)}。"
                 "動画か、PDF・Wordの書類を入れてください。")
        return
    kinds = {kind_of(n) for n in names}
    if len(kinds) > 1:
        st.error(MIXED)
        return
    if kinds == {"document"}:
        st.session_state[DROP_DOCS] = [
            (f.name, f.read_and_remove(settings) if isinstance(f, big_upload.BigPick)
             else f.getvalue()) for f in picks]
        big_upload.reset("drop")
        st.rerun()                    # the documents form asks 解析の細かさ itself
    model_key = default_model(settings)
    created = drop_videos(settings, picks, source == "inbox",
                          lambda paths: start_work_from_videos(repo, settings, paths, model_key))
    if created is None:
        return
    for key in KEYS:          # the next 「新しく始める」 starts from an empty drop zone
        st.session_state.pop(key, None)
    big_upload.reset("drop")
    flash(VIDEO_DROPPED)
    router.open_work(created[0])
    st.rerun()


def render_new(repo: Repo, settings: Settings) -> None:
    """The first screen of a new work; after a documents drop, step 1's form (kind fixed)."""
    docs = st.session_state.get(DROP_DOCS)
    if docs:
        from pfb_app.ui.wizard import step_assignment   # noqa: PLC0415
        step_assignment.render_new(repo, settings, docs=docs)
        return
    with theme.step_card(TITLE, GUIDE):
        source, picks = pick_files(settings, "drop", documents=True,
                                   used=used_video_paths(repo))
        st.caption(CAMERAS_NOTE)
        if st.button(PRIMARY, key="drop_go", type="primary", disabled=not picks,
                     width="stretch"):
            _drop(repo, settings, source, picks)
        if not picks:
            st.caption("ファイルを入れると押せます。")
    theme.help_expander(HELP)
