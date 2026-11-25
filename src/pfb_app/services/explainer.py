"""解説動画: who can have one, the request, the status a screen shows, and the worker's job.

- A person's video: an evaluated person (pipeline.auto_people.is_evaluated) of a video work
  whose text is approved now (review.subject_status) and cites at least one moment, while the
  tracks are not being redone. In a 「あとから訂正」 work: whose newest text is not reported
  wrong (services.corrections.usable_run) and who is in the PDF (its 「同一人物の確認」, with the
  scenes it uses); the video says that the text was made by AI, with its check state and version. The moments, their order and their words are the handout's
  (export.sheets / export.views "handout"); the frames come from the playable source
  (services.moment_clip.source: a small non-HEVC original or the ready proxy, never a 4K/HEVC
  original); what the VLM was given is export.explainer_ai's (never an "attention").
- 「追跡の様子」: one per group, the first camera with a usable track, from the analysed start
  for at most TRACKING_MAX_S, every evaluated person named, everyone else a nameless grey box.

Jobs live in `explainer_jobs` (storage.db), apart from the analysis jobs: they never show in a
work's steps, never hold up an approval, an erasure or retention, never move the finished date.
`request` queues one (one active per group, kind and person; refused for a work whose videos
were removed, being deleted, or a group being erased) with the requester's auth session; the
worker (`run_next`, worker.main's ExplainerRunner) checks that session again before it starts
and before it publishes (a revoked, expired or demoted one: cancelled, nothing written).

Files: `<group>/handout/explainer/person_<subject id>.mp4` (and `tracking.mp4`), each with a
`.json` sidecar holding the stamp it was made for. A screen shows a file only when that stamp is
the current one (`person_stamp` / `tracking_stamp`: approvals, texts, names, exclusion, erasure,
tracks, videos, the work's and group's settings, the drawing's VERSION). A job renders to a
part file and publishes under the group's lock (`handout/.explainer.lock`) after checking the
job, the session and the stamp again; the mp4 goes first, its sidecar last (a crash between
leaves a file whose stamp is old: shown as outdated, never as current). The folder is kept under
MAX_GROUP_BYTES (oldest first) and loses the files of people who no longer have a video; a new
video is refused when the disk is low (ops.disk.ensure_room). `drop_group` (erasure) and
`drop_work` (retention) cancel the jobs and remove the folders under the lock; the handout
folder is one of retention's HEAVY_DIRS as well.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, SubjectStatus
from pfb_app.export import explainer_video as ev
from pfb_app.export import labels
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import usable
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

PERSON, TRACKING = "person", "tracking"
FOLDER = "explainer"
LOCK = ".explainer.lock"
MAX_GROUP_BYTES = 600 * 1024 ** 2
MAX_VIDEO_BYTES = 120 * 1024 ** 2     # st.video holds the file in memory
ESTIMATE_BYTES = 40 * 1024 ** 2
MAX_MOMENTS = 10
TRACKING_MAX_S = 120.0
HUNG_SEC = 600                        # a running video without a heartbeat this long has stopped
PART_OLD_SEC = 3600                   # a part file this old was left by a stop
ACTIVE = ("queued", "running")
HALT = threading.Event()   # the worker is stopping: a video being made stops (not kept)

NOT_VIDEO = "解説動画は、動画の作業で作れます。"
NOT_EVALUATED = "この人は評価に入っていないため、解説動画は作れません。"
NOT_APPROVED = "文章を承認すると、解説動画を作れます。"
NOT_USABLE_LIVE = "文がまだないか、誤りの報告を確認中のため、解説動画は作れません。"   # 「あとから訂正」
NOT_PRINTED = "同じ人と確かめられていない（PDFに入っていない）ため、解説動画は作れません。"
NO_MOMENTS = "よかった場面の記録がないため、解説動画は作れません。"
RETRACKING = "追跡をやり直しているため、いまは作れません。終わってから作ってください。"
MEDIA_REMOVED = "この作業の動画は消してあるため、解説動画は作れません。"
DELETING = "この作業は削除の途中のため、作れません。"
ERASING = "このグループはデータを消している途中のため、いまは作れません。"
NO_TRACK = "追跡の記録がある動画がないため、追跡の様子は作れません。"
PREPARING = "動画の準備をしています（{n}%）。終わると作れます。"
CONVERT_FAILED = "この動画は再生用に変換できませんでした。"
NO_VIDEO = "動画が見つからないため、解説動画は作れません。"
CANNOT_READ = "この動画は読めないため、解説動画は作れません。"
NOBODY = "評価する人がまだいないため、追跡の様子は作れません。"
LOW_DISK = "保存場所の空きが少ないため、解説動画は作りませんでした。"
TOO_BIG = "解説動画が大きくなりすぎたため、保存しませんでした。"
SESSION_GONE = "依頼した人のログインが切れたため、作りませんでした。もう一度ログインして作ってください。"
CHANGED = "作っている間に文章や名前、追跡の記録が変わりました。もう一度作ってください。"
STOPPED = "途中で止まりました。もう一度作ってください。"
HUNG = "10分以上進まなかったため、止めました。もう一度作ってください。"
FAILED = "解説動画を作れませんでした。もう一度作っても直らないときは、管理者に知らせてください。"
CANCELLED = "データを消したため、作るのをやめました。"
REMOVED_MEDIA = "動画を消したため、作るのをやめました。"


class Refused(Exception):
    """No request (a plain Japanese message)."""


# --- places ----------------------------------------------------------------------------------

def folder(settings: Settings, aid: int, sid: int) -> Path:
    return files.submission_dir(settings, aid, sid, create=False) / "handout" / FOLDER


def file_of(settings: Settings, aid: int, sid: int, subject_id: int | None) -> tuple[Path, Path]:
    stem = TRACKING if subject_id is None else f"person_{int(subject_id)}"
    d = folder(settings, aid, sid)
    return d / f"{stem}.mp4", d / f"{stem}.json"


def _lock_path(settings: Settings, aid: int, sid: int) -> Path:
    return files.submission_dir(settings, aid, sid, create=False) / "handout" / LOCK


@contextmanager
def _locked(settings: Settings, aid: int, sid: int):
    path = _lock_path(settings, aid, sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _ids(repo: Repo, sid: int) -> tuple[dict, dict]:
    submission = repo.get_submission(sid)
    return submission, repo.get_assignment(submission["assignment_id"])


# --- who can have one ---------------------------------------------------------------------------

def _work_reason(repo: Repo, submission: dict, work: dict) -> str | None:
    from pfb_app.services import deletion   # noqa: PLC0415
    if work.get("kind") != Kind.video.value:
        return NOT_VIDEO
    if work.get("media_removed_at"):
        return MEDIA_REMOVED
    if deletion.is_requested(submission.get("meta")):
        return DELETING
    if repo.erasing(submission["id"]):
        return ERASING
    return None


def _retracking(repo: Repo, submission: dict) -> bool:
    from pfb_app.export.sheets import _boxes_newer   # noqa: PLC0415
    sid = submission["id"]
    cams = sorted(set((submission.get("meta") or {}).get("cameras") or {})
                  | set(repo.list_cameras(sid)))
    return any(repo.has_active_job(sid, Stage.track, c)
               or _boxes_newer(repo.latest_run(sid, Stage.track, c)) for c in cams)


@functools.lru_cache(maxsize=256)
def _probe(path: str, size: int, mtime_ns: int):
    from pfb_app.services import moment_clip   # noqa: PLC0415
    return moment_clip.probe(Path(path))


def _track_size(track) -> tuple[int, int] | None:
    from pfb_app.export import tracks   # noqa: PLC0415
    meta = tracks.track_meta((track.output or {}).get("boxes")) if track is not None else None
    return (meta.width, meta.height) if meta and meta.width and meta.height else None


def _sources(repo: Repo, submission: dict) -> list[tuple[Path, tuple[int, int] | None, str]]:
    """(video, size from its tracks, codec) of each camera with a usable track."""
    sid = submission["id"]
    videos = (submission.get("meta") or {}).get("cameras") or {}
    out = []
    for cam in sorted(set(videos) | set(repo.list_cameras(sid))):
        track = repo.latest_run(sid, Stage.track, cam)
        if not usable(track):
            continue
        out.append((Path(videos[cam]) if videos.get(cam) else None, _track_size(track), cam))
    return out


def source_state(settings: Settings, video: Path | None,
                 size: tuple[int, int] | None) -> tuple[str, float | None, str | None]:
    """(state, done 0..1, codec) of a camera's file for a player: ok | preparing | failed |
    missing | unreadable (services.moment_clip.source's rule, with the proxy's progress)."""
    from pfb_app.services import proxy   # noqa: PLC0415
    if video is None or not video.is_file():
        return "missing", None, None
    st = video.stat()
    found = _probe(str(video), st.st_size, st.st_mtime_ns)
    if found is None:
        return "unreadable", None, None
    w, h = size or (found.width, found.height)
    if not proxy.needs_playable(w, h, found.codec):
        return "ok", None, found.codec
    state, done = proxy.state(settings, video)
    if state == "ready":
        return "ok", 1.0, found.codec
    if state == "failed":
        return "failed", None, found.codec
    return "preparing", done or 0.0, found.codec


def _source_check(repo: Repo, submission: dict,
                  settings: Settings | None = None) -> tuple[str, float | None] | None:
    """Why the group's videos cannot be played yet (text, progress) or None."""
    settings = settings or repo.settings
    if settings is None:
        return None
    waiting = None
    for video, size, _ in _sources(repo, submission):
        state, done, _ = source_state(settings, video, size)
        if state == "missing":
            return NO_VIDEO, None
        if state == "unreadable":
            return CANNOT_READ, None
        if state == "failed":
            return CONVERT_FAILED, None
        if state == "preparing":
            waiting = done if waiting is None else min(waiting, done)
    if waiting is not None:
        return PREPARING.format(n=int(waiting * 100)), waiting
    return None


def person_reason(repo: Repo, subject_id: int, settings: Settings | None = None) -> str | None:
    """Why this person cannot have a video now (None: they can)."""
    from pfb_app.export import sheets   # noqa: PLC0415
    from pfb_app.services import review   # noqa: PLC0415
    subject = repo.get_subject(subject_id)
    submission, work = _ids(repo, subject["submission_id"])
    why = _work_reason(repo, submission, work)
    if why:
        return why
    if not is_evaluated(subject):
        return NOT_EVALUATED
    from pfb_app.services import corrections   # noqa: PLC0415
    if corrections.mode_of(work) == corrections.CORRECT_LATER:
        if corrections.usable_run(repo, subject, work) is None:
            return NOT_USABLE_LIVE
        found = sheets.person_moments(repo, subject_id)      # None: not in the PDF (the gate)
        if found is None:
            return NOT_PRINTED
        moments = found
    elif review.subject_status(repo, subject_id) != SubjectStatus.approved:
        return NOT_APPROVED
    else:
        moments = sheets.person_moments(repo, subject_id) or []
    if not any(m.cited for m in moments):
        return NO_MOMENTS
    if _retracking(repo, submission):
        return RETRACKING
    found = _source_check(repo, submission, settings)
    return found[0] if found else None


def tracking_reason(repo: Repo, sid: int, settings: Settings | None = None) -> str | None:
    submission, work = _ids(repo, sid)
    why = _work_reason(repo, submission, work)
    if why:
        return why
    if not any(is_evaluated(s) for s in repo.list_subjects(sid)):
        return NOBODY
    if not any(usable(repo.latest_run(sid, Stage.track, c)) for c in repo.list_cameras(sid)):
        return NO_TRACK
    if _retracking(repo, submission):
        return RETRACKING
    found = _source_check(repo, submission, settings)
    return found[0] if found else None


def ensure_sources(repo: Repo, settings: Settings) -> int:
    """The worker's pass: start the playable proxy (services.proxy: tone-mapped for HDR) of a
    video that needs one, in a group with someone approved (a 解説動画 or a dashboard clip can
    be wanted), one at a time (none while another is being made). Returns how many started."""
    from pfb_app.services import proxy   # noqa: PLC0415
    if proxy.busy(settings):
        return 0
    for work in repo.list_assignments():
        if work.get("kind") != Kind.video.value or work.get("media_removed_at"):
            continue
        for submission in repo.list_submissions(work["id"]):
            if _work_reason(repo, submission, work):
                continue
            if work.get("review_mode") == "correct_later":   # someone with a text in use
                from pfb_app.services import corrections   # noqa: PLC0415
                if not any(is_evaluated(s) and corrections.usable_run(repo, s, work) is not None
                           for s in repo.list_subjects(submission["id"])):
                    continue
            elif not any(is_evaluated(s) and s.get("approval_status") == "approved"
                         for s in repo.list_subjects(submission["id"])):
                continue
            for video, size, _ in _sources(repo, submission):
                state, _, codec = source_state(settings, video, size)
                if state != "preparing" or proxy.state(settings, video)[0] == "making":
                    continue
                w, h = size or (0, 0)
                found = _probe(str(video), video.stat().st_size, video.stat().st_mtime_ns)
                if not size and found is not None:
                    w, h = found.width, found.height
                if proxy.start(settings, video, w, h, codec):
                    log.info("再生用の動画を作り始めました: %s", video.name)
                    return 1
    return 0


# --- stamps ----------------------------------------------------------------------------------

def _stat(path) -> tuple | None:
    try:
        st = Path(path).stat()
        return (str(path), st.st_size, st.st_mtime_ns)
    except (OSError, TypeError):
        return None


def _common(repo: Repo, submission: dict, work: dict) -> list:
    meta = submission.get("meta") or {}
    sid = submission["id"]
    cams = []
    for cam in sorted(set(meta.get("cameras") or {}) | set(repo.list_cameras(sid))):
        track = repo.latest_run(sid, Stage.track, cam)
        ident = repo.latest_run(sid, Stage.identity, cam)
        boxes = (track.output or {}).get("boxes") if track is not None else None
        cams.append((cam, track.id if track else None, ident.id if ident else None,
                     _stat(boxes), _stat((meta.get("cameras") or {}).get(cam))))
    return [ev.VERSION,
            {k: work.get(k) for k in ("title", "kind", "domain", "sport", "profile",
                                       "media_removed_at", "created_at")},
            {k: meta.get(k) for k in ("cameras", "start_sec", "end_sec", "clock_offset_min")},
            cams]


def _chain(repo: Repo, run_id: int | None, depth: int = 5) -> list[int]:
    out, todo = [], [run_id] if run_id else []
    while todo and depth:
        nxt = []
        for rid in todo:
            run = repo.get_run(rid)
            if run is None:
                continue
            out.append(run.id)
            nxt += list(run.upstream_run_ids or [])
        todo, depth = nxt, depth - 1
    return out


def _subject_row(s: dict) -> tuple:
    return (s["id"], s.get("display_name"), s.get("participant_id"), s.get("excluded"),
            s.get("erased_at"), s.get("approval_status"), s.get("approved_run_id"),
            s.get("approved_at"))


def person_stamp(repo: Repo, subject_id: int) -> str:
    subject = repo.get_subject(subject_id)
    submission, work = _ids(repo, subject["submission_id"])
    latest = repo.latest_run(submission["id"], Stage.pfb, subject_id=subject_id)
    parts = [PERSON, _common(repo, submission, work), _subject_row(subject),
             latest.id if latest else None, _chain(repo, subject.get("approved_run_id"))]
    if work.get("review_mode") == "correct_later":   # the check state the video shows
        from pfb_app.services import corrections   # noqa: PLC0415
        rows = corrections.history(repo, subject_id)
        parts += [rows[-1].id if rows else None, _chain(repo, latest.id if latest else None),
                  sorted((submission.get("meta") or {}).get("people_checked") or [])]
    return hashlib.sha1(repr(parts).encode()).hexdigest()


def tracking_stamp(repo: Repo, sid: int) -> str:
    submission, work = _ids(repo, sid)
    parts = [TRACKING, _common(repo, submission, work),
             [_subject_row(s) for s in repo.list_subjects(sid)]]
    return hashlib.sha1(repr(parts).encode()).hexdigest()


def _stamp(repo: Repo, sid: int, subject_id: int | None) -> str:
    return tracking_stamp(repo, sid) if subject_id is None else person_stamp(repo, subject_id)


def _reason(repo: Repo, sid: int, subject_id: int | None,
            settings: Settings | None = None) -> str | None:
    return (tracking_reason(repo, sid, settings) if subject_id is None
            else person_reason(repo, subject_id, settings))


# --- what a screen shows -------------------------------------------------------------------------

@dataclass(frozen=True)
class Status:
    state: str      # unavailable | preparing | none | queued | running | done | stale | failed
    reason: str | None = None        # unavailable: why; failed: the job's words
    progress: float | None = None
    note: str | None = None
    path: Path | None = None
    seconds: float | None = None
    size: int | None = None
    job_id: int | None = None


def _sidecar(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(repo: Repo, settings: Settings, sid: int, subject_id: int | None) -> Status:
    """The video of a person (subject_id) or the group's 追跡の様子 (None), as a screen shows it."""
    if subject_id is not None and repo.get_subject(subject_id)["submission_id"] != sid:
        raise KeyError(f"subject {subject_id} is not in submission {sid}")
    why = _reason(repo, sid, subject_id, settings)
    if why:
        waiting = _source_check(repo, repo.get_submission(sid), settings)
        if waiting is not None and waiting[0] == why and waiting[1] is not None:
            return Status("preparing", reason=why, progress=waiting[1])
        return Status("unavailable", reason=why)
    kind = PERSON if subject_id is not None else TRACKING
    job = repo.explainer_latest(sid, kind, subject_id)
    if job is not None and job["status"] in ACTIVE:
        return Status(job["status"], progress=job["progress"], note=job["note"],
                      job_id=job["id"])
    aid = repo.get_submission(sid)["assignment_id"]
    mp4, side = file_of(settings, aid, sid, subject_id)
    meta = _sidecar(side)
    if mp4.is_file() and meta is not None and meta.get("stamp") == _stamp(repo, sid, subject_id):
        return Status("done", path=mp4, seconds=meta.get("seconds"), size=mp4.stat().st_size,
                      job_id=job["id"] if job else None)
    if job is not None and job["status"] in ("failed", "cancelled"):
        return Status("failed", reason=job["error"] or FAILED, job_id=job["id"])
    if mp4.is_file():
        return Status("stale")
    return Status("none")


# --- the request -------------------------------------------------------------------------------

def request(repo: Repo, sid: int, subject_id: int | None, *, session_id: int | None,
            actor: str | None) -> int:
    """Queue the video (or find the one already queued or being made). Refused: why not."""
    with repo._tx():
        if subject_id is not None and repo.get_subject(subject_id)["submission_id"] != sid:
            raise Refused(NOT_EVALUATED)
        why = _reason(repo, sid, subject_id)
        if why:
            raise Refused(why)
        kind = PERSON if subject_id is not None else TRACKING
        job, new = repo.explainer_enqueue(sid, kind, subject_id, _stamp(repo, sid, subject_id),
                                          session_id, actor)
        if new:
            repo.audit("explainer_request", f"subject:{subject_id}" if subject_id is not None
                       else f"submission:{sid}", {"job": job, "kind": kind})
    return job


# --- the plans ---------------------------------------------------------------------------------

def _handout(repo: Repo, subject_id: int):
    """(the group's sheets view, the person's sheet, the work, their moments in the handout's
    order): export.views.person "handout" shows the cited moments with a picture first (by
    their picture number: time, then id), then the other cited ones in time order. The pictures
    are made as build_report makes them (the same cache), so a picture that fails moves its
    moment to the end exactly as in the PDF."""
    from pfb_app.export import pictures, sheets   # noqa: PLC0415
    subject = repo.get_subject(subject_id)
    submission, work = _ids(repo, subject["submission_id"])
    # gate: a 「あとから訂正」 work's video has the PDF's scenes (its 同一人物の確認)
    sub, found = sheets.group_sheets(repo, work["id"], submission["id"], gate=True)
    sheet = next(s for s in found if s.subject_id == subject_id)
    if sheet.short:
        return sub, sheet, work, []
    asked = [m for m in sheet.moments if m._request is not None]
    made = pictures.make_pictures([m._request for m in asked], sub.cache_dir) if asked else {}
    for m in asked:
        m.picture = made.get(m._request)
    cited = [m for m in sheet.moments if m.cited]
    pictured = sorted((m for m in cited if m.picture is not None),
                      key=lambda m: (m.t_s or 0.0, m.ref_id))
    return sub, sheet, work, pictured + [m for m in cited if m.picture is None]


def handout_moments(repo: Repo, subject_id: int) -> list:
    """The person's cited moments in the handout's order (see _handout)."""
    return _handout(repo, subject_id)[3]


def _last_frame(cam, end_s: float | None) -> int | None:
    if cam.boxes is None or cam.meta is None:
        return None
    ends = [int(cam.boxes.boxes(lb)[0][-1]) for lb in cam.boxes.labels
            if len(cam.boxes.boxes(lb)[0])]
    if not ends:
        return None
    last = min(cam.meta.n_frames, max(ends))
    if end_s is not None:               # PFB3x3 registry: end_frame = round(end_s * fps) + 1
        last = min(last, round(end_s * cam.meta.fps) + 1)
    return last


def _source(settings: Settings, sub, cam) -> ev.Source | None:
    """The camera's playable file with its tracks; ev.RenderError when only a proxy would do
    and none is ready (or the file cannot be read)."""
    from pfb_app.services import moment_clip   # noqa: PLC0415
    if cam is None or cam.boxes is None or cam.video is None or not cam.fps or cam.meta is None \
            or not cam.meta.width or not cam.meta.height:
        return None
    size = (cam.meta.width, cam.meta.height)
    path, why = moment_clip.source(settings, cam.video, size)
    if path is None:
        raise ev.RenderError(why)
    return ev.Source(path, cam.fps, size, cam.boxes, sub.start_s, _last_frame(cam, sub.end_s))


def _summary(text: str | None) -> str:
    first = (text or "").strip().split("\n")[0]
    cut = first.find("。")
    return first[:cut + 1] if 0 <= cut < 80 else first[:80]


def person_plan(repo: Repo, settings: Settings, subject_id: int) -> ev.PersonPlan:
    from pfb_app.export import sheets   # noqa: PLC0415
    subject = repo.get_subject(subject_id)
    sub, sheet, work, ordered = _handout(repo, subject_id)
    pid = subject["participant_id"]
    parts, sources = [], {}
    for m in ordered[:MAX_MOMENTS]:
        part = ev.Part(m.when, m.category.label if m.category else None,
                       labels.self_words(m.what, work), labels.self_words(m.why, work))
        if m.t_s is not None and m._source is not None:
            cam = sub.camera(m._source.camera, sheets._by_stage(repo, m._source, Stage.track),
                             sheets._by_stage(repo, m._source, Stage.identity))
            key = (cam.name, cam.track.id if cam and cam.track else None) if cam else None
            if key not in sources:
                sources[key] = _source(settings, sub, cam)
            if sources[key] is not None:
                part.source = sources[key]
                part.at_s = sub.start_s + m.t_s
                part.mine = cam.labels_of(pid)
                part.stills = list(((m._source.output or {}).get("frames") or {})
                                   .get(m.ref_id) or [])
        parts.append(part)
    live = (f"確認状況：{sheet.live_label}　文の版：{sheet.live_version}" if sheet.live else None)
    return ev.PersonPlan(subject["display_name"], work["title"],
                         sheets.local_date(work.get("created_at")), _summary(sheet.text),
                         f"ほかの{labels.words_for(work).people}", parts, live)


def tracking_plan(repo: Repo, settings: Settings, sid: int) -> ev.TrackingPlan:
    from pfb_app.export import sheets   # noqa: PLC0415
    submission, work = _ids(repo, sid)
    sub = sheets._Submission(repo, work, submission)
    for cam in sub.cameras():
        src = _source(settings, sub, cam)
        if src is None:
            continue
        names = {}
        for s in repo.list_subjects(sid):
            if is_evaluated(s):
                for lb in cam.labels_of(s["participant_id"]):
                    names[lb] = s["display_name"]
        length = sub.duration_s() or TRACKING_MAX_S
        to_s = sub.start_s + min(length, TRACKING_MAX_S)
        if sub.end_s is not None:
            to_s = min(to_s, sub.end_s)
        offset = sub.clock_offset_s
        return ev.TrackingPlan(work["title"], submission.get("label") or "", src, names,
                               sub.start_s, to_s,
                               lambda t, o=offset: sheets.when_text(t, o))
    raise ev.RenderError(NO_TRACK)


# --- the worker's job --------------------------------------------------------------------------

def _session_ok(repo: Repo, session_id: int | None) -> bool:
    """The requester may still edit (None: queued without a browser session: tests, CLI)."""
    if session_id is None:
        return True
    from pfb_app.services import auth   # noqa: PLC0415
    user = auth.session_user(repo, session_id)
    return user is not None and auth.can(user.role, "edit")


def run_next(repo: Repo, settings: Settings) -> bool:
    """Make the oldest queued video. False when nothing was queued."""
    job = repo.explainer_claim()
    if job is None:
        return False
    run_job(repo, settings, job)
    return True


def run_pending(repo: Repo, settings: Settings) -> int:
    n = 0
    while run_next(repo, settings):
        n += 1
    return n


def _still_ours(repo: Repo, job: dict) -> bool:
    row = repo.explainer_job(job["id"])
    return row is not None and row["status"] == "running" and _session_ok(repo, job["session_id"])


def run_job(repo: Repo, settings: Settings, job: dict) -> None:
    sid, subject_id = job["submission_id"], job["subject_id"]
    part: Path | None = None
    try:
        if not _session_ok(repo, job["session_id"]):
            repo.explainer_finish(job["id"], "cancelled", SESSION_GONE)
            return
        why = _reason(repo, sid, subject_id, settings)
        if why:
            repo.explainer_finish(job["id"], "failed", why)
            return
        from pfb_app.ops import disk   # noqa: PLC0415
        try:
            disk.ensure_room(settings, ESTIMATE_BYTES)
        except disk.DiskLow:
            repo.explainer_finish(job["id"], "failed", LOW_DISK)
            return
        aid = repo.get_submission(sid)["assignment_id"]
        stamp = _stamp(repo, sid, subject_id)
        mp4, side = file_of(settings, aid, sid, subject_id)
        mp4.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=mp4.parent, prefix=f".{mp4.stem}.", suffix=".part.mp4")
        os.close(fd)
        part = Path(name)
        ticks = {"n": 0}

        def progress(fraction: float, note: str) -> None:
            repo.explainer_progress(job["id"], round(min(max(fraction, 0.0), 1.0), 3), note)

        def stop() -> bool:
            ticks["n"] += 1
            repo.explainer_beat(job["id"])
            return HALT.is_set() or not _still_ours(repo, job)

        if subject_id is None:
            made = ev.render_tracking(tracking_plan(repo, settings, sid), part,
                                      progress=progress, stop=stop)
        else:
            plan = person_plan(repo, settings, subject_id)
            made = ev.render_person(plan, part, progress=progress, stop=stop)
            made["moments"] = len(plan.parts)
        size = part.stat().st_size
        if size > MAX_VIDEO_BYTES:
            repo.explainer_finish(job["id"], "failed", TOO_BIG)
            return
        if repo.explainer_job(job["id"]) is None:
            return                                 # the work was deleted meanwhile
        if not mp4.parent.is_dir():                # its videos were removed meanwhile
            repo.explainer_finish(job["id"], "cancelled", CANCELLED)
            return
        with _locked(settings, aid, sid):
            if not _still_ours(repo, job):
                row = repo.explainer_job(job["id"])
                if row is not None and row["status"] == "running":
                    repo.explainer_finish(job["id"], "cancelled", SESSION_GONE)
                return
            if _reason(repo, sid, subject_id, settings) or _stamp(repo, sid, subject_id) != stamp:
                repo.explainer_finish(job["id"], "failed", CHANGED)
                return
            os.replace(part, mp4)
            part = None
            tmp = side.with_name(f".{side.name}.tmp")
            tmp.write_text(json.dumps({"stamp": stamp, "made_at": datetime.now(timezone.utc)
                                       .isoformat(timespec="seconds"), "bytes": size, **made},
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(side)
            _prune(repo, settings, aid, sid, keep=mp4)
            repo.explainer_finish(job["id"], "done")
        log.info("解説動画を作りました job=%s submission=%s subject=%s %.1f秒 %d bytes",
                 job["id"], sid, subject_id, made.get("seconds") or 0, size)
    except ev.Cancelled:
        row = repo.explainer_job(job["id"])
        if row is not None and row["status"] == "running":
            if HALT.is_set():
                repo.explainer_finish(job["id"], "failed", STOPPED)
            else:
                repo.explainer_finish(job["id"], "cancelled", SESSION_GONE)
    except ev.RenderError as e:
        repo.explainer_finish(job["id"], "failed", str(e))
    except Exception:   # noqa: BLE001 - one video never stops the worker
        log.exception("解説動画を作れませんでした job=%s", job["id"])
        try:
            repo.explainer_finish(job["id"], "failed", FAILED)
        except Exception:   # noqa: BLE001 - the group may be gone
            log.exception("解説動画のジョブの記録を更新できませんでした job=%s", job["id"])
    finally:
        if part is not None:
            part.unlink(missing_ok=True)


def _prune(repo: Repo, settings: Settings, aid: int, sid: int, keep: Path | None) -> None:
    """Under the lock: remove the videos nobody may see any more (a person who can no longer
    have one, a video whose stamp is not the current one: an excluded person's name may be in
    an old 追跡の様子), part files left by a stop, then the oldest until the group's videos take
    at most MAX_GROUP_BYTES (`keep`, just made, stays)."""
    import time   # noqa: PLC0415
    d = folder(settings, aid, sid)
    for part in d.glob(".*.part.mp4"):
        try:
            if time.time() - part.stat().st_mtime > PART_OLD_SEC:
                part.unlink(missing_ok=True)
        except OSError:
            continue
    found = []
    for mp4 in d.glob("*.mp4"):
        if mp4.name.startswith(".") or mp4 == keep:
            if mp4 == keep:
                found.append((float("inf"), mp4.stat().st_size, mp4))
            continue
        stem = mp4.stem
        try:
            subject_id = None if stem == TRACKING else int(stem[len("person_"):])
            meta = _sidecar(mp4.with_suffix(".json")) or {}
            reason = _reason(repo, sid, subject_id, settings)
            gone = reason in (NOT_EVALUATED, NOT_APPROVED, NO_MOMENTS, NOBODY, MEDIA_REMOVED,
                              NOT_USABLE_LIVE, NOT_PRINTED) \
                or (reason is None and meta.get("stamp") != _stamp(repo, sid, subject_id))
        except (KeyError, ValueError):
            gone = True
        if gone:
            _remove(mp4)
            continue
        try:
            st = mp4.stat()
        except OSError:
            continue
        found.append((st.st_mtime, st.st_size, mp4))
    total = sum(size for _, size, _ in found)
    for _, size, mp4 in sorted(found, key=lambda f: f[0]):
        if total <= MAX_GROUP_BYTES:
            break
        if mp4 == keep:
            continue
        _remove(mp4)
        total -= size


def sweep(repo: Repo, settings: Settings) -> int:
    """The worker's pass over every group's videos (_prune, under the group's lock); returns
    how many groups were looked at."""
    root = settings.data_dir / "assignments"
    n = 0
    for d in sorted(root.glob(f"*/submissions/*/handout/{FOLDER}")):
        try:
            aid, sid = int(d.parents[3].name), int(d.parents[1].name)
            if repo.get_submission(sid)["assignment_id"] != aid:
                continue
        except (KeyError, ValueError, IndexError):
            continue
        with _locked(settings, aid, sid):
            _prune(repo, settings, aid, sid, keep=None)
        n += 1
    return n


def fail_stale(repo: Repo, older_than_sec: float | None = None) -> int:
    """Videos whose job has not beaten for HUNG_SEC have failed (they can be made again)."""
    return repo.explainer_fail_stale(HUNG_SEC if older_than_sec is None else older_than_sec,
                                     HUNG)


def _remove(mp4: Path) -> None:
    try:
        mp4.unlink(missing_ok=True)
        mp4.with_suffix(".json").unlink(missing_ok=True)
    except OSError:
        log.warning("解説動画を消せませんでした: %s", mp4)


# --- erasure, retention, a worker start ---------------------------------------------------------

def drop_group(repo: Repo, settings: Settings, aid: int, sid: int,
               reason: str = CANCELLED) -> None:
    """Cancel the group's explainer jobs and remove its videos."""
    repo.explainer_cancel([sid], reason)
    remove_files(settings, aid, sid)


def remove_files(settings: Settings, aid: int, sid: int) -> None:
    """Remove the group's handout/explainer folder under the group's lock (services.erasure's
    file phase: every video of the group goes, the 追跡の様子 names everyone)."""
    d = folder(settings, aid, sid)
    if not d.exists() and not d.is_symlink():
        return
    with _locked(settings, aid, sid):
        root = files.assignment_dir(settings, aid, create=False).resolve()
        try:
            if d.is_symlink():
                d.unlink()
            elif d.is_dir() and d.resolve().is_relative_to(root):
                shutil.rmtree(d)
        except OSError:
            log.exception("解説動画のフォルダを消せませんでした: %s", d)


def drop_work(repo: Repo, settings: Settings, aid: int, reason: str = REMOVED_MEDIA) -> None:
    """Every group of the work (services.retention, before its files go)."""
    for s in repo.list_submissions(aid):
        drop_group(repo, settings, aid, s["id"], reason)


def recover(repo: Repo) -> int:
    """At worker start: a video left half made by a stopped worker has failed."""
    return repo.explainer_fail_running(STOPPED)
