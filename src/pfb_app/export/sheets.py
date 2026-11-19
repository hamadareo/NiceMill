"""What the PDFs and share pages show about each person, gathered from the recorded data.

`build_report(repo, assignment_id)` returns a Report: the work's facts and one Sheet per
evaluated person (approved or not, for the evaluator's contact sheet); `report.approved`
are the people the handouts are for, in name order.

Every number comes from a record; nothing is estimated or filled in:
- moments: the positive judgments of the person's feedback chain (the approved run's
  contribution runs and their behavior or evidence runs; for someone not approved, their
  latest feedback run, else the latest contribution runs);
- category: the engine's action_type of the moment's event (its behavior.engine.jsonl row,
  found next to the overlay frames and checked against the event's own source text); a
  moment without one is not counted by category;
- time: the engine's frame clock (else the event's video time) since the analysed start;
- on screen: the frames the person's tracks are boxed in (tracks/<id>/boxes.txt);
- court: only with the engine's line-call geometry and a 3x3 basketball work;
- pictures: the frame of each cited moment from the local video, around the person's box.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

from pfb_app.domain.models import Kind, Stage, StageRun, SubjectStatus
from pfb_app.domain.vocab import Words
from pfb_app.export import labels, pictures, tracks
from pfb_app.export.charts import Span
from pfb_app.export.labels import Category
from pfb_app.pipeline import identity_check
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.pfb3x3_bridge import (clock_seconds, format_game_clock,
                                           game_clock_offset_s, window_index)
from pfb_app.preprocess.frames import format_time
from pfb_app.pipeline.tracked_identity import label_mapping
from pfb_app.services import review
from pfb_app.services.review import MANUAL_WARNING, SHORT_MARK, subject_status
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

MAX_PICTURES = 4
ENGINE_ROWS = "behavior.engine.jsonl"
_EVENT_ID = re.compile(r"^(?P<cam>.+)-(?P<w>\d{4})-(?P<n>\d{2})$")
BASKETBALL = "basketball"


@dataclass
class Moment:
    ref_id: str
    camera: str | None
    t_s: float | None            # seconds since the analysed start (None: unknown)
    when: str                    # 「試合時間 12:30」, 「動画の時刻 03:05」, 「3ページ」
    what: str                    # as recorded (本人 / 別の参加者)
    why: str
    category: Category | None
    cited: bool
    page: int | None = None
    criteria: list[str] = field(default_factory=list)   # 評価の観点 the judgment named
    picture: Path | None = None  # the highlighted crop
    thumb: Path | None = None    # the evaluator's fallback image (review frames)
    number: int | None = None    # its picture number on the page
    other_person: bool = False   # the person's box was on another player then (no picture)
    _request: pictures.PictureRequest | None = None
    _event: dict | None = None
    _source: StageRun | None = None


@dataclass
class Sheet:
    subject_id: int
    name: str
    text: str | None
    approved: bool
    auto: bool = False
    manual: bool = False
    short: bool = False
    reason: str | None = None            # why someone is not in the PDF
    photo: Path | None = None
    moments: list[Moment] = field(default_factory=list)
    duration_s: float | None = None
    spans: list[Span] = field(default_factory=list)
    court_points: list[tuple[float, float]] = field(default_factory=list)
    distance_m: float | None = None
    court_camera: str | None = None      # the one camera court_points and distance_m are from
    team: str | None = None
    provenance: list[str] = field(default_factory=list)
    clock_offset_s: float | None = None
    judgments_false: int = 0
    retracking: bool = False              # tracks are being redone: no pictures, spans, court
    tips: list[str] = field(default_factory=list)        # 次にためしてみること the coach confirmed
    team_tips: list[str] = field(default_factory=list)   # the group's, as confirmed

    @property
    def chips(self) -> list[str]:
        if not self.approved:
            return [labels.LEFT_OUT]
        out = [labels.AUTO if self.auto else labels.APPROVED]
        if self.short:
            out.append(labels.SHORT)
        if self.manual:
            out.append(review.MANUAL_WARNING)
        return out

    @property
    def counted(self) -> int:
        return len(self.moments)

    def counts(self, categories: tuple[Category, ...]) -> list[tuple[Category, int]]:
        c = Counter(m.category.key for m in self.moments if m.category)
        return [(cat, c[cat.key]) for cat in categories if c[cat.key]]

    @property
    def untyped(self) -> int:
        return sum(1 for m in self.moments if m.category is None)

    def top_category(self, categories: tuple[Category, ...]) -> Category | None:
        counts = self.counts(categories)
        if not counts:
            return None
        best = max(n for _, n in counts)
        return next(cat for cat, n in counts if n == best)

    @property
    def on_screen_s(self) -> float | None:
        if not self.spans:
            return None
        return sum(s.end_s - s.start_s for s in self.spans)

    @property
    def pictures(self) -> list[Moment]:
        return [m for m in self.moments if m.number is not None]


@dataclass
class Report:
    work: dict
    words: Words
    categories: tuple[Category, ...]
    video: bool
    sheets: list[Sheet]                 # everyone evaluated, name order
    cameras: list[str]
    sports: bool
    # the groups' confirmed practice tips: (group name, tips), groups of approved people only
    team_tips: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def team_tips_title(self) -> str:
        from pfb_app.services import tips   # noqa: PLC0415
        return tips.team_title("sports" if self.sports else None)

    @property
    def approved(self) -> list[Sheet]:
        return [s for s in self.sheets if s.approved]

    @property
    def title(self) -> str:
        return self.work["title"]

    @property
    def date(self) -> str:
        return local_date(self.work.get("created_at"))

    @property
    def model(self) -> str:
        return labels.model_words(self.work.get("model_key"))

    @property
    def teams(self) -> list[str]:
        return sorted({s.team for s in self.sheets if s.team})


def local_date(stamp: str | None) -> str:
    try:
        d = datetime.fromisoformat(str(stamp)).astimezone()
    except (TypeError, ValueError):
        return ""
    return f"{d.year}年{d.month}月{d.day}日"


def local_time(stamp: str | None) -> str:
    try:
        d = datetime.fromisoformat(str(stamp)).astimezone()
    except (TypeError, ValueError):
        return ""
    return f"{d.month}月{d.day}日 {d.hour}:{d.minute:02d}"


def name_order(name: str) -> tuple:
    parts = re.split(r"(\d+)", name or "")
    return ([int(p) if i % 2 else p for i, p in enumerate(parts)], name or "")


# --- per camera: tracks, boxes, video, geometry, teams, engine rows ----------------------

@dataclass
class _Camera:
    name: str
    track: StageRun | None
    mapping: dict[str, str | None]
    boxes: tracks.BoxIndex | None
    meta: tracks.TrackMeta | None
    video: Path | None
    geometry: tracks.Geometry | None
    teams: dict[str, str]

    def labels_of(self, pid: str) -> list[str]:
        return sorted(lb for lb, p in self.mapping.items() if p == pid)

    @property
    def fps(self) -> float | None:
        return self.meta.fps if self.meta else None



def _by_stage(repo: Repo, run: StageRun | None, stage: Stage) -> StageRun | None:
    for rid in (run.upstream_run_ids if run else []):
        up = repo.get_run(rid)
        if up is not None and up.stage == stage:
            return up
    return None


def _probe_fps(video: Path | None) -> tracks.TrackMeta | None:
    if video is None:
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()
    if not fps or fps <= 0 or not n or n <= 0:
        return None
    return tracks.TrackMeta(float(fps), int(n), int(w) or None, int(h) or None)


class _Submission:
    """One submission's cameras and per-run caches, for every person in it."""

    def __init__(self, repo: Repo, work: dict, submission: dict, *, light: bool = False):
        """`light`: only what the moments need (person_moments): a camera is its track run,
        no boxes, video, court or teams are read."""
        self.repo, self.work, self.submission = repo, work, submission
        self.light = light
        self.sid = submission["id"]
        meta = submission.get("meta") or {}
        self.meta = meta
        self.start_s = float(meta.get("start_sec") or 0.0)
        self.end_s = None if meta.get("end_sec") is None else float(meta["end_sec"])
        # the game clock only when the coach gave it (clock_offset_min); else 開始から
        self.clock_offset_s = (game_clock_offset_s(meta, work.get("domain"))
                               if meta.get("clock_offset_min") is not None else None)
        self.camera_names = sorted(set(meta.get("cameras") or {}) | set(repo.list_cameras(self.sid)))
        self.cache_dir = (files.submission_dir(repo.settings, work["id"], self.sid, create=False)
                          / "handout" / "pictures")
        self._cams: dict[tuple, _Camera] = {}
        self._types: dict[int, dict[str, tuple[str | None, float | None]]] = {}
        self._photos: dict[int, Path] | None = None
        self._duration: float | None | bool = False
        self._retracking: bool | None = None
        self._foreign: dict[int, set[str]] = {}

    # cameras
    def camera(self, name: str | None, track: StageRun | None = None,
               identity: StageRun | None = None) -> _Camera | None:
        if not name:
            return None
        repo = self.repo
        track = track if usable(track) else repo.latest_run(self.sid, Stage.track, name)
        identity = identity if usable(identity) else repo.latest_run(self.sid, Stage.identity, name)
        key = (name, track.id if track else None, identity.id if identity else None)
        if key in self._cams:
            return self._cams[key]
        if self.light:
            cam = _Camera(name, track if usable(track) else None, {}, None, None, None, None, {})
            self._cams[key] = cam
            return cam
        try:
            mapping = label_mapping(identity.output) if usable(identity) else {}
        except (KeyError, TypeError, ValueError):
            mapping = {}
        out = (track.output or {}) if usable(track) else {}
        boxes_path = out.get("boxes")
        raw = (self.meta.get("cameras") or {}).get(name)
        video = Path(raw) if raw and Path(raw).is_file() else None
        meta = tracks.track_meta(boxes_path) or _probe_fps(video)
        geometry = (tracks.load_geometry(boxes_path)
                    if labels.sports(self.work) and self.work.get("sport") == BASKETBALL else None)
        teams: dict[str, str] = {}
        if usable(track):
            try:
                teams = review.names_view(repo, self.sid, name, track, fuse=False).teams
            except Exception:   # noqa: BLE001 - teams are optional for the PDF
                log.info("no teams for the PDF (submission %s, %s)", self.sid, name)
        cam = _Camera(name, track if usable(track) else None, mapping,
                      tracks.load_boxes(boxes_path), meta, video, geometry, teams)
        self._cams[key] = cam
        return cam

    def retracking(self, chain_tracks=()) -> bool:
        """True while the tracks may not belong to the runs the text stands on: a track job
        of the submission is active, or a boxes.txt is newer than its track run: the newest
        track runs, or `chain_tracks` (the track runs the text's chain stands on; every track
        run writes the same boxes.txt, so an older one's labels would be looked up in a later
        run's boxes; final review MED 6)."""
        if self._retracking is None:
            busy = any(self.repo.has_active_job(self.sid, Stage.track, c)
                       for c in self.camera_names)
            self._retracking = busy or any(_boxes_newer(c.track) for c in self.cameras())
        return self._retracking or any(_boxes_newer(t) for t in chain_tracks if t is not None)

    def cameras(self) -> list[_Camera]:
        return [c for c in (self.camera(n) for n in self.camera_names) if c is not None]

    def duration_s(self) -> float | None:
        """The analysed seconds: the track runs' analysed_s, else the video length from the
        task start (to its end, if set)."""
        if self._duration is not False:
            return self._duration
        found = []
        for cam in self.cameras():
            analysed = ((cam.track.output or {}) if cam.track else {}).get("analysed_s")
            if isinstance(analysed, (int, float)) and analysed > 0:
                found.append(float(analysed))
            elif cam.meta:
                length = cam.meta.n_frames / cam.meta.fps
                end = min(length, self.end_s) if self.end_s is not None else length
                if end - self.start_s > 0:
                    found.append(end - self.start_s)
        self._duration = max(found) if found else None
        return self._duration

    def foreign(self, source: StageRun) -> set[str]:
        """The behavior run's events that happened while the person's box was on another
        player (pipeline.identity_check; empty without a number timeline)."""
        if source.id not in self._foreign:
            try:
                self._foreign[source.id] = identity_check.foreign_events(self.repo, source)
            except Exception:   # noqa: BLE001 - the check only removes pictures
                log.info("no number timeline check for the PDF (run %s)", source.id)
                self._foreign[source.id] = set()
        return self._foreign[source.id]

    def photos(self) -> dict[int, Path]:
        if self._photos is None:
            from pfb_app.services.auto_people import person_crops
            try:
                self._photos = person_crops(self.repo, self.repo.settings, self.sid)
            except Exception:   # noqa: BLE001 - a photo is optional
                log.info("no person crops for the PDF (submission %s)", self.sid)
                self._photos = {}
        return self._photos

    # the engine's action types of a behavior run's events
    def action_types(self, source: StageRun, cam: _Camera | None) -> dict[str, tuple]:
        if source.id in self._types:
            return self._types[source.id]
        found: dict[str, tuple[str | None, float | None]] = {}
        out = source.output or {}
        rows_file = _rows_file(out, cam, self.repo, self.work["id"], self.sid)
        if rows_file is not None:
            texts = out.get("source_text") or {}
            rows = _engine_rows(rows_file)
            for e in out.get("events") or []:
                m = _EVENT_ID.match(str(e.get("event_id") or ""))
                if not m:
                    continue
                row = rows.get(int(m["w"]))
                n = int(m["n"])
                evs = (row or {}).get("events") or []
                if not 1 <= n <= len(evs) or not isinstance(evs[n - 1], dict):
                    continue
                ev = evs[n - 1]
                said = texts.get(e["event_id"])
                if said is not None and str(ev.get("observed_action") or "").strip() != said.strip():
                    continue           # not the row this event was made from
                found[e["event_id"]] = (ev.get("action_type"), clock_seconds(ev.get("clock")))
        self._types[source.id] = found
        return found


BOXES_SLACK_S = 2.0


def _boxes_newer(track: StageRun | None) -> bool:
    """boxes.txt written after its track run was saved (a new track is being made)."""
    if track is None:
        return False
    boxes = (track.output or {}).get("boxes")
    try:
        saved = datetime.fromisoformat(str(track.created_at)).timestamp()
        return Path(boxes).stat().st_mtime > saved + BOXES_SLACK_S
    except (TypeError, ValueError, OSError):
        return False


def _rows_file(out: dict, cam: _Camera | None, repo: Repo, aid: int, sid: int) -> Path | None:
    """The engine's behavior.engine.jsonl of a behavior run: next to (above) its overlay
    frames, else the only one for the camera's video id under the submission's engine dir."""
    for shots in (out.get("frames") or {}).values():
        for shot in shots[:1]:
            p = Path(shot).parent
            for _ in range(4):
                if (p / ENGINE_ROWS).is_file():
                    return p / ENGINE_ROWS
                p = p.parent
        break
    if cam is None or cam.track is None:
        return None
    vid = (cam.track.output or {}).get("video_id")
    if not vid:
        return None
    root = files.submission_dir(repo.settings, aid, sid, create=False) / "pfb3x3" / "exp"
    found = sorted(root.glob(f"*/*/{vid}/{ENGINE_ROWS}"))
    return found[0] if len(found) == 1 else None


def _engine_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rows[window_index(row)] = row
                except Exception:   # noqa: BLE001 - one bad row is skipped
                    continue
    except (OSError, UnicodeDecodeError):
        return {}
    return rows


def _merge(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start_s):
        if out and s.start_s <= out[-1].end_s + 1e-6:
            out[-1] = Span(out[-1].start_s, max(out[-1].end_s, s.end_s))
        else:
            out.append(s)
    return out


# --- one person ---------------------------------------------------------------------------

def when_text(t_s: float | None, clock_offset_s: float | None, video_time: str | None = None,
              camera: str | None = None) -> str:
    """A moment's time in the one clock of the page (the timeline's ticks use the same):
    試合時間 when the game clock is known, else 開始から (since the analysed start); 動画の時刻
    only for a moment whose place in the analysed part is unknown."""
    if t_s is not None and clock_offset_s is not None:
        when = f"試合時間 {format_game_clock(clock_offset_s + t_s)}"
    elif t_s is not None:
        when = f"開始から {format_time(max(t_s, 0.0))}"
    else:
        when = f"動画の時刻 {video_time or ''}".strip()
    return f"{labels.camera_word(camera)} {when}" if camera else when


def _chain(repo: Repo, sub: _Submission, run: StageRun | None) -> list[tuple[StageRun, StageRun]]:
    """(contribution run, its behavior / evidence run) pairs the person's text stands on."""
    if run is not None:
        cons = [repo.get_run(i) for i in run.upstream_run_ids]
    else:
        names = sub.camera_names or [None]
        cons = [repo.latest_run(sub.sid, Stage.contribution, c) for c in names]
    out = []
    for con in cons:
        if not usable(con) or con.stage != Stage.contribution or not con.upstream_run_ids:
            continue
        source = repo.get_run(con.upstream_run_ids[0])
        if usable(source):
            out.append((con, source))
    return out


def _category(work: dict, action_type: str | None, judgment: dict) -> Category | None:
    key = labels.category_of(work, action_type, judgment.get("rubric_criteria") or ())
    return None if key is None else labels.category(work, key)


def _moments(report_work: dict, sub: _Submission, subject: dict, chain, cited: list[str],
             several: bool) -> tuple[list[Moment], int]:
    pid = subject.get("participant_id")
    moments: dict[str, Moment] = {}
    false_count = 0
    for con, source in chain:
        out = source.output or {}
        events = {e["event_id"]: e for e in out.get("events") or []}
        items = {i["item_id"]: i for i in out.get("items") or []}
        cam = None
        types: dict = {}
        if events:
            track = _by_stage(sub.repo, source, Stage.track)
            identity = _by_stage(sub.repo, source, Stage.identity)
            cam = sub.camera(source.camera, track, identity)
            types = sub.action_types(source, cam)
        for j in (con.output or {}).get("judgments") or []:
            ref = j.get("ref_id")
            e, item = events.get(ref), items.get(ref)
            if e is not None and pid is not None and e.get("participant") != pid:
                continue
            if e is None and item is None:
                continue
            if not j.get("is_positive_contribution"):
                false_count += 1
                continue
            if ref in moments:
                continue
            why = j.get("contribution_interpretation") or ""
            if e is not None:
                kind, clock = types.get(ref, (None, None))
                t = clock if clock is not None else (
                    None if clock_seconds(e.get("video_time")) is None
                    else clock_seconds(e.get("video_time")) - sub.start_s)
                moments[ref] = Moment(ref, e.get("camera"), t,
                                      when_text(t, sub.clock_offset_s, e.get("video_time"),
                                                e.get("camera") if several else None),
                                      e.get("observed_behavior") or "", why,
                                      _category(report_work, kind, j),
                                      ref in cited, criteria=list(j.get("rubric_criteria") or []),
                                      _event=e, _source=source)
            else:
                moments[ref] = Moment(ref, None, None, f"{item['page']}ページ",
                                      item.get("quote") or item.get("observed_content") or "",
                                      why, _category(report_work, None, j), ref in cited,
                                      page=item.get("page"),
                                      criteria=list(j.get("rubric_criteria") or []))
    found = list(moments.values())
    found.sort(key=lambda m: (m.t_s is None, m.t_s or 0.0, m.page or 0, m.ref_id))
    return found, false_count


def _picture_request(sub: _Submission, m: Moment, pid: str) -> pictures.PictureRequest | None:
    if m.t_s is None or m._source is None:
        return None
    cam = sub.camera(m._source.camera, _by_stage(sub.repo, m._source, Stage.track),
                     _by_stage(sub.repo, m._source, Stage.identity))
    if cam is None or cam.boxes is None or cam.video is None or not cam.fps:
        return None
    frame = round((sub.start_s + m.t_s) * cam.fps) + 1
    hit = cam.boxes.nearest(cam.labels_of(pid), frame, tol=max(1, round(cam.fps)))
    if hit is None:
        return None
    return pictures.PictureRequest(cam.video, hit[0], hit[1])


def _sheet(repo: Repo, work: dict, sub: _Submission, subject: dict, video: bool,
           several: bool) -> Sheet:
    from pfb_app.services import short_text
    status = subject_status(repo, subject["id"])
    approved = status == SubjectStatus.approved
    run = (repo.get_run(subject["approved_run_id"]) if approved
           else repo.latest_run(sub.sid, Stage.pfb, subject_id=subject["id"]))
    warnings = run.warnings if run else []
    cited = list(((run.output or {}).get("evidence_refs") or []) if run else [])
    chain = _chain(repo, sub, run)
    moments, false_count = _moments(work, sub, subject, chain, cited, several)
    reason = None
    if not approved:
        reason = labels.WHY_NOT.get(status, "")
        if status == SubjectStatus.insufficient and short_text.offer(repo, subject, run):
            reason = "短い文の承認待ち"
    sheet = Sheet(subject_id=subject["id"], name=subject["display_name"],
                  text=(run.output or {}).get("pfb") if approved else None, approved=approved,
                  auto=approved and review.approved_automatically(repo, subject),
                  manual=MANUAL_WARNING in warnings, short=SHORT_MARK in warnings,
                  reason=reason, moments=moments, clock_offset_s=sub.clock_offset_s,
                  judgments_false=false_count)
    pid = subject.get("participant_id")
    chain_tracks = [_by_stage(repo, source, Stage.track) for _, source in chain]
    if video and pid and sub.retracking(chain_tracks):
        sheet.retracking = True
        sheet.photo = sub.photos().get(subject["id"])
    elif video and pid:
        sheet.photo = sub.photos().get(subject["id"])
        sheet.duration_s = sub.duration_s()
        spans, teams = [], Counter()
        court: list[tuple[int, list]] = []          # (fps, cleaned samples) per camera
        court_cams: list[str] = []
        for cam in sub.cameras():
            mine = cam.labels_of(pid)
            if not mine or cam.boxes is None or not cam.fps:
                continue
            if sheet.duration_s:
                spans += tracks.spans(cam.boxes.frames(mine), cam.fps, start_s=sub.start_s,
                                      duration_s=sheet.duration_s)
            if cam.geometry is not None:
                court.append((cam.fps, tracks.clean_samples(
                    tracks.court_samples(cam.boxes, mine, cam.geometry, cam.fps), cam.fps)))
                court_cams.append(cam.name)
            teams.update(cam.teams[lb] for lb in mine if cam.teams.get(lb) in ("T1", "T2"))
        sheet.spans = _merge(spans)
        fps, best = best_court(court)
        sheet.court_points = [(x, y) for _, x, y in best]
        sheet.distance_m = tracks.distance_m(best, fps) if best else None
        sheet.court_camera = court_cams[court.index((fps, best))] if best else None
        if teams:
            from pfb_app.services.auto_people import TEAM_WORDS
            sheet.team = TEAM_WORDS.get(teams.most_common(1)[0][0])
        if approved:
            shown = [m for m in sorted(moments, key=lambda m: cited.index(m.ref_id)
                                       if m.ref_id in cited else 99) if m.cited]
            for m in shown[:MAX_PICTURES]:
                if m._source is not None and m.ref_id in sub.foreign(m._source):
                    m.other_person = True     # never a picture of another player
                    continue
                m._request = _picture_request(sub, m, pid)
    if approved and run is not None:
        sheet.provenance = _provenance(repo, run, chain, sheet)
    if approved and video:
        from pfb_app.services import tips   # noqa: PLC0415 - tips import the pipeline
        sheet.tips = tips.for_pdf(repo, sub.sid, subject["id"])
        sheet.team_tips = tips.team_for_pdf(repo, sub.sid)
    return sheet


def person_moments(repo: Repo, subject_id: int) -> list[Moment] | None:
    """The good moments of the person's approved text (as the handout counts them, with their
    categories), or None when the person has no approved text now (review.subject_status:
    an approval the text has outgrown does not count). Nothing but the database and the
    engine's rows file is read (成長の記録 reads this for every work of a person)."""
    subject = repo.get_subject(subject_id)
    if subject_status(repo, subject_id) != SubjectStatus.approved:
        return None
    submission = repo.get_submission(subject["submission_id"])
    work = repo.get_assignment(submission["assignment_id"])
    sub = _Submission(repo, work, submission, light=True)
    run = repo.get_run(subject["approved_run_id"])
    cited = list((run.output or {}).get("evidence_refs") or [])
    moments, _ = _moments(work, sub, subject, _chain(repo, sub, run), cited,
                          len(sub.camera_names) > 1)
    return moments


def best_court(per_camera: list[tuple[float, list]]) -> tuple[float, list]:
    """The one camera's court samples to show and measure: the camera with the most samples
    (never two cameras added up: the same steps would be counted twice)."""
    if not per_camera:
        return 0.0, []
    return max(per_camera, key=lambda fs: len(fs[1]))


def _provenance(repo: Repo, run: StageRun, chain, sheet: Sheet) -> list[str]:
    how = "、".join(sheet.chips)
    lines = [f"文章：{how}（{local_time(run.created_at)}に作成）" if run.created_at
             else f"文章：{how}"]
    sources = []
    for con, source in chain:
        what = "行動の記録" if source.stage == Stage.behavior else "書いた所の記録"
        cam = f"{labels.camera_word(source.camera)}の" if source.camera else ""
        sources.append(f"{cam}{what} #{source.id}・よかった点の判断 #{con.id}")
    if sources:
        lines.append("もとにした解析：" + "、".join(sources) + f"・文章 #{run.id}")
    if sheet.judgments_false:
        lines.append(f"よかった点に入れなかった場面：{sheet.judgments_false}件")
    if sheet.untyped:
        lines.append(f"種類がわからない場面：{sheet.untyped}件（グラフには数えていません）")
    if sheet.retracking:
        lines.append(labels.RETRACK_NOTE)
    return lines


def group_sheets(repo: Repo, assignment_id: int, submission_id: int) -> tuple[_Submission,
                                                                             list[Sheet]]:
    """One video group's submission view and the sheets of its evaluated people (name
    order), without pictures (試合・活動のまとめ: services.dashboard)."""
    work = repo.get_assignment(assignment_id)
    submission = repo.get_submission(submission_id)
    if submission["assignment_id"] != assignment_id:
        raise KeyError(f"submission {submission_id} is not in assignment {assignment_id}")
    sub = _Submission(repo, work, submission)
    several = len(sub.camera_names) > 1
    sheets = [_sheet(repo, work, sub, subject, True, several)
              for subject in repo.list_subjects(submission_id) if is_evaluated(subject)]
    sheets.sort(key=lambda s: name_order(s.name))
    return sub, sheets


def build_report(repo: Repo, assignment_id: int, *, with_pictures: bool = True) -> Report:
    work = repo.get_assignment(assignment_id)
    video = work["kind"] == Kind.video.value
    sheets: list[Sheet] = []
    cameras: set[str] = set()
    requests: dict[Path, list[Moment]] = {}
    by_group: list[tuple[str, list[Sheet]]] = []
    for submission in repo.list_submissions(assignment_id):
        sub = _Submission(repo, work, submission)
        cameras.update(sub.camera_names)
        several = len(sub.camera_names) > 1
        mine: list[Sheet] = []
        by_group.append((submission["label"], mine))
        for subject in repo.list_subjects(submission["id"]):
            if video and not is_evaluated(subject):
                continue
            sheet = _sheet(repo, work, sub, subject, video, several)
            sheets.append(sheet)
            mine.append(sheet)
            for m in sheet.moments:
                if m._request is not None:
                    requests.setdefault(sub.cache_dir, []).append(m)
    if with_pictures:
        for cache_dir, ms in requests.items():
            made = pictures.make_pictures([m._request for m in ms], cache_dir)
            for m in ms:
                m.picture = made.get(m._request)
            pictures.prune(cache_dir, set(made.values()))
    for sheet in sheets:
        if not sheet.approved:
            continue
        numbered = sorted((m for m in sheet.moments if m.picture is not None),
                          key=lambda m: (m.t_s or 0.0, m.ref_id))
        for k, m in enumerate(numbered, 1):
            m.number = k
        for m in sheet.moments:
            if sheet.retracking:
                break
            if m.cited and m.picture is None and m._event is not None and m._source is not None \
                    and not m.other_person:
                shots = [f.path for f in review.frames_for(repo, m._source, m._event)
                         if f.path.is_file()]
                m.thumb = shots[0] if shots else None
    sheets.sort(key=lambda s: name_order(s.name))
    groups = [(label, next(t.team_tips for t in ss if t.approved and t.team_tips))
              for label, ss in by_group if any(t.approved and t.team_tips for t in ss)]
    return Report(work=work, words=labels.words_for(work), categories=labels.categories_for(work),
                  video=video, sheets=sheets, cameras=sorted(cameras), sports=labels.sports(work),
                  team_tips=groups)
