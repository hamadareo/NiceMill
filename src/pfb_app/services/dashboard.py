"""試合・活動のまとめ: the numbers of one video group, from what the handouts stand on.

`build(repo, aid, sid)` returns a Board. Every number is read, never estimated anew:
- people: the group's evaluated people (pipeline.auto_people.is_evaluated), so a person whose
  data was removed (erased_at), one taken out of the evaluation and any tracked label nobody
  is mapped to never appear and never count in a team total;
- moments, categories, on-screen time, court points and distance: export.sheets (the same
  chain as the handouts and the evaluator PDF; a waiting person's moments are those of the
  current draft, as on the evaluator PDF);
- distance: only with the court geometry, from one camera (Sheet.court_camera), with that
  camera's position error when geometry.json gives one (`h_unc_cm`);
- the image-plane positions (no geometry): the foot point (box bottom centre) of the person's
  boxes, one camera at a time (image planes are never merged), every POINT_STEP_S.

A work whose videos were removed (assignments.media_removed_at) lost its boxes with them, so
it keeps its moments and texts only, and nothing can be played.

A 「あとから訂正」 work (`Board.live`) has no approvals: a person's text is the newest one, and it
is used (`Person.approved`, the questions of services.ask) unless it is reported wrong; the
status is the check state (訂正なし / 訂正済み 版N / 誤りの報告あり・確認中), never 承認.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from pfb_app.domain.vocab import Words
from pfb_app.export import labels, sheets as sheet_mod, tracks
from pfb_app.export.labels import Category
from pfb_app.pipeline.pfb3x3_bridge import format_game_clock
from pfb_app.services.corrections import STATE_REPORT
from pfb_app.preprocess.frames import format_time
from pfb_app.storage.db import Repo

POINT_STEP_S = 0.5            # the image-plane samples, as export.tracks.court_samples
HIGHLIGHTS_EACH, HIGHLIGHTS_MAX = 2, 10
UNKNOWN_TEAM = "チームがわからない人"
APPROVED = "承認済み"
NOT_YET = "承認前"
SEVERAL_CAMERAS = ("カメラが複数あり、カメラどうしの時刻がそろっているとはかぎらないため、"
                   "合わせた映っていた時間はおおよそです")


@dataclass
class Person:
    subject_id: int
    name: str
    team: str | None
    approved: bool                     # the text is used (approved; live: not reported wrong)
    status: str                        # 承認済み, 確認待ち, ... (live: the check state)
    text: str | None                   # the text used (None before approval / while reported)
    total: int                         # good moments
    counts: dict[str, int]             # category key -> good moments (typed only)
    untyped: int
    on_screen_s: float | None
    distance_m: float | None
    court_camera: str | None
    position_error_cm: float | None
    court_points: list[tuple[float, float]] = field(default_factory=list)
    frame_points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    spans: list = field(default_factory=list)
    retracking: bool = False
    # seconds on screen per camera: cameras are not time-aligned, so on_screen_s (their
    # union) is approximate when there are several
    on_screen_by_camera: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Moment:
    subject_id: int
    name: str
    team: str | None
    t_s: float | None                  # seconds since the analysed start
    when: str                          # 試合時間 12:30 / 開始から 03:05
    category: Category | None
    what: str                          # as recorded (本人 / 別の参加者)
    cited: bool
    camera: str | None
    cite_rank: int                     # its place in the text's evidence_refs (99: not cited)
    approved: bool = True              # False: the person's text is not approved yet (承認前)
    state: str = ""                    # the person's status word for the list (live: check state)


@dataclass
class Board:
    work: dict
    submission_id: int
    group_label: str
    words: Words
    categories: tuple[Category, ...]
    sports: bool
    court: bool                        # someone has court points (the geometry exists)
    duration_s: float | None
    clock_offset_s: float | None
    start_s: float
    people: list[Person]
    moments: list[Moment]
    media_removed: bool
    retracking: bool
    cameras: list[str]
    videos: dict[str, Path]            # camera -> the local video still there
    frame_sizes: dict[str, tuple[int, int]]
    live: bool = False                 # a 「あとから訂正」 work: no approvals, the check states

    @property
    def several_cameras(self) -> bool:
        """Someone was seen by more than one camera (their minutes are then approximate)."""
        return any(len(p.on_screen_by_camera) > 1 for p in self.people)

    @property
    def minute_cameras(self) -> list[str]:
        return sorted({c for p in self.people for c in p.on_screen_by_camera})

    def tick(self, t: float) -> str:
        if self.clock_offset_s is not None:
            return format_game_clock(self.clock_offset_s + t)
        return format_time(max(t, 0.0))

    @property
    def untyped(self) -> int:
        return sum(p.untyped for p in self.people)

    @property
    def total(self) -> int:
        return sum(p.total for p in self.people)

    def person(self, subject_id: int) -> Person | None:
        return next((p for p in self.people if p.subject_id == subject_id), None)


_ACTIVE = ("queued", "running", "waiting_gpu")


def stamp(repo: Repo, aid: int) -> str:
    """What a board depends on in the database, as one hash: the work's row (every column:
    domain, sport, profile, task text, media removal, ...), its groups (label and meta), the
    runs (newest id and count), the people (name, exclusion, erasure, approval) and the jobs
    (newest id, and which are queued, running or waiting). Any change there changes it."""
    q = repo.conn.execute
    parts = [
        tuple(q("SELECT * FROM assignments WHERE id=?", (aid,)).fetchone() or ()),
        [tuple(r) for r in q("SELECT id, label, meta FROM submissions WHERE assignment_id=?"
                             " ORDER BY id", (aid,))],
        tuple(q("SELECT COALESCE(MAX(r.id), 0), COUNT(r.id) FROM stage_runs r JOIN submissions b"
                " ON r.submission_id = b.id WHERE b.assignment_id=?", (aid,)).fetchone()),
        [tuple(r) for r in q(
            "SELECT s.id, s.display_name, s.excluded, s.erased_at, s.approved_run_id,"
            " s.approved_at, s.participant_id FROM subjects s JOIN submissions b"
            " ON s.submission_id = b.id WHERE b.assignment_id=? ORDER BY s.id", (aid,))],
        tuple(q("SELECT COALESCE(MAX(j.id), 0) FROM jobs j JOIN submissions b"
                " ON j.submission_id = b.id WHERE b.assignment_id=?", (aid,)).fetchone()),
        tuple(q("SELECT COALESCE(MAX(v.id), 0) FROM text_versions v JOIN subjects s"   # corrections
                " ON v.subject_id = s.id JOIN submissions b ON s.submission_id = b.id"
                " WHERE b.assignment_id=?", (aid,)).fetchone()),
        [tuple(r) for r in q(
            "SELECT j.id, j.stage, j.camera, j.status FROM jobs j JOIN submissions b"
            " ON j.submission_id = b.id WHERE b.assignment_id=? AND j.status IN (?,?,?)"
            " ORDER BY j.id", (aid, *_ACTIVE))],
    ]
    return hashlib.sha1(repr(parts).encode()).hexdigest()


def groups(repo: Repo, aid: int) -> list[tuple[int, str]]:
    """The work's video groups: (submission id, label), in order."""
    return [(s["id"], s.get("label") or f"グループ{k}")
            for k, s in enumerate(repo.list_submissions(aid), 1)]


def _position_error(cam) -> float | None:
    """The camera's position error in cm (geometry.json h_unc_cm), None when not given."""
    boxes = ((cam.track.output or {}) if cam is not None and cam.track else {}).get("boxes")
    if not boxes:
        return None
    try:
        doc = json.loads((Path(boxes).parent / "linecall" / "geometry.json")
                         .read_text(encoding="utf-8"))
        v = float(doc.get("h_unc_cm"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _frame_points(cam, labels_: list[str]) -> list[tuple[float, float]]:
    if cam.boxes is None or not cam.fps or not labels_:
        return []
    step = max(1, round(POINT_STEP_S * cam.fps))
    picked: dict[int, tuple[float, float]] = {}
    for lb in labels_:
        frames, boxes = cam.boxes.boxes(lb)
        last = -10 ** 9
        for f, (x, y, w, h) in zip(frames, boxes):
            if f - last >= step and int(f) not in picked:
                picked[int(f)] = (float(x + w / 2), float(y + h))
                last = f
    return [picked[f] for f in sorted(picked)]


def build(repo: Repo, aid: int, sid: int | None = None) -> Board:
    """The board of group `sid` (None: the work's first group). KeyError: no such work or
    group, or a work without groups."""
    work = repo.get_assignment(aid)
    found = groups(repo, aid)
    if not found:
        raise KeyError(f"assignment {aid} has no group")
    sid = found[0][0] if sid is None else sid
    label = dict(found).get(sid)
    if label is None:
        raise KeyError(f"submission {sid} is not in assignment {aid}")
    sub, sheets = sheet_mod.group_sheets(repo, aid, sid)
    live = work.get("review_mode") == "correct_later"
    removed = bool(work.get("media_removed_at"))
    cams = {c.name: c for c in sub.cameras()}
    subjects = {s["id"]: s for s in repo.list_subjects(sid)}
    people, moments = [], []
    categories = labels.categories_for(work)
    for s in sheets:
        pid = subjects[s.subject_id].get("participant_id")
        frame_points: dict[str, list] = {}
        by_camera: dict[str, float] = {}
        if not s.retracking and not removed:
            for name, cam in cams.items():
                mine = cam.labels_of(pid)
                pts = _frame_points(cam, mine)
                if pts:
                    frame_points[name] = pts
                if mine and cam.boxes is not None and cam.fps and s.duration_s:
                    found = tracks.spans(cam.boxes.frames(mine), cam.fps, start_s=sub.start_s,
                                         duration_s=s.duration_s)
                    if found:
                        by_camera[name] = sum(sp.end_s - sp.start_s for sp in found)
        counts: dict[str, int] = {}
        for cat, n in s.counts(categories):
            counts[cat.key] = n
        court_cam = cams.get(s.court_camera) if s.court_camera else None
        # live: the newest text is used unless it is reported wrong (corrections.usable_run)
        used = s.approved or (s.live and s.live_label != STATE_REPORT)
        status = (s.live_label if s.live else (s.reason or "")) if not s.approved else APPROVED
        people.append(Person(
            subject_id=s.subject_id, name=s.name, team=s.team, approved=used,
            status=status, text=s.text if used else None,
            total=s.counted, counts=counts, untyped=s.untyped,
            on_screen_s=None if removed else s.on_screen_s,
            distance_m=None if removed else s.distance_m,
            court_camera=s.court_camera,
            position_error_cm=_position_error(court_cam) if s.distance_m is not None else None,
            court_points=[] if removed else list(s.court_points), frame_points=frame_points,
            spans=[] if removed else list(s.spans), retracking=s.retracking,
            on_screen_by_camera=by_camera))
        refs = _evidence_refs(repo, s)
        for m in s.moments:
            moments.append(Moment(s.subject_id, s.name, s.team, m.t_s, m.when, m.category,
                                  m.what, m.cited, m.camera,
                                  refs.index(m.ref_id) if m.ref_id in refs else 99,
                                  used, APPROVED if s.approved else (
                                      status if live else NOT_YET)))
    moments.sort(key=lambda m: (m.t_s is None, m.t_s or 0.0, sheet_mod.name_order(m.name)))
    videos = {} if removed else {n: c.video for n, c in cams.items() if c.video is not None}
    sizes = {n: (c.meta.width, c.meta.height) for n, c in cams.items()
             if c.meta and c.meta.width and c.meta.height}
    return Board(work=work, submission_id=sid, group_label=label,
                 words=labels.words_for(work), categories=categories,
                 sports=labels.sports(work), court=any(p.court_points for p in people),
                 duration_s=sub.duration_s(), clock_offset_s=sub.clock_offset_s,
                 start_s=sub.start_s, people=people, moments=moments, media_removed=removed,
                 retracking=any(p.retracking for p in people), cameras=sorted(cams),
                 videos=videos, frame_sizes=sizes, live=live)


def _evidence_refs(repo: Repo, sheet) -> list[str]:
    """The order the person's text cites its moments in (its pfb run's evidence_refs)."""
    subject = repo.get_subject(sheet.subject_id)
    from pfb_app.domain.models import Stage   # noqa: PLC0415
    run = (repo.get_run(subject["approved_run_id"]) if sheet.approved and
           subject.get("approved_run_id") else
           repo.latest_run(subject["submission_id"], Stage.pfb, subject_id=sheet.subject_id))
    return list(((run.output or {}).get("evidence_refs") or []) if run else [])


# --- aggregates -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Team:
    name: str                 # チーム1 / チームがわからない人 / the group as a whole
    people: int
    counts: dict[str, int]
    untyped: int
    total: int


def team_members(board: Board) -> list[tuple[str, list[Person]]]:
    """[(team, its evaluated people)] for a sports group whose teams are known (people
    without a team last, as 「チームがわからない人」); else one entry for the whole group."""
    known = sorted({p.team for p in board.people if p.team})
    if not board.sports or not known:
        return [(f"{board.group_label}の全員", list(board.people))]
    out = [(t, [p for p in board.people if p.team == t]) for t in known]
    rest = [p for p in board.people if not p.team]
    if rest:
        out.append((UNKNOWN_TEAM, rest))
    return out


def team_totals(board: Board) -> list[Team]:
    """Good moments per team_members entry; only the evaluated people are counted."""
    out = []
    for name, members in team_members(board):
        counts: dict[str, int] = {}
        for p in members:
            for k, n in p.counts.items():
                counts[k] = counts.get(k, 0) + n
        out.append(Team(name, len(members), counts, sum(p.untyped for p in members),
                        sum(p.total for p in members)))
    return out


def highlights(board: Board, each: int = HIGHLIGHTS_EACH,
               limit: int = HIGHLIGHTS_MAX) -> list[Moment]:
    """見どころ: the moments the texts cite first (evidence_refs order), at most `each` per
    person and `limit` in all, shown in time order."""
    picked: list[Moment] = []
    for p in board.people:
        mine = sorted((m for m in board.moments if m.subject_id == p.subject_id and m.cited),
                      key=lambda m: m.cite_rank)
        picked += mine[:each]
    picked.sort(key=lambda m: (m.cite_rank, m.t_s if m.t_s is not None else 1e12))
    picked = picked[:limit]
    return sorted(picked, key=lambda m: (m.t_s is None, m.t_s or 0.0))


@dataclass(frozen=True)
class Scene:
    start_s: float
    end_s: float
    label: str                                    # 「03:00〜04:00」
    who: list[tuple[str, list[tuple[Category | None, int]]]]   # (name, [(category, n)])
    first_t: float                                # the first moment in it (the jump)
    moments: int


def bin_size(duration_s: float | None) -> float:
    """The width of one 場面 of 場面ごとの様子 (the timeline's tick step)."""
    from pfb_app.export.charts import _tick_step   # noqa: PLC0415
    return float(_tick_step(duration_s or 60.0))


def scenes(board: Board) -> list[Scene]:
    """場面ごとの様子: who had which kind of good moment in each time bin (bins without a
    moment are left out; moments whose time is unknown are not placed)."""
    timed = [m for m in board.moments if m.t_s is not None]
    if not timed:
        return []
    size = bin_size(board.duration_s)
    order = {c.key: i for i, c in enumerate(board.categories)}
    bins: dict[int, list[Moment]] = {}
    for m in timed:
        bins.setdefault(int(max(m.t_s, 0.0) // size), []).append(m)
    out = []
    for k in sorted(bins):
        ms = bins[k]
        names: dict[str, dict] = {}
        for m in ms:
            key = m.category.key if m.category else ""
            entry = names.setdefault(m.name, {})
            entry.setdefault(key, [m.category, 0])[1] += 1
        who = [(name, sorted(((c, n) for c, n in cats.values()),
                             key=lambda cn: order.get(cn[0].key, 99) if cn[0] else 100))
               for name, cats in sorted(names.items(), key=lambda kv: sheet_mod.name_order(kv[0]))]
        a, b = k * size, (k + 1) * size
        if board.duration_s:
            b = min(b, board.duration_s)
        out.append(Scene(a, b, f"{board.tick(a)}〜{board.tick(b)}", who,
                         min(m.t_s for m in ms), len(ms)))
    return out


# --- CSV ------------------------------------------------------------------------------------

_FORMULA = ("=", "+", "-", "@", "\t", "\r")


def _cell(v) -> str:
    """A CSV cell; text that a spreadsheet would read as a formula gets a leading '."""
    if v is None:
        return ""
    s = str(v)
    return "'" + s if s.startswith(_FORMULA) else s


def _csv(rows: list[list]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    for r in rows:
        w.writerow([_cell(v) for v in r])
    return buf.getvalue().encode("utf-8-sig")     # the BOM: Excel reads it as UTF-8


def minutes(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds / 60, 1)


def distance_shown(metres: float | None) -> int | None:
    """Rounded as the handouts (export.views.distance_text): 1 m, 5 m from 50 m, 10 m from
    200 m; None below 1 m or without a distance."""
    if metres is None or metres < 1:
        return None
    step = 1 if metres < 50 else 5 if metres < 200 else 10
    return int(round(metres / step) * step)


def people_csv(board: Board) -> bytes:
    cats = board.categories
    several = board.several_cameras
    cams = board.minute_cameras if several else []
    head = ["名前", "チーム" if board.sports else "グループ", "状態", "映っていた時間（分）",
            *[f"映っていた時間・{labels.camera_word(c)}（分）" for c in cams],
            "よかった場面（回）", *[f"{c.label}（回）" for c in cats], "種類がわからない場面（回）"]
    if board.court:
        head += ["動いた距離（推定・m）", "位置の誤差（±cm）", "距離を出したカメラ"]
    head.append("メモ")
    rows = [head]
    for p in board.people:
        notes = []
        if p.on_screen_s is None:
            notes.append("映っていた時間の記録がありません")
        elif len(p.on_screen_by_camera) > 1:
            notes.append(SEVERAL_CAMERAS)
        if board.court and p.distance_m is None:
            notes.append("距離は出せません（コートの上の記録が30秒より短いか、ありません）")
        row = [p.name, p.team if board.sports else board.group_label, p.status,
               minutes(p.on_screen_s), *[minutes(p.on_screen_by_camera.get(c)) for c in cams],
               p.total, *[p.counts.get(c.key, 0) for c in cats], p.untyped]
        if board.court:
            row += [distance_shown(p.distance_m),
                    None if p.position_error_cm is None else round(p.position_error_cm),
                    labels.camera_word(p.court_camera) if p.distance_m is not None else None]
        row.append("・".join(notes))
        rows.append(row)
    return _csv(rows)


def moments_csv(board: Board) -> bytes:
    rows = [["時間", "名前", "チーム" if board.sports else "グループ", "場面の種類", "記録",
             "文章で取り上げた", "カメラ", "状態"]]
    for m in board.moments:
        rows.append([m.when, m.name, m.team if board.sports else board.group_label,
                     m.category.label if m.category else "種類がわからない", m.what,
                     "はい" if m.cited else "", labels.camera_word(m.camera),
                     m.state or (APPROVED if m.approved else NOT_YET)])
    return _csv(rows)


def teams_csv(board: Board) -> bytes:
    """The team totals (sports with known teams), else the whole group's, one row each."""
    cats = board.categories
    rows = [["チーム" if board.sports else "グループ", "人数", "よかった場面（回）",
             *[f"{c.label}（回）" for c in cats], "種類がわからない場面（回）"]]
    for t in team_totals(board):
        rows.append([t.name, t.people, t.total, *[t.counts.get(c.key, 0) for c in cats],
                     t.untyped])
    return _csv(rows)


# --- heatmap cells as a table (the charts' table twins) ----------------------------------------

def court_cells(points: list[tuple[float, float]], limit: int = 10) -> list[dict]:
    """The most recorded 1 m cells of a court heatmap (export.charts.court's cells), as rows."""
    from pfb_app.export.charts import CELL_CM, CH, CW   # noqa: PLC0415
    counts: dict[tuple[int, int], int] = {}
    for x, y in points:
        if math.isfinite(x) and math.isfinite(y) and 0 <= x <= CW and 0 <= y <= CH:
            k = (min(int(x // CELL_CM), math.ceil(CW / CELL_CM) - 1),
                 min(int(y // CELL_CM), math.ceil(CH / CELL_CM) - 1))
            counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    out = []
    for (i, j), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        out.append({"左のサイドラインから（m）": f"{i}〜{i + 1}", "ゴールの下の線から（m）": f"{j}〜{j + 1}",
                    "記録の割合（%）": round(100 * n / total)})
    return out


def frame_cells(points: list[tuple[float, float]], size: tuple[int, int],
                limit: int = 10) -> list[dict]:
    """The most recorded cells of an image-plane heatmap (dashboard_charts.frame_heatmap)."""
    from pfb_app.export.dashboard_charts import COLS   # noqa: PLC0415
    fw, fh = size
    if not fw or not fh:
        return []
    cell = fw / COLS
    rows_n = max(1, math.ceil(fh / cell))
    counts: dict[tuple[int, int], int] = {}
    for x, y in points:
        if 0 <= x <= fw and 0 <= y <= fh:
            k = (min(int(x // cell), COLS - 1), min(int(y // cell), rows_n - 1))
            counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    return [{"画面の左から（列）": f"{i + 1}/{COLS}", "画面の上から（行）": f"{j + 1}/{rows_n}",
             "記録の割合（%）": round(100 * n / total)}
            for (i, j), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]
