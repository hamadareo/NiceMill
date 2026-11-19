"""待ち時間の目安: how long a work's analysis may still take, from this installation's own past.

Everything here reads records and computes; nothing is written. It answers three things for
one work (`estimate`):

- **the time left** (`Estimate.seconds`): the jobs still to come (the active ones and the
  follow-ups they will start, per group and camera: `remaining_slots`) each priced from the
  finished runs of the same method and stage. The stages that follow the length of the video
  (preparing, tracking, behavior, contribution) are priced by a robust straight line through
  (length of the video, time taken) of the earlier runs (Theil-Sen: the median of the slopes
  between pairs of runs, so one slow first run or a run that was disturbed does not bend it).
  That needs `MIN_POINTS` earlier runs with a known length that do not all have the same
  length; with fewer, or without a known length of the group, the stage is priced by the
  median of its earlier runs, exactly as before this module. The stages that do not follow
  the length (the feedback of a person) always use the median. A stage that never finished
  before has no price, and then there is no estimate at all (`seconds` is None): the screen
  says so instead of guessing.
- **the queue** (`Estimate.ahead`): while none of the work's steps runs yet, how many other
  works have steps queued before its first one (or running now), and what those works still
  need (`wait_seconds`, None when one of them has no estimate). The time with waiting is the
  sum (`Estimate.total`); other works may in fact overlap, so it is an upper-ish guess and the
  words say 「かもしれません」.
- **a stalled start** (`Estimate.held`): a step waits for free room on the machine (the work's
  steps are 「ほかの処理が終わるまで待つ」), which no sum can predict.

The length of a group's video (`group_lengths`) is read from what the analysis already
recorded: the tracking run's analysed seconds, else the content check's duration (cut to the
group's own start and end), else the last prepared frame's time. It is read with
`json_extract`, never by decoding a run's whole output.

The history of a method (`history`) is kept per database and method while the number of
stage runs has not changed (a cache that is never older than the records; no schema change,
no index added to the live database).

How close it is (measured on a copy of this installation's records, 2026-11-14, works of one
group each: `scripts/eta_backtest.py` and the replay in tests/test_eta.py): with the length of
the video the sum of a work's steps was within about a quarter of what its steps really took
(the median of earlier runs alone: 40 to 70 % too low for the same works, and 3 to 4 times too
high for a 12 second clip). The real waiting time is a little longer than the sum of the steps
(about 15 %, the gaps between steps) and a work with several groups overlaps its groups, so the
words stay 「目安」 and 「かもしれません」, and the number is rounded up.

`lines` turns an estimate into the plain Japanese lines a card or a panel shows; rounding
is `rounded` (up to the minute, to 5 minutes from 10, to 10 from an hour).
"""
import math
import statistics
import threading
from collections import Counter
from dataclasses import dataclass, field

from pfb_app.domain.models import JobStatus, Kind, Stage
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import submission_engine
from pfb_app.services import deletion
from pfb_app.storage.db import NON_BLOCKING_STAGES, Repo

MIN_POINTS = 3         # earlier runs with a known length needed for the straight line
MIN_SPREAD = 1.25      # their longest length over their shortest, at least
HISTORY_LIMIT = 40     # newest runs of a stage (of a kind: served or not) that are read
# stages whose time follows the length of the video (per camera of a video)
LENGTH_STAGES = frozenset({"prepare", "track", "behavior", "contribution", "identity",
                           "evidence", "transcribe"})
SERVED_TIMED = ("contribution", "pfb")   # as storage.db: the served model's runs decide
ACTIVE = (JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value)
# what follows what, per method (per camera for videos); the pfb stage (one per person)
# comes after the last of them
CHAINS = {"pfb3x3": ["prepare", "track", "behavior", "contribution"],
          "paper": ["prepare", "identity", "behavior", "contribution"],
          "document": ["prepare", "evidence", "contribution"]}
FRONT_STAGES = {Stage.prepare.value, Stage.transcribe.value, Stage.track.value,
                Stage.identity.value}

EMPTY = "目安はまだ出せません（解析の実績がたまると出ます）"
HELD = "ほかの処理が終わるまで始まらないため、もっとかかるかもしれません。"


# --- the jobs still to come -------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    """One job that is active now (`job` is its row) or will be started by one that is."""
    submission_id: int
    engine: str                 # "pfb3x3", "paper" or "document"
    stage: str
    camera: str | None
    subject_id: int | None
    job: dict | None = None


def _latest_active(repo: Repo, submission_id: int) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for row in repo.submission_job_rows(submission_id):
        latest[(row["stage"], row["camera"], row["subject_id"])] = row
    return [r for r in latest.values() if r["status"] in ACTIVE]


def _gate(sub: dict, assignment: dict, engine: str) -> str | None:
    """The stage the chain waits before for the teacher (people mapping or confirmation)."""
    if engine == "pfb3x3":
        return "behavior"
    if engine == "paper" and assignment["require_identity_approval"] \
            and not sub["identity_confirmed"]:
        return "behavior"
    return None


def remaining_slots(repo: Repo, assignment_id: int, *, through: bool = False) -> list[Slot]:
    """Every active job of the work, plus the follow-ups it will start up to the next point
    where the teacher has to act. `through`: the tracking method goes on past the people
    mapping (the people are chosen automatically). The content check is never part of it (it
    has no chain, and a later group's check would otherwise hide the estimate)."""
    assignment = repo.get_assignment(assignment_id)
    kind = Kind(assignment["kind"])
    slots: list[Slot] = []
    for sub in repo.list_submissions(assignment_id):
        sid = sub["id"]
        active = [r for r in _latest_active(repo, sid) if r["stage"] != Stage.detect.value]
        if not active:
            continue
        engine = "document" if kind == Kind.document else submission_engine(sub)
        chain = CHAINS[engine]
        gate = None if through and engine == "pfb3x3" else _gate(sub, assignment, engine)
        cameras = list(sub["meta"].get("cameras") or {}) if kind == Kind.video else [None]
        planned = {(r["stage"], r["camera"], r["subject_id"]) for r in active}
        slots += [Slot(sid, engine, r["stage"], r["camera"], r["subject_id"], r) for r in active]
        to_pfb = False
        for row in active:
            stage = row["stage"]
            if stage == Stage.transcribe.value:      # the paper method identifies next
                stage = "prepare" if engine == "paper" else None
            if stage not in chain:
                continue
            where = [row["camera"]] if row["camera"] or kind == Kind.document else cameras
            for nxt in chain[chain.index(stage) + 1:]:
                if nxt == gate:
                    break
                for cam in where:
                    if (nxt, cam, None) not in planned:
                        planned.add((nxt, cam, None))
                        slots.append(Slot(sid, engine, nxt, cam, None))
            else:
                to_pfb = True
        if to_pfb:
            for subj in repo.list_subjects(sid):
                key = (Stage.pfb.value, None, subj["id"])
                if (kind == Kind.document or is_evaluated(subj)) and key not in planned:
                    planned.add(key)
                    slots.append(Slot(sid, engine, Stage.pfb.value, None, subj["id"]))
    return slots


def remaining_jobs(repo: Repo, assignment_id: int, *, front_only: bool = False,
                   through: bool = False) -> Counter:
    """Jobs still to come per (method, stage) (`remaining_slots` counted). `front_only`: only
    the stages up to the people mapping."""
    counts = Counter((s.engine, s.stage) for s in remaining_slots(repo, assignment_id,
                                                                  through=through))
    if front_only:
        counts = Counter({k: n for k, n in counts.items() if k[1] in FRONT_STAGES})
    return +counts


# --- how long a video is ----------------------------------------------------------------------

def window(duration: float | None, start: float | None, end: float | None) -> float | None:
    """The seconds of a video of `duration` that a group covers from `start` to `end` (None:
    to the end); None when the video's length is not known or nothing is left."""
    if not duration or duration <= 0:
        return None
    stop = min(float(end), float(duration)) if end is not None else float(duration)
    seconds = stop - float(start or 0.0)
    return seconds if seconds > 0 else None


def group_lengths(repo: Repo, submission_ids: list[int]) -> dict[tuple[int, str | None], float]:
    """{(group id, camera or None): seconds of video the analysis covers}, where known. A
    camera's own length (the tracking run's analysed seconds, else the last prepared frame);
    the None key is the group's (the content check's duration cut to the group's window)."""
    out: dict[tuple[int, str | None], float] = {}
    if not submission_ids:
        return out
    marks = ",".join("?" * len(submission_ids))
    rows = repo.conn.execute(
        "SELECT r.submission_id, r.stage, r.camera, json_extract(r.output, '$.analysed_s'),"
        " json_extract(r.output, '$.duration_s'), json_extract(r.output, '$.frames[#-1].video_sec'),"
        " json_extract(s.meta, '$.start_sec'), json_extract(s.meta, '$.end_sec')"
        " FROM stage_runs r JOIN submissions s ON r.submission_id = s.id"
        f" WHERE r.submission_id IN ({marks}) AND r.source = 'model' AND r.status = 'ok'"
        " AND r.stage IN ('track', 'detect', 'prepare') ORDER BY r.id", submission_ids).fetchall()
    tracked: dict[tuple[int, str | None], float] = {}
    prepared: dict[tuple[int, str | None], float] = {}
    for sid, stage, camera, analysed, duration, last, start, end in rows:   # newest wins
        if stage == "track" and analysed and analysed > 0:
            tracked[(sid, camera)] = float(analysed)
        elif stage == "prepare" and last is not None and last >= 0:
            prepared[(sid, camera)] = float(last)
        elif stage == "detect" and duration and duration > 0:
            seconds = window(float(duration), start, end)
            if seconds:
                out[(sid, None)] = seconds
    for key, value in prepared.items():
        out[key] = value
    out.update(tracked)
    return out


def length_of(lengths: dict[tuple[int, str | None], float], submission_id: int,
              camera: str | None) -> float | None:
    """The camera's length, else the group's, else the longest camera of the group; None."""
    for key in ((submission_id, camera), (submission_id, None)):
        if lengths.get(key):
            return lengths[key]
    known = [v for (sid, _), v in lengths.items() if sid == submission_id and v]
    return max(known) if known else None


# --- what earlier runs took -------------------------------------------------------------------

Points = list[tuple["float | None", float]]    # (length of the video or None, seconds taken)
_HISTORY: dict[tuple, dict[str, Points]] = {}
_HISTORY_LOCK = threading.Lock()
_HISTORY_KEEP = 16


def clear_history_cache() -> None:
    with _HISTORY_LOCK:
        _HISTORY.clear()


def _marker(repo: Repo) -> tuple:
    return tuple(repo.conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM stage_runs")
                 .fetchone())


def _db_path(repo: Repo) -> str:
    return next((r[2] for r in repo.conn.execute("PRAGMA database_list") if r[1] == "main"), "")


def history(repo: Repo, engine: str, limit: int = HISTORY_LIMIT) -> dict[str, Points]:
    """{stage: [(length, seconds)], newest first} of one method's finished model runs (a run
    a teacher saved, one that failed and one shorter than a second are left out, as in
    Repo.stage_durations). As there, the tracking method's contribution and feedback runs
    on the served model decide once any exist. Kept while the stage runs are unchanged."""
    path, marker = _db_path(repo), _marker(repo)
    key = (path, engine, limit)
    if path:
        with _HISTORY_LOCK:
            hit = _HISTORY.get(key)
        if hit is not None and hit[0] == marker:
            return {k: list(v) for k, v in hit[1].items()}
    data = _read_history(repo, engine, limit)
    if path:
        with _HISTORY_LOCK:
            if len(_HISTORY) >= _HISTORY_KEEP:
                _HISTORY.clear()
            _HISTORY[key] = (marker, {k: list(v) for k, v in data.items()})
    return data


def _records(repo: Repo, engine: str, limit: int) -> dict[str, list[tuple[int, int, float | None, float]]]:
    """{stage: [(group id, run id, length, seconds)], newest first} of one method's finished
    model runs (see `history`)."""
    if engine == "document":
        method, args = "a.kind = 'document'", ()
    else:
        method = ("a.kind = 'video' AND (COALESCE(json_extract(s.meta, '$.engine'),"
                  " 'paper') = 'pfb3x3') = ?")
        args = (engine == "pfb3x3",)
    rows = repo.conn.execute(
        "SELECT id, submission_id, stage, camera, served, duration_sec FROM (SELECT r.id,"
        " r.submission_id, r.stage, r.camera, r.duration_sec,"
        " COALESCE(r.model_id, '') LIKE 'served:%' AS served, ROW_NUMBER() OVER"
        " (PARTITION BY r.stage, COALESCE(r.model_id, '') LIKE 'served:%'"
        " ORDER BY r.id DESC) AS n FROM stage_runs r"
        " JOIN submissions s ON r.submission_id = s.id"
        " JOIN assignments a ON s.assignment_id = a.id WHERE r.source = 'model'"
        f" AND r.status = 'ok' AND r.duration_sec >= 1 AND {method}) WHERE n <= ?"
        " ORDER BY stage, served, n", (*args, limit)).fetchall()
    lengths = group_lengths(repo, sorted({r[1] for r in rows}))
    split: dict[str, dict[bool, list[tuple[int, int, float | None, float]]]] = {}
    for run_id, sid, stage, camera, served, seconds in rows:
        x = length_of(lengths, sid, camera) if stage in LENGTH_STAGES else None
        split.setdefault(stage, {}).setdefault(bool(served), []).append((run_id, sid, x, seconds))
    out: dict[str, list[tuple[int, int, float | None, float]]] = {}
    for stage, parts in split.items():
        if engine == "pfb3x3" and stage in SERVED_TIMED and parts.get(True):
            runs = parts[True]
        else:
            runs = sorted([*parts.get(False, []), *parts.get(True, [])], reverse=True)
        out[stage] = [(sid, run_id, x, sec) for run_id, sid, x, sec in runs[:limit]]
    return out


def _read_history(repo: Repo, engine: str, limit: int) -> dict[str, Points]:
    return {stage: [(x, sec) for _sid, _id, x, sec in runs]
            for stage, runs in _records(repo, engine, limit).items()}


# --- pricing one job --------------------------------------------------------------------------

def theil_sen(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """(intercept, slope) of the robust line through (x, y): the median of the slopes between
    pairs of points with different x, and the median intercept. The slope is never negative
    and the intercept never below zero (then the line goes through the origin: the median of
    y / x). None when no two points differ in x."""
    slopes = [(y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(points)
              for x2, y2 in points[i + 1:] if abs(x2 - x1) > 1e-9]
    if not slopes:
        return None
    slope = max(0.0, statistics.median(slopes))
    intercept = statistics.median(y - slope * x for x, y in points)
    if intercept < 0:
        return 0.0, statistics.median(y / x for x, y in points)
    return intercept, slope


def price(points: Points, length: float | None) -> float | None:
    """What one job of a stage may take, for a video of `length` seconds: the line through
    the earlier runs when there are enough of them with different lengths, else the median of
    the earlier runs (None without any)."""
    if not points:
        return None
    known = [(x, y) for x, y in points if x]
    if length and len(known) >= MIN_POINTS \
            and max(x for x, _ in known) >= MIN_SPREAD * min(x for x, _ in known):
        line = theil_sen(known)
        if line is not None:
            intercept, slope = line
            return max(1.0, intercept + slope * length)
    return float(statistics.median(y for _, y in points))


def price_of(hist: dict[str, Points], stage: str, length: float | None) -> float | None:
    """What one job of `stage` may take, from a method's `history` (None without any earlier
    run of it). A person's feedback job also makes that person's practice tips right after the
    text, and the tips are their own stage run (so not in the feedback run's time): the median
    time of the tips runs is added once per feedback, when there are any on record."""
    seconds = price(hist.get(stage) or [], length)
    if seconds is None:
        return None
    if stage == Stage.pfb.value and hist.get(Stage.tips.value):
        seconds += float(statistics.median(y for _, y in hist[Stage.tips.value]))
    return seconds


def _priced(repo: Repo, slots: list[Slot]) -> list[tuple[Slot, float]] | None:
    """Each slot with its price; None when a stage still to come has never finished before."""
    if not slots:
        return None
    lengths = group_lengths(repo, sorted({s.submission_id for s in slots}))
    hist: dict[str, dict[str, Points]] = {}
    out: list[tuple[Slot, float]] = []
    for slot in slots:
        x = length_of(lengths, slot.submission_id, slot.camera) \
            if slot.stage in LENGTH_STAGES else None
        seconds = price_of(hist.setdefault(slot.engine, history(repo, slot.engine)),
                           slot.stage, x)
        if seconds is None:
            return None
        out.append((slot, seconds))
    return out


def _seconds_left(priced: list[tuple[Slot, float]]) -> float:
    """The sum of the prices, less, for each running job with its own progress, that progress
    of its price; at least 0 (the content check's progress is not counted)."""
    total = sum(seconds for _, seconds in priced)
    for slot, seconds in priced:
        job = slot.job
        done = None if job is None else job.get("progress")
        if job is None or job["status"] != JobStatus.running.value or done is None \
                or slot.stage == Stage.detect.value:
            continue
        total -= min(max(float(done), 0.0), 1.0) * seconds
    return max(0.0, total)


def remaining_sec(repo: Repo, assignment_id: int) -> float | None:
    """The time left of the whole analysis (through the automatic people choice); None when
    there is nothing to wait for or a stage still to come has no history."""
    priced = _priced(repo, remaining_slots(repo, assignment_id, through=True))
    return None if priced is None else _seconds_left(priced)


# --- the queue --------------------------------------------------------------------------------

def queue_ahead(repo: Repo, assignment_id: int) -> tuple[list[int], bool]:
    """(the other works ahead of this one, a step of this one waits for room on the machine).
    Only while none of this work's steps runs: then the works with a step that runs now or was
    queued before this work's first waiting one are ahead (a step that never holds the coach
    up, like the names of a roster, is left out). Works waiting to be deleted are left out."""
    skip = ",".join("?" * len(NON_BLOCKING_STAGES))
    rows = repo.conn.execute(
        "SELECT j.id, j.status, s.assignment_id FROM jobs j JOIN submissions s"
        " ON j.submission_id = s.id WHERE j.status IN (?, ?, ?)"
        f" AND j.stage NOT IN ({skip}) ORDER BY j.id", (*ACTIVE, *NON_BLOCKING_STAGES)).fetchall()
    mine = [(i, st) for i, st, a in rows if a == assignment_id]
    if not mine or any(st == JobStatus.running.value for _, st in mine):
        return [], False
    held = any(st == JobStatus.waiting_gpu.value for _, st in mine)
    first = mine[0][0]
    gone = deletion.pending_deletions(repo)
    ahead = []
    for i, st, a in rows:
        if a != assignment_id and a not in gone and a not in ahead \
                and (i < first or st == JobStatus.running.value):
            ahead.append(a)
    return ahead, held


# --- the answer for one work ------------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    seconds: float | None            # this work's own time left; None: no estimate
    ahead: int = 0                   # other works in front of it
    wait_seconds: float | None = None   # what those works still need; None: not known
    held: bool = False               # a step waits for room on the machine
    idle: bool = False               # nothing of the work is waiting to be analysed
    ahead_ids: tuple[int, ...] = field(default=(), compare=False)

    @property
    def total(self) -> float | None:
        """The time left with the waiting in front; the work's own when nobody is ahead; None
        when either part is not known."""
        if self.seconds is None:
            return None
        if not self.ahead:
            return self.seconds
        return None if self.wait_seconds is None else self.seconds + self.wait_seconds


def estimate(repo: Repo, assignment_id: int) -> Estimate:
    """The work's time left and its place in the queue (`idle`: no step of it waits or runs,
    so there is nothing to estimate)."""
    slots = remaining_slots(repo, assignment_id, through=True)
    priced = _priced(repo, slots)
    own = None if priced is None else _seconds_left(priced)
    ahead_ids, held = queue_ahead(repo, assignment_id)
    wait: float | None = 0.0
    for other in ahead_ids:
        left = remaining_sec(repo, other)
        if left is None:
            wait = None
            break
        wait += left
    return Estimate(own, len(ahead_ids), wait if ahead_ids else None, held, not slots,
                    tuple(ahead_ids))


# --- how good the guesses are ---------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """One earlier group, guessed from all the others: what its runs really took, what the
    length-aware prices said and what the plain medians said (seconds)."""
    submission_id: int
    actual: float
    by_length: float
    by_median: float

    @property
    def error_length(self) -> float:
        return abs(self.by_length - self.actual) / self.actual

    @property
    def error_median(self) -> float:
        return abs(self.by_median - self.actual) / self.actual


def backtest(repo: Repo, engine: str) -> list[Check]:
    """For each group with a finished run in every stage its method has (each person's
    feedback counted), leave it out of the history and guess the sum of its runs, with and
    without the length of the video. Read-only; the way to see how far the estimates can be
    trusted on a copy of the real records (scripts/eta_backtest.py)."""
    records = _records(repo, engine, 10 ** 6)
    need = (set(CHAINS[engine]) - {"prepare"}) | {Stage.pfb.value}   # prepare can take < 1 s
    out: list[Check] = []
    for sid in sorted({sid for runs in records.values() for sid, *_ in runs}):
        if not all(any(r[0] == sid for r in records.get(stage, [])) for stage in need):
            continue
        actual = by_length = by_median = 0.0
        usable = True
        for runs in records.values():
            others = [(x, sec) for g, _id, x, sec in runs if g != sid]
            for g, _id, x, sec in runs:
                if g != sid:
                    continue
                a, b = price(others, x), price(others, None)
                if a is None or b is None:   # a stage no other group ever finished
                    usable = False
                    break
                actual, by_length, by_median = actual + sec, by_length + a, by_median + b
            if not usable:
                break
        if usable and actual > 0:
            out.append(Check(sid, actual, by_length, by_median))
    return out


# --- words ------------------------------------------------------------------------------------

def rounded(seconds: float) -> str:
    """「約15分」: rounded up, to 5 minutes from 10, to 10 from an hour."""
    minutes = max(1, math.ceil(seconds / 60))
    if minutes >= 60:
        minutes = math.ceil(minutes / 10) * 10
        hours, rest = divmod(minutes, 60)
        return f"約{hours}時間" + (f"{rest}分" if rest else "")
    if minutes >= 10:
        minutes = math.ceil(minutes / 5) * 5
    return f"約{minutes}分"


def time_left_text(seconds: float, *, waiting: bool = False) -> str:
    """「あと約15分（目安）」; with `waiting`: 「順番待ちをふくめて、あと約25分かかるかもしれません（目安）」."""
    if waiting:
        return f"順番待ちをふくめて、あと{rounded(seconds)}かかるかもしれません（目安）"
    return f"あと{rounded(seconds)}（目安）"


def queue_text(ahead: int) -> str:
    return f"前に{ahead}件の作業があります。順番が来ると始まります。"


def queue_lines(est: Estimate) -> list[str]:
    """Only what the queue adds to the work's own time left: who is in front (with the time
    left counted from the waiting, when it is known) and a start that has to wait for room
    on the machine. Empty when the work is not waiting for anything."""
    out: list[str] = []
    if est.ahead:
        out.append(queue_text(est.ahead))
        if est.total is not None:
            out.append(time_left_text(est.total, waiting=True))
    if est.held:
        out.append(HELD)
    return out


def lines(est: Estimate) -> list[str]:
    """The plain lines under a work's bar: where it stands in the queue, the time left (with
    the waiting when it is known), what makes it uncertain, or that no estimate can be given
    yet."""
    out: list[str] = []
    if est.ahead:
        out.append(queue_text(est.ahead))
    if est.idle:
        return out
    if est.seconds is None:
        out.append(EMPTY)
        return out
    total = est.total
    if est.ahead and total is not None:
        out.append(time_left_text(total, waiting=True))
    elif est.ahead:
        out.append(time_left_text(est.seconds) + "。順番待ちの時間は含みません")
    else:
        out.append(time_left_text(est.seconds))
    if est.held:
        out.append(HELD)
    return out
