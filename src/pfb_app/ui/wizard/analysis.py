"""How far a group's (and a work's) analysis has got: a pure derivation from the rows a
`progress._Snapshot` already holds (runs, jobs with their own progress, subjects).

- Stages follow the group's method (`PIPELINES`): the tracking engine's 内容の確認, 準備,
  人を見つける, 評価する人を選ぶ, したことの書き出し, よかった点を選ぶ, 文章づくり; the paper engine and documents
  in the same style. 確認と承認 is a line of its own (`Analysis.review`), outside the percentage.
- A stage has targets: a camera each for the per-camera stages of a video, an evaluated
  person each for pfb (every subject of a document), else one. A target is done when its
  latest run is usable and nothing upstream of it was superseded (as step 4 counts it), and
  no job of its key is active (one that is being redone counts by its job's progress).
- People (評価する人を選ぶ) is no stage of the pipeline: it is derived per camera from the mapping.
  Detect (内容の確認) runs while a content check job is active and is done otherwise: a check
  never stops a work.
- Fraction = Σ weight × stage completion / Σ weight; a stage's completion = (done targets +
  the own progress of running targets) / targets. Weights are the median durations of earlier
  runs of the method when every stage has a history, else `FALLBACK_WEIGHTS` (never mixed).
  The fraction stays at most 0.99 until every stage is done.
- `hold` keeps a shown bar from going back, except once at the start of a redo.

Every screen word here is plain Japanese; no internal term reaches the screen.
"""
import math
import statistics
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from pfb_app.domain.models import JobStatus, Kind, Stage
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.tracked_identity import mapping_is_current
from pfb_app.services import review
from pfb_app.storage.db import Repo

DONE, RUNNING, QUEUED, WAITING_GPU, STOPPED = "done", "running", "queued", "waiting_gpu", "stopped"

STAGE_NAMES: dict[str, str] = {
    "detect": "内容の確認", "prepare": "準備", "transcribe": "文字起こし", "track": "人を見つける",
    "people": "評価する人を選ぶ", "identity": "参加者の特定", "behavior": "したことの書き出し",
    "evidence": "場面の書き出し", "contribution": "よかった点を選ぶ", "pfb": "文章づくり",
}
REVIEW_NAME = "確認と承認"
LIVE_REVIEW_NAME = "文を読む・直す"   # 「あとから訂正」 works: no approvals
FALLBACK_WEIGHTS: dict[str, float] = {
    "detect": 1, "prepare": 1, "transcribe": 1, "track": 4, "people": 0.5, "identity": 1,
    "behavior": 10, "evidence": 3, "contribution": 3, "pfb": 3,
}
PIPELINES: dict[str, tuple[str, ...]] = {
    "pfb3x3": ("detect", "prepare", "track", "people", "behavior", "contribution", "pfb"),
    "paper": ("detect", "prepare", "transcribe", "identity", "behavior", "contribution", "pfb"),
    "document": ("prepare", "evidence", "contribution", "pfb"),
}
MOVING = "進んでいます"
REDOING = "やり直しています"
CAP = 0.99   # never 100 % before every stage is done

# every stage in one order, for the union of a work's groups
_ORDER = ("detect", "prepare", "transcribe", "track", "people", "identity", "behavior",
          "evidence", "contribution", "pfb")
# stages without a history of their own: weighed from another stage by their fallback ratio
# (people has no job; transcribe often never ran)
_DERIVED = {"people": "track", "transcribe": "prepare"}
_PER_CAMERA = {"prepare", "track", "identity", "behavior", "contribution"}   # videos only
_FRONT = {"prepare", "transcribe", "track", "identity"}   # jobs before the people are chosen
_ACTIVE = {JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value}
_FAILED = {JobStatus.failed.value, JobStatus.failed_retryable.value}
_PRECEDENCE = (STOPPED, RUNNING, WAITING_GPU, QUEUED, DONE)   # a work's stage over its groups


def _percent(fraction: float) -> int:
    return math.floor(round(fraction * 100, 6))


@dataclass
class StageLine:
    key: str
    name: str
    status: str      # DONE / RUNNING / QUEUED / WAITING_GPU / STOPPED
    fraction: float
    known: bool = True   # False: running, but how far is not known (the bar holds)
    note: str = ""

    @property
    def percent(self) -> int:
        return _percent(self.fraction)


@dataclass
class Analysis:
    fraction: float
    stages: list[StageLine]
    review: StageLine
    redo: bool = False
    cameras: dict[str, float] = field(default_factory=dict)   # camera -> its own fraction
    waiting_for_you: bool = False   # the coach's turn (「あなたの番」, no percentage shown)

    @property
    def percent(self) -> int:
        return _percent(self.fraction)

    @property
    def finished(self) -> bool:
        return bool(self.stages) and all(s.status == DONE for s in self.stages)

    @property
    def current(self) -> StageLine | None:
        """The first stage that is running, waiting for the GPU or stopped, else the first
        one not done; None when finished."""
        for s in self.stages:
            if s.status in (RUNNING, WAITING_GPU, STOPPED):
                return s
        return next((s for s in self.stages if s.status != DONE), None)

    @property
    def status(self) -> str:
        if self.finished:
            return DONE
        statuses = {s.status for s in self.stages}
        if STOPPED in statuses:
            return STOPPED
        if RUNNING not in statuses and WAITING_GPU in statuses:
            return WAITING_GPU
        return RUNNING

    @property
    def active(self) -> bool:
        """Moving on its own: not finished, not stopped and not waiting for the coach."""
        return not self.finished and self.status != STOPPED and not self.waiting_for_you


# --- weights ---

def stage_weights(durations: dict[str, list[float]], stages: Sequence[str],
                  counts: Mapping[str, int] | None = None) -> dict[str, float]:
    """Each stage's weight: the median of its earlier durations (one run each) times the
    stage's number of targets in `counts` (a run per camera or per person; 1 when not
    given), when every stage (but people and transcribe) has a history; people (and
    transcribe without a history) follow their anchor stage by the fallback ratio. Else
    FALLBACK_WEIGHTS for all."""
    need = [s for s in stages if s not in _DERIVED]
    anchors = {a for s, a in _DERIVED.items() if s in stages}
    if not need or not all(durations.get(s) for s in need) or not anchors <= set(need):
        return {s: float(FALLBACK_WEIGHTS[s]) for s in stages}
    counts = counts or {}

    def scaled(s: str) -> float:
        return float(statistics.median(durations[s])) * max(1, counts.get(s, 1))

    out: dict[str, float] = {}
    for s in stages:
        if durations.get(s) and s != "people":
            out[s] = scaled(s)
        else:
            anchor = _DERIVED[s]
            out[s] = scaled(anchor) * FALLBACK_WEIGHTS[s] / FALLBACK_WEIGHTS[anchor]
    return out


# pfb jobs of the tracking engine run this many at a time (worker.session.PARALLEL_JOBS)
PARALLEL_PFB = 8


def weight_counts(counts: Mapping[str, int], engine: str) -> dict[str, int]:
    """The counts the stage weights scale by: the tracking engine's pfb jobs run
    PARALLEL_PFB at a time, so n people take about ceil(n / PARALLEL_PFB) runs' time."""
    out = dict(counts)
    if engine == "pfb3x3" and out.get(Stage.pfb.value):
        out[Stage.pfb.value] = math.ceil(out[Stage.pfb.value] / PARALLEL_PFB)
    return out


def target_counts(repo: Repo, sub: dict, kind: Kind) -> dict[str, int]:
    """How many runs each stage of a group takes: one per camera for the per-camera stages
    of a video, one per evaluated person (every subject of a document) for pfb, else one."""
    kind = Kind(kind)
    cameras = len((sub.get("meta") or {}).get("cameras") or {}) if kind == Kind.video else 0
    counts = {s: cameras for s in _PER_CAMERA} if kind == Kind.video else {}
    counts[Stage.pfb.value] = len(_keys(repo, sub, kind, Stage.pfb.value, []))
    return counts


# --- one group ---

@dataclass
class _Target:
    camera: str | None
    done: bool
    job: dict | None = None
    redo: bool = False

    @property
    def running(self) -> bool:
        return self.job is not None and self.job["status"] == JobStatus.running.value

    @property
    def active(self) -> bool:
        return self.job is not None and self.job["status"] in _ACTIVE

    @property
    def completion(self) -> float:
        """1 when done; an active job's stored progress (a requeued or GPU-waiting job keeps
        how far it got); else 0."""
        if self.done:
            return 1.0
        if self.active and self.job.get("progress") is not None:
            return min(max(float(self.job["progress"]), 0.0), 1.0)
        return 0.0


def _pipeline(repo: Repo, sub: dict, engine: str, latest: dict[tuple, dict]) -> list[str]:
    stages = list(PIPELINES[engine])
    if engine == "paper":
        meta = sub.get("meta") or {}
        transcribing = (bool(meta.get("auto_transcribe"))
                        or any(k[0] == Stage.transcribe.value for k in latest)
                        or any(repo.latest_run(sub["id"], Stage.transcribe, cam) is not None
                               for cam in [None, *(meta.get("cameras") or {})]))
        if not transcribing:
            stages.remove("transcribe")
    return stages


def _keys(repo: Repo, sub: dict, kind: Kind, key: str,
          cameras: list[str]) -> list[tuple[str | None, int | None]]:
    """(camera, subject id) of each target of a stage."""
    if key == Stage.pfb.value:
        return [(None, s["id"]) for s in repo.list_subjects(sub["id"])
                if kind == Kind.document or is_evaluated(s)]
    if kind == Kind.video and key in _PER_CAMERA:
        return [(cam, None) for cam in cameras]
    return [(None, None)]


def _targets(repo: Repo, sub: dict, kind: Kind, key: str, cameras: list[str],
             latest: dict[tuple, dict]) -> list[_Target]:
    sid, stage = sub["id"], Stage(key)
    out = []
    for cam, subject in _keys(repo, sub, kind, key, cameras):
        job = latest.get((key, cam, subject))
        if key == Stage.prepare.value and cam is not None:
            # an older prepare job of the whole group (no camera) prepares every camera
            whole = latest.get((key, None, None))
            if whole is not None and (job is None or whole["id"] > job["id"]):
                job = whole
        active = job is not None and job["status"] in _ACTIVE
        run = repo.latest_run(sid, stage, cam, subject)
        done = not active and usable(run) and not review.is_stale_chain(repo, run)
        redo = active and any(r.status == "ok"
                              for r in repo.run_history(sid, stage, cam, subject))
        out.append(_Target(cam, done, job, redo))
    return out


def _status(targets: list[_Target]) -> str:
    statuses = {t.job["status"] for t in targets if t.job is not None}
    if JobStatus.running.value in statuses:
        return RUNNING
    if JobStatus.waiting_gpu.value in statuses:
        return WAITING_GPU
    if any(t.job is not None and t.job["status"] in _FAILED and not t.done for t in targets):
        return STOPPED
    if targets and all(t.done for t in targets):
        return DONE
    return QUEUED


def _running_note(key: str, targets: list[_Target]) -> tuple[str, bool]:
    """(the note of a running stage, whether its progress is known)."""
    running = [t for t in targets if t.running]
    note = next((t.job.get("progress_note") for t in running if t.job.get("progress_note")),
                None)
    if note:
        return note, True
    done = sum(t.done for t in targets)
    if key == Stage.pfb.value:
        return f"{done} / {len(targets)} 人", True
    if len(targets) > 1 and any(t.camera for t in targets):
        return f"{done} / {len(targets)} カメラ", True
    return MOVING, any(t.job.get("progress") is not None for t in running)


def _line(key: str, targets: list[_Target], status: str, reason: str) -> StageLine:
    fraction = sum(t.completion for t in targets) / len(targets) if targets else 0.0
    note, known = "", True
    if status == RUNNING:
        note, known = _running_note(key, targets)
    elif status in (WAITING_GPU, STOPPED):
        note = reason
    if any(t.redo for t in targets) and status in (RUNNING, QUEUED):
        note = REDOING + (f"（{note}）" if note and note != MOVING else "")
    return StageLine(key, STAGE_NAMES[key], status, fraction, known, note)


def _detect(latest: dict[tuple, dict]) -> list[_Target]:
    """One target: its active content check job (running, waiting for the GPU or queued),
    else done (a check never stops a work)."""
    active = [r for k, r in latest.items()
              if k[0] == Stage.detect.value and r["status"] in _ACTIVE]
    if not active:
        return [_Target(None, True)]
    order = (JobStatus.running.value, JobStatus.waiting_gpu.value, JobStatus.queued.value)
    return [_Target(None, False, min(active, key=lambda r: order.index(r["status"])))]


def _people(repo: Repo, sub: dict, cameras: list[str], track: list[_Target],
            front_active: bool, status: str) -> tuple[list[_Target], str]:
    """Per camera: done when confirmed or its mapping is current; running while its track is
    done but the mapping is not, nothing earlier runs and it is not the coach's turn; else
    queued. No job, no history."""
    tracked = {t.camera: t.done for t in track}
    out, running = [], False
    for cam in cameras:
        if sub["identity_confirmed"] or mapping_is_current(repo, sub["id"], cam):
            out.append(_Target(cam, True))
        else:
            out.append(_Target(cam, False))
            running |= tracked.get(cam, False) and not front_active and status != "your_turn"
    if out and all(t.done for t in out):
        return out, DONE
    return out, RUNNING if running else QUEUED


def _review(approved: int, total: int, at_review: bool, live: bool = False) -> StageLine:
    """The review line: 「確認と承認」 (承認 n / 全 m); for a 「あとから訂正」 work 「文を読む・直す」
    (文ができた人 n / 全 m), `approved` being the people with a text."""
    if total > 0 and approved == total:
        status = DONE
    elif approved > 0 or at_review:
        status = RUNNING
    else:
        status = QUEUED
    if live:
        return StageLine("review", LIVE_REVIEW_NAME, status, approved / total if total else 0.0,
                         note=f"文ができた人 {approved} / 全 {total}")
    return StageLine("review", REVIEW_NAME, status, approved / total if total else 0.0,
                     note=f"承認 {approved} / 全 {total}")


def submission_analysis(repo: Repo, sub: dict, kind: Kind, engine: str,
                        weights: dict[str, float], *, status: str, reason: str, approved: int,
                        total: int, step: str | None = None, live: bool = False) -> Analysis:
    """One group's analysis. `status` and `reason` are the group's own (SubProgress:
    "your_turn" sets waiting_for_you, "done" makes every stage done, "stopped" stops the
    first stage not done unless one has stopped already);
    `approved` / `total` its approvals and evaluated people; `step` its step (the review
    line runs from "review" or "pdf"; without it, once every stage is done)."""
    kind = Kind(kind)
    latest: dict[tuple, dict] = {}
    for row in repo.submission_job_rows(sub["id"]):
        latest[(row["stage"], row["camera"], row["subject_id"])] = row
    meta = sub.get("meta") or {}
    cameras = list(meta.get("cameras") or {}) if kind == Kind.video else []
    front_active = any(k[0] in _FRONT and r["status"] in _ACTIVE for k, r in latest.items())
    lines: list[StageLine] = []
    per_camera = {cam: [0.0, 0.0] for cam in cameras}   # camera -> [Σ w × completion, Σ w]
    tracks: list[_Target] = []
    redo = False
    for key in _pipeline(repo, sub, engine, latest):
        if key == "detect":
            targets = _detect(latest)
            line = _line(key, targets, _status(targets), reason)
        elif key == "people":
            targets, people = _people(repo, sub, cameras, tracks, front_active, status)
            line = _line(key, targets, people, reason)
        else:
            targets = _targets(repo, sub, kind, key, cameras, latest)
            line = _line(key, targets, _status(targets), reason)
            redo |= any(t.redo for t in targets)
            if key == "track":
                tracks = targets
        lines.append(line)
        for t in targets:
            if t.camera in per_camera:
                per_camera[t.camera][0] += weights[key] * t.completion
                per_camera[t.camera][1] += weights[key]
    if status == "done":      # complete (its PDF is out): nothing is left to analyse
        for line in lines:
            line.status, line.fraction, line.known, line.note = DONE, 1.0, True, ""
        per_camera = {cam: [1.0, 1.0] for cam in per_camera}
    elif status == "stopped" and not any(s.status == STOPPED for s in lines):
        # e.g. nobody to evaluate: where it stands, it has stopped
        first = next((s for s in lines if s.status != DONE), None)
        if first is not None:
            first.status, first.note, first.known = STOPPED, reason, True
    total_w = sum(weights[s.key] for s in lines)
    fraction = sum(weights[s.key] * s.fraction for s in lines) / total_w if total_w else 0.0
    analysis = Analysis(fraction, lines, StageLine("review", REVIEW_NAME, QUEUED, 0.0), redo,
                        {cam: (w_done / w if w else 0.0) for cam, (w_done, w) in
                         per_camera.items()}, waiting_for_you=status == "your_turn")
    analysis.fraction = 1.0 if analysis.finished else min(fraction, CAP)
    at_review = step in ("review", "pdf") if step is not None else analysis.finished
    analysis.review = _review(approved, total, at_review, live)
    return analysis


# --- the work ---

def work_analysis(parts: list[tuple[str, Analysis]], approved: int,
                  total: int, live: bool = False) -> Analysis | None:
    """The work's analysis from its groups' (label, analysis): the mean fraction, and per
    stage the most pressing status (stopped > running > waiting_gpu > queued > done) with
    the note of the first group in it (named when there are several groups)."""
    if not parts:
        return None
    several = len(parts) > 1
    stages = []
    for key in _ORDER:
        lines = [(label, s) for label, a in parts for s in a.stages if s.key == key]
        if not lines:
            continue
        found = {s.status for _, s in lines}
        status = next(st for st in _PRECEDENCE if st in found)
        label, first = next((lb, s) for lb, s in lines if s.status == status)
        note = f"{label}: {first.note}" if several and first.note else first.note
        stages.append(StageLine(key, STAGE_NAMES[key], status,
                                sum(s.fraction for _, s in lines) / len(lines), first.known,
                                note))
    at_review = any(a.review.status in (RUNNING, DONE) for _, a in parts)
    return Analysis(sum(a.fraction for _, a in parts) / len(parts), stages,
                    _review(approved, total, at_review, live), any(a.redo for _, a in parts),
                    dict(parts[0][1].cameras) if not several else {},
                    waiting_for_you=any(a.waiting_for_you for _, a in parts))


def hold(previous: tuple[float, bool] | None, analysis: Analysis) -> float:
    """The fraction to show, given the one shown before and whether it was a redo: never
    lower than before, except once when a redo starts; 1.0 once finished."""
    if analysis.finished:
        return 1.0
    if previous is None or (analysis.redo and not previous[1]):
        return analysis.fraction
    return max(previous[0], analysis.fraction)
