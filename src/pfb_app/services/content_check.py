"""The content check of dropped videos: the card, the coach's confirm and later files.

A work started from videos (`start_work_from_videos`) registers its first group with
`hold_for_check`: meta["check"] is "pending" and a `detect` job runs instead of `prepare`.
`build_card` turns the detect run into a plain, advisory card (the sentence and the pre-fills)
and marks the group "ready" once the check has finished (or failed). Nothing is decided by the
card: the coach confirms with 「この内容で始める」 (`confirm_check`), which writes the work and
the group's settings and enqueues the cameras' `prepare` jobs exactly as a registration does
(meta["check"] = "confirmed"). Names never run from here.

One work = one match or activity session. The first group confirmed sets the work's domain,
profile (the kind of activity, domain/profiles.py: from the scene verdict and, with an engine
that gives it, the kind of a non-sport activity), sport, title and activity line; later
groups are checked too, keep the work as it is and get a note when their own check disagrees
(`later_note`).

Engine error texts and job errors are never shown: a failed check gives the plain card that
asks the coach to pick.
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain import profiles
from pfb_app.domain.models import Kind, Stage
from pfb_app.domain.vocab import SCENE_LABELS, SPORT_KEYS, SPORT_LABELS
from pfb_app.engine import TARGETS
from pfb_app.pipeline.detect import PROBE_FAILED
from pfb_app.pipeline.runner import enqueue_once, mark_check_ready
from pfb_app.preprocess.frames import VideoError, probe_video, validate_interval
from pfb_app.services import eta
from pfb_app.services.intake import ENGINES, register_video_group
from pfb_app.services.review import BusyError, roster_subjects
from pfb_app.services.roster import Roster, apply_roster, roster_text
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

__all__ = ["AnalysedPart", "CHECK_CONFIRMED", "CHECK_PENDING", "CHECK_READY", "LONG_VIDEO_SEC", "MAX_CAMERAS",
           "MAX_PEOPLE", "MOTION_NOTE", "SHORT_PART_NOTE", "CheckCard", "CheckChoice",
           "add_next_part", "add_videos_to_work", "analysed_part", "build_card", "next_part_group", "card_sentence", "check_facts", "confirm_check",
           "default_title", "estimate_for", "later_note", "earlier_roster_text", "numbered",
           "roster_roles", "sport_phrase", "start_work_from_videos", "used_video_paths",
           "work_day"]

CHECK_PENDING, CHECK_READY, CHECK_CONFIRMED = "pending", "ready", "confirmed"
MAX_CAMERAS = 4
MAX_PEOPLE = {"school": 20, "sports": 30}       # people (or cars) tracked per camera
DOMAINS = tuple(MAX_PEOPLE)                     # the work domains (DB values)
LONG_VIDEO_SEC = 1200.0                         # above it, the first 20 minutes by default
ESTIMATE_STAGES = (Stage.prepare, Stage.track, Stage.behavior, Stage.contribution)  # per camera

MOTION_NOTE = "カメラが大きく動く動画は、人の取り違えが増えます"
SHORT_PART_NOTE = "短い部分だけを解析すると、取り違えが減ります"
LONG_NOTE = "20分より長い動画なので、はじめの20分を解析します。解析する部分は変えられます。"
WAITING_SENTENCE = "動画の内容を確かめています。しばらくお待ちください。"
NO_PROBE_SENTENCE = PROBE_FAILED   # 「内容を確かめられませんでした。どんな場面の動画か、下から選んでください。」
NO_PROBE_LATER = "内容を確かめられませんでした。"   # a later group: nothing to pick
PICK = "どんな場面の動画か、下から選んでください。"
FIRST_GROUP_FIRST = "先に最初の動画を決定してください"
TOO_MANY_CAMERAS = "カメラは4台までです。動画は4つまで選んでください。"

_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}_")   # save_uploaded_video's unique prefix


@dataclass
class CheckCard:
    """What the check thinks the video is, and the pre-fills of the confirm form.

    state: "waiting" (the check still runs), "ready" or "failed" (no counts: the coach picks).
    needs_pick: no scene verdict (GPU busy, the verdict or the whole check failed).
    people: about how many people (cars for a race) one camera shows; None without counts.
    domain/sport/title/activity: the work's; a later group shows the work's own values.
    profile: the suggested kind of activity (profiles.PROFILES key; None: the coach picks);
    a later group shows the work's."""
    state: str
    scene: str | None
    sport: str | None
    people: int | None
    numbers_visible: bool
    moving: bool | None
    duration_s: float | None
    sentence: str
    needs_pick: bool
    domain: str
    title: str
    activity: str
    method: str
    max_people: int | None
    target: str
    roster_roles: list[str]
    start_sec: float
    end_sec: float | None
    estimate_sec: float | None
    notes: list[str] = field(default_factory=list)
    sets_work: bool = True     # False: a later group, which keeps the work's kind and lines
    auto_title: bool = False   # the title is ours (default_title), not one the coach gave
    day: date | None = None    # the day of the work (default_title's date)
    taken_titles: frozenset[str] = frozenset()   # other works' titles (numbered() avoids them)
    profile: str | None = None


@dataclass
class CheckChoice:
    """The coach's answers on the card. domain/sport/title/activity/rubric_text only count for
    the group that sets the work; a later group keeps the work's."""
    title: str
    activity: str
    rubric_text: str
    domain: str
    sport: str | None
    method: str
    max_people: int | None          # None: the domain's limit
    target: str
    start_sec: float | None
    end_sec: float | None
    clock_offset_min: float | None = None   # 試合時間 at the file's start; None: not given
    model_key: str | None = None    # 解析の細かさ; None keeps the work's (a later group always)
    profile: str | None = None      # the kind of activity; None: the domain's default


# --- starting a work ---------------------------------------------------------------------------

def _title_of(settings: Settings, path: Path) -> str:
    """The file's name without extension (and without the upload's unique prefix)."""
    stem = path.stem
    try:
        if path.resolve().is_relative_to((settings.data_dir / "uploads").resolve()):
            stem = _UPLOAD_PREFIX.sub("", stem, count=1) or stem
    except OSError:
        pass
    return stem.strip() or "無題"


def start_work_from_videos(repo: Repo, settings: Settings, paths: list[Path],
                           model_key: str) -> tuple[int, int]:
    """A new work from dropped videos (one per camera, at most 4): a video assignment named
    after the first file (school until the check is confirmed) and its first group, held for
    the content check. Returns (assignment_id, submission_id). Nothing is left behind when the
    registration fails (VideoError, ValueError)."""
    paths = _camera_paths(paths)
    title = _title_of(settings, paths[0])
    aid = repo.create_assignment(title, Kind.video, "", "", model_key, True, domain="school")
    try:
        sid = _register_held(repo, settings, aid, paths)
    except BaseException:
        repo.delete_assignment(aid)
        raise
    return aid, sid


def add_videos_to_work(repo: Repo, settings: Settings, assignment_id: int,
                       paths: list[Path]) -> int:
    """A later file (one per camera, at most 4) of an existing work: one more group, labelled
    after the first file and held for the content check. Returns the submission id. The work
    itself never changes from here (see confirm_check)."""
    work = repo.get_assignment(assignment_id)
    if work["kind"] != Kind.video.value:
        raise ValueError("この活動には動画を追加できません。")
    if work.get("media_removed_at"):
        from pfb_app.services.intake import CLEARED   # noqa: PLC0415
        raise ValueError(CLEARED)
    paths = _camera_paths(paths)
    own = {_resolved(p) for sub in repo.list_submissions(assignment_id)
           for p in (sub["meta"].get("cameras") or {}).values()}
    again = [p.name for p in paths if _resolved(p) in own]
    if again:
        work_word = profiles.words(profiles.of_work(work)).work
        raise ValueError(f"この動画は、この{work_word}にもう入っています："
                         f"{'、'.join(again)}。別の動画を選んでください。")
    return _register_held(repo, settings, assignment_id, paths)


PART_REST_MIN_S = 30.0   # less video than this after the analysed part is not worth a message


@dataclass(frozen=True)
class AnalysedPart:
    """A group analysed only in part: seconds `start_sec`..`end_sec` of a video of
    `duration_s` seconds."""
    start_sec: float
    end_sec: float
    duration_s: float

    @property
    def sentence(self) -> str:
        """「動画は約30分です。はじめの20分だけを解析しました。」"""
        length = f"動画は約{max(1, math.ceil(self.duration_s / 60))}分です。"
        if self.start_sec <= 0:
            return f"{length}はじめの{_minutes(self.end_sec)}分だけを解析しました。"
        return f"{length}{_minutes(self.start_sec)}分〜{_minutes(self.end_sec)}分だけを解析しました。"

    @property
    def next_label(self) -> str:
        """The button: 「つづきも解析する（20分〜）」."""
        return f"つづきも解析する（{_minutes(self.end_sec)}分〜）"


def analysed_part(repo: Repo, submission_id: int) -> AnalysedPart | None:
    """The part of the video a group analyses when that is not all of it (the coach's range or
    the long-video default ended before the video did, by at least PART_REST_MIN_S); None for
    a group analysed to the end, one without a stated end, or a video of unknown length."""
    sub = repo.get_submission(submission_id)
    meta = sub["meta"] or {}
    end = _num(meta, "end_sec")
    out = _detect_output(repo, submission_id)[1]
    duration = _num(out, "duration_s")
    if end is None or duration is None or duration - end < PART_REST_MIN_S:
        return None
    return AnalysedPart(_num(meta, "start_sec") or 0.0, end, duration)


def next_part_group(repo: Repo, submission_id: int) -> dict | None:
    """The work's group that already continues this group's analysed part (the same videos,
    starting where it ended); None when there is none or the group is analysed to the end."""
    sub = repo.get_submission(submission_id)
    part = analysed_part(repo, submission_id)
    if part is None:
        return None
    cameras = {cam: Path(p) for cam, p in (sub["meta"].get("cameras") or {}).items()}
    for other in repo.list_submissions(sub["assignment_id"]):
        theirs = {cam: Path(p) for cam, p in (other["meta"].get("cameras") or {}).items()}
        if theirs == cameras and abs((_num(other["meta"], "start_sec") or 0.0)
                                     - part.end_sec) < 1.0:
            return other
    return None


def add_next_part(repo: Repo, settings: Settings, assignment_id: int, submission_id: int) -> int:
    """「つづきも解析する」: the same videos again as one more group of the work, from where
    the analysed part ended; held for the content check like any later file (its card
    proposes up to 20 minutes). ValueError when the group is analysed to the end, is not of
    this work, or its continuation is already there. Returns the new submission id."""
    sub = repo.get_submission(submission_id)
    part = analysed_part(repo, submission_id)
    if sub["assignment_id"] != assignment_id or part is None:
        raise ValueError("この動画には、つづきがありません。")
    if next_part_group(repo, submission_id) is not None:
        raise ValueError("つづきは、もう追加されています。")
    return register_video_group(
        repo, settings, assignment_id, label=f"{sub['label']}（{_minutes(part.end_sec)}分〜）",
        cameras={cam: Path(p) for cam, p in (sub["meta"].get("cameras") or {}).items()},
        start_sec=part.end_sec, end_sec=None, transcript_text=None,
        auto_transcribe=True, participants=[], engine="pfb3x3", hold_for_check=True)


def _resolved(path) -> Path:
    try:
        return Path(str(path)).resolve()
    except OSError:
        return Path(str(path))


def used_video_paths(repo: Repo) -> set[Path]:
    """Every video any work's group uses (resolved), so the receiving folder can tell new
    videos from those already in use."""
    return {_resolved(p) for work in repo.list_assignments()
            for sub in repo.list_submissions(work["id"])
            for p in (sub["meta"].get("cameras") or {}).values()}


def _camera_paths(paths: list[Path]) -> list[Path]:
    paths = [Path(p) for p in paths]
    if not paths:
        raise VideoError("動画が選ばれていません")
    if len(paths) > MAX_CAMERAS:
        raise ValueError(TOO_MANY_CAMERAS)
    return paths


def _register_held(repo: Repo, settings: Settings, assignment_id: int,
                   paths: list[Path]) -> int:
    return register_video_group(
        repo, settings, assignment_id, label=_title_of(settings, paths[0]),
        cameras={f"cam{i}": p for i, p in enumerate(paths, 1)}, start_sec=None, end_sec=None,
        transcript_text=None, auto_transcribe=True, participants=[], engine="pfb3x3",
        hold_for_check=True)


# --- reading the check -------------------------------------------------------------------------

def _detect_output(repo: Repo, submission_id: int) -> tuple[str, dict | None]:
    """("waiting", None) while a detect job is active; ("ready", output) for a usable run;
    ("failed", None) otherwise (failed run or job, or no check at all)."""
    if repo.has_active_job(submission_id, Stage.detect):
        return "waiting", None
    run = repo.latest_run(submission_id, Stage.detect)
    out = run.output if run is not None and run.status == "ok" else None
    if not isinstance(out, dict) or not isinstance(out.get("cameras"), dict) \
            or not out["cameras"]:
        return "failed", None
    return "ready", out


def _num(d, key) -> float | None:
    try:
        v = float((d or {}).get(key))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def _counts(out: dict | None, target: str) -> list[dict]:
    if not out:
        return []
    key = "vehicles" if target == "vehicle" else "persons"
    return [p.get(key) or {} for p in out["cameras"].values() if isinstance(p, dict)]


def _people(counts: list[dict]) -> int | None:
    medians = [m for m in (_num(c, "median") for c in counts) if m is not None]
    return int(math.floor(max(medians) + 0.5)) if medians else None


def _max_people(counts: list[dict], domain: str) -> int | None:
    if domain == "sports":
        p90 = [v for v in (_num(c, "p90") for c in counts) if v is not None]
        return (min(MAX_PEOPLE["sports"], math.ceil(max(p90))) or None) if p90 else None
    most = [v for v in (_num(c, "max") for c in counts) if v is not None]
    return (min(MAX_PEOPLE["school"], math.ceil(max(most))) or None) if most else None


def _moving(out: dict | None) -> bool | None:
    if not out:
        return None
    flags = [(p.get("camera_motion") or {}).get("moving") for p in out["cameras"].values()
             if isinstance(p, dict)]
    if any(f is True for f in flags):
        return True
    return False if any(f is False for f in flags) else None


def _scene(out: dict | None) -> tuple[str | None, str | None, bool]:
    """(scene, sport, numbers_visible) of the verdict; (None, None, False) without one."""
    verdict = (out or {}).get("scene")
    if not isinstance(verdict, dict) or verdict.get("scene") not in SCENE_LABELS:
        return None, None, False
    scene = verdict["scene"]
    sport = None
    if scene == "sport":
        sport = verdict.get("sport") if verdict.get("sport") in SPORT_KEYS else "other"
    return scene, sport, bool(verdict.get("numbers_visible"))


def _seen_profile(out: dict | None) -> str | None:
    """The profile the verdict suggests (profiles.for_scene); None without a verdict."""
    scene = _scene(out)[0]
    verdict = (out or {}).get("scene")
    activity = verdict.get("activity") if isinstance(verdict, dict) else None
    return profiles.for_scene(scene, activity if isinstance(activity, str) else None)


def _named(profile: str | None) -> bool:
    """A profile with its own scene wording (not 一般, not スポーツ, not None)."""
    return profile is not None and profile not in profiles.UNCHANGED


def _target_for(sport: str | None) -> str:
    return "vehicle" if sport == "motor_racing" else "person"


def sport_phrase(sport: str | None) -> str:
    if sport == "motor_racing":
        return "モーターレース"
    if sport in SPORT_LABELS and sport != "other":
        return f"{SPORT_LABELS[sport]}の試合"
    return "スポーツの試合"


def _activity(scene: str | None, sport: str | None, profile: str | None = None) -> str:
    if scene == "sport":
        return sport_phrase(sport)
    if _named(profile):
        return profiles.activity_text(profile)
    return "グループ活動" if scene == "group_work" else ""


def card_sentence(scene: str | None, sport: str | None, people: int | None, target: str,
                  gpu_busy: bool, *, ask: bool = True, profile: str | None = None) -> str:
    """The card's one plain sentence: 「サッカーの試合のようです（推定）。映っている人は約14人。」,
    「話し合い・会議のようです（推定）。…」 for a profile the verdict named.
    gpu_busy (or no verdict): the count only, and (with `ask`) the coach picks. `ask=False`
    for a later group, whose kind comes from the work."""
    if people is None:
        return NO_PROBE_SENTENCE if ask else NO_PROBE_LATER
    thing, unit = ("車", "台") if target == "vehicle" else ("人", "人")
    seen = f"映っている{thing}は約{people}{unit}。" if people else f"{thing}が見つかりませんでした。"
    if gpu_busy or scene not in SCENE_LABELS:
        return seen + (PICK if ask else "")
    if scene == "sport":
        return f"{sport_phrase(sport)}のようです（推定）。{seen}"
    if _named(profile):
        return f"{profiles.label(profile)}のようです（推定）。{seen}"
    if scene == "group_work":
        return f"グループ活動のようです（推定）。{seen}"
    return f"グループ活動やスポーツではないようです（推定）。{seen}"


def work_day(work: dict) -> date:
    """The local day the work was created (the date in its default title)."""
    try:
        return datetime.fromisoformat(work["created_at"]).astimezone().date()
    except (KeyError, TypeError, ValueError):
        return date.today()


def default_title(scene: str | None, sport: str | None, day: date,
                  profile: str | None = None) -> str:
    """The card's title until the coach writes one: 「サッカーの試合 10月22日」,
    「会議 10月22日」 (a profile with its own words), 「グループ活動 10月22日」, else
    「動画 10月22日」 (never the file name)."""
    if scene == "sport":
        what = sport_phrase(sport)
    elif _named(profile):
        what = profiles.title_word(profile)
    else:
        what = "グループ活動" if scene == "group_work" else "動画"
    return f"{what} {day.month}月{day.day}日"


def numbered(title: str, taken) -> str:
    """「動画 10月22日」, or 「動画 10月22日（2）」, 「（3）」… when another work has it."""
    if title not in taken:
        return title
    n = 2
    while f"{title}（{n}）" in taken:
        n += 1
    return f"{title}（{n}）"


def _is_file_title(settings: Settings, work: dict, submission: dict) -> bool:
    """True when the work's title is still the one start_work_from_videos gave it (the first
    file's name), so the card offers default_title instead."""
    cameras = list((submission["meta"].get("cameras") or {}).values())
    return bool(cameras) and work["title"] == _title_of(settings, Path(str(cameras[0])))


def _about(seconds: float) -> str:
    """「約15分」: rounded up, to 5 minutes from 10, to 10 from an hour (as the waiting
    screens round their estimates)."""
    minutes = max(1, math.ceil(seconds / 60))
    if minutes >= 60:
        minutes = math.ceil(minutes / 10) * 10
        hours, rest = divmod(minutes, 60)
        return f"約{hours}時間" + (f"{rest}分" if rest else "")
    if minutes >= 10:
        minutes = math.ceil(minutes / 5) * 5
    return f"約{minutes}分"


def _minutes(seconds: float) -> str:
    m = round(seconds / 60, 1)
    return f"{m:g}"


NOT_PICKED = "まだ選ばれていません"


def check_facts(*, scene: str | None, sport: str | None, people: int | None, target: str,
                start_sec: float, end_sec: float | None, estimate_sec: float | None,
                duration_s: float | None = None, limit: int | None = None,
                profile: str | None = None) -> list[tuple[str, str]]:
    """The short card's three facts, from what is chosen now:
    [("種目", "サッカー"), ("評価する人", "約14人"), ("解析する部分", "はじめの20分（解析の目安 約35分）")].
    `people` is the check's count, `limit` the coach's cap when lower."""
    if scene == "sport":
        first = ("種目", SPORT_LABELS.get(sport, NOT_PICKED) if sport else NOT_PICKED)
    elif _named(profile):
        first = ("場面", profiles.label(profile))
    else:
        first = ("場面", {"group_work": "グループ活動", "other": "その他の動画"}.get(
            scene or "", NOT_PICKED))
    thing, unit = ("車・バイク", "台") if target == "vehicle" else ("人", "人")
    if people is None:
        count = "動画から自動で数えます"
    elif people == 0:
        count = "見つかりませんでした"
    else:
        count = f"約{min(people, limit) if limit else people}{unit}"
    start = float(start_sec or 0.0)
    details = []
    if end_sec is None:
        part = "全体" if start <= 0 else f"{_minutes(start)}分から最後まで"
        if start <= 0 and duration_s:
            details.append(f"約{max(1, math.ceil(duration_s / 60))}分")
    elif start <= 0:
        part = f"はじめの{_minutes(end_sec)}分"
    else:
        part = f"{_minutes(start)}分〜{_minutes(end_sec)}分"
    if estimate_sec:
        details.append(f"解析の目安 {_about(estimate_sec)}")
    if details:
        part += f"（{'・'.join(details)}）"
    return [first, (f"評価する{thing}", count), ("解析する部分", part)]


def roster_roles(domain: str, sport: str | None, numbers_visible: bool) -> list[str]:
    if domain != "sports":
        roles = ["name", "appearance", "place"]
    elif sport == "baseball_softball":
        roles = ["name", "number", "position", "batting"]
    elif sport == "motor_racing":
        roles = ["name", "number", "grid"]
    else:
        roles = ["name", "number", "position", "team"]
    if numbers_visible:
        roles = ["number"] + [r for r in roles if r != "number"]
    return roles


def estimate_for(repo: Repo, method: str, cameras: int, people: int,
                 length: float | None = None) -> float | None:
    """A rough analysis time (seconds): the past time of prepare, track, behavior and
    contribution per camera, plus feedback (and its practice tips) per person. Each stage is priced for a video of
    `length` seconds by services.eta.price (a robust line through the earlier runs when
    there are enough of different lengths, else their median); None without that history."""
    history = eta.history(repo, method)
    total = 0.0
    stages = [(s, cameras) for s in ESTIMATE_STAGES] + ([(Stage.pfb, people)] if people else [])
    for stage, n in stages:
        seconds = eta.price_of(history, stage.value,
                               length if stage.value in eta.LENGTH_STAGES else None)
        if seconds is None:
            return None
        total += n * seconds
    return total


def _work_is_set(repo: Repo, submission: dict) -> bool:
    """True when another group of the work already set it: confirmed, or registered without
    the check (the work was set up by hand)."""
    return any(other["id"] != submission["id"]
               and other["meta"].get("check", CHECK_CONFIRMED) == CHECK_CONFIRMED
               for other in repo.list_submissions(submission["assignment_id"]))


def build_card(repo: Repo, settings: Settings, submission_id: int) -> CheckCard:
    """The card for one group. Reading it marks a pending group "ready" once its check has
    finished (or failed)."""
    sub = repo.get_submission(submission_id)
    meta = sub["meta"]
    work = repo.get_assignment(sub["assignment_id"])
    state, out = _detect_output(repo, submission_id)
    if state != "waiting" and meta.get("check") == CHECK_PENDING:
        try:   # fallback: the runner marks it when the detect job ends
            mark_check_ready(repo, submission_id)
        except sqlite3.OperationalError:
            log.exception("内容確認の完了を記録できませんでした submission=%s", submission_id)
    scene_seen, sport_seen, numbers_visible = _scene(out)
    profile_seen = _seen_profile(out)
    scene = scene_seen
    later = _work_is_set(repo, sub)
    if later:   # the work's kind: nothing to pick
        domain, sport = work["domain"], work["sport"]
        activity = work["task_text"]
        profile = profiles.of_work(work)
        scene = "sport" if domain == "sports" else (
            scene_seen if scene_seen in ("group_work", "other") else "group_work")
    else:
        domain = "sports" if scene == "sport" else "school"
        sport = sport_seen
        profile = profile_seen
        activity = work["task_text"] or _activity(scene, sport_seen, profile)
    target = _target_for(sport)
    counts = _counts(out, target)
    people = _people(counts)
    duration = _num(out, "duration_s") if out else None
    start = float(meta.get("start_sec") or 0.0)
    end = meta.get("end_sec")
    notes: list[str] = []
    if end is None and duration is not None and duration - start > LONG_VIDEO_SEC:
        end = start + LONG_VIDEO_SEC
        notes.append(LONG_NOTE)
    moving = _moving(out)
    if moving:
        notes += [MOTION_NOTE, SHORT_PART_NOTE]
    note = later_note(repo, submission_id)
    if note:
        notes.append(note)
    if state == "waiting":
        sentence = WAITING_SENTENCE
    else:
        sentence = card_sentence(scene_seen, sport_seen, people, target,
                                 gpu_busy=scene_seen is None, ask=not later,
                                 profile=profile_seen)
    method = "pfb3x3"
    auto_title = not later and _is_file_title(settings, work, sub)
    day = work_day(work)
    taken = frozenset(a["title"] for a in repo.list_assignments() if a["id"] != work["id"]) \
        if auto_title else frozenset()
    return CheckCard(
        state=state, scene=scene, sport=sport, people=people, numbers_visible=numbers_visible,
        moving=moving, duration_s=duration, sentence=sentence,
        needs_pick=not later and state != "waiting" and scene is None, domain=domain,
        title=numbered(default_title(scene, sport, day, profile), taken) if auto_title
        else work["title"],
        activity=activity, method=method, max_people=_max_people(counts, domain),
        target=target, roster_roles=roster_roles(domain, sport, numbers_visible),
        start_sec=start, end_sec=None if end is None else float(end),
        estimate_sec=estimate_for(repo, method, len(meta.get("cameras") or {}), people or 0,
                                  eta.window(duration, start, end)),
        notes=notes, sets_work=not later, auto_title=auto_title, day=day, taken_titles=taken,
        profile=profile)


def earlier_roster_text(repo: Repo, submission_id: int) -> str:
    """For a later group: the roster of the work's first confirmed group as pasted rows
    (roster.roster_text), the later card's starting point, which the coach can still edit.
    Empty for the group that sets the work, or when that group has no roster."""
    sub = repo.get_submission(submission_id)
    first = next((other for other in repo.list_submissions(sub["assignment_id"])
                  if other["id"] != submission_id
                  and other["meta"].get("check", CHECK_CONFIRMED) == CHECK_CONFIRMED), None)
    if first is None:
        return ""
    return roster_text(roster_subjects(repo.list_subjects(first["id"]), first["meta"]))


def later_note(repo: Repo, submission_id: int) -> str | None:
    """For a later group whose check disagrees with the work:
    「この動画は{new}のようですが、この{work}は{current}として進めます。」; else None."""
    sub = repo.get_submission(submission_id)
    if not _work_is_set(repo, sub):
        return None
    scene, sport, _ = _scene(_detect_output(repo, submission_id)[1])
    if scene is None:
        return None
    work = repo.get_assignment(sub["assignment_id"])
    if work["domain"] == "sports":
        if scene == "sport" and sport == (work["sport"] or sport):
            return None
        current = sport_phrase(work["sport"])
    else:
        if scene != "sport":
            return None
        profile = profiles.of_work(work)
        current = profiles.label(profile) if _named(profile) else "グループ活動"
    new = sport_phrase(sport) if scene == "sport" else (
        "グループ活動" if scene == "group_work" else "スポーツ以外の動画")
    w = profiles.words(profiles.of_work(work))
    return f"この動画は{new}のようですが、この{w.work}は{current}として進めます。"


# --- confirming --------------------------------------------------------------------------------

def _not_confirmed(submission: dict) -> None:
    if submission["meta"].get("check", CHECK_CONFIRMED) == CHECK_CONFIRMED:
        raise BusyError("この動画は、もう決定して解析を始めています。")


def _earlier_held(repo: Repo, submission: dict) -> None:
    """BusyError while an earlier group of the work still waits for its confirm: the first
    file sets the work."""
    if any(other["id"] < submission["id"]
           and other["meta"].get("check", CHECK_CONFIRMED) != CHECK_CONFIRMED
           for other in repo.list_submissions(submission["assignment_id"])):
        raise BusyError(FIRST_GROUP_FIRST)


def _float(value, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what}を数字で入れてください。") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{what}は0以上にしてください。")
    return v


def confirm_check(repo: Repo, settings: Settings, submission_id: int, choice: CheckChoice,
                  roster: Roster | None = None) -> None:
    """The coach's 「この内容で始める」 for one group.

    The group that sets the work writes the assignment's title, activity line, rubric, domain,
    profile (`choice.profile`, None: the domain's default; one of another domain is a
    ValueError) and sport (and 解析の細かさ when `choice.model_key` is given); a later group
    leaves the work untouched (its limits follow the work's domain).
    The group's meta gets engine, max_people and target (tracking method only), start_sec,
    end_sec, clock_offset_min and check = "confirmed"; the roster (when given) replaces the
    people; then `prepare` is enqueued per camera. ValueError for an empty title or activity
    or an invalid choice (nothing changes); BusyError once the group is confirmed."""
    title, activity = (choice.title or "").strip(), (choice.activity or "").strip()
    if not title:
        raise ValueError("タイトルを入れてください。")
    if not activity:
        raise ValueError("どんな活動かを入れてください。")
    sub = repo.get_submission(submission_id)
    _not_confirmed(sub)
    _earlier_held(repo, sub)
    work = repo.get_assignment(sub["assignment_id"])
    sets_work = not _work_is_set(repo, sub)
    domain, sport = (choice.domain, choice.sport) if sets_work else (work["domain"],
                                                                     work["sport"])
    if domain not in DOMAINS:
        raise ValueError("一般の活動かスポーツかを選んでください。")
    profile = profiles.check(choice.profile if sets_work else profiles.of_work(work), domain)
    if domain == "sports":
        if sport not in SPORT_KEYS:
            raise ValueError("スポーツの種類を選んでください。")
    else:
        sport = None
    model_key = choice.model_key if sets_work else None
    if model_key is not None and model_key not in settings.models:
        raise ValueError("解析の細かさを選んでください。")
    method = choice.method
    if method not in ENGINES:
        raise ValueError("解析の方式を選んでください。")
    if domain == "sports" and method != "pfb3x3":
        raise ValueError("スポーツの動画は、トラッキング方式で解析します。")
    limit = MAX_PEOPLE[domain]
    max_people = target = None
    if method == "pfb3x3":
        target = choice.target
        if target not in TARGETS:
            raise ValueError("追跡する対象（人か車）を選んでください。")
        max_people = limit if choice.max_people is None else choice.max_people
        if isinstance(max_people, bool) or not isinstance(max_people, (int, float)) \
                or max_people != int(max_people) or not 1 <= max_people <= limit:
            raise ValueError(f"追跡する人数は1〜{limit}で入れてください。")
        max_people = int(max_people)
    # only a value the coach entered is a known game clock (0 included); blank stays None
    offset = (None if choice.clock_offset_min is None
              else _float(choice.clock_offset_min, "この動画の始まりの試合時間（分）"))
    start = None if choice.start_sec is None else _float(choice.start_sec, "解析の始まり")
    end = None if choice.end_sec is None else _float(choice.end_sec, "解析の終わり")
    cameras: dict[str, str] = sub["meta"].get("cameras") or {}
    for i, (cam, path) in enumerate(cameras.items(), 1):
        try:
            validate_interval(probe_video(path), start, end, settings.frame_interval_sec)
        except VideoError as e:
            raise ValueError(f"カメラ{i}（{Path(path).name}）: {e}") from None
    with repo._tx():
        sub = repo.get_submission(submission_id)
        _not_confirmed(sub)
        _earlier_held(repo, sub)
        if sets_work == _work_is_set(repo, sub):
            raise BusyError("ほかの動画が先に決定されました。もう一度確かめてください。")
        if roster is not None:
            # in the same write transaction as the checks: a second tab confirming the same
            # group never replaces the people the first tab's confirm brought in
            apply_roster(repo, submission_id, roster, domain)
        if sets_work:
            repo.update_assignment_check(work["id"], title=title, task_text=activity,
                                         rubric_text=(choice.rubric_text or "").strip(),
                                         domain=domain, sport=sport, model_key=model_key,
                                         profile=profile)
        # read again: the roster wrote its own keys (roster_run_floor, added people)
        meta = dict(repo.get_submission(submission_id)["meta"])
        meta.update(engine=method, start_sec=start, end_sec=end, clock_offset_min=offset,
                    check=CHECK_CONFIRMED)
        if method == "pfb3x3":
            meta.update(max_people=max_people, target=target)
        else:
            meta.pop("max_people", None)
            meta.pop("target", None)
        repo.update_submission_meta(submission_id, meta)
        for cam in cameras:
            enqueue_once(repo, submission_id, Stage.prepare, camera=cam)
    repo.audit("confirm_check", f"submission:{submission_id}",
               {"sets_work": sets_work, "domain": domain, "profile": profile, "sport": sport,
                "method": method,
                "model": model_key,
                "max_people": max_people, "target": target, "roster": roster is not None})
