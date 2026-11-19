"""Automatic people (tracking engine): who in the video is evaluated, with no coach step.

After a camera's track run, `auto_identity` saves a mapping (an identity run with source
"auto") of the people worth evaluating to subjects named 「参加者n」 / 「選手n」, and
`auto_confirm` confirms every camera's mapping once all of them are current, which queues
behavior. `reselect` (「映っている人を自動で選び直す」) does both again on the existing track runs;
`rerun_behavior` (「もう一度解析する」) records the people's behavior again.
After a names run, `number_names` renames the automatic people whose shirt / car number was
read to 「#10」 / 「10号車」 (the coach's own names always win).
The coach can still change the mapping on the people screen; names are optional.
"""
import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from pfb_app.domain.models import Stage, StageRun
from pfb_app.engine.pfb3x3 import EngineError, Pfb3x3Engine
from pfb_app.export import tracks as track_files
from pfb_app.pipeline.auto_people import (box_stats, detect_p90, is_evaluated, people_cap,
                                          select_labels)
from pfb_app.pipeline import identity_check
from pfb_app.pipeline.participation import CHECK, NOT_PLAYING, participation
from pfb_app.pipeline.common import submission_engine, usable
from pfb_app.pipeline.name_fusion import MAX_DIGITS, MIN_FRAMES_AGREE
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.pipeline.tracked_identity import (TrackedIdentityOutput, label_mapping,
                                               mapped_track_ids, mapping_is_current)
from pfb_app.pipeline.tracking import video_id
from pfb_app.services import review
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

__all__ = ["auto_confirm", "auto_identity", "include_person", "is_evaluated", "name_notes",
           "number_names", "participation_view", "rename_person", "reselect", "person_crops",
           "rerun_behavior", "set_person_excluded", "short_people", "label_crop", "unused_roster"]

BUSY = "解析が進んでいる間は変更できません。終わってから、もう一度お試しください。"
BUSY_RERUN = "解析が進んでいる間はやり直せません。終わってから、もう一度お試しください。"
QUALITY_TIMEOUT_SEC = 120   # reselect's engine quality call: the coach is waiting for it
AUTO_SOURCE = "auto"   # the source of an identity run made here (not the coach's own mapping)
AUTO_KIND = "auto"   # meta["added_people"] kind of the people made here (as on the mapping screen)
PARTICIPATION = "participation"   # meta: {camera: {"track", "not_playing", "check", "included"}}
NAME_SOURCE = review.NAME_SOURCE   # meta: {participant_id: "number" | "coach"}
NUMBER_NOTE = {"person": "背番号から", "vehicle": "番号から"}   # shown under a number name
TEAM_WORDS = {"T1": "チーム1", "T2": "チーム2"}
_DIGITS = re.compile(rf"[0-9]{{1,{MAX_DIGITS}}}")
_NUMBER_NAME = re.compile(rf"(#[0-9]{{1,{MAX_DIGITS}}}|[0-9]{{1,{MAX_DIGITS}}}号車)(（[^（）]*）)?")


def _cameras(repo: Repo, submission_id: int, submission: dict) -> list[str]:
    cameras = list(submission["meta"].get("cameras") or {})
    return cameras + [c for c in repo.list_cameras(submission_id) if c not in cameras]


def _cap(repo: Repo, submission_id: int, submission: dict, camera: str, domain: str) -> int:
    detect = repo.latest_run(submission_id, Stage.detect)
    p90 = detect_p90(detect.output if usable(detect) else None, camera,
                     submission["meta"].get("target", "person"))
    return people_cap(submission["meta"].get("max_people"), p90, domain)


def _previous(repo: Repo, submission_id: int, camera: str, track: StageRun) -> dict:
    """{label: participant_id} of the camera's latest identity, when it was made for `track`."""
    identity = repo.latest_run(submission_id, Stage.identity, camera)
    if not usable(identity) or track.id not in mapped_track_ids(identity):
        return {}
    try:
        return {lb: pid for lb, pid in label_mapping(identity.output).items() if pid}
    except (KeyError, TypeError):
        log.warning("the identity output of %s is malformed; no earlier mapping is kept", camera)
        return {}


def current_mappings(repo: Repo, submission_id: int,
                     camera: str) -> dict[int, dict[str, str | None]]:
    """{track run id: {label: participant id}} of the newest usable identity run made for each
    of the camera's track runs (a label means a person only within its track run, and only
    the newest mapping counts: a corrected swap moves the person to another label)."""
    out: dict[int, dict[str, str | None]] = {}
    for run in reversed(repo.run_history(submission_id, Stage.identity, camera)):
        if not usable(run):
            continue
        try:
            mapping = label_mapping(run.output)
        except (KeyError, TypeError):
            continue
        for track_id in mapped_track_ids(run):
            out.setdefault(track_id, mapping)
    return out


def erased_labels(repo: Repo, submission_id: int, camera: str, track_id: int,
                  erased: set[str]) -> dict[str, str]:
    """{label: participant id} of this track run's labels that its current mapping gives to a
    person whose data was removed on request (services.erasure): they stay mapped to that
    (excluded) placeholder, so nobody new is made in their place."""
    if not erased:
        return {}
    mapping = current_mappings(repo, submission_id, camera).get(track_id) or {}
    return {lb: pid for lb, pid in mapping.items() if pid in erased}


def _mapped_everywhere(repo: Repo, submission_id: int, submission: dict) -> set[str]:
    mapped: set[str] = set()
    for cam in _cameras(repo, submission_id, submission):
        run = repo.latest_run(submission_id, Stage.identity, cam)
        if run is not None and isinstance(run.output, dict):
            mapped |= {p.get("participant_id") for p in run.output.get("participants") or []}
    return mapped - {None}


def _game(repo: Repo, submission: dict) -> bool:
    """A game of people: the participation rule applies (pipeline.participation)."""
    return (review._domain(repo, submission) == "sports"
            and submission["meta"].get("target", "person") == "person")


def _teams(repo: Repo, submission_id: int, camera: str, track: StageRun) -> dict[str, str]:
    for run in review._track_names_runs(repo, submission_id, camera, track):
        teams = (run.output or {}).get("teams")
        if isinstance(teams, dict) and teams:
            return {str(k): str(v) for k, v in teams.items()}
    return {}


def _participation(repo: Repo, submission_id: int, submission: dict, camera: str,
                   track: StageRun, labels: list[str], quality: dict | None) -> dict:
    """The camera's participation record for this track run: {"track", "not_playing":
    {label: reason}, "check": {label: reason}, "included": [labels the coach included]}.
    Labels the coach included (「この人も評価する」, kept while the track run is the same)
    are never not playing. When nobody would be left, everyone is to be checked instead."""
    old = (submission["meta"].get(PARTICIPATION) or {}).get(camera) or {}
    included = [lb for lb in old.get("included") or [] if lb in labels] \
        if old.get("track") == track.id else []
    boxes_path = (track.output or {}).get("boxes")
    meta = track_files.track_meta(boxes_path)
    status = participation(labels, boxes=track_files.load_boxes(boxes_path),
                           geometry=track_files.load_geometry(boxes_path),
                           fps=meta.fps if meta else None, quality=quality,
                           teams=_teams(repo, submission_id, camera, track))
    out = {lb: st for lb, st in status.items() if lb not in included}
    not_playing = {lb: why for lb, (st, why) in out.items() if st == NOT_PLAYING}
    check = {lb: why for lb, (st, why) in out.items() if st == CHECK}
    if labels and len(not_playing) == len(labels):
        not_playing, check = {}, {lb: "nobody" for lb in labels}
    return {"track": track.id, "not_playing": not_playing, "check": check,
            "included": included}


def _included(submission: dict, camera: str, track: StageRun, labels: list[str]) -> list[str]:
    """The labels the coach included on this camera's current track run (「この人も評価する」)."""
    old = (submission["meta"].get(PARTICIPATION) or {}).get(camera) or {}
    return [lb for lb in old.get("included") or [] if lb in labels] \
        if old.get("track") == track.id else []


def auto_identity(repo: Repo, submission_id: int, camera: str, *,
                  quality_fn=None) -> StageRun | None:
    """Save the camera's automatic mapping for its latest usable track run; None (nothing
    saved) when there is none. `quality_fn(track_run) -> dict | None` is asked only when the
    track output has no "quality" key (a track run made before quality existed).

    1. If the track output has rule defaults (`default_names`, `default_source` "rule"), the
       mapping is exactly those (label -> roster participant id), the other labels 対象外;
       no people are made.
    2. Else the labels are `select_labels(...)` (quality, boxes.txt, `people_cap`). A selected
       label already mapped (by the camera's latest identity made for this same track run) to
       a not-excluded subject keeps that subject; the others get new subjects with the next
       default names (「参加者n」 / 「選手n」, numbered on across cameras), participant ids
       "auto1", "auto2", ..., recorded in meta["added_people"] as "auto".
    3. One identity run is saved (source "auto", upstream [track.id], validated like a coach's
       mapping), all in one transaction with the people it made.
    4. Afterwards the auto people that no camera's latest mapping uses are removed when they
       have no runs, else marked excluded; roster subjects that no camera's latest mapping
       uses are marked excluded (roster subjects a rule mapping uses are evaluated again).
       Audited as "auto_identity" with counts only."""
    track = repo.latest_run(submission_id, Stage.track, camera)
    if not usable(track) or not isinstance(track.output, dict):
        return None
    out = track.output
    labels = list(out.get("labels") or [])
    submission = repo.get_submission(submission_id)
    domain = review._domain(repo, submission)
    defaults = dict(out.get("default_names") or {}) if out.get("default_source") == "rule" else {}
    method = "rule"
    selected: list[str] = []
    part: dict | None = None
    if not defaults:
        if "quality" in out:
            quality = out.get("quality")
        else:
            quality = quality_fn(track) if quality_fn is not None else None
        method = "quality" if quality is not None else "boxes"
        stats = box_stats(out["boxes"], labels) if out.get("boxes") else {}
        cap = _cap(repo, submission_id, submission, camera, domain)
        pool = labels
        if _game(repo, submission):
            part = _participation(repo, submission_id, submission, camera, track, labels,
                                  quality)
            pool = [lb for lb in labels if lb not in part["not_playing"]]
        selected = select_labels(pool, quality, stats, cap)
        if part is None:   # not a game: the coach's 「この人も評価する」 still counts
            kept = _included(submission, camera, track, labels)
            if kept:
                part = {"track": track.id, "not_playing": {}, "check": {}, "included": kept}
        selected += [lb for lb in part["included"] if lb not in selected] if part else []
    previous = _previous(repo, submission_id, camera, track)
    made: list[str] = []
    with repo._tx():
        submission = repo.get_submission(submission_id)
        meta = dict(submission["meta"] or {})
        subjects = repo.list_subjects(submission_id)
        if defaults:
            known = {s["participant_id"] for s in subjects if s["participant_id"]}
            mapping = {lb: (defaults[lb] if defaults.get(lb) in known else None)
                       for lb in labels}
        else:
            usable_ids = {s["participant_id"] for s in subjects
                          if s["participant_id"] and not s["excluded"]}
            # a person whose data was removed on request (services.erasure): their labels
            # stay mapped to their placeholder, so nobody new is made in their place
            erased = erased_labels(repo, submission_id, camera, track.id,
                                   {s["participant_id"] for s in subjects if s.get("erased_at")})
            planner = review.name_planner(domain, subjects, meta)
            number = planner.last["default"]
            taken = {s["participant_id"] for s in subjects if s["participant_id"]}
            added = review.added_people(meta)
            mapping = {lb: erased.get(lb) for lb in labels}
            for lb in selected:
                if lb in erased:
                    continue
                if previous.get(lb) in usable_ids:
                    mapping[lb] = previous[lb]
                    continue
                number += 1
                pid = review._free_id(AUTO_KIND, taken)
                repo.add_subject(submission_id, pid, planner._make("default", number), None,
                                 {}, {})
                taken.add(pid)
                added[pid] = AUTO_KIND
                made.append(pid)
                mapping[lb] = pid
            if made:
                meta[review.ADDED] = added
                review._forget_name_source(meta, made)
            if part is not None:
                meta[PARTICIPATION] = {**(meta.get(PARTICIPATION) or {}), camera: part}
            if made or part is not None:
                repo.update_submission_meta(submission_id, meta)
        parsed = review._parse(TrackedIdentityOutput, {
            "participants": [{"participant_id": pid, "label": lb, "display_name": None}
                             for lb, pid in mapping.items()],
            "labels": labels, "track_run_id": track.id})
        output, warnings = review._checked_mapping(repo, submission_id, camera, parsed)
        saved = repo.save_stage_run(StageRun(
            id=None, submission_id=submission_id, stage=Stage.identity, camera=camera,
            subject_id=None, version=0, source="auto", upstream_run_ids=[track.id],
            prompt_hash=None, model_id=None, output=output, status="ok", error=None,
            warnings=warnings, duration_sec=0.0, created_at=""))
    removed, excluded = _tidy(repo, submission_id,
                              {pid for pid in mapping.values() if pid} if defaults else set())
    repo.audit("auto_identity", f"run:{saved.id}", {
        "camera": camera, "track_run_id": track.id, "method": method, "labels": len(labels),
        "evaluated": sum(1 for pid in mapping.values() if pid), "people_made": len(made),
        "people_removed": removed, "people_excluded": excluded,
        "not_playing": len(part["not_playing"]) if part else 0,
        "to_check": len(part["check"]) if part else 0})
    return saved


def _tidy(repo: Repo, submission_id: int, restore: set[str]) -> tuple[int, int]:
    """Rule 4 of auto_identity; `restore`: roster participant ids to evaluate again.
    Returns (people removed, people newly excluded)."""
    submission = repo.get_submission(submission_id)
    mapped = _mapped_everywhere(repo, submission_id, submission)
    added = review.added_people(submission["meta"])
    # a person whose data was removed on request keeps their placeholder as it is
    subjects = [s for s in repo.list_subjects(submission_id) if not s.get("erased_at")]
    gone = [s["participant_id"] for s in subjects if added.get(s["participant_id"]) == AUTO_KIND
            and s["participant_id"] not in mapped and not review._has_runs(repo, s["id"])]
    review._remove_added(repo, submission_id, gone)
    drop = {s["id"] for s in subjects if added.get(s["participant_id"]) == AUTO_KIND
            and s["participant_id"] not in mapped and s["participant_id"] not in gone}
    roster = review.roster_subjects(subjects, submission["meta"])
    drop |= {s["id"] for s in roster if s["participant_id"] not in mapped}
    newly = 0
    with repo._tx():
        for s in subjects:
            if s["id"] in drop and not s["excluded"]:
                repo.set_excluded(s["id"], True)
                newly += 1
            elif s["participant_id"] in restore and s["excluded"]:
                repo.set_excluded(s["id"], False)
    return len(gone), newly


def _ready(repo: Repo, submission_id: int, cameras: list[str]) -> bool:
    for cam in cameras:
        if not mapping_is_current(repo, submission_id, cam):
            return False
        if any(repo.has_active_job(submission_id, st, c) for st, c in (
                (Stage.prepare, cam), (Stage.prepare, None), (Stage.track, cam),
                (Stage.identity, cam))):
            return False
    return True


def auto_confirm(repo: Repo, submission_id: int) -> list[int]:
    """review.confirm_identity once every registered camera has a current mapping and no
    prepare / track / identity job is active; returns the behavior job ids it queued, [] when
    not ready (never raises ApprovalError)."""
    submission = repo.get_submission(submission_id)
    cameras = list(submission["meta"].get("cameras") or {})
    if not cameras or not _ready(repo, submission_id, cameras):
        return []
    try:
        return list(review.confirm_identity(repo, submission_id))
    except review.ApprovalError as e:
        log.info("自動で人物を確定できませんでした submission=%s: %s", submission_id, e)
        return []


def reselect(repo: Repo, settings, submission_id: int) -> list[int]:
    """「映っている人を自動で選び直す」: a new auto identity for every camera on its existing
    track run (no re-tracking), then auto_confirm. The track output's quality is used, else
    the engine's `quality` (on the CPU, QUALITY_TIMEOUT_SEC; an EngineError or a timeout
    means none), else boxes.txt. BusyError
    while a blocking job runs. Returns the behavior job ids queued."""
    submission = repo.get_submission(submission_id)
    if submission_engine(submission) != "pfb3x3":
        raise review._invalid("このグループはトラッキング方式ではありません。")
    if repo.has_blocking_jobs(submission_id):
        raise review.BusyError(review._busy_message("映っている人の選び直し"))
    engine = None
    sub_dir = files.submission_dir(settings, submission["assignment_id"], submission_id)

    def quality_fn(track: StageRun):
        nonlocal engine
        if engine is None:
            engine = Pfb3x3Engine(settings)
        vid = track.output.get("video_id") or video_id(submission_id, track.camera)
        try:
            return engine.quality(sub_dir, vid, timeout_sec=QUALITY_TIMEOUT_SEC)
        except EngineError as e:
            log.warning("quality failed for %s: %s", vid, e)
            return None

    cameras = _cameras(repo, submission_id, submission)
    saved = [cam for cam in cameras
             if auto_identity(repo, submission_id, cam, quality_fn=quality_fn) is not None]
    ids = auto_confirm(repo, submission_id)
    try:   # people made now get the numbers already read
        number_names(repo, submission_id)
    except Exception:   # noqa: BLE001 - names stay as they were; the coach can name people
        log.exception("背番号から名前をつけられませんでした submission=%s", submission_id)
    repo.audit("auto_people_reselect", f"submission:{submission_id}",
               {"cameras": saved, "job_ids": ids})
    return ids


@dataclass(frozen=True)
class NotPlaying:
    """A tracked person the participation rule left out (試合に参加していない人)."""
    camera: str
    label: str
    track_run_id: int
    reason: str


@dataclass(frozen=True)
class ParticipationView:
    not_playing: list[NotPlaying]     # people not evaluated, camera then label order
    check: dict[int, str]             # {subject_id: reason} of evaluated people to check


def participation_view(repo: Repo, submission_id: int) -> ParticipationView:
    """Who the participation rule left out and whom it marked 要確認, for the cameras' latest
    track runs (a record made for an older track run does not count)."""
    submission = repo.get_submission(submission_id)
    records = submission["meta"].get(PARTICIPATION) or {}
    by_pid = {s["participant_id"]: s["id"] for s in repo.list_subjects(submission_id)
              if s.get("participant_id") and is_evaluated(s)}
    out, check = [], {}
    for camera in _cameras(repo, submission_id, submission):
        rec = records.get(camera) or {}
        track = repo.latest_run(submission_id, Stage.track, camera)
        if not usable(track) or rec.get("track") != track.id:
            continue
        mapping = current_mappings(repo, submission_id, camera).get(track.id) or {}
        for label in sorted(rec.get("not_playing") or {}, key=lambda lb: (len(lb), lb)):
            if not mapping.get(label):
                out.append(NotPlaying(camera, label, track.id, rec["not_playing"][label]))
        for label, why in (rec.get("check") or {}).items():
            sid_ = by_pid.get(mapping.get(label))
            if sid_ is not None:
                check.setdefault(sid_, why)
    return ParticipationView(out, check)


def include_person(repo: Repo, settings, submission_id: int, camera: str,
                   label: str) -> list[int]:
    """「この人も評価する」: a tracked person nobody evaluates (left out by the participation
    rule, or on screen too briefly: short_people) is evaluated after all. The label is
    remembered for the camera's current track run and the people are chosen again (reselect:
    the behavior is recorded again). ValueError when the label is not an unevaluated person of
    that camera's latest track run; BusyError while a blocking job runs."""
    if repo.has_blocking_jobs(submission_id):
        raise review.BusyError(review._busy_message("映っている人の選び直し"))
    track = repo.latest_run(submission_id, Stage.track, camera)
    with repo._tx():
        meta = dict(repo.get_submission(submission_id)["meta"] or {})
        records = dict(meta.get(PARTICIPATION) or {})
        rec = dict(records.get(camera) or {})
        if not usable(track) or label not in ((track.output or {}).get("labels") or []):
            raise ValueError("この人はもう評価に入っています。")
        current = rec.get("track") == track.id
        left_out = current and label in (rec.get("not_playing") or {})
        mapping = current_mappings(repo, submission_id, camera).get(track.id) or {}
        if not left_out and mapping.get(label):
            raise ValueError("この人はもう評価に入っています。")
        if not current:
            rec = {"track": track.id, "not_playing": {}, "check": {}, "included": []}
        rec["included"] = [*(rec.get("included") or []), label]
        rec["not_playing"] = {k: v for k, v in (rec.get("not_playing") or {}).items()
                              if k != label}
        records[camera] = rec
        meta[PARTICIPATION] = records
        repo.update_submission_meta(submission_id, meta)
    repo.audit("include_person", f"submission:{submission_id}", {"camera": camera})
    return reselect(repo, settings, submission_id)


@dataclass(frozen=True)
class ShortPerson:
    """A tracked person nobody evaluates although the participation rule did not leave them
    out: on screen too briefly (or beyond the cap). `seconds`: how long they were boxed."""
    camera: str
    label: str
    track_run_id: int
    seconds: float | None


SHORT_NOTE_MIN = 3   # this many unevaluated people (or half of those found) are worth a sentence


@dataclass(frozen=True)
class ShortView:
    people: list[ShortPerson]     # camera then longest-seen first
    found: int                    # people the tracker found (all cameras)
    evaluated: int                # of them, the people some mapping evaluates

    @property
    def much(self) -> bool:
        """So many people are not evaluated that the coach should be told why."""
        n = len(self.people)
        return n >= SHORT_NOTE_MIN or (n > 0 and 2 * n >= self.found)


def short_people(repo: Repo, submission_id: int) -> ShortView:
    """The people of the cameras' latest track runs that the automatic choice did not
    evaluate and the participation rule did not leave out (those are 試合に参加していない人).
    A camera whose mapping the coach made (or a rule mapping of listed students) has nobody
    here: every label left outside was decided."""
    submission = repo.get_submission(submission_id)
    records = submission["meta"].get(PARTICIPATION) or {}
    people, found, evaluated = [], 0, 0
    for camera in _cameras(repo, submission_id, submission):
        track = repo.latest_run(submission_id, Stage.track, camera)
        identity = repo.latest_run(submission_id, Stage.identity, camera)
        if not usable(track) or not isinstance(track.output, dict) or not usable(identity) \
                or track.id not in mapped_track_ids(identity):
            continue
        labels = list(track.output.get("labels") or [])
        found += len(labels)
        mapping = current_mappings(repo, submission_id, camera).get(track.id) or {}
        evaluated += sum(1 for lb in labels if mapping.get(lb))
        if identity.source != AUTO_SOURCE or track.output.get("default_source") == "rule":
            continue
        rec = records.get(camera) or {}
        left = set(rec.get("not_playing") or {}) if rec.get("track") == track.id else set()
        todo = [lb for lb in labels if not mapping.get(lb) and lb not in left]
        if not todo:
            continue
        seconds = _boxed_seconds(track, todo)
        people += sorted((ShortPerson(camera, lb, track.id, seconds.get(lb)) for lb in todo),
                         key=lambda p: (-(p.seconds or 0.0), len(p.label), p.label))
    return ShortView(people, found, evaluated)


def _boxed_seconds(track: StageRun, labels: list[str]) -> dict[str, float]:
    """{label: seconds boxed}: from the track's quality (frames), else its boxes; labels
    whose length is unknown are left out."""
    out = track.output or {}
    meta = track_files.track_meta(out.get("boxes"))
    if meta is None or not meta.fps:
        return {}
    quality = out.get("quality") if isinstance(out.get("quality"), dict) else {}

    def counted(lb: str):
        n = (quality.get(lb) or {}).get("frames")
        return n if isinstance(n, (int, float)) and not isinstance(n, bool) else None

    frames = {lb: counted(lb) for lb in labels}
    if any(n is None for n in frames.values()) and out.get("boxes"):
        stats = box_stats(out["boxes"], [lb for lb, n in frames.items() if n is None])
        frames = {lb: (n if n is not None else (stats[lb].frames if lb in stats else None))
                  for lb, n in frames.items()}
    return {lb: round(float(n) / meta.fps, 1) for lb, n in frames.items() if n is not None}


def rerun_behavior(repo: Repo, submission_id: int) -> list[int]:
    """「もう一度解析する」: behavior again for every registered camera (contribution and
    feedback follow). BusyError (BUSY_RERUN) while a blocking job runs. Returns the job ids
    queued."""
    submission = repo.get_submission(submission_id)
    if submission_engine(submission) != "pfb3x3":
        raise review._invalid("このグループはトラッキング方式ではありません。")
    cameras = list(submission["meta"].get("cameras") or {})
    with repo._tx():
        if repo.has_blocking_jobs(submission_id):
            raise review.BusyError(BUSY_RERUN)
        ids = [enqueue_once(repo, submission_id, Stage.behavior, cam) for cam in cameras]
    ids = [i for i in ids if i is not None]
    repo.audit("rerun_behavior", f"submission:{submission_id}", {"cameras": cameras,
                                                                 "job_ids": ids})
    return ids


def unused_roster(repo: Repo, submission_id: int) -> list[dict]:
    """The roster subjects that are excluded and that no camera's latest mapping uses (on a
    roster work without rule defaults, every roster subject until the coach names a person)."""
    submission = repo.get_submission(submission_id)
    mapped = _mapped_everywhere(repo, submission_id, submission)
    return [s for s in review.roster_subjects(repo.list_subjects(submission_id),
                                              submission["meta"])
            if s["excluded"] and s["participant_id"] not in mapped]


def rename_person(repo: Repo, subject_id: int, name: str, *, roster: bool = False) -> None:
    """「名前をつける」: change only `display_name` (PFB texts never contain names, so
    approval and runs stay). ValueError for a blank or over-long name, review.NameTaken when
    another person of the submission has it (compared as review.name_key). Allowed while
    jobs run.

    `roster`: the name was picked from the roster. A roster subject of that name that is
    excluded and mapped by no camera then does not count: it is deleted when it has no
    runs (in the same transaction), else kept as it is."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("名前を入れてください。")
    if len(name) > review.MAX_NAME:
        raise ValueError(f"名前は{review.MAX_NAME}文字までにしてください。")
    key = review.name_key(name)
    with repo._tx():
        subject = repo.get_subject(subject_id)
        sid = subject["submission_id"]
        others = [s for s in repo.list_subjects(sid)
                  if s["id"] != subject_id and review.name_key(s["display_name"]) == key]
        if roster and others:
            idle = {s["id"] for s in unused_roster(repo, sid)}
            spare = [s for s in others if s["id"] in idle]
            repo.delete_subjects(sid, [s["participant_id"] for s in spare
                                       if not review._has_runs(repo, s["id"])])
            others = [s for s in others if s["id"] not in idle]
        if others:
            raise review.NameTaken(name)
        repo.rename_subject(subject_id, name)
        if subject.get("participant_id"):   # the coach's name: number_names never changes it
            meta = dict(repo.get_submission(sid)["meta"] or {})
            meta[NAME_SOURCE] = {**(meta.get(NAME_SOURCE) or {}),
                                 subject["participant_id"]: "coach"}
            repo.update_submission_meta(sid, meta)
    repo.audit("rename_person", f"subject:{subject_id}", {"roster": roster})


# --- names from read numbers ---

def _vehicle(repo: Repo, submission: dict) -> bool:
    sport = repo.get_assignment(submission["assignment_id"]).get("sport") or ""
    return sport == "motor_racing" or submission["meta"].get("target") == "vehicle"


def number_name(number: str, vehicle: bool, team: str | None = None,
                camera: int | None = None) -> str:
    """「#10」 (「10号車」 for a vehicle); with a team (T1 / T2) 「#10（チーム1）」, with a camera
    number 「#10（カメラ2）」, with both 「#10（チーム1・カメラ2）」."""
    base = f"{number}号車" if vehicle else f"#{number}"
    notes = ([TEAM_WORDS[team]] if team in TEAM_WORDS else []) \
        + ([f"カメラ{camera}"] if camera else [])
    return f"{base}（{'・'.join(notes)}）" if notes else base


DOMINANT_NAME_SHARE = 0.5   # number timeline: the person wears the number this much of the time


def _read_number(read) -> str | None:
    """The engine's number read when MIN_FRAMES_AGREE frames agreed on it, as the digits it
    gave ("00" stays "00": it is not "0"); else None. With a number timeline (the read has
    segments) the dominant number decides instead, when it is worn at least
    DOMINANT_NAME_SHARE of the numbered time (a label whose number keeps changing is two or
    more players: no name)."""
    if not isinstance(read, dict):
        return None
    if read.get("segments"):
        share = read.get("dominant_share")
        dom = read.get("dominant")
        if dom is None or not isinstance(share, (int, float)) or share < DOMINANT_NAME_SHARE:
            return None
        number = unicodedata.normalize("NFKC", str(dom)).strip()
        return number if _DIGITS.fullmatch(number) else None
    agree = read.get("frames_agree")
    if isinstance(agree, bool) or not isinstance(agree, (int, float)) \
            or agree < MIN_FRAMES_AGREE or read.get("number") is None:
        return None
    number = unicodedata.normalize("NFKC", str(read["number"])).strip()
    return number if _DIGITS.fullmatch(number) else None


def _number_run(repo: Repo, submission_id: int, camera: str, track: StageRun):
    """The newest names run made for this track run whose numbers cue ran."""
    for run in review._track_names_runs(repo, submission_id, camera, track):
        out = run.output or {}
        if ((out.get("status") or {}).get("number") or {}).get("status") == "ok" \
                and isinstance((out.get("cues") or {}).get("number"), dict):
            return run
    return None


def _camera_reads(repo: Repo, submission_id: int,
                  camera: str) -> list[tuple[str, str, str | None]]:
    """[(participant_id, number, team or None)] of the camera's mapped people whose number was
    read, from its latest track run, the mapping made for it and the newest names run with
    numbers. The team is given only when the same number was read on both teams of this
    camera: the team clusters (T1 / T2) are made per camera and per names run, so チーム1 of one
    camera is not チーム1 of another and teams are never compared across cameras."""
    track = repo.latest_run(submission_id, Stage.track, camera)
    identity = repo.latest_run(submission_id, Stage.identity, camera)
    if not usable(track) or not usable(identity) or track.id not in mapped_track_ids(identity):
        return []
    run = _number_run(repo, submission_id, camera, track)
    if run is None:
        return []
    try:
        mapping = label_mapping(identity.output)
    except (KeyError, TypeError):
        return []
    teams = {str(k): str(v) for k, v in ((run.output or {}).get("teams") or {}).items()}
    numbers = {label: n for label, read in run.output["cues"]["number"].items()
               if (n := _read_number(read)) is not None}
    sides: dict[str, set[str]] = {}
    for label, n in numbers.items():
        if teams.get(label) in TEAM_WORDS:
            sides.setdefault(n, set()).add(teams[label])
    return [(mapping[label], n, teams.get(label) if len(sides.get(n, ())) > 1 else None)
            for label, n in numbers.items() if mapping.get(label)]


def _wanted_names(repo: Repo, submission_id: int, submission: dict, vehicle: bool,
                  eligible) -> dict[str, tuple[str, str | None]]:
    """{participant_id: (number name, fallback or None)} of the `eligible` people over every
    camera (camera k = its place in _cameras, from 1, as 「カメラk」 on the screens).

    1. Per camera, each read person gets a base name (「#10」, or 「#10（チーム1）」 when both
       of that camera's teams have a 10). People of one camera with the same base name cannot
       be told apart: none of them gets a name (they keep 選手n).
    2. A person mapped on several cameras gets a name only when every camera gave the same
       base name; its camera is the lowest of them.
    3. Eligible people of different cameras with the same base name are not merged (the same
       player seen by two cameras is two people): the lowest camera keeps the base name,
       camera k gets 「#10（カメラk）」 / 「#10（チーム2・カメラk）」.
    4. The fallback: the plain name's taker on camera k > 1 falls back to its own camera's
       name when someone who is not renamed keeps the plain one (a name the coach typed, an
       earlier number name of camera 1 no longer read); on camera 1 there is none (選手n)."""
    bases: dict[str, set[tuple]] = {}    # pid -> {(number, team)}
    first: dict[str, int] = {}          # pid -> its lowest camera with a read
    unclear: set[str] = set()
    for k, camera in enumerate(_cameras(repo, submission_id, submission), 1):
        reads = _camera_reads(repo, submission_id, camera)
        holders: dict[str, set[str]] = {}
        for pid, number, team in reads:
            holders.setdefault(review.name_key(number_name(number, vehicle, team)),
                               set()).add(pid)
        for pid, number, team in reads:
            if len(holders[review.name_key(number_name(number, vehicle, team))]) > 1:
                unclear.add(pid)
            bases.setdefault(pid, set()).add((number, team))
            first.setdefault(pid, k)
    chosen = {pid: next(iter(b)) for pid, b in bases.items()
              if pid in eligible and pid not in unclear and len(b) == 1}
    groups: dict[str, list[str]] = {}
    for pid, (number, team) in chosen.items():
        groups.setdefault(review.name_key(number_name(number, vehicle, team)), []).append(pid)
    wanted: dict[str, tuple[str, str | None]] = {}
    for pids in groups.values():
        pids.sort(key=lambda p: (first[p], p))
        for i, pid in enumerate(pids):
            number, team = chosen[pid]
            own = number_name(number, vehicle, team, first[pid])
            wanted[pid] = ((number_name(number, vehicle, team), own if first[pid] > 1 else None)
                           if i == 0 else (own, None))
    return wanted


def _default_name(planner: review.NamePlanner, name: str) -> bool:
    return any(re.fullmatch(re.escape(planner._make(kind, "")) + r"\d+", name or "")
               for kind in ("default", "numberless"))


def _number_named(name: str) -> bool:
    """True for a name number_names gives (「#10」, 「10号車」, with an optional bracket)."""
    return _NUMBER_NAME.fullmatch(name or "") is not None


def number_names(repo: Repo, submission_id: int) -> int:
    """Name the automatic people (「選手n」) of a sports work by the number read on them:
    「#10」, 「10号車」 for vehicles, 「#10（チーム1）」 when both teams of a camera have a 10,
    「#10（カメラ2）」 when another camera's person already is 「#10」 (_wanted_names). Only
    people made automatically (kind "auto") are renamed, and only while their name is the
    default one, or an earlier number name (NAME_SOURCE "number" and still of that form): a
    name the coach gave (NAME_SOURCE "coach", or any other name) is never changed. A name
    someone else keeps is not given; people who swap or rotate their numbers are renamed
    together. Display names only: the roster mapping is not touched. Returns how many were
    renamed."""
    renamed = 0
    with repo._tx():
        submission = repo.get_submission(submission_id)
        if review._domain(repo, submission) != "sports":
            return 0
        meta = dict(submission["meta"] or {})
        subjects = repo.list_subjects(submission_id)
        added, source = review.added_people(meta), dict(meta.get(NAME_SOURCE) or {})
        planner = review.name_planner("sports", subjects, meta)
        eligible = {s["participant_id"]: s for s in subjects
                    if added.get(s["participant_id"]) == AUTO_KIND
                    and source.get(s["participant_id"]) != "coach"
                    and ((source.get(s["participant_id"]) == "number"
                          and _number_named(s["display_name"]))
                         or _default_name(planner, s["display_name"]))}
        if not eligible:
            return 0
        wanted = _wanted_names(repo, submission_id, submission, _vehicle(repo, submission),
                               eligible)
        key = review.name_key
        current = {s["participant_id"]: s["display_name"] for s in subjects}
        holders: dict[str, set] = {}
        for s in subjects:
            holders.setdefault(key(s["display_name"]), set()).add(s["participant_id"])
        # the plain name kept by someone who is not moving: that person's camera-named
        # fallback (「#10（カメラ2）」); a fallback that is the current name makes it stay too
        names = {pid: primary for pid, (primary, _) in wanted.items()}
        changed = True
        while changed:
            changed = False
            staying = {pid for pid in current if pid not in names or current[pid] == names[pid]}
            for pid, (primary, fallback) in wanted.items():
                if fallback and names[pid] == primary \
                        and (holders.get(key(primary), set()) & staying) - {pid}:
                    names[pid], changed = fallback, True
        todo = {pid: name for pid, name in names.items() if current[pid] != name}
        wants = Counter(key(name) for name in todo.values())
        # a wanted name kept by someone outside todo blocks its taker; so, in turn, does a
        # wanted name that is the current name of a blocked person (swaps and rotations of
        # people who all move are renamed together)
        blocked = {pid for pid, name in todo.items()
                   if wants[key(name)] > 1 or holders.get(key(name), set()) - set(todo)}
        grown = True
        while grown:
            kept = {key(current[pid]) for pid in blocked}
            more = {pid for pid, name in todo.items() if pid not in blocked and key(name) in kept}
            blocked |= more
            grown = bool(more)
        for pid, name in sorted(todo.items()):
            if pid in blocked:
                continue
            repo.rename_subject(eligible[pid]["id"], name)
            source[pid] = "number"
            renamed += 1
        if renamed:
            meta[NAME_SOURCE] = source
            repo.update_submission_meta(submission_id, meta)
    if renamed:
        repo.audit("number_names", f"submission:{submission_id}", {"people": renamed})
    return renamed


def name_notes(repo: Repo, submission_id: int) -> dict[int, str]:
    """{subject_id: 「背番号から」 (「番号から」 for vehicles)} of the people named by number."""
    submission = repo.get_submission(submission_id)
    source = submission["meta"].get(NAME_SOURCE) or {}
    note = NUMBER_NOTE["vehicle" if _vehicle(repo, submission) else "person"]
    return {s["id"]: note for s in repo.list_subjects(submission_id)
            if source.get(s["participant_id"]) == "number"}


def set_person_excluded(repo: Repo, subject_id: int, excluded: bool) -> list[int]:
    """「この人は評価しない」 (excluded) / 「評価に戻す」. Excluding cancels the person's queued
    feedback job (BusyError while it is running); restoring queues the feedback job when the
    person has no usable one, nothing blocks and every camera's contribution is current.
    Returns the job ids queued."""
    subject = repo.get_subject(subject_id)
    sid = subject["submission_id"]
    if excluded:
        with repo._tx():
            if repo.has_active_job(sid, Stage.pfb, None, subject_id) \
                    and not repo.cancel_pending_jobs(sid, Stage.pfb, None, subject_id):
                raise review.BusyError(BUSY)
            repo.set_excluded(subject_id, True)
        repo.audit("set_person_excluded", f"subject:{subject_id}", {"excluded": True})
        return []
    ids: list[int] = []
    with repo._tx():
        repo.set_excluded(subject_id, False)
        submission = repo.get_submission(sid)
        cameras = list(submission["meta"].get("cameras") or {})
        pfb = repo.latest_run(sid, Stage.pfb, subject_id=subject_id)
        ready = bool(cameras) and not usable(pfb) and not repo.has_blocking_jobs(sid) and all(
            usable(run) and not repo.is_stale(run)
            for run in (repo.latest_run(sid, Stage.contribution, cam) for cam in cameras))
        if ready and is_evaluated(repo.get_subject(subject_id)):
            job = enqueue_once(repo, sid, Stage.pfb, None, subject_id)
            ids = [job] if job is not None else []
    repo.audit("set_person_excluded", f"subject:{subject_id}",
               {"excluded": False, "job_ids": ids})
    return ids


def _sheet_crop(settings, sub: dict, camera: str, label: str, track: StageRun) -> Path | None:
    """The person's box cut from the track sheet, cached once at
    `<submission dir>/people/<camera>_<label>_t<track run id>.jpg` (temp file + replace); None when the sheet or
    its geometry is missing or unreadable."""
    target = files.submission_dir(settings, sub["assignment_id"], sub["id"]) / "people" \
        / f"{camera}_{label}_t{track.id}.jpg"
    if target.is_file():
        return target
    out = track.output or {}
    try:
        x, y, w, h = (int(v) for v in (out.get("sheet_geometry") or {})[label])
        with Image.open(out["sheet"]) as sheet:
            box = (max(x, 0), max(y, 0), min(x + w, sheet.width), min(y + h, sheet.height))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            crop = sheet.convert("RGB").crop(box)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.stem}.{os.getpid()}.tmp.jpg")
        try:
            crop.save(tmp, "JPEG")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        return target
    except (KeyError, TypeError, ValueError, OSError, Image.DecompressionBombError):
        return None


def person_crops(repo: Repo, settings, submission_id: int) -> dict[int, Path]:
    """{subject_id: image}: the names run crop of the first camera that maps the subject, else
    a crop of that camera's track sheet (the people sheet shows only who was on screen at its
    frame), else a picture cut from the middle of the person's longest visible stretch
    (_stretch_crop); subjects with none of these are left out."""
    submission = repo.get_submission(submission_id)
    by_pid = {s["participant_id"]: s["id"] for s in repo.list_subjects(submission_id)
              if s.get("participant_id") and not s.get("erased_at")}
    found: dict[int, Path] = {}
    for camera in _cameras(repo, submission_id, submission):
        track = repo.latest_run(submission_id, Stage.track, camera)
        identity = repo.latest_run(submission_id, Stage.identity, camera)
        if not usable(track) or not usable(identity) or track.id not in mapped_track_ids(identity):
            continue
        try:
            mapping = label_mapping(identity.output)
        except (KeyError, TypeError):
            continue
        wanted = {lb: by_pid[pid] for lb, pid in mapping.items()
                  if pid in by_pid and by_pid[pid] not in found}
        if not wanted:
            continue
        crops = review.names_view(repo, submission_id, camera, track, fuse=False).crops
        reads = identity_check._names_reads(repo, submission_id, camera, track.id) or {}
        for label, subject_id in wanted.items():
            path = crops.get(label)
            if path is None:
                path = (_own_number_crop(settings, submission, camera, label, track,
                                         reads.get(label))
                        or label_crop(settings, submission, camera, label, track))
            if path is not None:
                found[subject_id] = path
    return found


def label_crop(settings, sub: dict, camera: str, label: str, track: StageRun) -> Path | None:
    """The picture of a tracked person: their box on the people sheet, else a picture from
    their longest stretch on screen; None when neither can be made."""
    return (_sheet_crop(settings, sub, camera, label, track)
            or _stretch_crop(settings, sub, camera, label, track))


def _own_number_crop(settings, sub: dict, camera: str, label: str, track: StageRun,
                     read) -> Path | None:
    """When the number timeline reads another number than the person's at the people
    sheet's frame (the sheet shows another player in the person's box): the person's picture
    from the middle of their longest stretch of their own number (export.pictures, cached
    under people/own/). None when the sheet frame is fine or nothing better can be made."""
    from pfb_app.export import pictures   # noqa: PLC0415 - cv2, only when needed
    out = track.output or {}
    meta = track_files.track_meta(out.get("boxes"))
    frame = out.get("sheet_frame")
    if not isinstance(read, dict) or meta is None or not meta.fps or not isinstance(frame, int):
        return None
    reach = max(float(read.get("timeline_s") or 0.0) * meta.fps * identity_check.REACH_STEPS,
                meta.fps)
    if not identity_check.foreign_at(read, frame, reach=reach):
        return None
    own = [seg for seg in read.get("segments") or []
           if len(seg) >= 3 and str(seg[2]) == str(read.get("dominant"))]
    boxes = track_files.load_boxes(out.get("boxes"))
    video = (sub["meta"].get("cameras") or {}).get(camera)
    if not own or boxes is None or not video or not Path(video).is_file():
        return None
    first, last = max(((int(a[0]), int(a[1])) for a in own), key=lambda ab: ab[1] - ab[0])
    hit = boxes.nearest([label], (first + last) // 2, tol=max(1, (last - first) // 2))
    if hit is None:
        return None
    req = pictures.PictureRequest(Path(video), hit[0], hit[1])
    target = files.submission_dir(settings, sub["assignment_id"], sub["id"]) / "people" / "own"
    return pictures.make_pictures([req], target).get(req)


STRETCH_GAP_S = 1.0   # a pause in a person's boxes longer than this ends their stretch


def _longest_stretch(frames, fps: float) -> tuple[int, int] | None:
    """(first, last) frame of the longest run of `frames` (sorted ints) whose gaps are at most
    STRETCH_GAP_S; the earliest on a tie; None without frames."""
    if len(frames) == 0:
        return None
    limit = max(1, round(STRETCH_GAP_S * fps))
    best = (0, 0)
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or int(frames[i]) - int(frames[i - 1]) > limit:
            if int(frames[i - 1]) - int(frames[start]) > int(frames[best[1]]) - int(frames[best[0]]):
                best = (start, i - 1)
            start = i
    return int(frames[best[0]]), int(frames[best[1]])


def _stretch_crop(settings, sub: dict, camera: str, label: str, track: StageRun) -> Path | None:
    """A person the people sheet does not show (it is made at one frame, where not everybody
    is on screen): their picture from the middle of their longest stretch on screen (the box
    of the track, cut from the camera's video; export.pictures, cached under people/own/).
    None when the boxes or the video are missing or the frame cannot be read."""
    from pfb_app.export import pictures   # noqa: PLC0415 - cv2, only when needed
    out = track.output or {}
    boxes = track_files.load_boxes(out.get("boxes"))
    video = (sub["meta"].get("cameras") or {}).get(camera)
    if boxes is None or not video or not Path(video).is_file():
        return None
    meta = track_files.track_meta(out.get("boxes"))
    frames, _ = boxes.boxes(label)
    stretch = _longest_stretch(frames, meta.fps if meta and meta.fps else 30.0)
    if stretch is None:
        return None
    hit = boxes.nearest([label], (stretch[0] + stretch[1]) // 2,
                        tol=max(1, (stretch[1] - stretch[0]) // 2))
    if hit is None:
        return None
    req = pictures.PictureRequest(Path(video), hit[0], hit[1])
    target = files.submission_dir(settings, sub["assignment_id"], sub["id"]) / "people" / "own"
    return pictures.make_pictures([req], target).get(req)
