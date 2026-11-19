"""Teacher review services: edit stage outputs, regenerate, confirm identity, approve, write by hand.

Identity confirmation is reset in Repo.save_stage_run (every new identity run, whether a model
regeneration or a teacher edit, passes through it), so nothing here has to remember to do it.
"""
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError

from pfb_app.domain import profiles
from pfb_app.domain.models import (BehaviorRunOutput, ContributionOutput, EvidenceItem,
                                   IdentityOutput, Kind, PfbOutput, Stage, StageRun,
                                   SubjectStatus)
from pfb_app.domain.vocab import default_person_name, numberless_name, opponent_name
from pfb_app.pipeline import checks, name_fusion, naming
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.pfb3x3_bridge import clock_is_known, english_event_ids
from pfb_app.pipeline.runner import enqueue_once, submission_engine
from pfb_app.pipeline.tracked_identity import (OUTSIDE, RETRACKED, TrackedIdentityOutput,
                                               _label_key, mapping_errors,
                                               mapping_is_current, not_current_message,
                                               roster_reset_message, roster_was_reset,
                                               student_names,
                                               unmapped_students)
from pfb_app.preprocess.frames import Frame
from pfb_app.storage import files
from pfb_app.storage.db import NON_BLOCKING_STAGES, Repo

log = logging.getLogger(__name__)

__all__ = ["AUTO", "ApprovalError", "BusyError", "EvidenceFrame", "NamesView", "OPP",
           "ValidationError", "add_named_subject", "approve", "approved_automatically",
           "auto_approve", "bulk_approve", "confirm_identity",
           "edit_stage_output", "frames_for", "is_stale_chain", "manual_pfb_warnings",
           "materialise_choices", "names_view", "plan_default_names", "regenerate_from",
           "save_tracked_identity", "set_auto_approve", "set_evaluate_opponents",
           "set_team_choice",
           "subject_status", "unapprove", "write_manual_pfb"]

MANUAL_WARNING = "手書き"
TRACK_REDO = "トラッキング方式では、人物の追跡からやり直してください。"
CHECK_DONE = "confirmed"      # submissions.meta["check"] once the coach confirmed the content check
CHECK_FIRST = "動画の内容を決定してから、もう一度お試しください。"
_VIDEO_ONLY = {Stage.transcribe, Stage.track, Stage.identity, Stage.behavior}
_PER_CAMERA = {Stage.prepare, Stage.track, Stage.identity, Stage.behavior, Stage.contribution}


class BusyError(Exception):
    """A job is still queued or running for the submission."""


class ApprovalError(Exception):
    """The requested confirmation or approval is not allowed yet."""


def _invalid(message: str) -> ValidationError:
    """A pydantic ValidationError carrying a Japanese message."""
    return ValidationError.from_exception_data(
        "編集内容", [{"type": "value_error", "loc": (), "input": None,
                     "ctx": {"error": ValueError(message)}}])


def _busy_message(what: str) -> str:
    return f"実行中または待機中のジョブがあるため{what}できません。完了するまでお待ちください。"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StoredEvidence(_Strict):
    items: list[EvidenceItem]


def _parse(model: type[BaseModel], data):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        places = sorted({".".join(str(p) for p in err["loc"]) or "全体" for err in e.errors()})
        raise _invalid("出力の形式が正しくありません: " + ", ".join(places)) from None


def _duplicates(ids: list[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def _raise_if(errors: list[str]) -> None:
    if errors:
        raise _invalid("\n".join(errors))


# --- per-stage validation of an edited output; each returns (output, warnings) ---

def _upstream(repo: Repo, run: StageRun, stage: Stage) -> list[StageRun]:
    ups = [repo.get_run(i) for i in run.upstream_run_ids]
    return [u for u in ups if u is not None and u.stage == stage]


def _check_identity(repo, run, data):
    submission = repo.get_submission(run.submission_id)
    if submission_engine(submission) == "pfb3x3":
        out = _parse(TrackedIdentityOutput, data)
        if out.track_run_id not in run.upstream_run_ids:
            raise _invalid(RETRACKED)
        return _checked_mapping(repo, run.submission_id, run.camera, out)
    out = _parse(IdentityOutput, data)
    _raise_if([f"参加者IDが重複しています: {i}"
               for i in _duplicates([p.participant_id for p in out.participants])])
    return out.model_dump(mode="json"), []


# tracking engine: per-event_id side data of a behavior run, with its name for messages
_BEHAVIOR_MAPS = {"frames": "フレーム画像", "source_text": "元の記録"}


def _check_behavior(repo, run, data):
    """Tracking-engine runs also hold `frames` (the overlays per event_id), `source_text`
    (the engine's English) and `provenance`. An edit always keeps the run's provenance (a
    supplied one is ignored); one that leaves frames or source_text out (the review table
    sends only the events) keeps the run's entries for the events that are still there. Maps given
    explicitly must name existing events, and frame paths must lie inside the submission's
    directory."""
    explicit = {k for k in _BEHAVIOR_MAPS if not isinstance(data, dict) or k in data}
    kept = {k: (run.output or {}).get(k) for k in _BEHAVIOR_MAPS if k not in explicit}
    if isinstance(data, dict):   # what made the run stays; an edit cannot rewrite it
        data = {k: v for k, v in data.items() if k != "provenance"}
        kept["provenance"] = (run.output or {}).get("provenance")
        data = {**data, **{k: v for k, v in kept.items() if v is not None}}
    out = _parse(BehaviorRunOutput, data)
    ids = [e.event_id for e in out.events]
    errors = [f"行動イベントのIDが重複しています: {i}" for i in _duplicates(ids)]
    errors += [f"行動イベントのカメラが一致しません: {e.event_id}"
               for e in out.events if e.camera != run.camera]
    identity = next((u for u in _upstream(repo, run, Stage.identity) if usable(u)), None)
    if identity is not None:
        allowed = {p["participant_id"] for p in identity.output["participants"]
                   if p.get("participant_id") is not None}   # 対象外 is never a participant
        errors += checks.unknown_participant_errors(out.events, allowed)
    for key in _BEHAVIOR_MAPS:
        side = getattr(out, key)
        if side is None:
            continue
        unknown = sorted(set(side) - set(ids))
        if key in explicit:
            errors += [f"{_BEHAVIOR_MAPS[key]}の行動イベントIDがありません: {i}" for i in unknown]
        else:   # rows the teacher deleted take their entries with them
            setattr(out, key, {k: v for k, v in side.items() if k not in unknown})
    if "frames" in explicit and out.frames:
        errors += _foreign_frames(repo, run.submission_id, out.frames)
    _raise_if(errors)
    return out.stored(), []


def _foreign_frames(repo: Repo, submission_id: int, frames: dict[str, list[str]]) -> list[str]:
    """Errors for frame paths that do not resolve inside the submission's directory."""
    submission = repo.get_submission(submission_id)
    root = files.submission_dir(repo.settings, submission["assignment_id"], submission_id,
                                create=False).resolve()
    return [f"登録したファイルのフォルダの外にあるフレーム画像は指定できません: {event_id} {p}"
            for event_id, paths in frames.items() for p in paths
            if not Path(p).resolve().is_relative_to(root)]


def _check_evidence(repo, run, data):
    out = _parse(_StoredEvidence, data)
    _raise_if([f"根拠項目のIDが重複しています: {i}"
               for i in _duplicates([i.item_id for i in out.items])])
    return out.model_dump(mode="json"), []


def _check_contribution(repo, run, data):
    out = _parse(ContributionOutput, data)
    upstream = [u for u in (repo.get_run(i) for i in run.upstream_run_ids[:1]) if u is not None]
    if not upstream or not usable(upstream[0]):
        raise _invalid("元になる結果がないため、参照IDを確認できません。先に上流の段階を実行してください。")
    key = "events" if upstream[0].stage == Stage.behavior else "items"
    id_key = "event_id" if key == "events" else "item_id"
    known = {d[id_key] for d in upstream[0].output[key]}
    refs = [j.ref_id for j in out.judgments]
    errors = [f"存在しない参照IDです: {r}" for r in sorted(set(refs) - known)]
    errors += [f"判定が重複しています: {r}" for r in _duplicates(refs)]
    _raise_if(errors)
    return out.model_dump(mode="json"), []


def _checked_mapping(repo: Repo, submission_id: int, camera: str | None,
                     out: TrackedIdentityOutput) -> tuple[dict, list[str]]:
    """Validate a tracking-engine mapping against the camera's latest track run and the
    students; returns the stored form (participants in the track's label order, display
    names from the student list) and a warning naming the students left without a label."""
    track = repo.latest_run(submission_id, Stage.track, camera)
    if not usable(track):
        raise _invalid(f"人物の追跡結果がありません（{camera}）。先に人物の追跡を実行してください。")
    if out.track_run_id != track.id:
        raise _invalid(RETRACKED)
    labels = list(track.output.get("labels") or [])
    students = student_names(repo.list_subjects(submission_id))
    _raise_if(mapping_errors(out, labels, students))
    order = {label: n for n, label in enumerate(labels)}
    participants = sorted(
        (p.model_copy(update={"display_name": students[p.participant_id]
                              if p.participant_id is not None else None})
         for p in out.participants), key=lambda p: order[p.label])
    stored = out.model_copy(update={"participants": participants, "labels": labels})
    submission = repo.get_submission(submission_id)
    own = student_names(roster_subjects(repo.list_subjects(submission_id), submission["meta"]))
    missing = unmapped_students(stored, own)   # people made on the screen are not expected
    people = profiles.words(profiles.of_work(repo.get_assignment(
        submission["assignment_id"]))).people
    warnings = ([f"人物が割り当てられていない{people}がいます（行動の記録がないため根拠不足になります）: "
                 + "、".join(missing)] if missing else [])
    return stored.model_dump(mode="json"), warnings


def save_tracked_identity(repo: Repo, submission_id: int, camera: str,
                          mapping: dict[str, str | None]) -> StageRun:
    """Save the teacher's people-to-student mapping of one camera (tracking engine).

    `mapping` is {P-label: participant_id, or None for 対象外}; every label of the camera's
    latest track run must be in it. A value may also be an AUTO / OPP option: its person is
    made first (materialise_choices) and removed again when the mapping cannot be saved. The
    new identity run (source "teacher", upstream [track_run_id]) resets the identity
    confirmation, so the teacher confirms again. Afterwards the automatic people (AUTO / OPP)
    that no camera's latest mapping uses and that have no runs are removed; those with runs
    are marked excluded. The people this mapping uses are evaluated again (not excluded).
    Last, automatic people get the numbers already read on them as names (_number_names)."""
    submission = repo.get_submission(submission_id)
    if submission_engine(submission) != "pfb3x3":
        raise _invalid("このグループはトラッキング方式ではありません。")
    if camera not in (submission["meta"].get("cameras") or {}):
        raise _invalid(f"登録されていないカメラです: {camera}")
    if repo.has_blocking_jobs(submission_id):   # a names job never holds the coach up
        raise BusyError(_busy_message("対応づけを保存"))
    track = repo.latest_run(submission_id, Stage.track, camera)
    if not usable(track):
        raise _invalid(f"人物の追跡結果がありません（{camera}）。先に人物の追跡を実行してください。")
    if any(isinstance(v, str) and v.startswith(OPP) for v in mapping.values()) \
            and not submission["meta"].get(EVALUATE_OPPONENTS):
        raise _invalid(OPPONENTS_OFF)
    mapping, created = _materialise(repo, submission_id, camera, mapping)
    try:
        out = _parse(TrackedIdentityOutput, {
            "participants": [{"participant_id": pid, "label": label, "display_name": None}
                             for label, pid in mapping.items()],
            "labels": list(track.output.get("labels") or []), "track_run_id": track.id})
        output, warnings = _checked_mapping(repo, submission_id, camera, out)
        saved = repo.save_stage_run(StageRun(
            id=None, submission_id=submission_id, stage=Stage.identity, camera=camera,
            subject_id=None, version=0, source="teacher", upstream_run_ids=[track.id],
            prompt_hash=None, model_id=None, output=output, status="ok", error=None,
            warnings=warnings, duration_sec=0.0, created_at=""))
    except BaseException:
        try:   # nobody is kept for a mapping that was not saved
            _remove_added(repo, submission_id, created)
        except Exception:   # noqa: BLE001 - the save's own error is the one to report
            log.exception("could not remove the people made for an unsaved mapping: %s",
                          created)
        raise
    # the coach has decided: a names job not yet started for this camera is no longer needed
    # (unless it reads the numbers that name the automatic people: _names_for_display)
    cancelled = (0 if _names_for_display(repo, submission)
                 else repo.cancel_pending_jobs(submission_id, Stage.names, camera))
    pruned = _prune_added(repo, submission_id)
    _settle_excluded(repo, submission_id, {p["participant_id"] for p in output["participants"]
                                           if p["participant_id"]})
    repo.audit("save_tracked_identity", f"run:{saved.id}", {
        "camera": camera, "track_run_id": track.id, "version": saved.version,
        "mapping": {p["label"]: p["participant_id"] for p in output["participants"]},
        "names_jobs_cancelled": cancelled, "people_made": created, "people_removed": pruned})
    _number_names(repo, submission_id)
    return saved


def _number_names(repo: Repo, submission_id: int) -> None:
    """After a saved mapping: automatic people made on the mapping screen get the numbers
    already read on them as names (services.auto_people.number_names). Any error is logged;
    the save never fails on it."""
    from pfb_app.services import auto_people   # auto_people imports this module
    try:
        auto_people.number_names(repo, submission_id)
    except Exception:   # noqa: BLE001 - names stay as they were; the coach can name people
        log.exception("背番号から名前をつけられませんでした submission=%s", submission_id)


# --- the mapping model: proposals, teams, default names (tracking engine) ---
#
# The mapping screen pre-fills each person from the names stage's cues (fused here, on the
# CPU, each time the screen opens), the coach's team answer and default names. Nothing here
# is saved as a mapping: only save_tracked_identity (the coach's 「これで決定して次へ」) does.

AUTO = "auto:"   # + label: a new person with a default name (選手n / 参加者n / 選手（番号なし）n)
OPP = "opp:"     # + label: a new opponent (相手選手n)
ADDED = "added_people"            # meta: {participant_id: "auto" | "opp" | "named"}
# meta: {participant_id: "number" | "coach"} who named an automatic person (absent: its
# default name); services.auto_people.number_names never overrides "coach"
NAME_SOURCE = "name_source"
OWN_TEAM = "own_team"             # meta: {camera: {"track_run_id", "labels"}} (a label set)
EVALUATE_OPPONENTS = "evaluate_opponents"   # meta: bool, off by default
CLUSTERS = ("T1", "T2")           # the team cue's two teams ("other": referees, onlookers)
MAX_NAME = 40
NAME_TAKEN = "同じ名前の人がすでにいます: "


class NameTaken(ValueError):
    """add_named_subject: the work already has someone of that name (`name`, as typed)."""

    def __init__(self, name: str):
        super().__init__(NAME_TAKEN + name)
        self.name = name
OPPONENTS_OFF = "「相手チームも評価する」がオフのため、相手選手は選べません。"
_OPTION_KIND = {AUTO: "auto", OPP: "opp"}


def is_auto_option(value) -> bool:
    return isinstance(value, str) and value.startswith((AUTO, OPP))


def option_label(option: str) -> str:
    """The person label an AUTO / OPP option was made for ('auto:P3' -> 'P3')."""
    return option.split(":", 1)[1]


def option_sort_key(option: str):
    return _label_key(option_label(option)), option


def added_people(meta: dict | None) -> dict[str, str]:
    """{participant_id: "auto" | "opp" | "named"} of the people made on the mapping screen."""
    return dict((meta or {}).get(ADDED) or {})


def roster_subjects(subjects: list[dict], meta: dict | None) -> list[dict]:
    """The work's own people (a roster, or the registered participants): the subjects with a
    participant id that were not made on the mapping screen."""
    added = added_people(meta)
    return [s for s in subjects if s.get("participant_id") and s["participant_id"] not in added]


def _domain(repo: Repo, submission: dict) -> str:
    return repo.get_assignment(submission["assignment_id"]).get("domain") or "school"


@dataclass
class NamePlanner:
    """Default names for AUTO / OPP options. Each kind is numbered on from the highest number
    of that kind among the submission's names, so names continue across cameras and never
    repeat a typed or roster name. AUTO is 「選手（番号なし）n」 for a sports work with its own
    people, else 「選手n」 / 「参加者n」; OPP is 「相手選手n」."""
    domain: str
    auto_kind: str                 # "numberless" | "default"
    last: dict[str, int]           # {kind: the highest number in use}

    def _make(self, kind: str, n) -> str:
        if kind == "opp":
            return opponent_name(n)
        if kind == "numberless":
            return numberless_name(n)
        return default_person_name(self.domain, n)

    def plan(self, options) -> dict[str, tuple[str, str]]:
        """{option: (kind, name)}, numbered in the order given (repeats and other values are
        skipped)."""
        last, plan = dict(self.last), {}
        for option in options:
            if option in plan or not is_auto_option(option):
                continue
            kind = "opp" if option.startswith(OPP) else self.auto_kind
            last[kind] += 1
            plan[option] = (kind, self._make(kind, last[kind]))
        return plan

    def names_for(self, chosen: dict, extra=()) -> dict[str, str]:
        """{option: name}: the AUTO / OPP options in `chosen` numbered in label order, exactly
        as materialise_choices numbers them when these choices are saved; the `extra` options
        (not chosen) get the numbers after them."""
        picked = sorted({v for v in chosen.values() if is_auto_option(v)}, key=option_sort_key)
        rest = sorted((o for o in extra if is_auto_option(o) and o not in picked),
                      key=option_sort_key)
        return {o: name for o, (_, name) in self.plan(picked + rest).items()}


def name_planner(domain: str, subjects: list[dict], meta: dict | None) -> NamePlanner:
    taken = [str(s.get("display_name") or "").strip() for s in subjects]
    planner = NamePlanner(domain, ("numberless" if domain == "sports"
                                   and roster_subjects(subjects, meta) else "default"), {})
    for kind in ("default", "numberless", "opp"):
        pattern = re.compile("^" + re.escape(planner._make(kind, "")) + r"(\d+)$")
        planner.last[kind] = max((int(m.group(1)) for n in taken if (m := pattern.match(n))),
                                 default=0)
    return planner


def plan_default_names(repo: Repo, submission_id: int, chosen: dict,
                       extra=()) -> dict[str, str]:
    """NamePlanner.names_for for the submission as it is now."""
    submission = repo.get_submission(submission_id)
    return name_planner(_domain(repo, submission), repo.list_subjects(submission_id),
                        submission["meta"]).names_for(chosen, extra)


def _free_id(prefix: str, taken: set[str]) -> str:
    pattern = re.compile("^" + re.escape(prefix) + r"(\d+)$")
    k = max((int(m.group(1)) for t in taken if (m := pattern.match(t))), default=0) + 1
    return f"{prefix}{k}"


def _materialise(repo: Repo, submission_id: int, camera: str,
                 chosen: dict) -> tuple[dict, list[str]]:
    """(chosen with AUTO / OPP options replaced by new participant ids, the ids made)."""
    wanted = sorted({v for v in chosen.values() if is_auto_option(v)}, key=option_sort_key)
    if not wanted:
        return dict(chosen), []
    made: dict[str, str] = {}
    with repo._tx():
        submission = repo.get_submission(submission_id)
        meta = dict(submission["meta"] or {})
        subjects = repo.list_subjects(submission_id)
        plan = name_planner(_domain(repo, submission), subjects, meta).plan(wanted)
        taken = {s["participant_id"] for s in subjects if s["participant_id"]}
        added = added_people(meta)
        for option in wanted:
            kind, name = plan[option]
            prefix = _OPTION_KIND[AUTO if option.startswith(AUTO) else OPP]
            pid = _free_id(prefix, taken)
            repo.add_subject(submission_id, pid, name, None, {}, {})
            taken.add(pid)
            added[pid] = prefix
            made[option] = pid
        meta[ADDED] = added
        _forget_name_source(meta, made.values())
        repo.update_submission_meta(submission_id, meta)
    return {label: made.get(v, v) for label, v in chosen.items()}, list(made.values())


def materialise_choices(repo: Repo, submission_id: int, camera: str,
                        chosen: dict[str, str | None]) -> dict[str, str | None]:
    """The choices with every AUTO / OPP option replaced by a new person's participant id
    (one person per option, named as the form showed: in label order, on from the highest
    default number of the submission). Other values (participant ids, OUTSIDE, None) are
    returned as they are. People no mapping uses are removed when a mapping is saved."""
    mapping, made = _materialise(repo, submission_id, camera, chosen)
    if made:
        repo.audit("materialise_choices", f"submission:{submission_id}",
                   {"camera": camera, "people": made})
    return mapping


def _names_for_display(repo: Repo, submission: dict) -> bool:
    """True when the work's names jobs read numbers for display names (naming.display_numbers):
    deciding the mapping then does not cancel them."""
    return naming.display_numbers(repo.settings,
                                  repo.get_assignment(submission["assignment_id"]))


def _forget_name_source(meta: dict, pids) -> None:
    """Drop the NAME_SOURCE records of these participant ids (people removed, or new people
    that got an id an earlier person had)."""
    pids = set(pids)
    if pids & set(meta.get(NAME_SOURCE) or {}):
        meta[NAME_SOURCE] = {p: k for p, k in meta[NAME_SOURCE].items() if p not in pids}


def _has_runs(repo: Repo, subject_id: int) -> bool:
    return repo.conn.execute("SELECT 1 FROM stage_runs WHERE subject_id=? LIMIT 1",
                             (subject_id,)).fetchone() is not None


def _remove_added(repo: Repo, submission_id: int, pids: list[str]) -> None:
    if not pids:
        return
    with repo._tx():
        meta = dict(repo.get_submission(submission_id)["meta"] or {})
        repo.delete_subjects(submission_id, list(pids))
        meta[ADDED] = {p: k for p, k in added_people(meta).items() if p not in pids}
        _forget_name_source(meta, pids)
        repo.update_submission_meta(submission_id, meta)


def _prune_added(repo: Repo, submission_id: int,
                 kinds: tuple[str, ...] = ("auto", "opp")) -> list[str]:
    """Remove the people made on the mapping screen of these kinds (by default the automatic
    ones; confirm_identity also passes "named") that no camera's latest mapping uses and that
    have no runs; returns their participant ids."""
    submission = repo.get_submission(submission_id)
    added = added_people(submission["meta"])
    automatic = [p for p, kind in added.items() if kind in kinds]
    if not automatic:
        return []
    cameras = set(submission["meta"].get("cameras") or {}) | set(repo.list_cameras(submission_id))
    mapped = set()
    for cam in cameras:
        run = repo.latest_run(submission_id, Stage.identity, cam)
        if run is not None and isinstance(run.output, dict):
            mapped |= {p.get("participant_id") for p in run.output.get("participants") or []}
    ids = {s["participant_id"]: s["id"] for s in repo.list_subjects(submission_id)}
    gone = [p for p in automatic
            if p not in mapped and (p not in ids or not _has_runs(repo, ids[p]))]
    _remove_added(repo, submission_id, gone)
    return gone


def _settle_excluded(repo: Repo, submission_id: int, used: set[str]) -> None:
    """After a coach's mapping: its people are evaluated again; automatic people (kind "auto")
    that no camera's latest mapping uses and that have runs are excluded."""
    submission = repo.get_submission(submission_id)
    added = added_people(submission["meta"])
    cameras = set(submission["meta"].get("cameras") or {}) | set(repo.list_cameras(submission_id))
    mapped = set(used)
    for cam in cameras:
        run = repo.latest_run(submission_id, Stage.identity, cam)
        if run is not None and isinstance(run.output, dict):
            mapped |= {p.get("participant_id") for p in run.output.get("participants") or []}
    with repo._tx():
        for s in repo.list_subjects(submission_id):
            pid = s["participant_id"]
            if pid in used and s["excluded"]:
                repo.set_excluded(s["id"], False)
            elif (added.get(pid) == "auto" and pid not in mapped and not s["excluded"]
                  and _has_runs(repo, s["id"])):
                repo.set_excluded(s["id"], True)


def name_key(text) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).split()).casefold()


def add_named_subject(repo: Repo, submission_id: int, display_name: str) -> str:
    """「名前をつける」: a new person with the coach's name (no roster), id n{k}. ValueError
    for a blank name or one the submission already has (compared after NFKC, white space
    and case); BusyError while a job other than names is queued or running. The names stage's
    roster fingerprint does not include such people, so the proposals stay current. Named
    people no camera maps are removed at the final confirm_identity."""
    name = str(display_name or "").strip()
    if not name:
        raise ValueError("名前を入れてください。")
    if len(name) > MAX_NAME:
        raise ValueError(f"名前は{MAX_NAME}文字までにしてください。")
    if repo.has_blocking_jobs(submission_id):   # a names job never holds the coach up
        raise BusyError(_busy_message("名前をつけることが"))
    with repo._tx():
        meta = dict(repo.get_submission(submission_id)["meta"] or {})
        subjects = repo.list_subjects(submission_id)
        if any(name_key(s["display_name"]) == name_key(name) for s in subjects):
            raise NameTaken(name)
        pid = _free_id("n", {s["participant_id"] for s in subjects if s["participant_id"]})
        repo.add_subject(submission_id, pid, name, None, {}, {})
        meta[ADDED] = {**added_people(meta), pid: "named"}
        repo.update_submission_meta(submission_id, meta)
    repo.audit("add_named_subject", f"submission:{submission_id}", {"participant_id": pid})
    return pid


def _track_names_runs(repo: Repo, submission_id: int, camera: str, track: StageRun):
    """The usable, not stale names runs of the camera made for this track run, newest first
    (whatever roster they were built from)."""
    for run in reversed(repo.run_history(submission_id, Stage.names, camera)):
        if (usable(run) and (run.output or {}).get("track_run_id") == track.id
                and not repo.is_stale(run)):
            yield run


def _team_run(repo: Repo, submission_id: int, camera: str, track: StageRun,
              current: StageRun | None) -> StageRun | None:
    """The names run whose team clusters the screen shows: the current one, else the newest
    usable run made for this track run that has clusters (the clusters do not depend on the
    roster, so a roster change does not hide the team question)."""
    if current is not None and (current.output or {}).get("teams"):
        return current
    return next((r for r in _track_names_runs(repo, submission_id, camera, track)
                 if (r.output or {}).get("teams")), None)


def _current_names(repo: Repo, submission_id: int, camera: str,
                   track: StageRun) -> StageRun | None:
    """The current names run (naming.current_names_run) when it was made for this track run."""
    run = naming.current_names_run(repo, submission_id, camera)
    if run is None or (run.output or {}).get("track_run_id") != track.id:
        return None
    return run


def set_team_choice(repo: Repo, submission_id: int, camera: str, team: str | None) -> None:
    """「自分のチームはどちらですか」: the coach's own team on this camera, "T1" or "T2" of the
    team clusters on screen, or None to clear the answer. Stored as the cluster's label set for
    the camera's track run (meta own_team), since T1/T2 may swap between engine runs. Starts
    no work: the proposals are fused again when the screen opens. ValueError for a camera or a
    team the screen does not have."""
    submission = repo.get_submission(submission_id)
    if camera not in (submission["meta"].get("cameras") or {}):
        raise ValueError(f"登録されていないカメラです: {camera}")
    track = repo.latest_run(submission_id, Stage.track, camera)
    if not usable(track):
        raise ValueError(f"人物の追跡結果がありません（{camera}）。")
    labels = []
    if team is not None:
        run = _team_run(repo, submission_id, camera, track,
                        _current_names(repo, submission_id, camera, track))
        teams = (run.output or {}).get("teams") or {} if run else {}
        labels = sorted((l for l, t in teams.items() if t == team), key=_label_key)
        if team not in CLUSTERS or not labels:
            raise ValueError("選べるチームではありません。")
    with repo._tx():
        meta = dict(repo.get_submission(submission_id)["meta"] or {})
        own = dict(meta.get(OWN_TEAM) or {})
        if team is None:
            own.pop(camera, None)
        else:
            own[camera] = {"track_run_id": track.id, "labels": labels}
        meta[OWN_TEAM] = own
        repo.update_submission_meta(submission_id, meta)
    repo.audit("set_team_choice", f"submission:{submission_id}",
               {"camera": camera, "labels": labels if team is not None else None})


def set_evaluate_opponents(repo: Repo, submission_id: int, on: bool) -> None:
    """「相手チームも評価する」 (off by default) for the whole work's submission."""
    with repo._tx():
        meta = dict(repo.get_submission(submission_id)["meta"] or {})
        meta[EVALUATE_OPPONENTS] = bool(on)
        repo.update_submission_meta(submission_id, meta)
    repo.audit("set_evaluate_opponents", f"submission:{submission_id}", {"on": bool(on)})


@dataclass(frozen=True)
class _Entry:
    """A subject as name_fusion reads a roster row."""
    participant_id: str
    display_name: str
    attrs: dict


def _min_frames_agree(names_output: dict) -> int:
    """max(2, the gate's frames-agree count for numbers).

    The engine's number gate counts frames: its `threshold` is frames_agree k, the number of
    frames that must read the same number (the engine Task 9 ruling), so it is the same unit
    as name_fusion's min_frames_agree. An explicit `min_frames_agree` field of the gate, when
    present, is read first; `threshold` is the fallback."""
    gate = (names_output.get("gate") or {}).get("number") or {}
    raw = gate.get("min_frames_agree")
    if raw is None:
        raw = gate.get("threshold")
    try:
        return max(name_fusion.MIN_FRAMES_AGREE, math.ceil(float(raw)))
    except (TypeError, ValueError, OverflowError):
        return name_fusion.MIN_FRAMES_AGREE


def _fused(names_output: dict, people: list[dict], labels: list[str], teams: dict[str, str],
           own: set[str] | None, target: str) -> dict[str, name_fusion.Proposal]:
    """The cues of a names run fused into {label: Proposal}. With a team answer (`own`, a label
    set) the order and position cues see only the own team's people (and, when the roster's
    team names are matched to the clusters, only the own team's rows), and every candidate is
    restricted to the own team."""
    status, cues = names_output.get("status") or {}, names_output.get("cues") or {}

    def ok(cue: str) -> dict | None:
        raw = cues.get(cue)
        return raw if (status.get(cue) or {}).get("status") == "ok" and isinstance(raw, dict) \
            else None

    entries = [_Entry(s["participant_id"], s["display_name"], dict(s.get("roster") or {}))
               for s in people]
    if not entries or not labels:
        return {}
    rows = [e.participant_id for e in entries]
    mfa = _min_frames_agree(names_output)
    # numbers propose roster rows only when the gate enabled them (a sports work reads them
    # for display names anyway: naming.display_numbers); a run without a gate record counts
    number_gate = (names_output.get("gate") or {}).get("number") or {}
    reads = (ok("number") or {}) if number_gate.get("enabled", True) is not False else {}
    team_names = (name_fusion.match_team_names(reads, entries, teams, min_frames_agree=mfa)
                  if teams else {})
    use = [l for l in labels if own is None or l in own]
    side = entries
    if own is not None:
        own_names = {unicodedata.normalize("NFKC", team_names[teams[l]]).strip()
                     for l in own if teams.get(l) in team_names}
        if own_names:
            side = [e for e in entries
                    if unicodedata.normalize("NFKC", str(e.attrs.get("team") or "")).strip()
                    in own_names]

    def restrict(cands):
        return (cands if own is None else
                name_fusion.restrict_to_team(cands, teams, own, team_names, entries))

    cands = name_fusion.number_candidates(reads, entries, min_frames_agree=mfa) if reads else []
    if (raw := ok("position")) is not None:
        pos = {l: p for l, p in (raw.get("positions") or {}).items() if l in use}
        cands += (name_fusion.position_candidates(pos, side, raw.get("slots"), raw.get("claims"))
                  if own is None else name_fusion.position_candidates(pos, side))
    if (raw := ok("grid")) is not None:
        order = [l for l in raw.get("order") or [] if l in use]
        cands += name_fusion.order_candidates(order, side, "grid")
    for cue in ("appearance", "place"):
        if (raw := ok(cue)) is not None:
            cands += name_fusion.mapped_candidates(raw.get("names") or {}, entries, cue)
    cands = restrict(cands)
    if (raw := ok("batting")) is not None:
        anchors = name_fusion.batting_anchors(cands, use, rows)
        order = [l for l in raw.get("order") or [] if l in use]
        cands += restrict(name_fusion.order_candidates(order, side, "batting", anchors))
    return name_fusion.fuse(cands, use, rows, target=target)


@dataclass
class NamesView:
    """What the names stage offers the mapping form of one camera (see names_view)."""
    run: StageRun | None = None                    # the current names run (proposals)
    proposals: dict[str, name_fusion.Proposal] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)      # {label: 「背番号10から推定」…}
    teams: dict[str, str] = field(default_factory=dict)        # {label: T1 | T2 | other}
    team_reason: str = ""
    strips: dict[str, Path] = field(default_factory=dict)      # {T1 | T2: sample strip}
    crops: dict[str, Path] = field(default_factory=dict)       # {label: crop}
    own_team: str | None = None                    # the cluster of the coach's answer
    own_labels: set[str] | None = None             # the answer's labels (None: no answer)
    pending: bool = False                          # a names job is queued or running


def names_view(repo: Repo, submission_id: int, camera: str, track: StageRun, *,
               fuse: bool = True) -> NamesView:
    """The proposals (fused now, from the current names run made for `track`), the team
    clusters and the coach's team answer for one camera. CPU only; nothing is saved.
    `fuse=False` skips the fusion (a saved mapping is shown as saved)."""
    submission = repo.get_submission(submission_id)
    meta = submission["meta"] or {}
    current = _current_names(repo, submission_id, camera, track)
    team_run = _team_run(repo, submission_id, camera, track, current)
    out = (team_run.output or {}) if team_run else {}
    # why there are no teams (e.g. one_colour) and the person crops: from the newest run
    newest = next(_track_names_runs(repo, submission_id, camera, track), None)
    last = (newest.output or {}) if newest else {}
    teams = {str(k): str(v) for k, v in (out.get("teams") or {}).items()}
    view = NamesView(run=current, teams=teams, team_reason=str(last.get("team_reason") or ""),
                     strips={k: Path(v) for k, v in (out.get("strips") or {}).items()
                             if v and Path(v).is_file()},
                     crops={k: Path(v) for k, v in (last.get("crops") or {}).items()
                            if v and Path(v).is_file()},
                     pending=repo.has_active_job(submission_id, Stage.names, camera))
    rec = (meta.get(OWN_TEAM) or {}).get(camera)
    if isinstance(rec, dict) and rec.get("track_run_id") == track.id:
        stored = set(rec.get("labels") or [])
        overlap = {c: len(stored & {l for l, t in teams.items() if t == c}) for c in CLUSTERS}
        best = max(overlap.values(), default=0)
        leaders = [c for c, n in overlap.items() if n == best]
        if best and len(leaders) == 1:
            view.own_team = leaders[0]
            view.own_labels = {l for l, t in teams.items() if t == view.own_team}
        else:
            view.own_labels = stored
    if fuse and current is not None:
        people = roster_subjects(repo.list_subjects(submission_id), meta)
        target = ("vehicle" if (repo.get_assignment(submission["assignment_id"]).get("sport")
                                or "") == "motor_racing" else "person")
        # the 「other」 cluster (referees, onlookers) is never named: it stays 対象外
        labels = [l for l in track.output.get("labels") or [] if teams.get(l) != "other"]
        view.proposals = _fused(current.output or {}, people, labels, teams, view.own_labels,
                                target)
        view.reasons = {l: "、".join(p.reasons) for l, p in view.proposals.items()}
    return view


def _positive_refs(contributions: list[StageRun]) -> tuple[set[str], set[str]]:
    positive, every = set(), set()
    for c in contributions:
        for j in (c.output or {}).get("judgments", []):
            every.add(j["ref_id"])
            if j["is_positive_contribution"]:
                positive.add(j["ref_id"])
    return positive, every


def _check_pfb(repo, run, data):
    out = _parse(PfbOutput, data)
    contributions = [c for c in _upstream(repo, run, Stage.contribution) if usable(c)]
    positive, every = _positive_refs(contributions)
    _raise_if(checks.evidence_ref_errors(out, positive))
    warnings = _text_warnings(repo, run.submission_id, every, out.pfb) if out.pfb else []
    return out.model_dump(mode="json"), warnings


def _text_warnings(repo: Repo, submission_id: int, refs: set[str], text: str) -> list[str]:
    """checks.pfb_text_warnings with the submission's internal ids (the judged refs and the
    participant ids), its cameras and, for the tracking engine, the person labels."""
    internal = refs | {s["participant_id"] for s in repo.list_subjects(submission_id)
                       if s["participant_id"]}
    sub = repo.get_submission(submission_id)
    tracked = submission_engine(sub) == "pfb3x3"
    domain = repo.get_assignment(sub["assignment_id"])["domain"]
    return checks.pfb_text_warnings(text, internal, set(repo.list_cameras(submission_id)),
                                    tracked, clock_is_known(sub["meta"], domain))


def manual_pfb_warnings(repo: Repo, subject_id: int, text: str) -> list[str]:
    """The text warnings an edit of the subject's PFB would get (as `_check_pfb` gives them),
    against the latest contribution runs `write_manual_pfb` would cite; nothing is saved."""
    subject = repo.get_subject(subject_id)
    sid = subject["submission_id"]
    kind = _kind(repo, repo.get_submission(sid))
    cameras = repo.list_cameras(sid) if kind == Kind.video else [None]
    contributions = [c for c in (repo.latest_run(sid, Stage.contribution, cam) for cam in cameras)
                     if usable(c)]
    _, every = _positive_refs(contributions)
    return _text_warnings(repo, sid, every, (text or "").strip()) if (text or "").strip() else []


_CHECKS = {Stage.identity: _check_identity, Stage.behavior: _check_behavior,
           Stage.evidence: _check_evidence, Stage.contribution: _check_contribution,
           Stage.pfb: _check_pfb}


def edit_stage_output(repo: Repo, run_id: int, new_output: dict) -> StageRun:
    """Save the teacher's corrected output as a new version of the run's stage."""
    run = repo.get_run(run_id)
    if run is None:
        raise _invalid(f"実行結果が見つかりません（ID {run_id}）。")
    check = _CHECKS.get(run.stage)
    if check is None:
        raise _invalid("この段階の出力は編集できません。")
    if repo.has_blocking_jobs(run.submission_id):
        raise BusyError(_busy_message("編集"))
    output, warnings = check(repo, run, new_output)
    saved = repo.save_stage_run(StageRun(
        id=None, submission_id=run.submission_id, stage=run.stage, camera=run.camera,
        subject_id=run.subject_id, version=0, source="teacher",
        upstream_run_ids=list(run.upstream_run_ids), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=warnings, duration_sec=0.0,
        created_at=""))
    repo.audit("edit_stage_output", f"run:{saved.id}", {
        "base_run_id": run.id, "stage": run.stage.value, "camera": run.camera,
        "subject_id": run.subject_id, "version": saved.version})
    return saved


# --- evidence frames (review) ---

@dataclass(frozen=True)
class EvidenceFrame:
    path: Path
    caption: str


def frames_for(repo: Repo, behavior_run: StageRun, event: dict) -> list[EvidenceFrame]:
    """The images shown for one behavior event of `behavior_run` in review.

    Tracking engine: the marked, zoomed frames PFB3x3 sent to the model for the event's
    window (the run's `frames[event_id]`), captioned like the engine's "frame k/n".
    Otherwise (the paper engine, or a row the teacher added): the prepared frame nearest to
    the event's video_time and its neighbours; also when none of the overlays exist any more
    (the engine moved a stale run directory aside). Prepared frames may be missing (the
    caller checks)."""
    shots = ((behavior_run.output or {}).get("frames") or {}).get(event["event_id"]) or []
    found = [EvidenceFrame(Path(p), f"フレーム {k}/{len(shots)}")
             for k, p in enumerate(shots, 1) if Path(p).is_file()]
    if found:
        return found
    prepare = repo.latest_run(behavior_run.submission_id, Stage.prepare, event["camera"])
    if prepare is None or not prepare.output:
        return []
    frames = [Frame.from_dict(f) for f in prepare.output.get("frames", [])]
    if not frames:
        return []
    video_time = event["video_time"]
    parts = [int(x) for x in video_time.split(":")] if video_time.replace(":", "").isdigit() else []
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    nearest = min(range(len(frames)), key=lambda i: abs(frames[i].video_sec - sec))
    return [EvidenceFrame(f.path, f.video_time) for f in frames[max(nearest - 1, 0):nearest + 2]]


# --- regenerate ---

def _kind(repo: Repo, submission: dict) -> Kind:
    return Kind(repo.get_assignment(submission["assignment_id"])["kind"])


def regenerate_from(repo: Repo, submission_id: int, stage: Stage, camera: str | None = None,
                    subject_id: int | None = None) -> list[int]:
    """Enqueue this stage; downstream stages follow via enqueue_followups.

    Without `camera` a per-camera video stage runs for every registered camera; without
    `subject_id` the pfb stage runs for every subject (video: those with a participant id).
    A group held for the content check (meta["check"] not "confirmed") may only re-run detect.
    """
    stage = Stage(stage)
    submission = repo.get_submission(submission_id)
    kind = _kind(repo, submission)
    one_text = None   # one person's text: only other people's texts may be being written
    if stage == Stage.pfb and subject_id is not None:
        one_text = next((s for s in repo.list_subjects(submission_id) if s["id"] == subject_id),
                        None)
        if one_text is None:
            raise _invalid("このファイルに属さない対象者です。")
    if (busy_for_subject(repo, one_text) if one_text is not None
            else repo.has_blocking_jobs(submission_id)):
        raise BusyError(_busy_message("再生成"))
    if submission["meta"].get("check", CHECK_DONE) != CHECK_DONE and stage != Stage.detect:
        raise _invalid(CHECK_FIRST)   # held for the content check: only the check itself
    if (kind == Kind.document and stage in _VIDEO_ONLY) or (
            kind == Kind.video and stage == Stage.evidence):
        raise _invalid("この活動の種類では実行できない段階です。")
    if kind == Kind.video:
        tracked = submission_engine(submission) == "pfb3x3"
        if tracked and stage == Stage.identity:
            raise _invalid(TRACK_REDO)   # never a paper-engine VLM identity job
        if not tracked and stage == Stage.track:
            raise _invalid("論文方式のグループでは人物の追跡は使いません。")
    targets: list[tuple[str | None, int | None]]
    if kind == Kind.video and stage in _PER_CAMERA:
        registered = list(submission["meta"].get("cameras") or {})
        if camera is not None and camera not in registered:
            raise _invalid(f"登録されていないカメラです: {camera}")
        targets = [(c, None) for c in ([camera] if camera else registered)]
    elif stage == Stage.pfb:
        subjects = repo.list_subjects(submission_id)
        if subject_id is not None:
            if subject_id not in {s["id"] for s in subjects}:
                raise _invalid("このファイルに属さない対象者です。")
            targets = [(None, subject_id)]
        else:
            targets = [(None, s["id"]) for s in subjects
                       if kind == Kind.document or s["participant_id"]]
    else:
        targets = [(None, None)]
    ids = [enqueue_once(repo, submission_id, stage, c, s) for c, s in targets]
    ids = [i for i in ids if i is not None]
    repo.audit("regenerate", f"submission:{submission_id}", {
        "stage": stage.value, "camera": camera, "subject_id": subject_id, "job_ids": ids})
    return ids


# --- identity confirmation ---

class Confirmed(list):
    """confirm_identity's result: the behavior job ids it queued, and `removed_named`, how many
    people named on the mapping screen were removed because no camera used them."""

    def __init__(self, ids=(), removed_named: int = 0):
        super().__init__(ids)
        self.removed_named = removed_named


def confirm_identity(repo: Repo, submission_id: int) -> Confirmed:
    """Confirm the identities the teacher has seen; with the approval gate, start behavior.

    The tracking engine always waits for this confirmation (by the coach, or by
    services.auto_people.auto_confirm after the automatic people), so its behavior jobs are
    enqueued here whatever the assignment's gate says.
    Tracking engine: people named on the mapping screen (「名前をつける」) that no camera's
    mapping uses and that have no runs are removed first (`removed_named` of the result).
    The queued or waiting names jobs of every camera are cancelled (a running one is left),
    except on a work whose names jobs read numbers for display names (_names_for_display)."""
    submission = repo.get_submission(submission_id)
    assignment = repo.get_assignment(submission["assignment_id"])
    if Kind(assignment["kind"]) != Kind.video:
        raise ApprovalError("動画の活動ではないため、人物の確認は不要です。")
    cameras = list(submission["meta"].get("cameras") or {})
    if not cameras:
        raise ApprovalError("カメラの動画が登録されていません。")
    tracked = submission_engine(submission) == "pfb3x3"
    for cam in cameras:
        if repo.has_active_job(submission_id, Stage.identity, cam):
            raise ApprovalError(f"人物の特定が実行中または待機中です（{cam}）。完了後に確認してください。")
        if tracked and any(repo.has_active_job(submission_id, st, c) for st, c in (
                (Stage.prepare, cam), (Stage.prepare, None), (Stage.track, cam))):
            raise ApprovalError(f"人物の追跡が実行中または待機中です（{cam}）。完了後に確認してください。")
    for cam in cameras:
        identity = repo.latest_run(submission_id, Stage.identity, cam)
        if not usable(identity):
            raise ApprovalError(
                f"人物と参加者の対応づけがありません（{cam}）。人物一覧で対応づけを保存してください。"
                if tracked else
                f"人物の特定結果がありません（{cam}）。先に人物の特定を実行してください。")
        if tracked:
            if not mapping_is_current(repo, submission_id, cam):
                if roster_was_reset(submission["meta"], identity):
                    raise ApprovalError(roster_reset_message(cam))
                raise ApprovalError(not_current_message(cam))
    # people named on the screen that no camera uses are not kept (they would get no feedback)
    removed = _prune_added(repo, submission_id, ("named",)) if tracked else []
    with repo._tx(join=True):
        repo.set_identity_confirmed(submission_id, True)
        # every camera is decided: names jobs not yet started are no longer needed, unless
        # they read the numbers that name the automatic people (_names_for_display)
        cancelled = 0 if _names_for_display(repo, submission) else sum(
            repo.cancel_pending_jobs(submission_id, Stage.names, cam) for cam in cameras)
    repo.audit("confirm_identity", f"submission:{submission_id}",
               {"cameras": cameras, "named_removed": removed, "names_jobs_cancelled": cancelled})
    ids = Confirmed(removed_named=len(removed))
    if not assignment["require_identity_approval"] and not tracked:
        return ids
    for cam in cameras:
        behavior = repo.latest_run(submission_id, Stage.behavior, cam)
        if usable(behavior) and not _is_stale(repo, behavior):
            continue  # already based on the identities just confirmed
        job_id = enqueue_once(repo, submission_id, Stage.behavior, cam)
        if job_id is not None:
            ids.append(job_id)
    return ids


# --- manual PFB, approval ---

def _is_stale(repo: Repo, run: StageRun) -> bool:
    """Stale if any run in its upstream chain was superseded (Repo.is_stale looks one level)."""
    seen: set[int] = set()
    todo = [run]
    while todo:
        r = todo.pop()
        if repo.is_stale(r):
            return True
        for up_id in r.upstream_run_ids:
            if up_id not in seen:
                seen.add(up_id)
                up = repo.get_run(up_id)
                if up is not None:
                    todo.append(up)
    return False


def is_stale_chain(repo: Repo, run: StageRun) -> bool:
    """Public form of the stale check: True if any run upstream of `run` was superseded."""
    return _is_stale(repo, run)


def _latest_pfb(repo: Repo, subject: dict) -> StageRun | None:
    return repo.latest_run(subject["submission_id"], Stage.pfb, subject_id=subject["id"])


def write_manual_pfb(repo: Repo, subject_id: int, text: str) -> StageRun:
    """Store the teacher's own PFB text as the subject's newest pfb run. One write transaction
    (joined when the caller has one): the busy check and the save see the same jobs, so a
    feedback job queued for this person in between cannot slip under the hand-written text."""
    if not text or not text.strip():
        raise _invalid("PFBの本文が空です。")
    with repo._tx(join=True):
        subject = repo.get_subject(subject_id)
        sid = subject["submission_id"]
        if busy_for_subject(repo, subject):   # other people's texts may still be written
            raise BusyError(_busy_message("手書きのPFBを保存"))
        kind = _kind(repo, repo.get_submission(sid))
        cameras = repo.list_cameras(sid) if kind == Kind.video else [None]
        contributions = [repo.latest_run(sid, Stage.contribution, cam) for cam in cameras]
        out = PfbOutput(pfb_status="generated", pfb=text.strip(), evidence_refs=[])
        saved = repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subject_id,
            version=0, source="teacher", upstream_run_ids=[c.id for c in contributions if c],
            prompt_hash=None, model_id=None, output=out.model_dump(), status="ok", error=None,
            warnings=[MANUAL_WARNING], duration_sec=0.0, created_at=""))
        repo.audit("manual_pfb", f"run:{saved.id}", {"subject_id": subject_id})
    return saved


UNTRANSLATED = "翻訳未完了"


def untranslated_behavior(repo: Repo, submission_id: int) -> dict[str, int]:
    """{camera: events still in English} of the latest behavior run per camera whose
    translation did not finish (tracking engine; provenance.translated false with events).
    A run the teacher saved is never counted: the teacher's review is the human check
    (english_left() still shows what stayed English)."""
    out = {}
    for cam in repo.list_cameras(submission_id):
        run = repo.latest_run(submission_id, Stage.behavior, cam)
        prov = (run.output or {}).get("provenance") if usable(run) else None
        if not prov or prov.get("translated") is not False or run.source == "teacher":
            continue
        left = int(prov.get("events") or 0) - int(prov.get("translated_events") or 0)
        if left > 0:
            out[cam] = left
    return out


def english_left(run) -> int:
    """Events of a tracking-engine behavior run whose text is still English (0 for paper)."""
    if run is None or not (run.output or {}).get("provenance"):
        return 0
    return len(english_event_ids(run.output))


def untranslated_message(cameras: dict[str, int]) -> str:
    where = "、".join(f"{cam}（{n}件）" for cam, n in cameras.items())
    return (f"{UNTRANSLATED}: 行動の記録の一部が英語のままです: {where}。"
            "「行動の観察」をこの段階から再生成してから承認してください。")


def approve(repo: Repo, subject_id: int) -> None:
    subject = repo.get_subject(subject_id)
    submission = repo.get_submission(subject["submission_id"])
    if _kind(repo, submission) == Kind.video and not submission["identity_confirmed"]:
        raise ApprovalError("人物の特定がまだ確認されていません。先に人物を確認してください。")
    untranslated = untranslated_behavior(repo, submission["id"])
    if untranslated:
        raise ApprovalError(untranslated_message(untranslated))
    run = _latest_pfb(repo, subject)
    if run is None:
        raise ApprovalError("PFBがまだありません。")
    if run.status == "failed":
        raise ApprovalError("PFBの生成に失敗しています。再生成するか、手書きしてください。")
    if _is_stale(repo, run):
        raise ApprovalError("PFBが古くなっています。再生成するか、手書きしてください。")
    if (run.output or {}).get("pfb_status") != "generated":
        raise ApprovalError("PFBを書くための根拠が足りません。手書きのPFBを保存してください。")
    repo.set_approval(subject_id, run.id)
    repo.audit("approve", f"subject:{subject_id}", {"run_id": run.id})
    _confirm_tips(repo, subject_id)


def _confirm_tips(repo: Repo, subject_id: int) -> None:
    """The person's own current practice tips are confirmed with their approval
    (services.tips.confirm_on_approval: never raises)."""
    from pfb_app.services import tips   # noqa: PLC0415 - tips import this module
    tips.confirm_on_approval(repo, subject_id)


def unapprove(repo: Repo, subject_id: int) -> None:
    repo.get_subject(subject_id)
    repo.set_approval(subject_id, None)
    repo.audit("unapprove", f"subject:{subject_id}", {})


# --- approval without a look at each text: 全員まとめて承認, 確認せずに自動で承認する ---

AUTO_APPROVED = "auto_approved"   # audit action of the worker's approvals
BULK_APPROVE = "bulk_approve"     # audit action of one 「全員まとめて承認」 press

# why a text is left for a person (plain words, shown on step 5 as they are)
SKIP_NEWER = "文章が新しくなりました。確かめてから承認してください。"
SKIP_WARNINGS = "文章に気になる点があります。確かめてから承認してください。"
SKIP_INSUFFICIENT = "根拠が足りないため、自動の文章がありません。"
SKIP_PENDING = "まだ文章ができていません。"
SKIP_FAILED = "文章を作れませんでした。"
SKIP_STALE = "文章を作ったあとで解析が更新されました。"
SKIP_BUSY = "文章を作り直しています。"
SKIP_ANALYSIS = "解析が終わるまでお待ちください。"
SKIP_IDENTITY = "映っている人と参加者のつなぎ方がまだ確定していません。"
SKIP_ENGLISH = "行動の記録の一部が英語のままです。"
SKIP_EXCLUDED = "評価しない人になっています。"
SKIP_GONE = "見つかりませんでした。"
SKIP_SHORT = "短い文です。一人ずつ確かめて承認してください。"

# a short ready-made text a person approved on step 5 (services.short_text): never unseen
SHORT_MARK = "短い文"               # its warnings marker (and the chip)
SHORT_MODEL_ID = "short_text:v1"    # its model_id: the template that made it


def text_warnings(run: StageRun | None) -> list[str]:
    """The text warnings of a feedback run (the hand-written marker is not one)."""
    return [w for w in (run.warnings if run else []) if w != MANUAL_WARNING]


def _skip_reason(repo: Repo, subject: dict, run: StageRun | None) -> str | None:
    """None when `run` (the subject's latest feedback) may be approved without a person
    reading it: the approval gates hold, it is fresh, generated and has no text warnings.
    Otherwise the plain reason, in the order `approve` checks."""
    submission = repo.get_submission(subject["submission_id"])
    video = _kind(repo, submission) == Kind.video
    if video and subject.get("participant_id") and subject.get("excluded"):
        return SKIP_EXCLUDED
    if video and not submission["identity_confirmed"]:
        return SKIP_IDENTITY
    if untranslated_behavior(repo, submission["id"]):
        return SKIP_ENGLISH
    if run is None:
        return SKIP_PENDING
    if run.status == "failed":
        return SKIP_FAILED
    if _is_stale(repo, run):
        return SKIP_STALE
    if (run.output or {}).get("pfb_status") != "generated":
        return SKIP_INSUFFICIENT
    if run.model_id == SHORT_MODEL_ID or SHORT_MARK in run.warnings:
        return SKIP_SHORT
    if text_warnings(run):
        return SKIP_WARNINGS
    return None


def busy_for_subject(repo: Repo, subject: dict) -> bool:
    """A step runs that feeds or rewrites this subject's text: its own feedback job, or any
    blocking step before the feedback. Other people's feedback jobs only write their own
    texts, so they never count (per-person results: the coach works on the ready people
    while the rest are written)."""
    return _being_written(repo, subject) is not None


def _being_written(repo: Repo, subject: dict) -> str | None:
    """SKIP_BUSY while the subject's text is being made again, SKIP_ANALYSIS while an earlier
    step that feeds it runs; None otherwise."""
    upstream = False
    for job in repo.submission_job_rows(subject["submission_id"]):
        if job["status"] not in ("queued", "running", "waiting_gpu"):
            continue
        if job["stage"] == Stage.pfb.value and job["subject_id"] == subject["id"]:
            return SKIP_BUSY
        if job["stage"] not in (Stage.pfb.value, Stage.names.value):
            upstream = True
    return SKIP_ANALYSIS if upstream else None


def _approve_unseen(repo: Repo, subject_id: int, run_id: int | None, *,
                    auto: bool) -> tuple[bool, str | None]:
    """Approve `run_id` for the subject without a person reading it (全員まとめて承認, or the
    worker with `auto`), in one write transaction (BEGIN IMMEDIATE) that reads the subject,
    its latest text, the running steps and the gates again before it writes. Returns
    (newly approved, None), (False, None) when that very run is already approved, or
    (False, plain reason)."""
    with repo._tx(join=True):
        try:
            subject = repo.get_subject(subject_id)
        except KeyError:
            return False, SKIP_GONE
        run = _latest_pfb(repo, subject)
        current = (run is not None and subject["approval_status"] == "approved"
                   and subject["approved_run_id"] == run.id and not _is_stale(repo, run))
        if current and run.id == run_id:
            return False, None
        reason = (_being_written(repo, subject)
                  or (SKIP_NEWER if (run.id if run else None) != run_id else None)
                  or _skip_reason(repo, subject, run))
        if reason is not None:
            return False, reason
        repo.set_approval(subject_id, run.id, auto=auto)
        if auto:
            repo.audit(AUTO_APPROVED, f"subject:{subject_id}", {"run_id": run.id})
        else:
            repo.audit("approve", f"subject:{subject_id}", {"run_id": run.id, "bulk": True})
        _confirm_tips(repo, subject_id)
    return True, None


@dataclass
class BulkResult:
    approved: list[int] = field(default_factory=list)            # subject ids
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (subject id, plain reason)


def bulk_approve(repo: Repo, shown: dict[int, int | None]) -> BulkResult:
    """「全員まとめて承認」: approve every subject of `shown` ({subject id: the feedback run id
    the screen showed}) whose latest text is still that run and needs no person's look
    (`_skip_reason`); the others are returned with the plain reason. A subject whose shown
    run is already approved counts as approved (a second press reports the same)."""
    result = BulkResult()
    fresh = []
    for subject_id, run_id in shown.items():
        new, reason = _approve_unseen(repo, subject_id, run_id, auto=False)
        if reason is None:
            result.approved.append(subject_id)
            if new:
                fresh.append(subject_id)
        else:
            result.skipped.append((subject_id, reason))
    if fresh or result.skipped:
        repo.audit(BULK_APPROVE, None, {"approved": fresh,
                                        "skipped": [sid for sid, _ in result.skipped]})
    return result


def set_auto_approve(repo: Repo, assignment_id: int, on: bool) -> None:
    """「確認せずに自動で承認する」 for the whole work (off by default)."""
    repo.set_auto_approve(assignment_id, bool(on))
    repo.audit("set_auto_approve", f"assignment:{assignment_id}", {"on": bool(on)})


def auto_approve(repo: Repo, run: StageRun) -> bool:
    """The worker, after a new feedback run: approve it when the work's option is on, it is
    still the subject's latest text, nothing that feeds it is running and it needs no
    person's look (`_skip_reason`: no text warnings, enough evidence, the approval gates
    hold). True when approved; audit-logged as 'auto_approved'. A text a person approved
    meanwhile stays theirs; anything else is left for a person on step 5."""
    if run.stage != Stage.pfb or run.subject_id is None or run.status != "ok":
        return False
    submission = repo.get_submission(run.submission_id)
    if not repo.get_assignment(submission["assignment_id"]).get("auto_approve"):
        return False
    new, _ = _approve_unseen(repo, run.subject_id, run.id, auto=True)
    return new


def approved_automatically(repo: Repo, subject: dict) -> bool:
    """True when the subject's current text is approved and that approval was the worker's
    (「自動で承認」), not a person's; an approval the text has outgrown is not shown."""
    return (bool(subject.get("approved_auto"))
            and subject_status(repo, subject["id"]) == SubjectStatus.approved)


# --- status ---

def _upstream_failed(repo: Repo, submission: dict, kind: Kind) -> bool:
    sid = submission["id"]
    if kind == Kind.video:
        cameras = set(submission["meta"].get("cameras") or {}) | set(repo.list_cameras(sid))
        keys = [(s, c) for s in (Stage.prepare, Stage.track, Stage.identity, Stage.behavior,
                                 Stage.contribution) for c in sorted(cameras)]
    else:
        keys = [(s, None) for s in (Stage.prepare, Stage.evidence, Stage.contribution)]
    runs = (repo.latest_run(sid, s, c) for s, c in keys)
    return any(r is not None and r.status == "failed" for r in runs)


def _feeding_job_statuses(repo: Repo, subject: dict) -> set[str]:
    """The statuses of the active jobs that feed the subject's text: an earlier step, or its
    own text job. A names job (NON_BLOCKING_STAGES) or another person's text job does not."""
    return {j["status"] for j in repo.submission_job_rows(subject["submission_id"])
            if j["status"] in ("queued", "running", "waiting_gpu")
            and j["stage"] not in NON_BLOCKING_STAGES
            and (j["stage"] != Stage.pfb.value or j["subject_id"] in (None, subject["id"]))}


def subject_status(repo: Repo, subject_id: int) -> SubjectStatus:
    """Derive the subject's state from its submission's jobs and its latest pfb run."""
    subject = repo.get_subject(subject_id)
    sid = subject["submission_id"]
    run = _latest_pfb(repo, subject)
    stale = run is not None and _is_stale(repo, run)
    if (subject["approval_status"] == "approved" and run is not None and not stale
            and run.id == subject["approved_run_id"]):
        return SubjectStatus.approved
    active = _feeding_job_statuses(repo, subject)
    if active:
        if "running" not in active and "waiting_gpu" in active:
            return SubjectStatus.waiting_gpu
        return SubjectStatus.processing
    if run is None:
        submission = repo.get_submission(sid)
        failed = _upstream_failed(repo, submission, _kind(repo, submission))
        return SubjectStatus.failed if failed else SubjectStatus.pending
    if run.status == "failed":
        return SubjectStatus.failed
    if stale:
        return SubjectStatus.review
    if (run.output or {}).get("pfb_status") != "generated":
        return SubjectStatus.insufficient
    return SubjectStatus.review
