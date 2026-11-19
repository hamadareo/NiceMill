"""The `names` stage: run the gated roster cues on one camera's tracked people.

After a usable `track` (camera c) the runner enqueues `(names, c)` when `names_wanted` holds
(`needs_names`, plus a gate-enabled cue that has its input for c). It runs in the
pfb3x3:<key> session when a wanted cue asks the vLLM server (SERVER_CUES), else under
NAMES_CPU_KEY in a CPU-only engine session (only `team` and/or `place`: no server is started).
The stage only stores the cues' raw engine results. It never maps anybody: the proposals are
fused (Task 8's name_fusion) when the mapping screen opens, and the coach confirms them there.
The names stage never blocks the coach (Repo.has_blocking_jobs leaves it out; saving a
camera's mapping cancels its names job that has not started; a roster change does not wait
for it).

A cue runs only when the engine's evaluation enabled it (`load_cue_gate`) AND its input exists,
with one exception: on a sports work the `number` cue always runs (`display_numbers`; the
settings' `number_names`, on by default), because the coach asked for automatic display names
from the read numbers (services.auto_people.number_names: 「#10」 / 「10号車」). The gate still
decides whether those reads propose roster rows (services.review._fused reads `gate`). When
the camera's mapping (made for its track run) maps no automatic person, the numbers are not
read only for display names (`_auto_mapped`: nobody would get one).

| gate cue   | engine call  | input                                           |
|------------|--------------|-------------------------------------------------|
| team       | teams        | the work's domain is sports                     |
| number     | numbers      | a roster `number`                               |
| position   | positions    | a roster `position` and the work's sport        |
| grid       | grid         | a roster `grid`                                 |
| batting    | batting      | a roster `batting`                              |
| appearance | propose      | a roster `appearance` ({participant_id: text})  |
| place      | rule_names   | roster places for this camera (place_positions) |

Per cue the output's `status` holds {"status": ...}:
- "ok": the engine ran (its raw result is in `cues`);
- "refused": the engine ran and refused (`error`; the empty raw result is still in `cues`);
- "failed": the call raised EngineError, ValueError, TypeError or KeyError (`error`; a warning
  is added; the stage still succeeds);
- "disabled": the gate disabled the cue (`reason`, also in `disabled`; a number cue run only for
  display names is "ok" and still listed in `disabled`);
- "no_roster_column" / "not_sports" / "no_sport": its input is missing;
- "no_server": a server cue in a CPU-only names job (the gate or the roster changed after it
  was queued).

Requeues, as for behavior: EngineServerDown (also an EngineError while the server is dead) and
GpuUnavailable (exit 3) / YieldRequested go back to the queue (runner.execute_job). There is
no per-cue resume: a requeued job runs every cue again (an accepted cost; the cues are short
next to track and behavior).

A run counts (current_names_run) only while it was made from the camera's latest track run,
after the roster floor, and from the same cue-feeding subjects (`roster_fingerprint`).

Error texts and gate reasons are for logs and the admin only; never shown to the coach.
"""
import dataclasses
import hashlib
import json
import logging
import time

from pfb_app.domain.models import Job, Stage, StageRun
from pfb_app.engine import SERVER_UNAVAILABLE, EngineError, EngineServerDown, load_cue_gate
from pfb_app.pipeline.common import StageContext, fail_run, save_run, usable
from pfb_app.pipeline.tracked_identity import label_mapping, mapped_track_ids
from pfb_app.pipeline.tracking import (_engine, _engine_signals, _first_line, _model_id,
                                       _on_tick, place_positions, video_id)
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

# roster roles that feed a cue (`team` alone does not: the team cue needs no roster)
CUE_ROLES = ("number", "position", "grid", "batting", "appearance", "place")
# the order the cues run in: the colour clusters first (no model), the CPU rule last
CUE_ORDER = ("team", "number", "position", "grid", "batting", "appearance", "place")
# what a failed cue is called in its warning (plain words, no internal terms)
CUE_NOUNS = {"team": "チームの色", "number": "背番号", "position": "守備位置",
             "grid": "スタート位置", "batting": "打順", "appearance": "外見",
             "place": "初期位置"}
SERVER_CUES = {"number", "position", "grid", "batting", "appearance"}
# the model key of a names job whose wanted cues need no vLLM server (team / place only)
NAMES_CPU_KEY = "names:cpu"


def _sports(assignment: dict) -> bool:
    return assignment.get("domain") == "sports"


def _roster(subject: dict) -> dict:
    return subject.get("roster") or {}


def needs_names(repo: Repo, submission: dict) -> bool:
    """True when the work is sports, or when a subject's roster has a cue role
    (number / position / grid / batting / appearance / place)."""
    if _sports(repo.get_assignment(submission["assignment_id"])):
        return True
    return any(str(_roster(s).get(role) or "").strip()
               for s in repo.list_subjects(submission["id"]) for role in CUE_ROLES)


def _values(subjects: list[dict], role: str) -> list[tuple[str, str]]:
    """[(participant_id, value)] of the roster rows with this role."""
    return [(s["participant_id"], str(_roster(s)[role]).strip()) for s in subjects
            if s.get("participant_id") and str(_roster(s).get(role) or "").strip()]


def cue_inputs(assignment: dict, subjects: list[dict], camera: str) -> dict[str, str | None]:
    """{cue: None when its input exists, else the status naming what is missing}."""
    sport = (assignment.get("sport") or "").strip()
    missing = {role: None if _values(subjects, role) else "no_roster_column"
               for role in ("number", "grid", "batting", "appearance")}
    return {"team": None if _sports(assignment) else "not_sports", **missing,
            "position": ("no_roster_column" if not _values(subjects, "position") else
                         None if sport else "no_sport"),
            "place": None if place_positions(subjects, camera) else "no_roster_column"}


def display_numbers(settings, assignment: dict) -> bool:
    """True when the number cue runs for display names whatever the gate says: a sports work,
    with the settings' `number_names` on."""
    return _sports(assignment) and bool(getattr(settings, "number_names", True))


def _auto_mapped(repo: Repo, submission: dict, camera: str,
                 track: StageRun | None = None) -> bool:
    """False when the camera's latest mapping, made for `track` (default: the camera's
    latest track run), maps no automatic person (meta "added_people" kind "auto", as
    services.review.ADDED): every evaluated person then has a roster name and reading numbers
    only for display names is wasted GPU time. True otherwise (no track run, or no mapping
    for it yet: it is not known who will be named)."""
    if track is None:
        track = repo.latest_run(submission["id"], Stage.track, camera)
    identity = repo.latest_run(submission["id"], Stage.identity, camera)
    if not usable(track) or not usable(identity) or track.id not in mapped_track_ids(identity):
        return True
    try:
        mapped = {pid for pid in label_mapping(identity.output).values() if pid}
    except (KeyError, TypeError):
        return True
    added = (submission.get("meta") or {}).get("added_people") or {}
    return any(added.get(pid) == "auto" for pid in mapped)


def wanted_cues(repo: Repo, submission: dict, camera: str) -> set[str] | None:
    """The cues the gate enables that have their input for this camera, plus `number` when
    display_numbers holds and an automatic person of the camera can get a number name
    (_auto_mapped); None when the repo does not know the settings (the gate cannot be read)."""
    if repo.settings is None:
        return None
    assignment = repo.get_assignment(submission["assignment_id"])
    gate = load_cue_gate(repo.settings, assignment["model_key"])
    inputs = cue_inputs(assignment, repo.list_subjects(submission["id"]), camera)
    wanted = {cue for cue in CUE_ORDER if gate[cue].enabled and inputs[cue] is None}
    if display_numbers(repo.settings, assignment) and _auto_mapped(repo, submission, camera):
        wanted.add("number")
    return wanted


def names_wanted(repo: Repo, submission: dict, camera: str) -> bool:
    """needs_names, and (when the repo knows the settings) some cue that the gate enables has
    its input for this camera: a names job never starts a vLLM server only to find every cue
    disabled (e.g. before the engine ships its gate file)."""
    if not needs_names(repo, submission):
        return False
    wanted = wanted_cues(repo, submission, camera)
    return wanted is None or bool(wanted)


def needs_server(repo: Repo, submission: dict, camera: str) -> bool:
    """True unless every wanted cue runs without the vLLM server (team / place only); True
    when the gate cannot be read."""
    wanted = wanted_cues(repo, submission, camera)
    return wanted is None or bool(wanted & SERVER_CUES)


def roster_fingerprint(subjects: list[dict]) -> str:
    """A hash of the subjects that feed the cues (a roster attr or an initial place):
    participant id, roster attrs, initial places and appearance. People the coach adds on the
    mapping screen (no roster attrs, no places) do not change it."""
    rows = sorted(
        [s.get("participant_id") or "", _roster(s), s.get("initial_positions") or {},
         s.get("appearance") or ""]
        for s in subjects
        if s.get("participant_id") and (_roster(s) or s.get("initial_positions")))
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def current_names_run(repo: Repo, submission_id: int, camera: str) -> StageRun | None:
    """The newest names run of the camera that still counts: ok, made from the camera's
    latest track run (not stale), newer than the roster (meta roster_run_floor) and built from
    the current cue-feeding subjects (roster_fingerprint); else None. A newer run built from
    other subjects (e.g. a job that was running while the roster changed) is passed over."""
    floor = int(repo.get_submission(submission_id)["meta"].get("roster_run_floor") or 0)
    current = roster_fingerprint(repo.list_subjects(submission_id))
    for run in reversed(repo.run_history(submission_id, Stage.names, camera)):
        if run.id <= floor:
            return None   # this one and every older one predate the roster
        if (usable(run) and run.output.get("roster_fingerprint") == current
                and not repo.is_stale(run)):
            return run
    return None


def run_names(ctx: StageContext, job: Job) -> StageRun:
    """The gated roster cues for one camera (see the module docstring).

    Output: {"track_run_id", "labels", "teams", "team_reason", "crops", "strips", "cues"
    ({cue: raw engine result as JSON}), "status" ({cue: {"status", "error"|"reason"}}),
    "disabled" ({cue: gate reason}), "gate" ({cue: {"enabled", "threshold"}}), "model" (None
    without a server), "roster_floor", "roster_fingerprint"}. Upstream: [track].

    A job under NAMES_CPU_KEY runs without ctx.server (its server cues get "no_server");
    any other names job needs the session's server."""
    started = time.monotonic()
    server = ctx.server
    if server is None and job.model_key != NAMES_CPU_KEY:
        raise RuntimeError("PFB3x3のサーバーが起動していません（pfb3x3セッションでのみ実行できます）")
    repo, sid, cam = ctx.repo, job.submission_id, job.camera
    submission = repo.get_submission(sid)
    track = repo.latest_run(sid, Stage.track, cam)
    if not usable(track):
        return fail_run(ctx, sid, Stage.names, f"人物の追跡結果がありません（{cam}）。",
                        camera=cam, started=started)
    assignment = repo.get_assignment(submission["assignment_id"])
    all_subjects = repo.list_subjects(sid)
    subjects = [s for s in all_subjects if s.get("participant_id")]
    gate = load_cue_gate(ctx.settings, assignment["model_key"])
    engine = _engine(ctx)
    sub_dir = files.submission_dir(ctx.settings, submission["assignment_id"], sid)
    vid = track.output.get("video_id") or video_id(sid, cam)
    tick = _on_tick(ctx, job)
    llm = {"base_url": server.base_url, "model": server.model} if server else {}
    sport = (assignment.get("sport") or "").strip()
    inputs = cue_inputs(assignment, subjects, cam)
    # numbers for display names only when an automatic person of this camera can get one
    display = display_numbers(ctx.settings, assignment) and _auto_mapped(repo, submission,
                                                                         cam, track)
    places = place_positions(subjects, cam)
    calls = {
        "team": lambda: engine.teams(sub_dir, vid, on_tick=tick),
        "number": lambda: engine.numbers(sub_dir, vid, on_tick=tick, **llm,
                                         **_timeline_kw(ctx.settings, assignment, submission)),
        "position": lambda: engine.positions(
            sub_dir, vid, sport=sport, positions=[v for _, v in _values(subjects, "position")],
            on_tick=tick, **llm),
        "grid": lambda: engine.grid(sub_dir, vid, on_tick=tick, **llm),
        "batting": lambda: engine.batting(sub_dir, vid, on_tick=tick, **llm),
        "appearance": lambda: engine.propose(sub_dir, vid, dict(_values(subjects, "appearance")),
                                             on_tick=tick, **llm),
        "place": lambda: engine.rule_names(
            sub_dir, vid, places, on_tick=tick,
            auto_start=track.output.get("sheet_frame_rule") == "auto_start"),
    }
    cues: dict[str, object] = {}
    status: dict[str, dict] = {}
    disabled: dict[str, str] = {}
    warnings: list[str] = []
    for cue in CUE_ORDER:
        missing, call = inputs[cue], calls[cue]
        g = gate[cue]
        for_display = cue == "number" and display
        if not g.enabled:
            disabled[cue] = g.reason
            status[cue] = {"status": "disabled", "reason": g.reason}
            if not for_display:
                continue
        if missing and not for_display:
            status[cue] = {"status": missing}
            continue
        if cue in SERVER_CUES and server is None:
            status[cue] = {"status": "no_server"}
            continue
        try:
            with _engine_signals():
                result = call()
        except (EngineError, ValueError, TypeError, KeyError) as e:
            if isinstance(e, EngineError) and cue in SERVER_CUES and (
                    SERVER_UNAVAILABLE in str(e) or not engine.server_alive(server)):
                raise EngineServerDown(f"PFB3x3のサーバーが応答しません（{cam}）: "
                                       f"{_first_line(e)}") from e
            log.warning("名前の手がかり %s を使えませんでした video=%s: %s", cue, vid, e)
            status[cue] = {"status": "failed", "error": _first_line(e)}
            warnings.append(f"{CUE_NOUNS[cue]}から名前を推定できませんでした")
            continue
        raw, error = _raw(cue, result, places)
        cues[cue] = raw
        status[cue] = {"status": "ok"} if error is None else {"status": "refused",
                                                              "error": error}
    team = cues.get("team") or {}
    output = {
        "track_run_id": track.id, "labels": list(track.output.get("labels") or []),
        "teams": dict(team.get("teams") or {}), "team_reason": team.get("reason"),
        "crops": dict(team.get("crops") or {}), "strips": dict(team.get("strips") or {}),
        "cues": cues, "status": status, "disabled": disabled,
        "gate": {cue: {"enabled": g.enabled, "threshold": g.threshold}
                 for cue, g in gate.items()},
        "model": server.model if server else None,
        "roster_floor": int(submission["meta"].get("roster_run_floor") or 0),
        "roster_fingerprint": roster_fingerprint(all_subjects),
    }
    return save_run(ctx, sid, Stage.names, camera=cam, upstream=[track.id],
                    model_id=_model_id(engine, server.model if server else "cpu"),
                    output=output, warnings=warnings, started=started)


def _timeline_kw(settings, assignment: dict, submission: dict) -> dict:
    """{"timeline_s": s} for a game of people (sports, target person) when
    settings.number_timeline_s > 0: the numbers cue also reads each person's number along
    the track, which shows the moments their box was on another player (identity_check)."""
    s = float(getattr(settings, "number_timeline_s", 0.0) or 0.0)
    game = ((assignment.get("domain") or "school") == "sports"
            and (submission.get("meta") or {}).get("target", "person") == "person")
    return {"timeline_s": s} if s > 0 and game else {}


def _raw(cue: str, result, places) -> tuple[object, str | None]:
    """(the engine's raw result as JSON, its refusal or None)."""
    if cue == "team":
        return ({"teams": dict(result.teams), "reason": result.reason,
                 "crops": {k: str(v) for k, v in result.crops.items()},
                 "strips": {k: str(v) for k, v in result.strips.items()},
                 "sizes": dict(result.sizes), "other_dist": result.other_dist}, result.error)
    if cue == "number":
        raw = {}
        for label, r in result.items():
            d = dataclasses.asdict(r)
            d["frames"], d["all"] = list(r.frames), list(r.all)
            if r.timeline_s:
                d["timeline"] = [list(t) for t in r.timeline]
                d["segments"] = [list(t) for t in r.segments]
            else:   # no timeline asked: the stored reads stay as they were
                for k in ("timeline_s", "timeline", "segments", "dominant", "dominant_share"):
                    d.pop(k, None)
            raw[label] = d
        return raw, result.error
    if cue == "position":
        return ({"positions": dict(result), "slots": dict(result.slots),
                 "claims": dict(result.claims)}, result.error)
    if cue == "grid":
        return ({"order": list(result), "frame": result.frame, "frame_rule": result.frame_rule,
                 "direction": result.direction, "direction_source": result.direction_source},
                result.error)
    if cue == "batting":
        return ({"order": list(result), "frames": result.frames, "errors": result.errors,
                 "first_frame": dict(result.first_frame), "seen": dict(result.seen),
                 "sequence": [list(x) for x in result.sequence]}, result.error)
    # appearance (propose) and place (rule_names): {P-label: participant_id}, None = refused
    raw = {"names": dict(result or {})}
    if cue == "place":
        raw["positions"] = dict(places)
    return raw, None if result is not None else "結果がありません"
