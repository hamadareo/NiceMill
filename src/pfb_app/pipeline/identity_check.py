"""Moments when a tracked person's box is on someone else (an ID swap).

The engine tracker links pieces of track into one person; on a game it often links pieces of
different players (work 6: one label showed #5, then #3, then #1). The names stage reads the
shirt number along each label's track (the engine's number timeline: "segments" of the same
number read in a row, and the label's "dominant" number). A moment of a label is the
person's own unless the timeline reads ANOTHER number there:

- `number_at`: the number of the segment the frame is in, else of the nearest segment within
  `reach` frames (between two segments: the nearer one), else None (nothing read near it);
- `foreign_at`: a number is read there and it is not the label's dominant number. No
  dominant number (a tie, or no timeline) or nothing read near the frame: never foreign.

`foreign_events` applies it to a behavior run: the events whose person's label reads another
number at the event's time. They are left out of the person's evidence (contribution), so no
feedback, picture or count stands on another player's moment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.pfb3x3_bridge import clock_seconds

log = logging.getLogger(__name__)

REACH_STEPS = 1.5    # a segment counts this many timeline steps beyond its first/last read


def number_at(read: dict | None, frame: int, *, reach: float) -> str | None:
    segs = (read or {}).get("segments") or []
    best = None
    for seg in segs:
        try:
            first, last, number = int(seg[0]), int(seg[1]), seg[2]
        except (TypeError, ValueError, IndexError):
            continue
        if number is None:
            continue
        d = 0 if first <= frame <= last else min(abs(frame - first), abs(frame - last))
        if d <= reach and (best is None or d < best[0]):
            best = (d, str(number))
    return None if best is None else best[1]


def foreign_at(read: dict | None, frame: int, *, reach: float) -> bool:
    dominant = (read or {}).get("dominant")
    if dominant is None:
        return False
    number = number_at(read, frame, reach=reach)
    return number is not None and number != str(dominant)


def _names_run(repo, submission_id: int, camera: str, track_id: int):
    """The newest usable names run made for this track run whose numbers have a timeline;
    None when there is none."""
    for run in reversed(repo.run_history(submission_id, Stage.names, camera)):
        out = run.output or {}
        if not usable(run) or out.get("track_run_id") != track_id or repo.is_stale(run):
            continue
        reads = (out.get("cues") or {}).get("number")
        if isinstance(reads, dict) and any(isinstance(r, dict) and r.get("segments")
                                           for r in reads.values()):
            return run
    return None


def _names_reads(repo, submission_id: int, camera: str, track_id: int) -> dict | None:
    """{label: number read} of the newest usable names run made for this track run whose
    numbers have a timeline; None when there is none."""
    run = _names_run(repo, submission_id, camera, track_id)
    return None if run is None else (run.output or {}).get("cues", {}).get("number")


SAME, FOREIGN, UNKNOWN = "same", "foreign", "unknown"


def _context(repo, behavior: StageRun):
    """(names run, track meta, {label: participant id}) of a tracked behavior run, or None
    when the run has no usable track, mapping or number timeline."""
    from pfb_app.export import tracks          # noqa: PLC0415 - export imports pipeline
    from pfb_app.pipeline.tracked_identity import label_mapping   # noqa: PLC0415
    ups = {r.stage: r for r in (repo.get_run(i) for i in behavior.upstream_run_ids) if r}
    track, identity = ups.get(Stage.track), ups.get(Stage.identity)
    if not usable(track) or not usable(identity) or not behavior.camera:
        return None
    names = _names_run(repo, behavior.submission_id, behavior.camera, track.id)
    meta = tracks.track_meta((track.output or {}).get("boxes"))
    if names is None or meta is None or not meta.fps:
        return None
    try:
        mapping = label_mapping(identity.output)
    except (KeyError, TypeError, ValueError):
        return None
    return names, meta, mapping


def event_states(repo, behavior: StageRun) -> dict[str, str]:
    """{event id: SAME / FOREIGN / UNKNOWN} for the events of a tracked behavior run: SAME when
    the number timeline reads the label's own (dominant) number at the event's time, FOREIGN
    when it reads another number there, UNKNOWN when nothing is read near the time, the label
    has no dominant number or the person has not exactly one label. Empty without a number
    timeline (nothing can be told)."""
    ctx = _context(repo, behavior)
    if ctx is None:
        return {}
    names, meta, mapping = ctx
    reads = (names.output or {}).get("cues", {}).get("number") or {}
    labels: dict[str, list[str]] = {}
    for label, pid in mapping.items():
        if pid:
            labels.setdefault(pid, []).append(label)
    out: dict[str, str] = {}
    for e in (behavior.output or {}).get("events") or []:
        eid = e.get("event_id")
        out[eid] = UNKNOWN
        sec = clock_seconds(e.get("video_time"))
        mine = labels.get(e.get("participant")) or []
        if sec is None or len(mine) != 1:
            continue
        read = reads.get(mine[0])
        if not isinstance(read, dict) or read.get("dominant") is None:
            continue
        step = float(read.get("timeline_s") or 0.0) * meta.fps
        frame = round(sec * meta.fps) + 1    # video_time is the time in the video file
        number = number_at(read, frame, reach=max(step * REACH_STEPS, meta.fps))
        if number is not None:
            out[eid] = SAME if number == str(read["dominant"]) else FOREIGN
    return out


def foreign_events(repo, behavior: StageRun) -> set[str]:
    """The event ids of a tracked behavior run that happened while the person's label showed
    another player (see the module doc); empty without a number timeline."""
    return {eid for eid, state in event_states(repo, behavior).items() if state == FOREIGN}


@dataclass(frozen=True)
class PersonFacts:
    """What the number timeline and the names stage say about one person (a participant)."""
    numbers: frozenset[str]     # the dominant numbers of the person's labels
    conflict: bool              # the person's labels read different dominant numbers
    duplicate: bool             # another person's label reads one of those numbers (same team, or a team unknown)
    other_team: bool            # every labelled track of the person is in neither team (officials, onlookers)
    unsure: bool                # the participation rule is not sure the person plays on the court


def person_facts(repo, behavior: StageRun, participant_id: str) -> PersonFacts | None:
    """See PersonFacts; None without a number timeline for the behavior run."""
    ctx = _context(repo, behavior)
    if ctx is None:
        return None
    names, _meta, mapping = ctx
    out = names.output or {}
    reads = (out.get("cues") or {}).get("number") or {}
    teams = out.get("teams") or {}
    mine = [lb for lb, pid in mapping.items() if pid == participant_id]
    numbers = {str(reads[lb]["dominant"]) for lb in mine
               if isinstance(reads.get(lb), dict) and reads[lb].get("dominant") is not None}
    duplicate = False
    for lb, read in reads.items():
        if lb in mine or not isinstance(read, dict) or read.get("dominant") is None:
            continue
        if str(read["dominant"]) not in numbers:
            continue
        theirs = teams.get(lb)
        if any(teams.get(m) and theirs and teams.get(m) != theirs for m in mine):
            continue    # the same number on the other team is another player's
        duplicate = True
    labelled = [teams.get(lb) for lb in mine if teams.get(lb)]
    sub_meta = repo.get_submission(behavior.submission_id).get("meta") or {}
    part = (sub_meta.get("participation") or {}).get(behavior.camera) or {}
    flagged = set(part.get("check") or {}) | set(part.get("not_playing") or {})
    return PersonFacts(frozenset(numbers), len(numbers) > 1, duplicate,
                       bool(labelled) and all(t == "other" for t in labelled),
                       any(lb in flagged for lb in mine))
