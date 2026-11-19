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


def _names_reads(repo, submission_id: int, camera: str, track_id: int) -> dict | None:
    """{label: number read} of the newest usable names run made for this track run whose
    numbers have a timeline; None when there is none."""
    for run in reversed(repo.run_history(submission_id, Stage.names, camera)):
        out = run.output or {}
        if not usable(run) or out.get("track_run_id") != track_id or repo.is_stale(run):
            continue
        reads = (out.get("cues") or {}).get("number")
        if isinstance(reads, dict) and any(isinstance(r, dict) and r.get("segments")
                                           for r in reads.values()):
            return reads
    return None


def foreign_events(repo, behavior: StageRun) -> set[str]:
    """The event ids of a tracked behavior run that happened while the person's label showed
    another player (see the module doc); empty without a number timeline."""
    from pfb_app.export import tracks          # noqa: PLC0415 - export imports pipeline
    from pfb_app.pipeline.tracked_identity import label_mapping   # noqa: PLC0415
    ups = {r.stage: r for r in (repo.get_run(i) for i in behavior.upstream_run_ids) if r}
    track, identity = ups.get(Stage.track), ups.get(Stage.identity)
    if not usable(track) or not usable(identity) or not behavior.camera:
        return set()
    reads = _names_reads(repo, behavior.submission_id, behavior.camera, track.id)
    meta = tracks.track_meta((track.output or {}).get("boxes"))
    if not reads or meta is None or not meta.fps:
        return set()
    try:
        mapping = label_mapping(identity.output)
    except (KeyError, TypeError, ValueError):
        return set()
    labels: dict[str, list[str]] = {}
    for label, pid in mapping.items():
        if pid:
            labels.setdefault(pid, []).append(label)
    out: set[str] = set()
    for e in (behavior.output or {}).get("events") or []:
        sec = clock_seconds(e.get("video_time"))
        mine = labels.get(e.get("participant")) or []
        if sec is None or len(mine) != 1:
            continue
        read = reads.get(mine[0])
        if not isinstance(read, dict):
            continue
        step = float(read.get("timeline_s") or 0.0) * meta.fps
        frame = round(sec * meta.fps) + 1    # video_time is the time in the video file
        if foreign_at(read, frame, reach=max(step * REACH_STEPS, meta.fps)):
            out.add(e["event_id"])
    return out
