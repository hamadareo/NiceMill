"""Automatic choice of the people to evaluate in a video (pure; no engine, no database).

The tracker boxes everyone it sees, including passers-by and fragments. `select_labels` keeps
the people who are on screen long enough and large enough (the engine's `quality` numbers), at
most `people_cap` of them, and orders them the way the naming screen lists them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

MIN_COVERAGE = 0.25   # share of the task range a person must be boxed in
# ... but never more than this share of what the best-seen person manages: in a long video
# (cuts away from the court, linked fragments that break up) nobody is boxed for a quarter of
# it, yet everybody was on screen for minutes (2026-11-14 trial, a 20 min analysis: best 0.34,
# 2 of 14 evaluated; every one of the 14 was boxed for more than 2 minutes). The bar only
# falls when the best person is below MIN_COVERAGE / REL_COVERAGE (about 0.56).
REL_COVERAGE = 0.45
MIN_HEIGHT = 0.06     # mean box height as a share of the frame
DEFAULT_CAP = 12      # people when neither max_people nor the detector's p90 is known


def is_evaluated(subject: dict) -> bool:
    """A subject that is evaluated: it has a participant id and is not excluded. The one
    predicate every list of students uses (a document's subjects have no participant id and
    are listed by the document flow itself). A person whose data was removed on request
    (services.erasure: erased_at) is never evaluated again."""
    return (bool(subject.get("participant_id")) and not subject.get("excluded")
            and not subject.get("erased_at"))


@dataclass(frozen=True)
class BoxStat:
    """One label's boxes in boxes.txt: how many frames, the first frame and its centre x,
    and the last frame."""
    frames: int
    first_frame: int
    first_x: float
    last_frame: int


def _hard_max(domain: str) -> int:
    # imported here: services.content_check imports the pipeline package at load time
    from pfb_app.services.content_check import MAX_PEOPLE
    return MAX_PEOPLE.get(domain, MAX_PEOPLE["school"])


def people_cap(max_people: int | None, detect_p90: float | None, domain: str) -> int:
    """How many people to keep: the teacher's max_people if a positive int, else the detector's
    p90 + 2, else DEFAULT_CAP; never above the domain's hard maximum, at least 1."""
    if isinstance(max_people, int) and not isinstance(max_people, bool) and max_people > 0:
        cap = max_people
    elif detect_p90 is not None:
        cap = math.ceil(detect_p90) + 2
    else:
        cap = DEFAULT_CAP
    return max(1, min(_hard_max(domain), cap))


def _number(label: str) -> int:
    digits = label[1:]
    return int(digits) if digits.isdigit() else 0


def box_stats(boxes_path: Path, labels: list[str]) -> dict[str, BoxStat]:
    """Per-label BoxStat from a MOT boxes.txt (`frame,track_id,x,y,w,h,conf`, label P{id});
    {} when the file is missing or unreadable. Labels without boxes are left out."""
    try:
        text = Path(boxes_path).read_text(encoding="utf-8")
    except OSError:
        return {}
    wanted = set(labels)
    found: dict[str, list] = {}   # label -> [frames, first_frame, first_x, last_frame]
    for line in text.splitlines():
        parts = line.split(",")
        try:
            frame, tid = int(float(parts[0])), int(float(parts[1]))
            x, w = float(parts[2]), float(parts[4])
        except (ValueError, IndexError):
            continue
        label = f"P{tid}"
        if label not in wanted:
            continue
        cur = found.get(label)
        if cur is None:
            found[label] = [1, frame, x + w / 2, frame]
            continue
        cur[0] += 1
        if frame < cur[1]:
            cur[1], cur[2] = frame, x + w / 2
        cur[3] = max(cur[3], frame)
    return {k: BoxStat(v[0], v[1], v[2], v[3]) for k, v in found.items()}


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def coverage_bar(scores) -> float:
    """The coverage a person must reach: MIN_COVERAGE, or REL_COVERAGE of the best score when
    that is lower (a video in which nobody is boxed for MIN_COVERAGE of the time). A video
    where somebody is boxed for about 56 % of it or more keeps MIN_COVERAGE exactly."""
    best = max(scores, default=0.0)
    return min(MIN_COVERAGE, REL_COVERAGE * best)


def select_labels(labels: list[str], quality: dict | None, stats: dict[str, BoxStat],
                  cap: int) -> list[str]:
    """The labels to evaluate, in naming order (first appearance, then first_x, then label
    number). With `quality`: keep coverage >= coverage_bar and a known mean height >=
    MIN_HEIGHT, best coverage first, at most `cap`; when nobody passes, the top `cap` by
    coverage. Without it: coverage = frames / file-wide span (from the stats), the same
    coverage bar and relaxation, no height filter."""
    cap = max(1, cap)
    if quality is None:
        span = 1
        if stats:
            span = max(1, max(st.last_frame for st in stats.values())
                       - min(st.first_frame for st in stats.values()) + 1)
        score = {lb: stats[lb].frames / span if lb in stats else 0.0 for lb in labels}
        bar = coverage_bar(score.values())
        pool = [lb for lb in labels if score[lb] >= bar] or list(labels)
    else:
        score, tall = {}, []
        for lb in labels:
            q = quality.get(lb) or {}
            cov, height = _num(q.get("coverage")), _num(q.get("mean_h_frac"))
            score[lb] = cov or 0.0
            if cov is not None and height is not None and height >= MIN_HEIGHT:
                tall.append(lb)
        bar = coverage_bar(score[lb] for lb in tall)
        pool = [lb for lb in tall if score[lb] >= bar] or list(labels)
    best = sorted(pool, key=lambda lb: (-score[lb], _number(lb)))[:cap]

    def naming_order(lb: str):
        st = stats.get(lb)
        return (st.first_frame, st.first_x, _number(lb)) if st else (math.inf, math.inf, _number(lb))
    return sorted(best, key=naming_order)


def detect_p90(detect_output: dict | None, camera: str, target: str = "person") -> float | None:
    """The detector's p90 head count of one camera (persons, or vehicles for target
    "vehicle"); None when absent, non-numeric or negative."""
    try:
        kind = "vehicles" if target == "vehicle" else "persons"
        value = _num(detect_output["cameras"][camera][kind]["p90"])
    except (KeyError, TypeError):
        return None
    return value if value is not None and value >= 0 else None
