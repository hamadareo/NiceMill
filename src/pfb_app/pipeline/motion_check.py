"""Re-check a movement word (走る・歩く・移動する・位置につく) against the tracks, with no model.

The recorded sentence said the person ran / walked / moved. The tracks (the person's box in every
frame) say how fast the box's foot point moved. A broadcast camera pans and zooms, so the raw
movement of a box is mostly the camera: the movement of everyone else on the same frames is taken
as the camera's (their median step), and only what is left is the person's own. A step with fewer
than MIN_OTHERS other people on both of its frames is not measured. Speeds are in body heights
per second (the box's height), so the picture's size and the person's distance do not matter much.

This is a necessary-condition check, not a measurement of the sport: a person who moves with the
whole pack (a fast break) looks still against it, which can only make the sentence wait for a
person, never pass. Pure functions over a `BoxIndex` (export.tracks); nothing is stored here.

Verdicts have the shape of pipeline.claim_check.Verdict.as_dict (plus `method`), so the automatic
approval reads both the same way: supported + keep = the movement is confirmed; unchecked =
could not be measured (the sentence waits for a person); contradicted = the tracks say otherwise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

METHOD = "motion"
POLICY_VERSION = "motion/1"

# movement word -> action. 「走」 is also an action of pipeline.claim_check (first match wins there),
# here it is checked by the tracks instead of a picture question.
_WORDS = (
    ("run", r"走っ|走る|走り|走れ|ダッシュ|駆け"),
    ("walk", r"歩い|歩く|歩き"),
    ("move", r"移動|位置につ|ポジション(?:に|を)|向かっ"),
)
ACTIONS = tuple(a for a, _ in _WORDS)

WINDOW_S = 1.5          # the steps are taken from t - WINDOW_S to t + WINDOW_S
STEP_S = 0.5
MIN_OTHERS = 3          # people needed on both frames of a step to read the camera's movement
MIN_STEPS = 4           # measured steps needed (of 6)
TOL_S = 0.12            # a box this close to a wanted time counts as on that frame
RUN_MIN = 1.4           # body heights per second (about a jog, 2.5 m/s)
STILL_MAX = 0.3         # below: the person stood (against the camera's movement)
WALK_MIN, WALK_MAX = 0.3, 1.2
MOVE_MIN = 0.3          # median speed, or ...
MOVE_NET = 0.5          # ... this much of a body height gone over the whole window


def motion_actions(text: str) -> set[str]:
    """Every movement action a sentence claims."""
    return {a for a, rx in _WORDS if re.search(rx, text or "")}


@dataclass
class Stats:
    steps: int = 0                  # measured steps
    median_speed: float | None = None   # body heights per second, camera movement removed
    net: float | None = None        # body heights between the first and last measured point


def _foot(box) -> tuple[float, float]:
    return box.x + box.w / 2.0, box.y + box.h


def stats_at(index, labels: list[str], vf: int, fps: float) -> Stats:
    """The person's own movement (camera removed) around frame `vf` (1-based, the tracks' fps)."""
    if index is None or not labels or not fps:
        return Stats()
    tol = max(1, int(round(TOL_S * fps)))
    others = [lb for lb in index.labels if lb not in set(labels)]
    n = int(round(2 * WINDOW_S / STEP_S))
    frames = [vf + int(round((-WINDOW_S + i * STEP_S) * fps)) for i in range(n + 1)]
    pts = [index.nearest(labels, f, tol) for f in frames]
    speeds: list[float] = []
    net: list[tuple[float, float]] = []
    heights: list[float] = []
    cam = np.zeros(2)
    total = np.zeros(2)
    used_first = None
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        if a is None or b is None:
            continue
        shifts = []
        for lb in others:
            oa, ob = index.nearest([lb], frames[i], tol), index.nearest([lb], frames[i + 1], tol)
            if oa is not None and ob is not None:
                fa, fb = _foot(oa[1]), _foot(ob[1])
                shifts.append((fb[0] - fa[0], fb[1] - fa[1]))
        if len(shifts) < MIN_OTHERS:
            continue
        cam = np.median(np.asarray(shifts), axis=0)
        fa, fb = _foot(a[1]), _foot(b[1])
        own = np.array([fb[0] - fa[0], fb[1] - fa[1]]) - cam
        h = max((a[1].h + b[1].h) / 2.0, 1.0)
        speeds.append(float(np.hypot(*own)) / h / STEP_S)
        total += own
        heights.append(h)
        used_first = used_first if used_first is not None else i
    if not speeds:
        return Stats()
    return Stats(len(speeds), float(np.median(speeds)),
                 float(np.hypot(*total)) / float(np.median(heights)))


def _v(action: str, status: str, show: str, reasons: list[str], conf: str = "high") -> dict:
    return {"action": action, "status": status, "confidence": conf, "show": show,
            "reasons": reasons, "method": METHOD, "version": POLICY_VERSION}


def verdict(action: str, s: Stats) -> dict:
    """The verdict for one movement action from the person's own movement."""
    if s.steps < MIN_STEPS or s.median_speed is None:
        return _v(action, "unchecked", "keep",
                  ["追跡の記録から、動きを測れませんでした（周りの人が少ない、または追跡が途切れている）"], "none")
    sp = s.median_speed
    shown = f"（体の高さの毎秒{sp:.2f}倍）"
    if action == "run":
        if sp >= RUN_MIN:
            return _v(action, "supported", "keep", ["追跡の記録で、カメラの動きを除いても速く動いていた" + shown])
        if sp < STILL_MAX:
            return _v(action, "contradicted", "fallback", ["追跡の記録で、ほとんど動いていなかった" + shown])
        return _v(action, "unclear", "hedge", ["走ったと言える速さには届かなかった" + shown], "low")
    if action == "walk":
        if WALK_MIN <= sp <= WALK_MAX:
            return _v(action, "supported", "keep", ["追跡の記録で、歩く程度の速さで動いていた" + shown])
        if sp < STILL_MAX:
            return _v(action, "contradicted", "fallback", ["追跡の記録で、ほとんど動いていなかった" + shown])
        return _v(action, "unclear", "hedge", ["歩く速さの範囲ではなかった" + shown], "low")
    if sp >= MOVE_MIN or (s.net or 0.0) >= MOVE_NET:
        return _v(action, "supported", "keep", ["追跡の記録で、場所が動いていた" + shown])
    return _v(action, "contradicted", "fallback", ["追跡の記録で、ほとんど動いていなかった" + shown])
