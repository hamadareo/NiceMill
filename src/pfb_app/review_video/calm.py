"""Calm over-the-head tags: a steady anchor point and a steady "what they seem to be doing" per person.

The raw boxes jitter (the detector's box size and position change from one picture to the next) and
the activity word flips between walking / running / standing every few tenths of a second. The player
draws a tag over each head, so the tags shook and their words flickered. Both are fixed here, once,
when the player is made (nothing is stored; the numbers are exact and testable):

* anchor: a One-Euro filter (smooth when the person is slow, quick when they move fast, so it does not
  lag) on the top-centre of the box, then a dead band: the anchor moves only when the filtered point is
  more than DEAD_PX away, and then only by the excess. Integer pixels.
* words: a state is shown at least MIN_SHOW_S; another state replaces it only after it has been the raw
  state for HOLD_S in a row; a short interruption keeps the earlier state; "not known" never replaces one.
"""
from __future__ import annotations

import math

MIN_CUT = 0.6  # Hz: smoothing when still
BETA = 0.008  # speed coefficient (per px/s): less smoothing when moving fast
D_CUT = 1.0
DEAD_PX = 4.0  # picture pixels (of the video's own size)
HOLD_S = 2.0  # a new word must be the raw one this long (measured: 1.0 s still gave 10 changes a minute)
MIN_SHOW_S = 4.0  # a word stays at least this long
GAP_S = 2.0            # a hole in a track longer than this starts the track again


def _alpha(cut: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cut)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    def __init__(self, mincut=None, beta=None, dcut=None):
        self.mincut = MIN_CUT if mincut is None else mincut
        self.beta = BETA if beta is None else beta
        self.dcut = D_CUT if dcut is None else dcut
        self.x = None
        self.dx = 0.0

    def __call__(self, v: float, dt: float) -> float:
        if self.x is None:
            self.x = v
            return v
        d = (v - self.x) / max(dt, 1e-6)
        self.dx += _alpha(self.dcut, dt) * (d - self.dx)
        cut = self.mincut + self.beta * abs(self.dx)
        self.x += _alpha(cut, dt) * (v - self.x)
        return self.x


class DeadBand:
    def __init__(self, width=None):
        self.w = DEAD_PX if width is None else width
        self.y = None

    def __call__(self, v: float) -> float:
        if self.y is None:
            self.y = v
        elif abs(v - self.y) > self.w:
            self.y = v - math.copysign(self.w, v - self.y)
        return self.y


def smooth_anchor(rows: list, fps: float) -> list[tuple[int, int, int]]:
    """rows [[frame, x, y, w, h, ...]] in frame order -> [(frame, ax, ay)] (the top-centre of the box, steadied)."""
    out = []
    fx = fy = dx = dy = None
    last = None
    for r in rows:
        t = r[0] / (fps or 30.0)
        if last is None or t - last > GAP_S:
            fx, fy, dx, dy = OneEuro(), OneEuro(), DeadBand(), DeadBand()
            dt = 1.0
        else:
            dt = max(t - last, 1e-3)
        last = t
        ax = dx(fx(r[1] + r[3] / 2.0, dt))
        ay = dy(fy(float(r[2]), dt))
        out.append((int(r[0]), int(round(ax)), int(round(ay))))
    return out


def stabilise(frames: list, acts: list, fps: float, hold_s: float | None = None, min_show_s: float | None = None) -> list[int]:
    """The shown activity code per row (0 = not known: never replaces a shown state)."""
    hold_s = HOLD_S if hold_s is None else hold_s
    min_show_s = MIN_SHOW_S if min_show_s is None else min_show_s
    shown = []
    cur = None
    since = cand = cand_since = None
    last = None
    for f, a in zip(frames, acts):
        t = f / (fps or 30.0)
        if cur is None or (last is not None and t - last > GAP_S):
            cur, since, cand = a, t, None
        elif a == cur or a == 0:
            cand = None
        elif cur == 0:
            cur, since, cand = a, t, None
        elif a == cand:
            if t - cand_since >= hold_s and t - since >= min_show_s:
                cur, since, cand = a, t, None
        else:
            cand, cand_since = a, t
        last = t
        shown.append(cur)
    return shown


def calm_entity(rows: list, acts: list, fps: float) -> list[list[int]]:
    """[[frame, ax, ay, act]] for one person's rows (in frame order)."""
    a = smooth_anchor(rows, fps)
    s = stabilise([r[0] for r in rows], acts, fps)
    return [[f, x, y, int(c)] for (f, x, y), c in zip(a, s)]
