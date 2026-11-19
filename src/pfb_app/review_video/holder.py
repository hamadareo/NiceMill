"""Who holds the ball (an estimate): ball detections + the people's boxes (pure; no model).

A frame counts only when the ball was seen with confidence >= MIN_CONF and exactly one listed
person's box has the ball clearly closest to its body (hands, waist or feet: inside the box
widened by 20 %, nearest to a point 65 % down the box, nearer than 0.6 box-widths/heights, and
the runner-up at least 1.3 times as far). Frames without a ball give no holder; a holder is
kept for a few frames around a sighting (HOLD_PAD) and a short gap between two sightings of the
same person is filled (MAX_GAP). Segments from one confident sighting only need conf >= SOLO_CONF.
Not checked against the scene sentences. It is an estimate, shown as one.
"""
from __future__ import annotations

MIN_CONF = 0.4
SOLO_CONF = 0.6
HOLD_PAD = 3
MAX_GAP = 6
NEAR = 0.6
RATIO = 1.3


def holder_of(ball, boxes: dict):
    """The label holding a ball [x1, y1, x2, y2, conf] among {label: (x, y, w, h)}, or None."""
    cx, cy = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
    cand = []
    for lb, (x, y, w, h) in boxes.items():
        if w <= 0 or h <= 0 or not (x - 0.2 * w <= cx <= x + 1.2 * w and y <= cy <= y + 1.05 * h):
            continue
        d = (((cx - (x + w / 2)) / w) ** 2 + ((cy - (y + 0.65 * h)) / h) ** 2) ** 0.5
        cand.append((d, lb))
    cand.sort()
    if not cand or cand[0][0] > NEAR or (len(cand) > 1 and cand[1][0] < RATIO * cand[0][0]):
        return None
    return cand[0][1]


def segments(ball: dict, boxes_at, labels=None) -> list[list]:
    """[[first_frame, last_frame, label], ...]; `boxes_at(frame)` -> {label: (x, y, w, h)}."""
    sights = []
    for f in sorted(int(k) for k in ball):
        b = ball[f] if f in ball else ball[str(f)]
        if b[4] < MIN_CONF:
            continue
        bx = boxes_at(f)
        if labels is not None:
            bx = {k: v for k, v in bx.items() if k in labels}
        lb = holder_of(b, bx)
        if lb is not None:
            sights.append((f, lb, b[4]))
    out: list[list] = []      # [first sighting, last sighting, label, best conf, sightings]
    for f, lb, conf in sights:
        if out and out[-1][2] == lb and f - out[-1][1] <= MAX_GAP:
            out[-1][1], out[-1][3], out[-1][4] = f, max(out[-1][3], conf), out[-1][4] + 1
        else:
            out.append([f, f, lb, conf, 1])
    return [[a - HOLD_PAD, b + HOLD_PAD, lb] for a, b, lb, c, n in out if c >= SOLO_CONF or n >= 2]


def holder_at(segs, frame: int):
    for a, b, lb in segs:
        if a <= frame <= b:
            return lb
    return None
