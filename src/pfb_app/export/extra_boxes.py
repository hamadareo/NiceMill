"""Boxes for the people the tracking kept no identity for (read only, numpy only).

The engine links the detector's tracklets into at most `max_people` persons (P1..Pn) and drops
the rest; on a game video the persons' boxes cover about half of the frames, so about half of
the people on the court have no box at a given moment although the detector saw most of them.
`extra_boxes` gives those detections back as *unnamed* boxes for drawing (the review player's
「全員の枠」): it never changes a person's boxes or identity, adds nothing to the database,
and a box here is never a person's evidence.

Input: the engine's raw_tracks.txt rows (frame, tracker_id, x, y, w, h, conf) and the persons'
boxes.txt rows (frame, person, x, y, w, h, conf). A raw row is kept when

1. it is not already a person's box (IoU >= DUP_IOU with a person's box of that frame, or its
   centre inside one) and not a second box on the same thing (non-maximum suppression);
2. its tracklet lasts MIN_TRACK_S (a flicker is not drawn);
3. its foot point is on the court: inside the area the persons' own foot points span within
   +-WINDOW_S (their convex hull, widened by HULL_MARGIN x a person's height; the camera
   moves and players spread out, so the area is local in time and generous), which leaves out spectators, the bench and the
   people at the table; and its height is plausible for that foot position (a line fitted to
   the persons' own height against foot y, x HEIGHT_LO..HEIGHT_HI);
4. the tracklet passes 2 and 3 in at least MAJORITY of its rows.

No court registration is needed (the games it is used on often have none).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUP_IOU = 0.3
NMS_IOU = 0.5
MIN_TRACK_S = 0.4
WINDOW_S = 4.0
HULL_MARGIN = 1.0
HEIGHT_LO, HEIGHT_HI = 0.6, 1.5
MAJORITY = 0.6
MIN_ANCHORS = 3          # persons' foot points a window needs to define the court area


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one (x, y, w, h) box with each row of b."""
    iw = np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]) - np.maximum(a[0], b[:, 0])
    ih = np.minimum(a[1] + a[3], b[:, 1] + b[:, 3]) - np.maximum(a[1], b[:, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise; the points themselves when < 3."""
    p = np.unique(np.round(pts, 1), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1]) \
                    - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(q)
        return out

    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])


def _dist_to_hull(pt: np.ndarray, hull: np.ndarray) -> float:
    """0 inside the (convex, counter-clockwise) hull, else the distance to it."""
    n = len(hull)
    if n == 1:
        return float(np.linalg.norm(pt - hull[0]))
    best, inside = np.inf, n >= 3
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0, 1))
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
        if n >= 3 and ab[0] * (pt[1] - a[1]) - ab[1] * (pt[0] - a[0]) < 0:
            inside = False
    return 0.0 if inside else best


def extra_boxes(raw: np.ndarray, kept: np.ndarray, fps: float) -> np.ndarray:
    """The unnamed boxes: rows (frame, extra id, x, y, w, h, conf); the id is the tracker's.
    Empty when there are too few persons' boxes to tell where the court is."""
    raw = np.asarray(raw, dtype=float).reshape(-1, 7)
    kept = np.asarray(kept, dtype=float).reshape(-1, 7)
    empty = np.zeros((0, 7))
    if not len(raw) or len(kept) < MIN_ANCHORS:
        return empty
    # 1. not a person's box, not a second box on the same thing
    by_frame: dict[int, np.ndarray] = {}
    order = np.argsort(kept[:, 0], kind="stable")
    ks = kept[order]
    frames, starts = np.unique(ks[:, 0].astype(int), return_index=True)
    for f, s, e in zip(frames, starts, list(starts[1:]) + [len(ks)]):
        by_frame[int(f)] = ks[s:e, 2:6]
    keep = np.ones(len(raw), bool)
    for i, r in enumerate(raw):
        b = by_frame.get(int(r[0]))
        if b is None:
            continue
        cx, cy = r[2] + r[4] / 2, r[3] + r[5] / 2
        inside = (b[:, 0] <= cx) & (cx <= b[:, 0] + b[:, 2]) & (b[:, 1] <= cy) & (cy <= b[:, 1] + b[:, 3])
        if inside.any() or _iou(r[2:6], b).max() >= DUP_IOU:
            keep[i] = False
    rest = raw[keep]
    if not len(rest):
        return empty
    # 3a. the persons' height against foot y (a line, fitted robustly)
    foot_y = kept[:, 3] + kept[:, 5]
    a, b0 = np.polyfit(foot_y, kept[:, 5], 1)
    res = kept[:, 5] - (a * foot_y + b0)
    ok = np.abs(res) < 2.5 * 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    if ok.sum() >= 3:
        a, b0 = np.polyfit(foot_y[ok], kept[ok, 5], 1)
    med_h = float(np.median(kept[:, 5]))
    # 3b. the court area per window: the persons' foot points within +-WINDOW_S
    feet = np.c_[kept[:, 0], kept[:, 2] + kept[:, 4] / 2, foot_y]
    feet = feet[np.argsort(feet[:, 0])]
    half = max(1, int(round(WINDOW_S * fps)))
    cache: dict[int, np.ndarray | None] = {}

    def area(frame: int):
        k = (frame // max(1, half // 2)) * max(1, half // 2)     # one hull per half window
        if k not in cache:
            w = half
            while True:
                m = (feet[:, 0] >= k - w) & (feet[:, 0] <= k + w)
                if m.sum() >= MIN_ANCHORS * 2 or w >= 8 * half:
                    break
                w *= 2
            cache[k] = _hull(feet[m][:, 1:3]) if m.sum() >= MIN_ANCHORS else None
        return cache[k]

    good = np.zeros(len(rest), bool)
    for i, r in enumerate(rest):
        h = area(int(r[0]))
        if h is None:
            continue
        foot = np.array([r[2] + r[4] / 2, r[3] + r[5]])
        exp = a * foot[1] + b0
        if exp <= 0:
            exp = med_h
        if not HEIGHT_LO <= r[5] / exp <= HEIGHT_HI:
            continue
        good[i] = _dist_to_hull(foot, h) <= HULL_MARGIN * exp
    # 2 + 4. whole tracklets
    out = []
    min_rows = max(2, int(round(MIN_TRACK_S * fps)))
    for tid in np.unique(rest[:, 1]):
        m = rest[:, 1] == tid
        if m.sum() >= min_rows and good[m].mean() >= MAJORITY:
            out.append(rest[m & good])
    if not out:
        return empty
    ex = np.vstack(out)
    # 1b. a second box on the same thing: keep the more confident (greedy, per frame)
    ex = ex[np.lexsort((-ex[:, 6], ex[:, 0]))]
    final = []
    cur, boxes = None, []
    for r in ex:
        if cur != r[0]:
            cur, boxes = r[0], []
        if boxes and _iou(r[2:6], np.array(boxes)).max() >= NMS_IOU:
            continue
        boxes.append(r[2:6])
        final.append(r)
    ex = np.array(final)
    return ex[np.lexsort((ex[:, 1], ex[:, 0]))]


def load_extra_boxes(tracks_dir: Path | str) -> np.ndarray:
    """`extra_boxes` of one camera's tracks directory (tracks/<video id>/: raw_tracks.txt,
    boxes.txt, meta.json); empty when a file is missing or unreadable (never raises)."""
    d = Path(tracks_dir)
    try:
        raw = np.loadtxt(d / "raw_tracks.txt", delimiter=",", ndmin=2)
        kept = np.loadtxt(d / "boxes.txt", delimiter=",", ndmin=2)
        fps = float(json.loads((d / "meta.json").read_text(encoding="utf-8"))["fps"])
    except (OSError, ValueError, KeyError, TypeError):
        return np.zeros((0, 7))
    if raw.size == 0 or kept.size == 0 or not fps > 0:
        return np.zeros((0, 7))
    return extra_boxes(raw, kept, fps)


def player_rows(ex: np.ndarray, step: int = 1) -> list[list[int]]:
    """[[frame, x, y, w, h], ...] of every `step`-th frame, rounded, for a page's JSON."""
    return [[int(r[0]), round(float(r[2])), round(float(r[3])), round(float(r[4])),
             round(float(r[5]))] for r in ex if int(r[0]) % max(1, step) == 0]
