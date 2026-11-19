"""The court's long sides (sidelines), estimated from the picture itself (OpenCV only).

Used when the submission has no registered court (the usual case). Nothing is registered or
stored: the result goes into the review player's JSON and is drawn as a red line marked
「コートの枠（推定）」.

How: the playing floor of a hall is wood-coloured and bounded by a darker painted strip (the
out-of-bounds apron) along each long side. For every column of a sampled frame the transitions
"painted strip above -> wood below" (the far side) and "wood above -> painted strip below" (the
near side) are collected; a robust line fit (RANSAC) per side gives the sideline. The frame's
own x-range of the supporting points (a little widened) is the visible length of the line.
Samples are every `SAMPLE_S` seconds because the camera pans; the player interpolates between them.

Honest limits (also said on screen): only the long sides that can be seen are drawn (the short
sides / corners are often outside the picture and are not guessed); a hall without a contrasting
strip, or a picture with strong glare, gives no line. A frame where the fit is not clear gives
no line for that side, and when too few frames give any line, no estimate at all
(`estimate` returns None): a wrong line is never drawn.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_S = 0.5
WORK_W = 640                  # the picture is analysed at this width
COL_STEP = 3                  # every 3rd column (of the working size)
MIN_STRIP = 3                 # the far strip is thin
MIN_RUN = 6                   # a painted strip / wood run must be this many px tall (working size)
FAR_STRIP_MAX = 26            # the far strip is thin; a big painted logo / key is not a strip
OUT_PX = 14                   # look this far beyond the strip
GAP_MAX = 6                   # a white line (a few px) may lie between the strip and the wood
MIN_INLIERS = 45              # columns that must agree on a side
MIN_SPAN = 0.40               # the supported part must span this share of the width
INLIER_PX = 3.0
MAX_SLOPE = 0.62              # |dy/dx| of a long side in the picture (about 32 degrees)
MIN_COVER = 0.25              # share of the samples that must give at least one line
NOTE = "推定"


def _masks(img):
    import cv2   # noqa: PLC0415
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    navy = ((h >= 100) & (h <= 128) & (s >= 80) & (v >= 25) & (v <= 150)).astype(np.uint8)
    wood = ((h >= 5) & (h <= 26) & (s >= 35) & (s <= 190) & (v >= 110)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(navy, cv2.MORPH_OPEN, k), cv2.morphologyEx(wood, cv2.MORPH_OPEN, k)


def _runs(col: np.ndarray):
    """(label, start, end) of the runs of a 1-D label array (0 other, 1 strip, 2 wood)."""
    out, start = [], 0
    for i in range(1, len(col) + 1):
        if i == len(col) or col[i] != col[start]:
            out.append((int(col[start]), start, i))
            start = i
    return out


def _transitions(navy: np.ndarray, wood: np.ndarray):
    """Points (x, y) where a painted strip meets wood: far (strip above, wood below) and near
    (wood above, strip below), in working-size pixels."""
    lab = np.zeros(navy.shape, np.uint8)
    lab[navy > 0] = 1
    lab[wood > 0] = 2
    far, near = [], []
    for x in range(0, lab.shape[1], COL_STEP):
        col = lab[:, x]
        runs = [r for r in _runs(col) if r[0] != 0 and r[2] - r[1] >= (MIN_STRIP if r[0] == 1 else MIN_RUN)]
        for a, b in zip(runs, runs[1:]):
            if b[1] - a[2] > GAP_MAX:
                continue
            y = (a[2] + b[1]) / 2.0
            if a[0] == 1 and b[0] == 2:
                above = col[max(0, a[1] - OUT_PX):a[1]]       # beyond the strip there must be no floor (a key's edge has)
                if len(above) >= 4 and (above == 2).mean() < 0.15 and a[2] - a[1] <= FAR_STRIP_MAX:
                    far.append((x, y))
            elif a[0] == 2 and b[0] == 1:
                below = col[b[2]:b[2] + OUT_PX]
                if len(below) >= 4 and (below == 2).mean() < 0.15:
                    near.append((x, y))
    return np.array(far, float).reshape(-1, 2), np.array(near, float).reshape(-1, 2)


def fit_line(pts: np.ndarray, rng: np.random.Generator, iters: int = 160):
    """Robust line y = a x + b through `pts`: (a, b, inlier mask) or None when no clear line."""
    n = len(pts)
    if n < MIN_INLIERS:
        return None
    best, best_in = None, None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        (x1, y1), (x2, y2) = pts[i], pts[j]
        if abs(x2 - x1) < 40:
            continue
        a = (y2 - y1) / (x2 - x1)
        if abs(a) > MAX_SLOPE:
            continue
        b = y1 - a * x1
        inl = np.abs(pts[:, 1] - (a * pts[:, 0] + b)) <= INLIER_PX
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (a, b), inl
    if best is None or best_in.sum() < MIN_INLIERS:
        return None
    a, b = np.polyfit(pts[best_in, 0], pts[best_in, 1], 1)
    inl = np.abs(pts[:, 1] - (a * pts[:, 0] + b)) <= INLIER_PX
    xs = pts[inl, 0]
    if inl.sum() < MIN_INLIERS or (xs.max() - xs.min()) < MIN_SPAN * WORK_W:
        return None
    return float(a), float(b), inl


def _agree(line, sign: int, feet: np.ndarray, k: float, inside: float = 0.9, tol_px: float = 10.0) -> bool:
    """Do the players (feet in the picture's pixels) all stand on the inner side of this line (working size )?"""
    if feet is None:
        return True
    x0, x1 = line[0] - 20, line[2] + 20
    mm = feet[(feet[:, 0] * k >= x0) & (feet[:, 0] * k <= x1)]
    if len(mm) < 3:
        return False
    ys = np.array([line[1] + (line[3] - line[1]) * (min(max(x * k, line[0]), line[2]) - line[0]) / (line[2] - line[0]) for x in mm[:, 0]])
    return float((sign * (mm[:, 1] * k - ys) >= -tol_px).mean()) >= inside


def detect(img, feet: np.ndarray | None = None) -> dict:
    """{"far": [x0, y0, x1, y1] | None, "near": ... } in the picture's own pixels."""
    import cv2   # noqa: PLC0415
    h0, w0 = img.shape[:2]
    k = WORK_W / float(w0)
    small = cv2.resize(img, (WORK_W, int(round(h0 * k))), interpolation=cv2.INTER_AREA)
    navy, wood = _masks(small)
    far, near = _transitions(navy, wood)
    rng = np.random.default_rng(7)
    out = {"far": None, "near": None}
    for name, pts in (("far", far), ("near", near)):
        cands, rest = [], pts
        for _ in range(4):        # several lines per side (a key's edge is one): the outermost wins
            f = fit_line(rest, rng)
            if f is None:
                break
            a, b, inl = f
            cands.append((a, b, rest[inl, 0]))
            rest = rest[~inl]
        if not cands:
            continue
        mid = WORK_W / 2.0
        sign = 1 if name == "far" else -1
        good = []
        for a_, b_, xs_ in cands:
            lo, hi = max(0.0, xs_.min() - 10), min(float(WORK_W), xs_.max() + 10)
            if _agree([lo, a_ * lo + b_, hi, a_ * hi + b_], sign, feet, k):
                good.append((a_, b_, xs_))
        if not good:
            continue
        # with players known, the side line is the nearest line that all of them stand inside of;
        # without, the outermost one
        pick = (max if name == "far" else min) if feet is not None else (min if name == "far" else max)
        a, b, xs = pick(good, key=lambda c: c[0] * mid + c[1])
        x0, x1 = max(0.0, xs.min() - 10), min(float(WORK_W), xs.max() + 10)
        out[name] = [round(x0 / k, 1), round((a * x0 + b) / k, 1), round(x1 / k, 1), round((a * x1 + b) / k, 1)]
    # the far side is above the near side where both exist (else one is a mistake: none is drawn)
    if out["far"] and out["near"]:
        xm = max(out["far"][0], out["near"][0]), min(out["far"][2], out["near"][2])
        if xm[1] > xm[0]:
            def y_at(l, x):
                return l[1] + (l[3] - l[1]) * (x - l[0]) / (l[2] - l[0])
            xc = (xm[0] + xm[1]) / 2
            if y_at(out["far"], xc) >= y_at(out["near"], xc) - 20:
                out = {"far": None, "near": None}
    return out


def _smooth(rows: list[list], key: int) -> None:
    """Drop a line whose height jumps against both neighbours (one bad frame), in place."""
    for i in range(1, len(rows) - 1):
        cur, a, b = rows[i][key], rows[i - 1][key], rows[i + 1][key]
        if not (cur and a and b):
            continue
        m = lambda l: (l[1] + l[3]) / 2.0
        if abs(m(cur) - m(a)) > 60 and abs(m(cur) - m(b)) > 60 and abs(m(a) - m(b)) < 60:
            rows[i][key] = None


def estimate(video, fps: float, last_frame: int | None = None, sample_s: float = SAMPLE_S,
             feet: np.ndarray | None = None, win_s: float = 0.7) -> dict | None:
    """The sidelines over the video: {"samples": [[frame, far, near], ...], "note", "far_share",
    "near_share"} where far/near are [x0, y0, x1, y1] or None; None when no clear line was found
    in enough frames (or the video cannot be read)."""
    import cv2   # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        total = int(last_frame or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round((fps or 30.0) * sample_s)))
        rows: list[list] = []
        for f in range(1, max(total, 2) + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, img = cap.read()
            if not ok:
                break
            fm = None
            if feet is not None:
                fm = feet[np.abs(feet[:, 0] - f) <= win_s * (fps or 30.0)][:, 1:3]
            d = detect(img, fm)
            rows.append([f, d["far"], d["near"]])
    except Exception:   # noqa: BLE001 - a picture that cannot be read: no estimate
        log.exception("court estimate failed")
        return None
    finally:
        cap.release()
    if not rows:
        return None
    _smooth(rows, 1)
    _smooth(rows, 2)
    got = sum(1 for r in rows if r[1] or r[2]) / len(rows)
    if got < MIN_COVER:
        return None
    return {"samples": rows, "note": NOTE, "far_share": round(sum(1 for r in rows if r[1]) / len(rows), 2),
            "near_share": round(sum(1 for r in rows if r[2]) / len(rows), 2)}


def _y_at(line, x: float) -> float:
    return line[1] + (line[3] - line[1]) * (x - line[0]) / (line[2] - line[0])


def validate(est: dict | None, feet: np.ndarray, fps: float, win_s: float = 0.7,
             inside: float = 0.88, tol_px: float = 14.0, min_feet: int = 3) -> dict | None:
    """Keep only the lines the players themselves agree with. `feet` rows (frame, x, y): the foot
    points of the people playing (listed players and unnamed detections in a team). Players stand
    on the court, so (nearly) all of them must lie below a far side line and above a near side line
    (where the line is in the picture); a line that most of them stand on the wrong side of is a
    mistake (a key's edge, a logo) and is dropped. A sample with fewer than `min_feet` players
    around is kept as it is only when it has no contradicting player. Returns the estimate with the
    surviving lines (None when too few remain)."""
    if not est:
        return None
    feet = np.asarray(feet, float).reshape(-1, 3)
    half = win_s * fps
    rows, dropped = [], 0
    for f, far, near in est["samples"]:
        m = feet[np.abs(feet[:, 0] - f) <= half] if len(feet) else feet
        new = []
        for line, sign in ((far, 1), (near, -1)):
            if line is None:
                new.append(None)
                continue
            x0, x1 = min(line[0], line[2]) - 20, max(line[0], line[2]) + 20
            mm = m[(m[:, 1] >= x0) & (m[:, 1] <= x1)] if len(m) else m
            if len(mm) < min_feet:
                new.append(None)       # nobody to confirm it: not drawn
                continue
            ys = np.array([_y_at(line, min(max(x, line[0]), line[2])) for x in mm[:, 1]])
            ok = (sign * (mm[:, 2] - ys) >= -tol_px).mean()      # far: feet below the line; near: above
            if ok >= inside:
                new.append(line)
            else:
                new.append(None)
                dropped += 1
        rows.append([f, new[0], new[1]])
    _smooth(rows, 1)
    _smooth(rows, 2)
    got = sum(1 for r in rows if r[1] or r[2]) / max(1, len(rows))
    if got < MIN_COVER:
        return None
    return {"samples": rows, "note": est["note"], "far_share": round(sum(1 for r in rows if r[1]) / len(rows), 2),
            "near_share": round(sum(1 for r in rows if r[2]) / len(rows), 2), "dropped": dropped}
