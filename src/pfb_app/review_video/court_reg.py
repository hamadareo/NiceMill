"""The court's outline (four corners) in the picture, for a panning / zooming camera.

A person looks at ONE picture of the video (or a few) and registers where the court's outer
edge is; the rest is followed by the machine. This is a registration, not an estimate from the
floor: the file says which line each corner belongs to and the page says "人が登録・動きは追跡".

How the outline follows the camera: a broadcast camera pans, tilts and zooms from a fixed spot,
so every picture is a plane homography of every other one (whatever the depth of what is seen).
Local features (SIFT) of consecutive samples are matched, a homography is fitted with RANSAC and
the chain is anchored to earlier "key" pictures (loop closure: the camera returns to the same
views) so the error does not pile up. A sample whose match is weak (few inliers, a wild
homography, a cut to another camera) gets NO outline: a wrong line is never drawn.

Registration file (JSON):
  {"ref_t": 20.0, "note": "...",
   "corners_ref": [[x, y], ...4...],         # in the reference picture's pixels (may lie outside it)
   "basis": ["far sideline: ...", ...]}      # which line each corner is, and the pictures they were read from
The corners are ordered around the court (far-left, far-right, near-right, near-left in the picture of
the reference).  Pure functions (no video needed) are separate from the video-reading ones so they can be tested.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_S = 0.5
WORK_W = 960
MASK_BOTTOM = 0.88            # below this share of the height: the score bug / captions, not matched
MIN_INLIERS = 40
MIN_INLIER_SHARE = 0.35
MAX_SCALE_STEP = 1.6          # a step between neighbours that scales / skews more than this is a cut
MAX_RMS_DRIFT = 6.0           # (working px) disagreement between the chain and a direct match to a key picture
NOTE = "登録"
COURT_REG_NOTE = ("赤い線「コートの枠（登録）」は、公開映像の1枚の画面で、人がコートの外周（サイドライン・エンドライン）の位置を登録し、"
                  "カメラの動き（パン・ズーム）は画像の特徴点の対応から機械が追跡して動かしたものです（推定）。追跡が確かでない時刻は、線を出していません。"
                  "線がずれる場面があります。外周の外に出ている辺や角は、画面の外になるので見える部分だけが出ます。")


def validate_registration(reg: dict) -> dict:
    """Raise ValueError on an unusable registration file, else return it."""
    if not isinstance(reg, dict):
        raise ValueError("registration must be an object")
    c = reg.get("corners_ref")
    if not (isinstance(c, list) and len(c) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in c)):
        raise ValueError("corners_ref must be 4 [x, y] points")
    if reg.get("ref_t") is None and reg.get("ref_frame") is None:
        raise ValueError("ref_t (seconds) or ref_frame is needed")
    pts = np.array(c, float)
    if not np.all(np.isfinite(pts)):
        raise ValueError("corners_ref must be finite")
    # a simple convex quadrilateral (no bow-tie): all cross products of consecutive edges have the same sign
    sg = []
    for i in range(4):
        a, b, d = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        sg.append(np.sign((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])))
    if len(set(sg)) != 1 or 0 in sg:
        raise ValueError("corners_ref must form a convex quadrilateral in order")
    return reg


def load_registration(path) -> dict:
    return validate_registration(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """Project Nx2 points by the 3x3 homography H; None when a point goes through infinity."""
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))] @ H.T
    if np.any(np.abs(p[:, 2]) < 1e-9):
        return None
    xy = p[:, :2] / p[:, 2:3]
    return xy if np.all(np.isfinite(xy)) else None


def sane(H: np.ndarray | None, w: float, h: float) -> bool:
    """A homography between two pictures of one panning camera is nearly a similarity: reject wild ones."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    H = H / H[2, 2]
    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    q = apply_h(H, box)
    if q is None:
        return False
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    s = area / (w * h)
    if not (1 / MAX_SCALE_STEP ** 2 <= s <= MAX_SCALE_STEP ** 2):
        return False
    # orientation kept (no mirror) and the picture stays a convex quadrilateral
    sg = []
    for i in range(4):
        a, b, d = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        sg.append((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]))
    return all(v > 0 for v in sg) or all(v < 0 for v in sg)


def quad_rows(corners_ref, h_to_ref: dict, w: float, h: float, limit: float = 1.0e5) -> list:
    """Rows [frame, x1, y1, ..., x4, y4] of the registered corners in each tracked picture.
    `h_to_ref[frame]` maps that picture's pixels to the reference picture's; here the inverse is used.
    The corners may lie far outside the picture (the court is bigger than what the camera shows); the
    straight edges between them are exact under a homography, so the visible part is right. A frame is
    skipped when a corner lies behind the camera (the sign of the homogeneous coordinate differs from
    the picture's own) or absurdly far (`limit` px): nothing is drawn rather than a wrong line."""
    out = []
    pts = np.asarray(corners_ref, float)
    ph = np.c_[pts, np.ones(4)]
    for f in sorted(h_to_ref):
        H = h_to_ref[f]
        if H is None:
            continue
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        s0 = (Hi @ (H @ np.array([w / 2.0, h / 2.0, 1.0])))[2]
        p = ph @ Hi.T
        if np.any(np.abs(p[:, 2]) < 1e-9) or np.any(p[:, 2] * s0 <= 0):
            continue
        q = p[:, :2] / p[:, 2:3]
        if not np.all(np.isfinite(q)) or np.any(np.abs(q) > limit):
            continue
        out.append([int(f)] + [round(float(v), 1) for v in q.reshape(-1)])
    return out


def register(video, fps: float, last_frame: int | None, reg: dict) -> dict | None:
    """The registered court followed over the video: {"frames": rows, "note", "text", "stats"} or None
    when the camera could not be followed at all."""
    reg = validate_registration(reg)
    ref_frame = int(reg["ref_frame"]) if reg.get("ref_frame") is not None else int(round(float(reg["ref_t"]) * (fps or 30.0))) + 1
    hs, w, h, st = track(video, fps, ref_frame, last_frame)
    rows = quad_rows(reg["corners_ref"], hs, w, h)
    if not rows:
        return None
    st = dict(st, rows=len(rows), samples_lost=sum(1 for v in hs.values() if v is None))
    return {"frames": rows, "note": "人が登録・動きは追跡", "text": COURT_REG_NOTE, "stats": st}


# --- who stands outside the court ---------------------------------------------------------------------
COURT_LEN_M, COURT_WID_M = 28.0, 15.0     # FIBA court
OUT_MARGIN_M = 1.0                         # beyond the line by this much (registration error + a step over the line)
OUT_MIN_S = 1.0                            # and for at least this long
IN_STAY_S = 2.0                            # a track that was INSIDE for this long is a player who stepped out
FOOT_EDGE_PX = 3                           # a box whose bottom touches the picture's edge: its feet are not seen
OUT_NOTE = ("「コート外（控え・観客かも）」は、足元がコートの外周の外（約1m以上）に1秒以上続いている人です。外にいても選手とは限りません"
            "（スローインで外に立つ選手もいます）。一度でもコートの中に2秒以上いた人は、外に出ても選手のままにしています。"
            "審判はコートの中にいるので、この手がかりでは分かりません（服の色などで推定）。")


def interp_quad(rows: list, frame: int):
    """The 4 corners at `frame` from the sample rows (sorted by frame): linear between two samples at most
    31 frames apart, else the nearest sample within 20 frames, else None."""
    if not rows:
        return None
    lo, hi = 0, len(rows)
    while lo < hi:
        m = (lo + hi) // 2
        if rows[m][0] < frame:
            lo = m + 1
        else:
            hi = m
    a = rows[lo - 1] if lo > 0 else None
    b = rows[lo] if lo < len(rows) else None
    if a and b and b[0] - a[0] <= 31:
        u = (frame - a[0]) / max(1, b[0] - a[0])
        return np.array([a[1 + i] + (b[1 + i] - a[1 + i]) * u for i in range(8)], float).reshape(4, 2)
    da = frame - a[0] if a else 1e9
    db = b[0] - frame if b else 1e9
    if da <= db and da <= 20:
        return np.array(a[1:9], float).reshape(4, 2)
    if db < da and db <= 20:
        return np.array(b[1:9], float).reshape(4, 2)
    return None


def plane_xy(quad: np.ndarray, pt) -> tuple[float, float] | None:
    """The point's place on the court in metres (x along the long side 0..28, y 0..15), from the quad
    (far-left, far-right, near-right, near-left) taken as the 28 x 15 m rectangle."""
    dst = np.array([[0, 0], [COURT_LEN_M, 0], [COURT_LEN_M, COURT_WID_M], [0, COURT_WID_M]], float)
    A = []
    for (x, y), (u, v) in zip(quad, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    try:
        _, _, vt = np.linalg.svd(np.array(A, float))
    except np.linalg.LinAlgError:
        return None
    Hm = vt[-1].reshape(3, 3)
    p = Hm @ np.array([pt[0], pt[1], 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def outside_m(xy) -> float:
    """How far outside the court (metres; negative: inside, the depth of the closest line)."""
    x, y = xy
    return max(-x, x - COURT_LEN_M, -y, y - COURT_WID_M)


def flag_rows(rows: list, court_rows: list, fps: float, vh: float, step: int,
              margin_m: float = OUT_MARGIN_M, min_s: float = OUT_MIN_S, stay_s: float = IN_STAY_S) -> list[int]:
    """0/1 per row of ONE person's track (rows [frame, x, y, w, h, ...], in frame order): 1 = standing outside
    the court for at least `min_s` seconds in a row (beyond `margin_m`), unless the same track stood inside
    for `stay_s` seconds or more (then it is a player who stepped out). Rows whose feet are not seen, or
    where the court's outline is not known, are never flagged and break a run."""
    n = len(rows)
    state = [None] * n            # outside distance in metres (None: not known)
    for i, r in enumerate(rows):
        if r[2] + r[4] >= vh - FOOT_EDGE_PX:
            continue
        q = interp_quad(court_rows, int(r[0]))
        if q is None:
            continue
        xy = plane_xy(q, (r[1] + r[3] / 2.0, r[2] + r[4]))
        if xy is not None and abs(xy[0]) < 200 and abs(xy[1]) < 200:
            state[i] = outside_m(xy)
    # seconds spent inside (a gap of more than two steps is not bridged)
    inside_s = 0.0
    for i in range(1, n):
        if state[i] is not None and state[i - 1] is not None and state[i] <= 0 and state[i - 1] <= 0 and rows[i][0] - rows[i - 1][0] <= 2 * step:
            inside_s += (rows[i][0] - rows[i - 1][0]) / (fps or 30.0)
    flags = [0] * n
    if inside_s >= stay_s:
        return flags
    i = 0
    while i < n:
        if state[i] is not None and state[i] > margin_m:
            j = i
            while j + 1 < n and state[j + 1] is not None and state[j + 1] > margin_m and rows[j + 1][0] - rows[j][0] <= 2 * step:
                j += 1
            if (rows[j][0] - rows[i][0]) / (fps or 30.0) + step / (fps or 30.0) >= min_s:
                for k in range(i, j + 1):
                    flags[k] = 1
            i = j + 1
        else:
            i += 1
    return flags


def _features(img, sift, mask):
    import cv2   # noqa: PLC0415
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return sift.detectAndCompute(g, mask)


def _match(fa, fb, matcher, ratio: float = 0.78):
    """H mapping b's pixels to a's pixels (RANSAC) and the inlier count / share, or (None, 0, 0)."""
    import cv2   # noqa: PLC0415
    ka, da = fa
    kb, db = fb
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None, 0, 0.0
    good = []
    for m in matcher.knnMatch(db, da, k=2):
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    if len(good) < 12:
        return None, 0, 0.0
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.5, maxIters=3000, confidence=0.999)
    if H is None:
        return None, 0, 0.0
    n = int(inl.sum())
    return H, n, n / len(good)


def track(video, fps: float, ref_frame: int, last_frame: int | None = None, sample_s: float = SAMPLE_S,
          progress=None) -> tuple[dict, float, float, dict]:
    """Follow the camera: ({frame: 3x3 H (frame's pixels -> the reference picture's pixels, in the video's own
    pixel size) or None}, width, height, stats). Frames are sampled every `sample_s`."""
    import cv2   # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}, 0.0, 0.0, {}
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    k = WORK_W / float(W0)
    wk, hk = WORK_W, int(round(H0 * k))
    S = np.diag([k, k, 1.0])           # video px -> working px
    Sinv = np.linalg.inv(S)
    total = int(last_frame or cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round((fps or 30.0) * sample_s)))
    frames = sorted(set(list(range(1, max(total, 2) + 1, step)) + [int(ref_frame)]))
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    mask = np.full((hk, wk), 255, np.uint8)
    mask[int(hk * MASK_BOTTOM):, :] = 0
    feats = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if not ok:
            continue
        feats[f] = _features(cv2.resize(img, (wk, hk), interpolation=cv2.INTER_AREA), sift, mask)
    cap.release()
    ref_frame = min(feats, key=lambda f: abs(f - ref_frame))
    Hw = {ref_frame: np.eye(3)}        # working px of the frame -> working px of the reference
    stats = {"samples": len(feats), "matched": 0, "lost": 0, "relinked": 0}
    keys = [ref_frame]
    fl = sorted(feats)
    ri = fl.index(ref_frame)
    # forward and backward from the reference, each picture chained to its neighbour, then to key pictures
    for order in (fl[ri + 1:], fl[:ri][::-1]):
        prev = ref_frame
        for f in order:
            best = None                    # (inliers, H_f_to_ref)
            cand = [prev] if prev in Hw and Hw[prev] is not None else []
            if cand:
                H, n, share = _match(feats[prev], feats[f], matcher)
                if H is not None and n >= MIN_INLIERS and share >= MIN_INLIER_SHARE and sane(H, wk, hk):
                    best = (n, Hw[prev] @ H)
            # loop closure: key pictures whose view overlaps this one's guess
            if best is not None:
                guess = apply_h(best[1], np.array([[wk / 2, hk / 2]]))
                near = []
                for kf in keys:
                    if kf == prev or Hw.get(kf) is None:
                        continue
                    c = apply_h(Hw[kf], np.array([[wk / 2, hk / 2]]))
                    if c is not None and guess is not None:
                        near.append((float(np.hypot(*(c[0] - guess[0]))), kf))
                near.sort()
                for _, kf in near[:3]:
                    H, n, share = _match(feats[kf], feats[f], matcher)
                    if H is not None and n >= max(MIN_INLIERS, 0.8 * best[0]) and share >= MIN_INLIER_SHARE and sane(H, wk, hk):
                        cand_h = Hw[kf] @ H
                        if n > best[0] * 0.9:
                            best = (n, cand_h)
                            stats["relinked"] += 1
                            break
            if best is None:
                # a hole: try a direct match to the key pictures (the camera may have cut away and come back)
                for kf in keys[::-1][:12]:
                    if Hw.get(kf) is None:
                        continue
                    H, n, share = _match(feats[kf], feats[f], matcher)
                    if H is not None and n >= 2 * MIN_INLIERS and share >= 0.5 and sane(H, wk, hk):
                        best = (n, Hw[kf] @ H)
                        break
            if best is None:
                Hw[f] = None
                stats["lost"] += 1
                continue                    # `prev` stays: the next picture is chained to the last good one
            Hw[f] = best[1]
            stats["matched"] += 1
            prev = f
            # a new key picture when it is farther than 40% of a screen from every key one
            c = apply_h(Hw[f], np.array([[wk / 2, hk / 2]]))
            if c is not None and all(Hw.get(kf) is None or np.hypot(*(apply_h(Hw[kf], np.array([[wk / 2, hk / 2]]))[0] - c[0])) > 0.35 * wk for kf in keys):
                keys.append(f)
            if progress and stats["matched"] % 50 == 0:
                progress(stats)
    out = {}
    for f, H in Hw.items():
        out[f] = None if H is None else Sinv @ H @ S
    stats["keys"] = len(keys)
    stats["ref_frame"] = ref_frame
    return out, float(W0), float(H0), stats
