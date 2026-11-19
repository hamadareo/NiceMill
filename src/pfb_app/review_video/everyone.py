"""Everyone on the court, in the colour of their kit, with a short "what they seem to be doing".

Everything here is computed from the picture and the tracking only (OpenCV + numpy), when the
review player is made; nothing is stored in the database and no cloud service is used.

* Kit colours: the median colour of the upper body (the central part of the box, between the
  shoulders and the waist) of every box, over the whole clip. The colour of a team is the median of
  its listed people. The frame colour of a team is that measured colour (a white kit is drawn
  white, a red kit red); it is only corrected where it could not be seen (very dark -> near black,
  grey -> mid grey) and, when the two kits are too alike to tell apart, the second team is moved
  to a clearly different colour (and the page says so). The colour name on the page ("白のチーム")
  comes from the measured colour and is an estimate from the picture, not a club's colour.
* Everybody: the unnamed detections (extras) and the tracks nobody listed get a team the same way
  (nearest kit colour, voted over their short track); a colour that is neither kit is not a
  player (referee, coach...): it is hidden unless "選手以外も表示" is on.
* What they seem to be doing: from the track (speed in body heights per second, the camera's own
  pan taken out by phase correlation of consecutive frames), from the body pose when a pose file is
  given (hands above the head, crouching) and from the ball holder segments when given. The words
  are hedged on purpose and never name a basketball action (shot, pass, goal): those exist only in
  scenes that passed the action check.
"""
from __future__ import annotations

import colorsys
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# activity codes (the page's words are ACT_WORDS[code]; 0 = not known: only the number is shown)
ACT_NONE, ACT_STAND, ACT_WALK, ACT_RUN, ACT_CROUCH, ACT_ARMS, ACT_BALL = range(7)
ACT_WORDS = ["", "立っているように見える", "歩いているように見える", "走っているように見える",
             "しゃがんでいるように見える", "腕を上げているように見える（推定）",
             "ボールを持っているように見える（推定）"]
ACT_SHORT = ["", "立っている", "歩いている", "走っている", "しゃがんでいる", "腕を上げている", "ボールを持っている"]
STAND_MAX, WALK_MAX = 0.35, 1.25       # body heights per second
TORSO_X, TORSO_Y = (0.25, 0.75), (0.18, 0.52)
MIN_BOX_H = 24
TEAM_THR = 18.0                        # Lab distance (L weighted 0.5) to a kit colour: still that team
SEP_MIN = 38.0                         # the two frame colours must differ at least this (Lab dE)
POSE_IOU = 0.4
MIN_CAND_ROWS = 5                      # a pose-model-only box must last about half a second

PALETTE = [("白", "#FFFFFF"), ("黒", "#141414"), ("青", "#1E5BD8"), ("黄", "#F2D400"), ("緑", "#1FA34A"),
           ("橙", "#F28C00"), ("水色", "#00B8D4"), ("ピンク", "#E0489E")]   # no red: red is the court's line


# --- colours ------------------------------------------------------------------------------------
def rgb_to_lab(rgb) -> np.ndarray:
    c = np.asarray(rgb, float).reshape(-1, 3) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]


def lab_dist(a, b, lw: float = 1.0) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(math.sqrt(lw * d[0] ** 2 + d[1] ** 2 + d[2] ** 2))


def hex_of(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(max(0, min(255, v)))) for v in rgb)


def rgb_of(hex_: str):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


def luminance(rgb) -> float:
    def lin(v):
        c = v / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def colour_name(rgb) -> str:
    """A plain Japanese colour word for a measured kit colour."""
    r, g, b = (v / 255.0 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.26:
        return "黒"
    if s < 0.17:
        return "白" if v >= 0.48 else ("灰色" if v >= 0.40 else "黒")
    if v < 0.40:
        return "紺" if (195 <= h <= 265 and s >= 0.45) else "黒"
    if h < 12 or h >= 345:
        return "ピンク" if (s < 0.45 and v > 0.75) else "赤"
    if h < 40:
        return "茶" if v < 0.6 else "橙"
    if h < 70:
        return "黄"
    if h < 165:
        return "緑"
    if h < 200:
        return "水色"
    if h < 255:
        return "青"
    if h < 300:
        return "紫"
    return "ピンク"


def frame_colour(rgb) -> tuple[str, str]:
    """(shown hex, colour name) of a measured kit colour: the measured colour itself, only made
    visible where the measurement cannot be (near-black, grey, white)."""
    name = colour_name(rgb)
    if name == "白":
        return "#FFFFFF", name
    if name == "黒":
        return "#141414", name
    if name == "灰色":
        return "#8E9196", name
    r, g, b = (v / 255.0 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s, v = max(s, 0.55), max(v, 0.50)         # a washed-out video colour: keep the hue, make it readable
    return hex_of([c * 255 for c in colorsys.hsv_to_rgb(h, s, v)]), name


def team_style(rgb_a, rgb_b) -> dict:
    """{"A": {hex, name, rim, measured}, "B": ..., "adjusted": bool}: the two teams' frame colours."""
    out, adjusted = {}, False
    ha, na = frame_colour(rgb_a)
    hb, nb = frame_colour(rgb_b)
    la, lb = rgb_to_lab(rgb_of(ha))[0], rgb_to_lab(rgb_of(hb))[0]
    if lab_dist(la, lb) < SEP_MIN or na == nb:
        best = None
        for name, hx in PALETTE:
            lc = rgb_to_lab(rgb_of(hx))[0]
            if lab_dist(lc, la) < 45:
                continue
            d = lab_dist(lc, rgb_to_lab(rgb_b)[0])
            if best is None or d < best[0]:
                best = (d, hx, name)
        if best is not None:
            hb, nb, adjusted = best[1], best[2], True
    for k, hx, nm, meas in (("A", ha, na, rgb_a), ("B", hb, nb, rgb_b)):
        out[k] = {"hex": hx, "name": nm, "rim": "#000000" if luminance(rgb_of(hx)) > 0.30 else "#FFFFFF",
                  "measured": hex_of(meas)}
    out["adjusted"] = adjusted
    return out


# --- the picture pass ------------------------------------------------------------------------------
def _torso(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, x1 = int(max(x + TORSO_X[0] * w, 0)), int(min(x + TORSO_X[1] * w, W))
    y0, y1 = int(max(y + TORSO_Y[0] * h, 0)), int(min(y + TORSO_Y[1] * h, H))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = img[y0:y1, x0:x1].reshape(-1, 3)
    if len(crop) > 400:
        crop = crop[:: len(crop) // 400]
    return np.median(crop[:, ::-1], axis=0)       # BGR -> RGB


def scan(video, step: int, boxes: dict[int, list]) -> tuple[dict, dict]:
    """One sequential pass over the video at the frames of `boxes` ({frame: [(key, x, y, w, h)]}):
    ({key: rgb}, {frame: cumulative scene shift (px, full size)}). A box that is too small gives no colour."""
    import cv2   # noqa: PLC0415
    cap = cv2.VideoCapture(str(video))
    cols: dict = {}
    shift: dict[int, tuple[float, float]] = {}
    if not cap.isOpened():
        return cols, shift
    try:
        prev, prev_f, cum, pos = None, None, np.zeros(2), 0
        win = None
        for f in sorted(boxes):
            if f - 1 < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                pos = f - 1
            while pos < f - 1:
                if not cap.grab():
                    return cols, shift
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                break
            small = cv2.cvtColor(cv2.resize(img, (320, max(2, int(img.shape[0] * 320 / img.shape[1])))), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None and prev.shape == small.shape and prev_f is not None and f - prev_f <= 2 * step:
                if win is None or win.shape != small.shape:
                    win = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
                (dx, dy), resp = cv2.phaseCorrelate(prev, small, win)
                if resp > 0.05 and abs(dx) < 80 and abs(dy) < 60:
                    cum = cum + np.array([dx, dy]) * (img.shape[1] / 320.0)
            prev, prev_f = small, f
            shift[f] = (float(cum[0]), float(cum[1]))
            for key, x, y, w, h in boxes[f]:
                if h < MIN_BOX_H:
                    continue
                c = _torso(img, x, y, w, h)
                if c is not None:
                    cols[key] = c
    finally:
        cap.release()
    return cols, shift


# --- linking, speed, activity -----------------------------------------------------------------------
def link_rows(rows: list, step: int, reach: float = 0.8) -> list[int]:
    """Track ids (0..) of unnamed detections [[frame, x, y, w, h], ...] (any order): the same thing
    in the frame before (or two before) is the nearest box within `reach` heights."""
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i][0])
    tid = [-1] * n
    last: dict[int, int] = {}
    nxt = 0
    i = 0
    while i < n:
        j = i
        f = rows[order[i]][0]
        while j < n and rows[order[j]][0] == f:
            j += 1
        cands = order[i:j]
        pairs = []
        for t, li in last.items():
            gap = f - rows[li][0]
            if not 0 < gap <= 2 * step:
                continue
            for c in cands:
                a, b = rows[li], rows[c]
                d = math.hypot(a[1] + a[3] / 2 - b[1] - b[3] / 2, a[2] + a[4] - b[2] - b[4]) / max(a[4], b[4], 1)
                if d <= reach * gap / step:
                    pairs.append((d, t, c))
        used_t, used_c = set(), set()
        for d, t, c in sorted(pairs):
            if t in used_t or c in used_c:
                continue
            used_t.add(t)
            used_c.add(c)
            tid[c] = t
            last[t] = c
        for c in cands:
            if tid[c] < 0:
                tid[c] = nxt
                last[nxt] = c
                nxt += 1
        i = j
    return tid


def speeds(rows: list, fps: float, step: int, shift: dict, k: int = 3) -> list:
    """Per row: ground speed in body heights per second (camera pan removed), None where the row
    has too little track around it. `rows` of ONE continuous thing, in frame order."""
    n = len(rows)
    pos = []
    for r in rows:
        s = shift.get(r[0])
        pos.append((r[1] + r[3] / 2 - (s[0] if s else 0.0), r[2] + r[4] - (s[1] if s else 0.0), s is not None))
    out = [None] * n
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if b - a < 2:
            continue
        df = rows[b][0] - rows[a][0]
        if df <= 0 or df > (2 * k + 2) * step or not (pos[a][2] and pos[b][2]):
            continue
        h = float(np.median([r[4] for r in rows[a:b + 1]]))
        if h < MIN_BOX_H:
            continue
        out[i] = math.hypot(pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]) / h / (df / fps)
    return out


def motion_code(v) -> int:
    if v is None:
        return ACT_NONE
    return ACT_STAND if v < STAND_MAX else (ACT_WALK if v < WALK_MAX else ACT_RUN)


def _mode(vals: list, i: int, r: int) -> int:
    seg = [v for v in vals[max(0, i - r): i + r + 1] if v]
    if not seg:
        return 0
    return max(set(seg), key=seg.count)


def activity(rows: list, fps: float, step: int, shift: dict, pose: list | None = None,
             holder: set | None = None) -> list[int]:
    """Activity code per row of one continuous thing: the ball (when the row is in a holder
    segment) > hands above the head > crouching (both from the pose file) > run / walk / stand
    (from the track). Smoothed over about 0.5 s so the words do not flicker."""
    n = len(rows)
    mot = [motion_code(v) for v in speeds(rows, fps, step, shift)]
    mot = [_mode(mot, i, 2) for i in range(n)]
    arms = [1 if (pose and pose[i] and pose[i] & 1) else 0 for i in range(n)]
    crouch = [1 if (pose and pose[i] and pose[i] & 2) else 0 for i in range(n)]
    out = []
    for i in range(n):
        a = sum(arms[max(0, i - 2): i + 3]) >= 3
        c = sum(crouch[max(0, i - 2): i + 3]) >= 3
        if holder and rows[i][0] in holder:
            out.append(ACT_BALL)
        elif a:
            out.append(ACT_ARMS)
        elif c:
            out.append(ACT_CROUCH)
        else:
            out.append(mot[i])
    return out


# --- pose file --------------------------------------------------------------------------------------
def load_pose(path) -> dict[int, list]:
    """{frame: [(x, y, w, h, flags)]} from a pose file (an external pose tool (not included)); {} when unreadable.
    flags: 1 = a wrist clearly above the head, 2 = crouching."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return {int(k): [tuple(v) for v in vs] for k, vs in (d.get("frames") or {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def _iou(a, b) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def pose_flags(rows: list, pose: dict) -> list:
    """Pose flags per row (None: no pose found for that box)."""
    out = []
    for r in rows:
        best, bf = POSE_IOU, None
        for d in pose.get(r[0], ()):
            v = _iou(r[1:5], d[:4])
            if v >= best:
                best, bf = v, int(d[4])
        out.append(bf)
    return out


def pose_candidates(pose: dict, existing: list, feet: list, step: int, fps: float) -> list:
    """Boxes the body-pose model saw that no tracked / unnamed box covers, standing inside the area the
    players themselves occupy (the box around the feet of the listed players and the unnamed boxes within
    +-2 s, widened by 0.8 of the person's height): [[frame, x, y, w, h], ...]. They are only candidates:
    `analyse` keeps one only when its kit colour is a team's and it lasts (a spectator or a coach is not)."""
    if not pose or not feet:
        return []
    by: dict[int, list] = {}
    for r in existing:
        by.setdefault(r[0], []).append(r)
    fa = np.asarray(feet, float).reshape(-1, 3)
    half = 2.0 * fps
    out = []
    for f, dets in pose.items():
        m = fa[np.abs(fa[:, 0] - f) <= half]
        if len(m) < 4:
            continue
        lo, hi = m[:, 1:3].min(axis=0), m[:, 1:3].max(axis=0)
        for d in dets:
            x, y, w, h = d[:4]
            if h < 40 or w <= 0:
                continue
            cx, fy, mg = x + w / 2, y + h, 0.8 * h
            if not (lo[0] - mg <= cx <= hi[0] + mg and lo[1] - mg <= fy <= hi[1] + mg):
                continue
            cover = False
            for r in by.get(f, ()):
                if _iou((x, y, w, h), r[1:5]) >= 0.25 or (x <= r[1] + r[3] / 2 <= x + w and y <= r[2] + r[4] / 2 <= y + h) \
                        or (r[1] <= cx <= r[1] + r[3] and r[2] <= y + h / 2 <= r[2] + r[4]):
                    cover = True
                    break
            if not cover:
                out.append([int(f), round(float(x)), round(float(y)), round(float(w)), round(float(h))])
    return out


# --- the whole thing --------------------------------------------------------------------------------
def analyse(video, fps: float, step: int, people: list[dict], tracks: dict, extras: list | None,
            holders: list | None = None, pose_path=None) -> dict | None:
    """What the player needs to show everyone (None when the picture cannot be read).

    people: the player's people dicts ("labels", "tm" team letter, "excluded");
    tracks: {label: [[frame, x, y, w, h], ...]}; extras: [[frame, x, y, w, h], ...].
    Returns {"style": team_style | None, "tracks": {label: [[f,x,y,w,h,act]]},
    "extras": [[f,x,y,w,h,team,act]], "others": {label: team}, "stats": {...}}."""
    extras = list(extras or [])
    listed = {lb: p for p in people for lb in p["labels"]}
    pose = load_pose(pose_path) if pose_path else {}
    n_real = len(extras)
    if pose:
        existing = [r for rows in tracks.values() for r in rows] + extras
        feet = [[r[0], r[1] + r[3] / 2, r[2] + r[4]] for lb, rows in tracks.items()
                if not (listed.get(lb) or {}).get("excluded") for r in rows]
        feet += [[r[0], r[1] + r[3] / 2, r[2] + r[4]] for r in extras]
        extras += pose_candidates(pose, existing, feet, step, fps)
    boxes: dict[int, list] = {}
    for lb, rows in tracks.items():
        for i, r in enumerate(rows):
            boxes.setdefault(r[0], []).append((("t", lb, i), r[1], r[2], r[3], r[4]))
    for i, r in enumerate(extras):
        boxes.setdefault(r[0], []).append((("e", i), r[1], r[2], r[3], r[4]))
    if not boxes:
        return None
    cols, shift = scan(video, step, boxes)
    if not cols:
        return None
    # team colours: the median over each team's listed people (a person's colour = median of its rows)
    per_team: dict[str, list] = {"A": [], "B": []}
    for lb, p in listed.items():
        if p.get("tm") in per_team and not p.get("excluded"):
            c = [cols[("t", lb, i)] for i in range(len(tracks.get(lb, []))) if ("t", lb, i) in cols]
            if c:
                per_team[p["tm"]].append(np.median(np.array(c), axis=0))
    style = None
    kit = {}
    if per_team["A"] and per_team["B"]:
        kit = {k: np.median(np.array(v), axis=0) for k, v in per_team.items()}
        style = team_style(kit["A"], kit["B"])
    labs = {k: rgb_to_lab(v)[0] for k, v in kit.items()}

    def team_of(rgb) -> int:      # 0: neither kit, 1: A, 2: B
        if not labs:
            return 0
        lab = rgb_to_lab(rgb)[0]
        da, db = lab_dist(lab, labs["A"], 0.5), lab_dist(lab, labs["B"], 0.5)
        if min(da, db) > TEAM_THR:
            return 0
        return 1 if da <= db else 2

    hold: dict[str, set] = {}
    for f0, f1, lb in holders or []:
        hold.setdefault(lb, set()).update(range(int(f0), int(f1) + 1))
    out_tracks, others = {}, {}
    for lb, rows in tracks.items():
        rows = sorted(rows)
        fl = pose_flags(rows, pose) if pose else None
        acts = activity(rows, fps, step, shift, fl, hold.get(lb))
        out_tracks[lb] = [r + [a] for r, a in zip(rows, acts)]
        if lb not in listed and kit:
            votes = [team_of(cols[("t", lb, i)]) for i in range(len(tracks[lb])) if ("t", lb, i) in cols]
            if votes:
                others[lb] = max(set(votes), key=votes.count)
    out_extras: list = []
    n_team = n_pose = 0
    if extras:
        tid = link_rows(extras, step)
        by: dict[int, list] = {}
        for i, t in enumerate(tid):
            by.setdefault(t, []).append(i)
        for t, idx in by.items():
            idx.sort(key=lambda i: extras[i][0])
            rows = [extras[i] for i in idx]
            votes = [team_of(cols[("e", i)]) for i in idx if ("e", i) in cols]
            tm = max(set(votes), key=votes.count) if votes else 0
            if all(i >= n_real for i in idx) and (tm == 0 or len(idx) < MIN_CAND_ROWS):
                continue          # a pose-model box nobody tracked: kept only as a team's player that lasts
            n_pose += sum(1 for i in idx if i >= n_real)
            fl = pose_flags(rows, pose) if pose else None
            acts = activity(rows, fps, step, shift, fl, None)
            for r, a in zip(rows, acts):
                out_extras.append(r + [tm, a])
        out_extras.sort(key=lambda r: (r[0], r[1]))
        n_team = sum(1 for r in out_extras if r[5])
    return {"style": style, "tracks": out_tracks, "extras": out_extras, "others": others,
            "stats": {"rows_with_colour": len(cols), "extras": len(out_extras), "extras_in_a_team": n_team,
                      "from_pose_only": n_pose,
                      "kit": {k: hex_of(v) for k, v in kit.items()}}}
