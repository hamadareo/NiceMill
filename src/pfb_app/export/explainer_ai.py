"""「AIに見せた範囲」 of one cited moment: what the engine actually gave the VLM for it.

Read from PFB3x3 (pin 58f8341, 2026-10-24): a behavior window is n frames (`CONDITIONS`:
clip_som 12 frames 0.5 s apart, the app's default; static_* 5 frames 5 s apart), frame k of
the run being `game_start + round(k * step * fps)` (game_start = round(start_s * fps) + 1),
grouped by n (windows.make_windows). Every frame is a crop of the whole frame at ONE region per
window (overlay.action_region: all boxes of all its frames + 15 %, at least 40 % of the width,
16:9, inside the frame), resized, with the set-of-mark boxes drawn. The engine saves exactly
the JPEGs it sent as `<run>/overlay/<window_id>/<k>.jpg` (the behavior run's
`frames[event_id]`); the frame numbers and the region are written nowhere.

So `ai_input` returns the recorded stills as they are (they ARE the input), plus the window's
frames and the region recomputed with the engine's formulas (`window_frames`, `action_region`,
ported 1:1); `verify` checks a recomputed region against a recorded still (the source frame
cropped there, resized to the still, must correlate with it), and only a verified region is
ever drawn. Nothing here is an "attention" of the model: the app has no such data.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pfb_app.export.tracks import Box, BoxIndex

# (seconds between frames, frames per window): PFB3x3 windows.CONDITIONS
CONDITIONS = {"static_ref": (5.0, 5), "static_som": (5.0, 5), "clip_som": (0.5, 12)}
MARGIN, MIN_FRAC, ASPECT = 0.15, 0.4, 16 / 9      # overlay.action_region's defaults
VERIFY_MIN = 0.80          # normalised correlation of the crop with the recorded still
VERIFY_SIZE = (64, 36)     # compared at this size, in grey
VARIANT_MARGIN = 0.03      # ... and it must beat moved and resized regions by this much
SHIFT, SCALES = 0.12, (0.85, 1.15)   # the variants: moved by 12 % of the region, resized
MAX_COVER = 0.90           # a region of 90 % of the frame or more is never drawn
MIN_MATCHES = 2            # stills that must match before the region is drawn
_WINDOW = re.compile(r"_w(\d+)$")


@dataclass
class AiInput:
    stills: list[Path] = field(default_factory=list)   # the images sent, frame order
    frames: list[int] | None = None    # the window's 1-based video frames (recomputed)
    region: tuple[float, float, float, float] | None = None   # source px (recomputed)
    verified: bool = False             # the region matched a recorded still

    def still_at(self, frame: float) -> int | None:
        """The 0-based still nearest to video frame `frame` (None: no stills; without the
        window's frames, the first)."""
        if not self.stills:
            return None
        if not self.frames or len(self.frames) != len(self.stills):
            return 0
        return min(range(len(self.frames)), key=lambda i: abs(self.frames[i] - frame))


def window_frames(index: int, step_s: float, per_window: int, fps: float, start_s: float,
                  last_frame: int) -> list[int] | None:
    """The 1-based frames of window `index` (None: the window would not exist)."""
    if fps <= 0 or index < 0 or per_window <= 0:
        return None
    start = round(start_s * fps) + 1
    out = []
    for k in range(index * per_window, index * per_window + per_window):
        f = start + round(k * step_s * fps)
        if f > last_frame:
            break
        out.append(f)
    return out if len(out) >= 2 else None


def action_region(boxes: list[Box], src_w: int, src_h: int, margin: float = MARGIN,
                  min_frac: float = MIN_FRAC, aspect: float = ASPECT
                  ) -> tuple[float, float, float, float]:
    """PFB3x3 overlay.action_region: (x0, y0, x1, y1) in source px covering all boxes."""
    if not boxes:
        return (0.0, 0.0, float(src_w), float(src_h))
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.w for b in boxes)
    y1 = max(b.y + b.h for b in boxes)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(w, h * aspect, min_frac * src_w)
    h = w / aspect
    if w > src_w or h > src_h:
        f = min(src_w / w, src_h / h)
        w, h = w * f, h * f
    rx = min(max(cx - w / 2, 0.0), src_w - w)
    ry = min(max(cy - h / 2, 0.0), src_h - h)
    return (rx, ry, rx + w, ry + h)


def boxes_at(index: BoxIndex, frame: int) -> list[tuple[str, Box]]:
    """Every label's box in exactly this 1-based frame."""
    out = []
    for label in index.labels:
        frames, boxes = index.boxes(label)
        i = int(np.searchsorted(frames, frame))
        if i < len(frames) and int(frames[i]) == frame:
            out.append((label, Box(*(float(v) for v in boxes[i]))))
    return out


def ai_input(stills: list[str | Path], boxes: BoxIndex | None, fps: float | None,
             size: tuple[int, int] | None, start_s: float, last_frame: int | None) -> AiInput:
    """The recorded stills of one event (the behavior run's frames[event_id]) and, when the
    run's condition and the window are known, the window's frames and its region recomputed
    from the same boxes (not yet verified)."""
    found = [Path(p) for p in stills if Path(p).is_file()]
    out = AiInput(stills=found)
    if not found or boxes is None or not fps or not size or not last_frame:
        return out
    window = _WINDOW.search(found[0].parent.name)
    try:
        condition = found[0].parents[3].name     # <condition>/<video>/overlay/<window>/k.jpg
    except IndexError:
        return out
    if window is None or condition not in CONDITIONS:
        return out
    step, per = CONDITIONS[condition]
    frames = window_frames(int(window.group(1)), step, per, fps, start_s, last_frame)
    if frames is None or len(frames) != len(found):
        return out
    out.frames = frames
    out.region = action_region([b for f in frames for _, b in boxes_at(boxes, f)],
                               size[0], size[1])
    return out


def _grey(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, VERIFY_SIZE, interpolation=cv2.INTER_AREA).astype(np.float64)
    g -= g.mean()
    return g


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = math.sqrt(float((a * a).sum())), math.sqrt(float((b * b).sum()))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float((a * b).sum()) / (na * nb)


def _clip(r, fw, fh):
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    x0, y0 = min(max(x0, 0.0), fw - w), min(max(y0, 0.0), fh - h)
    return (x0, y0, x0 + w, y0 + h)


def _variants(r, fw, fh) -> list[tuple[float, float, float, float]]:
    """The region moved by SHIFT of its size each way and resized by SCALES (inside the frame;
    a variant that the frame's edge brings back onto the region is left out)."""
    x0, y0, x1, y1 = r
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = [_clip((x0 + dx, y0 + dy, x1 + dx, y1 + dy), fw, fh)
           for dx, dy in ((SHIFT * w, 0), (-SHIFT * w, 0), (0, SHIFT * h), (0, -SHIFT * h))]
    for k in SCALES:
        nw, nh = min(w * k, fw), min(h * k, fh)
        out.append(_clip((cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2), fw, fh))
    return [v for v in out if max(abs(a - b) for a, b in zip(v, r)) >= 0.04 * min(w, h)]


def _score(frame_bgr: np.ndarray, r, target: np.ndarray) -> float | None:
    fh, fw = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in r)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), fw), min(max(y1, y0 + 1), fh)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return _ncc(_grey(frame_bgr[y0:y1, x0:x1]), target)


def verify(frame_bgr: np.ndarray, region_px: tuple[float, float, float, float],
           still: Path) -> bool:
    """The decoded frame cropped at `region_px` (decoded pixels) is the recorded still: it
    correlates with it (VERIFY_MIN) and better, by VARIANT_MARGIN, than the same region moved
    or resized (so a flat or repetitive picture never "matches" anywhere)."""
    shot = cv2.imread(str(still))
    if shot is None or frame_bgr is None:
        return False
    fh, fw = frame_bgr.shape[:2]
    x0, y0, x1, y1 = region_px
    if y1 - y0 <= 0 or abs((x1 - x0) / (y1 - y0) - shot.shape[1] / shot.shape[0]) > 0.08:
        return False                            # not the still's shape
    target = _grey(shot)
    score = _score(frame_bgr, region_px, target)
    if score is None or score < VERIFY_MIN:
        return False
    others = [v for v in (_score(frame_bgr, r, target) for r in _variants(region_px, fw, fh))
              if v is not None]
    return all(score >= v + VARIANT_MARGIN for v in others)


def cover(region: tuple[float, float, float, float], size: tuple[int, int]) -> float:
    """The share of the frame the region covers."""
    x0, y0, x1, y1 = region
    return max(0.0, (x1 - x0) * (y1 - y0)) / max(1.0, float(size[0] * size[1]))


@dataclass(frozen=True)
class Check:
    drawn: bool            # the dashed rectangle may be drawn
    matches: int           # stills that matched
    cover: float           # the region's share of the frame (0: no region)


def check_region(info: AiInput, read, size: tuple[int, int], scale: tuple[float, float],
                 near: float) -> Check:
    """Whether the recomputed region may be drawn: it covers less than MAX_COVER of the frame
    and at least MIN_MATCHES of up to three stills (the one nearest frame `near`, the first
    and the last) match their frames (`verify`). `read(frame)` returns that 1-based video
    frame decoded (BGR, or None); `scale` maps source px to its pixels."""
    if info.region is None or not info.frames:
        return Check(False, 0, 0.0)
    share = cover(info.region, size)
    if share >= MAX_COVER:
        return Check(False, 0, share)
    sx, sy = scale
    x0, y0, x1, y1 = info.region
    region_px = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
    picks = list(dict.fromkeys([info.still_at(near), 0, len(info.frames) - 1]))
    matches = 0
    for k in picks:
        frame = read(info.frames[k])
        if frame is not None and verify(frame, region_px, info.stills[k]):
            matches += 1
    return Check(matches >= MIN_MATCHES, matches, share)
