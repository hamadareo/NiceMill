"""What the tracking engine recorded about a person, read for the PDFs (read only).

- `load_boxes`: the engine's MOT boxes.txt (`frame,track_id,x,y,w,h,conf`, 1-based video
  frames, label P{track_id}) as a per-label index, cached by file size and mtime;
- `track_meta`: tracks/<id>/meta.json next to it (fps, n_frames, width, height);
- `spans`: the parts of the analysed time a person is boxed in;
- `load_geometry`: the court homography the engine's line-call run wrote
  (tracks/<id>/linecall/geometry.json, else the track stage's tracks/<id>/court/geometry.json:
  image px -> court cm, TID court frame; per-frame
  H_frames and a radial lens when it has them). No file, or an H that is not a finite,
  invertible 3x3, is no geometry: then there is no court diagram (never a guessed one);
- `court_samples` / `clean_samples` / `distance_m`: the person's foot point on the court,
  without points more than 1 m outside it and median-smoothed over about 1 s, and the
  distance walked between samples (steps faster than a sprint, or across a gap, are not
  counted; no distance from less than 30 s on the court).
"""
from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pfb_app.export.charts import Span

GEOMETRY = Path("linecall") / "geometry.json"
COURT_GEOMETRY = Path("court") / "geometry.json"   # the track stage's court (engine autocourt)
MAX_SPEED_MS = 9.0      # m/s: anything faster between two samples is a track swap
MAX_GAP_S = 2.0         # a longer gap between samples is not walked
MIN_ON_COURT_S = 30.0   # no distance from less than this much time on the court
OFF_COURT_CM = 100.0    # samples further outside the court than this are dropped
SMOOTH_S = 1.0          # the median window of the court samples
COURT_W, COURT_H = 1505.0, 1105.0   # the 3x3 half court (cm), as charts / linecall.court


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class TrackMeta:
    fps: float
    n_frames: int
    width: int | None
    height: int | None


class BoxIndex:
    """Boxes per label: frames sorted, with their (x, y, w, h)."""

    def __init__(self, rows: dict[str, list[tuple[int, float, float, float, float]]]):
        self._frames: dict[str, np.ndarray] = {}
        self._boxes: dict[str, np.ndarray] = {}
        for label, items in rows.items():
            items.sort()
            arr = np.asarray(items, dtype=float)
            self._frames[label] = arr[:, 0].astype(int)
            self._boxes[label] = arr[:, 1:5]

    @property
    def labels(self) -> list[str]:
        return list(self._frames)

    def frames(self, labels: list[str]) -> np.ndarray:
        parts = [self._frames[lb] for lb in labels if lb in self._frames]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int)

    def boxes(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        return self._frames.get(label, np.zeros(0, int)), self._boxes.get(label, np.zeros((0, 4)))

    def nearest(self, labels: list[str], frame: int, tol: int) -> tuple[int, Box] | None:
        """The box of any of `labels` closest to `frame`, within `tol` frames."""
        best: tuple[int, int, Box] | None = None
        for lb in labels:
            fr = self._frames.get(lb)
            if fr is None or not len(fr):
                continue
            i = int(np.searchsorted(fr, frame))
            for j in (i - 1, i):
                if 0 <= j < len(fr):
                    d = abs(int(fr[j]) - frame)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, int(fr[j]), Box(*(float(v) for v in self._boxes[lb][j])))
        return None if best is None else (best[1], best[2])


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


@functools.lru_cache(maxsize=16)
def _load_boxes(path: str, stamp: tuple[int, int]) -> BoxIndex | None:
    rows: dict[str, list] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split(",")
                try:
                    frame, tid = int(float(parts[0])), int(float(parts[1]))
                    box = tuple(float(v) for v in parts[2:6])
                except (ValueError, IndexError):
                    continue
                if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= 0 \
                        or box[3] <= 0:
                    continue
                rows.setdefault(f"P{tid}", []).append((frame, *box))
    except (OSError, UnicodeDecodeError):
        return None
    return BoxIndex(rows)


def load_boxes(path: Path | str | None) -> BoxIndex | None:
    if not path:
        return None
    stamp = _stamp(Path(path))
    return None if stamp is None else _load_boxes(str(path), stamp)


def track_meta(boxes_path: Path | str | None) -> TrackMeta | None:
    if not boxes_path:
        return None
    try:
        info = json.loads((Path(boxes_path).parent / "meta.json").read_text(encoding="utf-8"))
        fps, n = float(info["fps"]), int(info["n_frames"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (math.isfinite(fps) and fps > 0 and n > 0):
        return None
    w, h = info.get("width"), info.get("height")
    return TrackMeta(fps, n, int(w) if w else None, int(h) if h else None)


def spans(frames: np.ndarray, fps: float, *, start_s: float, duration_s: float,
          gap_s: float = 1.5) -> list[Span]:
    """The on-screen parts, in seconds since the analysed start, clipped to [0, duration_s];
    boxes less than `gap_s` apart are one part."""
    if not len(frames) or fps <= 0 or duration_s <= 0:
        return []
    t = (np.asarray(frames, dtype=float) - 1) / fps - start_s
    out: list[Span] = []
    a = b = float(t[0])
    for v in t[1:]:
        v = float(v)
        if v - b > gap_s:
            out.append(Span(a, b))
            a = v
        b = v
    out.append(Span(a, b))
    clipped = [Span(max(s.start_s, 0.0), min(s.end_s, duration_s)) for s in out]
    return [s for s in clipped if s.end_s >= s.start_s and s.end_s >= 0
            and s.start_s <= duration_s]


# --- court geometry -----------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    H: np.ndarray                  # image px (undistorted when lens) -> court cm
    lens: dict | None
    H_frames: np.ndarray | None    # (n, 3, 3), frame k at k - 1
    source: str


def _valid_H(H) -> np.ndarray | None:
    try:
        H = np.asarray(H, dtype=float)
    except (TypeError, ValueError):
        return None
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or H[2, 2] == 0:
        return None
    if abs(np.linalg.det(H)) < 1e-12:
        return None
    return H / H[2, 2]


def load_geometry(boxes_path: Path | str | None) -> Geometry | None:
    """The line-call run's geometry next to the boxes, else the track stage's court
    (tracks/<id>/court/geometry.json, written for basketball by the engine tracker)."""
    if not boxes_path:
        return None
    d = Path(boxes_path).parent
    return load_geometry_file(d / GEOMETRY) or load_geometry_file(d / COURT_GEOMETRY)


def load_geometry_file(f: Path) -> Geometry | None:
    """One geometry.json (as load_geometry; H_frames relative to its folder)."""
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    H = _valid_H(d.get("H"))
    if H is None:
        return None
    lens = d.get("lens") if isinstance(d.get("lens"), dict) else None
    if lens is not None:
        try:
            lens = {k: float(lens[k]) for k in ("f", "cx", "cy")} | {
                k: float(lens.get(k) or 0.0) for k in ("k1", "k2")}
        except (KeyError, TypeError, ValueError):
            lens = None
    frames = None
    if d.get("H_frames"):
        npy = Path(str(d["H_frames"]))
        if not npy.is_absolute():
            npy = f.parent / npy              # written relative to the geometry file
        try:
            arr = np.load(str(npy), mmap_mode="r")
            if arr.ndim == 3 and arr.shape[1:] == (3, 3) and len(arr):
                frames = arr
        except (OSError, ValueError):
            frames = None
    return Geometry(H, lens, frames, str(d.get("source") or ""))


def _undistort(uv: np.ndarray, cam: dict | None, iters: int = 20) -> np.ndarray:
    """Distorted -> undistorted px (radial k1, k2; Newton on the radius), as the engine's
    linecall.sources.undistort."""
    if not cam or (cam["k1"] == 0 and cam["k2"] == 0):
        return uv
    k1, k2 = cam["k1"], cam["k2"]
    xd, yd = (uv[:, 0] - cam["cx"]) / cam["f"], (uv[:, 1] - cam["cy"]) / cam["f"]
    rd = np.hypot(xd, yd)
    r = rd.copy()
    for _ in range(iters):
        r2 = r * r
        g = r * (1 + k1 * r2 + k2 * r2 * r2) - rd
        dg = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
        dg = np.where(np.abs(dg) < 1e-6, 1e-6, dg)
        r = np.maximum(r - g / dg, 0.0)
    scale = np.divide(r, rd, out=np.ones_like(rd), where=rd > 0)
    return np.c_[cam["cx"] + cam["f"] * xd * scale, cam["cy"] + cam["f"] * yd * scale]


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def court_samples(index: BoxIndex, labels: list[str], geom: Geometry, fps: float, *,
                  step_s: float = 0.5) -> list[tuple[int, float, float]]:
    """(frame, x cm, y cm) of the person's foot point (box bottom centre), one sample per
    `step_s` at most; non-finite projections are left out."""
    step = max(1, round(step_s * fps))
    picked: dict[int, tuple[float, float]] = {}
    for lb in labels:
        frames, boxes = index.boxes(lb)
        last = -10 ** 9
        for f, (x, y, w, h) in zip(frames, boxes):
            if f - last >= step and int(f) not in picked:
                picked[int(f)] = (x + w / 2, y + h)
                last = f
    if not picked:
        return []
    order = sorted(picked)
    uv = _undistort(np.asarray([picked[f] for f in order], dtype=float), geom.lens)
    out = []
    for f, p in zip(order, uv):
        if geom.H_frames is not None:
            k = min(max(f, 1), len(geom.H_frames)) - 1
            H = _valid_H(geom.H_frames[k])
            if H is None:
                continue
        else:
            H = geom.H
        x, y = _apply(H, p[None, :])[0]
        if math.isfinite(x) and math.isfinite(y):
            out.append((f, float(x), float(y)))
    return out


def clean_samples(samples: list[tuple[int, float, float]], fps: float, *,
                  window_s: float = SMOOTH_S) -> list[tuple[int, float, float]]:
    """The samples without those more than OFF_COURT_CM outside the court, each replaced by
    the median of the samples within `window_s` around it (one wild sample is not a walk)."""
    kept = [s for s in samples
            if -OFF_COURT_CM <= s[1] <= COURT_W + OFF_COURT_CM
            and -OFF_COURT_CM <= s[2] <= COURT_H + OFF_COURT_CM]
    if not kept or fps <= 0:
        return kept
    half = window_s / 2 * fps
    frames = np.asarray([s[0] for s in kept], dtype=float)
    xs = np.asarray([s[1] for s in kept])
    ys = np.asarray([s[2] for s in kept])
    out = []
    for k, f in enumerate(frames):
        lo, hi = np.searchsorted(frames, f - half), np.searchsorted(frames, f + half, "right")
        out.append((kept[k][0], float(np.median(xs[lo:hi])), float(np.median(ys[lo:hi]))))
    return out


def on_court_s(samples: list[tuple[int, float, float]], fps: float) -> float:
    """Seconds covered by the samples (gaps longer than MAX_GAP_S not counted)."""
    if fps <= 0:
        return 0.0
    return sum(dt for dt in ((b[0] - a[0]) / fps for a, b in zip(samples, samples[1:]))
               if 0 < dt <= MAX_GAP_S)


def distance_m(samples: list[tuple[int, float, float]], fps: float) -> float | None:
    """Metres walked between consecutive (cleaned) samples; None with less than
    MIN_ON_COURT_S of samples."""
    if fps <= 0 or on_court_s(samples, fps) < MIN_ON_COURT_S:
        return None
    total = 0.0
    for (f0, x0, y0), (f1, x1, y1) in zip(samples, samples[1:]):
        dt = (f1 - f0) / fps
        if dt <= 0 or dt > MAX_GAP_S:
            continue
        step = math.hypot(x1 - x0, y1 - y0) / 100.0
        if step / dt <= MAX_SPEED_MS:
            total += step
    return total
