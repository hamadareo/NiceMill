"""The pictures of the PDFs: one frame of the local video per moment, cropped around the
person and marked (read only on the video; the files stay on this machine).

`make_pictures` takes PictureRequests (video, 1-based frame, the person's box), reads each
frame once (one capture per video, frames in order), crops about 2.5x the box (`crop_rect`:
the box's area times 2.5^2 as a 3:4 portrait, inside the frame), softens what is
outside the person, draws a white-and-blue outline around them and writes a JPEG of at most
LONG_SIDE px. The result is cached under `cache_dir` by the video's size and mtime, the
frame and the box, so a second PDF of the same run reads no video at all.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import math
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pfb_app.export.tracks import Box

__all__ = ["Box", "PictureRequest", "crop_rect", "data_uri", "make_pictures", "prune"]

log = logging.getLogger(__name__)

VERSION = 2             # bump when the drawing changes: old cache files are not reused
SCALE = 2.5
ASPECT = 3 / 4      # width / height of every picture
LONG_SIDE = 1200
MIN_LONG_SIDE = 720     # small crops are enlarged (smoothly) to print well
QUALITY = 85
FORWARD_S = 4.0         # read on instead of seeking when the next picture is this close
OUTLINE = (42, 120, 214)    # categorical slot 1 (#2a78d6)
VEIL = 0.28                 # how much the area outside the person is lifted toward white


@dataclass(frozen=True)
class PictureRequest:
    video: Path
    frame: int          # 1-based
    box: Box


def crop_rect(box: Box, frame_w: int, frame_h: int, scale: float = SCALE
              ) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the area of `scale` x the box, as a 3:4 portrait around the box's
    centre (every picture of a page has the same shape), made smaller if it does not fit
    and moved to stay inside the frame."""
    area = max(box.w, 1.0) * scale * max(box.h, 1.0) * scale
    w, h = math.sqrt(area * ASPECT), math.sqrt(area / ASPECT)
    fit = min(1.0, frame_w / w, frame_h / h)
    w, h = w * fit, h * fit
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    x0 = min(max(cx - w / 2, 0.0), frame_w - w)
    y0 = min(max(cy - h / 2, 0.0), frame_h - h)
    return (int(round(x0)), int(round(y0)), int(round(min(x0 + w, frame_w))),
            int(round(min(y0 + h, frame_h))))


def _key(req: PictureRequest) -> str | None:
    try:
        st = Path(req.video).stat()
    except OSError:
        return None
    blob = (f"{Path(req.video).resolve()}|{st.st_size}|{st.st_mtime_ns}|{req.frame}|"
            f"{req.box.x:.1f},{req.box.y:.1f},{req.box.w:.1f},{req.box.h:.1f}|v{VERSION}")
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _draw(frame_bgr: np.ndarray, box: Box) -> Image.Image:
    fh, fw = frame_bgr.shape[:2]
    x0, y0, x1, y1 = crop_rect(box, fw, fh)
    crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop)
    k = 1.0
    long = max(img.size)
    if long > LONG_SIDE:
        k = LONG_SIDE / long
    elif long < MIN_LONG_SIDE:
        k = MIN_LONG_SIDE / long
    if k != 1.0:
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))),
                         Image.LANCZOS)
    pad = 0.08 * max(box.w, box.h)
    bx0 = (box.x - pad - x0) * k
    by0 = (box.y - pad - y0) * k
    bx1 = (box.x + box.w + pad - x0) * k
    by1 = (box.y + box.h + pad - y0) * k
    bx0, by0 = max(bx0, 2), max(by0, 2)
    bx1, by1 = min(bx1, img.width - 3), min(by1, img.height - 3)
    # lift everything outside the person toward white: they stand out, others recede
    veil = Image.new("RGB", img.size, (255, 255, 255))
    mask = Image.new("L", img.size, round(255 * VEIL))
    ImageDraw.Draw(mask).rounded_rectangle((bx0, by0, bx1, by1), radius=12, fill=0)
    img = Image.composite(veil, img, mask)
    line = max(3, round(max(img.size) / 240))
    draw = ImageDraw.Draw(img)
    radius = max(8, round(min(bx1 - bx0, by1 - by0) * 0.12))
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=radius, outline=(255, 255, 255),
                           width=line * 3)
    draw.rounded_rectangle((bx0 + line, by0 + line, bx1 - line, by1 - line),
                           radius=max(radius - line, 2), outline=OUTLINE, width=line)
    return img


def _save(img: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.{os.getpid()}.tmp.jpg")
    try:
        img.save(tmp, "JPEG", quality=QUALITY, optimize=True)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def make_pictures(requests: list[PictureRequest], cache_dir: Path) -> dict[PictureRequest, Path]:
    """{request: JPEG} for every request that could be made (unreadable videos and frames
    beyond the end are left out, logged)."""
    out: dict[PictureRequest, Path] = {}
    todo: dict[Path, list[tuple[PictureRequest, Path]]] = {}
    for req in dict.fromkeys(requests):
        key = _key(req)
        if key is None:
            continue
        target = Path(cache_dir) / f"{key}.jpg"
        if target.is_file():
            out[req] = target
        else:
            todo.setdefault(Path(req.video), []).append((req, target))
    for video, items in todo.items():
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                log.warning("a video for the PDF pictures cannot be opened: %s", video.name)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ahead_max = round(FORWARD_S * fps)
            pos = None
            for req, target in sorted(items, key=lambda it: it[0].frame):
                want = req.frame - 1
                ok = True
                if pos is not None and 0 <= want - pos <= ahead_max:
                    # close ahead: decode on from here (a seek would decode again from the
                    # keyframe, which with long keyframe gaps is behind us)
                    for _ in range(want - pos):
                        if not cap.grab():
                            ok = False
                            break
                elif pos != want:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
                ok, frame = cap.read() if ok else (False, None)
                pos = want + 1 if ok else None
                if not ok or frame is None:
                    continue
                try:
                    _save(_draw(frame, req.box), target)
                except (OSError, ValueError) as e:
                    log.warning("a PDF picture could not be written: %s", e)
                    continue
                out[req] = target
        finally:
            cap.release()
    return out


def prune(cache_dir: Path, keep: set[Path]) -> int:
    """Remove the cached pictures of `cache_dir` that are not in `keep` (the pictures of the
    current runs); files being written (".…tmp.jpg") are left alone. Returns how many went."""
    gone = 0
    keep = {Path(p).resolve() for p in keep}
    for f in Path(cache_dir).glob("*.jpg"):
        if f.name.startswith(".") or f.resolve() in keep:
            continue
        f.unlink(missing_ok=True)
        gone += 1
    return gone


def data_uri(path: Path) -> str:
    """The file as a data: URI (the share page carries its images inside)."""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
