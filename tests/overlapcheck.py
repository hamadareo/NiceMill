"""Mechanical layout audit of a PDF: text hidden by or sitting on a big shape, big flat blobs.

Occupancy and margin checks stay high when a page is broken (a stray filled ellipse over a table
passed both). This reads the PDF's drawing objects (pypdfium2) and its raster:

- shape_over_text   an opaque filled shape drawn AFTER a text object covers >= 60% of its box;
- low_contrast      text whose colour is too close to what is drawn behind it in the raster (white
                    text on white, text under a flat fill): contrast ratio < MIN_CONTRAST;
- flat_blob         a connected flat (single quantised colour), non-light region larger than
                    BLOB_MM2 outside the page's pictures.
Small badges (the numbered picture marks, 5 mm) are legitimate and stay below the limits.

CLI: python overlapcheck.py file.pdf [...]   -> prints hits, exit status 1 when any.
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as raw

BIG_PT2 = 600.0            # ~ 212 mm2
BLOB_MM2 = 2500.0
COVER = 0.60
MIN_CONTRAST = 2.5
MM = 72 / 25.4


@dataclass(frozen=True)
class Hit:
    page: int
    kind: str
    detail: str

    def __str__(self):
        return f"p{self.page} {self.kind}: {self.detail}"


def _color(o, getter):
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    ok = getter(o.raw, r, g, b, a)
    return (r.value, g.value, b.value, a.value) if ok else None


def _filled(o) -> bool:
    fm, st = ctypes.c_int(), ctypes.c_int()
    raw.FPDFPath_GetDrawMode(o.raw, fm, st)
    return fm.value != 0


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _light(c, lim=225):
    return min(c[:3]) >= lim


def audit(pdf: bytes, *, raster: bool = True) -> list[Hit]:
    doc = pdfium.PdfDocument(pdf)
    hits: list[Hit] = []
    for pi in range(len(doc)):
        page = doc[pi]
        objs = list(page.get_objects())
        texts, shapes, images = [], [], []
        for z, o in enumerate(objs):
            b = o.get_bounds()
            if o.type == raw.FPDF_PAGEOBJ_TEXT:
                if _area(b) > 0:
                    texts.append((z, b, _color(o, raw.FPDFPageObj_GetFillColor) or (0, 0, 0, 255)))
            elif o.type == raw.FPDF_PAGEOBJ_PATH:
                c = _color(o, raw.FPDFPageObj_GetFillColor)
                if c and c[3] > 0 and _filled(o) and _area(b) > 0:
                    shapes.append((z, b, c))
            elif o.type == raw.FPDF_PAGEOBJ_IMAGE:
                images.append(b)
        for tz, tb, tc in texts:
            ta = _area(tb)
            for sz, sb, sc in shapes:
                ov = _inter(tb, sb) / ta
                if ov < COVER:
                    continue
                white = sc[0] >= 250 and sc[1] >= 250 and sc[2] >= 250
                if sz > tz and sc[3] == 255 and not white:
                    hits.append(Hit(pi + 1, "shape_over_text", f"shape {tuple(round(v) for v in sb)} rgb{sc[:3]}"))
        if raster:
            hits += _blobs(page, pi + 1, images)
            hits += _contrast(page, pi + 1, texts)
    return _dedupe(hits)


def _lum(c):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def _contrast(page, n, texts) -> list[Hit]:
    sc = 2.0
    img = np.array(page.render(scale=sc).to_pil().convert("RGB"))
    H, W = img.shape[:2]
    hits = []
    for _z, b, tc in texts:
        x0, x1 = int(max(b[0], 0) * sc), int(min(b[2], W / sc) * sc)
        y0, y1 = int(max(H - b[3] * sc, 0)), int(min(H - b[1] * sc, H))
        if x1 - x0 < 16 * sc or y1 - y0 < 6 * sc:     # a lone glyph or mark: its box is mostly ink
            continue
        m = int(4 * sc)                  # the background is read on a ring just outside the text box
        ex0, ex1, ey0, ey1 = max(x0 - m, 0), min(x1 + m, W), max(y0 - m, 0), min(y1 + m, H)
        ring = np.ones((ey1 - ey0, ex1 - ex0), bool)
        ring[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
        px = img[ey0:ey1, ex0:ex1][ring]
        if len(px) < 20:
            continue
        bg = np.median(px, axis=0)
        lt, lb = _lum(tc), _lum(bg)
        ratio = (max(lt, lb) + 0.05) / (min(lt, lb) + 0.05)
        if ratio < MIN_CONTRAST:
            hits.append(Hit(n, "low_contrast", f"text box {tuple(round(v) for v in b)} text rgb{tc[:3]} on rgb{tuple(int(v) for v in bg)} ratio {ratio:.1f}"))
    return hits


def _dedupe(hits):
    seen, out = set(), []
    for h in hits:
        k = (h.page, h.kind, h.detail)
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def _blobs(page, n, images) -> list[Hit]:
    import cv2   # noqa: PLC0415
    sc = 1.0
    img = np.array(page.render(scale=sc).to_pil().convert("RGB"))
    H, W = img.shape[:2]
    for b in images:      # pictures are not flat shapes: leave them out
        x0, x1 = int(max(b[0], 0) * sc), int(min(b[2], W) * sc)
        y0, y1 = int(max(H - b[3] * sc, 0)), int(min(H - b[1] * sc, H))
        img[y0:y1, x0:x1] = 255
    q = (img >> 3).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    light = img.min(axis=2) >= 225
    hits = []
    vals, counts = np.unique(key[~light], return_counts=True)
    for v, cnt in zip(vals, counts):
        if cnt / (sc * sc) < BLOB_MM2 * MM * MM:
            continue
        mask = (key == v).astype(np.uint8)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, num):
            area_mm2 = stats[i, cv2.CC_STAT_AREA] / (sc * sc) / MM / MM
            if area_mm2 > BLOB_MM2:
                x, y, w, h = (int(stats[i, k]) for k in range(4))
                hits.append(Hit(n, "flat_blob", f"{area_mm2:.0f}mm2 at ({x},{y},{w},{h})"))
    return hits


if __name__ == "__main__":
    bad = 0
    for f in sys.argv[1:]:
        hs = audit(open(f, "rb").read())
        print(f, "pages", len(pdfium.PdfDocument(f)), "hits", len(hs))
        for h in hs:
            print("  ", h)
        bad += len(hs)
    sys.exit(1 if bad else 0)
