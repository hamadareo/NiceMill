"""Inline SVG charts for the PDFs, built by hand (pure functions, no plotting library).

Every function returns an SVG string (or "" when there is nothing to draw, so the caller
leaves the block out) that WeasyPrint draws as vector graphics:

- `bar_chart`: horizontal bars of one series in one colour, a direct value at each tip, the
  rows in the order given (category charts keep the categories' fixed order; summary charts
  pass rows sorted by value). A category row carries its marker shape beside the label.
- `timeline`: the analysed time as a strip, the parts where the person was on screen as a
  band, the good moments as markers above it (shape + colour of the category, a 2 px surface
  halo, never colour alone), clock ticks below, and a legend of words.
- `growth_grid`: 成長の記録's counts per evaluation point (rows, with their marker shape) and
  work (columns, by date) as a heatmap on one sequential hue, the count written in every cell
  (0 on the empty track), so the grid is its own table.
- `court`: a 3x3 half court (cm, TID court frame: y = 0 is the baseline under the basket)
  with 1 m cells on a one-hue sequential ramp (light = a short time, dark = a long time).

Design rules (dataviz): thin marks, 4 px rounded data ends square at the baseline, hairline
recessive axes, text in ink tokens and never in a series colour, an accessible <title> and
<desc> (the values as text) on every chart. Numbers are checked finite; a NaN is a bug in the
data and raises ValueError rather than being drawn.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from html import escape
from typing import Callable, Sequence

from pfb_app.export.labels import OTHER_SHAPE, SHAPES, Category

FONT = "Noto Sans CJK JP"
INK = "#23272e"
INK_2 = "#59606b"
RULE = "#cdd3db"
TRACK = "#eef1f5"
SURFACE = "#ffffff"
# ナイスミル 「A コートサイド」 (dataviz validator: BAR passes the categorical lightness band and
# chroma floor, RAMP the ordinal checks: one hue, monotone lightness, light end 2.07:1 on white)
BAR = "#2b5aa6"          # court blue (the navy's mid step): one series, one colour (6.7:1 on white)
BAND = "#cbd7ea"         # a pale court blue: the on-screen band under the moments' markers
LINE = "#7f8b9c"         # court lines (3.5:1 on white)
RAMP = ("#f2a173", "#e98350", "#de6327", "#b9400c", "#842a08")   # sequential ball orange
PLAIN = "#59606b"        # a moment without a category
# the growth grid's one-hue steps: RAMP 1 and 2 carry ink text (7.2 and 5.6:1), RAMP 4 and 5
# white text (5.5 and 9.1:1); RAMP 3 is skipped, neither text colour reaches 4.5:1 on it
GRID_STEPS = (RAMP[0], RAMP[1], RAMP[3], RAMP[4])
GRID_DARK_FROM = 2       # GRID_STEPS from this index on carry white text

# court geometry (cm), as pfb3x3/linecall/court.py (read only there)
CW, CH = 1505.0, 1105.0
BASKET = (752.5, 134.0)
R3, ARC_X = 674.5, 97.5
ARC_Y = BASKET[1] + math.sqrt(R3 ** 2 - (BASKET[0] - ARC_X) ** 2)
KEY_W, KEY_L, FT_R, RA_R, RIM_R = 490.0, 580.0, 180.0, 125.0, 22.5
BOARD_Y, BOARD_W = 120.0, 180.0
CELL_CM = 100.0


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    category: Category | None = None


@dataclass(frozen=True)
class Span:
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Mark:
    t_s: float
    category: Category | None
    number: int | None = None     # the picture number this moment has on the page


# --- primitives ---------------------------------------------------------------------------

def _f(x: float) -> str:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {x!r}")
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _text_w(text: str, size: float) -> float:
    """Rough advance width: CJK and full-width characters 1 em, the rest 0.6 em."""
    return sum(size * (1.0 if ord(c) >= 0x2E80 else 0.6) for c in text)


def _text(x: float, y: float, s: str, *, size: float = 9.5, fill: str = INK,
          anchor: str = "start", weight: str = "normal") -> str:
    bold = ' font-weight="bold"' if weight == "bold" else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-family="{FONT}" font-size="{_f(size)}" '
            f'fill="{fill}" text-anchor="{anchor}"{bold}>{escape(s)}</text>')


def _svg(width: float, height: float, title: str, desc: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{escape(title)}" width="{_f(width)}" height="{_f(height)}" '
            f'viewBox="0 0 {_f(width)} {_f(height)}">'
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>{"".join(body)}</svg>')


def _poly(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in points)


def _star(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.45
        a = -math.pi / 2 + k * math.pi / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _shape_el(shape: str, cx: float, cy: float, r: float, fill: str, *,
              stroke: str | None = None, width: float = 0.0) -> str:
    """One marker element of `shape`, centred at (cx, cy) with radius `r`."""
    paint = f'fill="{fill}"' + (f' stroke="{stroke}" stroke-width="{_f(width)}" '
                                f'stroke-linejoin="round"' if stroke else "")
    if shape in ("circle", "ring"):
        return f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" {paint}/>'
    if shape == "square":
        s = r * 0.86
        return (f'<rect x="{_f(cx - s)}" y="{_f(cy - s)}" width="{_f(2 * s)}" '
                f'height="{_f(2 * s)}" rx="1" {paint}/>')
    if shape == "plus":
        t = r * 0.42
        pts = [(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t), (cx + r, cy - t),
               (cx + r, cy + t), (cx + t, cy + t), (cx + t, cy + r), (cx - t, cy + r),
               (cx - t, cy + t), (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t)]
    elif shape == "triangle":
        pts = [(cx, cy - r * 1.1), (cx + r * 1.05, cy + r * 0.8), (cx - r * 1.05, cy + r * 0.8)]
    elif shape == "triangle_down":
        pts = [(cx, cy + r * 1.1), (cx + r * 1.05, cy - r * 0.8), (cx - r * 1.05, cy - r * 0.8)]
    elif shape == "diamond":
        pts = [(cx, cy - r * 1.2), (cx + r * 1.0, cy), (cx, cy + r * 1.2), (cx - r * 1.0, cy)]
    elif shape == "hexagon":
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a))
               for a in (math.pi / 6 + k * math.pi / 3 for k in range(6))]
    elif shape == "star":
        pts = _star(cx, cy, r * 1.25)
    elif shape == "pentagon":
        pts = [(cx + r * 1.08 * math.cos(a), cy + r * 1.08 * math.sin(a))
               for a in (-math.pi / 2 + k * 2 * math.pi / 5 for k in range(5))]
    elif shape == "cross":     # the plus turned 45 degrees
        t, c45 = r * 0.4, math.sqrt(0.5)
        arm = [(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r),
               (-t, t), (-r, t), (-r, -t), (-t, -t)]
        pts = [(cx + (x - y) * c45, cy + (x + y) * c45) for x, y in arm]
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return f'<polygon points="{_poly(pts)}" {paint}/>'


def _marker(shape: str, colour: str, cx: float, cy: float, r: float, *,
            cls: str | None = None) -> str:
    """A marker with its 2 px surface halo (legible where marks touch), as one group that
    carries the shape name."""
    attrs = f' class="{cls}"' if cls else ""
    halo = _shape_el("circle" if shape == "ring" else shape, cx, cy, r + 2, SURFACE,
                     stroke=SURFACE, width=2)
    if shape == "ring":
        mark = _shape_el("ring", cx, cy, r - 1, SURFACE, stroke=colour, width=2.4)
    else:
        mark = _shape_el(shape, cx, cy, r, colour)
    return (f'<g data-shape="{shape}" data-x="{_f(cx)}" data-y="{_f(cy)}"{attrs}>'
            f'{halo}{mark}</g>')


def _shape_of(category: Category | None) -> tuple[str, str]:
    return (category.shape, category.colour) if category else ("plus", PLAIN)


def glyph(category: Category | None, size: float = 12) -> str:
    """The category's marker as a small standalone SVG (for legends and lists in HTML)."""
    shape, colour = _shape_of(category)
    r = size / 2 - 2
    label = category.label if category else "よかった場面"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(label)}" '
            f'class="glyph" width="{_f(size)}" height="{_f(size)}" '
            f'viewBox="0 0 {_f(size)} {_f(size)}"><title>{escape(label)}</title>'
            f'{_marker(shape, colour, size / 2, size / 2, r)}</svg>')


def _rounded_bar(x: float, y: float, length: float, thick: float, fill: str) -> str:
    """A horizontal bar from x: square at the baseline, a 4 px rounded data end."""
    r = min(4.0, length / 2, thick / 2)
    d = (f"M{_f(x)},{_f(y)} h{_f(length - r)} a{_f(r)},{_f(r)} 0 0 1 {_f(r)},{_f(r)} "
         f"v{_f(thick - 2 * r)} a{_f(r)},{_f(r)} 0 0 1 {_f(-r)},{_f(r)} h{_f(r - length)} Z")
    return f'<path class="bar" d="{d}" fill="{fill}" data-length="{_f(length)}"/>'


# --- bar chart ----------------------------------------------------------------------------

def _count_text(unit: str) -> Callable[[float], str]:
    def fmt(v: float) -> str:
        return f"{_f(round(v, 1))}{unit}"
    return fmt


def bar_chart(rows: Sequence[Bar], *, title: str, unit: str = "回",
              value_text: Callable[[float], str] | None = None, width: float = 300,
              row_h: float = 24, thick: float = 12, colour: str = BAR) -> str:
    """Horizontal bars, one per row with a value > 0, in the order given; "" when no row
    has a value. `colour`: the one fill of the series (the organisation's accent may stand
    in for slot 1; categories are told apart by their labels and shapes, not the bars)."""
    for r in rows:
        _f(r.value)                        # refuse NaN / inf before anything is drawn
    rows = [r for r in rows if r.value > 0]
    if not rows:
        return ""
    fmt = value_text or _count_text(unit)
    size = 10.5
    glyph_w = 18 if any(r.category for r in rows) else 0
    label_w = min(max(_text_w(r.label, size) for r in rows) + 10, width * 0.45)
    value_w = max(_text_w(fmt(r.value), size) for r in rows) + 8
    x0 = glyph_w + label_w
    plot = max(width - x0 - value_w, 20)
    top = max(r.value for r in rows)
    height = len(rows) * row_h + 4
    body = [f'<line x1="{_f(x0)}" y1="2" x2="{_f(x0)}" y2="{_f(height - 2)}" '
            f'stroke="{RULE}" stroke-width="1"/>']
    for i, r in enumerate(rows):
        y = 2 + i * row_h
        mid = y + row_h / 2
        if r.category:
            shape, hue = _shape_of(r.category)
            body.append(_marker(shape, hue, 7, mid, 4.5))
        body.append(_text(glyph_w, mid + size * 0.36, r.label, size=size))
        length = max(plot * r.value / top, 2.0)
        body.append(_rounded_bar(x0, mid - thick / 2, length, thick, colour))
        body.append(_text(x0 + length + 5, mid + size * 0.36, fmt(r.value), size=size,
                          fill=INK_2))
    desc = "、".join(f"{r.label} {fmt(r.value)}" for r in rows)
    return _svg(width, height, title, desc, body)


# --- timeline -----------------------------------------------------------------------------

_TICK_STEPS = (10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600)


def _tick_step(duration: float, max_ticks: int = 7) -> float:
    return next((s for s in _TICK_STEPS if duration / s <= max_ticks - 1), 7200)


def _shape_rank(category: Category | None) -> int:
    if category is None:
        return 100
    return SHAPES.index(category.shape) if category.shape in SHAPES else (
        50 if category.shape == OTHER_SHAPE else 99)


def timeline(duration_s: float, spans: Sequence[Span], marks: Sequence[Mark], *,
             tick: Callable[[float], str], title: str, width: float = 640,
             band_label: str = "映っていた時間", plain_label: str = "よかった場面") -> str:
    """The analysed time [0, duration_s] as a strip; "" without a length."""
    _f(duration_s)
    if duration_s <= 0:
        return ""
    for s in spans:
        _f(s.start_s), _f(s.end_s)
    for m in marks:
        _f(m.t_s)
    pad_l, pad_r = 14.0, 18.0
    plot = width - pad_l - pad_r

    def x(t: float) -> float:
        return pad_l + plot * min(max(t, 0.0), duration_s) / duration_s

    shown = sorted((m for m in marks if 0 <= m.t_s <= duration_s), key=lambda m: m.t_s)
    r, gap = 5.5, 13.0
    rows_last: list[float] = []
    placed: list[tuple[Mark, float, int]] = []
    for m in shown:                          # dodge: the lowest row with room
        mx = x(m.t_s)
        row = next((k for k, last in enumerate(rows_last) if mx - last >= 2 * r + 2), None)
        if row is None:
            row = len(rows_last) if len(rows_last) < 4 else 3
            if row == len(rows_last):
                rows_last.append(-1e9)
        rows_last[row] = mx
        placed.append((m, mx, row))
    n_rows = max(len(rows_last), 1)
    num_h = 13.0 if any(m.number for m in shown) else 0.0
    band_y = 6 + num_h + n_rows * gap + 4
    band_h = 10.0
    body = [f'<rect x="{_f(pad_l)}" y="{_f(band_y)}" width="{_f(plot)}" height="{_f(band_h)}" '
            f'rx="2" fill="{TRACK}"/>']
    for s in spans:
        a, b = x(s.start_s), x(s.end_s)
        if b - a <= 0:
            continue
        w = max(b - a, 1.5)
        body.append(f'<rect class="span" x="{_f(a)}" y="{_f(band_y)}" width="{_f(w)}" '
                    f'height="{_f(band_h)}" fill="{BAND}"/>')
    for m, mx, row in placed:
        cy = band_y - 9 - row * gap
        body.append(f'<line x1="{_f(mx)}" y1="{_f(cy)}" x2="{_f(mx)}" y2="{_f(band_y + band_h)}" '
                    f'stroke="{RULE}" stroke-width="1"/>')
    for m, mx, row in placed:
        cy = band_y - 9 - row * gap
        shape, colour = _shape_of(m.category)
        body.append(_marker(shape, colour, mx, cy, r, cls="mark"))
        if m.number:
            body.append(_text(mx, cy - r - 3.5, str(m.number), size=9.5, weight="bold",
                              anchor="middle"))
    axis_y = band_y + band_h + 4
    step = _tick_step(duration_s)
    t, ticks = 0.0, []
    while t <= duration_s + 1e-6:
        ticks.append(t)
        t += step
    for t in ticks:
        tx = x(t)
        body.append(f'<line x1="{_f(tx)}" y1="{_f(axis_y)}" x2="{_f(tx)}" y2="{_f(axis_y + 4)}" '
                    f'stroke="{RULE}" stroke-width="1"/>')
        body.append(_text(tx, axis_y + 15, tick(t), size=9.5, fill=INK_2, anchor="middle"))
    # legend: words with every shape; only what the chart shows
    items: list[tuple[str, str]] = []
    if any(s.end_s > s.start_s for s in spans):
        items.append(("band", band_label))
    seen: dict[str, Category | None] = {}
    for m, _, _ in placed:
        seen.setdefault(m.category.key if m.category else "", m.category)
    for cat in sorted(seen.values(), key=_shape_rank):
        items.append((cat.key if cat else "", cat.label if cat else plain_label))
    leg_y, lx, size = axis_y + 36, pad_l, 10.5
    legend_rows = 1 if items else 0
    for key, label in items:
        w = 16 + _text_w(label, size) + 16
        if lx + w > width and lx > pad_l:
            lx, leg_y, legend_rows = pad_l, leg_y + 20, legend_rows + 1
        if key == "band":
            body.append(f'<rect x="{_f(lx)}" y="{_f(leg_y - 5)}" width="12" height="9" rx="1.5" '
                        f'fill="{BAND}"/>')
        else:
            cat = seen[key]
            shape, colour = _shape_of(cat)
            body.append(_marker(shape, colour, lx + 6, leg_y, 4.5))
        body.append(_text(lx + 16, leg_y + 3.8, label, size=size, fill=INK_2))
        lx += w
    height = (leg_y + 10) if items else (axis_y + 22)
    desc_parts = [f"{tick(m.t_s)} {m.category.label if m.category else plain_label}"
                  for m in shown]
    on = sum(max(min(s.end_s, duration_s) - max(s.start_s, 0), 0) for s in spans)
    desc = f"長さ {tick(duration_s)}。" + (f"{band_label} 約{round(on / 60, 1):g}分。" if spans
                                           else "") + "、".join(desc_parts)
    return _svg(width, height, title, desc, body)


# --- court --------------------------------------------------------------------------------

def _court_lines(s: float) -> list[str]:
    def p(x: float, y: float) -> str:
        return f"{_f(x * s)},{_f(y * s)}"
    stroke = f'fill="none" stroke="{LINE}" stroke-width="1.2"'
    bx, by = BASKET
    lines = [
        f'<rect x="0.6" y="0.6" width="{_f(CW * s - 1.2)}" height="{_f(CH * s - 1.2)}" {stroke}/>',
        f'<rect x="{_f((bx - KEY_W / 2) * s)}" y="0.6" width="{_f(KEY_W * s)}" '
        f'height="{_f(KEY_L * s)}" {stroke}/>',
        # free-throw circle: the half outside the key
        f'<path d="M{p(bx - FT_R, KEY_L)} A{_f(FT_R * s)},{_f(FT_R * s)} 0 0 0 '
        f'{p(bx + FT_R, KEY_L)}" {stroke}/>',
        # no-charge semicircle, backboard, rim
        f'<path d="M{p(bx - RA_R, by)} A{_f(RA_R * s)},{_f(RA_R * s)} 0 0 0 {p(bx + RA_R, by)}" '
        f'{stroke}/>',
        f'<line x1="{_f((bx - BOARD_W / 2) * s)}" y1="{_f(BOARD_Y * s)}" '
        f'x2="{_f((bx + BOARD_W / 2) * s)}" y2="{_f(BOARD_Y * s)}" stroke="{LINE}" '
        f'stroke-width="1.6"/>',
        f'<circle cx="{_f(bx * s)}" cy="{_f(by * s)}" r="{_f(RIM_R * s)}" {stroke}/>',
        # the 2-point line: two straight parts and the arc around the basket
        f'<path d="M{p(ARC_X, 0)} L{p(ARC_X, ARC_Y)} A{_f(R3 * s)},{_f(R3 * s)} 0 0 0 '
        f'{p(CW - ARC_X, ARC_Y)} L{p(CW - ARC_X, 0)}" {stroke}/>',
    ]
    return lines


def court(points_cm: Sequence[tuple[float, float]], *, title: str, width: float = 300,
          short_label: str = "短い", long_label: str = "長い",
          scale_label: str = "いた時間") -> str:
    """Where the person was: samples (court cm) counted in 1 m cells; "" when no sample is
    on the court."""
    nx, ny = math.ceil(CW / CELL_CM), math.ceil(CH / CELL_CM)
    counts: Counter[tuple[int, int]] = Counter()
    for px, py in points_cm:
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        if 0 <= px <= CW and 0 <= py <= CH:
            counts[(min(int(px // CELL_CM), nx - 1), min(int(py // CELL_CM), ny - 1))] += 1
    if not counts:
        return ""
    s = width / CW
    court_h = CH * s
    top = max(counts.values())
    body = [f'<rect x="0" y="0" width="{_f(CW * s)}" height="{_f(court_h)}" fill="{SURFACE}"/>']
    for (i, j), c in sorted(counts.items()):
        level = max(1, math.ceil(len(RAMP) * c / top))
        w = min(CELL_CM, CW - i * CELL_CM) * s
        h = min(CELL_CM, CH - j * CELL_CM) * s
        body.append(f'<rect class="cell" data-level="{level}" x="{_f(i * CELL_CM * s)}" '
                    f'y="{_f(j * CELL_CM * s)}" width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{RAMP[level - 1]}"/>')
    body += _court_lines(s)
    ly = court_h + 17
    size = 10
    body.append(_text(0, ly + 3, scale_label, size=size, fill=INK_2))
    lx = _text_w(scale_label, size) + 8
    body.append(_text(lx, ly + 3, short_label, size=size, fill=INK_2))
    lx += _text_w(short_label, size) + 4
    for k, colour in enumerate(RAMP):
        body.append(f'<rect x="{_f(lx + k * 13)}" y="{_f(ly - 6)}" width="12" height="10" '
                    f'fill="{colour}"/>')
    lx += len(RAMP) * 13 + 4
    body.append(_text(lx, ly + 3, long_label, size=size, fill=INK_2))
    desc = (f"{len(counts)}か所（1 mごと）に記録があります。いちばん長くいた所は "
            f"{scale_label}の{round(100 * top / sum(counts.values()))}%。")
    return _svg(width, court_h + 28, title, desc, body)


# --- growth grid --------------------------------------------------------------------------

def growth_grid(columns: Sequence[str],
                rows: Sequence[tuple[Category, Sequence[int | None]]], *,
                title: str, width: float = 640, cell_h: float = 24) -> str:
    """Rows = evaluation points, columns = works; "" when there is no column or no row. A
    None value: the work's activity has no such point (an outlined cell with 「—」)."""
    rows = [(cat, list(values)) for cat, values in rows if len(values) == len(columns)]
    if not columns or not rows:
        return ""
    for _, values in rows:
        for v in values:
            if v is not None:
                _f(v)
    size, head = 9.5, 18.0
    glyph_w = 18
    label_w = min(max(_text_w(cat.label, size) for cat, _ in rows) + 10, width * 0.36)
    x0 = glyph_w + label_w
    cell_w = min((width - x0) / len(columns), 64.0)
    width = x0 + cell_w * len(columns)
    top = max((v for _, values in rows for v in values if v is not None), default=0) or 1
    body = []
    for j, col in enumerate(columns):
        body.append(_text(x0 + (j + 0.5) * cell_w, head - 6, col, size=size - 1, fill=INK_2,
                          anchor="middle"))
    lines = []
    for i, (cat, values) in enumerate(rows):
        y = head + i * cell_h
        mid = y + cell_h / 2
        body.append(_marker(cat.shape, cat.colour, 7, mid, 4.5))
        body.append(_text(glyph_w, mid + size * 0.36, cat.label, size=size))
        for j, v in enumerate(values):
            x = x0 + j * cell_w
            if v is None:     # the work's activity has no such point: a dash, never a 0
                body.append(f'<rect class="cell" x="{_f(x + 1)}" y="{_f(y + 1)}" '
                            f'width="{_f(cell_w - 2)}" height="{_f(cell_h - 2)}" rx="3" '
                            f'fill="{SURFACE}" stroke="{TRACK}" stroke-width="1" data-value="">'
                            f'<title>{escape(columns[j])}・{escape(cat.label)}：対象外</title></rect>')
                body.append(_text(x + cell_w / 2, mid + size * 0.36, "—", size=size,
                                  fill=INK_2, anchor="middle"))
                continue
            step = None if v <= 0 else min(len(GRID_STEPS) - 1,
                                           math.ceil(v / top * len(GRID_STEPS)) - 1)
            fill = TRACK if step is None else GRID_STEPS[step]
            body.append(f'<rect class="cell" x="{_f(x + 1)}" y="{_f(y + 1)}" '
                        f'width="{_f(cell_w - 2)}" height="{_f(cell_h - 2)}" rx="3" '
                        f'fill="{fill}" data-value="{int(v)}"><title>{escape(columns[j])}・'
                        f'{escape(cat.label)}：{int(v)}回</title></rect>')
            ink = SURFACE if step is not None and step >= GRID_DARK_FROM else (INK if v else INK_2)
            body.append(_text(x + cell_w / 2, mid + size * 0.36, str(int(v)), size=size,
                              fill=ink, anchor="middle"))
        lines.append(f"{cat.label}：" + "、".join(
            f"{c} 対象外" if v is None else f"{c} {int(v)}回" for c, v in zip(columns, values)))
    height = head + len(rows) * cell_h + 2
    return _svg(width, height, title, "。".join(lines), body)
