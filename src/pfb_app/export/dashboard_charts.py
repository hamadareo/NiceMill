"""The dashboard's own SVG charts, in the hand-built style of export.charts (its palette,
markers and text helpers; an accessible <title>/<desc> with the values as text):

- `frame_heatmap`: where a person or a team was in one camera's picture (image px, the foot
  point), counted in cells on the court chart's one-hue ramp (light = short, dark = long),
  inside a frame outline. Used when there is no court geometry; it is a view from the camera,
  so it carries no court lines and no distance.
- `count_grid`: good moments per category (rows, with their marker shape) and person
  (columns), the number written in every cell (growth_grid's cell rules), so the grid is its
  own table. Rows with no moment are left out.
"""
from __future__ import annotations

import math
from collections import Counter
from html import escape
from typing import Sequence

from pfb_app.export.charts import (GRID_DARK_FROM, GRID_STEPS, INK, INK_2, RAMP, RULE, SURFACE, TRACK, _f,
                                   _marker, _svg, _text, _text_w)
from pfb_app.export.labels import Category

COLS = 16          # cells across the picture (rows follow its aspect)
NAME_CHARS = 5     # characters of a column name drawn; the full name stays in the desc


def frame_heatmap(points_px: Sequence[tuple[float, float]], size: tuple[int, int], *,
                  title: str, width: float = 320, short_label: str = "短い",
                  long_label: str = "長い", scale_label: str = "いた時間") -> str:
    fw, fh = size
    if not fw or not fh or fw <= 0 or fh <= 0:
        return ""
    for x, y in points_px:
        _f(x), _f(y)                        # NaN / inf is a bug in the data: refuse it
    nx = COLS
    cell = fw / nx
    ny = max(1, math.ceil(fh / cell))
    counts: Counter[tuple[int, int]] = Counter()
    for x, y in points_px:
        if 0 <= x <= fw and 0 <= y <= fh:
            counts[(min(int(x // cell), nx - 1), min(int(y // cell), ny - 1))] += 1
    if not counts:
        return ""
    s = width / fw
    plot_h = fh * s
    top = max(counts.values())
    body = [f'<rect x="0" y="0" width="{_f(width)}" height="{_f(plot_h)}" fill="{SURFACE}"/>']
    for (i, j), c in sorted(counts.items()):
        level = max(1, math.ceil(len(RAMP) * c / top))
        w = min(cell, fw - i * cell) * s
        h = min(cell, fh - j * cell) * s
        body.append(f'<rect class="cell" data-level="{level}" x="{_f(i * cell * s)}" '
                    f'y="{_f(j * cell * s)}" width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{RAMP[level - 1]}"/>')
    body.append(f'<rect x="0.6" y="0.6" width="{_f(width - 1.2)}" height="{_f(plot_h - 1.2)}" '
                f'fill="none" stroke="{RULE}" stroke-width="1.2"/>')
    ly, size_ = plot_h + 17, 10
    body.append(_text(0, ly + 3, scale_label, size=size_, fill=INK_2))
    lx = _text_w(scale_label, size_) + 8
    body.append(_text(lx, ly + 3, short_label, size=size_, fill=INK_2))
    lx += _text_w(short_label, size_) + 4
    for k, colour in enumerate(RAMP):
        body.append(f'<rect x="{_f(lx + k * 13)}" y="{_f(ly - 6)}" width="12" height="10" '
                    f'fill="{colour}"/>')
    lx += len(RAMP) * 13 + 4
    body.append(_text(lx, ly + 3, long_label, size=size_, fill=INK_2))
    desc = (f"画面を{nx}列に分けた{len(counts)}か所に記録があります。いちばん長くいた所は "
            f"{scale_label}の{round(100 * top / sum(counts.values()))}%。カメラから見た位置で、"
            "コートの上の位置ではありません。")
    return _svg(width, plot_h + 28, title, desc, body)


def _short(name: str) -> str:
    return name if len(name) <= NAME_CHARS else name[:NAME_CHARS] + "…"


def count_grid(columns: Sequence[str], rows: Sequence[tuple[Category, Sequence[int]]], *,
               title: str, width: float = 640, cell_h: float = 24) -> str:
    rows = [(cat, [int(v) for v in values]) for cat, values in rows
            if len(values) == len(columns) and any(v for v in values)]
    if not columns or not rows:
        return ""
    size, head = 9.5, 18.0
    glyph_w = 18
    label_w = min(max(_text_w(cat.label, size) for cat, _ in rows) + 10, width * 0.36)
    x0 = glyph_w + label_w
    cell_w = min((width - x0) / len(columns), 64.0)
    width = x0 + cell_w * len(columns)
    top = max(v for _, values in rows for v in values) or 1
    body = []
    for j, col in enumerate(columns):
        body.append(_text(x0 + (j + 0.5) * cell_w, head - 6, _short(col), size=size - 1,
                          fill=INK_2, anchor="middle"))
    lines = []
    for i, (cat, values) in enumerate(rows):
        y = head + i * cell_h
        mid = y + cell_h / 2
        body.append(_marker(cat.shape, cat.colour, 7, mid, 4.5))
        body.append(_text(glyph_w, mid + size * 0.36, cat.label, size=size))
        for j, v in enumerate(values):
            x = x0 + j * cell_w
            step = None if v <= 0 else min(len(GRID_STEPS) - 1,
                                           math.ceil(v / top * len(GRID_STEPS)) - 1)
            fill = TRACK if step is None else GRID_STEPS[step]
            body.append(f'<rect class="cell" x="{_f(x + 1)}" y="{_f(y + 1)}" '
                        f'width="{_f(cell_w - 2)}" height="{_f(cell_h - 2)}" rx="3" '
                        f'fill="{fill}" data-value="{v}"><title>{escape(columns[j])}・'
                        f'{escape(cat.label)}：{v}回</title></rect>')
            ink = SURFACE if step is not None and step >= GRID_DARK_FROM else (INK if v else INK_2)
            body.append(_text(x + cell_w / 2, mid + size * 0.36, str(v), size=size,
                              fill=ink, anchor="middle"))
        lines.append(f"{cat.label}：" + "、".join(f"{c} {v}回" for c, v in zip(columns, values)))
    height = head + len(rows) * cell_h + 2
    return _svg(width, height, title, "。".join(lines), body)
