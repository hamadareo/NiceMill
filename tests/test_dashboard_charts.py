"""The two charts the dashboard adds (export.dashboard_charts): the image-plane heatmap and the
category x people counts grid. Pure SVG strings, synthetic numbers only."""
import re

import pytest

from pfb_app.export import dashboard_charts as dc
from pfb_app.export.labels import SPORTS_CATEGORIES


def test_frame_heatmap_counts_points_in_cells_with_a_scale_legend():
    pts = [(10.0, 10.0)] * 5 + [(630.0, 350.0)] * 2
    svg = dc.frame_heatmap(pts, (640, 360), title="カメラ1から見た位置")
    assert svg.startswith("<svg") and "<title>カメラ1から見た位置</title>" in svg
    cells = re.findall(r'class="cell" data-level="(\d)"', svg)
    assert len(cells) == 2 and max(map(int, cells)) == 5
    assert "短い" in svg and "長い" in svg               # the scale is in words, not colour alone
    assert "2か所" in svg                               # the desc says what is drawn
    assert "court" not in svg


def test_frame_heatmap_is_empty_without_points_inside_the_frame():
    assert dc.frame_heatmap([], (640, 360), title="t") == ""
    assert dc.frame_heatmap([(-5.0, 2.0), (700.0, 2.0)], (640, 360), title="t") == ""
    assert dc.frame_heatmap([(1.0, 1.0)], (0, 0), title="t") == ""


def test_frame_heatmap_refuses_nan():
    with pytest.raises(ValueError):
        dc.frame_heatmap([(float("nan"), 1.0)], (640, 360), title="t")


def test_count_grid_writes_every_number_and_marks_each_category():
    cats = SPORTS_CATEGORIES[:3]
    svg = dc.count_grid(["山田", "#7"], [(cats[0], [2, 0]), (cats[1], [1, 3]), (cats[2], [0, 0])],
                        title="種類ごとのよかった場面")
    assert svg.startswith("<svg")
    values = re.findall(r'data-value="(\d+)"', svg)
    assert values == ["2", "0", "1", "3"]               # an all-zero category row is left out
    assert 'data-shape="circle"' in svg and 'data-shape="square"' in svg
    assert "山田" in svg and "#7" in svg
    assert "パス：山田 2回、#7 0回" in svg


def test_count_grid_is_empty_without_numbers():
    cats = SPORTS_CATEGORIES[:1]
    assert dc.count_grid([], [(cats[0], [])], title="t") == ""
    assert dc.count_grid(["a"], [(cats[0], [0])], title="t") == ""


def test_count_grid_shortens_long_names_but_keeps_them_in_the_desc():
    cats = SPORTS_CATEGORIES[:1]
    long_name = "とても長い名前の選手さん"
    svg = dc.count_grid([long_name], [(cats[0], [1])], title="t")
    assert "とても長い…" in svg and long_name in svg
