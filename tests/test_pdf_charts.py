"""export.charts: hand-built inline SVG (no plotting library). Pure functions, so the checks
read the SVG itself: well-formed, finite numbers only, every label present, and identity
never carried by colour alone (a shape and a text label go with every coloured mark)."""
import math
import re
import xml.etree.ElementTree as ET

import pytest

from pfb_app.export import charts
from pfb_app.export.charts import Bar, Mark, Span
from pfb_app.export.labels import SPORTS_CATEGORIES

NS = "{http://www.w3.org/2000/svg}"
SPORTS = SPORTS_CATEGORIES   # a fixed table, whatever the profiles say


def parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == f"{NS}svg"
    return root


def numbers_are_finite(svg: str) -> None:
    low = svg.lower()
    assert "nan" not in low and "inf" not in low and "None" not in svg
    for value in re.findall(r'(?<![\w-])(?:x|y|x1|x2|y1|y2|cx|cy|r|width|height|data-x|data-y)="([^"]*)"',
                            svg):
        assert math.isfinite(float(value)), value


def texts(root: ET.Element) -> list[str]:
    return ["".join(t.itertext()) for t in root.iter(f"{NS}text")]


def shapes(root: ET.Element) -> list[str]:
    return [e.get("data-shape") for e in root.iter() if e.get("data-shape")]


# --- bars ---------------------------------------------------------------------------------

def test_bar_chart_has_a_bar_label_and_value_per_row():
    rows = [Bar("パス", 3, SPORTS[0]), Bar("守り", 1, SPORTS[2]), Bar("声かけ", 2, SPORTS[4])]
    svg = charts.bar_chart(rows, title="どんな場面で活躍したか", unit="回")
    root = parse(svg)
    numbers_are_finite(svg)
    for label, value in (("パス", "3回"), ("守り", "1回"), ("声かけ", "2回")):
        assert label in texts(root) and value in texts(root)
    assert len([e for e in root.iter() if e.get("class") == "bar"]) == 3
    # the bars keep the given order (the categories' fixed order)
    assert [t for t in texts(root) if t in ("パス", "守り", "声かけ")] == ["パス", "守り", "声かけ"]
    # the accessible twin: title and a text description of every value
    assert root.find(f"{NS}title").text == "どんな場面で活躍したか"
    assert "パス 3回" in root.find(f"{NS}desc").text
    # a glyph (shape) beside each category label, never colour alone
    assert shapes(root) == ["circle", "triangle", "triangle_down"]


def test_bar_chart_bars_are_one_colour_and_scale_to_the_largest():
    svg = charts.bar_chart([Bar("A", 4), Bar("B", 2)], title="t", unit="回")
    root = parse(svg)
    bars = [e for e in root.iter() if e.get("class") == "bar"]
    assert {b.get("fill") for b in bars} == {charts.BAR}
    a, b = (float(e.get("data-length")) for e in bars)
    assert a == pytest.approx(2 * b)


def test_empty_rows_are_left_out_and_nothing_left_is_no_chart():
    svg = charts.bar_chart([Bar("パス", 0), Bar("守り", 2)], title="t", unit="回")
    assert "パス" not in svg and "守り" in svg
    assert charts.bar_chart([Bar("パス", 0)], title="t", unit="回") == ""
    assert charts.bar_chart([], title="t", unit="回") == ""


def test_non_finite_values_are_refused():
    with pytest.raises(ValueError):
        charts.bar_chart([Bar("A", float("nan"))], title="t", unit="回")
    with pytest.raises(ValueError):
        charts.bar_chart([Bar("A", float("inf"))], title="t", unit="回")


def test_value_format_and_escaping():
    svg = charts.bar_chart([Bar("<b>&人", 12.5)], title="a<b", unit="分",
                           value_text=lambda v: f"{v:.1f}分")
    root = parse(svg)
    assert "<b>&人" in texts(root) and "12.5分" in texts(root)


# --- timeline -----------------------------------------------------------------------------

def tick(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def test_timeline_marks_spans_ticks_and_legend():
    marks = [Mark(30, SPORTS[0], number=1), Mark(31, SPORTS[0]), Mark(200, SPORTS[2], number=2),
             Mark(400, SPORTS[7])]
    svg = charts.timeline(600, [Span(0, 120), Span(180, 420)], marks, tick=tick,
                          title="時間の流れ")
    root = parse(svg)
    numbers_are_finite(svg)
    found = [e for e in root.iter() if e.get("class") == "mark"]
    assert len(found) == 4
    assert [m.get("data-shape") for m in found] == ["circle", "circle", "triangle", "ring"]
    assert len([e for e in root.iter() if e.get("class") == "span"]) == 2
    t = texts(root)
    assert "00:00" in t and "1" in t and "2" in t           # ticks and the picture numbers
    for label in ("パス", "守り", "その他", "映っていた時間"):   # legend: shape + words
        assert label in t
    assert "シュート・得点" not in t                        # only categories that occur
    # two marks one second apart do not sit on top of each other
    a, b = found[0], found[1]
    assert (a.get("data-x"), a.get("data-y")) != (b.get("data-x"), b.get("data-y"))


def test_timeline_without_spans_has_no_band_legend_and_drops_marks_outside():
    svg = charts.timeline(100, [], [Mark(50, SPORTS[1]), Mark(150, SPORTS[1]),
                                    Mark(-3, SPORTS[1])], tick=tick, title="t")
    root = parse(svg)
    assert len([e for e in root.iter() if e.get("class") == "mark"]) == 1
    assert "映っていた時間" not in texts(root)


def test_timeline_needs_a_length():
    assert charts.timeline(0, [Span(0, 1)], [], tick=tick, title="t") == ""
    with pytest.raises(ValueError):
        charts.timeline(float("nan"), [], [], tick=tick, title="t")


def test_timeline_marks_without_a_category_are_plain_dots_with_a_legend_word():
    svg = charts.timeline(60, [Span(0, 60)], [Mark(10, None), Mark(40, SPORTS[0])],
                          tick=tick, title="t")
    root = parse(svg)
    assert "よかった場面" in texts(root)


# --- court --------------------------------------------------------------------------------

def test_court_heat_cells_only_where_the_person_was():
    pts = [(700.0, 300.0)] * 30 + [(200.0, 900.0)] * 3 + [(5000.0, 100.0)]   # last: off court
    svg = charts.court(pts, title="コートのどこにいたか")
    root = parse(svg)
    numbers_are_finite(svg)
    cells = [e for e in root.iter() if e.get("class") == "cell"]
    assert len(cells) == 2
    levels = sorted(int(c.get("data-level")) for c in cells)
    assert levels[0] < levels[-1] == len(charts.RAMP)
    assert {c.get("fill") for c in cells} <= set(charts.RAMP)
    t = texts(root)
    assert "短い" in t and "長い" in t                        # the ramp is labelled


def test_court_without_points_is_no_chart():
    assert charts.court([], title="t") == ""
    assert charts.court([(9999.0, 9999.0)], title="t") == ""


def test_glyph_is_a_small_standalone_svg():
    svg = charts.glyph(SPORTS[3])
    root = parse(svg)
    assert shapes(root) == ["diamond"]


def test_timeline_numbers_are_centred_over_their_marks():
    svg = charts.timeline(60, [], [Mark(30, SPORTS[0], number=1)], tick=tick, title="t")
    root = parse(svg)
    num = next(t for t in root.iter(f"{NS}text") if t.text == "1" and t.get("font-weight"))
    mark = next(e for e in root.iter() if e.get("class") == "mark")
    assert num.get("text-anchor") == "middle" and num.get("x") == mark.get("data-x")


def test_glyph_has_a_title():
    root = parse(charts.glyph(SPORTS[1]))
    assert root.find(f"{NS}title").text == "シュート・得点"
    assert parse(charts.glyph(None)).find(f"{NS}title").text == "よかった場面"


# --- growth grid (成長の記録) -----------------------------------------------------------------

def test_growth_grid_writes_every_count_in_its_cell():
    pass_, defense = SPORTS[0], SPORTS[2]
    svg = charts.growth_grid(["9/1", "9/8", "9/15"], [(pass_, [1, 0, 4]), (defense, [2, 2, 0])],
                             title="観点ごとの回数")
    numbers_are_finite(svg)
    root = parse(svg)
    cells = [e for e in root.iter(f"{NS}rect") if e.get("class") == "cell"]
    assert len(cells) == 6
    assert [c.get("data-value") for c in cells] == ["1", "0", "4", "2", "2", "0"]
    fills = {c.get("data-value"): c.get("fill") for c in cells}
    assert fills["0"] == charts.TRACK                      # nothing recorded: the empty track
    assert fills["4"] == charts.GRID_STEPS[-1]             # the most: the darkest step
    assert fills["1"] in charts.GRID_STEPS[:2]
    shown = texts(root)
    assert {"9/1", "9/8", "9/15", "パス", "守り"} <= set(shown)
    assert [t for t in shown if t.isdigit()] == ["1", "0", "4", "2", "2", "0"]
    assert sorted(shapes(root)) == sorted([pass_.shape, defense.shape])   # not colour alone
    desc = "".join(root.find(f"{NS}desc").itertext())
    assert "パス：9/1 1回、9/8 0回、9/15 4回" in desc


def test_growth_grid_text_contrast_on_the_dark_step():
    svg = charts.growth_grid(["9/1"], [(SPORTS[0], [3])], title="t")
    root = parse(svg)
    digit = next(t for t in root.iter(f"{NS}text") if "".join(t.itertext()) == "3")
    assert digit.get("fill") == charts.SURFACE


def test_growth_grid_without_data_is_empty():
    assert charts.growth_grid([], [(SPORTS[0], [])], title="t") == ""
    assert charts.growth_grid(["9/1"], [], title="t") == ""


def test_growth_grid_marks_a_point_the_work_does_not_have():
    """None: the work's activity has no such point (review MED5): a dash, never a 0."""
    svg = charts.growth_grid(["9/1", "9/8"], [(SPORTS[0], [None, 2])], title="t")
    root = parse(svg)
    cells = [e for e in root.iter(f"{NS}rect") if e.get("class") == "cell"]
    assert [c.get("data-value") for c in cells] == ["", "2"]
    assert cells[0].get("fill") == charts.SURFACE
    assert "—" in texts(root) and "0" not in texts(root)
    desc = "".join(root.find(f"{NS}desc").itertext())
    assert "9/1 対象外" in desc
