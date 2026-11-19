"""Automatic people selection: the cap, box statistics and the pure label choice."""
import pytest

from pfb_app.pipeline.auto_people import (BoxStat, box_stats, coverage_bar, detect_p90,
                                          people_cap, select_labels)


def test_people_cap():
    assert people_cap(5, 9.0, "sports") == 5
    assert people_cap(None, 9.2, "sports") == 12          # ceil(9.2) + 2 -> 12
    assert people_cap(None, None, "school") == 12
    assert people_cap(None, 40.0, "school") == 20 and people_cap(None, 40.0, "sports") == 30
    assert people_cap(0, None, "school") == 12            # 0/None means unknown
    assert people_cap(99, None, "school") == 20           # never above the hard max


def test_select_filters_sorts_and_caps():
    q = {"P1": {"coverage": .9, "mean_h_frac": .2}, "P2": {"coverage": .3, "mean_h_frac": .1},
         "P3": {"coverage": .1, "mean_h_frac": .3}, "P4": {"coverage": .8, "mean_h_frac": .03},
         "P5": {"coverage": .5, "mean_h_frac": None}}
    stats = {"P1": BoxStat(10, 50, 300., 60), "P2": BoxStat(5, 20, 900., 24)}
    assert select_labels(["P1", "P2", "P3", "P4", "P5"], q, stats, cap=2) == ["P2", "P1"]


def test_select_relaxes_to_top_n_when_nobody_passes():
    q = {"P1": {"coverage": .1, "mean_h_frac": .01}, "P2": {"coverage": .2, "mean_h_frac": .01}}
    assert select_labels(["P1", "P2"], q, {}, cap=1) == ["P2"]


def test_select_without_quality_uses_boxes(tmp_path):
    p = tmp_path / "boxes.txt"
    p.write_text("1,1,500,0,10,10,.9\n1,2,100,0,10,10,.9\n2,1,500,0,10,10,.9\n3,3,0,0,1,1,.9\n")
    stats = box_stats(p, ["P1", "P2", "P3"])
    assert stats["P1"] == BoxStat(2, 1, 505.0, 2)
    assert select_labels(["P1", "P2", "P3"], None, stats, cap=2) == ["P2", "P1"]


def test_same_first_frame_orders_left_to_right():
    q = {f"P{i}": {"coverage": .9, "mean_h_frac": .2} for i in (1, 2, 3)}
    stats = {"P1": BoxStat(5, 1, 400., 5), "P2": BoxStat(5, 1, 100., 5), "P3": BoxStat(5, 1, 100., 5)}
    assert select_labels(["P1", "P2", "P3"], q, stats, cap=3) == ["P2", "P3", "P1"]


def test_no_quality_low_coverage_is_excluded_below_cap():
    stats = {"P1": BoxStat(100, 1, 10., 100), "P2": BoxStat(50, 1, 20., 50),
             "P3": BoxStat(10, 1, 5., 10)}   # P3: 10 % of a 100-frame file
    assert select_labels(["P1", "P2", "P3"], None, stats, cap=3) == ["P1", "P2"]


def test_no_quality_relaxes_to_top_n_when_nobody_passes():
    stats = {"P1": BoxStat(10, 1, 10., 10), "P2": BoxStat(20, 1, 20., 20),
             "P3": BoxStat(5, 90, 5., 94), "P4": BoxStat(1, 100, 1., 100)}   # span 100
    assert select_labels(["P1", "P2", "P3", "P4"], None, stats, cap=2) == ["P1", "P2"]
    assert select_labels(["P1", "P2"], None, {}, cap=1) == ["P1"]


def test_box_stats_missing_file(tmp_path):
    assert box_stats(tmp_path / "none.txt", ["P1"]) == {}


def test_detect_p90():
    out = {"cameras": {"cam1": {"persons": {"p90": 7.0}, "vehicles": {"p90": 3}}}}
    assert detect_p90(out, "cam1") == 7.0 and detect_p90(out, "cam2") is None
    assert detect_p90(None, "cam1") is None
    assert detect_p90(out, "cam1", "vehicle") == 3.0
    bad = {"cameras": {"cam1": {"persons": {"p90": "x"}}}}
    assert detect_p90(bad, "cam1") is None
    assert detect_p90({"cameras": {"cam1": {"persons": {"p90": -1}}}}, "cam1") is None


def _q(cov, h=0.15):
    return {"coverage": cov, "mean_h_frac": h}


def test_coverage_bar_keeps_the_fixed_bar_when_somebody_is_well_seen():
    assert coverage_bar([0.9, 0.3]) == 0.25 and coverage_bar([0.65]) == 0.25
    assert coverage_bar([0.5]) == pytest.approx(0.225)
    assert coverage_bar([]) == 0.0
    assert coverage_bar([0.34, 0.2]) == pytest.approx(0.153)   # 0.45 of the best, below 0.25


def test_long_video_bar_follows_the_best_person():
    """2026-11-14 trial (30 min, a game with cuts away from the court): the best person was
    boxed for 34 % of the video, so the fixed 25 % left two of fourteen people evaluated."""
    cov = [0.34, 0.31, 0.27, 0.22, 0.2, 0.18, 0.17, 0.15, 0.12, 0.08, 0.05]
    labels = [f"P{i}" for i in range(1, len(cov) + 1)]
    q = {lb: _q(c) for lb, c in zip(labels, cov)}
    chosen = select_labels(labels, q, {}, cap=10)
    assert chosen == ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]    # coverage >= 0.153
    assert len(select_labels(labels, q, {}, cap=3)) == 3            # the cap still applies


def test_short_video_with_a_well_seen_person_is_unchanged():
    q = {"P1": _q(0.68), "P2": _q(0.3), "P3": _q(0.24), "P4": _q(0.19)}
    assert select_labels(["P1", "P2", "P3", "P4"], q, {}, cap=10) == ["P1", "P2"]


def test_a_small_person_does_not_set_the_bar():
    """The bar follows the best person who is big enough to count (MIN_HEIGHT)."""
    q = {"P1": _q(0.9, h=0.02), "P2": _q(0.3), "P3": _q(0.1)}
    assert select_labels(["P1", "P2", "P3"], q, {}, cap=5) == ["P2"]    # bar 0.135


def test_long_video_without_quality_follows_the_best_person_too():
    stats = {"P1": BoxStat(340, 1, 10., 1000), "P2": BoxStat(200, 5, 20., 900),
             "P3": BoxStat(60, 9, 30., 800)}                   # span 1000: 0.34 / 0.2 / 0.06
    assert select_labels(["P1", "P2", "P3"], None, stats, cap=3) == ["P1", "P2"]
