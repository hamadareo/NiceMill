"""Moments when a tracked person's box is on someone else (an ID swap), from the names
stage's number timeline: those events are not the person's and are left out."""
from pfb_app.pipeline.identity_check import foreign_at, number_at

# P2 of work 6 in small: #5 at the start, then #3, then #1 (the person's number) for most
READ = {"dominant": "1", "dominant_share": 0.67, "timeline_s": 6.0,
        "segments": [[1, 300, "5", 3], [400, 1800, "3", 8], [4000, 10000, "1", 24]]}


def test_number_at_inside_and_near_segments():
    assert number_at(READ, 100, reach=200) == "5"
    assert number_at(READ, 5000, reach=200) == "1"
    assert number_at(READ, 330, reach=200) == "5"          # between: the nearer segment
    assert number_at(READ, 380, reach=200) == "3"
    assert number_at(READ, 2900, reach=200) is None        # far from any read: unknown
    assert number_at(READ, 10150, reach=200) == "1"


def test_foreign_when_another_number_is_read_there():
    assert foreign_at(READ, 100, reach=200) is True
    assert foreign_at(READ, 1000, reach=200) is True
    assert foreign_at(READ, 6000, reach=200) is False
    assert foreign_at(READ, 2900, reach=200) is False      # unknown is never foreign


def test_no_dominant_or_no_timeline_is_never_foreign():
    assert foreign_at({**READ, "dominant": None}, 100, reach=200) is False
    assert foreign_at({"number": "5", "frames_agree": 3}, 100, reach=200) is False
    assert foreign_at(None, 100, reach=200) is False
