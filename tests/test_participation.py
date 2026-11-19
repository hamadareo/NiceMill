"""Who takes part in a game (players) and who does not (spectators, bench, officials):
court numbers from the boxes and the court geometry, and the rule on them."""
import json

import numpy as np

from pfb_app.export.tracks import BoxIndex, Geometry, load_geometry
from pfb_app.pipeline.participation import (CHECK, NOT_PLAYING, PLAYER, Court, court_numbers,
                                            decide, participation)

# image px == court cm (the identity homography) keeps the numbers readable
IDENTITY = Geometry(np.eye(3), None, None, "autocourt")
FPS = 10.0


def _walk(points, frames_each=60):
    """Rows (frame, x, y, w, h) whose box bottom centre walks through `points` (court cm)."""
    rows, f = [], 1
    for x, y in points:
        for _ in range(frames_each):
            rows.append((f, x - 20.0, y - 180.0, 40.0, 180.0))
            f += 1
    return rows


def test_a_player_moves_around_on_the_court():
    idx = BoxIndex({"P1": _walk([(200, 200), (700, 500), (1200, 900), (400, 800)])})
    c = court_numbers(idx, ["P1"], IDENTITY, FPS)["P1"]
    assert c.on_court == 1.0 and c.spread_cm >= 600 and c.samples > 0
    assert decide(c, None, None) == (PLAYER, "court")


def test_a_spectator_sits_outside_the_court():
    idx = BoxIndex({"P9": _walk([(-650, 0), (-640, 20), (-655, 10)])})
    c = court_numbers(idx, ["P9"], IDENTITY, FPS)["P9"]
    assert c.on_court == 0.0
    assert decide(c, None, None) == (NOT_PLAYING, "off_court")


def test_someone_standing_at_the_line_all_game_is_not_playing():
    # on the court with the margin, but never more than ~1.3 m from one spot (the near
    # sideline person of work 6: spread 132 cm)
    idx = BoxIndex({"P1": _walk([(1450, 850), (1500, 950), (1480, 980), (1440, 900)])})
    c = court_numbers(idx, ["P1"], IDENTITY, FPS)["P1"]
    assert c.on_court == 1.0 and c.spread_cm < 200
    assert decide(c, None, None) == (NOT_PLAYING, "static")


def test_the_margin_keeps_a_player_who_steps_out():
    idx = BoxIndex({"P2": _walk([(-80, 300), (600, 500), (1580, 700), (800, 1150)])})
    c = court_numbers(idx, ["P2"], IDENTITY, FPS)["P2"]
    assert c.on_court == 1.0          # 80 cm out is inside the 100 cm margin
    assert decide(c, None, None)[0] == PLAYER


def test_half_on_the_court_is_to_be_checked():
    idx = BoxIndex({"P5": _walk([(-500, 300), (-480, 900), (-520, 600), (300, 300),
                                 (900, 800), (1300, 200), (-500, 100), (-510, 400),
                                 (600, 600), (-490, 200)])})
    c = court_numbers(idx, ["P5"], IDENTITY, FPS)["P5"]
    assert 0.2 <= c.on_court < 0.5
    assert decide(c, None, None) == (CHECK, "court")


def test_a_team_colour_makes_a_player_even_off_the_court():
    off = Court(on_court=0.0, spread_cm=50.0, samples=100)
    assert decide(off, None, "T1") == (PLAYER, "team")


def test_without_geometry_motion_decides_and_static_is_to_be_checked():
    assert decide(None, 0.12, None) == (PLAYER, "moving")
    assert decide(None, 0.01, None) == (CHECK, "no_court")     # never dropped silently
    assert decide(None, None, None) == (CHECK, "no_court")
    assert decide(None, 0.01, "T2") == (PLAYER, "team")


def test_too_few_court_samples_fall_back_to_motion():
    few = Court(on_court=0.0, spread_cm=0.0, samples=3)
    assert decide(few, 0.12, None) == (PLAYER, "moving")


def test_participation_of_every_label():
    idx = BoxIndex({"P1": _walk([(200, 200), (700, 500), (1200, 900)]),
                    "P2": _walk([(-650, 0), (-640, 20), (-655, 10)]),
                    "P4": _walk([(900, 900), (300, 200), (1300, 400)])})
    out = participation(["P1", "P2", "P3", "P4"], boxes=idx, geometry=IDENTITY, fps=FPS,
                        quality={"P3": {"spread_frac": 0.2}}, teams={})
    assert out == {"P1": (PLAYER, "court"), "P2": (NOT_PLAYING, "off_court"),
                   "P3": (PLAYER, "moving"), "P4": (PLAYER, "court")}


def test_participation_without_geometry_or_boxes():
    out = participation(["P1"], boxes=None, geometry=None, fps=None, quality=None, teams=None)
    assert out == {"P1": (CHECK, "no_court")}


def test_load_geometry_reads_the_track_stage_court(tmp_path):
    boxes = tmp_path / "boxes.txt"
    boxes.write_text("")
    (tmp_path / "court").mkdir()
    (tmp_path / "court" / "geometry.json").write_text(json.dumps(
        {"H": np.eye(3).tolist(), "lens": None, "source": "autocourt"}))
    g = load_geometry(boxes)
    assert g is not None and g.source == "autocourt"
    # the line-call run's geometry wins when both are there
    (tmp_path / "linecall").mkdir()
    (tmp_path / "linecall" / "geometry.json").write_text(json.dumps(
        {"H": (np.eye(3) * 2).tolist(), "source": "tid"}))
    assert load_geometry(boxes).source == "tid"


def test_a_court_nobody_plays_on_is_not_trusted():
    """A wrong court registration would put every player outside: with fewer than two people
    playing on it, the court is ignored (motion decides, nobody is dropped)."""
    idx = BoxIndex({"P1": _walk([(-650, 0), (-640, 20), (-655, 10)]),
                    "P2": _walk([(-900, 0), (-300, 20), (-500, 10)])})
    out = participation(["P1", "P2"], boxes=idx, geometry=IDENTITY, fps=FPS,
                        quality={"P2": {"spread_frac": 0.2}}, teams={})
    assert out == {"P1": (CHECK, "no_court"), "P2": (PLAYER, "moving")}
