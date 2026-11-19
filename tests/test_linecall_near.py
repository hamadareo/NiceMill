"""Release 3 line-call conditions (PFB3x3 T11 review, 2026-10-23):

- a 踏んだ at a line toward the camera (where a foot reaching toward the camera is itself the
  box bottom, so the engine's ground check sees nothing) is shown as 要確認, never as a plain
  踏んだ. The engine has no near/far flag: the app derives it from geometry.json's H (image px
  -> court cm) at the episode's court point (near_court of its evidence frame in frames.jsonl,
  read by court_points; else the line's middle): a step image-down
  (toward the camera) that does not clearly move into the court side of the line means the
  line lies toward the camera. Without a usable geometry a 踏んだ is 要確認 too;
- sideline verdicts carry 「サイドラインの判定は、まだ正解データでの確認が済んでいません。」
  until a GT check of the sideline has side_agreement n > 0 (n = 0 today).

Only synthetic data: an invented pinhole camera behind the top line.
"""
import json
import math

import numpy as np
import pytest

from pfb_app.ui import linecall_view as lv
from test_linecall_view import RULING_WORDS
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN

CW, CH = 1505.0, 1105.0


def court_to_image(f=1000.0, cx=960.0, cy=540.0, h=300.0, x0=752.5, yc=CH + 600.0):
    """A pinhole camera h cm up at (x0, yc), looking along -y (over the top line, toward the
    basket): u = cx + f (x - x0) / D, v = cy + f h / D with D = yc - y."""
    return np.array([[f, -cx, cx * yc - f * x0], [0.0, -cy, cy * yc + f * h], [0.0, -1.0, yc]])


def write_geometry(lc_dir, M=None, **extra):
    """geometry.json as engine 58f8341 writes it (H = image -> court, h_unc_group)."""
    lc_dir.mkdir(parents=True, exist_ok=True)
    H = np.linalg.inv(court_to_image() if M is None else M)
    doc = {"status": "判定材料", "source": "tid_meta", "H": (H / H[2, 2]).tolist(),
           "H_frames": None, "lens": None, "lens_status": None, "h_unc_cm": 18.0,
           "h_unc_group": {"sideline": 15.0, "baseline": 15.0, "top_line": 15.0, "arc": 18.0},
           "notes": [], "autocourt": None, **extra}
    (lc_dir / "geometry.json").write_text(json.dumps(doc), encoding="utf-8")
    return lc_dir


def episode(line, verdict="stepped", frame=131, **extra):
    """An episode of engine 58f8341's episodes.json (evidence: [{side, method, d, unc}])."""
    return {"id": f"P2-{line}-120", "label": "P2", "line": line, "verdict": verdict,
            "frame": frame, "clock": "0:04", "video_s": 4.3,
            "evidence": [{"side": s, "method": "ankle", "d": -3.0, "unc": 4.0} for s in "LR"],
            "distances": {"L": -3.0, "R": 12.0}, "uncertainty_cm": {"L": 4.0, "R": 4.0},
            **extra}


def frame_row(eid, frame, foot, near):
    """A frames.jsonl row of engine 58f8341 (the T11 fields included)."""
    return {"episode": eid, "frame": frame, "foot": foot, "side": foot, "lo": 1.0, "hi": 9.0,
            "d_cm": 5.0, "near_court": near, "hi_raw": 8.0, "frame_would_step": False,
            "ground_gap_px": 3.0, "line_found": True, "frame_verdict": "undecidable",
            "frame_reason": "margin", "verdict": "undecidable", "note": None}


def toward(line, near, geom, verdict="stepped"):
    return lv.toward_camera(episode(line, verdict), geom, near)


@pytest.fixture
def geom(tmp_path):
    return lv.load_geometry(write_geometry(tmp_path / "linecall"))


def test_load_geometry(tmp_path, geom):
    assert geom is not None and geom.H.shape == (3, 3)
    assert lv.load_geometry(tmp_path / "nowhere") is None                # no file
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "geometry.json").write_text('{"H": [[1, 0], [0, 1]]}', encoding="utf-8")
    assert lv.load_geometry(bad) is None                                 # not a 3x3
    (bad / "geometry.json").write_text("{torn", encoding="utf-8")
    assert lv.load_geometry(bad) is None


def test_lines_toward_the_camera(geom):
    """Camera behind the top line: the top line is toward it, the baseline and the sidelines
    are not; the arc is toward it for a player inside it, not for one outside its top."""
    assert toward("top_line", [700.0, 1080.0], geom) is True
    assert toward("baseline", [700.0, 20.0], geom) is False
    assert toward("sideline_L", [40.0, 600.0], geom) is False
    assert toward("sideline_R", [1465.0, 600.0], geom) is False
    assert toward("arc", [752.5, 760.0], geom) is True          # inside the arc
    assert toward("arc", [752.5, 860.0], geom) is False         # outside its top
    # no court point: it cannot be told (a 踏んだ is then 要確認)
    assert toward("top_line", None, geom) is None
    assert toward("sideline_L", None, geom) is None


def test_the_player_s_side_and_the_band(geom):
    """Final review MED 4: within the band (the line group's H uncertainty, at least
    SIDE_BAND_MIN_CM) the player may be on either side: toward. Beyond it, outline lines use
    the player's side too: outside the far baseline the line lies toward the camera."""
    assert geom.band_cm("baseline") == 15.0 and geom.band_cm("arc") == 18.0
    assert toward("baseline", [700.0, -15.0], geom) is True      # 15 cm outside, in the band
    assert toward("baseline", [700.0, -40.0], geom) is True      # outside: line toward camera
    assert toward("baseline", [700.0, 40.0], geom) is False      # inside, beyond the band
    assert toward("top_line", [700.0, 1140.0], geom) is False    # outside the near line
    assert toward("arc", [752.5, 805.0], geom) is True           # the stripe: no sign flip
    assert toward("arc", [752.5, 812.0], geom) is True
    assert toward("sideline_L", [5.0, 600.0], geom) is True       # in the band


def test_band_floor_without_an_uncertainty(tmp_path):
    g = lv.load_geometry(write_geometry(tmp_path / "meta", h_unc_cm=0.0, h_unc_group=None))
    assert g.band_cm("arc") == lv.SIDE_BAND_MIN_CM == 15.0


def test_camera_at_the_side_makes_that_sideline_near(tmp_path):
    """A camera beside the left sideline (looking along +x): sideline_L is toward it."""
    f, cx, cy, h, xc = 1000.0, 960.0, 540.0, 300.0, -600.0
    # D = x - xc; u = cx + f (y - 552.5) / D, v = cy + f h / D
    M = np.array([[cx, f, -f * 552.5 - cx * xc], [cy, 0.0, f * h - cy * xc], [1.0, 0.0, -xc]])
    g = lv.load_geometry(write_geometry(tmp_path / "side", M))
    assert toward("sideline_L", [20.0, 500.0], g) is True
    assert toward("sideline_R", [1460.0, 500.0], g) is False
    assert toward("top_line", [700.0, 1060.0], g) is False


def test_unknown_geometry_is_none(geom):
    assert toward("top_line", [700.0, 1080.0], None) is None
    assert toward("nowhere", [700.0, 1080.0], geom) is None
    assert toward("baseline", ["x", None], geom) is None


def test_court_points_from_frames_jsonl(tmp_path):
    d = tmp_path / "linecall"
    d.mkdir()
    assert lv.court_points(d, {"a": 5}) == {}                     # no frames.jsonl
    rows = [frame_row("a", 4, "L", [1.0, 2.0]), frame_row("a", 5, "L", [700.0, 1080.0]),
            frame_row("a", 5, "R", [701.0, 1081.0]), frame_row("b", 5, "L", [9.0, 9.0]),
            frame_row("c", 5, "L", None), frame_row('q"x', 7, "L", [3.0, 4.0])]
    (d / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{torn",
                                    encoding="utf-8")
    assert lv.court_points(d, {"a": 5, "c": 5, 'q"x': 7, "z": 1}) == {
        "a": (700.0, 1080.0), 'q"x': (3.0, 4.0)}
    assert lv.court_points(d, {}) == {}
    eps = [episode("top_line"), episode("baseline", "clear"), {"id": "x", "verdict": "stepped"}]
    assert lv.stepped_frames(eps) == {"P2-top_line-120": 131}


def test_stepped_toward_the_camera_is_to_be_checked(geom):
    near, far = episode("top_line"), episode("baseline")
    pts = {near["id"]: (700.0, 1080.0)}
    assert lv.shown_verdict(near, geom, pts) == "check"
    assert lv.shown_verdict(far, geom, {far["id"]: (700.0, 40.0)}) == "stepped"
    assert lv.shown_verdict(far, geom) == "check"                # no court point: 要確認
    assert lv.shown_verdict(far, None, {far["id"]: (700.0, 20.0)}) == "check"   # no camera
    assert lv.shown_verdict(episode("top_line", "clear"), geom, pts) == "clear"
    assert lv.shown_verdict(episode("top_line", "undecidable"), None) == "undecidable"
    assert lv.shown_verdict({"verdict": "weird"}, geom) == "undecidable"
    assert lv.episode_row(near, geom, pts)["判定材料"] == "踏んだ（要確認）"
    assert lv.episode_row(far, geom, {far["id"]: (700.0, 40.0)})["判定材料"] == "踏んだ"
    assert lv.episode_row(far)["判定材料"] == "踏んだ（要確認）"   # no geometry given
    assert lv.SHOWN_LABELS["check"] == "踏んだ（要確認）"            # final review MED 5
    assert lv.NEAR_LINE_NOTE in lv.episode_notes(near, "check")
    assert lv.NEAR_LINE_NOTE not in lv.episode_notes(far, "stepped")


def test_sideline_verdicts_are_not_validated(geom):
    assert lv.SIDELINE_NOTE == "サイドラインの判定は、まだ正解データでの確認が済んでいません。"
    assert lv.SIDELINE_VALIDATED is False
    for line in ("sideline_L", "sideline_R"):
        for verdict in ("stepped", "clear"):
            ep = episode(line, verdict)
            assert lv.SIDELINE_NOTE in lv.episode_notes(ep, lv.shown_verdict(ep, geom))
        und = episode(line, "undecidable")
        assert lv.SIDELINE_NOTE not in lv.episode_notes(und, "undecidable")
    assert lv.SIDELINE_NOTE not in lv.episode_notes(episode("baseline", "clear"), "clear")


def test_new_copy_has_no_banned_words():
    for text in (lv.NEAR_LINE_NOTE, lv.SIDELINE_NOTE, *lv.SHOWN_LABELS.values()):
        found = [w for w in (*FORBIDDEN, *SCHOOL_WORDS, *RULING_WORDS, "ルーブリック")
                 if w in text]
        assert not found, (text, found)


def test_new_engine_rows_parse(tmp_path):
    """episodes.json / frames.jsonl / geometry.json of engine 58f8341 (rows with hi_raw /
    frame_would_step / ground_gap_px, geometry with h_unc_group) read as before."""
    d = write_geometry(tmp_path / "linecall")
    eps = [episode("top_line"), episode("arc", "undecidable", reason="margin")]
    (d / "episodes.json").write_text(json.dumps({"status": "判定材料", "episodes": eps}),
                                     encoding="utf-8")
    (d / "frames.jsonl").write_text(json.dumps(frame_row(eps[0]["id"], 131, "L",
                                                         [700.0, 1080.0])) + "\n",
                                    encoding="utf-8")
    got = lv.load_episodes(d / "episodes.json")
    assert [e["id"] for e in got] == [e["id"] for e in eps]
    assert lv.distance_text(got[0]).startswith("左足: 線の上（誤差 ±4 cm）")
    assert lv.reason_text(got[1]) == "判定不能の理由: 線に近すぎて決められません"
    g = lv.load_geometry(d)
    assert not math.isnan(g.H[0, 0])
    pts = lv.court_points(d, lv.stepped_frames(got))
    assert [lv.shown_verdict(e, g, pts) for e in got] == ["check", "undecidable"]
