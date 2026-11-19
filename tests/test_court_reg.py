"""The registered court that follows a panning camera, and who stands outside it
(pfb_app.review_video.court_reg). Only synthetic pictures and numbers."""
from types import SimpleNamespace

import numpy as np
import pytest

from pfb_app.review_video import court_reg as cr
from pfb_app.review_video import player

QUAD = [[100.0, 200.0], [1100.0, 200.0], [1500.0, 650.0], [-300.0, 650.0]]       # far-left, far-right, near-right, near-left


def _rows(court_xy_m, n, step=3, h=100.0, f0=1):
    """Track rows of a person whose feet are at the given court position (metres), seen with QUAD."""
    quad = np.array(QUAD, float)
    # the picture position of a court point: the inverse of plane_xy by bilinear-in-homography: use cv2-free solve
    dst = np.array([[0, 0], [cr.COURT_LEN_M, 0], [cr.COURT_LEN_M, cr.COURT_WID_M], [0, cr.COURT_WID_M]], float)
    A = []
    for (u, v), (x, y) in zip(dst, quad):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    Hm = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    p = Hm @ np.array([court_xy_m[0], court_xy_m[1], 1.0])
    fx, fy = p[0] / p[2], p[1] / p[2]
    return [[f0 + i * step, fx - 20, fy - h, 40.0, h] for i in range(n)]


COURT = [[f] + [v for pt in QUAD for v in pt] for f in range(1, 400, 15)]


def test_plane_position_of_the_corners_and_the_middle():
    q = np.array(QUAD, float)
    assert cr.plane_xy(q, q[0]) == pytest.approx((0, 0), abs=1e-6)
    assert cr.plane_xy(q, q[2]) == pytest.approx((28, 15), abs=1e-6)
    x, y = cr.plane_xy(q, _rows((14, 7.5), 1)[0][1:3] + np.array([20, 100]))
    assert (x, y) == pytest.approx((14, 7.5), abs=1e-3)
    assert cr.outside_m((14, 7.5)) < 0 < cr.outside_m((14, 16.2))


def test_a_registration_must_be_four_ordered_corners():
    ok = {"ref_t": 1.0, "corners_ref": QUAD}
    assert cr.validate_registration(ok) is ok
    with pytest.raises(ValueError):
        cr.validate_registration({"ref_t": 1.0, "corners_ref": QUAD[:3]})
    with pytest.raises(ValueError):                                   # a bow-tie
        cr.validate_registration({"ref_t": 1.0, "corners_ref": [QUAD[0], QUAD[2], QUAD[1], QUAD[3]]})
    with pytest.raises(ValueError):
        cr.validate_registration({"corners_ref": QUAD})


def test_the_outline_moves_with_the_camera_and_is_dropped_when_a_corner_is_behind_it():
    shift = np.array([[1, 0, 50.0], [0, 1, -20.0], [0, 0, 1]])        # picture -> reference: the camera panned
    rows = cr.quad_rows(QUAD, {5: shift, 6: None}, 1280, 720)
    assert [r[0] for r in rows] == [5]
    got = np.array(rows[0][1:]).reshape(4, 2)
    assert got == pytest.approx(np.array(QUAD) - [50.0, -20.0], abs=0.2)
    behind = np.array([[1, 0, 0], [0, 1, 0], [0.002, 0, 1.0]])         # reference points beyond the horizon of this view
    assert cr.quad_rows([[900, 100], [1100, 100], [1100, 600], [900, 600]], {7: behind}, 1280, 720) == []


def test_the_outline_between_two_samples_is_interpolated_and_not_made_up_far_from_any():
    rows = [[1] + [0.0] * 8, [31] + [30.0] * 8]
    mid = cr.interp_quad(rows, 16)
    assert mid is not None and mid[0] == pytest.approx([15.0, 15.0])
    assert cr.interp_quad(rows, 100) is None and cr.interp_quad([], 5) is None
    assert cr.interp_quad([[1] + [0.0] * 8, [200] + [1.0] * 8], 100) is None          # two far samples: not bridged


def test_standing_beyond_the_line_for_a_second_is_outside_but_not_less():
    far = _rows((14, -2.5), 12)                       # 2.5 m beyond the far sideline, 12 rows x 3 frames = 1.2 s
    assert cr.flag_rows(far, COURT, 30.0, 720, 3) == [1] * 12
    short = _rows((14, -2.5), 8)                      # 0.8 s: a step over the line, a ball chase
    assert sum(cr.flag_rows(short, COURT, 30.0, 720, 3)) == 0
    near_line = _rows((14, -0.6), 40)                 # 0.6 m beyond the line: inside the margin
    assert sum(cr.flag_rows(near_line, COURT, 30.0, 720, 3)) == 0
    inside = _rows((14, 7.5), 40)
    assert sum(cr.flag_rows(inside, COURT, 30.0, 720, 3)) == 0


def test_the_flag_does_not_blink_on_a_boundary_and_a_player_who_stood_inside_is_never_flagged():
    wobble = _rows((14, -1.6), 12) + _rows((14, -0.3), 1, f0=40) + _rows((14, -1.6), 12, f0=43)
    out = cr.flag_rows(wobble, COURT, 30.0, 720, 3)
    # the run is cut by the one inside row: both halves are 1.2 s long, each is flagged as a whole, and the
    # inside row itself is not; no single row flips on its own
    assert out[:12] == [1] * 12 and out[12] == 0 and out[13:] == [1] * 12
    stepped = _rows((14, 7.5), 24, f0=1) + _rows((14, -3.0), 30, f0=73)      # 2.4 s inside, then 3 s out
    assert sum(cr.flag_rows(stepped, COURT, 30.0, 720, 3)) == 0


def test_feet_that_are_not_seen_and_an_unknown_outline_are_never_flagged():
    cut = [[r[0], r[1], 720 - 100, r[3], 100.0] for r in _rows((14, -3.0), 15)]       # the box touches the picture's bottom
    assert sum(cr.flag_rows(cut, COURT, 30.0, 720, 3)) == 0
    assert sum(cr.flag_rows(_rows((14, -3.0), 15), [], 30.0, 720, 3)) == 0


def test_named_people_are_never_flagged_and_the_page_gets_the_flag_as_the_last_column():
    src = SimpleNamespace(fps=30.0, size=(1280, 720))
    people = [{"labels": ["P1"]}]
    tracks = {"P1": [r + [0] for r in _rows((14, -3.0), 15)], "P9": [r + [0] for r in _rows((14, -3.0), 15)]}
    extras = [r + [1, 0] for r in _rows((30.5, 7.0), 15)]
    t, e, info = player._flag_outside(COURT, src, 3, people, tracks, extras)
    assert [r[-1] for r in t["P1"]] == [0] * 15 and [r[-1] for r in t["P9"]] == [1] * 15
    assert len(t["P9"][0]) == 7 and len(e[0]) == 8 and [r[-1] for r in e] == [1] * 15
    assert info["n"] == 30 and info["margin_m"] == 1.0 and info["min_s"] == 1.0
    assert "選手とは限りません" in info["note"] and "審判はコートの中" in info["note"]


def test_the_camera_is_followed_on_a_synthetic_pan(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    world = np.full((900, 2200, 3), 120, np.uint8)
    for _ in range(900):                                    # texture: coloured blobs, lines
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, 2100)), int(rng.integers(0, 800))
        if rng.random() < .5:
            cv2.rectangle(world, (x, y), (x + int(rng.integers(6, 60)), y + int(rng.integers(6, 40))), c, -1)
        else:
            cv2.circle(world, (x, y), int(rng.integers(4, 25)), c, -1)
    path = str(tmp_path / "pan.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not vw.isOpened():
        pytest.skip("no video writer")
    offs = []
    for i in range(61):                                    # 2 s: the view moves 600 px right and 100 down
        ox, oy = int(10 + i * 10), int(20 + i * 1.5)
        offs.append((ox, oy))
        vw.write(world[oy:oy + 360, ox:ox + 640])
    vw.release()
    reg = {"ref_frame": 1, "corners_ref": [[100, 100], [500, 100], [560, 300], [40, 300]]}
    res = cr.register(path, 30.0, 61, reg)
    assert res and res["stats"]["lost"] == 0
    rows = {r[0]: r for r in res["frames"]}
    f = 46                                                  # (46 - 1) / 15 = 3: a sampled picture
    ox, oy = offs[f - 1]
    want = np.array([[100, 100], [500, 100], [560, 300], [40, 300]], float) - [ox - offs[0][0], oy - offs[0][1]]
    assert np.array(rows[f][1:]).reshape(4, 2) == pytest.approx(want, abs=4.0)


def test_a_cut_to_another_picture_gives_no_outline(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(5)
    a = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
    a = cv2.GaussianBlur(a, (0, 0), 2)
    b = cv2.GaussianBlur((rng.random((360, 640, 3)) * 255).astype(np.uint8), (0, 0), 2)
    path = str(tmp_path / "cut.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (640, 360))
    if not vw.isOpened():
        pytest.skip("no video writer")
    for i in range(46):
        vw.write(a if i < 16 else b)
    vw.release()
    res = cr.register(path, 30.0, 46, {"ref_frame": 1, "corners_ref": [[100, 100], [500, 100], [560, 300], [40, 300]]})
    frames = {r[0] for r in (res or {"frames": []})["frames"]}
    assert 31 not in frames and 46 not in frames          # after the cut: nothing is drawn
