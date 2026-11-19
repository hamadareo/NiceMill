"""Everyone framed in the colour of their kit, what they seem to be doing, the court's sidelines
(pfb_app.review_video.everyone / court_est). Only synthetic pictures: drawn rectangles."""
import re

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pfb_app.review_video import court_est, everyone as ev  # noqa: E402


def test_colour_words_follow_the_measured_colour():
    assert ev.colour_name((235, 235, 238)) == "白"
    assert ev.colour_name((130, 135, 148)) == "白"        # a white kit in a dim picture
    assert ev.colour_name((20, 20, 24)) == "黒"
    assert ev.colour_name((200, 30, 35)) == "赤"
    assert ev.colour_name((30, 80, 200)) == "青"
    assert ev.colour_name((40, 40, 90)) == "紺"
    assert ev.colour_name((240, 210, 30)) == "黄"


def test_a_white_team_is_drawn_white_with_a_black_rim_and_a_black_team_black_with_a_white_rim():
    st = ev.team_style((130, 135, 148), (64, 59, 61))
    assert st["A"]["hex"] == "#FFFFFF" and st["A"]["rim"] == "#000000" and st["A"]["name"] == "白"
    assert st["B"]["rim"] == "#FFFFFF" and st["B"]["name"] == "黒"
    assert st["adjusted"] is False
    red = ev.team_style((200, 30, 35), (240, 240, 240))
    assert red["A"]["name"] == "赤" and red["B"]["name"] == "白" and not red["adjusted"]


def test_two_alike_kits_are_moved_apart_and_it_is_said():
    st = ev.team_style((200, 30, 35), (205, 40, 30))
    assert st["adjusted"] is True
    d = ev.lab_dist(ev.rgb_to_lab(ev.rgb_of(st["A"]["hex"]))[0], ev.rgb_to_lab(ev.rgb_of(st["B"]["hex"]))[0])
    assert d >= ev.SEP_MIN
    assert st["B"]["name"] not in ("赤", "ピンク")          # red is the court's line


def test_the_words_never_name_a_basketball_action_and_are_hedged():
    for w in ev.ACT_WORDS[1:]:
        assert not re.search(r"シュート|パス|ドリブル|得点|ゴール|ブロック|リバウンド|ディフェンス|守", w), w
        assert w.endswith("見える") or w.endswith("（推定）"), w


def test_speed_with_the_camera_pan_taken_out_tells_standing_from_running():
    fps, step = 30.0, 3
    still = [[f, 500, 300, 40, 120] for f in range(3, 300, step)]
    run = [[f, 500 + 0.5 * f * 5, 300, 40, 120] for f in range(3, 300, step)]       # 2.5 px/frame = 75 px/s = 0.6 heights/s
    fast = [[f, 500 + f * 12, 300, 40, 120] for f in range(3, 150, step)]            # 360 px/s = 3 heights/s
    sh0 = {r[0]: (0.0, 0.0) for r in still}
    assert ev.activity(still, fps, step, sh0)[20] == ev.ACT_STAND
    assert ev.activity(run, fps, step, {r[0]: (0.0, 0.0) for r in run})[20] == ev.ACT_WALK
    assert ev.activity(fast, fps, step, {r[0]: (0.0, 0.0) for r in fast})[20] == ev.ACT_RUN
    # the same person standing still while the camera pans: the box moves, the pan is removed
    pan = [[f, 500 + f * 12, 300, 40, 120] for f in range(3, 150, step)]
    shift = {r[0]: (r[0] * 12.0, 0.0) for r in pan}
    assert ev.activity(pan, fps, step, shift)[20] == ev.ACT_STAND


def test_the_ball_and_the_pose_come_before_the_motion_and_a_single_frame_does_not_flicker():
    fps, step = 30.0, 3
    rows = [[f, 500, 300, 40, 120] for f in range(3, 200, step)]
    sh = {r[0]: (0.0, 0.0) for r in rows}
    n = len(rows)
    arms = [1 if 10 <= i < 20 else 0 for i in range(n)]
    one = [1 if i == 5 else 0 for i in range(n)]
    out = ev.activity(rows, fps, step, sh, arms)
    assert out[15] == ev.ACT_ARMS and out[3] == ev.ACT_STAND
    assert ev.ACT_ARMS not in ev.activity(rows, fps, step, sh, one)
    ball = {rows[15][0]}
    assert ev.activity(rows, fps, step, sh, arms, ball)[15] == ev.ACT_BALL
    crouch = [2 if 10 <= i < 20 else 0 for i in range(n)]
    assert ev.activity(rows, fps, step, sh, crouch)[15] == ev.ACT_CROUCH


def test_unnamed_boxes_are_linked_into_things_that_keep_their_own_track():
    step = 3
    rows = []
    for f in range(3, 90, step):
        rows.append([f, 100 + f, 300, 40, 120])
        rows.append([f, 900 - f, 320, 40, 120])
    tid = ev.link_rows(rows, step)
    assert len(set(tid)) == 2
    assert len({tid[i] for i in range(0, len(rows), 2)}) == 1 and len({tid[i] for i in range(1, len(rows), 2)}) == 1


def test_pose_is_matched_to_a_box_by_overlap_only():
    pose = {3: [(100, 100, 40, 120, 1)], 6: [(600, 100, 40, 120, 1)]}
    assert ev.pose_flags([[3, 102, 102, 40, 120], [6, 100, 100, 40, 120], [9, 100, 100, 40, 120]], pose) == [1, None, None]


def _clip(path, frames=30, w=640, h=360):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    for _ in range(frames):
        img = np.full((h, w, 3), (90, 140, 190), np.uint8)             # a wooden floor (BGR)
        cv2.rectangle(img, (100, 150), (140, 270), (235, 235, 235), -1)   # a white kit
        cv2.rectangle(img, (200, 150), (240, 270), (235, 235, 235), -1)
        cv2.rectangle(img, (300, 150), (340, 270), (30, 30, 30), -1)      # a black kit
        cv2.rectangle(img, (400, 150), (440, 270), (30, 30, 30), -1)
        cv2.rectangle(img, (500, 150), (540, 270), (200, 60, 150), -1)    # a purple shirt: neither kit
        vw.write(img)
    vw.release()


def test_everyone_gets_the_team_of_their_kit_and_a_shirt_of_neither_team_stays_out(tmp_path):
    clip = tmp_path / "c.avi"
    _clip(clip)
    tracks = {"P1": [[f, 100, 150, 40, 120] for f in range(3, 30, 3)], "P2": [[f, 300, 150, 40, 120] for f in range(3, 30, 3)],
              "P3": [[f, 400, 150, 40, 120] for f in range(3, 30, 3)], "P4": [[f, 200, 150, 40, 120] for f in range(3, 30, 3)]}
    people = [{"labels": ["P1"], "tm": "A", "excluded": False}, {"labels": ["P2"], "tm": "B", "excluded": False},
              {"labels": ["P3"], "tm": "B", "excluded": False}]
    extras = [[f, 500, 150, 40, 120] for f in range(3, 30, 3)] + [[f, 200, 150, 40, 120] for f in range(3, 30, 3)]
    out = ev.analyse(clip, 30.0, 3, people, tracks, extras)
    assert out["style"]["A"]["name"] == "白" and out["style"]["B"]["name"] == "黒"
    assert out["others"] == {"P4": 1}                                    # an unlisted track in the white kit
    teams = {r[1]: r[5] for r in out["extras"]}
    assert teams[200] == 1 and teams[500] == 0                           # white kit -> team A; the purple shirt: neither
    assert all(len(r) == 7 for r in out["extras"]) and all(len(r) == 6 for r in out["tracks"]["P1"])


def test_no_picture_no_invented_colours(tmp_path):
    assert ev.analyse(tmp_path / "none.avi", 30.0, 3, [], {"P1": [[3, 1, 1, 40, 120]]}, None) is None


def _hall(w=1280, h=720):
    img = np.full((h, w, 3), (120, 120, 120), np.uint8)               # boards / stands
    cv2.rectangle(img, (0, 300), (w, 316), (140, 60, 20), -1)         # the painted strip (BGR navy-ish blue)
    cv2.rectangle(img, (0, 316), (w, h), (90, 150, 205), -1)          # the wooden floor
    return img


def test_the_far_sideline_is_found_where_the_painted_strip_meets_the_wood():
    d = court_est.detect(_hall())
    assert d["far"] is not None
    y0, y1 = d["far"][1], d["far"][3]
    assert abs(y0 - 316) < 8 and abs(y1 - 316) < 8
    assert d["far"][2] - d["far"][0] > 500


def test_a_line_the_players_stand_on_the_wrong_side_of_is_not_drawn():
    est = {"samples": [[f, [100, 316, 1100, 316], None] for f in range(1, 200, 15)], "note": "推定"}
    below = np.array([[f, x, 450] for f in range(1, 200, 3) for x in (300, 500, 700, 900)], float)
    above = np.array([[f, x, 200] for f in range(1, 200, 3) for x in (300, 500, 700, 900)], float)
    assert court_est.validate(est, below, 30.0) is not None
    assert court_est.validate(est, above, 30.0) is None
    assert court_est.validate(est, np.zeros((0, 3)), 30.0) is None     # nobody to confirm it: nothing is drawn
    assert court_est.validate(None, below, 30.0) is None


def test_a_picture_without_a_painted_strip_gives_no_line():
    img = np.full((720, 1280, 3), (90, 150, 205), np.uint8)
    d = court_est.detect(img)
    assert d == {"far": None, "near": None}
