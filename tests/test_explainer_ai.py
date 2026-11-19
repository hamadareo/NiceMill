"""「AIに見せた範囲」: the engine's window frames and crop region ported 1:1, the recorded stills
and the check of a recomputed region against them (synthetic images only)."""
from pathlib import Path

import cv2
import numpy as np

from pfb_app.export import explainer_ai as ai
from pfb_app.export.tracks import Box, BoxIndex


def test_action_region_matches_the_engines_formula():
    full = ai.action_region([], 1920, 1080)
    assert full == (0.0, 0.0, 1920.0, 1080.0)
    # small boxes: at least 40 % of the width, 16:9, pushed inside the frame
    assert ai.action_region([Box(100, 100, 50, 100), Box(300, 120, 40, 80)], 1920, 1080) == \
        (0.0, 0.0, 768.0, 432.0)
    assert ai.action_region([Box(1500, 500, 100, 200)], 1920, 1080) == \
        (1152.0, 384.0, 1920.0, 816.0)
    # wide spread: the union + 15 % at 16:9
    x0, y0, x1, y1 = ai.action_region([Box(100, 400, 50, 100), Box(1400, 450, 50, 100)],
                                      1920, 1080)
    assert (round(x1 - x0, 3), round((x1 - x0) / (y1 - y0), 4)) == (1755.0, round(16 / 9, 4))


def test_window_frames_as_make_windows():
    assert ai.window_frames(0, 0.5, 12, 10.0, 0.0, 10_000) == [1 + 5 * k for k in range(12)]
    assert ai.window_frames(1, 0.5, 12, 10.0, 0.0, 10_000)[:2] == [61, 66]
    assert ai.window_frames(0, 0.5, 12, 30.0, 2.0, 10_000)[:3] == [61, 76, 91]
    assert ai.window_frames(1, 0.5, 12, 10.0, 0.0, 63) is None      # one frame left: no window
    assert ai.window_frames(1, 0.5, 12, 10.0, 0.0, 70) == [61, 66]


def _frame(seed=3, w=640, h=360):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _engine_still(frame, region, path: Path, boxes=()):
    """As PFB3x3 does: crop the region, resize to 960 wide, draw the marks, JPEG."""
    x0, y0, x1, y1 = (round(v) for v in region)
    crop = frame[y0:y1, x0:x1]
    out = cv2.resize(crop, (960, round(crop.shape[0] * 960 / crop.shape[1])))
    s = 960 / (region[2] - region[0])
    for b in boxes:
        cv2.rectangle(out, (round((b.x - region[0]) * s), round((b.y - region[1]) * s)),
                      (round((b.x + b.w - region[0]) * s), round((b.y + b.h - region[1]) * s)),
                      (70, 160, 0), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def _index(frames, box):
    return BoxIndex({"P1": [(f, box.x, box.y, box.w, box.h) for f in frames],
                     "P2": [(f, 400.0, 100.0, 30.0, 70.0) for f in frames[:3]]})


def test_ai_input_recomputes_and_verifies_against_the_recorded_stills(tmp_path):
    frames = [1 + 5 * k for k in range(12)]
    box = Box(100, 120, 30, 72)
    index = _index(frames, box)
    region = ai.action_region([Box(100, 120, 30, 72), Box(400, 100, 30, 70)], 640, 360)
    frame = _frame()
    run = tmp_path / "exp" / "qwen3vl-8b" / "clip_som" / "s1_cam1"
    stills = [_engine_still(frame, region, run / "overlay" / "s1_cam1_w0000" / f"{k}.jpg", [box])
              for k in range(1, 13)]
    got = ai.ai_input([str(p) for p in stills], index, 10.0, (640, 360), 0.0, 600)
    assert got.stills == stills and got.frames == frames
    assert got.region == region
    assert got.still_at(28) == 5 and got.still_at(-50) == 0
    # another picture is not what the model got
    assert not ai.verify(_frame(seed=9), region, stills[0])


def test_without_the_condition_or_with_other_counts_only_the_stills(tmp_path):
    frame = _frame()
    region = (0.0, 0.0, 640.0, 360.0)
    index = _index([1 + 5 * k for k in range(12)], Box(100, 120, 30, 72))
    odd = tmp_path / "exp" / "m" / "mystery" / "v" / "overlay" / "v_w0000"
    still = _engine_still(frame, region, odd / "1.jpg")
    got = ai.ai_input([str(still)], index, 10.0, (640, 360), 0.0, 600)
    assert got.stills == [still] and got.frames is None and got.region is None
    few = tmp_path / "exp" / "m" / "clip_som" / "v" / "overlay" / "v_w0000"
    shots = [_engine_still(frame, region, few / f"{k}.jpg") for k in (1, 2, 3)]
    got = ai.ai_input([str(p) for p in shots], index, 10.0, (640, 360), 0.0, 600)
    assert got.frames is None and got.still_at(40) == 0
    assert ai.ai_input([str(tmp_path / "gone.jpg")], index, 10.0, (640, 360), 0.0, 600).stills == []


def _window(tmp_path, region, frames_of, boxes_at_frames, name="v"):
    """12 stills as the engine sent them, frame k cut from frames_of(f_k)."""
    frames = [1 + 5 * k for k in range(12)]
    run = tmp_path / "exp" / "m" / "clip_som" / name / "overlay" / f"{name}_w0000"
    stills = [str(_engine_still(frames_of(f), region, run / f"{k}.jpg", boxes_at_frames))
              for k, f in enumerate(frames, 1)]
    return frames, stills


def test_a_region_is_drawn_only_when_two_stills_match_better_than_moved_ones(tmp_path):
    box, other = Box(100, 120, 30, 72), Box(300, 100, 30, 70)
    frames = [1 + 5 * k for k in range(12)]
    index = BoxIndex({"P1": [(f, box.x, box.y, box.w, box.h) for f in frames],
                      "P2": [(f, other.x, other.y, other.w, other.h) for f in frames]})
    region = ai.action_region([box, other], 640, 360)
    assert ai.cover(region, (640, 360)) < 0.9
    pictures = {f: _frame(seed=f) for f in frames}           # every frame its own picture
    _, stills = _window(tmp_path, region, pictures.get, [box, other])
    info = ai.ai_input(stills, index, 10.0, (640, 360), 0.0, 600)
    got = ai.check_region(info, pictures.get, (640, 360), (1.0, 1.0), near=30)
    assert got.drawn and got.matches >= 2
    # only one still is what the frames show: not drawn
    _, wrong = _window(tmp_path, region, lambda f: _frame(seed=1000 + f), [box, other], "w")
    mixed = [stills[0], *wrong[1:]]
    info = ai.ai_input(mixed, index, 10.0, (640, 360), 0.0, 600)
    info.stills = [Path(p) for p in mixed]
    assert not ai.check_region(info, pictures.get, (640, 360), (1.0, 1.0), near=1).drawn
    # a moved region does not beat the true one
    x0, y0, x1, y1 = region
    assert not ai.verify(pictures[1], (x0 + 40, y0, x1 + 40, y1), Path(stills[0]))


def test_a_near_full_frame_region_is_never_drawn(tmp_path):
    left, right = Box(5, 20, 40, 90), Box(590, 240, 40, 100)     # people at both edges
    frames = [1 + 5 * k for k in range(12)]
    index = BoxIndex({"P1": [(f, left.x, left.y, left.w, left.h) for f in frames],
                      "P2": [(f, right.x, right.y, right.w, right.h) for f in frames]})
    region = ai.action_region([left, right], 640, 360)
    assert ai.cover(region, (640, 360)) >= 0.9
    pictures = {f: _frame(seed=f) for f in frames}
    _, stills = _window(tmp_path, region, pictures.get, [left, right])
    info = ai.ai_input(stills, index, 10.0, (640, 360), 0.0, 600)
    got = ai.check_region(info, pictures.get, (640, 360), (1.0, 1.0), near=30)
    assert not got.drawn and got.cover >= 0.9
