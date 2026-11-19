"""Unnamed boxes for the people the tracking kept no identity for (export/extra_boxes.py)."""
import numpy as np

from pfb_app.export.extra_boxes import extra_boxes, load_extra_boxes, player_rows

FPS = 30.0


def _track(tid, frames, x, foot_y, h=100.0, w=40.0, conf=0.8, dx=0.0):
    return [(f, tid, x + dx * i, foot_y - h, w, h, conf) for i, f in enumerate(frames)]


def _scene():
    """Five persons on a 'court' (feet y 400..600, height grows with y); every other tracklet
    is added by the tests."""
    rows = []
    for i, (x, y) in enumerate([(100, 420), (300, 480), (500, 540), (700, 600), (900, 450)]):
        rows += _track(100 + i, range(1, 121), x, y, h=0.25 * y)
    return rows


def test_a_player_the_linking_dropped_gets_an_unnamed_box():
    kept = np.array(_scene())
    kept[:, 1] = np.arange(len(kept)) // 120 + 1            # person ids 1..5
    raw = np.vstack([kept, np.array(_track(9, range(1, 121), 400, 520, h=130.0))])
    ex = extra_boxes(raw, kept, FPS)
    assert set(ex[:, 1]) == {9} and len(ex) == 120


def test_a_person_that_is_already_boxed_is_not_boxed_twice():
    kept = np.array(_scene())
    raw = kept.copy()
    raw[:, 1] += 500                                          # the tracker's own ids
    assert len(extra_boxes(raw, kept, FPS)) == 0


def test_spectators_in_the_stands_and_flickers_get_no_box():
    kept = np.array(_scene())
    stands = np.array(_track(7, range(1, 121), 400, 150, h=45.0))      # far above, small
    huge = np.array(_track(8, range(1, 121), 600, 900, h=300.0))       # near the camera, off court
    flick = np.array(_track(6, range(10, 14), 400, 520, h=130.0))      # 4 frames only
    ex = extra_boxes(np.vstack([kept, stands, huge, flick]), kept, FPS)
    assert len(ex) == 0


def test_a_second_box_on_the_same_player_is_dropped():
    kept = np.array(_scene())
    a = np.array(_track(9, range(1, 121), 400, 520, h=130.0, conf=0.9))
    b = np.array(_track(10, range(1, 121), 404, 522, h=130.0, conf=0.4))
    ex = extra_boxes(np.vstack([kept, a, b]), kept, FPS)
    assert set(ex[:, 1]) == {9}


def test_too_few_persons_or_no_boxes_give_nothing():
    assert len(extra_boxes(np.zeros((0, 7)), np.zeros((0, 7)), FPS)) == 0
    two = np.array(_track(1, range(1, 5), 10, 400) + _track(2, range(1, 5), 200, 400))
    assert len(extra_boxes(two, two, FPS)) == 0


def test_loading_from_a_tracks_directory(tmp_path):
    kept = np.array(_scene())
    kept[:, 1] = np.arange(len(kept)) // 120 + 1
    raw = np.vstack([kept, np.array(_track(9, range(1, 121), 400, 520, h=130.0))])
    np.savetxt(tmp_path / "boxes.txt", kept, delimiter=",", fmt="%g")
    np.savetxt(tmp_path / "raw_tracks.txt", raw, delimiter=",", fmt="%g")
    (tmp_path / "meta.json").write_text('{"fps": 30.0}', encoding="utf-8")
    ex = load_extra_boxes(tmp_path)
    assert set(ex[:, 1]) == {9}
    rows = player_rows(ex, 3)
    assert rows[0][0] % 3 == 0 and len(rows[0]) == 5 and len(rows) == 40


def test_a_missing_or_broken_file_is_no_extra_boxes(tmp_path):
    assert len(load_extra_boxes(tmp_path)) == 0
    (tmp_path / "boxes.txt").write_text("not,a,box\n", encoding="utf-8")
    (tmp_path / "raw_tracks.txt").write_text("", encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    assert len(load_extra_boxes(tmp_path)) == 0
