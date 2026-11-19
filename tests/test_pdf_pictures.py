"""export.tracks and export.pictures on synthetic data only: a generated video with one
bright moving block as the "person", a MOT boxes.txt written here, a made-up homography."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pfb_app.export import pictures, tracks
from pfb_app.export.pictures import Box, PictureRequest

FPS, W, H = 10, 320, 240


def person_at(frame: int) -> tuple[int, int, int, int]:
    """The block's box (x, y, w, h) in 1-based `frame`: walks right, 2 px a frame."""
    return 20 + 2 * (frame - 1), 80, 24, 60


def synthetic_video(path: Path, seconds: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(1, seconds * FPS + 1):
        img = np.full((H, W, 3), 60, dtype=np.uint8)
        x, y, bw, bh = person_at(f)
        img[y:y + bh, x:x + bw] = (40, 200, 240)
        img[10:30, 250:300] = (60, 60, 220)            # someone else (red), far away
        w.write(img)
    w.release()
    return path


def boxes_file(path: Path, seconds: int = 6, gap=range(25, 35)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in range(1, seconds * FPS + 1):
        if f in gap:                                    # 1 s off screen
            continue
        x, y, bw, bh = person_at(f)
        rows.append(f"{f},1,{x},{y},{bw},{bh},0.9")
        rows.append(f"{f},2,250,10,50,20,0.8")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (path.parent / "meta.json").write_text(json.dumps({"fps": FPS, "n_frames": seconds * FPS,
                                                       "width": W, "height": H}))
    return path


# --- tracks -------------------------------------------------------------------------------

def test_box_index_nearest_and_spans(tmp_path):
    idx = tracks.load_boxes(boxes_file(tmp_path / "t" / "boxes.txt"))
    assert set(idx.labels) == {"P1", "P2"}
    frame, box = idx.nearest(["P1"], 12, tol=3)
    assert frame == 12 and (box.x, box.y, box.w, box.h) == person_at(12)
    frame, _ = idx.nearest(["P1"], 30, tol=10)       # inside the gap: the closest box
    assert frame in (24, 35)
    assert idx.nearest(["P1"], 30, tol=2) is None
    assert idx.nearest(["P9"], 10, tol=5) is None
    spans = tracks.spans(idx.frames(["P1"]), FPS, start_s=0.0, duration_s=6.0, gap_s=0.5)
    assert len(spans) == 2
    assert spans[0].start_s == pytest.approx(0.0) and spans[0].end_s == pytest.approx(2.3)
    assert spans[1].start_s == pytest.approx(3.4)
    merged = tracks.spans(idx.frames(["P1"]), FPS, start_s=0.0, duration_s=6.0, gap_s=2.0)
    assert len(merged) == 1


def test_spans_are_clipped_to_the_analysed_part(tmp_path):
    idx = tracks.load_boxes(boxes_file(tmp_path / "t" / "boxes.txt", gap=()))
    spans = tracks.spans(idx.frames(["P1"]), FPS, start_s=2.0, duration_s=3.0)
    assert spans[0].start_s == pytest.approx(0.0) and spans[-1].end_s <= 3.0


def test_unreadable_boxes_are_none(tmp_path):
    assert tracks.load_boxes(tmp_path / "missing.txt") is None
    assert tracks.track_meta(tmp_path / "missing.txt") is None


def test_no_geometry_means_no_court(tmp_path):
    boxes = boxes_file(tmp_path / "t" / "boxes.txt")
    assert tracks.load_geometry(boxes) is None
    (boxes.parent / "linecall").mkdir()
    (boxes.parent / "linecall" / "geometry.json").write_text(json.dumps(
        {"H": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}))
    assert tracks.load_geometry(boxes) is None          # a singular H is not a geometry
    (boxes.parent / "linecall" / "geometry.json").write_text("{not json")
    assert tracks.load_geometry(boxes) is None


def geometry(boxes: Path, H, **extra) -> None:
    (boxes.parent / "linecall").mkdir(exist_ok=True)
    (boxes.parent / "linecall" / "geometry.json").write_text(json.dumps(
        {"H": H, "lens": None, "source": "autocourt", **extra}))


def test_court_samples_and_distance(tmp_path):
    boxes = boxes_file(tmp_path / "t" / "boxes.txt", seconds=40, gap=())
    # 1 image px = 1 cm, no lens: the foot point walks 2 cm a frame = 0.2 m/s
    geometry(boxes, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    geom = tracks.load_geometry(boxes)
    idx = tracks.load_boxes(boxes)
    raw = tracks.court_samples(idx, ["P1"], geom, FPS, step_s=0.5)
    assert raw and all(len(s) == 3 for s in raw)
    f0, x0, y0 = raw[0]
    assert (x0, y0) == pytest.approx((person_at(f0)[0] + 12, 80 + 60))
    samples = tracks.clean_samples(raw, FPS)
    d = tracks.distance_m(samples, FPS)
    assert d == pytest.approx(7.9, abs=0.3)               # ~40 s at 0.2 m/s
    # a jump (a swapped track) is not walked
    jumpy = samples + [(samples[-1][0] + 5, 1400.0, 1000.0)]
    assert tracks.distance_m(jumpy, FPS) == pytest.approx(d, abs=0.01)


def test_distance_needs_thirty_seconds_on_the_court(tmp_path):
    walk = [(1 + 5 * k, 100.0 + 10 * k, 300.0) for k in range(59)]      # 29 s
    assert tracks.distance_m(walk, FPS) is None
    walk = [(1 + 5 * k, 100.0 + 10 * k, 300.0) for k in range(62)]      # 30.5 s
    assert tracks.distance_m(walk, FPS) == pytest.approx(6.1, abs=0.05)
    assert tracks.distance_m([(1, 0.0, 0.0), (11, 100.0, 0.0)], FPS) is None


def test_points_far_outside_the_court_are_dropped_and_spikes_smoothed():
    base = [(1 + 5 * k, 400.0, 300.0 + 2 * k) for k in range(80)]
    outside = [(base[-1][0] + 5, -150.0, 300.0)]                     # 1.5 m off the court
    near = [(base[-1][0] + 10, -50.0, 300.0)]                         # 0.5 m off: kept
    cleaned = tracks.clean_samples(base + outside + near, FPS)
    xs = [x for _, x, _ in cleaned]
    assert -150.0 not in xs and len(cleaned) == len(base) + 1
    spiky = list(base)
    f, x, y = spiky[40]
    spiky[40] = (f, x + 400.0, y)                                     # one wild sample
    smooth = tracks.clean_samples(spiky, FPS)
    assert tracks.distance_m(smooth, FPS) == pytest.approx(
        tracks.distance_m(tracks.clean_samples(base, FPS), FPS), abs=0.05)


def test_h_frames_are_found_next_to_the_geometry_file(tmp_path):
    boxes = boxes_file(tmp_path / "t" / "boxes.txt")
    (boxes.parent / "linecall").mkdir()
    np.save(boxes.parent / "linecall" / "H_frames.npy", np.stack([np.eye(3)] * 5))
    geometry(boxes, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], H_frames="H_frames.npy")
    geom = tracks.load_geometry(boxes)
    assert geom is not None and geom.H_frames is not None and len(geom.H_frames) == 5


# --- pictures -----------------------------------------------------------------------------

def test_crop_rect_is_about_two_and_a_half_boxes_and_stays_in_the_frame():
    x0, y0, x1, y1 = pictures.crop_rect(Box(100, 80, 24, 60), W, H)
    assert 0 <= x0 < 100 and x1 > 124 and 0 <= y0 < 80 and y1 > 140
    assert (x1 - x0) / (y1 - y0) == pytest.approx(0.75, abs=0.02)
    assert (x1 - x0) * (y1 - y0) == pytest.approx(2.5 * 24 * 2.5 * 60, rel=0.25)
    corner = pictures.crop_rect(Box(0, 0, 30, 70), W, H)
    assert corner[0] == 0 and corner[1] == 0 and corner[2] <= W and corner[3] <= H
    whole = pictures.crop_rect(Box(0, 0, W, H), W, H)     # bigger than the frame: fitted
    assert whole[1] == 0 and whole[3] == H and 0 <= whole[0] and whole[2] <= W
    assert (whole[2] - whole[0]) / H == pytest.approx(0.75, abs=0.01)


def test_highlighted_picture_from_the_video(tmp_path):
    video = synthetic_video(tmp_path / "v" / "cam1.mp4")
    req = PictureRequest(video=video, frame=21, box=Box(*person_at(21)))
    out = pictures.make_pictures([req], tmp_path / "cache")
    path = out[req]
    img = Image.open(path)
    assert img.format == "JPEG" and max(img.size) <= pictures.LONG_SIDE
    assert img.width / img.height == pytest.approx(0.75, abs=0.02)
    rgb = np.asarray(img.convert("RGB")).astype(int)
    blue = (abs(rgb[..., 0] - 42) < 40) & (abs(rgb[..., 1] - 120) < 40) & (rgb[..., 2] > 170)
    assert blue.sum() > 50                              # the outline is drawn
    # the far-away person is not in the picture
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 140) & (rgb[..., 2] < 140)
    assert red.sum() < 20                                # (JPEG edge noise only)


def test_pictures_are_cached(tmp_path, monkeypatch):
    video = synthetic_video(tmp_path / "v" / "cam1.mp4")
    req = PictureRequest(video=video, frame=11, box=Box(*person_at(11)))
    first = pictures.make_pictures([req], tmp_path / "cache")[req]

    def no_video(*a, **k):
        raise AssertionError("the cache should have been used")
    monkeypatch.setattr(pictures.cv2, "VideoCapture", no_video)
    assert pictures.make_pictures([req], tmp_path / "cache")[req] == first


def test_missing_video_or_frame_gives_no_picture(tmp_path):
    req = PictureRequest(video=tmp_path / "nope.mp4", frame=3, box=Box(1, 1, 5, 5))
    assert pictures.make_pictures([req], tmp_path / "cache") == {}
    video = synthetic_video(tmp_path / "v" / "cam1.mp4", seconds=1)
    late = PictureRequest(video=video, frame=999, box=Box(1, 1, 5, 5))
    assert pictures.make_pictures([late], tmp_path / "cache") == {}


def test_data_uri_and_asset_names(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(p, "JPEG")
    assert pictures.data_uri(p).startswith("data:image/jpeg;base64,")


def test_close_pictures_are_read_on_without_a_seek_and_match_a_seek(tmp_path, monkeypatch):
    video = synthetic_video(tmp_path / "v" / "cam1.mp4")
    reqs = [PictureRequest(video=video, frame=f, box=Box(*person_at(f))) for f in (5, 12, 30)]
    seeks = []
    real = pictures.cv2.VideoCapture

    class Counting:
        def __init__(self, *a):
            self._cap = real(*a)

        def set(self, prop, value):
            seeks.append(value)
            return self._cap.set(prop, value)

        def __getattr__(self, name):
            return getattr(self._cap, name)
    monkeypatch.setattr(pictures.cv2, "VideoCapture", Counting)
    read_on = pictures.make_pictures(reqs, tmp_path / "a")
    assert seeks == [4]                       # one seek, then read on (4 s at 10 fps)
    monkeypatch.setattr(pictures, "FORWARD_S", 0.0)
    seeks.clear()
    sought = pictures.make_pictures(reqs, tmp_path / "b")
    assert len(seeks) == 3
    for r in reqs:                            # the same frames either way
        a = np.asarray(Image.open(read_on[r])).astype(int)
        b = np.asarray(Image.open(sought[r])).astype(int)
        assert np.abs(a - b).mean() < 3
