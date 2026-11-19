from pathlib import Path

import cv2
import numpy as np
import pytest

from pfb_app.preprocess.frames import (
    Frame,
    VideoError,
    VideoInfo,
    chunk,
    extract_frames,
    format_time,
    probe_video,
    sample_points,
    validate_interval,
)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vid") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(200):  # 20 s at 10 fps
        img = np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path


def _frames(n):
    return [Frame(i, i * 5.0, format_time(i * 5.0), format_time(i * 5.0), Path(f"f{i}.jpg")) for i in range(n)]


def test_probe_video(video):
    info = probe_video(video)
    assert info.width == 320 and info.height == 240
    assert info.fps == pytest.approx(10)
    assert info.duration_sec == pytest.approx(20, abs=0.2)


def test_probe_video_unreadable(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError, match="動画"):
        probe_video(bad)
    with pytest.raises(VideoError, match="見つかりません"):
        probe_video(tmp_path / "missing.mp4")


def test_extract_a_time_within_half_a_frame_of_the_end(tmp_path):
    """1199 frames at 29.97 fps = 40.007 s: the sample at 40 s would be frame 1199, which does
    not exist (2026-11-15: 「40秒のフレームを読み取れません」 stopped a whole clip)."""
    path = tmp_path / "ends_on_a_multiple.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30000 / 1001, (64, 48))
    for i in range(1199):
        writer.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    info = probe_video(path)
    assert info.duration_sec > 40.0 > (1198.5 / info.fps)
    frames = extract_frames(path, tmp_path / "out", 0, 45)
    assert [f.video_sec for f in frames] == [0, 5, 10, 15, 20, 25, 30, 35]


def test_probe_video_refuses_a_codec_it_cannot_decode(video, monkeypatch):
    """AV1 10-bit: the header reads fine, no frame decodes (2026-11-15)."""
    real = cv2.VideoCapture

    class NoFrames:
        def __init__(self, *a):
            self._cap = real(*a)

        def read(self):
            return False, None

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", NoFrames)
    with pytest.raises(VideoError, match="形式は読み込めません.*MP4"):
        probe_video(video)


def test_extract_every_5s(video, tmp_path):
    frames = extract_frames(video, tmp_path / "out", 0, 20)
    assert [f.video_sec for f in frames] == [0, 5, 10, 15]
    for f in frames:
        img = cv2.imread(str(f.path))
        assert img.shape[:2] == (480, 854)
    assert [f.index for f in frames] == [0, 1, 2, 3]


def test_extract_reuses_existing(video, tmp_path):
    first = extract_frames(video, tmp_path / "out", 0, 20)
    mtimes = [f.path.stat().st_mtime_ns for f in first]
    second = extract_frames(video, tmp_path / "out", 0, 20)
    assert [f.path for f in second] == [f.path for f in first]
    assert [f.path.stat().st_mtime_ns for f in second] == mtimes


def test_task_elapsed_relative_to_start(video, tmp_path):
    frames = extract_frames(video, tmp_path / "out", 5, 20)
    assert frames[0].task_elapsed == "00:00"
    assert frames[0].video_time == "00:05"
    assert len(frames) == 3


def test_frame_dict_roundtrip():
    f = Frame(1, 5.0, "00:05", "00:00", Path("/x/a.jpg"))
    d = f.to_dict()
    assert d["path"] == "/x/a.jpg" and isinstance(d["path"], str)
    assert Frame.from_dict(d) == f


def test_format_time_hours():
    assert format_time(65) == "01:05"
    assert format_time(3599) == "59:59"
    assert format_time(3600) == "1:00:00"
    assert format_time(3725) == "1:02:05"


INFO = VideoInfo(duration_sec=20.0, fps=10.0, width=320, height=240)


def test_validate_interval_ok():
    assert validate_interval(INFO, None, None, 5) == (0.0, 20.0)
    assert validate_interval(INFO, 5, 20.3, 5) == (5.0, 20.0)  # clamped to the duration


def test_validate_interval_rejects_out_of_range():
    with pytest.raises(VideoError, match="動画の長さ"):
        validate_interval(INFO, 0, 30, 5)


def test_validate_interval_rejects_start_after_end():
    with pytest.raises(VideoError, match="開始"):
        validate_interval(INFO, 10, 10, 5)
    with pytest.raises(VideoError, match="開始"):
        validate_interval(INFO, 12, 5, 5)


def test_validate_interval_rejects_negative_start():
    with pytest.raises(VideoError, match="0秒以上"):
        validate_interval(INFO, -1, 10, 5)


def test_validate_interval_rejects_zero_frames():
    with pytest.raises(VideoError, match="フレーム"):
        validate_interval(INFO, 20.1, 20.4, 5)


def test_chunk_max_5():
    chunks = chunk(_frames(12))
    assert [len(c) for c in chunks] == [5, 5, 2]
    assert chunk([]) == []


def test_sample_points():
    fr = _frames(10)
    assert sample_points(fr, 3) == [fr[0], fr[5], fr[9]]
    assert sample_points(fr[:2], 3) == fr[:2]
    assert sample_points([], 3) == []
