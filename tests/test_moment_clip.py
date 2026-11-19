"""The short excerpt a dashboard jump plays (services.moment_clip): which file it is cut from
and the cut itself (PyAV stream copy). Generated videos only."""
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from pfb_app.services import moment_clip, proxy
from test_wizard_ui_basic import env  # noqa: F401 (env)


def make_video(path: Path, seconds: int = 30, fps: int = 10, size=(320, 240)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(seconds * fps):
        w.write(np.full((size[1], size[0], 3), (i * 3) % 256, dtype=np.uint8))
    w.release()
    return path


def seconds_of(path: Path) -> float:
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        last = None
        n = 0
        for frame in c.decode(v):
            last = frame.time
            n += 1
        assert n > 0
        return float(last)


def test_a_small_video_plays_from_the_original(env, tmp_path):
    settings, _ = env
    video = make_video(tmp_path / "a.mp4")
    assert moment_clip.source(settings, video, (320, 240)) == (video, None)


def test_an_unknown_size_is_probed(env, tmp_path, monkeypatch):
    settings, _ = env
    video = make_video(tmp_path / "a.mp4")
    assert moment_clip.source(settings, video, None) == (video, None)
    found = moment_clip.probe(video)
    assert (found.width, found.height) == (320, 240) and found.duration_s > 25
    monkeypatch.setattr(moment_clip, "probe", lambda p: moment_clip.Probe(3840, 2160, "h264", 30))
    assert moment_clip.source(settings, video, None) == (None, moment_clip.NOT_READY)


def test_an_unreadable_source_has_no_player(env, tmp_path):
    settings, _ = env
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert moment_clip.source(settings, bad, None) == (None, moment_clip.CANNOT_READ)


def test_an_hevc_source_plays_only_from_its_proxy(env, tmp_path, monkeypatch):
    settings, _ = env
    video = make_video(tmp_path / "a.mp4")
    monkeypatch.setattr(moment_clip, "probe", lambda p: moment_clip.Probe(1920, 1080, "hevc", 30))
    assert moment_clip.source(settings, video, (1920, 1080)) == (None, moment_clip.NOT_READY)
    small = make_video(tmp_path / "proxy.mp4")
    monkeypatch.setattr(proxy, "ready", lambda s, src, **k: small)
    assert moment_clip.source(settings, video, (1920, 1080)) == (small, None)


def test_no_video_no_player(env, tmp_path):
    settings, _ = env
    assert moment_clip.source(settings, None, None) == (None, moment_clip.NO_VIDEO)
    assert moment_clip.source(settings, tmp_path / "gone.mp4", (320, 240)) == (
        None, moment_clip.NO_VIDEO)


def test_a_big_video_plays_only_from_its_ready_proxy(env, tmp_path, monkeypatch):
    settings, _ = env
    video = make_video(tmp_path / "big.mp4")
    assert moment_clip.source(settings, video, (3840, 2160)) == (None, moment_clip.NOT_READY)
    small = make_video(tmp_path / "proxy.mp4")
    monkeypatch.setattr(proxy, "ready", lambda s, src, **k: small if Path(src) == video else None)
    assert moment_clip.source(settings, video, (3840, 2160)) == (small, None)


def test_the_clip_starts_before_the_moment_and_says_where_it_is(env, tmp_path):
    settings, _ = env
    video = make_video(tmp_path / "a.mp4", seconds=40)
    out = tmp_path / "clips"
    clip, offset = moment_clip.clip(out, "cam1", video, 20.0)
    assert clip.is_file() and clip.parent == out
    # the cut starts on a key frame at or before 5 s before the moment
    assert moment_clip.BEFORE_S <= offset <= moment_clip.BEFORE_S + 5
    length = seconds_of(clip)
    assert moment_clip.BEFORE_S + moment_clip.AFTER_S - 1 <= length + 0.2
    assert length <= offset + moment_clip.AFTER_S + 1.5
    again, offset2 = moment_clip.clip(out, "cam1", video, 20.0)      # cached
    assert again == clip and offset2 == pytest.approx(offset)


def test_a_moment_near_the_start_is_cut_from_zero(env, tmp_path):
    video = make_video(tmp_path / "a.mp4", seconds=20)
    clip, offset = moment_clip.clip(tmp_path / "clips", "cam1", video, 2.0)
    assert offset == pytest.approx(2.0, abs=0.15)


def test_a_changed_source_is_cut_again(env, tmp_path):
    video = make_video(tmp_path / "a.mp4", seconds=20)
    first, _ = moment_clip.clip(tmp_path / "clips", "cam1", video, 10.0)
    make_video(video, seconds=25)
    second, _ = moment_clip.clip(tmp_path / "clips", "cam1", video, 10.0)
    assert first != second and second.is_file()


def leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if not p.name.endswith(".lock"))


def test_a_failed_cut_leaves_nothing_behind(env, tmp_path, monkeypatch):
    video = make_video(tmp_path / "a.mp4", seconds=20)
    folder = tmp_path / "clips"

    def broken(src, dst, start_s, end_s):
        Path(dst).write_bytes(b"half a file")
        raise RuntimeError("stopped half way")
    monkeypatch.setattr(moment_clip, "_cut", broken)
    with pytest.raises(RuntimeError):
        moment_clip.clip(folder, "cam1", video, 10.0)
    assert leftovers(folder) == []


def test_a_broken_clip_is_never_cached(env, tmp_path, monkeypatch):
    video = make_video(tmp_path / "a.mp4", seconds=20)
    folder = tmp_path / "clips"

    def garbage(src, dst, start_s, end_s):
        Path(dst).write_bytes(b"\x00" * 64)
        return 0.0
    monkeypatch.setattr(moment_clip, "_cut", garbage)
    with pytest.raises(moment_clip.ClipFailed):
        moment_clip.clip(folder, "cam1", video, 10.0)
    assert leftovers(folder) == []
    monkeypatch.undo()
    clip, _ = moment_clip.clip(folder, "cam1", video, 10.0)     # a later try cuts it properly
    assert seconds_of(clip) > 5


def test_two_presses_at_once_cut_once(env, tmp_path, monkeypatch):
    import threading
    video = make_video(tmp_path / "a.mp4", seconds=30)
    folder = tmp_path / "clips"
    calls = []
    real = moment_clip._cut

    def counted(*a):
        calls.append(1)
        return real(*a)
    monkeypatch.setattr(moment_clip, "_cut", counted)
    out = []
    threads = [threading.Thread(target=lambda: out.append(moment_clip.clip(folder, "cam1", video,
                                                                           15.0)))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len({c for c, _ in out}) == 1 and len(out) == 3


def test_the_folder_keeps_under_its_cap_oldest_first(env, tmp_path, monkeypatch):
    import os
    import time
    video = make_video(tmp_path / "a.mp4", seconds=60)
    folder = tmp_path / "clips"
    first, _ = moment_clip.clip(folder, "cam1", video, 10.0)
    os.utime(first, (time.time() - 100, time.time() - 100))
    second, _ = moment_clip.clip(folder, "cam1", video, 30.0)
    monkeypatch.setattr(moment_clip, "MAX_FOLDER_BYTES", second.stat().st_size + 10)
    third, _ = moment_clip.clip(folder, "cam1", video, 45.0)
    assert not first.exists() and not first.with_suffix(".json").exists()
    assert not first.with_suffix(".lock").exists()     # its lock goes with it
    assert third.exists() and third.with_suffix(".lock").exists()


def test_low_disk_refuses_a_new_clip(env, tmp_path, monkeypatch):
    from pfb_app.ops import disk
    settings, _ = env
    video = make_video(tmp_path / "a.mp4", seconds=20)

    def low(*a, **k):
        raise disk.DiskLow("少ない")
    monkeypatch.setattr(disk, "ensure_room", low)
    with pytest.raises(moment_clip.ClipRefused) as e:
        moment_clip.clip(tmp_path / "clips", "cam1", video, 10.0, settings=settings)
    assert str(e.value) == moment_clip.LOW_DISK
    assert not (tmp_path / "clips").exists() or leftovers(tmp_path / "clips") == []


def test_a_clip_too_big_to_play_is_refused(env, tmp_path, monkeypatch):
    video = make_video(tmp_path / "a.mp4", seconds=20)
    monkeypatch.setattr(moment_clip, "MAX_CLIP_BYTES", 10)
    with pytest.raises(moment_clip.ClipRefused) as e:
        moment_clip.clip(tmp_path / "clips", "cam1", video, 10.0)
    assert str(e.value) == moment_clip.TOO_BIG
    assert leftovers(tmp_path / "clips") == []


def test_clips_live_in_the_groups_handout_folder(env):
    settings, _ = env
    d = moment_clip.clips_dir(settings, 3, 7)
    assert d.parts[-3:] == ("7", "handout", "clips")
    assert "handout" in d.parts   # a heavy folder: removed with the videos (retention)
