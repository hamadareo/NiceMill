"""The playable proxy for 解説動画 and the dashboard clips (services.proxy): HDR (HLG/PQ) sources
get a tone-mapped BT.709 SDR proxy (as OpenCV's frames, which the analysis reads), the child
reports its progress, a failure is recorded (not tried again for the same file), and an HEVC
video of any size gets one. Synthetic videos only (made here with PyAV)."""
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from pfb_app.config import Settings
from pfb_app.services import proxy


def hlg_clip(path: Path, w=640, h=360, fps=10, seconds=2) -> Path:
    """A 10-bit H.264 clip tagged HLG / BT.2020, as an iPhone's HDR video is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p10le"
        cc = s.codec_context
        cc.color_trc, cc.color_primaries, cc.colorspace, cc.color_range = 18, 9, 9, 1
        for n in range(fps * seconds):
            img = cv2.resize(rng.integers(30, 220, (h // 8, w // 8, 3), dtype=np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p10le")
            frame.pts = n
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


def _grey(path: Path, k: int = 5, size=(1920, 1080)) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(float)


def test_an_hdr_source_gets_a_tone_mapped_sdr_proxy(settings, tmp_path):
    src = hlg_clip(tmp_path / "inbox" / "hlg.mp4")
    assert proxy._first_frame_facts(src)[1]                     # it is HDR
    dst, meta_path, part = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    seen = []
    meta = proxy.make(src, dst, progress=seen.append)
    assert meta["complete"] and meta["hdr"] and meta["tonemapped"] == "bt709"
    assert not part.exists() and dst.is_file()
    with av.open(str(dst)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.name, cc.pix_fmt, cc.width, cc.height) == ("h264", "yuv420p", 1920, 1080)
        assert (int(cc.color_trc), int(cc.color_primaries)) == (1, 1)
    # the proxy looks like OpenCV's (tone-mapped) frames of the source, not like plain swscale
    ref, got = _grey(src), _grey(dst)
    with av.open(str(src)) as c:
        frame = next(f for k, f in enumerate(c.decode(video=0)) if k == 5)
    plain = cv2.resize(frame.reformat(format="bgr24").to_ndarray(), (1920, 1080),
                       interpolation=cv2.INTER_AREA).astype(float)
    assert abs(got.mean() - ref.mean()) < 3 < abs(plain.mean() - ref.mean())
    assert np.abs(got - ref).mean() < np.abs(plain - ref).mean()
    assert seen and seen[-1] == pytest.approx(1.0) and seen == sorted(seen)
    assert proxy.ready(settings, src) == dst                    # for a player
    assert proxy.ready(settings, src, for_analysis=True) is None   # never for the analysis


def test_progress_state_and_a_recorded_failure(settings, tmp_path):
    src = hlg_clip(tmp_path / "inbox" / "hlg.mp4", seconds=1)
    assert proxy.state(settings, src) == ("none", None)
    dst, meta_path, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({**proxy._fingerprint(src), "complete": False,
                                     "pending": True}))
    proxy.progress_path(settings, src).write_text("0.42")
    assert proxy.state(settings, src) == ("waiting", 0.42)       # no child running it now
    assert proxy.main([str(src), str(dst)]) == 0
    assert proxy.state(settings, src) == ("ready", 1.0)
    assert not proxy.progress_path(settings, src).exists()
    broken = tmp_path / "inbox" / "broken.mp4"
    broken.write_bytes(b"not a video" * 100)
    bdst, bmeta, _ = proxy.paths(settings, broken)
    bmeta.write_text(json.dumps({**proxy._fingerprint(broken), "complete": False,
                                 "pending": True}))
    assert proxy.main([str(broken), str(bdst)]) == 1
    assert proxy.state(settings, broken) == ("failed", None)
    assert not proxy.start(settings, broken, 3840, 2160)          # not tried again


def test_an_old_hdr_skip_is_made_now_and_hevc_of_any_size_needs_one(settings, tmp_path,
                                                                    monkeypatch):
    src = hlg_clip(tmp_path / "inbox" / "hlg.mp4", seconds=1)
    dst, meta_path, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps({**proxy._fingerprint(src), "complete": False,
                                     "skipped": "hdr", "hdr": True}))   # rel4's record
    started = []

    class Popen:
        def __init__(self, cmd, **kw):
            started.append(cmd)
            self.pid = 999_999
    monkeypatch.setattr(proxy.subprocess, "Popen", Popen)
    assert proxy.start(settings, src, 3840, 2160)
    assert proxy.needs_playable(1920, 1080, "hevc") and not proxy.needs_playable(1920, 1080, "h264")
    assert proxy.needs_playable(3840, 2160, "h264")
