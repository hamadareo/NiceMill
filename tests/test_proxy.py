"""The 1080p proxy of a 4K video (services/proxy.py) and the stages that read it.

Synthetic videos only (noise and boxes, made here with OpenCV)."""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Job, JobStatus, Kind, Stage
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import prepare_video
from pfb_app.preprocess.frames import probe_video
from pfb_app.services import proxy
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


def clip(path: Path, w=2688, h=1512, fps=5, seconds=3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(3)
    for n in range(fps * seconds):
        img = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        img[200:600, 100 + 40 * n:300 + 40 * n] = (30, 160, 90)
        out.write(img)
    out.release()
    return path


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


def test_only_big_videos_get_one():
    assert proxy.needs_proxy(3840, 2160) and proxy.needs_proxy(2160, 3840)
    assert not proxy.needs_proxy(1920, 1080) and not proxy.needs_proxy(2560, 1440)


def test_sizes_keep_the_aspect_and_are_even():
    assert proxy._size(3840, 2160) == (1920, 1080)
    assert proxy._size(2160, 3840) == (1080, 1920)      # a portrait phone video
    w, h = proxy._size(4096, 2160)
    assert h == 1080 and w % 2 == 0


def test_make_keeps_time_and_frames(settings, tmp_path):
    src = clip(tmp_path / "inbox" / "big.mp4")
    dst, meta_path, part = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    meta = proxy.make(src, dst)
    small, big = probe_video(dst), probe_video(src)
    assert (small.width, small.height) == (1920, 1080)
    assert abs(small.duration_sec - big.duration_sec) < 0.3 and meta["frames"] == 15
    assert not part.exists() and json.loads(meta_path.read_text())["complete"]
    assert proxy.ready(settings, src) == dst


def test_ready_needs_the_same_source(settings, tmp_path):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    dst, meta_path, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    proxy.make(src, dst)
    assert proxy.ready(settings, src) == dst
    os.utime(src, ns=(time.time_ns(), time.time_ns() + 10**9))   # the source changed
    assert proxy.ready(settings, src) is None
    assert proxy.sweep(settings) == 1 and not dst.exists() and not meta_path.exists()


def test_an_hdr_proxy_is_for_the_preview_only(settings, tmp_path):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    dst, meta_path, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    proxy.make(src, dst)
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "hdr": True}))
    assert proxy.ready(settings, src) == dst
    assert proxy.ready(settings, src, for_analysis=True) is None


def test_start_runs_in_the_background_and_never_twice(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    monkeypatch.setattr(proxy, "_low_priority", lambda: [])   # no idle class on a busy host
    assert proxy.start(settings, src, 2688, 1512)
    assert not proxy.start(settings, src, 2688, 1512)        # already being made
    assert not proxy.start(settings, src, 1920, 1080)        # not big
    deadline = time.time() + 300
    while proxy.ready(settings, src) is None and time.time() < deadline:
        time.sleep(0.3)
    assert proxy.ready(settings, src) is not None
    assert not proxy.start(settings, src, 2688, 1512)        # done


def test_start_never_raises(settings, tmp_path):
    assert not proxy.start(settings, tmp_path / "missing.mp4", 3840, 2160)


def video_ctx(settings, src):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("活動", Kind.video, "塔を作る", "", "standard", False)
    sid = repo.create_submission(aid, "A", {"cameras": {"cam1": str(src)}, "start_sec": None,
                                            "end_sec": None, "auto_transcribe": False})
    job = Job(id=1, submission_id=sid, stage=Stage.prepare, camera="cam1", subject_id=None,
              model_key="cpu", status=JobStatus.running, attempts=0, heartbeat_at=None,
              error=None)
    return StageContext(settings, repo, None, lambda: False), job


def test_prepare_reads_the_proxy_when_it_is_ready(settings, tmp_path):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=12)
    ctx, job = video_ctx(settings, src)
    run, = prepare_video(ctx, job)
    assert run.status == "ok" and "proxy" not in run.output        # none yet: the original
    dst, _, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    proxy.make(src, dst)
    for f in run.output["frames"]:
        Path(f["path"]).unlink()
    run2, = prepare_video(ctx, job)
    assert run2.output.get("proxy") is True
    assert [f["video_sec"] for f in run2.output["frames"]] == [
        f["video_sec"] for f in run.output["frames"]]
    assert cv2.imread(run2.output["frames"][0]["path"]).shape == (480, 854, 3)


def test_hdr_is_found_from_pyavs_numbers_and_names():
    assert proxy.is_hdr(18, 9) and proxy.is_hdr(16, 1) and proxy.is_hdr(1, 9)   # HLG, PQ, 2020
    assert proxy.is_hdr("arib-std-b67", None)
    assert not proxy.is_hdr(1, 1) and not proxy.is_hdr(2, 2) and not proxy.is_hdr(None, None)


# --- review round 1 ----------------------------------------------------------------------------

def rotated_clip(path: Path, rotation: int, w=2688, h=1512, fps=5, seconds=1) -> Path:
    """A synthetic landscape-coded clip with a display rotation, as phones store portrait."""
    import av
    from fractions import Fraction
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as out:
        v = out.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        v.options = {"preset": "ultrafast"}
        if rotation:
            v.set_display_rotation(rotation)
        for n in range(fps * seconds):
            img = np.zeros((h, w, 3), np.uint8)
            img[: h // 2, : w // 3] = (200, 30, 30)       # a marked corner
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = n, Fraction(1, fps)
            for p in v.encode(frame):
                out.mux(p)
        for p in v.encode():
            out.mux(p)
    return path


def test_the_proxy_keeps_the_display_rotation(settings, tmp_path):
    src = rotated_clip(tmp_path / "inbox" / "portrait.mp4", 90)
    assert proxy.orientation(src) in (90, 270)
    dst, meta_path, _ = proxy.paths(settings, src)
    dst.parent.mkdir(parents=True)
    meta = proxy.make(src, dst)
    assert meta["rotation"] in (90, 270)
    assert proxy.orientation(dst) == proxy.orientation(src)
    assert proxy.ready(settings, src, for_analysis=True) == dst
    big, small = probe_video(src), probe_video(dst)     # OpenCV turns both the same way
    assert (big.width > big.height) == (small.width > small.height)


def test_a_proxy_turned_otherwise_is_refused(settings, tmp_path):
    src = rotated_clip(tmp_path / "inbox" / "portrait.mp4", 90)
    flat = rotated_clip(tmp_path / "flat.mp4", 0)
    dst, meta_path, _ = proxy.paths(settings, flat)
    dst.parent.mkdir(parents=True)
    proxy.make(flat, dst)                                 # a proxy without the rotation
    want, want_meta, _ = proxy.paths(settings, src)
    dst.rename(want)
    meta = json.loads(meta_path.read_text())
    want_meta.write_text(json.dumps({**meta, **proxy._fingerprint(src)}))
    assert proxy.ready(settings, src) is None


def test_a_work_takes_its_proxies_along(tmp_path):
    from pfb_app.services.deletion import delete_work
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    mine = clip(tmp_path / "inbox" / "mine.mp4", seconds=1)
    shared = clip(tmp_path / "inbox" / "shared.mp4", seconds=1)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    meta = lambda *ps: {"cameras": {f"cam{i}": str(p) for i, p in enumerate(ps, 1)}}  # noqa: E731
    aid = repo.create_assignment("A", Kind.video, "t", "", "standard", False)
    repo.create_submission(aid, "g", meta(mine, shared))
    other = repo.create_assignment("B", Kind.video, "t", "", "standard", False)
    repo.create_submission(other, "g", meta(shared))
    made = {}
    for src in (mine, shared):
        dst, _, _ = proxy.paths(settings, src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        proxy.make(src, dst)
        made[src] = dst
    delete_work(repo, settings, aid)
    assert not made[mine].exists()
    assert made[shared].exists()                          # another work still uses it


def test_one_transcode_at_a_time(settings, tmp_path, monkeypatch):
    import fcntl
    import threading
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    dst, _, _ = proxy.paths(settings, src)
    monkeypatch.setattr(proxy.subprocess, "Popen", lambda argv, **kw: _NoChild())
    assert proxy.start(settings, src, 2688, 1512)            # its pending sidecar; we run main
    with open(dst.parent / proxy.RUN_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        th = threading.Thread(target=proxy.main, args=([str(src), str(dst)],))
        th.start()
        time.sleep(1.0)
        assert not dst.exists()                           # waits for the running one
        fcntl.flock(lock, fcntl.LOCK_UN)
    th.join(60)
    assert dst.exists()


def test_children_run_at_idle_priority(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    seen = {}

    class P:
        pid = 4242

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        return P()
    monkeypatch.setattr(proxy.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(proxy.shutil, "which", lambda name: "/usr/bin/ionice")
    assert proxy.start(settings, src, 2688, 1512)
    assert seen["argv"][:5] == ["ionice", "-c3", "nice", "-n", "10"]


def test_sweeps_are_hourly_at_most(settings, monkeypatch):
    monkeypatch.setattr(proxy, "_last_sweep", 0.0)
    assert proxy.maybe_sweep(settings, now=10_000.0)
    assert not proxy.maybe_sweep(settings, now=10_000.0 + proxy.SWEEP_EVERY_SEC - 1)
    assert proxy.maybe_sweep(settings, now=10_000.0 + proxy.SWEEP_EVERY_SEC + 1)



# --- review round 2: a proxy removed while queued or being made --------------------------------

class _NoChild:
    pid = 4243


def test_a_queued_proxy_is_removed_with_its_work(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    monkeypatch.setattr(proxy.subprocess, "Popen", lambda argv, **kw: _NoChild())
    assert proxy.start(settings, src, 2688, 1512)            # queued: the child has not run
    dst, meta_path, part = proxy.paths(settings, src)
    assert json.loads(meta_path.read_text())["pending"] is True
    assert proxy.remove_for(settings, [src]) == 1             # the work is deleted
    assert not meta_path.exists() and not dst.with_suffix(".pid").exists()
    assert proxy.main([str(src), str(dst)]) == 0              # the child runs late: gives up
    assert not dst.exists() and not part.exists() and not meta_path.exists()


def test_a_proxy_removed_while_being_made_leaves_nothing(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    monkeypatch.setattr(proxy.subprocess, "Popen", lambda argv, **kw: _NoChild())
    proxy.start(settings, src, 2688, 1512)
    dst, meta_path, part = proxy.paths(settings, src)
    real = proxy.make

    def make_then_delete(s, d, **kw):                         # the work goes mid-transcode
        def wanted(path, calls=[0]):
            calls[0] += 1
            if calls[0] == 1:
                proxy.remove_for(settings, [src])
            return path.exists()
        monkeypatch.setattr(proxy, "_still_wanted", wanted)
        return real(s, d, **kw)
    monkeypatch.setattr(proxy, "make", make_then_delete)
    assert proxy.main([str(src), str(dst)]) == 0
    assert not dst.exists() and not part.exists() and not meta_path.exists()


# --- review rel2: deleting a work stops its transcode (the part file holds the disk) -----------

needs_proc = pytest.mark.skipif(not Path("/proc/self/cmdline").exists(),
                                reason="the pid check reads /proc (Linux)")


def _sleeper(*argv):
    """A process with this command line after `python -c <sleep>` (a stand-in transcode)."""
    import subprocess
    import sys
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *argv],
                            start_new_session=True)


def _queued(settings, src, monkeypatch):
    monkeypatch.setattr(proxy.subprocess, "Popen", lambda argv, **kw: _NoChild())
    assert proxy.start(settings, src, 2688, 1512)
    monkeypatch.undo()


@needs_proc
def test_deleting_a_work_stops_its_transcode(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    _queued(settings, src, monkeypatch)
    dst, meta_path, part = proxy.paths(settings, src)
    child = _sleeper("-m", "pfb_app.services.proxy", str(src), str(dst))
    try:
        dst.with_suffix(".pid").write_text(str(child.pid))
        part.write_bytes(b"half a proxy")
        assert proxy.remove_for(settings, [src]) == 1
        assert child.wait(timeout=10) != 0                     # stopped, not left to finish
        assert not part.exists() and not meta_path.exists()
    finally:
        child.kill()
        child.wait()


@needs_proc
def test_only_our_transcode_of_that_file_is_signalled(settings, tmp_path, monkeypatch):
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    _queued(settings, src, monkeypatch)
    dst, _, _ = proxy.paths(settings, src)
    other = _sleeper("-m", "pfb_app.services.proxy", str(src), str(dst.with_name("other.mp4")))
    stranger = _sleeper("unrelated")
    try:
        for pid in (other.pid, stranger.pid):                 # a pid file naming someone else
            _queued(settings, src, monkeypatch)
            dst.with_suffix(".pid").write_text(str(pid))
            assert proxy.remove_for(settings, [src]) == 1
        time.sleep(0.5)
        assert other.poll() is None and stranger.poll() is None
    finally:
        for p in (other, stranger):
            p.kill()
            p.wait()


def test_no_proxy_is_started_when_the_disk_is_low(settings, tmp_path, monkeypatch, caplog):
    """The proxy is only an optimisation: on a low disk it is skipped (the original is used)."""
    from pfb_app.ops import disk
    src = clip(tmp_path / "inbox" / "big.mp4", seconds=1)
    asked, started = [], []

    def low(s, incoming_bytes=0, usage=None, **kw):
        asked.append(incoming_bytes)
        raise disk.DiskLow("空きが足りません")
    monkeypatch.setattr(disk, "ensure_room", low)
    monkeypatch.setattr(proxy.subprocess, "Popen", lambda argv, **kw: started.append(argv))
    with caplog.at_level("INFO"):
        assert not proxy.start(settings, src, 2688, 1512)
    assert not started and asked and 0 < asked[0] <= src.stat().st_size
    dst, meta_path, _ = proxy.paths(settings, src)
    assert not meta_path.exists() and not dst.with_suffix(".pid").exists()
    assert "空き" in caplog.text
