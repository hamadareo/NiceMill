"""LiveController (the engine's long-running `live` demo) against tests/fakes/fake_engine.py."""
import json
import os
import time
from pathlib import Path

import pytest
import yaml

from fakes.engine_env import alive, install_fake_engine
from pfb_app.config import Settings
from pfb_app.engine import Pfb3x3Engine
from pfb_app.engine.live import LiveBusy, LiveController, LiveSession

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                                reason="process checks need /proc (Linux)")


@pytest.fixture
def live(tmp_path, monkeypatch):
    fe = install_fake_engine(tmp_path, monkeypatch)
    video = tmp_path / "IMG_0105.MOV"
    video.write_bytes(b"not really a video")
    settings = Settings(data_dir=tmp_path / "data", engine_dir=fe.engine_dir,
                        engine_python=fe.engine_python, tid_python=tmp_path / "tid_python")
    ctl = LiveController(settings, Pfb3x3Engine(settings, tick_sec=0.05), stop_wait_sec=5)
    yield ctl, fe, video, settings
    ctl.stop()   # never leave a fake live run behind


def _wait(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _records(stream: Path) -> list[dict]:
    if not stream.exists():
        return []
    lines = stream.read_text(encoding="utf-8").split("\n")[:-1]   # whole lines only
    return [json.loads(line) for line in lines]


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_live_start_status_stop(live):
    ctl, fe, video, settings = live
    s = ctl.start(video, tid_video="IMG_0105")
    assert isinstance(s, LiveSession) and s.video_id == "linecall_demo"
    root = settings.data_dir / "live" / "pfb3x3"
    assert s.stream == root / "tracks" / "linecall_demo" / "live" / s.session / "stream.jsonl"
    assert ctl.status() == s
    # another controller (e.g. after a page rerun) sees the same run from the pidfile
    assert LiveController(settings, Pfb3x3Engine(settings)).status() == s
    pidfile = json.loads((settings.data_dir / "live" / "live.json").read_text("utf-8"))
    assert {"pid", "start_time", "session", "stream", "video_id", "boot_id"} <= set(pidfile)
    assert pidfile["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert pidfile["pid"] == s.pid and pidfile["stream"] == str(s.stream)
    assert _wait(lambda: len(_records(s.stream)) >= 3)
    n1 = len(_records(s.stream))
    assert _wait(lambda: len(_records(s.stream)) > n1)   # the stream grows
    argv = fe.calls()[-1]["argv"]
    assert argv[0] == "live"
    assert _opt(argv, "--video-id") == "linecall_demo"
    assert _opt(argv, "--registry") == str(root / "videos")
    assert _opt(argv, "--tracks") == str(root / "tracks")
    assert _opt(argv, "--session") == s.session
    assert _opt(argv, "--device") == "cuda"
    assert _opt(argv, "--tracker-python") == str(settings.engine_python)
    assert _opt(argv, "--tid-python") == str(settings.tid_python)
    assert _opt(argv, "--weights") == str(settings.data_dir / "models" / "yolo11l.pt")
    assert _opt(argv, "--tid-video") == "IMG_0105"
    entry = yaml.safe_load((root / "videos" / "linecall_demo.yaml").read_text("utf-8"))
    assert entry["path"] == str(video)
    assert entry["domain"] == "sports" and entry["sport"] == "basketball"
    assert ctl.last_outcome() is None   # still running
    assert ctl.stop() is True
    assert ctl.status() is None and not alive(s.pid)
    assert not (settings.data_dir / "live" / "live.json").exists()
    recs = _records(s.stream)
    assert recs[0]["type"] == "start" and recs[-1]["type"] == "end"
    assert recs[-1]["reason"] == "stop" and recs[-1]["status"] == "判定材料"
    assert ctl.last_outcome() == "stopped"   # a user stop, whatever the exit code
    assert LiveController(settings, Pfb3x3Engine(settings)).last_outcome() == "stopped"
    assert (root / "logs" / "live.out").is_file() and (root / "logs" / "live.err").is_file()


def test_live_stream_records(live, monkeypatch):
    ctl, _fe, video, settings = live
    monkeypatch.setenv("FAKE_ENGINE_LIVE_RECORDS", "40")
    s = ctl.start(video)
    assert _wait(lambda: ctl.status() is None, timeout=20)
    recs = _records(s.stream)
    assert recs[0]["type"] == "start" and recs[0]["alerts"] == "event"
    assert len(recs) == 42 and recs[-1]["type"] == "end" and recs[-1]["reason"] == "eof"
    kinds = {r["type"] for r in recs}
    assert {"positions", "alert", "alert_end", "stats", "dwell", "dwell_end"} <= kinds
    pos = next(r for r in recs if r["type"] == "positions")
    assert {"id", "x", "y", "in_play"} <= set(pos["players"][0])
    alert = next(r for r in recs if r["type"] == "alert")
    assert {"alert_id", "player", "line", "event", "side", "distance_cm",
            "needs_check"} <= set(alert)
    stats = next(r for r in recs if r["type"] == "stats")
    assert {"p50_ms", "p90_ms", "p95_ms", "max_ms", "n"} <= set(stats["latency"])
    assert ctl.last_outcome() == "ok"
    # the next start gets a session of its own even within the same second
    s2 = ctl.start(video)
    assert s2.session != s.session
    assert ctl.stop() is True


def test_live_busy(live):
    ctl, _fe, video, settings = live
    s = ctl.start(video)
    with pytest.raises(LiveBusy):
        ctl.start(video)
    with pytest.raises(LiveBusy):
        LiveController(settings, Pfb3x3Engine(settings)).start(video)
    assert ctl.status() == s
    assert ctl.stop() is True
    assert ctl.stop() is False   # nothing left to stop


def test_live_stale_pid_not_signalled(live, monkeypatch):
    ctl, _fe, _video, settings = live
    pidfile = settings.data_dir / "live" / "live.json"
    pidfile.parent.mkdir(parents=True)
    pidfile.write_text(json.dumps({   # our own pid, but not the start time recorded
        "pid": os.getpid(), "start_time": "1", "session": "20261022T000000Z",
        "stream": str(settings.data_dir / "x.jsonl"), "video_id": "linecall_demo",
        "started_at": "2026-10-22T00:00:00Z"}), encoding="utf-8")
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append(("killpg", pid, sig)))
    real_kill = os.kill
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append(("kill", pid, sig))
                        if sig else real_kill(pid, sig))
    assert ctl.status() is None
    assert ctl.stop() is False
    assert sent == []


def test_live_needs_gpu_outcome(live, monkeypatch):
    ctl, _fe, video, settings = live
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    monkeypatch.setenv("FAKE_ENGINE_ERROR",
                       "--device cuda: GPU 0 has 1000 MiB free < 38912 (32768 reserve + 6144)")
    s = ctl.start(video)
    assert _wait(lambda: ctl.status() is None and ctl.last_outcome() is not None)
    assert ctl.last_outcome() == "needs_gpu"
    assert not s.stream.exists()   # exit 3: nothing written
    assert LiveController(settings, Pfb3x3Engine(settings)).last_outcome() == "needs_gpu"
    # without the reaper's exit record (another app process): the engine's JSON line decides
    (settings.data_dir / "live" / "live.exit.json").unlink()
    assert ctl.last_outcome() == "needs_gpu"


def test_live_error_outcome(live, monkeypatch):
    ctl, _fe, video, _settings = live
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    ctl.start(video)
    assert _wait(lambda: ctl.status() is None and ctl.last_outcome() is not None)
    assert ctl.last_outcome() == "error"


def test_live_missing_video(live, tmp_path):
    ctl, _fe, _video, _settings = live
    with pytest.raises(FileNotFoundError):
        ctl.start(tmp_path / "missing.mov")
    assert ctl.status() is None


def test_live_status_during_exec(monkeypatch):
    """Right after Popen (or while a wrapper execs the engine) /proc/<pid>/cmdline is empty:
    the recorded pid + start time still count as ours; a wrong start time never does."""
    from pfb_app.engine import live as live_mod
    from pfb_app.engine.pfb3x3 import _start_time
    me = os.getpid()
    monkeypatch.setattr(live_mod, "_cmdline", lambda pid: [""])
    monkeypatch.setattr(live_mod, "_proc_age_s", lambda pid: 1.0)
    assert live_mod._ours(me, _start_time(me)) is True
    assert live_mod._ours(me, "1") is False
    assert live_mod._ours(me, None) is False
    # the grace is for a just-started process only (and never a kernel thread)
    monkeypatch.setattr(live_mod, "_proc_age_s", lambda pid: live_mod.EXEC_GRACE_SEC + 1)
    assert live_mod._ours(me, _start_time(me)) is False
    monkeypatch.setattr(live_mod, "_proc_age_s", lambda pid: 1.0)
    monkeypatch.setattr(live_mod, "_is_kthread", lambda pid: True)
    assert live_mod._ours(me, _start_time(me)) is False
    monkeypatch.setattr(live_mod, "_is_kthread", lambda pid: False)
    monkeypatch.setattr(live_mod, "_cmdline", lambda pid: ["python", "-m", "pytest"])
    assert live_mod._ours(me, _start_time(me)) is False


def test_live_pidfile_of_another_boot_is_stale(live):
    ctl, _fe, video, settings = live
    s = ctl.start(video)
    pidfile = settings.data_dir / "live" / "live.json"
    rec = json.loads(pidfile.read_text("utf-8"))
    pidfile.write_text(json.dumps({**rec, "boot_id": "another-boot"}), encoding="utf-8")
    assert ctl.status() is None
    pidfile.write_text(json.dumps(rec), encoding="utf-8")
    assert ctl.status() == s
    assert ctl.stop() is True


def test_live_concurrent_start_one_wins(live):
    import threading
    ctl, _fe, video, settings = live
    barrier = threading.Barrier(2)
    got: list = []

    def go():
        c = LiveController(settings, Pfb3x3Engine(settings))
        barrier.wait()
        try:
            got.append(c.start(video))
        except LiveBusy as e:
            got.append(e)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    sessions = [g for g in got if isinstance(g, LiveSession)]
    assert len(sessions) == 1 and sum(isinstance(g, LiveBusy) for g in got) == 1
    assert ctl.status() == sessions[0]
    assert ctl.stop() is True
    assert not alive(sessions[0].pid)


def test_live_stop_escalates_and_kills_the_orphaned_child(live, tmp_path, monkeypatch):
    """The engine ignores SIGTERM and its live_run.py child (own group, no parent-death
    signal here) would outlive the engine's SIGKILL: stop() kills it too, after checking its
    cmdline (live_run.py --parent-pid <engine>) and start time. A look-alike of another
    engine is left alone."""
    import subprocess
    import sys
    _ctl, _fe, video, settings = live
    pids = tmp_path / "stubborn.pids"
    monkeypatch.setenv("FAKE_ENGINE_STUBBORN", str(pids))
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              "/x/scripts/live_run.py", "--parent-pid", "1"])
    try:
        ctl = LiveController(settings, Pfb3x3Engine(settings), stop_wait_sec=1)
        s = ctl.start(video)
        assert _wait(lambda: pids.exists() and len(pids.read_text().split()) == 2)
        engine_pid, child_pid = map(int, pids.read_text().split())
        assert engine_pid == s.pid and alive(child_pid)
        assert ctl.stop() is True
        assert _wait(lambda: not alive(engine_pid) and not alive(child_pid), timeout=5)
        assert alive(decoy.pid)
        assert ctl.status() is None and ctl.last_outcome() == "stopped"
    finally:
        decoy.kill()
        decoy.wait()


def _stubborn_run(settings, video, pids, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_STUBBORN", str(pids))
    ctl = LiveController(settings, Pfb3x3Engine(settings), stop_wait_sec=1)
    s = ctl.start(video)
    assert _wait(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    engine_pid, child_pid = map(int, pids.read_text().split())
    assert engine_pid == s.pid
    return ctl, engine_pid, child_pid


def test_live_orphan_kill_needs_the_engine_as_parent(live, tmp_path, monkeypatch):
    """A process with the very cmdline of the engine's child (live_run.py --parent-pid
    <engine pid>, started after the engine) but another parent is not the engine's child."""
    import subprocess
    import sys
    _ctl, _fe, video, settings = live
    ctl, engine_pid, child_pid = _stubborn_run(settings, video, tmp_path / "p", monkeypatch)
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              "/x/scripts/live_run.py", "--parent-pid", str(engine_pid)])
    try:
        assert _wait(lambda: "live_run.py" in " ".join(
            Path(f"/proc/{decoy.pid}/cmdline").read_text().split("\0")))
        assert ctl.stop() is True
        assert _wait(lambda: not alive(engine_pid) and not alive(child_pid), timeout=5)
        assert alive(decoy.pid)
    finally:
        decoy.kill()
        decoy.wait()


def test_live_orphan_kill_permission_error_never_escapes(live, tmp_path, monkeypatch,
                                                         caplog):
    """A child we may not signal (another user's: PermissionError) is logged; stop() still
    returns True and removes the pidfile."""
    import signal
    from pfb_app.engine import live as live_mod
    _ctl, _fe, video, settings = live
    ctl, engine_pid, child_pid = _stubborn_run(settings, video, tmp_path / "p", monkeypatch)

    def denied(*_a, **_k):
        raise PermissionError(1, "Operation not permitted")

    real_kill = os.kill
    monkeypatch.setattr(os, "kill", lambda pid, sig: denied() if pid == child_pid
                        else real_kill(pid, sig))
    monkeypatch.setattr(os, "getpgid", denied)
    if hasattr(os, "pidfd_open"):
        monkeypatch.setattr(os, "pidfd_open", denied)
    if hasattr(signal, "pidfd_send_signal"):
        monkeypatch.setattr(signal, "pidfd_send_signal", denied)
    try:
        with caplog.at_level("WARNING", logger=live_mod.__name__):
            assert ctl.stop() is True
        assert not (settings.data_dir / "live" / "live.json").exists()
        assert str(child_pid) in caplog.text
        monkeypatch.undo()   # os.kill again (alive() uses it)
        assert _wait(lambda: not alive(engine_pid), timeout=5)
        assert alive(child_pid)   # it could not be signalled
    finally:
        monkeypatch.undo()
        real_kill(child_pid, signal.SIGKILL)


def test_stop_in_background_returns_at_once(live, tmp_path, monkeypatch):
    """The page's 「停止」: stop() can take stop_wait_sec (20 s) and more, so it runs in a
    thread; every controller of the same data dir sees the stop under way."""
    _ctl, _fe, video, settings = live
    pids = tmp_path / "stubborn.pids"
    monkeypatch.setenv("FAKE_ENGINE_STUBBORN", str(pids))
    ctl = LiveController(settings, Pfb3x3Engine(settings), stop_wait_sec=1.5)
    other = LiveController(settings, Pfb3x3Engine(settings))
    assert ctl.stopping() is False
    s = ctl.start(video)
    assert _wait(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    engine_pid, child_pid = map(int, pids.read_text().split())
    t0 = time.monotonic()
    thread = ctl.stop_in_background()
    assert time.monotonic() - t0 < 0.5
    assert ctl.stopping() and other.stopping()
    assert other.stop_in_background() is thread   # one stopper at a time
    assert ctl.status() == s   # SIGTERM is ignored: the run is still there while it stops
    thread.join(timeout=15)
    assert not thread.is_alive() and not ctl.stopping() and not other.stopping()
    assert ctl.status() is None and ctl.last_outcome() == "stopped"
    assert _wait(lambda: not alive(engine_pid) and not alive(child_pid), timeout=5)


def test_stop_in_background_with_nothing_running(live):
    ctl, _fe, _video, _settings = live
    thread = ctl.stop_in_background()
    thread.join(timeout=5)
    assert not ctl.stopping() and ctl.status() is None and ctl.last_outcome() is None


def test_stop_of_an_ended_session_leaves_the_next_run_alone(live):
    """The stale-stop race: 「停止」 was drawn for run A, A ended and run B started before
    the press arrived. stop(A) (also in the background) must not stop B."""
    ctl, _fe, video, _settings = live
    a = ctl.start(video)
    assert ctl.stop(expected_session=a.session) is True
    b = ctl.start(video)
    assert b.session != a.session
    assert ctl.stop(expected_session=a.session) is False
    ctl.stop_in_background(a.session).join(timeout=10)
    assert ctl.status() == b and alive(b.pid)
    assert ctl.stop(expected_session=b.session) is True and ctl.status() is None
