"""Pfb3x3Engine against tests/fakes/fake_engine.py (a stand-in `python -m pfb3x3.engine`)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from pfb_app.config import Settings
from pfb_app.engine import (
    LOG_PREFIX, BehaviorResult, EngineError, EngineNeedsGpu, EngineYield, Pfb3x3Engine,
    ServerHandle, SheetResult, TrackResult, engine_model_for,
)
from pfb_app.engine.pfb3x3 import (CONDITIONS, PFB_APP_ROOT, WINDOW_PLAN, count_rows,
                                  default_whisper_python, planned_windows)

FAKE = Path(__file__).resolve().parent / "fakes" / "fake_engine.py"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie is dead for our purposes
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(")")[-1].split()[0] == "Z")


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine_dir = tmp_path / "engines" / "pfb3x3"
    (engine_dir / "scripts").mkdir(parents=True)
    wrapper = tmp_path / "fake_python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    for name, cmd in (("serve.sh", "serve"), ("stop_server.sh", "stop")):
        (engine_dir / "scripts" / name).write_text(
            f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FAKE}" {cmd} "$@"\n', encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_ENGINE_LOG", str(log))
    for k in ("FAKE_ENGINE_EXIT", "FAKE_ENGINE_SLEEP", "FAKE_ENGINE_OUTSIDE", "FAKE_ENGINE_STDOUT",
              "FAKE_ENGINE_STUBBORN", "FAKE_ENGINE_SERVE_SLEEP", "FAKE_ENGINE_STOP_EXIT"):
        monkeypatch.delenv(k, raising=False)
    settings = Settings(data_dir=tmp_path / "data", engine_dir=engine_dir, engine_python=wrapper)
    sub_dir = tmp_path / "data" / "assignments" / "1" / "submissions" / "2"
    sub_dir.mkdir(parents=True)

    def calls() -> list[dict]:
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    return settings, Pfb3x3Engine(settings, tick_sec=0.05), sub_dir, calls


def _track(engine, sub_dir, **kw):
    return engine.track(sub_dir, "s2_cam1", Path("/videos/v.mp4"), 88, 496, 4, "紙タワー", **kw)


def test_track_writes_registry_and_parses(env):
    settings, engine, sub_dir, calls = env
    res = _track(engine, sub_dir, transcript=True)
    root = sub_dir / "pfb3x3"
    assert isinstance(res, TrackResult)
    assert res.people == 3 and res.labels == ["P1", "P2", "P3"]
    assert res.boxes == root / "tracks" / "s2_cam1" / "boxes.txt"
    assert res.track_stats.is_file()
    entry = yaml.safe_load((root / "videos" / "s2_cam1.yaml").read_text(encoding="utf-8"))
    assert entry == {"video_id": "s2_cam1", "path": "/videos/v.mp4", "domain": "generic",
                     "activity": "紙タワー", "start_s": 88.0, "end_s": 496.0, "max_people": 4,
                     "transcript": True, "camera_notes": ""}
    argv = calls()[-1]["argv"]
    assert argv[0] == "track"
    assert argv[argv.index("--registry") + 1] == str(root / "videos")
    assert argv[argv.index("--tracks") + 1] == str(root / "tracks")
    assert argv[argv.index("--device") + 1] == "auto"
    # the tracker runs in the engine venv unless tracker_python is set
    assert argv[argv.index("--tracker-python") + 1] == str(settings.engine_python)


def test_track_reports_the_tracker(env, monkeypatch):
    """Engine 58f8341: `tracker` (engine | trackid3x3) and the fallback's reason; an engine
    that reports neither (before 58f8341) gives None, None."""
    _, engine, sub_dir, _ = env
    res = _track(engine, sub_dir)
    assert (res.tracker, res.tracker_fallback) == ("engine", "not a TID dataset video")
    monkeypatch.setenv("FAKE_ENGINE_TRACKER", "trackid3x3")
    res = _track(engine, sub_dir)
    assert (res.tracker, res.tracker_fallback) == ("trackid3x3", None)
    monkeypatch.setenv("FAKE_ENGINE_TRACKER", "none")
    res = _track(engine, sub_dir)
    assert (res.tracker, res.tracker_fallback) == (None, None)
    assert res.people == 3 and res.labels == ["P1", "P2", "P3"]


def test_track_without_end_or_people(env):
    _, engine, sub_dir, _ = env
    engine.track(sub_dir, "s2_cam2", Path("/v.mp4"), 0, None, None, None)
    entry = yaml.safe_load((sub_dir / "pfb3x3" / "videos" / "s2_cam2.yaml").read_text("utf-8"))
    assert entry["end_s"] is None and entry["max_people"] is None and entry["activity"] is None


def test_rejects_unsafe_video_id(env):
    _, engine, sub_dir, _ = env
    with pytest.raises(ValueError):
        engine.track(sub_dir, "../x", Path("/v.mp4"), 0, None, None, None)


def test_env_has_no_foreign_paths(env, monkeypatch):
    settings, engine, sub_dir, calls = env
    abc = str(Path.home() / "extern" / "pfb3x3")
    monkeypatch.setenv("PFB_TRACKS", abc + "/runs/tracks")   # inherited values must not leak
    monkeypatch.setenv("PFB_VIDEOS", abc + "/configs/videos")
    monkeypatch.setenv("PYTHONPATH", abc + "/src")
    monkeypatch.setenv("VIRTUAL_ENV", "/home/x/pfb_app/.venv")
    _track(engine, sub_dir)
    child_env = calls()[-1]["env"]
    root = sub_dir / "pfb3x3"
    pfb = {k: v for k, v in child_env.items() if k.startswith("PFB_")}
    assert pfb == {
        "PFB_VIDEOS": str(root / "videos"),
        "PFB_TRACKS": str(root / "tracks"),
        "PFB_DET_CACHE": str(root / "cache"),
        "PFB_RUNTIME_DIR": str(settings.data_dir / "pfb3x3_runtime"),
        "PFB_YOLO_WEIGHTS": str(settings.data_dir / "models" / "yolo11l.pt"),
        "PFB_GPU_RESERVE_MIB": "32768",   # settings.engine_gpu_reserve_gb (default 32)
    }
    assert "PYTHONPATH" not in child_env and "VIRTUAL_ENV" not in child_env
    assert not any("extern" in v for v in child_env.values())
    assert child_env["HF_HUB_OFFLINE"] == "1"
    assert child_env["PYTHONDONTWRITEBYTECODE"] == "1"  # no __pycache__ inside engine_dir


def test_sheet(env):
    _, engine, sub_dir, calls = env
    res = engine.sheet(sub_dir, "s2_cam1")
    assert isinstance(res, SheetResult)
    assert res.sheet == sub_dir / "pfb3x3" / "sheets" / "s2_cam1" / "sheet.jpg"
    assert res.geometry["P1"] == [10, 10, 50, 100]
    # by default on the auto-start frame, the frame rule_names(auto_start=True) uses too
    assert "--auto-start" in calls()[-1]["argv"]
    assert (res.frame, res.frame_rule, res.frame_offset_s) == (152, "auto_start", 6.4)
    res = engine.sheet(sub_dir, "s2_cam1", auto_start=False)
    assert "--auto-start" not in calls()[-1]["argv"]
    assert (res.frame, res.frame_rule) == (120, "start")


def _engine_track_in_child(settings, sub_dir, tmp_path):
    """Run engine.track in a separate python process (a stand-in for a session child)."""
    script = tmp_path / "child.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from pfb_app.config import Settings\nfrom pfb_app.engine import Pfb3x3Engine\n"
        "s = Settings(data_dir=Path(sys.argv[1]), engine_dir=Path(sys.argv[2]),"
        " engine_python=Path(sys.argv[3]))\n"
        "Pfb3x3Engine(s, tick_sec=0.05).track(Path(sys.argv[4]), 's2_cam1', Path('/v.mp4'),"
        " 0, None, None, None)\n", encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script), str(settings.data_dir),
                             str(settings.engine_dir), str(settings.engine_python), str(sub_dir)])


def test_engine_call_is_recorded_and_removed(env):
    settings, engine, sub_dir, _ = env
    _track(engine, sub_dir)
    assert not engine.call_record.exists()   # removed once the command has finished


def test_stop_stale_call_after_the_session_was_killed(env, tmp_path, monkeypatch):
    settings, engine, sub_dir, _ = env
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids))
    child = _engine_track_in_child(settings, sub_dir, tmp_path)
    deadline = time.monotonic() + 30
    while not (pids.exists() and engine.call_record.exists()) and time.monotonic() < deadline:
        time.sleep(0.05)
    leader, tracker = (int(p) for p in pids.read_text().split())
    child.kill()   # the session dies hard: its finally never runs
    child.wait(timeout=10)
    assert _alive(leader) and _alive(tracker) and engine.call_record.exists()
    assert int(engine.call_record.read_text().split()[0]) == leader
    assert engine.stop_stale_call() is True
    assert not _alive(leader) and not _alive(tracker)
    assert not engine.call_record.exists()
    assert engine.stop_stale_call() is False   # nothing recorded any more


def test_stop_stale_call_never_signals_a_foreign_process(env):
    settings, engine, sub_dir, _ = env
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                             start_new_session=True)
    try:
        engine.call_record.parent.mkdir(parents=True, exist_ok=True)
        # right pid, wrong start time; then the right start time but not an engine command
        engine.call_record.write_text(f"{other.pid} 1\n", encoding="utf-8")
        assert engine.stop_stale_call() is False and _alive(other.pid)
        assert not engine.call_record.exists()
        start = Path(f"/proc/{other.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        engine.call_record.write_text(f"{other.pid} {start}\n", encoding="utf-8")
        assert engine.stop_stale_call() is False and _alive(other.pid)
    finally:
        other.kill()
        other.wait()


def test_rule_names_and_margin_guard(env, monkeypatch):
    _, engine, sub_dir, calls = env
    pos = {"left_front": "Pa", "left_back": "Pf", "right_front": "Pz", "right_back": "Pp"}
    assert engine.rule_names(sub_dir, "s2_cam1", pos) == {
        "P1": "Pa", "P2": "Pf", "P3": "Pz", "P4": "Pp"}
    argv = calls()[-1]["argv"]
    assert argv[0] == "rule-names" and json.loads(argv[argv.index("--positions") + 1]) == pos
    # the first frame >= the task start where everyone is boxed (the default frame can hit
    # the margin guard on real videos)
    assert "--auto-start" in argv
    engine.rule_names(sub_dir, "s2_cam1", pos, auto_start=False)
    assert "--auto-start" not in calls()[-1]["argv"]
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")   # exit 1 with {"error": ...}: no defaults
    assert engine.rule_names(sub_dir, "s2_cam1", pos) is None


def test_quality_returns_per_label(env):
    _, engine, sub_dir, calls = env
    _track(engine, sub_dir)
    q = engine.quality(sub_dir, "s2_cam1")
    assert set(q) == {"P1", "P2", "P3"}
    assert q["P1"]["coverage"] == 1.0 and q["P1"]["mean_h_frac"] == 0.2
    assert calls()[-1]["argv"][0] == "quality"


def test_quality_refused_is_none(env, monkeypatch):
    _, engine, sub_dir, _ = env
    _track(engine, sub_dir)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT_CMD", "quality")
    assert engine.quality(sub_dir, "s2_cam1") is None


def test_quality_takes_its_own_timeout(env, monkeypatch):
    settings, engine, sub_dir, _ = env
    _track(engine, sub_dir)
    seen = []
    real = engine._run
    monkeypatch.setattr(engine, "_run",
                        lambda *a, **kw: seen.append(kw["timeout"]) or real(*a, **kw))
    engine.quality(sub_dir, "s2_cam1", timeout_sec=120)
    engine.quality(sub_dir, "s2_cam1")
    assert seen == [120, settings.engine_timeout_sec]


def test_propose(env, monkeypatch):
    _, engine, sub_dir, calls = env
    names = engine.propose(sub_dir, "s2_cam1", {"Pa": "赤い服", "Pf": "眼鏡"},
                           base_url="http://localhost:8200/v1", model="qwen3vl-8b")
    assert names == {"P1": "Pa", "P2": "Pf"}
    argv = calls()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "qwen3vl-8b"
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    assert engine.propose(sub_dir, "s2_cam1", {}, base_url="u", model="m") is None


def test_transcribe(env):
    settings, engine, sub_dir, calls = env
    res = engine.transcribe(sub_dir, "s2_cam1")
    assert res.segments == 2
    assert res.transcript == sub_dir / "pfb3x3" / "tracks" / "s2_cam1" / "transcript.json"
    argv = calls()[-1]["argv"]
    # pfb_app's own venv has faster-whisper; never the experiment venvs under ~/extern
    whisper = argv[argv.index("--whisper-python") + 1]
    assert whisper == str(default_whisper_python()) and "extern" not in whisper
    assert default_whisper_python() in (PFB_APP_ROOT / ".venv" / "bin" / "python",
                                        Path(sys.executable))
    custom = Pfb3x3Engine(settings.model_copy(update={"whisper_python": Path("/opt/w/python")}))
    assert custom.whisper_python == Path("/opt/w/python")


def test_set_transcript_rewrites_the_entry(env):
    _, engine, sub_dir, _ = env
    _track(engine, sub_dir)
    f = sub_dir / "pfb3x3" / "videos" / "s2_cam1.yaml"
    engine.set_transcript(sub_dir, "s2_cam1", True)
    entry = yaml.safe_load(f.read_text(encoding="utf-8"))
    assert entry["transcript"] is True and entry["activity"] == "紙タワー"
    engine.set_transcript(sub_dir, "s2_cam1", False)
    assert yaml.safe_load(f.read_text(encoding="utf-8"))["transcript"] is False
    with pytest.raises(EngineError):
        engine.set_transcript(sub_dir, "s2_cam9", True)  # no entry: track first


def test_behavior(env):
    settings, engine, sub_dir, calls = env
    res = engine.behavior(sub_dir, "s2_cam1", base_url="http://localhost:8200/v1",
                          model="qwen3vl-8b")
    assert isinstance(res, BehaviorResult)
    assert res.windows == 2 and res.errors == 0 and res.transcript is False
    run_dir = sub_dir / "pfb3x3" / "exp" / "qwen3vl-8b" / "clip_som" / "s2_cam1"
    assert res.rows == run_dir / "behavior.engine.jsonl"
    assert res.overlay_dir == run_dir / "overlay"
    rows = res.load_rows()
    assert [(r["window_id"], r["window_index"]) for r in rows] == [
        ("s2_cam1_w0000", 0), ("s2_cam1_w0001", 1)]
    assert (res.overlay_dir / "s2_cam1_w0000" / "1.jpg").is_file()
    argv = calls()[-1]["argv"]
    assert argv[argv.index("--condition") + 1] == settings.pfb3x3_condition == "clip_som"
    assert argv[argv.index("--frames") + 1] == str(sub_dir / "pfb3x3" / "frames")
    assert argv[argv.index("--exp") + 1] == str(sub_dir / "pfb3x3" / "exp")
    engine.behavior(sub_dir, "s2_cam1", base_url="u", model="qwen3vl-8b", condition="static_ref")
    argv = calls()[-1]["argv"]
    assert argv[argv.index("--condition") + 1] == "static_ref"
    with pytest.raises(ValueError):
        engine.behavior(sub_dir, "s2_cam1", base_url="u", model="m", condition="nope")


def test_behavior_reports_the_transcript_flag(env):
    _, engine, sub_dir, _ = env
    _track(engine, sub_dir, transcript=True)
    engine.transcribe(sub_dir, "s2_cam1")
    res = engine.behavior(sub_dir, "s2_cam1", base_url="u", model="qwen3vl-8b")
    assert res.transcript is True


def test_exit_3_needs_gpu(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    with pytest.raises(EngineNeedsGpu):
        _track(engine, sub_dir)


def test_other_exit_raises_with_stderr_tail(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    with pytest.raises(EngineError) as ei:
        engine.sheet(sub_dir, "s2_cam1")
    assert ei.value.returncode == 1
    assert "failure 1" in ei.value.stderr_tail
    assert "failure 1" in str(ei.value)


def test_missing_engine_python_is_engine_error(env, tmp_path):
    settings, _, sub_dir, _ = env
    engine = Pfb3x3Engine(settings.model_copy(update={"engine_python": tmp_path / "nope"}))
    with pytest.raises(EngineError):
        engine.sheet(sub_dir, "s2_cam1")


def test_result_path_outside_sub_dir_is_rejected(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_OUTSIDE", "1")
    with pytest.raises(EngineError):
        _track(engine, sub_dir)


def test_on_tick_heartbeat_then_finish(env, monkeypatch, tmp_path):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "0.4")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(tmp_path / "pids"))
    ticks = []
    res = engine.sheet(sub_dir, "s2_cam1", on_tick=lambda: ticks.append(1) or False)
    assert res.frame == 152   # the auto-start frame (the default)
    assert len(ticks) >= 2
    child_pid = int((tmp_path / "pids").read_text().split()[1])
    deadline = time.monotonic() + 5
    while _alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child_pid)   # the fake itself leaves nothing behind


def test_yield_terminates_subprocess_and_its_children(env, monkeypatch, tmp_path):
    _, engine, sub_dir, _ = env
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids_file))
    t0 = time.monotonic()

    def tick() -> bool:
        return pids_file.is_file() and len(pids_file.read_text().split()) == 2

    with pytest.raises(EngineYield):
        engine.behavior(sub_dir, "s2_cam1", base_url="u", model="qwen3vl-8b", on_tick=tick)
    assert time.monotonic() - t0 < 30
    engine_pid, child_pid = (int(p) for p in pids_file.read_text().split())
    deadline = time.monotonic() + 5
    while _alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(engine_pid)
    assert not _alive(child_pid)


def test_timeout_terminates_and_raises(env, monkeypatch, tmp_path):
    settings, _, sub_dir, _ = env
    engine = Pfb3x3Engine(settings.model_copy(update={"engine_timeout_sec": 1}), tick_sec=0.05)
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids_file))
    with pytest.raises(EngineError, match="タイムアウト"):
        engine.sheet(sub_dir, "s2_cam1")
    engine_pid = int(pids_file.read_text().split()[0])
    assert not _alive(engine_pid)


def test_engine_model_for():
    assert engine_model_for("standard") == "qwen3vl-8b"
    assert engine_model_for("high") == "qwen3vl-32b"
    with pytest.raises(KeyError):
        engine_model_for("other")


def test_start_and_stop_server(env):
    settings, engine, _, calls = env
    handle = engine.start_server("qwen3vl-8b", 0.25)
    try:
        assert handle.port == 8200 and handle.model == "qwen3vl-8b"
        assert handle.base_url == "http://localhost:8200/v1"
        runtime = settings.data_dir / "pfb3x3_runtime"
        assert (runtime / "serve.pid").read_text().strip() == str(handle.pid)
        assert _alive(handle.pid)
        call = calls()[-1]
        assert call["argv"] == ["serve", "qwen3vl-8b", "8200"]
        assert call["env"]["PFB_GPU_UTIL"] == "0.25"
        assert call["env"]["PFB_SERVE_PORT"] == "8200"
        assert call["env"]["PFB_RUNTIME_DIR"] == str(runtime)
    finally:
        engine.stop_server(handle)
    deadline = time.monotonic() + 5
    while _alive(handle.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(handle.pid)
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()
    engine.stop_server(handle)   # already stopped: no error, nothing killed


def test_start_server_needs_gpu(env, monkeypatch):
    settings, engine, _, _ = env
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    with pytest.raises(EngineNeedsGpu):
        engine.start_server("qwen3vl-8b", 0.25)
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


def test_start_server_rejects_bad_util(env):
    _, engine, _, _ = env
    for util in (0, 1, 1.5):
        with pytest.raises(ValueError):
            engine.start_server("qwen3vl-8b", util)


def test_stop_server_refuses_foreign_pid(env):
    settings, engine, _, _ = env
    handle = engine.start_server("qwen3vl-8b", 0.25)
    try:
        foreign = handle.__class__(pid=handle.pid + 100000, port=handle.port, model=handle.model,
                                   base_url=handle.base_url)
        with pytest.raises(EngineError):
            engine.stop_server(foreign)
        assert _alive(handle.pid)
    finally:
        engine.stop_server(handle)


def test_stop_stale_server(env):
    settings, engine, _, _ = env
    handle = engine.start_server("qwen3vl-8b", 0.25)
    fresh = Pfb3x3Engine(settings, tick_sec=0.05)   # e.g. a new session after a crash
    fresh.stop_stale_server()
    deadline = time.monotonic() + 5
    while _alive(handle.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(handle.pid)
    fresh.stop_stale_server()   # nothing left: no-op


def test_engine_commit(env):
    settings, engine, _, _ = env
    assert engine.engine_commit is None
    (settings.engine_dir / "ENGINE_COMMIT").write_text("abc123\n", encoding="utf-8")
    assert engine.engine_commit == "abc123"


def test_engine_not_set_up(env, tmp_path):
    settings, _, sub_dir, _ = env
    engine = Pfb3x3Engine(settings.model_copy(update={"engine_dir": tmp_path / "missing"}))
    with pytest.raises(EngineError, match="セットアップ"):
        engine.sheet(sub_dir, "s2_cam1")


def test_json_summary_found_before_trailing_noise(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", "trailing")
    assert engine.sheet(sub_dir, "s2_cam1").frame == 152


@pytest.mark.parametrize("mode", ["empty", "nojson"])
def test_exit_0_without_json_summary(env, monkeypatch, mode):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", mode)
    with pytest.raises(EngineError, match="no JSON summary"):
        engine.sheet(sub_dir, "s2_cam1")


def test_exit_1_without_json(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_STDOUT", "nojson")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    with pytest.raises(EngineError) as ei:
        engine.sheet(sub_dir, "s2_cam1")
    assert ei.value.returncode == 1 and "failure 1" in ei.value.stderr_tail
    # without an {"error": ...} summary, rule-names is a real failure, not "no defaults"
    with pytest.raises(EngineError):
        engine.rule_names(sub_dir, "s2_cam1", {"left_front": "Pa"})


def test_yield_kills_grandchild_that_ignores_sigterm(env, monkeypatch, tmp_path):
    settings, _, sub_dir, _ = env
    engine = Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=0.5)
    pids_file, ready = tmp_path / "pids", tmp_path / "ready"
    monkeypatch.setenv("FAKE_ENGINE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_PIDS", str(pids_file))
    monkeypatch.setenv("FAKE_ENGINE_STUBBORN", str(ready))
    with pytest.raises(EngineYield):
        engine.behavior(sub_dir, "s2_cam1", base_url="u", model="qwen3vl-8b",
                        on_tick=ready.is_file)
    engine_pid, child_pid = (int(p) for p in pids_file.read_text().split())
    deadline = time.monotonic() + 5
    while _alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(engine_pid)
    assert not _alive(child_pid)


def test_yield_during_start_server_leaves_no_server(env, monkeypatch):
    settings, engine, _, _ = env
    pidf = settings.data_dir / "pfb3x3_runtime" / "serve.pid"
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "60")
    with pytest.raises(EngineYield):
        engine.start_server("qwen3vl-8b", 0.25, on_tick=pidf.is_file)
    assert not pidf.exists()


def test_failed_cleanup_does_not_mask_the_yield(env, monkeypatch):
    settings, engine, _, _ = env
    pidf = settings.data_dir / "pfb3x3_runtime" / "serve.pid"
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "60")
    monkeypatch.setenv("FAKE_ENGINE_STOP_EXIT", "1")
    with pytest.raises(EngineYield):
        engine.start_server("qwen3vl-8b", 0.25, on_tick=pidf.is_file)
    monkeypatch.delenv("FAKE_ENGINE_STOP_EXIT")
    pid = int(pidf.read_text())
    engine.stop_stale_server()   # clean up the fake server this test left behind
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_a_child_left_in_the_group_after_a_normal_exit_is_stopped(env, monkeypatch, tmp_path):
    _, engine, sub_dir, _ = env
    pid_file = tmp_path / "orphan.pid"
    monkeypatch.setenv("FAKE_ENGINE_ORPHAN", str(pid_file))
    _track(engine, sub_dir)   # the command itself succeeds
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)


def test_engine_errors_name_their_log_file(env, monkeypatch):
    _, engine, sub_dir, _ = env
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    with pytest.raises(EngineError) as e:
        _track(engine, sub_dir)
    lines = str(e.value).splitlines()
    log_file = sub_dir / "pfb3x3" / "logs" / "track.err"
    assert lines[0].startswith("エンジン track が失敗しました (exit 1)")
    assert lines[1] == f"{LOG_PREFIX}{log_file}" and e.value.log_path == log_file
    assert "fake engine track: failure 1" in str(e.value)   # the tail follows


def test_vllm_info_reads_the_serving_venv_without_importing(env, tmp_path):
    settings, engine, _, _ = env
    assert engine.vllm_info("qwen3vl-8b") == {"vllm_version": None, "vllm_venv": None}
    venv = tmp_path / "venvs" / "qwen3vl"
    meta = venv / "lib" / "python3.12" / "site-packages" / "vllm-0.11.0.dist-info" / "METADATA"
    meta.parent.mkdir(parents=True)
    meta.write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.11.0\n", encoding="utf-8")
    (settings.engine_dir / "configs").mkdir()
    (settings.engine_dir / "configs" / "models.yaml").write_text(
        yaml.safe_dump({"qwen3vl-8b": {"venv": str(venv)}}), encoding="utf-8")
    assert engine.vllm_info("qwen3vl-8b") == {"vllm_version": "0.11.0", "vllm_venv": str(venv)}
    # a model without its own venv: serve.sh's default (none here)
    assert engine.vllm_info("other")["vllm_venv"].endswith("extern/venvs/qwen3vl")


def test_write_transcript(env):
    _, engine, sub_dir, _ = env
    res = engine.write_transcript(sub_dir, "s2_cam1", [
        {"start": 1.0, "end": 2.0, "text": " はい "}, {"start": 3.0, "end": float("inf"),
                                                     "text": "どうぞ"},
        {"start": 4.0, "end": 5.0, "text": ""}])
    assert res.segments == 2
    assert json.loads(res.transcript.read_text(encoding="utf-8")) == [
        {"start": 1.0, "end": 2.0, "text": "はい"}, {"start": 3.0, "end": 86400.0, "text": "どうぞ"}]
    assert res.transcript == sub_dir / "pfb3x3" / "tracks" / "s2_cam1" / "transcript.json"


def test_server_alive(env):
    _, engine, _, _ = env
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    handle = ServerHandle(pid=proc.pid, port=8200, model="m", base_url="http://localhost:8200/v1")
    assert engine.server_alive(handle)
    proc.kill()
    proc.wait()
    assert not engine.server_alive(handle)


def test_terminate_logs_a_leader_that_will_not_be_reaped(env, monkeypatch, caplog):
    """The TimeoutExpired branch of _terminate (the leader survives SIGKILL, e.g. stuck in
    uninterruptible I/O): logged with the leader's pid, no exception."""
    _, engine, _, _ = env
    stopped = []
    monkeypatch.setattr(engine, "_stop_group", stopped.append)

    class Stuck:
        pid, returncode = 424242, None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("engine", timeout)

    with caplog.at_level("ERROR"):
        engine._terminate(Stuck())
    assert stopped == [424242]
    assert "424242" in caplog.text and "SIGKILL 後も終了しません" in caplog.text


# --- content check and naming cues (probe, scene, teams, numbers, positions, grid, batting) ---

CUE_FAKE = r'''
"""A stand-in engine for the check and cue commands: the real subparsers' options (an unknown
option is exit 1 with the engine's argparse-style error), canned outputs from FAKE_CUE_OUT."""
import argparse, json, os, sys
from collections import Counter

OPTS = {   # command -> (required, optional) besides --video-id --registry --tracks
    "track": ((), ("--device", "--tracker-python", "--weights")),
    "probe": ((), ("--samples", "--device", "--detector-python", "--weights")),
    "scene": (("--base-url", "--model"), ("--images",)),
    "teams": ((), ("--per-label", "--other-dist")),
    "numbers": (("--base-url", "--model"), ("--reads", "--min-agree", "--max-concurrency",
                                            "--timeline-s")),
    "positions": (("--base-url", "--model", "--positions"), ()),
    "grid": (("--base-url", "--model"), ("--direction",)),
    "batting": (("--base-url", "--model"), ("--every-s", "--windows")),
}


class Parser(argparse.ArgumentParser):
    def error(self, message):   # as the engine: exit 1 + JSON instead of argparse's exit 2
        print(json.dumps({"error": f"{self.prog}: {message}"}))
        sys.exit(1)


argv = sys.argv[1:]
if argv[:2] == ["-m", "pfb3x3.engine"]:
    argv = argv[2:]
with open(os.environ["FAKE_CUE_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"argv": argv}) + "\n")
p = Parser(prog="python -m pfb3x3.engine")
sub = p.add_subparsers(dest="cmd", required=True, parser_class=Parser)
for name, (req, opt) in OPTS.items():
    s = sub.add_parser(name)
    for o in ("--video-id", "--registry", "--tracks", *req):
        s.add_argument(o, required=True)
    for o in opt:
        s.add_argument(o)
    if name == "scene":   # the engine's `scene --activity` flag (profiles-engine)
        s.add_argument("--activity", action="store_true")
a = p.parse_args(argv)
code = int(os.environ.get("FAKE_CUE_EXIT") or 0)
if code:
    print(json.dumps({"error": os.environ.get("FAKE_CUE_ERROR") or f"{a.cmd}: refused"}))
    sys.exit(code)
d = os.path.join(a.tracks, a.video_id)
if os.environ.get("FAKE_CUE_OUTSIDE"):
    d = "/tmp/elsewhere/" + a.video_id
outs = json.loads(open(os.environ["FAKE_CUE_OUT"], encoding="utf-8").read())
out = json.loads(json.dumps(outs[a.cmd]).replace("{dir}", d))
if a.cmd == "positions":   # as the engine: slots counted from the list, every position claimed
    slots = dict(Counter(json.loads(a.positions)))
    claims = Counter(out["positions"].values())
    out.update(slots=slots, claims={k: claims[k] for k in slots})
print("loading ...")
print(json.dumps(out))
'''

CUE_OUTPUTS = {
    "track": {"people": 3, "labels": ["P1", "P2", "P3"], "track_stats": "{dir}/track_stats.json",
              "boxes": "{dir}/boxes.txt"},
    "probe": {"duration_s": 120.5, "fps": 30.0, "width": 1920, "height": 1080, "n_frames": 3615,
              "samples": 2, "persons": {"median": 14, "p90": 16, "max": 17},
              "vehicles": {"median": 0, "p90": 0, "max": 1},
              "camera_motion": {"median_shift_frac": 0.031, "moving": True},
              "frames": ["{dir}/probe/s00.jpg", "{dir}/probe/s01.jpg"]},
    "scene": {"scene": "sport", "sport": "soccer", "sport_text": "futsal",
              "numbers_visible": True, "model": "qwen3vl-8b"},
    "teams": {"teams": {"P1": "T1", "P2": "T2", "P3": "other"}, "reason": "",
              "crops": {"P1": "{dir}/teams/crop_P1.jpg", "P2": "{dir}/teams/crop_P2.jpg"},
              "strips": {"T1": "{dir}/teams/strip_T1.jpg", "T2": "{dir}/teams/strip_T2.jpg"},
              "sizes": {"T1": 1, "T2": 1, "other": 1}, "other_dist": 0.5},
    "numbers": {"numbers": {
        "P1": {"number": "10", "votes": 5, "frames_agree": 3, "reads": 6,
               "all": ["10", "10", "10", "10", "10", None], "frames": [10, 10, 80, 80, 150, 150],
               "failed": 0, "skipped": {}},
        "P2": {"number": None, "votes": 1, "frames_agree": 1, "reads": 4,
               "all": ["7", None, None, None], "frames": [5, 5, 90, 90], "failed": 2,
               "skipped": {"small": 1}},
        "P3": {"number": "00", "votes": 2, "frames_agree": 2, "reads": 2, "all": ["00", "00"],
               "frames": [3, 60], "failed": 0, "skipped": {},
               "timeline": [[3, "00"], [60, "00"], [120, "7"]], "timeline_s": 6.0,
               "segments": [[3, 60, "00", 2]], "dominant": "00", "dominant_share": 0.67}},
        "min_agree": 2, "model": "qwen3vl-8b"},
    "positions": {"positions": {"P1": "GK", "P3": "DF"}},   # slots and claims: from the argv
    "grid": {"order": ["P3", "P1", "P2"], "frame": 12, "frame_rule": "auto_start",
             "direction": "right", "direction_source": "tracks"},
    "batting": {"order": ["P2", "P1"], "frames": 40, "errors": 1,
                "first_frame": {"P2": 30, "P1": 90}, "seen": {"P2": 2, "P1": 1},
                "sequence": [[30, "P2"], [90, "P1"], [150, "P2"]]},
}
URL, MODEL = "http://localhost:8200/v1", "qwen3vl-8b"


@pytest.fixture
def cue_env(tmp_path, monkeypatch):
    from pfb_app.engine import Pfb3x3Engine as _Engine
    engine_dir = tmp_path / "engines" / "pfb3x3"
    engine_dir.mkdir(parents=True)
    script = tmp_path / "cue_fake.py"
    script.write_text(CUE_FAKE, encoding="utf-8")
    wrapper = tmp_path / "cue_python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    outs = tmp_path / "cue_outputs.json"
    outs.write_text(json.dumps(CUE_OUTPUTS), encoding="utf-8")
    log = tmp_path / "cue_calls.jsonl"
    monkeypatch.setenv("FAKE_CUE_LOG", str(log))
    monkeypatch.setenv("FAKE_CUE_OUT", str(outs))
    monkeypatch.delenv("FAKE_CUE_EXIT", raising=False)
    monkeypatch.delenv("FAKE_CUE_OUTSIDE", raising=False)
    monkeypatch.delenv("FAKE_CUE_ERROR", raising=False)
    settings = Settings(data_dir=tmp_path / "data", engine_dir=engine_dir, engine_python=wrapper)
    sub_dir = tmp_path / "data" / "assignments" / "1" / "submissions" / "2"
    sub_dir.mkdir(parents=True)

    def last_argv() -> list[str]:
        return json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["argv"]

    return settings, _Engine(settings, tick_sec=0.05), sub_dir, last_argv


def _opt(argv, name):
    return argv[argv.index(name) + 1]


def test_probe_writes_entry_and_parses(cue_env):
    from pfb_app.engine import ProbeResult
    settings, engine, sub_dir, last_argv = cue_env
    res = engine.probe(sub_dir, "s2_cam1", Path("/videos/v.mp4"), start_s=5, end_s=65,
                       samples=4)
    root = sub_dir / "pfb3x3"
    assert isinstance(res, ProbeResult)
    assert res.duration_s == 120.5
    assert res.persons == {"median": 14, "p90": 16, "max": 17}
    assert res.vehicles["max"] == 1
    assert res.camera_motion == {"median_shift_frac": 0.031, "moving": True}
    d = root / "tracks" / "s2_cam1__probe"   # the check's own registry id
    assert res.frames == [d / "probe" / "s00.jpg", d / "probe" / "s01.jpg"]
    entry = yaml.safe_load((root / "videos" / "s2_cam1__probe.yaml").read_text("utf-8"))
    assert entry["path"] == "/videos/v.mp4" and entry["start_s"] == 5.0
    assert entry["end_s"] == 65.0 and entry["domain"] == "generic"
    argv = last_argv()
    assert argv[0] == "probe" and _opt(argv, "--samples") == "4"
    # never waits for the GPU: the detector runs on the CPU unless asked otherwise
    assert _opt(argv, "--device") == "cpu"
    assert _opt(argv, "--detector-python") == str(settings.engine_python)
    assert _opt(argv, "--registry") == str(root / "videos")
    assert _opt(argv, "--video-id") == "s2_cam1__probe"


def test_probe_never_touches_the_track_entry(cue_env):
    _, engine, sub_dir, last_argv = cue_env
    engine.track(sub_dir, "s2_cam2", Path("/v2.mp4"), 10, 600, 12, "試合", transcript=True,
                 domain="sports", sport="soccer")
    f = sub_dir / "pfb3x3" / "videos" / "s2_cam2.yaml"
    before = f.read_bytes()
    engine.probe(sub_dir, "s2_cam2", Path("/v2.mp4"), start_s=30, end_s=90)
    assert f.read_bytes() == before
    probe = yaml.safe_load((sub_dir / "pfb3x3" / "videos" / "s2_cam2__probe.yaml")
                           .read_text(encoding="utf-8"))
    assert (probe["start_s"], probe["end_s"], probe["domain"]) == (30.0, 90.0, "generic")
    engine.scene(sub_dir, "s2_cam2", base_url=URL, model=MODEL)
    assert _opt(last_argv(), "--video-id") == "s2_cam2__probe"


def test_scene(cue_env):
    from pfb_app.engine import SceneResult
    _, engine, sub_dir, last_argv = cue_env
    res = engine.scene(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert res == SceneResult(scene="sport", sport="soccer", sport_text="futsal",
                              numbers_visible=True, model="qwen3vl-8b")
    argv = last_argv()
    assert argv[0] == "scene" and _opt(argv, "--base-url") == URL and _opt(argv, "--model") == MODEL
    assert _opt(argv, "--video-id") == "s2_cam1__probe"


def test_scene_asks_the_activity_when_the_engine_has_it(cue_env):
    """--activity only when the pinned engine has prompts/check/activity.md: an older engine
    would refuse the option (exit 1), and the check would lose its scene verdict."""
    from pfb_app.engine import SceneResult
    settings, engine, sub_dir, last_argv = cue_env
    engine.scene(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert "--activity" not in last_argv()
    prompt = settings.engine_dir / "prompts" / "check" / "activity.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("x", encoding="utf-8")
    outs = Path(os.environ["FAKE_CUE_OUT"])
    data = json.loads(outs.read_text(encoding="utf-8"))
    data["scene"] = {"scene": "group_work", "sport": None, "sport_text": None,
                     "numbers_visible": False, "model": "qwen3vl-8b", "activity": "meeting",
                     "activity_text": "Meeting"}
    outs.write_text(json.dumps(data), encoding="utf-8")
    res = engine.scene(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert "--activity" in last_argv()
    assert res == SceneResult(scene="group_work", sport=None, sport_text=None,
                              numbers_visible=False, model="qwen3vl-8b", activity="meeting",
                              activity_text="Meeting")


def test_teams(cue_env):
    from pfb_app.engine import TeamsResult
    _, engine, sub_dir, last_argv = cue_env
    res = engine.teams(sub_dir, "s2_cam1")
    d = sub_dir / "pfb3x3" / "tracks" / "s2_cam1" / "teams"
    assert isinstance(res, TeamsResult)
    assert res.teams == {"P1": "T1", "P2": "T2", "P3": "other"} and res.reason == ""
    assert res.crops == {"P1": d / "crop_P1.jpg", "P2": d / "crop_P2.jpg"}
    assert res.strips == {"T1": d / "strip_T1.jpg", "T2": d / "strip_T2.jpg"}
    assert res.sizes == {"T1": 1, "T2": 1, "other": 1} and res.other_dist == 0.5
    assert last_argv()[0] == "teams"


def test_teams_not_person(cue_env, tmp_path):
    _, engine, sub_dir, _ = cue_env
    outs = dict(CUE_OUTPUTS, teams={"teams": {}, "reason": "not_person", "crops": {},
                                    "strips": {}, "sizes": {"T1": 0, "T2": 0, "other": 0},
                                    "other_dist": 0.5})
    (tmp_path / "cue_outputs.json").write_text(json.dumps(outs), encoding="utf-8")
    res = engine.teams(sub_dir, "s2_cam1")
    assert (res.teams, res.reason, res.crops, res.strips) == ({}, "not_person", {}, {})
    assert res.sizes == {"T1": 0, "T2": 0, "other": 0}


def test_numbers(cue_env):
    from pfb_app.engine import NumberRead
    _, engine, sub_dir, last_argv = cue_env
    res = engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert res["P1"] == NumberRead(number="10", votes=5, reads=6, frames_agree=3,
                                   frames=(10, 10, 80, 80, 150, 150),
                                   all=("10", "10", "10", "10", "10", None), failed=0,
                                   skipped={})
    assert res["P2"] == NumberRead(number=None, votes=1, reads=4, frames_agree=1,
                                   frames=(5, 5, 90, 90), all=("7", None, None, None),
                                   failed=2, skipped={"small": 1})
    assert res["P3"].number == "00" and res["P3"].frames_agree == 2   # "00" is not "0"
    argv = last_argv()
    assert argv[0] == "numbers" and _opt(argv, "--base-url") == URL
    assert _opt(argv, "--model") == MODEL


def test_numbers_timeline(cue_env):
    _, engine, sub_dir, last_argv = cue_env
    res = engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL, timeline_s=6.0)
    assert _opt(last_argv(), "--timeline-s") == "6.0"
    p3 = res["P3"]
    assert (p3.timeline_s, p3.dominant, p3.dominant_share) == (6.0, "00", 0.67)
    assert p3.segments == ((3, 60, "00", 2),)
    assert p3.timeline == ((3, "00"), (60, "00"), (120, "7"))
    assert res["P1"].segments == () and res["P1"].dominant is None    # no timeline asked
    engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert "--timeline-s" not in last_argv()


def test_positions(cue_env):
    _, engine, sub_dir, last_argv = cue_env
    engine._write_entry(sub_dir / "pfb3x3", "s2_cam1", Path("/v.mp4"), 0, None, None, None,
                        False, domain="sports", sport="basketball")
    res = engine.positions(sub_dir, "s2_cam1", sport="soccer", positions=["GK", "DF", "DF"],
                           base_url=URL, model=MODEL)
    assert res == {"P1": "GK", "P3": "DF"} and res.error is None
    assert res.slots == {"GK": 1, "DF": 2} and res.claims == {"GK": 1, "DF": 1}
    argv = last_argv()
    assert argv[0] == "positions" and _opt(argv, "--model") == MODEL
    assert json.loads(_opt(argv, "--positions")) == ["GK", "DF", "DF"]
    assert _opt(argv, "--base-url") == URL
    # the engine reads the sport from the registry entry: the client keeps it current
    assert "--sport" not in argv
    entry = yaml.safe_load((sub_dir / "pfb3x3" / "videos" / "s2_cam1.yaml").read_text("utf-8"))
    assert entry["sport"] == "soccer"


def test_positions_sport_rewrite_warns_and_needs_the_entry(cue_env, caplog):
    _, engine, sub_dir, _ = cue_env
    with pytest.raises(EngineError, match="登録ファイル"):   # no entry: track first
        engine.positions(sub_dir, "s2_cam1", sport="soccer", positions=["GK"],
                         base_url=URL, model=MODEL)
    engine._write_entry(sub_dir / "pfb3x3", "s2_cam1", Path("/v.mp4"), 0, None, None, None,
                        False, domain="sports", sport="soccer")
    with caplog.at_level("WARNING", logger="pfb_app.engine.pfb3x3"):
        engine.positions(sub_dir, "s2_cam1", sport="soccer", positions=["GK"],
                         base_url=URL, model=MODEL)
    assert not [r for r in caplog.records if "soccer" in r.getMessage()]   # unchanged: quiet
    with caplog.at_level("WARNING", logger="pfb_app.engine.pfb3x3"):
        engine.positions(sub_dir, "s2_cam1", sport="baseball_softball", positions=["1"],
                         base_url=URL, model=MODEL)
    assert any("baseball_softball" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


def test_grid_and_batting(cue_env):
    _, engine, sub_dir, last_argv = cue_env
    grid = engine.grid(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert grid == ["P3", "P1", "P2"]
    assert (grid.frame, grid.frame_rule, grid.direction, grid.direction_source) == (
        12, "auto_start", "right", "tracks")
    argv = last_argv()
    assert argv[0] == "grid" and _opt(argv, "--base-url") == URL and _opt(argv, "--model") == MODEL
    bat = engine.batting(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert bat == ["P2", "P1"]
    assert (bat.frames, bat.errors) == (40, 1)
    assert bat.first_frame == {"P2": 30, "P1": 90} and bat.seen == {"P2": 2, "P1": 1}
    assert bat.sequence == [[30, "P2"], [90, "P1"], [150, "P2"]]
    argv = last_argv()
    assert argv[0] == "batting" and _opt(argv, "--base-url") == URL
    assert _opt(argv, "--model") == MODEL


def test_cue_soft_error_returns_empty(cue_env, monkeypatch):
    from pfb_app.engine import TeamsResult
    _, engine, sub_dir, _ = cue_env
    engine._write_entry(sub_dir / "pfb3x3", "s2_cam1", Path("/v.mp4"), 0, None, None, None,
                        False, domain="sports", sport="soccer")
    # it ran and found nothing: no error
    assert engine.teams(sub_dir, "s2_cam1").error is None
    assert engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL).error is None
    assert engine.grid(sub_dir, "s2_cam1", base_url=URL, model=MODEL).error is None
    monkeypatch.setenv("FAKE_CUE_EXIT", "1")   # a refusal: empty, with the engine's reason
    teams = engine.teams(sub_dir, "s2_cam1")
    assert isinstance(teams, TeamsResult)
    assert (teams.teams, teams.crops, teams.strips) == ({}, {}, {}) and teams.reason == "error"
    assert teams.error == "teams: refused"
    numbers = engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert numbers == {} and numbers.error == "numbers: refused"
    pos = engine.positions(sub_dir, "s2_cam1", sport="soccer", positions=["GK"],
                           base_url=URL, model=MODEL)
    assert pos == {} and pos.error == "positions: refused" and pos.claims == {}
    grid = engine.grid(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert grid == [] and grid.frame is None and grid.error == "grid: refused"
    bat = engine.batting(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    assert bat == [] and bat.sequence == [] and bat.frames == 0 and bat.error
    # the content check is not a cue: its failures are errors
    with pytest.raises(EngineError):
        engine.scene(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    with pytest.raises(EngineError):
        engine.probe(sub_dir, "s2_cam1", Path("/v.mp4"))
    monkeypatch.setenv("FAKE_CUE_EXIT", "3")   # a GPU shortage is never swallowed
    with pytest.raises(EngineNeedsGpu):
        engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL)


@pytest.mark.parametrize("error", [
    "python -m pfb3x3.engine teams: unrecognized arguments: --x",
    "no registry entry s2_cam1.yaml in /r", "invalid video id '../x'"])
def test_cue_broken_call_is_an_error_not_a_refusal(cue_env, monkeypatch, error):
    _, engine, sub_dir, _ = cue_env
    monkeypatch.setenv("FAKE_CUE_EXIT", "1")
    monkeypatch.setenv("FAKE_CUE_ERROR", error)
    with pytest.raises(EngineError):
        engine.teams(sub_dir, "s2_cam1")
    with pytest.raises(EngineError):
        engine.grid(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    with pytest.raises(EngineError):   # the names commands too
        engine.rule_names(sub_dir, "s2_cam1", {"left_front": "Pa"})


def test_cue_dead_server_is_server_down(cue_env, monkeypatch):
    from pfb_app.engine import EngineServerDown
    _, engine, sub_dir, _ = cue_env
    monkeypatch.setenv("FAKE_CUE_EXIT", "1")
    monkeypatch.setenv("FAKE_CUE_ERROR", "server unavailable (connection refused)")
    with pytest.raises(EngineServerDown):
        engine.numbers(sub_dir, "s2_cam1", base_url=URL, model=MODEL)
    with pytest.raises(EngineServerDown):
        engine.batting(sub_dir, "s2_cam1", base_url=URL, model=MODEL)


def test_cue_fake_rejects_a_stray_option(cue_env):
    _, engine, sub_dir, last_argv = cue_env
    with pytest.raises(EngineError, match="unrecognized arguments"):
        engine._soft(sub_dir / "pfb3x3", "teams", "s2_cam1", ["--sport", "soccer"], None)
    with pytest.raises(EngineError, match="required"):   # a missing required option too
        engine._soft(sub_dir / "pfb3x3", "grid", "s2_cam1", ["--model", MODEL], None)


def test_paths_outside_root_rejected(cue_env, monkeypatch):
    _, engine, sub_dir, _ = cue_env
    monkeypatch.setenv("FAKE_CUE_OUTSIDE", "1")
    with pytest.raises(EngineError):
        engine.probe(sub_dir, "s2_cam1", Path("/v.mp4"))
    with pytest.raises(EngineError):
        engine.teams(sub_dir, "s2_cam1")


def test_write_entry_domain_sport_target(env):
    _, engine, sub_dir, calls = env
    root = sub_dir / "pfb3x3"
    engine.track(sub_dir, "s2_cam1", Path("/v.mp4"), 0, None, 30, "試合",
                 domain="sports", sport="motor_racing", target="vehicle")
    entry = yaml.safe_load((root / "videos" / "s2_cam1.yaml").read_text(encoding="utf-8"))
    assert entry["domain"] == "sports" and entry["sport"] == "motor_racing"
    assert entry["target"] == "vehicle" and entry["max_people"] == 30
    engine.track(sub_dir, "s2_cam2", Path("/v.mp4"), 0, None, None, None, domain="sports")
    entry = yaml.safe_load((root / "videos" / "s2_cam2.yaml").read_text(encoding="utf-8"))
    assert entry["domain"] == "sports" and entry["sport"] is None
    assert entry["target"] == "person"
    with pytest.raises(ValueError):
        engine.track(sub_dir, "s2_cam3", Path("/v.mp4"), 0, None, None, None, target="boat")
    with pytest.raises(ValueError):
        engine.track(sub_dir, "s2_cam3", Path("/v.mp4"), 0, None, None, None, domain="school")


GATE_YAML = """\
qwen3vl-8b:
  number: {enabled: true, threshold: 3, precision: 0.95, coverage: 0.6, n: 40, reason: ""}
  team: {enabled: false, threshold: null, precision: 0.8, coverage: 0.9, n: 30,
         reason: "held-out precision 0.800 < 0.90"}
qwen3vl-32b:
  number: {enabled: true, threshold: 2, precision: 0.97, coverage: 0.7, n: 40, reason: ""}
"""


def test_load_cue_gate_missing_file_disables_all(env):
    from pfb_app.engine import CUES, CueGate, load_cue_gate
    settings, _, _, _ = env
    gate = load_cue_gate(settings, "standard")
    assert set(gate) == set(CUES) == {"number", "team", "position", "grid", "batting",
                                      "appearance", "place"}
    assert all(g == CueGate(enabled=False, threshold=None, reason="評価結果がありません")
               for g in gate.values())
    (settings.engine_dir / "configs").mkdir()
    (settings.engine_dir / "configs" / "cue_gate.yaml").write_text("{not: [yaml", "utf-8")
    assert not any(g.enabled for g in load_cue_gate(settings, "standard").values())


def test_load_cue_gate_reads_model_section(env):
    from pfb_app.engine import CueGate, load_cue_gate
    settings, _, _, _ = env
    (settings.engine_dir / "configs").mkdir()
    (settings.engine_dir / "configs" / "cue_gate.yaml").write_text(GATE_YAML, "utf-8")
    std = load_cue_gate(settings, "standard")
    assert std["number"] == CueGate(enabled=True, threshold=3.0, reason="")
    assert std["team"] == CueGate(enabled=False, threshold=None,
                                  reason="held-out precision 0.800 < 0.90")
    assert std["position"] == CueGate(enabled=False, threshold=None, reason="評価結果がありません")
    high = load_cue_gate(settings, "high")
    assert high["number"].threshold == 2.0 and not high["team"].enabled
    assert high["team"].reason == "評価結果がありません"
    # a model the file does not cover (or an unknown key): every cue disabled
    (settings.engine_dir / "configs" / "cue_gate.yaml").write_text(
        GATE_YAML.split("qwen3vl-32b:")[0], "utf-8")
    assert not any(g.enabled for g in load_cue_gate(settings, "high").values())
    assert not any(g.enabled for g in load_cue_gate(settings, "nope").values())


def test_check_calls_take_their_own_timeout(cue_env, monkeypatch):
    settings, engine, sub_dir, _ = cue_env
    seen = []
    real = engine._run
    monkeypatch.setattr(engine, "_run",
                        lambda *a, **kw: seen.append(kw["timeout"]) or real(*a, **kw))
    engine.probe(sub_dir, "s2_cam1", Path("/videos/v.mp4"), timeout_sec=600)
    engine.scene(sub_dir, "s2_cam1", base_url=URL, model=MODEL, timeout_sec=300)
    engine.probe(sub_dir, "s2_cam1", Path("/videos/v.mp4"))
    assert seen == [600, 300, settings.engine_timeout_sec]


def test_start_server_timeout_stops_it_again(env, monkeypatch):
    from pfb_app.engine import EngineTimeout
    settings, engine, _, _ = env
    monkeypatch.setenv("FAKE_ENGINE_SERVE_SLEEP", "60")
    t0 = time.monotonic()
    with pytest.raises(EngineTimeout, match="タイムアウト"):
        engine.start_server("qwen3vl-8b", 0.25, timeout_sec=1)
    assert time.monotonic() - t0 < 30
    assert not (settings.data_dir / "pfb3x3_runtime" / "serve.pid").exists()


# --- behavior progress: the planned windows and the rows written so far ---

def test_planned_windows_mirror_the_engine():
    assert planned_windows(30.0, 3000, 0.0, None, 3000, "clip_som") == 17
    assert planned_windows(30.0, 3000, 0.0, None, 3000, "static_ref") == 4
    assert planned_windows(30.0, 3000, 0.0, 60.0, 3000, "clip_som") == 10
    assert planned_windows(30.0, 3000, 10.0, None, 3000, "clip_som") == 15
    assert planned_windows(30.0, 3000, 0.0, None, None, "clip_som") == 0
    assert WINDOW_PLAN == {"clip_som": (0.5, 12), "static_som": (5.0, 5),
                           "static_ref": (5.0, 5)}
    assert set(WINDOW_PLAN) == set(CONDITIONS)


def test_count_rows_tolerates_partial_lines_and_duplicates(tmp_path):
    f = tmp_path / "behavior.jsonl"
    f.write_text('{"id": "v_w0000"}\n{"id": "v_w0001"}\n\n{"id": "v_w0001"}\n{"id": "v_w00',
                 encoding="utf-8")
    assert count_rows(f) == 2
    assert count_rows(tmp_path / "missing.jsonl") == 0
    f.write_bytes(b'{"id": "a"}\n{"id": "\xe3\x81')   # cut inside a character
    assert count_rows(f) == 1


def _engine_root(tmp_path, *, meta=True, boxes=True):
    sub_dir = tmp_path / "sub"
    root = Pfb3x3Engine.root(sub_dir)
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "v.yaml").write_text(
        yaml.safe_dump({"video_id": "v", "start_s": 0.0, "end_s": None}), encoding="utf-8")
    tracks = root / "tracks" / "v"
    tracks.mkdir(parents=True)
    if meta:
        (tracks / "meta.json").write_text(json.dumps({"fps": 30, "n_frames": 3000}),
                                          encoding="utf-8")
    if boxes:
        (tracks / "boxes.txt").write_text(
            "1,1,0,0,10,10,1\n1500,2,0,0,10,10,1\n3000,1,0,0,10,10,1\n12,2,0,0,10,10,1\n",
            encoding="utf-8")
    return sub_dir, root


def test_behavior_plan_reads_the_registry_and_tracks(tmp_path):
    engine = Pfb3x3Engine(Settings(data_dir=tmp_path / "data"))
    sub_dir, root = _engine_root(tmp_path)
    rows = root / "exp" / "m" / "clip_som" / "v" / "behavior.jsonl"
    assert engine.behavior_plan(sub_dir, "v", "m") == (rows, 17)
    assert engine.behavior_plan(sub_dir, "v", "m", "static_ref") == (
        root / "exp" / "m" / "static_ref" / "v" / "behavior.jsonl", 4)
    (root / "tracks" / "v" / "meta.json").unlink()
    assert engine.behavior_plan(sub_dir, "v", "m") == (rows, None)


def test_behavior_plan_is_none_without_its_inputs(tmp_path):
    engine = Pfb3x3Engine(Settings(data_dir=tmp_path / "data"))
    sub_dir, root = _engine_root(tmp_path, boxes=False)
    assert engine.behavior_plan(sub_dir, "v", "m")[1] is None
    (root / "tracks" / "v" / "boxes.txt").write_text("garbage\n", encoding="utf-8")
    assert engine.behavior_plan(sub_dir, "v", "m")[1] is None
    (root / "tracks" / "v" / "boxes.txt").write_text("", encoding="utf-8")
    assert engine.behavior_plan(sub_dir, "v", "m")[1] == 0     # nobody boxed: no windows
    (root / "videos" / "v.yaml").unlink()
    assert engine.behavior_plan(sub_dir, "v", "m")[1] is None


def test_count_rows_reads_a_file_again_only_when_it_changed(tmp_path, monkeypatch):
    from pfb_app.engine import pfb3x3 as eng
    f = tmp_path / "behavior.jsonl"
    f.write_text('{"id": "a"}\n', encoding="utf-8")
    reads = []
    real = eng._count_rows
    monkeypatch.setattr(eng, "_count_rows", lambda p: reads.append(p) or real(p))
    assert count_rows(f) == 1 and count_rows(f) == 1 and len(reads) == 1
    with open(f, "a", encoding="utf-8") as out:
        out.write('{"id": "b"}\n')
    assert count_rows(f) == 2 and len(reads) == 2


def test_analysed_seconds(tmp_path):
    engine = Pfb3x3Engine(Settings(data_dir=tmp_path / "data"))
    sub_dir, _ = _engine_root(tmp_path)          # 3000 frames at 30 fps: 100 s
    assert engine.analysed_seconds(sub_dir, "v", 10.0, None) == 90.0
    assert engine.analysed_seconds(sub_dir, "v", 10.0, 40.0) == 30.0
    assert engine.analysed_seconds(sub_dir, "v", 200.0, None) is None
    assert engine.analysed_seconds(sub_dir, "missing", 0.0, None) is None


# --- linecall (line-call evidence of a recorded video) ---

def _sports_track(engine, sub_dir):
    return engine.track(sub_dir, "s2_cam1", Path("/videos/v.mp4"), 0, None, None, None,
                        domain="sports", sport="basketball")


def test_linecall_call_args_and_result(env, tmp_path):
    from pfb_app.engine import LinecallResult
    settings, engine, sub_dir, calls = env
    _sports_track(engine, sub_dir)
    root = sub_dir / "pfb3x3"
    lc = root / "tracks" / "s2_cam1" / "linecall"
    labels = {"A6-sideline_L-120": {"label": "correct", "verdict": "stepped"}}
    lc.mkdir(parents=True)
    (lc / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    res = engine.linecall(sub_dir, "s2_cam1")
    assert isinstance(res, LinecallResult)
    assert res.episodes == 4
    assert res.verdicts == {"stepped": 1, "clear": 1, "undecidable": 2}
    assert res.episodes_json == lc / "episodes.json" and res.episodes_json.is_file()
    assert res.provenance == lc / "provenance.json" and res.provenance.is_file()
    argv = calls()[-1]["argv"]
    assert argv[0] == "linecall"
    assert argv[argv.index("--registry") + 1] == str(root / "videos")
    assert argv[argv.index("--tracks") + 1] == str(root / "tracks")
    assert argv[argv.index("--device") + 1] == "cuda"
    assert argv[argv.index("--tracker-python") + 1] == str(settings.engine_python)
    assert argv[argv.index("--tid-python") + 1] == str(
        Path.home() / "tid_tracker" / "venv" / "bin" / "python")
    assert argv[argv.index("--pose-weights") + 1] == str(
        settings.data_dir / "models" / "yolo11x-pose.pt")
    doc = json.loads(res.episodes_json.read_text(encoding="utf-8"))
    assert doc["status"] == "判定材料" and doc["video_id"] == "s2_cam1"
    eps = doc["episodes"]
    assert [(e["label"], e["verdict"], e["verdict_ja"]) for e in eps] == [
        ("A6", "stepped", "踏んだ"), ("P3", "clear", "踏んでいない"), ("B1", "undecidable", "判定不能"),
        ("P5", "undecidable", "判定不能")]
    last = eps[-1]   # no evidence frame: no crop, overlay, clip, clock or frame
    assert all(last[k] is None for k in ("crop", "overlay", "clip", "clock", "frame"))
    assert last["clip_note"] and last["evidence"] == []
    for e in eps[:3]:   # crop / overlay / clip paths are relative to linecall/ and exist
        for k in ("crop", "overlay", "clip"):
            assert not Path(e[k]).is_absolute() and (lc / e[k]).is_file()
        assert (lc / e["crop"]).read_bytes()[:2] == b"\xff\xd8"   # a real (tiny) JPEG
    # the app's labels.json is only read
    assert json.loads((lc / "labels.json").read_text(encoding="utf-8")) == labels


def test_linecall_tid_python_setting(env, tmp_path):
    settings, _engine, sub_dir, calls = env
    engine = Pfb3x3Engine(settings.model_copy(update={"tid_python": tmp_path / "tidpy"}),
                          tick_sec=0.05)
    _sports_track(engine, sub_dir)
    engine.linecall(sub_dir, "s2_cam1")
    argv = calls()[-1]["argv"]
    assert argv[argv.index("--tid-python") + 1] == str(tmp_path / "tidpy")


def test_linecall_needs_gpu(env, monkeypatch):
    _settings, engine, sub_dir, _calls = env
    _sports_track(engine, sub_dir)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    with pytest.raises(EngineNeedsGpu):
        engine.linecall(sub_dir, "s2_cam1")


def test_linecall_error_and_outside_path(env, monkeypatch):
    _settings, engine, sub_dir, _calls = env
    _sports_track(engine, sub_dir)
    monkeypatch.setenv("FAKE_ENGINE_OUTSIDE", "1")
    with pytest.raises(EngineError, match="の外にあります"):
        engine.linecall(sub_dir, "s2_cam1")
    monkeypatch.delenv("FAKE_ENGINE_OUTSIDE")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_ERROR", "linecall: in progress (pid 4242)")
    with pytest.raises(EngineError, match="in progress"):
        engine.linecall(sub_dir, "s2_cam1")


def test_linecall_malformed_summary_is_engine_error(env, monkeypatch):
    _settings, engine, sub_dir, _calls = env
    monkeypatch.setattr(engine, "_call", lambda *a, **k: {"status": "判定材料"})
    with pytest.raises(EngineError, match="linecall"):
        engine.linecall(sub_dir, "s2_cam1")
