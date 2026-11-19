"""The `names` stage: the gated roster cues run after `track` and their raw results are kept."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.engine import (BattingOrder, EngineError, EngineNeedsGpu, EngineServerDown,
                            GridOrder, NumberRead, NumberReads, PositionsResult, ServerHandle,
                            TeamsResult)
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline import naming
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import execute_job
from pfb_app.pipeline.tracking import GpuUnavailable
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

SERVER = ServerHandle(pid=1, port=8100, model="qwen3vl-8b", base_url="http://127.0.0.1:8100/v1")
KEY = "pfb3x3:standard"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=tmp_path / "engine")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def write_gate(settings, cues: dict, model="qwen3vl-8b"):
    f = settings.engine_dir / "configs" / "cue_gate.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump({model: cues}, allow_unicode=True), encoding="utf-8")


def on(threshold=None):
    return {"enabled": True, "threshold": threshold, "reason": "precision 0.95"}


def off(reason="precision 0.71 < 0.90"):
    return {"enabled": False, "threshold": None, "reason": reason}


ALL_ON = {c: on() for c in ("number", "team", "position", "grid", "batting", "appearance",
                             "place")}


def no_number_names(settings, repo):
    """The admin switch off: numbers are read only for roster proposals (the gate decides)."""
    settings = settings.model_copy(update={"number_names": False})
    repo.settings = settings
    return settings


class FakeEngine:
    """Canned cue results; `fail` maps a method to the exception it raises."""
    engine_commit = "0123456789abcdef0123"

    def __init__(self, sub_root: Path, fail: dict | None = None, refuse: set | None = None,
                 hook=None):
        self.root = sub_root
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail or {}
        self.refuse = refuse or set()
        self.alive = True
        self.hook = hook   # called with the cue name before it answers

    def _call(self, name, **kw):
        self.calls.append((name, kw))
        if self.hook is not None:
            self.hook(name)
        if name in self.fail:
            raise self.fail[name]

    def stop_stale_call(self):   # a session's start-up cleanup
        return False

    def start_server(self, *a, **kw):
        raise AssertionError("no server for a CPU-only names job")

    def names(self):
        return [c for c, _ in self.calls]

    def teams(self, sub_dir, video_id, *, on_tick=None):
        self._call("teams", video_id=video_id)
        if "teams" in self.refuse:
            return TeamsResult(teams={}, reason="error", crops={}, strips={}, error="no boxes")
        root = self.root
        return TeamsResult(teams={"P1": "T1", "P2": "T2", "P3": "other"}, reason="",
                           crops={"P1": root / "teams/P1.jpg", "P2": root / "teams/P2.jpg"},
                           strips={"T1": root / "teams/T1.jpg", "T2": root / "teams/T2.jpg"},
                           sizes={"T1": 1, "T2": 1, "other": 1}, other_dist=0.4)

    def numbers(self, sub_dir, video_id, *, base_url, model, on_tick=None, timeline_s=0.0):
        self._call("numbers", base_url=base_url, model=model,
                   **({"timeline_s": timeline_s} if timeline_s else {}))
        if "numbers" in self.refuse:
            return NumberReads(error="no crops")
        tl = ({"timeline_s": timeline_s, "timeline": ((5, "10"), (9, "10"), (40, "7")),
               "segments": ((5, 9, "10", 2),), "dominant": "10", "dominant_share": 0.67}
              if timeline_s else {})
        return NumberReads({"P1": NumberRead("10", 3, 4, frames_agree=3, frames=(5, 9, 13, 17),
                                             all=("10", "10", "10", "7"), failed=1,
                                             skipped={"small": 2}, **tl),
                            "P2": NumberRead(None, 1, 2, frames_agree=1, frames=(5, 9),
                                             all=("4", None))})

    def positions(self, sub_dir, video_id, *, sport, positions, base_url, model, on_tick=None):
        self._call("positions", sport=sport, positions=positions)
        return PositionsResult({"P1": "FW"}, slots={"FW": 1, "GK": 1},
                               claims={"FW": 1, "GK": 0})

    def grid(self, sub_dir, video_id, *, base_url, model, on_tick=None):
        self._call("grid")
        return GridOrder(["P2", "P1"], frame=30, frame_rule="auto_start", direction="left",
                         direction_source="tracks")

    def batting(self, sub_dir, video_id, *, base_url, model, on_tick=None):
        self._call("batting")
        return BattingOrder(["P1", "P2"], frames=12, errors=1, first_frame={"P1": 3, "P2": 7},
                            seen={"P1": 2, "P2": 1}, sequence=[[3, "P1"], [7, "P2"]])

    def propose(self, sub_dir, video_id, appearance, *, base_url, model, on_tick=None):
        self._call("propose", appearance=appearance)
        return {"P3": "r2"}

    def rule_names(self, sub_dir, video_id, positions, *, auto_start=True, on_tick=None):
        self._call("rule_names", positions=positions, auto_start=auto_start)
        return {"P1": "r1"}

    def server_alive(self, handle):
        return self.alive


def sports_submission(repo, tmp_path, rows, *, sport="soccer", domain="sports",
                      model_key="standard"):
    aid = repo.create_assignment("試合", Kind.video, "練習試合", "", model_key, True,
                                 domain=domain, sport=sport)
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": str(tmp_path / "cam1.mp4")},
                                             "engine": "pfb3x3", "start_sec": 0.0})
    for pid, name, attrs, places in rows:
        repo.add_subject(sid, pid, name, attrs.get("appearance"), places, attrs)
    return aid, sid


def save(repo, sid, stage, output, camera="cam1", upstream=(), status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "x", warnings=[],
        duration_sec=0.0, created_at=""))


def tracked(repo, sid, camera="cam1"):
    return save(repo, sid, Stage.track, {
        "labels": ["P1", "P2", "P3"], "people": 3, "video_id": f"s{sid}_{camera}",
        "sheet_frame_rule": "auto_start"}, camera=camera)


def make_ctx(settings, repo, engine, server=SERVER):
    return StageContext(settings, repo, FakeBackend(["{}"]), lambda: False, engine=engine,
                        server=server)


def claim(repo, sid, camera="cam1", key=KEY):
    repo.enqueue(sid, Stage.names, key, camera=camera)
    job = repo.claim_next_job({key})
    assert job is not None and job.stage == Stage.names
    return job


ROSTER = [("r1", "山田", {"name": "山田", "number": "10", "position": "FW"}, {}),
          ("r2", "佐藤", {"name": "佐藤", "number": "1", "position": "GK"}, {})]


# --- which cues run ---

def test_a_game_reads_the_number_timeline(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    settings = settings.model_copy(update={"number_timeline_s": 6.0})
    repo.settings = settings
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    call = next(kw for name, kw in engine.calls if name == "numbers")
    assert call["timeline_s"] == 6.0
    p1 = run.output["cues"]["number"]["P1"]
    assert p1["dominant"] == "10" and p1["segments"] == [[5, 9, "10", 2]]
    assert p1["timeline"] == [[5, "10"], [9, "10"], [40, "7"]] and p1["timeline_s"] == 6.0


def test_no_timeline_when_switched_off(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    settings = settings.model_copy(update={"number_timeline_s": 0.0})
    repo.settings = settings
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert "timeline_s" not in next(kw for name, kw in engine.calls if name == "numbers")
    assert "segments" not in run.output["cues"]["number"]["P1"]


def test_names_runs_only_enabled_cues_with_inputs(settings, repo, tmp_path):
    write_gate(settings, {**ALL_ON, "position": off()})
    settings = settings.model_copy(update={"number_timeline_s": 0.0})   # timeline: see above
    repo.settings = settings
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    track = tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))

    assert engine.names() == ["teams", "numbers"]
    assert engine.calls[1][1] == {"base_url": SERVER.base_url, "model": SERVER.model}
    assert run.status == "ok" and run.stage == Stage.names and run.camera == "cam1"
    assert run.upstream_run_ids == [track.id]
    out = run.output
    assert out["track_run_id"] == track.id and out["model"] == "qwen3vl-8b"
    assert out["disabled"] == {"position": "precision 0.71 < 0.90"}
    status = out["status"]
    assert status["position"] == {"status": "disabled", "reason": "precision 0.71 < 0.90"}
    assert status["team"] == {"status": "ok"} and status["number"] == {"status": "ok"}
    for cue in ("grid", "batting", "appearance", "place"):
        assert status[cue] == {"status": "no_roster_column"}
    # the raw engine fields, as JSON
    assert out["cues"]["number"]["P1"] == {
        "number": "10", "votes": 3, "reads": 4, "frames_agree": 3, "frames": [5, 9, 13, 17],
        "all": ["10", "10", "10", "7"], "failed": 1, "skipped": {"small": 2}}
    assert out["cues"]["number"]["P2"]["number"] is None
    assert set(out["cues"]) == {"team", "number"}
    assert out["teams"] == {"P1": "T1", "P2": "T2", "P3": "other"}
    assert out["team_reason"] == ""
    assert out["crops"] == {"P1": str(tmp_path / "teams/P1.jpg"),
                            "P2": str(tmp_path / "teams/P2.jpg")}
    assert out["strips"]["T1"] == str(tmp_path / "teams/T1.jpg")
    assert out["cues"]["team"]["sizes"] == {"T1": 1, "T2": 1, "other": 1}
    assert out["gate"]["number"] == {"enabled": True, "threshold": None}
    assert out["roster_floor"] == 0
    assert run.warnings == []


def test_names_runs_every_cue_with_its_inputs(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    rows = [("r1", "山田", {"name": "山田", "number": "10", "position": "FW", "grid": "1",
                           "batting": "1", "appearance": "赤い帽子", "place": "左前"},
             {"cam1": "左前"}),
            ("r2", "佐藤", {"name": "佐藤", "number": "1", "position": "GK", "grid": "2",
                           "batting": "2", "place": "右奥"}, {"cam1": "右奥"})]
    _, sid = sports_submission(repo, tmp_path, rows, sport="baseball")
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    out = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid)).output

    assert engine.names() == ["teams", "numbers", "positions", "grid", "batting", "propose",
                              "rule_names"]
    kw = dict(engine.calls)
    assert kw["positions"] == {"sport": "baseball", "positions": ["FW", "GK"]}
    assert kw["propose"] == {"appearance": {"r1": "赤い帽子"}}
    assert kw["rule_names"] == {"positions": {"left_front": "r1", "right_back": "r2"},
                                "auto_start": True}
    cues = out["cues"]
    assert cues["position"] == {"positions": {"P1": "FW"}, "slots": {"FW": 1, "GK": 1},
                                "claims": {"FW": 1, "GK": 0}}
    assert cues["grid"] == {"order": ["P2", "P1"], "frame": 30, "frame_rule": "auto_start",
                            "direction": "left", "direction_source": "tracks"}
    assert cues["batting"] == {"order": ["P1", "P2"], "frames": 12, "errors": 1,
                               "first_frame": {"P1": 3, "P2": 7}, "seen": {"P1": 2, "P2": 1},
                               "sequence": [[3, "P1"], [7, "P2"]]}
    assert cues["appearance"] == {"names": {"P3": "r2"}}
    assert cues["place"] == {"names": {"P1": "r1"},
                             "positions": {"left_front": "r1", "right_back": "r2"}}
    assert all(s == {"status": "ok"} for s in out["status"].values())
    assert out["disabled"] == {}


def test_school_work_runs_no_team_cue(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, [("r1", "A子", {"name": "A子", "number": "3"}, {})],
                               domain="school", sport=None)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    out = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid)).output
    assert engine.names() == ["numbers"]
    assert out["status"]["team"] == {"status": "not_sports"}
    assert out["teams"] == {} and out["team_reason"] is None
    assert out["crops"] == {} and out["strips"] == {}


def test_positions_need_the_sport(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER, sport=None)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    out = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid)).output
    assert "positions" not in engine.names()
    assert out["status"]["position"] == {"status": "no_sport"}


def test_no_gate_file_disables_every_cue(settings, repo, tmp_path):
    settings = no_number_names(settings, repo)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert engine.calls == [] and run.status == "ok"
    assert set(run.output["disabled"]) == {"number", "team", "position", "grid", "batting",
                                           "appearance", "place"}
    assert run.output["cues"] == {}


# --- errors ---

def test_names_cue_error_is_warning(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, fail={"numbers": EngineError("エンジン numbers が失敗しました")},
                        refuse={"teams"})
    ctx = make_ctx(settings, repo, engine)
    job = claim(repo, sid)
    execute_job(job, ctx)

    assert repo.get_job(job.id).status == JobStatus.done
    run = repo.latest_run(sid, Stage.names, "cam1")
    assert run.status == "ok"
    out = run.output
    assert engine.names() == ["teams", "numbers", "positions"]
    assert out["status"]["number"]["status"] == "failed"
    assert "numbers" in out["status"]["number"]["error"]
    assert out["status"]["team"] == {"status": "refused", "error": "no boxes"}
    assert out["status"]["position"] == {"status": "ok"}
    assert set(out["cues"]) == {"team", "position"}   # a refusal keeps its (empty) raw result
    assert out["cues"]["team"]["teams"] == {} and out["team_reason"] == "error"
    assert run.warnings == ["背番号から名前を推定できませんでした"]
    # no internal words in the warning shown on the detail screen
    assert not any(w in run.warnings[0] for w in ("numbers", "エンジン", "engine"))


def test_names_refusal_is_not_a_failure(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, refuse={"numbers"})
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert run.status == "ok" and run.warnings == []
    assert run.output["status"]["number"] == {"status": "refused", "error": "no crops"}
    assert run.output["cues"]["number"] == {}


def test_names_requires_server(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    ctx = make_ctx(settings, repo, engine, server=None)
    with pytest.raises(RuntimeError, match="サーバー"):
        naming.run_names(ctx, claim(repo, sid))
    assert engine.calls == []


def test_names_without_track_fails(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    run = naming.run_names(make_ctx(settings, repo, FakeEngine(tmp_path)), claim(repo, sid))
    assert run.status == "failed" and "追跡" in run.error


def test_names_server_down_requeues(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, fail={"numbers": EngineServerDown("server unavailable")})
    job = claim(repo, sid)
    with pytest.raises(EngineServerDown):
        execute_job(job, make_ctx(settings, repo, engine))
    assert repo.get_job(job.id).status == JobStatus.queued
    assert repo.latest_run(sid, Stage.names, "cam1") is None


def test_names_hard_error_from_a_dead_server_requeues(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, fail={"numbers": EngineError("connection refused")})
    engine.alive = False
    job = claim(repo, sid)
    with pytest.raises(EngineServerDown):
        execute_job(job, make_ctx(settings, repo, engine))
    assert repo.get_job(job.id).status == JobStatus.queued


def test_names_needs_gpu_requeues(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, fail={"numbers": EngineNeedsGpu("GPUの空きが足りません")})
    job = claim(repo, sid)
    with pytest.raises(GpuUnavailable):
        execute_job(job, make_ctx(settings, repo, engine))
    assert repo.get_job(job.id).status == JobStatus.queued


# --- the current names run ---

def test_current_names_run_respects_track_and_roster_floor(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    ctx = make_ctx(settings, repo, FakeEngine(tmp_path))
    run = naming.run_names(ctx, claim(repo, sid))
    assert naming.current_names_run(repo, sid, "cam1").id == run.id
    assert naming.current_names_run(repo, sid, "cam2") is None
    # a roster change (Task 7 writes the floor) makes it stale
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "roster_run_floor": run.id})
    assert naming.current_names_run(repo, sid, "cam1") is None
    again = naming.run_names(ctx, claim(repo, sid))
    assert again.output["roster_floor"] == run.id
    assert naming.current_names_run(repo, sid, "cam1").id == again.id
    # a new track run makes it stale too
    tracked(repo, sid)
    assert naming.current_names_run(repo, sid, "cam1") is None


def test_failed_names_run_is_not_current(settings, repo, tmp_path):
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    track = tracked(repo, sid)
    save(repo, sid, Stage.names, None, upstream=[track.id], status="failed")
    assert naming.current_names_run(repo, sid, "cam1") is None


# --- the pfb3x3 session claims names jobs ---

def test_pfb3x3_session_runs_a_names_job(tmp_path, monkeypatch):
    from fakes.engine_env import install_fake_engine
    from pfb_app.engine import Pfb3x3Engine
    from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
    from pfb_app.worker.session import EXIT_IDLE, run_session

    fake = install_fake_engine(tmp_path, monkeypatch)
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                        engine_dir=fake.engine_dir, engine_python=fake.engine_python,
                        session_idle_sec=0, number_names=False)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    jid = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    guard = GpuGuard(settings, query=lambda: GpuSnapshot(97.9, 7.9))
    engine = Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=2)

    code = run_session(None, KEY, settings=settings, guard=guard,
                       engine_factory=lambda s: engine, sleep=lambda s: None,
                       heartbeat_sec=0.05)

    assert code == EXIT_IDLE
    assert repo.get_job(jid).status == JobStatus.done
    run = repo.latest_run(sid, Stage.names, "cam1")
    # no gate file in the fake engine: every cue is disabled, nothing else is called
    assert run.status == "ok" and run.output["cues"] == {}
    assert run.output["model"] == "qwen3vl-8b"
    assert fake.commands() == ["serve", "stop"]   # only the server: no cue ran


def test_names_wanted_checks_the_gate_and_the_camera(settings, repo, tmp_path):
    rows = [("r1", "山田", {"name": "山田", "place": "左前"}, {"cam1": "左前"})]
    _, sid = sports_submission(repo, tmp_path, rows, domain="school", sport=None)
    sub = repo.get_submission(sid)
    assert naming.needs_names(repo, sub)
    assert not naming.names_wanted(repo, sub, "cam1")          # no gate file yet
    write_gate(settings, {"place": on()})
    assert naming.names_wanted(repo, sub, "cam1")
    assert not naming.names_wanted(repo, sub, "cam2")          # no place for this camera
    bare = Repo(repo.conn)                                     # no settings: needs_names only
    assert naming.names_wanted(bare, sub, "cam2")


# --- review round 1 ---

def test_names_cue_value_error_is_warning(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path, fail={"teams": ValueError("bad"), "numbers": KeyError("x"),
                                        "positions": TypeError("y")})
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert run.status == "ok"
    assert {c: run.output["status"][c]["status"] for c in ("team", "number", "position")} == {
        "team": "failed", "number": "failed", "position": "failed"}
    assert len(run.warnings) == 3


ONLY_NAMES = [("r1", "山田", {"name": "山田"}, {}), ("r2", "佐藤", {"name": "佐藤"}, {})]
CPU_GATE = {**ALL_ON, "number": off(), "appearance": off()}


def test_cpu_names_job_runs_teams_without_a_server(settings, repo, tmp_path):
    from pfb_app.pipeline import runner
    settings = no_number_names(settings, repo)
    write_gate(settings, CPU_GATE)
    _, sid = sports_submission(repo, tmp_path, ONLY_NAMES)
    assert runner.model_key_for(repo, sid, Stage.names, "cam1") == naming.NAMES_CPU_KEY
    assert runner.model_key_for(repo, sid, Stage.names) == KEY   # no camera: the brief's key
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    job = claim(repo, sid, key=naming.NAMES_CPU_KEY)
    run = naming.run_names(make_ctx(settings, repo, engine, server=None), job)
    assert engine.names() == ["teams"]
    assert run.status == "ok" and run.output["model"] is None
    assert run.output["teams"] == {"P1": "T1", "P2": "T2", "P3": "other"}
    assert run.model_id.startswith("pfb3x3:cpu")


def test_cpu_names_job_marks_server_cues_no_server(settings, repo, tmp_path):
    settings = no_number_names(settings, repo)
    write_gate(settings, ALL_ON)    # the gate changed after the job was queued
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine, server=None),
                           claim(repo, sid, key=naming.NAMES_CPU_KEY))
    assert engine.names() == ["teams"]
    assert run.output["status"]["number"] == {"status": "no_server"}
    assert run.output["status"]["position"] == {"status": "no_server"}


def test_sports_roster_with_only_names_starts_no_server(settings, repo, tmp_path):
    from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
    from pfb_app.pipeline.runner import enqueue_followups
    from pfb_app.worker.main import worker_loop
    from pfb_app.worker.session import EXIT_IDLE, run_session
    write_gate(settings, CPU_GATE)
    settings = settings.model_copy(update={"session_idle_sec": 0, "number_names": False})
    repo.settings = settings
    _, sid = sports_submission(repo, tmp_path, ONLY_NAMES)
    tracked(repo, sid)
    enqueue_followups(repo, SimpleNamespace(submission_id=sid, stage=Stage.track,
                                            camera="cam1", subject_id=None))
    jid, key = repo.conn.execute("SELECT id, model_key FROM jobs WHERE stage='names'").fetchone()
    assert key == "names:cpu"
    # the worker starts it at once, even with a full GPU
    full = GpuGuard(settings, query=lambda: GpuSnapshot(97.9, 97.0))
    spawned = []
    worker_loop(settings, guard=full, spawn=lambda k: spawned.append(k) or EXIT_IDLE,
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    assert spawned == ["names:cpu"]
    engine = FakeEngine(tmp_path)
    code = run_session(None, "names:cpu", settings=settings, guard=full,
                       engine_factory=lambda s: engine, sleep=lambda s: None,
                       heartbeat_sec=0.05)
    assert code == EXIT_IDLE and repo.get_job(jid).status == JobStatus.done
    assert engine.names() == ["teams"]   # start_server would have failed the test
    assert repo.latest_run(sid, Stage.names, "cam1").output["teams"]["P1"] == "T1"


def test_names_run_from_old_subjects_is_not_current(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)

    def roster_change(cue):   # the coach replaces the roster while the job runs
        if cue != "teams":
            return
        repo.delete_subjects(sid)
        repo.add_subject(sid, "r1", "鈴木", None, {}, {"name": "鈴木", "number": "9"})
        meta = repo.get_submission(sid)["meta"]
        floor = repo.conn.execute("SELECT MAX(id) FROM stage_runs").fetchone()[0]
        repo.update_submission_meta(sid, {**meta, "roster_run_floor": floor})

    run = naming.run_names(make_ctx(settings, repo, FakeEngine(tmp_path, hook=roster_change)),
                           claim(repo, sid))
    assert run.id > repo.get_submission(sid)["meta"]["roster_run_floor"]
    assert naming.current_names_run(repo, sid, "cam1") is None   # built from the old roster
    fresh = naming.run_names(make_ctx(settings, repo, FakeEngine(tmp_path)), claim(repo, sid))
    assert naming.current_names_run(repo, sid, "cam1").id == fresh.id
    # a person the coach names on the mapping screen does not make it stale
    repo.add_subject(sid, "n1", "飛び入り", None, {})
    assert naming.current_names_run(repo, sid, "cam1").id == fresh.id


# --- names jobs never hold the coach up ---

def mapping_ready(repo, sid):
    from pfb_app.pipeline.tracking import video_id
    prep = save(repo, sid, Stage.prepare, {"frames": []})
    return save(repo, sid, Stage.track, {"labels": ["P1", "P2"], "people": 2,
                                         "video_id": video_id(sid, "cam1")},
                upstream=[prep.id])


@pytest.mark.parametrize("number_names", [True, False])
@pytest.mark.parametrize("status", ["queued", "waiting_gpu", "running", "failed_retryable"])
def test_names_job_never_blocks_the_coach(repo, tmp_path, status, number_names):
    from pfb_app.domain.models import SubjectStatus
    from pfb_app.services.review import confirm_identity, save_tracked_identity, subject_status
    from pfb_app.ui.wizard.progress import work_progress
    if not number_names:
        no_number_names(repo.settings, repo)   # names jobs read only roster cues
    aid, sid = sports_submission(repo, tmp_path, ROSTER)
    mapping_ready(repo, sid)
    jid = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status=?, error=? WHERE id=?",
                      (status, "x" if status.startswith("failed") else None, jid))
    assert not repo.has_blocking_jobs(sid) and repo.has_active_jobs(sid) == (
        status != "failed_retryable")
    p = work_progress(repo, aid)
    assert (p.your_turn, p.stopped) == (True, False)   # the mapping is the coach's turn
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": None})
    left = repo.conn.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()
    # a names job not yet started is cancelled once the camera is mapped (unless it reads the
    # numbers that name the automatic people); a running one stays
    assert (left is None) == (status in ("queued", "waiting_gpu") and not number_names)
    confirm_identity(repo, sid)
    p = work_progress(repo, aid)   # behavior is queued; names never shows up or stops it
    assert not p.stopped and "名前" not in p.reason
    subj = repo.list_subjects(sid)[0]["id"]
    assert subject_status(repo, subj) == SubjectStatus.processing   # behavior, not names
    repo.conn.execute("UPDATE jobs SET status='done' WHERE stage!='names'")
    assert subject_status(repo, subj) != SubjectStatus.processing


def test_progress_names_job_line_is_plain():
    from pfb_app.ui.wizard.progress import job_line
    assert job_line("names", "running", "cam1", "video") == "名前の候補を作っています（カメラ1）"
    assert "名前の候補づくり" in job_line("names", "waiting_gpu", None, "video")


def test_apply_roster_requests_names_even_while_one_runs(settings, repo, tmp_path):
    from pfb_app.services.roster import apply_roster, parse_roster_text
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    running = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (running,))
    apply_roster(repo, sid, parse_roster_text("背番号,名前\n10,山田\n7,鈴木"), "sports")
    rows = repo.conn.execute("SELECT id, status, model_key FROM jobs WHERE stage='names'"
                             " ORDER BY id").fetchall()
    assert [tuple(r)[1:] for r in rows] == [("running", KEY), ("queued", KEY)]
    # a second change keeps the queued one
    apply_roster(repo, sid, parse_roster_text("背番号,名前\n10,山田\n8,鈴木"), "sports")
    assert repo.conn.execute("SELECT COUNT(*) FROM jobs WHERE stage='names'").fetchone()[0] == 2


def test_newest_matching_names_run_is_current(settings, repo, tmp_path):
    write_gate(settings, ALL_ON)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    track = tracked(repo, sid)
    ctx = make_ctx(settings, repo, FakeEngine(tmp_path))
    first = naming.run_names(ctx, claim(repo, sid))
    matching = naming.run_names(ctx, claim(repo, sid))
    # a later run built from other subjects (a job that started before a roster change and
    # reverted) is saved after the matching one: the matching one stays current
    save(repo, sid, Stage.names, {**matching.output, "roster_fingerprint": "old"},
         upstream=[track.id])
    save(repo, sid, Stage.names, None, upstream=[track.id], status="failed")
    assert naming.current_names_run(repo, sid, "cam1").id == matching.id
    # the floor still cuts off everything before it
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "roster_run_floor": matching.id})
    assert naming.current_names_run(repo, sid, "cam1") is None
    assert first.id < matching.id
    # a new track run makes them all stale
    repo.update_submission_meta(sid, meta)
    tracked(repo, sid)
    assert naming.current_names_run(repo, sid, "cam1") is None


# --- display names from numbers (the coach's request of 2026-10-22) ---

NUMBER_OFF = {**ALL_ON, "number": off()}


def test_sports_work_reads_numbers_for_names_even_with_the_gate_off(settings, repo, tmp_path):
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, [])            # no roster at all
    sub = repo.get_submission(sid)
    assert naming.wanted_cues(repo, sub, "cam1") == {"team", "number"}
    assert naming.names_wanted(repo, sub, "cam1") and naming.needs_server(repo, sub, "cam1")
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert engine.names() == ["teams", "numbers"]
    out = run.output
    assert out["status"]["number"] == {"status": "ok"}
    assert out["cues"]["number"]["P1"]["number"] == "10"
    # the gate still holds for roster proposals: it is recorded as it is
    assert out["gate"]["number"]["enabled"] is False
    assert out["disabled"] == {"number": "precision 0.71 < 0.90"}


def test_no_gate_file_still_reads_numbers_on_a_sports_work(settings, repo, tmp_path):
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert engine.names() == ["numbers"]
    assert run.output["status"]["number"] == {"status": "ok"}
    assert run.output["status"]["team"]["status"] == "disabled"


def test_school_work_keeps_the_number_gate(settings, repo, tmp_path):
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, [("r1", "A子", {"name": "A子", "number": "3"}, {})],
                               domain="school", sport=None)
    tracked(repo, sid)
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert "numbers" not in engine.names()
    assert run.output["status"]["number"]["status"] == "disabled"


def test_switch_off_keeps_the_number_gate_on_a_sports_work(settings, repo, tmp_path):
    settings = no_number_names(settings, repo)
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, [])
    assert naming.wanted_cues(repo, repo.get_submission(sid), "cam1") == {"team"}


def test_a_finished_track_queues_names_on_the_server_without_a_roster(settings, repo, tmp_path):
    from pfb_app.pipeline.runner import enqueue_followups
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, [])
    tracked(repo, sid)
    enqueue_followups(repo, SimpleNamespace(submission_id=sid, stage=Stage.track,
                                            camera="cam1", subject_id=None))
    keys = [k for (k,) in repo.conn.execute("SELECT model_key FROM jobs WHERE stage='names'")]
    assert keys == [KEY]


def test_numbers_are_not_read_for_names_when_no_automatic_person_is_mapped(settings, repo,
                                                                          tmp_path):
    from pfb_app.services.review import save_tracked_identity
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": "r2", "P3": None})
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert "numbers" not in engine.names()          # the roster names everyone: no GPU read
    assert run.output["status"]["number"]["status"] == "disabled"


def test_numbers_are_read_for_names_when_an_automatic_person_is_mapped(settings, repo,
                                                                      tmp_path):
    from pfb_app.services.review import AUTO, save_tracked_identity
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, ROSTER)
    tracked(repo, sid)
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": AUTO + "P2", "P3": None})
    engine = FakeEngine(tmp_path)
    run = naming.run_names(make_ctx(settings, repo, engine), claim(repo, sid))
    assert "numbers" in engine.names()
    assert run.output["status"]["number"] == {"status": "ok"}


def test_a_roster_mapped_camera_runs_its_team_cue_on_the_cpu(settings, repo, tmp_path):
    from pfb_app.pipeline import runner
    from pfb_app.services.review import AUTO, save_tracked_identity
    write_gate(settings, NUMBER_OFF)
    _, sid = sports_submission(repo, tmp_path, ONLY_NAMES)
    tracked(repo, sid)
    # no mapping yet: it is not known who will be named, so the numbers are read
    assert naming.wanted_cues(repo, repo.get_submission(sid), "cam1") == {"team", "number"}
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": "r2", "P3": None})
    assert naming.wanted_cues(repo, repo.get_submission(sid), "cam1") == {"team"}
    assert runner.model_key_for(repo, sid, Stage.names, "cam1") == naming.NAMES_CPU_KEY
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": AUTO + "P2", "P3": None})
    assert naming.wanted_cues(repo, repo.get_submission(sid), "cam1") == {"team", "number"}
    assert runner.model_key_for(repo, sid, Stage.names, "cam1") == KEY
