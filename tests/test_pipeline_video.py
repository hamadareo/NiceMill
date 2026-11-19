import json
import re
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline.video import (StageContext, run_behavior, run_contribution, run_identity,
                                    run_pfb)
from pfb_app.pipeline.prompts import load_prompt
from pfb_app.preprocess.frames import Frame, format_time
from pfb_app.storage.db import Repo, connect, init_schema

NO_TRANSCRIPT_TEXT = "文字起こしはありません"
OFFSET = 10.0  # the task starts 10 s into the video


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def make_frames(tmp_path, camera, n):
    d = tmp_path / "frames" / camera
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(n):
        p = d / f"f{k}.jpg"
        p.write_bytes(f"{camera}-{k}".encode())
        t = OFFSET + 5.0 * k
        frames.append(Frame(k, t, format_time(t), format_time(5.0 * k), p))
    return frames


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "x", warnings=[],
        duration_sec=0.0, created_at=""))


@pytest.fixture
def env(tmp_path, settings):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("課題", Kind.video, "紙で塔を作る課題", "ルーブリック項目Q",
                                 "standard", False)
    sid = repo.create_submission(aid, "G1", {})
    s_a = repo.add_subject(sid, "Pa", "A子", "縞模様の服", {"cam1": "左端", "cam2": "右奥"})
    s_z = repo.add_subject(sid, "Pz", "Z男", "無地の服", {"cam1": "右端", "cam2": "左奥"})
    return repo, sid, {"Pa": s_a, "Pz": s_z}


def ctx_for(settings, repo, fn, yield_check=lambda: False):
    backend = FakeBackend(fn)
    return StageContext(settings, repo, backend, yield_check), backend


def add_prepare(env_, tmp_path, camera, n):
    repo, sid, _ = env_
    frames = make_frames(tmp_path, camera, n)
    run = save(repo, sid, Stage.prepare, {"frames": [f.to_dict() for f in frames]}, camera=camera)
    return frames, run


def add_identity(repo, sid, camera, ids=("Pa", "Pz")):
    return save(repo, sid, Stage.identity, {"participants": [
        {"participant_id": i, "appearance": "外見", "initial_position": "位置",
         "verification_points": [], "identity_confidence": "high", "verified": True,
         "notes": ""} for i in ids]}, camera=camera)


def ident_json(ids):
    return json.dumps({"participants": [
        {"participant_id": i, "appearance": "外見", "initial_position": "位置",
         "verification_points": ["開始時点で確認"], "identity_confidence": "high",
         "verified": True, "notes": ""} for i in ids]}, ensure_ascii=False)


def ev(time, who="Pa", text="紙を折る"):
    return {"video_time": time, "participant": who, "observed_behavior": text, "utterance": "",
            "speaker": "unknown", "evidence_source": "video", "confidence": "high"}


def times_in(req):
    return re.findall(r"時刻 (\d\d:\d\d)", req.user_text)


def behavior_json(events):
    return json.dumps({"events": events}, ensure_ascii=False)


def judg(ids, true_ids=()):
    return json.dumps({"judgments": [
        {"ref_id": i, "contribution_interpretation": "解釈",
         "is_positive_contribution": i in true_ids, "rubric_criteria": []} for i in ids]})


def pfb_json(refs):
    return json.dumps({"pfb_status": "generated", "pfb": "紙を折って塔づくりを進めていました。",
                       "evidence_refs": refs}, ensure_ascii=False)


def events_run(repo, sid, camera, events):
    stored = [{**e, "event_id": f"{camera}-{n:04d}", "camera": camera, "task_elapsed": "00:00"}
              for n, e in enumerate(events, 1)]
    return save(repo, sid, Stage.behavior, {"events": stored}, camera=camera)


# --- identity ---

def test_identity_happy_path_and_request(settings, env, tmp_path):
    repo, sid, _ = env
    frames, prep = add_prepare(env, tmp_path, "cam1", 9)
    ctx, backend = ctx_for(settings, repo, lambda r: ident_json(["Pa", "Pz"]))
    run = run_identity(ctx, sid, "cam1")
    assert run.status == "ok" and run.camera == "cam1"
    assert [p["participant_id"] for p in run.output["participants"]] == ["Pa", "Pz"]
    assert run.upstream_run_ids == [prep.id]
    req = backend.calls[0]
    assert len(req.image_paths) == 3
    assert "縞模様の服" in req.user_text and "左端" in req.user_text
    assert "右奥" not in req.user_text  # other camera's position
    for t in (frames[0].video_time, frames[4].video_time, frames[8].video_time):
        assert t in req.user_text


def test_identity_requires_all_subjects(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 4)
    ctx, _ = ctx_for(settings, repo, lambda r: ident_json(["Pa"]))
    run = run_identity(ctx, sid, "cam1")
    assert run.status == "failed" and "Pz" in run.error and run.output is None


def test_identity_rejects_extra_id(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 4)
    ctx, _ = ctx_for(settings, repo, lambda r: ident_json(["Pa", "Pz", "Px"]))
    run = run_identity(ctx, sid, "cam1")
    assert run.status == "failed" and "Px" in run.error


def test_identity_without_prepare_fails(settings, env):
    repo, sid, _ = env
    ctx, backend = ctx_for(settings, repo, lambda r: ident_json(["Pa", "Pz"]))
    run = run_identity(ctx, sid, "cam1")
    assert run.status == "failed" and run.error and backend.calls == []


# --- behavior ---

def behavior_responder(req):
    return behavior_json([ev(t) for t in times_in(req)[:1]])


def test_behavior_chunks_of_five(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 12)
    add_identity(repo, sid, "cam1")
    ctx, backend = ctx_for(settings, repo, behavior_responder)
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "ok"
    assert [len(c.image_paths) for c in backend.calls] == [5, 5, 2]
    assert len(run.output["events"]) == 3


def test_behavior_merges_in_time_order_with_ids_and_elapsed(settings, env, tmp_path):
    repo, sid, _ = env
    frames, prep = add_prepare(env, tmp_path, "cam1", 7)
    ident = add_identity(repo, sid, "cam1")

    def fn(req):
        ts = times_in(req)
        return behavior_json([ev(ts[-1], "Pz"), ev(ts[0], "Pa")])  # reversed in the chunk

    ctx, _ = ctx_for(settings, repo, fn)
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "ok"
    assert run.upstream_run_ids == [ident.id, prep.id]
    evs = run.output["events"]
    assert [e["event_id"] for e in evs] == ["cam1-0001", "cam1-0002", "cam1-0003", "cam1-0004"]
    assert [e["video_time"] for e in evs] == [frames[0].video_time, frames[4].video_time,
                                              frames[5].video_time, frames[6].video_time]
    assert [e["task_elapsed"] for e in evs] == ["00:00", "00:20", "00:25", "00:30"]
    assert all(e["camera"] == "cam1" for e in evs)
    # the paper engine: no tracking-engine side maps or provenance; its own prompt's hash
    assert set(run.output) == {"events"}
    assert run.prompt_hash == load_prompt(settings, Kind.video, Stage.behavior).hash


def test_behavior_rejects_unknown_participant(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 3)
    add_identity(repo, sid, "cam1")
    ctx, _ = ctx_for(settings, repo, lambda r: behavior_json([ev(times_in(r)[0], "Px")]))
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "failed" and "Px" in run.error and run.output is None


def test_behavior_time_must_match_chunk_frame(settings, env, tmp_path):
    repo, sid, _ = env
    frames, _ = add_prepare(env, tmp_path, "cam1", 7)
    add_identity(repo, sid, "cam1")
    # second chunk's model answers with a time from the first chunk
    def fn(req):
        return behavior_json([ev(frames[0].video_time)])
    ctx, _ = ctx_for(settings, repo, fn)
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "failed" and frames[0].video_time in run.error


def test_behavior_failed_chunk_fails_run_but_keeps_cache(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 7)
    add_identity(repo, sid, "cam1")

    def bad_second(req):
        ts = times_in(req)
        who = "Px" if ts[0] == "00:35" else "Pa"
        return behavior_json([ev(ts[0], who)])

    ctx, _ = ctx_for(settings, repo, bad_second)
    assert run_behavior(ctx, sid, "cam1").status == "failed"
    ctx2, backend2 = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    again = run_behavior(ctx2, sid, "cam1")  # every attempt of every chunk is cached
    assert again.status == "failed" and backend2.calls == []
    assert again.version == 2


def test_behavior_without_identity_fails(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 3)
    ctx, backend = ctx_for(settings, repo, behavior_responder)
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "failed" and backend.calls == []


def test_transcript_slice_attached_per_chunk(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 7)  # chunk1 10-35 s, chunk2 35-45 s
    add_identity(repo, sid, "cam1")
    tr = save(repo, sid, Stage.transcribe, {"segments": [
        {"start": 12.0, "end": 14.0, "text": "前半の発言"},
        {"start": 36.0, "end": 38.0, "text": "後半の発言"}]})
    ctx, backend = ctx_for(settings, repo, behavior_responder)
    run = run_behavior(ctx, sid, "cam1")
    assert tr.id in run.upstream_run_ids
    first, second = backend.calls[0].user_text, backend.calls[1].user_text
    assert "前半の発言" in first and "後半の発言" not in first
    assert "後半の発言" in second and "前半の発言" not in second
    assert "cam1" in first


def test_behavior_without_transcript_still_runs(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 2)
    add_identity(repo, sid, "cam1")
    ctx, _ = ctx_for(settings, repo, behavior_responder)
    assert run_behavior(ctx, sid, "cam1").status == "ok"


def test_behavior_yield_propagates(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 2)
    add_identity(repo, sid, "cam1")
    ctx, _ = ctx_for(settings, repo, behavior_responder, yield_check=lambda: True)
    with pytest.raises(YieldRequested):
        run_behavior(ctx, sid, "cam1")


# --- contribution ---

def test_contribution_chunks_of_sixty_and_merges(settings, env):
    repo, sid, _ = env
    beh = events_run(repo, sid, "cam1", [ev("00:10") for _ in range(130)])
    ids = [e["event_id"] for e in beh.output["events"]]

    def fn(req):
        return judg(re.findall(r"ref_id: (\S+)", req.user_text), {"cam1-0001"})

    ctx, backend = ctx_for(settings, repo, fn)
    run = run_contribution(ctx, sid, "cam1")
    assert run.status == "ok" and run.upstream_run_ids == [beh.id]
    assert len(backend.calls) == 3
    assert [len(re.findall(r"ref_id:", c.user_text)) for c in backend.calls] == [60, 60, 10]
    assert [j["ref_id"] for j in run.output["judgments"]] == ids
    assert "紙で塔を作る課題" in backend.calls[0].user_text
    assert "ルーブリック項目Q" in backend.calls[0].user_text


def test_contribution_rejects_misaligned(settings, env):
    repo, sid, _ = env
    events_run(repo, sid, "cam1", [ev("00:10"), ev("00:15")])
    ctx, _ = ctx_for(settings, repo, lambda r: judg(["cam1-0001"]))
    run = run_contribution(ctx, sid, "cam1")
    assert run.status == "failed" and "cam1-0002" in run.error


def test_contribution_chunk_cannot_judge_other_chunks_events(settings, env):
    repo, sid, _ = env
    events_run(repo, sid, "cam1", [ev("00:10") for _ in range(61)])
    ctx, _ = ctx_for(settings, repo, lambda r: judg([f"cam1-{n:04d}" for n in range(1, 62)]))
    assert run_contribution(ctx, sid, "cam1").status == "failed"


def test_contribution_no_events_no_model_call(settings, env):
    repo, sid, _ = env
    events_run(repo, sid, "cam1", [])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_contribution(ctx, sid, "cam1")
    assert run.status == "ok" and run.output == {"judgments": []} and backend.calls == []


def test_contribution_without_behavior_fails(settings, env):
    repo, sid, _ = env
    ctx, backend = ctx_for(settings, repo, lambda r: judg([]))
    run = run_contribution(ctx, sid, "cam1")
    assert run.status == "failed" and backend.calls == []


# --- pfb ---

def prepare_cameras(repo, sid, cams=("cam1", "cam2")):
    for c in cams:
        save(repo, sid, Stage.prepare, {"frames": []}, camera=c)


def setup_two_cameras(repo, sid):
    prepare_cameras(repo, sid)
    e1 = events_run(repo, sid, "cam1", [ev("00:10", "Pa", "Aが紙を折る"),
                                        ev("00:15", "Pz", "Zが部品を配置する")])
    e2 = events_run(repo, sid, "cam2", [ev("00:10", "Pa", "Aが塔を支える"),
                                        ev("00:15", "Pz", "Zが位置を調整する")])
    c1 = save(repo, sid, Stage.contribution, {"judgments": json.loads(
        judg(["cam1-0001", "cam1-0002"], {"cam1-0001", "cam1-0002"}))["judgments"]},
        camera="cam1", upstream=[e1.id])
    c2 = save(repo, sid, Stage.contribution, {"judgments": json.loads(
        judg(["cam2-0001", "cam2-0002"], {"cam2-0001", "cam2-0002"}))["judgments"]},
        camera="cam2", upstream=[e2.id])
    return c1, c2


def test_pfb_merges_cameras(settings, env):
    repo, sid, subj = env
    c1, c2 = setup_two_cameras(repo, sid)
    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0001", "cam2-0001"]))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "ok" and run.subject_id == subj["Pa"]
    assert run.upstream_run_ids == [c1.id, c2.id]
    text = backend.calls[0].user_text
    assert "Aが紙を折る" in text and "Aが塔を支える" in text
    assert run.output["evidence_refs"] == ["cam1-0001", "cam2-0001"]
    assert repo.latest_run(sid, Stage.pfb, subject_id=subj["Pa"]).id == run.id


def test_pfb_excludes_other_participants(settings, env):
    repo, sid, subj = env
    setup_two_cameras(repo, sid)
    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0002"]))
    run = run_pfb(ctx, sid, subj["Pz"])
    assert run.status == "ok"
    text = backend.calls[0].user_text
    assert "Zが部品を配置する" in text and "Zが位置を調整する" in text
    assert "Aが紙を折る" not in text and "Aが塔を支える" not in text
    assert "cam1-0001" not in text
    ctx2, _ = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0001"]))
    ctx2.backend.model_id = "fake2"  # avoid the call_log cache hit from the previous call
    bad = run_pfb(ctx2, sid, subj["Pz"])
    assert bad.status == "failed" and "cam1-0001" in bad.error


def test_pfb_insufficient_without_model_call(settings, env):
    repo, sid, subj = env
    prepare_cameras(repo, sid)
    e1 = events_run(repo, sid, "cam1", [ev("00:10")])
    e2 = events_run(repo, sid, "cam2", [ev("00:10", "Pz")])
    save(repo, sid, Stage.contribution, {"judgments": json.loads(judg(["cam1-0001"]))["judgments"]},
         camera="cam1", upstream=[e1.id])
    save(repo, sid, Stage.contribution, {"judgments": json.loads(judg(["cam2-0001"], {"cam2-0001"}))["judgments"]},
         camera="cam2", upstream=[e2.id])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert backend.calls == [] and run.status == "ok" and run.model_id is None
    assert run.output["pfb_status"] == "insufficient_positive_evidence"


def test_pfb_text_warnings_use_ids_and_cameras(settings, env):
    repo, sid, subj = env
    setup_two_cameras(repo, sid)
    bad = json.dumps({"pfb_status": "generated", "pfb": "Pa は cam1 で赤い紙を折っていた",
                      "evidence_refs": ["cam1-0001"]}, ensure_ascii=False)
    ctx, _ = ctx_for(settings, repo, lambda r: bad)
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "ok"
    joined = "\n".join(run.warnings)
    assert "Pa" in joined and "cam1" in joined and "赤" in joined


def test_pfb_missing_contribution_run_fails_gracefully(settings, env):
    repo, sid, subj = env
    prepare_cameras(repo, sid)
    e1 = events_run(repo, sid, "cam1", [ev("00:10")])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[e1.id])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_pfb(ctx, sid, subj["Pa"])  # cam2 has no contribution run
    assert run.status == "failed" and "cam2" in run.error and backend.calls == []


def test_pfb_failed_behavior_run_fails_gracefully(settings, env):
    repo, sid, subj = env
    c1, c2 = setup_two_cameras(repo, sid)
    save(repo, sid, Stage.behavior, None, camera="cam2", status="failed")
    # re-point cam2's contribution at the failed behavior run
    bad = repo.latest_run(sid, Stage.behavior, "cam2")
    save(repo, sid, Stage.contribution, c2.output, camera="cam2", upstream=[bad.id])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "failed" and run.error and backend.calls == []


def test_pfb_dangling_ref_id_fails_gracefully(settings, env):
    repo, sid, subj = env
    prepare_cameras(repo, sid)
    e1 = events_run(repo, sid, "cam1", [ev("00:10")])
    e2 = events_run(repo, sid, "cam2", [ev("00:10")])
    save(repo, sid, Stage.contribution, {"judgments": json.loads(judg(["cam1-0001"], {"cam1-0001"}))["judgments"]},
         camera="cam1", upstream=[e1.id])
    save(repo, sid, Stage.contribution, {"judgments": json.loads(judg(["cam2-0009"], {"cam2-0009"}))["judgments"]},
         camera="cam2", upstream=[e2.id])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "failed" and "cam2-0009" in run.error and backend.calls == []


def test_pfb_uses_cameras_from_prepare_runs_not_subject_positions(settings, env):
    repo, sid, _ = env
    s3 = repo.add_subject(sid, "Pq", "Q子", None, {})  # no positions recorded at all
    prepare_cameras(repo, sid, ("cam3",))
    e = events_run(repo, sid, "cam3", [ev("00:10", "Pq", "Qが紙を折る")])
    c = save(repo, sid, Stage.contribution, {"judgments": json.loads(
        judg(["cam3-0001"], {"cam3-0001"}))["judgments"]}, camera="cam3", upstream=[e.id])
    # cam1/cam2 are in the subjects' positions but have no prepare run: they are not required
    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam3-0001"]))
    run = run_pfb(ctx, sid, s3)
    assert run.status == "ok" and run.upstream_run_ids == [c.id]
    assert "Qが紙を折る" in backend.calls[0].user_text


def test_pfb_without_any_camera_fails(settings, env):
    repo, sid, subj = env
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "failed" and run.error and backend.calls == []


def test_behavior_warns_when_transcript_run_failed(settings, env, tmp_path):
    repo, sid, _ = env
    add_prepare(env, tmp_path, "cam1", 2)
    add_identity(repo, sid, "cam1")
    save(repo, sid, Stage.transcribe, None, status="failed")
    ctx, backend = ctx_for(settings, repo, behavior_responder)
    run = run_behavior(ctx, sid, "cam1")
    assert run.status == "ok"
    assert run.warnings == ["文字起こしが失敗しているため、文字起こしなしで処理しました"]
    assert NO_TRANSCRIPT_TEXT in backend.calls[0].user_text


# --- sports: prompt variants and the game clock ---

from pfb_app.pipeline.video import _event_lines  # noqa: E402
from pfb_app.domain.models import BehaviorEvent  # noqa: E402


@pytest.fixture
def sports_env(tmp_path, settings):
    conn = connect(tmp_path / "sports.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("練習試合", Kind.video, "春季リーグ第2戦", "", "standard", False,
                                 domain="sports", sport="soccer")
    sid = repo.create_submission(aid, "G1", {"clock_offset_min": 0.0})   # a game clock the coach gave
    s_a = repo.add_subject(sid, "Pa", "#5", "", {})
    return repo, sid, {"Pa": s_a}


def sports_events_run(repo, sid, camera, events):
    stored = [{**e, "event_id": f"{camera}-{n:04d}", "camera": camera, "task_elapsed": "10:05"}
              for n, e in enumerate(events, 1)]
    return save(repo, sid, Stage.behavior, {"events": stored}, camera=camera)


def test_sports_event_lines_use_game_clock():
    e = BehaviorEvent(event_id="cam1-0001", camera="cam1", participant="Pa", video_time="00:42",
                      task_elapsed="50:30", observed_behavior="味方にパスを出す", utterance="",
                      speaker="unknown", evidence_source="video", confidence="high")
    sports = _event_lines([e], "sports")
    assert "試合時間: 50:30（51分ごろ）" in sports
    assert "00:42" not in sports and "時刻" not in sports
    for clock, minute in (("65:20", 65), ("65:40", 66), ("00:10", 0), ("1:05:20", 65)):
        assert f"（{minute}分ごろ）" in _event_lines([e.model_copy(update={"task_elapsed": clock})],
                                                  "sports"), clock
    school = _event_lines([e])
    assert school == _event_lines([e], "school")
    assert "時刻: 00:42" in school and "試合時間" not in school


def test_sports_contribution_and_pfb_use_the_sports_prompts(settings, sports_env):
    repo, sid, subj = sports_env
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    sports_events_run(repo, sid, "cam1", [ev("00:42", "Pa", "味方にパスを出す")])
    ctx, backend = ctx_for(settings, repo, lambda r: judg(["cam1-0001"], {"cam1-0001"}))
    run = run_contribution(ctx, sid, "cam1")
    sports_prompt = load_prompt(settings, Kind.video, Stage.contribution, domain="sports")
    assert run.status == "ok" and run.prompt_hash == sports_prompt.hash
    req = backend.calls[0]
    assert req.system == sports_prompt.text
    assert "【競技・データ開始】\nサッカー\n【競技・データ終了】" in req.user_text
    assert "春季リーグ第2戦" in req.user_text and "試合時間: 10:05（10分ごろ）" in req.user_text
    assert "【見てほしい観点・データ開始】" in req.user_text and "ルーブリック" not in req.user_text

    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0001"]))
    run = run_pfb(ctx, sid, subj["Pa"])
    pfb_prompt = load_prompt(settings, Kind.video, Stage.pfb, domain="sports")
    assert run.status == "ok" and run.prompt_hash == pfb_prompt.hash
    req = backend.calls[0]
    assert req.system == pfb_prompt.text
    assert "【競技・データ開始】\nサッカー\n【競技・データ終了】" in req.user_text
    assert "春季リーグ第2戦" in req.user_text and "試合時間: 10:05" in req.user_text


def test_school_contribution_keeps_the_school_prompt_and_sections(settings, env):
    repo, sid, _ = env
    events_run(repo, sid, "cam1", [ev("00:10")])
    ctx, backend = ctx_for(settings, repo, lambda r: judg(["cam1-0001"]))
    run = run_contribution(ctx, sid, "cam1")
    assert run.prompt_hash == load_prompt(settings, Kind.video, Stage.contribution).hash
    text = backend.calls[0].user_text
    assert "競技" not in text and "時刻: 00:10" in text and "【活動の内容・データ開始】" in text
    assert "【ルーブリック・データ開始】" in text and "見てほしい観点" not in text


def test_cited_evidence_when_sports(settings, sports_env, env):
    from pfb_app.ui.common import cited_evidence
    for (repo, sid, subj), when in ((sports_env, "試合時間 10:05"), (env, "00:42")):
        save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
        beh = (sports_events_run if when.startswith("試合") else events_run)(
            repo, sid, "cam1", [ev("00:42", "Pa", "味方にパスを出す")])
        con = save(repo, sid, Stage.contribution, {"judgments": json.loads(
            judg(["cam1-0001"], {"cam1-0001"}))["judgments"]}, camera="cam1", upstream=[beh.id])
        pfb = save(repo, sid, Stage.pfb, json.loads(pfb_json(["cam1-0001"])),
                   subject_id=subj["Pa"], upstream=[con.id])
        assert [i.when for i in cited_evidence(repo, pfb)] == [when]


# --- activity-type profiles: prompts, sections and the category of each moment ---

def test_meeting_contribution_and_pfb_use_the_profile(settings, env):
    repo, sid, subj = env
    aid = repo.get_submission(sid)["assignment_id"]
    repo.conn.execute("UPDATE assignments SET profile='meeting' WHERE id=?", (aid,))
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    events_run(repo, sid, "cam1", [ev("00:10", "Pa", "ボードに意見を書き出す")])

    def judged(req):
        return json.dumps({"judgments": [{
            "ref_id": "cam1-0001", "contribution_interpretation": "意見を整理した",
            "is_positive_contribution": True, "rubric_criteria": ["ルーブリック項目Q", "まとめ"]}]},
            ensure_ascii=False)

    ctx, backend = ctx_for(settings, repo, judged)
    run = run_contribution(ctx, sid, "cam1")
    prompt = load_prompt(settings, Kind.video, Stage.contribution, "school", "meeting")
    assert run.status == "ok" and run.prompt_hash == prompt.hash
    text = backend.calls[0].user_text
    assert backend.calls[0].system == prompt.text
    assert "【活動の種類・データ開始】\n話し合い・会議\n【活動の種類・データ終了】" in text
    assert "【見る観点・データ開始】\n発言\n傾聴\nまとめ\n提案\n合意づくり\n時間管理\n" in text
    assert "【評価の観点・データ開始】\nルーブリック項目Q" in text and "【ルーブリック" not in text

    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0001"]))
    run = run_pfb(ctx, sid, subj["Pa"])
    pfb_prompt = load_prompt(settings, Kind.video, Stage.pfb, "school", "meeting")
    assert run.status == "ok" and run.prompt_hash == pfb_prompt.hash
    text = backend.calls[0].user_text
    assert "当てはまる観点: まとめ" in text and "【見る観点・データ開始】" in text


def test_general_pfb_input_is_unchanged(settings, env):
    """一般 (and every work from before profiles): no category line, no new sections."""
    repo, sid, subj = env
    setup_two_cameras(repo, sid)
    ctx, backend = ctx_for(settings, repo, lambda r: pfb_json(["cam1-0001"]))
    run_pfb(ctx, sid, subj["Pa"])
    text = backend.calls[0].user_text
    assert "当てはまる観点" not in text and "活動の種類" not in text and "見る観点" not in text


# --- events on someone else (ID swap, the names stage's number timeline) ---

def tracked_behavior(repo, sid, tmp_path, events, segments, dominant):
    d = tmp_path / "trk"
    d.mkdir()
    (d / "boxes.txt").write_text("1,1,0,0,10,10,0.9\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"fps": 10.0, "n_frames": 1000}))
    track = save(repo, sid, Stage.track, {"labels": ["P1", "P2"], "boxes": str(d / "boxes.txt")},
                 camera="cam1")
    ident = save(repo, sid, Stage.identity, {"track_run_id": track.id, "labels": ["P1", "P2"],
                                             "participants": [
        {"participant_id": "Pa", "label": "P1", "display_name": None},
        {"participant_id": "Pz", "label": "P2", "display_name": None}]},
        camera="cam1", upstream=[track.id])
    save(repo, sid, Stage.names, {"track_run_id": track.id, "cues": {"number": {
        "P1": {"number": None, "frames_agree": 0, "timeline_s": 6.0, "segments": segments,
               "dominant": dominant, "dominant_share": 0.7}}}}, camera="cam1",
         upstream=[track.id])
    stored = [{**e, "event_id": f"cam1-{n:04d}", "camera": "cam1", "task_elapsed": "00:00"}
              for n, e in enumerate(events, 1)]
    return save(repo, sid, Stage.behavior, {"events": stored}, camera="cam1",
                upstream=[ident.id, track.id])


def test_contribution_leaves_out_moments_on_another_player(settings, env, tmp_path):
    repo, sid, _ = env
    # P1 (Pa) is #1 from 00:30 on; before that its box was on #5 (frames 1..200 = 0-20 s)
    tracked_behavior(repo, sid, tmp_path,
                     [ev("00:05"), ev("00:40"), ev("00:06", who="Pz")],
                     [[1, 200, "5", 4], [300, 900, "1", 12]], "1")
    seen = []

    def fn(req):
        ids = re.findall(r"ref_id: (\S+)", req.user_text)
        seen.extend(ids)
        return judg(ids, set(ids))

    ctx, _ = ctx_for(settings, repo, fn)
    run = run_contribution(ctx, sid, "cam1")
    assert run.status == "ok"
    assert seen == ["cam1-0002", "cam1-0003"]
    assert [j["ref_id"] for j in run.output["judgments"]] == ["cam1-0002", "cam1-0003"]
    assert any("別の人" in w for w in run.warnings)


def test_contribution_keeps_everything_without_a_timeline(settings, env, tmp_path):
    repo, sid, _ = env
    tracked_behavior(repo, sid, tmp_path, [ev("00:05"), ev("00:40")], [], None)
    ctx, _ = ctx_for(settings, repo, lambda r: judg(re.findall(r"ref_id: (\S+)", r.user_text)))
    run = run_contribution(ctx, sid, "cam1")
    assert [j["ref_id"] for j in run.output["judgments"]] == ["cam1-0001", "cam1-0002"]
