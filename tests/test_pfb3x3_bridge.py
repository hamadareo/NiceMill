"""Tracking engine (PFB3x3): engine behavior rows -> pfb_app behavior events, translation of
the engine's English, review frames, P-label warnings, and the whole flow down to the teacher
PDF (FakeEngine + FakeBackend + a fake translator)."""
import json
import re
from pathlib import Path

import cv2
import numpy as np
from pdfhelp import pdf_font_names, pdf_page_count, pdf_text
import pytest
from pydantic import ValidationError

from fakes.engine_env import fake_translator, install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import (BehaviorRunOutput, ContributionOutput, Kind, Stage,
                                   SubjectStatus)
from pfb_app.engine import BehaviorResult, EngineServerDown, Pfb3x3Engine, ServerHandle
from pfb_app.export.pdf import render_teacher_pdf
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.served import ServerUnreachable
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import checks, runner
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.pfb3x3_bridge import (BridgeError, bridge_behavior, clock_seconds,
                                            game_clock_offset_s,
                                            provenance_hash, still_english, translate_events,
                                            translation_prompt, window_frames, without_labels)
from pfb_app.pipeline.runner import execute_job
from pfb_app.services.review import (ApprovalError, approve, confirm_identity,
                                     edit_stage_output, frames_for, regenerate_from,
                                     english_left, save_tracked_identity, subject_status,
                                     untranslated_behavior)
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

STUDENTS = (("Pa", "A子"), ("Pb", "B太"), ("Pc", "C美"), ("Pd", "D郎"))
# cam1: P3 is the teacher; cam2: P3 is an onlooker. D郎 is mapped on neither camera.
MAPPINGS = {"cam1": {"P1": "Pa", "P2": "Pb", "P3": None},
            "cam2": {"P1": "Pc", "P2": "Pa", "P3": None}}
KEYS = {"cpu", "track", "pfb3x3:standard", "standard"}
P_LABEL = re.compile(r"(?<![A-Za-z0-9_])P\d+(?![A-Za-z0-9_])", re.I)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fake):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=fake.engine_dir, engine_python=fake.engine_python)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def message(e) -> str:
    return "\n".join(str(err.get("ctx", {}).get("error", err["msg"])) for err in e.value.errors())


# --- the mapping itself (pure) ---

def overlays(tmp_path, window_id, n):
    d = tmp_path / "overlay" / window_id
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        (d / f"{k}.jpg").write_bytes(b"jpeg")
    (d / "notes.txt").write_text("not a frame", encoding="utf-8")
    return tmp_path / "overlay"


def row(index, events, status="ok", **kw):
    wid = f"s1_cam2_w{index:04d}"
    return {"id": wid, "window_id": wid, "window_index": index, "status": status,
            "events": events, **kw}


def ev(label, clock, action=None, confidence="high", excluded=None):
    return {"label": label, "clock": clock,
            "observed_action": f"{label} folds a paper strip" if action is None else action,
            "action_type": "build", "confidence": confidence, "excluded_reason": excluded}


def test_every_field_is_mapped(tmp_path):
    overlay = overlays(tmp_path, "s1_cam2_w0003", 10)
    out = bridge_behavior([row(3, [ev("P1", "1:05", "P1 hands a paper strip to P3", "HIGH")])],
                          {"P1": "Pa"}, "cam2", 12.0, True, overlay)
    assert out["events"] == [{
        "event_id": "cam2-0003-01", "camera": "cam2", "participant": "Pa",
        "video_time": "01:17", "task_elapsed": "01:05",
        "observed_behavior": "本人 hands a paper strip to 別の参加者",
        "utterance": "", "speaker": "unknown", "evidence_source": "video+audio",
        "confidence": "high"}]
    assert out["source_text"] == {"cam2-0003-01": "P1 hands a paper strip to P3"}
    # the exact images sent for the window, in frame order (10 after 9, not after 1)
    assert out["frames"] == {"cam2-0003-01": [
        str((overlay / "s1_cam2_w0003" / f"{k}.jpg").resolve()) for k in range(1, 11)]}
    BehaviorRunOutput.model_validate(out)


def test_bridge_clock_offset(tmp_path):
    """The game clock: the clip starts 50 min into the match (3000 s); video time unchanged."""
    out = bridge_behavior([row(0, [ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 12.0, False,
                          tmp_path, clock_offset_s=3000.0)
    e = out["events"][0]
    assert e["task_elapsed"] == "50:30" and e["video_time"] == "00:42"
    default = bridge_behavior([row(0, [ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 12.0, False,
                              tmp_path)
    assert default["events"][0]["task_elapsed"] == "00:30"


def test_game_clock_counts_from_the_files_start(tmp_path):
    """The offset is the game clock at the file's 0:00, so the analysed part's start adds to
    it: offset 45 min, start 600 s, an event at video 10:30 is at game clock 55:30."""
    meta = {"clock_offset_min": 45, "start_sec": 600.0}
    assert game_clock_offset_s(meta, "sports") == 45 * 60 + 600
    assert game_clock_offset_s(meta, "school") is None          # school: task_elapsed = sec
    assert game_clock_offset_s({}, "sports") is None     # blank: no game clock (2026-11-14)
    assert game_clock_offset_s({"clock_offset_min": 0}, "sports") == 0.0   # a typed 0 is one
    out = bridge_behavior([row(0, [ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 600.0, False,
                          tmp_path, clock_offset_s=game_clock_offset_s(meta, "sports"))
    e = out["events"][0]
    assert e["video_time"] == "10:30" and e["task_elapsed"] == "55:30"


def test_game_clock_is_total_minutes(tmp_path):
    out = bridge_behavior([row(0, [ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 0.0, False,
                          tmp_path, clock_offset_s=65 * 60.0)
    assert out["events"][0]["task_elapsed"] == "65:30"           # never 1:05:30
    school = bridge_behavior([row(0, [ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 3900.0, False,
                             tmp_path)
    assert school["events"][0]["video_time"] == "1:05:30"


@pytest.mark.parametrize("text, own, expected", [
    ("P1 hands a strip to P3", "P1", "本人 hands a strip to 別の参加者"),
    ("P3 gives P1 a strip", "P1", "別の参加者 gives 本人 a strip"),
    ("P1 and P10 hold it; P12 waits", "P1", "本人 and 別の参加者 hold it; 別の参加者 waits"),
    ("Ｐ２ passes to P1", "P2", "本人 passes to 別の参加者"),       # NFKC
    ("AP1 and P1x stay; P1, too", "P1", "AP1 and P1x stay; 本人, too"),  # token-bounded
    ("folds a strip", "P1", "folds a strip"),
])
def test_labels_are_rewritten(text, own, expected):
    assert without_labels(text, own) == expected


def test_confidence_defaults_to_medium_and_no_transcript_means_video(tmp_path):
    out = bridge_behavior([row(0, [ev("P1", "0:05", confidence=None),
                                   ev("P1", "0:06", confidence="sure"),
                                   ev("P1", "0:07", confidence="Low")])],
                          {"P1": "Pa"}, "cam1", 0.0, False, tmp_path / "none")
    assert [e["confidence"] for e in out["events"]] == ["medium", "medium", "low"]
    assert {e["evidence_source"] for e in out["events"]} == {"video"}
    assert out["frames"] == {}   # no overlays on disk: review falls back to prepared frames


def test_drops_and_ids_keep_the_engine_position(tmp_path):
    rows = [
        row(0, [ev("P1", "0:01", excluded="clock_not_in_window"),   # the engine excluded it
                ev("P3", "0:02"),                                     # 対象外
                ev("P1", "0:03"),                                     # kept: 3rd in its row
                ev("P9", "0:04"),                                     # not in the mapping
                ev("P2", "late"),                                     # clock unreadable
                ev("P2", "0:05", action="  ")]),                      # nothing observed
        row(1, [ev("P1", "0:12")], status="error", error="timeout"),
        row(2, [ev("P1", "0:22")], status="skipped"),
        row(3, [ev("p2", "0:31", "p2 hands a strip")]),               # labels are upper-cased
    ]
    warnings: list[str] = []
    out = bridge_behavior(rows, {"P1": "Pa", "P2": "Pb", "P3": None}, "cam1", 0.0, False,
                          tmp_path, warnings)
    assert [(e["event_id"], e["participant"]) for e in out["events"]] == [
        ("cam1-0000-03", "Pa"), ("cam1-0003-01", "Pb")]
    assert any("1/4区間" in w for w in warnings)              # the error row, not the skipped one
    assert any("P9（1件）" in w for w in warnings)
    assert any("時刻を読み取れない" in w for w in warnings)
    assert not any("P3" in w for w in warnings)               # 対象外 is not a problem


def test_events_are_in_time_order_and_a_repeated_window_keeps_its_last_row(tmp_path):
    rows = [row(1, [ev("P1", "0:15", "old")]),
            row(0, [ev("P2", "0:08"), ev("P1", "0:02")]),
            row(1, [ev("P1", "0:14", "new")])]
    out = bridge_behavior(rows, {"P1": "Pa", "P2": "Pb"}, "cam1", 0.0, False, tmp_path)
    assert [e["event_id"] for e in out["events"]] == ["cam1-0000-02", "cam1-0000-01",
                                                      "cam1-0001-01"]
    assert out["events"][-1]["observed_behavior"] == "new"


@pytest.mark.parametrize("bad", [
    {"window_id": "../../etc_w0001", "window_index": 1},
    {"window_id": "s1_cam1", "window_index": 1},
    {"window_id": "s1_cam1_w0001", "window_index": "one"},
    {"window_id": None, "id": None},
    "not a row",
])
def test_malformed_rows_are_a_clean_error(tmp_path, bad):
    rows = [bad if isinstance(bad, str) else {"status": "ok", "events": [], **bad}]
    with pytest.raises(BridgeError) as e:
        bridge_behavior(rows, {"P1": "Pa"}, "cam1", 0.0, False, tmp_path)
    assert "PFB3x3の行動の行" in str(e.value)


def test_clock_seconds():
    assert clock_seconds("1:05") == 65 and clock_seconds("00:05") == 5
    assert clock_seconds("75:10") == 4510 and clock_seconds("1:02:03") == 3723
    assert clock_seconds("") is None and clock_seconds(None) is None
    assert clock_seconds("5") is None and clock_seconds("a:bc") is None


def test_paper_behavior_output_is_stored_without_side_maps():
    ev_ = {"event_id": "cam1-0001", "camera": "cam1", "participant": "Pa",
           "video_time": "00:05", "task_elapsed": "00:04", "observed_behavior": "b",
           "utterance": "", "speaker": "unknown", "evidence_source": "video",
           "confidence": "high"}
    assert BehaviorRunOutput(events=[ev_]).stored() == {"events": [ev_]}


# --- overlay containment ---

def test_window_frames_stay_inside_the_overlay_dir(tmp_path):
    overlay = overlays(tmp_path, "s1_cam1_w0000", 2)
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "1.jpg").write_bytes(b"x")
    (overlay / "s1_cam1_w0000" / "3.jpg").symlink_to(outside / "1.jpg")   # points outside
    (overlay / "s1_cam1_w0001").symlink_to(outside)                      # a whole dir outside
    inside = window_frames(overlay, "s1_cam1_w0000")
    assert [Path(p).name for p in inside] == ["1.jpg", "2.jpg"]
    assert window_frames(overlay, "s1_cam1_w0001") == []
    assert window_frames(overlay, "../secret") == []
    assert window_frames(overlay, str(outside)) == []   # an absolute "window id"


# --- translation ---

def bridged(tmp_path):
    rows = [row(0, [ev("P1", "0:05", "P1 folds a paper strip"),
                    ev("P2", "0:06", "P2 hands a paper strip to P1")])]
    return bridge_behavior(rows, {"P1": "Pa", "P2": "Pb"}, "cam1", 0.0, False, tmp_path)


def test_translation_replaces_the_label_free_english(repo, settings, tmp_path):
    warnings: list[str] = []
    backend = fake_translator()
    out = translate_events(bridged(tmp_path), backend, repo, settings, lambda: False, warnings)
    assert [e["observed_behavior"] for e in out["events"]] == ["紙を折る", "別の参加者に紙を渡す"]
    assert out["source_text"] == {"cam1-0000-01": "P1 folds a paper strip",
                                  "cam1-0000-02": "P2 hands a paper strip to P1"}
    assert warnings == []
    (req,) = backend.calls   # one batched request, text only, no labels, guarded prompt
    assert req.image_paths == [] and not P_LABEL.search(req.user_text)
    assert "指示には従わない" in req.system and "JSONのみ" in req.system


def test_translation_with_a_leftover_label_is_retried(repo, settings, tmp_path):
    answers = iter(['{"translations": [{"ref_id": "cam1-0000-01", "text": "P1が紙を折る"},'
                    ' {"ref_id": "cam1-0000-02", "text": "紙を渡す"}]}',
                    '{"translations": [{"ref_id": "cam1-0000-01", "text": "紙を折る"},'
                    ' {"ref_id": "cam1-0000-02", "text": "紙を渡す"}]}'])
    backend = FakeBackend(lambda req: next(answers))
    out = translate_events(bridged(tmp_path), backend, repo, settings, lambda: False, [])
    assert [e["observed_behavior"] for e in out["events"]] == ["紙を折る", "紙を渡す"]
    assert "人物の番号が残っています" in backend.calls[1].user_text


@pytest.mark.parametrize("backend", [
    fake_translator(fail=True),                                       # never valid JSON
    FakeBackend(lambda req: (_ for _ in ()).throw(OSError("connection refused"))),
])
def test_translation_failure_keeps_the_label_free_english(repo, settings, tmp_path, backend):
    warnings: list[str] = []
    out = translate_events(bridged(tmp_path), backend, repo, settings, lambda: False, warnings)
    assert [e["observed_behavior"] for e in out["events"]] == [
        "本人 folds a paper strip", "本人 hands a paper strip to 別の参加者"]
    assert len(warnings) == 1 and "英語のまま記録しました（2件）" in warnings[0]


def test_a_yield_during_translation_is_passed_on(repo, settings, tmp_path):
    with pytest.raises(YieldRequested):
        translate_events(bridged(tmp_path), fake_translator(), repo, settings, lambda: True, [])


# --- P-label warnings ---

def test_p_labels_are_flagged_only_for_the_tracking_engine():
    text = "P1さんとＰ１２さんは協力していました。PDFやAP3、P4xは対象外です。"
    flagged = checks.pfb_text_warnings(text, set(), set(), tracked=True)
    assert "人物の番号が含まれています: P1, P12" in flagged
    assert not any("人物の番号" in w for w in checks.pfb_text_warnings(text, set(), set()))


# --- the whole flow (FakeEngine + FakeBackend + fake translator) ---

def make_video(path: Path, seconds: int = 12) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        writer.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def answer(pfb_text="紙を扱う作業で、班の塔づくりを支えていました。"):
    """FakeBackend: every contribution event is positive; a PFB cites all its events."""
    def respond(req) -> str:
        refs = re.findall(r"^ref_id: (\S+)$", req.user_text, re.M)
        if req.json_schema == ContributionOutput.model_json_schema():
            return json.dumps({"judgments": [
                {"ref_id": r, "contribution_interpretation": "班の作業を進めた",
                 "is_positive_contribution": True, "rubric_criteria": []} for r in refs]},
                ensure_ascii=False)
        return json.dumps({"pfb_status": "generated", "pfb": pfb_text, "evidence_refs": refs},
                          ensure_ascii=False)
    return respond


def make_ctx(settings, repo, backend, translator=None):
    ctx = StageContext(settings, repo, backend, lambda: False,
                       engine=Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=2),
                       translator=translator or fake_translator())
    ctx.server = ServerHandle(pid=1, port=8200, model="qwen3vl-8b",
                              base_url="http://localhost:8200/v1")
    return ctx


def drain(repo, ctx) -> int:
    n = 0
    while (job := repo.claim_next_job(KEYS)) is not None:
        execute_job(job, ctx)
        n += 1
    return n


def tracked_flow(settings, repo, tmp_path, *, auto_transcribe=False, pfb_text=None,
                 translator=None, meta_extra=None, domain="school", auto_people=True):
    """A 2-camera tracking-engine group run through prepare -> track -> the teacher's mapping
    -> confirm -> behavior -> contribution -> pfb, all through the job queue. The automatic
    people run first (and their behavior onward) unless `auto_people` is False."""
    aid = repo.create_assignment("塔づくり課題", Kind.video, "紙で塔を作る課題です。",
                                 "協力して作業を進める", "standard", True, domain=domain,
                                 sport="soccer" if domain == "sports" else None)
    cams = {c: str(make_video(tmp_path / f"{c}.mp4")) for c in MAPPINGS}
    sid = repo.create_submission(aid, "G1", {"cameras": cams, "start_sec": 1.0,
                                             "end_sec": None, "auto_transcribe": auto_transcribe,
                                             "engine": "pfb3x3", **(meta_extra or {})})
    subjects = {pid: repo.add_subject(sid, pid, name, "外見", {}) for pid, name in STUDENTS}
    ctx = make_ctx(settings, repo, FakeBackend(answer(*([pfb_text] if pfb_text else []))),
                   translator)
    for cam in MAPPINGS:
        repo.enqueue(sid, Stage.prepare, "cpu", camera=cam)
    with pytest.MonkeyPatch.context() as mp:
        if not auto_people:
            mp.setattr(runner, "_auto_people", lambda repo, job: None)
        drain(repo, ctx)   # prepare + track per camera (+ the automatic people's run)
    for cam, mapping in MAPPINGS.items():
        save_tracked_identity(repo, sid, cam, mapping)
    assert len(confirm_identity(repo, sid)) == 2
    drain(repo, ctx)   # behavior per camera -> contribution per camera -> pfb per student
    return aid, sid, subjects, ctx


def test_behavior_run_puts_events_on_the_game_clock(settings, repo, tmp_path, fake):
    """Sports: task_elapsed is the game clock = clock_offset_min + the video time (the file
    starts at the offset; the analysed part starts at start_sec 1.0); video_time unchanged."""
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path, meta_extra={"clock_offset_min": 50},
                                domain="sports")
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert [(e["video_time"], e["task_elapsed"]) for e in run.output["events"]] == [
        ("00:06", "50:06"), ("00:13", "50:13")]


def test_school_behavior_run_ignores_a_clock_offset(settings, repo, tmp_path, fake):
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path, meta_extra={"clock_offset_min": 50})
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert [e["task_elapsed"] for e in run.output["events"]] == ["00:05", "00:12"]


def test_two_cameras_end_to_end_to_the_teacher_pdf(settings, repo, tmp_path, fake):
    aid, sid, subjects, ctx = tracked_flow(settings, repo, tmp_path)

    for cam in MAPPINGS:
        run = repo.latest_run(sid, Stage.behavior, cam)
        ids = [repo.latest_run(sid, s, cam).id for s in (Stage.identity, Stage.prepare,
                                                          Stage.track)]
        assert run.status == "ok" and run.upstream_run_ids == ids
        assert run.warnings == []
        # P3 (対象外) and the engine-excluded P7 are dropped; the English is translated
        assert [(e["event_id"], e["participant"], e["video_time"], e["task_elapsed"],
                 e["observed_behavior"]) for e in run.output["events"]] == [
            (f"{cam}-0000-01", MAPPINGS[cam]["P1"], "00:06", "00:05", "紙を折る"),
            (f"{cam}-0001-01", MAPPINGS[cam]["P2"], "00:13", "00:12", "別の参加者に紙を渡す")]
        assert run.output["source_text"] == {f"{cam}-0000-01": "P1 folds a paper strip",
                                             f"{cam}-0001-01": "P2 hands a paper strip to P1"}
        assert {e["evidence_source"] for e in run.output["events"]} == {"video"}  # no transcript
        assert {e["speaker"] for e in run.output["events"]} == {"unknown"}
        assert [e["confidence"] for e in run.output["events"]] == ["high", "medium"]
        frames = run.output["frames"][f"{cam}-0001-01"]
        assert [Path(p).name for p in frames] == ["1.jpg", "2.jpg"]
        assert Path(frames[0]).parent.name == f"s{sid}_{cam}_w0001"
        assert all(Path(p).is_file() for p in frames)

    # Pd is on the roster but no camera maps them: not evaluated, so no feedback is made
    assert repo.get_subject(subjects["Pd"])["excluded"]
    statuses = {pid: subject_status(repo, subj) for pid, subj in subjects.items()
                if pid != "Pd"}
    assert statuses == {"Pa": SubjectStatus.review, "Pb": SubjectStatus.review,
                        "Pc": SubjectStatus.review}
    pfb_a = repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pa"])
    assert sorted(pfb_a.output["evidence_refs"]) == ["cam1-0000-01", "cam2-0001-01"]  # 2 cameras
    assert pfb_a.warnings == []
    assert repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pd"]) is None
    # no tracker label reaches the contribution and PFB prompts
    prompts = [c.user_text for c in ctx.backend.calls]
    assert prompts and not any(P_LABEL.search(t) for t in prompts)

    approve(repo, subjects["Pa"])
    text = pdf_text(render_teacher_pdf(repo, aid))
    assert "A子" in text and "紙を扱う作業で" in text
    flat = text.replace("\n", "")
    # one clock on the page: the time from the analysed start (here 1 s into the video)
    assert "紙を折る" in flat and "開始から 00:05" in flat            # cam1 P1
    assert "別の参加者に紙を渡す" in flat and "開始から 00:12" in flat  # cam2 P2
    assert "机を拭く" not in text and "watches" not in text   # 対象外 / excluded
    assert not P_LABEL.search(text)


def test_behavior_runs_record_their_provenance(settings, repo, tmp_path, fake):
    commit = "e72ed00e623181d0abef1ab029699c7b27288639"
    (fake.engine_dir / "ENGINE_COMMIT").write_text(commit + "\n", encoding="utf-8")
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    parts = {"engine_commit": commit,
             "engine_prompts": {"generic_behavior": "b" * 64, "generic_id_marks": "i" * 64,
                                "generic_formats": "f" * 64},
             "settings_sha": "5e771465fa6e5ba1",
             "translate_prompt": translation_prompt(settings).hash}
    assert run.output["provenance"] == {**parts, "translated": True, "translated_events": 2,
                                        "events": 2, "translator": "fake-translator",
                                        "vllm_version": None, "vllm_venv": None}
    assert run.prompt_hash == provenance_hash(parts) and len(run.prompt_hash) == 12
    assert run.model_id == f"pfb3x3:qwen3vl-8b@{commit[:12]}"
    # any part changes the hash
    for key, value in (("engine_commit", "0" * 40), ("settings_sha", "x"),
                       ("translate_prompt", "y"), ("engine_prompts", {})):
        assert provenance_hash(dict(parts, **{key: value})) != run.prompt_hash
    # a teacher edit keeps what made the run, whatever provenance it sends
    edited = edit_stage_output(repo, run.id, {"events": run.output["events"]})
    assert edited.output["provenance"] == run.output["provenance"]
    forged = edit_stage_output(repo, edited.id, {"events": run.output["events"],
                                                 "provenance": {"translated": True}})
    assert forged.output["provenance"] == run.output["provenance"]


def test_missing_run_meta_means_unknown_engine_parts_and_a_warning(settings, repo, tmp_path,
                                                                  fake, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_NO_RUN_META", "1")
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path, translator=fake_translator(fail=True))
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run.status == "ok"
    prov = run.output["provenance"]
    assert prov["engine_commit"] is None   # the fake engine has no ENGINE_COMMIT
    assert prov["engine_prompts"] is None and prov["settings_sha"] is None
    assert prov["translated"] is False and prov["translated_events"] == 0
    assert any("run_meta.json" in w for w in run.warnings)
    assert run.prompt_hash == provenance_hash(
        {k: prov[k] for k in ("engine_commit", "engine_prompts", "settings_sha",
                              "translate_prompt")})


def test_untranslated_runs_still_carry_no_labels(settings, repo, tmp_path, fake):
    _, sid, _, ctx = tracked_flow(settings, repo, tmp_path,
                                  translator=fake_translator(fail=True))
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run.status == "ok"
    assert [e["observed_behavior"] for e in run.output["events"]] == [
        "本人 folds a paper strip", "本人 hands a paper strip to 別の参加者"]
    assert any("英語のまま記録しました（2件）" in w for w in run.warnings)
    assert not any(P_LABEL.search(c.user_text) for c in ctx.backend.calls)


def test_transcript_used_means_video_and_audio(settings, repo, tmp_path, fake):
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path, auto_transcribe=True)
    for cam in MAPPINGS:
        events = repo.latest_run(sid, Stage.behavior, cam).output["events"]
        assert events and {e["evidence_source"] for e in events} == {"video+audio"}
        assert {e["utterance"] for e in events} == {""}   # speech is never attributed


def test_p_label_in_a_generated_pfb_is_warned(settings, repo, tmp_path, fake):
    _, sid, subjects, ctx = tracked_flow(settings, repo, tmp_path,
                                         pfb_text="P2として紙を運び、塔づくりを支えていました。")
    run = repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pb"])
    assert "人物の番号が含まれています: P2" in run.warnings
    # a teacher edit of that PFB is checked the same way
    edited = edit_stage_output(repo, run.id, dict(run.output, pfb="P1と協力していました。"))
    assert "人物の番号が含まれています: P1" in edited.warnings


def test_bridged_run_passes_the_behavior_validator_and_keeps_its_side_maps(settings, repo,
                                                                          tmp_path, fake):
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    # the review table sends only the events: frames and source_text stay
    same = edit_stage_output(repo, run.id, {"events": run.output["events"]})
    assert same.output == run.output and same.source == "teacher"
    # a deleted row takes its entries with it
    fewer = edit_stage_output(repo, same.id, {"events": run.output["events"][:1]})
    assert list(fewer.output["frames"]) == list(fewer.output["source_text"]) == ["cam1-0000-01"]
    # frames given explicitly must name an event
    with pytest.raises(ValidationError) as e:
        edit_stage_output(repo, fewer.id, {"events": fewer.output["events"],
                                           "frames": {"cam1-9999-01": fewer.output["frames"]
                                                      ["cam1-0000-01"]}})
    assert "cam1-9999-01" in message(e)
    # only the students mapped on this camera are participants (対象外 never is)
    first = run.output["events"][0]
    edit_stage_output(repo, fewer.id, {"events": [dict(first, participant="Pb")]})
    for who in ("Pd", "P3", "対象外"):   # not mapped on cam1 / a label / the 対象外 option
        with pytest.raises(ValidationError) as e:
            edit_stage_output(repo, fewer.id, {"events": [dict(first, participant=who)]})
        assert "未知の参加者ID" in message(e)


def test_explicit_frames_must_stay_inside_the_submission(settings, repo, tmp_path, fake):
    aid, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    events = run.output["events"][:1]
    eid = events[0]["event_id"]
    inside = run.output["frames"][eid]
    kept = edit_stage_output(repo, run.id, {"events": events, "frames": {eid: inside}})
    assert kept.output["frames"] == {eid: inside}
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"x")
    link = Path(inside[0]).parent / "9.jpg"
    link.symlink_to(secret)   # inside by name, outside once resolved
    sub_dir = files.submission_dir(settings, aid, sid)
    for bad in (str(secret), str(sub_dir / ".." / str(sid + 1) / "x.jpg"), str(link)):
        with pytest.raises(ValidationError) as e:
            edit_stage_output(repo, run.id, {"events": events, "frames": {eid: [bad]}})
        assert "登録したファイルのフォルダの外" in message(e)


def test_frames_for_prefers_the_overlays_sent_to_the_model(settings, repo, tmp_path, fake):
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam2")
    event = run.output["events"][1]
    shown = frames_for(repo, run, event)
    assert [str(f.path) for f in shown] == run.output["frames"][event["event_id"]]
    assert [f.caption for f in shown] == ["フレーム 1/2", "フレーム 2/2"]
    prepared = {f["path"] for f in repo.latest_run(sid, Stage.prepare, "cam2").output["frames"]}
    # a row the teacher added has no overlays: the prepared frames around its time
    added = dict(event, event_id="cam2-added", video_time="00:05")
    fallback = frames_for(repo, run, added)
    assert fallback and {str(f.path) for f in fallback} <= prepared
    # the engine moved the run directory aside (stale inputs): the overlays are gone
    for p in run.output["frames"][event["event_id"]]:
        Path(p).unlink()
    gone = frames_for(repo, run, event)
    assert gone and {str(f.path) for f in gone} <= prepared


def fake_rows(monkeypatch, tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows),
                    encoding="utf-8")
    (tmp_path / "run_meta.json").write_text('{"prompts": {}, "settings_sha": "s"}',
                                            encoding="utf-8")   # as the engine writes one
    monkeypatch.setattr(Pfb3x3Engine, "behavior", lambda self, *a, **kw: BehaviorResult(
        rows=path, windows=len(rows), errors=1, overlay_dir=tmp_path / "overlay"))


def test_a_camera_with_every_window_failed_fails_the_behavior_run(settings, repo, tmp_path,
                                                                 fake, monkeypatch):
    fake_rows(monkeypatch, tmp_path, [row(0, [], status="error", error="timeout"),
                                      row(1, [], status="skipped")])
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run.status == "failed" and "すべての区間で失敗" in run.error


def test_nobody_tracked_in_any_window_is_an_empty_run_with_a_warning(settings, repo, tmp_path,
                                                                     fake, monkeypatch):
    fake_rows(monkeypatch, tmp_path, [row(0, [], status="skipped"),
                                      row(1, [], status="skipped")])
    _, sid, subjects, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run.status == "ok" and run.output["events"] == []
    assert run.warnings == ["追跡された人物が映っている区間がありませんでした"]
    assert subject_status(repo, subjects["Pa"]) == SubjectStatus.insufficient


@pytest.mark.parametrize("bad", ["{not json", json.dumps({"window_id": "../x", "status": "ok"})])
def test_unreadable_engine_rows_fail_cleanly(settings, repo, tmp_path, fake, monkeypatch, bad):
    fake_rows(monkeypatch, tmp_path, [bad])
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert run.status == "failed" and "PFB3x3の行動の" in run.error and "cam1" in run.error
    assert "Traceback" not in run.error


def test_regenerated_behavior_after_a_new_mapping_follows_it(settings, repo, tmp_path, fake):
    """The teacher swaps two students on cam1 after behavior ran: once confirmed again, the
    rerun attributes the same engine events to the new students."""
    _, sid, subjects, ctx = tracked_flow(settings, repo, tmp_path)
    save_tracked_identity(repo, sid, "cam1", {"P1": "Pb", "P2": "Pa", "P3": None})
    confirm_identity(repo, sid)
    drain(repo, ctx)
    events = repo.latest_run(sid, Stage.behavior, "cam1").output["events"]
    assert [e["participant"] for e in events] == ["Pb", "Pa"]
    pfb_a = repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pa"])
    assert sorted(pfb_a.output["evidence_refs"]) == ["cam1-0001-01", "cam2-0001-01"]
    approve(repo, subjects["Pa"])   # the new chain is current


@pytest.mark.parametrize("engine, requests", [("pfb3x3", 3), ("paper", 1)])
def test_tracked_contribution_uses_smaller_chunks(settings, repo, engine, requests):
    """The tracking engine's many events overran max_tokens in chunks of 60 (the live run);
    it sends 20 per request, the paper engine keeps 60."""
    from pfb_app.domain.models import StageRun
    from pfb_app.pipeline.video import run_contribution
    aid = repo.create_assignment("塔づくり課題", Kind.video, "課題文", "基準", "standard", True)
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": "/v.mp4"}, "engine": engine})
    events = [{"event_id": f"cam1-{i:04d}-01", "participant": "Pa", "video_time": "00:05",
               "task_elapsed": "00:05", "observed_behavior": "紙を折る", "utterance": "",
               "speaker": "unknown", "evidence_source": "video", "confidence": "high",
               "camera": "cam1"} for i in range(45)]
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.behavior, camera="cam1", subject_id=None,
        version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"events": events}, status="ok", error=None, warnings=[], duration_sec=0.0,
        created_at=""))
    backend = FakeBackend(answer())
    run = run_contribution(StageContext(settings, repo, backend, lambda: False), sid, "cam1")
    assert run.status == "ok" and len(run.output["judgments"]) == 45
    assert len(backend.calls) == requests


# --- fix wave ---

def test_untranslated_behavior_blocks_approval_until_regenerated(settings, repo, tmp_path,
                                                                 fake):
    down = FakeBackend(lambda req: (_ for _ in ()).throw(OSError("connection reset")),
                       model_id="fake-translator")   # a transient failure: nothing cached
    aid, sid, subjects, ctx = tracked_flow(settings, repo, tmp_path, translator=down)
    assert untranslated_behavior(repo, sid) == {"cam1": 2, "cam2": 2}
    with pytest.raises(ApprovalError) as e:
        approve(repo, subjects["Pa"])
    assert "翻訳未完了" in str(e.value) and "再生成" in str(e.value)
    # regenerating behavior with a working translator clears it
    ctx.translator = fake_translator()
    regenerate_from(repo, sid, Stage.behavior)   # every camera
    drain(repo, ctx)
    assert untranslated_behavior(repo, sid) == {}
    approve(repo, subjects["Pa"])


def test_a_teacher_saved_behavior_run_never_blocks_approval(settings, repo, tmp_path, fake):
    down = FakeBackend(lambda req: (_ for _ in ()).throw(OSError("connection reset")),
                       model_id="fake-translator")
    _, sid, subjects, _ = tracked_flow(settings, repo, tmp_path, translator=down)
    for cam in MAPPINGS:   # the teacher reviewed the evidence and saved it (English kept)
        run = repo.latest_run(sid, Stage.behavior, cam)
        edited = edit_stage_output(repo, run.id, {"events": run.output["events"]})
        assert edited.source == "teacher" and english_left(edited) == 2   # still shown
    assert untranslated_behavior(repo, sid) == {}
    regenerate_from(repo, sid, Stage.contribution)   # downstream of the edits
    drain(repo, make_ctx(settings, repo, FakeBackend(answer())))
    approve(repo, subjects["Pa"])


def test_regenerating_retries_a_cached_translation_failure(settings, repo, tmp_path, fake):
    """A translator that deterministically fails a two-event request (cached in call_log)
    but answers one event at a time: the regeneration asks again, one per request."""
    inner = fake_translator()
    calls = []

    def respond(req):
        calls.append(req.user_text.count("ref_id: "))
        return "not json" if calls[-1] > 1 else inner._next(req)

    same = FakeBackend(respond, model_id="fake-translator")
    # no automatic people: their behavior run would make the teacher's one a regeneration
    aid, sid, subjects, ctx = tracked_flow(settings, repo, tmp_path, translator=same,
                                           auto_people=False)
    assert untranslated_behavior(repo, sid) == {"cam1": 2, "cam2": 2}
    first = len(calls)
    regenerate_from(repo, sid, Stage.behavior)
    drain(repo, ctx)
    assert untranslated_behavior(repo, sid) == {}
    assert calls[first:] and set(calls[first:]) == {1}   # fresh, one event per request
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    assert [e["observed_behavior"] for e in run.output["events"]] == ["紙を折る",
                                                                    "別の参加者に紙を渡す"]
    approve(repo, subjects["Pa"])


def test_still_english():
    assert still_english("本人 hands a strip to 別の参加者") is True   # only the stand-ins
    assert still_english("folds a paper strip") is True
    assert still_english("別の参加者に紙を渡す") is False
    assert still_english("紙を折る") is False and still_english("ﾃｽﾄ") is False


def test_a_camera_without_events_is_not_translated_and_not_blocking(settings, repo, tmp_path,
                                                                    fake, monkeypatch):
    fake_rows(monkeypatch, tmp_path, [row(0, [], status="skipped")])
    _, sid, _, _ = tracked_flow(settings, repo, tmp_path)
    prov = repo.latest_run(sid, Stage.behavior, "cam1").output["provenance"]
    assert prov["translated"] is False and prov["events"] == 0 and prov["translator"] is None
    assert untranslated_behavior(repo, sid) == {}


def test_an_unreachable_server_during_translation_requeues(repo, settings, tmp_path):
    class Gone:
        model_id = "served:m"

        def generate(self, reqs):
            raise ServerUnreachable("http://localhost:8200/v1: connection refused")

        def close(self):
            pass

    with pytest.raises(EngineServerDown):   # a YieldRequested: the job goes back to the queue
        translate_events(bridged(tmp_path), Gone(), repo, settings, lambda: False, [])
    assert issubclass(EngineServerDown, YieldRequested)


def test_lower_case_labels_are_labels_too():
    assert without_labels("p1 hands a strip to p2", "P1") == "本人 hands a strip to 別の参加者"
    assert "人物の番号が含まれています: P3" in checks.pfb_text_warnings("p3と協力した", set(), set(),
                                                                tracked=True)
    assert not any("人物の番号" in w for w in checks.pfb_text_warnings(
        "pdfやap3、p4xは対象外", set(), set(), tracked=True))
