"""Review page, tracking engine (PFB3x3): the people sheet and the people-to-student mapping."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

import pfb_app.ui
from pfb_app.config import Settings
from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline.tracked_identity import OUTSIDE
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

UI = Path(pfb_app.ui.__file__).parent


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir)}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return settings, Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def tracked(settings, repo, tmp_path):
    """A tracking-engine group with 3 students; the tracker found 4 people on cam1."""
    aid = repo.create_assignment("授業", "video", "紙タワーを作る課題", "", "standard", False)
    sid = repo.create_submission(aid, "A班", {"cameras": {"cam1": "/v.mp4"}, "start_sec": 0.0,
                                              "end_sec": None, "auto_transcribe": False,
                                              "engine": "pfb3x3"})
    for pid, name in (("Pa", "花子"), ("Pb", "太郎"), ("Pc", "次郎")):
        repo.add_subject(sid, pid, name, "外見", {})
    sheet = tmp_path / "sheet.png"   # a tiny synthetic image, not a video frame
    cv2.imwrite(str(sheet), np.zeros((24, 32, 3), dtype=np.uint8))
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    track = save(repo, sid, Stage.track, {
        "people": 4, "labels": ["P1", "P2", "P3", "P4"], "sheet": str(sheet),
        "default_names": {"P1": "Pb", "P2": "Pa"}, "default_source": "rule"},
        camera="cam1", upstream=[prep.id])
    return aid, sid, track


def open_page(aid):
    at = AppTest.from_file(str(UI / "pages" / "review.py"), default_timeout=60).run()
    at.selectbox[0].select(aid).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_mapping_tab_shows_the_sheet_and_one_box_per_label(env, tmp_path):
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    at = open_page(aid)
    assert at.get("image")
    boxes = {label: at.selectbox(key=f"map-{track.id}-{label}") for label in ("P1", "P2", "P3", "P4")}
    assert boxes["P1"].value == "Pb" and boxes["P2"].value == "Pa"   # rule defaults
    assert boxes["P3"].value is None and boxes["P4"].value is None   # the teacher must choose
    assert boxes["P1"].options == ["花子", "太郎", "次郎", OUTSIDE]
    assert [b.label for b in boxes.values()] == ["1番の人", "2番の人", "3番の人", "4番の人"]
    captions = [c.value for c in at.caption]
    assert "推定した名前が入っています。確かめてから決定してください。" in captions
    assert captions.count("推定: 初期位置から推定") == 2          # P1 and P2, the rule's guesses
    assert not any(b.key == "regen-identity-cam1" for b in at.button)   # no VLM identity here
    assert any(b.key == "regen-track-cam1" for b in at.button)


def test_mapping_save_needs_every_label_then_saves(env, tmp_path):
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    at = open_page(aid)
    at.button(key=f"map-save-{track.id}").click().run()
    assert any("まだ選んでいない人物があります: 3番の人、4番の人" in e.value for e in at.error)
    assert repo.latest_run(sid, Stage.identity, "cam1") is None
    at.selectbox(key=f"map-{track.id}-P3").select("Pc").run()
    at.selectbox(key=f"map-{track.id}-P4").select(OUTSIDE).run()
    at.button(key=f"map-save-{track.id}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    run = repo.latest_run(sid, Stage.identity, "cam1")
    assert run.source == "teacher" and run.upstream_run_ids == [track.id]
    assert {p["label"]: p["participant_id"] for p in run.output["participants"]} == {
        "P1": "Pb", "P2": "Pa", "P3": "Pc", "P4": None}
    # the existing confirm button then confirms it
    next(b for b in at.button if b.label == "参加者特定を確認").click().run()
    assert not at.exception
    assert repo.get_submission(sid)["identity_confirmed"]


def test_mapping_save_shows_a_duplicate_student_error(env, tmp_path):
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    at = open_page(aid)
    at.selectbox(key=f"map-{track.id}-P3").select("Pa").run()
    at.selectbox(key=f"map-{track.id}-P4").select(OUTSIDE).run()
    at.button(key=f"map-save-{track.id}").click().run()
    assert not at.exception
    assert any("同じ参加者が複数の人物に割り当てられています: 花子" in e.value for e in at.error)
    assert repo.latest_run(sid, Stage.identity, "cam1") is None


def test_saved_mapping_is_shown_and_retrack_is_offered(env, tmp_path):
    from pfb_app.services.review import save_tracked_identity
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": None, "P4": "Pc"})
    at = open_page(aid)
    assert at.selectbox(key=f"map-{track.id}-P1").value == "Pa"
    assert at.selectbox(key=f"map-{track.id}-P3").value == OUTSIDE
    at.button(key="regen-track-cam1").click().run()
    assert not at.exception
    row = repo.conn.execute("SELECT stage, camera, model_key FROM jobs WHERE submission_id=?",
                            (sid,)).fetchone()
    assert tuple(row) == ("track", "cam1", "track")


def test_mapping_saved_before_a_roster_change_says_so(env, tmp_path):
    from pfb_app.pipeline.tracked_identity import RETRACKED, ROSTER_RESET
    from pfb_app.services.review import save_tracked_identity
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    run = save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": None,
                                                    "P4": "Pc"})
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "roster_run_floor": run.id})   # apply_roster
    at = open_page(aid)
    warnings = [w.value for w in at.warning]
    assert ROSTER_RESET in warnings and RETRACKED not in warnings


def test_paper_engine_identity_tab_is_unchanged(env, tmp_path):
    settings, repo = env
    aid = repo.create_assignment("授業", "video", "課題文", "", "standard", False)
    sid = repo.create_submission(aid, "B班", {"cameras": {"cam1": "/v.mp4"}, "start_sec": None,
                                              "end_sec": None, "auto_transcribe": False})
    repo.add_subject(sid, "Pa", "花子", "外見", {})
    at = open_page(aid)
    assert any(b.key == "regen-identity-cam1" for b in at.button)
    assert not any(b.key == "regen-track-cam1" for b in at.button)


def test_tracking_engine_tab_names_the_mapping_and_shows_the_overlays(env, tmp_path):
    from pfb_app.services.review import save_tracked_identity
    settings, repo = env
    aid, sid, track = tracked(settings, repo, tmp_path)
    identity = save_tracked_identity(repo, sid, "cam1",
                                     {"P1": "Pa", "P2": "Pb", "P3": None, "P4": "Pc"})
    shots = []
    for k in (1, 2):   # tiny synthetic images standing in for the engine's overlays
        shots.append(tmp_path / "overlay" / "s1_cam1_w0000" / f"{k}.jpg")
        shots[-1].parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(shots[-1]), np.zeros((24, 32, 3), dtype=np.uint8))
    event = {"event_id": "cam1-0000-01", "camera": "cam1", "participant": "Pa",
             "video_time": "00:05", "task_elapsed": "0:05", "observed_behavior": "紙を折る",
             "utterance": "", "speaker": "unknown", "evidence_source": "video",
             "confidence": "high"}
    prep = repo.latest_run(sid, Stage.prepare, "cam1")
    behavior = save(repo, sid, Stage.behavior,
                    {"events": [event], "frames": {"cam1-0000-01": [str(p) for p in shots]}},
                    camera="cam1", upstream=[identity.id, prep.id, track.id])
    at = open_page(aid)
    assert [t.label for t in at.tabs][:2] == ["人物と参加者の対応づけ", "行動の観察"]
    before = len(at.get("image"))
    at.selectbox(key=f"frames-{behavior.id}").select("cam1-0000-01").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("image")) == before + 2   # both overlays sent to the model
