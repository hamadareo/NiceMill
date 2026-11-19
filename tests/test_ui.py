from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

import pfb_app.ui
from pfb_app.config import Settings
from pfb_app.domain.models import Stage, StageRun
from pfb_app.services import review
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

UI = Path(pfb_app.ui.__file__).parent
PAGES = ["assignments", "submissions", "status", "review", "export"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3")   # never the real engine
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir)}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    # no GPU in unit tests: nvidia-smi may be missing or answer; the page must cope either way
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return settings, Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "失敗", warnings=["注意"],
        duration_sec=0.0, created_at=""))


def populate(settings, repo):
    # document assignment with an approved PFB
    doc = repo.create_assignment("レポート", "document", "課題文", "基準", "standard", False)
    dsid = repo.create_submission(doc, "山田.pdf", {"source_path": "/x.pdf", "original_filename": "山田.pdf"})
    dsub = repo.add_subject(dsid, None, "山田太郎", None, {})
    prep = save(repo, dsid, Stage.prepare, {"pages": [], "pdf_path": "/x.pdf"})
    ev = save(repo, dsid, Stage.evidence, {"items": [{
        "page": 1, "quote": "引用", "observed_content": "内容", "evidence_source": "text",
        "confidence": "high", "item_id": "p1-001", "quote_verified": False}]}, upstream=[prep.id])
    con = save(repo, dsid, Stage.contribution, {"judgments": [{
        "ref_id": "p1-001", "contribution_interpretation": "解釈", "is_positive_contribution": True,
        "rubric_criteria": ["基準A"]}]}, upstream=[ev.id])
    save(repo, dsid, Stage.pfb, {"pfb_status": "generated", "pfb": "よく考えて書けていました。",
                                 "evidence_refs": ["p1-001"]}, subject_id=dsub, upstream=[con.id])
    review.approve(repo, dsub)
    # video assignment with frames, a failed job and an unconfirmed identity
    vid = repo.create_assignment("授業", "video", "課題文", "", "standard", True)
    frame = files.frames_dir(settings, vid, 0, "cam1", create=True) / "f0.jpg"
    cv2.imwrite(str(frame), np.zeros((48, 85, 3), dtype=np.uint8))
    vsid = repo.create_submission(vid, "A班", {"cameras": {"cam1": "/v.mp4"}, "start_sec": None,
                                               "end_sec": None, "auto_transcribe": False})
    vsub = repo.add_subject(vsid, "Pa", "花子", "外見", {"cam1": "左"})
    vprep = save(repo, vsid, Stage.prepare, {"frames": [{
        "index": 0, "video_sec": 0.0, "video_time": "00:00", "task_elapsed": "00:00",
        "path": str(frame)}]}, camera="cam1")
    ident = save(repo, vsid, Stage.identity, {"participants": [{
        "participant_id": "Pa", "appearance": "外見", "initial_position": "左",
        "verification_points": ["開始時点"], "identity_confirmed": None, "identity_confidence": "high",
        "verified": True, "notes": ""}]}, camera="cam1", upstream=[vprep.id])
    beh = save(repo, vsid, Stage.behavior, {"events": [{
        "video_time": "00:00", "participant": "Pa", "observed_behavior": "紙を折る", "utterance": "",
        "speaker": "unknown", "evidence_source": "video", "confidence": "high",
        "event_id": "cam1-0001", "camera": "cam1", "task_elapsed": "00:00"}]},
        camera="cam1", upstream=[ident.id, vprep.id])
    save(repo, vsid, Stage.contribution, {"judgments": [{
        "ref_id": "cam1-0001", "contribution_interpretation": "解釈", "is_positive_contribution": True,
        "rubric_criteria": []}]}, camera="cam1", upstream=[beh.id])
    job = repo.enqueue(vsid, Stage.identity, "standard", camera="cam1")
    repo.claim_next_job({"standard"})
    repo.finish_job(job, "failed_retryable", "モデルの読み込みに失敗")
    return {"doc": doc, "dsub": dsub, "vid": vid, "vsid": vsid, "vsub": vsub, "job": job}


def run_page(name):
    at = AppTest.from_file(str(UI / "pages" / f"{name}.py"), default_timeout=60)
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_ui_pages_render_empty(env, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_ui_pages_render_with_data(env, page):
    populate(*env)
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_app_entry_renders(env):
    populate(*env)
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_status_page_survives_missing_gpu(env, monkeypatch):
    from pfb_app.inference import gpu
    def boom(*a, **k):
        raise gpu.GpuQueryError("none")
    monkeypatch.setattr(gpu, "query_nvidia_smi", boom)
    populate(*env)
    at = run_page("status")
    assert not at.exception
    assert any("GPU情報を取得できません" in i.value for i in at.info)


def test_status_page_shows_engine_session_readiness(env, monkeypatch):
    from pfb_app.inference import gpu
    monkeypatch.setattr(gpu, "query_nvidia_smi", lambda: gpu.GpuSnapshot(97.9, 40.0))
    at = run_page("status")
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "人物の追跡: 開始できます" in text            # 4 + 8 <= 57.9 GB free
    assert "PFB3x3（標準）: 開始できます" in text       # 34 + 8
    assert "PFB3x3（高精度）: 開始待ち" in text         # 60 + 8 > 57.9


def test_status_retry_requeues_failed_job(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("status")
    at.button(key=f"retry-{ids['job']}").click().run()
    assert not at.exception
    assert repo.get_job(ids["job"]).status.value == "queued"


def test_review_confirm_identity_button_and_frames(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("review")
    assert not at.exception
    # video is the second assignment; pick it
    at.selectbox[0].select(ids["vid"]).run()
    assert any(b.label == "参加者特定を確認" for b in at.button)
    assert not repo.get_submission(ids["vsid"])["identity_confirmed"]
    next(b for b in at.button if b.label == "参加者特定を確認").click().run()
    assert not at.exception
    assert repo.get_submission(ids["vsid"])["identity_confirmed"]


def test_review_shows_busy_error_as_message(env):
    settings, repo = env
    ids = populate(settings, repo)
    repo.enqueue(ids["vsid"], Stage.identity, "standard", camera="cam1")
    at = run_page("review")
    at.selectbox[0].select(ids["vid"]).run()
    at.text_area(key=f"manual-{ids['vsub']}").input("手書きの本文").run()
    next(b for b in at.button if b.label == "手書きのPFBを保存").click().run()
    assert not at.exception
    assert any("ジョブがあるため" in e.value for e in at.error)


def test_review_shows_frame_image_for_selected_behavior_row(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("review")
    at.selectbox[0].select(ids["vid"]).run()
    frames_box = next(s for s in at.selectbox if s.key and s.key.startswith("frames-"))
    assert not at.get("image")
    frames_box.select("cam1-0001").run()
    assert not at.exception
    assert at.get("image")


def test_review_shows_stale_badge_and_warnings(env):
    settings, repo = env
    ids = populate(settings, repo)
    # a newer prepare run makes everything downstream stale
    save(repo, ids["vsid"], Stage.prepare, {"frames": []}, camera="cam1")
    at = run_page("review")
    at.selectbox[0].select(ids["vid"]).run()
    assert not at.exception
    assert any("注意" in w.value for w in at.warning)
    assert any("badge" in m.value and "古い" in m.value for m in at.markdown)


def test_review_unapprove_document(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("review")
    at.button(key=f"unapprove-{ids['dsub']}").click().run()
    assert not at.exception
    assert repo.get_subject(ids["dsub"])["approval_status"] == "none"


def test_export_page_makes_pdf(env):
    settings, repo = env
    populate(settings, repo)
    at = run_page("export")
    at.button(key="make-teacher").click().run()
    assert not at.exception
    assert at.get("download_button")


def test_submissions_page_forms(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("submissions")  # first assignment: document
    assert not at.exception
    assert any(b.label == "登録" for b in at.button)
    at.selectbox[0].select(ids["vid"]).run()
    assert not at.exception
    assert at.get("form_submit_button") or at.button


def make_video(path: Path, seconds: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        writer.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def register_from_inbox(at, video, engine=None, people=None):
    if engine is not None:
        at.radio(key="engine").set_value(engine).run()
    source = next(r for r in at.radio if r.label == "動画の選び方")
    source.set_value("inboxフォルダから選ぶ").run()
    if people is not None:
        at.number_input(key="max_people").set_value(people)
    box = at.selectbox(key="inbox1")   # form widgets: set together, then submit
    box.select_index(box.options.index(video.name))
    at.text_input[0].input("B班")
    next(b for b in at.button if b.label == "登録").click().run()
    return at


@pytest.mark.parametrize("choice, engine, people, stored", [
    (None, "paper", None, None), ("paper", "paper", None, None),
    ("pfb3x3", "pfb3x3", 4, 4),
    ("pfb3x3", "pfb3x3", 0, None),      # 0: no cap
    ("pfb3x3", "pfb3x3", None, None),   # blank: the participants (none in this table)
])
def test_submissions_engine_radio_is_stored_in_the_meta(env, choice, engine, people, stored):
    settings, repo = env
    ids = populate(settings, repo)
    video = make_video(settings.inbox_dir / "group.mp4")
    at = run_page("submissions")
    at.selectbox[0].select(ids["vid"]).run()
    radio = at.radio(key="engine")
    assert radio.options == ["論文方式（標準）", "トラッキング方式（PFB3x3）"]
    assert radio.value == "paper"   # the paper engine is the default
    register_from_inbox(at, video, choice, people)
    assert not at.exception, [e.value for e in at.exception]
    subs = [s for s in repo.list_submissions(ids["vid"]) if s["label"] == "B班"]
    assert len(subs) == 1, [e.value for e in at.error]
    (sub,) = subs
    assert sub["meta"]["engine"] == engine
    assert sub["meta"].get("max_people") == stored
    assert ("max_people" in sub["meta"]) == (engine == "pfb3x3")
    captions = "\n".join(c.value for c in at.caption)
    if engine == "pfb3x3":
        assert "外見と初期位置は任意です" in captions
        # the engine is not set up in this test: the teacher is told
        assert any("セットアップされていません" in w.value for w in at.warning)
    else:
        assert "外見と初期位置（カメラごと）を手がかりに" in captions
    page = "\n".join(m.value for m in at.markdown)
    assert "分析の方式: " + ("トラッキング方式（PFB3x3）" if engine == "pfb3x3" else
                            "論文方式（標準）") in page


def test_export_never_offers_outdated_pdf_after_approval_change(env):
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("export")
    at.button(key="make-teacher").click().run()
    assert at.get("download_button")
    review.unapprove(repo, ids["dsub"])
    at.run()  # a rerun, as after any later interaction
    assert not at.exception
    assert not at.get("download_button")


def test_assignment_delete_refused_with_active_jobs(env):
    settings, repo = env
    ids = populate(settings, repo)
    repo.enqueue(ids["vsid"], Stage.identity, "standard", camera="cam1")
    at = run_page("assignments")
    at.selectbox(key="delete_target").select(ids["vid"]).run()
    at.checkbox(key="confirm_delete").check().run()
    next(b for b in at.button if b.label == "削除").click().run()
    assert not at.exception
    assert any("解析が待機中または実行中" in e.value for e in at.error)
    assert repo.get_assignment(ids["vid"])


def test_assignment_delete_goes_through_the_home_deletion(env, monkeypatch):
    from pfb_app.services import deletion
    settings, repo = env
    ids = populate(settings, repo)
    repo.conn.execute("DELETE FROM jobs")
    calls = []
    real = deletion.delete_work
    monkeypatch.setattr(deletion, "delete_work",
                        lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    at = run_page("assignments")
    at.selectbox(key="delete_target").select(ids["vid"]).run()
    at.checkbox(key="confirm_delete").check().run()
    next(b for b in at.button if b.label == "削除").click().run()
    assert not at.exception
    assert calls == [ids["vid"]]
    with pytest.raises(KeyError):
        repo.get_assignment(ids["vid"])
    assert repo.latest_audit("delete_work", f"assignment:{ids['vid']}") is not None


# --- fix wave: status page and untranslated behavior ---

def untranslated_submission(settings, repo):
    """A tracking-engine group whose cam1 behavior run kept 1 of 2 events in English."""
    aid = repo.create_assignment("塔づくり", "video", "課題文", "", "standard", True)
    sid = repo.create_submission(aid, "T班", {"cameras": {"cam1": "/v.mp4"}, "engine": "pfb3x3"})
    repo.add_subject(sid, "Pa", "花子", "", {})
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.behavior, {"events": [], "provenance": {
        "translated": False, "translated_events": 1, "events": 2}}, camera="cam1",
         upstream=[prep.id])
    return aid, sid


def test_status_page_shows_short_errors_the_model_column_and_untranslated(env):
    settings, repo = env
    ids = populate(settings, repo)
    untranslated_submission(settings, repo)
    jid = repo.enqueue(ids["vsid"], Stage.behavior, "pfb3x3:standard", camera="cam1")
    repo.claim_next_job({"pfb3x3:standard"})
    repo.finish_job(jid, "failed_retryable",
                    "EngineError: エンジン behavior が失敗しました (exit 1)\n"
                    "ログ: /data/x/pfb3x3/logs/behavior.err\n" + "\n".join(
                        f"traceback line {i}" for i in range(40)))
    repo.set_job_progress(jid, 0.3, "12 / 40 区間")
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    row = table[table["ID"] == jid].iloc[0]
    assert row["処理"] == "PFB3x3（標準）"
    assert row["進み具合"] == "30%（12 / 40 区間）"
    assert table[table["ID"] != jid]["進み具合"].eq("").all()      # nothing written: blank
    assert row["待機・失敗の理由"] == ("EngineError: エンジン behavior が失敗しました (exit 1)"
                                 "（ログ: /data/x/pfb3x3/logs/behavior.err）")
    text = "\n".join(m.value for m in at.markdown)
    assert "traceback line" not in text
    assert any("翻訳未完了" in w.value and "cam1（1件）" in w.value for w in at.warning)


def test_review_page_warns_about_untranslated_behavior(env):
    settings, repo = env
    aid, sid = untranslated_submission(settings, repo)
    at = run_page("review")
    at.selectbox[0].select(aid).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("翻訳未完了" in w.value for w in at.warning)


def test_review_page_does_not_block_a_teacher_saved_behavior_run(env):
    settings, repo = env
    aid, sid = untranslated_submission(settings, repo)
    run = repo.latest_run(sid, Stage.behavior, "cam1")
    repo.save_stage_run(run.model_copy(update={"id": None, "source": "teacher"}))
    assert review.untranslated_behavior(repo, sid) == {}
    at = run_page("review")
    at.selectbox[0].select(aid).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("翻訳未完了" in w.value for w in at.warning)


def test_status_page_renders_detect_and_names_jobs(env):
    settings, repo = env
    ids = populate(settings, repo)
    repo.enqueue(ids["vsid"], Stage.detect, "cpu")
    jid = repo.enqueue(ids["vsid"], Stage.names, "pfb3x3:standard")
    repo.claim_next_job({"pfb3x3:standard"})
    repo.finish_job(jid, "failed_retryable", "EngineError: names failed")
    repo.enqueue(ids["vsid"], Stage.names, "names:cpu")
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    table = at.dataframe[0].value
    assert {"内容の確認", "名前の候補づくり"} <= set(table["段階"])
    assert "名前の候補づくり" in set(table["処理"]) and "names:cpu" not in set(table["処理"])


def test_status_page_names_the_detect_session_plainly(env):
    settings, repo = env
    ids = populate(settings, repo)
    repo.enqueue(ids["vsid"], Stage.detect, "detect")
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    rows = at.dataframe[0].value
    detect_rows = rows[rows["段階"] == "内容の確認"]
    assert list(detect_rows["処理"]) == ["内容の確認"]   # not the raw key "detect"
    assert "detect" not in set(rows["処理"])


# --- UX round 2, L5: admin out of the way ---------------------------------------------------

def _done_only_work(repo, ids):
    """A second group whose jobs are all finished (closed work), next to the failed one."""
    sid = repo.create_submission(ids["doc"], "終わった班", {"source_path": "/y.pdf"})
    jid = repo.enqueue(sid, Stage.prepare, "standard")
    repo.claim_next_job({"standard"})
    repo.finish_job(jid, "done", None)
    return sid, jid


def test_status_page_lists_only_open_work_until_asked(env):
    settings, repo = env
    ids = populate(settings, repo)
    _, done_job = _done_only_work(repo, ids)
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    shown = set(at.dataframe[0].value["ID"])
    assert ids["job"] in shown and done_job not in shown      # the failed work is open
    toggle = next(t for t in at.toggle if t.label == "すべての作業を表示")
    assert toggle.value is False
    toggle.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert {ids["job"], done_job} <= set(at.dataframe[0].value["ID"])


def test_status_page_says_when_no_work_is_open(env):
    settings, repo = env
    ids = populate(settings, repo)
    repo.retry_job(ids["job"])
    repo.claim_next_job({"standard"})
    repo.finish_job(ids["job"], "done", None)
    at = run_page("status")
    assert not at.exception, [e.value for e in at.exception]
    assert not at.dataframe
    assert any("すべての作業を表示" in i.value for i in at.info)


def test_sidebar_starts_collapsed_and_the_admin_section_is_renamed():
    src = (UI / "app.py").read_text(encoding="utf-8")
    assert 'initial_sidebar_state="collapsed"' in src
    assert src.count("管理用の画面（くわしい操作）") >= 2     # the navigation group and the label
    assert '"詳細画面"' not in src and ">詳細画面<" not in src
