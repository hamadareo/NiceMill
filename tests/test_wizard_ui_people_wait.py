"""Wizard steps 3 (人を確かめる（任意） / 参加者を確認) and 4 (解析を待つ), AppTest on ui/app.py.

Only synthetic data: the people sheet is a tiny black PNG, no video frame is ever read."""
import os
import time
from collections import Counter

import cv2
import numpy as np

from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline.tracked_identity import OUTSIDE
from pfb_app.services.intake import register_documents
from pfb_app.services import auto_people, review
from pfb_app.services.review import (ApprovalError, BusyError, confirm_identity,
                                     save_tracked_identity)
from pfb_app.storage import db, files
from pfb_app.ui.wizard import live, router, step_people, step_wait
from pfb_app.ui.wizard.progress import RERUN, RESELECT, RETRY, View, work_progress
from test_wizard_ui_basic import (PRIMARY_LABELS, app, env, html,  # noqa: F401 (env)
                                  no_internal_terms, ok,
                                  pdf_bytes, primaries, register, screen_text, shown_step,
                                  video_assignment)

STUDENTS = (("Pa", "花子"), ("Pb", "太郎"), ("Pc", "次郎"))
LABELS = ["P1", "P2", "P3", "P4"]
DECIDE = "これで決定して次へ"


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), duration=0.0):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id,
        version=0, source="model", upstream_run_ids=list(upstream), prompt_hash=None,
        model_id=None, output=output, status="ok", error=None, warnings=[],
        duration_sec=duration, created_at=""))


def group(repo, aid, label="A班", cameras=("cam1",), engine="pfb3x3"):
    meta = {"cameras": {c: f"/v/{c}.mp4" for c in cameras}, "start_sec": 0.0,
            "end_sec": None, "auto_transcribe": False}
    if engine == "pfb3x3":
        meta["engine"] = engine
    sid = repo.create_submission(aid, label, meta)
    for pid, name in STUDENTS:
        repo.add_subject(sid, pid, name, "外見", {})
    return sid


def track(settings, repo, sid, camera="cam1", defaults=None):
    sheet = settings.data_dir / f"sheet_{sid}_{camera}.png"   # tiny synthetic image
    cv2.imwrite(str(sheet), np.zeros((24, 32, 3), dtype=np.uint8))
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    return save(repo, sid, Stage.track, {
        "people": 4, "labels": LABELS, "sheet": str(sheet),
        "default_names": defaults or {}, "default_source": "rule" if defaults else None},
        camera=camera, upstream=[prep.id])


def job(repo, sid, stage, status="queued", camera=None, subject_id=None, error=None,
        model="cpu"):
    jid = repo.enqueue(sid, stage, model, camera=camera, subject_id=subject_id)
    repo.conn.execute("UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, jid))
    return jid


def finish_all(repo, sid):
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))


def worker_alive(settings, seconds_ago=0):
    path = files.worker_beat_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


def mapped(repo, sid, cameras=("cam1",)):
    mapping = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": None}
    for cam in cameras:
        save_tracked_identity(repo, sid, cam, mapping)
    confirm_identity(repo, sid)


def texts(at, kind):
    return [e.value for e in at.get(kind)]


def links(at):
    return [link.proto.label for link in at.get("page_link")]


def primary(at):
    return next(b for b in at.button if b.proto.type == "primary")


def unmapped_check(at):
    return next(c for c in at.checkbox if c.label == step_people.UNMAPPED_CHECK)


def fill(at, track_run, values):
    for label, value in values.items():
        at.selectbox(key=f"pfb_map_{track_run.id}_{label}").select(value)
    return at.run()


# --- step 3, tracking -----------------------------------------------------------------------

def test_the_mapping_form_saves_confirms_and_moves_to_step_4(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid, defaults={"P1": "Pb", "P2": "Pa"})
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert primaries(at) == [DECIDE]
    assert len(at.get("image")) == 1                        # the people sheet, on the left
    boxes = {label: at.selectbox(key=f"pfb_map_{run.id}_{label}") for label in LABELS}
    assert boxes["P1"].value == "Pb" and boxes["P2"].value == "Pa"   # the rule's suggestion
    assert boxes["P3"].value is None
    assert step_people.SUGGESTED in screen_text(at)
    assert "初期位置から推定" in html(at)                     # the reason on the card
    assert "花子" in boxes["P1"].options and OUTSIDE in boxes["P1"].options
    no_internal_terms(at)

    primary(at).click().run()                               # P3 and P4 still unset
    ok(at)
    assert any("まだ選んでいない人がいます: 3番の人、4番の人" in e for e in texts(at, "error"))
    assert repo.latest_run(sid, Stage.identity, "cam1") is None

    fill(at, run, {"P3": "Pc", "P4": OUTSIDE})
    primary(at).click().run()
    ok(at)
    identity = repo.latest_run(sid, Stage.identity, "cam1")
    assert identity.source == "teacher"
    assert {p["label"]: p["participant_id"] for p in identity.output["participants"]} == {
        "P1": "Pb", "P2": "Pa", "P3": "Pc", "P4": None}
    assert repo.get_submission(sid)["identity_confirmed"]
    assert repo.has_active_job(sid, Stage.behavior, "cam1")
    assert work_progress(repo, aid).current == 4
    assert shown_step(at) == 4
    assert any("つなぎ方を決定しました" in s for s in texts(at, "success"))


def test_two_cameras_are_decided_one_after_the_other(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    runs = {cam: track(settings, repo, sid, cam) for cam in ("cam1", "cam2")}
    at = ok(app(session={"assignment_id": aid}))
    switcher = at.radio(key=f"pfb_cam_{sid}")
    assert switcher.value == "cam1" and "カメラ2" in html(at)
    assert primaries(at) == [DECIDE]
    values = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": OUTSIDE}
    fill(at, runs["cam1"], values)
    primary(at).click().run()
    ok(at)
    assert repo.latest_run(sid, Stage.identity, "cam1") is not None
    assert not repo.get_submission(sid)["identity_confirmed"]       # cam2 still to decide
    assert at.radio(key=f"pfb_cam_{sid}").value == "cam2"
    assert any("次はカメラ2です" in s for s in texts(at, "success"))
    assert shown_step(at) == 3
    fill(at, runs["cam2"], values)
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["identity_confirmed"]
    assert shown_step(at) == 4


def test_the_waiting_state_is_shown_while_the_track_job_is_queued(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, camera="cam1", model="track")
    worker_alive(settings)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    page = html(at)
    assert "人を見つけています" in page                        # the card title
    assert "順番を待っています（人を見つける・カメラ1）" in page
    assert primaries(at) == [DECIDE] and primary(at).disabled
    assert live.WORKER_STOPPED not in texts(at, "warning")
    assert not at.get("image")
    no_internal_terms(at)
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    at.run()
    assert "人を見つけています（カメラ1）" in html(at)
    # the track has finished: the next run shows the form
    track(settings, repo, sid)
    finish_all(repo, sid)
    at.run()
    ok(at)
    assert not primary(at).disabled and len(at.get("image")) == 1


def test_step_3_tracking_shows_the_panel(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, "running", camera="cam1", model="track")
    repo.set_job_progress(jid, 0.25, "人物を追跡しています")
    worker_alive(settings)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    page = html(at)
    assert "全体の進み具合" in page and "人を見つける" in page
    assert '<span class="pfb-stagelist__note">人物を追跡しています</span>' in page
    no_internal_terms(at)


def test_queued_work_without_a_worker_says_so(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    job(repo, sid, Stage.prepare, camera="cam1")
    at = ok(app(session={"assignment_id": aid}))
    assert live.WORKER_STOPPED in texts(at, "warning")       # never started
    worker_alive(settings, seconds_ago=10 * 60)
    assert live.WORKER_STOPPED in texts(at.run(), "warning")  # silent for 10 minutes
    worker_alive(settings)
    assert live.WORKER_STOPPED not in texts(at.run(), "warning")


def test_redoing_the_mapping_after_results_needs_the_confirmation(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)
    mapped(repo, sid)
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1",
               upstream=[repo.latest_run(sid, Stage.identity, "cam1").id])
    finish_all(repo, sid)
    old_identity = repo.latest_run(sid, Stage.identity, "cam1").id
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": None}))
    assert shown_step(at) == 3
    assert at.selectbox(key=f"pfb_map_{run.id}_P1").value == "Pa"     # the saved mapping
    assert not primary(at).disabled                       # unchanged: nothing to warn about
    assert not at.checkbox
    fill(at, run, {"P1": "Pb", "P2": "Pa"})
    ok(at)
    assert any("やり直しになります" in w for w in texts(at, "warning"))
    assert primary(at).disabled                            # until the teacher confirms
    no_internal_terms(at)
    at.checkbox[0].check().run()
    assert not primary(at).disabled
    primary(at).click().run()
    ok(at)
    identity = repo.latest_run(sid, Stage.identity, "cam1")
    assert identity.id != old_identity
    assert {p["label"]: p["participant_id"] for p in identity.output["participants"]}["P1"] == "Pb"
    assert repo.get_submission(sid)["identity_confirmed"]
    assert repo.is_stale(beh)
    assert repo.has_active_job(sid, Stage.behavior, "cam1")    # recomputed
    assert shown_step(at) == 4


def test_unchanged_mapping_is_not_saved_again(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    track(settings, repo, sid)
    mapped(repo, sid)
    finish_all(repo, sid)
    before = repo.latest_run(sid, Stage.identity, "cam1").id
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": None}))
    primary(at).click().run()
    ok(at)
    assert repo.latest_run(sid, Stage.identity, "cam1").id == before


def test_a_failed_track_stops_step_3_with_a_reason_and_a_link(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.track, "failed", camera="cam1", model="track",
        error="EngineError: 人物を追跡できませんでした")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert "解析が止まりました" in html(at)
    assert any("人物を追跡できませんでした（人を見つける・カメラ1）" in e
               for e in texts(at, "error"))
    assert router.DETAILS in links(at)
    assert primaries(at) == [RETRY]                 # no disabled 「これで決定して次へ」
    no_internal_terms(at)


def test_several_groups_can_be_switched(env):
    settings, repo = env
    aid = video_assignment(repo)
    a = group(repo, aid, "A班")
    b = group(repo, aid, "B班")
    track(settings, repo, a)
    run_b = track(settings, repo, b)
    at = ok(app(session={"assignment_id": aid}))
    assert at.button(key=f"pfb_group_{a}").disabled                   # shown
    at.button(key=f"pfb_group_{b}").click().run()
    ok(at)
    assert at.session_state["wizard"]["submission_id"] == b
    assert at.selectbox(key=f"pfb_map_{run_b.id}_P1") is not None
    assert primaries(at) == [DECIDE]


# --- step 3, paper --------------------------------------------------------------------------

def test_the_paper_identity_is_read_only_and_confirmed(env):
    settings, repo = env
    aid = video_assignment(repo)                 # with the identity gate
    sid = group(repo, aid, engine="paper")
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.identity, {"participants": [
        {"participant_id": pid, "appearance": "赤い服", "initial_position": "左前",
         "verification_points": ["赤い服"], "identity_confidence": "high", "verified": True,
         "notes": ""} for pid, _ in STUDENTS]}, camera="cam1", upstream=[prep.id])
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert primaries(at) == ["参加者の特定を確認"]
    assert not at.selectbox                                   # read-only
    table = at.dataframe[0].value
    assert list(table["参加者"]) == ["花子", "太郎", "次郎"]
    assert "管理用の画面で直す" in links(at)
    no_internal_terms(at)
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["identity_confirmed"]
    assert repo.has_active_job(sid, Stage.behavior, "cam1")
    assert shown_step(at) == 4


# --- step 4 ---------------------------------------------------------------------------------

def confirmed_and_analysing(settings, repo, aid):
    sid = group(repo, aid)
    track(settings, repo, sid)
    mapped(repo, sid)                  # behavior(cam1) queued
    return sid


def test_step_4_lists_stages_and_students_in_plain_words(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    worker_alive(settings)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    for word in ("全体の進み具合", "内容の確認", "人を見つける", "したことの書き出し",
                 "よかった点を選ぶ", "文章づくり", "確認と承認", "済み", "進行中", "待機中",
                 "したことを書き出しています（カメラ1）", "花子", "前の処理を待っています"):
        assert word in page, word
    assert page.index("全体の進み具合") < page.index("したことを書き出しています（カメラ1）")
    assert primaries(at) == []                      # nothing to press: only 「ホームへ戻る」
    assert at.button(key="pfb_wait_next").label == step_wait.HOME_WHILE_RUNNING
    assert any(e.label == "この画面の使い方" for e in at.expander)
    no_internal_terms(at)


def test_step_4_shows_the_gpu_wait_in_plain_words(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu', error='空き 3.0GB < 必要 20.0GB'"
                      " WHERE submission_id=?", (sid,))
    at = ok(app(session={"assignment_id": aid}))
    page = html(at)
    assert "順番を待っています（したことの書き出し）" in page and "順番待ち" in page
    assert "GPU" not in page
    assert "必要" not in page and "GB" not in page


def test_a_stopped_job_shows_a_reason_and_a_detail_link(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='failed', error=? WHERE submission_id=?",
                      ("ValueError: 行動を読み取れませんでした", sid))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    assert "止まっています" in page and "行動を読み取れませんでした" in page
    assert router.DETAILS in links(at) and primaries(at) == [RETRY]
    no_internal_terms(at)


def test_the_estimate_needs_history(env):
    settings, repo = env
    aid = video_assignment(repo)
    confirmed_and_analysing(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert "残り約" not in html(at)
    # finished runs of an earlier work: behavior 120 s, contribution 60 s, feedback 30 s
    old = group(repo, video_assignment(repo, "前の作業"), "前の班")
    subj = repo.list_subjects(old)[0]["id"]
    for seconds in (100, 120, 140):
        save(repo, old, Stage.behavior, {"events": []}, camera="cam1", duration=seconds)
    save(repo, old, Stage.contribution, {"judgments": []}, camera="cam1", duration=60)
    save(repo, old, Stage.pfb, {"pfb_status": "generated", "pfb": "良い", "evidence_refs": []},
         subject_id=subj, duration=30)
    # one behavior + one contribution + feedback for 3 students: 120 + 60 + 90 s
    assert live.remaining_jobs(repo, aid) == Counter({
        ("pfb3x3", "behavior"): 1, ("pfb3x3", "contribution"): 1, ("pfb3x3", "pfb"): 3})
    db.clear_stage_durations_cache()           # the history is kept for five minutes
    at.run()
    ok(at)
    assert "残り約5分（目安）" in html(at)
    assert live.ESTIMATE_NOTE in [c.value for c in at.caption]


def test_remaining_text():
    assert live.remaining_text(30) == "残り約1分（目安）"
    assert live.remaining_text(11 * 60) == "残り約15分（目安）"
    assert live.remaining_text(61 * 60) == "残り約1時間10分（目安）"


def test_remaining_jobs_through_the_automatic_people(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    job(repo, sid, Stage.prepare, camera="cam1")
    job(repo, sid, Stage.prepare, camera="cam2")
    assert live.remaining_jobs(repo, aid, through=True) == Counter({
        ("pfb3x3", "prepare"): 2, ("pfb3x3", "track"): 2, ("pfb3x3", "behavior"): 2,
        ("pfb3x3", "contribution"): 2, ("pfb3x3", "pfb"): 3})


def test_remaining_sec_takes_off_what_running_steps_have_done(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    assert live.remaining_sec(repo, aid) is None                 # no history yet
    old = group(repo, video_assignment(repo, "前の作業"), "前の班")
    subj = repo.list_subjects(old)[0]["id"]
    save(repo, old, Stage.behavior, {"events": []}, camera="cam1", duration=120)
    save(repo, old, Stage.contribution, {"judgments": []}, camera="cam1", duration=60)
    save(repo, old, Stage.pfb, {"pfb_status": "generated", "pfb": "良い", "evidence_refs": []},
         subject_id=subj, duration=30)
    db.clear_stage_durations_cache()
    assert live.remaining_sec(repo, aid) == 270                  # 120 + 60 + 3 x 30
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=?", (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    repo.set_job_progress(jid, 0.5, "20 / 40 区間")
    assert live.remaining_sec(repo, aid) == 210                  # half the behavior is done
    repo.set_job_progress(jid, 1.0, None)
    assert live.remaining_sec(repo, aid) == 150                  # what follows it is left


def test_the_estimate_counts_what_follows():
    assert live.estimate_text(30) == "残り時間の目安: 約1分"
    assert live.estimate_text(9 * 60) == "残り時間の目安: 約9分"
    assert live.estimate_text(11 * 60) == "残り時間の目安: 約15分"
    assert live.estimate_text(61 * 60) == "残り時間の目安: 約1時間10分"
    assert live.estimate_text(120 * 60) == "残り時間の目安: 約2時間"


def test_remaining_jobs_stop_at_the_teacher_and_cover_documents(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    job(repo, sid, Stage.prepare, camera="cam1")
    job(repo, sid, Stage.prepare, camera="cam2")
    # tracking: prepare -> track per camera, then the teacher maps the people
    front = Counter({("pfb3x3", "prepare"): 2, ("pfb3x3", "track"): 2})
    assert live.remaining_jobs(repo, aid) == front
    assert live.remaining_jobs(repo, aid, front_only=True) == front
    doc = repo.create_assignment("レポート", "document", "まとめる", "", "standard", False)
    register_documents(repo, settings, doc, [("山田太郎.pdf", pdf_bytes())], None)
    assert live.remaining_jobs(repo, doc) == Counter({
        ("document", "prepare"): 1, ("document", "evidence"): 1,
        ("document", "contribution"): 1, ("document", "pfb"): 1})


def test_step_4_follows_to_step_5_when_someone_can_be_checked(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    con = save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1",
               upstream=[beh.id])
    for s in repo.list_subjects(sid):
        save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "丁寧", "evidence_refs": []},
             subject_id=s["id"], upstream=[con.id])
    finish_all(repo, sid)
    at.run()                          # what the 10-second watch triggers
    ok(at)
    assert shown_step(at) == 5


def test_the_watch_opens_the_review_step_when_someone_is_ready(monkeypatch):
    calls = []
    monkeypatch.setattr(step_wait.router, "go", lambda step: calls.append(step))
    old = (4, ((1, "wait", "processing"), (2, "wait", "processing")), None)
    step_wait._moved(old, (4, ((1, "wait", "processing"), (2, "wait", "stopped")), None))
    assert calls == []
    # another group's people wait for the teacher (derived step 3): still open 確認して承認
    step_wait._moved(old, (3, ((1, "review", "your_turn"), (2, "people", "your_turn")), 5))
    assert calls == [5]


# --- every new screen -----------------------------------------------------------------------

def test_each_new_screen_has_one_primary_help_and_plain_words(env):
    settings, repo = env
    worker_alive(settings)
    waiting = video_assignment(repo, "追跡中")
    register(settings, repo, waiting)                    # prepare queued: step 3 waiting
    mapping = video_assignment(repo, "つなぐ")
    track(settings, repo, group(repo, mapping))
    analysing = video_assignment(repo, "解析中")
    confirmed_and_analysing(settings, repo, analysing)
    docs = repo.create_assignment("レポート", "document", "まとめる", "", "standard", False)
    register_documents(repo, settings, docs, [("山田太郎.pdf", pdf_bytes())], None)
    paper = video_assignment(repo, "論文方式")
    psid = group(repo, paper, engine="paper")
    save(repo, psid, Stage.identity, {"participants": []}, camera="cam1")
    for aid, step in ((waiting, 3), (mapping, 3), (analysing, 4), (analysing, 3), (docs, 3),
                      (paper, 3)):
        at = ok(app(session={"assignment_id": aid, "step": step, "submission_id": None}))
        assert shown_step(at) == step
        if (aid, step) in ((analysing, 4), (docs, 3)):   # nothing to press: 「ホームへ戻る」
            assert primaries(at) == [], primaries(at)
        else:
            assert len(primaries(at)) == 1, (aid, primaries(at))
            assert primaries(at)[0] in PRIMARY_LABELS
        assert any(e.label == "この画面の使い方" for e in at.expander), aid
        no_internal_terms(at)


# --- review round 1 -------------------------------------------------------------------------

def confirms(repo, sid):
    return repo.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='confirm_identity'"
                             " AND target=?", (f"submission:{sid}",)).fetchone()[0]


def behavior_jobs(repo, sid):
    return repo.conn.execute("SELECT camera FROM jobs WHERE submission_id=? AND stage='behavior'"
                             " ORDER BY id", (sid,)).fetchall()


def test_a_two_camera_redo_confirms_and_reanalyses_once(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    runs = {cam: track(settings, repo, sid, cam) for cam in ("cam1", "cam2")}
    mapped(repo, sid, ("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.behavior, {"events": []}, camera=cam,
             upstream=[repo.latest_run(sid, Stage.identity, cam).id])
    finish_all(repo, sid)
    before_confirms, before_jobs = confirms(repo, sid), len(behavior_jobs(repo, sid))
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": None}))
    assert at.radio(key=f"pfb_cam_{sid}").value == "cam1"
    for cam in ("cam1", "cam2"):
        fill(at, runs[cam], {"P1": "Pb", "P2": "Pa"})
        at.checkbox(key=f"pfb_redo_{runs[cam].id}").check().run()
        primary(at).click().run()
        ok(at)
        if cam == "cam1":       # saved, but not confirmed yet: カメラ2 comes next
            assert at.radio(key=f"pfb_cam_{sid}").value == "cam2"
            assert not repo.get_submission(sid)["identity_confirmed"]
            assert len(behavior_jobs(repo, sid)) == before_jobs
    assert confirms(repo, sid) == before_confirms + 1
    assert [r[0] for r in behavior_jobs(repo, sid)[before_jobs:]] == ["cam1", "cam2"]
    assert shown_step(at) == 4


def test_an_unchanged_later_camera_passes_straight_to_the_confirm(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    runs = {cam: track(settings, repo, sid, cam) for cam in ("cam1", "cam2")}
    mapped(repo, sid, ("cam1", "cam2"))
    save(repo, sid, Stage.behavior, {"events": []}, camera="cam1",
         upstream=[repo.latest_run(sid, Stage.identity, "cam1").id])
    finish_all(repo, sid)
    before = confirms(repo, sid)
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": None}))
    fill(at, runs["cam1"], {"P1": "Pb", "P2": "Pa"})
    at.checkbox(key=f"pfb_redo_{runs['cam1'].id}").check().run()
    primary(at).click().run()
    assert at.radio(key=f"pfb_cam_{sid}").value == "cam2" and confirms(repo, sid) == before
    assert not primary(at).disabled                        # unchanged: no warning
    primary(at).click().run()
    ok(at)
    assert confirms(repo, sid) == before + 1


def test_students_left_without_a_person_need_a_confirmation(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    fill(at, run, {"P1": "Pa", "P2": "Pb", "P3": OUTSIDE, "P4": OUTSIDE})
    ok(at)
    assert any("次郎さんに人物が割り当てられていません" in w and "場面が少ない" in w
               for w in texts(at, "warning"))
    assert primary(at).disabled
    unmapped_check(at).check().run()
    assert not primary(at).disabled
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["identity_confirmed"]


def test_across_cameras_only_students_missing_everywhere_are_named(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    runs = {cam: track(settings, repo, sid, cam) for cam in ("cam1", "cam2")}
    at = ok(app(session={"assignment_id": aid}))
    fill(at, runs["cam1"], {"P1": "Pa", "P2": "Pb", "P3": OUTSIDE, "P4": OUTSIDE})
    assert any("次郎さん" in w for w in texts(at, "warning"))    # not on this camera
    unmapped_check(at).check().run()
    primary(at).click().run()
    ok(at)
    fill(at, runs["cam2"], {"P1": "Pc", "P2": OUTSIDE, "P3": OUTSIDE, "P4": OUTSIDE})
    # the last camera: 次郎 is on カメラ2, 花子 and 太郎 on カメラ1 -> nobody is missing
    assert not any("割り当てられていません" in w for w in texts(at, "warning"))
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["identity_confirmed"]


def test_choices_are_kept_when_switching_cameras(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    runs = {cam: track(settings, repo, sid, cam) for cam in ("cam1", "cam2")}
    at = ok(app(session={"assignment_id": aid}))
    fill(at, runs["cam1"], {"P1": "Pc", "P2": "Pa"})
    at.radio(key=f"pfb_cam_{sid}").set_value("cam2").run()
    assert at.selectbox(key=f"pfb_map_{runs['cam2'].id}_P1").value is None   # cam1 not shown
    at.radio(key=f"pfb_cam_{sid}").set_value("cam1").run()
    ok(at)
    assert at.selectbox(key=f"pfb_map_{runs['cam1'].id}_P1").value == "Pc"
    assert at.selectbox(key=f"pfb_map_{runs['cam1'].id}_P2").value == "Pa"


def test_another_group_moving_on_does_not_move_an_open_form(env):
    settings, repo = env
    aid = video_assignment(repo)
    a = group(repo, aid, "A班", cameras=("cam1", "cam2"))
    track(settings, repo, a, "cam1")
    job(repo, a, Stage.track, "running", camera="cam2", model="track")   # still tracking
    b = group(repo, aid, "B班")
    track(settings, repo, b)
    mapped(repo, b)                           # B班 analysing
    signature = step_people._signature(aid, a, form=True)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3 and len(at.get("image")) == 1          # A班 カメラ1 form
    base = signature(View.load(repo, aid))   # one tick's view of the work
    ident = repo.latest_run(b, Stage.identity, "cam1")                  # B班 reaches review
    beh = save(repo, b, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    con = save(repo, b, Stage.contribution, {"judgments": []}, camera="cam1",
               upstream=[beh.id])
    for s in repo.list_subjects(b):
        save(repo, b, Stage.pfb, {"pfb_status": "generated", "pfb": "丁寧", "evidence_refs": []},
             subject_id=s["id"], upstream=[con.id])
    finish_all(repo, b)
    assert work_progress(repo, aid).turn_step == 5
    assert signature(View.load(repo, aid)) == base   # the watch does not rerun the page
    at.run()                                   # nor does any other rerun move the screen
    ok(at)
    assert shown_step(at) == 3 and len(at.get("image")) == 1


def test_a_failed_confirm_after_a_save_is_shown_after_the_rerun(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)

    def refuse(repo, sid):
        raise ApprovalError("人物の追跡が実行中または待機中です（cam1）。")
    monkeypatch.setattr(review, "confirm_identity", refuse)
    at = ok(app(session={"assignment_id": aid}))
    fill(at, run, {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": OUTSIDE})
    primary(at).click().run()
    ok(at)
    assert repo.latest_run(sid, Stage.identity, "cam1").source == "teacher"     # saved
    assert any("確定できませんでした" in e and "カメラ1" in e for e in texts(at, "error"))
    assert shown_step(at) == 3


def test_estimates_use_the_same_method_and_skip_runs_under_a_second(env):
    settings, repo = env
    paper = group(repo, video_assignment(repo, "論文"), "P班", engine="paper")
    tracking = group(repo, video_assignment(repo, "追跡"), "T班")
    doc = repo.create_assignment("レポート", "document", "まとめる", "", "standard", False)
    dsub = repo.create_submission(doc, "佐藤", {"source_path": "/x/a.pdf"})
    save(repo, paper, Stage.behavior, {"events": []}, camera="cam1", duration=900)
    save(repo, tracking, Stage.behavior, {"events": []}, camera="cam1", duration=60)
    save(repo, tracking, Stage.behavior, {"events": []}, camera="cam1", duration=0.2)
    save(repo, dsub, Stage.contribution, {"judgments": []}, duration=30)
    assert repo.stage_durations(Stage.behavior, "pfb3x3") == [60]
    assert repo.stage_durations(Stage.behavior, "paper") == [900]
    assert repo.stage_durations(Stage.contribution, "document") == [30]
    assert repo.stage_durations(Stage.contribution, "pfb3x3") == []


# --- polish (carried over from the review of steps 3 and 4) ----------------------------------

def test_a_busy_confirm_after_a_save_is_shown_after_the_rerun(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)

    def busy(repo, sid):
        raise BusyError("running")
    monkeypatch.setattr(review, "confirm_identity", busy)
    at = ok(app(session={"assignment_id": aid}))
    fill(at, run, {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": OUTSIDE})
    primary(at).click().run()
    ok(at)
    assert repo.latest_run(sid, Stage.identity, "cam1").source == "teacher"     # saved
    errors = texts(at, "error")
    assert any("保存しましたが、確定できませんでした" in e and step_people.BUSY in e
               for e in errors), errors
    assert shown_step(at) == 3
    assert not at.selectbox(key=f"pfb_map_{run.id}_P1").disabled        # the form is back


def test_the_unmapped_confirmation_is_dropped_when_the_choices_change(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    fill(at, run, {"P1": "Pa", "P2": "Pb", "P3": OUTSIDE, "P4": OUTSIDE})
    unmapped_check(at).check().run()
    assert not primary(at).disabled
    fill(at, run, {"P1": "Pb", "P2": "Pa"})              # other choices, 次郎 still unmapped
    ok(at)
    assert any("次郎さん" in w for w in texts(at, "warning"))
    assert not unmapped_check(at).value
    assert primary(at).disabled
    assert not repo.get_submission(sid)["identity_confirmed"]



# --- never stuck: the stopped states' action ------------------------------------------------

def age(repo, sid, hours=1):
    """Move the group's last activity an hour back (past the grace period)."""
    t = db._now(-hours * 3600)
    repo.conn.execute("UPDATE stage_runs SET created_at=? WHERE submission_id=?", (t, sid))
    repo.conn.execute("UPDATE jobs SET created_at=?, heartbeat_at=NULL WHERE submission_id=?",
                      (t, sid))


def work1_shape(settings, repo, aid):
    """No roster, everyone 対象外, behavior with no events, nothing running."""
    sid = repo.create_submission(aid, "A班", {"cameras": {"cam1": "/v/cam1.mp4"},
                                             "start_sec": 0.0, "end_sec": None,
                                             "auto_transcribe": False, "engine": "pfb3x3"})
    track(settings, repo, sid)
    save_tracked_identity(repo, sid, "cam1", {lb: None for lb in LABELS})
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    age(repo, sid)
    return sid


def test_wait_shows_reselect_as_the_only_primary(env, monkeypatch):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = work1_shape(settings, repo, aid)
    calls = []
    monkeypatch.setattr(auto_people, "reselect",
                        lambda r, s, i: calls.append(i) or [])
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    assert primaries(at) == [RESELECT]
    assert at.button(key="pfb_wait_next").label == step_wait.HOME_STOPPED
    assert at.button(key="pfb_wait_next").proto.type != "primary"
    page = html(at)
    assert "解析が止まりました" in page and "解析しています" not in page
    assert "評価する人が見つかりませんでした。" in page
    assert router.DETAILS not in links(at)             # the action is the way out
    no_internal_terms(at)
    primary(at).click().run()
    ok(at)
    assert calls == [sid]


def test_reselect_is_disabled_while_the_work_waits_for_deletion(env, monkeypatch):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    work1_shape(settings, repo, aid)
    repo.audit("delete_requested", f"assignment:{aid}", {})      # deleted once a step ends
    calls = []
    monkeypatch.setattr(auto_people, "reselect", lambda r, s, i: calls.append(i) or [])
    at = ok(app(session={"assignment_id": aid}))
    button = at.button(key=next(b.key for b in at.button if b.label == RESELECT))
    assert button.disabled and at.button(key="pfb_wait_next").proto.type != "primary"
    assert any("いまの処理が終わると削除されます" in i.value for i in at.info)
    assert calls == []


def test_wait_with_nobody_tracked_links_to_the_details(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = repo.create_submission(aid, "A班", {"cameras": {"cam1": "/v/cam1.mp4"},
                                             "start_sec": 0.0, "end_sec": None,
                                             "auto_transcribe": False, "engine": "pfb3x3"})
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.track, {"people": 0, "labels": [], "default_names": {}},
         camera="cam1", upstream=[prep.id])
    auto_people.auto_identity(repo, sid, "cam1")
    age(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    assert "動画に人が見つかりませんでした。管理用の画面でやり直してください" in page
    assert "解析が止まりました" in page
    assert RESELECT not in [b.label for b in at.button]
    assert router.DETAILS in links(at)
    assert primaries(at) == []                          # nothing to press here
    assert at.button(key="pfb_wait_next").label == step_wait.HOME_STOPPED
    no_internal_terms(at)


def test_a_busy_reselect_says_so_plainly(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    work1_shape(settings, repo, aid)

    def busy(*a, **k):
        raise BusyError(review._busy_message("映っている人の選び直し"))
    monkeypatch.setattr(auto_people, "reselect", busy)
    at = ok(app(session={"assignment_id": aid}))
    primary(at).click().run()
    ok(at)
    assert texts(at, "error") == [auto_people.BUSY]
    no_internal_terms(at)


def test_zero_events_offers_the_rerun(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    track(settings, repo, sid)
    mapped(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    age(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [RERUN]
    assert "人の動きが記録できませんでした。" in html(at)
    primary(at).click().run()
    ok(at)
    assert repo.has_active_job(sid, Stage.behavior, "cam1")
    assert shown_step(at) == 4 and primaries(at) == []
    assert at.button(key="pfb_wait_next").label == step_wait.HOME_WHILE_RUNNING
    assert "解析しています" in html(at)


def test_a_names_job_does_not_make_a_stopped_work_look_busy(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = work1_shape(settings, repo, aid)
    job(repo, sid, Stage.names, camera="cam1")          # never holds the work up
    age(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert "解析が止まりました" in html(at) and "解析しています" not in html(at)
    assert primaries(at) == [RESELECT]


def test_one_stopped_group_and_nothing_running_says_stopped(env):
    """Another group is about to go on (nothing runs yet): the title says the analysis has
    stopped, not that it is running."""
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = work1_shape(settings, repo, aid)
    other = work1_shape(settings, repo, aid)
    age(repo, other, hours=0)                           # just active: still preparing
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    assert "次の処理を準備しています" in page
    assert "解析が止まりました" in page and "解析しています" not in page
    job(repo, sid, Stage.behavior, camera="cam1")      # something runs: 解析しています
    at = ok(app(session={"assignment_id": aid}))
    assert "解析しています" in html(at) and "解析が止まりました" not in html(at)


def test_a_refused_reselect_says_so_plainly(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    work1_shape(settings, repo, aid)

    def refused(*a, **k):
        raise review._invalid("このグループはトラッキング方式ではありません。")
    monkeypatch.setattr(auto_people, "reselect", refused)
    at = ok(app(session={"assignment_id": aid}))
    primary(at).click().run()
    ok(at)
    assert texts(at, "error") == [step_wait.CANNOT_RESELECT]
    no_internal_terms(at)
