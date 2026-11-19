"""Wizard progress: which step a work is at, whether it is the teacher's turn, and why."""
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.export.pdf import collect_approved
from pfb_app.services.assignments import mark_pdf_downloaded, update_assignment
from pfb_app.services.review import (BusyError, approve, confirm_identity, save_tracked_identity,
                                     unapprove)
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.services import auto_people
from pfb_app.ui.wizard.progress import (DOCUMENT_STEPS, NO_EVENTS, NO_ONE_TRACKED, NOBODY,
                                        PAPER_STEPS,
                                        TRACKING_STEPS, all_work, plain_error, step_labels,
                                        work_progress)

FOUR = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": "Pd"}
STUDENTS = (("Pa", "A子"), ("Pb", "B太"), ("Pc", "C美"), ("Pd", "D郎"))
PFB = {"pfb_status": "generated", "pfb": "紙を丁寧に折っていました。", "evidence_refs": ["e1"]}
INSUFFICIENT = {"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []}


@pytest.fixture
def repo(tmp_path):
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), source="model",
         status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "失敗", warnings=[],
        duration_sec=0.0, created_at=""))


def job(repo, sid, stage, status="queued", camera=None, subject_id=None, error=None):
    jid = repo.enqueue(sid, stage, "cpu", camera=camera, subject_id=subject_id)
    repo.conn.execute("UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, jid))
    return jid


def finish_all(repo, sid):
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))


def video_work(repo, gate=False, title="動画の課題"):
    return repo.create_assignment(title, Kind.video, "紙タワーを作る課題", "ルーブリック",
                                  "standard", gate)


def group(repo, aid, engine="pfb3x3", cameras=("cam1",), label="G1", students=STUDENTS):
    meta = {"cameras": {c: f"/x/{c}.mp4" for c in cameras}, "start_sec": None,
            "end_sec": None, "auto_transcribe": False}
    if engine == "pfb3x3":
        meta["engine"] = engine
    sid = repo.create_submission(aid, label, meta)
    subjects = {pid: repo.add_subject(sid, pid, name, "外見", {}) for pid, name in students}
    return sid, subjects


def track(repo, sid, camera="cam1", labels=("P1", "P2", "P3", "P4")):
    prep = repo.latest_run(sid, Stage.prepare, camera) or save(
        repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    return save(repo, sid, Stage.track, {"people": len(labels), "labels": list(labels),
                                         "sheet": "/x/sheet.jpg", "default_names": {}},
                camera=camera, upstream=[prep.id])


def tracked_to_review(repo, sid, subjects, pfbs=None, cameras=("cam1",), behavior=None):
    """track -> mapping -> confirm -> behavior/contribution per camera -> pfb per student."""
    pids = list(subjects)
    mapping = {f"P{n}": pids[n - 1] if n <= len(pids) else None for n in range(1, 5)}
    for cam in cameras:
        track(repo, sid, cam)
        save_tracked_identity(repo, sid, cam, mapping)   # the rest are 対象外
    confirm_identity(repo, sid)
    cons = []
    for cam in cameras:
        ident = repo.latest_run(sid, Stage.identity, cam)
        beh = save(repo, sid, Stage.behavior, behavior or {"events": []}, camera=cam,
                   upstream=[ident.id])
        cons.append(save(repo, sid, Stage.contribution, {"judgments": []}, camera=cam,
                         upstream=[beh.id]))
    finish_all(repo, sid)
    for pid, subj in subjects.items():
        out = (pfbs or {}).get(pid, PFB)
        save(repo, sid, Stage.pfb, out, subject_id=subj, upstream=[c.id for c in cons])


def approve_short_text(repo, subject_id):
    """「この文で承認する」 for a person with too little evidence (UX round 2, L6)."""
    from pfb_app.services import short_text
    subject = repo.get_subject(subject_id)
    run = repo.latest_run(subject["submission_id"], Stage.pfb, subject_id=subject_id)
    short_text.approve_short(repo, subject_id, run.id, short_text.offer(repo, subject, run))


def paper_to_identity(repo, sid, cameras=("cam1",)):
    for cam in cameras:
        prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        save(repo, sid, Stage.identity, {"participants": []}, camera=cam, upstream=[prep.id])


def document_work(repo, names=("佐藤",)):
    aid = repo.create_assignment("書類の課題", Kind.document, "防災を調べる", "", "standard",
                                 False)
    subs = []
    for name in names:
        sid = repo.create_submission(aid, name, {"source_path": "/x/a.pdf"})
        subj = repo.add_subject(sid, None, name, None, {})
        subs.append((sid, subj))
    return aid, subs


def document_to_review(repo, sid, subj, pfb=PFB):
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x/a.pdf"})
    ev = save(repo, sid, Stage.evidence, {"items": []}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": []}, upstream=[ev.id])
    save(repo, sid, Stage.pfb, pfb, subject_id=subj, upstream=[con.id])


# --- step labels and the empty work ---

def test_a_video_work_without_submissions_waits_for_videos(repo):
    aid = video_work(repo)
    p = work_progress(repo, aid)
    assert p.steps == TRACKING_STEPS == ["活動を決める", "動画を入れる", "人を確かめる（任意）",
                                         "解析を待つ", "確認して承認", "PDFを受け取る"]
    assert (p.current, p.done, p.your_turn, p.stopped) == (2, {1}, True, False)
    assert p.reason == "動画を入れてください。" and p.subs == []
    assert (p.approved, p.total, p.complete) == (0, 0, False)
    assert p.title == "動画の課題" and p.kind == "video"


def test_a_document_work_has_its_own_steps(repo):
    aid, _ = document_work(repo, names=())
    p = work_progress(repo, aid)
    assert p.steps == DOCUMENT_STEPS == ["活動を決める", "書類を入れる", "解析を待つ",
                                         "確認して承認", "PDFを受け取る"]
    assert (p.current, p.your_turn, p.reason) == (2, True, "書類を入れてください。")


def test_paper_groups_show_the_paper_step_label(repo):
    aid = video_work(repo)
    group(repo, aid, engine="paper")
    assert work_progress(repo, aid).steps == PAPER_STEPS
    assert PAPER_STEPS[2] == "参加者を確認"
    group(repo, aid, engine="pfb3x3", label="G2")   # any tracking group: the tracking label
    assert work_progress(repo, aid).steps == TRACKING_STEPS


# --- tracking engine, step 3 ---

def test_tracking_prepare_and_track_jobs_mean_waiting_on_step_3(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    job(repo, sid, Stage.prepare, camera="cam1")
    p = work_progress(repo, aid)
    assert (p.current, p.done, p.your_turn, p.stopped) == (3, {1, 2}, False, False)
    assert p.subs[0].status == "processing" and p.subs[0].step == "people"
    assert p.subs[0].engine == "pfb3x3" and p.subs[0].label == "G1"
    assert "カメラ1" in p.reason and "順番" in p.reason
    finish_all(repo, sid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.track, "running", camera="cam1")
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (3, False)
    assert p.reason == "人を見つけています（カメラ1）"


def test_tracking_done_without_mapping_is_the_teachers_turn(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.stopped) == (3, True, False)
    assert p.reason == "4人の名前を選ぶのを待っています"
    assert p.subs[0].status == "your_turn"


def test_one_camera_still_tracking_keeps_the_work_waiting(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=("cam1", "cam2"))
    track(repo, sid, "cam1")
    job(repo, sid, Stage.track, "running", camera="cam2")
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (3, False)
    assert p.reason == "人を見つけています（カメラ2）"


def test_mapped_but_unconfirmed_asks_for_the_confirmation(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (3, True)
    assert p.reason == "選んだ名前の決定を待っています"


def test_a_mapping_of_an_older_track_needs_mapping_again(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    track(repo, sid, labels=("P1", "P2", "P3"))   # tracked again
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.reason) == (3, True, "3人の名前を選ぶのを待っています")


def test_a_mapping_saved_before_a_roster_change_needs_mapping_again(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    run = save_tracked_identity(repo, sid, "cam1", FOUR)
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "roster_run_floor": run.id})   # apply_roster
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.reason) == (3, True, "4人の名前を選ぶのを待っています")


def test_confirmed_tracking_with_behavior_jobs_is_step_4(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    confirm_identity(repo, sid)   # queues behavior
    p = work_progress(repo, aid)
    assert (p.current, p.done, p.your_turn) == (4, {1, 2, 3}, False)
    assert p.subs[0].step == "wait" and "したことの書き出し" in p.reason
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    assert work_progress(repo, aid).reason == "したことを書き出しています（カメラ1）"
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu', error='空き 3GB < 必要 20GB'"
                      " WHERE submission_id=?", (sid,))
    p = work_progress(repo, aid)
    assert p.subs[0].status == "waiting_gpu" and "順番を待っています" in p.reason
    assert "GPU" not in p.reason and "必要" not in p.reason


# --- paper engine, step 3 ---

def test_paper_identity_job_is_waiting_on_step_3(repo):
    aid = video_work(repo, gate=True)
    sid, _ = group(repo, aid, engine="paper")
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.identity, "running", camera="cam1")
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (3, False)
    assert p.reason == "参加者を特定しています（カメラ1）"


def test_paper_identity_done_with_the_gate_is_the_teachers_turn(repo):
    aid = video_work(repo, gate=True)
    sid, _ = group(repo, aid, engine="paper")
    paper_to_identity(repo, sid)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (3, True)
    assert p.reason == "参加者の特定の確認を待っています"
    confirm_identity(repo, sid)   # gate on: queues behavior
    assert work_progress(repo, aid).current == 4


def test_paper_without_the_gate_analyses_on_and_asks_at_the_end(repo):
    aid = video_work(repo, gate=False)
    sid, subjects = group(repo, aid, engine="paper", students=STUDENTS[:1])
    paper_to_identity(repo, sid)
    job(repo, sid, Stage.behavior, "running", camera="cam1")
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (4, False)
    finish_all(repo, sid)
    con = save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1")
    save(repo, sid, Stage.pfb, PFB, subject_id=subjects["Pa"], upstream=[con.id])
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.reason) == (3, True, "参加者の特定の確認を待っています")


# --- review, approval, PDF, complete ---

def test_students_in_review_are_step_5(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects)
    p = work_progress(repo, aid)
    assert (p.current, p.done, p.your_turn) == (5, {1, 2, 3, 4}, True)
    assert p.reason == "4人の承認を待っています"
    assert (p.approved, p.total, p.complete) == (0, 4, False)
    assert not p.reachable(6) and p.reachable(3) and p.reachable(5)


def test_excluded_leaves_counts_and_pdf(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects)
    approve(repo, subjects["Pa"])
    approve(repo, subjects["Pb"])
    assert (work_progress(repo, aid).total, len(collect_approved(repo, aid))) == (4, 2)
    repo.set_excluded(subjects["Pb"], True)
    p = work_progress(repo, aid)
    assert (p.approved, p.total) == (1, 3)
    assert [e.display_name for e in collect_approved(repo, aid)] == ["A子"]


def test_approval_opens_the_pdf_step_and_download_completes(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects, pfbs={"Pd": INSUFFICIENT})
    approve(repo, subjects["Pa"])
    p = work_progress(repo, aid)
    assert (p.current, p.approved, p.total) == (5, 1, 4)
    assert p.reachable(6) and p.reason == "2人の承認を待っています"
    approve(repo, subjects["Pb"])
    approve(repo, subjects["Pc"])
    p = work_progress(repo, aid)   # Pd has too little evidence: its short text waits (L6)
    assert (p.current, p.your_turn, p.complete) == (5, True, False)
    assert p.reachable(6) and p.reason == "1人の短い文が承認を待っています"
    approve_short_text(repo, subjects["Pd"])
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.complete) == (6, True, False)
    assert p.done == {1, 2, 3, 4, 5}
    assert "PDF" in p.reason
    mark_pdf_downloaded(repo, aid)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.complete, p.stopped) == (6, False, True, False)
    assert p.done == {1, 2, 3, 4, 5, 6} and p.subs[0].status == "done"


def test_an_approval_after_the_download_needs_a_new_download(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:2])
    tracked_to_review(repo, sid, subjects, pfbs={"Pb": INSUFFICIENT})
    approve(repo, subjects["Pa"])
    approve_short_text(repo, subjects["Pb"])      # else Pb's short text waits (L6)
    mark_pdf_downloaded(repo, aid)
    assert work_progress(repo, aid).complete
    write = repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pb"])
    save(repo, sid, Stage.pfb, PFB, subject_id=subjects["Pb"], upstream=write.upstream_run_ids,
         source="teacher")
    approve(repo, subjects["Pb"])
    assert not work_progress(repo, aid).complete
    mark_pdf_downloaded(repo, aid)
    assert work_progress(repo, aid).complete


def test_excluding_after_the_download_is_not_served_from_the_cache(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:2])
    tracked_to_review(repo, sid, subjects, pfbs={"Pb": INSUFFICIENT})
    approve(repo, subjects["Pa"])
    approve_short_text(repo, subjects["Pb"])      # else Pb's short text waits (L6)
    mark_pdf_downloaded(repo, aid)
    assert (work_progress(repo, aid).total, work_progress(repo, aid).complete) == (2, True)
    repo.set_excluded(subjects["Pa"], True)
    p = work_progress(repo, aid)
    assert p.total == 1 and p.approved == 1    # recounted, not the cached (2, 2); Pb's short
    #                                            text keeps it complete (L6)
    repo.set_excluded(subjects["Pa"], False)
    p = work_progress(repo, aid)
    assert (p.total, p.complete) == (2, True)


def test_only_insufficient_students_stay_on_step_5(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:1])
    tracked_to_review(repo, sid, subjects, pfbs={"Pa": INSUFFICIENT},
                      behavior={"events": [event("Pa")]})   # recorded, yet too little
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.complete) == (5, True, False)
    assert "短い文" in p.reason and not p.reachable(5 + 1)   # the short text waits (L6)


def test_document_steps_follow_the_document_numbering(repo):
    aid, [(sid, subj)] = document_work(repo)
    job(repo, sid, Stage.prepare)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.subs[0].engine) == (3, False, "document")
    assert p.reason == "順番を待っています（書類の読み込み）"
    finish_all(repo, sid)
    document_to_review(repo, sid, subj)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.reason) == (4, True, "1人の承認を待っています")
    approve(repo, subj)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (5, True)
    mark_pdf_downloaded(repo, aid)
    assert work_progress(repo, aid).complete


def test_document_reviews_add_up_across_submissions(repo):
    aid, subs = document_work(repo, names=("佐藤", "鈴木", "高橋"))
    for sid, subj in subs:
        document_to_review(repo, sid, subj)
    p = work_progress(repo, aid)
    assert (p.current, p.reason, p.total) == (4, "3人の承認を待っています", 3)


# --- mixed submissions ---

def test_mixed_submissions_take_the_earliest_step(repo):
    aid = video_work(repo)
    sid1, subjects1 = group(repo, aid, label="1班")
    tracked_to_review(repo, sid1, subjects1)
    sid2, _ = group(repo, aid, label="2班")
    job(repo, sid2, Stage.track, "running", camera="cam1")
    p = work_progress(repo, aid)
    assert p.current == 3 and p.done == {1, 2}
    assert [(s.label, s.step, s.status) for s in p.subs] == [
        ("1班", "review", "your_turn"), ("2班", "people", "processing")]
    assert p.your_turn   # 1班 can be approved meanwhile
    assert p.reason == "1班: 4人の承認を待っています"
    assert (p.turn_step, p.furthest) == (5, 5)
    assert p.reachable(p.turn_step) and p.reachable(4) and not p.reachable(6)
    finish_all(repo, sid2)
    track(repo, sid2)
    p = work_progress(repo, aid)
    assert (p.current, p.reason, p.turn_step) == (3, "2班: 4人の名前を選ぶのを待っています", 3)


# --- failures ---

def test_a_failed_job_stops_the_work_with_a_plain_reason(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.track, "failed_retryable", camera="cam1",
        error="EngineError: 人物を見つけられませんでした\nログ: /x/engine.log")
    p = work_progress(repo, aid)
    assert (p.stopped, p.your_turn, p.current) == (True, False, 3)
    assert p.subs[0].status == "stopped"
    assert p.reason == "人物を見つけられませんでした（人を見つける・カメラ1）"
    assert p.subs[0].action == "retry"


def test_a_retried_or_superseded_failure_no_longer_stops(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:1])
    tracked_to_review(repo, sid, subjects)
    jid = job(repo, sid, Stage.pfb, "failed", subject_id=subjects["Pa"],
              error="フィードバックの形式が正しくありません")
    assert work_progress(repo, aid).stopped
    repo.retry_job(jid)
    assert not work_progress(repo, aid).stopped
    repo.conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (jid,))
    save(repo, sid, Stage.pfb, PFB, subject_id=subjects["Pa"], source="teacher",
         upstream=repo.latest_run(sid, Stage.pfb, subject_id=subjects["Pa"]).upstream_run_ids)
    p = work_progress(repo, aid)   # the teacher wrote it by hand
    assert not p.stopped and p.current == 5


def test_failure_reasons_are_plain_japanese():
    assert plain_error("RuntimeError: CUDA out of memory. Tried 2 GiB") == \
        "ほかの解析と重なって止まりました。もう一度ためすと、たいてい進みます。"
    assert plain_error("heartbeat timeout") == "処理が途中で応答しなくなりました"
    assert plain_error("ValueError: bad thing happened") == "予期しないエラーが起きました"
    assert plain_error("VideoError: 動画を開けません（cam2）") == "動画を開けません（カメラ2）"
    assert plain_error("") == "原因がわかりません"


def test_a_stopped_work_also_reports_its_failed_document(repo):
    aid, [(sid, _)] = document_work(repo)
    job(repo, sid, Stage.prepare, "failed", error="書類を読み込めませんでした")
    p = work_progress(repo, aid)
    assert (p.stopped, p.current) == (True, 3)
    assert p.reason == "書類を読み込めませんでした（書類の読み込み）"


# --- the home list ---

def test_all_work_sorts_your_turn_stopped_processing_complete(repo):
    done = video_work(repo, title="完了")
    sid, subjects = group(repo, done, students=STUDENTS[:1])
    tracked_to_review(repo, sid, subjects)
    approve(repo, subjects["Pa"])
    mark_pdf_downloaded(repo, done)
    busy = video_work(repo, title="解析中")
    sid, _ = group(repo, busy)
    job(repo, sid, Stage.track, "running", camera="cam1")
    stopped = video_work(repo, title="停止")
    sid, _ = group(repo, stopped)
    job(repo, sid, Stage.prepare, "failed", camera="cam1", error="動画を開けません")
    video_work(repo, title="あなたの番")
    assert [w.title for w in all_work(repo)] == ["あなたの番", "停止", "解析中", "完了"]


# --- services ---

def test_update_assignment_until_a_job_exists(repo):
    aid = video_work(repo)
    update_assignment(repo, aid, "新しい課題文", "新しいルーブリック")
    a = repo.get_assignment(aid)
    assert (a["task_text"], a["rubric_text"]) == ("新しい課題文", "新しいルーブリック")
    with pytest.raises(ValueError):
        update_assignment(repo, aid, "  ", "")
    sid, _ = group(repo, aid)
    update_assignment(repo, aid, "まだ変えられる", "")   # a submission alone is no job
    jid = job(repo, sid, Stage.prepare, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (jid,))
    with pytest.raises(BusyError):
        update_assignment(repo, aid, "もう変えられない", "")
    assert repo.get_assignment(aid)["task_text"] == "まだ変えられる"


def test_mark_pdf_downloaded_records_the_time(repo):
    aid = video_work(repo)
    assert repo.get_assignment(aid)["pdf_downloaded_at"] is None
    mark_pdf_downloaded(repo, aid)
    assert repo.get_assignment(aid)["pdf_downloaded_at"]
    assert repo.latest_audit("pdf_downloaded", f"assignment:{aid}") is not None
    with pytest.raises(KeyError):
        mark_pdf_downloaded(repo, aid + 100)


def test_the_migration_is_idempotent_on_an_existing_db(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE assignments (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " title TEXT NOT NULL, kind TEXT NOT NULL, task_text TEXT NOT NULL,"
                " rubric_text TEXT NOT NULL, model_key TEXT NOT NULL,"
                " require_identity_approval INTEGER NOT NULL, created_at TEXT NOT NULL)")
    old.execute("INSERT INTO assignments (title, kind, task_text, rubric_text, model_key,"
                " require_identity_approval, created_at) VALUES ('既存', 'video', 't', 'r',"
                " 'standard', 0, '2026-10-01')")
    old.commit()
    old.close()
    for _ in range(3):
        conn = connect(path)
        init_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(assignments)")]
        assert cols.count("pdf_downloaded_at") == 1
        conn.close()
    repo = Repo(connect(path))
    a = repo.get_assignment(1)
    assert a["title"] == "既存" and a["pdf_downloaded_at"] is None


# --- review round 1 ---

def test_only_a_job_that_redoes_the_failed_stage_clears_it(repo):
    aid = video_work(repo, gate=True)
    sid, subjects = group(repo, aid, engine="paper", cameras=("cam1", "cam2"),
                          students=STUDENTS[:1])
    paper_to_identity(repo, sid, cameras=("cam1", "cam2"))
    repo.set_identity_confirmed(sid, True)
    job(repo, sid, Stage.behavior, "failed", camera="cam1", error="行動を読み取れませんでした")
    job(repo, sid, Stage.behavior, "running", camera="cam2")
    p = work_progress(repo, aid)
    assert p.stopped and p.reason == "行動を読み取れませんでした（したことの書き出し・カメラ1）"
    job(repo, sid, Stage.pfb, "queued", subject_id=subjects["Pa"])   # later in the pipeline
    job(repo, sid, Stage.identity, "queued", camera="cam2")          # another camera
    assert work_progress(repo, aid).stopped
    job(repo, sid, Stage.identity, "queued", camera="cam1")   # regenerated from before it
    p = work_progress(repo, aid)
    assert not p.stopped and p.current == 3
    assert p.reason == "順番を待っています（参加者の特定・カメラ1）"


def test_a_failed_pfb_is_cleared_only_by_the_same_student(repo):
    aid, subs = document_work(repo, names=("佐藤",))
    (sid, subj), = subs
    document_to_review(repo, sid, subj)
    job(repo, sid, Stage.pfb, "failed", subject_id=subj, error="フィードバックを書けませんでした")
    other = repo.add_subject(sid, None, "鈴木", None, {})
    job(repo, sid, Stage.pfb, "queued", subject_id=other)
    assert work_progress(repo, aid).stopped
    job(repo, sid, Stage.pfb, "queued", subject_id=subj)
    assert not work_progress(repo, aid).stopped


def test_stale_feedback_is_named_not_summed(repo):
    aid = video_work(repo)
    for label in ("1班", "2班"):
        sid, subjects = group(repo, aid, label=label)
        tracked_to_review(repo, sid, subjects)
    assert work_progress(repo, aid).reason == "8人の承認を待っています"
    con = repo.latest_run(sid, Stage.contribution, "cam1")   # 2班's results are recomputed
    save(repo, sid, Stage.contribution, con.output, camera="cam1", upstream=con.upstream_run_ids)
    p = work_progress(repo, aid)
    assert p.reason == "2班: 4人の承認を待っています（内容が古くなった人がいます）（ほか1件）"


def test_untranslated_records_are_named(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects, behavior={"events": [], "provenance": {
        "translated": False, "events": 3, "translated_events": 1}})
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn) == (5, True) and "英語" in p.reason


def test_a_missing_next_result_is_processing_for_a_short_while(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.prepare, "done", camera="cam1")   # track not queued (yet)
    p = work_progress(repo, aid)
    assert (p.subs[0].status, p.stopped, p.current) == ("processing", False, 3)
    p = work_progress(repo, aid, now=datetime.now(timezone.utc) + timedelta(seconds=60))
    assert p.stopped
    assert p.reason == "人を見つけた結果がありません（カメラ1）。管理用の画面でやり直してください"


def big_work(repo):
    """10 groups x 4 students x 2 cameras, 2 approvals per group."""
    aid = video_work(repo)
    for g in range(10):
        sid, subjects = group(repo, aid, cameras=("cam1", "cam2"), label=f"{g + 1}班")
        tracked_to_review(repo, sid, subjects, cameras=("cam1", "cam2"))
        for pid in ("Pa", "Pb"):
            approve(repo, subjects[pid])
    return aid


def traced(repo, fn):
    statements = []
    repo.conn.set_trace_callback(statements.append)
    try:
        started = time.perf_counter()
        result = fn()
        return result, statements, time.perf_counter() - started
    finally:
        repo.conn.set_trace_callback(None)


def test_work_progress_reads_a_big_work_in_few_statements(repo):
    aid = big_work(repo)
    p, statements, elapsed = traced(repo, lambda: work_progress(repo, aid))
    assert (p.current, p.approved, p.total, p.reason) == (5, 20, 40, "20人の承認を待っています")
    assert len(statements) < 300, len(statements)
    print(f"\nwork_progress 10 groups x 4 students x 2 cameras: {len(statements)} statements,"
          f" {elapsed * 1000:.1f} ms")


def test_a_complete_work_is_answered_from_the_cache_until_it_changes(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:1])
    tracked_to_review(repo, sid, subjects)
    approve(repo, subjects["Pa"])
    mark_pdf_downloaded(repo, aid)
    first = work_progress(repo, aid)
    assert first.complete
    assert first.analysis.finished and first.analysis.percent == 100
    again, statements, _ = traced(repo, lambda: work_progress(repo, aid))
    assert again == first and len(statements) <= 3
    unapprove(repo, subjects["Pa"])
    p = work_progress(repo, aid)
    assert (p.complete, p.current, p.your_turn) == (False, 5, True)


def test_complete_cache_key_comes_from_the_snapshot_read(repo, monkeypatch):
    from pfb_app.ui.wizard import progress as pr
    pr._COMPLETE.clear()
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:2])
    tracked_to_review(repo, sid, subjects, pfbs={"Pb": INSUFFICIENT})
    approve(repo, subjects["Pa"])
    approve_short_text(repo, subjects["Pb"])      # else Pb's short text waits (L6)
    mark_pdf_downloaded(repo, aid)
    snap = pr.snapshot(repo, aid)
    assert snap.signature is not None
    # a signature that changes after the snapshot's read must not become the key
    monkeypatch.setattr(pr, "_signature", lambda r, a: ("later",) + snap.signature[1:])
    assert pr.work_progress(snap, aid).complete
    assert [k for k in pr._COMPLETE] == [(snap.signature[0], aid)]
    assert next(iter(pr._COMPLETE.values()))[0] == snap.signature
    # a snapshot without a signature is never cached
    pr._COMPLETE.clear()
    snap.signature = None
    assert pr.work_progress(snap, aid).complete
    assert not pr._COMPLETE


def test_step_labels_sports():
    from pfb_app.ui.wizard.progress import step_labels
    sports = step_labels("video", "pfb3x3", "sports")
    assert sports[0] == "試合・練習を決める" and sports[1:] == TRACKING_STEPS[1:]
    assert step_labels("video", "pfb3x3", "school") == TRACKING_STEPS
    assert step_labels("document", "paper", "school") == DOCUMENT_STEPS
    assert step_labels("video", "paper", "school") == PAPER_STEPS


def test_a_sports_work_shows_its_own_first_step(repo):
    aid = repo.create_assignment("試合", Kind.video, "t", "", "standard", False,
                                 domain="sports", sport="soccer")
    p = work_progress(repo, aid)
    assert p.domain == "sports" and p.steps[0] == "試合・練習を決める"


# --- never stuck: stopped states with an action, step 3 optional ---

LATER = timedelta(hours=1)


def later():
    return datetime.now(timezone.utc) + LATER


def test_tracking_step_three_label():
    assert step_labels("video", "tracking", "school")[2] == "人を確かめる（任意）"
    assert step_labels("video", "tracking", "sports")[2] == "人を確かめる（任意）"
    assert step_labels("video", "paper", "school")[2] == "参加者を確認"


def test_work1_shape_is_stopped_with_reselect(repo):
    """No roster, 47 tracked people all saved as 対象外, behavior 0 events, contribution 0
    judgments, nobody evaluated, no jobs, last activity an hour ago."""
    aid = video_work(repo)
    sid, _ = group(repo, aid, students=())
    labels = [f"P{n}" for n in range(1, 48)]
    track(repo, sid, labels=labels)
    save_tracked_identity(repo, sid, "cam1", {lb: None for lb in labels})
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    p = work_progress(repo, aid, now=later())
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason, sub.action) == ("wait", "stopped", NOBODY,
                                                             "reselect")
    assert "しています" not in sub.reason and "しています" not in p.reason
    assert (p.current, p.stopped, p.your_turn) == (4, True, False)


def confirmed_with_behavior(repo, events):
    aid = video_work(repo)
    sid, subjects = group(repo, aid)
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": events}, camera="cam1", upstream=[ident.id])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    return aid, sid, subjects


def event(participant):
    return {"video_time": "00:01", "participant": participant, "observed_behavior": "話す",
            "utterance": "", "speaker": "unknown", "evidence_source": "video",
            "confidence": "high", "event_id": "cam1-0001-01", "camera": "cam1",
            "task_elapsed": "00:01"}


def test_zero_events_is_stopped_with_rerun(repo):
    aid, sid, subjects = confirmed_with_behavior(repo, [event("someone_else")])
    p = work_progress(repo, aid)                       # just finished: still preparing
    assert (p.subs[0].status, p.subs[0].action) == ("processing", None)
    p = work_progress(repo, aid, now=later())
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason, sub.action) == ("wait", "stopped", NO_EVENTS,
                                                             "rerun")
    assert "しています" not in p.reason


def test_events_of_an_excluded_person_only_are_zero_events(repo):
    aid, sid, subjects = confirmed_with_behavior(repo, [event("Pa")])
    p = work_progress(repo, aid, now=later())
    assert p.subs[0].reason != NO_EVENTS and p.subs[0].action is None   # as today
    repo.set_excluded(subjects["Pa"], True)
    p = work_progress(repo, aid, now=later())
    assert (p.subs[0].reason, p.subs[0].action) == (NO_EVENTS, "rerun")


def test_zero_events_with_a_stale_behavior_is_not_no_events(repo):
    aid, sid, subjects = confirmed_with_behavior(repo, [])
    track(repo, sid)                                   # tracked again: the behavior is stale
    repo.set_identity_confirmed(sid, True)
    p = work_progress(repo, aid, now=later())
    assert p.subs[0].reason != NO_EVENTS


def test_only_insufficient_feedback_with_zero_events_is_stopped_with_rerun(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:2])
    tracked_to_review(repo, sid, subjects, pfbs={"Pa": INSUFFICIENT, "Pb": INSUFFICIENT})
    p = work_progress(repo, aid, now=later())
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason, sub.action) == ("wait", "stopped", NO_EVENTS,
                                                             "rerun")
    assert (p.current, p.stopped, p.your_turn) == (4, True, False)


def test_zero_events_with_written_feedback_is_left_to_review(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:2])
    tracked_to_review(repo, sid, subjects, pfbs={"Pa": INSUFFICIENT})   # Pb: generated
    p = work_progress(repo, aid, now=later())
    assert (p.current, p.your_turn, p.subs[0].action) == (5, True, None)


def test_auto_identity_never_waits_for_the_coach(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, students=())
    track(repo, sid)
    auto_people.auto_identity(repo, sid, "cam1")        # not confirmed (yet)
    p = work_progress(repo, aid)
    assert (p.current, p.your_turn, p.subs[0].step) == (4, False, "wait")
    assert "待っています" not in p.reason
    p = work_progress(repo, aid, now=later())          # never confirmed: stopped, not waiting
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.action) == ("wait", "stopped", "reselect")
    assert (p.current, p.your_turn) == (4, False)
    assert "待っています" not in sub.reason and "しています" not in sub.reason
    assert p.reachable(3)                              # the mapping grid stays reachable


def test_auto_identity_with_nobody_evaluated_is_stopped(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, students=())
    track(repo, sid)
    auto_people.auto_identity(repo, sid, "cam1")
    for s in repo.list_subjects(sid):
        repo.set_excluded(s["id"], True)
    p = work_progress(repo, aid, now=later())
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason, sub.action) == ("wait", "stopped", NOBODY,
                                                             "reselect")


def test_nobody_tracked_is_stopped_without_an_action(repo):
    """Every camera's track run has 0 labels: choosing the people again cannot help."""
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=("cam1", "cam2"), students=())
    for cam in ("cam1", "cam2"):
        track(repo, sid, cam, labels=())
        auto_people.auto_identity(repo, sid, cam)
    p = work_progress(repo, aid, now=later())
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason, sub.action) == ("wait", "stopped",
                                                             NO_ONE_TRACKED, None)
    assert NO_ONE_TRACKED == ("動画に人が見つかりませんでした。"
                              "管理用の画面でやり直してください")
    repo.set_identity_confirmed(sid, True)              # confirmed: the same
    sub = work_progress(repo, aid, now=later()).subs[0]
    assert (sub.status, sub.reason, sub.action) == ("stopped", NO_ONE_TRACKED, None)


def test_one_camera_with_people_keeps_reselect(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=("cam1", "cam2"), students=())
    track(repo, sid, "cam1", labels=())
    track(repo, sid, "cam2")
    for cam in ("cam1", "cam2"):
        auto_people.auto_identity(repo, sid, cam)
    for s in repo.list_subjects(sid):
        repo.set_excluded(s["id"], True)
    sub = work_progress(repo, aid, now=later()).subs[0]
    assert (sub.reason, sub.action) == (NOBODY, "reselect")


def test_a_camera_without_a_mapping_still_waits_for_the_coach(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=("cam1", "cam2"), students=())
    track(repo, sid, "cam1")
    track(repo, sid, "cam2")
    auto_people.auto_identity(repo, sid, "cam1")        # cam2's automatic selection failed
    p = work_progress(repo, aid, now=later())
    assert (p.current, p.your_turn, p.subs[0].step) == (3, True, "people")
    assert p.reason == "4人の名前を選ぶのを待っています（カメラ2）"


def test_step_labels_speak_the_profile_words():
    from pfb_app.ui.wizard.progress import step_labels
    assert step_labels("video", "tracking", "meeting")[0] == "会議を決める"
    assert step_labels("video", "tracking", "workshop")[0] == "研修を決める"
    assert step_labels("video", "paper", "presentation")[2] == "発表者を確認"
    assert step_labels("video", "tracking", "general") == TRACKING_STEPS
    assert step_labels("document", "paper", "meeting") == DOCUMENT_STEPS


def test_a_meeting_work_has_meeting_steps(repo):
    aid = repo.create_assignment("定例会", "video", "週の定例会", "", "standard", True,
                                 profile="meeting")
    assert work_progress(repo, aid).steps[0] == "会議を決める"
