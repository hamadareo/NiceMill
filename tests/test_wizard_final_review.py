"""The final review's fix wave (findings 1-15 and T6-1..5).

Only synthetic data: invented names and texts, tiny black PNGs, throwaway databases."""
import importlib.util
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit import config as st_config
from streamlit.testing.v1 import AppTest

import test_wizard_progress as tp
from pfb_app.domain.models import Kind, Stage
from pfb_app.services import deletion, review
from pfb_app.services.assignments import mark_pdf_downloaded
from pfb_app.services.review import confirm_identity, save_tracked_identity
from pfb_app.storage import db as db_module
from pfb_app.storage import files
from pfb_app.ui.common import upload_note
from pfb_app.ui.wizard import live, progress, router, step_pdf, step_people, step_review, \
    step_wait, theme
from pfb_app.ui.wizard.progress import work_progress
from test_wizard_ui_basic import (UI, app, doc_assignment, env, html,  # noqa: F401 (env)
                                  ok, primaries, screen_text, shown_step, video_assignment)
from test_wizard_ui_people_wait import (OUTSIDE, fill, finish_all, group, job, primary, save,
                                        texts, track, worker_alive)
from test_wizard_ui_review_pdf import btn, has_button, reviewable, subject

ROOT = Path(__file__).resolve().parents[1]


# --- 1. a failed feedback job keeps the group at step 5 ---------------------------------------

def test_a_failed_feedback_job_keeps_the_group_at_step_5_with_its_counts(env):
    _, repo = env
    aid = tp.video_work(repo)
    sid, subjects = tp.group(repo, aid)
    tp.tracked_to_review(repo, sid, subjects)
    review.approve(repo, subjects["Pa"])
    tp.job(repo, sid, Stage.pfb, "failed", subject_id=subjects["Pb"],
           error="フィードバックを書けませんでした")
    p = work_progress(repo, aid)
    sub = p.subs[0]
    assert (sub.step, sub.status) == ("review", "stopped")
    assert sub.reason == "フィードバックを書けませんでした（文章づくり）" and sub.action == "retry"
    assert (p.approved, p.total, p.current) == (1, 4, 5)
    assert p.reachable(5) and p.reachable(6)
    # an earlier stage that failed still waits at step 4
    sid2, subjects2 = tp.group(repo, aid, label="G2")
    tp.tracked_to_review(repo, sid2, subjects2)
    tp.job(repo, sid2, Stage.contribution, "failed", camera="cam1", error="判定できませんでした")
    p = work_progress(repo, aid)
    assert (p.subs[1].step, p.subs[1].status) == ("wait", "stopped")


# --- 2. a group whose PDF is downloaded is done while another is analysing ---------------------

def test_a_downloaded_group_is_done_while_another_is_still_analysing(env):
    _, repo = env
    aid = tp.video_work(repo)
    sid, subjects = tp.group(repo, aid, label="1班")
    tp.tracked_to_review(repo, sid, subjects)
    for subj in subjects.values():
        review.approve(repo, subj)
    sid2, _ = tp.group(repo, aid, label="2班")
    tp.track(repo, sid2)
    tp.job(repo, sid2, Stage.track, "running", camera="cam1")
    mark_pdf_downloaded(repo, aid)
    p = work_progress(repo, aid)
    done = p.subs[0]
    assert (done.status, done.reason) == ("done", "PDFを受け取りました")
    assert not p.your_turn and not p.complete
    assert "参加者に渡すPDFを受け取れます" not in p.reason
    review.unapprove(repo, subjects["Pa"])       # approved again after the download
    review.approve(repo, subjects["Pa"])
    p = work_progress(repo, aid)
    assert (p.subs[0].status, p.subs[0].reason) == ("your_turn", "参加者に渡すPDFを受け取れます")


# --- 3. the old export page records the download; the backfill script --------------------------

def approved_document(repo, title="レポート", names=("佐藤",), insufficient=()):
    aid = repo.create_assignment(title, "document", "調べる", "", "standard", False)
    ids = []
    for name in names:
        sid = repo.create_submission(aid, name, {"source_path": "/x/a.pdf"})
        subj = repo.add_subject(sid, None, name, None, {})
        tp.document_to_review(repo, sid, subj,
                              tp.INSUFFICIENT if name in insufficient else tp.PFB)
        ids.append(subj)
    return aid, ids


def test_the_old_export_page_records_the_student_download(env):
    _, repo = env
    aid, (subj,) = approved_document(repo)
    review.approve(repo, subj)
    at = AppTest.from_file(str(UI / "pages" / "export.py"), default_timeout=60).run()
    at.button(key="make-teacher").click().run()
    ok(at)
    at.download_button(key="dl-teacher").click().run()
    ok(at)
    assert not repo.get_assignment(aid).get("pdf_downloaded_at")     # the teacher PDF: no
    at.button(key="make-student").click().run()
    ok(at)
    at.download_button(key="dl-student").click().run()
    ok(at)
    assert repo.get_assignment(aid)["pdf_downloaded_at"]


def _backfill():
    spec = importlib.util.spec_from_file_location(
        "backfill_pdf_downloaded", ROOT / "scripts" / "backfill_pdf_downloaded.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_backfill_sets_settled_works_only_and_is_idempotent(env, capsys):
    settings, repo = env
    settled, (a, b) = approved_document(repo, "済んだ作業", ("佐藤", "鈴木"),
                                        insufficient=("鈴木",))
    review.approve(repo, a)
    open_, (c, d) = approved_document(repo, "途中の作業", ("田中", "高橋"))
    review.approve(repo, c)                                   # d still to check
    short, _ = approved_document(repo, "根拠不足だけ", ("伊藤",), insufficient=("伊藤",))
    known, (e,) = approved_document(repo, "受け取り済み", ("山本",))
    review.approve(repo, e)
    repo.conn.execute("UPDATE assignments SET pdf_downloaded_at='2026-01-01T00:00:00+00:00'"
                      " WHERE id=?", (known,))
    backfill, db = _backfill(), str(files.db_path(settings))
    assert backfill.main(["--db", db]) == 0                  # a dry run by default
    out = capsys.readouterr().out
    assert "済んだ作業" in out and "途中の作業" not in out and "根拠不足だけ" not in out
    assert "受け取り済み" not in out
    assert not repo.get_assignment(settled).get("pdf_downloaded_at")
    assert backfill.main(["--db", db, "--apply"]) == 0
    capsys.readouterr()
    approved_at = repo.get_subject(a)["approved_at"]
    assert repo.get_assignment(settled)["pdf_downloaded_at"] == approved_at
    assert not repo.get_assignment(open_).get("pdf_downloaded_at")
    assert not repo.get_assignment(short).get("pdf_downloaded_at")
    assert repo.get_assignment(known)["pdf_downloaded_at"] == "2026-01-01T00:00:00+00:00"
    assert work_progress(repo, settled).complete
    assert backfill.main(["--db", db, "--apply"]) == 0       # idempotent
    assert "済んだ作業" not in capsys.readouterr().out
    assert repo.get_assignment(settled)["pdf_downloaded_at"] == approved_at


def test_the_backfill_never_creates_a_database(tmp_path, capsys):
    missing = tmp_path / "none.db"
    assert _backfill().main(["--db", str(missing)]) != 0
    assert not missing.exists()


# --- 4. a feedback built on an outdated contribution: 詳細画面 first ----------------------------

def test_an_outdated_contribution_gates_the_student_instead_of_looping(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    beh = repo.get_run(con.upstream_run_ids[0])
    save(repo, sid, Stage.behavior, beh.output, camera="cam1", upstream=beh.upstream_run_ids)
    students = step_review.students_of(repo, repo.get_submission(sid), Kind.video)
    assert {s.state for s in students} == {"gated"}
    assert all(s.gate == progress.REDO_CONTRIBUTION for s in students)
    assert progress.REDO_CONTRIBUTION == ("解析の途中の結果が直されたため、"
                                          "管理用の画面で『貢献の判定』から作り直してください")
    p = work_progress(repo, aid)
    assert p.subs[0].reason == progress.REDO_CONTRIBUTION and p.subs[0].step == "review"
    at = ok(app(session={"assignment_id": aid}))
    assert progress.REDO_CONTRIBUTION in texts(at, "warning")
    assert step_review.GATE_LINK in [link.proto.label for link in at.get("page_link")]
    assert not has_button(at, step_review.REDO) and not has_button(at, step_review.FIX)


def test_a_contribution_redone_since_is_plain_stale_and_can_be_redone(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    save(repo, sid, Stage.contribution, con.output, camera="cam1",
         upstream=con.upstream_run_ids)                     # a fresh newer contribution
    students = step_review.students_of(repo, repo.get_submission(sid), Kind.video)
    assert {s.state for s in students} == {"stale"}
    assert "内容が古くなった人がいます" in work_progress(repo, aid).subs[0].reason


# --- 5. a stopped group shows why above the form ----------------------------------------------

def test_a_stopped_tracking_group_shows_the_reason_above_the_form(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    track(settings, repo, sid, "cam1")
    job(repo, sid, Stage.track, "failed", camera="cam2", error="人物を見つけられませんでした")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert any("人物を見つけられませんでした（人を見つける・カメラ2）" in e for e in texts(at, "error"))
    assert at.selectbox                                      # the cam1 form is still there
    assert router.DETAILS in [link.proto.label for link in at.get("page_link")]


def test_a_stopped_paper_group_shows_the_reason_above_the_table(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, engine="paper")
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.identity, {"participants": [
        {"participant_id": "Pa", "appearance": "赤い服", "initial_position": "左前",
         "verification_points": [], "identity_confidence": "high", "verified": True,
         "notes": ""}]}, camera="cam1", upstream=[prep.id])
    job(repo, sid, Stage.transcribe, "failed", error="音声を読み取れませんでした")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert any("音声を読み取れませんでした（文字起こし）" in e for e in texts(at, "error"))
    assert at.dataframe


# --- 6. one snapshot per page run and per tick ------------------------------------------------

STUDENTS4 = (("Pa", "花子"), ("Pb", "太郎"), ("Pc", "次郎"), ("Pd", "三郎"))
CAMS = ("cam1", "cam2")
# one step-4 tick on 10 groups x 4 students x 2 cameras: 3092 queries before the fix wave;
# after it one snapshot (the assignment, BEGIN, 4 reads, COMMIT) and a cold stage_durations
# cache: measured 8, bound 8 + 20 %
TICK_QUERIES = 9


def big_work(settings, repo, groups=10):
    """10 tracking groups, 4 students, 2 cameras each, every one with its feedback written:
    every third writes two texts again, the next has a camera's records redone."""
    aid = video_assignment(repo, "大きな作業")
    for g in range(groups):
        meta = {"cameras": {c: f"/v/{c}.mp4" for c in CAMS}, "start_sec": 0.0,
                "end_sec": None, "auto_transcribe": False, "engine": "pfb3x3"}
        sid = repo.create_submission(aid, f"{g + 1}班", meta)
        subj = [repo.add_subject(sid, pid, name, "外見", {}) for pid, name in STUDENTS4]
        for cam in CAMS:
            track(settings, repo, sid, cam)
            save_tracked_identity(repo, sid, cam, {"P1": "Pa", "P2": "Pb", "P3": "Pc",
                                                   "P4": "Pd"})
        confirm_identity(repo, sid)
        finish_all(repo, sid)
        cons = []
        for cam in CAMS:
            ident = repo.latest_run(sid, Stage.identity, cam)
            beh = save(repo, sid, Stage.behavior, {"events": []}, camera=cam,
                       upstream=[ident.id])
            cons.append(save(repo, sid, Stage.contribution, {"judgments": []}, camera=cam,
                             upstream=[beh.id]))
        for s in subj:
            save(repo, sid, Stage.pfb, tp.PFB, subject_id=s, upstream=[c.id for c in cons])
        if g % 3 == 0:
            job(repo, sid, Stage.pfb, "running", subject_id=subj[0])
            job(repo, sid, Stage.pfb, "queued", subject_id=subj[1])
        elif g % 3 == 1:
            job(repo, sid, Stage.behavior, "running", camera="cam2")
    return aid


def test_one_wait_tick_reads_the_work_once(env):
    settings, repo = env
    worker_alive(settings)
    aid = big_work(settings, repo)
    db_module.clear_stage_durations_cache()
    queries = []
    repo.conn.set_trace_callback(queries.append)
    try:
        live.tick(repo, aid, "wait", step_wait._signature(aid),
                  step_wait._board(aid, settings), step_wait._moved)
    finally:
        repo.conn.set_trace_callback(None)
    warm = []
    repo.conn.set_trace_callback(warm.append)
    try:          # right after it: the stage durations are still known
        live.tick(repo, aid, "wait", step_wait._signature(aid),
                  step_wait._board(aid, settings), step_wait._moved)
    finally:
        repo.conn.set_trace_callback(None)
    print(f"one step-4 tick: {len(queries)} queries cold, {len(warm)} warm")
    assert len(queries) <= TICK_QUERIES, len(queries)
    assert len(warm) < len(queries), (len(warm), len(queries))


# all_work on 3 works of big_work(groups=2), cold stage durations: measured 25 statements at
# the branch base b05502e (1 list + 3 x 8 snapshot reads, no progress model); the analysis may
# add at most 2: it reads the stage durations once per method (+1), and the home render's
# pending_deletions (+1) is counted too: measured 27
HOME_BASE = 25


def test_home_reads_each_work_once(env):
    settings, repo = env
    worker_alive(settings)
    for _ in range(3):
        big_work(settings, repo, groups=2)
    db_module.clear_stage_durations_cache()
    queries = []
    repo.conn.set_trace_callback(queries.append)
    try:      # what home.render reads
        works = progress.all_work(repo)
        deletion.pending_deletions(repo)
    finally:
        repo.conn.set_trace_callback(None)
    print(f"home: {len(queries)} queries (base {HOME_BASE})")
    assert len(works) == 3 and all(w.analysis is not None for w in works)
    assert len(queries) <= HOME_BASE + 2, len(queries)


def test_the_router_builds_one_snapshot_and_the_screens_share_it(env, monkeypatch):
    settings, repo = env
    worker_alive(settings)
    aid = big_work(settings, repo, groups=2)
    made = []
    real = progress._Snapshot.__init__

    def counting(self, *a, **k):
        made.append(1)
        real(self, *a, **k)
    monkeypatch.setattr(progress._Snapshot, "__init__", counting)
    for step in (4, 5, 6):
        made.clear()
        ok(app(session={"assignment_id": aid, "step": step, "submission_id": None}))
        assert len(made) == 1, (step, len(made))


# --- 7. 「いまは押すものはありません」 only when nothing is stopped -------------------------------

def test_nothing_to_press_is_not_said_when_a_group_has_stopped(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    for label, failed in (("A班", False), ("B班", True)):
        sid = group(repo, aid, label)
        track(settings, repo, sid)
        save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": "Pc",
                                                  "P4": None})
        confirm_identity(repo, sid)                    # behavior queued
        if failed:
            repo.conn.execute("UPDATE jobs SET status='failed', error='止まりました'"
                              " WHERE submission_id=?", (sid,))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    assert live.NOTHING_TO_PRESS not in [c.value for c in at.caption]


# --- 8. the mapping choices -----------------------------------------------------------------

def test_kept_choices_never_hide_a_mapping_saved_elsewhere(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": None}))
    fill(at, run, {"P1": "Pc"})                       # chosen, not saved
    # 詳細画面 saves another mapping; the teacher comes back to step 3
    save_tracked_identity(repo, sid, "cam1", {"P1": "Pb", "P2": "Pa", "P3": "Pc", "P4": None})
    at.session_state["wizard"] = {"assignment_id": None, "step": None, "submission_id": None}
    ok(at.run())
    at.session_state["wizard"] = {"assignment_id": aid, "step": 3, "submission_id": None}
    ok(at.run())
    assert at.selectbox(key=f"pfb_map_{run.id}_P1").value == "Pb"
    assert at.selectbox(key=f"pfb_map_{run.id}_P2").value == "Pa"


def test_the_choices_are_dropped_after_a_decision(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    run = track(settings, repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    fill(at, run, {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": OUTSIDE})
    assert at.session_state[step_people._CHOICES]
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["identity_confirmed"]
    assert not any(key[0] == run.id for key in at.session_state[step_people._CHOICES])


# --- 9. a pending review action is dropped outside step 5 ------------------------------------

def test_a_review_action_is_dropped_outside_the_review_step(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    review.approve(repo, hanako["id"])
    action = ("approve", subject(repo, sid, "Pb")["id"], None)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None},
                extra={router.REVIEW_ACTION: action}))
    assert shown_step(at) == 6
    assert router.REVIEW_ACTION not in at.session_state
    assert step_review._ACTION == router.REVIEW_ACTION


# --- 10. a tick on a deleted work reruns the page --------------------------------------------

class _Rerun(Exception):
    pass


def _rerun():
    raise _Rerun


def test_a_tick_on_a_deleted_work_or_group_reruns_the_page(env, monkeypatch):
    settings, repo = env
    monkeypatch.setattr(live.st, "rerun", _rerun)
    aid = video_assignment(repo)
    sid = group(repo, aid)
    gone = video_assignment(repo, "消した作業")
    repo.delete_assignment(gone)
    with pytest.raises(_Rerun):
        live.tick(repo, gone, "wait", step_wait._signature(gone), step_wait._board(gone, settings))
    repo.conn.execute("DELETE FROM submissions WHERE id=?", (sid,))
    with pytest.raises(_Rerun):
        live.tick(repo, aid, "people", step_people._signature(aid, sid, True),
                  step_people._live(aid, sid, settings, "video"))


# --- 11. the shown group is dropped when the step changes ------------------------------------

def test_the_stepper_drops_the_picked_group(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid, "A班")
    b = reviewable(settings, repo, aid, "B班")
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": b}))
    next(x for x in at.button if x.key == "pfb_nav_4").click().run()
    ok(at)
    assert shown_step(at) == 4
    assert at.session_state["wizard"]["submission_id"] is None


def test_moving_on_by_itself_drops_the_picked_group(monkeypatch):
    s = {"assignment_id": 1, "step": None, "submission_id": 7}
    monkeypatch.setattr(router, "state", lambda: s)
    step_wait._moved((4, ((7, "wait", "processing"),), None),
                     (5, ((7, "review", "your_turn"),), 5))
    assert s == {"assignment_id": 1, "step": 5, "submission_id": None}


# --- 12. only students with too little evidence: their short text is the primary (L6) ---------

def test_a_group_of_insufficient_students_offers_writing_as_the_primary(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid, insufficient=("Pa", "Pb", "Pc"))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    assert primaries(at) == [step_review.USE_SHORT]
    assert not has_button(at, step_review.CONTINUE)
    btn(at, step_review.REWRITE).click().run()
    ok(at)
    assert primaries(at) == [step_review.FIX_APPROVE]


# --- 13. the download is recorded once the student PDF is made --------------------------------

def test_the_student_pdf_records_the_download_only_when_it_is_made(env, monkeypatch, caplog):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    with pytest.raises(step_pdf.ExportError):
        step_pdf.pdf_data(aid, "student")()
    assert "PDF" in caplog.text
    assert not repo.get_assignment(aid).get("pdf_downloaded_at")
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    step_pdf.pdf_data(aid, "teacher")()
    assert not repo.get_assignment(aid).get("pdf_downloaded_at")
    assert step_pdf.pdf_data(aid, "student")()[:4] == b"%PDF"
    assert repo.get_assignment(aid)["pdf_downloaded_at"]
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    student = next(b for b in at.download_button if b.label == step_pdf.STUDENT)
    assert student.proto.type == "primary"


# --- 14. stage durations are cached for five minutes -----------------------------------------

def test_stage_durations_are_cached_for_five_minutes(env, monkeypatch):
    settings, repo = env
    db_module.clear_stage_durations_cache()
    clock = [1000.0]
    monkeypatch.setattr(db_module, "_clock", lambda: clock[0])
    sid = group(repo, video_assignment(repo), "T班")
    save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", duration=60)
    assert repo.stage_durations(Stage.behavior, "pfb3x3") == [60]
    save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", duration=90)
    clock[0] += 299
    assert repo.stage_durations(Stage.behavior, "pfb3x3") == [60]
    assert repo.stage_durations(Stage.behavior, "paper") == []      # its own key
    clock[0] += 2
    assert repo.stage_durations(Stage.behavior, "pfb3x3") == [90, 60]


# --- 15 / T6-1. the upload limit is the server's -------------------------------------------------

def test_the_upload_note_reads_the_servers_limit():
    old = st.get_option("server.maxUploadSize")
    try:
        st_config.set_option("server.maxUploadSize", 321)
        assert upload_note("CSVファイル") == "CSVファイルを選べます（1ファイル最大 321 MB）。"
    finally:
        st_config.set_option("server.maxUploadSize", old)


def test_the_old_pages_say_the_upload_limit_too(env):
    settings, repo = env
    limit = st.get_option("server.maxUploadSize")

    def notes(at):
        return [c.value for c in at.caption if f"1ファイル最大 {limit} MB" in c.value]
    at = ok(AppTest.from_file(str(UI / "pages" / "assignments.py"), default_timeout=60).run())
    assert len(notes(at)) == 1
    doc_assignment(repo)
    vid = video_assignment(repo)
    at = ok(AppTest.from_file(str(UI / "pages" / "submissions.py"), default_timeout=60).run())
    assert len(notes(at)) == 2                       # the documents and the roster
    at.selectbox[0].select(vid).run()
    ok(at)
    assert len(notes(at)) == 1                       # one camera's video


# --- T6-2, T6-4, T6-5 -----------------------------------------------------------------------

def test_the_stop_button_is_not_hidden_globally_but_the_label_stays():
    css = theme.theme_css()
    assert not re.search(r'\[data-testid="stStatusWidget"\]\s*button\s*\{\s*display:\s*none', css)
    assert 'content: "処理中…"' in css


def test_the_reduced_motion_block_shortens_animations_and_transitions():
    css = theme.theme_css()
    start = css.index("@media (prefers-reduced-motion: reduce)")
    block = css[start:css.index("\n}\n", start)]
    assert re.search(r"animation-duration:\s*0\.01ms", block)
    assert re.search(r"transition-duration:\s*0\.01ms", block)


def test_the_guide_points_to_the_home_button_where_it_is():
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "左上の「ホーム」" not in guide
    assert "左のメニューの「ホーム」" in guide
    app_py = (UI / "app.py").read_text(encoding="utf-8")
    assert 'with st.sidebar:' in app_py and 'st.button("ホーム"' in app_py
