"""How far a work's analysis has got: stages, weights, sub-progress and the held bar."""
import pytest

from pfb_app.domain.models import Stage
from pfb_app.services.review import approve, confirm_identity, save_tracked_identity
from pfb_app.ui.wizard.analysis import (DONE, FALLBACK_WEIGHTS, MOVING, PIPELINES, QUEUED,
                                        REDOING, REVIEW_NAME, RUNNING, STAGE_NAMES, STOPPED,
                                        WAITING_GPU, Analysis, StageLine, hold, stage_weights,
                                        submission_analysis, target_counts, work_analysis)
from pfb_app.domain.models import Kind
from pfb_app.ui.wizard.progress import NOBODY, work_progress
from test_wizard_progress import (STUDENTS, document_work, finish_all, group, job, later,
                                  paper_to_identity, repo, save, track, tracked_to_review,
                                  video_work)

THREE = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": None}
TRACKING_NAMES = ["内容の確認", "準備", "人を見つける", "評価する人を選ぶ", "したことの書き出し",
                  "よかった点を選ぶ", "文章づくり"]


def confirmed(repo, cameras=("cam1",), label="G1", aid=None):
    """A tracking group of three students, tracked, mapped and confirmed on every camera; the
    behavior jobs the confirmation queued are finished (no behavior run yet)."""
    aid = aid or video_work(repo)
    sid, subjects = group(repo, aid, cameras=cameras, label=label, students=STUDENTS[:3])
    for cam in cameras:
        track(repo, sid, cam)
        save_tracked_identity(repo, sid, cam, THREE)
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    return aid, sid, subjects


def behavior_running(repo, sid, progress=None, note=None, camera="cam1", status="running",
                     error=None):
    jid = job(repo, sid, Stage.behavior, status, camera=camera, error=error)
    repo.set_job_progress(jid, progress, note)
    return jid


def behavior_done(repo, sid, camera="cam1"):
    ident = repo.latest_run(sid, Stage.identity, camera)
    return save(repo, sid, Stage.behavior, {"events": []}, camera=camera, upstream=[ident.id])


def sub_analysis(repo, aid, now=None, n=0):
    return work_progress(repo, aid, now=now).subs[n].analysis


# --- constants and weights ---

def test_names_and_pipelines():
    assert [STAGE_NAMES[k] for k in PIPELINES["pfb3x3"]] == TRACKING_NAMES
    assert REVIEW_NAME == "確認と承認" and MOVING == "進んでいます" and REDOING == "やり直しています"
    assert (DONE, RUNNING, QUEUED, WAITING_GPU, STOPPED) == (
        "done", "running", "queued", "waiting_gpu", "stopped")
    assert PIPELINES["paper"] == ("detect", "prepare", "transcribe", "identity", "behavior",
                                  "contribution", "pfb")
    assert PIPELINES["document"] == ("prepare", "evidence", "contribution", "pfb")
    assert {FALLBACK_WEIGHTS[k] for k in ("transcribe", "identity")} == {1}
    assert FALLBACK_WEIGHTS["evidence"] == 3


def test_fallback_weights_when_history_is_missing():
    assert stage_weights({}, PIPELINES["pfb3x3"]) == {
        "detect": 1, "prepare": 1, "track": 4, "people": 0.5, "behavior": 10,
        "contribution": 3, "pfb": 3}


def test_history_weights_use_medians_and_people_follows_track():
    d = {"detect": [10], "prepare": [20], "track": [80, 120, 100], "behavior": [600],
         "contribution": [60], "pfb": [30]}
    w = stage_weights(d, PIPELINES["pfb3x3"])
    assert w["track"] == 100 and w["people"] == 12.5 and w["behavior"] == 600
    assert w == {"detect": 10, "prepare": 20, "track": 100, "people": 12.5, "behavior": 600,
                 "contribution": 60, "pfb": 30}


def test_mixed_history_falls_back_entirely():
    assert stage_weights({"track": [100]}, PIPELINES["pfb3x3"])["track"] == 4


def test_transcribe_without_history_follows_prepare():
    d = {"detect": [10], "prepare": [20], "identity": [30], "behavior": [600],
         "contribution": [60], "pfb": [30]}
    w = stage_weights(d, PIPELINES["paper"])
    assert w["transcribe"] == 20 and w["identity"] == 30
    assert stage_weights({**d, "transcribe": [50]}, PIPELINES["paper"])["transcribe"] == 50


# --- one tracking group ---

def test_tracking_group_in_behavior_counts_rows(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, 0.3, "12 / 40 区間")
    p = work_progress(repo, aid)
    a = p.subs[0].analysis
    assert [s.name for s in a.stages] == TRACKING_NAMES
    assert [s.key for s in a.stages] == list(PIPELINES["pfb3x3"])
    assert a.percent == 42                      # (1+1+4+0.5+10*0.3)/22.5
    beh = a.stages[4]
    assert (beh.status, beh.note, beh.known) == ("running", "12 / 40 区間", True)
    assert beh.percent == 30
    assert [s.status for s in a.stages[:4]] == ["done"] * 4
    assert a.stages[5].status == "queued" and a.review.note == "承認 0 / 全 3"
    assert (a.review.name, a.review.status, a.review.fraction) == (REVIEW_NAME, "queued", 0)
    assert a.current is beh and a.status == "running" and a.active and not a.finished
    assert not a.redo
    assert p.subs[0].progress_fraction == a.fraction
    # one group: the work's analysis is the group's
    assert p.analysis.percent == 42 and p.progress_fraction == p.analysis.fraction
    assert p.analysis.stages[4].note == "12 / 40 区間"     # no label prefix for one group


def test_unknown_sub_progress_holds_the_stage_start(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid)
    a = sub_analysis(repo, aid)
    beh = a.stages[4]
    assert a.percent == 28                      # (1+1+4+0.5)/22.5
    assert (beh.status, beh.known, beh.note) == ("running", False, MOVING)


def test_a_progress_without_a_note_is_known(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, 0.5, None)
    beh = sub_analysis(repo, aid).stages[4]
    assert (beh.known, beh.note, beh.fraction) == (True, MOVING, 0.5)


def test_gpu_wait_keeps_the_bar_still_and_says_why(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, status="waiting_gpu", error="空き 3GB < 必要 20GB")
    p = work_progress(repo, aid)
    a = p.subs[0].analysis
    beh = a.stages[4]
    assert beh.status == "waiting_gpu" and a.percent == 28
    assert beh.note == p.subs[0].reason
    assert beh.note.startswith("順番を待っています（したことの書き出し）")
    assert a.status == "waiting_gpu" and a.active


def test_the_track_stage_shows_its_note(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, "running", camera="cam1")
    repo.set_job_progress(jid, 0.25, "人物を追跡しています")
    a = sub_analysis(repo, aid)
    tr = a.stages[2]
    assert (tr.key, tr.status, tr.note, tr.fraction) == ("track", "running",
                                                         "人物を追跡しています", 0.25)
    assert a.stages[3].status == "queued"     # people: the track is not done yet
    assert a.percent == 13                    # (1+1+4*0.25)/22.5


def test_people_run_after_the_track_until_the_mapping_is_current(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, students=())
    track(repo, sid)
    a = sub_analysis(repo, aid)       # the coach's turn: people wait for them
    assert a.stages[3].status == "queued"
    from pfb_app.services import auto_people
    auto_people.auto_identity(repo, sid, "cam1")
    a = sub_analysis(repo, aid)       # automatic people current: done
    assert a.stages[3].status == "done"


def test_cameras_and_students_are_counted(repo):
    aid, sid, _ = confirmed(repo, cameras=("cam1", "cam2"))
    behavior_done(repo, sid, "cam1")
    behavior_running(repo, sid, 0.5, None, camera="cam2")
    a = sub_analysis(repo, aid)
    beh = a.stages[4]
    assert beh.fraction == 0.75 and beh.note == "1 / 2 カメラ"
    assert set(a.cameras) == {"cam1", "cam2"} and a.cameras["cam1"] > a.cameras["cam2"]

    aid = video_work(repo, title="2")
    sid, subjects = group(repo, aid, students=STUDENTS[:3])
    tracked_to_review(repo, sid, subjects)
    pids = list(subjects)
    # two texts not written yet: one being written now, one waiting
    for pid in pids[1:]:
        repo.conn.execute("DELETE FROM stage_runs WHERE subject_id=?", (subjects[pid],))
    job(repo, sid, Stage.pfb, "running", subject_id=subjects[pids[1]])
    pfb = sub_analysis(repo, aid).stages[-1]
    assert (pfb.status, pfb.note, pfb.known) == ("running", "1 / 3 人", True)
    assert pfb.fraction == pytest.approx(1 / 3)


def test_stopped_stage_and_stopped_group(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, status="failed", error="行動の記録に失敗しました")
    p = work_progress(repo, aid)
    a = p.subs[0].analysis
    beh = a.stages[4]
    assert p.subs[0].status == "stopped"
    assert beh.status == "stopped" and beh.note == p.subs[0].reason
    assert a.status == "stopped" and not a.active and a.current is beh

    # nobody evaluated: the first stage that is not done stops with the group's reason
    aid = video_work(repo, title="2")
    sid, _ = group(repo, aid, students=())
    labels = [f"P{n}" for n in range(1, 5)]
    track(repo, sid, labels=labels)
    save_tracked_identity(repo, sid, "cam1", {lb: None for lb in labels})
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    p = work_progress(repo, aid, now=later())
    assert p.subs[0].reason == NOBODY
    a = p.subs[0].analysis
    assert [s.status for s in a.stages] == ["done"] * 6 + ["stopped"]
    assert a.stages[-1].note == NOBODY and not a.finished and a.percent < 100


def test_never_100_before_done(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:3])
    tracked_to_review(repo, sid, subjects)
    jid = job(repo, sid, Stage.pfb, "running", subject_id=subjects["Pa"])
    repo.set_job_progress(jid, 0.99, None)
    a = sub_analysis(repo, aid)
    assert a.percent <= 99 and a.fraction <= 0.99 and not a.finished
    finish_all(repo, sid)
    a = sub_analysis(repo, aid)
    assert a.fraction == 1.0 and a.finished and a.status == "done" and a.percent == 100
    assert not a.active and a.current is None
    assert a.review.status == "running"       # at the review step
    for pid in subjects:
        approve(repo, subjects[pid])
    a = sub_analysis(repo, aid)
    assert (a.review.status, a.review.fraction, a.review.note) == ("done", 1.0, "承認 3 / 全 3")


def test_redo_lowers_once_and_says_so(repo):
    aid, sid, _ = confirmed(repo)
    behavior_done(repo, sid)
    behavior_running(repo, sid)
    a = sub_analysis(repo, aid)
    beh = a.stages[4]
    assert a.redo and beh.status == "running" and beh.note == REDOING
    repo.conn.execute("UPDATE jobs SET progress=0.5, progress_note='20 / 40 区間'"
                      " WHERE status='running'")
    a = sub_analysis(repo, aid)
    assert a.stages[4].note == "やり直しています（20 / 40 区間）"
    assert a.stages[4].fraction == 0.5          # a target being redone is not done


def A(fraction, redo=False, finished=False):
    line = StageLine("pfb", "フィードバック作成", DONE if finished else RUNNING,
                     1.0 if finished else fraction)
    return Analysis(fraction, [line], StageLine("review", REVIEW_NAME, QUEUED, 0.0), redo=redo)


def test_hold_never_goes_back_without_a_redo():
    assert hold(None, A(0.5)) == 0.5
    assert hold((0.6, False), A(0.5)) == 0.6
    assert hold((0.6, False), A(0.7)) == 0.7
    assert hold((0.6, False), A(0.3, redo=True)) == 0.3
    assert hold((0.3, True), A(0.25, redo=True)) == 0.3
    assert hold((0.6, False), A(1.0, finished=True)) == 1.0


def test_work_analysis_stopped_group_wins_the_stage(repo):
    aid, sid1, _ = confirmed(repo, label="1班")
    behavior_running(repo, sid1, 0.5, "20 / 40 区間")
    _, sid2, _ = confirmed(repo, label="2班", aid=aid)
    behavior_running(repo, sid2, status="failed", error="失敗しました")
    p = work_progress(repo, aid)
    w = p.analysis
    one, two = p.subs[0].analysis, p.subs[1].analysis
    beh = w.stages[4]
    assert beh.status == "stopped" and beh.note.startswith("2班: ")
    assert w.fraction == pytest.approx((one.fraction + two.fraction) / 2)
    assert beh.fraction == pytest.approx((0.5 + 0) / 2)
    assert w.status == "stopped" and w.cameras == {}
    assert w.review.note == "承認 0 / 全 6"


def test_work_analysis_prefers_running_over_waiting_and_none_without_groups():
    def part(status, note):
        st = StageLine("behavior", "行動の記録", status, 0.5, note=note)
        return Analysis(0.5, [st], StageLine("review", REVIEW_NAME, QUEUED, 0.0),
                        cameras={"cam1": 0.5})
    w = work_analysis([("1班", part(WAITING_GPU, "GPU")), ("2班", part(RUNNING, "x"))], 0, 4)
    assert (w.stages[0].status, w.stages[0].note) == (RUNNING, "2班: x")
    assert w.review.note == "承認 0 / 全 4" and w.review.status == QUEUED
    one = work_analysis([("1班", part(RUNNING, "x"))], 1, 4)
    assert one.stages[0].note == "x" and one.cameras == {"cam1": 0.5}
    assert one.review.status == RUNNING
    assert work_analysis([], 0, 0) is None


def test_paper_and_document_pipelines(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, engine="paper")
    paper_to_identity(repo, sid)
    a = sub_analysis(repo, aid)
    assert [s.key for s in a.stages] == ["detect", "prepare", "identity", "behavior",
                                         "contribution", "pfb"]
    assert [s.name for s in a.stages][2] == "参加者の特定"
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "auto_transcribe": True})
    a = sub_analysis(repo, aid)
    assert [s.key for s in a.stages][2] == "transcribe" and a.stages[2].name == "文字起こし"

    aid, subs = document_work(repo)
    a = sub_analysis(repo, aid)
    assert [s.key for s in a.stages] == ["prepare", "evidence", "contribution", "pfb"]
    assert [s.name for s in a.stages] == ["準備", "場面の書き出し", "よかった点を選ぶ",
                                          "文章づくり"]
    assert a.review.note == "承認 0 / 全 1"


def test_old_work_without_detect_or_progress(repo):
    aid, sid, _ = confirmed(repo)
    assert not repo.conn.execute("SELECT 1 FROM jobs WHERE progress IS NOT NULL").fetchall()
    a = sub_analysis(repo, aid)
    assert a.stages[0].key == "detect" and a.stages[0].status == "done"


def test_a_running_check_is_the_detect_stage(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "check": "pending"})
    job(repo, sid, Stage.detect, "running")
    a = sub_analysis(repo, aid)
    assert a.stages[0].status == "running" and a.stages[1].status == "queued"
    finish_all(repo, sid)
    repo.update_submission_meta(sid, {**meta, "check": "ready"})
    assert sub_analysis(repo, aid).stages[0].status == "done"


def test_history_weights_reach_the_derivation(repo):
    from pfb_app.storage.db import clear_stage_durations_cache
    clear_stage_durations_cache()
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, 0.5, None)
    for stage in ("detect", "prepare", "track", "behavior", "contribution", "pfb"):
        repo.conn.execute(
            "INSERT INTO stage_runs (submission_id, stage, camera, version, source,"
            " upstream_run_ids, output, status, warnings, duration_sec, created_at)"
            " VALUES (?, ?, 'old', 1, 'model', '[]', '{}', 'ok', '[]', 100, '')",
            (sid, stage))
    clear_stage_durations_cache()
    a = sub_analysis(repo, aid)
    # every run 100 s, people 12.5, pfb 3 people at once (8 run in parallel) x 100:
    # (100*3 + 12.5 + 50) / 612.5
    assert a.fraction == pytest.approx(362.5 / 612.5)


# --- review round 1 ---

def test_history_weights_count_the_runs_of_each_stage(repo):
    d = {"detect": [10], "prepare": [20], "track": [100], "behavior": [600],
         "contribution": [60], "pfb": [30]}
    counts = {"prepare": 2, "track": 2, "behavior": 2, "contribution": 2, "pfb": 3}
    w = stage_weights(d, PIPELINES["pfb3x3"], counts)
    assert w == {"detect": 10, "prepare": 40, "track": 200, "people": 25, "behavior": 1200,
                 "contribution": 120, "pfb": 90}
    # the fallback is never scaled
    assert stage_weights({}, PIPELINES["pfb3x3"], counts)["pfb"] == 3
    aid, sid, _ = confirmed(repo, cameras=("cam1", "cam2"))
    sub = repo.get_submission(sid)
    assert target_counts(repo, sub, Kind.video) == {
        "prepare": 2, "track": 2, "identity": 2, "behavior": 2, "contribution": 2, "pfb": 3}


def test_a_requeued_or_gpu_waiting_job_keeps_its_progress(repo):
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, 0.3, "12 / 40 区間")
    assert sub_analysis(repo, aid).percent == 42
    repo.requeue_running("cpu")                     # the GPU was given back
    a = sub_analysis(repo, aid)
    assert a.percent == 42 and a.stages[4].status == "queued"
    repo.mark_waiting_gpu("cpu", "空き 3GB < 必要 20GB")
    a = sub_analysis(repo, aid)
    assert a.percent == 42 and a.stages[4].status == "waiting_gpu"


def test_a_real_redo_is_a_new_job_and_drops_once(repo):
    aid = video_work(repo)
    sid, subjects = group(repo, aid, students=STUDENTS[:3])
    tracked_to_review(repo, sid, subjects)
    before = sub_analysis(repo, aid)
    assert before.finished
    shown = hold(None, before)
    job(repo, sid, Stage.behavior, "running", camera="cam1")   # a new row: no progress yet
    a = sub_analysis(repo, aid)
    assert a.redo and a.fraction < 0.99
    shown = hold((shown, False), a)
    assert shown == a.fraction                       # lowered once
    repo.conn.execute("UPDATE jobs SET progress=0.1 WHERE status='running'")
    assert hold((shown, True), sub_analysis(repo, aid)) >= shown


def test_a_gpu_waiting_check_says_so(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    job(repo, sid, Stage.detect, "waiting_gpu", error="GPU")
    assert sub_analysis(repo, aid).stages[0].status == "waiting_gpu"
    finish_all(repo, sid)
    job(repo, sid, Stage.detect, "queued")
    assert sub_analysis(repo, aid).stages[0].status == "queued"


def test_a_group_stop_leaves_an_already_stopped_stage_alone(repo):
    aid, sid, _ = confirmed(repo)
    job(repo, sid, Stage.detect, "queued")       # a check still waiting
    behavior_running(repo, sid, status="failed", error="失敗しました")
    p = work_progress(repo, aid)
    assert p.subs[0].status == "stopped"
    a = p.subs[0].analysis
    assert [s.status for s in a.stages][:5] == ["queued", "done", "done", "done", "stopped"]


def test_a_done_group_is_all_done(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    sub = repo.get_submission(sid)
    a = submission_analysis(repo, sub, Kind.video, "pfb3x3", FALLBACK_WEIGHTS, status="done",
                            reason="完了しました", approved=4, total=4)
    assert a.finished and a.fraction == 1.0 and a.percent == 100
    assert {s.status for s in a.stages} == {"done"} and a.cameras == {"cam1": 1.0}
    assert a.review.status == "done"


def test_an_old_whole_group_prepare_job_counts_for_every_camera(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid, engine="paper", cameras=("cam1", "cam2"))
    job(repo, sid, Stage.prepare, "running")     # camera None: the group's cameras together
    a = sub_analysis(repo, aid)
    prep = a.stages[1]
    assert (prep.key, prep.status) == ("prepare", "running")


def test_the_coachs_turn_is_waiting_for_you(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)                              # mapping missing: the coach's turn
    p = work_progress(repo, aid)
    assert p.subs[0].status == "your_turn"
    a = p.subs[0].analysis
    assert a.waiting_for_you and not a.active
    assert p.analysis.waiting_for_you and not p.analysis.active
    aid, sid, _ = confirmed(repo)
    behavior_running(repo, sid, 0.3, None)
    assert not sub_analysis(repo, aid).waiting_for_you
