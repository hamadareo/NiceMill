"""「もう一度ためす」 (services/retry.py), the plain words of the progress, and the step model of
a work started from the drop zone. Synthetic data only."""
import pytest

from pfb_app.domain.models import Stage
from pfb_app.services import retry
from pfb_app.services.review import confirm_identity, save_tracked_identity
from pfb_app.ui.wizard import analysis, bars, router
from pfb_app.ui.wizard.progress import (GIVE_UP, RETRY, job_line, plain_error, work_progress)
from test_wizard_progress import (FOUR, finish_all, group, job, repo, save,  # noqa: F401 (repo)
                                  track, video_work)

OOM = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"


def failed_track(repo, error=OOM):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, "failed_retryable", camera="cam1", error=error)
    return aid, sid, jid


def fail_again(repo, jid, error=OOM):
    repo.conn.execute("UPDATE jobs SET status='failed', error=? WHERE id=?", (error, jid))


# --- the service ----------------------------------------------------------------------------

def test_retry_work_requeues_the_failed_jobs_and_is_audited(repo):
    aid, sid, jid = failed_track(repo)
    assert retry.retryable(repo, aid)
    assert retry.retry_work(repo, aid) == [jid]
    assert repo.get_job(jid).status.value == "queued"
    entry = repo.latest_audit(retry.ACTION, f"assignment:{aid}")
    assert entry["detail"]["job_ids"] == [jid]
    assert not retry.retryable(repo, aid)
    with pytest.raises(retry.CannotRetry):       # nothing failed any more
        retry.retry_work(repo, aid)


def test_retry_work_covers_every_group_of_the_work(repo):
    aid = video_work(repo)
    ids = []
    for label in ("G1", "G2"):
        sid, _ = group(repo, aid, label=label)
        save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
        ids.append(job(repo, sid, Stage.track, "failed", camera="cam1", error=OOM))
    assert sorted(retry.retry_work(repo, aid)) == sorted(ids)


def test_a_superseded_failure_is_not_retried(repo):
    aid, sid, jid = failed_track(repo)
    job(repo, sid, Stage.prepare, "running", camera="cam1")     # redoes the camera from before
    with pytest.raises(retry.CannotRetry):
        retry.retry_work(repo, aid)
    assert repo.get_job(jid).status.value == "failed_retryable"


def test_retry_is_refused_while_the_work_waits_for_deletion(repo):
    aid, _, jid = failed_track(repo)
    repo.audit("delete_requested", f"assignment:{aid}", {})
    with pytest.raises(retry.CannotRetry):
        retry.retry_work(repo, aid)
    assert repo.get_job(jid).status.value == "failed_retryable"


def test_times_retried_counts_this_failure_only(repo):
    aid, sid, jid = failed_track(repo)
    assert retry.times_retried(repo, aid, [jid]) == 0
    retry.retry_work(repo, aid)
    fail_again(repo, jid)
    retry.retry_work(repo, aid)
    assert retry.times_retried(repo, aid, [jid]) == 2
    assert retry.times_retried(repo, aid, [jid + 100]) == 0     # another failure starts afresh


# --- the progress: the action, the plain reason, the give-up copy ----------------------------

def test_a_failed_job_offers_the_retry_with_a_plain_reason(repo):
    aid, _, _ = failed_track(repo)
    p = work_progress(repo, aid)
    assert p.stopped and p.subs[0].action == "retry"
    assert p.reason == ("ほかの解析と重なって止まりました。もう一度ためすと、たいてい進みます。"
                        "（人を見つける・カメラ1）")
    assert RETRY == "もう一度ためす"


def test_after_two_failed_retries_it_says_to_tell_the_administrator(repo):
    aid, _, jid = failed_track(repo)
    for _ in range(2):
        retry.retry_work(repo, aid)
        assert not work_progress(repo, aid).stopped
        fail_again(repo, jid)
    p = work_progress(repo, aid)
    assert p.stopped and p.subs[0].action is None
    assert GIVE_UP == "何度ためしても止まります。管理者に知らせてください。"
    assert p.reason.startswith(GIVE_UP)
    assert "もう一度ためすと" not in p.reason


def test_one_failed_retry_still_offers_the_retry(repo):
    aid, _, jid = failed_track(repo)
    retry.retry_work(repo, aid)
    fail_again(repo, jid)
    assert work_progress(repo, aid).subs[0].action == "retry"


def test_other_errors_keep_their_own_plain_words(repo):
    aid, _, _ = failed_track(repo, error="EngineError: 人物を見つけられませんでした\nログ: /x")
    assert work_progress(repo, aid).reason == "人物を見つけられませんでした（人を見つける・カメラ1）"


# --- plain words (audit #6) ----------------------------------------------------------------

def test_stage_names_are_plain_words():
    names = analysis.STAGE_NAMES
    assert (names["track"], names["people"], names["behavior"], names["contribution"],
            names["evidence"], names["pfb"]) == (
        "人を見つける", "評価する人を選ぶ", "したことの書き出し", "よかった点を選ぶ",
        "場面の書き出し", "文章づくり")


def test_the_queue_and_the_wait_are_called_順番待ち():
    assert router.SUB_CHIP_TEXT["waiting_gpu"] == "順番待ち"
    assert bars.STATUS_CHIPS["waiting_gpu"][0] == "順番待ち"
    gpu = job_line("behavior", "waiting_gpu", "cam1", "video")
    assert gpu.startswith("順番を待っています（したことの書き出し") and "GPU" not in gpu
    assert job_line("track", "queued", "cam1", "video") == "順番を待っています（人を見つける・カメラ1）"
    assert job_line("track", "running", "cam1", "video") == "人を見つけています（カメラ1）"
    assert "GPU" not in plain_error(OOM)


def test_the_people_wait_names_what_is_chosen(repo):
    aid = video_work(repo)
    sid, _ = group(repo, aid)
    track(repo, sid)
    assert work_progress(repo, aid).reason == "4人の名前を選ぶのを待っています"


def test_the_app_title_is_plain():
    from pathlib import Path
    import pfb_app.ui
    source = (Path(pfb_app.ui.__file__).parent / "app.py").read_text(encoding="utf-8")
    assert "page_title=theme.APP_NAME" in source and "PFB作成アプリ" not in source


# --- the step model of a work started from the drop zone (audit #9) -------------------------

def checked_work(repo, check="pending", cameras=("cam1", "cam2")):
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=cameras)
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "check": check})
    return aid, sid


def test_a_dropped_video_marks_step_2_done_and_counts_the_videos(repo):
    aid, _ = checked_work(repo)
    p = work_progress(repo, aid)
    assert p.steps[0] == "動画の中身" and p.steps[1] == "動画（2本）"
    assert p.current == 1 and 2 in p.done and p.reachable(2)


def test_a_confirmed_check_keeps_the_names(repo):
    aid, sid = checked_work(repo, check="confirmed", cameras=("cam1",))
    track(repo, sid)
    save_tracked_identity(repo, sid, "cam1", FOUR)
    confirm_identity(repo, sid)
    p = work_progress(repo, aid)
    assert p.steps[:2] == ["動画の中身", "動画（1本）"] and {1, 2, 3} <= p.done


def test_a_work_without_the_check_keeps_its_steps(repo):
    aid = video_work(repo)
    group(repo, aid)
    assert work_progress(repo, aid).steps[:2] == ["活動を決める", "動画を入れる"]


# --- review round 1 (review-l1-findings.md #2-#5, #10) ---------------------------------------

def test_a_deletion_requested_just_before_the_transaction_refuses_the_retry(repo, monkeypatch):
    """Another tab asks to delete the work between the old check and the write transaction:
    the check inside BEGIN IMMEDIATE still sees it, so nothing goes back in the queue."""
    aid, _, jid = failed_track(repo)
    real_tx = repo._tx

    def tx_after_a_delete_request(*args, **kwargs):
        if not repo.conn.in_transaction:
            repo.audit("delete_requested", f"assignment:{aid}", {})   # the other tab
        return real_tx(*args, **kwargs)

    monkeypatch.setattr(repo, "_tx", tx_after_a_delete_request)
    with pytest.raises(retry.CannotRetry) as e:
        retry.retry_work(repo, aid)
    assert str(e.value) == retry.DELETING
    assert repo.get_job(jid).status.value == "failed_retryable"
    assert repo.latest_audit(retry.ACTION, f"assignment:{aid}") is None


def test_a_second_press_while_the_retry_runs_says_it_is_retrying(repo):
    aid, _, jid = failed_track(repo)
    retry.retry_work(repo, aid)
    with pytest.raises(retry.CannotRetry) as e:      # the job is queued again
        retry.retry_work(repo, aid)
    assert str(e.value) == retry.RETRYING and RETRY_WORDS in str(e.value)
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    with pytest.raises(retry.CannotRetry) as e:
        retry.retry_work(repo, aid)
    assert str(e.value) == retry.RETRYING


def test_nothing_to_retry_and_nothing_running_still_says_nothing(repo):
    aid, _, jid = failed_track(repo)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (jid,))
    with pytest.raises(retry.CannotRetry) as e:
        retry.retry_work(repo, aid)
    assert str(e.value) == retry.NOTHING


RETRY_WORDS = "もう一度ためしています。"


def test_give_up_needs_every_open_failure_to_have_been_retried(repo):
    """cam1 failed twice after retries, then cam2 fails for the first time: cam2 was never
    retried, so the group still offers the retry."""
    aid = video_work(repo)
    sid, _ = group(repo, aid, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    cam1 = job(repo, sid, Stage.track, "failed_retryable", camera="cam1", error=OOM)
    for _ in range(2):
        retry.retry_work(repo, aid)
        fail_again(repo, cam1)
    assert work_progress(repo, aid).subs[0].action is None       # cam1 alone: given up
    job(repo, sid, Stage.track, "failed", camera="cam2", error=OOM)
    p = work_progress(repo, aid)
    assert p.stopped and p.subs[0].action == "retry"
    assert not p.reason.startswith(GIVE_UP)


SERVER_DOWN = "解析サーバーが繰り返し停止しました。詳細画面から再実行してください。"


@pytest.mark.parametrize("error, plain", [
    (SERVER_DOWN, "解析サーバーが止まりました。もう一度ためすと、たいてい進みます。"),
    ("RuntimeError: " + SERVER_DOWN, "解析サーバーが止まりました。もう一度ためすと、たいてい進みます。"),
    ("人物の追跡結果がありません（cam2）。", "人を見つけた結果がまだありません。"),
    ("EngineError: 登録ファイルがありません（先に人物の追跡を実行してください）: "
     "/srv/pfb/data/subs/12/pfb3x3/videos/v_cam1.yaml", "先に人を見つける必要があります。"),
    ("モデルを読み込めませんでした: OSError: /models/x.bin が見つかりません。ほかの文",
     "モデルを読み込めませんでした。"),
], ids=["server_down", "server_down_wrapped", "no_track", "engine_registry", "model_load"])
def test_pipeline_texts_become_plain_words(error, plain):
    out = plain_error(error)
    assert out == plain
    for word in ("詳細画面", "再実行", "人物の追跡", "/", ".yaml", "OSError"):
        assert word not in out, word


def test_the_server_down_text_maps_from_the_runner_constant():
    from pfb_app.pipeline.runner import SERVER_DOWN_MESSAGE
    assert SERVER_DOWN_MESSAGE == SERVER_DOWN
    assert "詳細画面" not in plain_error(SERVER_DOWN_MESSAGE)
