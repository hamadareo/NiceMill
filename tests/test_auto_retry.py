"""One automatic retry of a step that stopped for a passing reason (services.auto_retry), and
how it sits with the coach's 「もう一度ためす」 give-up rule (services.retry)."""
from datetime import datetime, timedelta, timezone

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.ops import alerts
from pfb_app.pipeline import runner
from pfb_app.services import auto_retry, deletion, retry
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_runner import make_ctx

OOM = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def work(repo):
    aid = repo.create_assignment("作業", "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "グループ1", {"cameras": {"cam1": "/x.mp4"}})
    return aid, sid


def failed_job(repo, sid, error, stage=Stage.prepare, key="cpu", status=JobStatus.failed_retryable):
    jid = repo.enqueue(sid, stage, key, camera="cam1")
    claimed = repo.claim_next_job({key})
    assert claimed.id == jid
    repo.finish_job(jid, status, error)
    return jid


@pytest.mark.parametrize("error,want", [
    (OOM, True), ("heartbeat timeout", True), ("EngineError: エンジン track が失敗しました (exit -9)", True),
    ("ConnectionRefusedError: [Errno 111] Connection refused", True),
    ("sqlite3.OperationalError: database is locked", True),
    ("ValueError: 動画を読めません", False), ("", False), (None, False),
    ("解析サーバーが繰り返し停止しました。詳細画面から再実行してください。\nConnection refused", False),
    ("モデルを読み込めませんでした: CUDA out of memory", False),
])
def test_is_transient(error, want):
    assert auto_retry.is_transient(error) is want


def test_retries_once_per_job(repo):
    _, sid = work(repo)
    jid = failed_job(repo, sid, OOM)
    assert auto_retry.retry_once(repo, jid)
    job = repo.get_job(jid)
    assert job.status == JobStatus.queued and job.error is None
    entry = repo.latest_audit(auto_retry.ACTION, f"job:{jid}")
    assert entry["detail"] == {"stage": "prepare", "error": OOM}
    # it fails the same way again: no second automatic try
    repo.claim_next_job({"cpu"})
    repo.finish_job(jid, JobStatus.failed_retryable, OOM)
    assert not auto_retry.retry_once(repo, jid)
    assert repo.get_job(jid).status == JobStatus.failed_retryable


def test_not_for_lasting_reasons_detect_names_or_failed(repo):
    _, sid = work(repo)
    lasting = failed_job(repo, sid, "ValueError: 読めません")
    detect = failed_job(repo, sid, OOM, stage=Stage.detect, key="detect")
    failed = failed_job(repo, sid, OOM, stage=Stage.track, key="track", status=JobStatus.failed)
    for jid in (lasting, detect, failed):
        assert not auto_retry.retry_once(repo, jid)


def test_not_when_switched_off(tmp_path):
    s = Settings(data_dir=tmp_path / "data", auto_retry=False)
    conn = connect(files.db_path(s))
    init_schema(conn)
    repo = Repo(conn, s)
    _, sid = work(repo)
    assert not auto_retry.retry_once(repo, failed_job(repo, sid, OOM))


def test_not_for_a_work_to_be_deleted(repo):
    aid, sid = work(repo)
    jid = failed_job(repo, sid, OOM)
    repo.update_submission_meta(sid, {**repo.get_submission(sid)["meta"],
                                      deletion.REQUESTED: True})
    assert not auto_retry.retry_once(repo, jid)


def test_the_coach_keeps_both_tries(repo):
    """The automatic try is not a 「もう一度ためす」 press: GIVE_UP_AFTER counts the coach's own."""
    aid, sid = work(repo)
    jid = failed_job(repo, sid, OOM)
    auto_retry.retry_once(repo, jid)
    repo.claim_next_job({"cpu"})
    repo.finish_job(jid, JobStatus.failed_retryable, OOM)
    assert retry.times_retried(repo, aid, [jid]) == 0
    assert retry.retry_work(repo, aid) == [jid]
    assert retry.times_retried(repo, aid, [jid]) == 1


def test_sweep_takes_recent_failures_only(repo):
    _, sid = work(repo)
    recent = failed_job(repo, sid, "heartbeat timeout")
    old = failed_job(repo, sid, "heartbeat timeout", stage=Stage.transcribe, key="whisper")
    long_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    repo.conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (long_ago, old))
    assert auto_retry.sweep(repo) == [recent]
    assert repo.get_job(old).status == JobStatus.failed_retryable
    assert auto_retry.sweep(repo) == []


def test_execute_job_retries_an_oom_once_then_alerts(settings, repo, monkeypatch):
    _, sid = work(repo)
    jid = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")

    def boom(job, ctx):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
    monkeypatch.setattr(runner, "_run_stage", boom)
    ctx = make_ctx(settings, repo)
    with pytest.raises(runner.SessionMustEnd):
        runner.execute_job(repo.claim_next_job({"cpu"}), ctx)
    assert repo.get_job(jid).status == JobStatus.queued        # tried again automatically
    assert not [a for a in alerts.recent(settings) if a["key"] == f"job-stopped:{jid}"]
    with pytest.raises(runner.SessionMustEnd):
        runner.execute_job(repo.claim_next_job({"cpu"}), ctx)
    assert repo.get_job(jid).status == JobStatus.failed_retryable   # 「止まりました」 now
    assert [a["key"] for a in alerts.recent(settings)] == [f"job-stopped:{jid}"]


def test_execute_job_alerts_a_failed_step(settings, repo, monkeypatch):
    _, sid = work(repo)
    jid = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    monkeypatch.setattr(runner, "_run_stage", lambda job, ctx: (_ for _ in ()).throw(
        ValueError("動画を読めません")))
    runner.execute_job(repo.claim_next_job({"cpu"}), make_ctx(settings, repo))
    assert repo.get_job(jid).status == JobStatus.failed_retryable
    assert [a["key"] for a in alerts.recent(settings)] == [f"job-stopped:{jid}"]
