"""A job's own sub-progress: what the worker writes to jobs.progress / jobs.progress_note."""
import logging

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.job_progress import TRACKING_NOTE, behavior_progress, track_progress
from pfb_app.pipeline.tracking import _on_tick
from pfb_app.storage.db import Repo, connect, init_schema


def test_behavior_progress():
    assert behavior_progress(12, 40) == (0.3, "12 / 40 区間")
    assert behavior_progress(40, 40) == (0.99, "40 / 40 区間")    # translation still follows
    assert behavior_progress(45, 40) == (0.99, "40 / 40 区間")
    assert behavior_progress(3, None) == (None, None) and behavior_progress(0, 0) == (None, None)


def test_track_progress_is_time_based_and_capped():
    assert TRACKING_NOTE == "人物を追跡しています"
    assert track_progress(30, 120) == (0.25, "人物を追跡しています")
    assert track_progress(500, 120) == (0.95, "人物を追跡しています")
    assert track_progress(30, None) == (None, "人物を追跡しています")


@pytest.fixture
def repo_with_running_job(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("活動", Kind.video, "task", "rubric", "standard", False)
    sid = repo.create_submission(aid, "1班", {"cameras": {"cam1": "/x/cam1.mp4"}})
    jid = repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    job = repo.claim_next_job({"pfb3x3:standard"})
    assert job.id == jid
    yields = []
    ctx = StageContext(settings=settings, repo=repo, backend=None,
                       yield_check=lambda: bool(yields))
    return repo, ctx, job, yields


def test_on_tick_writes_progress_only_when_it_changes(repo_with_running_job, caplog):
    repo, ctx, job, yields = repo_with_running_job
    values = iter([(0.3, "12 / 40 区間"), (0.3, "12 / 40 区間"), (0.35, "14 / 40 区間")])
    calls = []

    def source():
        calls.append(1)
        return next(values)

    tick = _on_tick(ctx, job, source)
    statements = []
    repo.conn.set_trace_callback(statements.append)
    try:
        assert [tick(), tick(), tick()] == [False, False, False]
    finally:
        repo.conn.set_trace_callback(None)
    assert len(calls) == 3   # once per tick
    updates = [s for s in statements if "progress" in s and s.lstrip().startswith("UPDATE")]
    assert len(updates) == 2, updates
    row = repo.get_job(job.id)
    assert (row.progress, row.progress_note) == (0.35, "14 / 40 区間")

    def broken():
        raise RuntimeError("rows unreadable")

    yields.append(1)
    with caplog.at_level(logging.ERROR):
        assert _on_tick(ctx, job, broken)() is True   # still the yield check's answer
    assert "rows unreadable" in caplog.text
    assert repo.get_job(job.id).progress == 0.35


def test_on_tick_without_a_source_only_beats(repo_with_running_job):
    repo, ctx, job, _ = repo_with_running_job
    assert _on_tick(ctx, job)() is False
    assert repo.get_job(job.id).progress is None


def test_on_tick_never_lowers_and_logs_a_failure_once(repo_with_running_job, caplog):
    repo, ctx, job, _ = repo_with_running_job
    values = iter([(0.5, "人物を追跡しています"), (0.1, "人物を追跡しています"),
                   (None, "人物を追跡しています")])
    tick = _on_tick(ctx, job, lambda: next(values))
    tick(), tick(), tick()            # a reclaimed job's time restarted: the bar stays
    row = repo.get_job(job.id)
    assert (row.progress, row.progress_note) == (0.5, "人物を追跡しています")

    def broken():
        raise RuntimeError("rows unreadable")

    caplog.clear()
    tick = _on_tick(ctx, job, broken)
    with caplog.at_level(logging.ERROR):
        tick(), tick(), tick()
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


class _Stub:
    def __init__(self, rates, durations, seconds=None):
        self.rates, self.durations, self.seconds, self.asked = rates, durations, seconds, 0

    def track_rates(self):
        return self.rates

    def stage_durations(self, stage, engine):
        return self.durations

    def analysed_seconds(self, sub_dir, vid, start_s, end_s):
        self.asked += 1
        return self.seconds


def _source(stub, start_s=0.0, end_s=None, elapsed=30.0):
    import time
    from types import SimpleNamespace
    from pfb_app.pipeline.tracking import _track_source
    ctx = SimpleNamespace(repo=stub)
    return _track_source(ctx, stub, None, "v", start_s, end_s, time.monotonic() - elapsed)


def test_track_expected_time_follows_the_analysed_length():
    # 2 s of tracking per video second x a 60 s task = 120 s expected
    p, note = _source(_Stub([2.0, 1.0, 3.0], [500]), 10.0, 70.0)()
    assert note == TRACKING_NOTE and p == pytest.approx(0.25, abs=0.01)
    stub = _Stub([2.0], [500], seconds=60.0)     # no end: the video's length, once known
    src = _source(stub)
    assert src()[0] == pytest.approx(0.25, abs=0.01)
    src()
    assert stub.asked == 1                      # read once it is known
    stub = _Stub([2.0], [120], seconds=None)    # not known yet: the plain median
    assert _source(stub)()[0] == pytest.approx(0.25, abs=0.01)
    assert _source(_Stub([], []))()[0] is None
