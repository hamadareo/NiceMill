"""解説動画 in the worker: --once makes the queued videos after its pass, the resident worker's
thread makes them beside the sessions, and a start fails a video left half made."""
import time

import pytest

from pdf_synthetic import video_group, work
from pfb_app.services import explainer
from pfb_app.worker.main import ExplainerRunner, worker_loop
from test_wizard_ui_basic import env  # noqa: F401 (env)
from test_worker import StubGuard


@pytest.fixture(autouse=True)
def short(monkeypatch):
    monkeypatch.setattr(explainer, "MAX_MOMENTS", 1)


def _queued(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=30)
    subject = repo.list_subjects(sid)[0]["id"]
    return settings, repo, sid, subject


def test_once_makes_the_queued_videos_and_fails_a_half_made_one(env):
    settings, repo, sid, subject = _queued(env)
    other = repo.list_subjects(sid)[1]["id"]
    left = explainer.request(repo, sid, other, session_id=None, actor=None)
    repo.explainer_claim()                         # as a worker that stopped half way
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    assert repo.explainer_job(left)["status"] == "failed"
    assert repo.explainer_job(left)["error"] == explainer.STOPPED
    assert repo.explainer_job(job)["status"] == "done"


def test_the_resident_thread_makes_a_video_and_stops(env):
    settings, repo, sid, subject = _queued(env)
    runner = ExplainerRunner(settings, poll=0.05)
    runner.start()
    try:
        job = explainer.request(repo, sid, subject, session_id=None, actor=None)
        deadline = time.monotonic() + 60
        while repo.explainer_job(job)["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.1)
        assert repo.explainer_job(job)["status"] == "done"
    finally:
        runner.stop()
    assert not runner.alive


def test_a_halted_render_is_not_kept(env, monkeypatch):
    settings, repo, sid, subject = _queued(env)
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)
    explainer.HALT.set()
    try:
        explainer.run_next(repo, settings)
    finally:
        explainer.HALT.clear()
    assert (repo.explainer_job(job)["status"], repo.explainer_job(job)["error"]) == \
        ("failed", explainer.STOPPED)
    aid = repo.get_submission(sid)["assignment_id"]
    assert not list(explainer.folder(settings, aid, sid).glob("*.mp4"))
    assert not list(explainer.folder(settings, aid, sid).glob(".*"))


class _Hung:
    """A child that never ends and never beats."""

    def __init__(self):
        self.terminated = self.killed = False

    def is_alive(self):
        return not (self.terminated or self.killed)

    def join(self, timeout=None):
        time.sleep(min(timeout or 0, 0.05))

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_a_hung_render_is_stopped_failed_and_can_be_made_again(env):
    settings, repo, sid, subject = _queued(env)
    hung: list[_Hung] = []

    def launch(job):
        hung.append(_Hung())
        return hung[-1]
    runner = ExplainerRunner(settings, poll=0.05, hung_sec=0.3, launch=launch)
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)
    runner.start()
    try:
        deadline = time.monotonic() + 20
        while repo.explainer_job(job)["status"] == "queued" or (
                repo.explainer_job(job)["status"] == "running" and time.monotonic() < deadline):
            time.sleep(0.05)
        row = repo.explainer_job(job)
        assert (row["status"], row["error"]) == ("failed", explainer.HUNG)
        assert hung and hung[0].terminated
        again = explainer.request(repo, sid, subject, session_id=None, actor=None)
        assert again != job
    finally:
        runner.stop()


def test_a_failed_start_fails_the_job_at_once(env):
    settings, repo, sid, subject = _queued(env)
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)

    def launch(_job):
        raise OSError("cannot start")
    runner = ExplainerRunner(settings, poll=0.05, hung_sec=600, launch=launch)
    claimed = repo.explainer_claim()
    with pytest.raises(OSError):
        runner._supervise(repo, claimed)
    row = repo.explainer_job(job)
    assert (row["status"], row["error"]) == ("failed", explainer.FAILED)   # not 10 minutes later


def test_an_error_while_watching_stops_the_child_and_fails_the_job(env, monkeypatch):
    settings, repo, sid, subject = _queued(env)
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)
    child = _Hung()
    runner = ExplainerRunner(settings, poll=0.05, hung_sec=600, launch=lambda j: child)

    def broken(_repo, _job_id):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(runner, "_age", broken)
    claimed = repo.explainer_claim()
    with pytest.raises(RuntimeError, match="locked"):
        runner._supervise(repo, claimed)
    assert child.terminated                      # no child left making a video
    row = repo.explainer_job(job)
    assert (row["status"], row["error"]) == ("failed", explainer.FAILED)


def test_a_bare_timestamp_is_read_as_utc(env):
    settings, repo, sid, subject = _queued(env)
    job = explainer.request(repo, sid, subject, session_id=None, actor=None)
    repo.explainer_claim()
    repo.conn.execute("UPDATE explainer_jobs SET heartbeat_at=? WHERE id=?",
                      ("2000-01-01T00:00:00", job))
    age = ExplainerRunner(settings)._age(repo, job)
    assert age is not None and age > 3600
