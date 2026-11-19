"""Release-2 fix wave: before a live run starts, the live page asks the worker to release the
warm shared server (a request file in <data_dir>/live/ that the worker checks), and waits a
little for its answer. Fake engine only: no GPU."""
import json
import threading
import time

import pytest

from fakes.engine_env import alive, install_fake_engine
from pfb_app.config import Settings
from pfb_app.engine import Pfb3x3Engine
from pfb_app.engine import live
from pfb_app.engine.live import LiveController
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_shared_server import Clock, Guard, ready
from test_shared_server import Engine as SharedEngine
from pfb_app.worker.shared_server import SharedServer


@pytest.fixture
def fe(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fe):
    return Settings(data_dir=tmp_path / "data", engine_dir=fe.engine_dir,
                    engine_python=fe.engine_python, tid_python=tmp_path / "tid_python",
                    gpu_poll_sec=7)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    yield Repo(conn, settings)
    conn.close()


def test_a_release_request_is_recent_then_expires(settings):
    rid = live.request_gpu_release(settings, now=1000.0)
    rec = live.release_request(settings, now=1000.0 + live.RELEASE_HOLD_SEC - 1)
    assert rec is not None and rec["id"] == rid
    assert live.release_request(settings, now=1000.0 + live.RELEASE_HOLD_SEC + 1) is None
    assert not live.release_acked(settings, rid)
    live.ack_release(settings, rid)
    assert live.release_acked(settings, rid)


def test_the_worker_releases_the_warm_server_for_a_live_run(settings, repo, fe):
    guard = Guard(settings)
    shared = SharedServer(settings, guard, SharedEngine(settings), clock=Clock())
    try:
        shared.request()
        handle = ready(shared)
        rid = live.request_gpu_release(settings)
        shared.maintain(repo)
        assert shared.handle is None and not alive(handle.pid)
        assert live.release_acked(settings, rid)
        # while the request holds, the server is not started again for a job or a pre-warm
        state, why = shared.request()
        assert state == "no" and why == live.RELEASE_REASON
        assert not shared.holds_gpu()
    finally:
        shared.close()


def test_the_watcher_releases_it_while_a_session_runs_elsewhere(settings, repo, fe):
    guard = Guard(settings)
    shared = SharedServer(settings, guard, SharedEngine(settings), clock=Clock(),
                          watch_sec=0.2)
    try:
        shared.request()
        handle = ready(shared)
        shared.open(files.db_path(settings))
        rid = live.request_gpu_release(settings)
        deadline = time.monotonic() + 10
        while not live.release_acked(settings, rid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert live.release_acked(settings, rid) and not alive(handle.pid)
    finally:
        shared.close()


def test_a_leased_server_is_not_pulled_from_under_a_session(settings, repo, fe):
    guard = Guard(settings)
    shared = SharedServer(settings, guard, SharedEngine(settings), clock=Clock())
    try:
        shared.request()
        handle = ready(shared)
        with shared.lease():
            rid = live.request_gpu_release(settings)
            shared._release_for_live()
            assert alive(handle.pid) and not live.release_acked(settings, rid)
    finally:
        shared.close()


def _live(settings, **kw):
    return LiveController(settings, Pfb3x3Engine(settings, tick_sec=0.05), stop_wait_sec=5, **kw)


def test_the_live_start_asks_first_and_waits_for_the_worker(settings, tmp_path):
    video = tmp_path / "IMG_0105.MOV"
    video.write_bytes(b"x")
    files.worker_beat_path(settings).parent.mkdir(parents=True, exist_ok=True)
    files.worker_beat_path(settings).touch()                # a worker is running
    seen = {}

    def worker():   # answers the request as SharedServer._release_for_live does
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            rec = live.release_request(settings)
            if rec is not None:
                seen["asked_before_start"] = not (settings.data_dir / "live" / "live.json").exists()
                live.ack_release(settings, rec["id"])
                return
            time.sleep(0.02)
    t = threading.Thread(target=worker)
    t.start()
    ctl = _live(settings, release_wait_sec=5)
    t0 = time.monotonic()
    sess = ctl.start(video)
    t.join()
    try:
        assert seen == {"asked_before_start": True}
        assert time.monotonic() - t0 < 4                       # went on once answered
    finally:
        ctl.stop(sess.session)


def test_without_a_running_worker_the_live_start_does_not_wait(settings, tmp_path):
    video = tmp_path / "IMG_0105.MOV"
    video.write_bytes(b"x")
    ctl = _live(settings, release_wait_sec=30)
    t0 = time.monotonic()
    sess = ctl.start(video)
    try:
        assert time.monotonic() - t0 < 5
        assert live.release_request(settings) is not None      # still asked
    finally:
        ctl.stop(sess.session)


def test_without_a_shared_server_configured_the_live_start_neither_asks_nor_waits(
        settings, tmp_path):
    """No shared model in the settings (SharedServer.for_settings is None): there is no warm
    server to release, so a running worker would never answer; the start must not wait for it."""
    video = tmp_path / "IMG_0105.MOV"
    video.write_bytes(b"x")
    files.worker_beat_path(settings).parent.mkdir(parents=True, exist_ok=True)
    files.worker_beat_path(settings).touch()                # a worker is running
    bare = settings.model_copy(update={"pfb3x3_budget_gb": {"high": 60}})
    assert SharedServer.for_settings(bare, None) is None
    ctl = _live(bare, release_wait_sec=30)
    t0 = time.monotonic()
    sess = ctl.start(video)
    try:
        assert time.monotonic() - t0 < 5
        assert live.release_request(bare) is None              # nothing to release: not asked
    finally:
        ctl.stop(sess.session)
