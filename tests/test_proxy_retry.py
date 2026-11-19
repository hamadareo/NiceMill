"""A proxy build that failed for a passing reason (a full disk, no memory, a busy device) is
tried again later; one that failed because of the file stays failed until the file changes
(rel5 finding, fixed in rel7). Synthetic files only."""
import errno
import json
import time
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.services import proxy

W, H = 2688, 1512


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


@pytest.fixture
def src(tmp_path):
    p = tmp_path / "inbox" / "big.mp4"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not really a video")     # make() is replaced; start() only stats it
    return p


class _NoChild:
    pid = 4243


@pytest.fixture
def no_child(monkeypatch):
    started = []
    monkeypatch.setattr(proxy.subprocess, "Popen",
                        lambda argv, **kw: started.append(argv) or _NoChild())
    return started


def fail_with(monkeypatch, exc):
    def boom(*a, **kw):
        raise exc
    monkeypatch.setattr(proxy, "make", boom)


def run_child(settings, src):
    """What the spawned child does: main() on the pending sidecar start() wrote."""
    dst, _, _ = proxy.paths(settings, src)
    return proxy.main([str(src), str(dst)])


def sidecar(settings, src):
    return json.loads(proxy.paths(settings, src)[1].read_text(encoding="utf-8"))


def failed_once(settings, src, monkeypatch, no_child, exc):
    fail_with(monkeypatch, exc)
    assert proxy.start(settings, src, W, H)
    assert run_child(settings, src) == 1


@pytest.mark.parametrize("exc", [OSError(errno.ENOSPC, "No space left on device"),
                                 OSError(errno.ENOMEM, "Cannot allocate memory"),
                                 OSError(errno.EMFILE, "Too many open files"),
                                 MemoryError()])
def test_a_passing_failure_is_marked_to_be_tried_again(settings, src, monkeypatch, no_child, exc):
    failed_once(settings, src, monkeypatch, no_child, exc)
    meta = sidecar(settings, src)
    assert meta["failed"] and meta["transient"] and meta["attempts"] == 1
    assert meta["retry_at"] > time.time()


def test_it_waits_out_the_pause_and_then_starts_again(settings, src, monkeypatch, no_child):
    failed_once(settings, src, monkeypatch, no_child, OSError(errno.ENOSPC, "full"))
    assert proxy.state(settings, src) == ("failed", None)        # during the pause
    assert not proxy.start(settings, src, W, H)                  # not yet
    later = time.time() + proxy.RETRY_PAUSE_SEC + 1
    monkeypatch.setattr(proxy.time, "time", lambda: later)
    assert proxy.state(settings, src) == ("none", None)          # may be tried again
    before = len(no_child)
    assert proxy.start(settings, src, W, H)
    assert len(no_child) == before + 1
    assert sidecar(settings, src)["attempts"] == 1                # carried, not reset


def test_it_gives_up_after_a_few_passing_failures(settings, src, monkeypatch, no_child):
    exc = OSError(errno.ENOSPC, "full")
    now = [time.time()]
    monkeypatch.setattr(proxy.time, "time", lambda: now[0])
    for n in range(1, proxy.MAX_ATTEMPTS + 1):
        failed_once(settings, src, monkeypatch, no_child, exc)
        assert sidecar(settings, src)["attempts"] == n
        now[0] += proxy.RETRY_PAUSE_SEC + 1
        if n < proxy.MAX_ATTEMPTS:
            assert proxy.state(settings, src) == ("none", None)
    assert proxy.state(settings, src) == ("failed", None)         # now for good
    assert not proxy.start(settings, src, W, H)


@pytest.mark.parametrize("exc", [RuntimeError("bad stream"), IndexError("no video stream"),
                                 ValueError("odd"), OSError(errno.ENOENT, "gone")])
def test_a_failure_of_the_file_stays_until_the_file_changes(settings, src, monkeypatch, no_child,
                                                            exc):
    failed_once(settings, src, monkeypatch, no_child, exc)
    meta = sidecar(settings, src)
    assert meta["failed"] and not meta.get("transient")
    later = time.time() + 10 * proxy.RETRY_PAUSE_SEC
    monkeypatch.setattr(proxy.time, "time", lambda: later)
    assert proxy.state(settings, src) == ("failed", None)
    assert not proxy.start(settings, src, W, H)


def test_a_changed_file_is_tried_afresh_after_a_failure(settings, src, monkeypatch, no_child):
    failed_once(settings, src, monkeypatch, no_child, RuntimeError("bad"))
    src.write_bytes(b"a different file, longer")
    assert proxy.start(settings, src, W, H)
    assert sidecar(settings, src).get("attempts", 0) == 0
