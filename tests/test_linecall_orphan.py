"""Release-2 re-review MED3: a linecall child whose worker was SIGKILLed must not go on
claiming jobs. The child ends itself (PR_SET_PDEATHSIG on Linux, a getppid() watch as the
portable fallback) through its SIGTERM path, so its session's finally stops its engine command
cleanly; and a new worker never starts a second child while such an orphan (or the engine
command it holds in linecall_call.pgid) still runs. Real processes, no GPU, no engine."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from fakes.engine_env import alive
from pfb_app.config import Settings
from pfb_app.engine import Pfb3x3Engine
from pfb_app.worker import linecall_runner
from pfb_app.worker import main as worker_main
from pfb_app.worker.linecall_runner import LinecallRunner

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                                reason="process checks need /proc (Linux)")

TARGET = '''
import os, time
from pathlib import Path

def run(settings_path, model_key):   # stands in for session_entry: a session that never idles
    mark = Path(os.environ["ORPHAN_MARK"])
    mark.write_text(f"running {model_key}\\n")
    try:
        while True:
            time.sleep(0.05)
    finally:   # where the real session stops its engine command
        mark.write_text("stopped cleanly\\n")
'''

WORKER = '''
import multiprocessing, os, sys, time
sys.path.insert(0, sys.argv[1])
import orphan_target
from pfb_app.worker import linecall_runner
ctx = multiprocessing.get_context("spawn")
p = ctx.Process(target=linecall_runner.child_main, args=(sys.argv[2], os.getpid()),
                kwargs={"target": orphan_target.run, "pdeathsig": sys.argv[3] == "1",
                        "poll_sec": 0.2}, daemon=True)
p.start()
print(p.pid, flush=True)
time.sleep(600)
'''


def _wait(cond, timeout=15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def settings_file(tmp_path, settings):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir)}), encoding="utf-8")
    return path


@pytest.mark.parametrize("pdeathsig", [True, False], ids=["pdeathsig", "getppid-watch"])
def test_the_child_ends_cleanly_when_its_worker_is_killed(tmp_path, settings, settings_file,
                                                          monkeypatch, pdeathsig):
    (tmp_path / "orphan_target.py").write_text(TARGET, encoding="utf-8")
    mark = tmp_path / "mark"
    monkeypatch.setenv("ORPHAN_MARK", str(mark))
    worker = subprocess.Popen([sys.executable, "-c", WORKER, str(tmp_path), str(settings_file),
                               "1" if pdeathsig else "0"], stdout=subprocess.PIPE, text=True)
    child = None
    try:
        child = int(worker.stdout.readline())
        assert _wait(lambda: mark.exists() and mark.read_text().startswith("running linecall"))
        record = linecall_runner.session_record(settings)
        assert _wait(record.exists)
        assert record.read_text().split()[0] == str(child)
        worker.kill()   # SIGKILL: the worker's own stop() never runs
        worker.wait(timeout=10)
        assert _wait(lambda: not alive(child), timeout=10), "the orphan kept running"
        assert mark.read_text() == "stopped cleanly\n"      # through the SIGTERM path
        assert not record.exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if child is not None and alive(child):
            os.kill(child, signal.SIGKILL)


def test_the_parent_watch_calls_back_once_the_parent_changed():
    ppid = {"v": 4242}
    gone = []
    watch = linecall_runner.ParentWatch(4242, 0.01, on_gone=lambda: gone.append(1),
                                        getppid=lambda: ppid["v"])
    watch.start()
    time.sleep(0.1)
    assert gone == [] and watch.is_alive()
    ppid["v"] = 1      # re-parented: the worker is gone
    watch.join(timeout=5)
    assert gone == [1] and not watch.is_alive()


def test_a_child_whose_worker_is_already_gone_never_runs_its_session(tmp_path, settings_file):
    ran = []
    linecall_runner.child_main(str(settings_file), os.getppid() + 1_000_000,
                               target=lambda *a: ran.append(a), pdeathsig=False)
    assert ran == []


def _sleeper(*argv: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", *argv],
                            start_new_session=True)


def _start(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _runner(settings, started):
    return LinecallRunner(settings, None, start=lambda: started.append(1) or _Child(),
                          stop_stale=lambda: None)


class _Child:
    pid = 4243
    exitcode = None

    def is_alive(self):
        return True


def test_no_second_child_while_an_orphan_session_runs(settings):
    orphan = _sleeper("--multiprocessing-fork")   # how a spawn child's command line ends
    started = []
    try:
        record = linecall_runner.session_record(settings)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(f"{orphan.pid} {_start(orphan.pid)}\n", encoding="utf-8")
        runner = _runner(settings, started)
        assert linecall_runner.live_session(settings) == orphan.pid
        assert not runner.ready()
        orphan.kill()
        orphan.wait()
        assert linecall_runner.live_session(settings) is None and runner.ready()
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()
    assert started == []


def test_a_record_naming_another_process_holds_nothing_up(settings):
    other = _sleeper()   # right pid and start time, but not a linecall child
    try:
        record = linecall_runner.session_record(settings)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(f"{other.pid} {_start(other.pid)}\n", encoding="utf-8")
        assert linecall_runner.live_session(settings) is None
        record.write_text(f"{other.pid} 1\n", encoding="utf-8")   # a reused pid
        assert linecall_runner.live_session(settings) is None
        assert _runner(settings, []).ready() and alive(other.pid)   # never signalled
    finally:
        other.kill()
        other.wait()


def test_no_second_child_while_an_engine_linecall_we_did_not_start_runs(settings):
    engine = Pfb3x3Engine(settings)
    group = _sleeper("pfb3x3.engine")   # a live linecall engine command group of ours
    try:
        engine.linecall_call_record.parent.mkdir(parents=True, exist_ok=True)
        engine.linecall_call_record.write_text(f"{group.pid} {_start(group.pid)}\n",
                                               encoding="utf-8")
        started = []
        runner = _runner(settings, started)
        assert not runner.ready()
        group.kill()
        group.wait()
        assert runner.ready()
        runner.start()
        assert started == [1]
        # our own child's engine command is no reason to wait (it is ours)
        group2 = _sleeper("pfb3x3.engine")
        try:
            engine.linecall_call_record.write_text(f"{group2.pid} {_start(group2.pid)}\n",
                                                   encoding="utf-8")
            assert runner.busy and runner.foreign() is None
        finally:
            group2.kill()
            group2.wait()
    finally:
        if group.poll() is None:
            group.kill()
            group.wait()


def test_startup_cleanup_leaves_a_live_orphans_engine_command_alone(settings):
    engine = Pfb3x3Engine(settings)
    orphan = _sleeper("--multiprocessing-fork")
    group = _sleeper("pfb3x3.engine")
    try:
        record = linecall_runner.session_record(settings)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(f"{orphan.pid} {_start(orphan.pid)}\n", encoding="utf-8")
        engine.linecall_call_record.write_text(f"{group.pid} {_start(group.pid)}\n",
                                               encoding="utf-8")
        worker_main.startup_cleanup(settings)
        # the orphan stops its own command on its way out: nothing is signalled here
        assert alive(group.pid) and alive(orphan.pid)
        assert engine.linecall_call_record.exists()
        orphan.kill()
        orphan.wait()
        worker_main.startup_cleanup(settings)   # no orphan any more: stopped as a leftover
        assert _wait(lambda: not alive(group.pid))
        assert not engine.linecall_call_record.exists()
    finally:
        for p in (orphan, group):
            if p.poll() is None:
                p.kill()
                p.wait()


CLEANUP = '''
import sys, time
from pathlib import Path
from pfb_app.worker.session import exit_on_signals
exit_on_signals()
mark = Path(sys.argv[1])
mark.write_text("running")
try:
    while True:
        time.sleep(0.05)
finally:   # a session's cleanup (the engine group's stop) takes a while
    mark.write_text("cleaning")
    time.sleep(1.0)
    mark.write_text("cleaned")
'''


def test_a_second_sigterm_never_cuts_a_sessions_cleanup_short(tmp_path):
    """The worker's death sends SIGTERM twice (PR_SET_PDEATHSIG and the getppid watch), and a
    stop may too: only the first one ends the session, the finally blocks run to the end."""
    mark = tmp_path / "mark"
    proc = subprocess.Popen([sys.executable, "-c", CLEANUP, str(mark)])
    try:
        assert _wait(lambda: mark.exists() and mark.read_text() == "running")
        proc.send_signal(signal.SIGTERM)
        assert _wait(lambda: mark.read_text() == "cleaning")
        proc.send_signal(signal.SIGTERM)
        proc.send_signal(signal.SIGHUP)
        assert proc.wait(timeout=10) == 128 + signal.SIGTERM
        assert mark.read_text() == "cleaned"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
