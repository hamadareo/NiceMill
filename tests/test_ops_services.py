"""Keeping the app running: the generated systemd --user units, the install script (into
temp dirs only: systemctl is never called), the one-worker lock, the unclean-exit marker and
the worker's low-disk hold."""
import configparser
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.ops import alerts, disk, systemd
from pfb_app.ops.supervise import AlreadyRunning, RunningMarker, WorkerLock
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.worker import main as worker_main
from test_worker import StubGuard

ROOT = Path(__file__).resolve().parents[1]
UNITS = ("pfb-worker.service", "pfb-ui.service", "pfb-backup.service", "pfb-backup.timer",
         "pfb-guard.service", "pfb-guard.timer", "pfb-alert@.service")


def parse(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(strict=False, interpolation=None, delimiters=("=",))
    cp.optionxform = str
    cp.read_string(text.replace("\\\n", ""))
    return cp


def test_render_units(tmp_path):
    app = tmp_path / "pfb app"   # a space must survive the quoting
    units = systemd.render(app, config=tmp_path / "s.yaml", home=Path("/home/u"))
    assert tuple(units) == UNITS
    w = parse(units["pfb-worker.service"])
    assert w["Service"]["ExecStart"] == (f'"{app}/.venv/bin/python" -m pfb_app.worker.main'
                                         f" --settings {tmp_path}/s.yaml")
    assert w["Service"]["WorkingDirectory"] == f'"{app}"'
    assert w["Service"]["Restart"] == "always"
    assert w["Service"]["KillMode"] == "mixed"
    assert w["Service"]["SuccessExitStatus"] == "143 129"   # a stop on purpose is no failure
    assert w["Service"]["RestartPreventExitStatus"] == str(worker_main.EXIT_ALREADY_RUNNING)
    assert w["Unit"]["OnFailure"] == "pfb-alert@%n.service"
    assert w["Install"]["WantedBy"] == "default.target"
    text = units["pfb-worker.service"]
    assert "Environment=HF_HUB_OFFLINE=1" in text and "UnsetEnvironment=VIRTUAL_ENV" in text
    assert f"Environment=PFB_CONFIG={tmp_path}/s.yaml" in text
    assert "Environment=PATH=/home/u/.local/bin:" in text
    ui = parse(units["pfb-ui.service"])["Service"]["ExecStart"]
    # the default entry is ui/serve.py, so /pfb-upload exists under systemd too (phonefirst)
    assert "-m streamlit run src/pfb_app/ui/serve.py --server.address 127.0.0.1" in ui
    assert "--server.port 8501 --server.maxUploadSize 4096 --server.headless true" in ui
    timer = parse(units["pfb-backup.timer"])["Timer"]
    assert timer["OnCalendar"] == "*-*-* 03:30:00" and timer["Persistent"] == "true"
    assert timer["Unit"] == "pfb-backup.service"
    b = parse(units["pfb-backup.service"])["Service"]
    assert b["Type"] == "oneshot" and b["ExecStart"].endswith("-m pfb_app.ops.backup --settings "
                                                              f"{tmp_path}/s.yaml run")
    a = parse(units["pfb-alert@.service"])["Service"]
    assert a["ExecStart"].endswith("unit-failed %i")


def test_render_prefix_and_no_config(tmp_path):
    units = systemd.render(tmp_path, prefix="pfb-test", host="127.0.0.1", port=8591)
    assert set(units) == {u.replace("pfb-", "pfb-test-", 1) for u in UNITS}
    w = units["pfb-test-worker.service"]
    assert "--settings" not in w and "PFB_CONFIG" not in w
    assert "OnFailure=pfb-test-alert@%n.service" in w
    assert "--server.address 127.0.0.1 --server.port 8591" in units["pfb-test-ui.service"]


def test_render_ui_entry(tmp_path):
    """--entry app runs ui/app.py alone (start.sh's PFB_UI_ENTRY=app); anything else fails."""
    ui = systemd.render(tmp_path, entry="app")["pfb-ui.service"]
    assert "streamlit run src/pfb_app/ui/app.py " in ui and "serve.py" not in ui
    with pytest.raises(KeyError):
        systemd.render(tmp_path, entry="other")
    out = tmp_path / "units"
    assert systemd.main(["--app-dir", str(tmp_path), "--out", str(out), "--entry", "app"]) == 0
    assert "src/pfb_app/ui/app.py" in (out / "pfb-ui.service").read_text()


def install(*args, home, env_extra=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PFB_CONFIG", "PFB_UI_HOST", "PFB_UI_PORT")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), **(env_extra or {}))
    return subprocess.run(["bash", str(ROOT / "scripts" / "install_user_services.sh"),
                           "--python", sys.executable, *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_install_script_writes_files_into_a_given_folder_only(tmp_path):
    out = tmp_path / "units"
    (tmp_path / "s.yaml").write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    res = install("--unit-dir", str(out), "--prefix", "pfb-test", "--config",
                  str(tmp_path / "s.yaml"), home=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "systemctl は実行していません" in res.stdout
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(u.replace("pfb-", "pfb-test-", 1) for u in UNITS)
    text = (out / "pfb-test-worker.service").read_text()
    assert f"WorkingDirectory={ROOT}" in text and str(tmp_path / "s.yaml") in text
    assert not (tmp_path / ".config").exists()   # nothing in the (fake) home's systemd dir
    res = install("--unit-dir", str(out), "--prefix", "pfb-test", "--uninstall", home=tmp_path)
    assert res.returncode == 0 and list(out.iterdir()) == []


def test_install_script_dry_run_and_bad_options(tmp_path):
    res = install("--dry-run", home=tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.count("### pfb-") == len(UNITS) and "KillMode=mixed" in res.stdout
    assert not (tmp_path / ".config").exists()
    assert install("--prefix", "bad name!", home=tmp_path).returncode == 2
    assert install("--unit-dir", str(tmp_path / "u"), "--enable", home=tmp_path).returncode == 2
    assert install("--python", str(tmp_path / "nope"), "--unit-dir", str(tmp_path / "u"),
                   home=tmp_path).returncode == 1


def test_install_script_reads_the_ui_address_as_start_sh(tmp_path):
    """PFB_UI_HOST / PFB_UI_PORT as in scripts/start.sh; --host / --port still win."""
    def ui_line(*args, **env):
        res = install("--dry-run", *args, home=tmp_path, env_extra=env)
        assert res.returncode == 0, res.stderr
        return next(line for line in res.stdout.splitlines() if "--server.address" in line) \
            + res.stdout.split("--server.port", 1)[1].split()[0]
    assert "--server.address 127.0.0.1" in ui_line() and ui_line().endswith("8501")
    line = ui_line(PFB_UI_HOST="127.0.0.1", PFB_UI_PORT="8642")
    assert "--server.address 127.0.0.1" in line and line.endswith("8642")
    line = ui_line("--host", "127.0.0.2", "--port", "8643", PFB_UI_HOST="127.0.0.1",
                   PFB_UI_PORT="8642")
    assert "--server.address 127.0.0.2" in line and line.endswith("8643")
    res = install("--dry-run", home=tmp_path, env_extra={"PFB_UI_HOST": "a b"})
    assert res.returncode == 2 and "PFB_UI_HOST" in res.stderr

def test_systemd_analyze_accepts_the_units(tmp_path):
    """Where systemd is installed (the server), its own checker reads the files (no unit is loaded)."""
    if subprocess.run(["which", "systemd-analyze"], capture_output=True).returncode != 0:
        pytest.skip("no systemd here")
    out = tmp_path / "units"
    out.mkdir()
    for name, text in systemd.render(ROOT, prefix="pfbverify", python=Path(sys.executable)).items():
        (out / name).write_text(text)
    res = subprocess.run(["systemd-analyze", "--user", "verify", "--man=no",
                          *[str(p) for p in sorted(out.iterdir()) if "@" not in p.name]],
                         capture_output=True, text=True, timeout=60)
    problems = [line for line in (res.stdout + res.stderr).splitlines()
                if "pfbverify" in line and "alert@" not in line]
    assert not problems, problems


# --- one worker, unclean exits ---------------------------------------------------------------

def test_worker_lock_is_exclusive_and_released(tmp_path):
    path = tmp_path / "data" / "worker.lock"
    first = WorkerLock(path).acquire()
    assert path.read_text().strip() == str(os.getpid())
    with pytest.raises(AlreadyRunning, match=str(os.getpid())):
        WorkerLock(path).acquire()
    first.release()
    with WorkerLock(path):
        pass


def test_a_second_worker_exits_75(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(worker_main, "exit_on_signals", lambda: None)
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"data_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    held = WorkerLock(tmp_path / "data" / "worker.lock").acquire()
    try:
        with pytest.raises(SystemExit) as e:
            worker_main.main(["--settings", str(cfg), "--once"])
        assert e.value.code == worker_main.EXIT_ALREADY_RUNNING == 75
    finally:
        held.release()


def test_running_marker(tmp_path):
    path = tmp_path / "worker.running"
    with RunningMarker(path) as m:
        assert not m.unclean and path.exists()
    assert not path.exists()
    path.write_text("123")   # left by a worker that was killed
    with RunningMarker(path) as m:
        assert m.unclean
    assert not path.exists()


def test_worker_main_alerts_after_an_unclean_exit(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path / "data")
    cfg = tmp_path / "s.yaml"
    cfg.write_text(f"data_dir: {s.data_dir}\n", encoding="utf-8")
    s.data_dir.mkdir()
    (s.data_dir / "worker.running").write_text("1")
    ran = []
    monkeypatch.setattr(worker_main, "worker_loop", lambda settings, **kw: ran.append(kw["once"]))
    monkeypatch.setattr(worker_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(worker_main, "exit_on_signals", lambda: None)
    worker_main.main(["--settings", str(cfg), "--once"])
    assert ran == [True]
    assert [a["key"] for a in alerts.recent(s)] == ["worker-restarted"]
    assert not (s.data_dir / "worker.running").exists()
    assert not WorkerLock(s.data_dir / "worker.lock").acquire().release()


# --- the worker holds new work while the disk is too full ------------------------------------

@pytest.fixture
def wsettings(tmp_path):
    return Settings(data_dir=tmp_path / "data", gpu_poll_sec=0, session_idle_sec=0)


def test_worker_starts_nothing_while_the_disk_is_full(wsettings, tmp_path):
    conn = connect(files.db_path(wsettings))
    init_schema(conn)
    repo = Repo(conn, wsettings)
    aid = repo.create_assignment("作業", "video", "説明", "", "standard", False)
    sid = repo.create_submission(aid, "G", {"cameras": {}})
    jid = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    full = lambda s: disk.DiskState(s.data_dir, 3.0, 1000.0, "refuse")   # noqa: E731
    spawned = []
    worker_main.worker_loop(wsettings, guard=StubGuard(), spawn=lambda k, **kw: spawned.append(k),
                            sleep=lambda s: None, once=True, on_start=lambda s: None,
                            disk_check=full)
    assert repo.get_job(jid).status == JobStatus.queued and not spawned
    assert "disk-refuse" in [a["key"] for a in alerts.recent(wsettings)]


def test_disk_pass_states(wsettings):
    ok = lambda s: disk.DiskState(s.data_dir, 500.0, 1000.0, "ok")      # noqa: E731
    warn = lambda s: disk.DiskState(s.data_dir, 40.0, 1000.0, "warn")   # noqa: E731
    full = lambda s: disk.DiskState(s.data_dir, 3.0, 1000.0, "refuse")  # noqa: E731
    assert worker_main._disk_pass(wsettings, full, False) is True
    assert worker_main._disk_pass(wsettings, ok, True) is False
    assert "disk-refuse" not in [a["key"] for a in alerts.recent(wsettings)]   # cleared
    assert worker_main._disk_pass(wsettings, warn, False) is False
    assert [a["key"] for a in alerts.recent(wsettings)] == ["disk-warn"]

    def broken(_s):
        raise OSError("x")
    assert worker_main._disk_pass(wsettings, broken, False) is False
