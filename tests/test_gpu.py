import subprocess

import pytest

from pfb_app.config import Settings
from pfb_app.inference import gpu
from pfb_app.inference.gpu import GpuGuard, GpuQueryError, GpuSnapshot, parse_nvidia_smi

TOTAL = 95.59


def guard(used: float) -> GpuGuard:
    return GpuGuard(Settings(), query=lambda: GpuSnapshot(total_gb=TOTAL, used_gb=used))


def test_parse():
    snap = parse_nvidia_smi("97887, 35274\n")
    assert snap.total_gb == pytest.approx(95.59, abs=0.01)
    assert snap.used_gb == pytest.approx(34.45, abs=0.01)
    assert snap.free_gb == pytest.approx(61.14, abs=0.01)


def test_can_start_with_current_load():
    g = guard(34.45)
    assert g.can_start(24)[0] is True
    assert g.can_start(52)[0] is True
    g = guard(40.0)
    ok, reason = g.can_start(52)
    assert ok is False
    assert "空き 55.6GB < 必要 60GB（予算52+余白8）" == reason
    assert g.can_start(24)[0] is True


def test_should_yield_threshold():
    assert guard(TOTAL - 5.9).should_yield() is True
    assert guard(TOTAL - 6.1).should_yield() is False


def test_utilization_for():
    assert guard(0).utilization_for(24) == 0.251


def test_query_failure_raises(monkeypatch):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    with pytest.raises(GpuQueryError):
        gpu.query_nvidia_smi()


def test_missing_binary_raises(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    with pytest.raises(GpuQueryError):
        gpu.query_nvidia_smi()


def test_parse_garbage_raises():
    with pytest.raises(GpuQueryError):
        parse_nvidia_smi("not a number\n")


# -- machine_busy (the live page's 「混んでいるため」) ------------------------------------------

def _smi(gpu_line: str, apps: str):
    def run(cmd, **kw):
        out = gpu_line if any("--query-gpu" in c for c in cmd) else apps
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    return run


def test_machine_busy_needs_other_work(monkeypatch):
    own = 4242
    monkeypatch.setattr(gpu, "_ancestors", lambda pid: {pid, own} if pid == 777 else {pid, 1})
    calm = dict(loadavg=lambda: (2.0, 2.0, 2.0), cpus=lambda: 12)
    # only the live run (777, a child of 4242) and a small desktop program: not busy
    run = _smi("95\n", "777, 1200\n555, 556\n")
    assert gpu.machine_busy(own, run=run, **calm) is False
    # another program with 30 GiB and the GPU at 95 %: busy
    run = _smi("95\n", "777, 1200\n888, 30720\n")
    assert gpu.machine_busy(own, run=run, **calm) is True
    # the other program holds memory but the GPU is nearly idle (a warm, unused server)
    run = _smi("16\n", "777, 1200\n888, 30720\n")
    assert gpu.machine_busy(own, run=run, **calm) is False
    # the CPUs are fully booked
    assert gpu.machine_busy(own, run=_smi("16\n", "777, 1200\n"),
                            loadavg=lambda: (13.0, 9.0, 5.0), cpus=lambda: 12) is True


def test_machine_busy_unknown_is_false(monkeypatch):
    def broken(cmd, **kw):
        raise OSError("no nvidia-smi")
    assert gpu.machine_busy(1, run=broken, loadavg=lambda: (0.5, 0, 0), cpus=lambda: 12) is False
    garbage = _smi("n/a\n", "x, y\n")
    assert gpu.machine_busy(1, run=garbage, loadavg=lambda: (0.5, 0, 0), cpus=lambda: 12) is False
    assert gpu.machine_busy(None, run=_smi("99\n", "5, 9000\n"), loadavg=lambda: (0.5, 0, 0),
                            cpus=lambda: 12) is True     # no live run: every program is other
