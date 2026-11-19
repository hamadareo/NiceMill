"""GPU guard for the shared GPU: never push other users into out-of-memory."""
import math
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from pfb_app.config import Settings

_MIB_PER_GB = 1024


class GpuQueryError(RuntimeError):
    """nvidia-smi could not be run or its output could not be read."""


@dataclass
class GpuSnapshot:
    total_gb: float
    used_gb: float

    @property
    def free_gb(self) -> float:
        return self.total_gb - self.used_gb


def parse_nvidia_smi(csv_text: str) -> GpuSnapshot:
    """Parse `--query-gpu=memory.total,memory.used --format=csv,noheader,nounits` (MiB). Uses GPU 0."""
    lines = [line for line in csv_text.splitlines() if line.strip()]
    try:
        total_mib, used_mib = (float(v) for v in lines[0].split(","))
    except (IndexError, ValueError) as e:
        raise GpuQueryError(f"nvidia-smiの出力を読めません: {csv_text!r}") from e
    return GpuSnapshot(total_gb=total_mib / _MIB_PER_GB, used_gb=used_mib / _MIB_PER_GB)


def query_nvidia_smi() -> GpuSnapshot:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise GpuQueryError(f"nvidia-smiの実行に失敗しました: {e}") from e
    return parse_nvidia_smi(result.stdout)


class GpuGuard:
    def __init__(self, settings: Settings, query: Callable[[], GpuSnapshot] = query_nvidia_smi):
        self._settings = settings
        self._query = query

    def can_start(self, budget_gb: float) -> tuple[bool, str]:
        free = self._query().free_gb
        margin = self._settings.start_margin_gb
        need = budget_gb + margin
        if free >= need:
            return True, ""
        return False, f"空き {free:.1f}GB < 必要 {need:g}GB（予算{budget_gb:g}+余白{margin:g}）"

    def free_now_gb(self) -> float:
        """The GPU's free memory now (GB)."""
        return self._query().free_gb

    def should_yield(self) -> bool:
        return self._query().free_gb < self._settings.yield_threshold_gb

    def utilization_for(self, budget_gb: float) -> float:
        total = self._query().total_gb
        return math.floor(budget_gb / total * 1000 + 1e-9) / 1000


# -- is the machine busy with other work? (the live page's 「混んでいるため」) -----------------

BUSY_OTHER_MIB = 1024   # a GPU program this big that is not the live run is other work ...
BUSY_UTIL = 50.0        # ... when the GPU is at least this busy (an idle warm server is not)
# Measured 2026-10-24 (integration run, a 4K demo clip alone on the server): the live run held
# about 1.2 GB with the GPU at 15-17 % and dropped 4 of 1531 frames.


def _ancestors(pid: int) -> set[int]:
    """pid and its parents up to init (/proc/<pid>/stat ppid); empty when unreadable."""
    out, cur = set(), int(pid)
    while cur > 1 and cur not in out:
        out.add(cur)
        try:
            stat = Path(f"/proc/{cur}/stat").read_text(encoding="utf-8")
            cur = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def machine_busy(own_pid: int | None, *, run=subprocess.run, loadavg=os.getloadavg,
                 cpus=os.cpu_count) -> bool:
    """Whether other work keeps this machine busy: the 1-minute load average is at least the
    CPU count, or the GPU is at least BUSY_UTIL % busy while a program that is neither
    `own_pid` nor one of its children (the live run's engine and live_run.py) holds
    BUSY_OTHER_MIB. False when it cannot be told (no nvidia-smi, unreadable output)."""
    try:
        if loadavg()[0] >= (cpus() or 1):
            return True
    except OSError:
        pass
    try:
        util = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, check=True, timeout=10).stdout.split()
        if not util or float(util[0]) < BUSY_UTIL:
            return False
        apps = run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, check=True, timeout=10).stdout
        for line in apps.splitlines():
            pid, mib = (v.strip() for v in line.split(","))
            if float(mib) >= BUSY_OTHER_MIB and (own_pid is None
                                                  or own_pid not in _ancestors(int(pid))):
                return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return False
