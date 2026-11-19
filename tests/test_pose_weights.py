"""The line-call pose weights (yolo11x-pose.pt): put in place from a local copy by
scripts/provision_pose_weights.sh with the pinned sha256 (never downloaded), and checked by
the app before a line call (a missing or different file: 「準備ができていません」).
No real weights are used: small stand-in files, with the pin swapped in a copy of the script."""
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.engine import pfb3x3
from pfb_app.engine.pfb3x3 import Pfb3x3Engine, pose_weights_ready

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "provision_pose_weights.sh"
SETUP = ROOT / "scripts" / "setup_pfb3x3_engine.sh"
STAND_IN = b"stand-in pose weights"


def pose_weights(data_dir: Path, monkeypatch, data: bytes = STAND_IN) -> Path:
    """Stand-in weights at <data_dir>/models/yolo11x-pose.pt, pinned (for the tests that run a
    line call)."""
    path = Path(data_dir) / "models" / "yolo11x-pose.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    monkeypatch.setattr(pfb3x3, "POSE_SHA256", hashlib.sha256(data).hexdigest())
    return path


def test_the_pin_is_the_engines():
    # PFB3x3 scripts/linecall_run.py POSE_SHA256 (yolo11x-pose.pt, 118,481,010 bytes)
    assert pfb3x3.POSE_SHA256 == \
        "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"
    assert f'POSE_SHA256="{pfb3x3.POSE_SHA256}"' in SCRIPT.read_text(encoding="utf-8")


def test_ready_needs_the_file_with_the_pinned_sha(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data")
    engine = Pfb3x3Engine(settings)
    assert not engine.pose_ready()                                   # missing
    path = pose_weights(settings.data_dir, monkeypatch)
    assert engine.pose_ready() and pose_weights_ready(path)
    path.write_bytes(b"other weights, same name")                    # changed on disk
    assert not engine.pose_ready()


def test_ready_hashes_an_unchanged_file_once(tmp_path, monkeypatch):
    path = pose_weights(tmp_path, monkeypatch)
    seen = []
    real = pfb3x3._file_sha256
    monkeypatch.setattr(pfb3x3, "_file_sha256", lambda p: seen.append(p) or real(p))
    assert pose_weights_ready(path) and pose_weights_ready(path)
    assert len(seen) <= 1


def test_the_scripts_never_download():
    for script in (SCRIPT, SETUP):
        text = script.read_text(encoding="utf-8")
        for word in ("curl", "wget", "urlretrieve", "github.com/ultralytics"):
            assert word not in text, (script.name, word)


def provision(tmp_path, *args, src: Path | None = None, pin: bytes = STAND_IN):
    """Run a copy of the script whose pin is the stand-in's sha256."""
    copy = tmp_path / "app" / "scripts" / SCRIPT.name
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(SCRIPT.read_text(encoding="utf-8").replace(
        pfb3x3.POSE_SHA256, hashlib.sha256(pin).hexdigest()), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PFB_POSE_WEIGHTS_SRC"}
    env["PFB_DATA_DIR"] = str(tmp_path / "data")
    if src is not None:
        env["PFB_POSE_WEIGHTS_SRC"] = str(src)
    return subprocess.run(["bash", str(copy), *args], capture_output=True, text=True, env=env,
                          timeout=60)


def test_provision_copies_a_local_copy_and_checks_its_sha(tmp_path):
    src = tmp_path / "src" / "yolo11x-pose.pt"
    src.parent.mkdir()
    src.write_bytes(STAND_IN)
    dest = tmp_path / "data" / "models" / "yolo11x-pose.pt"
    res = provision(tmp_path, "--check")
    assert res.returncode == 1 and not dest.exists()             # check mode writes nothing
    res = provision(tmp_path, src=src)
    assert res.returncode == 0, res.stderr
    assert dest.read_bytes() == STAND_IN
    assert not [p for p in dest.parent.iterdir() if p.name != dest.name]   # no temp left
    res = provision(tmp_path, "--check")
    assert res.returncode == 0, res.stderr
    res = provision(tmp_path, src=src)                            # again: up to date
    assert res.returncode == 0 and "up to date" in res.stdout


def test_provision_refuses_another_file_or_no_source(tmp_path):
    src = tmp_path / "src" / "yolo11x-pose.pt"
    src.parent.mkdir()
    src.write_bytes(b"something else")
    dest = tmp_path / "data" / "models" / "yolo11x-pose.pt"
    res = provision(tmp_path, src=src)
    assert res.returncode == 1 and "sha256" in res.stderr and not dest.exists()
    res = provision(tmp_path)                                     # no source given
    assert res.returncode == 1 and "PFB_POSE_WEIGHTS_SRC" in res.stderr
    res = provision(tmp_path, str(tmp_path / "missing.pt"))
    assert res.returncode == 1 and not dest.exists()


def test_provision_replaces_a_changed_copy_only_from_a_good_source(tmp_path):
    dest = tmp_path / "data" / "models" / "yolo11x-pose.pt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"changed on disk")
    assert provision(tmp_path, "--check").returncode == 1
    good = tmp_path / "good.pt"
    good.write_bytes(STAND_IN)
    res = provision(tmp_path, str(good))                          # the source as an argument
    assert res.returncode == 0, res.stderr
    assert dest.read_bytes() == STAND_IN


@pytest.mark.parametrize("script", [SCRIPT, SETUP])
def test_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
