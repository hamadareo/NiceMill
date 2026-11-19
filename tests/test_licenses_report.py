import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("licenses_report", ROOT / "scripts" / "licenses_report.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)


def test_classify():
    assert lr.classify("MIT") == "permissive"
    assert lr.classify("Apache-2.0 OR BSD-3-Clause") == "permissive"
    assert lr.classify("AGPL-3.0-only") == "STRONG (AGPL)"
    assert lr.classify("GPL-3.0-or-later") == "STRONG (GPL)"
    assert lr.classify("LGPL-2.1+") == "weak (LGPL)"
    assert lr.classify("MPL-1.1 OR LGPL-2.1+ OR GPL-2.0+").startswith("weak")
    assert lr.classify("LicenseRef-NVIDIA-Proprietary") == "proprietary"
    assert lr.classify("UNKNOWN") == "unknown"


def test_app_runtime_closure_has_no_agpl_pdf_library():
    names = {n for n, _ in lr.lock_closure(ROOT / "uv.lock")}
    assert "pymupdf" not in names and "pypdfium2" in names
    assert "pypdf" not in names  # test-only (dev group)


def test_unreviewed_copyleft_is_flagged():
    rows = [("foo", "1", "GPL-3.0-only", ""), ("nvidia-x", "1", "LicenseRef-NVIDIA-Proprietary", ""),
            ("certifi", "1", "MPL-2.0", ""), ("bar", "1", "MIT", ""), ("baz", "1", "UNKNOWN", ""),
            ("gone", "1", "NOT INSTALLED HERE (run inside the synced venv)", "")]
    assert [r[0] for r in lr.unreviewed(rows)] == ["foo", "baz", "gone"]


def test_engine_rows_tolerate_missing_name(monkeypatch):
    import json
    import subprocess
    out = json.dumps([{"name": None, "version": "1", "expr": None, "license": None, "classifiers": []},
                      {"name": "Good_Pkg", "version": "2", "expr": "MIT", "license": None, "classifiers": []}])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    assert lr.engine_rows("python") == [("good-pkg", "2", "MIT", "")]


def test_override_that_disagrees_with_metadata_warns(capsys):
    f = {"name": "weasyprint", "version": "1", "expr": "GPL-3.0-only", "license": None, "classifiers": []}
    assert lr.licence_of(f) == lr.OVERRIDES["weasyprint"]
    assert "stricter" in capsys.readouterr().err


def test_check_detects_stale_doc(tmp_path, monkeypatch):
    import sys
    fake = [("a", "1", "MIT", "direct")]
    monkeypatch.setattr(lr, "app_rows", lambda lock: fake)
    out = tmp_path / "T.md"
    base = ["x", "--out", str(out), "--engine-setup", str(tmp_path / "none")]
    monkeypatch.setattr(sys, "argv", base)
    assert lr.main() == 0                                  # write
    monkeypatch.setattr(sys, "argv", base + ["--check"])
    assert lr.main() == 0                                  # fresh
    monkeypatch.setattr(lr, "app_rows", lambda lock: [("a", "2", "MIT", "direct")])
    assert lr.main() == 1                                  # dependency changed -> stale
    out.write_text(out.read_text().replace("| a | 1 |", "| a | 9 |"))
    monkeypatch.setattr(lr, "app_rows", lambda lock: fake)
    assert lr.main() == 1                                  # doc edited by hand -> stale
    monkeypatch.setattr(lr, "app_rows", lambda lock: [("evil", "1", "AGPL-3.0", "direct")])
    monkeypatch.setattr(sys, "argv", base)
    assert lr.main() == 0
    monkeypatch.setattr(sys, "argv", base + ["--check"])
    assert lr.main() == 1                                  # fresh doc but unreviewed AGPL


def test_committed_report_lists_every_lock_package():
    text = (ROOT / "docs" / "THIRD_PARTY.md").read_text()
    assert lr.APP_BEGIN in text and lr.APP_END in text
    for name, _ in lr.lock_closure(ROOT / "uv.lock"):
        assert f"| {name} |" in text, name
