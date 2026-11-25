#!/usr/bin/env python3
"""Generate docs/THIRD_PARTY.md: licence of every runtime dependency (re-runnable).

  uv run python scripts/licenses_report.py [--out docs/THIRD_PARTY.md]
        [--engine-python /path/to/engines/pfb3x3/.venv/bin/python]
        [--engine-pyproject /path/to/PFB3x3/pyproject.toml]
        [--engine-setup scripts/setup_pfb3x3_engine.sh]
        [--check]          # exit 1 when the file on disk differs or an unreviewed copyleft appears

App section: the runtime closure of `pfb-app` in uv.lock (dev group excluded), licence read
from the installed metadata of the environment running the script (run it in the synced venv).
Engine section: every distribution installed in the engine venv (read-only, metadata only),
plus the declared dependencies and the pins of scripts/setup_pfb3x3_engine.sh.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent.parent

# Copyleft class by pattern. Order matters (first hit wins).
_COPYLEFT = [
    (re.compile(r"\bAGPL|Affero", re.I), "STRONG (AGPL)"),
    (re.compile(r"\bSSPL|Server Side Public", re.I), "STRONG (SSPL)"),
    (re.compile(r"\bLGPL|Lesser General Public", re.I), "weak (LGPL)"),
    (re.compile(r"\bGPL|General Public License", re.I), "STRONG (GPL)"),
    (re.compile(r"\bMPL|Mozilla Public", re.I), "weak (MPL)"),
    (re.compile(r"\bEPL|Eclipse Public", re.I), "weak (EPL)"),
    (re.compile(r"\bCDDL", re.I), "weak (CDDL)"),
]
_PROPRIETARY = re.compile(r"proprietary|NVIDIA", re.I)

# Where the metadata is missing/ambiguous, the reviewed value (source: project page / LICENSE file).
OVERRIDES = {
    "ultralytics": "AGPL-3.0 (Ultralytics Enterprise licence available)",
    "pymupdf": "AGPL-3.0 / Artifex commercial",
    "pypdfium2": "Apache-2.0 OR BSD-3-Clause (PDFium: BSD-3-Clause / Apache-2.0)",
    "weasyprint": "BSD-3-Clause",
    "pydyf": "BSD-3-Clause",
    "av": "BSD-3-Clause (bundles FFmpeg libs: LGPL-2.1+ builds, see note)",
    "opencv-python-headless": "Apache-2.0 (bundles FFmpeg, LGPL-2.1)",
    "certifi": "MPL-2.0",
    "chardet": "LGPL-2.1+",
    "lap": "BSD-2-Clause",
    "humming-kernels": "Apache-2.0",
    "quack-kernels": "Apache-2.0",
    "openai-harmony": "Apache-2.0",
    "torchcodec": "BSD-3-Clause",
    "cuda-toolkit": "NVIDIA Proprietary Software (CUDA meta-package)",
    "colorama": "BSD-3-Clause",
    "pyreadline3": "BSD-3-Clause",
    "pywin32": "PSF-2.0",
    "win32-setctime": "MIT",
    "pyphen": "MPL-1.1 OR LGPL-2.1+ OR GPL-2.0+",
    "tqdm": "MPL-2.0 AND MIT",
    "text-unidecode": "Artistic-1.0-Perl OR GPL-2.0+",
}


# Non-permissive families that were reviewed as a group (NVIDIA CUDA runtime wheels).
REVIEWED_PREFIXES = ("nvidia-", "cuda-")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_fields(md) -> dict:
    return {"name": md["Name"], "version": md["Version"], "expr": md.get("License-Expression"),
            "license": md.get("License"),
            "classifiers": [c for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]}


def metadata_licence(f: dict) -> str:
    """Licence as the package metadata states it (no overrides)."""
    if f.get("expr"):
        return f["expr"].strip()
    classifiers = [c.split("::")[-1].strip() for c in f.get("classifiers", [])
                   if "OSI Approved" not in c.split("::")[-1]]
    lic = (f.get("license") or "").strip()
    if lic and len(lic) <= 60 and "\n" not in lic and lic.upper() != "UNKNOWN":
        return lic
    if classifiers:
        return " / ".join(dict.fromkeys(classifiers))
    if lic and lic.upper() != "UNKNOWN":
        return f"{lic.splitlines()[0].strip()[:60]} (from LICENSE text)"
    return "UNKNOWN"


def _RANK(c: str) -> int:
    for i, r in enumerate(("permissive", "weak", "proprietary", "unknown", "STRONG")):
        if c.startswith(r):
            return i
    return 9


def licence_of(f: dict) -> str:
    name = norm(f["name"])
    meta = metadata_licence(f)
    if name not in OVERRIDES:
        return meta
    ov = OVERRIDES[name]
    # Warn when the installed metadata is stricter than the reviewed override says.
    # (Choice licences "A OR B" are exempt: metadata lists all alternatives.)
    if meta != "UNKNOWN" and " OR " not in ov and _RANK(classify(meta)) > _RANK(classify(ov)):
        print(f"warning: override for {name} hides stricter installed metadata: "
              f"override={ov!r} metadata={meta!r}", file=sys.stderr)
    return ov


def classify(lic: str) -> str:
    if _PROPRIETARY.search(lic) and not re.search(r"\b(MIT|BSD|Apache)\b", lic):
        return "proprietary"
    # "A OR B": the user may choose the permissive side, so judge the weakest copyleft only
    # when every alternative is copyleft.
    if re.search(r"\sOR\s", lic):
        parts = [classify(p) for p in re.split(r"\sOR\s", lic)]
        return min(parts, key=_RANK)
    for rx, label in _COPYLEFT:
        if rx.search(lic):
            return label
    if lic == "UNKNOWN":
        return "unknown"
    return "permissive"


def lock_closure(lock_path: Path, root: str = "pfb-app") -> list[tuple[str, str]]:
    pkgs = {norm(p["name"]): p for p in tomllib.loads(lock_path.read_text())["package"]}
    seen: dict[str, str] = {}
    stack = [norm(d["name"]) for d in pkgs[root].get("dependencies", [])]
    while stack:
        n = stack.pop()
        if n in seen or n not in pkgs:
            continue
        seen[n] = pkgs[n].get("version", "?")
        stack += [norm(d["name"]) for d in pkgs[n].get("dependencies", [])]
    return sorted(seen.items())


def lock_direct(lock_path: Path, root: str = "pfb-app") -> set[str]:
    pkgs = {norm(p["name"]): p for p in tomllib.loads(lock_path.read_text())["package"]}
    return {norm(d["name"]) for d in pkgs[root].get("dependencies", [])}


def app_rows(lock_path: Path):
    rows, direct = [], lock_direct(lock_path)
    for name, ver in lock_closure(lock_path):
        try:
            dist = metadata.distribution(name)
            lic = licence_of(dist_fields(dist.metadata))
        except metadata.PackageNotFoundError:
            lic = OVERRIDES.get(name, "NOT INSTALLED HERE (run inside the synced venv)")
        rows.append((name, ver, lic, "direct" if name in direct else "transitive"))
    return rows


_ENGINE_PROBE = r"""
import json, re, sys
from importlib import metadata
out = []
for d in metadata.distributions():
    md = d.metadata
    out.append({"name": md["Name"], "version": md["Version"],
                "expr": md.get("License-Expression"), "license": md.get("License"),
                "classifiers": [c for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]})
print(json.dumps(out))
"""


def engine_rows(python: str):
    proc = subprocess.run([python, "-I", "-c", _ENGINE_PROBE], capture_output=True, text=True, check=True)
    rows = []
    for d in json.loads(proc.stdout):
        if not d.get("name"):  # broken dist-info without a Name
            continue
        rows.append((norm(d["name"]), d.get("version") or "?", licence_of(d), ""))
    return sorted(set(rows))


def md_table(rows, with_kind=False) -> str:
    head = "| Package | Version | Licence | Copyleft class |" + (" Kind |" if with_kind else "")
    sep = "|---|---|---|---|" + ("---|" if with_kind else "")
    lines = [head, sep]
    for name, ver, lic, kind in rows:
        cls = classify(lic)
        mark = f"**{cls}**" if cls != "permissive" else cls
        lines.append(f"| {name} | {ver} | {lic.replace('|', '/')} | {mark} |" + (f" {kind} |" if with_kind else ""))
    return "\n".join(lines)


def flags(rows):
    return [(n, v, lic, classify(lic)) for n, v, lic, _ in rows if classify(lic) != "permissive"]


APP_BEGIN = "<!-- app-section:begin -->"
APP_END = "<!-- app-section:end -->"


def unreviewed(rows):
    """Rows that are not permissive, unknown or not installed, and have no reviewed override."""
    bad = []
    for n, v, lic, _ in rows:
        if lic.startswith("NOT INSTALLED") or (classify(lic) != "permissive"
                                              and n not in OVERRIDES and not n.startswith(REVIEWED_PREFIXES)):
            bad.append((n, v, lic, classify(lic)))
    return bad


def build(args) -> str:
    out = ["# Third-party licences", "",
           "Generated by `scripts/licenses_report.py` (re-run it after any dependency change; "
           "`--check` fails when this file is stale). Source of truth for the app list: `uv.lock`; "
           "licences are read from the installed package metadata, with reviewed overrides in the script.",
           ""]
    app = app_rows(ROOT / "uv.lock")
    out += [APP_BEGIN, "## 1. pfb_app runtime dependencies", "",
            f"{len(app)} packages (runtime closure of `pfb-app` in `uv.lock`; dev group excluded; "
            "environment markers are not evaluated, so platform-only packages are listed too).", "",
            md_table(app, with_kind=True), ""]
    bad = flags(app)
    out += ["### Copyleft / non-permissive in the app (review required before distribution)", ""]
    out += [f"- **{n} {v}**: {lic} -> {c}" for n, v, lic, c in bad] or ["- none"]
    out += ["", "Notes: `certifi` (MPL-2.0) and LGPL parts bundled inside `av` / `opencv-python-headless` wheels are "
            "weak copyleft (file / dynamic-link level) and are met by shipping the notices; `pymupdf` (AGPL) was "
            "removed in favour of `pypdfium2`.", "", APP_END, ""]

    out += ["## 2. PFB3x3 engine (separate venv, read-only audit)", ""]
    if args.engine_pyproject and Path(args.engine_pyproject).is_file():
        t = tomllib.loads(Path(args.engine_pyproject).read_text())["project"]
        out += ["### Declared dependencies (`PFB3x3/pyproject.toml`)", "",
                f"- runtime: {', '.join(t.get('dependencies', []))}",
                f"- optional: {json.dumps(t.get('optional-dependencies', {}))}", ""]
    if args.engine_setup and Path(args.engine_setup).is_file():
        text = Path(args.engine_setup).read_text()
        pins = re.findall(r"^(TORCH_PKGS|TRACK_PKGS)=\((.*?)\)", text, re.M)
        out += ["### Pins in `scripts/setup_pfb3x3_engine.sh`", ""]
        out += [f"- `{k}`: {v.split('#')[0].strip()}" for k, v in pins]
        out += [""]
    if args.engine_python:
        eng = engine_rows(args.engine_python)
        out += ["### Installed in the engine venv", "", f"{len(eng)} distributions.", "", md_table(eng), ""]
        out += ["### Copyleft / non-permissive in the engine venv", ""]
        out += [f"- **{n} {v}**: {lic} -> {c}" for n, v, lic, c in flags(eng)] or ["- none"]
        out += [""]
    else:
        out += ["_Engine venv not given (`--engine-python`); only declared pins listed._", ""]
    out += ["## 3. Known blockers", "",
            "- **ultralytics (YOLO11) + `yolo11l.pt` weights: AGPL-3.0** (engine venv, "
            "`engine/pfb3x3.py` detector). Distributing it, or offering it as a network service, requires "
            "releasing the source under AGPL or buying the Ultralytics Enterprise licence. Not removed in this "
            "lane; decision pending (buy vs. swap to RTMDet / YOLOX / D-FINE, Apache-2.0).",
            "- NVIDIA CUDA wheels pulled by torch / vllm are under NVIDIA's proprietary redistribution terms "
            "(redistribution allowed with the notices; not OSI open source).",
            "- Model weights (Qwen3-VL, faster-whisper / large-v3-turbo) and fonts are not "
            "Python packages and are not scanned here: Apache-2.0 / MIT / OFL; keep their notices.", "",
            "## 4. Fonts and the logo", "",
            "| Font | Where | Licence | Notice |",
            "|---|---|---|---|",
            "| M PLUS 1 (400/500/700/800, subset) | screens: body text | SIL OFL 1.1 | "
            "`src/pfb_app/ui/static/fonts/OFL.txt` |",
            "| Dela Gothic One (subset) | screens: headings; the 「NiceMill」 wordmark (as outlines "
            "in `ui/static/brand/*.svg`) | SIL OFL 1.1 | `src/pfb_app/ui/static/fonts/OFL.txt` |",
            "| Barlow Condensed (600/700, Latin) | screens: numbers | SIL OFL 1.1 | "
            "`src/pfb_app/ui/static/fonts/OFL.txt` |",
            "| Noto Sans / Serif CJK JP | PDFs (system font on the server, not vendored) | SIL OFL 1.1 | "
            "the OS package |", "",
            "The woff2 files and the logo files are built by `scripts/brand_assets.py` from the TTFs in "
            "github.com/google/fonts (no Reserved Font Names). The NiceMill mark and wordmark are this "
            "project's own.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "THIRD_PARTY.md"))
    ap.add_argument("--engine-python")
    ap.add_argument("--engine-pyproject")
    ap.add_argument("--engine-setup", default=str(ROOT / "scripts" / "setup_pfb3x3_engine.sh"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = build(args)
    out = Path(args.out)
    if args.check:
        # Like for like: only the app section is compared (the engine section depends on
        # which engine venv was audited). Run inside the synced venv.
        app = app_rows(ROOT / "uv.lock")
        problems = [f"unreviewed licence: {n} {v}: {lic} ({c})" for n, v, lic, c in unreviewed(app)]
        fresh = text[text.index(APP_BEGIN):text.index(APP_END) + len(APP_END)]
        disk = out.read_text() if out.is_file() else ""
        if APP_BEGIN not in disk or APP_END not in disk or \
                disk[disk.index(APP_BEGIN):disk.index(APP_END) + len(APP_END)] != fresh:
            problems.append(f"{out} is stale: regenerate with scripts/licenses_report.py")
        for pr in problems:
            print(pr, file=sys.stderr)
        return 1 if problems else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
