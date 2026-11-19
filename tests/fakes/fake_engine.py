"""Stand-in for `python -m pfb3x3.engine` (and serve.sh / stop_server.sh) in tests.

Invoked as `<engine_python> -m pfb3x3.engine <cmd> ...` through a wrapper script, or as
`fake_engine.py serve|stop ...` from fake scripts/serve.sh and stop_server.sh. Stdlib only.
It writes canned outputs under the directories it is given and prints log lines followed by
one JSON object on the last stdout line, like the real engine.

Controlled by environment variables:
  FAKE_ENGINE_LOG    append {"argv": [...], "env": {...}} per call (JSON lines)
  FAKE_ENGINE_EXIT   exit with this code after printing an error (JSON + stderr)
  FAKE_ENGINE_EXIT_CMD  only apply FAKE_ENGINE_EXIT to this command (e.g. "behavior")
  FAKE_ENGINE_SLEEP  sleep this many seconds before answering; a child process is started
                     first, and both pids go to FAKE_ENGINE_PIDS (one per line)
  FAKE_ENGINE_OUTSIDE  report a result path outside the given directories
  FAKE_ENGINE_STDOUT   "empty": print nothing; "nojson": only log lines (also on failure);
                       "trailing": noise lines after the JSON summary
  FAKE_ENGINE_SLEEP_CMD  only apply FAKE_ENGINE_SLEEP to this command (e.g. "transcribe")
  FAKE_ENGINE_STUBBORN path: with FAKE_ENGINE_SLEEP, the child ignores SIGTERM and creates
                       this file once it does
  FAKE_ENGINE_SERVE_SLEEP  serve: sleep this long after writing serve.pid, before "ready"
  FAKE_ENGINE_STOP_EXIT    stop: exit with this code without stopping anything
  FAKE_ENGINE_PEOPLE   track: how many people (P1..Pn) the tracker finds (default 3)
  FAKE_ENGINE_TRACKER  track: "engine" (default) or "trackid3x3": the tracker the JSON and
                       track_stats.json `tracker` report (engine 58f8341); "none": neither
                       (engines before 58f8341)
  FAKE_ENGINE_QUALITY  quality: a JSON object that replaces the computed per-label answer
  FAKE_ENGINE_SHEET_NO_AUTO  sheet --auto-start: exit 1 (no frame with everyone boxed)
  FAKE_ENGINE_SHEET_NO_DEFAULT  sheet without --frame (--auto-start or the default frame):
                       exit 1 (nobody is boxed near frame 1, the people come in later)
  FAKE_ENGINE_BOXES_FRAME  track: the frame the boxes file puts the people on (default 1)
  FAKE_ENGINE_NO_RUN_META    behavior: write no run_meta.json next to the rows
  FAKE_ENGINE_ERROR    the {"error": ...} text of a FAKE_ENGINE_EXIT failure
  FAKE_ENGINE_ORPHAN   path: start a child that outlives the command (in its process group)
                       and write its pid there
  FAKE_ENGINE_STUBBORN (live) path: the engine ignores SIGTERM and starts a live_run.py
                       stand-in in its own session (`--parent-pid <engine pid>`, ignores
                       SIGTERM, no parent-death signal); "<engine pid>\n<child pid>\n" go there
  FAKE_ENGINE_LC_GEOMETRY  linecall: "none" writes no geometry.json (engines before 58f8341
                       wrote one too; tests of an unknown camera); default: a camera behind
                       the top line (the top line toward it, the sidelines and baseline not)
  FAKE_ENGINE_LIVE_RECORDS  live: end by itself (reason eof) after this many records between
                       `start` and `end` (default: run until SIGTERM, at most 1200 records)
"""
import argparse
import atexit
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# an 8x8 JPEG (the engine's crops and overlays are JPEGs written by PIL; tests never view them)
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERX"
    "RTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2Nj"
    "Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAEC"
    "AwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2Jy"
    "ggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWW"
    "l5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEA"
    "AwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMi"
    "MoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0"
    "dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn"
    "6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCKiiiuc5T/2Q==")
STATUS_JA = "判定材料"
VERDICT_JA = {"stepped": "踏んだ", "clear": "踏んでいない", "undecidable": "判定不能"}


def _log_call(argv: list[str]) -> None:
    log = os.environ.get("FAKE_ENGINE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "env": dict(os.environ)}) + "\n")


def _maybe_sleep(cmd: str) -> None:
    secs = os.environ.get("FAKE_ENGINE_SLEEP")
    only = os.environ.get("FAKE_ENGINE_SLEEP_CMD")
    if not secs or (only and only != cmd):
        return
    stubborn = os.environ.get("FAKE_ENGINE_STUBBORN")
    code = "import time; time.sleep(600)"
    if stubborn:
        code = ("import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w').close(); time.sleep(600)")
    child = subprocess.Popen([sys.executable, "-c", code] + ([stubborn] if stubborn else []))
    atexit.register(child.kill)   # a normal finish must not leave the child behind
    pids = os.environ.get("FAKE_ENGINE_PIDS")
    if pids:
        Path(pids).write_text(f"{os.getpid()}\n{child.pid}\n", encoding="utf-8")
    time.sleep(float(secs))


def _fail_if_asked(cmd: str) -> None:
    code = os.environ.get("FAKE_ENGINE_EXIT")
    only = os.environ.get("FAKE_ENGINE_EXIT_CMD")
    if code and (not only or only == cmd):
        print(f"fake engine {cmd}: Traceback (most recent call last)", file=sys.stderr)
        print(f"fake engine {cmd}: failure {code}", file=sys.stderr)
        mode = os.environ.get("FAKE_ENGINE_STDOUT")
        if mode == "nojson":
            print("Traceback: something broke")
        elif mode != "empty":
            print(json.dumps({"error": os.environ.get("FAKE_ENGINE_ERROR")
                              or f"fake failure {code}"}))
        sys.exit(int(code))


def _out(obj: dict) -> None:
    mode = os.environ.get("FAKE_ENGINE_STDOUT")
    if mode == "empty":
        return
    print("loading models ...")
    print("progress 1/1")
    if mode == "nojson":
        return
    print(json.dumps(obj, ensure_ascii=False))
    if mode == "trailing":
        print("[W atexit] destroying process group")
        print("{not json either")
        print("")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def cmd_track(a) -> None:
    entry = Path(a.registry) / f"{a.video_id}.yaml"
    if not entry.is_file():
        print(f"no registry entry {entry}", file=sys.stderr)
        sys.exit(1)
    d = Path(a.tracks) / a.video_id
    n = int(os.environ.get("FAKE_ENGINE_PEOPLE") or 3)
    labels = [f"P{i}" for i in range(1, n + 1)]
    frame = int(os.environ.get("FAKE_ENGINE_BOXES_FRAME") or 1)
    boxes = _write(d / "boxes.txt", "".join(f"{frame},{i},{10 + 70 * (i - 1)},10,50,100,0.9\n"
                                            for i in range(1, n + 1)))
    tracker = os.environ.get("FAKE_ENGINE_TRACKER") or "engine"
    summary = {}
    stats_doc = {p: {"frames": 110 - 10 * i} for i, p in enumerate(labels, 1)}
    if tracker != "none":   # engine 58f8341: which tracker ran (auto: TID only for a TID video)
        fallback = (None if tracker == "trackid3x3" else
                    {"reason": "not a TID dataset video", "stage": "eligibility", "tid_s": 0.0})
        record = {"used": tracker, "requested": "auto", "fallback": fallback}
        if tracker == "trackid3x3":
            record.update(source="existing", provenance="tid/provenance.json")
        stats_doc["tracker"] = record
        summary = {"tracker": tracker, "tracker_requested": "auto", "fallback": fallback}
    stats = _write(d / "track_stats.json", json.dumps(stats_doc))
    if os.environ.get("FAKE_ENGINE_OUTSIDE"):
        stats = Path("/tmp/elsewhere/track_stats.json")
    _out({"people": n, "labels": labels, "track_stats": str(stats), "boxes": str(boxes),
          **summary})


def cmd_frames(a) -> None:
    """behavior's frame cache ahead (engine `frames`): two marker files."""
    entry = Path(a.registry) / f"{a.video_id}.yaml"
    if not entry.is_file():
        print(f"no registry entry {entry}", file=sys.stderr)
        sys.exit(1)
    d = Path(a.frames) / a.video_id
    for f in (1, 16):
        _write(d / f"f{f:06d}_w1920.jpg", "jpg")
    _out({"frames": 2, "cached": 2, "decode": "auto"})


def cmd_quality(a) -> None:
    boxes = Path(a.tracks) / a.video_id / "boxes.txt"
    rows = [line.split(",") for line in boxes.read_text(encoding="utf-8").splitlines() if line]
    frames = [int(r[0]) for r in rows]
    span = max(1, max(frames) - min(frames) + 1) if frames else 1
    per: dict[str, set] = {}
    for r in rows:
        per.setdefault(f"P{int(r[1])}", set()).add(int(r[0]))
    quality = {lb: {"coverage": len(fs) / span, "mean_h_frac": 0.2, "fragments": 1,
                    "overlap_rate": 0.0, "spread_frac": 0.1, "frames": len(fs)}
               for lb, fs in per.items()}
    override = os.environ.get("FAKE_ENGINE_QUALITY")
    if override:
        quality = json.loads(override)
    _out({"quality": quality, "task_range": [1, max(frames) if frames else 1]})


def _frame_choice(a) -> dict:
    """Like the engine's pick_frame: --frame (manual) > --auto-start > the window rule."""
    if a.frame is not None:
        return {"frame": a.frame, "frame_rule": "manual", "frame_offset_s": 0.0}
    if a.auto_start:
        return {"frame": 152, "frame_rule": "auto_start", "frame_offset_s": 6.4}
    return {"frame": 120, "frame_rule": "start", "frame_offset_s": 0.0}


def cmd_sheet(a) -> None:
    if a.auto_start and os.environ.get("FAKE_ENGINE_SHEET_NO_AUTO"):
        print("no frame with all people within 60s after the task start", file=sys.stderr)
        print(json.dumps({"error": "sheet: no frame with all people within 60s"}))
        sys.exit(1)
    if a.frame is None and os.environ.get("FAKE_ENGINE_SHEET_NO_DEFAULT"):
        print("sheet: no boxes within 5.0s of frame 1", file=sys.stderr)
        print(json.dumps({"error": "sheet: no boxes within 5.0s of frame 1"}))
        sys.exit(1)
    sheet = _write(Path(a.out) / "sheet.jpg", "fake image")
    _out({"sheet": str(sheet), **_frame_choice(a),
          "geometry": {"P1": [10, 10, 50, 100], "P2": [80, 10, 50, 100]}})


def cmd_rule_names(a) -> None:
    positions = json.loads(a.positions)
    _out({"names": {f"P{i}": name for i, name in enumerate(positions.values(), start=1)},
          **_frame_choice(a)})


def cmd_propose(a) -> None:
    appearance = json.loads(a.appearance)
    _out({"names": {f"P{i}": name for i, name in enumerate(appearance, start=1)},
          "source": "vlm_proposal"})


def _transcript_used(a) -> bool:
    """Like the real engine: true only when the entry asks for it and segments exist."""
    entry = Path(a.registry) / f"{a.video_id}.yaml"
    wants = entry.is_file() and "transcript: true" in entry.read_text(encoding="utf-8")
    return wants and (Path(a.tracks) / a.video_id / "transcript.json").is_file()


def cmd_behavior(a) -> None:
    """Mirrors the real layout: <exp>/<model>/<cond>/<id>/behavior.engine.jsonl (one row per
    window, full window ids, "m:ss" task clocks, short English observations naming P-labels,
    one event the engine excluded, confidence null when the model gave none) and
    overlay/<window_id>/1..n.jpg (1-based)."""
    rows = [
        {"id": f"{a.video_id}_w0000", "window_id": f"{a.video_id}_w0000", "window_index": 0,
         "status": "ok", "clock_start": "0:00", "clock_end": "0:10",
         "events": [{"label": "P1", "clock": "0:05", "observed_action": "P1 folds a paper strip",
                     "action_type": "build", "confidence": "high", "excluded_reason": None},
                    {"label": "P7", "clock": "0:06", "observed_action": "P7 watches P1",
                     "action_type": "other", "confidence": "low",
                     "excluded_reason": "label_not_listed"}]},
        {"id": f"{a.video_id}_w0001", "window_id": f"{a.video_id}_w0001", "window_index": 1,
         "status": "ok", "clock_start": "0:10", "clock_end": "0:20",
         "events": [{"label": "P2", "clock": "0:12", "observed_action": "P2 hands a paper strip to P1",
                     "action_type": "hand_over", "confidence": None, "excluded_reason": None},
                    {"label": "P3", "clock": "0:15", "observed_action": "P3 wipes the desk",
                     "action_type": "other", "confidence": "medium",
                     "excluded_reason": None}]},
    ]
    run_dir = Path(a.exp) / a.model / a.condition / a.video_id
    out = _write(run_dir / "behavior.engine.jsonl",
                 "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if not os.environ.get("FAKE_ENGINE_NO_RUN_META"):   # the engine's traceability record
        _write(run_dir / "run_meta.json", json.dumps({
            "model": a.model, "served_model": a.model, "settings_sha": "5e771465fa6e5ba1",
            "prompts": {"generic_behavior": "b" * 64, "generic_id_marks": "i" * 64,
                        "generic_formats": "f" * 64}}))
    overlay = run_dir / "overlay"
    for r in rows:
        for k in (1, 2):
            _write(overlay / r["window_id"] / f"{k}.jpg", "fake image")
    _out({"rows": str(out), "windows": 2, "errors": 0, "overlay_dir": str(overlay),
          "transcript": _transcript_used(a)})


def cmd_transcribe(a) -> None:
    d = Path(os.environ["PFB_TRACKS"]) / a.video_id
    f = _write(d / "transcript.json", json.dumps([{"start": 1.0, "end": 2.0, "text": "はい"},
                                                  {"start": 3.0, "end": 4.0, "text": "どうぞ"}]))
    _out({"segments": 2, "transcript": str(f)})


def _entry_path(registry: str, video_id: str) -> Path | None:
    """The `path:` of a registry entry (pfb_app writes it with yaml.safe_dump); None when there
    is no entry. Stdlib only, so a plain line parse."""
    entry = Path(registry) / f"{video_id}.yaml"
    if not entry.is_file():
        return None
    for line in entry.read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            return Path(line.split(":", 1)[1].strip().strip("'\""))
    return None


def _label_precision(verdicts: dict, labels: dict) -> dict:
    """As the engine's verdict.label_precision (min_n 50): only {"label": correct|wrong,
    "verdict": v} matching the current verdict or {"truth": stepped|clear} count."""
    tally = {v: [0, 0] for v in VERDICT_JA}
    skipped = 0
    for eid, lab in labels.items():
        v = verdicts.get(eid)
        out = None
        if isinstance(lab, dict) and v in tally:
            if "truth" in lab:
                out = (None if lab["truth"] not in ("stepped", "clear")
                       else "correct" if lab["truth"] == v else "wrong")
            elif lab.get("verdict") == v and lab.get("label") in ("correct", "wrong"):
                out = lab["label"]
        if out is None:
            skipped += 1
            continue
        tally[v][0 if out == "correct" else 1] += 1
    n = {v: sum(tally[v]) for v in ("stepped", "clear")}
    correct = {v: tally[v][0] for v in ("stepped", "clear")}
    prec = {v: (correct[v] / n[v] if n[v] >= 50 else None) for v in n}
    return {"confirmed": sum(n.values()), "skipped": skipped, "min_n": 50, "n": n,
            "correct": correct, "precision": prec, "wilson_lo": {v: None for v in n},
            "undecidable": {"n": sum(tally["undecidable"]), "correct": tally["undecidable"][0]}}


def _inv3(m: list[list[float]]) -> list[list[float]]:
    """The inverse of a 3x3 matrix (stdlib only)."""
    (a, b, c), (d, e, f), (g, h, i) = m
    co = [[e * i - f * h, -(d * i - f * g), d * h - e * g],
          [-(b * i - c * h), a * i - c * g, -(a * h - b * g)],
          [b * f - c * e, -(a * f - c * d), a * e - b * d]]
    det = a * co[0][0] + b * co[0][1] + c * co[0][2]
    return [[co[j][k] / det for j in range(3)] for k in range(3)]


def behind_top_line_H() -> list[list[float]]:
    """image px -> court cm of a pinhole camera 3 m up, 6 m behind the top line's middle,
    looking at the basket (u = 960 + 1000 (x - 752.5) / D, v = 540 + 1000 * 300 / D,
    D = 1705 - y)."""
    fx, cx, cy, h, x0, yc = 1000.0, 960.0, 540.0, 300.0, 752.5, 1705.0
    m = [[fx, -cx, cx * yc - fx * x0], [0.0, -cy, cy * yc + fx * h], [0.0, -1.0, yc]]
    H = _inv3(m)
    return [[v / H[2][2] for v in row] for row in H]


def cmd_linecall(a) -> None:
    """Mirrors the engine's linecall outputs in <tracks>/<id>/linecall/: episodes.json (status
    判定材料; three episodes stepped A6 / clear P3 / undecidable B1; crop, overlay and clip paths
    relative to linecall/), crops/<id>/<frame>.jpg + <frame>_overlay.jpg (tiny JPEGs),
    clips/<id>.mp4 (empty), provenance.json and autocourt.json. labels.json is only read."""
    if _entry_path(a.registry, a.video_id) is None:
        print(f"no registry entry for {a.video_id}", file=sys.stderr)
        print(json.dumps({"error": f"no registry entry for {a.video_id}"}))
        sys.exit(1)
    lc = Path(a.tracks) / a.video_id / "linecall"
    fps = 30.0
    specs = [   # label, line, f0, f1, evidence frame, verdict, reason, d_L, d_R, court point
        ("A6", "sideline_L", 120, 141, 131, "stepped", None, -3.1, 12.4, [40.0, 600.0]),
        ("P3", "baseline", 900, 915, 907, "clear", None, 9.8, 21.0, [700.0, 15.0]),
        ("B1", "arc", 2400, 2433, 2410, "undecidable", "occluded", None, 4.2, [752.5, 790.0]),
    ]
    # and one without an evidence frame (no box): no crop, overlay, clip, clock or frame
    eps, rows = [], []
    for label, line, f0, f1, frame, verdict, reason, dl, dr, near in specs:
        eid = f"{label}-{line}-{f0}"
        crop = f"crops/{eid}/{frame}.jpg"
        overlay = f"crops/{eid}/{frame}_overlay.jpg"
        clip = f"clips/{eid}.mp4"
        for rel in (crop, overlay):
            (lc / rel).parent.mkdir(parents=True, exist_ok=True)
            (lc / rel).write_bytes(TINY_JPEG)
        _write(lc / clip, "")
        group = "sideline" if line.startswith("sideline") else line
        sec = int((frame - 1) / fps + 0.05)
        # frames.jsonl rows of the evidence frame, with the T11 fields (engine 58f8341)
        rows += [{"episode": eid, "frame": frame, "foot": s, "side": s, "method": "ankle",
                  "d_cm": d, "unc_cm": 3.5, "near_court": near, "line_found": True,
                  "ground_gap_px": 0.0, "hi_raw": None if d is None else d - 2.0,
                  "frame_would_step": verdict == "stepped", "frame_verdict": verdict,
                  "verdict": verdict} for s, d in (("L", dl), ("R", dr))]
        eps.append({
            "id": eid, "label": label, "line": line, "group": group, "f0": f0, "f1": f1,
            "verdict": verdict, "verdict_ja": VERDICT_JA[verdict], "status": STATUS_JA,
            "reason": reason, "frame": frame, "clock": f"{sec // 60}:{sec % 60:02d}",
            "video_s": (frame - 1) / fps, "decisiveness": 0.8 if reason is None else 0.0,
            "distances": {"L": dl, "R": dr}, "uncertainty_cm": {"L": 3.5, "R": 4.0},
            "method": {"L": "ankle", "R": "ankle"}, "crop": crop, "overlay": overlay,
            "clip": clip, "clip_note": None,
            "counts": {v: int(v == verdict) * 3 for v in VERDICT_JA},
            "evidence": [{"side": s, "method": "ankle", "d": d, "unc": 3.5}
                         for s, d in (("L", dl), ("R", dr))],
        })
    eps.append({
        "id": "P5-top_line-3000", "label": "P5", "line": "top_line", "group": "top_line",
        "f0": 3000, "f1": 3010, "verdict": "undecidable", "verdict_ja": VERDICT_JA["undecidable"],
        "status": STATUS_JA, "reason": "no_box", "frame": None, "clock": None, "video_s": None,
        "decisiveness": 0.0, "distances": {"L": None, "R": None},
        "uncertainty_cm": {"L": None, "R": None}, "method": {"L": None, "R": None},
        "crop": None, "overlay": None, "clip": None,
        "clip_note": "clips failed: RuntimeError: no frames in range",
        "counts": {"stepped": 0, "clear": 0, "undecidable": 11}, "evidence": []})
    _write(lc / "autocourt.json", json.dumps({"confidence": 0.9}))
    _write(lc / "frames.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    if os.environ.get("FAKE_ENGINE_LC_GEOMETRY") != "none":
        _write(lc / "geometry.json", json.dumps({
            "status": STATUS_JA, "source": "tid_meta", "H": behind_top_line_H(),
            "H_frames": None, "lens": None, "lens_status": None, "h_unc_cm": 0.0,
            "h_unc_group": None, "notes": [], "autocourt": None}, ensure_ascii=False))
    else:
        (lc / "geometry.json").unlink(missing_ok=True)
    prov = _write(lc / "provenance.json", json.dumps({
        "status": STATUS_JA, "video_id": a.video_id, "source": "engine",
        "parameters": {"source": a.source, "tid_video": a.tid_video,
                       "pose_weights": a.pose_weights, "device": a.device,
                       "limit_episodes": a.limit_episodes, "clips": not a.no_clips}},
        ensure_ascii=False))
    doc = _write(lc / "episodes.json", json.dumps(
        {"status": STATUS_JA, "video_id": a.video_id, "source": "engine", "fps": fps,
         "dir": str(lc), "episodes": eps}, ensure_ascii=False, indent=1))
    if os.environ.get("FAKE_ENGINE_OUTSIDE"):
        doc = Path("/tmp/elsewhere/episodes.json")
    labels = {}
    try:
        labels = json.loads((lc / "labels.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    verdicts = {e["id"]: e["verdict"] for e in eps}
    tally = {v: sum(1 for e in eps if e["verdict"] == v) for v in VERDICT_JA}
    _out({"episodes": len(eps), "verdicts": tally, "source": "engine", "clips_s": 0.0,
          "episodes_json": str(doc), "provenance": str(prov),
          "labels": _label_precision(verdicts, labels if isinstance(labels, dict) else {}),
          "status": STATUS_JA})


def _live_record(i: int, t: float) -> dict:
    """The i-th record (0-based) after `start`, in the shape of live_run.py's records."""
    k = 1 + int(t * 30)
    alert_id = f"2-sideline_L-{1 + 20 * (i // 20)}"   # one alert (touch, cross, end) per 20
    dwell_id = f"3-arc-dwell-{1 + 40 * (i // 40)}"     # one dwell (start, end) per 40
    if i % 10 == 9:   # FAKE_ENGINE_LIVE_DROPPED: frames dropped per stats (a busy machine)
        dropped = min(k, int(os.environ.get("FAKE_ENGINE_LIVE_DROPPED") or 0) * (1 + i // 10))
        return {"type": "stats", "t": t, "frames_in": k, "frames_processed": k - dropped,
                "dropped": dropped,
                "throughput_fps": 29.97, "input_fps": 29.98, "suppressed_tracks": 1,
                "suppressed_entries": 0, "in_play_tracks": 4, "inherited": 0,
                "reader_skipped": 0, "reader_lag_s": {"max": 0.01, "current": 0.0},
                "dropped_slot": 0,
                "latency": {"p50_ms": 16.1, "p90_ms": 18.5, "p95_ms": 19.1, "max_ms": 30.2,
                            "n": k}}
    if i % 20 == 3:
        return {"type": "alert", "t": t, "frame": k, "alert_id": alert_id,
                "player": 2, "line": "sideline_L", "event": "touch", "side": "in",
                "distance_cm": 4.0, "needs_check": True}
    if i % 20 == 6:
        return {"type": "alert", "t": t, "frame": k, "alert_id": alert_id, "player": 2,
                "line": "sideline_L", "event": "cross", "side": "out", "distance_cm": -18.5,
                "needs_check": False}
    if i % 20 == 8:
        return {"type": "alert_end", "t": t, "alert_id": alert_id,
                "result": "cross", "needs_check": False, "min_distance_cm": 4.0,
                "duration_s": 0.167, "reason": "exit"}
    if i % 40 == 12:
        return {"type": "dwell", "t": t, "frame": k, "dwell_id": dwell_id, "player": 3,
                "line": "arc", "label": "ライン付近に滞在", "t_start": max(0.0, t - 3.0),
                "side": "outside", "distance_cm": -15.2}
    if i % 40 == 16:
        return {"type": "dwell_end", "t": t, "dwell_id": dwell_id, "player": 3,
                "line": "arc", "label": "ライン付近に滞在", "duration_s": 3.4,
                "min_distance_cm": 8.1, "crossings": 0, "reason": "exit"}
    players = [{"id": p, "x": round(200.0 + 250 * p + 10 * (i % 7), 1),
                "y": round(150.0 + 120 * p, 1), "in_play": p != 5} for p in range(1, 6)]
    players[1].update(near_line="sideline_L", near_cm=12.0)
    return {"type": "positions", "t": t, "frame": k, "players": players}


def cmd_live(a) -> None:
    """Mirrors the engine's live: <tracks>/<id>/live/<session>/{geometry.json,stream.jsonl};
    a `start` record, then one record every 0.1 s (positions, alert, alert_end, dwell,
    dwell_end, stats) until SIGTERM or FAKE_ENGINE_LIVE_RECORDS, then `end`; prints the end
    record plus {stream, session, h_source}. Exit 0."""
    stop = {"set": False}
    stubborn = os.environ.get("FAKE_ENGINE_STUBBORN")
    signal.signal(signal.SIGTERM, signal.SIG_IGN if stubborn
                  else (lambda *_: stop.update(set=True)))
    signal.signal(signal.SIGINT, lambda *_: stop.update(set=True))

    def fail(msg: str) -> None:
        print(msg, file=sys.stderr)
        print(json.dumps({"error": msg}))
        sys.exit(1)

    entry = _entry_path(a.registry, a.video_id)
    if entry is None:
        fail(f"no registry entry for {a.video_id}")
    session = a.session or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session):
        fail(f"invalid --session {session!r}")
    d = Path(a.tracks) / a.video_id / "live" / session
    if d.exists():
        fail(f"live session {session} exists: {d}")
    source = a.source or str(entry)
    if "://" not in source and not Path(source).is_file():
        fail(f"--source {source}: no such file")
    d.mkdir(parents=True)
    if stubborn:   # the real child: <tracker python> scripts/live_run.py ... --parent-pid <us>
        child = subprocess.Popen(
            [sys.executable, "-c", "import signal, time; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)",
             str(Path(__file__).parent / "scripts" / "live_run.py"), "--out", str(d),
             "--parent-pid", str(os.getpid())], start_new_session=True,
            stdin=subprocess.DEVNULL)
        Path(stubborn).write_text(f"{os.getpid()}\n{child.pid}\n", encoding="utf-8")
    h_source = "tid_meta" if a.tid_video else "autocourt"
    _write(d / "geometry.json", json.dumps({"H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 1.0]],
                                            "ref_size": [1920, 1080], "lens": None,
                                            "h_unc_cm": 0.0 if a.tid_video else 15.0,
                                            "h_source": h_source}))
    stream = d / "stream.jsonl"
    limit = int(os.environ.get("FAKE_ENGINE_LIVE_RECORDS") or 1200)
    t0 = time.monotonic()
    n_alerts = 0
    with open(stream, "w", encoding="utf-8") as fh:
        def emit(rec: dict) -> None:
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()
        emit({"type": "start", "t": 0.0, "status": STATUS_JA, "fps": 30.0, "stride": a.stride,
              "unc_cm": {"sideline": 13.0, "arc": 12.0, "baseline": 32.0, "top_line": 10.0},
              "bias_cm": {"sideline": -6.1, "arc": 2.0, "baseline": -16.2, "top_line": 2.1},
              "alerts": "event", "dwell_s": 3.0, "all_tracks": a.all_tracks,
              "in_play_cm": 100.0, "source": source, "h_source": h_source})
        i, reason = 0, "eof"
        while i < limit:
            if stop["set"]:
                reason = "stop"
                break
            rec = _live_record(i, round(time.monotonic() - t0, 3))
            n_alerts += rec["type"] == "alert" and rec["event"] == "touch"
            emit(rec)
            i += 1
            time.sleep(0.1)
        if stop["set"]:
            reason = "stop"
        elapsed = round(time.monotonic() - t0, 3)
        frames = 1 + int(elapsed * 30)
        dropped = min(frames, int(os.environ.get("FAKE_ENGINE_LIVE_DROPPED") or 0) * (1 + i // 10))
        end = {"frames_in": frames, "frames_processed": frames - dropped, "dropped": dropped,
               "throughput_fps": 29.97, "input_fps": 29.98, "suppressed_tracks": 1,
               "suppressed_entries": 0, "in_play_tracks": 4, "inherited": 0,
               "reader_skipped": 0, "reader_lag_s": {"max": 0.01, "current": 0.0},
               "dropped_slot": 0,
               "latency": {"p50_ms": 16.1, "p90_ms": 18.5, "p95_ms": 19.1, "max_ms": 30.2,
                           "n": frames},
               "alerts": n_alerts, "events": {"touch": n_alerts, "cross": n_alerts,
                                              "dwell": 0, "dwell_crossings": 0},
               "stream": str(stream), "elapsed_s": elapsed, "fps": 30.0, "stride": a.stride,
               "skipped_stride": 0, "reason": reason, "type": "end", "t": elapsed,
               "status": STATUS_JA}
        emit(end)
    _out({**end, "stream": str(stream), "session": session, "h_source": h_source})


def cmd_serve(a) -> None:
    """Fake serve.sh: start a long-lived 'vllm' in its own session, record serve.pid."""
    _fail_if_asked("serve")
    runtime = Path(os.environ["PFB_RUNTIME_DIR"])
    runtime.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "vllm", "serve",
                             a.model, "--port", a.port],
                            start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    (runtime / "serve.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"started pid {proc.pid}")
    time.sleep(float(os.environ.get("FAKE_ENGINE_SERVE_SLEEP") or 0))
    print("ready after 0 s")


def cmd_stop(a) -> None:
    """Fake stop_server.sh: stop the process group named by serve.pid."""
    code = os.environ.get("FAKE_ENGINE_STOP_EXIT")
    if code:
        print("fake stop failure", file=sys.stderr)
        sys.exit(int(code))
    pidf = Path(os.environ["PFB_RUNTIME_DIR"]) / "serve.pid"
    if not pidf.is_file():
        print("nothing to stop")
        return
    pid = int(pidf.read_text().strip())
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    pidf.unlink()
    print("stopped")


def _frame_args(s) -> None:
    g = s.add_mutually_exclusive_group()
    g.add_argument("--auto-start", action="store_true")
    g.add_argument("--frame", type=int, default=None)
    s.add_argument("--window-s", type=float, default=5.0)


def main(argv: list[str]) -> None:
    if argv[:2] == ["-m", "pfb3x3.engine"]:
        argv = argv[2:]
    _log_call(argv)
    p = argparse.ArgumentParser(prog="fake_engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("track")
    for opt in ("--video-id", "--registry", "--tracks", "--device", "--tracker-python"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_track)

    s = sub.add_parser("sheet")
    for opt in ("--video-id", "--registry", "--tracks", "--out"):
        s.add_argument(opt, required=True)
    _frame_args(s)
    s.set_defaults(fn=cmd_sheet)

    s = sub.add_parser("frames")
    for opt in ("--video-id", "--registry", "--tracks", "--frames", "--condition"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_frames)

    s = sub.add_parser("quality")
    for opt in ("--video-id", "--registry", "--tracks"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_quality)

    s = sub.add_parser("rule-names")
    for opt in ("--video-id", "--registry", "--tracks", "--positions"):
        s.add_argument(opt, required=True)
    s.add_argument("--out")
    _frame_args(s)
    s.set_defaults(fn=cmd_rule_names)

    s = sub.add_parser("propose")
    for opt in ("--video-id", "--registry", "--tracks", "--appearance", "--base-url", "--model"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_propose)

    s = sub.add_parser("behavior")
    for opt in ("--video-id", "--registry", "--tracks", "--frames", "--exp", "--model",
                "--condition", "--base-url"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_behavior)

    s = sub.add_parser("transcribe")
    for opt in ("--video-id", "--registry", "--tracks", "--whisper-python"):
        s.add_argument(opt, required=True)
    s.set_defaults(fn=cmd_transcribe)

    s = sub.add_parser("linecall")   # the real subparser's options
    for opt in ("--video-id", "--registry", "--tracks"):
        s.add_argument(opt, required=True)
    s.add_argument("--source", choices=("auto", "tid", "engine"), default="auto")
    for opt in ("--tid-root", "--tid-video", "--tid-pred", "--tracker-python", "--tid-python",
                "--pose-weights"):
        s.add_argument(opt, default=None)
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit-episodes", type=int, default=None)
    s.add_argument("--no-clips", action="store_true")
    s.set_defaults(fn=cmd_linecall)

    s = sub.add_parser("live")   # the real subparser's options
    for opt in ("--video-id", "--registry", "--tracks"):
        s.add_argument(opt, required=True)
    for opt in ("--source", "--session", "--tid-root", "--tid-video", "--tracker-python",
                "--tid-python", "--weights"):
        s.add_argument(opt, default=None)
    s.add_argument("--duration-s", type=float, default=None)
    s.add_argument("--stride", type=int, choices=(1, 2), default=1)
    s.add_argument("--device", default="cuda")
    s.add_argument("--all-tracks", action="store_true")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("serve")
    s.add_argument("model")
    s.add_argument("port")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("stop")
    s.set_defaults(fn=cmd_stop)

    a = p.parse_args(argv)
    if a.cmd not in ("serve", "stop"):
        _maybe_sleep(a.cmd)
        _fail_if_asked(a.cmd)
    orphan = os.environ.get("FAKE_ENGINE_ORPHAN")
    if orphan and a.cmd not in ("serve", "stop"):   # a child left in our group after we exit
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        Path(orphan).write_text(f"{child.pid}\n", encoding="utf-8")
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
