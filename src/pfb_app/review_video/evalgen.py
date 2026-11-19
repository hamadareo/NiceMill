"""python -m pfb_app.review_video.evalgen SUBMISSION_ID --settings S.yaml --out TEXTS.json --work-dir DIR
       [--claim-cues CUES.json] [--server http://127.0.0.1:PORT/v1] [--model NAME]

Makes an evaluation text for every player of a submission with the app's own feedback stage
(pipeline.video.run_pfb, the same prompt and checks) and the LOCAL model, but from the counted scenes that
passed the action check only (claims.eligible): the paragraph the app already made restates the
actions of every counted scene, also those the check weakened, so it is not reused.

Why it is done on a COPY of the database: run_pfb reads the counted scenes from the contribution
stage's judgments, and writes a run. The live database and the opened (read only) one are never written:
the copy lives in --work-dir, and in it the judgments of every scene that did not pass are set to "not
a counted scene". Nothing else is changed.

A person the stage finds too few scenes for (insufficient_positive_evidence) gets the app's own fact-only
short text (services.short_text.compose), marked as such; nothing is invented. The result file is
{"<subject id>": {"text", "source": "ai" | "short", "model", "seconds", "scenes", "why"}}; the texts are
checked once more when the player is built (claims.apply_eval_texts). Only this machine is asked
(the served client refuses any other host); no cloud service is used."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from pfb_app.config import load_settings
from pfb_app.domain.models import Stage
from pfb_app.review_video import claims, collect, service
from pfb_app.storage import db as dbm


def _copy_db(src: Path, work_dir: Path) -> Path:
    """A consistent copy of the database (SQLite backup API, the source opened read only)."""
    (work_dir / "data").mkdir(parents=True, exist_ok=True, mode=0o700)
    dst = work_dir / "data" / "app.db"
    if dst.exists():
        dst.unlink()
    a = sqlite3.connect(f"file:{src.resolve()}?mode=ro", uri=True)
    b = sqlite3.connect(str(dst))
    try:
        a.backup(b)
    finally:
        a.close()
        b.close()
    return dst


def _keep_only(repo, sid: int, keep: set[str]) -> int:
    """In the COPY: every judgment of a scene not in `keep` becomes "not counted". Returns how many."""
    n = 0
    for cam in repo.list_cameras(sid):
        run = repo.latest_run(sid, Stage.contribution, cam)
        out = dict(run.output or {})
        js = []
        for j in out.get("judgments") or []:
            if j.get("is_positive_contribution") and j.get("ref_id") not in keep:
                j = dict(j, is_positive_contribution=False)
                n += 1
            js.append(j)
        out["judgments"] = js
        repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (json.dumps(out, ensure_ascii=False), run.id))
    return n


def generate(repo_ro, settings, sid: int, *, cues: str | None, work_dir: Path, server: str,
             model: str | None, timeout: float = 600.0, max_tokens: int = 4096, done: dict | None = None, log=print) -> dict:
    from pfb_app.inference import served   # noqa: PLC0415 - the one module that may talk to the network (this machine)
    from pfb_app.pipeline import video      # noqa: PLC0415
    from pfb_app.pipeline.common import StageContext   # noqa: PLC0415
    from pfb_app.services import short_text   # noqa: PLC0415
    from pfb_app.storage import files   # noqa: PLC0415

    pkg = collect.build_package(repo_ro, settings, sid, claim_cues=cues)
    scenes = {p.subject_id: claims.eligible(p) for p in pkg.people if not p.excluded}
    keep = {m.ref_id for ms in scenes.values() for m in ms}
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    copy = _copy_db(Path(files.db_path(settings)), work_dir)
    # a thinking model spends most of its tokens before the JSON: the answer is room for the thought + the text
    wsettings = settings.model_copy(update={"data_dir": copy.parent,
                                            "max_tokens": {**settings.max_tokens, Stage.pfb.value: max_tokens}})
    conn = dbm.connect(copy)
    repo = dbm.Repo(conn, wsettings)
    dropped = _keep_only(repo, sid, keep)
    log(f"counted scenes kept for the texts: {len(keep)}; set to 'not counted' in the copy: {dropped}")
    name = model or next(i for i in served.list_models(server) if i and "embed" not in i.lower())
    backend = served.ServedChatBackend(server, name, timeout_sec=timeout, max_concurrency=1)
    ctx = StageContext(settings=wsettings, repo=repo, backend=backend, yield_check=lambda: False)
    assignment = repo.get_assignment(repo.get_submission(sid)["assignment_id"])
    short = short_text.compose(assignment)
    subjects = {s["id"]: s for s in repo.list_subjects(sid)}
    result: dict = {}
    for p in pkg.people:
        if p.excluded:
            continue
        if (done or {}).get(str(p.subject_id), {}).get("text"):      # --resume: kept as made before
            result[str(p.subject_id)] = done[str(p.subject_id)]
            continue
        t0 = time.time()
        entry = {"scenes": len(scenes[p.subject_id]), "model": backend.model_id}
        if not scenes[p.subject_id] or subjects[p.subject_id]["participant_id"] is None:
            run = None
        else:
            run = video.run_pfb(ctx, sid, p.subject_id)
        log(f"  -> run status {run.status if run is not None else '-'}")
        out = (run.output or {}) if run is not None and run.status == "ok" else {}
        if out.get("pfb_status") == "generated" and out.get("pfb"):
            entry.update(text=out["pfb"], source="ai", warnings=list(run.warnings or []))
        elif run is not None and run.status != "ok":
            entry.update(text=None, source=None, why="文章の生成に失敗しました: " + str(run.error)[:200])
        else:
            entry.update(text=short, source="short")
        entry["seconds"] = round(time.time() - t0, 1)
        result[str(p.subject_id)] = entry
        log(f"{p.subject_id} {p.shown()} scenes={entry['scenes']} {entry['source']} {entry['seconds']}s")
    conn.close()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pfb_app.review_video.evalgen")
    ap.add_argument("submission_id", type=int)
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--claim-cues", type=Path, default=None)
    ap.add_argument("--server", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resume", action="store_true", help="keep the texts already in --out, make the missing ones")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="the answer budget (a thinking model needs ~12000)")
    a = ap.parse_args(argv)
    settings = load_settings(a.settings)
    repo = service.open_readonly(settings)
    done = json.loads(a.out.read_text(encoding="utf-8")) if a.resume and a.out.is_file() else None
    res = generate(repo, settings, a.submission_id, cues=str(a.claim_cues) if a.claim_cues else None,
                   work_dir=a.work_dir, server=a.server, model=a.model, max_tokens=a.max_tokens, done=done)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(res)} people -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
