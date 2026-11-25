"""Re-check the action words of the cited scenes of every generated feedback text of a DATABASE COPY.

    PYTHONPATH=src python scripts/claim_scan_run.py COPY_OF_APP.DB [--settings pfb.yaml]
        [--vlm] [--vlm-server URL] [--vlm-model NAME] [--max-scenes N] [--force] [--only-run ID ...]

The same code the worker runs after a new text (pipeline.claim_scan): movement words from the
tracks always, shoot / dribble / pass / defend (pose cue + two questions to a local vision model
on this machine) with --vlm. Writes only the table claim_checks of the copy. Prints the time spent."""
import argparse
import sys
import time
from pathlib import Path

from pfb_app.config import Settings, load_settings
from pfb_app.domain.models import Stage
from pfb_app.pipeline import claim_scan
from pfb_app.storage.db import Repo, connect, init_schema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--settings", type=Path)
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--vlm-server")
    ap.add_argument("--vlm-model")
    ap.add_argument("--max-scenes", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-run", type=int, action="append")
    a = ap.parse_args()
    settings = load_settings(a.settings) if a.settings else Settings()
    upd = {"claim_check_enabled": a.vlm}
    if a.vlm_server:
        upd["claim_check_vlm_server"] = a.vlm_server
    if a.vlm_model:
        upd["claim_check_vlm_model"] = a.vlm_model
    if a.max_scenes:
        upd["claim_check_max_scenes"] = a.max_scenes
    settings = settings.model_copy(update=upd)
    conn = connect(Path(a.db))
    init_schema(conn)
    repo = Repo(conn, settings)
    rows = conn.execute("SELECT id FROM stage_runs WHERE stage=? ORDER BY id", (Stage.pfb.value,)).fetchall()
    t0 = time.time()
    tot = [0, 0, 0, 0]
    for (rid,) in rows:
        if a.only_run and rid not in a.only_run:
            continue
        run = repo.get_run(rid)
        if (run.output or {}).get("pfb_status") != "generated":
            continue
        t1 = time.time()
        try:
            r = claim_scan.scan_run(repo, settings, run, force=a.force)
        except Exception as e:   # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"run {rid}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        conn.commit()
        for i, v in enumerate((r.scenes, r.motion, r.asked, len(r.skipped))):
            tot[i] += v
        print(f"run {rid}: scenes={r.scenes} motion={r.motion} asked={r.asked} skipped={len(r.skipped)}"
              f" ({time.time() - t1:.0f}s)", flush=True)
    print(f"total scenes={tot[0]} motion={tot[1]} asked={tot[2]} skipped={tot[3]} {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
