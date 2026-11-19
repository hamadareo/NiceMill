"""Choose the people automatically for tracking works tracked before automatic people existed.

  env -u VIRTUAL_ENV ~/.local/bin/uv run python scripts/auto_people_backfill.py [--settings PATH] [--apply]

The automatic people (services.auto_people) are chosen when a track run finishes, so a work
tracked earlier has none and still waits at step 3. This lists the tracking (pfb3x3) video
groups that
- are not confirmed (identity_confirmed),
- have a usable track run on every registered camera,
- have no active blocking job,
- have no camera whose current mapping was saved by the coach (source "teacher"),
- and have at least one camera without a current mapping,
and with `--apply` runs `services.auto_people.reselect` on each (a new automatic mapping on
the existing track runs, then the automatic confirm). Confirmed works are never touched.

- The database is the app's: the settings YAML named by --settings, else $PFB_CONFIG, else
  the defaults (as the app). It is never created and its schema is never changed.
- Dry run by default: prints what it would do and changes nothing. Back the database up
  before `--apply`.
- Prints submission ids, camera counts and outcomes only (no names).
"""
import argparse
import os
import sys
from pathlib import Path

from pfb_app.config import load_settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.common import submission_engine, usable
from pfb_app.pipeline.tracked_identity import mapping_is_current
from pfb_app.services import auto_people, review
from pfb_app.storage.db import Repo, connect

TEACHER = "teacher"


def candidates(repo: Repo) -> list[tuple[int, int]]:
    """[(submission id, registered cameras)] of the groups to choose the people for."""
    out = []
    for a in repo.list_assignments():
        if Kind(a["kind"]) != Kind.video:
            continue
        for sub in repo.list_submissions(a["id"]):
            sid = sub["id"]
            cameras = list((sub["meta"] or {}).get("cameras") or {})
            if (submission_engine(sub) != "pfb3x3" or sub["identity_confirmed"] or not cameras
                    or repo.has_blocking_jobs(sid)):
                continue
            if not all(usable(repo.latest_run(sid, Stage.track, cam)) for cam in cameras):
                continue
            current = [cam for cam in cameras if mapping_is_current(repo, sid, cam)]
            if any(repo.latest_run(sid, Stage.identity, cam).source == TEACHER
                   for cam in current):
                continue
            if len(current) < len(cameras):
                out.append((sid, len(cameras)))
    return out


def _outcome(repo: Repo, settings, sid: int) -> str:
    try:
        ids = auto_people.reselect(repo, settings, sid)
    except review.BusyError:
        return "busy (skipped)"
    except review.ValidationError:
        return "refused"
    confirmed = repo.get_submission(sid)["identity_confirmed"]
    return f"{'confirmed' if confirmed else 'not confirmed'}, {len(ids)} job(s) queued"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, default=None,
                        help="the settings YAML (default: $PFB_CONFIG, else the defaults)")
    parser.add_argument("--apply", action="store_true", help="act (default: dry run)")
    args = parser.parse_args(argv)
    path = args.settings or (Path(os.environ["PFB_CONFIG"]) if os.environ.get("PFB_CONFIG")
                             else None)
    settings = load_settings(path)
    db = settings.data_dir / "app.db"
    if not db.is_file():
        print(f"no database at {db}", file=sys.stderr)
        return 2
    conn = connect(db)
    try:
        if "excluded" not in {r[1] for r in conn.execute("PRAGMA table_info(subjects)")}:
            print("subjects.excluded is missing: start the app once first", file=sys.stderr)
            return 2
        repo = Repo(conn, settings)
        groups = candidates(repo)
        for sid, cameras in groups:
            if args.apply:
                print(f"submission {sid} ({cameras} camera(s)): {_outcome(repo, settings, sid)}")
            else:
                print(f"would choose the people of submission {sid} ({cameras} camera(s))")
        print(f"{len(groups)} group(s) {'done' if args.apply else 'to do (dry run)'}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
