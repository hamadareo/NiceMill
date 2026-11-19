"""Record a student-PDF download for works settled before downloads were recorded.

  env -u VIRTUAL_ENV ~/.local/bin/uv run python scripts/backfill_pdf_downloaded.py --db PATH [--apply]

The wizard's home screen shows a work as 完了 only once its student PDF was downloaded
(`assignments.pdf_downloaded_at`), and the old 「PDF出力」 page did not record that. For every
work without a recorded download whose students are all approved or have too little evidence
(and at least one is approved), this sets `pdf_downloaded_at` to the latest approval time.

- Dry run by default: prints the works it would change and changes nothing; `--apply` writes.
- Idempotent: a work that already has a value is never touched, so a second run changes
  nothing.
- Never creates a database and never changes the schema: the column must already be there
  (the app adds it on start). Back the database up before `--apply`.
"""
import argparse
import sys
from pathlib import Path

from pfb_app.domain.models import Kind, SubjectStatus
from pfb_app.services import review
from pfb_app.storage.db import Repo, connect

SETTLED = {SubjectStatus.approved, SubjectStatus.insufficient}


def settled_works(repo: Repo) -> list[tuple[int, str, str]]:
    """[(assignment id, title, latest approval time)] of the works to backfill."""
    out = []
    for a in repo.list_assignments():
        if a.get("pdf_downloaded_at"):
            continue
        kind = Kind(a["kind"])
        subjects = [s for sub in repo.list_submissions(a["id"])
                    for s in repo.list_subjects(sub["id"])
                    if kind == Kind.document or s["participant_id"]]
        statuses = {s["id"]: review.subject_status(repo, s["id"]) for s in subjects}
        if not subjects or not set(statuses.values()) <= SETTLED:
            continue
        approved = [s["approved_at"] for s in subjects
                    if statuses[s["id"]] == SubjectStatus.approved and s["approved_at"]]
        if approved:
            out.append((a["id"], a["title"], max(approved)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, type=Path, help="the SQLite database file")
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        print(f"no database at {args.db}", file=sys.stderr)
        return 2
    conn = connect(args.db)
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(assignments)")}
        if "pdf_downloaded_at" not in columns:
            print("assignments.pdf_downloaded_at is missing: start the app once first",
                  file=sys.stderr)
            return 2
        works = settled_works(Repo(conn))
        verb = "set" if args.apply else "would set"
        for aid, title, at in works:
            print(f"{verb} pdf_downloaded_at={at} on assignment {aid} ({title})")
        if args.apply and works:
            with conn:
                conn.execute("BEGIN")
                for aid, _, at in works:
                    conn.execute("UPDATE assignments SET pdf_downloaded_at=? WHERE id=?"
                                 " AND pdf_downloaded_at IS NULL", (at, aid))
        print(f"{len(works)} work(s) {'changed' if args.apply else 'to change (dry run)'}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
