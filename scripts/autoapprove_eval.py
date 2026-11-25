"""Run the automatic-approval checks over a COPY of a public-clip demo database and print the counts.

    PYTHONPATH=src python scripts/autoapprove_eval.py COPY_OF_APP.DB [--json OUT.json]

Reads every feedback run (`pfb`, every version) of the copy; never writes to it beyond what
`init_schema` adds (open a copy, never the live data/app.db). Prints, for the generated texts,
how many pass every check, which checks stopped the others (a text can be stopped by several),
the same counts if the action-claim check were waived (what the other checks alone would let
through), and sentence-level counts of the words the checks look for.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

from pfb_app.domain.models import Stage
from pfb_app.services import autoapprove, review
from pfb_app.storage.db import Repo, connect, init_schema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--json")
    args = ap.parse_args()
    conn = connect(Path(args.db))
    init_schema(conn)
    repo = Repo(conn)
    runs = conn.execute("SELECT id FROM stage_runs WHERE stage=? ORDER BY id", (Stage.pfb.value,)).fetchall()
    total = collections.Counter()
    why = collections.Counter()
    without_claims = []
    rows = []
    sentences = collections.Counter()
    for (rid,) in runs:
        run = repo.get_run(rid)
        subject = repo.get_subject(run.subject_id)
        out = run.output or {}
        total["runs"] += 1
        if out.get("pfb_status") != "generated":
            total["not_generated"] += 1
            continue
        total["generated"] += 1
        verdict = autoapprove.evaluate(repo, subject, run)
        ids = [c.id for c in verdict.held]
        total["passed"] += verdict.passed
        for i in ids:
            why[i] += 1
        if set(ids) <= {"claims"}:
            without_claims.append(rid)
        text = out.get("pfb") or ""
        parts = autoapprove._sentences(text)   # noqa: SLF001
        sentences["sentences"] += len(parts)
        for sentence in parts:
            sentences["claim_an_action"] += bool(autoapprove.actions_stated(sentence))
            sentences["jersey_as_fact"] += bool(autoapprove.jersey_problems(sentence))
            sentences["comparison_or_grading"] += bool(autoapprove.wording_problems(sentence))
        rows.append({"run": rid, "subject": subject["display_name"], "version": run.version,
                     "passed": verdict.passed, "held_by": ids,
                     "reasons": verdict.reasons, "text": text,
                     "refs": out.get("evidence_refs")})
    result = {"counts": dict(total), "held_by_check": dict(why),
              "would_pass_if_claims_were_waived": without_claims, "sentences": dict(sentences),
              "rows": rows}
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
