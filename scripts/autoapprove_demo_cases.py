"""Public-clip demo cases for the automatic approval (a COPY of the demo database is changed).

    PYTHONPATH=src python scripts/autoapprove_demo_cases.py COPY_OF_APP.DB [--inject]

The demo database holds the processing result of a public clip (Samsunspor, TBF, CC BY 3.0:
two cut-outs of 180 s and 60 s). Starting from one person whose scenes the number timeline reads
as the same player in every cited scene (the 60 s clip's 「#20」), this builds the cases the
checks must decide and prints each verdict:

  A   the person's own text as made (stopped only because its action words were never re-checked)
  A+  the same text with the action claims recorded as re-checked and supported (`--inject`:
      the verdicts are written by this script for the demo, no video was re-checked)
  B   a scene of another person among the cited scenes (a mixed-up text)
  C   the text hedged (「…ように見える（要確認）」)
  D   a jersey number written as a fact (「20番が…」)
  E   a comparison between players
  F   the text of a person whose number is read on three tracks (the 180 s clip's 「選手7」), as made

Only the copy is written (new feedback runs for the cases, the behavior run's claim_checks with
--inject). The live data/app.db is never opened by this script.
"""
import argparse
import json
import sys
from pathlib import Path

from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline import claim_check
from pfb_app.services import autoapprove, review
from pfb_app.storage.db import Repo, connect, init_schema


def put(repo: Repo, base: StageRun, text: str, refs) -> StageRun:
    return repo.save_stage_run(StageRun(
        id=None, submission_id=base.submission_id, stage=Stage.pfb, camera=None,
        subject_id=base.subject_id, version=0, source="model",
        upstream_run_ids=list(base.upstream_run_ids), prompt_hash=None, model_id=base.model_id,
        output={"pfb_status": "generated", "pfb": text, "evidence_refs": list(refs)},
        status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def show(repo: Repo, name: str, run: StageRun) -> dict:
    subject = repo.get_subject(run.subject_id)
    verdict = autoapprove.evaluate(repo, subject, run)
    row = {"case": name, "passed": verdict.passed, "held_by": [c.id for c in verdict.held],
           "reasons": verdict.reasons, "text": run.output["pfb"]}
    print(json.dumps(row, ensure_ascii=False))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--subject-a", type=int, default=21, help="the person of case A (a #20 of the 60 s clip)")
    ap.add_argument("--subject-f", type=int, default=7, help="the person of case F")
    args = ap.parse_args()
    conn = connect(Path(args.db))
    init_schema(conn)
    repo = Repo(conn)
    subject_a = repo.get_subject(args.subject_a)
    base = review._latest_pfb(repo, subject_a)   # noqa: SLF001
    own = list(base.output["evidence_refs"])
    ctx = autoapprove._context(repo, base)       # noqa: SLF001
    behavior = ctx.events[own[0]][0]
    others = [r for r, (_, e) in ctx.events.items() if e["participant"] != subject_a["participant_id"]]
    rows = []
    if args.inject:    # the re-check's verdicts for the person's scenes, written for the demo
        out = dict(behavior.output)
        verdicts = {}
        for ref in own:
            action = claim_check.claimed_action(ctx.events[ref][1].get("observed_behavior", ""))
            if action == claim_check.NO_CLAIM:
                continue
            verdicts[ref] = claim_check.Verdict(action, "supported", "high", "keep",
                                                ["デモ用に入れた値（実際の再確認ではない）"]).as_dict()
        text = base.output["pfb"]
        for action in ("defend", "pass", "shoot", "dribble", "run"):
            if action in autoapprove.actions_stated(text) and action not in {v["action"] for v in verdicts.values()}:
                verdicts[own[0]] = dict(claim_check.Verdict(action, "supported", "high", "keep", []).as_dict())
        out["claim_checks"] = verdicts
        conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (json.dumps(out, ensure_ascii=False), behavior.id))
        base = put(repo, base, base.output["pfb"], own)    # a fresh run, as the pipeline would write it
        rows.append(show(repo, "A+ (claims recorded as supported: demo values)", base))
    else:
        rows.append(show(repo, "A (as made)", base))
    text = base.output["pfb"]
    rows.append(show(repo, "B (a scene of another person cited)",
                     put(repo, base, text, [*own[:2], others[0]])))
    rows.append(show(repo, "C (hedged)", put(repo, base, claim_check.hedge(text.rstrip("。")), own)))
    rows.append(show(repo, "D (jersey number as a fact)",
                     put(repo, base, "20番の選手が柱の近くで動いていました。" + text, own)))
    rows.append(show(repo, "E (comparison)", put(repo, base, "ほかの選手より多く動いていました。" + text, own)))
    subject_f = repo.get_subject(args.subject_f)
    rows.append(show(repo, "F (one number read on three tracks)", review._latest_pfb(repo, subject_f)))   # noqa: SLF001
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
