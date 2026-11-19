#!/usr/bin/env python3
"""Fusion check on the engine's 3x3 cue dumps (numbers only).

    uv run python scripts/eval_fusion.py PROPOSALS_JSONL [--out DIR] [--min-agree K]
        [--quality-tsv QUALITY_TSV --min-spread X]

PROPOSALS_JSONL is the engine's results/cue_eval/<run>/proposals.jsonl (GT boxes) or
proposals.tracker*.jsonl (tracker boxes): one row per person label with the voted number,
frames_agree, the matched GT person (gt_label, jersey, team; null for a tracker label that
matches nobody) and the team cluster (null when the team cue made none).

Per video a GT roster is built (one row per GT person: number = jersey, team). Then, for each
GT team as the coach's own team:
- mode "clusters" (the dump has clusters): own = the cluster mapped one-to-one to that GT
  team; number_candidates -> restrict_to_team (team names from match_team_names) -> fuse,
  over the own cluster's labels and the whole roster;
- mode "own-team roster" (no clusters, the team cue off as in the app today): the roster is
  the own team's rows only, every label is a candidate (own=None) -> fuse.
A proposal is right when its row is the label's GT person. precision = correct / n;
coverage = n / GT people (the engine's definition). The grid THRESHOLD x MARGIN is scored on
the engine's tune videos; the point with the highest coverage at precision >= 0.90 is chosen
(the provisional constants first, so they win ties) and judged on the held-out videos
(precision >= 0.90 with n >= 20, the engine gate's rule).
--quality-tsv/--min-spread drop labels whose track spread is below X (the engine's
track-quality filter; dropped labels make no proposal, GT people still count).

Writes DIR/fusion.tsv and DIR/log.txt (text only). The last log line starts "held-out".
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.pipeline.name_fusion import (
    MARGIN, THRESHOLD, fuse, match_team_names, normalise_number, number_candidates,
    restrict_to_team,
)
from pfb_app.services.roster import RosterEntry

TUNE = ("IMG_0104", "IMG_0105", "IMG_0106", "IMG_0107")        # the engine's split
HELD_OUT = ("IMG_0108", "IMG_0109", "IMG_0110", "IMG_0111")
THRESHOLDS = (0.4, 0.6, 0.8, 1.0)
MARGINS = (0.1, 0.2, 0.3, 0.5)
GATE = 0.90
MIN_N = 20                      # the engine gate's minimum number of proposals
COLUMNS = ("split", "threshold", "margin", "precision", "wilson_lo", "wilson_hi", "coverage",
           "n", "correct", "people", "chosen")
log = logging.getLogger("eval_fusion")


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = correct / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def load(path: Path) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_video[row["video"]].append(row)
    return dict(by_video)


def load_spread(path: Path) -> dict[tuple[str, str], float]:
    with open(path, encoding="utf-8") as f:
        return {(r["video"], r["label"]): float(r["spread_frac"])
                for r in csv.DictReader(f, delimiter="\t")}


def roster(rows: list[dict]) -> tuple[list[RosterEntry], dict[str, str]]:
    """One roster row per GT person (row id = gt_label) and {gt_label: team}."""
    people: dict[str, RosterEntry] = {}
    team_of: dict[str, str] = {}
    for r in rows:
        gt = r.get("gt_label")
        if not gt or gt in people:
            continue
        number = normalise_number(str(r["jersey"])) if r.get("jersey") is not None else None
        team = "" if r.get("team") is None else str(r["team"])
        people[gt] = RosterEntry(gt, gt, {"number": number or "", "team": team})
        team_of[gt] = team
    return list(people.values()), team_of


def _cluster_of_team(rows: list[dict], gt_teams: list[str]) -> dict[str, str]:
    """{GT team: cluster}, one-to-one, by the most labels of that GT team in the cluster."""
    clusters = sorted({r["cluster"] for r in rows if r.get("cluster") not in (None, "other")})
    count: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r.get("cluster") in clusters and r.get("team") is not None:
            count[(str(r["team"]), r["cluster"])] += 1
    best, best_total = {}, -1
    k = min(len(gt_teams), len(clusters))
    for teams in itertools.permutations(gt_teams, k):
        for cl in itertools.permutations(clusters, k):
            total = sum(count[(t, c)] for t, c in zip(teams, cl))
            if total > best_total:
                best, best_total = dict(zip(teams, cl)), total
    return best


def cases(rows: list[dict], *, min_agree: int, keep=None):
    """[(cands, labels, row ids, people of the own team, {label: gt_label})], one per GT team."""
    entries, team_of = roster(rows)
    kept = [r for r in rows if keep is None or keep(r)]
    reads = {r["label"]: {"number": r.get("number"), "votes": r.get("votes", 0),
                          "frames_agree": r.get("frames_agree", 0)} for r in kept}
    truth = {r["label"]: r.get("gt_label") for r in rows}
    gt_teams = sorted(set(team_of.values()))
    out = []
    if any(r.get("cluster") is not None for r in rows):
        teams = {r["label"]: r.get("cluster") or "other" for r in rows}
        own_of = _cluster_of_team(rows, gt_teams)
        team_names = match_team_names(reads, entries, teams, min_frames_agree=min_agree)
        cands_all = number_candidates(reads, entries, min_frames_agree=min_agree)
        rows_ids = [e.participant_id for e in entries]
        for t in gt_teams:
            people = sum(1 for v in team_of.values() if v == t)
            own = own_of.get(t)
            if own is None:
                out.append(([], [], rows_ids, people, truth))
                continue
            cands = restrict_to_team(cands_all, teams, own, team_names, entries)
            labels = [l for l, c in teams.items() if c == own]
            out.append((cands, labels, rows_ids, people, truth))
    else:
        labels = [r["label"] for r in rows]
        for t in gt_teams:
            own_entries = [e for e in entries if team_of[e.participant_id] == t]
            cands = number_candidates(reads, own_entries, min_frames_agree=min_agree)
            out.append((cands, labels, [e.participant_id for e in own_entries],
                        len(own_entries), truth))
    return out


def score(prepared: list, threshold: float, margin: float) -> dict:
    n = correct = people = 0
    for cands, labels, rows_ids, ppl, truth in prepared:
        people += ppl
        for label, prop in fuse(cands, labels, rows_ids, threshold=threshold,
                                margin=margin).items():
            n += 1
            correct += int(truth.get(label) == prop.row)
    lo, hi = wilson(correct, n)
    return {"n": n, "correct": correct, "people": people,
            "precision": correct / n if n else None, "wilson_lo": lo, "wilson_hi": hi,
            "coverage": n / people if people else None}


def grid() -> list[tuple[float, float]]:
    points = [(t, m) for t in THRESHOLDS for m in MARGINS]
    points.remove((THRESHOLD, MARGIN))
    return [(THRESHOLD, MARGIN)] + points


def choose(tune: dict[tuple[float, float], dict]) -> tuple[float, float] | None:
    best, best_cov = None, -1.0
    for point in grid():
        s = tune[point]
        if s["n"] and s["precision"] >= GATE - 1e-9 and s["coverage"] > best_cov + 1e-9:
            best, best_cov = point, s["coverage"]
    return best


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _summary(s: dict) -> str:
    if not s["n"]:
        return f"no proposals (people {s['people']})"
    return (f"precision {s['precision']:.3f} ({s['correct']}/{s['n']}, 95% "
            f"{s['wilson_lo']:.3f}-{s['wilson_hi']:.3f}), coverage {s['coverage']:.3f} "
            f"(people {s['people']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("proposals", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-agree", type=int, default=2,
                    help="frames_agree a read needs (the engine's number gate k; default 2)")
    ap.add_argument("--quality-tsv", type=Path, default=None)
    ap.add_argument("--min-spread", type=float, default=None)
    args = ap.parse_args(argv)
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("results/fusion") / utc
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        log.info(text)

    dump = load(args.proposals)
    keep = None
    if args.quality_tsv is not None and args.min_spread is not None:
        spread = load_spread(args.quality_tsv)
        keep = lambda r: spread.get((r["video"], r["label"]), 0.0) >= args.min_spread  # noqa: E731
    mode = ("clusters" if any(r.get("cluster") is not None for rs in dump.values() for r in rs)
            else "own-team roster")
    say(f"eval_fusion {utc} dump {args.proposals}")
    say(f"mode {mode}; min_agree {args.min_agree}; quality "
        f"{'min_spread ' + str(args.min_spread) if keep else 'off'}; "
        f"provisional THRESHOLD {THRESHOLD} MARGIN {MARGIN}")
    unknown = sorted(set(dump) - set(TUNE) - set(HELD_OUT))
    if unknown:
        say(f"videos outside the split ignored: {', '.join(unknown)}")
    prepared = {split: [c for v in videos if v in dump
                        for c in cases(dump[v], min_agree=args.min_agree, keep=keep)]
                for split, videos in (("tune", TUNE), ("held_out", HELD_OUT))}
    results = {split: {p: score(prep, *p) for p in grid()} for split, prep in prepared.items()}
    chosen = choose(results["tune"])
    distinct = {(s["n"], s["correct"]) for s in results["tune"].values()}
    if len(distinct) == 1:
        say("tune: every grid point gives the same proposals (one cue: each pair scores 1.0 "
            "or 0, so the threshold and margin cannot discriminate)")
    with open(out / "fusion.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for split in ("tune", "held_out"):
            for point in grid():
                s = results[split][point]
                f.write("\t".join([split, _fmt(point[0]), _fmt(point[1])]
                                  + [_fmt(s[k]) for k in COLUMNS[3:10]]
                                  + ["1" if point == chosen else "0"]) + "\n")
    if chosen is None:
        point = (THRESHOLD, MARGIN)
        say(f"tune: no point reaches {GATE:.2f}; best at the provisional point: "
            + _summary(results["tune"][point]))
        say(f"constants stay provisional: THRESHOLD {THRESHOLD} MARGIN {MARGIN}")
    else:
        point = chosen
        say(f"tune: chosen THRESHOLD {point[0]} MARGIN {point[1]}: "
            + _summary(results["tune"][point]))
    held = results["held_out"][point]
    say(f"held-out at THRESHOLD {point[0]} MARGIN {point[1]}: " + _summary(held))
    if chosen is not None and held["n"] >= MIN_N and held["precision"] >= GATE - 1e-9:
        say(f"held-out precision ≥ {GATE:.2f}")
    elif chosen is not None and held["n"] and held["precision"] >= GATE - 1e-9:
        say(f"held-out precision {held['precision']:.3f} ≥ {GATE:.2f} but n {held['n']} < "
            f"{MIN_N} (the engine's gate minimum): the number cue stays gated by the engine file")
    else:
        prec = "n/a" if held["precision"] is None else f"{held['precision']:.3f}"
        say(f"held-out precision {prec} < {GATE:.2f} or not chosen on tune: the number cue "
            f"stays gated by the engine file")
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'fusion.tsv'}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
