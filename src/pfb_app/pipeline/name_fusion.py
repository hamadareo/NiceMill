"""Name fusion (pure): the roster cues' candidates -> one proposal per person, or none.

Each cue turns a roster column plus an engine cue result into scored candidates
(person label -> roster row id). Fusion sums the scores per (label, row) (each cue counts
once per pair), picks a one-to-one assignment that maximises the total (Hungarian), and keeps
a pair only when its score is at least THRESHOLD and beats the runner-up (the best other row
for that person, or the best other person for that row) by at least MARGIN. Everything else
stays unassigned: proposals are only ever pre-fills that the coach confirms.

Cue inputs are plain dicts (the engine's JSON):
- numbers: {label: {"number": str | None, "votes": n, "frames_agree": n}}; a number is used
  only with frames_agree >= min_frames_agree (default MIN_FRAMES_AGREE = 2; the caller passes
  max(2, the gate's threshold for numbers)). A read that is not such a dict (no
  "frames_agree") counts as no read and is logged once per call. Numbers are normalised as in
  the engine: NFKC, "#"/"No." dropped, an Excel float's ".0" dropped, leading zeros dropped
  unless the number is all zeros ("00" stays "00", distinct from "0").
- positions: {label: position} plus "slots" {position: roster rows} and "claims"
  {position: labels}. A person's position scores against every roster row with that
  position: the full score when the position has one row, else
  score x min(1, slots / claims) x SHARED_FACTOR (ties between same-position rows are left
  to the other cues and the margin).
- grid: the order front first; roster value k <-> order[k-1].
- batting: the order by first turn at bat. Only the relative order is trusted (the video
  may start mid-lineup, and a batter may be missed): the observed order is a rotation of the
  lineup, aligned through anchors (people already placed by the other cues, see
  batting_anchors). At least two anchors must agree on one rotation offset (and no other
  offset as many), and all anchors must be concordant (rank correlation 1 under it);
  otherwise the cue proposes nothing. Only unanchored people get batting candidates, and only
  when the nearest anchor on each side is on the agreed offset (a missed batter shifts the
  offset after it).
- position, grid and batting expect the labels already filtered to the coach's own team when
  a team answer exists (Task 10 does that). A unique position slot or a grid place alone
  (0.6) meets THRESHOLD on purpose; a shared position slot alone (<= 0.3) never does.
- appearance / place: {label: name} from the engine's propose / rule-names, matched to the
  roster by row id, then by the raw roster name (attrs["name"], never display_name).
- teams: {label: "T1" | "T2" | "other"}; T1/T2 may swap between engine runs, so the own team
  is passed as a set of labels (or a cluster name of the same teams dict).

THRESHOLD and MARGIN: checked on the 3x3 dumps (Task 13, scripts/eval_fusion.py, the
engine's tune/held-out split, numbers at frames_agree >= 2), and kept at the provisional
0.6 / 0.3. With the number cue alone each pair scores 1.0 or 0, so every grid point
(THRESHOLD 0.4-1.0 x MARGIN 0.1-0.5) gives the same proposals; the margin's work there is to
drop a number read on two people. GT boxes (oracle tracking): held-out precision 0.941
(16/17, qwen3vl-8b) and 1.000 (20/20, 32b), coverage 0.71 / 0.83. Tracker boxes (the
engine's own tracking, own-team roster, no team cue): held-out 0.667 (2/3, 8b) and 0.833
(5/6, 32b), coverage 0.13 / 0.25, n far below the engine gate's 20; with the track-quality
filter (spread >= 0.05, frames_agree 3) 6/8 (8b) and 6/6 (32b). No tracker-box point reaches
0.90 with n >= 20, so the number cue stays gated by the engine file. These constants start to
matter once cues are combined.

The shipped gate (engine 8815ccf, unchanged in 20f0a65, 7bba64a and 58f8341;
configs/cue_gate.yaml) disables every cue for both models:
number (tracker boxes + quality filter, k=3) held-out 6/7 = 0.857 (qwen3vl-8b) and 6/6 = 1.000
(32b), both with n < 20; team precision 0.909 (20/22) but coverage 0.18 < 0.50 (a filtering
cue needs 0.50); position, grid, batting, appearance and place have no labelled material.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

CUE_SCORES = {"number": 1.0, "place": 1.0, "position": 0.6, "grid": 0.6,
              "appearance": 0.45, "batting": 0.3}
THRESHOLD = 0.6
MARGIN = 0.3
SHARED_FACTOR = 0.5          # a position several roster rows share
MIN_FRAMES_AGREE = 2         # a read number counts only when read in this many frames
MAX_DIGITS = 3
MIN_ANCHORS = 2              # batting: anchors that must agree on one rotation
_EPS = 1e-9
log = logging.getLogger(__name__)

_REASONS = {"position": "守備位置から推定", "grid": "スタート位置から推定",
            "batting": "打順から推定（弱い）", "appearance": "外見から推定",
            "place": "初期位置から推定"}
_NUMBER = re.compile(r"(?:#|no\.?\s*)?\s*(\d+)(?:\.0+)?", re.I)
_INT = re.compile(r"(\d+)(?:\.0+)?")


@dataclass(frozen=True)
class Candidate:
    label: str
    row: str
    cue: str
    score: float
    detail: str = ""


@dataclass(frozen=True)
class Proposal:
    row: str
    score: float
    reasons: tuple[str, ...]


def reason_text(cue: str, detail: str, target: str = "person") -> str:
    if cue == "number":
        return f"番号{detail}から推定" if target == "vehicle" else f"背番号{detail}から推定"
    return _REASONS[cue]


# ---------------------------------------------------------------- normalisation

def _nfkc(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def normalise_number(text: Any) -> str | None:
    """A number as a digit string: NFKC, "#"/"No." dropped, leading zeros dropped unless all
    zeros; an Excel float's ".0" is dropped ("10.0" -> "10"). None for anything that is not
    one number of 1..MAX_DIGITS digits."""
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, float):
        if not text.is_integer() or text < 0:
            return None
        text = str(int(text))
    m = _NUMBER.fullmatch(_nfkc(text))
    if not m or len(m.group(1)) > MAX_DIGITS:
        return None
    digits = m.group(1)
    return digits.lstrip("0") or digits


def _norm_position(text: Any) -> str:
    return _nfkc(text).upper()


def _norm_name(text: Any) -> str:
    return " ".join(_nfkc(text).split()).casefold()


def _int_value(text: Any) -> int | None:
    s = _nfkc(text) if text is not None else ""
    m = _INT.fullmatch(s)
    return int(m.group(1)) if m else None


def _attr(entry, role: str) -> str:
    value = (getattr(entry, "attrs", None) or {}).get(role)
    return "" if value is None else str(value).strip()


def _usable_numbers(reads: Mapping[str, Any], min_frames_agree: int) -> dict[str, str]:
    """{label: number} for the engine reads with enough agreeing frames. A read that is not a
    dict with "frames_agree" is no read; such reads are logged once per call."""
    out, bad = {}, []
    for label, read in reads.items():
        if read is None:
            continue
        if not isinstance(read, Mapping) or "frames_agree" not in read:
            bad.append(label)
            continue
        agree = read.get("frames_agree")
        if isinstance(agree, bool) or not isinstance(agree, (int, float)) \
                or agree < min_frames_agree:
            continue
        n = normalise_number(read.get("number"))
        if n is not None:
            out[label] = n
    if bad:
        log.warning("number reads without frames_agree ignored for %d label(s): %s",
                    len(bad), ", ".join(map(str, bad[:10])))
    return out


# ---------------------------------------------------------------- candidate builders

def number_candidates(reads: Mapping[str, Any], entries, *,
                      min_frames_agree: int = MIN_FRAMES_AGREE) -> list[Candidate]:
    """Exact normalised number match; every roster row with that number is a candidate.
    reads: the engine's {label: {"number", "votes", "frames_agree"}}."""
    by_number: dict[str, list[str]] = {}
    for e in entries:
        n = normalise_number(_attr(e, "number"))
        if n is not None:
            by_number.setdefault(n, []).append(e.participant_id)
    out = []
    for label, n in _usable_numbers(reads, min_frames_agree).items():
        for row in by_number.get(n, []):
            out.append(Candidate(label, row, "number", CUE_SCORES["number"], n))
    return out


def position_candidates(pos: Mapping[str, str], entries, slots: Mapping[str, int] | None = None,
                        claims: Mapping[str, int] | None = None) -> list[Candidate]:
    """NFKC-upper equality; shared positions are weighted down (see the module docstring).
    slots/claims default to the counts from the roster and from `pos`. `pos` should hold only
    the own team's labels when a team answer exists."""
    rows: dict[str, list[str]] = {}
    for e in entries:
        p = _norm_position(_attr(e, "position"))
        if p:
            rows.setdefault(p, []).append(e.participant_id)
    n_slots = Counter({p: len(r) for p, r in rows.items()})
    if slots is not None:
        n_slots = Counter()
        for p, n in slots.items():
            n_slots[_norm_position(p)] += int(n)
    n_claims = Counter(_norm_position(p) for p in pos.values() if p)
    if claims is not None:
        n_claims = Counter()
        for p, n in claims.items():
            n_claims[_norm_position(p)] += int(n)
    base = CUE_SCORES["position"]
    out = []
    for label, p in pos.items():
        key = _norm_position(p) if p else ""
        targets = rows.get(key, [])
        if not targets:
            continue
        s, c = n_slots.get(key, len(targets)), n_claims.get(key, 1)
        score = base if s == 1 else base * min(1.0, s / max(c, 1)) * SHARED_FACTOR
        out.extend(Candidate(label, row, "position", score) for row in targets)
    return out


def order_candidates(order: Sequence[str], entries, role: str,
                     anchors: Mapping[str, str] | None = None) -> list[Candidate]:
    """grid: roster value k <-> order[k-1]. batting: relative order, aligned by at least
    MIN_ANCHORS agreeing anchors (see the module docstring). `order` should hold only the own
    team's labels when a team answer exists."""
    if role == "grid":
        out = []
        for e in entries:
            k = _int_value(_attr(e, "grid"))
            if k is not None and 1 <= k <= len(order):
                out.append(Candidate(order[k - 1], e.participant_id, "grid",
                                     CUE_SCORES["grid"]))
        return out
    if role != "batting":
        raise ValueError(f"unknown order role {role!r}")
    return _batting_candidates(order, entries, anchors or {})


def _batting_candidates(order: Sequence[str], entries,
                        anchors: Mapping[str, str]) -> list[Candidate]:
    ranked = [(k, i, e.participant_id) for i, e in enumerate(entries)
              if (k := _int_value(_attr(e, "batting"))) is not None]
    lineup = [row for _, _, row in sorted(ranked)]
    n = len(lineup)
    if not n:
        return []
    seen: list[str] = []
    for label in order:
        if label not in seen:
            seen.append(label)
    seen = seen[:n]
    index = {row: j for j, row in enumerate(lineup)}
    # anchors in observed order: (observed position, offset into the lineup)
    anch = [(i, (index[anchors[lab]] - i) % n) for i, lab in enumerate(seen)
            if anchors.get(lab) in index]
    top = Counter(d for _, d in anch).most_common(2)
    if not top or top[0][1] < MIN_ANCHORS or (len(top) > 1 and top[1][1] == top[0][1]):
        return []
    d0 = top[0][0]
    # concordance: under the agreed rotation the anchors' lineup ranks must increase with the
    # observed order (missed batters only push later offsets up)
    rotated = [(i + d - d0) % n for i, d in anch]
    if any(b <= a for a, b in zip(rotated, rotated[1:])):
        return []
    anchored = {i for i, _ in anch}
    score = CUE_SCORES["batting"]
    out = []
    for i, label in enumerate(seen):
        if i in anchored:
            continue
        before = [d for j, d in anch if j < i]
        after = [d for j, d in anch if j > i]
        if (before and before[-1] != d0) or (after and after[0] != d0):
            continue
        out.append(Candidate(label, lineup[(i + d0) % n], "batting", score))
    return out


def mapped_candidates(names: Mapping[str, str], entries, cue: str) -> list[Candidate]:
    """{label: roster row id or name} -> candidates; a name must match exactly one row.
    Names match the roster's raw name (attrs["name"]), never display_name, which may be a
    default (「選手3」, 「#10」) or disambiguated (「山田（#10）」); rows without a name take
    no part in name matching."""
    ids = {e.participant_id for e in entries}
    by_name: dict[str, list[str]] = {}
    for e in entries:
        raw = _attr(e, "name")
        if raw:
            by_name.setdefault(_norm_name(raw), []).append(e.participant_id)
    score = CUE_SCORES[cue]
    out = []
    for label, name in names.items():
        if name is None:
            continue
        if name in ids:
            out.append(Candidate(label, name, cue, score))
            continue
        rows = by_name.get(_norm_name(name), [])
        if len(rows) == 1:
            out.append(Candidate(label, rows[0], cue, score))
    return out


# ---------------------------------------------------------------- teams

def match_team_names(reads: Mapping[str, Any], entries, teams: Mapping[str, str], *,
                     min_frames_agree: int = MIN_FRAMES_AGREE) -> dict[str, str]:
    """{cluster: roster team name} by a majority of the number matches ("other" is never
    named). A tie names nothing; when both clusters lead with the same name, the one with
    more votes keeps it. Empty without a team column."""
    team_of = {e.participant_id: _nfkc(_attr(e, "team")) for e in entries}
    if not any(team_of.values()):
        return {}
    votes: dict[str, Counter] = {}
    for c in number_candidates(reads, entries, min_frames_agree=min_frames_agree):
        cluster, name = teams.get(c.label), team_of.get(c.row)
        if cluster in (None, "other") or not name:
            continue
        votes.setdefault(cluster, Counter())[name] += 1
    best: dict[str, tuple[str, int]] = {}
    for cluster, cnt in votes.items():
        top = cnt.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1]:
            best[cluster] = top[0]
    out: dict[str, str] = {}
    for cluster, (name, n) in best.items():
        rivals = [m for c, (nm, m) in best.items() if c != cluster and nm == name]
        if all(n > m for m in rivals):
            out[cluster] = name
    return out


def restrict_to_team(cands: Iterable[Candidate], teams: Mapping[str, str],
                     own: str | Collection[str] | None, team_names: Mapping[str, str],
                     entries) -> list[Candidate]:
    """Keep the own team's people (own: a set of labels, or a cluster name of `teams`) and,
    when the own team's roster name is known, only rows on that team. own=None keeps all."""
    cands = list(cands)
    if own is None:
        return cands
    labels = {l for l, t in teams.items() if t == own} if isinstance(own, str) else set(own)
    names = {team_names[teams[l]] for l in labels if teams.get(l) in team_names}
    team_of = {e.participant_id: _nfkc(_attr(e, "team")) for e in entries}
    names = {_nfkc(n) for n in names}
    return [c for c in cands
            if c.label in labels and (not names or team_of.get(c.row) in names)]


# ---------------------------------------------------------------- assignment

def hungarian(score: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-total one-to-one assignment of a rectangular score matrix, pure Python O(n^3)
    (shortest augmenting paths with potentials). Returns min(n, m) (row, column) pairs."""
    n = len(score)
    m = len(score[0]) if n else 0
    if not n or not m:
        return []
    if n > m:
        t = [[score[i][j] for i in range(n)] for j in range(m)]
        return sorted((i, j) for j, i in hungarian(t))
    cost = [[-x for x in r] for r in score]
    inf = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j])


def fuse(cands: Iterable[Candidate], labels: Sequence[str], rows: Sequence[str], *,
         threshold: float = THRESHOLD, margin: float = MARGIN,
         target: str = "person") -> dict[str, Proposal]:
    """{label: Proposal} for the pairs that clear the threshold and the margin."""
    li = {l: i for i, l in enumerate(labels)}
    ri = {r: j for j, r in enumerate(rows)}
    best: dict[tuple[int, int, str], Candidate] = {}
    for c in cands:
        if c.label in li and c.row in ri:
            key = (li[c.label], ri[c.row], c.cue)
            if key not in best or c.score > best[key].score:
                best[key] = c
    if not best:
        return {}
    s = [[0.0] * len(rows) for _ in labels]
    for (i, j, _), c in best.items():
        s[i][j] += c.score
    cue_rank = {cue: k for k, cue in enumerate(CUE_SCORES)}
    out: dict[str, Proposal] = {}
    for i, j in hungarian(s):
        total = s[i][j]
        if total <= 0:
            continue
        runner = max([s[i][k] for k in range(len(rows)) if k != j]
                     + [s[k][j] for k in range(len(labels)) if k != i] + [0.0])
        if total < threshold - _EPS or total - runner < margin - _EPS:
            continue
        mine = sorted((c for (a, b, _), c in best.items() if a == i and b == j),
                      key=lambda c: (-c.score, cue_rank.get(c.cue, len(cue_rank))))
        reasons = tuple(dict.fromkeys(reason_text(c.cue, c.detail, target) for c in mine))
        out[labels[i]] = Proposal(rows[j], total, reasons)
    return out


def batting_anchors(cands: Iterable[Candidate], labels: Sequence[str], rows: Sequence[str], *,
                    threshold: float = THRESHOLD, margin: float = MARGIN) -> dict[str, str]:
    """{label: row} the other cues already propose: the anchors for the batting order."""
    rest = [c for c in cands if c.cue != "batting"]
    return {l: p.row for l, p in fuse(rest, labels, rows, threshold=threshold,
                                      margin=margin).items()}
