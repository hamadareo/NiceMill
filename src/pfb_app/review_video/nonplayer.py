"""Who is probably not a player (referee, coach, bench), from the picture itself (pure OpenCV).

Used when the package is made (collect.build_package); the result goes into the package / the
player's JSON, never into the database. Nothing here removes anybody for good: a person scored
HIGH is only *set apart* ("選手以外（推定）"), the player can bring them back with one click; a
MID score is only a mark ("選手以外かも"); a low score changes nothing.

Cues (combined; one alone never sets a person apart):
  (a) shirt colour: the median torso colour (Lab) of the person's track over ~NSAMPLE frames,
      compared with the colour groups of everybody listed. The team-like groups are those with
      at least TEAM_MIN people (else the two biggest). A person far from every team-like group
      scores high on (a).
  (d) group size: a colour shared by only one (or two) listed people is not a team.
  score = 0.6 * a + 0.4 * d.   HIGH >= 0.65, MID >= 0.4.
  Only one team-like colour group found (the colours do not split into two teams): no team letters
  and no HIGH (the others may be the second team); a MID mark is all that is left.
Not used (no data in the demo submissions, so not claimed): court position (needs the court
registration) and where the person stays on the court.
"""
from __future__ import annotations

import cv2
import numpy as np

NSAMPLE = 10
MIN_PEOPLE = 5          # fewer listed people: no judgement (no team colour can be told)
LINK = 22.0             # same colour group: Lab distance below this (single link)
TEAM_MIN = 3
FAR_LO, FAR_HI = 12.0, 32.0
HIGH, MID = 0.65, 0.4


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(0.5 * d[0] ** 2 + d[1] ** 2 + d[2] ** 2))


def torso_colour(video, frames, boxes, fps: float, n: int = NSAMPLE):
    """Median torso colour (Lab, OpenCV scale shifted to L 0..100) of one track, or None."""
    rows = [(int(f), b) for f, b in zip(frames, boxes) if float(b[3]) >= 40 and float(b[2]) >= 12]
    if len(rows) < 3:
        return None
    pick = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, min(n, len(rows)))]
    cap = cv2.VideoCapture(str(video))
    cols = []
    try:
        for f, b in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(f - 1, 0))
            ok, img = cap.read()
            if not ok:
                continue
            x, y, w, h = (float(v) for v in b[:4])
            x0, x1 = int(max(x + 0.25 * w, 0)), int(min(x + 0.75 * w, img.shape[1]))
            y0, y1 = int(max(y + 0.2 * h, 0)), int(min(y + 0.55 * h, img.shape[0]))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            lab = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
            ab = lab[:, 1:] - 128
            chroma = np.hypot(ab[:, 0], ab[:, 1])
            top = ab[chroma >= np.percentile(chroma, 65)]      # the most coloured pixels: a coloured shirt shows here
            cols.append([float(np.median(lab[:, 0])) * 100 / 255, float(np.median(top[:, 0])), float(np.median(top[:, 1]))])
    finally:
        cap.release()
    return np.median(np.array(cols), axis=0) if len(cols) >= 2 else None


def groups(cols: dict) -> list[list[str]]:
    """Average-link colour groups of {key: Lab} (merging stops when the closest two groups are
    LINK apart: single link would chain a shirt colour into a team through one in-between person)."""
    gs = [[k] for k in cols]
    while len(gs) > 1:
        best = None
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                d = float(np.mean([_dist(cols[a], cols[b]) for a in gs[i] for b in gs[j]]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best[0] >= LINK:
            break
        _, i, j = best
        gs[i] = gs[i] + gs[j]
        del gs[j]
    return sorted(gs, key=len, reverse=True)


def score(cols: dict) -> dict[str, dict]:
    """{key: {"score", "a", "d", "level"}} for every key of `cols` ({} when too few people)."""
    if len(cols) < MIN_PEOPLE:
        return {}
    gs = groups(cols)
    big = [g for g in gs if len(g) >= TEAM_MIN]
    if not big:
        teams = gs[:2]
    else:
        teams = [big[0]]
        rest = [g for g in big[1:] if len(g) >= len(big[1]) - 1]    # about as big as the second biggest
        if rest:       # a tie for the second team: the less coloured kit (white / black) is the likelier team
            teams.append(min(rest, key=lambda g: float(np.hypot(*np.median(np.array([cols[k] for k in g]), axis=0)[1:]))))
    if not teams or len(teams[0]) < 2:
        return {}
    # a picture whose colours do not split into two teams (a dark kit against a dark kit, one team in
    # view, a bright hall): one colour group is not "team A", and the people outside it may simply be
    # the other team. Nobody gets a team letter and nobody is set apart (a mark at most).
    one_team = len(teams) < 2
    centres = [np.median(np.array([cols[k] for k in g]), axis=0) for g in teams]
    inteam = {k for g in teams for k in g}
    out = {}
    for g in gs:
        for k in g:
            if k in inteam:
                a = d = 0.0
            else:
                dist = min(_dist(cols[k], c) for c in centres)
                a = float(np.clip((dist - FAR_LO) / (FAR_HI - FAR_LO), 0, 1))
                d = 1.0 if len(g) == 1 else (0.6 if len(g) == 2 else 0.4)
            s = 0.6 * a + 0.4 * d
            out[k] = {"score": round(s, 2), "a": round(a, 2), "d": round(d, 2),
                      "team": ("AB"[[i for i, t in enumerate(teams) if k in t][0]] if k in inteam else None)
                      if not one_team else None,
                      "level": ("high" if (s >= HIGH and not one_team) else ("mid" if s >= MID else ""))}
    return out


def assess(people, sources) -> dict[int, dict]:
    """{subject_id: score dict} over the listed people (their tracks on the main camera)."""
    cols: dict[int, np.ndarray] = {}
    for p in people:
        for cam, labels in p.labels.items():
            src = sources.get(cam)
            if src is None or not labels:
                continue
            got = []
            for lb in labels:
                frames, boxes = src.boxes.boxes(lb)
                c = torso_colour(src.video, frames, boxes, src.fps)
                if c is not None:
                    got.append(c)
            if got:
                cols[p.subject_id] = np.median(np.array(got), axis=0)
            break
    return score(cols)
