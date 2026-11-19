"""Who takes part in a game (pure; no database).

A game video shows more people than the players: spectators, the bench, officials, people
walking by. Only the players are evaluated. The rule, per tracked person (a P-label):

- a team colour (T1 / T2 of the names stage) -> a player ("team");
- with the court geometry (image px -> court cm) and enough foot points on it:
  - at least MIN_ON_COURT of the foot points inside the court with COURT_MARGIN_CM around it,
    moving over at least MIN_SPREAD_CM -> a player ("court");
  - fewer than MAX_OFF_COURT inside -> not playing ("off_court");
  - never more than MIN_SPREAD_CM from one spot all the time (sitting or standing at the
    line) -> not playing ("static");
  - otherwise unsure: evaluated and marked to be checked (CHECK, "court");
- a court on which fewer than MIN_COURT_PLAYERS people play is not trusted (a wrong
  registration would put everyone outside): it is ignored;
- without a geometry: moving in the picture (the engine's spread_frac >= MOVING_SPREAD_FRAC)
  -> a player ("moving"); otherwise unsure (CHECK, "no_court"): nobody is dropped on motion
  alone.

Numbers behind the thresholds (work 6, IMG_0106, autocourt H): the 5 moving players had
98-100 % of their foot points on the court and spread 575-946 cm; the 9 others were
0.9-9 m outside (0-0.69 inside with a 100 cm margin, spectators 0) or, the one at the near
sideline, inside with the margin but within 132 cm of one spot for 6 minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfb_app.export.tracks import COURT_H, COURT_W, BoxIndex, Geometry, court_samples

PLAYER, NOT_PLAYING, CHECK = "player", "not_playing", "check"
COURT_MARGIN_CM = 100.0     # players step out of bounds; the bench and spectators sit further
MIN_ON_COURT = 0.5          # a clear majority of the person's time on the court
MAX_OFF_COURT = 0.2         # below this share on the court: not playing
MIN_SPREAD_CM = 200.0       # p10..p90 range of x or y: a player moves around the court
MIN_SAMPLES = 20            # foot points (one per 0.5 s): fewer is no court evidence
MOVING_SPREAD_FRAC = 0.08   # without a court: box centres' RMS spread / frame width
STEP_S = 0.5
MIN_COURT_PLAYERS = 2       # fewer people playing on the court: the court is not trusted


@dataclass(frozen=True)
class Court:
    on_court: float     # share of the foot points inside the court + COURT_MARGIN_CM
    spread_cm: float    # max(p90 - p10) of x and y of the foot points
    samples: int


def court_numbers(boxes: BoxIndex, labels: list[str], geometry: Geometry,
                  fps: float) -> dict[str, Court]:
    """{label: Court} of the labels with at least one foot point on the court plane."""
    out: dict[str, Court] = {}
    m = COURT_MARGIN_CM
    for lb in labels:
        pts = court_samples(boxes, [lb], geometry, fps, step_s=STEP_S)
        if not pts:
            continue
        xy = np.asarray([(x, y) for _, x, y in pts], dtype=float)
        inside = ((xy[:, 0] >= -m) & (xy[:, 0] <= COURT_W + m)
                  & (xy[:, 1] >= -m) & (xy[:, 1] <= COURT_H + m))
        lo, hi = np.percentile(xy, 10, axis=0), np.percentile(xy, 90, axis=0)
        out[lb] = Court(round(float(inside.mean()), 4), round(float((hi - lo).max()), 1),
                        len(xy))
    return out


def decide(court: Court | None, spread_frac: float | None,
           team: str | None) -> tuple[str, str]:
    """(PLAYER / NOT_PLAYING / CHECK, the reason) of one person (see the module doc)."""
    if team in ("T1", "T2"):
        return PLAYER, "team"
    if court is not None and court.samples >= MIN_SAMPLES:
        if court.on_court >= MIN_ON_COURT and court.spread_cm >= MIN_SPREAD_CM:
            return PLAYER, "court"
        if court.on_court < MAX_OFF_COURT:
            return NOT_PLAYING, "off_court"
        if court.spread_cm < MIN_SPREAD_CM:
            return NOT_PLAYING, "static"
        return CHECK, "court"
    if spread_frac is not None and spread_frac >= MOVING_SPREAD_FRAC:
        return PLAYER, "moving"
    return CHECK, "no_court"


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def participation(labels: list[str], *, boxes: BoxIndex | None, geometry: Geometry | None,
                  fps: float | None, quality: dict | None,
                  teams: dict | None) -> dict[str, tuple[str, str]]:
    """{label: (status, reason)} of every label (decide on its court numbers, the quality's
    spread_frac and its team)."""
    court = (court_numbers(boxes, labels, geometry, fps)
             if boxes is not None and geometry is not None and fps else {})
    quality, teams = quality or {}, teams or {}
    spread = {lb: _num((quality.get(lb) or {}).get("spread_frac")) for lb in labels}
    out = {lb: decide(court.get(lb), spread[lb], teams.get(lb)) for lb in labels}
    if court and sum(st == (PLAYER, "court") for st in out.values()) < MIN_COURT_PLAYERS:
        # nobody (or one person) plays on this court: the registration is not trusted
        out = {lb: decide(None, spread[lb], teams.get(lb)) for lb in labels}
    return out
