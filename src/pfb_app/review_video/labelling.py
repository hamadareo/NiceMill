"""How a person is shown: by the jersey number the AI read, never guessed.

The number is the one the app itself uses to name people (services.auto_people._read_number:
the names stage's read, kept only when enough frames agree, or the number worn most of the
time on the track), so a person shown here as 「3番」 is the one the app calls 「#3」. The
precision gate (configs/cue_gate.yaml) keeps the number cue officially OFF at this time (held-out
6/7 = 0.857, n 7 < 20), and the app's naming is provisional for that reason: so every screen says
the number is the AI's reading and may be wrong.

- number known:   「3番」, with the team letter when the team is known (「Aチーム 3番」)
- not known:      「番号不明（P2）」 (the tracker's own id, small, as a help)
- the same number twice in the same team (or twice with no team known):
  the label gets the note 「番号が重複・要確認」 (same number in different teams: told apart by team)
- one person's tracks reading different numbers: unknown, with 「番号が食い違う・要確認」
"""
from __future__ import annotations

from pfb_app.review_video.model import KIND_CHANGE, TEAM_LETTERS, UNKNOWN, Person, change_note

DUPLICATE = "番号が重複・要確認"
CONFLICT = "番号が食い違う・要確認"


def label_of(number: str | None, team: str | None, internal: str, verified: bool = False) -> str:
    """「3番？」 while the number cue has not passed the precision gate (an AI reading), 「3番」
    only when it has."""
    if number is None:
        return f"{UNKNOWN}（{internal}）" if internal else UNKNOWN
    side = TEAM_LETTERS.get(team or "")
    mark = "" if verified else "？"
    return f"{side} {number}番{mark}" if side else f"{number}番{mark}"


def assign(people: list[Person], verified: bool = False) -> None:
    """Set label / label_note on each person from number, team and internal (in place), and
    order the list (team, number, id: a neutral order, no counts)."""
    seen: dict[tuple, list[Person]] = {}
    for p in people:
        p.verified = verified
        p.label = label_of(p.number, p.team, p.internal, verified)
        p.label_note = ""
        for m in p.events:           # the number-change notes follow the same gate as the labels
            if m.kind == KIND_CHANGE and m.nums:
                m.note = change_note(m.nums[0], m.nums[1], verified)
        if p.number is not None:
            seen.setdefault((p.team, p.number), []).append(p)
        if p.conflict:
            p.label_note = CONFLICT
    for group in seen.values():
        if len(group) > 1:
            for p in group:
                p.label_note = DUPLICATE
    people.sort(key=lambda p: (p.team or "~", int(p.number) if (p.number or "").isdigit() else 10 ** 6,
                               p.number or "", p.internal, p.subject_id))
