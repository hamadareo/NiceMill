"""成長の記録: what one person's record says, from the recorded data only.

`for_person(repo, person_id)` / `for_subject(repo, subject_id)` return a Record: the person's
works in date order (services.growth links them), each with
- the work's title, date and activity;
- whether its text is approved now (review.subject_status); only an approved text feeds the
  counts, the chart and the quotes;
- the good moments of that approved text per evaluation point (the profile categories the
  handout's chart uses; export.sheets.person_moments);
- the sentences of the approved text (quotes).

`statements(record)` are the plain lines the screen and the handout page show: how many works,
the point recorded most, a point whose count in the newest work is above the first, a point
first recorded in the newest work, and always that the counts depend on the video. With fewer
than two approved works it says so and nothing else. No line ever claims more than the
numbers (no 「上手になった」).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from pfb_app.domain import profiles, vocab
from pfb_app.export import labels, sheets
from pfb_app.export.labels import Category
from pfb_app.services import growth
from pfb_app.services.review import approved_automatically
from pfb_app.storage.db import Repo

NO_WORK = "承認した文章のある作業はまだありません。"
ONE_WORK = "承認した文章のある作業は1件だけです。変化は、2件目の記録ができてからわかります。"
NOTE = "場面の数は、動画の長さや映り方でも変わります。"
NOT_APPROVED = "承認した文章がまだありません（数えていません）。"
GRID_WORKS = 8          # the chart's newest works (the list below it has every work)
MAX_RISES = 3
_ENDS = "。！？!?"
_OPEN, _CLOSE = "「『（(【", "」』）)】"
_MARKS = "①②③④⑤⑥⑦⑧⑨⑩"   # two works on one day: 9/1①, 9/1②


@dataclass
class Entry:
    subject_id: int
    assignment_id: int
    title: str
    group_label: str | None
    when: str
    stamp: str                      # the work's created_at (the chart's column label)
    order: tuple
    activity: str
    name: str                       # the person's name in this work
    how: str | None                 # "manual" | "suggested" | None (not linked)
    approved: bool = False
    auto: bool = False              # approved by the worker, not checked by a person
    counts: dict[str, int] = field(default_factory=dict)   # point label -> good moments
    moments: int = 0                # every good moment (also those without a point)
    quotes: list[str] = field(default_factory=list)
    known: frozenset[str] = frozenset()   # the points the work's activity has at all


@dataclass
class Record:
    person_id: int | None
    name: str
    entries: list[Entry]
    points: list[Category]          # every point of the approved entries, in a fixed order

    @property
    def approved(self) -> list[Entry]:
        return [e for e in self.entries if e.approved]

    @property
    def works(self) -> list[Entry]:
        """The approved entries one per work (two groups of one work are added up): what is
        counted and compared."""
        out: dict[int, Entry] = {}
        for e in self.approved:
            if e.assignment_id not in out:
                out[e.assignment_id] = Entry(**{**e.__dict__, "counts": dict(e.counts),
                                                "quotes": list(e.quotes)})
                continue
            w = out[e.assignment_id]
            for label, n in e.counts.items():
                w.counts[label] = w.counts.get(label, 0) + n
            w.moments += e.moments
            w.quotes += e.quotes
            w.known = w.known | e.known
        return list(out.values())


def activity_of(work: dict) -> str:
    key = profiles.of_work(work)
    text = profiles.label(key)
    sport = work.get("sport")
    if key == profiles.SPORTS and sport in vocab.SPORT_LABELS:
        text += f"（{vocab.SPORT_LABELS[sport]}）"
    return text


def sentences(text: str | None) -> list[str]:
    """The sentences of a text; a sentence end inside brackets (「やった！」) does not end it."""
    out, now, depth = [], "", 0
    for ch in " ".join((text or "").split()):
        now += ch
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(depth - 1, 0)
        elif ch in _ENDS and depth == 0:
            out.append(now.strip())
            now = ""
    if now.strip():
        out.append(now.strip())
    return [x for x in out if x]


def _entry(repo: Repo, member: growth.Member, how: str | None,
           points: dict[str, tuple[int, Category]]) -> Entry:
    work = repo.get_assignment(member.assignment_id)
    entry = Entry(member.subject_id, member.assignment_id, member.work_title, member.group_label,
                  member.when, str(work.get("created_at") or ""), member.order,
                  activity_of(work), member.name, how,
                  known=frozenset(c.label for c in labels.categories_for(work)))
    moments = sheets.person_moments(repo, member.subject_id)
    if moments is None:
        return entry
    subject = repo.get_subject(member.subject_id)
    run = repo.get_run(subject["approved_run_id"])
    entry.approved = True
    entry.auto = approved_automatically(repo, subject)
    entry.quotes = sentences((run.output or {}).get("pfb"))
    entry.moments = len(moments)
    counted = Counter(m.category.key for m in moments if m.category is not None)
    for rank, cat in enumerate(labels.categories_for(work)):     # the profile's fixed order
        if counted[cat.key]:
            entry.counts[cat.label] = counted[cat.key]
            known = points.get(cat.label)
            # a point keeps the place of its first work and the look of its newest work
            points[cat.label] = (min(known[0], rank) if known else rank, cat)
    return entry


def _record(repo: Repo, person_id: int | None, name: str, chosen: list[growth.Member],
            how: dict[int, str]) -> Record:
    points: dict[str, tuple[int, Category]] = {}
    entries = [_entry(repo, m, how.get(m.subject_id), points) for m in chosen]
    ranked = sorted(points.items(), key=lambda kv: (kv[1][0], kv[0]))
    return Record(person_id, name, entries, [cat for _, (_, cat) in ranked])


def _until(repo: Repo, up_to: int | None):
    if up_to is None:
        return lambda m: True
    work = repo.get_assignment(up_to)
    bound = (str(work.get("created_at") or ""), up_to)
    return lambda m: m.order[:2] <= bound


def for_person(repo: Repo, person_id: int, *, up_to: int | None = None) -> Record | None:
    """The record's works (`up_to`: only works up to and including that work's date)."""
    person = repo.growth_person(person_id)
    if person is None:
        return None
    links = {r["subject_id"]: r["how"] for r in repo.growth_links(person_id)}
    keep = _until(repo, up_to)
    chosen = [m for m in growth.members(repo, list(links)) if keep(m)]
    return _record(repo, person_id, person["name"], chosen, links)


def for_subject(repo: Repo, subject_id: int) -> Record:
    """The record of the person's record, or the person alone (one work) when unlinked."""
    person = repo.growth_person_of(subject_id)
    if person is not None:
        found = for_person(repo, person)
        if found is not None:
            return found
    member = next(m for m in growth.members(repo, [subject_id]))
    return _record(repo, None, member.name, [member], {})


def _top(rec: Record, done: list[Entry]) -> str | None:
    totals = {p.label: sum(e.counts.get(p.label, 0) for e in done) for p in rec.points}
    best = max(totals.values(), default=0)
    if not best:
        return None
    tops = [p.label for p in rec.points if totals[p.label] == best]
    if len(tops) == 1:
        works = sum(1 for e in done if e.counts.get(tops[0]))
        return f"いちばん多く記録された観点は「{tops[0]}」です（作業{works}件で合計{best}回）。"
    names = "と".join(f"「{t}」" for t in tops) if len(tops) == 2 else \
        "".join(f"「{t}」" for t in tops)
    each = "どちらも" if len(tops) == 2 else "どれも"
    return f"いちばん多く記録された観点は{names}です（{each}合計{best}回）。"


def statements(rec: Record) -> list[str]:
    """Counted per work (two groups of one work are one work). A point the activity of a work
    does not have is not a 0 there: nothing is compared with it."""
    done = rec.works
    if not done:
        return [NO_WORK]
    if len(done) == 1:
        return [ONE_WORK]
    first, last = done[0], done[-1]
    lines = [f"作業{len(done)}件の記録があります（{first.when}〜{last.when}）。"]
    top = _top(rec, done)
    if top:
        lines.append(top)
    rises, new = [], []
    for p in rec.points:
        b = last.counts.get(p.label, 0)
        before = [e for e in done[:-1] if p.label in e.known]
        if b and before and not any(e.counts.get(p.label) for e in before):
            new.append(p.label)
        elif p.label in first.known and p.label in last.known and b > first.counts.get(p.label, 0):
            rises.append(f"「{p.label}」のよかった場面は、最初の作業（{first.when}）では"
                         f"{first.counts.get(p.label, 0)}回、いちばん新しい作業（{last.when}）では{b}回でした。")
    lines += rises[:MAX_RISES]
    lines += [f"いちばん新しい作業で、はじめて「{label}」の場面が記録されました。" for label in new]
    lines.append(NOTE)
    return lines


def _column(stamp: str, with_year: bool) -> str:
    try:
        d = datetime.fromisoformat(stamp).astimezone()
    except (TypeError, ValueError):
        return "?"
    return f"{d.year}/{d.month}/{d.day}" if with_year else f"{d.month}/{d.day}"


def grid(rec: Record, last: int = GRID_WORKS
         ) -> tuple[list[str], list[tuple[Category, list[int | None]]]]:
    """(column labels: the newest `last` works by date, two on one day told apart by ①②;
    rows: each point and its count in each of those works, None where the work's activity
    does not have the point)."""
    shown = rec.works[-last:]
    years = {e.stamp[:4] for e in shown}
    columns = [_column(e.stamp, len(years) > 1) for e in shown]
    same = Counter(columns)
    seen: Counter = Counter()
    for i, c in enumerate(columns):
        if same[c] > 1:
            columns[i] = c + _MARKS[seen[c] % len(_MARKS)]
            seen[c] += 1
    rows = [(p, [e.counts.get(p.label, 0) if p.label in e.known else None for e in shown])
            for p in rec.points]
    return columns, rows


# --- the handout page 「成長のまとめ」 ----------------------------------------------------------

def handout_records(repo: Repo, assignment_id: int, subject_ids) -> dict[int, Record]:
    """{subject id: record up to this work} of the people whose record has two or more
    approved works up to and including this work (the others get no page)."""
    out = {}
    for sid in subject_ids:
        person = repo.growth_person_of(sid)
        rec = for_person(repo, person, up_to=assignment_id) if person is not None else None
        if rec is not None and len(rec.works) >= 2:
            out[sid] = rec
    return out


def handout_count(repo: Repo, assignment_id: int, subject_ids=None) -> int:
    """How many of the work's approved people would get the page: one query (step 6 shows it
    on every run). `subject_ids`: the people step 6 lists as in the PDF (their staleness is
    checked there). The earlier works' approvals are read as stored, so this is an upper
    bound: an earlier text an update outdated still counts here; the PDF leaves it out."""
    return repo.growth_handout_count(assignment_id, subject_ids)
