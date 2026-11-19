"""作業をさがす: narrow the works of the home screen by a name, a person, a sport or a day.

Pure reading: `load` reads what can be searched for every work in two statements, `matches`
decides one work. Nothing is written and nothing is shown that the home screen does not show
already (a work's name, the names of its people, its sport and kind, the day it was made).

- **Words** (`Query.text`): split at spaces (also the wide one); a work must match every word.
  A word matches when it is part of the work's name, one of its people's names (also written
  without the space inside the name), its sport (「バスケットボール」, or 「スポーツ」 for any
  sports work), its kind of activity, or its day. Letters are compared after Unicode
  normalisation (full-width and half-width forms are the same) and without regard to case.
  A person erased on request, and a person left out of the evaluation (「この人は評価しない」),
  are not matched by name.
- **Days** are matched as written in several ways: 2026-11-14, 2026/11/14, 11/14, 11月14日,
  2026年11月14日, and the month alone (11月). The day of a work is the local date it was made.
- **Sport** (`Query.sport`): a sport key of the works present (`sports_present`), or "none"
  for the works that are not about a sport.
- **Period** (`Query.period`): all, today, the last 7 days, the last 30 days (by that day).
"""
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

from pfb_app.domain import profiles, vocab
from pfb_app.storage.db import Repo

ALL, TODAY, WEEK, MONTH = "all", "today", "week", "month"
PERIODS = {ALL: "いつでも", TODAY: "今日", WEEK: "この7日", MONTH: "この30日"}
NO_SPORT = "none"
NO_SPORT_LABEL = "競技ではない作業"
ANY_SPORT_LABEL = "すべて"


@dataclass(frozen=True)
class Query:
    text: str = ""
    sport: str | None = None      # a sport key, NO_SPORT, or None (any)
    period: str = ALL

    @property
    def active(self) -> bool:
        return bool(self.text.strip()) or self.sport is not None or self.period != ALL


@dataclass(frozen=True)
class Facts:
    """What can be searched for one work."""
    assignment_id: int
    title: str
    sport: str | None             # the sport key of a sports work ("basketball"…), else None
    sport_label: str              # "" when the work is not about a sport
    kind_label: str               # its kind of activity (スポーツ, 話し合い・会議…)
    day: date | None              # the local date it was made
    names: tuple[str, ...] = field(default_factory=tuple)


def normalise(text: str) -> str:
    """NFKC, lower case, one space between words."""
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def _day(text: str | None) -> date | None:
    try:
        return datetime.fromisoformat(text).astimezone().date() if text else None
    except ValueError:
        return None


def day_forms(day: date | None) -> list[str]:
    """The ways a day can be written in a search word."""
    if day is None:
        return []
    y, m, d = day.year, day.month, day.day
    return [f"{y}-{m:02d}-{d:02d}", f"{y}/{m}/{d}", f"{y}/{m:02d}/{d:02d}", f"{m}/{d}",
            f"{m:02d}/{d:02d}", f"{m}月{d}日", f"{y}年{m}月{d}日", f"{y}年{m}月", f"{m}月",
            f"{y}-{m:02d}"]


def load(repo: Repo) -> dict[int, Facts]:
    """The searchable facts of every work, by work id (two statements)."""
    names: dict[int, list[str]] = {}
    for aid, name in repo.conn.execute(
            "SELECT sb.assignment_id, s.display_name FROM subjects s JOIN submissions sb"
            " ON s.submission_id = sb.id WHERE s.erased_at IS NULL AND COALESCE(s.excluded, 0) = 0"
            " AND TRIM(s.display_name) <> ''"
            " ORDER BY s.id"):
        names.setdefault(aid, [])
        if name not in names[aid]:
            names[aid].append(name)
    out: dict[int, Facts] = {}
    for work in repo.list_assignments():
        sport = work.get("sport") if work.get("domain") == profiles.SPORTS else None
        key = profiles.of_work(work)
        label = profiles.PROFILES[key].label if key in profiles.PROFILES else ""
        out[work["id"]] = Facts(
            work["id"], work["title"], sport or None,
            vocab.SPORT_LABELS.get(sport or "", "") if sport else "", label,
            _day(work.get("created_at")), tuple(names.get(work["id"], ())))
    return out


def sports_present(facts) -> list[tuple[str, str]]:
    """[(sport key, label)] of the sports among the works, in the label order of vocab."""
    keys = {f.sport for f in facts if f.sport}
    return [(k, vocab.SPORT_LABELS.get(k, k)) for k in vocab.SPORT_KEYS if k in keys] + [
        (k, k) for k in sorted(keys - set(vocab.SPORT_KEYS))]


def has_non_sport(facts) -> bool:
    return any(not f.sport for f in facts)


def _haystack(facts: Facts) -> str:
    parts = [facts.title, *facts.names, *(n.replace(" ", "").replace("　", "")
                                          for n in facts.names if " " in n or "　" in n),
             facts.sport_label, "スポーツ" if facts.sport else "", facts.kind_label,
             *day_forms(facts.day)]
    return "\n".join(normalise(p) for p in parts if p)


def _in_period(day: date | None, period: str, today: date) -> bool:
    if period == ALL:
        return True
    if day is None:
        return False
    age = (today - day).days
    return {TODAY: age == 0, WEEK: 0 <= age < 7, MONTH: 0 <= age < 30}.get(period, True)


def matches(facts: Facts, query: Query, today: date | None = None) -> bool:
    """True when the work passes the sport, the period and every word of the query."""
    today = today or date.today()
    if query.sport == NO_SPORT and facts.sport:
        return False
    if query.sport not in (None, NO_SPORT) and facts.sport != query.sport:
        return False
    if not _in_period(facts.day, query.period, today):
        return False
    words = normalise(query.text).split()
    if not words:
        return True
    haystack = _haystack(facts)
    return all(word in haystack for word in words)


def filter_ids(facts: dict[int, Facts], query: Query, today: date | None = None) -> set[int]:
    """The ids of the works that pass the query."""
    return {aid for aid, f in facts.items() if matches(f, query, today)}
