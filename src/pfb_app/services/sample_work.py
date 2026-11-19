"""「見本の作業」: a read-only sample work for first-time users.

It is built only from fictional material (sample_work.json: invented players, made-up texts
and scene layouts that ui/wizard/sample_art draws as illustrations). No real video, photo,
face or name is shipped with it, and nothing of it is stored in the works tables: it is never
counted on 「この解析の確かさ」, never in a PDF and never touched by a delete or a retry.

Whether it is shown is one per-install choice kept in audit_log (action `ACTION`, detail
{"shown": bool}; no entry: shown), so 「見本を消す」 is one tap and 「見本をもう一度出す」 on
「はじめての方へ」 brings it back.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pfb_app.storage.db import Repo

ACTION = "sample_work"
FIXTURE = Path(__file__).with_name("sample_work.json")
APPROVALS = ("as_is", "edited", "short")   # how each sample text was approved


@dataclass(frozen=True)
class Spot:
    id: str
    x: float          # 0..100 across the half court (left to right)
    y: float          # 0..100 from the baseline (top) towards half court (bottom)


@dataclass(frozen=True)
class Arrow:
    kind: str                         # "pass" (between two players) or "run" (a path)
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class Scene:
    spots: tuple[Spot, ...]
    focus: str | None
    arrow: Arrow | None


@dataclass(frozen=True)
class Moment:
    at: str
    did: str
    why: str
    scene: Scene


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    number: int
    team: str          # "blue" or "white"
    approval: str      # one of APPROVALS
    text: str
    draft: str | None
    moments: tuple[Moment, ...]


@dataclass(frozen=True)
class SampleWork:
    title: str
    about: str
    domain: str
    sport: str
    video_name: str
    minutes: int
    cameras: int
    check: str
    stages: tuple[str, ...]
    overview: Scene
    people: tuple[Person, ...]

    def team_of(self, person_id: str) -> str:
        return next(p.team for p in self.people if p.id == person_id)

    def number_of(self, person_id: str) -> int:
        return next(p.number for p in self.people if p.id == person_id)


def _scene(raw: dict) -> Scene:
    spots = tuple(Spot(p["id"], float(p["x"]), float(p["y"])) for p in raw["players"])
    where = {s.id: (s.x, s.y) for s in spots}
    arrow = None
    if raw.get("arrow"):
        a = raw["arrow"]
        start = where[a["from"]] if isinstance(a["from"], str) else tuple(a["from"])
        end = where[a["to"]] if isinstance(a["to"], str) else tuple(a["to"])
        arrow = Arrow(a["kind"], tuple(map(float, start)), tuple(map(float, end)))
    return Scene(spots, raw.get("focus"), arrow)


def parse(raw: dict) -> SampleWork:
    """The sample from its fixture. Refuses anything not marked synthetic."""
    if raw.get("synthetic") is not True:
        raise ValueError("the sample work must be built from synthetic material only")
    people = tuple(
        Person(p["id"], p["name"], int(p["number"]), p["team"], p["approval"], p["text"],
               p.get("draft"), tuple(Moment(m["at"], m["did"], m["why"], _scene(m["scene"]))
                                     for m in p["moments"]))
        for p in raw["people"])
    if any(p.approval not in APPROVALS for p in people):
        raise ValueError("unknown approval kind in the sample work")
    video = raw["video"]
    return SampleWork(raw["title"], raw["about"], raw["domain"], raw["sport"], video["name"],
                      int(video["minutes"]), int(video["cameras"]), raw["check"],
                      tuple(raw["stages"]), _scene(raw["overview"]), people)


@lru_cache(maxsize=1)
def load() -> SampleWork:
    return parse(json.loads(FIXTURE.read_text(encoding="utf-8")))


def is_shown(repo: Repo) -> bool:
    entry = repo.latest_audit(ACTION, None)
    return entry is None or bool(entry["detail"].get("shown"))


def set_shown(repo: Repo, shown: bool) -> None:
    """「見本を消す」 (False) or 「見本をもう一度出す」 (True)."""
    repo.audit(ACTION, None, {"shown": bool(shown)})
