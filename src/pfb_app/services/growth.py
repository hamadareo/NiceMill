"""成長の記録: one real person across several works (the links; export.growth reads them).

A record (growth_people) links subjects of different works or groups (growth_links). Links are
only ever made by the user: a suggestion accepted (`accept`, stored as "suggested") or a person
linked by hand (`link_manual`, "manual"). Nothing is merged by itself.

Who may be suggested as the same person (`link_key`), for evaluated people only (a video's
people with a participant id who are not excluded; every person of a document work):
- a plain number label (「#5」, 「5号車」) only between works of the same sport with the same
  「チーム・グループ」 (assignments.growth_group): numbers repeat across teams;
- a number label with a note (「#5（チーム1）」: team clusters and camera numbers belong to one
  work), an opponent (相手選手) and an anonymous default label (参加者n, 選手n, n番の人 …): never;
- any other name (roster, typed by the coach, a document's file name): the same name, compared
  after width, white space and case (review.name_key).

`suggestions` offers, per key, the unlinked people to the one record that has that key, or (no
record yet) to the earliest person with it; a key two records share is left to the user. A pair
answered 「別の人」 (`decline`) is never offered again.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from pfb_app.domain.models import Kind
from pfb_app.domain import vocab
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services.review import ADDED, NAME_SOURCE, name_key
from pfb_app.storage.db import Repo

MAX_GROUP = 40
MAX_NAME = 40
# a person called by a number: 「#5」「5」「5番」「背番号5」「No.5」 (a jersey) or 「5号車」 (a car);
# compared after NFKC (「＃５」 is 「#5」, （） are ())
_NUMBER = re.compile(r"^(?:#|背番号|no\.?)?\s*(\d+)(番|号車)?$", re.IGNORECASE)
_NOTED_NUMBER = re.compile(r"^(?:#|背番号|no\.?)?\s*\d+(?:番|号車)?\s*\(.*\)$", re.IGNORECASE)
APPROVED_ONLY = "文章を承認した人をつなげます。"
SAME_WORK = "同じ作業の人どうしはつなげません。"
OTHER_RECORD = "二人はそれぞれ別の記録に入っています。つなぎ直すときは、記録の画面で外してからつないでください。"
_PREFIXES = sorted({w.person_prefix for w in vocab.WORDS.values()} | {"参加者", "選手"},
                   key=len, reverse=True)
_ANONYMOUS = tuple(re.compile("^" + re.escape(unicodedata.normalize("NFKC", base)) + r"\d+$")
                   for base in (*_PREFIXES, vocab.numberless_name(""), vocab.opponent_name(""))
                   ) + (re.compile(r"^\d+番の人$"),)
UNNAMED = "名前のない人"


@dataclass(frozen=True)
class Member:
    """One evaluated person of one group of one work."""
    subject_id: int
    submission_id: int
    assignment_id: int
    name: str
    approved: bool               # the newest feedback text is the approved one (linkable)
    work_title: str
    group_label: str | None      # the group's name when the work has several groups
    when: str                    # the work's date, 「2026年9月1日」
    order: tuple                 # (the work's created_at, work id, group id): date order
    key: str | None
    reason: str | None           # why the key matches, in words
    person_id: int | None


@dataclass(frozen=True)
class Suggestion:
    """`subject_id` may be the same person as `other_id` (in record `person_id`, or with no
    record yet when None)."""
    subject_id: int
    other_id: int
    person_id: int | None
    reason: str


@dataclass
class Accepted:
    """What one accept linked (undo takes back exactly these)."""
    person_id: int
    linked: list[int] = field(default_factory=list)


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKC", str(name or "")).strip()


def anonymous(name: str) -> bool:
    text = _norm(name)
    return not text or text == UNNAMED or any(p.match(text) for p in _ANONYMOUS)


def link_key(work: dict, submission: dict, subject: dict) -> tuple[str, str] | None:
    """(key, reason) under which the person may be suggested as someone of another work, or
    None when the person is never suggested."""
    meta = submission.get("meta") or {}
    pid = subject.get("participant_id")
    if pid and (meta.get(ADDED) or {}).get(pid) == "opp":
        return None
    raw = str(subject.get("display_name") or "").strip()
    text = _norm(raw)
    if anonymous(text) or _NOTED_NUMBER.match(text):
        return None
    number = _NUMBER.match(text)
    if number:
        group = (work.get("growth_group") or "").strip()
        if not group:
            return None
        sport = work.get("sport") or work.get("profile") or ""
        kind = "car" if number.group(2) == "号車" else "no"
        return (f"number:{sport}|{name_key(group)}|{kind}{number.group(1)}",
                f"同じ番号（{raw}・チーム・グループ「{group}」）")
    return f"name:{name_key(text)}", f"同じ名前（{raw}）"


def date_text(stamp: str | None) -> str:
    try:
        d = datetime.fromisoformat(str(stamp)).astimezone()
    except (TypeError, ValueError):
        return ""
    return f"{d.year}年{d.month}月{d.day}日"


def members(repo: Repo, subject_ids=None) -> list[Member]:
    """Every evaluated person of every work (or only `subject_ids`), in date order; one query
    (Repo.growth_member_rows)."""
    out = []
    for r in repo.growth_member_rows(subject_ids):
        if r["kind"] == Kind.video.value and not is_evaluated(r):
            continue
        work = {"growth_group": r["growth_group"], "sport": r["sport"],
                "profile": r["profile"], "domain": r["domain"]}
        found = link_key(work, {"meta": r["meta"]}, r)
        out.append(Member(
            r["subject_id"], r["submission_id"], r["assignment_id"],
            r["display_name"] or UNNAMED, r["approved"], r["title"],
            r["group_label"] if r["groups"] > 1 else None, date_text(r["created_at"]),
            (str(r["created_at"] or ""), r["assignment_id"], r["submission_id"],
             r["subject_id"]),
            found[0] if found else None, found[1] if found else None, r["person_id"]))
    return sorted(out, key=lambda m: m.order)


def suggestions(repo: Repo, *, person_id: int | None = None, subject_id: int | None = None,
                everyone: list[Member] | None = None) -> list[Suggestion]:
    """The open suggestions (all, or those that would change `person_id`'s record or involve
    `subject_id`), in date order of the person to add. Only people with an approved text,
    never two people of one work. `everyone`: members(repo) when the caller has it."""
    everyone = members(repo) if everyone is None else everyone
    declined = repo.growth_declined()
    by_key: dict[str, list[Member]] = {}
    for m in everyone:
        if m.key is not None and m.approved:
            by_key.setdefault(m.key, []).append(m)
    out: list[Suggestion] = []
    for group in by_key.values():
        records = {m.person_id for m in group if m.person_id is not None}
        if len(records) > 1:
            continue
        record = next(iter(records), None)
        inside = [m for m in group if m.person_id is not None] if record else group[:1]
        for m in group:
            if m.person_id is not None or m in inside:
                continue
            if any(m.assignment_id == o.assignment_id for o in inside):
                continue
            if any(tuple(sorted((m.subject_id, o.subject_id))) in declined for o in inside):
                continue
            other = inside[-1]              # the latest one of the record (or the earliest)
            out.append(Suggestion(m.subject_id, other.subject_id, record, other.reason or ""))
    if person_id is not None:
        out = [s for s in out if s.person_id == person_id]
    if subject_id is not None:
        out = [s for s in out if subject_id in (s.subject_id, s.other_id)]
    order = {m.subject_id: m.order for m in everyone}
    return sorted(out, key=lambda s: order[s.subject_id])


def accept(repo: Repo, subject_id: int, other_id: int) -> Accepted:
    """「同じ人としてつなぐ」: `subject_id` joins `other_id`'s record (made, named after
    `other_id`'s name, when it has none). A person already in a record is never moved."""
    with repo._tx():
        person = repo.growth_person_of(other_id)
        mine = repo.growth_person_of(subject_id)
        record = person if person is not None else mine
        inside = [r["subject_id"] for r in repo.growth_links(record)
                  if r["subject_id"] not in (subject_id, other_id)] if record is not None else []
        _check(repo, [subject_id, other_id], inside)   # an outdated suggestion is refused
        if person is not None and mine is not None:
            if person == mine:
                return Accepted(person)
            raise ValueError(OTHER_RECORD)
        if person is None and mine is not None:
            person, subject_id, other_id = mine, other_id, subject_id
        done = Accepted(person if person is not None else repo.create_growth_person(
            repo.get_subject(other_id)["display_name"] or UNNAMED))
        for sid in (other_id, subject_id):
            if repo.growth_person_of(sid) is None:
                repo.link_growth(sid, done.person_id, "suggested")
                done.linked.append(sid)
    repo.audit("growth_accept", f"person:{done.person_id}", {"linked": done.linked})
    return done


def _check(repo: Repo, subject_ids: list[int], inside: list[int] = ()) -> None:
    """ValueError unless every one of `subject_ids` has an approved text (APPROVED_ONLY), and
    none of them shares a work with another of them or with `inside` (SAME_WORK)."""
    found = {m.subject_id: m for m in members(repo, [*subject_ids, *inside])}
    if any(not (found.get(sid) and found[sid].approved) for sid in subject_ids):
        raise ValueError(APPROVED_ONLY)
    works = [found[sid].assignment_id for sid in [*subject_ids, *inside] if sid in found]
    if len(works) != len(set(works)):
        raise ValueError(SAME_WORK)


def undo(repo: Repo, done: Accepted) -> None:
    """「元に戻す」 of one accept: the people it linked are unlinked again."""
    with repo._tx():
        for sid in done.linked:
            if repo.growth_person_of(sid) == done.person_id:
                repo.unlink_growth(sid)
    repo.audit("growth_undo", f"person:{done.person_id}", {"unlinked": done.linked})


def decline(repo: Repo, subject_id: int, other_id: int) -> None:
    """「別の人」: this pair is never suggested again."""
    repo.decline_growth(subject_id, other_id)
    repo.audit("growth_decline", f"subject:{subject_id}", {"other": other_id})


def link_manual(repo: Repo, subject_id: int, person_id: int | None) -> int:
    """Link the person to a record by hand (None: a new record named after them); a person in
    another record moves. Returns the record."""
    with repo._tx():
        if person_id is not None and repo.growth_person(person_id) is None:
            person_id = None
        inside = [r["subject_id"] for r in repo.growth_links(person_id)
                  if r["subject_id"] != subject_id] if person_id is not None else []
        _check(repo, [subject_id], inside)
        if person_id is None:
            person_id = repo.create_growth_person(
                repo.get_subject(subject_id)["display_name"] or UNNAMED)
        repo.link_growth(subject_id, person_id, "manual")
    repo.audit("growth_link", f"person:{person_id}", {"subject": subject_id})
    return person_id


def unlink(repo: Repo, subject_id: int) -> None:
    """「この作業を外す」."""
    person = repo.growth_person_of(subject_id)
    repo.unlink_growth(subject_id)
    repo.audit("growth_unlink", f"person:{person}", {"subject": subject_id})


def rename(repo: Repo, person_id: int, name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("名前を入れてください。")
    if len(text) > MAX_NAME:
        raise ValueError(f"名前は{MAX_NAME}文字までにしてください。")
    repo.rename_growth_person(person_id, text)
    return text


def set_group(repo: Repo, assignment_id: int, group: str | None) -> str | None:
    text = str(group or "").strip()
    if len(text) > MAX_GROUP:
        raise ValueError(f"チーム・グループの名前は{MAX_GROUP}文字までにしてください。")
    repo.set_growth_group(assignment_id, text or None)
    return text or None
