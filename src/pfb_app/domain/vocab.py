"""The one vocabulary table for the UI and the PDFs: general work vs sports, and the words of
each activity-type profile (domain/profiles.py).

The general vocabulary is stored under the domain value "school" (kept for compatibility with
existing databases); on screen it is the general wording for any field (活動・参加者・評価者),
never school words. A key of this table is a domain value ("school", "sports") or a profile key
("general", "meeting", "presentation", "workshop", "learning"); unknown keys get the general
words. Default person names stay 「参加者n」 for every profile that is not sports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Words:
    work: str
    people: str
    giver: str
    roster: str
    time_anchor: str
    rubric_label: str
    person_prefix: str
    moment: str = "場面"   # what one good moment is called (「よかった場面」)


GENERAL = "school"   # the DB value of the general (non-sports) domain; shown as general words

WORDS: dict[str, Words] = {
    GENERAL: Words("活動", "参加者", "評価者", "名簿", "開始から", "評価の観点（任意）", "参加者"),
    "sports": Words("試合・練習", "選手", "コーチ", "選手名簿", "試合時間",
                    "見てほしい観点（任意）", "選手"),
    # the activity-type profiles that are not sports (domain "school")
    "meeting": Words("会議", "参加者", "進行役", "名簿", "開始から", "評価の観点（任意）", "参加者"),
    "presentation": Words("発表", "発表者", "聞き手・評価者", "名簿", "開始から",
                          "評価の観点（任意）", "参加者"),
    "workshop": Words("研修", "受講者", "講師", "名簿", "開始から", "評価の観点（任意）", "参加者"),
    "learning": Words("活動", "参加者", "指導者", "名簿", "開始から", "評価の観点（任意）", "参加者"),
}
WORDS["general"] = WORDS[GENERAL]   # the general profile's key
_AS_IS = (GENERAL, "general")       # keys whose screen text is the general text unchanged


def words(domain: str | None) -> Words:
    return WORDS.get(domain or GENERAL, WORDS[GENERAL])


# Phrases that hold a base word in another sense and are never rewritten (「グループ活動」 is a
# kind of scene, not the work).
_KEEP = ("グループ活動",)
# A default person name (「参加者3」, every profile but sports) is a name, not a word: an error
# text that quotes one keeps it as the person is called.
_NAME = re.compile(r"参加者(?=[0-9０-９])")


def localize(text: str, domain: str | None) -> str:
    """Screen text written in general words, in the words of `domain` (a domain value or a
    profile key). General text is returned unchanged (byte-identical)."""
    if not text or (domain or GENERAL) in _AS_IS:
        return text
    w, base = words(domain), WORDS[GENERAL]
    kept = {f"\x00{i}\x00": phrase for i, phrase in enumerate(_KEEP)}
    name_mark = "\x00n\x00"
    text = _NAME.sub(name_mark, text)
    for mark, phrase in kept.items():
        text = text.replace(phrase, mark)
    for old, new in ((base.roster, w.roster), (base.work, w.work), (base.people, w.people),
                     (base.giver, w.giver)):
        text = text.replace(old, new)
    for mark, phrase in kept.items():
        text = text.replace(mark, phrase)
    return text.replace(name_mark, "参加者")


def default_person_name(domain: str | None, n: int) -> str:
    return f"{words(domain).person_prefix}{n}"


def opponent_name(n: int) -> str:
    return f"相手選手{n}"


def numberless_name(n: int) -> str:
    return f"選手（番号なし）{n}"


def number_name(number) -> str:
    return f"#{number}"


# Literal copy of the engine's list: pfb3x3/sports.py SPORT_KEYS (same order).
SPORT_KEYS = ("basketball", "soccer", "baseball_softball", "volleyball", "handball", "rugby",
              "tennis_badminton", "track", "swimming", "motor_racing", "other")

SPORT_LABELS = {
    "basketball": "バスケットボール", "soccer": "サッカー", "baseball_softball": "野球・ソフトボール",
    "volleyball": "バレーボール", "handball": "ハンドボール", "rugby": "ラグビー",
    "tennis_badminton": "テニス・バドミントン", "track": "陸上", "swimming": "水泳",
    "motor_racing": "モーターレース", "other": "その他のスポーツ",
}

SCENE_LABELS = {"group_work": "グループ活動", "sport": "スポーツ", "other": "その他"}


def mark_name(label: str) -> str:
    """'P7' -> '7番の人'."""
    return f"{label.lstrip('Pp')}番の人"
