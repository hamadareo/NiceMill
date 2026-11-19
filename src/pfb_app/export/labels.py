"""The words and category names of the PDFs, from one place.

`words_for(work)`, `categories_for(work)` and `category_of(work, action_type, criteria)` take
the work (its assignment row). When the activity profiles are there (pfb_app.domain.profiles,
branch feat/profiles) they answer from the work's profile (`profiles.of_work`, `.words`,
`.categories`, `.category_of`); until then from the domain mapping (sports / general) below.
Nothing else in the export reads the domain for words.

Categories are the plain words the charts use for the engine's action types
(pfb3x3/domain.py SPORTS_ACTION_TYPES and GENERIC_ACTION_TYPES) or a profile's own names. They
keep a fixed order: the order gives each category its colour slot and its marker shape, so a
category looks the same on every chart and page (colour follows the category, never its
rank), and the shape alone tells them apart in grayscale. Past seven, a category takes a hue
again but never a shape (colour + shape together; no generated ninth hue); past the nine
shapes the rest are folded into その他, which is gray.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from functools import lru_cache

from pfb_app.domain.models import SubjectStatus
from pfb_app.domain.vocab import Words, words

try:   # the activity profiles (feat/profiles); the domain mapping below until they are merged
    from pfb_app.domain import profiles as _profiles
except ImportError:   # pragma: no cover - depends on the branch
    _profiles = None

OTHER = "other"

# Categorical slots in the fixed order (dataviz reference palette, light mode); その他 is the
# neutral gray of an "Other" fold, never a ninth hue.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
OTHER_COLOUR = "#8b929c"
SHAPES = ("circle", "square", "triangle", "diamond", "triangle_down", "hexagon", "star",
          "pentagon", "cross")
OTHER_LABEL = "その他"
OTHER_SHAPE = "ring"


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    types: tuple[str, ...]     # the engine's action types counted here
    colour: str
    shape: str


def _table(rows: tuple[tuple[str, str, tuple[str, ...]], ...]) -> tuple[Category, ...]:
    out, i = [], 0
    for key, label, types in rows:
        if key == OTHER:
            out.append(Category(key, label, types, OTHER_COLOUR, OTHER_SHAPE))
            continue
        out.append(Category(key, label, types, SLOTS[i % len(SLOTS)], SHAPES[i % len(SHAPES)]))
        i += 1
    return tuple(out)


@lru_cache(maxsize=32)
def _named(names: tuple[str, ...]) -> tuple[Category, ...]:
    """A profile's category names as Categories (その他 last, whatever its place). Only as
    many named categories as there are marker shapes: the rest are folded into その他 (a
    shape is never used twice, so no category is told apart by colour alone)."""
    named = list(dict.fromkeys(n for n in names if n != OTHER_LABEL))
    rows = [(n, n, ()) for n in named[:len(SHAPES)]]
    if OTHER_LABEL in names or len(named) > len(SHAPES):
        rows.append((OTHER, OTHER_LABEL, ()))
    return _table(tuple(rows))


SPORTS_CATEGORIES = _table((
    ("pass", "パス", ("pass", "assist")),
    ("score", "シュート・得点", ("shot_or_score",)),
    ("defense", "守り", ("defense",)),
    ("position", "位置どり", ("positioning", "off_ball")),
    ("voice", "声かけ", ("communication",)),
    ("switch", "切りかえ", ("transition",)),
    ("effort", "がんばり・取り返し", ("effort_recovery",)),
    (OTHER, "その他", ("other",)),
))

GENERAL_CATEGORIES = _table((
    ("talk", "伝える・話し合う", ("speak_or_discuss", "point_or_gesture")),
    ("help", "手伝う・支える", ("hand_over", "hold_or_support")),
    ("make", "作る・組み立てる", ("build_or_assemble",)),
    ("fix", "直す・整える", ("adjust_or_fix",)),
    ("tool", "道具を使う", ("manipulate_object",)),
    ("watch", "よく見る・聞く", ("observe",)),
    ("move", "動く", ("move",)),
    (OTHER, "その他", ("other",)),
))


def _domain(work: dict | None) -> str:
    return (work or {}).get("domain") or "school"


def _profile(work: dict | None) -> str | None:
    return _profiles.of_work(dict(work or {})) if _profiles is not None else None


def words_for(work: dict | None) -> Words:
    """The work's words (participant / evaluator / activity / time anchor)."""
    if _profiles is not None:
        return _profiles.words(_profile(work))
    return words(_domain(work))


def categories_for(work: dict | None) -> tuple[Category, ...]:
    """The work's categories, in their fixed order (その他 last)."""
    if _profiles is not None:
        return _named(tuple(_profiles.categories(_profile(work))))
    return SPORTS_CATEGORIES if _domain(work) == "sports" else GENERAL_CATEGORIES


def category_of(work: dict | None, action_type: str | None, criteria=()) -> str | None:
    """The category key of one good moment. With profiles: the profile's answer from the
    judgment's 評価の観点 and the action type (None: not counted). Without: the action
    type's category, その他 for a type no category names, None when there is no type."""
    if _profiles is not None:
        name = _profiles.category_of(_profile(work), tuple(criteria or ()), action_type)
        if name is None:
            return None
        keys = {c.key for c in categories_for(work)}
        return name if name in keys and name != OTHER else OTHER     # folded -> その他
    t = (action_type or "").strip().lower()
    if not t:
        return None
    for c in categories_for(work):
        if t in c.types:
            return c.key
    return OTHER


def category(work: dict | None, key: str) -> Category | None:
    return next((c for c in categories_for(work) if c.key == key), None)


def sports(work: dict | None) -> bool:
    return _domain(work) == "sports"


# --- other words ------------------------------------------------------------------------

MODEL_WORDS = {"standard": "標準", "high": "高精度"}
_CAMERA = re.compile(r"(?<![A-Za-z0-9])cam(\d+)(?![A-Za-z0-9])")


def model_words(model_key: str | None) -> str:
    return MODEL_WORDS.get(model_key or "", "標準")


def camera_word(camera: str | None) -> str:
    """cam1 -> カメラ1 (as on the screens)."""
    return _CAMERA.sub(r"カメラ\1", camera or "")


# the evaluator PDF's status chips: always words, never a colour alone
APPROVED, AUTO, SHORT, LEFT_OUT = "承認", "自動で承認", "短い文", "入っていない"
WHY_NOT = {SubjectStatus.review: "確認待ち", SubjectStatus.insufficient: "場面が少ない",
           SubjectStatus.processing: "解析中", SubjectStatus.waiting_gpu: "解析の順番待ち",
           SubjectStatus.pending: "解析待ち", SubjectStatus.failed: "止まっています"}
SHORT_NOTE = "映っている場面が少なかったため、短い文にしています。"
RETRACK_NOTE = "人の追跡をやり直しているため、写真と図は入れていません。"


def footer_line(work: dict | None, video: bool, auto: bool = False) -> str:
    """A handout's footer; `auto`: the text was approved by the worker, not checked by a
    person, and the footer says so."""
    source = "動画" if video else "書いたもの"
    if auto:
        return f"このフィードバックは{source}をもとに自動で作り、そのまま渡しています。"
    return f"このフィードバックは{source}をもとに作り、{words_for(work).giver}が確かめたものです。"


def evaluator_footer_line(work: dict | None, video: bool, autos: list[bool]) -> str:
    """The evaluator PDF's cover footer, from the approved sheets' `auto` flags: a person's
    check only when every sheet was checked, the handouts' auto line when none was, else both
    counted (each person's page then carries that sheet's own footer_line)."""
    n_auto = sum(1 for a in autos if a)
    if n_auto == 0 or n_auto == len(autos):
        return footer_line(work, video, auto=n_auto > 0)
    source = "動画" if video else "書いたもの"
    return (f"このフィードバックは{source}をもとに作りました。{n_auto}人の分は確かめずに自動で承認し、"
            f"ほかは{words_for(work).giver}が確かめています。")


def self_words(text: str, work: dict | None) -> str:
    """An observation (written with 本人 / 別の参加者) for the person it is about: 本人 is
    「あなた」 and the others are 「ほかの<people>」."""
    from pfb_app.pipeline.pfb3x3_bridge import OTHER as OTHER_PERSON, SELF
    w = words_for(work)
    return (text or "").replace(SELF, "あなた").replace(OTHER_PERSON, f"ほかの{w.people}")


# --- file names -------------------------------------------------------------------------

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')
NAME_CHARS = 60


def handout_file_name(index: int, name: str, total: int, fallback: str = "参加者") -> str:
    """`<番号>_<名前>.pdf` of one person's handout: the number zero-padded to the count's width
    (so a file list sorts like the PDF), the name without characters file systems refuse."""
    safe = _UNSAFE.sub("_", unicodedata.normalize("NFC", name or ""))
    safe = re.sub(r"_+", "_", safe).strip(" ._")[:NAME_CHARS].strip(" ._") or fallback
    return f"{index:0{len(str(max(total, 1)))}d}_{safe}.pdf"
