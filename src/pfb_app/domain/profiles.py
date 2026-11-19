"""Activity-type profiles: what kind of activity a work is, and what that changes.

A work has one profile (assignments.profile), chosen from the content check's scene verdict
(`for_scene`) and changeable on the card (「内容を変える」):

    一般 / 話し合い・会議 / スポーツ / 発表・パフォーマンス / 研修・ワークショップ / 学習活動

A profile sets
1. the words on the screens and in the PDFs (vocab.WORDS under the profile's key: the people,
   the evaluator, the activity and the moment word);
2. the evaluation focus: the categories the contribution stage tags its judgments with
   (rubric_criteria) and the feedback emphasises, and an English focus line added to the
   tracking engine's activity description (`engine_activity`, the engine's existing generic
   `activity` hook) so the behavior stage looks for those actions;
3. the contribution and feedback prompts (prompts/video_<key>/, tone and length included).

一般 and スポーツ are exactly the behaviour before profiles (their prompts, the engine's
activity line and the model input are unchanged); their categories are the engine's action
types in plain words (`category_of` maps an action type). Every profile but スポーツ is the
"school" domain (the DB value of general works), so the domain logic (limits, roster roles,
methods) never changes with the profile.

`words(profile)` and `categories(profile)` are what the PDFs use (export.labels on the PDF
branch delegates to them).
"""
from __future__ import annotations

from dataclasses import dataclass

from pfb_app.domain import vocab

GENERAL, MEETING, SPORTS = "general", "meeting", "sports"
PRESENTATION, WORKSHOP, LEARNING = "presentation", "workshop", "learning"
SCHOOL = vocab.GENERAL   # the DB domain value of every profile but sports


@dataclass(frozen=True)
class Profile:
    key: str
    label: str                       # the card's choice and 「場面」 fact
    domain: str                      # "school" | "sports"
    categories: tuple[str, ...]      # plain category names, in chart order
    title_word: str                  # the default title 「{title_word} 10月23日」
    activity_text: str               # the default 「…の内容」 line
    focus: str | None = None         # English, appended to the engine's activity line


OTHER_CATEGORY = "その他"
# The categories of 一般 and スポーツ: the engine's action types (pfb3x3 domain.
# GENERIC_ACTION_TYPES / SPORTS_ACTION_TYPES) grouped under plain words, in chart order (the
# one table: export.labels on the PDF branch delegates to it). Only chart and PDF labels; the
# model input of 一般 and スポーツ never names them.
GENERAL_GROUPS = (
    ("伝える・話し合う", ("speak_or_discuss", "point_or_gesture")),
    ("手伝う・支える", ("hand_over", "hold_or_support")),
    ("作る・組み立てる", ("build_or_assemble",)),
    ("直す・整える", ("adjust_or_fix",)),
    ("道具を使う", ("manipulate_object",)),
    ("よく見る・聞く", ("observe",)),
    ("動く", ("move",)),
    (OTHER_CATEGORY, ("other",)),
)
SPORTS_GROUPS = (
    ("パス", ("pass", "assist")),
    ("シュート・得点", ("shot_or_score",)),
    ("守り", ("defense",)),
    ("位置どり", ("positioning", "off_ball")),
    ("声かけ", ("communication",)),
    ("切りかえ", ("transition",)),
    ("がんばり・取り返し", ("effort_recovery",)),
    (OTHER_CATEGORY, ("other",)),
)
GENERAL_ACTIONS = {t: name for name, types in GENERAL_GROUPS for t in types}
SPORTS_ACTIONS = {t: name for name, types in SPORTS_GROUPS for t in types}
_ACTIONS = {GENERAL: GENERAL_ACTIONS, SPORTS: SPORTS_ACTIONS}


def _names(groups) -> tuple[str, ...]:
    return tuple(name for name, _ in groups)


PROFILES: dict[str, Profile] = {p.key: p for p in (
    Profile(GENERAL, "一般", SCHOOL, _names(GENERAL_GROUPS), "グループ活動",
            "グループ活動"),
    Profile(MEETING, "話し合い・会議", SCHOOL,
            ("発言", "傾聴", "まとめ", "提案", "合意づくり", "時間管理"), "会議", "話し合い・会議",
            "Kind of activity: a meeting or discussion. Pay particular attention to visible "
            "actions such as speaking to the group, listening while turned toward the speaker, "
            "writing or summarising notes on paper or a board, showing or pointing at "
            "materials, and checking a clock or an agenda."),
    Profile(SPORTS, "スポーツ", "sports", _names(SPORTS_GROUPS), "スポーツの試合",
            "スポーツの試合"),
    Profile(PRESENTATION, "発表・パフォーマンス", SCHOOL,
            ("構成", "声と視線", "資料の使い方", "質疑応答", "聞き手への配慮"), "発表",
            "発表・パフォーマンス",
            "Kind of activity: a presentation or performance in front of listeners. Pay "
            "particular attention to visible actions such as presenting or performing, facing "
            "and looking at the listeners, gesturing, pointing at or operating slides or "
            "materials, and answering questions from the listeners."),
    Profile(WORKSHOP, "研修・ワークショップ", SCHOOL, ("取り組み", "質問", "協力", "実践"), "研修",
            "研修・ワークショップ",
            "Kind of activity: a training session or workshop. Pay particular attention to "
            "visible actions such as trying out or practising what is shown, asking a question "
            "(raising a hand, turning to the instructor), helping others or handing them "
            "materials, and working with the materials."),
    Profile(LEARNING, "学習活動", SCHOOL, ("考えを伝える", "聞く", "協力", "続けて取り組む"), "学習活動",
            "学習活動",
            "Kind of activity: a learning activity. Pay particular attention to visible actions "
            "such as telling others an idea, listening while turned toward the speaker, working "
            "together with others, writing or working on the material, and keeping at a task."),
)}
# the profiles whose model input and prompts are the ones from before profiles
UNCHANGED = frozenset({GENERAL, SPORTS})
# the engine's `scene --activity` keys (pfb3x3 sports.ACTIVITIES) -> profile
ACTIVITY_PROFILES = {"meeting": MEETING, "presentation": PRESENTATION, "workshop": WORKSHOP,
                     "learning": LEARNING}


def get(profile: str | None) -> Profile:
    """The profile; unknown or None is 一般."""
    return PROFILES.get(profile or GENERAL, PROFILES[GENERAL])


def words(profile: str | None) -> vocab.Words:
    """The profile's words (one table: vocab.WORDS)."""
    return vocab.words(get(profile).key)


def categories(profile: str | None) -> tuple[str, ...]:
    """The profile's evaluation categories in plain words, in chart order."""
    return get(profile).categories


def label(profile: str | None) -> str:
    return get(profile).label


def title_word(profile: str | None) -> str:
    return get(profile).title_word


def activity_text(profile: str | None) -> str:
    return get(profile).activity_text


def domain_of(profile: str | None) -> str:
    return get(profile).domain


def default_for_domain(domain: str | None) -> str:
    """The profile of a work from before profiles: sports -> スポーツ, else 一般."""
    return SPORTS if domain == "sports" else GENERAL


def of_work(work: dict) -> str:
    """The work's profile key: its stored profile when it belongs to the work's domain, else
    the domain's default (works from before profiles, or a profile that does not fit)."""
    domain = work.get("domain") or SCHOOL
    stored = work.get("profile")
    if stored in PROFILES and PROFILES[stored].domain == domain:
        return stored
    return default_for_domain(domain)


def check(profile: str | None, domain: str) -> str:
    """The profile to store for a work of `domain`: None is the domain's default; ValueError
    for an unknown profile or one of the other domain."""
    if profile is None:
        return default_for_domain(domain)
    if profile not in PROFILES or PROFILES[profile].domain != domain:
        raise ValueError("どんな場面の動画かを選んでください。")
    return profile


def for_scene(scene: str | None, activity: str | None = None) -> str | None:
    """The profile the content check suggests: a sport -> スポーツ; a meeting, presentation,
    workshop or learning activity -> that profile; any other scene -> 一般; no scene -> None
    (the coach picks)."""
    if scene is None:
        return None
    if scene == "sport":
        return SPORTS
    return ACTIVITY_PROFILES.get(activity or "", GENERAL)


def engine_activity(work: dict) -> str | None:
    """The activity description given to the tracking engine (its registry `activity`): the
    work's activity line, plus the profile's English focus line for the new profiles. 一般 and
    スポーツ give the line alone, as before (None when empty)."""
    text = (work.get("task_text") or "").strip()
    focus = get(of_work(work)).focus
    if focus:
        return f"{text}\n{focus}" if text else focus
    return text or None


def category_of(profile: str | None, criteria=(), action_type: str | None = None) -> str | None:
    """The category of one good moment: the first of the judgment's rubric_criteria that is
    one of the profile's categories; else (一般, スポーツ) the group of the engine action type,
    その他 for a type no group names; else None (no type, or a profile whose moments are
    categorised by the judgment only)."""
    cats = categories(profile)
    for c in criteria or ():
        if c in cats:
            return c
    table = _ACTIONS.get(get(profile).key)
    kind = (action_type or "").strip().lower()
    if table is None or not kind:
        return None
    return table.get(kind, OTHER_CATEGORY)


__all__ = ["ACTIVITY_PROFILES", "GENERAL", "GENERAL_ACTIONS", "GENERAL_GROUPS", "LEARNING",
           "MEETING", "OTHER_CATEGORY", "PRESENTATION", "SPORTS_GROUPS", "PROFILES", "Profile", "SPORTS", "SPORTS_ACTIONS", "UNCHANGED",
           "WORKSHOP", "activity_text", "categories", "category_of", "check",
           "default_for_domain", "domain_of", "engine_activity", "for_scene", "get", "label",
           "of_work", "title_word", "words"]
