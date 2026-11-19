"""Pure-function traceability and wording checks."""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from pfb_app.domain.models import (BehaviorEventDraft, Judgment, PfbOutput, TeamTipsOutput,
                                   TipsOutput)

MIN_QUOTE_CHARS = 10

COLOR_WORDS = ["赤", "青", "緑", "黄", "白", "黒", "紫", "橙", "オレンジ", "ピンク", "茶", "灰", "水色", "色"]
# Common words that contain a colour character but are not colour descriptions.
_COLOR_FALSE_POSITIVES = [
    "面白", "空白", "余白", "独白", "告白", "明白", "潔白", "白紙", "白板",
    "黒板", "赤ちゃん", "赤字", "黒字", "色々", "色んな", "お茶", "喫茶",
    "白熱", "白星", "黒星",
]
_CONFIDENCE_RE = re.compile(r"confidence|確信度|\d+(?:\.\d+)?\s*%", re.IGNORECASE)
ADVICE_WORDS = ["次回", "べき", "もっと", "改善", "ましょう"]
# a place in the game's time (NFKC text): wrong for a clip whose game clock is not known
TIME_PHRASE_RE = re.compile(
    r"試合開始|開始直後|試合終了|終了間際|試合の(?:始め|終わり)|試合時間\s*\d+"
    r"|序盤|中盤|終盤|前半|後半")
# 「N分ごろ」 as written (with its lead-in), and a start phrase: checked against the cited events
MINUTE_RE = re.compile(r"(試合時間|動画の約|動画の|約)?(\d+)分ごろ")
START_RE = re.compile(r"試合開始直後|開始直後")
EVIDENCE_LIMIT = 3     # plays a feedback text stands on (the prompt asks for two or three)
# Tracker labels P1, P12 (also p3: a model may lower-case them), token-bounded; NFKC text.
P_LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])P\d+(?![A-Za-z0-9_])", re.IGNORECASE)


def normalize_text(s: str) -> str:
    """NFKC-normalize and remove all whitespace and line breaks."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


def verify_quote(quote: str, page_text: str) -> bool:
    q = normalize_text(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return False
    return q in normalize_text(page_text)


def judgment_alignment_errors(ref_ids: list[str], judgments: list[Judgment]) -> list[str]:
    counts = Counter(j.ref_id for j in judgments)
    expected = set(ref_ids)
    errors = []
    for r in ref_ids:
        if r not in counts:
            errors.append(f"判定が欠落しています: {r}")
    for r, n in counts.items():
        if r not in expected:
            errors.append(f"余分な判定があります: {r}")
        elif n > 1:
            errors.append(f"判定が重複しています: {r}")
    return errors


def unknown_participant_errors(events: list[BehaviorEventDraft], allowed: set[str]) -> list[str]:
    errors = []
    for i, e in enumerate(events):
        if e.participant not in allowed:
            errors.append(f"未知の参加者IDです: {e.participant}（{i + 1}件目, {e.video_time}）")
    return errors


def _token_found(token: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text) is not None


def _p_labels(text: str) -> list[str]:
    found = {t.upper() for t in P_LABEL_RE.findall(unicodedata.normalize("NFKC", text))}
    return sorted(found, key=lambda t: int(t[1:]))


def pfb_text_warnings(text: str, internal_ids: set[str], camera_names: set[str],
                      tracked: bool = False, clock_known: bool = True) -> list[str]:
    """One warning per category found in the PFB text. `tracked` (tracking engine): also
    flag the tracker's person labels (P1, P2, ...). `clock_known` False (a sports work whose
    game clock the coach did not give): also flag a place in the game's time (試合開始直後,
    試合時間N分ごろ, 序盤, 後半, ...), which the text cannot know."""
    warnings = []
    ids = sorted(t for t in internal_ids if t and _token_found(t, text))
    if ids:
        warnings.append(f"内部IDが含まれています: {', '.join(ids)}")
    cams = sorted(t for t in camera_names if t and _token_found(t, text))
    if cams:
        warnings.append(f"カメラ名が含まれています: {', '.join(cams)}")
    labels = _p_labels(text) if tracked else []
    if labels:
        warnings.append(f"人物の番号が含まれています: {', '.join(labels)}")

    scrubbed = text
    for w in _COLOR_FALSE_POSITIVES:
        scrubbed = scrubbed.replace(w, "")
    colors = [c for c in COLOR_WORDS if c in scrubbed]
    if colors:
        warnings.append(f"色の語が含まれています: {', '.join(colors)}")

    norm = unicodedata.normalize("NFKC", text)
    conf = sorted({m.group(0) for m in _CONFIDENCE_RE.finditer(norm)})
    if conf:
        warnings.append(f"確信度の表現が含まれています: {', '.join(conf)}")

    when = [] if clock_known else sorted({m.group(0) for m in TIME_PHRASE_RE.finditer(norm)})
    if when:
        warnings.append("試合時間が不明の映像なのに、時間を表す言葉が含まれています: "
                        + ", ".join(when))

    advice = [w for w in ADVICE_WORDS if w in text]
    if advice:
        warnings.append(f"改善指導の表現が含まれています: {', '.join(advice)}")
    return warnings


def time_errors(text: str, minutes: set[int], clock_known: bool) -> list[str]:
    """What is wrong with the time words of a sports text, for the model to fix: every
    「N分ごろ」 must be one of `minutes` (the minutes of the events the text cites); 「開始直後」
    only with a known game clock and a 0-minute event; without a known clock a minute is the
    video's (「動画の約N分ごろ」) and 0 (under a minute) is not written."""
    norm = unicodedata.normalize("NFKC", text)
    have = "、".join(f"{m}分ごろ" for m in sorted(minutes)) or "なし"
    errors = []
    for m in MINUTE_RE.finditer(norm):
        n = int(m.group(2))
        if not clock_known and n == 0:
            errors.append("1分に満たない場面には、時間を書かないでください（「この場面で、」などで始める）")
        elif n not in minutes:
            errors.append(f"「{n}分ごろ」は、根拠にしたイベントにありません。書けるのは、根拠にした"
                          f"イベントの「N分ごろ」の数（{have}）だけです")
    if START_RE.search(norm):
        if not clock_known:
            errors.append("試合時間が分からない映像なので、「開始直後」とは書かないでください")
        elif 0 not in minutes:
            errors.append("「開始直後」は、根拠に0分ごろのイベントがあるときだけ書けます")
    return list(dict.fromkeys(errors))


def normalize_time_words(text: str, clock_known: bool) -> str:
    """Every 「N分ごろ」 is written one way: 「試合時間N分ごろ」 with a known game clock, else
    「動画の約N分ごろ」 (a minute of the video, not of the game)."""
    # matched and replaced on the text as written (\d and int() read full-width digits): the
    # rest of the sentence — 「！」「（）」「…」「～」 — is never touched
    lead = "試合時間" if clock_known else "動画の約"
    return MINUTE_RE.sub(lambda m: f"{lead}{int(m.group(2))}分ごろ", text)


def trim_refs(refs: list[str], text: str, minute_of: dict[str, int | None],
              limit: int = EVIDENCE_LIMIT) -> list[str]:
    """The evidence a text stands on: at most `limit` of `refs` (no repeats, the model's
    order), keeping one event for each minute the text names (and a 0-minute one for a
    「開始直後」) before the rest, so the list never grows past the few plays written about."""
    refs = list(dict.fromkeys(refs))
    if len(refs) <= limit:
        return refs
    norm = unicodedata.normalize("NFKC", text)
    named = {int(m.group(2)) for m in MINUTE_RE.finditer(norm)}
    if START_RE.search(norm):
        named.add(0)
    keep: list[str] = []
    covered: set[int] = set()
    for r in refs:
        minute = minute_of.get(r)
        if minute in named and minute not in covered and len(keep) < limit:
            keep.append(r)
            covered.add(minute)
    for r in refs:
        if len(keep) >= limit:
            break
        if r not in keep:
            keep.append(r)
    return [r for r in refs if r in keep]


def evidence_ref_errors(pfb: PfbOutput, true_ids: set[str]) -> list[str]:
    errors = [f"存在しない根拠IDです: {r}" for r in pfb.evidence_refs if r not in true_ids]
    if pfb.pfb_status == "generated" and not pfb.evidence_refs:
        errors.append("PFBが生成されていますが、根拠IDがありません")
    return errors


# --- practice tips (次にためしてみること) ---------------------------------------------------------

MAX_TIP_CHARS = 70          # one tip is a short line
MAX_PERSON_TIPS = 3         # a person's tips: one to three
MAX_TEAM_TIPS = 3           # the group's tips: one to three
MIN_TEAM_SUPPORT = 2        # a tip for the group stands on at least this many people's texts
# a tip ends the way a suggestion does: 「〜してみる」「〜を意識してみる」「〜してみよう」
TIP_END_RE = re.compile(r"(?:み(?:る|よう)|みても(?:よい|いい))$")
# counts and measures nobody observed (digits are never written; kanji numerals with a counter)
TIP_NUMBER_RE = re.compile(r"\d|[一二三四五六七八九十百千]+\s*(?:回|本|点|割|倍|秒|％|%)")
# words that grade a person: a tip suggests, it does not judge
TIP_GRADE_WORDS = ("上手", "下手", "へた", "うまい", "うまく", "巧み", "素晴らしい", "すばらしい",
                   "素敵", "完璧", "最高", "優秀", "苦手", "得意", "才能", "センス", "天才",
                   "ミス", "失敗", "弱点", "ダメ", "だめ", "悪い", "足りない", "不足", "問題",
                   "できていな", "できなかった")
# an order or a correction rather than a suggestion
TIP_ORDER_WORDS = ("なければ", "ないといけ", "ないとなら", "必ず", "絶対", "直す", "直して", "反省", "しろ", "せよ")
_TIP_TIME_RE = re.compile(r"動画の|分ごろ|秒")
# a person called by a name (「〇〇さん」); 「みなさん」「皆さん」 call everybody
_HONORIFIC_RE = re.compile(r"(?<![みなさ皆])(?:さん|くん|ちゃん)")
_RUN_RE = re.compile(r"[一-龥々]{2,}|[ァ-ヴー]{2,}")
_KATAKANA_RUN_RE = re.compile(r"[ァ-ヴー]{3,}")
# words too general to show that a tip stands on a scene (their pairs of letters are not used)
_GENERAL_RUNS = ("選手", "動き", "場面", "自分", "意識", "試合", "練習", "チーム", "周り", "仲間",
                 "味方", "相手", "全体", "様子", "気持ち", "状況", "時間", "理由", "目的", "方法",
                 "工夫", "動作", "行動", "今回", "次回", "全員", "一人", "二人", "それぞれ",
                 "ほか", "確認", "続け", "取り入れ", "大切", "大事", "場合", "周囲", "声かけ")
_GENERAL_PAIRS = {run[i:i + 2] for run in _GENERAL_RUNS for i in range(len(run) - 1)}
# katakana words that name no play, so no scene is needed to write them
_GENERAL_KATAKANA = ("チーム", "コート", "ボール", "プレー", "プレイ", "ゴール", "ポジション",
                     "スペース", "タイミング", "ライン", "メンバー", "コミュニケーション",
                     "サポート", "ベンチ")
# what a suggestion is made of besides the scene: its ending and the words that ask for it
_TIP_FILLER_RE = re.compile(r"ためして|してみ|みよう|みる|みても|意識|続け|ほかの|いつも|さらに")


@dataclass(frozen=True)
class TipContext:
    """What a tip's wording is checked against: the names it must not carry (people's display
    names), the ids and camera names of the work, and whether tracker labels count (tracking
    engine)."""
    names: frozenset[str] = frozenset()
    internal_ids: frozenset[str] = frozenset()
    cameras: frozenset[str] = frozenset()
    tracked: bool = False


def split_tips(text: str) -> list[str]:
    """The lines of a tips box as tips: one tip a line, blank lines and a leading bullet or
    number mark dropped, repeats kept once."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*・●○■□▪▶▷◆◇]|\d+[.)．）]|[①-⑩])\s*", "", line).strip()
        if line and line not in out:
            out.append(line)
    return out


def _tip_norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def tip_wording_problems(text: str, ctx: TipContext = TipContext()) -> list[str]:
    """What is wrong with one tip's wording, one plain line per kind of fault (also what the
    coach reads under a tip they wrote)."""
    norm = _tip_norm(text)
    flat = re.sub(r"\s+", "", norm)
    problems = []
    if len(flat) > MAX_TIP_CHARS:
        problems.append(f"{MAX_TIP_CHARS}文字より長い文があります。短く書き直してください")
    if not TIP_END_RE.search(flat.rstrip("。.！!")):
        problems.append("文の終わりを「〜してみる」「〜を意識してみる」の形にしてください")
    if TIP_NUMBER_RE.search(norm):
        problems.append("数字や回数が入っています。数えていないことは書かないでください")
    if _TIP_TIME_RE.search(norm) or TIME_PHRASE_RE.search(norm) or MINUTE_RE.search(norm):
        problems.append("時間を表す言葉が入っています。いつの場面かは書かないでください")
    grade = [w for w in TIP_GRADE_WORDS if w in norm]
    if grade:
        problems.append("人を評価する言葉が入っています（" + "、".join(grade) + "）")
    order = [w for w in (*TIP_ORDER_WORDS, *ADVICE_WORDS) if w in norm]
    if order:
        problems.append("指示や直す言い方が入っています（" + "、".join(dict.fromkeys(order))
                        + "）。「〜してみる」と、すすめる形にしてください")
    ids = sorted(t for t in ctx.internal_ids if t and _token_found(t, norm))
    if ids or (ctx.tracked and _p_labels(norm)):
        problems.append("記号や番号のような呼び方が入っています")
    if sorted(t for t in ctx.cameras if t and _token_found(t, norm)):
        problems.append("カメラの名前が入っています")
    named = [n for n in ctx.names if n and len(n) >= 2 and n in norm]
    if named or _HONORIFIC_RE.search(norm):
        problems.append("人の名前が入っています。名前は書かないでください")
    scrubbed = norm
    for w in _COLOR_FALSE_POSITIVES:
        scrubbed = scrubbed.replace(w, "")
    if any(c in scrubbed for c in COLOR_WORDS):
        problems.append("色の言葉が入っています")
    if _CONFIDENCE_RE.search(norm):
        problems.append("確かさの度合いが入っています")
    return problems


def _tip_units(tip: str) -> set[str]:
    """The pieces of a tip that can show it stands on a scene: pairs of letters inside its
    kanji / katakana words and runs of three letters holding one, without the general words
    and the suggestion's own filler."""
    body = _TIP_FILLER_RE.sub("、", re.sub(r"[\s。、，,.！!？?「」『』（）()・]", "、", tip))
    units: set[str] = set()
    for run in _RUN_RE.findall(body):
        if run not in _GENERAL_RUNS:
            units |= {run[i:i + 2] for i in range(len(run) - 1)}
    for part in body.split("、"):
        for i in range(len(part) - 2):
            gram = part[i:i + 3]
            if _RUN_RE.search(gram) or re.search(r"[一-龥々ァ-ヴー]", gram):
                units.add(gram)
    return {u for u in units if u not in _GENERAL_PAIRS
            and not any(pair in u for pair in _GENERAL_PAIRS)}


def tip_grounding_problems(text: str, source: str) -> list[str]:
    """Whether a tip stands on `source` (the cited events and the person's feedback): a
    katakana word (a play, a piece of equipment) must be written there, and some word of the
    tip must be one the source has (`_tip_units`)."""
    tip, base = _tip_norm(text), _tip_norm(source)
    problems = []
    for word in _KATAKANA_RUN_RE.findall(tip):
        if word not in base and word not in _GENERAL_KATAKANA:
            problems.append(f"「{word}」は、根拠にした場面にありません。場面に書かれていないことは書かないでください")
    if not any(u in base for u in _tip_units(tip)):
        problems.append("根拠にした場面と結びつくことばがありません。場面に書かれた動きにふれて書いてください")
    return list(dict.fromkeys(problems))


def person_tips_errors(out: TipsOutput, allowed_refs: set[str], sources: dict[str, str],
                       feedback: str, ctx: TipContext) -> list[str]:
    """What is wrong with the model's tips for one person, for it to fix: one to three tips,
    each citing events of the person's feedback, each well worded and standing on them."""
    errors = []
    if out.tips_status != "generated":
        return errors
    if len(out.tips) > MAX_PERSON_TIPS:
        errors.append(f"ヒントは{MAX_PERSON_TIPS}個までです")
    seen = set()
    for n, tip in enumerate(out.tips, 1):
        head = f"{n}個目: "
        refs = list(dict.fromkeys(tip.evidence_refs))
        if not refs:
            errors.append(head + "根拠にした場面のIDがありません")
        errors += [head + f"存在しない根拠IDです: {r}" for r in refs if r not in allowed_refs]
        key = normalize_text(tip.text)
        if key in seen:
            errors.append(head + "同じヒントが重なっています")
        seen.add(key)
        errors += [head + p for p in tip_wording_problems(tip.text, ctx)]
        cited = "\n".join(sources.get(r, "") for r in refs if r in allowed_refs)
        errors += [head + p for p in tip_grounding_problems(tip.text, cited + "\n" + feedback)]
    return errors


def trim_team_tips(out: TeamTipsOutput, sources: dict[int, str]) -> int:
    """Drop, in place, the group's tips that the people's texts cannot carry, keep the rest, and
    return how many were dropped. The model pads its answer to three tips: with a real one
    or two it adds tips that name a single person (so does it even with a prompt that tells it
    not to), and asked again after the refusal it gave up and returned none. A tip that names
    fewer than MIN_TEAM_SUPPORT existing people is dropped, and so is every tip after the
    MAX_TEAM_TIPS-th. Nothing is dropped when no tip would remain: `team_tips_errors` then
    says why. The tips that stay are checked as before."""
    if out.tips_status != "generated":
        return 0
    shared = [t for t in out.tips
              if len({p for p in t.people if p in sources}) >= MIN_TEAM_SUPPORT]
    if not shared:
        return 0
    shared = shared[:MAX_TEAM_TIPS]
    dropped = len(out.tips) - len(shared)
    out.tips = shared
    return dropped


def team_tips_errors(out: TeamTipsOutput, sources: dict[int, str], ctx: TipContext) -> list[str]:
    """What is wrong with the model's tips for the group: one to three tips, each standing on
    at least MIN_TEAM_SUPPORT people's texts (`sources`: {number: that person's text})."""
    errors = []
    if out.tips_status != "generated":
        return errors
    if len(out.tips) > MAX_TEAM_TIPS:
        errors.append(f"ヒントは{MAX_TEAM_TIPS}個までです")
    seen = set()
    for n, tip in enumerate(out.tips, 1):
        head = f"{n}個目: "
        people = list(dict.fromkeys(tip.people))
        errors += [head + f"存在しない人の番号です: {p}" for p in people if p not in sources]
        if len([p for p in people if p in sources]) < MIN_TEAM_SUPPORT:
            errors.append(head + f"チームのヒントは、{MIN_TEAM_SUPPORT}人以上の文章に共通することから"
                          "作ってください")
        key = normalize_text(tip.text)
        if key in seen:
            errors.append(head + "同じヒントが重なっています")
        seen.add(key)
        errors += [head + p for p in tip_wording_problems(tip.text, ctx)]
        base = "\n".join(sources.get(p, "") for p in people)
        errors += [head + p for p in tip_grounding_problems(tip.text, base)]
    return errors
