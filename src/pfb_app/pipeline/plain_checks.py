"""やさしい文章版: what a reworded feedback text must pass before anybody may read it.

An approved text can be reworded for a younger reader (a worker job: `pipeline.plain`; a
coach's own edit: `services.plain_text`). The reworded text may change the words, never the
facts, so every candidate goes through `problems`, a pure function that returns what is wrong
in plain Japanese (the model is shown the list to fix its answer; a coach sees it when an edit
is refused). It checks, against the approved text it was made from:

- shape: not empty, not much shorter or longer, not many more sentences, short sentences
  (a level's `max_sentence`), no talk about the rewriting itself, no brackets or readings in
  brackets;
- facts: every number of the original is there (as digits, a kanji numeral or, for a small
  number, its reading) and none is added; every name that the original has is written as it was
  (never turned into hiragana) and no other name of the work appears; a 「さん」 / 「くん」 never
  follows a word the original does not have; every loan word (katakana) of the original is kept
  and no other one appears, nor a Latin word;
- no new praise or claim: a word like 丁寧, 協力, すごい, 上手 or 最高 (or its hiragana) is allowed
  only when the original has it; colours, internal IDs, camera names, confidence, advice and a
  place in the game's time are refused exactly as for the feedback text itself
  (`checks.pfb_text_warnings`), but only what the original did not already have;
- the youngest level (`low`): kanji of the first two school years only (names excepted).

Nothing here decides whether the words are easy; that is the model's job, asked in the prompt.
These are the gates that make a wrong answer fall back to the original text.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Collection

from pfb_app.pipeline import checks

__all__ = ["GRADE1", "GRADE2", "LEVELS", "LOW_KANJI", "ORIGINAL", "ORIGINAL_LABEL", "Level",
           "fit", "kanji_for_prompt", "problems", "tidy"]

ORIGINAL = "original"
ORIGINAL_LABEL = "13歳以上向け（いまの文のまま）"

# the kanji of the first and the second year (80 and 160 characters)
GRADE1 = ("一右雨円王音下火花貝学気九休玉金空月犬見五口校左三山子四糸字耳七車手十出女小上森人水正生青夕石"
          "赤千川先早草足村大男竹中虫町天田土二日入年白八百文木本名目立力林六")
GRADE2 = ("引羽雲園遠何科夏家歌画回会海絵外角楽活間丸岩顔汽記帰弓牛魚京強教近兄形計元言原戸古午後語工公広"
          "交光考行高黄合谷国黒今才細作算止市矢姉思紙寺自時室社弱首秋週春書少場色食心新親図数西声星晴切雪船線"
          "前組走多太体台地池知茶昼長鳥朝直通弟店点電刀冬当東答頭同道読内南肉馬売買麦半番父風分聞米歩母方北毎妹"
          "万明鳴毛門夜野友用曜来里理話")
LOW_KANJI = GRADE1 + GRADE2 + "々"


@dataclass(frozen=True)
class Level:
    key: str
    label: str           # on the screen: the readers' ages and what changes
    short: str           # the ages alone
    max_sentence: int    # characters (spaces not counted) a sentence may have
    kanji: str | None    # the kanji that may be used (None: any)


LEVELS = {
    "low": Level("low", "6〜8歳向け（ひらがなが多く、短い文）", "6〜8歳向け", 40, LOW_KANJI),
    "mid": Level("mid", "9〜12歳向け（むずかしい言葉をやさしく）", "9〜12歳向け", 70, None),
}

MIN_RATIO, MAX_RATIO, MAX_EXTRA = 0.35, 2.2, 30   # length against the original's
SENTENCES_FACTOR, SENTENCES_EXTRA = 2, 2          # at most 2 x the original's + 2
MAX_LISTED = 8

# a claim or praise the analysis never recorded unless the original says it: each row is one
# idea in kanji and in hiragana
CLAIMS = (
    ("最後まで", "さいごまで"), ("仲間", "なかま"), ("一緒", "いっしょ"), ("協力", "きょうりょく"),
    ("頑張", "がんば"), ("努力", "どりょく"), ("積極", "せっきょく"), ("熱心", "ねっしん"),
    ("丁寧", "ていねい"), ("助け", "たすけ"), ("活躍", "かつやく"), ("貢献", "こうけん"),
    ("支え", "ささえ"), ("みんなで",), ("力を合わせ", "ちからをあわせ"), ("工夫", "くふう"),
    ("粘り", "ねばり"), ("諦め", "あきらめ"), ("上手", "じょうず"), ("成長", "せいちょう"),
    ("リーダー",), ("励ま", "はげま"), ("すごい", "すごく", "凄"), ("えらい", "偉い"),
    ("最高", "さいこう"), ("天才", "てんさい"), ("完璧", "かんぺき"), ("素晴らしい", "すばらしい"),
    ("さすが",), ("かっこいい", "格好いい"), ("ヒーロー",), ("一番", "いちばん"),
    ("ナンバーワン",), ("ナイス",), ("優しい", "思いやり", "おもいやり"),
)
# a claim word that a plainer word of the original licenses: 助け for 支援 or サポート, 仲間 for
# 味方 (the reword of a word the original has is not a new claim)
CLAIM_LICENCE = {
    "助け": ("支援", "サポート", "助", "手伝", "援"), "たすけ": ("支援", "サポート", "助", "手伝", "援"),
    "仲間": ("味方", "仲間", "チームメイト"), "なかま": ("味方", "仲間", "チームメイト"),
    "支え": ("支援", "サポート", "支"), "ささえ": ("支援", "サポート", "支"),
    "協力": ("連携", "連係", "協力", "共同"), "きょうりょく": ("連携", "連係", "協力", "共同"),
}
# loan words the original may have written plainly, and what licenses each: チャンス for 機会
NEW_KATAKANA_LICENCE = {"チャンス": ("機会", "可能性", "好機")}
# loan words of the original that a reader may find written plainly instead (and the plain
# words that stand for them); the faint ones (プレー, チーム) may be left out altogether
KATAKANA_OPTIONAL = frozenset({"プレー", "プレイ"})
KATAKANA_REPLACED = {
    "サポート": ("助", "たす", "手伝", "てつだ", "支", "ささ"),
    "ディフェンス": ("守", "まも", "防", "ふせ", "止", "とめ", "とま"),
    "オフェンス": ("攻", "せめ", "せま"),
}
# words that turn an action into a result or into another action (シュートを実行した -> 決めた,
# ボールを持たずに -> 手放して): allowed only when the original says them too
RESULT_WORDS = ("成功", "失敗", "決めた", "決まった", "決めて", "決め、", "手放", "奪", "勝っ", "勝ち",
                "負け", "ゴール", "入った", "入れた", "外した", "はずした", "きめた", "てばな")
# colours and advice in hiragana (checks.pfb_text_warnings reads them in kanji only)
HIRA_COLOURS = ("あかい", "あおい", "しろい", "くろい", "きいろ", "みどりいろ", "むらさき", "ちゃいろ",
                "はいいろ", "みずいろ", "だいだいいろ", "あかいろ", "あおいろ", "しろいろ", "くろいろ")
HIRA_ADVICE = ("つぎは", "こんどは", "しましょう", "してみよう", "がんばろう", "がんばって", "してね")
# talk about the rewriting itself, as an opening or a closing remark
META = ("言いかえ", "言い換え", "いいかえ", "書きかえ", "書き換え", "やさしく言", "やさしくいう",
        "かんたんに言", "かんたんにいう", "わかりやすく言", "わかりやすくいう", "つぎの文", "リライト")
# generic loan words that name no action, person or thing of the play
SAFE_KATAKANA = frozenset({"チーム", "グループ", "メンバー", "スポーツ", "ゲーム", "ルール"})

_SENTENCE_END = re.compile(r"[。！？!?\n]+")
_KANJI_RUN = re.compile(r"[一-鿿々]+")
_KATAKANA = re.compile(r"[ァ-ヴー]{2,}")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_DIGITS = re.compile(r"\d+")
_KNUM = "〇一二三四五六七八九十百"
_COUNTERS = "人回個本枚分秒点番度年日歳冊台階"
_KANJI_NUMBER = re.compile(rf"(?<![一-鿿])[{_KNUM}]+(?=[^一-鿿]|[{_COUNTERS}]|$)")
_DIGIT_VALUE = {c: i for i, c in enumerate("〇一二三四五六七八九")}
# small numbers a reader may find written as a word
READINGS = {"ひとり": 1, "ひとつ": 1, "ふたり": 2, "ふたつ": 2, "みっつ": 3, "よっつ": 4,
            "いつつ": 5, "むっつ": 6, "ななつ": 7, "やっつ": 8, "ここのつ": 9, "とお": 10}
_HONORIFIC = re.compile(r"([^\sぁ-ゖー、。!?「」()\[\],・]{1,8})(?:さん|くん|君|ちゃん)")


def norm(text: str) -> str:
    """NFKC without any space or line break (what every check reads)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def tidy(text: str) -> str:
    """The text as it is kept: trimmed, lines trimmed, no run of blank lines."""
    lines = [line.strip() for line in (text or "").strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fit(text: str, level: str, names: Collection[str] = ()) -> str:
    """`text` as the level reads it: for the youngest level every word of the readings table
    (`pipeline.plain_readings`) that holds a kanji of a later school year is written in
    hiragana. The model keeps those kanji however it is told, and reads them wrongly when asked
    for the reading, so this is done by table; a kanji the table lacks stays and `problems`
    refuses the text."""
    allowed = LEVELS[level].kanji
    if allowed is None:
        return text
    from pfb_app.pipeline import plain_readings   # noqa: PLC0415
    return plain_readings.hiragana(text, allowed, names)


def kanji_for_prompt(level: str) -> str | None:
    """The kanji the model may use at this level (None: any)."""
    return LEVELS[level].kanji


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(unicodedata.normalize("NFKC", text or "")) if s.strip()]


def _kanji_value(token: str) -> int:
    if "十" in token or "百" in token:
        total, current = 0, 0
        for c in token:
            if c == "百":
                total += (current or 1) * 100
                current = 0
            elif c == "十":
                total += (current or 1) * 10
                current = 0
            else:
                current = _DIGIT_VALUE[c]
        return total + current
    return int("".join(str(_DIGIT_VALUE[c]) for c in token))


def numbers(text: str) -> tuple[set[int], set[int]]:
    """({numbers written as digits or kanji numerals}, {small numbers written as a word})."""
    n = unicodedata.normalize("NFKC", text or "")
    exact = {int(m) for m in _DIGITS.findall(n)}
    exact |= {_kanji_value(m) for m in _KANJI_NUMBER.findall(n)}
    return exact, {v for word, v in READINGS.items() if word in n}


def has_name(name: str, text: str) -> bool:
    """`name` is in `text`; a name with Latin letters or digits (「#7」) only as a whole token."""
    if re.search(r"[A-Za-z0-9]", name):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None
    return name in text


def _names(names: Collection[str]) -> list[str]:
    out = []
    for raw in names:
        n = norm(raw)
        if n and (len(n) >= 2 or re.search(r"[A-Za-z0-9]", n)) and n not in out:
            out.append(n)
    return sorted(out, key=len, reverse=True)


def _listed(items) -> str:
    items = list(dict.fromkeys(items))
    return "、".join(items[:MAX_LISTED]) + ("など" if len(items) > MAX_LISTED else "")


def problems(original: str, rewrite: str, level: str, *, names: Collection[str] = (),
             internal_ids: Collection[str] = (), camera_names: Collection[str] = (),
             tracked: bool = False, clock_known: bool = True) -> list[str]:
    """What is wrong with `rewrite` of the approved text `original` for `level` (KeyError for
    an unknown level), in plain Japanese; [] when it may be used. `names`: the names of the
    work's people and groups (written as they are, never new ones); the rest as the feedback
    text's own checks (`checks.pfb_text_warnings`)."""
    spec = LEVELS[level]
    o, r = norm(original), norm(rewrite)
    if not r:
        return ["文章が空です"]
    found: list[str] = []
    # --- shape
    if len(r) < MIN_RATIO * len(o):
        found.append("短すぎます。元の文に書かれていることを落とさずに書いてください")
    if len(r) > MAX_RATIO * len(o) + MAX_EXTRA:
        found.append("長すぎます。元の文に書かれていることだけを書いてください")
    n_o, n_r = len(_sentences(original)) or 1, len(_sentences(rewrite))
    if n_r > SENTENCES_FACTOR * n_o + SENTENCES_EXTRA:
        found.append(f"文の数が多すぎます（元の文は{n_o}文）。元の文に出てくること以外は書かないでください")
    long = [s.replace(" ", "").replace("　", "") for s in _sentences(rewrite)
            if len(re.sub(r"\s+", "", s)) > spec.max_sentence]
    if long:
        found.append(f"1つの文が{spec.max_sentence}文字より長いところがあります。文を分けてください"
                     f"（「{long[0][:12]}…」）")
    meta = [m for m in META if m in r and m not in o]
    if meta:
        found.append("「言いかえると」のような前置きや説明は書かないでください。文章だけを書いてください")
    if re.search(r"[()]", unicodedata.normalize("NFKC", rewrite)) and not re.search(
            r"[()]", unicodedata.normalize("NFKC", original)):
        found.append("かっこ（ ）は使わないでください。読みがなもかっこでは付けません")
    # --- facts: names
    keep = [n for n in _names(names) if has_name(n, o)]
    for n in keep:
        if not has_name(n, r):
            found.append(f"元の文にある名前「{n}」を、書かれているとおりに書いてください")
    foreign = [n for n in _names(names) if n not in keep and has_name(n, r)]
    if foreign:
        found.append(f"元の文にない名前（{_listed(foreign)}）が入っています")
    nameless = r
    for n in keep:
        nameless = nameless.replace(n, "")
    new_people = [m.group(0) for m in _HONORIFIC.finditer(nameless)
                  if m.group(1) not in o and m.group(1) not in _names(names)]
    if new_people:
        found.append(f"元の文にない人の呼び方（{_listed(new_people)}）が入っています")
    # --- facts: numbers, loan words, Latin words
    exact_o, read_o = numbers(original)
    exact_r, read_r = numbers(rewrite)
    missing = sorted(v for v in exact_o if v not in exact_r and v not in read_r)
    if missing:
        found.append("元の文にある数字（" + "、".join(str(v) for v in missing) + "）が入っていません。"
                     "数字は元の文のまま書いてください")
    extra = sorted(v for v in exact_r if v not in exact_o)
    if extra:
        found.append("元の文にない数字（" + "、".join(str(v) for v in extra) + "）が入っています")
    kata_o = list(dict.fromkeys(k for k in _KATAKANA.findall(o) if len(k.replace("ー", "")) >= 2))
    for k in kata_o:
        if k in r or k in SAFE_KATAKANA or k in KATAKANA_OPTIONAL:
            continue
        if any(w in r for w in KATAKANA_REPLACED.get(k, ())):
            continue
        found.append(f"元の文にある言葉「{k}」が入っていません。そのまま使ってください")
    kata_new = [k for k in dict.fromkeys(_KATAKANA.findall(r)) if len(k.replace("ー", "")) >= 2
                and not any(k in w for w in kata_o) and k not in SAFE_KATAKANA
                and not any(w in o for w in NEW_KATAKANA_LICENCE.get(k, ()))]
    if kata_new:
        found.append(f"元の文にない言葉（{_listed(kata_new)}）が入っています")
    latin_o = {w.lower() for w in _LATIN.findall(unicodedata.normalize("NFKC", original))}
    latin = [w for w in _LATIN.findall(unicodedata.normalize("NFKC", rewrite))
             if w.lower() not in latin_o]
    if latin:
        found.append(f"元の文にない英字（{_listed(latin)}）が入っています")
    # --- no new praise or claim
    said = []
    for forms in CLAIMS:
        if any(f in r for f in forms) and not any(f in o for f in forms):
            word = next(f for f in forms if f in r)
            if not any(w in o for w in CLAIM_LICENCE.get(word, ())):
                said.append(word)
    if said:
        found.append(f"元の文にない評価や言い方（{_listed(said)}）が入っています。"
                     "元の文に書かれていないことは足さないでください")
    result = [w.rstrip("、") for w in RESULT_WORDS if w in r and w not in o]
    if result:
        found.append(f"元の文にない動作や結果（{_listed(result)}）が入っています。"
                     "元の文と同じ動作だけを書いてください")
    new_hira = [w for w in (*HIRA_COLOURS, *HIRA_ADVICE) if w in r and w not in o]
    new_warnings = [w for w in checks.pfb_text_warnings(
        rewrite, set(internal_ids), set(camera_names), tracked, clock_known)
        if w not in checks.pfb_text_warnings(original, set(internal_ids), set(camera_names),
                                             tracked, clock_known)]
    found += new_warnings
    if new_hira:
        found.append(f"色や、次への指示のような言葉（{_listed(new_hira)}）は書かないでください")
    # --- the youngest level's kanji
    if spec.kanji is not None:
        bad = [run for run in _KANJI_RUN.findall(unicodedata.normalize("NFKC", nameless))
               if any(c not in spec.kanji for c in run)]
        chars = [c for run in bad for c in run if c not in spec.kanji]
        if chars:
            found.append(f"{spec.short}では使えない漢字があります。ひらがなで書いてください: "
                         f"{_listed(bad)}（漢字「{_listed(chars)}」）")
    return found
