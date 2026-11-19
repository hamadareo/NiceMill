"""Readings for the youngest reader's text: a kanji the reader has not learned is written in hiragana.

The 8B model is poor at this on its own: told to use only the allowed kanji it keeps 試合 and 始
(0 of 33 texts passed the kanji check), and asked for readings it writes 味方 as あいがた and
守備 as しゅうび (about one in twenty words wrong). A reading the machine cannot verify is a
fact change in a child's text, so the conversion is a table: a word in it is written in
hiragana, a word that is not stays in kanji and the check (`plain_checks.problems`) refuses the
text, which then falls back to the original. Add a word here when a fallback names it.

Keys: a kanji word, or a kanji word with the hiragana that follows it where the reading depends on
it (試み, 行っ). A one-kanji key (no hiragana) applies only to a whole kanji run (進んだ, never
前進 or 動画), so a word of the table never changes the reading of a longer word.
"""
import re
import unicodedata
from collections.abc import Collection

# word -> reading; the value may keep a kanji the reader knows (分ごろ)
READINGS: dict[str, str] = {
    # the play and the game
    "試合": "しあい", "開始": "かいし", "直接的": "ちょくせつてき",
    "直接": "ちょくせつ", "攻撃": "こうげき", "守備": "しゅび", "得点": "とくてん",
    "味方": "みかた", "方向": "ほうこう", "可能性": "かのうせい", "可能": "かのう",
    "相手": "あいて", "参加者": "さんかしゃ", "参加": "さんか", "仲間": "なかま",
    "手伝": "てつだ", "展開": "てんかい", "貢献": "こうけん", "創出": "そうしゅつ",
    "準備": "じゅんび", "継続": "けいぞく", "選手": "せんしゅ", "具体的": "ぐたいてき",
    "具体": "ぐたい", "行動": "こうどう", "移動": "いどう", "制限": "せいげん",
    "実行": "じっこう", "様子": "ようす", "動画": "どうが", "連携": "れんけい",
    "促進": "そくしん", "機会": "きかい", "支援": "しえん", "維持": "いじ", "進行": "しんこう",
    "意図": "いと", "位置取り": "いちどり", "位置": "いち", "前進": "ぜんしん", "阻止": "そし",
    "確保": "かくほ", "状態": "じょうたい", "協力": "きょうりょく", "役立": "やくだ",
    "手放": "てばな", "分頃": "分ごろ", "守備行動": "しゅびこうどう",
    # general words of the other kinds of work
    "学習": "がくしゅう", "勉強": "べんきょう", "発表": "はっぴょう", "会議": "かいぎ",
    "説明": "せつめい", "質問": "しつもん", "意見": "いけん", "提案": "ていあん",
    "資料": "しりょう", "作業": "さぎょう", "練習": "れんしゅう", "活動": "かつどう",
    "発言": "はつげん", "理解": "りかい", "工夫": "くふう", "挑戦": "ちょうせん",
    "努力": "どりょく", "集中": "しゅうちゅう", "相談": "そうだん", "発見": "はっけん",
    "結果": "けっか", "目標": "もくひょう", "完成": "かんせい",
    "紹介": "しょうかい", "実験": "じっけん", "観察": "かんさつ", "記録": "きろく",
    "整理": "せいり", "感想": "かんそう", "全員": "ぜんいん",
    # a word whose reading depends on what follows it
    "経っ": "たっ", "経つ": "たつ", "経過": "けいか", "場所": "ばしょ", "場面": "ばめん",
    "全体": "ぜんたい", "試み": "こころみ", "試す": "ためす", "試し": "ためし",
    # one kanji (the stem of a verb or an adjective)
    "始": "はじ", "動": "うご", "進": "すす", "向": "む", "取": "と", "続": "つづ", "持": "も",
    "守": "まも", "受": "う", "狙": "ねら", "他": "ほか", "別": "べつ", "助": "たす", "防": "ふせ",
    "移": "うつ", "運": "はこ", "次": "つぎ", "決": "き", "整": "ととの", "投": "な", "密": "みつ",
    "突": "つ", "握": "にぎ", "挙": "あ", "映": "うつ", "流": "なが", "示": "しめ", "返": "かえ",
    "送": "おく", "戻": "もど", "離": "はな", "加": "くわ", "増": "ふ", "再": "ふたた",
    "役": "やく", "頃": "ごろ", "渡": "わた",
}

_KANJI = re.compile(r"[一-鿿々]")
_ORDER = sorted(READINGS, key=len, reverse=True)


def _is_kanji(c: str) -> bool:
    return _KANJI.match(c) is not None


def _match(text: str, i: int) -> tuple[str, str] | None:
    """The longest table word at `text[i:]` that fits (a one-kanji word only as a whole kanji
    run), as (the text it covers, its reading)."""
    for key in _ORDER:
        if not text.startswith(key, i):
            continue
        if len(key) == 1 and _is_kanji(key):
            before = text[i - 1] if i else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if _is_kanji(before) or _is_kanji(after):
                continue
        return key, READINGS[key]
    return None


def hiragana(text: str, allowed: Collection[str], names: Collection[str] = ()) -> str:
    """`text` with every table word that holds a kanji outside `allowed` written in hiragana
    (names are left as written). Other words are left as they are."""
    text = text or ""
    keep = sorted({n for n in (unicodedata.normalize("NFKC", n).strip() for n in names)
                   if n and _is_kanji(n[0])}, key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(text):
        name = next((n for n in keep if text.startswith(n, i)), None)
        if name:
            out.append(name)
            i += len(name)
            continue
        c = text[i]
        if _is_kanji(c):
            found = _match(text, i)
            if found and any(_is_kanji(k) and k not in allowed for k in found[0]):
                out.append(found[1])
                i += len(found[0])
                continue
        out.append(c)
        i += 1
    return "".join(out)
