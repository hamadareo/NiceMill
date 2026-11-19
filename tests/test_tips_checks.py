"""次にためしてみること: the pure wording and grounding checks (pipeline.checks)."""
import pytest

from pfb_app.domain.models import TeamTipsOutput, TipsOutput
from pfb_app.pipeline import checks
from pfb_app.pipeline.checks import TipContext

GOOD = "ドリブルのあと、周りの動きを見てパスを出してみる"
SOURCE = "行動: ドリブルでボールを運び、味方にパスを出す\n貢献の解釈: 攻めがつながった"


def problems(text, **kw):
    return checks.tip_wording_problems(text, TipContext(**kw))


def test_a_good_tip_has_no_fault():
    assert problems(GOOD) == []
    assert problems("声をかけ合う動きを続けてみよう") == []
    assert problems("ボールを持たないときの動きを意識してみる。") == []


@pytest.mark.parametrize("text, part", [
    ("ドリブルを続ける", "終わりを"),
    ("ドリブルをもっと続けてみる", "指示や直す"),
    ("ドリブルを続けるべきだ", "終わりを"),
    ("ドリブルを3回続けてみる", "数字"),
    ("ドリブルを２回続けてみる", "数字"),
    ("ドリブルを二回続けてみる", "数字"),
    ("序盤のドリブルを続けてみる", "時間"),
    ("5分ごろのドリブルを続けてみる", "数字"),
    ("上手なドリブルを続けてみる", "評価する"),
    ("得意なドリブルを続けてみる", "評価する"),
    ("ミスを減らしてみる", "評価する"),
    ("赤い服の人と動いてみる", "色"),
    ("自信度80%で動いてみる", "確かさ"),
    ("P12と声をかけてみる", "記号"),
])
def test_the_faults_are_found_in_plain_words(text, part):
    found = problems(text, tracked=True)
    assert any(part in p for p in found), found


def test_names_ids_and_cameras_are_not_written():
    ctx = dict(names=frozenset({"山田太郎"}), internal_ids=frozenset({"Pa", "cam1-0001"}),
               cameras=frozenset({"cam1"}))
    assert any("名前" in p for p in problems("山田太郎さんに声をかけてみる", **ctx))
    assert any("記号" in p for p in problems("Pa に声をかけてみる", **ctx))
    assert any("記号" in p for p in problems("cam1-0001 の動きを続けてみる", **ctx))
    assert any("カメラ" in p for p in problems("cam1 の動きを続けてみる", **ctx))
    assert problems("声をかけてみる", **ctx) == []   # a name part inside a longer word is fine
    # a made-up name is called by its ending
    assert any("名前" in p for p in problems("ケンタくんに声をかけてみる"))
    assert any("名前" in p for p in problems("田中さんと動いてみる"))
    assert problems("みなさんで声をかけてみる") == [] and problems("皆さんで動いてみる") == []


def test_long_tip_is_refused():
    assert any("文字より長い" in p for p in problems("ドリブル" * 30 + "してみる"))


def test_split_tips_takes_one_per_line_without_marks():
    assert checks.split_tips("・ひとつ目をためしてみる\n\n2. ふたつ目を意識してみる\n"
                             "- ひとつ目をためしてみる\n①みっつ目をやってみる") == [
        "ひとつ目をためしてみる", "ふたつ目を意識してみる", "みっつ目をやってみる"]
    assert checks.split_tips("") == [] and checks.split_tips(None) == []


def test_grounding_needs_the_scene():
    assert checks.tip_grounding_problems(GOOD, SOURCE) == []
    # a katakana word no scene mentions is an invented play
    assert any("ダンク" in p for p in checks.tip_grounding_problems(
        "ダンクを決めてみる", SOURCE))
    # nothing of the scene in the tip
    assert checks.tip_grounding_problems("周りの様子を見てみる", SOURCE)
    # general katakana words are allowed
    assert checks.tip_grounding_problems("チームの動きを見てパスを出してみる", SOURCE) == []
    assert checks.tip_grounding_problems("パスのあと、プレーの流れを見てみる", SOURCE) == []
    # a single-kanji noun is enough when the source has it with its neighbours
    assert checks.tip_grounding_problems(
        "全員で声をかけ合ってみる", "仲間に声をかけて合図を送っている") == []
    # words that only fill the suggestion are never the scene
    assert checks.tip_grounding_problems("意識して続けてみる", "意識して続けている") != []


def test_person_tips_errors():
    ctx = TipContext()
    sources = {"cam1-0001": SOURCE}
    ok = TipsOutput.model_validate({"tips_status": "generated", "tips": [
        {"text": GOOD, "evidence_refs": ["cam1-0001"]}]})
    assert checks.person_tips_errors(ok, {"cam1-0001"}, sources, "ドリブルとパスの話", ctx) == []
    bad = TipsOutput.model_validate({"tips_status": "generated", "tips": [
        {"text": GOOD, "evidence_refs": []},
        {"text": GOOD, "evidence_refs": ["cam9-0001"]},
        {"text": "上手に動く", "evidence_refs": ["cam1-0001"]},
        {"text": "パスを続けてみる", "evidence_refs": ["cam1-0001"]}]})
    errors = "\n".join(checks.person_tips_errors(bad, {"cam1-0001"}, sources, "", ctx))
    for part in ("3個までです", "1個目: 根拠にした場面のIDがありません", "2個目: 存在しない根拠ID",
                 "2個目: 同じヒント", "3個目: 文の終わり", "3個目: 人を評価する"):
        assert part in errors, (part, errors)


def test_team_tips_errors():
    ctx = TipContext()
    texts = {1: "ドリブルでボールを運びました", 2: "ドリブルから仲間にパスを出しました",
             3: "声をかけ合っていました"}
    ok = TeamTipsOutput.model_validate({"tips_status": "generated", "tips": [
        {"text": "ドリブルのあとにパスを出してみる", "people": [1, 2]}]})
    assert checks.team_tips_errors(ok, texts, ctx) == []
    bad = TeamTipsOutput.model_validate({"tips_status": "generated", "tips": [
        {"text": "ドリブルのあとにパスを出してみる", "people": [1]},
        {"text": "ドリブルのあとにパスを出してみる", "people": [1, 7]}]})
    errors = "\n".join(checks.team_tips_errors(bad, texts, ctx))
    assert "1個目: チームのヒントは、2人以上" in errors
    assert "2個目: 存在しない人の番号です: 7" in errors
    assert "同じヒントが重なっています" in errors


def test_trim_team_tips_keeps_the_shared_tips_and_drops_the_padding():
    texts = {1: "ドリブルでボールを運びました", 2: "ドリブルから仲間にパスを出しました",
             3: "声をかけ合っていました"}

    def draft(*tips):
        return TeamTipsOutput.model_validate({"tips_status": "generated", "tips": [
            {"text": t, "people": p} for t, p in tips]})

    out = draft(("ドリブルのあとにパスを出してみる", [1, 2]), ("声をかけてみる", [3]),
                ("ボールを運んでみる", [1, 9]))
    assert checks.trim_team_tips(out, texts) == 2
    assert [t.text for t in out.tips] == ["ドリブルのあとにパスを出してみる"]
    assert checks.team_tips_errors(out, texts, TipContext()) == []
    # more than three that stand: the first three
    many = draft(*[(f"ドリブルのあとにパスを出す{n}つめをためしてみる", [1, 2]) for n in range(5)])
    assert checks.trim_team_tips(many, texts) == 2 and len(many.tips) == 3
    # nothing would remain: left as it is, the checks say why
    none = draft(("声をかけてみる", [3]), ("ボールを運んでみる", [1]))
    assert checks.trim_team_tips(none, texts) == 0 and len(none.tips) == 2
    assert "2人以上" in "\n".join(checks.team_tips_errors(none, texts, TipContext()))
    # an answer of "insufficient evidence" is left alone
    nothing = TeamTipsOutput.model_validate({"tips_status": "insufficient_evidence", "tips": []})
    assert checks.trim_team_tips(nothing, texts) == 0


def test_output_schemas_keep_status_and_tips_consistent():
    with pytest.raises(ValueError):
        TipsOutput.model_validate({"tips_status": "generated", "tips": []})
    with pytest.raises(ValueError):
        TipsOutput.model_validate({"tips_status": "insufficient_evidence",
                                   "tips": [{"text": "x", "evidence_refs": []}]})
    with pytest.raises(ValueError):
        TeamTipsOutput.model_validate({"tips_status": "generated", "tips": []})
