from pfb_app.domain.models import BehaviorEventDraft, Judgment, PfbOutput
from pfb_app.pipeline.checks import (
    evidence_ref_errors,
    judgment_alignment_errors,
    normalize_text,
    pfb_text_warnings,
    unknown_participant_errors,
    verify_quote,
)


def _j(ref_id):
    return Judgment(ref_id=ref_id, contribution_interpretation="x", is_positive_contribution=True)


def _ev(participant):
    return BehaviorEventDraft(
        video_time="00:01", participant=participant, observed_behavior="x",
        evidence_source="video", confidence="high",
    )


def test_normalize_text_nfkc_and_removes_whitespace():
    assert normalize_text("ＡＢＣ １２３\n 　あ\tい") == "ABC123あい"


def test_verify_quote_ignores_whitespace_and_width():
    page = "本日の議論では、ＡＢＣ案を採用\nする方向で意見がまとまった。"
    assert verify_quote("ABC案を採用 する方向で意見", page)


def test_verify_quote_rejects_missing_and_short():
    page = "本日の議論では、ABC案を採用する方向で意見がまとまった。"
    assert not verify_quote("存在しない引用文章ですよこれは", page)
    assert not verify_quote("ABC案を採用", page)  # 7 chars < 10


def test_alignment_ok():
    assert judgment_alignment_errors(["a", "b"], [_j("b"), _j("a")]) == []


def test_alignment_detects_missing_and_duplicate():
    errs = judgment_alignment_errors(["a", "b", "c"], [_j("a"), _j("a"), _j("z")])
    assert any("b" in e and "欠落" in e for e in errs)
    assert any("c" in e and "欠落" in e for e in errs)
    assert any("a" in e and "重複" in e for e in errs)
    assert any("z" in e and "余分" in e for e in errs)


def test_unknown_participant():
    assert unknown_participant_errors([_ev("Pa")], {"Pa", "Pb"}) == []
    errs = unknown_participant_errors([_ev("Pa"), _ev("Px")], {"Pa", "Pb"})
    assert len(errs) == 1 and "Px" in errs[0]


def test_pfb_warnings_clean():
    assert pfb_text_warnings("議論をまとめ、全員の意見を引き出していました。", {"Pa"}, {"cam1"}) == []


def test_pfb_warnings_detects_id_and_advice():
    w = pfb_text_warnings("Pzさんは次回もっと", {"Pz", "Pa"}, {"cam1"})
    assert len(w) == 2
    assert any("Pz" in x for x in w)
    assert any("次回" in x for x in w)


def test_pfb_warnings_id_is_whole_token():
    assert pfb_text_warnings("Paper を読み上げました。", {"Pa"}, set()) == []
    assert len(pfb_text_warnings("Pa は発言した。", {"Pa"}, set())) == 1


def test_pfb_warnings_camera_color_confidence():
    assert any("cam1" in x for x in pfb_text_warnings("cam1 で撮影", set(), {"cam1"}))
    assert any("赤" in x for x in pfb_text_warnings("赤い服の人が話した", set(), set()))
    assert any("色" in x for x in pfb_text_warnings("明るい色の資料", set(), set()))
    assert pfb_text_warnings("確信度が高い", set(), set())
    assert pfb_text_warnings("confidence high", set(), set())
    assert pfb_text_warnings("確実性は９０％です", set(), set())


def test_pfb_warnings_color_false_positives():
    for t in ["面白い提案でした。", "黒板に整理しました。", "赤ちゃん向けの話題", "色々な意見を促した。"]:
        assert pfb_text_warnings(t, set(), set()) == [], t
    assert pfb_text_warnings("白い資料を示した", set(), set())


def test_pfb_warnings_sports_color_false_positives():
    for t in ["白熱した試合で、味方にパスを出していました。", "白星につながる守備でした。",
              "黒星の試合でも、最後まで走っていました。"]:
        assert pfb_text_warnings(t, set(), set()) == [], t


def test_pfb_warnings_advice_words():
    for t in ["〜すべきだ", "改善の余地", "頑張りましょう", "もっと話す"]:
        assert pfb_text_warnings(t, set(), set()), t


def _pfb(refs):
    return PfbOutput(pfb_status="generated", pfb="良かった。", evidence_refs=refs)


def test_evidence_refs():
    assert evidence_ref_errors(_pfb(["e1"]), {"e1", "e2"}) == []
    errs = evidence_ref_errors(_pfb(["e1", "e9"]), {"e1"})
    assert len(errs) == 1 and "e9" in errs[0]
    assert evidence_ref_errors(_pfb([]), {"e1"})  # generated without refs
    insuff = PfbOutput(pfb_status="insufficient_positive_evidence", pfb=None, evidence_refs=[])
    assert evidence_ref_errors(insuff, set()) == []
