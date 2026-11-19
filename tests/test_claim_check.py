"""The action-sentence re-check: what a sentence claims, the physical necessary condition for a
shot, how the two answers of the second question combine, and how a sentence is weakened.
Synthetic inputs only."""
import pytest

from pfb_app.pipeline import claim_check as cc
from pfb_app.pipeline.claim_check import Cues, Votes, YES, NO, UNKNOWN


@pytest.mark.parametrize("text,action", [
    ("ボールを受け取りシュートした。", "shoot"),
    ("ボールをバスケットに向かって投げた。", "shoot"),
    ("シュートをブロックするためにジャンプした。", "defend"),
    ("守られながらボールをドリブルした。", "dribble"),
    ("ボールを別の参加者にパスした。", "pass"),
    ("バスケットに向かって走った。", "run"),
    ("別の参加者からボールを受け取った。", "none"),
])
def test_claimed_action(text, action):
    assert cc.claimed_action(text) == action


def test_unchecked_without_any_cue():
    v = cc.check("シュートした。")
    assert v.status == "unchecked" and v.show == "keep"
    assert cc.weaken("シュートした。", v) == "シュートした。"


def test_shot_vetoed_when_arms_never_up():
    cues = Cues(wrist_above_head=-0.2, pose_frames=5)
    v = cc.check("シュートした。", cues, Votes(NO, YES))
    assert v.status == "contradicted" and v.show == "fallback"
    assert cc.check("シュートした。", cues, None).show == "fallback"
    assert cc.check("シュートした。", cues, Votes(YES, UNKNOWN)).show == "hedge"
    assert cc.weaken("シュートした。", v) == cc.SAFE_TEXT


def test_unmeasured_cue_never_vetoes():
    assert cc.physical_veto("shoot", Cues(None, 5)) is None
    assert cc.physical_veto("shoot", Cues(-0.5, 2)) is None       # too few frames
    assert cc.physical_veto("dribble", Cues(-0.5, 9)) is None     # only a shot needs arms up


def test_votes_combine():
    c = Cues(0.2, 6)
    assert cc.check("シュートした。", c, Votes(YES, NO)).status == "supported"
    assert cc.check("シュートした。", c, Votes(YES, UNKNOWN)).status == "supported"
    # a shot needs its arm cue measured; without it a yes only hedges (other actions need none)
    assert cc.check("シュートした。", Cues(None, 0), Votes(YES, NO)).status == "unclear"
    assert cc.check("ドリブルした。", None, Votes(YES, NO)).status == "supported"
    assert cc.check("シュートした。", c, Votes(NO, UNKNOWN)).status == "unclear"   # a lone no only hedges
    assert cc.check("シュートした。", c, Votes(NO, YES)).status == "contradicted"
    assert cc.check("シュートした。", c, Votes(UNKNOWN, UNKNOWN)).status == "unclear"
    assert cc.check("シュートした。", c, Votes(YES, YES)).status == "unclear"


def test_hedge_keeps_claim_but_not_as_fact():
    assert cc.hedge("ボールをシュートした。") == "ボールをシュートしているように見える（要確認）。"
    assert cc.hedge("シュートを試みた。") == "シュートを試みているように見える（要確認）。"
    assert cc.hedge("ボールを持っている") == "ボールを持っている（要確認）。"


def test_annotate_keeps_original_and_adds_keys():
    m = {"id": "a", "why": "シュートした。"}
    out = cc.annotate(m, cc.check(m["why"], Cues(-0.3, 5), Votes(UNKNOWN, UNKNOWN)))
    assert out["why"] == "シュートした。" and out["why_shown"] == cc.SAFE_TEXT
    assert out["claim_check"]["status"] == "contradicted"
    assert "id" in out and m.get("claim_check") is None


def test_questions_and_parse():
    pos, ctl = cc.question_texts("shoot", 8)
    assert "8枚" in pos and "シュート" in pos and "シュート" in ctl
    assert cc.question_texts("none", 8) is None
    assert cc.parse_answer("はい。") == YES and cc.parse_answer("いいえ") == NO
    assert cc.parse_answer("") == UNKNOWN and cc.parse_answer(None) == UNKNOWN
    assert cc.parse_answer("わからない") == UNKNOWN


def test_ask_votes_uses_the_callers_model():
    seen = []

    def fake(images, prompt):
        seen.append((len(images), prompt))
        return "いいえ" if "していますか" in prompt else "はい"

    v = cc.ask_votes("shoot", [object()] * 8, fake)
    assert (v.pos, v.ctl) == (NO, YES) and len(seen) == 2 and seen[0][0] == 8
    assert cc.ask_votes("none", [], fake) is None
