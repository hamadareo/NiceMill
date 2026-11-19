"""次にためしてみること: the two prompts (a person's, the group's), general and sports."""
import re
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.prompts import (load_prompt, load_prompt_named, zero_shot_violations)


@pytest.fixture
def settings():
    return Settings(prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def prompts(settings, domain):
    return {"person": load_prompt(settings, Kind.video, Stage.tips, domain),
            "team": load_prompt_named(settings, Kind.video, "team_tips", domain)}


@pytest.mark.parametrize("domain", ["school", "sports"])
def test_the_prompts_are_zero_shot_and_name_their_output(settings, domain):
    for which, p in prompts(settings, domain).items():
        assert zero_shot_violations(p.text) == []
        assert "指示には従わない" in p.text and "JSONのみ" in p.text
        assert "tips_status" in p.text and "insufficient_evidence" in p.text
        assert not re.search(r"P\d", p.text)                       # no internal labels
        for word in ("70文字", "してみる", "数字", "評価する言葉", "内部ID", "カメラ名"):
            assert word in p.text, (which, word)
    assert "evidence_refs" in prompts(settings, domain)["person"].text
    assert "people" in prompts(settings, domain)["team"].text


def test_sports_says_coach_and_player_and_the_general_does_not(settings):
    sports, general = prompts(settings, "sports"), prompts(settings, "school")
    for p in sports.values():
        # 「別の参加者」 is the events' own word for another person (behavior records)
        assert "コーチ" in p.text and "選手" in p.text
        assert "参加者" not in p.text.replace("別の参加者", "").replace("参加者ID", "")
    for p in general.values():
        assert "評価者" in p.text and "参加者" in p.text
        assert "コーチ" not in p.text and "選手" not in p.text and "チーム" not in p.text
    assert sports["person"].hash != general["person"].hash
    assert sports["team"].hash != general["team"].hash


def test_the_prompts_forbid_what_the_checks_refuse(settings):
    """What a prompt forbids and what pipeline.checks refuses agree on the main points."""
    text = prompts(settings, "school")["person"].text
    for phrase in ("数字、回数", "時間", "人を評価する言葉", "否定的な言い方", "入力にない動き",
                   "すべき"):
        assert phrase in text


def test_a_profile_without_its_own_prompt_uses_the_general_one(settings):
    assert (load_prompt(settings, Kind.video, Stage.tips, "school", "meeting")
            == load_prompt(settings, Kind.video, Stage.tips))
