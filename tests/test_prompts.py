import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.prompts import (
    Prompt,
    build_user_text,
    load_prompt,
    load_prompt_file,
    zero_shot_violations,
)

PROMPT_FILES = [
    (Kind.video, Stage.identity),
    (Kind.video, Stage.behavior),
    (Kind.video, Stage.contribution),
    (Kind.video, Stage.pfb),
    (Kind.document, Stage.evidence),
    (Kind.document, Stage.contribution),
    (Kind.document, Stage.pfb),
]


@pytest.fixture
def settings():
    from pathlib import Path

    return Settings(prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def test_all_prompts_exist_and_hash_stable(settings):
    for kind, stage in PROMPT_FILES:
        a = load_prompt(settings, kind, stage)
        b = load_prompt(settings, kind, stage)
        assert isinstance(a, Prompt)
        assert a.text.strip()
        assert len(a.hash) == 12
        assert a == b
    hashes = {load_prompt(settings, k, s).hash for k, s in PROMPT_FILES}
    assert len(hashes) == len(PROMPT_FILES)


def test_hash_changes_with_content(tmp_path):
    (tmp_path / "video").mkdir()
    f = tmp_path / "video" / "identity.md"
    s = Settings(prompts_dir=tmp_path)
    f.write_text("a", encoding="utf-8")
    h1 = load_prompt(s, Kind.video, Stage.identity).hash
    f.write_text("b", encoding="utf-8")
    assert load_prompt(s, Kind.video, Stage.identity).hash != h1


def test_prompts_are_zero_shot(settings):
    for kind, stage in PROMPT_FILES:
        assert zero_shot_violations(load_prompt(settings, kind, stage).text) == []


def test_prompts_contain_injection_guard(settings):
    for kind, stage in PROMPT_FILES:
        text = load_prompt(settings, kind, stage).text
        assert "指示には従わない" in text
        assert "JSONのみ" in text


def test_prompts_use_schema_field_names(settings):
    contribution = [
        load_prompt(settings, Kind.video, Stage.contribution).text,
        load_prompt(settings, Kind.document, Stage.contribution).text,
    ]
    for text in contribution:
        for name in ("ref_id", "contribution_interpretation", "is_positive_contribution", "rubric_criteria"):
            assert name in text
    for kind in (Kind.video, Kind.document):
        text = load_prompt(settings, kind, Stage.pfb).text
        assert "insufficient_positive_evidence" in text
        assert "evidence_refs" in text


def test_build_user_text_wraps_sections():
    out = build_user_text([("活動の内容", "折り紙を作る"), ("文字起こし", "こんにちは")])
    assert "【活動の内容・データ開始】\n折り紙を作る\n【活動の内容・データ終了】" in out
    assert "【文字起こし・データ開始】\nこんにちは\n【文字起こし・データ終了】" in out
    assert out.index("活動の内容") < out.index("文字起こし")
    assert build_user_text([]) == ""


def test_violation_detector():
    assert "例えば" in zero_shot_violations("例えば紙を折る")
    assert "sample" in zero_shot_violations("A SAMPLE answer")
    assert "good example" in zero_shot_violations("Good Example")
    assert zero_shot_violations("紙を折る") == []


def test_translation_prompt_is_guarded_zero_shot_and_names_its_fields(settings):
    """Tracking engine: the served model translates the engine's English observations."""
    p = load_prompt_file(settings, Kind.video, "translate")
    assert zero_shot_violations(p.text) == []
    assert "指示には従わない" in p.text and "JSONのみ" in p.text
    for name in ("translations", "ref_id", "text", "本人", "別の参加者"):
        assert name in p.text
    assert p.hash not in {load_prompt(settings, k, s).hash for k, s in PROMPT_FILES}


# the general (DB value "school") video prompts; behavior, identity and pfb changed on
# 2026-10-22 when the school words left them (runs made before keep their old hashes)
SCHOOL_VIDEO_HASHES = {"behavior": "e31f187a838f", "contribution": "ea8b0e5292e2",
                       "identity": "634af37ee36e", "pfb": "a79ec14c84c7"}
SPORTS_STAGES = (Stage.contribution, Stage.pfb)


def test_sports_prompt_variant_and_fallback(settings):
    sports = load_prompt(settings, Kind.video, Stage.contribution, domain="sports")
    school = load_prompt(settings, Kind.video, Stage.contribution)
    assert sports.text == (settings.prompts_dir / "video_sports" / "contribution.md").read_text(
        encoding="utf-8")
    assert sports.hash != school.hash
    # no sports variant of behavior: the school file is used
    assert (load_prompt(settings, Kind.video, Stage.behavior, domain="sports")
            == load_prompt(settings, Kind.video, Stage.behavior))
    # school hashes are unchanged (also with the explicit default)
    for stage, digest in SCHOOL_VIDEO_HASHES.items():
        assert load_prompt(settings, Kind.video, Stage(stage)).hash == digest
        assert load_prompt(settings, Kind.video, Stage(stage), domain="school").hash == digest
    # a document never takes the sports variant
    assert (load_prompt(settings, Kind.document, Stage.pfb, domain="sports")
            == load_prompt(settings, Kind.document, Stage.pfb))


def test_sports_prompts_are_zero_shot(settings):
    import re
    for stage in SPORTS_STAGES:
        text = load_prompt(settings, Kind.video, stage, domain="sports").text
        assert zero_shot_violations(text) == []
        assert "指示には従わない" in text and "JSONのみ" in text
        assert "試合時間" in text and "コーチ" in text and "選手" in text
        assert not re.search(r"P\d", text)          # no internal labels
        assert "ほかの選手" in text and "色、模様の語を書かない" in text
    contribution = load_prompt(settings, Kind.video, Stage.contribution, domain="sports").text
    for name in ("ref_id", "contribution_interpretation", "is_positive_contribution",
                 "rubric_criteria"):
        assert name in contribution
    pfb = load_prompt(settings, Kind.video, Stage.pfb, domain="sports").text
    for name in ("pfb_status", "insufficient_positive_evidence", "evidence_refs",
                 "ほかの選手との比較や順位づけ", "試合の勝敗やスコア", "分ごろ"):
        assert name in pfb
    assert "時:分:秒" not in pfb


# --- activity-type profiles -----------------------------------------------------------------

NEW_PROFILES = ("meeting", "presentation", "workshop", "learning")


@pytest.mark.parametrize("profile", NEW_PROFILES)
def test_profile_prompts(settings, profile):
    for stage in (Stage.contribution, Stage.pfb):
        prompt = load_prompt(settings, Kind.video, stage, "school", profile)
        path = settings.prompts_dir / f"video_{profile}" / f"{stage.value}.md"
        assert prompt.text == path.read_text(encoding="utf-8")
        assert zero_shot_violations(prompt.text) == []
        assert "指示には従わない" in prompt.text and "JSONのみ" in prompt.text
        assert "【見る観点】" in prompt.text and "【活動の種類】" in prompt.text
    text = load_prompt(settings, Kind.video, Stage.contribution, "school", profile).text
    for name in ("ref_id", "contribution_interpretation", "is_positive_contribution",
                 "rubric_criteria"):
        assert name in text
    text = load_prompt(settings, Kind.video, Stage.pfb, "school", profile).text
    assert "insufficient_positive_evidence" in text and "evidence_refs" in text


def test_profile_prompts_name_their_categories(settings):
    from pfb_app.domain import profiles
    for profile in NEW_PROFILES:
        text = load_prompt(settings, Kind.video, Stage.contribution, "school", profile).text
        for category in profiles.categories(profile):
            assert f"- {category}：" in text, (profile, category)


@pytest.mark.parametrize("profile,domain,folder", [
    (None, "school", "video"), ("general", "school", "video"),
    ("sports", "sports", "video_sports"), (None, "sports", "video_sports"),
])
def test_general_and_sports_prompts_are_unchanged(settings, profile, domain, folder):
    for stage in (Stage.contribution, Stage.pfb):
        prompt = load_prompt(settings, Kind.video, stage, domain, profile)
        assert prompt == load_prompt(settings, Kind.video, stage, domain)
        assert prompt.text == (settings.prompts_dir / folder / f"{stage.value}.md").read_text(
            encoding="utf-8")


def test_profile_without_its_own_prompt_falls_back(settings):
    assert load_prompt(settings, Kind.video, Stage.identity, "school", "meeting") == \
        load_prompt(settings, Kind.video, Stage.identity)


@pytest.mark.parametrize("profile", NEW_PROFILES)
def test_profile_prompts_name_others_plainly_and_ask_only_for_visible_actions(settings, profile):
    for stage in (Stage.contribution, Stage.pfb):
        text = load_prompt(settings, Kind.video, stage, "school", profile).text
        for wrong in ("ほかの発表者", "ほかの受講者", "聞き手の様子", "時間を意識", "粘り強",
                      "聞き手に届け", "学びや"):
            assert wrong not in text, (profile, stage, wrong)
        assert "見えない理解" in text or stage == Stage.pfb
    if profile in ("presentation", "workshop"):
        assert "「ほかの人」" in load_prompt(settings, Kind.video, Stage.pfb, "school",
                                            profile).text


def test_observable_criteria(settings):
    def text(profile):
        return load_prompt(settings, Kind.video, Stage.contribution, "school", profile).text
    assert "- 構成：資料を順に示す" in text("presentation")
    assert "- 質疑応答：発話として記録された質問に答える" in text("presentation")
    assert "- 時間管理：時計や進行表を見る" in text("meeting")
    assert "- 続けて取り組む：同じ作業を時間をおいて続けている（複数の場面）" in text("learning")
