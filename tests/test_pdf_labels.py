"""export.labels: the one place the PDFs take their words and category names from."""
import pytest

from pfb_app.export import labels
from pfb_app.export.labels import (OTHER, categories_for, category_of, handout_file_name,
                                   model_words, words_for)


@pytest.fixture
def no_profiles(monkeypatch):
    """The domain mapping used while the activity profiles are not there."""
    monkeypatch.setattr(labels, "_profiles", None)

SPORTS = {"domain": "sports", "sport": "basketball"}
GENERAL = {"domain": "school", "sport": None}


def test_words_follow_the_domain_for_now(no_profiles):
    assert words_for(SPORTS).people == "選手" and words_for(SPORTS).giver == "コーチ"
    assert words_for(GENERAL).people == "参加者" and words_for(GENERAL).giver == "評価者"
    assert words_for({}).people == "参加者"


def test_sports_categories_are_the_eight_plain_words_in_a_fixed_order(no_profiles):
    cats = categories_for(SPORTS)
    assert [c.label for c in cats] == ["パス", "シュート・得点", "守り", "位置どり", "声かけ",
                                       "切りかえ", "がんばり・取り返し", "その他"]
    assert cats[-1].key == OTHER


def test_general_categories_have_at_most_eight_and_end_with_other(no_profiles):
    cats = categories_for(GENERAL)
    assert 2 <= len(cats) <= 8 and cats[-1].key == OTHER
    assert "伝える・話し合う" in [c.label for c in cats]


def test_every_engine_action_type_has_a_category(no_profiles):
    """Every type of the engine's sports and generic enums (pfb3x3/domain.py) maps somewhere;
    an unknown type is その他, a missing one is no category at all."""
    sports = ("pass", "shot_or_score", "assist", "defense", "positioning", "off_ball",
              "transition", "communication", "effort_recovery", "other")
    generic = ("build_or_assemble", "hand_over", "hold_or_support", "adjust_or_fix",
               "manipulate_object", "point_or_gesture", "speak_or_discuss", "move", "observe",
               "other")
    for work, types in ((SPORTS, sports), (GENERAL, generic)):
        cats = {c.key for c in categories_for(work)}
        for t in types:
            assert category_of(work, t) in cats
    assert category_of(SPORTS, "assist") == category_of(SPORTS, "pass")
    assert category_of(SPORTS, "juggling") == OTHER
    assert category_of(SPORTS, None) is None and category_of(SPORTS, "") is None


def test_each_category_has_its_own_shape_and_colour_slot(no_profiles):
    for work in (SPORTS, GENERAL):
        cats = categories_for(work)
        assert len({c.shape for c in cats}) == len(cats)
        assert len({c.colour for c in cats}) == len(cats)
        assert cats[0].colour == "#2a78d6"   # slot 1 of the fixed order


def test_handout_file_names_are_numbered_and_safe():
    assert handout_file_name(3, "山田花子", 13) == "03_山田花子.pdf"
    assert handout_file_name(1, "#10", 9) == "1_#10.pdf"
    bad = handout_file_name(2, 'a/b\\c:d*e?"f<g>h|i\x00', 12)
    assert bad == "02_a_b_c_d_e_f_g_h_i.pdf"
    assert handout_file_name(4, " ..  ", 4) == "4_参加者.pdf"
    assert handout_file_name(4, "..", 4, fallback="選手") == "4_選手.pdf"
    assert len(handout_file_name(1, "長" * 200, 1)) <= 70


def test_model_and_camera_words():
    assert model_words("standard") == "標準" and model_words("high") == "高精度"
    assert labels.camera_word("cam2") == "カメラ2"


# --- the activity profiles (feat/profiles), through a stand-in module ------------------------

class FakeProfiles:
    """The feat/profiles API: of_work, words, categories, category_of."""
    CATS = {"meeting": ("発言", "傾聴", "まとめ", "提案", "合意づくり", "時間管理"),
            "sports": ("パス", "シュート・得点", "アシスト", "守り", "位置どり",
                       "ボールを持たない動き", "切りかえ", "声かけ", "がんばり・取り返し", "その他")}

    @staticmethod
    def of_work(work):
        return work.get("profile") or ("sports" if work.get("domain") == "sports" else "general")

    @staticmethod
    def words(profile):
        from pfb_app.domain.vocab import words
        return words("sports" if profile == "sports" else "school")

    @classmethod
    def categories(cls, profile):
        return cls.CATS.get(profile, ("発言・話し合い", "その他"))

    @classmethod
    def category_of(cls, profile, criteria=(), action_type=None):
        cats = cls.categories(profile)
        for c in criteria:
            if c in cats:
                return c
        if profile == "meeting":      # the new profiles count by 評価の観点 only
            return None
        return {"pass": "パス", "other": "その他"}.get(action_type or "")


def test_labels_follow_the_profile_when_profiles_exist(monkeypatch):
    monkeypatch.setattr(labels, "_profiles", FakeProfiles)
    meeting = {"domain": "school", "profile": "meeting"}
    cats = categories_for(meeting)
    assert [c.label for c in cats] == list(FakeProfiles.CATS["meeting"])
    assert category_of(meeting, None, ["傾聴"]) == "傾聴"
    assert category_of(meeting, "pass", ["関係ない観点"]) is None
    sports = categories_for(SPORTS)
    assert len(sports) == 10 and sports[-1].key == OTHER and sports[-1].label == "その他"
    # nine named categories: hues come round again, shapes never do
    assert len({c.shape for c in sports}) == len(sports)
    assert category_of(SPORTS, "pass") == "パス" and category_of(SPORTS, "other") == OTHER
    assert words_for(SPORTS).giver == "コーチ"


def test_every_shape_draws():
    from pfb_app.export import charts
    from pfb_app.export.labels import SHAPES
    for shape in (*SHAPES, "ring", "plus"):
        assert "nan" not in charts._marker(shape, "#2a78d6", 10, 10, 5).lower()


def test_more_categories_than_shapes_fold_into_other(monkeypatch):
    from pfb_app.export.labels import SHAPES

    class Many(FakeProfiles):
        CATS = {"general": tuple(f"観点{k}" for k in range(1, 13)) + ("その他",)}

        @classmethod
        def category_of(cls, profile, criteria=(), action_type=None):
            return next((c for c in criteria if c in cls.categories(profile)), None)
    monkeypatch.setattr(labels, "_profiles", Many)
    work = {"domain": "school"}
    cats = categories_for(work)
    assert len(cats) == len(SHAPES) + 1 and cats[-1].key == OTHER
    assert len({c.shape for c in cats}) == len(cats)
    assert category_of(work, None, ["観点1"]) == "観点1"
    assert category_of(work, None, ["観点12"]) == OTHER          # folded
    assert labels.category(work, category_of(work, None, ["観点12"])).label == "その他"
