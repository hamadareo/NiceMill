"""やさしい文章版 (rel8): the pure checks a reworded text must pass before it may be used.

A reworded text keeps the facts of the approved one: its numbers, names and loan words stay, it
adds no action, praise, person or number, and (for the youngest readers) it is written with the
kanji of the first two school years only. Synthetic texts only."""
import pytest

from pfb_app.pipeline import plain_checks as pc

ORIGINAL = "花子さんは柱を丁寧に折って、塔の土台を作りました。"
NAMES = ["花子", "太郎"]
LOW_OK = "花子さんは、はしらを ていねいに おって、とうの どだいを つくりました。"


def errors(rewrite, level="low", original=ORIGINAL, names=NAMES, **kw):
    return pc.problems(original, rewrite, level, names=names, **kw)


def said(found, word):
    return any(word in e for e in found)


# --- the kanji of the youngest level -----------------------------------------------------------

def test_the_first_two_years_have_the_standard_number_of_kanji():
    assert len(pc.GRADE1) == 80 and len(pc.GRADE2) == 160
    assert not set(pc.GRADE1) & set(pc.GRADE2)
    assert set("山川学校") <= set(pc.LOW_KANJI) and "柱" not in pc.LOW_KANJI


def test_a_text_in_plain_words_passes():
    assert errors(LOW_OK) == []


def test_a_kanji_beyond_the_second_year_is_refused_with_its_word():
    found = errors("花子さんは、柱を ていねいに おって、とうの どだいを つくりました。")
    assert said(found, "柱") and said(found, "ひらがな")


def test_a_name_keeps_its_kanji_even_when_they_are_hard():
    original = "鷹野さんは柱を丁寧に折りました。"
    ok = errors("鷹野さんは、はしらを ていねいに おりました。", original=original, names=["鷹野"])
    assert ok == []


def test_the_middle_level_may_use_any_kanji():
    assert errors("花子さんは柱を丁寧に折って、塔の土台を作りました。", level="mid") == []


# --- facts: numbers, names, loan words ---------------------------------------------------------

def test_the_numbers_of_the_original_stay_and_none_is_added():
    original = "花子さんは3回、パスを出しました。"
    good = "花子さんは、3かい、パスを だしました。"
    assert errors(good, original=original, names=["花子"]) == []
    assert said(errors("花子さんは、なんども、パスを だしました。", original=original,
                       names=["花子"]), "3")
    assert said(errors("花子さんは、4かい、パスを だしました。", original=original,
                       names=["花子"]), "4")


def test_a_number_may_be_written_in_kanji_or_as_a_reading():
    original = "花子さんは2回、パスを出しました。"
    for ok in ("花子さんは、二かい、パスを だしました。", "花子さんは、2かい、パスを だしました。"):
        assert errors(ok, original=original, names=["花子"]) == []
    original = "ふたりで、パスを出しました。".replace("ふたり", "2人")
    assert errors("ふたりで、パスを だしました。", original=original, names=[]) == []
    assert said(errors("三人で、パスを だしました。", original=original, names=[]), "3")


def test_a_kanji_numeral_inside_a_word_is_not_a_number():
    original = "花子さんは一緒にパスを出しました。"
    found = errors("花子さんは、一緒に、パスを だしました。", original=original, names=["花子"],
                   level="mid")
    assert not said(found, "数字")


def test_the_names_in_the_original_are_written_as_they_are():
    found = errors("はなこさんは、はしらを ていねいに おって、とうの どだいを つくりました。")
    assert said(found, "花子")


def test_a_name_that_is_not_in_the_original_is_refused():
    found = errors("花子さんは、太郎さんと、はしらを ていねいに おって、とうを つくりました。")
    assert said(found, "太郎")


def test_a_hash_number_name_is_matched_as_a_whole():
    original = "#7さんは、ボールを運びました。"
    assert errors("#7さんは、ボールを はこびました。", original=original, names=["#7", "#70"]) == []
    assert said(errors("#70さんは、ボールを はこびました。", original=original,
                       names=["#7", "#70"]), "#70")


def test_a_san_after_a_word_of_the_original_is_not_a_new_person_but_a_new_one_is():
    found = errors("花子さんは、お母さんと、はしらを ていねいに おって、とうを つくりました。")
    assert said(found, "母さん")


def test_loan_words_of_the_original_stay_and_new_ones_are_refused():
    original = "花子さんはドリブルでボールを運びました。"
    good = "花子さんは、ドリブルで、ボールを はこびました。"
    assert errors(good, original=original, names=["花子"]) == []
    assert said(errors("花子さんは、ボールを つきながら はこびました。", original=original,
                       names=["花子"]), "ドリブル")
    assert said(errors("花子さんは、ナイスドリブルで、ボールを はこびました。", original=original,
                       names=["花子"]), "ナイスドリブル")


def test_a_latin_word_that_is_not_in_the_original_is_refused():
    assert said(errors(LOW_OK + " OK!"), "OK")


# --- no new praise, no new claim ---------------------------------------------------------------

def test_praise_the_original_does_not_have_is_refused():
    found = errors("花子さんは、すごい はやさで、はしらを おって、とうを つくりました。")
    assert said(found, "すごい")
    original = "花子さんは上手にパスを出しました。"
    assert errors("花子さんは、じょうずに、パスを だしました。", original=original,
                  names=["花子"]) == []
    assert said(errors("花子さんは、パスを だしました。てんさいです。", original=original,
                       names=["花子"]), "てんさい")


def test_a_claim_in_hiragana_counts_like_the_kanji_one():
    found = errors("花子さんは、みんなで きょうりょくして、とうを つくりました。")
    assert said(found, "きょうりょく")


def test_the_same_checks_as_the_feedback_text_apply_to_what_is_new():
    assert said(errors("花子さんは、あかい はしらを ていねいに おって、とうを つくりました。"), "色")
    found = errors(LOW_OK, internal_ids={"cam1-0001-01", "Pa"}, camera_names={"cam1"})
    assert found == []
    found = errors(LOW_OK + " Pa", internal_ids={"Pa"})
    assert said(found, "Pa")
    assert said(errors(LOW_OK + " つぎは もっと がんばろう。"), "もっと")


def test_a_warning_the_original_already_has_is_not_blamed_on_the_rewrite():
    original = "花子さんは試合開始直後に、パスを出しました。"
    found = errors("花子さんは、しあいが はじまって すぐに、パスを だしました。",
                   original=original, names=["花子"], clock_known=False)
    assert found == []
    found = errors("花子さんは、序盤に、パスを だしました。", original=ORIGINAL, names=["花子"],
                   clock_known=False)
    assert found


def test_talk_about_the_rewriting_is_refused():
    assert said(errors("やさしく いうと、" + LOW_OK), "前置き")


def test_brackets_are_not_used_for_a_reading():
    assert said(errors("花子さんは、柱（はしら）を ていねいに おりました。"), "かっこ")


# --- shape --------------------------------------------------------------------------------------

def test_an_empty_text_is_refused():
    assert errors("   ")


def test_a_text_much_shorter_or_longer_than_the_original_is_refused():
    assert said(errors("つくりました。"), "短")
    assert said(errors(LOW_OK * 6), "長")


def test_the_youngest_level_wants_short_sentences():
    long_sentence = "花子さんは、はしらを ていねいに おって、とうの どだいを つくって、みんなが のぼれるように した。"
    found = errors(long_sentence, original=ORIGINAL + "花子さんは笑いました。")
    assert said(found, "文")
    assert errors(long_sentence, level="mid") == []


def test_more_sentences_than_the_original_could_hold_are_refused():
    many = "".join(f"花子さんは、はしらを おりました{n}。" for n in range(8))
    assert said(errors(many, level="mid"), "文の数")


def test_the_same_text_may_pass_unchanged_at_the_middle_level():
    assert errors(ORIGINAL, level="mid") == []


def test_words_for_the_three_versions():
    assert [pc.LEVELS[k].label for k in ("low", "mid")] == [
        "6〜8歳向け（ひらがなが多く、短い文）", "9〜12歳向け（むずかしい言葉をやさしく）"]
    assert pc.ORIGINAL_LABEL == "13歳以上向け（いまの文のまま）"


def test_the_kanji_list_the_model_is_given_is_the_checked_one():
    assert pc.kanji_for_prompt("low") == "".join(pc.LOW_KANJI)
    assert pc.kanji_for_prompt("mid") is None


@pytest.mark.parametrize("bad", ["x", "", None])
def test_an_unknown_level_is_an_error_not_a_pass(bad):
    with pytest.raises(KeyError):
        pc.problems(ORIGINAL, LOW_OK, bad)


# --- rel8-real: what a real model needed ---------------------------------------------------------

def test_a_faint_loan_word_may_be_left_out_and_a_plain_word_may_stand_for_the_others():
    original = "花子さんはドリブルのプレーで相手をディフェンスし、味方をサポートしました。"
    good = "花子さんは、ドリブルで、あいてを まもって、みかたを たすけました。"
    assert errors(good, original=original, names=["花子"]) == []
    found = errors("花子さんは、あいてを まもって、みかたを たすけました。", original=original,
                   names=["花子"])
    assert said(found, "ドリブル")


def test_a_missing_loan_word_is_reported_once():
    original = "花子さんはドリブルして、ドリブルして、またドリブルしました。"
    found = errors("花子さんは、ボールを はこびました。", original=original, names=["花子"])
    assert sum("ドリブル" in e for e in found) == 1


def test_a_plain_word_the_original_licenses_is_not_a_new_claim():
    original = "花子さんは味方のプレーを支援しました。"
    assert errors("花子さんは、なかまを たすけました。", original=original, names=["花子"]) == []
    found = errors("花子さんは、なかまを たすけました。", original="花子さんはボールを運びました。",
                   names=["花子"])
    assert said(found, "なかま") or said(found, "たすけ")


def test_a_loan_word_the_original_wrote_plainly_may_be_used_only_for_it():
    original = "花子さんは得点の機会をつくりました。"
    assert errors("花子さんは、てんの チャンスを つくりました。", original=original,
                  names=["花子"]) == []
    found = errors("花子さんは、チャンスを つくりました。", original="花子さんはボールを運びました。",
                   names=["花子"])
    assert said(found, "チャンス")


def test_a_result_or_another_action_the_original_does_not_say_is_refused():
    original = "花子さんはシュートを実行して、ボールを持たずに動きました。"
    found = errors("花子さんは、シュートを きめて、ボールを てばなして うごきました。",
                   original=original, names=["花子"])
    assert said(found, "動作や結果")
    assert errors("花子さんは、シュートを して、ボールを もたずに うごきました。",
                  original=original, names=["花子"]) == []
    assert errors("花子さんは、シュートを決めて、ボールを手放して動きました。",
                  original="花子さんはシュートを決めて、ボールを手放して動きました。",
                  names=["花子"], level="mid") == []
