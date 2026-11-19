"""やさしい文章版 (rel8-real): the youngest level's kanji are written in hiragana by a table.

The 8B model keeps 試合 however it is told and reads 味方 as あいがた when asked for a reading,
so the conversion is a table (pipeline.plain_readings) and a word it lacks stays in kanji, which
the check refuses. Synthetic texts only."""
import pytest

from pfb_app.pipeline import plain_checks as pc
from pfb_app.pipeline import plain_readings as pr


def fit(text, names=()):
    return pc.fit(text, "low", names)


def test_words_of_later_school_years_are_written_in_hiragana():
    assert fit("試合が始まって、相手の攻撃を防いだ。") == "しあいがはじまって、あいてのこうげきをふせいだ。"


def test_kanji_of_the_first_two_years_stay():
    assert fit("山で友だちと走った。") == "山で友だちと走った。"
    assert fit("試合開始直後") == "しあいかいし直後"      # 直後 is all first-two-year kanji


def test_a_word_may_keep_the_kanji_the_reader_knows():
    assert fit("1分頃、走った。") == "1分ごろ、走った。"


def test_a_one_kanji_entry_never_changes_a_longer_word():
    assert fit("動いた") == "うごいた"
    assert fit("動画を見た") == "どうがを見た"      # 動画 is its own entry, not 動 + 画
    assert fit("移動した") == "いどうした"          # not 移 + 動
    assert fit("動作を見た") == "動作を見た"          # unknown: left as written, never guessed


def test_a_word_the_table_lacks_stays_and_the_check_refuses_it():
    text = fit("柱を作った。")
    assert text == "柱を作った。"
    assert any("柱" in p for p in pc.problems("柱を作った。", text, "low"))


def test_the_reading_depends_on_what_follows():
    assert fit("シュートを試みた") == "シュートをこころみた"
    assert fit("5分経った") == "5分たった"


def test_names_are_never_converted():
    assert fit("試合太郎さんが走った。", ["試合太郎"]) == "試合太郎さんが走った。"
    assert fit("試合太郎さんが走った。") == "しあい太郎さんが走った。"


def test_the_other_levels_are_not_touched():
    assert pc.fit("試合が始まった", "mid") == "試合が始まった"


def test_every_entry_is_hiragana_but_for_kanji_the_reader_knows():
    for key, reading in pr.READINGS.items():
        assert all("ぁ" <= c <= "ゖ" or c == "ー" or c in pc.LOW_KANJI or c.isdigit()
                   for c in reading), (key, reading)
        assert any(pr._is_kanji(c) and c not in pc.LOW_KANJI for c in key), key   # a useful key
