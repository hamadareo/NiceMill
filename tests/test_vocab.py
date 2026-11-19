from pfb_app.domain.vocab import (SPORT_KEYS, SPORT_LABELS, default_person_name, mark_name,
                                  number_name, numberless_name, opponent_name, words)


def test_words():
    assert words("sports").work == "試合・練習"
    assert words(None).people == "参加者"      # the general words (DB value "school")
    assert words("school").work == "活動"
    assert words("zzz").giver == "評価者"


def test_names():
    assert default_person_name("sports", 3) == "選手3"
    assert default_person_name("school", 3) == "参加者3"
    assert opponent_name(2) == "相手選手2"
    assert numberless_name(1) == "選手（番号なし）1"
    assert number_name(10) == "#10"


def test_mark_name():
    assert mark_name("P12") == "12番の人"
    assert mark_name("P7") == "7番の人"


def test_sport_keys_match_engine():
    # literal copy of pfb3x3/sports.py SPORT_KEYS (the engine's list, same order)
    assert SPORT_KEYS == ("basketball", "soccer", "baseball_softball", "volleyball", "handball",
                          "rugby", "tennis_badminton", "track", "swimming", "motor_racing", "other")
    assert set(SPORT_LABELS) == set(SPORT_KEYS)
