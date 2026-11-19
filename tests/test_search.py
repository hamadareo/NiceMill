"""作業をさがす (services.search): words, names, sports and days. Synthetic works only."""
from datetime import date

from pfb_app.services import search
from pfb_app.services.search import Facts, Query
from test_wizard_ui_basic import env, video_assignment  # noqa: F401 (env)
from test_wizard_ui_people_wait import group


def facts(title="練習試合 11月", sport="basketball", day=date(2026, 11, 14), names=("山田 花子", "鈴木太郎"),
          kind="スポーツ"):
    label = {"basketball": "バスケットボール"}.get(sport, "") if sport else ""
    return Facts(1, title, sport, label, kind, day, tuple(names))


def test_every_word_must_match_and_letters_are_compared_loosely():
    f = facts()
    assert search.matches(f, Query("練習"))
    assert search.matches(f, Query("練習 花子"))            # a work's name and a person's
    assert search.matches(f, Query("練習　花子"))           # the wide space splits too
    assert not search.matches(f, Query("練習 次郎"))        # one word missing: not shown
    assert search.matches(f, Query("山田花子"))             # the name written without its space
    assert search.matches(f, Query("山田 花子"))
    assert search.matches(f, Query("バスケ"))               # a part of the sport's name
    assert search.matches(f, Query("スポーツ"))
    assert search.matches(f, Query(""))                    # nothing typed: everything passes
    assert search.matches(facts(title="Practice Match"), Query("ＰＲＡＣＴＩＣＥ match"))   # width, case


def test_days_are_matched_as_they_are_written():
    f = facts(day=date(2026, 11, 4))
    for word in ("11/4", "11/04", "11月4日", "2026-11-04", "2026/11/4", "2026年11月4日",
                 "11月", "2026年11月", "２０２６－１１－０４"):
        assert search.matches(f, Query(word)), word
    for word in ("11/5", "12月", "2025-11-04"):
        assert not search.matches(f, Query(word)), word
    assert not search.matches(facts(day=None), Query("11/4"))


def test_the_sport_and_the_period_narrow_further():
    today = date(2026, 11, 14)
    ball, other = facts(), facts(sport=None, kind="話し合い・会議", day=date(2026, 10, 1))
    assert search.matches(ball, Query(sport="basketball"), today)
    assert not search.matches(other, Query(sport="basketball"), today)
    assert search.matches(other, Query(sport=search.NO_SPORT), today)
    assert not search.matches(ball, Query(sport=search.NO_SPORT), today)
    assert search.matches(ball, Query(period=search.TODAY), today)
    assert not search.matches(facts(day=date(2026, 11, 13)), Query(period=search.TODAY), today)
    assert search.matches(facts(day=date(2026, 11, 8)), Query(period=search.WEEK), today)
    assert not search.matches(facts(day=date(2026, 11, 7)), Query(period=search.WEEK), today)
    assert search.matches(facts(day=date(2026, 10, 16)), Query(period=search.MONTH), today)
    assert not search.matches(other, Query(period=search.MONTH), today)
    assert not search.matches(facts(day=None), Query(period=search.WEEK), today)
    # all three together
    assert search.matches(ball, Query("花子", "basketball", search.WEEK), today)
    assert not search.matches(ball, Query("次郎", "basketball", search.WEEK), today)
    assert Query().active is False and Query(" ").active is False and Query("a").active is True
    assert Query(sport="soccer").active and Query(period=search.TODAY).active


def test_load_reads_names_sports_and_days_of_every_work(env):
    settings, repo = env
    plain = video_assignment(repo, "グループ活動の記録")
    group(repo, plain, "A班")
    sports = repo.create_assignment("県大会 準決勝", "video", "試合", "", "standard", False,
                                    domain="sports", sport="basketball")
    sid = group(repo, sports, "第1試合")
    loaded = search.load(repo)
    assert set(loaded) == {plain, sports}
    assert loaded[plain].names == ("花子", "太郎", "次郎") and loaded[plain].sport is None
    assert loaded[sports].sport == "basketball" and loaded[sports].sport_label == "バスケットボール"
    assert loaded[sports].day is not None
    assert search.filter_ids(loaded, Query("花子")) == {plain, sports}
    assert search.filter_ids(loaded, Query("バスケ")) == {sports}
    assert search.filter_ids(loaded, Query(sport=search.NO_SPORT)) == {plain}
    assert search.filter_ids(loaded, Query("県大会 第")) == set()      # a group's label is no word
    assert search.sports_present(loaded.values()) == [("basketball", "バスケットボール")]
    assert search.has_non_sport(loaded.values())
    # a person erased on request is no longer found by name
    repo.conn.execute("UPDATE subjects SET erased_at='2026-11-14T00:00:00+00:00'"
                      " WHERE submission_id=? AND display_name='花子'", (sid,))
    assert search.load(repo)[sports].names == ("太郎", "次郎")
    assert search.filter_ids(search.load(repo), Query("花子")) == {plain}


def test_a_person_left_out_of_the_evaluation_is_not_found_by_name(env):
    settings, repo = env
    aid = video_assignment(repo, "練習試合")
    sid = group(repo, aid, "A班")
    assert search.filter_ids(search.load(repo), Query("次郎")) == {aid}
    repo.conn.execute("UPDATE subjects SET excluded=1 WHERE submission_id=? AND display_name='次郎'",
                      (sid,))
    loaded = search.load(repo)
    assert loaded[aid].names == ("花子", "太郎")
    assert search.filter_ids(loaded, Query("次郎")) == set()
    assert search.filter_ids(loaded, Query("花子")) == {aid}


def test_sports_come_in_the_order_of_the_list_and_unknown_ones_last():
    fs = [facts(sport="soccer"), facts(sport="basketball"), facts(sport="hockey"),
          facts(sport=None)]
    present = search.sports_present(fs)
    assert [k for k, _ in present] == ["basketball", "soccer", "hockey"]
