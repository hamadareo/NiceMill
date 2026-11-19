"""Activity-type profiles: words, categories, the scene mapping and the work's profile."""
import pytest

from pfb_app.domain import profiles, vocab
from pfb_app.domain.profiles import (GENERAL, LEARNING, MEETING, PRESENTATION, PROFILES,
                                     SPORTS, WORKSHOP)
from test_neutral_wording import clean

ALL = (GENERAL, MEETING, SPORTS, PRESENTATION, WORKSHOP, LEARNING)


def test_six_profiles_in_screen_order():
    assert tuple(PROFILES) == ALL
    assert [profiles.label(p) for p in ALL] == [
        "一般", "話し合い・会議", "スポーツ", "発表・パフォーマンス", "研修・ワークショップ",
        "学習活動"]


@pytest.mark.parametrize("profile,expected", [
    (GENERAL, ("参加者", "評価者", "活動", "場面")),
    (MEETING, ("参加者", "進行役", "会議", "場面")),
    (SPORTS, ("選手", "コーチ", "試合・練習", "場面")),
    (PRESENTATION, ("発表者", "聞き手・評価者", "発表", "場面")),
    (WORKSHOP, ("受講者", "講師", "研修", "場面")),
    (LEARNING, ("参加者", "指導者", "活動", "場面")),
])
def test_words(profile, expected):
    w = profiles.words(profile)
    assert (w.people, w.giver, w.work, w.moment) == expected
    assert vocab.words(profile) == w            # one vocabulary table
    for field in w.__dataclass_fields__:
        clean(getattr(w, field), f"{profile}.{field}")


def test_general_and_sports_words_are_the_domain_words():
    assert profiles.words(GENERAL) == vocab.words("school")
    assert profiles.words(SPORTS) == vocab.words("sports")
    assert profiles.words(None) == vocab.words("school")
    assert profiles.words("zzz") == vocab.words("school")


def test_categories():
    assert profiles.categories(MEETING) == ("発言", "傾聴", "まとめ", "提案", "合意づくり",
                                            "時間管理")
    assert profiles.categories(PRESENTATION) == ("構成", "声と視線", "資料の使い方", "質疑応答",
                                                 "聞き手への配慮")
    assert profiles.categories(WORKSHOP) == ("取り組み", "質問", "協力", "実践")
    assert profiles.categories(LEARNING) == ("考えを伝える", "聞く", "協力", "続けて取り組む")
    # 一般 / スポーツ: the grouped tables the PDF charts use (pdf-v2 export.labels)
    assert profiles.categories(GENERAL) == ("伝える・話し合う", "手伝う・支える", "作る・組み立てる",
                                            "直す・整える", "道具を使う", "よく見る・聞く", "動く",
                                            "その他")
    assert profiles.categories(SPORTS) == ("パス", "シュート・得点", "守り", "位置どり", "声かけ",
                                           "切りかえ", "がんばり・取り返し", "その他")
    for p in ALL:
        cats = profiles.categories(p)
        assert len(cats) == len(set(cats)) >= 4
        for c in cats:
            clean(c, f"{p} category")
    assert profiles.categories(None) == profiles.categories(GENERAL)


def test_category_of_prefers_the_judgment_then_the_action_type():
    assert profiles.category_of(MEETING, ["協力", "まとめ", "発言"]) == "まとめ"
    assert profiles.category_of(MEETING, ["協力"]) is None
    assert profiles.category_of(MEETING, []) is None
    assert profiles.category_of(GENERAL, [], "hand_over") == "手伝う・支える"
    assert profiles.category_of(GENERAL, [], "observe") == "よく見る・聞く"
    assert profiles.category_of(GENERAL, [], "point_or_gesture") == "伝える・話し合う"
    assert profiles.category_of(SPORTS, [], "shot_or_score") == "シュート・得点"
    assert profiles.category_of(SPORTS, [], "assist") == "パス"
    assert profiles.category_of(SPORTS, [], "off_ball") == "位置どり"
    assert profiles.category_of(MEETING, [], "speak_or_discuss") is None   # judged only


@pytest.mark.parametrize("profile", [GENERAL, SPORTS])
def test_category_of_an_unknown_action_type_is_other(profile):
    """As pdf-v2's fallback: a type no group names is その他; no type and no tag is None."""
    assert profiles.category_of(profile, [], "nonsense") == "その他"
    assert profiles.category_of(profile, ["x"], " Other ") == "その他"
    assert profiles.category_of(profile, [], None) is None
    assert profiles.category_of(profile, [], "") is None


def test_every_engine_action_type_has_a_group():
    general = ("build_or_assemble", "hand_over", "hold_or_support", "adjust_or_fix",
               "manipulate_object", "point_or_gesture", "speak_or_discuss", "move", "observe",
               "other")
    sports = ("pass", "shot_or_score", "assist", "defense", "positioning", "off_ball",
              "transition", "communication", "effort_recovery", "other")
    assert set(profiles.GENERAL_ACTIONS) == set(general)
    assert set(profiles.SPORTS_ACTIONS) == set(sports)
    assert set(profiles.GENERAL_ACTIONS.values()) == set(profiles.categories(GENERAL))
    assert set(profiles.SPORTS_ACTIONS.values()) == set(profiles.categories(SPORTS))


@pytest.mark.parametrize("scene,activity,profile", [
    ("sport", None, SPORTS), ("sport", "meeting", SPORTS),
    ("group_work", "meeting", MEETING), ("group_work", "presentation", PRESENTATION),
    ("other", "presentation", PRESENTATION), ("group_work", "workshop", WORKSHOP),
    ("group_work", "learning", LEARNING), ("group_work", "group_task", GENERAL),
    ("group_work", "other", GENERAL), ("group_work", None, GENERAL), ("other", None, GENERAL),
    (None, None, None), (None, "meeting", None),
])
def test_for_scene(scene, activity, profile):
    assert profiles.for_scene(scene, activity) == profile


@pytest.mark.parametrize("work,profile", [
    ({"domain": "school"}, GENERAL), ({"domain": "sports"}, SPORTS), ({}, GENERAL),
    ({"domain": "school", "profile": None}, GENERAL),
    ({"domain": "school", "profile": MEETING}, MEETING),
    ({"domain": "sports", "profile": SPORTS}, SPORTS),
    ({"domain": "school", "profile": SPORTS}, GENERAL),      # never against the domain
    ({"domain": "sports", "profile": MEETING}, SPORTS),
    ({"domain": "school", "profile": "zzz"}, GENERAL),
])
def test_of_work(work, profile):
    assert profiles.of_work(work) == profile


def test_domain_of_and_check():
    assert profiles.domain_of(SPORTS) == "sports"
    assert {profiles.domain_of(p) for p in ALL if p != SPORTS} == {"school"}
    assert profiles.check(MEETING, "school") == MEETING
    assert profiles.check(None, "sports") == SPORTS
    assert profiles.check(None, "school") == GENERAL
    with pytest.raises(ValueError):
        profiles.check(SPORTS, "school")
    with pytest.raises(ValueError):
        profiles.check("zzz", "school")


def test_engine_activity_adds_the_focus_only_for_the_new_profiles():
    work = {"domain": "school", "task_text": " 紙で塔を作る \n"}
    assert profiles.engine_activity(work) == "紙で塔を作る"           # as before
    assert profiles.engine_activity({"domain": "school", "task_text": ""}) is None
    assert profiles.engine_activity({"domain": "sports", "task_text": "練習試合"}) == "練習試合"
    meeting = profiles.engine_activity({**work, "profile": MEETING})
    assert meeting.startswith("紙で塔を作る\n") and "meeting" in meeting
    alone = profiles.engine_activity({"domain": "school", "task_text": "", "profile": WORKSHOP})
    assert alone and "workshop" in alone


def test_scene_phrases():
    assert profiles.title_word(MEETING) == "会議"
    assert profiles.activity_text(PRESENTATION) == "発表・パフォーマンス"
    assert profiles.activity_text(GENERAL) == "グループ活動"
    for p in ALL:
        clean(profiles.title_word(p) + profiles.activity_text(p), p)


def test_localize_in_profile_words():
    assert vocab.localize("活動の名前", MEETING) == "会議の名前"
    assert vocab.localize("参加者ごとのフィードバック", PRESENTATION) == "発表者ごとのフィードバック"
    assert vocab.localize("グループ活動の参加者", WORKSHOP) == "グループ活動の受講者"
    assert vocab.localize("活動の名前", GENERAL) == "活動の名前"
    assert vocab.localize("評価者用", LEARNING) == "指導者用"


# --- storage ------------------------------------------------------------------------------------

def test_profile_column_is_added_to_an_old_database(tmp_path):
    import sqlite3

    from pfb_app.storage.db import Repo, connect, init_schema
    db = tmp_path / "app.db"
    old = sqlite3.connect(db)   # a database from before profiles (and before domain)
    old.executescript("""CREATE TABLE assignments (id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL, kind TEXT NOT NULL, task_text TEXT NOT NULL,
        rubric_text TEXT NOT NULL, model_key TEXT NOT NULL,
        require_identity_approval INTEGER NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO assignments (title, kind, task_text, rubric_text, model_key,
        require_identity_approval, created_at) VALUES ('a', 'video', '', '', 'standard', 1, 'x');""")
    old.commit()
    old.close()
    conn = connect(db)
    init_schema(conn)
    init_schema(conn)            # twice: guarded
    repo = Repo(conn)
    assert repo.get_assignment(1)["profile"] == GENERAL
    conn.execute("UPDATE assignments SET domain='sports' WHERE id=1")
    assert repo.get_assignment(1)["profile"] == SPORTS
    assert conn.execute("SELECT profile FROM assignments").fetchone()[0] is None   # read-time


def test_create_and_confirm_store_the_profile(tmp_path):
    from pfb_app.storage.db import Repo, connect, init_schema
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    repo = Repo(conn)
    aid = repo.create_assignment("t", "video", "", "", "standard", True)
    assert repo.get_assignment(aid)["profile"] == GENERAL
    aid = repo.create_assignment("t", "video", "", "", "standard", True, profile=MEETING)
    assert repo.get_assignment(aid)["profile"] == MEETING
    repo.update_assignment_check(aid, title="t", task_text="x", rubric_text="", domain="school",
                                 sport=None, profile=WORKSHOP)
    assert repo.get_assignment(aid)["profile"] == WORKSHOP
    repo.update_assignment_check(aid, title="t", task_text="x", rubric_text="", domain="sports",
                                 sport="soccer")
    assert repo.get_assignment(aid)["profile"] == SPORTS     # no profile given: the domain's


@pytest.mark.parametrize("profile", [PRESENTATION, WORKSHOP, MEETING, LEARNING, SPORTS])
def test_localize_never_renames_a_default_person_name(profile):
    """Error texts quote names (「参加者3」): the profile's words never rewrite them."""
    text = "参加者3 と 参加者１２ は、ほかの参加者と同じ名前です"
    out = vocab.localize(text, profile)
    assert out.startswith("参加者3 と 参加者１２ は、ほかの")
    assert out.endswith(f"ほかの{profiles.words(profile).people}と同じ名前です")


def test_default_person_names_stay_participant_n():
    for p in ALL:
        expected = "選手2" if p == SPORTS else "参加者2"
        assert vocab.default_person_name(profiles.domain_of(p), 2) == expected
