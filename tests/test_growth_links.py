"""services.growth: who may be suggested as the same person across works, and what accepting,
declining, undoing and linking by hand store. Suggestions never link anything by themselves."""
import pytest

from growth_synthetic import add_work
from pfb_app.config import Settings
from pfb_app.services import growth
from pfb_app.services.review import ADDED, NAME_SOURCE
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox"))


def pairs(repo, **kw) -> set[tuple[int, int]]:
    return {(s.subject_id, s.other_id) for s in growth.suggestions(repo, **kw)}


def test_the_same_name_in_two_works_is_a_suggestion_not_a_link(repo):
    _, _, one = add_work(repo, "定例会1", {"山田": ["発言"], "鈴木": ["傾聴"]}, day="2026-09-01")
    _, _, two = add_work(repo, "定例会2", {"山田": ["発言"], "佐藤": ["まとめ"]}, day="2026-09-08")
    found = growth.suggestions(repo)
    assert [(s.subject_id, s.other_id) for s in found] == [(two["山田"], one["山田"])]
    assert found[0].reason == "同じ名前（山田）"
    assert repo.growth_links() == []                      # nothing is linked by itself


def test_names_are_compared_after_width_space_and_case(repo):
    _, _, one = add_work(repo, "A", {"Ｙａｍａｄａ  Hanako": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"yamada hanako": ["発言"]}, day="2026-09-08")
    assert pairs(repo) == {(two["yamada hanako"], one["Ｙａｍａｄａ  Hanako"])}


def test_anonymous_labels_are_never_suggested(repo):
    people = {"参加者1": ["発言"], "選手2": ["発言"], "相手選手1": ["発言"], "選手（番号なし）1": ["発言"]}
    add_work(repo, "A", people, day="2026-09-01")
    add_work(repo, "B", people, day="2026-09-08")
    assert pairs(repo) == set()


def test_a_number_needs_the_same_team_or_group_on_both_works(repo):
    _, _, one = add_work(repo, "練習1", {"#5": ["パス"]}, profile="sports", day="2026-09-01",
                         group="U12 A")
    _, _, two = add_work(repo, "練習2", {"#5": ["パス"]}, profile="sports", day="2026-09-08",
                         group="u12  a")
    _, _, other = add_work(repo, "練習3", {"#5": ["パス"]}, profile="sports", day="2026-09-15",
                           group="U12 B")
    _, _, none = add_work(repo, "練習4", {"#5": ["パス"]}, profile="sports", day="2026-09-22")
    found = growth.suggestions(repo)
    assert [(s.subject_id, s.other_id) for s in found] == [(two["#5"], one["#5"])]
    assert found[0].reason == "同じ番号（#5・チーム・グループ「U12 A」）"


def test_a_number_with_a_note_is_never_suggested(repo):
    add_work(repo, "練習1", {"#5（チーム1）": ["パス"]}, profile="sports", day="2026-09-01", group="G")
    add_work(repo, "練習2", {"#5（チーム1）": ["パス"]}, profile="sports", day="2026-09-08", group="G")
    assert pairs(repo) == set()


def test_opponents_are_never_suggested(repo):
    meta = {ADDED: {"P1": "opp"}}
    add_work(repo, "練習1", {"#5": ["パス"]}, profile="sports", group="G", meta=meta)
    add_work(repo, "練習2", {"#5": ["パス"]}, profile="sports", group="G", meta=meta)
    assert pairs(repo) == set()


def test_a_name_the_coach_typed_counts_as_a_name(repo):
    meta = {NAME_SOURCE: {"P1": "coach"}}
    _, _, one = add_work(repo, "A", {"ハナ": ["発言"]}, meta=meta, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"ハナ": ["発言"]}, meta=meta, day="2026-09-08")
    assert pairs(repo) == {(two["ハナ"], one["ハナ"])}


def test_two_groups_of_one_work_are_never_suggested_or_linked(repo):
    """Review HIGH2: one work counts once; its groups are never joined as one person."""
    aid, _, first = add_work(repo, "大会", {"山田": ["発言"]}, label="前半")
    _, _, second = add_work(repo, "", {"山田": ["発言"]}, label="後半", aid=aid)
    assert pairs(repo) == set()
    person = growth.link_manual(repo, first["山田"], None)
    with pytest.raises(ValueError, match="同じ作業"):
        growth.link_manual(repo, second["山田"], person)
    assert repo.growth_person_of(second["山田"]) is None


def test_people_without_an_approved_text_are_not_suggested(repo):
    """Review HIGH1: only people with an approved text (they are never re-made by 自動で選び直す
    or a roster change, so their links last)."""
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08", approve=False)
    assert pairs(repo) == set()
    assert {m.subject_id: m.approved for m in growth.members(repo)} == {
        one["山田"]: True, two["山田"]: False}


def test_linking_needs_an_approved_text(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08", approve=False)
    with pytest.raises(ValueError, match=growth.APPROVED_ONLY):
        growth.accept(repo, two["山田"], one["山田"])
    with pytest.raises(ValueError, match=growth.APPROVED_ONLY):
        growth.link_manual(repo, two["山田"], None)
    person = growth.link_manual(repo, one["山田"], None)
    with pytest.raises(ValueError, match=growth.APPROVED_ONLY):
        growth.link_manual(repo, two["山田"], person)
    assert repo.growth_links() == [{**repo.growth_links()[0], "subject_id": one["山田"]}]


def test_an_approval_lost_later_keeps_the_link(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    person = growth.link_manual(repo, one["山田"], None)
    repo.set_approval(one["山田"], None)
    assert repo.growth_person_of(one["山田"]) == person


@pytest.mark.parametrize("label", ["5", "5番", "背番号5", "No.5", "no5", "＃5", "#5"])
def test_every_bare_number_follows_the_team_group_rule(repo, label):
    """Review MED7: a number the coach typed in any of these forms is a number, not a name."""
    add_work(repo, "練習1", {label: ["パス"]}, profile="sports", day="2026-09-01")
    add_work(repo, "練習2", {label: ["パス"]}, profile="sports", day="2026-09-08")
    assert pairs(repo) == set()                     # no team/group: never
    _, _, a = add_work(repo, "練習3", {label: ["パス"]}, profile="sports", day="2026-09-15", group="G")
    _, _, b = add_work(repo, "練習4", {"#5": ["パス"]}, profile="sports", day="2026-09-22", group="G")
    assert pairs(repo) == {(b["#5"], a[label])}     # the same number, whatever its form


def test_accepting_two_people_of_different_records_is_refused(repo):
    """Review LOW11: never a silent no-op that the screen calls a link."""
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    p1 = growth.link_manual(repo, one["山田"], None)
    p2 = growth.link_manual(repo, two["山田"], None)
    with pytest.raises(ValueError, match="別の記録"):
        growth.accept(repo, two["山田"], one["山田"])
    assert (repo.growth_person_of(one["山田"]), repo.growth_person_of(two["山田"])) == (p1, p2)


def test_accept_makes_a_record_and_later_works_join_it(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    _, _, three = add_work(repo, "C", {"山田": ["発言"]}, day="2026-09-15")
    first = growth.suggestions(repo)
    assert {(s.subject_id, s.other_id) for s in first} == {
        (two["山田"], one["山田"]), (three["山田"], one["山田"])}
    done = growth.accept(repo, two["山田"], one["山田"])
    person = repo.growth_person_of(one["山田"])
    assert person is not None and repo.growth_person_of(two["山田"]) == person
    assert repo.growth_person(person)["name"] == "山田"
    assert set(done.linked) == {one["山田"], two["山田"]}
    assert {r["how"] for r in repo.growth_links(person)} == {"suggested"}
    nxt = growth.suggestions(repo)
    assert [(s.subject_id, s.person_id) for s in nxt] == [(three["山田"], person)]
    growth.accept(repo, three["山田"], nxt[0].other_id)
    assert len(repo.growth_links(person)) == 3
    assert growth.suggestions(repo) == []


def test_undo_takes_back_only_what_the_accept_linked(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    _, _, three = add_work(repo, "C", {"山田": ["発言"]}, day="2026-09-15")
    growth.accept(repo, two["山田"], one["山田"])
    person = repo.growth_person_of(one["山田"])
    done = growth.accept(repo, three["山田"], one["山田"])
    assert done.linked == [three["山田"]]
    growth.undo(repo, done)
    assert repo.growth_person_of(three["山田"]) is None
    assert repo.growth_person_of(two["山田"]) == person
    assert [(s.subject_id, s.person_id) for s in growth.suggestions(repo)] == [
        (three["山田"], person)]                           # offered again after an undo


def test_declined_pairs_are_not_offered_again(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    growth.decline(repo, two["山田"], one["山田"])
    assert pairs(repo) == set()


def test_declined_with_a_member_of_the_record_is_declined_with_the_record(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    _, _, three = add_work(repo, "C", {"山田": ["発言"]}, day="2026-09-15")
    growth.decline(repo, three["山田"], one["山田"])
    growth.accept(repo, two["山田"], one["山田"])
    assert pairs(repo) == set()


def test_two_records_with_one_name_get_no_suggestion(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    _, _, three = add_work(repo, "C", {"山田": ["発言"]}, day="2026-09-15")
    growth.link_manual(repo, one["山田"], None)
    growth.link_manual(repo, two["山田"], None)        # a second, separate record
    assert pairs(repo) == set()
    assert repo.growth_person_of(three["山田"]) is None


def test_manual_link_joins_anonymous_people_too(repo):
    _, _, one = add_work(repo, "A", {"参加者1": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"参加者3": ["発言"]}, day="2026-09-08")
    person = growth.link_manual(repo, one["参加者1"], None)
    assert growth.link_manual(repo, two["参加者3"], person) == person
    assert {r["how"] for r in repo.growth_links(person)} == {"manual"}
    growth.unlink(repo, two["参加者3"])
    assert repo.growth_person_of(two["参加者3"]) is None


def test_suggestions_for_one_person_or_one_subject(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"], "鈴木": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"], "鈴木": ["発言"]}, day="2026-09-08")
    growth.accept(repo, two["山田"], one["山田"])
    person = repo.growth_person_of(one["山田"])
    assert pairs(repo, person_id=person) == set()
    assert pairs(repo, subject_id=one["鈴木"]) == {(two["鈴木"], one["鈴木"])}
    assert pairs(repo, subject_id=two["鈴木"]) == {(two["鈴木"], one["鈴木"])}


def test_people_not_evaluated_are_left_out(repo):
    _, _, one = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    _, _, two = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08")
    repo.set_excluded(two["山田"], True)
    assert pairs(repo) == set()
    assert two["山田"] not in {m.subject_id for m in growth.members(repo)}


def test_the_group_setting_is_cleaned(repo):
    aid, _, _ = add_work(repo, "A", {"山田": ["発言"]})
    growth.set_group(repo, aid, "  U12 A  ")
    assert repo.get_assignment(aid)["growth_group"] == "U12 A"
    with pytest.raises(ValueError):
        growth.set_group(repo, aid, "x" * (growth.MAX_GROUP + 1))


def test_an_outdated_suggestion_cannot_put_a_second_person_of_one_work_in_a_record(repo):
    """Re-review (b): both groups of work B were offered to the record; after one is accepted,
    the other (still on screen) is refused."""
    _, _, a = add_work(repo, "A", {"山田": ["発言"]}, day="2026-09-01")
    bid, _, b1 = add_work(repo, "B", {"山田": ["発言"]}, day="2026-09-08", label="1組")
    _, _, b2 = add_work(repo, "", {"山田": ["発言"]}, label="2組", aid=bid)
    person = growth.link_manual(repo, a["山田"], None)
    assert {(s.subject_id, s.person_id) for s in growth.suggestions(repo)} == {
        (b1["山田"], person), (b2["山田"], person)}
    growth.accept(repo, b1["山田"], a["山田"])
    with pytest.raises(ValueError, match=growth.SAME_WORK):
        growth.accept(repo, b2["山田"], a["山田"])
    assert repo.growth_person_of(b2["山田"]) is None
