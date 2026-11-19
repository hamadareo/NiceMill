"""export.growth: one person's record across works, from approved texts only, and the plain
statements it supports (exact numbers; never progress the records do not show)."""
import pytest

from growth_synthetic import add_work, text_of
from pfb_app.config import Settings
from pfb_app.export import growth as record
from pfb_app.services import growth
from pfb_app.storage.db import Repo, connect, init_schema

NOTE = "場面の数は、動画の長さや映り方でも変わります。"


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox"))


def linked(repo, *subject_ids) -> int:
    person = growth.link_manual(repo, subject_ids[0], None)
    for sid in subject_ids[1:]:
        growth.link_manual(repo, sid, person)
    return person


def test_entries_in_date_order_with_counts_and_quotes(repo):
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言", "傾聴", "傾聴"]}, day="2026-09-08")
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    rec = record.for_person(repo, linked(repo, b["山田"], a["山田"]))
    assert rec.name == "山田"
    assert [e.title for e in rec.entries] == ["定例会1", "定例会2"]
    assert [e.when for e in rec.entries] == ["2026年9月1日", "2026年9月8日"]
    assert rec.entries[0].activity == "話し合い・会議"
    assert rec.entries[1].counts == {"発言": 1, "傾聴": 2}
    assert [p.label for p in rec.points] == ["発言", "傾聴"]          # the profile's order
    assert rec.entries[0].quotes == [
        "山田さんは「定例会1」で、まわりの人の話をよく聞いていました。", "最後まで声をかけ続けたのもよかったです。"]
    assert all(q in text_of("山田", "定例会1") for q in rec.entries[0].quotes)


def test_only_approved_texts_count(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言", "発言"]}, day="2026-09-08")
    person = linked(repo, a["山田"], b["山田"])
    repo.set_approval(b["山田"], None)             # its approval was taken back after linking
    rec = record.for_person(repo, person)
    waiting = rec.entries[1]
    assert not waiting.approved and waiting.counts == {} and waiting.quotes == []
    assert len(rec.approved) == 1
    assert record.statements(rec)[0] == record.ONE_WORK


def test_no_approved_work(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, approve=False)
    rec = record.for_subject(repo, a["山田"])
    assert rec.person_id is None and len(rec.entries) == 1
    assert record.statements(rec) == [record.NO_WORK]


def test_one_work_says_so_plainly(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]})
    rec = record.for_subject(repo, a["山田"])
    assert record.statements(rec) == [
        "承認した文章のある作業は1件だけです。変化は、2件目の記録ができてからわかります。"]


def test_statements_state_only_what_the_counts_show(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言", "傾聴", "傾聴"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["傾聴"]}, day="2026-09-08")
    _, _, c = add_work(repo, "定例会3", {"山田": ["発言", "発言", "傾聴", "まとめ"]}, day="2026-09-15")
    rec = record.for_person(repo, linked(repo, a["山田"], b["山田"], c["山田"]))
    assert record.statements(rec) == [
        "作業3件の記録があります（2026年9月1日〜2026年9月15日）。",
        "いちばん多く記録された観点は「傾聴」です（作業3件で合計4回）。",
        "「発言」のよかった場面は、最初の作業（2026年9月1日）では1回、いちばん新しい作業（2026年9月15日）では2回でした。",
        "いちばん新しい作業で、はじめて「まとめ」の場面が記録されました。",
        NOTE]


def test_no_change_words_when_nothing_went_up(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言", "発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言"]}, day="2026-09-08")
    lines = record.statements(record.for_person(repo, linked(repo, a["山田"], b["山田"])))
    assert lines == ["作業2件の記録があります（2026年9月1日〜2026年9月8日）。",
                     "いちばん多く記録された観点は「発言」です（作業2件で合計3回）。", NOTE]
    text = "".join(lines)
    for word in ("成長", "伸び", "上達", "上手", "増え", "減"):
        assert word not in text


def test_up_to_leaves_out_later_works(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    aid2, _, b = add_work(repo, "定例会2", {"山田": ["発言"]}, day="2026-09-08")
    _, _, c = add_work(repo, "定例会3", {"山田": ["発言"]}, day="2026-09-15")
    person = linked(repo, a["山田"], b["山田"], c["山田"])
    rec = record.for_person(repo, person, up_to=aid2)
    assert [e.title for e in rec.entries] == ["定例会1", "定例会2"]


def test_an_unlinked_subject_is_its_own_record_and_a_linked_one_its_person(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言"]}, day="2026-09-08")
    assert [e.subject_id for e in record.for_subject(repo, b["山田"]).entries] == [b["山田"]]
    person = linked(repo, a["山田"], b["山田"])
    rec = record.for_subject(repo, b["山田"])
    assert rec.person_id == person and len(rec.entries) == 2


def test_grid_keeps_the_latest_works_and_every_point(repo):
    ids = []
    for k in range(1, 11):
        _, _, s = add_work(repo, f"会{k}", {"山田": ["発言"] * k}, day=f"2026-09-{k:02d}")
        ids.append(s["山田"])
    rec = record.for_person(repo, linked(repo, *ids))
    columns, rows = record.grid(rec)
    assert len(columns) == record.GRID_WORKS == 8
    assert columns[0] == "9/3" and columns[-1] == "9/10"
    assert [(cat.label, values) for cat, values in rows] == [("発言", list(range(3, 11)))]


def test_a_tie_for_the_point_recorded_most_names_both(repo):
    """Review LOW8."""
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言", "傾聴"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言", "傾聴"]}, day="2026-09-08")
    lines = record.statements(record.for_person(repo, linked(repo, a["山田"], b["山田"])))
    assert lines[1] == "いちばん多く記録された観点は「発言」と「傾聴」です（どちらも合計2回）。"


def test_a_point_another_activity_does_not_have_is_not_a_zero(repo):
    """Review MED5: a meeting has no 「パス」; it is shown as not applicable, never as 0, and no
    sentence compares it."""
    _, _, a = add_work(repo, "定例会", {"山田": ["発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "練習", {"山田": ["パス", "パス"]}, profile="sports", day="2026-09-08")
    rec = record.for_person(repo, linked(repo, a["山田"], b["山田"]))
    columns, rows = record.grid(rec)
    assert dict((cat.label, values) for cat, values in rows) == {"発言": [1, None],
                                                                   "パス": [None, 2]}
    text = "".join(record.statements(rec))
    assert "0回" not in text and "はじめて「パス」" not in text


def test_two_groups_of_one_work_count_as_one_work(repo):
    """Review HIGH2: links made before the rule (or by hand in the DB) still count works."""
    aid, _, first = add_work(repo, "大会", {"山田": ["発言"]}, label="前半", day="2026-09-01")
    _, _, second = add_work(repo, "", {"山田": ["傾聴"]}, label="後半", aid=aid)
    person = repo.create_growth_person("山田")
    repo.link_growth(first["山田"], person, "manual")
    repo.link_growth(second["山田"], person, "manual")
    rec = record.for_person(repo, person)
    assert len(rec.works) == 1 and rec.works[0].counts == {"発言": 1, "傾聴": 1}
    assert record.statements(rec) == [record.ONE_WORK]
    assert record.handout_records(repo, aid, [first["山田"], second["山田"]]) == {}


def test_counters_are_never_written_with_tsu():
    """Review HIGH3: 「作業12件」, never 「12つ」 (every string of the growth code)."""
    import re
    from pathlib import Path
    src = Path(record.__file__).resolve().parents[1]
    files = [src / "export" / "growth.py", src / "export" / "views.py",
             src / "services" / "growth.py", src / "ui" / "wizard" / "growth_view.py",
             src / "ui" / "wizard" / "step_pdf.py",
             src / "export" / "templates" / "_blocks.html.j2"]
    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    guide = (src.parents[1] / "docs" / "user_guide.md").read_text(encoding="utf-8")
    start = guide.index("## 成長の記録")       # this lane's section, and its line on step 6
    texts["user_guide.md"] = guide[start:guide.index("\n## ", start + 1)] + "".join(
        line for line in guide.splitlines() if "成長のまとめ" in line)
    for name, text in texts.items():
        found = re.findall(r"[0-9０-９}]つ", text)
        assert not found, (name, found)


@pytest.mark.parametrize("text,expected", [
    ("「やった！」と言えました。次も続けます。", ["「やった！」と言えました。", "次も続けます。"]),
    ("よく見ていました！（とても）すごい。", ["よく見ていました！", "（とても）すごい。"]),
    ("一文だけ", ["一文だけ"]),
])
def test_quotes_never_cut_inside_brackets(text, expected):
    """Review LOW12."""
    assert record.sentences(text) == expected


def test_two_works_on_one_day_get_columns_told_apart(repo):
    """Review LOW10."""
    _, _, a = add_work(repo, "午前", {"山田": ["発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "午後", {"山田": ["発言"]}, day="2026-09-01")
    columns, _ = record.grid(record.for_person(repo, linked(repo, a["山田"], b["山田"])))
    assert columns == ["9/1①", "9/1②"]
