"""Roster upload / paste: header synonyms, encodings, Excel, roles, entries and apply_roster."""
import io
import zipfile

import pytest

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.services.review import BusyError
from pfb_app.services.roster import (HEADER_SYNONYMS, MAX_BYTES, ROLE_LABELS, ROLES, Roster,
                                     apply_roster, guess_roles, normalise_header,
                                     normalise_number, parse_roster_file, parse_roster_text,
                                     roster_entries, roster_text, validate_roster)
from pfb_app.ui.common import mapping_form
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


@pytest.fixture
def sid(repo):
    aid = repo.create_assignment("練習試合", Kind.video, "", "", "standard", False,
                                 domain="sports", sport="soccer")
    return repo.create_submission(aid, "前半", {"cameras": {"cam1": "/x.mp4", "cam2": "/y.mp4"}})


def roles_of(r: Roster) -> dict[str, str]:
    return {c: r.roles[c] for c in r.columns}


# --- roles and synonyms ---------------------------------------------------------------------

def test_roles_and_labels():
    assert ROLES == ("name", "number", "position", "grid", "batting", "team", "appearance",
                     "place", "ignore")
    assert list(ROLE_LABELS.values()) == ["名前", "背番号", "守備位置・ポジション", "スタート位置",
                                          "打順", "チーム", "外見", "初期位置", "使わない"]


def test_guess_roles_synonyms():
    expected = {
        "name": ["名前", "氏名", "選手名", "name"],
        "number": ["背番号", "番号", "No", "No.", "#", "ゼッケン", "カーナンバー"],
        "position": ["ポジション", "守備位置", "守備番号", "position", "pos"],
        "grid": ["スタート位置", "グリッド", "grid"],
        "batting": ["打順", "打席順", "batting order"],
        "team": ["チーム", "team"],
        "appearance": ["外見", "服装"],
        "place": ["初期位置"],
    }
    for role, headers in expected.items():
        assert list(HEADER_SYNONYMS[role]) == headers
        for h in headers:
            assert guess_roles([h]) == {h: role}, h
    # matched after NFKC, trimming and lower-casing
    for h, role in [("NAME", "name"), (" Ｎｏ． ", "number"), ("ＰＯＳ", "position"),
                    ("Batting Order", "batting"), ("＃", "number"), ("初期位置（カメラ２）", "place")]:
        assert guess_roles([h]) == {h: role}, h
    assert guess_roles(["メモ", "学年"]) == {"メモ": "ignore", "学年": "ignore"}


def test_normalise_header():
    assert normalise_header("﻿ Ｎｏ．　") == "no."
    assert normalise_header("Batting  Order") == "batting order"


def test_normalise_number():
    assert normalise_number("#07") == "7"
    assert normalise_number("１０") == "10"
    assert normalise_number("10.0") == "10"
    assert normalise_number("No.5") == "5"
    assert normalise_number("00") == "00"
    assert normalise_number("0") == "0"
    assert normalise_number(" ") == ""
    assert normalise_number("10A") == "10A"     # not a plain number: kept as written (NFKC)


# --- files ----------------------------------------------------------------------------------

def test_shift_jis_fullwidth_headers():
    text = "Ｎｏ．,氏名,ポジション\r\n１０,山田　太郎,ＦＷ\r\n7,佐藤,GK\r\n"
    r = parse_roster_file("名簿.csv", text.encode("cp932"))
    assert r.columns == ["Ｎｏ．", "氏名", "ポジション"]
    assert roles_of(r) == {"Ｎｏ．": "number", "氏名": "name", "ポジション": "position"}
    assert r.rows[0] == {"Ｎｏ．": "１０", "氏名": "山田　太郎", "ポジション": "ＦＷ"}
    # padded header, UTF-8 with a BOM (Excel's "CSV UTF-8")
    bom = "﻿背番号 ,名前\n10,山田\n".encode("utf-8")
    r2 = parse_roster_file("roster.CSV", bom)
    assert roles_of(r2) == {"背番号": "number", "名前": "name"}     # headers are trimmed
    assert r2.rows == [{"背番号": "10", "名前": "山田"}]
    # cp932 with the padded header too
    r3 = parse_roster_file("x.csv", "背番号 ,名前\n10,山田\n".encode("cp932"))
    assert roles_of(r3) == roles_of(r2)


def test_csv_title_rows_blank_rows_and_extra_columns():
    text = ("2026年 春季大会 選手名簿,,,\n"
            ",,,\n"
            "背番号,名前,メモ,\n"
            "10,山田,主将,\n"
            ",,,\n"
            "00,鈴木,,\n"
            "0,高橋,,\n")
    r = parse_roster_file("r.csv", text.encode("utf-8"))
    assert r.columns == ["背番号", "名前", "メモ"]
    assert [row["背番号"] for row in r.rows] == ["10", "00", "0"]
    entries = roster_entries(r, "sports")
    assert [(e.participant_id, e.display_name, e.attrs) for e in entries] == [
        ("r1", "山田", {"name": "山田", "number": "10"}),
        ("r2", "鈴木", {"name": "鈴木", "number": "00"}),
        ("r3", "高橋", {"name": "高橋", "number": "0"})]


def test_csv_tab_separated_and_utf16():
    data = "背番号\t名前\n10\t山田\n".encode("utf-16")
    r = parse_roster_file("r.csv", data)
    assert roles_of(r) == {"背番号": "number", "名前": "name"}
    assert r.rows == [{"背番号": "10", "名前": "山田"}]


def test_unsupported_file_type():
    with pytest.raises(ValueError, match="名簿はCSVかExcel（.xlsx）で入れてください"):
        parse_roster_file("roster.pdf", b"%PDF")
    with pytest.raises(ValueError, match="名簿はCSVかExcel"):
        parse_roster_file("roster.xls", b"\xd0\xcf\x11\xe0")


def xlsx_bytes(rows, merge: str | None = None, sheets_before: int = 0) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "名簿"
    for row in rows:
        ws.append(row)
    if merge:
        ws.merge_cells(merge)
    other = wb.create_sheet("メモ")       # a second sheet is ignored
    other.append(["名前"])
    other.append(["別人"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_roundtrip():
    data = xlsx_bytes([
        ["2026 春季リーグ 選手名簿", None, None, None],      # merged title row
        [None, None, None, None],
        ["背番号", "名前", "ポジション", "初期位置（カメラ2）"],
        [10.0, "山田", "FW", "左前"],
        [None, None, None, None],
        [7, "佐藤", "GK", None],
        ["00", "鈴木", "DF", None],
        [0, None, None, None],
        [11.5, "田中", None, None],
    ], merge="A1:D1")
    r = parse_roster_file("名簿.xlsx", data)
    assert r.columns == ["背番号", "名前", "ポジション", "初期位置（カメラ2）"]
    assert roles_of(r) == {"背番号": "number", "名前": "name", "ポジション": "position",
                           "初期位置（カメラ2）": "place"}
    assert r.place_camera == {"初期位置（カメラ2）": "cam2"}
    assert [row["背番号"] for row in r.rows] == ["10", "7", "00", "0", "11.5"]
    assert all(isinstance(v, str) for row in r.rows for v in row.values())
    entries = roster_entries(r, "sports")
    assert [e.display_name for e in entries] == ["山田", "佐藤", "鈴木", "#0", "田中"]
    assert entries[0].attrs == {"name": "山田", "number": "10", "position": "FW", "place": "左前"}


def test_xlsx_two_row_header():
    data = xlsx_bytes([
        ["選手情報", None, "守備"],
        ["名前", "背番号", "ポジション"],
        ["山田", "１０", "ＦＷ"],
    ], merge="A1:B1")
    r = parse_roster_file("r.xlsx", data)
    assert r.columns == ["名前", "背番号", "ポジション"]
    assert roster_entries(r, "sports")[0].attrs == {"name": "山田", "number": "10",
                                                    "position": "ＦＷ"}


def test_xlsx_broken_file_is_a_value_error():
    with pytest.raises(ValueError, match="読めませんでした"):
        parse_roster_file("r.xlsx", b"not a zip")


# --- paste ----------------------------------------------------------------------------------

def test_paste_with_header_tab_and_comma():
    r = parse_roster_text("名前\t背番号\n山田\t10\n\n佐藤\t7\n")
    assert roles_of(r) == {"名前": "name", "背番号": "number"}
    assert len(r.rows) == 2
    r2 = parse_roster_text("名前,背番号\n山田,10\n")
    assert r2.rows == [{"名前": "山田", "背番号": "10"}]


def test_paste_headerless_number_name():
    r = parse_roster_text("10 山田 太郎\n#7\t佐藤\n\n  ５　鈴木\n")
    assert r.columns == ["背番号", "名前"]
    assert roles_of(r) == {"背番号": "number", "名前": "name"}
    assert r.rows == [{"背番号": "10", "名前": "山田 太郎"}, {"背番号": "#7", "名前": "佐藤"},
                      {"背番号": "５", "名前": "鈴木"}]
    entries = roster_entries(r, "sports")
    assert [(e.display_name, e.attrs["number"]) for e in entries] == [
        ("山田 太郎", "10"), ("佐藤", "7"), ("鈴木", "5")]


def test_paste_names_only():
    r = parse_roster_text("山田\n佐藤\n")
    assert r.columns == ["名前"]
    assert [e.display_name for e in roster_entries(r, "school")] == ["山田", "佐藤"]


def test_number_only_row_named_hash():
    r = parse_roster_text("背番号,名前,ポジション\n10,,FW\n#07,山田,\n,,GK\n")
    entries = roster_entries(r, "sports")
    assert [(e.participant_id, e.display_name) for e in entries] == [
        ("r1", "#10"), ("r2", "山田"), ("r3", "選手3")]
    assert entries[1].attrs == {"name": "山田", "number": "7"}
    school = Roster(columns=["名前", "メモ"], rows=[{"名前": "", "メモ": "x"}, {"名前": "a", "メモ": ""}],
                    roles={"名前": "name", "メモ": "ignore"}, place_camera={})
    assert [(e.participant_id, e.display_name) for e in roster_entries(school, "school")] == [
        ("r1", "a")]         # a row with only an unused column is blank


def test_validate_needs_name_or_number():
    r = parse_roster_text("ポジション,メモ\nFW,x\n")
    assert validate_roster(r) == ["名前か背番号の列を選んでください"]
    r.roles["メモ"] = "name"
    assert validate_roster(r) == []
    r2 = parse_roster_text("名前,氏名,背番号\na,b,1\n")
    assert validate_roster(r2) == ["同じ役割の列が2つあります: 名前（名前、氏名）"]
    r2.roles["氏名"] = "ignore"
    assert validate_roster(r2) == []


def test_place_camera_header():
    r = parse_roster_text("名前,初期位置,初期位置（カメラ2）,初期位置(カメラ３)\na,左,右,中\n")
    assert all(r.roles[c] == "place" for c in r.columns[1:])
    assert r.place_camera == {"初期位置": "cam1", "初期位置（カメラ2）": "cam2",
                              "初期位置(カメラ３)": "cam3"}
    assert validate_roster(r) == []
    e = roster_entries(r, "school")[0]
    assert e.attrs == {"name": "a", "place": "左"}
    # two place columns for the same camera are a repeat
    r2 = parse_roster_text("名前,初期位置,初期位置（カメラ1）\na,左,右\n")
    assert validate_roster(r2) == ["同じ役割の列が2つあります: 初期位置（初期位置、初期位置（カメラ1））"]


# --- apply_roster ---------------------------------------------------------------------------

def test_apply_roster_replaces_subjects(repo, sid):
    repo.add_subject(sid, "s1", "古い人", None, {})
    r = parse_roster_text("背番号,名前,外見,初期位置（カメラ2）,チーム\n"
                          "10,山田,赤い帽子,左前,A\n7,,,,B\n\n,,,,\n")
    assert apply_roster(repo, sid, r, "sports") == 2
    subs = repo.list_subjects(sid)
    assert [(s["participant_id"], s["display_name"]) for s in subs] == [("r1", "山田"), ("r2", "#7")]
    assert subs[0]["appearance"] == "赤い帽子"
    assert subs[0]["initial_positions"] == {"cam2": "左前"}
    assert subs[0]["roster"] == {"name": "山田", "number": "10", "appearance": "赤い帽子",
                                 "place": "左前", "team": "A"}
    assert subs[1]["appearance"] is None and subs[1]["initial_positions"] == {}
    assert subs[1]["roster"] == {"number": "7", "team": "B"}
    # a second roster replaces the first
    assert apply_roster(repo, sid, parse_roster_text("名前\nx\n"), "sports") == 1
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["x"]


def test_apply_roster_rejects_invalid(repo, sid):
    with pytest.raises(ValueError, match="名前か背番号"):
        apply_roster(repo, sid, parse_roster_text("メモ,備考\nx,y\n"), "sports")


def test_apply_roster_busy_after_confirm(repo, sid):
    r = parse_roster_text("名前\nx\n")
    repo.set_identity_confirmed(sid, True)
    with pytest.raises(BusyError):
        apply_roster(repo, sid, r, "sports")
    repo.set_identity_confirmed(sid, False)
    subject = repo.add_subject(sid, "s1", "a", None, {})
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subject, version=0,
        source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"text": "x"}, status="ok", error=None, warnings=[], duration_sec=0.0,
        created_at=""))
    with pytest.raises(BusyError):
        apply_roster(repo, sid, r, "sports")
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["a"]


def test_grid_and_batting_values_are_numbers():
    r = parse_roster_text("名前,グリッド,打順\na,１,1.0\n")
    assert roster_entries(r, "sports")[0].attrs == {"name": "a", "grid": "1", "batting": "1"}


# --- review round 1 -------------------------------------------------------------------------

def test_headerless_delimited_rows():
    r = parse_roster_text("10\t山田\tFW\n")
    assert r.columns == ["背番号", "名前", "列3"]
    assert r.rows == [{"背番号": "10", "名前": "山田", "列3": "FW"}]
    r2 = parse_roster_text("10,山田,FW\n7,佐藤,GK\n")
    assert r2.columns == ["背番号", "名前", "列3"]
    assert [e.display_name for e in roster_entries(r2, "sports")] == ["山田", "佐藤"]
    r3 = parse_roster_file("r.csv", "10,山田,FW\r\n7,佐藤,GK\r\n".encode("cp932"))
    assert r3.columns == ["背番号", "名前", "列3"] and len(r3.rows) == 2


def test_space_separated_header_line():
    r = parse_roster_text("背番号 ポジション 名前\n10 FW 山田 太郎\n7 GK 佐藤\n")
    assert r.columns == ["背番号", "ポジション", "名前"]
    assert r.rows[0] == {"背番号": "10", "ポジション": "FW", "名前": "山田 太郎"}


def test_delimiter_counted_over_lines_and_txt_tsv():
    data = "選手名簿\n背番号\t名前\n10\t山田\n".encode("utf-8")
    r = parse_roster_file("r.csv", data)
    assert r.columns == ["背番号", "名前"] and r.rows == [{"背番号": "10", "名前": "山田"}]
    assert parse_roster_file("r.tsv", data).rows == r.rows
    t = parse_roster_file("r.TXT", "10 山田\n7 佐藤\n".encode("cp932"))
    assert t.columns == ["背番号", "名前"] and len(t.rows) == 2


def test_size_limits():
    with pytest.raises(ValueError, match="大きすぎます"):
        parse_roster_file("r.csv", b"a" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="大きすぎます"):
        parse_roster_file("r.xlsx", b"a" * (MAX_BYTES + 1))
    with pytest.raises(ValueError, match="大きすぎます"):
        parse_roster_text("a" * (MAX_BYTES + 1))
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/big.bin", b"\0" * (51 * 1024 * 1024))
    assert len(bomb.getvalue()) < MAX_BYTES
    with pytest.raises(ValueError, match="Excelの名簿が大きすぎます"):
        parse_roster_file("r.xlsx", bomb.getvalue())
    rows = "名前\n" + "".join(f"人{i}\n" for i in range(5001))
    with pytest.raises(ValueError, match="5000人まで"):
        parse_roster_text(rows)
    assert len(parse_roster_text("名前\n" + "".join(f"人{i}\n" for i in range(5000))).rows) == 5000


def test_xlsx_columns_capped():
    data = xlsx_bytes([["名前"] + [f"c{i}" for i in range(2, 71)], ["山田"] + ["x"] * 69])
    r = parse_roster_file("r.xlsx", data)
    assert len(r.columns) == 64 and r.columns[-1] == "c64"


def test_xlsx_rows_capped():
    data = xlsx_bytes([["名前"]] + [[f"人{i}"] for i in range(5001)])
    with pytest.raises(ValueError, match="5000人まで"):
        parse_roster_file("r.xlsx", data)


def test_csv_error_is_friendly():
    with pytest.raises(ValueError, match="名簿を読めませんでした"):
        parse_roster_text("名前\n" + "a" * 200_000 + "\n")


def test_colliding_display_names():
    r = parse_roster_text("背番号,名前,チーム\n10,,チームA\n10,,チームB\n7,山田,チームA\n")
    assert [e.display_name for e in roster_entries(r, "sports")] == [
        "#10（チームA）", "#10（チームB）", "山田"]
    r2 = parse_roster_text("背番号,名前\n10,山田\n7,山田\n")
    entries = roster_entries(r2, "sports")
    assert [e.display_name for e in entries] == ["山田（#10）", "山田（#7）"]
    assert [e.attrs["name"] for e in entries] == ["山田", "山田"]
    r3 = parse_roster_text("名前,メモ\n山田,a\n佐藤,b\n山田,c\n山田,d\n")
    assert [e.display_name for e in roster_entries(r3, "school")] == [
        "山田", "佐藤", "山田（2）", "山田（3）"]


def test_apply_roster_busy_with_active_jobs(repo, sid):
    repo.enqueue(sid, Stage.track, "cpu", camera="cam1")
    with pytest.raises(BusyError, match="解析が進んでいるため"):
        apply_roster(repo, sid, parse_roster_text("名前\nx\n"), "sports")


def test_apply_roster_during_content_check(repo, sid):
    repo.enqueue(sid, Stage.detect, "cpu", camera="cam1")
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam2")
    assert apply_roster(repo, sid, parse_roster_text("名前\nx\n"), "sports") == 1
    # a names job never blocks a roster change (its run would be stale; Task 9 review)
    repo.enqueue(sid, Stage.names, "cpu", camera="cam1")
    assert apply_roster(repo, sid, parse_roster_text("名前\ny\n"), "sports") == 1
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    with pytest.raises(BusyError, match="解析が進んでいるため"):
        apply_roster(repo, sid, parse_roster_text("名前\nz\n"), "sports")
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["y"]


def test_apply_roster_empty(repo, sid):
    with pytest.raises(ValueError, match="選手名簿に人が見つかりませんでした"):
        apply_roster(repo, sid, parse_roster_text("名前,背番号\n"), "sports")


def _run(repo, sid, stage, output, camera="cam1", upstream=(), source="model"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def test_apply_roster_makes_saved_mapping_stale(repo, sid):
    roster = parse_roster_text("背番号,名前\n10,山田\n7,佐藤\n")
    apply_roster(repo, sid, roster, "sports")
    track = _run(repo, sid, Stage.track, {"labels": ["P1", "P2"], "default_names": {"P1": "r2"}})
    _run(repo, sid, Stage.identity, {
        "participants": [{"label": "P1", "participant_id": "r1", "display_name": None},
                         {"label": "P2", "participant_id": "r2", "display_name": None}],
        "labels": ["P1", "P2"], "track_run_id": track.id}, upstream=[track.id], source="teacher")
    assert mapping_form(repo, sid, "cam1").saved == {"P1": "r1", "P2": "r2"}
    # the same roster again changes nothing
    assert apply_roster(repo, sid, roster, "sports") == 2
    assert mapping_form(repo, sid, "cam1").saved == {"P1": "r1", "P2": "r2"}
    # a reordered roster: r1/r2 now mean other people, so the saved mapping is not current
    apply_roster(repo, sid, parse_roster_text("背番号,名前\n7,佐藤\n10,山田\n"), "sports")
    form = mapping_form(repo, sid, "cam1")
    assert form.saved == {} and not form.current
    assert form.prefill == {"P1": "r2"}                  # the rule's defaults
    # a mapping saved after the roster counts again
    _run(repo, sid, Stage.identity, {
        "participants": [{"label": "P1", "participant_id": "r2", "display_name": None},
                         {"label": "P2", "participant_id": "r1", "display_name": None}],
        "labels": ["P1", "P2"], "track_run_id": track.id}, upstream=[track.id], source="teacher")
    assert mapping_form(repo, sid, "cam1").saved == {"P1": "r2", "P2": "r1"}


# --- a later group's roster pre-fill (final review item 5) -----------------------------------

def test_roster_text_round_trips_the_entries(repo, sid):
    text = ("背番号,名前,ポジション,外見,初期位置（カメラ1）\n10,山田,FW,\"白, 長袖\",左\n"
            "7,,DF,,\n,佐藤,,,右\n")
    apply_roster(repo, sid, parse_roster_text(text), "sports")
    again = roster_text(repo.list_subjects(sid))
    assert again.splitlines()[0] == "名前\t背番号\tポジション\t外見"   # no places: per video
    entries = roster_entries(parse_roster_text(again), "sports")
    assert [(e.display_name, e.attrs) for e in entries] == [
        ("山田", {"name": "山田", "number": "10", "position": "FW", "appearance": "白、 長袖"}),
        ("#7", {"number": "7", "position": "DF"}),
        ("佐藤", {"name": "佐藤"})]
    assert roster_text([]) == ""
