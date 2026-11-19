"""試合・活動のまとめ (services.dashboard): the numbers of one group, on synthetic works only
(tests/pdf_synthetic.py). Every number must be one the handouts already stand on."""
import csv
import io
import json

import pytest

from pfb_app.export.sheets import build_report
from pfb_app.services import dashboard, erasure, retention
from pfb_app.storage import files
from pdf_synthetic import video_group, work
from test_wizard_ui_basic import env  # noqa: F401 (env)


def board_of(repo, aid, sid=None):
    return dashboard.build(repo, aid, sid)


# --- the model ------------------------------------------------------------------------------

def test_people_counts_and_minutes_are_the_handout_numbers(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60, teams=True)
    board = board_of(repo, aid)
    sheets = {s.subject_id: s for s in build_report(repo, aid, with_pictures=False).sheets}
    assert board.submission_id == sid and board.sports and board.court
    assert len(board.people) == 3
    for p in board.people:
        s = sheets[p.subject_id]
        assert p.total == s.counted == 4
        assert sum(p.counts.values()) + p.untyped == p.total
        assert p.on_screen_s == pytest.approx(s.on_screen_s)
        assert p.distance_m == pytest.approx(s.distance_m)
        assert p.team in ("チーム1", "チーム2")
        assert p.status == "承認済み" and p.text
    assert len(board.moments) == 12
    assert board.duration_s == pytest.approx(60)


def test_distance_names_its_one_camera_and_the_position_error(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=1, seconds=60, cameras=("cam1", "cam2"))
    g = (files.submission_dir(settings, aid, sid) / "pfb3x3" / "tracks" / f"s{sid}_cam1"
         / "linecall" / "geometry.json")
    doc = json.loads(g.read_text(encoding="utf-8"))
    doc["h_unc_cm"] = 42.0
    g.write_text(json.dumps(doc), encoding="utf-8")
    person = board_of(repo, aid).people[0]
    one = build_report(repo, aid, with_pictures=False).sheets[0]
    assert person.distance_m == pytest.approx(one.distance_m)      # never the two summed
    assert person.court_camera == "cam1" and person.position_error_cm == pytest.approx(42.0)


def test_no_geometry_no_distance_but_a_frame_heatmap(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, geometry=False)
    board = board_of(repo, aid)
    assert not board.court
    for p in board.people:
        assert p.distance_m is None and p.court_points == []
        assert p.frame_points["cam1"]                 # image px of the foot point
        assert all(0 <= x <= 640 and 0 <= y <= 360 for x, y in p.frame_points["cam1"])
    assert board.frame_sizes["cam1"] == (640, 360)


def test_a_general_work_has_no_court_and_no_teams(env):
    settings, repo = env
    aid = work(repo, domain="school", title="塔づくり")
    video_group(settings, repo, aid, people=2, seconds=60)
    board = board_of(repo, aid)
    assert not board.sports and not board.court
    assert all(p.distance_m is None and p.team is None for p in board.people)


def test_erased_and_left_out_people_never_appear(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60, teams=True)
    subjects = repo.list_subjects(sid)
    erasure.erase(repo, settings, subjects[0]["id"])
    repo.set_excluded(subjects[1]["id"], True)
    board = board_of(repo, aid)
    assert [p.subject_id for p in board.people] == [subjects[2]["id"]]
    assert {m.subject_id for m in board.moments} == {subjects[2]["id"]}
    assert sum(t.people for t in dashboard.team_totals(board)) == 1


def test_a_waiting_person_is_shown_with_their_status(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, unapproved=("Pb",))
    board = board_of(repo, aid)
    waiting = [p for p in board.people if not p.approved]
    assert len(waiting) == 1 and waiting[0].status == "確認待ち" and waiting[0].text is None


def test_removed_media_keep_the_moments_but_no_minutes_or_places(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60)
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-10-24T00:00:00+00:00'"
                      " WHERE id=?", (aid,))
    retention._finish(repo, settings, aid)
    for v in settings.inbox_dir.glob("*.mp4"):
        v.unlink()
    board = board_of(repo, aid)
    assert board.media_removed
    assert all(p.on_screen_s is None and p.court_points == [] and p.distance_m is None
               and not any(p.frame_points.values()) for p in board.people)
    assert len(board.moments) == 8 and all(m.category for m in board.moments)


def test_groups_are_shown_one_at_a_time(env):
    settings, repo = env
    aid = work(repo)
    a = video_group(settings, repo, aid, people=2, seconds=40, label="前半")
    b = video_group(settings, repo, aid, people=3, seconds=40, label="後半")
    assert [g[0] for g in dashboard.groups(repo, aid)] == [a, b]
    assert len(board_of(repo, aid, b).people) == 3 and board_of(repo, aid).submission_id == a


# --- aggregates -----------------------------------------------------------------------------

def test_team_totals_add_up_the_evaluated_people_of_each_team(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    board = board_of(repo, aid)
    teams = dashboard.team_totals(board)
    assert [t.name for t in teams] == ["チーム1", "チーム2"]
    assert [t.people for t in teams] == [2, 2]
    assert sum(t.total for t in teams) == board.total == 16
    for t in teams:
        members = [p for p in board.people if p.team == t.name]
        for key, n in t.counts.items():
            assert n == sum(p.counts.get(key, 0) for p in members)


def test_a_general_group_is_one_total(env):
    settings, repo = env
    aid = work(repo, domain="school", title="塔づくり")
    video_group(settings, repo, aid, people=3, seconds=60, label="班1")
    teams = dashboard.team_totals(board_of(repo, aid))
    assert [(t.name, t.people, t.total) for t in teams] == [("班1の全員", 3, 12)]


def test_highlights_are_cited_moments_at_most_two_each_in_time_order(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=6, seconds=90)
    board = board_of(repo, aid)
    hl = dashboard.highlights(board)
    assert 0 < len(hl) <= 10 and all(m.cited for m in hl)
    per = {}
    for m in hl:
        per[m.subject_id] = per.get(m.subject_id, 0) + 1
    assert max(per.values()) <= 2
    assert [m.t_s for m in hl] == sorted(m.t_s for m in hl)
    assert all(m.cite_rank < 2 for m in hl)        # the moments each text cites first


def test_scenes_say_who_did_what_in_each_time_bin(env):
    settings, repo = env
    aid = work(repo, domain="school", title="塔づくり")
    video_group(settings, repo, aid, people=2, seconds=120)
    board = board_of(repo, aid)
    found = dashboard.scenes(board)
    size = dashboard.bin_size(board.duration_s)
    assert sum(s.moments for s in found) == len(board.moments) == 8
    for s in found:
        assert s.end_s - s.start_s <= size and s.start_s <= s.first_t < s.start_s + size
        assert "〜" in s.label
        names = [name for name, _ in s.who]
        assert names and all(n in {p.name for p in board.people} for n in names)
        assert all(n > 0 and c is not None for _, cats in s.who for c, n in cats)


# --- CSV ------------------------------------------------------------------------------------

def rows_of(data: bytes) -> list[list[str]]:
    assert data.startswith(b"\xef\xbb\xbf")           # Excel reads UTF-8 with the BOM
    return list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))


def test_people_csv_has_the_numbers_and_says_what_is_missing(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, teams=True)
    board = board_of(repo, aid)
    rows = rows_of(dashboard.people_csv(board))
    head = rows[0]
    assert head[:5] == ["名前", "チーム", "状態", "映っていた時間（分）", "よかった場面（回）"]
    assert "動いた距離（推定・m）" in head and "位置の誤差（±cm）" in head
    assert len(rows) == 3
    for row, p in zip(rows[1:], board.people):
        assert row[0] == p.name and int(row[4]) == p.total
        assert float(row[3]) == pytest.approx(round(p.on_screen_s / 60, 1))


def test_people_csv_without_geometry_has_no_distance_column(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=60, geometry=False)
    head = rows_of(dashboard.people_csv(board_of(repo, aid)))[0]
    assert not any("距離" in h for h in head)


def test_csv_cells_never_start_a_formula(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=1, seconds=60)
    subject = repo.list_subjects(sid)[0]
    repo.conn.execute("UPDATE subjects SET display_name=? WHERE id=?",
                      ("=HYPERLINK(1)", subject["id"]))
    board = board_of(repo, aid)
    assert rows_of(dashboard.people_csv(board))[1][0] == "'=HYPERLINK(1)"
    assert rows_of(dashboard.moments_csv(board))[1][1] == "'=HYPERLINK(1)"


def test_moments_csv_lists_every_moment_in_time_order(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60)
    board = board_of(repo, aid)
    rows = rows_of(dashboard.moments_csv(board))
    assert rows[0][:4] == ["時間", "名前", "チーム", "場面の種類"]
    assert len(rows) == 1 + len(board.moments)
    assert all(r[0].startswith("開始から") for r in rows[1:])


def test_erasing_a_person_removes_the_groups_clips(env):
    from pfb_app.services import moment_clip
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    folder = moment_clip.clips_dir(settings, aid, sid)
    video = next(settings.inbox_dir.glob("*.mp4"))
    clip, _ = moment_clip.clip(folder, "cam1", video, 10.0)
    assert clip.is_file()
    erasure.erase(repo, settings, repo.list_subjects(sid)[0]["id"])
    assert not folder.exists()


def test_without_known_teams_the_whole_group_is_one_entry(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=3, seconds=60, label="A")
    board = board_of(repo, aid)
    members = dashboard.team_members(board)
    assert [(name, len(ps)) for name, ps in members] == [("Aの全員", 3)]


# --- the cache stamp ------------------------------------------------------------------------

def test_the_stamp_changes_with_what_the_board_reads(env):
    import json as _json
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=1, seconds=40)
    seen = {dashboard.stamp(repo, aid)}

    def changed():
        now = dashboard.stamp(repo, aid)
        assert now not in seen
        seen.add(now)

    meta = repo.get_submission(sid)["meta"]
    repo.conn.execute("UPDATE submissions SET meta=? WHERE id=?",
                      (_json.dumps({**meta, "clock_offset_min": 5}), sid))
    changed()
    for column, value in (("domain", "school"), ("sport", "soccer"), ("task_text", "別の内容"),
                          ("profile", "meeting")):
        repo.conn.execute(f"UPDATE assignments SET {column}=? WHERE id=?", (value, aid))
        changed()
    repo.conn.execute("INSERT INTO jobs (submission_id, stage, camera, model_key, status,"
                      " created_at) VALUES (?, 'track', 'cam1', 'standard', 'running', 'x')",
                      (sid,))
    changed()
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    changed()
    assert dashboard.stamp(repo, aid) == dashboard.stamp(repo, aid)


# --- review 2026-10-24 ----------------------------------------------------------------------

def test_moments_of_a_waiting_person_are_marked(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, unapproved=("Pb",))
    board = board_of(repo, aid)
    waiting = next(p for p in board.people if not p.approved)
    assert all(not m.approved for m in board.moments if m.subject_id == waiting.subject_id)
    assert all(m.approved for m in board.moments if m.subject_id != waiting.subject_id)
    rows = rows_of(dashboard.moments_csv(board))
    assert "状態" in rows[0]
    k = rows[0].index("状態")
    assert {r[k] for r in rows[1:] if r[1] == waiting.name} == {"承認前"}


def test_several_cameras_keep_their_own_minutes(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=60, cameras=("cam1", "cam2"))
    board = board_of(repo, aid)
    p = board.people[0]
    assert set(p.on_screen_by_camera) == {"cam1", "cam2"}
    assert board.several_cameras
    assert p.on_screen_s <= sum(p.on_screen_by_camera.values())
    head = rows_of(dashboard.people_csv(board))[0]
    assert "映っていた時間・カメラ1（分）" in head and "映っていた時間・カメラ2（分）" in head
    memo = rows_of(dashboard.people_csv(board))[1][-1]
    assert "おおよそ" in memo


def test_one_camera_has_no_per_camera_columns(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=60)
    board = board_of(repo, aid)
    assert not board.several_cameras
    assert not any("カメラ1（分）" in h for h in rows_of(dashboard.people_csv(board))[0])


def test_teams_csv_has_the_team_totals(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    board = board_of(repo, aid)
    rows = rows_of(dashboard.teams_csv(board))
    assert rows[0][:3] == ["チーム", "人数", "よかった場面（回）"]
    assert [r[0] for r in rows[1:]] == ["チーム1", "チーム2"]
    assert sum(int(r[2]) for r in rows[1:]) == board.total
    k = rows[0].index("パス（回）")
    key = board.categories[0].key
    assert [int(r[k]) for r in rows[1:]] == [t.counts.get(key, 0)
                                             for t in dashboard.team_totals(board)]


def test_heatmap_cells_as_a_table():
    rows = dashboard.court_cells([(50.0, 50.0)] * 3 + [(250.0, 150.0)])
    assert rows[0] == {"左のサイドラインから（m）": "0〜1", "ゴールの下の線から（m）": "0〜1",
                       "記録の割合（%）": 75}
    assert rows[1]["記録の割合（%）"] == 25 and len(rows) == 2
    assert dashboard.court_cells([]) == []
    rows = dashboard.frame_cells([(10.0, 10.0), (630.0, 350.0)], (640, 360))
    assert {r["画面の左から（列）"] for r in rows} == {"1/16", "16/16"}
    assert all(r["記録の割合（%）"] == 50 for r in rows)
