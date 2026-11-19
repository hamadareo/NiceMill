"""The people panel on 確認して承認 (synthetic images and invented names only)."""
import pytest
from PIL import Image

from pfb_app.domain.models import Stage
from pfb_app.services import auto_people, review
from pfb_app.storage import files
from test_wizard_ui_basic import app, env, html, no_internal_terms, ok, primaries, video_assignment  # noqa: F401
from test_wizard_ui_review_pdf import btn, reviewable, subject


def make_sheet(settings, sid, camera="cam1"):
    path = settings.data_dir / f"psheet_{sid}_{camera}.png"
    Image.new("RGB", (200, 100), (200, 30, 30)).save(path)
    return path


def set_track(repo, sid, **extra):
    run = repo.latest_run(sid, Stage.track, "cam1")
    out = {**run.output, **extra}
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (__import__("json").dumps(out), run.id))
    return repo.latest_run(sid, Stage.track, "cam1")


def test_crop_from_sheet_geometry(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    set_track(repo, sid, sheet=str(make_sheet(settings, sid)), sheet_geometry={"P1": [10, 10, 50, 80]})
    pa = subject(repo, sid, "Pa")
    crops = auto_people.person_crops(repo, settings, sid)
    path = crops[pa["id"]]
    sub = repo.get_submission(sid)
    assert path == (files.submission_dir(settings, sub["assignment_id"], sid) / "people"
                    / f"cam1_P1_t{repo.latest_run(sid, Stage.track, 'cam1').id}.jpg")
    with Image.open(path) as im:
        assert im.format == "JPEG" and im.size == (50, 80)
    assert subject(repo, sid, "Pb")["id"] not in crops           # no geometry for P2
    mtime = path.stat().st_mtime_ns
    auto_people.person_crops(repo, settings, sid)
    assert path.stat().st_mtime_ns == mtime                      # made once


def test_crop_prefers_names_run_crop(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    set_track(repo, sid, sheet=str(make_sheet(settings, sid)), sheet_geometry={"P1": [0, 0, 5, 5]})
    named = settings.data_dir / "named.jpg"
    Image.new("RGB", (4, 4)).save(named)

    class V:
        crops = {"P1": named}
    monkeypatch.setattr(review, "names_view", lambda *a, **k: V())
    crops = auto_people.person_crops(repo, settings, sid)
    assert crops[subject(repo, sid, "Pa")["id"]] == named


def open_panel(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    return settings, repo, aid, sid, ok(app(session={"assignment_id": aid}))


def test_no_picture_shows_shashin_nashi(env):
    *_, at = open_panel(env)
    assert "写真なし" in html(at)


def test_panel_buttons_are_secondary_and_copy_is_plain(env):
    *_, at = open_panel(env)
    assert any(e.label == "映っている人" for e in at.expander)
    labels = [b.label for b in at.button]
    assert labels.count("名前をつける") == 3 and labels.count("この人は評価しない") == 3
    assert primaries(at) == ["全員まとめて承認"]
    panel = [b for b in at.button if b.label in ("名前をつける", "この人は評価しない")]
    assert all(b.proto.type == "secondary" for b in panel)
    no_internal_terms(at)
    for word in ("P1", "ジョブ", "エンジン"):
        assert word not in html(at)


def test_exclude_then_restore_from_panel(env):
    settings, repo, aid, sid, at = open_panel(env)
    pb = subject(repo, sid, "Pb")
    at.button(key=f"pfb_people_excl_{pb['id']}").click().run()
    ok(at)
    assert repo.get_subject(pb["id"])["excluded"] is True
    assert at.button(key=f"pfb_people_excl_{pb['id']}").label == "評価に戻す"
    assert "（評価しない）" in html(at)
    at.button(key=f"pfb_people_excl_{pb['id']}").click().run()
    ok(at)
    assert repo.get_subject(pb["id"])["excluded"] is False


def test_rename_and_taken_name(env):
    settings, repo, aid, sid, at = open_panel(env)
    pa, pb = subject(repo, sid, "Pa"), subject(repo, sid, "Pb")
    at.text_input(key=f"pfb_people_name_{pa['id']}").set_value("新しい名前")
    at.button(key=f"pfb_people_rename_{pa['id']}").click().run()
    ok(at)
    assert repo.get_subject(pa["id"])["display_name"] == "新しい名前"
    at.text_input(key=f"pfb_people_name_{pb['id']}").set_value("新しい名前")
    at.button(key=f"pfb_people_rename_{pb['id']}").click().run()
    assert any("同じ名前の人がすでにいます。別の名前にしてください。" in e.value for e in at.error)
    assert repo.get_subject(pb["id"])["display_name"] != "新しい名前"


def test_retrack_makes_a_new_crop(env):
    """Same camera and label, a new track run with another sheet: a new file, not the old one."""
    from test_wizard_ui_people_wait import save
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    sub = repo.get_submission(sid)
    old = set_track(repo, sid, sheet=str(make_sheet(settings, sid)),
                    sheet_geometry={"P1": [10, 10, 50, 80]})
    other = settings.data_dir / "other_sheet.png"
    Image.new("RGB", (200, 100), (30, 30, 200)).save(other)
    new = save(repo, sid, Stage.track, {**old.output, "sheet": str(other),
                                        "sheet_geometry": {"P1": [0, 0, 30, 40]}}, camera="cam1")
    first = auto_people._sheet_crop(settings, sub, "cam1", "P1", old)
    second = auto_people._sheet_crop(settings, sub, "cam1", "P1", new)
    assert first != second and first.is_file()
    with Image.open(second) as im:
        assert im.size == (30, 40) and im.getpixel((5, 5))[2] > 150


def test_unused_roster_rows_have_no_card_and_their_names_can_be_picked(env):
    """An excluded roster row no camera maps is not a card; its name is in the select, the
    names evaluated people use are not, and picking it names the person."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    repo.add_subject(sid, "Pz", "山本", "外見", {})
    pz = subject(repo, sid, "Pz")
    repo.set_excluded(pz["id"], True)
    at = ok(app(session={"assignment_id": aid}))
    assert not any(b.key == f"pfb_people_rename_{pz['id']}" for b in at.button)
    assert [b.label for b in at.button].count("名前をつける") == 3
    pb = subject(repo, sid, "Pb")
    pick = at.selectbox(key=f"pfb_people_pick_{pb['id']}")
    assert pick.options == ["山本", "自分で入力"]
    pick.set_value("山本")
    at.button(key=f"pfb_people_rename_{pb['id']}").click().run()
    ok(at)
    assert repo.get_subject(pb["id"])["display_name"] == "山本"
    assert not any(s["participant_id"] == "Pz" for s in repo.list_subjects(sid))


def test_a_typed_name_is_shown_as_text(env):
    settings, repo, aid, sid, at = open_panel(env)
    pa = subject(repo, sid, "Pa")
    at.text_input(key=f"pfb_people_name_{pa['id']}").set_value("<i>x</i>*y*")
    at.button(key=f"pfb_people_rename_{pa['id']}").click().run()
    ok(at)
    page = html(at)
    assert "&lt;i&gt;x&lt;/i&gt;*y*" in page and "<i>x</i>" not in page


def test_panel_stays_after_excluding_the_last_person(env):
    """A group whose last person was excluded, picked on 確認して承認 (another group is
    there to check): the panel is still there, so 「評価に戻す」 can be pressed."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    reviewable(settings, repo, aid, label="B班")
    for pid in ("Pa", "Pb", "Pc"):
        auto_people.set_person_excluded(repo, subject(repo, sid, pid)["id"], True)
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": sid}))
    assert any(e.label == "映っている人" for e in at.expander)
    pa = subject(repo, sid, "Pa")
    assert at.button(key=f"pfb_people_excl_{pa['id']}").label == "評価に戻す"
    at.button(key=f"pfb_people_excl_{pa['id']}").click().run()
    ok(at)
    assert repo.get_subject(pa["id"])["excluded"] is False


def test_no_panel_for_the_paper_method(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    sub = repo.get_submission(sid)
    repo.update_submission_meta(sid, {**sub["meta"], "engine": "paper"})
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    assert "承認 0 / 全 3" in html(at)
    assert not any(e.label == "映っている人" for e in at.expander)
    assert "名前をつける" not in [b.label for b in at.button]


def test_a_number_name_shows_where_it_came_from(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, auto_people.NAME_SOURCE: {"Pa": "number"}})
    at = ok(app(session={"assignment_id": aid}))
    assert html(at).count("背番号から") == 1
    no_internal_terms(at)


def _left_out(repo, sid, not_playing=None, check=None):
    track_run = repo.latest_run(sid, Stage.track, "cam1")
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, auto_people.PARTICIPATION: {"cam1": {
        "track": track_run.id, "not_playing": not_playing or {}, "check": check or {},
        "included": []}}})


def test_people_not_in_the_game_have_their_own_part(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    labels = repo.latest_run(sid, Stage.track, "cam1").output["labels"]
    extra = "P9" if "P9" not in labels else "P99"
    _left_out(repo, sid, not_playing={extra: "off_court"}, check={"P1": "court"})
    called = []
    monkeypatch.setattr(auto_people, "include_person",
                        lambda repo_, settings_, sid_, cam, lb: called.append((sid_, cam, lb)) or [])
    at = ok(app(session={"assignment_id": aid}))
    page = html(at)
    assert "試合に参加していない人" in page and "コートの外にいました" in page
    assert page.count("要確認") == 1                      # Pa (P1) is to be checked
    button = at.button(key=f"pfb_people_incl_cam1_{extra}")
    assert button.label == "この人も評価する" and button.proto.type == "secondary"
    assert primaries(at) == ["全員まとめて承認"]
    no_internal_terms(at)
    assert extra not in page
    button.click().run()
    ok(at)
    assert called == [(sid, "cam1", extra)]


def test_no_part_when_everyone_takes_part(env):
    *_, at = open_panel(env)
    assert "試合に参加していない人" not in html(at) and "要確認" not in html(at)
