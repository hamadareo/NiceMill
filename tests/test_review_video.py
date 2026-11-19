"""振り返り用の編集動画と振り返りプレーヤー (pfb_app.review_video).

Synthetic works only (tests/pdf_synthetic.py: generated videos, invented names and texts): no
real video, frame or feedback is read.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from pdf_synthetic import SAYING, TEXT, save, video_group, work
from pfb_app.domain.models import Stage
from pfb_app.export import explainer_video as ev
from pfb_app.export import sheets
from pfb_app.export.explainer_video import Cancelled, RenderError
from pfb_app.export.tracks import Box
from pfb_app.review_video import claims, collect, edit, holder, labelling, model, player, service
from pfb_app.review_video.model import Moment, Package, Person
from test_wizard_ui_basic import env  # noqa: F401 (env)

SCHOOL_WORDS = ("生徒", "先生", "学生", "教員", "授業", "課題", "提出", "宿題", "学校",
                "教師", "児童", "担任", "クラス", "採点", "成績")
CREDIT = ["映像：テスト用の合成映像　作者：テスト", "ライセンス：CC BY 3.0 (テスト)"]


def _group(env, **kw):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=kw.pop("people", 3), seconds=40,
                      events_each=kw.pop("events_each", 3), **kw)
    return settings, repo, aid, sid


def _probe(path: Path) -> dict:
    import av
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        n = sum(1 for _ in c.decode(v))
        return {"codec": v.codec_context.name, "w": v.codec_context.width,
                "h": v.codec_context.height, "frames": n, "rate": float(v.average_rate),
                "audio": len(c.streams.audio)}


def _frame(path: Path, t: float) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        best, bd = None, 1e9
        for f in c.decode(v):
            d = abs(float(f.pts * v.time_base) - t)
            if d < bd:
                best, bd = f.to_ndarray(format="rgb24"), d
        return best


def _time_with_box(pkg, person, intro_s, scene=0):
    """Output time (s) of a moment in the video's first scene at which the person's box exists
    (the synthetic people leave the picture for a while)."""
    src = pkg.sources["cam1"]
    hs = [m for m in person.highlights()]
    seg = edit.segments(hs, lambda c: None)[scene]
    mine = set(person.labels["cam1"])
    t = seg.t0
    while t < seg.t1:
        found = {lb for lb, _ in edit.ai.boxes_at(src.boxes, round(t * src.fps) + 1)}
        if found & mine:
            return intro_s + (t - seg.t0)
        t += 0.1
    raise AssertionError("the person is never on screen in this scene")


# --- collect -------------------------------------------------------------------------------------

def test_package_holds_the_generated_sentences_unchanged(env):
    settings, repo, aid, sid = _group(env, people=3, unapproved=("Pc",))
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    assert [p.status for p in pkg.people].count("draft") == 1       # Pc is a draft, still included
    assert pkg.credit_lines == CREDIT and "cam1" in pkg.sources
    a = pkg.people[0]
    assert a.text and a.text == TEXT.format(name=a.name, work="試合")   # the generated text
    assert a.labels["cam1"]
    assert len(a.moments) == 3
    for m in a.moments:
        assert m.t_s is not None and m.cited
        assert m.what in SAYING.values()                            # as generated (本人 kept)
        assert m.why == "仲間の動きを助けた"
        assert m.when and m.suspect is False
    draft = next(p for p in pkg.people if p.status == "draft")
    assert draft.text == TEXT.format(name=draft.name, work="試合")   # a draft is shown as a draft
    assert draft.moments


def test_the_identity_check_flags_a_moment(env, monkeypatch):
    settings, repo, aid, sid = _group(env, people=2)
    first = None

    def foreign(self, source):
        nonlocal first
        events = (source.output or {}).get("events") or []
        first = first or events[0]["event_id"]
        return {first}
    monkeypatch.setattr(sheets._Submission, "foreign", foreign)
    pkg = collect.build_package(repo, settings, sid)
    flagged = [m for p in pkg.people for m in p.moments if m.suspect]
    assert [m.ref_id for m in flagged] == [first]


# --- scenes ----------------------------------------------------------------------------------------

def _m(t, ref="r", cam="cam1", **kw):
    return Moment(ref, cam, t, "開始から", "本人がパスした", "助けた", "パス", True, **kw)


def test_scenes_join_when_they_overlap_or_touch():
    ms = [_m(10, "a"), _m(15, "b"), _m(40, "c"), _m(41, "d", cam="cam2")]
    scenes = edit.segments(ms, lambda c: 100.0)
    assert [(s.camera, [m.ref_id for m in s.moments]) for s in scenes] == \
        [("cam1", ["a", "b"]), ("cam1", ["c"]), ("cam2", ["d"])]
    assert scenes[0].t0 == 6.0 and scenes[0].t1 == 18.0          # 4 s before, 3 s after
    # the clip's start / end are cut to the video
    s = edit.segments([_m(1.0), _m(99.0, "z")], lambda c: 100.0)
    assert s[0].t0 == 0.0 and s[1].t1 == 100.0
    assert edit.segments([_m(150.0)], lambda c: 100.0) == []     # past the end: none


def test_the_sentences_follow_the_moment():
    seg = edit.Segment("cam1", 6, 18, [_m(10, "a"), _m(15, "b")])
    assert edit.active(seg, 6).ref_id == "a"
    assert edit.active(seg, 12).ref_id == "a"
    assert edit.active(seg, 13.6).ref_id == "b"       # 1.5 s before the second one


# --- the video -----------------------------------------------------------------------------------

def test_highlight_video_is_phone_friendly_and_marked(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=3, events_each=2)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    p = pkg.people[0]
    target = tmp_path / "hl.mp4"
    ticks = []
    made = service.make_person_video(pkg, p.subject_id, target,
                                     progress=lambda f, n: ticks.append((f, n)))
    info = _probe(target)
    assert (info["codec"], info["w"], info["h"], info["audio"]) == ("h264", 1280, 720, 0)
    scenes = made["scenes"]
    assert 1 <= scenes <= 2 and made["moments"] == 2 and made["suspects"] == 0
    seconds = info["frames"] / info["rate"]
    assert abs(seconds - made["seconds"]) < 0.3
    assert seconds > edit.INTRO_S + edit.END_S + 3 * scenes
    assert ticks and ticks[-1][0] == 1.0
    assert not list(tmp_path.glob(".*.part.mp4"))                  # nothing left behind
    # a scene frame: the person's own-colour box (outline, coloured line) is drawn on the video
    t_scene = _time_with_box(pkg, p, edit.INTRO_S)
    frame = _frame(target, t_scene)
    video_part = frame[edit.HUD_H:400]
    blue = np.abs(video_part.astype(int) - np.array(model.team_color(p.team_letter)["rgb"])).sum(axis=2) < 40
    assert blue.sum() > 30
    # the band under the video is the caption's panel colour, with text drawn on it
    band = frame[-30:, 30:600]
    assert (np.abs(band.astype(int) - np.array(ev.PANEL)).sum(axis=2) > 60).sum() > 20


def test_suspect_scene_is_flagged_and_says_so(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    p = pkg.people[0]
    p.moments[0].suspect = True
    target = tmp_path / "hl.mp4"
    made = service.make_person_video(pkg, p.subject_id, target)
    assert made["suspects"] == 1
    frame = _frame(target, _time_with_box(pkg, p, edit.INTRO_S))
    pc = model.team_color(p.team_letter)
    own = np.abs(frame[:420].astype(int) - np.array(pc["rgb"])).sum(axis=2) < 40     # the person's colour, dotted box
    assert own.sum() > 30
    assert edit.SUSPECT_NOTE in model.SUSPECT_NOTE
    h = edit.band_height([p.moments[0]])
    plain = edit.band_height([_m(10)])
    assert h > plain                                                   # the extra line is there
    band = edit.caption_band(p.moments[0], h)
    assert (np.abs(band.astype(int) - np.array(edit.ORANGE)).sum(axis=2) < 40).sum() > 50


def test_the_band_shows_one_generated_sentence_whole_when_it_fits():
    m = _m(10)
    head, lines = edit.caption_lines(m)
    assert "".join(lines) == "助けた" == edit.main_text(m)         # the reason it counted, as generated
    only_what = Moment("x", "cam1", 10, "開始から", "本人がパスした", "", None, True)
    assert edit.main_text(only_what) == "本人がパスした"            # no reason: the observation
    long = Moment("x", "cam1", 10, "開始から", "あ" * 400, "い" * 400, None, True)
    _, lines = edit.caption_lines(long)
    assert len(lines) == 3 and lines[-1].endswith("…")
    f = ev.font(ev.BODY)
    assert all(f.getlength(s) <= ev.W - 2 * edit.PAD for s in lines)
    assert edit.band_height([long]) < int(ev.H * 0.45)             # worst case: still a big picture
    assert edit.band_height([m]) < edit.band_height([long])


def test_cards_say_it_is_a_draft_and_carry_the_credit(monkeypatch):
    drawn = []
    real = ev._text
    monkeypatch.setattr(ev, "_text", lambda d, xy, lines, *a, **k: (drawn.extend(lines),
                                                                    real(d, xy, lines, *a, **k))[1])
    pkg = Package("試合", "2026年11月17日", "ほかの選手", [], {}, CREDIT)
    p = Person(1, "選手1", {}, "draft", None, [])
    edit.intro_card(pkg, p, 2, 1)
    edit.end_card(pkg, p)
    text = "\n".join(drawn)
    assert model.DRAFT_NOTICE in text
    assert "点線の枠" in text                                     # only with a suspect
    assert "CC BY 3.0" in text and "作者" in text
    assert "まだ確認が済んでいません" in text                          # a draft says so
    assert "AIが映像を作り替えたものではありません" in text
    drawn.clear()
    edit.end_card(pkg, Person(1, "選手1", {}, "approved", "x", []))
    assert "まだ確認が済んでいません" not in "\n".join(drawn)
    drawn.clear()
    edit.intro_card(pkg, p, 2, 0)
    assert "点線の枠" not in "\n".join(drawn)


def test_wording_is_neutral_and_never_claims_a_human_check():
    sources = [Path(edit.__file__), Path(model.__file__), Path(player.__file__),
               Path(labelling.__file__), Path(collect.__file__)]
    for src in sources:
        text = src.read_text(encoding="utf-8")
        found = [w for w in SCHOOL_WORDS if w in text]
        assert not found, (src.name, found)
    assert "評価する人が確認してから渡します" in model.DRAFT_NOTICE
    assert "確認済み" not in model.DRAFT_NOTICE


def test_nothing_to_show_and_stop(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    p = pkg.people[0]
    p.moments.clear()
    out = tmp_path / "o"
    with pytest.raises(RenderError):
        service.make_person_video(pkg, p.subject_id, out / "x.mp4")
    assert not list(out.iterdir())
    q = pkg.people[1]
    with pytest.raises(Cancelled):
        service.make_person_video(pkg, q.subject_id, out / "y.mp4", stop=lambda: True)
    assert not list(out.iterdir())                                # a stop leaves nothing


def test_highlights_prefer_the_cited_moments():
    p = Person(1, "a", {}, "draft", None, [_m(5, "u", ), _m(9, "v"), _m(7, "w")])
    p.moments[0].cited = False
    assert [m.ref_id for m in p.highlights()] == ["w", "v"]
    for m in p.moments:
        m.cited = False
    assert [m.ref_id for m in p.highlights()] == ["u", "w", "v"]      # none cited: all, in time order
    p.moments.append(Moment("n", None, None, "", "", "", None, True))
    assert "n" not in [m.ref_id for m in p.highlights()]               # no time: not shown


# --- the player ----------------------------------------------------------------------------------

def _script_json(html: str) -> dict:
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_player_is_one_local_file_with_the_data_inside(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=3, events_each=2, unapproved=("Pc",))
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    made = player.build_player(pkg, tmp_path / "pl")
    html = Path(made["html"]).read_text(encoding="utf-8")
    data = _script_json(html)
    assert data == player.player_data(pkg, "video.mp4")
    assert (tmp_path / "pl" / "video.mp4").is_file()
    assert made["people"] == 3 and made["moments"] == 6
    # nothing from outside: no src/href/url() to another place, no import, no fetch
    assert not re.findall(r'(?:src|href)\s*=\s*["\']?(?:https?:)?//', html)
    assert "@import" not in html and "fetch(" not in html and "XMLHttpRequest" not in html
    assert not re.findall(r"url\(\s*[\"']?https?:", html)
    scripts = re.findall(r"<script[^>]*\bsrc=", html)
    assert scripts == []
    # sentences, times, tracks
    p0 = data["people"][0]
    assert [m["id"] for m in p0["moments"]] == [
        m.ref_id for m in sorted(pkg.people[0].moments, key=lambda m: (m.t_s, m.ref_id))]
    assert [m["what"] for m in p0["moments"]] == [
        m.what for m in sorted(pkg.people[0].moments, key=lambda m: (m.t_s, m.ref_id))]
    assert all(set(m) >= {"id", "t", "when", "what", "why", "suspect", "cited"}
               for p in data["people"] for m in p["moments"])
    labels = {lb for p in data["people"] for lb in p["labels"]}
    assert labels and set(data["tracks"]) == labels
    assert data["banner"] == model.DRAFT_NOTICE and data["credit"] == CREDIT
    # the controls the user asked for exist
    for needle in ('id="speed"', 'id="boxes"', 'id="tlinner"', "playbackRate", "currentTime"):
        assert needle in html


def test_player_escapes_names_that_could_break_the_page(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    pkg.people[0].name = "</script><b>x</b>"
    pkg.people[0].moments[0].what = "a</script>b c"
    pkg.title = "t</title>"
    made = player.build_player(pkg, tmp_path / "pl")
    html = Path(made["html"]).read_text(encoding="utf-8")
    assert html.count("</script>") == 2                                # the two real ones
    assert "<title>振り返りプレーヤー　t&lt;/title&gt;</title>" in html
    assert _script_json(html)["people"][0]["name"] == "</script><b>x</b>"


def test_player_thins_the_tracks(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    src = pkg.sources["cam1"]
    src.fps = 30.0                                                     # pretend: 30 fps tracks
    data = player.player_data(pkg, "video.mp4")
    assert data["step"] == 3
    for rows in data["tracks"].values():
        assert rows and all(r[0] % 3 == 0 for r in rows)


def test_player_needs_a_video_with_moments(env, tmp_path):
    pkg = Package("t", "", "ほかの選手", [], {})
    with pytest.raises(ValueError):
        player.build_player(pkg, tmp_path / "pl")


def test_make_all_writes_videos_and_the_player_and_changes_nothing_else(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    before = repo.conn.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0]
    made = service.make_all(repo, settings, sid, tmp_path / "out", credit_lines=CREDIT)
    assert len(made["videos"]) == 2 and all("error" not in v for v in made["videos"])
    assert (tmp_path / "out" / "player" / "review_player.html").is_file()
    assert sorted(p.suffix for p in (tmp_path / "out").iterdir() if p.is_file()) == [".mp4"] * 2
    assert repo.conn.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0] == before


# --- numbers, events, the timeline ---------------------------------------------------------------

def _names_run(repo, sid, reads, teams, gate_open=False):
    track = repo.latest_run(sid, Stage.track, "cam1")
    return save(repo, sid, Stage.names, {
        "track_run_id": track.id, "teams": teams, "status": {"number": {"status": "ok"}},
        "gate": {"number": {"enabled": gate_open, "threshold": 3.0}},
        "cues": {"number": reads}}, camera="cam1", upstream=[track.id])


def _read(number, agree=3, segments=None, dominant=None, share=1.0):
    r = {"number": number, "votes": agree, "frames_agree": agree}
    if segments:
        r.update(segments=segments, dominant=dominant, dominant_share=share, timeline_s=1.0)
    return r


def _numbered(env):
    settings, repo, aid, sid = _group(env, people=5, events_each=2)
    reads = {"P1": _read("3", segments=[[1, 200, "3"], [201, 400, "5"]], dominant="3", share=0.6),
             "P2": _read("3"), "P3": _read("7"), "P4": _read("7"), "P5": _read(None, agree=0)}
    teams = {"P1": "T1", "P2": "T2", "P3": "T1", "P4": "T1", "P5": "T2"}
    _names_run(repo, sid, reads, teams)
    return settings, repo, sid


def test_people_are_shown_by_the_number_the_ai_read(env):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    by = {p.internal: p for p in pkg.people}
    # the number cue's precision gate is closed: every number is a reading with a 「？」
    assert pkg.number_verified is False and not any(p.verified for p in pkg.people)
    assert by["P1"].label == "Aチーム 3番？" and by["P1"].number == "3" and not by["P1"].label_note
    assert by["P1"].dashed and by["P1"].tag() == "Aチーム 3番？"
    # the same number on the other team is told apart by the team, not merged
    assert by["P2"].label == "Bチーム 3番？" and not by["P2"].label_note
    # the same number twice in one team: flagged, both, and told apart by the tracker id
    assert by["P3"].label == by["P4"].label == "Aチーム 7番？"
    assert by["P3"].tag() == "Aチーム 7番？（P3）" and by["P4"].tag() == "Aチーム 7番？（P4）"
    assert by["P3"].label_note == by["P4"].label_note == labelling.DUPLICATE
    # a number that was not read (or not enough frames agreed): unknown, with the tracker id
    assert by["P5"].label == "番号不明（P5）" and by["P5"].number is None and not by["P5"].dashed
    # the app's own name is kept as it is (a separate field), and sentences are untouched
    assert by["P1"].name and all(m.what in SAYING.values() for m in by["P1"].moments)
    assert [p.internal for p in pkg.people][:2] == ["P1", "P3"]      # team, then number


def test_too_few_frames_agreeing_is_not_a_number(env):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    _names_run(repo, sid, {"P1": _read("9", agree=1), "P2": _read("4", agree=2)},
               {"P1": "T1", "P2": "T2"})
    pkg = collect.build_package(repo, settings, sid)
    by = {p.internal: p for p in pkg.people}
    assert by["P1"].label == "番号不明（P1）"          # one frame is not enough (MIN_FRAMES_AGREE)
    assert by["P2"].label == "Bチーム 4番？"


def test_a_number_change_on_a_track_is_marked_as_possibly_another_person(env):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    p1 = next(p for p in pkg.people if p.internal == "P1")
    changes = [m for m in p1.events if m.kind == model.KIND_CHANGE]
    assert len(changes) == 1
    c = changes[0]
    assert c.suspect and abs(c.t_s - 20.0) < 1e-6              # frame 201 at 10 fps
    assert "3番" in c.note and "5番" in c.note and "人物が別人かも・要確認" in c.note
    others = [m for p in pkg.people if p is not p1 for m in p.events if m.kind == model.KIND_CHANGE]
    assert others == []                                          # no timeline, no mark


def test_labels_without_any_number_run_are_unknown_not_guessed(env):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    assert all(p.number is None and p.label.startswith("番号不明（P") for p in pkg.people)
    assert all(p.team is None for p in pkg.people)


def test_conflicting_numbers_on_one_person_are_unknown_and_flagged():
    a = Person(1, "x", {}, "draft", None, [], number=None, internal="P1", conflict=True)
    b = Person(2, "y", {}, "draft", None, [], number="8", team=None, internal="P2")
    c = Person(3, "z", {}, "draft", None, [], number="8", team=None, internal="P3")
    labelling.assign([a, b, c])
    assert a.label == "番号不明（P1）" and a.label_note == labelling.CONFLICT
    assert b.label_note == c.label_note == labelling.DUPLICATE     # no team known: not told apart
    assert labelling.label_of("10", "T2", "P9") == "Bチーム 10番？"
    assert labelling.label_of("10", "T2", "P9", verified=True) == "Bチーム 10番"


def test_every_recorded_action_is_an_event_and_counted_ones_are_good(env):
    settings, repo, aid, sid = _group(env, people=2, events_each=3)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    out = dict(con.output)
    out["judgments"] = [dict(j) for j in out["judgments"]]
    out["judgments"][0]["is_positive_contribution"] = False
    out["judgments"][0]["contribution_interpretation"] = "今回は数えない"
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (json.dumps(out), con.id))
    pkg = collect.build_package(repo, settings, sid)
    events = [m for p in pkg.people for m in p.events if m.kind == model.KIND_EVENT]
    assert len(events) == 1 and events[0].what in SAYING.values() and events[0].why == "今回は数えない"
    assert events[0].t_s is not None and events[0].cited is False
    good = [m for p in pkg.people for m in p.moments]
    assert len(good) == 5 and events[0].ref_id not in {m.ref_id for m in good}
    data = player.player_data(pkg, "video.mp4")
    kinds = {m["kind"] for p in data["people"] for m in p["moments"] + p["events"]}
    assert kinds == {"good", "event"}
    assert data["duration"] > max(m["t"] for p in data["people"] for m in p["moments"])


def test_player_carries_labels_numbers_and_the_notes(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    made = player.build_player(pkg, tmp_path / "pl")
    html = Path(made["html"]).read_text(encoding="utf-8")
    data = _script_json(html)
    assert [p["label"] for p in data["people"]][:2] == ["Aチーム 3番？", "Aチーム 7番？"]
    assert [p["tag"] for p in data["people"]][1:3] == ["Aチーム 7番？（P3）", "Aチーム 7番？（P4）"]
    assert data["number_note"] == model.NUMBER_NOTE_UNVERIFIED and data["gate_note"] == model.GATE_NOTE
    assert "7人中6人" in data["gate_note"] and data["no_mark_note"] == model.NO_MARK_NOTE
    assert data["number_verified"] is False and data["people"][0]["dashed"] is True
    assert data["people"][0]["checked_note"] == model.CHECKED[True]      # P1 has a timeline
    assert data["people"][1]["checked_note"] == model.CHECKED[False]
    assert data["fix_note"] == model.FIX_NOTE
    changes = [m for p in data["people"] for m in p["events"] if m["kind"] == "change"]
    assert len(changes) == 1 and "人物が別人かも・要確認" in changes[0]["note"]
    for needle in ('id="tlinner"', 'id="zin"', 'id="zout"', 'id="allon"', "stepEvent", "ArrowLeft",
                   "tabIndex", 'aria-label', "ctrlKey"):
        assert needle in html
    # no counts or rankings of people on the page: no per-person totals are in the data
    assert not any(k in p for p in data["people"] for k in ("count", "total", "rank", "score"))


def test_cards_show_the_label_the_internal_id_and_the_number_note(monkeypatch):
    drawn = []
    real = ev._text
    monkeypatch.setattr(ev, "_text", lambda d, xy, lines, *a, **k: (drawn.extend(lines),
                                                                    real(d, xy, lines, *a, **k))[1])
    pkg = Package("試合", "2026年11月17日", "ほかの選手", [], {}, CREDIT)
    p = Person(1, "選手2", {}, "draft", None, [], number=None, internal="P2",
               label="番号不明（P2）", label_note=labelling.DUPLICATE)
    edit.intro_card(pkg, p, 1, 0)
    edit.end_card(pkg, p)
    text = "\n".join(drawn)
    flat = "".join(drawn)
    assert "ナイスミル上の名前：選手2　追跡ID：P2" in text
    assert labelling.DUPLICATE in text and model.NUMBER_NOTE_UNVERIFIED in text
    assert model.FIX_NOTE in text
    assert model.GATE_NOTE in flat and model.NO_MARK_NOTE in flat       # always on screen
    assert model.CHECKED[False] in text
    assert "この人の枠：番号不明（P2）" in text


def test_timeline_strip_marks_good_events_and_flags_by_shape_and_colour():
    p = Person(1, "x", {}, "draft", None, [_m(20, "g")],
               events=[_m(40, "e"), Moment("c", "cam1", 60, "", "", "", None, False, True,
                                           kind=model.KIND_CHANGE, note="n")])
    p.events[0].kind = model.KIND_EVENT
    p.events[0].cited = False
    strip = edit.timeline_strip(p, "cam1", 100.0)
    assert strip.shape == (edit.TL_H, ev.W, 3)
    def px(t, colour):
        x = edit.strip_x(t, 100.0)
        reg = strip[:, x - 10:x + 10].astype(int)
        return int((np.abs(reg - np.array(colour)).sum(axis=2) < 30).sum())
    assert px(20, ev.DASH) > 40 and px(60, edit.ORANGE) > 30 and px(40, (150, 170, 200)) > 20
    assert px(20, edit.ORANGE) == 0                               # not by colour alone: shapes differ


def test_the_highlight_video_has_the_whole_match_strip(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=2)
    pkg = collect.build_package(repo, settings, sid)
    p = pkg.people[0]
    target = tmp_path / "hl.mp4"
    service.make_person_video(pkg, p.subject_id, target)
    t = _time_with_box(pkg, p, edit.INTRO_S)
    frame = _frame(target, t)
    video_h = ev.H - edit.band_height(edit.segments(p.highlights(), lambda c: None)[0].moments)
    row = frame[video_h - edit.TL_H:video_h]
    assert (np.abs(row.astype(int) - np.array(ev.DASH)).sum(axis=2) < 60).sum() > 30   # a diamond
    assert (np.abs(row.astype(int) - 255).sum(axis=2) < 60).sum() > 10                 # the playhead


# --- the audit's findings -------------------------------------------------------------------------

def test_a_gate_that_is_open_gives_plain_numbers(env):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    _names_run(repo, sid, {"P1": _read("3"), "P2": _read("4")}, {"P1": "T1", "P2": "T2"},
               gate_open=True)
    pkg = collect.build_package(repo, settings, sid)
    assert pkg.number_verified is True
    by = {p.internal: p for p in pkg.people}
    assert by["P1"].label == "Aチーム 3番" and not by["P1"].dashed and by["P1"].verified
    data = player.player_data(pkg, "video.mp4")
    assert data["gate_note"] == "" and data["number_note"] == model.NUMBER_NOTE
    drawn = _drawn_texts(pkg, by["P1"])
    assert model.GATE_NOTE not in drawn and "？" not in by["P1"].label


def test_choosing_people_does_not_hide_a_duplicate_number(env):
    settings, repo, sid = _numbered(env)
    full = collect.build_package(repo, settings, sid)
    p3 = next(p for p in full.people if p.internal == "P3")
    only = collect.build_package(repo, settings, sid, subject_ids=[p3.subject_id])
    assert [p.internal for p in only.people] == ["P3"]
    assert only.people[0].label_note == labelling.DUPLICATE          # P4 is not in the video, still counts
    assert only.people[0].tag() == "Aチーム 7番？（P3）"


def test_outputs_are_copies_readable_by_the_owner_only(env, tmp_path):
    import os
    import stat
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    made = service.make_all(repo, settings, sid, tmp_path / "out")
    out = tmp_path / "out"
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert stat.S_IMODE((out / "player").stat().st_mode) == 0o700
    for f in list(out.glob("*.mp4")) + list((out / "player").iterdir()):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600, f
    src = next(iter(collect.build_package(repo, settings, sid).sources.values())).video
    copy = out / "player" / "video.mp4"
    assert copy.stat().st_nlink == 1 and not os.path.samefile(src, copy)    # a copy, not a link
    assert copy.read_bytes() == Path(src).read_bytes()
    assert made["player"]["video"] == str(copy)


def test_the_cli_opens_the_database_read_only(env, tmp_path):
    import hashlib
    import sqlite3
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    db = settings.data_dir / "app.db"
    repo.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    h = lambda: hashlib.sha256(db.read_bytes()).hexdigest()   # noqa: E731
    before = h()
    ro = service.open_readonly(settings)
    with pytest.raises(sqlite3.OperationalError):
        ro.conn.execute("UPDATE assignments SET title='x'")
    with pytest.raises(sqlite3.OperationalError):
        ro.conn.execute("CREATE TABLE zzz(a)")
    pkg = collect.build_package(ro, settings, sid)
    assert pkg.people
    made = service.make_all(ro, settings, sid, tmp_path / "out", with_player=False)
    assert made["videos"]
    assert h() == before                                       # not one byte changed
    with pytest.raises(FileNotFoundError):
        service.open_readonly(type(settings)(data_dir=tmp_path / "nowhere"))


def test_data_block_cannot_be_broken_by_html_comment_openers(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    pkg.people[0].moments[0].what = 'x<!--<script>alert(1)</script>y'
    pkg.people[0].moments[0].ref_id = 'a"b\\c'
    made = player.build_player(pkg, tmp_path / "pl")
    html = Path(made["html"]).read_text(encoding="utf-8")
    block = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert "<" not in block.group(1)
    assert html.count("<script") == 2 and "<!--" not in block.group(1)
    assert "CSS.escape" in html
    assert json.loads(block.group(1))["people"][0]["moments"][0]["what"].startswith("x<!--<script>")


def test_notes_and_windows_match_between_video_and_player():
    assert player.WINDOW_BEFORE_S == edit.BEFORE_S == player.PREROLL_S
    assert player.WINDOW_AFTER_S == edit.AFTER_S
    f = ev.font(ev.SMALL)
    for note in (model.BAND_NOTE, model.BAND_NOTE_VERIFIED):
        assert f.getlength(note) <= ev.W - 2 * edit.PAD            # one line in the band
    assert "未検証" in model.BAND_NOTE and "確認済み" not in model.BAND_NOTE


def test_a_dashed_box_has_gaps_a_solid_one_does_not():
    boxes = [("P1", Box(40, 40, 200, 100))]
    def top_edge(dashed):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        edit._draw_boxes(img, boxes, {"P1"}, (1.0, 1.0), ev.BLUE, ev._tag("x", ev.BLUE), dashed)
        row = img[40]
        on = (np.abs(row.astype(int) - np.array(ev.BLUE)).sum(axis=1) < 60)[60:230]
        return on
    solid, dashed = top_edge(False), top_edge(True)
    assert solid.all() and 0.2 < dashed.mean() < 0.85


def test_a_flagged_good_moment_keeps_its_diamond_and_rim_in_the_strip():
    g = _m(20, "g", suspect=True)
    p = Person(1, "x", {}, "draft", None, [g])
    strip = edit.timeline_strip(p, "cam1", 100.0)
    x = edit.strip_x(20, 100.0)
    reg = strip[:, x - 12:x + 12].astype(int)
    assert (np.abs(reg - np.array(edit.ORANGE)).sum(axis=2) < 30).sum() > 80     # a big diamond
    assert (np.abs(reg - 255).sum(axis=2) < 40).sum() > 15                        # the cited rim


def test_cards_fit_on_the_picture_and_do_not_repeat_the_date():
    drawn = []
    pkg = Package("バスケットボールの試合 11月17日", "2026年11月17日", "ほかの選手", [], {},
                  ["映像：" + "あ" * 60, "ライセンス：" + "い" * 90])
    p = Person(1, "選手2", {}, "draft", None, [], number="14", internal="P6", label="14番？",
               label_note=labelling.DUPLICATE)
    for card in (edit.intro_card(pkg, p, 3, 2), edit.end_card(pkg, p)):
        assert (card[-14:] == np.array(ev.BG)).all(), "text runs off the card"
    import numpy
    real = ev._text
    ev._text = lambda d, xy, lines, *a, **k: (drawn.extend(lines), real(d, xy, lines, *a, **k))[1]
    try:
        edit.intro_card(pkg, p, 3, 2)
    finally:
        ev._text = real
    title = [s for s in drawn if "振り返り用のハイライト" in s][0]
    assert "2026年" not in title and "11月17日" in title
    assert numpy is not None


def _drawn_texts(pkg, person):
    drawn = []
    real = ev._text
    ev._text = lambda d, xy, lines, *a, **k: (drawn.extend(lines), real(d, xy, lines, *a, **k))[1]
    try:
        edit.intro_card(pkg, person, 1, 1)
        edit.end_card(pkg, person)
        for m in person.highlights()[:1]:
            edit.caption_band(m, edit.band_height([m]), person.verified)
            drawn.extend(edit.caption_lines(m)[1] + [edit.caption_lines(m)[0]])
    finally:
        ev._text = real
    return "\n".join(drawn) + "".join(drawn)


def all_screen_text(pkg, person, tmp_path) -> str:
    """Every word a person can read in the video cards / bands and the player (the page text,
    the strings in the script and the data), for the wording checks."""
    made = player.build_player(pkg, tmp_path / "pl")
    html = Path(made["html"]).read_text(encoding="utf-8")
    return _drawn_texts(pkg, person) + "\n" + html


def test_every_word_on_the_screens_is_neutral(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    text = all_screen_text(pkg, pkg.people[0], tmp_path)
    assert not [w for w in SCHOOL_WORDS if w in text]
    assert "確認済み" not in text and "承認済み" not in text and "人の確認が済んだ" not in text
    # the docs a person may read carry no school words either
    doc = Path(__file__).resolve().parents[1] / "docs" / "review_video.md"
    assert not [w for w in SCHOOL_WORDS if w in doc.read_text(encoding="utf-8")]


# --- the redesign: fewer words, one reading order ----------------------------------------------------

def test_player_text_is_big_enough_and_notes_live_in_one_panel(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    ui_css = re.sub(r"\.cmt[^{]*\{[^}]*\}", "", html.split("<script")[0])   # the video caption overlay is deliberately small
    sizes = [int(x) for x in re.findall(r"font-size:(\d+)px", ui_css)]
    assert sizes and min(sizes) >= 14                                   # nothing else below 14 px
    big = re.search(r"#card \.main\{font-size:(\d+)px", html)
    assert big and int(big.group(1)) >= 18
    assert "AIの下書き・背番号は未検証 ⓘ" in html and 'id="panel"' in html and 'id="info"' in html
    for ref in ("D.banner", "D.number_note", "D.fix_note", "D.gate_note", "D.no_mark_note",
                "D.credit", "D.notes", "checked_note"):
        assert ref in html
    data = _script_json(html)
    assert data["gate_note"] == model.GATE_NOTE and data["no_mark_note"] == model.NO_MARK_NOTE
    static = html.split("<script")[0]
    for sentence in (model.GATE_NOTE, model.NO_MARK_NOTE, model.NUMBER_NOTE_UNVERIFIED,
                     model.DRAFT_NOTICE, model.FIX_NOTE):
        assert sentence not in static
    assert "chip" in html and '<details id="list">' in html
    assert 'try{return window.localStorage' in html and 'rv_tips_done' in html
    for tipsentence in ("① ここで再生します", "② ここで場面を選びます", "③ ここに文が出ます"):
        assert tipsentence in html


def test_the_band_note_is_one_short_line():
    assert model.BAND_NOTE == "AIの下書き・背番号は未検証" and model.BAND_NOTE_VERIFIED == "AIの下書き・要確認"
    assert len(model.BAND_NOTE) < 20



def test_player_has_live_comments_over_the_video(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    static = html.split("<script")[0]
    assert 'id="cm"' in static and 'aria-live="polite"' in static      # drawn inside #stage (fullscreen-safe)
    assert static.index('id="stage"') < static.index('id="cm"') < static.index('id="tools"')
    assert 'id="cm-on" type="checkbox" checked' in static and "コメント(C)" in static
    assert 'id="cm-sel" type="checkbox" checked' in static and 'id="fs"' in static   # captions follow the box: the chosen person's, by default
    css = static.split("<style>")[1]
    assert re.search(r"\.cmt\{[^}]*font-size:1[1-5]px", css)   # small caption-style (11-15 px), must not hide the match
    assert "prefers-reduced-motion:reduce" in css and "#stage:fullscreen" in css
    assert "AI下書き" in html and "要確認" in html
    assert 'ev.key==="c"' in html                                       # C toggles the comments
    assert _script_json(html)["window"] == [4.0, 3.0]                   # same window as the highlight video


def test_boxes_of_others_are_off_by_default_and_asked_for_by_a_switch(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    static = html.split("<script")[0]
    assert 'id="boxes" type="checkbox" checked' in static and "選んだ人の枠" in static
    assert 'id="allbox" type="checkbox">' in static and "全員の枠" in static   # not checked
    script = html.split("<script>")[1]
    # the grey boxes of everyone else are drawn only inside the allbox branch
    grey = script.index('rgba(150,156,166,.9)')
    assert script.rindex('getElementById("allbox").checked', 0, grey) > script.rindex("function draw", 0, grey)


def test_nonplayer_is_a_hint_from_the_apps_own_cues_never_a_removal(env):
    settings, repo, aid, sid = _group(env, people=4, events_each=1)
    reads = {f"P{i}": _read(str(i)) for i in range(1, 5)}
    teams = {"P1": "T1", "P2": "other", "P3": "T2", "P4": "other"}
    _names_run(repo, sid, reads, teams)
    meta = repo.get_submission(sid)["meta"]
    meta["participation"] = {"cam1": {"track": 1, "not_playing": {}, "check": {"P4": "no_court"},
                                      "included": []}}
    repo.update_submission_meta(sid, meta)
    pkg = collect.build_package(repo, settings, sid)
    by = {p.internal.rstrip("ほか"): p for p in pkg.people}
    assert len(pkg.people) == 4                                     # nobody is dropped
    assert by["P1"].nonplayer == "" and by["P3"].nonplayer == ""    # a team colour: never a hint
    assert by["P2"].nonplayer == model.NONPLAYER_WHY_OTHER
    assert all("選手以外かも" in v.nonplayer for v in (by["P2"], by["P4"]))


def test_no_cue_no_hint_and_the_player_says_so_plainly(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    assert all(p.nonplayer == "" for p in pkg.people)
    pkg.people[0].nonplayer = model.NONPLAYER_WHY_CHECK
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    assert [p["nonplayer"] for p in d["people"]][0] == model.NONPLAYER_WHY_CHECK
    assert "選手以外かも" in html and "選手の可能性もあります" in html


def test_highlight_video_draws_no_grey_boxes_on_others_unless_asked(monkeypatch):
    boxes = [("P1", Box(40, 40, 100, 100)), ("P9", Box(300, 40, 100, 100))]   # P9: a referee, say
    tag = ev._tag("P1", ev.BLUE)

    def draw():
        img = np.full((300, 600, 3), 255, np.uint8)
        edit._draw_boxes(img, boxes, {"P1"}, (1.0, 1.0), ev.BLUE, tag)
        return img
    assert edit.SHOW_OTHERS is False
    quiet = draw()
    assert (quiet[40:140, 300:400] == 255).all()          # nothing on the other person
    monkeypatch.setattr(edit, "SHOW_OTHERS", True)
    assert (draw()[40:140, 300:400] != 255).any()         # --others: the thin grey box


def test_team_colours_are_readable_distinct_and_not_reserved_colours():
    import itertools
    cols = {k: model.team_color(k) for k in ("A", "B", None)}
    for c in cols.values():
        assert model.contrast(c["hex"], c["ink"]) >= 4.5, c            # text on the colour chip
    assert len({c["hex"] for c in cols.values()}) == 3
    reserved = {"#e02020", "#f0e442", "#e8730c", "#e69f00"}            # red = ball, yellow = court, orange
    assert not reserved & {c["hex"].lower() for c in cols.values()}
    M = {"deut": [[.367322, .860646, -.227968], [.280085, .672501, .047413], [-.01182, .04294, .968881]],
         "prot": [[.152286, 1.052583, -.204868], [.114503, .786281, .099216], [-.003882, -.048116, 1.051998]],
         "trit": [[1.255528, -.076749, -.178779], [-.078411, .930809, .147602], [.004733, .691367, .3039]],
         "normal": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    def lin(v):
        c = v / 255
        return c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4

    def lab(rgb, m):
        l = [lin(x) for x in rgb]
        r, g, b = (max(0, min(1, sum(m[i][j] * l[j] for j in range(3)))) for i in range(3))
        x, y, z = (.4124 * r + .3576 * g + .1805 * b) / .9505, .2126 * r + .7152 * g + .0722 * b, (.0193 * r + .1192 * g + .9505 * b) / 1.089
        f = [t ** (1 / 3) if t > .008856 else 7.787 * t + 16 / 116 for t in (x, y, z)]
        return (116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2]))
    for name, m in M.items():
        for a, b in itertools.combinations(cols.values(), 2):
            d = sum((u - v) ** 2 for u, v in zip(lab(a["rgb"], m), lab(b["rgb"], m))) ** .5
            assert d >= 20, (name, a["hex"], b["hex"], d)             # strongly different, also for colour-blind eyes


def test_nonplayer_scoring_sets_apart_small_odd_colour_groups_only():
    from pfb_app.review_video import nonplayer
    team_a = {f"a{i}": np.array([55.0, 5, -18]) + i for i in range(4)}            # a light kit, 4 people
    team_b = {f"b{i}": np.array([15.0, 4, 2]) + i for i in range(4)}              # a dark kit, 4 people
    refs = {"r1": np.array([22.0, 14, -30]), "r2": np.array([21.0, 15, -31])}      # two in one odd colour
    out = nonplayer.score({**team_a, **team_b, **refs})
    assert out["r1"]["level"] in ("mid", "high") and out["r2"]["level"] in ("mid", "high")
    assert all(out[k]["level"] == "" and out[k]["score"] == 0 for k in {**team_a, **team_b})   # never a team member
    assert {out[k]["team"] for k in team_a} == {out["a0"]["team"]} and out["a0"]["team"] != out["b0"]["team"]
    assert nonplayer.score({"x": np.array([20.0, 15, -30]), "y": np.array([50.0, 0, 0])}) == {}   # too few people: no judgement
    one = nonplayer.score({**team_a, **team_b, "z": np.array([30.0, 30, 30])})
    assert one["z"]["level"] == "high"                                              # one person, far from both kits
    # audit 4 (wheelchair clips): the colours do not split into two teams -> no "team A" for everybody,
    # and the second team (here the odd-coloured three) is never set apart as non-players
    lone = nonplayer.score({**team_a, **{f"c{i}": np.array([30.0, 30, 30]) + i for i in range(2)}})
    assert all(v["team"] is None for v in lone.values()) and all(v["level"] != "high" for v in lone.values())


def test_a_set_apart_person_is_hidden_by_default_and_can_be_brought_back(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    pkg.people[0].excluded = True
    pkg.people[0].nonplayer = model.NONPLAYER_WHY_IMAGE
    res = service.make_all(repo, settings, sid, tmp_path / "o")
    assert all(not v.get("file") or "選手以外" not in v.get("skipped", "") for v in res["videos"])
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    assert d["people"][0]["excluded"] is True and d["people"][1]["excluded"] is False
    script = html.split("<script>")[1]
    assert "function isOut(" in script and "function players()" in script and 'id="restore"' in script
    assert 'id="np-on"' in html and "選手以外も表示" in html
    assert "選手以外（推定）" in script and "選手に戻す" in script
    assert "チームの名前ではありません" in html                                       # the page says what the team colours are (the legend / the colour note)


def test_the_picture_cues_never_override_the_apps_team_colour(env):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    for p in pkg.people:
        if p.team in ("T1", "T2"):
            assert not p.excluded and p.team_letter == {"T1": "A", "T2": "B"}[p.team]


def test_player_draws_bubbles_over_the_head_with_the_number_chip_in_the_teams_colour(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    legacy = {c.lower() for c in model.TEAM_COLORS.values()} | set(model.TEAM_COLORS.values())
    kit = {d["team_style"][k]["hex"] for k in "AB"} | {"#8E9196"} if d["team_style"] else set()
    assert all(p["color"] in (kit or legacy) for p in d["people"])        # the kit colours measured from the picture, else the fixed palette
    css, script = html.split("<style>")[1].split("</style>")[0], html.split("<script>")[1]
    assert re.search(r"\.bl\{[^}]*border:2px solid var\(--pc\)", css) and ".bl .cn{" in css and ".bl:after" in css   # a tail
    assert "headBox(" in script and "one bubble per person" in script and "slice(0,D.max_bubbles)" in script
    assert d["max_bubbles"] == player.MAX_BUBBLES == 2      # never more than two at once: the match must stay visible
    assert 'class="cn"' in script                                      # the number chip is inside the bubble
    assert "▲要確認" in script and 'ctx.setLineDash(warn&&!p.dashed?[2,4]' in script      # a flag is a symbol + a dotted box


def test_highlight_box_and_tag_use_the_teams_colour(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    p = pkg.people[1]
    p.team_letter = "B"
    pc = model.team_color("B")
    target = tmp_path / "hl.mp4"
    service.make_person_video(pkg, p.subject_id, target)
    frame = _frame(target, _time_with_box(pkg, p, edit.INTRO_S))
    near = np.abs(frame[:420].astype(int) - np.array(pc["rgb"])).sum(axis=2) < 40
    assert near.sum() > 30


def test_ball_holder_is_the_one_clearly_nearest_person_else_nobody():
    near_feet = {"P1": (100, 100, 40, 120), "P2": (400, 100, 40, 120)}
    ball = [118, 195, 130, 207, 0.8]                      # at P1's feet
    assert holder.holder_of(ball, near_feet) == "P1"
    assert holder.holder_of([250, 160, 262, 172, 0.8], near_feet) is None          # between them: nobody
    both = {"P1": (100, 100, 40, 120), "P2": (110, 100, 40, 120)}                   # two boxes equally near
    assert holder.holder_of([119, 160, 131, 172, 0.8], both) is None
    assert holder.holder_of([600, 160, 612, 172, 0.9], near_feet) is None           # far from everybody


def test_ball_segments_need_confidence_and_fill_only_short_gaps():
    boxes = {"P1": (100, 100, 40, 120)}
    at = lambda f: boxes
    hit = lambda c: [118, 195, 130, 207, c]
    assert holder.segments({10: hit(0.3)}, at) == []                               # too unsure
    assert holder.segments({10: hit(0.5)}, at) == []                               # one middling sighting: not enough
    assert holder.segments({10: hit(0.7)}, at) == [[10 - holder.HOLD_PAD, 10 + holder.HOLD_PAD, "P1"]]
    two = holder.segments({10: hit(0.5), 14: hit(0.5)}, at)                         # a short gap is bridged
    assert two == [[10 - holder.HOLD_PAD, 14 + holder.HOLD_PAD, "P1"]]
    apart = holder.segments({10: hit(0.5), 10 + holder.MAX_GAP + 5: hit(0.5)}, at)  # a long gap is not
    assert apart == []
    assert holder.holder_at([[5, 9, "P1"]], 7) == "P1" and holder.holder_at([[5, 9, "P1"]], 11) is None


def test_highlight_video_draws_a_red_ring_under_the_ball_holder_only():
    img = np.full((300, 600, 3), 90, np.uint8)
    edit._ball_ring(img, [("P1", Box(100, 50, 60, 150)), ("P2", Box(400, 50, 60, 150))], "P1", (1.0, 1.0))
    red = np.abs(img.astype(int) - np.array(edit.BALL_RED)).sum(axis=2) < 30
    assert red[:, :300].sum() > 40 and red[:, 300:].sum() == 0
    ys = np.where(red.any(axis=1))[0]
    assert ys.min() > 150                                                          # at the feet, not on the body
    assert edit.BALL_RED not in [tuple(c["rgb"]) for c in (model.team_color(k) for k in ("A", "B", None))]


def test_player_without_ball_data_shows_no_ball_switch_and_with_it_says_estimate(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    d = _script_json(Path(player.build_player(pkg, tmp_path / "a")["html"]).read_text(encoding="utf-8"))
    assert d["ball"] is None                                                       # no detection given: no feature
    pkg.holders, pkg.ball_frames = [[1, 9, "P1"]], 12
    html = Path(player.build_player(pkg, tmp_path / "b")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    assert d["ball"]["segs"] == [[1, 9, "P1"]] and "推定" in d["ball"]["note"] and "別の人を指すこと" in d["ball"]["note"]
    assert 'id="ball-on" type="checkbox" checked' in html and "#E02020" in html


def test_one_look_features_are_switches_fed_by_the_page_data_only(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    static = html.split("<script")[0]
    for i in ("f-auto", "f-slow", "f-spot", "f-trail", "f-live", "f-shot", "f-help", "prevm", "nextm", "helpbox"):
        assert f'id="{i}"' in static
    assert 'id="f-spot" type="checkbox">' in static and 'id="f-trail" type="checkbox">' in static   # off by default
    assert 'id="f-live" type="checkbox">' in static           # the live memo is off at first (fewer words over the match)
    script = html.split("<script>")[1]
    assert "記録がありません" in script and "実況メモ・AI下書き" in script                    # no data -> says so; a draft
    assert "ev.key===\"?\"" in script and "ev.key===\"f\"" in script


def test_court_outline_only_from_a_registered_court_and_never_estimated(env, tmp_path):
    settings, repo, aid, sid = _group(env, people=2, events_each=1)
    pkg = collect.build_package(repo, settings, sid)
    assert pkg.court is not None and pkg.court["frames"]                       # the synthetic submission has a court
    poly = model.court_at(pkg.court, pkg.court["frames"][0][0])
    assert poly is not None and len(poly) == 4
    assert model.court_at(pkg.court, pkg.court["frames"][-1][0] + 10000) is None   # too far from a sample: none
    assert model.court_at(None, 1) is None
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    assert d["court"]["frames"] and "推定した線ではありません" in d["court"]["text"]
    assert 'id="court-on" type="checkbox" checked' in html
    pkg.court = None
    assert _script_json(Path(player.build_player(pkg, tmp_path / "p2")["html"]).read_text(encoding="utf-8"))["court"] is None


def test_unnamed_detections_are_drawn_only_under_all_boxes_as_a_third_colour(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    cam = next(iter(pkg.sources))
    step = max(1, -(-int(pkg.sources[cam].fps) // 10))
    pkg.extras = {cam: np.array([[step * 3, 0, 10, 20, 30, 60, 0.9], [step * 3 + 1, 0, 10, 20, 30, 60, 0.9]])}
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    assert d["extras"][0][:5] == [step * 3, 10, 20, 30, 60] and "そのほかの人" in d["extra_note"]
    assert "途切れる" in d["extra_note"] and "直っていません" in d["extra_note"]      # the limits are stated
    pkg.extras = {}
    assert _script_json(Path(player.build_player(pkg, tmp_path / "p2")["html"]).read_text(encoding="utf-8"))["extras"] is None


def test_action_claims_the_cues_do_not_carry_are_weakened_everywhere(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    m = next(m for p in pkg.people for m in p.moments)
    m.why = "ボールをシュートした。"
    cues = {m.ref_id: {"wrist_above_head": -0.3, "pose_frames": 5}}      # no wrist above the nose: not a shot
    r = claims.apply(pkg.people, cues)
    assert r["weakened"] >= 1 and m.weak and m.why_orig == "ボールをシュートした。"
    assert "シュート" not in m.why and "要確認" in m.why                    # says less than the original, never more
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    shown = [x for q in d["people"] for x in q["moments"] if x["id"] == m.ref_id][0]
    assert shown["weak"] is True and "要確認" in shown["why"] and "断定しない" in d["claim_note"]
    assert edit.main_text(m).startswith("要確認：")                       # the highlight caption says it too
    other = next(x for p in pkg.people for x in p.moments if x is not m)
    before = other.why
    claims.apply(pkg.people, {})                                          # no cue: left exactly as it was (unchecked)
    assert other.why == before and not other.weak


# --- audit 2026-11-18: behaviour that only a real page can show -----------------------------------
def test_the_page_keeps_the_first_view_small_and_says_what_to_do_when_the_video_fails(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    static = html.split("<script")[0]
    # the switches live in a closed <details>; only the two moment buttons and the help stay outside
    outside = re.sub(r'<details id="opts">.*?</details>', "", static, flags=re.S)
    assert '<details id="opts">' in static and "<details id=\"opts\" open" not in static
    for i in ("prevm", "nextm", "f-help"):
        assert f'id="{i}"' in outside
    for i in ("f-auto", "f-slow", "f-spot", "f-trail", "f-shot", "boxes", "allbox", "cm-on", "cm-sel", "fs", "speed"):
        assert f'id="{i}"' not in outside and f'id="{i}"' in static
    assert 'id="vfail"' in static and "映像を読み込めませんでした" in static
    assert ".tg span{white-space:nowrap}" in static and "flex-wrap:wrap;gap:6px 8px" in static   # no one-character-per-line buttons on a phone
    script = html.split("<script>")[1]
    assert "ev.ctrlKey||ev.metaKey||ev.altKey" in script          # Cmd+F / Cmd+C / Ctrl+N are the browser's own
    assert "prefers-reduced-motion:reduce){.cmt,.bl{transition:none}" in static


def test_a_weakened_person_gets_no_full_paragraph_and_the_strip_marks_it(env, tmp_path):
    """Audit 3: the generated paragraph can restate the sentence the action check weakened, so a
    person with a weakened scene shows no full paragraph (the closed list says why); the highlight
    video's strip marks the weakened diamond in the flag colour, as the player's timeline does."""
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    person = next(p for p in pkg.people if p.moments)
    person.text = "この人はボールをシュートした。"
    m = person.moments[0]
    m.why, m.why_orig, m.weak = "この場面に映っていた（行動の内容は要確認）", "ボールをシュートした。", True
    d = player.player_data(pkg, "video.mp4")
    mine = next(q for q in d["people"] if q["id"] == person.subject_id)
    assert mine["text"] is None and mine["text_withheld"] is True
    assert "ボールをシュートした。" not in json.dumps(mine, ensure_ascii=False)   # the original sentence is nowhere in the data
    others = [q for q in d["people"] if q["id"] != person.subject_id and q["text"]]
    assert all(not q["text_withheld"] for q in others)
    cam = m.camera
    strip = edit.timeline_strip(person, cam, 40.0)
    x, mid = edit.strip_x(m.t_s, 40.0), edit.TL_H // 2
    assert tuple(int(v) for v in strip[mid, x]) == tuple(edit.ORANGE)


def test_no_original_sentence_of_a_weakened_scene_is_anywhere_in_the_page(env, tmp_path):
    """Audit 4: the page data carried the generated observation (`what`) of a weakened moment, which
    states the same action. Nothing the page embeds (all of the JSON, the script, the markup) may
    contain a sentence the action check weakened."""
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    m = next(m for p in pkg.people for m in p.moments)
    m.what, m.why, m.category = "本人はボールをバスケットに向かってシュートする。", "ボールをシュートした。", "専用の種類ラベル"
    r = claims.apply(pkg.people, {m.ref_id: {"wrist_above_head": -0.3, "pose_frames": 5}})
    assert r["weakened"] >= 1 and m.weak
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    for original in ("本人はボールをバスケットに向かってシュートする", "ボールをシュートした。", "シュートする。", "専用の種類ラベル"):
        assert original not in html, original
    shown = [x for q in player.player_data(pkg, "v.mp4")["people"] for x in q["moments"] if x["id"] == m.ref_id][0]
    assert shown["what"] == "" and "要確認" in shown["why"] and shown["weak"]
    assert "要確認" in edit.main_text(m) and "シュート" not in edit.main_text(m)      # the highlight caption: the same sentence
    assert "専用の種類ラベル" not in "".join(edit.caption_lines(m)[1]) + edit.caption_lines(m)[0]   # ...and its head line names no category


def test_an_unverified_action_of_the_other_recorded_actions_is_never_stated_as_a_fact(env, tmp_path):
    """Audit 4: the 行動の記録 sentences were outside the action check. A sentence that claims an action
    (shoot, dribble, pass, defend, run) is hedged and marked 要確認 unless a cue backs it; a shot is
    vetoed by the arm cue like a good moment; a sentence that claims no action is left alone."""
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    p = pkg.people[0]
    base = dict(camera="cam1", t_s=5.0, when="", what="", category=None)
    run = Moment(why="ボールを持って走った。", ref_id="e-run", kind=model.KIND_EVENT, **{**base, "what": "本人は走る。"})
    shot = Moment(why="ボールをシュートした。", ref_id="e-shot", kind=model.KIND_EVENT, **{**base, "what": "本人はシュートする。"})
    fell = Moment(why="地面に倒れた。", ref_id="e-fall", kind=model.KIND_EVENT, **{**base, "what": "地面に倒れる。"})
    p.events += [run, shot, fell]
    r = claims.apply(pkg.people, {"e-shot": {"wrist_above_head": -0.3, "pose_frames": 5}})
    assert run.weak and run.why.endswith("ているように見える（要確認）。") and run.claim["show"] == "hedge"
    assert shot.weak and shot.claim["show"] == "fallback" and "シュート" not in shot.why      # the arm cue vetoes it
    assert not fell.weak and fell.why == "地面に倒れた。"                                      # no action to check
    assert r["events"] >= 2
    d = player.player_data(pkg, "v.mp4")
    mine = next(q for q in d["people"] if q["id"] == p.subject_id)
    shown = {x["id"]: x for x in mine["events"]}
    assert shown["e-run"]["weak"] and shown["e-run"]["what"] == "" and not shown["e-fall"]["weak"]
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    assert "ボールを持って走った。" not in html and "ボールをシュートした。" not in html
    assert model.EVENT_WEAK_NOTE in html and model.EVENT_NOTE in html


def test_the_number_change_note_never_says_read_while_the_number_is_unverified(env):
    """Audit 4: CHANGE_NOTE said 「読み取った背番号が3番から5番に」 without a 「？」 while every other
    number on screen carries one (the gate is closed). It says what the AI saw until the gate opens."""
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    c = next(m for p in pkg.people for m in p.events if m.kind == model.KIND_CHANGE)
    assert not pkg.number_verified
    assert "3番？" in c.note and "5番？" in c.note and "読み取った" not in c.note and "要確認" in c.note
    settings2, repo2, aid2, sid2 = _group(env, people=2, events_each=1)
    _names_run(repo2, sid2, {"P1": _read("3", segments=[[1, 200, "3"], [201, 400, "5"]], dominant="3", share=0.6),
                             "P2": _read("4")}, {"P1": "T1", "P2": "T2"}, gate_open=True)
    pkg2 = collect.build_package(repo2, settings2, sid2)
    assert pkg2.number_verified
    c2 = next(m for p in pkg2.people for m in p.events if m.kind == model.KIND_CHANGE)
    assert "読み取った背番号が 3番 から 5番" in c2.note and "？" not in c2.note


def test_the_timeline_marks_have_a_touch_sized_hit_area_and_phones_get_taller_rows():
    html = player.TEMPLATE
    assert ".m:before{content:\"\";position:absolute;left:-11px;right:-11px;top:-11px;bottom:-11px}" in html   # 22px mark, 44px area
    assert "@media (pointer:coarse),(max-width:600px){.row,.nm{height:44px}}" in html
    assert "capY=Math.max(56," in html                                     # the caption stays above the browser's control bar


def _gecko():
    import shutil
    return shutil.which("geckodriver"), shutil.which("firefox")


@pytest.mark.skipif(not all(_gecko()), reason="needs geckodriver + firefox (headless)")
def test_in_a_real_browser_flags_keys_speed_and_the_failure_banner(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    m = next(m for p in pkg.people for m in p.moments)
    m.what, m.why = "ボールをシュートした（元の文）", "この場面に映っていた（行動の内容は要確認）"
    m.weak, m.why_orig = True, "ボールをシュートした。"
    import shutil, tempfile   # noqa: E401
    home_tmp = Path(tempfile.mkdtemp(prefix="pfbplayer_", dir=Path.home()))   # a snap Firefox cannot read /tmp
    try:
        _browser_check(player.build_player(pkg, home_tmp / "pl")["html"], m)
    finally:
        shutil.rmtree(home_tmp, ignore_errors=True)


def _browser_check(html_path, m):
    import os, subprocess, time   # noqa: E401
    import requests
    out = Path(html_path)
    (out.parent / "video.mp4").unlink()                           # the video cannot be loaded
    port = int(os.environ.get("PFB_TEST_GECKO_PORT", "8621"))
    gd = subprocess.Popen([_gecko()[0], "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    sess = None
    try:
        for _ in range(50):
            try:
                requests.get(base + "/status", timeout=1)
                break
            except requests.RequestException:
                time.sleep(0.2)
        caps = {"capabilities": {"alwaysMatch": {"browserName": "firefox", "moz:firefoxOptions": {"args": ["-headless"]}}}}
        sess = requests.post(base + "/session", json=caps).json()["value"]["sessionId"]

        def js(code, *a):
            return requests.post(f"{base}/session/{sess}/execute/sync", json={"script": code, "args": list(a)}).json()["value"]
        requests.post(f"{base}/session/{sess}/url", json={"url": "file://" + str(out)})
        time.sleep(2.5)
        assert js("return document.getElementById('vfail').classList.contains('on')") is True
        assert js("return document.getElementById('opts').open") is False
        # Cmd+C must not flip the comments switch; a plain C does
        assert js("var b=document.getElementById('cm-on'),a=b.checked;document.dispatchEvent(new KeyboardEvent('keydown',{key:'c',metaKey:true,bubbles:true}));return a===b.checked") is True
        assert js("var b=document.getElementById('cm-on'),a=b.checked;document.dispatchEvent(new KeyboardEvent('keydown',{key:'c',bubbles:true}));return a!==b.checked") is True
        # the weakened scene: the card never shows the original generated action ('したこと') and says 要確認 with a symbol
        pid = [p for p in js("return JSON.parse(document.getElementById('data').textContent).people") if any(x["id"] == m.ref_id for x in p["moments"])][0]["id"]
        js("window.__review.select(arguments[0]);window.__review.seek(arguments[1])", pid, m.t_s)
        time.sleep(.5)
        js("document.getElementById('morebtn') && document.getElementById('morebtn').click()")
        card = js("return document.getElementById('card').textContent")
        assert "ナイスミル上の名前" in card                                # the detail part is open
        assert "元の文" not in card and "▲" in card and "要確認" in card
        # a switch that sits in the closed settings shows its state in the summary line (text, not colour alone)
        assert js("return document.getElementById('optstate').textContent.indexOf('スポットライト')") == -1
        js("var b=document.getElementById('f-spot');b.checked=true;b.dispatchEvent(new Event('change',{bubbles:true}))")
        time.sleep(.5)
        assert "スポットライト" in js("return document.getElementById('optstate').textContent")
        # the person with the weakened scene gets no full generated paragraph in the list
        js("document.getElementById('list').open=true")
        assert "出しません" in js("return document.getElementById('listbody').textContent")
    finally:
        if sess:
            requests.delete(f"{base}/session/{sess}")
        gd.terminate()
        gd.wait(10)


# --- Chrome through the DevTools protocol (no chromedriver needed) --------------------------------
# Audit 2 found a Chrome-only failure (the saved picture) that a Firefox-only check had missed, and
# audit 3 added the Chrome checks by hand; this runs them as a (skippable) test.

def _chrome():
    import importlib.util
    import shutil
    exe = next((shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(n)), None)
    return exe if exe and importlib.util.find_spec("websockets") else None


class _Cdp:
    """A minimal DevTools-protocol client: one headless Chrome, one page."""

    def __init__(self, exe, port, profile):
        import json as _json
        import subprocess
        import time
        import urllib.request
        from websockets.sync.client import connect
        self.proc = subprocess.Popen([exe, "--headless=new", "--no-sandbox", "--disable-gpu", f"--remote-debugging-port={port}",
                                      f"--user-data-dir={profile}", "--autoplay-policy=no-user-gesture-required", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = None
        for _ in range(60):
            try:
                tabs = _json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1))
                url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:   # noqa: BLE001 - not up yet
                time.sleep(0.25)
        assert url, "Chrome did not start"
        self.ws, self.n, self._json = connect(url, max_size=None), 0, _json

    def cmd(self, method, **params):
        self.n += 1
        self.ws.send(self._json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            r = self._json.loads(self.ws.recv(timeout=30))
            if r.get("id") == self.n:
                return r

    def js(self, code):
        r = self.cmd("Runtime.evaluate", expression=code, returnByValue=True)
        assert "exceptionDetails" not in r["result"], r["result"]["exceptionDetails"]
        return r["result"]["result"].get("value")

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(10)


def _small_webm(path: Path, w: int = 320, h: int = 180, frames: int = 30) -> None:
    import av
    out = av.open(str(path), "w", format="webm")
    st = out.add_stream("libvpx", rate=10)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    for i in range(frames):
        for pk in st.encode(av.VideoFrame.from_ndarray(np.full((h, w, 3), i * 8 % 256, np.uint8), format="rgb24")):
            out.mux(pk)
    for pk in st.encode():
        out.mux(pk)
    out.close()


@pytest.mark.skipif(_chrome() is None, reason="needs Chrome (google-chrome / chromium) and the websockets package")
def test_in_chrome_touch_areas_phone_layout_banner_and_the_saved_picture(env, tmp_path):
    import os
    import shutil
    import tempfile
    import time
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    home_tmp = Path(tempfile.mkdtemp(prefix="pfbchrome_", dir=Path.home()))
    c = None
    try:
        html = Path(player.build_player(pkg, home_tmp / "pl")["html"])
        shape = _script_json(html.read_text(encoding="utf-8"))
        # a video every Chrome build can decode (the synthetic one is not), as long and as wide as the real one
        _small_webm(html.parent / "video.mp4", 320, round(320 * shape["size"][1] / shape["size"][0]) // 2 * 2,
                    frames=int(shape["video_s"] * 10) + 20)
        c = _Cdp(_chrome(), int(os.environ.get("PFB_TEST_CDP_PORT", "8623")), home_tmp / "profile")
        for width, height in ((1280, 1000), (390, 900)):
            c.cmd("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
            c.cmd("Page.navigate", url="file://" + str(html))
            time.sleep(2.5)
            c.js("document.getElementById('v').muted=true")
            assert c.js("document.documentElement.scrollWidth <= window.innerWidth + 1") is True      # no sideways scroll
            # the first-visit hints are one strip in the page flow: nothing floats over the picture
            assert c.js("document.getElementById('hints').classList.contains('on')") is True
            assert c.js("document.querySelectorAll('.bub').length") == 0
            r = c.js("(function(){var h=document.getElementById('hints').getBoundingClientRect(),s=document.getElementById('stage').getBoundingClientRect();return h.bottom<=s.top+1})()")
            assert r is True
            # a timeline mark looks 22px but is hit 20px away from its centre (a 44px touch area)
            c.js("window.__review.setZoom(32)")           # zoomed in: the marks are far enough apart to test one alone
            time.sleep(.3)
            hit = c.js("""(function(){var all=[].slice.call(document.querySelectorAll('.m')),best=null,bd=-1;
              all.forEach(function(b){var r=b.getBoundingClientRect(),d=1e9;all.forEach(function(o){if(o===b)return;var q=o.getBoundingClientRect();d=Math.min(d,Math.abs(q.left-r.left)+Math.abs(q.top-r.top))});if(d>bd){bd=d;best=b}});
              best.scrollIntoView({block:'center',inline:'center'});var r=best.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2,out=[];
              [[-20,0],[20,0],[0,-8],[0,8]].forEach(function(d){var e=document.elementFromPoint(x+d[0],y+d[1]);out.push(!!(e&&e.closest&&e.closest('.m')===best))});
              return {w:r.width,out:out,gap:bd}})()""")
            assert hit["w"] == 22 and all(hit["out"][:2]), hit                                       # left and right of the mark
            rowh = c.js("document.querySelector('.row').getBoundingClientRect().height")
            assert (rowh >= 44) if width == 390 else rowh >= 34
        # Cmd+C is the browser's own; a plain C flips the comments switch
        assert c.js("var b=document.getElementById('cm-on'),a=b.checked;document.dispatchEvent(new KeyboardEvent('keydown',{key:'c',metaKey:true,bubbles:true}));a===b.checked") is True
        assert c.js("var b=document.getElementById('cm-on'),a=b.checked;document.dispatchEvent(new KeyboardEvent('keydown',{key:'c',bubbles:true}));a!==b.checked") is True
        # the saved picture: the video frame of a local file taints the canvas in Chrome; the page must fall back
        # to boxes + text, say so, and never throw
        assert c.js("document.getElementById('v').readyState") >= 2, "the test video did not load"
        if True:
            c.js("window.__al=null;window.alert=function(m){window.__al=m};window.HTMLAnchorElement.prototype.click=function(){window.__dl=this.download}")
            c.js("document.getElementById('f-shot').click()")
            assert "枠と文だけ" in (c.js("window.__al") or "") and c.js("window.__dl").endswith("_boxes.png")
        assert c.js("document.getElementById('vmix').classList.contains('on')") is False      # the right video: no warning
        # at most two speech bubbles at once, however many people have a moment right now
        dense = _script_json(html.read_text(encoding="utf-8"))
        for p in dense["people"]:
            for m in p["moments"][:1]:
                m["t"] = 10.0
        assert sum(1 for p in dense["people"] if not p["excluded"] and any(m["t"] == 10.0 for m in p["moments"])) >= 3
        page = re.sub(r'(<script id="data" type="application/json">).*?(</script>)',
                      lambda mo: mo.group(1) + player._json_for_script(dense) + mo.group(2), html.read_text(encoding="utf-8"), flags=re.S)
        (html.parent / "dense.html").write_text(page, encoding="utf-8")
        c.cmd("Page.navigate", url="file://" + str(html.parent / "dense.html"))
        time.sleep(2.5)
        c.js("document.getElementById('cm-sel').checked=false;window.__review.seek(10)")
        time.sleep(1)
        assert 1 <= c.js("window.__review.state().comments") <= 2
        # another video put next to the page (a different shape): the page says the boxes will not fit
        _small_webm(html.parent / "video.mp4", 240, 320)
        c.cmd("Page.navigate", url="file://" + str(html))
        time.sleep(2.5)
        assert c.js("document.getElementById('vmix').classList.contains('on')") is True
        # the video cannot be loaded: the banner says so
        (home_tmp / "pl" / "video.mp4").unlink()
        c.cmd("Page.navigate", url="file://" + str(html))
        time.sleep(2.5)
        assert c.js("document.getElementById('vfail').classList.contains('on')") is True
    finally:
        if c:
            c.close()
        shutil.rmtree(home_tmp, ignore_errors=True)


# --- the command line (a plain message and nothing written when the arguments are wrong) -------------

def _cli_settings(settings, tmp_path) -> Path:
    import yaml
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir)}), encoding="utf-8")
    return path


def test_cli_wrong_arguments_say_why_and_write_nothing(env, tmp_path, capsys):
    from pfb_app.review_video.__main__ import main
    settings, repo, sid = _numbered(env)
    cfg = _cli_settings(settings, tmp_path)
    out = tmp_path / "out"
    cases = [([str(sid + 99), "--settings", str(cfg), "--out", str(out)], f"ID {sid + 99}"),
             ([str(sid), "--settings", str(tmp_path / "none.yaml"), "--out", str(out)], "--settings"),
             ([str(sid), "--settings", str(cfg), "--out", str(out), "--claim-cues", str(tmp_path / "none.json")], "--claim-cues"),
             ([str(sid), "--settings", str(cfg), "--out", str(out), "--ball", str(tmp_path / "none.json")], "--ball"),
             ([str(sid), "--settings", str(cfg), "--out", str(cfg)], "フォルダ")]
    for argv, word in cases:
        assert main(argv) == 2
        err = capsys.readouterr().err
        assert word in err and "Traceback" not in err, (argv, err)
        assert not out.exists()                       # nothing was created
    # an unknown person: said so, nothing made, still no folder
    assert main([str(sid), "--settings", str(cfg), "--out", str(out), "--subject", "9999"]) == 1
    cap = capsys.readouterr()
    assert "ID 9999" in cap.out and "作れたものがありません" in cap.err and not out.exists()


def test_cli_says_when_the_action_check_was_not_run_and_what_an_old_folder_still_holds(env, tmp_path):
    settings, repo, sid = _numbered(env)
    out = tmp_path / "out"
    first = service.make_all(repo, settings, sid, out)
    assert model.NO_CUES_NOTE in first["notes"]          # no cues: sentences are the AI's, and it says so
    one = first["videos"][0]["subject_id"]
    again = service.make_all(repo, settings, sid, out, subject_ids=[one], with_player=False)
    assert any("前回までの動画が残っています" in n for n in again["notes"])
    assert any("player/ は前回のまま" in n for n in again["notes"])
    cues = tmp_path / "cues.json"
    cues.write_text("{}", encoding="utf-8")
    checked = collect.build_package(repo, settings, sid, claim_cues=str(cues))
    assert model.NO_CUES_NOTE not in checked.notes and any(n.startswith("動作の確認") for n in checked.notes)



# --- everyone's evaluation text, grouped by team (2026-11-18) ----------------------------------------

def _scene(ref, why, *, weak=False, suspect=False, status="unchecked", t=10.0):
    m = Moment(ref, "cam1", t, "", why, why, None, False, suspect=suspect)
    m.weak, m.claim = weak, {"status": status, "action": claims.cc.claimed_action(why), "show": "keep", "reasons": []}
    return m


def test_the_text_may_only_come_from_checked_scenes():
    p = Person(1, "選手1", {"cam1": ["P1"]}, "draft", None)
    p.moments = [_scene("a", "ボールをパスした。"), _scene("b", "この場面に映っていた（行動の内容は要確認）", weak=True),
                 _scene("c", "別の人かもしれないドリブルをした。", suspect=True), _scene("d", "ボールを放った。"),
                 _scene("e", "ボールをシュートした。", status="supported")]
    got = [m.ref_id for m in claims.eligible(p)]
    assert got == ["a", "e"]                    # weakened, flagged and an unverified shot ("放った" is a shot) are out


def test_the_text_guard_hedges_what_no_checked_scene_says_and_never_compares():
    p = Person(1, "選手1", {"cam1": ["P1"]}, "draft", None)
    p.moments = [_scene("a", "ボールを別の参加者にパスした。"), _scene("b", "バスケットに向かって走った。")]
    scenes = claims.eligible(p)
    text = ("動画の約1分ごろ、ボールを別の参加者にパスして、チームに貢献しました。"
            "動画の約2分ごろ、ジャンプしてシュートを決めました。"
            "チームで一番よく走りました。3番がパスを受けました。")
    out, changes = claims.guard_text(text, scenes)
    assert "ボールを別の参加者にパスして、チームに貢献しました。" in out          # a checked action stays as written
    assert "シュートを決めました。" not in out
    shot = [s for s in out.split("。") if "シュート" in s]
    assert shot and all("要確認" in s for s in shot)                       # the shot is never a fact
    assert "一番" not in out and "チームで" not in out                      # comparison dropped
    assert "3番？" in out and "3番が" not in out                            # a number always carries 「？」
    assert len(changes) >= 2
    # a shot is only stated when a verified shot scene exists
    p.moments.append(_scene("s", "ボールをシュートした。", status="supported"))
    out2, _ = claims.guard_text("ボールをシュートしました。", claims.eligible(p))
    assert out2 == "ボールをシュートしました。"
    # a result no scene states is hedged
    out3, _ = claims.guard_text("得点しました。", claims.eligible(p))
    assert "要確認" in out3


def _eval_package(env):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    return pkg


def test_every_player_gets_a_text_or_a_reason_and_the_player_groups_them_by_team(env, tmp_path):
    pkg = _eval_package(env)
    for p, letter in zip(pkg.people, ["A", "B", "A", None, "B"]):
        p.team_letter, p.excluded = letter, False
    pkg.people[4].excluded = True
    data = {}
    for p in pkg.people[:3]:
        data[str(p.subject_id)] = {"text": "ボールを別の参加者にパスして、チームに貢献しました。", "source": "ai"}
    data[str(pkg.people[3].subject_id)] = {"text": "バスケットボールに参加している様子が、動画に映っていました。", "source": "short"}
    r = claims.apply_eval_texts(pkg.people, data)
    assert r["ai"] == 3 and r["short"] == 1 and r["none"] == []
    last = pkg.people[4]
    assert last.eval_text is None and "選手以外" in last.eval_why        # set apart: no text, and it says why
    d = player.player_data(pkg, "video.mp4")
    assert d["eval_on"] is True
    mine = d["people"][0]
    assert mine["eval"]["source"] == "ai" and mine["text"] is None and not mine["text_withheld"]
    assert d["people"][3]["eval"]["note"] == model.EVAL_SHORT_NOTE
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    assert 'id="evalbox"' in html and "function buildEval" in html and 'className="tgrp"' in html
    assert 'var TEAMS=[["A",STY?' in html and '[null,"チーム不明"]]' in html   # a fixed order: nobody is ranked
    assert "sort(function(a,b){return a.id-b.id})" in html                  # inside a team: by id, never by score


def test_a_player_without_a_text_says_so_and_the_old_paragraph_is_unchanged(env, tmp_path):
    pkg = _eval_package(env)
    d = player.player_data(pkg, "video.mp4")
    assert d["eval_on"] is False and all(q["eval"] is None for q in d["people"])   # without --eval-texts nothing changes
    r = claims.apply_eval_texts(pkg.people, {})
    assert len(r["none"]) == len([p for p in pkg.people if not p.excluded])
    assert all(p.eval_text is None and p.eval_why for p in pkg.people if not p.excluded)


def test_the_eval_texts_option_needs_an_existing_file(tmp_path, capsys):
    from pfb_app.review_video import __main__ as cli
    code = cli.main(["1", "--out", str(tmp_path / "o"), "--eval-texts", str(tmp_path / "nope.json")])
    assert code == 2 and "--eval-texts" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_eval_texts_are_neutral_words(env, tmp_path):
    pkg = _eval_package(env)
    claims.apply_eval_texts(pkg.people, {str(p.subject_id): {"text": "ボールをパスして、チームに貢献しました。", "source": "ai"} for p in pkg.people})
    text = all_screen_text(pkg, pkg.people[0], tmp_path)
    assert not [w for w in SCHOOL_WORDS if w in text]
    for w in (model.EVAL_AI_NOTE, model.EVAL_SHORT_NOTE, model.EVAL_NONE_NOTE):
        assert not [x for x in SCHOOL_WORDS if x in w] and "ランキング" not in w and "順位" not in w


def test_evalgen_works_on_a_copy_and_only_switches_off_the_scenes_that_did_not_pass(env, tmp_path):
    import sqlite3
    from pfb_app.review_video import evalgen
    from pfb_app.storage import files
    settings, repo, sid = _numbered(env)
    cam = repo.list_cameras(sid)[0]
    before = repo.latest_run(sid, Stage.contribution, cam).output["judgments"]
    positive = [j["ref_id"] for j in before if j["is_positive_contribution"]]
    assert len(positive) >= 2
    dst = evalgen._copy_db(Path(files.db_path(settings)), tmp_path / "work")
    assert dst != Path(files.db_path(settings)) and oct(dst.parent.stat().st_mode & 0o777) == "0o700"
    conn = evalgen.dbm.connect(dst)
    crepo = evalgen.dbm.Repo(conn, settings)
    n = evalgen._keep_only(crepo, sid, {positive[0]})
    now = crepo.latest_run(sid, Stage.contribution, cam).output["judgments"]
    assert n == len(positive) - 1
    assert [j["ref_id"] for j in now if j["is_positive_contribution"]] == [positive[0]]
    conn.close()
    # the database the scenes came from is untouched
    assert [j["ref_id"] for j in repo.latest_run(sid, Stage.contribution, cam).output["judgments"]
            if j["is_positive_contribution"]] == positive


def test_everyone_is_framed_in_the_kit_colour_with_a_hedged_always_on_bubble_and_a_red_court_line(env, tmp_path, monkeypatch):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    html = Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8")
    d = _script_json(html)
    script = html.split("<script>")[1]
    assert 'id="tags-on" type="checkbox" checked' in html and "全員の吹き出し" in html and "drawTags(" in script
    assert "コートの枠（推定）" in html and "#E02020" in script and "F0E442" not in html          # the court's line is red now
    # the words: hedged, no basketball action
    assert d["act_words"][0] == "" and all(w.endswith("見える") or w.endswith("（推定）") for w in d["act_words"][1:])
    assert not any(x in w for w in d["act_words"] for x in ("シュート", "パス", "得点", "ゴール", "ドリブル"))
    # the bubble's number: 「N番？」 while the gate is closed, else 「番号不明」; the tag text comes from the person's tag
    assert 'e.q?e.q.tag:"番号不明"' in script and not pkg.number_verified
    assert all(("番？" in p["tag"]) or ("番号不明" in p["tag"]) for p in d["people"])
    if d["every"]:
        assert d["team_style"] and d["team_style"]["note"] and "推定" in d["team_style"]["note"]
        extra_cols = 7 + bool(d["court_out"])          # the last column is the outside-the-court flag, only with a court
        assert all(len(r) == extra_cols for r in d["extras"] or [[0] * extra_cols])
        assert all(len(r) == extra_cols - 1 for rows in d["tracks"].values() for r in rows)
    # an analysis that fails leaves the plain player (nothing invented)
    pkg2 = collect.build_package(repo, settings, sid, credit_lines=CREDIT)
    monkeypatch.setattr(player.everyone, "analyse", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no picture")))
    d2 = _script_json(Path(player.build_player(pkg2, tmp_path / "p2")["html"]).read_text(encoding="utf-8"))
    assert d2["every"] is False and d2["team_style"] is None and d2["court_est"] is None
    assert all(len(r) == 5 for rows in d2["tracks"].values() for r in rows)


def test_a_court_estimate_is_only_drawn_when_it_is_asked_for_and_nothing_is_registered(env, tmp_path):
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid, estimate_court=False)
    assert pkg.court_est is False
    pkg = collect.build_package(repo, settings, sid)
    assert pkg.court_est == (pkg.court is None)
    d = _script_json(Path(player.build_player(pkg, tmp_path / "pl")["html"]).read_text(encoding="utf-8"))
    assert d["court_est"] is None or ("推定" in d["court_est"]["text"] and d["court_est"]["samples"])
