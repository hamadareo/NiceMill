"""「この人のデータを消す」 (services.erasure): a deletion request for one person, across the
works linked in 成長の記録. Synthetic works only (generated videos, invented names)."""
import json
from pathlib import Path

import pytest
from pdfhelp import pdf_text

from pfb_app.domain.models import Stage
from pfb_app.export import pdf
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import erasure, kept
from pfb_app.storage import files
from pdf_synthetic import document_work, video_group, work
from test_wizard_ui_basic import env  # noqa: F401 (env)

NAME = "山田花子"


def two_works(settings, repo):
    """Two tracked works with 山田花子 in both, linked as one person in 成長の記録."""
    first = work(repo, title="春の試合")
    s1 = video_group(settings, repo, first, people=2, seconds=20, label="A")
    second = work(repo, title="秋の試合")
    s2 = video_group(settings, repo, second, people=2, seconds=20, label="B")
    a = next(s for s in repo.list_subjects(s1) if s["display_name"] == NAME)
    b = next(s for s in repo.list_subjects(s2) if s["display_name"] == NAME)
    person = repo.create_growth_person(NAME)
    repo.link_growth(a["id"], person, "manual")
    repo.link_growth(b["id"], person, "manual")
    return (first, s1, a), (second, s2, b), person


def _crop(settings, aid, sid, name):
    path = files.submission_dir(settings, aid, sid) / "people" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpg")
    return path


def test_the_plan_lists_every_linked_work(env):
    settings, repo = env
    (first, _, a), (second, _, b), _ = two_works(settings, repo)
    plan = erasure.plan(repo, a["id"])
    assert [(i.title, i.name) for i in plan.items] == [("春の試合", NAME), ("秋の試合", NAME)]
    assert not plan.busy
    lines = erasure.what_goes(plan)
    assert any("フィードバックの文章" in line for line in lines)
    assert any("成長の記録" in line for line in lines)
    assert erasure.plan(repo, b["id"]).items == plan.items


def test_erase_a_tracked_person_in_two_works(env):
    settings, repo = env
    (first, s1, a), (second, s2, b), person = two_works(settings, repo)
    other = next(s for s in repo.list_subjects(s1) if s["id"] != a["id"])
    other_text = repo.latest_run(s1, Stage.pfb, subject_id=other["id"]).output["pfb"]
    kept.save(repo, first)
    track = repo.latest_run(s1, Stage.track, "cam1").id
    mine = _crop(settings, first, s1, f"cam1_P1_t{track}.jpg")   # 山田花子 is P1 of this track
    theirs = _crop(settings, first, s1, f"cam1_P2_t{track}.jpg")
    other_track = _crop(settings, first, s1, f"cam1_P1_t{track + 1000}.jpg")   # another run's P1
    cache = files.submission_dir(settings, first, s1) / "handout" / "pictures" / "x.jpg"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"jpg")
    pa = a["participant_id"]
    done = erasure.erase(repo, settings, a["id"])
    assert len(done.items) == 2
    for subject in (repo.get_subject(a["id"]), repo.get_subject(b["id"])):
        assert subject["display_name"] == erasure.ERASED_NAME and subject["erased_at"]
        assert subject["approval_status"] == "none" and not is_evaluated(subject)
        assert subject["appearance"] is None and subject["roster"] == {}
        assert repo.run_history(subject["submission_id"], Stage.pfb,
                                subject_id=subject["id"]) == []
    # the scenes about them in the shared analysis are gone; the others' stay
    beh = repo.latest_run(s1, Stage.behavior, "cam1")
    assert beh.output["events"] and all(e["participant"] != pa for e in beh.output["events"])
    kept_ids = {e["event_id"] for e in beh.output["events"]}
    assert set(beh.output["frames"]) <= kept_ids and set(beh.output["source_text"]) <= kept_ids
    con = repo.latest_run(s1, Stage.contribution, "cam1")
    assert {j["ref_id"] for j in con.output["judgments"]} == kept_ids
    assert repo.latest_run(s1, Stage.pfb, subject_id=other["id"]).output["pfb"] == other_text
    # files: their crop and the group's picture cache; the saved PDFs lose their pages
    assert not mine.exists() and theirs.exists() and not cache.exists()
    assert other_track.exists()
    # the name is nowhere in the group's runs any more
    for output, warnings in repo.conn.execute(
            "SELECT output, warnings FROM stage_runs WHERE submission_id=?", (s1,)):
        assert NAME not in (output or "") and NAME not in warnings
    record = kept.load(settings, first)
    assert [p["subject_id"] for p in record.people] == [other["id"]]
    assert record.evaluator is None
    # 成長の記録: the links and the record (now empty) are gone
    assert repo.growth_person_of(a["id"]) is None and repo.growth_person(person) is None
    # PDFs made now hold the others only
    text = pdf_text(pdf.render_student_pdf(repo, first))
    assert NAME not in text and other["display_name"] in text
    # the audit names nobody
    entries = [json.loads(r[0]) for r in repo.conn.execute(
        "SELECT detail FROM audit_log WHERE action='erase_person'")]
    assert len(entries) == 2 and NAME not in json.dumps(entries, ensure_ascii=False)
    # they cannot be brought back
    repo.set_excluded(a["id"], False)
    assert not is_evaluated(repo.get_subject(a["id"]))
    assert all(p["subject_id"] != a["id"] for p in erasure.people(repo))


def test_erase_a_document_person_removes_their_group(env):
    settings, repo = env
    aid = document_work(repo)
    sid = repo.list_submissions(aid)[0]["id"]
    subject = repo.list_subjects(sid)[0]
    folder = files.submission_dir(settings, aid, sid)
    (folder / "source.pdf").write_bytes(b"%PDF")
    (folder / "pages").mkdir()
    (folder / "pages" / "p1.png").write_bytes(b"png")
    repo.cache_put("h1", "m", "{}", 1, 1, 0.1, None)
    repo.conn.execute("INSERT INTO call_log_owner (request_hash, submission_id) VALUES ('h1', ?)",
                      (sid,))
    erasure.erase(repo, settings, subject["id"])
    assert repo.list_submissions(aid) == []
    assert not folder.exists()
    assert repo.conn.execute("SELECT COUNT(*) FROM call_log").fetchone()[0] == 0
    assert repo.get_assignment(aid)["title"] == "防災マップ"


def test_nothing_changes_while_a_linked_work_is_analysed(env):
    settings, repo = env
    (_, _, a), (_, s2, b), person = two_works(settings, repo)
    repo.enqueue(s2, Stage.pfb, "cpu")
    assert erasure.plan(repo, a["id"]).busy
    with pytest.raises(erasure.Busy):
        erasure.erase(repo, settings, a["id"])
    assert repo.get_subject(a["id"])["display_name"] == NAME
    assert repo.growth_person_of(b["id"]) == person


def test_the_people_to_choose_from(env):
    settings, repo = env
    (first, s1, a), _, _ = two_works(settings, repo)
    doc = document_work(repo)
    people = erasure.people(repo)
    assert {p["title"] for p in people} == {"春の試合", "秋の試合", "防災マップ"}
    one = next(p for p in people if p["subject_id"] == a["id"])
    assert one["name"] == NAME and one["linked"] == 2
    assert erasure.label(one) == f"{NAME}（春の試合ほか1件）"
    assert erasure.label(next(p for p in people if p["assignment_id"] == doc)) == \
        "山田花子（防災マップ）"


def test_an_erased_person_is_not_brought_back_by_the_people_selection(env):
    from pfb_app.services import auto_people
    settings, repo = env
    (first, s1, a), _, _ = two_works(settings, repo)
    erasure.erase(repo, settings, a["id"])
    for _ in range(3):   # again and again: the placeholder stays, nobody new is made
        auto_people.auto_identity(repo, s1, "cam1")
        ident = repo.latest_run(s1, Stage.identity, "cam1")
        mapping = {p["label"]: p["participant_id"] for p in ident.output["participants"]}
        assert mapping["P1"] == a["participant_id"]
        evaluated = {s["participant_id"] for s in repo.list_subjects(s1) if is_evaluated(s)}
        assert len(evaluated) == 1   # the other person only
        assert repo.get_subject(a["id"])["erased_at"]


def test_a_corrected_swap_never_touches_the_other_person(env):
    """Review HIGH 1: after the coach swapped P1 and P2, erasing 山田花子 (now P2) leaves
    鈴木一郎's crop and names entries (P1) alone, and the people selection still works."""
    from pfb_app.services import auto_people
    from pfb_app.services.review import save_tracked_identity
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    suzuki = next(s for s in repo.list_subjects(sid) if s["id"] != yamada["id"])
    track = repo.latest_run(sid, Stage.track, "cam1").id
    # a failed mapping run (never counts), then the coach's corrected swap
    failed = save(repo, sid, Stage.identity, {"participants": [
        {"participant_id": yamada["participant_id"], "label": "P1", "display_name": None}],
        "labels": ["P1", "P2"], "track_run_id": track}, camera="cam1", upstream=[track])
    repo.conn.execute("UPDATE stage_runs SET status='failed' WHERE id=?", (failed.id,))
    save_tracked_identity(repo, sid, "cam1", {"P1": suzuki["participant_id"],
                                              "P2": yamada["participant_id"]})
    p1 = _crop(settings, aid, sid, f"cam1_P1_t{track}.jpg")
    p2 = _crop(settings, aid, sid, f"cam1_P2_t{track}.jpg")
    save(repo, sid, Stage.names, {"track_run_id": track, "teams": {"P1": "T1", "P2": "T2"}},
         camera="cam1", upstream=[track])
    erasure.erase(repo, settings, yamada["id"])
    assert p1.exists() and not p2.exists()
    names = repo.latest_run(sid, Stage.names, "cam1").output
    assert names["teams"] == {"P1": "T1"}
    auto_people.auto_identity(repo, sid, "cam1")      # no 「同じ参加者が複数の人物に…」
    mapping = {p["label"]: p["participant_id"] for p in
               repo.latest_run(sid, Stage.identity, "cam1").output["participants"]}
    assert mapping == {"P1": suzuki["participant_id"], "P2": yamada["participant_id"]}


def test_the_engine_s_own_records_of_the_person_go_and_the_others_stay(env):
    """Review MED 5 / R27, R30: the current rows file's events, the pictures only their scenes
    use and the current sheet's files named after their label go; the others' stay in place,
    and the position records (tracks/) are never edited."""
    from pfb_app.export import sheets
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    other = next(s for s in repo.list_subjects(sid) if s["id"] != yamada["id"])
    beh = repo.latest_run(sid, Stage.behavior, "cam1").output
    shots = {e["participant"]: [] for e in beh["events"]}
    for e in beh["events"]:
        shots[e["participant"]] += [Path(p) for p in beh["frames"][e["event_id"]]]
    before = [(m.ref_id, m.category) for m in sheets.person_moments(repo, other["id"])]
    eng = files.submission_dir(settings, aid, sid) / "pfb3x3"
    vid = f"s{sid}_cam1"
    (eng / "sheets" / vid).mkdir(parents=True)
    (eng / "sheets" / vid / "people_sheet.jpg").write_bytes(b"x")
    track = repo.latest_run(sid, Stage.track, "cam1")
    repo.conn.execute("UPDATE stage_runs SET output=json_set(output, '$.sheet', ?) WHERE id=?",
                      (str(eng / "sheets" / vid / "people_sheet.jpg"), track.id))
    (eng / "sheets" / vid / "P1.jpg").write_bytes(b"x")
    (eng / "sheets" / vid / "P12.jpg").write_bytes(b"x")
    (eng / "tracks" / vid / "raw_appearance.json").write_text(
        json.dumps({"P1": "赤い服", "P2": "青い服"}), encoding="utf-8")
    erasure.erase(repo, settings, yamada["id"])
    rows = next(eng.glob(f"exp/*/*/{vid}/behavior.engine.jsonl")).read_text(encoding="utf-8")
    events = [e for line in rows.splitlines() for e in json.loads(line)["events"]]
    assert all(e.get("label") != "P1" for e in events)
    assert any(e.get("erased") for e in events) and any(e.get("label") == "P2" for e in events)
    assert not any(p.exists() for p in shots[yamada["participant_id"]])
    assert all(p.exists() for p in shots[other["participant_id"]])
    assert not (eng / "sheets" / vid / "P1.jpg").exists()
    assert (eng / "sheets" / vid / "P12.jpg").exists()
    assert json.loads((eng / "tracks" / vid / "raw_appearance.json").read_text()) == \
        {"P1": "赤い服", "P2": "青い服"}                     # position records stay (KEEPS)
    assert [(m.ref_id, m.category) for m in sheets.person_moments(repo, other["id"])] == before


def test_names_in_other_people_s_texts_are_disclosed(env):
    from pfb_app.services import review
    from pdf_synthetic import save
    settings, repo = env
    (first, s1, a), _, _ = two_works(settings, repo)
    other = next(s for s in repo.list_subjects(s1) if s["id"] != a["id"])
    run = save(repo, s1, Stage.pfb, {"pfb_status": "generated",
                                     "pfb": f"{NAME}さんにパスを出していました。",
                                     "evidence_refs": []}, subject_id=other["id"])
    repo.set_approval(other["id"], run.id)
    found = erasure.mentions(repo, erasure.plan(repo, a["id"]))
    assert found == [f"「春の試合」の{other['display_name']}さんの文章に「{NAME}」が出てきます。"]
    erasure.erase(repo, settings, a["id"])
    assert NAME in repo.get_run(run.id).output["pfb"]      # never edited automatically


def test_a_window_picture_other_scenes_use_stays(env):
    """Review R26: one overlay picture shared by a scene of each person stays; a picture only
    the erased person's scenes use goes."""
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    beh = repo.latest_run(sid, Stage.behavior, "cam1")
    out = beh.output
    mine = [e["event_id"] for e in out["events"] if e["participant"] == yamada["participant_id"]]
    theirs = [e["event_id"] for e in out["events"] if e["participant"] != yamada["participant_id"]]
    shared = Path(out["frames"][theirs[0]][0])
    only_mine = Path(out["frames"][mine[1]][0])
    out["frames"][mine[0]] = [str(shared)]          # the same window picture for both
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (json.dumps(out, ensure_ascii=False), beh.id))
    erasure.erase(repo, settings, yamada["id"])
    assert shared.exists() and not only_mine.exists()


def _rows_events(eng, sid):
    rows = next(eng.glob(f"exp/*/*/s{sid}_cam1/behavior.engine.jsonl"))
    return rows, [e for line in rows.read_text(encoding="utf-8").splitlines()
                  for e in json.loads(line)["events"]]


def test_rows_made_from_an_older_track_use_that_track_s_labels(env):
    """Review R27 / final HIGH 1: after a re-track that numbered the people again (no new
    behavior run), the rows file's labels mean the OLD numbering: it is edited with the labels
    the old track run's mapping gives the person (P1 there), not the new ones (P2)."""
    from pfb_app.services.review import save_tracked_identity
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    suzuki = next(s for s in repo.list_subjects(sid) if s["id"] != yamada["id"])
    old = repo.latest_run(sid, Stage.track, "cam1")
    save(repo, sid, Stage.track, dict(old.output), camera="cam1", upstream=old.upstream_run_ids)
    save_tracked_identity(repo, sid, "cam1", {"P1": suzuki["participant_id"],
                                              "P2": yamada["participant_id"]})
    eng = files.submission_dir(settings, aid, sid) / "pfb3x3"
    _, before = _rows_events(eng, sid)
    assert any(e.get("label") == "P1" for e in before)
    plan = erasure.plan(repo, yamada["id"])
    assert plan.engine_skipped == [] and erasure.engine_note(plan) == []
    done = erasure.erase(repo, settings, yamada["id"])
    _, after = _rows_events(eng, sid)
    assert all(e.get("label") != "P1" for e in after)          # yamada in the old numbering
    assert any(e.get("label") == "P2" for e in after)          # suzuki in the old numbering
    assert done.engine_skipped == []


def test_rows_after_a_re_track_without_a_new_mapping_are_edited(env):
    """Final review HIGH 1 (the reviewer's scenario): a re-track where the person has no
    mapping on the newest track run (the usual state after the automatic people): the rows of
    the behavior run, made from the old track run, are edited with the old labels."""
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    old = repo.latest_run(sid, Stage.track, "cam1")
    save(repo, sid, Stage.track, dict(old.output), camera="cam1", upstream=old.upstream_run_ids)
    eng = files.submission_dir(settings, aid, sid) / "pfb3x3"
    plan = erasure.plan(repo, yamada["id"])
    assert plan.engine_skipped == []
    done = erasure.erase(repo, settings, yamada["id"])
    _, after = _rows_events(eng, sid)
    assert all(e.get("label") != "P1" for e in after) and any(e.get("erased") for e in after)
    assert done.engine_skipped == []


def test_rows_whose_track_does_not_map_the_person_are_left_and_said(env):
    """A behavior run made from a track run where the person has no mapping: its rows cannot
    be edited (which label is theirs is unknown there); the confirm and the result say so."""
    from pfb_app.services.review import save_tracked_identity
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    suzuki = next(s for s in repo.list_subjects(sid) if s["id"] != yamada["id"])
    old = repo.latest_run(sid, Stage.track, "cam1")
    new = save(repo, sid, Stage.track, dict(old.output), camera="cam1",
               upstream=old.upstream_run_ids)
    save_tracked_identity(repo, sid, "cam1", {"P1": suzuki["participant_id"], "P2": None})
    beh = repo.latest_run(sid, Stage.behavior, "cam1")
    save(repo, sid, Stage.behavior, dict(beh.output), camera="cam1",
         upstream=[u for u in beh.upstream_run_ids if u != old.id] + [new.id])
    eng = files.submission_dir(settings, aid, sid) / "pfb3x3"
    rows, _ = _rows_events(eng, sid)
    before = rows.read_text(encoding="utf-8")
    plan = erasure.plan(repo, yamada["id"])
    assert plan.engine_skipped == ["春の試合"]
    assert any("番号をつけ直した" in line for line in erasure.engine_note(plan))
    done = erasure.erase(repo, settings, yamada["id"])
    assert rows.read_text(encoding="utf-8") == before
    assert done.engine_skipped == ["春の試合"]


def _linecall(settings, repo, aid, sid, source="engine"):
    """A linecall run of cam1 (upstream: the current track run) with an episode of P1 and one
    of P2 in episodes.json, each with its crop, overlay and clip."""
    from pdf_synthetic import save
    from pfb_app.pipeline.linecall import linecall_dir
    d = linecall_dir(settings, aid, sid, "cam1")
    eps = []
    for lb in ("P1", "P2"):
        eid = f"{lb}-arc-10"
        rels = {"crop": f"crops/{eid}/12.jpg", "overlay": f"crops/{eid}/12_overlay.jpg",
                "clip": f"clips/{eid}.mp4"}
        for rel in rels.values():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(b"x")
        eps.append({"id": eid, "label": lb, "line": "arc", "verdict": "undecidable", **rels})
    (d / "episodes.json").write_text(json.dumps({"status": "判定材料", "source": source,
                                                 "episodes": eps}), encoding="utf-8")
    track = repo.latest_run(sid, Stage.track, "cam1")
    save(repo, sid, Stage.linecall, {"episodes_json": str(d / "episodes.json"),
                                     "episodes": 2, "verdicts": {}},
         camera="cam1", upstream=[track.id])
    return d


def test_the_person_s_own_line_call_pictures_and_clips_go(env):
    """Final review LOW 8: the erased person's 判定支援 crops, overlays and clips (episodes of
    their label on the linecall run's track run) go; the others' stay; KEEPS says what stays."""
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    d = _linecall(settings, repo, aid, sid)
    erasure.erase(repo, settings, yamada["id"])
    assert not any((d / "crops" / "P1-arc-10").glob("*.jpg"))
    assert not (d / "clips" / "P1-arc-10.mp4").exists()
    assert (d / "crops" / "P2-arc-10" / "12.jpg").exists()
    assert (d / "clips" / "P2-arc-10.mp4").exists()
    assert any("判定支援の判定材料" in line for line in erasure.KEEPS)


def test_line_call_material_under_jersey_labels_stays(env):
    """A TrackID3x3 linecall run (labels A6 / B3, not the app's P-labels) cannot be tied to
    the person: its files stay (KEEPS discloses it)."""
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    d = _linecall(settings, repo, aid, sid, source="tid")
    erasure.erase(repo, settings, yamada["id"])
    assert (d / "clips" / "P1-arc-10.mp4").exists() and (d / "clips" / "P2-arc-10.mp4").exists()


def test_nothing_is_queued_while_the_files_are_edited(env, monkeypatch):
    """Review R29: the group's ERASING stamp holds enqueue_once off until the files are done."""
    from pfb_app.pipeline.runner import enqueue_once
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    seen = []
    real = erasure._video_files

    def video_files(settings, item, found):
        seen.append(enqueue_once(repo, sid, Stage.names, "cam1"))
        return real(settings, item, found)
    monkeypatch.setattr(erasure, "_video_files", video_files)
    erasure.erase(repo, settings, yamada["id"])
    assert seen == [None]
    assert "erasing" not in repo.get_submission(sid)["meta"]
    assert enqueue_once(repo, sid, Stage.names, "cam1") is not None


def test_an_erasure_stopped_before_its_files_is_finished_by_the_worker(env, monkeypatch):
    """Review round 3: a crash between the rows and the files; the next worker pass removes
    the files (also the person's saved PDF and phone page) and clears the stamp."""
    from test_worker import StubGuard
    from pfb_app.worker.main import worker_loop
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    yamada = next(s for s in repo.list_subjects(sid) if s["display_name"] == NAME)
    kept.save(repo, aid)
    track = repo.latest_run(sid, Stage.track, "cam1").id
    crop = _crop(settings, aid, sid, f"cam1_P1_t{track}.jpg")
    folder = kept.folder(settings, aid)

    def crash(*a, **k):
        raise KeyboardInterrupt("the process stops here")
    monkeypatch.setattr(erasure, "_file_phase", crash)
    with pytest.raises(KeyboardInterrupt):
        erasure.erase(repo, settings, yamada["id"])
    monkeypatch.undo()
    assert repo.get_subject(yamada["id"])["erased_at"]          # the rows went
    assert crop.exists() and (folder / f"{yamada['id']}.pdf").exists()
    repo.conn.execute("UPDATE submissions SET meta=json_set(meta, '$.erasing', ?) WHERE id=?",
                      ("2026-01-01T00:00:00.000000+00:00", sid))   # a real crash leaves it
    assert erasure.unfinished(repo) == []                       # within the grace period
    repo.conn.execute("UPDATE audit_log SET ts='2026-01-01T00:00:00.000000+00:00'"
                      " WHERE action='erase_person'")
    assert [r["subject_id"] for r in erasure.unfinished(repo)] == [yamada["id"]]
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    assert not crop.exists()
    assert not (folder / f"{yamada['id']}.pdf").exists()
    assert not (folder / f"{yamada['id']}.html").exists()
    assert erasure.unfinished(repo) == []
    assert "erasing" not in repo.get_submission(sid)["meta"]


def test_the_erasing_stamp_holds_off_names_and_retries(env):
    from pfb_app.domain.models import JobStatus
    from pfb_app.pipeline.runner import request_names
    from pfb_app.storage.db import _now
    settings, repo = env
    aid = work(repo, title="春の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    job = repo.enqueue(sid, Stage.names, "cpu", camera="cam1")
    repo.conn.execute("UPDATE jobs SET status=? WHERE id=?", (JobStatus.failed.value, job))
    repo.conn.execute("UPDATE submissions SET meta=json_set(meta, '$.erasing', ?) WHERE id=?",
                      (_now(), sid))
    assert repo.erasing(sid)
    assert request_names(repo, sid) == [] and not repo.retry_job(job)
    repo.conn.execute("UPDATE submissions SET meta=json_set(meta, '$.erasing', ?) WHERE id=?",
                      ("2026-01-01T00:00:00.000000+00:00", sid))
    assert not repo.erasing(sid)        # a stale stamp is logged and no longer counts
    assert repo.retry_job(job)


def test_the_erasure_notice_states_how_long_the_backups_keep_a_copy():
    from pfb_app.config import Settings
    assert any("最長42日" in line for line in erasure.keeps(Settings()))
    assert any("最長20日" in line for line in erasure.keeps(Settings(backup_keep=14, backup_keep_weekly=0))) is False
    assert any("最長14日" in line for line in erasure.keeps(Settings(backup_keep=14, backup_keep_weekly=0)))
