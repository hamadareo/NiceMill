"""Per-person results as they finish (phone-first lane, audit #4): while the feedback of some
people is still being written, the people whose text is ready can be checked and approved.

Only synthetic data: invented names and texts, frames are tiny black PNGs."""
from pfb_app.domain.models import Kind, Stage
from pfb_app.services import review
from pfb_app.ui.wizard import step_review
from pfb_app.ui.wizard.progress import work_progress
from test_wizard_ui_basic import (app, env, html, no_internal_terms, ok,  # noqa: F401 (env)
                                  primaries, shown_step, video_assignment)
from test_wizard_ui_people_wait import finish_all, group, job, mapped, save, track, worker_alive
from test_wizard_ui_review_pdf import EVENTS, TEXTS, btn, event, frame_files, subject


def partly_written(settings, repo, aid, written=("Pa", "Pb"), status="running", label="A班"):
    """A tracking group of three: the people in `written` have their text, the others' pfb
    jobs are `status` (the worker is writing them)."""
    sid = group(repo, aid, label)
    track(settings, repo, sid)
    mapped(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    output = {"events": [event(p) for p in EVENTS],
              "frames": {EVENTS[p][0]: frame_files(settings, f"{sid}_{EVENTS[p][0]}")
                         for p in EVENTS}}
    beh = save(repo, sid, Stage.behavior, output, camera="cam1", upstream=[ident.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": EVENTS[p][0], "contribution_interpretation": EVENTS[p][3],
         "is_positive_contribution": True, "rubric_criteria": []} for p in EVENTS]},
        camera="cam1", upstream=[beh.id])
    for s in repo.list_subjects(sid):
        pid = s["participant_id"]
        if pid in written:
            save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": TEXTS[pid],
                                        "evidence_refs": [EVENTS[pid][0]]},
                 subject_id=s["id"], upstream=[con.id])
        else:
            job(repo, sid, Stage.pfb, status, subject_id=s["id"])
    return sid


def write_rest(repo, sid):
    """The worker finishes the remaining texts."""
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    for s in repo.list_subjects(sid):
        if repo.latest_run(sid, Stage.pfb, subject_id=s["id"]) is None:
            pid = s["participant_id"]
            save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": TEXTS[pid],
                                        "evidence_refs": [EVENTS[pid][0]]},
                 subject_id=s["id"], upstream=[con.id])
    finish_all(repo, sid)


# --- the derivation ---------------------------------------------------------------------------

def test_ready_people_are_the_coachs_turn_while_others_are_written(env):
    settings, repo = env
    aid = video_assignment(repo)
    partly_written(settings, repo, aid)
    p = work_progress(repo, aid)
    sub = p.subs[0]
    assert (sub.step, sub.status) == ("review", "your_turn")
    assert p.your_turn and p.current == 5 and p.turn_step == 5
    assert sub.reason == "2人を確かめられます（ほかの1人の文章を作っています）"
    assert p.reason == sub.reason
    assert (p.approved, p.total) == (0, 3)


def test_nobody_ready_yet_is_still_the_analysis(env):
    settings, repo = env
    aid = video_assignment(repo)
    partly_written(settings, repo, aid, written=())
    p = work_progress(repo, aid)
    assert (p.subs[0].step, p.subs[0].status) == ("wait", "processing")
    assert not p.your_turn and p.reachable(5)


def test_queued_texts_count_as_being_written(env):
    settings, repo = env
    aid = video_assignment(repo)
    partly_written(settings, repo, aid, written=("Pa",), status="queued")
    sub = work_progress(repo, aid).subs[0]
    assert (sub.step, sub.status) == ("review", "your_turn")
    assert sub.reason == "1人を確かめられます（ほかの2人の文章を作っています）"


def test_approved_people_do_not_count_as_ready(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid, written=("Pa",))
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    p = work_progress(repo, aid)
    assert (p.subs[0].step, p.subs[0].status) == ("wait", "processing")
    assert (p.approved, p.total) == (1, 3) and p.reachable(5)


def test_upstream_work_keeps_the_group_in_the_analysis(env):
    """A text whose analysis is redone (behavior queued) is not offered meanwhile."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid)
    job(repo, sid, Stage.behavior, "queued", camera="cam1")
    sub = work_progress(repo, aid).subs[0]
    assert sub.step == "wait" and sub.status == "processing"


def test_documents_too(env):
    settings, repo = env
    aid = repo.create_assignment("書類の活動", Kind.document, "防災を調べる", "", "standard", False)
    subs = []
    for name in ("佐藤", "鈴木"):
        sid = repo.create_submission(aid, name, {"source_path": "/x/a.pdf"})
        subs.append((sid, repo.add_subject(sid, None, name, None, {})))
    for n, (sid, subj) in enumerate(subs):
        prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x/a.pdf"})
        ev = save(repo, sid, Stage.evidence, {"items": []}, upstream=[prep.id])
        con = save(repo, sid, Stage.contribution, {"judgments": []}, upstream=[ev.id])
        if n == 0:
            save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "よく調べていました。",
                                        "evidence_refs": []}, subject_id=subj, upstream=[con.id])
        else:
            job(repo, sid, Stage.pfb, "running", subject_id=subj)
    p = work_progress(repo, aid)
    assert p.your_turn and p.turn_step == p.number("review")


# --- the screens ------------------------------------------------------------------------------

def test_step_4_moves_on_when_the_first_text_is_ready(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid, written=())
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    hanako = subject(repo, sid, "Pa")
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": TEXTS["Pa"],
                                "evidence_refs": [EVENTS["Pa"][0]]},
         subject_id=hanako["id"], upstream=[con.id])
    repo.conn.execute("UPDATE jobs SET status='done' WHERE subject_id=?", (hanako["id"],))
    at.run()
    ok(at)
    assert shown_step(at) == 5
    assert TEXTS["Pa"] in html(at)


def test_step_5_shows_ready_people_and_who_is_still_written(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid, written=("Pa", "Pb"))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    page = html(at)
    assert TEXTS["Pa"] in page                              # the first ready one is shown
    assert "承認 0 / 全 3" in page
    assert step_review.WRITING_LINE.format(n=1) in page    # the others are coming
    labels = [o for w in at.radio for o in w.options]
    assert any("文章を書いています" in o for o in labels)
    assert primaries(at) == ["全員まとめて承認"]          # the two ready, as one press
    no_internal_terms(at)
    # the one still being written: its own line, no text, no approve
    jiro = subject(repo, sid, "Pc")
    at.radio(key=f"pfb_student_{sid}").set_value(jiro["id"]).run()
    ok(at)
    page = html(at)
    assert TEXTS["Pc"] not in page
    assert step_review.WRITING in [e.value for e in at.info]
    assert btn(at, "承認する").disabled


def test_bulk_approval_takes_only_the_ready_texts(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid, written=("Pa", "Pb"))
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "全員まとめて承認").click().run()
    ok(at)
    statuses = {s["participant_id"]: s["approval_status"] for s in repo.list_subjects(sid)}
    assert statuses["Pa"] == statuses["Pb"] == "approved"
    assert statuses["Pc"] != "approved"
    page = html(at)
    assert "2人を承認しました。" in page
    assert step_review.SKIP_WRITING in page                 # Pc, with why
    # the last text arrives: the screen offers it
    write_rest(repo, sid)
    at.run()
    ok(at)
    assert TEXTS["Pc"] in html(at)
    assert primaries(at) == ["承認する"]


def test_a_hand_written_text_is_one_transaction(env, monkeypatch):
    """The busy check and the save run inside one write transaction (review round 1, #8)."""
    from contextlib import contextmanager
    settings, repo = env
    aid = video_assignment(repo)
    sid = partly_written(settings, repo, aid, written=("Pa",))
    entered = []
    real = repo._tx

    @contextmanager
    def tx(join=False):
        entered.append(join)
        with real(join=join):
            yield
    monkeypatch.setattr(repo, "_tx", tx)
    review.write_manual_pfb(repo, subject(repo, sid, "Pa")["id"], "花子さんは丁寧でした。")
    assert entered and entered[0] is True
