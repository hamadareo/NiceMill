"""Wizard steps 5 (確認して承認) and 6 (PDFを受け取る), AppTest on ui/app.py.

Only synthetic data: invented names and texts, frames are tiny black PNGs."""
import re
from pathlib import Path

import cv2
import numpy as np
from pdfhelp import pdf_font_names, pdf_page_count, pdf_text
from streamlit.testing.v1 import AppTest

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.services import review
from pfb_app.services.intake import register_documents
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.ui.common import cited_evidence
from pfb_app.ui.wizard import router, step_pdf, step_review, theme
from pfb_app.ui.wizard.progress import work_progress
from test_wizard_ui_basic import (PRIMARY_LABELS, UI, app, env, html,  # noqa: F401 (env)
                                  no_internal_terms, ok, pdf_bytes, primaries, screen_text,
                                  shown_step, video_assignment)
from test_wizard_ui_people_wait import finish_all, group, job, mapped, save, track, worker_alive

TEXTS = {"Pa": "花子さんは柱を丁寧に折って、塔の土台を作りました。",
         "Pb": "太郎さんは崩れそうな土台を押さえて、みんなを助けました。",
         "Pc": "次郎さんは高さを測って、次の手を提案しました。"}
EVENTS = {"Pa": ("cam1-0001-01", "00:12", "紙を折って柱を作った", "塔の強さにつながった"),
          "Pb": ("cam1-0002-01", "01:05", "土台を両手で押さえた", "崩れるのを防いだ"),
          "Pc": ("cam1-0003-01", "02:30", "定規で高さを測った", "次の手を決める材料になった")}


def frame_files(settings, eid, n=4):
    paths = []
    for k in range(n):
        p = settings.data_dir / "frames" / f"{eid}_{k}.png"     # tiny synthetic image
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.zeros((12, 16, 3), dtype=np.uint8))
        paths.append(str(p))
    return paths


def event(pid):
    eid, t, what, _ = EVENTS[pid]
    return {"event_id": eid, "camera": "cam1", "video_time": t, "task_elapsed": t,
            "participant": pid, "observed_behavior": what, "utterance": "",
            "speaker": "unknown", "evidence_source": "video", "confidence": "high"}


def reviewable(settings, repo, aid, label="A班", insufficient=(), untranslated=False):
    """A tracking group whose three students each have a PFB to check."""
    sid = group(repo, aid, label)
    track(settings, repo, sid)
    mapped(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    output = {"events": [event(p) for p in EVENTS],
              "frames": {EVENTS[p][0]: frame_files(settings, f"{sid}_{EVENTS[p][0]}")
                         for p in EVENTS}}
    if untranslated:
        output["provenance"] = {"translated": False, "events": 3, "translated_events": 1}
    beh = save(repo, sid, Stage.behavior, output, camera="cam1", upstream=[ident.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": EVENTS[p][0], "contribution_interpretation": EVENTS[p][3],
         "is_positive_contribution": True, "rubric_criteria": []} for p in EVENTS]},
        camera="cam1", upstream=[beh.id])
    for s in repo.list_subjects(sid):
        pid = s["participant_id"]
        out = ({"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []}
               if pid in insufficient else
               {"pfb_status": "generated", "pfb": TEXTS[pid], "evidence_refs": [EVENTS[pid][0]]})
        save(repo, sid, Stage.pfb, out, subject_id=s["id"], upstream=[con.id])
    return sid


def subject(repo, sid, pid):
    return next(s for s in repo.list_subjects(sid) if s["participant_id"] == pid)


def btn(at, label):
    return next(b for b in at.button if b.label == label)


def editors(at):
    """The feedback editors on screen (the practice tips' boxes are text areas too)."""
    return [t for t in at.text_area if (t.key or "").startswith("pfb_edit_")]


def has_button(at, label):
    return any(b.label == label for b in at.button)


def texts(at, kind):
    return [e.value for e in at.get(kind)]


def shown_student(at):
    return next(w.value for w in [*at.radio, *at.selectbox] if (w.key or "").startswith("pfb_student_"))


# --- step 5 ---------------------------------------------------------------------------------

def test_one_student_with_large_text_evidence_and_small_frames(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    page = html(at)
    assert "承認 0 / 全 3" in page
    assert shown_student(at) == subject(repo, sid, "Pa")["id"]       # the first to check
    assert TEXTS["Pa"] in page and TEXTS["Pb"] not in page
    assert "00:12" in page and "紙を折って柱を作った" in page and "塔の強さにつながった" in page
    assert "01:05" not in page                                       # only what is cited
    images = at.get("image")
    assert len(images) == 3                                          # 4 frames, at most 3
    assert primaries(at) == ["全員まとめて承認"]   # two or more to check
    assert has_button(at, "直す") and has_button(at, "この人だけ作り直す")
    assert any(e.label == "この画面の使い方" for e in at.expander)
    no_internal_terms(at)


def test_approve_shows_the_stamp_once_and_moves_to_the_next_student(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "承認する").click().run()
    ok(at)
    hanako = subject(repo, sid, "Pa")
    assert review.subject_status(repo, hanako["id"]).value == "approved"
    page = html(at)
    assert "承認 1 / 全 3" in page
    assert '<progress class="pfb-bar pfb-bar--sm pfb-bar--approval" max="3" value="1"' in page
    assert "pfb-stamp--animate" in page and "花子さんを承認しました" in page
    assert shown_student(at) == subject(repo, sid, "Pb")["id"]
    assert TEXTS["Pb"] in html(at)
    assert shown_step(at) == 5
    at.run()                                                          # the stamp ran once
    ok(at)
    assert "pfb-stamp--animate" not in html(at)
    # an approved student shows the still stamp; 続ける leads to the next one to check
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    assert 'class="pfb-stamp"' in html(at) and "pfb-stamp--animate" not in html(at)
    assert primaries(at) == ["全員まとめて承認"]   # 2 left: the bulk press; 続ける stays
    btn(at, "続ける").click().run()
    ok(at)
    assert shown_student(at) == subject(repo, sid, "Pb")["id"]


def test_the_stamp_animation_is_guarded_by_reduced_motion():
    css = theme.theme_css()
    animated = css.index(".pfb-stamp--animate { animation")
    guard = css.rfind("@media (prefers-reduced-motion: no-preference)", 0, animated)
    assert guard != -1
    between = css[guard:animated]
    assert between.count("{") > between.count("}")          # the rule sits inside the guard
    assert "pfb-stamp--animate" in theme.approval_stamp(animate=True)
    assert "pfb-stamp--animate" not in theme.approval_stamp(animate=False)


def test_all_done_shows_a_completion_line_and_leads_to_the_pdf(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "承認する").click().run()
    btn(ok(at), "承認する").click().run()
    ok(at)
    page = html(at)
    assert "承認 2 / 全 3" in page
    assert "全員の確認が終わりました" not in page        # 次郎さん's short text waits (L6)
    assert shown_student(at) == subject(repo, sid, "Pc")["id"]
    assert primaries(at) == ["この文で承認する"]
    assert "pfb-stamp--animate" in page                               # 太郎さん, just now
    btn(at, "この文で承認する").click().run()
    ok(at)
    page = html(at)
    assert "承認 3 / 全 3" in page
    assert "全員の確認が終わりました" in page
    assert shown_step(at) == 5                                        # stays to show it
    assert primaries(at) == ["続ける"]
    no_internal_terms(at)
    btn(at, "続ける").click().run()
    ok(at)
    assert shown_step(at) == 6
    assert subject(repo, sid, "Pc")["approval_status"] == "approved"


def test_a_finished_group_stays_on_screen_and_leads_to_the_next_group(env):
    settings, repo = env
    aid = video_assignment(repo)
    a = reviewable(settings, repo, aid, "A班", insufficient=("Pc",))
    b = reviewable(settings, repo, aid, "B班")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "承認する").click().run()
    btn(ok(at), "承認する").click().run()
    btn(ok(at), "この文で承認する").click().run()                 # 次郎さん's short text (L6)
    ok(at)
    assert at.button(key=f"pfb_group_{a}").disabled              # still A班, now finished
    assert "A班は全員の確認が終わりました。次はB班です。" in html(at)
    assert primaries(at) == ["続ける"]
    btn(at, "続ける").click().run()
    ok(at)
    assert shown_step(at) == 5 and at.button(key=f"pfb_group_{b}").disabled
    assert shown_student(at) == subject(repo, b, "Pa")["id"]


def test_fix_then_approve_stores_the_teachers_text_and_approves(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "直す").click().run()
    ok(at)
    editor = at.text_area(key=f"pfb_edit_{hanako['id']}")
    assert editor.value == TEXTS["Pa"]                                 # prefilled
    assert primaries(at) == ["直して承認"]
    assert has_button(at, "やめる") and not has_button(at, "承認する")
    no_internal_terms(at)
    btn(at, "やめる").click().run()                                    # closes it again
    ok(at)
    assert primaries(at) == ["全員まとめて承認"] and not editors(at)
    btn(at, "直す").click().run()
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("花子さんは最後まで柱を支えました。").run()
    btn(at, "直して承認").click().run()
    ok(at)
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    assert run.source == "teacher" and run.output["pfb"] == "花子さんは最後まで柱を支えました。"
    assert review.subject_status(repo, hanako["id"]).value == "approved"
    assert subject(repo, sid, "Pa")["approved_run_id"] == run.id
    assert "承認 1 / 全 3" in html(at)
    assert not editors(at)


def test_an_empty_fix_is_a_plain_message(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "直す").click().run()
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("   ").run()
    btn(at, "直して承認").click().run()
    ok(at)
    assert any("文章が空です" in e for e in texts(at, "error"))
    assert repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]).source == "model"


def test_regenerate_shows_processing_until_a_new_text_is_there(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "この人だけ作り直す").click().run()
    ok(at)
    assert repo.has_active_job(sid, Stage.pfb, subject_id=hanako["id"])
    assert shown_step(at) == 5                         # stays although the group is busy
    assert shown_student(at) == hanako["id"]
    page = html(at)
    assert re.search(r'class="pfb-who".*?解析中.*?</div>', page, re.S)
    assert TEXTS["Pa"] not in page
    assert primaries(at) == ["全員まとめて承認"] and btn(at, "承認する").disabled
    no_internal_terms(at)
    # the worker writes a new text: the next run (what the watch triggers) shows it
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "花子さんは新しい柱を作りました。",
                                "evidence_refs": [EVENTS["Pa"][0]]},
         subject_id=hanako["id"], upstream=[con.id])
    finish_all(repo, sid)
    at.run()
    ok(at)
    assert "花子さんは新しい柱を作りました。" in html(at)
    assert not btn(at, "承認する").disabled


def test_step_5_stays_reachable_while_one_text_is_written_again(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.regenerate_from(repo, sid, Stage.pfb, subject_id=subject(repo, sid, "Pa")["id"])
    p = work_progress(repo, aid)
    assert p.reachable(5)


def test_an_insufficient_student_offers_writing_it_yourself(env):
    """A document work (no scenes, so no short text, L6): 「自分で書く」 from an empty editor."""
    settings, repo = env
    aid = documents(settings, repo)
    sub = repo.list_submissions(aid)[0]
    hanako = repo.list_subjects(sub["id"])[0]
    con = repo.latest_run(sub["id"], Stage.contribution)
    save(repo, sub["id"], Stage.pfb, {"pfb_status": "insufficient_positive_evidence",
                                      "pfb": None, "evidence_refs": []},
         subject_id=hanako["id"], upstream=[con.id])
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key="pfb_student_all").set_value(hanako["id"]).run()
    ok(at)
    assert step_review.INSUFFICIENT in screen_text(at)
    assert has_button(at, "自分で書く") and not has_button(at, "直す")
    assert len(primaries(at)) == 1
    no_internal_terms(at)
    btn(at, "自分で書く").click().run()
    ok(at)
    assert at.text_area(key=f"pfb_edit_{hanako['id']}").value == ""
    assert primaries(at) == ["直して承認"]
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("花子さんは道具を配りました。").run()
    btn(at, "直して承認").click().run()
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value == "approved"


def test_a_gate_shows_a_plain_message_and_the_detail_link(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, untranslated=True)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    text = screen_text(at)
    assert "英語のまま" in text and "翻訳未完了" not in text
    link = next(l for l in at.get("page_link") if l.proto.label == "根拠を管理用の画面で直す")
    assert "review" in link.proto.page
    assert link.proto.query_string == f"subject={hanako['id']}"
    assert primaries(at) == ["承認する"] and btn(at, "承認する").disabled
    no_internal_terms(at)


def test_the_review_page_opens_the_student_from_the_link(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    taro = subject(repo, sid, "Pb")
    at = AppTest.from_file(str(UI / "pages" / "review.py"), default_timeout=60)
    at.query_params["subject"] = str(taro["id"])
    at.run()
    ok(at)
    assert at.selectbox[0].value == aid
    assert at.selectbox(key="review_subject").value == taro["id"]
    assert "subject" not in at.query_params                 # read once; the picker is free


def test_busy_and_stale_errors_are_plain_messages(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))

    def busy(*a, **k):
        raise BusyError("実行中または待機中のジョブがあるため承認できません。")
    monkeypatch.setattr(review, "approve", busy)
    btn(at, "承認する").click().run()
    ok(at)
    errors = texts(at, "error")
    assert errors and all("ジョブ" not in e for e in errors)
    assert any("解析が進んでいる" in e for e in errors)

    def stale(*a, **k):
        raise ApprovalError("PFBが古くなっています。再生成するか、手書きしてください。")
    monkeypatch.setattr(review, "approve", stale)
    btn(at, "承認する").click().run()
    ok(at)
    errors = texts(at, "error")
    assert any("解析が更新されました" in e for e in errors)
    assert all("PFB" not in e for e in errors)


def test_while_another_text_is_written_fixing_and_redo_still_work(env):
    """Per-person results (phone-first lane): another person's text being written never holds
    this person up; only this person's own text, or an earlier step, does."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    taro = subject(repo, sid, "Pb")
    job(repo, sid, Stage.pfb, subject_id=taro["id"])               # another student is redone
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": None}))
    assert shown_student(at) == hanako["id"]
    btn(at, "直す").click().run()
    ok(at)
    assert not btn(at, "直して承認").disabled
    btn(at, "やめる").click().run()
    btn(ok(at), "この人だけ作り直す").click().run()
    ok(at)
    assert repo.has_active_job(sid, Stage.pfb, subject_id=hanako["id"])
    job(repo, sid, Stage.behavior, camera="cam1")                  # an earlier step: busy
    jiro = subject(repo, sid, "Pc")
    try:
        review.regenerate_from(repo, sid, Stage.pfb, subject_id=jiro["id"])
    except BusyError:
        pass
    else:
        raise AssertionError("an earlier step must hold the text back")


def test_a_names_job_or_another_text_job_leaves_ready_texts_ready(env):
    """A names job is not upstream of the texts, and another person's text job is not this
    person's: step 5, step 6 and the evaluator sheet keep a ready text 確認待ち."""
    from pfb_app.export.sheets import build_report   # noqa: PLC0415
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    job(repo, sid, Stage.names, status="running", camera="cam1")
    sub = repo.get_submission(sid)
    assert {s.state for s in step_review.students_of(repo, sub, Kind.video)} == {"review"}
    taro = subject(repo, sid, "Pb")
    job(repo, sid, Stage.pfb, status="running", subject_id=taro["id"])
    states = {s.subject["id"]: s.state for s in step_review.students_of(repo, sub, Kind.video)}
    assert states.pop(taro["id"]) == "regenerating" and set(states.values()) == {"review"}
    _, waiting = step_pdf.who(repo, aid, Kind.video)
    assert dict(waiting) == {s["display_name"]: ("解析中" if s["id"] == taro["id"] else "確認待ち")
                             for s in repo.list_subjects(sid)}
    reasons = {s.subject_id: s.reason for s in build_report(repo, aid).sheets}
    assert reasons.pop(taro["id"]) == "解析中" and set(reasons.values()) == {"確認待ち"}


def test_several_groups_have_a_picker_and_their_own_counter(env):
    settings, repo = env
    aid = video_assignment(repo)
    a = reviewable(settings, repo, aid, "A班")
    b = reviewable(settings, repo, aid, "B班")
    at = ok(app(session={"assignment_id": aid}))
    assert at.button(key=f"pfb_group_{a}").disabled
    at.button(key=f"pfb_group_{b}").click().run()
    ok(at)
    assert shown_student(at) == subject(repo, b, "Pa")["id"]
    assert "承認 0 / 全 3" in html(at)


def documents(settings, repo):
    """A document work: two reports, each with a cited item on page 2."""
    aid = repo.create_assignment("探究レポート", "document", "調べてまとめる", "", "standard", False)
    register_documents(repo, settings, aid, [("山田太郎.pdf", pdf_bytes()),
                                             ("佐藤花子.pdf", pdf_bytes())], None)
    for sub in repo.list_submissions(aid):
        finish_all(repo, sub["id"])
        ev = save(repo, sub["id"], Stage.evidence, {"items": [
            {"item_id": "p2-001", "page": 2, "quote": "図で比べた", "observed_content": "図で比べた",
             "evidence_source": "text", "confidence": "high", "quote_verified": True}]})
        con = save(repo, sub["id"], Stage.contribution, {"judgments": [
            {"ref_id": "p2-001", "contribution_interpretation": "分かりやすく示した",
             "is_positive_contribution": True, "rubric_criteria": []}]}, upstream=[ev.id])
        for s in repo.list_subjects(sub["id"]):
            save(repo, sub["id"], Stage.pfb, {"pfb_status": "generated", "pfb": "図で比べました。",
                                              "evidence_refs": ["p2-001"]},
                 subject_id=s["id"], upstream=[con.id])
    return aid


def test_documents_are_checked_as_one_list_with_pages(env):
    settings, repo = env
    aid = documents(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    assert "承認 0 / 全 2" in page and "2ページ" in page and "分かりやすく示した" in page
    assert not any(b.key and b.key.startswith("pfb_group_") for b in at.button)
    assert primaries(at) == ["全員まとめて承認"]


def test_document_pdfs_hold_the_approved_report(env):
    settings, repo = env
    aid = documents(settings, repo)
    yamada = next(s for sub in repo.list_submissions(aid) for s in repo.list_subjects(sub["id"])
                  if s["display_name"] == "山田太郎")
    review.approve(repo, yamada["id"])
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": None}))
    assert shown_step(at) == 5                                       # documents: PDF is 5
    assert primaries(at) == ["参加者に渡すPDFを受け取る"]
    assert "山田太郎さん" in screen_text(at) and "佐藤花子さん（確認待ち）" in screen_text(at)
    for which in ("student", "teacher"):
        text = pdf_text(step_pdf.pdf_data(aid, which)())
        assert "山田太郎" in text
        if which == "student":
            assert "佐藤花子" not in text
        else:   # on the contact sheet only, as not in the PDF
            assert "入っていない" in text
    assert "図で比べた" in pdf_text(step_pdf.pdf_data(aid, "teacher")())


def test_the_pdf_step_says_so_when_the_approvals_cannot_be_read(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])

    def broken(*a, **k):
        raise KeyError("run 99")
    monkeypatch.setattr(step_pdf, "collect_approved", broken)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert any("PDFを用意できませんでした" in e for e in texts(at, "error"))
    assert not at.download_button
    assert primaries(at) == ["参加者に渡すPDFを受け取る"] and at.button(key="pfb_pdf_student").disabled


def test_cited_evidence_follows_the_refs_and_caps_frames(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    run = repo.latest_run(sid, Stage.pfb, subject_id=subject(repo, sid, "Pb")["id"])
    items = cited_evidence(repo, run)
    assert [(i.when, i.what, i.why) for i in items] == [("01:05", "土台を両手で押さえた",
                                                         "崩れるのを防いだ")]
    assert len(items[0].frames) == 3 and all(Path(f.path).is_file() for f in items[0].frames)


def save_pfb(repo, sid, subject_id, text, warnings=(), refs=()):
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subject_id,
        version=0, source="model", upstream_run_ids=[con.id] if con else [], prompt_hash=None,
        model_id=None, output={"pfb_status": "generated", "pfb": text, "evidence_refs": list(refs)},
        status="ok", error=None, warnings=list(warnings), duration_sec=0.0, created_at=""))


def test_the_texts_warnings_are_shown_in_plain_words(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤", "内部IDが含まれています: cam1-0001-01"],
             [EVENTS["Pa"][0]])
    at = ok(app(session={"assignment_id": aid}))
    warnings = texts(at, "warning")
    assert any("色で人を表す言葉が入っています。" in w for w in warnings)
    assert any("記号や番号のような呼び方" in w for w in warnings)
    assert all("色の語" not in w and "内部ID" not in w and "cam1" not in w for w in warnings)
    assert primaries(at) == ["全員まとめて承認"]
    no_internal_terms(at)


def test_a_teachers_text_with_warnings_needs_a_second_press(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "直す").click().run()
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("赤い服のP2さんが柱を支えました。").run()
    btn(at, "直して承認").click().run()
    ok(at)
    assert repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]).source == "model"   # not yet
    warnings = texts(at, "warning")
    assert any("色で人を表す言葉" in w for w in warnings)
    assert any("人物の番号" in w for w in warnings)
    assert primaries(at) == ["このまま直して承認"]
    no_internal_terms(at)
    btn(at, "このまま直して承認").click().run()
    ok(at)
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    assert run.source == "teacher" and run.output["pfb"] == "赤い服のP2さんが柱を支えました。"
    assert review.subject_status(repo, hanako["id"]).value == "approved"


def test_a_clean_teachers_text_is_approved_at_once(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    assert review.manual_pfb_warnings(repo, hanako["id"], "花子さんは柱を支えました。") == []
    assert review.manual_pfb_warnings(repo, hanako["id"], "cam1-0001-01 で支えた") != []


def test_a_press_on_an_outdated_text_is_dropped(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    assert TEXTS["Pa"] in html(at)
    save_pfb(repo, sid, hanako["id"], "花子さんは別の場面でも活躍しました。", refs=[EVENTS["Pa"][0]])
    btn(at, "承認する").click().run()                    # pressed on the old text
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value != "approved"
    assert any("文章が新しくなりました。確かめてから押してください。" in w
               for w in texts(at, "warning"))
    assert "花子さんは別の場面でも活躍しました。" in html(at)
    btn(at, "承認する").click().run()                    # now on the text shown
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value == "approved"


def test_the_gate_comes_before_a_failed_text(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, untranslated=True)
    hanako = subject(repo, sid, "Pa")
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=hanako["id"],
        version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output=None, status="failed", error="x", warnings=[], duration_sec=0.0, created_at=""))
    student = next(s for s in step_review.students_of(repo, repo.get_submission(sid),
                                                       step_review.Kind.video)
                   if s.id == hanako["id"])
    assert student.state == "gated"


def test_the_draft_is_kept_when_switching_students(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako, taro = subject(repo, sid, "Pa"), subject(repo, sid, "Pb")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "直す").click().run()
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("下書きの文章です。").run()
    at.radio(key=f"pfb_student_{sid}").set_value(taro["id"]).run()
    ok(at)
    assert not editors(at)
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    assert at.text_area(key=f"pfb_edit_{hanako['id']}").value == "下書きの文章です。"


def test_counts_stay_while_one_text_is_written_again(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pb")["id"])
    review.regenerate_from(repo, sid, Stage.pfb, subject_id=subject(repo, sid, "Pa")["id"])
    p = work_progress(repo, aid)
    assert (p.approved, p.total) == (1, 3)
    assert p.reachable(6)


def test_the_stamp_is_dropped_outside_the_review_step(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    review.approve(repo, hanako["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None},
                extra={router.REVIEW_STAMP: (hanako["id"], "花子")}))
    assert shown_step(at) == 6
    assert router.REVIEW_STAMP not in at.session_state


# --- step 6 ---------------------------------------------------------------------------------

def test_the_pdf_step_needs_an_approval(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert shown_step(at) == 5                         # not reachable yet
    assert not at.download_button


def test_the_pdf_step_offers_both_pdfs_and_marks_the_download(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert shown_step(at) == 6
    assert primaries(at) == ["参加者に渡すPDFを受け取る"]
    labels = {b.label: b.proto.type for b in at.download_button}
    assert labels == {"参加者に渡すPDFを受け取る": "primary", "評価者用（根拠つき）PDF": "secondary",
                      "一人ずつのPDF（ZIPでまとめて）": "secondary"}
    assert any(e.label == "団体の設定" for e in at.expander)
    text = screen_text(at)
    assert "花子" in text and "太郎" in text and "次郎" in text
    assert "承認した参加者だけ" in text
    assert any(e.label == "この画面の使い方" for e in at.expander)
    no_internal_terms(at)
    assert not repo.get_assignment(aid).get("pdf_downloaded_at")
    at.download_button[0].click().run()
    ok(at)
    step_pdf.pdf_data(aid, "student")()       # the deferred download makes the bytes: recorded
    assert repo.get_assignment(aid)["pdf_downloaded_at"]


def test_the_one_person_zip_is_offered_and_counts_as_handed_out(env):
    import io
    import zipfile
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    data = step_pdf.pdf_data(aid, "each")()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.namelist() == ["1_花子.pdf", "1_花子.html"]
    assert repo.get_assignment(aid)["pdf_downloaded_at"]


def test_organisation_settings_are_saved_from_the_pdf_step(env):
    from pfb_app.services import organisation
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    at.text_input(key="pfb_org_name").input("みなと町の会")
    at.radio(key="pfb_org_accent").set_value("green")
    at = ok(at.button(key="FormSubmitter:pfb_org_form-団体の設定を保存").click().run())
    saved = organisation.load(settings)
    assert (saved.name, saved.accent) == ("みなと町の会", "green")
    assert "保存しました" in screen_text(at)
    assert primaries(at) == ["参加者に渡すPDFを受け取る"]


def test_the_pdf_data_is_the_student_and_teacher_export(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    for which in ("student", "teacher"):
        data = step_pdf.pdf_data(aid, which)()
        assert data.startswith(b"%PDF")
        text = pdf_text(data)
        assert "花子" in text and TEXTS["Pa"][:10] in text and TEXTS["Pb"][:10] not in text
        if which == "student":
            assert "太郎" not in text


def test_when_everyone_is_approved_the_pdf_line_says_so(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    for s in repo.list_subjects(sid):
        review.approve(repo, s["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    text = screen_text(at)
    assert "承認した参加者だけ" not in text and "全員" in text


# --- every new screen -----------------------------------------------------------------------

def test_every_review_and_pdf_screen_has_one_primary_help_and_plain_words(env):
    settings, repo = env
    worker_alive(settings)
    normal = video_assignment(repo, "通常")
    s_normal = reviewable(settings, repo, normal)
    short = video_assignment(repo, "場面が少ない")
    s_short = reviewable(settings, repo, short, insufficient=("Pa",))
    gated = video_assignment(repo, "英語")
    reviewable(settings, repo, gated, untranslated=True)
    redo = video_assignment(repo, "作り直し")
    s_redo = reviewable(settings, repo, redo)
    review.regenerate_from(repo, s_redo, Stage.pfb, subject_id=subject(repo, s_redo, "Pa")["id"])
    done = video_assignment(repo, "完了")
    s_done = reviewable(settings, repo, done)
    for s in repo.list_subjects(s_done):
        review.approve(repo, s["id"])
    screens = [
        dict(session={"assignment_id": normal}),                                     # normal
        dict(session={"assignment_id": normal},
             extra={step_review.EDITING: subject(repo, s_normal, "Pa")["id"]}),  # editing
        dict(session={"assignment_id": short}),                                      # insuff.
        dict(session={"assignment_id": gated}),                                      # gated
        dict(session={"assignment_id": redo, "step": 5, "submission_id": None}),     # redoing
        dict(session={"assignment_id": done, "step": 5, "submission_id": None}),     # all done
        dict(session={"assignment_id": done, "step": 6, "submission_id": None}),     # PDF
    ]
    for screen in screens:
        at = ok(app(**screen))
        assert len(primaries(at)) == 1, (screen, primaries(at))
        assert primaries(at)[0] in PRIMARY_LABELS, primaries(at)
        assert any(e.label == "この画面の使い方" for e in at.expander), screen
        no_internal_terms(at)
    at = ok(app(session={"assignment_id": short}))
    assert shown_student(at) != subject(repo, s_short, "Pa")["id"]   # the first to check


def test_pdf_buttons_sports_words(env):
    settings, repo = env
    aid = repo.create_assignment("試合", Kind.video, "t", "", "standard", False,
                                 domain="sports", sport="soccer")
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert primaries(at) == ["選手に渡すPDFを受け取る"]
    labels = {b.label: b.proto.type for b in at.download_button}
    assert labels == {"選手に渡すPDFを受け取る": "primary", "コーチ用（根拠つき）PDF": "secondary",
                      "一人ずつのPDF（ZIPでまとめて）": "secondary"}
    assert "生徒" not in screen_text(at) and "先生" not in screen_text(at)


def test_localize_leaves_general_text_alone():
    from pfb_app.domain.vocab import localize
    t = "参加者に渡す前に、評価者が活動の名簿を見ます"
    assert localize(t, "school") == t == localize(t, None)
    assert localize(t, "sports") == "選手に渡す前に、コーチが試合・練習の選手名簿を見ます"


GENERAL_WORDS = ("生徒", "先生", "課題", "参加者", "評価者", "活動", "名簿")


def _sports_work(settings, repo):
    aid = repo.create_assignment("試合", Kind.video, "メモ", "", "standard", False,
                                 domain="sports", sport="soccer")
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    return aid


def test_every_step_of_a_sports_work_uses_sports_words(env):
    settings, repo = env
    aid = _sports_work(settings, repo)
    for step in range(1, 7):
        at = ok(app(session={"assignment_id": aid, "step": step, "submission_id": None}))
        text = screen_text(at).replace("選手名簿", "")   # the sports word holds 名簿
        text = text.replace("グループ活動", "")          # a kind of scene, not the work
        assert not [w for w in GENERAL_WORDS if w in text], (step, text)


def test_the_domain_does_not_leak_to_home_or_the_new_work_screen(env):
    settings, repo = env
    aid = _sports_work(settings, repo)
    school = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    at.session_state["wizard"] = {"assignment_id": None, "step": None, "submission_id": None}
    at = ok(at.run())
    assert "活動" in screen_text(at)    # home help: general words
    at.session_state["wizard"] = {"assignment_id": None, "step": 1, "submission_id": None}
    at = ok(at.run())
    assert "評価したいファイルを入れてください" in screen_text(at)     # the drop zone
    assert "試合・練習" not in screen_text(at) and "選手" not in screen_text(at)


def test_a_flash_is_shown_in_the_work_vocabulary(env):
    settings, repo = env
    aid = _sports_work(settings, repo)
    at = app(session={"assignment_id": aid, "step": 5, "submission_id": None},
             extra={"_flash": ("参加者に渡せます。", "success", False)})
    assert any("選手に渡せます。" in e.value for e in at.success)


def test_a_raw_flash_keeps_the_users_group_label(env):
    settings, repo = env
    aid = _sports_work(settings, repo)
    msg = "「活動班」を登録しました。解析を始めています。"
    at = app(session={"assignment_id": aid, "step": 5, "submission_id": None},
             extra={"_flash": (msg, "success", True)})
    assert any(e.value == msg for e in at.success)


def test_registered_flash_is_raw_for_a_group_label(monkeypatch):
    from pfb_app.ui.wizard import step_files
    seen = {}
    monkeypatch.setattr(step_files, "flash", lambda m, kind="success", raw=False: seen.update(raw=raw))
    monkeypatch.setattr(step_files.router, "go", lambda n: None)
    monkeypatch.setattr(step_files.st, "rerun", lambda: None)
    monkeypatch.setattr(step_files.st, "session_state", {})
    step_files._registered("x", raw=True)
    assert seen["raw"] is True


def test_legacy_pages_show_flashes_in_general_words(env):
    settings, repo = env
    at = AppTest.from_file(str(UI / "pages/status.py"), default_timeout=60)
    at.session_state["pfb_domain"] = "sports"
    at.session_state["_flash"] = ("参加者に渡せます。", "success", False)
    at.run()
    assert any(e.value == "参加者に渡せます。" for e in at.success)
