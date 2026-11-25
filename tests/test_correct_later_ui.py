"""「あとから訂正」 (review_mode correct_later) down to the screens: the new-work default, the
mode switch and the 出典 on 作業の設定, step 5 「文を読む・直す」 (① read ② scenes ③ correct),
the worker's record of a 作り直し, 「再出力が必要」, step 6 and the PDFs; approve-mode works
stay as they were (their own tests, and the last ones here).

Only synthetic data: invented names and texts, frames are tiny black PNGs."""
import io
import json
import zipfile

import pytest
from tips_helpers import FEEDBACK, ctx_for, group, team_json, tips_json

from pfb_app.domain.models import Stage
from pfb_app.export import identity_gate as G
from pfb_app.export import labels
from pfb_app.export.pdf import render_student_pdf, render_student_zip, render_teacher_pdf
from pfb_app.services import corrections as cor
from pfb_app.services import organisation as org
from pfb_app.services import review
from pfb_app.services.assignments import mark_pdf_downloaded
from pfb_app.services.intake import create_assignment_from_form
from pfb_app.ui.wizard import progress as P
from pfb_app.ui.wizard.progress import work_progress
from test_export import text_of
from test_wizard_ui_basic import (app, env, make_video, ok, primaries, screen_text,  # noqa: F401
                                  shown_step, video_assignment)
from test_wizard_ui_review_pdf import TEXTS, btn, has_button, html, reviewable, subject

WORDS = ("承認", "渡す", "下書き", "未承認")


def no_approval_words(text: str) -> None:
    for word in WORDS:
        assert word not in text, (word, [ln for ln in text.splitlines() if word in ln][:3])


@pytest.fixture
def gate_ok(monkeypatch):
    """The PDFs' 同一人物の確認 passes for everybody (it has its own tests, test_identity_gate)."""
    monkeypatch.setattr(G, "assess", lambda repo, sub, subject, chain, pol=None: G.Verdict(
        G.OK, [], {}, {e["event_id"] for _c, src in chain
                       for e in (src.output or {}).get("events") or []}))


def live_work(settings, repo, *, checked=True):
    """A correct_later video work of three people. Its mapping was saved on 「人を確かめる」 (a
    person's decision); with checked=False it reads as made by the automatic people instead."""
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    if not checked:
        repo.conn.execute("UPDATE stage_runs SET source='auto' WHERE submission_id=? AND stage=?",
                          (sid, Stage.identity.value))
    return aid, sid


# --- the mode of a new work ---------------------------------------------------------------------

def test_new_works_start_as_correct_later_and_old_ones_stay(env):
    settings, repo = env
    made = create_assignment_from_form(repo, "探究レポート", "document", "調べてまとめる", "", None,
                                       "standard", False, settings)
    assert repo.get_assignment(made)["review_mode"] == cor.CORRECT_LATER
    from pfb_app.services.content_check import start_work_from_videos
    aid, _ = start_work_from_videos(repo, settings, [make_video(settings.inbox_dir / "a.mp4")],
                                    "standard")
    assert repo.get_assignment(aid)["review_mode"] == cor.CORRECT_LATER
    explicit = create_assignment_from_form(repo, "x", "document", "y", "", None, "standard", False,
                                           settings, review_mode=cor.APPROVE)
    assert repo.get_assignment(explicit)["review_mode"] == cor.APPROVE
    old = video_assignment(repo)                  # the storage default is still approve
    assert repo.get_assignment(old)["review_mode"] == cor.APPROVE
    repo.conn.execute("UPDATE assignments SET review_mode=NULL WHERE id=?", (old,))
    assert repo.get_assignment(old)["review_mode"] == cor.APPROVE    # NULL reads as approve


# --- the worker's record of a 作り直し -------------------------------------------------------------

def _answer(req):
    if "「次にためしてみること」を書く" in req.system:
        return tips_json("Pa")
    if "ためしてみること」を書く" in req.system:
        return team_json()
    return json.dumps({"pfb_status": "generated", "pfb": "作り直した文です。",
                       "evidence_refs": ["cam1-0001"]}, ensure_ascii=False)


def test_regenerate_by_a_person_is_a_version_with_who_and_the_way_back(tmp_path):
    from pfb_app.config import Settings
    from pfb_app.pipeline.runner import execute_job
    from pfb_app.storage.db import Repo, connect, init_schema
    from pathlib import Path
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid, sid, subj, runs = group(repo)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    repo.actor = (5, "coach5")
    review.regenerate_from(repo, sid, Stage.pfb, subject_id=subj["Pa"])
    repo.actor = None                               # the worker has no person
    ctx, _ = ctx_for(settings, repo, _answer)
    execute_job(repo.claim_next_job({"standard"}), ctx)
    rows = cor.history(repo, subj["Pa"])
    assert [(v.kind, v.reason, v.actor) for v in rows] == [("regenerate", cor.REDO_ASKED, "coach5")]
    assert rows[0].prev_run_id == runs["Pa"].id and rows[0].prev_text == FEEDBACK["Pa"]
    assert rows[0].text == "作り直した文です。"
    assert cor.state(repo, subj["Pa"]).label == "訂正済み 版1"
    cor.revert(repo, subj["Pa"], actor="coach5")    # the way back to the AI's first text
    latest = repo.latest_run(sid, Stage.pfb, subject_id=subj["Pa"])
    assert latest.output["pfb"] == FEEDBACK["Pa"] and cor.state(repo, subj["Pa"]).version == 2
    assert cor.history(repo, subj["Pz"]) == []      # nobody asked for the other one


def test_a_remake_by_the_analysis_is_recorded_and_first_texts_are_not(env):
    settings, repo = env
    aid, sid, subj, runs = group(repo)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    from tips_helpers import save
    first = save(repo, sid, Stage.pfb, {"pfb_status": "insufficient_positive_evidence",
                                        "pfb": None, "evidence_refs": []}, subject_id=subj["Pz"])
    assert cor.after_pfb(repo, [first], job_id=None) == 1          # Pz had a text before
    assert cor.history(repo, subj["Pz"])[0].reason == cor.REDO_ANALYSIS
    again = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "また作った文です。",
                                        "evidence_refs": []}, subject_id=subj["Pz"])
    assert cor.after_pfb(repo, [again], job_id=None) == 1          # Pz had a text before
    rows = cor.history(repo, subj["Pz"])
    assert [(v.prev_text, v.text) for v in rows] == [(FEEDBACK["Pz"], None), (None, "また作った文です。")]
    assert cor.state(repo, subj["Pz"]).label == "訂正済み 版2"      # the record explains the text
    newcomer = repo.add_subject(sid, "Pq", "テスト三郎", "", {})
    first_text = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "はじめての文。",
                                             "evidence_refs": []}, subject_id=newcomer)
    assert cor.after_pfb(repo, [first_text]) == 0                  # a person's first text
    other = repo.create_assignment("別の作業", "video", "x", "", "standard", False)
    sid2 = repo.create_submission(other, "B班", {})
    s2 = repo.add_subject(sid2, "Pa", "テスト一郎", "", {})
    one = save(repo, sid2, Stage.pfb, {"pfb_status": "generated", "pfb": "一つ目。",
                                       "evidence_refs": []}, subject_id=s2)
    two = save(repo, sid2, Stage.pfb, {"pfb_status": "generated", "pfb": "二つ目。",
                                       "evidence_refs": []}, subject_id=s2)
    assert one.id < two.id and cor.after_pfb(repo, [two]) == 0      # approve mode: no record
    assert cor.history(repo, s2) == []


# --- where a 「あとから訂正」 work stands -----------------------------------------------------------

def test_progress_has_no_approval_wait_and_asks_for_the_pdf_again_after_a_change(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    p = work_progress(repo, aid)
    assert p.live and p.steps[4] == P.LIVE_STEP and p.ready == 3 and p.approved == 0
    assert p.subs[0].step == "review" and p.subs[0].status == "your_turn"
    assert p.reason == P.LIVE_READY.format(n=3) and p.reachable(p.number("pdf"))
    assert p.analysis.review.name == "文を読む・直す" and "文ができた人 3 / 全 3" in p.analysis.review.note
    no_approval_words("\n".join([*p.steps, p.reason, p.analysis.review.note]))
    mark_pdf_downloaded(repo, aid)
    done = work_progress(repo, aid)
    assert done.complete and done.reason == "完了しました"
    assert work_progress(repo, aid).complete            # from the cache of complete works
    cor.report_wrong(repo, subject(repo, sid, "Pb")["id"], "動作が違う")   # a report after the PDF
    again = work_progress(repo, aid)
    assert not again.complete and again.subs[0].step == "pdf"
    assert again.subs[0].reason == P.LIVE_REISSUE and "再出力が必要" in again.reason
    assert cor.changed_since(repo, aid, repo.get_assignment(aid)["pdf_downloaded_at"]) \
        .keys() == {subject(repo, sid, "Pb")["id"]}
    mark_pdf_downloaded(repo, aid)
    assert work_progress(repo, aid).complete


def test_progress_of_an_approve_work_is_unchanged(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    p = work_progress(repo, aid)
    assert not p.live and p.steps[4] == "確認して承認" and p.reason == "3人の承認を待っています"
    assert not p.reachable(p.number("pdf"))


# --- step 5: 文を読む・直す ------------------------------------------------------------------------

def test_step5_reads_shows_the_scenes_and_offers_the_corrections(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    page = html(at)
    for part in ("① AIの文を読む", "② 根拠の場面を見る", "③ 直せる", TEXTS["Pa"], "訂正なし",
                 "文を読む・直す", "文ができた人 3 / 全 3", "紙を折って柱を作った"):
        assert part in page, part
    assert len(at.get("image")) == 3                         # the frames of the cited scene
    assert primaries(at) == ["PDFを受け取る"]
    for label in ("文を直す", "この文は間違い", "この人だけ作り直す"):
        assert has_button(at, label), label
    assert not has_button(at, "前の版に戻す")                 # nothing to go back to yet
    assert not any("訂正の記録" in e.label for e in at.expander)   # no record yet: no list
    no_approval_words(screen_text(at))
    btn(at, "PDFを受け取る").click().run()
    assert shown_step(ok(at)) == 6


def test_step5_edit_keeps_a_version_and_can_be_reverted(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    hanako = subject(repo, sid, "Pa")["id"]
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "文を直す").click().run()
    ok(at)
    assert primaries(at) == ["直した文を保存する"]
    editor = next(t for t in at.text_area if t.key == f"pfb_fix_text_{hanako}")
    assert editor.value == TEXTS["Pa"]
    editor.set_value("花子さんは柱をていねいに折り重ねて、塔の土台を作りました。")
    next(s for s in at.selectbox if s.key == f"pfb_fix_reason_{hanako}").set_value("動作が違う")
    next(t for t in at.text_input if t.key == f"pfb_fix_note_{hanako}").set_value("本数を直した")
    btn(at, "直した文を保存する").click().run()
    ok(at)
    state = cor.state(repo, hanako)
    assert state.label == "訂正済み 版1" and "動作が違う" in state.lines[0] and "本数を直した" in state.lines[0]
    page = screen_text(at)
    assert "折り重ねて" in page and "訂正済み 版1" in page and "直した文です（版1）" in page
    assert "次に受け取るPDFに入ります。" in page                          # in the PDF
    assert "直す前のAIの文の根拠" in page and len(at.get("image")) == 3   # the AI text's scenes
    assert any("訂正の記録（1件）" in e.label for e in at.expander)
    no_approval_words(screen_text(at))
    btn(at, "前の版に戻す").click().run()
    ok(at)
    assert TEXTS["Pa"] in html(at) and cor.state(repo, hanako).version == 2
    assert [v.kind for v in cor.history(repo, hanako)] == ["edit", "revert"]


def test_step5_report_then_resolve(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    hanako = subject(repo, sid, "Pa")["id"]
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "この文は間違い").click().run()
    ok(at)
    assert primaries(at) == ["報告する"]
    reasons = next(r for r in at.radio if r.key == f"pfb_report_reason_{hanako}")
    assert list(reasons.options) == list(cor.REASONS)
    btn(at, "報告する").click().run()                 # no reason chosen: refused, said
    ok(at)
    assert cor.open_report(repo, hanako) is None and any("理由" in e.value for e in at.error)
    reasons = next(r for r in at.radio if r.key == f"pfb_report_reason_{hanako}")
    reasons.set_value("人物の取り違え")
    next(t for t in at.text_input if t.key == f"pfb_report_note_{hanako}").set_value("別の人の場面")
    btn(at, "報告する").click().run()
    ok(at)
    assert cor.state(repo, hanako).label == cor.STATE_REPORT
    page = html(at)
    assert cor.STATE_REPORT in page and has_button(at, "確認して報告を閉じる")
    no_approval_words(screen_text(at))
    btn(at, "確認して報告を閉じる").click().run()
    ok(at)
    assert cor.state(repo, hanako).label == cor.STATE_NONE
    assert [v.kind for v in cor.history(repo, hanako)] == ["report", "resolve"]


def test_step5_redo_one_person_goes_through_the_worker_hook(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    hanako = subject(repo, sid, "Pa")["id"]
    old = repo.latest_run(sid, Stage.pfb, subject_id=hanako)
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "この人だけ作り直す").click().run()
    ok(at)
    job = repo.conn.execute("SELECT id FROM jobs WHERE stage='pfb' AND subject_id=?"
                            " AND status='queued'", (hanako,)).fetchone()
    assert job is not None and "作り直しています" in html(at)
    from tips_helpers import save
    new = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "作り直した花子さんの文です。",
                                      "evidence_refs": []}, subject_id=hanako, upstream=old.upstream_run_ids)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (job[0],))
    assert cor.after_pfb(repo, [new], job[0]) == 1      # what the worker does after the job
    row = cor.history(repo, hanako)[-1]
    assert (row.kind, row.reason, row.prev_run_id) == ("regenerate", cor.REDO_ASKED, old.id)
    at = ok(app(session={"assignment_id": aid}))
    assert "作り直した花子さんの文です" in html(at) and has_button(at, "前の版に戻す")


def test_step5_and_6_say_reoutput_needed_after_a_correction(env, gate_ok):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    mark_pdf_downloaded(repo, aid)
    taro = subject(repo, sid, "Pb")["id"]
    cor.edit_text(repo, taro, "太郎さんは土台を両手で押さえて、塔を守りました。")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 6                       # the derived step: get the PDF again
    page = screen_text(at)
    assert "再出力が必要" in page and "太郎さん" in page
    assert primaries(at) == ["参加者用のPDFを受け取る"]
    assert "PDFに入っている参加者（3人）" in page and "背番号の読み取り" not in page
    no_approval_words(page)
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    assert "再出力が必要" in screen_text(at)
    no_approval_words(screen_text(at))


def test_step6_without_anyone_confirmed_says_why(env, monkeypatch):
    settings, repo = env
    aid, sid = live_work(settings, repo, checked=False)
    # someone's number is read in the work, so the people nobody confirmed are left out
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: True)
    at = ok(app(session={"assignment_id": aid, "step": 6}))
    page = screen_text(at)
    assert "PDFに入っていない参加者（3人）" in page and "同一人物の確認が取れていない" in page
    assert "背番号の読み取りか" in page and "「人を確かめる」で確かめると、PDFに入ります" in page
    assert "作業の設定" in page                                         # the other way out
    assert primaries(at) == ["参加者用のPDFを受け取る"]
    assert next(b for b in at.button if b.key == "pfb_pdf_student").disabled
    no_approval_words(page)


def test_a_work_without_any_clue_prints_everyone_and_says_how_to_check(env):
    """No number read for anybody and nobody decided on 「人を確かめる」: everyone is printed
    (with the usual correct_later labels), and each person's screen says how to avoid a mix-up."""
    settings, repo = env
    aid, sid = live_work(settings, repo, checked=False)
    work = repo.get_assignment(aid)
    assert not G.work_has_clue(repo, work)
    at = ok(app(session={"assignment_id": aid, "step": 6}))
    page = screen_text(at)
    assert "PDFに入っている参加者（3人）" in page and "PDFに入っていない参加者" not in page
    assert primaries(at) == ["参加者用のPDFを受け取る"]
    assert not any(b.key == "pfb_pdf_student" for b in at.button)      # the download, not the stub
    no_approval_words(page)
    text = text_of(render_student_pdf(repo, aid))
    assert "背番号の読み取りも「人を確かめる」の記録もない作業のため" in text
    assert "AIが作成した文です" in text
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    hint = G.CHECK_HINT + "。"
    assert hint in [c.value for c in at.caption]                     # on the person's screen
    # once the evaluator decided the people, the work has a clue and the hint goes away;
    # everyone is still printed, now as checked by a person
    assert review.mark_people_checked(repo, sid)
    assert G.work_has_clue(repo, repo.get_assignment(aid))
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    assert hint not in [c.value for c in at.caption]
    text = text_of(render_student_pdf(repo, aid))
    assert "「人を確かめる」で確かめた人です" in text and "記録もない作業のため" not in text


def test_a_mapping_saved_on_the_people_step_counts_as_checked(env, monkeypatch):
    """The fixture's mapping was saved by a person (source teacher): printed, with the note,
    even when someone else's number is read in the work."""
    settings, repo = env
    aid, sid = live_work(settings, repo)
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: True)
    at = ok(app(session={"assignment_id": aid, "step": 6}))
    page = screen_text(at)
    assert "PDFに入っている参加者（3人）" in page and "PDFに入っていない参加者" not in page
    assert "「人を確かめる」で確かめた人です" in text_of(render_student_pdf(repo, aid))


def test_the_person_box_shows_the_new_state_after_a_correction(env):
    """More than six people: a box. Its words come from the state, and the box is drawn again
    when they change (a browser keeps showing the words it had when the person was picked)."""
    from pdf_synthetic import video_group, work
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=7, seconds=30)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    at = ok(app(session={"assignment_id": aid, "step": 5}))

    def box():
        return next(w for w in at.selectbox if (w.key or "").startswith("pfb_student_"))

    third = [s["id"] for s in repo.list_subjects(sid)][2]
    before = box()
    before.set_value(third).run()
    first_key = box().key
    cor.edit_text(repo, third, "直した文です。")
    at.run()
    assert box().key != first_key and box().value == third         # drawn again, same person
    assert sum("訂正済み 版1" in o for o in box().options) == 1


def test_other_screens_of_a_live_work_have_no_approval_words(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    at = ok(app())                                     # the home screen's card
    assert "3人の文ができました" in screen_text(at)
    assert not any("まとめて見る" in b.label for b in at.button)   # nothing to approve in bulk
    for step in (1, 4):
        at = ok(app(session={"assignment_id": aid, "step": step}))
        text = screen_text(at)
        if step == 1:     # the switch itself names the other way (the only place)
            text = text.replace(cor.MODE_WORDS[cor.APPROVE], "")
        no_approval_words(text)


# --- 作業の設定: the mode switch and the 出典 ------------------------------------------------------

def test_steps_5_and_6_show_the_way_back_to_check_people(env, monkeypatch):
    """Design (2026-11-23): the bar stays; steps 5 and 6 say that checking people puts more of
    them in the PDF, with a button back to 「人を確かめる」."""
    from pfb_app.ui.wizard import step_pdf
    settings, repo = env
    aid, sid = live_work(settings, repo, checked=False)
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: True)
    for step in (6, 5):
        at = ok(app(session={"assignment_id": aid, "step": step}))
        assert step_pdf.MORE_BY_CHECK in screen_text(at)
        btn(at, step_pdf.TO_PEOPLE).click().run()
        assert shown_step(at) == 3
    # a correction of someone the PDF leaves out never says 「次に受け取るPDFに入ります」
    hanako = subject(repo, sid, "Pa")["id"]
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    btn(at, "文を直す").click().run()
    next(t for t in at.text_area if t.key == f"pfb_fix_text_{hanako}").set_value(
        "花子さんは柱をていねいに折り重ねて、塔の土台を作りました。")
    btn(at, "直した文を保存する").click().run()
    page = screen_text(ok(at))
    assert "いまはPDFに入っていません" in page and "次に受け取るPDFに入ります。" not in page
    # nobody left out: neither the line nor the button
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: False)
    at = ok(app(session={"assignment_id": aid, "step": 6}))
    assert step_pdf.MORE_BY_CHECK not in screen_text(at) and not has_button(at, step_pdf.TO_PEOPLE)


CHANGED = "の文か確認状況が変わりました"     # the 「再出力が必要」 line of steps 5 and 6 (not the help)


def test_reoutput_only_for_people_the_pdf_carries(env, monkeypatch):
    """「再出力が必要」 is said when the text or check state of someone in the PDF changed, never
    for a person the PDF leaves out (here: not confirmed as one person)."""
    settings, repo = env
    aid, sid = live_work(settings, repo, checked=False)
    pa, pb = subject(repo, sid, "Pa")["id"], subject(repo, sid, "Pb")["id"]
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: True)
    monkeypatch.setattr(G, "assess", lambda repo_, sub, subj, chain, pol=None: G.Verdict(
        G.OK, [], {}, {e["event_id"] for _c, src in chain for e in (src.output or {}).get("events") or []})
        if subj["id"] == pa else G.Verdict(G.DOUBT, ["確かめる手がかりがありません"], {}, set()))
    assert cor.in_pdf_now(repo, aid) == {pa}
    mark_pdf_downloaded(repo, aid, sorted(cor.in_pdf_now(repo, aid)))
    assert cor.pdf_people(repo, aid) == {pa}
    assert work_progress(repo, aid).complete
    cor.edit_text(repo, pb, "太郎さんは土台を両手で押さえて、塔を守りました。")   # not in the PDF
    stamp = repo.get_assignment(aid)["pdf_downloaded_at"]
    assert cor.changed_since(repo, aid, stamp).keys() == {pb}
    assert cor.changed_since(repo, aid, stamp, pdf_only=True) == {}
    assert work_progress(repo, aid).complete
    for step in (5, 6):
        assert CHANGED not in screen_text(ok(app(session={"assignment_id": aid, "step": step})))
    cor.report_wrong(repo, pa, "動作が違う")      # in the received PDF (now out of the next one)
    assert cor.changed_since(repo, aid, stamp, pdf_only=True).keys() == {pa}
    assert not work_progress(repo, aid).complete
    for step in (5, 6):
        assert CHANGED in screen_text(ok(app(session={"assignment_id": aid, "step": step})))
    # a download that did not record who it carried: every change counts, as before
    mark_pdf_downloaded(repo, aid)
    cor.edit_text(repo, pb, "太郎さんは土台を押さえました。")
    assert cor.changed_since(repo, aid, repo.get_assignment(aid)["pdf_downloaded_at"],
                             pdf_only=True).keys() == {pb}


def test_settings_switch_the_mode_and_store_the_credits(env, gate_ok):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    at = ok(app(session={"assignment_id": aid, "step": 1}))
    mode = next(r for r in at.radio if r.key == f"pfb_review_mode_{aid}")
    assert mode.value == cor.CORRECT_LATER
    credits = next(t for t in at.text_area if t.key == f"pfb_credits_{aid}")
    credits.set_value("映像：テスト用の動画（CC BY 4.0）")
    btn(at, "出典を保存する").click().run()
    ok(at)
    assert repo.get_assignment(aid)["credits"] == "映像：テスト用の動画（CC BY 4.0）"
    assert "出典を保存しました" in screen_text(at)
    assert "映像：テスト用の動画（CC BY 4.0）" in text_of(render_student_pdf(repo, aid))
    next(t for t in at.text_area if t.key == f"pfb_credits_{aid}").set_value("  ")
    btn(at, "出典を保存する").click().run()
    ok(at)
    assert repo.get_assignment(aid)["credits"] is None and "出典を空にしました" in screen_text(at)
    mode = next(r for r in at.radio if r.key == f"pfb_review_mode_{aid}")
    mode.set_value(cor.APPROVE).run()
    ok(at)
    assert repo.get_assignment(aid)["review_mode"] == cor.APPROVE
    p = work_progress(repo, aid)
    assert p.steps[4] == "確認して承認" and p.reason == "3人の承認を待っています"


# --- the PDFs -----------------------------------------------------------------------------------

def test_zip_and_contact_of_a_live_work(env, gate_ok):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    names = zipfile.ZipFile(io.BytesIO(render_student_zip(repo, aid))).namelist()
    assert len([n for n in names if n.endswith(".pdf")]) == 3       # was empty before
    assert labels.CONTACT_MISSING in text_of(render_student_pdf(repo, aid))
    org.save(settings, name="", accent="blue", contact="  テスト窓口　000-0000 ")
    text = text_of(render_student_pdf(repo, aid))
    assert "訂正のご連絡先：テスト窓口 000-0000" in text and labels.CONTACT_MISSING not in text
    assert org.load(settings).contact == "テスト窓口 000-0000"
    org.save(settings, name="x", accent="blue")                    # left out: kept
    assert org.load(settings).contact == "テスト窓口 000-0000"
    with pytest.raises(org.OrgError):
        org.save(settings, name="x", accent="blue", contact="長" * (org.MAX_CONTACT + 1))
    teacher = text_of(render_teacher_pdf(repo, aid))
    no_approval_words(teacher)


def test_approve_work_pdf_has_no_live_lines_and_stored_credits_print(env):
    from test_export import add_video_subject
    settings, repo = env
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    text = text_of(render_student_pdf(repo, aid))
    assert labels.AI_LINE not in text and labels.CREDIT_MISSING not in text
    repo.set_credits(aid, "出典A\n\n出典B ")
    text = text_of(render_student_pdf(repo, aid))
    assert "出典A" in text and "出典B" in text
    repo.set_credits(aid, "  ")
    assert repo.get_assignment(aid)["credits"] is None


# --- the other screens of a 「あとから訂正」 work ---------------------------------------------------

def test_a_viewer_sees_the_texts_their_states_and_the_pdf(env, gate_ok, monkeypatch):
    from pfb_app.ui import access
    settings, repo = env
    aid, sid = live_work(settings, repo)
    cor.report_wrong(repo, subject(repo, sid, "Pb")["id"], "動作が違う")
    monkeypatch.setattr(access, "can_edit", lambda: False)
    at = ok(app(session={"assignment_id": aid}))
    page = screen_text(at)
    assert "AIが作った文" in page and TEXTS["Pa"] in page and cor.STATE_REPORT in page
    assert primaries(at) == ["参加者用のPDFを受け取る"]
    assert not has_button(at, "文を直す")
    no_approval_words(page)


def test_the_admin_review_page_of_a_live_work_saves_a_correction(env):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import pfb_app.ui
    settings, repo = env
    aid, sid = live_work(settings, repo)
    at = AppTest.from_file(str(Path(pfb_app.ui.__file__).parent / "pages" / "review.py"),
                           default_timeout=60).run()
    at.selectbox[0].select(aid).run()
    assert not at.exception, [e.value for e in at.exception]
    text = screen_text(at).replace("確認・承認", "")     # the page's own name (every work)
    no_approval_words(text)
    assert "訂正なし" in text and not [t for t in at.toggle if "自動承認" in t.label]
    hanako = subject(repo, sid, "Pa")["id"]
    at.text_area(key=f"manual-{hanako}").input("花子さんは柱をていねいに折り重ねて、塔の土台を作りました。")
    at.button(key=f"manual-save-{hanako}").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert cor.state(repo, hanako).label == "訂正済み 版1"
    assert repo.get_subject(hanako)["approval_status"] != "approved"


def test_a_document_work_prints_without_the_number_check(env):
    """A document has no video identity to confirm (no jersey numbers, no tracking): the
    same-person check of the videos does not keep its writer out of the PDF."""
    from test_export import judgment
    from tips_helpers import save
    settings, repo = env
    aid = create_assignment_from_form(repo, "防災のレポート", "document", "防災を調べる", "", None,
                                      "standard", False, settings)
    assert repo.get_assignment(aid)["review_mode"] == cor.CORRECT_LATER      # the new default
    sid = repo.create_submission(aid, "A", {"source_path": "/x", "original_filename": "r.pdf"})
    writer = repo.add_subject(sid, None, "テスト花子", None, {})
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x"})
    items = [{"page": 3, "quote": "避難経路を確認する", "observed_content": "c",
              "evidence_source": "text", "confidence": "high", "item_id": "p3-001",
              "quote_verified": True}]
    ev = save(repo, sid, Stage.evidence, {"items": items}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [judgment("p3-001")]},
               upstream=[ev.id])
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "避難経路をよく調べていました。",
                                "evidence_refs": ["p3-001"]}, subject_id=writer, upstream=[con.id])
    text = text_of(render_student_pdf(repo, aid))
    assert "避難経路をよく調べていました" in text and "確認状況：訂正なし" in text
    assert "同一人物" not in text
    from pfb_app.ui.wizard.step_pdf import who_live
    assert who_live(repo, aid) == ([(writer, "テスト花子")], [])
