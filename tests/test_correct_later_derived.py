"""「あとから訂正」 (correct_later) in the features that used only approved texts: each uses the
newest text that nobody reported wrong (services.corrections.usable_run), what the participants
receive says that the text is the AI's, and the automatic deletion never goes further than in an
「承認して出す」 work. Synthetic data only: invented names and texts, tiny generated videos."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_helpers import hand_out, iso
from growth_synthetic import add_work
from pdfhelp import pdf_text
from pfb_app.config import Settings
from pfb_app.domain import vocab
from pfb_app.domain.models import Stage
from pfb_app.export import explainer_video as ev
from pfb_app.export import growth as record
from pfb_app.export import identity_gate, labels, pdf, views
from pfb_app.pipeline import plain as pipeline_plain
from pfb_app.services import (ask, dashboard, explainer, growth, pending_review, plain_text,
                              quality_stats, retention, tips)
from pfb_app.services import corrections as cor
from pfb_app.storage.db import Repo, connect, init_schema
from test_correct_later_ui import live_work, no_approval_words
from test_export import add_video_subject
from test_export import video_assignment as export_work
from test_tips_pdf import ONE, TEAM, made
TEAM_OK = "パスのあとに全員で声をかけ合ってみる"     # TEAM has a wording fault (「直して」)
from tips_helpers import save as save_tips
from test_wizard_ui_basic import app, env, ok, screen_text  # noqa: F401 (env)
from test_wizard_ui_people_wait import save
from test_wizard_ui_review_pdf import TEXTS, subject
from pfb_app.ui.wizard import waiting_panel as panel

NOW = datetime(2026, 10, 31, 3, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                               prompts_dir=Path(__file__).resolve().parents[1] / "prompts"))


def ids(repo, sid):
    return [subject(repo, sid, p)["id"] for p in ("Pa", "Pb", "Pc")]


# --- the text in use ------------------------------------------------------------------------

def test_the_text_used_is_the_approved_one_or_the_newest_nobody_reported(repo):
    aid = export_work(repo)
    waiting = add_video_subject(repo, aid, "#8", approved=False)
    done = add_video_subject(repo, aid, "#9")
    assert cor.usable_run(repo, repo.get_subject(waiting["subj"])) is None
    assert cor.usable_run(repo, repo.get_subject(done["subj"])).id == done["pfb"].id
    live = export_work(repo)
    a = add_video_subject(repo, live, "#10", pfb_text="最初の文です。", approved=False)
    empty = add_video_subject(repo, live, "#11", pfb_text="", approved=False)
    cor.set_mode(repo, live, cor.CORRECT_LATER)

    def now():
        return cor.usable_run(repo, repo.get_subject(a["subj"]))

    assert now().id == a["pfb"].id                                   # nobody approves here
    assert cor.usable_run(repo, repo.get_subject(empty["subj"])) is None
    assert cor.unusable_why(repo, repo.get_subject(empty["subj"])) == cor.NO_TEXT_LIVE
    cor.report_wrong(repo, a["subj"], "動作が違う")
    assert now() is None
    assert cor.unusable_why(repo, repo.get_subject(a["subj"])) == cor.REPORTED_LIVE
    fixed = cor.edit_text(repo, a["subj"], "直した文です。")          # a correction ends the report
    assert now().id == fixed.id
    cor.report_wrong(repo, a["subj"], "その他")
    cor.resolve_report(repo, a["subj"])                              # looked at and closed
    assert now().id == fixed.id


# --- 成長の記録 -------------------------------------------------------------------------------

def _two_live_works(repo):
    a1, _, one = add_work(repo, "定例会1", {"山田": ["発言"]}, approve=False, day="2026-09-01")
    a2, _, two = add_work(repo, "定例会2", {"山田": ["発言", "発言"]}, approve=False,
                          day="2026-09-08")
    for aid in (a1, a2):
        cor.set_mode(repo, aid, cor.CORRECT_LATER)
    return one["山田"], two["山田"]


def test_the_growth_record_uses_the_texts_in_use_and_says_they_are_the_ai_s(repo):
    first, second = _two_live_works(repo)
    person = growth.link_manual(repo, first, None)
    growth.link_manual(repo, second, person)
    rec = record.for_person(repo, person)
    assert rec.live and [e.approved for e in rec.entries] == [True, True]
    assert [e.counts for e in rec.entries] == [{"発言": 1}, {"発言": 2}]
    assert all(e.live_label == cor.STATE_NONE and not e.auto for e in rec.entries)
    assert record.statements(rec)[0].startswith("作業2件の記録があります")
    page = views.growth_page(rec)
    assert page["footer"] == views.GROWTH_FOOTER_LIVE
    assert [w["state"] for w in page["works"]] == ["AIが作成した文・確認状況：訂正なし"] * 2
    # a text reported wrong is not counted (the work stays in the list, without numbers)
    cor.report_wrong(repo, second, "人物の取り違え")
    rec = record.for_person(repo, person)
    assert [e.approved for e in rec.entries] == [True, False] and rec.entries[1].counts == {}
    assert record.statements(rec) == [record.ONE_WORK_ANY]
    page = views.growth_page(rec)
    for line in [*page["lines"], page["footer"], *(w["state"] for w in page["works"])]:
        assert "承認" not in line, line
    # corrected: counted again, with its version
    cor.edit_text(repo, second, "山田さんは、意見をはっきり伝えていました。")
    rec = record.for_person(repo, person)
    assert [e.approved for e in rec.entries] == [True, True]
    assert rec.entries[1].live_label == "訂正済み 版1"


def test_linking_needs_a_text_nobody_reported_wrong(repo):
    first, second = _two_live_works(repo)
    cor.report_wrong(repo, second, "動作が違う")
    for link in (lambda: growth.accept(repo, second, first),
                 lambda: growth.link_manual(repo, second, None)):
        with pytest.raises(ValueError, match=re.escape(growth.LIVE_ONLY)):
            link()
    assert growth.link_manual(repo, first, None)                 # the other one links


# --- 試合・活動のまとめ and 記録への質問 ---------------------------------------------------------

def test_the_board_of_a_live_work_shows_each_person_s_check_state(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, pb, _ = ids(repo, sid)
    cor.report_wrong(repo, pa, "動作が違う")
    board = dashboard.build(repo, aid, sid)
    assert board.live
    by = {p.subject_id: p for p in board.people}
    assert not by[pa].approved and by[pa].text is None and by[pa].status == cor.STATE_REPORT
    assert by[pb].approved and by[pb].text == TEXTS["Pb"] and by[pb].status == cor.STATE_NONE
    states = {m.subject_id: (m.approved, m.state) for m in board.moments}
    assert states[pa] == (False, cor.STATE_REPORT) and states[pb] == (True, cor.STATE_NONE)
    rows = dashboard.moments_csv(board).decode("utf-8-sig")
    assert cor.STATE_REPORT in rows and "承認" not in rows


def _live_record():
    cats = {c.key: c for c in labels.SPORTS_CATEGORIES}
    people = [dashboard.Person(1, "山田花子", None, True, cor.STATE_NONE, "よく声をかけていました。",
                               0, {}, 0, None, None, None, None),
              dashboard.Person(2, "鈴木一郎", None, False, cor.STATE_REPORT, None,
                               0, {}, 0, None, None, None, None)]
    moments = [dashboard.Moment(1, "山田花子", None, 30.0, "開始から 00:30", cats["voice"],
                                "本人が声をかけた", True, "cam1", 0, True, cor.STATE_NONE),
               dashboard.Moment(2, "鈴木一郎", None, 60.0, "開始から 01:00", cats["voice"],
                                "本人が呼びかけた", True, "cam1", 0, False, cor.STATE_REPORT)]
    board = dashboard.Board(
        work={"id": 1, "domain": "sports"}, submission_id=1, group_label="A",
        words=vocab.words("sports"), categories=labels.SPORTS_CATEGORIES, sports=True,
        court=False, duration_s=600.0, clock_offset_s=None, start_s=0.0, people=people,
        moments=moments, media_removed=False, retracking=False, cameras=["cam1"], videos={},
        frame_sizes={}, live=True)
    return ask.from_board(board)


def test_asking_about_a_live_work_never_says_approval():
    rec = _live_record()
    f = ask.facts(ask.Plan(intent="feedback", person="鈴木一郎"), rec)
    assert f.kind == "message" and cor.STATE_REPORT in f.message and "承認" not in f.message
    assert "よく声" not in f.message
    f = ask.facts(ask.Plan(intent="feedback", person="山田花子"), rec)
    assert f.kind == "text" and "よく声をかけていました。" in f.message
    f = ask.facts(ask.Plan(intent="count", category="voice"), rec)
    assert ask.LIVE_WAITING_NOTE in f.notes and not any("承認" in n for n in f.notes)


# --- 次にためしてみること ---------------------------------------------------------------------

def model_tips(repo, sid, pid, items):
    """The model's own tips of one person, made from their newest text (as the worker does)."""
    s = subject(repo, sid, pid)
    pfb = repo.latest_run(sid, Stage.pfb, subject_id=s["id"])
    out = {"tips_status": "generated", "pfb_run_id": pfb.id,
           "items": [{"text": t, "evidence_refs": []} for t in items]}
    return save_tips(repo, sid, Stage.tips, out, subject_id=s["id"], upstream=pfb.upstream_run_ids,
                     source="model")


def model_team_tips(repo, sid, items):
    """The model's tips of the group: on the latest contribution and everyone's texts."""
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    texts = [repo.latest_run(sid, Stage.pfb, subject_id=s["id"]).id for s in repo.list_subjects(sid)]
    out = {"tips_status": "generated", "items": [{"text": t, "evidence_refs": []} for t in items]}
    return save_tips(repo, sid, Stage.tips, out, upstream=[con.id, *texts], source="model")


def test_live_pdfs_print_the_tips_an_approval_would_have_confirmed(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, pb, _ = ids(repo, sid)
    model_tips(repo, sid, "Pa", [ONE])                 # the model's own: nobody pressed anything
    assert tips.auto_ok(repo, sid, pa)
    assert tips.for_pdf(repo, sid, pa) == [] and tips.for_pdf(repo, sid, pa, live=True) == [ONE]
    model_team_tips(repo, sid, [TEAM])                 # a wording fault: an approval would not take it
    assert tips.wording(repo, sid, [TEAM]) and tips.team_for_pdf(repo, sid, live=True) == []
    model_team_tips(repo, sid, [TEAM_OK])
    assert tips.team_for_pdf(repo, sid) == [] and tips.team_for_pdf(repo, sid, live=True) == [TEAM_OK]
    text = pdf_text(pdf.render_student_pdf(repo, aid))
    assert ONE in text and TEAM_OK in text and TEAM not in text
    # reported wrong: the person's tips and the group's wait (the text stays, with its label)
    cor.report_wrong(repo, pa, "動作が違う")
    assert not tips.auto_ok(repo, sid, pa) and tips.for_pdf(repo, sid, pa, live=True) == []
    assert tips.team_for_pdf(repo, sid, live=True) == []
    text = pdf_text(pdf.render_student_pdf(repo, aid))
    assert ONE not in text and TEAM_OK not in text and cor.STATE_REPORT in text
    # a corrected text: the tips were made for the old one, so they wait for the coach
    cor.edit_text(repo, pa, "直した文です。")
    assert tips.view(repo, sid, pa).text_changed and tips.for_pdf(repo, sid, pa, live=True) == []
    # the coach's confirmation always wins
    made(repo, sid, "Pb", [ONE], source="teacher", confirmed=True)
    cor.report_wrong(repo, pb, "その他")
    assert tips.for_pdf(repo, sid, pb, live=True) == [ONE]


def test_step5_says_whether_the_tips_go_into_the_pdf(env):
    from pfb_app.ui.wizard import step_tips
    settings, repo = env
    aid, sid = live_work(settings, repo)
    for pid in ("Pa", "Pb", "Pc"):
        model_tips(repo, sid, pid, [ONE])
    at = ok(app(session={"assignment_id": aid, "step": 5}))
    page = screen_text(at)
    assert any(ONE in (t.value or "") for t in at.text_area)        # the tips' box
    assert step_tips.LIVE_AUTO in page
    no_approval_words(page)
    labels_ = [b.label for b in at.button]
    assert step_tips.DROP_LIVE in labels_ and step_tips.DROP not in labels_   # never 渡さない
    for text in (step_tips.DROP_LIVE, step_tips.TEAM_DROP_LIVE, step_tips.EMPTY_NOTE_LIVE,
                 step_tips.TEAM_EMPTY_NOTE_LIVE):
        assert "渡" not in text


# --- やさしい文章版 ----------------------------------------------------------------------------

def test_the_reworded_text_follows_the_text_in_use(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, _, _ = ids(repo, sid)
    found = pipeline_plain.approved_text(repo, repo.get_subject(pa))
    assert found is not None and found[1] == TEXTS["Pa"]
    cor.report_wrong(repo, pa, "動作が違う")
    assert pipeline_plain.approved_text(repo, repo.get_subject(pa)) is None
    with pytest.raises(plain_text.Refused) as e:
        plain_text.save_edit(repo, pa, "low", "やさしい文です。")
    assert "承認" not in str(e.value)
    fixed = cor.edit_text(repo, pa, "直した文です。")
    run, text = pipeline_plain.approved_text(repo, repo.get_subject(pa))
    assert run.id == fixed.id and text == "直した文です。"


# --- 解説動画 -------------------------------------------------------------------------------

def test_explainer_videos_use_the_text_in_use_and_say_so(env, monkeypatch):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, pb, _ = ids(repo, sid)
    texts = (explainer.NOT_APPROVED, explainer.NOT_USABLE_LIVE, explainer.NOT_PRINTED)
    assert explainer.person_reason(repo, pa, settings) not in texts
    stamp = explainer.person_stamp(repo, pa)
    cor.report_wrong(repo, pa, "動作が違う")
    assert explainer.person_reason(repo, pa, settings) == explainer.NOT_USABLE_LIVE
    cor.edit_text(repo, pa, "直した文です。")
    assert explainer.person_stamp(repo, pa) != stamp          # a video of the old text is outdated
    # nobody confirmed and someone else's number is read: not in the PDF, so no video either
    repo.conn.execute("UPDATE stage_runs SET source='auto' WHERE submission_id=? AND stage=?",
                      (sid, Stage.identity.value))
    monkeypatch.setattr(identity_gate, "work_has_clue", lambda repo_, work: True)
    assert explainer.person_reason(repo, pb, settings) == explainer.NOT_PRINTED
    lines = ev.end_lines(False, live="確認状況：訂正済み 版1　文の版：1")
    assert lines[:3] == [*ev.LIVE_END, "確認状況：訂正済み 版1　文の版：1"]
    assert not any("承認" in line for line in lines)


# --- 動画を自動で消す ---------------------------------------------------------------------------

def _handed_out(repo, received, *, live):
    aid = repo.create_assignment("練習試合", "video", "試合", "", "standard", False)
    sid = repo.create_submission(aid, "前半", {})
    subj = repo.add_subject(sid, "P1", "人1", None, {})
    run = save(repo, sid, Stage.pfb, {"pfb": "よかった。"}, subject_id=subj)
    if live:
        cor.set_mode(repo, aid, cor.CORRECT_LATER)
    else:
        repo.set_approval(subj, run.id)
    hand_out(repo, aid, received)
    return aid, subj


def test_live_works_are_never_cleared_sooner_or_more_than_approve_works(repo):
    received = NOW - timedelta(days=40)
    approve_aid, _ = _handed_out(repo, received, live=False)
    live_aid, subj = _handed_out(repo, received, live=True)
    work = repo.get_assignment(live_aid)
    videos, everything = retention.Policy(30, False), retention.Policy(30, True)
    # 「動画だけ消す」: an approve work's videos go; a live work's never go by themselves
    assert retention._due_all(repo, videos, NOW) == [approve_aid]
    assert retention.card_line(repo, work, videos, NOW) == "あとから訂正の作業のため、動画は自動では消しません"
    # 「作業ごと消す」: the same rule as an approve work ...
    assert retention._due_all(repo, everything, NOW) == [approve_aid, live_aid]
    # ... but an open report holds it
    cor.report_wrong(repo, subj, "動作が違う")
    repo.conn.execute("UPDATE text_versions SET created_at=? WHERE subject_id=?",
                      (iso(received - timedelta(minutes=30)), subj))     # reported before the PDF
    assert retention.finished_at(repo, live_aid) == received
    assert retention._due_all(repo, everything, NOW) == [approve_aid]
    assert (retention.card_line(repo, work, everything, NOW)
            == "誤りの報告を確認中のため、この作業は自動では消しません")
    # a correction after the PDF was received is activity: not finished until received again
    cor.resolve_report(repo, subj)
    assert retention.finished_at(repo, live_aid) is None
    assert retention._due_all(repo, everything, NOW) == [approve_aid]


# --- この解析の確かさ -------------------------------------------------------------------------

def test_the_quality_page_counts_live_works_by_their_corrections(env):
    from pfb_app.ui.wizard import quality_view
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, pb, _ = ids(repo, sid)
    cor.edit_text(repo, pa, "直した文です。")
    cor.report_wrong(repo, pb, "動作が違う")
    s = quality_stats.measure(repo)
    assert s.live == {quality_stats.LIVE_NONE: 2, quality_stats.LIVE_EDITED: 1,
                      quality_stats.LIVE_REDONE: 0}
    assert (s.live_total, s.live_reported, s.live_open) == (3, 1, 1)
    assert s.approved_total == 0                       # nothing is approved in a live work
    html = quality_view.live_html(s)
    assert "文を直した</dt><dd>1人・33%" in html and "訂正なし</dt><dd>2人・67%" in html
    assert "誤りの報告を受けた人：1人（うち確認中 1人）" in html
    assert "誤りがなかったという意味ではありません" in html
    assert quality_view.EMPTY not in quality_view.measured_html(s)


# --- the home screen: 誤りの報告がある人 --------------------------------------------------------

def test_the_home_screen_lists_the_texts_reported_wrong(env):
    settings, repo = env
    aid, sid = live_work(settings, repo)
    pa, _, _ = ids(repo, sid)
    assert pending_review.reported(repo, [aid]) == []
    at = ok(app())
    assert "誤りの報告がある人" not in screen_text(at)
    cor.report_wrong(repo, pa, "人物の取り違え", note="別の人の動きです")
    (work,) = pending_review.reported(repo, [aid])
    (p,) = work.people
    assert (p.subject_id, p.reason, p.note, p.text) == (pa, "人物の取り違え", "別の人の動きです",
                                                       TEXTS["Pa"])
    at = ok(app())
    page = screen_text(at)
    assert "誤りの報告がある人（1人）" in page
    assert "報告の理由：人物の取り違え（別の人の動きです）" in page and TEXTS["Pa"] in page
    no_approval_words(panel.REPORTS + panel.REPORTS_INTRO + panel.REPORT_WHY)
    at.button(key=f"pfb_wait_open_live_{aid}").click().run()
    assert "一人ずつ、AIの文を読んでください" in screen_text(at)          # step 5 of the work
    cor.resolve_report(repo, pa)
    assert pending_review.reported(repo, [aid]) == []
