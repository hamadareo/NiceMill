"""PDF export: only approved PFBs, teacher/student layouts, Japanese text embedded."""
from pathlib import Path

from pdfhelp import pdf_font_names, pdf_page_count, pdf_text
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.export.pdf import (ApprovedEntry, ExportError, collect_approved, name_order,
                                render_student_pdf, render_teacher_pdf)
from pfb_app.services.review import approve, edit_stage_output, write_manual_pfb
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def repo(tmp_path):
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def event(n, behavior="紙を折る", camera="cam1", participant="Pa"):
    return {"video_time": "00:05", "participant": participant, "observed_behavior": behavior,
            "utterance": "", "speaker": "unknown", "evidence_source": "video",
            "confidence": "high", "event_id": f"{camera}-{n:04d}", "camera": camera,
            "task_elapsed": "00:00"}


def judgment(ref, positive=True, text="解釈の文"):
    return {"ref_id": ref, "contribution_interpretation": text,
            "is_positive_contribution": positive, "rubric_criteria": []}


def video_assignment(repo, **kw):
    return repo.create_assignment("塔づくり課題", Kind.video,
                                  "紙で塔を作る課題の説明文です。" * 3, "r", "standard", False, **kw)


def add_video_subject(repo, aid, name, pfb_text="積極的に取り組んでいました。", approved=True):
    sid = repo.create_submission(aid, name, {"cameras": {"cam1": "/x"}, "start_sec": None,
                                             "end_sec": None, "auto_transcribe": False})
    subj = repo.add_subject(sid, "Pa", name, None, {})
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    ident = save(repo, sid, Stage.identity, {"participants": [{
        "participant_id": "Pa", "appearance": "外見", "initial_position": "左",
        "verification_points": [], "identity_confidence": "high", "verified": True,
        "notes": ""}]}, camera="cam1", upstream=[prep.id])
    beh = save(repo, sid, Stage.behavior,
               {"events": [event(1, "紙を折る"), event(2, "周りを見る")]}, camera="cam1",
               upstream=[ident.id, prep.id])
    con = save(repo, sid, Stage.contribution,
               {"judgments": [judgment("cam1-0001"), judgment("cam1-0002", False, "否定")]},
               camera="cam1", upstream=[beh.id])
    pfb = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": pfb_text,
                                      "evidence_refs": ["cam1-0001"]},
               subject_id=subj, upstream=[con.id])
    repo.set_identity_confirmed(sid, True)
    if approved:
        approve(repo, subj)
    return dict(sid=sid, subj=subj, beh=beh, con=con, pfb=pfb)


def text_of(pdf: bytes) -> str:
    return pdf_text(pdf)


def pages_of(pdf: bytes) -> int:
    return pdf_page_count(pdf)


def test_only_approved_included(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    add_video_subject(repo, aid, "未承認太郎", approved=False)
    entries = collect_approved(repo, aid)
    assert [e.display_name for e in entries] == ["山田花子"]
    assert isinstance(entries[0], ApprovedEntry) and not entries[0].manual
    assert entries[0].judgments_false_count == 1
    student = text_of(render_student_pdf(repo, aid))
    assert "山田花子" in student and "未承認太郎" not in student
    # the evaluator PDF lists everyone on its contact sheet, the others as not in the PDF,
    # and has a page only for the approved
    from pfb_app.export.pdf import _html
    teacher = _html("teacher.html.j2", repo, aid)
    assert "入っていない" in teacher and teacher.count('<section class="person-page">') == 1


def test_stale_approval_excluded_from_both_pdfs(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "残る人")
    stale = add_video_subject(repo, aid, "古くなった人")
    edit_stage_output(repo, stale["beh"].id, {"events": [event(1), event(2)]})
    # the DB column still says approved; only subject_status demotes it
    assert repo.get_subject(stale["subj"])["approval_status"] == "approved"
    assert [e.display_name for e in collect_approved(repo, aid)] == ["残る人"]
    student = text_of(render_student_pdf(repo, aid))
    assert "古くなった人" not in student and "残る人" in student
    from pfb_app.export.pdf import _html
    teacher = _html("teacher.html.j2", repo, aid)
    assert teacher.count('<section class="person-page">') == 1
    page = teacher.split('<section class="person-page">')[1]
    assert "残る人" in page and "古くなった人" not in page


def test_approved_run_text_is_used(repo):
    aid = video_assignment(repo)
    v = add_video_subject(repo, aid, "山田花子", pfb_text="承認した本文です。")
    # a newer, unapproved run exists -> subject is no longer approved at all
    save(repo, v["sid"], Stage.pfb, {"pfb_status": "generated", "pfb": "新しい本文",
                                     "evidence_refs": []}, subject_id=v["subj"],
         upstream=[v["con"].id])
    assert collect_approved(repo, aid) == []
    with pytest.raises(ExportError):
        render_student_pdf(repo, aid)


def test_student_pdf_one_page_per_student_and_no_internal_ids(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    add_video_subject(repo, aid, "鈴木次郎")
    pdf = render_student_pdf(repo, aid)
    assert pages_of(pdf) == 2
    t = text_of(pdf)
    for bad in ("Pa", "cam1", "high"):
        assert bad not in t
    assert "塔づくり課題" in t and "積極的に取り組んでいました" in t
    # the handout shows the moment the text cites, never the ones judged otherwise
    assert "紙を折る" in t and "00:05" in t and "周りを見る" not in t


def test_teacher_pdf_contains_evidence(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    t = text_of(render_teacher_pdf(repo, aid))
    assert "塔づくり課題" in t and "紙で塔を作る" in t
    assert "00:05" in t and "紙を折る" in t and "解釈の文" in t
    assert "周りを見る" not in t and "否定" not in t


def test_teacher_pdf_document_evidence(repo):
    aid = repo.create_assignment("防災課題", Kind.document, "防災を調べる", "r", "standard", False)
    sid = repo.create_submission(aid, "A", {"source_path": "/x", "original_filename": "r.pdf"})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x"})
    items = [{"page": 3, "quote": "避難経路を確認する", "observed_content": "c",
              "evidence_source": "text", "confidence": "high", "item_id": "p3-001",
              "quote_verified": True},
             {"page": 4, "quote": "", "observed_content": "避難所の地図", "evidence_source": "figure",
              "confidence": "high", "item_id": "p4-001", "quote_verified": False}]
    ev = save(repo, sid, Stage.evidence, {"items": items}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution,
               {"judgments": [judgment("p3-001"), judgment("p4-001")]}, upstream=[ev.id])
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "調べていました。",
                                "evidence_refs": ["p3-001"]}, subject_id=subj, upstream=[con.id])
    approve(repo, subj)
    t = text_of(render_teacher_pdf(repo, aid))
    assert "避難経路を確認する" in t and "避難所の地図" in t and "3" in t


def test_manual_pfb_flagged_and_escaped(repo):
    aid = video_assignment(repo)
    v = add_video_subject(repo, aid, "山田花子", approved=False)
    write_manual_pfb(repo, v["subj"], "<b>太字</b>&を書きました")
    approve(repo, v["subj"])
    entries = collect_approved(repo, aid)
    assert entries[0].manual
    assert "<b>太字</b>&を書きました" in text_of(render_student_pdf(repo, aid))


def test_japanese_text_extractable(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子", pfb_text="最後まで工夫していました。")
    for pdf in (render_teacher_pdf(repo, aid), render_student_pdf(repo, aid)):
        assert "していました" in text_of(pdf)
        assert any("Noto" in n for n in pdf_font_names(pdf))


def test_no_approved_raises(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "未承認", approved=False)
    for fn in (render_teacher_pdf, render_student_pdf):
        with pytest.raises(ExportError, match="承認済みのフィードバックがありません"):
            fn(repo, aid)


def test_video_entries_only_contain_own_participant_evidence(repo):
    aid = video_assignment(repo)
    cams = {"cam1": "/x", "cam2": "/y"}
    sid = repo.create_submission(aid, "G", {"cameras": cams, "start_sec": None, "end_sec": None,
                                            "auto_transcribe": False})
    subjects = {p: repo.add_subject(sid, p, name, None, {})
                for p, name in (("Pa", "山田花子"), ("Pb", "鈴木次郎"))}
    cons = []
    for cam in cams:
        prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        ident = save(repo, sid, Stage.identity, {"participants": [
            {"participant_id": p, "appearance": "外見", "initial_position": "左",
             "verification_points": [], "identity_confidence": "high", "verified": True,
             "notes": ""} for p in subjects]}, camera=cam, upstream=[prep.id])
        events = [event(1, f"{cam}Pa良い行動", cam, "Pa"), event(2, f"{cam}Pa悪い行動", cam, "Pa"),
                  event(3, f"{cam}Pb良い行動", cam, "Pb"), event(4, f"{cam}Pb悪い行動", cam, "Pb")]
        beh = save(repo, sid, Stage.behavior, {"events": events}, camera=cam,
                   upstream=[ident.id, prep.id])
        js = [judgment(e["event_id"], "良い" in e["observed_behavior"],
                       "解釈" + e["event_id"]) for e in events]
        cons.append(save(repo, sid, Stage.contribution, {"judgments": js}, camera=cam,
                         upstream=[beh.id]))
    repo.set_identity_confirmed(sid, True)
    for p, subj in subjects.items():
        save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": f"{p}の本文",
                                    "evidence_refs": []}, subject_id=subj,
             upstream=[c.id for c in cons])
        approve(repo, subj)
    entries = {e.display_name: e for e in collect_approved(repo, aid)}
    assert {e.judgments_false_count for e in entries.values()} == {2}
    assert len(entries["山田花子"].judgments_true) == 2
    from pfb_app.export.pdf import _html
    t = _html("teacher.html.j2", repo, aid)
    mine, theirs = t.split('<section class="person-page">')[1:]
    assert "山田花子" in mine and "鈴木次郎" in theirs
    assert "Pa良い行動" in mine and "Pb良い行動" not in mine
    assert "Pb良い行動" in theirs and "Pa良い行動" not in theirs
    assert "悪い行動" not in t
    assert text_of(render_teacher_pdf(repo, aid))



def test_student_pdf_html_general_words(repo):
    from pfb_app.export.pdf import _html
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    student = _html("student.html.j2", repo, aid)
    assert "活動「塔づくり課題」のふりかえり" in student and "評価者が確かめたもの" in student
    teacher = _html("teacher.html.j2", repo, aid)
    assert "参加者の一覧" in teacher and "開始から 00:05" in teacher
    for html in (student, teacher):
        assert "試合" not in html and "選手" not in html and "コーチ" not in html


def test_student_pdf_html_sports_words(repo):
    from pfb_app.export.pdf import _html
    aid = video_assignment(repo, domain="sports", sport="soccer")
    add_video_subject(repo, aid, "山田花子")
    html = _html("student.html.j2", repo, aid)
    assert "試合・練習「塔づくり課題」のふりかえり" in html and "活動「" not in html
    assert "コーチが確かめたもの" in html
    teacher = _html("teacher.html.j2", repo, aid)
    # no game clock was given: the times are from the start, never 試合時間
    assert "開始から 00:05" in teacher and "試合時間" not in teacher
    assert "選手の一覧" in teacher


def test_sports_teacher_pdf_uses_the_game_clock(repo):
    from pfb_app.export.pdf import _html
    aid = video_assignment(repo, domain="sports", sport="soccer")
    r = add_video_subject(repo, aid, "山田花子")
    sub = repo.get_submission(r["sid"])
    repo.update_submission_meta(r["sid"], {**sub["meta"], "clock_offset_min": 65.5})
    html = _html("teacher.html.j2", repo, aid)
    # 65:30 at the start + 5 s into the analysed part
    assert "試合時間 65:35" in html and "試合時間 00:05" not in html and "開始から" not in html


def test_names_are_listed_in_number_order():
    names = ["#10", "選手10", "#7", "#10（カメラ2）", "選手2", "山田", "#00", "#0"]
    assert sorted(names, key=name_order) == [
        "#0", "#00", "#7", "#10", "#10（カメラ2）", "山田", "選手2", "選手10"]
