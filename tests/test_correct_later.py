"""「あとから訂正」 (review_mode correct_later): output without approval, corrections kept as versions,
honest status in the PDFs; the approve mode is unchanged."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdfhelp import pdf_page_count  # noqa: E402
from test_export import add_video_subject, repo, text_of, video_assignment  # noqa: F401,E402

import pytest  # noqa: E402

from pfb_app.export import labels  # noqa: E402
from pfb_app.export.pdf import ExportError, _html, render_student_pdf, render_teacher_pdf  # noqa: E402
from pfb_app.services import corrections as cor  # noqa: E402


@pytest.fixture(autouse=True)
def _gate_ok(monkeypatch):
    """These tests are about the output without approval, not about who is confirmed (see
    test_identity_gate): everybody passes the same-person check here."""
    from pfb_app.export import identity_gate as G
    monkeypatch.setattr(G, "assess", lambda *a, **k: G.Verdict(
        G.OK, [], {}, {"cam1-0001", "cam1-0002"}))


def _work(repo, n=2):
    aid = video_assignment(repo)
    subs = [add_video_subject(repo, aid, f"選手{i + 2}", pfb_text=f"AIの文{i}です。", approved=False)
            for i in range(n)]
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    return aid, subs


def test_mode_defaults_to_approve_and_is_stored(repo):
    aid = video_assignment(repo)
    assert repo.get_assignment(aid)["review_mode"] == "approve"
    assert cor.approval_enabled(repo, aid)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    assert repo.get_assignment(aid)["review_mode"] == "correct_later"
    assert not cor.approval_enabled(repo, aid)
    with pytest.raises(cor.CorrectionError):
        cor.set_mode(repo, aid, "other")


def test_pdf_without_any_approval(repo):
    aid, _ = _work(repo)
    pdf = render_student_pdf(repo, aid)
    assert pdf_page_count(pdf) == 2                     # one page each, nobody approved
    t = text_of(pdf)
    assert "AIの文0です" in t and labels.AI_LINE in t
    assert "訂正なし" in t and "承認" not in t and "下書き" not in t and "未承認" not in t
    assert "文の版 0" in t and "出力日時" in t
    assert labels.CONTACT_MISSING in t                  # no contact set: says so
    teacher = text_of(render_teacher_pdf(repo, aid))
    assert "承認" not in teacher and "訂正なし" in teacher


def test_contact_is_printed(repo):
    aid, _ = _work(repo, 1)
    assert "訂正のご連絡先：相談窓口 03-0000" in text_of(
        render_student_pdf(repo, aid, contact="訂正のご連絡先：相談窓口 03-0000"))


def test_edit_keeps_versions_and_pdf_shows_the_newest(repo):
    aid, subs = _work(repo, 1)
    sid = subs[0]["subj"]
    cor.edit_text(repo, sid, "直した文です。", actor="コーチ", reason="動作が違う",
                  note="テスト用の訂正")
    st = cor.state(repo, sid)
    assert st.version == 1 and st.label == "訂正済み 版1" and st.corrected_at
    h = cor.history(repo, sid)
    assert len(h) == 1 and h[0].prev_text == "AIの文0です。" and h[0].text == "直した文です。"
    t = text_of(render_student_pdf(repo, aid))
    assert "直した文です" in t and "AIの文0です" not in t
    assert "訂正済み 版1" in t and "テスト用の訂正" in t and "文の版 1" in t
    cor.edit_text(repo, sid, "二度目の文です。")
    assert cor.state(repo, sid).version == 2 and len(cor.history(repo, sid)) == 2


def test_revert_restores_previous_text_and_keeps_history(repo):
    aid, subs = _work(repo, 1)
    sid = subs[0]["subj"]
    cor.edit_text(repo, sid, "直した文です。")
    cor.revert(repo, sid, actor="コーチ")
    t = text_of(render_student_pdf(repo, aid))
    assert "AIの文0です" in t and "直した文です" not in t
    h = cor.history(repo, sid)
    assert [v.kind for v in h] == ["edit", "revert"]        # nothing deleted
    assert cor.state(repo, sid).version == 2
    with pytest.raises(cor.CorrectionError):
        cor.revert(repo, add_video_subject(repo, aid, "選手9", approved=False)["subj"])


def test_report_shows_under_review_until_resolved_or_corrected(repo):
    aid, subs = _work(repo, 2)
    a, b = subs[0]["subj"], subs[1]["subj"]
    cor.report_wrong(repo, a, "背番号が違う", actor="コーチ")
    assert cor.state(repo, a).label == "誤りの報告あり・確認中"
    assert "誤りの報告あり・確認中" in text_of(render_student_pdf(repo, aid))
    with pytest.raises(cor.CorrectionError):
        cor.report_wrong(repo, a, "理由なし")
    cor.resolve_report(repo, a)
    assert cor.state(repo, a).label == "訂正なし"
    cor.report_wrong(repo, b, "その他")
    cor.edit_text(repo, b, "報告を受けて直した文です。")   # a correction closes the report
    assert cor.state(repo, b).label == "訂正済み 版1"


def test_approve_mode_unchanged_and_modes_do_not_mix(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    add_video_subject(repo, aid, "保留太郎", approved=False)
    t = text_of(render_student_pdf(repo, aid))
    assert "山田花子" in t and "保留太郎" not in t and "確かめたものです" in t
    assert labels.AI_LINE not in t and "出力日時" not in t
    # drafts=True on a correct_later work does not bring the 下書き bands (nobody is waiting)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    t2 = text_of(render_student_pdf(repo, aid, drafts=True))
    assert "下書き" not in t2 and "承認" not in t2 and "保留太郎" in t2


def test_no_text_no_export(repo):
    aid = video_assignment(repo)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    with pytest.raises(ExportError):
        render_student_pdf(repo, aid)
