"""Mechanical layout audit: no text hidden by or sitting on a stray shape, no flat blobs.
(Occupancy and margin checks stay high on a broken page: a stray filled ellipse over a table
passed both. Found on the demo edition's 検出の数字 page: a class name, `num`, collided with the
base stylesheet's picture-number badge.)"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdfhelp import pdf_page_count  # noqa: E402,F401
from test_export import add_video_subject, repo, video_assignment  # noqa: F401,E402
from test_booklet_demo import CREDIT, _fake_gather, _work  # noqa: F401,E402
import overlapcheck  # noqa: E402

import pytest  # noqa: E402
from weasyprint import HTML  # noqa: E402

from pfb_app.export import booklet, demo  # noqa: E402
from pfb_app.export.pdf import render_student_pdf, render_teacher_pdf  # noqa: E402

TPL = Path(__file__).parents[1] / "src" / "pfb_app" / "export" / "templates"


def _pdf(html: str) -> bytes:
    return HTML(string=html).write_pdf()


def test_audit_flags_the_original_bug_pattern():
    """The badge class on a table: a blue oval under white text, white text off it."""
    bad = _pdf("""<style>.num{width:80mm;height:30mm;border-radius:50%;background:#1c5cab;color:#fff;
      font-weight:bold}</style><p>本文</p><table class="num"><tr><td>項目はここに長く書く文字列です</td>
      <td>値の列</td></tr></table>""")
    kinds = {h.kind for h in overlapcheck.audit(bad)}
    assert "low_contrast" in kinds


def test_audit_flags_a_shape_drawn_over_text():
    bad = _pdf("""<div style="position:relative;width:100mm;height:20mm;font-size:16pt">この文字は隠される
      <div style="position:absolute;left:0;top:0;width:100mm;height:20mm;background:#c8410c"></div></div>""")
    assert any(h.kind == "shape_over_text" for h in overlapcheck.audit(bad))


def test_audit_flags_a_big_flat_blob():
    bad = _pdf('<div style="width:150mm;height:40mm;background:#2a78d6"></div>')
    assert any(h.kind == "flat_blob" for h in overlapcheck.audit(bad))


def test_audit_passes_plain_pages_and_small_badges():
    ok = _pdf("""<style>.b{display:inline-block;width:5mm;height:5mm;border-radius:50%;background:#1c5cab;
      color:#fff;text-align:center;font-size:8pt;line-height:5mm}</style>
      <p>普通の文章です。<span class="b">1</span> 表の例</p>
      <table><tr><td style="border-bottom:1px solid #ccc">項目</td><td><b>値</b></td></tr></table>""")
    assert overlapcheck.audit(ok) == []


@pytest.mark.parametrize("n", [1, 5])
def test_demo_booklet_has_no_overlaps(repo, monkeypatch, tmp_path, n):
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    pdf = demo.build(repo, _work(repo, n), credits=[CREDIT])
    assert overlapcheck.audit(pdf) == []


def test_full_booklet_and_handouts_have_no_overlaps(repo):
    aid = _work(repo, 3)
    assert overlapcheck.audit(booklet.build(repo, aid, credits=[CREDIT])) == []
    assert overlapcheck.audit(render_student_pdf(repo, aid, drafts=True, credits=[CREDIT])) == []
    assert overlapcheck.audit(render_teacher_pdf(repo, aid, drafts=True, credits=[CREDIT])) == []


def test_booklet_classes_do_not_collide_with_the_base_stylesheet():
    """Classes the booklet's own blocks use must not be styled by the base stylesheets (the
    badge `.num`, the chip, ...): the booklet scopes its CSS under `.bk` and uses its own names."""
    base = (TPL / "_base.html.j2").read_text() + (TPL / "_blocks.html.j2").read_text()
    defined = set(re.findall(r"(?<![\w-])\.([a-z][\w-]*)", "\n".join(
        re.findall(r"<style.*?</style>|\{% block style %\}.*?\{% endblock %\}", base, re.S) or [base])))
    own = {"figs", "est", "strip", "bars", "row", "lab", "track", "fill", "val"}
    assert not (own & defined), own & defined
    page = (TPL / "booklet.html.j2").read_text()
    assert 'class="num"' not in page and "table.num" not in page


def test_bar_chart_long_label_does_not_run_under_its_bar(tmp_path):
    from pfb_app.export.charts import Bar, bar_chart
    svg = bar_chart([Bar("背番号 不明（人物ID 15）", 3), Bar("背番号 8番？（推定）（人物ID 12）", 5)],
                    title="x", unit="回", width=240)
    pdf = _pdf(f'<div style="width:240px">{svg}</div>')
    assert [h for h in overlapcheck.audit(pdf) if h.kind != "flat_blob"] == []
    from pfb_app.export.charts import _text_w
    bar_x = float(re.search(r'<line x1="([\d.]+)"', svg).group(1))
    found = re.findall(r'<text x="([\d.]+)"[^>]*?font-size="([\d.]+)"[^>]*>(背番号[^<]*)<', svg)
    assert len(found) == 2
    for x, size, label in found:
        assert float(x) + _text_w(label, float(size)) <= bar_x + 1.0, label
