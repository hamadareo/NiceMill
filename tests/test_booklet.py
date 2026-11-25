"""The booklet output: cover, one spread per person (left page even), back cover; a multiple of 4."""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdfhelp import pdf_page_count  # noqa: E402
from test_export import add_video_subject, repo, video_assignment  # noqa: F401,E402

import pypdfium2 as pdfium  # noqa: E402
import pytest  # noqa: E402

from pfb_app.export import booklet  # noqa: E402
from pfb_app.export.pdf import render_student_pdf  # noqa: E402

CREDIT = "Samsunspor TBF / CC BY 3.0 / 切り出し・枠の描画あり"


def _pages(pdf: bytes) -> list[str]:
    doc = pdfium.PdfDocument(pdf)
    return [re.sub(r"\s+", "", p.get_textpage().get_text_range()) for p in doc]


def _work(repo, n):
    aid = video_assignment(repo)
    for i in range(1, n + 1):
        add_video_subject(repo, aid, f"#{i}", pfb_text=f"固有の文{i}です。", approved=False)
    return aid


@pytest.mark.parametrize("n", [0, 1, 5, 9, 10])
def test_total_is_a_multiple_of_four_and_layout(repo, n):
    aid = _work(repo, n) if n else video_assignment(repo)
    pdf = booklet.build(repo, aid, credits=[CREDIT], subtitle="公開クリップ")
    pages = _pages(pdf)
    total = len(pages)
    assert total % 4 == 0 and total == booklet.plan(n)["total"]
    assert "NiceMill処理出力" in pages[0]                      # the cover
    assert "奥付" in pages[-1] and re.sub(r"\s+", "", CREDIT) in pages[-1]           # the back cover
    first = booklet.plan(n)["first"]
    for k in range(n):
        left, right = first + 2 * k, first + 2 * k + 1          # 1-based page numbers
        assert left % 2 == 0                                    # the left page is even
        label = f"背番号{k + 1}番？（推定）"
        assert f"固有の文{k + 1}です" in pages[left - 1]
        assert label in pages[left - 1] and label in pages[right - 1]   # the same person
    if n % 2 == 0:                                              # a front spread fills the gap
        assert "目次" in pages[1] and "この冊子について" in pages[2]
    else:
        assert "目次" in pages[-1]
    joined = "\n".join(pages)
    for bad in ("承認", "未承認", "下書き", "ナイスミル", "さん"):
        assert bad not in joined, bad
    # page numbers run on (the page's own number is in its footer)
    for i, text in enumerate(pages[1:], 2):
        assert re.search(rf"(?<!\d){i}(?!\d)", text), i


def test_one_page_edition_is_unchanged_and_booklet_keeps_it_separate(repo):
    aid = _work(repo, 3)
    booklet.build(repo, aid, credits=[CREDIT])
    assert pdf_page_count(render_student_pdf(repo, aid, drafts=True)) == 3


def test_imposition_order_on_a3(repo):
    aid = _work(repo, 3)                                        # 1 + 2 + 6 + 1 = ... 4n
    pdf = booklet.build(repo, aid, credits=[CREDIT])
    n = pdf_page_count(pdf)
    sheets = booklet.impose_a3(pdf)
    doc = pdfium.PdfDocument(sheets)
    assert len(doc) == n // 2
    w, h = doc[0].get_size()
    assert w > h and abs(w - 2 * 595.276) < 1                    # A3 landscape
    first = re.sub(r"\s+", "", doc[0].get_textpage().get_text_range())
    assert "NiceMill処理出力" in first and "奥付" in first       # last | 1


# a booklet made from the public clip, when one is at hand (PFB_REAL_BOOKLET_DIR/booklet.pdf)
_COPY = Path(os.environ.get("PFB_REAL_BOOKLET_DIR", "/nonexistent")) / "booklet.pdf"


@pytest.mark.skipif(not _COPY.is_file(), reason="no public-clip booklet (set PFB_REAL_BOOKLET_DIR)")
def test_real_output_fills_the_pages_and_keeps_margins():
    """On the public-clip output: every spread page fills >= 90% of the body area (cover and
    back cover >= 80%), and nothing enters the gutter or the outer margin."""
    from PIL import Image, ImageChops
    mm, sc = 72 / 25.4, 2
    doc = pdfium.PdfDocument(str(_COPY))
    assert len(doc) % 4 == 0
    for i in range(len(doc)):
        n = i + 1
        img = doc[i].render(scale=sc).to_pil().convert("RGB")
        w, h = img.size
        body = img.crop((0, 0, w, int((297 - 20) * mm * sc)))
        l, t, r, b = ImageChops.difference(body, Image.new("RGB", body.size, (255, 255, 255))).getbbox()
        occ = (b / sc / mm - 14) / 263
        assert occ >= (0.8 if n in (1, len(doc)) else 0.9), (n, occ)
        lo, hi = (15, 210 - 22) if n % 2 == 0 else (22, 210 - 15)
        assert l / sc / mm >= lo - 0.6 and r / sc / mm <= hi + 0.6, n


def test_occupancy_function_reads_the_layout(repo):
    from pfb_app.export import pdf as _pdf
    aid = _work(repo, 1)
    booklet.build(repo, aid, credits=[CREDIT])
    log = booklet._render.log[-4:]
    assert all(0.0 < occ <= 1.0 for _, _, occ, _ in log) and all(s >= 0.6 for *_, s in log)


segno = pytest.importorskip("segno")


def test_cover_is_a_book_cover_with_authors_and_a_readable_qr(repo):
    import cv2
    import numpy as np
    aid = _work(repo, 3)
    pdf = booklet.build(repo, aid, credits=[CREDIT], subtitle="公開クリップ（Samsunspor 3分版）の処理結果",
                        credit_short="素材：Samsunspor TBF（CC BY 3.0）")
    doc = pdfium.PdfDocument(pdf)
    cover = re.sub(r"\s+", "", doc[0].get_textpage().get_text_range())
    assert "濱田怜央" in cover and "酒造正樹（湘南工科大学）" in cover
    assert booklet.REPO_URL in cover and "公開準備中" not in cover and "第1版" in cover
    assert not [o for o in doc[0].get_objects() if o.type == 3]       # no photo (the QR is a drawing)
    img = np.array(doc[0].render(scale=4).to_pil().convert("L"))
    value, points, _ = cv2.QRCodeDetector().detectAndDecode(img)
    assert value == booklet.REPO_URL                                   # it reads, to the right place
    p = points.reshape(-1, 2)
    assert (p[:, 0].max() - p[:, 0].min()) / 4 / (72 / 25.4) >= 30       # at least 30 mm a side
    back = re.sub(r"\s+", "", _pages(pdf)[-1])
    assert "濱田怜央" in back and booklet.REPO_URL in back


def test_note_is_off_by_default_and_can_be_added(repo):
    aid = _work(repo, 1)
    assert "公開準備中" not in _pages(booklet.build(repo, aid))[0]
    assert "公開準備中" in _pages(booklet.build(repo, aid, repo_note="公開準備中"))[0]


def test_booklet_lists_everyone_and_never_runs_the_identity_gate(repo, monkeypatch):
    from pfb_app.export import identity_gate as G
    from pfb_app.services import corrections as cor
    monkeypatch.setattr(G, "assess", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate ran")))
    aid = _work(repo, 9)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    pages = _pages(booklet.build(repo, aid, credits=[CREDIT]))
    assert len(pages) == 20 and booklet.plan(9)["total"] == 20
    assert "人物の追跡に取り違えが含まれることがあります。" in pages[-1]
    assert "同一人物の確認" not in "".join(pages)
    for k in range(9):
        assert f"固有の文{k + 1}です" in pages[2 * k + 1]


def test_tracking_note_is_a_single_small_line_on_the_back_only(repo):
    aid = _work(repo, 2)
    pages = _pages(booklet.build(repo, aid))
    assert sum(p.count("人物の追跡に取り違えが含まれることがあります。") for p in pages) == 1
    assert "人物の追跡に取り違えが含まれることがあります。" in pages[-1]
