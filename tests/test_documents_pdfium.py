"""Pins the behaviour of preprocess.documents after the pymupdf -> pypdfium2 swap."""
import io

import pypdfium2 as pdfium
import pytest
from PIL import Image
from pdfhelp import encrypt_pdf, make_pdf, make_pdf_bytes, pdf_page_count, pdf_text

from pfb_app.config import Settings
from pfb_app.preprocess.documents import DocumentError, extract_pages

LONG = "これは十分に長い日本語の文章です。\n" * 4


def test_text_has_unix_newlines_and_keeps_line_order(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", [LONG + "最後の行"])
    (page,) = extract_pages(pdf, tmp_path / "out", Settings())
    assert "\r" not in page.text
    lines = [ln for ln in page.text.split("\n") if ln.strip()]
    assert lines[0] == "これは十分に長い日本語の文章です。"
    assert len(lines) == 5 and lines[-1] == "最後の行"


def test_text_matches_test_reader(tmp_path):
    data = make_pdf_bytes([LONG + "一", LONG + "二"])
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(data)
    pages = extract_pages(pdf, tmp_path / "out", Settings())
    assert "".join(p.text for p in pages) == pdf_text(data)
    assert pdf_page_count(data) == len(pages) == 2


def test_empty_page_is_rendered_and_has_blank_text(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["", LONG])
    pages = extract_pages(pdf, tmp_path / "out", Settings())
    assert pages[0].text.strip() == ""
    assert pages[0].image_path is not None and pages[1].image_path is None


def test_image_page_is_detected_even_with_long_text(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", [LONG], with_image=True)
    (page,) = extract_pages(pdf, tmp_path / "out", Settings())
    assert page.image_path is not None and page.image_path.name == "page_0001.png"


@pytest.mark.parametrize("edge", [640, 1280, 1500])
def test_render_long_edge_and_aspect(tmp_path, edge):
    pdf = make_pdf(tmp_path / "a.pdf", ["短い"])
    (page,) = extract_pages(pdf, tmp_path / "out", Settings(doc_page_image_long_edge=edge))
    with Image.open(page.image_path) as im:
        w, h = im.size
        assert im.format == "PNG" and im.mode == "RGB"
    assert h == edge                       # A4 portrait: height is the long edge
    assert abs(w / h - 210 / 297) < 0.005  # aspect ratio kept


def test_landscape_page_long_edge_is_width(tmp_path):
    pdf = tmp_path / "land.pdf"
    doc = pdfium.PdfDocument.new()
    doc.new_page(842, 595)
    buf = io.BytesIO()
    doc.save(buf)
    pdf.write_bytes(buf.getvalue())
    (page,) = extract_pages(pdf, tmp_path / "out", Settings())
    with Image.open(page.image_path) as im:
        assert im.size[0] == 1280 and abs(im.size[1] / 1280 - 595 / 842) < 0.005


def test_rendered_page_is_white_background_with_dark_text(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["黒い文字"])
    (page,) = extract_pages(pdf, tmp_path / "out", Settings())
    with Image.open(page.image_path) as im:
        px = im.getpixel((2, 2))
        lo = im.convert("L").getextrema()[0]
    assert px == (255, 255, 255) and lo < 100


def test_encrypted_pdf_is_reported_as_password_protected(tmp_path):
    src = make_pdf(tmp_path / "plain.pdf", [LONG])
    enc = encrypt_pdf(src, tmp_path / "enc.pdf")
    with pytest.raises(DocumentError, match="パスワード"):
        extract_pages(enc, tmp_path / "out", Settings())


def test_concurrent_extraction_is_serialised_and_correct(tmp_path, monkeypatch):
    import threading
    import time

    from pfb_app.preprocess import documents

    pdfs = [make_pdf(tmp_path / f"{i}.pdf", [LONG + f"文書{i}", "短い"]) for i in range(4)]
    serial = {i: extract_pages(p, tmp_path / f"s{i}", Settings()) for i, p in enumerate(pdfs)}

    inside, overlaps = [0], []
    guard = threading.Lock()
    orig = documents._page_text

    def spy(page):
        with guard:
            inside[0] += 1
            if inside[0] > 1:
                overlaps.append(1)
        time.sleep(0.01)
        try:
            return orig(page)
        finally:
            with guard:
                inside[0] -= 1

    monkeypatch.setattr(documents, "_page_text", spy)
    results, errors = {}, []

    def run(k):
        try:
            i = k % 4
            results[k] = (i, extract_pages(pdfs[i], tmp_path / f"c{k}", Settings()))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run, args=(k,)) for k in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and not overlaps
    for i, pages in results.values():
        assert [p.text for p in pages] == [p.text for p in serial[i]]
        assert [p.image_path is not None for p in pages] == [p.image_path is not None for p in serial[i]]
