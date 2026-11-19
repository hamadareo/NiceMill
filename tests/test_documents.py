import shutil
from pathlib import Path

import pytest
from PIL import Image
from pdfhelp import encrypt_pdf, make_pdf

from pfb_app.config import Settings
from pfb_app.preprocess.documents import (
    DocumentError,
    Page,
    extract_pages,
    guess_student_name,
    to_pdf,
)


_make_pdf = make_pdf


LONG = "これは十分に長い日本語の文章です。\n" * 4


def test_extract_text_pdf(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", [LONG + "一ページ目", LONG + "二ページ目"])
    pages = extract_pages(pdf, tmp_path / "out", Settings())
    assert len(pages) == 2
    assert [p.number for p in pages] == [1, 2]
    assert "一ページ目" in pages[0].text
    assert "二ページ目" in pages[1].text
    assert all(p.image_path is None for p in pages)


def test_low_text_page_gets_image(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", [LONG, "短い"])
    pages = extract_pages(pdf, tmp_path / "out", Settings())
    assert pages[0].image_path is None
    assert pages[1].image_path is not None
    assert pages[1].image_path.is_file()
    with Image.open(pages[1].image_path) as im:
        assert max(im.size) == 1280
        assert im.mode == "RGB"


def test_page_with_image_gets_image(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", [LONG, LONG], with_image=True)
    pages = extract_pages(pdf, tmp_path / "out", Settings())
    assert pages[0].image_path is not None
    assert pages[1].image_path is None


def test_page_image_cap(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", ["短い"] * 5)
    s = Settings(doc_max_page_images=3)
    pages = extract_pages(pdf, tmp_path / "out", s)
    assert len(pages) == 5
    assert [p.image_path is not None for p in pages] == [True, True, True, False, False]


def test_page_dict_roundtrip(tmp_path):
    p = Page(number=1, text="x", image_path=tmp_path / "p.png")
    d = p.to_dict()
    assert d["image_path"] == str(tmp_path / "p.png")
    assert Page.from_dict(d) == p
    q = Page(number=2, text="y", image_path=None)
    assert q.to_dict()["image_path"] is None
    assert Page.from_dict(q.to_dict()) == q


def test_encrypted_pdf_raises(tmp_path):
    src = _make_pdf(tmp_path / "plain.pdf", [LONG])
    enc = encrypt_pdf(src, tmp_path / "enc.pdf")
    with pytest.raises(DocumentError, match="パスワード"):
        extract_pages(enc, tmp_path / "out", Settings())


def test_corrupt_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(bytes(range(256)) * 20)
    with pytest.raises(DocumentError, match="読み込めません"):
        extract_pages(bad, tmp_path / "out", Settings())


def test_zero_page_pdf_raises(tmp_path):
    z = tmp_path / "zero.pdf"
    z.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                  b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
                  b"trailer<</Root 1 0 R/Size 3>>\n%%EOF\n")
    with pytest.raises(DocumentError):
        extract_pages(z, tmp_path / "out", Settings())


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(DocumentError):
        extract_pages(tmp_path / "nope.pdf", tmp_path / "out", Settings())


def test_pdf_passthrough(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", [LONG])
    assert to_pdf(pdf, tmp_path / "out") == pdf


@pytest.mark.skipif(shutil.which("soffice") is None, reason="soffice not installed")
def test_docx_to_pdf(tmp_path):
    import docx

    src = tmp_path / "レポート.docx"
    d = docx.Document()
    d.add_paragraph("日本語の本文です。")
    d.save(src)
    before = src.read_bytes()
    out_dir = tmp_path / "out"
    pdf = to_pdf(src, out_dir)
    assert pdf.suffix == ".pdf" and pdf.parent == out_dir and pdf.is_file()
    assert src.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "レポート.docx"]
    pages = extract_pages(pdf, tmp_path / "img", Settings())
    assert "日本語の本文" in pages[0].text


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(DocumentError, match="対応していない"):
        to_pdf(f, tmp_path / "out")


def test_guess_student_name_with_roster():
    assert guess_student_name("課題3_山田太郎.pdf", ["山田太郎", "山田"]) == "山田太郎"


def test_guess_student_name_roster_no_match_falls_back():
    assert guess_student_name("課題3_佐藤花子.pdf", ["山田太郎"]) == "佐藤花子"


def test_guess_student_name_keeps_japanese():
    assert guess_student_name("21A1234_山田太郎_レポート.docx", None) == "山田太郎"
    assert guess_student_name("山田太郎.pdf", None) == "山田太郎"
