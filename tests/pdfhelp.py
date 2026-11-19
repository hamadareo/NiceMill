"""Synthetic-PDF helpers for tests (WeasyPrint writes, pypdfium2/pypdf read). No pymupdf."""
import base64
import html
import io
from pathlib import Path

import pypdfium2 as pdfium
from weasyprint import HTML

_PAGE_CSS = "@page{size:210mm 297mm;margin:20mm}body{font-family:'Noto Sans CJK JP','Noto Sans JP',sans-serif;font-size:12pt}"


def _png_data_uri(w=40, h=40, gray=200) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (gray, gray, gray)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_pdf_bytes(texts, with_image: bool = False) -> bytes:
    """One page per entry of texts (an empty entry -> an empty page)."""
    if isinstance(texts, str):
        texts = [texts]
    pages = []
    for i, t in enumerate(texts):
        body = "<br>".join(html.escape(line) for line in t.split("\n")) if t else "&nbsp;"
        img = f'<p><img src="{_png_data_uri()}" width="100" height="100"></p>' if with_image and i == 0 else ""
        pages.append(f'<div style="page-break-after:always">{body}{img}</div>')
    doc = f"<html><head><style>{_PAGE_CSS}</style></head><body>{''.join(pages)}</body></html>"
    return HTML(string=doc).write_pdf()


def make_pdf(path: Path, texts, with_image: bool = False) -> Path:
    Path(path).write_bytes(make_pdf_bytes(texts, with_image))
    return Path(path)


def encrypt_pdf(src: Path, dest: Path, user_pw="secret", owner_pw="owner") -> Path:
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(clone_from=PdfReader(str(src)))
    w.encrypt(user_pw, owner_pw, algorithm="AES-256")
    with open(dest, "wb") as f:
        w.write(f)
    return Path(dest)


def pdf_text(data: bytes) -> str:
    doc = pdfium.PdfDocument(data)
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            out.append(tp.get_text_bounded().replace("\r\n", "\n"))
            tp.close()
            page.close()
        return "".join(out)
    finally:
        doc.close()


def pdf_page_count(data: bytes) -> int:
    doc = pdfium.PdfDocument(data)
    try:
        return len(doc)
    finally:
        doc.close()


def pdf_font_names(data: bytes) -> list[str]:
    from pypdf import PdfReader
    names = []
    for page in PdfReader(io.BytesIO(data)).pages:
        fonts = (page.get("/Resources") or {}).get("/Font") or {}
        for ref in fonts.values():
            names.append(str(ref.get_object().get("/BaseFont", "")))
    return names
