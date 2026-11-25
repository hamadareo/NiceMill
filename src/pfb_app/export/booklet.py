"""The booklet (製本用) output: a cover, each person on a spread (left page = even), a back cover.

Page 1 is the cover (a right page); pages 2-3, 4-5, ... are spreads (the left page of each is
even, so a person's two pages face each other when bound); the last page is the back cover. The
total is a multiple of 4: with an even number of people a front spread (目次 / この冊子について)
is added, with an odd number the 奥付 carries 「この冊子について」 and the contents. Every page is
its own one-page document (fitted by shrinking its type in steps until it is one page), so the
gutter (inner margin 22 mm, outer 15 mm) and the page numbers are exact.
"""
from __future__ import annotations

import io
from pathlib import Path

from pfb_app.export import labels, pdf as _pdf, views
from pfb_app.export.sheets import build_report

AUTHORS = ("濱田 怜央", "酒造 正樹（湘南工科大学）")      # as given; no titles or contacts
REPO_URL = "https://github.com/hamadareo/NiceMill"
REPO_NOTE = None        # e.g. "公開準備中" next to the QR code while the repository is private
EDITION = "第1版"
SERIES = "NiceMill 処理出力シリーズ"
SCALES = tuple(round(3.0 - 0.03 * i, 2) for i in range(0, 80))     # 3.0 ... 0.63, large first


def _occupancy(doc) -> float:
    """The share of the body area the page's content (the .bk block) fills, from the layout."""
    page = doc.pages[0]._page_box
    top, height = page.content_box_y(), page.height
    bottom = top
    for box in page.descendants():
        el = getattr(box, "element", None)
        if el is not None and el.get("class") == "bk" and getattr(box, "height", None) is not None:
            bottom = max(bottom, box.content_box_y() + box.height)
    return max(0.0, min(1.0, (bottom - top) / height)) if height else 0.0


def _render(env_tmpl, assets_cls, variants=({},), **ctx):
    """One page: for each variant of the page (richer first) the largest scale that is still one
    page; the fullest result wins (a richer variant on a tie within 2 points)."""
    best = None
    for extra in variants:
        found = None
        cache = {}

        def attempt(sc):
            if sc not in cache:
                assets = assets_cls()
                doc = _pdf._document(env_tmpl.render(s=sc, src=assets.url, **{**ctx, **extra}), assets)
                cache[sc] = (_occupancy(doc), sc, doc) if len(doc.pages) == 1 else None
            return cache[sc]

        top = max(3.0, ctx.get("smax", 3.0))      # a page with little on it may grow further (smax)
        coarse = [round(top - 0.15 * i, 2) for i in range(0, int((top - 0.45) / 0.15) + 1)]       # top ... 0.6
        hit = next((c for c in coarse if attempt(c)), None)
        if hit is not None:      # then the finest step between it and the coarse step above
            for sc in (round(hit + 0.03 * k, 2) for k in (4, 3, 2, 1)):
                if attempt(sc):
                    hit = sc
                    break
            found = attempt(hit)
        if found is None:
            continue
        if best is None or found[0] > best[0] + 0.02:
            best = found
    if best is None:
        raise RuntimeError("a booklet page does not fit on one page")
    _render.log.append((ctx.get("kind"), ctx.get("n"), round(best[0], 2), best[1]))
    return best[2]


_render.log = []


def _today() -> str:
    from datetime import datetime   # noqa: PLC0415
    d = datetime.now()
    return f"{d.year}年{d.month}月{d.day}日"


def _qr_svg_path(url: str | None) -> str | None:
    """A black-on-white QR code of `url` as a small SVG file (None: no code)."""
    if not url:
        return None
    import tempfile   # noqa: PLC0415
    import segno      # noqa: PLC0415 - pip install segno
    path = Path(tempfile.mkdtemp(prefix="nicemill_qr_")) / "qr.svg"
    segno.make(url, error="m").save(str(path), scale=10, border=2, dark="#000000", light="#ffffff",
                                    xmldecl=False, svgns=True)
    return str(path)


def _facts(report, sheet, p) -> tuple[list[str], list[dict]]:
    """Facts already in the output's data: counts, on-screen time and spans, and each cited
    scene's time, what was done and why it was good (no ranking, nothing new)."""
    facts = []
    counts = sheet.counts(report.categories)
    total = sum(n for _, n in counts)
    if total:
        facts.append(f"よかった場面の数：{total}回")
        facts.append("場面の種類の内訳：" + "、".join(f"{c.label} {n}回" for c, n in counts))
    if sheet.on_screen_s:
        facts.append(f"映っていた時間：約{round(sheet.on_screen_s / 60, 1)}分")
    if sheet.spans:
        facts.append(f"映っていた区間：{len(sheet.spans)}か所")
    rows = []
    for it in list(p["pictures"]) + list(p["moments_text"]):
        m = it["m"]
        rows.append({"when": m.when, "what": it["caption"], "why": m.why or ""})
    return facts, rows


def plan(n_people: int) -> dict:
    """The page layout for n people: total pages (a multiple of 4), the front spread, the
    first page of person k (1-based)."""
    front = n_people % 2 == 0
    first = 4 if front else 2
    total = first + 2 * n_people          # the back cover is the last page
    assert total % 4 == 0, total
    return {"front": front, "first": first, "total": total}


def build(repo, assignment_id: int, *, credits=None, contact=None, subtitle=None,
          credit_short=None, title="NiceMill 処理出力", identity_gate: bool = False,
          authors=AUTHORS, repo_url=REPO_URL, repo_note=REPO_NOTE, edition=EDITION,
          series=SERIES) -> bytes:
    from pfb_app.export import identity_gate as gate_mod   # noqa: PLC0415
    from pfb_app.export import sheets as _sheets   # noqa: PLC0415
    keep, _sheets.MAX_PICTURES = _sheets.MAX_PICTURES, 8      # all of a person's cited scenes
    try:
        report = build_report(repo, assignment_id, drafts=True, identity_gate=identity_gate)
    finally:
        _sheets.MAX_PICTURES = keep
    org = _pdf._org(repo)
    people = report.exported
    left_out = [s for s in report.sheets if s not in people]
    unconfirmed = [s for s in left_out if s.ident_blocked]
    notext = [s for s in left_out if s not in unconfirmed]
    # a work where nothing tells people apart prints everyone, unchecked (identity_gate.NO_CLUE)
    unchecked = identity_gate and any(s.ident_how == gate_mod.NO_CLUE for s in people)
    lay = plan(len(people))
    env = _pdf._env.get_template("booklet.html.j2")
    stamp = _pdf._stamp()
    credit_lines = _pdf._credit_lines(credits)
    common = _pdf._common(report, org, None, credits, contact)
    common.pop("src", None)
    common.update(stamp=stamp, credits=credit_lines)
    qr = _qr_svg_path(repo_url)
    cover = {"title": title, "work": report.title, "subtitle": subtitle, "people": len(people),
             "credit_short": credit_short, "series": series, "authors": list(authors),
             "repo_url": repo_url, "repo_note": repo_note, "edition": edition, "qr": qr,
             "date": _today()}
    toc = [{"name": views.person(report, s, "handout")["name"], "id": s.jersey_id,
            "page": lay["first"] + 2 * k} for k, s in enumerate(people)]
    about = {
        "what": f"この冊子は、{report.title}の映像を NiceMill で処理した出力です。人ごとに見開き2ページで、"
                f"左に文と根拠の場面、右に場面の種類・時間の流れ・確認状況を載せています。",
        "ai": labels.AI_LINE + " 人の確認を経ていない場合があります。人の呼び方は背番号です。"
              "背番号は AI が読み取ったもので、「？（推定）」は確かめていない印です。",
        "fix": "誤りに気づいたら、評価する人に伝えてください。訂正は版として履歴に残り、確認状況"
               "（訂正なし／訂正済み 版N／誤りの報告あり）に出ます。",
        "left_out": " ".join(x for x in (
            ("背番号の読み取りも「人を確かめる」の記録もない作業のため、同じ人かどうかは確かめずに"
             "掲載しています。" if unchecked else
             "同一人物の確認を通った人だけを掲載しています。" if identity_gate else ""),
            (f"確認が取れていない{len(unconfirmed)}人は載せていません（理由は評価者用の一覧にあります）。"
             if unconfirmed else ""),
            ("同一人物と確かめられた人がいなかったため、人物ページはありません。"
             if unconfirmed and not people else ""),
            (f"場面が少なく文のない{len(notext)}人は、載せていません。" if notext else "")) if x),
        # one small, removable line (templates/booklet.html.j2: id="tracking-note")
        "tracking_note": ("" if identity_gate and not unchecked
                          else "人物の追跡に取り違えが含まれることがあります。"),
    }
    pages = []
    n = 1
    ctx = dict(common, cols=2, page_title=report.title, kind="cover", n=n, cover=cover, title=report.title, date=report.date,
               words=report.words, toc=toc, toc_back=None, about=about, p=None, contact=common["contact"],
               ai_line=labels.AI_LINE)
    pages.append(_render(env, _pdf._Assets, **ctx))
    n += 1
    if lay["front"]:
        pages.append(_render(env, _pdf._Assets, **dict(ctx, kind="front_l", n=n, toc=toc)))
        n += 1
        pages.append(_render(env, _pdf._Assets, **dict(ctx, kind="front_r", n=n)))
        n += 1
    for s in people:
        p = views.person(report, s, "handout", accent=org.accent_hex, width=520)
        all_rows = [{"when": m.when, "what": views.caption(report, s, m.what, "handout"),
                     "why": m.why or ""}
                    for m in sorted(s.moments, key=lambda m: (m.t_s or 0.0, m.ref_id))]
        left = dict(ctx, kind="left", p=p, credits=[], all_rows=all_rows, cols=1 if len(p["pictures"]) <= 1 else 2 if len(p["pictures"]) <= 4 else 3)
        facts, rows = _facts(report, s, p)
        right = dict(ctx, kind="right", p=p, credits=credit_lines[:1], facts=facts, rows=rows)
        assert n % 2 == 0, n                          # a person's left page is always even
        pages.append(_render(env, _pdf._Assets, **dict(left, n=n)))
        n += 1
        pages.append(_render(env, _pdf._Assets, variants=({"extras": True}, {"extras": False}),
                             **dict(right, n=n)))
        n += 1
    back = dict(ctx, kind="back", n=n, toc_back=None if lay["front"] else toc)
    pages.append(_render(env, _pdf._Assets, **back))
    assert n == lay["total"] and len(pages) == lay["total"], (n, len(pages), lay)
    combined = pages[0].copy([pg for d in pages for pg in d.pages])
    combined.metadata.title = f"{title} {report.title}"
    combined.metadata.generator = _pdf.BRAND_NAME
    return combined.write_pdf()


def impose_a3(pdf_bytes: bytes) -> bytes:
    """Saddle-stitch printing order on A3 landscape sheets, two A4 pages per side:
    outer sheet front = last | 1, back = 2 | last-1, and so on."""
    from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    r = PdfReader(io.BytesIO(pdf_bytes))
    N = len(r.pages)
    assert N % 4 == 0
    w, h = float(r.pages[0].mediabox.width), float(r.pages[0].mediabox.height)
    out = PdfWriter()
    for k in range(N // 4):
        for left, right in ((N - 2 * k, 1 + 2 * k), (2 + 2 * k, N - 1 - 2 * k)):
            sheet = PageObject.create_blank_page(width=2 * w, height=h)
            sheet.merge_transformed_page(r.pages[left - 1], Transformation())
            sheet.merge_transformed_page(r.pages[right - 1], Transformation().translate(w, 0))
            out.add_page(sheet)
    buf = io.BytesIO()
    out.write(buf)
    return buf.getvalue()
