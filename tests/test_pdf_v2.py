"""Feedback PDF v2: the visual handout, the evaluator PDF and the per-person share pack, on
synthetic works only (tests/pdf_synthetic.py)."""
import io
import os
import re
import time
import zipfile
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pdfhelp import pdf_page_count, pdf_text

from pfb_app.export import pdf, views
from pfb_app.export.labels import SHORT_NOTE
from pfb_app.export.sheets import build_report
from pfb_app.services import organisation
from pdf_synthetic import document_work, video_group, work
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN, env  # noqa: F401 (env)


def text_of(data: bytes) -> str:
    return pdf_text(data)


def pages_of(data: bytes) -> int:
    return pdf_page_count(data)


def page_texts(data: bytes) -> list[str]:
    doc = pdfium.PdfDocument(data)
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            out.append(tp.get_text_bounded().replace("\r\n", ""))   # a long line wraps
            tp.close()
            page.close()
        return out
    finally:
        doc.close()


def sections(html: str) -> list[str]:
    return re.findall(r'<article class="handout">(.*?)</article>', html, re.S)


def clean(text: str, where: str) -> None:
    found = [w for w in (*SCHOOL_WORDS, "ルーブリック", *FORBIDDEN) if w in text]
    assert not found, f"{where}: {found}"


# --- sports ---------------------------------------------------------------------------------

def test_sports_handout_has_pictures_charts_court_and_timeline(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=3, seconds=40)
    html = pdf._html("student.html.j2", repo, aid)
    parts = sections(html)
    assert len(parts) == 3
    for part in parts:
        assert "よかった場面" in part
        assert len(re.findall(r'<figure class="moment', part)) == 4
        assert len(re.findall(r'src="asset:[0-9a-f]+"', part)) >= 4
        assert "どんな場面で活躍したか" in part and "時間の流れ" in part
        assert "コートのどこにいたか" in part and "動いた距離 約" in part
        assert "あなた" in part and "本人" not in part and "別の参加者" not in part
        assert "ほかの選手" in part or "ほかの" not in part
        assert "さんへ" in part and "試合・練習「練習試合」のふりかえり" in part
        # pictures are numbered 1..n and the timeline carries the same numbers
        assert re.findall(r'<span class="num">(\d)</span>', part) == ["1", "2", "3", "4"]
    data = pdf.render_student_pdf(repo, aid)
    t = text_of(data)
    assert "このフィードバックは動画をもとに作り、コーチが確かめたものです。" in t
    assert "1 / " in t
    assert pages_of(data) >= 3


def test_every_handout_starts_on_its_own_page_with_its_own_numbers(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30)
    data = pdf.render_student_pdf(repo, aid)
    pages = page_texts(data)
    starts = [i for i, t in enumerate(pages) if "さんへ" in t]
    assert len(starts) == 2 and starts[0] == 0
    assert "1 / " in pages[starts[1]]                  # numbered from 1 again


CHECKED = "このフィードバックは動画をもとに作り、コーチが確かめたものです。"
AUTO_SENT = ("このフィードバックは動画をもとに自動で作りました。機械チェックは通りましたが、"
             "コーチは確かめていません（機械チェック済み・人は未確認）。")


def test_an_auto_approved_handout_says_so_on_its_own_pages_only(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30)
    first, second = build_report(repo, aid).approved
    subject = repo.get_subject(second.subject_id)
    with repo._tx():
        repo.set_approval(subject["id"], subject["approved_run_id"], auto=True)
    report = build_report(repo, aid)
    assert [s.auto for s in report.approved] == [False, True]
    owner, footers = None, {first.name: set(), second.name: set()}
    for page in page_texts(pdf.render_student_pdf(repo, aid)):
        if "さんへ" in page:
            owner = first.name if f"{first.name}さんへ" in page.replace("\r\n", "") \
                else second.name
        footers[owner] |= {line for line in (CHECKED, AUTO_SENT) if line in page}
    assert footers == {first.name: {CHECKED}, second.name: {AUTO_SENT}}
    org = organisation.load(repo.settings)
    assert AUTO_SENT in pdf.share_page(report, report.approved[1], org)
    shared = pdf.share_page(report, report.approved[0], org)
    assert CHECKED in shared and AUTO_SENT not in shared


MIXED = ("このフィードバックは動画をもとに作りました。1人の分は機械チェック済み・人は未確認の"
         "自動承認で、ほかはコーチが確かめています。")


def test_the_evaluator_pdf_never_says_a_person_checked_an_auto_approved_sheet(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30)
    data = pdf.render_teacher_pdf(repo, aid)       # all checked by a person: as before
    assert {line for page in page_texts(data) for line in (CHECKED, AUTO_SENT, MIXED)
            if line in page} == {CHECKED}
    first, second = build_report(repo, aid).approved
    subject = repo.get_subject(second.subject_id)
    with repo._tx():
        repo.set_approval(subject["id"], subject["approved_run_id"], auto=True)
    pages = page_texts(pdf.render_teacher_pdf(repo, aid))
    assert MIXED in pages[0] and CHECKED not in pages[0]    # the cover: both kinds, said so
    assert AUTO_SENT in pages[-1] and CHECKED not in pages[-1]   # the auto sheet's own page
    assert any(CHECKED in p for p in pages[1:-1])           # the checked sheet's own page
    # every sheet auto: no page claims a person's check
    subject = repo.get_subject(first.subject_id)
    with repo._tx():
        repo.set_approval(subject["id"], subject["approved_run_id"], auto=True)
    pages = page_texts(pdf.render_teacher_pdf(repo, aid))
    assert all(CHECKED not in p and MIXED not in p for p in pages)
    assert all(AUTO_SENT in p for p in pages)


def test_no_court_without_a_homography(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30, geometry=False)
    html = pdf._html("student.html.j2", repo, aid) + pdf._html("teacher.html.j2", repo, aid)
    assert "コートのどこにいたか" not in html and "動いた距離" not in html
    assert "時間の流れ" in html


def test_no_court_for_a_sport_other_than_3x3_basketball(env):
    settings, repo = env
    aid = work(repo, sport="soccer")
    video_group(settings, repo, aid, people=2, seconds=30, geometry=True)
    assert "コートのどこにいたか" not in pdf._html("student.html.j2", repo, aid)


def test_game_clock_on_pictures_and_ticks(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=30, clock_offset_min=10)
    html = pdf._html("student.html.j2", repo, aid)
    assert re.search(r"試合時間 1\d:\d\d", html)
    assert ">10:00<" in html                            # the timeline's first tick
    assert "目もりは試合時間です。" in html


def test_empty_categories_are_omitted(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30, events_each=2)
    report = build_report(repo, aid)
    for sheet in report.approved:
        p = views.person(report, sheet, "handout")
        present = {c.label for c, n in sheet.counts(report.categories)}
        assert len(present) == 2 and p["bars"] is not None
        for c in report.categories:
            assert (c.label in p["bars"]) == (c.label in present)


def test_a_single_category_is_one_line_not_a_one_bar_chart(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=30, events_each=1)
    report = build_report(repo, aid)
    p = views.person(report, report.approved[0], "handout")
    assert p["bars"] is None and p["single"]["n"] == 1


def test_short_text_person_gets_text_photo_and_timeline_only(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=30, short=("Pb",))
    report = build_report(repo, aid)
    short = next(s for s in report.approved if s.short)
    html = pdf._html("student.html.j2", repo, aid)
    part = next(p for p in sections(html) if short.name in p)
    assert "よかった場面" not in part and "どんな場面で活躍したか" not in part
    assert "コートのどこにいたか" not in part and "時間の流れ" in part
    assert 'class="mark"' not in part                   # the on-screen band only, no claims
    assert SHORT_NOTE not in html
    teacher = pdf._html("teacher.html.j2", repo, aid)
    assert SHORT_NOTE in teacher and "短い文" in teacher


def test_evaluator_pdf_cover_contact_sheet_and_people(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=30, unapproved=("Pd",), short=("Pc",))
    html = pdf._html("teacher.html.j2", repo, aid)
    for words in ("日付", "カメラ", "1台（カメラ1）", "選手", "4人", "承認したフィードバック",
                  "3人", "解析の設定", "標準", "よかった場面の数", "映っていた時間",
                  "選手の一覧", "根拠の一覧", "もとにした解析", "入っていない（確認待ち）"):
        assert words in html, words
    report = build_report(repo, aid)
    left_out = next(s for s in report.sheets if not s.approved)
    assert left_out.name in html
    # the evaluator's person pages are only the approved people
    assert html.count('<section class="person-page">') == 3
    data = pdf.render_teacher_pdf(repo, aid)
    assert pages_of(data) >= 4
    assert "このフィードバックは動画をもとに作り、コーチが確かめたものです。" in text_of(data)


def test_teams_split_the_contact_sheet_and_the_courts(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=30, teams=True)
    html = pdf._html("teacher.html.j2", repo, aid)
    assert '<div class="teams">' in html
    assert "チーム1" in html and "チーム2" in html and "チームごとのいた所" in html


# --- general and document works -------------------------------------------------------------

def test_general_work_uses_general_words_and_categories(env):
    settings, repo = env
    aid = work(repo, domain="school", title="ペーパータワー")
    video_group(settings, repo, aid, people=2, seconds=30, geometry=True)
    html = pdf._html("student.html.j2", repo, aid)
    assert "活動「ペーパータワー」のふりかえり" in html
    assert "コートのどこにいたか" not in html          # never a court outside sports
    assert "ほかの参加者" in html or "ほかの" not in html
    from pfb_app.export.labels import categories_for
    assert any(c.label in html for c in categories_for(repo.get_assignment(aid)))
    assert "パス" not in html and "シュート" not in html
    t = text_of(pdf.render_student_pdf(repo, aid))
    assert "このフィードバックは動画をもとに作り、評価者が確かめたものです。" in t


def test_document_work_pdfs_render(env):
    _, repo = env
    aid = document_work(repo)
    html = pdf._html("student.html.j2", repo, aid)
    assert "よかった所" in html and "3ページ" in html and "避難経路を家族で確認する" in html
    assert "時間の流れ" not in html and "どんな場面で活躍したか" not in html
    t = text_of(pdf.render_student_pdf(repo, aid))
    assert "このフィードバックは書いたものをもとに作り、評価者が確かめたものです。" in t
    teacher = text_of(pdf.render_teacher_pdf(repo, aid))
    assert "避難所の地図を手で描いた" in teacher and "調べた内容を自分の言葉にした" in teacher


# --- vocabulary ------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["sports", "general", "document"])
def test_vocabulary_scans_on_the_rendered_html_and_pdf_text(env, kind):
    settings, repo = env
    if kind == "document":
        aid = document_work(repo)
    else:
        aid = work(repo, domain="sports" if kind == "sports" else "school", title="塔づくり")
        video_group(settings, repo, aid, people=3, seconds=30, short=("Pc",),
                    teams=kind == "sports")
    for template in ("student.html.j2", "teacher.html.j2"):
        clean(pdf._html(template, repo, aid), f"{kind} {template}")
    clean(text_of(pdf.render_student_pdf(repo, aid)), f"{kind} handout pdf")
    clean(text_of(pdf.render_teacher_pdf(repo, aid)), f"{kind} evaluator pdf")
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        for name in z.namelist():
            clean(name, f"{kind} zip name")
            if name.endswith(".html"):
                clean(z.read(name).decode("utf-8"), f"{kind} {name}")


# --- the per-person pack ----------------------------------------------------------------------

def test_zip_holds_a_pdf_and_a_self_contained_page_per_person(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=3, seconds=30)
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        names = z.namelist()
        assert names == ["1_#7.pdf", "1_#7.html", "2_山田花子.pdf", "2_山田花子.html",
                         "3_鈴木一郎.pdf", "3_鈴木一郎.html"]
        pdfs = [n for n in names if n.endswith(".pdf")]
        assert len(pdfs) == 3 and len(names) == 6
        for n in pdfs:
            assert n.split("_", 1)[1][:-4] in text_of(z.read(n))
        page = z.read(pdfs[0][:-4] + ".html").decode("utf-8")
    assert 'name="viewport"' in page
    assert "data:image/jpeg;base64," in page and "asset:" not in page
    assert not re.search(r'(src|href)="(?!data:)', page)        # nothing to fetch
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", page)


def test_zip_names_follow_the_handout_order(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=12, seconds=30, events_each=1)
    report = build_report(repo, aid)
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        pdfs = [n for n in z.namelist() if n.endswith(".pdf")]
    assert pdfs == sorted(pdfs)
    assert [n.split("_", 1)[1][:-4] for n in pdfs] == [s.name for s in report.approved]
    assert pdfs[0].startswith("01_")


# --- organisation ----------------------------------------------------------------------------

def test_organisation_name_logo_and_accent_in_both_pdfs(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=30)
    logo = (b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
            b'<rect width="40" height="20" fill="#0d6670"/></svg>')
    organisation.save(settings, name="みなと町クラブ", accent="teal", logo=logo,
                      logo_name="l.svg")
    for template in ("student.html.j2", "teacher.html.j2"):
        html = pdf._html(template, repo, aid)
        assert "みなと町クラブ" in html and 'class="logo" src="asset:' in html
        assert "#0d6670" in html
    assert "みなと町クラブ" in text_of(pdf.render_student_pdf(repo, aid))
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        page = z.read([n for n in z.namelist() if n.endswith(".html")][0]).decode()
    assert "data:image/svg+xml;base64," in page


# --- time ------------------------------------------------------------------------------------

def test_thirteen_people_within_the_time_budget(env, capsys):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=13, seconds=120)
    times = {}
    for name, fn in (("handout", pdf.render_student_pdf), ("evaluator", pdf.render_teacher_pdf),
                     ("zip", pdf.render_student_zip), ("handout again", pdf.render_student_pdf)):
        start = time.monotonic()
        data = fn(repo, aid)
        times[name] = time.monotonic() - start
        assert data
    with capsys.disabled():
        print("\nPDF timing, 13 people:", {k: round(v, 1) for k, v in times.items()})
    assert max(times.values()) <= 60
    assert times["handout again"] <= times["handout"] + 1   # pictures come from the cache


# --- samples (synthetic only), written when PFB_PDF_SAMPLES is set ---------------------------

@pytest.mark.skipif(not os.environ.get("PFB_PDF_SAMPLES"), reason="sample PDFs on request")
def test_write_samples(env):
    settings, repo = env
    out = Path(os.environ["PFB_PDF_SAMPLES"])
    out.mkdir(parents=True, exist_ok=True)
    organisation.save(settings, name="みなと町バスケットボールクラブ", accent="blue")
    sports = work(repo, title="練習試合 10月23日")
    video_group(settings, repo, sports, people=6, seconds=90, teams=True, short=("Pf",),
                unapproved=("Pe",), clock_offset_min=0)
    general = work(repo, domain="school", title="ペーパータワー")
    video_group(settings, repo, general, people=4, seconds=60, label="B")
    document = document_work(repo)
    for name, aid in (("sports", sports), ("general", general), ("document", document)):
        (out / f"{name}_handout.pdf").write_bytes(pdf.render_student_pdf(repo, aid))
        (out / f"{name}_evaluator.pdf").write_bytes(pdf.render_teacher_pdf(repo, aid))
        (out / f"{name}_per_person.zip").write_bytes(pdf.render_student_zip(repo, aid))


# --- review round 1 -------------------------------------------------------------------------

def _picture_times(part: str) -> list[str]:
    return re.findall(r'<span class="when">([^<]*)</span>', part)


def test_one_clock_per_handout_with_a_late_start(env):
    settings, repo = env
    aid = work(repo, domain="school", title="ペーパータワー")
    video_group(settings, repo, aid, people=2, seconds=50, start_sec=12)
    html = pdf._html("student.html.j2", repo, aid)
    for part in sections(html):
        times = _picture_times(part)
        assert times and all(t.startswith("開始から ") for t in times)
        assert "動画の時刻" not in part and "試合時間" not in part
        svg = part[part.index('aria-label="時間の流れ"'):]
        desc = re.search(r"<desc>(.*?)</desc>", svg, re.S).group(1)
        for t in times:                                  # the same clock as the ticks
            assert t.removeprefix("開始から ") in desc
        assert ">00:00<" in part and "目もりは開始からの時間です。" in part
    report = build_report(repo, aid)
    for sheet in report.approved:                        # 開始から = seconds since the start
        for m in sheet.pictures:
            assert m.t_s is not None and m.t_s < 50 - 12


def test_sports_without_a_game_clock_never_says_game_time(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=40, start_sec=5)
    html = pdf._html("student.html.j2", repo, aid) + pdf._html("teacher.html.j2", repo, aid)
    assert "試合時間" not in html and "開始から" in html


def test_distance_comes_from_one_camera_never_the_sum(env):
    settings, repo = env
    one = work(repo, title="一台")
    video_group(settings, repo, one, people=1, seconds=60, label="A")
    two = work(repo, title="二台")
    video_group(settings, repo, two, people=1, seconds=60, label="B", cameras=("cam1", "cam2"))
    d1 = build_report(repo, one).approved[0].distance_m
    d2 = build_report(repo, two).approved[0].distance_m
    assert d1 is not None and d2 == pytest.approx(d1)
    assert len(build_report(repo, two).approved[0].court_points) == len(
        build_report(repo, one).approved[0].court_points)


def test_the_camera_with_most_court_samples_is_used():
    from pfb_app.export.sheets import best_court
    short, long_ = [(1, 0.0, 0.0)] * 3, [(1, 0.0, 0.0)] * 9
    assert best_court([(10.0, short), (25.0, long_)]) == (25.0, long_)
    assert best_court([]) == (0.0, [])


def _retrack_checks(repo, aid):
    html = pdf._html("student.html.j2", repo, aid)
    assert "<figure" not in html and "時間の流れ" not in html and "コートのどこ" not in html
    assert "さんへ" in html
    teacher = pdf._html("teacher.html.j2", repo, aid)
    assert RETRACK_NOTE in teacher and "時間の流れ" not in teacher


RETRACK_NOTE = "人の追跡をやり直しているため、写真と図は入れていません。"


def test_a_running_track_job_leaves_out_pictures_spans_and_court(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    assert "<figure" in pdf._html("student.html.j2", repo, aid)
    from pfb_app.domain.models import Stage
    jid = repo.enqueue(sid, Stage.track, "cpu", camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    _retrack_checks(repo, aid)


def test_boxes_newer_than_the_track_run_leave_out_pictures_spans_and_court(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    from pfb_app.domain.models import Stage
    boxes = Path(repo.latest_run(sid, Stage.track, "cam1").output["boxes"])
    later = time.time() + 120
    os.utime(boxes, (later, later))
    _retrack_checks(repo, aid)


def test_a_chain_on_an_older_track_run_counts_as_retracking(env):
    """Final review MED 6: every track run writes the same boxes.txt. A text whose chain stands
    on track run T1 would look up T1's labels in T2's boxes (with TID numbering: another
    person's box), so a boxes.txt newer than the chain's own track run is a re-track too."""
    from datetime import datetime, timedelta, timezone
    from pdf_synthetic import save
    from pfb_app.domain.models import Stage
    from pfb_app.export import sheets
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    t1 = repo.latest_run(sid, Stage.track, "cam1")
    hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    repo.conn.execute("UPDATE stage_runs SET created_at=? WHERE id=?", (hour_ago, t1.id))
    t1 = repo.get_run(t1.id)
    os.utime(Path(t1.output["boxes"]), (time.time(), time.time()))   # T2 wrote it just now
    t2 = save(repo, sid, Stage.track, dict(t1.output), camera="cam1",
              upstream=t1.upstream_run_ids)                          # ... and was saved

    def sub():
        return sheets._Submission(repo, repo.get_assignment(aid), repo.get_submission(sid))
    assert sub().retracking() is False                      # the newest track run: its boxes
    assert sub().retracking([t2]) is False
    assert sub().retracking([t1]) is True                   # the chain's T1: not its boxes
    assert sub().retracking([None]) is False


def test_the_picture_cache_keeps_only_the_current_pictures(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=1, seconds=40)
    pdf.render_student_pdf(repo, aid)
    from pfb_app.storage import files
    cache = files.submission_dir(settings, aid, sid) / "handout" / "pictures"
    current = set(cache.glob("*.jpg"))
    assert len(current) == 4
    stray = cache / "0123456789abcdef0123.jpg"
    stray.write_bytes(b"old")
    pdf.render_student_pdf(repo, aid)
    assert not stray.exists() and set(cache.glob("*.jpg")) == current


def test_the_combined_handout_is_titled_with_the_work(env):
    from pypdf import PdfReader
    settings, repo = env
    aid = work(repo, title="練習試合 本番")
    video_group(settings, repo, aid, people=2, seconds=30)
    meta = PdfReader(io.BytesIO(pdf.render_student_pdf(repo, aid))).metadata
    assert meta.title == "練習試合 本番"


def visible(html: str) -> str:
    """What a reader sees of an HTML page: no style sheet, no tags or attributes."""
    html = re.sub(r"<style>.*?</style>", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", html)


def test_no_other_person_in_anyones_handout_zip_pdf_or_page(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=3, seconds=40)
    report = build_report(repo, aid)
    from pfb_app.export.views import caption
    own = {s.name: {caption(report, s, m.what, "handout") for m in s.moments if m.cited}
           for s in report.approved}
    sections_by_name = {}
    for part in sections(pdf._html("student.html.j2", repo, aid)):
        name = next(n for n in own if f">{n}<" in part)
        sections_by_name[name] = part
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        files_ = {n: z.read(n) for n in z.namelist()}
    for name, mine in own.items():
        others = [n for n in own if n != name]
        foreign = set().union(*(own[n] for n in others)) - mine
        texts = [visible(sections_by_name[name])]
        texts += [pdf_text(v) if k.endswith(".pdf") else visible(v.decode())
                  for k, v in files_.items() if k.split("_", 1)[1].rsplit(".", 1)[0] == name]
        assert len(texts) == 3
        for t in texts:
            assert name in t
            for other in others:
                assert other not in t, (name, other)
            for c in foreign:
                assert c not in t, (name, c)


def test_a_confirm_with_the_game_clock_left_blank_gives_times_from_the_start(env):
    """The card's 「この動画の始まりの試合時間」 is blank by default: confirm stores None and
    the PDFs say 開始から; only a value the coach entered (0 included) is a game clock."""
    from pfb_app.domain.models import Stage
    from pfb_app.services.content_check import CheckChoice, confirm_check
    from test_content_check import detect_done
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=1, seconds=40)
    sub = repo.get_submission(sid)
    repo.update_submission_meta(sid, {**sub["meta"], "check": "pending"})
    detect_done(repo, sid, sport="basketball")

    def confirm(**kw):
        repo.update_submission_meta(sid, {**repo.get_submission(sid)["meta"], "check": "pending"})
        confirm_check(repo, settings, sid, CheckChoice(
            title="練習試合", activity="3人ずつで試合をする。", rubric_text="", domain="sports",
            sport="basketball", method="pfb3x3", max_people=None, target="person",
            start_sec=0.0, end_sec=None, **kw))
        repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
        assert repo.latest_run(sid, Stage.track, "cam1") is not None
        return pdf._html("student.html.j2", repo, aid) + pdf._html("teacher.html.j2", repo, aid)

    html = confirm()                                      # the card's default: left blank
    assert repo.get_submission(sid)["meta"]["clock_offset_min"] is None
    assert "開始から" in html and "試合時間" not in html
    html = confirm(clock_offset_min=0)                    # a 0 the coach typed counts
    assert repo.get_submission(sid)["meta"]["clock_offset_min"] == 0.0
    assert "試合時間 00:" in html and "開始から" not in html


def _drawn_images(data: bytes) -> list[list[str]]:
    """Per page: the sha1 of every image the page draws (in drawing order)."""
    import hashlib
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    out = []
    for page in reader.pages:
        res = page.get("/Resources").get_object()
        xo = res.get("/XObject")
        names = {}
        for name, ref in (xo.get_object().items() if xo else []):
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                names[name] = hashlib.sha1(obj.get_data()).hexdigest()
        drawn = re.findall(r"/(\w+)\s+Do", page.get_contents().get_data().decode("latin1"))
        out.append([names["/" + d] for d in drawn if "/" + d in names])
    return out


def test_each_person_in_the_combined_handout_gets_their_own_pictures(env):
    """Regression test for a past bug: every person's handout in the one PDF showed the first
    person's pictures (the asset URLs were numbered per person, and the combined PDF kept
    one image per URL)."""
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=3, seconds=40)
    data = pdf.render_student_pdf(repo, aid)
    texts, images = page_texts(data), _drawn_images(data)
    starts = [i for i, t in enumerate(texts) if "さんへ" in t] + [len(texts)]
    per_person = [set().union(*images[a:b]) for a, b in zip(starts, starts[1:])]
    assert len(per_person) == 3 and all(per_person)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not (per_person[i] & per_person[j]), (i, j)


def test_no_picture_of_a_moment_when_the_box_was_on_another_player(env):
    """Regression test for a past bug: one person's 4th picture showed another player
    (the track had swapped). With the names stage's number timeline, a cited moment whose
    label read another number is shown without a picture (and no fallback frame)."""
    from pfb_app.domain.models import Stage
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    track = repo.latest_run(sid, Stage.track, "cam1")
    beh = repo.latest_run(sid, Stage.behavior, "cam1")
    first = [e for e in beh.output["events"] if e["participant"] == "Pa"]
    other = first[1]
    sec = int(other["video_time"][:2]) * 60 + int(other["video_time"][3:5])
    f = sec * 10 + 1
    save(repo, sid, Stage.names, {"track_run_id": track.id, "cues": {"number": {"P1": {
        "number": "5", "frames_agree": 3, "timeline_s": 2.0, "dominant": "5",
        "dominant_share": 0.8,
        "segments": [[1, f - 25, "5", 6], [f - 5, f + 5, "8", 2], [f + 25, 400, "5", 6]]}}}},
         camera="cam1", upstream=[track.id])
    report = build_report(repo, aid)
    sheet = next(s for s in report.sheets if s.name == "山田花子")
    by_ref = {m.ref_id: m for m in sheet.moments}
    assert by_ref[other["event_id"]].picture is None and by_ref[other["event_id"]].thumb is None
    assert all(by_ref[e["event_id"]].picture is not None for e in first if e is not other)
    assert [m.number for m in sheet.pictures] == [1, 2, 3]


def test_the_portrait_is_taken_where_the_person_wears_their_own_number(env):
    """2026-10-24 (work 6): 選手2's portrait came from the people sheet at 1 s, when the
    track was on another player (#5; the person is #1 for most of the game). With a number
    timeline, a portrait taken on another number is replaced by one from the person's own
    longest stretch."""
    import json as _json
    from pfb_app.domain.models import Stage
    from pfb_app.services import auto_people
    from pdf_synthetic import save
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=40)
    track = repo.latest_run(sid, Stage.track, "cam1")
    out = {**track.output, "sheet_frame": 20, "sheet_geometry": {"P1": [10, 10, 20, 30]}}
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (_json.dumps(out), track.id))
    pa = next(s for s in repo.list_subjects(sid) if s["participant_id"] == "Pa")
    plain = auto_people.person_crops(repo, settings, sid)[pa["id"]]
    save(repo, sid, Stage.names, {"track_run_id": track.id, "cues": {"number": {"P1": {
        "number": None, "frames_agree": 1, "timeline_s": 2.0, "dominant": "1",
        "dominant_share": 0.7, "segments": [[1, 60, "5", 3], [100, 380, "1", 12]]}}}},
         camera="cam1", upstream=[track.id])
    own = auto_people.person_crops(repo, settings, sid)[pa["id"]]
    assert own != plain and own.is_file()
    pb = next(s for s in repo.list_subjects(sid) if s["participant_id"] == "Pb")
    # no sheet geometry for the second person: a picture from their own longest stretch
    assert auto_people.person_crops(repo, settings, sid).get(pb["id"]).is_file()


def test_every_evaluated_person_gets_a_photo_from_their_longest_stretch(env):
    """2026-11-14 long-video trial: the people sheet (one frame) showed 7 of 15 people, the
    others had no photo. A person without a sheet box gets a picture cut from the middle of
    their longest visible stretch, so everybody has one."""
    from pfb_app.services import auto_people
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=40)
    crops = auto_people.person_crops(repo, settings, sid)     # the group has no sheet geometry
    people = [s for s in repo.list_subjects(sid) if s.get("participant_id")]
    assert len(people) == 3 and all(crops[s["id"]].is_file() for s in people)
    assert len({crops[s["id"]] for s in people}) == 3           # one picture each


def test_longest_stretch_splits_on_a_pause_and_picks_the_earlier_tie():
    from pfb_app.services.auto_people import _longest_stretch
    assert _longest_stretch([], 10) is None
    assert _longest_stretch([5], 10) == (5, 5)
    # 10 fps: pauses up to 10 frames keep a stretch together; 1..20, then a 30-frame gap, 51..60
    frames = list(range(1, 21)) + list(range(51, 61))
    assert _longest_stretch(frames, 10) == (1, 20)
    assert _longest_stretch([1, 2, 3, 40, 41, 42], 10) == (1, 3)      # equal: the earlier one
    assert _longest_stretch([1, 8, 15, 22, 100], 10) == (1, 22)       # 7-frame pauses stay
