"""The evaluator PDF, the participant handouts and the per-person share pack
(Jinja2 + WeasyPrint, Noto CJK; charts are inline SVG from export.charts).

A subject is exported only when `review.subject_status` says approved; the raw approval_status
column is never read because a stale approval stays "approved" there. The PFB text comes from the
run whose id is `approved_run_id`.

- render_student_pdf: one handout per approved person (each starts on a new page and is
  numbered on its own: "1 / 2"), made person by person and put together;
- render_student_zip: per person `<番号>_<名前>.pdf` and `.html` (a self-contained page with
  its pictures inside, for a phone), in name order;
- render_teacher_pdf: the cover (facts and summary charts), the contact sheet of everyone
  evaluated, then each approved person with the evidence and where the text came from.

The handouts show the version of each person's text that the coach picked for the work
(`services.plain_text.handout_texts`: a text reworded for a younger reader, only while it is
checked and made from the approved text; otherwise the approved text); the evaluator PDF always
shows the approved text and says, under each person, which version their handout carries.

Images reach WeasyPrint only through `_Assets` ("asset:<id>" URLs mapped to files we chose):
the HTML carries no file paths, and nothing outside this machine is ever fetched.
"""
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from weasyprint.urls import URLFetcher, URLFetcherResponse

from pfb_app.domain.models import Kind, Stage, SubjectStatus
from pfb_app.export import labels, pictures, views
from pfb_app.export.sheets import Report, build_report
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import plain_text
from pfb_app.services.review import MANUAL_WARNING, SHORT_MARK, subject_status
from pfb_app.storage.db import Repo

__all__ = ["ApprovedEntry", "ExportError", "collect_approved", "handout_parts", "name_order",
           "render_student_pdf", "render_student_zip", "render_teacher_pdf", "share_page"]

_TASK_EXCERPT_CHARS = 200
# the app's name and its black-and-white mark, small at the start of every page's footer
BRAND_NAME = "ナイスミル"
BRAND_MARK = Path(__file__).resolve().parents[1] / "ui" / "static" / "brand" / "nicemiru_mark_mono.svg"
_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   autoescape=select_autoescape(["html", "xml", "j2"], default=True),
                   trim_blocks=True, lstrip_blocks=True)


class ExportError(Exception):
    pass


@dataclass
class ApprovedEntry:
    display_name: str
    pfb_text: str
    judgments_true: list[dict]
    judgments_false_count: int
    evidence: list[dict]
    manual: bool
    short: bool = False     # a short ready-made text approved on step 5 (services.short_text)


def _evidence_for(upstream_run, ref_id: str) -> dict | None:
    """Teacher-facing evidence for one reference id, or None if it cannot be found."""
    out = upstream_run.output or {}
    for e in out.get("events", []):
        if e["event_id"] == ref_id:
            return {"ref_id": ref_id, "participant": e["participant"],
                    "time": e["video_time"], "elapsed": e.get("task_elapsed") or e["video_time"],
                    "camera": e["camera"],
                    "text": e["observed_behavior"]}
    for i in out.get("items", []):
        if i["item_id"] == ref_id:
            return {"ref_id": ref_id, "page": i["page"],
                    "text": i["quote"] or i["observed_content"]}
    return None


def _entry(repo: Repo, subject: dict) -> ApprovedEntry:
    run = repo.get_run(subject["approved_run_id"])
    pid = subject["participant_id"]
    true_js, false_count, evidence = [], 0, []
    for con_id in run.upstream_run_ids:
        con = repo.get_run(con_id)
        if con is None or con.stage != Stage.contribution or not con.output:
            continue
        upstream = repo.get_run(con.upstream_run_ids[0]) if con.upstream_run_ids else None
        for j in con.output["judgments"]:
            # A video contribution run covers every participant: keep only this subject's.
            found = _evidence_for(upstream, j["ref_id"]) if upstream else None
            if found is None or found.get("participant", pid) != pid:
                continue
            if not j["is_positive_contribution"]:
                false_count += 1
                continue
            true_js.append(j)
            evidence.append(found)
    return ApprovedEntry(display_name=subject["display_name"], pfb_text=run.output["pfb"],
                         judgments_true=true_js, judgments_false_count=false_count,
                         evidence=evidence, manual=MANUAL_WARNING in run.warnings,
                         short=SHORT_MARK in run.warnings)


def name_order(name: str) -> tuple:
    """Sort key of a display name with its numbers compared as numbers: 「#7」 before
    「#10」, 「選手2」 before 「選手10」 (ties: the name itself, so 「#0」 before 「#00」)."""
    parts = re.split(r"(\d+)", name or "")
    return ([int(p) if i % 2 else p for i, p in enumerate(parts)], name or "")


def collect_approved(repo: Repo, assignment_id: int) -> list[ApprovedEntry]:
    entries = []
    video = repo.get_assignment(assignment_id)["kind"] == Kind.video.value
    for sub in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(sub["id"]):
            if video and not is_evaluated(subject):
                continue
            if subject_status(repo, subject["id"]) == SubjectStatus.approved:
                entries.append(_entry(repo, subject))
    return sorted(entries, key=lambda e: name_order(e.display_name))


class _Assets:
    """The images a document may load: "asset:<id>" -> a file we put there ourselves. The id
    comes from the file's path, never from its order: the handouts are separate documents
    joined into one PDF, which keeps one image per URL, so an order number gave every person
    the first person's pictures (production bug 2026-10-24)."""

    def __init__(self):
        self._paths: dict[str, Path] = {}

    def url(self, path) -> str:
        path = Path(path)
        key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
        self._paths.setdefault(key, path)
        return f"asset:{key}"

    def fetcher(self) -> URLFetcher:
        paths = self._paths

        class _Local(URLFetcher):
            def fetch(self, url, headers=None):
                if not url.startswith("asset:"):
                    raise ValueError("only the PDF's own images are loaded")
                path = paths[url.split(":", 1)[1]]
                mime = "image/svg+xml" if path.suffix.lower() == ".svg" else (
                    "image/png" if path.suffix.lower() == ".png" else "image/jpeg")
                return URLFetcherResponse(url, body=path.read_bytes(),
                                          headers={"Content-Type": mime})
        return _Local()


def _report(repo: Repo, assignment_id: int) -> Report:
    report = build_report(repo, assignment_id)
    if not report.approved:
        raise ExportError("承認済みのフィードバックがありません")
    return report


def _org(repo: Repo):
    from pfb_app.services import organisation
    return organisation.load(repo.settings)


def _common(report: Report, org, src) -> dict:
    return dict(title=report.title, date=report.date, words=report.words, org=org,
                accent=org.accent_hex, src=src, video=report.video, sports=report.sports,
                footer=labels.footer_line(report.work, report.video),
                brand_name=BRAND_NAME, brand_mark=BRAND_MARK)


def approved_subject_ids(repo: Repo, assignment_id: int) -> list[int]:
    """The subjects the handouts are for (as collect_approved picks them)."""
    video = repo.get_assignment(assignment_id)["kind"] == Kind.video.value
    return [subject["id"] for sub in repo.list_submissions(assignment_id)
            for subject in repo.list_subjects(sub["id"])
            if (not video or is_evaluated(subject))
            and subject_status(repo, subject["id"]) == SubjectStatus.approved]


def _growth_pages(repo: Repo, report: Report, org, on: bool) -> dict[int, dict]:
    """{subject id: its 「成長のまとめ」 page} when asked for (off by default)."""
    if not on:
        return {}
    from pfb_app.export import growth   # noqa: PLC0415
    records = growth.handout_records(repo, report.work["id"], [s.subject_id for s in report.approved])
    return {sid: views.growth_page(rec, accent=org.accent_hex) for sid, rec in records.items()}


def _handout_html(report: Report, sheets, org, src, *, screen: bool = False,
                  growth: dict[int, dict] | None = None,
                  plain: dict[int, str] | None = None) -> str:
    """`plain`: {subject id: the text their handout shows instead of the approved one}."""
    people = []
    for s in sheets:
        p = views.person(report, s, "handout", accent=org.accent_hex)
        if plain and s.subject_id in plain:
            p = {**p, "text": plain[s.subject_id]}
        people.append({**p, "growth": (growth or {}).get(s.subject_id)})
    title = sheets[0].name if len(sheets) == 1 else report.title
    return _env.get_template("share.html.j2" if screen else "student.html.j2").render(
        people=people, screen=screen, page_title=f"{title}（{report.title}）"
        if len(sheets) == 1 else report.title, **_common(report, org, src))


def _teacher_html(report: Report, org, src, notes: dict[int, dict] | None = None) -> str:
    task = (report.work.get("task_text") or "").strip()
    excerpt = task[:_TASK_EXCERPT_CHARS] + ("…" if len(task) > _TASK_EXCERPT_CHARS else "")
    groups = [(team, [views.contact(report, s) for s in people])
              for team, people in views.team_groups(report)]
    return _env.get_template("teacher.html.j2").render(
        screen=False, page_title=report.title, task_excerpt=excerpt,
        cameras=[labels.camera_word(c) for c in report.cameras], people_count=len(report.sheets),
        approved_count=len(report.approved), model=report.model,
        summary=views.summary(report, accent=org.accent_hex), groups=groups,
        courts=views.team_courts(report), short_note=labels.SHORT_NOTE,
        plain_label=f"{report.words.people}に渡すPDFの文章",
        team_tips=report.team_tips, team_tips_title=report.team_tips_title,
        people=[{**views.person(report, s, "evaluator", accent=org.accent_hex),
                 "plain": (notes or {}).get(s.subject_id)} for s in report.approved],
        **{**_common(report, org, src),   # never a person's check on an auto-approved sheet
           "footer": labels.evaluator_footer_line(report.work, report.video,
                                                  [s.auto for s in report.approved])})


def _html(template: str, repo: Repo, assignment_id: int) -> str:
    """The HTML of a PDF (tests read it): the handouts one after the other, or the
    evaluator PDF."""
    report, org = _report(repo, assignment_id), _org(repo)
    assets = _Assets()
    if template == "teacher.html.j2":
        return _teacher_html(report, org, assets.url, plain_text.handout_notes(repo, assignment_id))
    plain = plain_text.handout_texts(repo, assignment_id)
    return "\n".join(_handout_html(report, [s], org, assets.url, plain=plain)
                     for s in report.approved)


def _document(html: str, assets: _Assets):
    return HTML(string=html, url_fetcher=assets.fetcher()).render()


def _handout_documents(report: Report, org, growth: dict[int, dict] | None = None,
                       plain: dict[int, str] | None = None) -> list:
    docs = []
    for sheet in report.approved:
        assets = _Assets()
        docs.append(_document(_handout_html(report, [sheet], org, assets.url, growth=growth,
                                            plain=plain), assets))
    return docs


def render_teacher_pdf(repo: Repo, assignment_id: int) -> bytes:
    report, org = _report(repo, assignment_id), _org(repo)
    assets = _Assets()
    notes = plain_text.handout_notes(repo, assignment_id)
    return _document(_teacher_html(report, org, assets.url, notes), assets).write_pdf()


def render_student_pdf(repo: Repo, assignment_id: int, *, growth: bool = False) -> bytes:
    """`growth`: add the 「成長のまとめ」 page after the handout of each person whose record
    has two or more approved works up to this work (off by default)."""
    report, org = _report(repo, assignment_id), _org(repo)
    docs = _handout_documents(report, org, _growth_pages(repo, report, org, growth),
                              plain_text.handout_texts(repo, assignment_id))
    combined = docs[0].copy([page for d in docs for page in d.pages])
    combined.metadata.title = report.title          # not the first person's name
    return combined.write_pdf()


def share_page(report: Report, sheet, org, growth: dict[int, dict] | None = None,
               plain: dict[int, str] | None = None) -> str:
    """One person's handout as a self-contained HTML page (images inside as data: URIs)."""
    return _handout_html(report, [sheet], org, pictures.data_uri, screen=True, growth=growth,
                         plain=plain)


def handout_parts(repo: Repo, assignment_id: int) -> list[tuple[int, str, bytes, str]]:
    """[(subject id, name, the person's handout PDF, their phone page)] in the handout order
    (services.kept saves them before a work's videos go)."""
    report, org = _report(repo, assignment_id), _org(repo)
    plain = plain_text.handout_texts(repo, assignment_id)
    docs = _handout_documents(report, org, plain=plain)
    return [(sheet.subject_id, sheet.name, doc.write_pdf(),
             share_page(report, sheet, org, plain=plain))
            for sheet, doc in zip(report.approved, docs)]


def people_word(repo: Repo, assignment_id: int) -> str:
    """The work's word for its people (a handout file name's fallback)."""
    return labels.words_for(repo.get_assignment(assignment_id)).people


def render_student_zip(repo: Repo, assignment_id: int, *, growth: bool = False) -> bytes:
    """A ZIP of `<番号>_<名前>.pdf` and `<番号>_<名前>.html` per approved person, in the
    handout order (`growth`: as render_student_pdf)."""
    report, org = _report(repo, assignment_id), _org(repo)
    pages = _growth_pages(repo, report, org, growth)
    plain = plain_text.handout_texts(repo, assignment_id)
    docs = _handout_documents(report, org, pages, plain)
    buf = io.BytesIO()
    total = len(report.approved)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, (sheet, doc) in enumerate(zip(report.approved, docs), 1):
            name = labels.handout_file_name(k, sheet.name, total, fallback=report.words.people)
            z.writestr(name, doc.write_pdf())
            z.writestr(name[:-4] + ".html", share_page(report, sheet, org, pages, plain))
    return buf.getvalue()
