"""確認待ちの人の一覧と、読んで確認した人のまとめて承認 (across works).

`waiting` lists, work by work, the people whose feedback text waits for a person's look (the
status 「確認待ち」: a first or newer text exists and is not approved) and says for each whether
it can go with a bulk approval. `approve` approves what a person picked, through the same
service as step 5's 「全員まとめて承認」 (`review.bulk_approve`): its audit entries
(「approve」 with bulk, and one 「bulk_approve」 for the press), its run-id check (a text that
changed after the screen showed it is left for a person) and its gates apply unchanged.

A person is offered for a bulk approval only when nothing needs a second thought, the same
bar as step 5's: the text is generated and current, has no text warnings (colour words,
unsupported times and the like), is not a short ready-made text, the identity and language
gates hold, and nothing that feeds it is running. Everyone else is listed with the plain
reason (`Person.why`) and has to be opened and approved one by one.

Nothing here decides who may approve: the screen shows the controls only to a person who can
edit (ui.access) and re-checks in the press's callback; the repository handed to `approve`
carries the signed-in person for the audit.

A 「あとから訂正」 work has nothing to approve and is never in `waiting`. What waits for a look
there is a text someone reported wrong: `reported` lists those people (誤りの報告あり・確認中),
work by work, with the reason; each is opened and handled in the work (step 5), never in bulk.
"""
from dataclasses import dataclass

from pfb_app.domain.models import Kind, SubjectStatus
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import deletion, review
from pfb_app.storage.db import Repo

UNDO = ("まちがえて承認したときは、作業の「確認して承認」の画面でその人を開き、「直す」で書き直して"
        "承認し直せます。管理者は、管理用の画面の「確認・承認」で「承認を取り消す」もできます。"
        "PDFは、承認を変えたあとにもう一度受け取ってください。")


@dataclass(frozen=True)
class Person:
    subject_id: int
    name: str
    group: str                 # the group's label (a work may have several groups)
    run_id: int | None         # the feedback text shown: what an approval would approve
    text: str
    why: str | None            # None: can go with a bulk approval; else the plain reason

    @property
    def ready(self) -> bool:
        return self.why is None


@dataclass(frozen=True)
class WorkWaiting:
    assignment_id: int
    title: str
    people: tuple[Person, ...]

    @property
    def ready(self) -> list[Person]:
        return [p for p in self.people if p.ready]


def waiting(repo: Repo, assignment_ids) -> list[WorkWaiting]:
    """The people waiting for a look in these works (works waiting to be deleted are left
    out), work by work in the order given; a work with nobody waiting is left out."""
    gone = deletion.pending_deletions(repo)
    out: list[WorkWaiting] = []
    for aid in assignment_ids:
        if aid in gone:
            continue
        try:
            work = repo.get_assignment(aid)
        except KeyError:
            continue
        video = Kind(work["kind"]) == Kind.video
        people: list[Person] = []
        for sub in repo.list_submissions(aid):
            for subject in repo.list_subjects(sub["id"]):
                if video and not is_evaluated(subject):
                    continue
                if review.subject_status(repo, subject["id"]) != SubjectStatus.review:
                    continue
                run = review._latest_pfb(repo, subject)   # noqa: SLF001 (the same bar as bulk_approve)
                why = (review._being_written(repo, subject)   # noqa: SLF001
                       or review._skip_reason(repo, subject, run))   # noqa: SLF001
                people.append(Person(subject["id"], subject["display_name"], sub["label"],
                                     run.id if run else None,
                                     ((run.output or {}).get("pfb") or "") if run else "", why))
        if people:
            out.append(WorkWaiting(aid, work["title"], tuple(people)))
    return out


def approve(repo: Repo, picked: dict[int, int | None]) -> review.BulkResult:
    """Approve the people the screen offered and the person picked ({subject id: the run id
    shown}), by `review.bulk_approve`: a text that is no longer that run, or that no longer
    passes the bar, is skipped with its plain reason."""
    return review.bulk_approve(repo, dict(picked))


@dataclass(frozen=True)
class Reported:
    subject_id: int
    name: str
    group: str
    reason: str
    note: str | None
    text: str                  # the text the report is about


@dataclass(frozen=True)
class WorkReported:
    assignment_id: int
    title: str
    people: tuple[Reported, ...]


def reported(repo: Repo, assignment_ids) -> list[WorkReported]:
    """The people of these 「あとから訂正」 works whose newest correction row is a report (one
    query), work by work in the order given; works waiting to be deleted are left out."""
    ids = [int(a) for a in assignment_ids]
    if not ids:
        return []
    gone = deletion.pending_deletions(repo)
    marks = ",".join("?" * len(ids))
    rows = repo.conn.execute(
        "SELECT b.assignment_id, s.id, s.display_name, b.label, v.reason, v.note, v.text,"
        " a.title FROM text_versions v JOIN subjects s ON v.subject_id = s.id"
        " JOIN submissions b ON s.submission_id = b.id JOIN assignments a ON b.assignment_id = a.id"
        " WHERE v.kind = 'report' AND a.review_mode = 'correct_later'"
        " AND v.id = (SELECT MAX(w.id) FROM text_versions w WHERE w.subject_id = s.id)"
        f" AND b.assignment_id IN ({marks}) ORDER BY s.id", ids).fetchall()
    by_work: dict[int, list[Reported]] = {}
    titles: dict[int, str] = {}
    for aid, sid, name, group, reason, note, text, title in rows:
        if aid in gone:
            continue
        titles[aid] = title
        by_work.setdefault(aid, []).append(Reported(sid, name, group, reason or "", note,
                                                    text or ""))
    return [WorkReported(a, titles[a], tuple(by_work[a])) for a in ids if a in by_work]
