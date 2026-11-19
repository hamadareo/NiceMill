"""The saved handouts of a work whose videos are to go (services.kept): one PDF and one phone
page per approved person and the evaluator PDF, kept in data/assignments/<aid>/kept/.
Synthetic works only (generated videos, invented names)."""
import io
import json
import zipfile
from datetime import datetime, timedelta

from pdfhelp import pdf_page_count, pdf_text

from pfb_app.services import kept
from pfb_app.storage import files
from pdf_synthetic import video_group, work
from test_wizard_ui_basic import env  # noqa: F401 (env)


def _work(settings, repo, people=3):
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=people, seconds=30)
    return aid, sid


def test_save_and_serve_the_handouts(env):
    settings, repo = env
    aid, sid = _work(settings, repo)
    saved = kept.save(repo, aid)
    assert [p["name"] for p in saved.people] == ["#7", "山田花子", "鈴木一郎"]
    folder = files.assignment_dir(settings, aid, create=False) / "kept"
    manifest = json.loads((folder / "kept.json").read_text(encoding="utf-8"))
    assert manifest["saved_at"] == saved.saved_at and len(manifest["people"]) == 3
    assert kept.current(repo, aid) == saved
    combined = kept.student_pdf(settings, aid, saved)
    assert pdf_page_count(combined) >= 3
    assert "山田花子" in pdf_text(combined) and "鈴木一郎" in pdf_text(combined)
    with zipfile.ZipFile(io.BytesIO(kept.student_zip(settings, aid, saved))) as z:
        names = sorted(z.namelist())
    assert names == ["1_#7.html", "1_#7.pdf", "2_山田花子.html", "2_山田花子.pdf",
                     "3_鈴木一郎.html", "3_鈴木一郎.pdf"]
    assert "山田花子" in pdf_text(kept.evaluator_pdf(settings, aid, saved))


def test_saving_again_replaces_the_folder(env):
    settings, repo = env
    aid, _ = _work(settings, repo, people=2)
    first = kept.save(repo, aid)
    second = kept.save(repo, aid)
    folder = files.assignment_dir(settings, aid, create=False) / "kept"
    assert second.saved_at >= first.saved_at
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        ["kept.json", "evaluator.pdf", *(f"{p['subject_id']}.pdf" for p in second.people),
         *(f"{p['subject_id']}.html" for p in second.people)])
    assert not list(folder.parent.glob("kept.*"))   # no half-written folder left


def test_a_newer_approval_makes_the_saved_files_out_of_date(env):
    settings, repo = env
    aid, sid = _work(settings, repo, people=2)
    saved = kept.save(repo, aid)
    subject = repo.list_subjects(sid)[0]
    later = (datetime.fromisoformat(saved.saved_at) + timedelta(seconds=5)).isoformat(
        timespec="microseconds")
    repo.conn.execute("UPDATE subjects SET approved_at=? WHERE id=?", (later, subject["id"]))
    assert kept.current(repo, aid) is None
    assert kept.load(settings, aid) == saved     # still on disk


def test_an_approval_taken_back_makes_the_saved_files_out_of_date(env):
    settings, repo = env
    aid, sid = _work(settings, repo, people=2)
    kept.save(repo, aid)
    repo.set_approval(repo.list_subjects(sid)[0]["id"], None)
    assert kept.current(repo, aid) is None


def test_drop_one_person(env):
    settings, repo = env
    aid, sid = _work(settings, repo)
    saved = kept.save(repo, aid)
    gone = saved.people[1]["subject_id"]
    assert kept.drop_person(settings, aid, gone) == 3   # pdf, html and the evaluator PDF
    left = kept.load(settings, aid)
    assert [p["subject_id"] for p in left.people] == [saved.people[0]["subject_id"],
                                                      saved.people[2]["subject_id"]]
    assert left.evaluator is None and kept.evaluator_pdf(settings, aid, left) is None
    assert "山田花子" not in pdf_text(kept.student_pdf(settings, aid, left))
    assert kept.drop_person(settings, aid, gone) == 0


def test_a_tampered_manifest_never_reads_outside_the_folder(env, tmp_path):
    settings, repo = env
    aid, _ = _work(settings, repo, people=1)
    kept.save(repo, aid)
    folder = files.assignment_dir(settings, aid, create=False) / "kept"
    secret = tmp_path / "secret.pdf"
    secret.write_bytes(b"%PDF-1.4 secret")
    manifest = json.loads((folder / "kept.json").read_text(encoding="utf-8"))
    manifest["people"][0]["subject_id"] = "../../../../secret"
    manifest["evaluator"] = str(secret)
    (folder / "kept.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = kept.load(settings, aid)
    assert loaded is None or (loaded.people == [] and loaded.evaluator is None)


def test_a_renamed_person_makes_the_saved_files_out_of_date(env):
    """Review MED 7: a fixed name must reach the next PDF (made anew from the texts)."""
    settings, repo = env
    aid, sid = _work(settings, repo, people=2)
    kept.save(repo, aid)
    assert kept.current(repo, aid) is not None
    repo.rename_subject(repo.list_subjects(sid)[0]["id"], "山田はなこ")
    assert kept.current(repo, aid) is None
