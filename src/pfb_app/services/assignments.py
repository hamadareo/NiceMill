"""Small assignment services for the wizard: edit the task before analysis, rename a work,
record PDF downloads."""
from pfb_app.services.review import BusyError
from pfb_app.storage.db import Repo

__all__ = ["BusyError", "mark_pdf_downloaded", "rename_work", "update_assignment"]

STARTED = "解析が始まっているため、活動の内容と評価の観点は変更できません。"
NO_TITLE = "名前を入れてください。"
MAX_TITLE = 80


def update_assignment(repo: Repo, aid: int, task_text: str, rubric_text: str) -> None:
    """Change the task text and rubric while no job has been made for any submission of the
    assignment (results already computed would no longer match). Raises BusyError once a job
    exists, ValueError for an empty task text and KeyError for an unknown assignment."""
    task = (task_text or "").strip()
    if not task:
        raise ValueError("活動の内容が空です。")
    rubric = (rubric_text or "").strip()
    if not repo.update_assignment_texts(aid, task, rubric):
        raise BusyError(STARTED)
    repo.audit("update_assignment", f"assignment:{aid}", {})


def rename_work(repo: Repo, aid: int, title: str) -> str:
    """Change only the work's name (what its home card and header show), at any time: spaces
    and line breaks collapse to one space, at most MAX_TITLE characters. Returns the name
    saved. ValueError for an empty name, KeyError for an unknown work."""
    name = " ".join((title or "").split())[:MAX_TITLE].strip()
    if not name:
        raise ValueError(NO_TITLE)
    if not repo.rename_assignment(aid, name):
        raise KeyError(f"assignment {aid} not found")
    repo.audit("rename_work", f"assignment:{aid}", {})
    return name


def mark_pdf_downloaded(repo: Repo, aid: int) -> None:
    """Record that the student PDF of the assignment was downloaded (now)."""
    at = repo.set_pdf_downloaded(aid)
    repo.audit("pdf_downloaded", f"assignment:{aid}", {"at": at})
