"""同意の記録: how the consent of the people in a work was obtained (optional, per work).

The record is a choice and a free note; nothing is ever blocked by it. When the choice is
「未確認」, the PDF step reminds the user, gently, to check before handing anything out.
The audit log records the choice only (the note may hold names).
"""
from pfb_app.storage.db import Repo

CHOICES = {"consented": "本人・保護者の同意あり", "policy": "団体の規定による",
           "unconfirmed": "未確認"}
NOT_RECORDED = "記録しない"
MAX_NOTE = 200
CARD_NOTE = 30   # characters of the note on the home card
TITLE = "同意の記録"
SAVE = "同意の記録を保存"
SAVED = "同意の記録を保存しました。"
REMINDER = ("この作業の同意の記録は「未確認」です。PDFを渡す前に、本人（未成年のときは保護者）の"
            "同意を確かめてください。確かめたら、ホームの「その他」→「同意の記録」で変えられます。")


class ConsentError(ValueError):
    """A value that cannot be saved (plain Japanese message)."""


NONE = ""   # 「記録しない」 as a choice on screen (a selectbox reads None as no choice)


def options() -> list[str]:
    """The choices in the order shown, 「記録しない」 (NONE) first."""
    return [NONE, *CHOICES]


def label(choice: str | None) -> str:
    return CHOICES.get(choice or "", NOT_RECORDED)


def _choice(work: dict) -> str | None:
    value = work.get("consent")
    return value if value in CHOICES else None


def record(repo: Repo, aid: int, choice: str | None, note: str | None) -> None:
    """Save the work's record; `choice` None (or NONE) clears it (and its note). KeyError: no
    such work."""
    choice = choice or None
    if choice is not None and choice not in CHOICES:
        raise ConsentError("同意の記録は一覧から選んでください。")
    note = " ".join((note or "").split())
    if len(note) > MAX_NOTE:
        raise ConsentError(f"メモは{MAX_NOTE}文字までにしてください。")
    with repo._tx():
        repo.set_consent(aid, choice, note if choice is not None else None)
        repo.audit("set_consent", f"assignment:{aid}",
                   {"consent": choice, "note": bool(note and choice is not None)})


def card_text(work: dict) -> str | None:
    """「同意：本人・保護者の同意あり（メモ）」 for the home card; None when not recorded."""
    choice = _choice(work)
    if choice is None:
        return None
    note = (work.get("consent_note") or "").strip()
    if len(note) > CARD_NOTE:
        note = note[:CARD_NOTE] + "…"
    return f"同意：{CHOICES[choice]}" + (f"（{note}）" if note else "")


def needs_reminder(work: dict) -> bool:
    """The PDF step reminds (never blocks) when the record says 「未確認」."""
    return _choice(work) == "unconfirmed"
