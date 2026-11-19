"""同意の記録: the work's optional record of how consent was obtained (synthetic rows only)."""
import pytest

from pfb_app.services import consent
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn)


def _work(repo):
    return repo.create_assignment("練習試合", "video", "試合", "", "standard", False)


def test_the_choices_are_the_brief_s_words():
    assert list(consent.CHOICES.values()) == ["本人・保護者の同意あり", "団体の規定による", "未確認"]
    assert consent.options() == ["", *consent.CHOICES]
    assert consent.label(None) == consent.label("") == "記録しない"


def test_record_show_and_clear(repo):
    aid = _work(repo)
    assert consent.card_text(repo.get_assignment(aid)) is None
    assert not consent.needs_reminder(repo.get_assignment(aid))
    consent.record(repo, aid, "consented", "  4月の説明会で書面  ")
    work = repo.get_assignment(aid)
    assert consent.card_text(work) == "同意：本人・保護者の同意あり（4月の説明会で書面）"
    assert not consent.needs_reminder(work)
    consent.record(repo, aid, "unconfirmed", "")
    work = repo.get_assignment(aid)
    assert consent.card_text(work) == "同意：未確認"
    assert consent.needs_reminder(work)
    entry = repo.latest_audit("set_consent", f"assignment:{aid}")
    assert entry["detail"] == {"consent": "unconfirmed", "note": False}   # never the note
    consent.record(repo, aid, None, "メモだけ")
    work = repo.get_assignment(aid)
    assert consent.card_text(work) is None and work["consent_note"] is None


def test_a_long_note_is_shortened_on_the_card_only(repo):
    aid = _work(repo)
    consent.record(repo, aid, "policy", "あ" * 60)
    work = repo.get_assignment(aid)
    assert work["consent_note"] == "あ" * 60
    assert consent.card_text(work) == "同意：団体の規定による（" + "あ" * 30 + "…）"


def test_wrong_values_are_refused(repo):
    aid = _work(repo)
    with pytest.raises(consent.ConsentError):
        consent.record(repo, aid, "yes", "")
    with pytest.raises(consent.ConsentError):
        consent.record(repo, aid, "policy", "x" * (consent.MAX_NOTE + 1))
    with pytest.raises(KeyError):
        consent.record(repo, aid + 9, "policy", "")


def test_an_unknown_stored_value_reads_as_not_recorded(repo):
    aid = _work(repo)
    repo.conn.execute("UPDATE assignments SET consent='odd' WHERE id=?", (aid,))
    work = repo.get_assignment(aid)
    assert consent.card_text(work) is None and not consent.needs_reminder(work)
