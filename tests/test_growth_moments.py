"""export.sheets.person_moments: the approved text's good moments with their categories, read
without videos, boxes or pictures (成長の記録 counts them for many works)."""
import pytest

from growth_synthetic import add_work
from pfb_app.export import sheets
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.config import Settings


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox"))


def test_moments_of_an_approved_person_match_the_handout(repo):
    aid, _, ids = add_work(repo, "定例会", {"山田": ["発言", "発言", "傾聴"], "鈴木": ["まとめ"]})
    got = sheets.person_moments(repo, ids["山田"])
    assert [m.category.label for m in got] == ["発言", "発言", "傾聴"]
    report = sheets.build_report(repo, aid, with_pictures=False)
    sheet = next(s for s in report.sheets if s.name == "山田")
    assert [m.category.key for m in got] == [m.category.key for m in sheet.moments]


def test_a_person_not_approved_has_no_moments(repo):
    _, _, ids = add_work(repo, "定例会", {"山田": ["発言"]}, approve=False)
    assert sheets.person_moments(repo, ids["山田"]) is None


def test_an_approval_the_text_outgrew_counts_as_not_approved(repo):
    from growth_synthetic import save
    from pfb_app.domain.models import Stage
    _, sid, ids = add_work(repo, "定例会", {"山田": ["発言"]})
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "新しい文", "evidence_refs": []},
         subject_id=ids["山田"])
    assert sheets.person_moments(repo, ids["山田"]) is None
