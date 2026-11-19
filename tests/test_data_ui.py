"""「データの扱い」 on the screens: the home card's lines and 「同意の記録」, the reminder and the
saved PDFs on step 6, and the admin page (setting, preview, 「この人のデータを消す」).
Synthetic works only."""
from datetime import datetime, timedelta, timezone

from streamlit.testing.v1 import AppTest

from data_helpers import hand_out
from pfb_app.services import erasure, kept, retention, review
from pfb_app.ui.wizard import step_pdf
from pdf_synthetic import video_group, work
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import (FORBIDDEN, UI, app, env, html, ok, primaries,  # noqa: F401
                                  screen_text, video_assignment)
from test_wizard_ui_review_pdf import reviewable, subject


def _clean(text: str) -> None:
    found = [w for w in (*SCHOOL_WORDS, "ルーブリック", *FORBIDDEN) if w in text]
    assert not found, found


def finished(repo, aid, days_ago=40):
    hand_out(repo, aid, datetime.now(timezone.utc) - timedelta(days=days_ago))


def test_home_card_lines_and_the_consent_record(env):
    settings, repo = env
    aid = video_assignment(repo, "春の練習")
    at = ok(app())
    assert "同意：" not in html(at)
    at.selectbox(key=f"pfb_consent_{aid}").set_value("unconfirmed").run()
    at.text_input(key=f"pfb_consent_note_{aid}").input("保護者に確認中").run()
    ok(at.button(key=f"pfb_consent_go_{aid}").click().run())
    assert repo.get_assignment(aid)["consent"] == "unconfirmed"
    assert "同意：未確認（保護者に確認中）" in html(at)
    retention.save(repo, 90, False)
    at = ok(app())
    assert "PDFを受け取ってから90日後に、動画が消えます" in html(at)
    assert len(primaries(at)) == 1
    _clean(screen_text(at))


def test_step6_reminds_when_consent_is_unconfirmed(env):
    from pfb_app.services import consent
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert not any("未確認" in i.value for i in at.info)
    consent.record(repo, aid, "unconfirmed", "")
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert any("同意の記録は「未確認」" in i.value for i in at.info)
    assert primaries(at) == ["参加者に渡すPDFを受け取る"]      # never blocked
    assert not at.download_button[0].proto.disabled


def test_step6_hands_out_the_saved_pdfs_once_the_videos_are_gone(env):
    from pdfhelp import pdf_text
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=20)
    finished(repo, aid)
    retention.clear_media(repo, settings, aid)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert any("消す前に保存したPDF" in c.value for c in at.caption)
    assert not any(c.key == "pfb_pdf_growth" for c in at.checkbox)   # no growth page offered
    record = kept.current(repo, aid)
    data = step_pdf.kept_data(aid, "student", record)()
    assert "山田花子" in pdf_text(data)
    assert step_pdf.kept_data(aid, "teacher", record)()[:4] == b"%PDF"
    _clean(screen_text(at))


def _page() -> AppTest:
    return AppTest.from_file(str(UI / "pages" / "data.py"), default_timeout=60)


def test_the_data_page_explains_sets_and_erases(env):
    settings, repo = env
    aid = work(repo, title="秋の試合")
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    finished(repo, aid)
    at = ok(_page().run())
    text = screen_text(at)
    assert "外のサービスに送ることはありません" in text and str(settings.data_dir) in text
    assert "設定されていません" in text
    assert primaries(at) == ["この設定にする"]
    _clean(text)
    at.radio(key="pfb_data_days").set_value(30).run()
    assert any("1件の作業の動画が消えます" in c.value for c in at.caption)
    assert "「秋の試合」の動画" in html(at)                  # named before anything goes
    assert at.button(key="pfb_data_save").proto.disabled     # asked first (review MED 4)
    at.checkbox(key="pfb_data_sure_30_0").check().run()
    ok(at.button(key="pfb_data_save").click().run())
    assert retention.load(repo) == retention.Policy(30, False)
    at.checkbox(key="pfb_data_all").check().run()            # すべて消す: asked again
    assert "「秋の試合」" in html(at) and at.button(key="pfb_data_save").proto.disabled
    at.checkbox(key="pfb_data_all").uncheck().run()
    # 「この人のデータを消す」
    target = next(s for s in repo.list_subjects(sid) if s["display_name"] == "山田花子")
    at.selectbox(key="pfb_data_person").set_value(target["id"]).run()
    text = screen_text(at)
    assert "「秋の試合」の山田花子さん" in text and "元に戻せません" in text
    assert at.button(key=f"pfb_data_erase_{target['id']}").proto.disabled
    at.checkbox(key=f"pfb_data_sure_{target['id']}").check().run()
    ok(at.button(key=f"pfb_data_erase_{target['id']}").click().run())
    assert repo.get_subject(target["id"])["display_name"] == erasure.ERASED_NAME
    _clean(screen_text(at))


def test_the_data_page_is_in_the_admin_menu(env):
    at = ok(app())
    assert any(link.proto.label == "データの扱い" for link in at.get("page_link"))


def test_every_step_of_a_work_without_its_videos_still_opens(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    finished(repo, aid)
    retention.clear_media(repo, settings, aid)
    at = ok(app())
    assert "動画は" in html(at) and "に消しました" in html(at)
    for step in range(1, 7):
        ok(app(session={"assignment_id": aid, "step": step, "submission_id": sid}))


def test_every_step_after_a_person_was_erased_still_opens(env):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=20)
    erasure.erase(repo, settings, repo.list_subjects(sid)[0]["id"])
    for step in range(1, 7):
        at = ok(app(session={"assignment_id": aid, "step": step, "submission_id": sid}))
        assert "山田花子" not in screen_text(at)


def test_the_timing_text_counts_the_hourly_passes():
    from pfb_app.ui.wizard import data_view
    assert data_view.timing_text(0, False) == "この設定で、いますぐ消えるものはありません。"
    assert "次の1時間のうちに3件の作業の動画" in data_view.timing_text(3, False)
    assert data_view.timing_text(7, True) == \
        "この設定にすると、7件の作業が、1時間に3件ずつ、約3時間かけて消えます。"
