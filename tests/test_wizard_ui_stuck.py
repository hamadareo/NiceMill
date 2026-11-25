"""A stopped analysis is no dead end, and the waiting screen lets the coach go home (UX round 2,
lane L1). AppTest on ui/app.py, synthetic data only (no video frame is ever read)."""
from pfb_app.domain.models import Stage
from pfb_app.services import retry
from pfb_app.ui.wizard.progress import GIVE_UP, RETRY
from pfb_app.ui.wizard.router import DETAILS
from pfb_app.ui.wizard.step_wait import HOME_WHILE_RUNNING
from test_wizard_ui_basic import (PRIMARY_LABELS, app, env, html,  # noqa: F401 (env)
                                  no_internal_terms, ok, primaries, shown_step,
                                  video_assignment)
from test_wizard_ui_people_wait import (confirmed_and_analysing, group, job, links, save,
                                        worker_alive)

OOM = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
STOPPED = "解析が止まりました"


def failed_track(repo):
    aid = video_assignment(repo)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, "failed_retryable", camera="cam1", model="track",
              error=OOM)
    return aid, sid, jid


def retried_twice(repo, aid, jid):
    for _ in range(2):
        retry.retry_work(repo, aid)
        repo.conn.execute("UPDATE jobs SET status='failed', error=? WHERE id=?", (OOM, jid))


def retries(repo, aid):
    return repo.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action=? AND target=?",
                             (retry.ACTION, f"assignment:{aid}")).fetchone()[0]


def test_retry_is_a_primary_label():
    assert RETRY in PRIMARY_LABELS
    assert DETAILS == "くわしく見る（管理用の画面）"


def test_a_stopped_track_opens_with_the_retry_as_the_only_primary(env):
    settings, repo = env
    aid, _, jid = failed_track(repo)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    page = html(at)
    assert STOPPED in page and "映っている人の名前を選んでください" not in page
    assert any("ほかの解析と重なって止まりました" in e.value for e in at.error)
    assert primaries(at) == [RETRY]
    assert "これで決定して次へ" not in [b.label for b in at.button]
    assert DETAILS in links(at) and "詳細画面で直す" not in links(at)
    no_internal_terms(at)
    next(b for b in at.button if b.label == RETRY).click().run()
    ok(at)
    assert repo.get_job(jid).status.value == "queued" and retries(repo, aid) == 1


def test_a_stopped_behavior_offers_the_retry_on_step_4(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='failed', error=? WHERE submission_id=?",
                      ("ValueError: 行動を読み取れませんでした", sid))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    page = html(at)
    assert STOPPED in page and "行動を読み取れませんでした" in page
    assert primaries(at) == [RETRY]
    assert DETAILS in links(at)
    no_internal_terms(at)
    next(b for b in at.button if b.label == RETRY).click().run()
    ok(at)
    assert retries(repo, aid) == 1
    assert all(j["status"] != "failed" for j in repo.submission_job_rows(sid))


def test_after_two_failed_retries_the_wizard_says_to_tell_the_administrator(env):
    settings, repo = env
    aid, _, jid = failed_track(repo)
    retried_twice(repo, aid, jid)
    at = ok(app(session={"assignment_id": aid}))
    assert any(GIVE_UP in e.value for e in at.error)
    assert RETRY not in [b.label for b in at.button]
    assert DETAILS in links(at)


def test_the_home_card_of_a_stopped_work_offers_the_retry(env):
    settings, repo = env
    aid, _, jid = failed_track(repo)
    at = ok(app())
    retry_button = at.button(key=f"pfb_retry_{aid}")
    assert retry_button.label == RETRY and retry_button.proto.type != "primary"
    assert len(primaries(at)) == 1                 # the home keeps one primary
    assert DETAILS in links(at) and "詳細画面で直す" not in links(at)
    assert "ほかの解析と重なって止まりました" in html(at)
    no_internal_terms(at)
    retry_button.click().run()
    ok(at)
    assert repo.get_job(jid).status.value == "queued" and retries(repo, aid) == 1


def test_the_home_card_gives_up_after_two_failed_retries(env):
    settings, repo = env
    aid, _, jid = failed_track(repo)
    retried_twice(repo, aid, jid)
    at = ok(app())
    assert f"pfb_retry_{aid}" not in [b.key for b in at.button]
    assert GIVE_UP in html(at) and DETAILS in links(at)


def test_the_waiting_screen_lets_the_coach_go_home(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 4
    home = at.button(key="pfb_wait_next")
    assert home.label == HOME_WHILE_RUNNING == "ホームへ戻る（解析は続きます）"
    assert home.proto.type != "primary" and not home.disabled
    assert primaries(at) == []
    home.click().run()
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] is None
    assert at.query_params.get("work") is None


def test_the_stepper_and_words_are_plain_while_analysing(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu' WHERE submission_id=?", (sid,))
    at = ok(app(session={"assignment_id": aid}))
    page = html(at)
    for word in ("人を見つける", "したことの書き出し", "よかった点を選ぶ", "文章づくり", "順番待ち"):
        assert word in page, word
    for word in ("GPU", "人物の追跡", "貢献の判定", "フィードバック作成"):
        assert word not in page, word


HOSTILE = "*太字* _下線_ [リンク](x) <b>タグ</b>"


def test_the_home_retry_flash_escapes_the_title(env):
    from pfb_app.ui.wizard import home
    settings, repo = env
    aid = video_assignment(repo, title=HOSTILE)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    job(repo, sid, Stage.track, "failed_retryable", camera="cam1", model="track", error=OOM)
    at = ok(app())
    at.button(key=f"pfb_retry_{aid}").click().run()
    ok(at)
    assert [s.value for s in at.success] == [f"「{home.md_escape(HOSTILE)}」をもう一度ためしています。"]
    for ch in ("*", "_", "[", "]", "<", ">"):
        assert f"\\{ch}" in at.success[0].value, ch


def test_md_escape():
    from pfb_app.ui.wizard import home
    assert home.md_escape("IMG_4821") == "IMG\\_4821"
    assert home.md_escape("練習試合") == "練習試合"
    assert home.md_escape("*a* [b] <c>") == "\\*a\\* \\[b\\] \\<c\\>"


def test_md_escape_keeps_colons_from_making_emoji():
    from pfb_app.ui.wizard import home
    assert home.md_escape(":smile: 練習") == "\\:smile\\: 練習"


def test_a_second_retry_press_is_shown_as_info_not_error(monkeypatch):
    """retry.RETRYING (the retry already runs) is st.info on step 3/4; other refusals st.error."""
    from pfb_app.ui.wizard import step_wait
    shown = []
    monkeypatch.setattr(step_wait.st, "button", lambda *a, **k: True)
    monkeypatch.setattr(step_wait.st, "info", lambda m: shown.append(("info", m)))
    monkeypatch.setattr(step_wait.st, "error", lambda m: shown.append(("error", m)))
    for reason in (retry.RETRYING, retry.NOTHING):
        def refuse(repo, aid, reason=reason):
            raise retry.CannotRetry(reason)
        monkeypatch.setattr(step_wait.retry, "retry_work", refuse)
        step_wait.retry_button(None, 1, "k", primary=False)
    assert [kind for kind, _ in shown] == ["info", "error"]
    assert retry.RETRYING in shown[0][1]
