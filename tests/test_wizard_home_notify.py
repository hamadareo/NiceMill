"""UX round 2, lane L2: home in groups, 「その他」 (名前を変える, 削除), and telling the coach
that the analysis has finished (tab title, optional chime, 「あと約n分（目安）」 on cards).

Only synthetic data: invented titles and names, generated tiny videos."""
import html as htmllib
import json
import re
from collections import Counter
from types import SimpleNamespace

import pytest

from pfb_app.domain.models import Stage
from pfb_app.services import assignments
from pfb_app.services.assignments import mark_pdf_downloaded
from pfb_app.services.review import approve
from pfb_app.storage import db
from pfb_app.ui.wizard import home, live, notify
from test_wizard_progress import group as progress_group, tracked_to_review
from test_wizard_ui_basic import (app, env, html, ok, primaries, register,  # noqa: F401 (env)
                                  screen_text, video_assignment)
from test_wizard_ui_people_wait import (confirmed_and_analysing, group, save, worker_alive)


def review_ready(repo, title="見直しの作業"):
    aid = video_assignment(repo, title)
    sid, subjects = progress_group(repo, aid)
    tracked_to_review(repo, sid, subjects)
    return aid, subjects


def complete_work(repo, title="終わった作業"):
    aid = video_assignment(repo, title)
    sid, subjects = progress_group(repo, aid, students=(("Pa", "A子"),))
    tracked_to_review(repo, sid, subjects)
    approve(repo, subjects["Pa"])
    mark_pdf_downloaded(repo, aid)
    return aid


def analysing(settings, repo, title="解析中の作業", fraction=0.3):
    aid = video_assignment(repo, title)
    sid = confirmed_and_analysing(settings, repo, aid)
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=?", (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    repo.set_job_progress(jid, fraction, None)
    return aid


def notice(at) -> dict:
    """The hidden notice element's data (title, sound, chimes) of this run."""
    bodies = [el.proto.body for el in at.get("html") if "data-pfb-notify" in el.proto.body]
    assert len(bodies) == 1, bodies
    raw = re.search(r'data-pfb-notify="([^"]*)"', bodies[0]).group(1)
    return json.loads(htmllib.unescape(raw))


def notice_element(at):
    return next(el for el in at.get("html") if "data-pfb-notify" in el.proto.body)


# --- #7 home in groups --------------------------------------------------------------------

def test_home_groups_your_turn_running_and_finished_collapsed(env):
    settings, repo = env
    worker_alive(settings)
    done = complete_work(repo)
    running = analysing(settings, repo)
    turn = video_assignment(repo, "あなたの番の作業")         # no video yet: the coach's turn
    at = ok(app())
    page = html(at)
    assert "あなたの番" in page and "解析中" in page
    yours = page.index('<h2 class="pfb-homegroup"')
    card = '<h3 class="pfb-workcard__title">{}</h3>'.format
    assert yours < page.index(card("あなたの番の作業")) < page.index(card("解析中の作業"))
    heads = re.findall(r'<h2 class="pfb-homegroup">(.*?)</h2>', page)
    assert [re.sub(r"<[^>]+>", "", h) for h in heads] == ["あなたの番（1件）", "解析中（1件）"]
    finished = next(e for e in at.expander if e.label.startswith("終わった作業"))
    assert finished.label == "終わった作業（1件）"
    assert not finished.proto.expanded                         # collapsed
    inside = "\n".join(el.proto.body for el in finished.get("html"))
    assert '<h3 class="pfb-workcard__title">終わった作業</h3>' in inside   # its card inside
    assert any(b.key == f"pfb_open_{done}" for b in finished.button)
    assert primaries(at) == ["続ける"]                        # still one primary
    assert at.button(key=f"pfb_open_{turn}").proto.type == "primary"
    assert at.button(key=f"pfb_open_{running}").proto.type == "secondary"


def test_a_stopped_work_is_in_your_turn(env):
    settings, repo = env
    aid = video_assignment(repo, "止まった作業")
    sid = register(settings, repo, aid)
    job = repo.claim_next_job({"cpu"})
    assert job is not None and job.submission_id == sid
    repo.finish_job(job.id, "failed", "VideoError: 動画を読めません")
    at = ok(app())
    page = html(at)
    heads = [re.sub(r"<[^>]+>", "", h)
             for h in re.findall(r'<h2 class="pfb-homegroup">(.*?)</h2>', page)]
    assert heads == ["あなたの番（1件）"]
    assert not [e for e in at.expander if e.label.startswith("終わった作業")]


def test_phones_drop_the_mini_stepper(env):
    _, repo = env
    video_assignment(repo)
    at = ok(app())
    css = "\n".join(el.proto.body for el in at.get("html") if "<style>" in el.proto.body)
    rule = re.search(r"@media \(max-width: 40rem\)\s*\{([^@]*)\}", css)
    assert rule and re.search(r"\.pfb-mini\s*\{\s*display:\s*none", rule.group(1))
    assert 'class="pfb-mini"' in html(at)                     # wider screens keep it


# --- #7 「その他」: 名前を変える and 削除 -----------------------------------------------------

def test_more_popover_holds_rename_and_delete(env):
    _, repo = env
    aid = video_assignment(repo, "IMG_4821")
    at = ok(app())
    more = [p for p in at.get("popover") if p.proto.popover.label == "その他"]
    assert len(more) == 1
    assert at.text_input(key=f"pfb_rename_{aid}").value == "IMG_4821"
    assert at.text_input(key=f"pfb_rename_{aid}").label == "名前を変える"
    delete = at.button(key=f"pfb_del_{aid}")
    assert delete.label == "削除" and delete.proto.type == "tertiary"
    assert at.button(key=f"pfb_rename_go_{aid}").proto.type != "primary"


def test_rename_from_home_changes_the_title_only(env):
    _, repo = env
    aid = video_assignment(repo, "IMG_4821")
    before = repo.get_assignment(aid)
    at = ok(app())
    at.text_input(key=f"pfb_rename_{aid}").input("  練習試合 \n")
    at = ok(at.button(key=f"pfb_rename_go_{aid}").click().run())
    after = repo.get_assignment(aid)
    assert after["title"] == "練習試合"
    assert {k: v for k, v in after.items() if k != "title"} == \
        {k: v for k, v in before.items() if k != "title"}
    assert any(s.value == "名前を「練習試合」に変えました。" for s in at.success)
    assert "練習試合" in html(at) and "IMG_4821" not in html(at)
    assert at.session_state["wizard"]["assignment_id"] is None      # still home
    assert at.text_input(key=f"pfb_rename_{aid}").value == "練習試合"   # the cleaned name


def test_rename_with_an_empty_name_keeps_the_title(env):
    _, repo = env
    aid = video_assignment(repo, "IMG_4821")
    at = ok(app())
    at.text_input(key=f"pfb_rename_{aid}").input("   ")
    at = ok(at.button(key=f"pfb_rename_go_{aid}").click().run())
    assert repo.get_assignment(aid)["title"] == "IMG_4821"
    assert any(assignments.NO_TITLE in w.value for w in at.warning)


def test_rename_work_service(env):
    settings, repo = env
    aid = video_assignment(repo, "IMG_4821")
    register(settings, repo, aid)                  # analysis started: the name may still change
    assert assignments.rename_work(repo, aid, " 練習　試合 ") == "練習 試合"
    assert repo.get_assignment(aid)["title"] == "練習 試合"
    assert repo.get_assignment(aid)["task_text"] == "塔を高く作る"
    with pytest.raises(ValueError):
        assignments.rename_work(repo, aid, " \n ")
    with pytest.raises(KeyError):
        assignments.rename_work(repo, aid + 99, "名前")
    long = "あ" * (assignments.MAX_TITLE + 10)
    assert assignments.rename_work(repo, aid, long) == "あ" * assignments.MAX_TITLE
    actions = [r[0] for r in repo.conn.execute("SELECT action FROM audit_log")]
    assert actions.count("rename_work") == 2


def test_help_mentions_more_and_finished(env):
    _, repo = env
    video_assignment(repo)
    text = screen_text(ok(app()))
    assert "「その他」" in text and "名前を変える" in text


# --- #4(c) 「あと約n分（目安）」 on cards ----------------------------------------------------

def test_card_shows_time_left_when_known(env):
    settings, repo = env
    worker_alive(settings)
    aid = analysing(settings, repo, fraction=0.5)
    at = ok(app())
    assert "あと約" not in html(at)                          # no history yet: nothing
    old = group(repo, video_assignment(repo, "前の作業"), "前の班")
    subj = repo.list_subjects(old)[0]["id"]
    save(repo, old, Stage.behavior, {"events": []}, camera="cam1", duration=120)
    save(repo, old, Stage.contribution, {"judgments": []}, camera="cam1", duration=60)
    save(repo, old, Stage.pfb, {"pfb_status": "generated", "pfb": "良い", "evidence_refs": []},
         subject_id=subj, duration=30)
    db.clear_stage_durations_cache()
    assert live.remaining_sec(repo, aid) == 210
    at = ok(app())
    page = html(at)
    assert '<p class="pfb-workcard__eta">あと約4分（目安）</p>' in page
    assert page.count("pfb-workcard__eta") == 1            # only the analysing card


def test_time_left_text():
    assert home.time_left_text(30) == "あと約1分（目安）"
    assert home.time_left_text(11 * 60) == "あと約15分（目安）"
    assert home.time_left_text(61 * 60) == "あと約1時間10分（目安）"


# --- #4(a) the tab title ---------------------------------------------------------------------

def test_tab_title_of_a_work_follows_its_state(env):
    settings, repo = env
    worker_alive(settings)
    ready, _ = review_ready(repo, "練習試合")
    at = ok(app(session={"assignment_id": ready}))
    assert notice(at)["title"] == "（確認できます）練習試合｜NiceMill"
    running = analysing(settings, repo, "紅白戦")
    at = ok(app(session={"assignment_id": running}))
    assert notice(at)["title"] == "（解析中 42%）紅白戦｜NiceMill"
    empty = video_assignment(repo, "まだ空の作業")
    at = ok(app(session={"assignment_id": empty}))
    assert notice(at)["title"] == "（あなたの番）まだ空の作業｜NiceMill"
    done = complete_work(repo, "決勝")
    at = ok(app(session={"assignment_id": done}))
    assert notice(at)["title"] == "（完了）決勝｜NiceMill"


def test_tab_title_on_home_counts_your_turn(env):
    settings, repo = env
    worker_alive(settings)
    at = ok(app())
    assert notify.APP == "NiceMill"   # the app's name, as app.py's page_title
    assert notice(at)["title"] == notify.APP
    analysing(settings, repo)
    at = ok(app())
    assert notice(at)["title"] == f"（解析中 42%）{notify.APP}"
    review_ready(repo)
    video_assignment(repo, "空の作業")
    at = ok(app())
    assert notice(at)["title"] == f"（あなたの番 2件）{notify.APP}"


def test_notice_element_is_hidden_and_runs_its_script(env):
    _, repo = env
    video_assignment(repo)
    at = ok(app())
    el = notice_element(at)
    assert el.proto.unsafe_allow_javascript
    body = el.proto.body
    assert ".st-key-pfb_notify" in body and "display: none" in body
    assert "document.title" in body
    assert "prefers-reduced-motion: reduce" in body           # no sound under reduced motion


# --- #4(b) the optional chime ---------------------------------------------------------------

def W(aid, your_turn=False, stopped=False, complete=False, analysis=None):
    return SimpleNamespace(assignment_id=aid, your_turn=your_turn, stopped=stopped,
                           complete=complete, analysis=analysis)


def test_newly_ready_only_on_a_running_to_ready_transition():
    seen = {}
    assert notify.newly_ready(seen, [W(1, your_turn=True), W(2)], everyone=True) == []
    assert notify.newly_ready(seen, [W(1, your_turn=True), W(2)], everyone=True) == []
    assert notify.newly_ready(seen, [W(1, your_turn=True), W(2, your_turn=True)],
                              everyone=True) == [2]
    # again (another refresh): no second chime
    assert notify.newly_ready(seen, [W(1, your_turn=True), W(2, your_turn=True)],
                              everyone=True) == []
    # stopped or complete is not "ready"
    assert notify.newly_ready(seen, [W(3)], everyone=False) == []
    assert notify.newly_ready(seen, [W(3, stopped=True)], everyone=False) == []
    assert notify.newly_ready(seen, [W(3, your_turn=True)], everyone=False) == []
    # one work's screen keeps what it knows of the others; home forgets deleted ones
    assert set(seen) == {1, 2, 3}
    notify.newly_ready(seen, [W(1, your_turn=True)], everyone=True)
    assert set(seen) == {1}


def test_sound_is_off_by_default_with_its_toggle_on_home(env):
    settings, repo = env
    worker_alive(settings)
    analysing(settings, repo)
    at = ok(app())
    toggle = at.toggle(key=notify.SOUND_WIDGET)
    assert toggle.label == "終わったら音で知らせる" and toggle.value is False
    assert notice(at)["sound"] is False


def test_one_chime_when_a_work_becomes_ready_with_sound_on(env):
    settings, repo = env
    ready, _ = review_ready(repo, "練習試合")
    # the session saw it analysing before; it is ready now
    at = ok(app(extra={notify.SEEN: {ready: notify.RUNNING}}))
    assert notice(at)["chimes"] == []                         # sound off: nothing
    at = ok(app(extra={notify.SEEN: {ready: notify.RUNNING}, notify.SOUND: True}))
    chimes = notice(at)["chimes"]
    assert len(chimes) == 1 and notice(at)["sound"] is True
    at = ok(at.run())                                         # a refresh: the same chime id,
    assert notice(at)["chimes"] == chimes                     # the page plays each id once


def test_chime_ids_expire(env, monkeypatch):
    settings, repo = env
    ready, _ = review_ready(repo)
    clock = [1000.0]
    monkeypatch.setattr(notify, "_now", lambda: clock[0])
    at = ok(app(extra={notify.SEEN: {ready: notify.RUNNING}, notify.SOUND: True}))
    assert len(notice(at)["chimes"]) == 1
    clock[0] += notify.CHIME_KEEP_SEC + 1
    at = ok(at.run())
    assert notice(at)["chimes"] == []


def test_toggle_keeps_its_choice_when_it_is_not_shown(env):
    settings, repo = env
    worker_alive(settings)
    aid = analysing(settings, repo)
    at = ok(app())
    at = ok(at.toggle(key=notify.SOUND_WIDGET).set_value(True).run())
    assert at.session_state[notify.SOUND] is True
    at.session_state["wizard"] = {"assignment_id": aid, "step": None, "submission_id": None}
    at.query_params["work"] = str(aid)
    at = ok(at.run())                                         # a work screen: no toggle
    assert notice(at)["sound"] is True


def test_home_signature_moves_with_state_and_tens_of_percent(env):
    settings, repo = env
    worker_alive(settings)
    aid = analysing(settings, repo)
    from pfb_app.ui.wizard.progress import all_work
    before = notify.signature(all_work(repo))
    assert before == notify.signature(all_work(repo))
    jid = repo.conn.execute(
        "SELECT j.id FROM jobs j JOIN submissions s ON j.submission_id = s.id"
        " WHERE s.assignment_id=? AND j.status='running'", (aid,)).fetchone()[0]
    repo.set_job_progress(jid, 0.9, None)
    assert notify.signature(all_work(repo)) != before
    assert Counter(s for _, s, _ in notify.signature(all_work(repo))) == Counter({"running": 1})


# --- review fixes (2026-10-22) ---------------------------------------------------------------

HOSTILE = """"><img src=x onerror=alert(1)></script><script>alert(2)</script>'`${x}"""


def test_a_hostile_name_stays_text(env):
    _, repo = env
    aid = video_assignment(repo, "IMG_4821")
    at = ok(app())
    at.text_input(key=f"pfb_rename_{aid}").input(HOSTILE)
    at = ok(at.button(key=f"pfb_rename_go_{aid}").click().run())
    assert repo.get_assignment(aid)["title"] == HOSTILE
    bodies = "\n".join(el.proto.body for el in at.get("html"))
    assert bodies.count("<script") == 1                      # only the notice's own script
    assert "<img" not in bodies
    assert notice(at)["title"] == f"（あなたの番 1件）{notify.APP}"
    message = next(s.value for s in at.success if s.value.startswith("名前を"))
    assert message == f"名前を「{home.md_escape(HOSTILE)}」に変えました。"
    assert "\\<img" in message and "\\`" in message and "\\$" in message
    # on the work's own screen the tab title carries the name as plain data
    at = ok(app(session={"assignment_id": aid}))
    assert notice(at)["title"] == f"（あなたの番）{HOSTILE}｜NiceMill"
    bodies = "\n".join(el.proto.body for el in at.get("html"))
    assert bodies.count("<script") == 1 and "<img" not in bodies


def test_md_escape():
    assert home.md_escape("IMG_4821") == "IMG\\_4821"
    assert home.md_escape("練習試合") == "練習試合"
    assert home.md_escape("*a* [b](c) `d` <e> $f$ \\") == \
        "\\*a\\* \\[b\\]\\(c\\) \\`d\\` \\<e\\> \\$f\\$ \\\\"


def test_a_home_run_reads_every_work_once(env, monkeypatch):
    from pfb_app.ui.wizard import progress
    settings, repo = env
    worker_alive(settings)
    analysing(settings, repo)
    review_ready(repo)
    calls = []
    real = progress.all_work

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(home, "all_work", counted)
    monkeypatch.setattr(notify, "all_work", counted)
    at = ok(app())
    assert notify.watching(home.groups(real(repo))[1])       # the refresh is on
    assert len(calls) == 1


def test_the_refresh_runs_only_while_something_is_analysed(env, monkeypatch):
    settings, repo = env
    worker_alive(settings)
    started = []
    monkeypatch.setattr(notify, "_home_watch", lambda base, first: started.append(base))
    video_assignment(repo, "空の作業")
    complete_work(repo)
    review_ready(repo)
    ok(app())
    assert started == []
    analysing(settings, repo)
    ok(app())
    assert len(started) == 1


def test_watching_follows_active_analysis_not_only_the_state():
    active = SimpleNamespace(active=True)
    idle = SimpleNamespace(active=False)
    assert notify.watching([W(1)])                                   # analysing
    assert notify.watching([W(1, your_turn=True, analysis=active)])  # one group still moving
    assert not notify.watching([W(1, your_turn=True, analysis=idle)])
    assert not notify.watching([W(1, complete=True), W(2, stopped=True)])


def test_needs_rerun():
    assert notify.needs_rerun((1,), (2,), held=False)
    assert not notify.needs_rerun((1,), (1,), held=False)
    assert not notify.needs_rerun((1,), (2,), held=True)


def test_no_rerun_while_deleting_or_renaming(env):
    settings, repo = env
    worker_alive(settings)
    aid = analysing(settings, repo, "紅白戦")
    at = ok(app())
    assert at.session_state[notify.HOLD] is False
    at = ok(at.button(key=f"pfb_del_{aid}").click().run())         # the confirm area opens
    assert at.session_state[notify.HOLD] is True
    at = ok(at.button(key=f"pfb_delete_cancel_{aid}").click().run())
    assert at.session_state[notify.HOLD] is False
    at = ok(at.text_input(key=f"pfb_rename_{aid}").input("紅白戦（前半）").run())
    assert at.session_state[notify.HOLD] is True                    # typing a new name
    at = ok(at.button(key=f"pfb_rename_go_{aid}").click().run())
    assert at.session_state[notify.HOLD] is False                   # saved: the field matches
