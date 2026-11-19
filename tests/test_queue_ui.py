"""待ち時間の目安・順番待ち・終わりの帯・作業をさがす on the screens (AppTest on ui/app.py).

Only synthetic data: invented works, names and numbers."""
import re
from types import SimpleNamespace

from pfb_app.services import eta
from pfb_app.ui.wizard import done_band, find
from test_eta import REAL_RUNS, history_of, queued_work
from test_wizard_ui_basic import app, env, html, no_internal_terms, ok  # noqa: F401 (env)
from test_wizard_ui_basic import video_assignment
from test_wizard_ui_people_wait import confirmed_and_analysing, group, worker_alive
from test_wizard_ui_review_pdf import reviewable


def etas(at) -> list[str]:
    return re.findall(r'<p class="pfb-workcard__eta">([^<]*)</p>', html(at))


# --- the cards ------------------------------------------------------------------------------

def test_cards_say_the_place_in_the_queue_and_the_time_with_the_waiting(env):
    settings, repo = env
    worker_alive(settings)
    history_of(repo, REAL_RUNS)
    a, *_ = queued_work(settings, repo, "走っている作業", status="running")
    b, *_ = queued_work(settings, repo, "順番待ちの作業")
    at = ok(app())
    lines = etas(at)
    assert eta.queue_text(1) in lines
    assert any(l.startswith("順番待ちをふくめて、あと約") and l.endswith("かもしれません（目安）")
               for l in lines)
    assert sum(l.startswith("あと約") for l in lines) == 1        # the running work's own time
    assert not any("目安はまだ出せません" in l for l in lines)
    no_internal_terms(at)


def test_a_card_without_enough_history_says_so_and_an_idle_one_says_nothing(env):
    settings, repo = env
    worker_alive(settings)
    queued_work(settings, repo, "解析を待つ作業")
    video_assignment(repo, "何も入っていない作業")
    at = ok(app())
    assert etas(at) == [eta.EMPTY]


def test_the_waiting_screen_says_the_queue_too(env):
    settings, repo = env
    worker_alive(settings)
    history_of(repo, REAL_RUNS)
    queued_work(settings, repo, "走っている作業", status="running")
    b, *_ = queued_work(settings, repo, "順番待ちの作業")
    at = ok(app(session={"assignment_id": b}))
    captions = [c.value for c in at.caption]
    assert eta.queue_text(1) in captions
    assert any(c.startswith("順番待ちをふくめて、あと約") for c in captions)
    assert "残り約" in html(at)            # the work's own time stays in the bar


def test_the_waiting_screen_without_history_says_no_estimate_yet(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo, "解析中の作業")
    confirmed_and_analysing(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert eta.EMPTY in [c.value for c in at.caption]


# --- the band ---------------------------------------------------------------------------------

def fake(aid, state, review_step=True, title="作業"):
    """A WorkProgress with only what the band reads."""
    keys = ["assignment", "files", "people", "wait", "review", "pdf"]
    return SimpleNamespace(
        assignment_id=aid, title=title, complete=state == "done", your_turn=state == "ready",
        stopped=state == "stopped", keys=keys,
        turn_step=(5 if review_step else 3) if state == "ready" else None)


def fresh():
    return {"seen": {}, "shown": {}, "dismissed": {}}


def test_a_work_that_was_analysing_and_now_waits_is_announced_once_until_closed():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    marker = lambda aid: "2026-11-14T11:59:00+00:00"          # noqa: E731
    state = fresh()
    assert done_band.announce(state, [fake(1, "running")], marker, now) == []
    assert done_band.announce(state, [fake(1, "ready")], marker, now) == [1]
    assert done_band.announce(state, [fake(1, "ready")], marker, now) == [1]    # stays
    state["dismissed"][1] = state["shown"][1]
    assert done_band.announce(state, [fake(1, "ready")], marker, now) == []     # closed
    assert done_band.announce(state, [fake(1, "running")], marker, now) == []
    newer = lambda aid: "2026-11-14T12:30:00+00:00"            # noqa: E731
    assert done_band.announce(state, [fake(1, "ready")], newer, now) == [1]     # analysed again


def test_a_work_seen_first_is_announced_only_when_it_finished_lately():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    state = fresh()
    late = lambda aid: "2026-11-14T08:00:00+00:00"             # noqa: E731
    old = lambda aid: "2026-11-13T08:00:00+00:00"              # noqa: E731
    assert done_band.announce(state, [fake(1, "ready")], late, now) == [1]
    assert done_band.announce(fresh(), [fake(1, "ready")], old, now) == []
    assert done_band.announce(fresh(), [fake(1, "ready")], lambda a: "", now) == []
    # the coach's turn at an earlier step is not a finished analysis
    assert done_band.announce(fresh(), [fake(1, "ready", review_step=False)], late, now) == []
    # a work that is gone is forgotten
    state = fresh()
    done_band.announce(state, [fake(1, "ready")], late, now)
    assert done_band.announce(state, [], late, now) == [] and state["shown"] == {}


def test_the_home_screen_shows_the_band_for_a_work_that_just_finished(env):
    settings, repo = env
    aid = video_assignment(repo, "県大会")
    reviewable(settings, repo, aid)
    at = ok(app())
    page = html(at)
    assert "「県大会」の解析が終わりました。確認できます" in page
    labels = {b.key: b.label for b in at.button}
    assert labels[f"pfb_band_open_{aid}"] == "確認する" and labels[f"pfb_band_close_{aid}"] == "閉じる"
    assert at.button(key=f"pfb_band_open_{aid}").proto.type == "secondary"
    assert [b.label for b in at.button if b.proto.type == "primary"] == ["続ける"]
    no_internal_terms(at)
    at = ok(at.button(key=f"pfb_band_close_{aid}").click().run())
    assert "の解析が終わりました" not in html(at)
    assert "の解析が終わりました" not in html(ok(at.run()))              # still closed


def test_an_old_finish_is_not_announced_and_opening_the_work_closes_its_band(env):
    settings, repo = env
    old = video_assignment(repo, "先週の試合")
    reviewable(settings, repo, old)
    repo.conn.execute("UPDATE stage_runs SET created_at='2020-01-01T00:00:00+00:00'")
    at = ok(app())
    assert "の解析が終わりました" not in html(at)
    new = video_assignment(repo, "県大会")
    reviewable(settings, repo, new, label="B班")
    at = ok(at.run())
    assert "「県大会」の解析が終わりました" in html(at) and "先週の試合」の解析が" not in html(at)
    at = ok(at.button(key=f"pfb_band_open_{new}").click().run())
    assert at.session_state["wizard"]["assignment_id"] == new
    assert "の解析が終わりました" not in html(at)                    # the open work needs no band
    at.session_state["wizard"] = {"assignment_id": None, "step": None, "submission_id": None}
    at = ok(at.run())
    assert "の解析が終わりました" not in html(at)                    # and it stays closed at home


def test_another_work_that_finished_shows_its_band_on_a_work_screen(env):
    settings, repo = env
    other = video_assignment(repo, "県大会")
    reviewable(settings, repo, other)
    waiting = video_assignment(repo, "解析中の作業")
    confirmed_and_analysing(settings, repo, waiting)
    at = ok(app(session={"assignment_id": waiting}))
    assert "「県大会」の解析が終わりました。確認できます" in html(at)
    assert "「解析中の作業」の解析が終わりました" not in html(at)


def test_a_work_screen_does_not_read_every_work_on_every_rerun(env, monkeypatch):
    settings, repo = env
    other = video_assignment(repo, "県大会")
    reviewable(settings, repo, other)
    waiting = video_assignment(repo, "解析中の作業")
    confirmed_and_analysing(settings, repo, waiting)
    calls, real = [], done_band.work_progress

    def counting(r, aid, *a, **k):
        calls.append(aid)
        return real(r, aid, *a, **k)
    monkeypatch.setattr(done_band, "work_progress", counting)
    clock = [1000.0]
    monkeypatch.setattr(done_band, "_monotonic", lambda: clock[0])
    band = "「県大会」の解析が終わりました。確認できます"
    at = ok(app(session={"assignment_id": waiting}))
    assert calls == [other] and band in html(at)          # the open work is not read again
    for _ in range(3):                                    # a refresh reads nothing, the band stays
        at = ok(at.run())
        assert band in html(at)
    assert calls == [other]
    clock[0] += done_band.CHECK_EVERY + 1                 # after a while it looks again
    at = ok(at.run())
    assert calls == [other, other] and band in html(at)
    # closing the band still works from the remembered state
    at = ok(at.button(key=f"pfb_band_close_{other}").click().run())
    assert band not in html(at)


# --- 作業をさがす -------------------------------------------------------------------------------

def three_works(repo):
    a = repo.create_assignment("県大会 準決勝", "video", "試合", "", "standard", False,
                               domain="sports", sport="basketball")
    group(repo, a, "第1試合")
    b = repo.create_assignment("朝の練習", "video", "練習", "", "standard", False,
                               domain="sports", sport="soccer")
    group(repo, b, "紅白戦")
    c = video_assignment(repo, "ペーパータワー")
    group(repo, c, "A班")
    return a, b, c


def cards(at) -> set[str]:
    return {k for k in (b.key for b in at.button) if k and k.startswith("pfb_open_")}


def test_the_search_row_appears_with_two_works_and_narrows_them(env):
    settings, repo = env
    video_assignment(repo, "ひとつだけ")
    assert not [t for t in ok(app()).text_input if t.key == find.TEXT]      # one work: nothing to narrow
    a, b, c = three_works(repo)
    at = ok(app())
    assert at.text_input(key=find.TEXT).label == "作業をさがす"
    assert len(cards(at)) == 4
    at = ok(at.text_input(key=find.TEXT).input("県大会").run())
    assert cards(at) == {f"pfb_open_{a}"}
    assert "1件の作業が見つかりました（全部で4件）。" in [c.value for c in at.caption]
    at = ok(at.text_input(key=find.TEXT).input("花子").run())                # a person's name
    assert len(cards(at)) == 3 and "pfb_open_1" not in cards(at)    # the three with people
    at = ok(at.text_input(key=find.TEXT).input("該当なし").run())
    assert cards(at) == set() and find.NOTHING in [c.value for c in at.caption]
    at = ok(at.button(key="pfb_find_clear").click().run())
    assert len(cards(at)) == 4 and at.text_input(key=find.TEXT).value == ""


def test_the_sport_and_the_period_choices_narrow_too(env):
    settings, repo = env
    a, b, c = three_works(repo)
    at = ok(app())
    sport = at.selectbox(key=find.SPORT)
    assert list(sport.options) == ["すべて", "バスケットボール", "サッカー", "競技ではない作業"]
    at = ok(sport.select("サッカー").run())
    assert cards(at) == {f"pfb_open_{b}"}
    at = ok(at.selectbox(key=find.SPORT).select("競技ではない作業").run())
    assert f"pfb_open_{c}" in cards(at) and f"pfb_open_{a}" not in cards(at)
    at = ok(at.selectbox(key=find.SPORT).select("すべて").run())
    repo.conn.execute("UPDATE assignments SET created_at='2020-01-01T00:00:00+00:00' WHERE id=?", (a,))
    at = ok(at.selectbox(key=find.PERIOD).select("この7日").run())
    assert f"pfb_open_{a}" not in cards(at) and f"pfb_open_{b}" in cards(at)
    no_internal_terms(at)


def test_a_work_found_can_still_be_opened_and_the_chosen_words_stay(env):
    settings, repo = env
    a, b, c = three_works(repo)
    at = ok(app())
    at = ok(at.text_input(key=find.TEXT).input("朝の").run())
    at = ok(at.button(key=f"pfb_open_{b}").click().run())
    assert at.session_state["wizard"]["assignment_id"] == b
