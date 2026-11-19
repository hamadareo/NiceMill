"""待ち時間の目安 (services.eta): length-aware prices, the queue, the words, and how far the
guesses are from what earlier groups really took (a backtest on synthetic histories).

Only synthetic data: invented works and numbers (the rows of the real-run table in
REAL_RUNS are plain seconds, no names or files)."""
import statistics

import pytest

from pfb_app.domain.models import Stage
from pfb_app.services import deletion, eta
from test_wizard_ui_basic import env, video_assignment  # noqa: F401 (env)
from test_wizard_ui_people_wait import STUDENTS, confirmed_and_analysing, group, job, save

# (video seconds, prepare, track, behavior, contribution) of finished runs of the tracking
# method, as the real installation recorded them (the first, slow track run is the kind of
# outlier the line must not follow); feedback took about 2 s a person
REAL_RUNS = [
    (191.4, 19.6, 412.2, 92.3, 62.9),
    (191.4, 10.7, 110.1, 106.0, 18.9),
    (376.0, 26.6, 97.0, 165.7, 34.1),
    (376.0, 27.7, 129.0, 167.1, 32.0),
    (11.8, 1.5, 17.7, 12.8, 5.8),
    (15.4, 1.6, 16.0, 16.4, 12.5),
    (13.1, 1.2, 17.8, 40.3, 15.7),
]


def finished(repo, aid, label, length, prepare, track, behavior, contribution, feedback=2.0):
    """A group whose steps all finished, with the length of its video on record."""
    sid = group(repo, aid, label)
    save(repo, sid, Stage.prepare, {"frames": [{"video_sec": length}]}, camera="cam1",
         duration=prepare)
    save(repo, sid, Stage.track, {"analysed_s": length}, camera="cam1", duration=track)
    save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", duration=behavior)
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", duration=contribution)
    for subject in repo.list_subjects(sid):
        save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "良い", "evidence_refs": []},
             subject_id=subject["id"], duration=feedback)
    return sid


def history_of(repo, rows, feedback=2.0):
    old = video_assignment(repo, "これまでの作業")
    for n, row in enumerate(rows):
        finished(repo, old, f"班{n}", *row, feedback=feedback)


# --- the line -----------------------------------------------------------------------------------

def test_theil_sen_ignores_an_outlier_and_stays_inside_zero():
    points = [(10, 25.0), (20, 31.0), (30, 37.0), (40, 43.0), (50, 600.0)]   # y = 19 + 0.6 x
    intercept, slope = eta.theil_sen(points)
    assert slope == pytest.approx(0.6) and intercept == pytest.approx(19.0)
    assert eta.theil_sen([(5, 1.0), (5, 2.0)]) is None                      # one length only
    # a falling line would predict less time for a longer video: never; and no negative base
    assert eta.theil_sen([(10, 30.0), (20, 20.0), (30, 10.0)]) == (pytest.approx(20.0), 0.0)
    intercept, slope = eta.theil_sen([(100, 20.0), (200, 60.0), (300, 100.0)])
    assert intercept == 0.0 and slope == pytest.approx(0.3)     # through the origin: median y / x


def test_price_uses_the_line_only_with_enough_different_lengths():
    spread = [(10.0, 20.0), (100.0, 50.0), (200.0, 83.0), (300.0, 116.0)]   # about 17 + 0.33 x
    assert eta.price(spread, 150.0) == pytest.approx(17 + 0.33 * 150, abs=2.0)
    assert eta.price(spread, None) == statistics.median(y for _, y in spread)   # length unknown
    assert eta.price(spread[:2], 150.0) == statistics.median([20.0, 50.0])      # two runs only
    same = [(100.0, 50.0), (100.0, 60.0), (100.0, 55.0)]                       # one length only
    assert eta.price(same, 400.0) == 55.0
    assert eta.price([(None, 5.0), (None, 7.0), (None, 9.0)], 100.0) == 7.0     # no lengths
    assert eta.price([], 100.0) is None
    assert eta.price([(10.0, 5.0), (400.0, 6.0), (800.0, 7.0)], 1.0) >= 1.0     # never below 1 s


# --- how long the video is ------------------------------------------------------------------------

def test_group_lengths_prefer_the_track_then_the_frames_then_the_check(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, "A班")
    assert eta.group_lengths(repo, [sid]) == {}
    assert eta.length_of({}, sid, "cam1") is None
    # the content check: the whole video 600 s, the group's window 60 s to 360 s
    repo.update_submission_meta(sid, {**repo.get_submission(sid)["meta"],
                                      "start_sec": 60.0, "end_sec": 360.0})
    save(repo, sid, Stage.detect, {"duration_s": 600.0})
    lengths = eta.group_lengths(repo, [sid])
    assert lengths == {(sid, None): 300.0}
    assert eta.length_of(lengths, sid, "cam1") == 300.0                     # the group's own
    save(repo, sid, Stage.prepare, {"frames": [{"video_sec": 0.0}, {"video_sec": 295.0}]},
         camera="cam1")
    lengths = eta.group_lengths(repo, [sid])
    assert eta.length_of(lengths, sid, "cam1") == 295.0                     # the prepared frames
    save(repo, sid, Stage.track, {"analysed_s": 301.5}, camera="cam1")
    lengths = eta.group_lengths(repo, [sid])
    assert eta.length_of(lengths, sid, "cam1") == 301.5                     # the track decides
    assert eta.length_of(lengths, sid, "cam2") == 300.0                     # an unknown camera


def test_the_history_follows_new_runs_without_clearing_anything(env):
    settings, repo = env
    history_of(repo, REAL_RUNS[:3])
    first = eta.history(repo, "pfb3x3")
    assert len(first["track"]) == 3
    history_of(repo, REAL_RUNS[3:5])
    assert len(eta.history(repo, "pfb3x3")["track"]) == 5
    assert all(x for x, _ in eta.history(repo, "pfb3x3")["track"])           # every length known
    assert [x for x, _ in eta.history(repo, "pfb3x3")["pfb"]] == [None] * 15   # feedback: per person


# --- the time left ----------------------------------------------------------------------------------

def analysing_group(settings, repo, length=None):
    aid = video_assignment(repo, "いまの作業")
    sid = confirmed_and_analysing(settings, repo, aid)     # behavior (cam1) queued
    if length is not None:
        save(repo, sid, Stage.detect, {"duration_s": length})
    return aid, sid


def test_no_history_no_estimate(env):
    settings, repo = env
    aid, _ = analysing_group(settings, repo, 200.0)
    assert eta.remaining_sec(repo, aid) is None
    assert eta.lines(eta.estimate(repo, aid)) == [eta.EMPTY]
    assert "実績" in eta.EMPTY and "目安" in eta.EMPTY


def test_a_short_video_is_not_priced_like_the_long_ones_before_it(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    short, _ = analysing_group(settings, repo, 12.0)
    long, _ = analysing_group(settings, repo, 376.0)
    quick, slow = eta.remaining_sec(repo, short), eta.remaining_sec(repo, long)
    # behavior + contribution + three people's feedback, by the line through the 7 runs
    assert quick < 40 and slow > 150 and slow > 5 * quick
    # without the length (a group registered without its check) it is the plain median again
    aid, sid = analysing_group(settings, repo)
    median = (statistics.median(r[3] for r in REAL_RUNS) + statistics.median(r[4] for r in REAL_RUNS)
              + 3 * 2.0)
    assert eta.remaining_sec(repo, aid) == pytest.approx(median)


def add_tips(repo, seconds):
    """A tips run per person of every group of the earlier work (the hints the worker makes
    right after each person's feedback, as its own stage run)."""
    for sub in repo.list_submissions(repo.list_assignments()[0]["id"]):
        for subject in repo.list_subjects(sub["id"]):
            save(repo, sub["id"], Stage.tips, {"tips": ["もう一歩"]}, subject_id=subject["id"],
                 duration=seconds)


def test_the_hints_made_after_each_feedback_are_part_of_its_time(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    aid, _ = analysing_group(settings, repo)           # no length: the plain medians
    median = (statistics.median(r[3] for r in REAL_RUNS) + statistics.median(r[4] for r in REAL_RUNS))
    assert eta.remaining_sec(repo, aid) == pytest.approx(median + 3 * 2.0)   # no hints on record
    add_tips(repo, 3.0)
    assert eta.remaining_sec(repo, aid) == pytest.approx(median + 3 * (2.0 + 3.0))
    # the card's estimate before the work starts follows the same price
    assert eta.price_of(eta.history(repo, "pfb3x3"), "pfb", None) == pytest.approx(5.0)
    assert eta.price_of(eta.history(repo, "pfb3x3"), "behavior", None) is not None
    assert eta.price_of(eta.history(repo, "pfb3x3"), "nothing", None) is None


def test_the_hints_estimate_is_never_negative_or_not_a_number(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    add_tips(repo, 1.0)
    for x in (None, 0.0, 5.0, 12.0, 600.0):
        aid, _ = analysing_group(settings, repo, x)
        left = eta.remaining_sec(repo, aid)
        assert left is not None and left == left and left >= 0


def test_running_progress_takes_off_its_part_of_the_price(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    aid, sid = analysing_group(settings, repo, 200.0)
    before = eta.remaining_sec(repo, aid)
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=? AND stage='behavior'",
                            (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    repo.set_job_progress(jid, 0.5, "半分")
    behavior = eta.price([(x, y) for x, y in eta.history(repo, "pfb3x3")["behavior"]], 200.0)
    assert eta.remaining_sec(repo, aid) == pytest.approx(before - 0.5 * behavior)
    repo.set_job_progress(jid, 1.0, None)
    assert eta.remaining_sec(repo, aid) == pytest.approx(before - behavior)


def test_remaining_jobs_are_what_the_work_still_needs(env):
    settings, repo = env
    aid, sid = analysing_group(settings, repo, 200.0)
    assert [(s.stage, s.camera) for s in eta.remaining_slots(repo, aid, through=True)] == [
        ("behavior", "cam1"), ("contribution", "cam1")] + [("pfb", None)] * len(STUDENTS[:3])
    assert sum(eta.remaining_jobs(repo, aid, through=True).values()) == 5


# --- the queue --------------------------------------------------------------------------------------

def queued_work(settings, repo, title, status="queued"):
    aid = video_assignment(repo, title)
    sid = group(repo, aid, "班")
    jid = job(repo, sid, Stage.prepare, status=status, camera="cam1", model="pfb3x3:x")
    return aid, sid, jid


def test_works_in_front_and_what_they_still_need(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    a, *_ = queued_work(settings, repo, "走っている作業", status="running")
    b, *_ = queued_work(settings, repo, "順番待ちの作業")
    c, *_ = queued_work(settings, repo, "その次の作業")
    assert eta.queue_ahead(repo, a) == ([], False)          # it runs: nobody is in front
    assert eta.queue_ahead(repo, b) == ([a], False)
    assert eta.queue_ahead(repo, c) == ([a, b], False)
    est = eta.estimate(repo, c)
    assert est.ahead == 2 and est.ahead_ids == (a, b)
    assert est.seconds is not None and est.wait_seconds is not None
    assert est.total == pytest.approx(est.seconds + est.wait_seconds)
    assert est.wait_seconds == pytest.approx(eta.remaining_sec(repo, a) + eta.remaining_sec(repo, b))
    assert eta.estimate(repo, a).ahead == 0 and eta.estimate(repo, a).total == eta.estimate(repo, a).seconds


def test_a_step_waiting_for_room_is_said_and_a_work_to_be_deleted_is_not_counted(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    a, *_ = queued_work(settings, repo, "走っている作業", status="running")
    b, *_ = queued_work(settings, repo, "待っている作業", status="waiting_gpu")
    assert eta.queue_ahead(repo, b) == ([a], True)
    assert eta.HELD in eta.lines(eta.estimate(repo, b))
    with pytest.raises(deletion.StillRunning):     # its running step ends first
        deletion.delete_work(repo, settings, a, stop_analysis=True)
    assert deletion.pending_deletions(repo) == {a}
    assert eta.queue_ahead(repo, b) == ([], True)


def test_the_wait_is_unknown_when_a_work_in_front_has_no_estimate(env):
    settings, repo = env
    # history for the later stages only: the work in front still needs a prepare step
    old = video_assignment(repo, "これまで")
    sid = group(repo, old, "班")
    save(repo, sid, Stage.track, {"analysed_s": 100.0}, camera="cam1", duration=50)
    save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", duration=60)
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", duration=20)
    save(repo, sid, Stage.pfb, {"pfb_status": "generated"}, subject_id=repo.list_subjects(sid)[0]["id"],
         duration=3)
    a, *_ = queued_work(settings, repo, "前の作業", status="running")
    own, _ = analysing_group(settings, repo, 100.0)     # its job is younger than a's
    est = eta.estimate(repo, own)
    assert est.seconds is not None
    assert est.ahead == 1 and est.wait_seconds is None and est.total is None
    text = eta.lines(est)
    assert text[0] == eta.queue_text(1) and "順番待ちの時間は含みません" in text[1]


# --- the words ----------------------------------------------------------------------------------------

def test_the_words_are_plain_and_say_it_is_a_guess():
    assert eta.rounded(30) == "約1分" and eta.rounded(11 * 60) == "約15分"
    assert eta.rounded(61 * 60) == "約1時間10分" and eta.rounded(120 * 60) == "約2時間"
    assert eta.time_left_text(11 * 60) == "あと約15分（目安）"
    assert eta.time_left_text(11 * 60, waiting=True) == \
        "順番待ちをふくめて、あと約15分かかるかもしれません（目安）"
    assert eta.lines(eta.Estimate(600.0)) == ["あと約10分（目安）"]
    assert eta.lines(eta.Estimate(None)) == [eta.EMPTY]
    assert eta.lines(eta.Estimate(600.0, ahead=2, wait_seconds=1200.0)) == [
        "前に2件の作業があります。順番が来ると始まります。",
        "順番待ちをふくめて、あと約30分かかるかもしれません（目安）"]
    assert eta.lines(eta.Estimate(None, ahead=1)) == [eta.queue_text(1), eta.EMPTY]
    for text in (eta.EMPTY, eta.HELD, eta.queue_text(3)):
        for word in ("GPU", "ジョブ", "job", "engine", "キュー", "SQL", "/", "http"):
            assert word not in text


# --- how far the guesses are -----------------------------------------------------------------------

def test_backtest_on_the_real_runs_the_length_beats_the_median(env):
    settings, repo = env
    history_of(repo, REAL_RUNS)
    checks = eta.backtest(repo, "pfb3x3")
    assert len(checks) == len(REAL_RUNS)
    by_length = statistics.median(c.error_length for c in checks)
    by_median = statistics.median(c.error_median for c in checks)
    assert by_length < by_median
    assert by_length < 0.5                                   # a guess within half, at the median
    short = [c for c in checks if c.actual < 120]            # the three short clips
    assert len(short) == 3
    # the plain median prices a 12 s clip like the 190-376 s ones; the line does not
    assert statistics.median(c.error_length for c in short) < 0.5 * statistics.median(
        c.error_median for c in short)


def test_backtest_on_a_regular_history_is_close(env):
    settings, repo = env
    lengths = [20, 45, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600]
    wobble = [1.05, 0.93, 1.1, 0.97, 1.0, 1.08, 0.92, 1.04, 0.96, 1.02, 0.99, 1.06, 0.94, 1.01]
    rows = [(x, 5 + 0.05 * x, (14 + 0.30 * x) * w, (8 + 0.45 * x) * w, (4 + 0.12 * x) * w)
            for x, w in zip(lengths, wobble)]
    rows[5] = (rows[5][0], rows[5][1], rows[5][2] * 6, rows[5][3], rows[5][4])   # a disturbed run
    history_of(repo, rows)
    checks = eta.backtest(repo, "pfb3x3")
    assert len(checks) == len(rows)
    errors = sorted(c.error_length for c in checks)
    assert statistics.median(errors) < 0.12
    assert errors[int(0.9 * len(errors))] < 0.35             # nine in ten within about a third
    assert statistics.median(errors) < 0.5 * statistics.median(c.error_median for c in checks)


def test_the_estimate_of_a_new_work_matches_what_it_then_takes(env):
    """Predict a work from an earlier history, then record its real run times and compare."""
    settings, repo = env
    lengths = [30, 60, 90, 150, 240, 300, 420, 540]
    history_of(repo, [(x, 4 + 0.05 * x, 12 + 0.30 * x, 8 + 0.45 * x, 4 + 0.12 * x)
                      for x in lengths])
    for x in (75.0, 200.0, 480.0):
        aid, sid = analysing_group(settings, repo, x)
        guess = eta.remaining_sec(repo, aid)
        truth = (8 + 0.45 * x) + (4 + 0.12 * x) + 3 * 2.0
        assert guess == pytest.approx(truth, rel=0.15)


# --- the content check card ---------------------------------------------------------------------

def test_window_is_the_part_of_the_video_that_is_analysed():
    assert eta.window(600.0, 60.0, 360.0) == 300.0
    assert eta.window(600.0, 0.0, None) == 600.0
    assert eta.window(600.0, None, 900.0) == 600.0          # an end past the video's end
    assert eta.window(600.0, 700.0, None) is None           # a start after the end
    assert eta.window(None, 0.0, 60.0) is None and eta.window(0.0, 0.0, None) is None


def test_the_check_cards_estimate_follows_the_part_chosen(env):
    from pfb_app.services.content_check import estimate_for
    settings, repo = env
    assert estimate_for(repo, "pfb3x3", 1, 3, 100.0) is None            # no history yet
    history_of(repo, REAL_RUNS)
    short = estimate_for(repo, "pfb3x3", 1, 3, 12.0)
    long = estimate_for(repo, "pfb3x3", 1, 3, 376.0)
    assert short < 60 and long > 250 and long > 5 * short
    # the length unknown (or one camera more): the median, as it was before
    plain = estimate_for(repo, "pfb3x3", 1, 3)
    medians = [statistics.median(r[i] for r in REAL_RUNS) for i in (1, 2, 3, 4)]
    assert plain == pytest.approx(sum(medians) + 3 * 2.0)
    assert estimate_for(repo, "pfb3x3", 2, 3) == pytest.approx(2 * sum(medians) + 3 * 2.0)
