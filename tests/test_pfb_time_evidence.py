"""Long-video findings (2026-11-14): the feedback's time words must agree with the evidence it
cites (「N分ごろ」 among the cited events' minutes; 「開始直後」 only for a 0-minute event of a
known game clock; 「試合時間」 only with a known clock, else 「動画の約N分ごろ」), the evidence
list stays at three plays at most, runs of the same recorded action count once, and the prompts
bound the interpretation. Synthetic data only."""
import json
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import BehaviorEvent, Kind, Stage
from pfb_app.pipeline import checks
from pfb_app.pipeline.pfb3x3_bridge import clock_is_known
from pfb_app.pipeline.prompts import load_prompt
from pfb_app.pipeline.video import _event_lines, _fold_runs, run_contribution, run_pfb
from test_pipeline_video import ctx_for, judg, save
from test_unknown_game_clock import make_env


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def ev(eid, video, elapsed, text="相手を防ぐために動く", cam="cam1"):
    return BehaviorEvent(event_id=eid, camera=cam, participant="Pa", video_time=video,
                         task_elapsed=elapsed, observed_behavior=text, utterance="",
                         speaker="unknown", evidence_source="video", confidence="high")


# --- the pure checks -------------------------------------------------------------------------

def test_a_minute_must_be_one_of_the_cited_events_minutes():
    ok = checks.time_errors("試合時間1分ごろ、守りました。試合時間3分ごろ、走りました。", {1, 3}, True)
    assert ok == []
    bad = checks.time_errors("試合時間2分ごろ、守りました。", {1, 3}, True)
    assert len(bad) == 1 and "2分ごろ" in bad[0] and "1" in bad[0] and "3" in bad[0]
    assert checks.time_errors("試合時間２分ごろ、守りました。", {1, 3}, True)     # full width
    assert checks.time_errors("2分ごろ、守りました。", {1, 3}, True)              # bare too


def test_a_start_phrase_needs_a_zero_minute_event_of_a_known_clock():
    assert checks.time_errors("試合開始直後、走りました。", {0, 5}, True) == []
    assert checks.time_errors("試合開始直後、走りました。", {1, 5}, True)
    assert checks.time_errors("開始直後に走りました。", {1}, True)             # 8:08 -> 開始直後
    assert checks.time_errors("試合開始直後、走りました。", {0}, False)        # no game clock


def test_without_a_known_clock_a_zero_minute_is_not_written():
    assert checks.time_errors("動画の約0分ごろ、走りました。", {0}, False)
    assert checks.time_errors("動画の約13分ごろ、走りました。", {13}, False) == []


def test_text_without_a_time_has_no_error():
    assert checks.time_errors("この場面で、守りました。", {1}, True) == []
    assert checks.time_errors("この場面で、守りました。", set(), False) == []


@pytest.mark.parametrize("known,text,want", [
    (True, "13分ごろ、守りました。", "試合時間13分ごろ、守りました。"),
    (True, "動画の約13分ごろ、守りました。", "試合時間13分ごろ、守りました。"),
    (True, "約13分ごろ、守りました。", "試合時間13分ごろ、守りました。"),
    (True, "試合時間13分ごろ、守りました。", "試合時間13分ごろ、守りました。"),
    (False, "試合時間13分ごろ、守りました。", "動画の約13分ごろ、守りました。"),
    (False, "13分ごろ、守りました。", "動画の約13分ごろ、守りました。"),
    (False, "動画の13分ごろ、守りました。", "動画の約13分ごろ、守りました。"),
    (True, "試合時間１３分ごろ、守りました。", "試合時間13分ごろ、守りました。"),
    (True, "この場面で、守りました。", "この場面で、守りました。"),
])
def test_time_mentions_are_written_one_way(known, text, want):
    assert checks.normalize_time_words(text, known) == want


def test_a_mixed_text_is_made_uniform():
    text = "試合時間1分ごろ、守り、3分ごろ、走りました。"
    assert checks.normalize_time_words(text, True) == "試合時間1分ごろ、守り、試合時間3分ごろ、走りました。"


def test_evidence_is_cut_to_three_keeping_the_plays_the_text_names():
    minutes = {"a": 1, "b": 1, "c": 2, "d": 3, "e": 3, "f": 4}
    refs = list(minutes)
    kept = checks.trim_refs(refs, "試合時間1分ごろ、守り、試合時間4分ごろ、走りました。", minutes)
    assert len(kept) == 3 and "a" in kept and "f" in kept       # a minute named stays covered
    assert kept == [r for r in refs if r in kept]                 # the model's order
    assert checks.trim_refs(["a", "c"], "x", minutes) == ["a", "c"]
    assert checks.trim_refs(["a", "a", "c"], "x", minutes) == ["a", "c"]    # no repeats
    zero = {"a": 5, "b": 0, "c": 6, "d": 7}
    kept = checks.trim_refs(list(zero), "試合開始直後、守りました。", zero)
    assert "b" in kept and len(kept) == 3


# --- runs of the same recorded action ------------------------------------------------------------

def test_consecutive_same_actions_count_once():
    events = [ev(f"e{n}", f"13:{7 + n // 5:02d}", "x") for n in range(15)]        # 5 per second
    events += [ev("z", "20:00", "x", text="パスを出す")]
    folded = _fold_runs(events)
    assert [(e.event_id, n) for e, n in folded] == [("e0", 15), ("z", 1)]


def test_a_gap_a_different_action_or_another_camera_starts_a_new_run():
    a = ev("a", "13:07", "x")
    far = ev("b", "13:40", "x")                     # 33 s later
    other = ev("c", "13:41", "x", text="パスを出す")
    cam2 = ev("d", "13:41", "x", cam="cam2")
    folded = _fold_runs([a, far, other, cam2])
    assert [e.event_id for e, _ in folded] == ["a", "b", "c", "d"]
    again = ev("e", "13:09", "x")
    assert [(e.event_id, n) for e, n in _fold_runs([a, again])] == [("a", 2)]


def test_the_same_action_with_other_spacing_or_width_is_the_same_action():
    a = ev("a", "13:07", "x", text="相手を防ぐために動く")
    b = ev("b", "13:08", "x", text="相手を防ぐために動く。")
    c = ev("c", "13:09", "x", text="相手を 防ぐために動く")
    assert [(e.event_id, n) for e, n in _fold_runs([a, b, c])] == [("a", 3)]


# --- the event lines -----------------------------------------------------------------------------

def test_event_lines_give_the_video_time_without_a_game_clock():
    lines = _event_lines([ev("a", "13:07", "00:12")], "sports", clock_known=False)
    assert "動画の時間: 13:07（約13分ごろ）" in lines and "試合時間" not in lines
    short = _event_lines([ev("a", "00:12", "00:12")], "sports", clock_known=False)
    assert "動画の時間: 00:12（1分未満）" in short and "分ごろ" not in short
    known = _event_lines([ev("a", "13:07", "50:30")], "sports", clock_known=True)
    assert "試合時間: 50:30（51分ごろ）" in known and "動画の時間" not in known


# --- through the stage ----------------------------------------------------------------------------

def pfb_stage(tmp_path, settings, meta, events, answers):
    """Run contribution (all positive) and then pfb; `answers` is a list of the model's pfb
    answers in turn (the last one repeats). Returns (run, backend)."""
    repo, sid, subj = make_env(tmp_path, settings, meta)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    beh = save(repo, sid, Stage.behavior, {"events": [e.model_dump() for e in events]},
               camera="cam1")
    ids = [e.event_id for e in events]
    con = save(repo, sid, Stage.contribution, {"judgments": json.loads(judg(ids, set(ids)))[
        "judgments"]}, camera="cam1", upstream=[beh.id])
    turn = iter(answers)
    last = {}

    def answer(_r):
        last["a"] = next(turn, last.get("a", answers[-1]))
        return json.dumps(last["a"], ensure_ascii=False)

    ctx, backend = ctx_for(settings, repo, answer)
    return run_pfb(ctx, sid, subj["Pa"]), backend


def say(text, refs):
    return {"pfb_status": "generated", "pfb": text, "evidence_refs": refs}


def test_a_wrong_start_phrase_is_asked_again_and_the_fix_is_kept(tmp_path, settings):
    events = [ev("e1", "01:10", "01:10", "声をかける"), ev("e2", "03:20", "03:20", "パスを出す")]
    run, backend = pfb_stage(
        tmp_path, settings, {"clock_offset_min": 0.0}, events,
        [say("試合開始直後、声をかけ、試合時間3分ごろ、パスを出していました。", ["e1", "e2"]),
         say("試合時間1分ごろ、声をかけ、試合時間3分ごろ、パスを出していました。", ["e1", "e2"])])
    assert run.status == "ok" and len(backend.calls) == 2
    assert "開始直後" in backend.calls[1].user_text and "エラー" in backend.calls[1].user_text
    assert run.output["pfb"].startswith("試合時間1分ごろ、") and run.warnings == []
    assert run.output["evidence_refs"] == ["e1", "e2"]


def test_a_text_that_stays_wrong_is_kept_with_a_warning_not_failed(tmp_path, settings):
    events = [ev("e1", "01:10", "01:10", "声をかける"), ev("e2", "03:20", "03:20", "パスを出す")]
    run, backend = pfb_stage(tmp_path, settings, {"clock_offset_min": 0.0}, events,
                             [say("試合時間2分ごろ、声をかけていました。", ["e1"])])
    assert run.status == "ok" and run.output["pfb_status"] == "generated"
    assert len(backend.calls) == 3                       # the first and two retries
    assert any("2分ごろ" in w for w in run.warnings), run.warnings


def test_an_unknown_clock_says_the_video_minute(tmp_path, settings):
    events = [ev("e1", "13:07", "00:12", "声をかける")]
    run, backend = pfb_stage(tmp_path, settings, {}, events,
                             [say("試合時間13分ごろ、声をかけていました。", ["e1"])])
    assert run.status == "ok" and len(backend.calls) == 1
    assert run.output["pfb"] == "動画の約13分ごろ、声をかけていました。"
    assert run.warnings == []
    assert "動画の時間: 13:07（約13分ごろ）" in backend.calls[0].user_text


def test_an_unknown_clock_never_gets_a_start_phrase(tmp_path, settings):
    events = [ev("e1", "00:12", "00:12", "声をかける")]
    run, backend = pfb_stage(tmp_path, settings, {}, events,
                             [say("試合開始直後、声をかけていました。", ["e1"]),
                              say("この場面で、声をかけていました。", ["e1"])])
    assert len(backend.calls) == 2 and run.output["pfb"].startswith("この場面で")
    assert run.warnings == []


def test_a_run_of_one_action_is_one_line_and_the_evidence_is_three_at_most(tmp_path, settings):
    run_events = [ev(f"r{n}", f"13:{7 + n // 5:02d}", f"13:{7 + n // 5:02d}") for n in range(13)]
    others = [ev("p1", "14:30", "14:30", "パスを出す"), ev("p2", "16:10", "16:10", "声をかける"),
              ev("p3", "18:00", "18:00", "ボールを運ぶ")]
    refs = [e.event_id for e in run_events[:6]] + ["p1", "p2", "p3"]
    run, backend = pfb_stage(
        tmp_path, settings, {"clock_offset_min": 0.0}, run_events + others,
        [say("試合時間13分ごろ、守り、試合時間18分ごろ、ボールを運んでいました。", refs)])
    text = backend.calls[0].user_text
    assert text.count("ref_id:") == 4                    # the run counts once
    assert "13:07" in text and "r0" in text and "r1\n" not in text
    assert run.status == "ok"
    kept = run.output["evidence_refs"]
    assert len(kept) <= 3 and "p3" in kept and any(r.startswith("r") for r in kept)


def test_the_prompts_bound_the_interpretation_and_the_wording(settings):
    pfb = load_prompt(settings, Kind.video, Stage.pfb, domain="sports").text
    assert "動画の約N分ごろ" in pfb and "1分未満" in pfb
    assert "行動イベントに書かれていない結果・効果" in pfb
    assert "可能性を高め" in pfb                          # named as an example of too much
    assert "実行" in pfb and "打って" in pfb              # 「シュートを実行した」 reworded
    contribution = load_prompt(settings, Kind.video, Stage.contribution, domain="sports").text
    assert "可能性を高め" in contribution and "行動の文に書かれた内容" in contribution


def test_normalising_time_words_leaves_the_rest_of_the_sentence_alone():
    text = "試合時間10分ごろ、パスを出しました！（すばらしい）…次も頑張りましょう～"
    assert checks.normalize_time_words(text, True) == text
    assert checks.normalize_time_words("１０分ごろ、パス！（すばらしい）", True) == \
        "試合時間10分ごろ、パス！（すばらしい）"
