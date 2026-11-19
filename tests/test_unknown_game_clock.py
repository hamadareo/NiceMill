"""A clip cut from the middle of a game (the coach left 「この動画の始まりの試合時間」 blank) has
no game clock: the feedback must not say 「試合開始直後」 or 「試合時間N分ごろ」 for it
(observed 2026-11-13 on free clips of 12-15 s). The time phrase comes from real evidence only:
the work's clock_offset_min (a typed 0 counts) plus the event's time; else it is left out.
Synthetic data only."""
import json
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import BehaviorEvent, Kind, Stage
from pfb_app.pipeline import checks
from pfb_app.pipeline.pfb3x3_bridge import bridge_behavior, game_clock_offset_s
from pfb_app.pipeline.prompts import load_prompt
from pfb_app.pipeline.video import _event_lines, run_contribution, run_pfb
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.common import cited_evidence
from test_pipeline_video import ctx_for, ev, judg, save, sports_events_run
from test_pfb3x3_bridge import ev as bridge_ev, row

PHRASES = ("試合開始直後", "試合時間0分ごろ")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def make_env(tmp_path, settings, meta, db="clock.db"):
    conn = connect(tmp_path / db)
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("練習試合", Kind.video, "春季リーグ第2戦", "", "standard", False,
                                 domain="sports", sport="soccer")
    sid = repo.create_submission(aid, "G1", meta)
    return repo, sid, {"Pa": repo.add_subject(sid, "Pa", "#5", "", {})}


def event(task_elapsed="00:05"):
    return BehaviorEvent(event_id="cam1-0001", camera="cam1", participant="Pa",
                         video_time="00:42", task_elapsed=task_elapsed,
                         observed_behavior="味方にパスを出す", utterance="", speaker="unknown",
                         evidence_source="video", confidence="high")


# --- the clock offset: only a value the coach gave is a game clock ---------------------------

def test_a_blank_offset_is_no_game_clock():
    assert game_clock_offset_s({}, "sports") is None
    assert game_clock_offset_s({"clock_offset_min": None, "start_sec": 30.0}, "sports") is None
    assert game_clock_offset_s({"start_sec": 30.0}, "sports") is None


def test_a_typed_zero_is_a_game_clock():
    assert game_clock_offset_s({"clock_offset_min": 0.0}, "sports") == 0.0
    assert game_clock_offset_s({"clock_offset_min": 0, "start_sec": 5.0}, "sports") == 5.0
    assert game_clock_offset_s({"clock_offset_min": 45, "start_sec": 600.0}, "sports") == 3300.0
    assert game_clock_offset_s({"clock_offset_min": 45}, "school") is None


def test_events_without_a_clock_keep_the_time_since_the_analysed_start(tmp_path):
    out = bridge_behavior([row(0, [bridge_ev("P1", "0:30")])], {"P1": "Pa"}, "cam1", 12.0, False,
                          tmp_path, clock_offset_s=game_clock_offset_s({"start_sec": 12.0},
                                                                       "sports"))
    assert out["events"][0]["task_elapsed"] == "00:30"


# --- what the model is told ---------------------------------------------------------------

def test_the_event_line_says_the_game_time_is_unknown_when_it_is():
    lines = _event_lines([event("00:05")], "sports", clock_known=False)
    assert "動画の時間: 00:42（約1分ごろ）" in lines       # the video's own time, not a game's
    assert "試合時間" not in lines and "00:05" not in lines
    known = _event_lines([event("50:30")], "sports", clock_known=True)
    assert "試合時間: 50:30（51分ごろ）" in known
    assert "試合時間: 50:30（51分ごろ）" in _event_lines([event("50:30")], "sports")   # default
    assert "不明" not in _event_lines([event("00:05")], "school", clock_known=False)


def test_the_prompts_tell_what_to_write_for_each_case(settings):
    pfb = load_prompt(settings, Kind.video, Stage.pfb, domain="sports").text
    assert "動画の時間" in pfb
    assert "この場面で" in pfb
    for word in ("試合開始直後", "序盤", "終盤", "前半", "後半"):
        assert word in pfb          # named as what not to write when the time is unknown
    # the 0-minute rule is kept for a game clock that is known
    assert "N分ごろ" in pfb and "試合開始直後、」と書く" in pfb
    contribution = load_prompt(settings, Kind.video, Stage.contribution, domain="sports").text
    assert "動画の時間" in contribution


@pytest.mark.parametrize("meta,known", [({}, False), ({"clock_offset_min": None}, False),
                                        ({"clock_offset_min": 0.0}, True),
                                        ({"clock_offset_min": 12.0}, True)])
def test_the_stages_send_the_clock_only_when_it_is_known(tmp_path, settings, meta, known):
    repo, sid, subj = make_env(tmp_path, settings, meta)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    sports_events_run(repo, sid, "cam1", [ev("00:42", "Pa", "味方にパスを出す")])
    ctx, backend = ctx_for(settings, repo, lambda r: judg(["cam1-0001"], {"cam1-0001"}))
    assert run_contribution(ctx, sid, "cam1").status == "ok"
    text = backend.calls[0].user_text
    assert ("試合時間: 10:05（10分ごろ）" in text) is known
    assert ("動画の時間:" in text) is (not known)
    ctx, backend = ctx_for(settings, repo, lambda r: json.dumps(
        {"pfb_status": "generated", "pfb": "この場面で、パスを出していました。",
         "evidence_refs": ["cam1-0001"]}, ensure_ascii=False))
    assert run_pfb(ctx, sid, subj["Pa"]).status == "ok"
    text = backend.calls[0].user_text
    assert ("試合時間: 10:05（10分ごろ）" in text) is known
    assert ("動画の時間:" in text) is (not known)
    assert "10:05" not in text or known


# --- the check on what came back -------------------------------------------------------------

@pytest.mark.parametrize("text", ["試合開始直後、パスを出していました。", "試合時間0分ごろ、パスを出しました。",
                                  "試合の序盤に声をかけていました。", "後半、ボールを運びました。",
                                  "試合時間１２分ごろ、守っていました。", "終盤でも走っていました。",
                                  "試合終了間際に声を出していました。"])
def test_a_time_phrase_in_a_text_without_a_clock_is_warned(text):
    warnings = checks.pfb_text_warnings(text, set(), set(), clock_known=False)
    assert any("時間" in w and "不明" in w for w in warnings), warnings
    # with a known clock the same text is the model's honest reading of the clock
    assert checks.pfb_text_warnings(text, set(), set(), clock_known=True) == []


@pytest.mark.parametrize("text", ["この場面で、パスを出していました。",
                                  "味方へ声をかけて、守りを整えていました。"])
def test_a_neutral_text_without_a_clock_has_no_warning(text):
    assert checks.pfb_text_warnings(text, set(), set(), clock_known=False) == []


def test_the_default_changes_nothing():
    assert checks.pfb_text_warnings("試合開始直後、走りました。", set(), set()) == []


def test_run_pfb_warns_when_the_model_invents_a_time(tmp_path, settings):
    repo, sid, subj = make_env(tmp_path, settings, {})
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    sports_events_run(repo, sid, "cam1", [ev("00:42", "Pa", "味方にパスを出す")])
    ctx, _ = ctx_for(settings, repo, lambda r: judg(["cam1-0001"], {"cam1-0001"}))
    run_contribution(ctx, sid, "cam1")
    ctx, _ = ctx_for(settings, repo, lambda r: json.dumps(
        {"pfb_status": "generated", "pfb": "試合開始直後、パスを出していました。",
         "evidence_refs": ["cam1-0001"]}, ensure_ascii=False))
    run = run_pfb(ctx, sid, subj["Pa"])
    assert run.status == "ok"
    assert any("不明" in w for w in run.warnings), run.warnings


def test_the_review_path_warns_too(tmp_path, settings):
    from pfb_app.services import review
    repo, sid, subj = make_env(tmp_path, settings, {})
    assert any("不明" in w for w in review._text_warnings(
        repo, sid, set(), "試合開始直後、パスを出していました。"))
    repo2, sid2, _ = make_env(tmp_path, settings, {"clock_offset_min": 0.0}, db="known.db")
    assert review._text_warnings(repo2, sid2, set(), "試合開始直後、パスを出していました。") == []


# --- what the coach reads next to the text ---------------------------------------------------

@pytest.mark.parametrize("meta,when", [({"clock_offset_min": 0.0}, "試合時間 10:05"),
                                       ({}, "00:42")])
def test_cited_evidence_names_the_game_time_only_when_it_is_one(tmp_path, settings, meta, when):
    repo, sid, subj = make_env(tmp_path, settings, meta)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    beh = sports_events_run(repo, sid, "cam1", [ev("00:42", "Pa", "味方にパスを出す")])
    con = save(repo, sid, Stage.contribution, {"judgments": json.loads(
        judg(["cam1-0001"], {"cam1-0001"}))["judgments"]}, camera="cam1", upstream=[beh.id])
    pfb = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "x",
                                      "evidence_refs": ["cam1-0001"]},
               subject_id=subj["Pa"], upstream=[con.id])
    assert [i.when for i in cited_evidence(repo, pfb)] == [when]
