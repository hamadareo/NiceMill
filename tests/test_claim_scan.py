"""動作の再確認 (pipeline.motion_check, pipeline.claim_scan, the claim_checks table) and how the
automatic approval reads it. Synthetic tracks and a fake vision model: no video, no GPU."""
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from pfb_app.domain.models import Stage
from pfb_app.export.tracks import BoxIndex
from pfb_app.pipeline import claim_check as cc
from pfb_app.pipeline import claim_scan as cs
from pfb_app.pipeline import motion_check as mc
from pfb_app.services import autoapprove, review
from pfb_app.storage import db
from test_autoapprove import env, make, stubbed  # noqa: F401 (fixtures)
from test_wizard_ui_review_pdf import subject

FPS = 10.0


def index(person_dx=0.0, camera_dx=0.0, others=5, seconds=6):
    """A person (P1) and `others` people in boxes 100 px high; the camera pans camera_dx px a
    second (everyone shifts), the person also walks person_dx px a second."""
    rows = {}
    for f in range(1, int(seconds * FPS) + 1):
        t = f / FPS
        rows.setdefault("P1", []).append((f, 200 + (camera_dx + person_dx) * t, 300.0, 40.0, 100.0))
        for k in range(others):
            rows.setdefault(f"P{k + 2}", []).append((f, 100 + 90 * k + camera_dx * t, 280.0 + 5 * k, 40.0, 100.0))
    return BoxIndex(rows)


def test_movement_words():
    assert mc.motion_actions("ゴールに向かって走る。") == {"run", "move"}
    assert mc.motion_actions("ゆっくり歩いて位置についた。") == {"walk", "move"}
    assert mc.motion_actions("紙を折った。") == set()


def test_a_standing_person_under_a_panning_camera_did_not_run():
    s = mc.stats_at(index(person_dx=0, camera_dx=300), ["P1"], 30, FPS)   # the camera moves, not the person
    assert s.steps >= mc.MIN_STEPS and s.median_speed < 0.05
    v = mc.verdict("run", s)
    assert v["status"] == "contradicted" and v["show"] != "keep"


def test_a_running_person_is_confirmed_and_the_camera_is_removed():
    s = mc.stats_at(index(person_dx=300, camera_dx=-200), ["P1"], 30, FPS)   # 3 body heights a second
    assert s.median_speed == pytest.approx(3.0, abs=0.2)
    for action in ("run", "move"):
        v = mc.verdict(action, s)
        assert (v["status"], v["show"], v["action"], v["method"]) == ("supported", "keep", action, "motion")
    assert mc.verdict("walk", s)["status"] == "unclear"      # too fast for a walk: not confirmed


def test_a_walk_is_confirmed_only_in_the_walking_range():
    s = mc.stats_at(index(person_dx=70), ["P1"], 30, FPS)
    assert mc.verdict("walk", s)["status"] == "supported"
    assert mc.verdict("run", s)["status"] == "unclear" and mc.verdict("run", s)["show"] == "hedge"


def test_without_enough_other_people_nothing_is_measured():
    s = mc.stats_at(index(person_dx=300, others=1), ["P1"], 30, FPS)   # the camera cannot be read
    assert s.steps == 0
    v = mc.verdict("run", s)
    assert v["status"] == "unchecked" and v["show"] == "keep"
    assert mc.stats_at(None, ["P1"], 30, FPS).steps == 0


# --- the scan --------------------------------------------------------------------------------

class FakeAsk:
    def __init__(self, pos, ctl):
        self.pos, self.ctl, self.calls = pos, ctl, 0

    def __call__(self, images, prompt):
        self.calls += 1
        return self.ctl if "別の動きだけ" in prompt or "しており" in prompt else self.pos


def scene_of(work, ref, text, actions, boxes):
    beh = work.repo.latest_run(work.sid, Stage.behavior, "cam1")
    src = SimpleNamespace(boxes=boxes, fps=FPS, size=(640, 360), video="none.mp4")
    return cs.Scene(ref, beh.id, text, actions, "cam1", 3.0, ["P1"], src)


@pytest.fixture
def scan(monkeypatch, env, stubbed):
    settings, repo = env
    work = make(env, stubbed)
    run = review._latest_pfb(repo, subject(repo, work.sid, "Pa"))   # noqa: SLF001
    state = SimpleNamespace(scenes=[], cues={})
    monkeypatch.setattr(cs, "scenes_of", lambda r, s, ru: list(state.scenes))
    monkeypatch.setattr(cs, "tiles", lambda sc: [np.zeros((4, 4, 3), np.uint8)] * 8)
    return SimpleNamespace(settings=settings, repo=repo, work=work, run=run, state=state)


def rows(scan, event):
    beh = scan.repo.latest_run(scan.work.sid, Stage.behavior, "cam1")
    return scan.repo.claim_checks(beh.id).get(event, {})


def test_movement_is_checked_without_any_model(scan):
    e = scan.work.events["Pa"][0]
    scan.state.scenes = [scene_of(scan.work, e, "ゴールに向かって走る。", {"run", "move"},
                                  index(person_dx=300))]
    ask = FakeAsk("はい", "いいえ")
    r = cs.scan_run(scan.repo, scan.settings, scan.run, ask=ask)
    assert (r.scenes, r.motion, r.asked, ask.calls) == (1, 2, 0, 0)
    got = rows(scan, e)
    assert got["motion:run"]["status"] == "supported" and got["motion:move"]["status"] == "supported"
    cs.scan_run(scan.repo, scan.settings, scan.run, ask=ask)          # already checked: not written again
    assert len(scan.repo.conn.execute("SELECT * FROM claim_checks").fetchall()) == 2


def test_the_model_is_asked_only_when_enabled(scan):
    e = scan.work.events["Pa"][0]
    scan.state.scenes = [scene_of(scan.work, e, "ジャンプしてシュートした。", {"shoot"}, index())]
    ask = FakeAsk("はい", "いいえ")
    cs.scan_run(scan.repo, scan.settings, scan.run, ask=ask, cues=lambda s: {})
    assert ask.calls == 0 and rows(scan, e) == {}          # off: nothing asked, nothing written
    assert scan.settings.claim_check_enabled is False      # the default
    r = cs.scan_run(scan.repo, scan.settings, scan.run, ask=ask, vlm=True,
                    cues=lambda s: {e: cc.Cues(0.3, 7)})
    assert r.asked == 1 and ask.calls == 2
    v = rows(scan, e)["claim:shoot"]
    assert (v["status"], v["show"]) == ("supported", "keep")
    assert v["votes"] == {"pos": "はい", "ctl": "いいえ"} and v["cues"]["pose_frames"] == 7


def test_a_shot_with_no_arm_cue_or_with_the_arms_down_is_not_confirmed(scan):
    e = scan.work.events["Pa"][0]
    scan.state.scenes = [scene_of(scan.work, e, "シュートした。", {"shoot"}, index())]
    cs.scan_run(scan.repo, scan.settings, scan.run, ask=FakeAsk("はい", "いいえ"), vlm=True,
                cues=lambda s: {})                         # the pose could not be measured
    assert rows(scan, e)["claim:shoot"]["status"] == "unclear"
    cs.scan_run(scan.repo, scan.settings, scan.run, ask=FakeAsk("はい", "いいえ"), vlm=True, force=True,
                cues=lambda s: {e: cc.Cues(0.0, 7)})       # arms never above the head
    v = rows(scan, e)["claim:shoot"]
    assert v["status"] in ("unclear", "contradicted") and v["show"] != "keep"


def test_a_failing_model_leaves_no_row(scan):
    e = scan.work.events["Pa"][0]
    scan.state.scenes = [scene_of(scan.work, e, "シュートした。", {"shoot"}, index())]

    def broken(images, prompt):
        raise OSError("connection refused")
    r = cs.scan_run(scan.repo, scan.settings, scan.run, ask=broken, vlm=True,
                    cues=lambda s: {e: cc.Cues(0.3, 7)})
    assert r.skipped == [e] and rows(scan, e) == {}


def test_only_a_server_on_this_machine_is_used(scan):
    s = scan.settings.model_copy(update={"claim_check_vlm_server": "http://192.0.2.5:1234/v1"})
    with pytest.raises(ValueError):
        cs.make_ask(s)
    cs.make_ask(scan.settings)                              # the default is local


# --- how the automatic approval reads the rows -----------------------------------------------

def verdict_row(action, status, show="keep", method="claim"):
    return {"action": action, "status": status, "confidence": "high", "show": show, "reasons": [],
            "method": method}


def test_the_recorded_rechecks_let_a_text_through_or_hold_it(scan):
    work, repo = scan.work, scan.repo
    beh = repo.latest_run(work.sid, Stage.behavior, "cam1")
    text = "ボールを受け取ってシュートし、ゴールに向かって走りました。"
    work.write("Pa", text, work.events["Pa"])
    assert "確かめられていません" in work.held()["claims"]
    for i, e in enumerate(work.events["Pa"]):
        repo.put_claim_check(beh.id, e, "motion:run", verdict_row("run", "supported", method="motion"))
        repo.put_claim_check(beh.id, e, "motion:move", verdict_row("move", "supported", method="motion"))
    held = work.held()["claims"]
    assert "シュート" in held and "走る" not in held and "移動" not in held     # the shot is still open
    repo.put_claim_check(beh.id, work.events["Pa"][0], "claim:shoot", verdict_row("shoot", "supported"))
    assert "claims" not in work.held()
    repo.put_claim_check(beh.id, work.events["Pa"][1], "claim:shoot",
                         verdict_row("shoot", "unclear", show="hedge"))          # a doubtful scene holds it
    assert "弱めた" in work.held()["claims"]


def test_a_text_that_states_a_walk_needs_a_walk_confirmed(scan):
    work, repo = scan.work, scan.repo
    beh = repo.latest_run(work.sid, Stage.behavior, "cam1")
    work.write("Pa", "ゆっくり歩いて、折った紙を机にそろえました。", work.events["Pa"])
    assert "歩く" in work.held()["claims"]
    repo.put_claim_check(beh.id, work.events["Pa"][0], "motion:walk",
                         verdict_row("walk", "supported", method="motion"))
    assert "claims" not in work.held()


# --- the table ----------------------------------------------------------------------------------

def test_the_table_is_added_to_an_existing_database_with_a_copy(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.init_schema(conn)
    conn.execute("DROP TABLE claim_checks")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    db.init_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM claim_checks").fetchone()[0] == 0
    assert list(tmp_path.glob("old.db.pre-migrate-*"))
    conn.close()
    again = sqlite3.connect(path)
    assert again.execute("SELECT name FROM sqlite_master WHERE name='claim_checks'").fetchone()
