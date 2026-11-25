"""自動承認 (services.autoapprove): the machine checks, the record, the undo, the migration, the screen.

Only synthetic data: invented names and texts, tiny generated images. The tracker's number
timeline is stubbed (`states` / `facts`), the real functions have their own tests below.
"""
import re
from types import SimpleNamespace

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline import identity_check as ident
from pfb_app.pipeline.identity_check import PersonFacts
from pfb_app.services import autoapprove, review
from pfb_app.storage import db
from test_wizard_ui_basic import app, env, html, ok, primaries, screen_text, video_assignment  # noqa: F401 (env)
from test_wizard_ui_people_wait import finish_all, group, mapped, save, track
from test_wizard_ui_review_pdf import btn, has_button, subject

OWN_TEXT = "紙を折って柱を作り、塔の土台を整えていました。"   # no action word, no number, 20+ chars
OTHER_TEXTS = {"Pb": "定規で高さを測りながら、塔の柱の長さをそろえていました。",
               "Pc": "折った紙を重ねて、塔のてっぺんの形を整えていました。"}
CLEAN = PersonFacts(frozenset({"7"}), False, False, False, False)


def event(eid, pid, what="紙を折った", conf="high", t="00:12"):
    return {"event_id": eid, "camera": "cam1", "video_time": t, "task_elapsed": t,
            "participant": pid, "observed_behavior": what, "utterance": "", "speaker": "unknown",
            "evidence_source": "video", "confidence": conf}


class Work:
    """A tracking group of Pa, Pb, Pc, three scenes each, every scene judged good."""

    def __init__(self, settings, repo, texts=None, refs=None, conf=None, claims=None):
        self.repo = repo
        self.aid = video_assignment(repo)
        self.sid = group(repo, self.aid)
        track(settings, repo, self.sid)
        mapped(repo, self.sid)
        finish_all(repo, self.sid)
        ident_run = repo.latest_run(self.sid, Stage.identity, "cam1")
        self.events = {p: [f"cam1-{k}{i:02d}-01" for i in range(1, 4)]
                       for k, p in enumerate(("Pa", "Pb", "Pc"))}
        evs = [event(e, p, conf=(conf or {}).get(e, "high"), t=f"00:{k * 10 + i:02d}")
               for k, p in enumerate(self.events) for i, e in enumerate(self.events[p])]
        out = {"events": evs}
        if claims:
            out["claim_checks"] = claims
        beh = save(repo, self.sid, Stage.behavior, out, camera="cam1", upstream=[ident_run.id])
        con = save(repo, self.sid, Stage.contribution, {"judgments": [
            {"ref_id": e["event_id"], "contribution_interpretation": "つながった",
             "is_positive_contribution": True, "rubric_criteria": []} for e in evs]},
            camera="cam1", upstream=[beh.id])
        self.con = con
        texts = texts or {}
        self.states: dict[str, str] = {}
        self.facts: dict[str, PersonFacts] = {}
        for s in repo.list_subjects(self.sid):
            pid = s["participant_id"]
            run = self.write(pid, texts.get(pid, OTHER_TEXTS.get(pid, OWN_TEXT)),
                             (refs or {}).get(pid, self.events[pid][:3]))
            assert run.id
            for e in self.events[pid]:
                self.states[e] = ident.SAME
            self.facts[pid] = CLEAN if pid == "Pa" else PersonFacts(frozenset({str(7 + len(pid))}), False, False, False, False)

    def write(self, pid, text, refs):
        s = subject(self.repo, self.sid, pid)
        return save(self.repo, self.sid, Stage.pfb,
                    {"pfb_status": "generated", "pfb": text, "evidence_refs": list(refs)},
                    subject_id=s["id"], upstream=[self.con.id])

    def verdict(self, pid="Pa", policy=autoapprove.Policy()):
        s = subject(self.repo, self.sid, pid)
        return autoapprove.evaluate(self.repo, s, review._latest_pfb(self.repo, s), policy)  # noqa: SLF001

    def held(self, pid="Pa", **kw) -> dict[str, str]:
        return {c.id: c.detail for c in self.verdict(pid, **kw).held}


@pytest.fixture
def stubbed(monkeypatch):
    """The number timeline is stubbed: `work.states` (event id -> state), `work.facts`."""
    holder = SimpleNamespace(work=None)
    monkeypatch.setattr(autoapprove.ident, "event_states",
                        lambda repo, behavior: dict(holder.work.states))
    monkeypatch.setattr(autoapprove.ident, "person_facts",
                        lambda repo, behavior, pid: holder.work.facts.get(pid))
    return holder


def make(env, stubbed, **kw) -> Work:
    settings, repo = env
    work = Work(settings, repo, **kw)
    stubbed.work = work
    return work


# --- every condition -------------------------------------------------------------------------

def test_a_text_that_passes_every_check_is_auto_approved(env, stubbed):
    work = make(env, stubbed)
    verdict = work.verdict()
    assert verdict.passed and verdict.reasons == []
    assert [c.id for c in verdict.checks] == ["base", "evidence", "scenes", "confidence", "identity",
                                              "claims", "wording", "jersey", "player", "text"]
    review.set_auto_approve(work.repo, work.aid, True)
    run = review._latest_pfb(work.repo, subject(work.repo, work.sid, "Pa"))  # noqa: SLF001
    assert review.auto_approve(work.repo, run) is True
    s = subject(work.repo, work.sid, "Pa")
    assert review.subject_status(work.repo, s["id"]).value == "approved"
    assert review.approved_automatically(work.repo, s)


def test_a_scene_of_another_person_stops_the_text(env, stubbed):        # the failure example
    work = make(env, stubbed)
    work.states[work.events["Pa"][1]] = ident.FOREIGN                   # 別人かも
    assert "別の人かもしれない" in work.held()["identity"]
    work.states[work.events["Pa"][1]] = ident.SAME
    work.write("Pa", OWN_TEXT, [*work.events["Pa"][:2], work.events["Pb"][0]])   # a scene of Pb
    held = work.held()
    assert "別の人の記録が混ざって" in held["evidence"]


def test_one_undecidable_scene_stops_the_text(env, stubbed):
    work = make(env, stubbed)
    work.states[work.events["Pa"][2]] = ident.UNKNOWN                   # 判定不能
    assert "判定できない" in work.held()["identity"]
    work.states.clear()                                                  # no number timeline at all
    assert "identity" in work.held()


def test_the_same_number_on_two_people_or_two_numbers_on_one_person_stop_it(env, stubbed):
    work = make(env, stubbed)
    work.facts["Pa"] = PersonFacts(frozenset({"7"}), False, True, False, False)
    assert "同じ背番号" in work.held()["identity"]
    work.facts["Pa"] = PersonFacts(frozenset({"7", "9"}), True, False, False, False)
    assert "ちがう背番号" in work.held()["identity"]


def test_an_unchecked_action_claim_stops_the_text(env, stubbed):
    work = make(env, stubbed)
    work.write("Pa", "ボールを受け取ってシュートしたあと、ゴールに向かって走りました。", work.events["Pa"])
    detail = work.held()["claims"]
    assert "シュート" in detail and "確かめられていません" in detail


def test_an_action_claim_rechecked_as_supported_passes(env, stubbed):
    claims = {f"cam1-0{i:02d}-01": {"action": "shoot", "status": "supported", "confidence": "high",
                                    "show": "keep", "reasons": []} for i in range(1, 4)}
    work = make(env, stubbed, claims=claims)
    work.write("Pa", "ボールを受け取って、ゴールにシュートしました。二回目のシュートも続けました。",
               work.events["Pa"])
    assert "claims" not in work.held()
    work.write("Pa", "ボールを受け取ってドリブルし、ゴールにシュートしました。", work.events["Pa"])
    assert "ドリブル" in work.held()["claims"]                          # only the shot was checked


def test_a_weakened_scene_or_a_hedged_sentence_is_never_auto_approved(env, stubbed):
    claims = {"cam1-001-01": {"action": "shoot", "status": "unclear", "confidence": "low",
                              "show": "hedge", "reasons": ["映像からは判断できない"]}}
    work = make(env, stubbed, claims=claims)
    assert "弱めた" in work.held()["claims"]
    work.write("Pb", "ボールを持っているように見える（要確認）。紙を折って柱を作りました。", work.events["Pb"])
    assert "要確認" in work.held("Pb")["claims"]


def test_a_comparison_or_grading_word_stops_the_text(env, stubbed):
    work = make(env, stubbed)
    for text in ("ほかの選手より多く柱を折って、塔の土台を整えていました。",
                 "チームで一番早く柱を折って、塔の土台を整えていました。",
                 "柱を折るのが上手で、塔の土台を整えていました。",
                 "柱を折って塔の土台を整え、チームで1位の高さでした。"):
        work.write("Pa", text, work.events["Pa"])
        assert "wording" in work.held(), text
    work.write("Pa", OWN_TEXT, work.events["Pa"])
    assert "wording" not in work.held()


def test_a_jersey_number_must_carry_the_question_mark(env, stubbed):
    work = make(env, stubbed)
    for text in ("7番が紙を折って柱を作り、塔の土台を整えていました。",
                 "背番号7の人が紙を折って柱を作り、塔の土台を整えました。",
                 "#7が紙を折って柱を作り、塔の土台を整えていました。"):
        work.write("Pa", text, work.events["Pa"])
        assert "jersey" in work.held(), text
    work.write("Pa", "7番？の人が紙を折って柱を作り、塔の土台を整えていました。", work.events["Pa"])
    assert "jersey" not in work.held()
    work.write("Pa", "一番目の柱を折り、塔の土台を整えていました。1番目の工程です。", work.events["Pa"])
    assert "jersey" not in work.held()


def test_evidence_scenes_confidence_and_policy(env, stubbed):
    work = make(env, stubbed, conf={"cam1-001-01": "medium"})
    assert "confidence" in work.held()
    assert "confidence" not in work.held(policy=autoapprove.Policy(min_confidence="medium"))
    work.write("Pa", OWN_TEXT, ["cam1-001-01"])
    held = work.held(policy=autoapprove.Policy(min_confidence="low"))
    assert "1件" in held["scenes"] and "2件" in held["scenes"]
    work.write("Pa", OWN_TEXT, ["cam1-999-01", "cam1-001-01"])
    assert "見つかりません" in work.held(policy=autoapprove.Policy(min_confidence="low"))["evidence"]


def test_a_scene_not_judged_good_is_not_a_basis(env, stubbed):
    settings, repo = env
    work = make(env, stubbed)
    out = dict(work.con.output)
    out["judgments"] = [dict(j, is_positive_contribution=j["ref_id"] != "cam1-001-01")
                        for j in out["judgments"]]
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (db._json(out), work.con.id))  # noqa: SLF001
    assert "よかった場面として記録されていない" in work.held()["evidence"]


def test_player_checks_text_checks_and_short_texts(env, stubbed):
    work = make(env, stubbed)
    work.facts["Pa"] = PersonFacts(frozenset({"7"}), False, False, True, False)
    assert "選手以外" in work.held()["player"]
    work.facts["Pa"] = PersonFacts(frozenset({"7"}), False, False, False, True)
    assert "選手以外" in work.held()["player"]
    work.facts["Pa"] = CLEAN
    work.write("Pa", "柱を折りました。", work.events["Pa"])
    assert "短すぎ" in work.held()["text"]
    work.write("Pa", "紙を折って柱を作りました。紙を折って柱を作りました。塔の土台を整えました。", work.events["Pa"])
    assert "くり返" in work.held()["text"]
    work.write("Pa", OWN_TEXT, work.events["Pa"])
    work.write("Pb", OWN_TEXT, work.events["Pb"])
    assert "ほかの人と同じ" in work.held()["text"]


def test_the_base_gates_still_hold(env, stubbed):
    work = make(env, stubbed)
    work.write("Pa", "赤い服の人が紙を折って柱を作り、塔の土台を整えました。", work.events["Pa"])
    run = review._latest_pfb(work.repo, subject(work.repo, work.sid, "Pa"))  # noqa: SLF001
    work.repo.conn.execute("UPDATE stage_runs SET warnings=? WHERE id=?",
                           (db._json(["色の語が含まれています: 赤"]), run.id))  # noqa: SLF001
    assert set(work.held()) == {"base"}
    work.write("Pa", OWN_TEXT, work.events["Pa"])
    work.repo.set_identity_confirmed(work.sid, False)
    assert set(work.held()) == {"base"}


def test_documents_are_never_auto_approved(env):
    settings, repo = env
    aid = repo.create_assignment("探究レポート", "document", "調べる", "観点", "standard", False)
    sid = repo.create_submission(aid, "a.pdf", {"source_path": "/x.pdf", "original_filename": "a.pdf"})
    sub = repo.add_subject(sid, None, "山田", None, {})
    run = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": OWN_TEXT, "evidence_refs": []},
               subject_id=sub)
    verdict = autoapprove.evaluate(repo, repo.get_subject(sub), run)
    assert not verdict.passed and "動画以外" in verdict.reasons[0]


# --- off means as before; the record; the undo ------------------------------------------------

def test_the_switch_is_off_by_default_and_off_means_nothing_is_auto_approved(env, stubbed):
    work = make(env, stubbed)
    assert work.repo.get_assignment(work.aid)["auto_approve"] is False
    run = review._latest_pfb(work.repo, subject(work.repo, work.sid, "Pa"))  # noqa: SLF001
    assert review.auto_approve(work.repo, run) is False
    assert autoapprove.apply_now(work.repo, work.aid) == ([], 0)
    assert autoapprove.counts(work.repo, work.aid) == (0, 3)
    assert Settings().auto_approve_min_scenes == 2 and Settings().auto_approve_min_confidence == "high"


def test_a_persons_bulk_approval_is_still_a_persons_approval(env, stubbed):
    work = make(env, stubbed)
    shown = {s["id"]: review._latest_pfb(work.repo, s).id for s in work.repo.list_subjects(work.sid)}  # noqa: SLF001
    result = review.bulk_approve(work.repo, shown)
    assert len(result.approved) == 3
    assert all(not s["approved_auto"] for s in work.repo.list_subjects(work.sid))
    assert work.repo.list_auto_approvals(subject(work.repo, work.sid, "Pa")["id"]) == []


def test_the_record_keeps_every_check_the_time_and_the_policy(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    approved, left = autoapprove.apply_now(work.repo, work.aid)
    assert len(approved) == 3 and left == 0
    s = subject(work.repo, work.sid, "Pa")
    rec = work.repo.active_auto_approval(s["id"], s["approved_run_id"] or work.repo.get_subject(s["id"])["approved_run_id"])
    assert rec["policy_version"] == autoapprove.POLICY_VERSION and rec["approved_at"]
    assert rec["policy"] == autoapprove.Policy().as_dict()
    assert [c["id"] for c in rec["checks"]] == ["base", "evidence", "scenes", "confidence", "identity",
                                                "claims", "wording", "jersey", "player", "text"]
    assert all(c["ok"] for c in rec["checks"])
    assert work.repo.latest_audit("auto_approved", f"subject:{s['id']}")
    listed = autoapprove.listing(work.repo, work.aid)
    assert {item.name for item in listed} == {"花子", "太郎", "次郎"} and all(i.recorded for i in listed)
    assert autoapprove.counts(work.repo, work.aid) == (3, 0)


def test_apply_now_approves_only_what_passes_and_says_how_many_wait(env, stubbed):
    work = make(env, stubbed)
    work.states[work.events["Pb"][0]] = ident.UNKNOWN
    review.set_auto_approve(work.repo, work.aid, True)
    approved, left = autoapprove.apply_now(work.repo, work.aid)
    assert len(approved) == 2 and left == 1
    assert review.subject_status(work.repo, subject(work.repo, work.sid, "Pb")["id"]).value == "review"
    assert autoapprove.counts(work.repo, work.aid) == (2, 1)


def test_revoke_puts_every_automatic_approval_back_and_keeps_a_persons_own(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    pc = subject(work.repo, work.sid, "Pc")
    review.approve(work.repo, pc["id"])               # a person confirms Pc: no longer automatic
    assert not work.repo.get_subject(pc["id"])["approved_auto"]
    assert [r["status"] for r in work.repo.list_auto_approvals(pc["id"])] == ["confirmed"]
    back = autoapprove.revoke(work.repo, work.aid)
    assert sorted(back) == sorted(subject(work.repo, work.sid, p)["id"] for p in ("Pa", "Pb"))
    assert review.subject_status(work.repo, pc["id"]).value == "approved"          # untouched
    for p in ("Pa", "Pb"):
        s = subject(work.repo, work.sid, p)
        assert review.subject_status(work.repo, s["id"]).value == "review"
        assert [r["status"] for r in work.repo.list_auto_approvals(s["id"])] == ["revoked"]
    assert work.repo.latest_audit(autoapprove.AUDIT_REVOKED, f"assignment:{work.aid}")["detail"]["subjects"] == back
    assert autoapprove.counts(work.repo, work.aid) == (0, 2)
    assert autoapprove.revoke(work.repo, work.aid) == []                            # nothing left


def test_a_revoked_text_is_not_approved_automatically_again(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    pa = subject(work.repo, work.sid, "Pa")
    run = review._latest_pfb(work.repo, pa)  # noqa: SLF001
    assert autoapprove.revoke(work.repo, work.aid, {pa["id"]}) == [pa["id"]]
    assert review.auto_approve(work.repo, run) is False
    assert autoapprove.apply_now(work.repo, work.aid)[0] == []
    assert "取り消された" in work.held()["base"]
    new = work.write("Pa", "紙を折って柱を作り、塔の土台を整え、高さを測っていました。", work.events["Pa"])
    assert review.auto_approve(work.repo, new) is True                                # a new text is checked anew
    assert [r["status"] for r in work.repo.list_auto_approvals(pa["id"])] == ["revoked", "active"]


def test_a_newer_automatic_approval_replaces_the_older_record(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    pa = subject(work.repo, work.sid, "Pa")
    new = work.write("Pa", "紙を折って柱を作り、塔の土台を整え、高さを測っていました。", work.events["Pa"])
    assert review.auto_approve(work.repo, new) is True
    assert [r["status"] for r in work.repo.list_auto_approvals(pa["id"])] == ["replaced", "active"]


def test_unapprove_by_a_person_closes_the_record(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    pa = subject(work.repo, work.sid, "Pa")
    review.unapprove(work.repo, pa["id"])
    assert [r["status"] for r in work.repo.list_auto_approvals(pa["id"])] == ["revoked"]


# --- the migration: only added, the old rows stay --------------------------------------------

def test_the_migration_only_adds_a_table_and_keeps_every_row(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.init_schema(conn)
    repo = db.Repo(conn)
    aid = repo.create_assignment("A", "video", "t", "r", "standard", True)
    sid = repo.create_submission(aid, "g", {})
    repo.add_subject(sid, "P1", "X", "", {})
    conn.execute("DROP TABLE auto_approvals")
    before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("assignments", "submissions", "subjects", "stage_runs", "audit_log")}
    schema = {r[0]: r[1] for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")}
    conn.close()
    conn = db.connect(path)
    db.init_schema(conn)
    after = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in before}
    assert after == before
    now = {r[0]: r[1] for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")}
    assert {k for k in now if k not in schema} == {"auto_approvals"}
    assert all(now[k] == schema[k] for k in schema)                    # no existing table changed
    assert any(p.name.startswith("old.db.pre-migrate") for p in tmp_path.iterdir())   # a copy first
    db.init_schema(conn)                                                # idempotent


def test_deleting_a_work_removes_its_records(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    assert work.repo.conn.execute("SELECT COUNT(*) FROM auto_approvals").fetchone()[0] == 3
    work.repo.conn.execute("DELETE FROM assignments WHERE id=?", (work.aid,))
    assert work.repo.conn.execute("SELECT COUNT(*) FROM auto_approvals").fetchone()[0] == 0


# --- identity_check on a fake timeline --------------------------------------------------------

def test_event_states_and_person_facts_read_the_number_timeline(monkeypatch):
    reads = {
        "P1": {"dominant": "5", "timeline_s": 6.0, "segments": [[1, 600, "5", 3]]},
        "P2": {"dominant": "5", "timeline_s": 6.0, "segments": [[1, 600, "5", 3]]},     # same number again
        "P3": {"dominant": "9", "timeline_s": 6.0, "segments": [[1, 100, "4", 2], [200, 600, "9", 3]]},
        "P4": {"dominant": None, "timeline_s": 6.0, "segments": []},
    }
    names = SimpleNamespace(output={"cues": {"number": reads}, "teams": {"P1": "T1", "P2": "T2"}})
    meta = SimpleNamespace(fps=10.0)
    mapping = {"P1": "Pa", "P2": "Pb", "P3": "Pc", "P4": "Pd"}
    monkeypatch.setattr(ident, "_context", lambda repo, behavior: (names, meta, mapping))
    behavior = SimpleNamespace(output={"events": [
        {"event_id": "e1", "participant": "Pa", "video_time": "00:10"},      # frame 101: P1 reads 5
        {"event_id": "e2", "participant": "Pc", "video_time": "00:05"},      # frame 51: P3 reads 4 (not 9)
        {"event_id": "e3", "participant": "Pc", "video_time": "00:40"},      # frame 401: P3 reads 9
        {"event_id": "e4", "participant": "Pd", "video_time": "00:10"},      # no number at all
        {"event_id": "e5", "participant": "Pa", "video_time": None},
    ]}, submission_id=1, camera="cam1")
    states = ident.event_states(None, behavior)
    assert states == {"e1": ident.SAME, "e2": ident.FOREIGN, "e3": ident.SAME, "e4": ident.UNKNOWN,
                      "e5": ident.UNKNOWN}
    assert ident.foreign_events(None, behavior) == {"e2"}
    repo = SimpleNamespace(get_submission=lambda sid: {"meta": {}})
    facts = ident.person_facts(repo, behavior, "Pa")
    assert facts.numbers == {"5"} and not facts.duplicate        # the same number on the other team
    names.output["teams"] = {}
    assert ident.person_facts(repo, behavior, "Pa").duplicate    # teams unknown: two people with 5
    assert not ident.person_facts(repo, behavior, "Pc").conflict
    reads["P3b"] = {"dominant": "2", "timeline_s": 6.0, "segments": [[1, 10, "2", 2]]}
    mapping["P3b"] = "Pc"
    assert ident.person_facts(repo, behavior, "Pc").conflict
    names.output["teams"] = {"P1": "other"}
    assert ident.person_facts(repo, behavior, "Pa").other_team
    repo = SimpleNamespace(get_submission=lambda sid: {"meta": {"participation": {"cam1": {"check": {"P1": "x"}}}}})
    assert ident.person_facts(repo, behavior, "Pa").unsure


def test_without_a_timeline_nothing_can_be_told(monkeypatch):
    monkeypatch.setattr(ident, "_context", lambda repo, behavior: None)
    behavior = SimpleNamespace(output={"events": [{"event_id": "e1"}]})
    assert ident.event_states(None, behavior) == {} and ident.foreign_events(None, behavior) == set()
    assert ident.person_facts(None, behavior, "Pa") is None


# --- words -----------------------------------------------------------------------------------

def test_word_lists():
    assert autoapprove.wording_problems("ほかの選手より多い") and autoapprove.wording_problems("MVPでした")
    assert autoapprove.wording_problems("紙を折りました") == []
    assert autoapprove.jersey_problems("７番が走る") == ["7番"]
    assert autoapprove.jersey_problems("7番？が走る") == [] and autoapprove.jersey_problems("1番目") == []
    assert autoapprove.actions_stated("パスをして走った。") == {"pass", "run"}
    assert autoapprove.actions_stated("紙を折った。") == set()


def test_the_screens_words_are_neutral():
    from pfb_app.ui.wizard import step_review as sr
    words = [autoapprove.LABEL, autoapprove.SWITCH, sr.AUTO_HELP, sr.AUTO_NOTE, sr.AUTO_HELD, sr.AUTO_REVOKE,
             sr.AUTO_REVOKE_HELP, sr.AUTO_PASSED, sr.AUTO_LINE, sr.AUTO_ONLY_WAITING, sr.AUTO_APPLY]
    for w in words:
        assert not [x for x in ("先生", "生徒", "学生", "採点", "成績", "提出") if x in w], w
    assert "機械チェック済み・人は未確認" in autoapprove.LABEL
    from pfb_app.export import labels
    assert "機械チェック済み・人は未確認" in labels.footer_line(None, True, auto=True)
    assert "機械チェック済み・人は未確認" in labels.evaluator_footer_line(None, True, [True, False])
    assert "機械チェック済み・人は未確認" in labels.AUTO


# --- the screen --------------------------------------------------------------------------------

def test_the_review_screen_without_the_switch_is_as_before(env, stubbed):
    work = make(env, stubbed)
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    page = screen_text(at)
    assert "自動承認 " not in page and "人の確認待ちだけを見る" not in page
    assert "機械チェックを通った文を自動承認する" in page             # the switch itself (off)
    assert primaries(at) == ["全員まとめて承認"]


def test_the_review_screen_shows_the_summary_the_marks_and_the_reasons(env, stubbed):
    work = make(env, stubbed)
    work.states[work.events["Pb"][0]] = ident.FOREIGN
    review.set_auto_approve(work.repo, work.aid, True)
    approved, left = autoapprove.apply_now(work.repo, work.aid)
    assert len(approved) == 2 and left == 1
    pb = subject(work.repo, work.sid, "Pb")
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    page = html(at)
    assert "自動承認 2件／人の確認待ち 1件" in page
    assert "承認 0 / 全 3" in page and "自動承認 2（人は未確認）" in page   # a person's count apart
    # the waiting person is the one shown, with the reasons
    assert "自動承認しなかった理由" in page and "別の人かもしれない場面があります" in page
    assert "pfb-stamped" not in page
    # an automatic approval never wears a person's stamp
    at.session_state["pfb_student_%s" % work.sid] = subject(work.repo, work.sid, "Pa")["id"]
    at = at.run()
    page = html(at)
    assert "自動承認（機械チェック済み・人は未確認）" in page
    assert "この文章は自動承認です" in screen_text(at)
    assert has_button(at, "人が確認して承認する") and has_button(at, "この人の自動承認を取り消す")
    assert review.subject_status(work.repo, pb["id"]).value == "review"


def test_the_filter_shows_only_people_waiting_for_a_person(env, stubbed):
    work = make(env, stubbed)
    work.states[work.events["Pb"][0]] = ident.UNKNOWN
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    options = [str(o) for w in [*at.radio, *at.selectbox] if (w.key or "").startswith("pfb_student_") for o in w.options]
    assert len(options) == 3
    key = f"pfb_waiting_only_{work.sid}"
    next(t for t in at.toggle if t.key == key).set_value(True)
    at = ok(at.run())
    assert not any((w.key or "").startswith("pfb_student_") for w in [*at.radio, *at.selectbox])  # one person left: no switcher
    heads = re.findall(r'pfb-who__name">([^<]*)<', html(at))
    assert heads == ["太郎さん"]                                      # the people who wait for a person


def test_the_revoke_button_puts_everyone_back(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    assert "自動承認 3件／人の確認待ち 0件" in html(at)
    btn(at, "自動承認を取り消して人の確認待ちに戻す").click()
    at = ok(at.run())
    assert autoapprove.counts(work.repo, work.aid) == (0, 3)
    assert all(review.subject_status(work.repo, s["id"]).value == "review"
               for s in work.repo.list_subjects(work.sid))
    assert "3件の自動承認を取り消しました" in screen_text(at)


def test_a_person_can_confirm_an_automatic_approval(env, stubbed):
    work = make(env, stubbed)
    review.set_auto_approve(work.repo, work.aid, True)
    autoapprove.apply_now(work.repo, work.aid)
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    btn(at, "人が確認して承認する").click()
    at = ok(at.run())
    pa = subject(work.repo, work.sid, "Pa")
    assert not work.repo.get_subject(pa["id"])["approved_auto"]
    assert work.repo.list_auto_approvals(pa["id"])[0]["status"] == "confirmed"


def test_the_work_switch_changes_through_the_screen(env, stubbed):
    work = make(env, stubbed)
    at = ok(app(session={"assignment_id": work.aid, "step": 5, "submission_id": None}))
    toggle = next(t for t in at.toggle if t.key == f"pfb_auto_approve_{work.aid}")
    assert toggle.value is False
    toggle.set_value(True)
    at = ok(at.run())
    assert work.repo.get_assignment(work.aid)["auto_approve"] is True
    assert "自動承認 0件／人の確認待ち 3件" in html(at)
    assert has_button(at, "いま待っている文章にも機械チェックをかける")
