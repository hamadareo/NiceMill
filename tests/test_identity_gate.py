"""同一人物の確認: only a confirmed person is printed in a correct_later output."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from test_export import add_video_subject, repo, text_of, video_assignment  # noqa: F401,E402

import pytest  # noqa: E402

from pfb_app.export import identity_gate as G  # noqa: E402
from pfb_app.export.pdf import ExportError, _html, render_student_pdf, render_teacher_pdf  # noqa: E402
from pfb_app.pipeline import identity_check as ident  # noqa: E402
from pfb_app.services import corrections as cor  # noqa: E402


def _fake(monkeypatch, *, timeline, dominant="8", states=None, other=None):
    """A person with one label P1 whose number reads are `timeline` and whose events are `states`."""
    cam = SimpleNamespace(track=SimpleNamespace(id=1), labels_of=lambda pid: ["P1"], name="cam1")
    sub = SimpleNamespace(sid=1, camera=lambda name, *a, **k: cam)
    events = [{"event_id": f"cam1-{i:04d}", "participant": "Pa"} for i in range(len(states))]
    source = SimpleNamespace(id=5, camera="cam1", output={"events": events})
    read = {"dominant": dominant, "timeline": timeline,
            "segments": [[timeline[0][0], timeline[-1][0], dominant, 2]] if dominant else []}
    reads = {"P1": read, **(other or {})}
    monkeypatch.setattr(ident, "event_states",
                        lambda repo, src: {e["event_id"]: s for e, s in zip(events, states)})
    from pfb_app.services import auto_people
    monkeypatch.setattr(auto_people, "_number_run",
                        lambda *a, **k: SimpleNamespace(output={"cues": {"number": reads}}))
    repo = SimpleNamespace(settings=None)
    return repo, sub, {"participant_id": "Pa"}, [(None, source)]


def test_confirmed(monkeypatch):
    tl = [[0, "8"], [180, None], [360, "8"], [540, "8"], [720, "16"]]
    v = G.assess(*_fake(monkeypatch, timeline=tl, states=["same", "same", "same", "unknown"]))
    assert v.state == G.OK and v.facts["reads"] == 3 and v.facts["numbered"] == 4
    assert v.same_refs == {"cam1-0000", "cam1-0001", "cam1-0002"}
    assert len(v.restrict("cam1", __import__("numpy").arange(0, 800))) == 721   # only the read spans


def test_doubtful_without_enough_reads_or_scenes_or_number(monkeypatch):
    v = G.assess(*_fake(monkeypatch, timeline=[[0, "8"], [180, "8"]], states=["same"] * 5))
    assert v.state == G.DOUBT and "足りません" in v.reasons[0]
    v = G.assess(*_fake(monkeypatch, timeline=[[0, "8"]] * 4, states=["same", "same"]))
    assert v.state == G.DOUBT and any("場面" in r for r in v.reasons)
    v = G.assess(*_fake(monkeypatch, timeline=[[0, None]], dominant=None, states=["unknown"]))
    assert v.state == G.DOUBT and "手がかり" in v.reasons[0]
    v = G.assess(*_fake(monkeypatch, timeline=[[0, "8"]] * 4, states=["same"] * 3,
                        other={"P2": {"dominant": "8"}}))
    assert v.state == G.DOUBT and v.facts["duplicate"]


def test_mixed_when_another_player_is_in_it(monkeypatch):
    v = G.assess(*_fake(monkeypatch, timeline=[[0, "8"]] * 4, states=["same", "foreign", "same", "same"]))
    assert v.state == G.MIXED and "別の人" in v.reasons[0]


def test_policy_override():
    p = G.policy(SimpleNamespace(identity_gate={"min_reads": 1, "min_share": 0.3, "bogus": 1}))
    assert p.min_reads == 1 and p.min_share == 0.3 and p.min_same_scenes == 3


def _verdict(state, same=()):
    return lambda *a, **k: G.Verdict(state, ["確かめる手がかりがありません"] if state != G.OK else [],
                                     {"reads": 1}, set(same))


def _clue(monkeypatch, value=True):
    """Somebody's number is read in the work (or nobody's, value=False)."""
    monkeypatch.setattr(G, "work_has_clue", lambda repo_, work: value)


def test_only_confirmed_people_are_printed(repo, monkeypatch):
    _clue(monkeypatch)
    aid = video_assignment(repo)
    ok = add_video_subject(repo, aid, "#8", pfb_text="確かな人の文です。", approved=False)
    add_video_subject(repo, aid, "#9", pfb_text="疑わしい人の文です。", approved=False)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    ok_participant = repo.get_subject(ok["subj"])["participant_id"]
    monkeypatch.setattr(G, "assess", lambda repo_, sub, subject, chain, pol=None:
                        _verdict(G.OK, {"cam1-0001"})() if subject["id"] == ok["subj"] else _verdict(G.DOUBT)())
    student = text_of(render_student_pdf(repo, aid))
    assert "確かな人の文です" in student and "疑わしい人の文です" not in student
    assert "同一人物と確かめられた場面" in student                     # the small note on the page
    teacher = _html("teacher.html.j2", repo, aid)
    assert "同一人物の確認が取れていない" in teacher and "確かめる手がかりがありません" in teacher


def test_nobody_confirmed_is_an_error_for_the_pdfs(repo, monkeypatch):
    _clue(monkeypatch)
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "#9", pfb_text="疑わしい人の文です。", approved=False)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    monkeypatch.setattr(G, "assess", lambda *a, **k: _verdict(G.MIXED)())
    with pytest.raises(ExportError):
        render_student_pdf(repo, aid)


def test_approve_mode_never_runs_the_gate(repo, monkeypatch):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "山田花子")
    monkeypatch.setattr(G, "assess", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate ran")))
    assert "山田花子" in text_of(render_student_pdf(repo, aid))
    assert render_teacher_pdf(repo, aid)


# --- the three ways through: the number, a person's check, no clue at all --------------------

def _chain(events, identity_id=7):
    source = SimpleNamespace(camera="cam1", output={"events": events}, upstream_run_ids=[identity_id])
    return [(None, source)]


def _identity(repo_runs, run_id=7, source="model", pids=("Pa",)):
    run = SimpleNamespace(id=run_id, stage="identity", status="ok", source=source, error=None,
                          output={"participants": [{"participant_id": p} for p in pids]})
    repo_runs[run_id] = run
    return run


class _Repo:
    def __init__(self):
        self.runs = {}

    def get_run(self, i):
        return self.runs.get(i)


EVENTS = [{"event_id": "cam1-0001", "participant": "Pa"}, {"event_id": "cam1-0002", "participant": "Pa"},
          {"event_id": "cam1-0003", "participant": "Pb"}]


def test_decide_number_checked_no_clue_and_blocked(monkeypatch):
    monkeypatch.setattr(G, "usable", lambda run: run is not None and run.status == "ok")
    r = _Repo()
    subject = {"participant_id": "Pa"}
    ok = G.Verdict(G.OK, [], {}, {"cam1-0001"})
    doubt = G.Verdict(G.DOUBT, ["確かめる手がかりがありません"], {}, set())
    mixed = G.Verdict(G.MIXED, ["別の人が入っています"], {}, set(), foreign_refs={"cam1-0002"})
    _identity(r.runs)                                           # an automatic mapping
    d = G.decide(r, {"meta": {}}, subject, _chain(EVENTS), ok, clue=True)
    assert d.passed and d.how == G.NUMBER and d.keep == {"cam1-0001"} and d.restrict
    d = G.decide(r, {"meta": {}}, subject, _chain(EVENTS), doubt, clue=True)
    assert not d.passed and d.how == ""                         # a clue elsewhere: left out
    d = G.decide(r, {"meta": {}}, subject, _chain(EVENTS), doubt, clue=False)
    assert d.passed and d.how == G.NO_CLUE and d.keep is None and not d.restrict
    # the evaluator decided the mapping (its id kept on the group): printed with every scene,
    # as in an approved work, even one where another number was read (decided 2026-11-23)
    checked = {"meta": {G.PEOPLE_CHECKED: [7]}}
    d = G.decide(r, checked, subject, _chain(EVENTS), doubt, clue=True)
    assert d.passed and d.how == G.CHECKED and d.keep is None and not d.restrict
    d = G.decide(r, checked, subject, _chain(EVENTS), mixed, clue=True)
    assert d.passed and d.how == G.CHECKED and d.keep is None and not d.restrict
    # a mapping saved on 「人を確かめる」 (source teacher) counts the same
    _identity(r.runs, source="teacher")
    assert G.decide(r, {"meta": {}}, subject, _chain(EVENTS), doubt, clue=True).how == G.CHECKED
    # ... but not when that mapping does not map the person, or a camera's mapping is automatic
    _identity(r.runs, source="teacher", pids=("Pb",))
    assert not G.decide(r, {"meta": {}}, subject, _chain(EVENTS), doubt, clue=True).passed
    _identity(r.runs, source="teacher")
    _identity(r.runs, run_id=8)
    two = _chain(EVENTS) + [(None, SimpleNamespace(camera="cam2", upstream_run_ids=[8],
                                                   output={"events": [{"event_id": "cam2-0001",
                                                                       "participant": "Pa"}]}))]
    assert not G.decide(r, {"meta": {}}, subject, two, doubt, clue=True).passed


def test_checked_ids_read_only_numbers():
    assert G.checked_ids({"meta": {G.PEOPLE_CHECKED: [3, "4", "x", None]}}) == {3, 4}
    assert G.checked_ids({"meta": None}) == set()


def test_marking_the_people_checked_records_the_latest_mappings(repo):
    from pfb_app.services import review
    aid = video_assignment(repo)
    made = add_video_subject(repo, aid, "#8", approved=False)
    sid = made["sid"]
    ident = repo.latest_run(sid, __import__("pfb_app.domain.models", fromlist=["Stage"]).Stage.identity,
                            "cam1")
    assert review.mark_people_checked(repo, sid) == [ident.id]
    assert repo.get_submission(sid)["meta"][G.PEOPLE_CHECKED] == [ident.id]
    assert review.mark_people_checked(repo, sid, ["cam1"]) == []          # once is enough
    assert repo.get_submission(sid)["meta"]["cameras"] == {"cam1": "/x"}   # the rest is kept
    rows = repo.conn.execute("SELECT action FROM audit_log WHERE action='people_checked'").fetchall()
    assert len(rows) == 1


def test_a_checked_person_is_printed_with_the_note_and_no_clue_prints_everyone(repo, monkeypatch):
    from pfb_app.services import review
    aid = video_assignment(repo)
    a = add_video_subject(repo, aid, "#8", pfb_text="一人目の文です。", approved=False)
    add_video_subject(repo, aid, "#9", pfb_text="二人目の文です。", approved=False)
    cor.set_mode(repo, aid, cor.CORRECT_LATER)
    monkeypatch.setattr(G, "assess", lambda *a_, **k: _verdict(G.DOUBT)())
    # nothing in the work tells people apart: everyone, with the note on the page
    assert not G.work_has_clue(repo, repo.get_assignment(aid))
    text = text_of(render_student_pdf(repo, aid))
    assert "一人目の文です" in text and "二人目の文です" in text
    assert "背番号の読み取りも「人を確かめる」の記録もない作業のため" in text
    # the evaluator decided the first group's people: that group's person is checked, and
    # now the work has a clue, so the person nobody checked is left out
    review.mark_people_checked(repo, a["sid"])
    assert G.work_has_clue(repo, repo.get_assignment(aid))
    text = text_of(render_student_pdf(repo, aid))
    assert "一人目の文です" in text and "二人目の文です" not in text
    assert "「人を確かめる」で確かめた人です" in text
