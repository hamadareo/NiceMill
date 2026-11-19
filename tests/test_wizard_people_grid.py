"""Step 3's mapping grid for many people (Task 12), AppTest on ui/app.py.

A sports work with 12 tracked people, an 8-player roster and a canned names run (number reads,
two teams plus referees, crops and team strips). Only synthetic data: tiny black PNGs.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline import naming
from pfb_app.pipeline.tracked_identity import OUTSIDE
from pfb_app.services import review
from pfb_app.services.review import AUTO, OPP
from pfb_app.ui.wizard import people_grid, step_people
from test_wizard_ui_basic import app, env, html, ok, screen_text, shown_step  # noqa: F401

LABELS = [f"P{i}" for i in range(1, 13)]
ROSTER = [("r1", "山田", "10"), ("r2", "佐藤", "7"), ("r3", "鈴木", "5"), ("r4", "高田", "9"),
          ("r5", "伊藤", "11"), ("r6", "渡辺", "3"), ("r7", "中村", "4"), ("r8", "小林", "8")]
NUMBERS = {"P1": "10", "P2": "7", "P3": "5", "P6": "9"}
TEAMS = {**{f"P{i}": "T1" for i in range(1, 6)}, **{f"P{i}": "T2" for i in range(6, 11)},
         "P11": "other", "P12": "other"}
DECIDE = "これで決定して次へ"
P_LABEL = re.compile(r"\bP\d+\b")


def png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((16, 8, 3), dtype=np.uint8))
    return str(path)


def save(repo, sid, stage, output, camera="cam1", upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def sports_work(repo, groups=("A",)):
    aid = repo.create_assignment("練習試合", Kind.video, "練習試合", "", "standard", True,
                                 domain="sports", sport="soccer")
    sids = []
    for g in groups:
        sid = repo.create_submission(aid, f"{g}戦", {
            "cameras": {"cam1": "/v/cam1.mp4"}, "engine": "pfb3x3", "start_sec": 0.0,
            "end_sec": None, "auto_transcribe": False, "check": review.CHECK_DONE})
        for pid, name, number in ROSTER:
            repo.add_subject(sid, pid, name, None, {}, {"name": name, "number": number})
        sids.append(sid)
    return aid, sids


def tracked(settings, repo, sid, labels=LABELS):
    sheet = png(settings.data_dir / f"sheet_{sid}.png")
    prep = save(repo, sid, Stage.prepare, {"frames": []})
    return save(repo, sid, Stage.track, {"people": len(labels), "labels": labels, "sheet": sheet,
                                         "default_names": {}, "default_source": None},
                upstream=[prep.id])


def read(number):
    return {"number": number, "votes": 3, "reads": 3, "frames_agree": 3, "frames": [],
            "all": [], "failed": 0, "skipped": {}}


def names_run(settings, repo, sid, tr, teams=TEAMS, team_reason=""):
    labels = tr.output["labels"]
    crops = {l: png(settings.data_dir / f"crop_{sid}_{l}.png") for l in labels}
    strips = ({t: png(settings.data_dir / f"strip_{sid}_{t}.png") for t in ("T1", "T2")}
              if teams else {})
    cues = {"number": {l: read(n) for l, n in NUMBERS.items()}}
    status = {"number": {"status": "ok"}}
    if teams is not None:
        cues["team"] = {"teams": teams, "reason": team_reason, "crops": crops,
                        "strips": strips, "sizes": {}, "other_dist": None}
        status["team"] = {"status": "ok"}
    return save(repo, sid, Stage.names, {
        "track_run_id": tr.id, "labels": labels, "teams": dict(teams or {}),
        "team_reason": team_reason, "crops": crops, "strips": strips, "cues": cues,
        "status": status, "disabled": {}, "gate": {"number": {"enabled": True,
                                                              "threshold": None}},
        "model": "qwen3vl-8b", "roster_floor": 0,
        "roster_fingerprint": naming.roster_fingerprint(repo.list_subjects(sid))},
        upstream=[tr.id])


def scene(env, *, teams=TEAMS, team_reason="", groups=("A",), labels=LABELS, names=True):
    settings, repo = env
    aid, sids = sports_work(repo, groups)
    runs = {}
    for sid in sids:
        runs[sid] = tr = tracked(settings, repo, sid, labels)
        if names:
            names_run(settings, repo, sid, tr, teams=teams, team_reason=team_reason)
    return aid, sids, runs


def box(at, tr, label):
    return at.selectbox(key=f"pfb_map_{tr.id}_{label}")


def values(at, tr):
    return {l: box(at, tr, l).value for l in tr.output["labels"]}


def bulk(at, tr):
    return at.button(key=f"pfb_bulk_{tr.id}")


def shown_labels(at, tr):
    prefix = f"pfb_map_{tr.id}_"
    return [w.key[len(prefix):] for w in at.selectbox if (w.key or "").startswith(prefix)]


def primary(at):
    return next(b for b in at.button if b.proto.type == "primary")


def button(at, label):
    return next(b for b in at.button if b.label == label)


def tick_unmapped(at):
    for c in at.checkbox:
        if (c.key or "").startswith("pfb_unmapped_"):
            c.check()
    return at.run()


def rendered(at) -> str:
    """screen_text plus the text of every html element (tags stripped) and aria-labels."""
    page = html(at)
    return "\n".join([screen_text(at), re.sub(r"<[^>]+>", " ", page),
                      *re.findall(r'(?:aria-label|title)="([^"]*)"', page)])


# --- the grid ------------------------------------------------------------------------------

def test_grid_shows_estimate_chip_and_reason(env):
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert shown_labels(at, tr) == LABELS                    # one card per person
    page = html(at)
    assert "背番号10から推定" in page and "背番号9から推定" in page
    assert page.count(">推定</span>") == 4                   # P1, P2, P3, P6
    assert "1番の人" in page and "12番の人" in page
    v = values(at, tr)
    assert v["P1"] == "r1" and v["P2"] == "r2" and v["P6"] == "r4"
    assert v["P4"] is None and v["P11"] == OUTSIDE           # referees: 対象外
    assert box(at, tr, "P1").label == "1番の人"
    assert len(at.get("image")) >= 12                        # the crops
    assert step_people.SUGGESTED in screen_text(at)
    assert "初期位置から推定した" not in screen_text(at)
    # changing a proposal keeps the reason in view, no longer as the chip
    box(at, tr, "P1").select(OUTSIDE)
    ok(at.run())
    assert html(at).count(">推定</span>") == 3
    assert "山田（背番号10から推定）" in html(at)


def test_filter_unassigned_shows_only_unset(env):
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"pfb_filter_{tr.id}").set_value("unset")
    ok(at.run())
    assert shown_labels(at, tr) == ["P4", "P5", "P7", "P8", "P9", "P10"]
    box(at, tr, "P4").select(OUTSIDE)
    ok(at.run())
    assert shown_labels(at, tr) == ["P5", "P7", "P8", "P9", "P10"]
    at.radio(key=f"pfb_filter_{tr.id}").set_value("guess")
    ok(at.run())
    assert shown_labels(at, tr) == ["P1", "P2", "P3", "P6"]
    at.radio(key=f"pfb_filter_{tr.id}").set_value("outside")
    ok(at.run())
    assert shown_labels(at, tr) == ["P4", "P11", "P12"]
    at.radio(key=f"pfb_filter_{tr.id}").set_value("all")
    ok(at.run())
    assert values(at, tr)["P4"] == OUTSIDE                   # kept while it was hidden


def test_bulk_outside_sets_rest(env):
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    box(at, tr, "P5").select("r5")
    ok(at.run())
    b = bulk(at, tr)
    assert b.proto.type == "secondary"
    assert b.label == "残りを対象外にする（5人）"           # P4 P7 P8 P9 P10
    b.click()
    ok(at.run())
    v = values(at, tr)
    assert v["P5"] == "r5" and v["P1"] == "r1"               # chosen ones stay
    assert all(v[l] == OUTSIDE for l in ("P4", "P7", "P8", "P9", "P10", "P11", "P12"))
    assert repo_identity(env, sid) is None                   # nothing saved yet


def repo_identity(env, sid):
    return env[1].latest_run(sid, Stage.identity, "cam1")


def test_name_a_person_creates_subject_and_selects_it(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    box(at, tr, "P2").select("r3")                          # an unsaved choice elsewhere
    ok(at.run())
    at.text_input(key=f"pfb_name_{tr.id}_P4").input("高橋")
    at.button(key=f"pfb_nameadd_{tr.id}_P4").click()
    ok(at.run())
    made = [s for s in repo.list_subjects(sid) if s["display_name"] == "高橋"]
    assert len(made) == 1
    pid = made[0]["participant_id"]
    assert box(at, tr, "P4").value == pid
    assert "高橋" in box(at, tr, "P4").options
    assert box(at, tr, "P2").value == "r3"                  # not thrown away
    # a name the work already has is refused, in words
    at.text_input(key=f"pfb_name_{tr.id}_P5").input("山田")
    at.button(key=f"pfb_nameadd_{tr.id}_P5").click()
    ok(at.run())
    assert any("同じ名前" in e.value for e in at.error)
    assert box(at, tr, "P5").value is None


def test_a_refused_name_is_shown_as_typed(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    review.add_named_subject(repo, sid, "生徒会長")         # school words inside a typed name
    at = ok(app(session={"assignment_id": aid}))
    at.text_input(key=f"pfb_name_{tr.id}_P5").input("生徒会長")
    at.button(key=f"pfb_nameadd_{tr.id}_P5").click()
    ok(at.run())
    errors = [e.value for e in at.error]
    assert review.NAME_TAKEN + "生徒会長" in errors, errors   # never 「選手会長」


def test_team_answer_changes_prefill(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    team = at.radio(key=f"pfb_team_{tr.id}")
    assert team.label == people_grid.TEAM_QUESTION
    assert team.value is None                                # not answered yet
    assert team.options == ["左", "右", "分けない"]
    assert people_grid.TEAM_QUESTION in html(at)
    box(at, tr, "P8").select("r8")                          # a choice of the coach's own
    ok(at.run())
    v = values(at, tr)
    assert v["P6"] == "r4" and v["P7"] is None
    at.radio(key=f"pfb_team_{tr.id}").set_value("T1")
    ok(at.run())
    own = repo.get_submission(sid)["meta"]["own_team"]["cam1"]
    assert own["labels"] == ["P1", "P2", "P3", "P4", "P5"]
    v = values(at, tr)
    assert v["P6"] == OUTSIDE and v["P7"] == OUTSIDE        # the other team: 対象外
    assert v["P1"] == "r1"
    assert v["P8"] == "r8"                                  # the coach's choice is kept
    assert at.radio(key=f"pfb_team_{tr.id}").value == "T1"
    assert html(at).count(">推定</span>") == 3              # P6's guess is the other team's
    # 分けない is an answer of its own: the defaults come back, and it stays picked
    at.radio(key=f"pfb_team_{tr.id}").set_value("none")
    ok(at.run())
    assert values(at, tr)["P7"] is None
    meta = repo.get_submission(sid)["meta"]
    assert "cam1" not in meta.get("own_team", {})
    assert meta[people_grid.NO_TEAMS] == {"cam1": tr.id}
    assert at.radio(key=f"pfb_team_{tr.id}").value == "none"
    ok(at.run())
    assert at.radio(key=f"pfb_team_{tr.id}").value == "none"


def test_team_reason_is_said_in_plain_words(env):
    aid, (sid,), runs = scene(env, teams={}, team_reason="one_colour")
    at = ok(app(session={"assignment_id": aid}))
    assert not [r for r in at.radio if (r.key or "").startswith("pfb_team_")]
    assert people_grid.TEAM_REASONS["one_colour"] in screen_text(at)
    assert "one_colour" not in rendered(at)


def test_team_reason_not_person_is_silent(env):
    aid, (sid,), runs = scene(env, teams={}, team_reason="not_person")
    at = ok(app(session={"assignment_id": aid}))
    text = rendered(at)
    assert "not_person" not in text and "チーム分け" not in text
    assert people_grid.TEAM_QUESTION not in text


def test_opponents_option_names_opponents(env):
    settings, repo = env
    aid, (a, b), runs = scene(env, groups=("A", "B"))
    tr = runs[a]
    at = ok(app(session={"assignment_id": aid}))
    opp = at.checkbox(key=f"pfb_opp_{a}")
    assert opp.label == people_grid.OPPONENTS and opp.value is False   # off by default
    at.radio(key=f"pfb_team_{tr.id}").set_value("T1")
    ok(at.run())
    at.checkbox(key=f"pfb_opp_{a}").check()
    ok(at.run())
    # every group of the work
    assert all(repo.get_submission(s)["meta"][review.EVALUATE_OPPONENTS] for s in (a, b))
    v = values(at, tr)
    assert [v[f"P{i}"] for i in range(6, 11)] == [OPP + f"P{i}" for i in range(6, 11)]
    assert v["P4"] == AUTO + "P4" and v["P11"] == OUTSIDE
    shown = {l: box(at, tr, l).options[box(at, tr, l).index] for l in ("P4", "P6", "P10")}
    assert shown == {"P4": "選手（番号なし）1", "P6": "相手選手1", "P10": "相手選手5"}
    ok(tick_unmapped(at))
    primary(at).click()
    ok(at.run())
    names = {s["display_name"] for s in repo.list_subjects(a)}
    assert {f"相手選手{i}" for i in range(1, 6)} <= names and "選手（番号なし）1" in names
    ident = repo_identity(env, a)
    assert ident is not None and repo.get_submission(a)["identity_confirmed"]
    # off again: the opponent options go away for the other group as well
    at2 = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": b}))
    at2.checkbox(key=f"pfb_opp_{b}").uncheck()
    ok(at2.run())
    assert not any(repo.get_submission(s)["meta"][review.EVALUATE_OPPONENTS] for s in (a, b))
    assert not any(o.startswith("相手選手") for o in box(at2, runs[b], "P7").options)


def test_decide_enabled_while_names_pending(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    repo.enqueue(sid, Stage.names, "pfb3x3:standard", camera="cam1")
    at = ok(app(session={"assignment_id": aid}))
    assert people_grid.NAMES_PENDING in screen_text(at)
    assert not primary(at).disabled
    bulk(at, tr).click()
    ok(at.run())
    ok(tick_unmapped(at))
    assert not primary(at).disabled
    primary(at).click()
    ok(at.run())
    assert repo_identity(env, sid) is not None
    assert repo.get_submission(sid)["identity_confirmed"]
    assert shown_step(at) == 4


def test_roster_change_requests_names(env):
    settings, repo = env
    gate = settings.engine_dir / "configs" / "cue_gate.yaml"
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(yaml.safe_dump({"qwen3vl-8b": {"number": {
        "enabled": True, "threshold": None, "reason": "ok"}}}), encoding="utf-8")
    aid, (sid,), runs = scene(env)
    at = ok(app(session={"assignment_id": aid}))
    assert any(e.label == "選手名簿を入れる・直す" for e in at.expander)
    at.text_area(key=f"pfb_roster_{sid}_text").input("背番号,名前\n10,山田\n23,木村\n")
    assert "選び直しになります" not in screen_text(at)            # nothing decided yet
    ok(at.run())
    apply = at.button(key=f"pfb_roster_apply_{sid}")
    assert apply.label == "この選手名簿を使う" and apply.proto.type == "secondary"
    apply.click()
    ok(at.run())
    names = {s["display_name"] for s in repo.list_subjects(sid)}
    assert names == {"山田", "木村"}
    queued = repo.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE submission_id=? AND stage=? AND status='queued'",
        (sid, Stage.names.value)).fetchone()[0]
    assert queued == 1
    assert people_grid.NAMES_PENDING in screen_text(at)


def test_no_p_labels_in_rendered_text(env):
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    assert not P_LABEL.search(rendered(at)), P_LABEL.search(rendered(at))
    primary(at).click()                                      # people still unset
    ok(at.run())
    errors = [e.value for e in at.error]
    assert any("まだ選んでいない人がいます: 4番の人、5番の人" in e for e in errors), errors
    assert not P_LABEL.search(rendered(at)), P_LABEL.search(rendered(at))
    # with the opponents on, every option is a name too
    at.radio(key=f"pfb_team_{tr.id}").set_value("T1")
    ok(at.run())
    at.checkbox(key=f"pfb_opp_{sid}").check()
    ok(at.run())
    assert not P_LABEL.search(rendered(at)), P_LABEL.search(rendered(at))


def test_messages_name_people_by_their_number():
    assert people_grid.marks("人物の追跡結果にない番号です: P7") == \
        "人物の追跡結果にない番号です: 7番の人"
    assert people_grid.marks("同じ人物が複数回あります: P3、P12") == \
        "同じ人物が複数回あります: 3番の人、12番の人"
    assert people_grid.marks("PFB") == "PFB"


# --- fix round 1 ----------------------------------------------------------------------------

def test_proposals_landing_keep_picks_and_fill_untouched_people(env):
    settings, repo = env
    aid, (sid,), runs = scene(env, names=False)
    tr = runs[sid]
    repo.enqueue(sid, Stage.names, "pfb3x3:standard", camera="cam1")
    at = ok(app(session={"assignment_id": aid}))
    assert people_grid.NAMES_PENDING in screen_text(at)
    assert values(at, tr)["P1"] is None
    box(at, tr, "P2").select("r6")                          # the coach's own pick
    ok(at.run())
    box(at, tr, "P8").select("r8")
    ok(at.run())
    # the names job finishes while the screen is open (the live part reruns the page)
    names_run(settings, repo, sid, tr)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    ok(at.run())
    v = values(at, tr)
    assert v["P1"] == "r1" and v["P3"] == "r3" and v["P6"] == "r4"   # filled in
    assert v["P2"] == "r6" and v["P8"] == "r8"                       # kept
    assert people_grid.NAMES_PENDING not in screen_text(at)


def test_the_watch_reruns_when_names_land(env):
    settings, repo = env
    aid, (sid,), runs = scene(env, names=False)
    tr = runs[sid]
    repo.enqueue(sid, Stage.names, "pfb3x3:standard", camera="cam1")
    from pfb_app.ui.wizard.progress import View
    signature = step_people._signature(aid, sid, form=True)
    base = signature(View.load(repo, aid))
    names_run(settings, repo, sid, tr)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    assert signature(View.load(repo, aid)) != base


def test_thirty_people_render_and_decide(env):
    settings, repo = env
    labels = [f"P{i}" for i in range(1, 31)]
    teams = {l: ("T1" if i < 13 else "T2" if i < 26 else "other")
             for i, l in enumerate(labels)}
    aid, (sid,), runs = scene(env, labels=labels, teams=teams)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    assert shown_labels(at, tr) == labels
    assert len([c for c in at.get("image")]) >= 30
    assert primary(at).label == DECIDE and len([b for b in at.button
                                                if b.proto.type == "primary"]) == 1
    bulk(at, tr).click()
    ok(at.run())
    ok(tick_unmapped(at))
    primary(at).click()
    ok(at.run())
    ident = repo_identity(env, sid)
    assert len(ident.output["participants"]) == 30
    assert repo.get_submission(sid)["identity_confirmed"]


def _expander(at, label):
    return next(e for e in at.expander if e.label == label)


def test_the_sheet_and_the_card_pictures_keep_their_place(env):
    settings, repo = env
    aid, (sid,), runs = scene(env, names=False)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    sheet = _expander(at, people_grid.SHEET_MORE)
    assert len(sheet.get("image")) == 1 and sheet.proto.expanded   # open: no crops
    assert html(at).count(people_grid.NO_PHOTO) == len(LABELS)    # a box where each crop goes
    names_run(settings, repo, sid, tr)                             # the crops arrive
    ok(at.run())
    sheet = _expander(at, people_grid.SHEET_MORE)
    assert len(sheet.get("image")) == 1 and not sheet.proto.expanded
    assert people_grid.NO_PHOTO not in html(at)
    assert len(at.get("image")) == len(LABELS) + 1 + 2             # crops, sheet, strips


def test_the_guess_chip_is_a_theme_chip():
    from pfb_app.ui.wizard import theme
    chip = theme.status_chip("guess", "推定")
    assert 'class="pfb-chip pfb-chip--guess"' in chip and ">推定</span>" in chip
    assert ".pfb-chip--guess" in theme.theme_css()


def test_the_step_card_is_wider_on_this_screen():
    from pfb_app.ui.wizard import theme
    css = theme.theme_css()
    assert '.stApp [class*="st-key-pfb_step_card"].st-key-pfb_step_card_people' in css
    assert "max-width: 76rem" in css


def test_a_decided_camera_shows_the_team_answer_read_only(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    review.set_team_choice(repo, sid, "cam1", "T2")
    mapping = {l: None for l in LABELS}
    review.save_tracked_identity(repo, sid, "cam1", mapping)
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": sid}))
    team = at.radio(key=f"pfb_team_{tr.id}")
    assert team.value == "T2" and team.disabled
    # the roster expander says a new roster resets the decided camera
    assert people_grid.ROSTER_RESETS.replace("名簿", "選手名簿") in screen_text(at)


def test_opponents_only_with_teams_and_decided_groups_keep_theirs(env):
    settings, repo = env
    aid, (a,), runs = scene(env, teams={}, team_reason="one_colour")
    at = ok(app(session={"assignment_id": aid}))
    assert not [c for c in at.checkbox if (c.key or "").startswith("pfb_opp_")]
    aid, (b, c), runs = scene(env, groups=("B", "C"))
    repo.conn.execute("UPDATE submissions SET identity_confirmed=1 WHERE id=?", (b,))
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": c}))
    assert at.checkbox(key=f"pfb_opp_{c}")
    assert people_grid.OPPONENTS_KEPT in screen_text(at)


def test_a_roster_reset_says_so_rather_than_retracked(env):
    from pfb_app.services.roster import apply_roster, parse_roster_text
    settings, repo = env
    aid, (sid,), runs = scene(env)
    review.save_tracked_identity(repo, sid, "cam1", {l: None for l in LABELS})
    apply_roster(repo, sid, parse_roster_text("背番号,名前\n10,山田\n23,木村\n"), "sports")
    at = ok(app(session={"assignment_id": aid, "step": 3, "submission_id": sid}))
    warnings = [w.value for w in at.warning]
    assert any(step_people.ROSTER_RESET.replace("名簿", "選手名簿") in w for w in warnings)
    assert not any("追跡がやり直された" in w for w in warnings)


def test_a_busy_roster_change_says_so_in_plain_words(env):
    settings, repo = env
    aid, (sid,), runs = scene(env)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    at = ok(app(session={"assignment_id": aid}))
    at.text_area(key=f"pfb_roster_{sid}_text").input("背番号,名前\n10,山田\n23,木村\n")
    ok(at.run())
    at.button(key=f"pfb_roster_apply_{sid}").click()
    ok(at.run())
    errors = [e.value for e in at.error]
    assert any(people_grid.ROSTER_BUSY.replace("名簿", "選手名簿") in e for e in errors), errors
    assert not any("ジョブ" in e for e in errors)
    assert {s["display_name"] for s in repo.list_subjects(sid)} >= {"佐藤", "鈴木"}   # kept


def test_name_it_says_who_it_is_for(env):
    aid, (sid,), runs = scene(env)
    at = ok(app(session={"assignment_id": aid}))
    helps = [p.proto.popover.help for p in at.get("popover")
             if p.proto.popover.label not in ("ほかの段階", "アカウント")]   # the phone
    # stepper's own popover, and the sidebar's account box (ui/auth_ui)
    assert len(helps) == len(LABELS)
    assert "4番の人に名前をつける" in helps


def test_step_3_never_shows_the_service_words_for_tracking():
    """services/review.py and tracked_identity.py say 人物の追跡; step 3 shows 人を見つける."""
    from pfb_app.ui.wizard.progress import plain_service
    for text in ("人物の追跡結果がありません（cam1）。先に人物の追跡を実行してください。",
                 "人物の追跡が実行中または待機中です（cam2）。完了後に確認してください。",
                 "人物の追跡がやり直されています。最新の人物一覧で対応づけを保存してください。",
                 "人物の追跡結果にない番号です: P9",
                 "トラッキング方式では、人物の追跡からやり直してください。"):
        assert "人物の追跡" not in plain_service(text), text
    src = (Path(__file__).parents[1] / "src/pfb_app/ui/wizard/step_people.py").read_text(
        encoding="utf-8")
    assert src.count("plain_service(") >= 3


# --- what a track run's warnings say -------------------------------------------------------

RAW_SHEET = "開始直後に全員が写っている画面が見つからないため、人物一覧は既定の画面で作りました"
RAW_AUDIO = "音声から発言が見つからなかったため、映像だけで観察します"
RAW_FAIL = "文字起こしに失敗したため、映像だけで観察します: Traceback in whisper_python"


def test_track_notes_are_plain_sentences_and_unknown_warnings_are_dropped():
    notes = step_people.track_notes([RAW_SHEET, RAW_AUDIO, RAW_FAIL, RAW_SHEET, "internal: x"])
    assert notes == ["全員が同時に映る場面がなかったため、写真に写っていない人がいます。",
                     "音声に聞き取れる話し声がなかったため、映像だけで観察します。",
                     "音声を文字にできなかったため、映像だけで観察します。"]
    assert step_people.track_notes([]) == []


def test_the_grid_shows_the_plain_notice_not_the_raw_warning(env):
    import json
    settings, repo = env
    aid, (sid,), runs = scene(env)
    repo.conn.execute("UPDATE stage_runs SET warnings=? WHERE id=?",
                      (json.dumps([RAW_SHEET, RAW_FAIL]), runs[sid].id))
    at = ok(app(session={"assignment_id": aid}))
    page = rendered(at)
    assert "全員が同時に映る場面がなかったため" in page
    assert "音声を文字にできなかったため" in page
    assert "全員が写っている画面が見つからない" not in page and "Traceback" not in page
