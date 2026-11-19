"""Name fusion (pure): cue candidates -> summed scores -> Hungarian -> threshold and margin."""
import itertools
import random
from dataclasses import dataclass, field

import pytest

from pfb_app.pipeline.name_fusion import (
    CUE_SCORES, MARGIN, SHARED_FACTOR, THRESHOLD, Candidate, Proposal, batting_anchors,
    fuse, hungarian,
    match_team_names, mapped_candidates, normalise_number, number_candidates,
    order_candidates, position_candidates, reason_text, restrict_to_team,
)


@dataclass
class Entry:
    """Stand-in for services.roster.RosterEntry (participant_id, display_name, attrs)."""
    participant_id: str
    display_name: str
    attrs: dict = field(default_factory=dict)


def _e(pid, display, **attrs):
    return Entry(pid, display, {k: str(v) for k, v in attrs.items()})


def _read(number, agree=2, votes=None):
    return {"number": number, "votes": votes if votes is not None else agree,
            "frames_agree": agree}


# ---------------------------------------------------------------- constants and reasons

def test_constants():
    assert CUE_SCORES == {"number": 1.0, "place": 1.0, "position": 0.6, "grid": 0.6,
                          "appearance": 0.45, "batting": 0.3}
    assert THRESHOLD == 0.6 and MARGIN == 0.3 and SHARED_FACTOR == 0.5


def test_reason_texts():
    assert reason_text("number", "10") == "背番号10から推定"
    assert reason_text("position", "") == "守備位置から推定"
    assert reason_text("grid", "") == "スタート位置から推定"
    assert reason_text("batting", "") == "打順から推定（弱い）"
    assert reason_text("appearance", "") == "外見から推定"
    assert reason_text("place", "") == "初期位置から推定"


def test_vehicle_reason():
    assert reason_text("number", "7", target="vehicle") == "番号7から推定"
    assert reason_text("grid", "", target="vehicle") == "スタート位置から推定"
    entries = [_e("r1", "A", number=7)]
    out = fuse(number_candidates({"P1": _read("7")}, entries), ["P1"], ["r1"],
               target="vehicle")
    assert out["P1"].reasons == ("番号7から推定",)


def test_normalise_number():
    assert normalise_number("#07") == "7"
    assert normalise_number("１０") == "10"
    assert normalise_number("No. 5") == "5"
    assert normalise_number("00") == "00"
    assert normalise_number("0") == "0"
    assert normalise_number("") is None
    assert normalise_number("ab") is None
    assert normalise_number(None) is None
    # Excel cells read as floats
    assert normalise_number("10.0") == "10" and normalise_number("7.00") == "7"
    assert normalise_number(10.0) == "10" and normalise_number("00") == "00"
    assert normalise_number("10.5") is None


# ---------------------------------------------------------------- number cue

def test_number_match_proposes_with_reason():
    entries = [_e("r1", "山田", number=10), _e("r2", "佐藤", number=4)]
    cands = number_candidates({"P1": _read("10"), "P2": _read(None, agree=1)}, entries)
    assert cands == [Candidate("P1", "r1", "number", 1.0, "10")]
    out = fuse(cands, ["P1", "P2"], ["r1", "r2"])
    assert out == {"P1": Proposal("r1", 1.0, ("背番号10から推定",))}


def test_number_needs_two_agreeing_frames():
    entries = [_e("r1", "山田", number=10)]
    assert number_candidates({"P1": _read("10", agree=1, votes=3)}, entries) == []
    assert len(number_candidates({"P1": _read("10", agree=2)}, entries)) == 1
    # the caller may ask for more agreeing frames (the gate's threshold)
    assert number_candidates({"P1": _read("10", agree=2)}, entries, min_frames_agree=3) == []
    assert len(number_candidates({"P1": _read("10", agree=3)}, entries,
                                 min_frames_agree=3)) == 1


def test_number_read_without_frames_agree_is_no_read_and_warned(caplog):
    entries = [_e("r1", "山田", number=10)]
    with caplog.at_level("WARNING", logger="pfb_app.pipeline.name_fusion"):
        got = number_candidates({"P1": "10", "P2": {"number": "10", "votes": 3},
                                 "P3": None}, entries)
    assert got == []
    warned = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 1 and "frames_agree" in warned[0].getMessage()


def test_number_00_distinct_from_0_and_roster_normalised():
    entries = [_e("r1", "A", number="#00"), _e("r2", "B", number="０"),
               _e("r3", "C", number="07")]
    reads = {"P1": _read("00"), "P2": _read("0"), "P3": _read("7")}
    got = {(c.label, c.row) for c in number_candidates(reads, entries)}
    assert got == {("P1", "r1"), ("P2", "r2"), ("P3", "r3")}


def test_same_number_both_teams_no_proposal_without_team():
    entries = [_e("r1", "A", number=10, team="東"), _e("r2", "B", number=10, team="西")]
    cands = number_candidates({"P1": _read("10")}, entries)
    assert len(cands) == 2
    assert fuse(cands, ["P1"], ["r1", "r2"]) == {}


def test_same_number_both_teams_proposed_with_own_team():
    entries = [_e("r1", "A", number=10, team="東"), _e("r2", "B", number=10, team="西"),
               _e("r3", "C", number=4, team="東"), _e("r4", "D", number=5, team="西")]
    reads = {"P1": _read("10"), "P2": _read("4"), "P3": _read("5")}
    teams = {"P1": "T1", "P2": "T1", "P3": "T2"}
    names = match_team_names(reads, entries, teams)
    assert names == {"T1": "東", "T2": "西"}
    cands = restrict_to_team(number_candidates(reads, entries), teams, "T1", names, entries)
    out = fuse(cands, ["P1", "P2", "P3"], ["r1", "r2", "r3", "r4"])
    assert out["P1"].row == "r1" and out["P2"].row == "r3" and "P3" not in out


def test_restrict_to_team_label_set_and_none():
    entries = [_e("r1", "A", number=10, team="東"), _e("r2", "B", number=10, team="西")]
    cands = [Candidate("P1", "r1", "number", 1.0, "10"),
             Candidate("P1", "r2", "number", 1.0, "10"),
             Candidate("P2", "r2", "number", 1.0, "10")]
    teams = {"P1": "T2", "P2": "T1"}
    assert restrict_to_team(cands, teams, None, {}, entries) == cands
    # the chosen team as a label set (T1/T2 names may swap between runs)
    assert restrict_to_team(cands, teams, {"P1"}, {}, entries) == cands[:2]
    assert restrict_to_team(cands, teams, {"P1"}, {"T2": "東", "T1": "西"}, entries) == \
        cands[:1]
    # a cluster name: its labels
    assert restrict_to_team(cands, teams, "T1", {}, entries) == cands[2:]


def test_match_team_names():
    entries = [_e("r1", "A", number=1, team="東"), _e("r2", "B", number=2, team="東"),
               _e("r3", "C", number=3, team="西"), _e("r4", "D", number=4, team="西"),
               _e("r5", "E", number=9, team="東"), _e("r6", "F", number=9, team="西")]
    teams = {"P1": "T1", "P2": "T1", "P3": "T2", "P4": "T2", "P5": "T1", "P6": "other"}
    reads = {"P1": _read("1"), "P2": _read("2"), "P3": _read("3"), "P4": _read("9"),
             "P5": _read("3"), "P6": _read("4"), "P7": _read("1", agree=1)}
    # T1: 東 2 (1, 2) vs 西 1 (3); T2: 西 2 (3, 9) vs 東 1 (9); "other" is never named
    assert match_team_names(reads, entries, teams) == {"T1": "東", "T2": "西"}
    # no team column -> empty
    plain = [_e("r1", "A", number=1)]
    assert match_team_names({"P1": _read("1")}, plain, {"P1": "T1"}) == {}
    # a tie is no majority
    assert match_team_names({"P1": _read("9")}, entries, {"P1": "T1"}) == {}
    # both clusters on the same name: the stronger keeps it
    reads2 = {"P1": _read("1"), "P2": _read("2"), "P3": _read("1")}
    assert match_team_names(reads2, entries, {"P1": "T1", "P2": "T1", "P3": "T2"}) == \
        {"T1": "東"}


# ---------------------------------------------------------------- position / order / mapped

def test_position_candidates_unique_and_shared():
    entries = [_e("r1", "A", position="ＧＫ"), _e("r2", "B", position="DF"),
               _e("r3", "C", position="df"), _e("r4", "D", position="FW")]
    pos = {"P1": "GK", "P2": "DF", "P3": "DF", "P4": "DF", "P5": "MF"}
    cands = position_candidates(pos, entries, slots={"GK": 1, "DF": 2, "FW": 1},
                                claims={"GK": 1, "DF": 3})
    by = {(c.label, c.row): c.score for c in cands}
    assert by[("P1", "r1")] == pytest.approx(0.6)
    w = 0.6 * (2 / 3) * SHARED_FACTOR
    for lab in ("P2", "P3", "P4"):
        assert by[(lab, "r2")] == pytest.approx(w) and by[(lab, "r3")] == pytest.approx(w)
    assert len(by) == 7
    # without slots/claims they are counted from the roster and the answers
    cands = position_candidates({"P1": "gk", "P2": "DF"}, entries)
    by = {(c.label, c.row): c.score for c in cands}
    assert by == {("P1", "r1"): pytest.approx(0.6),
                  ("P2", "r2"): pytest.approx(0.6 * SHARED_FACTOR),
                  ("P2", "r3"): pytest.approx(0.6 * SHARED_FACTOR)}


def test_grid_order_candidates():
    entries = [_e("r1", "A", grid=2), _e("r2", "B", grid=1), _e("r3", "C", grid="x")]
    cands = order_candidates(["P5", "P3", "P9"], entries, "grid")
    assert {(c.label, c.row, c.cue, c.score) for c in cands} == {
        ("P5", "r2", "grid", 0.6), ("P3", "r1", "grid", 0.6)}


def _lineup(n=9):
    return [_e(f"r{k}", f"B{k}", batting=k) for k in range(1, n + 1)]


def test_batting_relative_order_follows_anchors():
    entries = _lineup()
    # the video starts at the 4th batter: P1 bats 4th, P2 5th, ...
    order = [f"P{i}" for i in range(1, 10)]
    anchors = {"P1": "r4", "P3": "r6"}
    cands = order_candidates(order, entries, "batting", anchors=anchors)
    got = {c.label: c.row for c in cands}
    # the anchored people get no batting candidate of their own
    assert "P1" not in got and "P3" not in got
    assert got["P2"] == "r5" and got["P6"] == "r9"
    assert got["P7"] == "r1" and got["P9"] == "r3"
    assert all(c.score == CUE_SCORES["batting"] for c in cands)
    # no anchors, or only one: no batting candidates
    assert order_candidates(order, entries, "batting") == []
    assert order_candidates(order, entries, "batting", anchors={"P1": "r4"}) == []
    # anchors that do not agree on one rotation: no batting candidates
    assert order_candidates(order, entries, "batting",
                            anchors={"P1": "r4", "P2": "r1", "P3": "r8"}) == []
    assert order_candidates(order, entries, "batting",
                            anchors={"P1": "r4", "P3": "r7"}) == []


def test_batting_skipped_batter_leaves_the_gap_open():
    entries = _lineup()
    # batters 4, 5, 6, 7, (8 missed), 9, 1 were seen
    order = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    cands = order_candidates(order, entries, "batting",
                             anchors={"Q1": "r4", "Q3": "r6", "Q6": "r1"})
    # Q2 sits between two anchors on the agreed rotation; Q4, Q5 and Q7 are next to the
    # anchor after the missed batter
    assert {c.label: c.row for c in cands} == {"Q2": "r5"}


def test_batting_anchors_from_other_cues():
    entries = [_e(f"r{k}", f"B{k}", batting=k, number=k + 10) for k in range(1, 6)]
    cands = number_candidates({"P1": _read("13"), "P2": _read("14")}, entries)
    anchors = batting_anchors(cands, ["P1", "P2", "P3"], [e.participant_id for e in entries])
    assert anchors == {"P1": "r3", "P2": "r4"}
    got = {c.label: c.row for c in order_candidates(["P1", "P2", "P3"], entries, "batting",
                                                    anchors=anchors)}
    assert got == {"P3": "r5"}


def test_mapped_candidates():
    entries = [_e("r1", "山田 太郎", name="山田 太郎"), _e("r2", "Sato", name="Sato"),
               _e("r3", "dup", name="dup"), _e("r4", "dup", name="dup")]
    cands = mapped_candidates({"P1": "山田　太郎", "P2": "sato", "P3": "dup", "P4": "誰"},
                              entries, "appearance")
    assert cands == [Candidate("P1", "r1", "appearance", 0.45),
                     Candidate("P2", "r2", "appearance", 0.45)]
    cands = mapped_candidates({"P1": "r2"}, entries, "place")
    assert cands == [Candidate("P1", "r2", "place", 1.0)]


def test_mapped_candidates_match_the_roster_name_not_the_display_name():
    entries = [_e("r1", "選手3", number=10),                  # no name: a default name
               _e("r2", "#11", number=11),                   # number only
               _e("r3", "山田（#10）", name="山田", number=10),  # disambiguated
               _e("r4", "山田（#12）", name="山田", number=12),
               _e("r5", "佐藤（#7）", name="佐藤", number=7)]
    names = {"P1": "選手3", "P2": "#11", "P3": "山田（#10）", "P4": "山田", "P5": "佐藤"}
    cands = mapped_candidates(names, entries, "appearance")
    # default and disambiguated display names never match; 山田 is two rows
    assert cands == [Candidate("P5", "r5", "appearance", 0.45)]


# ---------------------------------------------------------------- fusion

def test_batting_alone_never_passes():
    entries = _lineup(4)
    cands = order_candidates(["P1", "P2", "P3", "P4"], entries, "batting",
                             anchors={"P1": "r1", "P2": "r2"})
    assert {(c.label, c.row) for c in cands} == {("P3", "r3"), ("P4", "r4")}
    assert fuse(cands, ["P1", "P2", "P3", "P4"], ["r1", "r2", "r3", "r4"]) == {}


def test_shared_position_alone_never_passes():
    entries = [_e("r1", "A", position="DF"), _e("r2", "B", position="DF"),
               _e("r3", "C", position="GK")]
    cands = position_candidates({"P1": "DF", "P2": "GK"}, entries,
                                slots={"DF": 2, "GK": 1}, claims={"DF": 1, "GK": 1})
    assert max(c.score for c in cands if c.label == "P1") <= 0.3 + 1e-9
    out = fuse(cands, ["P1", "P2"], ["r1", "r2", "r3"])
    # a unique slot alone meets the threshold; a shared one never does
    assert set(out) == {"P2"} and out["P2"].row == "r3"


def test_position_plus_batting_passes_with_margin():
    entries = [_e("r1", "A", number=11, batting=1), _e("r2", "B", number=12, batting=2),
               _e("r3", "C", position="OF", batting=3), _e("r4", "D", position="OF", batting=4),
               _e("r5", "E", batting=5)]
    labels, rows = ["P1", "P2", "P3", "P4", "P5"], [e.participant_id for e in entries]
    cands = number_candidates({"P1": _read("11"), "P2": _read("12")}, entries) + \
        position_candidates({"P3": "OF", "P4": "OF"}, entries)
    anchors = batting_anchors(cands, labels, rows)
    assert anchors == {"P1": "r1", "P2": "r2"}
    cands += order_candidates(labels, entries, "batting", anchors=anchors)
    out = fuse(cands, labels, rows)
    # the shared OF slot (0.3 each) is split by the batting order (0.3)
    assert out["P3"] == Proposal("r3", pytest.approx(0.6),
                                 ("守備位置から推定", "打順から推定（弱い）"))
    assert out["P4"].row == "r4" and out["P1"].row == "r1" and "P5" not in out


def test_margin_blocks_close_runner_up():
    cands = [Candidate("P1", "r1", "appearance", 0.45), Candidate("P1", "r1", "batting", 0.3),
             Candidate("P1", "r2", "position", 0.6)]
    assert fuse(cands, ["P1"], ["r1", "r2"]) == {}
    # the runner-up in the row's column counts too
    cands = [Candidate("P1", "r1", "number", 1.0, "4"), Candidate("P2", "r1", "grid", 0.6),
             Candidate("P2", "r1", "batting", 0.3)]
    assert fuse(cands, ["P1", "P2"], ["r1"]) == {}
    assert fuse(cands, ["P1", "P2"], ["r1"], margin=0.05)["P1"].row == "r1"


def test_fuse_same_cue_twice_counts_once():
    cands = [Candidate("P1", "r1", "number", 1.0, "4"), Candidate("P1", "r1", "number", 1.0, "4")]
    out = fuse(cands, ["P1"], ["r1"])
    assert out["P1"].score == pytest.approx(1.0) and out["P1"].reasons == ("背番号4から推定",)


def test_fuse_ignores_unknown_labels_and_rows():
    cands = [Candidate("P1", "r9", "number", 1.0, "4"), Candidate("P8", "r1", "number", 1.0)]
    assert fuse(cands, ["P1"], ["r1"]) == {}


# ---------------------------------------------------------------- hungarian

def test_hungarian_matches_bruteforce():
    rng = random.Random(8)
    for _ in range(20):
        s = [[rng.random() for _ in range(6)] for _ in range(6)]
        pairs = hungarian(s)
        assert sorted(i for i, _ in pairs) == list(range(6))
        assert sorted(j for _, j in pairs) == list(range(6))
        best = max(sum(s[i][p[i]] for i in range(6)) for p in itertools.permutations(range(6)))
        assert sum(s[i][j] for i, j in pairs) == pytest.approx(best)


def test_hungarian_rectangular():
    rng = random.Random(3)
    for n, m in ((3, 5), (5, 3), (1, 4), (4, 1)):
        s = [[rng.random() for _ in range(m)] for _ in range(n)]
        pairs = hungarian(s)
        assert len(pairs) == min(n, m)
        assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == min(n, m)
        if n <= m:
            best = max(sum(s[i][p[i]] for i in range(n))
                       for p in itertools.permutations(range(m), n))
        else:
            best = max(sum(s[p[j]][j] for j in range(m))
                       for p in itertools.permutations(range(n), m))
        assert sum(s[i][j] for i, j in pairs) == pytest.approx(best)
    assert hungarian([]) == [] and hungarian([[]]) == []


# ---------------------------------------------------------------- scripts/eval_fusion.py

def _eval_fusion():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_fusion.py"
    spec = importlib.util.spec_from_file_location("eval_fusion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dump(tmp_path, rows):
    import json
    p = tmp_path / "proposals.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _row(label, gt, jersey, team, number, agree, cluster, video="IMG_0104"):
    return {"video": video, "label": label, "gt_label": gt, "jersey": jersey, "team": team,
            "number": number, "votes": agree * 2, "frames_agree": agree, "reads": 6,
            "cluster": cluster}


def _tsv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:]]


def test_eval_fusion_reads_dump(tmp_path):
    """GT-box dump (clusters present): own team = the cluster of the GT team. P2 misreads 4 as
    5 (P3's number, P3 has no read) -> one wrong proposal; precision 2/3, coverage 3/4."""
    ev = _eval_fusion()
    dump = _dump(tmp_path, [
        _row("P1", "P1", 10, "A", "10", 2, "T1"),
        _row("P2", "P2", 4, "A", "5", 2, "T1"),
        _row("P3", "P3", 5, "A", None, 0, "T1"),
        _row("P4", "P4", 9, "B", "9", 3, "T2"),
    ])
    out = tmp_path / "out"
    assert ev.main([str(dump), "--out", str(out)]) == 0
    rows = _tsv(out / "fusion.tsv")
    tune = [r for r in rows if r["split"] == "tune"
            and float(r["threshold"]) == THRESHOLD and float(r["margin"]) == MARGIN]
    assert len(tune) == 1
    assert (tune[0]["n"], tune[0]["correct"], tune[0]["people"]) == ("3", "2", "4")
    assert float(tune[0]["precision"]) == pytest.approx(2 / 3, abs=1e-4)
    assert float(tune[0]["coverage"]) == pytest.approx(0.75, abs=1e-4)
    log_text = (out / "log.txt").read_text(encoding="utf-8")
    assert "mode clusters" in log_text
    assert log_text.rstrip().splitlines()[-1].startswith("held-out")


def test_eval_fusion_tracker_dump_uses_own_team_roster(tmp_path):
    """Tracker-box dump (no clusters): the roster is the own team's, every label is a
    candidate; a label without a GT person that reads an own number is a wrong proposal."""
    ev = _eval_fusion()
    dump = _dump(tmp_path, [
        _row("P1", "P2", 10, "A", "10", 2, None, video="IMG_0108"),
        _row("P2", None, None, None, "4", 3, None, video="IMG_0108"),
        _row("P3", "P5", 4, "A", None, 1, None, video="IMG_0108"),
        _row("P4", "P7", 10, "B", "10", 2, None, video="IMG_0108"),
    ])
    out = tmp_path / "out"
    assert ev.main([str(dump), "--out", str(out)]) == 0
    held = [r for r in _tsv(out / "fusion.tsv") if r["split"] == "held_out"
            and float(r["threshold"]) == THRESHOLD and float(r["margin"]) == MARGIN]
    # team A: P1 and P4 both read 10 -> ambiguous, no proposal; P2 (no person) -> P5 wrong.
    # team B: P1 and P4 both read 10 -> ambiguous again. 1 proposal, 0 correct, 3 people.
    assert (held[0]["n"], held[0]["correct"], held[0]["people"]) == ("1", "0", "3")
    assert "mode own-team roster" in (out / "log.txt").read_text(encoding="utf-8")
