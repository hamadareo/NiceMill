"""Pure view helpers of the 判定支援 pages (no Streamlit): names, episode rows, agreement copy,
the court mini-map SVG, the stream.jsonl tail and the live state.

Only synthetic data: invented episodes and stream records in tmp files.
"""
import ast
import base64
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pfb_app.ui import linecall_view as lv
from pfb_app.ui.wizard.theme import TOKENS
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN

RULING_WORDS = ("公式", "判定結果", "確定", "判決", "審判の判定")
MODULE = Path(lv.__file__)


# -- names and copy -------------------------------------------------------------------------

def test_player_names():
    assert lv.player_name("P7") == "7番の人"
    assert lv.player_name("A6") == "Aチーム #6"
    assert lv.player_name("B12") == "Bチーム #12"
    assert lv.player_name(3) == "3番の人"
    assert lv.player_name("3") == "3番の人"


def test_copy_constants():
    assert lv.NOTE == "ここに出るのは判定材料です。最後に決めるのは審判とコーチです。"
    assert lv.LABEL_BUTTONS == (("correct", "正しい"), ("wrong", "違う"),
                                ("undecidable", "判定不能"))
    assert lv.VERDICT_LABELS == {"stepped": "踏んだ", "clear": "踏んでいない",
                                 "undecidable": "判定不能"}
    assert lv.LINE_LABELS == {"sideline_L": "サイドライン（左）", "sideline_R": "サイドライン（右）",
                              "baseline": "エンドライン", "top_line": "反対側のライン",
                              "arc": "2ポイントライン"}
    assert lv.SIDE_LABELS == {"in": "内側", "out": "外側", "inside": "内側", "outside": "外側"}
    assert lv.coach_label_text("correct") == "コーチの確認: 正しい"
    assert lv.coach_label_text("undecidable") == "コーチの確認: 判定不能"
    assert lv.coach_label_text(None) is None
    assert lv.latency_text(0.0161) == "遅れ 約0.0秒"
    assert lv.latency_text(1.26) == "遅れ 約1.3秒"
    assert lv.latency_text(None) is None
    assert lv.LIVE_NO_GPU == ("解析用の機械が混んでいるため開始できません。"
                              "少し待ってからもう一度押してください。")


def test_copy_has_no_banned_words():
    """Every string constant of the module (docstrings aside) is free of the school words,
    the ruling words and the wizard screens' FORBIDDEN words."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)}
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docstrings]
    assert any("判定材料" in s for s in strings)          # the scan sees the copy
    for s in strings:
        found = [w for w in (*SCHOOL_WORDS, "ルーブリック", *RULING_WORDS, *FORBIDDEN) if w in s]
        assert not found, (s, found)


# -- episodes -------------------------------------------------------------------------------

def ep(i, verdict="stepped", label="P7", line="sideline_L", clock="1:05", video_s=65.2):
    return {"id": f"{label}-{line}-{i}", "label": label, "line": line, "group": "sideline",
            "f0": i, "f1": i + 9, "verdict": verdict, "verdict_ja": lv.VERDICT_LABELS[verdict],
            "status": "判定材料", "reason": None, "frame": i + 4, "clock": clock,
            "video_s": video_s, "crop": f"crops/{i}.jpg", "overlay": f"crops/{i}_ov.jpg",
            "clip": f"clips/{i}.mp4", "clip_note": None, "uncertainty_cm": 6.0,
            "distances": {"L": -3.0, "R": 12.0}, "evidence": []}


def test_episodes_missing_or_torn(tmp_path):
    p = tmp_path / "episodes.json"
    assert lv.load_episodes(p) is None                           # missing
    p.write_text('{"status": "判定材料", "episodes": [{"id": "a"', encoding="utf-8")
    assert lv.load_episodes(p) is None                           # torn mid-write
    p.write_bytes(b"\xff\xfe\x00garbage")
    assert lv.load_episodes(p) is None                           # not text
    p.write_text(json.dumps({"status": "判定材料"}), encoding="utf-8")
    assert lv.load_episodes(p) is None                           # no episodes list
    p.write_text(json.dumps({"episodes": {"a": 1}}), encoding="utf-8")
    assert lv.load_episodes(p) is None
    p.write_text(json.dumps([ep(1)]), encoding="utf-8")
    assert lv.load_episodes(p) is None                           # not a document
    (tmp_path / "dir.json").mkdir()
    assert lv.load_episodes(tmp_path / "dir.json") is None       # unreadable
    p.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    assert lv.load_episodes(p) == []
    p.write_text(json.dumps({"status": "判定材料", "episodes": [ep(1), "junk", ep(2)]}),
                 encoding="utf-8")
    assert [e["id"] for e in lv.load_episodes(p)] == ["P7-sideline_L-1", "P7-sideline_L-2"]


def test_episode_row():
    # a 踏んだ without a geometry: 要確認 (test_linecall_near: with one)
    assert lv.episode_row(ep(1)) == {"時間": "1:05", "選手": "7番の人",
                                     "線": "サイドライン（左）", "判定材料": "踏んだ（要確認）"}
    row = lv.episode_row(ep(2, verdict="clear", label="A6", line="arc", clock=None,
                            video_s=125.9))
    assert row == {"時間": "2:05", "選手": "Aチーム #6", "線": "2ポイントライン",
                   "判定材料": "踏んでいない"}
    row = lv.episode_row({"label": "B3", "line": "top_line", "verdict": "undecidable",
                          "clock": None, "video_s": None})
    assert row == {"時間": "—", "選手": "Bチーム #3", "線": "反対側のライン", "判定材料": "判定不能"}
    assert lv.episode_row({})["判定材料"] == "判定不能"           # never raises


def test_agreement_copy():
    stepped = [ep(i, verdict="stepped") for i in range(60)]
    clear = [ep(100 + i, verdict="clear") for i in range(60)]
    und = [ep(200 + i, verdict="undecidable") for i in range(60)]
    eps = stepped + clear + und
    ok = lambda es: {e["id"]: "correct" for e in es}            # noqa: E731
    assert lv.agreement(eps, {}) == "コーチの確認: 0件（あと50件で一致率を表示）"
    # below min_n in every class: the count form; n = 正しい + 違う on stepped/clear only
    labels = {**ok(stepped[:49]), **ok(clear[:10]), **ok(und[:30]),
              stepped[49]["id"]: "undecidable", "gone-episode": "correct"}
    assert lv.agreement(eps, labels) == "コーチの確認: 59件（あと1件で一致率を表示）"
    # (a) a stored label made for another verdict (the episode changed on a rerun) is skipped
    labels[stepped[49]["id"]] = {"label": "correct", "verdict": "clear"}
    assert lv.agreement(eps, labels) == "コーチの確認: 59件（あと1件で一致率を表示）"
    labels[stepped[49]["id"]] = {"label": "correct", "verdict": "stepped"}
    # (c) a class reaching min_n shows its rate; the other class waits with a dash
    labels[stepped[0]["id"]] = labels[stepped[1]["id"]] = "wrong"
    assert lv.agreement(eps, labels) == \
        "コーチの確認との一致率: 踏んだ（要確認を含む） 96%（50件） / 踏んでいない —（あと40件）"
    # (b) 判定不能 episodes never count, however many are confirmed
    assert lv.agreement(und, ok(und)) == "コーチの確認: 0件（あと50件で一致率を表示）"
    labels.update(ok(clear[10:50]))
    labels[clear[0]["id"]] = {"label": "wrong", "verdict": "clear"}
    assert lv.agreement(eps, labels) == \
        "コーチの確認との一致率: 踏んだ（要確認を含む） 96%（50件） / 踏んでいない 98%（50件）"
    assert lv.agreement(stepped[:5], ok(stepped[:5]), min_n=5) == \
        "コーチの確認との一致率: 踏んだ（要確認を含む） 100%（5件） / 踏んでいない —（あと5件）"
    # the copy lives in constants
    assert lv.AGREEMENT_COUNT.format(n=3, m=47) == "コーチの確認: 3件（あと47件で一致率を表示）"
    assert lv.AGREEMENT_RATE.format(stepped="a", clear="b") == \
        "コーチの確認との一致率: 踏んだ（要確認を含む） a / 踏んでいない b"
    assert lv.AGREEMENT_CLASS.format(p=90, n=50) == "90%（50件）"
    assert lv.AGREEMENT_CLASS_WAIT.format(m=7) == "—（あと7件）"


# -- mini-map -------------------------------------------------------------------------------

def circles(svg):
    return [dict(re.findall(r'(\w[\w-]*)="([^"]*)"', m))
            for m in re.findall(r"<circle ([^>]*)/?>", svg)]


def test_minimap_has_numbered_dots():
    players = [{"id": 3, "x": 752.5, "y": 0.0, "in_play": True},
               {"id": 7, "x": 10.0, "y": 1100.0, "in_play": True},
               {"id": 12, "x": 700.0, "y": 600.0, "in_play": False},
               {"id": 5, "x": float("nan"), "y": 3.0}]
    svg = lv.minimap_svg(players, flagged={7})
    assert svg.startswith("<svg") and 'viewBox="0 0 1505 1105"' in svg
    root = ET.fromstring(svg)                  # well-formed: no duplicate attributes
    assert root.tag.endswith("svg") and root.get("viewBox") == "0 0 1505 1105"
    assert re.search(r'role="img"', svg) and "aria-labelledby=" in svg
    title = re.search(r'<title id="([^"]+)">([^<]+)</title>', svg)
    desc = re.search(r'<desc id="([^"]+)">([^<]+)</desc>', svg)
    assert title and desc
    assert f'aria-labelledby="{title.group(1)} {desc.group(1)}"' in svg
    assert "7番の人" in desc.group(2)                   # the flagged player is named in words
    dots = [c for c in circles(svg) if c.get("class") == "player"]
    assert [d["data-id"] for d in dots] == ["3", "7", "12"]      # NaN row skipped
    numbers = re.findall(r'<text class="num"[^>]*>(\d+)</text>', svg)
    assert numbers == ["3", "7", "12"]
    # flagged: the warning token AND a ring, never colour alone
    by_id = {d["data-id"]: d for d in dots}
    assert by_id["7"]["fill"] == TOKENS["wait"]
    assert by_id["3"]["fill"] != TOKENS["wait"]
    rings = [c for c in circles(svg) if c.get("class") == "flag-ring"]
    assert len(rings) == 1
    assert (rings[0]["cx"], rings[0]["cy"]) == (by_id["7"]["cx"], by_id["7"]["cy"])
    assert float(rings[0]["r"]) > float(by_id["7"]["r"])
    # off play: dashed outline, not the in-play fill
    assert "stroke-dasharray" in by_id["12"]
    # far baseline (y = 0, under the basket) at the top, the near top line at the bottom
    assert float(by_id["3"]["cy"]) < float(by_id["7"]["cy"])
    assert float(by_id["3"]["cy"]) < 1105 / 4 and float(by_id["7"]["cy"]) > 1105 * 3 / 4
    assert float(by_id["7"]["cx"]) < float(by_id["3"]["cx"])     # left sideline on the left
    # court outline, key and arc are drawn; colours come from the theme tokens
    assert 'class="court"' in svg and 'class="key"' in svg and 'class="arc"' in svg
    assert TOKENS["board"] in svg and TOKENS["chalk"] in svg


def test_minimap_geometry_matches_engine_court():
    """The arc path runs through the engine court model's junctions (court.py: R3 674.5,
    ARC_X 97.5, basket (752.5, 134)), mapped through the same transform as the dots."""
    arc_y = 134 + math.sqrt(674.5 ** 2 - 655 ** 2)
    assert lv.ARC_Y == pytest.approx(arc_y) and lv.ARC_Y == pytest.approx(295.0132, abs=1e-3)
    svg = lv.minimap_svg([], set())
    arc = re.search(r'<path class="arc" d="([^"]+)"', svg).group(1)
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", arc)]
    x0, y0 = lv.to_svg(97.5, 0.0)
    x1, y1 = lv.to_svg(97.5, arc_y)
    x2, y2 = lv.to_svg(1505 - 97.5, arc_y)
    assert nums[:4] == pytest.approx([x0, y0, x1, y1], abs=0.1)
    assert nums[-4:-2] == pytest.approx([x2, y2], abs=0.1)
    # the arc's lowest point (752.5, 808.5) is inside the drawn area, below the key
    _, ybot = lv.to_svg(752.5, 134 + 674.5)
    assert 0 < ybot < 1105
    # everything fits the viewBox, with room for players just outside the lines
    assert lv.to_svg(-40, -40)[0] > 0 and lv.to_svg(-40, -40)[1] > 0
    assert lv.to_svg(1545, 1145)[0] < 1505 and lv.to_svg(1545, 1145)[1] < 1105


def test_minimap_alt_and_data_uri():
    """st.html drops inline SVG, so the page shows the map as an image: a data URI for
    st.image / <img>, with the desc sentence as its alt text."""
    players = [{"id": 3, "x": 700.0, "y": 300.0}, {"id": 7, "x": 5.0, "y": 500.0}]
    alt = lv.minimap_alt(players, {7})
    assert alt == "上が奥のエンドライン、下が手前のライン。選手 2人。線の近くでアラート中: 7番の人。"
    assert lv.minimap_alt([], set()).endswith("選手 0人。アラート中の選手はいません。")
    svg = lv.minimap_svg(players, {7})
    assert f">{alt}</desc>" in svg                     # the same sentence as the SVG desc
    uri = lv.minimap_data_uri(players, {7})
    assert uri.startswith("data:image/svg+xml;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]).decode("utf-8") == svg
    # fonts do not load inside an <img> SVG: a generic family only
    assert 'font-family="sans-serif"' in svg and "Zen Kaku" not in svg


def test_minimap_escapes_and_ids_unique():
    a = lv.minimap_svg([{"id": 1, "x": 0, "y": 0}], set(), uid="a")
    b = lv.minimap_svg([{"id": 1, "x": 0, "y": 0}], set(), uid="b")
    assert re.search(r'<title id="([^"]+)"', a).group(1) != \
        re.search(r'<title id="([^"]+)"', b).group(1)
    assert "<script" not in lv.minimap_svg([{"id": "<script>", "x": 1, "y": 1}], set())


# -- stream tail ----------------------------------------------------------------------------

def line(rec):
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def test_stream_tail_partial_line(tmp_path):
    p = tmp_path / "stream.jsonl"
    tail = lv.StreamTail(p)
    assert tail.read_new() == []                                  # not there yet
    a = line({"type": "start", "t": 0.0, "status": "判定材料"})
    b = line({"type": "positions", "t": 0.2, "frame": 7, "players": []})
    p.write_bytes(a + b[: len(b) // 2])                           # 1.5 lines
    assert tail.read_new() == [json.loads(a)]
    assert tail.read_new() == []                                  # the half line is kept
    with p.open("ab") as f:
        f.write(b[len(b) // 2:])
    assert tail.read_new() == [json.loads(b)]
    assert tail.read_new() == []
    # a multi-byte character split across two writes is not mangled
    c = line({"type": "dwell", "label": "ライン付近に滞在", "player": 3})
    cut = c.index("滞".encode()) + 1
    with p.open("ab") as f:
        f.write(c[:cut])
    assert tail.read_new() == []
    with p.open("ab") as f:
        f.write(c[cut:])
    assert tail.read_new() == [json.loads(c)]


def test_stream_tail_restarts_on_replace_and_skips_bad_lines(tmp_path):
    p = tmp_path / "stream.jsonl"
    p.write_bytes(line({"type": "start", "t": 0.0}) + b"not json\n" + line([1, 2])
                  + line({"type": "stats", "t": 1.0}))
    tail = lv.StreamTail(p)
    assert [r["type"] for r in tail.read_new()] == ["start", "stats"]
    # replaced by a new session (new inode): read from the beginning again
    q = tmp_path / "new.jsonl"
    q.write_bytes(line({"type": "start", "t": 0.0, "session": "s2"})
                  + line({"type": "stats", "t": 1.0}) + line({"type": "stats", "t": 2.0}))
    os.replace(q, p)
    assert [r.get("session") for r in tail.read_new()] == ["s2", None, None]
    # truncated in place (size shrank): start over
    p.write_bytes(line({"type": "start", "t": 0.0, "session": "s3"}))
    assert [r.get("session") for r in tail.read_new()] == ["s3"]
    p.unlink()
    assert tail.read_new() == []


# -- live state -----------------------------------------------------------------------------

def alert(t, aid, player=7, line="sideline_L", event="touch", side="in", d=8.4, nc=True):
    return {"type": "alert", "t": t, "frame": int(t * 30) + 1, "alert_id": aid, "player": player,
            "line": line, "event": event, "side": side, "distance_cm": d, "needs_check": nc}


def test_live_state_rolls_alerts():
    s = lv.LiveState()
    assert s.players == [] and s.alerts == [] and s.latency_s is None and not s.ended
    s.apply([{"type": "start", "t": 0.0, "status": "判定材料", "alerts": "event"},
             {"type": "positions", "t": 0.2, "frame": 7,
              "players": [{"id": 7, "x": 3.0, "y": 500.0, "in_play": True,
                           "near_line": "sideline_L", "near_cm": 9.1}]},
             alert(0.3, "7-sideline_L-10"),
             {"type": "stats", "t": 1.0, "latency": {"p50_ms": 16.1, "p95_ms": 19.1}},
             {"type": "positions", "t": 0.4, "frame": 13,
              "players": [{"id": 7, "x": 4.0, "y": 501.0}, {"id": 9, "x": 900.0, "y": 300.0}]}])
    assert [p["id"] for p in s.players] == [7, 9]                 # latest positions only
    assert s.latency_s == pytest.approx(0.0161)
    assert s.flagged == {7}
    a = s.alerts[0]
    assert a["id"] == "7-sideline_L-10" and not a["ended"] and a["needs_check"]
    assert lv.alert_text(a) == "7番の人　サイドライン（左）まで 8 cm（内側）"
    # the same alert crosses: updated in place, needs_check off
    s.apply([alert(0.5, "7-sideline_L-10", event="cross", side="out", d=-14.6, nc=False)])
    assert len(s.alerts) == 1 and s.alerts[0]["record"]["event"] == "cross"
    assert not s.alerts[0]["needs_check"]
    assert lv.alert_text(s.alerts[0]) == "7番の人　サイドライン（左）まで 15 cm（外側）"
    s.apply([{"type": "alert_end", "t": 0.9, "alert_id": "7-sideline_L-10", "result": "cross",
              "needs_check": False, "min_distance_cm": 2.0, "duration_s": 0.6,
              "reason": "exit"}])
    assert s.alerts[0]["ended"] and s.flagged == set()
    # 25 more alerts: only the last 20 ids are kept, newest first
    s.apply([alert(1 + i, f"9-arc-{i}", player=9, line="arc", side="outside") for i in range(25)])
    assert len(s.alerts) == 20
    assert [a["id"] for a in s.alerts[:2]] == ["9-arc-24", "9-arc-23"]
    assert s.alerts[-1]["id"] == "9-arc-5"
    assert lv.alert_text(s.alerts[0]) == "9番の人　2ポイントラインまで 8 cm（外側）"
    # a dwell is listed with its own words; its end closes it
    s.apply([{"type": "dwell", "t": 30.0, "frame": 901, "dwell_id": "9-arc-dwell-901",
              "player": 9, "line": "arc", "label": "ライン付近に滞在", "t_start": 26.9,
              "side": "inside", "distance_cm": 12.0}])
    assert s.alerts[0]["id"] == "9-arc-dwell-901" and s.alerts[0]["kind"] == "dwell"
    assert lv.alert_text(s.alerts[0]) == "9番の人　2ポイントライン付近に滞在"
    s.apply([{"type": "dwell_end", "t": 40.0, "dwell_id": "9-arc-dwell-901", "player": 9,
              "line": "arc", "duration_s": 13.1, "reason": "exit"}])
    assert s.alerts[0]["ended"]
    # an end for an id already rolled out is ignored; a stats without latency keeps the value
    s.apply([{"type": "alert_end", "t": 41.0, "alert_id": "9-arc-0"},
             {"type": "stats", "t": 41.0, "latency": {"p50_ms": None}},
             {"type": "unknown", "t": 41.0}, {"no": "type"}])
    assert len(s.alerts) == 20 and s.latency_s == pytest.approx(0.0161)
    s.apply([{"type": "end", "t": 42.0, "reason": "stop", "status": "判定材料"}])
    assert s.ended and s.flagged == set()
    # a new start (new session in the same stream) clears everything
    s.apply([{"type": "start", "t": 0.0}])
    assert s.alerts == [] and s.players == [] and not s.ended and s.latency_s is None


def test_live_state_keeps_the_drop_rate_of_the_latest_stats_or_end():
    s = lv.LiveState()
    assert s.drop_rate is None and lv.drop_texts(s) == (None, None)
    s.apply([{"type": "stats", "t": 1.0, "frames_in": 200, "frames_processed": 200,
              "dropped": 0}])
    assert s.drop_rate == 0.0
    rate, note = lv.drop_texts(s)
    assert rate == "飛ばした映像 0%" and note is None          # nothing dropped: no warning
    s.apply([{"type": "stats", "t": 2.0, "frames_in": 400, "frames_processed": 388,
              "dropped": 12}])
    assert s.drop_rate == pytest.approx(0.03)
    rate, note = lv.drop_texts(s)
    assert rate == "飛ばした映像 3%"
    assert note == lv.LIVE_DROPS_NOTE == ("映像の一部を飛ばしています。"
                                          "アラートが出ないことは、線を踏んでいないことを意味しません。")
    # 「混んでいるため」 only when the machine was busy with other work while it dropped
    assert lv.drop_texts(s, busy=True) == ("飛ばした映像 3%", lv.LIVE_DROPS_NOTE_BUSY)
    assert lv.LIVE_DROPS_NOTE_BUSY == "混んでいるため" + lv.LIVE_DROPS_NOTE
    # a torn stats record keeps the last rate; the end record's summary replaces it
    s.apply([{"type": "stats", "t": 3.0, "frames_in": "x", "dropped": None},
             {"type": "stats", "t": 3.5, "frames_in": 0, "dropped": 0}])
    assert s.drop_rate == pytest.approx(0.03)
    s.apply([{"type": "end", "t": 9.0, "frames_in": 1000, "frames_processed": 995,
              "dropped": 5, "reason": "eof"}])
    assert s.ended and s.drop_rate == pytest.approx(0.005)
    assert lv.drop_texts(s) == ("飛ばした映像 0.5%", lv.LIVE_DROPS_NOTE)   # a drop: always said
    s.apply([{"type": "start", "t": 0.0}])
    assert s.drop_rate is None


def test_alert_text_raw_record():
    assert lv.alert_text(alert(1.0, "x", player=12, line="baseline", side="out", d=-0.4)) \
        == "12番の人　エンドラインまで 0 cm（外側）"


# -- live sources ---------------------------------------------------------------------------

def test_tid_video_for():
    """The settings demo is a TrackID video (IMG_*), or T10's 1920 proxy of one: the engine
    then takes the court from TrackID's meta (--tid-video)."""
    assert lv.tid_video_for(Path("/v/IMG_0105.MOV")) == "IMG_0105"
    assert lv.tid_video_for(Path("/runs/live_proxy/IMG_0105_1920.mp4")) == "IMG_0105"
    assert lv.tid_video_for(Path("/v/game.mp4")) is None
    assert lv.tid_video_for(Path("/v/my_IMG_0105.mp4")) is None
    assert lv.tid_video_for(None) is None


def test_live_copy():
    assert lv.LIVE_STOPPING == "停止しています…"
    assert lv.LIVE_DEMO_LABEL == "デモ動画"
    for text in (lv.LIVE_FAILED, lv.LIVE_START_FAILED, lv.LIVE_VIDEO_MISSING):
        assert text.endswith("。") and "エラー" not in text


# -- the 「試合後の判定材料」 page helpers ------------------------------------------------------

def test_review_copy():
    assert lv.REVIEW_TITLE == "試合後の判定材料"
    assert lv.NO_SPORTS_WORK == "判定支援を使える試合・練習がありません。"
    assert lv.NEXT_PAGE == "次の20件" and lv.PAGE_SIZE == 20
    assert lv.NO_PHOTO == "写真なし" and lv.NO_CLIP == "動画なし"
    assert lv.STALE_VERDICT == "判定材料が新しくなりました。もう一度確かめてください。"
    assert lv.NOT_QUEUED == "この試合・練習は削除の手続き中のため、判定材料を作れません。"
    assert lv.REASON_LABELS["no_box"] == "一部の場面で人が見つかりません"
    assert lv.REASON_LABELS["line_unstable"] == "線の位置が場面ごとにずれています"
    assert set(lv.VERDICT_CHIPS) == set(lv.SHOWN_LABELS) == {*lv.VERDICT_LABELS, "check"}
    from pfb_app.ui.wizard.theme import CHIP_COLOURS
    assert set(lv.VERDICT_CHIPS.values()) <= set(CHIP_COLOURS)
    assert len(set(lv.VERDICT_CHIPS.values())) == 4          # each verdict looks different


def test_review_copy_has_no_banned_words():
    """The page copy, scanned directly (the AST scan above covers the module as a whole)."""
    copy = [*lv.REASON_LABELS.values(), lv.REVIEW_TITLE, lv.REVIEW_GUIDE, lv.NO_SPORTS_WORK,
            lv.NEEDS_TRACK, lv.NOT_QUEUED, lv.STALE_VERDICT, lv.NO_PHOTO, lv.NO_CLIP,
            lv.CROP_CAPTION, lv.OVERLAY_CAPTION, lv.NEXT_PAGE, lv.PREV_PAGE, lv.PAGE_RANGE,
            lv.PICK_GROUP, lv.PICK_CAMERA, lv.REASON_TEXT, lv.NO_MEASURE, lv.ON_LINE,
            lv.UNDER_1CM, lv.GAP_CM, lv.UNC_TEXT, *lv.SIDE_WORDS.values(), lv.NOTE, lv.BUILD,
            lv.BUILDING, lv.BUILD_FAILED, lv.NO_EPISODES, lv.WATCH_CLIP, lv.VERDICT_PREFIX,
            *lv.SHOWN_LABELS.values(), *(t for _, t in lv.LABEL_BUTTONS),
            lv.AGREEMENT_COUNT, lv.AGREEMENT_RATE, lv.NEAR_LINE_NOTE, lv.SIDELINE_NOTE,
            lv.LIVE_SIDELINE_NOTE, lv.MEDIA_REMOVED, lv.ERASING]
    for text in copy:
        found = [w for w in (*FORBIDDEN, *SCHOOL_WORDS, *RULING_WORDS, "ルーブリック")
                 if w in text]
        assert not found, (text, found)


def test_evidence_file_stays_inside(tmp_path):
    base = tmp_path / "linecall"
    (base / "crops" / "e1").mkdir(parents=True)
    (base / "clips").mkdir()
    crop = base / "crops" / "e1" / "131.jpg"
    crop.write_bytes(b"jpeg bytes")
    (base / "crops" / "e1" / "131.png").write_bytes(b"png bytes")
    (base / "crops" / "e1" / "131.JPEG").write_bytes(b"JPEG bytes")
    (base / "crops" / "e1" / "notes.txt").write_bytes(b"x")
    (base / "clips" / "e1.mp4").write_bytes(b"mp4 bytes")
    (tmp_path / "secret.jpg").write_bytes(b"x")
    assert lv.evidence_file(base, "crops/e1/131.jpg", "photo") == b"jpeg bytes"
    assert lv.evidence_file(base, "crops/e1/131.png", "photo") == b"png bytes"
    assert lv.evidence_file(base, "crops/e1/131.JPEG", "photo") == b"JPEG bytes"
    assert lv.evidence_file(base, "clips/e1.mp4", "clip") == b"mp4 bytes"
    assert lv.evidence_path(base, "clips/e1.mp4", "clip") == (base / "clips" / "e1.mp4").resolve()
    # only pictures as photos and only mp4 as clips
    assert lv.evidence_file(base, "crops/e1/notes.txt", "photo") is None
    assert lv.evidence_file(base, "clips/e1.mp4", "photo") is None
    assert lv.evidence_file(base, "crops/e1/131.jpg", "clip") is None
    for rel in (None, "", 3, "crops/e1/none.jpg", "crops/e1", "../secret.jpg",
                "crops/../../secret.jpg", str(tmp_path / "secret.jpg"), "a\x00b.jpg"):
        assert lv.evidence_file(base, rel, "photo") is None, rel
        assert lv.evidence_path(base, rel, "photo") is None, rel
    (base / "crops" / "link.jpg").symlink_to(tmp_path / "secret.jpg")   # a link out of it
    assert lv.evidence_file(base, "crops/link.jpg", "photo") is None
    assert lv.evidence_file(tmp_path / "missing", "crops/e1/131.jpg", "photo") is None
    crop.chmod(0)                                    # unreadable: no photo, no exception
    try:
        if os.geteuid() != 0:
            assert lv.evidence_file(base, "crops/e1/131.jpg", "photo") is None
    finally:
        crop.chmod(0o644)


OUT = {"distances": {"L": -3.1, "R": 12.4}, "uncertainty_cm": {"L": 3.5, "R": 4.0}}


def test_distance_text_outline():
    e = {"line": "sideline_L", **OUT}
    assert lv.distance_text(e) == ("左足: 線の上（誤差 ±4 cm）"
                                   "　右足: 線の内側（コート側）に12 cm（誤差 ±4 cm）")

    def foot(d, line="baseline"):
        return lv.distance_text({"line": line, "distances": {"L": d, "R": None}})

    assert foot(0.0) == "左足: 線の上　右足: 測れませんでした"            # gap = -5
    assert foot(-5.0) == "左足: 線の上　右足: 測れませんでした"           # gap = 0
    assert foot(-5.3).startswith("左足: 線の外側に1 cm未満")              # gap 0.3
    assert foot(-7.0).startswith("左足: 線の外側に2 cm")
    assert foot(0.2).startswith("左足: 線の内側（コート側）に1 cm未満")
    assert foot(12.5).startswith("左足: 線の内側（コート側）に13 cm")     # half up
    assert foot(-5.5, "top_line").startswith("左足: 線の外側に1 cm　")   # gap 0.5 -> 1
    for line in ("sideline_L", "sideline_R", "baseline", "top_line"):
        assert foot(3.0, line).startswith("左足: 線の内側（コート側）に3 cm")


def test_distance_text_arc():
    e = {"line": "arc", "distances": {"L": None, "R": 4.2},
         "uncertainty_cm": {"L": None, "R": 4.0}}
    assert lv.distance_text(e) == "左足: 測れませんでした　右足: 線の内側（ゴール側）に1 cm（誤差 ±4 cm）"

    def foot(d):
        return lv.distance_text({"line": "arc", "distances": {"L": d, "R": d}})

    assert foot(-3.0) == "左足: 線の上　右足: 線の上"                     # |d| - 3.4 < 0
    assert foot(3.4) == "左足: 線の上　右足: 線の上"
    assert foot(3.6).startswith("左足: 線の内側（ゴール側）に1 cm未満")
    assert foot(-10.0).startswith("左足: 線の外側に7 cm")                 # 6.6 -> 7


def test_distance_text_uncertainty_and_gaps():
    # a scalar uncertainty is for both feet; ceil; None / inf / nan are left out
    e = {"line": "baseline", "distances": {"L": 8.0, "R": 9.0}, "uncertainty_cm": 6.2}
    assert lv.distance_text(e) == ("左足: 線の内側（コート側）に8 cm（誤差 ±7 cm）"
                                   "　右足: 線の内側（コート側）に9 cm（誤差 ±7 cm）")
    e = {"line": "baseline", "distances": {"L": 8.0, "R": 9.0},
         "uncertainty_cm": {"L": float("inf"), "R": None}}
    assert lv.distance_text(e) == "左足: 線の内側（コート側）に8 cm　右足: 線の内側（コート側）に9 cm"
    assert lv.distance_text({"line": "baseline", "distances": {"L": 8.0, "R": 9.0},
                             "uncertainty_cm": float("nan")}).count("誤差") == 0
    # nothing measured, nothing usable or an unknown line: no line at all
    assert lv.distance_text({"line": "arc", "distances": {"L": None, "R": None},
                             "uncertainty_cm": {"L": None, "R": None}}) is None
    assert lv.distance_text({}) is None
    assert lv.distance_text({"line": "arc", "distances": "junk"}) is None
    assert lv.distance_text({"line": "arc", "distances": {"L": float("inf"), "R": "x"}}) is None
    assert lv.distance_text({"line": "free_throw", **OUT}) is None


def test_reason_text():
    assert lv.reason_text({"verdict": "undecidable", "reason": "occluded"}) == \
        "判定不能の理由: 足が隠れています"
    assert lv.reason_text({"verdict": "undecidable", "reason": "no_box"}) == \
        "判定不能の理由: 一部の場面で人が見つかりません"
    for reason in ("occluded", "no_box", "no_pose", "low_conf", "line_not_found",
                   "line_unstable", "margin"):   # pfb3x3.linecall.verdict.REASONS
        assert reason in lv.REASON_LABELS
    assert lv.reason_text({"verdict": "undecidable", "reason": "new_one"}) is None
    assert lv.reason_text({"verdict": "stepped", "reason": "occluded"}) is None


def test_agreement_matches_ids_as_text():
    eps = [{"id": 7, "verdict": "stepped"}]
    assert lv.agreement(eps, {"7": {"label": "correct", "verdict": "stepped"}}) == \
        "コーチの確認: 1件（あと49件で一致率を表示）"


def test_page_slice():
    eps = list(range(45))
    assert lv.page_slice(eps, 0) == (0, list(range(20)))
    assert lv.page_slice(eps, 20) == (20, list(range(20, 40)))
    assert lv.page_slice(eps, 40) == (40, list(range(40, 45)))
    assert lv.page_slice(eps, 60) == (40, list(range(40, 45)))     # fewer now: the last page
    assert lv.page_slice(eps, 7) == (0, list(range(20)))           # always a page start
    assert lv.page_slice([], 20) == (0, [])
    assert lv.page_range_text(20, 20, 45) == "21〜40件目（全45件）"


def test_listed_needs_unique_ids():
    eps = [{"id": "a"}, {"id": None}, {}, {"id": ""}, {"id": "a", "x": 1}, {"id": 7}, {"id": True}]
    assert lv.listed(eps) == [{"id": "a"}, {"id": 7}]
