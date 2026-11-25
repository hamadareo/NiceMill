"""The demo edition of the booklet: scene-level confirmation, layout, the selection note, the numbers."""
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pdfhelp import pdf_page_count  # noqa: E402,F401
from test_export import add_video_subject, repo, video_assignment  # noqa: F401,E402

import pypdfium2 as pdfium  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from pfb_app.export import booklet, demo, demo_scenes as D  # noqa: E402

CREDIT = "Samsunspor TBF / CC BY 3.0 / 切り出し・枠の描画あり"
NUMBERS = {"fps": 30.0, "frames": 5400, "clip_s": 180, "ids": 16, "ids_with_number": 8,
           "boxes_per_frame": 6.8, "longest_run_s": 71.8, "longest_run_id": "P11", "runs_4s": 87,
           "scenes_all": 78, "scenes_ok": 22}


def _norm(s):
    return re.sub(r"\s+", "", s)


def _pages(pdf: bytes) -> list[str]:
    doc = pdfium.PdfDocument(pdf)
    return [_norm(p.get_textpage().get_text_range()) for p in doc]


# --- scene-level confirmation (pure functions on boxes) ------------------------------------

def _series(frames, x=100.0, step=2.0, w=40.0, h=100.0):
    fr = np.array(list(frames))
    return fr, np.array([[x + step * i, 50.0, w, h] for i in range(len(fr))])


def test_continuity_needs_the_boxes_and_no_jump():
    fps = 30.0
    fr, bx = _series(range(1, 401))
    assert D.continuity_ok(fr, bx, 150, fps)                      # every frame, small moves
    gap = np.array([f for f in fr if not 140 <= f <= 170])        # 31 of 61 frames gone
    assert not D.continuity_ok(gap, _series(gap)[1], 150, fps)
    bx2 = bx.copy()
    bx2[150:, 0] += 300.0                                         # the box jumps 3 body heights
    assert not D.continuity_ok(fr, bx2, 150, fps)
    assert D.continuity_ok(fr, bx2, 340, fps)                     # the jump is outside this scene's window


def test_unique_needs_no_box_overlap_and_no_other_reader():
    mine = _series(range(1, 301))
    same_place = {"P2": (mine[0], mine[1].copy())}               # another label on the same body
    far = {"P3": (mine[0], mine[1] + np.array([500.0, 0, 0, 0]))}
    assert not D.unique_ok(100, 160, mine, same_place, [])
    assert D.unique_ok(100, 160, mine, far, [])
    assert not D.unique_ok(100, 160, mine, far, [(150, 400)])    # someone else reads our number there
    assert D.unique_ok(100, 160, mine, far, [(200, 400)])        # ... but only later


def test_windows_are_unbroken_unique_and_without_another_number():
    series = _series(range(1, 601))
    ps = D.PersonScenes([], "14", 3, 4, 9.0, ["P1"], series, {}, [],
                        [(100, 200, "14"), (400, 450, "9")])
    ps.scenes = [D.Scene("e1", 150, True, True, True)]
    found = D.windows(ps, [(1, 600)], max_n=2)
    assert found and all(w["b"] - w["a"] == D.WINDOW_FRAMES for w in found)
    assert found[0]["scene"] and found[0]["number"]               # the one holding a confirmed scene first
    assert all(not (w["a"] <= 450 and w["b"] >= 400) for w in found)   # another number is read there
    assert len(found) == 2 and found[0]["b"] < found[1]["a"] or found[1]["b"] < found[0]["a"]
    assert D.windows(ps, [(1, 100)]) == []                        # a run shorter than the window


# --- who gets a page ------------------------------------------------------------------------

class _S:      # a stand-in for a Sheet in choose()
    def __init__(self, name):
        self.shown_long = name


def _entry(name, ok, reads=0):
    return demo.Entry(_S(name), {f"{name}{i}" for i in range(ok)}, 5, "7", reads, 5)


def test_choose_keeps_two_confirmed_scenes_strongest_first():
    es = [_entry("a", 1), _entry("b", 2), _entry("c", 5), _entry("d", 3), _entry("e", 0), _entry("f", 2)]
    shown, used = demo.choose(es, 2, floor=4)
    assert used == 2 and [e.sheet.shown_long for e in shown] == ["c", "d", "b", "f"]
    assert not any(e.estimated for e in shown)
    assert all(e.n_ok >= 2 for e in shown)                        # nobody without confirmed scenes


def test_choose_lowers_the_minimum_to_one_and_marks_the_page_as_estimate():
    es = [_entry("a", 1), _entry("b", 3), _entry("c", 0), _entry("d", 1)]
    shown, used = demo.choose(es, 2, floor=4)
    assert used == 1 and [e.sheet.shown_long for e in shown] == ["b", "a", "d"]
    assert [e.estimated for e in shown] == [False, True, True]    # 推定
    none, used = demo.choose([_entry("x", 0)], 2, floor=4)
    assert none == []                                             # no confirmed scene: no page at all


# --- the booklet ---------------------------------------------------------------------------

def _jpg(tmp_path, k):
    p = tmp_path / f"f{k}.jpg"
    Image.new("RGB", (60, 80), (200 - k * 10, 200, 200)).save(p)
    return p


def _fake_gather(tmp_path, n_examples):
    def gather(repo, aid, report):
        entries = []
        for i, s in enumerate(report.exported):
            e = demo.Entry(s, {"a", "b", "c"}, 4, "7", 3, 4, 9.0, [], 14.8, 3)
            entries.append(e)
        k = 0
        while k < n_examples and entries:
            e = entries[k % len(entries)]
            e.examples.append(demo.Example(
                e.sheet.shown, e.sheet.jersey_id, 100 + k, 220 + k, 30.0,
                [(f"0:{10 + j:02d}.0", _jpg(tmp_path, j)) for j in range(6)], 14.8, "7", True))
            k += 1
        return demo.Evidence(entries, dict(NUMBERS))
    return gather


def _work(repo, n):
    aid = video_assignment(repo)
    for i in range(1, n + 1):
        add_video_subject(repo, aid, f"#{i}", pfb_text=f"固有の文{i}です。", approved=False)
    return aid


@pytest.mark.parametrize("n,ex", [(0, 0), (1, 0), (1, 2), (1, 1), (5, 8), (5, 3), (9, 8), (9, 0)])
def test_total_is_a_multiple_of_four_and_no_page_is_blank(repo, monkeypatch, tmp_path, n, ex):
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, ex))
    aid = _work(repo, n) if n else video_assignment(repo)
    pdf = demo.build(repo, aid, credits=[CREDIT])
    pages = _pages(pdf)
    shown_n = demo.build.last["people_shown"]
    assert shown_n == n
    n_ex = min(ex, demo.MAX_EXAMPLES)
    lay = demo.layout(n, n_ex if n else 0)
    assert len(pages) % 4 == 0 and len(pages) == lay["total"]
    assert all(len(re.sub(r"NiceMill\d+", "", p)) > 15 for p in pages)    # padding spreads are never blank
    assert "NiceMill処理出力" in pages[0] and "デモ版" in pages[0]
    assert "NiceMillとは" in "".join(pages) and "検出の数字" in "".join(pages)
    assert "できること（今）" in "".join(pages) and "これから（開発中）" in "".join(pages)
    assert "奥付" in pages[-1]
    first = lay["first"]
    for k in range(n):                                            # a person's left page is even
        assert (first + 2 * k) % 2 == 0 and f"固有の文{k + 1}です" not in pages[0]
    if lay["front"]:
        assert "目次" in pages[1] and "この冊子について" in pages[2]
    joined = "".join(pages)
    for bad in ("承認", "下書き", "ナイスミル", "さん", "公開準備", "渡す"):
        assert bad not in joined, bad
    assert not re.search(r"\d+\.\d+\.\d+\.\d+|/home|/Users|pfb_", joined)


def test_demo_is_a_separate_output_and_the_full_booklet_is_unchanged(repo, monkeypatch, tmp_path):
    aid = _work(repo, 3)
    full = booklet.build(repo, aid, credits=[CREDIT], subtitle="公開クリップ（Samsunspor 3分版）の処理結果")
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    dm = demo.build(repo, aid, credits=[CREDIT])
    fp, dp = _pages(full), _pages(dm)
    assert len(fp) == booklet.plan(3)["total"]                    # 3 people: 8 pages, as before
    assert "デモ版" not in "".join(fp) and "検出の例" not in "".join(fp) and "検出の数字" not in "".join(fp)
    assert "デモ版" in dp[0] and dp != fp and len(dp) != len(fp)
    assert "公開クリップ（Samsunspor3分版）の処理結果デモ版" in dp[0]
    assert pdfium.PdfDocument(dm).get_metadata_dict().get("Title", "").endswith("デモ版")
    assert demo.SUBTITLE == "公開クリップ（Samsunspor 3分版）の処理結果 デモ版"


NEGATIVE = ("？", "?", "推定", "不明", "かも", "未確認", "誤", "取り違", "証明", "経ていない", "読めない")


def test_colophon_states_the_selection_with_measured_counts(repo, monkeypatch, tmp_path):
    aid = _work(repo, 4)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    back = pages[-1]
    assert _norm("掲載は、同一人物と確認できた場面のある人物の例です。全4人のうち、人物ページを載せたのは4人です。"
                 "本資料は公開動画1本による実処理例です。人物の対応づけには手がかりに基づく部分があります。") in back
    assert "確認できた場面は、全4人の記録された場面16件のうち12件（75%）" in back   # 4 people x (3 of 4)
    assert "背番号は約6秒ごとに読み取っています" in back
    assert _norm(CREDIT) in back


def test_no_uncertain_or_negative_wording_anywhere(repo, monkeypatch, tmp_path):
    """Pages carry confirmed information only: no 「？」「推定」「不明」「誤り」「取り違え」...; a person without a
    confirmed number is named 選手 ID n; the one limit is stated neutrally on the colophon."""
    aid = _work(repo, 3)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    joined = "".join(pages)
    for w in NEGATIVE:
        assert w not in joined, w
    assert "#1" in joined and not re.search(r"背番号\d+番|背番号不明", joined)     # the plain name, never an uncertain number
    assert joined.count("本資料は公開動画1本による実処理例です") >= 1


def test_numbers_pages_show_only_measured_values_and_the_branch_figures_as_such(repo, monkeypatch, tmp_path):
    aid = _work(repo, 2)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    big = next(p for p in pages if "検出の数字" in p and "1本の動画から" in p)
    future = next(p for p in pages if "これから（開発中）" in p)
    for text in ("180秒", "5400フレーム", "16人物ID", "8／16人物ID", "6.8人／コマ", "71.8秒", "87区間",
                 "22場面", "記録された78場面", "2人分"):
        assert text in big, text                                   # exactly the values handed in
    assert "この1本）で測った値です" in big
    for text in ("47%", "85%", "74%", "86%", "35／74人", "63／74人", "26／35人", "30／35人",
                 "開発版", "本番への反映と、この3分版での測定はこれからです", "選手以外の枠"):
        assert text in future, text


def test_intro_spread_states_the_claim_and_the_flow(repo, monkeypatch, tmp_path):
    aid = _work(repo, 2)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 4))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    first = demo.layout(2, 4)["first"]
    left, right = pages[first - 1], pages[first]
    assert "NiceMillとは" in left and "動画を入れるだけで、選手ごとの個別フィードバックが自動で出る。" in left
    for step in ("動画を1本入れる", "人を見つけて、追い続ける", "背番号を読む", "よかった場面を選ぶ",
                 "ほめる文章を書く", "選手1人に1枚"):
        assert step in left, step
    assert "実際の出力" in right and "2人分の個別フィードバックが出ました" in right


def test_branch_recall_figures_are_consistent_with_their_counts():
    for r in demo.RECALL:
        assert round(100 * r["n_before"] / r["total"]) == r["before"]
        assert round(100 * r["n_after"] / r["total"]) == r["after"]
    assert [(r["before"], r["after"]) for r in demo.RECALL] == [(47, 85), (74, 86)]


def test_strip_pages_state_their_continuity_criterion(repo, monkeypatch, tmp_path):
    aid = _work(repo, 2)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 2))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    strip = next(p for p in pages if "検出の例1／2" in p)
    assert "121フレーム" in strip and "途切れず" in strip and "半分以上重なることもありませんでした" in strip
    assert "6枚で、0.8秒おきです" in strip


def test_odd_number_of_examples_gets_a_reading_guide_page_as_partner(repo, monkeypatch, tmp_path):
    aid = _work(repo, 2)
    monkeypatch.setattr(demo, "gather", _fake_gather(tmp_path, 3))
    pages = _pages(demo.build(repo, aid, credits=[CREDIT]))
    assert any("検出の例の見方" in p for p in pages) and len(pages) % 4 == 0


# the demo edition made from the public clip, when one is at hand (PFB_REAL_BOOKLET_DIR/demo.pdf)
_REAL = Path(os.environ.get("PFB_REAL_BOOKLET_DIR", "/nonexistent")) / "demo.pdf"


@pytest.mark.skipif(not _REAL.is_file(), reason="the real demo output is not on this machine")
def test_real_demo_output_numbers_and_structure():
    pages = _pages(_REAL.read_bytes())
    assert len(pages) % 4 == 0 and len(pages) == 24
    joined = "".join(pages)
    assert "全9人のうち、人物ページを載せたのは4人です" in joined
    for w in NEGATIVE:
        assert w not in joined, w
    m = re.search(r"記録された場面(\d+)件のうち(\d+)件（(\d+)%）", joined)
    assert m and int(m.group(1)) == 78 and round(100 * int(m.group(2)) / 78) == int(m.group(3))
    assert 0 < int(m.group(2)) < 78                                # a real share, neither all nor none
    assert "検出の例1／8" in joined and "検出の例8／8" in joined
