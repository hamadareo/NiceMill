"""Calm over-the-head tags (pfb_app.review_video.calm and the page's tag layout): a steady anchor and steady words."""
import math

import numpy as np

from pfb_app.review_video import calm, everyone, player


def _rows(n=120, step=3, jitter=3.0, seed=1, vx=0.0):
    rng = np.random.default_rng(seed)
    return [[1 + i * step, 400 + vx * i * step + rng.normal(0, jitter), 200 + rng.normal(0, jitter), 40 + rng.normal(0, 2), 120 + rng.normal(0, 3)] for i in range(n)]


def _shake(pts):
    a = np.array(pts, float)
    return float(np.mean(np.hypot(*(a[2:] - 2 * a[1:-1] + a[:-2]).T)))


def test_the_anchor_shakes_less_than_half_as_much_and_does_not_lag_behind_a_runner():
    rows = _rows(jitter=3.0)
    raw = [(r[1] + r[3] / 2, r[2]) for r in rows]
    sm = [(x, y) for _, x, y in calm.smooth_anchor(rows, 30.0)]
    assert _shake(sm) < 0.5 * _shake(raw)
    run = _rows(jitter=0.0, vx=0.6)                       # 18 px per second, steady: the lag stays small
    out = calm.smooth_anchor(run, 30.0)
    lag = [abs((r[1] + r[3] / 2) - x) for r, (_, x, _) in zip(run[40:], out[40:])]
    assert max(lag) < 25


def test_a_small_wobble_does_not_move_the_anchor_at_all():
    rows = [[1 + 3 * i, 400 + (2 if i % 2 else -2), 200 + (1 if i % 3 else -1), 40, 120] for i in range(60)]
    out = calm.smooth_anchor(rows, 30.0)
    assert len({(x, y) for _, x, y in out[20:]}) <= 2


def test_a_word_is_shown_at_least_the_minimum_and_a_short_interruption_is_ignored():
    fps = 30.0
    frames = list(range(1, 1 + 3 * 200, 3))                                    # 20 s, 10 rows a second
    raw = [3] * 50 + [1] * 5 + [3] * 145                                       # a half second of "standing" inside "running"
    shown = calm.stabilise(frames, raw, fps)
    assert set(shown) == {3}
    flick = [3 if (i // 3) % 2 else 2 for i in range(200)]                      # running / walking every 0.3 s
    sh = calm.stabilise(frames, flick, fps)
    assert sum(1 for a, b in zip(sh, sh[1:]) if a != b) <= 1
    steady = [3] * 60 + [2] * 140
    sh = calm.stabilise(frames, steady, fps)
    ch = [i for i in range(1, len(sh)) if sh[i] != sh[i - 1]]
    assert len(ch) == 1 and (frames[ch[0]] - frames[60]) / fps >= calm.HOLD_S           # a real change comes, after the hold


def test_every_shown_state_lasts_at_least_the_minimum_and_unknown_never_replaces_one():
    fps = 30.0
    frames = list(range(1, 1 + 3 * 300, 3))
    rng = np.random.default_rng(4)
    raw = [int(rng.choice([1, 2, 3], p=[.3, .4, .3])) for _ in range(300)]
    sh = calm.stabilise(frames, raw, fps)
    ch = [0] + [i for i in range(1, len(sh)) if sh[i] != sh[i - 1]] + [len(sh)]
    assert all((frames[b - 1] - frames[a]) / fps + 0.1 >= calm.MIN_SHOW_S - 1e-6 for a, b in zip(ch[1:-2], ch[2:-1]))
    assert calm.stabilise(frames[:30], [3] * 10 + [0] * 20, fps) == [3] * 30


def test_the_page_data_carries_a_calm_row_for_every_track_and_unnamed_person():
    tracks = {"P1": [[1 + 3 * i, 100 + i, 50, 30, 90, 3] for i in range(40)]}
    extras = [[1 + 3 * i, 300, 60, 30, 90, 1, 2] for i in range(40)] + [[1 + 3 * i, 600, 60, 30, 90, 2, 1] for i in range(40)]
    out, tid = player._calm(30.0, 3, tracks, extras)
    assert set(out) == {"P1", "e0", "e1"} and len(tid) == 80
    assert all(len(r) == 4 for rows in out.values() for r in rows)
    assert out["P1"][-1][3] == 3 and out["e0"][0][3] == 2


def test_the_page_draws_numbers_for_everyone_and_the_sentence_only_for_a_few_with_a_big_font():
    t = player.TEMPLATE
    assert 'id="act-all" type="checkbox"' in t and 'id="act-all" type="checkbox" checked' not in t     # off by default
    assert "全員の様子を常に表示" in t and "選んだ人・ボールを持っていそうな人・見せ場の場面の人だけ" in t
    assert "L.ch*0.026" in t                                                  # the font is 2.6 % of the picture's height
    assert "ease(" in t and "wantUntil" in t and "noSent" in t               # the box eases, a sentence stays, the fallback waits
    assert "calmAt(" in t and 'e.q?e.q.tag:"番号不明"' in t
