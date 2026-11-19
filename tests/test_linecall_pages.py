"""The 判定支援 pages (AppTest) against tests/fakes/fake_engine.py.

「ライブ判定」 (pages/linecall_live.py): start / stop the engine's live demo, the court
mini-map, rolling alerts and latency, and the plain copy for needs-GPU, errors and a
missing demo video.

Only synthetic data: a fake engine, a few bytes standing in for the demo video, a generated
tiny video for a camera. No image is looked at; the map is checked as SVG text.
"""
import base64
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from fakes.engine_env import alive, install_fake_engine
from pfb_app.config import load_settings
from pfb_app.domain.models import Stage
from pfb_app.engine import EngineError, Pfb3x3Engine
from pfb_app.engine import live as live_mod
from pfb_app.engine.live import LiveController
from pfb_app.inference import gpu
from pfb_app.services.intake import register_video_group
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui import linecall_view as lv
from test_neutral_wording import SCHOOL_WORDS, everything
from test_wizard_ui_basic import FORBIDDEN, UI, html, make_video, ok, screen_text

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                                reason="process checks need /proc (Linux)")

LIVE_PAGE = UI / "pages" / "linecall_live.py"
RULING_WORDS = ("公式", "判定結果", "確定", "判決", "審判の判定")


@pytest.fixture
def lc(tmp_path, monkeypatch):
    """Settings (via PFB_CONFIG) with the fake engine and a demo video named like T10's
    1920 proxy; `lc.configure(demo_video=...)` rewrites them."""
    fe = install_fake_engine(tmp_path, monkeypatch)
    demo = tmp_path / "live_proxy" / "IMG_0105_1920.mp4"
    demo.parent.mkdir()
    demo.write_bytes(b"not really a video")
    cfg_path = tmp_path / "pfb.yaml"

    def configure(demo_video: Path | None = demo):
        cfg = {"data_dir": str(tmp_path / "data"), "inbox_dir": str(tmp_path / "inbox"),
               "engine_dir": str(fe.engine_dir), "engine_python": str(fe.engine_python),
               "tid_python": str(tmp_path / "tid_python")}
        if demo_video is not None:
            cfg["linecall_demo_video"] = str(demo_video)
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setenv("PFB_CONFIG", str(cfg_path))
        ns.settings = load_settings(cfg_path)
        return ns.settings

    # the machine is not busy with other work unless a test says so (never the real GPU)
    monkeypatch.setattr(gpu, "machine_busy", lambda pid: False)
    ns = SimpleNamespace(fe=fe, demo=demo, configure=configure, tmp=tmp_path)
    settings = configure()
    from test_pose_weights import pose_weights
    ns.pose = pose_weights(settings.data_dir, monkeypatch)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    ns.repo = Repo(conn, settings)
    yield ns
    ctl = controller(settings)   # never leave a fake live run (or its stopper) behind
    ctl.stop_in_background().join(timeout=30)
    ctl.stop()


def controller(settings) -> LiveController:
    return LiveController(settings, Pfb3x3Engine(settings))


def page() -> AppTest:
    return AppTest.from_file(str(LIVE_PAGE), default_timeout=60).run()


def wait(cond, timeout=15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def records(stream: Path) -> list[dict]:
    if not stream.exists():
        return []
    return [json.loads(x) for x in stream.read_text(encoding="utf-8").split("\n")[:-1]]


def live_argv(lc) -> list[str]:
    """The argv of the fake live run (it logs itself once it is up)."""
    assert wait(lambda: any(c["argv"][0] == "live" for c in lc.fe.calls()))
    return [c for c in lc.fe.calls() if c["argv"][0] == "live"][-1]["argv"]


def opt(argv, name):
    return argv[argv.index(name) + 1]


def map_svgs(at: AppTest) -> list[tuple[str, str]]:
    """(svg text, caption) of each image that is an SVG data URI."""
    out = []
    for img in at.get("image"):
        for url, caption in zip(img.value, img.captions):
            prefix = "data:image/svg+xml;base64,"
            if url.startswith(prefix):
                out.append((base64.b64decode(url[len(prefix):]).decode("utf-8"), caption))
    return out


def keys(at: AppTest) -> set[str]:
    return {b.key for b in at.button}


def texts(at: AppTest) -> str:
    return everything(at) + "\n" + "\n".join(
        [w.value for w in at.warning] + [e.value for e in at.error] + [s.label for s in at.status])


def stopped_cleanly(lc) -> bool:
    ctl = controller(lc.settings)
    return ctl.status() is None and not ctl.stopping()


# -- ライブ判定 ------------------------------------------------------------------------------

def test_live_start_shows_map_and_alerts(lc, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_LIVE_RECORDS", "25")
    at = ok(page())
    assert at.title[0].value == "ライブ判定"
    assert lv.NOTE in screen_text(at)
    assert lv.LIVE_SIDELINE_NOTE in [c.value for c in at.caption]   # final review LOW 11
    assert lv.LIVE_STOPPED in html(at) and "lc_live_stop" not in keys(at)
    start = at.button(key="lc_live_start")
    assert start.label == "デモを開始" and start.proto.type == "primary" and not start.disabled
    at = ok(start.click().run())
    sess = controller(lc.settings).status()
    assert sess is not None
    argv = live_argv(lc)
    assert argv[0] == "live" and opt(argv, "--video-id") == "linecall_demo"
    assert opt(argv, "--tid-video") == "IMG_0105"   # the 1920 proxy of IMG_0105
    entry = yaml.safe_load((lc.settings.data_dir / "live" / "pfb3x3" / "videos"
                            / "linecall_demo.yaml").read_text("utf-8"))
    assert entry["path"] == str(lc.demo)
    assert lv.LIVE_RUNNING in html(at)
    assert at.button(key="lc_live_stop").label == "停止"
    assert "lc_live_start" not in keys(at)
    assert wait(lambda: len(records(sess.stream)) >= 15)
    at = ok(at.run())
    svgs = map_svgs(at)
    assert len(svgs) == 1
    svg, caption = svgs[0]
    assert svg.startswith("<svg") and 'viewBox="0 0 1505 1105"' in svg
    for n in range(1, 6):   # numbered dots
        assert f'data-id="{n}"' in svg and f">{n}</text>" in svg
    assert caption.startswith("上が奥のエンドライン") and "選手 5人" in caption
    # an alert shows its latest record: the touch (4 cm inside) became a cross (19 cm out)
    assert "2番の人　サイドライン（左）まで 19 cm（外側）" in html(at)
    assert "遅れ 約0.0秒" in screen_text(at)
    # the demo ends by itself (25 records: the 2nd alert stops at its touch): 停止中, and
    # the last picture stays, newest alert first
    assert wait(lambda: stopped_cleanly(lc), timeout=20)
    at = ok(at.run())
    assert lv.LIVE_STOPPED in html(at) and lv.LIVE_RUNNING not in html(at)
    assert "lc_live_start" in keys(at) and "lc_live_stop" not in keys(at)
    items = re.findall(r'<li class="pfb-lc-alert">(.*?)</li>', html(at))
    assert len(items) == 3 and "まで 4 cm（内側）" in items[0] and lv.NEEDS_CHECK in items[0]
    assert "付近に滞在" in items[1] and lv.NEEDS_CHECK not in items[2]
    assert map_svgs(at) and 'class="flag-ring"' not in map_svgs(at)[0][0]   # ended: none open
    assert not at.warning and not at.error


def test_live_stop(lc):
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    sess = controller(lc.settings).status()
    assert wait(lambda: len(records(sess.stream)) >= 3)
    at = ok(at.run())
    at = ok(at.button(key="lc_live_stop").click().run())
    assert wait(lambda: stopped_cleanly(lc), timeout=30)
    assert not alive(sess.pid)
    at = ok(at.run())
    assert lv.LIVE_STOPPED in html(at)
    assert controller(lc.settings).last_outcome() == "stopped"
    assert not at.warning and not at.error   # a user stop is no failure
    assert at.button(key="lc_live_start").label == "デモを開始"
    assert records(sess.stream)[-1]["type"] == "end"
    # the stream was read through its end record: nothing is open on the map any more
    state = at.session_state["_lc_live_follow"]["state"]
    assert state.ended and state.flagged == set()
    assert 'class="flag-ring"' not in map_svgs(at)[0][0]
    assert '<span class="pfb-chip pfb-chip--stopped">' in html(at)


def test_live_stop_keeps_the_page_responsive(lc, monkeypatch):
    """stop() waits up to STOP_WAIT_SEC (20 s) for the engine: the press returns at once and
    the page says 「停止しています…」 until the stop has finished."""
    pids = lc.tmp / "stubborn.pids"
    monkeypatch.setenv("FAKE_ENGINE_STUBBORN", str(pids))   # ignores SIGTERM: SIGKILL later
    monkeypatch.setattr(live_mod, "STOP_WAIT_SEC", 4.0)
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    assert wait(lambda: pids.exists() and len(pids.read_text().split()) == 2)
    engine_pid, child_pid = map(int, pids.read_text().split())
    at = ok(at.run())
    t0 = time.monotonic()
    at = ok(at.button(key="lc_live_stop").click().run())
    assert time.monotonic() - t0 < 3.0
    assert [s.label for s in at.status] == [lv.LIVE_STOPPING]
    assert at.status[0].state == "running"
    assert "lc_live_stop" not in keys(at) and at.button(key="lc_live_start").disabled
    at = ok(at.run())   # still stopping: the page keeps drawing
    assert [s.label for s in at.status] == [lv.LIVE_STOPPING]
    assert at.radio(key="lc_live_source").disabled
    assert wait(lambda: stopped_cleanly(lc), timeout=20)
    assert wait(lambda: not alive(engine_pid) and not alive(child_pid), timeout=5)
    at = ok(at.run())
    assert not at.status and lv.LIVE_STOPPED in html(at)
    assert not at.button(key="lc_live_start").disabled
    assert not at.warning and not at.error


def test_live_shows_the_drop_rate_and_says_what_drops_mean(lc, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_LIVE_RECORDS", "25")
    at = ok(ok(page()).button(key="lc_live_start").click().run())
    sess = controller(lc.settings).status()
    assert wait(lambda: any(r.get("type") == "stats" for r in records(sess.stream)))
    at = ok(at.run())
    assert "飛ばした映像 0%" in screen_text(at)
    assert lv.LIVE_DROPS_NOTE not in screen_text(at)        # nothing dropped: no warning
    assert wait(lambda: stopped_cleanly(lc), timeout=20)
    # the same demo where the machine was busy: frames were dropped
    monkeypatch.setenv("FAKE_ENGINE_LIVE_DROPPED", "6")
    at = ok(ok(page()).button(key="lc_live_start").click().run())
    sess = controller(lc.settings).status()
    assert wait(lambda: any(r.get("type") == "stats" and r.get("dropped")
                            for r in records(sess.stream)))
    at = ok(at.run())
    assert [i.value for i in at.info] == [lv.LIVE_DROPS_NOTE]   # nothing else busy: neutral
    assert re.search(r"飛ばした映像 \d+(\.\d)?%", screen_text(at))
    assert "リアルタイム" not in texts(at) and "デモ" in texts(at)
    assert wait(lambda: stopped_cleanly(lc), timeout=20)


def test_live_says_busy_only_when_other_work_was_running(lc, monkeypatch):
    """Drops while other work kept the machine busy (gpu.machine_busy, checked at most every
    lv.LIVE_LOAD_CHECK_SEC while frames are dropped): 「混んでいるため…」."""
    seen = []

    def busy(pid):
        seen.append(pid)
        return True
    monkeypatch.setattr(gpu, "machine_busy", busy)
    monkeypatch.setenv("FAKE_ENGINE_LIVE_DROPPED", "6")
    monkeypatch.setenv("FAKE_ENGINE_LIVE_RECORDS", "25")
    at = ok(ok(page()).button(key="lc_live_start").click().run())
    sess = controller(lc.settings).status()
    assert wait(lambda: any(r.get("type") == "stats" and r.get("dropped")
                            for r in records(sess.stream)))
    at = ok(at.run())
    assert [i.value for i in at.info] == [lv.LIVE_DROPS_NOTE_BUSY]
    assert seen and seen[0] == sess.pid               # the live run itself is not other work
    assert wait(lambda: stopped_cleanly(lc), timeout=20)


def test_live_needs_gpu_copy(lc, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    monkeypatch.setenv("FAKE_ENGINE_ERROR",
                       "--device cuda: GPU 0 has 1000 MiB free < 38912 (32768 reserve + 6144)")
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    ctl = controller(lc.settings)
    assert wait(lambda: ctl.status() is None and ctl.last_outcome() == "needs_gpu")
    at = ok(at.run())
    assert [w.value for w in at.warning] == [lv.LIVE_NO_GPU]
    assert not at.error
    assert "MiB" not in texts(at) and "GPU 0" not in texts(at)
    assert lv.LIVE_STOPPED in html(at) and not at.button(key="lc_live_start").disabled
    # an old failure is not news: the banner goes after LIVE_OUTCOME_SHOWN_SEC
    old = time.time() - lv.LIVE_OUTCOME_SHOWN_SEC - 60
    os.utime(ctl.exit_record, (old, old))
    at = ok(at.run())
    assert not at.warning and not at.error and lv.LIVE_STOPPED in html(at)


def test_live_error_copy(lc, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    monkeypatch.setenv("FAKE_ENGINE_ERROR", "RuntimeError: boom in live_run")
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    ctl = controller(lc.settings)
    assert wait(lambda: ctl.status() is None and ctl.last_outcome() == "error")
    at = ok(at.run())
    assert [e.value for e in at.error] == [lv.LIVE_FAILED]
    for raw in ("boom", "RuntimeError", "Traceback", "live_run", "fake"):
        assert raw not in texts(at), raw


def test_live_start_failed_copy(lc, monkeypatch):
    def broken(self, *args, **kwargs):
        raise EngineError("エンジンがセットアップされていません: /secret/engines/pfb3x3")
    monkeypatch.setattr(LiveController, "start", broken)
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    assert [w.value for w in at.warning] == [lv.LIVE_START_FAILED]
    assert "secret" not in texts(at) and "エンジン" not in texts(at)
    assert controller(lc.settings).status() is None and lv.LIVE_STOPPED in html(at)


def test_live_run_started_from_another_session(lc):
    """A run started elsewhere (another browser, or this page before a reload) is followed:
    the radio shows its video, the caption names it, and 「停止」 stops it."""
    sid, video = _sports_group(lc)
    at = ok(page())
    assert at.radio(key="lc_live_source").value == "demo"
    sess = controller(lc.settings).start(video, video_id=f"live_s{sid}_cam1")
    assert wait(lambda: len(records(sess.stream)) >= 12)
    at = ok(at.run())
    radio = at.radio(key="lc_live_source")
    assert radio.value == f"s{sid}_cam1" and radio.disabled
    assert "実行中の動画: 練習試合（A班）カメラ1" in [c.value for c in at.caption]
    assert lv.LIVE_RUNNING in html(at) and map_svgs(at) and "まで" in html(at)
    at = ok(at.button(key="lc_live_stop").click().run())
    assert wait(lambda: stopped_cleanly(lc), timeout=30) and not alive(sess.pid)
    at = ok(at.run())
    assert lv.LIVE_STOPPED in html(at) and not at.warning and not at.error


def test_live_stale_stop_leaves_the_next_run_alone(lc):
    """「停止」 was drawn for run A; A is stopped and run B started elsewhere before the
    press arrives: the press must not stop B."""
    at = ok(page())
    at = ok(at.button(key="lc_live_start").click().run())
    ctl = controller(lc.settings)
    a = ctl.status()
    assert a is not None and "lc_live_stop" in keys(at)
    assert ctl.stop() is True
    b = ctl.start(lc.demo, tid_video="IMG_0105")
    assert b.session != a.session
    at = ok(at.button(key="lc_live_stop").click().run())   # the button drawn for A
    assert wait(lambda: not ctl.stopping(), timeout=10)
    assert ctl.status() == b and alive(b.pid)
    at = ok(at.run())
    assert lv.LIVE_RUNNING in html(at) and "lc_live_stop" in keys(at)


def test_live_no_demo_configured(lc):
    lc.configure(demo_video=None)
    at = ok(page())
    assert lv.LIVE_NO_DEMO in [i.value for i in at.info]
    assert "linecall_demo_video" not in lv.LIVE_NO_DEMO and "設定:" not in lv.LIVE_NO_DEMO
    assert at.button(key="lc_live_start").disabled
    assert lc.fe.calls() == []


def test_live_demo_video_missing(lc):
    lc.configure(demo_video=lc.tmp / "gone" / "IMG_0105.MOV")
    at = ok(page())
    assert lv.LIVE_VIDEO_MISSING in [i.value for i in at.info]
    assert "gone" not in texts(at)   # no path shown
    assert at.button(key="lc_live_start").disabled


def _sports_group(lc, domain="sports", title="練習試合") -> tuple[int, Path]:
    aid = lc.repo.create_assignment(title, "video", "3x3の試合", "", "standard", True,
                                    domain=domain,
                                    sport="basketball" if domain == "sports" else None)
    video = make_video(lc.settings.inbox_dir / f"{title}.mp4", seconds=2)
    sid = register_video_group(lc.repo, lc.settings, aid, "A班", {"cam1": video}, 0.0, None,
                               "", False, [], engine="pfb3x3")
    return sid, video


def test_live_camera_source(lc):
    sid, video = _sports_group(lc)
    _sports_group(lc, domain="school", title="グループ活動")   # not a sports work: not offered
    at = ok(page())
    radio = at.radio(key="lc_live_source")
    assert list(radio.options) == ["デモ動画", "練習試合（A班）カメラ1"]
    at = ok(radio.set_value(f"s{sid}_cam1").run())
    at = ok(at.button(key="lc_live_start").click().run())
    argv = live_argv(lc)
    assert opt(argv, "--video-id") == f"live_s{sid}_cam1" and "--tid-video" not in argv
    entry = yaml.safe_load((lc.settings.data_dir / "live" / "pfb3x3" / "videos"
                            / f"live_s{sid}_cam1.yaml").read_text("utf-8"))
    assert entry["path"] == str(video) and entry["domain"] == "sports"
    assert at.radio(key="lc_live_source").disabled   # no switching while it runs


def test_live_sources_without_demo(lc):
    lc.configure(demo_video=None)
    sid, _video = _sports_group(lc)
    at = ok(page())
    # the demo stays listed (with its note) so the choice is visible
    assert list(at.radio(key="lc_live_source").options) == ["デモ動画", "練習試合（A班）カメラ1"]
    at = ok(at.radio(key="lc_live_source").set_value(f"s{sid}_cam1").run())
    assert not at.button(key="lc_live_start").disabled and not at.info


def test_live_page_wording(lc, monkeypatch):
    """Idle, running (alerts on screen) and stopped: no internal, school or ruling words."""
    monkeypatch.setenv("FAKE_ENGINE_LIVE_RECORDS", "25")
    _sports_group(lc)
    at = ok(page())
    shots = [texts(at)]
    at = ok(at.button(key="lc_live_start").click().run())
    sess = controller(lc.settings).status()
    assert wait(lambda: len(records(sess.stream)) >= 15)
    at = ok(at.run())
    shots.append(texts(at))
    assert wait(lambda: stopped_cleanly(lc), timeout=20)
    shots.append(texts(at := ok(at.run())))
    for text in shots:
        for word in (*FORBIDDEN, *SCHOOL_WORDS, *RULING_WORDS, "ルーブリック"):
            assert word not in text, word


# -- 試合後の判定材料 --------------------------------------------------------------------------

REVIEW_PAGE = UI / "pages" / "linecall_review.py"
EPISODE_IDS = ("A6-sideline_L-120", "P3-baseline-900", "B1-arc-2400", "P5-top_line-3000")


def review_page(session: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(REVIEW_PAGE), default_timeout=60)
    for key, value in (session or {}).items():
        at.session_state[key] = value
    return at.run()


def episode_keys(at: AppTest) -> list[str]:
    """The episode ids on screen, in order (one 「正しい」 button each)."""
    return [b.key[3:-len("_correct")] for b in at.button
            if b.key and b.key.startswith("lc_") and b.key.endswith("_correct")]


def in_order(at: AppTest) -> list:
    """Every element of the page, depth first in the order they are drawn."""
    out = []

    def walk(node):
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for k in sorted(children):
                walk(children[k])
        else:
            out.append(node)

    walk(at._tree)
    return out


def position(at: AppTest, *, key: str | None = None, warning: str | None = None) -> int:
    """Where the button `key` (or the warning with text `warning`) is drawn."""
    for i, el in enumerate(in_order(at)):
        if key is not None and getattr(el, "type", None) == "button" and el.key == key:
            return i
        if warning is not None and getattr(el, "type", None) == "warning" \
                and getattr(el, "value", None) == warning:
            return i
    raise AssertionError((key, warning))


def tracked_group(lc, cam="cam1"):
    from test_linecall_stage import tracked
    return tracked(lc.settings, lc.repo, lc.tmp, cam=cam)


def built(lc, cam="cam1"):
    """A tracked sports group whose camera went through the fake engine's linecall."""
    from test_linecall_stage import run_linecall_job
    aid, sid = tracked_group(lc, cam)
    assert run_linecall_job(lc.settings, lc.repo, sid, cam).status.value == "done"
    return aid, sid


def lc_dir(lc, sid, cam="cam1") -> Path:
    from pfb_app.services import linecall
    return linecall.episodes_path(lc.settings, lc.repo, sid, cam).parent


def write_episodes(lc, sid, eps, cam="cam1") -> Path:
    d = lc_dir(lc, sid, cam)
    d.mkdir(parents=True, exist_ok=True)
    (d / "episodes.json").write_text(json.dumps({"status": "判定材料", "episodes": eps},
                                                ensure_ascii=False), encoding="utf-8")
    return d


def synthetic(i, verdict="stepped", **extra):
    return {"id": f"P{i}-arc-{i}", "label": f"P{i}", "line": "arc", "verdict": verdict,
            "clock": None, "video_s": float(i), "crop": None, "overlay": None, "clip": None,
            "distances": {"L": None, "R": None}, "uncertainty_cm": {"L": None, "R": None},
            **extra}


def test_review_lists_fake_episodes(lc):
    _, sid = built(lc)
    at = ok(review_page())
    assert at.title[0].value == "試合後の判定材料"
    assert lv.NOTE in screen_text(at)
    assert episode_keys(at) == list(EPISODE_IDS)
    text, body = screen_text(at), html(at)
    for word in ("踏んだ", "踏んでいない", "判定不能", "Aチーム #6", "3番の人", "5番の人",
                 "サイドライン（左）", "2ポイントライン"):
        assert word in text, word
    # every verdict chip comes after 「判定材料」, with its words (never colour only)
    chips = re.findall(r'判定材料 <span class="pfb-chip pfb-chip--(\w+)">'
                       r'<span class="pfb-chip__mark" aria-hidden="true"></span>([^<]+)</span>',
                       body)
    assert [w for _, w in chips] == ["踏んだ", "踏んでいない", "判定不能", "判定不能"]
    assert len({k for k, _ in chips}) == 3
    assert lv.REVIEW_GUIDE in text
    assert "左足: 線の上（誤差 ±4 cm）　右足: 線の内側（コート側）に12 cm（誤差 ±4 cm）" in text
    assert "左足: 測れませんでした　右足: 線の内側（ゴール側）に1 cm（誤差 ±4 cm）" in text
    assert "判定不能の理由: 足が隠れています" in text
    # crop + overlay of the three episodes with a frame (files inside linecall/)
    assert len(at.get("image")) == 6
    # the 4th episode has no frame: no photo and no clip, said in words
    assert text.count(lv.NO_PHOTO) == 1 and text.count(lv.NO_CLIP) == 1
    assert [e.label for e in at.expander] == [lv.WATCH_CLIP] * 3
    assert not at.get("video")            # clips load only when opened
    # one primary button: 判定材料を作る, ready again after the run
    assert [b.label for b in at.button if b.proto.type == "primary"] == [lv.BUILD]
    assert not at.button(key="lc_build").disabled
    assert "コーチの確認: 0件（あと50件で一致率を表示）" in text
    assert not at.warning and not at.error and not at.exception


CHIP = (r'判定材料 <span class="pfb-chip pfb-chip--(\w+)">'
        r'<span class="pfb-chip__mark" aria-hidden="true"></span>([^<]+)</span>')


def test_review_sideline_and_near_line(lc):
    """Engine 58f8341 outputs (geometry.json, T11 evidence rows): with the camera behind the top
    line, the sideline 踏んだ is a plain 踏んだ with the sideline note; a 踏んだ at the top line
    (toward the camera) is 要確認 with its reason, and its buttons store the engine's verdict."""
    _, sid = built(lc)
    at = ok(review_page())
    text = screen_text(at)
    assert [w for _, w in re.findall(CHIP, html(at))][:2] == ["踏んだ", "踏んでいない"]
    assert text.count(lv.SIDELINE_NOTE) == 1 and lv.NEAR_LINE_NOTE not in text

    d = lc_dir(lc, sid)
    doc = json.loads((d / "episodes.json").read_text(encoding="utf-8"))
    near = {**doc["episodes"][0], "id": "P9-top_line-50", "label": "P9", "line": "top_line"}
    doc["episodes"] = [near, *doc["episodes"]]
    (d / "episodes.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with open(d / "frames.jsonl", "a", encoding="utf-8") as fh:   # its court point (T11 row)
        fh.write(json.dumps({"episode": near["id"], "frame": near["frame"], "foot": "L",
                             "near_court": [700.0, 1090.0], "hi_raw": -2.0,
                             "frame_would_step": True}) + "\n")
    at = ok(review_page())
    chips = re.findall(CHIP, html(at))
    assert chips[0] == ("guess", "踏んだ（要確認）") and chips[1][1] == "踏んだ"
    text = screen_text(at)
    assert text.count(lv.NEAR_LINE_NOTE) == 1
    for word in RULING_WORDS:
        assert word not in text, word
    at.button(key="lc_P9-top_line-50_correct").click().run()
    assert lc.repo.linecall_label_records(sid, "cam1")["P9-top_line-50"] == {
        "label": "correct", "verdict": "stepped"}


def test_review_without_geometry_checks_every_step(lc, monkeypatch):
    """No geometry.json: where the camera is cannot be told, so a 踏んだ is 要確認."""
    monkeypatch.setenv("FAKE_ENGINE_LC_GEOMETRY", "none")
    _, sid = built(lc)
    assert not (lc_dir(lc, sid) / "geometry.json").exists()
    at = ok(review_page())
    words = [w for _, w in re.findall(CHIP, html(at))]
    assert words == ["踏んだ（要確認）", "踏んでいない", "判定不能", "判定不能"]
    text = screen_text(at)
    assert lv.NEAR_LINE_NOTE in text and lv.SIDELINE_NOTE in text
    assert "踏んだ" not in words


def test_review_clip_opens_lazily(lc):
    _, sid = built(lc)
    at = ok(review_page({f"lc_clip_{EPISODE_IDS[0]}": True}))
    videos = at.get("video")
    assert len(videos) == 1


def test_label_buttons_store(lc):
    _, sid = built(lc)
    at = ok(review_page())
    assert [b.label for b in at.button if b.key.startswith(f"lc_{EPISODE_IDS[1]}_")] == [
        "正しい", "違う", "判定不能"]
    at = ok(at.button(key=f"lc_{EPISODE_IDS[1]}_wrong").click().run())
    assert lc.repo.linecall_labels(sid, "cam1") == {EPISODE_IDS[1]: "wrong"}
    assert "コーチの確認: 違う" in screen_text(at)
    assert "コーチの確認: 1件（あと49件で一致率を表示）" in screen_text(at)
    at = ok(at.button(key=f"lc_{EPISODE_IDS[1]}_correct").click().run())   # last press wins
    assert lc.repo.linecall_labels(sid, "cam1") == {EPISODE_IDS[1]: "correct"}
    assert "コーチの確認: 正しい" in screen_text(at) and "コーチの確認: 違う" not in screen_text(at)
    at = ok(at.button(key=f"lc_{EPISODE_IDS[3]}_undecidable").click().run())
    assert lc.repo.linecall_labels(sid, "cam1")[EPISODE_IDS[3]] == "undecidable"
    mirror = json.loads((lc_dir(lc, sid) / "labels.json").read_text("utf-8"))
    assert mirror == {EPISODE_IDS[1]: {"label": "correct", "verdict": "clear"}}


def test_label_for_a_changed_verdict_reruns(lc):
    """The episodes were rebuilt after the page was drawn: the press stores nothing and the
    page says the material is new, in that episode's box."""
    _, sid = built(lc)
    at = ok(review_page())
    at = ok(at.button(key=f"lc_{EPISODE_IDS[1]}_correct").click().run())   # unchanged: stored
    path = lc_dir(lc, sid) / "episodes.json"
    doc = json.loads(path.read_text("utf-8"))
    doc["episodes"][2]["verdict"] = "stepped"         # was 判定不能 when the page was drawn
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    at = ok(at.button(key=f"lc_{EPISODE_IDS[2]}_correct").click().run())
    assert lc.repo.linecall_labels(sid, "cam1") == {EPISODE_IDS[1]: "correct"}
    assert [w.value for w in at.warning] == [lv.STALE_VERDICT]
    assert [t.value for t in at.toast] == [lv.STALE_VERDICT]
    # drawn in that episode's box, above its buttons (after the episode before it)
    here = position(at, warning=lv.STALE_VERDICT)
    assert position(at, key=f"lc_{EPISODE_IDS[1]}_undecidable") < here \
        < position(at, key=f"lc_{EPISODE_IDS[2]}_correct")
    assert screen_text(at).count("コーチの確認: 正しい") == 1   # the other episode's label
    at = ok(at.run())                                          # shown once
    assert not at.warning and not at.toast


def test_stale_label_is_not_shown_as_current(lc):
    """A label given for the old verdict (a rerun flipped it) is not shown for the new one."""
    _, sid = built(lc)
    lc.repo.set_linecall_label(sid, "cam1", EPISODE_IDS[0], "correct", verdict="clear")
    lc.repo.set_linecall_label(sid, "cam1", EPISODE_IDS[1], "wrong", verdict="clear")
    text = screen_text(ok(review_page()))
    assert "コーチの確認: 正しい" not in text and "コーチの確認: 違う" in text


def test_build_button_queues_and_shows_progress(lc):
    from pfb_app.services import linecall
    _, sid = tracked_group(lc)
    at = ok(review_page())
    assert lv.NO_EPISODES in [i.value for i in at.info]
    build = at.button(key="lc_build")
    assert build.label == lv.BUILD and build.proto.type == "primary" and not build.disabled
    at = ok(build.click().run())
    assert linecall.job_state(lc.repo, sid, "cam1") == "active"
    assert lv.BUILDING in html(at)
    assert at.button(key="lc_build").disabled
    # no worker heartbeat ever: the page says the worker seems stopped (own check)
    from pfb_app.ui.wizard.live import WORKER_STOPPED
    assert WORKER_STOPPED in [w.value for w in at.warning]


def test_build_says_not_ready_without_the_pose_weights(lc):
    _, sid = tracked_group(lc)
    lc.pose.unlink()
    at = ok(review_page())
    assert at.button(key="lc_build").disabled
    assert lv.NOT_READY in [w.value for w in at.warning]
    assert "準備ができていません" in lv.NOT_READY
    assert lv.BUILD_FAILED not in texts(at)

def test_build_needs_a_tracked_camera(lc):
    _sports_group(lc)
    at = ok(review_page())
    assert at.button(key="lc_build").disabled
    assert lv.NEEDS_TRACK in screen_text(at)


def test_build_refused_while_the_work_is_being_deleted(lc):
    from pfb_app.services import deletion
    _, sid = tracked_group(lc)
    at = ok(review_page())
    sub = lc.repo.get_submission(sid)
    lc.repo.update_submission_meta(sid, {**sub["meta"], deletion.REQUESTED: True})
    at = ok(at.button(key="lc_build").click().run())          # requested after the draw
    assert lc.repo.latest_job_row(sid, Stage.linecall, "cam1") is None
    assert [w.value for w in at.warning] == [lv.NOT_QUEUED]
    assert [t.value for t in at.toast] == [lv.NOT_QUEUED]
    assert position(at, key="lc_build") < position(at, warning=lv.NOT_QUEUED)
    # drawn again: the button is off and says why
    at = ok(at.run())
    assert at.button(key="lc_build").disabled and not at.warning
    assert lv.NOT_QUEUED in [c.value for c in at.caption]


def test_build_says_why_for_removed_videos_and_an_erasure(lc):
    """Final review LOW 9: a work whose videos were removed, or a group whose person is being
    erased, says that (not 削除の手続き中), and the button is off."""
    from datetime import datetime, timezone
    _, sid = tracked_group(lc)
    aid = lc.repo.get_submission(sid)["assignment_id"]
    at = ok(review_page())
    lc.repo.conn.execute("UPDATE assignments SET media_removed_at=? WHERE id=?",
                         (datetime.now(timezone.utc).isoformat(), aid))
    at = ok(at.button(key="lc_build").click().run())          # removed after the draw
    assert lc.repo.latest_job_row(sid, Stage.linecall, "cam1") is None
    assert [w.value for w in at.warning] == [lv.MEDIA_REMOVED]
    at = ok(at.run())
    assert at.button(key="lc_build").disabled
    assert lv.MEDIA_REMOVED in [c.value for c in at.caption]
    assert lv.NOT_QUEUED not in screen_text(at)
    lc.repo.conn.execute("UPDATE assignments SET media_removed_at=NULL WHERE id=?", (aid,))
    sub = lc.repo.get_submission(sid)
    lc.repo.update_submission_meta(sid, {**sub["meta"],
                                         "erasing": datetime.now(timezone.utc).isoformat()})
    at = ok(review_page())
    assert at.button(key="lc_build").disabled
    assert lv.ERASING in [c.value for c in at.caption]


def test_watch_rereads_only_on_its_ticks(lc, monkeypatch):
    """While the job is active the page's watcher does not read the DB on the page run (its
    first tick); its signature is the job state and whether the worker seems stopped."""
    from pfb_app.services import linecall
    from pfb_app.storage import files as storage_files
    from pfb_app.ui import common
    _, sid = tracked_group(lc)
    linecall.request(lc.repo, sid, "cam1")
    opened = []
    real = common.get_repo
    monkeypatch.setattr(common, "get_repo", lambda: opened.append(1) or real())
    at = ok(review_page())
    assert lv.BUILDING in html(at) and len(opened) == 1     # the page's own read only
    assert lv.watch_signature(lc.settings, lc.repo, sid, "cam1") == ("active", True)
    beat = storage_files.worker_beat_path(lc.settings)
    beat.parent.mkdir(parents=True, exist_ok=True)
    beat.touch()
    assert lv.watch_signature(lc.settings, lc.repo, sid, "cam1") == ("active", False)
    at = ok(at.run())
    assert not at.warning                                    # the worker is alive now


def test_failed_job_copy(lc, monkeypatch):
    from pfb_app.services import linecall
    from test_linecall_stage import run_linecall_job
    _, sid = tracked_group(lc)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    job = run_linecall_job(lc.settings, lc.repo, sid)
    assert linecall.job_state(lc.repo, sid, "cam1") == "failed" and job.error
    at = ok(review_page())
    text = texts(at)
    assert lv.BUILD_FAILED in [e.value for e in at.error]
    assert lv.NO_EPISODES not in text
    for raw in ("EngineError", "error", "exit", "Traceback", job.error.splitlines()[0]):
        assert raw not in text, raw
    assert not at.button(key="lc_build").disabled          # pressing again queues anew


def test_failed_rerun_keeps_the_old_episodes(lc, monkeypatch):
    from test_linecall_stage import run_linecall_job
    _, sid = built(lc)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    run_linecall_job(lc.settings, lc.repo, sid)
    at = ok(review_page())
    assert lv.BUILD_FAILED in [e.value for e in at.error]
    assert episode_keys(at) == list(EPISODE_IDS)


def with_photos(lc, sid, eps, cam="cam1"):
    """Tiny synthetic JPEG bytes for every episode's crop and overlay."""
    from fakes.fake_engine import TINY_JPEG
    d = lc_dir(lc, sid, cam)
    for e in eps:
        e["crop"], e["overlay"] = f"crops/{e['id']}/1.jpg", f"crops/{e['id']}/1_overlay.jpg"
        for rel in (e["crop"], e["overlay"]):
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(TINY_JPEG)
    return eps


def test_episodes_paged_by_20(lc):
    _, sid = tracked_group(lc)
    write_episodes(lc, sid, with_photos(lc, sid, [synthetic(i) for i in range(1, 46)]))
    at = ok(review_page())
    assert episode_keys(at) == [f"P{i}-arc-{i}" for i in range(1, 21)]
    assert "1〜20件目（全45件）" in screen_text(at)
    assert "lc_prev" not in keys(at) and "lc_prev_top" not in keys(at)
    assert len(at.get("image")) == 40                       # crop + overlay of 20
    assert position(at, key="lc_next_top") < position(at, key="lc_P1-arc-1_correct")
    assert position(at, key="lc_next") > position(at, key="lc_P20-arc-20_undecidable")
    at = ok(at.button(key="lc_next").click().run())
    assert at.button(key="lc_next").label == lv.NEXT_PAGE
    assert episode_keys(at) == [f"P{i}-arc-{i}" for i in range(21, 41)]
    at = ok(at.button(key="lc_next").click().run())
    assert episode_keys(at) == [f"P{i}-arc-{i}" for i in range(41, 46)]
    assert "lc_next" not in keys(at) and "41〜45件目（全45件）" in screen_text(at)
    at = ok(at.button(key="lc_prev_top").click().run())
    assert episode_keys(at) == [f"P{i}-arc-{i}" for i in range(21, 41)]
    assert len(at.get("image")) == 40 and lv.NO_PHOTO not in screen_text(at)


def test_episodes_without_photos_say_so(lc):
    _, sid = tracked_group(lc)
    write_episodes(lc, sid, [synthetic(i) for i in range(1, 4)])
    at = ok(review_page())
    assert len(at.get("image")) == 0
    assert screen_text(at).count(lv.NO_PHOTO) == 3 and screen_text(at).count(lv.NO_CLIP) == 3


def test_unreadable_or_odd_evidence_files(lc):
    """A photo that is not a picture, a clip that is not mp4, an unreadable crop: words."""
    _, sid = tracked_group(lc)
    d = write_episodes(lc, sid, [synthetic(1, crop="crops/1/a.txt", overlay="crops/1/b.jpg",
                                           clip="clips/1.mov")])
    (d / "crops" / "1").mkdir(parents=True)
    (d / "clips").mkdir()
    (d / "crops" / "1" / "a.txt").write_text("x")
    (d / "crops" / "1" / "b.jpg").mkdir()          # a directory named like a photo
    (d / "clips" / "1.mov").write_bytes(b"x")
    at = ok(review_page({"lc_clip_P1-arc-1": True}))
    assert not at.get("image") and not at.get("video")
    assert lv.NO_PHOTO in screen_text(at) and lv.NO_CLIP in screen_text(at)


def test_evidence_paths_never_leave_linecall(lc):
    _, sid = tracked_group(lc)
    outside = lc.tmp / "outside.jpg"
    outside.write_bytes(b"\xff\xd8 not shown")
    d = write_episodes(lc, sid, [synthetic(1, crop="../../../../../../outside.jpg",
                                           overlay=str(outside), clip="../outside.mp4")])
    (d / "crops").mkdir()
    at = ok(review_page({"lc_clip_P1-arc-1": True}))
    assert not at.get("image") and not at.get("video")
    assert lv.NO_PHOTO in screen_text(at) and lv.NO_CLIP in screen_text(at)


def test_torn_episodes_file(lc):
    _, sid = tracked_group(lc)
    d = lc_dir(lc, sid)
    d.mkdir(parents=True)
    (d / "episodes.json").write_text('{"status": "判定材料", "episodes": [{"id"', "utf-8")
    at = ok(review_page())
    assert lv.NO_EPISODES in [i.value for i in at.info] and not at.error


def test_camera_picker(lc):
    aid = lc.repo.create_assignment("交流戦", "video", "3x3の試合", "", "standard", True,
                                    domain="sports", sport="basketball")
    v1 = make_video(lc.settings.inbox_dir / "c1.mp4", seconds=2)
    v2 = make_video(lc.settings.inbox_dir / "c2.mp4", seconds=2)
    sid = register_video_group(lc.repo, lc.settings, aid, "前半", {"cam1": v1, "cam2": v2},
                               0.0, None, "", False, [], engine="pfb3x3")
    write_episodes(lc, sid, [synthetic(7)], cam="cam2")
    at = ok(review_page())
    assert list(at.selectbox(key="lc_work").options) == ["交流戦"]
    assert list(at.selectbox(key="lc_group").options) == ["前半"]
    assert list(at.selectbox(key="lc_camera").options) == ["カメラ1", "カメラ2"]
    assert episode_keys(at) == []
    at = ok(at.selectbox(key="lc_camera").set_value("cam2").run())
    assert episode_keys(at) == ["P7-arc-7"]


def test_general_work_hidden(lc):
    _sports_group(lc, domain="school", title="グループ活動")
    at = ok(review_page())
    assert lv.NO_SPORTS_WORK in [i.value for i in at.info]
    assert "lc_build" not in keys(at) and not at.selectbox
    assert "グループ活動" not in screen_text(at)


def test_review_page_wording(lc):
    _, sid = built(lc)
    lc.repo.set_linecall_label(sid, "cam1", EPISODE_IDS[0], "wrong", verdict="stepped")
    for at in (ok(review_page()), ok(review_page({f"lc_clip_{EPISODE_IDS[0]}": True}))):
        text = texts(at) + "\n".join(e.label for e in at.expander)
        for word in (*FORBIDDEN, *SCHOOL_WORDS, *RULING_WORDS, "ルーブリック"):
            assert word not in text, word
