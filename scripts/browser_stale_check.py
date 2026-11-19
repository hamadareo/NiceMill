"""Browser check: step 5's elements must not stay on step 6 (stale elements, stuck tooltips).

Not run in CI. It builds a SYNTHETIC sports work in a temp data dir (invented names and texts,
tiny black PNGs, no video is ever read), creates a temp 管理者 with `pfb-admin create-admin`,
starts the app on 127.0.0.1:<free port> and drives the system Chrome headless over CDP:

  sign in -> the work's step 5 -> 「この文で承認する」 for each short-text person (with the
  pointer on the button, as a coach's click) -> 「続ける」 -> step 6 -> wait --wait seconds ->
  count the elements Streamlit marks stale (data-stale="true"), the leftover copies of keyed
  blocks (a `st-key-*` class on more than one node: a block React lost, see ui/keyed) and the
  tooltips still on screen with the pointer off the page. --decline-first presses
  「この人は入れない」 for the first person instead (its one-time message moves the step card).

    env -u VIRTUAL_ENV uv run python scripts/browser_stale_check.py [--wait 25] [--shot out.png]
        [--decline-first] [--verbose] [--trace]

Exit 0 when step 6 shows none of them, else 1. It stops only the processes it started and
removes its temp dirs (the Chrome profile too). The Chrome runs with a bare environment: with
the login session's (D-Bus, keyring) the headless Chrome on the server never loads a page.
"""
import argparse
import asyncio
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

CHROME = "/opt/google/chrome/chrome"
LOGIN, PASSWORD = "stale-check", "stale-check-password-1234"
NAMES = ["花子", "太郎", "次郎", "三郎", "四郎", "五郎", "六子", "七海"]
SHORT = 4                       # people on the short-text path; the others are approved
TOOLTIP = "この短い文を、この人の文章として承認します"


def say(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- the synthetic work ------------------------------------------------------------------------

def build(tmp: Path) -> tuple[Path, int]:
    """A temp config and a sports work at step 5: SHORT people with a short text offered, the
    others approved. Returns (config path, assignment id)."""
    import cv2
    import numpy as np
    import yaml

    from pfb_app.config import Settings
    from pfb_app.domain.models import Stage
    from pfb_app.services import review
    from pfb_app.services.review import confirm_identity, save_tracked_identity
    from pfb_app.storage import files
    from pfb_app.storage.db import Repo, connect, init_schema
    from test_wizard_ui_people_wait import finish_all, save

    settings = Settings(data_dir=tmp / "data", inbox_dir=tmp / "inbox",
                        engine_dir=tmp / "engines" / "pfb3x3")
    cfg = tmp / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir)}), encoding="utf-8")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("練習試合", "video", "サッカーの試合", "", "standard", True,
                                 domain="sports", sport="soccer")
    people = [(f"P{chr(97 + i)}", n) for i, n in enumerate(NAMES)]
    meta = {"cameras": {"cam1": "/v/cam1.mp4"}, "start_sec": 0.0, "end_sec": None,
            "auto_transcribe": False, "engine": "pfb3x3"}
    sid = repo.create_submission(aid, "A班", meta)
    for pid, name in people:
        repo.add_subject(sid, pid, name, "外見", {})
    labels = [f"P{i + 1}" for i in range(len(people))]
    sheet = settings.data_dir / f"sheet_{sid}.png"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet), np.zeros((24, 32, 3), dtype=np.uint8))
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.track, {"people": len(labels), "labels": labels, "sheet": str(sheet),
                                  "default_names": {}, "default_source": None},
         camera="cam1", upstream=[prep.id])
    save_tracked_identity(repo, sid, "cam1", dict(zip(labels, [p for p, _ in people])))
    confirm_identity(repo, sid)
    finish_all(repo, sid)
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    events, frames = [], {}
    for k, (pid, _) in enumerate(people):
        eid = f"cam1-{k + 1:04d}-01"
        events.append({"event_id": eid, "camera": "cam1", "video_time": f"00:{10 + k:02d}",
                       "task_elapsed": f"00:{10 + k:02d}", "participant": pid,
                       "observed_behavior": "ボールを前へ運んだ", "utterance": "",
                       "speaker": "unknown", "evidence_source": "video", "confidence": "high"})
        png = settings.data_dir / "frames" / f"{sid}_{eid}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png), np.zeros((12, 16, 3), dtype=np.uint8))
        frames[eid] = [str(png)]
    beh = save(repo, sid, Stage.behavior, {"events": events, "frames": frames}, camera="cam1",
               upstream=[ident.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": e["event_id"], "contribution_interpretation": "攻撃につながった",
         "is_positive_contribution": True, "rubric_criteria": []} for e in events]},
        camera="cam1", upstream=[beh.id])
    for k, s in enumerate(repo.list_subjects(sid)):
        if k < SHORT:
            out = {"pfb_status": "insufficient_positive_evidence", "pfb": None,
                   "evidence_refs": []}
        else:
            out = {"pfb_status": "generated", "pfb": f"{s['display_name']}さんはボールを前へ運びました。",
                   "evidence_refs": [events[k]["event_id"]]}
        save(repo, sid, Stage.pfb, out, subject_id=s["id"], upstream=[con.id])
        if k >= SHORT:
            review.approve(repo, s["id"])
    conn.commit()
    conn.close()
    return cfg, aid


# --- CDP -----------------------------------------------------------------------------------

class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    frames: list = []          # Streamlit's ForwardMsgs as received (with --trace)

    def _event(self, msg: dict) -> None:
        if msg.get("method") == "Network.webSocketFrameReceived":
            self.frames.append((time.time(), msg["params"]["response"]))

    async def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            self._event(msg)
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                            awaitPromise=True)
        return r.get("result", {}).get("value")

    async def wait(self, expr: str, timeout: float = 60.0, what: str = ""):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expr):
                return
            await asyncio.sleep(0.25)
        text = await self.js("document.body ? document.body.innerText.slice(0, 600) : ''")
        raise TimeoutError(f"{what or expr}; page: {text!r}")

    async def idle(self, quiet: float = 1.5, timeout: float = 60.0):
        """Wait until Streamlit is not running a script for `quiet` seconds."""
        expr = ("!document.querySelector('[data-testid=\"stStatusWidget\"] "
                "[data-testid=\"stStatusWidgetRunningIcon\"]') && "
                "!!document.querySelector('[data-testid=\"stApp\"]')")
        end, since = time.time() + timeout, None
        while time.time() < end:
            if await self.js(expr):
                since = since or time.time()
                if time.time() - since >= quiet:
                    return
            else:
                since = None
            await asyncio.sleep(0.25)
        raise TimeoutError("the page never went idle")

    async def center(self, selector_js: str):
        return await self.js(f"""(() => {{ const el = {selector_js};
            if (!el) return null; el.scrollIntoView({{block: 'center'}});
            const r = el.getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2]; }})()""")

    async def click(self, selector_js: str, what: str):
        """A real pointer click: move onto the element (its tooltip opens), press, release.
        The pointer then stays where it was, as a coach's mouse does."""
        xy = await self.center(selector_js)
        if xy is None:
            raise RuntimeError(f"not on screen: {what}")
        x, y = xy
        await self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        await asyncio.sleep(1.2)                      # the help tooltip shows on hover
        for t in ("mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left",
                            clickCount=1)

    async def type_into(self, selector_js: str, text: str):
        await self.js(f"{selector_js}.focus()")
        await self.call("Input.insertText", text=text)


def button(label: str) -> str:
    return (f"[...document.querySelectorAll('button')].find(b => b.innerText.trim() === "
            f"{json.dumps(label)} && !b.disabled && !b.closest('[data-stale=\"true\"]'))")


PROBE = """(() => {
  const stale = [...document.querySelectorAll('[data-stale="true"]')];
  const tips = [...document.querySelectorAll('[data-testid="stTooltipContent"], [role="tooltip"]')]
    .filter(e => e.offsetParent !== null || e.getClientRects().length);
  return {
    stale: stale.length,
    stale_text: stale.map(e => e.innerText.trim().slice(0, 40)).filter(Boolean).slice(0, 12),
    stale_where: stale.slice(0, 5).map(e => { const out = []; let n = e;
      while (n && n !== document.body && out.length < 14) {
        const key = [...(n.classList || [])].find(c => c.startsWith('st-key-'));
        const tid = n.getAttribute && n.getAttribute('data-testid');
        if (key || tid) out.push((tid || '') + (key ? '.' + key : '') + (n.getAttribute('data-stale') ? '[stale]' : ''));
        n = n.parentElement; }
      return out.join(' < '); }),
    // a key is on one block or element per run: a second copy is a leftover React lost
    dupes: (() => { const n = {}; document.querySelectorAll('[class*="st-key-"]').forEach(e =>
      [...e.classList].filter(c => c.startsWith('st-key-')).forEach(c => { n[c] = (n[c] || 0) + 1; }));
      return Object.entries(n).filter(([, v]) => v > 1).map(([k, v]) => `${k} x${v}`); })(),
    stale_short_buttons: stale.filter(e => e.innerText.includes('この文で承認する')).length,
    tooltips: tips.map(e => e.innerText.trim()).filter(Boolean),
    step6: document.body.innerText.includes('PDFを受け取ってください'),
  };
})()"""


def summarize(frames: list) -> list[str]:
    """Each Streamlit ForwardMsg in a line: new_session (fragment ids), script_finished,
    deltas counted per run."""
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
    out, deltas = [], 0
    t0 = frames[0][0] if frames else 0
    for t, resp in frames:
        if resp.get("opcode") != 2:
            continue
        msg = ForwardMsg()
        try:
            msg.ParseFromString(base64.b64decode(resp["payloadData"]))
        except Exception:   # noqa: BLE001
            continue
        kind = msg.WhichOneof("type")
        if kind == "delta":
            deltas += 1
            continue
        if deltas:
            out.append(f"  {deltas} deltas")
            deltas = 0
        if kind == "new_session":
            ns = msg.new_session
            out.append(f"{t - t0:7.2f} new_session run={ns.script_run_id[:8]} "
                       f"fragments={list(ns.fragment_ids_this_run)}")
        elif kind == "script_finished":
            out.append(f"{t - t0:7.2f} script_finished "
                       f"{ForwardMsg.ScriptFinishedStatus.Name(msg.script_finished)}")
        elif kind not in ("page_info_changed", "page_profile"):
            out.append(f"{t - t0:7.2f} {kind}")
    return out


async def drive(port: int, cdp: int, aid: int, wait_sec: float, shot: Path | None,
                trace: bool = False, decline_first: bool = False,
                verbose: bool = False) -> dict:
    import websockets
    base = f"http://127.0.0.1:{port}"
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{cdp}/json").read())
    ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
    async with websockets.connect(ws_url, max_size=2 ** 26) as ws:
        page = Page(ws)
        await page.call("Page.enable")
        await page.call("Runtime.enable")
        if trace:
            await page.call("Network.enable")
        await page.call("Emulation.setDeviceMetricsOverride", width=1280, height=1600,
                        deviceScaleFactor=1, mobile=False)
        say("navigate"); await page.call("Page.navigate", url=f"{base}/?work={aid}")
        await page.wait("!!document.querySelector('input[type=password]')", 90, "sign-in form")
        say("sign-in form"); await page.idle()
        inputs = "document.querySelectorAll('input')"
        await page.type_into(f"[...{inputs}].find(i => i.type !== 'password')", LOGIN)
        await page.type_into(f"[...{inputs}].find(i => i.type === 'password')", PASSWORD)
        await page.call("Input.dispatchKeyEvent", type="keyDown", key="Tab", code="Tab")
        await page.call("Input.dispatchKeyEvent", type="keyUp", key="Tab", code="Tab")
        await asyncio.sleep(0.5)
        await page.click(button("ログイン") + " || " + "[...document.querySelectorAll('button')]"
                         ".find(b => b.innerText.includes('ログイン'))", "ログイン")
        say("signing in"); await page.wait(f"!!({button('この文で承認する')})", 90, "step 5 with a short text")
        await page.idle()
        for k in range(SHORT):
            label = "この人は入れない" if decline_first and k == 0 else "この文で承認する"
            say(label, k + 1); await page.click(button(label), f"{label} #{k + 1}")
            await asyncio.sleep(0.5)
            await page.idle()
            probe = await page.js(PROBE)
            say("  stale", probe["stale"], "leftover copies", probe["dupes"],
                "tooltips", len(probe["tooltips"]))
            if verbose:
                text = await page.js("document.body.innerText")
                say("  page:", text[text.find("A班"):][:700].replace("\n", " | "))
        await page.wait(f"!!({button('続ける')})", 60, "「続ける」 after the last approval")
        before = await page.js(PROBE)
        say("continue"); await page.click(button("続ける"), "続ける")
        await page.wait("document.body.innerText.includes('PDFを受け取ってください')", 60, "step 6")
        await page.idle()
        say("step 6; waiting", wait_sec); await asyncio.sleep(wait_sec)
        # the pointer off the page: a tooltip still drawn now is a leftover, not a hover
        await page.call("Input.dispatchMouseEvent", type="mouseMoved", x=1, y=1)
        await asyncio.sleep(1.0)
        after = await page.js(PROBE)
        if shot is not None:
            data = await page.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            shot.write_bytes(base64.b64decode(data["data"]))
        if trace:
            print("\n".join(summarize(page.frames)), file=sys.stderr)
        return {"step5_before_continue": before, "step6_after_wait": after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=float, default=25.0)
    ap.add_argument("--shot", type=Path)
    ap.add_argument("--trace", action="store_true", help="print Streamlit's messages")
    ap.add_argument("--verbose", action="store_true", help="print step 5's text after each press")
    ap.add_argument("--decline-first", action="store_true",
                    help="「この人は入れない」 for the first person (its message moves the card)")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # the cleanup below still runs
    tmp = Path(tempfile.mkdtemp(prefix="pfb_stale_check_"))
    procs: list[subprocess.Popen] = []
    try:
        cfg, aid = build(tmp)
        env = {**os.environ, "PFB_CONFIG": str(cfg)}
        env.pop("VIRTUAL_ENV", None)
        subprocess.run([sys.executable, "-m", "pfb_app.admin_cli", "create-admin", "--login",
                        LOGIN, "--name", "確認用", "--password-stdin"], input=PASSWORD + "\n",
                       text=True, env=env, check=True, capture_output=True, cwd=ROOT)
        port, cdp = free_port(), free_port()
        log = open(tmp / "streamlit.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "src/pfb_app/ui/serve.py",
             "--server.address", "127.0.0.1", "--server.port", str(port),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        profile = tmp / "chrome-profile"
        procs.append(subprocess.Popen(
            [CHROME, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
             "--disable-gpu", "--remote-allow-origins=*", f"--remote-debugging-port={cdp}",
             f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
             "--no-proxy-server", "--password-store=basic", "--lang=ja", "about:blank"],
            # a bare environment: with the login session's (D-Bus, keyring) the headless
            # Chrome on the server never loads a page
            env={"HOME": str(tmp), "PATH": "/usr/bin:/bin"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cdp}/json/version", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        say("app on", port, "chrome cdp on", cdp)
        result = asyncio.run(asyncio.wait_for(drive(port, cdp, aid, args.wait, args.shot, args.trace,
                                                     args.decline_first, args.verbose), 400))
        print(json.dumps(result, ensure_ascii=False, indent=1))
        after = result["step6_after_wait"]
        bad = after["stale"] or after["dupes"] or after["tooltips"] or not after["step6"]
        print("FAIL" if bad else "OK")
        return 1 if bad else 0
    finally:
        for p in procs:              # only what this script started (its own process groups)
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
