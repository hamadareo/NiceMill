"""Phone-first lane: HTTPS-ready start script and device notifications (no network used)."""
import json
import re
import subprocess
from pathlib import Path

from pfb_app.ui.wizard import notify

ROOT = Path(__file__).resolve().parents[1]


def test_the_notice_rides_with_the_chime_and_needs_https():
    html = notify._html("（確認できます）A", ["1-abc"])
    data = json.loads(re.search(r'data-pfb-notify="([^"]*)"', html).group(1)
                      .replace("&quot;", '"'))
    assert data["notice"] == notify.NOTICE and data["chimes"] == ["1-abc"]
    assert "isSecureContext" in notify._JS and "requestPermission" in notify._JS
    assert "visibilityState" in notify._JS            # never while the tab is in view


def test_start_sh_keeps_the_live_address_by_default_and_can_bind_loopback():
    text = (ROOT / "scripts" / "start.sh").read_text()
    assert 'UI_HOST="${PFB_UI_HOST:-127.0.0.1}"' in text
    assert 'UI_PORT="${PFB_UI_PORT:-8501}"' in text
    assert "src/pfb_app/ui/serve.py" in text and "src/pfb_app/ui/app.py" in text
    for script in ("start.sh", "https_serve.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)


def test_https_serve_refuses_a_bad_port_without_touching_anything():
    r = subprocess.run(["bash", str(ROOT / "scripts" / "https_serve.sh"), "on", "--port", "80;x"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin",
                                                             "TAILSCALE": "/bin/false"})
    assert r.returncode == 2 and "ポート" in r.stderr


# --- https_serve.sh against a fake tailscale (review round 1, #7) -----------------------------

FAKE_TS = r'''#!/usr/bin/env python3
import json, os, sys
state_path = os.environ["FAKE_TS_STATE"]
state = json.load(open(state_path))
args = sys.argv[1:]
def save():
    json.dump(state, open(state_path, "w"))
if args[:2] == ["status", "--json"]:
    print(json.dumps({"Self": {"DNSName": "myhost.example.ts.net."}, "CertDomains": state["certs"]}))
elif args[:2] == ["debug", "prefs"]:
    print(json.dumps({"OperatorUser": state["operator"]}))
elif args[:3] == ["serve", "status", "--json"]:
    print(json.dumps(state["serve"]))
elif args[:2] == ["serve", "status"]:
    print("serve config")
elif args and args[0] == "serve":
    state["calls"].append(args[1:])
    port = next(a.split("=", 1)[1] for a in args if a.startswith("--https="))
    key = "myhost.example.ts.net:" + port
    web = state["serve"].setdefault("Web", {})
    if args[-1] == "off":
        web.pop(key, None)
    else:
        web[key] = {"Handlers": {"/": {"Proxy": args[-1]}}}
    save()
else:
    sys.exit(9)
'''


def run_https(tmp_path, args, *, serve=None, certs=("myhost.example.ts.net",), operator="tester"):
    import os
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "tailscale"
    fake.write_text(FAKE_TS)
    fake.chmod(0o755)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"certs": list(certs), "operator": operator,
                                 "serve": serve or {}, "calls": []}))
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FAKE_TS_STATE": str(state), "USER": "tester",
           "HOME": os.environ.get("HOME", "/tmp")}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "https_serve.sh"), *args,
                        "--port", "18499"], capture_output=True, text=True, env=env)
    return r, json.loads(state.read_text())


def mapped(proxy, port="443"):
    return {"Web": {f"myhost.example.ts.net:{port}": {"Handlers": {"/": {"Proxy": proxy}}}}}


def test_on_never_replaces_another_mapping(tmp_path):
    r, state = run_https(tmp_path, ["on"], serve=mapped("http://127.0.0.1:9000"))
    assert r.returncode == 1 and state["calls"] == []
    assert "ほかのもの" in r.stderr


def test_on_maps_a_free_port_and_leaves_the_same_mapping(tmp_path):
    r, state = run_https(tmp_path, ["on"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert state["calls"] == [["--bg", "--https=443", "http://127.0.0.1:18499"]]
    r, state = run_https(tmp_path, ["on"], serve=mapped("http://127.0.0.1:18499"))
    assert r.returncode == 0 and state["calls"] == [] and "すでに" in r.stdout


def test_on_refuses_without_certificates_or_operator(tmp_path):
    r, state = run_https(tmp_path, ["on"], certs=())
    assert r.returncode == 1 and state["calls"] == []
    r, state = run_https(tmp_path, ["on"], operator="")
    assert r.returncode == 1 and state["calls"] == []


def test_off_only_removes_this_apps_mapping(tmp_path):
    r, state = run_https(tmp_path, ["off"], serve=mapped("http://127.0.0.1:9000"))
    assert r.returncode == 1 and state["calls"] == []
    r, state = run_https(tmp_path, ["off"])
    assert r.returncode == 1 and state["calls"] == []                 # nothing mapped
    r, state = run_https(tmp_path, ["off"], serve=mapped("http://127.0.0.1:18499"))
    assert r.returncode == 0 and state["calls"] == [["--https=443", "off"]]


def test_another_https_port_is_its_own_mapping(tmp_path):
    serve = mapped("http://127.0.0.1:9000")                           # someone's 443
    r, state = run_https(tmp_path, ["on", "--https-port", "8443"], serve=serve)
    assert r.returncode == 0
    assert state["calls"] == [["--bg", "--https=8443", "http://127.0.0.1:18499"]]
    assert state["serve"]["Web"]["myhost.example.ts.net:443"]["Handlers"]["/"]["Proxy"] \
        == "http://127.0.0.1:9000"


def test_a_tcp_forward_on_the_port_counts_as_taken(tmp_path):
    serve = {"TCP": {"443": {"TCPForward": "127.0.0.1:9000"}}}
    r, state = run_https(tmp_path, ["on"], serve=serve)
    assert r.returncode == 1 and state["calls"] == [] and "tcp:127.0.0.1:9000" in r.stderr
    r, state = run_https(tmp_path, ["off"], serve=serve)
    assert r.returncode == 1 and state["calls"] == []
