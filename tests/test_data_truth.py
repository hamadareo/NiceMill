"""The 「データの扱い」 page says the data stays on this machine. These checks keep that true:
only three modules of the app can open a network connection (the local model server's
client, which refuses other hosts; the optional operator notice; the check whether the app's
own screen port answers), the models load offline, and the screens send no usage statistics."""
import ast
import re
from pathlib import Path

import pytest

import pfb_app
from pfb_app.inference import served

SRC = Path(pfb_app.__file__).parent
ROOT = SRC.parents[1]
NETWORK = {"urllib.request", "http.client", "requests", "httpx", "aiohttp", "smtplib",
           "ftplib", "websocket", "websockets", "paramiko"}
ALLOWED = {"inference/served.py", "ops/alerts.py",
           # whether this app's own screen port answers before a restore (connects, sends
           # nothing; ops.systemd._listening)
           "ops/systemd.py"}


def _imports(tree) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def test_only_the_known_modules_can_open_a_connection():
    senders = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        if _imports(tree) & NETWORK or re.search(r"socket\.(create_connection|socket)\(", text):
            senders.add(str(path.relative_to(SRC)))
    assert senders == ALLOWED


def test_the_model_client_talks_only_to_this_machine():
    assert served.LOCAL_HOSTS == {"localhost", "127.0.0.1", "::1"}
    with pytest.raises(ValueError):
        served.ServedChatBackend("http://example.com:8000/v1", "m")


def test_the_models_load_offline_and_the_screens_send_no_statistics():
    assert "HF_HUB_OFFLINE=1" in (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    from pfb_app.engine import pfb3x3
    assert pfb3x3._OFFLINE_ENV.get("HF_HUB_OFFLINE") == "1"
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r"^gatherUsageStats\s*=\s*false", config, re.M)


def test_the_operator_notice_never_carries_names_or_texts():
    doc = (SRC / "ops" / "alerts.py").read_text(encoding="utf-8")
    assert "Never a name, a\nfeedback text, a file name or anything from the videos" in doc


def test_the_page_says_so(tmp_path):
    from pfb_app.config import Settings
    from pfb_app.ui.wizard import data_view
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")
    lines = "\n".join(data_view.where_lines(settings))
    assert "外のサービスに送ることはありません" in lines
    assert "設定されていません" in lines
    assert f"{settings.backup_keep}回分" in lines
    on = settings.model_copy(update={"alert_webhook_file": tmp_path / "hook"})
    assert "設定されています" in "\n".join(data_view.where_lines(on))


def test_every_engine_subprocess_gets_the_offline_switches(tmp_path, monkeypatch):
    """Review MED 6: the tracker (ultralytics 8.4 sends usage events unless YOLO_OFFLINE),
    vLLM and the hub libraries run offline in every subprocess the engine starts."""
    from pfb_app.config import Settings
    from pfb_app.engine import pfb3x3
    from pfb_app.ops import offline
    for key, value in {"YOLO_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1"}.items():
        assert offline.OFFLINE_ENV[key] == value
        monkeypatch.setenv(key, "0")          # whatever the environment said before
    engine = pfb3x3.Pfb3x3Engine(Settings(data_dir=tmp_path, engine_dir=tmp_path / "e"))
    for env in (engine._env(None), engine._env(tmp_path, {"X": "1"})):
        assert {k: env[k] for k in offline.OFFLINE_ENV} == offline.OFFLINE_ENV
    # every way the engine starts a process goes through _env
    for name in ("engine/pfb3x3.py", "engine/live.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        for call in re.findall(r"subprocess\.Popen\([^)]*", text):
            assert "env=env" in call or "env=engine._env(" in call, (name, call)
        for call in re.findall(r"self\._run\(\s*\w+,\s*([^,]+),", text):
            assert call.strip() == "env" or call.strip().startswith("self._env("), (name, call)
        for assign in re.findall(r"^\s+env = (.+)$", text, re.M):
            # _env's own first line, or a call of it
            assert assign.startswith(("self._env(", "{k: v for k, v in os.environ")), \
                (name, assign)


def test_the_worker_and_its_sessions_run_offline(monkeypatch):
    from pfb_app.ops import offline
    env = {"YOLO_OFFLINE": "0"}
    offline.apply(env)
    assert env == {**offline.OFFLINE_ENV}
    for name in ("worker/main.py", "worker/session.py"):
        assert "offline.apply()" in (SRC / name).read_text(encoding="utf-8"), name
