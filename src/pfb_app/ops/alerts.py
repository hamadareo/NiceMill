"""Optional operator alerts (off by default), in the Discord-webhook style.

`alert_webhook_file` (settings) names a file that holds the webhook URL (keep it mode 600,
outside git); the URL is never a setting, never logged and never shown (an error names only
its type). Without the file every alert is only logged.

What is sent: a short title and a plain sentence about the server's own state (a step
stopped, the worker restarted, the disk is nearly full, a backup failed). Never a name, a
feedback text, a file name or anything from the videos: callers pass ids and counts only.

The same `key` is handled at most once per alert_repeat_sec (<data_dir>/alerts.json keeps
when each key last came up, also without a webhook: the status page lists them), so a
flapping state cannot flood the channel or the log.

CLI (systemd OnFailure= and checks):
  python -m pfb_app.ops.alerts [--settings PATH] unit-failed UNIT   (restarting or given up)
  python -m pfb_app.ops.alerts [--settings PATH] test
"""
import argparse
import fcntl
import json
import logging
import os
import socket
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings, load_settings

log = logging.getLogger(__name__)

KINDS = {"info": 0x3498DB, "ok": 0x2ECC71, "warn": 0xE67E22, "error": 0xE74C3C}
FOOTER = "NiceMill（運用のお知らせ）"
TIMEOUT_SEC = 10


def state_path(settings: Settings) -> Path:
    return settings.data_dir / "alerts.json"


def _read_state(settings: Settings) -> dict:
    try:
        data = json.loads(state_path(settings).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


_THREAD_LOCK = threading.Lock()


@contextmanager
def _locked(settings: Settings):
    """One read-modify-write of alerts.json at a time: across threads (a lock) and across the
    worker, its sessions, the UI and the backup (flock on alerts.json.lock)."""
    path = state_path(settings)
    with _THREAD_LOCK:
        fd = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            log.exception("お知らせの記録をロックできませんでした")
        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)   # releases the flock


def _write_state(settings: Settings, state: dict) -> None:
    path = state_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        log.exception("お知らせの記録を書けませんでした")


def webhook_url(settings: Settings) -> str | None:
    """The URL from alert_webhook_file, or None (not configured, unreadable or not https)."""
    path = settings.alert_webhook_file
    if path is None:
        return None
    try:
        url = Path(path).read_text(encoding="utf-8").strip()
    except OSError as e:
        log.warning("お知らせの送り先ファイルを読めません（%s）", type(e).__name__)
        return None
    if not url.startswith("https://"):
        log.warning("お知らせの送り先ファイルの中身が https:// で始まっていません")
        return None
    try:
        if Path(path).stat().st_mode & 0o077:
            log.warning("お知らせの送り先ファイル %s をほかの人も読めます。"
                        "chmod 600 にしてください", path)
    except OSError:
        pass
    return url


def payload(title: str, text: str, kind: str = "warn") -> dict:
    embed = {"title": title[:240], "color": KINDS.get(kind, KINDS["warn"]),
             "footer": {"text": f"{FOOTER} · {socket.gethostname()}"}}
    if text.strip():
        embed["description"] = text.strip()[:3900]
    return {"embeds": [embed]}


def _post(url: str, body: dict) -> int:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "pfb-app-alert"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return resp.status


KEEP_SEC = 7 * 24 * 3600   # entries older than this are dropped from alerts.json
EXIT_ALREADY_RUNNING = 75   # worker.main: another worker holds worker.lock


def _entry_at(entry) -> float | None:
    if isinstance(entry, dict) and isinstance(entry.get("at"), (int, float)):
        return float(entry["at"])
    return None


def send(settings: Settings, key: str, title: str, text: str = "", kind: str = "warn", *,
         post: Callable[[str, dict], int] | None = None,
         now: Callable[[], float] = time.time) -> bool:
    """Record and log the alert and, when a webhook is configured, post it; the same `key`
    is handled at most once per alert_repeat_sec (later calls do nothing). Returns True when
    it was posted. Never raises."""
    try:
        t = now()
        with _locked(settings):   # claim the key: a second caller meanwhile does nothing
            state = _read_state(settings)
            last = _entry_at(state.get(key))
            if last is not None and t - last < settings.alert_repeat_sec:
                return False
            state = {k: v for k, v in state.items()
                     if (_entry_at(v) or 0) > t - max(KEEP_SEC, settings.alert_repeat_sec)}
            state[key] = {"at": t, "title": title, "kind": kind, "sent": False}
            _write_state(settings, state)
        log.warning("お知らせ[%s] %s %s", key, title, text)
        url = webhook_url(settings)
        if url is None:
            return False
        try:   # outside the lock: the network may take a while
            (post or _post)(url, payload(title, text, kind))
        except Exception as e:   # noqa: BLE001 - its text may quote the URL (the secret)
            log.warning("お知らせを送れませんでした（%s）", type(e).__name__)
            return False
        with _locked(settings):
            state = _read_state(settings)
            if isinstance(state.get(key), dict):
                state[key]["sent"] = True
                _write_state(settings, state)
        return True
    except Exception:   # noqa: BLE001 - an alert must never break the caller
        log.exception("お知らせの処理に失敗しました")
        return False


def recent(settings: Settings, limit: int = 10) -> list[dict]:
    """The newest alerts recorded (sent or only logged), newest first:
    [{"key", "at", "title", "kind", "sent"}]."""
    rows = [{"key": k, **v} for k, v in _read_state(settings).items()
            if _entry_at(v) is not None]
    return sorted(rows, key=lambda r: r["at"], reverse=True)[:limit]


def clear(settings: Settings, key: str) -> None:
    """Forget `key` (the state recovered: the next problem alerts at once)."""
    with _locked(settings):
        state = _read_state(settings)
        if state.pop(key, None) is not None:
            _write_state(settings, state)


def unit_state(unit: str) -> dict[str, str]:
    """ActiveState / SubState / NRestarts of a systemd --user unit ({} when unknown)."""
    try:
        out = subprocess.run(["systemctl", "--user", "show", unit, "-p",
                              "ActiveState,SubState,NRestarts,ExecMainStatus,Result"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)


def unit_failed_text(unit: str, state: dict[str, str]) -> tuple[str, str]:
    """OnFailure= runs on every failure, also when systemd restarts the unit right after."""
    if state.get("ExecMainStatus") == str(EXIT_ALREADY_RUNNING):
        return ("ワーカーを起動しませんでした",
                f"{unit} は、同じデータで別のワーカーが動いていたため起動しませんでした"
                "（二重に動かさないための停止です）。tmux と systemd の両方で"
                "起動していないか確かめてください。")
    if state.get("ActiveState") == "failed":
        return ("サービスが止まりました",
                f"{unit} が繰り返し止まったため、自動での再起動をやめました。"
                "サーバーで状態を確かめてください。")
    restarts = state.get("NRestarts", "")
    return ("サービスが止まり、自動で再起動します",
            f"{unit} が止まったため、自動で再起動します"
            + (f"（これまでの再起動: {restarts}回）。" if restarts.isdigit() else "。")
            + "続くときはサーバーの記録を確かめてください。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pfb_app.ops.alerts")
    parser.add_argument("--settings", help="settings YAML (default: built-in defaults)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    failed = sub.add_parser("unit-failed", help="a systemd unit failed (OnFailure=)")
    failed.add_argument("unit")
    sub.add_parser("test", help="send a test alert")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = load_settings(Path(args.settings) if args.settings else None)
    if args.cmd == "unit-failed":
        title, text = unit_failed_text(args.unit, unit_state(args.unit))
        sent = send(settings, f"unit-failed:{args.unit}", title, text, "error")
    else:
        sent = send(settings, f"test:{time.time()}", "お知らせのテスト",
                    "この通知が届けば、お知らせの設定はできています。", "info")
    print("sent" if sent else "not sent (logged only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
