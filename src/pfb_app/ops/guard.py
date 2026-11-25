"""The watch that does not depend on the worker: one plain status of the server, written to
files, so someone who cannot open the screens can still read it (`cat ops_status.txt`).

`collect(settings)` looks (read-only) at
- the worker: its beat file (the worker touches it every 30 s); a worker that is not there is
  an error after STALE_ERROR_SEC, since a restart (systemd) or a deploy takes less than that;
- the screens: whether the settings' host:port answers;
- the disk: ops.disk.disk_state (warn / refuse levels of the settings);
- the nightly backup: its last result and age (BACKUP_STALE_HOURS);
- the restore rehearsal: how long ago `scripts/backup.sh rehearse` last ran (REHEARSAL_STALE_DAYS).

`check` (what the pfb-guard timer runs every few minutes) writes <data_dir>/ops_status.json and
ops_status.txt (replaced atomically) and raises the matching alerts (ops.alerts: recorded for
the status page, posted only when a webhook is configured, the same key at most once per
alert_repeat_sec; it clears the key when the state is good again). The disk keys are the
worker's own ("disk-warn" / "disk-refuse"), so a worker that is down does not make a second
alert. `report` only prints (check_ops.sh). No network, no external notification of its own.

CLI:  python -m pfb_app.ops.guard [--settings PATH] check | report [--json]
  check   always exits 0 (a systemd oneshot must not "fail" because the server has a problem)
  report  exits 0 (all good), 1 (something to look at), 2 (something is wrong)
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.config import Settings, load_settings
from pfb_app.ops import alerts, backup, disk
from pfb_app.storage import files

log = logging.getLogger(__name__)

STATUS_JSON = "ops_status.json"
STATUS_TXT = "ops_status.txt"
BEAT_FRESH_SEC = 120          # as ops.health.ALIVE_SEC
STALE_ERROR_SEC = 600         # a worker gone this long is an error (a restart takes seconds)
BACKUP_STALE_HOURS = 36       # as ui.ops_panel
BACKUP_ERROR_HOURS = 72
REHEARSAL_STALE_DAYS = 45
STATUS_FRESH_SEC = 900        # the screens call a status file older than this "not updated"
LEVELS = {"ok": 0, "info": 0, "warn": 1, "error": 2}


@dataclass
class Item:
    key: str
    level: str      # ok | info | warn | error
    text: str

    def as_dict(self) -> dict:
        return {"key": self.key, "level": self.level, "text": self.text}


def _age_text(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}秒前"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}分前"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.0f}時間前"
    return f"{seconds / 86400:.0f}日前"


def _parse(iso: str | None) -> datetime | None:
    try:
        at = datetime.fromisoformat(iso) if iso else None
    except ValueError:
        return None
    return at.replace(tzinfo=timezone.utc) if at is not None and at.tzinfo is None else at


def worker_item(settings: Settings, now: datetime) -> Item:
    beat = files.worker_beat_at(settings)
    if beat is None:
        return Item("worker", "error", "ワーカー（解析を進める部分）が一度も動いた記録がありません。")
    age = max(0.0, (now - beat).total_seconds())
    if age < BEAT_FRESH_SEC:
        return Item("worker", "ok", f"ワーカーは動いています（最後の応答 {_age_text(age)}）。")
    if age < STALE_ERROR_SEC:
        return Item("worker", "warn", f"ワーカーの応答が {_age_text(age)} から途絶えています。"
                    "再起動中かもしれません。")
    return Item("worker", "error", f"ワーカーが止まっています（最後の応答 {_age_text(age)}）。")


def ui_item(settings: Settings, listening=None) -> Item:
    if listening is None:
        from pfb_app.ops.systemd import _listening   # noqa: PLC0415
        listening = _listening
    if listening(settings.host, settings.port):
        return Item("ui", "ok", f"画面は開けます（{settings.host}:{settings.port}）。")
    return Item("ui", "error", f"画面（{settings.host}:{settings.port}）が応答しません。")


def disk_item(settings: Settings, disk_check=disk.disk_state) -> Item:
    state = disk_check(settings)
    if state.level == "unknown":
        return Item("disk", "warn", "保存場所の空きを調べられませんでした。")
    text = f"保存場所の空きは約{state.free_gb:.0f}GBです（全体 {state.total_gb:.0f}GB）。"
    if state.level == "refuse":
        return Item("disk", "error", text + "足りないため、新しい動画や書類を受け付けていません。")
    if state.level == "warn":
        return Item("disk", "warn", text + "少なくなっています。使い終わった作業を削除してください。")
    return Item("disk", "ok", text)


def backup_item(settings: Settings, now: datetime) -> Item:
    last = backup.last_result(settings)
    if last is None:
        return Item("backup", "warn", "バックアップはまだ一度も作られていません。")
    if not last.get("ok"):
        return Item("backup", "error", "前回のバックアップに失敗しました"
                    f"（{last.get('error') or '理由不明'}）。")
    at = _parse(last.get("at"))
    if at is None:
        return Item("backup", "warn", "最後のバックアップの時刻を読めません。")
    hours = (now - at).total_seconds() / 3600
    text = f"最後のバックアップは{_age_text(hours * 3600)}です。"
    if hours > BACKUP_ERROR_HOURS:
        return Item("backup", "error", text + "夜間のバックアップが動いていません。")
    if hours > BACKUP_STALE_HOURS:
        return Item("backup", "warn", text + "夜間のバックアップが動いているか確かめてください。")
    return Item("backup", "ok", text)


def rehearsal_item(settings: Settings, now: datetime) -> Item:
    last = backup.last_rehearsal(settings)
    if last is None:
        return Item("rehearsal", "info", "バックアップからの復元の試験（scripts/backup.sh "
                    "rehearse）は、まだしていません。")
    if not last.get("ok"):
        return Item("rehearsal", "error", "復元の試験に失敗しました: "
                    + "; ".join(last.get("problems") or ["理由不明"]))
    at = _parse(last.get("at"))
    days = (now - at).total_seconds() / 86400 if at else None
    if days is None or days > REHEARSAL_STALE_DAYS:
        return Item("rehearsal", "warn", "復元の試験をしてから時間がたっています。"
                    "scripts/backup.sh rehearse を実行してください。")
    return Item("rehearsal", "ok", f"復元の試験は{_age_text(days * 86400)}にして、成功しています。")


def collect(settings: Settings, now: datetime | None = None, *, listening=None,
            disk_check=disk.disk_state) -> dict:
    now = now or datetime.now(timezone.utc)
    items = [worker_item(settings, now), ui_item(settings, listening),
             disk_item(settings, disk_check), backup_item(settings, now),
             rehearsal_item(settings, now)]
    worst = max(items, key=lambda i: LEVELS[i.level])
    level = worst.level if LEVELS[worst.level] else "ok"
    summary = {"ok": "すべて問題ありません。",
               "warn": "確かめたほうがよいことがあります。",
               "error": "対応が必要なことがあります。"}[level]
    return {"checked_at": now.isoformat(timespec="seconds"), "level": level, "summary": summary,
            "items": [i.as_dict() for i in items], "data_dir": str(settings.data_dir)}


MARK = {"ok": "[OK]  ", "info": "[--]  ", "warn": "[注意]", "error": "[要対応]"}


def render_text(status: dict) -> str:
    when = _parse(status["checked_at"])
    shown = when.astimezone().strftime("%Y-%m-%d %H:%M:%S") if when else status["checked_at"]
    lines = [f"NiceMillの状態（{shown} に確認）", f"まとめ: {status['summary']}", ""]
    lines += [f"{MARK[i['level']]} {i['text']}" for i in status["items"]]
    return "\n".join(lines) + "\n"


def status_paths(settings: Settings) -> tuple[Path, Path]:
    return settings.data_dir / STATUS_JSON, settings.data_dir / STATUS_TXT


def write_status(settings: Settings, status: dict) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for path, text in zip(status_paths(settings),
                          (json.dumps(status, ensure_ascii=False, indent=1) + "\n",
                           render_text(status)), strict=True):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def read_status(settings: Settings, now: datetime | None = None) -> tuple[dict, float] | None:
    """(the status, its age in seconds) the pfb-guard timer last wrote; None: none yet."""
    path = status_paths(settings)[0]
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        age = ((now or datetime.now(timezone.utc)).timestamp() - path.stat().st_mtime)
    except (OSError, ValueError):
        return None
    return (status, max(0.0, age)) if isinstance(status, dict) and "items" in status else None


# what a problem item raises: (alert key, title, text); the key is cleared when it is ok again
ALERT_FOR = {
    "worker": ("guard-worker", "ワーカーが止まっています",
               "解析を進めるワーカーが応答していません。サーバーの状態を確かめてください。"),
    "ui": ("guard-ui", "画面が応答しません", "画面（ウェブ）が応答していません。サーバーの状態を確かめてください。"),
}
DISK_KEYS = {"warn": ("disk-warn", "保存場所の空きが少なくなっています"),
             "error": ("disk-refuse", "保存場所の空きが足りません")}


def raise_alerts(settings: Settings, status: dict) -> None:
    for item in status["items"]:
        if item["key"] in ALERT_FOR:
            key, title, text = ALERT_FOR[item["key"]]
            if item["level"] == "error":
                alerts.send(settings, key, title, text, "error")
            elif item["level"] == "ok":
                alerts.clear(settings, key)
        elif item["key"] == "disk":
            if item["level"] in DISK_KEYS:
                key, title = DISK_KEYS[item["level"]]
                alerts.send(settings, key, title, item["text"], "error"
                            if item["level"] == "error" else "warn")
            elif item["level"] == "ok":
                alerts.clear(settings, "disk-warn")
                alerts.clear(settings, "disk-refuse")


def check(settings: Settings, now: datetime | None = None, **kw) -> dict:
    """Collect, write the status files, raise the alerts; returns the status."""
    status = collect(settings, now, **kw)
    write_status(settings, status)
    raise_alerts(settings, status)
    log.info("見張り: %s %s", status["level"], status["summary"])
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pfb_app.ops.guard")
    parser.add_argument("--settings", help="settings YAML (default: PFB_CONFIG or built-ins)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="write ops_status.json / .txt and raise alerts")
    r = sub.add_parser("report", help="print the status (read-only); exit 0 / 1 / 2")
    r.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    path = args.settings or os.environ.get("PFB_CONFIG")
    settings = load_settings(Path(path) if path else None)
    logging.basicConfig(level=logging.INFO)
    if args.cmd == "check":
        started = time.monotonic()
        check(settings)
        log.info("見張りの確認は %.1f 秒でした", time.monotonic() - started)
        return 0
    status = collect(settings)
    print(json.dumps(status, ensure_ascii=False, indent=1) if args.json else render_text(status),
          end="")
    return LEVELS[status["level"]]


if __name__ == "__main__":
    sys.exit(main())
