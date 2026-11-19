"""Operator alerts (ops.alerts): off by default, a webhook URL only from a file, never logged,
once per key per alert_repeat_sec. No network: the post function is a stub."""
import logging

import pytest

from pfb_app.config import Settings
from pfb_app.ops import alerts

URL = "https://discord.example.invalid/api/webhooks/123/SECRET-TOKEN"


@pytest.fixture
def settings(tmp_path):
    hook = tmp_path / "hook"
    hook.write_text(URL + "\n", encoding="utf-8")
    return Settings(data_dir=tmp_path / "data", alert_webhook_file=hook, alert_repeat_sec=600)


class Post:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, url, body):
        self.calls.append((url, body))
        if self.fail:
            raise self.fail
        return 204


def test_default_settings_only_log_and_record(tmp_path, caplog):
    s = Settings(data_dir=tmp_path / "data")
    post = Post()
    with caplog.at_level(logging.WARNING):
        assert alerts.send(s, "disk-warn", "空きが少ない", "残り40GB", post=post) is False
    assert post.calls == []
    assert "空きが少ない" in caplog.text
    assert [a["key"] for a in alerts.recent(s)] == ["disk-warn"]
    assert alerts.recent(s)[0]["sent"] is False


def test_posts_the_discord_payload(settings):
    post = Post()
    assert alerts.send(settings, "job-stopped:7", "解析が止まりました", "ジョブ 7", "error",
                       post=post)
    (url, body), = post.calls
    assert url == URL
    embed = body["embeds"][0]
    assert embed["title"] == "解析が止まりました" and embed["description"] == "ジョブ 7"
    assert embed["color"] == alerts.KINDS["error"]
    assert URL not in str(alerts.recent(settings))


def test_same_key_once_per_window(settings):
    post = Post()
    t = [1000.0]
    for _ in range(3):
        alerts.send(settings, "disk-refuse", "満杯", post=post, now=lambda: t[0])
    assert len(post.calls) == 1
    alerts.send(settings, "other", "別の件", post=post, now=lambda: t[0])
    assert len(post.calls) == 2
    t[0] += 601
    alerts.send(settings, "disk-refuse", "満杯", post=post, now=lambda: t[0])
    assert len(post.calls) == 3
    alerts.clear(settings, "disk-refuse")
    alerts.send(settings, "disk-refuse", "満杯", post=post, now=lambda: t[0])
    assert len(post.calls) == 4


def test_a_failed_post_never_raises_or_leaks_the_url(settings, caplog):
    post = Post(fail=OSError(f"cannot reach {URL}"))
    with caplog.at_level(logging.WARNING):
        assert alerts.send(settings, "k", "題", post=post) is False
    assert "SECRET-TOKEN" not in caplog.text
    assert "OSError" in caplog.text


def test_bad_or_missing_hook_file(tmp_path, caplog):
    bad = tmp_path / "bad"
    bad.write_text("http://plain.example.invalid/hook", encoding="utf-8")
    s = Settings(data_dir=tmp_path / "data", alert_webhook_file=bad)
    assert alerts.webhook_url(s) is None
    assert "plain.example" not in caplog.text
    s2 = Settings(data_dir=tmp_path / "data", alert_webhook_file=tmp_path / "missing")
    assert alerts.webhook_url(s2) is None
    post = Post()
    assert alerts.send(s2, "k", "題", post=post) is False and post.calls == []


def test_old_entries_are_dropped(settings):
    post = Post()
    alerts.send(settings, "old", "古い", post=post, now=lambda: 0.0)
    alerts.send(settings, "new", "新しい", post=post, now=lambda: alerts.KEEP_SEC + 10.0)
    assert [a["key"] for a in alerts.recent(settings)] == ["new"]


def test_cli_unit_failed(settings, monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(f"data_dir: {settings.data_dir}\n", encoding="utf-8")
    assert alerts.main(["--settings", str(cfg), "unit-failed", "pfb-worker.service"]) == 0
    assert "not sent" in capsys.readouterr().out
    rec, = alerts.recent(Settings(data_dir=settings.data_dir))
    assert rec["key"] == "unit-failed:pfb-worker.service"


def test_the_setting_never_holds_a_url():
    fields = Settings.model_fields
    assert "alert_webhook_file" in fields
    assert not [n for n in fields if "url" in n.lower()]


def test_unit_failed_text_tells_a_restart_from_giving_up():
    title, text = alerts.unit_failed_text("pfb-worker.service",
                                          {"ActiveState": "activating", "SubState": "auto-restart",
                                           "NRestarts": "2"})
    assert "自動で再起動します" in title and "2回" in text
    title, text = alerts.unit_failed_text("pfb-worker.service", {"ActiveState": "failed"})
    assert title == "サービスが止まりました" and "再起動をやめました" in text
    assert alerts.unit_failed_text("x", {})[0] == "サービスが止まり、自動で再起動します"
