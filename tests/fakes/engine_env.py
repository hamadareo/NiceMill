"""Install tests/fakes/fake_engine.py as the PFB3x3 engine of a temporary Settings.

`install_fake_engine(tmp_path, monkeypatch)` creates `<tmp>/engines/pfb3x3/scripts/{serve,stop_server}.sh`
and a `fake_python` wrapper, clears the FAKE_ENGINE_* switches and returns the pieces a test
needs. Every engine call is appended to the returned log (see `engine_calls`).
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

FAKE = Path(__file__).resolve().parent / "fake_engine.py"
SWITCHES = ("FAKE_ENGINE_EXIT", "FAKE_ENGINE_EXIT_CMD", "FAKE_ENGINE_SLEEP", "FAKE_ENGINE_OUTSIDE",
            "FAKE_ENGINE_STDOUT", "FAKE_ENGINE_STUBBORN", "FAKE_ENGINE_SERVE_SLEEP",
            "FAKE_ENGINE_STOP_EXIT", "FAKE_ENGINE_PIDS", "FAKE_ENGINE_SLEEP_CMD",
            "FAKE_ENGINE_PEOPLE", "FAKE_ENGINE_SHEET_NO_AUTO", "FAKE_ENGINE_NO_RUN_META",
            "FAKE_ENGINE_ERROR", "FAKE_ENGINE_ORPHAN", "FAKE_ENGINE_LIVE_RECORDS")


@dataclass
class FakeEngineEnv:
    engine_dir: Path
    engine_python: Path
    log: Path

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def commands(self) -> list[str]:
        return [c["argv"][0] for c in self.calls()]


def install_fake_engine(tmp_path: Path, monkeypatch) -> FakeEngineEnv:
    engine_dir = tmp_path / "engines" / "pfb3x3"
    (engine_dir / "scripts").mkdir(parents=True)
    wrapper = tmp_path / "fake_python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    for name, cmd in (("serve.sh", "serve"), ("stop_server.sh", "stop")):
        (engine_dir / "scripts" / name).write_text(
            f'#!/usr/bin/env bash\nexec "{sys.executable}" "{FAKE}" {cmd} "$@"\n', encoding="utf-8")
    log = tmp_path / "engine_calls.jsonl"
    monkeypatch.setenv("FAKE_ENGINE_LOG", str(log))
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)
    return FakeEngineEnv(engine_dir, wrapper, log)


def alive(pid: int) -> bool:
    """True while `pid` is a live, non-zombie process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split(")")[-1].split()[0] == "Z")


# What fake_translator answers for the fake engine's (label-free) observations.
TRANSLATIONS = {"本人 folds a paper strip": "紙を折る",
                "本人 hands a paper strip to 別の参加者": "別の参加者に紙を渡す",
                "本人 wipes the desk": "机を拭く"}


def fake_translator(table: dict[str, str] | None = None, fail: bool = False):
    """A FakeBackend standing in for the session's served model when the engine's English
    is translated: known sentences from `table`, others as 「（訳）<text>」."""
    import re
    from pfb_app.inference.fake import FakeBackend

    table = TRANSLATIONS if table is None else table

    def respond(req) -> str:
        if fail:
            return "not json"
        title = req.json_schema.get("title")
        ids = re.findall(r"^ref_id: (\S+)$", req.user_text, re.M)
        if title == "ContributionOutput":   # the session's server also judges contributions
            return json.dumps({"judgments": [
                {"ref_id": i, "contribution_interpretation": "解釈",
                 "is_positive_contribution": True, "rubric_criteria": []} for i in ids]},
                ensure_ascii=False)
        if title == "PfbOutput":            # ... and writes the feedback
            return json.dumps({"pfb_status": "generated", "pfb": "よく動いていました。",
                               "evidence_refs": ids[:1]}, ensure_ascii=False)
        pairs = re.findall(r"^ref_id: (\S+)\n記録: (.*)$", req.user_text, re.M)
        return json.dumps({"translations": [{"ref_id": r, "text": table.get(t, f"（訳）{t}")}
                                            for r, t in pairs]}, ensure_ascii=False)

    return FakeBackend(respond, model_id="fake-translator")
