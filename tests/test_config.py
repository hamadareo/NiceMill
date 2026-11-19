from pathlib import Path

from pfb_app.config import load_settings


def test_load_settings_defaults():
    s = load_settings(None)
    assert s.models["standard"].budget_gb == 24
    assert s.models["high"].hf_id == "Qwen/Qwen3-VL-32B-Instruct-FP8"
    assert s.port == 8501
    assert s.host == "127.0.0.1"
    assert s.frame_size == (854, 480)
    assert s.data_dir == Path("~/pfb_app/data").expanduser()
    assert "~" not in str(s.inbox_dir)
    assert set(s.max_tokens.values()) == {4096}


def test_load_settings_yaml_override(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("yield_threshold_gb: 10\ndata_dir: ~/x\n", encoding="utf-8")
    s = load_settings(p)
    assert s.yield_threshold_gb == 10
    assert s.port == 8501
    assert s.data_dir == Path("~/x").expanduser()


def test_repo_settings_yaml_matches_defaults():
    repo = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    assert load_settings(repo) == load_settings(None)


def test_engine_settings_defaults():
    s = load_settings(None)
    assert s.engine_dir == Path("~/pfb_app/engines/pfb3x3").expanduser()
    assert s.engine_python is None and s.tracker_python is None and s.whisper_python is None
    assert s.track_budget_gb == 4
    assert s.pfb3x3_condition == "clip_som"
    assert s.pfb3x3_port == 8200


def test_repo_settings_yaml_lists_number_names():
    import yaml
    repo = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    assert yaml.safe_load(repo.read_text(encoding="utf-8"))["number_names"] is True
