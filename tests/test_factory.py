import os

from pfb_app.config import Settings
from pfb_app.inference import transformers_backend, vllm_backend
from pfb_app.inference.factory import make_backend
from pfb_app.inference.transformers_backend import strip_json_fence


def test_strip_json_fence():
    assert strip_json_fence("{\"a\": 1}") == "{\"a\": 1}"
    assert strip_json_fence("```json\n{\"a\": 1}\n```") == "{\"a\": 1}"
    assert strip_json_fence("  ```\n{\"a\": 1}\n```  ") == "{\"a\": 1}"


def _recorder(store):
    class Recorder:
        def __init__(self, *args):
            store.append(args)
    return Recorder


def test_make_backend_dispatch(monkeypatch, tmp_path):
    tf, vl = [], []
    monkeypatch.setattr(transformers_backend, "TransformersBackend", _recorder(tf))
    monkeypatch.setattr(vllm_backend, "VllmBackend", _recorder(vl))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    settings = Settings(data_dir=tmp_path / "data")
    assert settings.backend == "transformers"

    make_backend(settings, "standard", 95.59)
    assert vl == []
    spec, fraction, max_images, media_root = tf[0]
    assert spec == settings.models["standard"]
    assert fraction == 0.251
    assert max_images == 20
    assert media_root == settings.data_dir
    assert os.environ["TRITON_CACHE_DIR"] == str(settings.data_dir / "triton_cache")

    settings = Settings(data_dir=tmp_path / "data", backend="vllm")
    make_backend(settings, "standard", 95.59)
    assert len(tf) == 1
    assert vl[0][1] == 0.251
