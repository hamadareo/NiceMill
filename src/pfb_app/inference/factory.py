"""Build the configured inference backend."""
import os

from pfb_app.config import Settings
from pfb_app.inference.backend import InferenceBackend
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot


def make_backend(settings: Settings, model_key: str, total_gb: float) -> InferenceBackend:
    spec = settings.models[model_key]
    fraction = GpuGuard(settings, lambda: GpuSnapshot(total_gb, 0.0)).utilization_for(spec.budget_gb)
    max_images = max(settings.frames_per_chunk, settings.identity_sample_points,
                     settings.doc_max_page_images)
    media_root = settings.data_dir
    triton_cache = media_root / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)  # before importing either backend
    if settings.backend == "vllm":
        from pfb_app.inference.vllm_backend import VllmBackend
        return VllmBackend(spec, fraction, max_images, media_root)
    from pfb_app.inference.transformers_backend import TransformersBackend
    return TransformersBackend(spec, fraction, max_images, media_root)
