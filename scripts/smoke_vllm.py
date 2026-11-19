"""Manual GPU smoke test: one image, JSON-schema output, VRAM measurement. Run as its own process."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from pfb_app.config import load_settings
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.gpu import GpuGuard


class Scene(BaseModel):
    main_color: str
    shape: str


def vram_used() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
                          capture_output=True, text=True).stdout.strip().replace("\n", " | ")


def main() -> int:
    settings = load_settings(Path("config/settings.yaml"))
    spec = settings.models["standard"]
    guard = GpuGuard(settings)
    ok, reason = guard.can_start(spec.budget_gb)
    if not ok:
        print(f"smoke skipped: {reason}")
        return 0
    triton_cache = settings.data_dir / "triton_cache"
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)  # before vllm/triton import
    print("VRAM before:", vram_used())

    from PIL import Image, ImageDraw
    from pfb_app.inference.vllm_backend import VllmBackend

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=settings.data_dir, prefix="smoke_") as tmp:
        img = Path(tmp) / "test.jpg"
        im = Image.new("RGB", (448, 448), "white")
        ImageDraw.Draw(im).rectangle((100, 100, 340, 340), fill="red")
        im.save(img)
        backend = VllmBackend(spec, guard.utilization_for(spec.budget_gb), 5, settings.data_dir)
        print("VRAM during (model loaded):", vram_used())
        res = backend.generate([ChatRequest(
            system="You describe images.", user_text="Describe the main shape and its color as JSON.",
            image_paths=[img], json_schema=Scene.model_json_schema(), max_tokens=128)])
        print("RESULT:", res[0])
        print(Scene.model_validate_json(res[0].text))
        print("VRAM during (after generate):", vram_used())
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
