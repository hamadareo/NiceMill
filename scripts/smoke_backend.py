"""Manual GPU smoke test of the configured backend. Run as its own process (see usage below).

  cd ~/pfb_app; nvidia-smi --query-gpu=memory.used --format=csv; \
  uv run python scripts/smoke_backend.py --model standard; \
  nvidia-smi --query-gpu=memory.used --format=csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from pfb_app.config import load_settings
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.gpu import GpuGuard, query_nvidia_smi


class Scene(BaseModel):
    description: str
    objects: list[str]


def vram_used() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def own_vram() -> str:
    import os
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    mine = [l.strip() for l in out.splitlines() if l.split(",")[0].strip() == str(os.getpid())]
    return mine[0] if mine else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="standard")
    args = ap.parse_args()
    settings = load_settings(Path("config/settings.yaml"))
    spec = settings.models[args.model]
    ok, reason = GpuGuard(settings).can_start(spec.budget_gb)
    if not ok:
        print(f"smoke skipped: {reason}")
        return 0
    print("VRAM before:", vram_used())

    from PIL import Image, ImageDraw
    from pfb_app.inference.factory import make_backend
    from pfb_app.inference.structured import call_structured
    from pfb_app.storage.db import Repo, connect, init_schema

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    img = settings.data_dir / "smoke_test.jpg"
    im = Image.new("RGB", (854, 480), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((100, 100, 340, 380), fill="red")
    d.ellipse((500, 100, 760, 360), fill="blue")
    im.save(img)

    backend = make_backend(settings, args.model, query_nvidia_smi().total_gb)
    print("VRAM after model load:", vram_used())
    conn = connect(settings.data_dir / "smoke.db")
    init_schema(conn)
    req = ChatRequest(system="You describe images.", user_text="Describe the image.",
                      image_paths=[img], json_schema=Scene.model_json_schema(), max_tokens=256)
    res = call_structured(backend, Repo(conn, settings), req, Scene,
                          max_retries=settings.max_retries, yield_check=lambda: False)
    print("PARSED:", res.parsed, "| errors:", res.errors, "| raw:", res.raw_responses)
    print("VRAM after generation:", vram_used())
    import torch
    print("OWN process VRAM:", own_vram(),
          f"| torch peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
          f"| reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB (cap {spec.budget_gb} GB)")
    backend.close()
    return 0 if res.parsed is not None else 1


if __name__ == "__main__":
    sys.exit(main())
