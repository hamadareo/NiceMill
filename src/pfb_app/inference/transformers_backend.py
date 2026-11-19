"""Transformers backend. torch/transformers are imported lazily so other modules never load them."""
import json
import os
import re
import time
from pathlib import Path

from pfb_app.config import ModelSpec
from pfb_app.inference.backend import ChatRequest, RawResponse

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_json_fence(text: str) -> str:
    """Remove a surrounding ```json fence, if any."""
    m = _FENCE.match(text)
    return m.group(1).strip() if m else text


class TransformersBackend:
    def __init__(self, spec: ModelSpec, memory_fraction: float, max_images: int,
                 media_root: Path):
        # App-local Triton cache (the shared ~/.triton is never used); must precede torch/triton use.
        triton_cache = Path(media_root) / "triton_cache"
        triton_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = spec.hf_id
        self._max_images = max_images
        self._media_root = Path(media_root).resolve()
        self._torch = torch
        # VRAM cap: must be set before the weights are loaded.
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                spec.hf_id, torch_dtype="auto", device_map="cuda:0", attn_implementation="sdpa")
        except Exception as e:  # no fallback to another model, ever
            raise RuntimeError(f"モデル {spec.hf_id} をTransformersで読み込めません: {e}") from e
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(spec.hf_id)

    def _load_images(self, paths: list[Path]) -> list:
        from PIL import Image

        if len(paths) > self._max_images:
            raise ValueError(f"画像が多すぎます: {len(paths)} > {self._max_images}")
        images = []
        for p in paths:
            resolved = Path(p).resolve()
            if not resolved.is_relative_to(self._media_root):
                raise ValueError(f"media_root の外の画像は読めません: {p}")
            with Image.open(resolved) as im:
                images.append(im.convert("RGB"))
        return images

    def _messages(self, req: ChatRequest) -> list[dict]:
        schema = json.dumps(req.json_schema, ensure_ascii=False)
        text = (f"{req.user_text}\n\n次のJSONスキーマに従うJSONのみを出力してください。"
                f"説明文やコードブロックは付けないでください。\n{schema}")
        content = [{"type": "image", "image": im} for im in self._load_images(req.image_paths)]
        content.append({"type": "text", "text": text})
        return [{"role": "system", "content": [{"type": "text", "text": req.system}]},
                {"role": "user", "content": content}]

    def _generate_one(self, req: ChatRequest) -> RawResponse:
        torch = self._torch
        start = time.monotonic()
        inputs = self._processor.apply_chat_template(
            self._messages(req), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self._model.generate(**inputs, do_sample=False, max_new_tokens=req.max_tokens)
        new_tokens = out[0, prompt_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        resp = RawResponse(text=strip_json_fence(text), prompt_tokens=prompt_len,
                           completion_tokens=int(new_tokens.shape[0]),
                           duration_sec=time.monotonic() - start)
        del inputs, out
        torch.cuda.empty_cache()
        return resp

    def generate(self, reqs: list[ChatRequest]) -> list[RawResponse]:
        return [self._generate_one(r) for r in reqs]

    def close(self) -> None:
        pass  # VRAM is released when the owning child process exits
