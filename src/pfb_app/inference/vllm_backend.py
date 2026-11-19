"""vLLM backend. vllm is imported lazily so unit tests never load it."""
import os
import time
from pathlib import Path

from pfb_app.config import ModelSpec
from pfb_app.inference.backend import ChatRequest, RawResponse


class VllmBackend:
    def __init__(self, spec: ModelSpec, gpu_memory_utilization: float, max_images: int,
                 media_root: Path):
        # App-local Triton cache (the shared ~/.triton is never used); must precede vllm/triton import.
        triton_cache = Path(media_root) / "triton_cache"
        triton_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
        from vllm import LLM  # lazy: heavy import, needs GPU stack

        self.model_id = spec.hf_id
        self._llm = LLM(
            model=spec.hf_id,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images},
            allowed_local_media_path=str(media_root),
        )

    @staticmethod
    def _messages(req: ChatRequest) -> list[dict]:
        content = [
            {"type": "image_url", "image_url": {"url": Path(p).resolve().as_uri()}}
            for p in req.image_paths
        ]
        content.append({"type": "text", "text": req.user_text})
        return [{"role": "system", "content": req.system}, {"role": "user", "content": content}]

    def generate(self, reqs: list[ChatRequest]) -> list[RawResponse]:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        params = [
            SamplingParams(
                temperature=0,
                max_tokens=r.max_tokens,
                structured_outputs=StructuredOutputsParams(json=r.json_schema),
            )
            for r in reqs
        ]
        start = time.monotonic()
        outputs = self._llm.chat([self._messages(r) for r in reqs], params, use_tqdm=False)
        # llm.chat batches internally, so wall time is shared; split evenly per request.
        per_request = (time.monotonic() - start) / max(len(reqs), 1)
        return [
            RawResponse(
                text=o.outputs[0].text,
                prompt_tokens=len(o.prompt_token_ids) if o.prompt_token_ids is not None else None,
                completion_tokens=len(o.outputs[0].token_ids),
                duration_sec=per_request,
            )
            for o in outputs
        ]

    def close(self) -> None:
        pass  # VRAM is released when the owning child process exits
