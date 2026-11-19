"""Inference backend protocol and request/response types."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ChatRequest:
    system: str
    user_text: str
    image_paths: list[Path]
    json_schema: dict
    max_tokens: int


@dataclass
class RawResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_sec: float
    error: str | None = None   # this request failed on its own (never cached; retried)


class InferenceBackend(Protocol):
    model_id: str

    def generate(self, reqs: list[ChatRequest]) -> list[RawResponse]: ...

    def close(self) -> None: ...
