"""Deterministic in-memory backend for tests."""
from typing import Callable

from pfb_app.inference.backend import ChatRequest, RawResponse


class FakeBackend:
    def __init__(self, responses: Callable[[ChatRequest], str] | list[str], model_id: str = "fake"):
        self.model_id = model_id
        self._responses = responses
        self._index = 0
        self.calls: list[ChatRequest] = []

    def _next(self, req: ChatRequest) -> str:
        if callable(self._responses):
            return self._responses(req)
        text = self._responses[self._index]
        self._index += 1
        return text

    def generate(self, reqs: list[ChatRequest]) -> list[RawResponse]:
        out = []
        for req in reqs:
            self.calls.append(req)
            out.append(RawResponse(text=self._next(req), prompt_tokens=None,
                                   completion_tokens=None, duration_sec=0.0))
        return out

    def close(self) -> None:
        pass
