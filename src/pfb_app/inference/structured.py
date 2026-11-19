"""Structured (JSON-schema) calls with caching, validation and retry."""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, ValidationError

from pfb_app.inference.backend import ChatRequest, InferenceBackend
from pfb_app.storage.db import Repo


class YieldRequested(Exception):
    """The shared GPU is under pressure; the caller must end the session."""


@dataclass
class StructuredResult:
    parsed: BaseModel | None
    raw_responses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FailedCall(str):
    """The raw text ("") of a request that failed on its own (RawResponse.error)."""

    def __new__(cls, error: str):
        obj = super().__new__(cls, "")
        obj.error = error
        return obj


def _no_errors(_model: BaseModel) -> list[str]:
    return []


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request_hash(model_id: str, req: ChatRequest, attempt: int,
                 variant: str | None = None) -> str:
    """The call_log key. `variant`: what the backend adds to every request beyond `req` (e.g.
    the served backend's schema text); None leaves the key exactly as it always was."""
    payload = {
        "model_id": model_id,
        "system": req.system,
        "user_text": req.user_text,
        "images": [_file_sha256(Path(p)) for p in req.image_paths],
        "schema": json.dumps(req.json_schema, sort_keys=True, ensure_ascii=False),
        "attempt": attempt,
        "max_tokens": req.max_tokens,
    }
    if variant is not None:
        payload["variant"] = variant
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _evaluate(raw: str, schema_cls: type[BaseModel],
              validate: Callable[[BaseModel], list[str]]) -> tuple[BaseModel | None, list[str]]:
    if isinstance(raw, FailedCall):
        return None, [f"モデルの呼び出しに失敗しました: {raw.error}"]
    try:
        parsed = schema_cls.model_validate(json.loads(raw))
    except json.JSONDecodeError as e:
        return None, [f"JSONとして読めません: {e}"]
    except ValidationError as e:
        return None, [f"スキーマに合いません: {e}"]
    errors = validate(parsed)
    if errors:
        return None, list(errors)
    return parsed, []


def _with_errors(req: ChatRequest, errors: list[str]) -> ChatRequest:
    note = "\n\n前回の出力にエラーがありました。修正して、JSONのみを出力してください。\n" + "\n".join(
        f"- {e}" for e in errors)
    return ChatRequest(req.system, req.user_text + note, req.image_paths, req.json_schema,
                       req.max_tokens)


def _generate_cached(backend: InferenceBackend, repo: Repo, reqs: list[ChatRequest],
                     attempt: int, yield_check: Callable[[], bool],
                     fresh: bool = False) -> list[str]:
    """Return raw text per request: cache hits are reused, misses are generated in one batch.
    fresh: never reuse a cached answer (the new one replaces it)."""
    if yield_check():
        raise YieldRequested()
    variant = getattr(backend, "cache_variant", None)
    hashes = [request_hash(backend.model_id, r, attempt, variant) for r in reqs]
    raws: list[str | None] = [None if fresh else repo.cache_get(h) for h in hashes]
    todo = [i for i, raw in enumerate(raws) if raw is None]
    if todo:
        responses = backend.generate([reqs[i] for i in todo])
        if len(responses) != len(todo):
            raise RuntimeError(
                f"バックエンドの応答数が一致しません: {len(responses)} != {len(todo)}")
        for i, resp in zip(todo, responses):
            if resp.error is not None:   # not cached: a rerun asks the model again
                raws[i] = FailedCall(resp.error)
                continue
            repo.cache_put(hashes[i], backend.model_id, resp.text, resp.prompt_tokens,
                           resp.completion_tokens, resp.duration_sec, None)
            raws[i] = resp.text
    return [raw if raw is not None else "" for raw in raws]


def _run_attempts(backend, repo, req, schema_cls, result: StructuredResult, first_attempt: int,
                  max_retries: int, yield_check, validate, fresh: bool = False) -> StructuredResult:
    """Continue retrying from `first_attempt`; `req` already carries previous errors."""
    for attempt in range(first_attempt, 1 + max_retries):
        raw = _generate_cached(backend, repo, [req], attempt, yield_check, fresh)[0]
        result.raw_responses.append(raw)
        parsed, errors = _evaluate(raw, schema_cls, validate)
        if parsed is not None:
            result.parsed = parsed
            return result
        result.errors.extend(errors)
        req = _next_request(req, raw, errors)
    return result


def _next_request(req: ChatRequest, raw: str, errors: list[str]) -> ChatRequest:
    """The retry of `req`: with the model's errors noted, or unchanged when the call itself
    failed (a transport error says nothing about the answer)."""
    return req if isinstance(raw, FailedCall) else _with_errors(req, errors)


def call_structured(backend: InferenceBackend, repo: Repo, req: ChatRequest,
                    schema_cls: type[BaseModel], *, max_retries: int,
                    yield_check: Callable[[], bool],
                    validate: Callable[[BaseModel], list[str]] = _no_errors) -> StructuredResult:
    return _run_attempts(backend, repo, req, schema_cls, StructuredResult(None), 0, max_retries,
                         yield_check, validate)


def call_structured_batch(backend: InferenceBackend, repo: Repo, reqs: list[ChatRequest],
                          schema_cls: type[BaseModel], *, max_retries: int,
                          yield_check: Callable[[], bool],
                          validate: (Callable[[BaseModel], list[str]]
                                     | Sequence[Callable[[BaseModel], list[str]]]) = _no_errors,
                          fresh: bool = False) -> list[StructuredResult]:
    """First attempt for all requests in one batch; the failures are retried in rounds, each
    round one batch of every request still failing (a served backend answers a batch
    concurrently). Each request sees the same attempts, texts and cache keys as when retried
    on its own.

    `validate` is one function for all requests, or one function per request. fresh: ask
    the model again even where call_log has an answer (used to retry cached failures).
    """
    validators = list(validate) if isinstance(validate, (list, tuple)) else [validate] * len(reqs)
    if len(validators) != len(reqs):
        raise ValueError("validateの数がリクエスト数と一致しません")
    results = [StructuredResult(None) for _ in reqs]
    pending = list(enumerate(reqs))   # (index, the request as this attempt sends it)
    for attempt in range(0, 1 + max_retries):
        if not pending:
            break
        raws = _generate_cached(backend, repo, [r for _, r in pending], attempt, yield_check,
                                fresh)
        still = []
        for (i, req), raw in zip(pending, raws):
            results[i].raw_responses.append(raw)
            parsed, errors = _evaluate(raw, schema_cls, validators[i])
            if parsed is not None:
                results[i].parsed = parsed
                continue
            results[i].errors.extend(errors)
            still.append((i, _next_request(req, raw, errors)))
        pending = still
    return results
