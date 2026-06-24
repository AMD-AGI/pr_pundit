"""
Shared LLM client for all pipeline stages.

Calls the LiteLLM proxy (OpenAI-compatible endpoint).
Requires LITELLM_BASE_URL and LITELLM_MASTER_KEY in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def make_client(timeout: float = 300.0):
    import httpx
    from openai import OpenAI

    return OpenAI(
        base_url=os.environ["LITELLM_BASE_URL"],
        api_key=os.environ.get("LITELLM_MASTER_KEY", ""),
        http_client=httpx.Client(verify=False, timeout=httpx.Timeout(timeout)),
    )


def llm_call(
    prompt: str,
    model: str,
    *,
    client=None,
    max_tokens: int = 16384,
    temperature: float = 0.2,
    json_mode: bool = False,
    timeout: float | None = None,
    stream: bool | None = None,
) -> str:
    if client is None:
        client = make_client(timeout=timeout or 300.0)
    elif timeout is not None:
        # Caller wants a longer timeout but passed an existing client — rebuild from env
        client = make_client(timeout=timeout)
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Stream large requests so the HTTP connection stays alive during generation,
    # avoiding read-timeout on long responses (fused_moe.py rewrites, etc.).
    use_stream = stream if stream is not None else (max_tokens >= 8192)
    t0 = time.monotonic()

    _max_retries = 5
    for _attempt in range(_max_retries):
      if use_stream:
        chunks = []
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0
        try:
            for chunk in client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True}):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        chunks.append(delta.content)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
        except Exception as _stream_err:
            if _attempt < _max_retries - 1:
                logger.warning("Stream error on attempt %d/%d: %s — retrying", _attempt + 1, _max_retries, _stream_err)
                time.sleep(min(120, 30 * (2 ** _attempt)))
                continue
            raise
        if finish_reason == "length":
            logger.warning(
                "LLM hit max_tokens=%d (finish_reason=length) — output truncated", max_tokens
            )
        try:
            from pipeline.telemetry import CURRENT_CALL_ID, record_tokens
            record_tokens(
                call_id=CURRENT_CALL_ID.get(),
                tool="",
                model=model,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
            )
        except Exception:
            pass
        content = "".join(chunks)
        if not content.strip() and _attempt < _max_retries - 1:
            logger.warning("LLM returned empty response on attempt %d/%d — retrying", _attempt + 1, _max_retries)
            time.sleep(5 * (_attempt + 1))
            continue
        _record_trace(kwargs["messages"], content, model, prompt_tokens, completion_tokens,
                      finish_reason or "end_turn", (time.monotonic() - t0) * 1000)
        return content
      break  # non-streaming path exits the retry loop immediately

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        logger.warning(
            "LLM hit max_tokens=%d (finish_reason=length) — output truncated", max_tokens
        )
    # Record token usage against the current MCP tool call if one is active.
    prompt_tokens = 0
    completion_tokens = 0
    if getattr(resp, "usage", None):
        prompt_tokens = resp.usage.prompt_tokens or 0
        completion_tokens = resp.usage.completion_tokens or 0
        try:
            from pipeline.telemetry import CURRENT_CALL_ID, record_tokens
            record_tokens(
                call_id=CURRENT_CALL_ID.get(),
                tool="",  # filled in by telemetry context; empty here is fine
                model=model,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
            )
        except Exception:
            pass
    content = choice.message.content or ""
    _record_trace(kwargs["messages"], content, model, prompt_tokens, completion_tokens,
                  choice.finish_reason or "end_turn", (time.monotonic() - t0) * 1000)
    return content


def _record_trace(messages, content, model, input_tokens, output_tokens, stop_reason, latency_ms):
    try:
        from pipeline.tracing import record_llm_call
        record_llm_call(
            messages=messages,
            response_content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
        )
    except Exception:
        pass


def _make_dspy_lm(model: str):
    """Create a DSPy LM pointing at the LiteLLM gateway."""
    import dspy
    gateway = os.environ.get("LITELLM_GATEWAY_URL", os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"))
    key = os.environ.get("LITELLM_MASTER_KEY", "dummy")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    return dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
    )


def embed(texts: list[str], model: str = "text-embedding-3-small", client=None) -> list[list[float]]:
    """Embed a list of strings, returns list of float vectors."""
    if client is None:
        client = make_client()
    response = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]


def parse_json(text: str) -> dict | list:
    """Best-effort extract JSON from LLM output, repairing truncation."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    if not text:
        raise json.JSONDecodeError(
            "LLM returned an empty response — the model may have refused to produce output "
            "(e.g. due to a dedup finding or content policy). "
            "If this is a duplicate-check failure, relaunch with force=True to bypass the duplicate gate.",
            "", 0,
        )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # model may return JSON followed by prose — extract first complete JSON object/array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        break

    lines = text.splitlines()
    for drop in range(1, min(len(lines), 20)):
        candidate = "\n".join(lines[: len(lines) - drop])
        candidate = candidate.rstrip().rstrip(",")
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        candidate += "".join(reversed(stack))
        try:
            result = json.loads(candidate)
            logger.warning("Repaired truncated JSON (dropped last %d lines)", drop)
            return result
        except json.JSONDecodeError:
            continue

    # Last resort: Python dict/list notation (single quotes from DSPy SUBMIT calls)
    import ast
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (dict, list)):
            return result
    except Exception:
        pass

    raise json.JSONDecodeError("Could not repair truncated JSON", text, 0)
