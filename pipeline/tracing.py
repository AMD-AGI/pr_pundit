"""
Pipeline execution tracing — captures every LLM call (inputs, outputs, tool uses)
to JSONL files on NFS for later supervised fine-tuning and reinforcement learning.

Trace layout on disk:
  /app/runtime/traces/<run_id>.jsonl    — one JSON line per LLM call
  /app/runtime/traces/<run_id>.outcome.json — written at run end

Each trace line (llm_call):
  {
    "trace_id": "uuid4",
    "run_id": "...",
    "plan_id": "...",
    "timestamp": "ISO8601",
    "stage": "rewrite_file | harness_run | intent_extract | ...",
    "model": "claude-opus-4-7",
    "pr_index": 1 | null,
    "file": "vllm/attention/ops.py" | null,
    "messages": [{"role": "user", "content": "..."}],
    "tool_uses": [
      {"name": "fetch_upstream_file", "input": {...}, "output": "...", "latency_ms": 210}
    ],
    "response": {"content": "...", "stop_reason": "end_turn",
                 "usage": {"input_tokens": 4200, "output_tokens": 1800}},
    "latency_ms": 3400
  }

DSPy ReAct/RLM calls are captured via the DSPy history mechanism after each predict.

Outcome file:
  {
    "run_id": "...",
    "plan_id": "...",
    "upstream": "vllm-project/vllm",
    "timestamp": "ISO8601",
    "n_prs": 3,
    "phases": {
      "rules": {"pass": true, "iters": 2, "harness_hits": [...]},
      "arch":  {"pass": true, "iters": 1},
      "critic": {"pass": true, "iters": 1}
    },
    "final_status": "success | max_iters | stopped | error",
    "pr_urls": ["https://github.com/..."]
  }
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── context vars set by the pipeline harness at run start ─────────────────────

CURRENT_RUN_ID:  contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_run_id",  default=None)
CURRENT_PLAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_plan_id", default=None)
CURRENT_STAGE:   contextvars.ContextVar[str]         = contextvars.ContextVar("trace_stage",   default="unknown")
CURRENT_PR_IDX:  contextvars.ContextVar[int | None]  = contextvars.ContextVar("trace_pr_idx",  default=None)
CURRENT_FILE:    contextvars.ContextVar[str | None]  = contextvars.ContextVar("trace_file",    default=None)

# ── trace directory ────────────────────────────────────────────────────────────

_RUNTIME_DIR = Path(os.environ.get("RUNTIME_DIR", "/app/runtime"))
_TRACES_DIR  = _RUNTIME_DIR / "traces"


def _ensure_traces_dir() -> Path:
    try:
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return _TRACES_DIR


# ── public context manager helpers ────────────────────────────────────────────

class trace_stage:
    """Context manager: set the stage label (and optionally pr_index / file) for tracing."""
    def __init__(self, stage: str, pr_index: int | None = None, file: str | None = None):
        self._stage = stage
        self._pr_index = pr_index
        self._file = file
        self._tokens: list = []

    def __enter__(self):
        self._tokens = [
            CURRENT_STAGE.set(self._stage),
            CURRENT_PR_IDX.set(self._pr_index),
            CURRENT_FILE.set(self._file),
        ]
        return self

    def __exit__(self, *_):
        for t in self._tokens:
            t.var.reset(t)


def set_run_context(run_id: str, plan_id: str | None = None) -> None:
    """Call once at run start (from mcp_server harness) to bind run/plan IDs."""
    CURRENT_RUN_ID.set(run_id)
    CURRENT_PLAN_ID.set(plan_id)


# ── core write function ────────────────────────────────────────────────────────

def record_llm_call(
    *,
    messages: list[dict],
    response_content: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str = "end_turn",
    tool_uses: list[dict] | None = None,
    latency_ms: float = 0.0,
    extra: dict | None = None,
    stage_override: str | None = None,
) -> None:
    """Append one LLM call trace line to the run's JSONL file.

    Safe to call from any thread — the file write is atomic (single os.write).
    No-op if no run_id is set in context.
    stage_override: when provided, used instead of CURRENT_STAGE (useful for DSPy
        calls that run in threads where the ContextVar is not propagated).
    """
    run_id = CURRENT_RUN_ID.get()
    if not run_id:
        return

    record: dict[str, Any] = {
        "trace_id": str(uuid.uuid4()),
        "run_id": run_id,
        "plan_id": CURRENT_PLAN_ID.get(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage_override or CURRENT_STAGE.get(),
        "model": model,
        "pr_index": CURRENT_PR_IDX.get(),
        "file": CURRENT_FILE.get(),
        "messages": messages,
        "tool_uses": tool_uses or [],
        "response": {
            "content": response_content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
        "latency_ms": round(latency_ms, 1),
    }
    if extra:
        record.update(extra)

    _append_trace_line(run_id, record)


def _append_trace_line(run_id: str, record: dict) -> None:
    try:
        trace_path = _ensure_traces_dir() / f"{run_id}.jsonl"
        line = json.dumps(record, default=str) + "\n"
        # Open in append mode — safe for concurrent writes within one process
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.debug("Trace write failed (non-fatal): %s", exc)


# ── DSPy history capture ───────────────────────────────────────────────────────

_DSPY_HISTORY_CURSOR: int = 0  # index into lm.history of last flushed entry


def flush_dspy_history(model: str, stage: str | None = None) -> None:
    """After a DSPy ReAct/RLM call, pull its history and emit trace lines.

    DSPy stores the full message history on the LM object (dspy.settings.lm.history).
    Each history entry is one LM call (one tool-call round in a ReAct loop).
    We track a cursor so every call since the last flush is recorded.
    """
    global _DSPY_HISTORY_CURSOR
    run_id = CURRENT_RUN_ID.get()
    if not run_id:
        return
    try:
        import dspy
        lm = dspy.settings.lm
        if lm is None or not hasattr(lm, "history"):
            return
        history = lm.history
        if not history:
            return
        new_entries = history[_DSPY_HISTORY_CURSOR:]
        _DSPY_HISTORY_CURSOR = len(history)
        for entry in new_entries:
            messages = entry.get("messages", [])
            outputs = entry.get("outputs", [])
            response_content = "\n".join(str(o) for o in outputs) if outputs else ""
            usage = entry.get("usage", {}) or {}

            # Extract tool calls from the message thread
            tool_uses: list[dict] = []
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls", []) or []:
                    fn = tc.get("function", {})
                    tool_uses.append({
                        "name": fn.get("name", ""),
                        "input": _safe_json(fn.get("arguments", "")),
                        "output": None,
                    })
            result_msgs = [m for m in messages if m.get("role") == "tool"]
            for i, r in enumerate(result_msgs):
                if i < len(tool_uses):
                    tool_uses[i]["output"] = r.get("content", "")

            # DSPy usage dict keys vary by LM backend; try both naming conventions.
            _in_tok = (
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or 0
            )
            _out_tok = (
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or 0
            )
            # If DSPy calls run in a thread/async context, CURRENT_STAGE may be
            # "unknown" (ContextVar not inherited). Use the explicit stage arg when set.
            _effective_stage = stage or CURRENT_STAGE.get()
            record_llm_call(
                messages=messages,
                response_content=response_content,
                model=model,
                input_tokens=_in_tok,
                output_tokens=_out_tok,
                tool_uses=tool_uses,
                extra={"source": "dspy", "stage_override": _effective_stage},
                stage_override=_effective_stage,
            )
    except Exception as exc:
        logger.debug("DSPy history flush failed (non-fatal): %s", exc)


def _safe_json(s: str | dict) -> Any:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


# ── outcome writer ─────────────────────────────────────────────────────────────

def write_outcome(
    *,
    run_id: str,
    plan_id: str | None,
    upstream: str,
    n_prs: int,
    phases: dict,
    final_status: str,
    pr_urls: list[str] | None = None,
) -> None:
    """Write the run-level outcome file. Called once at pipeline completion."""
    record = {
        "run_id": run_id,
        "plan_id": plan_id,
        "upstream": upstream,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_prs": n_prs,
        "phases": phases,
        "final_status": final_status,
        "pr_urls": pr_urls or [],
    }
    try:
        outcome_path = _ensure_traces_dir() / f"{run_id}.outcome.json"
        outcome_path.write_text(json.dumps(record, indent=2, default=str))
    except Exception as exc:
        logger.debug("Outcome write failed (non-fatal): %s", exc)
