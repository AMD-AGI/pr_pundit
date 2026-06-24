"""
Stage B2 — Recipe normalizer.

Reads data/bronze_recipes/{owner}_{repo}/files.jsonl and produces
data/silver_recipes/{owner}_{repo}/records.jsonl

Each silver record adds structured metadata extracted from the raw file:
  - test_type: throughput | accuracy | latency | kernel | e2e | unknown
  - tools: list of tools detected (vllm, sglang, lm_eval, pytest, ...)
  - models: model names mentioned
  - flags: key CLI flags extracted (tp, pp, quantization, dtype, ...)
  - summary: one-line human description (from comments/docstring)

Usage:
    python -m pipeline.normalize_recipes --repo vllm-project/vllm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BRONZE = Path(__file__).resolve().parent.parent / "data" / "bronze_recipes"
SILVER = Path(__file__).resolve().parent.parent / "data" / "silver_recipes"

# ── performance signal filter ────────────────────────────────────────
# Files from companion recipe repos (source_type="recipes") are often
# operational guides. Only keep them if they contain concrete performance
# or benchmarking signal.
_PERF_SIGNAL_KEYWORDS = {
    "throughput", "tok/s", "tokens/s", "tokens per second", "req/s",
    "requests per second", "benchmark", "latency", "ttft", "tpot", "itl",
    "p99", "p95", "accuracy", "gsm8k", "mmlu", "humaneval", "lm_eval",
    "lm-eval", "microbenchmark", "profil", "flops", "tflops", "bandwidth",
    "speedup", "performance", "optimal", "tuning", "sweep",
    "--max-num-seqs", "--num-prompts", "--request-rate",
    "benchmark_throughput", "benchmark_latency", "benchmark_serving",
}

def _has_perf_signal(content: str) -> bool:
    lower = content.lower()
    return any(kw in lower for kw in _PERF_SIGNAL_KEYWORDS)


# ── classification patterns ──────────────────────────────────────────

_TEST_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    # accuracy must come before throughput — lm_eval scripts often mention throughput secondarily
    # "accuracy" word alone is too noisy (model docs say "accuracy on benchmarks") — require specific eval framework/dataset names
    ("accuracy",   ["gsm8k", "mmlu", "lm_eval", "lm-eval", "benchmark_accuracy",
                    "humaneval", "mbpp", "truthfulqa", "hellaswag", "winogrande", "arc_challenge",
                    "lm_evaluation_harness", "atom accuracy", "accuracy_test_threshold",
                    "evaluate_harness", "eleutherai"]),
    ("latency",    ["latency", "ttft", "time_to_first_token", "tpot", "itl", "p99", "p95"]),
    # kernel before throughput — aiter perftest files report ops/s but are kernel microbenchmarks
    ("kernel",     ["benchmark_kernel", "microbenchmark", "ops_per_second",
                    "hip_graph", "cuda_graph",
                    "benchmarks/kernels/", "fused_kernels/", "op_tests/",
                    "triton.ops", "triton.language",
                    "benchmark_rope", "benchmark_gemm", "benchmark_moe",
                    "benchmark_activation", "benchmark_quant",
                    # aiter-specific patterns
                    "aiter.test_common", "from aiter.test_common", "perftest(",
                    "run_perftest(", "checkallclose",
                    "ops/triton/", "ops/ck/", "csrc/kernels/",
                    # GEMM kernel benchmarks
                    "gemm_benchmark", "benchmark_gemm", "ck_gemm", "deepgemm",
                    "a8w8", "w8a8", "fp8_gemm", "int8_gemm", "bf16_gemm",
                    "gemm tuner", "gemm kernel", "grouped gemm",
                    "gemm_tuner", "batched_gemm", "ck_batched_gemm",
                    # other kernel-level indicators
                    "flops_per_element", "tflops", "gbps", "gb/s", "bandwidth_gb"]),
    ("throughput", ["throughput", "tokens_per_second", "tok/s", "requests_per_second",
                    "benchmark_serving", "benchmark_throughput", "tput",
                    "output_tokens_per_second", "input_tokens_per_second"]),
    ("e2e",        ["e2e", "end_to_end", "end-to-end", "integration", "serving"]),
]

_TOOL_PATTERNS: list[tuple[str, list[str]]] = [
    ("vllm",     ["from vllm", "import vllm", "vllm serve", "vllm.llm", "llm(", "asyncllmengine"]),
    ("sglang",   ["import sglang", "sgl.engine", "sglang.launch", "python -m sglang"]),
    ("lm_eval",  ["lm_eval", "lm-eval", "lm_evaluation_harness"]),
    ("pytest",   ["import pytest", "def test_", "@pytest"]),
    ("docker",   ["docker run", "docker exec", "subprocess.*docker"]),
]

_FLAG_RE = re.compile(
    r"""
    --(?P<flag>
        tensor[_-]parallel(?:[_-]size)?|tp(?:[_-]size)?|
        pipeline[_-]parallel(?:[_-]size)?|pp(?:[_-]size)?|
        data[_-]parallel(?:[_-]size)?|dp(?:[_-]size)?|
        quantization|quant|dtype|data[_-]type|
        max[_-]model[_-]len|max[_-]num[_-]seqs|
        gpu[_-]memory[_-]utilization|kv[_-]cache[_-]dtype|
        enable[_-]prefix[_-]caching|enable[_-]chunked[_-]prefill|
        num[_-]gpu[_-]blocks|block[_-]size|
        input[_-]len|output[_-]len|max[_-]tokens|
        concurrency|num[_-]prompts|batch[_-]size|
        trust[_-]remote[_-]code|revision|tokenizer
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MODEL_RE = re.compile(
    r"""
    (?:
        # Path-like model names
        [A-Za-z0-9_\-]+/[A-Za-z0-9_\.\-]+(?:-[A-Za-z0-9_]+)*  |
        # Known family names
        (?:llama|mistral|qwen|deepseek|gemma|phi|falcon|mixtral|kimi|yi|solar|
           claude|gpt|starcoder|codellama|vicuna|wizardlm|internlm|baichuan)
        [A-Za-z0-9_\.\-]*
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Noise model strings to filter
_MODEL_NOISE = {"the", "a", "an", "for", "with", "and", "or", "of", "to", "in", "is"}


def _classify_test_type(text: str) -> str:
    lower = text.lower()
    for test_type, patterns in _TEST_TYPE_PATTERNS:
        if any(p in lower for p in patterns):
            return test_type
    return "unknown"


def _detect_tools(text: str) -> list[str]:
    lower = text.lower()
    return [tool for tool, patterns in _TOOL_PATTERNS if any(p in lower for p in patterns)]


def _extract_flags(text: str) -> list[str]:
    flags = {m.group("flag").lower().replace("-", "_") for m in _FLAG_RE.finditer(text)}
    return sorted(flags)


def _extract_models(text: str) -> list[str]:
    candidates = {m.group(0) for m in _MODEL_RE.finditer(text)}
    # Filter noise and very short matches
    return sorted(
        m for m in candidates
        if len(m) > 3 and m.lower() not in _MODEL_NOISE and "/" in m or
        any(kw in m.lower() for kw in ("llama", "mistral", "qwen", "deepseek", "gemma",
                                        "kimi", "phi", "gpt", "starcoder", "falcon"))
    )[:20]  # cap at 20


def _extract_summary(content: str, path: str) -> str:
    """Extract a one-line summary from shebang comments or docstring."""
    lines = content.splitlines()
    # Shell scripts: look for first meaningful comment line
    if path.endswith(".sh"):
        for line in lines[:30]:
            line = line.strip().lstrip("#").strip()
            if len(line) > 15 and not line.startswith("!"):
                return line[:200]
    # Python: look for module docstring
    if path.endswith(".py"):
        in_docstring = False
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                text = stripped.strip('"\'').strip()
                if text:
                    return text[:200]
                in_docstring = True
                continue
            if in_docstring and stripped:
                return stripped[:200]
    # Markdown: first non-heading line
    if path.endswith(".md"):
        for line in lines[:10]:
            stripped = line.strip().lstrip("#").strip()
            if stripped and not stripped.startswith("!["):
                return stripped[:200]
    return Path(path).stem.replace("_", " ").replace("-", " ")


def normalize_recipes(repo: str):
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"

    src = BRONZE / repo_slug / "files.jsonl"
    if not src.exists():
        logger.error("Bronze not found: %s — run scrape-recipes first", src)
        return

    dst_dir = SILVER / repo_slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "records.jsonl"

    # Load already-normalized paths for resume
    done: set[str] = set()
    if dst.exists():
        for line in dst.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add(f"{rec['source_repo']}:{rec['path']}")

    if done:
        logger.info("Resuming — %d records already normalized", len(done))

    count = 0
    with open(dst, "a") as out_fh:
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            key = f"{raw['source_repo']}:{raw['path']}"
            if key in done:
                continue

            content = raw.get("content", "")
            path = raw["path"]

            # Drop operational-only files from recipe companion repos
            # CI configs always pass — they contain real benchmark commands
            if raw.get("source_type") == "recipes" and not _has_perf_signal(content):
                logger.debug("Skipping operational recipe (no perf signal): %s", path)
                continue

            record = {
                **raw,
                "repo": raw["repo"] if "/" in raw.get("repo", "") else raw["repo"].replace("_", "/", 1),
                "test_type": _classify_test_type(path + "\n" + content),
                "tools": _detect_tools(content),
                "models": _extract_models(content),
                "flags": _extract_flags(content),
                "summary": _extract_summary(content, path),
            }
            # Drop raw content from silver to save space — distill reads bronze directly
            # Actually keep a truncated version for distill to use without re-reading bronze
            record["content"] = content[:8000] if len(content) > 8000 else content

            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            done.add(key)
            count += 1

    logger.info("Normalized %d records → %s", count, dst)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Normalize recipe bronze to silver")
    p.add_argument("--repo", required=True, help="owner/name")
    args = p.parse_args()
    normalize_recipes(args.repo)


if __name__ == "__main__":
    main()
