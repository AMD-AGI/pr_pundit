"""
Stage C2 — Recipe distiller.

Reads data/silver_recipes/{owner}_{repo}/records.jsonl and produces
data/gold/{owner}_{repo}/test_knowledge.json

The gold output is a structured knowledge base of test patterns:
  {
    "repo": "vllm-project/vllm",
    "generated_at": "...",
    "categories": {
      "throughput": [ <TestRecipe>, ... ],
      "accuracy":   [ <TestRecipe>, ... ],
      "latency":    [ <TestRecipe>, ... ],
      "kernel":     [ <TestRecipe>, ... ],
      "e2e":        [ <TestRecipe>, ... ],
    },
    "test_expectations": [ <TestExpectation>, ... ]  # mined from PR reviews
  }

A TestRecipe:
  {
    "name": "vLLM offline throughput benchmark",
    "use_case": "Measure tokens/sec for a given model with tensor parallelism",
    "tools": ["vllm"],
    "flavor": "python_api" | "lm_eval" | "cli",
    "models": ["meta-llama/Llama-3-70B"],
    "key_flags": {"tp": 4, "quantization": "fp8", "max_num_seqs": 256},
    "code_template": "...",    # canonical Python/shell template
    "source_files": ["benchmarks/benchmark_throughput.py"],
    "metrics_reported": ["throughput_tok_s", "latency_ms"],
  }

A TestExpectation (mined from reviews):
  {
    "pr_category": "new model",     # what type of PR triggers this
    "expected_tests": ["throughput benchmark", "accuracy on GSM8K"],
    "typical_hardware": ["A100", "MI300X"],
    "reviewer_quotes": ["Please add a throughput benchmark with fp8"]
  }

Usage:
    python -m pipeline.distill_recipes --repo vllm-project/vllm
    python -m pipeline.distill_recipes --repo vllm-project/vllm --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import threading
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.llm import llm_call, make_client, parse_json
from pipeline.notify import (
    notify_distill_recipes_start,
    notify_distill_recipes_type_done,
    notify_distill_recipes_done,
    notify_distill_recipes_error,
)

logger = logging.getLogger(__name__)

SILVER = Path(__file__).resolve().parent.parent / "data" / "silver_recipes"
BRONZE = Path(__file__).resolve().parent.parent / "data" / "bronze"
GOLD   = Path(__file__).resolve().parent.parent / "data" / "gold"

DEFAULT_MODEL = "claude-opus-4-7"

# How many silver records to send per LLM batch
BATCH_SIZE = 10
CONSOLIDATE_CHUNK = 20  # max recipes per consolidation call

# ── prompts ──────────────────────────────────────────────────────────

_RECIPE_DISTILL_PROMPT = """You are distilling benchmark and test scripts from an open-source ML repository into a structured knowledge base of test patterns.

Below are {n} raw files from the repository "{repo}". Each file includes its path, detected test type, tools used, and a content excerpt.

Your task: extract DISTINCT, REUSABLE test recipe templates that measure or validate performance.

INCLUDE:
- Throughput benchmarks (tokens/sec, requests/sec) with specific flag combinations
- Accuracy benchmarks (GSM8K, MMLU, HumanEval, lm_eval harness)
- Latency benchmarks (TTFT, TPOT, ITL, p99)
- Kernel microbenchmarks (GEMM, attention, quantization ops)
- End-to-end serving benchmarks with measurable metrics
- Scripts showing optimal flag combinations for specific hardware or workloads

EXCLUDE — do NOT extract recipes for:
- Operational server setup (vllm serve without benchmarking)
- Installation or deployment guides
- API usage examples that don't measure performance
- Documentation or README files
- Configuration-only scripts with no measurable output

If a file contains no performance measurement, skip it entirely and extract nothing from it.
Consolidate similar scripts into one recipe with notes about variants.

Return a JSON array of recipes with this schema:
[
  {{
    "name": "short descriptive name",
    "use_case": "one sentence: what this test measures and when to use it",
    "tools": ["vllm" | "sglang" | "lm_eval" | "triton" | ...],
    "flavor": "python_api" | "lm_eval" | "cli" | "shell_script",
    "models": ["example model names this is typically run with"],
    "key_flags": {{"flag_name": "typical_value_or_description"}},
    "code_template": "a clean, parameterized code template (Python or shell) that a developer can adapt. Use {{MODEL}}, {{TP}}, {{OUTPUT_DIR}} etc as placeholders. Keep it runnable and complete.",
    "source_files": ["path/to/source.py"],
    "metrics_reported": ["list of metrics this test outputs"],
    "trigger_pr_types": ["types of PR changes that should trigger this test: e.g. 'kernel_optimization', 'new_model', 'quantization', 'attention_mechanism', 'amd_hardware', 'serving_feature'"],
    "hardware_requirements": ["specific GPU/hardware needed: e.g. 'A100', 'MI300X', 'any'"],
    "notes": "any important caveats, hardware requirements, or flag interactions"
  }}
]

FILES:
{files_block}

Return ONLY the JSON array. No prose.
"""

_CONSOLIDATE_PROMPT = """You are consolidating benchmark/test recipes extracted in separate batches from the same repository. Some recipes may be duplicates or near-duplicates extracted from similar files.

REPOSITORY: {repo}
CATEGORY: {test_type}
RECIPES TO CONSOLIDATE ({n} total):
{recipes_json}

Merge duplicates into a clean, non-redundant set:
- Merge recipes that test the same thing into one recipe; combine their source_files and note variants in "notes"
- Keep recipes that genuinely test different things (e.g. offline vs online throughput are different)
- The final set should have no two recipes that measure the same metric with the same tool
- For code_template: pick the most complete and parameterized template from the input recipes; do not generate new code

Return a JSON array using the full recipe schema including code_template. Return ONLY the JSON array.
"""

_EXPECTATION_DISTILL_PROMPT = """You are analyzing PR review comments and PR descriptions from the "{repo}" repository to extract patterns about what supporting tests reviewers typically require.

Below are {n} PR review excerpts that mention tests, benchmarks, or performance evaluation.

Extract a list of "test expectations" — patterns of what tests are expected for what type of PR.

Return a JSON array:
[
  {{
    "pr_category": "what type of PR triggers this expectation (e.g. 'new model support', 'kernel optimization', 'quantization', 'attention mechanism')",
    "expected_tests": ["list of test types expected"],
    "typical_metrics": ["tokens/sec", "GSM8K accuracy", ...],
    "typical_hardware": ["A100", "MI300X", "H100", ...],
    "reviewer_quotes": ["short direct quotes from reviewers requesting these tests"]
  }}
]

PR EXCERPTS:
{excerpts_block}

Return ONLY the JSON array. No prose.
"""


def _format_files_block(records: list[dict]) -> str:
    parts = []
    for r in records:
        parts.append(
            f"--- {r['path']} (type={r['test_type']}, tools={r['tools']}) ---\n"
            f"{r['content'][:3000]}"
        )
    return "\n\n".join(parts)


def _mine_review_expectations(repo: str, model: str, client) -> list[dict]:
    """Mine existing bronze PR reviews for test expectation patterns."""
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    bronze_dir = BRONZE / repo_slug

    # Collect review comments that mention tests/benchmarks
    excerpts: list[str] = []
    for fname in ("reviews.jsonl", "review_threads.jsonl", "issue_comments.jsonl"):
        fpath = bronze_dir / fname
        if not fpath.exists():
            continue
        for line in fpath.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            body = r.get("body", "") or ""
            if not body:
                # review_threads have nested comments
                for c in r.get("comments", {}).get("nodes", []):
                    body += c.get("body", "") + "\n"
            if any(kw in body.lower() for kw in
                   ("benchmark", "throughput", "accuracy", "test", "perf", "latency",
                    "gsm8k", "mmlu", "tok/s", "tokens per second")):
                excerpts.append(body[:500])
                if len(excerpts) >= 200:
                    break
        if len(excerpts) >= 200:
            break

    if not excerpts:
        logger.info("No relevant review excerpts found for expectation mining")
        return []

    logger.info("Mining test expectations from %d review excerpts", len(excerpts))
    excerpts_block = "\n\n---\n\n".join(excerpts[:100])
    prompt = _EXPECTATION_DISTILL_PROMPT.format(
        repo=repo, n=len(excerpts[:100]), excerpts_block=excerpts_block
    )
    try:
        raw = llm_call(prompt, model, client=client, max_tokens=16384, json_mode=True)
        result = parse_json(raw)
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("Expectation mining failed: %s", exc)
        return []


def _check_connectivity(client) -> bool:
    """Verify LiteLLM proxy is reachable before starting expensive batches."""
    try:
        client.models.list()
        return True
    except Exception as exc:
        logger.error("Cannot reach LiteLLM proxy: %s", exc)
        return False


def _distill_batch(
    batch: list[dict], batch_num: int, total_batches: int,
    test_type: str, repo: str, model: str,
) -> list[dict]:
    client = make_client()
    files_block = _format_files_block(batch)
    prompt = _RECIPE_DISTILL_PROMPT.format(n=len(batch), repo=repo, files_block=files_block)
    logger.info("  [%s] calling LLM batch %d/%d (%d files)...", test_type, batch_num, total_batches, len(batch))
    try:
        raw = llm_call(prompt, model, client=client, max_tokens=16384, json_mode=False)
        recipes = parse_json(raw)
        if isinstance(recipes, list):
            logger.info("  [%s] batch %d/%d → %d recipes", test_type, batch_num, total_batches, len(recipes))
            return recipes
        # Model wrapped array in an object — unwrap common keys
        if isinstance(recipes, dict):
            for key in ("recipes", "items", "results", "data"):
                if isinstance(recipes.get(key), list):
                    return recipes[key]
        logger.warning("  [%s] batch %d/%d: LLM returned non-list — got: %.200s", test_type, batch_num, total_batches, raw)
    except Exception as exc:
        logger.warning("  [%s] batch %d/%d failed: %s", test_type, batch_num, total_batches, exc)
    return []


_EMBED_INSTRUCTION = "Represent the benchmark test recipe for deduplication and similarity comparison:"


def _cluster_by_embedding(
    recipes: list[dict], embed_model: str, sim_threshold: float = 0.82
) -> list[list[dict]]:
    """Cluster recipes using complete-linkage agglomerative clustering on embeddings.

    Complete linkage: two clusters merge only if ALL pairs across them exceed
    sim_threshold — avoids chaining (A~B, B~C pulling in unrelated A and C).
    Falls back to one-cluster-per-recipe if scipy/embedding unavailable.
    """
    from pipeline.llm import embed
    texts = [
        f"{_EMBED_INSTRUCTION} {r.get('name', '')}. {r.get('use_case', '')}"
        for r in recipes
    ]
    try:
        vectors = embed(texts, model=embed_model)
    except Exception as exc:
        logger.warning("Embedding failed (%s) — skipping clustering", exc)
        return [[r] for r in recipes]

    if len(vectors) == 1:
        return [recipes]

    try:
        import numpy as np
        from collections import defaultdict
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        arr = np.array(vectors, dtype=float)
        dist_condensed = pdist(arr, metric="cosine")
        Z = linkage(dist_condensed, method="complete")
        labels = fcluster(Z, t=1.0 - sim_threshold, criterion="distance")

        groups: dict[int, list[dict]] = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[int(label)].append(recipes[idx])
        return list(groups.values())

    except ImportError:
        logger.warning("scipy not available — skipping clustering")
        return [[r] for r in recipes]


def _merge_cluster(
    cluster: list[dict], test_type: str, repo: str, model: str,
) -> list[dict]:
    """LLM-merge a cluster of similar recipes into one canonical recipe.
    Passes full code_templates so the LLM can pick the best one directly.
    """
    if len(cluster) == 1:
        return cluster
    client = make_client()
    prompt = _CONSOLIDATE_PROMPT.format(
        repo=repo, test_type=test_type, n=len(cluster),
        recipes_json=json.dumps(cluster, indent=2),
    )
    try:
        raw = llm_call(prompt, model, client=client, max_tokens=16384, json_mode=False)
        result = parse_json(raw)
        if isinstance(result, dict):
            for key in ("recipes", "items", "results", "data"):
                if isinstance(result.get(key), list):
                    result = result[key]
                    break
        if isinstance(result, list):
            return result
    except Exception as exc:
        logger.warning("  [%s] cluster merge failed (%s) — keeping largest recipe", test_type, exc)
    return [max(cluster, key=lambda r: len(r.get("code_template", "")))]


def _consolidate_recipes(
    recipes: list[dict], test_type: str, repo: str, model: str,
    workers: int = 4, embed_model: str = "text-embedding-3-small",
) -> list[dict]:
    """Embedding-based clustering + LLM merge per cluster.

    1. Embed name+use_case for all recipes (one API call)
    2. Greedy cosine-similarity clustering (no LLM needed)
    3. LLM merge call only for clusters with >1 recipe (parallel)
    """
    if len(recipes) <= 1:
        return recipes

    logger.info("  [%s] embedding %d recipes for clustering...", test_type, len(recipes))
    clusters = _cluster_by_embedding(recipes, embed_model)
    multi = [c for c in clusters if len(c) > 1]
    singles = [c[0] for c in clusters if len(c) == 1]
    logger.info("  [%s] %d clusters: %d to merge, %d already unique",
                test_type, len(clusters), len(multi), len(singles))

    merged: list[dict] = list(singles)
    if multi:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_merge_cluster, cluster, test_type, repo, model)
                for cluster in multi
            ]
            for fut in concurrent.futures.as_completed(futures):
                merged.extend(fut.result())

    logger.info("  [%s] consolidated %d → %d recipes", test_type, len(recipes), len(merged))
    return merged


def distill_recipes(repo: str, model: str = DEFAULT_MODEL, workers: int = 4, embed_model: str = "instructor-xl"):
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"

    silver_path = SILVER / repo_slug / "records.jsonl"
    if not silver_path.exists():
        logger.error("Silver not found: %s — run normalize-recipes first", silver_path)
        return

    gold_dir = GOLD / repo_slug
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "test_knowledge.json"

    # Load silver records grouped by test_type
    records: list[dict] = []
    for line in silver_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    logger.info("Loaded %d silver recipe records", len(records))

    # Verify connectivity before starting
    logger.info("Checking LiteLLM connectivity...")
    client = make_client()
    if not _check_connectivity(client):
        logger.error("Aborting — fix LiteLLM connection first")
        return
    logger.info("LiteLLM reachable. Starting distillation with %d workers.", workers)
    notify_distill_recipes_start(repo, model, len(records), workers)

    try:
        # Group by test_type for targeted distillation
        by_type: dict[str, list[dict]] = {}
        for r in records:
            by_type.setdefault(r.get("test_type", "unknown"), []).append(r)

        all_recipes: dict[str, list[dict]] = {}
        lock = threading.Lock()

        for test_type, type_records in by_type.items():
            logger.info("Distilling %d '%s' records (%d workers)...", len(type_records), test_type, workers)
            batches = [type_records[i:i + BATCH_SIZE] for i in range(0, len(type_records), BATCH_SIZE)]
            total_batches = len(batches)
            type_recipes: list[dict] = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_distill_batch, batch, idx + 1, total_batches, test_type, repo, model): idx
                    for idx, batch in enumerate(batches)
                }
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    with lock:
                        type_recipes.extend(result)

            type_recipes = _consolidate_recipes(type_recipes, test_type, repo, model, workers=workers, embed_model=embed_model)
            all_recipes[test_type] = type_recipes
            logger.info("  '%s' final: %d recipes", test_type, len(type_recipes))
            notify_distill_recipes_type_done(repo, test_type, len(type_recipes))

        # Mine test expectations from existing bronze review data
        logger.info("Mining test expectations from PR reviews...")
        test_expectations = _mine_review_expectations(repo, model, client)
        logger.info("Extracted %d test expectation patterns", len(test_expectations))

        output = {
            "repo": repo,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "total_source_files": len(records),
            "categories": all_recipes,
            "test_expectations": test_expectations,
        }

        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info("Test knowledge written → %s", out_path)

        total = sum(len(v) for v in all_recipes.values())
        logger.info("Done: %d recipes across %d categories + %d expectation patterns",
                    total, len(all_recipes), len(test_expectations))
        notify_distill_recipes_done(repo, total, len(all_recipes), len(test_expectations))

    except Exception as exc:
        logger.exception("Recipe distillation failed")
        notify_distill_recipes_error(repo, str(exc))
        raise


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Distill recipe silver into gold test knowledge base")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=20, help="Parallel LLM workers per test_type batch")
    p.add_argument("--embed-model", default="instructor-xl",
                   help="Embedding model for deduplication clustering")
    args = p.parse_args()
    distill_recipes(args.repo, args.model, workers=args.workers, embed_model=args.embed_model)


if __name__ == "__main__":
    main()
