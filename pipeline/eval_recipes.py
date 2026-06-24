"""
Evaluation harness for recipe distillation quality.

Three checks:
  1. syntax_validity   — all code_template fields parse (Python: ast, shell: basic checks)
  2. coverage          — sampled source files are represented in the gold recipes
  3. suggestion_relevance — suggest_tests matches what reviewers actually asked for in PRs

Each check returns an EvalResult with a 0-1 score and textual feedback (for GEPA).

Also provides a DSPy module + GEPA-compatible metric for prompt optimization.

Usage:
    # Evaluate current gold output
    python -m pipeline.eval_recipes --repo vllm-project/vllm

    # Run GEPA optimization over the distillation prompt
    python -m pipeline.eval_recipes --repo vllm-project/vllm --optimize
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

SILVER = Path(__file__).resolve().parent.parent / "data" / "silver_recipes"
BRONZE = Path(__file__).resolve().parent.parent / "data" / "bronze"
GOLD   = Path(__file__).resolve().parent.parent / "data" / "gold"
EVAL_LOG = Path(__file__).resolve().parent.parent / "data" / "eval_logs"

DEFAULT_MODEL = "claude-opus-4-7"
COVERAGE_SAMPLE_SIZE = 20
RELEVANCE_SAMPLE_SIZE = 15


# ── result types ─────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    score: float          # 0.0 – 1.0
    feedback: str         # natural-language description of failures (for GEPA)
    details: list[dict] = field(default_factory=list)


@dataclass
class EvalResult:
    repo: str
    timestamp: str
    model: str
    checks: list[CheckResult]

    @property
    def overall_score(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.score for c in self.checks) / len(self.checks)

    @property
    def combined_feedback(self) -> str:
        parts = [f"[{c.name}] score={c.score:.2f}: {c.feedback}" for c in self.checks]
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "timestamp": self.timestamp,
            "model": self.model,
            "overall_score": self.overall_score,
            "checks": [
                {"name": c.name, "score": c.score, "feedback": c.feedback, "details": c.details}
                for c in self.checks
            ],
        }


# ── check 1: syntax validity ─────────────────────────────────────────

def _check_python_template(template: str) -> tuple[bool, str]:
    try:
        ast.parse(template)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def _check_shell_template(template: str) -> tuple[bool, str]:
    # Minimal shell checks: has at least one command, no obvious truncation
    if not template.strip():
        return False, "empty template"
    if template.count("{") != template.count("}"):
        return False, "unbalanced braces (likely truncated)"
    return True, ""


def check_syntax_validity(knowledge: dict) -> CheckResult:
    recipes: list[dict] = []
    for cat_recipes in knowledge.get("categories", {}).values():
        recipes.extend(cat_recipes)

    if not recipes:
        return CheckResult("syntax_validity", 0.0, "No recipes found in knowledge base")

    failures: list[dict] = []
    for r in recipes:
        template = r.get("code_template", "")
        flavor = r.get("flavor", "")
        name = r.get("name", "unnamed")

        if not template:
            failures.append({"recipe": name, "error": "missing code_template"})
            continue

        if flavor in ("python_api", "lm_eval") or template.strip().startswith(
            ("import ", "from ", "def ", "class ", "#!")
        ):
            ok, err = _check_python_template(template)
        else:
            ok, err = _check_shell_template(template)

        if not ok:
            failures.append({"recipe": name, "error": err, "flavor": flavor})

    score = 1.0 - len(failures) / len(recipes)
    if failures:
        feedback = (
            f"{len(failures)}/{len(recipes)} recipes have invalid code templates. "
            f"Examples: {'; '.join(f['recipe'] + ': ' + f['error'] for f in failures[:3])}"
        )
    else:
        feedback = f"All {len(recipes)} recipe templates are syntactically valid."

    return CheckResult("syntax_validity", score, feedback, failures)


# ── check 2: source coverage ─────────────────────────────────────────

_COVERAGE_PROMPT = """You are checking whether a gold test knowledge base adequately captures the content of a source file.

SOURCE FILE: {path}
CONTENT (first 2000 chars):
{content}

GOLD KNOWLEDGE BASE RECIPES (names and use cases):
{recipe_list}

Question: Is the benchmark/test knowledge from this source file captured in any of the recipes above?
Answer with a JSON object:
{{
  "covered": true | false,
  "matching_recipe": "recipe name if covered, else null",
  "gap": "if not covered, what knowledge from this file is missing from the recipes"
}}
"""


def check_coverage(
    knowledge: dict, silver_path: Path, model: str, n: int = COVERAGE_SAMPLE_SIZE
) -> CheckResult:
    from pipeline.llm import llm_call, parse_json

    if not silver_path.exists():
        return CheckResult("coverage", 0.0, "Silver file not found")

    records = [json.loads(l) for l in silver_path.read_text().splitlines() if l.strip()]
    if not records:
        return CheckResult("coverage", 0.0, "No silver records to sample")

    # Only sample files that actually have benchmark content
    benchmark_records = [
        r for r in records
        if r.get("test_type", "unknown") != "unknown"
        and r.get("source_type") != "docs"
    ]
    sample = random.sample(benchmark_records, min(n, len(benchmark_records)))

    # Build a compact recipe list for the prompt
    all_recipes = []
    for cat_recipes in knowledge.get("categories", {}).values():
        all_recipes.extend(cat_recipes)
    recipe_list = "\n".join(
        f"- {r.get('name', '')}: {r.get('use_case', '')}" for r in all_recipes
    )

    covered = 0
    details = []
    for rec in sample:
        prompt = _COVERAGE_PROMPT.format(
            path=rec["path"],
            content=rec.get("content", "")[:2000],
            recipe_list=recipe_list,
        )
        try:
            raw = llm_call(prompt, model, max_tokens=512, json_mode=True)
            result = parse_json(raw)
            is_covered = result.get("covered", False)
            if is_covered:
                covered += 1
            details.append({
                "path": rec["path"],
                "covered": is_covered,
                "matching_recipe": result.get("matching_recipe"),
                "gap": result.get("gap"),
            })
        except Exception as exc:
            logger.warning("Coverage check failed for %s: %s", rec["path"], exc)
            details.append({"path": rec["path"], "covered": False, "gap": str(exc)})

    score = covered / len(sample) if sample else 0.0
    gaps = [d for d in details if not d["covered"] and d.get("gap")]
    if gaps:
        feedback = (
            f"{covered}/{len(sample)} sampled source files are covered by recipes. "
            f"Missing knowledge from: {'; '.join(d['path'] + ': ' + str(d.get('gap', ''))[:80] for d in gaps[:3])}"
        )
    else:
        feedback = f"All {covered}/{len(sample)} sampled source files are covered by recipes."

    return CheckResult("coverage", score, feedback, details)


# ── check 3: suggestion relevance ────────────────────────────────────

_RELEVANCE_PROMPT = """You are evaluating whether a test suggestion system correctly identified the types of supporting tests that were actually run for a PR.

The ground truth comes from the PR description itself — the author reported what tests they ran and what results they got. This is the best signal: if the author ran a throughput benchmark and pasted results, that test type was clearly needed for this PR.

GROUND TRUTH — tests actually performed (from PR description and any reviewer acknowledgements):
{ground_truth}

CHANGED FILES in the PR (to understand what the PR touches):
{changed_files}

SUGGESTED TESTS (from the test suggestion agent, given only the changed file list):
{suggested_tests}

Rate how well the suggestions match what was actually needed.
Return a JSON object:
{{
  "score": 0.0 to 1.0,
  "matched": ["test types the agent correctly suggested"],
  "missed": ["test types that were done but agent didn't suggest"],
  "extra": ["agent suggested these but they weren't needed — not a penalty, just FYI"],
  "feedback": "one sentence explaining the score"
}}

Scoring guide:
- 1.0: agent suggested all the test types that were actually performed
- 0.7: agent suggested most of them, missed one minor one
- 0.4: agent missed the primary test type
- 0.0: agent suggested completely irrelevant tests
"""

# Coarse pre-filter: does this PR description contain evidence of test results?
# Intentionally permissive — false positives are fine, the LLM judge handles them.
# False negatives (missed PRs) don't matter since we only need 15-20 candidates.
_TEST_EVIDENCE_RE = re.compile(
    r"""
    (?:
        # Markdown table headers with perf/accuracy columns
        \|\s*(?:throughput|tok(?:ens)?[/_]s|req(?:uests)?[/_]s|latency|accuracy|
                gsm8k|mmlu|humaneval|score|tpot|ttft|itl|p99|p95)
        |
        # Numeric results with units  e.g. "1234 tok/s", "72.3%", "45ms"
        \d+\.?\d*\s*(?:tok(?:ens)?[/_]s|req[/_]s|ms\b|tokens?\s+per\s+second|%\s+acc)
        |
        # Accuracy improvement patterns  e.g. "from 72.3% to 74.1%"
        from\s+\d+\.?\d*%\s+to\s+\d+\.?\d*%
        |
        # Explicit benchmark tool names
        (?:lm[_-]eval|benchmark_serving|benchmark_throughput|
           vllm\.llm|sglang\.engine|python\s+-m\s+vllm\.entrypoints)
        |
        # "ran/ran/measured/evaluated ... benchmark/accuracy/throughput"
        (?:i\s+(?:ran|tested|measured|evaluated)|we\s+(?:ran|tested)|
           benchmark(?:ed|ing|s)?|perf(?:ormance)?\s+test)
        .{0,100}
        (?:throughput|latency|accuracy|gsm8k|mmlu|\d+\s*tok)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _mine_pr_test_evidence(bronze_dir: Path, n: int) -> list[dict]:
    """
    Find PRs where the author reported test results in the PR description.
    These are our ground-truth examples: we know what tests were done.
    """
    prs_path = bronze_dir / "pull_requests.jsonl"
    if not prs_path.exists():
        return []

    candidates: list[dict] = []
    for line in prs_path.read_text().splitlines():
        if not line.strip():
            continue
        pr = json.loads(line)
        body = pr.get("body", "") or ""
        pr_num = pr.get("number")
        if not pr_num or not body:
            continue
        if _TEST_EVIDENCE_RE.search(body):
            candidates.append({"pr_number": pr_num, "description": body[:1500]})
        if len(candidates) >= n * 3:
            break

    return candidates


def check_suggestion_relevance(
    repo: str, knowledge: dict, model: str, n: int = RELEVANCE_SAMPLE_SIZE,
    fast: bool = False,
) -> CheckResult:
    from pipeline.llm import llm_call, parse_json
    from pipeline.suggest_tests import _ANALYZE_PROMPT, _load_repo_config, _format_repo_context
    from pipeline.suggest_tests import suggest_tests

    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    bronze_dir = BRONZE / repo_slug

    # Primary signal: PRs where authors reported test results in the description
    candidates = _mine_pr_test_evidence(bronze_dir, n)

    if not candidates:
        return CheckResult(
            "suggestion_relevance", 0.5,
            "No PRs with test evidence found in descriptions (neutral score)"
        )

    # Load changed files per PR for synthetic patch
    files_path = bronze_dir / "files.jsonl"
    pr_files: dict[int, list[str]] = {}
    if files_path.exists():
        for line in files_path.read_text().splitlines():
            if not line.strip():
                continue
            f = json.loads(line)
            pn = f.get("_pr_number")
            if pn:
                pr_files.setdefault(pn, []).append(f.get("path", ""))

    # Filter to PRs where we also have file info
    candidates = [c for c in candidates if pr_files.get(c["pr_number"])]
    if not candidates:
        return CheckResult("suggestion_relevance", 0.5,
                           "No PRs with both test evidence and file info (neutral score)")

    sample = random.sample(candidates, min(n, len(candidates)))
    scores = []
    details = []

    for item in sample:
        pr_num = item["pr_number"]
        changed_files = pr_files.get(pr_num, [])
        synthetic_patch = "\n".join(f"diff --git a/{f} b/{f}" for f in changed_files[:20])

        try:
            if fast:
                # Fast mode: 1 LLM call (analyze only) instead of 3 full pipeline calls
                repo_context = _format_repo_context(_load_repo_config(repo))
                analyze_prompt = _ANALYZE_PROMPT.format(
                    repo=repo,
                    repo_context=repo_context,
                    blurb_section="",
                    patch_truncated=synthetic_patch[:6000],
                )
                raw_analysis = llm_call(analyze_prompt, model, max_tokens=1024, json_mode=True)
                analysis = parse_json(raw_analysis)
                cats = analysis.get("suggested_test_categories", [])
                suggested_str = (
                    "Suggested test categories: " + ", ".join(cats)
                    if cats else "No suggestions generated"
                )
            else:
                suggestion = suggest_tests(repo, synthetic_patch, model=model)
                suggested_names = [
                    s.get("name", "") + ": " + s.get("description", "")
                    for s in suggestion.get("scripts", [])
                ]
                suggested_str = "\n".join(f"- {s}" for s in suggested_names) or "No suggestions generated"

            prompt = _RELEVANCE_PROMPT.format(
                ground_truth=item["description"],
                changed_files="\n".join(changed_files[:20]),
                suggested_tests=suggested_str,
            )
            raw = llm_call(prompt, model, max_tokens=512, json_mode=True)
            result = parse_json(raw)
            item_score = float(result.get("score", 0.5))
            scores.append(item_score)
            details.append({
                "pr_number": pr_num,
                "score": item_score,
                "matched": result.get("matched", []),
                "missed": result.get("missed", []),
                "feedback": result.get("feedback", ""),
            })
        except Exception as exc:
            logger.warning("Relevance check failed for PR #%d: %s", pr_num, exc)

    if not scores:
        return CheckResult("suggestion_relevance", 0.5, "Could not evaluate relevance (neutral)")

    avg_score = sum(scores) / len(scores)
    misses = [d for d in details if d.get("missed")]
    if misses:
        feedback = (
            f"Average relevance score: {avg_score:.2f} across {len(scores)} PRs. "
            f"Common misses: {'; '.join(str(d['missed'][:2]) for d in misses[:3])}"
        )
    else:
        feedback = f"Average relevance score: {avg_score:.2f} across {len(scores)} PRs."

    return CheckResult("suggestion_relevance", avg_score, feedback, details)


# ── full eval ────────────────────────────────────────────────────────

def evaluate(
    repo: str, model: str = DEFAULT_MODEL, skip_relevance: bool = False, fast: bool = False
) -> EvalResult:
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"

    knowledge_path = GOLD / repo_slug / "test_knowledge.json"
    if not knowledge_path.exists():
        raise FileNotFoundError(f"No test knowledge found for {repo} — run distill-recipes first")

    knowledge = json.loads(knowledge_path.read_text())
    silver_path = SILVER / repo_slug / "records.jsonl"

    checks: list[CheckResult] = []

    logger.info("Check 1/3: syntax validity...")
    checks.append(check_syntax_validity(knowledge))
    logger.info("  score=%.2f — %s", checks[-1].score, checks[-1].feedback[:100])

    logger.info("Check 2/3: source coverage (sampling %d files)...", COVERAGE_SAMPLE_SIZE)
    checks.append(check_coverage(knowledge, silver_path, model))
    logger.info("  score=%.2f — %s", checks[-1].score, checks[-1].feedback[:100])

    if not skip_relevance:
        mode = "fast" if fast else "full"
        logger.info("Check 3/3: suggestion relevance (%s mode, sampling %d PRs)...", mode, RELEVANCE_SAMPLE_SIZE)
        checks.append(check_suggestion_relevance(repo, knowledge, model, fast=fast))
        logger.info("  score=%.2f — %s", checks[-1].score, checks[-1].feedback[:100])

    result = EvalResult(
        repo=repo,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        checks=checks,
    )

    logger.info("Overall score: %.2f", result.overall_score)

    # Persist to eval log
    EVAL_LOG.mkdir(parents=True, exist_ok=True)
    log_path = EVAL_LOG / f"{repo_slug}_recipe_eval.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
    logger.info("Eval log appended → %s", log_path)

    return result


# ── DSPy module + GEPA optimization ──────────────────────────────────

def build_dspy_optimizer(repo: str, model: str = DEFAULT_MODEL):
    """
    Build a GEPA-optimized DSPy module for the distillation prompt.

    GEPA is an evolutionary optimizer that uses natural language feedback
    to evolve prompts — ideal here because our metric produces rich textual
    explanations of recipe quality failures.
    """
    import dspy
    from dspy import GEPA, GEPAFeedbackMetric

    from pipeline.llm import make_client

    # Configure DSPy to use LiteLLM proxy
    import os
    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000"),
        api_key=os.environ.get("LITELLM_MASTER_KEY", ""),
    )
    dspy.configure(lm=lm)

    # ── DSPy signatures matching our distillation prompts ────────────

    class RecipeDistillSignature(dspy.Signature):
        """
        Distill a batch of benchmark/test source files from an ML repository
        into structured, reusable test recipe templates.

        Focus on throughput, accuracy, latency, and kernel benchmark patterns.
        Extract canonical code templates with correct flags and placeholder variables.
        Consolidate similar files into one recipe. Skip unit tests.
        """
        repo: str = dspy.InputField(desc="Repository name e.g. vllm-project/vllm")
        test_type: str = dspy.InputField(desc="Category being distilled: throughput|accuracy|latency|kernel|e2e")
        files_block: str = dspy.InputField(desc="Batch of source files with path, detected metadata, and content")
        recipes_json: str = dspy.OutputField(desc="JSON array of TestRecipe objects")

    class RecipeDistiller(dspy.Module):
        def __init__(self):
            self.distill = dspy.ChainOfThought(RecipeDistillSignature)

        def forward(self, repo: str, test_type: str, files_block: str) -> dspy.Prediction:
            return self.distill(repo=repo, test_type=test_type, files_block=files_block)

    # ── training examples from silver records ────────────────────────

    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    silver_path = SILVER / repo_slug / "records.jsonl"

    if not silver_path.exists():
        raise FileNotFoundError(f"Silver not found for {repo} — run normalize-recipes first")

    records = [json.loads(l) for l in silver_path.read_text().splitlines() if l.strip()]

    # Group into batches by test_type for training examples
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r.get("test_type", "unknown"), []).append(r)

    train_examples = []
    for test_type, type_records in by_type.items():
        if test_type == "unknown":
            continue
        batch = type_records[:10]  # small batch per type
        files_block = "\n\n".join(
            f"--- {r['path']} ---\n{r['content'][:1500]}" for r in batch
        )
        train_examples.append(dspy.Example(
            repo=repo,
            test_type=test_type,
            files_block=files_block,
        ).with_inputs("repo", "test_type", "files_block"))

    logger.info("Built %d training examples for GEPA", len(train_examples))

    # ── GEPA metric with textual feedback ────────────────────────────

    knowledge_path = GOLD / repo_slug / "test_knowledge.json"

    def gepa_metric(
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace=None,
        pred_name=None,
        pred_trace=None,
    ):
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback
        from pipeline.llm import parse_json

        recipes_json = getattr(pred, "recipes_json", "") or ""
        feedback_parts = []
        score = 0.0

        try:
            recipes = parse_json(recipes_json)
            if not isinstance(recipes, list):
                return ScoreWithFeedback(score=0.0, feedback="Output is not a JSON array of recipes")

            # Sub-check 1: syntax validity
            syntax_result = check_syntax_validity({"categories": {gold.test_type: recipes}})
            score += syntax_result.score * 0.4
            if syntax_result.score < 1.0:
                feedback_parts.append(f"Syntax issues: {syntax_result.feedback}")

            # Sub-check 2: spot coverage — are input files referenced in output recipes?
            input_paths = re.findall(r"--- (.+?) ---", gold.files_block)
            recipe_source_files = []
            for r in recipes:
                recipe_source_files.extend(r.get("source_files", []))
            source_text = " ".join(recipe_source_files)
            covered = sum(1 for p in input_paths if any(
                Path(p).name in sf for sf in recipe_source_files
            ))
            coverage_score = covered / len(input_paths) if input_paths else 0.5
            score += coverage_score * 0.3
            if coverage_score < 0.8:
                uncovered = [p for p in input_paths if not any(Path(p).name in sf for sf in recipe_source_files)]
                feedback_parts.append(
                    f"Coverage: {covered}/{len(input_paths)} source files referenced in recipes. "
                    f"Uncovered: {uncovered[:3]}"
                )

            # Sub-check 3: recipe quality heuristics
            quality_issues = []
            for r in recipes:
                if not r.get("code_template"):
                    quality_issues.append(f"'{r.get('name', '?')}' has no code_template")
                if not r.get("key_flags"):
                    quality_issues.append(f"'{r.get('name', '?')}' has no key_flags")
                if not r.get("metrics_reported"):
                    quality_issues.append(f"'{r.get('name', '?')}' has no metrics_reported")
            quality_score = max(0.0, 1.0 - len(quality_issues) / max(len(recipes), 1) * 0.5)
            score += quality_score * 0.3
            if quality_issues:
                feedback_parts.append(f"Quality issues: {'; '.join(quality_issues[:3])}")

        except Exception as exc:
            feedback_parts.append(f"Failed to parse output as JSON: {exc}")
            score = 0.0

        score = min(1.0, max(0.0, score))
        feedback = " | ".join(feedback_parts) if feedback_parts else "Good output."
        return ScoreWithFeedback(score=score, feedback=feedback)

    # ── Assemble GEPA optimizer ──────────────────────────────────────

    metric = GEPAFeedbackMetric(gepa_metric)
    optimizer = GEPA(
        metric=metric,
        auto="medium",
        log_dir=str(EVAL_LOG / f"{repo_slug}_gepa_logs"),
        track_stats=True,
    )

    return RecipeDistiller(), optimizer, train_examples


def run_optimization(repo: str, model: str = DEFAULT_MODEL, out_path: Path | None = None):
    """Run GEPA to optimize the distillation prompt, save the compiled module."""
    logger.info("Building DSPy module and GEPA optimizer...")
    module, optimizer, train_examples = build_dspy_optimizer(repo, model)

    logger.info("Running GEPA optimization (%d training examples)...", len(train_examples))
    optimized = optimizer.compile(module, trainset=train_examples)

    save_path = out_path or (EVAL_LOG / f"{repo.replace('/', '_')}_optimized_distiller.json")
    optimized.save(str(save_path))
    logger.info("Optimized module saved → %s", save_path)
    return optimized


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Evaluate and optimize recipe distillation")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--optimize", action="store_true",
                   help="Run GEPA prompt optimization after evaluation")
    p.add_argument("--skip-relevance", action="store_true",
                   help="Skip suggestion relevance check (slower, needs LLM calls)")
    p.add_argument("--fast", action="store_true",
                   help="Fast relevance check: 1 LLM call per PR (analyze only) vs 3 (full pipeline). "
                        "Good for GEPA optimization loops.")
    p.add_argument("--out", default=None, help="Path to save optimized module (with --optimize)")
    args = p.parse_args()

    result = evaluate(args.repo, model=args.model, skip_relevance=args.skip_relevance, fast=args.fast)

    print(f"\n{'='*60}")
    print(f"Eval results for {args.repo}")
    print(f"Overall score: {result.overall_score:.2f}")
    for c in result.checks:
        print(f"  {c.name}: {c.score:.2f} — {c.feedback[:120]}")
    print(f"{'='*60}\n")

    if args.optimize:
        run_optimization(args.repo, model=args.model,
                        out_path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
