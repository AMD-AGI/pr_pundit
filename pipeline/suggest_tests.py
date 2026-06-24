"""
Stage D2 — Test suggestion and generation agent.

Three-stage pipeline:
  1. PR Analyzer   — understands what the PR changes and what claims need evidence
  2. Recipe Selector — picks relevant test recipes from the gold knowledge base
  3. Code Generator  — produces runnable Python test scripts with correct flags

Generates two flavors of test code when applicable:
  - python_api:  uses vllm.LLM / sglang.Engine directly (offline, good for kernels)
  - lm_eval:     uses lm-eval harness (good for accuracy benchmarks)

Usage:
    python -m pipeline.suggest_tests \\
        --repo vllm-project/vllm \\
        --patch diff.patch \\
        [--blurb "Testing AMD MI300X kernel optimizations"] \\
        [--model claude-sonnet-4-6] \\
        [--out tests_output.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.llm import llm_call, make_client, parse_json

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

DEFAULT_MODEL = "claude-opus-4-7"

# ── prompts ──────────────────────────────────────────────────────────

_ANALYZE_PROMPT = """You are analyzing a pull request to understand what supporting tests or benchmarks would be needed to validate it for acceptance into a public open-source ML repository.

REPOSITORY: {repo}
{repo_context}{blurb_section}
PR DIFF (first 6000 chars):
{patch_truncated}

Analyze the PR and return a JSON object:
{{
  "pr_summary": "one sentence describing what the PR does",
  "change_categories": ["list from: new_model, kernel_optimization, quantization, attention_mechanism, serving_feature, bug_fix, documentation, memory_optimization, parallelism, other"],
  "hardware_hints": ["GPU types implied by the diff and repo focus: e.g. A100, H100, MI300X, MI250X"],
  "model_hints": ["model families that would be natural to test with"],
  "claims_needing_evidence": ["performance claims, correctness claims, etc. that reviewers will want validated"],
  "suggested_test_categories": ["from: throughput, accuracy, latency, kernel, e2e"],
  "priority": "high | medium | low — how strongly would reviewers demand benchmarks?"
}}

Return ONLY the JSON object.
"""

_SELECT_PROMPT = """You are selecting the most relevant benchmark recipes for a PR.

PR ANALYSIS:
{analysis_json}

{repo_context}
AVAILABLE TEST KNOWLEDGE BASE (recipes from the repository):
{knowledge_truncated}

Select the most relevant recipes for this PR. Rules:
- Pick at most ONE recipe per test_type (kernel / throughput / latency / accuracy / e2e).
- Prefer variety: if two recipes test the same thing, pick only the best-fitting one.
- Prefer recipes whose hardware_requirements match the hardware_hints in the PR analysis.
- Total selected recipes should be 2-4 maximum.

Return a JSON object:
{{
  "selected_recipes": [
    {{
      "recipe_name": "name from knowledge base",
      "test_type": "kernel | throughput | latency | accuracy | e2e",
      "relevance": "why this recipe is relevant to the PR",
      "adaptations_needed": ["what to change from the template for this PR"],
      "priority": "must_have | nice_to_have"
    }}
  ],
  "test_strategy": "2-3 sentence description of the overall test strategy for this PR",
  "expected_metrics": ["specific metrics to report: e.g. throughput_tok_s, GSM8K_5shot_accuracy"]
}}

Return ONLY the JSON object.
"""

_GENERATE_PROMPT = """You are generating ONE runnable benchmark/test script for a PR submission.

REPOSITORY: {repo}
PR SUMMARY: {pr_summary}
TEST STRATEGY: {test_strategy}
{repo_context}{blurb_section}
RECIPE TO IMPLEMENT:
{recipe_block}

Generate a single complete, runnable script. Requirements:
- Self-contained and well-commented
- Realistic placeholder values with clear comments explaining what to substitute
- Show how to interpret the output / what metrics to report
- For Python: use vllm.LLM or sglang.Engine APIs (NOT Docker)
- For accuracy: use lm_eval with vllm or sglang backend
- Include a results reporting section that prints a clean summary table
- Flag interactions must be correct (e.g. quantization + dtype combinations)
- If the repo targets ROCm/AMD hardware, use ROCm-compatible flags and avoid CUDA-only features

Return a JSON object:
{{
  "name": "descriptive filename (e.g. benchmark_throughput_tp4.py)",
  "flavor": "python_api | lm_eval | shell_script",
  "test_type": "throughput | accuracy | latency | kernel | e2e",
  "description": "what this script tests and when to run it",
  "code": "complete runnable code",
  "how_to_run": "exact command to run this script",
  "what_to_report": "what numbers to paste into the PR description"
}}

Return ONLY the JSON object.
"""

_PR_TEMPLATE_PROMPT = """You are writing a PR description template for benchmark results.

REPOSITORY: {repo}
PR SUMMARY: {pr_summary}
TEST STRATEGY: {test_strategy}
SCRIPTS GENERATED: {script_names}

Write a markdown template the PR author can paste into their PR description to report results.
Include placeholders like TBD or <fill in> for the actual numbers.
Keep it concise — one table per script maximum.

Return ONLY the markdown string (no JSON wrapper).
"""


def _load_repo_config(repo: str) -> dict:
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    path = GOLD / repo_slug / "repo_config.yaml"
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _format_repo_context(config: dict) -> str:
    """Build a compact repo-context block for prompts from repo_config."""
    if not config:
        return ""
    parts = ["REPOSITORY CONTEXT:"]
    if desc := config.get("description", "").strip():
        parts.append(f"Description: {desc}")
    if kws := config.get("focus_keywords", []):
        parts.append(f"Key platform terms: {', '.join(kws[:30])}")
    if areas := config.get("focus_areas", []):
        parts.append("Focus areas:")
        for a in areas[:8]:
            parts.append(f"  - {a}")
    if tg := config.get("test_guidance", {}):
        if hw := tg.get("primary_hardware", []):
            parts.append(f"Primary test hardware: {', '.join(hw)}")
        if notes := tg.get("platform_notes", []):
            parts.append("Platform notes:")
            for n in notes[:5]:
                parts.append(f"  - {n}")
        if scripts := tg.get("known_benchmark_scripts", []):
            parts.append(f"Known benchmark entry points: {', '.join(scripts[:8])}")
        if profiles := tg.get("model_profiles", {}):
            parts.append("Model-family flags for AMD (use exact flags shown — all sourced from AMD/vllm docs):")
            for profile_name, profile in profiles.items():
                desc = profile.get("description", profile_name)
                flags = profile.get("flags", [])
                env = profile.get("env", {})
                notes = profile.get("notes", [])
                line = f"  [{profile_name}] {desc}"
                if flags:
                    line += f" | flags: {' '.join(flags)}"
                if env:
                    line += f" | env: {' '.join(f'{k}={v}' for k, v in env.items())}"
                parts.append(line)
                for note in notes:
                    parts.append(f"    NOTE: {note}")
    return "\n".join(parts) + "\n\n"


def _load_knowledge(repo: str, repo_config: dict | None = None) -> dict:
    """Load primary KB and merge supplemental KBs declared in repo_config.

    Supplemental recipes are tagged with _source_kb so the generator knows
    to adapt their templates to the target repo's API.
    """
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    path = GOLD / repo_slug / "test_knowledge.json"
    if not path.exists():
        primary: dict = {}
    else:
        primary = json.loads(path.read_text())

    if repo_config is None:
        repo_config = _load_repo_config(repo)
    supplemental_repos: list[str] = (
        repo_config.get("test_guidance", {}).get("supplemental_knowledge", [])
    )
    if not supplemental_repos:
        return primary

    # Deep-copy categories so we can annotate without mutating the loaded JSON
    merged_categories: dict[str, list[dict]] = {
        cat: list(recipes)
        for cat, recipes in primary.get("categories", {}).items()
    }

    for sup_repo in supplemental_repos:
        sup_owner, sup_name = sup_repo.split("/", 1)
        sup_slug = f"{sup_owner}_{sup_name}"
        sup_path = GOLD / sup_slug / "test_knowledge.json"
        if not sup_path.exists():
            logger.debug("Supplemental KB not found for %s — run distill-recipes first", sup_repo)
            continue
        sup_kb = json.loads(sup_path.read_text())
        logger.debug("Loading supplemental KB: %s", sup_repo)
        for cat, recipes in sup_kb.get("categories", {}).items():
            tagged = [{**r, "_source_kb": sup_repo} for r in recipes]
            merged_categories.setdefault(cat, []).extend(tagged)

    return {**primary, "categories": merged_categories}


def _format_knowledge_for_prompt(knowledge: dict, categories: list[str]) -> str:
    """Build a compact knowledge string focused on relevant categories."""
    if not knowledge:
        return "No test knowledge base available for this repository."

    parts = []
    all_categories = knowledge.get("categories", {})

    # Include requested categories first, then others up to token budget
    ordered = list(categories) + [c for c in all_categories if c not in categories]

    for cat in ordered:
        recipes = all_categories.get(cat, [])
        if not recipes:
            continue
        parts.append(f"\n=== {cat.upper()} RECIPES ===")
        for r in recipes[:5]:  # cap per category
            source_label = f" [source: {r['_source_kb']}]" if "_source_kb" in r else ""
            parts.append(
                f"\nName: {r.get('name', '')}{source_label}\n"
                f"Use case: {r.get('use_case', '')}\n"
                f"Triggers for: {r.get('trigger_pr_types', [])}\n"
                f"Hardware: {r.get('hardware_requirements', ['any'])}\n"
                f"Tools: {r.get('tools', [])}\n"
                f"Flavor: {r.get('flavor', '')}\n"
                f"Key flags: {json.dumps(r.get('key_flags', {}))}\n"
                f"Metrics: {r.get('metrics_reported', [])}\n"
                f"Template:\n{r.get('code_template', '')[:1500]}"
            )

    # Include expectations
    expectations = knowledge.get("test_expectations", [])
    if expectations:
        parts.append("\n=== REVIEWER EXPECTATIONS ===")
        for e in expectations[:10]:
            parts.append(
                f"PR type: {e.get('pr_category', '')}\n"
                f"Expected tests: {e.get('expected_tests', [])}\n"
                f"Typical metrics: {e.get('typical_metrics', [])}"
            )

    return "\n".join(parts)


def _format_selected_recipes(selection: dict, knowledge: dict, target_repo: str) -> str:
    """Build a block with full templates for the selected recipes."""
    selected = selection.get("selected_recipes", [])
    if not selected:
        return ""

    all_recipes: list[dict] = []
    for recipes in knowledge.get("categories", {}).values():
        all_recipes.extend(recipes)

    name_to_recipe = {r.get("name", ""): r for r in all_recipes}
    parts = []
    for sel in selected:
        name = sel.get("recipe_name", "")
        recipe = name_to_recipe.get(name, {})
        source_kb = recipe.get("_source_kb")
        source_note = (
            f"SOURCE: {source_kb} — this template uses {source_kb}'s API. "
            f"Adapt it to {target_repo}'s API when generating code.\n"
            if source_kb else ""
        )
        parts.append(
            f"--- {name} ---\n"
            f"{source_note}"
            f"Relevance: {sel.get('relevance', '')}\n"
            f"Adaptations needed: {sel.get('adaptations_needed', [])}\n"
            f"Template:\n{recipe.get('code_template', 'No template available')}"
        )
    return "\n\n".join(parts)


def _format_single_recipe(sel: dict, knowledge: dict, target_repo: str) -> str:
    """Build a prompt block for a single selected recipe."""
    all_recipes: list[dict] = []
    for recipes in knowledge.get("categories", {}).values():
        all_recipes.extend(recipes)
    name_to_recipe = {r.get("name", ""): r for r in all_recipes}

    name = sel.get("recipe_name", "")
    recipe = name_to_recipe.get(name, {})
    source_kb = recipe.get("_source_kb")
    source_note = (
        f"SOURCE: {source_kb} — this template uses {source_kb}'s API. "
        f"Adapt it to {target_repo}'s API when generating code.\n"
        if source_kb else ""
    )
    return (
        f"Recipe: {name}\n"
        f"{source_note}"
        f"Relevance: {sel.get('relevance', '')}\n"
        f"Adaptations needed: {sel.get('adaptations_needed', [])}\n"
        f"Template:\n{recipe.get('code_template', 'No template available')}"
    )


def suggest_tests(
    repo: str,
    patch_text: str,
    blurb: str = "",
    model: str = DEFAULT_MODEL,
    log_callback=None,
) -> dict:
    def log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    client = make_client()
    blurb_section = f"USER CONTEXT: {blurb}\n" if blurb else ""
    repo_config = _load_repo_config(repo)
    repo_context = _format_repo_context(repo_config)

    # Stage 1: Analyze the PR
    log("Stage 1/3: Analyzing PR...")
    analyze_prompt = _ANALYZE_PROMPT.format(
        repo=repo,
        repo_context=repo_context,
        blurb_section=blurb_section,
        patch_truncated=patch_text[:6000],
    )
    raw_analysis = llm_call(analyze_prompt, model, client=client, max_tokens=2048, json_mode=True)
    analysis = parse_json(raw_analysis)
    log(f"  PR: {analysis.get('pr_summary', '')}")
    log(f"  Categories: {analysis.get('change_categories', [])}")
    log(f"  Suggested tests: {analysis.get('suggested_test_categories', [])}")

    # Load knowledge base (primary + supplemental repos declared in repo_config)
    knowledge = _load_knowledge(repo, repo_config)
    if not knowledge:
        log(f"WARNING: No test knowledge base found for {repo}. Run distill-recipes first.")

    suggested_categories = analysis.get("suggested_test_categories", [])
    knowledge_str = _format_knowledge_for_prompt(knowledge, suggested_categories)

    # Stage 2: Select relevant recipes
    log("Stage 2/3: Selecting relevant recipes...")
    select_prompt = _SELECT_PROMPT.format(
        analysis_json=json.dumps(analysis, indent=2),
        repo_context=repo_context,
        knowledge_truncated=knowledge_str[:12000],
    )
    raw_selection = llm_call(select_prompt, model, client=client, max_tokens=4096, json_mode=True)
    selection = parse_json(raw_selection)
    selected = selection.get("selected_recipes", [])

    # Dedup: keep only first recipe per test_type
    seen_types: set[str] = set()
    deduped = []
    for r in selected:
        tt = r.get("test_type", r.get("recipe_name", ""))
        if tt not in seen_types:
            seen_types.add(tt)
            deduped.append(r)
    if len(deduped) < len(selected):
        log(f"  Deduped {len(selected)} → {len(deduped)} recipes (one per test_type)")
    selected = deduped
    selection["selected_recipes"] = selected

    log(f"  Selected {len(selected)} recipes")
    log(f"  Strategy: {selection.get('test_strategy', '')}")

    # Stage 3: Generate one script per selected recipe
    log(f"Stage 3/3: Generating {len(selected)} script(s) one at a time...")
    scripts: list[dict] = []
    pr_summary = analysis.get("pr_summary", "")
    test_strategy = selection.get("test_strategy", "")

    for i, sel in enumerate(selected):
        recipe_name = sel.get("recipe_name", f"recipe_{i+1}")
        log(f"  [{i+1}/{len(selected)}] Generating: {recipe_name}...")
        recipe_block = _format_single_recipe(sel, knowledge, target_repo=repo)
        generate_prompt = _GENERATE_PROMPT.format(
            repo=repo,
            pr_summary=pr_summary,
            test_strategy=test_strategy,
            repo_context=repo_context,
            blurb_section=blurb_section,
            recipe_block=recipe_block[:8000],
        )
        raw = llm_call(generate_prompt, model, client=client, max_tokens=16384, json_mode=True)
        script = parse_json(raw)
        if isinstance(script, dict) and script.get("code"):
            scripts.append(script)
            log(f"    → {script.get('name', recipe_name)}")
        else:
            log(f"    ✗ skipped (empty output)")

    # Generate PR description template
    log("  Generating PR description template...")
    pr_template_prompt = _PR_TEMPLATE_PROMPT.format(
        repo=repo,
        pr_summary=pr_summary,
        test_strategy=test_strategy,
        script_names=", ".join(s.get("name", "") for s in scripts),
    )
    pr_description_template = llm_call(
        pr_template_prompt, model, client=client, max_tokens=2048, json_mode=False
    )

    log(f"  Done — {len(scripts)} scripts generated")
    return {
        "repo": repo,
        "analysis": analysis,
        "selection": selection,
        "scripts": scripts,
        "pr_description_template": pr_description_template,
    }


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Suggest and generate supporting tests for a PR")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--patch", required=True, help="Path to .patch / .diff file")
    p.add_argument("--blurb", default="", help="Optional context about what to test")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = p.parse_args()

    patch_text = Path(args.patch).read_text()
    result = suggest_tests(args.repo, patch_text, blurb=args.blurb, model=args.model)

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out_json)
        logger.info("Results written to %s", args.out)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
