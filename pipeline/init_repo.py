"""
init-repo — intelligent repo onboarding command.

Given a GitHub repository, discovers its contributing guide, dev setup docs,
pre-commit config, PR template, CI workflows, and documentation site via:

  1. GitHub file tree scan + LLM categorization
  2. Web search for docs site + httpx fetch of relevant pages
  3. LLM synthesis → repo_config.yaml with pr_preparation section

Usage:
    init-repo --repo vllm-project/vllm
    init-repo --repo sgl-project/sglang --docs-url https://docs.example.com/contributing/
    init-repo --repo sgl-project/sglang --apply      # writes pipeline_config.yaml entry
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"

# ── prompts ──────────────────────────────────────────────────────────

_CATEGORIZE_PROMPT = """\
You are analyzing the full file tree of the GitHub repository "{repo}" to identify files
that contain contributor guidance and development setup information.

We are looking for files in these categories:
- contributing: CONTRIBUTING.md, CONTRIBUTING.rst, DEVELOPMENT.md, HACKING.md, etc.
- dev_setup: installation guides, dev environment setup, requirements-dev.txt, Makefile
- pre_commit: .pre-commit-config.yaml, .pre-commit-hooks.yaml
- pr_template: .github/pull_request_template.md, .github/PULL_REQUEST_TEMPLATE/*.md
- ci_workflows: .github/workflows/*.yml (pick 1-3 most relevant: tests, lint, CI)
- agent_files: CLAUDE.md, .cursorrules, .github/copilot-instructions.md, AGENTS.md
- lint_config: pyproject.toml (if it has [tool.ruff] or [tool.mypy]), .flake8, mypy.ini, setup.cfg

FILE TREE (top-level structure + samples):
{tree_summary}

ALL TOP-LEVEL AND GITHUB FILES (exact paths):
{github_files}

Return a JSON object with paths to fetch for each category. Only include paths that exist.
Use null for categories where no file was found.

{{
  "contributing": "path/to/CONTRIBUTING.md or null",
  "dev_setup": ["path1", "path2"],
  "pre_commit": ".pre-commit-config.yaml or null",
  "pr_template": "path or null",
  "ci_workflows": ["path1", "path2"],
  "agent_files": ["path1"],
  "lint_config": ["pyproject.toml"],
  "docs_search_query": "search query to find the official contributing/developer docs website for this repo"
}}

Return ONLY the JSON object.
"""

_SYNTHESIZE_PROMPT = """\
You are synthesizing contributor guidance for the GitHub repository "{repo}" into a
structured configuration. Your goal: produce a complete repo_config.yaml that will help
a tool automatically guide contributors through PR preparation.

REPOSITORY: {repo}
GITHUB URL: https://github.com/{repo}

DISCOVERED FILES:
{file_contents}

DOCUMENTATION PAGES (from official docs site):
{doc_pages}

EXISTING REPO CONFIG (if any — extend rather than replace):
{existing_config}

Synthesize everything into a JSON object with these exact top-level keys:

{{
  "name": "{repo}",
  "description": "one sentence describing what this repo is and does",
  "focus_keywords": ["keyword1", "keyword2"],
  "focus_areas": ["area1", "area2"],
  "trusted_reviewers": [],
  "scope_guidance": {{
    "repo_wide": true
  }},
  "pr_preparation": {{
    "contributing_urls": ["https://..."],
    "arch_doc_paths": ["CONTRIBUTING.md", "docs/ARCHITECTURE.md"],
    "dev_setup_commands": ["pip install -e '.[dev]'", "pre-commit install"],
    "pre_commit_run_command": "pre-commit run --files {{changed_files}}",
    "lint_commands": ["ruff check {{changed_files}}", "mypy {{changed_files}}"],
    "test_commands": ["pytest tests/ -x -q"],
    "branch_naming_convention": "one-sentence description of the branch naming rule, including any required prefix (e.g. feature/, fix/) and slug style",
    "pr_title_format": "one-sentence description of the PR title format, including required bracketed prefixes if any and max length",
    "commit_message_format": "short description of the commit format used in this repo",
    "commit_message_components": ["[Component]", "[Fix]"],
    "pr_template_sections": ["## Section from PR template"],
    "pr_checklist": ["[ ] checklist item from CONTRIBUTING.md"]
  }},
  "contribution_rules": {{
    "tuning_data": {{
      "description": "Where tuning/config data lives, naming convention, and whether it needs a separate PR from kernel logic",
      "separate_pr": true,
      "target_directory": "path/to/configs/",
      "merge_into_existing": true
    }},
    "hardware_gating": {{
      "description": "How hardware-specific code paths are gated (feature flags, runtime detection, etc.)",
      "mechanism": "describe the gating mechanism"
    }},
    "model_optimization": {{
      "description": "Where model-specific optimizations go and whether they need separate PRs",
      "separate_pr": false
    }},
    "pr_split_guidance": "one sentence describing how to split a combined patch that touches multiple concern types"
  }}
}}

Rules for pr_preparation:
- contributing_urls: official contributing guide URL(s), not GitHub file paths
- arch_doc_paths: repo-relative paths to architecture/contributing docs to mine for structural principles (CONTRIBUTING.md, docs/ARCHITECTURE.md, etc.)
- dev_setup_commands: exact shell commands a new contributor runs to set up dev env
- pre_commit_run_command: the exact command to run pre-commit against changed files
  (e.g. "pre-commit run --files {{changed_files}}"). Set to null if the repo does NOT
  use pre-commit (i.e. no .pre-commit-config.yaml found) — in that case populate
  lint_commands instead.
- lint_commands: ONLY populate when pre_commit_run_command is null. Extract the exact
  commands from CONTRIBUTING.md, CI workflows, or Makefile. Include version-pinned
  install steps if the repo specifies exact versions (e.g. "pip install black==26.3.0
  ruff==0.15.7"). Use {{changed_files}} placeholder. For repos using custom git hooks
  (e.g. .githooks/), include the manual equivalents (black, ruff, clang-format) not
  the hook installer — the installer is a dev_setup_command, not a lint_command.
- test_commands: the standard command to run tests for this repo
- branch_naming_convention: describe the branch naming rule. Look for it in CONTRIBUTING.md,
  README, or PR examples in git history. If the repo shows examples like "feature/my-feature"
  extract the prefix. If no rule is documented write "No enforced prefix — use a short
  descriptive slug with hyphens." Never invent a convention not present in the docs.
- pr_title_format: describe the PR title format. If the repo requires bracketed prefixes
  like [Bugfix] or [Core], list them. If no format is enforced, write "Short imperative
  title. No required prefix." Never invent prefix tags not present in CONTRIBUTING.md.
- commit_message_format: describe the format (e.g. "[Component] Short description (DCO signed)")
- commit_message_components: common prefix tags like [ROCm], [Core], [Bugfix] — empty list if not used

Rules for contribution_rules:
- tuning_data.separate_pr: true if CONTRIBUTING.md or PR history shows tuning/config data in separate PRs from code
- tuning_data.merge_into_existing: true if new configs must be appended to existing files (not new files per model)
- tuning_data.target_directory: the directory where config/tuning data files live
- hardware_gating.mechanism: how hardware-specific paths are guarded (e.g. "get_gfx() string comparison", "IS_CUDA flag", "triton.runtime.driver.active.get_current_target()")
- model_optimization.separate_pr: true if model-specific optimizations need their own PR; usually false
- pr_split_guidance: plain-English rule for how to split a patch that mixes kernel code + tuning data + docs
- Set fields to null if not applicable or not found in docs. Do not invent rules.
- pr_template_sections: actual section headings from the PR template
- pr_checklist: items from the PR checklist in CONTRIBUTING.md or pr_template

Be specific and grounded in what you found. Do not invent items not present in the docs.
If you could not find information for a field, use an empty list or null.

Return ONLY the JSON object.
"""

# ── GitHub discovery ─────────────────────────────────────────────────


def _fetch_tree_and_categorize(
    owner: str,
    name: str,
    model: str,
    gh_token: str,
) -> dict:
    """Fetch file tree, ask LLM to categorize relevant files."""
    from scraper.scrape_recipes import (
        RestClient,
        _fetch_full_tree,
        _get_default_branch,
        _summarize_tree,
    )
    from pipeline.llm import llm_call, parse_json

    with RestClient(gh_token) as client:
        branch = _get_default_branch(client, owner, name)
        logger.info("Default branch: %s", branch)
        tree = _fetch_full_tree(client, owner, name, branch)
        logger.info("Fetched %d tree items", len(tree))

    tree_summary = _summarize_tree(tree)

    # Collect top-level files + all .github/ files for the LLM to see exact paths
    github_files: list[str] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        parts = path.split("/")
        # top-level files
        if len(parts) == 1:
            github_files.append(path)
        # .github/** files
        elif parts[0] == ".github":
            github_files.append(path)

    prompt = _CATEGORIZE_PROMPT.format(
        repo=f"{owner}/{name}",
        tree_summary=tree_summary,
        github_files="\n".join(github_files) if github_files else "(none found)",
    )

    logger.info("Asking LLM to categorize contributing files...")
    raw = llm_call(prompt, model, max_tokens=2048, json_mode=True)
    try:
        categorized = parse_json(raw)
    except Exception as exc:
        logger.warning("LLM categorization failed: %s", exc)
        categorized = {}

    return categorized, tree, branch


def _fetch_github_files(
    owner: str,
    name: str,
    categorized: dict,
    gh_token: str,
) -> dict[str, str]:
    """Fetch content of the categorized files from GitHub."""
    from scraper.scrape_recipes import RestClient, _fetch_content

    # Flatten all paths from categorized
    paths_to_fetch: list[str] = []
    for key, val in categorized.items():
        if key == "docs_search_query" or val is None:
            continue
        if isinstance(val, str):
            paths_to_fetch.append(val)
        elif isinstance(val, list):
            paths_to_fetch.extend(val)

    contents: dict[str, str] = {}
    with RestClient(gh_token) as client:
        for path in paths_to_fetch:
            if not path or not isinstance(path, str):
                continue
            logger.info("Fetching %s/%s: %s", owner, name, path)
            content = _fetch_content(client, owner, name, path)
            if content:
                contents[path] = content[:20_000]  # cap per file
            else:
                logger.warning("Could not fetch %s", path)

    return contents


# ── Web search + doc fetching ─────────────────────────────────────────


def _fetch_doc_page(url: str) -> str | None:
    """Fetch a web page and strip HTML to plain text."""
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; pr-pundit/1.0)"
        })
        if resp.status_code != 200:
            return None
        text = resp.text
        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:15_000]
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def _discover_and_fetch_docs(
    categorized: dict,
    docs_url_override: str | None,
) -> list[dict]:
    """Discover the docs site via real web search (or override) and fetch pages."""
    from pipeline.web_search import web_search

    if docs_url_override:
        candidate_urls = [docs_url_override]
        logger.info("Using provided docs URL: %s", docs_url_override)
    else:
        query = categorized.get("docs_search_query", "")
        if not query:
            return []
        logger.info("Searching for docs: %s", query)
        results = web_search(query, max_results=6)
        candidate_urls = [
            r["href"] for r in results
            if r.get("href", "").startswith("http")
            and "github.com" not in r["href"]
            and "stackoverflow.com" not in r["href"]
            and "reddit.com" not in r["href"]
        ]
        logger.info("Found %d candidate doc URLs", len(candidate_urls))

    pages: list[dict] = []
    for url in candidate_urls[:5]:
        logger.info("Fetching doc page: %s", url)
        content = _fetch_doc_page(url)
        if content:
            pages.append({"url": url, "content": content})
        if sum(len(p["content"]) for p in pages) > 60_000:
            break

    return pages


# ── Synthesis ─────────────────────────────────────────────────────────


def _synthesize_config(
    owner: str,
    name: str,
    file_contents: dict[str, str],
    doc_pages: list[dict],
    existing_config: dict,
    model: str,
) -> dict:
    """Call LLM to synthesize repo_config fields from discovered content."""
    from pipeline.llm import llm_call, parse_json

    # Format file contents
    file_sections: list[str] = []
    total_chars = 0
    for path, content in file_contents.items():
        section = f"=== {path} ===\n{content}"
        file_sections.append(section)
        total_chars += len(section)
        if total_chars > 80_000:
            break
    file_contents_str = "\n\n".join(file_sections) if file_sections else "(no files found)"

    # Format doc pages
    doc_sections: list[str] = []
    for page in doc_pages:
        doc_sections.append(f"=== {page['url']} ===\n{page['content']}")
    doc_pages_str = "\n\n".join(doc_sections) if doc_sections else "(no doc pages found)"

    # Format existing config (only pr_preparation and non-pr-prep fields)
    existing_str = yaml.dump(existing_config, default_flow_style=False) if existing_config else "(none)"

    prompt = _SYNTHESIZE_PROMPT.format(
        repo=f"{owner}/{name}",
        file_contents=file_contents_str,
        doc_pages=doc_pages_str,
        existing_config=existing_str,
    )

    logger.info("Synthesizing repo_config...")
    raw = llm_call(prompt, model, max_tokens=4096, json_mode=True)
    result = parse_json(raw)
    if not isinstance(result, dict):
        raise ValueError(f"LLM returned non-dict: {type(result)}")
    return result


# ── Write outputs ─────────────────────────────────────────────────────


_KERNEL_LIB_SLUGS = {"ROCm_aiter", "ROCm_hipBLASLt", "ROCm_composable_kernel"}

def _infer_targeting(repo: str) -> dict:
    """Infer a targeting section based on the repo's known role.

    Inference rules:
    - Kernel/hardware libraries (aiter, hipBLASLt) → long-term, use_sparingly
    - Inference engines and benchmarking platforms → fast-adoption
    - Unknown repos → fast-adoption with priority=99
    """
    slug = repo.replace("/", "_", 1)
    if slug in _KERNEL_LIB_SLUGS or "aiter" in repo.lower():
        return {
            "tier": "long-term",
            "description": f"{repo} — kernel/hardware library; use as target sparingly; prefer vLLM/SGLang for fast adoption",
            "use_sparingly": True,
            "priority": 10,
            "notify_repos": [],
        }
    # Inference engines and benchmark platforms get fast-adoption by default
    return {
        "tier": "fast-adoption",
        "description": f"{repo} — inference engine or benchmarking platform; primary target for fast upstream adoption",
        "use_sparingly": False,
        "priority": 5,
        "notify_repos": ["ROCm/aiter"],
    }


def _write_repo_config(slug: str, synthesized: dict, existing_config: dict) -> Path:
    """Merge synthesized config into existing repo_config.yaml and write."""
    config_path = GOLD / slug / "repo_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Start from existing config, overlay pr_preparation and missing fields
    merged = dict(existing_config)
    for key in ("name", "description", "focus_keywords", "focus_areas", "scope_guidance"):
        if key in synthesized and not merged.get(key):
            merged[key] = synthesized[key]

    # Always update pr_preparation (the new section)
    if "pr_preparation" in synthesized:
        merged["pr_preparation"] = synthesized["pr_preparation"]

    # Set targeting section if not already present in existing config
    if "targeting" not in merged:
        repo_name = synthesized.get("name", slug.replace("_", "/", 1))
        merged["targeting"] = _infer_targeting(repo_name)

    config_path.write_text(yaml.dump(merged, default_flow_style=False, allow_unicode=True))
    return config_path


def _detect_recipes_repo(owner: str, name: str, tree: list[dict]) -> str | None:
    """Try to auto-detect companion recipes repo from README or known patterns."""
    from scraper.scrape_recipes import KNOWN_RECIPE_REPOS
    slug = f"{owner}/{name}"
    if slug in KNOWN_RECIPE_REPOS:
        return KNOWN_RECIPE_REPOS[slug]
    return None


def _print_pipeline_config_entry(
    owner: str,
    name: str,
    recipes_repo: str | None,
    config_path: Path,
    apply: bool,
) -> None:
    """Print (and optionally write) the pipeline_config.yaml entry."""
    slug = f"{owner}/{name}"
    entry_lines = [f"  {slug}:"]
    if recipes_repo:
        entry_lines.append(f"    recipes_repo: {recipes_repo}  # auto-detected companion repo")

    entry = "\n".join(entry_lines)

    print(f"\nWritten: {config_path}  (review pr_preparation section)")
    print("\nAdd to pipeline_config.yaml under target_repos:")
    print("  target_repos:")
    print(entry)

    if apply:
        pipeline_cfg_path = ROOT / "pipeline_config.yaml"
        try:
            cfg_text = pipeline_cfg_path.read_text()
            if slug in cfg_text:
                print(f"\n(already in pipeline_config.yaml — skipped)")
            else:
                # Append after target_repos: key or at end
                if "target_repos:" in cfg_text:
                    idx = cfg_text.index("target_repos:") + len("target_repos:")
                    # find end of target_repos block — next top-level key or EOF
                    rest = cfg_text[idx:]
                    next_key_match = re.search(r"\n[a-z]", rest)
                    if next_key_match:
                        insert_pos = idx + next_key_match.start()
                        new_text = cfg_text[:insert_pos] + "\n" + entry + cfg_text[insert_pos:]
                    else:
                        new_text = cfg_text.rstrip() + "\n" + entry + "\n"
                    pipeline_cfg_path.write_text(new_text)
                    print(f"\nUpdated pipeline_config.yaml with entry for {slug}")
        except FileNotFoundError:
            logger.warning("pipeline_config.yaml not found — skipping --apply")

    print("\nNext steps:")
    print(f"  1. Review {config_path}")
    print(f"  2. Add the pipeline_config.yaml entry above (or use --apply)")
    print(f"  3. Run: run-pipeline")


# ── Main ──────────────────────────────────────────────────────────────


def init_repo(
    repo: str,
    *,
    model: str = "claude-opus-4-7",
    docs_url: str | None = None,
    apply: bool = False,
) -> None:
    """Discover and write repo_config.yaml for a new repo."""
    owner, name = repo.split("/", 1)
    slug = f"{owner}_{name}"

    gh_token = os.environ.get("GITHUB_TOKEN", "")

    # Load existing config if any
    config_path = GOLD / slug / "repo_config.yaml"
    existing_config: dict = {}
    if config_path.exists():
        try:
            existing_config = yaml.safe_load(config_path.read_text()) or {}
            logger.info("Loaded existing repo_config.yaml")
        except Exception:
            pass

    # Step 1: GitHub file tree scan + LLM categorization
    print(f"[1/3] Scanning GitHub file tree for {repo}...")
    categorized, tree, branch = _fetch_tree_and_categorize(owner, name, model, gh_token)

    contributing_files = [
        k for k, v in categorized.items()
        if k != "docs_search_query" and v and (isinstance(v, str) or len(v) > 0)
    ]
    print(f"      Found categories: {', '.join(contributing_files)}")

    # Step 2: Fetch GitHub file contents
    file_contents = _fetch_github_files(owner, name, categorized, gh_token)
    print(f"      Fetched {len(file_contents)} file(s): {', '.join(file_contents.keys())}")

    # Step 3: Web search + doc page fetching
    print(f"[2/3] Discovering documentation site...")
    doc_pages = _discover_and_fetch_docs(categorized, docs_url)
    if doc_pages:
        print(f"      Fetched {len(doc_pages)} doc page(s): {', '.join(p['url'] for p in doc_pages)}")
    else:
        print("      No external doc pages found")

    # Step 4: LLM synthesis
    print(f"[3/3] Synthesizing repo_config.yaml...")
    synthesized = _synthesize_config(owner, name, file_contents, doc_pages, existing_config, model)

    # Inject raw PR template verbatim — LLM must not paraphrase the template structure.
    # Find whichever fetched file is the PR template and embed it directly.
    pr_template_raw = None
    pr_template_key = categorized.get("pr_template")
    if isinstance(pr_template_key, str) and pr_template_key in file_contents:
        pr_template_raw = file_contents[pr_template_key]
    elif isinstance(pr_template_key, list):
        for p in pr_template_key:
            if p in file_contents:
                pr_template_raw = file_contents[p]
                break
    if pr_template_raw:
        synthesized.setdefault("pr_preparation", {})["pr_template_raw"] = pr_template_raw
        logger.info("Stored raw PR template (%d chars)", len(pr_template_raw))

    # Write repo_config.yaml
    out_path = _write_repo_config(slug, synthesized, existing_config)

    # Detect companion recipes repo
    recipes_repo = _detect_recipes_repo(owner, name, tree)

    # Print pipeline_config.yaml entry + next steps
    _print_pipeline_config_entry(owner, name, recipes_repo, out_path, apply)


def main():
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Intelligently onboard a GitHub repo into PR Pundit"
    )
    p.add_argument("--repo", required=True, help="owner/name (e.g. sgl-project/sglang)")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument(
        "--docs-url",
        default=None,
        help="Override auto-discovered docs URL (e.g. https://docs.vllm.ai/en/latest/contributing/)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Automatically add entry to pipeline_config.yaml",
    )
    args = p.parse_args()

    if "/" not in args.repo:
        print(f"Error: --repo must be owner/name (got {args.repo!r})", file=sys.stderr)
        sys.exit(1)

    init_repo(args.repo, model=args.model, docs_url=args.docs_url, apply=args.apply)


if __name__ == "__main__":
    main()
