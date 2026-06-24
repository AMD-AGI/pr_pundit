"""
PR preparation package generator.

Given a diff and a repo slug (with pr_preparation section in repo_config.yaml),
produces a complete PR prep package:
  - contributing_checklist: items + shell commands to run
  - commit_message: a draft commit message in the repo's style
  - pr_description: a PR description draft with placeholders for benchmark results
  - commands_to_run: exact shell commands the contributor should run locally

Called by the prepare_pr MCP tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

_DIFF_CAP = 600_000

_PREPARE_PROMPT = """\
You are helping a contributor prepare a pull request for the GitHub repository "{repo}".
Your goal is to produce a complete PR preparation package grounded in this repo's
contributing guide, PR template, and pre-commit configuration.

REPOSITORY: {repo}
GITHUB URL: https://github.com/{repo}

PR DIFF (first {diff_cap} chars):
{diff_truncated}

{blurb_section}{pr_template_section}CONTRIBUTING GUIDANCE FOR THIS REPO:
{pr_prep_section}

{judge_section}{test_section}{benchmark_section}\
Generate a JSON object with these exact keys:

{{
  "pr_title": "complete PR title with the repo's required category prefix (e.g. [Kernel], [Core], [Model], [Bugfix])",
  "contributing_checklist": [
    {{
      "item": "short description of checklist item",
      "command": "exact shell command to run (or null if manual)",
      "required": true
    }}
  ],
  "commit_message": "complete commit message in the repo's style with correct prefix component",
  "pr_description": "complete PR description draft in markdown, following the repo's PR template sections. Include actual benchmark results where provided; use 'TBD' placeholders only where data is genuinely missing.",
  "commands_to_run": [
    "exact lint/format/test command 1",
    "exact lint/format/test command 2"
  ],
  "submission_instructions": "step-by-step instructions for opening the PR: fork URL, branch push command, where to paste the PR description, DCO or CLA requirements",
  "verification_gaps": [
    "list any claims in the PR that could not be verified from the diff or benchmark data alone"
  ]
}}

Rules:
- pr_title: MUST start with the repo's required category prefix in square brackets.
  For vllm-project/vllm the valid prefixes are: [Bugfix] [CI/Build] [Doc] [Model]
  [Frontend] [Kernel] [Core] [Hardware][Vendor] [Misc]. Choose the prefix(es) that
  match the diff content — [Kernel] for Triton/CUDA kernel files, [Core] for engine
  logic, [Model] for model class changes, [Hardware][AMD] for ROCm/HIP-specific work.
  If multiple categories apply, include all (e.g. "[Kernel][Model]"). The title after
  the prefix should be a concise imperative phrase (no trailing period).
- contributing_checklist: include all items from the repo checklist that are relevant
  to this diff (skip items clearly not applicable, e.g. docs update if no docs changed)
- For each checklist item with a known command, include the exact command
- commit_message: use the repo's commit message format — extract the right [Component]
  prefix from the diff content (e.g. [ROCm] for ROCm-specific changes)
- pr_description: follow the ACTUAL PR TEMPLATE structure exactly — use the same section
  headings and ordering as in PR_TEMPLATE above. Do not invent sections. If benchmark_results
  are provided, use the real numbers. Distinguish clearly between "path broken before PR"
  and "path works after PR" — do not conflate baseline and post-PR numbers.
  CRITICAL: do NOT include a "Checklist" or "Contributing checklist" section inside
  pr_description. Checklist items go in the separate contributing_checklist JSON field only.
  The PR body that maintainers see must not contain internal contributor checklists.
- For kernel/dispatch PRs: always include a before/after table. If before numbers are
  unavailable because the code path was broken pre-PR, say so explicitly.
- CRITICAL — no fabricated performance numbers: ONLY include throughput, latency,
  speedup percentages, or benchmark scores that appear in the benchmark_results block
  above or are quoted verbatim in the diff/README source material. If no measured data
  is available, write "TBD — to be filled after benchmarking" in the relevant section.
  Do NOT estimate, extrapolate, or invent metrics. A fabricated number in a PR description
  is a false claim to maintainers and reviewers.
- CRITICAL — attribute third-party comparison data: if the PR description references
  performance of other hardware, vendors, or systems (e.g. NVIDIA, Intel, reference
  implementations), add a parenthetical or footnote naming the exact source (benchmark
  suite, CSV, date) and stating the numbers are for context only. The PR makes no claims
  about systems other than the one being patched.
- commands_to_run: EXACT commands grounded in this repo's contributing guide. Do NOT
  default to "pre-commit run" if the repo uses a different mechanism (e.g. manual black/ruff,
  a .githooks/ installer, or a Makefile target). Check pr_preparation.lint_commands and
  pr_preparation.pre_commit_run_command. Include ruff/black/clang-format commands with
  the exact version flags the repo specifies.
- submission_instructions: concrete steps — fork the repo on GitHub, add fork remote,
  push the branch, open PR against the correct base branch. Include any DCO sign-off
  (`git commit -s`) or CLA requirement. Mention where to paste the pr_description.
- verification_gaps: identify claims that need further evidence

Return ONLY the JSON object.
"""


def _load_repo_config(slug: str) -> dict:
    config_path = GOLD / slug / "repo_config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def prepare_pr(
    repo: str,
    diff: str,
    *,
    blurb: str = "",
    judge_findings: list[dict] | None = None,
    test_scripts: list[dict] | None = None,
    benchmark_results: list[dict] | None = None,
    series_pr_urls: dict[int, str] | None = None,
    parent_issue_url: str = "",
    sibling_titles: list[str] | None = None,
    model: str = "claude-opus-4-7",
) -> dict:
    """Generate a PR preparation package for a diff.

    Args:
        repo:              owner/name slug (e.g. "vllm-project/vllm")
        diff:              unified diff text
        blurb:             optional short description of what the PR does
        judge_findings:    optional list of judge findings (from judge_patch)
        test_scripts:      optional list of test scripts (from suggest_tests)
        benchmark_results: optional list of measured benchmark dicts, each with:
                           - phase: "before" | "after" | "comparison"
                           - hardware: str (e.g. "MI355X gfx950 256CUs")
                           - config: str (e.g. "E=256 H=7168 TOP_K=8")
                           - rows: list of {label, latency_ms, throughput, notes}
                           Before-PR numbers are especially valuable for kernel PRs
                           where the baseline path may be broken or suboptimal.
        series_pr_urls:    optional {pr_index: github_url} mapping — after the LLM
                           generates the description, any "PR N" references in the
                           description are replaced with live links to the real PRs.
                           Pass this after PRs are opened (second-pass update).
        model:             LiteLLM model name

    Returns:
        dict with keys: contributing_checklist, commit_message, pr_description,
        commands_to_run, verification_gaps
    """
    from pipeline.llm import llm_call, make_client, parse_json

    slug = repo.replace("/", "_", 1)
    repo_config = _load_repo_config(slug)
    pr_prep = repo_config.get("pr_preparation", {})

    # Format pr_preparation section for the prompt
    if pr_prep:
        pr_prep_section = yaml.dump(pr_prep, default_flow_style=False)
    else:
        pr_prep_section = (
            "(No pr_preparation section found in repo_config.yaml. "
            "Use general best practices for open-source contribution.)"
        )

    # Pull out the raw PR template if available — used verbatim in the prompt
    pr_template_raw = pr_prep.get("pr_template_raw", "")

    # Resolve {changed_files} placeholder in pr_preparation commands using actual diff paths
    changed_files = " ".join(
        line.split()[-1].lstrip("b/")
        for line in diff.splitlines()
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null")
    )
    if changed_files and pr_prep:
        import copy
        pr_prep = copy.deepcopy(pr_prep)
        for key in ("lint_commands", "test_commands"):
            pr_prep[key] = [
                cmd.replace("{changed_files}", changed_files)
                for cmd in pr_prep.get(key, [])
            ]
        if pr_prep.get("pre_commit_run_command"):
            pr_prep["pre_commit_run_command"] = pr_prep["pre_commit_run_command"].replace(
                "{changed_files}", changed_files
            )
        pr_prep_section = yaml.dump(pr_prep, default_flow_style=False)

    # Blurb section
    blurb_section = f"PR BLURB: {blurb}\n\n" if blurb else ""
    if parent_issue_url:
        blurb_section += f"PARENT ISSUE (include 'Part of {parent_issue_url}' near the top of pr_description): {parent_issue_url}\n\n"

    # Judge findings section
    judge_section = ""
    if judge_findings:
        fails = [f for f in judge_findings if f.get("result") == "fail"]
        if fails:
            lines = ["KNOWN VIOLATIONS (from judge — address these in the PR prep):\n"]
            for f in fails[:10]:
                lines.append(f"- [{f.get('severity', '?')}] {f.get('rule_text', '')[:100]}")
                if f.get("fix_hint"):
                    lines.append(f"  Fix: {f['fix_hint'][:120]}")
            judge_section = "\n".join(lines) + "\n\n"

    # Test scripts section
    test_section = ""
    if test_scripts:
        lines = ["SUGGESTED TEST SCRIPTS (include placeholders in PR description):\n"]
        for s in test_scripts[:4]:
            lines.append(f"- {s.get('name', '')} ({s.get('test_type', '')}): {s.get('description', '')[:100]}")
            if s.get("what_to_report"):
                lines.append(f"  Report: {s['what_to_report'][:120]}")
        test_section = "\n".join(lines) + "\n\n"

    # Benchmark results section (real measured numbers, before and after)
    benchmark_section = ""
    if benchmark_results:
        lines = ["BENCHMARK RESULTS (real measured numbers — use these in the PR description):\n"]
        for r in benchmark_results:
            phase = r.get("phase", "")       # "before" | "after" | "comparison"
            config = r.get("config", "")
            hardware = r.get("hardware", "")
            rows = r.get("rows", [])          # list of {label, latency_ms, throughput, notes}
            if config or hardware:
                lines.append(f"\n{phase.upper()} — {hardware} — {config}")
            for row in rows:
                notes = f"  [{row['notes']}]" if row.get("notes") else ""
                lines.append(
                    f"  {row.get('label','')}: {row.get('latency_ms','?')} ms  "
                    f"{row.get('throughput','?')}{notes}"
                )
        benchmark_section = "\n".join(lines) + "\n\n"

    diff_truncated = diff[:_DIFF_CAP]

    pr_template_section = (
        f"PR TEMPLATE (reproduce this structure exactly in pr_description):\n{pr_template_raw}\n\n"
        if pr_template_raw else ""
    )

    # Sibling titles section — ensures consistent conventional prefix across PR series
    sibling_section = ""
    if sibling_titles:
        sibling_section = (
            "SIBLING PR TITLES (already finalized for other PRs in this series):\n"
            + "\n".join(f"  - {t}" for t in sibling_titles)
            + "\n\nUse the same conventional prefix (e.g. perf:, feat:, fix:) as the sibling "
            "PRs unless this PR's change type is fundamentally different.\n\n"
        )

    prompt = _PREPARE_PROMPT.format(
        repo=repo,
        diff_truncated=diff_truncated,
        diff_cap=_DIFF_CAP,
        blurb_section=blurb_section,
        pr_template_section=pr_template_section,
        pr_prep_section=pr_prep_section,
        judge_section=judge_section,
        test_section=test_section,
        benchmark_section=benchmark_section + sibling_section,
    )

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=16384, json_mode=True)
    result = parse_json(raw)

    if not isinstance(result, dict):
        raise ValueError(f"LLM returned non-dict: {type(result)}")

    # Ensure all expected keys exist
    result.setdefault("pr_title", "")
    result.setdefault("contributing_checklist", [])
    result.setdefault("commit_message", "")
    result.setdefault("pr_description", "")
    result.setdefault("commands_to_run", [])
    result.setdefault("submission_instructions", "")

    # Pass test_scripts through so callers (get_plan) can surface them to the IDE agent.
    if test_scripts:
        result["test_scripts"] = test_scripts

    # Override commands_to_run with verbatim lint/test commands from repo_config.
    # The LLM should not guess these — they are repo-specific and version-pinned.
    # pr_prep already has {changed_files} resolved at this point.
    repo_commands: list[str] = []
    if pr_prep.get("pre_commit_run_command"):
        repo_commands.append(pr_prep["pre_commit_run_command"])
    elif pr_prep.get("lint_commands"):
        repo_commands.extend(pr_prep["lint_commands"])
    if pr_prep.get("test_commands"):
        repo_commands.extend(pr_prep["test_commands"])

    if repo_commands:
        # Prepend repo-known commands; keep any LLM-generated ones that aren't duplicates
        llm_cmds = [c for c in result["commands_to_run"] if c not in repo_commands]
        result["commands_to_run"] = repo_commands + llm_cmds

    # Strip any "## Checklist"-variant section from pr_description — checklist items belong
    # in contributing_checklist JSON only. Applied defensively regardless of LLM output.
    _desc_raw = result.get("pr_description", "")
    if _desc_raw:
        import re as _re_checklist
        _desc_stripped = _re_checklist.sub(
            r"\n##\s+(?:Contributing\s+|PR\s+|Review\s+)?Checklist\b[\s\S]*?(?=\n##\s|\Z)",
            "",
            _desc_raw,
            flags=_re_checklist.IGNORECASE,
        ).rstrip()
        if _desc_stripped != _desc_raw:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Stripped '## Checklist' section from pr_description (belongs in contributing_checklist only)"
            )
        result["pr_description"] = _desc_stripped

    # Inject parent issue reference at the top of the PR description if not already there.
    if parent_issue_url and parent_issue_url not in result.get("pr_description", ""):
        import re as _re
        issue_ref = f"Part of {parent_issue_url}\n\n"
        result["pr_description"] = issue_ref + result.get("pr_description", "")

    # Second-pass: replace "PR N" placeholders with real GitHub PR links.
    # This runs immediately if series_pr_urls is given (caller already has the URLs),
    # or can be called as a post-open pass after PRs are created on GitHub.
    if series_pr_urls:
        import re as _re
        desc = result.get("pr_description", "")
        for pr_idx, url in series_pr_urls.items():
            pr_num = url.rstrip("/").rsplit("/", 1)[-1]
            # Replace "PR {idx}" with linked "#num" — preserve surrounding context
            desc = _re.sub(
                rf"\bPR {pr_idx}\b(?!\d)",
                f"[#{pr_num}]({url})",
                desc,
            )
        result["pr_description"] = desc

    return result


def main():
    import argparse
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Generate PR preparation package")
    p.add_argument("--repo", required=True, help="owner/name (e.g. vllm-project/vllm)")
    p.add_argument("--patch", required=True, help="Path to unified diff file")
    p.add_argument("--blurb", default="", help="Short description of what the PR does")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--out", default=None, help="Write JSON output to this file")
    args = p.parse_args()

    diff = Path(args.patch).read_text()
    result = prepare_pr(args.repo, diff, blurb=args.blurb, model=args.model)

    print("\n=== COMMIT MESSAGE ===")
    print(result["commit_message"])

    print("\n=== CONTRIBUTING CHECKLIST ===")
    for item in result["contributing_checklist"]:
        check = "[x]" if item.get("required") else "[ ]"
        cmd = f"  →  {item['command']}" if item.get("command") else ""
        print(f"  {check} {item['item']}{cmd}")

    print("\n=== COMMANDS TO RUN ===")
    for cmd in result["commands_to_run"]:
        print(f"  $ {cmd}")

    print("\n=== PR DESCRIPTION ===")
    print(result["pr_description"])

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
