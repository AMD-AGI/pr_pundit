"""
Iterative patch fixer — judge-in-the-loop code generator.

Uses DSPy RLM (Recursive Language Model) to rewrite a patch until it
satisfies all distilled rules, with the judge callable as a tool
mid-loop so the agent can self-check before committing to a final diff.

Optionally runs real tests/benchmarks inside the loop if repo_path is
provided — patches are applied to an isolated git worktree, the command
runs, then the worktree is destroyed. No risk to the main checkout.

Usage:
    python -m pipeline.fix --repo owner_name --patch diff.patch
    python -m pipeline.fix --repo owner_name --patch diff.patch \\
        --repo-path /path/to/vllm --comments "Add benchmark"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_OUTER_TRIES = 4
MAX_RLM_ITERS = 30
BUDGET_WARN_1 = int(MAX_RLM_ITERS * 0.70)   # warn at 70%
BUDGET_WARN_2 = int(MAX_RLM_ITERS * 0.90)   # warn harder at 90%
TEST_TIMEOUT_SEC = 300


# ── diff validity ────────────────────────────────────────────────────

def _is_valid_diff(patch: str) -> tuple[bool, str]:
    """Return (valid, reason). Checks for minimum unified diff structure."""
    if not patch or not patch.strip():
        return False, "patch is empty"
    lines = patch.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    if not has_hunk:
        return False, "no hunk headers (@@ ... @@) found — not a valid unified diff"
    has_content = any(line.startswith(("+", "-")) and not line.startswith(("+++", "---")) for line in lines)
    if not has_content:
        return False, "no added/removed lines found in diff"
    return True, "ok"


# ── DSPy setup ───────────────────────────────────────────────────────

def _make_dspy_lm(model: str):
    import dspy

    base_url = os.environ["LITELLM_BASE_URL"]
    api_key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    return dspy.LM(
        dspy_model,
        api_base=base_url,
        api_key=api_key,
        cache=False,
        timeout=300,
        max_tokens=16384,
    )


# ── DSPy signatures ──────────────────────────────────────────────────

def _make_signatures(has_test_runner: bool = False):
    import dspy

    test_tool_doc = ""
    if has_test_runner:
        test_tool_doc = """\
          run_test(patch_text, cmd) -> str
            Apply patch_text to an isolated repo checkout and run cmd (e.g. pytest,
            a benchmark script). Returns stdout/stderr and exit code.
            Use to verify correctness or measure performance before submitting.
          run_shell_cmd(cmd) -> str
            Run a shell command without applying the patch (for linting, syntax
            checks, inspecting the environment, etc.).
        """

    class FixPlanSignature(dspy.Signature):
        """Plan how to fix a code patch to satisfy rules and reviewer comments.

        Read the patch, findings, and reviewer comments carefully.
        Identify exactly what needs to change and in which files/lines.
        If the correct API or syntax is unclear, note that the executor should
        call web_search() to look it up before writing the fix.
        Do NOT write code yet — produce a precise, ordered plan.
        """

        patch: str = dspy.InputField(desc="The original unified diff being reviewed")
        findings: str = dspy.InputField(
            desc="JSON list of rule violations from the judge, each with rule_text, "
                 "violation, fix_hint, file, line_start, line_end"
        )
        reviewer_comments: str = dspy.InputField(
            desc="Additional free-form reviewer comments to address (may be empty)"
        )
        fix_history: str = dspy.InputField(
            desc="JSON list of previous fix attempts and why they still failed. "
                 "Empty on first attempt."
        )
        has_test_runner: str = dspy.InputField(
            desc="'yes' if run_test() tool is available for running real tests, 'no' otherwise"
        )

        failure_summary: str = dspy.OutputField(desc="2-3 sentences: what is wrong and why")
        fix_plan: str = dspy.OutputField(
            desc="Numbered, ordered list of concrete changes. Each item: which file/section, "
                 "what to change, why it fixes the violation. "
                 "If has_test_runner=yes, include test commands to verify each change."
        )
        risk_notes: str = dspy.OutputField(
            desc="Brief note on anything that could break if changed naively"
        )

    executor_doc = f"""\
Rewrite a code patch to satisfy all rule violations and reviewer comments.

Available tools — call them as Python in this REPL:
  apply_search_replace(patch, search, replace) -> str
    Edit the patch text. Returns the modified patch or an error if search not found.
  show_patch(patch) -> str
    Print the current patch for inspection (first 100 lines).
  run_judge(patch_text) -> str
    Run the rule judge on patch_text. Returns JSON findings.
    You MUST call this at least once before submitting to verify your fix.
  web_search(query) -> str
    Search the web for documentation, API references, or error explanations.
    Use when you need to look up correct syntax, library APIs, or best practices
    before writing a fix (e.g. "TORCH_CHECK error message format vllm",
    "pre-commit ruff config options", "pytest parametrize syntax").
{test_tool_doc}

Strategy:
  1. Read fix_plan carefully.
  2. Apply changes using apply_search_replace().
  3. {"run_test() to verify correctness, then " if has_test_runner else ""}call run_judge() on your modified patch.
  4. If violations remain, fix and re-check.
  5. SUBMIT only after run_judge() confirms no violations (or you have exhausted options).

Rules:
  - The patch must remain a valid unified diff.
  - Never invent lines that weren't in the original diff.
  - Only modify +/- lines, not context lines (lines without a leading + or -).
  - If a search string isn't found, show_patch() to inspect current state first.
"""

    class FixProposalSignature(dspy.Signature):
        patch: str = dspy.InputField(desc="The original unified diff to fix")
        findings: str = dspy.InputField(desc="JSON rule violations to resolve")
        reviewer_comments: str = dspy.InputField(desc="Additional reviewer comments")
        fix_plan: str = dspy.InputField(desc="Ordered plan from the planning stage")
        fix_history: str = dspy.InputField(desc="Previous failed attempts with reasons")

        fixed_patch: str = dspy.OutputField(
            desc="The complete rewritten unified diff. Must be a valid diff. "
                 "Return the original if you could not improve it."
        )
        changes_made: str = dspy.OutputField(
            desc="Bullet list: what you changed and which violation each change addresses"
        )
        confidence: str = dspy.OutputField(desc="high | medium | low")
        unresolved: str = dspy.OutputField(
            desc="Violations you could NOT resolve and why. Empty string if all resolved."
        )
        judge_was_called: str = dspy.OutputField(
            desc="'yes' if you called run_judge() at least once to verify, 'no' otherwise"
        )

    FixProposalSignature.__doc__ = executor_doc
    return FixPlanSignature, FixProposalSignature


# ── tools ────────────────────────────────────────────────────────────

def _build_tools(repo_slug: str, model: str, repo_path: str | None = None) -> list:
    """Build tools for the DSPy RLM agent.

    All tools share a call counter for budget warnings.
    run_judge() tracks whether self-verification has happened.
    run_test() is only included when repo_path is provided.
    """
    state = {"tool_calls": 0, "judge_calls": 0}

    def _budget_suffix() -> str:
        calls = state["tool_calls"]
        if calls >= BUDGET_WARN_2:
            remaining = MAX_RLM_ITERS - calls
            return (
                f"\n\n⚠️ CRITICAL BUDGET WARNING: only ~{remaining} tool calls remain. "
                "Finalize your fix and SUBMIT now."
            )
        if calls >= BUDGET_WARN_1:
            remaining = MAX_RLM_ITERS - calls
            return (
                f"\n\n⚠️ Budget warning: ~{remaining} tool calls remaining. "
                "Start wrapping up — call run_judge() and submit soon."
            )
        return ""

    def run_judge(patch_text: str) -> str:
        """Run the PR Pundit judge on patch_text. Returns JSON findings."""
        state["tool_calls"] += 1
        state["judge_calls"] += 1
        try:
            from pipeline.judge import judge_patch
            result = judge_patch(repo_slug, patch_text, model)
            findings = result.get("findings", [])
            summary = result.get("summary", {})
            out = json.dumps({
                "fail": summary.get("fail", 0),
                "uncertain": summary.get("uncertain", 0),
                "pass": summary.get("pass", 0),
                "findings": [
                    {
                        "rule_text": f["rule_text"],
                        "severity": f["severity"],
                        "result": f["result"],
                        "violation": f["violation"],
                        "fix_hint": f.get("fix_hint", ""),
                        "file": f.get("file", ""),
                        "line_start": f.get("line_start"),
                        "line_end": f.get("line_end"),
                    }
                    for f in findings
                ],
            }, indent=2)
        except Exception as exc:
            out = json.dumps({"error": str(exc)})
        return out + _budget_suffix()

    def apply_search_replace(patch: str, search: str, replace: str) -> str:
        """Apply a search/replace to the patch text. Returns modified patch or error."""
        state["tool_calls"] += 1
        if search not in patch:
            # fuzzy: try ignoring leading whitespace on each line
            search_stripped = search.strip()
            for line in patch.splitlines():
                if line.strip() == search_stripped:
                    search = line
                    break
        if search not in patch:
            return (
                "ERROR: search string not found in patch. "
                "Call show_patch() to inspect the current patch state and check exact whitespace."
            ) + _budget_suffix()
        return patch.replace(search, replace, 1) + _budget_suffix()

    def show_patch(patch: str) -> str:
        """Print the current patch for inspection (first 100 lines)."""
        state["tool_calls"] += 1
        lines = patch.splitlines()
        preview = "\n".join(lines[:100])
        if len(lines) > 100:
            preview += f"\n... ({len(lines) - 100} more lines)"
        return preview + _budget_suffix()

    def web_search(query: str) -> str:
        """Search the web for documentation or API references. Returns top results."""
        state["tool_calls"] += 1
        from pipeline.web_search import web_search as _ws, format_search_results
        results = _ws(query, max_results=5)
        return format_search_results(results) + _budget_suffix()

    tools = [run_judge, apply_search_replace, show_patch, web_search]

    if repo_path:
        rp = Path(repo_path).resolve()

        def run_test(patch_text: str, cmd: str) -> str:
            """Apply patch_text to an isolated git worktree of repo_path and run cmd.

            Returns combined stdout/stderr and exit code. The worktree is always
            destroyed after the command finishes — no persistent side effects.

            Example commands:
              pytest tests/kernels/test_attention.py -x -q
              python benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-hf
            """
            state["tool_calls"] += 1
            if not rp.exists():
                return f"ERROR: repo_path {rp} does not exist" + _budget_suffix()

            # validate the patch before trying to apply it
            valid, reason = _is_valid_diff(patch_text)
            if not valid:
                return f"ERROR: invalid diff — {reason}" + _budget_suffix()

            with tempfile.TemporaryDirectory(prefix="pr-pundit-fix-") as tmpdir:
                worktree = Path(tmpdir) / "worktree"
                try:
                    # create isolated worktree from HEAD
                    add = subprocess.run(
                        ["git", "worktree", "add", "--detach", str(worktree)],
                        cwd=str(rp), capture_output=True, text=True,
                    )
                    if add.returncode != 0:
                        return f"ERROR creating worktree: {add.stderr.strip()}" + _budget_suffix()

                    # write and apply patch
                    patch_file = Path(tmpdir) / "fix.patch"
                    patch_file.write_text(patch_text)
                    apply = subprocess.run(
                        ["git", "apply", "--check", str(patch_file)],
                        cwd=str(worktree), capture_output=True, text=True,
                    )
                    if apply.returncode != 0:
                        return (
                            f"ERROR: patch does not apply cleanly:\n{apply.stderr.strip()}\n"
                            "Fix the diff format before running tests."
                        ) + _budget_suffix()

                    subprocess.run(
                        ["git", "apply", str(patch_file)],
                        cwd=str(worktree), check=True, capture_output=True,
                    )

                    # run the test command
                    proc = subprocess.run(
                        cmd, shell=True, cwd=str(worktree),
                        capture_output=True, text=True, timeout=TEST_TIMEOUT_SEC,
                    )
                    output = (proc.stdout + proc.stderr).strip()
                    # cap output to avoid blowing context
                    if len(output) > 8000:
                        output = output[:4000] + "\n...(truncated)...\n" + output[-4000:]
                    result = f"exit code: {proc.returncode}\n{output}"

                except subprocess.TimeoutExpired:
                    result = f"ERROR: command timed out after {TEST_TIMEOUT_SEC}s"
                except Exception as exc:
                    result = f"ERROR: {exc}"
                finally:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(worktree)],
                        cwd=str(rp), capture_output=True,
                    )

            return result + _budget_suffix()

        def run_shell_cmd(cmd: str) -> str:
            """Run a shell command without applying the patch (linting, env checks, etc.).

            Runs in the original repo directory. Do NOT use this to run tests that
            depend on the patched code — use run_test() for that.
            """
            state["tool_calls"] += 1
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=str(rp),
                    capture_output=True, text=True, timeout=60,
                )
                output = (proc.stdout + proc.stderr).strip()
                if len(output) > 4000:
                    output = output[:4000] + "\n...(truncated)..."
                return f"exit code: {proc.returncode}\n{output}" + _budget_suffix()
            except subprocess.TimeoutExpired:
                return "ERROR: command timed out after 60s" + _budget_suffix()
            except Exception as exc:
                return f"ERROR: {exc}" + _budget_suffix()

        tools += [run_test, run_shell_cmd]

    return tools, state


# ── main fix loop ────────────────────────────────────────────────────

def fix_patch(
    repo_slug: str,
    patch_text: str,
    reviewer_comments: str = "",
    model: str | None = None,
    max_tries: int = MAX_OUTER_TRIES,
    repo_path: str | None = None,
    log_callback=None,
) -> dict:
    """
    Iteratively fix a patch until the judge is satisfied.

    Args:
        repo_slug:         gold data repo slug (owner_name)
        patch_text:        the unified diff to fix
        reviewer_comments: additional free-form reviewer requests
        model:             LiteLLM model name
        max_tries:         outer loop iteration limit
        repo_path:         local path to the repo for running real tests (optional)

    Returns:
        fixed_patch, final_findings, attempts, history, success
    """
    import dspy

    model = model or DEFAULT_MODEL
    _dspy_lm = _make_dspy_lm(model)

    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    has_test_runner = bool(repo_path)
    FixPlanSignature, FixProposalSignature = _make_signatures(has_test_runner)
    tools, tool_state = _build_tools(repo_slug, model, repo_path)

    from pipeline.judge import judge_patch

    history: list[dict] = []
    current_patch = patch_text
    best_patch = patch_text
    best_finding_count = None

    for attempt in range(1, max_tries + 1):
        _log(f"── Attempt {attempt} / {max_tries} ──")

        valid, reason = _is_valid_diff(current_patch)
        if not valid:
            _log(f"  Patch invalid ({reason}) — reverting to last good patch")
            current_patch = best_patch
            history.append({"attempt": attempt, "outcome": "invalid_diff", "reason": reason})
            continue

        _log("  Running judge...")
        judge_result = judge_patch(repo_slug, current_patch, model)
        findings = judge_result.get("findings", [])
        summary = judge_result.get("summary", {})
        finding_count = summary.get("fail", 0) + summary.get("uncertain", 0)

        _log(
            f"  Judge: {summary.get('fail', 0)} fail, "
            f"{summary.get('uncertain', 0)} uncertain, "
            f"{summary.get('pass', 0)} pass"
        )

        if best_finding_count is None or finding_count < best_finding_count:
            best_finding_count = finding_count
            best_patch = current_patch

        if not findings:
            _log(f"  Judge satisfied — done in {attempt} attempt(s).")
            return {
                "fixed_patch": current_patch,
                "final_findings": [],
                "attempts": attempt,
                "history": history,
                "success": True,
            }

        findings_json = json.dumps(findings, indent=2)
        fix_history_json = json.dumps(history, indent=2) if history else "[]"

        tool_state["tool_calls"] = 0
        tool_state["judge_calls"] = 0

        _log(f"  Planning fix for {len(findings)} violations...")
        planner = dspy.ChainOfThought(FixPlanSignature)
        with dspy.context(lm=_dspy_lm):
            plan = planner(
                patch=current_patch,
                findings=findings_json,
                reviewer_comments=reviewer_comments or "(none)",
                fix_history=fix_history_json,
                has_test_runner="yes" if has_test_runner else "no",
            )
        _log(f"  Plan: {plan.fix_plan[:120]}...")

        _log(f"  RLM agent running (up to {MAX_RLM_ITERS} tool calls)...")
        rlm = dspy.RLM(FixProposalSignature, tools=tools, max_iterations=MAX_RLM_ITERS)
        with dspy.context(lm=_dspy_lm):
            result = rlm(
                patch=current_patch,
                findings=findings_json,
                reviewer_comments=reviewer_comments or "(none)",
                fix_plan=plan.fix_plan,
                fix_history=fix_history_json,
            )
        _log(f"  RLM done — {tool_state['tool_calls']} tool calls, "
             f"{tool_state['judge_calls']} judge calls")

        if tool_state["judge_calls"] == 0:
            _log("  WARNING: agent did not call run_judge() — fix unverified")

        fixed_patch = (result.fixed_patch or "").strip()

        # validate agent output before using it
        if not fixed_patch:
            _log(f"  Agent returned empty patch — stopping")
            history.append({
                "attempt": attempt, "findings_count": finding_count,
                "outcome": "empty_output",
                "judge_called": tool_state["judge_calls"] > 0,
            })
            break

        valid, reason = _is_valid_diff(fixed_patch)
        if not valid:
            _log(f"  Agent produced invalid diff ({reason}) — discarding, retrying")
            history.append({
                "attempt": attempt, "findings_count": finding_count,
                "outcome": "invalid_diff_output", "reason": reason,
                "judge_called": tool_state["judge_calls"] > 0,
            })
            continue

        if fixed_patch == current_patch:
            _log("  Agent returned unchanged patch — stopping")
            history.append({
                "attempt": attempt, "findings_count": finding_count,
                "outcome": "no_change",
                "changes_made": getattr(result, "changes_made", ""),
                "unresolved": getattr(result, "unresolved", ""),
                "judge_called": tool_state["judge_calls"] > 0,
            })
            break

        history.append({
            "attempt": attempt,
            "findings_count": finding_count,
            "outcome": "changed",
            "changes_made": getattr(result, "changes_made", ""),
            "confidence": getattr(result, "confidence", ""),
            "unresolved": getattr(result, "unresolved", ""),
            "judge_called": tool_state["judge_calls"] > 0,
        })
        current_patch = fixed_patch

    # final judge pass on best patch
    final_result = judge_patch(repo_slug, best_patch, model)
    final_findings = final_result.get("findings", [])

    return {
        "fixed_patch": best_patch,
        "final_findings": final_findings,
        "attempts": len(history),
        "history": history,
        "success": len(final_findings) == 0,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Iteratively fix a patch using the judge")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    p.add_argument("--patch", required=True, help="path to .patch or .diff file")
    p.add_argument("--comments", default="", help="additional reviewer comments")
    p.add_argument("--model", default=None)
    p.add_argument("--max-tries", type=int, default=MAX_OUTER_TRIES)
    p.add_argument("--repo-path", default=None,
                   help="local path to the repo for running real tests inside the loop")
    p.add_argument("--out", default=None, help="write fixed patch to file")
    args = p.parse_args()

    patch_text = Path(args.patch).read_text()
    result = fix_patch(
        args.repo, patch_text, args.comments, args.model,
        args.max_tries, args.repo_path,
    )

    print(f"\n{'='*60}")
    print(f"FIX RESULT: {'SUCCESS' if result['success'] else 'PARTIAL'}")
    print(f"  Attempts: {result['attempts']}")
    print(f"  Remaining violations: {len(result['final_findings'])}")
    for f in result["final_findings"]:
        print(f"  [{f['severity']}] {f['rule_text'][:70]}")
        print(f"    {f['violation'][:100]}")
    print(f"{'='*60}\n")

    if args.out:
        Path(args.out).write_text(result["fixed_patch"])
        logger.info("Fixed patch written to %s", args.out)
    else:
        print(result["fixed_patch"])


if __name__ == "__main__":
    main()
