"""
Structured code reviewer — evaluate a patch against distilled rules.

Produces machine-readable findings consumable by both humans and code
generator bots.  Each finding includes exact file, line range, violation
description, fix hint, and supporting examples.

Usage:
    python -m pipeline.judge --repo owner_name --patch diff.patch
    python -m pipeline.judge --repo owner_name --patch diff.patch --model claude-sonnet-4-6 --workers 10
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.llm import llm_call, make_client, parse_json

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

DEFAULT_MODEL = "claude-opus-4-7"


# ── diff parser ─────────────────────────────────────────────────────

def _parse_diff(patch_text: str) -> list[dict]:
    """Parse unified diff into per-file structures with hunks."""
    files: list[dict] = []
    current_file: dict | None = None
    current_hunk: dict | None = None

    for line in patch_text.splitlines(keepends=True):
        m = re.match(r"^diff --git a/.+ b/(.+)$", line)
        if m:
            if current_hunk and current_file:
                current_file["hunks"].append(current_hunk)
            if current_file:
                files.append(current_file)
            current_file = {"path": m.group(1), "hunks": []}
            current_hunk = None
            continue

        if line.startswith("+++ b/") and current_file:
            current_file["path"] = line[6:].rstrip()
            continue

        m = re.match(
            r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)", line
        )
        if m:
            if current_hunk and current_file:
                current_file["hunks"].append(current_hunk)
            current_hunk = {
                "header": line.rstrip(),
                "old_start": int(m.group(1)),
                "old_lines": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_lines": int(m.group(4) or 1),
                "body": "",
                "added_lines": [],
            }
            continue

        if current_hunk is not None:
            current_hunk["body"] += line
            if line.startswith("+") and not line.startswith("+++"):
                lineno = current_hunk["new_start"] + current_hunk["body"].count("\n") - 1
                current_hunk["added_lines"].append((lineno, line[1:].rstrip()))

    if current_hunk and current_file:
        current_file["hunks"].append(current_hunk)
    if current_file:
        files.append(current_file)

    return files


# ── rule loading & matching ─────────────────────────────────────────

EXT_MAP = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "go": ".go", "rust": ".rs", "java": ".java", "cpp": ".cpp",
    "c": ".c", "ruby": ".rb", "kotlin": ".kt", "swift": ".swift",
}


def _load_rules(repo_slug: str, include_pending: bool = False) -> list[dict]:
    rules_path = GOLD / repo_slug / "rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"No rules at {rules_path}. Run distill first.")
    rules = json.loads(rules_path.read_text())
    allowed = {"accepted", "pending"} if include_pending else {"accepted"}
    return [r for r in rules if r.get("human_review_status") in allowed]


def _match_rule(rule: dict, changed_paths: list[str]) -> bool:
    """Return True if the rule applies to any of the changed paths.

    Uses AND logic across constraint types: a file must satisfy ALL of
    path_prefixes AND languages AND file_patterns that are specified.
    This prevents a rule scoped to vllm/attention/ from matching
    vllm/config.py just because config.py is also a Python file.

    If a constraint type is absent it is not required (no constraint = any).
    """
    scope = rule.get("scope", {}) or {}
    path_prefixes = scope.get("path_prefixes") or []
    languages = scope.get("languages") or []
    file_patterns = scope.get("file_patterns") or []

    # no constraints at all → repo-wide
    if not path_prefixes and not languages and not file_patterns:
        return True

    for p in changed_paths:
        path_ok = any(p.startswith(prefix) for prefix in path_prefixes) if path_prefixes else True
        lang_ok = any(p.endswith(EXT_MAP.get(lang, f".{lang}")) for lang in languages) if languages else True
        pat_ok = any(
            (stem := pat.replace("*", "").replace("?", "")) and stem in p
            for pat in file_patterns
        ) if file_patterns else True

        if path_ok and lang_ok and pat_ok:
            return True

    return False


# ── evaluators ──────────────────────────────────────────────────────

def _eval_deterministic(rule: dict, diff_files: list[dict]) -> dict | None:
    """Run regex/pattern check. Returns a finding dict or None (pass)."""
    verifier = rule.get("verifier", {})
    pattern = verifier.get("pattern")
    if not pattern:
        return None

    patch_text = "\n".join(
        h["body"] for f in diff_files for h in f["hunks"]
    )

    try:
        matches = re.findall(pattern, patch_text)
    except re.error:
        return None

    directive = rule.get("directive_type", "")

    if "forbidden" in directive and matches:
        first_file = diff_files[0]["path"] if diff_files else "unknown"
        return {
            "rule_id": rule.get("rule_id", ""),
            "rule_text": rule.get("rule_text", ""),
            "directive_type": directive,
            "severity": rule.get("severity", "advisory"),
            "result": "fail",
            "confidence": 1.0,
            "file": first_file,
            "line_start": None,
            "line_end": None,
            "violation": f"Forbidden pattern found: {matches[:3]}",
            "fix_hint": rule.get("rationale", "Remove the forbidden pattern"),
            "examples": _get_examples(rule),
            "rationale": rule.get("rationale", ""),
        }

    # For required/preferred patterns: absence from the diff is not a violation.
    # The diff may simply not touch the relevant code. Only forbidden-pattern
    # violations are detectable by pure regex on the diff.
    return None


JUDGE_PROMPT = """\
You are a code reviewer enforcing a project-specific merge rule.

Rule: {rule_text}
Severity: {severity}
Rationale: {rationale}

{examples_section}

Diff to review (what changed):
```
{diff_text}
```
{file_context_section}
Evaluate whether this diff violates the rule. Use the full file context (if provided) to judge whether the rule is actually applicable to the code being changed — return result="pass" if the rule does not apply to what this diff is doing.

Respond with your answer inside <answer> tags. The content must be a single JSON object with these fields:
- result: "pass", "fail", or "uncertain"
- confidence: float 0.0–1.0
- violation: one sentence describing the specific violation (empty string if pass)
- fix_hint: one sentence on how to fix it (empty string if pass)
- file: the file where the violation occurs (empty string if pass)
- line_start: integer line number or null
- line_end: integer line number or null
- violating_snippet: 5–15 lines of plain code context around the violation, with +/- markers stripped (empty string if pass)

Example:
<answer>{{"result":"fail","confidence":0.9,"violation":"","fix_hint":"","file":"","line_start":null,"line_end":null,"violating_snippet":""}}</answer>
"""


def _format_examples(rule: dict) -> str:
    sections = []
    pos = rule.get("positive_examples") or []
    neg = rule.get("negative_examples") or []

    if pos:
        sections.append("Positive examples (code that follows this rule):")
        for ex in pos[:2]:
            desc = ex.get("description", "")
            before = ex.get("code_before", "")
            after = ex.get("code_after", "")
            entry = f"  - {desc}"
            if before:
                entry += f"\n    Before: {before[:300]}"
            if after:
                entry += f"\n    After:  {after[:300]}"
            sections.append(entry)

    if neg:
        sections.append("Negative examples (code that violates this rule):")
        for ex in neg[:2]:
            desc = ex.get("description", "")
            before = ex.get("code_before", "")
            entry = f"  - {desc}"
            if before:
                entry += f"\n    Code: {before[:300]}"
            sections.append(entry)

    return "\n".join(sections) if sections else "No examples available."


def _get_examples(rule: dict) -> list[dict]:
    examples = []
    for ex in (rule.get("positive_examples") or [])[:2]:
        examples.append({
            "description": ex.get("description", ""),
            "code_before": ex.get("code_before"),
            "code_after": ex.get("code_after"),
        })
    for ex in (rule.get("negative_examples") or [])[:2]:
        examples.append({
            "description": ex.get("description", ""),
            "code_before": ex.get("code_before"),
            "code_after": ex.get("code_after"),
        })
    return examples


_FILE_CONTENT_CAP = 40_000   # chars per file in full-context mode


def _build_file_context_section(diff_files: list[dict], repo_root: Path | None) -> str:
    if not repo_root:
        return ""
    parts = []
    for f in diff_files:
        fp = repo_root / f["path"]
        if not fp.exists():
            continue
        content = fp.read_text(errors="replace")
        if len(content) > _FILE_CONTENT_CAP:
            content = content[:_FILE_CONTENT_CAP] + "\n... (truncated)"
        parts.append(f"=== {f['path']} ===\n{content}")
    if not parts:
        return ""
    return "\nFull file context (use this to judge whether the rule applies to what changed):\n" + "\n\n".join(parts) + "\n"


def _eval_llm_judge(
    rule: dict, diff_files: list[dict], client, model: str,
    repo_root: Path | None = None,
) -> dict | None:
    """Use LLM to evaluate a rule against the diff. Returns finding or None."""
    diff_text = "\n".join(
        f"--- {f['path']} ---\n" + "".join(h["body"] for h in f["hunks"])
        for f in diff_files
    )
    # cap diff size to model context limit (per-rule diffs are already scoped to matching files)
    if len(diff_text) > 600_000:
        diff_text = diff_text[:600_000] + "\n... (truncated)"

    prompt = JUDGE_PROMPT.format(
        rule_text=rule.get("rule_text", ""),
        severity=rule.get("severity", "advisory"),
        rationale=rule.get("rationale", ""),
        examples_section=_format_examples(rule),
        diff_text=diff_text,
        file_context_section=_build_file_context_section(diff_files, repo_root),
    )

    raw = llm_call(prompt, model, client=client, max_tokens=1024)
    if not raw.strip():
        logger.debug("Empty LLM response for rule %s", rule.get("rule_id", "?")[:12])
        return None

    # prefer content inside <answer>...</answer> tags; fall back to full text
    import re as _re
    tag_match = _re.search(r"<answer>(.*?)</answer>", raw, _re.DOTALL)
    json_text = tag_match.group(1).strip() if tag_match else raw

    try:
        result = parse_json(json_text)
    except Exception:
        logger.warning("Unparseable LLM response for rule %s: %.300s", rule.get("rule_id", "?")[:12], raw)
        raise

    if result.get("result") == "pass":
        return None

    return {
        "rule_id": rule.get("rule_id", ""),
        "rule_text": rule.get("rule_text", ""),
        "directive_type": rule.get("directive_type", ""),
        "severity": rule.get("severity", "advisory"),
        "result": result.get("result", "uncertain"),
        "confidence": result.get("confidence", 0.5),
        "file": result.get("file", diff_files[0]["path"] if diff_files else "unknown"),
        "line_start": result.get("line_start"),
        "line_end": result.get("line_end"),
        "violating_snippet": result.get("violating_snippet", ""),
        "violation": result.get("violation", ""),
        "fix_hint": result.get("fix_hint", ""),
        "examples": _get_examples(rule),
        "rationale": rule.get("rationale", ""),
    }


# ── holistic verification pass ──────────────────────────────────────

_VERIFY_PROMPT = """\
You are a senior code reviewer doing a final sanity check on a set of rule violations \
flagged by an automated checker.

The diff being reviewed:
```
{diff_text}
```

Applicable rules for this diff:
{rules_summary}

Violations flagged so far:
{findings_json}

Your tasks:
1. For each flagged violation, decide if it is a genuine violation given the full diff context.
   A finding should be DROPPED if the rule clearly does not apply to what this diff is actually \
doing (e.g. the rule is about a code pattern that wasn't changed, or the flagged line is in a \
comment/test that the rule exempts).
2. Identify any clear violations that were MISSED — a rule in the applicable list that is \
obviously violated but not already flagged. Only add a finding if you are confident (>0.7).

Respond with a single JSON object inside <answer> tags:
{{
  "drop_rule_ids": ["rule_id_1", ...],
  "additional_findings": [
    {{
      "rule_id": "...",
      "rule_text": "...",
      "severity": "blocker|strong|advisory",
      "directive_type": "...",
      "result": "fail",
      "confidence": 0.0,
      "file": "...",
      "line_start": null,
      "line_end": null,
      "violation": "...",
      "fix_hint": "...",
      "violating_snippet": ""
    }}
  ]
}}

If nothing to drop and nothing to add, return {{"drop_rule_ids": [], "additional_findings": []}}.
"""

_FINDINGS_CAP = 30     # don't send more than this many findings to the verifier
_DIFF_CAP = 600_000   # chars — model context limit; per-rule eval is scoped so this mainly affects verification pass


def _verify_findings(
    findings: list[dict],
    patch_text: str,
    scoped_rules: list[dict],
    model: str,
    client,
) -> list[dict]:
    """Holistic pass: drop false positives and surface obvious missed violations."""
    diff_text = patch_text[:_DIFF_CAP] + ("\n... (truncated)" if len(patch_text) > _DIFF_CAP else "")

    # summarise applicable rules (rule_id + rule_text only, to stay compact)
    rules_summary = "\n".join(
        f"- [{r.get('severity','?')}] {r.get('rule_id','')[:8]}: {r.get('rule_text','')[:120]}"
        for r in scoped_rules
    )

    # send at most _FINDINGS_CAP findings; keep the highest-severity ones
    severity_order = {"blocker": 0, "strong": 1, "advisory": 2}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f["severity"], 9))
    capped = sorted_findings[:_FINDINGS_CAP]
    findings_json = json.dumps(
        [{"rule_id": f["rule_id"], "rule_text": f["rule_text"], "severity": f["severity"],
          "result": f["result"], "confidence": f["confidence"],
          "file": f["file"], "line_start": f.get("line_start"), "violation": f["violation"]}
         for f in capped],
        indent=2,
    )

    prompt = _VERIFY_PROMPT.format(
        diff_text=diff_text,
        rules_summary=rules_summary,
        findings_json=findings_json,
    )

    try:
        raw = llm_call(prompt, model, client=client, max_tokens=2048)
    except Exception as exc:
        logger.warning("Verification pass failed: %s — keeping original findings", exc)
        return findings

    import re as _re
    tag_match = _re.search(r"<answer>(.*?)</answer>", raw, _re.DOTALL)
    json_text = tag_match.group(1).strip() if tag_match else raw

    try:
        verdict = parse_json(json_text)
    except Exception:
        logger.warning("Verification pass returned unparseable JSON — keeping original findings")
        return findings

    drop_ids = set(verdict.get("drop_rule_ids") or [])
    if drop_ids:
        logger.info("Verification pass dropped %d finding(s): %s", len(drop_ids), drop_ids)
    kept = [f for f in findings if f["rule_id"] not in drop_ids]

    # build a lookup for additional findings that reference existing scoped rules
    rule_by_id = {r.get("rule_id", ""): r for r in scoped_rules}
    for af in verdict.get("additional_findings") or []:
        rid = af.get("rule_id", "")
        rule = rule_by_id.get(rid, {})
        kept.append({
            "rule_id": rid,
            "rule_text": af.get("rule_text") or rule.get("rule_text", ""),
            "directive_type": af.get("directive_type") or rule.get("directive_type", ""),
            "severity": af.get("severity") or rule.get("severity", "advisory"),
            "result": af.get("result", "uncertain"),
            "confidence": af.get("confidence", 0.7),
            "file": af.get("file", ""),
            "line_start": af.get("line_start"),
            "line_end": af.get("line_end"),
            "violating_snippet": af.get("violating_snippet", ""),
            "violation": af.get("violation", ""),
            "fix_hint": af.get("fix_hint", ""),
            "examples": _get_examples(rule),
            "rationale": rule.get("rationale", ""),
        })
        logger.info("Verification pass added finding: %s — %s", rid[:8], af.get("violation", "")[:80])

    return kept


# ── orchestrator ────────────────────────────────────────────────────

def judge_patch(
    repo_slug: str,
    patch_text: str,
    model: str | None = None,
    workers: int = 10,
    include_pending: bool = False,
    repo_root: str | Path | None = None,
) -> dict:
    """Evaluate a patch against all applicable rules.

    Returns a dict with:
      - summary: {total_rules_checked, pass, fail, uncertain}
      - findings: list of structured violation dicts
      - scoped_rules: all rules that matched (for generator context)
    """
    model = model or DEFAULT_MODEL
    diff_files = _parse_diff(patch_text)
    changed_paths = [f["path"] for f in diff_files]

    if not changed_paths:
        return {
            "summary": {"total_rules_checked": 0, "pass": 0, "fail": 0, "uncertain": 0},
            "findings": [],
            "scoped_rules": [],
        }

    rules = _load_rules(repo_slug, include_pending=include_pending)
    matched = [r for r in rules if _match_rule(r, changed_paths)]

    logger.info(
        "Judge: %d rules loaded, %d matched for %d changed files",
        len(rules), len(matched), len(changed_paths),
    )

    findings: list[dict] = []
    results_counter = {"pass": 0, "fail": 0, "uncertain": 0}

    # split into deterministic vs LLM rules
    deterministic_rules = []
    llm_rules = []
    for r in matched:
        vtype = (r.get("verifier") or {}).get("verifier_type", "")
        if vtype in ("regex", "ast", "metadata"):
            deterministic_rules.append(r)
        else:
            llm_rules.append(r)

    # run deterministic checks (fast, no concurrency needed)
    for r in deterministic_rules:
        applicable_files = [f for f in diff_files if _match_rule(r, [f["path"]])]
        if not applicable_files:
            results_counter["pass"] += 1
            continue
        finding = _eval_deterministic(r, applicable_files)
        if finding:
            findings.append(finding)
            results_counter[finding["result"]] = results_counter.get(finding["result"], 0) + 1
        else:
            results_counter["pass"] += 1

    # run LLM judge concurrently
    if llm_rules:
        client = make_client()
        lock = threading.Lock()
        completed = 0
        _repo_root = Path(repo_root) if repo_root else None

        def _process(rule):
            nonlocal completed
            applicable_files = [f for f in diff_files if _match_rule(rule, [f["path"]])]
            if not applicable_files:
                return None
            finding = _eval_llm_judge(rule, applicable_files, client, model, repo_root=_repo_root)
            with lock:
                completed += 1
                if completed % 10 == 0:
                    logger.info("  judge progress: %d / %d rules", completed, len(llm_rules))
            return finding

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, r): r for r in llm_rules}
            for fut in concurrent.futures.as_completed(futures):
                rule = futures[fut]
                try:
                    finding = fut.result()
                    if finding:
                        findings.append(finding)
                        results_counter[finding["result"]] = results_counter.get(finding["result"], 0) + 1
                    else:
                        results_counter["pass"] += 1
                except Exception as exc:
                    logger.warning(
                        "LLM judge failed for rule %s: %s",
                        rule.get("rule_id", "?")[:8], exc,
                    )
                    applicable_paths = [f["path"] for f in diff_files if _match_rule(rule, [f["path"]])]
                    findings.append({
                        "rule_id": rule.get("rule_id", ""),
                        "rule_text": rule.get("rule_text", ""),
                        "directive_type": rule.get("directive_type", ""),
                        "severity": rule.get("severity", "advisory"),
                        "result": "uncertain",
                        "confidence": 0.0,
                        "file": applicable_paths[0] if applicable_paths else (changed_paths[0] if changed_paths else "unknown"),
                        "line_start": None,
                        "line_end": None,
                        "violating_snippet": "",
                        "violation": "LLM evaluation failed",
                        "fix_hint": "",
                        "examples": _get_examples(rule),
                        "rationale": rule.get("rationale", ""),
                    })
                    results_counter["uncertain"] += 1

    # holistic verification pass: drop false positives, surface missed issues
    if findings:
        _verify_client = client if llm_rules else make_client()
        findings = _verify_findings(findings, patch_text, scoped_rules=matched, model=model, client=_verify_client)
        # recount after verification (additional findings may push total above matched count)
        results_counter = {"pass": 0, "fail": 0, "uncertain": 0}
        for f in findings:
            results_counter[f["result"]] = results_counter.get(f["result"], 0) + 1
        results_counter["pass"] = max(0, len(matched) - results_counter["fail"] - results_counter["uncertain"])

    # sort findings: blockers first, then by confidence
    severity_order = {"blocker": 0, "strong": 1, "advisory": 2}
    findings.sort(key=lambda f: (severity_order.get(f["severity"], 9), -f["confidence"]))

    scoped_rules = [
        {"rule_id": r.get("rule_id", ""), "rule_text": r.get("rule_text", ""),
         "severity": r.get("severity", ""), "directive_type": r.get("directive_type", "")}
        for r in matched
    ]

    summary = {
        "total_rules_checked": len(matched),
        "pass": results_counter["pass"],
        "fail": results_counter["fail"],
        "uncertain": results_counter["uncertain"],
    }

    logger.info(
        "Judge complete: %d checked, %d pass, %d fail, %d uncertain",
        summary["total_rules_checked"], summary["pass"],
        summary["fail"], summary["uncertain"],
    )

    return {"summary": summary, "findings": findings, "scoped_rules": scoped_rules}


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Judge a patch against distilled rules")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    p.add_argument("--patch", required=True, help="path to .patch or .diff file")
    p.add_argument("--model", default=None, help="model name from litellm_config.yaml")
    p.add_argument("--workers", type=int, default=10, help="concurrent LLM evaluations")
    p.add_argument("--json-out", default=None, help="write JSON output to file")
    p.add_argument("--repo-root", default=None, help="path to local clone of the repo being reviewed; when set, full file contents are included in LLM prompts for better applicability judgment")
    p.add_argument("--include-pending", action="store_true", help="include rules with pending human review status")
    args = p.parse_args()

    patch_text = Path(args.patch).read_text()
    result = judge_patch(args.repo, patch_text, args.model, args.workers, include_pending=args.include_pending, repo_root=args.repo_root)

    output = json.dumps(result, indent=2, default=str)

    if args.json_out:
        Path(args.json_out).write_text(output)
        logger.info("Output written to %s", args.json_out)

    # human-readable summary to stdout
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"JUDGE SUMMARY: {s['total_rules_checked']} rules checked")
    print(f"  PASS: {s['pass']}  |  FAIL: {s['fail']}  |  UNCERTAIN: {s['uncertain']}")
    print(f"{'='*60}")

    for f in result["findings"]:
        icon = {"fail": "FAIL", "uncertain": "WARN"}.get(f["result"], f["result"].upper())
        loc = f["file"]
        if f.get("line_start"):
            loc += f":{f['line_start']}"
            if f.get("line_end") and f["line_end"] != f["line_start"]:
                loc += f"-{f['line_end']}"
        print(f"\n  [{icon}] [{f['severity']}] {loc}")
        print(f"    Rule: {f['rule_text'][:80]}")
        print(f"    Violation: {f['violation'][:120]}")
        if f.get("fix_hint"):
            print(f"    Fix: {f['fix_hint'][:120]}")

    if not result["findings"]:
        print("\n  All rules passed.")

    print()


if __name__ == "__main__":
    main()
