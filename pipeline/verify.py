"""
Gate verifier — evaluate a proposed patch against gold rules.

Usage:
    python -m pipeline.verify --repo owner_name --patch diff.patch

Or import and call programmatically:

    from pipeline.verify import verify_patch
    results = verify_patch("owner_name", patch_text)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.notify import notify_verify_done

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"


def _load_rules(repo_slug: str) -> list[dict]:
    rules_path = GOLD / repo_slug / "rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"No rules at {rules_path}. Run distill first.")
    rules = json.loads(rules_path.read_text())
    # only enforce accepted or pending rules (never rejected/deprecated)
    return [
        r for r in rules
        if r.get("human_review_status") in ("pending", "accepted")
    ]


def _match_rule(rule: dict, changed_paths: list[str]) -> bool:
    """Check whether a rule applies to any of the changed paths."""
    scope = rule.get("scope", {})

    # repo-wide rules always match
    if scope.get("repo_wide", True) and not scope.get("path_prefixes") and not scope.get("languages"):
        return True

    for p in changed_paths:
        # path prefix check
        for prefix in scope.get("path_prefixes", []):
            if p.startswith(prefix):
                return True
        # language check
        for lang in scope.get("languages", []):
            ext_map = {"python": ".py", "typescript": ".ts", "javascript": ".js",
                       "go": ".go", "rust": ".rs", "java": ".java"}
            if p.endswith(ext_map.get(lang, f".{lang}")):
                return True
        # glob patterns (simplified: just prefix match on pattern stem)
        for pat in scope.get("file_patterns", []):
            stem = pat.replace("*", "").replace("?", "")
            if stem and stem in p:
                return True

    return False


def _run_deterministic_check(rule: dict, patch_text: str) -> dict:
    """Run regex/pattern-based verification."""
    import re as re_mod
    verifier = rule.get("verifier", {})
    pattern = verifier.get("pattern")
    if not pattern:
        return {"result": "skipped", "confidence": 0.0, "explanation": "No pattern defined"}

    try:
        matches = re_mod.findall(pattern, patch_text)
        vtype = rule.get("directive_type", "")
        if "forbidden" in vtype:
            if matches:
                return {"result": "fail", "confidence": 1.0,
                        "explanation": f"Forbidden pattern found: {matches[:3]}"}
            return {"result": "pass", "confidence": 1.0, "explanation": "No forbidden pattern found"}
        else:
            if matches:
                return {"result": "pass", "confidence": 1.0,
                        "explanation": f"Required pattern found: {matches[:3]}"}
            return {"result": "fail", "confidence": 0.9,
                    "explanation": "Required pattern not found in patch"}
    except re_mod.error:
        return {"result": "skipped", "confidence": 0.0, "explanation": "Invalid regex pattern"}


def verify_patch(repo_slug: str, patch_text: str) -> list[dict]:
    """Run all applicable rules against a patch and return eval results."""
    rules = _load_rules(repo_slug)

    # extract changed paths from unified diff
    import re as re_mod
    changed_paths = re_mod.findall(r"^\+\+\+ b/(.+)$", patch_text, re_mod.MULTILINE)
    if not changed_paths:
        changed_paths = re_mod.findall(r"^diff --git a/.+ b/(.+)$", patch_text, re_mod.MULTILINE)

    results = []
    for rule in rules:
        if not _match_rule(rule, changed_paths):
            continue

        verifier = rule.get("verifier", {})
        vtype = verifier.get("verifier_type", "")

        if vtype in ("regex", "ast", "metadata"):
            check = _run_deterministic_check(rule, patch_text)
        elif vtype == "llm_judge":
            # placeholder — plug in your LLM judge here
            check = {"result": "uncertain", "confidence": 0.5,
                     "explanation": "LLM judge not configured"}
        else:
            check = {"result": "skipped", "confidence": 0.0,
                     "explanation": f"Verifier type '{vtype}' not implemented"}

        results.append({
            "eval_id": str(uuid.uuid4()),
            "rule_id": rule.get("rule_id", ""),
            "rule_text": rule.get("rule_text", ""),
            "rule_version": rule.get("rule_version", 1),
            "human_review_status": rule.get("human_review_status", "pending"),
            "severity": rule.get("severity", "advisory"),
            "result": check["result"],
            "confidence": check["confidence"],
            "explanation": check["explanation"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        })

    return results


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Verify a patch against gold rules")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    p.add_argument("--patch", required=True, help="path to .patch or .diff file")
    args = p.parse_args()

    patch_text = Path(args.patch).read_text()
    results = verify_patch(args.repo, patch_text)

    # summary
    by_result = {}
    for r in results:
        by_result.setdefault(r["result"], []).append(r)

    passed = len(by_result.get("pass", []))
    failed = len(by_result.get("fail", []))
    uncertain = len(by_result.get("uncertain", []))

    notify_verify_done(
        args.repo.replace("_", "/", 1),
        Path(args.patch).name,
        passed, failed, uncertain,
    )

    print(json.dumps(results, indent=2))
    print(f"\n--- Summary ---")
    for status in ("fail", "pass", "uncertain", "skipped"):
        items = by_result.get(status, [])
        if items:
            print(f"  {status.upper()}: {len(items)}")
            for it in items:
                flag = f" [{it['human_review_status']}]" if it["human_review_status"] != "accepted" else ""
                print(f"    • {it['rule_text'][:80]}{flag}")


if __name__ == "__main__":
    main()
