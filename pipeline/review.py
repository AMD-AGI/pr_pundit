"""
Human review tool — non-blocking post-distillation review of candidate rules.

Usage:
    # List pending rules
    python -m pipeline.review --repo owner_name --list

    # Accept a rule
    python -m pipeline.review --repo owner_name --accept RULE_ID

    # Reject a rule
    python -m pipeline.review --repo owner_name --reject RULE_ID --reason "too broad"

    # Mark as needs-splitting
    python -m pipeline.review --repo owner_name --split RULE_ID
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"


def _load_rules(repo_slug: str) -> tuple[Path, list[dict]]:
    path = GOLD / repo_slug / "rules.json"
    if not path.exists():
        raise FileNotFoundError(f"No rules at {path}")
    return path, json.loads(path.read_text())


def _save_rules(path: Path, rules: list[dict]):
    path.write_text(json.dumps(rules, indent=2, default=str))


def list_rules(repo_slug: str, status_filter: str | None = None):
    _, rules = _load_rules(repo_slug)
    for r in rules:
        if status_filter and r.get("human_review_status") != status_filter:
            continue
        status = r.get("human_review_status", "?")
        severity = r.get("severity", "?")
        print(f"[{status:15s}] [{severity:8s}] {r.get('rule_id', '?')[:8]}…  {r.get('rule_text', '')[:70]}")


def update_rule(repo_slug: str, rule_id: str, new_status: str, reason: str | None = None):
    path, rules = _load_rules(repo_slug)
    found = False
    for r in rules:
        if r.get("rule_id", "").startswith(rule_id):
            r["human_review_status"] = new_status
            r["human_review_notes"] = reason
            r["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            logger.info("Updated rule %s → %s", r["rule_id"][:8], new_status)
            break

    if not found:
        logger.error("Rule %s not found", rule_id)
        return

    _save_rules(path, rules)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Review distilled rules (non-blocking)")
    p.add_argument("--repo", required=True)
    p.add_argument("--list", action="store_true", dest="list_rules")
    p.add_argument("--status", default=None, help="filter by status")
    p.add_argument("--accept", metavar="RULE_ID")
    p.add_argument("--reject", metavar="RULE_ID")
    p.add_argument("--split", metavar="RULE_ID")
    p.add_argument("--deprecate", metavar="RULE_ID")
    p.add_argument("--reason", default=None)
    args = p.parse_args()

    if args.list_rules:
        list_rules(args.repo, args.status)
    elif args.accept:
        update_rule(args.repo, args.accept, "accepted", args.reason)
    elif args.reject:
        update_rule(args.repo, args.reject, "rejected", args.reason)
    elif args.split:
        update_rule(args.repo, args.split, "needs_splitting", args.reason)
    elif args.deprecate:
        update_rule(args.repo, args.deprecate, "deprecated", args.reason)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
