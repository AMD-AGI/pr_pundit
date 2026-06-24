"""
Microsoft Teams notification via Incoming Webhook (Adaptive Cards).

Sends pipeline status updates as rich adaptive cards.
Set TEAMS_WEBHOOK_URL in .env.  If unset, notifications are silently skipped.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _webhook_url() -> str | None:
    return os.environ.get("TEAMS_WEBHOOK_URL") or os.environ.get("TEAMS_WEBHOOK")


def _post_card(card: dict[str, Any]):
    """Post an adaptive card to the configured Teams webhook."""
    url = _webhook_url()
    if not url:
        logger.debug("TEAMS_WEBHOOK_URL not set — skipping notification")
        return

    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": card,
            }
        ],
    }

    try:
        resp = httpx.post(url, json=payload, timeout=10.0)
        resp.raise_for_status()
    except Exception:
        logger.warning("Teams notification failed", exc_info=True)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _make_card(
    title: str,
    status: str,
    facts: list[tuple[str, str]],
    *,
    color: str = "default",   # "good" | "attention" | "warning" | "default"
    body_text: str | None = None,
) -> dict:
    """Build an Adaptive Card with a header, fact set, and optional body."""
    header_color = {
        "good": "Good",
        "attention": "Attention",
        "warning": "Warning",
    }.get(color, "Default")

    items: list[dict] = [
        {
            "type": "TextBlock",
            "size": "Medium",
            "weight": "Bolder",
            "text": title,
            "color": header_color,
        },
        {
            "type": "TextBlock",
            "text": f"Status: **{status}**",
            "wrap": True,
        },
        {
            "type": "FactSet",
            "facts": [{"title": k, "value": v} for k, v in facts],
        },
    ]

    if body_text:
        items.append({
            "type": "TextBlock",
            "text": body_text,
            "wrap": True,
            "size": "Small",
        })

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": items,
    }


# ── Public notification helpers ──────────────────────────────────────

def notify_scrape_start(repo: str, *, resumed: bool = False, already_done: int = 0):
    status = "Resumed" if resumed else "Started"
    facts = [
        ("Stage", "Bronze — Scrape"),
        ("Repository", repo),
        ("Status", status),
        ("Timestamp", _timestamp()),
    ]
    if resumed:
        facts.append(("Already scraped", str(already_done)))
    _post_card(_make_card(f"🔄 Scrape {status}", status, facts))


def notify_scrape_progress(repo: str, scraped: int, total: str | int):
    facts = [
        ("Stage", "Bronze — Scrape"),
        ("Repository", repo),
        ("Progress", f"{scraped} / {total} PRs"),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("📥 Scrape Progress", "In Progress", facts))


def notify_scrape_done(repo: str, total_scraped: int):
    facts = [
        ("Stage", "Bronze — Scrape"),
        ("Repository", repo),
        ("PRs scraped", str(total_scraped)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("✅ Scrape Complete", "Done", facts, color="good"))


def notify_scrape_error(repo: str, error: str, scraped_so_far: int):
    facts = [
        ("Stage", "Bronze — Scrape"),
        ("Repository", repo),
        ("PRs before error", str(scraped_so_far)),
        ("Error", error[:200]),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("❌ Scrape Failed", "Error", facts, color="attention"))


def notify_scrape_rate_limited(repo: str, remaining: int, wait_seconds: int):
    facts = [
        ("Stage", "Bronze — Scrape"),
        ("Repository", repo),
        ("Remaining budget", str(remaining)),
        ("Sleeping", f"{wait_seconds}s"),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("⏳ Rate Limit Pause", "Waiting", facts, color="warning"))


def notify_normalize_done(repo: str, threads: int, examples: int):
    facts = [
        ("Stage", "Silver — Normalize"),
        ("Repository", repo),
        ("Threads", str(threads)),
        ("Review examples", str(examples)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("✅ Normalize Complete", "Done", facts, color="good"))


def notify_distill_start(repo: str, run_id: str, provider: str, model: str, examples: int):
    facts = [
        ("Stage", "Gold — Distill"),
        ("Repository", repo),
        ("Run ID", run_id[:8]),
        ("Provider", f"{provider} / {model}"),
        ("Input examples", str(examples)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("🔄 Distillation Started", "Started", facts))


def notify_distill_filter_progress(repo: str, done: int, total: int, kept: int):
    facts = [
        ("Stage", "Gold — Distill (filter)"),
        ("Repository", repo),
        ("Progress", f"{done} / {total} examples"),
        ("Normative so far", str(kept)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("📊 Filter Progress", "In Progress", facts))


def notify_distill_done(repo: str, run_id: str, clusters: int, rules: int):
    facts = [
        ("Stage", "Gold — Distill"),
        ("Repository", repo),
        ("Run ID", run_id[:8]),
        ("Clusters", str(clusters)),
        ("Rules (pending review)", str(rules)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("✅ Distillation Complete", "Done", facts, color="good"))


def notify_distill_error(repo: str, run_id: str, error: str):
    facts = [
        ("Stage", "Gold — Distill"),
        ("Repository", repo),
        ("Run ID", run_id[:8]),
        ("Error", error[:200]),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("❌ Distillation Failed", "Error", facts, color="attention"))


def notify_verify_done(repo: str, patch_ref: str, passed: int, failed: int, uncertain: int):
    total = passed + failed + uncertain
    facts = [
        ("Stage", "Verify"),
        ("Repository", repo),
        ("Patch", patch_ref[:60]),
        ("Rules checked", str(total)),
        ("Passed", str(passed)),
        ("Failed", str(failed)),
        ("Uncertain", str(uncertain)),
        ("Timestamp", _timestamp()),
    ]
    color = "good" if failed == 0 else "attention"
    _post_card(_make_card("🔍 Verification Complete", "Done", facts, color=color))


def notify_distill_recipes_start(repo: str, model: str, n_records: int, workers: int):
    facts = [
        ("Stage", "Gold — Distill Recipes"),
        ("Repository", repo),
        ("Model", model),
        ("Silver records", str(n_records)),
        ("Workers", str(workers)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("🔄 Recipe Distillation Started", "Started", facts))


def notify_distill_recipes_type_done(repo: str, test_type: str, n_recipes: int):
    facts = [
        ("Stage", "Gold — Distill Recipes"),
        ("Repository", repo),
        ("Test type", test_type),
        ("Recipes", str(n_recipes)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card(f"📦 {test_type} Done", "In Progress", facts))


def notify_distill_recipes_done(repo: str, total_recipes: int, n_categories: int, n_expectations: int):
    facts = [
        ("Stage", "Gold — Distill Recipes"),
        ("Repository", repo),
        ("Total recipes", str(total_recipes)),
        ("Categories", str(n_categories)),
        ("Expectation patterns", str(n_expectations)),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("✅ Recipe Distillation Complete", "Done", facts, color="good"))


def notify_distill_recipes_error(repo: str, error: str):
    facts = [
        ("Stage", "Gold — Distill Recipes"),
        ("Repository", repo),
        ("Error", error[:200]),
        ("Timestamp", _timestamp()),
    ]
    _post_card(_make_card("❌ Recipe Distillation Failed", "Error", facts, color="attention"))
