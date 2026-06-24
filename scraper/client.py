"""
GitHub GraphQL client with rate-limit handling and cursor pagination.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GITHUB_GQL_URL = "https://api.github.com/graphql"

# Pre-emptively pause when remaining points drop below this threshold.
# GitHub GraphQL budget is 5,000 points/hour; a single paginated PR
# detail fetch costs ~5-10 points depending on nesting.
RATE_LIMIT_FLOOR = 100


class GitHubClient:
    def __init__(self, token: str, *, max_retries: int = 5):
        self._token = token
        self._max_retries = max_retries
        self._client = httpx.Client(
            headers={
                "Authorization": f"bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._rate_remaining: int | None = None
        self._rate_reset: int | None = None

    # ── rate-limit awareness ─────────────────────────────────────────

    def _update_rate_limit(self, headers: httpx.Headers):
        """Track remaining budget from response headers."""
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining is not None:
            self._rate_remaining = int(remaining)
        if reset is not None:
            self._rate_reset = int(reset)

    def _wait_if_near_limit(self):
        """Sleep pre-emptively when we're close to exhausting the budget."""
        if self._rate_remaining is not None and self._rate_remaining < RATE_LIMIT_FLOOR:
            reset = self._rate_reset or int(time.time()) + 60
            wait = max(reset - int(time.time()), 1)
            logger.warning(
                "Rate budget low (%d remaining). Sleeping %ds until reset.",
                self._rate_remaining, wait,
            )
            time.sleep(wait)

    # ── low-level query ──────────────────────────────────────────────

    def query(self, gql: str, variables: dict[str, Any] | None = None) -> dict:
        payload = {"query": gql}
        if variables:
            payload["variables"] = variables

        self._wait_if_near_limit()

        for attempt in range(1, self._max_retries + 1):
            resp = self._client.post(GITHUB_GQL_URL, json=payload)
            self._update_rate_limit(resp.headers)

            # rate-limit back-off
            if resp.status_code in (403, 429):
                reset = int(resp.headers.get("x-ratelimit-reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                logger.warning("Rate limited. Sleeping %ds (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue

            # transient server errors — exponential back-off
            if resp.status_code >= 500:
                wait = 2 ** attempt
                logger.warning("Server error %d. Retrying in %ds (attempt %d)",
                               resp.status_code, wait, attempt)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            body = resp.json()

            if "errors" in body:
                logger.error("GraphQL errors: %s", json.dumps(body["errors"], indent=2))
                raise RuntimeError(f"GraphQL errors: {body['errors']}")

            return body["data"]

        raise RuntimeError("Exhausted retries on GitHub API")

    # ── paginated helper ─────────────────────────────────────────────

    def paginate(
        self,
        gql: str,
        variables: dict[str, Any],
        *,
        path: list[str],          # path to the connection, e.g. ["repository", "pullRequests"]
    ) -> list[dict]:
        """Iterate through all pages and return a flat list of nodes."""
        all_nodes: list[dict] = []
        cursor: str | None = None

        while True:
            vars_copy = {**variables, "cursor": cursor}
            data = self.query(gql, vars_copy)

            # walk down to the connection
            connection = data
            for key in path:
                connection = connection[key]

            nodes = connection.get("nodes", [])
            all_nodes.extend(nodes)

            page_info = connection["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]

        return all_nodes

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
