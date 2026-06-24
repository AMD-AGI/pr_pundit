"""
PR review response generator.

Fetches all open review comments from a GitHub PR, then for each comment
produces a verdict (valid / needs_discussion / not_applicable), reasoning
grounded in the repo's rules and the diff, a ready-to-post reply, and an
optional code fix.

Called by the review_pr MCP tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"
_DIFF_CAP = 200_000
_COMMENT_CAP = 800


def _load_repo_config(slug: str) -> dict:
    config_path = GOLD / slug / "repo_config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def _fetch_pr_comments(repo: str, pr_number: int, gh_token: str) -> list[dict]:
    """Fetch inline review comments + top-level review bodies from GitHub API."""
    import urllib.request

    owner, name = repo.split("/", 1)
    headers = {"Authorization": f"token {gh_token}", "Accept": "application/vnd.github+json"}

    def gh_get(url: str) -> list:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    comments = []

    # Inline review comments (line-level)
    try:
        inline = gh_get(f"https://api.github.com/repos/{owner}/{name}/pulls/{pr_number}/comments")
        for c in inline:
            if c.get("user", {}).get("type") == "Bot" or c.get("user", {}).get("login", "").endswith("[bot]"):
                comments.append({
                    "id": c["id"],
                    "author": c["user"]["login"],
                    "path": c.get("path", ""),
                    "line": c.get("line") or c.get("original_line") or 0,
                    "body": c["body"],
                    "source": "inline",
                })
    except Exception as e:
        logger.warning("Failed to fetch inline comments: %s", e)

    # Top-level review bodies (non-empty)
    try:
        reviews = gh_get(f"https://api.github.com/repos/{owner}/{name}/pulls/{pr_number}/reviews")
        for r in reviews:
            body = r.get("body", "").strip()
            if body and (r.get("user", {}).get("type") == "Bot" or r.get("user", {}).get("login", "").endswith("[bot]")):
                comments.append({
                    "id": r["id"],
                    "author": r["user"]["login"],
                    "path": "",
                    "line": 0,
                    "body": body,
                    "source": "review",
                })
    except Exception as e:
        logger.warning("Failed to fetch reviews: %s", e)

    return comments


_REVIEW_PROMPT = """\
You are a senior engineer helping respond to automated reviewer comments on a pull request
for the GitHub repository "{repo}".

REPOSITORY: {repo}
GITHUB URL: https://github.com/{repo}

PR DIFF (first {diff_cap} chars):
{diff_truncated}

CONTRIBUTING GUIDANCE FOR THIS REPO:
{pr_prep_section}

REVIEWER COMMENTS TO RESPOND TO:
{comments_section}

For each comment produce a JSON object in the "responses" array:

{{
  "responses": [
    {{
      "comment_id": <id>,
      "path": "<file path>",
      "line": <line number>,
      "comment_body": "<first 100 chars of comment>",
      "verdict": "valid" | "needs_discussion" | "not_applicable",
      "reasoning": "1-3 sentences grounding your verdict in the diff and repo rules",
      "suggested_reply": "ready-to-post reply to paste on GitHub (polite, specific, ≤150 words)",
      "code_fix": "exact code snippet to fix the issue, or null"
    }}
  ],
  "summary": "2-3 sentence summary: how many valid/invalid, which to fix before merge"
}}

Verdict definitions:
- valid: the comment identifies a real bug, correctness risk, or clear style violation
  that should be fixed before merge
- needs_discussion: technically possible concern but debatable given the context,
  use case, or stated constraints — reply should explain the trade-off
- not_applicable: wrong assumption about the code, already handled, or irrelevant
  to this PR's scope

Rules:
- Be specific: quote the relevant line or variable name in reasoning
- For "valid" comments, always provide a code_fix
- For "not_applicable", explain exactly why the concern does not apply
- suggested_reply should be professional and collaborative, not defensive
- If the comment is from an automated bot (Copilot, etc.), note that in the reply
  when relevant
- Ground reasoning in the diff and repo rules, not general principles alone

Return ONLY the JSON object.
"""


def review_pr(
    repo: str,
    pr_number: int,
    diff: str = "",
    *,
    gh_token: str = "",
    model: str = "claude-opus-4-7",
) -> dict:
    """Fetch PR review comments and generate grounded responses.

    Args:
        repo:       owner/name slug (e.g. "ROCm/aiter")
        pr_number:  PR number
        diff:       unified diff text (optional, improves analysis)
        gh_token:   GitHub API token
        model:      LiteLLM model name

    Returns:
        dict with keys: responses (list), summary
    """
    from pipeline.llm import llm_call, make_client, parse_json

    slug = repo.replace("/", "_", 1)
    repo_config = _load_repo_config(slug)
    pr_prep = repo_config.get("pr_preparation", {})
    pr_prep_section = yaml.dump(pr_prep, default_flow_style=False) if pr_prep else "(no pr_preparation found)"

    comments = _fetch_pr_comments(repo, pr_number, gh_token)
    if not comments:
        return {"responses": [], "summary": "No bot/automated reviewer comments found on this PR."}

    logger.info("Fetched %d reviewer comment(s)", len(comments))

    comments_section_lines = []
    for c in comments:
        body_preview = c["body"][:_COMMENT_CAP]
        loc = f"{c['path']}:{c['line']}" if c["path"] else "(top-level review)"
        comments_section_lines.append(
            f"[id={c['id']}] [{c['author']}] {loc}\n{body_preview}\n"
        )
    comments_section = "\n".join(comments_section_lines)

    diff_truncated = diff[:_DIFF_CAP] if diff else "(diff not provided)"

    prompt = _REVIEW_PROMPT.format(
        repo=repo,
        diff_truncated=diff_truncated,
        diff_cap=_DIFF_CAP,
        pr_prep_section=pr_prep_section,
        comments_section=comments_section,
    )

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=8192, json_mode=True)
    result = parse_json(raw)

    if not isinstance(result, dict):
        raise ValueError(f"LLM returned non-dict: {type(result)}")

    result.setdefault("responses", [])
    result.setdefault("summary", "")

    # Backfill comment metadata from fetched comments for convenience
    id_to_comment = {c["id"]: c for c in comments}
    for r in result["responses"]:
        cid = r.get("comment_id")
        if cid and cid in id_to_comment:
            c = id_to_comment[cid]
            r.setdefault("path", c["path"])
            r.setdefault("line", c["line"])
            r.setdefault("comment_body", c["body"][:200])

    return result
