"""
pr-pundit-apply — thin local client for PR Pundit.

Downloads a computed PR plan from the MCP server and executes the git/gh phases
(fork, push, open PR, create/update tracking issue) using the developer's own
GitHub credentials. No Anthropic API key required — all LLM work already happened
on the server.

Usage:
    uvx --from git+https://github.com/AMD-AGI/pr-scraper pr-pundit-apply \\
        --server https://your-mcp-server \\
        --plan-id abc12345 \\
        [--upstream owner/repo]   # override target repo if needed
        [--no-draft]              # open as ready-for-review instead of draft
        [--dry-run]               # show what would happen, skip git/gh

Prerequisites:
    - gh CLI installed and authenticated (gh auth login)
    - git user.name and user.email configured
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Apply a PR Pundit plan — fork, push, and open PRs using your GitHub credentials."
    )
    p.add_argument("--server", required=True, help="Base URL of the MCP server, e.g. http://10.x.x.x:8502")
    p.add_argument("--plan-id", required=True, help="Plan ID returned by plan_pr_series")
    p.add_argument("--upstream", default="", help="Override target repo (owner/name)")
    p.add_argument("--no-draft", action="store_true", help="Open as ready-for-review instead of draft")
    p.add_argument("--dry-run", action="store_true", help="Show actions without executing git/gh commands")
    args = p.parse_args()

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is required. Install with: pip install httpx", file=sys.stderr)
        sys.exit(1)

    # 1. Download plan artifacts from the MCP server
    url = f"{args.server.rstrip('/')}/plans/{args.plan_id}"
    print(f"Fetching plan {args.plan_id} from {args.server} ...")
    try:
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            print(f"ERROR: Plan '{args.plan_id}' not found on server. It may have expired (24h TTL).", file=sys.stderr)
        else:
            print(f"ERROR: Server returned {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as exc:
        print(f"ERROR: Could not reach server at {args.server}: {exc}", file=sys.stderr)
        sys.exit(1)

    plan = resp.json()
    target_repo = args.upstream or plan.get("target_repo", "")
    if not target_repo:
        print("ERROR: No target repo in plan and --upstream not provided.", file=sys.stderr)
        sys.exit(1)

    prs = plan.get("prs_created", [])
    if not prs:
        print("ERROR: Plan contains no PRs.", file=sys.stderr)
        sys.exit(1)

    issue_title = plan.get("issue_title", "")
    pr_plan = plan.get("pr_plan", {})
    draft = not args.no_draft

    print(f"Target repo:  {target_repo}")
    print(f"PRs to open:  {len(prs)}")
    print(f"Draft:        {draft}")
    if args.dry_run:
        print("(dry-run mode — no git/gh commands will run)")
    print()

    # Import git/gh functions from the pipeline (available since we installed from git)
    from pipeline.create_pr_from_seed import (
        _create_issue,
        _create_pr,
        _fork_and_push,
        _gh_token,
        _update_issue_body,
    )

    # Resolve GitHub token from environment or gh CLI keyring
    try:
        token = _gh_token()
    except Exception as exc:
        print(f"ERROR: Could not get GitHub token: {exc}", file=sys.stderr)
        print("Run 'gh auth login' or set GITHUB_TOKEN in your environment.", file=sys.stderr)
        sys.exit(1)

    # 2. Create a stub tracking issue now so PRs can reference it
    parent_issue_url = ""
    if issue_title and not args.dry_run:
        stub_body = (
            plan.get("issue_instruction", {}).get("stub_body")
            or (
                f"This issue tracks a series of {len(prs)} pull request(s) "
                f"targeting `{target_repo}`.\n\n"
                f"**Status:** PRs being opened — full description will be added shortly.\n\n"
                + "\n".join(f"- PR {pr['index']}: {pr['title']}" for pr in prs)
            )
        )
        try:
            parent_issue_url = _create_issue(target_repo, issue_title, stub_body)
            print(f"Tracking issue created (stub): {parent_issue_url}")
        except Exception as exc:
            logger.warning("Could not create tracking issue: %s", exc)
    elif issue_title and args.dry_run:
        print(f"[dry-run] Would create tracking issue: {issue_title!r}")

    # 3. Fork, push, and open each PR in index order
    pr_urls_by_index: dict[int, str] = {}
    for pr in sorted(prs, key=lambda x: x.get("index", 0)):
        pr_idx = pr["index"]
        pr_branch = pr["branch"]
        pr_title = pr["title"]
        pr_diff = pr.get("diff", "")
        ancestor_diffs = pr.get("ancestor_diffs") or None
        pr_description = pr.get("pr_description", "")
        commit_message = pr.get("commit_message", pr_title)

        print(f"Preparing PR {pr_idx}/{len(prs)}: {pr_title}")

        if args.dry_run:
            print(f"  [dry-run] Would fork {target_repo}, create branch {pr_branch}, push and open PR")
            print(f"  Commit: {commit_message[:80]}")
            continue

        try:
            fork_slug = _fork_and_push(
                target_repo, pr_branch, pr_diff, token,
                ancestor_diffs=ancestor_diffs,
            )
        except Exception as exc:
            print(f"ERROR: fork/push failed for PR {pr_idx}: {exc}", file=sys.stderr)
            sys.exit(1)

        # Inject parent issue reference into description if not already there
        if parent_issue_url and parent_issue_url not in pr_description:
            pr_description = f"Part of {parent_issue_url}\n\n{pr_description}"

        try:
            pr_url = _create_pr(target_repo, fork_slug, pr_branch, pr_title, pr_description, draft=draft)
        except Exception as exc:
            print(f"ERROR: PR creation failed for PR {pr_idx}: {exc}", file=sys.stderr)
            sys.exit(1)

        pr_urls_by_index[pr_idx] = pr_url
        print(f"  PR created: {pr_url}")

    # 4. Update tracking issue with real PR links
    if parent_issue_url and pr_urls_by_index:
        # Build a simple update body with real PR links (no LLM — keeps this client thin)
        pr_list = "\n".join(
            f"- PR {idx}: {pr_urls_by_index[idx]}" for idx in sorted(pr_urls_by_index)
        )
        updated_body = (
            f"This issue tracks a series of {len(prs)} pull request(s) "
            f"targeting `{target_repo}`.\n\n"
            f"## PRs\n{pr_list}\n\n"
            f"*Generated by PR Pundit — plan ID: {args.plan_id}*"
        )
        try:
            _update_issue_body(target_repo, parent_issue_url, updated_body)
            print(f"Tracking issue updated: {parent_issue_url}")
        except Exception as exc:
            logger.warning("Could not update tracking issue: %s", exc)

    if not args.dry_run:
        print(f"\nDone. {len(pr_urls_by_index)}/{len(prs)} PR(s) opened.")
        for idx in sorted(pr_urls_by_index):
            print(f"  PR {idx}: {pr_urls_by_index[idx]}")
    else:
        print(f"\n[dry-run] Would have opened {len(prs)} PR(s) on {target_repo}.")


if __name__ == "__main__":
    main()
