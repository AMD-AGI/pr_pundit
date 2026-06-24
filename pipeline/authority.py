"""
Compute reviewer authority scores from bronze data.

Produces data/gold/{repo}/reviewer_authority.json with per-user stats
and tier labels.

Usage:
    python -m pipeline.authority --repo owner_name
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BRONZE = Path(__file__).resolve().parent.parent / "data" / "bronze"
GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

BOT_SUFFIXES = ["[bot]", "-bot", "bot"]
BOT_NAMES = {"dependabot", "renovate", "codecov", "github-actions",
             "gemini-code-assist", "copilot"}

# GitHub orgs whose members get a boosted authority score.
# Override via --trusted-orgs if your target repos are maintained by different orgs.
TRUSTED_ORGS: list[str] = []
TRUSTED_ORG_SCORE_MULTIPLIER = 1.5


def _fetch_org_members(orgs: list[str], token: str) -> set[str]:
    """Fetch members of GitHub orgs. Returns set of logins.

    Tries the public /members endpoint first. If the org has private membership
    visibility (403), falls back to /orgs/{org}/teams or the GraphQL search —
    ultimately populates at least the reviewers seen in the bronze data who are
    confirmed org members via /orgs/{org}/memberships/{login}.
    """
    import httpx
    members: set[str] = set()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    for org in orgs:
        # Try paginated public-members list first (requires read:org for private orgs)
        url: str | None = f"https://api.github.com/orgs/{org}/members?per_page=100"
        fetched_via_list = False
        while url:
            try:
                r = httpx.get(url, headers=headers, timeout=30)
                if r.status_code == 404:
                    logger.warning("Org not found: %s", org)
                    url = None
                    break
                if r.status_code == 403:
                    # Private org membership — token lacks read:org scope.
                    # Fall through to the per-user membership check below.
                    logger.info(
                        "Org %s has private membership (403); will verify members individually",
                        org,
                    )
                    url = None
                    break
                r.raise_for_status()
                for m in r.json():
                    members.add(m["login"])
                fetched_via_list = True
                next_url = None
                for part in r.headers.get("link", "").split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip().strip("<>")
                url = next_url
            except Exception as exc:
                logger.warning("Failed to fetch members list for %s: %s", org, exc)
                url = None
                break

        if not fetched_via_list:
            # Private org: confirm membership for the authenticated token owner only.
            # Reviewers seen in bronze data will be verified individually in compute_authority.
            try:
                me = httpx.get("https://api.github.com/user", headers=headers, timeout=10)
                me.raise_for_status()
                my_login = me.json().get("login", "")
                if my_login:
                    check = httpx.get(
                        f"https://api.github.com/orgs/{org}/memberships/{my_login}",
                        headers=headers, timeout=10,
                    )
                    if check.status_code == 200 and check.json().get("state") == "active":
                        members.add(my_login)
                        logger.info("Confirmed %s is an active member of %s", my_login, org)
            except Exception as exc:
                logger.warning("Could not verify own membership in %s: %s", org, exc)

    logger.info("Fetched %d org members across %s", len(members), orgs)
    return members


def _is_bot(login: str) -> bool:
    low = login.lower()
    if low in BOT_NAMES:
        return True
    return any(low.endswith(s) for s in BOT_SUFFIXES)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_authority(repo_slug: str, amd_orgs: list[str] | None = None) -> dict:
    src = BRONZE / repo_slug
    prs = _read_jsonl(src / "pull_requests.jsonl")
    reviews = _read_jsonl(src / "reviews.jsonl")

    # load trusted reviewers from config
    config_path = GOLD / repo_slug / "repo_config.yaml"
    trusted: set[str] = set()
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        trusted = set(config.get("trusted_reviewers", []))

    # fetch trusted org members
    orgs = amd_orgs if amd_orgs is not None else TRUSTED_ORGS
    amd_members: set[str] = set()
    if orgs:
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            amd_members = _fetch_org_members(orgs, token)
        else:
            logger.warning("GITHUB_TOKEN not set — skipping org membership lookup")

    # count merged PRs per author
    merged_prs: dict[str, int] = {}
    for pr in prs:
        login = (pr.get("author") or {}).get("login", "")
        if not login or _is_bot(login):
            continue
        if pr.get("mergedAt"):
            merged_prs[login] = merged_prs.get(login, 0) + 1

    # count reviews given per reviewer
    reviews_given: dict[str, int] = {}
    for rev in reviews:
        login = (rev.get("author") or {}).get("login", "")
        if not login or _is_bot(login):
            continue
        reviews_given[login] = reviews_given.get(login, 0) + 1

    # combine all known users
    all_users = set(merged_prs.keys()) | set(reviews_given.keys())

    # compute authority score
    scores: dict[str, dict] = {}
    for user in all_users:
        mp = merged_prs.get(user, 0)
        rg = reviews_given.get(user, 0)
        is_trusted = user in amd_members
        base_score = mp * 2 + rg
        score = math.ceil(base_score * TRUSTED_ORG_SCORE_MULTIPLIER) if is_trusted else base_score
        scores[user] = {
            "login": user,
            "merged_prs": mp,
            "reviews_given": rg,
            "authority_score": score,
            "trusted_org_member": is_trusted,
        }

    # compute tier thresholds from percentiles
    all_scores = sorted(s["authority_score"] for s in scores.values())
    if all_scores:
        p95 = all_scores[int(len(all_scores) * 0.95)]
        p80 = all_scores[int(len(all_scores) * 0.80)]
    else:
        p95 = p80 = 0

    for user, s in scores.items():
        if user in trusted or s["authority_score"] >= p95:
            s["tier"] = "core maintainer"
        elif s["authority_score"] >= p80:
            s["tier"] = "frequent contributor"
        elif s["merged_prs"] >= 5 or s["reviews_given"] >= 10:
            s["tier"] = "contributor"
        else:
            s["tier"] = "occasional contributor"

    # sort by score descending
    ranked = dict(sorted(scores.items(), key=lambda x: -x[1]["authority_score"]))

    result = {
        "repo": repo_slug.replace("_", "/", 1),
        "total_users": len(ranked),
        "thresholds": {"p95": p95, "p80": p80},
        "trusted_reviewers": sorted(trusted),
        "users": ranked,
    }

    # write
    out = GOLD / repo_slug / "reviewer_authority.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # log top 10
    top = list(ranked.values())[:10]
    logger.info("Authority computed for %d users (p95=%d, p80=%d)", len(ranked), p95, p80)
    for u in top:
        logger.info("  %s: %d merged, %d reviews, score=%d [%s]",
                     u["login"], u["merged_prs"], u["reviews_given"],
                     u["authority_score"], u["tier"])

    return result


def main():
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Compute reviewer authority scores")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    p.add_argument("--trusted-orgs", default=",".join(TRUSTED_ORGS),
                   help="comma-separated GitHub orgs whose members get a boosted authority score")
    p.add_argument("--no-trusted-orgs", action="store_true", help="disable org membership lookup")
    args = p.parse_args()
    orgs = [] if args.no_trusted_orgs else [o for o in args.trusted_orgs.split(",") if o]
    compute_authority(args.repo, amd_orgs=orgs)


if __name__ == "__main__":
    main()
