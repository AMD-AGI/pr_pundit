"""
Stage A — Bronze scraper.

Scrapes all merged PRs for a repo via GraphQL and writes raw JSON to
data/bronze/{owner}_{repo}/.  Each entity type gets its own JSONL file.

Supports checkpoint/resume: a checkpoint.json tracks which PR numbers
have been fully scraped.  On restart the scraper skips completed PRs
and resumes the PR listing from where it left off.

Usage:
    python -m scraper.scrape_bronze --repo owner/name --token ghp_...
    python -m scraper.scrape_bronze --repo owner/name --limit 50   # test run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.notify import (
    notify_scrape_done,
    notify_scrape_error,
    notify_scrape_progress,
    notify_scrape_rate_limited,
    notify_scrape_start,
)
from scraper.client import GitHubClient
from scraper.queries import (
    CLOSED_PRS_QUERY,
    MERGED_PRS_QUERY,
    PR_COMMITS_QUERY,
    PR_FILES_QUERY,
    PR_ISSUE_COMMENTS_QUERY,
    PR_REVIEWS_QUERY,
    PR_REVIEW_THREADS_QUERY,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze"


# ── checkpoint ───────────────────────────────────────────────────────

class Checkpoint:
    """Tracks completed PR numbers and the PR-list cursor for resume."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict = {"completed_prs": [], "pr_list_cursor": None}
        if path.exists():
            self._data = json.loads(path.read_text())

    @property
    def completed(self) -> set[int]:
        return set(self._data.get("completed_prs", []))

    @property
    def pr_list_cursor(self) -> str | None:
        return self._data.get("pr_list_cursor")

    def mark_pr_done(self, pr_number: int):
        completed = self._data.setdefault("completed_prs", [])
        if pr_number not in completed:
            completed.append(pr_number)
        self._save()

    def save_pr_list_cursor(self, cursor: str | None):
        self._data["pr_list_cursor"] = cursor
        self._save()

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2))


# ── JSONL writer ─────────────────────────────────────────────────────

def _jsonl_writer(path: Path):
    """Return a callable that appends one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")

    def _write(obj: dict):
        fh.write(json.dumps(obj, default=str) + "\n")
        fh.flush()

    _write._close = fh.close  # type: ignore[attr-defined]
    return _write


# ── main scrape loop ─────────────────────────────────────────────────

def scrape_repo(owner: str, name: str, token: str, *, limit: int | None = None):
    repo_display = f"{owner}/{name}"
    out = DATA_DIR / f"{owner}_{name}"
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Checkpoint(out / "checkpoint.json")
    done = ckpt.completed

    resumed = bool(done)
    if resumed:
        logger.info("Resuming — %d PRs already scraped, skipping those", len(done))

    notify_scrape_start(repo_display, resumed=resumed, already_done=len(done))

    write_pr = _jsonl_writer(out / "pull_requests.jsonl")
    write_commit = _jsonl_writer(out / "commits.jsonl")
    write_file = _jsonl_writer(out / "files.jsonl")
    write_review = _jsonl_writer(out / "reviews.jsonl")
    write_thread = _jsonl_writer(out / "review_threads.jsonl")
    write_issue_comment = _jsonl_writer(out / "issue_comments.jsonl")

    scraped_count = 0
    rate_limit_notified = False

    try:
        with GitHubClient(token) as gh:
            logger.info("Fetching merged PRs for %s/%s …", owner, name)

            # paginate through the PR listing, resuming from saved cursor
            cursor = ckpt.pr_list_cursor
            page_num = 0

            while True:
                data = gh.query(MERGED_PRS_QUERY, {"owner": owner, "name": name, "cursor": cursor})
                connection = data["repository"]["pullRequests"]
                prs = connection.get("nodes", [])
                page_info = connection["pageInfo"]
                total = connection.get("totalCount", "?")
                page_num += 1

                logger.info("PR list page %d  (%d PRs on page, %s total merged)", page_num, len(prs), total)

                for pr in prs:
                    pr_num = pr["number"]

                    # skip already-completed PRs
                    if pr_num in done:
                        logger.debug("  PR #%d already scraped, skipping", pr_num)
                        continue

                    if limit and scraped_count >= limit:
                        logger.info("Reached --limit %d, stopping", limit)
                        break

                    logger.info("  PR #%d  (scraped %d so far)", pr_num, scraped_count + 1)
                    write_pr(pr)

                    base_vars = {"owner": owner, "name": name, "number": pr_num}

                    # commits
                    commits = gh.paginate(PR_COMMITS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "commits"])
                    for c in commits:
                        c["_pr_number"] = pr_num
                        write_commit(c)

                    # files
                    files = gh.paginate(PR_FILES_QUERY, base_vars,
                                        path=["repository", "pullRequest", "files"])
                    for f in files:
                        f["_pr_number"] = pr_num
                        write_file(f)

                    # reviews
                    reviews = gh.paginate(PR_REVIEWS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "reviews"])
                    for r in reviews:
                        r["_pr_number"] = pr_num
                        write_review(r)

                    # review threads (resolution state + inline comments)
                    threads = gh.paginate(PR_REVIEW_THREADS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "reviewThreads"])
                    for t in threads:
                        t["_pr_number"] = pr_num
                        write_thread(t)

                    # issue-level comments
                    issue_comments = gh.paginate(PR_ISSUE_COMMENTS_QUERY, base_vars,
                                                 path=["repository", "pullRequest", "comments"])
                    for ic in issue_comments:
                        ic["_pr_number"] = pr_num
                        write_issue_comment(ic)

                    # checkpoint this PR as complete
                    ckpt.mark_pr_done(pr_num)
                    scraped_count += 1

                    # periodic Teams progress update
                    if scraped_count % 50 == 0:
                        notify_scrape_progress(repo_display, scraped_count, total)
                        rate_limit_notified = False  # allow one more rate-limit notif per batch

                    # notify if rate-limited (throttle: at most once per pause)
                    if gh._rate_remaining is not None and gh._rate_remaining < 200:
                        if not rate_limit_notified:
                            notify_scrape_rate_limited(
                                repo_display, gh._rate_remaining,
                                max((gh._rate_reset or 0) - int(__import__("time").time()), 0),
                            )
                            rate_limit_notified = True

                # check if we hit limit or exhausted pages
                if limit and scraped_count >= limit:
                    break

                if not page_info["hasNextPage"]:
                    break

                cursor = page_info["endCursor"]
                ckpt.save_pr_list_cursor(cursor)

    except Exception as exc:
        notify_scrape_error(repo_display, str(exc), scraped_count)
        raise
    finally:
        # close file handles
        for w in [write_pr, write_commit, write_file, write_review, write_thread, write_issue_comment]:
            w._close()  # type: ignore[attr-defined]

    notify_scrape_done(repo_display, scraped_count)
    logger.info("Bronze scrape complete — %d new PRs scraped → %s", scraped_count, out)


def scrape_repo_closed(owner: str, name: str, token: str, *, limit: int | None = None):
    """Scrape CLOSED (rejected) PRs into the same JSONL files as merged PRs.

    Uses a separate checkpoint_closed.json so it never interferes with the
    merged-PR checkpoint.  Records are distinguished by state: CLOSED.
    """
    repo_display = f"{owner}/{name}"
    out = DATA_DIR / f"{owner}_{name}"
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Checkpoint(out / "checkpoint_closed.json")
    done = ckpt.completed

    resumed = bool(done)
    if resumed:
        logger.info("Closed PRs — resuming: %d already scraped, skipping those", len(done))
    else:
        logger.info("Closed PRs — starting fresh scrape for %s", repo_display)

    # Append to the same JSONL files; state field distinguishes CLOSED records
    write_pr = _jsonl_writer(out / "pull_requests.jsonl")
    write_commit = _jsonl_writer(out / "commits.jsonl")
    write_file = _jsonl_writer(out / "files.jsonl")
    write_review = _jsonl_writer(out / "reviews.jsonl")
    write_thread = _jsonl_writer(out / "review_threads.jsonl")
    write_issue_comment = _jsonl_writer(out / "issue_comments.jsonl")

    scraped_count = 0
    rate_limit_notified = False

    try:
        with GitHubClient(token) as gh:
            logger.info("Fetching closed PRs for %s/%s …", owner, name)

            cursor = ckpt.pr_list_cursor
            page_num = 0

            while True:
                data = gh.query(CLOSED_PRS_QUERY, {"owner": owner, "name": name, "cursor": cursor})
                connection = data["repository"]["pullRequests"]
                prs = connection.get("nodes", [])
                page_info = connection["pageInfo"]
                total = connection.get("totalCount", "?")
                page_num += 1

                logger.info("Closed PR list page %d  (%d PRs on page, %s total closed)", page_num, len(prs), total)

                for pr in prs:
                    pr_num = pr["number"]

                    if pr_num in done:
                        logger.debug("  Closed PR #%d already scraped, skipping", pr_num)
                        continue

                    if limit and scraped_count >= limit:
                        logger.info("Reached --limit %d, stopping", limit)
                        break

                    logger.info("  Closed PR #%d  (scraped %d so far)", pr_num, scraped_count + 1)
                    write_pr(pr)

                    base_vars = {"owner": owner, "name": name, "number": pr_num}

                    commits = gh.paginate(PR_COMMITS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "commits"])
                    for c in commits:
                        c["_pr_number"] = pr_num
                        write_commit(c)

                    files = gh.paginate(PR_FILES_QUERY, base_vars,
                                        path=["repository", "pullRequest", "files"])
                    for f in files:
                        f["_pr_number"] = pr_num
                        write_file(f)

                    reviews = gh.paginate(PR_REVIEWS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "reviews"])
                    for r in reviews:
                        r["_pr_number"] = pr_num
                        write_review(r)

                    threads = gh.paginate(PR_REVIEW_THREADS_QUERY, base_vars,
                                          path=["repository", "pullRequest", "reviewThreads"])
                    for t in threads:
                        t["_pr_number"] = pr_num
                        write_thread(t)

                    issue_comments = gh.paginate(PR_ISSUE_COMMENTS_QUERY, base_vars,
                                                 path=["repository", "pullRequest", "comments"])
                    for ic in issue_comments:
                        ic["_pr_number"] = pr_num
                        write_issue_comment(ic)

                    ckpt.mark_pr_done(pr_num)
                    scraped_count += 1

                    if scraped_count % 50 == 0:
                        notify_scrape_progress(repo_display, scraped_count, total)
                        rate_limit_notified = False

                    if gh._rate_remaining is not None and gh._rate_remaining < 200:
                        if not rate_limit_notified:
                            notify_scrape_rate_limited(
                                repo_display, gh._rate_remaining,
                                max((gh._rate_reset or 0) - int(__import__("time").time()), 0),
                            )
                            rate_limit_notified = True

                if limit and scraped_count >= limit:
                    break

                if not page_info["hasNextPage"]:
                    break

                cursor = page_info["endCursor"]
                ckpt.save_pr_list_cursor(cursor)

    except Exception as exc:
        notify_scrape_error(repo_display, str(exc), scraped_count)
        raise
    finally:
        for w in [write_pr, write_commit, write_file, write_review, write_thread, write_issue_comment]:
            w._close()  # type: ignore[attr-defined]

    notify_scrape_done(repo_display, scraped_count)
    logger.info("Closed PR scrape complete — %d new PRs scraped → %s", scraped_count, out)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Scrape merged PRs to bronze JSONL")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub PAT")
    p.add_argument("--limit", type=int, default=None, help="Max PRs to scrape (for testing)")
    p.add_argument("--include-closed", action="store_true", help="Also scrape CLOSED (rejected) PRs")
    args = p.parse_args()

    if not args.token:
        p.error("Provide --token or set GITHUB_TOKEN env var")

    owner, name = args.repo.split("/", 1)
    scrape_repo(owner, name, args.token, limit=args.limit)
    if args.include_closed:
        scrape_repo_closed(owner, name, args.token, limit=args.limit)


if __name__ == "__main__":
    main()
