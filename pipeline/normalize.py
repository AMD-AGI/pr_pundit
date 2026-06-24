"""
Stage B — Normalize bronze → silver.

Reads raw JSONL from data/bronze/{repo}/ and produces:
  • data/silver/{repo}/threads.json    — normalized threads
  • data/silver/{repo}/examples.json   — denormalized review examples

Usage:
    python -m pipeline.normalize --repo owner_name
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

from pipeline.notify import notify_normalize_done

logger = logging.getLogger(__name__)

BRONZE = Path(__file__).resolve().parent.parent / "data" / "bronze"
SILVER = Path(__file__).resolve().parent.parent / "data" / "silver"

LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".kt": "kotlin",
    ".swift": "swift", ".sh": "shell", ".md": "markdown",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _lang(path: str | None) -> str | None:
    if not path:
        return None
    for ext, lang in LANG_MAP.items():
        if path.endswith(ext):
            return lang
    return None


def _parse_diff_hunks(patch_text: str | None) -> list[dict]:
    """Split a unified diff patch into individual hunks."""
    if not patch_text:
        return []
    hunks = []
    current: dict | None = None
    for line in patch_text.splitlines(keepends=True):
        m = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)", line)
        if m:
            if current:
                hunks.append(current)
            current = {
                "old_start": int(m.group(1)),
                "old_lines": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_lines": int(m.group(4) or 1),
                "header": line.rstrip(),
                "body": "",
            }
        elif current is not None:
            current["body"] += line
    if current:
        hunks.append(current)
    return hunks


def normalize(repo_slug: str):
    src = BRONZE / repo_slug
    dst = SILVER / repo_slug
    dst.mkdir(parents=True, exist_ok=True)

    prs = {pr["number"]: pr for pr in _read_jsonl(src / "pull_requests.jsonl")}
    threads_raw = _read_jsonl(src / "review_threads.jsonl")
    files_raw = _read_jsonl(src / "files.jsonl")
    reviews_raw = _read_jsonl(src / "reviews.jsonl")

    # index file patches by (pr_number, path)
    patches: dict[tuple[int, str], str] = {}
    for f in files_raw:
        key = (f["_pr_number"], f.get("path", ""))
        patches[key] = f.get("patch", "") or ""

    # index reviews by id
    review_states: dict[str, str] = {}
    for r in reviews_raw:
        review_states[r.get("id", "")] = r.get("state", "")

    normalized_threads: list[dict] = []
    examples: list[dict] = []

    for t in threads_raw:
        pr_num = t["_pr_number"]
        pr = prs.get(pr_num, {})
        path = t.get("path")

        # build comment list
        comments = []
        for c in t.get("comments", {}).get("nodes", []):
            rev = c.get("pullRequestReview") or {}
            comments.append({
                "comment_id": c.get("id"),
                "author": (c.get("author") or {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("createdAt"),
                "is_reply": c.get("replyTo") is not None,
                "review_state": rev.get("state"),
            })

        # linked hunk
        patch_text = patches.get((pr_num, path or ""), "")
        hunks = _parse_diff_hunks(patch_text)
        target_line = t.get("originalLine") or t.get("line")
        matched_hunk = None
        if target_line and hunks:
            for h in hunks:
                if h["new_start"] <= target_line <= h["new_start"] + h["new_lines"]:
                    matched_hunk = h
                    break
            if not matched_hunk:
                matched_hunk = hunks[0]  # fallback to first hunk

        thread_obj = {
            "thread_id": t.get("id"),
            "pr_id": pr.get("id"),
            "pr_number": pr_num,
            "pr_title": pr.get("title", ""),
            "pr_author": (pr.get("author") or {}).get("login", "unknown"),
            "repo": repo_slug.replace("_", "/", 1),
            "path": path,
            "line": t.get("line"),
            "start_line": t.get("startLine"),
            "diff_side": t.get("diffSide"),
            "is_resolved": t.get("isResolved", False),
            "resolved_by": (t.get("resolvedBy") or {}).get("login"),
            "is_outdated": t.get("isOutdated", False),
            "hunk_header": matched_hunk["header"] if matched_hunk else None,
            "hunk_body": matched_hunk["body"] if matched_hunk else None,
            "comments": comments,
        }
        normalized_threads.append(thread_obj)

        # build review example if thread has at least one non-author comment
        pr_author = thread_obj["pr_author"]
        reviewer = next((c["author"] for c in comments if c["author"] != pr_author), None)
        if reviewer and path:
            example = {
                "example_id": f"{repo_slug}_{pr_num}_{t.get('id', '')}",
                "repo": thread_obj["repo"],
                "pr_number": pr_num,
                "pr_title": pr.get("title", ""),
                "pr_description": pr.get("body", ""),
                "pr_author": pr_author,
                "pr_labels": [l["name"] for l in (pr.get("labels", {}).get("nodes") or [])],
                "path": path,
                "language": _lang(path),
                "hunk_before": matched_hunk["body"] if matched_hunk else None,
                "file_context": patch_text[:2000] if patch_text else None,
                "thread_id": t.get("id"),
                "comments": comments,
                "reviewer": reviewer,
                "is_resolved": t.get("isResolved", False),
                "code_changed_after_comment": t.get("isOutdated", False),
                "final_review_state": pr.get("reviewDecision"),
                "pr_merged": True,
            }
            examples.append(example)

    # write outputs
    (dst / "threads.json").write_text(json.dumps(normalized_threads, indent=2, default=str))
    (dst / "examples.json").write_text(json.dumps(examples, indent=2, default=str))

    logger.info("Silver: %d threads, %d examples → %s", len(normalized_threads), len(examples), dst)
    notify_normalize_done(repo_slug.replace("_", "/", 1), len(normalized_threads), len(examples))


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Normalize bronze → silver")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    args = p.parse_args()
    normalize(args.repo)


if __name__ == "__main__":
    main()
