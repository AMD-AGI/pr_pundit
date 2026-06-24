"""
Bronze layer — raw GitHub API response schemas.

These dataclasses mirror what we store verbatim from the GraphQL scrape.
Every record keeps a `raw_payload` dict so we never lose fields we
didn't model yet.  Bronze data is append-only and never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── repository ───────────────────────────────────────────────────────

@dataclass
class Repository:
    repo_id: str
    owner: str
    name: str
    default_branch: str
    url: str
    scraped_at: str          # ISO-8601
    schema_version: int = 1
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── pull request ─────────────────────────────────────────────────────

@dataclass
class PullRequest:
    pr_id: str               # GitHub node ID
    number: int
    repo_id: str
    title: str
    author: str
    state: str               # MERGED
    created_at: str
    merged_at: str | None
    closed_at: str | None
    base_branch: str
    head_branch: str
    merge_commit_sha: str | None
    description_raw: str      # full markdown body
    labels: list[str]
    additions: int
    deletions: int
    changed_files: int
    review_decision: str | None  # APPROVED, CHANGES_REQUESTED, …
    url: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── commit ───────────────────────────────────────────────────────────

@dataclass
class Commit:
    sha: str
    pr_id: str
    author: str
    committed_at: str
    message: str
    parents: list[str]
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── file change ──────────────────────────────────────────────────────

@dataclass
class FileChange:
    file_change_id: str       # synthetic: pr_id + path
    pr_id: str
    path: str
    previous_path: str | None
    change_type: str          # ADDED, DELETED, MODIFIED, RENAMED
    additions: int
    deletions: int
    patch_text: str | None    # unified diff text
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── diff hunk ────────────────────────────────────────────────────────

@dataclass
class DiffHunk:
    hunk_id: str              # synthetic: file_change_id + ordinal
    file_change_id: str
    path: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    header: str               # @@ line
    body: str                 # hunk text
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── review ───────────────────────────────────────────────────────────

@dataclass
class Review:
    review_id: str            # GitHub node ID
    pr_id: str
    reviewer: str
    submitted_at: str
    state: str                # APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED
    body: str | None
    commit_sha: str | None
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── review thread ────────────────────────────────────────────────────

@dataclass
class ReviewThread:
    thread_id: str            # GitHub node ID
    pr_id: str
    path: str | None
    line: int | None
    original_line: int | None
    start_line: int | None
    original_start_line: int | None
    diff_side: str | None     # LEFT or RIGHT
    is_resolved: bool
    is_outdated: bool
    resolved_by: str | None
    subject_type: str | None  # LINE, FILE
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── review comment ───────────────────────────────────────────────────

@dataclass
class ReviewComment:
    comment_id: str           # GitHub node ID
    thread_id: str | None
    review_id: str | None
    pr_id: str
    author: str
    body_raw: str             # full markdown
    created_at: str
    updated_at: str | None
    in_reply_to_id: str | None
    path: str | None
    line: int | None
    original_line: int | None
    diff_side: str | None
    commit_sha: str | None
    original_commit_sha: str | None
    diff_hunk: str | None     # GitHub-returned context hunk
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ── issue (top-level PR) comment ─────────────────────────────────────

@dataclass
class IssueComment:
    comment_id: str
    pr_id: str
    author: str
    body_raw: str
    created_at: str
    updated_at: str | None
    raw_payload: dict[str, Any] = field(default_factory=dict)
