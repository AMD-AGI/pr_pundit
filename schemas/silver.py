"""
Silver layer — normalized, linkable records ready for analysis.

Silver records are derived deterministically from bronze.  They add:
  • stable synthetic IDs for entities GitHub doesn't ID (hunks)
  • parsed / split diff hunks
  • resolved thread ↔ hunk linkage
  • denormalized "review example" rows for distillation input
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Normalized review thread with linked hunk context ────────────────

@dataclass
class NormalizedThread:
    thread_id: str
    pr_id: str
    pr_number: int
    pr_title: str
    pr_author: str
    repo: str                     # owner/name

    # location
    path: str | None
    line: int | None
    start_line: int | None
    diff_side: str | None

    # resolution
    is_resolved: bool
    resolved_by: str | None
    is_outdated: bool

    # linked hunk context (filled during normalization)
    hunk_header: str | None
    hunk_body: str | None         # diff text the comment anchors to
    file_context_before: str | None   # ~10 lines before hunk
    file_context_after: str | None    # ~10 lines after hunk

    # all comments in chronological order
    comments: list[ThreadComment] = field(default_factory=list)


@dataclass
class ThreadComment:
    comment_id: str
    author: str
    body: str                     # raw markdown preserved
    created_at: str
    is_reply: bool
    review_state: str | None      # state of the parent review


# ── Denormalized review example (one per actionable thread) ──────────
# This is the primary input format for the distillation step.

@dataclass
class ReviewExample:
    example_id: str               # synthetic
    repo: str
    pr_number: int
    pr_title: str
    pr_description: str
    pr_author: str
    pr_labels: list[str]

    # code context
    path: str
    language: str | None          # inferred from extension
    hunk_before: str | None       # diff hunk text
    hunk_after: str | None        # code state after resolution
    file_context: str | None      # surrounding lines

    # review conversation
    thread_id: str
    comments: list[ThreadComment]
    reviewer: str                 # first non-author commenter
    is_resolved: bool
    code_changed_after_comment: bool  # did the relevant hunk change?

    # outcome
    final_review_state: str | None    # APPROVED after fix?
    pr_merged: bool
