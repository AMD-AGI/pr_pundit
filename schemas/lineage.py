"""
Lineage layer — PR evolution trees mined from bronze data.

A LineageTree captures the full chain of failed attempts (CLOSED PRs) that
preceded a successfully merged PR.  Multiple closed PRs can converge on one
merged PR; chains can be multi-hop (A → B → merged C).

Trees are the input to the supervisor agent, which extracts architecture audit
harness candidates from each tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── enums ────────────────────────────────────────────────────────────

class LineageEdgeSignal(str, Enum):
    body_cross_reference = "body_cross_reference"
    revert_pair          = "revert_pair"
    topic_similarity     = "topic_similarity"


# ── nodes ────────────────────────────────────────────────────────────

@dataclass
class PRReview:
    reviewer: str
    state: str          # APPROVED | CHANGES_REQUESTED | COMMENTED
    body: str
    submitted_at: str


@dataclass
class ReviewThreadComment:
    path: str
    diff_hunk: str      # the code context the reviewer was looking at
    comments: list[str] # comment bodies in thread order


@dataclass
class PRNode:
    repo: str
    number: int
    title: str
    body: str
    author: str
    state: str                           # "CLOSED" or "MERGED"
    created_at: str
    closed_at: str | None
    merged_at: str | None
    files_changed: list[str]
    reviews: list[PRReview] = field(default_factory=list)
    review_thread_comments: list[ReviewThreadComment] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "state": self.state,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "merged_at": self.merged_at,
            "files_changed": self.files_changed,
            "reviews": [
                {"reviewer": r.reviewer, "state": r.state, "body": r.body, "submitted_at": r.submitted_at}
                for r in self.reviews
            ],
            "review_thread_comments": [
                {"path": t.path, "diff_hunk": t.diff_hunk, "comments": t.comments}
                for t in self.review_thread_comments
            ],
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PRNode:
        # backward-compat: old trees stored rejection_reviews without state
        raw_reviews = d.get("reviews") or [
            {**r, "state": "CHANGES_REQUESTED"} for r in d.get("rejection_reviews", [])
        ]
        return cls(
            repo=d["repo"],
            number=d["number"],
            title=d["title"],
            body=d["body"],
            author=d["author"],
            state=d["state"],
            created_at=d["created_at"],
            closed_at=d.get("closed_at"),
            merged_at=d.get("merged_at"),
            files_changed=d.get("files_changed", []),
            reviews=[PRReview(**r) for r in raw_reviews],
            review_thread_comments=[
                ReviewThreadComment(**t) if "diff_hunk" in t
                else ReviewThreadComment(path=t["path"], diff_hunk="", comments=t["comments"])
                for t in d.get("review_thread_comments", [])
            ],
            url=d.get("url", ""),
        )


# ── edges ────────────────────────────────────────────────────────────

@dataclass
class LineageEdge:
    from_pr: int              # earlier / failed PR number
    to_pr: int                # successor PR number (closer to root)
    signal: LineageEdgeSignal
    confidence: float

    def to_dict(self) -> dict:
        return {
            "from_pr": self.from_pr,
            "to_pr": self.to_pr,
            "signal": self.signal.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageEdge:
        return cls(
            from_pr=d["from_pr"],
            to_pr=d["to_pr"],
            signal=LineageEdgeSignal(d["signal"]),
            confidence=d["confidence"],
        )


# ── tree ─────────────────────────────────────────────────────────────

@dataclass
class LineageTree:
    """Full evolution chain leading to a successfully merged PR.

    depth = longest path from any leaf (most-failed attempt) to root.
    A depth-1 tree has one closed PR directly preceding the merged PR.
    A depth-2+ tree has a chain: closed A → closed B → … → merged PR.
    Deeper trees signal harder architectural problems and richer lessons.
    """
    tree_id: str                      # "{repo_slug}-{root_pr}", e.g. "vllm-37646"
    repo: str
    root_pr: int                      # the final merged PR at the top of the tree
    nodes: dict[int, PRNode]          # all PRs keyed by number
    edges: list[LineageEdge]          # directed from_pr → to_pr (toward root)
    depth: int                        # longest path from leaf to root
    created_at: str

    def failed_nodes(self) -> list[PRNode]:
        """Return all CLOSED nodes, ordered by created_at ascending."""
        closed = [n for n in self.nodes.values() if n.state == "CLOSED"]
        return sorted(closed, key=lambda n: n.created_at)

    def root_node(self) -> PRNode:
        return self.nodes[self.root_pr]

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "repo": self.repo,
            "root_pr": self.root_pr,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "depth": self.depth,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageTree:
        nodes = {int(k): PRNode.from_dict(v) for k, v in d["nodes"].items()}
        return cls(
            tree_id=d["tree_id"],
            repo=d["repo"],
            root_pr=d["root_pr"],
            nodes=nodes,
            edges=[LineageEdge.from_dict(e) for e in d.get("edges", [])],
            depth=d.get("depth", 1),
            created_at=d["created_at"],
        )
