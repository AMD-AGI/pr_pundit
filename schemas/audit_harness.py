"""
Audit harness layer — architecture principles discovered from PR lineage trees.

An AuditHarness is an executable LLM-based check extracted from a LineageTree.
It encodes a structural principle that was violated in the failed PRs and
correctly applied in the merged PR.

Unlike Rules (which check code patterns mechanically), harnesses are:
  - Higher-level: architecture and design philosophy, not syntax patterns
  - LLM-executed: the audit_prompt_template is rendered and called at runtime
  - Discovered: categories are not pre-defined; the LLM names each harness
  - Incremental: new harnesses are admitted only if they add a genuinely new
    dimension not already covered by existing harnesses (LLM sequential review)

Harnesses plug into the rewrite loop: after each file rewrite, the pipeline
selects relevant harnesses for this PR series (one LLM call), runs them
sequentially (one at a time, fixing incrementally), then runs the existing
wholistic audit + critic pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── enums ────────────────────────────────────────────────────────────

class HarnessStatus(str, Enum):
    ACTIVE     = "active"
    DEPRECATED = "deprecated"
    REJECTED   = "rejected"


# ── supporting types ─────────────────────────────────────────────────

@dataclass
class LineageRef:
    """Reference to the lineage tree that motivated this harness."""
    tree_id: str        # e.g. "vllm-project_vllm-37646"
    repo: str           # e.g. "vllm-project/vllm"
    depth: int          # chain length (depth=3 means 2 failed attempts before merge)
    failed_prs: list[int]   # PR numbers of failed attempts, chronological
    merged_pr: int      # PR number of the successfully merged PR

    def to_dict(self) -> dict:
        return {
            "tree_id": self.tree_id,
            "repo": self.repo,
            "depth": self.depth,
            "failed_prs": self.failed_prs,
            "merged_pr": self.merged_pr,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageRef:
        return cls(
            tree_id=d["tree_id"],
            repo=d.get("repo", ""),
            depth=d.get("depth", 1),
            failed_prs=d.get("failed_prs", []),
            merged_pr=d.get("merged_pr", 0),
        )


@dataclass
class HarnessExample:
    """Concrete (failed → correct) example that motivated the harness."""
    description: str
    anti_pattern: str          # prose: what the failed PR did wrong
    correct_pattern: str       # prose: what the merged PR did right
    source_tree_id: str
    source_failed_pr: int
    source_merged_pr: int

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "anti_pattern": self.anti_pattern,
            "correct_pattern": self.correct_pattern,
            "source_tree_id": self.source_tree_id,
            "source_failed_pr": self.source_failed_pr,
            "source_merged_pr": self.source_merged_pr,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HarnessExample:
        return cls(**d)


# ── harness ──────────────────────────────────────────────────────────

@dataclass
class AuditHarness:
    """An LLM-callable architecture check discovered from lineage trees.

    audit_prompt_template is a Python str.format()-compatible template.
    The pipeline renders it with:
      {diff}          — unified diff of the PR being checked
      {intent}        — PR objective from the planner
      {files_changed} — list of files modified in this PR

    The prompt must instruct the LLM to return JSON:
      {"hints": ["hint1", "hint2", ...], "clean": true/false}

    hints are fed directly into rewrite_file_for_pr() via audit_hints.
    """
    harness_id: str
    name: str                       # short kebab-case, e.g. "compiler-pass-locus"
    description: str                # what architectural pattern this checks (2-3 sentences)
    relevance_criteria: str         # prose: when to apply (LLM uses this to decide)
    audit_prompt_template: str      # template rendered at runtime; see docstring above
    lineage_refs: list[LineageRef]  # which trees produced this harness
    examples: list[HarnessExample]  # concrete (failed → correct) examples

    status: HarnessStatus = HarnessStatus.ACTIVE
    pre_push_only: bool = False         # skip rewrite loop; surface as post-push checklist item
    supersedes_harness_id: str | None = None
    harness_version: int = 1
    distillation_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def discovered_from(self) -> list[str]:
        """Tree IDs — derived from lineage_refs for backwards compatibility."""
        return [r.tree_id for r in self.lineage_refs]

    @property
    def supporting_tree_count(self) -> int:
        return len(self.lineage_refs)

    def to_dict(self) -> dict:
        return {
            "harness_id": self.harness_id,
            "name": self.name,
            "description": self.description,
            "relevance_criteria": self.relevance_criteria,
            "audit_prompt_template": self.audit_prompt_template,
            "lineage_refs": [r.to_dict() for r in self.lineage_refs],
            "examples": [e.to_dict() for e in self.examples],
            "status": self.status.value,
            "pre_push_only": self.pre_push_only,
            "supersedes_harness_id": self.supersedes_harness_id,
            "harness_version": self.harness_version,
            "distillation_run_id": self.distillation_run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AuditHarness:
        # Handle old format: rebuild lineage_refs from discovered_from if needed
        if "lineage_refs" in d:
            lineage_refs = [LineageRef.from_dict(r) for r in d["lineage_refs"]]
        else:
            lineage_refs = [
                LineageRef(tree_id=tid, repo="", depth=1, failed_prs=[], merged_pr=0)
                for tid in d.get("discovered_from", [])
            ]

        # Handle old status values (pending → active, accepted → active)
        raw_status = d.get("status", "active")
        if raw_status in ("pending", "accepted"):
            raw_status = "active"
        status = HarnessStatus(raw_status)

        return cls(
            harness_id=d["harness_id"],
            name=d["name"],
            description=d["description"],
            relevance_criteria=d["relevance_criteria"],
            audit_prompt_template=d["audit_prompt_template"],
            lineage_refs=lineage_refs,
            examples=[HarnessExample.from_dict(e) for e in d.get("examples", [])],
            status=status,
            pre_push_only=d.get("pre_push_only", False),
            supersedes_harness_id=d.get("supersedes_harness_id"),
            harness_version=d.get("harness_version", 1),
            distillation_run_id=d.get("distillation_run_id"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )
