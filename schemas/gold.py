"""
Gold layer — distilled merge rules and verification specs.

Every rule is a self-contained JSON-serialisable object that:
  1. States a directive in natural language.
  2. Links back to the raw evidence that justifies it.
  3. Carries a machine-executable verification spec.
  4. Has a non-blocking human_review_status flag (never blocks the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── enums ────────────────────────────────────────────────────────────

class Severity(str, Enum):
    BLOCKER = "blocker"
    STRONG = "strong"
    ADVISORY = "advisory"


class DirectiveType(str, Enum):
    REQUIRED_PATTERN = "required_pattern"
    FORBIDDEN_PATTERN = "forbidden_pattern"
    PREFERRED_PATTERN = "preferred_pattern"
    NAMING_CONVENTION = "naming_convention"
    ERROR_HANDLING = "error_handling"
    TESTING_REQUIREMENT = "testing_requirement"
    SECURITY_RULE = "security_rule"
    PERFORMANCE_RULE = "performance_rule"
    API_CONTRACT = "api_contract"
    PROCESS_REQUIREMENT = "process_requirement"
    DOCUMENTATION = "documentation"
    MAINTAINABILITY = "maintainability"


class VerifierType(str, Enum):
    REGEX = "regex"
    AST = "ast"
    SEMANTIC = "semantic"         # embedding / search based
    TEST_ORACLE = "test_oracle"   # run a test suite
    METADATA = "metadata"         # check labels, files present, etc.
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class HumanReviewStatus(str, Enum):
    """Non-blocking flag set *after* distillation.  Pipeline never waits."""
    PENDING = "pending"           # freshly distilled, not yet reviewed
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_SPLITTING = "needs_splitting"
    DEPRECATED = "deprecated"


class GateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


# ── evidence cluster ─────────────────────────────────────────────────
# Intermediate: groups of review threads expressing the same concern.

@dataclass
class EvidenceCluster:
    cluster_id: str
    canonical_issue: str          # one-sentence summary of the concern
    canonical_fix_pattern: str    # one-sentence summary of the accepted fix
    supporting_example_ids: list[str]   # → silver.ReviewExample.example_id
    supporting_pr_numbers: list[int]
    support_count: int
    contradictory_example_ids: list[str]
    extraction_notes: str | None
    distillation_run_id: str      # links to the run that created this


# ── distilled rule ───────────────────────────────────────────────────

@dataclass
class Rule:
    rule_id: str
    rule_text: str                # concise directive (imperative mood)
    directive_type: DirectiveType
    severity: Severity
    scope: RuleScope
    applicability: RuleApplicability
    rationale: str                # why reviewers enforce this

    # evidence provenance
    evidence_cluster_ids: list[str]
    evidence_pr_numbers: list[int]
    support_count: int            # how many threads back this rule
    extraction_confidence: float  # 0.0–1.0

    # examples (serialisable for LLM context)
    positive_examples: list[RuleExample]
    negative_examples: list[RuleExample]

    # verification
    verifier: VerifierSpec

    # human review — non-blocking post-distillation flag
    human_review_status: HumanReviewStatus = HumanReviewStatus.PENDING
    human_review_notes: str | None = None

    # lifecycle
    rule_version: int = 1
    supersedes_rule_id: str | None = None
    distillation_run_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class RuleScope:
    """Where in the repo this rule applies."""
    repo_wide: bool = True
    path_prefixes: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)  # globs


@dataclass
class RuleApplicability:
    """Conditions under which the rule triggers."""
    when_adding: list[str] = field(default_factory=list)      # e.g. "exported_function"
    when_modifying: list[str] = field(default_factory=list)
    when_deleting: list[str] = field(default_factory=list)
    pr_labels: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)       # prose exceptions


@dataclass
class RuleExample:
    """Concrete code example for LLM context."""
    description: str
    code_before: str | None
    code_after: str | None
    source_pr: int | None         # PR number
    source_path: str | None


# ── verifier spec ────────────────────────────────────────────────────

@dataclass
class VerifierSpec:
    verifier_type: VerifierType
    # deterministic checkers
    pattern: str | None = None             # regex or tree-sitter query
    engine: str | None = None              # e.g. "tree-sitter-python"
    # LLM judge
    prompt_template_id: str | None = None
    required_inputs: list[str] = field(default_factory=list)
    output_schema: str | None = None       # JSON schema for judge output
    confidence_threshold: float = 0.8
    # metadata checks
    metadata_checks: list[str] = field(default_factory=list)


# ── gate evaluation result ───────────────────────────────────────────

@dataclass
class RuleEvalResult:
    """Result of running one rule against a proposed patch."""
    eval_id: str
    rule_id: str
    rule_version: int
    patch_ref: str                # commit SHA or PR reference
    result: GateResult
    confidence: float
    explanation: str
    cited_evidence: list[str]     # rule example IDs or evidence cluster IDs
    evaluated_at: str
    evaluator_version: str        # verifier code/model version
