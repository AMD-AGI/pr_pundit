"""
Stage L2 — Supervisor agent.

For each LineageTree, runs one LLM call to extract an AuditHarness candidate:
an architectural principle that was violated in the failed PRs and correctly
applied in the merged PR.

The supervisor is explicitly told what harnesses already exist and must output
null if the existing harnesses fully cover the pattern.  This is the core
mechanism for learning unknowns only.

Usage (via distill-design-rules CLI):
    distill-design-rules --repo vllm-project/vllm --supervisor-only
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from pipeline.llm import llm_call, make_client, parse_json
from schemas.audit_harness import AuditHarness, HarnessExample, HarnessStatus
from schemas.lineage import LineageTree, PRNode

logger = logging.getLogger(__name__)

LINEAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "lineage"
_DEFAULT_MODEL = "claude-opus-4-7"
_MAX_WORKERS = 4
_MAX_BODY = 800
_MAX_REVIEW = 400
_MAX_FILES = 20


# ── prompt assembly ───────────────────────────────────────────────────

def _format_node(node: PRNode) -> str:
    lines = [
        f"PR #{node.number}: {node.title}",
        f"Files: {node.files_changed[:_MAX_FILES]}",
    ]

    # PR-level reviews: CHANGES_REQUESTED first, then COMMENTED
    cr_reviews = [r for r in (node.reviews or []) if r.state == "CHANGES_REQUESTED"]
    other_reviews = [r for r in (node.reviews or []) if r.state == "COMMENTED" and r.body]
    if cr_reviews:
        lines.append("Review decisions (CHANGES_REQUESTED):")
        for r in cr_reviews[:5]:
            body = r.body[:_MAX_REVIEW].replace("\n", " ")
            lines.append(f"  - {r.reviewer}: {body}")
    if other_reviews and len(cr_reviews) < 2:
        lines.append("Reviewer comments:")
        for r in other_reviews[:4]:
            body = r.body[:_MAX_REVIEW].replace("\n", " ")
            lines.append(f"  - {r.reviewer}: {body}")

    # Inline thread comments with the code hunk they reference
    if node.review_thread_comments:
        lines.append("Inline code review threads:")
        for t in node.review_thread_comments[:8]:
            if t.diff_hunk:
                hunk_preview = t.diff_hunk[-300:].replace("\n", "\n    ")
                lines.append(f"  [{t.path}] code context:\n    {hunk_preview}")
            for c in t.comments[:2]:
                snippet = c[:_MAX_REVIEW].replace("\n", " ")
                lines.append(f"    reviewer: {snippet}")

    return "\n".join(lines)


def _build_prompt(tree: LineageTree, existing_harnesses: list[AuditHarness]) -> str:
    failed_nodes = tree.failed_nodes()
    root = tree.root_node()

    # Order failed nodes chronologically
    failed_section = []
    for i, node in enumerate(failed_nodes, 1):
        failed_section.append(f"[Attempt {i}]\n{_format_node(node)}")
    failed_text = "\n\n".join(failed_section)

    root_text = (
        f"PR #{root.number}: {root.title}\n"
        f"Files: {root.files_changed[:_MAX_FILES]}\n"
        f"Description: {root.body[:_MAX_BODY].replace(chr(10), ' ')}"
    )

    harness_list = ""
    if existing_harnesses:
        items = [
            f"  [{h.harness_id[:8]}] {h.name}: {h.description}"
            for h in existing_harnesses
            if h.status == HarnessStatus.ACTIVE
        ]
        harness_list = "\n".join(items) if items else "  (none yet)"
    else:
        harness_list = "  (none yet)"

    return f"""You are analyzing a PR lineage tree — a chain of failed attempts that
eventually led to a successfully merged PR.

LINEAGE TREE: {tree.tree_id}  (chain depth: {tree.depth})
Repository: {tree.repo}

FAILED ATTEMPTS (earliest first):
{failed_text}

SUCCESSFULLY MERGED PR #{root.number}:
{root_text}

EXISTING AUDIT HARNESSES (already known — do NOT re-derive these):
{harness_list}

TASK:
Identify the single most important architectural principle that would have guided
the authors of the failed PRs toward the correct approach in PR #{root.number}.

Requirements:
- The principle must NOT already be captured by any existing harness above
- It must be CHECKABLE: you must be able to write a concrete LLM prompt that
  takes a PR diff + PR intent and returns specific hints about whether this
  principle is violated
- It must be GENERALIZABLE: applicable to other similar PRs, not just this one
- It must be ARCHITECTURAL: about code design, layer ownership, abstraction
  boundaries — not about syntax, style, or testing conventions
- Deeper chains (depth > 1) indicate harder architectural problems; weight your
  analysis accordingly

If the existing harnesses already fully explain why the failed attempts failed,
output null.

If you identify a genuinely new principle, output JSON with this exact structure:
{{
  "name": "short-kebab-case-name (3-5 words)",
  "description": "2-3 sentences: what architectural pattern this checks and why it matters",
  "relevance_criteria": "1-2 sentences: what kinds of PRs or intents trigger this harness",
  "audit_prompt_template": "Full LLM prompt template that checks a single PR diff.\\nMust include placeholders: {{diff}}, {{intent}}, {{files_changed}}\\nMust instruct the LLM to return JSON: {{\\\"hints\\\": [\\\"...\\\"], \\\"clean\\\": true/false}}\\nBe specific — hints should be actionable rewrites, not vague warnings.",
  "confidence": 0.0-1.0,
  "anti_pattern": "1-2 sentences: what the failed PRs did wrong architecturally",
  "correct_pattern": "1-2 sentences: what the merged PR did correctly"
}}

Output ONLY the JSON object or null. No prose before or after."""


# ── per-tree extraction ───────────────────────────────────────────────

def _extract_candidate(
    tree: LineageTree,
    existing_harnesses: list[AuditHarness],
    model: str,
    client,
) -> dict | None:
    """Run the supervisor LLM call for one tree. Returns raw candidate dict or None."""
    prompt = _build_prompt(tree, existing_harnesses)
    try:
        raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=False)
        stripped = raw.strip()
        if stripped.lower() == "null" or not stripped:
            logger.info("Tree %s → null (existing harnesses cover it)", tree.tree_id)
            return None
        candidate = parse_json(stripped)
        if not isinstance(candidate, dict):
            logger.warning("Tree %s → unexpected output type: %s", tree.tree_id, type(candidate))
            return None
        candidate["_tree_id"] = tree.tree_id
        candidate["_repo"] = tree.repo
        candidate["_depth"] = tree.depth
        candidate["_failed_prs"] = [n.number for n in tree.failed_nodes()]
        candidate["_merged_pr"] = tree.root_pr
        logger.info("Tree %s (depth=%d) → candidate: %s", tree.tree_id, tree.depth,
                    candidate.get("name", "?"))
        return candidate
    except Exception as exc:
        logger.warning("Tree %s → supervisor error: %s", tree.tree_id, exc)
        return None


# ── batch extraction ──────────────────────────────────────────────────

def run_supervisor(
    trees: list[LineageTree],
    existing_harnesses: list[AuditHarness],
    repo_slug: str,
    model: str = _DEFAULT_MODEL,
    workers: int = _MAX_WORKERS,
) -> list[dict]:
    """Run supervisor on all trees; write candidates.jsonl; return non-null candidates."""
    out_dir = LINEAGE_DIR / repo_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "harness_candidates.jsonl"

    client = make_client()

    logger.info("Running supervisor on %d trees (workers=%d) …", len(trees), workers)

    candidates: list[dict] = []

    with open(out_path, "a") as f_out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_tree = {
                executor.submit(_extract_candidate, tree, existing_harnesses, model, client): tree
                for tree in trees
            }
            for future in as_completed(future_to_tree):
                tree = future_to_tree[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Tree %s → unexpected error: %s", tree.tree_id, exc)
                    result = None
                # Always write to JSONL (null as sentinel)
                f_out.write(json.dumps({"tree_id": tree.tree_id, "result": result}, default=str) + "\n")
                f_out.flush()
                if result is not None:
                    candidates.append(result)

    logger.info("Supervisor complete: %d/%d trees produced candidates",
                len(candidates), len(trees))
    return candidates


# ── load existing candidates ──────────────────────────────────────────

def load_candidates(repo_slug: str) -> list[dict]:
    """Load non-null candidates from harness_candidates.jsonl."""
    path = LINEAGE_DIR / repo_slug / "harness_candidates.jsonl"
    if not path.exists():
        return []
    candidates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("result") is not None:
                candidates.append(record["result"])
    return candidates
