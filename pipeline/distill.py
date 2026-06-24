"""
Stage C — Distill silver examples → gold rules.

Reads data/silver/{repo}/examples.json and produces:
  • data/gold/{repo}/reviewer_authority.json — reviewer scores (computed first)
  • data/gold/{repo}/clusters.json           — evidence clusters
  • data/gold/{repo}/rules.json             — distilled rules (human_review_status=pending)

The distiller uses an LLM to:
  1. Filter out non-normative comments (pure chat, approvals, questions).
  2. Cluster remaining comments by intent.
  3. Promote each cluster into a candidate rule with verifier spec.

Human review is a *post-distillation flag* — the pipeline never blocks.

Usage:
    python -m pipeline.distill --repo owner_name [--provider openai|anthropic]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.notify import (
    notify_distill_done,
    notify_distill_error,
    notify_distill_filter_progress,
    notify_distill_start,
)

logger = logging.getLogger(__name__)

SILVER = Path(__file__).resolve().parent.parent / "data" / "silver"
GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

# ── LLM interaction ─────────────────────────────────────────────────
# Kept minimal: callers can swap in any OpenAI-compatible or Anthropic
# client.  The prompts are the important part.

FILTER_PROMPT = """\
You are an expert code reviewer analyst specializing in the following repository:

Repository: {repo_description}

Focus areas for this repository:
{focus_areas}

Given the following review comment thread from a merged pull request,
decide TWO things:

1. Does it contain a **normative directive** — a statement that implies a rule
   or requirement that future code changes should follow?

2. Is this directive **repository-specific** — something unique to this codebase's
   architecture, domain, or conventions that a generic coding assistant would NOT
   already know?

REJECT generic best practices that any linter or coding agent already enforces:
- "add comments", "add tests", "improve naming", "handle errors"
- "reduce duplication", "follow style guide", "add type hints"
- "remove dead code", "use constants instead of magic numbers"

KEEP rules specific to this project's domain, architecture, or conventions:
- Correct usage of project-specific APIs, configs, or abstractions
- Platform-specific constraints (GPU backends, hardware compatibility)
- Domain-specific correctness (parallelism strategies, memory management)
- Project-specific patterns that differ from common practice

Return ONLY JSON:
{{"is_normative": true/false, "is_repo_specific": true/false, "reason": "..."}}

Weight the reviewer's authority when evaluating — directives from core maintainers
and frequent contributors are stronger signals than those from occasional contributors.

---
PR: #{pr_number} {pr_title}
File: {path}
Reviewer: {reviewer} ({reviewer_merged_prs} merged PRs, {reviewer_reviews} reviews — {reviewer_tier}{reviewer_amd})

Diff hunk:
```
{hunk}
```

Comment thread:
{thread_text}
"""


def _load_repo_config(repo_slug: str) -> dict:
    """Load repo_config.yaml if it exists, else return defaults."""
    import yaml
    config_path = GOLD / repo_slug / "repo_config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {"name": repo_slug.replace("_", "/", 1), "description": "", "focus_keywords": [], "focus_areas": []}


def _load_authority(repo_slug: str) -> dict[str, dict]:
    """Load reviewer authority. Returns {login: {merged_prs, reviews_given, tier}}."""
    path = GOLD / repo_slug / "reviewer_authority.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("users", {})

CLUSTER_PROMPT = """\
You are a senior engineering manager distilling recurring code review
feedback into a structured knowledge base.

Below are {count} review examples from the same repository.  Each one
was marked as containing a normative directive.

Group them into clusters where each cluster represents ONE distinct
merge rule.  For each cluster output:

{{
  "cluster_id": "<uuid>",
  "canonical_issue": "one sentence: what reviewers flag",
  "canonical_fix_pattern": "one sentence: what the accepted fix looks like",
  "supporting_example_ids": ["..."],
  "support_count": <int>,
  "contradictory_example_ids": []
}}

Return a JSON array of clusters.

---
Examples:
{examples_json}
"""

RULE_PROMPT = """\
You are a principal engineer writing verifiable merge rules for a specific repository.

Repository: {repo_description}

Focus areas:
{focus_areas}

Given the following evidence cluster, produce ONE rule object.
The rule MUST be specific to this repository's domain and architecture — not a generic coding best practice.

{scope_guidance}

IMPORTANT: Set repo_wide to false and specify precise path_prefixes and languages
based on the evidence. Narrow scope prevents false positives on unrelated files.

{{
  "rule_id": "<uuid>",
  "rule_text": "imperative directive (e.g., 'Always add tests for new public APIs')",
  "directive_type": one of {directive_types},
  "severity": "blocker" | "strong" | "advisory",
  "scope": {{
    "repo_wide": true/false,
    "path_prefixes": [...],
    "languages": [...],
    "file_patterns": [...]
  }},
  "applicability": {{
    "when_adding": [...],
    "when_modifying": [...],
    "when_deleting": [...],
    "pr_labels": [],
    "exceptions": [...]
  }},
  "rationale": "why reviewers enforce this",
  "positive_examples": [
    {{"description": "...", "code_before": "...", "code_after": "...", "source_pr": <int>, "source_path": "..."}}
  ],
  "negative_examples": [...],
  "verifier": {{
    "verifier_type": "regex" | "ast" | "semantic" | "llm_judge" | "metadata" | "hybrid",
    "pattern": "...",
    "engine": "...",
    "prompt_template_id": null,
    "confidence_threshold": 0.8
  }},
  "extraction_confidence": 0.0-1.0
}}

Evidence cluster:
{cluster_json}

Supporting examples:
{examples_json}
"""


from pipeline.llm import make_client as _make_client
from pipeline.llm import llm_call as _llm_call_raw
from pipeline.llm import parse_json as _parse_json


def _llm_call(prompt: str, provider: str, model: str, *, client=None, max_tokens: int = 16384) -> str:
    return _llm_call_raw(prompt, model, client=client, max_tokens=max_tokens)


# ── pipeline steps ───────────────────────────────────────────────────

def _filter_one(
    ex: dict, provider: str, model: str, client,
    repo_config: dict, authority: dict[str, dict],
) -> tuple[str, bool]:
    """Filter a single example. Returns (example_id, keep)."""
    thread_text = "\n".join(
        f"  {c['author']}: {c['body']}" for c in ex.get("comments", [])
    )
    focus_areas = "\n".join(f"- {a}" for a in repo_config.get("focus_areas", []))
    reviewer = ex.get("reviewer", "")
    rev_info = authority.get(reviewer, {})
    prompt = FILTER_PROMPT.format(
        repo_description=repo_config.get("description", ""),
        focus_areas=focus_areas or "No specific focus areas configured.",
        pr_number=ex["pr_number"],
        pr_title=ex["pr_title"],
        path=ex["path"],
        reviewer=reviewer,
        reviewer_merged_prs=rev_info.get("merged_prs", 0),
        reviewer_reviews=rev_info.get("reviews_given", 0),
        reviewer_tier=rev_info.get("tier", "unknown"),
        reviewer_amd=", AMD org member" if rev_info.get("amd_org_member") else "",
        hunk=ex.get("hunk_before", "") or "",
        thread_text=thread_text,
    )
    try:
        raw = _llm_call(prompt, provider, model, client=client, max_tokens=512)
        result = _parse_json(raw)
        keep = bool(result.get("is_normative")) and bool(result.get("is_repo_specific"))
        return (ex["example_id"], keep)
    except Exception:
        logger.warning("Filter failed for example %s, skipping", ex["example_id"])
        return (ex["example_id"], False)


def _load_filter_checkpoint(path: Path) -> dict[str, bool]:
    """Load existing filter checkpoint. Returns {example_id: is_normative}."""
    done: dict[str, bool] = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["example_id"]] = rec["is_normative"]
    return done


def _filter_normative(
    examples: list[dict], provider: str, model: str, repo_slug: str,
    workers: int = 10, repo_config: dict | None = None,
) -> list[dict]:
    """Step 1: keep only examples with repo-specific normative content."""
    repo_config = repo_config or _load_repo_config(repo_slug)
    authority = _load_authority(repo_slug)
    if authority:
        logger.info("Loaded authority data for %d users", len(authority))
    ckpt_path = GOLD / repo_slug / "filter_checkpoint.jsonl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_filter_checkpoint(ckpt_path)

    todo = [ex for ex in examples if ex["example_id"] not in done]
    logger.info(
        "Normative filter: %d already checkpointed, %d to process (%d workers)",
        len(done), len(todo), workers,
    )

    total_count = len(examples)
    checkpoint_count = len(done)
    repo_display = repo_slug.replace("_", "/", 1)

    if todo:
        client = _make_client()
        lock = threading.Lock()
        completed = 0

        def _process(ex):
            nonlocal completed
            eid, is_norm = _filter_one(ex, provider, model, client, repo_config, authority)
            with lock:
                done[eid] = is_norm
                with open(ckpt_path, "a") as f:
                    f.write(json.dumps({"example_id": eid, "is_normative": is_norm}) + "\n")
                completed += 1
                if completed % 100 == 0:
                    kept_so_far = sum(1 for v in done.values() if v)
                    total_done = len(done)
                    logger.info("  filter progress: %d / %d (%d unique IDs)", completed, len(todo), total_done)
                    notify_distill_filter_progress(
                        repo_display, completed + checkpoint_count, total_count, kept_so_far,
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, ex): ex for ex in todo}
            for fut in concurrent.futures.as_completed(futures):
                exc = fut.exception()
                if exc:
                    ex = futures[fut]
                    logger.error("Unexpected error filtering %s: %s", ex["example_id"], exc)

    kept = [ex for ex in examples if done.get(ex["example_id"], False)]
    logger.info("Normative filter: %d / %d kept", len(kept), len(examples))
    return kept


def _cluster_one_batch(
    batch_idx: int, batch: list[dict], provider: str, model: str, run_id: str,
) -> tuple[int, list[dict]]:
    """Cluster a single batch. Returns (batch_idx, clusters)."""
    slim = [
        {
            "example_id": e["example_id"],
            "pr_number": e["pr_number"],
            "path": e["path"],
            "language": e.get("language"),
            "reviewer": e.get("reviewer"),
            "comments": [c["body"] for c in e.get("comments", [])],
            "hunk": (e.get("hunk_before") or "")[:500],
            "is_resolved": e.get("is_resolved"),
        }
        for e in batch
    ]
    prompt = CLUSTER_PROMPT.format(
        count=len(slim),
        examples_json=json.dumps(slim, indent=2),
    )
    raw = _llm_call(prompt, provider, model, max_tokens=16384)
    clusters = _parse_json(raw)
    if not isinstance(clusters, list):
        clusters = [clusters] if isinstance(clusters, dict) else []
    valid = []
    for cl in clusters:
        if not isinstance(cl, dict):
            continue
        cl["distillation_run_id"] = run_id
        if "cluster_id" not in cl:
            cl["cluster_id"] = str(uuid.uuid4())
        valid.append(cl)
    return (batch_idx, valid)


def _cluster_examples(
    examples: list[dict], provider: str, model: str, run_id: str,
    repo_slug: str, workers: int = 10,
) -> list[dict]:
    """Step 2: group normative examples into evidence clusters."""
    batch_size = 40
    ckpt_path = GOLD / repo_slug / "cluster_checkpoint.jsonl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    done_batches: dict[int, list[dict]] = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done_batches[rec["batch_index"]] = rec["clusters"]

    batches = []
    for i in range(0, len(examples), batch_size):
        batch_idx = i // batch_size
        batches.append((batch_idx, examples[i : i + batch_size]))

    todo = [(idx, b) for idx, b in batches if idx not in done_batches]
    total_batches = len(batches)
    logger.info(
        "Clustering: %d batches total, %d already checkpointed, %d to process (%d workers)",
        total_batches, len(done_batches), len(todo), workers,
    )

    failed_batches: list[int] = []

    if todo:
        lock = threading.Lock()
        completed = 0

        def _process(batch_idx: int, batch: list[dict]):
            nonlocal completed
            idx, clusters = _cluster_one_batch(batch_idx, batch, provider, model, run_id)
            with lock:
                done_batches[idx] = clusters
                with open(ckpt_path, "a") as f:
                    f.write(json.dumps({"batch_index": idx, "clusters": clusters}) + "\n")
                completed += 1
                logger.info("  cluster progress: %d / %d batches done", completed + (total_batches - len(todo)), total_batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, idx, batch): idx for idx, batch in todo}
            for fut in concurrent.futures.as_completed(futures):
                exc = fut.exception()
                if exc:
                    batch_idx = futures[fut]
                    logger.warning("Cluster batch %d failed: %s", batch_idx, exc)
                    failed_batches.append(batch_idx)

    all_clusters: list[dict] = []
    for idx, _ in batches:
        if idx in done_batches:
            all_clusters.extend(done_batches[idx])

    if failed_batches:
        logger.warning("Failed batches: %s (re-run to retry)", failed_batches)
    logger.info("Clustered into %d evidence groups (%d batches failed)", len(all_clusters), len(failed_batches))
    return all_clusters


# ── cluster dedup ───────────────────────────────────────────────────

STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "of", "to", "for",
    "is", "it", "be", "as", "at", "by", "this", "that", "with", "from",
    "are", "was", "were", "been", "has", "have", "had", "do", "does",
    "not", "no", "all", "any", "can", "will", "should", "must", "when",
    "if", "so", "than", "then", "use", "using", "used", "also", "each",
    "which", "their", "its", "into", "such", "only", "other", "more",
    "new", "may", "e.g.", "i.e.", "etc", "e.g", "ie", "eg",
})


def _tokenize(text: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9_]+", " ", text.lower()).split() if w not in STOP_WORDS and len(w) > 1}


def _dedup_clusters(clusters: list[dict], threshold: float = 0.5) -> list[dict]:
    """Merge duplicate clusters using Jaccard similarity on canonical_issue."""
    if not clusters:
        return clusters

    tokens = [_tokenize(cl.get("canonical_issue", "")) for cl in clusters]

    # union-find
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # inverted index for fast candidate lookup
    index: dict[str, list[int]] = {}
    for i, toks in enumerate(tokens):
        for t in toks:
            index.setdefault(t, []).append(i)

    # compare candidates sharing tokens
    seen_pairs: set[tuple[int, int]] = set()
    for _, indices in index.items():
        if len(indices) > 200:
            continue
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                a, b = indices[ii], indices[jj]
                pair = (min(a, b), max(a, b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                ta, tb = tokens[a], tokens[b]
                if not ta or not tb:
                    continue
                jaccard = len(ta & tb) / len(ta | tb)
                if jaccard >= threshold:
                    union(a, b)

    # group by root
    groups: dict[int, list[int]] = {}
    for i in range(len(clusters)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # merge each group
    merged: list[dict] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(clusters[members[0]])
            continue

        group = [clusters[i] for i in members]
        best = max(group, key=lambda c: c.get("support_count", 0))

        all_example_ids: list[str] = []
        all_contradictory: list[str] = []
        total_support = 0
        for cl in group:
            all_example_ids.extend(cl.get("supporting_example_ids", []))
            all_contradictory.extend(cl.get("contradictory_example_ids", []))
            total_support += cl.get("support_count", 0)

        merged_cl = dict(best)
        merged_cl["supporting_example_ids"] = list(dict.fromkeys(all_example_ids))
        merged_cl["contradictory_example_ids"] = list(dict.fromkeys(all_contradictory))
        merged_cl["support_count"] = total_support
        merged.append(merged_cl)

    logger.info("Dedup: %d clusters → %d (merged %d duplicates)",
                len(clusters), len(merged), len(clusters) - len(merged))
    return merged


def _promote_one(
    cl: dict, examples_by_id: dict, directive_types: list[str],
    provider: str, model: str, run_id: str, repo_config: dict | None = None,
) -> dict:
    """Promote a single cluster into a rule."""
    repo_config = repo_config or {}
    supporting = [
        examples_by_id[eid]
        for eid in cl.get("supporting_example_ids", [])
        if eid in examples_by_id
    ]
    slim_supporting = [
        {
            "example_id": e["example_id"],
            "pr_number": e["pr_number"],
            "path": e["path"],
            "comments": [c["body"] for c in e.get("comments", [])],
            "hunk": (e.get("hunk_before") or "")[:500],
        }
        for e in supporting[:5]
    ]
    focus_areas = "\n".join(f"- {a}" for a in repo_config.get("focus_areas", []))
    scope_guidance = repo_config.get("scope_guidance", "")
    prompt = RULE_PROMPT.format(
        repo_description=repo_config.get("description", ""),
        focus_areas=focus_areas or "No specific focus areas configured.",
        scope_guidance=f"Scope guidance: {scope_guidance}" if scope_guidance else "",
        directive_types=json.dumps(directive_types),
        cluster_json=json.dumps(cl, indent=2),
        examples_json=json.dumps(slim_supporting, indent=2),
    )
    raw = _llm_call(prompt, provider, model)
    rule = _parse_json(raw)
    rule["human_review_status"] = "pending"
    rule["human_review_notes"] = None
    rule["rule_version"] = 1
    rule["supersedes_rule_id"] = None
    rule["distillation_run_id"] = run_id
    rule["created_at"] = datetime.now(timezone.utc).isoformat()
    rule["evidence_cluster_ids"] = [cl["cluster_id"]]
    rule["evidence_pr_numbers"] = list({e["pr_number"] for e in supporting})
    rule["support_count"] = cl.get("support_count", len(supporting))
    return rule


def _promote_to_rules(
    clusters: list[dict],
    examples: list[dict],
    provider: str,
    model: str,
    run_id: str,
    repo_slug: str,
    workers: int = 10,
    repo_config: dict | None = None,
) -> list[dict]:
    """Step 3: turn each cluster into a candidate rule."""
    from schemas.gold import DirectiveType

    repo_config = repo_config or _load_repo_config(repo_slug)
    directive_types = [d.value for d in DirectiveType]
    examples_by_id = {e["example_id"]: e for e in examples}

    ckpt_path = GOLD / repo_slug / "promote_checkpoint.jsonl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["cluster_id"]] = rec["rule"]

    todo = [cl for cl in clusters if cl.get("cluster_id") not in done]
    logger.info(
        "Rule promotion: %d clusters total, %d already checkpointed, %d to process (%d workers)",
        len(clusters), len(done), len(todo), workers,
    )

    failed = 0

    if todo:
        lock = threading.Lock()
        completed = 0

        def _process(cl):
            nonlocal completed, failed
            try:
                rule = _promote_one(cl, examples_by_id, directive_types, provider, model, run_id, repo_config)
                with lock:
                    cid = cl["cluster_id"]
                    done[cid] = rule
                    with open(ckpt_path, "a") as f:
                        f.write(json.dumps({"cluster_id": cid, "rule": rule}, default=str) + "\n")
                    completed += 1
                    if completed % 100 == 0:
                        logger.info("  promote progress: %d / %d", completed, len(todo))
            except Exception:
                with lock:
                    completed += 1
                    failed += 1
                logger.warning("Rule promotion failed for cluster %s", cl.get("cluster_id", "?")[:12])

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, cl) for cl in todo]
            concurrent.futures.wait(futures)

    rules = list(done.values())
    logger.info("Promoted %d rules (%d failed)", len(rules), failed)
    return rules


# ── main ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-opus-4-7"


def _upload_gold_to_nfs(repo_slug: str) -> None:
    import shutil
    import subprocess

    host = os.environ.get("NFS_HOST")
    user = os.environ.get("NFS_USER")
    password = os.environ.get("NFS_PASS")
    remote_base = os.environ.get("NFS_PATH", "/nfs/data/pr-pundit")

    if not all([host, user, password]):
        raise EnvironmentError(
            "NFS_HOST, NFS_USER, and NFS_PASS must be set in .env to use --upload"
        )
    if not shutil.which("sshpass"):
        raise EnvironmentError("sshpass not found — install with: apt install sshpass")

    local_src = str(GOLD / repo_slug)
    remote_dst = f"{user}@{host}:{remote_base}/{repo_slug}"

    cmd = [
        "sshpass", "-p", password,
        "scp", "-r",
        "-o", "StrictHostKeyChecking=no",
        local_src,
        remote_dst,
    ]

    logger.info("Uploading gold/%s → %s:%s/%s", repo_slug, host, remote_base, repo_slug)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Upload failed:\n{result.stderr}")
    logger.info("Upload complete → %s:%s/%s", host, remote_base, repo_slug)


def distill(repo_slug: str, model: str | None = None, workers: int = 10, upload: bool = False):
    model = model or DEFAULT_MODEL
    run_id = str(uuid.uuid4())
    logger.info("Distillation run %s (model=%s, workers=%d)", run_id, model, workers)

    src = SILVER / repo_slug / "examples.json"
    if not src.exists():
        raise FileNotFoundError(f"No silver examples at {src}. Run normalize first.")

    examples = json.loads(src.read_text())
    logger.info("Loaded %d silver examples", len(examples))

    provider = "litellm"
    repo_config = _load_repo_config(repo_slug)
    notify_distill_start(repo_slug.replace("_", "/", 1), run_id, provider, model, len(examples))

    repo_display = repo_slug.replace("_", "/", 1)
    logger.info("Repo config: %s — %d focus areas, %d keywords",
                repo_config.get("name", repo_slug),
                len(repo_config.get("focus_areas", [])),
                len(repo_config.get("focus_keywords", [])))

    try:
        # step 0 — compute reviewer authority (needed by step 1 filter)
        try:
            from pipeline.authority import compute_authority
            compute_authority(repo_slug)
            logger.info("Authority scores computed")
        except Exception as exc:
            logger.warning("Authority computation failed (continuing without it): %s", exc)

        # step 1 — filter (repo-specific normative only)
        normative = _filter_normative(examples, provider, model, repo_slug, workers, repo_config)

        # step 2 — cluster
        clusters = _cluster_examples(normative, provider, model, run_id, repo_slug, workers)

        # step 2.5 — dedup clusters (no LLM, fast)
        clusters = _dedup_clusters(clusters)

        # step 3 — promote
        rules = _promote_to_rules(clusters, normative, provider, model, run_id, repo_slug, workers, repo_config)

        # write gold
        dst = GOLD / repo_slug
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "clusters.json").write_text(json.dumps(clusters, indent=2, default=str))
        (dst / "rules.json").write_text(json.dumps(rules, indent=2, default=str))

        logger.info("Gold output → %s  (%d clusters, %d rules)", dst, len(clusters), len(rules))
        logger.info("All rules have human_review_status='pending'. Review at your convenience.")
        notify_distill_done(repo_display, run_id, len(clusters), len(rules))

        if upload:
            _upload_gold_to_nfs(repo_slug)

    except Exception as exc:
        notify_distill_error(repo_display, run_id, str(exc))
        raise


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Distill silver examples → gold rules")
    p.add_argument("--repo", required=True, help="repo slug (owner_name)")
    p.add_argument("--model", default=None, help="model name from litellm_config.yaml")
    p.add_argument("--workers", type=int, default=10, help="concurrent LLM requests (default 10)")
    p.add_argument("--upload", action="store_true", help="upload gold output to NFS cluster after distillation")
    args = p.parse_args()
    distill(args.repo, args.model, args.workers, args.upload)


if __name__ == "__main__":
    main()
