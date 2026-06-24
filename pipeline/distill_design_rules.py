"""
Stage L3 — Incremental architecture harness distillation.

Takes harness candidates from the supervisor (per lineage tree) and distills
them into a compact set of AuditHarnesses using sequential LLM review.

The key invariant: we only admit NEW dimensions.  Candidates are presented
to an LLM one at a time (deepest lineage trees first) against the growing bank:

  "Does this candidate add a genuinely new architectural dimension
   not already covered by the existing bank?"

  → keep   : added to the bank as a new harness
  → discard: already covered; the existing harness absorbs the evidence

This avoids embedding heuristics entirely. The LLM understands architectural
orthogonality directly. The result is a small, non-redundant bank.

Usage:
    distill-design-rules --repo vllm-project/vllm [--dry-run] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.llm import llm_call, make_client, parse_json
from pipeline.pr_lineage import build_lineage_trees
from pipeline.pr_supervisor import load_candidates, run_supervisor
from schemas.audit_harness import AuditHarness, HarnessExample, HarnessStatus, LineageRef

logger = logging.getLogger(__name__)

LINEAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "lineage"
RANKED_PATH = LINEAGE_DIR / "ranked_new_harnesses.json"

_DEFAULT_MODEL = "claude-opus-4-7"


def _harnesses_path(repo_slug: str) -> Path:
    return LINEAGE_DIR / repo_slug / "audit_harnesses.json"


# ── harness store ─────────────────────────────────────────────────────

def load_harnesses(repo_slug: str) -> list[AuditHarness]:
    path = _harnesses_path(repo_slug)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [AuditHarness.from_dict(h) for h in data]


def load_all_harnesses() -> list[AuditHarness]:
    """Merge harnesses from all repo dirs — used at runtime across the full bank."""
    all_h: list[AuditHarness] = []
    seen_ids: set[str] = set()
    if not LINEAGE_DIR.exists():
        return []
    for d in sorted(LINEAGE_DIR.iterdir()):
        path = d / "audit_harnesses.json"
        if path.exists():
            data = json.loads(path.read_text())
            for h in data:
                harness = AuditHarness.from_dict(h)
                if harness.harness_id not in seen_ids:
                    seen_ids.add(harness.harness_id)
                    all_h.append(harness)
    return all_h


def save_harnesses(repo_slug: str, harnesses: list[AuditHarness]) -> None:
    path = _harnesses_path(repo_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([h.to_dict() for h in harnesses], indent=2, default=str)
    )


# ── sequential LLM distillation ───────────────────────────────────────

def _bank_summary(bank: list[AuditHarness]) -> str:
    """Compact listing of the current bank for the LLM prompt."""
    if not bank:
        return "  (empty — this is the first candidate)"
    lines = []
    for h in bank:
        lines.append(f"  [{h.harness_id[:8]}] {h.name}")
        lines.append(f"    {h.description[:120]}")
    return "\n".join(lines)


def _llm_review_candidate(
    candidate: dict,
    bank: list[AuditHarness],
    model: str,
    client,
) -> tuple[str, str | None]:
    """Ask the LLM whether the candidate adds a new dimension to the bank.

    Returns:
        ("keep", None)            — new principle, add to bank
        ("discard", harness_id)   — covered by existing harness_id
    """
    prompt = f"""You are building a compact bank of architecture audit harnesses — executable
checks that an LLM can run on a PR diff to detect specific architectural anti-patterns.

CURRENT BANK ({len(bank)} harness{"es" if len(bank) != 1 else ""}):
{_bank_summary(bank)}

NEW CANDIDATE:
Name: {candidate.get("name")}
Description: {candidate.get("description")}
Relevance criteria: {candidate.get("relevance_criteria")}
Anti-pattern detected: {candidate.get("anti_pattern")}
Correct pattern: {candidate.get("correct_pattern")}

TASK:
Does this candidate add a genuinely NEW architectural dimension to the bank —
a principle that existing harnesses do NOT already check?

Rules:
- "New" means the candidate would catch a class of architectural mistake that
  none of the existing harnesses would catch.
- "Covered" means an existing harness already addresses the same failure mode,
  even if phrased differently.
- Be conservative: prefer a small, orthogonal bank over an exhaustive one.
  If in doubt, discard.

Output JSON (one of these two forms, nothing else):
{{"decision": "keep"}}
or
{{"decision": "discard", "covered_by": "<harness_id_8chars>"}}"""

    try:
        raw = llm_call(prompt, model, client=client, max_tokens=512, json_mode=False)
        result = parse_json(raw.strip())
        if not isinstance(result, dict):
            return "keep", None
        decision = result.get("decision", "keep")
        if decision == "discard":
            return "discard", result.get("covered_by")
        return "keep", None
    except Exception as exc:
        logger.warning("LLM review error for '%s': %s — defaulting to keep", candidate.get("name"), exc)
        return "keep", None


def _sequential_llm_dedup(
    candidates: list[dict],
    existing_bank: list[AuditHarness],
    model: str,
    client,
    dry_run: bool = False,
) -> list[AuditHarness]:
    """Review candidates one-by-one; build a compact, orthogonal harness bank.

    Candidates are processed deepest-tree-first (depth desc), then by tree_id
    for determinism. Deeper chains signal harder architectural problems and tend
    to produce more fundamental principles, so they seed the bank first.
    Returns the list of newly admitted harnesses.
    """
    run_id = str(uuid.uuid4())
    bank = list(existing_bank)  # starts with whatever exists
    new_harnesses: list[AuditHarness] = []

    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-c.get("_depth", 1), c.get("_tree_id", "")),
    )

    for i, candidate in enumerate(sorted_candidates):
        logger.info(
            "Reviewing candidate %d/%d: '%s' (depth=%d, tree=%s)",
            i + 1, len(sorted_candidates),
            candidate.get("name"),
            candidate.get("_depth", 1),
            candidate.get("_tree_id", ""),
        )

        decision, covered_by = _llm_review_candidate(candidate, bank, model, client)

        if decision == "discard":
            logger.info("  → DISCARD (covered by %s)", covered_by or "existing")
        else:
            logger.info("  → KEEP as new harness")
            if not dry_run:
                harness = _candidate_to_harness(candidate, run_id=run_id)
                bank.append(harness)
                new_harnesses.append(harness)
            else:
                logger.info("  [dry-run] Would add: '%s'", candidate.get("name"))

    return new_harnesses


def _candidate_to_harness(c: dict, run_id: str) -> AuditHarness:
    now = datetime.now(timezone.utc).isoformat()

    lineage_ref = LineageRef(
        tree_id=c.get("_tree_id", ""),
        repo=c.get("_repo", ""),
        depth=c.get("_depth", 1),
        failed_prs=c.get("_failed_prs") or [],
        merged_pr=c.get("_merged_pr", 0),
    )

    example = None
    if c.get("anti_pattern") and c.get("correct_pattern"):
        example = HarnessExample(
            description=c.get("name", ""),
            anti_pattern=c.get("anti_pattern", ""),
            correct_pattern=c.get("correct_pattern", ""),
            source_tree_id=lineage_ref.tree_id,
            source_failed_pr=(lineage_ref.failed_prs or [0])[0],
            source_merged_pr=lineage_ref.merged_pr,
        )

    return AuditHarness(
        harness_id=str(uuid.uuid4()),
        name=c.get("name", "unknown"),
        description=c.get("description", ""),
        relevance_criteria=c.get("relevance_criteria", ""),
        audit_prompt_template=c.get("audit_prompt_template", ""),
        lineage_refs=[lineage_ref],
        examples=[example] if example else [],
        status=HarnessStatus.ACTIVE,
        distillation_run_id=run_id,
        created_at=now,
        updated_at=now,
    )


# ── main distillation ─────────────────────────────────────────────────

def distill(
    repo_slugs: list[str],
    model: str = _DEFAULT_MODEL,
    dry_run: bool = False,
    workers: int = 4,
) -> tuple[list[AuditHarness], list[AuditHarness]]:
    """Run sequential LLM distillation pipeline per-repo.

    For each repo, loads that repo's existing harnesses plus all harnesses from
    other repos as the starting bank (so cross-repo duplicates are discarded).
    New harnesses are saved per-repo.

    Returns:
        (all_harnesses_across_repos, newly_admitted_harnesses)
    """
    client = make_client()

    # Start bank from all existing harnesses across all repos
    existing = load_all_harnesses()
    logger.info("Loaded %d existing harnesses across all repos", len(existing))

    all_new_harnesses: list[AuditHarness] = []

    for slug in repo_slugs:
        candidates = load_candidates(slug)
        logger.info("Repo %s: %d candidates loaded", slug, len(candidates))
        if not candidates:
            continue

        # Bank for this repo = all existing (cross-repo) + what we've admitted so far
        current_bank = existing + all_new_harnesses
        logger.info("Starting sequential LLM review for %s (%d candidates, bank size=%d)",
                    slug, len(candidates), len(current_bank))

        new_for_repo = _sequential_llm_dedup(
            candidates, current_bank, model=model, client=client, dry_run=dry_run
        )

        if not dry_run and new_for_repo:
            # Merge with this repo's existing harnesses and save
            repo_existing = load_harnesses(slug)
            repo_all = repo_existing + new_for_repo
            repo_all.sort(key=lambda h: (-max((r.depth for r in h.lineage_refs), default=1), h.name))
            save_harnesses(slug, repo_all)
            logger.info("Saved %d harnesses (%d new) for %s → %s",
                        len(repo_all), len(new_for_repo), slug, _harnesses_path(slug))
            all_new_harnesses.extend(new_for_repo)

    if not dry_run:
        all_harnesses = load_all_harnesses()
        LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
        RANKED_PATH.write_text(
            json.dumps([h.to_dict() for h in all_new_harnesses], indent=2, default=str)
        )
        logger.info("Wrote %d new harnesses → %s", len(all_new_harnesses), RANKED_PATH)
        return all_harnesses, all_new_harnesses
    else:
        return existing, []


# ── upstream architecture principle ingestion ─────────────────────────────

_ARCH_DOC_PATHS = [
    "CONTRIBUTING.md",
    "docs/source/contributing/architecture.md",
    "docs/source/design/arch_overview.rst",
    "docs/source/design/arch_overview.md",
    "ARCHITECTURE.md",
    "docs/ARCHITECTURE.md",
]

_EXTRACT_ARCH_PRINCIPLES_PROMPT = """\
You are reading upstream documentation for the GitHub repository "{repo}".

Your task: extract concrete architectural principles that would help a code reviewer
detect structural violations in contributed PRs. Focus on:
- Layer separation rules (what modules/APIs must NOT be crossed directly)
- Seam or boundary rules (how subsystems must communicate)
- Naming or location conventions with hard enforcement
- Extension points that must be used instead of direct modifications
- Patterns explicitly called out as forbidden or strongly discouraged

Do NOT extract:
- Style preferences (naming conventions that are soft suggestions)
- Process instructions (how to run tests, how to submit PRs)
- Performance recommendations without structural enforcement

For each principle you find, output a JSON object with:
  {{
    "name": "short-kebab-case-slug",
    "description": "One to two sentence description of the structural law",
    "relevance_criteria": "When this harness should be applied — what files, subsystems, or change types trigger it",
    "audit_prompt_template": "Specific question to ask when auditing a PR diff for violations of this principle"
  }}

Return a JSON array of these objects. If no concrete architectural principles are
present in the documentation, return an empty array [].

DOCUMENTATION SOURCE: {source_path}

---

{content}

---

Output ONLY the JSON array."""


def _load_repo_config_for_distill(repo_slug: str) -> dict:
    """Load repo_config.yaml for a given repo slug (owner_name format)."""
    import yaml
    config_path = LINEAGE_DIR.parent / "gold" / repo_slug / "repo_config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def ingest_upstream_arch_principles(
    repo: str,
    *,
    token: str | None = None,
    model: str = _DEFAULT_MODEL,
    dry_run: bool = False,
) -> list[AuditHarness]:
    """Fetch upstream documentation and extract architectural principles into the harness bank.

    Reads arch_doc_paths from repo_config.yaml (pr_preparation.arch_doc_paths), falls back
    to a default list. Asks an LLM to extract structural principles in AuditHarness format,
    deduplicates against the existing bank, and saves new harnesses to the repo's bank file.

    Returns the list of newly added harnesses.
    """
    import base64
    import httpx

    repo_slug = repo.replace("/", "_", 1)
    repo_config = _load_repo_config_for_distill(repo_slug)
    doc_paths = (
        repo_config.get("pr_preparation", {}).get("arch_doc_paths")
        or _ARCH_DOC_PATHS
    )

    client = make_client()
    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _headers = {"Authorization": f"token {_token}", "Accept": "application/vnd.github.v3+json"}

    def _fetch_doc(path: str) -> str | None:
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            r = httpx.get(url, headers=_headers, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception as exc:
            logger.debug("Failed to fetch %s from %s: %s", path, repo, exc)
        return None

    # Collect all found docs
    docs: list[tuple[str, str]] = []
    for path in doc_paths:
        content = _fetch_doc(path)
        if content:
            logger.info("Found upstream doc: %s (%d chars)", path, len(content))
            docs.append((path, content))

    if not docs:
        logger.info("No upstream architecture docs found for %s", repo)
        return []

    existing_bank = load_all_harnesses()
    existing_names = {h.name.lower() for h in existing_bank}
    run_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    newly_added: list[AuditHarness] = []

    for source_path, content in docs:
        prompt = _EXTRACT_ARCH_PRINCIPLES_PROMPT.format(
            repo=repo,
            source_path=source_path,
            content=content[:40000],
        )
        try:
            raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=False)
            candidates = parse_json(raw.strip())
            if not isinstance(candidates, list):
                continue
        except Exception as exc:
            logger.warning("LLM extraction failed for %s/%s: %s", repo, source_path, exc)
            continue

        for c in candidates:
            if not isinstance(c, dict):
                continue
            name = c.get("name", "").strip()
            if not name:
                continue
            if name.lower() in existing_names:
                logger.info("Skipping '%s' — already in bank", name)
                continue

            harness = AuditHarness(
                harness_id=str(uuid.uuid4()),
                name=name,
                description=c.get("description", ""),
                relevance_criteria=c.get("relevance_criteria", ""),
                audit_prompt_template=c.get("audit_prompt_template", ""),
                lineage_refs=[],
                examples=[],
                status=HarnessStatus.ACTIVE,
                distillation_run_id=run_id,
                created_at=now,
                updated_at=now,
            )
            existing_names.add(name.lower())
            newly_added.append(harness)
            logger.info("Extracted new harness from docs: '%s'", name)

    if newly_added and not dry_run:
        repo_existing = load_harnesses(repo_slug)
        repo_all = repo_existing + newly_added
        save_harnesses(repo_slug, repo_all)
        logger.info(
            "Saved %d new arch-principle harnesses from upstream docs → %s",
            len(newly_added), _harnesses_path(repo_slug),
        )

    return newly_added


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Distill audit harnesses from lineage tree candidates"
    )
    p.add_argument("--repo", action="append", dest="repos",
                   help="owner/name (repeat for multiple repos; default: all with candidates)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would happen without writing files")
    p.add_argument("--supervisor-only", action="store_true",
                   help="Run lineage + supervisor but stop before distillation")
    p.add_argument("--distill-only", action="store_true",
                   help="Skip lineage + supervisor; distill from existing candidates files only")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--passes", default="1,2",
                   help="Lineage passes to run (1=body_xref, 2=revert)")
    p.add_argument("--limit-trees", type=int, default=None,
                   help="Cap trees per repo (for testing)")
    p.add_argument("--ingest-arch-docs", action="store_true",
                   help="Fetch upstream architecture docs and add principles to the harness bank")
    args = p.parse_args()

    if args.ingest_arch_docs:
        if not args.repos:
            p.error("--ingest-arch-docs requires at least one --repo owner/name")
        for repo in args.repos:
            added = ingest_upstream_arch_principles(repo, model=args.model, dry_run=args.dry_run)
            print(f"{'(dry-run) ' if args.dry_run else ''}Ingested {len(added)} arch principles from {repo}:")
            for h in added:
                print(f"  [NEW] {h.name}: {h.description[:100]}")
        return

    passes = [int(x.strip()) for x in args.passes.split(",")]
    existing = load_all_harnesses()

    # Resolve repos
    if args.repos:
        repos = args.repos
    else:
        # Auto-discover repos that have trees.jsonl or candidates
        repos = []
        if LINEAGE_DIR.exists():
            for d in LINEAGE_DIR.iterdir():
                if d.is_dir() and (d / "trees.jsonl").exists():
                    repos.append(d.name.replace("_", "/", 1))
        if not repos:
            p.error("No repos specified and no lineage data found. Use --repo owner/name")

    repo_slugs: list[str] = []
    for repo in repos:
        owner, name = repo.split("/", 1)
        repo_slug = f"{owner}_{name}"
        repo_slugs.append(repo_slug)

    if not args.distill_only:
        for repo in repos:
            owner, name = repo.split("/", 1)
            repo_slug = f"{owner}_{name}"

            # Run lineage tree builder
            logger.info("Building lineage trees for %s …", repo)
            trees = build_lineage_trees(repo, passes=passes, limit=args.limit_trees)

            if not trees:
                logger.info("No trees found for %s", repo)
                continue

            # Write trees to disk
            out_dir = LINEAGE_DIR / repo_slug
            out_dir.mkdir(parents=True, exist_ok=True)
            trees_path = out_dir / "trees.jsonl"
            with open(trees_path, "w") as f:
                for tree in trees:
                    f.write(json.dumps(tree.to_dict(), default=str) + "\n")
            logger.info("Wrote %d trees → %s", len(trees), trees_path)

            # Run supervisor
            logger.info("Running supervisor for %s …", repo)
            run_supervisor(trees, existing, repo_slug, model=args.model, workers=args.workers)

    if args.supervisor_only:
        logger.info("--supervisor-only: stopping before distillation")
        return

    # Distill across all repos
    logger.info("Distilling across repos: %s", repo_slugs)
    all_harnesses, new_harnesses = distill(repo_slugs, model=args.model,
                                           dry_run=args.dry_run, workers=args.workers)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Distillation summary:")
    print(f"  Total harnesses: {len(all_harnesses)}")
    print(f"  New (admitted): {len(new_harnesses)}")
    if new_harnesses:
        print("\n  New harnesses:")
        for h in new_harnesses:
            trees = ", ".join(f"{r.repo}#{r.merged_pr}(depth={r.depth})" for r in h.lineage_refs)
            print(f"    [NEW] {h.name}")
            print(f"      {h.description[:120]}")
            print(f"      From: {trees}")


if __name__ == "__main__":
    main()
