"""
Architectural-layer awareness for the PR generation pipeline.

Classifies seed files by architectural layer, detects when model-layer changes
can be replaced by a compiler-pass pattern, and audits generated diffs for
layer-policy violations.

Four public functions:
  load_layer_policy       — load policy from pipeline_config.yaml
  classify_seed_files     — assign each seed file (or patch path) to a layer
  check_compiler_pass_sufficiency — LLM check: can these objectives skip model layer?
  audit_layer_distribution — post-rewrite: flag warn_if_present layer files in output
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TypedDict

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "pipeline_config.yaml"


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class LayerRule(TypedDict):
    name: str
    path_prefixes: list[str]
    preferred_for: list[str]
    warn_if_present: bool
    compiler_pass_infra: list[str]


class LayerPolicy(TypedDict):
    repo: str
    layers: list[LayerRule]
    enabled: bool


class FileLayerMap(TypedDict):
    by_file: dict[str, str]          # file_path -> layer name (or "unknown")
    by_layer: dict[str, list[str]]   # layer name -> [file_paths]
    has_model_layer: bool
    has_compiler_pass_layer: bool


class SufficiencyVerdict(TypedDict):
    objective: str
    verdict: str                     # "compiler_pass_sufficient" | "model_layer_required"
    reasoning: str
    new_pass_pattern_needed: str
    matchable_subgraph: str


class SufficiencyResult(TypedDict):
    verdicts: list[SufficiencyVerdict]
    demote_to_excluded: list[str]    # objectives the LLM said can be done via compiler pass
    keep_as_objectives: list[str]    # objectives that still need model-layer changes
    compiler_pass_files_fetched: list[str]
    skipped: bool


class LayerAuditResult(TypedDict):
    warnings: list[str]
    model_layer_files_in_output: list[str]
    clean: bool


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SUFFICIENCY_PROMPT = """\
You are an expert in PyTorch FX graph compiler passes for ML inference.

## Context: How vLLM compiler passes work

vLLM uses an FX-graph compilation infrastructure under `vllm/compilation/passes/`.
Each pass subclasses `InductorPass` (or similar) and implements `__call__(graph)`.
Inside the pass, pattern functions describe a small subgraph using placeholder inputs
and calls to real PyTorch ops. The pass uses `torch.fx.subgraph_rewriter.replace_pattern`
(or a custom registry) to pattern-match the NATURAL forward-pass graph — the graph
torch.compile sees when it traces the model's forward() method.

Key invariant: a compiler pass can only match subgraphs that appear in the NATURAL
forward-pass graph. It cannot introduce structure that the model code never generates.

For example, for a fused allreduce+rmsnorm kernel:
- The natural forward-pass graph for a DeepSeek TP layer already contains:
    allreduce_op(hidden_states) → rms_norm(hidden_states)
  because each TransformerLayer calls allreduce then calls the subsequent RMSNorm.
- A compiler pass can pattern-match `allreduce → rms_norm` and replace it with a
  single fused kernel — WITHOUT any model code changes.
- HOWEVER: if the model code was restructured to pass reduce_results=False or to
  set fused_allreduce=True on the RMSNorm constructor, that restructuring IS a model
  change — even if it ultimately enables a compiler pass to fire. Such passes are NOT
  pure compiler-pass solutions.

## What you must decide

For each stated objective below, decide:

  "compiler_pass_sufficient" — The natural forward-pass graph already exposes the
    right subgraph pattern (e.g. allreduce → rms_norm). A compiler pass can match
    it and replace it with the fused kernel WITHOUT any changes to model code
    (model_executor/models/, model_executor/layers/).
    ONLY say this if you can name the exact matchable subgraph.

  "model_layer_required" — The objective requires wiring changes in model code
    that change what the natural forward-pass graph looks like. E.g.:
    - Setting reduce_results=False on a projection layer
    - Adding a fused_allreduce parameter to a norm layer constructor
    - Routing activations through a new module that didn't previously exist
    These cannot be achieved by pattern-matching the existing forward-pass graph alone.

WARNING: Do NOT say "compiler_pass_sufficient" unless you can describe the specific
matchable subgraph (e.g. "allreduce_op → rms_norm" with their op names). If you are
uncertain, say "model_layer_required" — it is safer to keep the model changes than
to mistakenly drop them.

## Existing compiler-pass infrastructure in the target repo

The following files show what passes, patterns, and registration mechanisms already exist.
Use them to understand what patterns are ALREADY matchable and what the pass structure looks like.

{compiler_pass_files}

## Model-layer files in the seed (files we are considering whether to keep or drop)

{model_layer_files}

## Stated objectives

{objectives_section}

---

Respond with ONLY a JSON object:

{{
  "verdicts": [
    {{
      "objective": "exact text of the objective",
      "verdict": "compiler_pass_sufficient" | "model_layer_required",
      "reasoning": "1-2 sentences: why the natural graph does/doesn't expose the right subgraph",
      "new_pass_pattern_needed": "describe the FX pattern function signature if compiler_pass_sufficient, else empty string",
      "matchable_subgraph": "e.g. 'allreduce_op(x) → rms_norm(x, weight)' — exact op names — else empty string"
    }}
  ],
  "summary": "1-2 sentences summarizing which objectives need model changes and which don't"
}}
"""


# ---------------------------------------------------------------------------
# load_layer_policy
# ---------------------------------------------------------------------------

def load_layer_policy(repo_slug: str) -> LayerPolicy:
    """Load layer policy for repo_slug from pipeline_config.yaml.

    Never raises. Returns enabled=False if no policy found.
    repo_slug may be 'owner/repo' (canonical) or 'owner_repo' (underscore form).
    """
    try:
        config = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
        policies = config.get("layer_policies", {})
        # normalise slug: accept both 'owner/repo' and 'owner_repo'
        slug_slash = repo_slug.replace("_", "/", 1) if "/" not in repo_slug else repo_slug
        slug_under = repo_slug.replace("/", "_", 1) if "/" in repo_slug else repo_slug
        raw = policies.get(slug_slash) or policies.get(slug_under)
        if not raw:
            return {"repo": repo_slug, "layers": [], "enabled": False}
        layers: list[LayerRule] = []
        for layer_raw in raw.get("layers", []):
            layers.append({
                "name": layer_raw.get("name", "unknown"),
                "path_prefixes": layer_raw.get("path_prefixes", []),
                "preferred_for": layer_raw.get("preferred_for", []),
                "warn_if_present": bool(layer_raw.get("warn_if_present", False)),
                "compiler_pass_infra": layer_raw.get("compiler_pass_infra", []),
            })
        return {"repo": repo_slug, "layers": layers, "enabled": bool(layers)}
    except Exception as exc:
        logger.warning("load_layer_policy failed for %s: %s", repo_slug, exc)
        return {"repo": repo_slug, "layers": [], "enabled": False}


# ---------------------------------------------------------------------------
# classify_seed_files
# ---------------------------------------------------------------------------

def _match_layer(path: str, policy: LayerPolicy) -> str:
    """Return the layer name for path, longest-prefix match wins."""
    best_name = "unknown"
    best_len = -1
    for layer in policy["layers"]:
        for prefix in layer["path_prefixes"]:
            if path.startswith(prefix) and len(prefix) > best_len:
                best_len = len(prefix)
                best_name = layer["name"]
    return best_name


def _extract_patch_paths(patch_text: str) -> list[str]:
    """Return all b-paths from a unified diff (+++ b/... lines)."""
    paths = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            p = line[6:].strip()
            if p:
                paths.append(p)
    return paths


def classify_seed_files(
    file_names: list[str],
    policy: LayerPolicy,
    patch_contents: dict[str, str] | None = None,
) -> FileLayerMap:
    """Assign each seed file to an architectural layer.

    For .patch files, parse +++ b/ lines and use the majority layer from those
    actual paths rather than the patch filename.

    patch_contents: {filename: content} for patch files in the seed.
    """
    if not policy["enabled"]:
        return {
            "by_file": {},
            "by_layer": {},
            "has_model_layer": False,
            "has_compiler_pass_layer": False,
        }

    by_file: dict[str, str] = {}
    by_layer: dict[str, list[str]] = {}

    for fname in file_names:
        if fname.endswith((".patch", ".diff")) and patch_contents and fname in patch_contents:
            paths = _extract_patch_paths(patch_contents[fname])
            if paths:
                # For patch files: each internal path contributes to layer membership.
                # We store the patch file under the majority layer name for by_file,
                # but also populate by_layer for ALL layers seen in the patch so that
                # has_model_layer / has_compiler_pass_layer reflect the patch contents.
                layer_votes: dict[str, int] = {}
                for p in paths:
                    ln = _match_layer(p, policy)
                    layer_votes[ln] = layer_votes.get(ln, 0) + 1
                majority_layer = max(layer_votes, key=lambda k: layer_votes[k])
                by_file[fname] = majority_layer
                # Populate all represented layers so has_model_layer etc. are accurate
                for ln, count in layer_votes.items():
                    # Use virtual names like "<patch>:model" for the specific paths
                    for p in paths:
                        if _match_layer(p, policy) == ln:
                            by_layer.setdefault(ln, [])
                            if p not in by_layer[ln]:
                                by_layer[ln].append(p)
            else:
                layer = _match_layer(fname, policy)
                by_file[fname] = layer
                by_layer.setdefault(layer, []).append(fname)
        else:
            layer = _match_layer(fname, policy)
            by_file[fname] = layer
            by_layer.setdefault(layer, []).append(fname)

    return {
        "by_file": by_file,
        "by_layer": by_layer,
        "has_model_layer": bool(by_layer.get("model")),
        "has_compiler_pass_layer": bool(by_layer.get("compiler_pass")),
    }


# ---------------------------------------------------------------------------
# check_compiler_pass_sufficiency
# ---------------------------------------------------------------------------

def _fetch_file_from_github(path: str, repo: str, token: str, max_lines: int = 200) -> str | None:
    """Fetch up to max_lines of a file from GitHub via raw.githubusercontent.com."""
    import urllib.request

    # repo may be 'owner/repo' — handle both
    owner_repo = repo if "/" in repo else repo.replace("_", "/", 1)
    url = f"https://raw.githubusercontent.com/{owner_repo}/main/{path}"
    auth_header = f"Bearer {token}"
    req = urllib.request.Request(url, headers={"Authorization": auth_header})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            lines = content.splitlines()[:max_lines]
            return "\n".join(lines)
    except Exception as exc:
        logger.debug("_fetch_file_from_github %s: %s", path, exc)
        return None


def _list_github_dir(path: str, repo: str, token: str) -> list[str]:
    """Return file paths in a GitHub directory (one level, .py files only)."""
    import json
    import urllib.request

    owner_repo = repo if "/" in repo else repo.replace("_", "/", 1)
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path.rstrip('/')}"
    auth_header = f"Bearer {token}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": auth_header, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read())
            return [
                item["path"]
                for item in items
                if item["type"] == "file" and item["name"].endswith(".py")
            ]
    except Exception as exc:
        logger.debug("_list_github_dir %s: %s", path, exc)
        return []


def _fetch_compiler_pass_files(
    policy: LayerPolicy, target_repo: str, token: str, max_files: int = 10
) -> tuple[dict[str, str], list[str]]:
    """Fetch content of compiler_pass_infra files for the policy.

    Returns (content_map, fetched_path_list).
    """
    compiler_layer = next(
        (layer for layer in policy["layers"] if layer["name"] == "compiler_pass"),
        None,
    )
    if not compiler_layer:
        return {}, []

    infra_paths: list[str] = []
    for entry in compiler_layer["compiler_pass_infra"]:
        if entry.endswith("/"):
            # Directory — list .py files
            dir_files = _list_github_dir(entry, target_repo, token)
            infra_paths.extend(dir_files[:5])  # cap per-dir
        else:
            infra_paths.append(entry)

    fetched: dict[str, str] = {}
    for path in infra_paths[:max_files]:
        if path in fetched:
            continue
        content = _fetch_file_from_github(path, target_repo, token)
        if content:
            fetched[path] = content

    return fetched, list(fetched.keys())


def check_compiler_pass_sufficiency(
    objectives: list[str],
    seed_layer_map: FileLayerMap,
    target_repo: str,
    token: str,
    policy: LayerPolicy,
    model: str = "claude-opus-4-7",
) -> SufficiencyResult:
    """Check whether stated objectives can be achieved purely via compiler passes.

    Only runs when:
    - policy.enabled is True
    - seed_layer_map.has_model_layer is True (there are model-layer files to potentially drop)

    On any error, returns skipped=True (conservative — keep all objectives).
    """
    _empty: SufficiencyResult = {
        "verdicts": [],
        "demote_to_excluded": [],
        "keep_as_objectives": objectives[:],
        "compiler_pass_files_fetched": [],
        "skipped": True,
    }

    if not policy["enabled"]:
        logger.debug("check_compiler_pass_sufficiency: policy disabled, skipping")
        return _empty

    if not seed_layer_map["has_model_layer"]:
        logger.debug("check_compiler_pass_sufficiency: no model-layer files in seed, skipping")
        _empty["skipped"] = False
        return _empty

    if not objectives:
        _empty["skipped"] = False
        return _empty

    try:
        from pipeline.llm import llm_call, make_client, parse_json

        # Fetch compiler-pass infrastructure from the target repo
        compiler_files, fetched_paths = _fetch_compiler_pass_files(policy, target_repo, token)
        logger.info(
            "Sufficiency check: fetched %d compiler-pass infra files from %s",
            len(compiler_files), target_repo,
        )

        if not compiler_files:
            logger.warning("check_compiler_pass_sufficiency: no compiler-pass files fetched — skipping")
            return _empty

        # Format compiler pass files for the prompt
        cp_parts = []
        for path, content in compiler_files.items():
            cp_parts.append(f"### {path}\n```python\n{content}\n```")
        compiler_pass_section = "\n\n".join(cp_parts)

        # Format model-layer seed files
        model_layer_files = seed_layer_map["by_layer"].get("model", [])
        model_layer_section = "\n".join(f"  - {f}" for f in model_layer_files)
        if not model_layer_section:
            model_layer_section = "(none)"

        objectives_section = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(objectives))

        prompt = _SUFFICIENCY_PROMPT.format(
            compiler_pass_files=compiler_pass_section,
            model_layer_files=model_layer_section,
            objectives_section=objectives_section,
        )

        client = make_client()
        raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=True, temperature=0)
        parsed = parse_json(raw)

        if not isinstance(parsed, dict):
            logger.warning("check_compiler_pass_sufficiency: LLM returned non-dict — skipping")
            return _empty

        verdicts: list[SufficiencyVerdict] = parsed.get("verdicts", [])
        demote: list[str] = []
        keep: list[str] = []

        # Guard: only demote objectives that are actually about model-layer files.
        # An objective not referencing any model-layer path prefix should never be
        # demoted — it may be a compiler-pass or config objective that the LLM
        # mistakenly marks sufficient.
        model_layer_prefixes = [
            p
            for layer in policy.get("layers", [])
            if layer.get("warn_if_present")
            for p in layer.get("path_prefixes", [])
        ]
        model_layer_files_in_seed = set(seed_layer_map["by_layer"].get("model", []))

        def _objective_is_model_layer(obj_text: str) -> bool:
            """Return True iff the objective text references a model-layer path or file."""
            obj_lower = obj_text.lower()
            for prefix in model_layer_prefixes:
                if prefix.lower() in obj_lower:
                    return True
            for fname in model_layer_files_in_seed:
                if fname.lower().split("/")[-1] in obj_lower:
                    return True
            return False

        # Wiring keywords indicate the objective requires explicit model-code changes
        # (constructor args, explicit calls, line-level wiring) — a compiler pass alone
        # cannot satisfy these. Force-keep any verdict that matches these regardless of
        # what the LLM returned.
        _WIRING_KEYWORDS = (
            "wire", "register", "line ~", "line~", "pass ", " to ", " into ",
            "set ", "=true", "=false", "=none", "in constructor",
            "add parameter", "add param", "add argument", "add arg",
        )

        def _has_wiring_keyword(text: str) -> bool:
            lower = text.lower()
            return any(kw in lower for kw in _WIRING_KEYWORDS)

        for v in verdicts:
            obj = v.get("objective", "")
            verdict = v.get("verdict", "model_layer_required")
            matchable_subgraph = (v.get("matchable_subgraph") or "").strip()

            if verdict == "compiler_pass_sufficient":
                matched = _find_matching_objective(obj, objectives)
                candidate = matched or obj

                # Force-keep: wiring keyword in the objective text overrides the LLM verdict
                if _has_wiring_keyword(candidate):
                    logger.info(
                        "  [pass-sufficient→keep] wiring keyword detected — overriding to model_layer_required: %s",
                        candidate[:100],
                    )
                    keep.append(candidate)
                    continue

                # Force-keep: LLM must provide a non-empty matchable_subgraph to demote
                if not matchable_subgraph:
                    logger.info(
                        "  [pass-sufficient→keep] empty matchable_subgraph — overriding to model_layer_required: %s",
                        candidate[:100],
                    )
                    keep.append(candidate)
                    continue

                # Only demote if we can match the objective to the original list
                if not _objective_is_model_layer(candidate):
                    logger.info(
                        "  [pass-sufficient] skipping demotion — objective not about model layer: %s",
                        candidate[:100],
                    )
                    keep.append(candidate)
                elif matched:
                    demote.append(matched)
                    logger.info("  [pass-sufficient] %s (subgraph: %s)", matched[:100], matchable_subgraph[:80])
                else:
                    demote.append(obj)
                    logger.info("  [pass-sufficient] (unmatched) %s", obj[:100])
            else:
                matched = _find_matching_objective(obj, objectives)
                keep.append(matched or obj)

        # Any objectives not mentioned in verdicts → keep
        mentioned = {v.get("objective", "") for v in verdicts}
        for obj in objectives:
            if not _find_matching_objective(obj, list(mentioned)):
                keep.append(obj)

        return {
            "verdicts": verdicts,
            "demote_to_excluded": demote,
            "keep_as_objectives": keep,
            "compiler_pass_files_fetched": fetched_paths,
            "skipped": False,
        }

    except Exception as exc:
        logger.warning("check_compiler_pass_sufficiency failed: %s — skipping (conservative)", exc)
        return _empty


def _find_matching_objective(candidate: str, objectives: list[str]) -> str | None:
    """Find best matching objective string from the list (exact or substring)."""
    # Exact match
    if candidate in objectives:
        return candidate
    # Substring match in either direction
    cand_lower = candidate.lower()
    for obj in objectives:
        obj_lower = obj.lower()
        if cand_lower in obj_lower or obj_lower in cand_lower:
            return obj
        # Check first 60 chars overlap
        if len(cand_lower) > 30 and len(obj_lower) > 30:
            if cand_lower[:60] == obj_lower[:60]:
                return obj
    return None


# ---------------------------------------------------------------------------
# audit_layer_distribution
# ---------------------------------------------------------------------------

_JUSTIFIED_INTENTS = {"enables_compiler_pass", "wiring", "refactor"}
_SUSPICIOUS_INTENTS = {"new_model_logic"}

_ARCH_AUDIT_PROMPT = """\
You are auditing a set of pull-request diffs against this repository's architectural conventions.

## Repository contributing conventions (fetched from the repo itself)

{repo_docs}

## Known layer-policy violation patterns (learned from past PRs to this repo)

{violation_patterns}

## PR diffs under review

{diff_summary}

## Planner's stated intent for each flagged file

{intent_summary}

## Your task

For each flagged file listed below, decide whether its presence in the diff is a genuine
architectural concern or a justified exception.

Flagged files: {flagged_list}

Use the repository's own contributing docs and conventions above as your primary guide.
Only flag a file as a real warning if it clearly violates how this repo organises its
architecture. Err on the side of permitting changes when the repo docs don't address
the specific pattern.

Respond with ONLY a JSON object:
{{
  "warnings": [
    {{
      "file": "path/to/file.py",
      "pr_index": 1,
      "message": "one-sentence warning grounded in the repo's own conventions"
    }}
  ],
  "clean_files": ["path/to/file.py that looks fine, ..."],
  "reasoning": "2-3 sentences summarising your overall judgment"
}}
"""

_REPO_DOC_PATHS = [
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    "docs/contributing.md",
    "docs/contributing.rst",
    "docs/design.md",
    "docs/architecture.md",
    "docs/developer_guide.md",
    "docs/source/contributing/overview.rst",
    "docs/source/contributing/index.rst",
    "ADR.md",
    "docs/adr/index.md",
]


def _fetch_repo_docs(target_repo: str, token: str, max_chars: int = 16000) -> str:
    """Fetch CONTRIBUTING.md and key architectural docs from the target repo.

    Returns concatenated text capped at max_chars, or empty string on failure.
    """
    collected: list[str] = []
    total = 0
    for path in _REPO_DOC_PATHS:
        if total >= max_chars:
            break
        content = _fetch_file_from_github(path, target_repo, token, max_lines=300)
        if content:
            snippet = content[: max_chars - total]
            collected.append(f"### {path}\n{snippet}")
            total += len(snippet)
            if total >= max_chars:
                break
    return "\n\n".join(collected)


def audit_layer_distribution(
    pr_diffs: list[str],
    plan: dict,
    policy: LayerPolicy,
    *,
    target_repo: str = "",
    token: str = "",
    model: str = "claude-opus-4-7",
) -> LayerAuditResult:
    """Audit generated PR diffs for files in warn_if_present layers.

    Two-stage audit:
    1. Rule-based pre-filter: identify files in warn_if_present layers.
    2. LLM judgment (when target_repo + token provided): fetch the repo's own
       CONTRIBUTING.md and architectural docs, then ask an LLM to judge whether
       each flagged file is a genuine violation given the repo's own conventions.
       Falls back to pure rule-based if the LLM call fails or no token is given.

    pr_diffs: list of unified diff strings (one per PR in the series).
    plan: the pr_plan dict including per-PR model_layer_touches annotations.
    target_repo: 'owner/repo' string used to fetch repo docs from GitHub.
    token: GitHub token for fetching docs.
    model: LLM model name for the judgment call.
    """
    if not policy["enabled"]:
        return {"warnings": [], "model_layer_files_in_output": [], "clean": True}

    warn_layers = {layer["name"] for layer in policy["layers"] if layer["warn_if_present"]}
    if not warn_layers:
        return {"warnings": [], "model_layer_files_in_output": [], "clean": True}

    # Build a lookup: path → touch annotation, keyed by (pr_index, path)
    pr_series = plan.get("pr_series", [])
    touch_by_pr_path: dict[tuple[int, str], dict] = {}
    for pr in pr_series:
        for touch in pr.get("model_layer_touches", []):
            key = (pr.get("index", 0), touch.get("file", ""))
            touch_by_pr_path[key] = touch

    # Fallback: synthesize touch entries for files listed in affected_files / new_files
    # that have no explicit model_layer_touches annotation. These files are authorized by
    # the planner — treating them as intent=unknown would incorrectly fail the arch check.
    for pr in pr_series:
        pr_idx = pr.get("index", 0)
        authorized: set[str] = set(pr.get("affected_files", []))
        for nf in pr.get("new_files", []):
            if isinstance(nf, dict):
                authorized.add(nf.get("path", ""))
            elif isinstance(nf, str):
                authorized.add(nf)
        for path in authorized:
            if path and (pr_idx, path) not in touch_by_pr_path:
                touch_by_pr_path[(pr_idx, path)] = {
                    "file": path,
                    "intent": "planned",
                    "rationale": "authorized in PR plan (affected_files / new_files)",
                }

    # Stage 1: collect all files in warn_if_present layers and their per-PR stats
    class _Candidate:
        def __init__(self, pr_idx: int, path: str, added: int, removed: int):
            self.pr_idx = pr_idx
            self.path = path
            self.added = added
            self.removed = removed

    candidates: list[_Candidate] = []
    flagged_files: list[str] = []

    for i, diff_text in enumerate(pr_diffs, 1):
        current_file: str | None = None
        per_file_added: dict[str, int] = {}
        per_file_removed: dict[str, int] = {}

        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
                per_file_added.setdefault(current_file, 0)
                per_file_removed.setdefault(current_file, 0)
            elif current_file and line.startswith("+") and not line.startswith("+++"):
                per_file_added[current_file] = per_file_added.get(current_file, 0) + 1
            elif current_file and line.startswith("-") and not line.startswith("---"):
                per_file_removed[current_file] = per_file_removed.get(current_file, 0) + 1

        for path, added in per_file_added.items():
            if not path:
                continue
            layer = _match_layer(path, policy)
            if layer not in warn_layers:
                continue
            flagged_files.append(path)
            candidates.append(_Candidate(i, path, added, per_file_removed.get(path, 0)))

    # Deduplicate files while preserving order
    seen: set[str] = set()
    unique_files: list[str] = []
    for f in flagged_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    if not candidates:
        return {"warnings": [], "model_layer_files_in_output": [], "clean": True}

    # Stage 2: LLM-grounded judgment when repo docs are available
    if target_repo and token:
        try:
            warnings = _llm_arch_audit(
                candidates=candidates,
                touch_by_pr_path=touch_by_pr_path,
                plan=plan,
                target_repo=target_repo,
                token=token,
                model=model,
            )
            return {
                "warnings": warnings,
                "model_layer_files_in_output": unique_files,
                "clean": len(warnings) == 0,
            }
        except Exception as exc:
            logger.warning("LLM arch audit failed (%s) — falling back to rule-based", exc)

    # Fallback: rule-based audit (original logic)
    warnings = _rule_based_audit(candidates, touch_by_pr_path)
    return {
        "warnings": warnings,
        "model_layer_files_in_output": unique_files,
        "clean": len(warnings) == 0,
    }


def _rule_based_audit(
    candidates: list,
    touch_by_pr_path: dict[tuple[int, str], dict],
) -> list[str]:
    """Original rule-based audit logic — used as fallback when LLM audit fails."""
    warnings: list[str] = []
    for c in candidates:
        touch = touch_by_pr_path.get((c.pr_idx, c.path))
        if touch is None:
            warnings.append(
                f"PR {c.pr_idx}: {c.path} — no model_layer_touches annotation in plan. "
                f"Verify this cannot be achieved via a compiler pass instead."
            )
            continue
        intent = touch.get("intent", "new_model_logic")
        rationale = touch.get("rationale", "")
        companion = touch.get("companion_pr_index")
        if intent == "new_model_logic":
            warnings.append(
                f"PR {c.pr_idx}: {c.path} intent=new_model_logic — {rationale or 'no rationale provided'}. "
                f"Verify this cannot be achieved via a compiler pass instead."
            )
        elif intent == "enables_compiler_pass":
            if not companion:
                warnings.append(
                    f"PR {c.pr_idx}: {c.path} intent=enables_compiler_pass but no companion_pr_index — "
                    f"which PR contains the pass? {rationale}"
                )
            elif c.added > c.removed * 2 + 20:
                warnings.append(
                    f"PR {c.pr_idx}: {c.path} intent=enables_compiler_pass but diff is heavily additive "
                    f"(+{c.added}/-{c.removed}) — verify no new model logic was added. {rationale}"
                )
        elif intent == "refactor":
            if c.added > c.removed + 30:
                warnings.append(
                    f"PR {c.pr_idx}: {c.path} intent=refactor but diff adds significantly more than it removes "
                    f"(+{c.added}/-{c.removed}) — verify this is not new_model_logic. {rationale}"
                )
        elif intent == "wiring":
            if c.added > 30:
                warnings.append(
                    f"PR {c.pr_idx}: {c.path} intent=wiring but {c.added} lines added — "
                    f"consider classifying as new_model_logic if substantial. {rationale}"
                )
    return warnings


def _llm_arch_audit(
    candidates: list,
    touch_by_pr_path: dict[tuple[int, str], dict],
    plan: dict,
    target_repo: str,
    token: str,
    model: str,
) -> list[str]:
    """LLM-grounded arch audit using the repo's own contributing docs."""
    from pipeline.llm import llm_call, make_client, parse_json

    repo_docs = _fetch_repo_docs(target_repo, token)
    if not repo_docs:
        repo_docs = "(Could not fetch repo docs — no CONTRIBUTING.md or docs/ found.)"
    else:
        logger.info("arch audit: fetched %d chars of repo docs from %s", len(repo_docs), target_repo)

    # Collect known violation patterns from the plan's arch_principles field
    arch_principles = plan.get("arch_principles") or {}
    violation_patterns = ""
    if arch_principles:
        vp_parts = []
        for cat, items in arch_principles.items():
            if isinstance(items, list) and items:
                vp_parts.append(f"{cat}:\n" + "\n".join(f"  - {it}" for it in items[:5]))
        violation_patterns = "\n\n".join(vp_parts) if vp_parts else "(none)"
    else:
        violation_patterns = "(none recorded)"

    # Build diff summary and intent summary for flagged files
    diff_parts: list[str] = []
    intent_parts: list[str] = []
    flagged_list_items: list[str] = []

    for c in candidates:
        touch = touch_by_pr_path.get((c.pr_idx, c.path))
        intent = (touch or {}).get("intent", "unknown")
        rationale = (touch or {}).get("rationale", "")
        diff_parts.append(
            f"PR {c.pr_idx}: {c.path} (+{c.added}/-{c.removed} lines, intent={intent})"
        )
        intent_parts.append(
            f"PR {c.pr_idx} / {c.path}: intent={intent}, rationale={rationale or '(none)'}"
        )
        flagged_list_items.append(f"{c.path} (PR {c.pr_idx})")

    prompt = _ARCH_AUDIT_PROMPT.format(
        repo_docs=repo_docs,
        violation_patterns=violation_patterns,
        diff_summary="\n".join(diff_parts),
        intent_summary="\n".join(intent_parts),
        flagged_list=", ".join(flagged_list_items),
    )

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=True, temperature=0)
    parsed = parse_json(raw)

    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-dict: {type(parsed)}")

    warnings: list[str] = []
    for w in parsed.get("warnings", []):
        file_path = w.get("file", "")
        pr_idx = w.get("pr_index", "?")
        message = w.get("message", "")
        if file_path and message:
            warnings.append(f"PR {pr_idx}: {file_path} — {message}")

    reasoning = parsed.get("reasoning", "")
    if reasoning:
        logger.info("arch audit reasoning: %s", reasoning)

    return warnings
