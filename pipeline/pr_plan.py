"""
PR planning: split a large patch into a series of focused, independently-mergeable PRs.

Given a combined diff and judge findings, produces a PR series plan where each entry
describes exactly what objective it serves and why the boundary is drawn there.

This is a pure planning step — it produces prose descriptions of atomic objectives,
not hunk assignments. Diff reconstruction is handled separately by pr_rewrite.py,
which rewrites each changed file from the base state for each PR's objective.

Called by create_pr_from_seed before any fork/push/PR-creation happens.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"

_DIFF_CAP = 200_000  # chars shown to planner for context

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_INTENT_PROMPT = """\
You are reviewing a seed folder that someone prepared to submit upstream as one or more pull requests.

Your job is NOT to look at what lines changed.
Your job is to understand WHAT this seed is trying to accomplish.

You have access to:
- The seed README (describes the motivation and goals)
- The list of files in the seed (tells you what was changed)
- Short excerpts from the non-CSV files (enough to see the shape of the change, not a full diff)

SEED README:
{readme_section}

FILES IN SEED:
{file_list}

FILE EXCERPTS (first ~80 lines of each non-CSV file):
{excerpts_section}

---

Answer these questions by reasoning from the README and file names — NOT from diff noise:

1. What problem is this seed solving? (hardware bug, missing kernel dispatch, missing tuning data, etc.)
2. What is each objective in plain English — one sentence per objective, specific enough that a reviewer
   can later check whether a proposed code change actually serves it.
3. Are there any changes visible in the file listing that appear INCIDENTAL — refactors, unrelated
   cleanups, removals of code that is not mentioned in the README as an objective? List them explicitly
   so the planner knows to DROP them.
4. What is the most likely target repository for these changes?

CRITICAL RULES:
- Objectives must come from the README and the nature of the problem, not from "lines that were deleted."
- If the seed contains a whole-file replacement (e.g. fused_moe.py), there will be many deletions
  relative to upstream that are NOT objectives — they are just cleanup the author did locally.
  Do NOT promote those deletions into objectives.
- Be conservative: prefer fewer, narrower objectives over broad ones. The planner will be more
  conservative if the objectives are precise.

Respond with ONLY a JSON object:

{{
  "objectives": [
    "One sentence describing objective 1",
    "One sentence describing objective 2"
  ],
  "excluded_changes": [
    "Description of incidental change 1 — should be dropped",
    "Description of incidental change 2 — should be dropped"
  ],
  "target_repo_hint": "owner/name or null",
  "summary": "2-3 sentence human-readable summary of what this seed is trying to do."
}}
"""

_PLAN_PROMPT = """\
You are a senior open-source maintainer reviewing a patch before it is split
into pull requests for the GitHub repository "{repo}".

REPOSITORY: {repo}

STATED OBJECTIVES (extracted from seed README and verified against upstream — treat these as ground truth):
{objectives_section}

OBJECTIVE UPSTREAM ROUTING:
Some objectives above are prefixed with [owner/repo] — this prefix is a routing instruction,
not a label. It means that objective MUST be implemented in the named repo.

Rules (apply before building any PR):
1. An objective prefixed [owner/repo] MUST appear in exactly one PR whose `upstream` field
   is set to "owner/repo". Do not place it in a PR targeting a different repo.
2. An objective with NO prefix targets the primary repo ({repo}).
3. The ONLY valid reason to exclude an objective from all PRs is if it appears in
   OBJECTIVES CONFIRMED ALREADY PRESENT IN UPSTREAM below. "Doesn't belong to {repo}"
   is NOT a valid reason to drop an objective — it is a reason to route it to the correct
   upstream via the PR spec's `upstream` field.
4. If a prefixed objective has no corresponding patch file, still create the PR spec with
   `upstream` set to that repo and `in_scope` derived from the objective text — the rewriter
   will generate the diff. Never silently drop a non-satisfied objective.

UPSTREAM PATTERNS TO FOLLOW (naming conventions, base classes, registration patterns found in target repo):
{upstream_patterns_section}

OBJECTIVES CONFIRMED ALREADY PRESENT IN UPSTREAM — HARD BANNED FROM ALL PRS:
{already_satisfied_section}

CRITICAL: The items above were verified against the live upstream repo and confirmed ALREADY IMPLEMENTED.
Do NOT create any PR for these — not even a cleanup PR, not even a "minor improvement" PR.
If the diff contains code that implements a banned item, move it to scope_creep with recommendation "drop".
Creating a PR for a banned item is a correctness error that will cause immediate rejection.

CHANGES FLAGGED AS INCIDENTAL OR ALREADY SATISFIED UPSTREAM (must be DROPPED):
{excluded_section}

CRITICAL: Every item above is BANNED from appearing in any PR's in_scope or affected_files.
A prior verification step confirmed these are already present upstream or are incidental.
Do NOT include any diff hunks that implement these items in any PR — treat those code changes
as if they do not exist in the diff. If the diff contains changes that implement a banned item,
move those changes to scope_creep with recommendation "drop".

PATCH SUMMARY (files changed):
{file_summary}

FULL DIFF (first {diff_cap} chars — for context only):
{diff_truncated}

JUDGE FINDINGS:
{judge_section}

CONTRIBUTING GUIDANCE FOR THIS REPO:
{pr_prep_section}

CONTRIBUTION RULES FOR THIS REPO (how PRs must be structured):
{contribution_rules_section}

KNOWN DOWNSTREAM CONSUMERS (repos that import from this repo directly):
{downstream_section}

PATCH UPSTREAM GROUPS (which patches belong to which upstream repo):
{patch_upstream_section}

DEVELOPER NOTES (additional guidance from the submitter — follow these closely):
{notes_section}

---

## PHASE 0 — CONTRIBUTION SHAPE

Before splitting into PRs, apply the repo's contribution_rules:
- If the repo separates tuning data from kernel logic (separate_pr: true for tuning_data),
  plan a dedicated tuning PR that stacks AFTER all kernel PRs are complete.
- If tuning data must be merged into existing files (merge_into_existing: true),
  the tuning PR must modify those existing files, not create new ones.
- If the repo uses hardware gating via a specific mechanism (e.g. get_gfx()), ensure
  new hardware-specific code paths follow that mechanism — do not introduce feature flags.
- Include these shape decisions in each PR's rationale.

## PHASE 0b — OBJECTIVE VALIDITY CHECK  ← DO THIS BEFORE WRITING ANY PR

For EVERY stated objective, trace it to the root in the target repo before accepting it.
Do not assume the objective is valid just because it appears in the seed README.

Ask for each objective:
1. **Is the optimization already present upstream?**
   - Find every symbol the objective proposes to optimize. Search not only in the patched
     file but also in every module it imports — trace the import chain to the definition site.
   - At each definition site, check for existing memoization (`@lru_cache`, `@cache`,
     `functools.lru_cache`, module-level dicts, class-level caches, singleton patterns).
   - Check for the new code path, fast path, or dispatch branch the objective proposes to add.
   - If the optimization is already present anywhere in the import chain → DROP.

2. **Would this actually produce a measurable performance improvement?**
   - Identify the runtime call frequency: is this called once at import time, once per model
     load, or once per decode step? An optimization only matters if it fires on the hot path.
   - Identify whether the underlying cost is already eliminated by the runtime or framework
     (e.g., Python's memory allocator pools same-shape allocations, the OS caches syscalls).
   - If the optimization targets a path that is not a measurable bottleneck at runtime → DROP.

3. **Is the objective's premise true in the current upstream?**
   - Trace the actual call path in the upstream code to confirm the inefficiency or bug the
     objective claims to fix is real and not already corrected. If the premise is false → DROP.

RULE: Any objective that fails checks (1), (2), or (3) must be moved to `excluded_changes`
with a one-line explanation citing the upstream evidence. Only objectives that survive all
three checks proceed to PHASE 1.

## PHASE 1 — DOWNSTREAM SAFETY CHECK  ← DO THIS FIRST

Before anything else, identify every change in the diff that:
- Removes a function, class, or module-level name from a public API
- Removes, renames, or reorders parameters of a public function
- Changes the default value of a parameter that callers may rely on

For each such change, assume that known downstream consumers (listed above) may
be passing that parameter or calling that function. A parameter that has no
caller *inside this repo* may still be passed by vLLM, SGLang, or other consumers.

RULE: Any removal of public API surface (function parameters, exported names,
callable signatures) MUST be marked as DROP in scope_creep UNLESS you have
explicit evidence in the diff or stated objectives that the removal is safe for
all downstream consumers. The bar for "safe" is high: the parameter must not
appear anywhere in the known downstream consumers listed above.

IMPORTANT: When the seed uses a whole-file replacement, the diff will contain many
deletions that are upstream-cleanup artifacts, NOT intentional removals. Treat ALL
deletions with suspicion unless they are explicitly named in a stated objective above.
If a deletion is not named in any objective, mark it as DROP.

Perf and bugfix goals are ALWAYS achievable with purely ADDITIVE changes:
add a new dispatch path, add a cache, add a fast path — never remove existing ones.

## PHASE 2 — NECESSITY FILTER

IMPORTANT: The objectives listed above were extracted by reading the seed's README and
file listing, NOT from the diff. If the diff contains changes that do not map to any
objective above, they are incidental and must be DROPPED — do NOT promote them to a
new objective just because they appear in the diff.

For every remaining (non-dropped) logical group of changes, ask:
  "Which stated objective does this change directly serve?"

If the answer is "none" → recommend DROP. Do NOT create a separate PR for it.

## PHASE 3 — ATOMIC OBJECTIVE SPLIT

Rules:
1. **One coherent objective per PR.** A PR should be the smallest unit that
   is meaningfully reviewable on its own. Do NOT create a PR for a change so
   small it reads as a footnote (e.g. "add 2 module-level constants", "rename
   one variable"). Bundle micro-changes that share the same objective theme and
   the same reviewer mental model into one PR. A reviewer should be able to say
   "I understand this PR — it does X" in one sentence, not "it adds a constant,
   oh and also a dict, and also tweaks a heuristic."
2. **Additive-only by default.** PRs must add new code paths, caches, or fast
   paths on top of existing API. Never plan a PR that removes or renames public
   API surface (function parameters, exported names, callable signatures) unless:
   (a) the stated objectives explicitly require it, AND
   (b) you have confirmed the symbol does not appear in any known downstream consumer.
   When in doubt, keep the existing code and add alongside it. The existing code
   path can be deprecated by maintainers later — that is their call to make.
3. **Minimum richness floor — absorb rather than split.** Every PR must have
   at least one of:
   - A non-trivial function added or meaningfully rewritten (>10 lines of net
     new logic, not counting renamed or reformatted lines)
   - A measurable, testable behavioral change (kernel dispatch path, allocation
     pattern, numeric precision)
   If a proposed PR fails this floor, **fold its changes into the most thematically
   related PR in the series** — do not leave them as a standalone PR. Lean toward
   absorption: a reviewer inspecting a properly-sized PR can absorb a small related
   change at no extra cost; a separate trivial PR burns a full CI GPU run for nothing.
3b. **No scaffolding-only PRs.** Do NOT create a PR whose only changes are
   constants, imports, or variable substitutions that are exclusively consumed by
   another PR in this series. Such changes have zero observable impact without the
   consuming PR — they are internal implementation details and must be attached to
   the PR that actually uses them. Ask: "if this PR merged alone, would any
   existing behaviour change?" If no, merge it into the consuming PR.
3c. **CI cost discipline.** GPU CI runs are expensive. A PR that a reviewer
   would describe as "this just tweaks a constant" or "this only adds a comment"
   is not worth its own CI slot. When in doubt between splitting and merging,
   merge — the reviewer can always ask for a split; they cannot un-burn GPU time.
4. **Independence first.** Prefer PRs that all target `main`. Only use
   `stack_on_pr_index` when PR N literally cannot compile or run correctly without PR M.
5. **Minimum viable PR first.** Most conservative, obviously-correct change is PR 1.
6. **Label:** bugfix | perf | tuning | refactor | docs | new-feature
7. **Describe changes precisely.** In `in_scope`, name the exact functions,
   constants, and logic blocks to ADD or CHANGE — not file names alone.
   The rewriter will use these descriptions as instructions. Write "Add X" not
   "Remove Y and replace with X" — the rewriter must keep Y intact.
   When an objective replaces a call (e.g. replacing `get_gfx()` with a cached
   variable) across multiple functions in a file, enumerate EVERY function that
   must be modified by name. A missing function is a plan-consistency failure.
8. **Target 2–4 PRs for a typical patch.** Fewer, simpler PRs get reviewed
   faster. Consolidate PRs that share a theme (e.g. all caching changes → one
   PR, all new dispatch paths → one PR). If you reach 5+ PRs, stop and merge
   the most closely related pair before adding another. Removing dead code is
   almost never worth a standalone PR — maintainers do cleanup after perf PRs land.
   **Do not produce more than 5 PRs under any circumstances.**

9. **`model_layer_touches`** — for EVERY model-layer file a PR touches, you
   MUST produce one entry explaining WHY. Use one of these intents:

   - `"enables_compiler_pass"` — pure structural change that exposes a stable
     graph shape or entry point for a companion fusion pass. No new model logic.
     Requires `companion_pr_index` pointing to the pass PR.
   - `"wiring"` — adds/changes a config flag, imports a new op, or threads
     an argument through existing call sites. No new model behavior on its own.
   - `"refactor"` — moves or renames existing code without adding new behavior.
     Diff should show near-equal additions and deletions.
   - `"new_model_logic"` — genuinely adds new model-layer behavior (e.g. a new
     kernel dispatch path, a new normalization step). Use ONLY when the objective
     cannot be achieved via compiler pass. Justify in `rationale`.

   If a model-layer file is touched only to add an import or a one-line call to
   a compiler-pass op, classify as `"wiring"`. If you cannot determine why the
   model-layer file is touched, classify as `"new_model_logic"` — the audit will
   flag it for human review.

   Omit `model_layer_touches` (empty list or absent) if the PR touches no
   model-layer files (i.e. only kernel, compiler-pass, test, or config files).

10. **`commit_message`** — write a single commit message for the PR following
    conventional commits style (`type(scope): subject`). Include a short body
    paragraph. End with a `Signed-off-by:` trailer using a plausible author name
    and email (the developer who would submit this patch). Example:
    `Signed-off-by: Jane Smith <jsmith@example.com>`

11. **`upstream`** (REQUIRED) — set to the target upstream repo for this PR using the
    PATCH UPSTREAM GROUPS provided above. If a PR's affected_files all come from one
    upstream's patch set, set `upstream` to that repo. If this is a single-upstream seed,
    set `upstream` to `{repo}` for all PRs.
    CRITICAL RULES for multi-upstream seeds:
    - Each PR's `upstream` must match the upstream where its `affected_files` live.
    - Do NOT put cross-upstream objectives in `scope_creep` — assign them to the correct
      upstream. `scope_creep` is ONLY for changes that are genuinely out-of-scope for their
      assigned upstream (not useful to any objective of that upstream).
    - A patch file may only appear in ONE PR's `affected_files` — verify no double-assignment.
    - An objective prefixed [owner/repo] that is NOT in the already_satisfied list must
      appear in a PR. It may never be silently dropped. If there are no patch files for that
      upstream, still create the PR spec with upstream=owner/repo — the rewriter will produce
      the diff from the objective description.

12. **`new_files`** — for any file this PR creates that does NOT already exist upstream,
    add one entry per new file: `path`, `intent` (new_kernel|new_module|new_model|helper|test|config),
    `justification`. Leave empty list `[]` if no new files are created.

---
Respond with ONLY a JSON object:

{{
  "pr_series": [
    {{
      "index": 1,
      "label": "bugfix",
      "title": "Short PR title (<72 chars)",
      "commit_message": "type(scope): short subject line\n\nOne short paragraph describing the what and why.\n\nSigned-off-by: Author Name <author@example.com>",
      "objective": "One sentence: the single atomic thing this PR achieves.",
      "serves_objective": "Quote the specific stated objective this addresses.",
      "upstream": "owner/repo — the target upstream repo for this PR (from PATCH UPSTREAM GROUPS below, or {repo} if single-upstream)",
      "new_files": [
        {{
          "path": "relative/path/to/new_file.py",
          "intent": "new_kernel | new_module | new_model | helper | test | config",
          "justification": "Why this new file is needed to serve the objective."
        }}
      ],
      "in_scope": [
        "Exact description of what to ADD or CHANGE: function name, logic block, constant. E.g.: 'Remove the gfx950/M<256 conditional in q_dtype_a resolution; replace with: q_dtype_a = dtypes.fp4x2 if per_1x32 else q_dtype_a'",
        "Another precise change..."
      ],
      "out_of_scope": [
        "What must NOT be changed in this PR, and why."
      ],
      "affected_files": ["aiter/fused_moe.py"],
      "stack_on_pr_index": null,
      "rationale": "Why this boundary. Why can it be reviewed independently.",
      "cross_reference_note": "One sentence linking the series.",
      "model_layer_touches": [
        {{
          "file": "vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py",
          "intent": "enables_compiler_pass",
          "rationale": "Exposes forward_static so the fusion matcher in PR 3 can trace a stateless graph — no new model logic added.",
          "companion_pr_index": 3
        }}
      ]
    }}
  ],
  "scope_creep": [
    {{
      "description": "What the change does and which objective it was checked against.",
      "recommendation": "drop | revert | justify-and-keep",
      "reason": "Why it doesn't serve any stated objective for its assigned upstream, or why it regresses behaviour."
    }}
  ],
  "objectives_coverage": "One sentence confirming every stated objective is covered.",
  "summary": "2-3 sentences: how many PRs, what is dropped, merge order."
}}
"""


_VERIFY_PROMPT = """\
You are checking whether a list of proposed objectives are still needed, given the current
state of the upstream target repository.

PROPOSED OBJECTIVES (extracted from the seed README — what the seed claims to add or fix):
{objectives_section}

UPSTREAM FILE EXCERPTS (current state of the target repo — what already exists):
{excerpts_section}

---

For each proposed objective, decide ONE of:
  (a) confirmed    — the upstream does NOT already do this; the objective is genuinely new
  (b) already_satisfied — the upstream already does this (cite the exact code that satisfies it)
  (c) partial      — the upstream has something similar but incomplete (describe the gap)
  (d) wrong_upstream — the objective's target file does NOT exist in THIS upstream and appears
                       to belong to a different repo named in the seed (cite the suspected
                       target repo, e.g. "belongs to ROCm/aiter, not sgl-project/sglang").
                       CRITICAL: classify by the GIT PATH of the file being modified, NOT by
                       what that file imports. A file at python/sglang/srt/... belongs to
                       sgl-project/sglang even if it imports from aiter or ROCm libraries.
                       A change that calls into aiter but modifies a sglang file is a SGLANG
                       change, not an aiter change.

Rules:
- Be precise. "get_gfx() is already @lru_cached at line 42 of fused_moe.py" is a valid reason
  to mark an objective as already_satisfied. "it might be cached somewhere" is not.
- For new-model seeds (objectives that add new architecture): confirmed is almost always
  correct unless the upstream already has an identical model class or utility function.
- For optimization seeds: check decorators, module-level constants, and caching patterns
  carefully — these are the most common sources of false positives.
- Partial objectives should still be demoted (the planner will treat them as drop candidates).
  Describe the gap so the rewriter understands what remains to do.

CLAIM VALIDATION:

For optimization objectives (caching, fast paths, allocation reduction), also verify the
underlying claim — not just whether the feature exists, but whether the problem it purports
to solve is real:
- If the objective claims "X is called repeatedly and wasteful", check whether X already has
  @lru_cache, @functools.cache, or module-level caching at its DEFINITION SITE (which may be
  in an imported helper, not the file being patched). If found, mark already_satisfied.
- If the objective claims "torch.empty() / allocation Y is wasteful", apply the following
  reasoning carefully:
  (1) PyTorch's CUDA/ROCm caching allocator pools freed GPU blocks. torch.empty() of the
      same shape does NOT call hipMalloc after the first use — the allocator returns the
      cached block immediately.
  (2) The pooled block is only available AFTER the previous tensor holding that block has
      been released. In a real inference or training loop (vLLM, Megatron, SGLang, etc.),
      step N's output tensors are fully consumed and go out of scope before step N+1 begins
      allocating — so the allocator CAN reuse them and a Python-level buffer cache adds no
      GPU-side benefit.
  (3) A microbenchmark that holds references to previous-call tensors across iterations
      creates artificial lifetime overlap that does not exist in production. Benchmark data
      showing a non-zero "alloc delta per call" in such a setup is a measurement artifact,
      not evidence of a real problem.
  (4) The only legitimate gain from a Python-level tensor cache in this context is skipping
      the CPU-side allocator bookkeeping (free-list scan, metadata update) — a small µs-range
      benefit that is unlikely to be a meaningful bottleneck.
  Mark allocation-reduction objectives as already_satisfied unless the seed provides an
  end-to-end inference-loop benchmark (not a microbenchmark) showing a real latency delta.
- If the objective claims "a dispatch key mismatch causes lookups to fail", verify the mismatch
  is demonstrably visible in the upstream code (the key used in the call vs. the key in the
  lookup table must differ). If you cannot confirm the mismatch from the excerpts, mark partial.

Also identify any useful PATTERNS from the upstream excerpts that the rewriter should follow:
naming conventions, base classes to inherit from, registration patterns, decorator usage, etc.

Respond with ONLY a JSON object:

{{
  "confirmed": [
    "Objective text — reason it is not yet present upstream"
  ],
  "already_satisfied": [
    "Objective text — upstream already does this at <file>:<line> because <reason>"
  ],
  "partial": [
    "Objective text — upstream has <X> but is missing <Y>"
  ],
  "wrong_upstream": [
    "Objective text — target file <path> not present in this upstream; belongs to <owner/repo>"
  ],
  "upstream_patterns": [
    "Useful pattern the rewriter should follow: e.g. 'all hardware gating uses get_gfx() string compare, not enums'"
  ]
}}
"""


def _load_repo_config(slug: str) -> dict:
    config_path = GOLD / slug / "repo_config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text()) or {}
    return {}


def extract_intent(
    readme: str,
    file_names: list[str],
    content_excerpts: dict[str, str],
    *,
    layer_map: "dict | None" = None,
    model: str = "claude-opus-4-7",
) -> dict:
    """Stage 0: extract objectives from the seed README and file listing — diff-blind.

    Returns dict with keys:
        objectives:        list of plain-English objective sentences
        excluded_changes:  list of incidental changes to drop
        target_repo_hint:  best-guess target repo slug or None
        summary:           human-readable 2-3 sentence summary
    """
    from pipeline.llm import llm_call, make_client, parse_json

    readme_section = readme[:4000] if readme else "(no README found)"

    # Annotate file names with layer if a layer_map was provided
    if layer_map and layer_map.get("by_file"):
        by_file = layer_map["by_file"]
        annotated = []
        for n in file_names:
            layer = by_file.get(n, "unknown")
            annotated.append(f"  - {n}  [layer: {layer}]")
        file_list = "\n".join(annotated) or "  (none)"
    else:
        file_list = "\n".join(f"  - {n}" for n in file_names) or "  (none)"

    excerpts_parts = []
    for name, text in content_excerpts.items():
        lines = text.splitlines()[:80]
        excerpts_parts.append(f"### {name}\n" + "\n".join(lines))
    excerpts_section = "\n\n".join(excerpts_parts) or "(no excerpts available)"

    # Build layer summary section for the prompt
    layer_prompt_section = ""
    if layer_map and layer_map.get("by_layer"):
        by_layer = layer_map["by_layer"]
        layer_parts = []
        for layer_name, files in sorted(by_layer.items()):
            layer_parts.append(f"  {layer_name}: {', '.join(files)}")
        if layer_parts:
            layer_summary = "\n".join(layer_parts)
            layer_prompt_section = (
                "\n\nARCHITECTURAL LAYER CLASSIFICATION:\n"
                + layer_summary
                + "\n\nFor this repository, kernel-fusion and perf improvements should prefer the"
                " compiler-pass layer (vllm/compilation/passes/). When model-layer files"
                " (vllm/model_executor/models/, vllm/model_executor/layers/) appear alongside"
                " compiler-pass files, ask for EACH objective: is the model-layer file the"
                " PRIMARY mechanism, or is it wiring that a compiler pass could instead achieve"
                " by pattern-matching the natural forward-pass graph? Mark model-layer wiring"
                " as excluded_changes if a pass approach is feasible; the pipeline will run a"
                " sufficiency check to confirm."
            )

    base_prompt = _INTENT_PROMPT.format(
        readme_section=readme_section,
        file_list=file_list,
        excerpts_section=excerpts_section,
    )
    # Insert layer section before the closing JSON instruction block
    if layer_prompt_section:
        # Append before the final "Respond with ONLY a JSON object" marker
        split_marker = "\nRespond with ONLY a JSON object:"
        if split_marker in base_prompt:
            prompt = base_prompt.replace(split_marker, layer_prompt_section + split_marker, 1)
        else:
            prompt = base_prompt + layer_prompt_section
    else:
        prompt = base_prompt

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=True)
    result = parse_json(raw)

    if not isinstance(result, dict):
        logger.warning("extract_intent returned non-dict: %s", type(result))
        return {"objectives": [], "excluded_changes": [], "target_repo_hint": None, "summary": ""}

    result.setdefault("objectives", [])
    result.setdefault("excluded_changes", [])
    result.setdefault("target_repo_hint", None)
    result.setdefault("summary", "")
    return result


def extract_intent_rlm(
    readme: str,
    file_names: list[str],
    content_excerpts: dict[str, str],
    target_repo: str,
    token: str,
    *,
    layer_map: "dict | None" = None,
    model: str = "claude-opus-4-7",
    max_iters: int = 40,
) -> dict:
    """Stage 0a (RLM variant): iteratively extract objectives from seed material + upstream.

    Uses a DSPy ReAct agent with fetch_file / search_symbol / grep_repo tools so the
    agent can look up upstream architecture (existing passes, base classes, registration
    patterns) before committing to its final list of objectives.  This avoids the static
    excerpt pre-fetch that misses context 2+ imports deep.

    Falls back to extract_intent() if dspy is unavailable or the agent errors.

    Returns same schema as extract_intent().
    """
    try:
        import dspy
    except ImportError:
        logger.warning("dspy not available — falling back to static extract_intent")
        return extract_intent(readme, file_names, content_excerpts, layer_map=layer_map, model=model)

    import base64
    import httpx as _httpx

    owner, repo_name = target_repo.split("/", 1)
    _headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    _fetched_cache: dict[str, str] = {}

    def _gh_get(url: str) -> dict | list | None:
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("GitHub API error %s: %s", url, exc)
            return None

    def _gh_get_with_status(url: str) -> tuple[dict | list | None, int | None]:
        """Like _gh_get but exposes the HTTP status (or None for transport error)
        so callers can distinguish 404 vs 403/429 vs network blip."""
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            status = r.status_code
            if status >= 400:
                logger.debug("GitHub API %s -> %s", url, status)
                return (None, status)
            return (r.json(), status)
        except Exception as exc:
            logger.debug("GitHub API transport error %s: %s", url, exc)
            return (None, None)

    def fetch_file(path: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Fetch a slice of a file from the upstream repo.
        For large files, call iteratively with increasing start_line to page through:
          1. First call (start_line=0) returns the header showing total line count.
          2. Call again with start_line=200, 400, etc. to read further sections.
        Args:
            path: file path relative to repo root
            start_line: 0-indexed line to start from (default 0)
            num_lines: number of lines to return (default 200)
        Returns a [Lines X–Y of Z] header followed by content, or an error message."""
        cache_key = path
        if cache_key not in _fetched_cache:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            data, status = _gh_get_with_status(url)
            # Retry ONCE on 404 / transient transport error to dodge GitHub Contents API flakes.
            # Do NOT retry on 403/429 (rate limit) — those will not recover in 1s.
            if (status == 404 or status is None) and not (isinstance(data, dict) and data.get("encoding") == "base64"):
                import time as _time
                _time.sleep(1.0)
                data, status = _gh_get_with_status(url)
            if status in (403, 429):
                # Do NOT cache rate-limit results — they are transient.
                return f"(fetch rate-limited for {path} — treat as unknown, do not conclude absent)"
            if not data or not isinstance(data, dict) or data.get("encoding") != "base64":
                # Do NOT cache negative results — let later calls retry.
                logger.info("fetch_file 404: %s — confirmed absent on retry (url=%s)", path, url)
                return f"(file not found at {path} on default branch; verified via {url})"
            try:
                import base64 as _b64
                text = _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
                _fetched_cache[cache_key] = text
            except Exception as exc:
                return f"(decode error: {exc})"
        lines = _fetched_cache[cache_key].splitlines()
        total = len(lines)
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(chunk)
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def search_symbol(symbol: str) -> str:
        """Search for a symbol (function, class, decorator, or pass name) in the upstream repo.
        Use this to find where compiler passes, model layers, or kernel dispatch functions are
        defined — critical for judging whether an objective should target a pass or a model file.
        Returns matching file paths."""
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {symbol})"
        items = data.get("items", [])
        if not items:
            return f"(not found: {symbol})"
        return "\n".join(f"  {item['path']}" for item in items[:5])

    def grep_repo(pattern: str) -> str:
        """Search for a regex pattern or string literal in the upstream repo.
        Use this to find existing registration patterns, decorator usages, or
        import chains relevant to the objectives.
        Returns file paths where the pattern was found."""
        url = f"https://api.github.com/search/code?q={pattern}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {pattern})"
        items = data.get("items", [])
        if not items:
            return "(not found)"
        return "\n".join(f"  {i['path']}" for i in items[:5])

    def read_patch(name: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Read lines from a seed patch file to understand what the diff actually changes.
        Call this iteratively with increasing start_line to page through large diffs.
        Args:
            name: patch file name (as listed in FILES IN SEED)
            start_line: 0-indexed line to start from (default 0)
            num_lines: lines to return per page (default 200; no hard cap — page as needed)
        Returns the lines with numbers and a total-line count."""
        if name not in content_excerpts:
            return f"(patch not found: {name}. Available: {list(content_excerpts.keys())})"
        lines = content_excerpts[name].splitlines()
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {len(lines)} lines total)"
        numbered = "\n".join(f"{start_line + i + 1:4d}  {l}" for i, l in enumerate(chunk))
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {len(lines)}]\n{numbered}"

    # Configure DSPy
    import os
    gateway = os.environ.get("LITELLM_GATEWAY_URL", "http://localhost:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    _lm = dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
        temperature=0.2,
    )

    # Build static context blocks (same as extract_intent)
    readme_section = readme[:4000] if readme else "(no README found)"
    if layer_map and layer_map.get("by_file"):
        by_file = layer_map["by_file"]
        file_list = "\n".join(
            f"  - {n}  [layer: {by_file.get(n, 'unknown')}]" for n in file_names
        ) or "  (none)"
    else:
        file_list = "\n".join(f"  - {n}" for n in file_names) or "  (none)"

    # List available patch names so the RLM knows what to call read_patch on.
    # No pre-fetched excerpts — the RLM pages through files on demand with read_patch.
    available_patches = "\n".join(f"  - {n}" for n in content_excerpts) or "  (none)"

    layer_guidance = ""
    if layer_map and layer_map.get("by_layer"):
        by_layer = layer_map["by_layer"]
        layer_parts = [
            f"  {layer_name}: {', '.join(files)}"
            for layer_name, files in sorted(by_layer.items())
        ]
        if layer_parts:
            layer_guidance = (
                "\n\nARCHITECTURAL LAYER CLASSIFICATION:\n"
                + "\n".join(layer_parts)
                + "\n\nNote which files belong to which layer. Describe objectives for ALL"
                " layers — architectural suitability is evaluated separately."
            )

    seed_context = (
        f"SEED README:\n{readme_section}\n\n"
        f"FILES IN SEED:\n{file_list}\n\n"
        f"AVAILABLE PATCH FILES (call read_patch to read any of these):\n{available_patches}"
        + layer_guidance
    )

    class IntentSignature(dspy.Signature):
        """You are analyzing a seed folder to extract every code-level change it makes.
        Your job is to faithfully describe what the seed does — not to judge whether it
        belongs upstream or whether it "generalizes". Those judgments happen later.

        REQUIRED TOOL USE — do not skip any step:
        1. Call read_patch on EVERY file listed in AVAILABLE PATCH FILES to understand
           what actually changed. For large files, call with increasing start_line (0, 150,
           300, …) until you have read every hunk. The first call shows total line count.
           Do NOT rely only on the README — the README may be incomplete or misleading.
        2. For CSV, YAML, JSON, or any data file: read its full content with read_patch.
           Describe the data structure (columns, row count, sample values) and embed the
           full data verbatim in the objective (see DATA EMBEDDING below).
        3. Call fetch_file on any upstream file referenced by the seed diff to understand
           existing architecture (base classes, compiler pass structure, kernel registration).
           For large files, call fetch_file iteratively with increasing start_line.
        4. Call search_symbol for any new symbol introduced in the diff to confirm it does
           not already exist upstream.
        5. Call grep_repo on import patterns to verify layer boundaries.

        DELETION VALIDATION — required before finalizing objectives:
        For each file where the patch contains deletions (lines starting with "-"):
        a. Read the deleted lines via read_patch.
        b. Call fetch_file on the same upstream path to see the current upstream state.
        c. Classify each deletion:
             - Coherent with a stated objective (replacing old impl with new) → include in that objective
             - Local cleanup the author did incidentally, unrelated to stated goals → add to excluded_changes
        Do NOT promote incidental deletions into objectives. A whole-file replacement
        will contain many deletions that are simply the author's local cleanup — exclude them.

        Rules:
        - Objectives describe what the seed changes, taken from README intent AND actual diff content.
        - Prefer fewer, narrower objectives over broad ones.
        - Include ALL changes the seed makes — kernel parameters, lookup tables, tuning data,
          algorithmic constants, dispatch logic, etc. Do NOT exclude a change because it looks
          model-specific or hardware-specific. Upstream generalizability is NOT your concern.
        - If model-layer files appear alongside compiler-pass files, fetch both and describe
          the model-layer changes as objectives; note which are wiring vs. primary logic.

        DATA EMBEDDING — critical for non-code changes:
        The rewriter is a code LLM. It can implement algorithmic changes from a description,
        but it CANNOT reproduce externally measured data (tuned kernel parameters, benchmark
        results, lookup table values, calibration coefficients, etc.).
        For any objective whose change is data-driven (CSV rows, JSON lookup tables, constant
        tables derived from hardware runs), you MUST embed the full data in the objective
        description so the rewriter can reproduce the change exactly:
          - Include every row/value from the seed's data artifact files verbatim.
          - Format as a fenced code block with the correct file extension so the rewriter
            knows the exact content to write (e.g. ```csv ... ```).
          - Do NOT paraphrase or summarize — the rewriter needs the raw values.
        Example: if the seed adds 29 tuned CSV rows, the objective must contain all 29 data rows
        verbatim (no header/column-name line — the upstream file already has its own structure
        and the rewriter must append data rows only),
        not a description like "add rows for model_dim=3072".

        BUDGET SKILL — maintain throughout your session:
          At the start of your REPL session run:
              budget_used = 0; budget_cap = 150
          Before every llm_query() call check:
              if budget_used >= budget_cap * 0.8:
                  # Low budget — skip llm_query(); reason from already-fetched text instead.
                  pass
          After every llm_query() call: budget_used += 1

        Return a JSON object: {objectives: [...], excluded_changes: [...],
                               target_repo_hint: "owner/name or null", summary: "..."}"""

        seed_context: str = dspy.InputField(desc="Seed README, file listing, and file excerpts")
        upstream_repo: str = dspy.InputField(desc="Target upstream repo (owner/name)")
        result: str = dspy.OutputField(
            desc='JSON: {"objectives": [...], "excluded_changes": [...], "target_repo_hint": "...", "summary": "..."}'
        )

    rlm = dspy.RLM(
        IntentSignature,
        tools=[fetch_file, search_symbol, grep_repo, read_patch],
        max_iterations=max_iters,
        max_llm_calls=150,
    )

    try:
        with dspy.context(lm=_lm):
            prediction = rlm(seed_context=seed_context, upstream_repo=target_repo)
        raw = prediction.result
        from pipeline.llm import parse_json
        result = parse_json(raw)
        if not isinstance(result, dict):
            raise ValueError(f"non-dict result: {type(result)}")
        result.setdefault("objectives", [])
        result.setdefault("excluded_changes", [])
        result.setdefault("target_repo_hint", None)
        result.setdefault("summary", "")
        logger.info(
            "RLM extract_intent: %d objectives, %d excluded (fetched %d upstream files)",
            len(result["objectives"]), len(result["excluded_changes"]), len(_fetched_cache),
        )
        return result
    except Exception as exc:
        raise RuntimeError(f"RLM extract_intent failed: {exc}") from exc


def verify_objectives(
    objectives: list[str],
    excluded_changes: list[str],
    upstream_file_excerpts: dict[str, str],
    *,
    model: str = "claude-opus-4-7",
) -> dict:
    """Stage 0b: check which objectives are already satisfied by the upstream target repo.

    upstream_file_excerpts: {upstream_path: first ~200 lines of content}

    Returns dict with keys:
        confirmed:          objectives genuinely not yet present upstream
        already_satisfied:  objectives the upstream already covers (with citation)
        partial:            objectives where upstream has something similar but incomplete
        upstream_patterns:  useful patterns the rewriter should follow
    """
    from pipeline.llm import llm_call, make_client, parse_json

    if not objectives:
        return {"confirmed": [], "already_satisfied": [], "partial": [], "wrong_upstream": [], "wrong_file_in_repo": [], "upstream_patterns": []}

    objectives_section = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(objectives))

    if upstream_file_excerpts:
        parts = []
        for path, content in upstream_file_excerpts.items():
            parts.append(f"### {path}\n```\n{content}\n```")
        excerpts_section = "\n\n".join(parts)
    else:
        excerpts_section = "(no upstream files fetched — treat all objectives as confirmed)"

    prompt = _VERIFY_PROMPT.format(
        objectives_section=objectives_section,
        excerpts_section=excerpts_section,
    )

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=True)
    result = parse_json(raw)

    if not isinstance(result, dict):
        logger.warning("verify_objectives returned non-dict: %s — treating all as confirmed", type(result))
        return {"confirmed": objectives, "already_satisfied": [], "partial": [], "wrong_upstream": [], "wrong_file_in_repo": [], "upstream_patterns": []}

    result.setdefault("confirmed", objectives)
    result.setdefault("already_satisfied", [])
    result.setdefault("partial", [])
    result.setdefault("wrong_upstream", [])
    result.setdefault("wrong_file_in_repo", [])
    result.setdefault("upstream_patterns", [])
    return result


def verify_objectives_rlm(
    objectives: list[str],
    excluded_changes: list[str],
    target_repo: str,
    token: str,
    *,
    model: str = "claude-opus-4-7",
    max_iters: int = 40,
    seed_files: dict[str, str] | None = None,
) -> dict:
    """Stage 0b (RLM variant): iteratively verify objectives against upstream.

    Uses a DSPy ReAct agent with fetch_file / search_symbol / grep_repo tools
    so the agent can trace imports and look up definitions on demand — avoiding
    the static excerpt fetch that misses symbols defined in imported helpers.

    Falls back to verify_objectives() if dspy is unavailable.

    Returns same schema as verify_objectives().
    """
    if not objectives:
        return {"confirmed": [], "already_satisfied": [], "partial": [], "wrong_upstream": [], "wrong_file_in_repo": [], "upstream_patterns": []}

    try:
        import dspy
    except ImportError:
        logger.warning("dspy not available — falling back to static verify_objectives")
        _fallback = verify_objectives(objectives, excluded_changes, {}, model=model)
        _fallback.setdefault("wrong_file_in_repo", [])
        return _fallback

    import re
    import httpx as _httpx

    owner, repo_name = target_repo.split("/", 1)
    _headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    _fetched_cache: dict[str, str] = {}

    def _gh_get(url: str) -> dict | list | None:
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("GitHub API error %s: %s", url, exc)
            return None

    def _gh_get_with_status(url: str) -> tuple[dict | list | None, int | None]:
        """Like _gh_get but exposes the HTTP status (or None for transport error)
        so callers can distinguish 404 vs 403/429 vs network blip."""
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            status = r.status_code
            if status >= 400:
                logger.debug("GitHub API %s -> %s", url, status)
                return (None, status)
            return (r.json(), status)
        except Exception as exc:
            logger.debug("GitHub API transport error %s: %s", url, exc)
            return (None, None)

    def fetch_file(path: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Fetch a slice of a file from the upstream repo.
        For large files call iteratively: first call shows [Lines 1–N of TOTAL];
        continue with start_line=200, 400, … until you've read all relevant sections.
        IMPORTANT: Always read far enough to see all decorators — @lru_cache and similar
        often appear many lines after the def statement.
        Args:
            path: file path relative to repo root
            start_line: 0-indexed line to start from (default 0)
            num_lines: lines to return per call (default 200)
        Returns [Lines X–Y of Z] header + content, or an error message."""
        if path not in _fetched_cache:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            data, status = _gh_get_with_status(url)
            # Retry ONCE on 404 / transient transport error to dodge GitHub Contents API flakes.
            # Do NOT retry on 403/429 (rate limit) — those will not recover in 1s.
            if (status == 404 or status is None) and not (isinstance(data, dict) and data.get("encoding") == "base64"):
                import time as _time
                _time.sleep(1.0)
                data, status = _gh_get_with_status(url)
            if status in (403, 429):
                # Do NOT cache rate-limit results — they are transient.
                return f"(fetch rate-limited for {path} — treat as unknown, do not conclude absent)"
            if not data or not isinstance(data, dict) or data.get("encoding") != "base64":
                # Check if this is a new file added by the seed PR (not yet in upstream).
                if seed_files:
                    _seed_content = seed_files.get(path, "")
                    if _seed_content:
                        _fetched_cache[path] = _seed_content
                        logger.info("fetch_file seed-fallback: %s — serving from seed content (%d chars)", path, len(_seed_content))
                        # Fall through to the slice-return below — _fetched_cache[path] is now set.
                    else:
                        logger.info("fetch_file 404: %s — confirmed absent on retry (url=%s)", path, url)
                        return f"[NEW FILE — not yet in upstream {owner}/{repo_name}]\n(file not found at {path}; no seed content available)"
                else:
                    logger.info("fetch_file 404: %s — confirmed absent on retry (url=%s)", path, url)
                    return f"(file not found at {path} on default branch; verified via {url})"
            import base64
            try:
                text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                _fetched_cache[path] = text
            except Exception as exc:
                return f"(decode error: {exc})"
        lines = _fetched_cache[path].splitlines()
        total = len(lines)
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(chunk)
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def search_symbol(symbol: str) -> str:
        """Search for a symbol (function, class, or decorator name) across the upstream repo.
        Use this when you see an import or reference to a name and want to know where it's defined.
        Returns a list of matching file paths and line snippets, or 'not found'."""
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {symbol})"
        items = data.get("items", [])
        if not items:
            return f"(not found: {symbol})"
        results = []
        for item in items[:5]:
            results.append(f"  {item['path']}")
        return "\n".join(results)

    def grep_repo(pattern: str) -> str:
        """Search for a regex pattern in the upstream repo source code.
        Use this to find where a specific string, decorator, or function signature appears.
        Returns file paths where the pattern was found."""
        url = f"https://api.github.com/search/code?q={pattern}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {pattern})"
        items = data.get("items", [])
        if not items:
            return "(not found)"
        return "\n".join(f"  {i['path']}" for i in items[:5])

    # Configure DSPy with the LiteLLM gateway
    import os
    gateway = os.environ.get("LITELLM_GATEWAY_URL", "http://localhost:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    _lm = dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
        temperature=0.2,
    )

    objectives_text = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(objectives))

    class VerifySignature(dspy.Signature):
        """You are verifying whether each objective in a list is (a) already implemented
        upstream and (b) would actually provide a meaningful performance improvement.

        REQUIRED TOOL USE — do not skip:
        1. For every objective, call search_symbol for the key function/class/decorator name.
        2. Call fetch_file on EVERY file returned by the search. For large files, call
           iteratively with increasing start_line (0, 200, 400, …) — the first call shows
           "[Lines 1–N of TOTAL]" so you know how much more to read. Continue until you
           have covered the entire file or confirmed the symbol/decorator you need.
        3. For any symbol referenced in the objective, trace its import chain:
           - Find the import statement in the patched file
           - Call fetch_file on the imported module to find the actual definition
           - Check if the definition already has @lru_cache, @cache, a module-level dict,
             or any other caching/memoization applied at the definition site
           MANDATORY — for ANY objective that proposes to cache a value to avoid
           repeated computation or allocation, identify the UNDERLYING SOURCE of that
           value (the function being called, the allocation being avoided) and verify
           it at its definition site, not just at the call site. Examples:
           - Objective "cache get_gfx() in a module variable" → fetch chip_info.py,
             check whether get_gfx() itself has @lru_cache. If yes: already_satisfied.
           - Objective "cache moe_sorting buffers to avoid torch.empty re-allocation"
             → check whether PyTorch's caching allocator already reuses same-shape
             tensors between calls (it does for same device/dtype/shape). If yes and
             the objective provides no end-to-end benchmark proving GPU benefit: already_satisfied.
           - Objective "pre-resolve kernel name at import time" → check whether the
             kernel lookup is already O(1) (dict lookup, @lru_cache). If yes: already_satisfied.
           Exception: if the objective also changes BEHAVIOR (not just avoids overhead),
           it must be confirmed even if the overhead it targets is already cached elsewhere.
        4. Call grep_repo for the exact decorator or pattern (e.g. "lru_cache" "functools.cache")
           to catch caching applied anywhere in the import chain.
        5. For objectives that add data to an existing data store (CSV rows, JSON entries,
           YAML blocks, Python lookup-table entries, config file sections, etc.):

           GENERAL RULE — burden of proof is on already_satisfied/partial, NOT on confirmed:
           - If the file does not exist at all → confirmed (data must be created).
           - If the file exists but you cannot confirm the EXACT entries are present → confirmed.
             Do not classify as partial just because the file exists or has related entries.
             Only classify as partial when you have positive evidence that SOME (but not all)
             of the exact requested data is already there.
           - Only classify as already_satisfied when you can PROVE the exact entries exist
             (file fetched, grep confirmed, every specified key/dimension/value matched).
           The rewriter appending data that already exists is a cheap mistake to fix;
           silently dropping an objective is invisible and permanent.

           VERIFICATION STEPS (adapt to the data format):
           - Fetch the target file with fetch_file (read the full file iteratively if needed).
           - Use grep_repo to search for the EXACT identifying values from the objective —
             not just a filename or a related value, but the specific combination of keys
             that would uniquely identify the requested entries.
           - Entries for DIFFERENT key combinations do NOT satisfy the objective.
             Only the exact match counts.

           Examples by format:
           - CSV (e.g. "Add tuned GEMM rows for N=3072, K=768"):
             grep for ",3072," AND ",768," together; rows with N=7168 or K=384 do not count.
           - JSON lookup table (e.g. "Add kernel config for dtype=fp8, arch=gfx950"):
             grep for "fp8" AND "gfx950" in the same block; a gfx942 entry does not count.
           - YAML config (e.g. "Add section for model_type=minimax_m25"):
             grep for "minimax_m25"; a "minimax_m2" or "minimax" partial key does not count.
           - Python dict / registry (e.g. "Register kernel for (QuantType.per_1x128, bf16)"):
             grep for the exact tuple; a (per_tensor, bf16) entry does not count.
        6. After establishing what upstream does, ask for each objective:
           PERF VALIDITY: "Even if not already done, would this change produce a real,
           measurable GPU performance improvement?" Consider:
           - If the function is already @lru_cache'd at definition, calling it once per
             request has zero overhead — caching the call site adds nothing.
           - If PyTorch's caching allocator already pools same-shape torch.empty() calls,
             a Python-level buffer cache adds no GPU benefit.
           - If the optimization only fires on a code path that is never the bottleneck
             (e.g. import-time setup), it has no runtime perf impact.
           If perf benefit is negligible or already handled, classify as already_satisfied.

        For each objective determine:
          confirmed       — not yet upstream AND gives real perf benefit; also use this
                             when you cannot confirm the exact data entries are present
          already_satisfied — upstream already has it OR the perf benefit is negligible
                             (cite file:line and one-sentence reason in both cases)
          partial         — you have positive evidence that SOME (but not all) of the
                             exact requested data/code is already present; do NOT use
                             this just because the file exists or has related entries
          wrong_upstream   — the objective's target file does NOT exist in THIS upstream
                             and appears to belong to a different repo named in the seed
                             (cite the suspected target repo, e.g. "belongs to ROCm/aiter,
                             not sgl-project/sglang"). Use this when fetch_file/search_symbol
                             return 404 for the file AND the seed clearly tags or references
                             a different upstream as its home.
                             CRITICAL: classify by the GIT PATH of the file being modified,
                             NOT by what that file imports. python/sglang/srt/layers/communicator.py
                             belongs to sgl-project/sglang even if it imports from aiter.
                             A change that calls into aiter but edits a sglang file is a SGLANG
                             objective. Only mark wrong_upstream when the file PATH itself is
                             absent from this repo (fetch_file returns 404).
          new_file_confirmed — SPECIAL CASE: if fetch_file returns a "[NEW FILE]" header,
                             the file is newly introduced by this seed and does not yet exist
                             upstream. This means the objective is VALID and IMPLEMENTABLE —
                             classify it as confirmed. A new file that doesn't exist upstream
                             yet is not wrong_upstream or wrong_file_in_repo; it is simply new.

          wrong_file_in_repo — the objective's target file path does NOT exist in THIS
                             upstream, but the objective itself does belong here (no
                             clear other-repo signal). When fetch_file returns a plain
                             "(file not found...)" message (no [NEW FILE] header) for a
                             file path in this repo:
                             (1) use search_symbol on any named function, class, or dict
                                 key mentioned in the objective to find the correct file;
                             (2) use grep_repo on distinctive string literals from the
                                 objective;
                             (3) if a match is found, return wrong_file_in_repo with the
                                 found path as suggested_file_path;
                             (4) if no match is found and there's no other-repo signal,
                                 still return wrong_file_in_repo with suggested_file_path
                                 null rather than marking the objective as confirmed.
                             Each wrong_file_in_repo entry must be an object:
                             {"objective_index": N, "objective_text": "...",
                              "original_path": "...", "suggested_file_path": "..." or null}

        BUDGET SKILL — maintain throughout your session:
          At the start of your REPL session run:
              budget_used = 0; budget_cap = 150
          Before every llm_query() call check:
              if budget_used >= budget_cap * 0.8:
                  # Low budget — skip llm_query(); reason from already-fetched text instead.
                  pass
          After every llm_query() call: budget_used += 1

        Return a JSON object with keys: confirmed, already_satisfied, partial, wrong_upstream, wrong_file_in_repo, upstream_patterns."""

        objectives: str = dspy.InputField(desc="Numbered list of objectives to verify")
        repo: str = dspy.InputField(desc="Target upstream repo (owner/name)")
        result: str = dspy.OutputField(
            desc="JSON object: {confirmed: [...], already_satisfied: [...], partial: [...], wrong_upstream: [...], wrong_file_in_repo: [{objective_index, objective_text, original_path, suggested_file_path}], upstream_patterns: [...]}"
        )

    rlm = dspy.RLM(
        VerifySignature,
        tools=[fetch_file, search_symbol, grep_repo],
        max_iterations=max_iters,
        max_llm_calls=150,
    )

    try:
        with dspy.context(lm=_lm):
            prediction = rlm(objectives=objectives_text, repo=target_repo)
        raw = prediction.result
        from pipeline.llm import parse_json
        result = parse_json(raw)
        if not isinstance(result, dict):
            raise ValueError(f"non-dict result: {type(result)}")
        def _coerce_str_list(lst: list, fallback_list: list | None = None) -> list[str]:
            result_items = []
            for item in (lst or []):
                if isinstance(item, dict):
                    # Prefer objective_text, then title, then id — never str(dict) which produces unparseable output
                    result_items.append(
                        item.get("objective_text") or item.get("title") or item.get("id") or str(item)
                    )
                elif isinstance(item, int) and fallback_list and 1 <= item <= len(fallback_list):
                    fb = fallback_list[item - 1]
                    result_items.append(fb if isinstance(fb, str) else str(fb))
                else:
                    result_items.append(str(item))
            return result_items

        result.setdefault("confirmed", objectives)
        result.setdefault("already_satisfied", [])
        result.setdefault("partial", [])
        result.setdefault("wrong_upstream", [])
        result.setdefault("wrong_file_in_repo", [])
        result.setdefault("upstream_patterns", [])
        result["confirmed"] = _coerce_str_list(result["confirmed"], fallback_list=objectives)
        result["already_satisfied"] = _coerce_str_list(result["already_satisfied"], fallback_list=objectives)
        result["partial"] = _coerce_str_list(result["partial"], fallback_list=objectives)
        result["wrong_upstream"] = _coerce_str_list(result["wrong_upstream"], fallback_list=objectives)
        # wrong_file_in_repo is a list of objects, not strings — normalize shape only.
        _wfir_normalized: list[dict] = []
        for _item in (result.get("wrong_file_in_repo") or []):
            if isinstance(_item, dict):
                _wfir_normalized.append({
                    "objective_index": _item.get("objective_index"),
                    "objective_text": str(_item.get("objective_text") or ""),
                    "original_path": str(_item.get("original_path") or ""),
                    "suggested_file_path": _item.get("suggested_file_path"),
                })
            elif isinstance(_item, str):
                _wfir_normalized.append({
                    "objective_index": None,
                    "objective_text": _item,
                    "original_path": "",
                    "suggested_file_path": None,
                })
        result["wrong_file_in_repo"] = _wfir_normalized
        # RLM sometimes returns upstream_patterns as a single string — coerce to list
        if isinstance(result["upstream_patterns"], str):
            result["upstream_patterns"] = [result["upstream_patterns"]] if result["upstream_patterns"].strip() else []
        logger.info("RLM verify_objectives: %d confirmed, %d satisfied, %d partial, %d wrong_upstream, %d wrong_file_in_repo (fetched %d files)",
                    len(result["confirmed"]), len(result["already_satisfied"]),
                    len(result["partial"]), len(result["wrong_upstream"]),
                    len(result["wrong_file_in_repo"]), len(_fetched_cache))
        for item in result["already_satisfied"]:
            logger.info("  [dropped] %s", item)
        for item in result["confirmed"]:
            logger.info("  [confirmed] %s", item)
        for item in result["wrong_upstream"]:
            logger.info("  [wrong_upstream] %s", item)
        for item in result["wrong_file_in_repo"]:
            logger.info("  [wrong_file_in_repo] %s (orig=%s suggested=%s)",
                        item.get("objective_text", "")[:120],
                        item.get("original_path", ""),
                        item.get("suggested_file_path"))
        return result
    except Exception as exc:
        raise RuntimeError(f"RLM verify_objectives failed: {exc}") from exc




def _file_summary(diff: str) -> str:
    lines = []
    current_file = ""
    hunk_count = add_count = del_count = 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current_file:
                lines.append(f"  {current_file}: {hunk_count} hunks (+{add_count}/-{del_count})")
            current_file = line.split(" b/", 1)[-1] if " b/" in line else line
            hunk_count = add_count = del_count = 0
        elif line.startswith("@@"):
            hunk_count += 1
        elif line.startswith("+") and not line.startswith("+++"):
            add_count += 1
        elif line.startswith("-") and not line.startswith("---"):
            del_count += 1
    if current_file:
        lines.append(f"  {current_file}: {hunk_count} hunks (+{add_count}/-{del_count})")
    return "\n".join(lines) if lines else "(no files detected)"


def plan_prs(
    repo: str,
    diff: str,
    *,
    objectives: list[str] | None = None,
    excluded_changes: list[str] | None = None,
    already_satisfied: list[str] | None = None,
    upstream_patterns: list[str] | None = None,
    judge_findings: dict | None = None,
    layer_audit_warnings: list[str] | None = None,
    notes: str = "",
    model: str = "claude-opus-4-7",
    patch_upstream_groups: dict[str, list[str]] | None = None,
) -> dict:
    """Plan a PR series from a unified diff.

    Pure planning step — produces prose descriptions of atomic objectives and
    precise in_scope instructions for the rewriter. No hunk parsing.

    Returns dict with keys:
        pr_series:           list of PR specs with precise in_scope instructions
        scope_creep:         changes to drop/revert
        objectives_coverage: confirmation all objectives are covered
        summary:             human-readable summary
    """
    from pipeline.llm import llm_call, make_client, parse_json

    slug = repo.replace("/", "_", 1)
    repo_config = _load_repo_config(slug)
    pr_prep = repo_config.get("pr_preparation", {})
    pr_prep_section = yaml.dump(pr_prep, default_flow_style=False) if pr_prep else "(no pr_preparation found)"
    contribution_rules = repo_config.get("contribution_rules", {})
    contribution_rules_section = (
        yaml.dump(contribution_rules, default_flow_style=False)
        if contribution_rules
        else "(no contribution_rules found in repo config)"
    )

    objectives_section = (
        "\n".join(f"- {o}" for o in objectives)
        if objectives
        else "(not explicitly stated — infer from the diff, but be conservative)"
    )

    excluded_section = (
        "\n".join(f"- {e}" for e in excluded_changes)
        if excluded_changes
        else "(none flagged — apply extra caution to any deletions in the diff)"
    )

    already_satisfied_section = (
        "\n".join(f"- {s}" for s in already_satisfied)
        if already_satisfied
        else "(none)"
    )

    if isinstance(upstream_patterns, str):
        upstream_patterns = [upstream_patterns] if upstream_patterns.strip() else []
    upstream_patterns_section = (
        "\n".join(f"- {p}" for p in upstream_patterns)
        if upstream_patterns
        else "(no upstream patterns identified)"
    )

    judge_section = "(no judge findings provided)"
    if judge_findings and isinstance(judge_findings, dict):
        violations = judge_findings.get("violations", [])
        jsum = judge_findings.get("summary", "")
        if violations or jsum:
            parts = []
            if jsum:
                parts.append(f"Summary: {jsum}")
            for v in violations[:20]:
                parts.append(f"- [{v.get('severity','?')}] {v.get('file','')}: {v.get('message','')}")
            judge_section = "\n".join(parts)

    if layer_audit_warnings:
        layer_warning_section = "\n".join(f"- {w}" for w in layer_audit_warnings)
        judge_section = (
            judge_section
            + "\n\nARCHITECTURAL LAYER WARNINGS (from pre-planning layer analysis):\n"
            + layer_warning_section
            + "\nObjectives marked above as 'Demoted to excluded' have been moved to"
            " excluded_changes because the compiler-pass layer is sufficient — do NOT"
            " include model-layer wiring for those objectives in any PR's in_scope."
        )

    # Build downstream consumer list from repo_config
    downstream_consumers = repo_config.get("supplemental_knowledge", {}).get("downstream_consumers", [])
    if not downstream_consumers:
        # Fall back to well-known defaults for common repos
        downstream_consumers = repo_config.get("downstream_consumers", [])
    if downstream_consumers:
        downstream_lines = [
            f"- {c}" if isinstance(c, str) else f"- {c.get('repo', str(c))} ({c.get('note', '')})"
            for c in downstream_consumers
        ]
        downstream_section = (
            "These repos import directly from this repo. Any public symbol removed from this "
            "repo will break them silently:\n" + "\n".join(downstream_lines)
        )
    else:
        downstream_section = (
            "(no downstream consumers listed in repo config — assume any public function "
            "parameter or exported name may be used by external callers you cannot see)"
        )

    notes_section = notes.strip() if notes else "(none)"

    if patch_upstream_groups:
        _pug_lines = [
            f"  {ups}: {', '.join(patches)}"
            for ups, patches in patch_upstream_groups.items()
        ]
        patch_upstream_section = (
            "This seed contains patches for multiple upstream repos. "
            "Each PR must have its `upstream` field set to the correct repo:\n"
            + "\n".join(_pug_lines)
        )
    else:
        patch_upstream_section = f"Single-upstream seed — all PRs target: {repo}"

    prompt = _PLAN_PROMPT.format(
        repo=repo,
        objectives_section=objectives_section,
        upstream_patterns_section=upstream_patterns_section,
        excluded_section=excluded_section,
        already_satisfied_section=already_satisfied_section,
        file_summary=_file_summary(diff),
        diff_truncated=diff[:_DIFF_CAP],
        diff_cap=_DIFF_CAP,
        judge_section=judge_section,
        pr_prep_section=pr_prep_section,
        contribution_rules_section=contribution_rules_section,
        downstream_section=downstream_section,
        patch_upstream_section=patch_upstream_section,
        notes_section=notes_section,
    )

    client = make_client()
    raw = llm_call(prompt, model, client=client, max_tokens=16384, json_mode=True)
    result = parse_json(raw)

    if not isinstance(result, dict):
        raise ValueError(f"plan_prs LLM returned non-dict: {type(result)}")

    return _postprocess_plan(result, repo, objectives, patch_upstream_groups, already_satisfied=already_satisfied)


def _postprocess_plan(
    result: dict,
    repo: str,
    objectives: list | None,
    patch_upstream_groups: dict | None,
    already_satisfied: list | None = None,
) -> dict:
    """Apply setdefaults, objective-dropped milestones, and double-assignment checks."""
    result.setdefault("pr_series", [])
    result.setdefault("scope_creep", [])
    result.setdefault("objectives_coverage", "")
    result.setdefault("summary", "")

    # Detect objectives that survived verification but didn't make it into any
    # planned PR (planner LLM dropped them silently). Emit per-objective milestones so
    # the developer can see the loss instead of guessing.
    #
    # Uses an LLM batched call to determine coverage rather than token-overlap so that
    # shared vocabulary across repos (e.g. aiter/sglang both say "fused_moe") doesn't
    # produce false positives.  Falls back to token-overlap on LLM error.
    #
    # Also enforces the upstream-tag invariant: if an objective is tagged [owner/repo]
    # then it is only "covered" if a PR with `upstream` = owner/repo includes it.
    if objectives:
        # Already-satisfied objectives were hard-banned from the planner — they will
        # never appear in any PR and should not be flagged as "silently dropped".
        _already_satisfied_set: set[str] = set(
            (s if isinstance(s, str) else str(s)) for s in (already_satisfied or [])
        )
        _skip_indices: set[int] = {
            i for i, o in enumerate(objectives)
            if (o if isinstance(o, str) else str(o)) in _already_satisfied_set
        }

        _pr_summaries = []
        for pr in result["pr_series"]:
            _pr_summaries.append(
                f"PR {pr.get('index',0)} (upstream={pr.get('upstream',repo)}): "
                f"title={pr.get('title','')!r}; "
                f"objective={pr.get('objective','')!r}; "
                f"in_scope={pr.get('in_scope',[])!r}"
            )
        _planned_text_raw = "\n".join(_pr_summaries)

        _uncovered_from_llm: set[int] = set()
        try:
            from pipeline.llm import llm_call, make_client
            _checkable_objs = [(i, o) for i, o in enumerate(objectives) if i not in _skip_indices]
            _cov_prompt = (
                "You are checking whether each stated objective is addressed by at least one planned PR.\n\n"
                "PLANNED PRs:\n" + _planned_text_raw + "\n\n"
                "STATED OBJECTIVES:\n"
                + "\n".join(f"  {i}. {o}" for i, o in _checkable_objs) + "\n\n"
                "For each objective index, output 'covered' or 'not_covered'. "
                "An objective tagged [owner/repo] is only covered by a PR whose upstream field = owner/repo.\n"
                "Respond ONLY with a JSON object: {\"coverage\": {\"0\": \"covered\", \"1\": \"not_covered\", ...}}"
            )
            _client = make_client()
            _raw = llm_call(_cov_prompt, "claude-haiku-4-5-20251001", client=_client, max_tokens=1024, json_mode=True, temperature=0)
            from pipeline.llm import parse_json
            _cov_result = parse_json(_raw)
            _cov_map: dict = _cov_result.get("coverage", {})
            for _i in range(len(objectives)):
                if _cov_map.get(str(_i), "covered").strip().lower() == "not_covered":
                    _uncovered_from_llm.add(_i)
        except Exception as _cov_exc:
            logger.debug("LLM coverage check failed (%s), falling back to token-overlap", _cov_exc)
            # Fallback: token-overlap (original logic)
            _planned_text = " ".join(
                (pr.get("objective", "") or "")
                + " " + (pr.get("serves_objective", "") or "")
                + " " + " ".join(pr.get("in_scope") or [])
                for pr in result["pr_series"]
            ).lower()
            for _i, _obj in enumerate(objectives):
                if _i in _skip_indices:
                    continue
                _obj_str = _obj if isinstance(_obj, str) else str(_obj)
                _tokens = [t.lower() for t in _obj_str.replace(",", " ").split() if len(t) >= 5]
                _covered = any(t in _planned_text for t in _tokens) if _tokens else True
                if not _covered:
                    _uncovered_from_llm.add(_i)

        # Upstream-tag invariant: [owner/repo]-tagged objective must have a PR with matching upstream.
        _upstream_tagged_uncovered: set[int] = set()
        import re as _re
        for _i, _obj in enumerate(objectives):
            if _i in _skip_indices:
                continue
            _obj_str = _obj if isinstance(_obj, str) else str(_obj)
            _tag_m = _re.match(r"^\[([^\]]+/[^\]]+)\]", _obj_str.strip())
            if _tag_m:
                _tagged_repo = _tag_m.group(1)
                _has_matching_pr = any(
                    pr.get("upstream") == _tagged_repo for pr in result["pr_series"]
                )
                if not _has_matching_pr:
                    _upstream_tagged_uncovered.add(_i)
                    logger.info(
                        "planner_objective_dropped (upstream_tag_invariant): no PR with upstream=%s for: %s",
                        _tagged_repo, _obj_str[:120],
                    )

        for _idx, _obj in enumerate(objectives):
            if _idx in _skip_indices:
                continue
            _obj_str = _obj if isinstance(_obj, str) else str(_obj)
            if _idx in _uncovered_from_llm or _idx in _upstream_tagged_uncovered:
                _reason = (
                    f"no PR with upstream={_tagged_repo} planned"
                    if _idx in _upstream_tagged_uncovered
                    else "objective survived verification but no planned PR references it"
                )
                try:
                    from mcp_server import _emit_milestone
                    _emit_milestone("planner_objective_dropped", {
                        "objective_text": _obj_str,
                        "original_index": _idx,
                        "dropped_at_stage": "planner_filter",
                        "reason": _reason,
                        "upstream": repo,
                    })
                except Exception:
                    pass
                logger.warning(
                    "planner_objective_dropped (planner_filter): %s",
                    _obj_str[:200],
                )

    for i, pr in enumerate(result["pr_series"], 1):
        pr.setdefault("index", i)
        pr.setdefault("label", "unknown")
        pr.setdefault("title", f"PR {i}")
        pr.setdefault("commit_message", "")
        pr.setdefault("objective", "")
        pr.setdefault("serves_objective", "")
        pr.setdefault("upstream", repo)
        pr.setdefault("new_files", [])
        pr.setdefault("in_scope", [])
        pr.setdefault("out_of_scope", [])
        pr.setdefault("affected_files", [])
        pr.setdefault("stack_on_pr_index", None)
        pr.setdefault("rationale", "")
        pr.setdefault("cross_reference_note", "")
        # RLM may return list fields as [{"path": "..."}, ...] or [{"text": "..."}, ...].
        # Normalize all list-of-str fields to plain strings.
        def _coerce_str(v: object) -> str:
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return v.get("path") or v.get("text") or v.get("description") or str(v)
            return str(v)
        pr["affected_files"] = [_coerce_str(v) for v in pr["affected_files"]]
        # new_files is a list of dicts {path, intent, justification} per schema — do not coerce.
        # If the RLM returned plain strings, normalize to dicts so downstream .get("path") works.
        if isinstance(pr["new_files"], list):
            pr["new_files"] = [
                v if isinstance(v, dict) else {"path": str(v), "intent": "unknown", "justification": ""}
                for v in pr["new_files"]
            ]
        pr["in_scope"] = [_coerce_str(v) for v in pr["in_scope"]]
        pr["out_of_scope"] = [_coerce_str(v) for v in pr["out_of_scope"]]

    for c in result["scope_creep"]:
        c.setdefault("description", "")
        c.setdefault("recommendation", "drop")
        c.setdefault("reason", "")

    # Double-assignment consistency check: no patch file should appear in more than one PR.
    # Each warning is tagged with severity="info" when PR b stacks on PR a (intentional
    # stacked refactor) and severity="warning" otherwise (real planner bug).
    if patch_upstream_groups:
        _pr_by_index: dict[int, dict] = {_pr["index"]: _pr for _pr in result["pr_series"]}

        def _ancestor_chain(_idx: int) -> set[int]:
            """Return the set of PR indices that PR _idx transitively stacks on."""
            _chain: set[int] = set()
            _cur = _pr_by_index.get(_idx)
            _safety = 0
            while _cur and _safety < 64:
                _parent = _cur.get("stack_on_pr_index")
                if not isinstance(_parent, int) or _parent in _chain:
                    break
                _chain.add(_parent)
                _cur = _pr_by_index.get(_parent)
                _safety += 1
            return _chain

        _seen_files: dict[str, int] = {}
        _double_assigned: list[dict] = []
        for _pr in result["pr_series"]:
            for _fp in _pr.get("affected_files", []):
                if _fp in _seen_files:
                    _pr_a = _seen_files[_fp]
                    _pr_b = _pr["index"]
                    _is_stacked = _pr_a in _ancestor_chain(_pr_b)
                    _double_assigned.append({
                        "file": _fp,
                        "pr_a": _pr_a,
                        "pr_b": _pr_b,
                        "message": f"{_fp} assigned to both PR {_pr_a} and PR {_pr_b}",
                        "severity": "info" if _is_stacked else "warning",
                        "reason": "intentional_stacked_refactor" if _is_stacked else "unrelated_double_assignment",
                    })
                else:
                    _seen_files[_fp] = _pr["index"]
        if _double_assigned:
            _warn_items = [w for w in _double_assigned if w["severity"] == "warning"]
            if _warn_items:
                logger.warning(
                    "plan_prs double-assignment detected: %s",
                    "; ".join(w["message"] for w in _warn_items),
                )
            _info_items = [w for w in _double_assigned if w["severity"] == "info"]
            if _info_items:
                logger.info(
                    "plan_prs intentional stacked double-assignment: %s",
                    "; ".join(w["message"] for w in _info_items),
                )
            result["double_assignment_warnings"] = _double_assigned

    # Observability: flag suspiciously under-divided plans (1 PR, many objectives with bugfix verbs).
    _bugfix_verbs = ("fix", "correct", "handle", "resolve", "patch", "revert")
    _n_prs = len(result["pr_series"])
    _n_objs = len(objectives) if objectives else 0
    if _n_prs == 1 and _n_objs >= 5 and objectives:
        _bugfix_obj_count = sum(
            1 for o in objectives
            if any(v in (o if isinstance(o, str) else str(o)).lower() for v in _bugfix_verbs)
        )
        if _bugfix_obj_count >= 3:
            try:
                from mcp_server import _emit_milestone
                _emit_milestone("planner_underdivided", {
                    "n_prs": _n_prs,
                    "n_objectives": _n_objs,
                    "n_bugfix_objectives": _bugfix_obj_count,
                    "upstream": repo,
                    "warning": f"plan has 1 PR for {_n_objs} objectives ({_bugfix_obj_count} bugfix-verb) — likely under-divided",
                })
            except Exception:
                pass
            logger.warning(
                "planner_underdivided: 1 PR for %d objectives (%d with bugfix verbs) — check PR BOUNDARIES rule",
                _n_objs, _bugfix_obj_count,
            )

    return result


def plan_prs_rlm(
    repo: str,
    diff: str,
    *,
    objectives: list[str] | None = None,
    excluded_changes: list[str] | None = None,
    already_satisfied: list[str] | None = None,
    upstream_patterns: list[str] | None = None,
    judge_findings: dict | None = None,
    layer_audit_warnings: list[str] | None = None,
    notes: str = "",
    model: str = "claude-opus-4-7",
    patch_upstream_groups: dict[str, list[str]] | None = None,
    token: str = "",
    max_iters: int = 30,
) -> dict:
    """Plan a PR series using a DSPy RLM that verifies upstream paths before committing them.

    Unlike plan_prs(), the RLM can fetch upstream directory listings and file content to
    confirm that every affected_files entry and every symbol referenced in in_scope
    instructions actually exists at the expected path.  This eliminates the deepseek-class
    failure (wrong affected_files path) and the glm5-class failure (wrong symbol reference).

    No fallback to plan_prs() — errors propagate to the caller unchanged.
    """
    import dspy
    from pipeline.llm import parse_json

    # -------------------------------------------------------------------------
    # Build _patch_index: {filename → full hunk text} from the unified diff
    # -------------------------------------------------------------------------
    _patch_index: dict[str, str] = {}
    _current_file: str | None = None
    _current_lines: list[str] = []
    for _line in diff.splitlines(keepends=True):
        if _line.startswith("+++ b/"):
            if _current_file and _current_lines:
                _patch_index[_current_file] = "".join(_current_lines)
            _current_file = _line[6:].strip()
            _current_lines = []
        elif _current_file is not None:
            _current_lines.append(_line)
    if _current_file and _current_lines:
        _patch_index[_current_file] = "".join(_current_lines)

    # -------------------------------------------------------------------------
    # Shared GitHub helper (auth + JSON decode)
    # -------------------------------------------------------------------------
    import httpx as _httpx
    import base64 as _b64

    _headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    _file_cache: dict[str, str] = {}
    _dir_cache: dict[str, str] = {}

    def _gh_get(url: str):
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("GitHub API %s: %s", url, exc)
            return None

    def _gh_get_with_status(url: str) -> tuple[object | None, int | None]:
        """Like _gh_get but exposes the HTTP status (or None for transport error)
        so callers can distinguish 404 vs 403/429 vs network blip."""
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            status = r.status_code
            if status >= 400:
                logger.debug("GitHub API %s -> %s", url, status)
                return (None, status)
            return (r.json(), status)
        except Exception as exc:
            logger.debug("GitHub API transport error %s: %s", url, exc)
            return (None, None)

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    def fetch_upstream_file(repo_name: str, path: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Fetch a file or directory listing from a GitHub upstream repo.

        Pass a file path to read file content (paginated — call with increasing
        start_line to page through large files; first call shows total line count).
        Pass a directory path to get a JSON listing of its contents (name, type, path).
        Use this to explore the repo tree and locate where files actually live.

        Args:
            repo_name: GitHub repo in owner/name form (e.g. "ROCm/aiter")
            path: file or directory path relative to repo root (e.g. "aiter/ops/triton" or "")
            start_line: for file reads, 0-indexed line to start from (default 0)
            num_lines: for file reads, number of lines to return (default 200)
        Returns file content slice with [Lines X–Y of Z] header, directory JSON listing,
        or an error string."""
        import json as _json
        url = f"https://api.github.com/repos/{repo_name}/contents/{path}"
        cache_key = f"{repo_name}:{path}"
        if cache_key in _file_cache:
            raw_text = _file_cache[cache_key]
        elif cache_key in _dir_cache:
            return _dir_cache[cache_key]
        else:
            data, status = _gh_get_with_status(url)
            # Retry ONCE on 404 / transient transport error to dodge GitHub Contents API flakes.
            # Do NOT retry on 403/429 (rate limit) — surface that distinctly instead.
            if (status == 404 or status is None) and not isinstance(data, (dict, list)):
                import time as _time
                _time.sleep(1.0)
                data, status = _gh_get_with_status(url)
            if status in (403, 429):
                return f"(fetch rate-limited for {repo_name}/{path} — treat as unknown, do not conclude absent)"
            if data is None:
                logger.info("fetch_upstream_file 404: %s/%s — confirmed absent on retry (url=%s)", repo_name, path, url)
                return f"(file not found at {repo_name}/{path} on default branch; verified via {url})"
            if isinstance(data, list):
                # Directory listing
                entries = [{"name": e["name"], "type": e["type"], "path": e["path"]} for e in data]
                listing = _json.dumps(entries, indent=2)
                _dir_cache[cache_key] = listing
                return listing
            if isinstance(data, dict) and data.get("encoding") == "base64":
                try:
                    raw_text = _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    _file_cache[cache_key] = raw_text
                except Exception as exc:
                    return f"(decode error: {exc})"
            else:
                return f"(unexpected response for {repo_name}/{path})"

        lines = raw_text.splitlines()
        total = len(lines)
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(chunk)
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def read_diff_section(file_path: str) -> str:
        """Return the diff hunks for a specific file from the seed patch.

        Use this to read what the seed actually changes for a given file path
        without needing the entire diff in context.

        Args:
            file_path: path as it appears in the seed diff (after '+++ b/')
        Returns all hunk lines (@@, context, +, -) for that file, or a not-found message."""
        text = _patch_index.get(file_path)
        if text is None:
            available = list(_patch_index.keys())[:10]
            return f"(not found: {file_path!r}. Available: {available})"
        return text[:8000]  # cap at 8k chars per file section

    # -------------------------------------------------------------------------
    # Build plan_context (everything from _PLAN_PROMPT except the diff body)
    # -------------------------------------------------------------------------
    slug = repo.replace("/", "_", 1)
    repo_config = _load_repo_config(slug)
    pr_prep = repo_config.get("pr_preparation", {})
    pr_prep_section = yaml.dump(pr_prep, default_flow_style=False) if pr_prep else "(no pr_preparation found)"
    contribution_rules = repo_config.get("contribution_rules", {})
    contribution_rules_section = (
        yaml.dump(contribution_rules, default_flow_style=False)
        if contribution_rules
        else "(no contribution_rules found in repo config)"
    )

    objectives_section = (
        "\n".join(f"- {o}" for o in objectives)
        if objectives
        else "(not explicitly stated — infer from the diff, but be conservative)"
    )
    excluded_section = (
        "\n".join(f"- {e}" for e in excluded_changes)
        if excluded_changes
        else "(none flagged)"
    )
    already_satisfied_section = (
        "\n".join(f"- {s}" for s in already_satisfied)
        if already_satisfied
        else "(none)"
    )
    if isinstance(upstream_patterns, str):
        upstream_patterns = [upstream_patterns] if upstream_patterns.strip() else []
    upstream_patterns_section = (
        "\n".join(f"- {p}" for p in upstream_patterns)
        if upstream_patterns
        else "(no upstream patterns identified)"
    )

    judge_section = "(no judge findings provided)"
    if judge_findings and isinstance(judge_findings, dict):
        violations = judge_findings.get("violations", [])
        jsum = judge_findings.get("summary", "")
        if violations or jsum:
            parts = []
            if jsum:
                parts.append(f"Summary: {jsum}")
            for v in violations[:20]:
                parts.append(f"- [{v.get('severity','?')}] {v.get('file','')}: {v.get('message','')}")
            judge_section = "\n".join(parts)
    if layer_audit_warnings:
        judge_section = (
            judge_section
            + "\n\nARCHITECTURAL LAYER WARNINGS:\n"
            + "\n".join(f"- {w}" for w in layer_audit_warnings)
        )

    downstream_consumers = repo_config.get("supplemental_knowledge", {}).get("downstream_consumers", [])
    if not downstream_consumers:
        downstream_consumers = repo_config.get("downstream_consumers", [])
    if downstream_consumers:
        downstream_lines = [
            f"- {c}" if isinstance(c, str) else f"- {c.get('repo', str(c))} ({c.get('note', '')})"
            for c in downstream_consumers
        ]
        downstream_section = (
            "These repos import directly from this repo — any public symbol removed will break them:\n"
            + "\n".join(downstream_lines)
        )
    else:
        downstream_section = "(no downstream consumers listed)"

    notes_section = notes.strip() if notes else "(none)"

    if patch_upstream_groups:
        _pug_lines = [f"  {ups}: {', '.join(patches)}" for ups, patches in patch_upstream_groups.items()]
        patch_upstream_section = (
            "This seed contains patches for multiple upstream repos. "
            "Each PR must have its `upstream` field set to the correct repo:\n"
            + "\n".join(_pug_lines)
        )
    else:
        patch_upstream_section = f"Single-upstream seed — all PRs target: {repo}"

    diff_files = list(_patch_index.keys())
    file_summary_text = _file_summary(diff)

    plan_context = (
        f"REPOSITORY: {repo}\n\n"
        f"STATED OBJECTIVES:\n{objectives_section}\n\n"
        f"UPSTREAM PATTERNS:\n{upstream_patterns_section}\n\n"
        f"ALREADY SATISFIED (hard-banned from all PRs):\n{already_satisfied_section}\n\n"
        f"EXCLUDED CHANGES (must be dropped):\n{excluded_section}\n\n"
        f"PATCH FILES IN SEED DIFF (call read_diff_section on any of these):\n"
        + "\n".join(f"  - {f}" for f in diff_files) + "\n\n"
        f"PATCH SUMMARY:\n{file_summary_text}\n\n"
        f"JUDGE FINDINGS:\n{judge_section}\n\n"
        f"CONTRIBUTING GUIDANCE:\n{pr_prep_section}\n\n"
        f"CONTRIBUTION RULES:\n{contribution_rules_section}\n\n"
        f"DOWNSTREAM CONSUMERS:\n{downstream_section}\n\n"
        f"PATCH UPSTREAM GROUPS:\n{patch_upstream_section}\n\n"
        f"DEVELOPER NOTES:\n{notes_section}\n\n"
        + _PLAN_PROMPT[_PLAN_PROMPT.index("## PHASE 0"):].replace("{repo}", repo)
    )

    # -------------------------------------------------------------------------
    # DSPy RLM
    # -------------------------------------------------------------------------
    import os
    gateway = os.environ.get("LITELLM_GATEWAY_URL", "http://localhost:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    _lm = dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
        temperature=0.2,
    )

    class PlanSignature(dspy.Signature):
        """Your primary job is to RELIEVE PRESSURE ON THE DOWNSTREAM REWRITER by
        pre-computing exact (file, upstream_repo) pairs that are confirmed to exist —
        so the rewriter starts from verified paths and never burns iterations searching
        for where to make changes.

        CORE TASK — BUILD A CONFIRMED FILE-UPSTREAM MAP:

        You have a seed diff that touches files across one or more upstream repos. Before
        writing any PR, build a mapping: {seed_file_path → (confirmed_upstream_repo,
        confirmed_upstream_path)}. To do this:

        - Use read_diff_section to read what the seed changes for each file.
        - Use fetch_upstream_file to explore each candidate repo's directory tree and
          locate where the file actually lives. Pass a directory path to get a listing;
          pass a file path to read its content. Navigate iteratively — start at the
          repo root or the likely subdirectory, then drill down.
        - If the seed path doesn't exist in the obvious repo, the file may have been
          renamed or restructured upstream. Write Python in your REPL to search through
          fetched directory listings for the best match, or fetch sibling directories
          to find an equivalent file that serves the same purpose.
        - A path that belongs to one repo must not appear in another repo's PR.
          (e.g. aiter/ops/... → ROCm/aiter; python/sglang/srt/... → sgl-project/sglang
          regardless of what it imports.)

        PATH-NOT-FOUND ESCALATION — when fetch_upstream_file returns "not found" for a
        seed file path, do NOT immediately drop the objective. Instead:
          1. Fetch the parent directory (one level up) to get a listing of what exists.
          2. Search siblings and sub-directories for a file serving the same purpose
             (same base name, similar naming convention, same file type).
          3. If you find an equivalent: use that confirmed path in affected_files or
             new_files and adapt in_scope instructions to the upstream's actual path.
          4. If the file is genuinely new (no equivalent exists upstream): confirm the
             right parent directory exists, choose a path that fits the upstream's naming
             convention, and place the file in new_files with that confirmed parent path.
          5. Only output objective_needs_human_review if after fetching the parent dir
             AND at least two sibling dirs you still cannot find a reasonable mapping.

        Only output a path in affected_files or new_files after fetch_upstream_file
        has confirmed the parent directory exists. Never guess a path without first
        fetching its parent.

        For in_scope instructions: fetch the actual file content to see what symbols,
        base classes, and patterns are present before writing descriptions. If a symbol
        referenced in the seed is renamed upstream, use the upstream name.

        Every PR's `upstream` field must match the repo where its files were confirmed.
        An objective prefixed [owner/repo] must land in a PR with `upstream` = owner/repo.

        PR BOUNDARIES — each PR must address exactly one coherent, independently
        reviewable change. Distinct bug fixes MUST be SEPARATE PRs even when they
        touch the same file. Do NOT merge multiple unrelated fixes into one PR.

        One-shot example (WRONG → RIGHT):
          WRONG: PR1 {title: "Fix dispatch and add kernel", affected_files: ["moe.py"],
                       in_scope: ["fix dispatch bug", "add FP8 kernel"]}
          RIGHT: PR1 {title: "Fix dispatch bug", affected_files: ["moe.py"],
                       in_scope: ["fix dispatch bug"]}
                 PR2 {title: "Add FP8 kernel",   affected_files: ["moe.py"],
                       in_scope: ["add FP8 kernel"]}

        If you have N objectives that each contain bugfix verbs (fix, correct, handle,
        resolve, patch, revert) targeting different functions or behaviors, produce at
        least N separate PRs — do not consolidate them.

        SCOPE CREEP DISCIPLINE — enforce these before finalizing any PR:

        - Read plan_context sections ALREADY SATISFIED and EXCLUDED CHANGES carefully.
          Any change implementing a banned item must go to `scope_creep` with recommendation
          "drop", regardless of whether you think it's useful. No exceptions.
        - Any diff hunk that doesn't map to a stated objective is incidental and must go
          to `scope_creep` with recommendation "drop". Do not promote incidental changes
          to new objectives.
        - DOWNSTREAM SAFETY (PHASE 1): Any removal of public API surface (function
          parameters, exported names, callable signatures) must be in `scope_creep` with
          "drop" unless a stated objective explicitly requires it. When in doubt, keep the
          existing code and add alongside it.
        - If you are unsure whether a change belongs to any objective, put it in
          `scope_creep`. The rewriter will ignore anything in `scope_creep`.

        BUDGET SKILL — maintain throughout your session:
          At the start of your REPL session run:
              budget_used = 0; budget_cap = 100
          Before every llm_query() call check:
              if budget_used >= budget_cap * 0.8:
                  # Low budget — reason from already-fetched text instead of querying.
                  pass
          After every llm_query() call: budget_used += 1

        Apply the PHASE 0 / PHASE 1 / PHASE 2 / PHASE 3 planning rules from plan_context.
        Return ONLY a JSON object matching the schema in plan_context."""

        plan_context: str = dspy.InputField(
            desc="Repo rules, objectives, patch file list, contribution guidelines, and planning instructions"
        )
        upstream_repo: str = dspy.InputField(desc="Primary upstream repo (owner/name)")
        result: str = dspy.OutputField(
            desc='Validated JSON plan: {"pr_series": [...], "scope_creep": [...], "objectives_coverage": "...", "summary": "..."}'
        )

    rlm = dspy.RLM(
        PlanSignature,
        tools=[fetch_upstream_file, read_diff_section],
        max_iterations=max_iters,
        max_llm_calls=100,
    )

    try:
        with dspy.context(lm=_lm):
            prediction = rlm(plan_context=plan_context, upstream_repo=repo)
        raw = prediction.result
        result = parse_json(raw)
        if not isinstance(result, dict):
            raise ValueError(f"non-dict result: {type(result)}")
        logger.info(
            "RLM plan_prs: %d PRs, %d scope_creep (fetched %d upstream files)",
            len(result.get("pr_series", [])),
            len(result.get("scope_creep", [])),
            len(_file_cache) + len(_dir_cache),
        )
        return _postprocess_plan(result, repo, objectives, patch_upstream_groups, already_satisfied=already_satisfied)
    except Exception as exc:
        raise RuntimeError(f"RLM plan_prs failed: {exc}") from exc


def format_plan(plan: dict, repo: str) -> str:
    """Render a PR plan as human-readable text."""
    lines = [
        "=" * 70,
        f"PR PLAN  —  {repo}",
        "=" * 70,
        "",
        plan.get("summary", ""),
        "",
    ]

    if plan.get("objectives_coverage"):
        lines += [f"Objectives: {plan['objectives_coverage']}", ""]

    _LABEL_ICON = {
        "bugfix": "🐛", "perf": "⚡", "tuning": "📊",
        "refactor": "🔧", "docs": "📝", "new-feature": "✨", "unknown": "❓",
    }

    for pr in plan.get("pr_series", []):
        idx = pr["index"]
        icon = _LABEL_ICON.get(pr["label"], "❓")
        stack = f"  [stacks on PR {pr['stack_on_pr_index']}]" if pr["stack_on_pr_index"] else "  [targets main]"
        files = ", ".join(pr.get("affected_files", [])) or "?"
        lines.append(f"PR {idx}: {icon} [{pr['label'].upper()}]  {pr['title']}{stack}")
        lines.append(f"  Files:     {files}")
        lines.append(f"  Objective: {pr['objective']}")
        if pr.get("serves_objective"):
            lines.append(f"  Serves:    {pr['serves_objective']}")
        lines.append(f"  In scope (rewriter instructions):")
        for s in pr["in_scope"]:
            lines.append(f"    + {s}")
        if pr["out_of_scope"]:
            lines.append(f"  Out of scope:")
            for s in pr["out_of_scope"]:
                lines.append(f"    - {s}")
        if pr["rationale"]:
            lines.append(f"  Rationale: {pr['rationale']}")
        lines.append("")

    creep = plan.get("scope_creep", [])
    if creep:
        drops = [c for c in creep if c.get("recommendation") == "drop"]
        reverts = [c for c in creep if c.get("recommendation") == "revert"]
        others = [c for c in creep if c.get("recommendation") not in ("drop", "revert")]
        if drops:
            lines.append(f"🗑  DROP ({len(drops)} groups — exclude from all PRs):")
            for c in drops:
                lines.append(f"  {c.get('description', '')}")
                if c.get("reason"):
                    lines.append(f"    → {c['reason']}")
            lines.append("")
        if reverts:
            lines.append(f"↩  REVERT ({len(reverts)} groups — must not be applied):")
            for c in reverts:
                lines.append(f"  {c.get('description', '')}")
                if c.get("reason"):
                    lines.append(f"    → {c['reason']}")
            lines.append("")
        if others:
            lines.append("⚠  REVIEW:")
            for c in others:
                lines.append(f"  [{c.get('recommendation','?')}] {c.get('description','')}")
                if c.get("reason"):
                    lines.append(f"    → {c['reason']}")
            lines.append("")

    if plan.get("layer_audit_warnings"):
        lines.append("🏗  LAYER WARNINGS (model-layer objectives demoted to excluded):")
        for w in plan["layer_audit_warnings"]:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
