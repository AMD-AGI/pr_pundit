"""
PR rewrite: generate a surgical per-PR file from base + patched reference.

Strategy — section-based rewrite:
  1. Extract the relevant sections of the BASE file (functions/classes named in
     in_scope) plus surrounding context lines, rather than the entire file.
  2. Extract the corresponding sections from the PATCHED file (same names).
  3. Ask the LLM to rewrite ONLY those sections, applying this PR's objective.
  4. Splice the rewritten sections back into the full base file.
  5. Diff base vs spliced result with difflib to produce a clean unified patch.

This keeps each LLM call manageable (section + context vs. whole file), avoids
max_tokens truncation, and produces output that is byte-identical to base except
where the objective was applied.

After each file rewrite, the patch is validated with judge_patch (same rules
engine used by the iterative-fix loop). Violations are fed back to the LLM
for correction — up to max_retries times. This mirrors the fix-it loop pattern.
"""

from __future__ import annotations

import ast
import difflib
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# Context lines to include above/below each extracted section
_SECTION_CONTEXT = 30

# Unresolved scope-creep violations from the last rewrite_file_for_pr call.
# Keyed by (file_path, pr_index) → list[str] of removed symbol names.
# rewrite_all_files_for_plan reads this after each call to surface violations
# as arch_constraints for the next composite-loop iteration.
_unresolved_scope_creep: dict[tuple[str, int], list[str]] = {}
# Max chars of the patched reference section shown per LLM call
_SECTION_CAP = 80_000
# Max lines in a section before escalating to whole-file rewrite
_SECTION_LINE_CAP = 500

_REWRITE_PROMPT = """\
You are an expert contributor to the GitHub repository "{repo}".

Your task: rewrite **only the sections shown below** from `{file_path}`, as they
should appear in **PR {pr_index} — {pr_title}**.

---

## THIS PR'S SINGLE OBJECTIVE

{objective}

## WHAT TO CHANGE (precise instructions):

{in_scope_block}

## WHAT MUST NOT CHANGE IN THIS PR:

{out_of_scope_block}

---
{critic_hints_block}{arch_constraints_block}## REPOSITORY CONTRIBUTION RULES

{contrib_rules}

---

## BASE SECTIONS (`{file_path}` on `main` — your authoritative starting point)

Lines {base_line_start}–{base_line_end} of the file (1-indexed):

```
{base_section}
```

Lines not shown are UNCHANGED from the base file — do not touch them.

---

## SEED DIFF HINT (what the seed author changed — for reference only)

The seed implemented this objective differently. Use this ONLY as a conceptual
hint. Do NOT copy it literally — implement the objective from scratch against
the BASE SECTIONS above, using idioms and patterns from the upstream repo.
If the hint is empty, ignore this section entirely.

```
{seed_diff_hint}
```

---

## YOUR TASK

Rewrite the base section above, applying ONLY the changes listed in "WHAT TO
CHANGE". Preserve all other lines exactly.

Rules:
1. Start from the BASE SECTIONS exactly.
2. Apply ONLY the changes that serve the single objective above.
3. Every line not touched by this PR must be identical to the base section.
4. Follow the repo's contribution rules above.
5. Do NOT include changes from the reference that are outside this objective.
6. Do NOT add comments explaining what you changed.
7. **CRITICAL — do not remove, rename, or restructure any function, class, or
   variable that is not explicitly named in "WHAT TO CHANGE".** If a function
   appears in the base section but is not in "WHAT TO CHANGE", it must appear
   unchanged in your output — even if the reference section omits it.
8. The reference may contain refactors unrelated to this PR's objective. Ignore
   all reference changes that are not described in "WHAT TO CHANGE".
9. **NEVER remove function parameters, even if they appear unused in this file.**
   Downstream callers (vLLM, SGLang, and others) may pass these parameters.
   Keep all existing function signatures exactly as in the base section.
10. **NEVER alter macro names, preprocessor symbols, or intrinsic names in C/CUDA
    files.** Names like `VLLM_LDG`, `VLLM_DISPATCH_FLOATING_TYPES`, `__syncthreads`,
    etc. are exact identifiers — any case change (e.g. `VLLm_LDG`) causes a
    compilation error. Copy them letter-for-letter from the base section.
11. **NEVER rename struct fields, config keys, dataclass attributes, or dict keys
    that appear in the base section**, unless that rename is explicitly listed in
    "WHAT TO CHANGE". Field names are public API — an unrequested rename is a
    silent breaking change across every caller. If you are unsure of the correct
    name, fetch the defining file with `fetch_upstream_file` to verify before writing.
12. **NEVER replace a function body with `raise NotImplementedError` or `pass`**
    unless the base section already has that stub. If a function is not in "WHAT TO
    CHANGE", copy its entire body verbatim from the base section.

Wrap your output in a single code fence:

```
<rewritten section here>
```

No explanation outside the fence.
"""

_FIX_PROMPT = """\
Your previous rewrite of the section from `{file_path}` for PR {pr_index}
("{pr_title}") has rule violations. Fix them and return the corrected section.

## VIOLATIONS TO FIX

{violations_block}

## CURRENT REWRITTEN SECTION (your previous output — fix the violations)

```
{current_content}
```

## REMINDER: THIS PR'S OBJECTIVE

{objective}

## IN SCOPE

{in_scope_block}

## HARD CONSTRAINTS (apply even when fixing violations)

- Do NOT rename any struct field, config key, dataclass attribute, or dict key
  that was in the original base — field names are public API.
- Do NOT remove any function parameter.
- Fix ONLY the listed violations; leave everything else exactly as-is.

Wrap your output in a single code fence:

```
<corrected section here>
```

No explanation outside the fence.
"""

_WHOLE_FILE_PROMPT = """\
You are an expert contributor to the GitHub repository "{repo}".

Your task: produce the exact content of `{file_path}` as it should appear in
**PR {pr_index} — {pr_title}**.

---

## THIS PR'S SINGLE OBJECTIVE

{objective}

## WHAT TO CHANGE (precise instructions):

{in_scope_block}

## WHAT MUST NOT CHANGE IN THIS PR:

{out_of_scope_block}

---

{critic_hints_block}{arch_constraints_block}## REPOSITORY CONTRIBUTION RULES

{contrib_rules}

---

## BASE FILE (`{file_path}` on `main` — your authoritative starting point)

```
{base_content}
```

---

## SEED DIFF HINT (what the seed author changed — for reference only)

The seed implemented this objective differently. Use this ONLY as a conceptual
hint. Do NOT copy it literally — implement the objective from scratch against
the BASE FILE above, using idioms and patterns from the upstream repo.
If the hint is empty, ignore this section entirely.

```
{seed_diff_hint}
```

---

## YOUR TASK

Produce the complete file content after applying ONLY the changes above.

Rules:
1. Start from BASE FILE exactly.
2. Apply ONLY the changes that serve the single objective.
3. Every line not touched by this PR must be identical to BASE FILE.
4. Follow the repo's contribution rules.
5. Do NOT include changes from REFERENCE outside this objective.
6. Do NOT add comments explaining what you changed.
7. **CRITICAL — do not remove, rename, or restructure any function, class, or
   variable that is not explicitly named in "WHAT TO CHANGE".** Every symbol in
   the base file that is not in "WHAT TO CHANGE" must appear unchanged.
8. The reference may contain refactors unrelated to this PR's objective. Ignore
   all reference changes not described in "WHAT TO CHANGE".
9. **NEVER remove function parameters, even if they appear unused.** Downstream
   callers (vLLM, SGLang, and others) may pass these parameters and will break
   silently if they are removed. Keep all existing function signatures intact.
10. **CRITICAL — if the REFERENCE FILE is missing something that BASE FILE has
    (a parameter, a type, a function, an import), and that removal is NOT listed
    in "WHAT TO CHANGE", the BASE FILE is authoritative — keep it.** The
    reference may have been built against an older tree; trust the base for
    everything not explicitly in scope.
11. **NEVER rename struct fields, config keys, dataclass attributes, or dict keys
    that appear in BASE FILE**, unless that rename is explicitly listed in "WHAT TO
    CHANGE". Field names are public API — an unrequested rename silently breaks every
    caller. If uncertain, fetch the file that defines the config class or struct to
    verify the exact field name before writing it.
12. **NEVER replace a function body with `raise NotImplementedError` or `pass`**
    unless the BASE FILE already has that stub. Functions not in "WHAT TO CHANGE"
    must be copied verbatim — body and all.

Return ONLY the complete file content. No explanation, no markdown fences.
"""


def _extract_seed_diff_hint(
    base_content: str,
    seed_content: str,
    in_scope_symbols: set[str],
) -> str:
    """Compute a filtered structural diff of seed vs upstream base.

    Returns only the unified-diff hunks whose added lines reference at least one
    in-scope symbol, capped at 4000 chars. Returns "" when there is no delta or
    no in-scope overlap — the caller omits the hint from the prompt entirely.
    """
    if not seed_content or seed_content == base_content:
        return ""

    base_lines = base_content.splitlines(keepends=True)
    seed_lines = seed_content.splitlines(keepends=True)
    raw_diff = list(difflib.unified_diff(base_lines, seed_lines, lineterm=""))
    if not raw_diff:
        return ""

    # Split into per-hunk groups and keep only those touching in-scope symbols.
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in raw_diff:
        if line.startswith("@@") and current:
            hunks.append(current)
            current = [line]
        elif line.startswith(("---", "+++")) and not current:
            current.append(line)
        else:
            current.append(line)
    if current:
        hunks.append(current)

    header = hunks[0] if hunks and hunks[0][0].startswith(("---", "+++")) else []
    body_hunks = hunks[1:] if header else hunks

    if not in_scope_symbols:
        # No symbols identified — return the full diff (capped) as-is
        kept = body_hunks
    else:
        kept = [
            h for h in body_hunks
            if any(
                sym in line
                for line in h
                if line.startswith("+")
                for sym in in_scope_symbols
            )
        ]

    if not kept:
        return ""

    result = "".join(header) + "".join("".join(h) for h in kept)
    return result[:4000]


def _build_contrib_rules(repo_config: dict) -> str:
    """Build a contribution rules block from repo_config.yaml — no hardcoded rules."""
    pr_prep = repo_config.get("pr_preparation", {})
    if not pr_prep:
        return "(no contributing rules configured)"

    parts = []

    if pr_prep.get("contributing_urls"):
        parts.append("Contributing guide: " + ", ".join(pr_prep["contributing_urls"]))

    if pr_prep.get("commit_message_format"):
        parts.append(f"Commit format: {pr_prep['commit_message_format']}")

    if pr_prep.get("lint_commands"):
        parts.append("Lint commands: " + "; ".join(pr_prep["lint_commands"]))

    if pr_prep.get("pr_checklist"):
        parts.append("PR checklist:\n" + "\n".join(f"  {c}" for c in pr_prep["pr_checklist"]))

    for rule in repo_config.get("architecture_rules", []):
        parts.append(f"Architecture rule: {rule}")

    return "\n".join(parts) if parts else "(no contributing rules configured)"


def _format_violations(findings: list[dict]) -> str:
    lines = []
    for f in findings:
        sev = f.get("severity", "?")
        file_ = f.get("file", "")
        msg = f.get("message", "")
        hint = f.get("fix_hint", "")
        line = f"[{sev}] {file_}: {msg}"
        if hint:
            line += f"\n  Fix: {hint}"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def _strip_fences(text: str) -> str:
    """Extract content from the first markdown code fence block found.

    If there are no fences, return the text as-is (whole-file rewrite
    that the LLM correctly returned without fences).
    """
    # Find the first opening fence and extract to its closing fence
    import re
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # No fence found — strip any trailing ``` if it exists
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip("\n")
    return text


def _is_additive_only(base_content: str, patched_content: str, file_path: str) -> bool:
    """Return True if patched_content only adds lines relative to base_content.

    Covers two cases:
    - New file: base is empty, patched has content.
    - Append-only: every line in base appears in patched in the same order
      (no edits or deletions, only additions — e.g. CSV row appends).

    In both cases the LLM adds no value; use patched_content directly.
    """
    if not patched_content or patched_content == base_content:
        return False
    if not base_content.strip():
        return True  # new file
    base_lines = base_content.splitlines()
    patched_lines = patched_content.splitlines()
    if len(patched_lines) < len(base_lines):
        return False  # patched has fewer lines — not purely additive
    # Walk both sequences: every base line must appear in order in patched
    bi = 0
    for pl in patched_lines:
        if bi < len(base_lines) and pl == base_lines[bi]:
            bi += 1
    return bi == len(base_lines)


def _data_store_sanity_check(rewritten: str, base_content: str, file_path: str, fallback: str | None = None) -> str:
    """Verify that the rewritten output of a structured data store file is sane.

    Applies format-specific checks to prevent LLM-introduced schema drift or
    truncation.  Currently handles CSV (most common); other formats (YAML, JSON,
    Python dicts) pass through — add format branches as needed.

    CSV checks (in order):
      1. Schema: column count must not change.  If same count but header text
         differs (casing/spacing), strip the spurious LLM header so rows still land
         correctly when appended to the original file.
      2. Row count must not decrease (truncation detection).
      3. Prose guard: first non-empty line must not start with a prose sentence.
    Falls back to `fallback` (default: base_content) on any hard failure.
    Pass patched_content as fallback when the objective is to append entries.

    Example — CSV:
      base_content  = canonical upstream CSV (623 rows + header)
      rewritten     = LLM output (should be base + new rows, possibly with a
                      re-generated header at the top)
      → strip spurious header if column count matches, then validate row count.
    """
    import re
    _fallback = fallback if fallback is not None else base_content

    # ── CSV ──────────────────────────────────────────────────────────────────
    if file_path.endswith(".csv"):
        def _hdr(t):
            for ln in t.splitlines():
                if ln.strip():
                    return [c.strip() for c in ln.split(",")]
            return []

        def _rows(t):
            return sum(1 for ln in t.splitlines() if ln.strip() and "," in ln)

        oh, nh = _hdr(base_content), _hdr(rewritten)

        if oh and len(oh) != len(nh):
            logger.warning(
                "  Data-store sanity (%s): column count changed %d → %d — falling back",
                file_path, len(oh), len(nh),
            )
            return _fallback

        if oh and nh and [c.lower() for c in nh] != [c.lower() for c in oh]:
            # Same column count but LLM re-generated the header with different
            # casing or spacing.  Strip it so new rows append cleanly to the
            # original file which already has its own header.
            lines = rewritten.splitlines()
            stripped = []
            removed = False
            for ln in lines:
                if not removed and ln.strip() and [c.strip().lower() for c in ln.split(",")] == [c.lower() for c in nh]:
                    removed = True
                    continue
                stripped.append(ln)
            rewritten = "\n".join(stripped)
            logger.info("  Data-store sanity (%s): stripped LLM-generated header", file_path)

        if _rows(base_content) > 0 and _rows(rewritten) < _rows(base_content):
            # If the output is much shorter than base AND looks like valid CSV data
            # (comma-delimited, no prose), treat it as append-only rows rather than
            # a truncated full-file rewrite.  This handles the case where DSPy
            # correctly returns only the new rows instead of the entire file.
            _r_lines = [ln for ln in rewritten.splitlines() if ln.strip()]
            _first_r = _r_lines[0] if _r_lines else ""
            _looks_like_data = "," in _first_r and not re.match(r'^[a-z][a-z]+ ', _first_r)
            if _looks_like_data:
                logger.info(
                    "  Data-store sanity (%s): output looks like append-only rows "
                    "(%d rows) — appending to base (%d rows)",
                    file_path, _rows(rewritten), _rows(base_content),
                )
                return base_content.rstrip("\n") + "\n" + rewritten.strip() + "\n"
            logger.warning(
                "  Data-store sanity (%s): row count dropped %d → %d — falling back",
                file_path, _rows(base_content), _rows(rewritten),
            )
            return _fallback

        first = next((ln for ln in rewritten.splitlines() if ln.strip()), "")
        if re.match(r'^[a-z][a-z]+ ', first):
            logger.warning("  Data-store sanity (%s): output looks like prose — falling back", file_path)
            return _fallback

    # ── Other structured formats (YAML, JSON, Python dicts) ──────────────────
    # Add format-specific checks here as needed.  For now, pass through and let
    # the diff / critic stages catch problems.

    return rewritten


# Keep the old name as an alias so existing call sites don't break.
_csv_sanity_check = _data_store_sanity_check


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _identify_symbols_llm(in_scope: list[str], top_level_defs: set[str]) -> set[str]:
    """Ask a model which top-level symbols the in_scope instructions directly touch."""
    from pipeline.llm import llm_call, make_client, parse_json
    defs_list = "\n".join(sorted(top_level_defs))
    instructions = "\n".join(f"- {s}" for s in in_scope)
    prompt = (
        "You are a precise code-analysis assistant.\n\n"
        "INSTRUCTIONS (what needs to change in this file):\n"
        f"{instructions}\n\n"
        "TOP-LEVEL SYMBOLS in the file (one per line):\n"
        f"{defs_list}\n\n"
        "TASK: Return ONLY the symbols that must be DIRECTLY MODIFIED (body changed, "
        "parameter added/removed, field added/removed) to implement the instructions.\n\n"
        "STRICT RULES:\n"
        "- Return the MINIMAL set. If only one class needs a new field, return ONLY that class.\n"
        "- Do NOT include symbols that merely call or import the changed symbol.\n"
        "- Do NOT include symbols that are mentioned as context or background.\n"
        "- Do NOT include parent classes unless they themselves need changes.\n"
        "- A symbol that only READS a new field does not need to be returned.\n\n"
        "Return ONLY a JSON object: {\"symbols\": [\"SymbolA\", \"SymbolB\"]}. "
        "If no symbols need modification, return {\"symbols\": []}."
    )
    try:
        raw = llm_call(prompt, "claude-sonnet-4-6", client=make_client(), max_tokens=1024)
        result = parse_json(raw)
        names = result.get("symbols", [])
        return {n for n in names if n in top_level_defs}
    except Exception as e:
        logger.warning("_identify_symbols_llm failed (%s: %s) — falling back to empty set", type(e).__name__, e)
        return set()


def _find_symbol_ranges(
    content: str,
    in_scope: list[str],
    *,
    context: int = _SECTION_CONTEXT,
) -> list[tuple[int, int]]:
    """Return (start, end) line ranges (0-indexed, end exclusive) covering
    all top-level symbols (functions/classes) mentioned in in_scope text.

    Falls back to returning empty list if no symbols are found,
    which triggers the whole-file rewrite path.

    context: lines of surrounding code to include on each side for LLM reading.
    Pass context=0 to get bare symbol ranges (for splicing without pre-context).
    """
    lines = content.splitlines()
    n = len(lines)

    # Build the set of all top-level def/class names in the file.
    top_level_defs: set[str] = set()
    for line in lines:
        m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if m:
            top_level_defs.add(m.group(2))

    if not top_level_defs or not in_scope:
        return []

    symbol_names = _identify_symbols_llm(in_scope, top_level_defs)

    if not symbol_names:
        return []

    # Find top-level def/class blocks that define any symbol.
    # Walk back from each def to include any preceding decorator lines so the
    # splice boundary never cuts inside a decorator expression.
    raw_ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        line = lines[i]
        m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if m and m.group(2) in symbol_names:
            # Walk back to include decorators (lines starting with @)
            sym_start = i
            while sym_start > 0 and lines[sym_start - 1].lstrip().startswith("@"):
                sym_start -= 1
            # Find end of block: next top-level def/class or EOF
            j = i + 1
            while j < n:
                if re.match(r"^(def|class)\s+", lines[j]):
                    break
                j += 1
            raw_ranges.append((sym_start, j))
            i = j
        else:
            i += 1

    # Also include module-level assignments that match symbol names
    for i, line in enumerate(lines):
        m = re.match(r"^([A-Z_][A-Z0-9_]+)\s*[:=]", line)
        if m and m.group(1) in symbol_names:
            # Include the constant + a few lines after it
            end = min(i + 5, n)
            raw_ranges.append((i, end))

    if not raw_ranges:
        return []

    # Merge overlapping/adjacent ranges and add context
    raw_ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in raw_ranges:
        s = max(0, start - context)
        e = min(n, end + context)
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _build_depth_map(content: str) -> list[int]:
    """Build bracket/paren/brace depth at the START of each line (0-indexed list).

    Uses Python's tokenize module so string literals, f-strings, and comments
    don't confuse the bracket counting. Falls back to a simple char-count scan
    if the file has tokenizer errors (e.g. incomplete/malformed Python).

    depth_map[i] = bracket depth immediately before line i starts.
    """
    import io
    import tokenize as _tokenize

    lines_raw = content.splitlines()
    n = len(lines_raw)
    depth_at = [0] * (n + 2)   # 1-indexed; [0] unused, [n+1] sentinel

    try:
        depth = 0
        last_recorded = 0  # last line (1-indexed) whose depth was recorded

        def _fill(up_to: int) -> None:
            nonlocal last_recorded
            for ln in range(last_recorded + 1, up_to + 1):
                if ln <= n:
                    depth_at[ln] = depth
            last_recorded = max(last_recorded, up_to)

        readline = io.StringIO(content).readline
        for tok_type, tok_string, (srow, _sc), (_er, _ec), _ in _tokenize.generate_tokens(readline):
            if tok_type in (_tokenize.NEWLINE, _tokenize.NL, _tokenize.COMMENT,
                            _tokenize.ENCODING, _tokenize.ENDMARKER):
                continue
            # Record depth at start of this line (before processing this token)
            _fill(srow - 1)          # fill lines before srow
            if srow <= n and last_recorded < srow:
                depth_at[srow] = depth
                last_recorded = srow
            if tok_type == _tokenize.OP:
                if tok_string in "([{":
                    depth += 1
                elif tok_string in ")]}":
                    depth = max(0, depth - 1)
        _fill(n)
    except Exception:
        # Fallback: naive char-count (inaccurate inside strings, but better than nothing)
        depth = 0
        for i, line in enumerate(lines_raw):
            depth_at[i + 1] = depth
            for ch in line:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)

    return depth_at   # 1-indexed: depth_at[i] = depth at start of line i


_CONTINUATION_RE = re.compile(r"^\s*(elif|else|except|finally|case)\b")
_DECORATOR_RE = re.compile(r"^\s*@")


def _safe_boundary(
    lines: list[str],
    idx: int,
    direction: int,
    depth_map: list[int] | None = None,
    max_shift: int = 40,
) -> int:
    """Shift idx in direction (+1/-1) until we land on a syntactically safe splice point.

    A safe point satisfies:
    1. Bracket/paren/brace depth at the start of line idx is 0
       (so we are not cutting inside a multiline expression).
    2. The line is not a bare continuation keyword (elif/else/except/finally/case)
       that requires a preceding block to be syntactically valid.
    3. When walking backward (direction=-1), not a decorator line (@...)
       because decorators must stay attached to the def/class they precede.

    depth_map: pre-built result of _build_depth_map() for O(1) lookup per line.
                If None, a simple naive O(n) scan is used.

    Returns the adjusted index (clamped to [0, len(lines)]).
    """
    n = len(lines)

    def _depth(i: int) -> int:
        if depth_map is not None:
            # depth_map is 1-indexed; line i (0-indexed) → depth_map[i+1]
            dm_idx = i + 1
            if 0 <= dm_idx < len(depth_map):
                return depth_map[dm_idx]
        # Fallback: naive cumulative scan
        d = 0
        for ln in lines[:i]:
            for ch in ln:
                if ch in "([{":
                    d += 1
                elif ch in ")]}":
                    d = max(0, d - 1)
        return d

    orig = idx
    for _ in range(max_shift + 1):
        idx = max(0, min(n, idx))
        if _depth(idx) == 0:
            line_text = lines[idx].rstrip() if idx < n else ""
            if _CONTINUATION_RE.match(line_text):
                idx += direction
                continue
            if direction < 0 and _DECORATOR_RE.match(line_text):
                idx += direction
                continue
            return idx
        idx += direction

    return max(0, min(n, orig))   # best effort: return original


def _extract_section(content: str, ranges: list[tuple[int, int]]) -> tuple[str, int, int]:
    """Extract the union of all ranges as a single text block.

    Returns (section_text, first_line_1indexed, last_line_1indexed).
    Boundaries are snapped to syntactically safe splice points using the
    tokenize-based depth map.
    """
    if not ranges:
        lines = content.splitlines(keepends=True)
        return content, 1, len(lines)

    lines = content.splitlines(keepends=True)
    n = len(lines)
    first = ranges[0][0]
    last = ranges[-1][1]

    depth_map = _build_depth_map(content)
    first = _safe_boundary(lines, first, direction=-1, depth_map=depth_map)
    last  = _safe_boundary(lines, last,  direction=+1, depth_map=depth_map)
    first = max(0, first)
    last  = min(n, last)

    section_lines = lines[first:last]
    return "".join(section_lines), first + 1, last


def _splice_section(
    base_content: str,
    rewritten_section: str,
    ranges: list[tuple[int, int]],
    *,
    file_path: str = "",
    validate: bool = True,
    max_expansion: int = 8,
) -> str:
    """Replace the extracted section in base_content with rewritten_section.

    For Python files (validate=True), performs an ast.parse check on the
    spliced result. If it fails, automatically expands the splice window by
    up to max_expansion lines on each side and retries.  This catches cases
    where the rewritten section is internally valid but creates a syntax error
    at its junction with the surrounding code (e.g. dangling elif).

    Returns the spliced string. On repeated ast.parse failures, returns the
    best result anyway (with a warning) so the structural_errors checker
    upstream can handle it.
    """
    if not ranges:
        return rewritten_section

    lines = base_content.splitlines(keepends=True)
    n = len(lines)
    first = ranges[0][0]
    last  = ranges[-1][1]

    rewritten_lines = rewritten_section.splitlines(keepends=True)
    if rewritten_lines and not rewritten_lines[-1].endswith("\n"):
        rewritten_lines[-1] += "\n"

    do_validate = validate and (not file_path or file_path.endswith(".py"))

    if not do_validate:
        result = lines[:first] + rewritten_lines + lines[last:]
        return "".join(result)

    # Build the depth map once for efficient boundary snapping
    depth_map = _build_depth_map(base_content)
    best: str | None = None

    for expansion in range(max_expansion + 1):
        # Expand window and re-snap to safe boundaries
        f = _safe_boundary(lines, max(0, first - expansion),  -1, depth_map)
        l = _safe_boundary(lines, min(n, last  + expansion),  +1, depth_map)
        f = max(0, f)
        l = min(n, l)

        candidate = "".join(lines[:f] + rewritten_lines + lines[l:])
        try:
            ast.parse(candidate)
            if expansion > 0:
                logger.debug(
                    "_splice_section %s: needed expansion=%d to pass ast.parse",
                    file_path, expansion,
                )
            return candidate
        except SyntaxError:
            if best is None:
                best = candidate   # keep the unexpanded version for the error reporter

    # All expansions failed — return the unexpanded splice so structural_errors
    # can report the exact error back to the LLM for correction.
    logger.warning(
        "_splice_section %s: all %d expansions failed ast.parse — returning raw splice",
        file_path, max_expansion,
    )
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main rewrite function
# ---------------------------------------------------------------------------

def _find_top_level_symbols(content: str) -> set[str]:
    """Return names of all top-level def/class statements."""
    symbols: set[str] = set()
    for line in content.splitlines():
        m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if m:
            symbols.add(m.group(2))
    return symbols


def _apply_linters(content: str, file_path: str, repo_config: dict) -> str:
    """Apply code formatters based on repo_config lint_commands (black, ruff).

    Only runs formatters that can operate in-process and are safe to apply
    without side effects. Silently skips if unavailable.
    """
    if not file_path.endswith(".py"):
        return content

    lint_cmds = " ".join(
        repo_config.get("pr_preparation", {}).get("lint_commands", [])
    ).lower()
    # Always run black if it's listed; fall through to ruff after
    if "black" in lint_cmds or not lint_cmds:
        try:
            import black
            content = black.format_str(content, mode=black.Mode())
            logger.debug("black formatted %s", file_path)
        except Exception as e:
            logger.debug("black formatting skipped for %s: %s", file_path, e)

    if "ruff" in lint_cmds:
        try:
            import subprocess, tempfile
            from pathlib import Path as _P
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(content)
                tmp = f.name
            r = subprocess.run(
                ["ruff", "check", "--fix", "--quiet", tmp],
                capture_output=True, text=True,
            )
            if r.returncode in (0, 1):  # 1 = fixed some issues
                content = _P(tmp).read_text()
            _P(tmp).unlink(missing_ok=True)
        except Exception as e:
            logger.debug("ruff skipped for %s: %s", file_path, e)

    return content


def _validate_rewrite(
    base_content: str,
    rewritten: str,
    file_path: str,
    pr_index: int,
    in_scope: list[str] | None = None,
) -> list[str]:
    """Fast structural checks before running the expensive judge.

    Returns a list of error descriptions (empty = passes all checks).
    """
    import subprocess, tempfile
    from pathlib import Path as _Path

    errors: list[str] = []

    # 1. Python syntax check
    if file_path.endswith(".py"):
        try:
            ast.parse(rewritten)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e}")
            return errors  # no point continuing if syntax is broken

    # 2. Scope-creep and signature-mutation checks for Python files.
    if file_path.endswith(".py") and in_scope is not None:
        in_scope_text = " ".join(in_scope)
        base_syms = _find_top_level_symbols(base_content)
        out_syms = _find_top_level_symbols(rewritten)

        # 2a. Symbol removal: base symbols absent from rewrite and not in in_scope.
        missing = base_syms - out_syms
        if missing:
            unexpected = {s for s in missing if s not in in_scope_text}
            if unexpected:
                errors.append(
                    f"Scope-creep: rewrite removed symbols not in in_scope: "
                    f"{sorted(unexpected)}. These must appear unchanged."
                )

        # 2b. Signature mutation: existing symbols whose def/class line changed
        #     but are not mentioned in in_scope (catches stub replacements).
        def _first_def_line(code: str, sym: str) -> str:
            for line in code.splitlines():
                m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                if m and m.group(2) == sym:
                    return line.rstrip()
            return ""

        mutated = []
        for sym in base_syms & out_syms:
            if sym in in_scope_text:
                continue
            base_sig = _first_def_line(base_content, sym)
            out_sig = _first_def_line(rewritten, sym)
            if base_sig and out_sig and base_sig != out_sig:
                mutated.append(sym)
        if mutated:
            errors.append(
                f"Scope-creep: rewrite mutated signatures of out-of-scope symbols: "
                f"{sorted(mutated)}. Preserve their def/class lines verbatim."
            )

        # 2c. Stub-body detection: out-of-scope function replaced with raise/pass stub.
        def _is_stub_body(code: str, sym: str) -> bool:
            """Return True if sym's body is only raise NotImplementedError or pass."""
            lines = code.splitlines()
            in_body = False
            body_lines: list[str] = []
            for line in lines:
                m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                if m:
                    if m.group(2) == sym:
                        in_body = True
                        body_lines = []
                        continue
                    elif in_body:
                        break
                if in_body:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                        body_lines.append(stripped)
            if not body_lines:
                return False
            non_trivial = [l for l in body_lines if l not in ("pass", "...")]
            return len(non_trivial) == 1 and non_trivial[0].startswith("raise NotImplementedError")

        stubbed = [
            sym for sym in base_syms & out_syms
            if sym not in in_scope_text and _is_stub_body(rewritten, sym) and not _is_stub_body(base_content, sym)
        ]
        if stubbed:
            errors.append(
                f"Scope-creep: rewrite replaced body of out-of-scope symbol(s) with stub: "
                f"{sorted(stubbed)}. Copy the original body verbatim."
            )

    # 3. Patch must apply cleanly to base (use a real git repo context)
    patch = generate_unified_patch(base_content, rewritten, file_path)
    if patch:
        with tempfile.TemporaryDirectory(prefix="pr_validate_") as tmpdir:
            # Init a minimal git repo and write the base file at the correct path
            nested = _Path(tmpdir) / _Path(file_path).parent
            nested.mkdir(parents=True, exist_ok=True)
            base_path = _Path(tmpdir) / file_path
            base_path.write_text(base_content, encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=tmpdir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=test@test.com", "-c", "user.name=test",
                 "commit", "-q", "-m", "base"],
                cwd=tmpdir, capture_output=True,
            )
            patch_path = _Path(tmpdir) / "check.patch"
            patch_path.write_text(patch, encoding="utf-8")
            result = subprocess.run(
                ["git", "apply", "--check", str(patch_path)],
                capture_output=True, text=True,
                cwd=tmpdir,
            )
            if result.returncode != 0:
                errors.append(f"git apply --check failed: {result.stderr.strip()[:200]}")

    return errors


def validate_rewrite(
    base_content: str,
    rewritten_content: str,
    file_path: str,
    in_scope: list[str] | None = None,
    pr_index: int = 0,
) -> list[str]:
    """Public wrapper around _validate_rewrite for use as an RLM tool.

    Returns a list of structural error descriptions. Empty list = passes all checks.
    Checks: Python syntax, symbol drops, signature mutations, stub-body replacements,
    patch applicability.
    """
    return _validate_rewrite(base_content, rewritten_content, file_path, pr_index, in_scope)


def _dspy_rewrite_file(
    file_path: str,
    base_content: str,
    pr_spec: dict,
    repo: str,
    contrib_rules: str,
    *,
    model: str = "claude-opus-4-7",
    upstream_repo: str | None = None,
    token: str | None = None,
    critic_hints_block: str = "",
    arch_constraints_block: str = "",
    use_whole_file: bool = True,
    base_section: str = "",
    seed_diff_hint: str = "",
    in_scope_block: str = "",
    out_of_scope_block: str = "",
    line_start: int = 1,
    line_end: int = 0,
) -> str | None:
    """DSPy ReAct rewriter with upstream fetch capability.

    The agent can call fetch_upstream_file and search_upstream_symbol to look up
    related headers, function signatures, and type definitions before producing
    the rewritten content. This prevents hallucinated call signatures, wrong
    tensor shapes, and other bugs that stem from the rewriter being blind to
    cross-file context.

    Returns the rewritten content string, or None if dspy is unavailable or fails
    (caller should fall back to static llm_call).
    """
    try:
        import dspy
    except ImportError:
        return None

    import base64
    import os
    import httpx as _httpx

    gateway = os.environ.get("LITELLM_GATEWAY_URL", "http://localhost:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    # Whole-file rewrites need 128k output tokens to avoid row/line truncation.
    _dspy_max_tokens = 128000 if use_whole_file else 32768
    _lm = dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
        max_tokens=_dspy_max_tokens,
    )

    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _headers = {"Authorization": f"token {_token}", "Accept": "application/vnd.github.v3+json"}
    _fetched: dict[str, str] = {}

    def _gh_get(url: str) -> dict | list | None:
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("GitHub API error %s: %s", url, exc)
            return None

    def fetch_upstream_file(path: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Fetch a slice of an upstream file to verify signatures, shapes, or conventions.
        For large files call iteratively: first call shows [Lines 1–N of TOTAL]; continue
        with start_line=200, 400, … to read further sections.
        Args:
            path: file path relative to repo root (e.g. 'csrc/cache.h', 'vllm/attention/ops.py')
            start_line: 0-indexed line to start from (default 0)
            num_lines: lines to return per call (default 200)
        Returns [Lines X–Y of Z] header + content, or an error message."""
        if not upstream_repo:
            return "(upstream_repo not available — cannot fetch)"
        cache_key = path
        if cache_key not in _fetched:
            owner, repo_name = upstream_repo.split("/", 1)
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            data = _gh_get(url)
            if not data or not isinstance(data, dict) or data.get("encoding") != "base64":
                return f"(file not found: {path})"
            try:
                text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                _fetched[cache_key] = text
            except Exception as exc:
                return f"(decode error: {exc})"
        lines = _fetched[cache_key].splitlines()
        total = len(lines)
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(chunk)
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def search_upstream_symbol(symbol: str) -> str:
        """Search the upstream repo for where a function, struct, or type is defined.
        Use this to find call signatures, parameter counts, or type definitions you
        need to reference correctly in the rewritten code.
        Args:
            symbol: exact symbol name (e.g. 'concat_and_cache_mla_rope_fused', 'KVCache')
        Returns file paths where the symbol appears."""
        if not upstream_repo:
            return "(upstream_repo not available — cannot search)"
        owner, repo_name = upstream_repo.split("/", 1)
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {symbol})"
        items = data.get("items", [])
        if not items:
            return f"(not found upstream: {symbol})"
        return "\n".join(f"  {item['path']}" for item in items[:5])

    _DATA_ARTIFACT_EXTS = {".csv", ".json", ".yaml", ".yml", ".toml"}
    _is_data_artifact = any(file_path.endswith(ext) for ext in _DATA_ARTIFACT_EXTS)

    if _is_data_artifact and seed_content and seed_content != base_content:
        # For data/config files the exact values ARE the implementation — show
        # a line-by-line diff between upstream base and seed so the model
        # reproduces the change precisely rather than guessing structure.
        _seed_lines = seed_content.splitlines()
        _base_lines_da = base_content.splitlines()
        _added = [l for l in _seed_lines if l not in set(_base_lines_da)]
        _removed = [l for l in _base_lines_da if l not in set(_seed_lines)]
        _da_diff_parts = [f"+ {l}" for l in _added[:80]] + [f"- {l}" for l in _removed[:20]]
        _da_diff_preview = "\n".join(_da_diff_parts) or "(no row-level differences detected)"
        _hint_block = (
            f"DATA ARTIFACT CHANGE REQUIRED — apply the exact values from the seed:\n"
            f"The seed version of `{file_path}` differs from upstream. "
            f"You MUST reproduce these exact additions/removals (do not paraphrase or summarise):\n"
            f"```\n{_da_diff_preview}\n```\n"
            f"RULE: Data file correctness requires exact field values. "
            f"Do not omit rows or change values."
        )
    else:
        _hint_block = (
            f"SEED DIFF HINT (what the seed author changed — implement from scratch, don't copy):\n```\n{seed_diff_hint[:_SECTION_CAP]}\n```"
            if seed_diff_hint
            else "SEED DIFF HINT: (none — implement the objective from scratch against the base)"
        )
    if use_whole_file:
        task_description = (
            f"BASE FILE (`{file_path}` on `main`):\n```\n{base_content}\n```\n\n"
            + _hint_block
        )
        output_instruction = "Return the COMPLETE rewritten file content. No markdown fences, no explanation — just the file."
    else:
        task_description = (
            f"BASE SECTION (`{file_path}` lines {line_start}–{line_end}):\n```\n{base_section}\n```\n\n"
            + _hint_block
        )
        output_instruction = (
            f"Return ONLY the rewritten section (lines {line_start}–{line_end}). "
            "No markdown fences, no explanation — just the section content."
        )

    class RewriteSignature(dspy.Signature):
        f"""You are an expert contributor to the GitHub repository "{repo}".

        Your task: rewrite `{file_path}` for PR {pr_spec['index']} — {pr_spec['title']}.

        OBJECTIVE:
        {pr_spec['objective']}

        WHAT TO CHANGE:
        {in_scope_block}

        WHAT MUST NOT CHANGE:
        {out_of_scope_block}

        {critic_hints_block}{arch_constraints_block}CONTRIBUTION RULES:
        {contrib_rules}

        {task_description}

        The SEED DIFF HINT above (if present) shows how the seed author implemented this.
        Use it only to understand the intent — do NOT copy it. Your implementation must
        use upstream repo idioms and patterns, derived from fetching the actual upstream files.

        REQUIRED — you MUST call fetch_upstream_file at least once before producing
        any output. Do NOT write a single line of code until you have fetched and
        read the upstream source. Skipping this step will produce wrong field names,
        wrong call signatures, and wrong type annotations — the PR will be rejected.

        Mandatory fetch sequence (do this before writing ANYTHING):
        1. For every function, config class, struct, or Pydantic model that the
           changed code calls, subclasses, or reads fields from — call
           fetch_upstream_file on the file that DEFINES it. Verify the exact field
           names, parameter order, and types from the live upstream source.
        2. For C/CUDA files, fetch every .h header that declares functions or structs
           you are calling or extending.
        3. Use search_upstream_symbol if you cannot find the definition by path.
        4. Only AFTER completing all fetches above, produce the rewritten content.

        Rules:
        - The BASE shown above is authoritative — it is the exact file content this PR
          must build on. It may already contain changes from earlier PRs in the stack.
          Use fetch_upstream_file ONLY to look up types, call signatures, and idioms from
          OTHER files — NEVER to replace or override the BASE content above.
        - Start from the BASE exactly. Apply ONLY changes that serve the objective.
        - Every line not touched by this PR must be identical to BASE.
        - NEVER remove function parameters — downstream callers depend on them.
        - NEVER alter macro names or intrinsic names in C/CUDA (exact letter-for-letter copy).
        - NEVER rename struct fields, config keys, dataclass attributes, or dict keys
          unless the rename is explicitly in "WHAT TO CHANGE". The fetched upstream
          source is the authority — do not guess or infer names from context.
        - Do NOT add comments explaining what you changed.
        - {output_instruction}"""

        task: str = dspy.InputField(desc="Rewrite task description with base and reference content")
        answer: str = dspy.OutputField(desc="The rewritten file or section content (no fences)")

    try:
        agent = dspy.ReAct(
            RewriteSignature,
            tools=[fetch_upstream_file, search_upstream_symbol],
            max_iters=8,
        )
        _in_scope_summary = "; ".join(pr_spec.get("in_scope", []))[:300]
        _task_prompt = (
            f"STEP 1 (MANDATORY): Call fetch_upstream_file on every file that defines "
            f"a config class, struct, function, or type that `{file_path}` imports or "
            f"calls into. Do this NOW before writing anything.\n\n"
            f"STEP 2: Rewrite `{file_path}` for PR {pr_spec['index']} — {pr_spec['title']}.\n"
            f"Objective: {pr_spec['objective']}\n"
            f"In scope: {_in_scope_summary}\n\n"
            "STEP 3 — SELF-CHECK before returning:\n"
            "- If your output contains entries with fields that follow a mathematical "
            "relationship, verify each entry satisfies it and fix any that do not.\n"
            "- Entries that represent independent measurements must not be byte-identical "
            "to each other unless their inputs are also identical."
        )
        with dspy.context(lm=_lm):
            prediction = agent(task=_task_prompt)
        result = _strip_fences(prediction.answer or "")
        logger.info(
            "  DSPy rewrite %s PR %d: fetched %d upstream files",
            file_path, pr_spec["index"], len(_fetched),
        )
        try:
            from pipeline.tracing import flush_dspy_history
            flush_dspy_history(model, stage="rewrite_file")
        except Exception:
            pass
        return result if result.strip() else None
    except Exception as exc:
        logger.warning("DSPy rewrite failed for %s PR %d (%s) — falling back to static", file_path, pr_spec["index"], exc)
        return None


def rewrite_file_for_pr(
    file_path: str,
    base_content: str,
    patched_content: str,
    pr_spec: dict,
    repo: str,
    repo_config: dict,
    *,
    model: str = "claude-opus-4-7",
    max_retries: int = _MAX_RETRIES,
    extra_violations: list[dict] | None = None,
    audit_hints: list[str] | None = None,
    critic_hints: list[str] | None = None,
    arch_constraints: list[str] | None = None,
    upstream_repo: str | None = None,
    token: str | None = None,
    seed_content: str = "",
    prior_pr_summary: str = "",
) -> str:
    """Produce the target file content for one PR's objective.

    Uses section-based rewriting: extracts only the relevant functions/classes,
    rewrites them in isolation, then splices back into the base file. Falls back
    to whole-file rewrite for small files.

    After each attempt, the resulting patch is validated with judge_patch.
    Violations are fed back for correction (up to max_retries).

    Returns the complete rewritten file as a string.
    """
    from pipeline.llm import llm_call, make_client
    from pipeline.judge import judge_patch
    from pipeline.tracing import trace_stage as _trace_stage

    repo_slug = repo.replace("/", "_", 1)
    in_scope = pr_spec.get("in_scope", [])

    in_scope_block = "\n".join(f"  - {s}" for s in in_scope)
    out_of_scope_block = "\n".join(f"  - {s}" for s in pr_spec.get("out_of_scope", []))
    if not out_of_scope_block:
        out_of_scope_block = "  (see plan scope_creep — all changes not listed above)"

    contrib_rules = _build_contrib_rules(repo_config)
    client = make_client()

    _trace_ctx = _trace_stage("rewrite_file", pr_index=pr_spec.get("index"), file=file_path)
    _trace_ctx.__enter__()

    base_lines = base_content.splitlines()
    n_lines = len(base_lines)
    # Threshold: use whole-file rewrite for files that fit comfortably
    # ~60k chars / 4 chars per token ≈ 15k tokens; leave room for prompt overhead
    # Whole-file mode: use when file is ≤ 800 lines
    use_whole_file = n_lines <= 800

    # Short-circuit: purely additive changes (new file, append-only CSV/data)
    # don't need an LLM — the seed patch is already the correct output.
    if _is_additive_only(base_content, patched_content, file_path):
        logger.info(
            "  %s: additive-only change — using patched_content directly (no LLM rewrite)",
            file_path,
        )
        _trace_ctx.__exit__(None, None, None)
        return patched_content

    # Pre-compute a filtered structural diff of seed vs base to use as a
    # low-weight implementation hint. Symbol names resolved later (section path);
    # for whole-file path we pass all in-scope text as the filter.
    _in_scope_symbols_text = set(" ".join(in_scope).split())
    _seed_diff_hint_full = _extract_seed_diff_hint(base_content, seed_content, _in_scope_symbols_text)

    _critic_hints_block = ""
    _empty_diff_escalation = any(
        "empty_diff" in (h or "").lower() or "diff is empty" in (h or "").lower()
        for h in (critic_hints or [])
    )
    if _empty_diff_escalation:
        # Previous iteration produced an empty diff for this file. Force whole-file
        # rewrite mode so the model can see the full context, and append an escape-hatch
        # directive: produce a real edit OR explicitly justify / defer the file.
        use_whole_file = True
    if critic_hints:
        _hints_fmt = "\n".join(f"  - {h}" for h in critic_hints)
        _critic_hints_block = (
            "## MANDATORY FIXES — PREVIOUS REWRITE HAD THESE BUGS\n\n"
            "A code reviewer rejected the last rewrite for these specific bugs.\n"
            "You MUST fix ALL of them. They are hard requirements, not suggestions.\n\n"
            f"{_hints_fmt}\n\n---\n\n"
        )
        if _empty_diff_escalation:
            _critic_hints_block += (
                "## EMPTY-DIFF ESCALATION\n\n"
                "The previous iteration emitted no changes for this file. Re-read the "
                "in_scope instructions below — they describe exactly what must be added "
                "or changed. Implement that now. Either:\n"
                "  (a) Produce a concrete, non-empty edit that advances the PR objective, OR\n"
                "  (b) If the file genuinely requires no change for this objective, return\n"
                "      the base content unchanged and add a one-line justification at the\n"
                "      top of your reasoning explaining why no change is needed.\n"
                "Do NOT silently re-emit the base. Do NOT invent edits to escape this loop.\n\n---\n\n"
            )

    _arch_constraints_block = ""
    if arch_constraints:
        _ac_fmt = "\n".join(f"  - {c}" for c in arch_constraints)
        _arch_constraints_block = (
            "## ARCHITECTURAL CONSTRAINTS (HARD — violations will be rejected)\n\n"
            "The following architectural principles MUST be respected. Any violation\n"
            "will cause the PR to be rejected during review. These are not style\n"
            "suggestions — they are mandatory structural laws for this codebase:\n\n"
            f"{_ac_fmt}\n\n---\n\n"
        )

    if prior_pr_summary:
        # Prepend ancestor-PR preservation note to arch_constraints_block so it
        # reaches every call path (DSPy task prompt, static whole-file prompt, section
        # prompt) without updating each call site individually.
        _prior_pr_block = (
            "## ANCESTOR PR CHANGES — PRESERVE THESE\n\n"
            "This PR stacks on top of earlier PRs in the series. The BASE content above "
            "already includes their changes. Do NOT remove, revert, or duplicate any of "
            "the following — they must be present unchanged in your output:\n\n"
            f"{prior_pr_summary}\n\n---\n\n"
        )
        _arch_constraints_block = _prior_pr_block + _arch_constraints_block

    # Pre-compute symbol ranges to route non-Python files (CSV/headers/data) to the
    # 128k whole-file path before reaching the use_whole_file / section branches.
    core_ranges: list[tuple[int, int]] = _find_symbol_ranges(base_content, in_scope, context=0)
    ranges: list[tuple[int, int]] = _find_symbol_ranges(base_content, in_scope)

    if not ranges:
        # Non-Python file (CSV, C/C++ header, data file, etc.).

        # DSPy whole-file rewrite with 128k tokens — for non-Python files (CSV, headers, data files).
        logger.info(
            "  No symbols identified for %s (%d lines) — DSPy whole-file rewrite (128k tokens)",
            file_path, n_lines,
        )
        _wf_hint = _extract_seed_diff_hint(base_content, seed_content, _in_scope_symbols_text)
        dspy_result = _dspy_rewrite_file(
            file_path, base_content, pr_spec, repo, contrib_rules,
            model=model, upstream_repo=upstream_repo, token=token,
            critic_hints_block=_critic_hints_block,
            arch_constraints_block=_arch_constraints_block,
            use_whole_file=True,
            seed_diff_hint=_wf_hint,
            in_scope_block=in_scope_block,
            out_of_scope_block=out_of_scope_block,
        )
        if dspy_result:
            candidate = dspy_result
        else:
            _wf_prompt = _WHOLE_FILE_PROMPT.format(
                repo=repo,
                file_path=file_path,
                pr_index=pr_spec["index"],
                pr_title=pr_spec["title"],
                objective=pr_spec["objective"],
                in_scope_block=in_scope_block,
                out_of_scope_block=out_of_scope_block,
                critic_hints_block=_critic_hints_block,
                arch_constraints_block=_arch_constraints_block,
                contrib_rules=contrib_rules,
                base_content=base_content,
                seed_diff_hint=_wf_hint[:_SECTION_CAP],
            )
            candidate = _strip_fences(llm_call(_wf_prompt, model, client=client, max_tokens=128000))
        rewritten = _csv_sanity_check(candidate, base_content, file_path, fallback=patched_content)
        use_whole_file = True
        ranges = []
        core_ranges = []
    elif use_whole_file:
        dspy_result = _dspy_rewrite_file(
            file_path, base_content, pr_spec, repo, contrib_rules,
            model=model, upstream_repo=upstream_repo, token=token,
            critic_hints_block=_critic_hints_block,
            arch_constraints_block=_arch_constraints_block,
            use_whole_file=True,
            seed_diff_hint=_seed_diff_hint_full,
            in_scope_block=in_scope_block,
            out_of_scope_block=out_of_scope_block,
        )
        if dspy_result:
            rewritten = _csv_sanity_check(dspy_result, base_content, file_path, fallback=patched_content)
        else:
            prompt = _WHOLE_FILE_PROMPT.format(
                repo=repo,
                file_path=file_path,
                pr_index=pr_spec["index"],
                pr_title=pr_spec["title"],
                objective=pr_spec["objective"],
                in_scope_block=in_scope_block,
                out_of_scope_block=out_of_scope_block,
                critic_hints_block=_critic_hints_block,
                arch_constraints_block=_arch_constraints_block,
                contrib_rules=contrib_rules,
                base_content=base_content,
                seed_diff_hint=_seed_diff_hint_full[:_SECTION_CAP],
            )
            rewritten = _csv_sanity_check(_strip_fences(llm_call(prompt, model, client=client, max_tokens=32768, timeout=900.0)), base_content, file_path, fallback=patched_content)
    else:
        # Section-based rewrite (large Python file, ranges computed above).
        # Use core_ranges (no padding) for LLM display and splicing — avoids leaving
        # pre-context lines in the base that the LLM omits (causing unclosed-paren errors).
        base_section, line_start, line_end = _extract_section(base_content, core_ranges)
        section_lines = line_end - line_start + 1

        if section_lines > _SECTION_LINE_CAP:
            # Section too large to splice reliably — use two-pass:
            # Pass 1: whole-file LLM rewrite (128k tokens) on base_content (not seed).
            # Pass 2: extract same logical section from valid candidate, splice into base.
            # This preserves all symbols PR N-1 added that aren't in the seed.
            logger.info(
                "  Section too large (%d of %d) — two-pass: whole-file rewrite then section extract",
                section_lines, n_lines,
            )
            _surgical_note = (
                "## PATCH-ONLY CONSTRAINT\n\n"
                "The seed for this PR was a whole-file replacement, so there is no patch\n"
                "to apply mechanically. You must implement the objective as a MINIMAL,\n"
                "SURGICAL PATCH against the BASE FILE:\n\n"
                "- Touch only the exact lines needed to implement 'WHAT TO CHANGE'.\n"
                "- Every function, class, constant, and import NOT mentioned in\n"
                "  'WHAT TO CHANGE' must appear byte-for-byte identical in your output.\n"
                "- If you are adding a new constant or helper, insert it in the most\n"
                "  natural location without moving any surrounding code.\n"
                "- Think of your output as a git patch: the reviewer should see only\n"
                "  the new lines and nothing else changed.\n\n---\n\n"
            )
            _arch_constraints_block = _surgical_note + _arch_constraints_block
            _wf_hint = _extract_seed_diff_hint(base_content, seed_content, _in_scope_symbols_text)
            _wf_prompt = _WHOLE_FILE_PROMPT.format(
                repo=repo,
                file_path=file_path,
                pr_index=pr_spec["index"],
                pr_title=pr_spec["title"],
                objective=pr_spec["objective"],
                in_scope_block=in_scope_block,
                out_of_scope_block=out_of_scope_block,
                critic_hints_block="",
                arch_constraints_block=_arch_constraints_block,
                contrib_rules=contrib_rules,
                base_content=base_content,
                seed_diff_hint=_wf_hint[:_SECTION_CAP],
            )
            # Use 128k output tokens (model max) — the full file is ~20k tokens;
            # 32768 truncated the output causing syntax errors.
            candidate = _strip_fences(llm_call(_wf_prompt, model, client=client, max_tokens=128000))
            # Validate candidate before splicing — truncation produces syntax errors.
            try:
                ast.parse(candidate)
                _candidate_valid = True
            except SyntaxError:
                _candidate_valid = False
            # Symbols already anywhere in the base file (new PR can reference them).
            _all_base_syms = _find_top_level_symbols(base_content)
            # Symbols the planner declared as in-scope (resolved from instruction text).
            _allowed_new_syms = _identify_symbols_llm(in_scope, _all_base_syms) if _all_base_syms else set()

            def _get_def_signatures(code: str) -> dict[str, str]:
                """Return {symbol: first def/class line} for top-level symbols."""
                sigs: dict[str, str] = {}
                for line in code.splitlines():
                    m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                    if m and m.group(2) not in sigs:
                        sigs[m.group(2)] = line.rstrip()
                return sigs

            def _candidate_section_ok(cand: str, base_sec: str) -> bool:
                """Return True iff candidate section doesn't drop, add-creep, or silently mutate base symbols."""
                base_syms = _find_top_level_symbols(base_sec)
                cand_syms = _find_top_level_symbols(cand)
                dropped = base_syms - cand_syms
                if dropped:
                    logger.warning(
                        "  Two-pass candidate section drops base symbols: %s — rejecting candidate",
                        sorted(dropped),
                    )
                    return False
                # Scope-creep check: new symbols must be explicitly declared in-scope.
                added = cand_syms - base_syms
                creep = added - _allowed_new_syms
                if creep:
                    logger.warning(
                        "  Two-pass candidate section adds undeclared symbols: %s — rejecting candidate",
                        sorted(creep),
                    )
                    return False
                # Signature-mutation check: existing symbols not in in_scope must keep
                # the same def/class line (catches signature rewrites like get_inter_dim).
                base_sigs = _get_def_signatures(base_sec)
                cand_sigs = _get_def_signatures(cand)
                mutated = [
                    sym for sym, sig in base_sigs.items()
                    if sym not in _allowed_new_syms and sym in cand_sigs and cand_sigs[sym] != sig
                ]
                if mutated:
                    logger.warning(
                        "  Two-pass candidate section mutated signatures of out-of-scope symbols: %s — rejecting candidate",
                        sorted(mutated),
                    )
                    return False
                # Stub-body check: out-of-scope function replaced with raise NotImplementedError.
                def _sec_is_stub(code: str, sym: str) -> bool:
                    lines = code.splitlines()
                    in_body = False
                    body_lines: list[str] = []
                    for line in lines:
                        m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                        if m:
                            if m.group(2) == sym:
                                in_body = True
                                body_lines = []
                                continue
                            elif in_body:
                                break
                        if in_body:
                            stripped = line.strip()
                            if stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                                body_lines.append(stripped)
                    if not body_lines:
                        return False
                    non_trivial = [l for l in body_lines if l not in ("pass", "...")]
                    return len(non_trivial) == 1 and non_trivial[0].startswith("raise NotImplementedError")

                stubbed = [
                    sym for sym in base_syms & cand_syms
                    if sym not in _allowed_new_syms and _sec_is_stub(cand, sym) and not _sec_is_stub(base_sec, sym)
                ]
                if stubbed:
                    logger.warning(
                        "  Two-pass candidate section stubs out-of-scope symbols: %s — rejecting candidate",
                        sorted(stubbed),
                    )
                    return False
                return True

            if _candidate_valid:
                candidate_core = _find_symbol_ranges(candidate, in_scope, context=0)
                if candidate_core:
                    candidate_section, _, _ = _extract_section(candidate, candidate_core)
                    base_section_syms, _, _ = _extract_section(base_content, core_ranges)
                    if _candidate_section_ok(candidate_section, base_section_syms):
                        rewritten = _splice_section(base_content, candidate_section, core_ranges, file_path=file_path)
                    else:
                        # Whole-file candidate adds/drops/mutates out-of-scope symbols —
                        # fall back to a section-only rewrite so the objective still lands.
                        logger.warning("  Two-pass: candidate section rejected; falling back to section-only rewrite")
                        _sec_prompt = _REWRITE_PROMPT.format(
                            repo=repo,
                            file_path=file_path,
                            pr_index=pr_spec["index"],
                            pr_title=pr_spec["title"],
                            objective=pr_spec["objective"],
                            in_scope_block=in_scope_block,
                            out_of_scope_block=out_of_scope_block,
                            critic_hints_block="",
                            arch_constraints_block=_arch_constraints_block,
                            contrib_rules=contrib_rules,
                            base_line_start=line_start,
                            base_line_end=line_end,
                            base_section=base_section[:_SECTION_CAP],
                            seed_diff_hint=_extract_seed_diff_hint(
                                base_content, seed_content,
                                set(in_scope) if isinstance(in_scope, list) else {str(in_scope)},
                            )[:_SECTION_CAP],
                        )
                        _sec_result = _strip_fences(llm_call(_sec_prompt, model, client=client, max_tokens=32768))
                        rewritten = _splice_section(base_content, _sec_result, core_ranges, file_path=file_path)
                else:
                    rewritten = candidate
                    use_whole_file = True
                    ranges = []
                    core_ranges = []
            else:
                # Candidate truncated/invalid — keep ranges so the retry loop's section
                # fix pass works on the section, not the truncated whole file.
                logger.warning(
                    "  Two-pass candidate has syntax errors (likely truncated) — will fix section in retry loop"
                )
                # Extract the portion of candidate that corresponds to our core range
                # and splice as a starting point for the fix loop.
                candidate_core = _find_symbol_ranges(candidate, in_scope, context=0)
                if candidate_core:
                    candidate_section, _, _ = _extract_section(candidate, candidate_core)
                    base_section_syms, _, _ = _extract_section(base_content, core_ranges)
                    if _candidate_section_ok(candidate_section, base_section_syms):
                        rewritten = _splice_section(base_content, candidate_section, core_ranges, file_path=file_path)
                    else:
                        rewritten = base_content
                else:
                    # Can't extract anything useful — fall back to base for fix loop
                    rewritten = base_content
        else:
                # Normal section rewrite (section <= _SECTION_LINE_CAP lines)
                _section_symbols = set()
                for line in base_section.splitlines():
                    m = re.match(r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                    if m:
                        _section_symbols.add(m.group(2))
                _seed_diff_hint_section = _extract_seed_diff_hint(
                    base_content, seed_content, _section_symbols or _in_scope_symbols_text
                )

                logger.info(
                    "  Section rewrite: lines %d–%d (%d lines of %d total)",
                    line_start, line_end, section_lines, n_lines,
                )

                _critic_hints_block = ""
                if critic_hints:
                    _hints_fmt = "\n".join(f"  - {h}" for h in critic_hints)
                    _critic_hints_block = (
                        "## MANDATORY FIXES — PREVIOUS REWRITE HAD THESE BUGS\n\n"
                        "A code reviewer rejected the last rewrite for these specific bugs.\n"
                        "You MUST fix ALL of them. They are hard requirements, not suggestions.\n\n"
                        f"{_hints_fmt}\n\n---\n\n"
                    )
                dspy_section = _dspy_rewrite_file(
                    file_path, base_content, pr_spec, repo, contrib_rules,
                    model=model, upstream_repo=upstream_repo, token=token,
                    critic_hints_block=_critic_hints_block,
                    arch_constraints_block=_arch_constraints_block,
                    use_whole_file=False,
                    base_section=base_section[:_SECTION_CAP],
                    seed_diff_hint=_seed_diff_hint_section,
                    in_scope_block=in_scope_block,
                    out_of_scope_block=out_of_scope_block,
                    line_start=line_start,
                    line_end=line_end,
                )
                if dspy_section:
                    rewritten_section = dspy_section
                else:
                    prompt = _REWRITE_PROMPT.format(
                        repo=repo,
                        file_path=file_path,
                        pr_index=pr_spec["index"],
                        pr_title=pr_spec["title"],
                        objective=pr_spec["objective"],
                        in_scope_block=in_scope_block,
                        out_of_scope_block=out_of_scope_block,
                        critic_hints_block=_critic_hints_block,
                        arch_constraints_block=_arch_constraints_block,
                        contrib_rules=contrib_rules,
                        base_line_start=line_start,
                        base_line_end=line_end,
                        base_section=base_section[:_SECTION_CAP],
                        seed_diff_hint=_seed_diff_hint_section[:_SECTION_CAP],
                    )
                    rewritten_section = _strip_fences(
                        llm_call(prompt, model, client=client, max_tokens=32768)
                    )
                rewritten = _splice_section(base_content, rewritten_section, core_ranges, file_path=file_path)

    # If caller provided pre-known violations (e.g. from Copilot review), run a fix
    # pass before the normal validation loop.
    if extra_violations:
        logger.info(
            "  %s PR %d: applying %d pre-known violations from review before validation",
            file_path, pr_spec["index"], len(extra_violations),
        )
        current_for_fix = rewritten if (use_whole_file or not core_ranges) else _extract_section(rewritten, core_ranges)[0]
        fix_prompt = _FIX_PROMPT.format(
            file_path=file_path,
            pr_index=pr_spec["index"],
            pr_title=pr_spec["title"],
            violations_block=_format_violations(extra_violations),
            current_content=current_for_fix[:_SECTION_CAP],
            objective=pr_spec["objective"],
            in_scope_block=in_scope_block,
        )
        fixed = _strip_fences(llm_call(fix_prompt, model, client=client, max_tokens=32768))
        rewritten = fixed if (use_whole_file or not core_ranges) else _splice_section(base_content, fixed, core_ranges, file_path=file_path)

    # If layer-audit feedback was provided, run a targeted fix pass to address warnings.
    if audit_hints:
        hints_block = "\n".join(f"- {h}" for h in audit_hints)
        logger.info(
            "  %s PR %d: applying %d layer-audit hint(s) before validation",
            file_path, pr_spec["index"], len(audit_hints),
        )
        current_for_audit = rewritten if (use_whole_file or not core_ranges) else _extract_section(rewritten, core_ranges)[0]
        audit_prompt = (
            f"You are fixing a PR rewrite for {repo}/{file_path} (PR {pr_spec['index']}: {pr_spec['title']}).\n\n"
            f"The layer-audit flagged these concerns with the current rewrite:\n{hints_block}\n\n"
            f"Objective: {pr_spec['objective']}\n\n"
            f"Current file content:\n```\n{current_for_audit[:_SECTION_CAP]}\n```\n\n"
            "Produce the corrected file content only (no fences, no explanation)."
        )
        audited = _strip_fences(llm_call(audit_prompt, model, client=client, max_tokens=32768))
        rewritten = audited if (use_whole_file or not core_ranges) else _splice_section(base_content, audited, core_ranges, file_path=file_path)

    # Structural validation + judge feedback loop
    for attempt in range(max_retries):
        # Fast structural checks first (syntax, scope-creep, patch applicability)
        structural_errors = _validate_rewrite(base_content, rewritten, file_path, pr_spec["index"], in_scope)
        if structural_errors:
            logger.warning(
                "  %s PR %d attempt %d: structural errors — %s",
                file_path, pr_spec["index"], attempt + 1, "; ".join(structural_errors),
            )
            if not use_whole_file and core_ranges:
                # Surgical mode (no applicable patch) — section splice produced syntax errors.
                # Extract just the broken section and fix it, then re-splice. Keep core_ranges
                # alive so every retry uses the section approach (not whole-file truncation).
                logger.info(
                    "  %s PR %d: section splice failed (surgical); running section fix pass",
                    file_path, pr_spec["index"],
                )
                bad_section, _, _ = _extract_section(rewritten, core_ranges)
                blocking = [{"severity": "error", "file": file_path, "message": e, "fix_hint": ""} for e in structural_errors]
                fix_prompt = _FIX_PROMPT.format(
                    file_path=file_path,
                    pr_index=pr_spec["index"],
                    pr_title=pr_spec["title"],
                    violations_block=_format_violations(blocking),
                    current_content=bad_section[:_SECTION_CAP],
                    objective=pr_spec["objective"],
                    in_scope_block=in_scope_block,
                )
                fixed_section = _strip_fences(llm_call(fix_prompt, model, client=client, max_tokens=32768))
                rewritten = _splice_section(base_content, fixed_section, core_ranges, file_path=file_path)
            else:
                # Already in whole-file mode — run a targeted fix pass
                blocking = [{"severity": "error", "file": file_path, "message": e, "fix_hint": ""} for e in structural_errors]
                fix_prompt = _FIX_PROMPT.format(
                    file_path=file_path,
                    pr_index=pr_spec["index"],
                    pr_title=pr_spec["title"],
                    violations_block=_format_violations(blocking),
                    current_content=rewritten[:_SECTION_CAP],
                    objective=pr_spec["objective"],
                    in_scope_block=in_scope_block,
                )
                rewritten = _csv_sanity_check(_strip_fences(llm_call(fix_prompt, model, client=client, max_tokens=32768)), base_content, file_path, fallback=patched_content)
            continue

        patch = generate_unified_patch(base_content, rewritten, file_path)
        if not patch:
            break  # no changes — nothing to validate

        try:
            judge_result = judge_patch(repo_slug, patch, model=model)
            violations = judge_result.get("findings", [])
            blocking = [v for v in violations if v.get("severity") in ("error", "critical", "high")]
        except Exception as e:
            logger.warning("judge_patch failed on attempt %d: %s", attempt + 1, e)
            break

        if not blocking:
            if violations:
                logger.info(
                    "  %s PR %d attempt %d: %d non-blocking warnings",
                    file_path, pr_spec["index"], attempt + 1, len(violations),
                )
            break

        logger.info(
            "  %s PR %d attempt %d: %d blocking violations — retrying",
            file_path, pr_spec["index"], attempt + 1, len(blocking),
        )

        # For the fix loop, we feed just the changed region back
        current_for_fix = rewritten if use_whole_file or not core_ranges else _extract_section(rewritten, core_ranges)[0]
        fix_prompt = _FIX_PROMPT.format(
            file_path=file_path,
            pr_index=pr_spec["index"],
            pr_title=pr_spec["title"],
            violations_block=_format_violations(blocking),
            current_content=current_for_fix[:_SECTION_CAP],
            objective=pr_spec["objective"],
            in_scope_block=in_scope_block,
        )
        fixed_section = _strip_fences(llm_call(fix_prompt, model, client=client, max_tokens=32768))

        if use_whole_file or not core_ranges:
            rewritten = fixed_section
        else:
            rewritten = _splice_section(base_content, fixed_section, core_ranges, file_path=file_path)
    else:
        logger.warning(
            "  %s PR %d: still has violations after %d retries",
            file_path, pr_spec["index"], max_retries,
        )
        # Surface unresolved scope-creep as arch_constraints for the composite loop.
        # Parse removed symbol names from the last structural_errors list so the
        # next rewrite iteration can explicitly name them in the out-of-scope block.
        _creep_syms: list[str] = []
        for _err in (structural_errors if "structural_errors" in dir() else []):
            import re as _re2
            _m = _re2.search(r"removed symbols not in in_scope: \[(.+?)\]", _err)
            if _m:
                _creep_syms.extend(s.strip().strip("'\"") for s in _m.group(1).split(","))
        if _creep_syms:
            _unresolved_scope_creep[(file_path, pr_spec["index"])] = _creep_syms

    rewritten = _apply_linters(rewritten, file_path, repo_config)
    _trace_ctx.__exit__(None, None, None)
    return rewritten


def generate_unified_patch(
    base_content: str,
    rewritten_content: str,
    file_path: str,
) -> str:
    """Diff base vs rewritten and return a unified patch string."""
    base_lines = base_content.splitlines()
    rewritten_lines = rewritten_content.splitlines()

    is_new_file = not base_content.strip()

    diff = list(difflib.unified_diff(
        base_lines,
        rewritten_lines,
        fromfile="/dev/null" if is_new_file else f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    ))

    if not diff:
        return ""

    patch = "\n".join(diff) + "\n"
    if is_new_file:
        patch = f"new file mode 100644\n{patch}"
    return patch


def _fetch_upstream_file_content(file_path: str, upstream_repo: str, token: str | None) -> str | None:
    """Fetch a file from the upstream GitHub repo. Returns decoded text or None on failure."""
    import base64 as _b64
    import httpx as _httpx

    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _headers = {"Authorization": f"token {_token}", "Accept": "application/vnd.github.v3+json"}
    try:
        owner, repo_name = upstream_repo.split("/", 1)
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}"
        r = _httpx.get(url, headers=_headers, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or data.get("encoding") != "base64":
            return None
        return _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("_fetch_upstream_file_content: %s/%s → %s", upstream_repo, file_path, exc)
        return None


def rewrite_pr_series(
    plan: dict,
    changed_files: dict[str, tuple[str, str]],
    repo: str,
    repo_config: dict,
    *,
    model: str = "claude-opus-4-7",
    max_retries: int = _MAX_RETRIES,
    audit_feedback: dict[int, list[str]] | None = None,
    critic_feedback: dict[int, list[str]] | None = None,
    pr_indices: list[int] | None = None,
    upstream_repo: str | None = None,
    token: str | None = None,
    arch_constraints_per_pr: dict[int, list[str]] | None = None,
    seed_files: dict[str, str] | None = None,
) -> dict[int, str]:
    """Rewrite all changed files for each PR and produce surgical patches.

    Args:
        plan:          output of plan_prs()
        changed_files: {repo_relative_path: (base_content, patched_content)}
        repo:          owner/name slug
        repo_config:   loaded repo_config.yaml dict
        model:         LiteLLM model name
        max_retries:   max judge-feedback iterations per file
        pr_indices:    if set, only rewrite PRs with these indices (others skipped)
        upstream_repo: owner/name for fetching related upstream files during rewrite
        token:         GitHub token for upstream fetches
        arch_constraints_per_pr: {pr_index: [constraint_description, ...]} injected into prompts

    Returns:
        {pr_index: unified_diff_string}
    """
    pr_diffs: dict[int, list[str]] = {}
    # Track the accumulated rewritten content per file so stacked PRs use the
    # previous PR's output as their base rather than the original upstream file.
    accumulated: dict[str, str] = {}  # file_path -> latest rewritten content

    # Build a unified seed-content index. Seed paths often carry repo prefixes
    # (e.g. "vllm/csrc/foo.cu") while the planner uses upstream-relative paths
    # ("csrc/foo.cu"). Index by exact path, normalized path (suffix matching a
    # path in any pr_spec["affected_files"]), and basename so lookups via any
    # of these spellings resolve to the same content. Applies to any extension
    # — no data-format-specific branches.
    _all_affected_paths: set[str] = set()
    for _ps in plan.get("pr_series", []):
        for _f in _ps.get("affected_files", []):
            _all_affected_paths.add(_f)
    _affected_segments: set[str] = set()
    for _ap in _all_affected_paths:
        for _seg in _ap.split("/"):
            if _seg:
                _affected_segments.add(_seg)
    _seed_index: dict[str, str] = {}
    if seed_files:
        for _sp, _sc in seed_files.items():
            _seed_index.setdefault(_sp, _sc)
            _parts = _sp.split("/")
            for _i, _seg in enumerate(_parts):
                if _seg in _affected_segments:
                    _norm = "/".join(_parts[_i:])
                    _seed_index.setdefault(_norm, _sc)
                    break
            _base_name = _parts[-1]
            _seed_index.setdefault(_base_name, _sc)

    def _seed_lookup(file_path: str) -> str:
        if file_path in _seed_index:
            return _seed_index[file_path]
        _bn = file_path.rsplit("/", 1)[-1]
        return _seed_index.get(_bn, "")

    for pr_spec in plan.get("pr_series", []):
        idx = pr_spec["index"]
        if pr_indices is not None and idx not in pr_indices:
            continue
        affected = pr_spec.get("affected_files", list(changed_files.keys()))
        patch_parts: list[str] = []

        for file_path in affected:
            if file_path not in changed_files:
                # File is in the plan but not in the seed patch. Try to fetch it
                # from upstream so the rewriter can implement the objective from scratch.
                if upstream_repo:
                    _upstream_text = _fetch_upstream_file_content(file_path, upstream_repo, token)
                    if _upstream_text is not None:
                        logger.info(
                            "PR %d: %s not in seed patch — fetched from upstream (%d chars); "
                            "rewriter will implement objective from scratch",
                            idx, file_path, len(_upstream_text),
                        )
                        # Insert as a changed_files entry so the rest of the loop works normally.
                        # base = upstream content, patched = "" (model must derive from in_scope)
                        changed_files = dict(changed_files)  # avoid mutating caller's dict
                        changed_files[file_path] = (_upstream_text, "")
                    else:
                        logger.warning(
                            "PR %d: affected file %s not in changed_files and not found upstream — skipping",
                            idx, file_path,
                        )
                        continue
                else:
                    logger.warning("PR %d: affected file %s not in changed_files dict", idx, file_path)
                    continue

            orig_base, patched_content = changed_files[file_path]
            # For stacked PRs, use the accumulated rewrite from the previous PR
            # as the base so each patch applies cleanly on top of the previous one.
            base_content = accumulated.get(file_path, orig_base)

            # New files (base is empty) — include verbatim if in this PR's scope.
            # If patched_content is also empty (data artifact not in file_edits),
            # fall back to seed_files content so new data-artifact files are created.
            if not orig_base.strip():
                _new_file_content = patched_content or _seed_lookup(file_path)
                if _new_file_content:
                    patch_parts.append(generate_unified_patch("", _new_file_content, file_path))
                    accumulated[file_path] = _new_file_content
                else:
                    logger.warning("PR %d: new file %s has no content — skipping", idx, file_path)
                continue

            logger.info("Rewriting %s for PR %d: %s", file_path, idx, pr_spec["title"])
            try:
                from mcp_server import _emit_milestone
                _emit_milestone("rewrite_file", {
                    "pr_index": idx, "file": file_path,
                    "n_files": len(affected), "n_prs": len(plan.get("pr_series", [])),
                    "has_feedback": bool((critic_feedback or {}).get(idx) or (audit_feedback or {}).get(idx)),
                })
            except Exception:
                pass
            pr_audit_hints = (audit_feedback or {}).get(idx)
            pr_critic_hints = (critic_feedback or {}).get(idx)
            pr_arch_constraints = (arch_constraints_per_pr or {}).get(idx)

            # Build a plain-text summary of what ancestor PRs changed in this file
            # so the rewriter doesn't accidentally revert those changes.
            _ancestor_summary_parts: list[str] = []
            for _anc_idx in sorted(pr_diffs.keys()):
                if _anc_idx < idx:
                    _anc_diff = "\n".join(pr_diffs[_anc_idx])
                    # Collect only +/- lines whose hunk header matches this file.
                    _file_lines = []
                    _in_target_hunk = False
                    for ln in _anc_diff.splitlines():
                        if ln.startswith("+++ "):
                            _in_target_hunk = file_path in ln
                            continue
                        if ln.startswith("--- "):
                            continue
                        if _in_target_hunk and ln.startswith(("+", "-")):
                            _file_lines.append(ln)
                    if _file_lines:
                        _anc_spec = next(
                            (ps for ps in plan.get("pr_series", []) if ps.get("index") == _anc_idx),
                            {},
                        )
                        _anc_title = _anc_spec.get("title", f"PR {_anc_idx}")
                        _added_syms = [
                            ln[1:].strip() for ln in _file_lines
                            if ln.startswith("+") and ("def " in ln or "class " in ln)
                        ]
                        _summary_line = f"PR {_anc_idx} ({_anc_title})"
                        if _added_syms:
                            _summary_line += " added: " + ", ".join(_added_syms[:8])
                        _ancestor_summary_parts.append(_summary_line)
            _prior_pr_summary = "\n".join(_ancestor_summary_parts)

            rewritten = rewrite_file_for_pr(
                file_path, base_content, patched_content,
                pr_spec, repo, repo_config,
                model=model, max_retries=max_retries,
                audit_hints=pr_audit_hints,
                critic_hints=pr_critic_hints,
                arch_constraints=pr_arch_constraints,
                upstream_repo=upstream_repo,
                token=token,
                seed_content=_seed_lookup(file_path),
                prior_pr_summary=_prior_pr_summary,
            )

            patch = generate_unified_patch(base_content, rewritten, file_path)
            if patch:
                patch_parts.append(patch)
                accumulated[file_path] = rewritten
            else:
                logger.info("  No changes for %s in PR %d", file_path, idx)
                accumulated[file_path] = base_content

            # Surface unresolved scope-creep as arch_constraints so the composite
            # loop can inject them into the next iteration's rewrite prompt.
            _creep = _unresolved_scope_creep.pop((file_path, idx), None)
            if _creep:
                if arch_constraints_per_pr is None:
                    arch_constraints_per_pr = {}
                _sym_list = ", ".join(f"`{s}`" for s in _creep)
                _constraint = (
                    f"CRITICAL — previous rewrite attempt removed {_sym_list} from "
                    f"`{file_path}`. These symbols MUST appear unchanged in the output. "
                    "Do NOT remove any symbol that is not explicitly listed in 'WHAT TO CHANGE'."
                )
                arch_constraints_per_pr.setdefault(idx, []).append(_constraint)
                logger.warning(
                    "  PR %d: injecting scope-creep constraint for next iter: %s", idx, _sym_list
                )

        pr_diffs[idx] = patch_parts

    return {idx: "\n".join(parts) for idx, parts in pr_diffs.items() if parts}


# ---------------------------------------------------------------------------
# Post-rewrite critic
# ---------------------------------------------------------------------------

def _apply_diff_to_content(base: str, diff: str, file_path: str) -> str | None:
    """Apply a unified diff to base content in a temp dir. Returns new content or None."""
    import tempfile, subprocess, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / file_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(base)
        pf = pathlib.Path(tmp) / "patch.diff"
        pf.write_text(diff)
        r = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(pf)],
            cwd=tmp, capture_output=True, text=True,
        )
        return p.read_text() if r.returncode == 0 and p.exists() else None


def critic_pr_series(
    pr_diffs: dict[int, str],
    pr_plan: dict,
    changed_files: dict[str, tuple[str, str]],
    *,
    upstream_repo: str | None = None,
    token: str | None = None,
) -> dict[int, list[str]]:
    """Post-rewrite critic. Returns {pr_index: [issue_string]}. Empty list = clean.

    Checks per PR:
    - Non-empty diff (objective was applied)
    - CSV: header columns match original; row count must not drop; no prose on line 1
    - Python: no public symbols dropped; no syntax errors in rewritten content
    - Stacking: symbols added in PR N must still exist after PR N+1 applies
    - DSPy ReAct code review: fetches upstream source to verify shapes/types/logic
    """
    import re, ast

    def _syms(src):
        try:
            return {n.name for n in ast.walk(ast.parse(src))
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        except SyntaxError:
            return None  # None = parse failure, distinct from empty set

    def _csv_hdr(t):
        for ln in t.splitlines():
            if ln.strip():
                return [c.strip() for c in ln.split(",")]
        return []

    def _csv_rows(t):
        return sum(1 for ln in t.splitlines() if ln.strip() and "," in ln)

    issues: dict[int, list[str]] = {}
    accumulated: dict[str, str] = {p: base for p, (base, _) in changed_files.items()}
    added_syms: dict[int, dict[str, set[str]]] = {}

    for spec in pr_plan.get("pr_series", []):
        idx = spec["index"]
        issues.setdefault(idx, [])
        diff = pr_diffs.get(idx, "")

        if not diff.strip():
            issues[idx].append("PR diff is empty — objective may not have been applied")
            continue

        affected = list(dict.fromkeys(
            p for p in re.findall(r'^(?:\+\+\+|---) [ab]/(.+)$', diff, re.MULTILINE)
            if not p.startswith("/dev/null")
        ))

        for fp in affected:
            base_content, _ = changed_files.get(fp, ("", ""))
            new_content = _apply_diff_to_content(accumulated.get(fp, base_content), diff, fp)
            if new_content is None:
                continue

            # CSV checks
            if fp.endswith(".csv"):
                oh, nh = _csv_hdr(base_content), _csv_hdr(new_content)
                if oh and nh != oh:
                    issues[idx].append(
                        f"{fp}: CSV header changed ({len(oh)} cols → {len(nh)} cols)"
                    )
                or_, nr = _csv_rows(base_content), _csv_rows(new_content)
                if or_ > 0 and nr < or_:
                    issues[idx].append(f"{fp}: CSV rows dropped ({or_} → {nr})")
                first = next((ln for ln in new_content.splitlines() if ln.strip()), "")
                if re.match(r'^[a-z][a-z]+ ', first):
                    issues[idx].append(f"{fp}: CSV content looks like prose on line 1")

            # Python symbol checks
            if fp.endswith(".py") and base_content:
                bs, ns = _syms(base_content), _syms(new_content)
                if ns is None:
                    issues[idx].append(f"{fp}: syntax error in rewritten content")
                elif bs is not None:
                    dropped = bs - ns
                    # Don't flag symbols the plan explicitly says to remove (scope_creep
                    # drop list) — those are intentional, not bugs.
                    _plan_drops: set[str] = set()
                    for _drop_group in spec.get("scope_creep", {}).get("drop", []):
                        # scope_creep drop entries are free-form strings; extract any
                        # symbol names that appear in the dropped set.
                        for _sym in list(dropped):
                            if _sym in _drop_group:
                                _plan_drops.add(_sym)
                    dropped -= _plan_drops
                    if dropped:
                        issues[idx].append(
                            f"{fp}: public symbols dropped: {sorted(dropped)}"
                        )
                    added_syms.setdefault(idx, {})[fp] = ns - bs

            accumulated[fp] = new_content

        # Stacking check: symbols added by earlier PRs must survive this PR
        for prev_idx, prev_files in added_syms.items():
            if prev_idx >= idx:
                continue
            for fp, syms in prev_files.items():
                curr = accumulated.get(fp, "")
                if not curr or not syms:
                    continue
                cs = _syms(curr)
                if cs is not None:
                    reverted = syms - cs
                    if reverted:
                        issues[idx].append(
                            f"{fp}: symbols from PR {prev_idx} reverted: {sorted(reverted)}"
                        )

    # DSPy ReAct code-review pass: catches logic bugs structural checks miss
    # (wrong array dimensions, inconsistent types, index errors, etc.)
    # Fetches upstream source files to verify shapes/signatures before reporting bugs.
    _llm_issues = _llm_code_review(
        pr_diffs, pr_plan, upstream_repo=upstream_repo, token=token,
    )
    for idx, llm_iss in _llm_issues.items():
        issues.setdefault(idx, []).extend(llm_iss)

    return issues


def _llm_code_review(
    pr_diffs: dict[int, str],
    pr_plan: dict,
    model: str = "claude-sonnet-4-6",
    upstream_repo: str | None = None,
    token: str | None = None,
) -> dict[int, list[str]]:
    """DSPy ReAct code review. Returns {pr_index: [issue]} for real correctness bugs.

    The agent fetches upstream source files to verify tensor shapes, function signatures,
    and calling conventions — enabling it to catch bugs like wrong dim() checks or size()
    indexing that are invisible from the diff alone.

    For stacked PRs, ancestor diffs are passed as explicit stacking_context so the agent
    does not flag mismatches that are already resolved by an earlier PR in the series.

    Falls back to a single-shot LLM call if dspy is unavailable.
    """
    try:
        import dspy
    except ImportError:
        return _llm_code_review_static(pr_diffs, pr_plan, model)

    import base64
    import os
    import httpx as _httpx

    gateway = os.environ.get("LITELLM_GATEWAY_URL", "http://localhost:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    dspy_model = model if model.startswith("openai/") else f"openai/{model}"
    _lm = dspy.LM(
        dspy_model,
        api_base=f"{gateway}/",
        api_key=key or "dummy",
        cache=False,
    )

    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _headers = {"Authorization": f"token {_token}", "Accept": "application/vnd.github.v3+json"}
    _fetched: dict[str, str] = {}

    def _gh_get(url: str) -> dict | list | None:
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("GitHub API error %s: %s", url, exc)
            return None

    def fetch_upstream_file(path: str) -> str:
        """Fetch a slice of an upstream file to verify shapes, signatures, and conventions.
        For large files call iteratively: first call shows [Lines 1–N of TOTAL]; continue
        with start_line=200, 400, … to read further sections.
        Args:
            path: file path relative to repo root (e.g. 'csrc/cache.h')
            start_line: 0-indexed line to start from (default 0)
            num_lines: lines to return per call (default 200)
        Returns [Lines X–Y of Z] header + content, or an error message."""
        if not upstream_repo:
            return "(upstream_repo not available — cannot fetch)"
        cache_key = path
        if cache_key not in _fetched:
            owner, repo_name = upstream_repo.split("/", 1)
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
            data = _gh_get(url)
            if not data or not isinstance(data, dict) or data.get("encoding") != "base64":
                return f"(file not found: {path})"
            try:
                text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                _fetched[cache_key] = text
            except Exception as exc:
                return f"(decode error: {exc})"
        lines = _fetched[cache_key].splitlines()
        total = len(lines)
        chunk = lines[start_line:start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(chunk)
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def search_upstream_symbol(symbol: str) -> str:
        """Search the upstream repo for a function, struct, or type definition.
        Use this to find how an existing tensor is allocated (shape, dtype, strides)
        or how a called function is defined. Returns matching file paths.
        Args:
            symbol: exact symbol name (e.g. 'concat_and_cache_mla_rope_fused', 'k_pe_out')"""
        if not upstream_repo:
            return "(upstream_repo not available — cannot search)"
        owner, repo_name = upstream_repo.split("/", 1)
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=5"
        data = _gh_get(url)
        if not data or not isinstance(data, dict):
            return f"(search failed for: {symbol})"
        items = data.get("items", [])
        if not items:
            return f"(not found upstream: {symbol})"
        return "\n".join(f"  {item['path']}" for item in items[:5])

    def fetch_patched_file(path: str, start_line: int = 0, num_lines: int = 300) -> str:
        """Return the full post-patch content of a changed file (upstream + this PR's diff applied).

        Unlike fetch_upstream_file (pre-patch), this shows the file *as it will look after
        the PR lands* — use it to see new code in its full surrounding context and spot
        inconsistencies with nearby code (e.g., a missing cast that already exists elsewhere
        in the same file, a wrong return type, an unconventional pattern).

        Falls back to fetch_upstream_file if the path is not in this PR's diff.
        Args:
            path: file path (e.g. 'python/sglang/srt/layers/quantization/fp8_kernel.py')
            start_line: 0-indexed line to start from (default 0)
            num_lines: lines to return per call (default 300)
        """
        import re as _re

        # Resolve the diff text for this file
        key = path.strip()
        if key not in _diff_by_file_p3:
            for candidate in _diff_by_file_p3:
                if candidate.endswith(key) or key.endswith(candidate):
                    key = candidate
                    break

        file_diff = _diff_by_file_p3.get(key, "")
        if not file_diff:
            # Not in this PR's diff — fall back to upstream fetch
            return fetch_upstream_file(path)

        # Fetch the upstream (pre-patch) file
        upstream_text = fetch_upstream_file(path)
        if upstream_text.startswith("("):
            # File not found upstream — it's a new file; reconstruct from diff additions
            added_lines: list[str] = []
            for line in file_diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])
            patched = "\n".join(added_lines)
        else:
            # Strip the [Lines X–Y of Z] header that fetch_upstream_file prepends
            raw_upstream = _fetched.get(path) or _fetched.get(key, "")
            if not raw_upstream:
                # fetch_upstream_file already populated _fetched; retry lookup
                for k, v in _fetched.items():
                    if k.endswith(key) or key.endswith(k):
                        raw_upstream = v
                        break

            if not raw_upstream:
                return upstream_text  # fall back gracefully

            # Apply unified diff hunks to the upstream content in-memory
            src_lines = raw_upstream.splitlines()
            out_lines: list[str] = []
            src_idx = 0  # 0-indexed pointer into src_lines

            for hunk_match in _re.finditer(
                r"^@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@[^\n]*\n(.*?)(?=^@@|\Z)",
                file_diff,
                _re.MULTILINE | _re.DOTALL,
            ):
                old_start = int(hunk_match.group(1)) - 1  # convert to 0-indexed
                hunk_body = hunk_match.group(3)

                # Copy unchanged lines before this hunk
                out_lines.extend(src_lines[src_idx:old_start])
                src_idx = old_start

                for hunk_line in hunk_body.splitlines():
                    if hunk_line.startswith("-"):
                        src_idx += 1  # consume from source, skip in output
                    elif hunk_line.startswith("+"):
                        out_lines.append(hunk_line[1:])
                    else:
                        # Context line — advance source and keep
                        if src_idx < len(src_lines):
                            out_lines.append(src_lines[src_idx])
                        src_idx += 1

            # Copy any trailing lines after the last hunk
            out_lines.extend(src_lines[src_idx:])
            patched = "\n".join(out_lines)

        lines = patched.splitlines()
        total = len(lines)
        chunk = lines[start_line: start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — patched file has {total} lines)"
        return f"[Post-patch lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n" + "\n".join(chunk)

    class CodeReviewSignature(dspy.Signature):
        """You are a senior engineer doing an adversarial code review of a pull request diff.
        Your sole job is to find REAL correctness bugs that would cause a crash or wrong result.
        Do NOT report style issues, naming, or missing tests.

        Types of bugs to look for:
        - Wrong tensor dimensions: diff says dim()==3 but the tensor is 2D
        - Wrong size() index: diff accesses size(2) but the tensor has only 2 dims (0,1)
        - Shape/type mismatches between a newly allocated buffer and how it is stored/used
        - Off-by-one errors in stride or index calculations
        - TORCH_CHECK conditions that are wrong (e.g. check for wrong shape/dtype)
        - Inconsistency between variable name comments and what the code actually does
        - Logic bugs that produce silently wrong results (not just crashes)

        IMPORTANT: If stacking_context is provided, this PR stacks on top of those ancestor PRs.
        Changes shown in stacking_context will be present in the repo before this PR lands.
        Do NOT flag mismatches that are fully explained by an ancestor diff in stacking_context.

        Strategy:
        1. Read stacking_context first (if provided) to understand what ancestor PRs change.
        2. Call read_diff_section for EACH file listed in diff_file_list to get its full hunks.
           Do not skip files — a bug may be in any changed file.
        3. For each file's diff: identify every new tensor allocation, TORCH_CHECK, size()
           call, dim() comparison, type cast, or arithmetic expression added or modified.
        4. For each changed file, call fetch_patched_file to read the full post-patch content
           in context. Look for inconsistencies with nearby code: missing casts that already
           exist in the same file, wrong return types relative to call sites, unconventional
           patterns compared to identical operations elsewhere in the file.
        5. For each suspicious check, use fetch_upstream_file to retrieve pre-patch definitions
           of tensors, structs, or callers — to verify shape/dtype/type expectations.
           Note: fetch_upstream_file returns pre-patch state; account for ancestor diffs.
        6. Use search_upstream_symbol to trace a type or shape through the codebase.
        7. After gathering context, reason step by step about whether each check is correct.
        8. Report ONLY bugs you are confident about after checking upstream source AND
           accounting for ancestor PR changes.

        Return a JSON object: {"issues": ["<concise bug description with file:line if possible>", ...]}
        Return {"issues": []} if the diff looks correct after checking upstream.
        Return ONLY the JSON — no markdown, no prose."""

        pr_objective: str = dspy.InputField(desc="PR title and objective")
        diff_file_list: str = dspy.InputField(desc="Files changed in this PR (call read_diff_section to fetch each file's hunks)")
        stacking_context: str = dspy.InputField(
            desc="Diffs from ancestor PRs that land before this one (empty if none). "
                 "Do not flag mismatches that are fully explained by these ancestor changes."
        )
        result: str = dspy.OutputField(desc='JSON: {"issues": [...]}')

    results: dict[int, list[str]] = {}

    for spec in pr_plan.get("pr_series", []):
        idx = spec["index"]
        diff = pr_diffs.get(idx, "")
        if not diff.strip():
            continue

        # Index this PR's diff by file so the agent can fetch each file on demand.
        _diff_by_file_p3: dict[str, str] = {}
        _cur_file_p3: str | None = None
        _cur_lines_p3: list[str] = []

        def _flush_p3() -> None:
            if _cur_file_p3 and _cur_lines_p3:
                _diff_by_file_p3[_cur_file_p3] = "".join(_cur_lines_p3)

        for _ln in diff.splitlines(keepends=True):
            if _ln.startswith("+++ b/"):
                _flush_p3()
                _cur_file_p3 = _ln[6:].rstrip("\n")
                _cur_lines_p3 = [_ln]
            elif _cur_file_p3 is not None:
                _cur_lines_p3.append(_ln)
        _flush_p3()
        del _flush_p3, _cur_file_p3, _cur_lines_p3, _ln

        def read_diff_section(file_path: str) -> str:
            """Return all diff hunks for a specific file in this PR's diff.
            Call this for each file you want to inspect — the diff may be large.
            Args:
                file_path: path as it appears in the diff header, e.g. 'vllm/model_executor/models/qwen2_moe.py'
            """
            key = file_path.strip()
            if key not in _diff_by_file_p3:
                for candidate in _diff_by_file_p3:
                    if candidate.endswith(key) or key.endswith(candidate):
                        key = candidate
                        break
            content = _diff_by_file_p3.get(key, "")
            if not content:
                available = ", ".join(sorted(_diff_by_file_p3)) or "(none)"
                return f"(file not found: {file_path!r}. Files in diff: {available})"
            return content[:12000]

        _diff_file_list_p3 = "\n".join(f"  - {f}" for f in sorted(_diff_by_file_p3))

        # Build stacking context: diffs from all ancestor PRs (lower index) that land first.
        # The agent uses this to avoid flagging mismatches already resolved by ancestor changes.
        ancestor_parts: list[str] = []
        for anc_idx in sorted(pr_diffs.keys()):
            if anc_idx < idx:
                anc_diff = pr_diffs[anc_idx]
                if anc_diff.strip():
                    ancestor_parts.append(
                        f"--- Ancestor PR {anc_idx} diff (lands before this PR) ---\n"
                        + anc_diff[:3000]
                    )
        stacking_context = "\n\n".join(ancestor_parts)

        objective = (spec.get("title", "") + "\n" + spec.get("description", ""))[:600]
        try:
            agent = dspy.ReAct(
                CodeReviewSignature,
                tools=[fetch_upstream_file, search_upstream_symbol, read_diff_section, fetch_patched_file],
                max_iters=len(_diff_by_file_p3) + 10,
            )
            with dspy.context(lm=_lm):
                prediction = agent(
                    pr_objective=objective,
                    diff_file_list=_diff_file_list_p3,
                    stacking_context=stacking_context,
                )
            from pipeline.llm import parse_json
            parsed = parse_json(prediction.result)
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
            if issues:
                results[idx] = [f"[code-review] {iss}" for iss in issues]
            logger.info(
                "ReAct code review PR %d: %d issue(s), fetched %d upstream files, %d ancestor PRs in context",
                idx, len(issues), len(_fetched), len(ancestor_parts),
            )
            _fetched.clear()  # reset per-PR so fetch budget applies per PR
        except Exception as exc:
            logger.warning("ReAct code review failed for PR %d (%s) — falling back", idx, exc)
            fallback = _llm_code_review_static({idx: diff}, {"pr_series": [spec]}, model)
            results.update(fallback)

    return results


# ---------------------------------------------------------------------------
# Phase 3b: Cross-PR data artifacts review
# ---------------------------------------------------------------------------

def _data_artifacts_review(
    pr_diffs: dict[int, str],
    pr_plan: dict,
    model: str = "claude-sonnet-4-6",
    upstream_repo: str | None = None,
    token: str | None = None,
) -> dict[int, list[str]]:
    """Cross-PR data artifact review using DSPy RLM.

    Inspects data artifact files (configs, tuning tables, lookup data, etc.) added or
    modified across all PRs in the series together. Passes base + patched content as
    Python objects directly into the REPL so the agent can load them with pandas, json,
    yaml, etc. and compare data programmatically — not as text.

    Returns {pr_index: [issue_string]}. Empty list = clean.
    Falls back to empty dict if dspy is unavailable.
    """
    try:
        import dspy
    except ImportError:
        return {}

    import re as _re
    import os
    import base64
    import httpx as _httpx
    from pipeline.llm import _make_dspy_lm, parse_json

    _lm = _make_dspy_lm(model)

    _token = token or os.environ.get("GITHUB_TOKEN", "")
    _gh_headers = {"Authorization": f"token {_token}", "Accept": "application/vnd.github.v3+json"}
    _fetched_upstream: dict[str, str] = {}

    def _fetch_upstream_text(path: str) -> str:
        if path in _fetched_upstream:
            return _fetched_upstream[path]
        if not upstream_repo:
            return ""
        try:
            _owner, _rname = upstream_repo.split("/", 1)
            url = f"https://api.github.com/repos/{_owner}/{_rname}/contents/{path}"
            r = _httpx.get(url, headers=_gh_headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("encoding") == "base64":
                text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                _fetched_upstream[path] = text
                return text
        except Exception:
            pass
        _fetched_upstream[path] = ""
        return ""

    def _apply_diff(base: str, diff: str, path: str) -> str:
        """Apply unified diff hunks for a single file to base content. Returns patched text."""
        src_lines = base.splitlines()
        out_lines: list[str] = []
        src_idx = 0
        for hunk in _re.finditer(
            r"^@@\s*-(\d+)(?:,\d+)?\s*\+(\d+)(?:,\d+)?\s*@@[^\n]*\n(.*?)(?=^@@|\Z)",
            diff, _re.MULTILINE | _re.DOTALL,
        ):
            old_start = int(hunk.group(1)) - 1
            out_lines.extend(src_lines[src_idx:old_start])
            src_idx = old_start
            for ln in hunk.group(3).splitlines():
                if ln.startswith("-"):
                    src_idx += 1
                elif ln.startswith("+"):
                    out_lines.append(ln[1:])
                else:
                    if src_idx < len(src_lines):
                        out_lines.append(src_lines[src_idx])
                    src_idx += 1
        out_lines.extend(src_lines[src_idx:])
        return "\n".join(out_lines)

    # Determine which file extensions/paths look like data artifacts.
    _DATA_SUFFIXES = (".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
                      ".parquet", ".arrow", ".feather", ".npy", ".npz")
    _DATA_PATH_HINTS = ("config", "tuning", "lookup", "param", "profile", "table",
                        "fixture", "benchmark", "weight", "calibr")

    def _is_data_artifact(path: str) -> bool:
        p = path.lower()
        return any(p.endswith(s) for s in _DATA_SUFFIXES) or any(h in p for h in _DATA_PATH_HINTS)

    # Build the cross-PR artifact data structure passed directly into the REPL:
    #   artifact_data: dict[int, dict[str, {"base": str, "patched": str, "path": str}]]
    #     pr_index → {file_path → {"base": upstream_text, "patched": post-patch_text, "path": file_path}}
    # The REPL agent loads these with io.StringIO + pandas/json/yaml to compare data as data.
    _artifact_data: dict[int, dict[str, dict]] = {}

    for _pr_idx, _diff in pr_diffs.items():
        if not _diff.strip():
            continue

        # Index diff text by file path.
        _cur_file: str | None = None
        _cur_lines: list[str] = []
        _file_diffs: dict[str, str] = {}

        def _flush_fd() -> None:
            if _cur_file and _cur_lines:
                _file_diffs[_cur_file] = "".join(_cur_lines)

        for _ln in _diff.splitlines(keepends=True):
            if _ln.startswith("+++ b/"):
                _flush_fd()
                _cur_file = _ln[6:].rstrip("\n")
                _cur_lines = [_ln]
            elif _cur_file is not None:
                _cur_lines.append(_ln)
        _flush_fd()
        del _flush_fd, _cur_file, _cur_lines, _ln

        _file_map: dict[str, dict] = {}
        for _fp, _fdiff in _file_diffs.items():
            if not _is_data_artifact(_fp):
                continue
            _base = _fetch_upstream_text(_fp)
            _patched = _apply_diff(_base, _fdiff, _fp)
            _file_map[_fp] = {"path": _fp, "base": _base, "patched": _patched}

        if _file_map:
            _artifact_data[_pr_idx] = _file_map

    if not _artifact_data:
        return {}

    def fetch_upstream_artifact(path: str) -> str:
        """Fetch additional context from an upstream data artifact file not already in artifact_data.
        Use this if you need to cross-reference a file not touched by any PR.
        Args:
            path: file path relative to repo root
        Returns file content (up to 8000 chars) or an error string.
        """
        text = _fetch_upstream_text(path)
        return text[:8000] if text else f"(not found upstream: {path})"

    class DataArtifactsReviewSignature(dspy.Signature):
        """You are reviewing data artifact files added or modified across a pull request series.

        You have direct access to `artifact_data` as a Python variable in the REPL:

          artifact_data: dict[int, dict[str, dict]]
            pr_index → {
              file_path → {
                "path":    str,   # file path relative to repo root
                "base":    str,   # upstream file content BEFORE this PR (empty if new file)
                "patched": str,   # full file content AFTER this PR's diff is applied
              }
            }

        Treat the data AS DATA, not as text. The REPL sandbox has standard libraries available.
        Load files using the appropriate parser for the format:
          - CSV / TSV  →  import io, csv; reader = csv.DictReader(io.StringIO(content))
                       or  import pandas as pd; df = pd.read_csv(io.StringIO(content))
          - JSON       →  import json; data = json.loads(content)
          - YAML       →  import yaml; data = yaml.safe_load(content)
          - TOML       →  import tomllib; data = tomllib.loads(content)  # Python 3.11+
          - Parquet    →  import io, pandas as pd; df = pd.read_parquet(io.BytesIO(content.encode("latin-1")))
                       or  import pyarrow.parquet as pq, io; tbl = pq.read_table(io.BytesIO(...))
          - Feather    →  import pyarrow.feather as ft, io; df = ft.read_feather(io.BytesIO(...))
          - NPY / NPZ  →  import numpy as np, io; arr = np.load(io.BytesIO(content.encode("latin-1")))

        Your job is to find real data quality problems that would cause wrong runtime behavior.
        Focus on cross-PR issues that per-PR review cannot detect:

          - Copy-paste between separate files: two PRs add entries to separate files that are
            supposed to cover different configurations or hardware paths, but the metric/value
            columns are numerically identical row-for-row. The key discriminator column differs
            but all metric columns are the same — indicating the second file was generated by
            copying the first rather than independently measuring/deriving values.

          - Placeholder / zero-filled values: an added block has zero variance in numeric
            metric columns across all rows — every cell is 0.0, 1, or the same constant —
            indicating no real measurement was performed.

          - Cross-PR contradictions: two PRs set conflicting values for the same config key
            (e.g. both claim to be the authoritative value for the same model_dim row).

        One-shot example of a copy-paste bug:
          artifact_data[1]["configs/path_A.csv"]["patched"] parses to a DataFrame:
            dim    heads  mode  latency_us  throughput
            7168   128    fast  12.3        142.1
            3584   64     fast  6.8         139.7
          artifact_data[2]["configs/path_B.csv"]["patched"] parses to a DataFrame:
            dim    heads  mode  latency_us  throughput
            7168   128    slow  12.3        142.1    ← same as path_A row 1
            3584   64     slow  6.8         139.7    ← same as path_A row 2
          Detection: after loading both DataFrames, drop the "mode" discriminator column,
          compute set intersection of remaining rows — 100% overlap.
          path_B.csv covers mode=slow (different kernel dispatch), so latency_us and throughput
          MUST differ from path_A.csv. Identical values = copy-paste, not independent measurement.

        Strategy — use the REPL to analyze the data:
          1. Enumerate artifact_data to discover which PRs touch which files.
             {pr: list(files.keys()) for pr, files in artifact_data.items()}
          2. For each file, load "patched" content using the appropriate parser.
             Extract only the rows that are NEW (i.e. present in "patched" but not in "base"):
             use set difference or DataFrame anti-join on the parsed representation.
          3. Compare new-entry sets across PR pairs. Use numeric comparison (not string),
             e.g. pd.DataFrame.equals(), np.allclose(), or set intersection after rounding.
          4. Identify key-discriminator columns vs metric columns using llm_query:
             llm_query("Given this header: [...], which columns are key/discriminators
             and which are metric/measurement columns?")
          5. Report copy-paste only if metric columns are suspiciously similar across PRs
             after the analysis confirms the files are supposed to cover different cases.
          6. Call fetch_upstream_artifact for upstream context if needed.

        Report only issues you are confident about after loading and comparing the actual data.
        Do NOT flag:
          - Columns that are legitimately constant by design (e.g. a dtype field always "fp16").
          - Files with different schemas or key sets (not structurally comparable).
          - Metric values that could plausibly be identical for a legitimate reason (e.g. two
            files share the same hardware target and the measured latency genuinely agrees) —
            use llm_query to judge whether the context makes identical values plausible.

        SUBMIT a JSON object:
          {"issues": [{"pr_index": N, "description": "concise description"}, ...]}
        SUBMIT {"issues": []} if no problems found.
        """

        artifact_data: dict = dspy.InputField(
            desc=(
                "Cross-PR data artifact contents: {pr_index: {file_path: "
                "{path, base, patched}}}. Load 'patched' with pandas/json/yaml."
            )
        )
        series_context: str = dspy.InputField(
            desc="Title and objective for each PR in the series"
        )
        result: str = dspy.OutputField(desc='JSON: {"issues": [{"pr_index": N, "description": "..."}, ...]}')

    # Build a brief series context (title per PR).
    _series_ctx_parts: list[str] = []
    for spec in pr_plan.get("pr_series", []):
        title = spec.get("title", f"PR {spec['index']}")
        desc = spec.get("description", "")[:200]
        _series_ctx_parts.append(f"PR {spec['index']}: {title} — {desc}" if desc else f"PR {spec['index']}: {title}")
    _series_context = "\n".join(_series_ctx_parts)

    _n_files = sum(len(fm) for fm in _artifact_data.values())
    try:
        rlm = dspy.RLM(
            DataArtifactsReviewSignature,
            max_iterations=_n_files * 4 + 12,
            max_llm_calls=_n_files * 3 + 8,
            max_output_chars=30_000,
            tools=[fetch_upstream_artifact],
        )
        with dspy.context(lm=_lm):
            prediction = rlm(
                artifact_data=_artifact_data,
                series_context=_series_context,
            )
        parsed = parse_json(prediction.result)
        raw_issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
        results: dict[int, list[str]] = {}
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            pr_idx = item.get("pr_index")
            desc = item.get("description", "")
            if pr_idx is not None and desc:
                results.setdefault(int(pr_idx), []).append(f"[data-artifact] {desc}")
        logger.info(
            "Cross-PR data artifacts review: %d issue(s) across %d artifact files in %d PRs",
            sum(len(v) for v in results.values()), _n_files, len(_artifact_data),
        )
        return results
    except Exception as exc:
        logger.warning("Cross-PR data artifacts review failed (%s) — skipping", exc)
        return {}


# ---------------------------------------------------------------------------
# Phase 4: Plan-consistency critic
# ---------------------------------------------------------------------------

def plan_critic_pr_series(
    pr_diffs: dict[int, str],
    pr_plan: dict,
    model: str = "claude-sonnet-4-6",
    upstream_repo: str = "",
    token: str = "",
    deferred_files: set[str] | None = None,
) -> dict[int, list[str]]:
    """Phase 4: verify each PR diff satisfies its plan's explicit constraints.

    Unlike the Phase 3 ReAct critic (which checks correctness vs upstream source),
    this critic checks whether the diff matches what the plan *said to do* — catching
    bugs where the rewriter used the right structure but the wrong values or wrong data
    (e.g., a field value is 0 when the plan says 1, or a symbol was added that the
    plan said to exclude).

    Uses DSPy RLM with read_diff_section + upstream fetch tools so it can verify
    symbol names and values against the real upstream before flagging violations.

    Returns {pr_index: [issue_string]}. Empty list = clean.
    """
    import dspy
    from pipeline.llm import make_client, parse_json, _make_dspy_lm

    _owner, _repo_name = (upstream_repo.split("/", 1) if "/" in upstream_repo else ("", ""))
    _headers: dict = {}
    if token:
        _headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    _fetched_cache: dict[str, str] = {}

    results: dict[int, list[str]] = {}

    for spec in pr_plan.get("pr_series", []):
        idx = spec["index"]
        diff = pr_diffs.get(idx, "")
        if not diff.strip():
            continue

        # Build a focused plan block: title + objective + upstream + new_files + in_scope + out_of_scope
        plan_parts = []
        if spec.get("title"):
            plan_parts.append(f"PR title: {spec['title']}")
        if spec.get("description"):
            plan_parts.append(f"Objective: {spec['description']}")
        _pr_upstream = spec.get("upstream", upstream_repo) or upstream_repo
        plan_parts.append(f"Target upstream repo for this PR: {_pr_upstream}")
        if spec.get("new_files"):
            _nf_lines = []
            for nf in spec["new_files"]:
                if isinstance(nf, dict):
                    _nf_lines.append(f"  - {nf['path']} (intent: {nf.get('intent','?')}) — {nf.get('justification','')}")
                else:
                    _nf_lines.append(f"  - {nf}")
            plan_parts.append("Explicitly planned new files (do NOT flag as scope creep):\n" + "\n".join(_nf_lines))
        if spec.get("objective"):
            plan_parts.append(f"Objective: {spec['objective']}")
        if spec.get("serves_objective"):
            plan_parts.append(f"Serves objective: {spec['serves_objective']}")
        if spec.get("in_scope"):
            in_scope = spec["in_scope"]
            if isinstance(in_scope, list):
                plan_parts.append("In scope (must be present):\n" + "\n".join(f"  - {s}" for s in in_scope))
            else:
                plan_parts.append(f"In scope (must be present):\n{in_scope}")
        if spec.get("out_of_scope"):
            out = spec["out_of_scope"]
            if isinstance(out, list):
                plan_parts.append("Out of scope (must NOT be present):\n" + "\n".join(f"  - {s}" for s in out))
            else:
                plan_parts.append(f"Out of scope (must NOT be present):\n{out}")
        if spec.get("scope_creep"):
            drops = spec["scope_creep"].get("drop", [])
            if drops:
                plan_parts.append("Explicitly excluded (must NOT appear in diff):\n" +
                                  "\n".join(f"  - {d[:200]}" for d in drops[:5]))
        if spec.get("csv_seed_rows"):
            _rows = spec["csv_seed_rows"]
            _sample = "\n".join(_rows[:4]) + ("\n..." if len(_rows) > 4 else "")
            plan_parts.append(
                f"CSV SEED ROWS (authoritative — supersedes any row count in the plan description):\n"
                f"The diff MUST add EXACTLY {len(_rows)} rows to the CSV file(s) — "
                f"these are the only real hardware measurements available from the seed.\n"
                f"First rows:\n{_sample}"
            )
        if spec.get("seed_files"):
            _sf = spec["seed_files"]
            _sf_list = "\n".join(
                f"  - {_fp} ({len(_fc)} chars)" for _fp, _fc in _sf.items()
            )
            plan_parts.append(
                f"SEED FILE CONTENT (pre-extracted verbatim — must appear in diff):\n"
                f"The diff MUST include ALL of the following files with the exact content provided "
                f"in pr_plan[\"pr_series\"][n][\"seed_files\"]. Omitting any file will fail plan-consistency.\n"
                f"Required files:\n{_sf_list}"
            )

        # Inform the critic about legitimately deferred files so it doesn't flag them as omissions.
        if deferred_files:
            _spec_files = set(spec.get("affected_files") or [])
            _pr_deferred = sorted(_spec_files & deferred_files)
            if _pr_deferred:
                plan_parts.append(
                    "DEFERRED FILES (do NOT flag as omissions — these were intentionally deferred "
                    "to another upstream run and will not appear in this diff):\n" +
                    "\n".join(f"  - {f}" for f in _pr_deferred)
                )

        plan_block = "\n\n".join(plan_parts)

        # Index the diff by file path so the agent can read each file's hunks on demand.
        # Key: file path string (e.g. "aiter/configs/tuned_fmoe.csv")
        # Value: the full unified diff text for that file (all hunks)
        _diff_by_file: dict[str, str] = {}
        _current_file: str | None = None
        _current_lines: list[str] = []

        def _flush() -> None:
            if _current_file and _current_lines:
                _diff_by_file[_current_file] = "".join(_current_lines)

        for _line in diff.splitlines(keepends=True):
            if _line.startswith("+++ b/"):
                _flush()
                _current_file = _line[6:].rstrip("\n")
                _current_lines = [_line]
            elif _current_file is not None:
                _current_lines.append(_line)
        _flush()
        del _flush, _current_file, _current_lines, _line

        # List files present in this diff (shown to the agent up front).
        _diff_file_list = "\n".join(f"  - {f}" for f in sorted(_diff_by_file))

        def read_diff_section(file_path: str) -> str:
            """Return all diff hunks for a specific file in this PR.
            Use this to inspect any file the plan mentions — the full diff may be large,
            so call this once per file rather than reading a truncated whole-diff.
            Args:
                file_path: path as it appears in the diff header, e.g. 'aiter/configs/tuned_fmoe.csv'
            """
            # Fuzzy match: accept basename or suffix match.
            key = file_path.strip()
            if key not in _diff_by_file:
                for candidate in _diff_by_file:
                    if candidate.endswith(key) or key.endswith(candidate):
                        key = candidate
                        break
            content = _diff_by_file.get(key, "")
            if not content:
                available = ", ".join(sorted(_diff_by_file)) or "(none)"
                return f"(file not found in diff: {file_path!r}. Files in this diff: {available})"
            return content[:12000]  # cap per-file to avoid single-tool overflow

        def fetch_upstream_file_critic(path: str, start_line: int = 0, num_lines: int = 200) -> str:
            """Fetch a slice of a file from the upstream repo to verify what actually exists there.
            Uses the PR's assigned upstream repo (from the plan's upstream field).
            Use this before flagging a violation about a name, ID, or value — confirm the
            upstream state first. Returns file content with a line-range header."""
            import base64 as _b64
            import httpx as _httpx

            # Use per-PR upstream (set in plan block above as _pr_upstream).
            _critic_owner, _critic_repo = (
                _pr_upstream.split("/", 1) if "/" in _pr_upstream else (_owner, _repo_name)
            )
            if not _critic_owner or not _critic_repo:
                return "(upstream repo not configured — cannot fetch)"
            _cache_key = f"{_critic_owner}/{_critic_repo}/{path}"
            if _cache_key not in _fetched_cache:
                url = f"https://api.github.com/repos/{_critic_owner}/{_critic_repo}/contents/{path}"
                try:
                    r = _httpx.get(url, headers=_headers, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    if not isinstance(data, dict) or data.get("encoding") != "base64":
                        return f"(file not found or not text: {path})"
                    _fetched_cache[_cache_key] = _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception as exc:
                    return f"(fetch error for {path}: {exc})"
            lines = _fetched_cache[_cache_key].splitlines()
            total = len(lines)
            chunk = lines[start_line: start_line + num_lines]
            if not chunk:
                return f"(no content at line {start_line} — file has {total} lines)"
            return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n" + "\n".join(chunk)

        def search_upstream_symbol_critic(symbol: str) -> str:
            """Search for a symbol in the upstream repo to confirm whether it exists and its exact name.
            Use this before flagging a violation about a symbol name in the diff vs the plan."""
            import httpx as _httpx

            if not _owner or not _repo_name:
                return "(upstream repo not configured — cannot search)"
            url = f"https://api.github.com/search/code?q={symbol}+repo:{_owner}/{_repo_name}&per_page=5"
            try:
                r = _httpx.get(url, headers=_headers, timeout=15)
                r.raise_for_status()
                items = r.json().get("items", [])
                if not items:
                    return f"(not found in upstream: {symbol!r})"
                return "\n".join(f"  {item['path']}" for item in items[:5])
            except Exception as exc:
                return f"(search error: {exc})"

        class PlanConsistencySignature(dspy.Signature):
            """You are a plan-consistency reviewer for pull request diffs.

            IMPORTANT: The plan was produced by a fast cursory pass without deep upstream
            knowledge. Symbol names, IDs, file paths, and exact values in the plan may be
            wrong. Before flagging a violation:
              1. Read the relevant diff section with read_diff_section.
              2. If the diff uses a name or value that differs from the plan, fetch the
                 upstream file with fetch_upstream_file_critic or search for the symbol
                 with search_upstream_symbol_critic to confirm which is correct.
              3. Only flag a violation if you can confirm the diff is wrong relative to
                 UPSTREAM STATE — not just relative to what the plan says.

            EXHAUSTIVE ENUMERATION — CRITICAL:
            You MUST check EVERY item listed under "In scope (must be present)" and report
            ALL omissions in a single response. Do not stop after finding the first missing
            item. For each in-scope item: read the relevant file's diff with read_diff_section,
            confirm whether the change is present or absent, and if absent add it to issues.
            Reporting only one omission when multiple exist will cause the rewriter to loop
            indefinitely — it fixes the one you report and you find a different one next time.
            Return ALL confirmed omissions at once.

            UPSTREAM ASSIGNMENT CHECK: The plan specifies a "Target upstream repo for this PR".
            Verify that every file in the diff (read the diff_file_list) is a file that
            belongs to that upstream repo. Use the following path-prefix rules:
              - "python/sglang/" or "sglang/srt/" → sgl-project/sglang
              - "python/vllm/" or "vllm/" → vllm-project/vllm
              - "aiter/" → ROCm/aiter
            If a file's path belongs to a DIFFERENT upstream than the plan's target, flag it as:
            "upstream_mismatch: {file} belongs to {correct_upstream}, not {pr_upstream} — move to correct PR"

            NEW FILES CHECK: The plan lists "Explicitly planned new files" — these are expected
            and must NOT be flagged as scope creep. Before flagging any new file addition,
            check whether it appears in that list. If it does, it is `intent=planned` — skip.
            If the diff introduces a file via `--- /dev/null` (i.e. creates a new file) that is
            NOT listed in "Explicitly planned new files", use fetch_upstream_file_critic to check
            whether that path already exists in the upstream repo. If it does exist upstream,
            flag it as: "scope_creep: creates new file {path} but {path} already exists upstream
            — edit the existing file instead of creating a new one". Only flag this if you
            confirmed via fetch_upstream_file_critic that the file really exists upstream.

            What to check (plan is a hint, upstream is the authority):
            1. Does the diff add things the plan says NOT to add? (genuine scope creep — but
               exclude files declared in "Explicitly planned new files")
            2. Does the diff omit things the plan explicitly requires AND which actually
               belong in this PR based on the upstream structure? CHECK ALL IN-SCOPE ITEMS.
            3. For data files: do field values match the objective (verify against upstream schema)?
            4. Upstream assignment: are all modified files in the correct upstream repo?

            Do NOT flag:
            - Symbol names or IDs that the diff uses correctly per upstream (plan may have
              wrong names)
            - Numbering that follows upstream conventions (plan may have wrong IDs)
            - Structural choices that match upstream idioms even if they differ from the plan
            - Files listed in "Explicitly planned new files" — those are expected

            Return a JSON object: {"issues": ["<confirmed violation>", ...]}
            Return {"issues": []} if the diff correctly implements the objective.
            Return ONLY the JSON — no markdown, no prose."""

            plan: str = dspy.InputField(desc="The plan for this PR (title, objective, in/out of scope) — treat as a hint")
            diff_file_list: str = dspy.InputField(desc="Files present in this PR's diff")
            result: str = dspy.OutputField(desc='JSON {"issues": [...]} — empty list if diff is correct')

        try:
            _lm = _make_dspy_lm(model)
            # Each in-scope item may need read_diff_section + fetch call; give generous headroom
            # so the RLM can check all items exhaustively without hitting the iteration cap.
            _n_inscope = len(spec.get("in_scope") or []) if isinstance(spec.get("in_scope"), list) else 4
            _max_iters = len(_diff_by_file) * 3 + _n_inscope * 2 + 10
            rlm = dspy.RLM(
                PlanConsistencySignature,
                max_iterations=_max_iters,
                max_llm_calls=_max_iters + 4,
                max_output_chars=30_000,
                tools=[read_diff_section, fetch_upstream_file_critic, search_upstream_symbol_critic],
            )
            with dspy.context(lm=_lm):
                prediction = rlm(
                    plan=plan_block[:4000],
                    diff_file_list=_diff_file_list,
                )
            parsed = parse_json(prediction.result)
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
            # Filter out findings that mention a deferred file — these are not omissions.
            if deferred_files and issues:
                issues = [
                    iss for iss in issues
                    if not any(df in iss for df in deferred_files)
                ]
            if issues:
                results[idx] = [f"[plan-consistency] {iss}" for iss in issues]
            logger.info(
                "Plan consistency review PR %d: %d issue(s), %d file(s) in diff",
                idx, len(issues), len(_diff_by_file),
            )
        except Exception as exc:
            logger.warning("Plan consistency review failed for PR %d: %s", idx, exc)

    return results


def _llm_code_review_static(
    pr_diffs: dict[int, str],
    pr_plan: dict,
    model: str = "claude-sonnet-4-6",
) -> dict[int, list[str]]:
    """Single-shot LLM code review fallback (no upstream fetching)."""
    from pipeline.llm import llm_call, make_client, parse_json

    _PROMPT = """\
You are a senior engineer doing a code review of a pull request diff.
Find REAL correctness bugs only — not style, not naming, not missing tests.

Focus on:
- Wrong array dimensions, indices, or sizes (e.g. dim() == 3 but tensor is 2D)
- Type or shape mismatches between a new output tensor and how it is used/indexed
- Off-by-one errors or wrong stride calculations
- Inconsistency between comments/docs and the actual code change
- Missing or incorrect validation checks (TORCH_CHECK, assert, etc.)
- Logic errors that would cause a crash or wrong result at runtime

PR objective: {objective}

Diff:
{diff}

Return a JSON object: {{"issues": ["<concise bug description>", ...]}}
Return {{"issues": []}} if the diff looks correct.
Return ONLY the JSON.
"""
    client = make_client()
    results: dict[int, list[str]] = {}

    for spec in pr_plan.get("pr_series", []):
        idx = spec["index"]
        diff = pr_diffs.get(idx, "")
        if not diff.strip():
            continue
        objective = spec.get("title", "") + "\n" + spec.get("description", "")
        prompt = _PROMPT.format(objective=objective[:400], diff=diff[:8000])
        try:
            raw = llm_call(prompt, model, client=client, max_tokens=1024, json_mode=True)
            parsed = parse_json(raw)
            issues = parsed.get("issues", []) if isinstance(parsed, dict) else []
            if issues:
                results[idx] = [f"[code-review] {iss}" for iss in issues]
                logger.info("Static code review PR %d: %d issue(s)", idx, len(issues))
        except Exception as exc:
            logger.warning("Static code review failed for PR %d: %s", idx, exc)

    return results
