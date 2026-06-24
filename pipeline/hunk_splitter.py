"""
Hunk-level diff splitting.

Takes a unified diff and a PR plan (from pr_plan.plan_prs) and:
  1. Parses the diff into discrete, numbered hunks
  2. Asks an LLM to assign each hunk to a PR index, "drop", or "revert"
  3. Reconstructs one clean per-PR diff per assigned PR index

The LLM works at hunk granularity — not at file or prose-description
granularity — so the output diffs are deterministic and apply cleanly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HUNK_PREVIEW_LINES = 30   # max content lines shown to LLM per hunk
_ASSIGN_BATCH = 40         # hunks per LLM call (keeps prompt manageable)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Hunk:
    hunk_id: int          # global sequential id (1-based across whole diff)
    file: str             # repo-relative path
    hunk_num: int         # per-file hunk number (1-based)
    header: str           # "@@ -old,count +new,count @@ context"
    old_start: int        # first line of old file covered by this hunk
    new_start: int        # first line of new file covered by this hunk
    lines: list[str]      # content lines (not including the @@ header line)
    file_header: str      # diff --git / index / --- / +++ block for this file


@dataclass
class HunkAssignment:
    hunk_id: int
    action: str           # "pr_1", "pr_2", ..., "drop", "revert"
    pr_index: int | None  # None when action is "drop" or "revert"
    reason: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_hunks(diff: str) -> list[Hunk]:
    """Parse a unified diff into a flat list of Hunk objects."""
    hunks: list[Hunk] = []
    hunk_id = 0

    # Split into per-file sections on "diff --git"
    file_sections = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)

    for section in file_sections:
        if not section.strip() or not section.startswith("diff --git "):
            continue

        # Extract file path from "diff --git a/path b/path"
        m = re.match(r"diff --git a/(.+?) b/(.+)", section.splitlines()[0])
        file_path = m.group(2) if m else "unknown"

        # File header = everything up to the first @@ line
        hunk_split = re.split(r"(?=^@@)", section, flags=re.MULTILINE)
        file_header = hunk_split[0].rstrip()
        hunk_blocks = hunk_split[1:]  # each starts with @@

        for hunk_num, block in enumerate(hunk_blocks, 1):
            block_lines = block.splitlines()
            if not block_lines:
                continue

            header_line = block_lines[0]
            content_lines = block_lines[1:]

            # Parse @@ -old_start,count +new_start,count @@ ...
            hm = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", header_line)
            old_start = int(hm.group(1)) if hm else 0
            new_start = int(hm.group(2)) if hm else 0

            hunk_id += 1
            hunks.append(Hunk(
                hunk_id=hunk_id,
                file=file_path,
                hunk_num=hunk_num,
                header=header_line,
                old_start=old_start,
                new_start=new_start,
                lines=content_lines,
                file_header=file_header,
            ))

    return hunks


def format_hunks_for_llm(hunks: list[Hunk]) -> str:
    """Render hunks as a compact numbered list for the assignment prompt."""
    parts = []
    for h in hunks:
        preview = h.lines[:_HUNK_PREVIEW_LINES]
        truncated = len(h.lines) > _HUNK_PREVIEW_LINES
        content = "\n".join(preview)
        if truncated:
            content += f"\n... ({len(h.lines) - _HUNK_PREVIEW_LINES} more lines)"
        parts.append(
            f"HUNK {h.hunk_id}  {h.file}  {h.header}\n{content}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Assignment prompt
# ---------------------------------------------------------------------------

_ASSIGN_PROMPT = """\
You are assigning hunks from a unified diff to pull requests.

## PR PLAN

{plan_section}

## HUNKS TO ASSIGN

Each hunk is prefixed with its ID, file, and @@ header.

{hunks_section}

---

VALID PR INDICES: {valid_pr_indices}
You MUST only use these indices. Do NOT invent new ones (no pr_4 if the plan
only has pr_1, pr_2, pr_3). If a hunk fits a PR that shares scope with
another planned PR, assign it to the closest matching planned index.

For EVERY hunk listed above, assign it to exactly one of:
  - pr_N     (where N is one of the VALID PR INDICES above)
  - drop     (serves no stated objective — exclude from all PRs)
  - revert   (actively regresses existing behaviour — must be reverted)

Use the plan's `in_scope` descriptions as your guide.
If a hunk is a necessary companion to an in-scope change (e.g. a one-line
fix required for an in-scope function to compile), include it in that PR.
If in doubt between "drop" and including in a PR, choose "drop".

Respond with ONLY JSON:

{{
  "assignments": [
    {{
      "hunk_id": <int>,
      "action": "pr_1" | "pr_2" | ... | "drop" | "revert",
      "reason": "one sentence"
    }}
  ]
}}
"""


def _format_plan_for_assignment(plan: dict) -> str:
    """Render the PR plan concisely for the assignment prompt."""
    lines = []
    for pr in plan.get("pr_series", []):
        lines.append(f"PR {pr['index']} [{pr['label']}]: {pr['title']}")
        lines.append(f"  Objective: {pr['objective']}")
        for s in pr.get("in_scope", []):
            lines.append(f"  in_scope: {s}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM assignment
# ---------------------------------------------------------------------------

def assign_hunks(
    plan: dict,
    hunks: list[Hunk],
    *,
    model: str = "claude-opus-4-7",
) -> list[HunkAssignment]:
    """Ask the LLM to assign every hunk to a PR index, drop, or revert.

    Batches hunks in groups of _ASSIGN_BATCH to keep prompts manageable.
    Returns one HunkAssignment per hunk, covering all hunk_ids.
    """
    from pipeline.llm import llm_call, make_client, parse_json

    plan_section = _format_plan_for_assignment(plan)
    client = make_client()
    results: list[HunkAssignment] = []

    # Process in batches
    for batch_start in range(0, len(hunks), _ASSIGN_BATCH):
        batch = hunks[batch_start : batch_start + _ASSIGN_BATCH]
        hunks_section = format_hunks_for_llm(batch)

        valid_indices = sorted(pr["index"] for pr in plan.get("pr_series", []))
        valid_pr_indices = ", ".join(f"pr_{i}" for i in valid_indices)

        prompt = _ASSIGN_PROMPT.format(
            plan_section=plan_section,
            hunks_section=hunks_section,
            valid_pr_indices=valid_pr_indices,
        )

        raw = llm_call(prompt, model, client=client, max_tokens=4096, json_mode=True)
        data = parse_json(raw)

        valid_indices = sorted(pr["index"] for pr in plan.get("pr_series", []))
        max_plan_index = valid_indices[-1] if valid_indices else 0

        for entry in data.get("assignments", []):
            hunk_id = entry.get("hunk_id")
            action = entry.get("action", "drop").lower()
            reason = entry.get("reason", "")

            # Parse "pr_N" → pr_index=N, clamp to plan's max
            pr_index: int | None = None
            m = re.match(r"pr_(\d+)", action)
            if m:
                pr_index = int(m.group(1))
                if pr_index not in valid_indices:
                    # Fold overflow index into the highest planned PR
                    clamped = max(i for i in valid_indices if i <= pr_index) if any(i <= pr_index for i in valid_indices) else valid_indices[-1]
                    reason = f"[clamped from pr_{pr_index} to pr_{clamped}] {reason}"
                    pr_index = clamped
                    action = f"pr_{clamped}"

            results.append(HunkAssignment(
                hunk_id=hunk_id,
                action=action,
                pr_index=pr_index,
                reason=reason,
            ))

    # Any hunk not mentioned by the LLM is treated as drop
    assigned_ids = {a.hunk_id for a in results}
    for h in hunks:
        if h.hunk_id not in assigned_ids:
            results.append(HunkAssignment(
                hunk_id=h.hunk_id,
                action="drop",
                pr_index=None,
                reason="not assigned by LLM — defaulting to drop",
            ))

    results.sort(key=lambda a: a.hunk_id)
    return results


# ---------------------------------------------------------------------------
# Diff reconstruction
# ---------------------------------------------------------------------------

def build_per_pr_diffs(
    hunks: list[Hunk],
    assignments: list[HunkAssignment],
) -> dict[int, str]:
    """Reconstruct one unified diff per PR index from hunk assignments.

    Returns {pr_index: diff_string}. Hunks assigned "drop" or "revert"
    are excluded.

    Note on line numbers: the reconstructed @@ headers retain their original
    old_start/new_start values. When non-contiguous hunks are combined, these
    numbers will be offset from reality, but `git apply --3way` resolves this
    via context matching rather than line numbers.
    """
    id_to_hunk = {h.hunk_id: h for h in hunks}
    # Group assignments by pr_index
    by_pr: dict[int, list[HunkAssignment]] = {}
    for a in assignments:
        if a.pr_index is not None:
            by_pr.setdefault(a.pr_index, []).append(a)

    diffs: dict[int, str] = {}
    for pr_index, pr_assignments in sorted(by_pr.items()):
        pr_assignments.sort(key=lambda a: a.hunk_id)

        # Group by file to emit file headers once
        by_file: dict[str, list[Hunk]] = {}
        for a in pr_assignments:
            h = id_to_hunk.get(a.hunk_id)
            if h:
                by_file.setdefault(h.file, []).append(h)

        parts: list[str] = []
        for file_path, file_hunks in by_file.items():
            file_hunks.sort(key=lambda h: h.hunk_id)
            # File header (diff --git / index / --- / +++ lines)
            parts.append(file_hunks[0].file_header)
            for h in file_hunks:
                parts.append(h.header)
                parts.extend(h.lines)

        diffs[pr_index] = "\n".join(parts) + "\n"

    return diffs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def split_diff_by_plan(
    plan: dict,
    diff: str,
    *,
    model: str = "claude-opus-4-7",
) -> dict:
    """Reconstruct per-PR diffs from plan hunk assignments (no LLM call).

    plan_prs() now emits hunk_assignments alongside the PR series plan,
    so this function is purely deterministic reconstruction.

    Returns:
        hunks:        list of all parsed Hunk objects
        assignments:  list of HunkAssignment (hunk_id → action/pr_index)
        pr_diffs:     {pr_index: diff_string} — one diff per planned PR
        drop_hunks:   hunk_ids marked "drop"
        revert_hunks: hunk_ids marked "revert"
        stats:        summary counts
    """
    # Use hunks already parsed by plan_prs if available (avoids re-parsing)
    hunks = plan.get("_hunks") or parse_hunks(diff)
    if not hunks:
        return {"hunks": [], "assignments": [], "pr_diffs": {}, "stats": {"total": 0}}

    # Convert plan hunk_assignments to HunkAssignment objects
    assignments: list[HunkAssignment] = []
    for entry in plan.get("hunk_assignments", []):
        hunk_id = entry.get("hunk_id", 0)
        action = entry.get("action", "drop").lower()
        reason = entry.get("reason", "")
        pr_index: int | None = None
        m = re.match(r"pr_(\d+)", action)
        if m:
            pr_index = int(m.group(1))
        assignments.append(HunkAssignment(hunk_id=hunk_id, action=action, pr_index=pr_index, reason=reason))

    # Fall back to the old LLM-based assignment if plan has no hunk_assignments
    if not assignments:
        assignments = assign_hunks(plan, hunks, model=model)

    pr_diffs = build_per_pr_diffs(hunks, assignments)

    drop_ids = [a.hunk_id for a in assignments if a.action == "drop"]
    revert_ids = [a.hunk_id for a in assignments if a.action == "revert"]
    pr_counts = {}
    for a in assignments:
        if a.pr_index is not None:
            pr_counts[a.pr_index] = pr_counts.get(a.pr_index, 0) + 1

    return {
        "hunks": hunks,
        "assignments": assignments,
        "pr_diffs": pr_diffs,
        "drop_hunks": drop_ids,
        "revert_hunks": revert_ids,
        "stats": {
            "total": len(hunks),
            "assigned": sum(pr_counts.values()),
            "dropped": len(drop_ids),
            "reverted": len(revert_ids),
            "by_pr": pr_counts,
        },
    }


def format_assignment_summary(result: dict, plan: dict) -> str:
    """Render hunk assignment results as a human-readable summary."""
    stats = result.get("stats", {})
    assignments = result.get("assignments", [])
    id_to_hunk = {h.hunk_id: h for h in result.get("hunks", [])}

    lines = [
        "=" * 70,
        "HUNK ASSIGNMENTS",
        "=" * 70,
        f"Total hunks: {stats.get('total', 0)}  |  "
        f"Assigned: {stats.get('assigned', 0)}  |  "
        f"Dropped: {stats.get('dropped', 0)}  |  "
        f"Reverted: {stats.get('reverted', 0)}",
        "",
    ]

    pr_titles = {pr["index"]: pr["title"] for pr in plan.get("pr_series", [])}

    # Group by action
    by_pr: dict[int, list] = {}
    drops: list = []
    reverts: list = []
    for a in sorted(assignments, key=lambda x: x.hunk_id):
        h = id_to_hunk.get(a.hunk_id)
        loc = f"{h.file}  {h.header[:60]}" if h else f"hunk {a.hunk_id}"
        if a.pr_index is not None:
            by_pr.setdefault(a.pr_index, []).append((a, loc))
        elif a.action == "revert":
            reverts.append((a, loc))
        else:
            drops.append((a, loc))

    for pr_idx, entries in sorted(by_pr.items()):
        title = pr_titles.get(pr_idx, f"PR {pr_idx}")
        lines.append(f"PR {pr_idx} — {title}  ({len(entries)} hunks)")
        for a, loc in entries:
            lines.append(f"  [{a.hunk_id:3}] {loc}")
            if a.reason:
                lines.append(f"        {a.reason}")
        lines.append("")

    if reverts:
        lines.append(f"↩  REVERT  ({len(reverts)} hunks — must not be applied)")
        for a, loc in reverts:
            lines.append(f"  [{a.hunk_id:3}] {loc}")
            if a.reason:
                lines.append(f"        {a.reason}")
        lines.append("")

    if drops:
        lines.append(f"🗑  DROP  ({len(drops)} hunks — excluded from all PRs)")
        for a, loc in drops:
            lines.append(f"  [{a.hunk_id:3}] {loc}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)
