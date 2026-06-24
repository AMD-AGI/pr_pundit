"""
RLM-based PR rewrite pipeline.

Replaces the outer composite loop (rewrite + Phase 1 rules + Phase 3 code review)
with a single DSPy RLM agent that keeps all diff/file content as Python REPL
variables — no prompt injection, no char caps.

Phase 2 (arch audit) and Phase 4 (plan consistency) remain separate steps in
create_pr_from_seed.py and run after this module returns.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def run_rlm_pipeline(
    seed: dict,
    pr_plan: dict,
    resolved_upstream: str,
    repo_config: dict,
    model: str,
    token: str,
    *,
    seed_token: str = "",
    critic_feedback: dict[int, list[str]] | None = None,
    accepted_harnesses: list | None = None,
    seed_files: dict[str, str] | None = None,
    iter_history: list[dict] | None = None,
    already_written_prs: list[dict] | None = None,
    active_pr_indices: list[int] | None = None,
    prior_upstream_context: list[dict] | None = None,
    passing_file_diffs: dict[str, str] | None = None,
    max_iterations: int = 200,
    max_llm_calls: int = 400,
    verbose: bool = False,
) -> tuple[dict[int, str], list[dict]]:
    """Run the DSPy RLM rewrite pipeline.

    Returns (pr_diffs, deferred_files):
      pr_diffs: dict[int, str] — unified diff string per PR index.
      deferred_files: list[dict] — files the RLM deferred to a different upstream.
    active_pr_indices: if set, RLM is instructed to write only these PR indices (multi-upstream mode).
    prior_upstream_context: list of dicts from prior upstream runs to inject as read-only context.
    Raises on any failure; no fallback to the old rewrite_pr_series path.
    """
    import dspy
    from pipeline.llm import _make_dspy_lm
    from pipeline.pr_rewrite import (
        validate_rewrite,
        generate_unified_patch,
    )
    from pipeline.judge import judge_patch

    owner, repo_name = resolved_upstream.split("/", 1)
    repo_slug = resolved_upstream
    _headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    _fetched_cache: dict[str, str] = {}
    _seed_sourced_cache_keys: set[str] = set()  # cache_keys populated from seed content (new files)

    # ── Upstream file fetch ────────────────────────────────────────────────────

    def fetch_upstream_file(path: str, start_line: int = 0, num_lines: int = 200, repo: str = "") -> str:
        """Fetch a slice of a file from an upstream repo.
        For large files, call iteratively with increasing start_line to page through.
        The first call returns a [Lines X–Y of TOTAL] header showing total line count.
        Args:
            path: file path relative to repo root (e.g. "vllm/model_executor/models/qwen2_moe.py")
            start_line: 0-indexed line to start from
            num_lines: number of lines to return per page (default 200; no hard cap — page as needed)
            repo: upstream repo in "owner/name" format. Always set this explicitly — read
                  pr_plan["pr_series"][n]["upstream"] to get the correct repo per PR. Every upstream
                  in the bundle (including submodule repos like ROCm/composable_kernel) is first-class
                  — there is no "primary" upstream. If omitted, defaults to the current run's upstream.
        Returns lines with a header, or an error string."""
        import base64 as _b64
        import httpx as _httpx

        # Use per-call repo override if provided (multi-upstream support).
        if repo and "/" in repo:
            _fetch_owner, _fetch_repo = repo.split("/", 1)
        else:
            _fetch_owner, _fetch_repo = owner, repo_name

        cache_key = f"{_fetch_owner}/{_fetch_repo}/{path}"
        if cache_key not in _fetched_cache:
            url = f"https://api.github.com/repos/{_fetch_owner}/{_fetch_repo}/contents/{path}"
            try:
                r = _httpx.get(url, headers=_headers, timeout=15)
                if r.status_code == 404:
                    # Brand-new file: not yet in the upstream repo (added by the seed PR).
                    # If seed content was pre-extracted, surface it directly so the RLM
                    # can commit it verbatim without needing to synthesise it from scratch.
                    _seed_content = _base_content_index.get("seed:" + path, "")
                    if _seed_content:
                        _fetched_cache[cache_key] = _seed_content
                        _seed_sourced_cache_keys.add(cache_key)
                    else:
                        return f"(file not found: {path} in {_fetch_owner}/{_fetch_repo})"
                else:
                    r.raise_for_status()
                    data = r.json()
                    if not isinstance(data, dict) or data.get("encoding") != "base64":
                        return f"(file not found or not text: {path})"
                    text = _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    _fetched_cache[cache_key] = text
            except Exception as exc:
                return f"(fetch error for {path}: {exc})"
        _is_new_file = cache_key in _seed_sourced_cache_keys
        lines = _fetched_cache[cache_key].splitlines()
        total = len(lines)
        chunk = lines[start_line: start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        body = "\n".join(f"{start_line + i + 1:5d}  {l}" for i, l in enumerate(chunk))
        if _is_new_file:
            header = (
                f"[NEW FILE — seed PR adds this; not yet in {_fetch_owner}/{_fetch_repo}. "
                f"Lines {start_line+1}–{start_line+len(chunk)} of {total}]"
            )
            instruction = (
                "\n\n[INSTRUCTION] This file does not exist upstream yet — it is brand new. "
                "The no-copy rule applies only to files that already exist upstream. "
                "For new files, use this content as-is. "
                "Call write_file_rewrite_tool(pr_index, path, <content>, base_content='')."
                if start_line == 0 else ""
            )
            return f"{header}\n{body}{instruction}"
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{body}"

    def search_upstream_symbol(symbol: str) -> str:
        """Search for a symbol (function, class, variable) in the upstream repo.
        Use to confirm whether a symbol already exists before adding it.
        Returns matching file paths or a not-found message."""
        import httpx as _httpx
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=5"
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            if not items:
                return f"(not found: {symbol})"
            return "\n".join(f"  {item['path']}" for item in items[:5])
        except Exception as exc:
            return f"(search error: {exc})"

    def fetch_symbol_definition(symbol: str, context_lines: int = 10) -> str:
        """Fetch the full definition block of a function or class from the upstream repo.
        Searches for the symbol, fetches the containing file, then extracts just the
        relevant def/class block plus surrounding context. Follows indentation to find
        the end of the block — no manual pagination needed.
        Args:
            symbol: exact function or class name (e.g. "fused_moe_kernel", "MoEConfig")
            context_lines: lines of context to include before the def/class line (default 10)
        Returns: the definition block with file path and line numbers as a header,
                 or an error if not found."""
        import httpx as _httpx
        import re as _re
        import base64 as _b64

        # Search for the file containing this symbol
        url = f"https://api.github.com/search/code?q={symbol}+repo:{owner}/{repo_name}&per_page=8"
        try:
            r = _httpx.get(url, headers=_headers, timeout=15)
            r.raise_for_status()
            items = r.json().get("items", [])
        except Exception as exc:
            return f"(search error for {symbol!r}: {exc})"

        if not items:
            return f"(symbol not found: {symbol!r})"

        # Find the best candidate: prefer .py files with an exact def/class match
        candidate_path = None
        for item in items:
            p = item["path"]
            if p.endswith(".py"):
                candidate_path = p
                break
        if candidate_path is None:
            candidate_path = items[0]["path"]

        # Fetch full file content (use cache)
        if candidate_path not in _fetched_cache:
            furl = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{candidate_path}"
            try:
                fr = _httpx.get(furl, headers=_headers, timeout=15)
                fr.raise_for_status()
                data = fr.json()
                if data.get("encoding") != "base64":
                    return f"(cannot decode {candidate_path})"
                _fetched_cache[candidate_path] = _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception as exc:
                return f"(fetch error for {candidate_path}: {exc})"

        lines = _fetched_cache[candidate_path].splitlines()
        total = len(lines)

        # Find the def/class line
        pattern = _re.compile(rf"^(\s*)(def|class)\s+{_re.escape(symbol)}\b")
        def_lineno = None
        base_indent = None
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                def_lineno = i
                base_indent = len(m.group(1))
                break

        if def_lineno is None:
            # Symbol exists in file but not as a top-level def/class — return surrounding lines
            ctx_pattern = _re.compile(rf"\b{_re.escape(symbol)}\b")
            for i, line in enumerate(lines):
                if ctx_pattern.search(line):
                    start = max(0, i - context_lines)
                    end = min(total, i + context_lines + 1)
                    block = "\n".join(f"{start + j + 1:5d}  {lines[start + j]}" for j in range(end - start))
                    return f"[{candidate_path} — {symbol!r} referenced at line {i + 1} (not a def/class)]\n{block}"
            return f"(symbol {symbol!r} found in search but not located in {candidate_path})"

        # Collect lines until indentation returns to base or EOF
        end_lineno = def_lineno + 1
        for i in range(def_lineno + 1, total):
            line = lines[i]
            stripped = line.lstrip()
            if stripped == "":
                end_lineno = i + 1
                continue
            indent = len(line) - len(stripped)
            if indent <= base_indent and stripped:
                break
            end_lineno = i + 1

        ctx_start = max(0, def_lineno - context_lines)
        block_lines = lines[ctx_start:end_lineno]
        numbered = "\n".join(f"{ctx_start + j + 1:5d}  {bl}" for j, bl in enumerate(block_lines))
        return f"[{candidate_path} lines {ctx_start + 1}–{end_lineno} of {total}]\n{numbered}"

    # ── Seed inspection tools ──────────────────────────────────────────────────

    # Build a name → content index for all seed files so the RLM can page through
    # patches, file_edits, and data_artifacts without us pre-injecting any content.
    # Data artifacts with empty content but a download_url are fetched lazily here
    # using seed_token so the RLM has full CSV/JSON content available.
    _seed_index: dict[str, str] = {}
    for _p in seed.get("patches", []):
        _seed_index[_p["name"]] = _p.get("content", "")
    for _e in seed.get("file_edits", []):
        _seed_index[_e["name"]] = _e.get("content", "")
    _eff_seed_token = seed_token or token
    for _d in seed.get("data_artifacts", []):
        _content = _d.get("content", "")
        if not _content and _d.get("download_url") and _eff_seed_token:
            try:
                import httpx as _httpx_art
                _art_headers = {"Authorization": f"token {_eff_seed_token}"}
                _art_r = _httpx_art.get(_d["download_url"], headers=_art_headers, timeout=30, follow_redirects=True)
                if _art_r.status_code == 200:
                    _content = _art_r.text
            except Exception:
                pass
        _seed_index[_d["name"]] = _content

    def inspect_seed() -> str:
        """Return a structured summary of the seed: README excerpt, file listing with sizes,
        and the type of each file (patch, file_edit, data_artifact).
        Call this first to understand what is in the seed before reading individual files."""
        lines = []
        readme = seed.get("readme") or ""
        if readme:
            lines.append("## README (first 60 lines)")
            lines.append("\n".join(readme.splitlines()[:60]))
            lines.append("")
        lines.append("## Seed files")
        for p in seed.get("patches", []):
            n_lines = len(p.get("content", "").splitlines())
            lines.append(f"  [patch]       {p['name']}  ({n_lines} lines) — use read_seed_file to inspect")
        for e in seed.get("file_edits", []):
            n_lines = len(e.get("content", "").splitlines())
            lines.append(f"  [file_edit]   {e['name']}  ({n_lines} lines) — use read_seed_file to inspect")
        for d in seed.get("data_artifacts", []):
            n_lines = len(d.get("content", "").splitlines())
            lines.append(f"  [data_artifact] {d['name']}  ({n_lines} lines) — use read_seed_file to inspect")
        if not (seed.get("patches") or seed.get("file_edits") or seed.get("data_artifacts")):
            lines.append("  (no files found)")
        return "\n".join(lines)

    def read_seed_file(name: str, start_line: int = 0, num_lines: int = 200) -> str:
        """Page through the content of a seed file (patch, file_edit, or data_artifact).
        For large files, call iteratively with increasing start_line.
        The first call returns a [Lines X–Y of TOTAL] header showing total line count.
        Args:
            name: file name as listed by inspect_seed() (e.g. "fused_moe.py", "tuned_fmoe.csv")
            start_line: 0-indexed line to start from
            num_lines: lines to return per page (default 200; no hard cap — page as needed)
        Returns numbered lines with a header, or an error if name not found."""
        if name not in _seed_index:
            # fuzzy match by suffix
            for candidate in _seed_index:
                if candidate.endswith(name) or name.endswith(candidate):
                    name = candidate
                    break
        content = _seed_index.get(name, "")
        if not content and name not in _seed_index:
            available = ", ".join(sorted(_seed_index)) or "(none)"
            return f"(seed file not found: {name!r}. Available: {available})"
        all_lines = content.splitlines()
        total = len(all_lines)
        chunk = all_lines[start_line: start_line + num_lines]
        if not chunk:
            return f"(no content at line {start_line} — file has {total} lines total)"
        numbered = "\n".join(f"{start_line + i + 1:5d}  {l}" for i, l in enumerate(chunk))
        return f"[Lines {start_line + 1}–{start_line + len(chunk)} of {total}]\n{numbered}"

    def detect_target_repo() -> str:
        """Detect the upstream GitHub repository this seed is targeting.
        Reasons over the seed README, patch file names, and data artifact names.
        Returns the detected repo slug (e.g. "ROCm/aiter", "vllm-project/vllm") and confidence.
        Call this if you are unsure which upstream repo the plan's changes belong to."""
        from pipeline.create_pr_from_seed import _detect_target_repo
        patches = seed.get("patches", []) + seed.get("file_edits", [])
        readme = seed.get("readme") or ""
        artifact_names = [d["name"] for d in seed.get("data_artifacts", [])]
        slug, confidence, reasoning = _detect_target_repo(
            readme, patches, data_artifact_names=artifact_names
        )
        if slug:
            return f"Target repo: {slug}\nConfidence: {confidence}\nReasoning: {reasoning}"
        return f"No target repo detected (confidence={confidence}). Reasoning: {reasoning}"

    # ── Diff index built from seed patches ─────────────────────────────────────
    # Build a per-file diff index from the generated patches so the RLM can
    # read any file's diff without us injecting the whole thing into the prompt.
    _patch_index: dict[str, str] = {}  # file_path → full diff hunks for that file
    _base_content_index: dict[str, str] = {}  # file_path → upstream base content

    def _build_patch_index(pr_diffs: dict[int, str]) -> None:
        _patch_index.clear()
        for diff_text in pr_diffs.values():
            _current_file = None
            _current_lines: list[str] = []
            for line in diff_text.splitlines(keepends=True):
                if line.startswith("+++ b/"):
                    if _current_file and _current_lines:
                        _patch_index.setdefault(_current_file, "")
                        _patch_index[_current_file] += "".join(_current_lines)
                    _current_file = line[6:].rstrip("\n")
                    _current_lines = [line]
                elif _current_file is not None:
                    _current_lines.append(line)
            if _current_file and _current_lines:
                _patch_index.setdefault(_current_file, "")
                _patch_index[_current_file] += "".join(_current_lines)

    def read_diff_section(file_path: str) -> str:
        """Return the full diff hunks for a specific file across all PRs.
        No character cap — the full content is returned as a string variable.
        Args:
            file_path: path as it appears in the diff (e.g. "vllm/model_executor/models/qwen2_moe.py")
        Returns all +/- lines for that file, or an error with available file list."""
        key = file_path.strip()
        if key not in _patch_index:
            for candidate in _patch_index:
                if candidate.endswith(key) or key.endswith(candidate):
                    key = candidate
                    break
        content = _patch_index.get(key, "")
        if not content:
            available = ", ".join(sorted(_patch_index)) or "(none)"
            return f"(file not found: {file_path!r}. Files in diff: {available})"
        return content

    def get_pr_plan_section(pr_index: int) -> str:
        """Return the full plan section for a specific PR: objective, in_scope, out_scope.
        No character cap.
        Args:
            pr_index: 1-based PR index as in the plan"""
        for spec in pr_plan.get("pr_series", []):
            if spec.get("index") == pr_index:
                import json
                return json.dumps(spec, indent=2)
        available = [s.get("index") for s in pr_plan.get("pr_series", [])]
        return f"(PR index {pr_index} not found. Available: {available})"

    def update_pr_plan_field(pr_index: int, field_path: str, value: str) -> str:
        """Update a field in the PR plan for the given PR index. Mutates pr_plan in-place
        so the arch critic sees the fix on its next check.

        Use this in STEP 0a when critic_feedback contains a PLAN-LEVEL issue such as
        intent=unknown, missing rationale, wrong upstream, or a new file not listed in
        affected_files. Call this before writing any diffs.

        Args:
            pr_index:   1-based PR index as in the plan
            field_path: one of:
                        "rationale"              → sets spec["rationale"]
                        "upstream"               → sets spec["upstream"]
                        "new_files.<path>"       → adds/updates an entry in spec["new_files"]
                                                   where path is the exact file path.
                                                   Pass value as JSON: '{"intent":"planned","justification":"..."}'
                        "new_files.<path>.intent" → sets the intent field on an existing or
                                                    new entry in spec["new_files"] for that path
            value:      new string value to set at that path (or JSON for new_files.<path>)
        Returns: confirmation string, or an error message starting with ERROR:
        """
        specs = pr_plan.get("pr_series", [])
        spec = next((s for s in specs if s.get("index") == pr_index), None)
        if spec is None:
            available = [s.get("index") for s in specs]
            return f"ERROR: PR {pr_index} not found in plan. Available: {available}"

        # Special handling for new_files — file paths contain dots so can't use naive split
        if field_path.startswith("new_files."):
            remainder = field_path[len("new_files."):]
            # Check if there's a trailing .field like ".intent" or ".justification"
            # Strategy: find an existing new_files entry whose path is a prefix of remainder,
            # OR treat remainder up to the last dot-segment as the file path if it ends with a known field
            _known_fields = {"intent", "justification", "path"}
            _nf_list = spec.setdefault("new_files", [])
            # Try to match a trailing known field
            _file_path_part = remainder
            _sub_field = None
            for _kf in _known_fields:
                if remainder.endswith(f".{_kf}"):
                    _file_path_part = remainder[:-(len(_kf) + 1)]
                    _sub_field = _kf
                    break
            # Find or create entry in new_files list
            _entry = next((e for e in _nf_list if isinstance(e, dict) and e.get("path") == _file_path_part), None)
            if _entry is None:
                _entry = {"path": _file_path_part, "intent": "planned", "justification": ""}
                _nf_list.append(_entry)
            if _sub_field:
                _entry[_sub_field] = value
            else:
                # value is the full entry or a single value for the path itself
                import json as _json
                try:
                    _parsed = _json.loads(value)
                    if isinstance(_parsed, dict):
                        _entry.update(_parsed)
                    else:
                        _entry["intent"] = value
                except Exception:
                    _entry["intent"] = value
            logger.info("plan_revision: PR %d new_files[%s] updated", pr_index, _file_path_part)
            return f"Updated PR {pr_index} plan: new_files entry for '{_file_path_part}' → {_entry}"

        # Simple top-level field update
        spec[field_path] = value
        logger.info("plan_revision: PR %d %s = %r", pr_index, field_path, value)
        return f"Updated PR {pr_index} plan: {field_path} = {value!r}"

    # ── Validation tools ───────────────────────────────────────────────────────

    def validate_rewrite_tool(
        base_content: str,
        rewritten_content: str,
        file_path: str,
        in_scope_json: str = "[]",
    ) -> str:
        """Run structural validation on a rewritten file section.
        Checks: Python syntax, symbol drops, signature mutations, stub-body replacements,
        patch applicability against git.
        Args:
            base_content: original upstream file content
            rewritten_content: your proposed rewritten content
            file_path: path used for file-type detection (e.g. "foo/bar.py")
            in_scope_json: JSON array of symbol names that are intentionally modified
        Returns: "OK" if all checks pass, or a list of error descriptions separated by newlines."""
        import json as _json
        try:
            in_scope = _json.loads(in_scope_json)
        except Exception:
            in_scope = []
        errors = validate_rewrite(base_content, rewritten_content, file_path, in_scope)

        # Check for new symbols added beyond in_scope.
        # Any new `def` or `class` not named in in_scope is blocking scope creep.
        import re as _re
        import difflib as _difflib
        _ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        if _ext in ("py", "cpp", "cu", "h", "cuh", "c"):
            _diff_lines = list(_difflib.unified_diff(
                base_content.splitlines(), rewritten_content.splitlines(), lineterm=""
            ))
            _new_symbols = []
            for _line in _diff_lines:
                if not _line.startswith("+") or _line.startswith("+++"):
                    continue
                _m = _re.match(r"^\+\s*(def|class)\s+(\w+)", _line)
                if _m:
                    _sym = _m.group(2)
                    if not any(_sym in s for s in in_scope):
                        _new_symbols.append(_sym)
            if _new_symbols:
                errors = list(errors) + [
                    f"Scope creep: '{s}' is a new symbol not listed in in_scope — "
                    "remove it or add it to in_scope before calling write_file_rewrite_tool."
                    for s in _new_symbols
                ]

        if not errors:
            return "OK"
        return "\n".join(errors)

    def judge_rewrite_tool(patch_text: str) -> str:
        """Run the rules harness against a unified diff patch.
        Returns blocking findings (severity: error/critical/high) or "PASS" if none.
        Args:
            patch_text: unified diff string (output of generate_patch_tool)"""
        try:
            result = judge_patch(repo_slug, patch_text)
            findings = result.get("findings", [])
            blocking = [f for f in findings if f.get("severity") in ("error", "critical", "high")]
            if not blocking:
                return "PASS"
            lines = []
            for f in blocking:
                lines.append(
                    f"[{f.get('severity', '?')}] {f.get('rule_id', '?')}: {f.get('message', '')}"
                    + (f" (file: {f.get('file', '')}, line: {f.get('line', '')})" if f.get("file") else "")
                )
            return "\n".join(lines)
        except Exception as exc:
            raise RuntimeError(f"judge_rewrite failed: {exc}") from exc

    def run_harness_tool(pr_index: int, patch_text: str) -> str:
        """Run the Phase 1 architecture audit harnesses for a specific PR.
        Args:
            pr_index: which PR to audit (1-based)
            patch_text: unified diff string for this PR
        Returns: "PASS" if all harnesses pass, or descriptions of failing harnesses."""
        if not accepted_harnesses:
            return "PASS (no harnesses loaded)"
        try:
            from pipeline.create_pr_from_seed import _select_relevant_harnesses, _run_harness
            pr_diffs_single = {pr_index: patch_text}
            selection = _select_relevant_harnesses(accepted_harnesses, pr_plan, pr_diffs_single)
            selected = selection.get(pr_index, [])
            if not selected:
                return "PASS (no harnesses selected for this PR)"
            # Find the PR spec for this index
            pr_spec = next(
                (s for s in pr_plan.get("pr_series", []) if s.get("index") == pr_index),
                {"title": "", "objective": "", "files": []},
            )
            failures = []
            for harness in selected:
                if getattr(harness, "pre_push_only", False):
                    continue
                violations = _run_harness(
                    harness, patch_text, pr_spec,
                    upstream_repo=resolved_upstream, token=token,
                )
                for v in (violations or []):
                    failures.append(f"[{harness.name}] {v}")
            if not failures:
                return "PASS"
            return "\n".join(failures)
        except Exception as exc:
            logger.warning("run_harness_tool failed for PR %d: %s", pr_index, exc)
            return f"(harness error: {exc})"

    def generate_patch_tool(base_content: str, new_content: str, file_path: str) -> str:
        """Generate a unified diff patch string between base and new content.
        Args:
            base_content: original file content (empty string for new files)
            new_content: proposed file content
            file_path: path used in the diff header
        Returns: unified diff string."""
        return generate_unified_patch(base_content, new_content, file_path) or "(no changes)"

    # ── Tools pulled from create_pr_from_seed private API ─────────────────────

    def find_upstream_path(filename: str) -> str:
        """Search the upstream repo for a file by name.
        Useful when you know a seed file's basename but not its full repo-relative path.
        Args:
            filename: bare filename to search for (e.g. "fused_moe.py", "qwen2_moe.py")
        Returns: repo-relative path (e.g. "vllm/model_executor/models/qwen2_moe.py") or an error."""
        from pipeline.create_pr_from_seed import _find_upstream_path
        result = _find_upstream_path(filename, resolved_upstream, token)
        if result:
            return result
        return f"(not found: {filename} in {resolved_upstream})"

    def check_patch_applies(patch_text: str) -> str:
        """Shallow-clone the upstream repo and verify that a patch applies cleanly.
        Use this before finalising a rewrite to confirm there are no conflicts.
        Args:
            patch_text: unified diff string to test
        Returns: JSON-like summary — 'applies: true/false', per-patch status, next steps."""
        from pipeline.create_pr_from_seed import _check_patch_applies
        import json as _json
        patches = [{"name": "rewrite.patch", "content": patch_text}]
        result = _check_patch_applies(resolved_upstream, patches)
        return _json.dumps(result, indent=2)

    def check_duplicate_prs(keywords_json: str, objectives_json: str = "[]") -> str:
        """Check whether the upstream repo already has open or merged PRs implementing the same objective.
        Call this before committing a rewrite to avoid duplicating existing work.
        Args:
            keywords_json: JSON array of search keywords (e.g. '["fused_moe", "MoE sorting"]')
            objectives_json: JSON array of objective strings (used for semantic dedup filtering)
        Returns: summary of any duplicate PRs found, or "No duplicates found"."""
        from pipeline.create_pr_from_seed import _check_duplicates
        import json as _json
        try:
            keywords = _json.loads(keywords_json)
            objectives = _json.loads(objectives_json)
        except Exception:
            return f"(invalid JSON args)"
        result = _check_duplicates(resolved_upstream, keywords, objectives or None)
        if result.get("blocked"):
            return result["message"]
        open_c = len(result.get("open_prs", []))
        merged_c = len(result.get("merged_prs", []))
        return f"No duplicates found (checked {open_c + merged_c} candidate PR(s))"

    def apply_seed_patch_to_base(base_content: str, patch_text: str, file_path: str) -> str:
        """Apply a seed patch (unified diff) to an upstream base file's content.
        Use this to compute what the seed author's final file looks like against the
        current upstream base — useful as a diff hint when the seed is a .patch file.
        Args:
            base_content: current upstream file content (from fetch_upstream_file)
            patch_text: unified diff content (from read_seed_file)
            file_path: file path for context (used in error messages only)
        Returns: patched file content, or an error message if the patch does not apply."""
        from pipeline.create_pr_from_seed import _apply_patch_to_base
        try:
            return _apply_patch_to_base(base_content, patch_text, file_path)
        except Exception as exc:
            return f"(patch apply failed for {file_path}: {exc})"

    # ── Architecture principles harnesses (prefixed harness_) ─────────────────
    # Tools tagged harness_ run the architectural rule checks from layer_policy.py.
    # The RLM should call these after rewriting, the same way judge_rewrite_tool runs
    # the rules harnesses. Call harness_classify_files first to understand layer
    # assignments, then harness_check_compiler_pass_sufficiency for compiler-pass PRs,
    # then harness_audit_layer_distribution to run the full arch principles check.

    def _planned_new_file_paths() -> set[str]:
        """Live read from pr_plan so update_pr_plan_field additions are visible immediately."""
        paths: set[str] = set()
        for _pr_spec_nf in pr_plan.get("pr_series", []):
            for _nf in _pr_spec_nf.get("new_files", []):
                if isinstance(_nf, dict) and _nf.get("path"):
                    paths.add(_nf["path"])
        return paths

    def harness_classify_files(file_paths_json: str, pr_index: int = 0) -> str:
        """HARNESS: Classify a list of file paths into architectural layers for the upstream repo.
        Tells you which files are kernel-layer, model-layer, tooling, etc., and whether
        any are in layers that should not appear in a submission PR. Also annotates files
        as intent=planned (declared in the PR plan's new_files) or intent=unknown (new, not declared).
        Args:
            file_paths_json: JSON array of upstream file paths (e.g. '["vllm/model_executor/models/qwen2_moe.py"]')
            pr_index: optional PR index (1-based) to scope new_files check to a specific PR
        Returns: JSON mapping each path to its layer name, flagged status, and intent."""
        from pipeline.layer_policy import load_layer_policy, classify_seed_files, _match_layer
        import json as _json
        try:
            paths = _json.loads(file_paths_json)
        except Exception:
            return "(invalid JSON — expected array of file paths)"
        policy = load_layer_policy(resolved_upstream)

        # Per-PR new_files: if pr_index given, use that PR's new_files; else use all.
        if pr_index:
            _pr_nf_spec = next((s for s in pr_plan.get("pr_series", []) if s.get("index") == pr_index), None)
            _pr_new_files = {nf["path"] for nf in (_pr_nf_spec or {}).get("new_files", []) if isinstance(nf, dict) and nf.get("path")}
        else:
            _pr_new_files = _planned_new_file_paths()

        if not policy["enabled"]:
            annotated = {
                p: {
                    "layer": "unknown",
                    "flagged": False,
                    "intent": "planned" if p in _pr_new_files else "existing",
                }
                for p in paths
            }
            return _json.dumps(annotated, indent=2) + "\n(layer policy not enabled for this repo)"
        result = {p: _match_layer(p, policy) for p in paths}
        warn_layers = {layer["name"] for layer in policy["layers"] if layer.get("warn_if_present")}
        annotated = {
            p: {
                "layer": layer,
                "flagged": layer in warn_layers,
                "intent": "planned" if p in _pr_new_files else "unknown" if layer in warn_layers else "existing",
            }
            for p, layer in result.items()
        }
        return _json.dumps(annotated, indent=2)

    def harness_check_compiler_pass_sufficiency(
        pr_index: int, patch_text: str, objective: str
    ) -> str:
        """HARNESS: For compiler-pass objectives, verify the diff touches the pass registration file.
        A kernel-side change without a corresponding compiler-pass registration is insufficient
        for the objective to be mergeable upstream.
        Args:
            pr_index: which PR to check (1-based)
            patch_text: unified diff string for this PR
            objective: the PR's objective text
        Returns: "SUFFICIENT" or a description of what registration file is missing."""
        from pipeline.layer_policy import (
            load_layer_policy, check_compiler_pass_sufficiency,
        )
        import json as _json
        policy = load_layer_policy(resolved_upstream)
        if not policy["enabled"]:
            return "SUFFICIENT (layer policy not enabled)"
        pr_spec = next(
            (s for s in pr_plan.get("pr_series", []) if s.get("index") == pr_index),
            {"title": "", "objective": objective, "files": []},
        )
        result = check_compiler_pass_sufficiency(
            [patch_text], pr_plan, policy,
            target_repo=resolved_upstream, token=token,
        )
        verdicts = result.get("verdicts", [])
        issues = [v for v in verdicts if v.get("verdict") not in ("sufficient", "not_applicable")]
        if not issues:
            return "SUFFICIENT"
        return "\n".join(
            f"[insufficient] {v.get('objective', '')}: {v.get('reason', '')}"
            for v in issues
        )

    def harness_audit_layer_distribution(patch_text_json: str) -> str:
        """HARNESS: Run the full architecture principles audit against all PR diffs.
        Checks whether any diffs touch architectural layers that should not appear in
        upstream submissions (e.g. model-layer files for a kernel-only PR).
        The audit fetches the repo's own CONTRIBUTING.md and architectural docs to judge.
        Args:
            patch_text_json: JSON object mapping PR index (as string) to unified diff string
                             (e.g. '{"1": "diff --git ...", "2": "diff --git ..."}')
        Returns: "CLEAN" or a list of architectural warnings."""
        from pipeline.layer_policy import load_layer_policy, audit_layer_distribution
        import json as _json
        try:
            diffs_by_pr = _json.loads(patch_text_json)
        except Exception:
            return "(invalid JSON — expected {pr_index_str: patch_text})"
        policy = load_layer_policy(resolved_upstream)
        if not policy["enabled"]:
            return "CLEAN (layer policy not enabled)"
        diff_list = [v for _, v in sorted(diffs_by_pr.items(), key=lambda kv: int(kv[0]))]
        result = audit_layer_distribution(
            diff_list, pr_plan, policy,
            target_repo=resolved_upstream, token=token,
        )
        if result["clean"]:
            return "CLEAN"
        return "\n".join(result.get("warnings", []))

    # ── Output accumulators ────────────────────────────────────────────────────
    _committed_rewrites: dict[str, dict[str, str]] = {}  # pr_index_str → {file_path: content}
    _committed_bases: dict[str, dict[str, str]] = {}     # pr_index_str → {file_path: base_content}
    _deferred_files: list[dict] = []                     # files routed to a different upstream

    def write_file_rewrite_tool(
        pr_index: int,
        file_path: str,
        new_content: str,
        base_content: str = "",
    ) -> str:
        """Commit a validated file rewrite for a specific PR.
        Call this ONLY after validate_rewrite_tool returns "OK" and judge_rewrite_tool returns "PASS".
        Args:
            pr_index: which PR this file belongs to (1-based)
            file_path: upstream file path
            new_content: complete rewritten file content
            base_content: the original file content you started from before editing
                          (pass "" only for brand-new files that do not exist upstream).
                          ALWAYS pass this whenever you fetched the file with
                          fetch_upstream_file — regardless of whether the file is in the
                          primary upstream or a cross-upstream repo. It must match exactly
                          what fetch_upstream_file returned and is required for accurate diffs.
        Returns: "committed" confirmation."""
        if file_path.endswith(".json") and new_content.strip():
            try:
                import json as _json_mod
                _json_mod.loads(new_content)
            except Exception as _json_err:
                return (
                    f"ERROR: The content you wrote to {file_path} is not valid JSON — "
                    f"json.loads() failed: {_json_err}. "
                    f"Ensure the file begins with '{{' or '[' and ends with '}}' or ']', "
                    f"with all key-value pairs properly enclosed in an object or array. "
                    f"Do NOT write bare 'KEY: VALUE' fragments without enclosing braces."
                )
        key = str(pr_index)
        _committed_rewrites.setdefault(key, {})
        _committed_rewrites[key][file_path] = new_content
        _committed_bases.setdefault(key, {})
        if base_content:
            _committed_bases[key][file_path] = base_content
        logger.info("RLM committed rewrite: PR %d / %s (%d chars)", pr_index, file_path, len(new_content))
        return f"committed: PR {pr_index} / {file_path}"

    def defer_file_to_upstream(
        file_path: str,
        source_pr_idx: int,
        target_upstream: str,
        reason: str,
    ) -> str:
        """Defer a file's implementation to a different upstream's PR run.
        Use ONLY when fetch_upstream_file returns 404 for this file in the current upstream
        and you know it belongs to target_upstream.
        Do NOT call write_file_rewrite_tool for a deferred file.
        Args:
            file_path: the file path as it appears in the PR plan
            source_pr_idx: the PR index this file was originally assigned to
            target_upstream: full 'owner/repo' string for the correct upstream
            reason: brief explanation (e.g. 'file lives in ROCm/aiter not sgl-project/sglang')
        Returns: confirmation string."""
        _deferred_files.append({
            "file_path": file_path,
            "source_pr": source_pr_idx,
            "target_upstream": target_upstream,
            "reason": reason,
        })
        logger.info("RLM deferred %s → %s (PR %d): %s", file_path, target_upstream, source_pr_idx, reason)
        return f"deferred: {file_path} → {target_upstream}"

    def run_python_code(code: str, timeout_secs: int = 30) -> str:
        """Execute Python code in a sandboxed subprocess and return stdout+stderr (capped at 4000 chars).

        Use this to run codegen scripts after fetching them with fetch_upstream_file.
        Typical workflow for auto-generated files (files with 'DO NOT EDIT — generated by ...' headers):
          1. fetch_upstream_file the generator script (e.g. 'scripts/gen_lookup_header.py')
          2. Write the modified generator code as a Python string
          3. Call run_python_code with the modified code to get the generated output
          4. Diff the output against the existing generated file to produce the patch

        The code runs in a fresh temp directory — write any dependency files using open() first.
        Do NOT use for GPU operations, network downloads, or operations requiring HIP/CUDA libraries.
        Args:
            code: Python source code to execute
            timeout_secs: max execution time in seconds (default 30, max 60)
        Returns: combined stdout+stderr output string, or error message."""
        import subprocess as _subprocess
        import tempfile as _tempfile
        import textwrap as _textwrap
        _timeout = min(int(timeout_secs), 60)
        with _tempfile.TemporaryDirectory() as _tmpdir:
            _script = _tmpdir + "/run.py"
            try:
                with open(_script, "w") as _f:
                    _f.write(_textwrap.dedent(code))
                _r = _subprocess.run(
                    ["python3", _script],
                    capture_output=True, text=True,
                    timeout=_timeout, cwd=_tmpdir,
                )
                _out = (_r.stdout + _r.stderr).strip()
                if len(_out) > 4000:
                    _out = _out[:4000] + f"\n... (truncated, {len(_out)} total chars)"
                return _out or "(no output)"
            except _subprocess.TimeoutExpired:
                return f"[timeout after {_timeout}s — script did not complete]"
            except Exception as _exc:
                return f"[error: {_exc}]"

    # ── Build initial diff index from seed patches ─────────────────────────────
    # Fetch base content for all files touched by the plan so the RLM has it.
    pr_series = pr_plan.get("pr_series", [])
    all_affected: list[str] = []
    for spec in pr_series:
        all_affected.extend(spec.get("affected_files", spec.get("files", [])))

    for fp in set(all_affected):
        if fp not in _base_content_index:
            # Fetch full file content for diff computation — bypass the paged tool to avoid
            # line-count caps producing truncated base content and spurious diff additions.
            import base64 as _b64pre, httpx as _httpxpre
            _pre_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{fp}"
            try:
                _pre_r = _httpxpre.get(_pre_url, headers=_headers, timeout=15)
                if _pre_r.status_code == 200:
                    _pre_data = _pre_r.json()
                    if isinstance(_pre_data, dict) and _pre_data.get("encoding") == "base64":
                        _base_content_index[fp] = _b64pre.b64decode(
                            _pre_data["content"]
                        ).decode("utf-8", errors="replace")
                    else:
                        _base_content_index[fp] = ""
                else:
                    _base_content_index[fp] = ""
            except Exception:
                _base_content_index[fp] = ""

    # Seed files (pre-computed seed content for data artifacts and file edits)
    if seed_files:
        for fp, content in seed_files.items():
            _base_content_index.setdefault("seed:" + fp, content)

    # ── Build critic feedback string ───────────────────────────────────────────
    import json as _json
    feedback_block = ""
    if critic_feedback:
        parts = []
        for pr_idx, issues in sorted(critic_feedback.items()):
            has_empty_diff = any("diff is empty" in iss.lower() for iss in issues)
            if has_empty_diff:
                parts.append(f"⚠ MUST IMPLEMENT — diff was empty last iteration (PR {pr_idx}):")
            else:
                parts.append(f"PR {pr_idx}:")
            for iss in issues:
                parts.append(f"  - {iss}")
        feedback_block = "\n".join(parts)

    # ── Build iteration history block ──────────────────────────────────────────
    history_block = ""
    if iter_history:
        parts = []
        for entry in iter_history:
            n = entry.get("iter", "?")
            parts.append(f"=== Iteration {n} ===")
            files = entry.get("files_touched", [])
            if files:
                parts.append(f"Files touched: {', '.join(files)}")
            outcomes = entry.get("phase_outcomes", [])
            for o in outcomes:
                parts.append(f"  [{o['phase']}] {o['result']}: {o.get('detail', '')}")
            note = entry.get("note", "")
            if note:
                parts.append(f"  Note: {note}")
        history_block = "\n".join(parts)

    written_prs_block = ""
    if already_written_prs:
        written_prs_block = json.dumps(already_written_prs)

    # ── Build locked-diffs block (passing-file anchoring) ─────────────────────
    # Files whose diffs passed critic review in the previous iteration are
    # injected verbatim so the RLM can re-apply them mechanically rather than
    # re-deriving from memory — eliminating the primary cause of rewrite_exhausted
    # on multi-file PRs where the RLM drops previously-correct files.
    locked_diffs_block = ""
    if passing_file_diffs:
        parts = ["── LOCKED DIFFS (do not re-derive these — they passed critic review) ────────"]
        parts.append(
            "The following file diffs passed all critic checks in the previous iteration.\n"
            "Re-apply them EXACTLY as shown. Do NOT re-fetch, re-derive, or re-generate.\n"
            "Only modify files the critic explicitly flagged in prior_critic_feedback."
        )
        for filepath, diff_content in passing_file_diffs.items():
            parts.append(f"\n=== {filepath} ===\n{diff_content}")
        locked_diffs_block = "\n".join(parts)

    # ── Build bundle/multi-upstream context block ──────────────────────────────
    bundle_context_block = ""
    if active_pr_indices is not None or prior_upstream_context:
        parts = ["── MULTI-UPSTREAM BUNDLE CONTEXT ────────────────────────────────────────"]
        if active_pr_indices is not None:
            parts.append(
                f"This run writes PRs for upstream: {resolved_upstream}\n"
                f"Your active PRs (write ONLY these indices): {active_pr_indices}\n"
                f"The full bundle plan above contains PRs for other upstreams too — "
                f"do NOT write those. Use defer_file_to_upstream for any file that "
                f"returns 404 in {resolved_upstream}."
            )
        if prior_upstream_context:
            parts.append("Prior upstream runs (read-only context — do NOT re-implement these):")
            for entry in prior_upstream_context:
                if "deferred_to" in entry:
                    parts.append(
                        f"  [deferred] PR {entry['source_pr']} file {entry['file_path']} "
                        f"→ {entry['deferred_to']}: {entry['reason']}"
                    )
                else:
                    diff_preview = (entry.get("diff", "")[:200] + "...") if entry.get("diff") else "(empty)"
                    parts.append(
                        f"  [committed] upstream={entry['upstream']} PR {entry['pr_idx']} "
                        f"'{entry['title']}' diff_preview={diff_preview}"
                    )
        parts.append("─────────────────────────────────────────────────────────────────────────")
        bundle_context_block = "\n".join(parts)

    # ── DSPy RLM setup ─────────────────────────────────────────────────────────

    class RewriteSignature(dspy.Signature):
        """You are a code rewriting agent that implements upstream pull requests from seed material.

        You have direct access to `seed` and `pr_plan` as Python variables in the REPL.
        Never wait for tools to tell you something you can compute from these variables directly.

        VARIABLE SHAPES:
          seed: dict with keys:
            "readme"         → str: seed README / motivation text
            "patches"        → list of {"name": str, "content": str}   # unified diffs
            "file_edits"     → list of {"name": str, "content": str}   # full file replacements
            "data_artifacts" → list of {"name": str, "content": str}   # CSV/YAML/JSON data

          pr_plan: dict with keys:
            "pr_series" → list of {
              "index":          int,          # 1-based PR number
              "title":          str,
              "objective":      str,          # what this PR accomplishes
              "in_scope":       list[str],    # symbols / changes explicitly authorised
              "out_scope":      list[str],    # must NOT touch
              "affected_files": list[str],    # HINT — may be incomplete or wrong
            }

          pr_plan is a LOOSE STARTING POINT. It was produced by a fast cursory pass
          over the seed without deep upstream knowledge. You have stronger tools and
          must override it when your exploration reveals:
            • files it listed that are not actually needed
            • files it missed that the objective clearly requires
            • a grouping of objectives into PRs that does not make sense
          Your Change Tree (Step 1) is authoritative — pr_plan["affected_files"] is a hint.

        ── STEP 0: ORIENT AND PLAN ───────────────────────────────────────────────
        IMPORTANT — if prior_iter_history is non-empty, the upstream pipeline has
        ALREADY completed intent extraction and dedup for this run. Do NOT call
        check_duplicate_prs, detect_target_repo for dedup purposes, or re-derive
        objectives. Instead, read prior_iter_history to understand:
          - which files were touched in previous iterations
          - which phase checks failed and why (coverage, arch, plan-consistency)
          - what the previous iteration's note says about what went wrong
        Use this history to build a targeted fix plan for this iteration.

        Run Python in the REPL before touching any file:
          seed["readme"][:3000]                           # read motivation
          [p["name"] for p in seed["patches"]]            # enumerate patches
          [d["name"] for d in seed["data_artifacts"]]     # enumerate data files
          [s["objective"] for s in pr_plan["pr_series"]]  # enumerate objectives

        After orienting, write a detailed work plan in the REPL as a Python list
        before fetching any upstream file or writing any code. Each item is one
        concrete unit of work: a file to fetch, a symbol to implement, a call
        site to wire, a validate or judge call. If prior_critic_feedback is
        non-empty, each feedback issue becomes its own item in the same list.
        If prior_iter_history is non-empty, each phase failure becomes its own
        fix item — address them explicitly before rewriting from scratch.

          plan = [
              {"id": 1, "pr": 1, "task": "fetch upstream <file>", "done": False},
              {"id": 2, "pr": 1, "task": "implement <symbol>",    "done": False},
              {"id": 3, "pr": 1, "task": "wire <symbol> into <call site>", "done": False},
              {"id": 4, "pr": 1, "task": "validate PR 1",         "done": False},
              ...
          ]
          print("PLAN:", plan)

        Work through the plan strictly in order. After completing each item, set
        plan[i]["done"] = True and print the updated plan before advancing to the
        next item. Never combine multiple items into one step or skip ahead.
        Never call write_file_rewrite_tool for a PR until every plan item for
        that PR is marked done — print the open items and confirm the list is
        clear before writing.

        ── STEP 0b: WRITE-SEQUENCE AND CONSOLIDATION PLANNING (run ONCE on first iter) ──
        Skip this step if prior_iter_history is non-empty — sequencing was decided on iter 1.
        If this is the first iteration (prior_iter_history is empty):

        1. SEQUENCE: Read pr_plan["pr_series"] and identify dependency edges.
           If PR B calls or imports a symbol that PR A introduces, A must be written first.
           Treat the planner's order as a strong hint — override only when you see a clear
           dependency violation. Store your write order as a Python list in the REPL:
               write_order = [1, 2, 3]   # 1-based PR indices in the order you will write them
               print("WRITE ORDER:", write_order)

        2. CONSOLIDATION: For each adjacent pair in write_order ask:
           "Would a reviewer prefer these as one PR or two?"
           Consolidate ONLY when ALL of these hold:
             - Both PRs touch the same logical unit (same file or tightly coupled module pair)
             - The combined diff would be under ~400 lines
             - There is no reason a reviewer would want to bisect between them
             - Combining does not mix unrelated concerns (e.g. a kernel and a CSV table)
           If consolidating, merge the two specs' in_scope lists and affected_files.
           Use llm_query() to reason about a borderline case. Store your decision:
               consolidation_plan = {"merged": [(1,2)], "rationale": "..."}  # or {"merged": []}
               print("CONSOLIDATION:", consolidation_plan)

        3. COMMIT: Your final_write_order (post-consolidation) is locked for this run.
               final_write_order = [...]   # may be shorter than pr_plan["pr_series"]
               print("FINAL WRITE ORDER:", final_write_order)

        4. CONTEXT CHAINING: When writing PR at position N in final_write_order, load
           already_written_prs (available as a Python variable) for context. These are
           diffs that were locked in previous outer iterations. Do not re-introduce symbols
           or boilerplate already present in those diffs.

        ── STEP 1: BUILD A CHANGE TREE ───────────────────────────────────────────
        For each objective in pr_plan["pr_series"], construct a Change Tree entry:

          objective → [files you will actually touch] → target upstream repository

        Use detect_target_repo() to confirm the primary upstream. Each file must be
        assigned to exactly one upstream repo. Lean toward a single repo; split only
        if the seed clearly touches two separate projects (justify this in the tree).

        Derive the file list from seed exploration and upstream symbol tracing — do NOT
        copy pr_plan["affected_files"] blindly. Add files the plan missed; drop files the
        plan listed but the objective does not require.

        Store the Change Tree as a Python dict in the REPL — you will use it throughout.

        ── STEP 2: FETCH BASE CONTEXT ────────────────────────────────────────────
        For each (objective, file) pair in your Change Tree:

        a. Call fetch_upstream_file(path, repo=<upstream>) to read the upstream base for that file.
           ALWAYS pass repo= explicitly — read pr_plan["pr_series"][n]["upstream"] for the correct
           repo per PR. Every upstream in the bundle is first-class (there is no "primary"):
           ROCm/composable_kernel is as authoritative as ROCm/aiter. Page through large files
           with start_line. Store the full content as a variable.
        b. Follow imports, definitions, and dependencies that the changed code touches.
           Use fetch_symbol_definition(symbol) to pull the exact def/class block for any
           function or class the objective references — no manual file search needed.
           Use search_upstream_symbol(symbol) when you need to locate which file defines it.
           Use fetch_upstream_file on imported modules to understand calling conventions.
           Fetch enough to understand what already exists in the base before writing anything.
        c. Read the seed material for that file:
             patch_obj = next((p for p in seed["patches"] if <name matches>), None)
           Call read_seed_file(patch_obj["name"]) to page through it.
           The seed shows the author's INTENT — what they wanted to achieve.
           Do NOT copy the seed code verbatim — adapt it to the upstream base's idioms
           and calling conventions. EXCEPTION: if fetch_upstream_file returns a "[NEW FILE]"
           header, the file does not exist upstream yet and the seed content IS the intended
           implementation. Use it as-is and commit with base_content='' — no adaptation needed.
           EXCEPTION for deletion-only objectives: If the PR objective is to *remove* specific
           lines (debug prints, logging calls, dead code, unused imports), write the complete
           upstream file with only those targeted lines deleted. All other lines must be
           preserved exactly as fetched from fetch_upstream_file. The 'do not copy verbatim'
           rule prohibits wholesale copying of seed additions; it does NOT mean you should
           avoid reproducing existing upstream code — you must copy all preserved lines exactly.

        ── STEP 3: IMPLEMENT THE OBJECTIVE ──────────────────────────────────────
        For each file in the current PR's Change Tree node:

        a. Reason about the base context you fetched. What existing symbols, patterns,
           and extension points does the base provide for this kind of change?
        b. Write the rewrite as a Python string variable in the REPL. Implement ONLY
           what is listed in spec["in_scope"]. Do not touch anything in spec["out_scope"].
        c. Validate immediately — call validate_rewrite_tool(base, rewritten, path, in_scope_json).
           Fix every reported error before continuing.
        d. Generate a patch: call generate_patch_tool(base, rewritten, path).
        e. Run rules harnesses: call judge_rewrite_tool(patch). Fix all blocking findings.
        f. Run harness validation (see ARCHITECTURE HARNESSES below).
        g. Only after ALL checks pass: call write_file_rewrite_tool(pr_index, path, rewritten, base_content=base).
           The `base_content` argument must be the exact string you fetched from fetch_upstream_file
           before editing. ALWAYS pass it for any file you fetched via fetch_upstream_file —
           regardless of whether the file lives in the primary upstream or a cross-upstream repo.
           This is required for accurate diff generation in every case.
           Pass "" only for brand-new files that do not exist anywhere upstream.

        ── AUTO-GENERATED FILES ──────────────────────────────────────────────────
        If a file you need to modify contains a header like "DO NOT EDIT — generated by <script>"
        or "# auto-generated", it is produced by a codegen script. To produce a correct diff:
          1. Fetch the generator script: fetch_upstream_file("scripts/<gen_script>.py")
          2. Write modified generator code as a Python string, incorporating the new kernel/entry
             the objective requires (e.g. new kernel name, new dispatch table entry)
          3. Call run_python_code(code) to execute it and capture the generated output
          4. Compare the generated output to the existing generated file (fetched via
             fetch_upstream_file) to produce the correct patch — only the added/changed lines
          5. Proceed to validate_rewrite_tool and write_file_rewrite_tool as normal
        Only use run_python_code for PURE PYTHON codegen (template instantiation, table generation).
        Do NOT use it for scripts requiring GPU, HIP, C++ compilation, or network access.

        ── TUNING CSV AND BENCHMARK DATA FILES ──────────────────────────────────
        If any file you need to write is a CSV or JSON file of GPU tuning measurements
        (paths containing: aiter/configs/, model_configs/, ops/triton/configs/, or filenames matching
        *_tuned_*.csv, *_gemm*.csv, *_fmoe*.csv, *_moe*.csv, *.json in a configs/ directory,
        or containing columns/fields like us1/us2/bw/latency/tflops/kernelId/buckets):

          RULE: Do NOT generate, estimate, or approximate any numeric performance values.
                GPU kernel benchmark data can only come from real hardware runs.
                For new JSON config files that fetch_upstream_file returns as "[NEW FILE]",
                use the seed content verbatim — it contains pre-measured tuning values.

          FIRST: check whether the PR plan spec for this PR has a `csv_seed_rows` field:
            rows = pr_plan["pr_series"][n]["csv_seed_rows"]   # pre-extracted from seed diff
          If `csv_seed_rows` is present and non-empty:
            → These are the ONLY rows you are permitted to add. Copy them verbatim.
            → Your diff MUST add EXACTLY len(csv_seed_rows) rows — no more, no fewer.
            → Do NOT search the seed diff for additional rows; all rows are already here.

          FALLBACK (only if csv_seed_rows is absent or empty):
            1. Find this file in seed["patches"] (look for `+++ b/<filename>` in the diff).
            2. Extract ONLY rows marked with leading `+` in the seed diff.
            3. If the seed diff adds N rows, your diff MUST add EXACTLY those N rows.
            4. If brand-new file: header row (from seed) + verbatim `+` rows only.

          If neither csv_seed_rows nor seed patch rows exist for this CSV:
            → Call defer_file_to_upstream(file_path=..., source_pr_idx=...,
                                          target_upstream="needs_hardware_benchmark",
                                          reason="CSV tuning data requires GPU runs; no seed rows available")
            → Do NOT write any rows. Do NOT approximate.

        ── VERBATIM SEED FILE CONTENT (JSON / YAML / new C++ files) ─────────────
        Some PRs include a `seed_files` dict in their spec:
          seed_files = pr_plan["pr_series"][n].get("seed_files", {})  # {path: content}

        These are pre-extracted verbatim from the seed patch. Rules:
        - If seed_files is present and non-empty, check each planned file against it.
        - For a file listed in seed_files: include it in your diff with EXACTLY the
          provided content — do NOT synthesize, estimate, or modify any values.
        - If the file is NEW (not yet in upstream): your diff creates it with this content.
        - If the file MODIFIES an existing upstream file: the provided content shows the
          lines that were added/changed — incorporate them into the patched file.
        - The plan_consistency critic will verify each seed_files entry appears in the diff.
          Omitting any of these files will cause rewrite_exhausted.
        - Do NOT approximate JSON tuning values. Do NOT reformat or reorder keys.
          Copy the content byte-for-byte as provided.

        ── C++ FILES IN SECONDARY-UPSTREAM PRs (e.g. ROCm/composable_kernel) ─────
        For PRs targeting a secondary upstream (e.g. ROCm/composable_kernel), the seed
        patches in seed["patches"] contain the exact diffs to apply. Workflow:
        1. Identify the patch entry: find the patch whose content has "+++ b/<file_path>"
           matching the file you need to modify. Do this via Python:
             patch_text = next(p["content"] for p in seed["patches"]
                               if "<file_path>" in p["content"])
        2. Fetch the upstream base: fetch_upstream_file(file_path, repo="ROCm/composable_kernel")
        3. Apply the patch: apply_seed_patch_to_base(base_content, patch_text, file_path)
        4. Use the result as your new_content in write_file_rewrite_tool.
        This applies to ALL C++ header/source modifications in composable_kernel or any
        other secondary upstream where the seed patch is the authoritative change source.

        ── STEP 4: ARCHITECTURE HARNESSES ───────────────────────────────────────
        After generating each PR's patch, before calling write_file_rewrite_tool:

        The upstream repo has layer policies that govern its software architecture —
        e.g. in vLLM, the separation between atom (kernel) layer, FX graph compilation
        passes, and model-layer wiring. Use the harness_ tools to discover and enforce
        these policies:

        1. harness_classify_files(file_paths_json, pr_index=N) — classify each changed
           file into its architectural layer. Pass pr_index to get intent annotations:
           - intent="planned" → declared in pr_plan["pr_series"][N]["new_files"]; do NOT
             flag this file as unknown scope. The plan explicitly expects this new file.
           - intent="existing" → already in upstream; normal modification.
           - intent="unknown" → NEW file NOT in new_files; investigate before proceeding.
           Flagged layers AND unknown-intent new files both require justification before
           write_file_rewrite_tool is called. For intent=planned files, no justification
           is needed — the plan declared them explicitly.
        2. harness_check_compiler_pass_sufficiency(pr_index, patch, objective) — for
           objectives involving a new kernel dispatch or compiler pass, verify the diff
           also registers the pass. Skip for data-only or model-wiring PRs.
        3. harness_audit_layer_distribution(patch_text_json) — full architectural audit.
           Returns "CLEAN" or a list of warnings. Revise the diff to remove flagged
           files or justify their inclusion before calling write_file_rewrite_tool.

        ── BANNED ACTIONS ────────────────────────────────────────────────────────
        Violating these will fail validation:
        - Do NOT remove a function or class not listed in spec["in_scope"].
        - Do NOT replace a function body with `raise NotImplementedError` or `pass`
          unless the upstream base already has that stub.
        - Do NOT add helper functions, classes, or module-level constants not in spec["in_scope"].
        - Do NOT rename parameters, struct fields, config keys, or dictionary keys.
        - Do NOT remove function parameters — downstream callers depend on them.

        ── DELETION AUDIT ────────────────────────────────────────────────────────
        Before calling write_file_rewrite_tool, run Python to diff your rewrite against base.
        For every deleted line confirm:
          (a) Named in spec["in_scope"] → intentional.
          (b) Whitespace/comment only, no semantic effect → ok.
          (c) Anything else → restore verbatim from base.
        The seed author may have incidentally deleted lines while focused on their core
        objective. Do not propagate those deletions.

        ── DATA FILES (CSV, YAML, JSON) ──────────────────────────────────────────
        Access data artifact content directly in Python from seed["data_artifacts"].
        Skip validate_rewrite_tool and judge_rewrite_tool for pure data files.
        Use run_python to verify: row count, column structure, no existing rows dropped.
        Semantics are append-only unless spec["in_scope"] explicitly authorises deletion.

        CRITICAL — CSV row completeness: when a PR spec contains csv_seed_rows, you
        MUST write every row verbatim, in order, without reconstruction or arithmetic.
        Do NOT regenerate the sequence from a pattern — copy directly from
        spec["csv_seed_rows"]. After writing, run a Python assertion:
            assert len(appended_rows) == len(spec.get("csv_seed_rows", [])), \
                f"CSV row count mismatch: wrote {len(appended_rows)}, expected {len(spec.get('csv_seed_rows', []))}"
        If the assertion fails, re-read spec["csv_seed_rows"] and rewrite the append
        block completely from the list, not from memory or reconstruction.

        ── STEP 0a: PLAN REVISION (run this before STEP 0 when prior_critic_feedback is non-empty) ─

        Read prior_critic_feedback carefully. For each issue, decide whether it is
        PLAN-LEVEL or DIFF-LEVEL:

          PLAN-LEVEL — the issue is about what the plan *declares*, not about code:
            • "intent=unknown" for a new file — the plan never said why this file was added
            • "no stated rationale" — the PR spec is missing a justification field
            • "wrong upstream" — the PR's `upstream` field points to the wrong repo
            • "new file not listed in affected_files" — plan and diff are out of sync
            Example: critic says "utils/rope_helpers.py — New utility module added with
              intent=unknown; all new files must declare their purpose in the plan."
              → Fix: update_pr_plan_field(1, "new_files.utils/rope_helpers.py.intent", "planned")
            Example: critic says "PR 2 has no rationale field; the plan must explain why
              this change is split from PR 1."
              → Fix: update_pr_plan_field(2, "rationale", "Separates config loading from ...")
            Call update_pr_plan_field() to fix the plan metadata. The plan is read by the
            arch critic on every check — this is the only way to make the critic see the fix.
            Do NOT try to resolve plan-level issues through code changes alone.

          DIFF-LEVEL — the issue is about what the diff *does* in code:
            • Wrong function signature, missing import, incorrect logic
            • Test added but assertions are wrong
            • Scope creep: code changed outside in_scope
            Example: "PR 1: quantize_weights() call missing scale_factor argument"
            Example: "PR 2: import added for module not listed in in_scope"
            Fix: rewrite the relevant file(s) in STEP 1+ as usual.
            CRITICAL — file preservation rule: when fixing diff-level issues, only
            rewrite the specific file(s) the critic explicitly flagged. For every
            other file that you modified in the previous iteration, re-apply your
            previous changes unchanged — do not re-fetch from upstream, do not
            re-derive the diff. If fixing file A requires a cascading change to
            file B (e.g. a new symbol that must be imported), state this cascade
            explicitly in your STEP 0 plan before touching file B.

        For every PLAN-LEVEL issue:
          1. Identify the PR index and the field to update (e.g. "rationale", "new_files.<name>.intent").
          2. Call update_pr_plan_field(pr_index, field_path, value) — updates pr_plan in-place.
          3. Print the returned confirmation string.
          Complete ALL plan updates before moving to STEP 0 and diff writing.

        For DIFF-LEVEL issues: handle in STEP 1+ as usual.
        If prior_critic_feedback is empty: skip this step entirely.

        ── PRIOR CRITIC FEEDBACK ─────────────────────────────────────────────────
        If prior_critic_feedback is non-empty, include one plan item per feedback
        issue (labelled with its PR number) when building your STEP 0 work plan.
        Apply the same sequential discipline: fix one issue, verify it with
        validate_rewrite_tool, mark the plan item done, then advance to the next.
        Do not bundle multiple issues into a single file write.

        ── BUDGET SKILL (maintain throughout) ────────────────────────────────────
        At the start of your REPL session run:
            budget_used = 0; budget_cap = 400
        Before every llm_query() call check:
            if budget_used >= budget_cap * 0.8:
                # Low budget — skip llm_query(); reason from already-fetched text instead.
                # Prioritize: validation harnesses > critic feedback fixes > new file fetches.
                pass
        After every llm_query() call: budget_used += 1

        Return JSON: {"committed": [{"pr_index": N, "files": ["path1", ...]}, ...]}
        """

        seed: dict = dspy.InputField(
            desc="Full seed dict — patches, file_edits, data_artifacts, readme accessible as Python variables"
        )
        pr_plan: dict = dspy.InputField(
            desc="Loose starting-point plan from a cursory seed pass. pr_series has index, objective, upstream (target repo for this PR), new_files (explicitly planned new file paths with intent), in_scope, out_scope, affected_files — treat affected_files as a hint; your Change Tree is authoritative. new_files and upstream are authoritative."
        )
        prior_critic_feedback: str = dspy.InputField(
            desc="Violations from the Phase 4 plan-consistency critic (empty string if first run)"
        )
        prior_iter_history: str = dspy.InputField(
            desc="Structured summary of previous rewrite iterations: files touched, phase outcomes, and notes per iter. Empty string on the first run. Use this to avoid repeating mistakes and to skip tasks already confirmed done (e.g. dedup, intent extraction) by the upstream pipeline."
        )
        already_written_prs: str = dspy.InputField(
            desc="JSON list of {title, pr_index, diff} for PRs whose diffs were locked in a prior outer iteration. Available as a Python variable in the REPL for context chaining. Empty string if no prior diffs exist."
        )
        upstream_repo: str = dspy.InputField(desc="Primary target upstream repo (owner/name)")
        bundle_context: str = dspy.InputField(
            desc="Multi-upstream bundle context: which PR indices to write in this run, what prior upstream runs committed, and defer_file_to_upstream instructions. Empty string for single-upstream seeds."
        )
        locked_diffs: str = dspy.InputField(
            desc="Diffs for files that passed all critic checks in the previous iteration. Re-apply these verbatim — do not re-derive. Empty string on first iteration or when no files have passed yet."
        )
        result: str = dspy.OutputField(
            desc='JSON: {"committed": [{"pr_index": N, "files": [...]}, ...]}'
        )

    import shutil as _shutil
    if not _shutil.which("deno"):
        raise RuntimeError(
            "dspy.RLM requires Deno for its sandboxed Python REPL but 'deno' was not found in PATH. "
            "Install Deno (https://docs.deno.com/runtime/getting_started/installation/) and restart."
        )

    _lm = _make_dspy_lm(model)
    rlm = dspy.RLM(
        RewriteSignature,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
        max_output_chars=50_000,
        verbose=verbose,
        tools=[
            # Seed inspection (tools still useful for fuzzy match / pagination ergonomics)
            inspect_seed,
            read_seed_file,
            detect_target_repo,
            # Upstream reads + navigation
            fetch_upstream_file,
            search_upstream_symbol,
            fetch_symbol_definition,
            find_upstream_path,
            # Seed → upstream reconciliation
            apply_seed_patch_to_base,
            check_duplicate_prs,
            # Plan + diff access (tool still useful for formatted JSON display)
            get_pr_plan_section,
            update_pr_plan_field,
            read_diff_section,
            # Validation — structural + rules
            validate_rewrite_tool,
            judge_rewrite_tool,
            run_harness_tool,
            check_patch_applies,
            # Harnesses — architecture principles (prefixed harness_)
            harness_classify_files,
            harness_check_compiler_pass_sufficiency,
            harness_audit_layer_distribution,
            # Output
            generate_patch_tool,
            write_file_rewrite_tool,
            defer_file_to_upstream,
            # Codegen: run auto-generated file scripts (pure Python only, no GPU/network)
            run_python_code,
        ],
    )

    logger.info(
        "RLM pipeline: %d PR(s), %d file(s), model=%s, feedback_issues=%d, max_iter=%d",
        len(pr_series), len(set(all_affected)), model,
        sum(len(v) for v in (critic_feedback or {}).values()),
        max_iterations,
    )

    with dspy.context(lm=_lm):
        prediction = rlm(
            seed=seed,
            pr_plan=pr_plan,
            prior_critic_feedback=feedback_block,
            prior_iter_history=history_block,
            already_written_prs=written_prs_block,
            upstream_repo=resolved_upstream,
            bundle_context=bundle_context_block,
            locked_diffs=locked_diffs_block,
        )

    # ── Build pr_diffs from committed rewrites ─────────────────────────────────
    pr_diffs: dict[int, str] = {}
    for pr_idx_str, file_map in _committed_rewrites.items():
        pr_idx = int(pr_idx_str)
        parts = []
        for fp, new_content in file_map.items():
            # Prefer the base the RLM explicitly passed (required for cross-upstream files
            # that 404 in _base_content_index); fall back to the pre-fetched index.
            base = (
                _committed_bases.get(pr_idx_str, {}).get(fp)
                or _base_content_index.get(fp, "")
            )
            patch = generate_unified_patch(base, new_content, fp)
            if patch:
                parts.append(patch)
        pr_diffs[pr_idx] = "\n".join(parts)

    # Fill in any PRs the RLM skipped (empty diff = no changes for that PR).
    for spec in pr_series:
        pr_diffs.setdefault(spec["index"], "")

    logger.info(
        "RLM pipeline complete: %d PR(s) with diffs, %d with changes, %d deferred",
        len(pr_diffs), sum(1 for d in pr_diffs.values() if d.strip()), len(_deferred_files),
    )
    return pr_diffs, _deferred_files
