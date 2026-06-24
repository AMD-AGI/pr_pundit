"""
Pipeline orchestrator for the recipe knowledge base.

Reads pipeline_config.yaml, tracks state in pipeline_state.json, and runs
all stages in dependency order with automatic retry and Teams notifications.

Usage:
    python -m pipeline.orchestrate              # run / resume from last failure
    python -m pipeline.orchestrate --restart    # clear state and start over
    python -m pipeline.orchestrate --dry-run    # show planned steps, don't run
    python -m pipeline.orchestrate --list       # show step statuses
    python -m pipeline.orchestrate --step scrape_recipes:ROCm/aiter  # one step only
    python -m pipeline.orchestrate --config my.yaml  # custom config file
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from pipeline.notify import _make_card, _post_card

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "pipeline_config.yaml"
DEFAULT_STATE  = Path(__file__).resolve().parent.parent / "pipeline_state.json"
DATA_ROOT      = Path(__file__).resolve().parent.parent / "data"

MAX_RETRIES   = 3
RETRY_BACKOFF = 30   # seconds between retries


# ── Teams helpers ─────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _notify(title: str, status: str, facts: list[tuple[str, str]], color: str = "default"):
    try:
        _post_card(_make_card(title, status, facts, color=color))
    except Exception:
        pass


def _notify_start(total_steps: int, repos: list[str]):
    _notify("🔄 Pipeline Started", "Started", [
        ("Repos", ", ".join(repos)),
        ("Steps", str(total_steps)),
        ("Timestamp", _ts()),
    ])


def _notify_step_done(step: str, attempt: int, elapsed: float):
    _notify(f"✅ {step}", "Done", [
        ("Step", step),
        ("Attempt", str(attempt)),
        ("Elapsed", f"{elapsed:.0f}s"),
        ("Timestamp", _ts()),
    ], color="good")


def _notify_step_failed(step: str, attempt: int, retrying: bool):
    status = f"Retrying ({attempt}/{MAX_RETRIES})" if retrying else "Failed"
    color  = "warning" if retrying else "attention"
    _notify(f"{'⚠️' if retrying else '❌'} {step}", status, [
        ("Step", step),
        ("Attempt", str(attempt)),
        ("Timestamp", _ts()),
    ], color=color)


def _notify_done(total: int, elapsed: float):
    _notify("✅ Pipeline Complete", "Done", [
        ("Steps completed", str(total)),
        ("Total elapsed", f"{elapsed / 60:.1f} min"),
        ("Timestamp", _ts()),
    ], color="good")


def _notify_pipeline_failed(step: str, attempts: int):
    _notify("❌ Pipeline Failed", "Error", [
        ("Failed step", step),
        ("Attempts", str(attempts)),
        ("Timestamp", _ts()),
    ], color="attention")


# ── Step builder ──────────────────────────────────────────────────────────────

def _build_steps(config: dict) -> list[dict]:
    """Return ordered list of step descriptors."""
    distill_cfg = config.get("distill", {})
    model       = distill_cfg.get("model", "claude-opus-4-7")
    workers     = str(distill_cfg.get("workers", 20))
    embed_model = distill_cfg.get("embed_model", "instructor-xl")

    steps: list[dict] = []

    def add(key: str, cmd: list[str]):
        steps.append({"key": key, "cmd": cmd})

    # Supplemental repos — scraped as their own recipe source
    for sup_repo in config.get("supplemental_repos", []):
        add(f"scrape_recipes:{sup_repo}", [
            "python", "-m", "scraper.scrape_recipes",
            "--repo", sup_repo,
            "--recipes-repo", sup_repo,
        ])
        add(f"normalize_recipes:{sup_repo}", [
            "python", "-m", "pipeline.normalize_recipes",
            "--repo", sup_repo,
        ])
        add(f"distill_recipes:{sup_repo}", [
            "python", "-m", "pipeline.distill_recipes",
            "--repo", sup_repo,
            "--model", model,
            "--workers", workers,
            "--embed-model", embed_model,
        ])

    # Rules repos — crawl merged PR review threads and distill reviewer rules
    import os
    # scrape_bronze uses a single-token GraphQL client — take the first token only
    gh_token = os.environ.get("GITHUB_TOKEN", "").split(",")[0].strip()
    for rules_entry in config.get("rules_repos", []):
        # Accept either a plain string or a dict with repo + options
        if isinstance(rules_entry, str):
            rules_repo = rules_entry
            include_closed = False
        else:
            rules_repo = rules_entry["repo"]
            include_closed = rules_entry.get("include_closed", False)
        slug = _repo_slug(rules_repo)
        scrape_bronze_cmd = [
            "python", "-m", "scraper.scrape_bronze",
            "--repo", rules_repo,
        ]
        if gh_token:
            scrape_bronze_cmd += ["--token", gh_token]
        add(f"scrape_bronze:{rules_repo}", scrape_bronze_cmd)
        # Closed PRs are a separate step so the merged-PR scrape checkpoint is preserved
        if include_closed:
            scrape_closed_cmd = [
                "python", "-m", "scraper.scrape_bronze",
                "--repo", rules_repo,
                "--include-closed",
            ]
            if gh_token:
                scrape_closed_cmd += ["--token", gh_token]
            add(f"scrape_bronze_closed:{rules_repo}", scrape_closed_cmd)
        add(f"normalize_rules:{rules_repo}", [
            "python", "-m", "pipeline.normalize",
            "--repo", slug,
        ])
        add(f"distill_rules:{rules_repo}", [
            "python", "-m", "pipeline.distill",
            "--repo", slug,
            "--model", model,
            "--workers", workers,
        ])

    # Target repos — recipe KB (test suggestions)
    for repo, repo_cfg in config.get("target_repos", {}).items():
        repo_cfg = repo_cfg or {}
        scrape_cmd = [
            "python", "-m", "scraper.scrape_recipes",
            "--repo", repo,
        ]
        if recipes_repo := repo_cfg.get("recipes_repo"):
            scrape_cmd += ["--recipes-repo", recipes_repo]

        add(f"scrape_recipes:{repo}", scrape_cmd)
        add(f"normalize_recipes:{repo}", [
            "python", "-m", "pipeline.normalize_recipes",
            "--repo", repo,
        ])
        add(f"distill_recipes:{repo}", [
            "python", "-m", "pipeline.distill_recipes",
            "--repo", repo,
            "--model", model,
            "--workers", workers,
            "--embed-model", embed_model,
        ])

    # PR lineage — build dependency trees for each rules repo (runs after distill_rules)
    lineage_model = distill_cfg.get("lineage_model", "claude-sonnet-4-6")
    rules_repo_names = [
        e if isinstance(e, str) else e["repo"]
        for e in config.get("rules_repos", [])
    ]
    for rules_repo in rules_repo_names:
        add(f"pr_lineage:{rules_repo}", [
            "python", "-m", "pipeline.pr_lineage",
            "--repo", rules_repo,
        ])

    # Architecture principles — distill design harnesses from all lineage data
    if rules_repo_names:
        design_cmd = ["python", "-m", "pipeline.distill_design_rules"]
        for repo in rules_repo_names:
            design_cmd += ["--repo", repo]
        design_cmd += ["--model", lineage_model, "--workers", workers]
        add("distill_design_rules:all", design_cmd)

    return steps


# ── Output detection ─────────────────────────────────────────────────────────

def _repo_slug(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"{owner}_{name}"


def _detect_done(steps: list[dict], state: dict, state_path: Path) -> int:
    """Mark steps as done based on reliable output sentinels.

    Only test_knowledge.json is a reliable completion marker — it is written
    atomically at the very end of distill_recipes. bronze/silver files are
    written incrementally so a non-empty file could be from a partial run.

    Strategy: if test_knowledge.json exists for a repo, all three stages
    (scrape, normalize, distill) succeeded — mark all three done.
    If it doesn't exist, leave scrape/normalize pending so they re-run;
    the scraper's own checkpoint makes re-runs fast and safe.
    """
    # Collect repos whose recipe distillation is provably complete
    complete_recipe_repos: set[str] = set()
    for step in steps:
        stage, repo = step["key"].split(":", 1)
        if stage != "distill_recipes":
            continue
        slug = _repo_slug(repo)
        kb = DATA_ROOT / "gold" / slug / "test_knowledge.json"
        if kb.exists() and kb.stat().st_size > 0:
            complete_recipe_repos.add(repo)

    # Collect repos whose rules distillation is provably complete
    complete_rules_repos: set[str] = set()
    for step in steps:
        stage, repo = step["key"].split(":", 1)
        if stage != "distill_rules":
            continue
        slug = _repo_slug(repo)
        rules = DATA_ROOT / "gold" / slug / "rules.json"
        if rules.exists() and rules.stat().st_size > 0:
            complete_rules_repos.add(repo)

    # Collect repos whose closed-PR scrape is provably complete
    complete_closed_repos: set[str] = set()
    for step in steps:
        stage, repo = step["key"].split(":", 1)
        if stage != "scrape_bronze_closed":
            continue
        slug = _repo_slug(repo)
        ckpt = DATA_ROOT / "bronze" / slug / "checkpoint_closed.json"
        if ckpt.exists() and ckpt.stat().st_size > 0:
            complete_closed_repos.add(repo)

    # Collect repos whose PR lineage is provably complete
    complete_lineage_repos: set[str] = set()
    for step in steps:
        stage, repo = step["key"].split(":", 1)
        if stage != "pr_lineage":
            continue
        slug = _repo_slug(repo)
        trees = DATA_ROOT / "lineage" / slug / "trees.jsonl"
        if trees.exists() and trees.stat().st_size > 0:
            complete_lineage_repos.add(repo)

    # Check if distill_design_rules:all is complete
    all_harnesses_exist = False
    for step in steps:
        if step["key"] != "distill_design_rules:all":
            continue
        # Complete if at least one repo has audit_harnesses.json
        lineage_dir = DATA_ROOT / "lineage"
        if lineage_dir.exists():
            harness_files = list(lineage_dir.glob("*/audit_harnesses.json"))
            all_harnesses_exist = len(harness_files) > 0

    newly_done = 0
    for step in steps:
        key = step["key"]
        if _step_status(state, key) == "done":
            continue
        stage, repo = key.split(":", 1)
        if stage == "scrape_bronze_closed":
            complete = complete_closed_repos
        elif stage in ("scrape_bronze", "normalize_rules", "distill_rules"):
            complete = complete_rules_repos
        elif stage == "pr_lineage":
            complete = complete_lineage_repos
        elif key == "distill_design_rules:all":
            if all_harnesses_exist:
                _mark(state, state_path, key, "done",
                      completed_at="detected", detected=True)
                print(f"  DETECTED  {key}")
                newly_done += 1
            continue
        else:
            complete = complete_recipe_repos
        if repo in complete:
            _mark(state, state_path, key, "done",
                  completed_at="detected", detected=True)
            print(f"  DETECTED  {key}")
            newly_done += 1

    return newly_done


# ── State management ──────────────────────────────────────────────────────────

def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"steps": {}}


def _save_state(state: dict, path: Path):
    path.write_text(json.dumps(state, indent=2))


def _step_status(state: dict, key: str) -> str:
    return state["steps"].get(key, {}).get("status", "pending")


def _mark(state: dict, path: Path, key: str, status: str, **extra):
    state["steps"].setdefault(key, {})
    state["steps"][key].update({"status": status, **extra})
    _save_state(state, path)


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_step(step: dict, dry_run: bool) -> bool:
    """Run one step with retry. Returns True on success."""
    key = step["key"]
    cmd = step["cmd"]

    print(f"\n{'─' * 60}")
    print(f"  STEP: {key}")
    print(f"  CMD:  {' '.join(cmd)}")
    if dry_run:
        print("  [dry-run — skipping]")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Attempt {attempt}/{MAX_RETRIES}  [{_ts()}]")
        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  ✓ Done in {elapsed:.0f}s")
            _notify_step_done(key, attempt, elapsed)
            return True

        retrying = attempt < MAX_RETRIES
        _notify_step_failed(key, attempt, retrying)
        if retrying:
            print(f"  ✗ Failed (exit {result.returncode}) — retrying in {RETRY_BACKOFF}s")
            time.sleep(RETRY_BACKOFF)
        else:
            print(f"  ✗ Failed after {MAX_RETRIES} attempts")

    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Run the full recipe KB pipeline")
    p.add_argument("--config",   default=str(DEFAULT_CONFIG), help="Config YAML")
    p.add_argument("--state",    default=str(DEFAULT_STATE),  help="State JSON")
    p.add_argument("--restart",  action="store_true", help="Ignore saved state and start over")
    p.add_argument("--dry-run",  action="store_true", help="Print steps without running")
    p.add_argument("--list",     action="store_true", help="Show step statuses and exit")
    p.add_argument("--step",     default=None, metavar="KEY", help="Run only this step key")
    p.add_argument("--detect",   action="store_true",
                   help="Scan output files and mark existing work as done, then exit")
    args = p.parse_args()

    config_path = Path(args.config)
    state_path  = Path(args.state)

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text()) or {}
    steps  = _build_steps(config)

    if args.restart:
        state_path.unlink(missing_ok=True)
    state = _load_state(state_path)

    # --detect: scan outputs and mark existing work done
    if args.detect:
        n = _detect_done(steps, state, state_path)
        print(f"\nDetected {n} completed steps. Run --list to review.")
        return

    # --list
    if args.list:
        print(f"{'STEP':<50}  STATUS")
        print("─" * 60)
        for s in steps:
            print(f"{s['key']:<50}  {_step_status(state, s['key'])}")
        return

    # --step: filter to a single step
    if args.step:
        steps = [s for s in steps if s["key"] == args.step]
        if not steps:
            print(f"Unknown step: {args.step}")
            print("Known steps:", ", ".join(s["key"] for s in _build_steps(config)))
            sys.exit(1)
        # allow re-running done steps when explicitly named
        for s in steps:
            state["steps"].pop(s["key"], None)

    all_repos = (
        list(config.get("supplemental_repos", []))
        + [e if isinstance(e, str) else e["repo"] for e in config.get("rules_repos", [])]
        + list((config.get("target_repos") or {}).keys())
    )

    # Auto-detect completed work from output files (catches manual runs)
    _detect_done(steps, state, state_path)

    pending = [s for s in steps if _step_status(state, s["key"]) != "done"]
    done_already = len(steps) - len(pending)

    print(f"\nPipeline: {len(steps)} steps total, {done_already} already done, {len(pending)} to run")
    if not args.dry_run:
        _notify_start(len(pending), all_repos)

    pipeline_start = time.time()

    for step in steps:
        key = step["key"]

        if _step_status(state, key) == "done" and not args.step:
            print(f"  SKIP  {key}  (already done)")
            continue

        _mark(state, state_path, key, "running",
              started_at=datetime.now(timezone.utc).isoformat(),
              attempts=state["steps"].get(key, {}).get("attempts", 0))

        success = _run_step(step, dry_run=args.dry_run)

        attempts = state["steps"][key].get("attempts", 0) + 1
        if success or args.dry_run:
            _mark(state, state_path, key, "done",
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  attempts=attempts)
        else:
            _mark(state, state_path, key, "failed", attempts=attempts)
            _notify_pipeline_failed(key, attempts)
            print(f"\nPipeline stopped at: {key}")
            print(f"Fix the issue and re-run to resume from this step.")
            sys.exit(1)

    total_elapsed = time.time() - pipeline_start
    completed = sum(1 for s in steps if _step_status(state, s["key"]) == "done")
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete — {completed} steps in {total_elapsed / 60:.1f} min")
    if not args.dry_run:
        _notify_done(completed, total_elapsed)


if __name__ == "__main__":
    main()
