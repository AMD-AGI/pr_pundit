"""
Stage A2 — Recipe / test-knowledge scraper.

Fetches three sources for a repo and writes raw files to
data/bronze_recipes/{owner}_{repo}/:

  1. A paired "recipes" repository (e.g. vllm-project/recipes)
     → every file in the repo
  2. Directories in the MAIN repo identified by LLM as containing benchmark
     or supporting test knowledge (not unit/functional tests)
  3. Inline documentation files identified by LLM as describing test patterns

The LLM inspects the full file tree of the main repo and selects which
directories to scrape — no hardcoded directory names.

Each scraped file becomes one JSON record in files.jsonl:
  { "repo": "vllm-project/vllm",
    "source_repo": "vllm-project/recipes",
    "source_type": "recipes" | "benchmarks" | "examples" | "docs" | "tests",
    "llm_reason": "why the LLM chose this directory",
    "path": "benchmarks/benchmark_throughput.py",
    "extension": ".sh",
    "size": 1234,
    "content": "...",
    "sha": "abc123" }

Usage:
    python -m scraper.scrape_recipes --repo vllm-project/vllm
    python -m scraper.scrape_recipes --repo vllm-project/vllm \\
        --recipes-repo vllm-project/recipes
    python -m scraper.scrape_recipes --repo vllm-project/vllm \\
        --override-dirs benchmarks/ sglang/bench_serving.py  # skip LLM discovery
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze_recipes"
GITHUB_API = "https://api.github.com"

KEEP_EXTENSIONS = {
    ".sh", ".py", ".md", ".rst", ".yaml", ".yml", ".toml", ".txt",
    # Native kernel benchmark repos (aiter, composable_kernel, hipBLAS)
    ".cpp", ".cu", ".hip", ".h", ".hpp",
}
MAX_FILE_BYTES = 200_000

KNOWN_RECIPE_REPOS: dict[str, str] = {
    "vllm-project/vllm": "vllm-project/recipes",
    "sgl-project/sglang": "sgl-project/sgl-cookbook",
}

_DISCOVER_PROMPT = """You are analyzing the file tree of the GitHub repository "{repo}" to identify which directories and files contain SUPPORTING TEST KNOWLEDGE useful for generating performance benchmarks, accuracy tests, or other supporting tests for new PR contributions.

We are NOT interested in:
- Unit tests (test correctness of individual functions)
- Functional/regression tests (pytest-based test suites)
- CI/CD configuration
- Build scripts

We ARE interested in:
- Throughput benchmarks (tokens/sec, requests/sec)
- Accuracy benchmarks (GSM8K, MMLU, lm-eval scripts)
- Latency / performance sweep scripts
- Kernel microbenchmarks
- End-to-end serving benchmarks
- Example scripts showing how to run models with specific flags
- Documentation describing benchmark procedures or standard configurations

FILE TREE (top-level structure):
{tree_summary}

Return a JSON array of directories/files to scrape:
[
  {{
    "path": "benchmarks/",
    "source_type": "benchmarks" | "examples" | "docs" | "tests",
    "reason": "one sentence why this is relevant"
  }},
  ...
]

Rules:
- Use trailing "/" for directories (will scrape all files inside recursively)
- Use exact path for individual files
- Include only paths that clearly contain performance/benchmark/supporting test knowledge
- If a tests/ directory has a benchmarks/ or performance/ subdirectory, include that subdirectory specifically rather than all of tests/
- Be selective — quality over quantity

Return ONLY the JSON array.
"""


class RestClient:
    """Thin GitHub REST client with rate-limit handling and token rotation.

    Accepts one or more tokens. On rate limit, rotates to the next available
    token instead of sleeping. Sleeps only when all tokens are exhausted,
    until the earliest reset time.
    """

    def __init__(self, tokens: str | list[str]):
        if isinstance(tokens, str):
            tokens = [t.strip() for t in tokens.split(",") if t.strip()]
        if not tokens:
            tokens = [""]
        self._clients: list[httpx.Client] = []
        for token in tokens:
            headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            self._clients.append(httpx.Client(headers=headers, timeout=30.0))
        self._idx = 0
        self._resets: dict[int, int] = {}  # client index → reset epoch
        if len(self._clients) > 1:
            logger.info("Token rotation enabled — %d tokens", len(self._clients))

    def _next_available(self) -> int | None:
        now = int(time.time())
        for i in range(len(self._clients)):
            idx = (self._idx + i) % len(self._clients)
            if self._resets.get(idx, 0) <= now:
                return idx
        return None

    def get(self, path: str, **params) -> dict | list:
        url = f"{GITHUB_API}{path}"
        for attempt in range(1, 6):
            resp = self._clients[self._idx].get(url, params=params or None)
            if resp.status_code == 404:
                return {}
            if resp.status_code in (403, 429):
                if "x-github-sso" in resp.headers:
                    sso_url = resp.headers["x-github-sso"].split("url=")[-1]
                    raise RuntimeError(
                        f"GitHub SSO enforcement: your token is not authorized for this org.\n"
                        f"Authorize it at: {sso_url}"
                    )
                if resp.status_code == 403 and int(resp.headers.get("x-ratelimit-remaining", 1)) > 0:
                    raise RuntimeError(
                        f"GitHub 403 Forbidden (not a rate limit): {resp.json().get('message', '')}"
                    )
                reset = int(resp.headers.get("x-ratelimit-reset", time.time() + 60))
                self._resets[self._idx] = reset
                next_idx = self._next_available()
                if next_idx is not None:
                    logger.info(
                        "Token %d rate limited — switching to token %d", self._idx, next_idx
                    )
                    self._idx = next_idx
                    continue
                wait = max(min(self._resets.values()) - int(time.time()), 1)
                logger.warning("All tokens rate limited, sleeping %ds (attempt %d)", wait, attempt)
                time.sleep(wait)
                next_idx = self._next_available()
                if next_idx is not None:
                    self._idx = next_idx
                continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Exhausted retries for {url}")

    def close(self):
        for c in self._clients:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _get_default_branch(client: RestClient, owner: str, name: str) -> str:
    info = client.get(f"/repos/{owner}/{name}")
    return info.get("default_branch", "main") if info else "main"


def _fetch_full_tree(client: RestClient, owner: str, name: str, branch: str) -> list[dict]:
    """Get the complete recursive file tree."""
    tree_data = client.get(f"/repos/{owner}/{name}/git/trees/{branch}", recursive="1")
    return tree_data.get("tree", []) if tree_data else []


def _summarize_tree(tree: list[dict]) -> str:
    """Build a compact tree summary for the LLM: top-level dirs + notable files."""
    # Collect top-level entries
    top_dirs: dict[str, int] = {}
    notable_files: list[str] = []

    for item in tree:
        path = item["path"]
        parts = path.split("/")
        if len(parts) == 1:
            if item["type"] == "blob":
                notable_files.append(path)
        else:
            top_dir = parts[0]
            top_dirs[top_dir] = top_dirs.get(top_dir, 0) + 1

    # For each top-level dir, show a sample of its contents (up to 8 files)
    dir_samples: dict[str, list[str]] = {}
    for item in tree:
        path = item["path"]
        parts = path.split("/")
        if len(parts) >= 2 and item["type"] == "blob":
            top = parts[0]
            if top in top_dirs:
                dir_samples.setdefault(top, [])
                if len(dir_samples[top]) < 8:
                    dir_samples[top].append(path)

    lines = ["TOP-LEVEL DIRECTORIES:"]
    for d, count in sorted(top_dirs.items()):
        lines.append(f"  {d}/  ({count} files total)")
        for sample in dir_samples.get(d, [])[:5]:
            lines.append(f"    {sample}")

    if notable_files:
        lines.append("\nTOP-LEVEL FILES:")
        for f in notable_files[:20]:
            lines.append(f"  {f}")

    return "\n".join(lines)


def _discover_dirs_with_llm(repo: str, tree: list[dict], model: str) -> list[dict]:
    """Ask the LLM which directories/files contain benchmark/test knowledge."""
    from pipeline.llm import llm_call, parse_json

    tree_summary = _summarize_tree(tree)
    prompt = _DISCOVER_PROMPT.format(repo=repo, tree_summary=tree_summary)

    logger.info("Asking LLM to discover relevant directories in %s...", repo)
    try:
        raw = llm_call(prompt, model, max_tokens=2048, json_mode=True)
        result = parse_json(raw)
        if isinstance(result, list):
            logger.info("LLM identified %d directories/files to scrape:", len(result))
            for item in result:
                logger.info("  %s (%s) — %s", item.get("path"), item.get("source_type"), item.get("reason", ""))
            return result
    except Exception as exc:
        logger.warning("LLM discovery failed: %s — falling back to empty list", exc)
    return []


def _fetch_content(client: RestClient, owner: str, name: str, path: str) -> str | None:
    """Fetch and decode file content. Returns None on failure or if too large."""
    data = client.get(f"/repos/{owner}/{name}/contents/{path}")
    if not data or isinstance(data, list):
        return None
    size = data.get("size", 0)
    if size > MAX_FILE_BYTES:
        logger.debug("Skipping large file %s (%d bytes)", path, size)
        return None
    content_b64 = data.get("content", "")
    if not content_b64:
        return None
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return None


def _scrape_selection(
    client: RestClient,
    source_repo: str,
    repo: str,
    tree: list[dict],
    selection: list[dict],
    write,
    checkpoint: set[str],
) -> int:
    """Scrape files from the repo that match the LLM-selected paths."""
    owner, name = repo.split("/", 1)

    # Build lookup: path → tree item
    tree_by_path = {item["path"]: item for item in tree if item.get("type") == "blob"}

    count = 0
    for sel in selection:
        sel_path = sel.get("path", "").rstrip("/")
        source_type = sel.get("source_type", "benchmarks")
        reason = sel.get("reason", "")
        is_dir = sel.get("path", "").endswith("/")

        # Find matching files
        if is_dir:
            matches = [
                item for item in tree
                if item.get("type") == "blob" and item["path"].startswith(sel_path + "/")
            ]
        else:
            matches = [tree_by_path[sel_path]] if sel_path in tree_by_path else []

        for item in matches:
            path = item["path"]
            ext = Path(path).suffix.lower()
            if ext not in KEEP_EXTENSIONS:
                continue
            record_key = f"{source_repo}:{path}"
            if record_key in checkpoint:
                continue

            content = _fetch_content(client, owner, name, path)
            if content is None:
                continue

            write({
                "repo": repo,
                "source_repo": source_repo,
                "source_type": source_type,
                "llm_reason": reason,
                "path": path,
                "extension": ext,
                "size": item.get("size", 0),
                "sha": item.get("sha", ""),
                "content": content,
            })
            checkpoint.add(record_key)
            count += 1
            if count % 20 == 0:
                logger.info("  %d files scraped so far...", count)

    return count


def _scrape_full_repo(
    client: RestClient,
    source_repo: str,
    repo: str,
    source_type: str,
    write,
    checkpoint: set[str],
) -> int:
    """Scrape every relevant file in a repo (used for recipe repos)."""
    owner, name = source_repo.split("/", 1)
    branch = _get_default_branch(client, owner, name)
    tree = _fetch_full_tree(client, owner, name, branch)

    count = 0
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item["path"]
        ext = Path(path).suffix.lower()
        if ext not in KEEP_EXTENSIONS:
            continue
        record_key = f"{source_repo}:{path}"
        if record_key in checkpoint:
            continue
        content = _fetch_content(client, owner, name, path)
        if content is None:
            continue
        write({
            "repo": repo,
            "source_repo": source_repo,
            "source_type": source_type,
            "llm_reason": "recipe repository — all files are relevant",
            "path": path,
            "extension": ext,
            "size": item.get("size", 0),
            "sha": item.get("sha", ""),
            "content": content,
        })
        checkpoint.add(record_key)
        count += 1
        if count % 20 == 0:
            logger.info("  recipe repo: %d files scraped...", count)

    return count


def scrape_recipes(
    repo: str,
    token: str,
    *,
    recipes_repo: str | None = None,
    override_dirs: list[str] | None = None,
    model: str = "claude-sonnet-4-6",
):
    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    out_dir = DATA_DIR / repo_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "files.jsonl"
    ckpt_file = out_dir / "checkpoint.json"

    checkpoint: set[str] = set()
    if ckpt_file.exists():
        ckpt_data = json.loads(ckpt_file.read_text())
        checkpoint = set(ckpt_data.get("scraped_keys", []))
        logger.info("Resuming — %d files already scraped", len(checkpoint))

    def write(record: dict):
        with open(out_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint():
        ckpt_file.write_text(json.dumps({"scraped_keys": list(checkpoint)}, indent=2))

    if recipes_repo is None:
        recipes_repo = KNOWN_RECIPE_REPOS.get(repo)

    total = 0
    with RestClient(token) as client:
        # 1. Recipe / cookbook repo — scrape everything (it's curated by definition)
        if recipes_repo:
            logger.info("Scraping recipe repo: %s", recipes_repo)
            n = _scrape_full_repo(client, recipes_repo, repo, "recipes", write, checkpoint)
            logger.info("Recipe repo: %d files", n)
            total += n
            save_checkpoint()

        # 2. Main repo — use LLM to discover relevant directories
        branch = _get_default_branch(client, owner, name)
        logger.info("Fetching file tree for %s (branch: %s)...", repo, branch)
        tree = _fetch_full_tree(client, owner, name, branch)
        logger.info("File tree: %d total files", len(tree))

        if override_dirs:
            # Manual override — build selection without LLM
            logger.info("Using override dirs: %s", override_dirs)
            selection = [
                {"path": d, "source_type": "benchmarks", "reason": "user-specified"}
                for d in override_dirs
            ]
        else:
            selection = _discover_dirs_with_llm(repo, tree, model)

        if selection:
            n = _scrape_selection(client, repo, repo, tree, selection, write, checkpoint)
            logger.info("Main repo selected dirs: %d files", n)
            total += n
            save_checkpoint()
        else:
            logger.warning("No directories selected — nothing scraped from main repo")

        # 3. CI configs — fixed dirs, always scraped regardless of LLM selection.
        # These contain real benchmark commands with exact models, flags, and hardware targets.
        ci_dirs = [
            {"path": ".buildkite", "source_type": "ci", "reason": "CI benchmark pipeline definitions"},
            {"path": ".github",    "source_type": "ci", "reason": "GitHub Actions workflow configs"},
        ]
        n = _scrape_selection(client, repo, repo, tree, ci_dirs, write, checkpoint)
        if n:
            logger.info("CI dirs: %d files", n)
            total += n
            save_checkpoint()

    logger.info("Recipe scrape complete — %d files → %s", total, out_file)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Scrape recipe repos and test dirs to bronze_recipes")
    p.add_argument("--repo", required=True, help="Main repo: owner/name")
    p.add_argument("--recipes-repo", default=None,
                   help="Recipe/cookbook repo (auto-detected for known repos)")
    p.add_argument("--override-dirs", nargs="+", default=None,
                   help="Skip LLM discovery and scrape these dirs/files directly")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Model for LLM directory discovery")
    p.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    args = p.parse_args()

    if not args.token:
        p.error("Provide --token or set GITHUB_TOKEN")

    scrape_recipes(
        args.repo,
        args.token,
        recipes_repo=args.recipes_repo,
        override_dirs=args.override_dirs,
        model=args.model,
    )


if __name__ == "__main__":
    main()
