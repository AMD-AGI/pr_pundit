# Contributing to PR Pundit

## Setup

```bash
git clone https://github.com/AMD-AGI/pr_pundit.git
cd pr_pundit
uv sync --extra mcp
cp .env.example .env   # fill in GITHUB_TOKEN and LITELLM_BASE_URL
```

Python 3.11+ required. [uv](https://docs.astral.sh/uv/) is the recommended package manager.

## Project layout

```
pipeline/       # core pipeline stages (distill, judge, conform, pr_plan, pr_rewrite, …)
scraper/        # GitHub GraphQL client and bronze-data scrapers
schemas/        # Pydantic models for bronze / silver / gold data layers
mcp_server.py   # FastMCP server exposing pipeline tools over HTTP
```

Data flows bronze → silver → gold:
- **bronze** — raw scraped PR/review JSON from GitHub
- **silver** — normalized, deduplicated review comments
- **gold** — distilled rules, authority scores, test knowledge bases

## Making changes

Open an issue before starting significant work so we can discuss approach. For small fixes, a PR is fine directly.

**Run the judge on your own diff before opening a PR** — that's the whole point of this tool:

```bash
git diff main...HEAD > my.diff
python -m pipeline.judge --repo AMD-AGI_pr_pundit --patch my.diff
```

(You'll need to have scraped and distilled rules for this repo first via `init-repo --repo AMD-AGI/pr_pundit`.)

## Code style

- `ruff` for linting, `black` for formatting — both run via pre-commit
- No type annotations required but they're welcome
- Default to no comments; add one only when the *why* is non-obvious

```bash
ruff check pipeline/ scraper/ schemas/ mcp_server.py
black --check pipeline/ scraper/ schemas/ mcp_server.py
```

## Adding a new MCP tool

1. Implement the pipeline logic in `pipeline/`
2. Add the `@mcp.tool()` wrapper in `mcp_server.py` following the existing pattern
3. Add a row to the tool table in `README.md` and `MCP.md`

## Pipeline stages

Each stage in `pipeline/` is a standalone Python module with a `main()` entry point. Stages read from `data/bronze|silver|gold/{repo_slug}/` and write to the next layer. The MCP server calls these stages directly, not as subprocesses.

## Submitting a PR

- Keep PRs focused — one logical change per PR
- Include a short description of *why*, not just *what*
- If your change affects MCP tool behavior, update `MCP.md` parameter docs
