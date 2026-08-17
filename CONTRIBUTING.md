# Contributing to PR Pundit

Thank you for your interest in contributing to PR Pundit. This document covers
how to set up a development environment, run the test suite, and submit changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Running Tests](#running-tests)
- [Branch and Commit Conventions](#branch-and-commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Adding a New Pipeline Stage](#adding-a-new-pipeline-stage)
- [Adding a New MCP Tool](#adding-a-new-mcp-tool)

---

## Development Setup

```bash
git clone https://github.com/AMD-AGI/pr_pundit.git
cd pr_pundit

# Install with all optional dependencies
pip install -e ".[ui,mcp]"
```

Required environment variables (copy `.env.example` if present):

| Variable | Required for | Description |
|---|---|---|
| `GITHUB_TOKEN` | scraping | GitHub PAT with `repo` scope |
| `LITELLM_BASE_URL` | distill, judge, conform | LiteLLM proxy URL |
| `LITELLM_MASTER_KEY` | distill, judge, conform | LiteLLM proxy auth key |

The MCP server can run locally in stdio mode for development:

```bash
python mcp_server.py
```

---

## Project Layout

```
mcp_server.py          # MCP tool definitions (entry point for IDE integration)
pipeline/              # Core pipeline stages
  distill.py           # Rule distillation from reviewer feedback
  judge.py             # Rule violation detection
  fix.py               # Rewrite loop (judge → fix until clean)
  pr_plan.py           # Seed-to-PR series planner
  pr_prepare.py        # PR description and contributing checklist generation
  suggest_tests.py     # Test/benchmark script suggestion
scraper/               # GitHub data ingestion
  scrape_bronze.py     # Raw PR scraping
schemas/               # Pydantic data schemas
  bronze.py            # Raw GitHub API payloads
  silver.py            # Normalized reviewer threads
  gold.py              # Distilled rules and evidence
data/
  bronze/              # Raw GitHub API payloads (git-ignored)
  silver/              # Normalized reviewer threads (git-ignored)
  gold/                # Distilled rules — only repo_config.yaml is committed
ui/                    # Streamlit UI
k8s/                   # Kubernetes deployment manifests and deploy script
```

---

## Running Tests

```bash
pytest
```

Integration tests that hit the LLM require `LITELLM_BASE_URL` and
`LITELLM_MASTER_KEY` to be set. Unit tests run without them.

To run only unit tests:

```bash
pytest -m "not integration"
```

---

## Branch and Commit Conventions

**Branch naming**: `<type>/<scope>/<short-description>`

| Type | When to use |
|---|---|
| `feat` | New feature or pipeline stage |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behaviour change |
| `test` | New or updated tests |
| `chore` | Build, CI, dependency changes |

Examples:
- `feat/pipeline/add-cochange-signal`
- `fix/mcp/handle-empty-diff`
- `docs/readme/update-mcp-tool-table`

**Commit messages**: follow [Conventional Commits](https://www.conventionalcommits.org/).

```
feat(pipeline): add co-change signal to judge findings
fix(mcp): return 400 when diff is empty instead of 500
docs(readme): add plan_pr_series parameter table
```

---

## Pull Request Process

1. Fork the repository and create a branch following the naming convention above.
2. Make your changes with appropriately scoped commits.
3. Run `pytest` and confirm all tests pass.
4. Open a pull request against `main` with a description that covers:
   - What the change does and why.
   - Which pipeline stages or MCP tools are affected.
   - Any new environment variables or configuration keys introduced.
5. Address reviewer feedback promptly.
6. A maintainer will merge once the PR is approved.

For significant changes (new pipeline stages, new MCP tools, schema changes),
open an issue first to discuss the approach before writing code.

---

## Adding a New Pipeline Stage

Each pipeline stage lives in `pipeline/` as a standalone Python module with a
`__main__` entry point and a corresponding function callable from
`mcp_server.py`.

Steps:

1. Add the module to `pipeline/<stage_name>.py`.
2. Register a CLI entry point in `pyproject.toml` under `[project.scripts]`.
3. If the stage is exposed via MCP, add a tool function in `mcp_server.py`
   following the existing pattern (docstring → parameter schema → handler).
4. Add the stage to `README.md`'s pipeline overview and the MCP tool table.
5. Write tests under `tests/` that cover the happy path and the empty-input edge case.

---

## Adding a New MCP Tool

MCP tools are defined in `mcp_server.py`. Each tool must:

- Have a clear, single-sentence docstring (this becomes the tool description
  shown to IDE assistants).
- Accept and return only JSON-serialisable types.
- Return instructions and patch content for the IDE agent to execute; the MCP
  server must not run `git` or `gh` subprocesses itself.
- Be documented in [MCP.md](MCP.md) with its full parameter reference and a
  usage example.

---

## License

By contributing to PR Pundit, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE).
