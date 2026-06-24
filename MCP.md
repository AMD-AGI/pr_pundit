# PR Pundit — MCP Setup

Connect PR Pundit's MCP server to your IDE so you can judge diffs, get test suggestions, and open PRs without leaving the editor.

## Option A — Self-hosted server (Claude Code, recommended)

After deploying the server (see [Deploying the server](#deploying--updating-the-server)), add it to Claude Code:

```bash
claude mcp add --transport http pr-pundit http://<your-server>/pr-pundit-mcp/mcp
```

Or add manually to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "pr-pundit": {
      "type": "http",
      "url": "http://<your-server>/pr-pundit-mcp/mcp"
    }
  }
}
```

Verify with `/mcp` in Claude Code — you should see `pr-pundit` with its tools listed.

## Option B — Local stdio (Claude Code, for development)

Run the server in your repo directory:

```bash
cd /path/to/pr-pundit
uv sync --extra mcp
source .env          # sets LITELLM_BASE_URL, GITHUB_TOKEN
```

**`~/.claude/settings.json`**
```json
{
  "mcpServers": {
    "pr-pundit": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "mcp-server"],
      "cwd": "/path/to/pr-pundit",
      "env": {
        "LITELLM_BASE_URL": "http://localhost:4000"
      }
    }
  }
}
```

## Option C — Cursor

Cursor reads MCP config from `~/.cursor/mcp.json` (global, all projects) or `.cursor/mcp.json` (project-local).

**Global setup** — works in every Cursor project:

1. Open `~/.cursor/mcp.json` (create it if it doesn't exist) and add:

```json
{
  "mcpServers": {
    "pr-pundit": {
      "url": "http://<your-server>/pr-pundit-mcp/mcp"
    }
  }
}
```

2. In Cursor, open **Settings → Cursor Settings → MCP** and confirm `pr-pundit` appears with a green status dot.

**Project-local setup** — only active when this repo is open:

Create `.cursor/mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "pr-pundit": {
      "url": "http://<your-server>/pr-pundit-mcp/mcp"
    }
  }
}
```

Then restart Cursor (or reload the window with `Ctrl+Shift+P → Developer: Reload Window`).

> **Note:** Cursor reads the `url` field only — omit `type`. The `.mcp.json` at the repo root (Option D) uses `"type": "http"` which is ignored by Cursor but required by Claude Code; both point to the same `/mcp` endpoint.

## Option D — VS Code / project-shared config

A `.mcp.json` at the repo root works for Claude Code and is already committed:

```json
{
  "mcpServers": {
    "pr-pundit": {
      "type": "http",
      "url": "http://<your-server>/pr-pundit-mcp/mcp"
    }
  }
}
```

Clone the repo and open it — Claude Code picks this up automatically.

## Verify it works

In your IDE chat:
```
Use the get_rules tool for https://github.com/vllm-project/vllm
```

You should get a list of distilled reviewer rules.

---

## Available tools

| Tool | What to say |
|------|-------------|
| `get_rules` | "Show me the rules for vllm-project/vllm" |
| `judge_diff` | "Judge my diff against the vllm rules" |
| `conform_diff` | "Fix my diff to pass the vllm rules" |
| `suggest_tests` | "Suggest benchmarks for my ROCm changes" |
| `prepare_pr` | "Write a PR description for my diff" |
| `review_pr` | "Review this PR: https://github.com/vllm-project/vllm/pull/123" |
| `plan_pr_series` | "Plan PRs from this seed (see usage below)" |
| `get_plan` | "Fetch the plan for plan_id abc123" |

The tools detect the repo automatically from `git remote get-url origin`. If you're in a fork, tell the assistant which upstream to use: _"judge against vllm-project/vllm rules"_.

---

### `plan_pr_series` — creating PRs from a seed

`plan_pr_series` accepts three kinds of input via `seed_url`:

| Input type | Example |
|---|---|
| GitHub PR URL | `https://github.com/vllm-project/vllm/pull/38646` |
| GitHub tree URL (seed folder) | `https://github.com/your-org/your-repo/tree/main/MyFeature` |
| Local folder path | `/home/user/my-feature-seed` |

When a PR URL is given, the tool fetches the PR's unified diff and description and uses those as the seed — no separate patch files needed.

#### Parameters

| Parameter | Description |
|---|---|
| `seed_url` | PR URL, seed folder URL, or local path (required) |
| `upstream_repo_url` | The canonical upstream repo whose rules and gold data drive all LLM stages. Auto-detected; override when detection is wrong or when the seed comes from a fork. Example: `https://github.com/vllm-project/vllm` |
| `staging_repo_url` | Where to fork/push and open the PR. Defaults to `upstream_repo_url`. Override to use a personal fork as a staging area while still using the upstream's rules. Example: `https://github.com/yourname/vllm` |
| `blurb` | Short one-line description of the PR series (supplements the README/PR body) |
| `notes` | Free-form guidance forwarded to every LLM stage — planner, test suggester, and PR preparer. Use for hardware targets, framing constraints, or any submitter context. |
| `force` | Skip the duplicate-PR check and proceed anyway |
| `target_tier` | `"fast-adoption"` (vLLM/SGLang) or `"long-term"` (aiter) — filters auto-detection |
| `seed_github_token` | GitHub token used **only** to fetch the seed repo. Supply when the seed is in a private org repo the server's PAT cannot read. See [Private seed repos](#private-seed-repos) below. |

> **upstream vs staging:** `upstream_repo_url` controls which rules/gold data to use. `staging_repo_url` controls where the branch is pushed and the PR is opened. When staging is a repo you already own, the tool pushes directly without re-forking.

#### Private seed repos

If the seed folder lives in a private org, the server's built-in PAT may not have read access. The tool will return a `SEED_AUTH_REQUIRED` error with instructions for the IDE agent.

**The IDE agent handles this automatically** — it runs `gh auth token` locally and re-calls the tool with the result as `seed_github_token`. The token is:
- held only in memory for the duration of the pipeline run
- used exclusively to fetch the seed (push/PR operations always use the server's own credentials)
- never written to log files or persisted to disk

> **Do not ask the user to paste a token in chat.** The IDE agent obtains it silently via `gh auth token`.

#### Example prompts

```
# From a PR URL, with hardware framing notes
plan_pr_series(
  seed_url="https://github.com/vllm-project/vllm/pull/38646",
  notes="prepare this PR for ROCm MI300X"
)

# Open the PR on a personal fork (staging) while using vLLM rules (upstream)
plan_pr_series(
  seed_url="https://github.com/vllm-project/vllm/pull/38646",
  upstream_repo_url="https://github.com/vllm-project/vllm",
  staging_repo_url="https://github.com/yourname/vllm",
  notes="prepare for ROCm MI300X"
)

# With a local seed folder
plan_pr_series(
  seed_url="/home/user/my-patch-folder",
  notes="keep changes minimal — targeting a stable branch"
)
```

#### How it works (async, two-phase)

**Phase 1 — upstream detection** (fast, synchronous): The tool fetches the seed, detects the upstream repo, and returns a confirmation prompt. The assistant surfaces the detected repo and asks you to confirm before running the expensive pipeline.

**Phase 2 — full pipeline** (background, 5–15 min): After you confirm the upstream, the tool launches the LLM pipeline in a background thread and returns immediately with a `run_id` and `plan_id`. Your IDE assistant then:

1. Starts a Monitor on the returned log file path to watch pipeline progress
2. Calls `get_plan(plan_id)` when `"artifacts ready"` appears in the log
3. Executes the git/gh commands from the plan locally (fork → push → open PRs) using your credentials

The MCP server does all LLM and analysis work. Your IDE does all git/gh shell work — no org repo access is needed on the server side.

---

## Deploying / updating the server

```bash
# From the repo root — builds Docker image, deploys to k8s
./k8s/deploy.sh

# Check logs after deploy
kubectl logs -f deployment/pr-pundit -c pr-pundit-mcp
```

## Required environment variables (for local development)

| Variable | Description |
|----------|-------------|
| `LITELLM_BASE_URL` | LiteLLM proxy URL, e.g. `http://localhost:4000` |
| `LITELLM_MASTER_KEY` | API key for the LiteLLM proxy |
| `GITHUB_TOKEN` | GitHub PAT with `repo` scope (for fetching PRs and opening PRs) |

Copy `.env.example` to `.env` and fill in your values.
