# PR Pundit

PR Pundit learns the review standards of a GitHub repository from its merged
PR history, then judges and conforms your code against those standards before
you open a PR.

Rules are distilled from real reviewer feedback — not generic best practices —
so findings reflect what the actual maintainers of that repo care about.

## Using PR Pundit

### From your IDE (recommended)

Connect PR Pundit's MCP server and your IDE assistant can judge or conform your
changes without leaving the editor. Add this to your MCP config (e.g. Claude
Code `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "pr-pundit": {
      "url": "http://<server>/pr-pundit-mcp/mcp"
    }
  }
}
```

Then ask your assistant:
> "Create an upstream PR bundle for the working directory to vllm. Use xxx/vllm github repository as staging repo"
> "Judge my changes against the repo rules"
> "Conform my diff to the repo standards before I open this PR"

The assistant detects the repo from `git remote get-url origin` and gets your
diff from `git diff main...HEAD` automatically — nothing to configure per
project.

Tools available:

| Tool | What it does |
|---|---|
| `get_rules(repo_url)` | Show the distilled rules for the repo |
| `judge_diff(diff, repo_url)` | Find violations with file, line, and fix hint |
| `conform_diff(diff, repo_url)` | Rewrite the diff to satisfy all rules |
| `suggest_tests(diff, repo_url)` | Generate runnable benchmark/accuracy test scripts |
| `prepare_pr(diff, repo_url)` | Write a PR description grounded in measured results |
| `review_pr(pr_url)` | Review open reviewer comments and suggest replies |
| `upload_seed(patches, readme=…)` | Upload local patch files so `plan_pr_series` can use them as a seed |
| `plan_pr_series(seed_url, upstream_repo_url=…, staging_repo_url=…, notes=…)` | Split a GitHub PR, seed folder, or local path into a focused PR series; `upstream_repo_url` sets which repo's rules to use; `staging_repo_url` sets where the PR is opened (defaults to upstream) |
| `get_plan(plan_id)` | Fetch a completed plan and push instructions |
| `stop_pipeline(run_id, stop_token)` | Cancel a running `plan_pr_series` pipeline |

See [MCP.md](MCP.md) for full parameter reference and usage examples.

The assistant reads your repo URL from `git remote get-url origin` to identify
which rules to load. If you work in a fork, `origin` points at the fork rather
than the upstream repo — just tell the assistant which repo to use:

> "Judge my changes against the vllm-project/vllm rules"

It will pass the upstream URL as an override so the correct rules are loaded.

`conform_diff` runs a judge-in-the-loop agent that iterates until violations
are resolved. It can take 1–3 minutes; the SSE connection keeps the IDE from
timing out.

### From the CLI

```bash
# Judge a diff
python -m pipeline.judge --repo owner_name --patch changes.diff --model claude-sonnet-4

# Conform a diff (rewrite until rules are satisfied)
python -m pipeline.fix --repo owner_name --patch changes.diff

# Write judge output as JSON
python -m pipeline.judge --repo owner_name --patch changes.diff --json-out results.json
```

Each finding includes exact file and line location, what the violation is, a
fix hint, and severity (`blocker | strong | advisory`).

### From the UI

```bash
pip install -e ".[ui]"
streamlit run ui/app.py --server.port 8501
```

Four pages are available from the sidebar:

| Page | What it does |
|---|---|
| **Rule Review** | Browse, edit, and accept/reject distilled rules |
| **Judge Dashboard** | Paste a diff or PR URL and run the judge interactively; Fix It agent iterates until violations resolve |
| **Test Suggestions** | Generate runnable benchmark/accuracy scripts for a PR diff |
| **Architecture Principles** | Browse the LLM-discovered architecture principles mined from PR lineage trees |

---

## Adding a New Repo

To teach PR Pundit about a repo, run the distillation pipeline once. It scrapes
merged PRs, extracts reviewer feedback, and distills it into verifiable rules.

```bash
# 1. Scrape merged PRs from GitHub
python -m scraper.scrape_bronze --repo owner/name

# 2. Normalize reviewer threads
python -m pipeline.normalize --repo owner_name

# 3. Configure focus areas (see data/gold/vllm-project_vllm/repo_config.yaml)
#    Create data/gold/{owner_name}/repo_config.yaml

# 4. Score reviewer authority
python -m pipeline.authority --repo owner_name

# 5. Distill rules
python -m pipeline.distill --repo owner_name --model claude-sonnet-4 --workers 10

# 6. Review and accept rules in the UI
streamlit run ui/app.py --server.port 8501
```

`repo_config.yaml` tells the distiller what the repo is about so it filters
out generic feedback and keeps only repo-specific standards:

```yaml
name: owner/name
description: "What this repo does"
focus_keywords:
  - keyword1
focus_areas:
  - Domain-specific concern
trusted_reviewers:
  - github-login
```

Reviewer authority is computed from `merged_prs * 2 + reviews_given` and used
to weight feedback from core maintainers over occasional contributors.

Distillation checkpoints progress — kill and re-run to resume. To start over:

```bash
rm data/gold/owner_name/{filter,cluster,promote}_checkpoint.jsonl
```

All distilled rules start as `pending`. Review them before they are enforced:

```bash
python -m pipeline.review --repo owner_name --list --status pending
python -m pipeline.review --repo owner_name --accept RULE_ID
python -m pipeline.review --repo owner_name --reject RULE_ID --reason "too broad"
```

---

## Test Suggestion (Recipe Pipeline)

PR Pundit can suggest runnable benchmark and accuracy tests for a PR before it
is submitted. The suggestions are grounded in a knowledge base of test recipes
distilled from the repo's own benchmark scripts, related recipe repos, and
reviewer expectations mined from past PR discussions.

### How it works

```
scrape-recipes → bronze_recipes/   (raw benchmark files)
      ↓
normalize-recipes → silver_recipes/  (structured recipe records)
      ↓
distill-recipes → gold/{repo}/test_knowledge.json  (deduplicated recipe KB)
      ↓
suggest-tests --patch diff.patch  →  runnable scripts + PR description template
```

The knowledge base is split by test category (`throughput`, `accuracy`,
`latency`, `kernel`, `e2e`). Within each category, similar recipes are
clustered using embeddings and merged by LLM into canonical templates.
Reviewer expectations (what kinds of tests past reviewers demanded for each PR
type) are mined from bronze PR reviews and stored alongside recipes.

### Building the knowledge base — orchestrated (recommended)

The pipeline orchestrator runs all stages in order, resumes from failures, and
sends Teams notifications. Configure everything in `pipeline_config.yaml`:

```yaml
# Scraped once, shared across all target repos at query time
supplemental_repos:
  - ROCm/aiter

# Each target gets its own KB; recipes_repo is baked in at distillation time
target_repos:
  vllm-project/vllm:
    recipes_repo: vllm-project/recipes
  sgl-project/sglang:
    recipes_repo: sgl-project/sgl-cookbook

distill:
  model: claude-sonnet-4-6
  workers: 20
  embed_model: instructor-xl
```

```bash
# Check what will run
run-pipeline --dry-run

# Detect work already done from previous manual runs (checks gold KB output)
run-pipeline --detect

# Run everything (resumes automatically on failure)
nohup run-pipeline > pipeline.out 2>&1 &

# Check progress
run-pipeline --list

# Re-run a single step
run-pipeline --step distill_recipes:vllm-project/vllm

# Start from scratch
run-pipeline --restart
```

State is saved in `pipeline_state.json` — kill and re-run to resume.
Detection is based on `test_knowledge.json` only (written atomically at the
end of distillation) so partial runs don't get incorrectly marked as done.

### Building the knowledge base for a repo — manual

```bash
# 1. Scrape benchmark scripts from the repo and its recipe companion
scrape-recipes --repo vllm-project/vllm

# 2. (Optional) Scrape supplemental sources scraped independently
scrape-recipes --repo ROCm/aiter --recipes-repo ROCm/aiter
scrape-recipes --repo ROCm/composable_kernel --recipes-repo ROCm/composable_kernel

# 3. Normalize into structured recipe records
normalize-recipes --repo vllm-project/vllm

# 4. Distill into the knowledge base (parallel, ~20 min for vllm)
distill-recipes --repo vllm-project/vllm --workers 20 --embed-model instructor-xl

# 5. Build supplemental KBs (scraped independently, merged at query time)
normalize-recipes --repo ROCm/aiter
distill-recipes --repo ROCm/aiter --workers 20
```

The distiller runs in two passes per test category:

1. **Extract** — batches of 10 silver records → LLM extracts structured recipes
   (parallel, 20 workers)
2. **Consolidate** — embeddings cluster near-duplicate recipes → LLM merges
   each cluster into one canonical recipe (complete-linkage at cosine 0.82)

### Supplemental knowledge bases

Repos like `ROCm/aiter` and `ROCm/composable_kernel` contain ROCm-native kernel
benchmarks not present in the vllm repo itself. They are scraped and distilled
independently (once), then declared in `repo_config.yaml`:

```yaml
test_guidance:
  supplemental_knowledge:
    - ROCm/aiter        # merged at query time for vllm and sglang
    # vllm-project/vllm and sgl-project/sglang used by InferenceX
```

At suggestion time, supplemental recipes are merged into the selection pool and
labelled with their source. The code generator is instructed to adapt their
templates to the target repo's API.

### Suggesting tests for a PR

```bash
suggest-tests --repo vllm-project/vllm --patch changes.diff

# With context about what you are testing
suggest-tests --repo vllm-project/vllm --patch changes.diff \
    --blurb "Testing AMD MI300X FP8 GEMM kernel optimization"

# Save output
suggest-tests --repo vllm-project/vllm --patch changes.diff --out tests.json
```

Output includes:

- **scripts** — complete runnable Python/shell scripts with a `how_to_run` command
  and `what_to_report` guide for pasting results into the PR description
- **pr_description_template** — markdown table the author can paste directly

### Evaluating the knowledge base

```bash
# Full eval: runs the 3-stage pipeline on sampled PRs with known test evidence
eval-recipes --repo vllm-project/vllm --model claude-sonnet-4-6

# Fast mode: 1 LLM call per PR instead of 3 (good for iteration)
eval-recipes --repo vllm-project/vllm --fast
```

The evaluator samples PRs where the author already included benchmark results
in the description, runs the test suggester on each, and scores whether the
suggestions would have matched the actual tests that were run.

### Repo config for test guidance

`data/gold/{owner_name}/repo_config.yaml` controls what the test suggester
knows about the repo's hardware environment and platform constraints:

```yaml
test_guidance:
  primary_hardware:
    - MI300X    # primary benchmark target
    - H100      # cross-platform validation

  platform_notes:
    - ROCm benchmarks use --device rocm; do NOT use CUDA_VISIBLE_DEVICES
    - aiter ops are ROCm-only; include import guards in generated scripts

  known_benchmark_scripts:
    - benchmarks/benchmark_throughput.py
    - benchmarks/kernels/benchmark_fp8_gemm.py

  supplemental_knowledge:
    - ROCm/aiter
```

Platform notes flow into all three suggestion stages so the analyzer correctly
identifies hardware targets, the selector prefers hardware-matched recipes, and
the generator emits the right flags and import guards.

---

## Seed-to-PR: Automated PR Series from a Seed Folder

PR Pundit can take a seed folder (README + patch files) and produce a series of
focused, independently-mergeable upstream PRs — entirely from your IDE.

```
# In your IDE assistant:
"Plan PRs from this seed: /path/to/my-feature-seed"
```

The pipeline runs server-side and returns immediately. Your IDE monitors progress
and presents the full plan when ready. You then push with a single command.

**What the pipeline does:**

1. **Intent extraction** (DSPy RLM agent) — reads the seed README and fetches upstream
   architecture files to extract precise objectives and exclude incidental changes
2. **Upstream reality check** (DSPy RLM agent) — traces imports in the target repo to
   confirm each objective isn't already implemented
3. **Layer separation** — classifies files by layer (model, compiler_pass, kernel, test)
   and checks whether model-layer changes are required or can be replaced by a compiler pass
4. **Planning** — splits confirmed objectives into N atomic PRs grouped by reviewer boundary
5. **Rewrite with feedback loops** — generates per-PR diffs with three self-correcting loops:
   - *Rewrite judge loop*: judge violations → targeted fix per file (up to 3 retries)
   - *Layer audit loop*: model-layer leakage → rewrite with audit hints (up to 3 iters)
   - *Critic loop*: structural issues (symbol drop, empty diff) → targeted fix (up to 2 iters)
6. **Handoff** — benchmark test scripts + PR descriptions ready for your review before push

See [DESIGN.md](DESIGN.md) for the full stage map and [LAYER_SEPARATION.md](LAYER_SEPARATION.md)
for the compiler-pass vs. model-layer decision logic.

---

## Architecture Principles (Meta-Learning)

PR Pundit mines PR lineage trees — chains of failed attempts that eventually
led to a merged PR — to discover architectural principles that no set of regex
or LLM rules could encode. These are higher-level, executable checks applied
incrementally during the rewrite loop.

### How it works

```
scrape-bronze --include-closed   # scrape CLOSED (rejected) PRs alongside MERGED
      ↓
pr-pundit-lineage --repo owner/name   # build lineage trees from PR evolution chains
      ↓
data/lineage/{repo}/trees.jsonl       # LineageTree: failed attempts → merged PR

      ↓
distill-design-rules --repo owner/name   # supervisor + sequential LLM distillation
      ↓
data/lineage/audit_harnesses.json        # global harness bank (cross-repo)
```

**Lineage tree**: one merged PR as root; closed PRs that were explicitly
superseded or reverted as leaves. Chain depth reflects how many failed attempts
preceded the correct approach — deeper chains signal harder architectural
problems.

**Supervisor agent**: for each tree, an LLM identifies the architectural
principle that would have guided the failed PRs toward the merged approach —
one principle per tree, conditioned on all existing harnesses so it only
produces genuinely new ones.

**Sequential LLM distillation**: candidates are reviewed one-by-one (deepest
chains first) against the growing bank. The LLM answers: "does this add a new
architectural dimension not already covered?" Keep or discard — no embeddings,
no thresholds. The result is a compact, non-redundant bank (~46 harnesses from
88 vLLM candidates).

### CLI

```bash
# Full run: build lineage trees, run supervisor, distill
distill-design-rules --repo vllm-project/vllm

# Build trees + supervisor only (inspect candidates before distilling)
distill-design-rules --repo vllm-project/vllm --supervisor-only

# Distill from existing candidates (skip tree rebuild)
distill-design-rules --repo vllm-project/vllm --distill-only

# Dry run (show what would be admitted without writing)
distill-design-rules --repo vllm-project/vllm --dry-run
```

### Harness bank

Each harness encodes:
- **name** — short kebab-case identifier (e.g. `compiler-pass-locus`)
- **description** — what architectural anti-pattern this checks
- **relevance_criteria** — when to apply (the LLM uses this to select harnesses per PR)
- **audit_prompt_template** — rendered at runtime with `{diff}`, `{intent}`, `{files_changed}`; returns `{"hints": [...], "clean": true/false}`
- **lineage_refs** — provenance: which PR chains produced this harness, with clickable links to failed and merged PRs
- **examples** — anti-pattern / correct-pattern pairs from real PR history

### Runtime integration

During `create_pr_from_seed`, after each rewrite pass:

1. One LLM call selects which harnesses are relevant to this PR series' intent and files
2. Each selected harness runs its `audit_prompt_template` against the diff → returns hints
3. Hints feed back into a targeted rewrite of just the affected PR
4. The existing layer audit runs last as the wholistic check

### Browsing harnesses

The **Architecture Principles** page in the UI lets you browse the full bank:
filter by repo, chain depth, or free-text search; click through to the
failed/merged PR links on GitHub.

---

## Deploying the Server

The server hosts the MCP endpoint and the Streamlit UI. It runs on Kubernetes
using the existing cluster.

```bash
./k8s/deploy.sh
```

- UI: `http://<cluster-ip>/pr-pundit`
- MCP: `http://<cluster-ip>/pr-pundit-mcp/mcp`

The MCP server is a sidecar container in the same pod — no separate image or
deploy step. Both containers share the gold data volume.

### Run MCP server locally

For development, the MCP server can run in stdio mode against a local
`data/gold/` directory. You need to have run the full pipeline for each repo
first, or copy `data/gold/` from the server.

```bash
pip install -e ".[mcp]"
python mcp_server.py
```

IDE config for stdio mode:

```json
{
  "mcpServers": {
    "pr-pundit": {
      "command": "python",
      "args": ["/path/to/pr-scraper/mcp_server.py"]
    }
  }
}
```

---

## Environment Variables

| Variable | Required for | Description |
|---|---|---|
| `GITHUB_TOKEN` | scraping | GitHub PAT with `repo` scope |
| `LITELLM_BASE_URL` | distill, judge, conform | LiteLLM proxy URL |
| `LITELLM_MASTER_KEY` | distill, judge, conform | LiteLLM proxy auth key |
| `TEAMS_WEBHOOK_URL` | optional | Microsoft Teams webhook for notifications |

## Install

```bash
pip install -e .           # core pipeline + CLI
pip install -e ".[ui]"     # adds Streamlit UI
pip install -e ".[mcp]"    # adds MCP server
```

## Data Layout

```
data/
├── bronze/   # raw GitHub API payloads
├── silver/   # normalized reviewer threads
└── gold/     # distilled rules (what the judge runs against)
```

## Schemas

- [`schemas/bronze.py`](schemas/bronze.py) — PR, commit, file, review, thread, comment
- [`schemas/silver.py`](schemas/silver.py) — normalized threads and review examples
- [`schemas/gold.py`](schemas/gold.py) — rules, evidence clusters, verifier specs
