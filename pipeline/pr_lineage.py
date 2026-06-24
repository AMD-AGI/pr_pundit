"""
Stage L1 — Lineage tree builder.

Reads bronze JSONL files and constructs LineageTree objects capturing the
evolution of ideas from failed (CLOSED) PRs to a successfully merged PR.

Two passes (in confidence order):

  Pass 1 — Body cross-reference (confidence 1.0)
    Scan MERGED PR bodies for "#N" within 120 chars of supersede keywords.
    If PR #N is CLOSED, add edge N → merged.  Recurse: if the closed PR body
    also mentions another closed PR, extend the tree upward.

  Pass 2 — Revert pairs (confidence 0.9)
    Scan MERGED PR titles for "Revert …".  The reverted merged PR acts as a
    failed intermediate; find the fix PR (next merged PR by same author on
    overlapping files within 30 days).

Output: data/lineage/{owner}_{name}/trees.jsonl — one tree per line.

Usage:
    pr-pundit-lineage --repo vllm-project/vllm [--passes 1,2] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from schemas.lineage import (
    LineageEdge,
    LineageEdgeSignal,
    LineageTree,
    PRNode,
    PRReview,
    ReviewThreadComment,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BRONZE_DIR = DATA_DIR / "bronze"
LINEAGE_DIR = DATA_DIR / "lineage"

# Keywords that, near a PR number in a body, signal supersession/follow-up
_SUPERSEDE_RE = re.compile(
    r"(supersedes?|replaces?|alternative\s+to|instead\s+of|follow[- ]?up\s+to|"
    r"follow[- ]?up\s+#\d+|this\s+replaces?|closes?\s+#\d+|supercedes?|"
    r"based\s+on|re[- ]?submit|re[- ]?open|re[- ]?work|retry|attempt\s+\d+|"
    r"v\d+\s+of|second\s+try|updated\s+version\s+of)",
    re.IGNORECASE,
)

# Match "#N" PR references anywhere in text
_PR_REF_RE = re.compile(r"#(\d+)")

# Match owner/repo#N or bare GitHub PR URLs
_FULL_PR_REF_RE = re.compile(
    r"(?:https?://github\.com/[\w.-]+/[\w.-]+/pull/(\d+)|[\w.-]+/[\w.-]+#(\d+))"
)

# Match revert PR titles
_REVERT_TITLE_RE = re.compile(r'^Revert\s+".*?"\s*\(#(\d+)\)|^Revert\s+#(\d+)', re.IGNORECASE)
_REVERT_BODY_RE = re.compile(r'Reverts\s+(?:\w+/\w+)?#(\d+)', re.IGNORECASE)


# ── bronze loader ────────────────────────────────────────────────────

def _load_bronze(repo_slug: str) -> tuple[dict[int, dict], dict[int, list[dict]], dict[int, list[dict]], dict[int, list[dict]]]:
    """Load bronze data for a repo.

    Returns:
        prs:            {pr_number: pr_record}
        reviews:        {pr_number: [review_record]} — all states
        threads:        {pr_number: [thread_record]}
        files:          {pr_number: [file_record]}
    """
    bronze = BRONZE_DIR / repo_slug
    if not bronze.exists():
        raise FileNotFoundError(f"Bronze directory not found: {bronze}")

    def _load_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    pr_records = _load_jsonl(bronze / "pull_requests.jsonl")
    review_records = _load_jsonl(bronze / "reviews.jsonl")
    thread_records = _load_jsonl(bronze / "review_threads.jsonl")
    file_records = _load_jsonl(bronze / "files.jsonl")

    prs = {r["number"]: r for r in pr_records}
    reviews: dict[int, list[dict]] = defaultdict(list)
    for r in review_records:
        reviews[r["_pr_number"]].append(r)
    threads: dict[int, list[dict]] = defaultdict(list)
    for t in thread_records:
        threads[t["_pr_number"]].append(t)
    files: dict[int, list[dict]] = defaultdict(list)
    for f in file_records:
        files[f["_pr_number"]].append(f)

    logger.info("Loaded %d PRs (%d MERGED, %d CLOSED) from %s",
                len(prs),
                sum(1 for p in prs.values() if p.get("state") == "MERGED"),
                sum(1 for p in prs.values() if p.get("state") == "CLOSED"),
                repo_slug)
    return prs, reviews, threads, files


# ── PRNode builder ───────────────────────────────────────────────────

def _build_node(pr: dict, repo: str, reviews: list[dict], threads: list[dict]) -> PRNode:
    # All review-level comments (CHANGES_REQUESTED, COMMENTED, APPROVED) with a body.
    # Bot authors are excluded — they add noise without architectural signal.
    _BOT_LOGINS = {"github-actions", "codecov", "dependabot", "gemini-code-assist", "claude",
                   "copilot", "deepsource-autofix", "sourcery-ai"}
    pr_reviews = [
        PRReview(
            reviewer=(r.get("author") or {}).get("login", "unknown"),
            state=r.get("state", "COMMENTED"),
            body=r.get("body", "").strip(),
            submitted_at=r.get("submittedAt", ""),
        )
        for r in reviews
        if r.get("body", "").strip()
        and (r.get("author") or {}).get("login", "") not in _BOT_LOGINS
    ]

    # Inline thread comments — preserve path, diff hunk, and all comment bodies.
    thread_comments = []
    for t in threads:
        comments_list = (t.get("comments") or {}).get("nodes", [])
        # Collect non-bot comment bodies; use first comment's hunk for the thread.
        bodies = []
        hunk = ""
        for c in comments_list:
            body = (c.get("body") or "").strip()
            author = (c.get("author") or {}).get("login", "")
            if not hunk:
                hunk = (c.get("diffHunk") or "").strip()
            if body and author not in _BOT_LOGINS:
                bodies.append(body)
        if bodies:
            thread_comments.append(ReviewThreadComment(
                path=t.get("path", ""),
                diff_hunk=hunk,
                comments=bodies,
            ))

    return PRNode(
        repo=repo,
        number=pr["number"],
        title=pr.get("title", ""),
        body=pr.get("body", "") or "",
        author=(pr.get("author") or {}).get("login", "unknown"),
        state=pr.get("state", ""),
        created_at=pr.get("createdAt", ""),
        closed_at=pr.get("closedAt"),
        merged_at=pr.get("mergedAt"),
        files_changed=[],  # filled in separately from files.jsonl
        reviews=pr_reviews,
        review_thread_comments=thread_comments,
        url=pr.get("url", ""),
    )


def _attach_files(node: PRNode, files: list[dict]) -> None:
    node.files_changed = list({f["path"] for f in files if f.get("path")})


# ── Pass 1: body cross-reference ──────────────────────────────────────

def _extract_pr_refs_near_supersede(body: str) -> set[int]:
    """Return PR numbers that indicate supersession/follow-up in body.

    Two signals:
    - A #N or owner/repo#N reference within 150 chars of a supersede keyword.
    - A bare GitHub PR URL anywhere in the body (always a concrete signal).
    """
    found: set[int] = set()
    if not body:
        return found

    # Signal 1: keyword-proximity
    for m in _SUPERSEDE_RE.finditer(body):
        window_start = max(0, m.start() - 150)
        window_end = min(len(body), m.end() + 150)
        window = body[window_start:window_end]
        for ref in _PR_REF_RE.finditer(window):
            found.add(int(ref.group(1)))

    # Signal 2: full GitHub PR URLs or owner/repo#N — always concrete
    for m in _FULL_PR_REF_RE.finditer(body):
        num = m.group(1) or m.group(2)
        if num:
            found.add(int(num))

    return found


def _build_trees_pass1(
    prs: dict[int, dict],
    reviews: dict[int, list[dict]],
    threads: dict[int, list[dict]],
    files: dict[int, list[dict]],
    repo: str,
) -> list[LineageTree]:
    """Body cross-reference pass — confidence 1.0."""
    # Map: closed_pr_number → set of merged_pr_numbers that reference it
    edges_raw: list[tuple[int, int]] = []  # (from_closed, to_merged_or_closed)

    for pr_num, pr in prs.items():
        if pr.get("state") != "MERGED":
            continue
        body = pr.get("body") or ""
        refs = _extract_pr_refs_near_supersede(body)
        for ref in refs:
            if ref in prs and prs[ref].get("state") == "CLOSED":
                edges_raw.append((ref, pr_num))
                logger.debug("Pass 1: MERGED #%d references CLOSED #%d", pr_num, ref)

    # Also check closed PR bodies for references to other closed PRs (chains)
    for pr_num, pr in prs.items():
        if pr.get("state") != "CLOSED":
            continue
        body = pr.get("body") or ""
        refs = _extract_pr_refs_near_supersede(body)
        for ref in refs:
            if ref in prs and prs[ref].get("state") == "CLOSED" and ref != pr_num:
                edges_raw.append((ref, pr_num))
                logger.debug("Pass 1 chain: CLOSED #%d references CLOSED #%d", pr_num, ref)

    if not edges_raw:
        logger.info("Pass 1: no body cross-reference edges found")
        return []

    # Build connected components to form trees
    return _assemble_trees(edges_raw, prs, reviews, threads, files, repo,
                           signal=LineageEdgeSignal.body_cross_reference, confidence=1.0)


# ── Pass 2: revert pairs ──────────────────────────────────────────────

def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build_trees_pass2(
    prs: dict[int, dict],
    reviews: dict[int, list[dict]],
    threads: dict[int, list[dict]],
    files: dict[int, list[dict]],
    repo: str,
) -> list[LineageTree]:
    """Revert pairs pass — confidence 0.9.

    For each MERGED PR whose title starts with 'Revert', extract the reverted
    PR number.  The reverted merged PR acts as the 'failed intermediate'.
    Find the fix PR: next merged PR by same author on overlapping files within
    30 days.
    """
    edges_raw: list[tuple[int, int]] = []

    # Index: author → list of merged PR records (for fix PR lookup)
    by_author: dict[str, list[dict]] = defaultdict(list)
    for pr in prs.values():
        if pr.get("state") == "MERGED":
            author = (pr.get("author") or {}).get("login", "")
            by_author[author].append(pr)

    # File index for overlap checking
    pr_files: dict[int, set[str]] = {
        num: {f["path"] for f in flist if f.get("path")}
        for num, flist in files.items()
    }

    for pr_num, pr in prs.items():
        if pr.get("state") != "MERGED":
            continue
        title = pr.get("title", "")
        m = _REVERT_TITLE_RE.match(title)
        if not m:
            # Also try body
            body = pr.get("body") or ""
            m2 = _REVERT_BODY_RE.search(body)
            reverted_num = int(m2.group(1)) if m2 else None
        else:
            reverted_num = int(m.group(1) or m.group(2))

        if reverted_num is None or reverted_num not in prs:
            continue

        reverted_pr = prs[reverted_num]
        if reverted_pr.get("state") != "MERGED":
            continue

        # The revert PR is now the 'failed intermediate'
        # Find fix: merged by same author, overlapping files, within 30 days after revert
        revert_author = (pr.get("author") or {}).get("login", "")
        revert_merged_at = _parse_date(pr.get("mergedAt"))
        if not revert_merged_at:
            continue

        revert_files = pr_files.get(pr_num, set())
        best_fix: dict | None = None
        best_delta = None

        for candidate in by_author.get(revert_author, []):
            cnum = candidate["number"]
            if cnum == pr_num or cnum == reverted_num:
                continue
            cmerged = _parse_date(candidate.get("mergedAt"))
            if not cmerged:
                continue
            delta = (cmerged - revert_merged_at).total_seconds()
            if delta < 0 or delta > 30 * 86400:
                continue
            overlap = revert_files & pr_files.get(cnum, set())
            if not overlap:
                continue
            if best_delta is None or delta < best_delta:
                best_fix = candidate
                best_delta = delta

        if best_fix:
            # edge: reverted → revert (intermediate) → fix
            edges_raw.append((reverted_num, pr_num))
            edges_raw.append((pr_num, best_fix["number"]))
            logger.debug("Pass 2: Revert chain #%d → #%d → #%d",
                         reverted_num, pr_num, best_fix["number"])

    if not edges_raw:
        logger.info("Pass 2: no revert pairs found")
        return []

    return _assemble_trees(edges_raw, prs, reviews, threads, files, repo,
                           signal=LineageEdgeSignal.revert_pair, confidence=0.9)


# ── tree assembly ─────────────────────────────────────────────────────

def _compute_depth(nodes_in_component: set[int], edges: list[tuple[int, int]]) -> int:
    """Longest path from any leaf to root using iterative topological sort (cycle-safe)."""
    from collections import deque

    # Build adjacency and in-degree, ignoring self-loops
    children: dict[int, set[int]] = defaultdict(set)
    in_degree: dict[int, int] = {n: 0 for n in nodes_in_component}
    for f, t in edges:
        if f in nodes_in_component and t in nodes_in_component and f != t:
            if t not in children[f]:
                children[f].add(t)
                in_degree[t] = in_degree.get(t, 0) + 1

    # Kahn's algorithm — nodes in cycles never reach in_degree 0, so they're skipped
    queue = deque(n for n in nodes_in_component if in_degree.get(n, 0) == 0)
    dist: dict[int, int] = {n: 1 for n in nodes_in_component}

    while queue:
        node = queue.popleft()
        for succ in children.get(node, set()):
            dist[succ] = max(dist.get(succ, 1), dist[node] + 1)
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    return max(dist.values(), default=1)


def _assemble_trees(
    edges_raw: list[tuple[int, int]],
    prs: dict[int, dict],
    reviews: dict[int, list[dict]],
    threads: dict[int, list[dict]],
    files: dict[int, list[dict]],
    repo: str,
    signal: LineageEdgeSignal,
    confidence: float,
) -> list[LineageTree]:
    """Group edges into connected components; identify root; build LineageTree objects."""
    # Union-Find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    all_nodes: set[int] = set()
    for f, t in edges_raw:
        all_nodes.add(f)
        all_nodes.add(t)
        union(f, t)

    # Group by component
    components: dict[int, set[int]] = defaultdict(set)
    for n in all_nodes:
        components[find(n)].add(n)

    repo_slug = repo.replace("/", "_")
    now = datetime.now(timezone.utc).isoformat()
    trees: list[LineageTree] = []

    for component_nodes in components.values():
        # Determine in-degree within component (to find root = node with no outgoing edge
        # that is pointed to by others, i.e. the MERGED terminal)
        has_outgoing: set[int] = set()
        has_incoming: set[int] = set()
        local_edges: list[tuple[int, int]] = []
        for f, t in edges_raw:
            if f in component_nodes and t in component_nodes:
                has_outgoing.add(f)
                has_incoming.add(t)
                local_edges.append((f, t))

        # Root candidates: in component, has incoming, no outgoing (or is MERGED)
        roots = [
            n for n in component_nodes
            if prs.get(n, {}).get("state") == "MERGED"
        ]
        if not roots:
            # Fallback: node with most incoming edges
            roots = [max(component_nodes, key=lambda n: sum(1 for _, t in local_edges if t == n))]

        root = roots[0]  # take first MERGED; typically only one per tree

        # Build node objects
        node_objects: dict[int, PRNode] = {}
        for n in component_nodes:
            if n not in prs:
                continue
            node = _build_node(prs[n], repo, reviews.get(n, []), threads.get(n, []))
            _attach_files(node, files.get(n, []))
            node_objects[n] = node

        if root not in node_objects:
            continue

        edge_objects = [
            LineageEdge(from_pr=f, to_pr=t, signal=signal, confidence=confidence)
            for f, t in local_edges
        ]

        depth = _compute_depth(component_nodes, local_edges)
        repo_slug_safe = repo_slug
        tree_id = f"{repo_slug_safe}-{root}"

        trees.append(LineageTree(
            tree_id=tree_id,
            repo=repo,
            root_pr=root,
            nodes=node_objects,
            edges=edge_objects,
            depth=depth,
            created_at=now,
        ))

    return trees


# ── dedup ─────────────────────────────────────────────────────────────

def _merge_trees(trees: list[LineageTree]) -> list[LineageTree]:
    """Merge trees with the same root PR, combining nodes and edges."""
    by_root: dict[int, LineageTree] = {}
    for tree in trees:
        root = tree.root_pr
        if root not in by_root:
            by_root[root] = tree
        else:
            existing = by_root[root]
            # Merge nodes (nodes from both, no duplicates)
            for num, node in tree.nodes.items():
                if num not in existing.nodes:
                    existing.nodes[num] = node
            # Merge edges (deduplicate by (from, to))
            existing_edge_keys = {(e.from_pr, e.to_pr) for e in existing.edges}
            for edge in tree.edges:
                if (edge.from_pr, edge.to_pr) not in existing_edge_keys:
                    existing.edges.append(edge)
                    existing_edge_keys.add((edge.from_pr, edge.to_pr))
            # Recompute depth
            all_node_nums = set(existing.nodes.keys())
            local_edges = [(e.from_pr, e.to_pr) for e in existing.edges]
            existing.depth = _compute_depth(all_node_nums, local_edges)
    return list(by_root.values())


# ── main ──────────────────────────────────────────────────────────────

def build_lineage_trees(
    repo: str,
    passes: list[int] | None = None,
    limit: int | None = None,
) -> list[LineageTree]:
    """Build and return LineageTree objects for a repo.

    Args:
        repo:   "owner/name"
        passes: which passes to run (default [1, 2])
        limit:  cap number of trees (for testing)
    """
    if passes is None:
        passes = [1, 2]

    owner, name = repo.split("/", 1)
    repo_slug = f"{owner}_{name}"

    prs, reviews, threads, files = _load_bronze(repo_slug)

    all_trees: list[LineageTree] = []

    if 1 in passes:
        logger.info("Running Pass 1 (body cross-reference) for %s …", repo)
        t1 = _build_trees_pass1(prs, reviews, threads, files, repo)
        logger.info("Pass 1: %d trees found", len(t1))
        all_trees.extend(t1)

    if 2 in passes:
        logger.info("Running Pass 2 (revert pairs) for %s …", repo)
        t2 = _build_trees_pass2(prs, reviews, threads, files, repo)
        logger.info("Pass 2: %d trees found", len(t2))
        all_trees.extend(t2)

    # Merge trees with same root (from different passes)
    merged = _merge_trees(all_trees)
    # Filter: keep trees that have at least one CLOSED node (actual failed attempt)
    merged = [t for t in merged if any(n.state == "CLOSED" for n in t.nodes.values())]
    # Sort: deepest trees first (richest signal)
    merged.sort(key=lambda t: t.depth, reverse=True)

    logger.info("Total trees after merge+filter: %d (depth range %d–%d)",
                len(merged),
                merged[-1].depth if merged else 0,
                merged[0].depth if merged else 0)

    if limit:
        merged = merged[:limit]
        logger.info("Applied --limit %d → %d trees", limit, len(merged))

    return merged


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build PR lineage trees from bronze data")
    p.add_argument("--repo", required=True, help="owner/name")
    p.add_argument("--passes", default="1,2",
                   help="Comma-separated pass numbers to run (default: 1,2)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max trees to output (for testing)")
    args = p.parse_args()

    passes = [int(x.strip()) for x in args.passes.split(",")]
    trees = build_lineage_trees(args.repo, passes=passes, limit=args.limit)

    owner, name = args.repo.split("/", 1)
    repo_slug = f"{owner}_{name}"
    out_dir = LINEAGE_DIR / repo_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trees.jsonl"

    with open(out_path, "w") as f:
        for tree in trees:
            f.write(json.dumps(tree.to_dict(), default=str) + "\n")

    logger.info("Wrote %d trees → %s", len(trees), out_path)
    print(f"\nSummary:")
    print(f"  Trees: {len(trees)}")
    if trees:
        depth_hist: dict[int, int] = defaultdict(int)
        for t in trees:
            depth_hist[t.depth] += 1
        for d in sorted(depth_hist):
            print(f"  depth={d}: {depth_hist[d]} trees")


if __name__ == "__main__":
    main()
