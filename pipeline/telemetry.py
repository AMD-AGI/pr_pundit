"""
Lightweight usage telemetry for PR Pundit MCP server.

Writes to SQLite at $DATA_DIR/telemetry.db (NFS-backed in k8s).
No external dependencies — stdlib only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("DATA_DIR", "/app/data")) / "telemetry.db"

# Carries the current tool_calls.id down into LLM calls without changing signatures.
CURRENT_CALL_ID: ContextVar[int | None] = ContextVar("current_call_id", default=None)

_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    tool        TEXT    NOT NULL,
    repo_slug   TEXT,
    developer   TEXT,
    forwarded_ip TEXT,
    socket_ip   TEXT,
    client      TEXT,
    duration_ms INTEGER,
    success     INTEGER NOT NULL,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS token_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    tool        TEXT    NOT NULL,
    model       TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    call_id     INTEGER REFERENCES tool_calls(id)
);

-- One row per plan_pr_series invocation. Captures the key inputs and outcome.
CREATE TABLE IF NOT EXISTS plan_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    run_id          TEXT    NOT NULL,
    plan_id         TEXT,
    developer       TEXT,
    client          TEXT,
    seed_url        TEXT,
    upstream_repo   TEXT,
    staging_repo    TEXT,
    tokens_in       INTEGER DEFAULT 0,
    tokens_out      INTEGER DEFAULT 0,
    wall_time_s     REAL,
    outcome         TEXT,   -- 'done' | 'error' | 'blocked'
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_tc_ts    ON tool_calls(ts);
CREATE INDEX IF NOT EXISTS idx_tc_dev   ON tool_calls(developer);
CREATE INDEX IF NOT EXISTS idx_tc_tool  ON tool_calls(tool);
CREATE INDEX IF NOT EXISTS idx_tu_ts    ON token_usage(ts);
CREATE INDEX IF NOT EXISTS idx_pr_ts    ON plan_runs(ts);
CREATE INDEX IF NOT EXISTS idx_pr_dev   ON plan_runs(developer);
CREATE INDEX IF NOT EXISTS idx_pr_seed  ON plan_runs(seed_url);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    try:
        with _connect() as con:
            con.executescript(_DDL)
        logger.info("Telemetry DB ready at %s", DB_PATH)
    except Exception:
        logger.exception("Telemetry DB init failed — metrics will not be recorded")


def developer_id(forwarded_ip: str, socket_ip: str, user_agent: str) -> str:
    """Stable 16-char hex derived from (trusted) forwarded IP + user-agent."""
    identity_ip = forwarded_ip or socket_ip
    raw = f"{identity_ip}|{user_agent}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_client(user_agent: str) -> str:
    ua = user_agent.lower()
    if "cursor" in ua:
        return "cursor"
    if "claude-code" in ua or "claude_code" in ua:
        return "claude-code"
    return "unknown"


def record_call(
    *,
    tool: str,
    repo_slug: str | None,
    developer: str | None,
    forwarded_ip: str | None,
    socket_ip: str | None,
    client: str,
    duration_ms: int,
    success: bool,
    error: str | None,
) -> int | None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as con:
            cur = con.execute(
                """INSERT INTO tool_calls
                   (ts, tool, repo_slug, developer, forwarded_ip, socket_ip,
                    client, duration_ms, success, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ts, tool, repo_slug, developer, forwarded_ip, socket_ip,
                 client, duration_ms, int(success), error),
            )
            return cur.lastrowid
    except Exception:
        logger.exception("telemetry.record_call failed")
        return None


def record_tokens(
    *,
    call_id: int | None,
    tool: str,
    model: str | None,
    tokens_in: int,
    tokens_out: int,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as con:
            con.execute(
                """INSERT INTO token_usage (ts, tool, model, tokens_in, tokens_out, call_id)
                   VALUES (?,?,?,?,?,?)""",
                (ts, tool, model, tokens_in, tokens_out, call_id),
            )
    except Exception:
        logger.exception("telemetry.record_tokens failed")


def record_plan_run(
    *,
    run_id: str,
    plan_id: str | None,
    developer: str | None,
    client: str | None,
    seed_url: str,
    upstream_repo: str | None,
    staging_repo: str | None,
    wall_time_s: float,
    outcome: str,
    error: str | None = None,
) -> None:
    """Record one plan_pr_series invocation with its aggregate token usage."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _connect() as con:
            # Sum all token_usage rows produced during this run (linked via plan_id prefix
            # on tool name is unreliable — use wall-clock window starting ~wall_time_s ago).
            row = con.execute(
                """SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0)
                   FROM token_usage
                   WHERE ts >= datetime('now', ? || ' seconds')""",
                (f"-{wall_time_s + 5:.0f}",),
            ).fetchone()
            tokens_in, tokens_out = (row[0], row[1]) if row else (0, 0)
            con.execute(
                """INSERT INTO plan_runs
                   (ts, run_id, plan_id, developer, client, seed_url, upstream_repo,
                    staging_repo, tokens_in, tokens_out, wall_time_s, outcome, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, run_id, plan_id, developer, client, seed_url, upstream_repo,
                 staging_repo, tokens_in, tokens_out, round(wall_time_s, 1), outcome, error),
            )
    except Exception:
        logger.exception("telemetry.record_plan_run failed")


def analytics_summary(days: int = 30) -> dict:
    """Return aggregated analytics for the last N days."""
    try:
        with _connect() as con:
            con.row_factory = sqlite3.Row
            since = f"-{days} days"

            # Tool call summary
            tool_rows = con.execute(
                """SELECT tool, COUNT(*) as calls,
                          SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as ok,
                          ROUND(AVG(duration_ms)) as avg_ms
                   FROM tool_calls WHERE ts >= datetime('now', ?)
                   GROUP BY tool ORDER BY calls DESC""",
                (since,),
            ).fetchall()

            # Unique developers per client
            dev_rows = con.execute(
                """SELECT client, COUNT(DISTINCT developer) as unique_devs,
                          COUNT(*) as calls
                   FROM tool_calls WHERE ts >= datetime('now', ?)
                   GROUP BY client ORDER BY calls DESC""",
                (since,),
            ).fetchall()

            # Token usage by model
            token_rows = con.execute(
                """SELECT model, SUM(tokens_in) as tin, SUM(tokens_out) as tout,
                          COUNT(*) as llm_calls
                   FROM token_usage WHERE ts >= datetime('now', ?)
                   GROUP BY model ORDER BY (tin+tout) DESC""",
                (since,),
            ).fetchall()

            # plan_pr_series runs — seed URLs, staging repos, outcomes
            plan_rows = con.execute(
                """SELECT seed_url, upstream_repo, staging_repo, developer, client,
                          tokens_in, tokens_out, wall_time_s, outcome, ts
                   FROM plan_runs WHERE ts >= datetime('now', ?)
                   ORDER BY ts DESC LIMIT 100""",
                (since,),
            ).fetchall()

            plan_stats = con.execute(
                """SELECT outcome, COUNT(*) as n,
                          ROUND(AVG(tokens_in+tokens_out)) as avg_tokens,
                          ROUND(AVG(wall_time_s)) as avg_wall_s
                   FROM plan_runs WHERE ts >= datetime('now', ?)
                   GROUP BY outcome""",
                (since,),
            ).fetchall()

            # Top seed URLs
            top_seeds = con.execute(
                """SELECT seed_url, COUNT(*) as n FROM plan_runs
                   WHERE ts >= datetime('now', ?) AND seed_url IS NOT NULL
                   GROUP BY seed_url ORDER BY n DESC LIMIT 20""",
                (since,),
            ).fetchall()

            # Top staging repos (where users are pushing to)
            top_staging = con.execute(
                """SELECT staging_repo, COUNT(*) as n FROM plan_runs
                   WHERE ts >= datetime('now', ?) AND staging_repo IS NOT NULL
                   GROUP BY staging_repo ORDER BY n DESC LIMIT 20""",
                (since,),
            ).fetchall()

        return {
            "period_days": days,
            "tool_calls": [dict(r) for r in tool_rows],
            "clients": [dict(r) for r in dev_rows],
            "token_usage": [dict(r) for r in token_rows],
            "plan_runs": {
                "recent": [dict(r) for r in plan_rows],
                "by_outcome": [dict(r) for r in plan_stats],
                "top_seeds": [dict(r) for r in top_seeds],
                "top_staging_repos": [dict(r) for r in top_staging],
            },
        }
    except Exception:
        logger.exception("telemetry.analytics_summary failed")
        return {"error": "analytics query failed"}
