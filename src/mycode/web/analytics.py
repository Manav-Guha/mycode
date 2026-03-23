"""SQLite-backed job analytics for the myCode web backend.

Logs every web job to a local SQLite database for usage tracking.
The database is created on first access — no migration framework needed.

Configuration:
    MYCODE_DB_PATH: Path to SQLite database file.
        Default: /data/mycode_analytics.db
        NOTE: On Railway, this requires a persistent volume mounted at /data/.
        Without a volume, the database resets on every deploy. Attach a
        Railway volume at /data/ before deploying, or set this env var to
        a path on an attached volume.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

VALID_SOURCES = frozenset({"internal", "hn", "public", "cli", "test_group"})
_INTERNAL_SOURCES = frozenset({"internal", "test_group"})

_DB_PATH = os.environ.get("MYCODE_DB_PATH", "/data/mycode_analytics.db")

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS job_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'public',
    repo_url TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'started',
    findings_critical INTEGER,
    findings_warning INTEGER,
    findings_info INTEGER,
    scenarios_run INTEGER,
    pdf_downloaded BOOLEAN NOT NULL DEFAULT 0,
    json_downloaded BOOLEAN NOT NULL DEFAULT 0
);
"""

# Per-thread connections — SQLite connections are not thread-safe.
_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating the DB if needed."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    db_path = os.environ.get("MYCODE_DB_PATH", _DB_PATH)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    conn.commit()
    _local.conn = conn
    return conn


def validate_source(source: str) -> str:
    """Return source if valid, otherwise 'public'."""
    return source if source in VALID_SOURCES else "public"


def log_job_started(job_id: str, source: str, repo_url: str) -> None:
    """Insert a new job_log row with status='started'."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO job_log (job_id, source, repo_url, started_at, status) "
            "VALUES (?, ?, ?, ?, 'started')",
            (job_id, validate_source(source), repo_url, _now_iso()),
        )
        db.commit()
    except Exception as exc:
        logger.warning("Analytics: failed to log job start for %s: %s", job_id, exc)


def log_job_completed(
    job_id: str,
    status: str,
    findings_critical: int = 0,
    findings_warning: int = 0,
    findings_info: int = 0,
    scenarios_run: int = 0,
) -> None:
    """Update job_log row with completion data."""
    try:
        db = get_db()
        db.execute(
            "UPDATE job_log SET completed_at=?, status=?, "
            "findings_critical=?, findings_warning=?, findings_info=?, "
            "scenarios_run=? WHERE job_id=?",
            (
                _now_iso(), status,
                findings_critical, findings_warning, findings_info,
                scenarios_run, job_id,
            ),
        )
        db.commit()
    except Exception as exc:
        logger.warning("Analytics: failed to log job completion for %s: %s", job_id, exc)


def log_download(job_id: str, download_type: str) -> None:
    """Mark pdf_downloaded or json_downloaded as true."""
    col = {"pdf": "pdf_downloaded", "json": "json_downloaded"}.get(download_type)
    if col is None:
        return
    try:
        db = get_db()
        db.execute(f"UPDATE job_log SET {col}=1 WHERE job_id=?", (job_id,))
        db.commit()
    except Exception as exc:
        logger.warning("Analytics: failed to log download for %s: %s", job_id, exc)


def get_admin_stats() -> dict:
    """Compute all admin stats from the job_log table."""
    db = get_db()

    def scalar(sql: str, params: tuple = ()) -> int | float | None:
        row = db.execute(sql, params).fetchone()
        return row[0] if row else 0

    total_jobs = scalar("SELECT COUNT(*) FROM job_log")

    external_jobs = scalar(
        "SELECT COUNT(*) FROM job_log WHERE source NOT IN ('internal', 'test_group')"
    )

    # by_source
    by_source: dict[str, int] = {}
    for row in db.execute("SELECT source, COUNT(*) FROM job_log GROUP BY source"):
        by_source[row[0]] = row[1]

    # by_status (exclude 'started' — those are still in-flight)
    by_status: dict[str, int] = {}
    for row in db.execute(
        "SELECT status, COUNT(*) FROM job_log WHERE status != 'started' GROUP BY status"
    ):
        by_status[row[0]] = row[1]

    # last_24h
    last_24h_total = scalar(
        "SELECT COUNT(*) FROM job_log WHERE started_at > datetime('now', '-1 day')"
    )
    last_24h_external = scalar(
        "SELECT COUNT(*) FROM job_log "
        "WHERE started_at > datetime('now', '-1 day') "
        "AND source NOT IN ('internal', 'test_group')"
    )

    # last_7d
    last_7d_total = scalar(
        "SELECT COUNT(*) FROM job_log WHERE started_at > datetime('now', '-7 days')"
    )
    last_7d_external = scalar(
        "SELECT COUNT(*) FROM job_log "
        "WHERE started_at > datetime('now', '-7 days') "
        "AND source NOT IN ('internal', 'test_group')"
    )

    # Download rates (among completed jobs only)
    completed = scalar("SELECT COUNT(*) FROM job_log WHERE status='completed'") or 0
    if completed > 0:
        pdf_rate = round(
            scalar("SELECT SUM(pdf_downloaded) FROM job_log WHERE status='completed'")
            / completed, 2
        )
        json_rate = round(
            scalar("SELECT SUM(json_downloaded) FROM job_log WHERE status='completed'")
            / completed, 2
        )
    else:
        pdf_rate = 0.0
        json_rate = 0.0

    # Average findings (completed jobs)
    avg_critical = round(
        scalar("SELECT AVG(findings_critical) FROM job_log WHERE status='completed'") or 0, 1
    )
    avg_warning = round(
        scalar("SELECT AVG(findings_warning) FROM job_log WHERE status='completed'") or 0, 1
    )

    # Unique repos (excluding zip uploads)
    unique_repos = scalar(
        "SELECT COUNT(DISTINCT repo_url) FROM job_log "
        "WHERE repo_url != 'zip_upload' AND repo_url != ''"
    )

    # Return repos (distinct URLs appearing more than once)
    return_repos = scalar(
        "SELECT COUNT(*) FROM ("
        "  SELECT repo_url FROM job_log "
        "  WHERE repo_url != 'zip_upload' AND repo_url != '' "
        "  GROUP BY repo_url HAVING COUNT(*) > 1"
        ")"
    )

    return {
        "total_jobs": total_jobs,
        "external_jobs": external_jobs,
        "by_source": by_source,
        "by_status": by_status,
        "last_24h": {"total": last_24h_total, "external": last_24h_external},
        "last_7d": {"total": last_7d_total, "external": last_7d_external},
        "pdf_download_rate": pdf_rate,
        "json_download_rate": json_rate,
        "avg_findings_critical": avg_critical,
        "avg_findings_warning": avg_warning,
        "unique_repos": unique_repos,
        "return_repos": return_repos,
    }


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def close_connection() -> None:
    """Close the per-thread connection (for testing cleanup)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
