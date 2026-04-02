"""SQLite-backed API key store and override audit trail for CI gate.

Uses the same database as analytics.py (MYCODE_DB_PATH).  Tables are
created on first access via ``ensure_ci_tables()``.

Keys are ``mck_``-prefixed, stored as SHA-256 hashes.  The full key is
returned only once at creation time.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone

from mycode.web.analytics import get_db

logger = logging.getLogger(__name__)

# ── Table DDL ──

_CREATE_CI_KEYS_TABLE = """\
CREATE TABLE IF NOT EXISTS ci_api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_used TIMESTAMP,
    total_runs INTEGER NOT NULL DEFAULT 0,
    total_overrides INTEGER NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT 1
);
"""

_CREATE_CI_OVERRIDES_TABLE = """\
CREATE TABLE IF NOT EXISTS ci_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL
);
"""


def ensure_ci_tables(conn) -> None:
    """Create CI tables if they don't exist (idempotent)."""
    conn.execute(_CREATE_CI_KEYS_TABLE)
    conn.execute(_CREATE_CI_OVERRIDES_TABLE)


# ── Helpers ──


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Key CRUD ──


def create_api_key() -> str:
    """Generate a new ``mck_``-prefixed API key and store its hash.

    Returns the full key (shown only once).
    """
    raw = secrets.token_hex(16)
    key = f"mck_{raw}"
    key_hash = _hash_key(key)
    prefix = key[:8]

    db = get_db()
    db.execute(
        "INSERT INTO ci_api_keys (key_hash, key_prefix, created_at) VALUES (?, ?, ?)",
        (key_hash, prefix, _now_iso()),
    )
    db.commit()
    return key


def validate_api_key(key: str) -> bool:
    """Return True if *key* matches an active row in ci_api_keys."""
    if not key or not key.startswith("mck_"):
        return False
    key_hash = _hash_key(key)
    db = get_db()
    row = db.execute(
        "SELECT 1 FROM ci_api_keys WHERE key_hash = ? AND active = 1",
        (key_hash,),
    ).fetchone()
    return row is not None


def record_key_usage(key: str) -> None:
    """Update last_used and increment total_runs for *key*."""
    key_hash = _hash_key(key)
    try:
        db = get_db()
        db.execute(
            "UPDATE ci_api_keys SET last_used = ?, total_runs = total_runs + 1 "
            "WHERE key_hash = ?",
            (_now_iso(), key_hash),
        )
        db.commit()
    except Exception as exc:
        logger.warning("Failed to record key usage: %s", exc)


def list_api_keys() -> list[dict]:
    """Return all keys (prefix only — never expose hash)."""
    db = get_db()
    rows = db.execute(
        "SELECT key_prefix, created_at, last_used, total_runs, "
        "total_overrides, active FROM ci_api_keys ORDER BY created_at DESC"
    ).fetchall()
    return [
        {
            "key_prefix": r[0],
            "created_at": r[1],
            "last_used": r[2],
            "total_runs": r[3],
            "total_overrides": r[4],
            "active": bool(r[5]),
        }
        for r in rows
    ]


# ── Overrides ──


def record_override(job_id: str, key: str, reason: str) -> None:
    """Insert an override record and increment the key's total_overrides."""
    key_hash = _hash_key(key)
    db = get_db()
    db.execute(
        "INSERT INTO ci_overrides (job_id, key_hash, reason, created_at) "
        "VALUES (?, ?, ?, ?)",
        (job_id, key_hash, reason, _now_iso()),
    )
    db.execute(
        "UPDATE ci_api_keys SET total_overrides = total_overrides + 1 "
        "WHERE key_hash = ?",
        (key_hash,),
    )
    db.commit()


def get_override(job_id: str) -> dict | None:
    """Return the override record for *job_id*, or None."""
    db = get_db()
    row = db.execute(
        "SELECT o.job_id, k.key_prefix, o.reason, o.created_at "
        "FROM ci_overrides o "
        "JOIN ci_api_keys k ON o.key_hash = k.key_hash "
        "WHERE o.job_id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "job_id": row[0],
        "key_prefix": row[1],
        "reason": row[2],
        "created_at": row[3],
    }


def list_overrides() -> list[dict]:
    """Return all override records (for admin view)."""
    db = get_db()
    rows = db.execute(
        "SELECT o.job_id, k.key_prefix, o.reason, o.created_at "
        "FROM ci_overrides o "
        "JOIN ci_api_keys k ON o.key_hash = k.key_hash "
        "ORDER BY o.created_at DESC"
    ).fetchall()
    return [
        {
            "job_id": r[0],
            "key_prefix": r[1],
            "reason": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]
