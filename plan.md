# Plan: Analytics instrumentation — server-side job logging with source tagging

## Overview

Add SQLite-backed job logging to track every web job, with source tagging from URL query params and an admin stats endpoint. No UI, no accounts, no migration framework.

---

## New file: `src/mycode/web/analytics.py`

Single module for all analytics logic: DB init, log writes, stats queries.

### Database

- SQLite at `os.environ.get("MYCODE_DB_PATH", "/data/mycode_analytics.db")`
- Created on first access if missing (including parent dirs)
- One table `job_log` with columns per spec

### Functions

```python
def get_db() -> sqlite3.Connection
    # Returns connection. Creates DB + table on first call.
    # Uses threading.local() for per-thread connections (SQLite is not thread-safe).

def log_job_started(job_id: str, source: str, repo_url: str) -> None
    # INSERT row with status="started", started_at=now

def log_job_completed(job_id: str, status: str, findings_critical: int,
                      findings_warning: int, findings_info: int,
                      scenarios_run: int) -> None
    # UPDATE row: completed_at=now, status, finding counts, scenarios_run

def log_download(job_id: str, download_type: str) -> None
    # UPDATE pdf_downloaded=true or json_downloaded=true

def get_admin_stats() -> dict
    # Single function that runs all stat queries and returns the JSON shape from the spec
```

### Schema

```sql
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
```

---

## Changes to existing files

### `src/mycode/web/jobs.py`

- Add `source: str = "public"` field to the `Job` dataclass

### `src/mycode/web/app.py`

- Add `source: str = Query(default="public")` parameter to the `preflight` endpoint
- Pass `source` through to `handle_preflight`
- Add new endpoint: `GET /api/admin/stats` with `key` query param
- Add new endpoint: `GET /api/report/{job_id}/report.json` for JSON download tracking (or track via existing report endpoint — see design note below)

**Design note on JSON download tracking:** The current JSON download is client-side only (`downloadJSON()` in app.js creates a blob from already-fetched report data). Two options:

1. Add a lightweight `POST /api/report/{job_id}/download-log` endpoint that the frontend calls when downloading JSON
2. Track it on the existing `GET /api/report/{job_id}` call (but this fires on report view, not download)

**Decision: Option 1** — small dedicated endpoint. The frontend calls it fire-and-forget when JSON is downloaded. Same endpoint used for PDF download tracking (called from the PDF download handler).

### `src/mycode/web/routes.py`

- `handle_preflight()`: accept `source` parameter, set `job.source = source`, call `log_job_started(job.id, source, repo_url)`
- `handle_download_pdf()`: call `log_download(job_id, "pdf")` on successful download
- New `handle_admin_stats()`: validate admin key, call `get_admin_stats()`, return result
- New `handle_log_download()`: accept job_id + download_type, call `log_download()`

### `src/mycode/web/worker.py`

- At end of `run_analysis()`: on success, call `log_job_completed(job.id, "completed", critical, warning, info, scenarios_run)`. On failure, call `log_job_completed(job.id, "failed", ...)`. Extract finding counts from `job.result.report`.
- Detect timeout: if `job.error` contains timeout indicators or job exceeded budget, use status="timeout" instead of "failed"

### `web/app.js`

- Read `?source=` from `window.location.search` on page load
- Append `source` to the FormData in `submitUrl()` and `submitFile()`
- In `downloadJSON()`: fire `fetch("/api/report/{jobId}/download-log", {method: "POST", body: ...})` (fire-and-forget, no await needed)
- In `downloadUnderstanding()`: same fire-and-forget call after successful PDF download

---

## New endpoint: `GET /api/admin/stats`

```python
@app.get("/api/admin/stats")
async def admin_stats(key: str = Query(default="")):
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    stats = get_admin_stats()
    return JSONResponse(content=stats)
```

### Stats queries (all in `get_admin_stats()`)

| Stat | Query |
|------|-------|
| `total_jobs` | `SELECT COUNT(*) FROM job_log` |
| `external_jobs` | `SELECT COUNT(*) FROM job_log WHERE source != 'internal' AND source != 'test_group'` |
| `by_source` | `SELECT source, COUNT(*) FROM job_log GROUP BY source` |
| `by_status` | `SELECT status, COUNT(*) FROM job_log WHERE status != 'started' GROUP BY status` |
| `last_24h.total` | `WHERE started_at > datetime('now', '-1 day')` |
| `last_24h.external` | Same + `AND source NOT IN ('internal', 'test_group')` |
| `last_7d` | Same pattern with `-7 days` |
| `pdf_download_rate` | `SUM(pdf_downloaded) / COUNT(*)` where status='completed' |
| `json_download_rate` | `SUM(json_downloaded) / COUNT(*)` where status='completed' |
| `avg_findings_critical` | `AVG(findings_critical)` where status='completed' |
| `avg_findings_warning` | `AVG(findings_warning)` where status='completed' |
| `unique_repos` | `SELECT COUNT(DISTINCT repo_url) FROM job_log WHERE repo_url != 'zip_upload'` |
| `return_repos` | `SELECT COUNT(*) FROM (SELECT repo_url FROM job_log WHERE repo_url != 'zip_upload' GROUP BY repo_url HAVING COUNT(*) > 1)` |

---

## New endpoint: `POST /api/report/{job_id}/download-log`

```python
@app.post("/api/report/{job_id}/download-log")
async def log_download_endpoint(job_id: str, type: str = Form(default="")):
    if type in ("pdf", "json"):
        log_download(job_id, type)
    return JSONResponse(content={"ok": True})
```

---

## Source validation

Valid sources: `"internal"`, `"hn"`, `"public"`, `"cli"`, `"test_group"`. If an unrecognised source is passed, default to `"public"`. Validation in `handle_preflight()`.

---

## Tests: `tests/test_analytics.py`

All tests use a temp SQLite DB (tmp file, not `/data/`).

| Test | What it verifies |
|------|------------------|
| `test_log_job_started` | Row inserted with correct job_id, source, repo_url, status="started", started_at set, completed_at null |
| `test_log_job_completed` | Row updated with status, finding counts, completed_at set |
| `test_log_job_completed_timeout` | status="timeout" stored correctly |
| `test_log_download_pdf` | pdf_downloaded flipped to true |
| `test_log_download_json` | json_downloaded flipped to true |
| `test_source_validation` | Invalid source defaults to "public" |
| `test_admin_stats_no_key` | Returns 403 when MYCODE_ADMIN_KEY not set |
| `test_admin_stats_wrong_key` | Returns 403 when key doesn't match |
| `test_admin_stats_valid` | Returns correct JSON shape with accurate counts |
| `test_admin_stats_external_excludes_internal` | internal/test_group jobs excluded from external_jobs count |
| `test_admin_stats_return_repos` | Repos appearing 2+ times counted correctly |
| `test_admin_stats_download_rates` | pdf/json download rates calculated correctly |
| `test_db_created_on_first_access` | DB file + table created if path doesn't exist |

Tests for source propagation through the API (using FastAPI TestClient):

| Test | What it verifies |
|------|------------------|
| `test_preflight_source_param` | source query param flows to job_log |
| `test_preflight_default_source` | Missing source defaults to "public" |

---

## Files modified (summary)

| File | Change |
|------|--------|
| `src/mycode/web/analytics.py` | **NEW** — DB init, log writes, stats queries |
| `src/mycode/web/jobs.py` | Add `source` field to Job |
| `src/mycode/web/app.py` | Add source param to preflight, add admin stats + download-log endpoints |
| `src/mycode/web/routes.py` | Wire analytics calls into preflight, PDF download, add admin stats handler |
| `src/mycode/web/worker.py` | Log completion/failure at end of run_analysis |
| `web/app.js` | Read source from URL, pass to preflight, fire download-log on PDF/JSON download |
| `tests/test_analytics.py` | **NEW** — all tests listed above |

## What is NOT changed

- No admin dashboard UI
- No user accounts or auth beyond admin key
- No data retention / cleanup policies
- No migration framework
- Existing tests untouched (no breaking changes)
