# Admin Jobs Endpoint — Implementation Plan

## Goal
New `GET /api/admin/jobs` endpoint returning per-job detail records from the analytics SQLite DB, filtered by source/status. Enables tracking individual test group users in India.

---

## Current State

- **Analytics DB** (`analytics.py`): `job_log` table has columns: `id`, `job_id`, `source`, `repo_url`, `started_at`, `completed_at`, `status`, `findings_critical`, `findings_warning`, `findings_info`, `scenarios_run`, `pdf_downloaded`, `json_downloaded`.
- **Missing from DB**: languages detected, deps found count, individual finding titles, predictions count, repo identifier (partially — `repo_url` stores the GitHub URL or `"zip_upload"`, but not the uploaded zip filename).
- **Admin auth pattern** (`app.py:277-284`): `MYCODE_ADMIN_KEY` env var, passed as `?key=` query param, returns 403 if missing or wrong.

---

## What's Already Stored vs What's Needed

| Field needed | Already in `job_log`? | Source |
|---|---|---|
| job_id | Yes | `job_id` column |
| source | Yes | `source` column |
| status | Yes | `status` column |
| created_at | Yes | `started_at` column |
| completed_at | Yes | `completed_at` column |
| duration_seconds | Derivable | `completed_at - started_at` |
| repo_identifier | **Partial** — GitHub URL stored, zip filename not | `repo_url` column |
| languages_detected | **No** | Not stored |
| deps_found | **No** | Not stored |
| findings counts by severity | Yes | `findings_critical`, `findings_warning`, `findings_info` |
| finding titles list | **No** | Not stored |
| predictions_count | **No** | Not stored |
| pdf_downloaded | Yes | `pdf_downloaded` column |
| json_downloaded | Yes | `json_downloaded` column |

---

## Implementation Steps

### Step 1: Schema migration — add missing columns to `job_log`

Add 4 new columns to the CREATE TABLE and an `ALTER TABLE` migration for existing DBs:

```sql
languages_detected TEXT DEFAULT ''       -- comma-separated, e.g. "python,javascript"
deps_found INTEGER DEFAULT 0            -- count of non-dev dependencies
finding_titles TEXT DEFAULT ''           -- JSON array of finding title strings
predictions_count INTEGER DEFAULT 0     -- number of predictions returned
```

**File:** `analytics.py`

- Update `_CREATE_TABLE` to include the new columns
- Add `_ensure_columns()` function called from `get_db()` that runs `ALTER TABLE ADD COLUMN` for each missing column (wrapped in try/except for "duplicate column" — idempotent migration)
- This is the standard pattern for SQLite schema evolution without a migration framework

### Step 2: Capture `repo_identifier` for zip uploads

**File:** `routes.py` — `handle_preflight()`

Currently line 104: `repo_url = "zip_upload"` — change to `repo_url = f"zip:{filename}"` where `filename` is the uploaded file's original name. Falls back to `"zip_upload"` if filename is empty.

This means `repo_url` will now contain either:
- A GitHub URL (already works)
- `"zip:myproject.zip"` for uploads (new — captures filename)
- `"zip_upload"` for historical records (no migration needed — returned as-is)

### Step 3: Capture new fields at job completion

**File:** `analytics.py`

Update `log_job_completed()` signature to accept optional new fields:

```python
def log_job_completed(
    job_id: str,
    status: str,
    findings_critical: int = 0,
    findings_warning: int = 0,
    findings_info: int = 0,
    scenarios_run: int = 0,
    *,
    languages_detected: str = "",
    deps_found: int = 0,
    finding_titles: str = "",       # JSON array string
    predictions_count: int = 0,
) -> None:
```

Update the SQL UPDATE to set the new columns.

**File:** `worker.py` — `_log_completion()`

Extract additional data from `job` and pass to `log_job_completed()`:
- `languages_detected`: `",".join(sorted(job.detected_languages))`
- `deps_found`: `len([d for d in job.ingestion.dependencies if not d.is_dev])` if ingestion exists
- `finding_titles`: `json.dumps([f.title for f in report.findings])` if report exists
- `predictions_count`: count from job result predictions if available

### Step 4: New query function `get_admin_jobs()`

**File:** `analytics.py`

```python
def get_admin_jobs(
    source: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
```

- Query `job_log` with optional WHERE clauses for source and status
- ORDER BY `started_at DESC`
- LIMIT/OFFSET with cap: `limit = min(limit, 200)`
- For each row, compute `duration_seconds` as `(completed_at - started_at)` in seconds (None if not completed)
- Parse `finding_titles` from JSON string back to list
- Map `repo_url` → `repo_identifier` in the output (rename for API clarity)
- Return list of dicts matching the spec

### Step 5: Wire up endpoint in `app.py`

**File:** `app.py`

```python
@app.get("/api/admin/jobs")
async def admin_jobs(
    key: str = Query(default=""),
    source: str = Query(default=""),
    status: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Return individual job records. Requires MYCODE_ADMIN_KEY."""
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    jobs = await asyncio.to_thread(
        get_admin_jobs,
        source=source or None,
        status=status or None,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(content={"jobs": jobs, "count": len(jobs)})
```

### Step 6: Tests

**File:** `tests/test_admin_jobs.py` (new)

1. **`test_admin_jobs_auth_required`** — no key → 403
2. **`test_admin_jobs_wrong_key`** — wrong key → 403
3. **`test_admin_jobs_empty_db`** — valid key, no jobs → empty list
4. **`test_admin_jobs_returns_records`** — insert 3 job_log rows, verify all returned with correct fields
5. **`test_admin_jobs_filter_by_source`** — insert jobs with different sources, filter by `source=test_group`, verify only matching returned
6. **`test_admin_jobs_filter_by_status`** — filter by `status=completed`
7. **`test_admin_jobs_pagination`** — insert 5 jobs, `limit=2&offset=0` → 2 records, `offset=2` → next 2
8. **`test_admin_jobs_limit_cap`** — request `limit=500`, verify capped to 200
9. **`test_admin_jobs_newest_first`** — verify ordering by started_at DESC
10. **`test_admin_jobs_duration_computed`** — job with both started_at and completed_at, verify duration_seconds is correct
11. **`test_admin_jobs_null_fields_for_incomplete`** — started job with no completion data, verify nulls for completed_at, duration, findings
12. **`test_admin_jobs_finding_titles_parsed`** — verify finding_titles comes back as list, not JSON string
13. **`test_zip_filename_captured`** — verify zip upload stores `"zip:filename.zip"` not just `"zip_upload"`
14. **`test_log_completion_new_fields`** — verify languages_detected, deps_found, finding_titles, predictions_count are stored and retrievable

---

## Response Shape

```json
{
  "jobs": [
    {
      "job_id": "j_abc123def456",
      "source": "test_group",
      "status": "completed",
      "created_at": "2026-03-30T08:15:00+00:00",
      "completed_at": "2026-03-30T08:18:42+00:00",
      "duration_seconds": 222,
      "repo_identifier": "https://github.com/user/repo",
      "languages_detected": "python,javascript",
      "deps_found": 12,
      "findings_critical": 1,
      "findings_warning": 3,
      "findings_info": 2,
      "finding_titles": ["Memory accumulation under sustained load", "Response time degradation at 500 concurrent users", ...],
      "predictions_count": 5,
      "pdf_downloaded": true,
      "json_downloaded": false
    }
  ],
  "count": 1
}
```

For historical jobs where new columns don't exist: `languages_detected` → `""`, `deps_found` → `0`, `finding_titles` → `[]`, `predictions_count` → `0`. `repo_identifier` returns whatever `repo_url` contains (GitHub URL, `"zip_upload"`, or `"zip:filename.zip"`).

---

## Files Modified

| File | Change |
|---|---|
| `src/mycode/web/analytics.py` | 4 new columns + migration, `log_job_completed()` expanded, new `get_admin_jobs()` |
| `src/mycode/web/worker.py` | `_log_completion()` extracts and passes new fields |
| `src/mycode/web/routes.py` | Zip upload captures filename in `repo_url` |
| `src/mycode/web/app.py` | New `/api/admin/jobs` endpoint |
| `tests/test_admin_jobs.py` | **New file** — 14 tests |

## Files NOT Modified

| File | Reason |
|---|---|
| `src/mycode/web/jobs.py` | In-memory Job dataclass unchanged — all new data flows through analytics DB |
| `src/mycode/report.py` | Data already exists on report objects |
| `src/mycode/prediction.py` | Read-only — no changes needed |
| `web/app.js` | No frontend for admin endpoint |
