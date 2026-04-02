# Plan: CI Gate API

## Overview

Add a CI/CD integration layer so users can run myCode checks from GitHub Actions (or any CI system). Four new endpoints, a SQLite-backed API key system, and two GitHub Action YAML templates.

The CI flow reuses the existing pipeline: preflight → non-interactive intent → analyze → poll status → read report. The new endpoints wrap this into a single-submission + polling model suited for CI runners.

---

## Architecture

### Request flow

```
GitHub Action
  → POST /api/ci/check  (submit repo URL + threshold + tier)
  → internally: preflight → submit-intent (non-interactive defaults) → analyze
  → GET  /api/ci/result/{job_id}  (poll every 15s)
  → when complete: evaluate findings against threshold → pass/fail
  → optional: POST /api/ci/override/{job_id}  (flip fail → pass)
```

### What's new vs what's reused

| Component | New or Reused |
|-----------|---------------|
| Project clone, ingestion, library matching, viability | **Reused** — `handle_preflight()` |
| Non-interactive constraints from tier | **New** — maps tier 1/2/3 → analysis_depth quick/standard/deep |
| Scenario gen, execution, report gen | **Reused** — `run_analysis()` via `handle_analyze()` |
| Job store, status polling | **Reused** — existing `JobStore` + `handle_status()` |
| Threshold evaluation (pass/fail) | **New** — compares finding severities against threshold |
| API key validation | **New** — SQLite table `ci_api_keys` |
| Override mechanism | **New** — SQLite table `ci_overrides` |
| GitHub Action YAML | **New** — two template files |

---

## File Changes

### 1. `src/mycode/web/ci.py` (NEW)

All CI-specific business logic lives here. Keeps `routes.py` clean.

**Functions:**

#### `handle_ci_check(repo_url, threshold, tier, api_key) → dict`
- Validates `api_key` against SQLite key store → 403 if invalid
- Validates `threshold` ∈ {"report_only", "critical", "warning", "any"} → 400 if invalid
- Validates `tier` ∈ {1, 2, 3} → 400 if invalid
- Calls `handle_preflight(github_url=repo_url, source="ci")` — reuses existing preflight
- If preflight errors, returns error immediately
- Maps tier to analysis_depth: 1→"quick", 2→"standard", 3→"deep"
- Calls `handle_submit_intent(job_id, answers)` with non-interactive defaults (same as CLI `--non-interactive`) and the mapped depth
- Calls `handle_analyze(job_id)` to kick off background execution
- Stores threshold on the Job object (new field: `ci_threshold`)
- Updates `ci_api_keys.last_used` and increments `total_runs`
- Returns `{"job_id": "j_xxx", "status": "queued"}`

#### `handle_ci_result(job_id) → dict`
- Calls `handle_status(job_id)` to get current state
- If status is "running" / "preflight_running" / "conversing" / etc → returns `{"status": "running"}`
- If status is "failed" → returns `{"status": "error", "error": "..."}`
- If status is "completed":
  - Reads report from `handle_report(job_id)` to get finding counts
  - Counts findings by severity: critical, warning, info
  - Checks for override (query `ci_overrides` table)
  - Evaluates pass/fail based on `ci_threshold` stored on the job:
    - `report_only` → always `"pass"` (status field says `"report_only"`)
    - `critical` → fail if any critical findings
    - `warning` → fail if any critical or warning findings
    - `any` → fail if any findings at all
  - If override exists → status is `"pass"`, `"override": true`
  - Returns response dict with `status`, `summary`, `findings_count`, `threshold_applied`, `override`, `report_url`

#### `handle_ci_override(job_id, api_key, reason) → dict`
- Validates `api_key` → 403 if invalid
- Validates job exists and is completed → 400 otherwise
- Inserts row into `ci_overrides` table: job_id, api_key (hashed), reason, timestamp
- Increments `ci_api_keys.total_overrides` for this key
- Returns `{"ok": true, "job_id": job_id}`

#### `handle_ci_keys_create(admin_key) → dict`
- Validates `admin_key` against `MYCODE_ADMIN_KEY` env var → 403 if wrong
- Generates key: `mck_` + 32 hex chars (via `secrets.token_hex(16)`)
- Inserts into `ci_api_keys` table: key_hash (sha256), key_prefix (first 8 chars for display), created_at, last_used=null, total_runs=0, total_overrides=0
- Returns `{"key": "mck_xxx..."}` — full key shown only once

#### `handle_ci_keys_list(admin_key) → dict`
- Validates admin key → 403
- Returns list of all keys with: key_prefix, created_at, last_used, total_runs, total_overrides, active status
- Never returns the full key or hash

#### `handle_admin_overrides(admin_key) → dict`
- Validates admin key → 403
- Returns all override records: job_id, key_prefix, reason, timestamp

### 2. `src/mycode/web/ci_keys.py` (NEW)

SQLite operations for the CI key store. Same pattern as `analytics.py` — uses `get_db()` from analytics.

**Tables (created via migration in `get_db()`):**

```sql
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

CREATE TABLE IF NOT EXISTS ci_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL
);
```

**Functions:**
- `ensure_ci_tables(conn)` — CREATE TABLE IF NOT EXISTS for both tables. Called from `get_db()`.
- `validate_api_key(key) → bool` — hash key with sha256, query `ci_api_keys` where `key_hash=? AND active=1`
- `record_key_usage(key)` — update `last_used`, increment `total_runs`
- `create_api_key() → str` — generate `mck_` + hex, insert hash, return full key
- `list_api_keys() → list[dict]` — return all keys (prefix only, no hash)
- `record_override(job_id, key, reason)` — insert into `ci_overrides`, increment key's `total_overrides`
- `get_override(job_id) → dict | None` — check if override exists for a job
- `list_overrides() → list[dict]` — all overrides for admin view

### 3. `src/mycode/web/analytics.py` (MODIFY)

- Add call to `ensure_ci_tables(conn)` inside `get_db()` after the existing `_ensure_columns(conn)` call
- Add `"ci"` to `VALID_SOURCES` frozenset

### 4. `src/mycode/web/jobs.py` (MODIFY)

- Add `ci_threshold: str = ""` field to `Job` dataclass — stores the threshold for CI result evaluation

### 5. `src/mycode/web/app.py` (MODIFY)

Add six new endpoints, all following the existing pattern (async wrappers calling sync handlers via `asyncio.to_thread`):

```python
# ── CI Gate ──

@app.post("/api/ci/check")
async def ci_check(request: Request):
    body = await request.json()
    result = await asyncio.to_thread(
        handle_ci_check,
        body.get("repo_url", ""),
        body.get("threshold", "report_only"),
        body.get("tier", 2),
        body.get("api_key", ""),
    )
    if "error" in result:
        code = result.pop("_status", 400)
        return JSONResponse(content=result, status_code=code)
    return JSONResponse(content=result)

@app.get("/api/ci/result/{job_id}")
async def ci_result(job_id: str):
    result = await asyncio.to_thread(handle_ci_result, job_id)
    return JSONResponse(content=result)

@app.post("/api/ci/override/{job_id}")
async def ci_override(job_id: str, request: Request):
    body = await request.json()
    result = await asyncio.to_thread(
        handle_ci_override,
        job_id,
        body.get("api_key", ""),
        body.get("reason", ""),
    )
    if "error" in result:
        code = result.pop("_status", 400)
        return JSONResponse(content=result, status_code=code)
    return JSONResponse(content=result)

@app.post("/api/admin/ci-keys")
async def admin_ci_keys_create(key: str = Query(default="")):
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    result = await asyncio.to_thread(handle_ci_keys_create)
    return JSONResponse(content=result)

@app.get("/api/admin/ci-keys")
async def admin_ci_keys_list(key: str = Query(default="")):
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    result = await asyncio.to_thread(handle_ci_keys_list)
    return JSONResponse(content=result)

@app.get("/api/admin/overrides")
async def admin_overrides(key: str = Query(default="")):
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    result = await asyncio.to_thread(handle_admin_overrides)
    return JSONResponse(content=result)
```

### 6. `docs/ci/mycode-check.yml` (NEW)

GitHub Action template users copy into `.github/workflows/`:

```yaml
name: myCode Stress Test
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  mycode-check:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Run myCode stress test
        env:
          MYCODE_API_KEY: ${{ secrets.MYCODE_API_KEY }}
          MYCODE_THRESHOLD: ${{ vars.MYCODE_THRESHOLD || 'report_only' }}
          MYCODE_API_URL: ${{ vars.MYCODE_API_URL || 'https://mycode-api.up.railway.app' }}
          MYCODE_TIER: ${{ vars.MYCODE_TIER || '2' }}
        run: |
          set -euo pipefail

          # Submit check
          RESPONSE=$(curl -sf -X POST "${MYCODE_API_URL}/api/ci/check" \
            -H "Content-Type: application/json" \
            -d "{
              \"repo_url\": \"https://github.com/${{ github.repository }}\",
              \"threshold\": \"${MYCODE_THRESHOLD}\",
              \"tier\": ${MYCODE_TIER},
              \"api_key\": \"${MYCODE_API_KEY}\"
            }")

          JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
          echo "myCode job: $JOB_ID"

          if [ "$JOB_ID" = "null" ] || [ -z "$JOB_ID" ]; then
            echo "::error::Failed to submit myCode check: $RESPONSE"
            exit 1
          fi

          # Poll for result (15s intervals, 15min timeout)
          DEADLINE=$((SECONDS + 900))
          while [ $SECONDS -lt $DEADLINE ]; do
            sleep 15
            RESULT=$(curl -sf "${MYCODE_API_URL}/api/ci/result/${JOB_ID}")
            STATUS=$(echo "$RESULT" | jq -r '.status')

            if [ "$STATUS" = "running" ]; then
              echo "Still running..."
              continue
            fi

            echo "$RESULT" | jq .

            if [ "$STATUS" = "pass" ] || [ "$STATUS" = "report_only" ]; then
              echo "::notice::myCode check passed"
              exit 0
            elif [ "$STATUS" = "fail" ]; then
              SUMMARY=$(echo "$RESULT" | jq -r '.summary')
              echo "::error::myCode check failed: $SUMMARY"
              exit 1
            elif [ "$STATUS" = "error" ]; then
              echo "::warning::myCode check encountered an error"
              exit 0  # Don't block CI on myCode infrastructure errors
            fi
          done

          echo "::warning::myCode check timed out after 15 minutes"
          exit 0  # Don't block CI on timeout
```

### 7. `.github/workflows/mycode-self-check.yml` (NEW)

Dogfood workflow running myCode on its own repo for PRs:

```yaml
name: myCode Self-Check
on:
  pull_request:
    branches: [main]

jobs:
  self-check:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Run myCode on this repo
        env:
          MYCODE_API_KEY: ${{ secrets.MYCODE_API_KEY }}
          MYCODE_API_URL: ${{ vars.MYCODE_API_URL || 'https://mycode-api.up.railway.app' }}
        run: |
          set -euo pipefail

          RESPONSE=$(curl -sf -X POST "${MYCODE_API_URL}/api/ci/check" \
            -H "Content-Type: application/json" \
            -d "{
              \"repo_url\": \"https://github.com/${{ github.repository }}\",
              \"threshold\": \"report_only\",
              \"tier\": 2,
              \"api_key\": \"${MYCODE_API_KEY}\"
            }")

          JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
          echo "myCode job: $JOB_ID"

          if [ "$JOB_ID" = "null" ] || [ -z "$JOB_ID" ]; then
            echo "::error::Failed to submit: $RESPONSE"
            exit 1
          fi

          DEADLINE=$((SECONDS + 900))
          while [ $SECONDS -lt $DEADLINE ]; do
            sleep 15
            RESULT=$(curl -sf "${MYCODE_API_URL}/api/ci/result/${JOB_ID}")
            STATUS=$(echo "$RESULT" | jq -r '.status')

            if [ "$STATUS" = "running" ]; then
              echo "Still running..."
              continue
            fi

            echo "## myCode Self-Check Results" >> $GITHUB_STEP_SUMMARY
            echo '```json' >> $GITHUB_STEP_SUMMARY
            echo "$RESULT" | jq . >> $GITHUB_STEP_SUMMARY
            echo '```' >> $GITHUB_STEP_SUMMARY

            echo "$RESULT" | jq .
            exit 0  # report_only — never fails
          done

          echo "::warning::Timed out"
          exit 0
```

---

## Tests: `tests/test_ci_gate.py` (NEW)

Covers all CI gate logic without hitting the real pipeline or SQLite on disk.

### Key store tests
- `test_create_api_key` — generates `mck_`-prefixed key, 36 chars total
- `test_validate_key_valid` — created key passes validation
- `test_validate_key_invalid` — random string fails validation
- `test_key_usage_tracking` — `record_key_usage` updates last_used and total_runs
- `test_list_keys_no_hash_exposed` — list endpoint never returns key_hash

### CI check endpoint tests (mock preflight + analyze)
- `test_ci_check_missing_api_key` — 403
- `test_ci_check_invalid_api_key` — 403
- `test_ci_check_invalid_threshold` — 400
- `test_ci_check_invalid_tier` — 400
- `test_ci_check_success` — returns job_id + status "queued", source="ci" on job
- `test_ci_check_preflight_failure` — returns error from preflight

### CI result endpoint tests
- `test_ci_result_running` — returns `{"status": "running"}`
- `test_ci_result_completed_report_only` — always status "pass" regardless of findings
- `test_ci_result_completed_critical_pass` — no critical findings → pass
- `test_ci_result_completed_critical_fail` — has critical finding → fail
- `test_ci_result_completed_warning_fail` — has warning finding → fail with threshold "warning"
- `test_ci_result_completed_any_fail` — has info finding → fail with threshold "any"
- `test_ci_result_with_override` — overridden job returns status "pass", override=true
- `test_ci_result_not_found` — 404
- `test_ci_result_job_failed` — returns status "error"

### CI override endpoint tests
- `test_ci_override_success` — flips status, records audit trail
- `test_ci_override_invalid_key` — 403
- `test_ci_override_job_not_completed` — 400
- `test_ci_override_audit_trail` — admin endpoint shows the override

### Admin endpoint tests
- `test_admin_ci_keys_create` — requires admin key, returns mck_ key
- `test_admin_ci_keys_list` — requires admin key, returns key list
- `test_admin_overrides_list` — requires admin key, returns override list

### Source tagging test
- `test_ci_jobs_tagged_source_ci` — job_log row has source="ci"

---

## What Does NOT Change

- Existing endpoints (`/api/preflight`, `/api/submit-intent`, `/api/analyze`, `/api/status`, `/api/report`) — untouched
- Web frontend — no changes
- Pipeline internals (ingester, scenario generator, engine, report generator) — untouched
- Existing analytics schema — only additive (new tables)
- Existing admin endpoints — untouched
- CLI interface — untouched

## Dependencies

No new dependencies. Uses stdlib `secrets`, `hashlib` for key generation/hashing. SQLite tables added to the existing analytics database.

## Deployment Notes

- Railway persistent volume at `/data/` must already be attached (same as analytics requirement)
- `MYCODE_ADMIN_KEY` env var must be set (already required for existing admin endpoints)
- After deploy: generate first CI key via `POST /api/admin/ci-keys?key=<MYCODE_ADMIN_KEY>`
- Add `MYCODE_API_KEY` as a GitHub secret on any repo using the action

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/mycode/web/ci.py` | NEW | CI endpoint handlers |
| `src/mycode/web/ci_keys.py` | NEW | SQLite key store + override table |
| `src/mycode/web/analytics.py` | MODIFY | Add "ci" to VALID_SOURCES, wire CI tables into get_db() |
| `src/mycode/web/jobs.py` | MODIFY | Add `ci_threshold` field to Job |
| `src/mycode/web/app.py` | MODIFY | Register 6 new endpoints |
| `docs/ci/mycode-check.yml` | NEW | GitHub Action template for users |
| `.github/workflows/mycode-self-check.yml` | NEW | Dogfood workflow |
| `tests/test_ci_gate.py` | NEW | ~30 tests covering all CI gate logic |
