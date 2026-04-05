# Plan: Two Bug Fixes

**STATUS: AWAITING REVIEW**

---

## Bug 1: Fix Prompt Too Generic for Slow Endpoint Findings

### Root Cause

When an HTTP endpoint errors at low concurrency (e.g., "Get Slow-Report (http)" at load_level=1), the finding gets category `http_load_testing` and affected_dependencies `["http"]`. The prompt generation flow is:

1. `generate_finding_prompt()` (`documents.py:1003-1035`) calls `_build_fix(f)` (line 1031)
2. `_build_fix()` (`documents.py:1366-1375`) calls `_match_remediation(f)` → no pattern matches because:
   - `_pat_external_timeout` (`documents.py:1278-1289`) requires `"skipped" in title` — this finding's title is `"Get Slow-Report (http)"`, not a skipped finding
   - `_pat_flask_concurrency` requires framework="flask" — but `_detect_framework(["http"])` returns `""` since "http" is not a framework name
   - No other pattern matches `http_load_testing` findings with non-framework deps
3. Falls back to `_FIX_OBJECTIVES["http_load_testing"]` (`documents.py:1063-1066`): `"handle concurrent HTTP load without crashes, error spikes, or excessive response time."`

This fallback is useless to a coding agent. The endpoint errored at concurrency=1, which means it's not a concurrency problem at all — it's a blocking call that times out even on a single request.

### Evidence

- `http_load_driver.py:90`: `_SLOW_BASELINE_MS = 10_000` — endpoints >10s at concurrency=1 are either skipped (info) or error (critical)
- `report.py:3944-3948`: `http_get_slow-report` → title `"Get Slow-Report (http)"`, deps `["http"]`
- `documents.py:1063-1066`: Fallback produces the generic text
- The test report JSON confirms: `"Get Slow-Report (http)"` with `load_level: 1`, `severity: "critical"`, `prompt: "Fix: handle concurrent HTTP load..."`

### Proposed Fix

Add a new remediation pattern that matches HTTP endpoint findings that failed at **low concurrency** (load_level ≤ 5). At this level, the problem is a blocking call, not a concurrency issue.

**File:** `src/mycode/documents.py`
**Insert:** Before `_pat_external_timeout` (line 1277), so it matches first for low-concurrency errors while the existing skipped pattern still handles the info-severity case.

```python
@_register_pattern
def _pat_http_endpoint_blocking(f, framework, fields):
    """Match HTTP endpoint findings that failed at very low concurrency."""
    if f.category != "http_load_testing":
        return None
    if f._load_level is None or f._load_level > 5:
        return None
    if "could not start" in f.title.lower():
        return None  # handled by _pat_startup_failure
    endpoint = fields.get("endpoint", "this endpoint")
    return (
        f"Your {endpoint} took too long to respond even at {f._load_level} "
        f"concurrent connection(s). This usually means the route handler "
        f"contains a blocking call — a synchronous operation that holds the "
        f"thread until it completes.",
        "Check the route handler for: (1) time.sleep() calls — remove or "
        "replace with async alternatives, (2) synchronous database queries "
        "without timeouts — add connection timeouts and consider async DB "
        "drivers, (3) external API calls without timeouts — add "
        "requests.get(url, timeout=5) or equivalent. The blocking call "
        "must be removed or made non-blocking for the endpoint to respond "
        "under load.",
    )
```

Also update the `_FIX_OBJECTIVES["http_load_testing"]` fallback to be more specific for the cases that reach it (higher concurrency failures without a framework match):

```python
"http_load_testing": (
    "handle concurrent HTTP load without crashes or excessive response "
    "time. Check for blocking I/O in route handlers, add connection "
    "timeouts to external calls, and consider running with multiple "
    "workers (e.g. gunicorn -w 4)."
),
```

### Acceptance Criteria

1. An endpoint that errors at load_level=1 gets: "Check the route handler for: time.sleep() calls..."
2. An endpoint that errors at load_level=50 still gets framework-specific advice (Flask/FastAPI patterns) or the updated generic fallback
3. The skipped "Endpoint X skipped — slow response" info finding still gets `_pat_external_timeout` advice (unchanged, since it's info severity and won't generate a prompt anyway)
4. Existing tests pass

---

## Bug 2: Edition Counter Not Incrementing for Zip Re-uploads

### Root Cause

**File:** `src/mycode/web/routes.py`, line 701:
```python
edition = get_next_edition(github_url=job.github_url or None)
```

For zip uploads, `job.github_url` is never set (line 100-105 shows zip path skips the `if github_url:` branch). So `get_next_edition()` receives `github_url=None, project_path=None` and hits the early return at `documents.py:584-585`:
```python
else:
    return 1
```

Every zip upload always gets Edition 1.

### Why the current design doesn't work for zips

The edition system has two identification mechanisms:
1. **CLI:** Hashes the resolved local `project_path` — works because the user re-runs from the same directory
2. **Web/GitHub:** Hashes the normalized GitHub URL — works because the URL is a stable identifier

Zip uploads have neither: the project is extracted to a temp directory (different path each time) and there's no URL. The system has no stable identifier to match re-uploads against.

### Proposed Fix

Use `job.project_name` as the stable identifier for zip uploads. When two zip uploads have the same project name (from `_infer_project_name()` reading `package.json`/`pyproject.toml`/dirname), they're treated as the same project.

**File:** `src/mycode/web/routes.py`, line 701

**Change:**
```python
# Before:
edition = get_next_edition(github_url=job.github_url or None)

# After:
if job.github_url:
    edition = get_next_edition(github_url=job.github_url)
else:
    # Zip uploads: use project_name as stable identifier
    edition = get_next_edition(
        project_path=Path(job.project_name or "untitled"),
    )
```

**Why `project_path=Path(job.project_name)`:** `get_next_edition` hashes `str(project_path.resolve())` for the CLI case. Passing a `Path` constructed from the project name produces a hash of the name string. This is a slight abuse of the parameter semantics, but it avoids changing the `get_next_edition` API.

**Cleaner alternative:** Add a `project_name` parameter to `get_next_edition`:

**File:** `src/mycode/documents.py`, lines 569-585

```python
def get_next_edition(project_path: Optional[Path] = None,
                     github_url: Optional[str] = None,
                     project_name: Optional[str] = None) -> int:
    if github_url:
        key_input = _normalize_github_url(github_url)
    elif project_path:
        key_input = str(project_path.resolve())
    elif project_name:
        key_input = f"name:{project_name.strip().lower()}"
    else:
        return 1
```

Then in `routes.py:701`:
```python
edition = get_next_edition(
    github_url=job.github_url or None,
    project_name=job.project_name if not job.github_url else None,
)
```

**I recommend the cleaner alternative.** It makes the intent explicit and avoids conflating a project name with a filesystem path.

### Edge Cases

- **Same project name, different projects:** Two unrelated zip uploads named "My App" would share an edition counter. This is acceptable — the name is chosen by the user (or inferred from package.json), and name collisions at the per-user level are unlikely. The edition counter is informational, not a security boundary.
- **Different zip filenames, same project:** A project uploaded as `dashboard-v1.zip` then `dashboard-v2.zip` would match by the *project name inside the zip* (from package.json/pyproject.toml), not by the zip filename. This is correct behavior.
- **No project name at all:** Falls through to `return 1` — same as today. Unnamed zips don't get edition tracking.

### Acceptance Criteria

1. First zip upload of "Test Dashboard" → Edition 1
2. Second zip upload of "Test Dashboard" → Edition 2
3. First zip upload of a different project → Edition 1 (independent counter)
4. GitHub URL submissions still use URL-based edition tracking (unchanged)
5. Existing tests pass

---

## Implementation Order

1. Bug 1 (fix prompt) — 10 minutes, add one pattern + update fallback text in documents.py
2. Bug 2 (edition counter) — 10 minutes, add `project_name` param to `get_next_edition`, update call site

## Files Modified

| Bug | File | Lines | Change |
|-----|------|-------|--------|
| 1 | `src/mycode/documents.py` | ~1277 | Add `_pat_http_endpoint_blocking` pattern |
| 1 | `src/mycode/documents.py` | ~1063 | Update `_FIX_OBJECTIVES["http_load_testing"]` fallback |
| 2 | `src/mycode/documents.py` | ~569 | Add `project_name` param to `get_next_edition` |
| 2 | `src/mycode/web/routes.py` | ~701 | Pass `project_name` for zip uploads |
