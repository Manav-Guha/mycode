# Plan: Three Bug Fixes

**STATUS: AWAITING REVIEW**

---

## Bug 1: Fix Prompt Mismatch for "Application server could not start — missing dependency"

### Root Cause

`_pat_flask_concurrency()` in `documents.py:1213-1229` matches **all** Flask findings with `category == "http_load_testing"`, including startup failures. It does not check the title or `failure_pattern`. Since there is no `_pat_flask_startup()` pattern, a Flask startup failure (with `failure_pattern="missing_server_dependency"`) matches the concurrency pattern and gets: "Use database connection pooling (SQLAlchemy pool_size)..."

FastAPI has a startup-specific pattern (`_pat_fastapi_startup`, `documents.py:1157-1171`) that checks `"could not start" in f.title.lower()`. Flask does not.

### Evidence

- `http_load_driver.py:1149-1154`: When server fails with "missing dependency" in error, sets `failure_pattern="missing_server_dependency"` and `diagnosis="...a required dependency is missing."`
- `documents.py:1213-1220`: `_pat_flask_concurrency` checks only `framework == "flask"` and `f.category in ("http_load_testing", ...)` — no exclusion for startup failures
- Pattern registry iterates in registration order; `_pat_flask_concurrency` (line 1213) fires before any startup check could run

### Proposed Fix

Add `_pat_startup_failure()` as a **framework-agnostic** startup pattern, registered **before** `_pat_flask_concurrency`. This handles all frameworks, not just Flask, and checks `failure_pattern` for specific remediation.

**File:** `src/mycode/documents.py`
**Insert:** Before `_pat_flask_concurrency` (line 1213)

```python
@_register_pattern
def _pat_startup_failure(f, framework, fields):
    if "could not start" not in f.title.lower():
        return None
    fp = f.failure_pattern or ""
    if fp == "missing_server_dependency":
        return (
            f"Your {framework} app failed to start because a required "
            f"dependency is missing from your project.",
            f"Check that all imports in your main app module are listed "
            f"in your requirements.txt or package.json. Run your app "
            f"locally to see which import fails, then add the missing "
            f"package.",
        )
    if fp == "missing_env_config":
        return (
            f"Your {framework} app failed to start because required "
            f"environment variables are not set.",
            f"Create a .env file with the required variables, or set "
            f"them in your hosting platform's environment settings. "
            f"Check your app's documentation or config file for the "
            f"expected variable names.",
        )
    if fp == "missing_external_service":
        return (
            f"Your {framework} app failed to start because it cannot "
            f"connect to an external service (database, cache, API).",
            f"Make sure your database or external service is running "
            f"and the connection string is correct. For local "
            f"development, check that Docker containers or local "
            f"services are started.",
        )
    if fp == "server_syntax_error":
        return (
            f"Your {framework} app failed to start due to a syntax "
            f"error in the code.",
            f"Run your app locally — the error message will point to "
            f"the exact file and line. Fix the syntax error and retry.",
        )
    # Generic startup failure
    return (
        f"Your {framework} app failed to start. This prevents all "
        f"users from accessing your application.",
        f"Run your app locally to reproduce the error. Check the "
        f"startup logs for the specific failure reason.",
    )
```

**Also remove** `_pat_fastapi_startup` (lines 1157-1171) since the new pattern covers all frameworks including FastAPI.

### Acceptance Criteria

1. Flask project with missing dependency gets: "Check that all imports...are listed in your requirements.txt"
2. Flask project with env variable issue gets: "Create a .env file..."
3. FastAPI startup failures still get correct prompts (covered by same pattern)
4. Flask concurrency findings (non-startup) still get the pooling/gunicorn advice
5. Existing tests pass

---

## Bug 2: React/JS Packages Flagged as Missing pip Dependencies

### Root Cause

When a project is detected as **multi-language** (has both `requirements.txt` and `package.json`), or when a Python project has a `package.json` from a build dependency, the Python ingester extracts dependencies only from `requirements.txt` — this is correct. But the viability gate and report may still reference JS packages.

The actual bug path: The report's missing-dependency rendering in `report.py:2088` uses a heuristic `is_js = any(d.name.startswith("@") for d in ingestion.dependencies)` to decide whether to show "declared but not installed" (Python) vs "no stress profile available" (JS). Packages like `react`, `react-dom`, `react-scripts` do NOT start with `@`, so they're treated as Python packages and flagged as "declared but not installed."

This happens when:
1. A React project is misclassified as Python (unlikely but possible if it has a `requirements.txt`)
2. The multi-language ingestion picks up both Python and JS deps, but renders all missing deps using the Python template
3. The `@` heuristic misses non-scoped JS packages

### Evidence

- `report.py:2088`: `is_js = any(d.name.startswith("@") for d in ingestion.dependencies)` — fails for `react`, `react-dom`, `react-scripts`
- `ingester.py:683`: Sets `is_missing=True` when a dep is not found via `importlib.metadata` (Python-only check)
- `pipeline.py:847-860`: Language router sends Python projects through `ProjectIngester` which only knows pip

### Proposed Fix

Replace the `@`-prefix heuristic with the actual language from the pipeline. The `IngestionResult` already has a `language` field (`ingester.py:209`).

**File:** `src/mycode/report.py`, line ~2088
**Change:** Replace `is_js = any(d.name.startswith("@") ...` with `is_js = ingestion.language == "javascript"`

The `_build_confidence_note` function and the `_record_unrecognized_deps` method both receive `ingestion` — the language is available.

Additionally, in the dependency coverage section (`report.py:2203-2230`), when `ingestion.language == "javascript"`, deps should not be flagged as "declared in requirements but not installed" — they should be flagged as "no stress profile available" (info, not warning).

### Acceptance Criteria

1. A JavaScript project's deps (react, react-dom) show as "no stress profile" (info), not "declared but not installed" (warning)
2. A Python project's missing deps still show as "declared but not installed" (warning)
3. Multi-language projects use per-dep language awareness (deps from package.json are JS, deps from requirements.txt are Python)
4. Existing tests pass

---

## Bug 3: Project Name Extraction Pulling Wrong Name

### Root Cause

`_infer_project_name()` in `pipeline.py:993-1035` reads `package.json` `name` field as the project name. If the repo's `package.json` has `"name": "sumo-unity3d-connection"`, the project is named "Sumo Unity3d Connection" regardless of what the user typed.

The user's description ("TRAFFIC REGULATOR APPLICATION") goes into `constraints.project_description` at `routes.py:806`. This is stored in `report.user_project_description` at `report.py:1073`. But the report's `display_name` at `report.py:438` uses `self.project_description` (auto-generated) first, which incorporates the `package.json` name.

The priority chain is:
1. `report.py:438`: `display_name = self.project_description or project_name or "Your Project"`
2. `self.project_description` is set by `_generate_project_description()` at `report.py:1097` which uses `project_name` (from `_infer_project_name()` → package.json)
3. The user's stated name in `constraints.project_description` is stored but **never used as the display name**

### Evidence

- `pipeline.py:1020-1028`: Reads `package.json` `name` field, title-cases it → "Sumo Unity3d Connection"
- `routes.py:246`: `project_name = _infer_project_name(project_path)` — set at preflight, before user submits intent
- `routes.py:806`: User's description stored as `constraints.project_description`
- `report.py:1073`: Stored as `report.user_project_description`
- `report.py:438`: `display_name` uses `self.project_description` (auto-generated), never `user_project_description`

### Proposed Fix

When `user_project_description` is set, use it as the display name in preference to the auto-generated description. The auto-generated description can still appear as supplementary context.

**File:** `src/mycode/report.py`, line ~438
**Change:**

```python
# Before:
display_name = self.project_description or project_name or "Your Project"

# After:
display_name = (
    self.user_project_description
    or self.project_description
    or project_name
    or "Your Project"
)
```

Apply the same change in `as_text()` (line ~215 area, wherever display_name is computed for text output).

Also apply in `_generate_project_description()` (`report.py:3315`): when a `user_project_description` exists, use it as the project name parameter instead of the inferred name.

### Acceptance Criteria

1. A project submitted with description "TRAFFIC REGULATOR APPLICATION" shows that as the report title, not "Sumo Unity3d Connection"
2. A project submitted without a description still shows the auto-detected name from package.json/pyproject.toml
3. The auto-generated technical description still appears in the report body (not lost)
4. Existing tests pass

---

## Implementation Order

1. Bug 1 (fix prompt) — 15 minutes, isolated to documents.py pattern registry
2. Bug 3 (project name) — 10 minutes, isolated to report.py display name logic
3. Bug 2 (JS deps) — 20 minutes, touches report rendering and needs careful language-awareness threading

## Files Modified

| Bug | File | Lines | Change |
|-----|------|-------|--------|
| 1 | `src/mycode/documents.py` | ~1157-1229 | Add `_pat_startup_failure`, remove `_pat_fastapi_startup` |
| 3 | `src/mycode/report.py` | ~438, ~215 | Prefer `user_project_description` for display name |
| 2 | `src/mycode/report.py` | ~2088 | Replace `@`-prefix heuristic with `ingestion.language` |
