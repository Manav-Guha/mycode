# Plan: Two Fixes — Coverage Warning + Health Endpoint

**STATUS: AWAITING REVIEW**

---

## Fix 1: Coverage Warning for Low Scenario Count

### Problem

A project with 0-1 recognized dependencies and no coupling points can produce 0-1 scenarios. The report says "All stress test scenarios completed cleanly" identically whether 1 scenario ran or 30 ran. A 1-scenario clean pass gives false confidence — the user thinks their project was thoroughly tested.

Confirmed path to 1 scenario:
- `scenario.py:1284-1307` — the only guaranteed scenario is `unrecognized_deps_generic_stress`, which fires when the `unrecognized` list is non-empty.
- If a project has zero recognized deps, zero unrecognized deps, and no coupling points (≥3 callers or ≥2 global mutators, `ingester.py:1020-1092`), the scenario list is **empty**.
- The pipeline handles this with only a warning string (`pipeline.py:1152-1155`), not a user-visible alert.

### Threshold

**≤3 scenarios.** Rationale:
- A single recognized dep with a profile generates 2-5 template scenarios plus 1-2 failure mode scenarios. A project with even one profiled dep gets ≥5 scenarios.
- ≤3 scenarios means: at most one generic stress test + one or two coupling scenarios. This is not meaningful coverage.
- The median scenario count across corpus reports is ~15-20. Anything ≤3 is bottom-5th-percentile.

### Where the warning should appear

**Both the report and the web UI.**

#### A. Report (PDF + text + JSON)

**File:** `src/mycode/report.py`
**Location:** After the stats bar and before findings, in the `as_text()` method at line ~244 and `as_markdown()` method at line ~472.

**Proposed check:** Add after the existing scenario coverage summary block (`report.py:244-252`):

```python
# Low coverage warning
if self.scenarios_run <= 3 and not self.findings:
    sections.append(
        "\n⚠ Limited test coverage: myCode ran only "
        f"{self.scenarios_run} scenario{'s' if self.scenarios_run != 1 else ''}. "
        "This usually means your project's dependencies don't have "
        "detailed test profiles in myCode's library yet. A clean "
        "result with limited coverage means fewer things were checked, "
        "not that nothing can go wrong."
    )
```

Mirror the same text in `as_markdown()` at line ~472.

**Also in JSON output:** Add a boolean field `low_coverage` to the `as_dict()` method (`report.py:696+`) under `statistics`:

```python
"low_coverage": self.scenarios_run <= 3,
```

#### B. Web UI

**File:** `src/mycode/web/worker.py` (line 94 sets `progress_scenarios_total`)
**Location:** After job completion, the web frontend reads the report JSON. The `low_coverage` field in the JSON output is sufficient — the frontend can check it and display a banner.

No backend change needed beyond the JSON field. Frontend change: check `report.statistics.low_coverage` and show a yellow banner with the same text.

### Visual distinction

**Yes — a low-scenario clean pass should look different from a full-coverage clean pass.**

Currently, the report executive summary uses a score label system. Proposed change in `_compute_score_label()` or equivalent: when `scenarios_run <= 3` and no findings, use a label like **"Limited check — passed"** instead of the unconditional **"Passed"**.

**File:** `src/mycode/report.py`, in the summary/score rendering section (~line 450-468).

Add before the existing score logic:

```python
if total <= 3 and not self.findings:
    score_label = "Limited coverage — no issues found"
```

### Acceptance criteria

1. A project with 1 scenario and no findings shows "Limited test coverage" warning in report text, markdown, and PDF.
2. Report JSON includes `"low_coverage": true` in statistics.
3. Score label distinguishes limited-coverage pass from full-coverage pass.
4. A project with 10+ scenarios and no findings shows no warning (no regression).
5. Existing tests pass.

---

## Fix 2: /api/health Endpoint Slowness

### Problem

The `/api/health` endpoint takes >10 seconds on first hit (or every 5 minutes when cache expires). This causes:
1. The HTTP load driver flags myCode's own health endpoint as slow when self-testing (`_SLOW_BASELINE_MS = 10_000` at `http_load_driver.py:90`).
2. External uptime monitors (Railway, UptimeRobot) see timeouts.
3. The endpoint blocks the async event loop during the slow path.

### What it currently does

**File:** `src/mycode/web/routes.py:893-919` (`handle_health()`)
**Called from:** `src/mycode/web/app.py:400-404` (`health()` async endpoint)

```python
def handle_health() -> HealthResponse:
    now = time.time()
    if "result" not in _docker_cache or now - _docker_cache.get("ts", 0) > 300:
        from mycode.container import is_docker_available   # lazy import
        _docker_cache["result"] = is_docker_available()    # BLOCKING: subprocess
        _docker_cache["ts"] = now
    ...
```

`is_docker_available()` (`container.py:35-49`) runs `docker info` as a **blocking subprocess** with a **10-second timeout**. On Railway (where Docker is not available), this blocks for the full 10 seconds before timing out.

The async endpoint calls `handle_health()` **synchronously** — not via `asyncio.to_thread()`. This blocks the entire event loop for up to 10 seconds.

The 5-minute cache (`_docker_cache`) means this only happens once per 5 minutes, but that's enough to trigger the slow-baseline detector on self-tests and cause uptime monitor alerts.

### What a health endpoint should do

Return immediately with server status. No subprocess calls, no external dependencies, no blocking I/O. Health endpoints exist for load balancers and monitors to verify the process is alive and accepting requests.

### Proposed fix

**Remove the Docker check from the health endpoint entirely.** Docker availability is irrelevant to server health — it's a feature flag, not a liveness signal.

**File:** `src/mycode/web/routes.py:893-919`

Replace `handle_health()` with:

```python
def handle_health() -> HealthResponse:
    """Return server health status. Must be fast — no subprocess calls."""
    version = "0.1.2"
    try:
        import importlib.metadata
        version = importlib.metadata.version("mycode-ai")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        docker_available=False,  # Railway doesn't have Docker; remove field in next schema bump
        version=version,
        active_jobs=store.active_count(),
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
    )
```

This removes:
- The `_docker_cache` dict (lines 889-890) — no longer needed
- The `is_docker_available()` import and call (lines 896-902)
- The blocking subprocess on the event loop

**`docker_available` field:** Set to `False` unconditionally. Railway doesn't have Docker. If Docker detection is needed elsewhere (e.g., for the `--containerised` flag), it should be checked at job submission time, not on every health poll. The `HealthResponse` schema (`web/schemas.py:156-161`) can keep the field for backward compatibility; remove it in a future schema bump.

**Alternative (if Docker status must stay on health):** Move the check to a background task that runs on startup and every 5 minutes, storing the result in a module-level variable. The health endpoint reads the cached value without blocking. But this is overengineering — Docker isn't available on Railway and the field serves no current purpose.

### Acceptance criteria

1. `GET /api/health` returns in <100ms consistently (no 10-second spikes).
2. Response still includes `status`, `version`, `active_jobs`, `max_concurrent_jobs`.
3. `docker_available` field is `False` (matches Railway reality).
4. No blocking subprocess call on the async event loop.
5. Existing health-endpoint tests pass (update expected `docker_available` if tests assert `True`).

---

## Implementation Order

1. Fix 2 first (health endpoint) — 10 minutes, isolated change, no risk.
2. Fix 1 second (coverage warning) — 30 minutes, touches report rendering in multiple output formats.

## Files Modified

| Fix | File | Lines | Change |
|-----|------|-------|--------|
| 2 | `src/mycode/web/routes.py` | 889-919 | Remove Docker check, simplify `handle_health()` |
| 1 | `src/mycode/report.py` | ~244, ~472, ~696 | Add low-coverage warning in text/markdown/JSON |
| 1 | `src/mycode/report.py` | ~450-468 | Distinguish score label for limited coverage |
