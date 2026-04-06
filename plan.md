# Plan: Two Bug Fixes

**STATUS: IMPLEMENTED — verified and committed**

---

## Bug A: Flask Sandbox Endpoint Interference

### Symptom

When a Flask project has multiple endpoints with varying response times, fast endpoints are falsely flagged as slow. Confirmed across 4 test runs with `~/Desktop/test-dashboard-fixed.zip`: `/slow-report` (no `time.sleep`, confirmed absent) consistently measures ~15s because `/process-data` (pandas `iterrows()`) occupies the Flask dev server's single worker thread.

### How Endpoints Are Currently Tested

Endpoints are tested **sequentially** in a loop (`http_load_driver.py:475`). Each endpoint goes through all concurrency levels before the next one starts. Within each level, concurrent requests are fired via `ThreadPoolExecutor` (`http_load_driver.py:241-245`).

The server is started **once** (`server_manager.py:432-438`) and shared across all endpoint tests. There is **no isolation** between endpoint tests — no server restart, no drain/cooldown between endpoints.

### Root Cause

**Flask dev server defaults to single-threaded mode.** The startup command at `server_manager.py:432-438`:

```python
if fw == "flask":
    cmd = [
        "python", "-m", "flask", "run",
        "--port", str(port),
    ]
    env = {"FLASK_APP": detection.entry_file}
    return cmd, env
```

No `--with-threads` flag. Werkzeug's dev server runs single-threaded by default: it handles one request at a time, queuing all others.

**The interference mechanism:** Endpoint A (`/process-data`) is tested first with concurrent requests. At higher concurrency levels, some requests may still be processing or the server thread may still be occupied when Endpoint B (`/slow-report`) starts its baseline test. On a single-threaded server, Endpoint B's baseline request (concurrency=1) queues behind Endpoint A's lingering work. With `_REQUEST_TIMEOUT_SECONDS = 15` and `_SLOW_BASELINE_MS = 10_000`, a fast endpoint waiting 10+ seconds in queue triggers `external_dependency_timeout` at line 361-371 and is skipped.

Even without cross-endpoint queueing, the single-threaded server means that *within* an endpoint's own test at concurrency > 1, only one request is processed at a time. Every concurrent request waits for all prior ones. This inflates response times for ALL endpoints at ANY concurrency level, not just the falsely-flagged fast ones. The measurements are fundamentally wrong for any Flask app tested this way.

### Files and Line Numbers

| File | Lines | What |
|------|-------|------|
| `src/mycode/server_manager.py` | 432-438 | Flask startup command — missing `--with-threads` |
| `src/mycode/http_load_driver.py` | 76 | `_ROUNDS_PER_LEVEL = 3` |
| `src/mycode/http_load_driver.py` | 82 | `_REQUEST_TIMEOUT_SECONDS = 15` |
| `src/mycode/http_load_driver.py` | 90 | `_SLOW_BASELINE_MS = 10_000` |
| `src/mycode/http_load_driver.py` | 230-285 | `_drive_single_round()` — concurrent requests via ThreadPoolExecutor |
| `src/mycode/http_load_driver.py` | 288-321 | `drive_load_level()` — 3 rounds, median selection |
| `src/mycode/http_load_driver.py` | 324-391 | `drive_endpoint()` — per-endpoint loop with baseline abort check |
| `src/mycode/http_load_driver.py` | 361-371 | Baseline >10s → `external_dependency_timeout` skip |
| `src/mycode/http_load_driver.py` | 473-498 | Main endpoint iteration loop — sequential, shared server |

### Proposed Fix

**Enforce `--with-threads` on Flask dev server startup.**

In `server_manager.py:432-438`, add the threading flag:

```python
if fw == "flask":
    cmd = [
        "python", "-m", "flask", "run",
        "--port", str(port),
        "--with-threads",
    ]
    env = {"FLASK_APP": detection.entry_file}
    return cmd, env
```

**Why this is the correct fix:**

1. **Matches real deployment.** Production Flask apps always run behind multi-worker/threaded WSGI servers (gunicorn, uwsgi). Single-threaded testing produces misleading results that don't represent how the app will actually behave.
2. **Fixes the root cause.** Endpoint requests won't queue behind each other. Each request gets its own thread.
3. **One-line change.** Minimal blast radius — only affects Flask apps, only adds threading.
4. **No threshold tuning needed.** The measurements themselves become correct, so existing thresholds work as designed.

**Alternatives considered and rejected:**

- *Restart server between endpoint tests:* Adds 30-120s per endpoint (health check wait). Wastes time budget. Doesn't fix within-endpoint concurrency measurements being wrong.
- *Detect single-threaded Flask and adjust thresholds:* Masks the problem. Measurements are fundamentally wrong on a single-threaded server at any concurrency > 1 — no threshold adjustment can correct that.
- *Add drain/cooldown between endpoint tests:* Band-aid for cross-endpoint interference only. Doesn't fix within-endpoint concurrency inflation.

### Verification

**Pass condition with `~/Desktop/test-dashboard-fixed.zip`:**
- `/slow-report` must NOT be flagged as slow — no `external_dependency_timeout`, baseline median well below 10,000ms
- `/process-data` (pandas `iterrows()`) must STILL be flagged as genuinely slow
- Both endpoints must receive load testing at all concurrency levels (no premature skip for `/slow-report`)

**Unit test:** Assert `build_startup_command()` includes `--with-threads` when framework is `"flask"`.

---

## Bug B: Edition N Footer Says "This is your first myCode report"

### Symptom

Edition 2, 3, and 4 reports all display: *"This is your first myCode report. Future reports will show changes from your previous assessment."* The edition number in the header increments correctly, but the footer text never updates.

### Root Cause

**The footer text is hardcoded with no conditional on edition number.** At `documents.py:2763-2773`:

```python
# ── Historical comparison placeholder ──
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*_SUBTLE)
pdf.multi_cell(
    0, 4,
    _safe_text(
        "This is your first myCode report. Future reports will show "
        "changes from your previous assessment. Run myCode again after "
        "making improvements to track your progress."
    ),
)
```

The `edition` parameter is in scope — the function signature at line 2417 is `def render_understanding_pdf(report, edition, ...)`, and `edition` is already used at line 2466 for the header info row. But the footer block at line 2763 ignores it.

### Files and Line Numbers

| File | Lines | What |
|------|-------|------|
| `src/mycode/documents.py` | 2417 | Function signature — `edition: int` parameter available |
| `src/mycode/documents.py` | 2466 | Edition used in header (works correctly) |
| `src/mycode/documents.py` | 2763-2773 | Hardcoded footer text — **the bug** |
| `src/mycode/documents.py` | 586-634 | `get_next_edition()` — edition counter logic |
| `src/mycode/documents.py` | 2977 | `write_edition_documents()` — calls `render_understanding_pdf()` |

### "Changes from previous" Comparison Feature

The comment at line 2763 (`# ── Historical comparison placeholder ──`) confirms this was planned but never built. No comparison logic exists anywhere in the codebase. The edition 2+ footer text should NOT reference "changes noted above" — that feature doesn't exist yet.

### Proposed Fix

Replace the hardcoded text at `documents.py:2766-2773` with an edition-aware conditional:

```python
# ── Historical comparison placeholder ──
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*_SUBTLE)
if edition <= 1:
    footer_text = (
        "This is your first myCode report. Future reports will show "
        "changes from your previous assessment. Run myCode again after "
        "making improvements to track your progress."
    )
else:
    footer_text = (
        f"This is Edition {edition} of your myCode report. "
        "Run myCode again after making improvements to continue "
        "tracking your progress."
    )
pdf.multi_cell(0, 4, _safe_text(footer_text))
```

**Why this wording for edition 2+:** It acknowledges the edition number without promising a comparison feature that doesn't exist. When the historical comparison feature ships later, update the text to reference it (e.g., "Changes from your previous assessment are noted above.").

### Verification

**Pass condition:**
- Edition 1 report: footer says "This is your first myCode report. Future reports will show changes..."
- Edition 2 report: footer says "This is Edition 2 of your myCode report. Run myCode again..."
- Edition 5 report: footer says "This is Edition 5 of your myCode report. Run myCode again..."
- Edition number in footer matches edition number in header

**Unit test:** Call `render_understanding_pdf()` with `edition=1` and `edition=3`, extract text from each PDF, assert correct footer text appears.

---

## Implementation Order

1. **Bug A** — one-line change in `server_manager.py`, plus unit test
2. **Bug B** — conditional in `documents.py`, plus unit test

## Files Modified

| Bug | File | Lines | Change |
|-----|------|-------|--------|
| A | `src/mycode/server_manager.py` | 435 | Add `"--with-threads"` to Flask startup command |
| B | `src/mycode/documents.py` | 2766-2773 | Replace hardcoded footer with edition-aware conditional |
