# Plan: Improve unsupported-language error message

## Problem

When a user submits a project in an unsupported language (Kotlin, Java, Go, etc.), the error they see is:

> "Preflight analysis failed: Could not determine project language — no Python or JavaScript files found."

This is confusing because:
1. "Could not determine project language" implies myCode failed to detect, not that the language isn't supported
2. "No Python or JavaScript files found" doesn't tell the user what to do about it
3. The "Preflight analysis failed:" prefix (added by the generic `except Exception` handler in routes.py) makes it sound like an internal error

## Root cause

- **`src/mycode/pipeline.py:276-280`** — `detect_languages()` raises `LanguageDetectionError("Could not determine project language — no Python or JavaScript files found.")`
- **`src/mycode/web/routes.py:301-309`** — `handle_preflight()` catches this via the generic `except Exception` handler, wrapping it as `"Preflight analysis failed: {exc}"`
- **`web/app.js:145-148`** — Frontend displays `data.error` in a red `viability-banner fail` div. This part works fine — it cleanly renders whatever string the backend sends.

## Changes

### 1. `src/mycode/pipeline.py` — Update error message in `detect_languages()` (line 277-280)

Change the `LanguageDetectionError` message from:
```
"Could not determine project language — no Python or JavaScript files found."
```
to:
```
"myCode currently supports Python and JavaScript/TypeScript projects. Your project doesn't appear to use a supported language. If your project does use Python or JS/TS, make sure it has a requirements.txt, pyproject.toml, setup.py, or package.json in the repository."
```

### 2. `src/mycode/web/routes.py` — Catch `LanguageDetectionError` specifically in `handle_preflight()`

Add a dedicated `except` clause for `LanguageDetectionError` **before** the generic `except Exception` clause (between lines 296 and 302). This returns the message directly without the "Preflight analysis failed:" prefix:

```python
except LanguageDetectionError as exc:
    job.status = "preflight_failed"
    job.error = str(exc)
    return PreflightResponse(job_id=job.id, error=str(exc))
```

Requires adding `LanguageDetectionError` to the existing import from `mycode.pipeline` at line 19.

### 3. `web/app.js` — No changes needed

The frontend already displays preflight errors cleanly in a styled red banner using `escapeHtml()`. The longer message text will wrap naturally. No raw JSON is shown — the `data.error` string is rendered directly into the banner HTML.

## Files touched

| File | Change | Lines affected |
|------|--------|----------------|
| `src/mycode/pipeline.py` | Update error message string | ~3 lines |
| `src/mycode/web/routes.py` | Add import + dedicated except clause | ~5 lines |

## Tests

- Any existing tests that assert on the old `LanguageDetectionError` message text will need their expected string updated
- No new test files needed
- Run fast suite to confirm no regressions
