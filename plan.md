# Plan: Promote Confirmed Corpus Patterns into Library Profiles

## Overview

271 unique patterns extracted from 2,081 repos. 33 patterns appear in 10+ repos. This plan fixes classification gaps, enriches profiles with corpus-confirmed counts, and adds 2 new remediation patterns.

---

## Step 1: Fix Classification Gaps in `classifiers.py`

**File:** `src/mycode/classifiers.py`

The root cause: `http_load_testing` is not in `_CATEGORY_DOMAIN_MAP`, and `"missing"` / `"could not start"` / `"degradation"` are not in `_NAME_DOMAIN_KEYWORDS`. So these findings fall through to "unclassified".

### Changes to `_NAME_DOMAIN_KEYWORDS` (line ~170):
Add these entries (order matters — more specific first):
```python
("could not start", "dependency_failure"),       # 1,218 repos — server start failures
("missing depend", "dependency_failure"),          # 190+102 repos — missing deps
("degradation", "scaling_collapse"),               # 82 repos — response time degradation
```

**Why `_NAME_DOMAIN_KEYWORDS` and not `_CATEGORY_DOMAIN_MAP`?**
`http_load_testing` findings span multiple domains (start failures = dependency_failure, response time = scaling_collapse, concurrency = concurrency_failure). Adding `http_load_testing` to `_CATEGORY_DOMAIN_MAP` would force one domain for all. Title-based keywords are more precise.

### Changes to `_PATTERN_RULES` (line ~257):
Add pattern rules so the failure_pattern classifier also fires:
```python
("dependency_failure", ["could not start", "server", "startup"], "missing_server_dependency"),
("dependency_failure", ["missing depend", "unresolvable"], "unresolvable_dependency"),
("scaling_collapse", ["response time", "degradation", "latency"], "response_time_cliff"),
```

### No changes to `_CATEGORY_DOMAIN_MAP` — `http_load_testing` intentionally excluded.

---

## Step 2: Enrich Existing Profiles with `corpus_confirmed`

**Files:** 8 profile JSONs in `src/mycode/profiles/python/`

For each profile, add `corpus_confirmed` to entries in `known_failure_modes` where the corpus extraction has matching patterns. This is a new field — profiles currently have `name`, `description`, `trigger_conditions`, `severity`, `versions_affected`, `detection_hint`.

### pandas.json
Add `corpus_confirmed` to existing failure modes, and add new entries for patterns confirmed in corpus but missing from profile:

| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| data_volume_scaling / scaling_collapse | 283 | Add `corpus_confirmed: 283` to `memory_error_on_operations` (closest match) |
| silent_data_type_changes / input_handling | 60 | Add `corpus_confirmed: 60` to `silent_dtype_coercion` |
| memory_profiling_over_time / resource_exhaustion | 52 | New entry: `memory_growth_over_time` |
| merge_memory_stress / resource_exhaustion | 49 | Add `corpus_confirmed: 49` to `memory_error_on_operations` (merge is covered) |
| iterrows_vs_vectorized / scaling_collapse | 49 | Add `corpus_confirmed: 49` to `apply_performance_cliff` (same antipattern class) |
| memory_crash_on_data_ops / resource_exhaustion | 57 | Covered by `memory_error_on_operations`, add count |

### numpy.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| array_size_scaling / scaling_collapse | 142 | Add `corpus_confirmed: 142` to closest existing mode |
| repeated_allocation_memory / resource_exhaustion | 41 | Add `corpus_confirmed: 41` |

### streamlit.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| cache_memory_growth / resource_exhaustion | 296 | Add `corpus_confirmed: 296` to existing cache mode |

### requests.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| concurrent_request_load / concurrency_failure | 36 | Add `corpus_confirmed: 36` |
| large_download_memory / resource_exhaustion | 32 | Add `corpus_confirmed: 32` |
| timeout_behavior / integration_failure | 29 | Add `corpus_confirmed: 29` |
| session_vs_individual_performance / resource_exhaustion | 27 | Add `corpus_confirmed: 27` |

### httpx.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| async_concurrent_load / concurrency_failure | 19 | Add `corpus_confirmed: 19` |

### pydantic.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| validation_throughput / scaling_collapse | 22 | Add `corpus_confirmed: 22` |

### fastapi.json
| Pattern | Corpus Count | Action |
|---------|-------------|--------|
| server_start_failure / dependency_failure | shared 1,218 | Add `corpus_confirmed: 1218` (shared across frameworks) |
| response_time_degradation / scaling_collapse | shared 82 | Add `corpus_confirmed: 82` (shared) |

### flask.json
Same two patterns as fastapi — `corpus_confirmed: 1218` and `corpus_confirmed: 82`.

**Approach:** Read each profile, identify the matching `known_failure_modes` entry, add the `corpus_confirmed` field. If no matching entry exists (e.g., pandas `memory_growth_over_time`), add a new `known_failure_modes` entry with full schema.

---

## Step 3: Add 2 New Remediation Patterns in `documents.py`

**File:** `src/mycode/documents.py` (after line ~1260, before `_match_remediation`)

### Pattern 1: Pandas silent data type changes (60 repos)
```python
@_register_pattern
def _pat_pandas_silent_dtypes(f, framework, fields):
    if "pandas" in (f.affected_dependencies or []) and (
        f.failure_pattern == "silent_data_type_changes"
        or ("dtype" in f.title.lower() or "type" in f.title.lower())
        and f.category == "edge_case_input"
    ):
        return (
            "Pandas silently converts data types when input values don't match "
            "expected types. A single non-numeric value in an integer column "
            "converts the entire column to object dtype, increasing memory 10x "
            "and producing incorrect numeric operations without raising errors.",
            "Specify dtypes explicitly in read_csv(dtype={...}), use "
            "pd.to_numeric(errors='coerce') for controlled conversion, and "
            "validate column dtypes after loading with df.dtypes checks.",
        )
    return None
```

### Pattern 2: Requests concurrent load (36 repos)
```python
@_register_pattern
def _pat_requests_concurrent(f, framework, fields):
    if "requests" in (f.affected_dependencies or []) and (
        f.failure_domain == "concurrency_failure"
        or f.category == "concurrent_execution"
    ):
        return (
            f"The requests library is synchronous — each call blocks its "
            f"thread until the response arrives. At {fields['load']} concurrent "
            f"requests, all threads are occupied waiting on I/O and new "
            f"requests queue.",
            "Use requests.Session() for connection pooling, switch to "
            "httpx.AsyncClient or aiohttp for async I/O, or use "
            "concurrent.futures.ThreadPoolExecutor with a bounded pool size.",
        )
    return None
```

---

## Step 4: New Tests

**File:** `tests/test_classifiers.py` (or wherever classifier tests live)

### Test 1: Server start failure gets `dependency_failure`
```python
def test_server_start_classified_as_dependency_failure():
    result = classify_finding(
        scenario_name="Application server could not start",
        scenario_category="http_load_testing",
    )
    assert result["failure_domain"] == "dependency_failure"
    assert result["failure_pattern"] == "missing_server_dependency"
```

### Test 2: Response time degradation gets `scaling_collapse`
```python
def test_response_time_degradation_classified_as_scaling_collapse():
    result = classify_finding(
        scenario_name="Response time degradation on your application",
        scenario_category="http_load_testing",
    )
    assert result["failure_domain"] == "scaling_collapse"
    assert result["failure_pattern"] == "response_time_cliff"
```

### Test 3: Missing dependencies gets `dependency_failure`
```python
def test_missing_deps_classified_as_dependency_failure():
    result = classify_finding(
        scenario_name="4 missing dependencies",
        scenario_category="",
    )
    assert result["failure_domain"] == "dependency_failure"
    assert result["failure_pattern"] == "unresolvable_dependency"
```

---

## Step 5: Run Tests

Run the fast test suite to verify nothing breaks:
```
pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_session.py --ignore=tests/test_pipeline.py -k "not (TestPipelineIntegration or TestCLIExitCode)"
```

---

## What Is NOT Changed

- `scripts/corpus_extract.py` — untouched
- `corpus_extraction/` output files — untouched
- Test logic / scenario generation code — untouched
- Profile JSON schema — only additive (`corpus_confirmed` field added, no restructuring)

---

## Risk Assessment

- **Low risk:** Classifier changes only affect _new_ reports. Existing extraction data and reports are unaffected.
- **Low risk:** Profile `corpus_confirmed` is a new field — nothing reads it yet. It's metadata for human review and future library enrichment.
- **Low risk:** Remediation patterns are additive — new patterns checked after existing ones, so no change to current behaviour.
- **Medium risk:** The `"could not start"` keyword in `_NAME_DOMAIN_KEYWORDS` is broad. But it's checked against `name_lower` (the finding title), and this exact phrase only appears in HTTP load testing findings. No false positives expected.
