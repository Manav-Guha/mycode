# Plan: Phase 4 — Measurement-Based Pattern Gates (Revised)

**STATUS: AWAITING REVIEW (Revision 3)**

## Overview

Implement measurement-based gates for six remediation patterns in `src/mycode/documents.py`, per ADR-001 through ADR-006. This adds a threshold provider abstraction, three blanket rules, per-pattern gate logic tied to corpus-calibrated thresholds, updated narrative templates, and the ADR-004 pattern rename.

**Authoritative specs:**
- myCode-Architecture-Decision-Record.md (ADR-001 through ADR-006)
- myCode-Product-Architecture.md (Sections 1 and 2)
- phase4-pattern-audit.md (original weak-gate analysis)
- phase4-corpus-distributions.md (corpus distributions informing Method B thresholds)

---

## 1. Threshold Provider Abstraction

### File location

`src/mycode/thresholds.py` — new file.

### Interface

```python
def get_thresholds(pattern_name: str) -> dict:
    """Return threshold values for a given pattern.

    Returns a dict of threshold key → value, e.g.:
        {"peak_memory_mb": 47, "error_count": 11}
    """

def describe_calibration(pattern_name: str) -> str:
    """Return a human-readable description of how thresholds were set.

    Used for explainability in diagnostic output (Product Architecture
    Section 2 user-comprehension principle).
    """
```

### v1 implementation

Static dict lookup. No project_metadata parameter in v1 — thresholds are fixed per-pattern from the ADR values. The signature accepts `pattern_name` only.

```python
_THRESHOLDS = {
    "data_volume": {
        "peak_memory_mb": 47,
        "error_count": 11,
    },
    "cascading_timeout": {
        "error_count": 3,
    },
    "unbounded_cache_growth": {
        "peak_memory_mb": 50,
    },
    "input_handling_failure": {
        "error_count": 39,
    },
    "flask_concurrency": {
        "load_level": 5,
    },
    "requests_concurrent": {
        "load_level": 2,
        "error_count": 91,
    },
}

_CALIBRATION_DESCRIPTIONS = {
    "data_volume": "Thresholds set at p25 of corpus distribution (9,297 reports, N=2,921 data_volume_scaling findings). peak_memory_mb=47 is the 25th percentile; error_count=11 is the 25th percentile.",
    "cascading_timeout": "Threshold set at p25 of corpus error_count distribution (N=467 cascading_timeout findings). error_count=3 captures 75% of real cascading timeouts.",
    "unbounded_cache_growth": "Threshold set at 50MB — below p25 of general memory findings. Requires has_cache_decorator to confirm a cache mechanism exists in user code.",
    "input_handling_failure": "error_count=39 is the p25 of corpus distribution (N=143 unvalidated_type_crash findings). Exception marker requirement ensures the diagnosis names the actual exception type.",
    "flask_concurrency": "load_level=5 is the p25 of corpus distribution (N=886 flask concurrency findings). Thread saturation at <5 concurrent requests is implausible.",
    "requests_concurrent": "load_level=2 and error_count=91 are from corpus distribution (N=219). Requires I/O marker in details to confirm requests-specific failure.",
}
```

### Convention alignment

Follows the same swappable-backend pattern as the LLM backend (`src/mycode/llm.py`) and component library loader — a module-level interface with a static implementation that can be replaced by a dynamic backend (e.g., per-project calibration from corpus data) without changing call sites.

### Why not a class?

The LLM backend uses module-level functions, not a class. The threshold provider follows the same convention for consistency. If a future version needs per-project calibration, the function signatures stay the same; only the internal implementation changes.

### describe_calibration() rendering (Product Architecture Section 1.6)

The ADRs require the threshold-setting method to be visible to the user. `describe_calibration()` is consumed in Phase 4, not deferred to Phase 8.

**Rendering approach: footer line appended to each diagnosis string.**

Each narrative helper function appends a calibration disclosure as the final line of the `diagnosis` string:

```python
from mycode.thresholds import get_thresholds, describe_calibration

def _flask_concurrency_narrative(f, fields) -> str:
    # ... build sections as before ...
    calibration = describe_calibration("flask_concurrency")
    sections.append(f"How this threshold was set: {calibration}")
    return " ".join(sections)
```

The same pattern applies to all six narrative functions/templates. For patterns that use inline gate logic rather than a helper function (e.g., `_pat_data_volume`), the calibration line is appended directly:

```python
calibration = describe_calibration("data_volume")
diagnosis = f"... {calibration}"
```

**Why a footer line, not a separate section or inline:**
- **Not a separate report section:** The calibration text is per-finding, not per-report. Each finding may use a different pattern with different thresholds. A separate section would require cross-referencing ("see Appendix B, row 3") which violates the user-comprehension principle.
- **Not inline in the narrative body:** Embedding "this threshold was set at p25 of 9,297 reports" mid-sentence interrupts the user's understanding of what happened. The user needs the "what" and "why it matters" first, then the "how we decided this matters" as supporting context.
- **Footer line:** Appears after the escape hatch (for 4-question narratives) or after the main diagnosis text (for simpler patterns). The user reads the finding, understands its relevance, then sees the methodological basis. This mirrors how scientific papers present results then methods.

**Output surface impact:**

| Surface | Rendering |
|---|---|
| **PDF "Why this matters"** | Calibration line appears as the final sentence of the diagnosis paragraph |
| **JSON `"diagnosis"` key** | Calibration line included in the diagnosis string |
| **Prompt for coding agent** | Calibration line is present but harmless — coding agents ignore methodological context |
| **Markdown** | Calibration line appears at end of diagnosis |

**Label prefix:** `"How this threshold was set: "` — uses a consistent label across all six patterns so the user learns to recognise it. The prefix is a plain-language question that frames the calibration text as an answer.

---

## 2. Blanket Rule Enforcement

### Approach: Pre-dispatch filter in `_match_remediation()`

A single function `_blanket_rules_pass(f, pattern_fn)` called at the top of the dispatch loop in `_match_remediation()`. This is preferable to a decorator because:
- The pattern functions are already registered via `@_register_pattern` — adding a second decorator creates ordering ambiguity
- A pre-dispatch filter keeps the blanket rules in one place, visible and auditable
- The concurrency-specific Rule 3 needs to know *which* pattern is being evaluated, which a decorator on the individual function couldn't determine without self-awareness

### Location

Inside `_match_remediation()` in `src/mycode/documents.py` (line 1870).

### Current code (lines 1870–1881):

```python
def _match_remediation(f: Finding) -> tuple[str, str] | None:
    framework = _detect_framework(f.affected_dependencies)
    fields = _remediation_fields(f)
    for pattern_fn in _REMEDIATION_PATTERNS:
        result = pattern_fn(f, framework, fields)
        if result is not None:
            return result
    return None
```

### New code:

```python
# Patterns subject to concurrency blanket rule (Rule 3)
_CONCURRENCY_PATTERNS = {
    _pat_flask_concurrency,
    _pat_requests_concurrent,
    _pat_cascading_timeout,
}

def _blanket_rules_pass(f: Finding, pattern_fn) -> bool:
    """Return False if blanket rules suppress this pattern for this finding."""
    # Rule 1: no pattern fires on info severity
    if f.severity == "info":
        return False
    # Rule 2: no pattern fires on clean findings
    if f._finding_type == "clean":
        return False
    # Rule 3: concurrency patterns require load_level >= 2 and non-null
    if pattern_fn in _CONCURRENCY_PATTERNS:
        if f._load_level is None or f._load_level < 2:
            return False
    return True

def _match_remediation(f: Finding) -> tuple[str, str] | None:
    framework = _detect_framework(f.affected_dependencies)
    fields = _remediation_fields(f)
    for pattern_fn in _REMEDIATION_PATTERNS:
        if not _blanket_rules_pass(f, pattern_fn):
            continue
        result = pattern_fn(f, framework, fields)
        if result is not None:
            return result
    return None
```

### Impact

- **Rule 1** suppresses all six patterns on `severity == "info"`. The corpus showed 25.6% of flask_concurrency matches were info-severity — these will now fall through to the generic fallback (which is the correct behaviour for a finding where nothing went wrong).
- **Rule 2** suppresses patterns on `_finding_type == "clean"`. This field was just added to serialization (commit 71fa8e4) and is populated by the engine.
- **Rule 3** suppresses concurrency patterns at `_load_level < 2` or null. The corpus showed median load_level=1 for requests_concurrent — single-connection findings will now be excluded from the thread-exhaustion narrative.
- Non-targeted patterns (e.g., `_pat_memory_baseline`, `_pat_pandas_silent_dtypes`) also benefit from Rules 1 and 2 as a side effect. This is correct — no pattern should diagnose an info/clean finding.

### Commit 1 test fixtures

The blanket rules will immediately suppress some existing test findings (e.g., tests that create info-severity findings and expect a pattern to match). However, examining the 12 affected tests from Section 6, none of the tests that currently pass use `severity="info"` or `_finding_type="clean"` — the blanket rules only affect real findings from production runs. The concurrency Rule 3 affects tests without `_load_level` set, but those tests are covered in Commits 2 and 3.

Therefore: Commit 1 adds `thresholds.py` and `_blanket_rules_pass` to `_match_remediation()` with 2 minimal test fixture additions (7.1, 7.2). Most existing tests pass unchanged because:
- No existing test creates a Finding with `severity="info"` and expects a pattern match
- No existing test creates a Finding with `_finding_type="clean"` (field defaults to `""`, which is not `"clean"`)
- Rule 3 only checks `_CONCURRENCY_PATTERNS` — the `_CONCURRENCY_PATTERNS` set references the pattern functions, which are unchanged in Commit 1. Tests that don't set `_load_level` will be fixed in Commits 2/3 when the pattern-specific gates are added.

Wait — Rule 3 will break `test_cascading_timeout_minimal` (no `_load_level`), `test_requests_concurrent_minimal` (no `_load_level`), and `test_requests_concurrent_full` (`_load_level=20` passes but it's a concurrency pattern — actually 20 ≥ 2, so it passes Rule 3). Let me re-check:

- `test_cascading_timeout_minimal` (line 2899): `_load_level` not set → defaults to `None` → Rule 3 blocks. **BREAKS.**
- `test_cascading_timeout_full` (line 2876): `_load_level=10` → 10 ≥ 2 → Rule 3 passes. OK.
- `test_requests_concurrent_minimal` (line 3017): `_load_level` not set → defaults to `None` → Rule 3 blocks. **BREAKS.**
- `test_requests_concurrent_full` (line 2994): `_load_level=20` → 20 ≥ 2 → Rule 3 passes. OK.
- `test_flask_concurrency_minimal` (line 2668): `_load_level=25` → 25 ≥ 2 → Rule 3 passes. OK.
- `test_flask_concurrency_full` (line 2645): `_load_level=25` → 25 ≥ 2 → Rule 3 passes. OK.

So Commit 1 breaks 2 tests via Rule 3. These need fixtures in Commit 1:
- `test_cascading_timeout_minimal`: add `f._load_level = 5`
- `test_requests_concurrent_minimal`: add `f._load_level = 5`

These are minimal fixture additions that satisfy the blanket rule without changing the pattern-specific gate (which doesn't change until Commits 2/3).

---

## 3. Per-Pattern Gate Logic

### 3.1 `_pat_data_volume` (ADR-001) — Commit 2

**Current gate** (`documents.py:1843`):
```python
if f.category == "data_volume_scaling":
```

**New gate:**
```python
if f.category == "data_volume_scaling":
    t = get_thresholds("data_volume")
    if not (f._peak_memory_mb >= t["peak_memory_mb"]
            or f._error_count >= t["error_count"]):
        return None
```

Gate keeps the category check, adds threshold check: peak_memory_mb ≥ 47 OR error_count ≥ 11. Either metric crossing its threshold is sufficient (OR logic per ADR-001).

**Fields read:** `f.category`, `f._peak_memory_mb`, `f._error_count`

### 3.2 `_pat_cascading_timeout` (ADR-002) — Commit 2

**Current gate** (`documents.py:1721`):
```python
if f.failure_pattern == "cascading_timeout":
```

**New gate:**
```python
if f.failure_pattern == "cascading_timeout":
    t = get_thresholds("cascading_timeout")
    if f._error_count < t["error_count"]:
        return None
```

Keeps the failure_pattern check, adds error_count ≥ 3.

**Narrative branching:** cascade vs single-point based on `len(f.call_chain) >= 2`. See Section 4.2.

**Fields read:** `f.failure_pattern`, `f._error_count`, `f.call_chain`

### 3.3 `_pat_unbounded_cache_growth` (ADR-003) — Commit 2

**Current gate** (`documents.py:1634`):
```python
if f.failure_pattern == "unbounded_cache_growth":
```

**New gate:**
```python
if f.failure_pattern == "unbounded_cache_growth":
    t = get_thresholds("unbounded_cache_growth")
    has_cache = fields["has_cache_decorator"]
    if not (has_cache and f._peak_memory_mb >= t["peak_memory_mb"]):
        return None
```

Requires both `has_cache_decorator == True` AND `peak_memory_mb >= 50`. The corpus showed 0% of unbounded_cache_growth findings had extractable peak_memory — this pattern will only fire on post-71fa8e4 enriched runs where the measurement fields are populated.

**Fields read:** `f.failure_pattern`, `f._peak_memory_mb`, `fields["has_cache_decorator"]`

### 3.4 `_pat_input_handling_failure` (ADR-004) — Commit 3

**Current gate** (`documents.py:1783`):
```python
if f.failure_pattern == "unvalidated_type_crash":
```

**New gate:**
```python
_EXCEPTION_MARKERS = (
    "TypeError", "ValueError", "AttributeError", "KeyError",
    "IndexError", "ValidationError", "JSONDecodeError",
)

if f.failure_pattern == "unvalidated_type_crash":
    t = get_thresholds("input_handling_failure")
    if f._error_count < t["error_count"]:
        return None
    details = f.details or ""
    matched_marker = None
    for marker in _EXCEPTION_MARKERS:
        if marker in details:
            matched_marker = marker
            break
    if matched_marker is None:
        return None
```

Requires error_count ≥ 39 AND at least one of 7 exception markers present in `f.details`. The `matched_marker` variable is used in the narrative for exception-type branching.

**Fields read:** `f.failure_pattern`, `f._error_count`, `f.details`

### 3.5 `_pat_flask_concurrency` (ADR-005) — Commit 3

**Current gate** (`documents.py:1530`):
```python
if (
    framework == "flask"
    and f.category in (
        "http_load_testing", "blocking_io", "concurrent_execution",
    )
):
```

**New gate:**
```python
if (
    framework == "flask"
    and f.category in (
        "http_load_testing", "blocking_io", "concurrent_execution",
    )
):
    t = get_thresholds("flask_concurrency")
    if f._load_level is None or f._load_level < t["load_level"]:
        return None
```

Adds load_level ≥ 5 AND non-null. Note: Rule 3 (blanket) already requires load_level ≥ 2 for this pattern; this pattern-specific gate raises the bar to ≥ 5.

**Fields read:** `f.category`, `f._load_level`, framework (derived from `f.affected_dependencies`)

### 3.6 `_pat_requests_concurrent` (ADR-006) — Commit 3

**Current gate** (`documents.py:1812`):
```python
if "requests" in (f.affected_dependencies or []) and (
    f.failure_domain == "concurrency_failure"
    or f.category == "concurrent_execution"
):
```

**New gate:**
```python
_IO_MARKERS = (
    "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout",
    "requests.exceptions", "ConnectionResetError", "SSLError",
)
```

Reverted to ADR spec markers. The original plan listed `urllib3` and `socket` instead of `ConnectionResetError` and `SSLError`. While `urllib3` and `socket` are legitimate `requests` internals, the ADR spec chose `ConnectionResetError` and `SSLError` because they are the exception types the user sees in their traceback — which aligns with the user-comprehension principle. The lower-level library names would appear in stack frames but are not the exception type the user needs to act on.

```python
if "requests" in (f.affected_dependencies or []) and (
    f.failure_domain == "concurrency_failure"
    or f.category == "concurrent_execution"
):
    t = get_thresholds("requests_concurrent")
    if f._load_level is None or f._load_level < t["load_level"]:
        return None
    if f._error_count < t["error_count"]:
        return None
    details = f.details or ""
    if not any(marker in details for marker in _IO_MARKERS):
        return None
```

Requires load_level ≥ 2 AND error_count ≥ 91 AND at least one I/O marker in details.

**Fields read:** `f.affected_dependencies`, `f.failure_domain`, `f.category`, `f._load_level`, `f._error_count`, `f.details`

---

## 4. Per-Pattern Narrative and Fix-Prompt Template Updates

### Architecture note: diagnosis vs fix output surfaces

The pattern functions return a `(diagnosis, fix)` tuple. These flow to different output surfaces:

| Surface | Consumes | Audience |
|---|---|---|
| **PDF "Why this matters"** section | `_build_diagnosis(f)` → `diagnosis` | Non-engineer user reading the Understanding Report |
| **JSON `"diagnosis"` key** | `_build_diagnosis(f)` → `diagnosis` | Developer/tool reading the JSON report |
| **JSON `"prompt"` key** | `generate_finding_prompt(f)` → embeds both `diagnosis` and `fix` | Coding agent (Claude Code, Cursor, etc.) |
| **PDF "What to do"** section | `generate_finding_prompt(f)` → includes `fix` within the prompt block | Non-engineer user pasting into their coding agent |
| **Markdown investigation** | `_build_diagnosis(f)` → `diagnosis` | Developer reading markdown output |
| **LLM fix enrichment** | `_build_diagnosis(f)` → provides context to LLM | LLM generating a more specific fix suggestion |

**Key observation:** The `diagnosis` string serves two audiences simultaneously:
1. Non-engineer users reading the PDF (plain language, operational framing)
2. Coding agents receiving the prompt (needs actionable technical detail)

The `fix` string serves one audience: coding agents. The user sees `fix` only indirectly — embedded inside the prompt block they paste into their coding agent.

**Implication for Phase 4:** The ADR's narrative-vs-fix-prompt distinction maps directly to the existing `(diagnosis, fix)` tuple:
- **Narrative** = `diagnosis` (first tuple element) — written for the non-engineer user, plain language, answers "what happened and why it matters"
- **Fix prompt** = `fix` (second tuple element) — written for the coding agent, technical, answers "what to change"

The current single-return-value structure is sufficient. No new fields or structural changes needed. The richer narratives from ADR-005 and ADR-006 go into the `diagnosis` string; the exception-branched fix advice from ADR-004 goes into the `fix` string.

### Narrative architecture: helper functions for multi-section narratives

ADR-005 and ADR-006 require multi-section narratives (4 questions, production symptoms, capacity framing, escape hatch). These are too complex for inline f-strings. Each will use a dedicated helper function:

```python
from mycode.thresholds import get_thresholds, describe_calibration

def _flask_concurrency_narrative(f, fields) -> str:
    """Build 4-question narrative for Flask concurrency findings.
    Appends calibration disclosure as footer."""
    ...

def _requests_concurrent_narrative(f, fields, io_marker) -> str:
    """Build structured narrative for requests concurrent findings.
    Appends calibration disclosure as footer."""
    ...

def _input_handling_narrative(f, fields, matched_marker) -> str:
    """Build exception-branched narrative for input handling findings.
    Appends calibration disclosure as footer."""
    ...
```

These helpers are called by their respective pattern functions and return the diagnosis string (including calibration footer). The pattern function still returns `(diagnosis, fix)` as before.

---

### 4.1 `_pat_data_volume` (ADR-001) — Commit 2

**Current narrative (representative):**
```python
f"Processing time grows with input size"
f"{'Peak memory: ' + mem + 'MB. ' if mem not in ('', 'high') else ''}"
```

**New narrative:** Keep the current template. The gate change (threshold) is sufficient — the template already interpolates the actual measurements. No narrative text change needed per ADR-001. Append calibration footer:

```python
calibration = describe_calibration("data_volume")
diagnosis += f" How this threshold was set: {calibration}"
```

**Current fix:** Chunked/streaming processing, pagination, cache intermediate results.
**New fix:** No change. The fix advice is correct regardless of the threshold.

### 4.2 `_pat_cascading_timeout` (ADR-002) — Commit 2

**Current narrative:**
```python
f"A slow dependency call triggers cascading timeouts in downstream functions. "
f"the slow call blocks its caller, which blocks its caller, "
f"until the entire request chain times out."
```

**New narrative — cascade branch** (when `len(f.call_chain) >= 2`):
Keep the current cascade narrative. The call_chain depth confirms the cascade claim.

**New narrative — single-point branch** (when `len(f.call_chain) < 2`):
```python
f"{loc + ' — a' if loc else 'A'} dependency call timed out"
f"{', taking ' + resp if resp else ''}"
f"{'at load level ' + fields['load'] if fields['load'] != 'high' else ''}. "
f"The slow response blocked the request handler, but myCode did not "
f"detect a multi-layer call chain — this appears to be a single-point "
f"timeout rather than a cascading failure."
f"{' ' + detail_excerpt if detail_excerpt else ''}"
f"{' (' + trigger + ')' if trigger else ''}"
```
This avoids claiming "blocks its caller, which blocks its caller" when there's no evidence of chain depth. The "appears to be" language preserves uncertainty.

Both branches append calibration footer:

```python
calibration = describe_calibration("cascading_timeout")
diag += f" How this threshold was set: {calibration}"
```

**Current fix:** Timeouts + circuit breakers + partial results.
**New fix:** No change — the fix advice applies to both branches.

### 4.3 `_pat_unbounded_cache_growth` (ADR-003) — Commit 2

**Current narrative:**
```python
f"Memory grows without bound because cached data is never evicted. "
```

**New narrative (with causal-uncertainty language per ADR-003):**

```python
# Extract the specific decorator name for the narrative
cache_decorator = ""
if f.source_decorators:
    for dec in f.source_decorators:
        if dec in _CACHE_DECORATORS:
            cache_decorator = dec
            break

diag = (
    f"{loc + ' — memory' if loc else 'Memory'} reached "
    f"{mem}MB during testing. The code uses a `{cache_decorator}` "
    f"cache decorator, and memory grew as the cache accumulated "
    f"entries. This correlation suggests the cache may lack an "
    f"eviction bound — but myCode cannot confirm this is the sole "
    f"cause of the growth."
    f"{'At load level ' + fields['load'] + ', ' if fields['load'] != 'high' else ''}"
    f"if the cache grows with every unique input, it will eventually "
    f"exhaust available memory."
    f"{' ' + detail_excerpt if detail_excerpt else ''}"
    f"{' (' + trigger + ')' if trigger else ''}"
)
```

Key changes from current:
- "grows without bound because cached data is never evicted" (strong causal claim) → "grew as the cache accumulated entries. This correlation suggests the cache may lack an eviction bound — but myCode cannot confirm this is the sole cause" (correlational, acknowledges uncertainty)
- Names the specific decorator
- States peak memory as a measured fact, not a prediction

Append calibration footer:

```python
calibration = describe_calibration("unbounded_cache_growth")
diag += f" How this threshold was set: {calibration}"
```

**Current fix:** lru_cache, TTLCache, Redis/Memcached.
**New fix:** No change.

### 4.4 `_pat_input_handling_failure` (ADR-004) — Commit 3

**Current narrative:**
```python
f"The application crashes when it receives input of an unexpected type. "
f"{errs + ' errors occurred'} because the code assumes a specific input type"
```

**New narrative — exception-type branched via helper function:**

```python
def _input_handling_narrative(f, fields, matched_marker) -> str:
    """Build exception-branched narrative for input handling findings."""
    loc = fields["loc"]
    errs = fields["errs"]
    trigger = fields["trigger"]
    detail_excerpt = fields["detail_excerpt"]

    # Exception-type-specific operational explanation
    _MARKER_EXPLANATIONS = {
        "TypeError": (
            "a value of the wrong type was passed to an operation — "
            "for example, a string where an integer was expected, or "
            "None where an object was required"
        ),
        "ValueError": (
            "a value had the right type but was outside the expected range "
            "or format — for example, a negative number where only "
            "positive was valid, or an unrecognised enum value"
        ),
        "AttributeError": (
            "the code tried to access a property or method on an object "
            "that doesn't have it — typically because the object was None "
            "or a different type than expected"
        ),
        "KeyError": (
            "the code tried to access a dictionary key that doesn't exist "
            "— the input structure was missing an expected field"
        ),
        "IndexError": (
            "the code tried to access a list or array position that "
            "doesn't exist — the input was shorter than expected"
        ),
        "ValidationError": (
            "input failed schema validation — the data structure or "
            "field types didn't match the expected Pydantic or "
            "validation model"
        ),
        "JSONDecodeError": (
            "the code received text that isn't valid JSON — this "
            "typically happens when an API returns an error page, "
            "empty response, or HTML instead of JSON"
        ),
    }

    explanation = _MARKER_EXPLANATIONS.get(matched_marker, "")

    diag = (
        f"{loc + ' raised' if loc else 'The application raised'} "
        f"`{matched_marker}` — "
    )
    if explanation:
        diag += f"{explanation}. "
    diag += (
        f"{errs + ' errors occurred' if errs else 'The error occurred'} "
        f"{'at load level ' + fields['load'] + ' ' if fields['load'] != 'high' else ''}"
        f"during stress testing"
        f"{' (' + trigger + ')' if trigger else ''}."
        f"{' ' + detail_excerpt if detail_excerpt else ''}"
    )
    # Calibration disclosure (Product Architecture Section 1.6)
    calibration = describe_calibration("input_handling_failure")
    diag += f" How this threshold was set: {calibration}"
    return diag
```

Each exception type gets an operational explanation that tells the non-engineer user what the exception means in terms they can understand. This replaces the generic "crashes when it receives input of an unexpected type" with specific, accurate descriptions.

**New fix — exception-type branched:**

```python
_MARKER_FIXES = {
    "TypeError": (
        "Add type checks (`isinstance(x, str)`) or use Pydantic "
        "models for structured input. Wrap type-sensitive operations "
        "in try/except TypeError."
    ),
    "ValueError": (
        "Add value range validation before processing. Use "
        "try/except ValueError for conversion operations like "
        "int(), float(), or datetime parsing."
    ),
    "AttributeError": (
        "Check for None before attribute access (`if obj is not None: "
        "obj.method()`). Use `getattr(obj, 'attr', default)` for "
        "optional attributes."
    ),
    "KeyError": (
        "Use `dict.get('key', default)` instead of `dict['key']`. "
        "For nested structures, validate key existence at each level "
        "or use a schema validator."
    ),
    "IndexError": (
        "Validate sequence length before indexing "
        "(`if len(items) > i: items[i]`). Use `next(iter(items), None)` "
        "for safe first-element access."
    ),
    "ValidationError": (
        "Check that input data matches the Pydantic model constraints. "
        "Add `try/except ValidationError` with a user-friendly error "
        "message explaining which fields are invalid."
    ),
    "JSONDecodeError": (
        "Wrap JSON parsing in try/except JSONDecodeError. Check "
        "`response.status_code` before parsing. Log the raw response "
        "body when parsing fails for debugging."
    ),
}
```

### 4.5 `_pat_flask_concurrency` (ADR-005) — Commit 3

**Current narrative:**
```python
f"Flask handles requests synchronously — each request blocks a worker thread. "
f"All threads are occupied and new requests queue."
```

**New narrative — full 4-question structure via helper function:**

```python
def _flask_concurrency_narrative(f, fields) -> str:
    """Build 4-question narrative for Flask concurrency findings.

    Structure follows the ADR-005 user-comprehension principle:
    1. What happens to your users
    2. At what point it matters
    3. What to do about it
    4. When it does not matter
    """
    loc = fields["loc"]
    resp = fields["resp"]
    errs = fields["errs"]
    trigger = fields["trigger"]
    detail_excerpt = fields["detail_excerpt"]
    call_chain = fields["call_chain"]
    load = fields["load"]

    sections = []

    # Opening — what happened (measured fact)
    opening = (
        f"Flask handles requests synchronously — each request blocks "
        f"a worker thread until it finishes."
    )
    if call_chain:
        opening += f" Call chain: {call_chain}."
    sections.append(opening)

    # 1. What happens to your users
    user_impact = (
        f"What happens to your users: "
        f"{loc + ' took' if loc else 'Response time reached'} "
        f"{resp if resp else 'degraded levels'} at {load} concurrent "
        f"requests"
        f"{', with ' + errs + ' errors' if errs else ''}. "
        f"When all worker threads are busy, new requests wait in a "
        f"queue — users see either slow responses or timeouts."
    )
    sections.append(user_impact)

    # 2. At what point it matters
    threshold_context = (
        f"At what point it matters: your application showed "
        f"degradation at {load} concurrent users"
        f"{' (' + trigger + ')' if trigger else ''}. "
        f"The default Flask development server uses a single thread. "
        f"A typical production deployment with gunicorn uses 2-4 "
        f"workers × 2 threads = 4-8 concurrent requests before "
        f"queuing begins — actual capacity depends on your "
        f"configuration."
    )
    sections.append(threshold_context)

    # 3. What to do about it — kept brief here, full detail in fix
    action = (
        f"What to do about it: deploy with a production server "
        f"(gunicorn, waitress) and add connection pooling for "
        f"database and HTTP calls. See the fix prompt below for "
        f"specific code changes."
    )
    sections.append(action)

    # 4. When it does not matter
    escape = (
        f"When this does not matter: if your application serves a "
        f"small number of sequential users (internal tool, personal "
        f"project) and {load} concurrent users is far above your "
        f"actual usage, this finding is informational."
    )
    sections.append(escape)

    # Calibration disclosure (Product Architecture Section 1.6)
    calibration = describe_calibration("flask_concurrency")
    sections.append(f"How this threshold was set: {calibration}")

    if detail_excerpt:
        sections.append(detail_excerpt)

    return " ".join(sections)
```

**Current fix:** Connection pooling, gunicorn, timeouts.
**New fix:** No change — the fix advice is correct. The narrative's "what to do about it" section provides the bridge.

### 4.6 `_pat_requests_concurrent` (ADR-006) — Commit 3

**Current narrative:**
```python
f"The `requests` library is synchronous — each call blocks its thread "
f"until the response arrives. At {fields['load']} concurrent requests, "
f"all threads are occupied waiting on I/O"
```

**New narrative — structured via helper function:**

```python
def _requests_concurrent_narrative(f, fields, io_marker) -> str:
    """Build structured narrative for requests concurrent findings.

    Includes: observable symptoms, capacity framing, remediation
    options with effort/gain, and escape hatch.
    """
    loc = fields["loc"]
    resp = fields["resp"]
    errs = fields["errs"]
    trigger = fields["trigger"]
    detail_excerpt = fields["detail_excerpt"]
    load = fields["load"]

    sections = []

    # Observable production symptoms
    symptoms = (
        f"{loc + ' — the' if loc else 'The'} `requests` library is "
        f"synchronous: each HTTP call blocks its thread until the "
        f"response arrives. At {load} concurrent operations, "
        f"{errs + ' I/O errors occurred' if errs else 'threads were blocked'}"
        f"{' (' + io_marker + ')' if io_marker else ''}"
        f"{' with response times reaching ' + resp if resp else ''}."
    )
    sections.append(symptoms)

    # Capacity framing
    capacity = (
        f"In production, this means your application can handle at "
        f"most {load} simultaneous outbound HTTP calls before new "
        f"requests queue or fail. If your application makes external "
        f"API calls on every user request, this limits your effective "
        f"concurrency to {load} users."
    )
    sections.append(capacity)

    # Remediation options with effort/gain tradeoff
    remediation = (
        f"Three options, from least to most effort: "
        f"(1) Add `timeout=5` to every `requests.get/post` call — "
        f"low effort, prevents indefinite blocking but doesn't "
        f"increase throughput. "
        f"(2) Use `requests.Session()` with "
        f"`concurrent.futures.ThreadPoolExecutor(max_workers=N)` — "
        f"moderate effort, adds bounded parallelism. "
        f"(3) Replace `requests` with `httpx.AsyncClient` and "
        f"`async/await` — highest effort, highest throughput gain."
    )
    sections.append(remediation)

    # Escape hatch
    escape = (
        f"When this does not matter: if your application makes "
        f"external calls infrequently (e.g., on startup, on admin "
        f"action) rather than on every user request, the synchronous "
        f"behaviour is acceptable and this finding is informational."
    )
    sections.append(escape)

    # Calibration disclosure (Product Architecture Section 1.6)
    calibration = describe_calibration("requests_concurrent")
    sections.append(f"How this threshold was set: {calibration}")

    if detail_excerpt:
        sections.append(detail_excerpt)

    return " ".join(sections)
```

**Current fix:** httpx.AsyncClient, requests.Session, ThreadPoolExecutor.
**New fix:** The fix string keeps the same technical content but the narrative's remediation section now provides the effort/gain context for the user. The fix string remains coding-agent-oriented:

```python
fix = (
    f"{loc + ': replace' if loc else 'Replace'} `requests.get/post` "
    f"with `httpx.AsyncClient` for async I/O, or use "
    f"`requests.Session()` for connection pooling with "
    f"`concurrent.futures.ThreadPoolExecutor"
    f"(max_workers={load})` for bounded parallelism. "
    f"Add `timeout=5` to every `requests` call as a minimum safety net."
    f"{' Other dependencies involved: ' + deps_str + '.' if deps_str else ''}"
)
```

---

## 5. ADR-004 Pattern Rename

### Function rename

```python
# Before:
@_register_pattern
def _pat_unvalidated_type_crash(f, framework, fields):

# After:
@_register_pattern
def _pat_input_handling_failure(f, framework, fields):
```

### Dispatch mapping

The `failure_pattern` classifier label `"unvalidated_type_crash"` in `classifiers.py` is NOT renamed (corpus continuity). The pattern function's gate still checks `f.failure_pattern == "unvalidated_type_crash"`. The function name is the only thing that changes — this is a narrative-layer rename only.

### No changes to classifiers.py

The `_PATTERN_RULES` entry at `classifiers.py:289` stays as:
```python
("input_handling_failure", ["type", "TypeError"], "unvalidated_type_crash"),
```

---

## 6. Current Output Surface Structure (Revision 2 addition)

### What each pattern function returns

Each pattern function (`_pat_flask_concurrency`, etc.) returns either:
- `(diagnosis: str, fix: str)` — a 2-tuple of plain strings
- `None` — if the pattern's gate rejects the finding

**`diagnosis`** is the human-readable narrative explaining what happened and why it matters. It is the "Why this matters" section in the PDF. It is written for non-engineer users but also consumed by coding agents via the prompt.

**`fix`** is the coding-agent-oriented instruction explaining what code changes to make. It is embedded inside the `generate_finding_prompt()` output under the label `"Fix:"`.

### Where each return value flows

```
(diagnosis, fix)
  │
  ├─→ diagnosis
  │   ├─ PDF: "Why this matters" section (user reads directly)
  │   ├─ JSON: "diagnosis" key (tools consume)
  │   ├─ Markdown: "**Why this matters:**" section
  │   ├─ generate_finding_prompt(): embedded as "Diagnosis: {diagnosis}"
  │   └─ fix_enrichment.py: context for LLM-generated fix suggestions
  │
  └─→ fix
      └─ generate_finding_prompt(): embedded as "Fix: {fix}"
         ├─ JSON: appears inside the "prompt" key value
         └─ PDF: appears inside the "What to do" code block
```

The `fix` string does NOT have its own top-level key in the JSON output. It is only accessible as part of the `"prompt"` string. This is intentional — the fix is addressed to the coding agent, not to the user or to downstream tools.

### How the ADR's narrative-vs-fix-prompt distinction maps

The ADR distinguishes:
1. **Narrative** (for users) → maps to `diagnosis` (first tuple element)
2. **Fix prompt** (for coding agents) → maps to `fix` (second tuple element)

This is a clean 1:1 mapping. No new fields or structural changes are needed. The richer multi-section narratives from ADR-005 and ADR-006 go into the `diagnosis` string; the exception-branched fix advice from ADR-004 goes into the `fix` string.

---

## 7. Test Impact Analysis

### Tests that will break and their fixes

#### Commit 1 breaks (blanket rules)

**7.1 `test_cascading_timeout_minimal` (test_documents.py:2899)**
**Why:** `_load_level` is `None` → Rule 3 blocks cascading_timeout (a concurrency pattern).
**Fix:** Add `f._load_level = 5` to the Finding construction.

**7.2 `test_requests_concurrent_minimal` (test_documents.py:3017)**
**Why:** `_load_level` is `None` → Rule 3 blocks requests_concurrent.
**Fix:** Add `f._load_level = 5` to the Finding construction.

#### Commit 2 breaks (patterns 1-3: data_volume, cascading_timeout, unbounded_cache_growth)

**7.3 `test_data_volume_minimal` (test_documents.py:3057)**
**Why:** `_peak_memory_mb` is 0.0 (< 47) and `_error_count` is 0 (< 11). Gate rejects.
**Fix:** Add `f._peak_memory_mb = 100.0`.

**7.4 `test_data_volume_scaling` (test_documents.py:2092)**
**Why:** Same as 7.3 — no measurements cross threshold.
**Fix:** Add `f._peak_memory_mb = 100.0`.

**7.5 `test_cascading_timeout_full` (test_documents.py:2876)**
**Why:** `_error_count` not set (default 0, < 3).
**Fix:** Add `f._error_count = 5`.

**7.6 `test_cascading_timeout_minimal` (test_documents.py:2899) — ADDITIONAL for Commit 2**
**Why:** After Commit 1 added `_load_level = 5`, Commit 2 adds error_count ≥ 3 gate. Default is 0.
Also: with empty call_chain, the single-point narrative fires instead of cascade narrative.
**Fix:** Add `f._error_count = 5`. Update assertion:
- `assert "cascading" in d.lower()` → `assert "timed out" in d.lower()` (single-point branch)
- `assert "timeout" in fix.lower()` — still passes.

**7.7 `test_unbounded_cache_growth_full` (test_documents.py:2763)**
**Why:** No `source_decorators` → `has_cache_decorator` is False.
**Fix:** Add `source_decorators=["lru_cache"]`. Assertions:
- `"utils.py" in d` — passes (loc unchanged)
- `"450MB" in d` — passes (memory in new narrative)
- `"lru_cache" in fix` — passes (fix unchanged)
No assertion changes needed — the _full test doesn't assert on "never evicted".

**7.8 `test_unbounded_cache_growth_minimal` (test_documents.py:2786)**
**Why:** No decorators, no peak memory. Gate requires both.
**Fix:** Add `f.source_decorators = ["cache"]` and `f._peak_memory_mb = 100.0`.
**Assertion changes:**
- `assert "never evicted" in d` → `assert "cache" in d.lower()` (new narrative mentions the cache decorator and correlation language, not "never evicted")
- `assert "eviction" in fix` — still passes (fix text unchanged)

#### Commit 3 breaks (patterns 4-6: input_handling_failure, flask_concurrency, requests_concurrent)

**7.9 `test_unvalidated_type_crash_full` (test_documents.py:2953)**
**Why:** `_error_count=12` < 39.
**Fix:** Set `f._error_count = 45`.
**Assertion changes:**
- `assert "12 errors" in d` → `assert "45 errors" in d`
- `assert "TypeError" in d` — still passes (the marker appears in the exception-branched narrative)
- `assert "Pydantic" in fix or "pydantic" in fix.lower()` — still passes (TypeError fix branch recommends Pydantic)

**7.10 `test_unvalidated_type_crash_minimal` (test_documents.py:2977)**
**Why:** `_error_count` is 0 (< 39), no details (no exception marker).
**Fix:** Set `f._error_count = 50`, `f.details = "TypeError: cannot convert None to int"`.
**Assertion changes:**
- `assert "unexpected type" in d` → `assert "TypeError" in d` (new narrative names the actual exception)
- `assert "validation" in fix.lower()` — still passes (TypeError fix branch mentions type checks)

**7.11 `test_requests_concurrent_full` (test_documents.py:2994)**
**Why:** `_error_count=4` < 91, no I/O marker in details.
**Fix:** Set `f._error_count = 95`, update details to include an I/O marker:
`f.details = "All 20 threads blocked waiting for responses. Timeout: read timed out"`
**Assertion changes:**
- `assert "client.py" in d` — passes (loc unchanged)
- `assert "6000ms" in d` — passes (new narrative includes response time)
- `assert "httpx" in fix` — passes (fix unchanged)

**7.12 `test_requests_concurrent_minimal` (test_documents.py:3017) — ADDITIONAL for Commit 3**
**Why:** After Commit 1 added `_load_level = 5`, Commit 3 adds error_count ≥ 91 and I/O marker gates.
**Fix:** Set `f._error_count = 95`, `f.details = "ConnectionError: connection refused"`.
**Assertion changes:**
- `assert "synchronous" in d` → `assert "requests" in d.lower()` (new narrative uses different wording but still names the library)
- `assert "requests" in d.lower()` — still passes.

**7.13 `test_flask_concurrency_full` (test_documents.py:2645)**
**Why:** `_load_level=25` ≥ 5, severity=critical → gates pass. But the 4-question narrative changes the template structure.
**Assessment:** Check each assertion:
- `assert "synchronously" in d` — passes (new narrative opens with "Flask handles requests synchronously")
- `assert "views.py" in d` — passes (loc field populates into "What happens to your users" section)
- `assert "4200ms" in d` — passes (response time interpolated in opening)
- `assert "gunicorn" in fix` — passes (fix unchanged)
**Result:** No changes needed.

**7.14 `test_flask_concurrency_minimal` (test_documents.py:2668)**
**Why:** `_load_level=25` ≥ 5 → passes. Check assertions:
- `assert "synchronously" in d` — passes
- `assert "25 concurrent" in d` — passes ("at 25 concurrent requests" appears in user impact section)
**Result:** No changes needed.

**7.15 `test_flask_concurrency_with_call_chain` (test_documents.py:502)**
**Why:** `_load_level=50` ≥ 5, severity=critical → passes.
- `assert "Call chain:" in diag` — passes (new narrative includes call chain in opening section)
- `assert "index()" in diag` — passes
**Result:** No changes needed.

### Summary table by commit

#### Commit 1 (blanket rules)
| # | Test | Fix |
|---|---|---|
| 7.1 | test_cascading_timeout_minimal | Add `f._load_level = 5` |
| 7.2 | test_requests_concurrent_minimal | Add `f._load_level = 5` |

#### Commit 2 (patterns 1-3)
| # | Test | Fix |
|---|---|---|
| 7.3 | test_data_volume_minimal | Add `f._peak_memory_mb = 100.0` |
| 7.4 | test_data_volume_scaling | Add `f._peak_memory_mb = 100.0` |
| 7.5 | test_cascading_timeout_full | Add `f._error_count = 5` |
| 7.6 | test_cascading_timeout_minimal | Add `f._error_count = 5`, update "cascading" → "timed out" assertion |
| 7.7 | test_unbounded_cache_growth_full | Add `source_decorators=["lru_cache"]` |
| 7.8 | test_unbounded_cache_growth_minimal | Add decorators + memory, update "never evicted" → "cache" assertion |

#### Commit 3 (patterns 4-6)
| # | Test | Fix |
|---|---|---|
| 7.9 | test_unvalidated_type_crash_full | Set `_error_count=45`, update "12 errors" → "45 errors" |
| 7.10 | test_unvalidated_type_crash_minimal | Add error_count + details, update "unexpected type" → "TypeError" |
| 7.11 | test_requests_concurrent_full | Set `_error_count=95`, add I/O marker to details |
| 7.12 | test_requests_concurrent_minimal | Add `_error_count=95` + I/O marker details, update "synchronous" assertion |
| 7.13-15 | flask_concurrency tests | No changes needed |

#### Calibration footer — cross-cutting test impact

The calibration footer (`"How this threshold was set: ..."`) is appended to every diagnosis string. Existing `in` assertions are unaffected (original text is still present). Each test that checks a diagnosis string gains one additional assertion:

```python
assert "How this threshold was set:" in d
```

This is added to all 15 tests listed above. No existing assertions change because of the footer — the footer only adds text after the existing content.

---

## 8. Verification Sequence Before Each Commit

### After Commit 1 (threshold provider + blanket rules)

```bash
pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_session.py --ignore=tests/test_pipeline.py -k "not (TestPipelineIntegration or TestCLIExitCode)" -q --tb=short
```
Expected: All fast tests pass. The 2 test fixtures (7.1, 7.2) ensure blanket rules don't break existing tests.

### After Commit 2 (patterns 1-3)

```bash
pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_session.py --ignore=tests/test_pipeline.py -k "not (TestPipelineIntegration or TestCLIExitCode)" -q --tb=short
```
Expected: All fast tests pass with the 6 test updates (7.3-7.8).

### After Commit 3 (patterns 4-6)

Full test suite:
```bash
pytest tests/ -q --tb=short
```
Expected: 2,846+ passing, 5 skipped. Only accepted failure: `test_empty_project_raises` (pre-existing).

### Self-test (after Commit 3)

Run myCode against itself. Compare against baseline at `~/Desktop/MyCode Bug Test Work/mycode-report_post_Phase2.json`.

### Expected-changes manifest

| Finding type | Pre-Phase-4 | Post-Phase-4 | Reason |
|---|---|---|---|
| severity="info" | Has diagnosis from pattern | Empty diagnosis (generic fallback) | Blanket Rule 1 |
| _finding_type="clean" | Has diagnosis from pattern | Empty diagnosis | Blanket Rule 2 |
| Flask at load_level < 5 | Flask concurrency diagnosis | Falls through | ADR-005 threshold |
| Flask at load_level ≥ 5 | Old one-sentence narrative | 4-question narrative | ADR-005 narrative |
| Data volume with low memory/errors | Data volume diagnosis | Falls through | ADR-001 threshold |
| Cascading timeout without errors | Timeout diagnosis | Falls through | ADR-002 threshold |
| Cascading timeout without call_chain | "blocks its caller, which blocks its caller" | "single-point timeout" | ADR-002 branching |

---

## 9. Commit Structure

**Three commits, landed sequentially on main. Each leaves the test suite passing.**

### Commit 1: Infrastructure — threshold provider + blanket rules
```
feat: add threshold provider and blanket rules for Phase 4 pattern gates

Add src/mycode/thresholds.py with static threshold config for six
patterns (ADR-001 through ADR-006). Add blanket rule enforcement in
_match_remediation(): no pattern fires on info severity, clean findings,
or low load_level for concurrency patterns.
```
**Files:** `src/mycode/thresholds.py` (new), `src/mycode/documents.py` (blanket rules in `_match_remediation`), `tests/test_documents.py` (2 fixture additions: 7.1, 7.2)

### Commit 2: Patterns 1-3 (data_volume, cascading_timeout, unbounded_cache_growth)
```
feat: implement measurement-based gates for data_volume, cascading_timeout, unbounded_cache_growth

ADR-001: peak_memory_mb >= 47 OR error_count >= 11 for data_volume.
ADR-002: error_count >= 3 + cascade vs single-point narrative branching.
ADR-003: has_cache_decorator AND peak_memory_mb >= 50 + causal-uncertainty language.
```
**Files:** `src/mycode/documents.py` (3 pattern functions), `tests/test_documents.py` (6 test updates: 7.3-7.8)

### Commit 3: Patterns 4-6 (input_handling_failure, flask_concurrency, requests_concurrent)
```
feat: implement measurement-based gates for input_handling_failure, flask_concurrency, requests_concurrent

ADR-004: error_count >= 39 + exception marker detection + exception-type
branching in narrative and fix. Rename _pat_unvalidated_type_crash →
_pat_input_handling_failure.
ADR-005: load_level >= 5 + 4-question user-comprehension narrative.
ADR-006: load_level >= 2 + error_count >= 91 + I/O marker + structured
narrative with production symptoms, capacity framing, and escape hatch.
```
**Files:** `src/mycode/documents.py` (3 pattern functions + 3 helper functions), `tests/test_documents.py` (4 test updates: 7.9-7.12)

---

## Scope Boundary

- **Files modified:** `src/mycode/documents.py`, `tests/test_documents.py`
- **File created:** `src/mycode/thresholds.py`
- **Files NOT modified:** `classifiers.py`, `report.py`, `engine.py`, `http_load_driver.py`, or any other source file
- **No changes to the Finding dataclass** — all measurement fields already exist
- **No changes to the execution engine** — thresholds are applied at the narrative layer only
- **No changes to generate_finding_prompt()** — it already consumes diagnosis and fix via `_build_diagnosis()` and `_build_fix()`
- **`describe_calibration()` consumed in Phase 4** — calibration disclosure appears as a footer line on each finding's diagnosis string, satisfying Product Architecture Section 1.6. Not deferred to Phase 8.
