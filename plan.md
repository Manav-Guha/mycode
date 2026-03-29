# Plan: E1-E4 Constraint Wiring + Intent-Driven Testing + Prediction Box

## Current State

**What exists:**
- `OperationalConstraints` has: `user_scale`, `usage_pattern`, `max_payload_mb`, `data_type`, `deployment_context`, `analysis_depth`, etc.
- Web conversation is sequential Q&A: Turn 1 (free text about project), Turn 2 (free text about usage), Turn 3+ (follow-up questions for missing fields: user_scale, data_type, usage_pattern, max_payload_mb, analysis_depth).
- Scenario generator uses `user_scale` for concurrent test levels via `_scale_levels()`, derives `data_sizes` from `max_payload_mb` or `user_scale * 10`, and applies `data_type` filtering/boost.
- Report's `_contextualise_findings` classifies severity by comparing load_level against user_scale (≤1x = critical, ≤3x = warning, >3x = info).
- Degradation curves have no user-intent markers.
- No prediction system exists.
- `corpus_patterns_ranked.json` exists with confirmed_count, severity_distribution, affected_dependencies per pattern.

**What's missing:**
- No `current_users` vs `max_users` distinction — only `user_scale`.
- No `per_user_data` or `max_total_data` — only `max_payload_mb`.
- Scenario generator doesn't use 10% increment logic from current→max.
- Usage pattern doesn't shape load distribution (bursts vs steady vs growing).
- Report doesn't compute headroom or frame findings as capacity assessments.
- No prediction box or corpus lookup by dependency stack.
- Web UI is sequential Q&A, not grouped form sections.

---

## PART 1: Web Frontend — Grouped Form Sections

### 1.1 Replace conversation HTML (index.html)

Replace the `converse-section` div with four form sections. Each section has a heading and all its inputs visible at once.

```
Section 1 — Your Project
  - "What does your project do?" → textarea (id: q-description)
  - "What type of data does it handle?" → pill group (text, images, numerical, documents, mixed)

Section 2 — Your Users
  - "How many users do you expect currently?" → number input (id: q-current-users)
  - "Maximum users over the next year?" → number input (id: q-max-users)
  - "How will they use it?" → pill group (steady, bursts, on-demand, growing)

Section 3 — Your Data
  - "Typical data per user?" → dropdown (small: <1MB, medium: 1-50MB, large: 50MB+, or free text)
  - "Maximum total data for the application?" → dropdown with examples + free text option

Section 4 — Analysis
  - "How thorough?" → pill group (quick scan ~2 min, standard ~5 min, deep ~10 min)
```

Bottom of form: single "Run Stress Tests" button that submits all answers at once.

### 1.2 Replace conversation JS (app.js)

Remove the `beginConversation()` / `requestConverseTurn()` / `sendReply()` sequential flow for the web path.

New flow:
1. After preflight completes → show the grouped form (not conversation).
2. User fills in form fields → clicks "Run Stress Tests".
3. `submitForm()` collects all field values into a single `FormData` POST to a **new endpoint** `/api/submit-intent`.
4. `/api/submit-intent` parses all answers into `OperationalConstraints`, stores on job, marks `conversation_done`, then triggers analysis.
5. Right panel shows progress → results.

The sequential conversation UI is removed from the web path entirely. The left panel becomes a structured intake form that stays visible (sticky) during analysis.

### 1.3 Pill component in CSS/HTML

Simple pill buttons using radio inputs styled as pills. Each group is a `<div class="pill-group">` with `<label class="pill"><input type="radio" name="..." value="..."><span>Label</span></label>`.

Selected state: accent background. Unselected: dark border.

### 1.4 New endpoint: `/api/submit-intent` (routes.py)

```python
def handle_submit_intent(job_id: str, answers: dict) -> dict:
    """Parse grouped form answers into OperationalConstraints, start analysis."""
```

Accepts:
- `job_id`
- `description` (free text)
- `data_type` (pill value)
- `current_users` (number)
- `max_users` (number)
- `usage_pattern` (pill value)
- `per_user_data` (dropdown value or free text)
- `max_total_data` (dropdown value or free text)
- `analysis_depth` (pill value)

Parses into `OperationalConstraints` using existing parsers + new parsers for new fields.

Does NOT remove `/api/converse` — the CLI still uses it via the sequential path.

### 1.5 No change to CLI

The CLI's sequential Q&A in `interface.py` continues to work. The CLI path will ask the new questions (current_users, max_users, per_user_data, max_total_data) as follow-ups in `_FOLLOWUP_FIELDS`. The existing `user_scale` follow-up is split into two: current_users then max_users.

---

## PART 2: OperationalConstraints Changes

### 2.1 New fields on `OperationalConstraints`

```python
@dataclass
class OperationalConstraints:
    # Existing (some renamed/repurposed):
    user_scale: Optional[int] = None          # KEPT for backward compat, computed as max_users
    usage_pattern: Optional[str] = None       # values: "steady", "burst", "on_demand", "growing"
    max_payload_mb: Optional[float] = None    # KEPT, now derived from per_user_data if not set directly
    data_type: Optional[str] = None
    analysis_depth: Optional[str] = None
    # ... other existing fields unchanged ...

    # NEW fields:
    current_users: Optional[int] = None       # user's current scale
    max_users: Optional[int] = None           # user's growth target
    per_user_data: Optional[str] = None       # e.g. "small", "medium", "large" or "50 rows"
    max_total_data: Optional[str] = None      # e.g. "10GB", "1M rows"
    project_description: Optional[str] = None # free text from Section 1
```

**Backward compatibility:** `user_scale` is computed as `max_users` when `max_users` is set. Existing code that reads `user_scale` continues to work. `current_users` is the new field for baseline.

### 2.2 New parsers in constraints.py

- `parse_per_user_data(text) -> Optional[str]`: Handles "small", "medium", "large", free text. Stores the raw category or description.
- `parse_max_total_data(text) -> Optional[str]`: Handles "1GB", "1M rows", descriptive text.
- `parse_current_users(text) -> Optional[int]`: Reuses `parse_user_scale` logic.
- `per_user_data_to_items(per_user_data: str) -> int`: Converts category to item count for scenario generation (small=50, medium=500, large=5000, or parse numeric).
- `max_total_data_to_items(max_total_data: str) -> int`: Converts to item count.

### 2.3 Usage pattern value changes

Current values: `sustained`, `burst`, `periodic`, `growing`.
New values (web pills): `steady`, `bursts`, `on_demand`, `growing`.

Map: `sustained` → `steady`, `periodic` → `on_demand`, keep `burst`/`bursts` and `growing`.

Add aliases in `parse_usage_pattern` so both old and new values work. CLI questions updated to match new labels.

### 2.4 Update _FOLLOWUP_FIELDS for CLI

```python
_FOLLOWUP_FIELDS = (
    "current_users", "max_users", "data_type", "usage_pattern",
    "per_user_data", "max_total_data", "analysis_depth",
)
```

Remove `user_scale` and `max_payload_mb` from follow-ups. `user_scale` is derived from `max_users`. `max_payload_mb` is derived from `per_user_data`/`max_total_data` if needed.

---

## PART 3: Scenario Generator Parameterisation

### 3.1 User scaling — 10% increment logic

Replace `_scale_levels(user_scale)` with new `_user_scale_levels(current_users, max_users)`:

```python
def _user_scale_levels(current: int, maximum: int) -> list[int]:
    """Generate test levels from current to max in 10% increments of max, plus 10-20% buffer."""
    step = max(1, maximum // 10)
    levels = list(range(current, maximum + 1, step))
    if levels[-1] != maximum:
        levels.append(maximum)
    # Add 10-20% buffer (use 15% = midpoint)
    buffer = maximum + max(1, int(maximum * 0.15))
    levels.append(buffer)
    return sorted(set(levels))
```

Example: current=50, max=200, step=20 → [50, 70, 90, 110, 130, 150, 170, 190, 200, 230]

### 3.2 Data scaling — dual axis

Replace `_data_scale_levels(base_items)` with intent-driven logic:

```python
def _data_scale_levels_intent(per_user_items: int, max_total_items: int) -> list[int]:
    """Generate data test levels from per_user up to max_total in 10% increments."""
    step = max(1, max_total_items // 10)
    levels = list(range(per_user_items, max_total_items + 1, step))
    if levels[-1] != max_total_items:
        levels.append(max_total_items)
    buffer = max_total_items + max(1, int(max_total_items * 0.15))
    levels.append(buffer)
    return sorted(set(levels))
```

Existing `_data_scale_levels` kept as fallback when no intent is provided.

### 3.3 Usage pattern → load shape

In `_apply_constraints`, when generating concurrent_execution scenarios:

```python
if usage_pattern == "burst" or usage_pattern == "bursts":
    # Spike pattern: ramp to max instantly, hold, drop, repeat
    params["load_shape"] = "spike"
elif usage_pattern == "steady":
    # Even distribution across all levels
    params["load_shape"] = "even"
elif usage_pattern == "growing":
    # Progressive ramp: start low, increase over duration
    params["load_shape"] = "ramp"
elif usage_pattern == "on_demand":
    # Intermittent: random spikes with gaps
    params["load_shape"] = "intermittent"
```

The engine reads `load_shape` from test_config and adjusts how it schedules concurrent operations within each tier. **This is a new parameter the engine must handle.**

Engine changes:
- `load_shape=even` (default): spread iterations evenly across the test window.
- `load_shape=spike`: launch all iterations simultaneously at the start of each tier.
- `load_shape=ramp`: linearly increase iteration rate from 0 to max over the tier duration.
- `load_shape=intermittent`: 3 bursts at 33%, 66%, 100% of tier duration, each launching N/3 iterations.

### 3.4 Non-interactive fallback

When `--non-interactive` is used (no constraints provided), the scenario generator:

1. Looks up the project's dependency names in `corpus_patterns_ranked.json`.
2. Finds the median `confirmed_count`-weighted failure thresholds for similar stacks.
3. If corpus lookup fails, uses hardcoded defaults: current_users=10, max_users=100, per_user_data="medium", max_total_data="10000 items".
4. Sets `constraints.assumptions_used = True` (new bool field).
5. Stores the assumed values in `constraints.assumed_values` (new dict field).

The report checks `assumptions_used` and prepends the assumptions disclaimer.

### 3.5 Store intent on scenario for report access

Each scenario gets `test_config["user_intent"]` dict:
```python
{
    "current_users": 50,
    "max_users": 200,
    "per_user_data": "medium",
    "max_total_data": "10000 items",
}
```

This lets the report generator frame each finding relative to the user's stated scale without re-reading constraints.

---

## PART 4: Report Framing Relative to User Intent

### 4.1 Rewrite `_contextualise_findings`

Current: compares load_level to user_scale as a ratio.

New: computes headroom and frames findings as capacity assessments.

```python
def _contextualise_findings(self, report, constraints, profile_matches):
    current = constraints.current_users
    maximum = constraints.max_users or constraints.user_scale

    for finding in report.findings:
        load_level, is_concurrency = _extract_load_level(finding)
        if load_level is None or not is_concurrency or maximum is None:
            continue

        if current and load_level > current:
            headroom_pct = ((load_level - current) / current) * 100
            finding.description = (
                f"This issue occurs at {load_level:,} concurrent users. "
                f"At your current scale of {current:,} users, this does not affect you. "
                f"You have {headroom_pct:.0f}% headroom above your current usage. "
                f"Your code is reliable up to {load_level:,} users."
            )
            if load_level < maximum:
                finding.description += (
                    f" At your maximum target of {maximum:,} users, "
                    f"this becomes relevant."
                )
                finding.severity = "critical"  # breaks within target range
            else:
                finding.severity = "warning" if load_level < maximum * 1.5 else "info"
        elif current and load_level <= current:
            finding.description = (
                f"This issue occurs at {load_level:,} concurrent users — "
                f"below your current {current:,} users. "
                f"This is affecting your application now."
            )
            finding.severity = "critical"
```

### 4.2 Report language shift

In `report.py`, rename section headers and terminology:

| Current | New |
|---------|-----|
| "Fix Before Launch" | "Priority Improvements" |
| "Findings" | "Recommendations" |
| "Findings at Default Test Range" | "Recommendations at Default Test Range" |
| severity "critical" display text | "Priority improvement" |
| severity "warning" display text | "Improvement opportunity" |
| "Your code breaks at X" | "Your code is reliable up to X. To extend to Y, here is what to improve." |
| "Degradation curves" | "Scaling roadmap" |

**Where:** `DiagnosticReport.as_text()` section headers (report.py ~270-300), `documents.py` PDF section headers, `app.js` report rendering.

**Severity internal values remain `critical`/`warning`/`info`.** Only display labels change.

### 4.3 Headroom reporting in descriptions

Every contextualised finding includes:
- Percentage headroom: "40% headroom above current usage"
- Plain language: "reliable up to 70 users against your stated 50"
- Relationship to max target: "at your maximum target of 200 users, this becomes relevant"

### 4.4 Degradation curve intent markers

In `DiagnosticReport`'s JSON output (used by web frontend) and `documents.py` PDF:

Add to each `DegradationPoint`:
```python
user_baseline: Optional[int] = None   # current_users
user_ceiling: Optional[int] = None    # max_users
```

These are populated from constraints when the report is generated.

**Web frontend (app.js):** When rendering SVG degradation charts, draw two vertical dashed lines:
- Green dashed line at `user_baseline` x-position, labeled "Current"
- Orange dashed line at `user_ceiling` x-position, labeled "Target"

**documents.py PDF:** Add annotation text below charts: "Green line = your current scale (50 users). Orange line = your target (200 users)."

### 4.5 Update documents.py section headers

Match the language changes from 4.2. Limited to:
- "Findings" → "Recommendations"
- "Fix Before Launch" → "Priority Improvements"
- Degradation chart section title: "Performance Under Load" → "Scaling Roadmap"

No other documents.py changes.

---

## PART 5: Prediction Box

### 5.1 Corpus lookup module (new file: `src/mycode/prediction.py`)

```python
def predict_issues(
    dependency_names: list[str],
    corpus_path: str = "corpus_extraction/corpus_patterns_ranked.json",
    constraints: Optional[OperationalConstraints] = None,
) -> PredictionResult:
    """Look up corpus patterns for projects with similar dependency stacks."""
```

Logic:
1. Load `corpus_patterns_ranked.json`.
2. For each pattern, check if `affected_dependencies` overlaps with the user's deps.
3. Score by overlap count × `confirmed_count`.
4. Return top 5 patterns with:
   - title
   - probability (confirmed_count for matching deps / total projects with those deps)
   - severity distribution
   - matching deps

```python
@dataclass
class PredictionResult:
    predictions: list[PredictionItem]
    total_similar_projects: int
    matching_deps: list[str]

@dataclass
class PredictionItem:
    title: str
    probability_pct: float
    severity: str  # most common severity
    confirmed_count: int
    relevant_to_scale: str  # framed relative to user's stated scale, or generic
```

### 5.2 Prediction endpoint: `/api/predict` (routes.py)

Called by frontend after preflight completes and form is submitted, but before analysis finishes.

```python
def handle_predict(job_id: str) -> dict:
    """Return corpus-based predictions for the job's dependency stack."""
```

Uses `job.ingestion.dependencies` + `job.constraints` to call `predict_issues()`.

Returns JSON:
```json
{
    "total_similar_projects": 47,
    "matching_deps": ["flask", "sqlalchemy", "pandas"],
    "predictions": [
        {
            "title": "Memory growth under repeated requests",
            "probability_pct": 68,
            "severity": "warning",
            "scale_note": "At your stated 200 users, this is the most likely issue"
        }
    ]
}
```

### 5.3 Prediction box in web frontend (app.js + index.html)

In `index.html`, add a `prediction-section` div at the top of `col-right`, above `progress-section`:

```html
<div class="section hidden" id="prediction-section">
    <div class="section-title">Predictive Analysis</div>
    <div id="prediction-content"></div>
</div>
```

In `app.js`:
1. After form submission triggers analysis, immediately call `/api/predict`.
2. Render prediction box with:
   - "Based on [N] projects with similar technology stack ([dep list]):"
   - List of top 3-5 predictions with probability bars
   - Scale-relative framing if constraints available
3. Prediction box stays pinned at top of right column.
4. After tests complete, annotate each prediction:
   - If a matching finding exists → "Confirmed by testing" (check mark)
   - If not → "Not observed in your project — your code handles this better than X% of similar projects"

Matching logic: compare prediction title keywords against finding titles/categories.

### 5.4 Non-interactive prediction

When `--non-interactive`, prediction still runs (dependency-only, no user intent framing). Included in report text:
```
"Based on 47 similar projects, the most common issues are:
1. Memory growth under repeated requests (68% of similar projects)
2. ..."
```

### 5.5 Styling (style.css)

Prediction box: bordered container with subtle background, prediction items as rows with probability percentage badge on the right. Confirmed/not-confirmed annotations in green/grey after tests complete.

---

## PART 6: Implementation Order

### Phase A: Data model + parsers (no UI changes, no behavior changes)
1. Add new fields to `OperationalConstraints` (`current_users`, `max_users`, `per_user_data`, `max_total_data`, `project_description`, `assumptions_used`, `assumed_values`)
2. Add new parsers in `constraints.py` (`parse_per_user_data`, `parse_max_total_data`)
3. Update `_FOLLOWUP_FIELDS` and `_FOLLOWUP_QUESTIONS` in `interface.py` for CLI
4. Backward compat: `user_scale` property returns `max_users` when set
5. Write tests for new parsers and backward compat

### Phase B: Scenario generator parameterisation
1. Add `_user_scale_levels(current, maximum)` function
2. Add `_data_scale_levels_intent(per_user, max_total)` function
3. Update `_apply_constraints` to use new fields when available, fall back to existing logic
4. Add `load_shape` parameter threading through to engine
5. Engine: implement load_shape variations (spike, even, ramp, intermittent)
6. Non-interactive fallback with corpus-derived defaults
7. Write tests for new scaling functions and constraint application

### Phase C: Report framing
1. Rewrite `_contextualise_findings` for headroom-based framing
2. Add `user_baseline`/`user_ceiling` to `DegradationPoint`
3. Populate from constraints during report generation
4. Update section headers / display labels (report.py, documents.py, app.js)
5. Non-interactive assumptions disclaimer
6. Write tests for finding framing and headroom calculations

### Phase D: Prediction box
1. Create `src/mycode/prediction.py` with corpus lookup
2. Add `/api/predict` endpoint
3. Implement prediction matching + probability calculation
4. Write tests for prediction module

### Phase E: Web frontend
1. Replace conversation HTML with grouped form sections
2. Replace conversation JS with form submission flow
3. Add `/api/submit-intent` endpoint
4. Add prediction box rendering
5. Add intent markers on degradation charts (SVG vertical lines)
6. Add prediction confirmation annotations after test completion
7. Styling for pills, form sections, prediction box

### Phase F: Integration + acceptance
1. Run full test suite (2,290 with standard filters)
2. Verify CLI sequential flow still works
3. End-to-end web test: form → analysis → report with intent framing
4. Non-interactive test with assumptions disclaimer
5. Prediction box test with corpus data

---

## Files Modified

| File | Changes |
|------|---------|
| `src/mycode/constraints.py` | New fields, new parsers, usage_pattern aliases |
| `src/mycode/interface.py` | Updated followup fields/questions for CLI, new parsers wired |
| `src/mycode/scenario.py` | New scaling functions, load_shape, non-interactive fallback |
| `src/mycode/engine.py` | Load shape handling in concurrent execution |
| `src/mycode/report.py` | Rewritten contextualise_findings, headroom, DegradationPoint fields, section headers |
| `src/mycode/documents.py` | Section header language changes only |
| `src/mycode/prediction.py` | **NEW** — corpus lookup + prediction logic |
| `src/mycode/web/routes.py` | New `/api/submit-intent` and `/api/predict` endpoints |
| `web/index.html` | Grouped form sections replacing conversation |
| `web/app.js` | Form submission, prediction box, intent markers on charts |
| `web/style.css` | Pills, form sections, prediction box styling |
| `tests/` | New tests for parsers, scaling, prediction, report framing |

## Files NOT Modified

| File | Reason |
|------|--------|
| `corpus_extraction/*` | Per task spec — no extraction pipeline changes |
| `scripts/batch_mine.py` | Per task spec — no mining script changes |
| `corpus_patterns_ranked.json` | Read-only — no format changes |
| `src/mycode/session.py` | No session changes needed |
| `src/mycode/ingester.py` | No ingester changes needed |
| `src/mycode/http_load_driver.py` | No HTTP testing changes |

---

## Risk Areas

1. **Backward compatibility:** `user_scale` is used in ~15 places across scenario.py and report.py. Making it a derived property from `max_users` requires careful testing.
2. **Load shape in engine:** Adding spike/ramp/intermittent patterns to the execution engine is the most complex new behavior. Each pattern must produce valid, measurable results.
3. **Corpus data quality:** `corpus_patterns_ranked.json` may not have enough data for meaningful probability percentages for all dependency combinations. Need graceful fallback when overlap is thin (<5 similar projects).
4. **Form → sequential migration:** Removing sequential conversation from web while keeping it in CLI means two parallel input paths feeding the same `OperationalConstraints`. Must ensure both produce identical constraint objects for the same inputs.
5. **Prediction matching:** Matching predictions to test results requires fuzzy title/category comparison. Could produce false positives ("confirmed") if matching is too loose.
