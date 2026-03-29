# Plan: Multi-Language Project Support

## Problem

myCode picks one language per project. Multi-language projects (FastAPI + React, Django + Vue) get classified as one language and the other half is missed. The fastapi/full-stack-fastapi-template was classified as JavaScript due to package.json in the frontend directory — the Python backend was never analysed.

---

## Current Flow

```
project_path
  │
  ├→ detect_language(path) → "python" | "javascript"    [pipeline.py:210]
  │   Checks indicator files (requirements.txt vs package.json),
  │   counts .py vs .js/.ts files, Python wins ties.
  │   Returns EXACTLY ONE language.
  │
  ├→ SessionManager.setup()                              [session.py:179]
  │   Creates Python venv (always).
  │   Installs pip deps (_install_dependencies).
  │   Installs npm deps (_install_js_dependencies) IF package.json exists.
  │   Session already handles both runtimes.
  │
  ├→ _run_ingestion(session, language)                   [pipeline.py:546]
  │   if language == "python":
  │       ProjectIngester(installed_packages=venv_packages)
  │   else:
  │       JsProjectIngester(installed_packages=None)
  │   Returns ONE IngestionResult with deps from ONE language only.
  │
  ├→ library.match_dependencies(language, dep_dicts)     [pipeline.py:411]
  │   Matches against Python OR JavaScript profiles, not both.
  │
  ├→ generator.generate(ingestion, matches, language=language)  [scenario.py:597]
  │   _get_categories(language) → PYTHON_CATEGORIES or JAVASCRIPT_CATEGORIES
  │   Generates scenarios for ONE language's categories only.
  │
  ├→ ExecutionEngine(session, ingestion, language=language)     [engine.py]
  │   is_js = category in _JS_CATEGORIES or self.language == "javascript"
  │   Routes to "python" or "node" runner per scenario.
  │   Already handles mixed runners — if a JS-specific category appears
  │   in a Python project, it uses node. This is correct.
  │
  └→ ReportGenerator.generate(execution, ingestion, matches)    [report.py:1014]
      No language parameter. Infers from single-language execution.
```

**Key observation:** SessionManager and ExecutionEngine already handle both runtimes. The bottleneck is steps 1 (detection), 3 (ingestion), 4 (matching), and 6 (scenario generation) — they are mutually exclusive.

---

## Proposed Flow

```
project_path
  │
  ├→ detect_languages(path) → {"python", "javascript"} | {"python"} | {"javascript"}
  │   NEW function. Returns a SET of detected languages.
  │   Falls back to detect_language() for single-language.
  │
  ├→ SessionManager.setup()   [NO CHANGE]
  │   Already installs both pip and npm deps.
  │
  ├→ FOR EACH detected language:                         [NEW in pipeline.py]
  │   ├→ _run_ingestion(session, lang) → IngestionResult
  │   └→ library.match_dependencies(lang, deps)
  │
  ├→ _merge_ingestion_results(py_result, js_result) → IngestionResult  [NEW]
  │   Combined deps (tagged), combined files, combined LOC.
  │   Primary language determined by heuristic.
  │
  ├→ _merge_profile_matches(py_matches, js_matches) → list[ProfileMatch]
  │
  ├→ generator.generate(merged_ingestion, merged_matches, language=primary)
  │   CHANGE: accept languages: set[str] parameter alongside language.
  │   When languages has both, use ALL_CATEGORIES (union).
  │   Tag each scenario with its execution language.
  │
  ├→ ExecutionEngine(session, ingestion, language=primary)
  │   MINIMAL CHANGE: engine already routes by category.
  │   Scenarios tagged with execution language override self.language.
  │
  └→ ReportGenerator.generate(execution, ingestion, matches)
      MINIMAL CHANGE: findings already carry category and dep info.
      Summary generation uses primary language for server context.
```

---

## Detailed Design

### 1. Language Detection

**New function: `detect_languages()` in pipeline.py**

```python
def detect_languages(project_path: Path) -> set[str]:
    """Detect all languages present in a project.

    Returns a set: {"python"}, {"javascript"}, or {"python", "javascript"}.
    """
```

Logic:
- Check for Python indicators: requirements.txt, pyproject.toml, setup.py, Pipfile, *.py source files
- Check for JavaScript indicators: package.json with actual source files (not just build tooling)
- **Critical filter:** Only detect JavaScript if there are actual .js/.ts/.jsx/.tsx source files (not just node_modules or a lone package.json from a build dependency). Scan for ≥3 JS source files OR a package.json with a "main" or "scripts" field indicating a real JS project.
- Return set of detected languages

**Backward compatibility:** `detect_language()` (singular) remains unchanged. It calls `detect_languages()` and picks one via the existing heuristic (Python wins ties). All single-language paths continue to use it.

**Where called:** `handle_preflight()` in routes.py and `run_pipeline()` in pipeline.py both switch to `detect_languages()`.

### 2. IngestionResult Changes

**No new dataclass.** The existing `IngestionResult` is language-agnostic by design — it has `dependencies`, `file_analyses`, `coupling_points`, etc. without any language tag.

**Add one field:**

```python
@dataclass
class IngestionResult:
    # ... existing fields ...
    language: str = ""  # NEW: "python", "javascript", or "multi" for merged
    secondary_languages: list[str] = field(default_factory=list)  # NEW
```

**Merging strategy:** Create a new function `merge_ingestion_results()`:

```python
def merge_ingestion_results(
    primary: IngestionResult,
    secondary: IngestionResult,
    primary_language: str,
) -> IngestionResult:
    """Merge two single-language IngestionResults into one."""
```

- `dependencies`: concatenate both lists. DependencyInfo already has `name` — no collision risk since Python and JS deps have different names (flask vs react). Tag each with a `_language` marker.
- `file_analyses`: concatenate. FileAnalysis has `file_path` which naturally distinguishes .py from .js files.
- `coupling_points`: concatenate. Coupling points are intra-language — no cross-language coupling analysis.
- `function_flows`: concatenate.
- `files_analyzed`: sum both.
- `total_lines`: sum both.
- `files_failed`: sum both.
- `warnings`: concatenate.
- `parse_errors`: concatenate.
- `language`: set to primary_language.
- `secondary_languages`: set to [secondary_language].

**Where called:** After running both ingesters in pipeline.py / routes.py.

### 3. Primary Language Heuristic

```python
def _determine_primary_language(
    py_result: Optional[IngestionResult],
    js_result: Optional[IngestionResult],
    py_deps: list[str],
    js_deps: list[str],
) -> str:
    """Determine which language is the primary stress target."""
```

Rules (in priority order):
1. If Python has a server framework (FastAPI, Flask, Django, Streamlit, Gradio) → Python is primary
2. If JavaScript has a server framework (Express, Next.js) and Python does NOT → JavaScript is primary
3. If neither has a server framework → language with more source lines is primary
4. Tie → Python (matches current behavior)

The primary language determines:
- HTTP server testing target
- Report ordering (primary findings first)
- `language` field in the merged IngestionResult

### 4. Scenario Generator Changes

**`generate()` signature change:**

```python
def generate(
    self,
    ingestion_result: IngestionResult,
    profile_matches: list[ProfileMatch],
    operational_intent: str,
    language: str = "python",
    languages: Optional[set[str]] = None,  # NEW
    constraints: Optional[OperationalConstraints] = None,
) -> ScenarioGeneratorResult:
```

**Behavior:**
- If `languages` is None or has one element → existing behavior (backward compatible).
- If `languages` has both "python" and "javascript" → use `ALL_CATEGORIES` (union of PYTHON_CATEGORIES and JAVASCRIPT_CATEGORIES).
- Each generated scenario is tagged with an `execution_language` field:
  - Scenarios from Python profiles → `execution_language = "python"`
  - Scenarios from JavaScript profiles → `execution_language = "javascript"`
  - Coupling scenarios → inherit from the coupling point's file type (.py → python, .js → javascript)
  - Shared-category scenarios for profiled deps → tagged by the dep's language

**New field on StressTestScenario:**

```python
@dataclass
class StressTestScenario:
    # ... existing fields ...
    execution_language: str = ""  # NEW: "python" or "javascript"
```

This field is advisory — the engine already routes by category. But it makes the intent explicit and allows the engine to override its default `self.language` check.

### 5. Execution Engine Changes

**Minimal.** The engine already handles mixed runners:

```python
is_js = (
    scenario.category in _JS_CATEGORIES
    or self.language == "javascript"
)
```

**One change:** Also check `scenario.execution_language`:

```python
is_js = (
    scenario.category in _JS_CATEGORIES
    or scenario.execution_language == "javascript"
    or (not scenario.execution_language and self.language == "javascript")
)
```

This ensures Python scenarios from the JS half of a multi-language project still use the correct runner, and vice versa.

### 6. Report Generator Changes

**No signature change.** The report doesn't need to know about languages directly — it operates on findings, which are produced by the execution engine from scenarios that are already correctly routed.

**One enhancement:** When generating the project description and plain summary, detect multi-language projects:

```python
if ingestion.secondary_languages:
    # "Your FastAPI + React application..."
    # instead of "Your Python web application..."
```

This is a string formatting change in `_generate_project_description()` and `_generate_plain_summary()`.

### 7. Prediction Module Changes

**No model retraining needed.** The XGBoost model uses binary dependency features. If a project has both `flask` and `react`, both features are 1 — the model handles this correctly because its training data includes projects that happen to have both (even if they were classified as one language).

**One change in `predict_issues()`:** The `dependency_names` list will now include both Python and JS deps. The existing code already passes all deps to the model — no change needed.

**Architectural inference:** `infer_architectural_type()` already works on description keywords. For multi-language projects, the dependency-based classifier in `classifiers.py` will see both framework types and resolve correctly (e.g., FastAPI + React → the classifier sees both and picks based on scoring).

### 8. Library Matching Changes

Currently `library.match_dependencies(language, dep_dicts)` filters profiles by language. For multi-language projects:

```python
# Run matching for each language, merge results
py_matches = library.match_dependencies("python", py_dep_dicts)
js_matches = library.match_dependencies("javascript", js_dep_dicts)
merged_matches = py_matches + js_matches
```

This happens before the merge, using each language's own dep list.

### 9. HTTP Load Testing

Currently `run_http_testing_phase(language=language)` determines which server to start.

For multi-language projects:
- Use the primary language for HTTP testing
- If the primary side is Python (FastAPI/Flask), start the Python server
- If the primary side is JS (Express/Next.js), start the Node.js server
- Do NOT start two servers — one is enough for stress testing

This requires no code change beyond passing `language=primary_language` (which is already what happens).

### 10. Web Routes Changes

**`handle_preflight()`:**

```python
# Current:
language = detect_language(project_path)

# New:
languages = detect_languages(project_path)
primary_language = _pick_primary(languages)  # for backward compat
job.language = primary_language
job.detected_languages = languages  # NEW field on Job
```

Then run both ingesters if both languages detected:

```python
if "python" in languages:
    py_ingestion = _run_python_ingestion(session)
if "javascript" in languages:
    js_ingestion = _run_js_ingestion(session)

if py_ingestion and js_ingestion:
    ingestion = merge_ingestion_results(py_ingestion, js_ingestion, primary_language)
elif py_ingestion:
    ingestion = py_ingestion
else:
    ingestion = js_ingestion
```

**`run_analysis()` in worker.py:**

```python
# Pass languages set to scenario generator
languages = getattr(job, 'detected_languages', {job.language})
result = generator.generate(
    ...,
    language=job.language,
    languages=languages,
    ...
)
```

---

## What Does NOT Change

| Component | Why |
|-----------|-----|
| PythonProjectIngester internals | Works correctly for Python projects |
| JsProjectIngester internals | Works correctly for JS projects |
| SessionManager | Already installs both pip and npm deps |
| Profile JSON files | Language-specific, no change needed |
| Corpus data / extraction | Read-only |
| XGBoost model | Handles combined dep features naturally |
| Web frontend (app.js, index.html) | Receives report data as before |
| PDF generation (documents.py) | Receives DiagnosticReport as before |

---

## Regression Risk Assessment

### High Risk
1. **IngestionResult merge logic** — If the merge produces an invalid state (e.g., duplicate deps, mismatched coupling points), everything downstream breaks. Mitigate: extensive unit tests for merge function.
2. **Scenario generator category expansion** — Using ALL_CATEGORIES for multi-language projects means more scenarios. If the time budget doesn't account for this, tests may time out. Mitigate: keep existing per-depth coupling caps.
3. **Engine runner routing** — If `execution_language` is set incorrectly, Python code runs through Node.js or vice versa, producing garbage results. Mitigate: default to existing `self.language` fallback when `execution_language` is empty.

### Medium Risk
4. **Library matching with merged deps** — Python deps accidentally matched against JS profiles or vice versa. Mitigate: match each language's deps separately, then merge results.
5. **HTTP testing with wrong server** — Primary language heuristic picks wrong server framework. Mitigate: explicit priority ordering (Python server frameworks > JS server frameworks).

### Low Risk
6. **Report description strings** — "Your Python web application" when it should say "Your FastAPI + React application." Cosmetic, not functional.
7. **Prediction accuracy** — Combined dep list may produce slightly different predictions. The model already handles this; no retraining needed.

### Zero Risk (no change)
8. Single-language Python projects — identical code path (detect_languages returns {"python"}).
9. Single-language JS projects — identical code path (detect_languages returns {"javascript"}).

---

## Test Plan

### New Tests

1. **`test_detect_languages()`** — Fixtures with Python-only, JS-only, and mixed project structures. Verify set output.
2. **`test_detect_languages_ignores_build_tooling()`** — A Python project with a lone package.json from a build tool should NOT detect JavaScript.
3. **`test_merge_ingestion_results()`** — Merge two IngestionResults, verify combined deps, files, LOC, coupling points.
4. **`test_primary_language_heuristic()`** — FastAPI+React → Python primary. Express+Python scripts → JS primary. No frameworks → more LOC wins.
5. **`test_scenario_generator_multi_language()`** — With languages={"python", "javascript"}, verify ALL_CATEGORIES used, scenarios tagged with execution_language.
6. **`test_engine_execution_language_routing()`** — Scenario with execution_language="python" in a JS project → uses Python runner.
7. **`test_single_language_unchanged()`** — Regression test: single-language project produces identical results to baseline.

### Existing Tests That May Need Updating

- Tests that mock `detect_language()` → may need to also mock `detect_languages()`.
- Tests that assert exact scenario counts → may get more scenarios if test fixtures trigger multi-language detection.
- Tests that check `IngestionResult` field counts → new `language` and `secondary_languages` fields.

**Approach:** Add new fields with defaults so existing tests pass without modification. Only tests that explicitly check the new behavior need updating.

---

## Implementation Order

### Phase 1: Detection (no behavior change)
- Add `detect_languages()` function to pipeline.py
- Add `language` and `secondary_languages` fields to IngestionResult
- `detect_language()` calls `detect_languages()` internally, picks one
- All existing code still uses `detect_language()` — zero behavior change
- Tests: new detection tests only

### Phase 2: Dual Ingestion (pipeline.py and routes.py)
- Add `merge_ingestion_results()` function
- Add `_determine_primary_language()` heuristic
- Update `handle_preflight()` to run both ingesters when both detected
- Update `_run_ingestion()` to accept languages set
- Merge results before passing downstream
- Tests: merge function tests, preflight with mixed fixtures

### Phase 3: Scenario Tagging
- Add `execution_language` field to StressTestScenario
- Update `generate()` to accept `languages` parameter
- Use ALL_CATEGORIES when both languages present
- Tag each scenario with execution_language
- Tests: multi-language scenario generation

### Phase 4: Engine Routing
- Update `_execute_scenario()` to check `scenario.execution_language`
- Fallback to `self.language` when `execution_language` is empty
- Tests: execution language routing

### Phase 5: Report Polish
- Update `_generate_project_description()` for multi-language
- Update `_generate_plain_summary()` for multi-language
- Tests: report text for multi-language projects

### Phase 6: Integration Testing
- End-to-end test with a real multi-language fixture
- Verify single-language regression (identical output)
- Run full test suite (2,397+)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/mycode/pipeline.py` | `detect_languages()`, `merge_ingestion_results()`, `_determine_primary_language()`, update `_run_ingestion()` |
| `src/mycode/ingester.py` | Add `language`, `secondary_languages` fields to IngestionResult |
| `src/mycode/scenario.py` | Add `execution_language` to StressTestScenario, `languages` param to `generate()`, ALL_CATEGORIES routing |
| `src/mycode/engine.py` | Check `scenario.execution_language` in runner selection |
| `src/mycode/report.py` | Multi-language project description text |
| `src/mycode/web/routes.py` | Dual ingestion in `handle_preflight()` |
| `src/mycode/web/worker.py` | Pass `languages` to scenario generator |
| `src/mycode/web/jobs.py` | Add `detected_languages` field to Job |
| `tests/test_multi_language.py` | **NEW** — all multi-language tests |

## Files NOT Modified

| File | Reason |
|------|--------|
| `src/mycode/ingester.py` (internals) | Python ingester works correctly |
| `src/mycode/js_ingester.py` (internals) | JS ingester works correctly |
| `src/mycode/session.py` | Already handles both runtimes |
| `src/mycode/prediction.py` | Handles combined dep list naturally |
| `src/mycode/documents.py` | Receives DiagnosticReport as before |
| `src/mycode/constraints.py` | No language dependency |
| `web/app.js`, `web/index.html` | Frontend is language-agnostic |
| Profile JSON files | Language-specific, no change needed |
| Corpus data | Read-only |
