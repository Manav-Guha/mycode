# Readability Refactor Plan

**Constraint:** No behaviour changes. All 2,294 tests must pass. No new files, no deleted files, no test changes.

---

## 1. engine.py (3,925 lines)

### 1a. Consolidate duplicate budget/timeout-capping logic
- `_execute_scenario` (lines 917–1174) applies `hit_user_cap` / `hit_budget` / `failure_reason = "budget_exceeded"` logic twice — once for the callable JS harness path (lines 1031–1047) and once for the standard path (lines 1146–1159). Extract a helper `_apply_timeout_labels(result, hit_user_cap, hit_budget)` to eliminate the duplication (~20 duplicated lines).

### 1b. Extract harness cleanup into a helper
- Harness file cleanup (`for path in (harness_path, config_path): try: path.unlink(...)`) appears at lines 1073–1078 and 1134–1139. Extract `_cleanup_harness_files(*paths)`.

### 1c. Break up `_execute_scenario` (~250 lines)
- Currently one monolithic method. Split into:
  - `_prepare_harness(scenario)` → returns (harness_content, harness_config, runner, is_js, src_files, src_funcs)
  - `_run_and_parse_harness(scenario, harness_content, harness_config, runner, timeout)` → returns ScenarioResult
  - Keep `_execute_scenario` as the orchestrator calling these two helpers.
- Each piece is ~80 lines — well within the 80-line guideline.

### 1d. Add docstrings to undocumented methods
- `_build_harness_config` (line 1398) — add a one-liner.
- `_get_target_modules` (line 1777) — add a one-liner (currently has a docstring already ✓).
- `_write_harness` (line 1506) — has docstring ✓.

### 1e. Unused imports check
- All imports confirmed used. No removals.

### 1f. Do NOT touch harness template strings
- Per spec: "Do not refactor the harness body templates — they're ugly by nature."
- Lines 1892–3925 are template strings and their dicts. Leave them alone.

---

## 2. report.py (4,512 lines)

### 2a. Consolidate `_FAILURE_REASON_EXPLANATIONS` and `_FAILURE_REASON_HEADERS`
- Lines 713–779: Two parallel dicts keyed by the same failure reasons. Merge into a single `_FAILURE_REASON_INFO: dict[str, tuple[str, str]]` mapping `reason → (header, explanation)`. Update `_render_incomplete_text` and `_render_incomplete_markdown` to use the merged dict. This removes the risk of the two dicts getting out of sync.

### 2b. Extract shared logic from `_render_incomplete_text` and `_render_incomplete_markdown`
- Lines 793–882: Nearly identical structure. Extract common logic into a shared helper that returns structured data (header, explanation, formatted items list, is_summarised), then have each renderer just format for its output medium.

### 2c. Consolidate duplicate severity-filtering pattern
- The pattern `criticals = [f for f in findings if f.severity == "critical"]` / `warnings = [...]` / `infos = [...]` appears 4 times. Extract `_partition_by_severity(findings) → (criticals, warnings, infos)`.

### 2d. Refactor `_describe_step` (lines 3809–3943, ~134 lines) to data-driven table
- This is a chain of 25+ `re.match` + `if m: return ...` blocks all following the same pattern. Refactor into a list of `(compiled_pattern, format_function)` tuples with a single loop. Cuts ~80 lines of repetitive code. Keep the special cases (literal step names like `"render_memory_growth"`, `"api_timeout_handling"`) in a simple dict lookup before the regex table.

### 2e. Extract concurrency contextualisation from `_contextualise_findings`
- `_contextualise_findings` (lines 1754–1901, ~147 lines) has three distinct branches: concurrency findings (76 lines), data-size findings, and corpus stats enrichment. Extract the concurrency branch into `_contextualise_concurrency_finding(finding, user_scale, ratio, prior_state)` — this is a natural split since it has its own severity classification logic.

### 2f. Deduplicate `_COUPLING_PREFIXES`
- Defined as a local tuple in both `_describe_scenario()` (line 3525) and `_humanize_scenario_name()` (line 3755). Move to a single module-level constant.

### 2g. Clarifying comments on the three `_humanize_*` functions
- `_humanize_title_name`, `_humanize_scenario_name`, `_humanize_identifier` serve different purposes. Add a one-line "When to use" comment to each.

### 2h. Unused imports check
- All imports confirmed used.

---

## 3. documents.py (2,065 lines)

### 3a. Remove dead colour aliases
- Lines 1519–1522: `_DARK_BLUE = _BRAND`, `_BODY_GREY = _BODY`, `_LIGHT_GREY = _SUBTLE`. **Confirmed unused** — grep across all `.py` files shows only the definitions, no usage. Remove along with the "Keep old names" comment.

### 3b. Extract shared finding-card data preparation
- `_render_understanding_finding` (markdown, line 779) and `_render_pdf_finding` (PDF, line 1897) both follow the same 5-part structure: title → what we found → diagnosis → consequence → prompt → after you fix it. Extract a shared helper `_finding_card_data(f) → dict` that computes all the display fields once, then have each renderer just format the dict for its medium. Reduces risk of the two renderers diverging.

### 3c. Remove redundant local `import re` in `_extract_mb_from_text`
- Line 1084: `import re` — `re` is already imported at module level (line 14). Remove the local import.

### 3d. Remove redundant local `import re as _re` in `_short_step`
- Line 95: `import re as _re` — same issue. Use the module-level `re`. Remove local import.

### 3e. Merge `_fmt_cell` and `_fmt_cell_short`
- Lines 123–134: Identical except for which step-label formatter they call. Merge into `_fmt_cell(value, label, metric, short=False)`.

---

## 4. js_ingester.py (1,523 lines)

### 4a. Add section comment before regex fallback class
- Add `# ── Regex Fallback Parser ──` comment before `class _JsFileAnalyzer` (line 574) to clearly delineate the AST path from the regex fallback path.

### 4b. Document the AST-first-then-regex fallback strategy
- In `JsProjectIngester.ingest()` (line 933), the comments are minimal. Add a 2-line comment block explaining: "Phase 1: batch AST parse all files via Node.js subprocess. Phase 2: for any file where AST failed, fall back to regex analysis."

### 4c. Remove dead regex `_ES6_REEXPORT_RE`
- Line 132: `_ES6_REEXPORT_RE` — **confirmed unused** (only defined, never referenced). Remove.

### 4d. Remove unused import `shutil`
- Line 15: `import shutil` — **confirmed unused** (`shutil.` never called in this file). Remove.

---

## 5. ingester.py (1,124 lines)

### 5a. Verify no dead code
- All methods in `PythonProjectIngester` are called from `ingest()`. Confirmed no dead methods.
- All module-level helpers (`_get_static_call_name`, `_extract_string_list`, `_normalize_package_name`, `_is_version_outdated`, `_read_text_safe`) are used. No removals.

### 5b. Unused imports check
- All imports confirmed used. No changes needed.

**Verdict:** ingester.py is already clean. No changes.

---

## 6. scenario.py (1,833 lines)

### 6a. Verify module-level constants are used
- `_DATA_TYPE_BOOST`, `_DATA_TYPE_LOW`, `_DATA_TYPE_KEEP`, `_MAX_COUPLING_SCENARIOS_FILTERED` — **all confirmed used** in `_generate_offline`. No removals.

### 6b. Check `depth_to_coupling_cap` import
- **Confirmed used** at line 1349. No change needed.

### 6c. Check for dead code in `_generate_offline`
- This is a long function but generates scenarios from templates — it's inherently verbose. Verify no unreachable branches from previous iterations.

---

## Cross-File

### C1. Deduplicate `_COUPLING_PREFIXES` within report.py
- Two local definitions in different functions within report.py. Move to a single module-level tuple constant `_COUPLING_PREFIXES`.

---

## Execution Plan

1. **Verify baseline:** `pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_session.py --ignore=tests/test_pipeline.py -k "not (TestPipelineIntegration or TestCLIExitCode)"` — all must pass.
2. **engine.py** (1a, 1b, 1c, 1d) — highest priority, most-patched file.
3. **report.py** (2a, 2b, 2c, 2d, 2e, 2f, 2g) — most helper cleanup.
4. **documents.py** (3a, 3b, 3c, 3d, 3e) — reduce duplication.
5. **js_ingester.py** (4a, 4b, 4c, 4d) — section comments and dead code removal.
6. **scenario.py** (6a, 6b, 6c) — verify and minor cleanup.
7. **ingester.py** — no changes needed.
8. **Re-run full test suite** after each file.

---

## What I Will NOT Do

- Change test files
- Change profile JSON files
- Change web frontend (app.js, style.css, index.html)
- Change API endpoints or response formats
- Rename files
- Change public interfaces (function signatures, class names, module exports)
- Add dependencies
- Touch harness body template strings (lines 1892–3925 in engine.py)
- Create or delete files (except this plan.md)
