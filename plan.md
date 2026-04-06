# Plan: Fix Prompt Pipeline Enrichment — Ingestion Data Flow

**STATUS: AWAITING REVIEW**

Scope: Flow four categories of ingestion-level data into Finding objects during report generation, so remediation patterns and fix prompts can reference installed versions, outdated status, call chains, and decorator presence.

Background: The fix prompt rebuild (f84978f) addressed finding-object-level data. The Gemini integration (58b3348) addressed source code reading. This task addresses ingestion-level data that exists in the pipeline but never reaches fix prompts.

---

## 1. Where IngestionResult Exists Alongside Findings

**`ReportGenerator.generate()` (report.py:1058)** receives `ingestion: IngestionResult` and has it in scope for the entire method.

**The gap:** The finding assembly loop lives inside `_analyze_execution()` (report.py:1184), which receives only `dep_file_map` — a narrow derivative of ingestion. The full `IngestionResult` is not available inside the loop.

**However, ingestion IS available after the loop returns.** The existing enrichment point (report.py:1271) already iterates over `report.findings` with access to `ingestion`. The Gemini enrichment (`enrich_finding`) is called here.

**Proposed insertion point:** A new deterministic enrichment step at report.py, between the existing sort (line 1237) and the Gemini enrichment (line 1258). This runs after all findings are assembled, grouped, and sorted — but before any LLM calls.

```
Line 1237: report.findings.sort(...)          ← findings finalised

NEW:       _enrich_findings_from_ingestion()  ← deterministic, fast

Line 1258: for f in report.findings:          ← existing Gemini enrichment
               enrich_finding(f, ingestion, ...)
```

This avoids modifying `_analyze_execution()`'s signature, keeps the enrichment concern separate, and runs before Gemini so the LLM prompt can include version/decorator context.

---

## 2. Data Flow: installed_version / latest_version / is_outdated

### Current state
- `DependencyInfo` (ingester.py:159) has `installed_version`, `latest_version`, `is_outdated` fields — all populated during ingestion.
- `_flag_version_discrepancies()` (report.py:2103) reads these fields and converts them to strings in `report.version_flags`. Structured data is lost.
- Remediation patterns in documents.py reference `deps_str` (comma-separated names) but never version numbers.

### Proposed data flow

**New fields on Finding:**

```python
# Version context for affected dependencies (populated by ingestion enrichment)
dep_versions: dict[str, str] = field(default_factory=dict)
# e.g. {"flask": "2.0.1", "pandas": "1.5.3"}

dep_latest_versions: dict[str, str] = field(default_factory=dict)
# e.g. {"flask": "3.1.0", "pandas": "2.2.0"}

dep_outdated: list[str] = field(default_factory=list)
# e.g. ["flask", "pandas"] — subset of affected_dependencies that are outdated
```

**Lookup mechanism:**

Pre-build a `dict[str, DependencyInfo]` from `ingestion.dependencies` at the start of the enrichment step:

```python
dep_lookup = {d.name: d for d in ingestion.dependencies}
```

For each finding, iterate `f.affected_dependencies` and look up each in `dep_lookup`:

```python
for dep_name in f.affected_dependencies:
    dep = dep_lookup.get(dep_name)
    if dep and dep.installed_version:
        f.dep_versions[dep_name] = dep.installed_version
    if dep and dep.latest_version:
        f.dep_latest_versions[dep_name] = dep.latest_version
    if dep and dep.is_outdated:
        f.dep_outdated.append(dep_name)
```

**Integration with `_remediation_fields()`** (documents.py:1140):

Add three new fields to the returned dict:

```python
# Version-aware dependency string
# e.g. "flask 2.0.1 (latest: 3.1.0), pandas 1.5.3 (latest: 2.2.0)"
"deps_versioned": ...,

# Outdated count
# e.g. "2 of 3 dependencies are outdated"
"deps_outdated_note": ...,

# Simple boolean for template branching
"has_outdated": ...,
```

**Integration with remediation patterns:**

Patterns that currently end with `Dependencies involved: {deps_str}` can conditionally append version context:

- `_pat_startup_failure` (documents.py:1280) — `missing_server_dependency` sub-pattern: append "Installed version: X, latest: Y" when outdated
- `_pat_streamlit_memory` (documents.py:1350) — append "Note: {dep} is outdated (X → Y), upgrading may reduce memory footprint"
- `_pat_flask_concurrency` (documents.py:1419) — append "Note: Flask {X} → {Y} includes concurrency improvements"
- `_pat_fastapi_concurrency` (documents.py:1247) — append version note when FastAPI is outdated

**Pattern change example** (before/after):

```
Before: "Dependencies involved: flask, sqlalchemy."
After:  "Dependencies involved: flask 2.0.1 (outdated — latest 3.1.0), sqlalchemy 2.0.0."
```

### Serialisation

`_finding_dict()` (report.py:685) — add:

```python
"dep_versions": dict(f.dep_versions) if f.dep_versions else None,
"dep_outdated": list(f.dep_outdated) if f.dep_outdated else None,
```

---

## 3. Data Flow: FunctionFlow Call Graph

### Current state
- `FunctionFlow` (ingester.py:173) captures caller→callee edges with file and line number.
- `ingestion.function_flows` is a flat list of edges — the full call graph.
- Never accessed by report generation.

### Proposed data flow

**New field on Finding:**

```python
call_chain: list[str] = field(default_factory=list)
# e.g. ["index()", "get_data()", "requests.get()"]
# Ordered from entry point → leaf call
```

**Traversal mechanism:**

Build an adjacency list from `ingestion.function_flows`:

```python
# caller → [callee1, callee2, ...]
call_graph: dict[str, list[str]] = defaultdict(list)
for flow in ingestion.function_flows:
    call_graph[flow.caller].append(flow.callee)
```

For each finding with `source_function`, traverse the call graph depth-first from that function. Collect the chain up to depth limit of **4** (entry point → 3 levels of callees). Stop at:
- Depth limit reached
- No further edges
- Cycle detected (mark with "→ ...")

**Why depth 4?** Most vibe-coded apps are shallow. The useful information is "your route handler calls X which calls Y which blocks" — 3 hops covers this. Beyond 4, the chain becomes noise.

**Matching challenge:** `FunctionFlow.caller` uses qualified names like `"app.handle_request"` or `"app.DataProcessor.fetch"`. `Finding.source_function` is just `"handle_request"`. Resolution approach:
1. Build a reverse lookup: `{simple_name: [qualified_name1, qualified_name2]}`
2. When finding has `source_file` + `source_function`, prefer the qualified name that matches `source_file`'s module
3. If ambiguous (same name in multiple files), skip enrichment for that finding

**Integration with remediation patterns:**

Add `"call_chain"` to `_remediation_fields()`:

```python
"call_chain": " → ".join(f.call_chain) if f.call_chain else "",
```

Patterns that benefit:
- `_pat_flask_concurrency` — "Flask handles requests synchronously. `index()` → `get_data()` → `requests.get()` — the blocking call is 2 levels deep"
- `_pat_fastapi_concurrency` — "The blocking work is in: `handler()` → `sync_db_query()`"
- `_pat_blocking_io` (if exists) — trace the blocking I/O to its source

**Pattern change example:**

```
Before: "In `app.py`, the `index()` function delegates blocking work..."
After:  "In `app.py`, the call chain `index()` → `get_data()` → `requests.get()` includes a blocking call..."
```

### Serialisation

```python
"call_chain": list(f.call_chain) if f.call_chain else None,
```

---

## 4. Data Flow: Decorators

### Current state
- `FunctionInfo.decorators` (ingester.py:114) is a `list[str]` of decorator names — populated during AST parsing.
- Examples: `["cache", "lru_cache", "st.cache_data", "st.cache_resource", "app.route", "retry"]`
- Never accessed by report generation.

### Proposed data flow

**New field on Finding:**

```python
source_decorators: list[str] = field(default_factory=list)
# e.g. ["app.route", "st.cache_data"]
```

**Lookup mechanism:**

Reuse the same `FileAnalysis` lookup used by `fix_enrichment.extract_function_body()`:

```python
for fa in ingestion.file_analyses:
    if fa.file_path == f.source_file:
        for fi in fa.functions:
            if fi.name == f.source_function:
                f.source_decorators = list(fi.decorators)
                break
        break
```

**Integration with remediation patterns:**

Add to `_remediation_fields()`:

```python
"has_cache_decorator": any(
    d in ("cache", "lru_cache", "st.cache_data", "st.cache_resource",
          "functools.cache", "functools.lru_cache")
    for d in f.source_decorators
),
"decorators_str": ", ".join(f"@{d}" for d in f.source_decorators) if f.source_decorators else "",
```

**Patterns that change behaviour based on decorators:**

- `_pat_streamlit_memory` (documents.py:1350) — currently always recommends `@st.cache_data`. With decorator data:
  - If `@st.cache_data` already present → "Caching is applied but memory still grows — check cache size limits with `max_entries` parameter or `ttl`"
  - If no cache decorator → "Wrap with `@st.cache_data`" (current behaviour)

- `_pat_streamlit_response_time` (documents.py:1384) — same logic: detect if caching is already attempted.

- `_pat_unbounded_cache_growth` (if the pattern exists) — if `@lru_cache` is present without `maxsize`, the fix prompt can say "Add `maxsize=128` to the existing `@lru_cache`"

**Pattern change example:**

```
Before: "Wrap expensive computations with `@st.cache_data`"
After:  "`@st.cache_data` is already applied to `process_data()` — add `max_entries=100` or `ttl=300` to bound cache growth"
```

### Serialisation

```python
"source_decorators": list(f.source_decorators) if f.source_decorators else None,
```

---

## 5. Implementation Structure

### New function in report.py

```python
def _enrich_findings_from_ingestion(
    findings: list[Finding],
    ingestion: IngestionResult,
) -> None:
    """Attach ingestion-level data to findings (in-place, deterministic)."""
```

This is a **single function** that handles all four enrichments in one pass. It:
1. Pre-builds lookup structures (dep_lookup, call_graph adjacency list, function decorator index)
2. Iterates findings once, enriching each

**Why not a separate module?** Unlike Gemini enrichment (which has LLM calls, timeout handling, fallback logic), this is pure deterministic data wiring — 40-60 lines. It belongs in report.py alongside the other finding assembly logic.

### Call site in `ReportGenerator.generate()`

After line 1237 (sort), before line 1258 (Gemini enrichment):

```python
# 4b. Attach ingestion-level data to findings (versions, call graph, decorators)
_enrich_findings_from_ingestion(report.findings, ingestion)
```

### Changes to `_remediation_fields()` in documents.py

Add 5 new keys to the returned dict:
- `deps_versioned` — version-annotated dependency string
- `deps_outdated_note` — "X of Y dependencies are outdated"
- `has_outdated` — boolean for template branching
- `call_chain` — formatted call chain string
- `has_cache_decorator` — boolean for cache-aware patterns

### Pattern modifications in documents.py

| Pattern | Line | Change |
|---------|------|--------|
| `_pat_fastapi_concurrency` | 1247 | Append call chain to diagnosis, version note to fix |
| `_pat_startup_failure` | 1280 | Version note on `missing_server_dependency` sub-pattern |
| `_pat_streamlit_memory` | 1350 | Cache-decorator-aware fix, version note |
| `_pat_streamlit_response_time` | 1384 | Cache-decorator-aware fix |
| `_pat_flask_concurrency` | 1419 | Append call chain, version note |
| Generic `_build_fix` fallback | ~1750 | Append outdated note when `has_outdated` |

---

## 6. Integration with Gemini Fix Enrichment (58b3348)

The deterministic enrichment runs **before** Gemini enrichment. This means:
- The Gemini prompt (in `fix_enrichment.py`) can access `f.dep_versions`, `f.dep_outdated`, `f.call_chain`, `f.source_decorators` if we update the user message template.
- **Proposal:** Do NOT change the Gemini prompt template in this task. The source code already contains decorator and version information implicitly. Adding structured metadata to the LLM prompt is a separate improvement — keep this task focused on deterministic enrichment.

The `_build_diagnosis()` call inside `get_llm_fix_suggestion()` will automatically pick up the improved diagnosis text (since patterns now include version/chain/decorator data).

---

## 7. Risk Assessment

### Performance
- **dep_lookup construction:** O(n) where n = number of dependencies. Typically 5-30 dependencies. Negligible.
- **call_graph construction:** O(e) where e = number of function flow edges. For a 50-file project, typically 200-500 edges. Negligible.
- **Per-finding enrichment:** O(d + depth) per finding where d = affected_dependencies count, depth ≤ 4. Typically 5-10 findings × 2-3 deps × depth 4. Negligible.
- **Total added time:** <1ms for typical projects. No I/O, no network calls, pure in-memory data wiring.

### Memory
- New fields on Finding add ~200 bytes per finding (dict + list of short strings). With 10 findings, that's 2KB. Negligible even for large repos.
- Lookup structures (dep_lookup, call_graph) are temporary — built at enrichment time, garbage collected after. For a 1000-function project with 5000 edges, ~100KB. Acceptable.

### Correctness risks
- **Call graph cycle:** Handled by cycle detection with depth limit. Worst case: chain is truncated, not incorrect.
- **Ambiguous function names:** When same function name exists in multiple files, disambiguation via `source_file` resolves most cases. If still ambiguous, skip — silent no-op, not incorrect.
- **Missing version data:** `installed_version` may be `None` for some dependencies (not installed, version detection failed). All lookups use `.get()` with fallback — no KeyError risk. Fields simply stay empty.
- **Stale decorator data:** Decorators reflect the state at ingestion time, which is the project copy in the session venv — matches what the stress test ran against. No staleness risk.

### Breaking change risk
- New fields on Finding all default to empty (dict/list). Existing code that constructs Finding objects (tests, report assembly) is unaffected.
- `_remediation_fields()` adds new keys — existing patterns that don't use them are unaffected.
- JSON output adds new nullable keys — backwards compatible for consumers.

---

## 8. Files Changed

| File | Change |
|------|--------|
| `src/mycode/report.py` | 4 new fields on Finding, `_enrich_findings_from_ingestion()` function, call site in generate(), serialisation in `_finding_dict()` |
| `src/mycode/documents.py` | 5 new keys in `_remediation_fields()`, 6 pattern modifications (version/chain/decorator aware) |
| `tests/test_fix_enrichment.py` | Tests for ingestion enrichment (version lookup, call chain traversal, decorator attachment, edge cases) |
| `tests/test_documents.py` | Tests for enriched remediation patterns (version-aware deps_str, cache-decorator branching, call chain in diagnosis) |

---

## Not In Scope

- Gemini prompt template changes (source code already contains this data implicitly)
- New remediation patterns (only modifying existing ones)
- `CouplingPoint` usage (lower value than the four fields above — defer)
- `FunctionInfo.is_async` flow (marginal value over decorator detection)
- `DependencyInfo.required_version` flow (marginal value — installed vs latest is what matters)
- Cross-finding version deduplication (if 3 findings reference flask, each gets version data independently — acceptable)
