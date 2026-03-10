# Baseline Viability Gate — Implementation Plan

## Problem

myCode currently runs stress scenarios even when the sandbox environment is fundamentally broken — missing most dependencies, core imports failing, syntax errors in user code. This wastes time and produces reports full of harness failures that obscure real findings. A viability gate stops the pipeline early when the environment cannot produce meaningful results.

## Where It Sits in the Pipeline

```
Stage 1: Language Detection
Stage 2: Session Setup (venv creation + dependency installation)
Stage 3: Project Ingestion (parses code, detects deps, checks installed versions)
Stage 4: Component Library Matching
  ──── NEW: Stage 4.5 — Baseline Viability Gate ────
Stage 5: Conversational Interface
Stage 6: Scenario Generation
Stage 7: Scenario Review
Stage 8: Execution
Stage 9: Report Generation
```

**Why after Stage 4, not after Stage 2:**
- The ingester (Stage 3) is what discovers declared dependencies and checks which are actually installed (`is_missing` field on `DependencyInfo`).
- Library matching (Stage 4) tells us which deps have profiles (i.e., which deps actually matter for scenario generation).
- We need both signals: what's missing, and what's important.

**Why before Stage 5 (Conversation):**
- No point asking the user about their operational intent if we already know the environment can't run tests. Fail fast.

## What the Gate Checks

Three checks, in order. Each produces a pass/fail plus diagnostic detail.

### Check 1: Dependency Installation Rate

Using `DependencyInfo.is_missing` from the ingestion result:

```python
non_dev_deps = [d for d in ingestion.dependencies if not d.is_dev]
missing = [d for d in non_dev_deps if d.is_missing]
install_rate = (len(non_dev_deps) - len(missing)) / len(non_dev_deps) if non_dev_deps else 1.0
```

**Threshold:** `install_rate < 0.50` = FAIL (less than 50% of declared dependencies installed).

**Rationale:** Below 50%, most scenarios will fail due to import errors, not due to actual stress. The signal-to-noise ratio is too low.

**Edge case:** If the project declares 0 non-dev dependencies, this check passes (install_rate = 1.0). Projects with no declared deps are either stdlib-only or have inline imports — both are testable.

### Check 2: Core Dependency Importability

For dependencies that *are* installed (not missing), verify they're actually importable in the sandbox venv. This catches cases where pip reports success but the package is broken (native extension build failures, missing system libraries, etc.).

```python
def _check_importability(session, deps, language):
    """Try importing each installed dep in the sandbox. Return (importable, failed) lists."""
    if language == "python":
        # Build a small script that tries `import <name>` for each dep
        import_script = "import sys\nresults = {}\n"
        for dep in deps:
            module_name = dep.name.replace("-", "_")
            import_script += (
                f"try:\n"
                f"    __import__('{module_name}')\n"
                f"    results['{dep.name}'] = True\n"
                f"except Exception:\n"
                f"    results['{dep.name}'] = False\n"
            )
        import_script += "import json\nprint(json.dumps(results))\n"
        result = session.run_in_session(["python", "-c", import_script], timeout=15)
        # Parse JSON output...
    elif language == "javascript":
        # Build a Node script that tries require() for each dep
        # Similar pattern using session.run_in_session(["node", "-e", ...])
```

**Threshold:** If >50% of *installed* dependencies fail to import, the check fails. This is a secondary signal — it catches the "pip said OK but nothing works" case.

**Note:** This check runs inside the sandbox venv (via `session.run_in_session`), so it tests the actual execution environment.

### Check 3: Basic Syntax Validation of User Code

Verify at least some user source files parse without syntax errors. The ingester already tracks `files_failed` — we reuse that:

```python
if ingestion.files_analyzed == 0 and ingestion.files_failed > 0:
    # All files have syntax errors — nothing is testable
    syntax_viable = False
elif ingestion.files_analyzed > 0:
    syntax_rate = ingestion.files_analyzed / (ingestion.files_analyzed + ingestion.files_failed)
    syntax_viable = syntax_rate >= 0.25  # At least 25% of files must parse
else:
    syntax_viable = True  # No files found is handled elsewhere
```

**Threshold:** `syntax_rate < 0.25` = FAIL (less than 25% of source files parse).

**Rationale:** Lower threshold than deps because partial file parsing is normal (e.g., Python 3.12 syntax in a 3.11 parser). But if 75%+ of files are unparseable, the ingestion data is too thin to generate meaningful scenarios.

## Composite Gate Decision

```python
@dataclass
class ViabilityResult:
    viable: bool
    install_rate: float          # 0.0–1.0
    import_rate: float           # 0.0–1.0 (of installed deps)
    syntax_rate: float           # 0.0–1.0
    missing_deps: list[str]      # names of missing deps
    unimportable_deps: list[str] # names of import-failed deps
    reason: str                  # human-readable explanation if not viable
    suggest_docker: bool         # True if not containerised and gate failed

# Gate fails if ANY check fails
viable = (
    install_rate >= 0.50
    and import_rate >= 0.50
    and syntax_viable
)
```

## What Happens When the Gate Fails

### 1. Pipeline stops before scenario generation

The pipeline does NOT proceed to Stages 5–9. This is a hard stop, not a warning.

```python
# In run_pipeline(), after _run_library_matching():
viability = _run_viability_gate(session, ingestion, matches, language, result, config)
if not viability.viable:
    # Generate a "baseline failed" report and return
    _run_baseline_failed_report(viability, ingestion, matches, config, result)
    result.total_duration_ms = _elapsed_ms(pipeline_start)
    _safe_save(recorder, result)
    return result
```

### 2. Produce a "baseline failed" report

Instead of the normal diagnostic report, the report generator produces a special short report explaining why testing couldn't proceed. This is a new method on `ReportGenerator` (or a standalone function).

The report includes:
- **What happened:** "myCode could not establish a healthy test environment for your project."
- **Dependency status:** "4 of 9 dependencies installed successfully (44%). Missing: pandas, scikit-learn, tensorflow, opencv-python."
- **Import status:** (if relevant) "2 of 5 installed dependencies failed to import: numpy (missing libopenblas), scipy (missing liblapack)."
- **Syntax status:** (if relevant) "3 of 12 source files could not be parsed."
- **Docker suggestion:** If `--containerised` was NOT used: "Try running with `--containerised` for better dependency isolation. Docker can install system-level libraries that may be missing from your local environment."
- **Next steps:** "Check that your project's dependencies install correctly in a clean environment. Run `pip install -r requirements.txt` in a fresh venv to verify."

This report is stored in `result.report` as a `DiagnosticReport` with a special `baseline_failed` flag so the CLI can format it appropriately.

### 3. PipelineResult records the gate outcome

```python
@dataclass
class PipelineResult:
    # ... existing fields ...
    viability: Optional[ViabilityResult] = None  # NEW
```

The viability result is always recorded (even when the gate passes), so the report generator can use it for the existing confidence note.

## Thresholds Summary

| Check | Threshold | Rationale |
|---|---|---|
| Dependency install rate | < 50% = FAIL | Below half, most scenarios will be import-error noise |
| Import success rate | < 50% of installed = FAIL | Catches broken installs pip didn't flag |
| Syntax parse rate | < 25% = FAIL | Lower bar because partial parse failures are normal |

## Files That Need to Change

### 1. `src/mycode/pipeline.py` — Main orchestration changes

**New function:** `_run_viability_gate()`

```python
def _run_viability_gate(
    session: SessionManager,
    ingestion: IngestionResult,
    matches: list[ProfileMatch],
    language: str,
    result: PipelineResult,
    config: PipelineConfig,
) -> ViabilityResult:
    """Stage 4.5: Check whether the sandbox can produce meaningful results."""
    stage_start = time.monotonic()

    non_dev = [d for d in ingestion.dependencies if not d.is_dev]
    missing = [d for d in non_dev if d.is_missing]
    installed = [d for d in non_dev if not d.is_missing]

    # Check 1: Install rate
    install_rate = (len(non_dev) - len(missing)) / len(non_dev) if non_dev else 1.0

    # Check 2: Import rate (only for installed deps)
    importable, unimportable = _check_importability(session, installed, language)
    import_rate = len(importable) / len(installed) if installed else 1.0

    # Check 3: Syntax rate
    total_files = ingestion.files_analyzed + ingestion.files_failed
    syntax_rate = ingestion.files_analyzed / total_files if total_files > 0 else 1.0

    viable = install_rate >= 0.50 and import_rate >= 0.50 and syntax_rate >= 0.25

    # Build reason string
    reasons = []
    if install_rate < 0.50:
        reasons.append(
            f"Only {len(non_dev) - len(missing)} of {len(non_dev)} "
            f"dependencies installed ({install_rate:.0%})."
        )
    if import_rate < 0.50:
        reasons.append(
            f"Only {len(importable)} of {len(installed)} installed "
            f"dependencies are importable ({import_rate:.0%})."
        )
    if syntax_rate < 0.25:
        reasons.append(
            f"Only {ingestion.files_analyzed} of {total_files} source "
            f"files could be parsed ({syntax_rate:.0%})."
        )

    is_containerised = os.environ.get("MYCODE_CONTAINERISED") == "1"

    viability = ViabilityResult(
        viable=viable,
        install_rate=install_rate,
        import_rate=import_rate,
        syntax_rate=syntax_rate,
        missing_deps=[d.name for d in missing],
        unimportable_deps=unimportable,
        reason=" ".join(reasons),
        suggest_docker=not viable and not is_containerised,
    )

    result.viability = viability
    result.stages.append(StageResult(
        stage="viability_gate",
        success=viable,
        duration_ms=_elapsed_ms(stage_start),
        error=viability.reason if not viable else "",
    ))

    if not viable:
        logger.warning("Viability gate FAILED: %s", viability.reason)
    else:
        logger.info(
            "Viability gate passed: install=%.0f%% import=%.0f%% syntax=%.0f%%",
            install_rate * 100, import_rate * 100, syntax_rate * 100,
        )

    return viability
```

**New function:** `_check_importability()`

```python
def _check_importability(
    session: SessionManager,
    installed_deps: list[DependencyInfo],
    language: str,
) -> tuple[list[str], list[str]]:
    """Test-import installed deps in the sandbox. Returns (importable, failed) name lists."""
    if not installed_deps:
        return [], []

    if language == "python":
        return _check_python_imports(session, installed_deps)
    else:
        return _check_js_imports(session, installed_deps)


def _check_python_imports(
    session: SessionManager,
    deps: list[DependencyInfo],
) -> tuple[list[str], list[str]]:
    """Try importing each Python dep in the sandbox venv."""
    lines = ["import json", "results = {}"]
    for dep in deps:
        module = dep.name.replace("-", "_").lower()
        lines.append(f"try:")
        lines.append(f"    __import__('{module}')")
        lines.append(f"    results['{dep.name}'] = True")
        lines.append(f"except Exception:")
        lines.append(f"    results['{dep.name}'] = False")
    lines.append("print(json.dumps(results))")
    script = "\n".join(lines)

    try:
        result = session.run_in_session(
            ["python", "-c", script], timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().splitlines()[-1])
            importable = [name for name, ok in data.items() if ok]
            failed = [name for name, ok in data.items() if not ok]
            return importable, failed
    except Exception as exc:
        logger.debug("Import check failed: %s", exc)

    # If the check itself fails, assume all installed deps are importable
    # (don't block the pipeline on a meta-failure)
    return [d.name for d in deps], []


def _check_js_imports(
    session: SessionManager,
    deps: list[DependencyInfo],
) -> tuple[list[str], list[str]]:
    """Try requiring each JS dep in the sandbox node_modules."""
    lines = ["const results = {};"]
    for dep in deps:
        lines.append(f"try {{ require('{dep.name}'); results['{dep.name}'] = true; }}")
        lines.append(f"catch(e) {{ results['{dep.name}'] = false; }}")
    lines.append("console.log(JSON.stringify(results));")
    script = "\n".join(lines)

    try:
        result = session.run_in_session(
            ["node", "-e", script], timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().splitlines()[-1])
            importable = [name for name, ok in data.items() if ok]
            failed = [name for name, ok in data.items() if not ok]
            return importable, failed
    except Exception as exc:
        logger.debug("JS import check failed: %s", exc)

    return [d.name for d in deps], []
```

**Modify `run_pipeline()`** — insert gate between library matching and conversation:

```python
# After Stage 4 (library matching), before Stage 5 (conversation):

# ── Stage 4.5: Viability Gate ──
viability = _run_viability_gate(
    session, ingestion, matches, language, result, config,
)
if not viability.viable:
    _run_baseline_failed_report(
        viability, ingestion, matches, config, result,
    )
    result.total_duration_ms = _elapsed_ms(pipeline_start)
    _safe_save(recorder, result)
    return result
```

**New function:** `_run_baseline_failed_report()`

```python
def _run_baseline_failed_report(
    viability: ViabilityResult,
    ingestion: IngestionResult,
    matches: list[ProfileMatch],
    config: PipelineConfig,
    result: PipelineResult,
) -> None:
    """Generate a short diagnostic report explaining why testing couldn't proceed."""
    stage_start = time.monotonic()

    project_name = _infer_project_name(Path(config.project_path))

    sections: list[str] = []
    sections.append(
        f"# {project_name} — Baseline Viability Report\n\n"
        "myCode could not establish a healthy test environment for your project. "
        "Stress testing was not attempted because the results would not be meaningful.\n"
    )

    # Dependency status
    if viability.install_rate < 1.0:
        installed_count = len(ingestion.dependencies) - len(viability.missing_deps)
        total_count = len([d for d in ingestion.dependencies if not d.is_dev])
        sections.append(
            f"## Dependency Installation\n\n"
            f"{installed_count} of {total_count} dependencies installed "
            f"({viability.install_rate:.0%}).\n\n"
            f"**Could not install:** {', '.join(viability.missing_deps[:10])}"
            + ("..." if len(viability.missing_deps) > 10 else "") + "\n"
        )

    # Import status
    if viability.unimportable_deps:
        sections.append(
            f"## Import Failures\n\n"
            f"These dependencies installed but could not be imported:\n"
            + "\n".join(f"- {name}" for name in viability.unimportable_deps[:10])
            + "\n\nThis typically means a native library or system dependency is missing.\n"
        )

    # Syntax status
    if viability.syntax_rate < 1.0:
        total = ingestion.files_analyzed + ingestion.files_failed
        sections.append(
            f"## Source File Parsing\n\n"
            f"{ingestion.files_analyzed} of {total} source files parsed "
            f"successfully ({viability.syntax_rate:.0%}).\n"
        )

    # Docker suggestion
    if viability.suggest_docker:
        sections.append(
            "## Suggestion: Try Docker Mode\n\n"
            "Run with `--containerised` for better dependency isolation. "
            "Docker can install system-level libraries (C extensions, native "
            "bindings) that may be missing from your local environment.\n\n"
            "```\nmycode /path/to/project --containerised\n```\n"
        )

    # Next steps
    sections.append(
        "## Next Steps\n\n"
        "1. Check that your project's dependencies install correctly in a "
        "clean virtual environment.\n"
        "2. Run `pip install -r requirements.txt` (or `npm install`) in a "
        "fresh environment to verify.\n"
        "3. Fix any missing system libraries or build dependencies.\n"
        "4. Re-run myCode once the environment is healthy.\n"
    )

    report_text = "\n".join(sections)

    # Create a DiagnosticReport with baseline_failed flag
    report = DiagnosticReport(
        project_name=project_name,
        report_text=report_text,
        baseline_failed=True,
    )
    result.report = report

    result.stages.append(StageResult(
        stage="report_generation",
        duration_ms=_elapsed_ms(stage_start),
    ))
```

**Modify `PipelineResult`** — add viability field:

```python
@dataclass
class PipelineResult:
    # ... existing fields ...
    viability: Optional["ViabilityResult"] = None  # NEW
```

### 2. `src/mycode/report.py` — Add `baseline_failed` flag to DiagnosticReport

```python
@dataclass
class DiagnosticReport:
    # ... existing fields ...
    baseline_failed: bool = False  # NEW — True when viability gate stopped the pipeline
```

Also update `_build_confidence_note()` to use the viability data when available (passed through from PipelineResult), instead of re-computing missing dep counts.

### 3. `src/mycode/cli.py` — Display baseline-failed report differently

In the CLI output logic, detect `result.report.baseline_failed` and:
- Print the report text directly (it's already formatted as plain text/markdown)
- Set exit code to 2 (distinct from 0=success, 1=pipeline error)
- Skip the normal report formatting path

```python
# In the CLI's result display section:
if result.report and result.report.baseline_failed:
    print(result.report.report_text)
    if result.viability and result.viability.suggest_docker:
        print("\nHint: try --containerised for better isolation.")
    sys.exit(2)
```

### 4. `tests/test_pipeline.py` — New test cases

```python
class TestViabilityGate:
    """Tests for the baseline viability gate."""

    def test_gate_passes_all_deps_installed(self):
        """Gate passes when all deps are installed."""
        ...

    def test_gate_fails_below_50_percent_installed(self):
        """Gate fails when <50% of deps are installed."""
        ...

    def test_gate_fails_below_50_percent_importable(self):
        """Gate fails when <50% of installed deps can be imported."""
        ...

    def test_gate_fails_below_25_percent_syntax(self):
        """Gate fails when <25% of source files parse."""
        ...

    def test_gate_passes_no_dependencies(self):
        """Gate passes when project has zero declared deps."""
        ...

    def test_gate_ignores_dev_dependencies(self):
        """Only non-dev deps count toward install rate."""
        ...

    def test_gate_suggests_docker_when_not_containerised(self):
        """suggest_docker is True when gate fails outside Docker."""
        ...

    def test_gate_no_docker_suggestion_when_containerised(self):
        """suggest_docker is False when already in Docker."""
        ...

    def test_pipeline_stops_on_gate_failure(self):
        """Pipeline returns early with baseline_failed report."""
        ...

    def test_pipeline_continues_on_gate_pass(self):
        """Pipeline proceeds to conversation when gate passes."""
        ...

    def test_import_check_fallback_on_error(self):
        """If the import check script itself fails, assume all OK."""
        ...

    def test_baseline_failed_report_content(self):
        """Report includes missing deps, rates, and docker suggestion."""
        ...
```

Most of these can be fast tests (mocked session, no real venv needed). The import check tests need a mock `session.run_in_session` that returns controlled JSON output.

## Design Decisions

1. **Hard gate, not a warning.** Below the thresholds, the environment is too broken for scenarios to be informative. Running them anyway wastes time and produces a report full of "Tests myCode Could Not Run" — we already have that section, this gate prevents it from dominating the report.

2. **Import check is cheap.** A single subprocess call with 15s timeout. For 20 deps, this takes <2s. Worth the cost for the signal it provides.

3. **Fail-open on meta-errors.** If the import check script itself crashes, we don't block the pipeline. The check failing is not the same as deps failing.

4. **Viability result stored even on pass.** This lets the report generator use the exact import/install rates for confidence notes, replacing the current `_build_confidence_note()` approximation.

5. **Exit code 2 for baseline failure.** Distinguishes "your project needs fixing" (exit 2) from "myCode itself broke" (exit 1) and "all good" (exit 0). Useful for CI integration.

6. **Docker suggestion only when not already containerised.** Avoids the unhelpful "try Docker" message when you're already in Docker and deps still fail.

Notes to address:
1. On the import check: add a common name-mapping dict for known cases (scikit-learn→sklearn, Pillow→PIL, python-dotenv→dotenv, beautifulsoup4→bs4, opencv-python→cv2). The replace("-", "_") heuristic will miss these. This doesn't need to be exhaustive — just cover the common ones.
2. Make the baseline failed report format consistent with the normal report format (plain text with ASCII dividers, not markdown headers)
3. Add stack context to ViabilityResult: include language, framework, and the full dependency list with declared versions. This data is not needed for the v1 gate decision, but it makes the result self-contained for future corpus integration. Design the dataclass so it can be serialised to JSON and stored alongside failure signatures.