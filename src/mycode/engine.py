"""Execution Engine (D2) — Runs stress tests inside the Session Manager's sandbox.

Takes StressTestScenario objects from the Scenario Generator, generates and
executes test harness scripts inside the session's virtual environment,
monitors resources, captures errors, and returns structured results for the
Report Generator.

Pure Python. No LLM dependency.
"""

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.ingester import IngestionResult
from mycode.scenario import StressTestScenario
from mycode.session import SessionManager, SessionResult

logger = logging.getLogger(__name__)

# Markers for extracting structured output from harness scripts
_RESULTS_START = "__MYCODE_RESULTS_START__"
_RESULTS_END = "__MYCODE_RESULTS_END__"

# Truncation limits
_SNIPPET_MAX = 2000

# Hard per-scenario timeout cap — prevents any single scenario from hanging
# the entire run, regardless of what resource_limits say.
_SCENARIO_TIMEOUT_CAP = 30

# JavaScript-specific categories — executed via Node.js harness
_JS_CATEGORIES = frozenset({
    "async_failures",
    "event_listener_accumulation",
    "state_management_degradation",
})


# ── Data Classes ──


@dataclass
class StepResult:
    """Result of a single step within a scenario execution.

    Attributes:
        step_name: Identifier for this step (e.g. "data_size_1000").
        parameters: Parameters used for this step (e.g. {"data_size": 1000}).
        execution_time_ms: Wall-clock time for this step.
        memory_peak_mb: Peak memory usage during this step (via tracemalloc).
        error_count: Number of errors recorded during this step.
        errors: List of error dicts with type, message, and optional traceback.
        measurements: Additional measurements collected during this step.
        resource_cap_hit: Which cap was hit ("memory", "timeout", or "").
        stdout_snippet: Tail of stdout from harness (for debugging).
        stderr_snippet: Tail of stderr from harness (for debugging).
    """

    step_name: str
    parameters: dict = field(default_factory=dict)
    execution_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    error_count: int = 0
    errors: list = field(default_factory=list)
    measurements: dict = field(default_factory=dict)
    resource_cap_hit: str = ""
    stdout_snippet: str = ""
    stderr_snippet: str = ""


@dataclass
class ScenarioResult:
    """Complete execution result for a single stress test scenario.

    Attributes:
        scenario_name: Name of the scenario that was executed.
        scenario_category: Stress category (e.g. "data_volume_scaling").
        status: "completed", "partial", "failed", or "skipped".
        steps: Individual step results.
        total_execution_time_ms: Total wall-clock time including harness overhead.
        peak_memory_mb: Highest peak memory across all steps.
        total_errors: Sum of errors across all steps.
        failure_indicators_triggered: Which failure indicators from the scenario fired.
        resource_cap_hit: Whether any resource cap was hit.
        summary: Plain-language summary of what happened.
    """

    scenario_name: str
    scenario_category: str
    status: str
    steps: list[StepResult] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    total_errors: int = 0
    failure_indicators_triggered: list[str] = field(default_factory=list)
    resource_cap_hit: bool = False
    summary: str = ""


@dataclass
class ExecutionEngineResult:
    """Complete output from the Execution Engine.

    Attributes:
        scenario_results: Results for each executed scenario.
        total_execution_time_ms: Total wall-clock time for all scenarios.
        scenarios_completed: Count of fully completed scenarios.
        scenarios_failed: Count of failed or partially completed scenarios.
        scenarios_skipped: Count of skipped scenarios.
        warnings: Non-fatal issues encountered during execution.
    """

    scenario_results: list[ScenarioResult] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    scenarios_completed: int = 0
    scenarios_failed: int = 0
    scenarios_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


class EngineError(Exception):
    """Base exception for Execution Engine errors."""


# ── Execution Engine ──


class ExecutionEngine:
    """Runs stress test scenarios inside the Session Manager's sandbox.

    Takes StressTestScenario objects from the Scenario Generator, generates
    self-contained test harness scripts, executes them inside the session's
    sandbox, and returns structured results.

    The engine:
    - Generates synthetic data based on scenario configurations
    - Monitors resources: memory, CPU timing, error counts
    - Enforces resource caps: memory ceiling, timeout
    - Records controlled termination when caps are exceeded (finding, not crash)
    - Handles user code crashes gracefully — catches, records, continues

    Usage::

        engine = ExecutionEngine(session, ingestion, language="javascript")
        result = engine.execute(scenarios)
    """

    def __init__(
        self,
        session: SessionManager,
        ingestion: IngestionResult,
        language: str = "python",
    ):
        if not session._setup_complete:
            raise EngineError(
                "Session must be set up before creating ExecutionEngine."
            )
        self.session = session
        self.ingestion = ingestion
        self.language = language.lower()

    def execute(
        self,
        scenarios: list[StressTestScenario],
    ) -> ExecutionEngineResult:
        """Execute all scenarios and return aggregated results.

        Each scenario is executed independently. A failure in one scenario
        does not prevent subsequent scenarios from running.
        """
        if not scenarios:
            return ExecutionEngineResult(warnings=["No scenarios to execute."])

        logger.info("Starting execution of %d scenarios", len(scenarios))
        sys.stderr.flush()

        results: list[ScenarioResult] = []
        warnings: list[str] = []
        start = time.perf_counter()

        for scenario in scenarios:
            try:
                result = self._execute_scenario(scenario)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Scenario '%s' engine error: %s", scenario.name, e,
                )
                results.append(ScenarioResult(
                    scenario_name=scenario.name,
                    scenario_category=scenario.category,
                    status="failed",
                    summary=f"Engine error: {type(e).__name__}: {str(e)[:500]}",
                ))
                warnings.append(
                    f"Scenario '{scenario.name}' encountered an engine error: {e}"
                )

        total_ms = (time.perf_counter() - start) * 1000
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status in ("failed", "partial"))
        skipped = sum(1 for r in results if r.status == "skipped")

        return ExecutionEngineResult(
            scenario_results=results,
            total_execution_time_ms=round(total_ms, 2),
            scenarios_completed=completed,
            scenarios_failed=failed,
            scenarios_skipped=skipped,
            warnings=warnings,
        )

    def _execute_scenario(self, scenario: StressTestScenario) -> ScenarioResult:
        """Execute a single stress test scenario."""
        config = scenario.test_config
        resource_limits = config.get("resource_limits", {})
        timeout = resource_limits.get(
            "timeout_seconds", self.session.resource_caps.timeout_seconds,
        )
        # Enforce hard cap — no single scenario may exceed _SCENARIO_TIMEOUT_CAP
        timeout = min(timeout, _SCENARIO_TIMEOUT_CAP)

        logger.info(
            "Executing scenario: %s [%s] (timeout=%ds)",
            scenario.name, scenario.category, timeout,
        )
        sys.stderr.flush()

        start = time.perf_counter()

        # Build harness script and config — route to correct runtime
        # JS-specific categories always use Node.js; shared categories
        # use the project language to pick the right harness.
        is_js = (
            scenario.category in _JS_CATEGORIES
            or self.language == "javascript"
        )
        harness_config = self._build_harness_config(scenario)

        if is_js:
            harness_content = self._build_js_harness(
                scenario.category,
                harness_body=harness_config.get("harness_body", ""),
                behavior=harness_config.get("behavior", ""),
            )
            harness_path, config_path = self._write_harness(
                harness_content, harness_config, scenario.name, ext=".cjs",
            )
            runner = "node"
        else:
            harness_content = self._build_harness(
                scenario.category,
                behavior=harness_config.get("behavior", ""),
            )
            harness_path, config_path = self._write_harness(
                harness_content, harness_config, scenario.name,
            )
            runner = "python"

        # Build command — for Node.js, cap V8 heap to match resource caps
        if runner == "node":
            heap_mb = self.session.resource_caps.memory_mb
            cmd = [
                runner,
                f"--max-old-space-size={heap_mb}",
                str(harness_path),
                str(config_path),
            ]
        else:
            cmd = [runner, str(harness_path), str(config_path)]

        # Run harness in session sandbox
        session_result = self.session.run_in_session(
            cmd,
            timeout=timeout,
        )

        total_ms = (time.perf_counter() - start) * 1000

        # Parse output
        result = self._parse_harness_output(session_result, scenario)
        result.total_execution_time_ms = round(total_ms, 2)

        # Check failure indicators
        result.failure_indicators_triggered = self._check_failure_indicators(
            scenario, result.steps, session_result,
        )

        # Clean up harness files
        for path in (harness_path, config_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        logger.info(
            "Scenario '%s' %s: %d steps, %d errors, %.0f ms",
            scenario.name, result.status, len(result.steps),
            result.total_errors, result.total_execution_time_ms,
        )
        return result

    def _build_harness_config(self, scenario: StressTestScenario) -> dict:
        """Build the configuration dict passed to the harness script."""
        config = scenario.test_config
        behavior = config.get("behavior", "")
        skip_imports = config.get("skip_imports", False)

        # Coupling scenarios are standalone — no user imports needed
        if behavior:
            skip_imports = True

        if skip_imports:
            target_modules: list[str] = []
        else:
            target_modules = self._get_target_modules(scenario)

        harness_config: dict = {
            "category": scenario.category,
            "parameters": config.get("parameters", {}),
            "resource_limits": config.get("resource_limits", {}),
            "measurements": config.get("measurements", []),
            "target_modules": target_modules,
            "synthetic_data": config.get("synthetic_data", {}),
            "scenario_name": scenario.name,
        }

        if skip_imports:
            harness_config["target_functions"] = []
        else:
            # Add function info from ingestion
            target_funcs = []
            for analysis in self.ingestion.file_analyses:
                if self.language == "javascript":
                    mod = analysis.file_path
                else:
                    mod = (
                        analysis.file_path
                        .replace(".py", "")
                        .replace("/", ".")
                        .replace("\\", ".")
                    )
                if mod in target_modules or not target_modules:
                    for func in analysis.functions:
                        if not func.is_method and not func.name.startswith("_"):
                            target_funcs.append({
                                "module": mod,
                                "name": func.name,
                                "args": func.args,
                                "is_async": func.is_async,
                            })
            harness_config["target_functions"] = target_funcs[:50]

        # Pass harness_body override if present
        harness_body = config.get("harness_body", "")
        if harness_body:
            harness_config["harness_body"] = harness_body

        # Pass coupling metadata for standalone coupling bodies
        if behavior:
            harness_config["behavior"] = behavior
            if "coupling_source" in config:
                harness_config["coupling_source"] = config["coupling_source"]
            if "coupling_sources" in config:
                harness_config["coupling_sources"] = config["coupling_sources"]
            if "coupling_targets" in config:
                harness_config["coupling_targets"] = config["coupling_targets"]

        return harness_config

    def _build_harness(self, category: str, behavior: str = "") -> str:
        """Generate a self-contained Python test harness script."""
        if behavior and behavior in _PY_COUPLING_BODIES:
            body = _PY_COUPLING_BODIES[behavior]
        elif behavior and behavior in _PY_LIB_BODIES:
            body = _PY_LIB_BODIES[behavior]
        else:
            body = _CATEGORY_BODIES.get(category, _BODY_GENERIC)
        return _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE

    def _build_js_harness(
        self,
        category: str,
        harness_body: str = "",
        behavior: str = "",
    ) -> str:
        """Generate a self-contained JavaScript test harness script.

        Body selection priority:
        1. behavior → _JS_COUPLING_BODIES (standalone coupling tests)
        2. harness_body → _JS_CATEGORY_BODIES (browser-only node-safe tests)
        3. category → _JS_CATEGORY_BODIES (standard category tests)
        """
        if behavior and behavior in _JS_COUPLING_BODIES:
            body = _JS_COUPLING_BODIES[behavior]
        elif harness_body and harness_body in _JS_CATEGORY_BODIES:
            body = _JS_CATEGORY_BODIES[harness_body]
        else:
            body = _JS_CATEGORY_BODIES.get(category, _JS_BODY_GENERIC)
        return _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE

    def _write_harness(
        self,
        script_content: str,
        harness_config: dict,
        scenario_name: str,
        ext: str = ".py",
    ) -> tuple[Path, Path]:
        """Write harness script and config to the session workspace."""
        safe = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in scenario_name
        )[:60]
        uid = uuid.uuid4().hex[:8]

        harness_path = (
            self.session.project_copy_dir / f"_mycode_harness_{safe}_{uid}{ext}"
        )
        config_path = (
            self.session.project_copy_dir / f"_mycode_config_{safe}_{uid}.json"
        )

        harness_path.write_text(script_content, encoding="utf-8")
        config_path.write_text(json.dumps(harness_config), encoding="utf-8")

        return harness_path, config_path

    def _parse_harness_output(
        self,
        session_result: SessionResult,
        scenario: StressTestScenario,
    ) -> ScenarioResult:
        """Parse harness stdout into a ScenarioResult."""
        # Handle timeout
        if session_result.timed_out:
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_category=scenario.category,
                status="partial",
                resource_cap_hit=True,
                summary="Scenario timed out before completion.",
                steps=[StepResult(
                    step_name="timeout",
                    resource_cap_hit="timeout",
                    error_count=1,
                    errors=[{"type": "Timeout", "message": "Execution timed out"}],
                    stdout_snippet=session_result.stdout[-_SNIPPET_MAX:],
                    stderr_snippet=session_result.stderr[-_SNIPPET_MAX:],
                )],
                total_errors=1,
            )

        stdout = session_result.stdout
        stderr = session_result.stderr

        # Extract JSON between markers
        start_idx = stdout.find(_RESULTS_START)
        end_idx = stdout.find(_RESULTS_END)

        if start_idx == -1 or end_idx == -1:
            is_crash = session_result.returncode != 0
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_category=scenario.category,
                status="failed" if is_crash else "completed",
                summary=(
                    f"Harness did not produce structured output. "
                    f"Exit code: {session_result.returncode}."
                ),
                steps=[StepResult(
                    step_name="harness_crash",
                    error_count=1 if is_crash else 0,
                    errors=[{
                        "type": "HarnessCrash",
                        "message": stderr[-500:] if stderr else "No error output",
                    }] if is_crash else [],
                    stdout_snippet=stdout[-_SNIPPET_MAX:],
                    stderr_snippet=stderr[-_SNIPPET_MAX:],
                )],
                total_errors=1 if is_crash else 0,
            )

        # Parse JSON
        json_str = stdout[start_idx + len(_RESULTS_START):end_idx].strip()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_category=scenario.category,
                status="failed",
                summary=f"Failed to parse harness output: {e}",
                total_errors=1,
            )

        # Build steps from parsed data
        steps: list[StepResult] = []
        total_errors = 0
        peak_memory = 0.0

        for raw in data.get("steps", []):
            step = StepResult(
                step_name=raw.get("step_name", "unknown"),
                parameters=raw.get("parameters", {}),
                execution_time_ms=raw.get("execution_time_ms", 0.0),
                memory_peak_mb=raw.get("memory_peak_mb", 0.0),
                error_count=raw.get("error_count", 0),
                errors=raw.get("errors", []),
                measurements=raw.get("measurements", {}),
                resource_cap_hit=raw.get("resource_cap_hit", ""),
            )
            steps.append(step)
            total_errors += step.error_count
            peak_memory = max(peak_memory, step.memory_peak_mb)

        # Add import errors as a step if present
        import_errors = data.get("import_errors", [])
        if import_errors:
            steps.insert(0, StepResult(
                step_name="module_import",
                error_count=len(import_errors),
                errors=import_errors,
            ))
            total_errors += len(import_errors)

        # Determine status
        has_cap_hit = any(s.resource_cap_hit for s in steps)
        if has_cap_hit:
            status = "partial"
        elif total_errors > 0 and steps and all(s.error_count > 0 for s in steps):
            status = "failed"
        else:
            status = "completed"

        # Build summary
        parts = []
        if steps:
            parts.append(f"{len(steps)} steps executed")
        if total_errors:
            parts.append(f"{total_errors} errors recorded")
        if peak_memory > 0:
            parts.append(f"peak memory {peak_memory:.1f} MB")
        if has_cap_hit:
            caps = {s.resource_cap_hit for s in steps if s.resource_cap_hit}
            parts.append(f"resource cap hit: {', '.join(sorted(caps))}")

        return ScenarioResult(
            scenario_name=scenario.name,
            scenario_category=scenario.category,
            status=status,
            steps=steps,
            peak_memory_mb=peak_memory,
            total_errors=total_errors,
            resource_cap_hit=has_cap_hit,
            summary=". ".join(parts) + "." if parts else "No results.",
        )

    def _get_target_modules(self, scenario: StressTestScenario) -> list[str]:
        """Determine which user modules to target for this scenario."""
        config = scenario.test_config

        # Prefer explicit target_files from test_config
        target_files = config.get("target_files", [])
        if target_files:
            if self.language == "javascript":
                # Skip .jsx/.tsx — Node.js can't natively require JSX syntax
                return [f for f in target_files if not f.endswith((".jsx", ".tsx"))]
            return [
                f.replace(".py", "").replace("/", ".").replace("\\", ".")
                for f in target_files
            ]

        # Fall back to all analyzed files (excluding tests/private)
        modules = []
        for analysis in self.ingestion.file_analyses:
            if analysis.parse_error:
                continue
            if self.language == "javascript":
                # JS: keep relative file paths as-is for require()
                # Skip .jsx/.tsx — Node.js can't natively require JSX syntax
                fp = analysis.file_path
                if fp.endswith((".jsx", ".tsx")):
                    continue
                base = fp.split("/")[-1].split("\\")[-1]
                if not base.startswith("test") and not base.startswith("_"):
                    modules.append(fp)
            else:
                mod = (
                    analysis.file_path
                    .replace(".py", "")
                    .replace("/", ".")
                    .replace("\\", ".")
                )
                if not mod.startswith("test_") and not mod.startswith("_"):
                    modules.append(mod)

        return modules[:20]

    def _check_failure_indicators(
        self,
        scenario: StressTestScenario,
        steps: list[StepResult],
        session_result: SessionResult,
    ) -> list[str]:
        """Check which failure indicators from the scenario were triggered."""
        indicators = scenario.failure_indicators
        if not indicators:
            return []

        triggered: list[str] = []

        # Collect text from all errors for keyword matching
        all_text = (session_result.stderr + " " + session_result.stdout).lower()
        for step in steps:
            for error in step.errors:
                all_text += " " + str(error.get("type", "")).lower()
                all_text += " " + str(error.get("message", "")).lower()

        any_timeout = (
            session_result.timed_out
            or any(s.resource_cap_hit == "timeout" for s in steps)
        )
        any_memory_cap = any(s.resource_cap_hit == "memory" for s in steps)
        any_crash = session_result.returncode not in (0, -1)
        total_errors = sum(s.error_count for s in steps)

        # Detect monotonic memory growth
        memory_values = [s.memory_peak_mb for s in steps if s.memory_peak_mb > 0]
        memory_growing = False
        if len(memory_values) >= 3:
            increases = sum(
                1 for i in range(1, len(memory_values))
                if memory_values[i] > memory_values[i - 1]
            )
            memory_growing = increases >= len(memory_values) * 0.7

        for indicator in indicators:
            ind = indicator.lower()
            if ind == "timeout" and any_timeout:
                triggered.append(indicator)
            elif ind == "crash" and any_crash:
                triggered.append(indicator)
            elif ind in ("memory", "oom", "memory_error") and any_memory_cap:
                triggered.append(indicator)
            elif ind in ("memory_growth_unbounded", "memory_leak") and memory_growing:
                triggered.append(indicator)
            elif ind in ("error", "errors") and total_errors > 0:
                triggered.append(indicator)
            elif ind in ("error_propagation", "cascade_failure"):
                # More than half of steps have errors
                if steps and total_errors > len(steps) * 0.5:
                    triggered.append(indicator)
            elif ind in all_text:
                triggered.append(indicator)

        return triggered


# ── Harness Templates ──
#
# The harness is a self-contained Python script that runs inside the
# session's virtual environment.  It uses ONLY the standard library.
#
# Structure:  _HARNESS_PREAMBLE  +  category body  +  _HARNESS_POSTAMBLE
#
# Configuration is passed via a JSON file (sys.argv[1]).

_HARNESS_PREAMBLE = '''\
#!/usr/bin/env python
"""Auto-generated myCode stress test harness."""
import sys
import json
import os
import time
import traceback
import tracemalloc
import importlib

CONFIG = json.loads(open(sys.argv[1]).read())

tracemalloc.start()
sys.path.insert(0, os.getcwd())

_steps = []
_step_errors = []
_import_errors = []
_modules = {}
_callables = []

# ── Module Import ──
for _mod_name in CONFIG.get("target_modules", []):
    try:
        _mod = importlib.import_module(_mod_name)
        _modules[_mod_name] = _mod
    except Exception as _e:
        _import_errors.append({
            "type": type(_e).__name__,
            "message": str(_e)[:500],
            "module": _mod_name,
        })

# ── Callable Discovery ──
for _fi in CONFIG.get("target_functions", []):
    _mn = _fi["module"]
    _fn = _fi["name"]
    if _mn in _modules:
        _attr = getattr(_modules[_mn], _fn, None)
        if callable(_attr):
            _callables.append({
                "name": _mn + "." + _fn,
                "func": _attr,
                "args": _fi.get("args", []),
                "is_async": _fi.get("is_async", False),
            })

# ── Helpers ──

def _record_error(exc, target=""):
    """Record an error within the current step."""
    _step_errors.append({
        "type": type(exc).__name__,
        "message": str(exc)[:500],
        "target": target,
    })

def _measure_step(step_name, params, func):
    """Execute func and record resource measurements."""
    global _step_errors
    _step_errors = []
    tracemalloc.reset_peak()
    _t0 = time.perf_counter()
    _cap = ""
    try:
        func()
    except MemoryError:
        _cap = "memory"
        _step_errors.append({"type": "MemoryError", "message": "Memory limit exceeded"})
    except Exception as _e:
        _step_errors.append({
            "type": type(_e).__name__,
            "message": str(_e)[:500],
            "traceback": traceback.format_exc()[-1000:],
        })
    _elapsed = (time.perf_counter() - _t0) * 1000
    _, _peak = tracemalloc.get_traced_memory()
    _steps.append({
        "step_name": step_name,
        "parameters": params,
        "execution_time_ms": round(_elapsed, 2),
        "memory_peak_mb": round(_peak / 1048576, 2),
        "error_count": len(_step_errors),
        "errors": list(_step_errors),
        "resource_cap_hit": _cap,
    })

def _call_safely(entry, args=None):
    """Call a discovered function with optional arguments."""
    _f = entry["func"]
    _params = [p for p in entry.get("args", []) if p != "self"]
    _is_async = entry.get("is_async", False)
    if _is_async:
        import asyncio
        _loop = asyncio.new_event_loop()
        try:
            if args is not None:
                _coro = _f(*args) if isinstance(args, (list, tuple)) else _f(args)
            elif not _params:
                _coro = _f()
            else:
                _coro = _f(*([None] * len(_params)))
            _loop.run_until_complete(_coro)
        finally:
            _loop.close()
        return
    if args is not None:
        if isinstance(args, (list, tuple)):
            _f(*args)
        else:
            _f(args)
    elif not _params:
        _f()
    else:
        try:
            _f()
        except TypeError:
            _f(*([None] * len(_params)))

# ── Test Body ──
'''

_HARNESS_POSTAMBLE = '''
# ── Output ──
print("__MYCODE_RESULTS_START__")
print(json.dumps({"steps": _steps, "import_errors": _import_errors}))
print("__MYCODE_RESULTS_END__")
'''

# ── Category-Specific Test Bodies ──

_BODY_DATA_VOLUME_SCALING = '''\
_params = CONFIG.get("parameters", {})
_sizes = _params.get("data_sizes", [100, 1000, 10000])
for _sz in _sizes:
    def _run(_s=_sz):
        _data = list(range(_s))
        if _callables:
            for _e in _callables[:5]:
                try:
                    _call_safely(_e, [_data])
                except Exception as _exc:
                    _record_error(_exc, _e["name"])
        else:
            _ = [x * 2 for x in _data]
    _measure_step("data_size_%d" % _sz, {"data_size": _sz}, _run)
'''

_BODY_MEMORY_PROFILING = '''\
_params = CONFIG.get("parameters", {})
_iterations = _params.get("iterations", 50)
_batch = max(1, _iterations // 10)
for _b in range(0, _iterations, _batch):
    _count = min(_batch, _iterations - _b)
    def _run(_n=_count):
        for _i in range(_n):
            if _callables:
                for _e in _callables[:3]:
                    try:
                        _call_safely(_e)
                    except Exception as _exc:
                        _record_error(_exc, _e["name"])
            else:
                for _mn, _m in list(_modules.items()):
                    try:
                        importlib.reload(_m)
                    except Exception:
                        pass
    _measure_step("batch_%d" % _b, {"batch_start": _b, "batch_count": _count}, _run)
'''

_BODY_EDGE_CASE_INPUT = '''\
_edge_cases = [
    ("none", None),
    ("empty_string", ""),
    ("empty_list", []),
    ("empty_dict", {}),
    ("zero", 0),
    ("negative", -1),
    ("large_string", "x" * 100000),
    ("nested_none", [None, None, None]),
    ("mixed_types", [1, "two", 3.0, None, True]),
    ("boolean_false", False),
    ("very_long_list", list(range(50000))),
]
if _callables:
    for _case_name, _case_data in _edge_cases:
        def _run(_d=_case_data):
            for _e in _callables[:5]:
                try:
                    _call_safely(_e, [_d])
                except Exception as _exc:
                    _record_error(_exc, _e["name"])
        _measure_step("edge_%s" % _case_name, {"edge_case": _case_name}, _run)
else:
    _measure_step("edge_no_callables", {}, lambda: None)
'''

_BODY_CONCURRENT_EXECUTION = '''\
import concurrent.futures
_params = CONFIG.get("parameters", {})
_levels = _params.get("concurrent", [1, 5, 10])
for _lvl in _levels:
    def _run(_n=_lvl):
        def _worker():
            if _callables:
                _call_safely(_callables[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n) as _pool:
            _futs = [_pool.submit(_worker) for _ in range(_n)]
            for _f in concurrent.futures.as_completed(_futs, timeout=30):
                try:
                    _f.result()
                except Exception as _exc:
                    _record_error(_exc, "concurrent_worker")
    _measure_step("concurrent_%d" % _lvl, {"concurrent_count": _lvl}, _run)
'''

_BODY_BLOCKING_IO = '''\
import tempfile
_params = CONFIG.get("parameters", {})
_sizes = _params.get("data_sizes", [1000, 10000, 100000])
for _sz in _sizes:
    def _run(_s=_sz):
        _data = "x" * _s
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=True) as _tf:
            _tf.write(_data)
            _tf.flush()
        if _callables:
            for _e in _callables[:3]:
                try:
                    _call_safely(_e)
                except Exception as _exc:
                    _record_error(_exc, _e["name"])
    _measure_step("io_size_%d" % _sz, {"data_size": _sz}, _run)
'''

_BODY_GIL_CONTENTION = '''\
import concurrent.futures
_params = CONFIG.get("parameters", {})
_thread_counts = _params.get("concurrent", [1, 2, 4, 8])
for _tc in _thread_counts:
    def _run(_n=_tc):
        def _cpu_work():
            _total = 0
            for _i in range(100000):
                _total += _i * _i
            if _callables:
                try:
                    _call_safely(_callables[0])
                except Exception as _exc:
                    _record_error(_exc, "gil_worker")
            return _total
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n) as _pool:
            _futs = [_pool.submit(_cpu_work) for _ in range(_n)]
            for _f in concurrent.futures.as_completed(_futs, timeout=30):
                try:
                    _f.result()
                except Exception as _exc:
                    _record_error(_exc, "gil_worker")
    _measure_step("gil_threads_%d" % _tc, {"thread_count": _tc}, _run)
'''

_BODY_GENERIC = '''\
_params = CONFIG.get("parameters", {})
_iterations = _params.get("iterations", 10)
for _i in range(_iterations):
    def _run(_it=_i):
        if _callables:
            for _e in _callables[:5]:
                try:
                    _call_safely(_e)
                except Exception as _exc:
                    _record_error(_exc, _e["name"])
    _measure_step("iteration_%d" % _i, {"iteration": _i}, _run)
'''

# Category -> body mapping
_CATEGORY_BODIES: dict[str, str] = {
    "data_volume_scaling": _BODY_DATA_VOLUME_SCALING,
    "memory_profiling": _BODY_MEMORY_PROFILING,
    "edge_case_input": _BODY_EDGE_CASE_INPUT,
    "concurrent_execution": _BODY_CONCURRENT_EXECUTION,
    "blocking_io": _BODY_BLOCKING_IO,
    "gil_contention": _BODY_GIL_CONTENTION,
}


# ── Python Coupling Test Bodies ──
#
# Standalone bodies for Python coupling scenarios.  They do NOT reference
# _callables or _modules — all operations are synthetic, driven by
# the coupling metadata in CONFIG (coupling_source, coupling_sources,
# coupling_targets, behavior).

_PY_COUPLING_BODY_PURE_COMPUTATION = '''\
_params = CONFIG.get("parameters", {})
_source = CONFIG.get("coupling_source", "")
_sizes = _params.get("data_sizes", [100, 1000, 10000, 100000])

def _workload_for_source(name):
    _lower = name.lower()
    if "json" in _lower:
        return "json"
    if "sort" in _lower or "filter" in _lower or "map" in _lower:
        return "list_ops"
    if "fetch" in _lower or "request" in _lower:
        return "fetch"
    return "generic"

_workload = _workload_for_source(_source)

for _sz in _sizes:
    def _run(_s=_sz, _wl=_workload):
        if _wl == "json":
            import json as _json
            _obj = {("key_%d" % _i): {"value": _i, "label": "item_%d" % _i, "nested": {"a": _i}} for _i in range(_s)}
            _str = _json.dumps(_obj)
            _parsed = _json.loads(_str)
            _ = len(_parsed)
        elif _wl == "list_ops":
            _arr = list(range(_s))
            _mapped = [x * 2 + 1 for x in _arr]
            _filtered = [x for x in _mapped if x % 3 != 0]
            _reduced = sum(_filtered)
            _ = _reduced
        elif _wl == "fetch":
            _responses = [
                {"status": 200, "headers": {"content-type": "application/json"},
                 "body": {"id": _i, "data": "x" * 100}}
                for _i in range(_s)
            ]
            import json as _json
            _total = 0
            for _r in _responses:
                _total += len(_json.dumps(_r["body"]))
            _ = _total
        else:
            _arr = [{"v": _i, "s": str(_i)} for _i in range(_s)]
            _arr.sort(key=lambda x: x["v"])
            _total = sum(item["v"] for item in _arr)
            _ = _total
    _measure_step("compute_%d" % _sz, {"data_size": _sz, "workload": _workload, "source": _source}, _run)
'''

_PY_COUPLING_BODY_STATE_SETTER = '''\
import threading
_params = CONFIG.get("parameters", {})
_sources = CONFIG.get("coupling_sources", [CONFIG.get("coupling_source", "setState")])
_cycle_counts = _params.get("cycle_counts", [10, 100, 1000])

for _cycles in _cycle_counts:
    def _run(_n=_cycles, _setters=_sources):
        _state = {s: 0 for s in _setters}
        _lock = threading.Lock()
        _race_detected = [False]

        def _mutate(setter_name, count):
            for _c in range(count):
                with _lock:
                    _state[setter_name] = _c
                    # Simulate derived state
                    _derived = {k + "_derived": v * 2 for k, v in _state.items()}
                    _ = _derived

        _threads = []
        for _setter in _setters:
            _t = threading.Thread(target=_mutate, args=(_setter, _n))
            _threads.append(_t)
            _t.start()
        for _t in _threads:
            _t.join(timeout=30)

        # Verify final state consistency
        for _setter in _setters:
            if _state[_setter] != _n - 1:
                _record_error(
                    RuntimeError("State inconsistency: %s = %s" % (_setter, _state[_setter])),
                    "state_check",
                )
    _measure_step("state_cycles_%d" % _cycles, {"cycles": _cycles, "setter_count": len(_sources)}, _run)
'''

_PY_COUPLING_BODY_API_CALLER = '''\
import concurrent.futures
_params = CONFIG.get("parameters", {})
_source = CONFIG.get("coupling_source", "fetch")
_concurrency_levels = _params.get("concurrency_levels", [1, 5, 10, 50])

def _mock_api_call(call_id, fail_rate=0.05):
    import random
    _data = {("field_%d" % _i): "value_%d_%d" % (call_id, _i) for _i in range(100)}
    if random.random() < fail_rate:
        raise RuntimeError("API_ERROR_%d" % call_id)
    return {"status": 200, "data": _data, "id": call_id}

for _conc in _concurrency_levels:
    def _run(_n=_conc):
        import json as _json
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n) as _pool:
            _futs = [_pool.submit(_mock_api_call, _i) for _i in range(_n)]
            _success = 0
            _errors = 0
            for _f in concurrent.futures.as_completed(_futs, timeout=30):
                try:
                    _result = _f.result()
                    _json.dumps(_result["data"])
                    _success += 1
                except Exception as _e:
                    _errors += 1
                    _record_error(_e, "api_worker")
            _ = _success
    _measure_step("api_concurrency_%d" % _conc, {"concurrency": _conc, "source": _source}, _run)

# Timeout handling test
def _run_timeout():
    import concurrent.futures
    import time as _time
    _timeout_ms = 100

    def _slow_call():
        _time.sleep((_timeout_ms + 50) / 1000.0)
        return {"status": 200}

    _results = []
    for _i in range(10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            _fut = _pool.submit(_slow_call)
            try:
                _fut.result(timeout=_timeout_ms / 1000.0)
                _results.append("ok")
            except concurrent.futures.TimeoutError:
                _results.append("timeout")
    _ = len(_results)

_measure_step("api_timeout_handling", {"source": _source}, _run_timeout)
'''

_PY_COUPLING_BODY_DOM_RENDER = '''\
_params = CONFIG.get("parameters", {})
_source = CONFIG.get("coupling_source", "render")
_node_counts = _params.get("node_counts", [10, 100, 500, 1000])

def _create_vnode(vtype, props=None, children=None):
    import random
    return {"type": vtype, "props": props or {}, "children": children or [], "_key": random.random()}

def _build_tree(depth, breadth):
    if depth <= 0:
        return _create_vnode("span", {"text": "leaf"})
    _children = []
    for _i in range(breadth):
        _children.append(_build_tree(depth - 1, max(1, breadth - 1)))
    return _create_vnode("div", {"className": "node_%d" % depth}, _children)

def _count_nodes(tree):
    _count = 1
    for _child in tree.get("children", []):
        _count += _count_nodes(_child)
    return _count

def _diff_trees(old_tree, new_tree):
    _patches = 0
    if old_tree["type"] != new_tree["type"]:
        return 1
    for _k in new_tree["props"]:
        if old_tree["props"].get(_k) != new_tree["props"][_k]:
            _patches += 1
    _max_ch = max(len(old_tree["children"]), len(new_tree["children"]))
    for _i in range(_max_ch):
        if _i >= len(old_tree["children"]) or _i >= len(new_tree["children"]):
            _patches += 1
        else:
            _patches += _diff_trees(old_tree["children"][_i], new_tree["children"][_i])
    return _patches

for _node_count in _node_counts:
    _depth = 2
    _breadth = 2
    while _breadth ** _depth < _node_count and _depth < 8:
        _breadth += 1
        if _breadth > 10:
            _breadth = 3
            _depth += 1

    def _run(_d=_depth, _b=_breadth):
        _tree1 = _build_tree(_d, _b)
        _tree2 = _build_tree(_d, _b)
        _patch_count = _diff_trees(_tree1, _tree2)
        _ = _patch_count
    _measure_step("render_nodes_%d" % _node_count, {"target_nodes": _node_count, "source": _source}, _run)

# Memory growth across repeated render cycles
_render_cycles = _params.get("render_cycles", 50)
def _run_memory():
    _retained = []
    for _c in range(_render_cycles):
        _tree = _build_tree(3, 4)
        _retained.append(_tree)
        if len(_retained) > 20:
            _retained.pop(0)
    _ = len(_retained)
_measure_step("render_memory_growth", {"cycles": _render_cycles, "source": _source}, _run_memory)
'''

_PY_COUPLING_BODY_ERROR_HANDLER = '''\
_params = CONFIG.get("parameters", {})
_source = CONFIG.get("coupling_source", "handle_error")
_batch_sizes = _params.get("batch_sizes", [10, 100, 1000, 5000])

def _generate_errors(count):
    _errors = []
    _types = [
        lambda: TypeError("Cannot read attribute of NoneType"),
        lambda: ValueError("invalid literal for int()"),
        lambda: KeyError("missing_key"),
        lambda: RuntimeError("unexpected state"),
        lambda: OSError("connection refused"),
        lambda: TimeoutError("operation timed out"),
        lambda: IndexError("list index out of range"),
        lambda: AttributeError("object has no attribute"),
        lambda: ZeroDivisionError("division by zero"),
    ]
    for _i in range(count):
        _errors.append(_types[_i % len(_types)]())
    return _errors

def _handle_error(error):
    _type = type(error).__name__
    _msg = str(error)[:200]
    _severity = "unknown"
    if _type in ("TypeError", "AttributeError"):
        _severity = "bug"
    elif _type in ("MemoryError", "RecursionError"):
        _severity = "resource"
    elif _type in ("OSError", "TimeoutError"):
        _severity = "transient"
    elif _type in ("ValueError", "KeyError", "IndexError"):
        _severity = "input"
    _action = "retry" if _severity == "transient" else "report"
    return {"type": _type, "severity": _severity, "action": _action}

for _batch_size in _batch_sizes:
    def _run(_n=_batch_size):
        _errors = _generate_errors(_n)
        _retry_count = 0
        _report_count = 0
        for _err in _errors:
            _result = _handle_error(_err)
            if _result["action"] == "retry":
                _retry_count += 1
            else:
                _report_count += 1
        _ = _retry_count
        _ = _report_count
    _measure_step("error_flood_%d" % _batch_size, {"batch_size": _batch_size, "source": _source}, _run)
'''

# Coupling behavior -> body mapping (separate from category bodies)
_PY_COUPLING_BODIES: dict[str, str] = {
    "pure_computation": _PY_COUPLING_BODY_PURE_COMPUTATION,
    "state_setter": _PY_COUPLING_BODY_STATE_SETTER,
    "api_caller": _PY_COUPLING_BODY_API_CALLER,
    "dom_render": _PY_COUPLING_BODY_DOM_RENDER,
    "error_handler": _PY_COUPLING_BODY_ERROR_HANDLER,
}


# ── Python Library-Specific Standalone Bodies ──
#
# Standalone bodies for server framework library scenarios.  They do NOT
# reference _callables or _modules — all operations are synthetic, stdlib-only,
# exercising the same computational patterns the framework uses internally.
# This avoids importing user code that would start a blocking server.

_PY_LIB_BODY_FLASK_SERVER_STRESS = '''\
import concurrent.futures
import json as _json
import hashlib
_params = CONFIG.get("parameters", {})

# 1. Request parsing + JSON body processing at scaling data sizes
_payload_sizes = _params.get("payload_sizes_kb", [1, 10, 100, 1000])
for _sz in _payload_sizes:
    def _run(_s=_sz):
        _body = {("field_%d" % _i): {"value": _i, "label": "item_%d" % _i} for _i in range(_s * 10)}
        _encoded = _json.dumps(_body).encode("utf-8")
        _parsed = _json.loads(_encoded)
        # Simulate response serialization
        _response = _json.dumps({"status": "ok", "data": _parsed}).encode("utf-8")
        _ = len(_response)
    _measure_step("wsgi_payload_%dkb" % _sz, {"payload_size_kb": _sz}, _run)

# 2. Concurrent request handler simulation
_concurrency_levels = _params.get("concurrent", [1, 5, 10, 50])
for _conc in _concurrency_levels:
    def _run(_n=_conc):
        def _handle_request(req_id):
            _body = {("key_%d" % _i): "val_%d" % _i for _i in range(50)}
            _resp = _json.dumps(_body)
            _hash = hashlib.sha256(_resp.encode()).hexdigest()
            return {"id": req_id, "hash": _hash}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n) as _pool:
            _futs = [_pool.submit(_handle_request, _i) for _i in range(_n)]
            for _f in concurrent.futures.as_completed(_futs, timeout=30):
                try:
                    _f.result()
                except Exception as _exc:
                    _record_error(_exc, "request_handler")
    _measure_step("wsgi_concurrent_%d" % _conc, {"concurrent_requests": _conc}, _run)

# 3. Session state serialization under concurrent access
import threading
_session_cycles = _params.get("session_cycles", [10, 100, 1000])
for _sc in _session_cycles:
    def _run(_n=_sc):
        _session = {}
        _lock = threading.Lock()
        def _session_write(writer_id, count):
            for _c in range(count):
                with _lock:
                    _session["user_%d" % writer_id] = {
                        "counter": _c,
                        "data": _json.dumps({"ts": _c, "payload": "x" * 100}),
                    }
        _threads = []
        for _w in range(4):
            _t = threading.Thread(target=_session_write, args=(_w, _n))
            _threads.append(_t)
            _t.start()
        for _t in _threads:
            _t.join(timeout=30)
    _measure_step("session_writes_%d" % _sc, {"session_cycles": _sc}, _run)
'''

_PY_LIB_BODY_FASTAPI_SERVER_STRESS = '''\
import concurrent.futures
import json as _json
import asyncio
_params = CONFIG.get("parameters", {})

# 1. Pydantic-style nested dict validation at scaling complexity
_field_counts = _params.get("field_counts", [5, 20, 50, 100, 500])
for _fc in _field_counts:
    def _run(_n=_fc):
        # Simulate Pydantic model validation with nested dicts
        _model = {}
        for _i in range(_n):
            _model["field_%d" % _i] = {
                "type": "string" if _i % 3 == 0 else ("integer" if _i % 3 == 1 else "nested"),
                "required": _i % 2 == 0,
                "default": None,
            }
            if _i % 3 == 2:
                _model["field_%d" % _i]["children"] = {
                    ("sub_%d" % _j): {"type": "string", "max_length": 255}
                    for _j in range(min(10, _n // 5 + 1))
                }
        # Validate an input against the model
        _input_data = {("field_%d" % _i): ("value_%d" % _i) for _i in range(_n)}
        _errors = []
        for _key, _spec in _model.items():
            if _spec["required"] and _key not in _input_data:
                _errors.append({"field": _key, "error": "required"})
            if _spec["type"] == "nested" and "children" in _spec:
                for _ck in _spec["children"]:
                    pass  # validate children
        _ = len(_errors)
    _measure_step("validation_fields_%d" % _fc, {"field_count": _fc}, _run)

# 2. Async task simulation (concurrent coroutines via asyncio)
_async_levels = _params.get("concurrent", [10, 50, 100, 500])
for _al in _async_levels:
    def _run(_n=_al):
        _loop = asyncio.new_event_loop()
        async def _handler(_id):
            _data = {("k_%d" % _i): _i for _i in range(100)}
            _resp = _json.dumps(_data)
            return {"id": _id, "size": len(_resp)}
        async def _main():
            _tasks = [_handler(_i) for _i in range(_n)]
            return await asyncio.gather(*_tasks)
        try:
            _results = _loop.run_until_complete(_main())
            _ = len(_results)
        finally:
            _loop.close()
    _measure_step("async_handlers_%d" % _al, {"concurrent_async": _al}, _run)

# 3. Thread pool exhaustion simulation (sync handlers)
_sync_levels = _params.get("sync_concurrent", [10, 20, 40, 80])
for _sl in _sync_levels:
    def _run(_n=_sl):
        import time as _time
        def _sync_handler(handler_id):
            _time.sleep(0.001)  # simulate minimal I/O
            _data = {("field_%d" % _i): "value" for _i in range(50)}
            return _json.dumps(_data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_n, 40)) as _pool:
            _futs = [_pool.submit(_sync_handler, _i) for _i in range(_n)]
            _completed = 0
            for _f in concurrent.futures.as_completed(_futs, timeout=30):
                try:
                    _f.result()
                    _completed += 1
                except Exception as _exc:
                    _record_error(_exc, "sync_handler")
            _ = _completed
    _measure_step("sync_threadpool_%d" % _sl, {"sync_concurrent": _sl}, _run)
'''

_PY_LIB_BODY_STREAMLIT_SERVER_STRESS = '''\
import json as _json
import time as _time
import hashlib
_params = CONFIG.get("parameters", {})

# 1. Full script rerun simulation (repeated computation, no caching vs caching)
_data_sizes = _params.get("data_row_counts", [100, 1000, 10000, 100000])
for _ds in _data_sizes:
    def _run(_n=_ds):
        # Simulate expensive computation run on every rerun (no cache)
        _data = [{"id": _i, "value": _i * 3.14, "label": "row_%d" % _i} for _i in range(_n)]
        _filtered = [r for r in _data if r["value"] > _n * 0.5]
        _sorted_data = sorted(_filtered, key=lambda r: r["value"], reverse=True)
        _serialized = _json.dumps(_sorted_data[:100])
        _ = len(_serialized)
    _measure_step("rerun_rows_%d" % _ds, {"data_rows": _ds}, _run)

# Simulate with caching (memoization dict)
_cache = {}
for _ds in _data_sizes:
    def _run(_n=_ds, _c=_cache):
        _key = "data_%d" % _n
        if _key in _c:
            _result = _c[_key]
        else:
            _data = [{"id": _i, "value": _i * 3.14, "label": "row_%d" % _i} for _i in range(_n)]
            _filtered = [r for r in _data if r["value"] > _n * 0.5]
            _result = sorted(_filtered, key=lambda r: r["value"], reverse=True)
            _c[_key] = _result
        _ = len(_result)
    _measure_step("cached_rows_%d" % _ds, {"data_rows": _ds, "cached": True}, _run)

# 2. Session state accumulation (growing dicts per "rerun")
_rerun_counts = _params.get("rerun_counts", [10, 50, 100, 500])
for _rc in _rerun_counts:
    def _run(_n=_rc):
        _session_state = {}
        for _rerun in range(_n):
            # Each rerun appends to session state (common anti-pattern)
            _session_state["history"] = _session_state.get("history", [])
            _session_state["history"].append({
                "rerun": _rerun,
                "data": list(range(100)),
                "timestamp": _time.perf_counter(),
            })
            # Simulate widget state accumulation
            for _w in range(10):
                _session_state["widget_%d" % _w] = _rerun
        _ = len(_session_state.get("history", []))
    _measure_step("session_reruns_%d" % _rc, {"rerun_count": _rc}, _run)

# 3. Large tabular data serialization at scale
_table_sizes = _params.get("table_sizes_kb", [10, 100, 1000, 10000])
for _ts in _table_sizes:
    def _run(_s=_ts):
        _rows = _s * 10  # ~100 bytes per row
        _table = [
            {"col_a": _i, "col_b": "x" * 50, "col_c": _i * 1.5, "col_d": _i % 2 == 0}
            for _i in range(_rows)
        ]
        _serialized = _json.dumps(_table)
        _hash = hashlib.sha256(_serialized.encode()).hexdigest()
        _ = _hash
    _measure_step("table_serialize_%dkb" % _ts, {"table_size_kb": _ts}, _run)
'''

# Server framework library -> body mapping
_PY_LIB_BODIES: dict[str, str] = {
    "flask_server_stress": _PY_LIB_BODY_FLASK_SERVER_STRESS,
    "fastapi_server_stress": _PY_LIB_BODY_FASTAPI_SERVER_STRESS,
    "streamlit_server_stress": _PY_LIB_BODY_STREAMLIT_SERVER_STRESS,
}


# ── JavaScript Harness Templates ──
#
# The JS harness is a self-contained Node.js script that runs inside the
# session sandbox.  It uses ONLY Node.js built-in modules.
#
# Structure:  _JS_HARNESS_PREAMBLE  +  category body  +  _JS_HARNESS_POSTAMBLE
#
# Configuration is passed via a JSON file (process.argv[2]).

_JS_HARNESS_PREAMBLE = '''\
#!/usr/bin/env node
"use strict";
const fs = require("fs");
const path = require("path");
const { EventEmitter } = require("events");

const CONFIG = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const _steps = [];
let _stepErrors = [];
const _importErrors = [];
const _modules = {};
const _callables = [];

// ── Module Import ──
for (const _modName of (CONFIG.target_modules || [])) {
    try {
        const _modPath = path.resolve(process.cwd(), _modName);
        _modules[_modName] = require(_modPath);
    } catch (_e) {
        _importErrors.push({
            type: _e.constructor.name,
            message: String(_e).slice(0, 500),
            module: _modName,
        });
    }
}

// ── Callable Discovery ──
for (const _fi of (CONFIG.target_functions || [])) {
    const _mn = _fi.module;
    const _fn = _fi.name;
    if (_modules[_mn]) {
        const _attr = _modules[_mn][_fn];
        if (typeof _attr === "function") {
            _callables.push({
                name: _mn + "." + _fn,
                func: _attr,
                args: _fi.args || [],
                is_async: _fi.is_async || false,
            });
        }
    }
}

// ── Helpers ──

function _recordError(exc, target) {
    _stepErrors.push({
        type: (exc && exc.constructor) ? exc.constructor.name : "Error",
        message: String(exc).slice(0, 500),
        target: target || "",
    });
}

async function _measureStep(stepName, params, func) {
    _stepErrors = [];
    const _t0 = performance.now();
    let _cap = "";
    try {
        const _result = func();
        if (_result && typeof _result.then === "function") {
            await _result;
        }
    } catch (_e) {
        if (_e instanceof RangeError && /call stack/i.test(String(_e))) {
            _cap = "memory";
        }
        _stepErrors.push({
            type: (_e && _e.constructor) ? _e.constructor.name : "Error",
            message: String(_e).slice(0, 500),
            traceback: (_e && _e.stack) ? _e.stack.slice(-1000) : "",
        });
    }
    const _elapsed = performance.now() - _t0;
    const _mem = process.memoryUsage();
    const _peakMb = _mem.heapUsed / 1048576;
    _steps.push({
        step_name: stepName,
        parameters: params,
        execution_time_ms: Math.round(_elapsed * 100) / 100,
        memory_peak_mb: Math.round(_peakMb * 100) / 100,
        error_count: _stepErrors.length,
        errors: [..._stepErrors],
        resource_cap_hit: _cap,
    });
}

function _callSafely(entry, args) {
    const _f = entry.func;
    if (args !== undefined) {
        if (Array.isArray(args)) return _f(...args);
        return _f(args);
    }
    const _params = (entry.args || []).filter(function(p) { return p !== "this"; });
    if (_params.length === 0) return _f();
    try {
        return _f();
    } catch (_e) {
        return _f(...new Array(_params.length).fill(null));
    }
}

// ── Test Body ──
(async () => {
'''

_JS_HARNESS_POSTAMBLE = '''
})().then(() => {
    console.log("__MYCODE_RESULTS_START__");
    console.log(JSON.stringify({steps: _steps, import_errors: _importErrors}));
    console.log("__MYCODE_RESULTS_END__");
}).catch((_fatalErr) => {
    console.error("Harness error:", _fatalErr);
    console.log("__MYCODE_RESULTS_START__");
    console.log(JSON.stringify({steps: _steps, import_errors: _importErrors}));
    console.log("__MYCODE_RESULTS_END__");
    process.exitCode = 1;
});
'''

# ── JS Category-Specific Test Bodies ──

_JS_BODY_ASYNC_FAILURES = '''\
const _params = CONFIG.parameters || {};
const _levels = _params.concurrent || [10, 50, 100];
for (const _lvl of _levels) {
    await _measureStep("async_load_" + _lvl, {concurrent_promises: _lvl}, () => {
        const _promises = [];
        for (let _i = 0; _i < _lvl; _i++) {
            if (_callables.length > 0) {
                _promises.push(
                    Promise.resolve()
                        .then(() => _callSafely(_callables[_i % _callables.length]))
                        .catch((_e) => _recordError(_e, _callables[_i % _callables.length].name))
                );
            } else {
                _promises.push(
                    new Promise((resolve, reject) => {
                        setTimeout(() => {
                            if (_i % 7 === 0) reject(new Error("async_failure_" + _i));
                            else resolve(_i);
                        }, 0);
                    }).catch((_e) => _recordError(_e, "promise_" + _i))
                );
            }
        }
        return Promise.allSettled(_promises);
    });
}
const _chainLengths = _params.chain_lengths || [5, 10, 20];
for (const _depth of _chainLengths) {
    await _measureStep("rejection_chain_" + _depth, {chain_depth: _depth}, () => {
        let _chain = Promise.resolve();
        for (let _i = 0; _i < _depth; _i++) {
            _chain = _chain.then(() => {
                if (_callables.length > 0) return _callSafely(_callables[0]);
                return _i;
            });
        }
        _chain = _chain.then(() => { throw new Error("end_of_chain_rejection"); });
        return _chain.catch((_e) => _recordError(_e, "rejection_chain"));
    });
}
'''

_JS_BODY_EVENT_LISTENER_ACCUMULATION = '''\
const _params = CONFIG.parameters || {};
const _listenerCounts = _params.listener_counts || [10, 100, 500, 1000];
for (const _count of _listenerCounts) {
    await _measureStep("listeners_" + _count, {listener_count: _count}, () => {
        const _emitter = new EventEmitter();
        _emitter.setMaxListeners(0);
        for (let _i = 0; _i < _count; _i++) {
            _emitter.on("test_event", () => {});
        }
        for (let _j = 0; _j < 10; _j++) {
            _emitter.emit("test_event", {iteration: _j});
        }
        if (_callables.length > 0) {
            for (const _e of _callables.slice(0, 3)) {
                try { _callSafely(_e); } catch (_exc) { _recordError(_exc, _e.name); }
            }
        }
    });
}
const _iterations = _params.iterations || 50;
const _batch = Math.max(1, Math.floor(_iterations / 10));
for (let _b = 0; _b < _iterations; _b += _batch) {
    const _batchCount = Math.min(_batch, _iterations - _b);
    await _measureStep("leak_batch_" + _b, {batch_start: _b, batch_count: _batchCount}, () => {
        const _emitter = new EventEmitter();
        _emitter.setMaxListeners(0);
        for (let _i = 0; _i < _batchCount * 100; _i++) {
            _emitter.on("data", () => {});
        }
        for (let _j = 0; _j < _batchCount; _j++) {
            _emitter.emit("data", {value: _j});
        }
    });
}
'''

_JS_BODY_STATE_MANAGEMENT_DEGRADATION = '''\
const _params = CONFIG.parameters || {};
const _iterations = _params.iterations || 50;
const _batch = Math.max(1, Math.floor(_iterations / 10));
for (let _b = 0; _b < _iterations; _b += _batch) {
    const _count = Math.min(_batch, _iterations - _b);
    await _measureStep("state_batch_" + _b, {batch_start: _b, batch_count: _count}, () => {
        const _store = {};
        for (let _i = 0; _i < _count * 1000; _i++) {
            _store["key_" + (_b * 1000 + _i)] = {
                value: "x".repeat(100),
                timestamp: Date.now(),
                nested: {a: _i, b: [_i, _i + 1, _i + 2]},
            };
        }
        if (_callables.length > 0) {
            for (const _e of _callables.slice(0, 3)) {
                try { _callSafely(_e); } catch (_exc) { _recordError(_exc, _e.name); }
            }
        }
        const _keys = Object.keys(_store);
        for (let _j = 0; _j < Math.min(1000, _keys.length); _j++) {
            void _store[_keys[Math.floor(Math.random() * _keys.length)]];
        }
    });
}
const _closureCounts = _params.closure_counts || [100, 500, 1000];
for (const _cc of _closureCounts) {
    await _measureStep("closures_" + _cc, {closure_count: _cc}, () => {
        const _retained = [];
        for (let _i = 0; _i < _cc; _i++) {
            const _largeData = new Array(1000).fill(_i);
            _retained.push(() => _largeData.reduce((a, b) => a + b, 0));
        }
        let _sum = 0;
        for (const _fn of _retained) { _sum += _fn(); }
    });
}
'''

_JS_BODY_GENERIC = '''\
const _params = CONFIG.parameters || {};
const _iterations = _params.iterations || 10;
for (let _i = 0; _i < _iterations; _i++) {
    await _measureStep("iteration_" + _i, {iteration: _i}, () => {
        if (_callables.length > 0) {
            for (const _e of _callables.slice(0, 5)) {
                try { _callSafely(_e); } catch (_exc) { _recordError(_exc, _e.name); }
            }
        }
    });
}
'''

# ── Node.js-Compatible Test Bodies (browser-only deps) ──
#
# These bodies use ONLY Node.js built-ins (no DOM, no canvas, no window).
# They stress-test the computational/data patterns that browser-only
# libraries rely on, without importing the libraries themselves.

_JS_BODY_NODE_DATA_PROCESSING = '''\
const _params = CONFIG.parameters || {};
const _sizes = _params.data_sizes || [1000, 10000, 100000];
for (const _sz of _sizes) {
    await _measureStep("data_size_" + _sz, {data_size: _sz}, () => {
        // Create {x,y} data arrays
        const _data = [];
        for (let _i = 0; _i < _sz; _i++) {
            _data.push({x: _i, y: Math.sin(_i * 0.01) * 100 + Math.random() * 10});
        }
        // Transform
        const _transformed = _data.map(d => ({x: d.x * 2, y: d.y + 10}));
        // Sort by y
        _transformed.sort((a, b) => a.y - b.y);
        // Filter
        const _filtered = _transformed.filter(d => d.y > 0);
        // Aggregate
        let _sum = 0;
        for (const d of _filtered) _sum += d.y;
        // Deep compare (subset)
        const _subset = _data.slice(0, Math.min(100, _data.length));
        JSON.stringify(_subset);
        // Typed array ops
        const _buf = new Float64Array(_sz);
        for (let _i = 0; _i < _sz; _i++) _buf[_i] = _data[_i].y;
    });
}
'''

_JS_BODY_NODE_OBJECT_LIFECYCLE = '''\
const _params = CONFIG.parameters || {};
const _cycles = _params.cycle_count || 100;
const _iterations = _params.iterations || 50;
const _batch = Math.max(1, Math.floor(_iterations / 10));
for (let _b = 0; _b < _iterations; _b += _batch) {
    const _count = Math.min(_batch, _iterations - _b);
    await _measureStep("lifecycle_batch_" + _b, {batch_start: _b, batch_count: _count}, () => {
        for (let _i = 0; _i < _count; _i++) {
            const _objects = [];
            for (let _j = 0; _j < _cycles; _j++) {
                const _obj = {
                    id: _j,
                    data: new Array(100).fill(0).map((_, k) => ({value: k, label: "item_" + k})),
                    metadata: {created: Date.now(), tags: ["a", "b", "c"]},
                    subscriptions: [() => {}, () => {}, () => {}],
                    cleanup: () => { _obj.data = null; _obj.subscriptions = null; },
                };
                _objects.push(_obj);
            }
            // Discard — trigger cleanup callbacks
            for (const _o of _objects) {
                if (_o.cleanup) _o.cleanup();
            }
            _objects.length = 0;
        }
    });
}
'''

_JS_BODY_NODE_RAPID_UPDATES = '''\
const _params = CONFIG.parameters || {};
const _hzLevels = _params.update_hz || [30, 60, 120];
const _duration = _params.duration_ms || 2000;
for (const _hz of _hzLevels) {
    await _measureStep("updates_" + _hz + "hz", {update_hz: _hz, duration_ms: _duration}, () => {
        return new Promise((resolve) => {
            const _interval = Math.max(1, Math.floor(1000 / _hz));
            let _count = 0;
            let _backlog = 0;
            let _lastTime = performance.now();
            const _state = {value: 0, history: []};
            const _maxUpdates = Math.ceil(_hz * _duration / 1000);
            const _timer = setInterval(() => {
                const _now = performance.now();
                const _elapsed = _now - _lastTime;
                if (_elapsed > _interval * 2) _backlog++;
                _lastTime = _now;
                _state.value++;
                _state.history.push({v: _state.value, t: _now});
                if (_state.history.length > 1000) _state.history = _state.history.slice(-500);
                _count++;
                if (_count >= _maxUpdates) {
                    clearInterval(_timer);
                    if (_backlog > _maxUpdates * 0.1) {
                        _recordError(new Error("Update backlog: " + _backlog + "/" + _count), "rapid_updates");
                    }
                    resolve();
                }
            }, _interval);
            // Safety timeout
            setTimeout(() => { clearInterval(_timer); resolve(); }, _duration + 1000);
        });
    });
}
'''

_JS_BODY_NODE_EDGE_CASE_DATA = '''\
const _edgeCases = [
    {name: "null", value: null},
    {name: "undefined", value: undefined},
    {name: "NaN", value: NaN},
    {name: "Infinity", value: Infinity},
    {name: "neg_Infinity", value: -Infinity},
    {name: "empty_string", value: ""},
    {name: "empty_array", value: []},
    {name: "empty_object", value: {}},
    {name: "zero", value: 0},
    {name: "negative_zero", value: -0},
    {name: "false", value: false},
    {name: "very_long_string", value: "x".repeat(100000)},
    {name: "deeply_nested", value: (function() {
        let o = {v: 1}; for (let i = 0; i < 50; i++) o = {child: o}; return o;
    })()},
    {name: "wide_object", value: Object.fromEntries(
        Array.from({length: 1000}, (_, i) => ["key_" + i, i])
    )},
    {name: "mixed_array", value: [1, "two", null, undefined, true, {a: 1}, [1,2]]},
    {name: "negative", value: -1},
    {name: "float_precision", value: 0.1 + 0.2},
    {name: "max_safe_int", value: Number.MAX_SAFE_INTEGER},
    {name: "min_safe_int", value: Number.MIN_SAFE_INTEGER},
    {name: "sparse_array", value: (function() { const a = []; a[100] = 1; return a; })()},
];
for (const _ec of _edgeCases) {
    await _measureStep("edge_" + _ec.name, {edge_case: _ec.name}, () => {
        const _v = _ec.value;
        // Serialization
        try { JSON.stringify(_v); } catch(_e) { _recordError(_e, "stringify_" + _ec.name); }
        // Iteration
        try {
            if (_v && typeof _v === "object") {
                if (Array.isArray(_v)) { for (const _item of _v) { void _item; } }
                else { for (const _k in _v) { void _v[_k]; } }
            }
        } catch(_e) { _recordError(_e, "iterate_" + _ec.name); }
        // Property access
        try {
            if (_v != null) { void _v.toString(); void _v.valueOf(); }
        } catch(_e) { _recordError(_e, "access_" + _ec.name); }
    });
}
'''

_JS_BODY_NODE_TREE_SCALING = '''\
const _params = CONFIG.parameters || {};
const _sizes = _params.tree_sizes || [100, 1000, 10000];
const _depth = _params.depth || 5;
const _childCount = _params.child_count || 3;
function _buildTree(size, maxDepth, children) {
    let _nodeCount = 0;
    function _node(depth) {
        if (_nodeCount >= size || depth >= maxDepth) return null;
        _nodeCount++;
        const _n = {type: "div", props: {id: "n" + _nodeCount, className: "item"}, children: []};
        if (depth < maxDepth) {
            for (let _i = 0; _i < children && _nodeCount < size; _i++) {
                const _child = _node(depth + 1);
                if (_child) _n.children.push(_child);
            }
        }
        return _n;
    }
    return _node(0);
}
function _traverseTree(node) {
    if (!node) return 0;
    let _count = 1;
    if (node.children) {
        for (const _c of node.children) _count += _traverseTree(_c);
    }
    return _count;
}
for (const _sz of _sizes) {
    await _measureStep("tree_size_" + _sz, {tree_size: _sz, depth: _depth, child_count: _childCount}, () => {
        const _tree = _buildTree(_sz, _depth, _childCount);
        const _count = _traverseTree(_tree);
        // Serialize to check memory
        JSON.stringify(_tree);
    });
}
'''

_JS_BODY_NODE_MATH_COMPUTATION = '''\
const _params = CONFIG.parameters || {};
const _opCounts = _params.op_counts || [1000, 10000, 100000];
// 4x4 matrix multiply (pure JS)
function _mat4Multiply(a, b) {
    const _r = new Float64Array(16);
    for (let _i = 0; _i < 4; _i++) {
        for (let _j = 0; _j < 4; _j++) {
            let _sum = 0;
            for (let _k = 0; _k < 4; _k++) _sum += a[_i * 4 + _k] * b[_k * 4 + _j];
            _r[_i * 4 + _j] = _sum;
        }
    }
    return _r;
}
function _vec3Transform(v, m) {
    return [
        v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12],
        v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13],
        v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14],
    ];
}
function _vec3Dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function _vec3Cross(a, b) {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function _vec3Normalize(v) {
    const _len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) || 1;
    return [v[0]/_len, v[1]/_len, v[2]/_len];
}
for (const _ops of _opCounts) {
    await _measureStep("math_ops_" + _ops, {op_count: _ops}, () => {
        let _mat = Float64Array.from([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        const _rot = Float64Array.from([
            Math.cos(0.1), -Math.sin(0.1), 0, 0,
            Math.sin(0.1), Math.cos(0.1), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);
        for (let _i = 0; _i < _ops; _i++) {
            _mat = _mat4Multiply(_mat, _rot);
            const _v = [_i * 0.01, _i * 0.02, _i * 0.03];
            _vec3Transform(_v, _mat);
            _vec3Dot(_v, [1, 0, 0]);
            _vec3Cross(_v, [0, 1, 0]);
            _vec3Normalize(_v);
        }
    });
}
'''

_JS_BODY_NODE_PUBSUB_REACTIVITY = '''\
const _params = CONFIG.parameters || {};
const _subscriberCounts = _params.subscriber_counts || [10, 100, 1000];
const _updateCount = _params.update_count || 100;
for (const _sc of _subscriberCounts) {
    await _measureStep("subscribers_" + _sc, {subscriber_count: _sc, update_count: _updateCount}, () => {
        const _emitter = new EventEmitter();
        _emitter.setMaxListeners(0);
        let _delivered = 0;
        // Subscribe
        for (let _i = 0; _i < _sc; _i++) {
            _emitter.on("update", (data) => { _delivered++; void data; });
        }
        // Publish updates
        for (let _j = 0; _j < _updateCount; _j++) {
            _emitter.emit("update", {value: _j, timestamp: Date.now()});
        }
        const _expected = _sc * _updateCount;
        if (_delivered < _expected * 0.99) {
            _recordError(
                new Error("Missed notifications: " + _delivered + "/" + _expected),
                "pubsub_delivery"
            );
        }
        // Cleanup
        _emitter.removeAllListeners();
    });
}
'''

_JS_BODY_NODE_CLOSURE_MEMORY = '''\
const _params = CONFIG.parameters || {};
const _closureCounts = _params.closure_counts || [100, 500, 1000];
const _scopeSize = _params.scope_size || 1000;
for (const _cc of _closureCounts) {
    await _measureStep("closures_" + _cc, {closure_count: _cc, scope_size: _scopeSize}, () => {
        const _closures = [];
        // useState-like setter closures
        for (let _i = 0; _i < _cc; _i++) {
            const _scopeData = new Array(_scopeSize).fill(_i);
            let _state = {value: _i, data: _scopeData};
            const _setter = (newVal) => { _state = {value: newVal, data: _scopeData}; };
            _closures.push({setter: _setter, state: _state});
        }
        // useEffect-like cleanup closures
        const _cleanups = [];
        for (let _i = 0; _i < _cc; _i++) {
            const _resources = {timer: _i, subscription: new Array(100).fill(_i)};
            _cleanups.push(() => { _resources.timer = null; _resources.subscription = null; });
        }
        // useMemo-like factory closures
        const _memos = [];
        for (let _i = 0; _i < _cc; _i++) {
            const _deps = [_i, _i + 1, _i + 2];
            const _compute = () => _deps.reduce((a, b) => a + b, 0);
            _memos.push({compute: _compute, value: _compute()});
        }
        // Exercise closures
        for (const _c of _closures) _c.setter(_c.state.value + 1);
        for (const _fn of _cleanups) _fn();
        for (const _m of _memos) _m.value = _m.compute();
    });
}
'''

_JS_BODY_NODE_ANIMATION_LOOP = '''\
const _params = CONFIG.parameters || {};
const _frameCount = _params.frame_count || 1000;
const _allocPerFrame = _params.allocations_per_frame || 10;
// Anti-pattern: allocate new objects every frame
await _measureStep("alloc_per_frame", {frame_count: _frameCount, alloc_per_frame: _allocPerFrame}, () => {
    for (let _f = 0; _f < _frameCount; _f++) {
        for (let _a = 0; _a < _allocPerFrame; _a++) {
            // Simulate creating new vectors/matrices per frame (anti-pattern)
            const _v = [Math.random(), Math.random(), Math.random()];
            const _m = new Float64Array(16);
            for (let _i = 0; _i < 16; _i++) _m[_i] = Math.random();
            void _v;
            void _m;
        }
    }
});
// Good pattern: reuse objects
await _measureStep("reuse_objects", {frame_count: _frameCount, alloc_per_frame: _allocPerFrame}, () => {
    const _v = [0, 0, 0];
    const _m = new Float64Array(16);
    for (let _f = 0; _f < _frameCount; _f++) {
        for (let _a = 0; _a < _allocPerFrame; _a++) {
            _v[0] = Math.random(); _v[1] = Math.random(); _v[2] = Math.random();
            for (let _i = 0; _i < 16; _i++) _m[_i] = Math.random();
        }
    }
});
'''

# JS Category -> body mapping
_JS_CATEGORY_BODIES: dict[str, str] = {
    "async_failures": _JS_BODY_ASYNC_FAILURES,
    "event_listener_accumulation": _JS_BODY_EVENT_LISTENER_ACCUMULATION,
    "state_management_degradation": _JS_BODY_STATE_MANAGEMENT_DEGRADATION,
    # Node.js-compatible bodies for browser-only deps (keyed with node_ prefix)
    "node_data_processing": _JS_BODY_NODE_DATA_PROCESSING,
    "node_object_lifecycle": _JS_BODY_NODE_OBJECT_LIFECYCLE,
    "node_rapid_updates": _JS_BODY_NODE_RAPID_UPDATES,
    "node_edge_case_data": _JS_BODY_NODE_EDGE_CASE_DATA,
    "node_tree_scaling": _JS_BODY_NODE_TREE_SCALING,
    "node_math_computation": _JS_BODY_NODE_MATH_COMPUTATION,
    "node_pubsub_reactivity": _JS_BODY_NODE_PUBSUB_REACTIVITY,
    "node_closure_memory": _JS_BODY_NODE_CLOSURE_MEMORY,
    "node_animation_loop": _JS_BODY_NODE_ANIMATION_LOOP,
}

# ── JS Coupling Test Bodies ──
#
# Standalone bodies for coupling scenarios.  They do NOT reference
# _callables or _modules — all operations are synthetic, driven by
# the coupling metadata in CONFIG (coupling_source, coupling_sources,
# coupling_targets, behavior).

_JS_COUPLING_BODY_PURE_COMPUTATION = '''\
const _params = CONFIG.parameters || {};
const _source = CONFIG.coupling_source || "";
const _sizes = _params.data_sizes || [100, 1000, 10000, 100000];

// Pick a workload based on the coupling_source function name
function _workloadForSource(name) {
    const _lower = name.toLowerCase();
    if (_lower.includes("json") && _lower.includes("stringify")) {
        return "json_stringify";
    }
    if (_lower.includes("json") && _lower.includes("parse")) {
        return "json_parse";
    }
    if (_lower.includes("fetch")) {
        return "fetch_like";
    }
    if (_lower.includes("keys") || _lower.includes("values") || _lower.includes("entries")) {
        return "object_enum";
    }
    if (_lower.includes("sort")) {
        return "sort";
    }
    if (_lower.includes("filter") || _lower.includes("map") || _lower.includes("reduce")) {
        return "array_transform";
    }
    return "generic";
}

const _workload = _workloadForSource(_source);

for (const _sz of _sizes) {
    await _measureStep("compute_" + _sz, {data_size: _sz, workload: _workload, source: _source}, () => {
        if (_workload === "json_stringify") {
            const _obj = {};
            for (let _i = 0; _i < _sz; _i++) {
                _obj["key_" + _i] = {value: _i, label: "item_" + _i, nested: {a: _i}};
            }
            const _str = JSON.stringify(_obj);
            void _str.length;
        } else if (_workload === "json_parse") {
            const _obj = {};
            for (let _i = 0; _i < _sz; _i++) {
                _obj["key_" + _i] = {value: _i, label: "item_" + _i};
            }
            const _str = JSON.stringify(_obj);
            const _parsed = JSON.parse(_str);
            void Object.keys(_parsed).length;
        } else if (_workload === "fetch_like") {
            // Simulate request/response data processing at scale
            const _responses = [];
            for (let _i = 0; _i < _sz; _i++) {
                _responses.push({
                    status: 200,
                    headers: {"content-type": "application/json"},
                    body: {id: _i, data: "x".repeat(100)},
                });
            }
            // Process responses: extract, transform, aggregate
            let _totalSize = 0;
            for (const _r of _responses) {
                _totalSize += JSON.stringify(_r.body).length;
            }
            void _totalSize;
        } else if (_workload === "object_enum") {
            const _obj = {};
            for (let _i = 0; _i < _sz; _i++) {
                _obj["prop_" + _i] = _i;
            }
            const _keys = Object.keys(_obj);
            const _vals = Object.values(_obj);
            let _sum = 0;
            for (const _v of _vals) _sum += _v;
            void _keys.length;
        } else if (_workload === "sort") {
            const _arr = [];
            for (let _i = 0; _i < _sz; _i++) {
                _arr.push({id: _i, value: Math.random()});
            }
            _arr.sort((a, b) => a.value - b.value);
        } else if (_workload === "array_transform") {
            const _arr = [];
            for (let _i = 0; _i < _sz; _i++) _arr.push(_i);
            const _mapped = _arr.map(x => x * 2 + 1);
            const _filtered = _mapped.filter(x => x % 3 !== 0);
            const _reduced = _filtered.reduce((acc, x) => acc + x, 0);
            void _reduced;
        } else {
            // Generic computation
            const _arr = [];
            for (let _i = 0; _i < _sz; _i++) _arr.push({v: _i, s: String(_i)});
            _arr.sort((a, b) => a.v - b.v);
            let _sum = 0;
            for (const _item of _arr) _sum += _item.v;
            void _sum;
        }
    });
}
'''

_JS_COUPLING_BODY_STATE_SETTER = '''\
const _params = CONFIG.parameters || {};
const _sources = CONFIG.coupling_sources || [CONFIG.coupling_source || "setState"];
const _cycleCounts = _params.cycle_counts || [10, 100, 1000];

for (const _cycles of _cycleCounts) {
    await _measureStep("state_cycles_" + _cycles, {cycles: _cycles, setter_count: _sources.length}, () => {
        // Shared state object — multiple "setters" mutate concurrently
        const _state = {};
        for (let _i = 0; _i < _sources.length; _i++) {
            _state[_sources[_i]] = 0;
        }

        // Simulate rapid state mutations from multiple sources
        for (let _c = 0; _c < _cycles; _c++) {
            for (const _setter of _sources) {
                // Each setter writes a new value
                _state[_setter] = _c;
                // Simulate derived state computation
                const _derived = {};
                for (const _k of Object.keys(_state)) {
                    _derived[_k + "_derived"] = _state[_k] * 2;
                }
                // Simulate subscriber notification
                const _snapshot = JSON.parse(JSON.stringify(_state));
                void _snapshot;
            }
        }

        // Verify final state consistency
        for (const _setter of _sources) {
            if (_state[_setter] !== _cycles - 1) {
                _recordError(
                    new Error("State inconsistency: " + _setter + " = " + _state[_setter]),
                    "state_check"
                );
            }
        }
    });
}
'''

_JS_COUPLING_BODY_API_CALLER = '''\
const _params = CONFIG.parameters || {};
const _source = CONFIG.coupling_source || "fetch";
const _concurrencyLevels = _params.concurrency_levels || [1, 5, 10, 50];

// Mock async API call with configurable latency
function _mockApiCall(id, failRate) {
    return new Promise((resolve, reject) => {
        // Simulate processing work (not just setTimeout)
        const _data = {};
        for (let _i = 0; _i < 100; _i++) {
            _data["field_" + _i] = "value_" + id + "_" + _i;
        }
        if (Math.random() < failRate) {
            reject(new Error("API_ERROR_" + id));
        } else {
            resolve({status: 200, data: _data, id: id});
        }
    });
}

for (const _conc of _concurrencyLevels) {
    await _measureStep("api_concurrency_" + _conc, {concurrency: _conc, source: _source}, async () => {
        const _promises = [];
        const _failRate = 0.05;  // 5% error rate
        for (let _i = 0; _i < _conc; _i++) {
            _promises.push(
                _mockApiCall(_i, _failRate).catch(_e => ({status: "error", error: String(_e)}))
            );
        }
        const _results = await Promise.all(_promises);

        // Process results — simulate dependents consuming API responses
        let _successCount = 0;
        let _errorCount = 0;
        for (const _r of _results) {
            if (_r.status === "error") {
                _errorCount++;
            } else {
                _successCount++;
                // Simulate dependent processing
                JSON.stringify(_r.data);
            }
        }
        void _successCount;
    });
}

// Timeout handling test
await _measureStep("api_timeout_handling", {source: _source}, async () => {
    const _timeoutMs = 100;
    function _slowCall() {
        return new Promise((resolve) => {
            setTimeout(() => resolve({status: 200}), _timeoutMs + 50);
        });
    }
    function _withTimeout(promise, ms) {
        return Promise.race([
            promise,
            new Promise((_, reject) => setTimeout(() => reject(new Error("TIMEOUT")), ms)),
        ]);
    }

    const _results = [];
    for (let _i = 0; _i < 10; _i++) {
        try {
            const _r = await _withTimeout(_slowCall(), _timeoutMs);
            _results.push({status: "ok"});
        } catch (_e) {
            _results.push({status: "timeout"});
        }
    }
    void _results.length;
});
'''

_JS_COUPLING_BODY_DOM_RENDER = '''\
const _params = CONFIG.parameters || {};
const _source = CONFIG.coupling_source || "render";
const _nodeCounts = _params.node_counts || [10, 100, 500, 1000];

// Virtual DOM node factory — no real DOM
function _createVNode(type, props, children) {
    return {type, props: props || {}, children: children || [], _key: Math.random()};
}

function _buildTree(depth, breadth) {
    if (depth <= 0) {
        return _createVNode("span", {text: "leaf"}, []);
    }
    const _children = [];
    for (let _i = 0; _i < breadth; _i++) {
        _children.push(_buildTree(depth - 1, Math.max(1, breadth - 1)));
    }
    return _createVNode("div", {className: "node_" + depth}, _children);
}

function _countNodes(tree) {
    let _count = 1;
    for (const _child of tree.children) {
        _count += _countNodes(_child);
    }
    return _count;
}

function _diffTrees(oldTree, newTree) {
    let _patches = 0;
    if (oldTree.type !== newTree.type) { _patches++; return _patches; }
    const _oldKeys = Object.keys(oldTree.props);
    const _newKeys = Object.keys(newTree.props);
    for (const _k of _newKeys) {
        if (oldTree.props[_k] !== newTree.props[_k]) _patches++;
    }
    const _maxChildren = Math.max(oldTree.children.length, newTree.children.length);
    for (let _i = 0; _i < _maxChildren; _i++) {
        if (!oldTree.children[_i] || !newTree.children[_i]) {
            _patches++;
        } else {
            _patches += _diffTrees(oldTree.children[_i], newTree.children[_i]);
        }
    }
    return _patches;
}

for (const _nodeCount of _nodeCounts) {
    // Determine tree dimensions to approximate target node count
    let _depth = 2, _breadth = 2;
    while (_breadth ** _depth < _nodeCount && _depth < 8) {
        _breadth++;
        if (_breadth > 10) { _breadth = 3; _depth++; }
    }

    await _measureStep("render_nodes_" + _nodeCount, {target_nodes: _nodeCount, source: _source}, () => {
        // Build initial tree
        const _tree1 = _buildTree(_depth, _breadth);
        const _actualNodes = _countNodes(_tree1);

        // Build updated tree (simulates re-render with changed props)
        const _tree2 = _buildTree(_depth, _breadth);

        // Diff
        const _patchCount = _diffTrees(_tree1, _tree2);
        void _patchCount;
    });
}

// Memory growth across repeated render cycles
const _renderCycles = _params.render_cycles || 50;
await _measureStep("render_memory_growth", {cycles: _renderCycles, source: _source}, () => {
    const _retained = [];
    for (let _c = 0; _c < _renderCycles; _c++) {
        const _tree = _buildTree(3, 4);
        _retained.push(_tree);
        if (_retained.length > 20) {
            _retained.shift();  // Simulate keeping last 20 renders
        }
    }
    void _retained.length;
});
'''

_JS_COUPLING_BODY_ERROR_HANDLER = '''\
const _params = CONFIG.parameters || {};
const _source = CONFIG.coupling_source || "handleError";
const _batchSizes = _params.batch_sizes || [10, 100, 1000, 5000];

// Error generator factory
function _generateErrors(count) {
    const _errors = [];
    const _types = [
        () => new TypeError("Cannot read properties of undefined"),
        () => new RangeError("Maximum call stack size exceeded"),
        () => new SyntaxError("Unexpected token"),
        () => new ReferenceError("x is not defined"),
        () => new Error("NETWORK_ERROR"),
        () => new Error("TIMEOUT"),
        () => { const e = new Error("nested"); e.cause = new Error("root"); return e; },
        () => { try { null.x; } catch(e) { return e; } },
        () => { try { undefined(); } catch(e) { return e; } },
    ];
    for (let _i = 0; _i < count; _i++) {
        _errors.push(_types[_i % _types.length]());
    }
    return _errors;
}

// Simulate error handler: classify, log, decide action
function _handleError(error) {
    const _type = error.constructor.name;
    const _msg = String(error).slice(0, 200);
    const _hasCause = !!(error.cause);
    // Classify
    let _severity = "unknown";
    if (_type === "TypeError" || _type === "ReferenceError") _severity = "bug";
    else if (_type === "RangeError") _severity = "resource";
    else if (_msg.includes("NETWORK") || _msg.includes("TIMEOUT")) _severity = "transient";
    else _severity = "unknown";
    // Decide action
    const _action = _severity === "transient" ? "retry" : "report";
    return {type: _type, severity: _severity, action: _action, hasCause: _hasCause};
}

for (const _batchSize of _batchSizes) {
    await _measureStep("error_flood_" + _batchSize, {batch_size: _batchSize, source: _source}, () => {
        const _errors = _generateErrors(_batchSize);
        let _retryCount = 0;
        let _reportCount = 0;
        for (const _err of _errors) {
            const _result = _handleError(_err);
            if (_result.action === "retry") _retryCount++;
            else _reportCount++;
        }
        void _retryCount;
        void _reportCount;
    });
}
'''

# Coupling behavior -> body mapping (separate from category bodies)
_JS_COUPLING_BODIES: dict[str, str] = {
    "pure_computation": _JS_COUPLING_BODY_PURE_COMPUTATION,
    "state_setter": _JS_COUPLING_BODY_STATE_SETTER,
    "api_caller": _JS_COUPLING_BODY_API_CALLER,
    "dom_render": _JS_COUPLING_BODY_DOM_RENDER,
    "error_handler": _JS_COUPLING_BODY_ERROR_HANDLER,
}
