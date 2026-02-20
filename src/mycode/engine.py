"""Execution Engine (D2) — Runs stress tests inside the Session Manager's sandbox.

Takes StressTestScenario objects from the Scenario Generator, generates and
executes test harness scripts inside the session's virtual environment,
monitors resources, captures errors, and returns structured results for the
Report Generator.

Pure Python. No LLM dependency.
"""

import json
import logging
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

# JavaScript-only categories — skipped until JS execution support is added
_JS_ONLY_CATEGORIES = frozenset({
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
    self-contained Python test harness scripts, executes them inside the
    session's virtual environment, and returns structured results.

    The engine:
    - Generates synthetic data based on scenario configurations
    - Monitors resources: memory, CPU timing, error counts
    - Enforces resource caps: memory ceiling, timeout
    - Records controlled termination when caps are exceeded (finding, not crash)
    - Handles user code crashes gracefully — catches, records, continues

    Usage::

        engine = ExecutionEngine(session, ingestion)
        result = engine.execute(scenarios)
    """

    def __init__(
        self,
        session: SessionManager,
        ingestion: IngestionResult,
    ):
        if not session._setup_complete:
            raise EngineError(
                "Session must be set up before creating ExecutionEngine."
            )
        self.session = session
        self.ingestion = ingestion

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
        logger.info(
            "Executing scenario: %s [%s]", scenario.name, scenario.category,
        )

        # Skip JS-only categories
        if scenario.category in _JS_ONLY_CATEGORIES:
            return ScenarioResult(
                scenario_name=scenario.name,
                scenario_category=scenario.category,
                status="skipped",
                summary=(
                    "JavaScript-specific scenario — "
                    "JS execution not yet supported."
                ),
            )

        config = scenario.test_config
        resource_limits = config.get("resource_limits", {})
        timeout = resource_limits.get(
            "timeout_seconds", self.session.resource_caps.timeout_seconds,
        )

        start = time.perf_counter()

        # Build harness script and config
        harness_config = self._build_harness_config(scenario)
        harness_content = self._build_harness(scenario.category)
        harness_path, config_path = self._write_harness(
            harness_content, harness_config, scenario.name,
        )

        # Run harness in session sandbox
        session_result = self.session.run_in_session(
            ["python", str(harness_path), str(config_path)],
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
        target_modules = self._get_target_modules(scenario)

        harness_config = {
            "category": scenario.category,
            "parameters": config.get("parameters", {}),
            "resource_limits": config.get("resource_limits", {}),
            "measurements": config.get("measurements", []),
            "target_modules": target_modules,
            "synthetic_data": config.get("synthetic_data", {}),
            "scenario_name": scenario.name,
        }

        # Add function info from ingestion
        target_funcs = []
        for analysis in self.ingestion.file_analyses:
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

        return harness_config

    def _build_harness(self, category: str) -> str:
        """Generate a self-contained Python test harness script."""
        body = _CATEGORY_BODIES.get(category, _BODY_GENERIC)
        return _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE

    def _write_harness(
        self,
        script_content: str,
        harness_config: dict,
        scenario_name: str,
    ) -> tuple[Path, Path]:
        """Write harness script and config to the session workspace."""
        safe = "".join(
            c if c.isalnum() or c == "_" else "_"
            for c in scenario_name
        )[:60]
        uid = uuid.uuid4().hex[:8]

        harness_path = (
            self.session.project_copy_dir / f"_mycode_harness_{safe}_{uid}.py"
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
            return [
                f.replace(".py", "").replace("/", ".").replace("\\", ".")
                for f in target_files
            ]

        # Fall back to all analyzed Python files (excluding tests/private)
        modules = []
        for analysis in self.ingestion.file_analyses:
            if analysis.parse_error:
                continue
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
