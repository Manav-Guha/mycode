"""Tests for the Execution Engine (D2)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycode.engine import (
    EngineError,
    ExecutionEngine,
    ExecutionEngineResult,
    ScenarioResult,
    StepResult,
    _CATEGORY_BODIES,
    _BODY_GENERIC,
    _HARNESS_PREAMBLE,
    _HARNESS_POSTAMBLE,
    _JS_CATEGORY_BODIES,
    _JS_COUPLING_BODIES,
    _JS_BODY_GENERIC,
    _JS_HARNESS_PREAMBLE,
    _JS_HARNESS_POSTAMBLE,
    _PY_COUPLING_BODIES,
    _PY_LIB_BODIES,
    _SCENARIO_TIMEOUT_CAP,
    _RESULTS_START,
    _RESULTS_END,
    _validate_harness,
    _validate_python_harness,
    _validate_js_harness,
)
from mycode.ingester import (
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    FunctionInfo,
    IngestionResult,
)
from mycode.scenario import StressTestScenario
from mycode.session import ResourceCaps, SessionResult


# ── Fixtures ──


def _make_session(tmp_path):
    """Create a mock SessionManager."""
    session = MagicMock()
    session._setup_complete = True
    session.project_copy_dir = tmp_path / "project"
    session.project_copy_dir.mkdir(parents=True, exist_ok=True)
    session.resource_caps = ResourceCaps()
    session.run_in_session = MagicMock(return_value=SessionResult(
        returncode=0, stdout="", stderr="",
    ))
    return session


def _make_ingestion(files=None):
    """Create a sample IngestionResult."""
    if files is None:
        files = [
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(
                        name="hello",
                        file_path="app.py",
                        lineno=1,
                        args=[],
                    ),
                    FunctionInfo(
                        name="process_data",
                        file_path="app.py",
                        lineno=10,
                        args=["data", "count"],
                    ),
                    FunctionInfo(
                        name="_private_helper",
                        file_path="app.py",
                        lineno=20,
                        args=["x"],
                    ),
                ],
                classes=[],
                imports=[],
                lines_of_code=30,
            ),
        ]
    return IngestionResult(
        project_path="/fake/project",
        files_analyzed=len(files),
        file_analyses=files,
    )


def _make_scenario(
    name="test_scenario",
    category="data_volume_scaling",
    test_config=None,
    failure_indicators=None,
    priority="medium",
):
    """Create a StressTestScenario for testing."""
    if test_config is None:
        test_config = {
            "parameters": {"data_sizes": [100, 1000]},
            "measurements": ["memory_mb", "execution_time_ms"],
            "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
        }
    return StressTestScenario(
        name=name,
        category=category,
        description=f"Test scenario for {category}",
        target_dependencies=["flask"],
        test_config=test_config,
        expected_behavior="Should complete without errors",
        failure_indicators=failure_indicators or [],
        priority=priority,
        source="offline",
    )


def _make_harness_output(steps=None, import_errors=None):
    """Create simulated harness stdout with JSON result markers."""
    data = {
        "steps": steps or [],
        "import_errors": import_errors or [],
    }
    return (
        "Some debug output\n"
        f"{_RESULTS_START}\n"
        f"{json.dumps(data)}\n"
        f"{_RESULTS_END}\n"
    )


def _make_step_data(
    step_name="step_1",
    execution_time_ms=42.5,
    memory_peak_mb=10.3,
    error_count=0,
    errors=None,
    resource_cap_hit="",
):
    """Create a raw step dict as the harness would output."""
    return {
        "step_name": step_name,
        "parameters": {"data_size": 1000},
        "execution_time_ms": execution_time_ms,
        "memory_peak_mb": memory_peak_mb,
        "error_count": error_count,
        "errors": errors or [],
        "measurements": {},
        "resource_cap_hit": resource_cap_hit,
    }


# ── Data Class Tests ──


class TestDataClasses:
    """Verify data class defaults and field types."""

    def test_test_step_result_defaults(self):
        step = StepResult(step_name="test")
        assert step.step_name == "test"
        assert step.parameters == {}
        assert step.execution_time_ms == 0.0
        assert step.memory_peak_mb == 0.0
        assert step.error_count == 0
        assert step.errors == []
        assert step.measurements == {}
        assert step.resource_cap_hit == ""
        assert step.stdout_snippet == ""
        assert step.stderr_snippet == ""

    def test_scenario_result_defaults(self):
        result = ScenarioResult(
            scenario_name="test",
            scenario_category="data_volume_scaling",
            status="completed",
        )
        assert result.scenario_name == "test"
        assert result.steps == []
        assert result.total_execution_time_ms == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.total_errors == 0
        assert result.failure_indicators_triggered == []
        assert result.resource_cap_hit is False
        assert result.summary == ""

    def test_execution_engine_result_defaults(self):
        result = ExecutionEngineResult()
        assert result.scenario_results == []
        assert result.total_execution_time_ms == 0.0
        assert result.scenarios_completed == 0
        assert result.scenarios_failed == 0
        assert result.scenarios_skipped == 0
        assert result.warnings == []


# ── Engine Init Tests ──


class TestEngineInit:
    """Test ExecutionEngine initialization."""

    def test_requires_setup_session(self, tmp_path):
        session = _make_session(tmp_path)
        session._setup_complete = False
        ingestion = _make_ingestion()

        with pytest.raises(EngineError, match="Session must be set up"):
            ExecutionEngine(session, ingestion)

    def test_accepts_setup_session(self, tmp_path):
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()
        engine = ExecutionEngine(session, ingestion)
        assert engine.session is session
        assert engine.ingestion is ingestion


# ── Execute Tests ──


class TestExecute:
    """Test the execute() orchestration method."""

    def test_empty_scenarios(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        result = engine.execute([])
        assert result.scenarios_completed == 0
        assert result.warnings == ["No scenarios to execute."]

    def test_single_scenario_completed(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[
            _make_step_data("data_size_100"),
            _make_step_data("data_size_1000"),
        ])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine.execute([_make_scenario()])

        assert result.scenarios_completed == 1
        assert result.scenarios_failed == 0
        assert result.scenarios_skipped == 0
        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].status == "completed"
        assert result.total_execution_time_ms > 0

    def test_multiple_scenarios(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        scenarios = [
            _make_scenario(name="scenario_1"),
            _make_scenario(name="scenario_2", category="memory_profiling"),
            _make_scenario(name="scenario_3", category="edge_case_input"),
        ]
        engine = ExecutionEngine(session, _make_ingestion())
        result = engine.execute(scenarios)

        assert len(result.scenario_results) == 3
        assert result.scenarios_completed == 3

    def test_js_scenario_executed_via_node(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        scenario = _make_scenario(name="async_test", category="async_failures")
        result = engine.execute([scenario])

        assert result.scenarios_skipped == 0
        assert result.scenarios_completed == 1
        assert result.scenario_results[0].status == "completed"
        # run_in_session should be called with node + --max-old-space-size
        call_args = session.run_in_session.call_args[0][0]
        assert call_args[0] == "node"
        assert call_args[1].startswith("--max-old-space-size=")
        assert call_args[2].endswith(".cjs")

    def test_engine_error_during_scenario(self, tmp_path):
        session = _make_session(tmp_path)
        session.run_in_session.side_effect = OSError("Disk full")

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine.execute([_make_scenario()])

        assert result.scenarios_failed == 1
        assert result.scenario_results[0].status == "failed"
        assert "Engine error" in result.scenario_results[0].summary
        assert len(result.warnings) == 1

    def test_continues_after_scenario_failure(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first scenario blows up")
            return SessionResult(
                returncode=0, stdout=harness_stdout, stderr="",
            )

        session.run_in_session.side_effect = side_effect

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine.execute([
            _make_scenario(name="fail_scenario"),
            _make_scenario(name="ok_scenario"),
        ])

        assert result.scenarios_failed == 1
        assert result.scenarios_completed == 1
        assert len(result.scenario_results) == 2


# ── Harness Generation Tests ──


class TestHarnessGeneration:
    """Test harness script generation and config building."""

    def test_all_category_bodies_compile(self):
        """Every category body + generic must produce valid Python."""
        bodies = list(_CATEGORY_BODIES.items()) + [("generic", _BODY_GENERIC)]
        for name, body in bodies:
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            try:
                compile(script, f"harness_{name}.py", "exec")
            except SyntaxError as e:
                pytest.fail(f"Harness for '{name}' has syntax error: {e}")

    def test_build_harness_uses_category_body(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness = engine._build_harness("data_volume_scaling")
        assert "data_sizes" in harness
        assert "_RESULTS_START" in harness or "__MYCODE_RESULTS_START__" in harness

    def test_build_harness_falls_back_to_generic(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness = engine._build_harness("unknown_category")
        assert "iteration_" in harness

    def test_build_harness_config_includes_target_functions(self, tmp_path):
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)

        assert "target_modules" in config
        assert "target_functions" in config
        # Should include 'hello' and 'process_data' but not '_private_helper'
        func_names = [f["name"] for f in config["target_functions"]]
        assert "hello" in func_names
        assert "process_data" in func_names
        assert "_private_helper" not in func_names

    def test_build_harness_config_respects_target_files(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        scenario = _make_scenario(test_config={
            "target_files": ["models.py"],
            "parameters": {},
            "resource_limits": {},
        })
        config = engine._build_harness_config(scenario)
        assert config["target_modules"] == ["models"]

    def test_write_harness_creates_files(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness_path, config_path = engine._write_harness(
            "print('hello')",
            {"key": "value"},
            "test_scenario",
        )

        assert harness_path.exists()
        assert config_path.exists()
        assert harness_path.read_text() == "print('hello')"
        assert json.loads(config_path.read_text()) == {"key": "value"}
        assert "_mycode_harness_" in harness_path.name
        assert "_mycode_config_" in config_path.name

    def test_write_harness_safe_names(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness_path, _ = engine._write_harness(
            "pass", {}, "weird/name with spaces & symbols!",
        )
        # Should not have slashes or spaces in filename
        assert "/" not in harness_path.name
        assert " " not in harness_path.name

    def test_all_js_category_bodies_valid_structure(self):
        """Every JS category body + generic produces a complete script with markers."""
        bodies = list(_JS_CATEGORY_BODIES.items()) + [("generic", _JS_BODY_GENERIC)]
        for name, body in bodies:
            script = _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE
            # Must contain the output markers
            assert "__MYCODE_RESULTS_START__" in script, f"JS '{name}' missing start marker"
            assert "__MYCODE_RESULTS_END__" in script, f"JS '{name}' missing end marker"
            # Must have the async IIFE wrapper
            assert "(async () => {" in script, f"JS '{name}' missing async IIFE"
            assert "})().then(" in script, f"JS '{name}' missing .then()"
            # Must not have unclosed braces (basic structural check)
            opens = script.count("{")
            closes = script.count("}")
            assert opens == closes, (
                f"JS '{name}' brace mismatch: {opens} opens vs {closes} closes"
            )

    def test_build_js_harness_uses_category_body(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness = engine._build_js_harness("async_failures")
        assert "async_load_" in harness
        assert "Promise.allSettled" in harness
        assert "__MYCODE_RESULTS_START__" in harness

    def test_build_js_harness_falls_back_to_generic(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness = engine._build_js_harness("unknown_js_category")
        assert "iteration_" in harness

    def test_write_harness_js_extension(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        harness_path, config_path = engine._write_harness(
            "console.log('hello')",
            {"key": "value"},
            "test_js_scenario",
            ext=".cjs",
        )

        assert harness_path.exists()
        assert config_path.exists()
        assert harness_path.name.endswith(".cjs")
        assert "_mycode_harness_" in harness_path.name
        assert harness_path.read_text() == "console.log('hello')"


# ── Output Parsing Tests ──


class TestOutputParsing:
    """Test _parse_harness_output with various session results."""

    def _make_engine(self, tmp_path):
        session = _make_session(tmp_path)
        return ExecutionEngine(session, _make_ingestion())

    def test_parse_successful_output(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(steps=[
            _make_step_data("step_a", execution_time_ms=10.0, memory_peak_mb=5.0),
            _make_step_data("step_b", execution_time_ms=20.0, memory_peak_mb=15.0),
        ])
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "completed"
        assert len(result.steps) == 2
        assert result.steps[0].step_name == "step_a"
        assert result.steps[1].step_name == "step_b"
        assert result.peak_memory_mb == 15.0
        assert result.total_errors == 0
        assert result.resource_cap_hit is False

    def test_parse_with_errors(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(steps=[
            _make_step_data("ok_step"),
            _make_step_data(
                "error_step",
                error_count=2,
                errors=[
                    {"type": "ValueError", "message": "bad input"},
                    {"type": "TypeError", "message": "wrong type"},
                ],
            ),
        ])
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "completed"  # Not all steps have errors
        assert result.total_errors == 2

    def test_parse_all_steps_errored(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(steps=[
            _make_step_data("s1", error_count=1, errors=[{"type": "E", "message": ""}]),
            _make_step_data("s2", error_count=1, errors=[{"type": "E", "message": ""}]),
        ])
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "failed"

    def test_parse_timeout(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        session_result = SessionResult(
            returncode=-1, stdout="partial output", stderr="", timed_out=True,
        )

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "partial"
        assert result.resource_cap_hit is True
        assert result.total_errors == 1
        assert result.steps[0].resource_cap_hit == "timeout"

    def test_parse_crash_no_markers(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        session_result = SessionResult(
            returncode=1, stdout="", stderr="Traceback: SomeError",
        )

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "failed"
        assert result.total_errors == 1
        assert result.steps[0].step_name == "harness_crash"
        assert "SomeError" in result.steps[0].errors[0]["message"]

    def test_parse_exit_zero_no_markers(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        session_result = SessionResult(
            returncode=0, stdout="no markers here", stderr="",
        )

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "completed"
        assert result.total_errors == 0

    def test_parse_invalid_json(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = f"{_RESULTS_START}\nnot json at all\n{_RESULTS_END}\n"
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "failed"
        assert "parse" in result.summary.lower()

    def test_parse_with_import_errors(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(
            steps=[_make_step_data("step_1")],
            import_errors=[{"type": "ModuleNotFoundError", "message": "No module named 'flask'", "module": "app"}],
        )
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        # Import errors inserted as first step
        assert result.steps[0].step_name == "module_import"
        assert result.steps[0].error_count == 1
        assert result.total_errors == 1
        assert len(result.steps) == 2  # import_step + step_1

    def test_parse_resource_cap_hit(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(steps=[
            _make_step_data("step_1"),
            _make_step_data("step_2", resource_cap_hit="memory", error_count=1,
                            errors=[{"type": "MemoryError", "message": "limit"}]),
        ])
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert result.status == "partial"
        assert result.resource_cap_hit is True
        assert "memory" in result.summary.lower()

    def test_parse_summary_content(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario()
        stdout = _make_harness_output(steps=[
            _make_step_data("s1", memory_peak_mb=25.0),
            _make_step_data("s2", memory_peak_mb=50.0, error_count=1,
                            errors=[{"type": "E", "message": "err"}]),
        ])
        session_result = SessionResult(returncode=0, stdout=stdout, stderr="")

        result = engine._parse_harness_output(session_result, scenario)

        assert "2 steps executed" in result.summary
        assert "1 errors recorded" in result.summary
        assert "50.0 MB" in result.summary


# ── Target Module Discovery Tests ──


class TestTargetModules:
    """Test _get_target_modules logic."""

    def test_uses_target_files_from_config(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        scenario = _make_scenario(test_config={
            "target_files": ["app.py", "models/user.py"],
            "parameters": {},
            "resource_limits": {},
        })
        modules = engine._get_target_modules(scenario)
        assert modules == ["app", "models.user"]

    def test_discovers_from_ingestion(self, tmp_path):
        session = _make_session(tmp_path)
        ingestion = _make_ingestion(files=[
            FileAnalysis(file_path="app.py", lines_of_code=10),
            FileAnalysis(file_path="utils.py", lines_of_code=20),
            FileAnalysis(file_path="test_app.py", lines_of_code=5),
            FileAnalysis(file_path="_internal.py", lines_of_code=5),
            FileAnalysis(file_path="broken.py", parse_error="syntax error"),
        ])
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario(test_config={"parameters": {}, "resource_limits": {}})

        modules = engine._get_target_modules(scenario)

        assert "app" in modules
        assert "utils" in modules
        assert "test_app" not in modules
        assert "_internal" not in modules
        assert "broken" not in modules

    def test_limits_to_20_modules(self, tmp_path):
        session = _make_session(tmp_path)
        files = [
            FileAnalysis(file_path=f"mod_{i}.py", lines_of_code=10)
            for i in range(30)
        ]
        ingestion = _make_ingestion(files=files)
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario(test_config={"parameters": {}, "resource_limits": {}})

        modules = engine._get_target_modules(scenario)
        assert len(modules) == 20


# ── Failure Indicator Tests ──


class TestFailureIndicators:
    """Test _check_failure_indicators detection."""

    def _make_engine(self, tmp_path):
        session = _make_session(tmp_path)
        return ExecutionEngine(session, _make_ingestion())

    def test_no_indicators_returns_empty(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=[])
        result = engine._check_failure_indicators(
            scenario, [], SessionResult(returncode=0, stdout="", stderr=""),
        )
        assert result == []

    def test_timeout_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["timeout"])
        session_result = SessionResult(
            returncode=-1, stdout="", stderr="", timed_out=True,
        )
        result = engine._check_failure_indicators(scenario, [], session_result)
        assert "timeout" in result

    def test_crash_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["crash"])
        session_result = SessionResult(returncode=137, stdout="", stderr="Killed")
        result = engine._check_failure_indicators(scenario, [], session_result)
        assert "crash" in result

    def test_memory_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["memory"])
        steps = [StepResult(step_name="s", resource_cap_hit="memory")]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "memory" in result

    def test_error_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["errors"])
        steps = [StepResult(step_name="s", error_count=3)]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "errors" in result

    def test_memory_growth_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["memory_growth_unbounded"])
        # Monotonically increasing memory
        steps = [
            StepResult(step_name=f"s{i}", memory_peak_mb=float(i * 10))
            for i in range(1, 6)
        ]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "memory_growth_unbounded" in result

    def test_memory_growth_not_triggered_flat(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["memory_growth_unbounded"])
        # Flat memory usage
        steps = [
            StepResult(step_name=f"s{i}", memory_peak_mb=10.0)
            for i in range(5)
        ]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "memory_growth_unbounded" not in result

    def test_cascade_failure_indicator(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["cascade_failure"])
        # More than half of steps have errors
        steps = [
            StepResult(step_name="s1", error_count=1),
            StepResult(step_name="s2", error_count=1),
            StepResult(step_name="s3", error_count=0),
        ]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "cascade_failure" in result

    def test_keyword_match_in_stderr(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["deadlock"])
        session_result = SessionResult(
            returncode=0, stdout="", stderr="Warning: potential deadlock detected",
        )
        result = engine._check_failure_indicators(scenario, [], session_result)
        assert "deadlock" in result

    def test_keyword_match_in_error_messages(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["corruption"])
        steps = [StepResult(
            step_name="s",
            error_count=1,
            errors=[{"type": "RuntimeError", "message": "Data corruption detected"}],
        )]
        session_result = SessionResult(returncode=0, stdout="", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert "corruption" in result

    def test_not_triggered_when_absent(self, tmp_path):
        engine = self._make_engine(tmp_path)
        scenario = _make_scenario(failure_indicators=["timeout", "crash", "memory"])
        steps = [StepResult(step_name="s")]
        session_result = SessionResult(returncode=0, stdout="ok", stderr="")
        result = engine._check_failure_indicators(scenario, steps, session_result)
        assert result == []


# ── End-to-End Scenario Execution Tests ──


class TestScenarioExecution:
    """Test _execute_scenario with mocked session."""

    def test_harness_written_and_cleaned(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine._execute_scenario(_make_scenario())

        # Harness and config files should be cleaned up
        project_dir = session.project_copy_dir
        remaining = list(project_dir.glob("_mycode_*"))
        assert remaining == []

    def test_timeout_capped_when_config_exceeds_cap(self, tmp_path):
        """Config timeout of 900 exceeds the category cap → should be capped."""
        from mycode.engine import _CATEGORY_TIMEOUT_CAPS
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        # data_volume_scaling has a 600s cap
        scenario = _make_scenario(test_config={
            "parameters": {},
            "resource_limits": {"timeout_seconds": 900},
        })
        engine = ExecutionEngine(session, _make_ingestion())
        engine._execute_scenario(scenario)

        call_kwargs = session.run_in_session.call_args
        actual = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert actual == _CATEGORY_TIMEOUT_CAPS["data_volume_scaling"]

    def test_timeout_preserved_when_under_cap(self, tmp_path):
        """Config timeout of 15 is under the 30s cap → should be preserved."""
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        scenario = _make_scenario(test_config={
            "parameters": {},
            "resource_limits": {"timeout_seconds": 15},
        })
        engine = ExecutionEngine(session, _make_ingestion())
        engine._execute_scenario(scenario)

        call_kwargs = session.run_in_session.call_args
        actual = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert actual == 15

    def test_default_timeout_from_session_capped(self, tmp_path):
        """Session default timeout of 900 exceeds category cap → should be capped."""
        from mycode.engine import _CATEGORY_TIMEOUT_CAPS
        session = _make_session(tmp_path)
        session.resource_caps = ResourceCaps(timeout_seconds=900)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        # data_volume_scaling category → 600s cap
        scenario = _make_scenario(test_config={"parameters": {}, "resource_limits": {}})
        engine = ExecutionEngine(session, _make_ingestion())
        engine._execute_scenario(scenario)

        call_kwargs = session.run_in_session.call_args
        actual = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert actual == _CATEGORY_TIMEOUT_CAPS["data_volume_scaling"]

    def test_scenario_timeout_cap_value(self):
        """_SCENARIO_TIMEOUT_CAP should be 300 seconds."""
        assert _SCENARIO_TIMEOUT_CAP == 300

    def test_js_category_executed_via_node(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        for cat in ("async_failures", "event_listener_accumulation", "state_management_degradation"):
            session.run_in_session.reset_mock()
            session.run_in_session.return_value = SessionResult(
                returncode=0, stdout=harness_stdout, stderr="",
            )
            scenario = _make_scenario(name=f"js_{cat}", category=cat)
            result = engine._execute_scenario(scenario)
            assert result.status == "completed", f"JS category {cat} should complete"
            # Verify node was used, not python
            call_args = session.run_in_session.call_args[0][0]
            assert call_args[0] == "node", f"JS category {cat} should use node"
            assert call_args[1].startswith("--max-old-space-size="), f"JS category {cat} should set heap limit"
            assert call_args[2].endswith(".cjs"), f"JS category {cat} should write .cjs file"

    def test_all_js_categories_generate_valid_harness(self, tmp_path):
        """Verify that every JS category produces a harness and completes."""
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[
            _make_step_data("async_load_10"),
            _make_step_data("async_load_50"),
        ])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        categories = [
            "async_failures",
            "event_listener_accumulation",
            "state_management_degradation",
        ]

        for cat in categories:
            scenario = _make_scenario(name=f"test_{cat}", category=cat)
            result = engine._execute_scenario(scenario)
            assert result.status == "completed", f"JS category {cat} should complete"

    def test_js_harness_files_cleaned_up(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        scenario = _make_scenario(name="js_async", category="async_failures")
        engine._execute_scenario(scenario)

        # Harness .cjs and config files should be cleaned up
        project_dir = session.project_copy_dir
        remaining = list(project_dir.glob("_mycode_*"))
        assert remaining == []

    def test_all_python_categories_generate_valid_harness(self, tmp_path):
        """Verify that every Python category produces a runnable harness."""
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        categories = [
            "data_volume_scaling",
            "memory_profiling",
            "edge_case_input",
            "concurrent_execution",
            "blocking_io",
            "gil_contention",
        ]

        for cat in categories:
            scenario = _make_scenario(name=f"test_{cat}", category=cat)
            result = engine._execute_scenario(scenario)
            assert result.status == "completed", f"Category {cat} should complete"

    def test_js_language_shared_categories_use_node(self, tmp_path):
        """Shared categories (data_volume_scaling, etc.) should use node for JS projects."""
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")
        shared_categories = [
            "data_volume_scaling", "memory_profiling",
            "edge_case_input", "concurrent_execution",
        ]
        for cat in shared_categories:
            session.run_in_session.reset_mock()
            session.run_in_session.return_value = SessionResult(
                returncode=0, stdout=harness_stdout, stderr="",
            )
            scenario = _make_scenario(name=f"js_{cat}", category=cat)
            engine._execute_scenario(scenario)
            call_args = session.run_in_session.call_args[0][0]
            assert call_args[0] == "node", (
                f"Shared category '{cat}' should use node for JS projects"
            )
            assert call_args[1].startswith("--max-old-space-size="), (
                f"Shared category '{cat}' should set heap limit for JS projects"
            )
            assert call_args[2].endswith(".cjs"), (
                f"Shared category '{cat}' should write .cjs harness for JS projects"
            )

    def test_python_language_shared_categories_use_python(self, tmp_path):
        """Shared categories should still use python for Python projects."""
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        engine = ExecutionEngine(session, _make_ingestion(), language="python")
        scenario = _make_scenario(name="py_scaling", category="data_volume_scaling")
        engine._execute_scenario(scenario)
        call_args = session.run_in_session.call_args[0][0]
        assert call_args[0] == "python", "Shared category should use python for Python projects"

    def test_js_target_modules_preserve_file_paths(self, tmp_path):
        """JS projects should keep file paths as-is, not convert to dot notation."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion(files=[
            FileAnalysis(file_path="src/app.js", lines_of_code=100),
            FileAnalysis(file_path="utils/helpers.js", lines_of_code=50),
        ])
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenario = _make_scenario(test_config={"parameters": {}, "resource_limits": {}})
        modules = engine._get_target_modules(scenario)
        assert "src/app.js" in modules
        assert "utils/helpers.js" in modules

    def test_js_target_modules_excludes_jsx_tsx(self, tmp_path):
        """JSX/TSX files should be excluded — Node.js can't natively require them."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion(files=[
            FileAnalysis(file_path="src/App.jsx", lines_of_code=100),
            FileAnalysis(file_path="src/api.js", lines_of_code=50),
            FileAnalysis(file_path="src/types.tsx", lines_of_code=30),
            FileAnalysis(file_path="src/utils.ts", lines_of_code=40),
        ])
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenario = _make_scenario(test_config={"parameters": {}, "resource_limits": {}})
        modules = engine._get_target_modules(scenario)
        assert "src/api.js" in modules
        assert "src/utils.ts" in modules
        assert "src/App.jsx" not in modules
        assert "src/types.tsx" not in modules

    def test_failure_indicators_attached_to_result(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[
            _make_step_data(
                "step",
                error_count=1,
                errors=[{"type": "RuntimeError", "message": "timeout occurred"}],
            ),
        ])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="something timed out",
        )

        scenario = _make_scenario(failure_indicators=["timeout"])
        engine = ExecutionEngine(session, _make_ingestion())
        result = engine._execute_scenario(scenario)

        assert "timeout" in result.failure_indicators_triggered


# ── Harness Config Structure Tests ──


class TestHarnessConfig:
    """Test that harness configs contain the right data."""

    def test_config_has_all_required_keys(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)

        assert "category" in config
        assert "parameters" in config
        assert "resource_limits" in config
        assert "measurements" in config
        assert "target_modules" in config
        assert "target_functions" in config
        assert "synthetic_data" in config
        assert "scenario_name" in config

    def test_target_functions_have_correct_shape(self, tmp_path):
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)

        for func in config["target_functions"]:
            assert "module" in func
            assert "name" in func
            assert "args" in func
            assert "is_async" in func

    def test_target_functions_limited_to_50(self, tmp_path):
        session = _make_session(tmp_path)
        funcs = [
            FunctionInfo(name=f"func_{i}", file_path="big.py", lineno=i, args=[])
            for i in range(60)
        ]
        ingestion = _make_ingestion(files=[
            FileAnalysis(file_path="big.py", functions=funcs, lines_of_code=100),
        ])
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)
        assert len(config["target_functions"]) == 50

    def test_method_functions_excluded(self, tmp_path):
        session = _make_session(tmp_path)
        ingestion = _make_ingestion(files=[
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(name="standalone", file_path="app.py", lineno=1, args=[]),
                    FunctionInfo(
                        name="method", file_path="app.py", lineno=5,
                        args=["self"], is_method=True,
                    ),
                ],
                lines_of_code=10,
            ),
        ])
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)
        func_names = [f["name"] for f in config["target_functions"]]
        assert "standalone" in func_names
        assert "method" not in func_names

    def test_async_functions_marked(self, tmp_path):
        session = _make_session(tmp_path)
        ingestion = _make_ingestion(files=[
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(
                        name="async_handler", file_path="app.py", lineno=1,
                        args=["request"], is_async=True,
                    ),
                ],
                lines_of_code=10,
            ),
        ])
        engine = ExecutionEngine(session, ingestion)
        scenario = _make_scenario()

        config = engine._build_harness_config(scenario)
        async_func = next(
            f for f in config["target_functions"] if f["name"] == "async_handler"
        )
        assert async_func["is_async"] is True


# ── Edge Case & Robustness Tests ──


class TestEdgeCases:
    """Test edge cases and error resilience."""

    def test_empty_ingestion_result(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        ingestion = IngestionResult(project_path="/fake")
        engine = ExecutionEngine(session, ingestion)

        config = engine._build_harness_config(_make_scenario())
        assert config["target_modules"] == []
        assert config["target_functions"] == []

    def test_scenario_with_empty_test_config(self, tmp_path):
        session = _make_session(tmp_path)
        harness_stdout = _make_harness_output(steps=[_make_step_data()])
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout=harness_stdout, stderr="",
        )

        scenario = _make_scenario(test_config={})
        engine = ExecutionEngine(session, _make_ingestion())
        result = engine.execute([scenario])

        assert result.scenarios_completed == 1

    def test_very_long_stderr_truncated_in_result(self, tmp_path):
        session = _make_session(tmp_path)
        long_stderr = "x" * 10000
        session.run_in_session.return_value = SessionResult(
            returncode=1, stdout="", stderr=long_stderr,
        )

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine._execute_scenario(_make_scenario())

        crash_step = result.steps[0]
        # Error message should be truncated
        assert len(crash_step.errors[0]["message"]) <= 500

    def test_partial_harness_output_before_crash(self, tmp_path):
        """Harness writes start marker but crashes before end marker."""
        session = _make_session(tmp_path)
        stdout = f"debug output\n{_RESULTS_START}\n{{\"steps\": []\n"  # Incomplete
        session.run_in_session.return_value = SessionResult(
            returncode=1, stdout=stdout, stderr="Segmentation fault",
        )

        engine = ExecutionEngine(session, _make_ingestion())
        result = engine._execute_scenario(_make_scenario())

        # No end marker → treated as crash
        assert result.status == "failed"
        assert result.steps[0].step_name == "harness_crash"


# ── JavaScript Harness Template Tests ──


class TestJSHarnessTemplates:
    """Test JS harness template content and structure."""

    def test_async_failures_body_has_promise_testing(self):
        harness = _JS_HARNESS_PREAMBLE + "\n" + _JS_CATEGORY_BODIES["async_failures"] + "\n" + _JS_HARNESS_POSTAMBLE
        assert "Promise.allSettled" in harness
        assert "async_load_" in harness
        assert "rejection_chain_" in harness
        assert "_callSafely" in harness

    def test_event_listener_body_has_emitter_testing(self):
        harness = _JS_HARNESS_PREAMBLE + "\n" + _JS_CATEGORY_BODIES["event_listener_accumulation"] + "\n" + _JS_HARNESS_POSTAMBLE
        assert "EventEmitter" in harness
        assert "setMaxListeners" in harness
        assert "listeners_" in harness
        assert "leak_batch_" in harness

    def test_state_management_body_has_state_testing(self):
        harness = _JS_HARNESS_PREAMBLE + "\n" + _JS_CATEGORY_BODIES["state_management_degradation"] + "\n" + _JS_HARNESS_POSTAMBLE
        assert "state_batch_" in harness
        assert "closures_" in harness
        assert "_store" in harness

    def test_js_generic_body_has_iteration_loop(self):
        harness = _JS_HARNESS_PREAMBLE + "\n" + _JS_BODY_GENERIC + "\n" + _JS_HARNESS_POSTAMBLE
        assert "iteration_" in harness
        assert "_callSafely" in harness

    def test_js_preamble_has_module_loading(self):
        assert "require(" in _JS_HARNESS_PREAMBLE
        assert "target_modules" in _JS_HARNESS_PREAMBLE
        assert "target_functions" in _JS_HARNESS_PREAMBLE
        assert "_recordError" in _JS_HARNESS_PREAMBLE
        assert "_measureStep" in _JS_HARNESS_PREAMBLE

    def test_js_postamble_outputs_json(self):
        assert "__MYCODE_RESULTS_START__" in _JS_HARNESS_POSTAMBLE
        assert "__MYCODE_RESULTS_END__" in _JS_HARNESS_POSTAMBLE
        assert "JSON.stringify" in _JS_HARNESS_POSTAMBLE

    def test_js_category_bodies_mapping_complete(self):
        expected_categories = {"async_failures", "event_listener_accumulation", "state_management_degradation"}
        expected_node = {
            "node_data_processing", "node_object_lifecycle", "node_rapid_updates",
            "node_edge_case_data", "node_tree_scaling", "node_math_computation",
            "node_pubsub_reactivity", "node_closure_memory", "node_animation_loop",
        }
        assert set(_JS_CATEGORY_BODIES.keys()) == expected_categories | expected_node

    def test_js_harness_uses_only_builtins(self):
        """JS harness preamble static imports should only use Node.js built-ins."""
        # Only check top-level require statements (before module import loop)
        preamble_setup = _JS_HARNESS_PREAMBLE.split("// ── Module Import")[0]
        for line in preamble_setup.split("\n"):
            if "require(" in line and not line.strip().startswith("//"):
                assert any(m in line for m in ("fs", "path", "events")), (
                    f"JS preamble requires non-builtin: {line.strip()}"
                )


# ── Browser-Only / skip_imports Tests ──


class TestSkipImports:
    """Test skip_imports and harness_body override for browser-only deps."""

    def test_skip_imports_empty_modules(self, tmp_path):
        """skip_imports → empty target_modules and target_functions."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion())

        scenario = _make_scenario(test_config={
            "skip_imports": True,
            "parameters": {"data_sizes": [100]},
            "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
        })
        config = engine._build_harness_config(scenario)

        assert config["target_modules"] == []
        assert config["target_functions"] == []

    def test_harness_body_override(self, tmp_path):
        """harness_body overrides category-based body selection."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")

        # Build with category "data_volume_scaling" but harness_body "node_data_processing"
        harness_default = engine._build_js_harness("data_volume_scaling")
        harness_override = engine._build_js_harness(
            "data_volume_scaling", harness_body="node_data_processing",
        )

        # Default should use the generic body (data_volume_scaling isn't in JS bodies)
        assert "iteration_" in harness_default  # generic body
        # Override should use node_data_processing body
        assert "Float64Array" in harness_override
        assert "data_size_" in harness_override

    def test_all_node_bodies_registered(self):
        """All 9 node_* keys should be in _JS_CATEGORY_BODIES."""
        expected_node_keys = {
            "node_data_processing",
            "node_object_lifecycle",
            "node_rapid_updates",
            "node_edge_case_data",
            "node_tree_scaling",
            "node_math_computation",
            "node_pubsub_reactivity",
            "node_closure_memory",
            "node_animation_loop",
        }
        actual_node_keys = {
            k for k in _JS_CATEGORY_BODIES if k.startswith("node_")
        }
        assert actual_node_keys == expected_node_keys

    def test_node_harness_produces_valid_structure(self):
        """Every node_* body produces a complete harness with markers and balanced braces."""
        node_bodies = {
            k: v for k, v in _JS_CATEGORY_BODIES.items() if k.startswith("node_")
        }
        for name, body in node_bodies.items():
            script = _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE
            assert "__MYCODE_RESULTS_START__" in script, f"'{name}' missing start marker"
            assert "__MYCODE_RESULTS_END__" in script, f"'{name}' missing end marker"
            assert "(async () => {" in script, f"'{name}' missing async IIFE"
            opens = script.count("{")
            closes = script.count("}")
            assert opens == closes, (
                f"'{name}' brace mismatch: {opens} opens vs {closes} closes"
            )


class TestJSCouplingBodies:
    """Test standalone JS coupling body templates."""

    def test_js_coupling_skips_imports(self, tmp_path):
        """JS scenario with behavior key → empty target_modules/functions."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")

        scenario = _make_scenario(
            name="coupling_compute_fetch",
            category="data_volume_scaling",
            test_config={
                "behavior": "pure_computation",
                "coupling_source": "fetch",
                "coupling_targets": ["processResponse"],
                "coupling_type": "data_flow",
                "measurements": ["memory_mb", "execution_time_ms"],
                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
            },
        )
        config = engine._build_harness_config(scenario)

        assert config["target_modules"] == []
        assert config["target_functions"] == []

    def test_js_coupling_passes_metadata(self, tmp_path):
        """Coupling metadata passed through to harness_config."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")

        scenario = _make_scenario(
            name="coupling_compute_JSON_stringify",
            category="data_volume_scaling",
            test_config={
                "behavior": "pure_computation",
                "coupling_source": "JSON.stringify",
                "coupling_targets": ["sendData", "logOutput"],
                "coupling_type": "data_flow",
                "measurements": ["memory_mb"],
                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
            },
        )
        config = engine._build_harness_config(scenario)

        assert config["behavior"] == "pure_computation"
        assert config["coupling_source"] == "JSON.stringify"
        assert config["coupling_targets"] == ["sendData", "logOutput"]

    def test_js_coupling_passes_grouped_sources(self, tmp_path):
        """State setter scenarios pass coupling_sources (plural)."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")

        scenario = _make_scenario(
            name="coupling_state_setters_group_1",
            category="concurrent_execution",
            test_config={
                "behavior": "state_setter",
                "coupling_sources": ["setCount", "setName", "setItems"],
                "coupling_targets": ["Dashboard", "Sidebar"],
                "coupling_type": "shared_state",
                "measurements": ["memory_mb"],
                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
            },
        )
        config = engine._build_harness_config(scenario)

        assert config["coupling_sources"] == ["setCount", "setName", "setItems"]
        assert config["target_modules"] == []

    def test_python_coupling_skips_imports(self, tmp_path):
        """Python scenario with behavior key skips user module imports."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        scenario = _make_scenario(
            name="coupling_compute_process",
            category="data_volume_scaling",
            test_config={
                "behavior": "pure_computation",
                "coupling_source": "process_data",
                "coupling_targets": ["save_result"],
                "coupling_type": "data_flow",
                "measurements": ["memory_mb"],
                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
            },
        )
        config = engine._build_harness_config(scenario)

        # Python coupling now also skips user module imports
        assert config["target_modules"] == []
        assert config["target_functions"] == []

    def test_js_coupling_body_routing(self, tmp_path):
        """behavior parameter routes to _JS_COUPLING_BODIES."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")

        harness = engine._build_js_harness(
            "data_volume_scaling",
            behavior="pure_computation",
        )
        # Should contain coupling body content, not generic body
        assert "_workloadForSource" in harness
        assert "json_stringify" in harness

    def test_all_coupling_bodies_registered(self):
        """All 5 behavior keys present in _JS_COUPLING_BODIES."""
        expected = {
            "pure_computation",
            "state_setter",
            "api_caller",
            "dom_render",
            "error_handler",
        }
        assert set(_JS_COUPLING_BODIES.keys()) == expected

    def test_coupling_bodies_valid_structure(self):
        """Every coupling body produces a complete harness with balanced braces."""
        for name, body in _JS_COUPLING_BODIES.items():
            script = _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE
            assert "__MYCODE_RESULTS_START__" in script, f"'{name}' missing start marker"
            assert "__MYCODE_RESULTS_END__" in script, f"'{name}' missing end marker"
            opens = script.count("{")
            closes = script.count("}")
            assert opens == closes, (
                f"'{name}' brace mismatch: {opens} opens vs {closes} closes"
            )


class TestNodeAvailabilityCheck:
    """Test Node.js availability check for JS projects."""

    def test_node_unavailable_returns_all_failed(self, tmp_path):
        """When Node.js is not available, all JS scenarios should fail immediately."""
        session = _make_session(tmp_path)
        # Simulate node --version failing
        session.run_in_session.return_value = SessionResult(
            returncode=-1, stdout="", stderr="No such file or directory: 'node'",
        )
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")
        scenarios = [
            StressTestScenario(
                name="react_dom_data_volume",
                category="data_volume_scaling",
                description="Test",
            ),
            StressTestScenario(
                name="coupling_render_App",
                category="state_management_degradation",
                description="Test",
            ),
        ]
        result = engine.execute(scenarios)

        assert len(result.scenario_results) == 2
        for sr in result.scenario_results:
            assert sr.status == "failed"
            assert sr.failure_reason == "dependency_unavailable"
            assert "node.js" in sr.summary.lower()
        assert any("node.js" in w.lower() for w in result.warnings)

    def test_node_available_proceeds_normally(self, tmp_path):
        """When Node.js is available, execution proceeds."""
        session = _make_session(tmp_path)
        # First call: node --version check succeeds
        # Subsequent calls: harness execution
        version_result = SessionResult(
            returncode=0, stdout="v18.19.0\n", stderr="",
        )
        harness_result = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n" + json.dumps({
                "steps": [], "import_errors": [], "probe_skipped": [],
            }) + f"\n{_RESULTS_END}",
            stderr="",
        )
        session.run_in_session.side_effect = [version_result, harness_result]
        engine = ExecutionEngine(session, _make_ingestion(), language="javascript")
        scenarios = [
            StressTestScenario(
                name="test_scenario",
                category="data_volume_scaling",
                description="Test",
                test_config={"behavior": "pure_computation"},
            ),
        ]
        result = engine.execute(scenarios)

        # Should have proceeded past the check
        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].failure_reason != "dependency_unavailable"


class TestPyCouplingBodies:
    """Test standalone Python coupling body templates."""

    def test_python_coupling_body_routing(self, tmp_path):
        """behavior parameter routes to _PY_COUPLING_BODIES."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "data_volume_scaling",
            behavior="pure_computation",
        )
        # Should contain coupling body content, not category body
        assert "_workload_for_source" in harness
        assert "json" in harness

    def test_all_py_coupling_bodies_registered(self):
        """All 6 behavior keys present in _PY_COUPLING_BODIES."""
        expected = {
            "pure_computation",
            "state_setter",
            "api_caller",
            "db_connector",
            "dom_render",
            "error_handler",
        }
        assert set(_PY_COUPLING_BODIES.keys()) == expected

    def test_py_coupling_bodies_compile(self):
        """Every Python coupling body must produce valid Python."""
        for name, body in _PY_COUPLING_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            try:
                compile(script, f"harness_coupling_{name}.py", "exec")
            except SyntaxError as e:
                pytest.fail(f"Python coupling body '{name}' has syntax error: {e}")

    def test_py_coupling_bodies_valid_structure(self):
        """Every Python coupling body produces complete harness with markers."""
        for name, body in _PY_COUPLING_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            assert "__MYCODE_RESULTS_START__" in script, f"'{name}' missing start marker"
            assert "__MYCODE_RESULTS_END__" in script, f"'{name}' missing end marker"
            assert "CONFIG" in body, f"'{name}' does not reference CONFIG"

    def test_py_coupling_body_does_not_use_callables(self):
        """Python coupling bodies must not reference _callables or _modules."""
        for name, body in _PY_COUPLING_BODIES.items():
            assert "_callables" not in body, f"'{name}' references _callables"
            assert "_modules" not in body, f"'{name}' references _modules"
            assert "importlib" not in body, f"'{name}' references importlib"

    def test_behavior_fallback_to_category(self, tmp_path):
        """Unknown behavior falls back to category body."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "data_volume_scaling",
            behavior="nonexistent_behavior",
        )
        # Should fall back to data_volume_scaling body
        assert "data_sizes" in harness


class TestPyLibBodies:
    """Test standalone Python library body templates for server frameworks."""

    def test_flask_server_stress_routing(self, tmp_path):
        """behavior='flask_server_stress' routes to Flask lib body."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "concurrent_execution",
            behavior="flask_server_stress",
        )
        # Should contain Flask body, not concurrent_execution body
        assert "wsgi_payload" in harness
        assert "session_writes" in harness

    def test_fastapi_server_stress_routing(self, tmp_path):
        """behavior='fastapi_server_stress' routes to FastAPI lib body."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "data_volume_scaling",
            behavior="fastapi_server_stress",
        )
        assert "validation_fields" in harness
        assert "async_handlers" in harness

    def test_streamlit_server_stress_routing(self, tmp_path):
        """behavior='streamlit_server_stress' routes to Streamlit lib body."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "memory_profiling",
            behavior="streamlit_server_stress",
        )
        assert "rerun_rows" in harness
        assert "session_reruns" in harness

    def test_unknown_behavior_falls_back_to_category(self, tmp_path):
        """Unknown behavior not in coupling or lib bodies falls back to category."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "concurrent_execution",
            behavior="unknown_server_stress",
        )
        # Should fall back to concurrent_execution body
        assert "ThreadPoolExecutor" in harness
        assert "wsgi_payload" not in harness

    def test_all_lib_bodies_registered(self):
        """All 3 server framework keys present in _PY_LIB_BODIES."""
        expected = {
            "flask_server_stress",
            "fastapi_server_stress",
            "streamlit_server_stress",
        }
        assert set(_PY_LIB_BODIES.keys()) == expected

    def test_lib_bodies_compile(self):
        """Every Python lib body must produce valid Python."""
        for name, body in _PY_LIB_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            try:
                compile(script, f"harness_lib_{name}.py", "exec")
            except SyntaxError as e:
                pytest.fail(f"Python lib body '{name}' has syntax error: {e}")

    def test_lib_bodies_valid_structure(self):
        """Every Python lib body produces complete harness with markers."""
        for name, body in _PY_LIB_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            assert "__MYCODE_RESULTS_START__" in script, f"'{name}' missing start marker"
            assert "__MYCODE_RESULTS_END__" in script, f"'{name}' missing end marker"
            assert "CONFIG" in body, f"'{name}' does not reference CONFIG"

    def test_lib_bodies_do_not_use_callables(self):
        """Python lib bodies must not reference _callables or _modules."""
        for name, body in _PY_LIB_BODIES.items():
            assert "_callables" not in body, f"'{name}' references _callables"
            assert "_modules" not in body, f"'{name}' references _modules"
            assert "importlib" not in body, f"'{name}' references importlib"

    def test_coupling_body_takes_priority(self, tmp_path):
        """A behavior in _PY_COUPLING_BODIES takes priority over _PY_LIB_BODIES."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        harness = engine._build_harness(
            "data_volume_scaling",
            behavior="pure_computation",
        )
        # pure_computation is in _PY_COUPLING_BODIES, should use that
        assert "_workload_for_source" in harness


# ── Harness Syntax Validation Tests ──


class TestHarnessValidation:
    """Tests for pre-execution harness syntax validation."""

    def test_valid_python_passes(self):
        """Valid Python code passes validation."""
        code = "import sys\nprint('hello')\nx = 1 + 2\n"
        assert _validate_python_harness(code) == ""

    def test_invalid_python_detected(self):
        """Python syntax errors are detected."""
        code = "def foo(\n  x = 1\n"  # unclosed paren
        result = _validate_python_harness(code)
        assert result != ""
        assert "syntax error" in result.lower()

    def test_valid_js_passes(self):
        """Valid JS code passes validation."""
        code = "function foo() { return [1, 2, 3]; }\nfoo();\n"
        assert _validate_js_harness(code) == ""

    def test_js_unclosed_brace_detected(self):
        """Unclosed JS brace is detected."""
        code = "function foo() {\n  console.log('hello');\n"
        result = _validate_js_harness(code)
        assert result != ""
        assert "unclosed" in result.lower() or "still open" in result.lower()

    def test_js_mismatched_bracket_detected(self):
        """Mismatched JS brackets are detected."""
        code = "var x = [1, 2, 3};\n"
        result = _validate_js_harness(code)
        assert result != ""
        assert "mismatched" in result.lower() or "unexpected" in result.lower()

    def test_js_strings_not_false_positive(self):
        """Brackets inside JS strings don't cause false positives."""
        code = 'var x = "hello { world }";\nvar y = { a: 1 };\n'
        assert _validate_js_harness(code) == ""

    def test_js_comments_not_false_positive(self):
        """Brackets inside JS comments don't cause false positives."""
        code = "// this has { unclosed\nvar x = { a: 1 };\n/* also { here */\n"
        assert _validate_js_harness(code) == ""

    def test_validate_harness_routes_correctly(self):
        """_validate_harness dispatches to correct validator."""
        assert _validate_harness("x = 1", "python") == ""
        assert _validate_harness("var x = 1;", "node") == ""
        assert _validate_harness("def foo(:", "python") != ""

    def test_harness_generation_failure_result(self, tmp_path):
        """Invalid harness produces HarnessGenerationError, not project failure."""
        session = _make_session(tmp_path)
        engine = ExecutionEngine(session, _make_ingestion(), language="python")

        # Monkey-patch _build_harness to return invalid Python
        original = engine._build_harness
        engine._build_harness = lambda *a, **kw: "def broken(\n"

        scenario = StressTestScenario(
            name="test_broken_harness",
            description="Test with broken harness",
            category="data_volume_scaling",
            test_config={"steps": [], "resource_limits": {}},
        )
        result = engine._execute_scenario(scenario)

        assert result.status == "failed"
        assert "harness generation failure" in result.summary.lower()
        assert result.steps[0].errors[0]["type"] == "HarnessGenerationError"
        assert "mycode limitation" in result.steps[0].errors[0]["message"].lower()

        # Restore
        engine._build_harness = original

    def test_all_python_harness_bodies_valid(self):
        """Every built-in Python harness body produces valid syntax."""
        for name, body in _CATEGORY_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            error = _validate_python_harness(script)
            assert error == "", f"Python body '{name}' has syntax error: {error}"

    def test_all_python_coupling_bodies_valid(self):
        """Every Python coupling body produces valid syntax."""
        for name, body in _PY_COUPLING_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            error = _validate_python_harness(script)
            assert error == "", f"Python coupling body '{name}' has syntax error: {error}"

    def test_all_python_lib_bodies_valid(self):
        """Every Python lib body produces valid syntax."""
        for name, body in _PY_LIB_BODIES.items():
            script = _HARNESS_PREAMBLE + "\n" + body + "\n" + _HARNESS_POSTAMBLE
            error = _validate_python_harness(script)
            assert error == "", f"Python lib body '{name}' has syntax error: {error}"

    def test_all_js_harness_bodies_valid(self):
        """Every built-in JS harness body has balanced brackets."""
        for name, body in _JS_CATEGORY_BODIES.items():
            script = _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE
            error = _validate_js_harness(script)
            assert error == "", f"JS body '{name}' has bracket error: {error}"

    def test_all_js_coupling_bodies_valid(self):
        """Every JS coupling body has balanced brackets."""
        for name, body in _JS_COUPLING_BODIES.items():
            script = _JS_HARNESS_PREAMBLE + "\n" + body + "\n" + _JS_HARNESS_POSTAMBLE
            error = _validate_js_harness(script)
            assert error == "", f"JS coupling body '{name}' has bracket error: {error}"


# ── Harness Failure Classification Tests ──


from mycode.engine import _classify_harness_failure


class TestClassifyHarnessFailure:
    """Tests for _classify_harness_failure() stderr classification."""

    def test_syntax_error_classified(self):
        stderr = "SyntaxError: invalid syntax (line 5)"
        assert _classify_harness_failure(stderr) == "harness_generation_error"

    def test_name_error_classified(self):
        stderr = "NameError: name 'foo' is not defined"
        assert _classify_harness_failure(stderr) == "harness_generation_error"

    def test_module_not_found_framework(self):
        stderr = "ModuleNotFoundError: No module named 'django'"
        assert _classify_harness_failure(stderr) == "unsupported_framework"

    def test_module_not_found_expo(self):
        stderr = "ModuleNotFoundError: No module named 'expo'"
        assert _classify_harness_failure(stderr) == "unsupported_framework"

    def test_module_not_found_with_pip_suggestion(self):
        stderr = (
            "ModuleNotFoundError: No module named 'celery'\n"
            "pip install celery"
        )
        assert _classify_harness_failure(stderr) == "dependency_unavailable"

    def test_module_not_found_user_module(self):
        stderr = "ModuleNotFoundError: No module named 'myapp'"
        assert _classify_harness_failure(stderr) == "module_import_failure"

    def test_npm_error(self):
        stderr = "npm ERR! code E404\nnpm ERR! 404 Not Found"
        assert _classify_harness_failure(stderr) == "dependency_unavailable"

    def test_pip_error(self):
        stderr = "pip install failed: No matching distribution"
        assert _classify_harness_failure(stderr) == "dependency_unavailable"

    def test_empty_stderr(self):
        assert _classify_harness_failure("") == "unknown"

    def test_unrecognized_error(self):
        stderr = "RuntimeError: something bad happened"
        assert _classify_harness_failure(stderr) == "unknown"

    def test_cannot_find_module_js(self):
        stderr = "Error: Cannot find module 'express'"
        assert _classify_harness_failure(stderr) == "unsupported_framework"

    def test_scenario_result_has_failure_reason_field(self):
        sr = ScenarioResult(
            scenario_name="test",
            scenario_category="test_cat",
            status="failed",
            failure_reason="harness_generation_error",
        )
        assert sr.failure_reason == "harness_generation_error"

    def test_scenario_result_failure_reason_default_empty(self):
        sr = ScenarioResult(
            scenario_name="test",
            scenario_category="test_cat",
            status="completed",
        )
        assert sr.failure_reason == ""


# ── Probe-and-Skip Tests ──


class TestProbeAndSkip:
    """Tests for the runtime context probe in harness preamble."""

    def test_probe_code_in_preamble(self):
        """Preamble contains probe section."""
        assert "_probe_results" in _HARNESS_PREAMBLE
        assert "_probed_callables" in _HARNESS_PREAMBLE
        assert "_CONTEXT_ATTRS" in _HARNESS_PREAMBLE
        assert "Probe timeout" in _HARNESS_PREAMBLE

    def test_postamble_includes_probe_results(self):
        """Postamble outputs probe_skipped in JSON."""
        assert "probe_skipped" in _HARNESS_POSTAMBLE
        assert "probe_results" in _HARNESS_POSTAMBLE

    def test_parse_all_probed_out(self, tmp_path):
        """When all callables fail probe, result is skipped with runtime_context_required."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        probe_output = json.dumps({
            "steps": [],
            "import_errors": [],
            "probe_skipped": [
                {"name": "app.render_dashboard", "error": {"type": "AttributeError", "message": "st.session_state"}},
                {"name": "app.load_data", "error": {"type": "ConnectionError", "message": "Connection refused"}},
            ],
        })
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{probe_output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"
        assert len(sr.probe_skipped) == 2
        assert sr.probe_skipped[0]["name"] == "app.render_dashboard"

    def test_parse_partial_probe_out(self, tmp_path):
        """When some callables are probed out but steps still run, probe_skipped is populated."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = json.dumps({
            "steps": [
                {
                    "step_name": "data_size_100",
                    "parameters": {"data_size": 100},
                    "execution_time_ms": 5.0,
                    "memory_peak_mb": 1.0,
                    "error_count": 0,
                    "errors": [],
                    "resource_cap_hit": "",
                },
            ],
            "import_errors": [],
            "probe_skipped": [
                {"name": "app.render_dashboard", "error": {"type": "AttributeError", "message": "st.session_state"}},
            ],
        })
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        # Scenario ran (not skipped), but has probe_skipped
        assert sr.status == "completed"
        assert sr.failure_reason == ""
        assert len(sr.probe_skipped) == 1
        assert "runtime context" in sr.summary.lower()

    def test_parse_no_probes_skipped(self, tmp_path):
        """Normal scenario with no probe failures."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = json.dumps({
            "steps": [
                {
                    "step_name": "data_size_100",
                    "parameters": {"data_size": 100},
                    "execution_time_ms": 5.0,
                    "memory_peak_mb": 1.0,
                    "error_count": 0,
                    "errors": [],
                    "resource_cap_hit": "",
                },
            ],
            "import_errors": [],
            "probe_skipped": [],
        })
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "completed"
        assert sr.probe_skipped == []
        assert "runtime context" not in sr.summary.lower()

    def test_scenario_result_probe_skipped_default(self):
        """ScenarioResult.probe_skipped defaults to empty list."""
        sr = ScenarioResult(
            scenario_name="test",
            scenario_category="test_cat",
            status="completed",
        )
        assert sr.probe_skipped == []

    def test_harness_syntax_valid(self):
        """The updated preamble with probe code is syntactically valid."""
        from mycode.engine import _BODY_GENERIC
        full_harness = _HARNESS_PREAMBLE + "\n" + _BODY_GENERIC + "\n" + _HARNESS_POSTAMBLE
        import ast
        # Should not raise SyntaxError
        ast.parse(full_harness)


class TestIdenticalErrorDetection:
    """Tests for the post-execution identical-error-at-every-step detection."""

    def _make_steps_output(
        self, error_counts, import_errors=None,
        error_type="ImportError", error_message="No module named 'yfinance'",
    ):
        """Build harness JSON output with given error counts per step."""
        steps = []
        for i, ec in enumerate(error_counts):
            steps.append({
                "step_name": f"batch_{i}",
                "parameters": {"batch_start": i * 5, "batch_count": 5},
                "execution_time_ms": 10.0,
                "memory_peak_mb": 2.0,
                "error_count": ec,
                "errors": [
                    {"type": error_type, "message": error_message}
                    for j in range(ec)
                ],
                "resource_cap_hit": "",
            })
        return json.dumps({
            "steps": steps,
            "import_errors": import_errors or [],
            "probe_skipped": [],
        })

    def test_identical_errors_reclassified(self, tmp_path):
        """10 steps all with 30 errors → runtime_context_required."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        # 10 batches, each with 30 identical errors (the Financial Dashboard pattern)
        output = self._make_steps_output([30] * 10, import_errors=[
            {"type": "ModuleNotFoundError", "message": "No module named 'yfinance'"},
        ])
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="pandas_memory_profiling_over_time",
            category="memory_profiling",
            description="Memory profiling test",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"
        assert "identical" in sr.summary.lower()

    def test_identical_errors_without_import_error(self, tmp_path):
        """Identical errors at every step even without module_import → reclassified."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output([5] * 5)
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="data_volume_scaling_numpy",
            category="data_volume_scaling",
            description="Data volume test",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"

    def test_varying_errors_not_reclassified(self, tmp_path):
        """Error counts that vary across steps → real scaling issue, NOT reclassified."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        # Errors increase with data size — genuine scaling failure
        output = self._make_steps_output([0, 2, 5, 8, 15])
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="data_volume_scaling_pandas",
            category="data_volume_scaling",
            description="Data volume test",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_some_steps_zero_errors_not_reclassified(self, tmp_path):
        """Some steps pass, some fail → not identical, NOT reclassified."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output([0, 0, 3, 3, 3])
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""

    def test_two_steps_not_enough_for_detection(self, tmp_path):
        """Only 2 test steps with same errors → NOT reclassified (need ≥3)."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output([5, 5])
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""

    def test_timeout_steps_excluded_from_check(self, tmp_path):
        """Time-budget-exceeded steps are excluded from the identical-error check."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        # 5 real steps with identical errors + 2 timeout skips
        steps = []
        for i in range(5):
            steps.append({
                "step_name": f"batch_{i}",
                "parameters": {},
                "execution_time_ms": 10.0,
                "memory_peak_mb": 2.0,
                "error_count": 10,
                "errors": [{"type": "ImportError", "message": "No module named 'yfinance'"}] * 10,
                "resource_cap_hit": "",
            })
        for i in range(5, 7):
            steps.append({
                "step_name": f"batch_{i}",
                "parameters": {},
                "execution_time_ms": 0,
                "memory_peak_mb": 0,
                "error_count": 1,
                "errors": [{"type": "Skipped", "message": "time budget exceeded"}],
                "resource_cap_hit": "timeout",
            })

        output = json.dumps({
            "steps": steps,
            "import_errors": [],
            "probe_skipped": [],
        })
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="memory_profiling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        # The 5 real steps have identical errors → reclassified
        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"

    def test_typeerror_not_reclassified(self, tmp_path):
        """TypeError at every step is a data shape issue, NOT runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [10] * 5,
            error_type="TypeError",
            error_message="expected str, got int",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_valueerror_not_reclassified(self, tmp_path):
        """ValueError at every step is a data issue, NOT runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [8] * 4,
            error_type="ValueError",
            error_message="invalid literal for int()",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_attributeerror_with_framework_keyword_reclassified(self, tmp_path):
        """AttributeError mentioning 'st.' (Streamlit) → runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [5] * 5,
            error_type="AttributeError",
            error_message="module 'streamlit' has no attribute 'session_state'",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="memory_profiling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"

    def test_attributeerror_without_framework_keyword_not_reclassified(self, tmp_path):
        """AttributeError without framework keywords → real bug, NOT reclassified."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [5] * 5,
            error_type="AttributeError",
            error_message="'NoneType' object has no attribute 'append'",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_connectionerror_reclassified(self, tmp_path):
        """ConnectionError at every step → runtime context (needs server)."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [3] * 4,
            error_type="ConnectionRefusedError",
            error_message="Connection refused",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="concurrent_access",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.status == "skipped"
        assert sr.failure_reason == "runtime_context_required"

    def test_keyerror_not_reclassified(self, tmp_path):
        """KeyError at every step is a data structure issue, NOT runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [6] * 4,
            error_type="KeyError",
            error_message="'missing_column'",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_unknown_error_type_not_reclassified(self, tmp_path):
        """Unrecognized error type should NOT default to runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        output = self._make_steps_output(
            [4] * 5,
            error_type="CustomAppError",
            error_message="something went wrong",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"

    def test_no_error_type_not_reclassified(self, tmp_path):
        """Errors with no type field should NOT default to runtime context."""
        session = _make_session(tmp_path)
        ingestion = _make_ingestion()

        # Build steps with errors that have empty type
        output = self._make_steps_output(
            [3] * 4,
            error_type="",
            error_message="something failed",
        )
        session.run_in_session.return_value = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n{output}\n{_RESULTS_END}",
            stderr="",
        )

        engine = ExecutionEngine(session=session, ingestion=ingestion)
        scenario = StressTestScenario(
            name="test_scenario",
            category="data_volume_scaling",
            description="Test scenario",
        )
        sr = engine._execute_scenario(scenario)

        assert sr.failure_reason == ""
        assert sr.status != "skipped"


class TestBrowserOnlySkip:
    """Test that browser-only projects skip callable harnesses."""

    def test_react_project_skips_harness(self, tmp_path):
        """A React project with only browser deps should skip all scenarios."""
        session = _make_session(tmp_path)
        # Node.js available
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        ingestion = IngestionResult(
            project_path="/fake/react-app",
            files_analyzed=1,
            file_analyses=[],
            dependencies=[
                DependencyInfo(name="react"),
                DependencyInfo(name="react-dom"),
                DependencyInfo(name="react-scripts"),
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="coupling_render_App",
                category="state_management_degradation",
                description="Test",
            ),
            StressTestScenario(
                name="react_data_volume",
                category="data_volume_scaling",
                description="Test",
            ),
        ]
        result = engine.execute(scenarios)

        assert len(result.scenario_results) == 2
        for sr in result.scenario_results:
            assert sr.status == "skipped"
            assert sr.failure_reason == "browser_framework"
            assert "browser environment" in sr.summary.lower()

    def test_express_project_not_skipped(self, tmp_path):
        """An Express project should NOT be considered browser-only."""
        session = _make_session(tmp_path)
        version_result = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        harness_result = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n" + json.dumps({
                "steps": [], "import_errors": [], "probe_skipped": [],
            }) + f"\n{_RESULTS_END}",
            stderr="",
        )
        session.run_in_session.side_effect = [version_result, harness_result]
        ingestion = IngestionResult(
            project_path="/fake/express-app",
            files_analyzed=1,
            file_analyses=[
                FileAnalysis(
                    file_path="index.js",
                    functions=[FunctionInfo(name="handler", file_path="index.js", lineno=1, args=[])],
                    classes=[], imports=[], lines_of_code=10,
                ),
            ],
            dependencies=[
                DependencyInfo(name="express"),
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="express_load",
                category="data_volume_scaling",
                description="Test",
            ),
        ]
        result = engine.execute(scenarios)
        # Should NOT be skipped as browser-only
        assert result.scenario_results[0].failure_reason != "browser_framework"

    def test_mixed_deps_not_skipped(self, tmp_path):
        """A project with both browser and server deps should NOT skip."""
        session = _make_session(tmp_path)
        version_result = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        harness_result = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n" + json.dumps({
                "steps": [], "import_errors": [], "probe_skipped": [],
            }) + f"\n{_RESULTS_END}",
            stderr="",
        )
        session.run_in_session.side_effect = [version_result, harness_result]
        ingestion = IngestionResult(
            project_path="/fake/fullstack",
            files_analyzed=1,
            file_analyses=[
                FileAnalysis(
                    file_path="server.js",
                    functions=[FunctionInfo(name="serve", file_path="server.js", lineno=1, args=[])],
                    classes=[], imports=[], lines_of_code=10,
                ),
            ],
            dependencies=[
                DependencyInfo(name="react"),
                DependencyInfo(name="express"),  # server dep -> not browser-only
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="test_load",
                category="data_volume_scaling",
                description="Test",
            ),
        ]
        result = engine.execute(scenarios)
        assert result.scenario_results[0].failure_reason != "browser_framework"

    def test_dev_deps_excluded_from_check(self, tmp_path):
        """Dev dependencies should not count toward browser-only detection."""
        session = _make_session(tmp_path)
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        ingestion = IngestionResult(
            project_path="/fake/react-app",
            files_analyzed=1,
            file_analyses=[],
            dependencies=[
                DependencyInfo(name="react"),
                DependencyInfo(name="react-dom"),
                DependencyInfo(name="react-scripts"),
                DependencyInfo(name="jest", is_dev=True),  # dev dep ignored
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="test_scenario", category="data_volume_scaling", description="Test",
            ),
        ]
        result = engine.execute(scenarios)
        # Still browser-only (jest is dev)
        assert result.scenario_results[0].status == "skipped"
        assert result.scenario_results[0].failure_reason == "browser_framework"

    def test_react_with_utility_deps_still_browser_only(self, tmp_path):
        """React + utility packages like sass should still be browser-only."""
        session = _make_session(tmp_path)
        session.run_in_session.return_value = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        ingestion = IngestionResult(
            project_path="/fake/react-app",
            files_analyzed=1,
            file_analyses=[],
            dependencies=[
                DependencyInfo(name="react"),
                DependencyInfo(name="react-dom"),
                DependencyInfo(name="react-scripts"),
                DependencyInfo(name="sass"),  # utility, not server-side
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="test_scenario", category="data_volume_scaling", description="Test",
            ),
        ]
        result = engine.execute(scenarios)
        assert result.scenario_results[0].status == "skipped"
        assert result.scenario_results[0].failure_reason == "browser_framework"

    def test_no_browser_framework_not_skipped(self, tmp_path):
        """A project with only utility deps and no browser framework is NOT skipped."""
        session = _make_session(tmp_path)
        version_result = SessionResult(
            returncode=0, stdout="v20.11.0\n", stderr="",
        )
        harness_result = SessionResult(
            returncode=0,
            stdout=f"{_RESULTS_START}\n" + json.dumps({
                "steps": [], "import_errors": [], "probe_skipped": [],
            }) + f"\n{_RESULTS_END}",
            stderr="",
        )
        session.run_in_session.side_effect = [version_result, harness_result]
        ingestion = IngestionResult(
            project_path="/fake/utility-app",
            files_analyzed=1,
            file_analyses=[
                FileAnalysis(
                    file_path="index.js",
                    functions=[FunctionInfo(name="run", file_path="index.js", lineno=1, args=[])],
                    classes=[], imports=[], lines_of_code=10,
                ),
            ],
            dependencies=[
                DependencyInfo(name="lodash"),
                DependencyInfo(name="axios"),
            ],
        )
        engine = ExecutionEngine(session, ingestion, language="javascript")
        scenarios = [
            StressTestScenario(
                name="test_scenario", category="data_volume_scaling", description="Test",
            ),
        ]
        result = engine.execute(scenarios)
        assert result.scenario_results[0].failure_reason != "browser_framework"
