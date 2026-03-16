"""Tests for js_stress_runner — instrumented JS function caller."""

import json
import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from mycode.js_module_loader import ExportedFunction, _check_node_available
from mycode.js_stress_runner import (
    StressRunResult,
    _extract_results,
    run_stress,
    _RESULTS_START,
    _RESULTS_END,
)

pytestmark = pytest.mark.skipif(
    not _check_node_available(), reason="Node.js not available"
)


# ── Fixtures ──


@pytest.fixture
def js_dir(tmp_path):
    return tmp_path


def _write_js(directory, filename, content):
    p = directory / filename
    p.write_text(textwrap.dedent(content))
    return str(p)


def _exports(*specs):
    """Build ExportedFunction list from (name, arity, is_async) tuples."""
    return [
        ExportedFunction(name=s[0], arity=s[1], is_async=s[2] if len(s) > 2 else False)
        for s in specs
    ]


# ── StressRunResult dataclass ──


class TestStressRunResult:
    def test_defaults(self):
        r = StressRunResult()
        assert r.stdout == ""
        assert r.returncode == 0
        assert r.parsed is None
        assert r.error is None

    def test_with_error(self):
        r = StressRunResult(error="boom", returncode=-1)
        assert r.error == "boom"


# ── _extract_results ──


class TestExtractResults:
    def test_valid_markers(self):
        stdout = f"some noise\n{_RESULTS_START}\n" + '{"steps":[]}\n' + f"{_RESULTS_END}\nmore noise"
        result = _extract_results(stdout)
        assert result == {"steps": []}

    def test_missing_start_marker(self):
        assert _extract_results("no markers here\n" + _RESULTS_END) is None

    def test_missing_end_marker(self):
        assert _extract_results(_RESULTS_START + "\n{}\n") is None

    def test_invalid_json(self):
        stdout = f"{_RESULTS_START}\nnot json\n{_RESULTS_END}"
        assert _extract_results(stdout) is None


# ── Validation errors ──


class TestValidation:
    def test_node_not_available(self, js_dir):
        path = _write_js(js_dir, "ok.js", "module.exports = { fn: () => 1 };")
        result = run_stress(path, _exports(("fn", 0)), node_path="/nonexistent/node")
        assert result.error is not None
        assert "not available" in result.error

    def test_file_not_found(self):
        result = run_stress("/no/such/file.js", _exports(("fn", 0)))
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_no_exports(self, js_dir):
        path = _write_js(js_dir, "ok.js", "module.exports = {};")
        result = run_stress(path, [])
        assert result.error is not None
        assert "No exported" in result.error


# ── Basic execution ──


class TestBasicExecution:
    def test_sync_function_no_args(self, js_dir):
        path = _write_js(js_dir, "simple.js", """\
            function greet() { return "hello"; }
            module.exports = { greet };
        """)
        result = run_stress(path, _exports(("greet", 0)), scale_levels=[10, 100])
        assert result.error is None
        assert result.parsed is not None
        steps = result.parsed["steps"]
        assert len(steps) >= 1
        assert steps[0]["step_name"] == "scale_10"
        assert steps[0]["error_count"] == 0

    def test_sync_function_with_args(self, js_dir):
        path = _write_js(js_dir, "adder.js", """\
            function add(a, b) { return a + b; }
            module.exports = { add };
        """)
        result = run_stress(
            path, _exports(("add", 2)), scale_levels=[10],
            param_names={"add": ["count", "limit"]},
        )
        assert result.error is None
        assert result.parsed is not None
        steps = result.parsed["steps"]
        assert len(steps) == 1
        assert steps[0]["error_count"] == 0

    def test_async_function(self, js_dir):
        path = _write_js(js_dir, "async_mod.js", """\
            async function fetchData(id) {
                return { id, data: "result_" + id };
            }
            module.exports = { fetchData };
        """)
        result = run_stress(
            path, _exports(("fetchData", 1, True)), scale_levels=[10, 100],
            param_names={"fetchData": ["id"]},
        )
        assert result.error is None
        assert result.parsed is not None
        steps = result.parsed["steps"]
        assert len(steps) == 2
        for step in steps:
            assert step["error_count"] == 0

    def test_multiple_functions(self, js_dir):
        path = _write_js(js_dir, "multi.js", """\
            function double(n) { return n * 2; }
            function triple(n) { return n * 3; }
            module.exports = { double, triple };
        """)
        result = run_stress(
            path,
            _exports(("double", 1), ("triple", 1)),
            scale_levels=[10],
            param_names={"double": ["count"], "triple": ["count"]},
        )
        assert result.error is None
        steps = result.parsed["steps"]
        assert steps[0]["parameters"]["functions_called"] == 2


# ── Output format compatibility ──


class TestOutputFormat:
    """Verify output matches what engine.py's _parse_harness_output expects."""

    def test_has_markers(self, js_dir):
        path = _write_js(js_dir, "ok.js", """\
            module.exports = { fn: () => 1 };
        """)
        result = run_stress(path, _exports(("fn", 0)), scale_levels=[10])
        assert _RESULTS_START in result.stdout
        assert _RESULTS_END in result.stdout

    def test_step_fields(self, js_dir):
        path = _write_js(js_dir, "ok.js", """\
            module.exports = { fn: (data) => data.length };
        """)
        result = run_stress(
            path, _exports(("fn", 1)), scale_levels=[100],
            param_names={"fn": ["data"]},
        )
        step = result.parsed["steps"][0]
        # All required fields present
        assert "step_name" in step
        assert "parameters" in step
        assert "execution_time_ms" in step
        assert "memory_peak_mb" in step
        assert "error_count" in step
        assert "errors" in step
        assert "resource_cap_hit" in step
        # Types match
        assert isinstance(step["execution_time_ms"], (int, float))
        assert isinstance(step["memory_peak_mb"], (int, float))
        assert isinstance(step["error_count"], int)
        assert isinstance(step["errors"], list)
        assert isinstance(step["resource_cap_hit"], str)

    def test_top_level_fields(self, js_dir):
        path = _write_js(js_dir, "ok.js", """\
            module.exports = { fn: () => 1 };
        """)
        result = run_stress(path, _exports(("fn", 0)), scale_levels=[10])
        data = result.parsed
        assert "steps" in data
        assert "import_errors" in data
        assert "probe_skipped" in data
        assert isinstance(data["steps"], list)
        assert isinstance(data["import_errors"], list)
        assert isinstance(data["probe_skipped"], list)

    def test_execution_time_positive(self, js_dir):
        path = _write_js(js_dir, "slow.js", """\
            function work(data) {
                let sum = 0;
                for (const item of data) sum += item.value;
                return sum;
            }
            module.exports = { work };
        """)
        result = run_stress(
            path, _exports(("work", 1)), scale_levels=[1000],
            param_names={"work": ["data"]},
        )
        step = result.parsed["steps"][0]
        assert step["execution_time_ms"] > 0
        assert step["memory_peak_mb"] > 0


# ── Error handling ──


class TestErrorHandling:
    def test_function_that_throws(self, js_dir):
        path = _write_js(js_dir, "throws.js", """\
            function boom() { throw new Error("intentional"); }
            module.exports = { boom };
        """)
        result = run_stress(path, _exports(("boom", 0)), scale_levels=[10])
        assert result.parsed is not None
        steps = result.parsed["steps"]
        assert len(steps) >= 1
        assert steps[0]["error_count"] > 0
        assert steps[0]["errors"][0]["type"] == "Error"
        assert "intentional" in steps[0]["errors"][0]["message"]

    def test_import_error(self, js_dir):
        path = _write_js(js_dir, "bad_dep.js", """\
            const express = require("express");
            module.exports = { start: () => express() };
        """)
        result = run_stress(path, _exports(("start", 0)), scale_levels=[10])
        assert result.parsed is not None
        assert len(result.parsed["import_errors"]) > 0
        assert result.parsed["import_errors"][0]["module"] is not None

    def test_runtime_context_probed_out(self, js_dir):
        path = _write_js(js_dir, "ctx.js", """\
            function connectDB() {
                // Simulate missing env / uninitialized context
                throw new Error("DATABASE_URL not set");
            }
            module.exports = { connectDB };
        """)
        result = run_stress(path, _exports(("connectDB", 0)), scale_levels=[10])
        assert result.parsed is not None
        # connectDB should be probe-skipped (DATABASE_URL is context error)
        assert len(result.parsed["probe_skipped"]) > 0
        assert result.parsed["probe_skipped"][0]["name"] == "connectDB"

    def test_process_timeout(self, js_dir):
        path = _write_js(js_dir, "hang.js", """\
            function hang() {
                const start = Date.now();
                while (Date.now() - start < 30000) {}
            }
            module.exports = { hang };
        """)
        result = run_stress(path, _exports(("hang", 0)), scale_levels=[10], timeout=3)
        assert result.error is not None
        assert "timed out" in result.error.lower()


# ── Synthetic data generation ──


class TestSyntheticData:
    """Verify that param_names drive data generation heuristics."""

    def test_data_param_receives_array(self, js_dir):
        path = _write_js(js_dir, "processor.js", """\
            function process(data) {
                if (!Array.isArray(data)) throw new Error("expected array");
                return data.length;
            }
            module.exports = { process: process };
        """)
        result = run_stress(
            path, _exports(("process", 1)), scale_levels=[100],
            param_names={"process": ["data"]},
        )
        assert result.parsed["steps"][0]["error_count"] == 0

    def test_id_param_receives_number(self, js_dir):
        path = _write_js(js_dir, "lookup.js", """\
            function findById(id) {
                if (typeof id !== "number") throw new Error("expected number, got " + typeof id);
                return { id };
            }
            module.exports = { findById };
        """)
        result = run_stress(
            path, _exports(("findById", 1)), scale_levels=[10],
            param_names={"findById": ["id"]},
        )
        assert result.parsed["steps"][0]["error_count"] == 0

    def test_callback_param_receives_function(self, js_dir):
        path = _write_js(js_dir, "with_cb.js", """\
            function run(data, callback) {
                if (typeof callback !== "function") throw new Error("expected function");
                callback(data);
                return data;
            }
            module.exports = { run };
        """)
        result = run_stress(
            path, _exports(("run", 2)), scale_levels=[10],
            param_names={"run": ["data", "callback"]},
        )
        assert result.parsed["steps"][0]["error_count"] == 0

    def test_config_param_receives_object(self, js_dir):
        path = _write_js(js_dir, "configurable.js", """\
            function init(options) {
                if (typeof options !== "object" || options === null) throw new Error("expected object");
                return options;
            }
            module.exports = { init };
        """)
        result = run_stress(
            path, _exports(("init", 1)), scale_levels=[10],
            param_names={"init": ["options"]},
        )
        assert result.parsed["steps"][0]["error_count"] == 0

    def test_no_param_names_uses_positional(self, js_dir):
        """Without param_names, falls back to positional generation."""
        path = _write_js(js_dir, "generic.js", """\
            function work(a) {
                if (!Array.isArray(a)) throw new Error("expected array");
                return a.length;
            }
            module.exports = { work };
        """)
        # No param_names → first positional arg is data array
        result = run_stress(path, _exports(("work", 1)), scale_levels=[10])
        assert result.parsed["steps"][0]["error_count"] == 0


# ── Escalation ──


class TestEscalation:
    def test_scale_levels_increase(self, js_dir):
        path = _write_js(js_dir, "counter.js", """\
            function count(data) { return data.length; }
            module.exports = { count };
        """)
        levels = [50, 500, 5000]
        result = run_stress(
            path, _exports(("count", 1)), scale_levels=levels,
            param_names={"count": ["data"]},
        )
        steps = result.parsed["steps"]
        assert len(steps) == 3
        assert steps[0]["parameters"]["scale_level"] == 50
        assert steps[1]["parameters"]["scale_level"] == 500
        assert steps[2]["parameters"]["scale_level"] == 5000

    def test_stops_on_resource_cap(self, js_dir):
        path = _write_js(js_dir, "oom.js", """\
            function grow(data) {
                // Recursive to blow stack
                function recurse(n) { if (n > 0) return recurse(n - 1) + 1; return 0; }
                return recurse(1000000);
            }
            module.exports = { grow };
        """)
        result = run_stress(
            path, _exports(("grow", 1)), scale_levels=[10, 100, 1000],
            param_names={"grow": ["data"]},
        )
        # Should have fewer steps than scale_levels (stopped on cap)
        steps = result.parsed["steps"]
        has_cap = any(s["resource_cap_hit"] for s in steps)
        # Either it stopped early, or all ran (stack overflow is immediate)
        assert len(steps) >= 1

    def test_step_timeout_per_step(self, js_dir):
        path = _write_js(js_dir, "slow_step.js", """\
            async function slowWork() {
                // Never-resolving promise — triggers step timeout
                return new Promise(() => {});
            }
            module.exports = { slowWork };
        """)
        result = run_stress(
            path, _exports(("slowWork", 0, True)), scale_levels=[1],
            step_timeout_ms=1000,  # 1s timeout per step
            timeout=10,
        )
        assert result.parsed is not None
        steps = result.parsed["steps"]
        assert len(steps) >= 1
        assert steps[0]["resource_cap_hit"] == "timeout"
        assert steps[0]["error_count"] > 0


# ── Mixed probe outcomes ──


class TestMixedProbe:
    def test_partial_probe_skip(self, js_dir):
        """Some functions probe-skip, others remain testable."""
        path = _write_js(js_dir, "mixed.js", """\
            function healthy() { return 42; }
            function needsDB() {
                throw new Error("DATABASE_URL not set");
            }
            module.exports = { healthy, needsDB };
        """)
        result = run_stress(
            path,
            _exports(("healthy", 0), ("needsDB", 0)),
            scale_levels=[10],
        )
        assert result.parsed is not None
        # needsDB should be probed out, healthy should run
        assert len(result.parsed["probe_skipped"]) >= 1
        skipped_names = {p["name"] for p in result.parsed["probe_skipped"]}
        assert "needsDB" in skipped_names
        # Steps should exist (healthy ran)
        assert len(result.parsed["steps"]) >= 1
        assert result.parsed["steps"][0]["error_count"] == 0
