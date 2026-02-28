"""Tests for Scenario Discovery Engine (Component 9) — discovery logging.

Tests cover:
  - DiscoveryCandidate serialization and schema completeness
  - Helper functions: _parse_typical_limit, _extract_load_level, _detect_superlinear
  - Detection methods: crash at safe level, memory growth anomaly,
    curve shape mismatch, interaction failure, unrecognized dep failure
  - Exclusion logic: known failures, extreme scale, skipped scenarios
  - Save method: JSON written to correct dir with correct schema
  - End-to-end analyse integration with mixed scenario results
"""

import json
from pathlib import Path
from typing import Optional

import pytest

from mycode.constraints import OperationalConstraints
from mycode.discovery import (
    DiscoveryCandidate,
    DiscoveryEngine,
    _build_dep_strings,
    _constraints_to_dict,
    _detect_superlinear,
    _extract_load_level,
    _is_known_failure,
    _parse_typical_limit,
)
from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.library.loader import DependencyProfile, ProfileMatch
from mycode.scenario import ScenarioGeneratorResult, StressTestScenario


# ── Test Helpers ──


def _make_step(
    name: str = "step_1",
    params: Optional[dict] = None,
    memory_peak_mb: float = 0.0,
    execution_time_ms: float = 0.0,
    error_count: int = 0,
    errors: Optional[list] = None,
    resource_cap_hit: str = "",
) -> StepResult:
    return StepResult(
        step_name=name,
        parameters=params or {},
        memory_peak_mb=memory_peak_mb,
        execution_time_ms=execution_time_ms,
        error_count=error_count,
        errors=errors or [],
        resource_cap_hit=resource_cap_hit,
    )


def _make_scenario_result(
    name: str = "test_scenario",
    category: str = "data_volume_scaling",
    status: str = "completed",
    steps: Optional[list[StepResult]] = None,
    total_errors: int = 0,
) -> ScenarioResult:
    steps = steps or []
    return ScenarioResult(
        scenario_name=name,
        scenario_category=category,
        status=status,
        steps=steps,
        total_errors=total_errors,
    )


def _make_scenario(
    name: str = "test_scenario",
    category: str = "data_volume_scaling",
    target_deps: Optional[list[str]] = None,
    expected_behavior: str = "",
) -> StressTestScenario:
    return StressTestScenario(
        name=name,
        category=category,
        description="Test scenario",
        target_dependencies=target_deps or [],
        expected_behavior=expected_behavior,
    )


def _make_profile_data(
    name: str = "testlib",
    baseline_footprint_mb: float = 50.0,
    scaling_limits: Optional[list] = None,
    known_failure_modes: Optional[list] = None,
    growth_pattern: str = "Linear memory growth with data size",
    known_conflicts: Optional[list] = None,
) -> dict:
    """Build a raw profile dict for constructing DependencyProfile."""
    return {
        "identity": {
            "name": name,
            "category": "testing",
            "current_stable_version": "1.0.0",
        },
        "scaling_characteristics": {
            "description": "Test scaling",
            "concurrency_model": "single_threaded",
            "bottlenecks": [],
            "scaling_limits": scaling_limits or [],
        },
        "memory_behavior": {
            "baseline_footprint_mb": baseline_footprint_mb,
            "growth_pattern": growth_pattern,
            "known_leaks": [],
            "gc_behavior": "standard",
        },
        "known_failure_modes": known_failure_modes or [],
        "edge_case_sensitivities": [],
        "interaction_patterns": {
            "commonly_used_with": [],
            "known_conflicts": known_conflicts or [],
            "dependency_chain_risks": [],
        },
        "stress_test_templates": [],
    }


def _make_profile(raw: dict) -> DependencyProfile:
    """Construct a DependencyProfile from a raw dict."""
    return DependencyProfile(
        identity=raw["identity"],
        scaling_characteristics=raw["scaling_characteristics"],
        memory_behavior=raw["memory_behavior"],
        known_failure_modes=raw["known_failure_modes"],
        edge_case_sensitivities=raw["edge_case_sensitivities"],
        interaction_patterns=raw["interaction_patterns"],
        stress_test_templates=raw["stress_test_templates"],
        raw=raw,
    )


def _make_profile_match(
    name: str = "testlib",
    profile_data: Optional[dict] = None,
    installed_version: str = "1.0.0",
) -> ProfileMatch:
    if profile_data is not None:
        profile = _make_profile(profile_data)
    else:
        profile = _make_profile(_make_profile_data(name=name))
    return ProfileMatch(
        dependency_name=name,
        profile=profile,
        installed_version=installed_version,
    )


def _make_unrecognized_match(name: str = "unknown_lib") -> ProfileMatch:
    """ProfileMatch with no profile (unrecognized dependency)."""
    return ProfileMatch(
        dependency_name=name,
        profile=None,
        installed_version="0.1.0",
    )


# ── TestDiscoveryCandidate ──


class TestDiscoveryCandidate:
    def test_to_dict_returns_all_fields(self):
        dc = DiscoveryCandidate(
            discovery_id="test-uuid",
            timestamp="2026-02-28T00:00:00+00:00",
            mycode_version="0.1.1",
            language="python",
            dependencies_involved=["pandas==2.2.1"],
            scenario_category="memory_profiling",
            expected_behavior="Linear growth",
            actual_behavior="Exponential growth",
            deviation_factor=4.8,
            load_level_at_discovery="14 concurrent sessions",
            reproducible=True,
            constraint_context={"user_scale": 200},
            raw_metrics={"mem": 100},
            suggested_template="Test memory growth",
        )
        d = dc.to_dict()

        assert d["discovery_id"] == "test-uuid"
        assert d["language"] == "python"
        assert d["deviation_factor"] == 4.8
        assert d["reproducible"] is True
        assert isinstance(d["dependencies_involved"], list)
        assert isinstance(d["constraint_context"], dict)
        assert isinstance(d["raw_metrics"], dict)

    def test_to_dict_is_json_serializable(self):
        dc = DiscoveryCandidate(
            discovery_id="uuid-1",
            timestamp="2026-02-28T00:00:00+00:00",
            mycode_version="0.1.1",
            language="javascript",
            dependencies_involved=["express==4.18.0"],
            scenario_category="async_failures",
            expected_behavior="Expected",
            actual_behavior="Actual",
            deviation_factor=1.5,
            load_level_at_discovery="10 users",
            reproducible=False,
            constraint_context={},
            raw_metrics={},
            suggested_template="Template",
        )
        # Should not raise
        result = json.dumps(dc.to_dict())
        assert isinstance(result, str)

    def test_schema_completeness(self):
        """All spec-required fields are present."""
        dc = DiscoveryCandidate(
            discovery_id="id",
            timestamp="ts",
            mycode_version="v",
            language="python",
            dependencies_involved=[],
            scenario_category="cat",
            expected_behavior="exp",
            actual_behavior="act",
            deviation_factor=0.0,
            load_level_at_discovery="lvl",
            reproducible=True,
            constraint_context={},
            raw_metrics={},
            suggested_template="tmpl",
        )
        d = dc.to_dict()
        expected_keys = {
            "discovery_id", "timestamp", "mycode_version", "language",
            "dependencies_involved", "scenario_category",
            "expected_behavior", "actual_behavior", "deviation_factor",
            "load_level_at_discovery", "reproducible",
            "constraint_context", "raw_metrics", "suggested_template",
        }
        assert set(d.keys()) == expected_keys


# ── TestParseTypicalLimit ──


class TestParseTypicalLimit:
    def test_range_string(self):
        assert _parse_typical_limit("10-50") == 50.0

    def test_range_string_with_spaces(self):
        assert _parse_typical_limit("10 - 50") == 50.0

    def test_integer(self):
        assert _parse_typical_limit(150000) == 150000.0

    def test_float(self):
        assert _parse_typical_limit(1.5) == 1.5

    def test_string_number(self):
        assert _parse_typical_limit("1000") == 1000.0

    def test_invalid_string(self):
        assert _parse_typical_limit("unlimited") is None

    def test_none(self):
        assert _parse_typical_limit(None) is None

    def test_empty_string(self):
        assert _parse_typical_limit("") is None

    def test_range_with_decimals(self):
        assert _parse_typical_limit("1.5-3.5") == 3.5


# ── TestExtractLoadLevel ──


class TestExtractLoadLevel:
    def test_from_parameters_concurrent_users(self):
        step = _make_step(params={"concurrent_users": 20})
        sr = _make_scenario_result()
        desc, val = _extract_load_level(sr, step)
        assert val == 20.0
        assert "concurrent_users" in desc

    def test_from_parameters_data_size(self):
        step = _make_step(params={"data_size": 1000})
        sr = _make_scenario_result()
        desc, val = _extract_load_level(sr, step)
        assert val == 1000.0
        assert "data_size" in desc

    def test_from_step_name_pattern(self):
        step = _make_step(name="load_100")
        sr = _make_scenario_result()
        desc, val = _extract_load_level(sr, step)
        assert val == 100.0
        assert "load_100" in desc

    def test_no_load_info(self):
        step = _make_step(name="warmup")
        sr = _make_scenario_result()
        desc, val = _extract_load_level(sr, step)
        assert val is None
        assert desc == "warmup"


# ── TestDetectSuperlinear ──


class TestDetectSuperlinear:
    def test_linear_sequence_not_superlinear(self):
        # 10, 20, 30, 40, 50 — ratios are constant (2.0, 1.5, 1.33, 1.25)
        # Actually decreasing ratios → NOT superlinear
        assert _detect_superlinear([10, 20, 30, 40, 50]) is False

    def test_exponential_sequence_is_superlinear(self):
        # 1, 2, 8, 64 — ratios: 2, 4, 8 — increasing
        assert _detect_superlinear([1, 2, 8, 64]) is True

    def test_too_few_values(self):
        assert _detect_superlinear([10, 20]) is False

    def test_empty_values(self):
        assert _detect_superlinear([]) is False

    def test_constant_values_not_superlinear(self):
        assert _detect_superlinear([5, 5, 5, 5]) is False

    def test_zeros_handled(self):
        assert _detect_superlinear([0, 0, 0]) is False

    def test_mixed_with_zeros_filtered(self):
        # Positive values: 1, 3, 27 — ratios: 3, 9 — increasing
        assert _detect_superlinear([0, 1, 3, 27]) is True


# ── TestCrashAtSafeLevel ──


class TestCrashAtSafeLevel:
    def test_crash_within_safe_range_flagged(self):
        """Crash at load 10 when profile says safe up to 50 → discovery."""
        profile_data = _make_profile_data(
            name="streamlit",
            scaling_limits=[{
                "metric": "concurrent_sessions",
                "typical_limit": "10-50",
            }],
        )
        pm = _make_profile_match("streamlit", profile_data)
        step = _make_step(
            name="load_10",
            params={"concurrent_users": 10},
            error_count=1,
            errors=[{"type": "RuntimeError", "message": "Server crashed"}],
        )
        sr = _make_scenario_result(
            name="concurrency_test",
            category="concurrent_execution",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(
            name="concurrency_test",
            category="concurrent_execution",
            target_deps=["streamlit"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_crash_at_safe_level(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert results[0].scenario_category == "concurrent_execution"
        assert "streamlit" in results[0].expected_behavior

    def test_crash_beyond_safe_range_ignored(self):
        """Crash at load 100 when profile says safe up to 50 → no discovery."""
        profile_data = _make_profile_data(
            name="streamlit",
            scaling_limits=[{
                "metric": "concurrent_sessions",
                "typical_limit": "10-50",
            }],
        )
        pm = _make_profile_match("streamlit", profile_data)
        step = _make_step(
            name="load_100",
            params={"concurrent_users": 100},
            error_count=1,
            errors=[{"type": "RuntimeError", "message": "Server crashed"}],
        )
        sr = _make_scenario_result(
            name="concurrency_test",
            category="concurrent_execution",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(
            name="concurrency_test",
            target_deps=["streamlit"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_crash_at_safe_level(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_no_errors_no_discovery(self):
        """No errors → no discovery."""
        profile_data = _make_profile_data(
            name="flask",
            scaling_limits=[{
                "metric": "concurrent_requests",
                "typical_limit": 100,
            }],
        )
        pm = _make_profile_match("flask", profile_data)
        step = _make_step(name="load_10", params={"concurrent_users": 10})
        sr = _make_scenario_result(steps=[step])
        scenario = _make_scenario(target_deps=["flask"])
        engine = DiscoveryEngine()
        results = engine._detect_crash_at_safe_level(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_known_failure_excluded(self):
        """Crash matching known_failure_modes → excluded."""
        profile_data = _make_profile_data(
            name="streamlit",
            scaling_limits=[{
                "metric": "concurrent_sessions",
                "typical_limit": "10-50",
            }],
            known_failure_modes=[{
                "name": "session_memory_overflow",
                "description": "Sessions use too much memory",
                "trigger_conditions": "Many sessions",
                "severity": "high",
                "versions_affected": "all",
                "detection_hint": "session memory overflow",
            }],
        )
        pm = _make_profile_match("streamlit", profile_data)
        step = _make_step(
            name="load_10",
            params={"concurrent_users": 10},
            error_count=1,
            errors=[{"type": "MemoryError", "message": "session memory overflow detected"}],
        )
        sr = _make_scenario_result(
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(target_deps=["streamlit"])
        engine = DiscoveryEngine()
        results = engine._detect_crash_at_safe_level(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0


# ── TestMemoryGrowthAnomaly ──


class TestMemoryGrowthAnomaly:
    def test_high_growth_flagged(self):
        """Memory growth >2x baseline → discovery."""
        profile_data = _make_profile_data(
            name="pandas",
            baseline_footprint_mb=50.0,
        )
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="step_1", memory_peak_mb=60.0),
            _make_step(name="step_2", memory_peak_mb=100.0),
            _make_step(name="step_3", memory_peak_mb=200.0),
        ]
        sr = _make_scenario_result(
            name="mem_test",
            category="memory_profiling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="mem_test",
            category="memory_profiling",
            target_deps=["pandas"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_memory_growth_anomaly(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert results[0].deviation_factor > 2.0

    def test_normal_growth_not_flagged(self):
        """Memory growth within 2x baseline → no discovery."""
        profile_data = _make_profile_data(
            name="pandas",
            baseline_footprint_mb=100.0,
        )
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="step_1", memory_peak_mb=100.0),
            _make_step(name="step_2", memory_peak_mb=120.0),
            _make_step(name="step_3", memory_peak_mb=140.0),
        ]
        sr = _make_scenario_result(
            name="mem_test",
            category="memory_profiling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="mem_test",
            category="memory_profiling",
            target_deps=["pandas"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_memory_growth_anomaly(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_wrong_category_ignored(self):
        """Non-memory_profiling category → no detection."""
        profile_data = _make_profile_data(name="pandas", baseline_footprint_mb=50.0)
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="s1", memory_peak_mb=50.0),
            _make_step(name="s2", memory_peak_mb=500.0),
        ]
        sr = _make_scenario_result(
            name="scale_test",
            category="data_volume_scaling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="scale_test",
            category="data_volume_scaling",
            target_deps=["pandas"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_memory_growth_anomaly(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0


# ── TestCurveShapeMismatch ──


class TestCurveShapeMismatch:
    def test_exponential_vs_linear_flagged(self):
        """Superlinear growth when profile says linear → discovery."""
        profile_data = _make_profile_data(
            name="pandas",
            growth_pattern="Linear memory growth with data size",
        )
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="s1", memory_peak_mb=10.0),
            _make_step(name="s2", memory_peak_mb=20.0),
            _make_step(name="s3", memory_peak_mb=80.0),
            _make_step(name="s4", memory_peak_mb=640.0),
        ]
        sr = _make_scenario_result(
            name="mem_test",
            category="memory_profiling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="mem_test",
            category="memory_profiling",
            target_deps=["pandas"],
            expected_behavior="Linear memory growth",
        )
        engine = DiscoveryEngine()
        results = engine._detect_curve_shape_mismatch(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert "Superlinear" in results[0].actual_behavior

    def test_linear_vs_linear_not_flagged(self):
        """Linear growth when profile says linear → no discovery."""
        profile_data = _make_profile_data(
            name="pandas",
            growth_pattern="Linear memory growth with data size",
        )
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="s1", memory_peak_mb=10.0),
            _make_step(name="s2", memory_peak_mb=20.0),
            _make_step(name="s3", memory_peak_mb=30.0),
            _make_step(name="s4", memory_peak_mb=40.0),
        ]
        sr = _make_scenario_result(
            name="mem_test",
            category="memory_profiling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="mem_test",
            category="memory_profiling",
            target_deps=["pandas"],
            expected_behavior="Linear memory growth",
        )
        engine = DiscoveryEngine()
        results = engine._detect_curve_shape_mismatch(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_nonlinear_profile_ignored(self):
        """Profile doesn't say 'linear' → no detection attempted."""
        profile_data = _make_profile_data(
            name="chromadb",
            growth_pattern="Grows with embedding index size",
        )
        pm = _make_profile_match("chromadb", profile_data)
        steps = [
            _make_step(name="s1", memory_peak_mb=10.0),
            _make_step(name="s2", memory_peak_mb=20.0),
            _make_step(name="s3", memory_peak_mb=80.0),
            _make_step(name="s4", memory_peak_mb=640.0),
        ]
        sr = _make_scenario_result(
            name="mem_test",
            category="memory_profiling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="mem_test",
            category="memory_profiling",
            target_deps=["chromadb"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_curve_shape_mismatch(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_data_volume_scaling_uses_execution_time(self):
        """data_volume_scaling checks execution_time_ms for curve shape."""
        profile_data = _make_profile_data(
            name="pandas",
            growth_pattern="Linear processing time growth",
        )
        pm = _make_profile_match("pandas", profile_data)
        steps = [
            _make_step(name="s1", execution_time_ms=10.0),
            _make_step(name="s2", execution_time_ms=20.0),
            _make_step(name="s3", execution_time_ms=80.0),
            _make_step(name="s4", execution_time_ms=640.0),
        ]
        sr = _make_scenario_result(
            name="scale_test",
            category="data_volume_scaling",
            steps=steps,
        )
        scenario = _make_scenario(
            name="scale_test",
            category="data_volume_scaling",
            target_deps=["pandas"],
            expected_behavior="Linear scaling",
        )
        engine = DiscoveryEngine()
        results = engine._detect_curve_shape_mismatch(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert "execution time" in results[0].actual_behavior


# ── TestInteractionFailure ──


class TestInteractionFailure:
    def test_unknown_interaction_flagged(self):
        """Failure between 2 deps not in any profile's known_conflicts → discovery."""
        pm1 = _make_profile_match("streamlit", _make_profile_data(name="streamlit"))
        pm2 = _make_profile_match("pandas", _make_profile_data(name="pandas"))
        step = _make_step(
            error_count=1,
            errors=[{"type": "TypeError", "message": "Unexpected type conflict"}],
        )
        sr = _make_scenario_result(
            name="interaction_test",
            category="edge_case_inputs",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(
            name="interaction_test",
            target_deps=["streamlit", "pandas"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_interaction_failure(
            sr, scenario, [pm1, pm2], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert "streamlit" in results[0].expected_behavior
        assert "pandas" in results[0].expected_behavior

    def test_known_conflict_not_flagged(self):
        """Failure matching a known_conflict → not a discovery."""
        pm1_data = _make_profile_data(
            name="streamlit",
            known_conflicts=[{
                "dependency": "asyncio",
                "description": "Event loop conflict",
                "severity": "high",
            }],
        )
        pm1 = _make_profile_match("streamlit", pm1_data)
        pm2 = _make_profile_match("asyncio", _make_profile_data(name="asyncio"))
        step = _make_step(
            error_count=1,
            errors=[{"type": "RuntimeError", "message": "Event loop conflict"}],
        )
        sr = _make_scenario_result(
            name="interaction_test",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(
            name="interaction_test",
            target_deps=["streamlit", "asyncio"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_interaction_failure(
            sr, scenario, [pm1, pm2], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_single_dep_ignored(self):
        """Single-dependency scenario → no interaction detection."""
        pm = _make_profile_match("pandas", _make_profile_data(name="pandas"))
        step = _make_step(error_count=1, errors=[{"message": "fail"}])
        sr = _make_scenario_result(
            name="single_test",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(name="single_test", target_deps=["pandas"])
        engine = DiscoveryEngine()
        results = engine._detect_interaction_failure(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0


# ── TestUnrecognizedDepFailure ──


class TestUnrecognizedDepFailure:
    def test_unrecognized_dep_with_errors_flagged(self):
        """Unrecognized dep (no profile) with errors → discovery."""
        pm = _make_unrecognized_match("scikit-learn")
        step = _make_step(
            error_count=1,
            errors=[{"type": "ImportError", "message": "Module not found"}],
        )
        sr = _make_scenario_result(
            name="sklearn_test",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(
            name="sklearn_test",
            target_deps=["scikit-learn"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_unrecognized_dep_failure(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 1
        assert "scikit-learn" in results[0].expected_behavior

    def test_recognized_dep_not_flagged_here(self):
        """Recognized dep with profile → not an unrecognized discovery."""
        pm = _make_profile_match("pandas", _make_profile_data(name="pandas"))
        step = _make_step(error_count=1, errors=[{"message": "fail"}])
        sr = _make_scenario_result(
            name="pd_test",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(name="pd_test", target_deps=["pandas"])
        engine = DiscoveryEngine()
        results = engine._detect_unrecognized_dep_failure(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_unrecognized_dep_no_errors_not_flagged(self):
        """Unrecognized dep but no errors → no discovery."""
        pm = _make_unrecognized_match("scikit-learn")
        step = _make_step(name="step_1")
        sr = _make_scenario_result(name="sklearn_test", steps=[step])
        scenario = _make_scenario(
            name="sklearn_test",
            target_deps=["scikit-learn"],
        )
        engine = DiscoveryEngine()
        results = engine._detect_unrecognized_dep_failure(
            sr, scenario, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0


# ── TestExclusions ──


class TestExclusions:
    def test_skipped_scenario_excluded(self):
        """Skipped scenarios produce no discoveries."""
        sr = _make_scenario_result(status="skipped")
        scenario = _make_scenario(target_deps=["pandas"])
        pm = _make_profile_match("pandas", _make_profile_data(name="pandas"))
        execution = ExecutionEngineResult(scenario_results=[sr])
        scenarios = ScenarioGeneratorResult(scenarios=[scenario])
        engine = DiscoveryEngine()
        results = engine.analyse(
            execution, scenarios, [pm], None, "python", "0.1.1",
        )
        assert len(results) == 0

    def test_extreme_scale_excluded(self):
        """Crash at >3x user_scale → excluded from crash detection."""
        profile_data = _make_profile_data(
            name="flask",
            scaling_limits=[{
                "metric": "concurrent_requests",
                "typical_limit": 1000,
            }],
        )
        pm = _make_profile_match("flask", profile_data)
        step = _make_step(
            name="load_400",
            params={"concurrent_users": 400},
            error_count=1,
            errors=[{"type": "Error", "message": "Overloaded"}],
        )
        sr = _make_scenario_result(
            name="scale_test",
            steps=[step],
            total_errors=1,
        )
        scenario = _make_scenario(name="scale_test", target_deps=["flask"])
        constraints = OperationalConstraints(user_scale=100)
        engine = DiscoveryEngine()
        # 400 > 100 * 3 = 300, so excluded
        results = engine._detect_crash_at_safe_level(
            sr, scenario, [pm], constraints, "python", "0.1.1",
        )
        assert len(results) == 0


# ── TestSaveDiscoveries ──


class TestSaveDiscoveries:
    def test_save_writes_json_files(self, tmp_path):
        """Each discovery is written as a separate JSON file."""
        disc_dir = tmp_path / "discoveries"
        engine = DiscoveryEngine(discoveries_dir=disc_dir)

        dc = DiscoveryCandidate(
            discovery_id="test-save-uuid",
            timestamp="2026-02-28T00:00:00+00:00",
            mycode_version="0.1.1",
            language="python",
            dependencies_involved=["pandas==2.2.1"],
            scenario_category="memory_profiling",
            expected_behavior="Linear growth",
            actual_behavior="Exponential growth",
            deviation_factor=4.8,
            load_level_at_discovery="14 sessions",
            reproducible=True,
            constraint_context={"user_scale": 200},
            raw_metrics={"mem": 100},
            suggested_template="Template",
        )

        paths = engine.save([dc])
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].name == "test-save-uuid.json"

        data = json.loads(paths[0].read_text())
        assert data["discovery_id"] == "test-save-uuid"
        assert data["language"] == "python"
        assert data["deviation_factor"] == 4.8

    def test_save_creates_directory(self, tmp_path):
        """Directory is created lazily on first save."""
        disc_dir = tmp_path / "nested" / "discoveries"
        assert not disc_dir.exists()

        engine = DiscoveryEngine(discoveries_dir=disc_dir)
        dc = DiscoveryCandidate(
            discovery_id="uuid-2",
            timestamp="ts",
            mycode_version="v",
            language="python",
            dependencies_involved=[],
            scenario_category="cat",
            expected_behavior="exp",
            actual_behavior="act",
            deviation_factor=0.0,
            load_level_at_discovery="lvl",
            reproducible=True,
            constraint_context={},
            raw_metrics={},
            suggested_template="tmpl",
        )
        paths = engine.save([dc])
        assert disc_dir.exists()
        assert len(paths) == 1

    def test_save_empty_list_returns_empty(self, tmp_path):
        """Saving empty list does not create directory."""
        disc_dir = tmp_path / "should_not_exist"
        engine = DiscoveryEngine(discoveries_dir=disc_dir)
        paths = engine.save([])
        assert paths == []
        assert not disc_dir.exists()

    def test_save_multiple_discoveries(self, tmp_path):
        """Multiple discoveries produce multiple files."""
        disc_dir = tmp_path / "discoveries"
        engine = DiscoveryEngine(discoveries_dir=disc_dir)

        discoveries = [
            DiscoveryCandidate(
                discovery_id=f"uuid-{i}",
                timestamp="ts",
                mycode_version="v",
                language="python",
                dependencies_involved=[],
                scenario_category="cat",
                expected_behavior="exp",
                actual_behavior="act",
                deviation_factor=0.0,
                load_level_at_discovery="lvl",
                reproducible=True,
                constraint_context={},
                raw_metrics={},
                suggested_template="tmpl",
            )
            for i in range(3)
        ]
        paths = engine.save(discoveries)
        assert len(paths) == 3
        assert all(p.exists() for p in paths)


# ── TestAnalyseIntegration ──


class TestAnalyseIntegration:
    def test_mixed_scenario_results(self, tmp_path):
        """End-to-end: mix of discoverable and non-discoverable results."""
        disc_dir = tmp_path / "discoveries"
        engine = DiscoveryEngine(discoveries_dir=disc_dir)

        # Profile: streamlit with safe range 10-50
        profile_data = _make_profile_data(
            name="streamlit",
            baseline_footprint_mb=80.0,
            scaling_limits=[{
                "metric": "concurrent_sessions",
                "typical_limit": "10-50",
            }],
            growth_pattern="Linear memory growth per session",
        )
        pm_streamlit = _make_profile_match("streamlit", profile_data, "1.41.0")

        # Scenario 1: crash at safe level → discovery
        crash_step = _make_step(
            name="load_5",
            params={"concurrent_users": 5},
            error_count=1,
            errors=[{"type": "RuntimeError", "message": "Unexpected crash"}],
        )
        crash_sr = _make_scenario_result(
            name="crash_scenario",
            category="concurrent_execution",
            steps=[crash_step],
            total_errors=1,
        )
        crash_scenario = _make_scenario(
            name="crash_scenario",
            category="concurrent_execution",
            target_deps=["streamlit"],
        )

        # Scenario 2: clean run → no discovery
        clean_step = _make_step(name="load_10", params={"concurrent_users": 10})
        clean_sr = _make_scenario_result(
            name="clean_scenario",
            category="concurrent_execution",
            steps=[clean_step],
        )
        clean_scenario = _make_scenario(
            name="clean_scenario",
            category="concurrent_execution",
            target_deps=["streamlit"],
        )

        # Scenario 3: skipped → no discovery
        skipped_sr = _make_scenario_result(
            name="skipped_scenario",
            status="skipped",
        )
        skipped_scenario = _make_scenario(name="skipped_scenario")

        execution = ExecutionEngineResult(
            scenario_results=[crash_sr, clean_sr, skipped_sr],
        )
        scenarios = ScenarioGeneratorResult(
            scenarios=[crash_scenario, clean_scenario, skipped_scenario],
        )
        constraints = OperationalConstraints(user_scale=20)

        discoveries = engine.analyse(
            execution, scenarios, [pm_streamlit],
            constraints, "python", "0.1.1",
        )

        assert len(discoveries) >= 1
        # The crash at 5 (within safe 50) should be found
        crash_disc = [
            d for d in discoveries
            if d.scenario_category == "concurrent_execution"
            and "Crash" in d.actual_behavior
        ]
        assert len(crash_disc) == 1

        # Save and verify
        paths = engine.save(discoveries)
        assert len(paths) == len(discoveries)
        assert all(p.exists() for p in paths)


class TestNoDiscoveries:
    def test_clean_run_produces_no_discoveries(self):
        """A clean run with no errors produces zero discoveries."""
        profile_data = _make_profile_data(name="flask")
        pm = _make_profile_match("flask", profile_data)

        step = _make_step(name="load_10", params={"concurrent_users": 10})
        sr = _make_scenario_result(
            name="clean_test",
            category="concurrent_execution",
            steps=[step],
        )
        scenario = _make_scenario(
            name="clean_test",
            category="concurrent_execution",
            target_deps=["flask"],
        )

        execution = ExecutionEngineResult(scenario_results=[sr])
        scenarios = ScenarioGeneratorResult(scenarios=[scenario])

        engine = DiscoveryEngine()
        discoveries = engine.analyse(
            execution, scenarios, [pm], None, "python", "0.1.1",
        )
        assert discoveries == []


# ── TestHelpers ──


class TestBuildDepStrings:
    def test_with_version(self):
        pm = ProfileMatch(
            dependency_name="pandas",
            profile=None,
            installed_version="2.2.1",
        )
        scenario = _make_scenario(target_deps=["pandas"])
        result = _build_dep_strings(scenario, [pm])
        assert result == ["pandas==2.2.1"]

    def test_without_version(self):
        pm = ProfileMatch(
            dependency_name="pandas",
            profile=None,
            installed_version=None,
        )
        scenario = _make_scenario(target_deps=["pandas"])
        result = _build_dep_strings(scenario, [pm])
        assert result == ["pandas"]


class TestConstraintsToDict:
    def test_none_constraints(self):
        assert _constraints_to_dict(None) == {}

    def test_populated_constraints(self):
        c = OperationalConstraints(user_scale=20, data_type="tabular")
        d = _constraints_to_dict(c)
        assert d["user_scale"] == 20
        assert d["data_type"] == "tabular"
        assert d["usage_pattern"] is None


class TestIsKnownFailure:
    def test_matching_detection_hint(self):
        profile = _make_profile(_make_profile_data(
            known_failure_modes=[{
                "name": "oom",
                "detection_hint": "MemoryError",
                "description": "OOM",
                "trigger_conditions": "",
                "severity": "critical",
                "versions_affected": "all",
            }],
        ))
        errors = [{"type": "MemoryError", "message": "out of memory"}]
        assert _is_known_failure(errors, profile) is True

    def test_no_match(self):
        profile = _make_profile(_make_profile_data(
            known_failure_modes=[{
                "name": "oom",
                "detection_hint": "MemoryError",
                "description": "OOM",
                "trigger_conditions": "",
                "severity": "critical",
                "versions_affected": "all",
            }],
        ))
        errors = [{"type": "TypeError", "message": "wrong type"}]
        assert _is_known_failure(errors, profile) is False

    def test_empty_errors(self):
        profile = _make_profile(_make_profile_data(
            known_failure_modes=[{
                "name": "oom",
                "detection_hint": "MemoryError",
                "description": "OOM",
                "trigger_conditions": "",
                "severity": "critical",
                "versions_affected": "all",
            }],
        ))
        assert _is_known_failure([], profile) is False
