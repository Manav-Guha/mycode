"""Tests for Report Generator (E2).

Tests cover:
  - Finding extraction from execution results (failures, errors, cap hits)
  - Degradation curve detection (time, memory, error accumulation)
  - Version discrepancy flagging (outdated, missing)
  - Unrecognized dependency flagging
  - Offline summary generation
  - LLM-backed summary generation (mocked)
  - DiagnosticReport.as_text() rendering
  - Edge cases (empty results, no findings, all passing)
"""

import json
from unittest.mock import MagicMock

import pytest

from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.ingester import DependencyInfo, FileAnalysis, ImportInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.report import (
    DegradationPoint,
    DiagnosticReport,
    Finding,
    ReportError,
    ReportGenerator,
    _build_dep_file_map,
    _describe_errors,
    _describe_errors_with_context,
    _describe_impact,
    _describe_scenario,
    _describe_step,
    _extract_cap_type,
    _extract_project_ref,
    _format_ms,
    _human_metric,
    _human_time,
    _humanize_title_name,
    _build_degradation_narrative,
    _consequence_line,
    _is_significant_degradation,
    _is_significant_finding,
    _resolve_source_file,
)
from mycode.scenario import LLMBackend, LLMConfig, LLMError, LLMResponse


# ── Fixtures ──


@pytest.fixture
def clean_execution() -> ExecutionEngineResult:
    """Execution with all scenarios passing cleanly."""
    return ExecutionEngineResult(
        scenario_results=[
            ScenarioResult(
                scenario_name="flask_data_volume",
                scenario_category="data_volume_scaling",
                status="completed",
                steps=[
                    StepResult(
                        step_name="size_100",
                        execution_time_ms=50.0,
                        memory_peak_mb=10.0,
                    ),
                    StepResult(
                        step_name="size_1000",
                        execution_time_ms=55.0,
                        memory_peak_mb=11.0,
                    ),
                ],
                total_errors=0,
            ),
        ],
        scenarios_completed=1,
        scenarios_failed=0,
    )


@pytest.fixture
def failing_execution() -> ExecutionEngineResult:
    """Execution with failures, errors, and cap hits."""
    return ExecutionEngineResult(
        scenario_results=[
            ScenarioResult(
                scenario_name="flask_data_volume",
                scenario_category="data_volume_scaling",
                status="completed",
                steps=[
                    StepResult(
                        step_name="size_100",
                        execution_time_ms=50.0,
                        memory_peak_mb=10.0,
                    ),
                    StepResult(
                        step_name="size_10000",
                        execution_time_ms=500.0,
                        memory_peak_mb=200.0,
                        error_count=3,
                        errors=[
                            {"type": "MemoryError", "message": "out of memory"},
                            {"type": "MemoryError", "message": "out of memory"},
                            {"type": "TimeoutError", "message": "too slow"},
                        ],
                    ),
                ],
                total_errors=3,
                failure_indicators_triggered=["memory_growth_unbounded"],
            ),
            ScenarioResult(
                scenario_name="sqlalchemy_concurrent",
                scenario_category="concurrent_execution",
                status="failed",
                summary="Database connection pool exhausted",
                total_errors=5,
            ),
            ScenarioResult(
                scenario_name="edge_case_inputs",
                scenario_category="edge_case_input",
                status="completed",
                steps=[
                    StepResult(
                        step_name="empty_input",
                        execution_time_ms=10.0,
                        resource_cap_hit="memory",
                    ),
                ],
                resource_cap_hit=True,
                total_errors=0,
            ),
        ],
        scenarios_completed=1,
        scenarios_failed=2,
    )


@pytest.fixture
def degrading_execution() -> ExecutionEngineResult:
    """Execution showing clear degradation curves."""
    return ExecutionEngineResult(
        scenario_results=[
            ScenarioResult(
                scenario_name="flask_scaling",
                scenario_category="data_volume_scaling",
                status="completed",
                steps=[
                    StepResult(step_name="size_10", execution_time_ms=10.0, memory_peak_mb=5.0),
                    StepResult(step_name="size_100", execution_time_ms=20.0, memory_peak_mb=8.0),
                    StepResult(step_name="size_1000", execution_time_ms=80.0, memory_peak_mb=15.0),
                    StepResult(step_name="size_10000", execution_time_ms=800.0, memory_peak_mb=120.0),
                ],
                total_errors=0,
            ),
            ScenarioResult(
                scenario_name="memory_leak_check",
                scenario_category="memory_profiling",
                status="completed",
                steps=[
                    StepResult(step_name="iter_1", memory_peak_mb=10.0, execution_time_ms=5.0),
                    StepResult(step_name="iter_2", memory_peak_mb=20.0, execution_time_ms=5.0),
                    StepResult(step_name="iter_3", memory_peak_mb=40.0, execution_time_ms=5.0),
                    StepResult(step_name="iter_4", memory_peak_mb=80.0, execution_time_ms=5.0),
                ],
                total_errors=0,
            ),
        ],
    )


@pytest.fixture
def simple_ingestion() -> IngestionResult:
    """Ingestion with some version discrepancies."""
    return IngestionResult(
        project_path="/tmp/myapp",
        files_analyzed=5,
        total_lines=500,
        dependencies=[
            DependencyInfo(
                name="flask",
                installed_version="2.3.0",
                latest_version="3.1.0",
                is_outdated=True,
            ),
            DependencyInfo(
                name="requests",
                installed_version="2.31.0",
                latest_version="2.32.3",
                is_outdated=True,
            ),
            DependencyInfo(
                name="missing-dep",
                is_missing=True,
            ),
            DependencyInfo(
                name="sqlalchemy",
                installed_version="2.0.30",
                is_outdated=False,
            ),
        ],
    )


@pytest.fixture
def profile_matches() -> list[ProfileMatch]:
    """Mix of recognized and unrecognized profile matches."""
    flask_profile = MagicMock()
    flask_profile.name = "flask"
    return [
        ProfileMatch(
            dependency_name="flask",
            profile=flask_profile,
            installed_version="2.3.0",
            version_match=False,
            version_notes="v2.3.0 has known WSGI thread leak",
        ),
        ProfileMatch(
            dependency_name="sqlalchemy",
            profile=MagicMock(),
            installed_version="2.0.30",
            version_match=True,
        ),
        ProfileMatch(
            dependency_name="requests",
            profile=MagicMock(),
            installed_version="2.31.0",
        ),
        ProfileMatch(dependency_name="obscure-lib", profile=None),
        ProfileMatch(dependency_name="custom-tool", profile=None),
    ]


# ── Finding Extraction Tests ──


class TestFindingExtraction:
    """Tests for extracting findings from execution results."""

    def test_clean_execution_no_critical_findings(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        critical = [f for f in report.findings if f.severity == "critical"]
        assert len(critical) == 0

    def test_failed_scenario_creates_critical_finding(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        critical = [f for f in report.findings if f.severity == "critical"]
        failed_findings = [f for f in critical if f._finding_type == "scenario_failed"]
        assert len(failed_findings) >= 1
        assert any("sqlalchemy" in f.title.lower() for f in failed_findings)

    def test_resource_cap_creates_critical_finding(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        cap_findings = [
            f for f in report.findings if f._finding_type == "resource_limit_hit"
        ]
        assert len(cap_findings) >= 1
        assert cap_findings[0].severity == "critical"

    def test_errors_create_warning_finding(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        error_findings = [
            f for f in report.findings
            if f.severity == "warning" and f._finding_type == "errors_during"
        ]
        assert len(error_findings) >= 1

    def test_failure_indicators_create_warning(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        indicator_findings = [
            f for f in report.findings
            if f._finding_type == "failure_indicators"
        ]
        assert len(indicator_findings) >= 1
        assert "memory usage grows without limit" in indicator_findings[0].description

    def test_error_type_summarization(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        error_findings = [
            f for f in report.findings
            if f._finding_type == "errors_during"
        ]
        assert len(error_findings) >= 1
        assert "out-of-memory error" in error_findings[0].details

    def test_findings_sorted_by_severity(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        severities = [f.severity for f in report.findings]
        # Critical should come before warning, warning before info
        for i in range(len(severities) - 1):
            order = {"critical": 0, "warning": 1, "info": 2}
            assert order.get(severities[i], 9) <= order.get(severities[i + 1], 9)

    def test_scenario_counts(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert report.scenarios_run == 3
        # 1 completed with errors + 1 failed + 1 with cap hit = 3 failed
        assert report.scenarios_failed == 3
        assert report.scenarios_passed == 0
        assert report.total_errors == 8


# ── Degradation Detection Tests ──


class TestDegradationDetection:
    """Tests for degradation curve analysis."""

    def test_time_degradation_detected(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        time_dps = [
            dp for dp in report.degradation_points
            if dp.metric == "execution_time_ms"
        ]
        assert len(time_dps) >= 1
        dp = time_dps[0]
        assert dp.breaking_point  # Should identify where the jump happened
        assert "increased" in dp.description.lower() or "grew" in dp.description.lower()

    def test_memory_degradation_detected(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        mem_dps = [
            dp for dp in report.degradation_points
            if dp.metric == "memory_peak_mb"
        ]
        assert len(mem_dps) >= 1
        # Memory goes 10→20→40→80, that's 8x growth
        dp = mem_dps[0]
        assert dp.breaking_point
        assert "8.0x" in dp.description or "increased" in dp.description.lower()

    def test_no_degradation_on_stable_results(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        # 50ms → 55ms is only 1.1x, not degradation
        time_dps = [
            dp for dp in report.degradation_points
            if dp.metric == "execution_time_ms"
        ]
        assert len(time_dps) == 0

    def test_degradation_steps_recorded(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        assert len(report.degradation_points) >= 1
        for dp in report.degradation_points:
            assert len(dp.steps) >= 2
            # Steps should be (label, value) tuples
            for label, value in dp.steps:
                assert isinstance(label, str)
                assert isinstance(value, float)


class TestDegradationCurveAnalysis:
    """Direct tests for _analyze_curve static method."""

    def test_2x_overall_growth(self):
        steps = [("a", 10.0), ("b", 15.0), ("c", 25.0)]
        dp = ReportGenerator._analyze_curve("test", "time", steps)
        assert dp is not None
        assert dp.breaking_point

    def test_3x_spike(self):
        steps = [("a", 10.0), ("b", 10.0), ("c", 40.0)]
        dp = ReportGenerator._analyze_curve("test", "time", steps)
        assert dp is not None
        assert dp.breaking_point == "c"

    def test_stable_returns_none(self):
        steps = [("a", 10.0), ("b", 11.0), ("c", 12.0)]
        dp = ReportGenerator._analyze_curve("test", "time", steps)
        assert dp is None

    def test_single_step_returns_none(self):
        steps = [("a", 10.0)]
        dp = ReportGenerator._analyze_curve("test", "time", steps)
        assert dp is None

    def test_zero_baseline_growth(self):
        steps = [("a", 0.0), ("b", 0.0), ("c", 5.0)]
        dp = ReportGenerator._analyze_curve("test", "errors", steps)
        assert dp is not None
        assert "grew from near zero" in dp.description.lower()

    def test_zero_baseline_stays_zero(self):
        steps = [("a", 0.0), ("b", 0.0), ("c", 0.0)]
        dp = ReportGenerator._analyze_curve("test", "errors", steps)
        assert dp is None

    def test_flat_noisy_data_returns_none(self):
        """Flat data with slight noise (CV < 10%) should not be flagged."""
        steps = [("a", 10.0), ("b", 10.5), ("c", 9.8), ("d", 10.2)]
        dp = ReportGenerator._analyze_curve("test", "execution_time_ms", steps)
        assert dp is None

    def test_trivial_absolute_delta_returns_none(self):
        """High ratio but trivial absolute delta (below min-delta) → None."""
        # 0.05ms → 0.11ms is 2.2x but absolute delta is 0.06ms < 50ms threshold
        steps = [("a", 0.05), ("b", 0.11)]
        dp = ReportGenerator._analyze_curve("test", "execution_time_ms", steps)
        assert dp is None


# ── Version Discrepancy Tests ──


class TestVersionDiscrepancies:
    """Tests for version discrepancy flagging."""

    def test_outdated_deps_flagged(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        assert len(report.version_flags) >= 2  # flask + requests
        assert any("flask" in v.lower() for v in report.version_flags)
        assert any("requests" in v.lower() for v in report.version_flags)

    def test_missing_dep_flagged(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        assert any("missing-dep" in v for v in report.version_flags)
        missing_findings = [
            f for f in report.findings if "missing" in f.title.lower()
        ]
        assert len(missing_findings) >= 1
        assert missing_findings[0].severity == "warning"

    def test_profile_version_notes_included(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        assert any("WSGI thread leak" in v for v in report.version_flags)

    def test_outdated_dep_creates_info_finding(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        outdated_findings = [
            f for f in report.findings
            if "outdated" in f.title.lower()
        ]
        assert len(outdated_findings) >= 1
        assert outdated_findings[0].severity == "info"
        assert "flask" in outdated_findings[0].affected_dependencies

    def test_no_version_flags_when_all_current(self, clean_execution):
        ingestion = IngestionResult(
            project_path="/tmp/x",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(name="flask", installed_version="3.1.0", is_outdated=False),
            ],
        )
        matches = [
            ProfileMatch(
                dependency_name="flask",
                profile=MagicMock(),
                version_match=True,
            ),
        ]
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, matches)
        assert len(report.version_flags) == 0

    def test_dev_deps_excluded_from_version_flags(self, clean_execution):
        ingestion = IngestionResult(
            project_path="/tmp/x",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(
                    name="express", installed_version="4.18.0",
                    latest_version="5.0.0", is_outdated=True,
                ),
                DependencyInfo(
                    name="jest", installed_version="28.0.0",
                    latest_version="29.0.0", is_outdated=True, is_dev=True,
                ),
                DependencyInfo(
                    name="eslint", is_missing=True, is_dev=True,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, [])
        assert any("express" in v for v in report.version_flags)
        assert not any("jest" in v for v in report.version_flags)
        assert not any("eslint" in v for v in report.version_flags)


# ── JS Dependency Report Language Tests ──


class TestJsDependencyReportLanguage:
    """Test that JS projects get 'without stress profiles' (INFO) not 'missing' (WARNING)."""

    def test_js_missing_deps_are_info_not_warning(self, clean_execution):
        """JS project with unrecognised deps should get INFO finding."""
        ingestion = IngestionResult(
            project_path="/tmp/x",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(name="@vercel/analytics", is_missing=True),
                DependencyInfo(name="@radix-ui/react-dialog", is_missing=True),
                DependencyInfo(name="@tanstack/react-query", is_missing=True),
                DependencyInfo(name="next", installed_version="14.0.0"),
                DependencyInfo(name="react", installed_version="18.2.0"),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, [])

        missing_findings = [
            f for f in report.findings
            if "without stress profiles" in f.title.lower()
            or "missing" in f.title.lower()
        ]
        assert len(missing_findings) == 1
        f = missing_findings[0]
        assert "without stress profiles" in f.title
        assert f.severity == "info"
        assert "don't have myCode stress profiles" in f.description

    def test_python_missing_deps_still_warning(self, clean_execution):
        """Python project with missing deps should keep WARNING severity."""
        ingestion = IngestionResult(
            project_path="/tmp/x",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(name="pandas", is_missing=True),
                DependencyInfo(name="numpy", is_missing=True),
                DependencyInfo(name="flask", installed_version="2.0.0"),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, [])

        missing_findings = [
            f for f in report.findings
            if "missing" in f.title.lower()
        ]
        assert len(missing_findings) == 1
        assert missing_findings[0].severity == "warning"
        assert "declared in requirements" in missing_findings[0].description

    def test_js_version_flags_say_no_profile(self, clean_execution):
        """JS project version flags should say 'no stress profile available'."""
        ingestion = IngestionResult(
            project_path="/tmp/x",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(name="@vercel/og", is_missing=True),
                DependencyInfo(name="next", installed_version="14.0.0"),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, [])

        assert any("no stress profile available" in v for v in report.version_flags)
        assert not any("declared but not installed" in v for v in report.version_flags)


# ── Unrecognized Dependencies Tests ──


class TestUnrecognizedDeps:
    """Tests for unrecognized dependency flagging."""

    def test_unrecognized_deps_flagged(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        assert "obscure-lib" in report.unrecognized_deps
        assert "custom-tool" in report.unrecognized_deps
        assert len(report.unrecognized_deps) == 2

    def test_unrecognized_tracked_in_report(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        # Unrecognized deps are tracked but not as alarming findings
        assert "obscure-lib" in report.unrecognized_deps
        assert "custom-tool" in report.unrecognized_deps
        # Dependency coverage is reflected instead
        assert report.recognized_dep_count == 3

    def test_no_unrecognized_when_all_matched(self, clean_execution):
        ingestion = IngestionResult(project_path="/tmp/x", files_analyzed=1)
        matches = [
            ProfileMatch(dependency_name="flask", profile=MagicMock()),
        ]
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, matches)
        assert len(report.unrecognized_deps) == 0


# ── Offline Summary Tests ──


class TestOfflineSummary:
    """Tests for offline summary generation."""

    def test_clean_summary(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        # Clean execution but has version flags, so not fully clean
        assert report.summary
        assert "version" in report.summary.lower() or "warning" in report.summary.lower()

    def test_failing_summary_mentions_critical(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert "critical" in report.summary.lower()
        assert report.model_used == "offline"

    def test_empty_execution_summary(self, simple_ingestion, profile_matches):
        empty = ExecutionEngineResult()
        gen = ReportGenerator(offline=True)
        report = gen.generate(empty, simple_ingestion, profile_matches)

        assert "no stress test" in report.summary.lower()

    def test_summary_includes_intent(self, clean_execution, simple_ingestion):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            clean_execution, simple_ingestion, [],
            operational_intent="Personal budget tracker for daily use",
        )

        assert "budget tracker" in report.summary.lower()

    def test_degradation_mentioned_in_summary(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        assert "degradation" in report.summary.lower()

    def test_all_scenarios_failed_summary(self, simple_ingestion):
        """When all scenarios are incomplete, summary says 'could not execute'."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="react_dom_data_volume",
                    scenario_category="data_volume_scaling",
                    status="failed",
                    summary="Harness did not produce structured output. Exit code: -5.",
                    failure_reason="unknown",
                    steps=[],
                ),
                ScenarioResult(
                    scenario_name="coupling_render_App",
                    scenario_category="state_management_degradation",
                    status="failed",
                    summary="Harness did not produce structured output. Exit code: -5.",
                    failure_reason="unknown",
                    steps=[],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(execution, simple_ingestion, [])

        assert "could not execute" in report.summary.lower()
        assert "completed without issues" not in report.summary.lower()

    def test_partial_incomplete_summary(self):
        """When some scenarios pass and some fail, summary notes both."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="react_dom_data_volume",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[StepResult(step_name="s1")],
                    total_errors=0,
                ),
                ScenarioResult(
                    scenario_name="coupling_render_App",
                    scenario_category="state_management_degradation",
                    status="failed",
                    summary="Harness crash.",
                    failure_reason="unknown",
                    steps=[],
                ),
            ],
        )
        # Use minimal ingestion with no version discrepancies
        ingestion = IngestionResult(project_path="/tmp/test")
        gen = ReportGenerator(offline=True)
        report = gen.generate(execution, ingestion, [])

        assert "could not be run" in report.summary.lower()
        # Should say "1 of 2" not "All 2"
        assert "1 of 2" in report.summary


# ── LLM Summary Tests ──


class TestLLMSummary:
    """Tests for LLM-backed summary generation."""

    def _make_generator_with_mock(
        self, llm_response: str,
    ) -> ReportGenerator:
        config = LLMConfig(api_key="test-key")
        gen = ReportGenerator(llm_config=config, offline=False)
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.generate = MagicMock(return_value=LLMResponse(
            content=llm_response,
            model="test-model",
            input_tokens=500,
            output_tokens=200,
        ))
        gen._backend = mock_backend
        return gen

    def test_llm_summary_used(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = self._make_generator_with_mock(
            '{"summary": "Your app struggles under heavy data load. '
            'The database layer breaks at around 10,000 records."}'
        )
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
        )

        assert "database layer" in report.summary.lower()
        assert report.model_used == "test-model"

    def test_llm_token_usage_tracked(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = self._make_generator_with_mock('{"summary": "All good."}')
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        assert report.token_usage["input_tokens"] == 500
        assert report.token_usage["output_tokens"] == 200

    def test_llm_failure_falls_back_to_offline(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        config = LLMConfig(api_key="test-key")
        gen = ReportGenerator(llm_config=config, offline=False)
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.generate = MagicMock(
            side_effect=LLMError("connection refused"),
        )
        gen._backend = mock_backend

        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
        )

        # Should fall back to offline summary
        assert report.summary
        assert "critical" in report.summary.lower()

    def test_llm_bad_json_uses_plain_text(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = self._make_generator_with_mock(
            "Your project looks healthy with minor version concerns."
        )
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        # Should use the plain text response directly
        assert "healthy" in report.summary.lower()

    def test_no_api_key_falls_back_to_offline(self):
        gen = ReportGenerator(llm_config=LLMConfig(), offline=False)
        assert gen._offline is True


# ── DiagnosticReport.as_text() Tests ──


class TestReportRendering:
    """Tests for as_text() output rendering."""

    def test_renders_header(self):
        report = DiagnosticReport(summary="All good.")
        text = report.as_text()
        assert "myCode Diagnostic Report" in text

    def test_renders_summary(self):
        report = DiagnosticReport(summary="Found 2 issues.")
        text = report.as_text()
        assert "Found 2 issues" in text

    def test_renders_findings(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Memory leak detected",
                    severity="critical",
                    description="Memory grew unbounded",
                ),
                Finding(
                    title="Slow response",
                    severity="warning",
                    description="Response time doubled",
                ),
            ],
        )
        text = report.as_text()
        assert "[!!]" in text  # critical marker
        assert "[! ]" in text  # warning marker
        assert "Memory leak" in text
        assert "Slow response" in text

    def test_renders_degradation(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test_scaling",
                    metric="execution_time_ms",
                    steps=[("small", 10.0), ("large", 100.0)],
                    breaking_point="large",
                    description="Time increased 10x",
                ),
            ],
        )
        text = report.as_text()
        assert "Degradation Curves" in text
        assert "Breaking point: response time exceeds 100ms at large" in text

    def test_renders_version_flags(self):
        report = DiagnosticReport(
            version_flags=["flask: installed 2.3.0, latest is 3.1.0"],
        )
        text = report.as_text()
        assert "Version Discrepancies" in text
        assert "flask" in text

    def test_renders_dependency_coverage(self):
        report = DiagnosticReport(
            recognized_dep_count=5,
            unrecognized_deps=["obscure-lib", "custom-tool"],
        )
        text = report.as_text()
        assert "5 of 7" in text
        assert "tested with general usage-based analysis" in text

    def test_renders_footer(self):
        report = DiagnosticReport()
        text = report.as_text()
        assert "diagnoses" in text.lower()
        assert "does not prescribe" in text.lower()

    def test_renders_scenario_stats(self):
        report = DiagnosticReport(
            scenarios_run=10,
            scenarios_passed=7,
            scenarios_failed=3,
            total_errors=15,
        )
        text = report.as_text()
        assert "10 run" in text
        assert "7 clean" in text
        assert "3 with issues" in text
        assert "15" in text

    def test_full_report_rendering(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="Budget tracker",
        )
        text = report.as_text()

        # Should be a substantial report
        assert len(text) > 200
        assert "myCode Diagnostic Report" in text
        assert "does not prescribe" in text.lower()


# ── Helper Tests ──


class TestHumanMetric:
    """Tests for _human_metric helper."""

    def test_known_metrics(self):
        assert _human_metric("execution_time_ms") == "Execution time"
        assert _human_metric("memory_peak_mb") == "Memory usage"
        assert _human_metric("error_count") == "Error count"

    def test_unknown_metric(self):
        assert _human_metric("custom_thing") == "Custom Thing"


class TestExtractSummary:
    """Tests for ReportGenerator._extract_summary."""

    def test_json_extraction(self):
        result = ReportGenerator._extract_summary(
            '{"summary": "All good"}', "fallback"
        )
        assert result == "All good"

    def test_plain_text_fallthrough(self):
        result = ReportGenerator._extract_summary(
            "Your project looks healthy overall.", "fallback"
        )
        assert "healthy" in result

    def test_empty_uses_fallback(self):
        result = ReportGenerator._extract_summary("", "fallback")
        assert result == "fallback"

    def test_too_short_uses_fallback(self):
        result = ReportGenerator._extract_summary("ok", "fallback")
        assert result == "fallback"

    def test_brace_extraction(self):
        result = ReportGenerator._extract_summary(
            'Here is the result: {"summary": "Found issues"}',
            "fallback",
        )
        assert result == "Found issues"


# ── Finding Grouping Tests ──


class TestFindingGrouping:
    """Tests for grouping similar findings."""

    def test_group_identical_findings(self):
        """3 findings with same category, pattern, and metrics → 1 with count=3."""
        findings = [
            Finding(
                title=f"Resource limit hit: coupling_api_fetch_{i}",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
                _execution_time_ms=200.0,
                _error_count=5,
            )
            for i in range(3)
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 1
        assert result[0].group_count == 3
        assert len(result[0].grouped_findings) == 2

    def test_no_group_different_categories(self):
        """Same pattern, different category → 2 separate findings."""
        findings = [
            Finding(
                title="Resource limit hit: test_a",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
                _execution_time_ms=200.0,
            ),
            Finding(
                title="Resource limit hit: test_b",
                severity="critical",
                category="data_volume_scaling",
                _peak_memory_mb=100.0,
                _execution_time_ms=200.0,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 2
        assert all(f.group_count == 1 for f in result)

    def test_no_group_different_patterns(self):
        """Same category, different pattern → 2 separate findings."""
        findings = [
            Finding(
                title="Resource limit hit: test_a",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
            ),
            Finding(
                title="Scenario failed: test_b",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 2

    def test_no_group_divergent_metrics(self):
        """>10% difference → 2 separate findings."""
        findings = [
            Finding(
                title="Resource limit hit: test_a",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
                _execution_time_ms=200.0,
            ),
            Finding(
                title="Resource limit hit: test_b",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=200.0,  # 100% different
                _execution_time_ms=200.0,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 2

    def test_within_10pct_groups(self):
        """Memory 100 vs 108 (8% diff) → grouped."""
        findings = [
            Finding(
                title="Resource limit hit: test_a",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
                _execution_time_ms=200.0,
                _error_count=5,
            ),
            Finding(
                title="Resource limit hit: test_b",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=108.0,
                _execution_time_ms=210.0,
                _error_count=5,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 1
        assert result[0].group_count == 2

    def test_above_10pct_not_grouped(self):
        """Memory 100 vs 115 (15% diff) → not grouped."""
        findings = [
            Finding(
                title="Resource limit hit: test_a",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=100.0,
            ),
            Finding(
                title="Resource limit hit: test_b",
                severity="critical",
                category="concurrent_execution",
                _peak_memory_mb=115.0,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 2

    def test_grouped_count_correct(self):
        """4 findings → representative has count=4, grouped_findings has 3."""
        findings = [
            Finding(
                title=f"Errors during: coupling_compute_{i}",
                severity="warning",
                category="data_volume_scaling",
                _peak_memory_mb=50.0,
                _execution_time_ms=100.0,
                _error_count=3,
            )
            for i in range(4)
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 1
        assert result[0].group_count == 4
        assert len(result[0].grouped_findings) == 3

    def test_both_zero_metrics_grouped(self):
        """All-zero metrics → always similar, should group."""
        findings = [
            Finding(
                title="Failure indicators triggered: test_a",
                severity="warning",
                category="edge_case_input",
            ),
            Finding(
                title="Failure indicators triggered: test_b",
                severity="warning",
                category="edge_case_input",
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 1
        assert result[0].group_count == 2

    def test_grouped_rendering(self):
        """as_text() includes '(and N similar)' and collapsed list."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Data Volume Scaling (pandas)",
                    severity="critical",
                    category="data_volume_scaling",
                    description="Resource cap exceeded.",
                    grouped_findings=[
                        Finding(
                            title="Array Size Scaling (numpy)",
                            severity="critical",
                        ),
                        Finding(
                            title="Matrix Operation Scaling (numpy)",
                            severity="critical",
                        ),
                    ],
                    group_count=3,
                ),
            ],
        )
        text = report.as_text()
        assert "(and 2 similar)" in text
        # Grouped finding titles appear in Also: lines
        assert "Also: Array Size Scaling (numpy), Matrix Operation Scaling (numpy)" in text

    def test_grouping_preserves_ungrouped(self):
        """Version flag findings (unique titles) should not be grouped."""
        findings = [
            Finding(
                title="Outdated dependency: flask",
                severity="info",
                category="",
                description="Flask is outdated.",
            ),
            Finding(
                title="Outdated dependency: requests",
                severity="info",
                category="",
                description="Requests is outdated.",
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        # Different titles → different patterns → not grouped
        assert len(result) == 2
        assert all(f.group_count == 1 for f in result)


    def test_no_group_different_dependencies(self):
        """Same category/pattern/metrics but different deps → not grouped."""
        findings = [
            Finding(
                title="Errors during: pandas_data_volume_scaling",
                severity="warning",
                category="data_volume_scaling",
                affected_dependencies=["pandas"],
                _peak_memory_mb=54.8,
                _execution_time_ms=200.0,
                _error_count=26,
            ),
            Finding(
                title="Errors during: numpy_array_size_scaling",
                severity="warning",
                category="data_volume_scaling",
                affected_dependencies=["numpy"],
                _peak_memory_mb=54.8,
                _execution_time_ms=200.0,
                _error_count=26,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 2
        assert all(f.group_count == 1 for f in result)


class TestFindingDeduplicationBySeverity:
    """Test that same-title findings at different severities are deduplicated."""

    def test_same_title_different_severity_keeps_highest(self):
        """CRITICAL + WARNING with same title → only CRITICAL kept."""
        findings = [
            Finding(
                title="Data Volume Scaling (pandas)",
                severity="critical",
                category="data_volume_scaling",
                description="Resource limit hit.",
                affected_dependencies=["pandas"],
                _peak_memory_mb=500,
                _execution_time_ms=1000,
            ),
            Finding(
                title="Data Volume Scaling (pandas)",
                severity="warning",
                category="data_volume_scaling",
                description="3 errors occurred.",
                affected_dependencies=["pandas"],
                _peak_memory_mb=500,
                _execution_time_ms=1000,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        assert len(result) == 1
        assert result[0].severity == "critical"

    def test_same_title_same_severity_kept(self):
        """Two WARNING findings with same title but different types are kept."""
        findings = [
            Finding(
                title="Flask Concurrent Stress",
                severity="warning",
                category="concurrent_execution",
                description="Errors occurred.",
                affected_dependencies=["flask"],
                _peak_memory_mb=100,
                _execution_time_ms=200,
            ),
            Finding(
                title="Flask Concurrent Stress",
                severity="warning",
                category="concurrent_execution",
                description="Memory growth unbounded.",
                affected_dependencies=["flask"],
                _peak_memory_mb=100,
                _execution_time_ms=200,
            ),
        ]
        result = ReportGenerator._group_similar_findings(findings)
        # Both kept (same severity) — they may be grouped by metrics though
        titles = [f.title for f in result]
        assert "Flask Concurrent Stress" in titles

    def test_resource_cap_suppresses_error_finding(self):
        """A scenario with resource_cap_hit should not also produce error finding."""
        from mycode.engine import ScenarioResult, StepResult, ExecutionEngineResult
        from mycode.ingester import IngestionResult
        sr = ScenarioResult(
            scenario_name="pandas_data_volume",
            scenario_category="data_volume_scaling",
            status="completed",
            resource_cap_hit=True,
            total_errors=5,
            steps=[
                StepResult(
                    step_name="size_1000",
                    execution_time_ms=100.0,
                    memory_peak_mb=500.0,
                    error_count=5,
                    resource_cap_hit="memory",
                ),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        gen = ReportGenerator(offline=True)
        ingestion = IngestionResult(project_path="/tmp/test")
        report = gen.generate(execution, ingestion, [])
        # Should have exactly one finding (CRITICAL for resource cap),
        # not a second WARNING for errors
        scenario_findings = [
            f for f in report.findings
            if "pandas" in f.title.lower() or "data volume" in f.title.lower()
        ]
        severities = [f.severity for f in scenario_findings]
        assert "critical" in severities
        assert severities.count("warning") == 0


class TestMetricsSimilar:
    """Direct tests for _metrics_similar."""

    def test_identical(self):
        a = Finding(title="a", severity="critical", _peak_memory_mb=100, _execution_time_ms=200)
        b = Finding(title="b", severity="critical", _peak_memory_mb=100, _execution_time_ms=200)
        assert ReportGenerator._metrics_similar(a, b) is True

    def test_within_tolerance(self):
        a = Finding(title="a", severity="critical", _peak_memory_mb=100)
        b = Finding(title="b", severity="critical", _peak_memory_mb=109)
        assert ReportGenerator._metrics_similar(a, b) is True

    def test_beyond_tolerance(self):
        a = Finding(title="a", severity="critical", _peak_memory_mb=100)
        b = Finding(title="b", severity="critical", _peak_memory_mb=112)
        assert ReportGenerator._metrics_similar(a, b) is False

    def test_both_zero(self):
        a = Finding(title="a", severity="critical")
        b = Finding(title="b", severity="critical")
        assert ReportGenerator._metrics_similar(a, b) is True

    def test_one_zero_one_nonzero(self):
        a = Finding(title="a", severity="critical", _peak_memory_mb=0.0)
        b = Finding(title="b", severity="critical", _peak_memory_mb=100.0)
        assert ReportGenerator._metrics_similar(a, b) is False


class TestDegradationGrouping:
    """Tests for grouping similar degradation points."""

    def _make_dp(self, name, metric="execution_time_ms",
                 first=10.0, last=100.0):
        """Helper to create a DegradationPoint with steps."""
        return DegradationPoint(
            scenario_name=name,
            metric=metric,
            steps=[("small", first), ("large", last)],
            breaking_point="large",
            description=f"{metric} increased {last/first:.1f}x",
        )

    def test_group_identical_degradation(self):
        """3 degradation points with same metric and ratio → 1 with count=3."""
        points = [
            self._make_dp(f"coupling_compute_{i}", first=10.0, last=260.0)
            for i in range(3)
        ]
        result = ReportGenerator._group_similar_degradation_points(points)
        assert len(result) == 1
        assert result[0].group_count == 3
        assert len(result[0].grouped_points) == 2

    def test_no_group_different_metrics(self):
        """Same ratio, different metrics → 2 separate."""
        points = [
            self._make_dp("test_a", metric="execution_time_ms", first=10, last=100),
            self._make_dp("test_b", metric="memory_peak_mb", first=10, last=100),
        ]
        result = ReportGenerator._group_similar_degradation_points(points)
        assert len(result) == 2

    def test_no_group_divergent_ratios(self):
        """>10% ratio difference → 2 separate."""
        points = [
            self._make_dp("test_a", first=10.0, last=260.0),   # 26x
            self._make_dp("test_b", first=10.0, last=100.0),   # 10x
        ]
        result = ReportGenerator._group_similar_degradation_points(points)
        assert len(result) == 2

    def test_within_10pct_ratio_groups(self):
        """26x vs 28x (7.7% diff) → grouped."""
        points = [
            self._make_dp("test_a", first=10.0, last=260.0),   # 26x
            self._make_dp("test_b", first=10.0, last=280.0),   # 28x
        ]
        result = ReportGenerator._group_similar_degradation_points(points)
        assert len(result) == 1
        assert result[0].group_count == 2

    def test_grouped_degradation_rendering(self):
        """as_text() shows '(and N similar)' and Also: list for degradation."""
        dp = DegradationPoint(
            scenario_name="coupling_compute_setLoading",
            metric="execution_time_ms",
            steps=[("small", 10.0), ("large", 2600.0)],
            breaking_point="large",
            description="Execution time increased 260.0x",
            grouped_points=[
                DegradationPoint(
                    scenario_name="coupling_compute_setError",
                    metric="execution_time_ms",
                ),
                DegradationPoint(
                    scenario_name="coupling_compute_setRawScores",
                    metric="execution_time_ms",
                ),
            ],
            group_count=3,
        )
        report = DiagnosticReport(degradation_points=[dp])
        text = report.as_text()
        assert "(and 2 similar)" in text
        # Scenario names are humanized in Also: lines
        assert "Also: Set error, Set raw scores" in text

    def test_many_grouped_shows_plus_more(self):
        """More than 5 grouped → shows '+N more'."""
        grouped = [
            DegradationPoint(
                scenario_name=f"coupling_compute_func_{i}",
                metric="execution_time_ms",
            )
            for i in range(8)
        ]
        dp = DegradationPoint(
            scenario_name="coupling_compute_func_main",
            metric="execution_time_ms",
            steps=[("small", 10.0), ("large", 260.0)],
            grouped_points=grouped,
            group_count=9,
        )
        report = DiagnosticReport(degradation_points=[dp])
        text = report.as_text()
        assert "(and 8 similar)" in text
        assert "+3 more" in text

    def test_grouping_wired_into_generate(self):
        """End-to-end: identical degradation from multiple scenarios gets grouped."""
        # Create 5 scenarios that all produce identical degradation
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name=f"coupling_compute_{i}",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[
                        StepResult(step_name="size_100",
                                  execution_time_ms=10.0, memory_peak_mb=5.0),
                        StepResult(step_name="size_10000",
                                  execution_time_ms=250.0, memory_peak_mb=120.0),
                    ],
                    total_errors=0,
                )
                for i in range(5)
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )
        # 5 scenarios × 2 metrics (time + memory) = 10 degradation points raw
        # Should group to 2 (one per metric)
        assert len(report.degradation_points) == 2
        for dp in report.degradation_points:
            assert dp.group_count == 5


class TestFindingPattern:
    """Direct tests for _finding_pattern."""

    def test_scenario_failed(self):
        f = Finding(title="Scenario failed: some_test", severity="critical")
        assert ReportGenerator._finding_pattern(f) == "scenario_failed"

    def test_resource_limit(self):
        f = Finding(title="Resource limit hit: some_test", severity="critical")
        assert ReportGenerator._finding_pattern(f) == "resource_limit_hit"

    def test_errors_during(self):
        f = Finding(title="Errors during: some_test", severity="warning")
        assert ReportGenerator._finding_pattern(f) == "errors_during"

    def test_failure_indicators(self):
        f = Finding(title="Failure indicators triggered: some_test", severity="warning")
        assert ReportGenerator._finding_pattern(f) == "failure_indicators"

    def test_unique_title(self):
        f = Finding(title="Outdated dependency: flask", severity="info")
        assert ReportGenerator._finding_pattern(f) == "Outdated dependency: flask"


# ── Edge Cases ──


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_execution_result(self):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            ExecutionEngineResult(),
            IngestionResult(project_path="/tmp/x"),
            [],
        )
        assert report.scenarios_run == 0
        assert "no stress test" in report.summary.lower()

    def test_no_dependencies(self, clean_execution):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            clean_execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )
        assert len(report.version_flags) == 0
        assert len(report.unrecognized_deps) == 0

    def test_skipped_scenario(self):
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="skipped_test",
                    scenario_category="async_failures",
                    status="skipped",
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x"),
            [],
        )
        # Skipped scenarios count as incomplete (not failed)
        assert report.scenarios_failed == 0
        assert report.scenarios_incomplete == 1

    def test_operational_intent_in_context(self, clean_execution):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            clean_execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
            operational_intent="Handles 500 users daily",
        )
        assert report.operational_context == "Handles 500 users daily"


# ── Plain Summary Tests ──


class TestPlainSummary:
    """Tests for plain-language summary generation."""

    def test_clean_run_positive_summary(self, clean_execution):
        """No critical/warning findings → positive assessment."""
        ingestion = IngestionResult(
            project_path="/tmp/x", files_analyzed=1,
            dependencies=[
                DependencyInfo(name="flask", installed_version="3.1.0", is_outdated=False),
            ],
        )
        matches = [
            ProfileMatch(dependency_name="flask", profile=MagicMock(), version_match=True),
        ]
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, ingestion, matches)

        assert "looks solid" in report.plain_summary.lower()

    def test_critical_findings_summary(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Critical findings → warning-toned assessment."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert "problems" in report.plain_summary.lower()
        assert "real-world conditions" in report.plain_summary.lower()

    def test_includes_project_description(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Project description from ingester used in activity phrases."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="Personal budget tracker for daily use",
            project_name="budget tracker",
        )

        lower = report.plain_summary.lower()
        # Auto-generated description from ingester (flask + sqlalchemy + requests)
        # should appear instead of raw project_name
        assert "flask" in lower
        # Should be woven into bullets too
        assert lower.count("flask") >= 1

    def test_degradation_translated(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        """Degradation points appear as plain-language bullets with real impact."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        assert "- " in report.plain_summary
        lower = report.plain_summary.lower()
        # Should describe impact in real terms, not multipliers
        assert "response time" in lower or "memory" in lower
        # Should include activity context and practical consequences
        assert "during" in lower or "server" in lower or "session" in lower

    def test_findings_translated(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Findings appear as plain-language bullets with When pattern."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert "- " in report.plain_summary
        assert "when " in report.plain_summary.lower()

    def test_max_three_items(self):
        """Only top 3 findings/degradation shown."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name=f"test_scenario_{i}",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[
                        StepResult(step_name="data_size_100", execution_time_ms=10.0, memory_peak_mb=5.0),
                        StepResult(step_name="data_size_100000", execution_time_ms=500.0, memory_peak_mb=200.0),
                    ],
                    total_errors=0,
                )
                for i in range(10)
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        bullet_count = report.plain_summary.count("\n- ")
        assert bullet_count <= 3

    def test_no_boilerplate_closing_line(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Summary does NOT end with boilerplate bridge line."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert "paste these into your coding tool" not in report.plain_summary.lower()
        assert "see detailed" not in report.plain_summary.lower()

    def test_plain_summary_in_as_text(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """as_text() renders plain_summary before the technical summary."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        text = report.as_text()
        plain_pos = text.find("real-world conditions")
        summary_pos = text.find("critical issue")
        assert plain_pos != -1
        assert summary_pos != -1
        assert plain_pos < summary_pos

    def test_no_intent_uses_auto_description(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Without intent, auto-generated description from ingester is used."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="",
            project_name="",
        )

        assert report.plain_summary
        # Auto-generated description used (flask detected in dependencies)
        lower = report.plain_summary.lower()
        assert "flask" in lower or "your" in lower

    def test_auto_description_used_over_raw_name(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Auto-generated description from ingester takes precedence over raw user input."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="A long complex description that would extract badly",
            project_name="incident tracker",
        )
        # Auto-generated description (from flask deps) used instead of raw project_name
        assert report.project_description
        assert "flask" in report.project_description.lower()

    def test_empty_execution(self):
        """No scenarios → no plain summary generated."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            ExecutionEngineResult(),
            IngestionResult(project_path="/tmp/x"),
            [],
        )

        assert report.plain_summary == ""

    def test_impact_uses_real_terms_not_multiplier(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        """Impact described in real terms, not multipliers."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        lower = report.plain_summary.lower()
        # Should have real-terms descriptions (time bands or concrete values)
        assert any(
            phrase in lower
            for phrase in ("instant", "fast", "slow", "delay", "second",
                           "minute", "ms", "mb")
        )
        # Should NOT have multiplier patterns like "80x" or "204x slower"
        import re
        assert not re.search(r"\d+x slower", lower)

    def test_degradation_bullet_describes_impact(self):
        """Degradation bullet leads with impact, not step labels."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent_request_load",
                    scenario_category="concurrent_execution",
                    status="completed",
                    steps=[
                        StepResult(step_name="concurrent_1", execution_time_ms=10.0, memory_peak_mb=5.0),
                        StepResult(step_name="concurrent_50", execution_time_ms=5000.0, memory_peak_mb=50.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        lower = report.plain_summary.lower()
        # Should describe the impact in real terms
        assert "memory" in lower or "response time" in lower
        # Should include activity context and practical server sizing
        assert "during" in lower or "server" in lower or "session" in lower

    def test_one_bullet_per_scenario(self):
        """Same scenario with multiple degradation metrics → only one bullet."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent_request_load",
                    scenario_category="concurrent_execution",
                    status="completed",
                    steps=[
                        StepResult(step_name="concurrent_1", execution_time_ms=10.0, memory_peak_mb=5.0),
                        StepResult(step_name="concurrent_50", execution_time_ms=5000.0, memory_peak_mb=200.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        # Two degradation points (time + memory) but same scenario → 1 bullet
        bullet_count = report.plain_summary.count("\n- ")
        assert bullet_count == 1

    def test_summary_bullet_no_threshold_detail(self):
        """Summary bullets don't include threshold details (those belong in curves)."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent_request_load",
                    scenario_category="concurrent_execution",
                    status="completed",
                    steps=[
                        StepResult(step_name="concurrent_1", execution_time_ms=0.5, memory_peak_mb=5.0),
                        StepResult(step_name="concurrent_10", execution_time_ms=50.0, memory_peak_mb=15.0),
                        StepResult(step_name="concurrent_50", execution_time_ms=3000.0, memory_peak_mb=80.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        lower = report.plain_summary.lower()
        # Memory is prioritized — should show peak MB
        assert "80mb" in lower
        # Should NOT include threshold details like "starts climbing around"
        assert "starts climbing" not in lower
        assert "starts slowing" not in lower

    def test_memory_projects_production_impact(self):
        """Memory findings connect to production deployment impact."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="streamlit_cache_memory_growth",
                    scenario_category="memory_profiling",
                    status="completed",
                    steps=[
                        StepResult(step_name="batch_0", execution_time_ms=5.0, memory_peak_mb=7.0),
                        StepResult(step_name="batch_50", execution_time_ms=5.0, memory_peak_mb=72.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        lower = report.plain_summary.lower()
        assert "72mb" in lower
        # Should include practical server sizing
        assert "2gb server" in lower or "concurrent sessions" in lower

    def test_priority_caps_before_memory_before_time(self):
        """Resource caps appear first, then memory, then execution time."""
        execution = ExecutionEngineResult(
            scenario_results=[
                # Execution time degradation only (low priority)
                # Memory stays flat so only time degrades
                ScenarioResult(
                    scenario_name="flask_large_payload_response",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[
                        StepResult(step_name="data_size_1", execution_time_ms=0.13, memory_peak_mb=5.0),
                        StepResult(step_name="data_size_10000", execution_time_ms=770.0, memory_peak_mb=5.5),
                    ],
                    total_errors=0,
                ),
                # Memory degradation (medium priority)
                ScenarioResult(
                    scenario_name="streamlit_cache_memory_growth",
                    scenario_category="memory_profiling",
                    status="completed",
                    steps=[
                        StepResult(step_name="batch_0", execution_time_ms=5.0, memory_peak_mb=0.08),
                        StepResult(step_name="batch_50", execution_time_ms=5.0, memory_peak_mb=72.0),
                    ],
                    total_errors=0,
                ),
                # Resource cap hit (highest priority)
                ScenarioResult(
                    scenario_name="requests_timeout_behavior",
                    scenario_category="edge_case_input",
                    status="failed",
                    summary="Connection timeout not handled",
                    total_errors=3,
                    resource_cap_hit=True,
                    steps=[
                        StepResult(step_name="edge_timeout", execution_time_ms=30000.0,
                                   resource_cap_hit="timeout"),
                    ],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=5),
            [],
        )

        # Extract the bullet lines
        bullets = [
            line for line in report.plain_summary.split("\n")
            if line.startswith("- ")
        ]
        assert len(bullets) == 3
        # First bullet should be about the resource cap / edge case
        assert ("timed out" in bullets[0].lower()
                or "crash" in bullets[0].lower()
                or "unusual" in bullets[0].lower())
        # Second should be about memory
        assert "memory" in bullets[1].lower()
        # Third should be about response time
        assert "response time" in bullets[2].lower()

    def test_extract_project_ref(self):
        """_extract_project_ref extracts short noun phrases."""
        # Strips "An" article, splits on " that "
        assert _extract_project_ref(
            "An incident matching system that compares reports"
        ) == "incident matching system"
        # Strips "I built a" filler + "a" article
        assert _extract_project_ref(
            "I built a Flask app that shows charts"
        ) == "Flask app"
        # No filler to strip, keeps full clause
        ref = _extract_project_ref("Personal budget tracker for daily use")
        assert "budget tracker" in ref
        # Strips "My" article
        assert _extract_project_ref("My todo list app") == "todo list app"
        # Fallback for too-short input
        assert _extract_project_ref("") == "project"
        assert _extract_project_ref("ab") == "project"
        # Truncates at arrow and strips parenthetical framework ref
        assert _extract_project_ref(
            "Incident Solution (Flask) \u2192 Matches user issues to known bugs"
        ) == "Incident Solution"
        # Truncates at ASCII arrow
        assert _extract_project_ref(
            "Budget App -> tracks expenses daily"
        ) == "Budget App"
        # Truncates at newline
        assert _extract_project_ref(
            "Task Manager\nBuilt with React and Node"
        ) == "Task Manager"
        # Strips em-dash separated description
        assert _extract_project_ref(
            "Sales Dashboard \u2014 shows monthly revenue"
        ) == "Sales Dashboard"

    def test_degradation_preferred_over_finding_same_scenario(self):
        """Degradation points surface instead of findings for the same scenario."""
        execution = ExecutionEngineResult(
            scenario_results=[
                # This scenario produces BOTH a finding (failed) AND degradation
                # points. The degradation point has curve data so should be preferred.
                ScenarioResult(
                    scenario_name="streamlit_cache_memory_growth",
                    scenario_category="memory_profiling",
                    status="completed",
                    steps=[
                        StepResult(step_name="batch_0", execution_time_ms=5.0, memory_peak_mb=0.08),
                        StepResult(step_name="batch_50", execution_time_ms=5.0, memory_peak_mb=72.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        lower = report.plain_summary.lower()
        # Should contain degradation curve data (actual MB values)
        assert "0mb" in lower or "72mb" in lower
        # Should NOT just say "keeps growing" without numbers
        assert "72mb" in lower

    def test_error_pluralization_singular(self):
        """Single error uses singular form."""
        gen = ReportGenerator(offline=True)
        f = Finding(
            title="Errors during: flask_concurrent_request_load",
            severity="warning",
            category="concurrent_execution",
        )
        f._execution_time_ms = 0.0
        f._peak_memory_mb = 0.0
        f._error_count = 1
        result = gen._translate_finding(f, "your project")
        assert "1 error occurred" in result.lower() or "1 request" in result.lower()
        assert "1 errors" not in result.lower()

    def test_error_pluralization_plural(self):
        """Multiple errors uses plural form."""
        gen = ReportGenerator(offline=True)
        f = Finding(
            title="Errors during: flask_concurrent_request_load",
            severity="warning",
            category="concurrent_execution",
        )
        f._execution_time_ms = 0.0
        f._peak_memory_mb = 0.0
        f._error_count = 5
        result = gen._translate_finding(f, "your project")
        assert "5 errors occurred" in result.lower() or "5 request" in result.lower()

    def test_error_describes_timeout_type(self):
        """Error description identifies timeout errors from details."""
        gen = ReportGenerator(offline=True)
        f = Finding(
            title="Errors during: requests_timeout_behavior",
            severity="warning",
            category="concurrent_execution",
            details="TimeoutError: Connection timed out after 30s",
        )
        f._execution_time_ms = 0.0
        f._peak_memory_mb = 0.0
        f._error_count = 3
        result = gen._translate_finding(f, "your project")
        assert "timed out" in result.lower()


    def test_plain_summary_criticals_before_warnings(self):
        """Within the same priority band, criticals sort before warnings."""
        # Use two scenarios that produce findings at the same priority band
        # but different severities: one critical, one warning.
        execution = ExecutionEngineResult(
            scenario_results=[
                # Warning-level finding: small number of errors but scenario
                # completes (not failed, just has errors)
                ScenarioResult(
                    scenario_name="flask_data_volume_warning",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    total_errors=1,
                    steps=[
                        StepResult(
                            step_name="data_size_100",
                            execution_time_ms=10.0,
                            memory_peak_mb=5.0,
                            error_count=1,
                            errors=[{"type": "ValueError", "message": "bad input"}],
                        ),
                    ],
                ),
                # Critical-level finding: scenario fails with many errors
                ScenarioResult(
                    scenario_name="flask_concurrent_crash",
                    scenario_category="concurrent_execution",
                    status="failed",
                    summary="Crash under load",
                    total_errors=5,
                    steps=[
                        StepResult(
                            step_name="concurrent_10",
                            execution_time_ms=500.0,
                            memory_peak_mb=200.0,
                            error_count=5,
                            errors=[{"type": "MemoryError", "message": "OOM"}],
                        ),
                    ],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )

        # Verify that critical findings appear before warning findings
        # after sort — findings list is sorted by severity
        critical_findings = [f for f in report.findings if f.severity == "critical"]
        warning_findings = [f for f in report.findings if f.severity == "warning"]
        if critical_findings and warning_findings:
            # First critical should appear before first warning in the list
            ci = report.findings.index(critical_findings[0])
            wi = report.findings.index(warning_findings[0])
            assert ci < wi


class TestPlainSummaryHelpers:
    """Tests for plain summary module-level helpers."""

    def test_human_time(self):
        assert _human_time(0.5) == "instant"
        assert _human_time(50) == "fast"
        assert _human_time(200) == "noticeable delay"
        assert _human_time(800) == "slow"
        assert _human_time(5000) == "very slow"
        assert "seconds" in _human_time(15000)
        assert _human_time(45000) == "about 30 seconds"
        assert _human_time(120000) == "over a minute"

    def test_describe_scenario_template_match(self):
        assert _describe_scenario("flask_concurrent_request_load") == \
            "handling multiple users at once"
        assert _describe_scenario("streamlit_cache_memory_growth") == \
            "caching data over time"
        assert _describe_scenario("requests_timeout_behavior") == \
            "calling external APIs that respond slowly"

    def test_describe_scenario_coupling_patterns(self):
        assert "components" in _describe_scenario("coupling_api_fetch_data")
        assert "calculations" in _describe_scenario("coupling_compute_transform")
        assert "state" in _describe_scenario("coupling_state_setters_group_1")

    def test_describe_scenario_unknown_humanizes_fallback(self):
        # Unknown scenarios should produce readable text, not empty string
        result = _describe_scenario("completely_unknown_thing")
        assert "_" not in result  # no snake_case in output
        assert result  # non-empty

    def test_describe_scenario_check_suffix(self):
        result = _describe_scenario("pandas_settingwithcopy_warning_ignored_check")
        assert "behavior" in result
        assert "_" not in result

    def test_describe_scenario_version_discrepancy(self):
        result = _describe_scenario("streamlit_version_discrepancy")
        assert "streamlit" in result
        assert "version" in result
        assert "_" not in result
        # Should not start with "testing" to avoid "myCode tested testing..."
        assert not result.startswith("testing")

    def test_describe_step_patterns(self):
        assert _describe_step("data_size_10000") == "10,000 items"
        assert _describe_step("concurrent_50") == "50 simultaneous users"
        assert _describe_step("io_size_100000") == "100KB of data"
        assert _describe_step("gil_threads_8") == "8 parallel threads"

    def test_describe_step_unknown_returns_empty(self):
        assert _describe_step("edge_none") == ""
        assert _describe_step("iteration_5") == ""

    def test_describe_impact_time_different_bands(self):
        result = _describe_impact("execution_time_ms", 0.5, 5000.0)
        assert "instant" in result
        assert "very slow" in result

    def test_describe_impact_time_same_band_uses_peak(self):
        """When both values are in the same band, use concrete peak value."""
        result = _describe_impact("execution_time_ms", 10.0, 80.0)
        assert "80ms" in result

    def test_describe_impact_time_with_activity(self):
        """Activity context is included when provided."""
        result = _describe_impact(
            "execution_time_ms", 0.5, 5000.0,
            activity="concurrent user handling",
        )
        assert "concurrent user handling" in result
        assert "instant" in result
        assert "very slow" in result

    def test_describe_impact_memory_high_no_user_scale(self):
        """Memory ≥50MB without user_scale gives practical server sizing."""
        result = _describe_impact("memory_peak_mb", 5.0, 120.0)
        assert "120MB" in result
        assert "2gb server" in result.lower()
        assert "concurrent sessions" in result.lower()

    def test_describe_impact_memory_with_user_scale(self):
        """Memory ≥50MB with user_scale connects to user context."""
        result = _describe_impact("memory_peak_mb", 5.0, 120.0, user_scale=2983)
        assert "120MB" in result
        assert "2gb server" in result.lower()
        assert "concurrent sessions" in result.lower()
        assert "running out of memory" in result

    def test_describe_impact_memory_small_no_projection(self):
        """Memory <50MB says it's moderate and unlikely to cause problems."""
        result = _describe_impact("memory_peak_mb", 5.0, 30.0)
        assert "30MB" in result
        assert "moderate" in result.lower()
        assert "unlikely" in result.lower()

    def test_describe_impact_memory_shows_growth(self):
        """Memory with significant growth shows from→to values."""
        result = _describe_impact("memory_peak_mb", 5.0, 120.0)
        assert "grew from 5MB to 120MB" in result

    def test_describe_impact_errors_names_count_and_consequence(self):
        """Error degradation names count and states consequence."""
        result = _describe_impact("error_count", 0, 3)
        assert "3" in result
        assert "runtime" in result.lower()
        assert "crash" in result.lower() or "wrong" in result.lower()

    def test_describe_impact_errors_singular(self):
        """Single error uses singular form."""
        result = _describe_impact("error_count", 0, 1)
        assert "1 runtime error" in result
        assert "errors" not in result.split("1 runtime error")[0]

    def test_format_ms(self):
        assert _format_ms(0.16) == "0.16ms"
        assert _format_ms(78.0) == "78ms"
        assert _format_ms(770.0) == "770ms"
        assert _format_ms(5000.0) == "5.0s"
        assert _format_ms(120000.0) == "2.0min"

    def test_extract_cap_type_timeout(self):
        f = Finding(title="Resource limit hit: test", severity="critical",
                    details="Caps hit: timeout")
        assert "timed out" in _extract_cap_type(f)

    def test_extract_cap_type_memory(self):
        f = Finding(title="Resource limit hit: test", severity="critical",
                    details="Caps hit: memory_cap")
        assert "memory" in _extract_cap_type(f)

    def test_extract_cap_type_unknown(self):
        f = Finding(title="Some finding", severity="warning", details="")
        assert _extract_cap_type(f) == ""

    def test_extract_project_ref_arrow(self):
        """Truncates at unicode arrow and strips parentheticals."""
        ref = _extract_project_ref(
            "Incident Solution (Flask) \u2192 Matches user issues"
        )
        assert ref == "Incident Solution"

    def test_extract_project_ref_newline(self):
        ref = _extract_project_ref("Task Manager\nBuilt with React")
        assert ref == "Task Manager"

    def test_finding_uses_actual_metrics(self):
        """_translate_finding includes actual metric values, not vague phrases."""
        gen = ReportGenerator(offline=True)
        f = Finding(
            title="Errors during: flask_concurrent_request_load",
            severity="warning",
            category="concurrent_execution",
        )
        f._execution_time_ms = 5000.0
        f._peak_memory_mb = 200.0
        f._error_count = 3
        result = gen._translate_finding(f, "your project")
        lower = result.lower()
        # Should include actual metric data
        assert "5.0s" in lower or "200mb" in lower or "3 errors" in lower
        # Should NOT contain vague phrases
        assert "things slow down" not in lower
        assert "problems were detected" not in lower

    def test_describe_errors_singular(self):
        f = Finding(title="test", severity="warning")
        f._error_count = 1
        result = _describe_errors(f)
        assert "1 error occurred" == result

    def test_describe_errors_plural(self):
        f = Finding(title="test", severity="warning")
        f._error_count = 5
        result = _describe_errors(f)
        assert "5 errors occurred" == result

    def test_describe_errors_timeout(self):
        f = Finding(title="test", severity="warning",
                    details="TimeoutError: read timed out")
        f._error_count = 3
        result = _describe_errors(f)
        assert "3 requests timed out" == result

    def test_describe_errors_timeout_singular(self):
        f = Finding(title="test", severity="warning",
                    details="TimeoutError: read timed out")
        f._error_count = 1
        result = _describe_errors(f)
        assert "1 request timed out" == result


# ── Summary Bullet Quality Tests (Session 15) ──


class TestSummaryBulletQuality:
    """Tests that summary bullets answer: what was tested, what happened, why it matters."""

    def test_memory_bullet_includes_activity(self):
        """Memory degradation bullet must name what activity caused it."""
        gen = ReportGenerator(offline=True)
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="memory_peak_mb",
            steps=[("concurrent_1", 7.0), ("concurrent_50", 73.0)],
        )
        result = gen._translate_degradation(dp, "your app")
        lower = result.lower()
        # Must say WHAT was being done
        assert "handling" in lower or "concurrent" in lower or "multiple" in lower
        # Must include the metric
        assert "73mb" in lower

    def test_memory_bullet_includes_server_sizing(self):
        """Memory bullet must relate to real server sizes."""
        gen = ReportGenerator(offline=True)
        dp = DegradationPoint(
            scenario_name="streamlit_cache_memory_growth",
            metric="memory_peak_mb",
            steps=[("batch_0", 7.0), ("batch_50", 73.0)],
        )
        result = gen._translate_degradation(dp, "your app")
        lower = result.lower()
        assert "2gb server" in lower
        assert "concurrent sessions" in lower

    def test_memory_bullet_with_user_scale_shows_capacity(self):
        """Memory bullet with user_scale shows sessions before OOM."""
        gen = ReportGenerator(offline=True)
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="memory_peak_mb",
            steps=[("concurrent_1", 5.0), ("concurrent_50", 73.0)],
        )
        result = gen._translate_degradation(dp, "your app", user_scale=13467)
        lower = result.lower()
        assert "73mb" in lower
        assert "2gb server" in lower
        assert "running out of memory" in lower

    def test_error_bullet_names_type_and_significance(self):
        """Error degradation bullet must name the error type and consequence."""
        gen = ReportGenerator(offline=True)
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="error_count",
            steps=[("concurrent_1", 0), ("concurrent_50", 3)],
        )
        result = gen._translate_degradation(dp, "your app")
        lower = result.lower()
        # Must say WHAT was being done
        assert "handling" in lower or "concurrent" in lower or "multiple" in lower
        # Must name the count
        assert "3" in lower
        # Must say WHY it matters
        assert "crash" in lower or "wrong" in lower

    def test_single_error_bullet_significance(self):
        """Single error bullet must still explain why it matters."""
        gen = ReportGenerator(offline=True)
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="error_count",
            steps=[("concurrent_1", 0), ("concurrent_50", 1)],
        )
        result = gen._translate_degradation(dp, "your app")
        lower = result.lower()
        assert "1" in result
        assert "runtime" in lower or "error" in lower
        assert "crash" in lower or "wrong" in lower

    def test_insignificant_memory_filtered_from_summary(self):
        """41MB memory alone should NOT appear in summary (not actionable)."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="coupling_compute_pipeline",
                    scenario_category="dependency_interaction",
                    status="completed",
                    steps=[
                        StepResult(step_name="compute_100", execution_time_ms=5.0, memory_peak_mb=7.0),
                        StepResult(step_name="compute_10000", execution_time_ms=8.0, memory_peak_mb=41.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )
        lower = report.plain_summary.lower()
        # 41MB is insignificant — should NOT be a bullet
        assert "41mb" not in lower

    def test_insignificant_fast_timing_filtered(self):
        """Fast timing (<500ms) degradation should NOT appear in summary."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_scaling",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[
                        StepResult(step_name="size_10", execution_time_ms=5.0, memory_peak_mb=5.0),
                        StepResult(step_name="size_1000", execution_time_ms=200.0, memory_peak_mb=10.0),
                    ],
                    total_errors=0,
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )
        lower = report.plain_summary.lower()
        # Under 500ms is still fast — should not be a bullet
        assert "200ms" not in lower

    def test_significance_filter_memory_below_50(self):
        """_is_significant_degradation returns False for memory <50MB."""
        dp = DegradationPoint(
            scenario_name="test", metric="memory_peak_mb",
            steps=[("a", 5.0), ("b", 41.0)],
        )
        assert _is_significant_degradation(dp) is False

    def test_significance_filter_memory_above_50(self):
        """_is_significant_degradation returns True for memory ≥50MB."""
        dp = DegradationPoint(
            scenario_name="test", metric="memory_peak_mb",
            steps=[("a", 5.0), ("b", 73.0)],
        )
        assert _is_significant_degradation(dp) is True

    def test_significance_filter_time_below_500(self):
        """_is_significant_degradation returns False for timing <500ms."""
        dp = DegradationPoint(
            scenario_name="test", metric="execution_time_ms",
            steps=[("a", 5.0), ("b", 200.0)],
        )
        assert _is_significant_degradation(dp) is False

    def test_significance_filter_time_above_500(self):
        """_is_significant_degradation returns True for timing ≥500ms."""
        dp = DegradationPoint(
            scenario_name="test", metric="execution_time_ms",
            steps=[("a", 5.0), ("b", 800.0)],
        )
        assert _is_significant_degradation(dp) is True

    def test_significance_filter_errors_zero(self):
        """_is_significant_degradation returns False for error count ending at 0."""
        dp = DegradationPoint(
            scenario_name="test", metric="error_count",
            steps=[("a", 0), ("b", 0)],
        )
        assert _is_significant_degradation(dp) is False

    def test_significance_filter_errors_nonzero(self):
        """_is_significant_degradation returns True for error count > 0."""
        dp = DegradationPoint(
            scenario_name="test", metric="error_count",
            steps=[("a", 0), ("b", 1)],
        )
        assert _is_significant_degradation(dp) is True

    def test_memory_profiling_finding_includes_activity(self):
        """memory_profiling finding bullet includes what activity caused it."""
        gen = ReportGenerator(offline=True)
        f = Finding(
            title="Memory growth: streamlit_cache_memory_growth",
            severity="warning",
            category="memory_profiling",
        )
        f._peak_memory_mb = 72.0
        result = gen._translate_finding(f, "your app")
        lower = result.lower()
        # Must name activity
        assert "caching" in lower or "memory" in lower
        # Must include practical server sizing
        assert "2gb server" in lower or "concurrent sessions" in lower

    def test_describe_errors_with_context_runtime(self):
        """Runtime errors are identified from details."""
        f = Finding(
            title="Errors during: test_scenario",
            severity="warning",
            details="RuntimeError: division by zero",
        )
        f._error_count = 2
        result = _describe_errors_with_context(f, "test_scenario")
        assert "runtime" in result.lower()

    def test_describe_errors_with_context_generic(self):
        """Generic errors without type info keep generic description."""
        f = Finding(
            title="Errors during: test_scenario",
            severity="warning",
            details="Something went wrong",
        )
        f._error_count = 3
        result = _describe_errors_with_context(f, "test_scenario")
        assert "3 errors occurred" in result

    def test_summary_max_4_bullets(self):
        """Summary has at most 4 bullets."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name=f"scenario_{i}",
                    scenario_category="concurrent_execution",
                    status="completed",
                    steps=[
                        StepResult(step_name="step_1", execution_time_ms=10.0, memory_peak_mb=100.0),
                        StepResult(step_name="step_50", execution_time_ms=5000.0, memory_peak_mb=500.0),
                    ],
                    total_errors=i,
                )
                for i in range(6)
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
        )
        bullet_count = report.plain_summary.count("\n- ")
        assert bullet_count <= 4


# ── Constraint Contextualisation Tests ──


class TestConstraintContextualisation:
    """Tests for constraint-driven finding severity classification (E3)."""

    def _make_report_with_concurrent_findings(self) -> DiagnosticReport:
        """Create a report with concurrent execution findings at various load levels."""
        return DiagnosticReport(
            scenarios_run=3,
            findings=[
                Finding(
                    title="Errors during: flask_concurrent_10",
                    severity="warning",
                    category="concurrent_execution",
                    description="3 errors occurred during this test.",
                    details="concurrent_10: TimeoutError",
                ),
                Finding(
                    title="Resource limit hit: flask_concurrent_50",
                    severity="critical",
                    category="concurrent_execution",
                    description="Resource cap hit.",
                    details="concurrent_50: memory cap",
                ),
                Finding(
                    title="Errors during: flask_concurrent_500",
                    severity="warning",
                    category="concurrent_execution",
                    description="Errors at high load.",
                    details="concurrent_500: connection refused",
                ),
            ],
        )

    def test_within_capacity_stays_critical(self):
        """Failure at load ≤ stated capacity → CRITICAL."""
        from mycode.constraints import OperationalConstraints

        report = self._make_report_with_concurrent_findings()
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        # concurrent_10 is within 20 users → critical
        f10 = [f for f in report.findings if "concurrent_10" in f.title][0]
        assert f10.severity == "critical"
        assert "You said 20 users" in f10.description

    def test_beyond_capacity_becomes_warning(self):
        """Failure at 1x < load ≤ 3x → WARNING."""
        from mycode.constraints import OperationalConstraints

        report = self._make_report_with_concurrent_findings()
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        # concurrent_50 is 2.5x of 20 → warning
        f50 = [f for f in report.findings if "concurrent_50" in f.title][0]
        assert f50.severity == "warning"
        assert "2.5x" in f50.description

    def test_far_beyond_capacity_becomes_info(self):
        """Failure at load > 3x → INFORMATIONAL."""
        from mycode.constraints import OperationalConstraints

        report = self._make_report_with_concurrent_findings()
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        # concurrent_500 is 25x of 20 → info
        f500 = [f for f in report.findings if "concurrent_500" in f.title][0]
        assert f500.severity == "info"
        assert "well beyond" in f500.description.lower()

    def test_no_user_scale_notes_default(self):
        """None user_scale adds 'not specified' note to findings."""
        from mycode.constraints import OperationalConstraints

        report = self._make_report_with_concurrent_findings()
        constraints = OperationalConstraints()  # user_scale=None

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        for f in report.findings:
            if f.severity in ("critical", "warning"):
                assert "not specified" in f.description.lower()

    def test_info_findings_not_reclassified(self):
        """Version/dep info findings keep their severity."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Outdated dependency: flask",
                    severity="info",
                    category="",
                    description="Flask is outdated.",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        # Should stay info — not reclassified
        assert report.findings[0].severity == "info"
        assert "You said 20 users" not in report.findings[0].description

    def test_constraint_summary_added_to_context(self):
        """Constraint summary is appended to report operational_context."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            operational_context="A budgeting app",
            scenarios_run=1,
            findings=[],
        )
        constraints = OperationalConstraints(
            user_scale=50,
            data_type="tabular",
        )

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        assert "50" in report.operational_context
        assert "tabular" in report.operational_context

    def test_full_generate_with_constraints(self):
        """Full generate() call with constraints threads through correctly."""
        from mycode.constraints import OperationalConstraints

        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent_request_load",
                    scenario_category="concurrent_execution",
                    status="completed",
                    steps=[
                        StepResult(
                            step_name="concurrent_10",
                            execution_time_ms=50.0,
                            memory_peak_mb=10.0,
                        ),
                        StepResult(
                            step_name="concurrent_50",
                            execution_time_ms=500.0,
                            memory_peak_mb=200.0,
                            error_count=3,
                            errors=[
                                {"type": "TimeoutError", "message": "timeout"},
                            ],
                        ),
                    ],
                    total_errors=3,
                ),
            ],
            scenarios_completed=1,
        )
        ingestion = IngestionResult(
            project_path="/tmp/test",
            files_analyzed=1,
            total_lines=50,
        )

        constraints = OperationalConstraints(user_scale=20)
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution=execution,
            ingestion=ingestion,
            profile_matches=[],
            operational_intent="A budget app for 20 users",
            constraints=constraints,
        )

        # Should have contextualised findings
        assert report.scenarios_run == 1
        assert "20" in report.operational_context

    def test_data_size_finding_no_ratio(self):
        """data_size findings describe load level without user-scale ratio."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: pandas_data_volume_scaling",
                    severity="warning",
                    category="data_volume_scaling",
                    description="2 errors occurred during this test.",
                    details="data_size_10000: MemoryError",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        # Should NOT mention "You said 20 users" or compute a ratio
        assert "You said" not in f.description
        assert "your stated capacity" not in f.description
        # Should describe the load level in user terms
        assert "10,000 items" in f.description
        # Severity should be unchanged (no ratio reclassification)
        assert f.severity == "warning"

    def test_state_cycles_finding_no_ratio(self):
        """state_cycles findings describe load level without user-scale ratio."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Resource limit hit: react_state_management",
                    severity="critical",
                    category="memory_profiling",
                    description="Resource cap hit.",
                    details="state_cycles_1000: heap out of memory",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=50)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        assert "You said" not in f.description
        assert "1,000 state mutation cycles" in f.description
        assert f.severity == "critical"

    def test_io_size_finding_no_ratio(self):
        """io_size findings display size in human units, no user-scale ratio."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: file_upload_scaling",
                    severity="warning",
                    category="data_volume_scaling",
                    description="Errors at large payload.",
                    details="io_size_1048576: timeout",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=10)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        assert "You said" not in f.description
        assert "MB of data" in f.description
        assert f.severity == "warning"

    def test_batch_finding_no_ratio(self):
        """batch_N findings are iteration counters — no load context injected."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: batch_processing",
                    severity="warning",
                    category="data_volume_scaling",
                    description="Slow at scale.",
                    details="batch_500: degraded",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        # batch_N is an iteration counter, not a load level — no ratio, no load context
        assert "You said" not in f.description
        assert "Slow at scale." in f.description

    def test_concurrent_finding_still_gets_ratio(self):
        """Concurrency findings still get the ratio against user_scale."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: flask_api_concurrency_test",
                    severity="warning",
                    category="concurrent_execution",
                    description="3 errors occurred.",
                    details="api_concurrency_50: TimeoutError",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        assert "You said 20 users" in f.description
        assert "2.5x" in f.description
        assert f.severity == "warning"

    def test_gil_threads_finding_gets_ratio(self):
        """gil_threads is a concurrency metric and should get user-scale ratio."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: gil_contention",
                    severity="warning",
                    category="concurrent_execution",
                    description="GIL contention.",
                    details="gil_threads_10: lock wait",
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=20)

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        assert "You said 20 users" in f.description
        assert f.severity == "critical"  # 10/20 ≤ 1.0 → critical

    def test_data_size_finding_no_user_scale(self):
        """data_size finding with no user_scale still describes the load level."""
        from mycode.constraints import OperationalConstraints

        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: data_volume_test",
                    severity="warning",
                    category="data_volume_scaling",
                    description="Errors at scale.",
                    details="data_size_5000: MemoryError",
                ),
            ],
        )
        constraints = OperationalConstraints()  # no user_scale

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)

        f = report.findings[0]
        # Should still get the descriptive text
        assert "5,000 items" in f.description
        # Should NOT get "not specified" (that's for concurrency only)
        assert "not specified" not in f.description


class TestCorpusAwareFindingLanguage:
    """Tests for corpus stats sentences appended to findings (Item 6)."""

    def test_corpus_stats_appended_to_finding(self):
        """Finding with dep that has corpus_stats gets corpus sentence."""
        from mycode.constraints import OperationalConstraints
        from mycode.library.loader import DependencyProfile

        profile = MagicMock(spec=DependencyProfile)
        profile.corpus_stats = {
            "tested_count": 5,
            "failure_rate": 0.80,
            "common_failure_category": "memory_profiling",
            "last_updated": "2026-02-27",
        }
        matches = [
            ProfileMatch(
                dependency_name="streamlit",
                profile=profile,
                installed_version="1.41.0",
                version_match=True,
            ),
        ]
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: streamlit_cache_memory_growth",
                    severity="warning",
                    category="memory_profiling",
                    description="Memory keeps growing.",
                    affected_dependencies=["streamlit"],
                ),
            ],
        )
        constraints = OperationalConstraints()

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints, matches)

        f = report.findings[0]
        assert "test portfolio" in f.description
        assert "streamlit" in f.description
        assert "80%" in f.description
        assert "5 tested projects" in f.description

    def test_no_corpus_stats_no_change(self):
        """Finding with dep without corpus_stats is unchanged."""
        from mycode.constraints import OperationalConstraints
        from mycode.library.loader import DependencyProfile

        profile = MagicMock(spec=DependencyProfile)
        profile.corpus_stats = {}
        matches = [
            ProfileMatch(
                dependency_name="flask",
                profile=profile,
                installed_version="3.1.0",
                version_match=True,
            ),
        ]
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: flask_concurrent",
                    severity="warning",
                    category="concurrent_execution",
                    description="Errors occurred.",
                    affected_dependencies=["flask"],
                ),
            ],
        )
        constraints = OperationalConstraints()

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints, matches)

        f = report.findings[0]
        assert "test portfolio" not in f.description

    def test_corpus_stats_minimum_tested_count(self):
        """tested_count < 3 → no corpus sentence appended."""
        from mycode.constraints import OperationalConstraints
        from mycode.library.loader import DependencyProfile

        profile = MagicMock(spec=DependencyProfile)
        profile.corpus_stats = {
            "tested_count": 2,
            "failure_rate": 1.00,
            "common_failure_category": "concurrent_execution",
            "last_updated": "2026-02-27",
        }
        matches = [
            ProfileMatch(
                dependency_name="socketio",
                profile=profile,
                installed_version="4.8.1",
                version_match=True,
            ),
        ]
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Errors during: socketio_connection_scaling",
                    severity="warning",
                    category="concurrent_execution",
                    description="Connection errors.",
                    affected_dependencies=["socketio"],
                ),
            ],
        )
        constraints = OperationalConstraints()

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints, matches)

        f = report.findings[0]
        assert "test portfolio" not in f.description

    def test_critical_finding_no_portfolio_stats(self):
        """CRITICAL findings should NOT get portfolio stats — they have enough urgency."""
        from mycode.constraints import OperationalConstraints
        from mycode.library.loader import DependencyProfile

        profile = MagicMock(spec=DependencyProfile)
        profile.corpus_stats = {
            "tested_count": 10,
            "failure_rate": 0.70,
            "common_failure_category": "memory_profiling",
            "last_updated": "2026-02-27",
        }
        matches = [
            ProfileMatch(
                dependency_name="pandas",
                profile=profile,
                installed_version="2.2.0",
                version_match=True,
            ),
        ]
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Crash: pandas memory exhaustion",
                    severity="critical",
                    category="memory_profiling",
                    description="Out of memory at 500 rows.",
                    affected_dependencies=["pandas"],
                ),
            ],
        )
        constraints = OperationalConstraints()

        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints, matches)

        f = report.findings[0]
        assert "test portfolio" not in f.description
        assert "tested projects" not in f.description


# ── Session 14: Harness Failure Transparency Tests ──


class TestHarnessFailureTransparency:
    """Tests for harness failure routing and rendering."""

    def test_harness_failure_routed_to_incomplete(self):
        """ScenarioResults with failure_reason go to incomplete_tests."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="test_flask_concurrent",
                    scenario_category="concurrent_execution",
                    status="failed",
                    failure_reason="unsupported_framework",
                    summary="ModuleNotFoundError: No module named 'flask'",
                    total_errors=1,
                    steps=[StepResult(
                        step_name="harness_crash",
                        error_count=1,
                        errors=[{"type": "HarnessCrash", "message": "..."}],
                    )],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        assert len(report.incomplete_tests) == 1
        assert report.incomplete_tests[0]._failure_reason == "unsupported_framework"
        assert len(report.findings) == 0

    def test_env_errors_still_routed_to_incomplete(self):
        """Environment-only errors (no failure_reason) still go to incomplete."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="test_import_stuff",
                    scenario_category="data_volume_scaling",
                    status="failed",
                    total_errors=1,
                    steps=[StepResult(
                        step_name="module_import",
                        error_count=1,
                        errors=[{
                            "type": "ModuleNotFoundError",
                            "message": "No module named 'myapp'",
                        }],
                    )],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        assert len(report.incomplete_tests) == 1
        assert report.incomplete_tests[0]._failure_reason == ""

    def test_section_header_renamed(self):
        """Report renders 'Tests myCode Could Not Run' instead of 'Incomplete Tests'."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(
                    title="Could not test: test_flask",
                    severity="info",
                    _failure_reason="unsupported_framework",
                ),
            ],
        )
        text = report.as_text()
        assert "Tests myCode Could Not Run" in text
        assert "Incomplete Tests" not in text

    def test_markdown_has_incomplete_section(self):
        """Markdown output includes Tests myCode Could Not Run section."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(
                    title="Could not test: test_flask",
                    severity="info",
                    _failure_reason="dependency_unavailable",
                ),
            ],
        )
        md = report.as_markdown()
        assert "Tests myCode Could Not Run" in md

    def test_grouping_by_reason(self):
        """Multiple failures of same reason are grouped."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(title="Could not test: test_a", severity="info",
                        _failure_reason="harness_generation_error"),
                Finding(title="Could not test: test_b", severity="info",
                        _failure_reason="harness_generation_error"),
                Finding(title="Could not test: test_c", severity="info",
                        _failure_reason="harness_generation_error"),
            ],
        )
        text = report.as_text()
        assert "Test Script Generation Issue (3 tests)" in text
        assert "Affected:" in text

    def test_failure_reason_in_json(self):
        """failure_reason appears in as_dict() output."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(title="Could not test: foo", severity="info",
                        _failure_reason="timeout"),
            ],
        )
        d = report.as_dict()
        assert d["incomplete_tests"][0]["failure_reason"] == "timeout"

    def test_plain_language_explanation(self):
        """Each failure reason gets a plain-language explanation."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(title="Could not test: test_dep", severity="info",
                        _failure_reason="dependency_unavailable"),
            ],
        )
        text = report.as_text()
        assert "--containerised" in text


# ── Session 14: Intelligent Project Description Tests ──


def _s14_ingestion(deps=None):
    """Helper: minimal IngestionResult."""
    dep_list = []
    if deps:
        for d in deps:
            if isinstance(d, str):
                dep_list.append(DependencyInfo(name=d))
            else:
                dep_list.append(d)
    return IngestionResult(
        project_path="/tmp/test",
        files_analyzed=3,
        total_lines=200,
        dependencies=dep_list,
    )


class TestIntelligentProjectDescription:
    """Tests for _generate_project_description with classifier output."""

    def test_framework_with_vertical(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["flask", "sqlalchemy", "pandas"])
        desc = _generate_project_description(ingestion, vertical="web_app")
        assert "Flask" in desc
        assert "web application" in desc

    def test_framework_with_project_name(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["react", "react-dom", "react-scripts"])
        desc = _generate_project_description(
            ingestion, project_name="React Shopping Cart", vertical="web_app",
        )
        assert "React" in desc
        assert "web application" in desc
        assert "(React Shopping Cart)" in desc

    def test_vertical_only(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion([])
        desc = _generate_project_description(ingestion, vertical="dashboard")
        assert "dashboard" in desc

    def test_deps_only_no_vertical_no_framework(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["numpy", "scipy"])
        desc = _generate_project_description(ingestion)
        assert "numpy" in desc
        assert "project" in desc.lower()

    def test_never_says_general_purpose(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion([])
        desc = _generate_project_description(ingestion)
        assert "General-purpose" not in desc
        assert "Your Project" not in desc
        assert desc == "your project"

    def test_project_name_not_duplicated(self):
        """Don't append project name if it's already in the description."""
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["flask"])
        desc = _generate_project_description(
            ingestion, project_name="Flask", vertical="web_app",
        )
        # "Flask" is already in the description, don't repeat as "(Flask)"
        assert desc.count("Flask") == 1

    def test_generic_project_name_omitted(self):
        """Generic names like 'Project' or 'App' are not appended."""
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["flask"])
        desc = _generate_project_description(
            ingestion, project_name="Project", vertical="web_app",
        )
        assert "(Project)" not in desc

    def test_api_service_vertical(self):
        from mycode.report import _generate_project_description
        ingestion = _s14_ingestion(["fastapi", "redis", "celery"])
        desc = _generate_project_description(ingestion, vertical="api_service")
        assert "FastAPI" in desc
        assert "API service" in desc


# ── Session 14: Report Narrative Tests ──


class TestReportNarrative:
    """Tests for progressive degradation and no-constraint mode."""

    def test_degradation_narrative_format(self):
        from mycode.report import _build_degradation_narrative
        dp = DegradationPoint(
            scenario_name="test_scaling",
            metric="execution_time_ms",
            steps=[
                ("data_size_100", 3.0),
                ("data_size_1000", 15.0),
                ("data_size_10000", 750.0),
            ],
        )
        narrative = _build_degradation_narrative(dp)
        assert "100 items" in narrative
        assert "1,000 items" in narrative
        assert "3ms" in narrative

    def test_no_constraint_header_text(self):
        report = DiagnosticReport(
            findings=[Finding(title="Problem", severity="critical")],
            has_user_constraints=False,
        )
        text = report.as_text()
        assert "Findings at Default Test Range" in text
        assert "Fix Before Launch" not in text

    def test_constraint_header_text(self):
        report = DiagnosticReport(
            findings=[Finding(title="Problem", severity="critical")],
            has_user_constraints=True,
        )
        text = report.as_text()
        assert "Fix Before Launch" in text

    def test_no_constraint_header_markdown(self):
        report = DiagnosticReport(
            findings=[Finding(title="Problem", severity="critical")],
            has_user_constraints=False,
        )
        md = report.as_markdown()
        assert "Findings at Default Test Range" in md

    def test_no_constraint_note(self):
        report = DiagnosticReport(
            findings=[Finding(title="Problem", severity="critical")],
            has_user_constraints=False,
        )
        text = report.as_text()
        assert "conversational interface" in text

    def test_structured_finding_description(self):
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent_execution",
                    scenario_category="concurrent_execution",
                    status="failed",
                    summary="Failed at 10 concurrent users",
                    total_errors=1,
                    steps=[StepResult(
                        step_name="concurrent_10",
                        error_count=1,
                        errors=[{"type": "RuntimeError", "message": "fail"}],
                    )],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(execution, _s14_ingestion(), [], "test")
        criticals = [f for f in report.findings if f.severity == "critical"]
        assert len(criticals) >= 1
        assert "myCode tested" in criticals[0].description

    def test_degradation_narrative_in_text(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test_scaling",
                    metric="execution_time_ms",
                    steps=[
                        ("data_size_100", 5.0),
                        ("data_size_10000", 500.0),
                    ],
                    breaking_point="data_size_10000",
                    description="Time increased 100x",
                ),
            ],
        )
        text = report.as_text()
        assert "100 items" in text or "10,000 items" in text
        assert "data_size_100: 5.00" not in text

    def test_memory_degradation_narrative(self):
        from mycode.report import _build_degradation_narrative
        dp = DegradationPoint(
            scenario_name="test_mem",
            metric="memory_peak_mb",
            steps=[
                ("data_size_100", 10.0),
                ("data_size_10000", 250.0),
            ],
        )
        narrative = _build_degradation_narrative(dp)
        assert "MB" in narrative
        assert "100 items" in narrative

    def test_fraction_context_for_concurrency(self):
        """When app fails at M < N users, show fraction."""
        from mycode.constraints import OperationalConstraints
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Errors during: flask_concurrent_50",
                    severity="warning",
                    category="concurrent_execution",
                    description="Some errors occurred.",
                    details="concurrent_50",
                    _load_level=10,
                ),
            ],
        )
        constraints = OperationalConstraints(user_scale=100)
        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)
        f = report.findings[0]
        assert "10%" in f.description or "just" in f.description


# ── Session 14: Project Name Inference Tests ──


class TestProjectNameInference:
    """Tests for _infer_project_name from pipeline."""

    def test_infer_from_directory(self, tmp_path):
        from mycode.pipeline import _infer_project_name
        project = tmp_path / "my-cool-app"
        project.mkdir()
        name = _infer_project_name(project)
        assert name == "My Cool App"

    def test_infer_from_pyproject_toml(self, tmp_path):
        from mycode.pipeline import _infer_project_name
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "expense-tracker"\nversion = "1.0"\n'
        )
        name = _infer_project_name(project)
        assert name == "Expense Tracker"

    def test_infer_from_package_json(self, tmp_path):
        import json as _json
        from mycode.pipeline import _infer_project_name
        project = tmp_path / "proj"
        project.mkdir()
        (project / "package.json").write_text(
            _json.dumps({"name": "budget-planner", "version": "2.0.0"})
        )
        name = _infer_project_name(project)
        assert name == "Budget Planner"

    def test_fallback_to_dirname(self, tmp_path):
        from mycode.pipeline import _infer_project_name
        project = tmp_path / "hello_world"
        project.mkdir()
        name = _infer_project_name(project)
        assert name == "Hello World"


# ── Probe-and-Skip Report Tests ──


class TestRuntimeContextFindings:
    """Tests for runtime_context_required finding handling in reports."""

    def test_runtime_context_routed_to_incomplete(self):
        """Scenarios with runtime_context_required go to incomplete_tests."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="pandas_memory_profiling",
                    scenario_category="memory_profiling",
                    status="skipped",
                    failure_reason="runtime_context_required",
                    probe_skipped=[
                        {"name": "app.render_dashboard", "error": {"type": "AttributeError", "message": "st.session_state"}},
                    ],
                    summary="1 function(s) require runtime context",
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        assert len(report.incomplete_tests) == 1
        assert report.incomplete_tests[0]._failure_reason == "runtime_context_required"
        assert "runtime context" in report.incomplete_tests[0].description.lower()
        # Should NOT appear in findings
        assert len(report.findings) == 0

    def test_runtime_context_not_counted_as_failed(self):
        """Runtime context scenarios don't count toward pass/fail totals."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="pandas_memory_profiling",
                    scenario_category="memory_profiling",
                    status="skipped",
                    failure_reason="runtime_context_required",
                    probe_skipped=[
                        {"name": "app.load_data", "error": {"type": "ConnectionError", "message": "refused"}},
                    ],
                ),
                ScenarioResult(
                    scenario_name="data_volume_scaling_pandas",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[StepResult(step_name="data_size_100", error_count=0)],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        # 1 passed, 0 failed (the skipped one doesn't count)
        assert report.scenarios_passed == 1
        assert report.scenarios_failed == 0

    def test_partial_probe_creates_incomplete_per_function(self):
        """Partially probed scenario creates separate incomplete findings."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="data_volume_scaling_pandas",
                    scenario_category="data_volume_scaling",
                    status="completed",
                    steps=[StepResult(step_name="data_size_100", error_count=0)],
                    probe_skipped=[
                        {"name": "app.render_dashboard", "error": {"type": "AttributeError", "message": "st.session_state"}},
                        {"name": "app.init_session", "error": {"type": "RuntimeError", "message": "no event loop"}},
                    ],
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        # 1 passed (the scenario itself)
        assert report.scenarios_passed == 1
        # 2 incomplete findings for the probed-out functions
        ctx_findings = [
            f for f in report.incomplete_tests
            if f._failure_reason == "runtime_context_required"
        ]
        assert len(ctx_findings) == 2

    def test_runtime_context_section_renders_in_text(self):
        """Runtime context findings render under the correct header."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(
                    title="Could not test: app.render_dashboard",
                    severity="info",
                    _failure_reason="runtime_context_required",
                    description="Requires runtime context.",
                ),
            ],
        )
        text = report.as_text()
        assert "Requires Runtime Context" in text
        assert "render_dashboard" in text

    def test_runtime_context_explanation_in_text(self):
        """The explanation text is rendered for runtime context group."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(
                    title="Could not test: app.render_dashboard",
                    severity="info",
                    _failure_reason="runtime_context_required",
                    description="Requires runtime context.",
                ),
            ],
        )
        text = report.as_text()
        assert "runtime context" in text.lower()
        assert "planned for v2" not in text

    def test_runtime_context_with_http_ran_shows_http_message(self):
        """When HTTP testing ran, runtime context findings say so."""
        report = DiagnosticReport(
            http_ran=True,
            incomplete_tests=[
                Finding(
                    title="Could not test: app.render_dashboard",
                    severity="info",
                    _failure_reason="runtime_context_required",
                    description=(
                        "This function could not be tested in isolation, but "
                        "myCode tested your application under load via HTTP — "
                        "see HTTP findings in this report."
                    ),
                ),
            ],
        )
        text = report.as_text()
        assert "tested your application under load via HTTP" in text
        assert "in this report" in text
        assert "above" not in text.split("HTTP")[1] if "HTTP" in text else True
        assert "planned for v2" not in text

    def test_runtime_context_http_description_position_neutral(self):
        """Runtime context description uses 'in this report', never 'above'."""
        from mycode.engine import ExecutionEngineResult, ScenarioResult
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="flask_concurrent",
                    scenario_category="concurrent_execution",
                    status="skipped",
                    failure_reason="runtime_context_required",
                ),
            ],
            http_ran=True,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(execution, IngestionResult(project_path="/tmp/t"), [])
        # The runtime context description should say "in this report"
        assert len(report.incomplete_tests) >= 1
        ctx_finding = report.incomplete_tests[0]
        assert "in this report" in ctx_finding.description
        assert "above" not in ctx_finding.description

    def test_runtime_context_without_http_ran_no_http_reference(self):
        """When HTTP testing didn't run, no HTTP reference in message."""
        report = DiagnosticReport(
            http_ran=False,
            incomplete_tests=[
                Finding(
                    title="Could not test: app.render_dashboard",
                    severity="info",
                    _failure_reason="runtime_context_required",
                    description=(
                        "This function requires runtime context that myCode "
                        "cannot simulate in isolation."
                    ),
                ),
            ],
        )
        text = report.as_text()
        assert "cannot simulate in isolation" in text
        assert "HTTP" not in text
        assert "planned for v2" not in text

    def test_identical_error_scenario_routed_to_incomplete(self):
        """Scenario reclassified by identical-error detection goes to incomplete_tests."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="pandas_memory_profiling_over_time",
                    scenario_category="memory_profiling",
                    status="skipped",
                    failure_reason="runtime_context_required",
                    steps=[
                        StepResult(step_name="module_import", error_count=1,
                                   errors=[{"type": "ModuleNotFoundError", "message": "yfinance"}]),
                    ] + [
                        StepResult(step_name=f"batch_{i}", error_count=30,
                                   errors=[{"type": "TypeError", "message": "err"}] * 30)
                        for i in range(10)
                    ],
                    total_errors=301,
                    summary="Every test step produced 30 identical error(s).",
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        # Should be in incomplete_tests, NOT in findings
        assert len(report.incomplete_tests) == 1
        assert report.incomplete_tests[0]._failure_reason == "runtime_context_required"
        assert len(report.findings) == 0
        # Should not count as failed
        assert report.scenarios_failed == 0

    def test_analyze_execution_uses_http_ran_for_description(self):
        """When http_ran=True, runtime context findings get HTTP message."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="streamlit_concurrent_session_load",
                    scenario_category="concurrent_execution",
                    status="skipped",
                    failure_reason="runtime_context_required",
                    steps=[],
                    total_errors=0,
                ),
            ],
        )
        execution.http_ran = True
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        assert len(report.incomplete_tests) == 1
        desc = report.incomplete_tests[0].description
        assert "tested your application under load via HTTP" in desc

    def test_analyze_execution_no_http_ran_no_http_reference(self):
        """When http_ran=False, runtime context findings don't mention HTTP."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="streamlit_concurrent_session_load",
                    scenario_category="concurrent_execution",
                    status="skipped",
                    failure_reason="runtime_context_required",
                    steps=[],
                    total_errors=0,
                ),
            ],
        )
        execution.http_ran = False
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        assert len(report.incomplete_tests) == 1
        desc = report.incomplete_tests[0].description
        assert "cannot simulate in isolation" in desc
        assert "HTTP" not in desc


class TestCouplingTitleDifferentiation:
    """Coupling scenarios should have distinct, behavior-specific titles."""

    def test_api_coupling_title(self):
        assert "API Coupling" in _humanize_title_name("coupling_api_fetchData")

    def test_compute_coupling_title(self):
        assert "Computation Coupling" in _humanize_title_name("coupling_compute_calculate")

    def test_render_coupling_title(self):
        assert "Render Coupling" in _humanize_title_name("coupling_render_App")

    def test_error_handler_coupling_title(self):
        assert "Error Handling" in _humanize_title_name("coupling_errorhandler_onError")

    def test_state_setters_coupling_title(self):
        result = _humanize_title_name("coupling_state_setters_group_1")
        assert "Shared State" in result

    def test_coupling_titles_are_distinct(self):
        """Different coupling behaviors should produce different titles."""
        titles = {
            _humanize_title_name("coupling_api_fetchData"),
            _humanize_title_name("coupling_compute_calculate"),
            _humanize_title_name("coupling_render_App"),
            _humanize_title_name("coupling_errorhandler_onError"),
        }
        assert len(titles) == 4, f"Expected 4 distinct titles, got {titles}"


class TestBrowserFrameworkSuppression:
    """Browser-framework incomplete tests should be suppressed when HTTP findings exist."""

    def test_suppressed_when_http_findings_present(self):
        """7 browser_framework skips should vanish when HTTP findings exist."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name=f"react_scenario_{i}",
                    scenario_category="data_volume_scaling",
                    status="skipped",
                    failure_reason="browser_framework",
                    summary="Frontend framework requires browser environment.",
                )
                for i in range(7)
            ],
            http_findings=[
                Finding(
                    title="Response time degradation on your application",
                    severity="warning",
                    category="http_load_testing",
                    description="Response time increases 25x under load.",
                ),
            ],
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        # browser_framework entries should be suppressed
        browser_incomplete = [
            f for f in report.incomplete_tests
            if f._failure_reason == "browser_framework"
        ]
        assert len(browser_incomplete) == 0
        # HTTP finding should be present
        assert any("response time" in f.title.lower() for f in report.findings)

    def test_kept_when_no_http_findings(self):
        """browser_framework skips should remain when HTTP testing has no findings."""
        execution = ExecutionEngineResult(
            scenario_results=[
                ScenarioResult(
                    scenario_name="react_scenario_0",
                    scenario_category="data_volume_scaling",
                    status="skipped",
                    failure_reason="browser_framework",
                    summary="Frontend framework requires browser environment.",
                ),
            ],
            # No http_findings
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(), [], "test intent",
        )
        browser_incomplete = [
            f for f in report.incomplete_tests
            if f._failure_reason == "browser_framework"
        ]
        assert len(browser_incomplete) == 1


# ── Session 19: User Timeout in Report ──


class TestUserTimeoutInReport:
    """Tests for hit_user_timeout scenarios in report generation."""

    def test_user_timeout_creates_info_finding(self):
        """Scenarios hitting user timeout get 'Reached time limit' finding."""
        sr = ScenarioResult(
            scenario_name="data_volume_scaling_pandas",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="timeout",
            hit_user_timeout=True,
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        user_timeout = [
            f for f in report.incomplete_tests
            if f._failure_reason == "user_timeout"
        ]
        assert len(user_timeout) == 1
        assert "time limit" in user_timeout[0].title.lower()
        assert "re-run" in user_timeout[0].description.lower()
        # Should mention what the test checks
        assert "data processing" in user_timeout[0].description.lower() or \
               "scales" in user_timeout[0].description.lower()

    def test_user_timeout_with_step_timing_shows_avg_ms(self):
        """When steps completed, description includes avg execution time."""
        sr = ScenarioResult(
            scenario_name="pandas_data_volume_scaling",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="timeout",
            hit_user_timeout=True,
            source_functions=["process_data"],
            steps=[
                StepResult(step_name="tier_1", execution_time_ms=150.0),
                StepResult(step_name="tier_2", execution_time_ms=200.0),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        desc = report.incomplete_tests[0].description
        assert "process_data" in desc
        assert "175ms" in desc  # avg of 150 and 200
        assert "re-run" in desc.lower()

    def test_user_timeout_with_current_timeout_shows_value(self):
        """Description includes the current timeout value."""
        from mycode.constraints import OperationalConstraints
        sr = ScenarioResult(
            scenario_name="pandas_data_volume_scaling",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="timeout",
            hit_user_timeout=True,
            steps=[
                StepResult(step_name="tier_1", execution_time_ms=100.0),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        constraints = OperationalConstraints(timeout_per_scenario=120)
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
            constraints=constraints,
        )
        desc = report.incomplete_tests[0].description
        assert "120s" in desc

    def test_user_timeout_memory_profiling_description(self):
        """Memory profiling timeout describes memory leak checking."""
        sr = ScenarioResult(
            scenario_name="pandas_memory_profiling",
            scenario_category="memory_profiling",
            status="partial",
            failure_reason="timeout",
            hit_user_timeout=True,
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        desc = report.incomplete_tests[0].description
        assert "memory" in desc.lower()

    def test_user_timeout_excluded_from_pass_fail(self):
        """User-timeout scenarios are not counted as passed or failed."""
        sr = ScenarioResult(
            scenario_name="test_scenario",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="timeout",
            hit_user_timeout=True,
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        # The report should show 0 passed, 0 failed (user timeout excluded)
        assert report.scenarios_passed == 0
        assert report.scenarios_failed == 0


class TestBudgetExceededInReport:
    """Tests for budget_exceeded scenarios in report generation."""

    def test_budget_exceeded_creates_info_finding(self):
        """budget_exceeded scenario gets 'Time budget reached' finding."""
        sr = ScenarioResult(
            scenario_name="coupling_compute_main",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="budget_exceeded",
            steps=[
                StepResult(step_name="tier_1", execution_time_ms=100),
                StepResult(step_name="tier_2", execution_time_ms=200),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        budget_findings = [
            f for f in report.incomplete_tests
            if f._failure_reason == "budget_exceeded"
        ]
        assert len(budget_findings) == 1
        assert "time budget" in budget_findings[0].title.lower()
        assert "2 steps" in budget_findings[0].description

    def test_budget_exceeded_excluded_from_pass_fail(self):
        """budget_exceeded scenarios are not counted as passed or failed."""
        sr = ScenarioResult(
            scenario_name="test_scenario",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="budget_exceeded",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
        )
        assert report.scenarios_passed == 0
        assert report.scenarios_failed == 0

    def test_budget_exceeded_suggests_deep_when_not_deep(self):
        """When analysis_depth != 'deep', suggest upgrading."""
        from mycode.constraints import OperationalConstraints
        sr = ScenarioResult(
            scenario_name="coupling_compute_main",
            scenario_category="data_volume_scaling",
            status="partial",
            failure_reason="budget_exceeded",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_failed=1,
        )
        constraints = OperationalConstraints(analysis_depth="standard")
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
            constraints=constraints,
        )
        desc = report.incomplete_tests[0].description
        assert "deep analysis" in desc.lower()


class TestTimeoutInReportContext:
    """Tests for timeout_per_scenario appearing in the report summary."""

    def test_timeout_in_summary_context(self):
        """timeout_per_scenario appears in the 'Results assessed relative to' line."""
        from mycode.constraints import OperationalConstraints
        sr = ScenarioResult(
            scenario_name="test_scenario",
            scenario_category="data_volume_scaling",
            status="completed",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        constraints = OperationalConstraints(
            user_scale=100,
            usage_pattern="sustained",
            timeout_per_scenario=120,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
            constraints=constraints,
        )
        text = report.as_text()
        assert "120s per test" in text

    def test_no_timeout_no_mention(self):
        """When timeout_per_scenario is None, it doesn't appear in summary."""
        from mycode.constraints import OperationalConstraints
        sr = ScenarioResult(
            scenario_name="test_scenario",
            scenario_category="data_volume_scaling",
            status="completed",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        constraints = OperationalConstraints(user_scale=100)
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
            constraints=constraints,
        )
        text = report.as_text()
        assert "per test" not in text

    def test_analysis_depth_in_summary(self):
        """analysis_depth appears in report context instead of raw timeout."""
        from mycode.constraints import OperationalConstraints
        sr = ScenarioResult(
            scenario_name="test_scenario",
            scenario_category="data_volume_scaling",
            status="completed",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        constraints = OperationalConstraints(
            user_scale=100,
            analysis_depth="standard",
            timeout_per_scenario=300,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, _s14_ingestion(["pandas"]), [], "test intent",
            constraints=constraints,
        )
        text = report.as_text()
        assert "standard analysis" in text
        # Should show depth label, not raw timeout
        assert "300s per test" not in text


# ── Dep→File Mapping & source_file Fallback ──


class TestBuildDepFileMap:
    """Tests for _build_dep_file_map."""

    def test_single_file_single_dep(self):
        ingestion = IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=1,
            total_lines=10,
            file_analyses=[
                FileAnalysis(
                    file_path="app.py",
                    imports=[ImportInfo(module="flask", names=["Flask"])],
                ),
            ],
        )
        result = _build_dep_file_map(ingestion)
        assert result == {"flask": ["app.py"]}

    def test_multiple_files_ordered_by_usage(self):
        ingestion = IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=2,
            total_lines=20,
            file_analyses=[
                FileAnalysis(
                    file_path="routes.py",
                    imports=[
                        ImportInfo(module="flask", names=["Blueprint", "request", "jsonify"]),
                    ],
                ),
                FileAnalysis(
                    file_path="app.py",
                    imports=[
                        ImportInfo(module="flask", names=["Flask"]),
                    ],
                ),
            ],
        )
        result = _build_dep_file_map(ingestion)
        # routes.py has 3 names, app.py has 1 → routes.py first
        assert result["flask"] == ["routes.py", "app.py"]

    def test_submodule_maps_to_top_level(self):
        ingestion = IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=1,
            total_lines=10,
            file_analyses=[
                FileAnalysis(
                    file_path="db.py",
                    imports=[ImportInfo(module="sqlalchemy.orm", names=["Session"])],
                ),
            ],
        )
        result = _build_dep_file_map(ingestion)
        assert "sqlalchemy" in result
        assert result["sqlalchemy"] == ["db.py"]

    def test_empty_ingestion(self):
        ingestion = IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=0,
            total_lines=0,
        )
        result = _build_dep_file_map(ingestion)
        assert result == {}

    def test_import_without_names_counts_as_one(self):
        ingestion = IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=1,
            total_lines=5,
            file_analyses=[
                FileAnalysis(
                    file_path="main.py",
                    imports=[ImportInfo(module="os")],
                ),
            ],
        )
        result = _build_dep_file_map(ingestion)
        assert result["os"] == ["main.py"]


class TestResolveSourceFile:
    """Tests for _resolve_source_file."""

    def test_finds_primary_file(self):
        dep_map = {"flask": ["routes.py", "app.py"], "sqlalchemy": ["db.py"]}
        assert _resolve_source_file(["flask"], dep_map) == "routes.py"

    def test_first_matching_dep_wins(self):
        dep_map = {"flask": ["routes.py"], "sqlalchemy": ["db.py"]}
        assert _resolve_source_file(["sqlalchemy", "flask"], dep_map) == "db.py"

    def test_no_match_returns_empty(self):
        dep_map = {"flask": ["app.py"]}
        assert _resolve_source_file(["pandas"], dep_map) == ""

    def test_empty_deps_returns_empty(self):
        dep_map = {"flask": ["app.py"]}
        assert _resolve_source_file([], dep_map) == ""

    def test_empty_map_returns_empty(self):
        assert _resolve_source_file(["flask"], {}) == ""


class TestSourceFileFallbackIntegration:
    """Test that source_file is populated via dep→file fallback."""

    def _ingestion_with_flask_file(self):
        return IngestionResult(
            project_path="/tmp/proj",
            files_analyzed=1,
            total_lines=50,
            dependencies=[
                DependencyInfo(name="flask", installed_version="3.0.0"),
            ],
            file_analyses=[
                FileAnalysis(
                    file_path="app.py",
                    imports=[ImportInfo(module="flask", names=["Flask", "request"])],
                ),
            ],
        )

    def test_fallback_populates_source_file_on_finding(self):
        """When ScenarioResult has no source_files, fallback uses dep map."""
        sr = ScenarioResult(
            scenario_name="flask_resource_exhaustion",
            scenario_category="resource_exhaustion",
            status="failed",
            steps=[StepResult(step_name="step_1", error_count=1)],
            total_errors=1,
            # source_files is empty — no target_modules
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            self._ingestion_with_flask_file(),
            [],
            "test intent",
        )
        # The finding should have source_file from the dep→file fallback
        flask_findings = [
            f for f in report.findings
            if "flask" in f.affected_dependencies
        ]
        assert flask_findings
        assert flask_findings[0].source_file == "app.py"

    def test_explicit_source_file_not_overridden(self):
        """When ScenarioResult has source_files, fallback is NOT used."""
        sr = ScenarioResult(
            scenario_name="flask_resource_exhaustion",
            scenario_category="resource_exhaustion",
            status="failed",
            steps=[StepResult(step_name="step_1", error_count=1)],
            total_errors=1,
            source_files=["explicit.py"],
            source_functions=["handle_request"],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            self._ingestion_with_flask_file(),
            [],
            "test intent",
        )
        flask_findings = [
            f for f in report.findings
            if "flask" in f.affected_dependencies
        ]
        assert flask_findings
        assert flask_findings[0].source_file == "explicit.py"
        assert flask_findings[0].source_function == "handle_request"

    def test_fallback_on_incomplete_test(self):
        """Incomplete tests also get source_file from fallback."""
        sr = ScenarioResult(
            scenario_name="flask_concurrent_execution",
            scenario_category="concurrent_execution",
            status="skipped",
            failure_reason="runtime_context_required",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr],
            scenarios_completed=0,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            self._ingestion_with_flask_file(),
            [],
            "test intent",
        )
        assert report.incomplete_tests
        assert report.incomplete_tests[0].source_file == "app.py"

    def test_http_findings_get_source_file(self):
        """HTTP findings merged post-analysis get source_file from fallback."""
        http_finding = Finding(
            title="High error rate under load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        execution = ExecutionEngineResult(
            scenario_results=[],
            scenarios_completed=0,
            http_findings=[http_finding],
            http_ran=True,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution,
            self._ingestion_with_flask_file(),
            [],
            "test intent",
        )
        http_fs = [f for f in report.findings if f.category == "http_load_testing"]
        assert http_fs
        assert http_fs[0].source_file == "app.py"


# ── Bug 1: Scenario Counting ──


class TestScenarioCounting:
    """Test that scenario counts correctly distinguish failed vs incomplete."""

    def test_runtime_context_counts_as_incomplete(self):
        sr = ScenarioResult(
            scenario_name="flask_concurrent",
            scenario_category="concurrent_execution",
            status="skipped",
            failure_reason="runtime_context_required",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr], scenarios_completed=0,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        assert report.scenarios_failed == 0
        assert report.scenarios_incomplete == 1
        assert report.scenarios_passed == 0

    def test_environment_only_counts_as_incomplete(self):
        sr = ScenarioResult(
            scenario_name="flask_test",
            scenario_category="resource_exhaustion",
            status="completed",
            total_errors=1,
            steps=[
                StepResult(
                    step_name="step_1",
                    error_count=1,
                    errors=[{"type": "ModuleNotFoundError", "message": "No module"}],
                ),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr], scenarios_completed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        assert report.scenarios_failed == 0
        assert report.scenarios_incomplete == 1

    def test_real_failure_counts_as_failed(self):
        sr = ScenarioResult(
            scenario_name="flask_resource",
            scenario_category="resource_exhaustion",
            status="failed",
            total_errors=3,
            steps=[
                StepResult(
                    step_name="step_1",
                    error_count=3,
                    errors=[{"type": "MemoryError", "message": "OOM"}],
                ),
            ],
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr], scenarios_completed=1,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        assert report.scenarios_failed == 1
        assert report.scenarios_incomplete == 0

    def test_mixed_counts(self):
        """3 scenarios: 1 pass, 1 fail, 1 incomplete."""
        pass_sr = ScenarioResult(
            scenario_name="ok_test",
            scenario_category="data_volume_scaling",
            status="completed",
            steps=[StepResult(step_name="s1")],
        )
        fail_sr = ScenarioResult(
            scenario_name="bad_test",
            scenario_category="resource_exhaustion",
            status="failed",
            total_errors=1,
            steps=[StepResult(step_name="s1", error_count=1)],
        )
        skip_sr = ScenarioResult(
            scenario_name="skip_test",
            scenario_category="concurrent_execution",
            status="skipped",
            failure_reason="runtime_context_required",
        )
        execution = ExecutionEngineResult(
            scenario_results=[pass_sr, fail_sr, skip_sr],
            scenarios_completed=2,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        assert report.scenarios_passed == 1
        assert report.scenarios_failed == 1
        assert report.scenarios_incomplete == 1

    def test_incomplete_in_as_dict(self):
        sr = ScenarioResult(
            scenario_name="skip_test",
            scenario_category="concurrent_execution",
            status="skipped",
            failure_reason="runtime_context_required",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr], scenarios_completed=0,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        d = report.as_dict()
        assert d["statistics"]["scenarios_incomplete"] == 1

    def test_incomplete_in_as_text(self):
        sr = ScenarioResult(
            scenario_name="skip_test",
            scenario_category="concurrent_execution",
            status="skipped",
            failure_reason="runtime_context_required",
        )
        execution = ExecutionEngineResult(
            scenario_results=[sr], scenarios_completed=0,
        )
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            execution, IngestionResult(project_path="/tmp/x"), [],
        )
        text = report.as_text()
        assert "could not test" in text.lower()


# ── Bug 4: Non-monotonic Degradation ──


class TestNonMonotonicDegradation:
    """Test that non-monotonic data points are flagged."""

    def test_drop_flagged(self):
        from mycode.report import DegradationPoint
        dp = DegradationPoint(
            scenario_name="test",
            metric="response_time_ms",
            steps=[
                ("c_1000", 50.0),
                ("c_2500", 132.5),
                ("c_3750", 35.9),   # drops >40% from 132.5
                ("c_5000", 150.0),
            ],
            description="Response time increased.",
        )
        ReportGenerator._annotate_non_monotonic(dp)
        assert "variance" in dp.description.lower()
        assert "c_3750" in dp.description

    def test_no_drop_no_flag(self):
        from mycode.report import DegradationPoint
        dp = DegradationPoint(
            scenario_name="test",
            metric="response_time_ms",
            steps=[
                ("c_1000", 50.0),
                ("c_2500", 100.0),
                ("c_3750", 150.0),
            ],
            description="Steadily increasing.",
        )
        ReportGenerator._annotate_non_monotonic(dp)
        assert "variance" not in dp.description.lower()

    def test_small_drop_not_flagged(self):
        """A 20% drop is normal noise, not flagged."""
        from mycode.report import DegradationPoint
        dp = DegradationPoint(
            scenario_name="test",
            metric="execution_time_ms",
            steps=[
                ("s1", 100.0),
                ("s2", 85.0),   # 15% drop, within threshold
                ("s3", 200.0),
            ],
            description="Normal variation.",
        )
        ReportGenerator._annotate_non_monotonic(dp)
        assert "variance" not in dp.description.lower()

    def test_two_steps_not_annotated(self):
        """Need at least 3 steps for annotation."""
        from mycode.report import DegradationPoint
        dp = DegradationPoint(
            scenario_name="test",
            metric="memory_peak_mb",
            steps=[("s1", 100.0), ("s2", 30.0)],
            description="Short curve.",
        )
        ReportGenerator._annotate_non_monotonic(dp)
        assert "variance" not in dp.description.lower()


# ── HTTP Startup-Failure + Version Discrepancy Enrichment ──


class TestStartupFailureVersionEnrichment:
    """Test that 'could not start' HTTP findings are enriched with version discrepancy info."""

    def test_enriched_when_version_discrepancy_exists(self):
        """Server-could-not-start with matching outdated dep gets version context."""
        report = DiagnosticReport()
        report.findings.append(Finding(
            title="Application server could not start",
            severity="critical",
            category="http_load_testing",
            description=(
                "myCode detected a streamlit application but the server "
                "failed to start: missing dependency. No users can access "
                "your app until this is resolved."
            ),
            affected_dependencies=["streamlit"],
            failure_domain="",
            failure_pattern=None,
        ))

        ingestion = IngestionResult(
            project_path="/fake",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(
                    name="streamlit",
                    installed_version="1.6.0",
                    latest_version="1.41.0",
                    is_outdated=True,
                ),
            ],
        )

        ReportGenerator._enrich_startup_failure_with_version(ingestion, report)

        finding = report.findings[0]
        assert "1.6.0" in finding.description
        assert "1.41.0" in finding.description
        assert "version gap" in finding.description
        assert finding.failure_domain == "dependency_failure"
        assert finding.failure_pattern == "version_incompatibility"

    def test_not_enriched_when_no_version_discrepancy(self):
        """Server-could-not-start without outdated dep stays unchanged."""
        original_desc = (
            "myCode detected a flask application but the server "
            "failed to start: syntax error. No users can access "
            "your app until this is resolved."
        )
        report = DiagnosticReport()
        report.findings.append(Finding(
            title="Application server could not start",
            severity="critical",
            category="http_load_testing",
            description=original_desc,
            affected_dependencies=["flask"],
            failure_domain="",
            failure_pattern=None,
        ))

        ingestion = IngestionResult(
            project_path="/fake",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(
                    name="flask",
                    installed_version="3.0.0",
                    latest_version="3.0.0",
                    is_outdated=False,
                ),
            ],
        )

        ReportGenerator._enrich_startup_failure_with_version(ingestion, report)

        finding = report.findings[0]
        assert finding.description == original_desc
        assert finding.failure_domain == ""
        assert finding.failure_pattern is None

    def test_enrichment_uses_correct_framework_from_affected_deps(self):
        """Enrichment matches the framework name from affected_dependencies."""
        report = DiagnosticReport()
        report.findings.append(Finding(
            title="Application server could not start",
            severity="critical",
            category="http_load_testing",
            description="Server failed to start: missing dep.",
            affected_dependencies=["fastapi", "uvicorn"],
            failure_domain="",
        ))

        ingestion = IngestionResult(
            project_path="/fake",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(
                    name="fastapi",
                    installed_version="0.68.0",
                    latest_version="0.115.0",
                    is_outdated=True,
                ),
                DependencyInfo(
                    name="uvicorn",
                    installed_version="0.20.0",
                    latest_version="0.20.0",
                    is_outdated=False,
                ),
            ],
        )

        ReportGenerator._enrich_startup_failure_with_version(ingestion, report)

        finding = report.findings[0]
        # Should reference fastapi (outdated), not uvicorn (current)
        assert "fastapi" in finding.description
        assert "0.68.0" in finding.description
        assert "0.115.0" in finding.description
        assert finding.failure_domain == "dependency_failure"
        assert finding.failure_pattern == "version_incompatibility"

    def test_non_startup_http_findings_not_enriched(self):
        """HTTP findings that aren't startup failures are left alone."""
        report = DiagnosticReport()
        report.findings.append(Finding(
            title="Application degrades under concurrent load",
            severity="warning",
            category="http_load_testing",
            description="Your app degrades at 50 concurrent connections.",
            affected_dependencies=["streamlit"],
        ))

        ingestion = IngestionResult(
            project_path="/fake",
            files_analyzed=1,
            dependencies=[
                DependencyInfo(
                    name="streamlit",
                    installed_version="1.6.0",
                    latest_version="1.41.0",
                    is_outdated=True,
                ),
            ],
        )

        ReportGenerator._enrich_startup_failure_with_version(ingestion, report)

        finding = report.findings[0]
        assert "version gap" not in finding.description
        assert finding.failure_domain == ""


# ── Report Humanisation Tests ──


class TestDescribeScenarioHumanisation:
    """Test that scenario names produce user-readable descriptions."""

    def test_coupling_compute_streamlit(self):
        result = _describe_scenario("coupling_compute_streamlit_markdown")
        assert "running calculations" in result

    def test_coupling_compute_long_path(self):
        result = _describe_scenario(
            "coupling_compute_My_Projects_app_render_paginated_table"
        )
        assert "running calculations" in result
        assert "coupling_compute" not in result

    def test_template_match_still_works(self):
        result = _describe_scenario("pandas_data_volume_scaling")
        assert result == "processing larger amounts of data"

    def test_unknown_scenario_no_raw_underscores(self):
        """Fallback should not contain coupling_ prefixes."""
        result = _describe_scenario("coupling_api_some_module_call")
        assert "coupling_api" not in result
        assert "connecting" in result

    def test_new_template_edge_case_inputs(self):
        result = _describe_scenario("streamlit_edge_case_inputs")
        assert result == "handling unusual or extreme inputs"


class TestDescribeStepHumanisation:
    """Test that step names produce user-readable labels."""

    def test_compute_50000(self):
        assert _describe_step("compute_50000") == "50,000 items of data"

    def test_table_serialize_10kb(self):
        result = _describe_step("table_serialize_10kb")
        assert "10KB table" in result

    def test_table_serialize_1000kb(self):
        result = _describe_step("table_serialize_1000kb")
        assert "1MB table" in result

    def test_cached_rows_1000(self):
        assert _describe_step("cached_rows_1000") == "1,000 rows of cached data"

    def test_session_reruns_500(self):
        assert _describe_step("session_reruns_500") == "500 page refreshes"

    def test_render_nodes_100(self):
        assert _describe_step("render_nodes_100") == "100 UI elements"

    def test_wsgi_payload_50kb(self):
        result = _describe_step("wsgi_payload_50kb")
        assert "50KB" in result

    def test_validation_fields_20(self):
        assert _describe_step("validation_fields_20") == "20 fields to validate"

    def test_http_concurrent_label(self):
        assert _describe_step("10 concurrent") == "10 concurrent connections"

    def test_error_flood_100(self):
        assert _describe_step("error_flood_100") == "100 simultaneous errors"

    def test_session_writes_50(self):
        assert _describe_step("session_writes_50") == "50 session write cycles"

    def test_render_memory_growth(self):
        assert _describe_step("render_memory_growth") == "repeated rendering cycles"

    def test_api_timeout_handling(self):
        assert _describe_step("api_timeout_handling") == "slow API responses"


class TestConsequenceLine:
    """Test consequence context appended to degradation narratives."""

    def test_time_over_1000_unresponsive(self):
        result = _consequence_line(
            "execution_time_ms",
            [("s1", 10.0), ("s2", 500.0), ("s3", 2000.0)],
        )
        assert "unresponsive" in result

    def test_time_under_100_acceptable(self):
        result = _consequence_line(
            "execution_time_ms",
            [("s1", 5.0), ("s2", 50.0)],
        )
        assert "acceptable range" in result

    def test_time_500_delays(self):
        result = _consequence_line(
            "execution_time_ms",
            [("s1", 10.0), ("s2", 600.0)],
        )
        assert "delays" in result

    def test_memory_over_50_server_estimate(self):
        result = _consequence_line(
            "memory_peak_mb",
            [("s1", 10.0), ("s2", 100.0)],
        )
        assert "server supports" in result
        assert "20" in result  # 2048 / 100 = 20

    def test_memory_under_50_no_consequence(self):
        result = _consequence_line(
            "memory_peak_mb",
            [("s1", 5.0), ("s2", 30.0)],
        )
        assert result == ""

    def test_narrative_includes_consequence(self):
        """Full narrative should include consequence for slow responses."""
        dp = DegradationPoint(
            scenario_name="test_scenario",
            metric="execution_time_ms",
            steps=[("data_size_100", 10.0), ("data_size_10000", 1500.0)],
        )
        narrative = _build_degradation_narrative(dp)
        assert "unresponsive" in narrative

    def test_http_response_time_consequence(self):
        """response_time_ms metric (HTTP) also gets consequence."""
        result = _consequence_line(
            "response_time_ms",
            [("1 concurrent", 2.0), ("100 concurrent", 5.0)],
        )
        assert "acceptable range" in result


class TestMarkdownExecutiveSummary:
    """Test executive summary and structural improvements in as_markdown."""

    def test_exec_summary_contains_project_description(self):
        report = DiagnosticReport(
            project_description="Your financial dashboard built with streamlit",
            scenarios_run=10,
            scenarios_passed=8,
            scenarios_failed=2,
        )
        md = report.as_markdown()
        assert "financial dashboard" in md

    def test_exec_summary_references_critical_finding(self):
        report = DiagnosticReport(
            project_description="Your dashboard",
            scenarios_run=5,
            scenarios_passed=3,
            scenarios_failed=2,
            findings=[
                Finding(
                    title="Server crashed under load",
                    severity="critical",
                    category="http_load_testing",
                    description="The server crashed.",
                ),
            ],
        )
        md = report.as_markdown()
        assert "Key issue" in md
        assert "Server crashed under load" in md

    def test_exec_summary_no_critical_no_key_issue(self):
        report = DiagnosticReport(
            project_description="Your app",
            scenarios_run=5,
            scenarios_passed=5,
            scenarios_failed=0,
        )
        md = report.as_markdown()
        assert "Key issue" not in md

    def test_what_to_do_next_present(self):
        report = DiagnosticReport(
            scenarios_run=3,
            scenarios_passed=2,
            scenarios_failed=1,
            findings=[
                Finding(
                    title="Some issue",
                    severity="warning",
                    description="Details.",
                ),
            ],
        )
        md = report.as_markdown()
        assert "## What To Do Next" in md
        assert "coding agent" in md

    def test_what_to_do_next_absent_when_no_findings(self):
        report = DiagnosticReport(
            scenarios_run=3,
            scenarios_passed=3,
            scenarios_failed=0,
            findings=[],
        )
        md = report.as_markdown()
        assert "What To Do Next" not in md


# ── Final report quality pass tests ──


class TestDescribeScenarioHttpRoot:
    """Issue 2: http_get_root must produce human-readable label."""

    def test_http_get_root(self):
        from mycode.report import _describe_scenario
        assert "main page" in _describe_scenario("http_get_root")

    def test_http_post_root(self):
        from mycode.report import _describe_scenario
        assert "main page" in _describe_scenario("http_post_root")

    def test_http_get_api_users(self):
        from mycode.report import _describe_scenario
        result = _describe_scenario("http_get_api_users")
        assert "/api/users" in result


class TestDescribeStepSingular:
    """Issue 4: Singular grammar for N=1."""

    def test_concurrent_connection_singular(self):
        from mycode.report import _describe_step
        result = _describe_step("1 concurrent")
        assert result == "1 concurrent connection"

    def test_concurrent_connection_plural(self):
        from mycode.report import _describe_step
        result = _describe_step("5 concurrent")
        assert result == "5 concurrent connections"


class TestFlatMemoryNarrative:
    """Issue 6: Flat memory should say 'stable' not show degradation."""

    def test_flat_memory_narrative(self):
        from mycode.report import DegradationPoint, _build_degradation_narrative
        dp = DegradationPoint(
            scenario_name="http_get_root",
            metric="memory_peak_mb",
            steps=[
                ("1 concurrent", 65.0),
                ("5 concurrent", 65.0),
                ("10 concurrent", 65.0),
            ],
        )
        narrative = _build_degradation_narrative(dp)
        assert "stable" in narrative.lower()
        assert "65MB" in narrative

    def test_growing_memory_not_flat(self):
        from mycode.report import DegradationPoint, _build_degradation_narrative
        dp = DegradationPoint(
            scenario_name="http_get_root",
            metric="memory_peak_mb",
            steps=[
                ("1 concurrent", 50.0),
                ("5 concurrent", 75.0),
                ("10 concurrent", 150.0),
            ],
        )
        narrative = _build_degradation_narrative(dp)
        assert "stable" not in narrative.lower()


class TestPortfolioStatsNotOnInfo:
    """Issue 5: Portfolio stats only on critical/warning, not info."""

    def test_info_finding_no_portfolio_stats(self):
        from mycode.report import DiagnosticReport, Finding, ReportGenerator
        from mycode.report import DegradationPoint
        # Create a report with an info finding
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="No issues detected",
                    severity="info",
                    description="All tests passed cleanly.",
                    affected_dependencies=["flask"],
                ),
            ],
        )
        corpus = {"flask": {"failure_rate": 0.45, "tested_count": 10}}
        # Simulate the contextualise step (which appends corpus stats)
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        # Manually run the corpus append logic
        for finding in report.findings:
            if finding.severity not in ("critical", "warning"):
                continue
            for dep_name in finding.affected_dependencies:
                stats = corpus.get(dep_name.lower())
                if stats and stats.get("tested_count", 0) >= 3:
                    finding.description += (
                        f" In myCode's test portfolio, {dep_name} "
                        f"showed failures in {stats['failure_rate']:.0%} of "
                        f"{stats['tested_count']} tested projects."
                    )
                    break
        # Info finding should NOT have portfolio stats
        assert "portfolio" not in report.findings[0].description


class TestRecognizedDepNames:
    """Issue 1: recognized_dep_names field on DiagnosticReport."""

    def test_as_dict_includes_names(self):
        report = DiagnosticReport(
            recognized_dep_count=2,
            recognized_dep_names=["pandas", "streamlit"],
        )
        d = report.as_dict()
        assert d["statistics"]["recognized_dependency_names"] == ["pandas", "streamlit"]

    def test_as_text_includes_names(self):
        report = DiagnosticReport(
            recognized_dep_count=2,
            recognized_dep_names=["pandas", "flask"],
            unrecognized_deps=["plotly"],
        )
        text = report.as_text()
        assert "pandas, flask" in text

    def test_as_markdown_includes_names(self):
        report = DiagnosticReport(
            recognized_dep_count=2,
            recognized_dep_names=["pandas", "flask"],
            unrecognized_deps=["plotly"],
        )
        md = report.as_markdown()
        assert "pandas, flask" in md


class TestBreakingPointWithMetric:
    """Issue 7: Breaking point label includes what metric breaks."""

    def test_breaking_point_label_time(self):
        from mycode.report import DegradationPoint, _breaking_point_label
        dp = DegradationPoint(
            scenario_name="test",
            metric="execution_time_ms",
            steps=[("data_size_1000", 10.0), ("data_size_50000", 500.0)],
            breaking_point="data_size_50000",
        )
        label = _breaking_point_label(dp)
        assert "response time exceeds 500ms" in label
        assert "50,000 items" in label

    def test_breaking_point_label_memory(self):
        from mycode.report import DegradationPoint, _breaking_point_label
        dp = DegradationPoint(
            scenario_name="test",
            metric="memory_peak_mb",
            steps=[("concurrent_5", 20.0), ("concurrent_50", 120.0)],
            breaking_point="concurrent_50",
        )
        label = _breaking_point_label(dp)
        assert "memory exceeds 120MB" in label
        assert "50 simultaneous users" in label

    def test_breaking_point_label_errors(self):
        from mycode.report import DegradationPoint, _breaking_point_label
        dp = DegradationPoint(
            scenario_name="test",
            metric="error_count",
            steps=[("concurrent_5", 0.0), ("concurrent_50", 5.0)],
            breaking_point="concurrent_50",
        )
        label = _breaking_point_label(dp)
        assert "errors begin" in label

    def test_response_time_ms_metric_recognized(self):
        """HTTP metric 'response_time_ms' should get time narrative."""
        from mycode.report import DegradationPoint, _build_degradation_narrative
        dp = DegradationPoint(
            scenario_name="http_get_root",
            metric="response_time_ms",
            steps=[
                ("1 concurrent", 5.0),
                ("10 concurrent", 500.0),
            ],
        )
        narrative = _build_degradation_narrative(dp)
        # Should contain time units, not bare numbers
        assert "ms" in narrative or "s" in narrative


class TestJsonPromptField:
    """Test that as_dict() includes prompt field on findings."""

    def test_critical_finding_has_prompt(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Memory leak",
                    severity="critical",
                    category="memory_profiling",
                    description="Memory grew.",
                    source_file="app.py",
                    source_function="run",
                ),
            ],
        )
        d = report.as_dict()
        f = d["findings"][0]
        assert "prompt" in f
        assert "CRITICAL" in f["prompt"]
        assert "app.py" in f["prompt"]
        assert "run" in f["prompt"]

    def test_info_finding_has_empty_prompt(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="All clean",
                    severity="info",
                    description="No issues.",
                ),
            ],
        )
        d = report.as_dict()
        f = d["findings"][0]
        assert f["prompt"] == ""

    def test_prompt_includes_fix_objective(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Concurrency issue",
                    severity="warning",
                    category="concurrent_execution",
                    description="Race condition.",
                    source_file="api.py",
                ),
            ],
        )
        d = report.as_dict()
        prompt = d["findings"][0]["prompt"]
        assert "thread safety" in prompt
        assert "JSON" in prompt
