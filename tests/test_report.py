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
from mycode.ingester import DependencyInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.report import (
    DegradationPoint,
    DiagnosticReport,
    Finding,
    ReportError,
    ReportGenerator,
    _human_metric,
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
        failed_findings = [f for f in critical if "failed" in f.title.lower()]
        assert len(failed_findings) >= 1
        assert any("sqlalchemy" in f.title.lower() for f in failed_findings)

    def test_resource_cap_creates_critical_finding(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        cap_findings = [
            f for f in report.findings if "resource" in f.title.lower()
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
            if f.severity == "warning" and "error" in f.title.lower()
        ]
        assert len(error_findings) >= 1

    def test_failure_indicators_create_warning(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        indicator_findings = [
            f for f in report.findings
            if "indicator" in f.title.lower()
        ]
        assert len(indicator_findings) >= 1
        assert "memory_growth_unbounded" in indicator_findings[0].description

    def test_error_type_summarization(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        error_findings = [
            f for f in report.findings
            if "errors during" in f.title.lower()
        ]
        assert len(error_findings) >= 1
        assert "MemoryError" in error_findings[0].details

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

    def test_unrecognized_creates_info_finding(
        self, clean_execution, simple_ingestion, profile_matches,
    ):
        gen = ReportGenerator(offline=True)
        report = gen.generate(clean_execution, simple_ingestion, profile_matches)

        unrec_findings = [
            f for f in report.findings
            if "unrecognized" in f.title.lower()
        ]
        assert len(unrec_findings) == 1
        assert unrec_findings[0].severity == "info"
        assert "obscure-lib" in unrec_findings[0].description

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
        assert "Breaking point: large" in text
        assert "10.00" in text

    def test_renders_version_flags(self):
        report = DiagnosticReport(
            version_flags=["flask: installed 2.3.0, latest is 3.1.0"],
        )
        text = report.as_text()
        assert "Version Discrepancies" in text
        assert "flask" in text

    def test_renders_unrecognized_deps(self):
        report = DiagnosticReport(
            unrecognized_deps=["obscure-lib", "custom-tool"],
        )
        text = report.as_text()
        assert "Unrecognized Dependencies" in text
        assert "obscure-lib" in text

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
        # Skipped scenarios count as failed (not clean)
        assert report.scenarios_failed == 1

    def test_operational_intent_in_context(self, clean_execution):
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            clean_execution,
            IngestionResult(project_path="/tmp/x", files_analyzed=1),
            [],
            operational_intent="Handles 500 users daily",
        )
        assert report.operational_context == "Handles 500 users daily"
