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
    _describe_impact,
    _describe_scenario,
    _describe_step,
    _extract_project_ref,
    _format_ms,
    _human_metric,
    _human_time,
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
                    title="Resource limit hit: coupling_api_fetch_data",
                    severity="critical",
                    category="concurrent_execution",
                    description="Resource cap exceeded.",
                    grouped_findings=[
                        Finding(
                            title="Resource limit hit: coupling_api_send",
                            severity="critical",
                        ),
                        Finding(
                            title="Resource limit hit: coupling_api_post",
                            severity="critical",
                        ),
                    ],
                    group_count=3,
                ),
            ],
        )
        text = report.as_text()
        assert "(and 2 similar)" in text
        assert "Also: coupling_api_send, coupling_api_post" in text

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
        assert "Also: coupling_compute_setError, coupling_compute_setRawScores" in text

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

    def test_includes_operational_intent(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Project ref derived from intent appears in the summary."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="Personal budget tracker for daily use",
        )

        assert "budget tracker" in report.plain_summary.lower()

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
        # Should use "When" pattern for activity framing
        assert "when " in lower

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

    def test_closing_line_present(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Summary ends with the bridge line."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(failing_execution, simple_ingestion, profile_matches)

        assert "paste these into your coding tool" in report.plain_summary.lower()

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

    def test_no_intent_still_works(
        self, failing_execution, simple_ingestion, profile_matches,
    ):
        """Works without operational_intent — uses 'your project'."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            failing_execution, simple_ingestion, profile_matches,
            operational_intent="",
        )

        assert report.plain_summary
        assert "your project" in report.plain_summary.lower()

    def test_empty_execution(self):
        """No scenarios → no plain summary generated."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(
            ExecutionEngineResult(),
            IngestionResult(project_path="/tmp/x"),
            [],
        )

        assert report.plain_summary == ""

    def test_impact_uses_real_time_not_multiplier(
        self, degrading_execution, simple_ingestion, profile_matches,
    ):
        """Impact described in real terms, not multipliers."""
        gen = ReportGenerator(offline=True)
        report = gen.generate(degrading_execution, simple_ingestion, profile_matches)

        lower = report.plain_summary.lower()
        # Should have real-terms descriptions from the _human_time scale
        assert any(
            phrase in lower
            for phrase in ("instant", "fast", "slow", "delay", "second",
                           "minute", "ms", "MB")
        )
        # Should NOT have multiplier patterns like "80x" or "204x slower"
        import re
        assert not re.search(r"\d+x slower", lower)

    def test_step_translated_to_user_terms(self):
        """Step names translated to user-meaningful descriptions."""
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
        assert "simultaneous users" in lower

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

    def test_breaking_point_describes_impact(self):
        """Breaking point includes what happens there, not just the label."""
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
        # Should describe what happens at the breaking point
        assert "simultaneous users" in lower
        assert "slowing down" in lower or "slow" in lower

    def test_memory_projects_multi_user(self):
        """Memory findings project multi-user impact."""
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
        assert "10 users" in lower
        assert "720mb" in lower

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

    def test_describe_scenario_unknown_returns_empty(self):
        assert _describe_scenario("completely_unknown_thing") == ""

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

    def test_describe_impact_time_same_band_uses_concrete(self):
        """When both values are in the same band, use concrete ms values."""
        result = _describe_impact("execution_time_ms", 10.0, 80.0)
        assert "10ms" in result
        assert "80ms" in result

    def test_describe_impact_memory_projects_multi_user(self):
        """Memory ≥50MB projects 10-user impact."""
        result = _describe_impact("memory_peak_mb", 5.0, 120.0)
        assert "5MB" in result
        assert "120MB" in result
        assert "10 users" in result
        assert "1200MB" in result

    def test_describe_impact_memory_small_no_projection(self):
        """Memory <50MB doesn't project multi-user."""
        result = _describe_impact("memory_peak_mb", 5.0, 30.0)
        assert "30MB" in result
        assert "10 users" not in result

    def test_format_ms(self):
        assert _format_ms(0.16) == "0.16ms"
        assert _format_ms(78.0) == "78ms"
        assert _format_ms(770.0) == "770ms"
        assert _format_ms(5000.0) == "5.0s"
        assert _format_ms(120000.0) == "2.0min"
