"""Tests for PDF content generation helpers in documents.py.

Tests pure logic functions — no PDF rendering required.
"""

import pytest

from mycode.documents import (
    _build_architecture_assessment,
    _build_corpus_context,
    _build_dependency_profile,
    _build_executive_summary,
    _build_test_methodology,
)
from mycode.report import DiagnosticReport, Finding


# ── Fixtures ──


@pytest.fixture
def report_with_criticals():
    """Report with 2 critical findings."""
    return DiagnosticReport(
        user_project_description="Financial Dashboard",
        scenarios_run=12,
        recognized_dep_count=5,
        unrecognized_deps=["some-lib"],
        findings=[
            Finding(
                title="Memory accumulation over sessions",
                severity="critical",
                category="memory_profiling",
            ),
            Finding(
                title="Response time degradation under load",
                severity="critical",
                category="http_load_testing",
            ),
            Finding(
                title="Edge case input handling",
                severity="warning",
                category="edge_case_inputs",
            ),
        ],
    )


@pytest.fixture
def report_no_findings():
    """Report with no findings."""
    return DiagnosticReport(
        user_project_description="Simple API",
        scenarios_run=8,
        recognized_dep_count=3,
        findings=[],
    )


@pytest.fixture
def report_warnings_only():
    """Report with only warnings, no critical."""
    return DiagnosticReport(
        user_project_description="Chat Application",
        scenarios_run=10,
        recognized_dep_count=4,
        findings=[
            Finding(
                title="Slow query under concurrent load",
                severity="warning",
                category="concurrent_execution",
            ),
        ],
    )


@pytest.fixture
def predictions_arch_filtered():
    """Predictions dict with architecture-specific layer."""
    return {
        "total_similar_projects": 120,
        "matching_deps": ["flask", "pandas"],
        "architectural_type": "dashboard",
        "arch_filtered": True,
        "predictions": [
            {
                "title": "Memory accumulation over sessions",
                "probability_pct": 35.0,
                "severity": "critical",
                "confirmed_count": 42,
            },
            {
                "title": "Data volume scaling degradation",
                "probability_pct": 28.0,
                "severity": "warning",
                "confirmed_count": 34,
            },
        ],
        "tech_wide_total": 960,
        "tech_wide_predictions": [
            {
                "title": "Memory accumulation over sessions",
                "probability_pct": 15.0,
                "severity": "critical",
                "confirmed_count": 144,
            },
        ],
    }


@pytest.fixture
def predictions_no_arch():
    """Predictions dict without architecture filtering."""
    return {
        "total_similar_projects": 500,
        "matching_deps": ["express"],
        "architectural_type": "general",
        "arch_filtered": False,
        "predictions": [
            {
                "title": "Response time degradation under load",
                "probability_pct": 20.0,
                "severity": "warning",
                "confirmed_count": 100,
            },
        ],
        "tech_wide_total": 0,
        "tech_wide_predictions": [],
    }


@pytest.fixture
def sample_corpus_patterns():
    """Minimal corpus patterns for testing dependency profiles."""
    return [
        {
            "title": "Memory accumulation over sessions",
            "category": "memory_profiling",
            "confirmed_count": 150,
            "affected_dependencies": ["flask", "streamlit"],
        },
        {
            "title": "Data volume scaling degradation",
            "category": "data_volume_scaling",
            "confirmed_count": 80,
            "affected_dependencies": ["pandas", "flask"],
        },
        {
            "title": "Response time degradation under load",
            "category": "http_load_testing",
            "confirmed_count": 120,
            "affected_dependencies": ["flask", "express"],
        },
        {
            "title": "Application handled HTTP load without issues",
            "category": "http_load_testing",
            "confirmed_count": 321,
            "affected_dependencies": ["flask"],
        },
        {
            "title": "Tiny pattern",
            "category": "edge_case_input",
            "confirmed_count": 1,
            "affected_dependencies": ["flask"],
        },
    ]


# ── Executive Summary Tests ──


class TestExecutiveSummary:

    def test_with_critical_findings(self, report_with_criticals):
        result = _build_executive_summary(report_with_criticals)
        assert "12 scenarios" in result
        assert "2 priority improvements" in result
        assert "Memory accumulation" in result

    def test_no_findings(self, report_no_findings):
        result = _build_executive_summary(report_no_findings)
        assert "No priority improvements" in result
        assert "8 scenarios" in result

    def test_with_constraints(self, report_with_criticals):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints(max_users=500, current_users=50)
        result = _build_executive_summary(
            report_with_criticals, constraints=constraints,
        )
        assert "500" in result
        assert "priority improvement" in result

    def test_no_critical_with_constraints(self, report_warnings_only):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints(max_users=200)
        result = _build_executive_summary(
            report_warnings_only, constraints=constraints,
        )
        assert "200" in result
        assert "No priority issues" in result
        assert "1 improvement opportunity" in result


# ── Corpus Context Tests ──


class TestCorpusContext:

    def test_matched_finding(self, predictions_arch_filtered):
        finding = Finding(
            title="Memory accumulation over sessions",
            severity="critical",
            category="memory_profiling",
        )
        result = _build_corpus_context(finding, predictions_arch_filtered)
        assert "120" in result  # project_count
        assert "35%" in result  # probability_pct
        assert "42" in result   # confirmed_count

    def test_no_match(self, predictions_arch_filtered):
        finding = Finding(
            title="Completely unrelated issue",
            severity="warning",
            category="edge_case_inputs",
        )
        result = _build_corpus_context(finding, predictions_arch_filtered)
        assert result == ""

    def test_no_predictions(self):
        finding = Finding(
            title="Some finding",
            severity="critical",
            category="memory_profiling",
        )
        result = _build_corpus_context(finding, None)
        assert result == ""

    def test_flat_predictions_format(self):
        """Test with flat predictions_pdf_dict format from routes.py."""
        predictions = {
            "total_similar_projects": 500,
            "architectural_type": "api_service",
            "arch_filtered": False,
            "predictions": [
                {
                    "title": "Response time degradation under load",
                    "probability_pct": 20.0,
                    "severity": "warning",
                    "confirmed_count": 100,
                },
            ],
            "tech_wide_total": 0,
            "tech_wide_predictions": [],
        }
        finding = Finding(
            title="Response time degradation under load",
            severity="warning",
            category="http_load_testing",
        )
        result = _build_corpus_context(finding, predictions)
        assert "500" in result
        assert "20%" in result


# ── Dependency Profile Tests ──


class TestDependencyProfile:

    def test_with_data(self, sample_corpus_patterns):
        report = DiagnosticReport(
            recognized_dep_names=["flask", "pandas"],
        )
        result = _build_dependency_profile(report, sample_corpus_patterns)
        assert len(result) >= 1
        # Flask should have patterns
        flask_entry = next(
            (e for e in result if e["dep"] == "flask"), None,
        )
        assert flask_entry is not None
        assert len(flask_entry["patterns"]) >= 1
        # Verify skip titles are excluded
        for entry in result:
            for pat in entry["patterns"]:
                assert pat["title"] != "Application handled HTTP load without issues"
        # Verify minimum confirmed_count filter
        for entry in result:
            for pat in entry["patterns"]:
                assert pat["count"] >= 3

    def test_empty_corpus(self):
        report = DiagnosticReport(recognized_dep_names=["flask"])
        result = _build_dependency_profile(report, [])
        assert result == []

    def test_no_deps(self, sample_corpus_patterns):
        report = DiagnosticReport(recognized_dep_names=[])
        result = _build_dependency_profile(report, sample_corpus_patterns)
        assert result == []

    def test_cap_at_six_deps(self, sample_corpus_patterns):
        """Even with many deps, result is capped at 6."""
        report = DiagnosticReport(
            recognized_dep_names=[
                "flask", "pandas", "numpy", "streamlit",
                "fastapi", "express", "react", "sqlalchemy",
            ],
        )
        result = _build_dependency_profile(report, sample_corpus_patterns)
        assert len(result) <= 6

    def test_patterns_capped_at_three(self, sample_corpus_patterns):
        """Each dep gets at most 3 patterns."""
        report = DiagnosticReport(
            recognized_dep_names=["flask"],
        )
        result = _build_dependency_profile(report, sample_corpus_patterns)
        for entry in result:
            assert len(entry["patterns"]) <= 3


# ── Architecture Assessment Tests ──


class TestArchitectureAssessment:

    def test_with_arch_data(self, report_with_criticals, predictions_arch_filtered):
        result = _build_architecture_assessment(
            report_with_criticals, predictions_arch_filtered,
        )
        assert "dashboard" in result
        assert "120" in result   # project_count
        assert "5" in result     # recognized_dep_count

    def test_general_arch(self, report_with_criticals, predictions_no_arch):
        result = _build_architecture_assessment(
            report_with_criticals, predictions_no_arch,
        )
        assert result == ""

    def test_no_predictions(self, report_with_criticals):
        result = _build_architecture_assessment(report_with_criticals, None)
        assert result == ""


# ── Test Methodology Tests ──


class TestTestMethodology:

    def test_with_constraints(self, report_with_criticals):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints(
            current_users=50,
            max_users=500,
            analysis_depth="deep",
        )
        result = _build_test_methodology(report_with_criticals, constraints)
        assert result["scenarios_run"] == 12
        assert result["analysis_depth"] == "deep"
        assert result["baseline"] == 50
        assert result["ceiling"] == 500
        assert result["buffer"] == 575  # 500 * 1.15
        assert result["fully_tested"] == 5
        assert result["usage_tested"] == 1

    def test_no_constraints(self, report_with_criticals):
        result = _build_test_methodology(report_with_criticals, None)
        assert result["scenarios_run"] == 12
        assert result["analysis_depth"] == "standard"
        assert result["baseline"] is None
        assert result["ceiling"] is None
        assert result["buffer"] is None
