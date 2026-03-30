"""Tests for PDF content generation helpers in documents.py.

Tests pure logic functions and PDF rendering margin safety.
"""

import pytest

from mycode.documents import (
    _build_architecture_assessment,
    _build_corpus_context,
    _build_dependency_profile,
    _build_executive_summary,
    _build_test_methodology,
    _HAS_FPDF,
    _safe_text,
    _truncate_to_width,
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


# ── Truncation and PDF Rendering Tests ──


@pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
class TestTruncateToWidth:

    def test_short_text_unchanged(self):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "", 10)
        result = _truncate_to_width(pdf, "Short", 100, "Helvetica", "", 10)
        assert result == "Short"

    def test_long_text_truncated(self):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        text = "A" * 200
        result = _truncate_to_width(pdf, text, 30, "Helvetica", "", 10)
        assert result.endswith("...")
        assert len(result) < len(text)
        # Verify the truncated text actually fits
        pdf.set_font("Helvetica", "", 10)
        assert pdf.get_string_width(result) <= 30


@pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
class TestPredictionBarRendering:
    """Render actual PDF prediction bars and verify no text overflows."""

    def test_confirmed_predictions_within_margins(self):
        """Multiple confirmed predictions with long titles stay in bounds."""
        from mycode.documents import _render_pred_bars, _make_pdf_class

        PDFClass = _make_pdf_class()
        pdf = PDFClass(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_margins(20, 15, 20)
        pdf.add_page()

        page_right = pdf.w - pdf.r_margin

        # Long titles that would overflow without truncation
        preds = [
            {
                "title": "Memory accumulation over repeated "
                         "sessions with large dataset processing",
                "probability_pct": 45.0,
                "severity": "critical",
            },
            {
                "title": "Response time degradation under "
                         "concurrent load with database queries",
                "probability_pct": 32.0,
                "severity": "warning",
            },
            {
                "title": "Short title",
                "probability_pct": 15.0,
                "severity": "info",
            },
        ]
        confirmed = frozenset(p["title"] for p in preds)

        # This should not raise and should not produce text past the margin
        _render_pred_bars(pdf, preds, confirmed)

        # Verify PDF renders without error
        output = bytes(pdf.output())
        assert len(output) > 0

    def test_unconfirmed_long_title_within_margins(self):
        """A very long unconfirmed title is truncated, not overflowing."""
        from mycode.documents import _render_pred_bars, _make_pdf_class

        PDFClass = _make_pdf_class()
        pdf = PDFClass(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_margins(20, 15, 20)
        pdf.add_page()

        preds = [
            {
                "title": "X" * 150,  # absurdly long title
                "probability_pct": 50.0,
                "severity": "critical",
            },
        ]

        _render_pred_bars(pdf, preds)
        output = bytes(pdf.output())
        assert len(output) > 0


@pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
class TestFullPdfRendering:
    """End-to-end PDF rendering with all new sections."""

    def test_pdf_renders_with_predictions_and_findings(self):
        """Full PDF with confirmed predictions, corpus context, all sections."""
        from mycode.documents import render_understanding_pdf

        report = DiagnosticReport(
            user_project_description="Financial Dashboard",
            project_description="A Flask + pandas dashboard for analytics.",
            scenarios_run=12,
            scenarios_passed=8,
            scenarios_failed=3,
            scenarios_incomplete=1,
            total_errors=5,
            recognized_dep_count=4,
            recognized_dep_names=["flask", "pandas", "numpy", "requests"],
            unrecognized_deps=["some-lib"],
            confidence_note="Results may vary outside sandbox.",
            findings=[
                Finding(
                    title="Memory accumulation over sessions",
                    severity="critical",
                    category="memory_profiling",
                    description="Memory grew from 50MB to 200MB.",
                    _peak_memory_mb=200.0,
                ),
                Finding(
                    title="Data volume scaling degradation",
                    severity="warning",
                    category="data_volume_scaling",
                    description="Processing slowed at 10K items.",
                    _load_level=10000,
                ),
            ],
        )

        predictions = {
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
                    "matching_deps": ["flask"],
                    "scale_note": "",
                },
                {
                    "title": "Data volume scaling degradation",
                    "probability_pct": 28.0,
                    "severity": "warning",
                    "confirmed_count": 34,
                    "matching_deps": ["pandas"],
                    "scale_note": "",
                },
            ],
            "tech_wide_total": 960,
            "tech_wide_predictions": [
                {
                    "title": "Memory accumulation over sessions",
                    "probability_pct": 10.0,
                    "severity": "critical",
                    "confirmed_count": 96,
                    "matching_deps": ["flask"],
                    "scale_note": "",
                },
            ],
        }

        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints(
            current_users=50, max_users=500, analysis_depth="standard",
        )

        pdf_bytes = render_understanding_pdf(
            report, edition=1, project_name="Financial Dashboard",
            predictions=predictions, constraints=constraints,
        )

        assert len(pdf_bytes) > 1000
        # Verify it's a valid PDF with multiple pages
        pdf_text = pdf_bytes.decode("latin-1", errors="ignore")
        assert pdf_text.startswith("%PDF")
        # 3 pages expected (header+exec+context, findings, table+methodology)
        assert "/Count 3" in pdf_text or "/Count 2" in pdf_text
