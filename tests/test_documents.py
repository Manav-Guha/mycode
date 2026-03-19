"""Tests for the document generation and edition counter module."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mycode.documents import (
    _EDITIONS_DIR,
    _MAX_EDITIONS,
    _REPORTS_DIR,
    _edition_number,
    _hash_key,
    _normalize_github_url,
    _prune_old_editions,
    _sanitize_dirname,
    generate_finding_prompt,
    get_next_edition,
    render_understanding,
    write_edition_documents,
)
from mycode.report import DiagnosticReport, DegradationPoint, Finding


# ── Fixtures ──


@pytest.fixture
def tmp_mycode_dir(tmp_path):
    """Redirect ~/.mycode to a temp dir for isolated tests."""
    editions_dir = tmp_path / "editions"
    reports_dir = tmp_path / "reports"
    with patch("mycode.documents._EDITIONS_DIR", editions_dir), \
         patch("mycode.documents._REPORTS_DIR", reports_dir):
        yield tmp_path


@pytest.fixture
def sample_report():
    """A minimal DiagnosticReport with findings at all severity levels."""
    return DiagnosticReport(
        project_name="Test Project",
        summary="Test summary.",
        plain_summary="Your project has issues.",
        project_description="A Flask web API for managing widgets.",
        recognized_dep_count=3,
        unrecognized_deps=["some-lib", "another-lib"],
        scenarios_run=5,
        scenarios_passed=2,
        scenarios_failed=2,
        total_errors=4,
        confidence_note="Results may vary outside sandbox.",
        findings=[
            Finding(
                title="Memory leak in data processing",
                severity="critical",
                category="memory_profiling",
                description="myCode tested data processing. Memory grew unbounded.",
                details="Peak memory: 450MB at step compute_10000",
                affected_dependencies=["pandas"],
                source_file="src/processing.py",
                source_function="process_data",
                _load_level=10000,
            ),
            Finding(
                title="Slow response under concurrent load",
                severity="warning",
                category="concurrent_execution",
                description="myCode tested concurrent access. Response time degraded 5x.",
                details="Median response: 2400ms at 25 concurrent users",
                affected_dependencies=["flask"],
                source_file="src/api.py",
                source_function="handle_request",
                _load_level=25,
            ),
            Finding(
                title="Edge case handled gracefully",
                severity="info",
                category="edge_case_inputs",
                description="Empty input handled without errors.",
                affected_dependencies=["flask"],
                source_file="src/api.py",
                source_function="validate_input",
            ),
        ],
        incomplete_tests=[
            Finding(
                title="Could not test: database queries",
                severity="info",
                category="data_volume_scaling",
                description="Requires runtime context.",
                affected_dependencies=["sqlalchemy"],
                source_file="src/db.py",
                source_function="query_all",
                _failure_reason="runtime_context_required",
            ),
        ],
        degradation_points=[
            DegradationPoint(
                scenario_name="concurrent_flask_requests",
                metric="execution_time_ms",
                steps=[("1", 50.0), ("10", 120.0), ("25", 2400.0)],
                breaking_point="25",
                description="Response time increased 48x from baseline.",
            ),
        ],
    )


# ── Edition Counter Tests ──


class TestEditionCounter:
    """Tests for the edition counter logic."""

    def test_first_run_returns_1(self, tmp_mycode_dir):
        edition = get_next_edition(project_path=Path("/tmp/my-project"))
        assert edition == 1

    def test_second_run_returns_2(self, tmp_mycode_dir):
        path = Path("/tmp/my-project")
        e1 = get_next_edition(project_path=path)
        e2 = get_next_edition(project_path=path)
        assert e1 == 1
        assert e2 == 2

    def test_different_projects_independent(self, tmp_mycode_dir):
        path_a = Path("/tmp/project-a")
        path_b = Path("/tmp/project-b")
        e1a = get_next_edition(project_path=path_a)
        e1b = get_next_edition(project_path=path_b)
        e2a = get_next_edition(project_path=path_a)
        assert e1a == 1
        assert e1b == 1
        assert e2a == 2

    def test_github_url_hashing(self, tmp_mycode_dir):
        url = "https://github.com/user/repo"
        e1 = get_next_edition(github_url=url)
        e2 = get_next_edition(github_url=url)
        assert e1 == 1
        assert e2 == 2

    def test_github_url_normalization(self, tmp_mycode_dir):
        """Same repo with different URL formats get same edition."""
        e1 = get_next_edition(github_url="https://github.com/User/Repo")
        e2 = get_next_edition(github_url="https://github.com/user/repo.git")
        e3 = get_next_edition(github_url="https://github.com/user/repo/")
        # All normalized to same key → sequential
        assert e1 == 1
        assert e2 == 2
        assert e3 == 3

    def test_no_path_or_url_returns_1(self, tmp_mycode_dir):
        assert get_next_edition() == 1

    def test_edition_file_persisted(self, tmp_mycode_dir):
        path = Path("/tmp/test-persist")
        get_next_edition(project_path=path)
        # Check that a file was created in editions dir
        editions_dir = tmp_mycode_dir / "editions"
        assert editions_dir.exists()
        files = list(editions_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["edition"] == 1


class TestNormalizeGithubUrl:
    """Tests for GitHub URL normalization."""

    def test_strips_git_suffix(self):
        assert _normalize_github_url("https://github.com/user/repo.git") == \
               "github.com/user/repo"

    def test_strips_trailing_slash(self):
        assert _normalize_github_url("https://github.com/user/repo/") == \
               "github.com/user/repo"

    def test_lowercases(self):
        assert _normalize_github_url("https://GitHub.com/User/Repo") == \
               "github.com/user/repo"


class TestSanitizeDirname:
    """Tests for project name sanitization."""

    def test_basic_name(self):
        assert _sanitize_dirname("my-project") == "my-project"

    def test_scoped_package(self):
        assert _sanitize_dirname("@my-org/cool-app") == "my-org-cool-app"

    def test_spaces(self):
        assert _sanitize_dirname("My Cool Project") == "my-cool-project"

    def test_special_chars(self):
        assert _sanitize_dirname("project!@#$%") == "project"

    def test_empty_returns_project(self):
        assert _sanitize_dirname("") == "project"


# ── Document 1: Understanding Your Results ──


class TestRenderUnderstanding:
    """Tests for the human-readable diagnostic document."""

    def test_header(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "# myCode" in md
        assert "Stress test your AI-generated code before it breaks" in md
        assert "**Edition 1**" in md

    def test_footer(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "myCode by Machine Adjacent Systems" in md
        assert "does not guarantee code correctness" in md

    def test_project_summary(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "A Flask web API for managing widgets" in md

    def test_dependency_stack(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "3 dependencies with targeted stress profiles" in md
        assert "some-lib" in md

    def test_findings_grouped_by_severity(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "## CRITICAL" in md
        assert "## WARNING" in md
        assert "## INFO" in md

    def test_finding_details_present(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Memory leak in data processing" in md
        assert "Peak memory: 450MB" in md

    def test_degradation_table_present(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Performance Under Load" in md
        assert "What we tested" in md
        assert "Verdict" in md
        # Raw scenario name should NOT appear
        assert "concurrent_flask_requests" not in md

    def test_confidence_note(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Results may vary outside sandbox" in md

    def test_test_overview(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Scenarios run: 5" in md
        assert "Passed: 2" in md
        assert "Failed: 2" in md

    def test_edition_number_in_output(self, sample_report):
        md = render_understanding(sample_report, 42)
        assert "**Edition 42**" in md

    def test_empty_report(self):
        report = DiagnosticReport()
        md = render_understanding(report, 1)
        assert "# myCode" in md
        assert "myCode by Machine Adjacent Systems" in md

    def test_incomplete_tests_in_info(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Could not test: database queries" in md


# ── Prompt Generation ──


class TestGenerateFindingPrompt:
    """Tests for the coding agent prompt generator."""

    def test_critical_finding_prompt(self):
        f = Finding(
            title="Memory leak in data processing",
            severity="critical",
            category="memory_profiling",
            description="Memory grew unbounded under load.",
            source_file="src/processing.py",
            source_function="process_data",
            affected_dependencies=["pandas"],
            _load_level=10000,
            _peak_memory_mb=450.0,
        )
        prompt = generate_finding_prompt(f)
        assert "CRITICAL" in prompt
        assert "src/processing.py" in prompt
        assert "process_data" in prompt
        assert "pandas" in prompt
        assert "10000" in prompt
        assert "Investigate:" in prompt
        assert "The fix should:" in prompt
        assert "The attached JSON" in prompt

    def test_warning_finding_prompt(self):
        f = Finding(
            title="Slow response",
            severity="warning",
            category="concurrent_execution",
            description="Response degraded.",
            source_file="api.py",
        )
        prompt = generate_finding_prompt(f)
        assert "WARNING" in prompt
        assert "api.py" in prompt
        assert "thread safety" in prompt  # fix objective

    def test_info_finding_no_prompt(self):
        f = Finding(
            title="All good",
            severity="info",
            description="No issues found.",
        )
        prompt = generate_finding_prompt(f)
        assert prompt == ""

    def test_finding_without_source_file(self):
        f = Finding(
            title="Generic issue",
            severity="critical",
            category="edge_case_inputs",
            description="Bad input crashed it.",
        )
        prompt = generate_finding_prompt(f)
        assert "CRITICAL" in prompt
        assert "File:" not in prompt
        assert "The fix should:" in prompt

    def test_http_finding_prompt(self):
        f = Finding(
            title="Server crashed",
            severity="critical",
            category="http_load_testing",
            description="Crash under load.",
            source_file="server.py",
        )
        prompt = generate_finding_prompt(f)
        assert "concurrent HTTP load" in prompt


class TestFindingLayout:
    """Test the new 5-part finding layout in markdown."""

    def test_critical_finding_has_all_sections(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Memory leak",
                    severity="critical",
                    category="memory_profiling",
                    description="Memory grew.",
                    source_file="app.py",
                    source_function="run",
                    _peak_memory_mb=200.0,
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "**What we found:**" in md
        assert "**What this means for you:**" in md
        assert "**What to do:**" in md
        assert "**Prompt:**" in md
        assert "**After you fix it:**" in md
        assert "Run myCode again" in md

    def test_info_finding_no_action_sections(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="All clean",
                    severity="info",
                    description="No issues.",
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "**What we found:**" in md
        assert "**What to do:**" not in md
        assert "**Prompt:**" not in md
        assert "**After you fix it:**" not in md

    def test_json_callout_present_for_actionable(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Issue",
                    severity="critical",
                    description="Problem.",
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Download the JSON report" in md

    def test_no_json_callout_for_info_only(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Clean",
                    severity="info",
                    description="Fine.",
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Download the JSON report" not in md


# ── File Output Tests ──


class TestWriteEditionDocuments:
    """Tests for writing documents to ~/.mycode/reports/."""

    def test_writes_file(self, tmp_mycode_dir, sample_report):
        path, edition = write_edition_documents(
            report=sample_report,
            project_name="Test Project",
            project_path=Path("/tmp/test-project"),
        )
        assert path.exists()
        assert edition == 1

    def test_correct_filename(self, tmp_mycode_dir, sample_report):
        path, edition = write_edition_documents(
            report=sample_report,
            project_name="test-project",
            project_path=Path("/tmp/test-project"),
        )
        from mycode.documents import _HAS_FPDF
        if _HAS_FPDF:
            assert path.name.startswith("myCode-test-project-Results-")
            assert path.name.endswith(".pdf")
        else:
            assert path.name == "mycode-understanding-your-results-edition-1.md"

    def test_correct_directory_structure(self, tmp_mycode_dir, sample_report):
        path, edition = write_edition_documents(
            report=sample_report,
            project_name="my-app",
            project_path=Path("/tmp/my-app"),
        )
        assert "my-app" in str(path)
        assert "edition-1" in str(path)

    def test_sequential_editions(self, tmp_mycode_dir, sample_report):
        proj = Path("/tmp/seq-test")
        _, e1 = write_edition_documents(
            report=sample_report, project_name="seq", project_path=proj,
        )
        _, e2 = write_edition_documents(
            report=sample_report, project_name="seq", project_path=proj,
        )
        assert e1 == 1
        assert e2 == 2

    def test_never_writes_to_project_dir(self, tmp_mycode_dir, sample_report):
        project = Path("/tmp/user-project")
        path, _ = write_edition_documents(
            report=sample_report,
            project_name="user-project",
            project_path=project,
        )
        assert not str(path).startswith(str(project))

    def test_content_matches_renderer(self, tmp_mycode_dir, sample_report):
        path, edition = write_edition_documents(
            report=sample_report,
            project_name="content-test",
            project_path=Path("/tmp/content-test"),
        )
        from mycode.documents import _HAS_FPDF
        if _HAS_FPDF:
            assert path.read_bytes()[:5] == b"%PDF-"
        else:
            content = path.read_text(encoding="utf-8")
            assert content == render_understanding(sample_report, edition)

    def test_github_url_edition(self, tmp_mycode_dir, sample_report):
        _, e1 = write_edition_documents(
            report=sample_report,
            project_name="web-test",
            github_url="https://github.com/user/repo",
        )
        _, e2 = write_edition_documents(
            report=sample_report,
            project_name="web-test",
            github_url="https://github.com/user/repo",
        )
        assert e1 == 1
        assert e2 == 2

    def test_special_chars_in_project_name(self, tmp_mycode_dir, sample_report):
        path, _ = write_edition_documents(
            report=sample_report,
            project_name="@org/my app!",
            project_path=Path("/tmp/special"),
        )
        assert path.exists()
        assert "@" not in str(path.parent)
        assert " " not in str(path.parent)


# ── Source File/Function on Finding ──


class TestFindingSourceFields:
    """Tests for source_file and source_function on Finding."""

    def test_default_empty(self):
        f = Finding(title="test", severity="info")
        assert f.source_file == ""
        assert f.source_function == ""

    def test_set_on_construction(self):
        f = Finding(
            title="test",
            severity="critical",
            source_file="app.py",
            source_function="main",
        )
        assert f.source_file == "app.py"
        assert f.source_function == "main"

    def test_as_dict_includes_source(self):
        """source_file and source_function appear in report JSON."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="test",
                    severity="critical",
                    source_file="app.py",
                    source_function="main",
                ),
            ],
        )
        d = report.as_dict()
        f = d["findings"][0]
        assert f["source_file"] == "app.py"
        assert f["source_function"] == "main"


# ── Prompt-specific data tests ──


class TestPromptSpecificData:
    """Test that generated prompts use finding-specific data."""

    def test_memory_finding_mentions_memory(self):
        f = Finding(
            title="Memory Exhaustion",
            severity="critical",
            category="memory_profiling",
            description="Memory grew under load.",
            source_file="app.py",
            _peak_memory_mb=256.0,
        )
        prompt = generate_finding_prompt(f)
        assert "256 MB" in prompt
        assert "memory" in prompt.lower()

    def test_slow_response_finding_mentions_time(self):
        f = Finding(
            title="Slow Response",
            severity="warning",
            category="blocking_io",
            description="Response time degraded.",
            source_file="routes.py",
            _execution_time_ms=6000.0,
        )
        prompt = generate_finding_prompt(f)
        assert "6000 ms" in prompt

    def test_error_count_in_prompt(self):
        f = Finding(
            title="Error Storm",
            severity="critical",
            category="edge_case_inputs",
            description="Many errors occurred.",
            source_file="handler.py",
            _error_count=42,
        )
        prompt = generate_finding_prompt(f)
        assert "42 errors" in prompt


# ── Bug 1: Document 1 incomplete count ──


class TestDocumentIncompleteCount:
    """Test that Document 1 shows scenarios_incomplete."""

    def test_incomplete_shown_in_understanding(self):
        report = DiagnosticReport(
            scenarios_run=8,
            scenarios_passed=1,
            scenarios_failed=2,
            scenarios_incomplete=5,
        )
        md = render_understanding(report, 1)
        assert "Could not test: 5" in md

    def test_no_incomplete_line_when_zero(self):
        report = DiagnosticReport(
            scenarios_run=3,
            scenarios_passed=2,
            scenarios_failed=1,
            scenarios_incomplete=0,
        )
        md = render_understanding(report, 1)
        assert "Could not test" not in md


# ── Edition Pruning ──


class TestEditionPruning:
    """Test that old editions are pruned to keep at most 10."""

    def test_no_pruning_under_limit(self, tmp_path):
        project_dir = tmp_path / "my-project"
        for i in range(1, 6):
            (project_dir / f"edition-{i}").mkdir(parents=True)
        removed = _prune_old_editions(project_dir)
        assert removed == 0
        assert len(list(project_dir.iterdir())) == 5

    def test_pruning_at_limit(self, tmp_path):
        project_dir = tmp_path / "my-project"
        for i in range(1, 11):
            (project_dir / f"edition-{i}").mkdir(parents=True)
        removed = _prune_old_editions(project_dir)
        assert removed == 0
        assert len(list(project_dir.iterdir())) == 10

    def test_pruning_over_limit(self, tmp_path):
        project_dir = tmp_path / "my-project"
        for i in range(1, 14):
            d = project_dir / f"edition-{i}"
            d.mkdir(parents=True)
            (d / "report.md").write_text("content")
        removed = _prune_old_editions(project_dir)
        assert removed == 3
        remaining = {d.name for d in project_dir.iterdir()}
        assert remaining == {f"edition-{i}" for i in range(4, 14)}

    def test_preserves_newest(self, tmp_path):
        project_dir = tmp_path / "my-project"
        for i in range(1, 16):
            (project_dir / f"edition-{i}").mkdir(parents=True)
        _prune_old_editions(project_dir)
        remaining = {d.name for d in project_dir.iterdir()}
        # Editions 6-15 should remain
        for i in range(6, 16):
            assert f"edition-{i}" in remaining
        for i in range(1, 6):
            assert f"edition-{i}" not in remaining

    def test_nonexistent_dir(self, tmp_path):
        removed = _prune_old_editions(tmp_path / "does-not-exist")
        assert removed == 0

    def test_non_edition_dirs_ignored(self, tmp_path):
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        for i in range(1, 13):
            (project_dir / f"edition-{i}").mkdir()
        (project_dir / "other-dir").mkdir()
        (project_dir / "notes.txt").write_text("hi")
        _prune_old_editions(project_dir)
        # other-dir and notes.txt should still exist
        assert (project_dir / "other-dir").exists()
        assert (project_dir / "notes.txt").exists()
        # 10 edition dirs remain
        edition_dirs = [d for d in project_dir.iterdir() if d.name.startswith("edition-")]
        assert len(edition_dirs) == 10

    def test_edition_number_parsing(self):
        assert _edition_number("edition-1") == 1
        assert _edition_number("edition-42") == 42
        assert _edition_number("edition-") == 0
        assert _edition_number("other") == 0


# ── PDF Generation Tests ──


from mycode.documents import _HAS_FPDF, _sanitize_filename, pdf_filename


class TestPdfFilename:
    """Test PDF filename generation and sanitization."""

    def test_basic_name(self):
        result = pdf_filename("Financial Dashboard", "Results", "2026-03-19")
        assert result == "myCode-Financial-Dashboard-Results-2026-03-19.pdf"

    def test_special_chars(self):
        result = pdf_filename("my@project/v2", "Fixes", "2026-03-19")
        assert result == "myCode-my-project-v2-Fixes-2026-03-19.pdf"

    def test_empty_name(self):
        result = pdf_filename("", "Results", "2026-03-19")
        assert result == "myCode-Project-Results-2026-03-19.pdf"

    def test_spaces_to_hyphens(self):
        result = pdf_filename("My Cool App", "Fixes", "2026-01-01")
        assert result == "myCode-My-Cool-App-Fixes-2026-01-01.pdf"

    def test_sanitize_filename_preserves_case(self):
        assert _sanitize_filename("MyApp") == "MyApp"

    def test_sanitize_filename_strips_specials(self):
        assert _sanitize_filename("a@b/c\\d") == "a-b-c-d"


@pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
class TestPdfRendering:
    """Test PDF document rendering (requires fpdf2)."""

    def test_understanding_pdf_returns_bytes(self):
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            project_description="Test dashboard",
            scenarios_run=5,
            scenarios_passed=3,
            scenarios_failed=2,
        )
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"

    def test_understanding_pdf_has_content(self):
        """PDF should be non-trivial size for a report with findings."""
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            project_description="Financial dashboard built with streamlit",
            scenarios_run=10,
            scenarios_passed=7,
            scenarios_failed=3,
            findings=[
                Finding(
                    title="Memory grows unbounded",
                    severity="critical",
                    description="Memory usage reaches 500MB under load.",
                    affected_dependencies=["pandas"],
                ),
                Finding(
                    title="Response time degrades",
                    severity="warning",
                    description="Response time exceeds 2 seconds at 50 users.",
                ),
            ],
        )
        pdf_bytes = render_understanding_pdf(report, edition=2)
        # Should be at least 1KB for a report with content
        assert len(pdf_bytes) > 1000

    def test_write_edition_creates_pdf(self, tmp_mycode_dir):
        """write_edition_documents creates .pdf file when fpdf2 is available."""
        report = DiagnosticReport(
            scenarios_run=3,
            scenarios_passed=3,
        )
        path, edition = write_edition_documents(
            report=report,
            project_name="Test App",
            project_path=Path("/tmp/test-app"),
        )
        assert path.suffix == ".pdf"
        assert path.read_bytes()[:5] == b"%PDF-"

    def test_pdf_with_unicode_in_findings(self):
        """PDF handles Unicode characters in finding text gracefully."""
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Memory crash \u2014 out of memory",
                    severity="critical",
                    description="App uses \u201csmart quotes\u201d and \u2026 ellipsis.",
                ),
            ],
        )
        # Should not raise FPDFUnicodeEncodingException
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"


# ── Issue 1: Degradation humanisation in documents ──


class TestDegradationHumanisation:
    """Test that degradation curves use humanised labels in the table."""

    def test_markdown_table_humanised_labels(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="coupling_compute_streamlit_markdown",
                    metric="execution_time_ms",
                    steps=[("compute_50000", 100.0), ("compute_100000", 500.0)],
                    breaking_point="compute_100000",
                ),
            ],
        )
        md = render_understanding(report, 1)
        # Raw names should NOT appear
        assert "coupling_compute_streamlit_markdown" not in md
        assert "compute_50000" not in md
        assert "compute_100000" not in md
        # Table should have humanised step labels
        assert "items of data" in md
        # Table format
        assert "| What we tested |" in md

    def test_markdown_verdict_column(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_concurrent_request_load",
                    metric="execution_time_ms",
                    steps=[("concurrent_10", 50.0), ("concurrent_50", 2000.0)],
                    breaking_point="concurrent_50",
                ),
            ],
        )
        md = render_understanding(report, 1)
        # 2000ms = unresponsive
        assert "Unresponsive at peak load" in md
        # Step label humanised in cell context
        assert "50 simultaneous users" in md
        assert "concurrent_50" not in md

    def test_markdown_memory_label_prefix(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_concurrent_request_load",
                    metric="memory_peak_mb",
                    steps=[("concurrent_5", 10.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Memory usage" in md

    @pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
    def test_pdf_degradation_table(self):
        """PDF degradation section should render as table."""
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="coupling_compute_streamlit_markdown",
                    metric="execution_time_ms",
                    steps=[("compute_50000", 100.0), ("compute_100000", 500.0)],
                    breaking_point="compute_100000",
                ),
            ],
        )
        # Should not raise — and should produce valid PDF
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"


# ── Issue 2: http_tested collapse ──


class TestHttpTestedCollapse:
    """http_tested findings should collapse into a single summary line."""

    def _make_http_tested_report(self, n=6, framework="streamlit"):
        return DiagnosticReport(
            scenarios_run=10,
            scenarios_passed=4,
            scenarios_incomplete=n,
            incomplete_tests=[
                Finding(
                    title=f"Could not test: scenario_{i}",
                    severity="info",
                    _failure_reason="http_tested",
                    affected_dependencies=[framework],
                )
                for i in range(n)
            ],
        )

    def test_understanding_md_no_individual_http_tested(self):
        report = self._make_http_tested_report()
        md = render_understanding(report, 1)
        # Individual findings should not appear
        assert "Could not test: scenario_0" not in md
        assert "Could not test: scenario_5" not in md
        # Summary line should appear
        assert "6 streamlit scenarios were tested via HTTP load testing." in md

    def test_mixed_incomplete_preserves_others(self):
        """Non-http_tested incomplete findings still appear individually."""
        report = DiagnosticReport(
            incomplete_tests=[
                Finding(
                    title="Could not test: db query",
                    severity="info",
                    _failure_reason="runtime_context_required",
                    affected_dependencies=["sqlalchemy"],
                    source_file="db.py",
                ),
                Finding(
                    title="Could not test: api call",
                    severity="info",
                    _failure_reason="http_tested",
                    affected_dependencies=["flask"],
                ),
            ],
        )
        md = render_understanding(report, 1)
        # runtime_context_required finding still appears
        assert "Could not test: db query" in md
        # http_tested collapsed
        assert "Could not test: api call" not in md
        assert "1 flask scenario was tested via HTTP load testing." in md

    @pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
    def test_pdf_understanding_no_individual_http_tested(self):
        from mycode.documents import render_understanding_pdf
        report = self._make_http_tested_report()
        # Should produce valid PDF without individual http_tested findings
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"



# ── Issue 1: Dependency stack names recognized deps ──


class TestDepStackNames:
    """Recognized dependencies should be named in the dependency stack."""

    def test_markdown_names_recognized_deps(self):
        report = DiagnosticReport(
            recognized_dep_count=2,
            recognized_dep_names=["pandas", "streamlit"],
            unrecognized_deps=["plotly"],
        )
        md = render_understanding(report, 1)
        assert "pandas, streamlit" in md
        assert "targeted stress profiles: pandas, streamlit" in md

    @pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
    def test_pdf_names_recognized_deps(self):
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            recognized_dep_count=2,
            recognized_dep_names=["pandas", "streamlit"],
            unrecognized_deps=["plotly"],
        )
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"


# ── Issue 7: Breaking point includes metric context ──


class TestBreakingPointInTable:
    """Breaking point info now appears in verdict column."""

    def test_time_verdict_and_data(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_scaling",
                    metric="execution_time_ms",
                    steps=[("data_size_1000", 10.0), ("data_size_50000", 350.0)],
                    breaking_point="data_size_50000",
                ),
            ],
        )
        md = render_understanding(report, 1)
        # 350ms → noticeable, with breaking point context
        assert "Noticeable above 50,000 items" in md
        # Cell data present
        assert "350ms" in md
        assert "10ms" in md

    def test_memory_verdict_and_data(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="pandas_memory",
                    metric="memory_peak_mb",
                    steps=[("data_size_100", 5.0), ("data_size_10000", 120.0)],
                    breaking_point="data_size_10000",
                ),
            ],
        )
        md = render_understanding(report, 1)
        # Memory >50MB verdict
        assert "Heavy" in md
        # Cell data
        assert "120MB" in md
        assert "10,000 items" in md


# ── Performance Summary Table ──


class TestPerfSummaryTable:
    """Test the performance summary table generation."""

    def test_single_curve_produces_table(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test_scaling",
                    metric="execution_time_ms",
                    steps=[("data_size_100", 5.0), ("data_size_1000", 50.0), ("data_size_10000", 500.0)],
                    breaking_point="data_size_10000",
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "| What we tested |" in md
        assert "| Verdict |" in md
        # 3 data cells
        assert "5ms" in md
        assert "50ms" in md
        assert "500ms" in md

    def test_multiple_curves_multi_row(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_scaling",
                    metric="execution_time_ms",
                    steps=[("concurrent_1", 10.0), ("concurrent_50", 200.0)],
                ),
                DegradationPoint(
                    scenario_name="flask_scaling",
                    metric="memory_peak_mb",
                    steps=[("concurrent_1", 20.0), ("concurrent_50", 80.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        # Should have table header once
        assert md.count("| What we tested |") == 1
        # Two data rows (time + memory)
        assert "200ms" in md
        assert "80MB" in md
        assert "Memory usage" in md

    def test_verdict_no_issues(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="http_get_root",
                    metric="response_time_ms",
                    steps=[("1 concurrent", 3.0), ("5 concurrent", 6.0), ("15 concurrent", 7.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "No issues" in md

    def test_verdict_unresponsive(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_load",
                    metric="execution_time_ms",
                    steps=[("concurrent_1", 100.0), ("concurrent_50", 5000.0)],
                    breaking_point="concurrent_50",
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Unresponsive at peak load" in md

    def test_verdict_memory_fine(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test",
                    metric="memory_peak_mb",
                    steps=[("data_size_100", 5.0), ("data_size_10000", 30.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Fine at your scale" in md

    def test_verdict_memory_heavy(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test",
                    metric="memory_peak_mb",
                    steps=[("data_size_100", 10.0), ("data_size_100000", 250.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "Very heavy" in md

    def test_verdict_errors(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test",
                    metric="error_count",
                    steps=[("concurrent_1", 0.0), ("concurrent_50", 12.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        assert "12 errors at peak" in md

    def test_two_step_curve_mid_dash(self):
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test",
                    metric="execution_time_ms",
                    steps=[("data_size_100", 5.0), ("data_size_10000", 50.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        # Mid column should be dash for 2-step curves
        lines = [l for l in md.split("\n") if l.startswith("| ") and "5ms" in l]
        assert len(lines) == 1
        # Should contain a dash cell
        assert "| — |" in lines[0]

    def test_empty_degradation_no_table(self):
        report = DiagnosticReport(degradation_points=[])
        md = render_understanding(report, 1)
        assert "Performance Under Load" not in md

    @pytest.mark.skipif(not _HAS_FPDF, reason="fpdf2 not installed")
    def test_pdf_table_renders(self):
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="test_scaling",
                    metric="execution_time_ms",
                    steps=[("data_size_100", 5.0), ("data_size_1000", 50.0), ("data_size_10000", 500.0)],
                ),
                DegradationPoint(
                    scenario_name="test_scaling",
                    metric="memory_peak_mb",
                    steps=[("data_size_100", 5.0), ("data_size_1000", 20.0), ("data_size_10000", 80.0)],
                ),
            ],
        )
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"
