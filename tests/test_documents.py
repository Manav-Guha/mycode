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
    get_next_edition,
    render_fixes,
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

    def test_degradation_points(self, sample_report):
        md = render_understanding(sample_report, 1)
        assert "Performance Degradation" in md
        assert "concurrent_flask_requests" in md

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


# ── Document 2: Recommended Fixes ──


class TestRenderFixes:
    """Tests for the agent-parseable investigation document."""

    def test_attribution_comment(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert md.startswith("<!-- Generated by myCode")
        assert "Machine Adjacent Systems" in md
        assert "Edition 1" in md

    def test_no_branding_in_body(self, sample_report):
        md = render_fixes(sample_report, 1)
        # After the first HTML comment, no branding
        body = md.split("-->", 1)[1]
        assert "Machine Adjacent Systems" not in body

    def test_grouped_by_file(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "## src/processing.py" in md
        assert "## src/api.py" in md
        assert "## src/db.py" in md

    def test_files_ordered_by_severity(self, sample_report):
        md = render_fixes(sample_report, 1)
        # src/processing.py has critical → should come first
        proc_pos = md.index("## src/processing.py")
        api_pos = md.index("## src/api.py")
        db_pos = md.index("## src/db.py")
        assert proc_pos < api_pos
        assert api_pos < db_pos

    def test_severity_tags(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "[CRITICAL]" in md
        assert "[WARNING]" in md
        assert "[INFO]" in md

    def test_investigation_prompt_for_critical(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "**Investigate:**" in md
        assert "Examine" in md

    def test_info_findings_are_context(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "informational" in md
        assert "No investigation required" in md

    def test_source_function_shown(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "`process_data`" in md
        assert "`handle_request`" in md

    def test_load_level_shown(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "10000" in md  # load level for critical finding

    def test_category_shown(self, sample_report):
        md = render_fixes(sample_report, 1)
        assert "memory_profiling" in md

    def test_empty_findings(self):
        report = DiagnosticReport()
        md = render_fixes(report, 1)
        assert "No findings to investigate" in md

    def test_findings_without_source_file(self):
        """Findings without source_file go to catch-all group."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Generic issue",
                    severity="warning",
                    category="edge_case_inputs",
                    description="Something broke.",
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "(no file identified)" in md

    def test_multiple_findings_same_file(self):
        """Multiple findings in same file grouped together."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Issue A",
                    severity="critical",
                    source_file="app.py",
                    source_function="func_a",
                    category="memory_profiling",
                    description="Memory issue.",
                ),
                Finding(
                    title="Issue B",
                    severity="warning",
                    source_file="app.py",
                    source_function="func_b",
                    category="concurrent_execution",
                    description="Concurrency issue.",
                ),
            ],
        )
        md = render_fixes(report, 1)
        # Only one file header
        assert md.count("## app.py") == 1
        # Both findings present
        assert "Issue A" in md
        assert "Issue B" in md

    def test_investigation_prompt_categories(self):
        """Category-specific investigation prompts are generated."""
        for category in [
            "data_volume_scaling", "memory_profiling", "edge_case_inputs",
            "concurrent_execution", "blocking_io", "async_failures",
        ]:
            report = DiagnosticReport(
                findings=[
                    Finding(
                        title="Test finding",
                        severity="critical",
                        category=category,
                        description="Test.",
                        source_file="test.py",
                        source_function="test_func",
                    ),
                ],
            )
            md = render_fixes(report, 1)
            assert "**Investigate:**" in md


# ── File Output Tests ──


class TestWriteEditionDocuments:
    """Tests for writing documents to ~/.mycode/reports/."""

    def test_writes_both_files(self, tmp_mycode_dir, sample_report):
        path1, path2, edition = write_edition_documents(
            report=sample_report,
            project_name="Test Project",
            project_path=Path("/tmp/test-project"),
        )
        assert path1.exists()
        assert path2.exists()
        assert edition == 1

    def test_correct_filenames(self, tmp_mycode_dir, sample_report):
        path1, path2, edition = write_edition_documents(
            report=sample_report,
            project_name="test-project",
            project_path=Path("/tmp/test-project"),
        )
        assert path1.name == "mycode-understanding-your-results-edition-1.md"
        assert path2.name == "mycode-recommended-fixes-edition-1.md"

    def test_correct_directory_structure(self, tmp_mycode_dir, sample_report):
        path1, path2, edition = write_edition_documents(
            report=sample_report,
            project_name="my-app",
            project_path=Path("/tmp/my-app"),
        )
        reports_dir = tmp_mycode_dir / "reports"
        assert "my-app" in str(path1)
        assert "edition-1" in str(path1)

    def test_sequential_editions(self, tmp_mycode_dir, sample_report):
        proj = Path("/tmp/seq-test")
        _, _, e1 = write_edition_documents(
            report=sample_report, project_name="seq", project_path=proj,
        )
        _, _, e2 = write_edition_documents(
            report=sample_report, project_name="seq", project_path=proj,
        )
        assert e1 == 1
        assert e2 == 2

    def test_never_writes_to_project_dir(self, tmp_mycode_dir, sample_report):
        project = Path("/tmp/user-project")
        path1, path2, _ = write_edition_documents(
            report=sample_report,
            project_name="user-project",
            project_path=project,
        )
        # Neither file is inside the project directory
        assert not str(path1).startswith(str(project))
        assert not str(path2).startswith(str(project))

    def test_content_matches_renderers(self, tmp_mycode_dir, sample_report):
        path1, path2, edition = write_edition_documents(
            report=sample_report,
            project_name="content-test",
            project_path=Path("/tmp/content-test"),
        )
        understanding_content = path1.read_text(encoding="utf-8")
        fixes_content = path2.read_text(encoding="utf-8")
        assert understanding_content == render_understanding(sample_report, edition)
        assert fixes_content == render_fixes(sample_report, edition)

    def test_github_url_edition(self, tmp_mycode_dir, sample_report):
        _, _, e1 = write_edition_documents(
            report=sample_report,
            project_name="web-test",
            github_url="https://github.com/user/repo",
        )
        _, _, e2 = write_edition_documents(
            report=sample_report,
            project_name="web-test",
            github_url="https://github.com/user/repo",
        )
        assert e1 == 1
        assert e2 == 2

    def test_special_chars_in_project_name(self, tmp_mycode_dir, sample_report):
        path1, _, _ = write_edition_documents(
            report=sample_report,
            project_name="@org/my app!",
            project_path=Path("/tmp/special"),
        )
        assert path1.exists()
        # Directory name should be sanitized
        assert "@" not in str(path1.parent)
        assert " " not in str(path1.parent)


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


# ── Bug 2: Finding-specific Investigation Prompts ──


class TestFindingSpecificPrompts:
    """Test that investigation prompts use finding-specific data."""

    def test_memory_finding_mentions_memory(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Memory Exhaustion",
                    severity="critical",
                    category="memory_profiling",
                    description="Memory grew under load.",
                    source_file="app.py",
                    _peak_memory_mb=256.0,
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "256 MB" in md
        assert "memory" in md.lower()

    def test_slow_response_finding_mentions_time(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Slow Response",
                    severity="warning",
                    category="blocking_io",
                    description="Response time degraded.",
                    source_file="routes.py",
                    _execution_time_ms=6000.0,
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "6000 ms" in md
        assert "latency" in md.lower() or "synchronous" in md.lower()

    def test_http_memory_finding_specific(self):
        """HTTP finding about memory gets memory-specific guidance."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="High baseline memory",
                    severity="warning",
                    category="http_load_testing",
                    description="Memory baseline is high at startup.",
                    source_file="server.py",
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "startup" in md.lower() or "bundle" in md.lower()

    def test_http_response_time_finding_specific(self):
        """HTTP finding about response time gets latency-specific guidance."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Response time degradation",
                    severity="warning",
                    category="http_load_testing",
                    description="Response time increases under concurrent load.",
                    source_file="server.py",
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "concurrent" in md.lower() or "middleware" in md.lower()

    def test_different_http_findings_get_different_prompts(self):
        """Two HTTP findings with different data produce different prompts."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="High memory baseline",
                    severity="warning",
                    category="http_load_testing",
                    description="Memory baseline is high.",
                    source_file="server.py",
                ),
                Finding(
                    title="Response time degradation",
                    severity="warning",
                    category="http_load_testing",
                    description="Response time increases under load.",
                    source_file="server.py",
                ),
            ],
        )
        md = render_fixes(report, 1)
        # Extract investigate prompts
        prompts = [
            line for line in md.split("\n")
            if line.startswith("**Investigate:**")
        ]
        assert len(prompts) == 2
        # They should be different
        assert prompts[0] != prompts[1]

    def test_error_count_in_prompt(self):
        """Finding with many errors mentions error count."""
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Error Storm",
                    severity="critical",
                    category="edge_case_inputs",
                    description="Many errors occurred.",
                    source_file="handler.py",
                    _error_count=42,
                ),
            ],
        )
        md = render_fixes(report, 1)
        assert "42 errors" in md


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
