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
    _finding_severity_for_dp,
    _hash_key,
    _integrate_details,
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
        assert "## PRIORITY IMPROVEMENTS" in md
        assert "## IMPROVEMENT OPPORTUNITIES" in md
        assert "## GOOD TO KNOW" in md

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
        assert "Fix:" in prompt
        assert "JSON" in prompt

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
        assert "thread safety" in prompt

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
        assert "Location:" not in prompt
        assert "Fix:" in prompt

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

    def test_prompt_includes_llm_suggestion(self):
        f = Finding(
            title="Memory leak",
            severity="critical",
            category="memory_profiling",
            source_file="app.py",
            source_function="handler",
            llm_fix_suggestion="Add a cache eviction on line 12 to bound memory growth.",
        )
        prompt = generate_finding_prompt(f)
        assert "Suggested fix:" in prompt
        assert "cache eviction" in prompt

    def test_prompt_excludes_empty_llm_suggestion(self):
        f = Finding(
            title="Memory leak",
            severity="critical",
            category="memory_profiling",
            source_file="app.py",
        )
        prompt = generate_finding_prompt(f)
        assert "Suggested fix:" not in prompt

    def test_understanding_caveat_when_llm_suggestions_present(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Issue",
                    severity="critical",
                    category="memory_profiling",
                    llm_fix_suggestion="Fix line 5.",
                ),
            ],
        )
        md = render_understanding(report, edition=1)
        assert "generated by an AI model (Gemini)" in md

    def test_understanding_no_caveat_without_suggestions(self):
        report = DiagnosticReport(
            findings=[
                Finding(
                    title="Issue",
                    severity="critical",
                    category="memory_profiling",
                ),
            ],
        )
        md = render_understanding(report, edition=1)
        assert "generated by an AI model (Gemini)" not in md


class TestRemediationFieldsEnrichment:
    """Tests for version/call-chain/decorator-aware remediation fields."""

    def test_deps_versioned_with_outdated(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            affected_dependencies=["flask", "pandas"],
            dep_versions={"flask": "2.0.1", "pandas": "2.2.0"},
            dep_latest_versions={"flask": "3.1.0", "pandas": "2.2.0"},
            dep_outdated=["flask"],
        )
        fields = _remediation_fields(f)
        assert "flask 2.0.1 (outdated — latest 3.1.0)" in fields["deps_versioned"]
        assert "pandas 2.2.0" in fields["deps_versioned"]
        assert "outdated" not in fields["deps_versioned"].split("pandas")[1]
        assert fields["has_outdated"] is True
        assert "1 of 2" in fields["deps_outdated_note"]

    def test_deps_versioned_none_outdated(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            affected_dependencies=["flask"],
            dep_versions={"flask": "3.1.0"},
            dep_latest_versions={"flask": "3.1.0"},
            dep_outdated=[],
        )
        fields = _remediation_fields(f)
        assert "flask 3.1.0" in fields["deps_versioned"]
        assert "outdated" not in fields["deps_versioned"]
        assert fields["has_outdated"] is False
        assert fields["deps_outdated_note"] == ""

    def test_call_chain_field(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            call_chain=["index", "get_data", "requests.get"],
        )
        fields = _remediation_fields(f)
        assert fields["call_chain"] == "index() → get_data() → requests.get()"

    def test_call_chain_empty(self):
        from mycode.documents import _remediation_fields

        f = Finding(title="Issue", severity="critical", category="memory_profiling")
        fields = _remediation_fields(f)
        assert fields["call_chain"] == ""

    def test_has_cache_decorator_streamlit(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            source_decorators=["app.route", "st.cache_data"],
        )
        fields = _remediation_fields(f)
        assert fields["has_cache_decorator"] is True

    def test_has_cache_decorator_lru_cache(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            source_decorators=["lru_cache"],
        )
        fields = _remediation_fields(f)
        assert fields["has_cache_decorator"] is True

    def test_no_cache_decorator(self):
        from mycode.documents import _remediation_fields

        f = Finding(
            title="Issue",
            severity="critical",
            category="memory_profiling",
            source_decorators=["app.route"],
        )
        fields = _remediation_fields(f)
        assert fields["has_cache_decorator"] is False


class TestPatternVersionAwareness:
    """Tests that patterns include version/call-chain/decorator context."""

    def test_flask_concurrency_with_call_chain(self):
        from mycode.documents import _build_diagnosis

        f = Finding(
            title="Response time degradation",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
            source_file="app.py",
            source_function="index",
            call_chain=["index", "get_data", "requests.get"],
            _load_level=50,
        )
        diag = _build_diagnosis(f)
        assert "Call chain:" in diag
        assert "index()" in diag

    def test_flask_concurrency_with_outdated_dep(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Response time degradation",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
            dep_versions={"flask": "2.0.1"},
            dep_latest_versions={"flask": "3.1.0"},
            dep_outdated=["flask"],
            _load_level=50,
        )
        fix = _build_fix(f)
        assert "flask 2.0.1" in fix
        assert "outdated" in fix.lower()

    def test_streamlit_memory_with_cache_decorator(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Memory growth",
            severity="critical",
            category="memory_profiling",
            failure_pattern="memory_accumulation_over_sessions",
            affected_dependencies=["streamlit"],
            source_decorators=["st.cache_data"],
            _peak_memory_mb=100.0,
            _load_level=10,
        )
        fix = _build_fix(f)
        assert "already applied" in fix.lower()

    def test_streamlit_memory_without_cache_decorator(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Memory growth",
            severity="critical",
            category="memory_profiling",
            failure_pattern="memory_accumulation_over_sessions",
            affected_dependencies=["streamlit"],
            source_decorators=[],
            _peak_memory_mb=100.0,
            _load_level=10,
        )
        fix = _build_fix(f)
        assert "@st.cache_data" in fix

    def test_streamlit_response_time_with_cache_decorator(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Response time degradation",
            severity="critical",
            category="http_load_testing",
            failure_domain="concurrency_failure",
            affected_dependencies=["streamlit"],
            source_decorators=["st.cache_data"],
            _load_level=10,
        )
        fix = _build_fix(f)
        assert "already applied" in fix.lower()

    def test_fastapi_concurrency_with_version_note(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Concurrency failure",
            severity="critical",
            category="http_load_testing",
            failure_domain="concurrency_failure",
            affected_dependencies=["fastapi"],
            dep_versions={"fastapi": "0.95.0"},
            dep_latest_versions={"fastapi": "0.115.0"},
            dep_outdated=["fastapi"],
            _load_level=100,
        )
        fix = _build_fix(f)
        assert "fastapi 0.95.0" in fix
        assert "outdated" in fix.lower()

    def test_startup_failure_missing_dep_with_version(self):
        from mycode.documents import _build_fix

        f = Finding(
            title="Server could not start",
            severity="critical",
            category="http_load_testing",
            failure_pattern="missing_server_dependency",
            affected_dependencies=["flask"],
            dep_versions={"flask": "2.0.1"},
            dep_latest_versions={"flask": "3.1.0"},
            dep_outdated=["flask"],
        )
        fix = _build_fix(f)
        assert "flask 2.0.1" in fix


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

    def test_memory_finding_has_fix_objective(self):
        f = Finding(
            title="Memory Exhaustion",
            severity="critical",
            category="memory_profiling",
            description="Memory grew under load.",
            source_file="app.py",
            _peak_memory_mb=256.0,
        )
        prompt = generate_finding_prompt(f)
        assert "memory growth" in prompt.lower()
        assert "app.py" in prompt

    def test_blocking_io_finding_has_fix_objective(self):
        f = Finding(
            title="Slow Response",
            severity="warning",
            category="blocking_io",
            description="Response time degraded.",
            source_file="routes.py",
            _execution_time_ms=6000.0,
        )
        prompt = generate_finding_prompt(f)
        assert "blocking" in prompt.lower() or "event loop" in prompt.lower()
        assert "routes.py" in prompt

    def test_prompt_includes_load_level(self):
        f = Finding(
            title="Error Storm",
            severity="critical",
            category="edge_case_inputs",
            description="Many errors occurred.",
            source_file="handler.py",
            _load_level=42,
        )
        prompt = generate_finding_prompt(f)
        assert "42" in prompt


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

    def test_pdf_with_arrow_and_math_unicode(self):
        """PDF handles → ← ≤ ≥ × and other non-Latin-1 characters."""
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            scenarios_run=1,
            findings=[
                Finding(
                    title="Scale test \u2192 failure",
                    severity="critical",
                    description=(
                        "Testing from 50 \u2192 150 sessions. "
                        "Threshold \u2264 100 users. "
                        "Rate \u2265 3\u00d7 baseline."
                    ),
                ),
            ],
        )
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"

    def test_safe_text_replaces_arrows(self):
        """_safe_text replaces → and ← with ASCII equivalents."""
        from mycode.documents import _safe_text
        assert _safe_text("a \u2192 b") == "a -> b"
        assert _safe_text("a \u2190 b") == "a <- b"
        assert _safe_text("\u2264 100") == "<= 100"
        assert _safe_text("\u2265 100") == ">= 100"

    def test_safe_text_replaces_dashes_and_quotes(self):
        """_safe_text replaces em-dash, en-dash, smart quotes."""
        from mycode.documents import _safe_text
        assert _safe_text("a \u2014 b") == "a -- b"       # em-dash → --
        assert _safe_text("a \u2013 b") == "a - b"        # en-dash → -
        assert _safe_text("\u201chello\u201d") == '"hello"'  # smart double quotes
        assert _safe_text("\u2018hi\u2019") == "'hi'"        # smart single quotes

    def test_safe_text_replaces_bullets_and_math(self):
        """_safe_text replaces bullets, math symbols."""
        from mycode.documents import _safe_text
        assert _safe_text("\u2022 item") == "* item"       # bullet → *
        assert _safe_text("\u00b7 dot") == "* dot"         # middle dot → *
        assert _safe_text("3 \u00d7 4") == "3 x 4"        # × → x
        assert _safe_text("6 \u00f7 2") == "6 / 2"        # ÷ → /
        assert _safe_text("\u00b1 5") == "+/- 5"           # ± → +/-
        assert _safe_text("\u2026") == "..."               # ellipsis

    def test_safe_text_strips_unknown_unicode(self):
        """_safe_text strips non-Latin-1 characters not in the replacement map."""
        from mycode.documents import _safe_text
        # Snowman U+2603 is not in Latin-1 or the replacement map
        assert _safe_text("hello \u2603 world") == "hello  world"


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
        assert "Memory" in md

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
        assert "Monitor under load" in md
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
        assert "Memory" in md

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

    def test_dedup_same_label_keeps_worst(self):
        """Multiple curves with same label → one row with highest peak."""
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="coupling_compute_foo",
                    metric="execution_time_ms",
                    steps=[("compute_1000", 10.0), ("compute_10000", 50.0)],
                ),
                DegradationPoint(
                    scenario_name="coupling_compute_bar",
                    metric="execution_time_ms",
                    steps=[("compute_1000", 15.0), ("compute_10000", 200.0)],
                ),
                DegradationPoint(
                    scenario_name="coupling_compute_baz",
                    metric="execution_time_ms",
                    steps=[("compute_1000", 8.0), ("compute_10000", 30.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        # All three produce "Running calculations across components"
        # Should appear as ONE row, not three
        table_rows = [
            l for l in md.split("\n")
            if l.startswith("| ") and "calculations" in l.lower()
        ]
        assert len(table_rows) == 1
        # Should show the worst peak (200ms from bar)
        assert "200ms" in table_rows[0]

    def test_dedup_different_labels_kept(self):
        """Curves with different labels each get their own row."""
        report = DiagnosticReport(
            degradation_points=[
                DegradationPoint(
                    scenario_name="flask_data_volume_scaling",
                    metric="execution_time_ms",
                    steps=[("data_size_100", 5.0), ("data_size_10000", 100.0)],
                ),
                DegradationPoint(
                    scenario_name="flask_data_volume_scaling",
                    metric="memory_peak_mb",
                    steps=[("data_size_100", 10.0), ("data_size_10000", 60.0)],
                ),
            ],
        )
        md = render_understanding(report, 1)
        # Time and memory have different labels due to "Memory usage —" prefix
        data_rows = [
            l for l in md.split("\n")
            if l.startswith("| ") and "---" not in l and "What we tested" not in l
        ]
        assert len(data_rows) == 2


class TestPerfRowLabelNoTruncation:
    """Tests for _perf_row_label — labels should never be truncated."""

    def test_long_label_not_truncated(self):
        from mycode.documents import _perf_row_label
        dp = DegradationPoint(
            scenario_name="coupling_compute_very_long_function_name_with_details",
            metric="execution_time_ms",
            steps=[("compute_1000", 100.0)],
        )
        label = _perf_row_label(dp)
        assert "..." not in label

    def test_pdf_renders_long_label_without_truncation(self):
        """PDF performance table renders long labels with wrapping, not ellipsis."""
        from mycode.documents import render_understanding_pdf
        report = DiagnosticReport(
            scenarios_run=1,
            degradation_points=[
                DegradationPoint(
                    scenario_name="coupling_compute_streamlit_markdown_render_with_extra_detail",
                    metric="execution_time_ms",
                    steps=[
                        ("compute_1000", 50.0),
                        ("compute_5000", 200.0),
                        ("compute_10000", 800.0),
                    ],
                    breaking_point="compute_10000",
                ),
            ],
        )
        pdf_bytes = render_understanding_pdf(report, edition=1)
        assert pdf_bytes[:5] == b"%PDF-"

    def test_memory_label_preserved(self):
        from mycode.documents import _perf_row_label
        dp = DegradationPoint(
            scenario_name="pandas_large_dataframe_memory_growth_under_concurrent_load",
            metric="memory_peak_mb",
            steps=[("batch_100", 50.0)],
        )
        label = _perf_row_label(dp)
        assert label.startswith("Memory")
        assert "..." not in label


class TestVerdictFindingCrossRef:
    """Tests for _verdict cross-referencing findings."""

    def test_warning_finding_overrides_no_issues(self):
        """A WARNING finding for the same scenario overrides 'No issues' verdict."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="execution_time_ms",
            steps=[("concurrent_10", 3.0), ("concurrent_50", 29.0)],
        )
        findings = [
            Finding(
                title="Flask concurrent request load",
                severity="warning",
                category="concurrent_execution",
                description="9x degradation detected.",
            ),
        ]
        result = _verdict(dp, findings)
        assert "Warning" in result
        assert "see above" in result

    def test_critical_finding_overrides_threshold(self):
        """A CRITICAL finding takes precedence over threshold verdict."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="pandas_data_scaling",
            metric="execution_time_ms",
            steps=[("data_100", 10.0), ("data_10000", 50.0)],
        )
        findings = [
            Finding(
                title="Pandas data scaling",
                severity="critical",
                category="data_volume_scaling",
                description="Crashed at 10k rows.",
            ),
        ]
        result = _verdict(dp, findings)
        assert "Critical" in result

    def test_no_matching_finding_uses_threshold(self):
        """When no finding matches, threshold-based verdict applies."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="execution_time_ms",
            steps=[("concurrent_10", 3.0), ("concurrent_50", 29.0)],
        )
        findings = [
            Finding(
                title="Unrelated scenario",
                severity="critical",
                category="memory_profiling",
                description="Something else.",
            ),
        ]
        result = _verdict(dp, findings)
        assert result == "No issues"

    def test_info_finding_does_not_override(self):
        """INFO findings should not override the threshold verdict."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="execution_time_ms",
            steps=[("concurrent_10", 3.0), ("concurrent_50", 29.0)],
        )
        findings = [
            Finding(
                title="Flask concurrent request load",
                severity="info",
                category="concurrent_execution",
                description="Informational.",
            ),
        ]
        result = _verdict(dp, findings)
        assert result == "No issues"

    def test_no_findings_uses_threshold(self):
        """When findings=None, threshold-based verdict applies."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="flask_test",
            metric="execution_time_ms",
            steps=[("concurrent_10", 3000.0)],
        )
        assert _verdict(dp) == "Unresponsive at peak load"
        assert _verdict(dp, None) == "Unresponsive at peak load"
        assert _verdict(dp, []) == "Unresponsive at peak load"

    def test_md_table_includes_finding_verdict(self):
        """Markdown perf table shows finding-based verdict."""
        from mycode.documents import _render_perf_table_md
        points = [
            DegradationPoint(
                scenario_name="flask_concurrent_request_load",
                metric="execution_time_ms",
                steps=[("concurrent_10", 3.0), ("concurrent_50", 29.0)],
            ),
        ]
        findings = [
            Finding(
                title="Flask concurrent request load",
                severity="warning",
                category="concurrent_execution",
                description="9x degradation.",
            ),
        ]
        lines: list[str] = []
        _render_perf_table_md(lines, points, findings)
        table_text = "\n".join(lines)
        assert "Warning" in table_text
        assert "No issues" not in table_text

    def test_http_finding_matches_http_curve(self):
        """HTTP finding matches http_ prefixed scenario even with different title."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="http_get_root",
            metric="response_time_ms",
            steps=[("concurrent_1", 3.0), ("concurrent_50", 29.0)],
        )
        findings = [
            Finding(
                title="Response time degradation on your application",
                severity="warning",
                category="http_load_testing",
                description="9x degradation detected.",
            ),
        ]
        result = _verdict(dp, findings)
        assert "Warning" in result
        assert "see above" in result

    def test_memory_finding_matches_memory_curve(self):
        """Memory finding matches any curve with a memory metric."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="streamlit_session_growth",
            metric="memory_peak_mb",
            steps=[("batch_10", 50.0), ("batch_100", 300.0)],
        )
        findings = [
            Finding(
                title="Memory exhaustion under load",
                severity="critical",
                category="memory_profiling",
                description="Out of memory.",
            ),
        ]
        result = _verdict(dp, findings)
        assert "Critical" in result

    def test_dependency_overlap_matches(self):
        """Finding with matching dependency matches the curve."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="pandas_large_dataframe_scaling",
            metric="execution_time_ms",
            steps=[("data_100", 10.0), ("data_10000", 500.0)],
        )
        findings = [
            Finding(
                title="Data processing failure",
                severity="warning",
                category="data_volume_scaling",
                description="Slowed down.",
                affected_dependencies=["pandas"],
            ),
        ]
        result = _verdict(dp, findings)
        assert "Warning" in result

    def test_no_match_when_category_and_deps_differ(self):
        """No match when category, metric, and deps all differ."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="http_get_root",
            metric="response_time_ms",
            steps=[("concurrent_1", 3.0), ("concurrent_50", 29.0)],
        )
        findings = [
            Finding(
                title="Memory crash",
                severity="critical",
                category="memory_profiling",
                description="Out of memory.",
                affected_dependencies=["numpy"],
            ),
        ]
        result = _verdict(dp, findings)
        assert result == "No issues"

    def test_verdict_color_for_finding_verdicts(self):
        from mycode.documents import _verdict_color, _RED, _AMBER_TEXT
        assert _verdict_color("Critical -- see above") == _RED
        assert _verdict_color("Warning -- see above") == _AMBER_TEXT

    def test_flat_curve_overrides_critical_finding(self):
        """A near-flat curve (<15% growth) returns 'Stable' despite CRITICAL finding."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="http_memory_profiling",
            metric="memory_mb",
            steps=[("concurrent_1", 65.0), ("concurrent_1500", 70.0)],
        )
        findings = [
            Finding(
                title="Memory baseline limits concurrent capacity",
                severity="critical",
                category="http_load_testing",
            ),
        ]
        result = _verdict(dp, findings)
        assert result == "Stable"

    def test_growing_curve_still_gets_finding_severity(self):
        """A curve with significant growth (>15%) still gets finding severity."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="http_memory_profiling",
            metric="memory_mb",
            steps=[("concurrent_1", 65.0), ("concurrent_1500", 130.0)],
        )
        findings = [
            Finding(
                title="Memory baseline limits concurrent capacity",
                severity="critical",
                category="http_load_testing",
            ),
        ]
        result = _verdict(dp, findings)
        assert "Critical" in result

    def test_flat_time_curve_stable(self):
        """A near-flat time curve returns 'Stable' despite WARNING finding."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="flask_concurrent_request_load",
            metric="execution_time_ms",
            steps=[("concurrent_10", 50.0), ("concurrent_100", 55.0)],
        )
        findings = [
            Finding(
                title="Flask concurrent request degradation",
                severity="warning",
                category="concurrent_execution",
            ),
        ]
        result = _verdict(dp, findings)
        assert result == "Stable"

    def test_zero_baseline_skips_flat_check(self):
        """When first value is 0, flat-curve check is skipped (can't compute ratio)."""
        from mycode.documents import _verdict
        dp = DegradationPoint(
            scenario_name="http_error_rate",
            metric="error_count",
            steps=[("concurrent_1", 0.0), ("concurrent_100", 5.0)],
        )
        result = _verdict(dp, [])
        assert result != "Stable"


class TestConsequenceForUser:
    """Tests for _consequence_for_user consequence text generation."""

    def test_data_volume_scaling_with_load_level(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Data volume crash",
            severity="critical",
            category="data_volume_scaling",
            description="myCode tested data scaling.",
        )
        f._load_level = 5000
        result = _consequence_for_user(f)
        assert "5,000 items" in result
        assert "crashes or slowdowns" in result

    def test_memory_with_user_scale(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Memory exhaustion",
            severity="critical",
            category="memory_profiling",
            description="You said 500 users. Memory keeps growing.",
        )
        f._peak_memory_mb = 200.0
        result = _consequence_for_user(f)
        assert "500" in result
        assert "GB of RAM" in result

    def test_memory_without_user_scale(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Memory exhaustion",
            severity="critical",
            category="memory_profiling",
            description="Memory keeps growing.",
        )
        f._peak_memory_mb = 200.0
        result = _consequence_for_user(f)
        assert "200 MB" in result
        assert "exhaust server memory" in result

    def test_concurrent_with_response_time(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Slow under load",
            severity="warning",
            category="concurrent_execution",
            description="You said 100 users. Slowing down.",
        )
        f._execution_time_ms = 3500.0
        f._load_level = 100
        result = _consequence_for_user(f)
        assert "100" in result
        assert "3,500 ms" in result
        assert "slow" in result.lower()

    def test_http_load_testing(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="HTTP response degradation",
            severity="warning",
            category="http_load_testing",
            description="Response times increased.",
        )
        f._execution_time_ms = 6000.0
        f._load_level = 50
        result = _consequence_for_user(f)
        assert "50" in result
        assert "6,000 ms" in result
        assert "unresponsive" in result

    def test_within_capacity_framing(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Crash under load",
            severity="critical",
            category="concurrent_execution",
            description="You said 100 users. This breaks at 80.",
        )
        f._load_level = 80
        result = _consequence_for_user(f)
        assert "at the scale you described" in result

    def test_beyond_capacity_framing(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Break at high load",
            severity="info",
            category="concurrent_execution",
            description="You said 20 users. Breaks at 250.",
        )
        f._load_level = 250
        result = _consequence_for_user(f)
        assert "grows beyond" in result
        assert "20" in result

    def test_fallback_error_count(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Errors",
            severity="warning",
            category="edge_case_input",
            description="Errors occurred.",
        )
        f._error_count = 12
        result = _consequence_for_user(f)
        assert "12 errors" in result
        assert "real traffic" in result

    def test_response_time_qualifiers(self):
        from mycode.documents import _response_time_qualifier
        assert _response_time_qualifier(200) == "noticeable but acceptable"
        assert _response_time_qualifier(1000) == "noticeably slow"
        assert _response_time_qualifier(3000) == "slow"
        assert _response_time_qualifier(8000) == "unresponsive"

    def test_no_metrics_returns_empty(self):
        from mycode.documents import _consequence_for_user
        f = Finding(
            title="Something",
            severity="warning",
            category="edge_case_input",
            description="A test ran.",
        )
        result = _consequence_for_user(f)
        assert result == ""

    def test_extract_user_scale_from_desc(self):
        from mycode.documents import _extract_user_scale_from_desc
        assert _extract_user_scale_from_desc("You said 500 users. Breaks.") == 500
        assert _extract_user_scale_from_desc(
            "You said 10,000 users. Breaks."
        ) == 10000
        assert _extract_user_scale_from_desc("No user info here.") is None


# ── Architecture-aware remediation ──


class TestBuildRemediation:
    """Verify diagnosis + fix are framework-specific and separated."""

    def test_fastapi_concurrency(self):
        """FastAPI + concurrency_failure → FastAPI-specific diagnosis + fix."""
        from mycode.documents import _build_diagnosis, _build_fix, generate_finding_prompt
        f = Finding(
            title="Endpoint /api/preflight degrades under concurrent load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["fastapi", "uvicorn"],
        )
        f.failure_domain = "concurrency_failure"
        f._load_level = 50
        diagnosis = _build_diagnosis(f)
        assert "thread pool" in diagnosis
        assert "50 concurrent" in diagnosis
        fix = _build_fix(f)
        assert "ThreadPoolExecutor" in fix
        assert "async" in fix
        # Prompt contains both
        prompt = generate_finding_prompt(f)
        assert "Diagnosis:" in prompt
        assert "Fix:" in prompt

    def test_streamlit_memory(self):
        """Streamlit + memory_accumulation → Streamlit-specific diagnosis + fix."""
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory baseline limits concurrent capacity",
            severity="warning",
            category="http_load_testing",
            description="Your application uses 65MB per process.",
            affected_dependencies=["streamlit"],
        )
        f.failure_pattern = "memory_accumulation_over_sessions"
        diagnosis = _build_diagnosis(f)
        assert "Streamlit" in diagnosis
        assert "65MB" in diagnosis
        fix = _build_fix(f)
        assert "@st.cache_data" in fix

    def test_streamlit_response_time_unclassified(self):
        """Streamlit + http + 'response time' in title → matches pattern #4."""
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Response time degradation on your application",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=["streamlit"],
        )
        f.failure_domain = "unclassified"
        f.failure_pattern = None
        diagnosis = _build_diagnosis(f)
        assert "Streamlit reruns" in diagnosis
        fix = _build_fix(f)
        assert "@st.cache_data" in fix
        assert "@st.cache_resource" in fix

    def test_flask_http_load(self):
        """Flask + http_load_testing → Flask-specific diagnosis + fix."""
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application degrades under concurrent load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f._load_level = 25
        diagnosis = _build_diagnosis(f)
        assert "synchronously" in diagnosis
        assert "25 concurrent" in diagnosis
        fix = _build_fix(f)
        assert "gunicorn" in fix

    def test_memory_baseline_any_framework(self):
        """Any framework + memory baseline → diagnosis + fix."""
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory baseline limits concurrent capacity",
            severity="warning",
            category="http_load_testing",
            description="Your application uses 54MB per process.",
            affected_dependencies=["fastapi", "uvicorn"],
        )
        diagnosis = _build_diagnosis(f)
        assert "54MB" in diagnosis
        assert "not a leak" in diagnosis
        fix = _build_fix(f)
        assert "module-level" in fix

    def test_data_volume_scaling(self):
        """data_volume_scaling category → diagnosis + fix."""
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Pandas data scaling",
            severity="warning",
            category="data_volume_scaling",
            affected_dependencies=["pandas"],
        )
        diagnosis = _build_diagnosis(f)
        assert "time grows" in diagnosis
        fix = _build_fix(f)
        assert "chunked" in fix or "streaming" in fix
        assert "pagination" in fix

    def test_unknown_framework_falls_back(self):
        """Unknown framework + HTTP → empty diagnosis, generic fix."""
        from mycode.documents import _build_diagnosis, _build_fix, generate_finding_prompt
        f = Finding(
            title="Some HTTP issue",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=["unknown_dep"],
        )
        diagnosis = _build_diagnosis(f)
        assert diagnosis == ""
        fix = _build_fix(f)
        assert "concurrent HTTP load" in fix
        # Prompt has Fix but no Diagnosis
        prompt = generate_finding_prompt(f)
        assert "Diagnosis:" not in prompt
        assert "Fix:" in prompt


# ── _finding_severity_for_dp metric-aware matching ──


class TestFindingSeverityForDpMetricMatching:
    """Verify that _finding_severity_for_dp matches by category AND metric."""

    def _make_dp(self, scenario="http_endpoint_load", metric="response_time_ms"):
        return DegradationPoint(
            scenario_name=scenario,
            metric=metric,
            steps=[("1 user", 50), ("10 users", 200)],
        )

    def _make_finding(self, title, severity="critical", category="http_load_testing"):
        return Finding(
            title=title,
            severity=severity,
            category=category,
            description="test",
        )

    def test_response_time_dp_picks_warning_not_critical_memory(self):
        """Response time dp should match WARNING response-time finding,
        not CRITICAL memory finding, even though both are http_load_testing."""
        dp = self._make_dp(metric="response_time_ms")
        findings = [
            self._make_finding("Memory usage exceeds safe limits", severity="critical"),
            self._make_finding("Response time degradation under load", severity="warning"),
        ]
        assert _finding_severity_for_dp(dp, findings) == "warning"

    def test_memory_dp_picks_critical_memory_not_warning_response(self):
        """Memory dp should match CRITICAL memory finding, not WARNING response."""
        dp = self._make_dp(metric="memory_peak_mb")
        findings = [
            self._make_finding("Memory usage exceeds safe limits", severity="critical"),
            self._make_finding("Response time degradation under load", severity="warning"),
        ]
        assert _finding_severity_for_dp(dp, findings) == "critical"

    def test_response_time_dp_with_only_critical_response_finding(self):
        """When response time finding is critical, dp should return critical."""
        dp = self._make_dp(metric="response_time_ms")
        findings = [
            self._make_finding("Response time degradation under load", severity="critical"),
        ]
        assert _finding_severity_for_dp(dp, findings) == "critical"

    def test_fallback_to_category_when_no_metric_match(self):
        """If no finding title matches the metric, fall back to category match."""
        dp = self._make_dp(metric="response_time_ms")
        findings = [
            self._make_finding("Server crashed under load", severity="critical"),
        ]
        # "Server crashed" doesn't mention response time or memory,
        # so metric filter doesn't match — should fall back to category
        assert _finding_severity_for_dp(dp, findings) == "critical"

    def test_execution_time_matches_latency_keyword(self):
        """execution_time_ms metric should match 'latency' in title."""
        dp = self._make_dp(metric="execution_time_ms")
        findings = [
            self._make_finding("Memory usage exceeds safe limits", severity="critical"),
            self._make_finding("High latency detected at scale", severity="warning"),
        ]
        assert _finding_severity_for_dp(dp, findings) == "warning"

    def test_memory_growth_matches_memory_keyword(self):
        """memory_growth_mb metric should match 'memory' in title."""
        dp = self._make_dp(metric="memory_growth_mb")
        findings = [
            self._make_finding("Response time degradation", severity="critical"),
            self._make_finding("Memory leak detected", severity="warning"),
        ]
        assert _finding_severity_for_dp(dp, findings) == "warning"

    def test_non_http_categories_unaffected(self):
        """Non-HTTP categories should still work as before."""
        dp = self._make_dp(
            scenario="concurrent_request_load", metric="execution_time_ms",
        )
        findings = [
            Finding(
                title="Concurrent access fails",
                severity="critical",
                category="concurrent_execution",
                description="test",
            ),
        ]
        assert _finding_severity_for_dp(dp, findings) == "critical"

    def test_dep_overlap_respects_metric_time_vs_memory(self):
        """Dependency overlap with memory finding should not match time curve."""
        dp = self._make_dp(
            scenario="coupling_compute_streamlit_markdown",
            metric="execution_time_ms",
        )
        findings = [
            self._make_finding(
                "Memory baseline limits concurrent capacity",
                severity="critical",
                category="http_load_testing",
            ),
        ]
        # streamlit is in the dp name but the finding is about memory
        # and the curve is execution_time — should NOT match
        findings[0].affected_dependencies = ["streamlit"]
        assert _finding_severity_for_dp(dp, findings) is None

    def test_dep_overlap_matches_compatible_metric(self):
        """Dependency overlap with time finding matches time curve."""
        dp = self._make_dp(
            scenario="coupling_compute_streamlit_markdown",
            metric="execution_time_ms",
        )
        findings = [
            self._make_finding(
                "Response time degradation under load",
                severity="warning",
                category="http_load_testing",
            ),
        ]
        findings[0].affected_dependencies = ["streamlit"]
        assert _finding_severity_for_dp(dp, findings) == "warning"

    def test_dep_overlap_memory_finding_matches_memory_curve(self):
        """Dependency overlap with memory finding matches memory curve."""
        dp = self._make_dp(
            scenario="coupling_compute_streamlit_markdown",
            metric="memory_peak_mb",
        )
        findings = [
            self._make_finding(
                "Memory baseline limits concurrent capacity",
                severity="critical",
                category="http_load_testing",
            ),
        ]
        findings[0].affected_dependencies = ["streamlit"]
        assert _finding_severity_for_dp(dp, findings) == "critical"


# ── _integrate_details ──


class TestIntegrateDetails:
    def test_at_prefix_becomes_sentence(self):
        desc = "Memory usage grows without bound, eventually crashing."
        detail = "at first iteration"
        result = _integrate_details(desc, detail)
        assert result == (
            "Memory usage grows without bound, eventually crashing."
            " This issue begins at first iteration."
        )

    def test_at_prefix_with_number(self):
        desc = "Data processing slows significantly"
        detail = "at 25,000 items"
        result = _integrate_details(desc, detail)
        assert result == (
            "Data processing slows significantly."
            " This issue begins at 25,000 items."
        )

    def test_at_prefix_strips_trailing_period(self):
        desc = "Response time degrades."
        detail = "at 10 concurrent users."
        result = _integrate_details(desc, detail)
        assert result == (
            "Response time degrades."
            " This issue begins at 10 concurrent users."
        )

    def test_non_at_detail_appended_as_sentence(self):
        desc = "The function fails under load"
        detail = "Error count reached 47"
        result = _integrate_details(desc, detail)
        assert result == "The function fails under load. Error count reached 47."

    def test_empty_details_returns_description(self):
        assert _integrate_details("Some description.", "") == "Some description."

    def test_empty_description_returns_details(self):
        assert _integrate_details("", "some detail") == "some detail"

    def test_duplicate_detail_skipped(self):
        desc = "Memory usage grows at first iteration and keeps going."
        detail = "at first iteration"
        result = _integrate_details(desc, detail)
        # "at first iteration" is already in desc, so should be skipped
        assert result == desc


# ── Enriched & new pattern quality tests ──


def _assert_prompt_quality(diagnosis: str, fix: str, label: str):
    """Assert no None, no empty parens, no broken sentences, no trailing artifacts."""
    import re as _re
    for name, text in [("diagnosis", diagnosis), ("fix", fix)]:
        assert text, f"{label} {name} is empty"
        # Strip backtick-quoted spans (and surrounding parens if present)
        stripped = _re.sub(r"\(`[^`]*`\)", "", text)
        stripped = _re.sub(r"`[^`]*`", "", stripped)
        # Check for bare "None" as standalone word (not NoneType, not in error messages)
        # Allow NoneType and None inside known error patterns
        no_errors = _re.sub(r"\w+Error:.*", "", stripped)
        no_errors = _re.sub(r"\w+Exception:.*", "", no_errors)
        assert not _re.search(r"\bNone\b", no_errors), (
            f"{label} {name} contains bare 'None': {text!r}"
        )
        # Check for empty "()" — sign of failed interpolation
        # Remove function-call-like patterns (word followed by parens with content)
        no_calls = _re.sub(r"\w+\([^)]*\)", "", stripped)
        assert "()" not in no_calls, f"{label} {name} contains '()': {text!r}"
        # No broken sentences: ". ." or ".." at non-ellipsis positions
        assert ". ." not in text, f"{label} {name} has broken sentence: {text!r}"
        # No trailing punctuation artifacts: multiple punctuation at end
        assert not _re.search(r"[.!?]{2,}$", text.rstrip()), (
            f"{label} {name} trailing punctuation artifact: {text!r}"
        )
        # No empty conditional fragments like " — " at start or "  " double-space
        assert "  " not in text, f"{label} {name} has double space: {text!r}"
        # No dangling " —" at very start
        assert not text.startswith(" —"), f"{label} {name} starts with ' —': {text!r}"
        assert not text.startswith(" "), f"{label} {name} starts with space: {text!r}"


class TestEnrichedPatternQuality:
    """Each of the 15 patterns tested with fully-populated AND minimal findings.

    Asserts: no None, no empty (), no broken sentences, no trailing
    punctuation artifacts in diagnosis and fix output.
    """

    # ── 1. _pat_fastapi_concurrency ──

    def test_fastapi_concurrency_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /api/data degrades under concurrent load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["fastapi", "uvicorn"],
            source_file="main.py",
            source_function="get_data",
            details="ConnectionError: pool exhausted",
        )
        f.failure_domain = "concurrency_failure"
        f._load_level = 50
        f._execution_time_ms = 8500.0
        f._error_count = 12
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "fastapi_concurrency_full")
        assert "main.py" in d
        assert "get_data" in d
        assert "8500ms" in d
        assert "12 errors" in d
        assert "sustained load" in d
        assert "ThreadPoolExecutor" in fix

    def test_fastapi_concurrency_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /api degrades",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["fastapi"],
        )
        f.failure_domain = "concurrency_failure"
        f._load_level = 10
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "fastapi_concurrency_minimal")
        assert "thread pool" in d
        assert "10 concurrent" in d

    # ── 2. _pat_startup_failure (5 sub-patterns) ──

    def test_startup_missing_dep_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
            source_file="app.py",
            details="Traceback (most recent call last):\n  File \"app.py\", line 1\nModuleNotFoundError: No module named 'flask_cors'",
        )
        f.failure_pattern = "missing_server_dependency"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_dep_full")
        assert "app.py" in d
        assert "ModuleNotFoundError" in d
        assert "app.py" in fix

    def test_startup_missing_dep_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f.failure_pattern = "missing_server_dependency"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_dep_minimal")
        assert "missing" in d.lower()

    def test_startup_missing_env_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
            source_file="config.py",
            details="KeyError: 'DATABASE_URL'",
        )
        f.failure_pattern = "missing_env_config"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_env_full")
        assert "environment" in d.lower()
        assert "config.py" in d

    def test_startup_missing_env_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["express"],
        )
        f.failure_pattern = "missing_env_config"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_env_minimal")

    def test_startup_missing_service_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask", "sqlalchemy"],
            source_file="app.py",
            details="OperationalError: could not connect to server",
        )
        f.failure_pattern = "missing_external_service"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_service_full")
        assert "external service" in d.lower()
        assert "app.py" in d

    def test_startup_missing_service_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f.failure_pattern = "missing_external_service"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_missing_service_minimal")

    def test_startup_syntax_error_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
            source_file="views.py",
            details="SyntaxError: invalid syntax (views.py, line 42)",
        )
        f.failure_pattern = "server_syntax_error"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_syntax_error_full")
        assert "syntax" in d.lower()
        assert "views.py" in d

    def test_startup_syntax_error_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f.failure_pattern = "server_syntax_error"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_syntax_error_minimal")

    def test_startup_generic_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask", "redis"],
            source_file="run.py",
            details="RuntimeError: unexpected failure on boot",
        )
        f.failure_pattern = "unknown_reason"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_generic_full")
        assert "run.py" in d

    def test_startup_generic_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application could not start",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f.failure_pattern = "unknown_reason"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "startup_generic_minimal")

    # ── 3. _pat_streamlit_memory ──

    def test_streamlit_memory_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory accumulation over sessions",
            severity="warning",
            category="memory_profiling",
            description="Uses 80MB per session",
            affected_dependencies=["streamlit", "pandas"],
            source_file="dashboard.py",
            source_function="load_data",
            details="Memory keeps growing with each new session.",
        )
        f.failure_pattern = "memory_accumulation_over_sessions"
        f._peak_memory_mb = 80.0
        f._load_level = 10
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "streamlit_memory_full")
        assert "dashboard.py" in d
        assert "load_data" in d
        assert "80MB" in d
        assert "~800MB" in d
        assert "@st.cache_data" in fix

    def test_streamlit_memory_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory baseline limits concurrent capacity",
            severity="warning",
            category="http_load_testing",
            description="Your application uses 65MB per process.",
            affected_dependencies=["streamlit"],
        )
        f.failure_pattern = "memory_accumulation_over_sessions"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "streamlit_memory_minimal")
        assert "65MB" in d
        assert "Streamlit" in d

    # ── 4. _pat_streamlit_response_time ──

    def test_streamlit_response_time_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Response time degradation under load",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=["streamlit", "plotly"],
            source_file="app.py",
            source_function="render_chart",
            details="Slow chart rendering under concurrent sessions.",
        )
        f.failure_domain = "concurrency_failure"
        f._load_level = 20
        f._execution_time_ms = 5000.0
        f._error_count = 3
        f.operational_trigger = "burst_traffic"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "streamlit_response_time_full")
        assert "Streamlit reruns" in d
        assert "app.py" in d
        assert "5000ms" in d
        assert "3 errors" in d
        assert "@st.cache_data" in fix

    def test_streamlit_response_time_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Response time degradation on your application",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=["streamlit"],
        )
        f.failure_domain = "unclassified"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "streamlit_response_time_minimal")
        assert "Streamlit reruns" in d
        assert "@st.cache_resource" in fix

    # ── 5. _pat_flask_concurrency ──

    def test_flask_concurrency_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application degrades under concurrent load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask", "sqlalchemy"],
            source_file="views.py",
            source_function="get_users",
            details="All workers busy, requests queuing.",
        )
        f._load_level = 25
        f._execution_time_ms = 4200.0
        f._error_count = 5
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "flask_concurrency_full")
        assert "synchronously" in d
        assert "views.py" in d
        assert "4200ms" in d
        assert "gunicorn" in fix

    def test_flask_concurrency_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Application degrades under concurrent load",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["flask"],
        )
        f._load_level = 25
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "flask_concurrency_minimal")
        assert "synchronously" in d
        assert "25 concurrent" in d

    # ── 6. _pat_response_time_cliff_generic (NEW) ──

    def test_response_time_cliff_generic_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Response time cliff at 25 connections",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["express"],
            source_file="server.js",
            source_function="handleRequest",
            details="Latency spike from 50ms to 8500ms.",
        )
        f.failure_pattern = "response_time_cliff"
        f._load_level = 25
        f._execution_time_ms = 8500.0
        f._error_count = 7
        f.operational_trigger = "burst_traffic"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "response_time_cliff_generic_full")
        assert "server.js" in d
        assert "handleRequest" in d
        assert "8500ms" in d
        assert "burst traffic" in d
        assert "25 concurrent" in fix

    def test_response_time_cliff_generic_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Response time cliff",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=[],
        )
        f.failure_pattern = "response_time_cliff"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "response_time_cliff_generic_minimal")
        assert "degrades sharply" in d
        assert "bottleneck" in d

    # ── 7. _pat_memory_baseline ──

    def test_memory_baseline_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory baseline limits concurrent capacity",
            severity="warning",
            category="memory_profiling",
            description="54MB per process.",
            affected_dependencies=["fastapi", "numpy"],
            source_file="app.py",
            source_function="create_app",
            details="numpy alone contributes 40MB.",
        )
        f._peak_memory_mb = 54.0
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "memory_baseline_full")
        assert "54MB" in d
        assert "app.py" in d
        assert "not a leak" in d

    def test_memory_baseline_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Memory baseline limits concurrent capacity",
            severity="warning",
            category="http_load_testing",
            description="Your application uses 54MB per process.",
            affected_dependencies=["fastapi", "uvicorn"],
        )
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "memory_baseline_minimal")
        assert "54MB" in d

    # ── 8. _pat_unbounded_cache_growth (NEW) ──

    def test_unbounded_cache_growth_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Cache memory grows without bound",
            severity="critical",
            category="memory_profiling",
            affected_dependencies=["requests"],
            source_file="utils.py",
            source_function="fetch_user_profile",
            details="Dict cache grows with every unique user ID.",
        )
        f.failure_pattern = "unbounded_cache_growth"
        f._peak_memory_mb = 450.0
        f._load_level = 50
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "unbounded_cache_growth_full")
        assert "utils.py" in d
        assert "fetch_user_profile" in d
        assert "450MB" in d
        assert "lru_cache" in fix

    def test_unbounded_cache_growth_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Cache grows",
            severity="warning",
            category="memory_profiling",
            affected_dependencies=[],
        )
        f.failure_pattern = "unbounded_cache_growth"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "unbounded_cache_growth_minimal")
        assert "never evicted" in d
        assert "eviction" in fix

    # ── 9. _pat_http_endpoint_blocking ──

    def test_http_endpoint_blocking_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /health slow response",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["requests"],
            source_file="routes.py",
            source_function="health_check",
            details="Handler calls external API without timeout.",
        )
        f._load_level = 2
        f._execution_time_ms = 15000.0
        f._error_count = 1
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "http_endpoint_blocking_full")
        assert "routes.py" in d
        assert "15000ms" in d
        assert "2 concurrent" in d

    def test_http_endpoint_blocking_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /api slow",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=[],
        )
        f._load_level = 1
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "http_endpoint_blocking_minimal")
        assert "1 concurrent" in d
        assert "blocking" in d.lower() or "thread" in d.lower()

    # ── 10. _pat_external_timeout ──

    def test_external_timeout_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /data skipped due to slow response",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=["requests"],
            source_file="api.py",
            source_function="fetch_data",
            details="Timeout waiting for upstream service.",
        )
        f._execution_time_ms = 12000.0
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "external_timeout_full")
        assert "api.py" in d
        assert "12000ms" in d
        assert "timeout" in fix.lower()

    def test_external_timeout_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Endpoint /x skipped due to slow response",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=[],
        )
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "external_timeout_minimal")
        assert "external service" in d.lower()

    # ── 11. _pat_cascading_timeout (NEW) ──

    def test_cascading_timeout_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Cascading timeout in request chain",
            severity="critical",
            category="http_load_testing",
            affected_dependencies=["requests", "httpx"],
            source_file="app.py",
            source_function="get_dashboard",
            details="Chain: get_dashboard -> fetch_user -> external API.",
        )
        f.failure_pattern = "cascading_timeout"
        f._load_level = 10
        f._execution_time_ms = 12500.0
        f.operational_trigger = "sustained_load"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "cascading_timeout_full")
        assert "app.py" in d
        assert "get_dashboard" in d
        assert "12500ms" in d
        assert "circuit breakers" in fix

    def test_cascading_timeout_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Cascading timeout",
            severity="warning",
            category="http_load_testing",
            affected_dependencies=[],
        )
        f.failure_pattern = "cascading_timeout"
        f._load_level = 10
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "cascading_timeout_minimal")
        assert "cascading" in d.lower()
        assert "timeout" in fix.lower()

    # ── 12. _pat_pandas_silent_dtypes ──

    def test_pandas_silent_dtypes_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Silent dtype conversion in data pipeline",
            severity="warning",
            category="edge_case_input",
            affected_dependencies=["pandas", "numpy"],
            source_file="etl.py",
            source_function="clean_data",
            details="Column 'age' converted from int64 to object.",
        )
        f.failure_pattern = "silent_data_type_changes"
        f._peak_memory_mb = 120.0
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "pandas_silent_dtypes_full")
        assert "etl.py" in d
        assert "120MB" in d
        assert "read_csv" in fix

    def test_pandas_silent_dtypes_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Pandas dtype issue",
            severity="warning",
            category="edge_case_input",
            affected_dependencies=["pandas"],
        )
        f.failure_pattern = "silent_data_type_changes"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "pandas_silent_dtypes_minimal")
        assert "Pandas" in d
        assert "object" in d

    # ── 13. _pat_unvalidated_type_crash (NEW) ──

    def test_unvalidated_type_crash_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Type crash on user input",
            severity="critical",
            category="edge_case_inputs",
            affected_dependencies=["flask", "pydantic"],
            source_file="routes.py",
            source_function="create_user",
            details="TypeError: int() argument must be a string, not NoneType",
        )
        f.failure_pattern = "unvalidated_type_crash"
        f._load_level = 1
        f._error_count = 12
        f.operational_trigger = "format_variation"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "unvalidated_type_crash_full")
        assert "routes.py" in d
        assert "create_user" in d
        assert "12 errors" in d
        assert "TypeError" in d
        assert "Pydantic" in fix or "pydantic" in fix.lower()

    def test_unvalidated_type_crash_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Type crash",
            severity="warning",
            category="edge_case_inputs",
            affected_dependencies=[],
        )
        f.failure_pattern = "unvalidated_type_crash"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "unvalidated_type_crash_minimal")
        assert "unexpected type" in d
        assert "validation" in fix.lower()

    # ── 14. _pat_requests_concurrent ──

    def test_requests_concurrent_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Concurrent request bottleneck",
            severity="warning",
            category="concurrent_execution",
            affected_dependencies=["requests", "httpx"],
            source_file="client.py",
            source_function="fetch_all",
            details="All 20 threads blocked waiting for responses.",
        )
        f.failure_domain = "concurrency_failure"
        f._load_level = 20
        f._execution_time_ms = 6000.0
        f._error_count = 4
        f.operational_trigger = "concurrent_access"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "requests_concurrent_full")
        assert "client.py" in d
        assert "6000ms" in d
        assert "httpx" in fix

    def test_requests_concurrent_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Concurrent bottleneck",
            severity="warning",
            category="concurrent_execution",
            affected_dependencies=["requests"],
        )
        f._load_level = 5
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "requests_concurrent_minimal")
        assert "synchronous" in d
        assert "requests" in d.lower()

    # ── 15. _pat_data_volume ──

    def test_data_volume_full(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Data scaling degradation",
            severity="warning",
            category="data_volume_scaling",
            affected_dependencies=["pandas", "sqlalchemy"],
            source_file="pipeline.py",
            source_function="process_batch",
            details="Processing 100k rows takes 45s.",
        )
        f._load_level = 5
        f._execution_time_ms = 45000.0
        f._peak_memory_mb = 512.0
        f._error_count = 2
        f.operational_trigger = "large_input"
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "data_volume_full")
        assert "pipeline.py" in d
        assert "45000ms" in d
        assert "512MB" in d
        assert "pagination" in fix

    def test_data_volume_minimal(self):
        from mycode.documents import _build_diagnosis, _build_fix
        f = Finding(
            title="Data scaling issue",
            severity="warning",
            category="data_volume_scaling",
            affected_dependencies=[],
        )
        d = _build_diagnosis(f)
        fix = _build_fix(f)
        _assert_prompt_quality(d, fix, "data_volume_minimal")
        assert "time grows" in d.lower()
        assert "chunked" in fix or "streaming" in fix


class TestDetailExcerptErrorExtraction:
    """Verify detail_excerpt extracts error type+message from tracebacks."""

    def test_traceback_extracts_error_line(self):
        from mycode.documents import _remediation_fields
        f = Finding(
            title="Test",
            severity="warning",
            details=(
                "Traceback (most recent call last):\n"
                "  File \"app.py\", line 42, in handler\n"
                "    result = int(value)\n"
                "TypeError: int() argument must be a string, not NoneType"
            ),
        )
        fields = _remediation_fields(f)
        assert fields["detail_excerpt"] == "TypeError: int() argument must be a string, not NoneType"

    def test_error_colon_extracts_line(self):
        from mycode.documents import _remediation_fields
        f = Finding(
            title="Test",
            severity="warning",
            details="ConnectionError: Connection refused to localhost:5432",
        )
        fields = _remediation_fields(f)
        assert "ConnectionError" in fields["detail_excerpt"]

    def test_exception_extracts_line(self):
        from mycode.documents import _remediation_fields
        f = Finding(
            title="Test",
            severity="warning",
            details=(
                "Traceback (most recent call last):\n"
                "  File \"db.py\", line 10\n"
                "OperationalException: database locked"
            ),
        )
        fields = _remediation_fields(f)
        assert fields["detail_excerpt"] == "OperationalException: database locked"

    def test_no_error_uses_truncation(self):
        from mycode.documents import _remediation_fields
        long_text = "Memory keeps growing with each session. " * 10
        f = Finding(
            title="Test",
            severity="warning",
            details=long_text,
        )
        fields = _remediation_fields(f)
        assert len(fields["detail_excerpt"]) <= 200
        assert fields["detail_excerpt"].endswith(".")

    def test_empty_details(self):
        from mycode.documents import _remediation_fields
        f = Finding(title="Test", severity="warning", details="")
        fields = _remediation_fields(f)
        assert fields["detail_excerpt"] == ""
