"""Tests for Baseline Viability Gate — sandbox health checks.

Tests cover:
  - Gate pass/fail logic for each threshold
  - Import check with name mapping
  - Composite gate decisions
  - Baseline failed report generation
  - Pipeline integration (early return on failure)
  - Edge cases (zero deps, all dev deps, meta-failures)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mycode.ingester import DependencyInfo, FileAnalysis, IngestionResult
from mycode.session import SessionResult
from mycode.viability import (
    IMPORT_RATE_THRESHOLD,
    INSTALL_RATE_THRESHOLD,
    SYNTAX_RATE_THRESHOLD,
    ViabilityResult,
    _is_js_project,
    _python_import_name,
    _run_js_viability_gate,
    build_baseline_failed_text,
    check_importability,
    check_js_imports,
    check_python_imports,
    run_viability_gate,
)


# ── Helpers ──


def _make_dep(name: str, missing: bool = False, is_dev: bool = False,
              installed_version: str | None = "1.0.0",
              required_version: str | None = None) -> DependencyInfo:
    """Create a DependencyInfo for testing."""
    return DependencyInfo(
        name=name,
        installed_version=None if missing else installed_version,
        is_missing=missing,
        is_dev=is_dev,
        required_version=required_version,
    )


def _make_ingestion(
    deps: list[DependencyInfo] | None = None,
    files_analyzed: int = 10,
    files_failed: int = 0,
    file_analyses: list | None = None,
) -> IngestionResult:
    """Create an IngestionResult with the given dep/file counts."""
    return IngestionResult(
        project_path="/tmp/fake_project",
        dependencies=deps or [],
        files_analyzed=files_analyzed,
        files_failed=files_failed,
        file_analyses=file_analyses or [],
        total_lines=100,
    )


def _make_session(import_results: dict[str, bool] | None = None,
                  returncode: int = 0,
                  raise_error: bool = False) -> MagicMock:
    """Create a mock SessionManager that returns controlled import check results."""
    session = MagicMock()
    if raise_error:
        session.run_in_session.side_effect = Exception("subprocess failed")
    elif import_results is not None:
        session.run_in_session.return_value = SessionResult(
            returncode=returncode,
            stdout=json.dumps(import_results) + "\n",
            stderr="",
        )
    else:
        session.run_in_session.return_value = SessionResult(
            returncode=returncode,
            stdout="{}",
            stderr="",
        )
    return session


# ── Test: Python Import Name Mapping ──


class TestPythonImportName:
    """Test the package name → import name mapping."""

    def test_known_mapping_scikit_learn(self):
        assert _python_import_name("scikit-learn") == "sklearn"

    def test_known_mapping_pillow(self):
        assert _python_import_name("Pillow") == "PIL"

    def test_known_mapping_beautifulsoup4(self):
        assert _python_import_name("beautifulsoup4") == "bs4"

    def test_known_mapping_opencv(self):
        assert _python_import_name("opencv-python") == "cv2"

    def test_known_mapping_pyyaml(self):
        assert _python_import_name("PyYAML") == "yaml"

    def test_known_mapping_python_dotenv(self):
        assert _python_import_name("python-dotenv") == "dotenv"

    def test_known_mapping_pyjwt(self):
        assert _python_import_name("PyJWT") == "jwt"

    def test_known_mapping_attrs(self):
        assert _python_import_name("attrs") == "attr"

    def test_hyphen_to_underscore_fallback(self):
        assert _python_import_name("my-package") == "my_package"

    def test_simple_name_passthrough(self):
        assert _python_import_name("flask") == "flask"

    def test_case_insensitive_lookup(self):
        assert _python_import_name("PILLOW") == "PIL"
        assert _python_import_name("Scikit-Learn") == "sklearn"


# ── Test: Import Checking ──


class TestCheckPythonImports:
    """Test Python import checking in sandbox."""

    def test_all_importable(self):
        deps = [_make_dep("flask"), _make_dep("requests")]
        session = _make_session({"flask": True, "requests": True})
        importable, failed = check_python_imports(session, deps)
        assert importable == ["flask", "requests"]
        assert failed == []

    def test_some_failed(self):
        deps = [_make_dep("flask"), _make_dep("broken_pkg")]
        session = _make_session({"flask": True, "broken_pkg": False})
        importable, failed = check_python_imports(session, deps)
        assert importable == ["flask"]
        assert failed == ["broken_pkg"]

    def test_all_failed(self):
        deps = [_make_dep("a"), _make_dep("b")]
        session = _make_session({"a": False, "b": False})
        importable, failed = check_python_imports(session, deps)
        assert importable == []
        assert failed == ["a", "b"]

    def test_empty_deps(self):
        importable, failed = check_python_imports(MagicMock(), [])
        assert importable == []
        assert failed == []

    def test_subprocess_failure_fail_open(self):
        """If the import check script itself crashes, assume all importable."""
        deps = [_make_dep("flask"), _make_dep("django")]
        session = _make_session(raise_error=True)
        importable, failed = check_python_imports(session, deps)
        assert set(importable) == {"flask", "django"}
        assert failed == []

    def test_nonzero_returncode_fail_open(self):
        """If subprocess returns non-zero, assume all importable."""
        deps = [_make_dep("flask")]
        session = _make_session(import_results=None, returncode=1)
        session.run_in_session.return_value = SessionResult(
            returncode=1, stdout="", stderr="error",
        )
        importable, failed = check_python_imports(session, deps)
        assert importable == ["flask"]
        assert failed == []


class TestCheckJsImports:
    """Test JavaScript import checking in sandbox."""

    def test_all_importable(self):
        deps = [_make_dep("express"), _make_dep("lodash")]
        session = _make_session({"express": True, "lodash": True})
        importable, failed = check_js_imports(session, deps)
        assert importable == ["express", "lodash"]
        assert failed == []

    def test_some_failed(self):
        deps = [_make_dep("express"), _make_dep("broken")]
        session = _make_session({"express": True, "broken": False})
        importable, failed = check_js_imports(session, deps)
        assert importable == ["express"]
        assert failed == ["broken"]

    def test_empty_deps(self):
        importable, failed = check_js_imports(MagicMock(), [])
        assert importable == []
        assert failed == []


class TestCheckImportability:
    """Test the language-dispatch wrapper."""

    def test_dispatches_python(self):
        deps = [_make_dep("flask")]
        session = _make_session({"flask": True})
        importable, failed = check_importability(session, deps, "python")
        assert importable == ["flask"]

    def test_dispatches_javascript(self):
        deps = [_make_dep("express")]
        session = _make_session({"express": True})
        importable, failed = check_importability(session, deps, "javascript")
        assert importable == ["express"]

    def test_empty_deps_returns_empty(self):
        importable, failed = check_importability(MagicMock(), [], "python")
        assert importable == []
        assert failed == []


# ── Test: Gate Logic ──


class TestViabilityGate:
    """Test the composite viability gate."""

    def test_passes_all_deps_installed_and_importable(self):
        deps = [_make_dep("flask"), _make_dep("requests"), _make_dep("sqlalchemy")]
        ingestion = _make_ingestion(deps, files_analyzed=10, files_failed=0)
        session = _make_session({"flask": True, "requests": True, "sqlalchemy": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.install_rate == 1.0
        assert result.import_rate == 1.0
        assert result.syntax_rate == 1.0
        assert result.reason == ""

    def test_fails_below_50_percent_installed(self):
        deps = [
            _make_dep("flask"),
            _make_dep("pandas", missing=True),
            _make_dep("numpy", missing=True),
            _make_dep("scipy", missing=True),
        ]
        ingestion = _make_ingestion(deps, files_analyzed=10)
        session = _make_session({"flask": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is False
        assert result.install_rate == 0.25
        assert "1 of 4" in result.reason

    def test_passes_at_exactly_50_percent_installed(self):
        deps = [
            _make_dep("flask"),
            _make_dep("requests"),
            _make_dep("pandas", missing=True),
            _make_dep("numpy", missing=True),
        ]
        ingestion = _make_ingestion(deps)
        session = _make_session({"flask": True, "requests": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.install_rate == 0.50

    def test_fails_below_50_percent_importable(self):
        deps = [
            _make_dep("flask"),
            _make_dep("numpy"),
            _make_dep("scipy"),
            _make_dep("pandas"),
        ]
        ingestion = _make_ingestion(deps)
        session = _make_session({
            "flask": True, "numpy": False, "scipy": False, "pandas": False,
        })

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is False
        assert result.import_rate == 0.25
        assert "importable" in result.reason

    def test_passes_at_exactly_50_percent_importable(self):
        deps = [_make_dep("flask"), _make_dep("numpy")]
        ingestion = _make_ingestion(deps)
        session = _make_session({"flask": True, "numpy": False})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.import_rate == 0.50

    def test_fails_below_25_percent_syntax(self):
        deps = [_make_dep("flask")]
        ingestion = _make_ingestion(deps, files_analyzed=1, files_failed=5)
        session = _make_session({"flask": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is False
        assert result.syntax_rate == pytest.approx(1 / 6)
        assert "parsed" in result.reason

    def test_passes_at_exactly_25_percent_syntax(self):
        deps = [_make_dep("flask")]
        ingestion = _make_ingestion(deps, files_analyzed=1, files_failed=3)
        session = _make_session({"flask": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.syntax_rate == 0.25

    def test_passes_no_dependencies(self):
        ingestion = _make_ingestion([], files_analyzed=5)
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.install_rate == 1.0
        assert result.import_rate == 1.0

    def test_ignores_dev_dependencies(self):
        """Dev deps don't count toward install rate."""
        deps = [
            _make_dep("flask"),
            _make_dep("pytest", missing=True, is_dev=True),
            _make_dep("black", missing=True, is_dev=True),
            _make_dep("mypy", missing=True, is_dev=True),
        ]
        ingestion = _make_ingestion(deps)
        session = _make_session({"flask": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        assert result.install_rate == 1.0  # Only flask (non-dev) counts

    def test_suggests_docker_when_not_containerised(self):
        deps = [_make_dep("a", missing=True), _make_dep("b", missing=True)]
        ingestion = _make_ingestion(deps)
        session = MagicMock()

        result = run_viability_gate(
            session, ingestion, "python", is_containerised=False,
        )
        assert result.viable is False
        assert result.suggest_docker is True

    def test_no_docker_suggestion_when_containerised(self):
        deps = [_make_dep("a", missing=True), _make_dep("b", missing=True)]
        ingestion = _make_ingestion(deps)
        session = MagicMock()

        result = run_viability_gate(
            session, ingestion, "python", is_containerised=True,
        )
        assert result.viable is False
        assert result.suggest_docker is False

    def test_no_files_passes_syntax(self):
        """If there are zero source files, syntax check passes."""
        ingestion = _make_ingestion([], files_analyzed=0, files_failed=0)
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "python")
        assert result.syntax_rate == 1.0

    def test_multiple_failures_combine_reasons(self):
        deps = [
            _make_dep("a", missing=True),
            _make_dep("b", missing=True),
            _make_dep("c", missing=True),
        ]
        ingestion = _make_ingestion(deps, files_analyzed=1, files_failed=5)
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is False
        # Both install and syntax reasons should appear
        assert "dependencies installed" in result.reason
        assert "parsed" in result.reason

    def test_stack_context_populated(self):
        """ViabilityResult includes language and declared deps for corpus."""
        deps = [_make_dep("flask", required_version=">=2.0")]
        ingestion = _make_ingestion(deps)
        session = _make_session({"flask": True})

        result = run_viability_gate(
            session, ingestion, "python", framework="web_backend",
        )
        assert result.language == "python"
        assert result.framework == "web_backend"
        assert len(result.declared_deps) == 1
        assert result.declared_deps[0]["name"] == "flask"
        assert result.declared_deps[0]["version"] == ">=2.0"


# ── Test: ViabilityResult Serialisation ──


class TestViabilityResultSerialisation:
    """Test JSON serialisation of ViabilityResult."""

    def test_as_dict_roundtrip(self):
        vr = ViabilityResult(
            viable=False,
            install_rate=0.4,
            import_rate=0.75,
            syntax_rate=1.0,
            missing_deps=["pandas", "numpy"],
            unimportable_deps=["scipy"],
            reason="Only 2 of 5 dependencies installed (40%).",
            suggest_docker=True,
            language="python",
            framework="data_science",
            declared_deps=[
                {"name": "flask", "version": "2.0.0"},
                {"name": "pandas", "version": ""},
            ],
        )
        d = vr.as_dict()
        assert d["viable"] is False
        assert d["install_rate"] == 0.4
        assert d["missing_deps"] == ["pandas", "numpy"]
        assert d["language"] == "python"
        # Should be JSON-serialisable
        json_str = json.dumps(d)
        assert json.loads(json_str) == d


# ── Test: Baseline Failed Report ──


class TestBaselineFailedReport:
    """Test the plain-text baseline-failed report."""

    def test_report_has_header(self):
        vr = ViabilityResult(
            viable=False, install_rate=0.3, import_rate=1.0,
            syntax_rate=1.0, missing_deps=["pandas", "numpy"],
        )
        deps = [_make_dep("flask"), _make_dep("pandas", missing=True),
                _make_dep("numpy", missing=True)]
        ingestion = _make_ingestion(deps)
        text = build_baseline_failed_text(vr, ingestion, "My Project")
        assert "myCode Baseline Viability Report" in text
        assert "My Project" in text

    def test_report_shows_missing_deps(self):
        vr = ViabilityResult(
            viable=False, install_rate=0.33, import_rate=1.0,
            syntax_rate=1.0, missing_deps=["pandas", "numpy"],
        )
        deps = [_make_dep("flask"), _make_dep("pandas", missing=True),
                _make_dep("numpy", missing=True)]
        ingestion = _make_ingestion(deps)
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "pandas" in text
        assert "numpy" in text
        assert "1 of 3 dependencies installed" in text

    def test_report_shows_import_failures(self):
        vr = ViabilityResult(
            viable=False, install_rate=1.0, import_rate=0.25,
            syntax_rate=1.0, unimportable_deps=["scipy", "numpy"],
        )
        ingestion = _make_ingestion([_make_dep("flask")])
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "Import Failures" in text
        assert "scipy" in text
        assert "native library" in text

    def test_report_shows_syntax_issues(self):
        vr = ViabilityResult(
            viable=False, install_rate=1.0, import_rate=1.0,
            syntax_rate=0.2,
        )
        ingestion = _make_ingestion([], files_analyzed=2, files_failed=8)
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "Source File Parsing" in text
        assert "2 of 10" in text

    def test_report_shows_docker_suggestion(self):
        vr = ViabilityResult(
            viable=False, install_rate=0.3, import_rate=1.0,
            syntax_rate=1.0, suggest_docker=True,
            missing_deps=["a"],
        )
        ingestion = _make_ingestion([_make_dep("a", missing=True)])
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "--containerised" in text

    def test_report_no_docker_when_not_suggested(self):
        vr = ViabilityResult(
            viable=False, install_rate=0.3, import_rate=1.0,
            syntax_rate=1.0, suggest_docker=False,
            missing_deps=["a"],
        )
        ingestion = _make_ingestion([_make_dep("a", missing=True)])
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "--containerised" not in text

    def test_report_has_next_steps(self):
        vr = ViabilityResult(
            viable=False, install_rate=0.3, import_rate=1.0,
            syntax_rate=1.0, missing_deps=["a"],
        )
        ingestion = _make_ingestion([_make_dep("a", missing=True)])
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "Next Steps" in text
        assert "Re-run myCode" in text

    def test_report_uses_ascii_dividers_not_markdown(self):
        """Report uses ---- dividers, not # headers."""
        vr = ViabilityResult(
            viable=False, install_rate=0.3, import_rate=1.0,
            syntax_rate=1.0, missing_deps=["a"],
        )
        ingestion = _make_ingestion([_make_dep("a", missing=True)])
        text = build_baseline_failed_text(vr, ingestion, "Test")
        assert "----" in text
        assert "====" in text
        # Should not contain markdown-style headers
        lines = text.split("\n")
        for line in lines:
            assert not line.startswith("# "), f"Found markdown header: {line}"
            assert not line.startswith("## "), f"Found markdown header: {line}"


# ── Test: JS/TS Viability Gate ──


class TestIsJsProject:
    """Test JS/TS project detection."""

    def test_detects_package_json_in_file_analyses(self):
        ingestion = _make_ingestion(file_analyses=[
            FileAnalysis(file_path="package.json"),
        ])
        assert _is_js_project(ingestion) is True

    def test_detects_scoped_npm_packages(self):
        deps = [_make_dep("@vercel/analytics"), _make_dep("react")]
        ingestion = _make_ingestion(deps)
        assert _is_js_project(ingestion) is True

    def test_python_project_not_detected(self):
        deps = [_make_dep("flask"), _make_dep("requests")]
        ingestion = _make_ingestion(deps)
        assert _is_js_project(ingestion) is False

    def test_empty_project_not_detected(self):
        ingestion = _make_ingestion([])
        assert _is_js_project(ingestion) is False


class TestJsViabilityGate:
    """Test JS/TS viability gate — no Python import checking."""

    def test_passes_with_scoped_npm_packages(self):
        """20 scoped npm packages + 2 recognised should pass."""
        deps = [
            _make_dep("@vercel/analytics"),
            _make_dep("@vercel/og"),
            _make_dep("@t3-oss/env-nextjs"),
            _make_dep("@tanstack/react-query"),
            _make_dep("@radix-ui/react-dialog"),
            _make_dep("@radix-ui/react-dropdown-menu"),
            _make_dep("@radix-ui/react-slot"),
            _make_dep("@radix-ui/react-tabs"),
            _make_dep("@hookform/resolvers"),
            _make_dep("@auth/prisma-adapter"),
            _make_dep("@prisma/client"),
            _make_dep("@trpc/client"),
            _make_dep("@trpc/server"),
            _make_dep("@trpc/react-query"),
            _make_dep("@types/node", is_dev=True),
            _make_dep("@testing-library/react", is_dev=True),
            _make_dep("@typescript-eslint/parser", is_dev=True),
            _make_dep("@tailwindcss/forms"),
            _make_dep("@emotion/react"),
            _make_dep("@emotion/styled"),
            # 2 recognised (non-scoped)
            _make_dep("next"),
            _make_dep("react"),
        ]
        ingestion = _make_ingestion(deps, files_analyzed=30, files_failed=0,
                                    file_analyses=[FileAnalysis(file_path="package.json")])
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "javascript")
        assert result.viable is True
        assert result.install_rate == 1.0
        assert result.import_rate >= 0.8
        assert result.language == "javascript"

    def test_passes_without_node_modules(self):
        """Should pass even without node_modules (npm install runs later)."""
        deps = [_make_dep("next"), _make_dep("react"), _make_dep("@vercel/og")]
        ingestion = _make_ingestion(deps, files_analyzed=5)
        session = MagicMock()
        session.project_copy_dir = Path("/nonexistent/path")

        result = run_viability_gate(session, ingestion, "javascript")
        assert result.viable is True

    def test_fails_with_zero_dependencies(self):
        """package.json with no dependencies should fail."""
        ingestion = _make_ingestion([], files_analyzed=5,
                                    file_analyses=[FileAnalysis(file_path="package.json")])
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "javascript")
        assert result.viable is False
        assert "no dependencies" in result.reason

    def test_fails_with_low_syntax_rate(self):
        """JS gate still checks syntax rate."""
        deps = [_make_dep("next"), _make_dep("react")]
        ingestion = _make_ingestion(deps, files_analyzed=1, files_failed=10)
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "javascript")
        assert result.viable is False
        assert "parsed" in result.reason

    def test_does_not_run_python_imports(self):
        """JS gate should NOT try to import packages in Python."""
        deps = [_make_dep("@vercel/og"), _make_dep("next")]
        ingestion = _make_ingestion(deps, files_analyzed=5)
        session = MagicMock()

        run_viability_gate(session, ingestion, "javascript")
        # run_in_session should never be called (no import check)
        session.run_in_session.assert_not_called()

    def test_auto_detects_js_from_scoped_deps(self):
        """Even if language='python', scoped deps trigger JS gate."""
        deps = [_make_dep("@vercel/og"), _make_dep("next")]
        ingestion = _make_ingestion(deps, files_analyzed=5)
        session = MagicMock()

        result = run_viability_gate(session, ingestion, "python")
        # Should use JS gate (not try Python imports on npm packages)
        assert result.viable is True
        session.run_in_session.assert_not_called()

    def test_python_project_unchanged(self):
        """Python projects still use Python import-based gate."""
        deps = [_make_dep("flask"), _make_dep("requests")]
        ingestion = _make_ingestion(deps, files_analyzed=10)
        session = _make_session({"flask": True, "requests": True})

        result = run_viability_gate(session, ingestion, "python")
        assert result.viable is True
        # Should have called run_in_session for import check
        session.run_in_session.assert_called_once()


# ── Test: Pipeline Integration ──


class TestPipelineIntegration:
    """Test viability gate integration with the pipeline."""

    def test_pipeline_stops_on_gate_failure(self):
        """Pipeline returns early with baseline_failed report when gate fails."""
        from mycode.pipeline import PipelineConfig, run_pipeline

        # Create a minimal project that will fail the viability gate.
        # We mock the session to simulate massive dep failure.
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python project with many deps that will all be "missing"
            proj = Path(tmpdir) / "project"
            proj.mkdir()
            (proj / "requirements.txt").write_text(
                "nonexistent_pkg_aaa\n"
                "nonexistent_pkg_bbb\n"
                "nonexistent_pkg_ccc\n"
                "nonexistent_pkg_ddd\n"
            )
            (proj / "app.py").write_text("print('hello')\n")

            config = PipelineConfig(
                project_path=proj,
                language="python",
                operational_intent="test",
                offline=True,
                skip_version_check=True,
                auto_approve_scenarios=True,
                temp_base=Path(tmpdir) / "sessions",
            )

            result = run_pipeline(config)

            # The gate should have failed (all 4 deps missing = 0% install rate)
            assert result.viability is not None
            assert result.viability.viable is False
            assert result.viability.install_rate == 0.0

            # Should have a baseline_failed report
            assert result.report is not None
            assert result.report.baseline_failed is True

            # Should NOT have reached execution
            stage_names = [s.stage for s in result.stages]
            assert "execution" not in stage_names
            assert "viability_gate" in stage_names

    def test_pipeline_continues_on_gate_pass(self):
        """Pipeline proceeds past viability gate when environment is healthy."""
        from mycode.pipeline import PipelineConfig, run_pipeline

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a project with no deps — should pass gate
            proj = Path(tmpdir) / "project"
            proj.mkdir()
            (proj / "app.py").write_text("x = 1 + 1\n")

            config = PipelineConfig(
                project_path=proj,
                language="python",
                operational_intent="test",
                offline=True,
                skip_version_check=True,
                auto_approve_scenarios=True,
                temp_base=Path(tmpdir) / "sessions",
            )

            result = run_pipeline(config)

            # Gate should pass (no deps = 100% install rate)
            assert result.viability is not None
            assert result.viability.viable is True

            # Should have reached execution stage
            stage_names = [s.stage for s in result.stages]
            assert "viability_gate" in stage_names
            assert "execution" in stage_names

            # Should NOT have baseline_failed
            if result.report:
                assert result.report.baseline_failed is False


# ── Test: CLI Exit Code ──


class TestCLIExitCode:
    """Test that CLI returns exit code 2 for baseline failure."""

    def test_exit_code_2_on_baseline_failure(self):
        from mycode.cli import main

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Path(tmpdir) / "project"
            proj.mkdir()
            (proj / "requirements.txt").write_text(
                "nonexistent_pkg_aaa\n"
                "nonexistent_pkg_bbb\n"
                "nonexistent_pkg_ccc\n"
            )
            (proj / "app.py").write_text("print('hello')\n")

            exit_code = main([
                str(proj),
                "--offline",
                "--non-interactive",
                "--skip-version-check",
                "--yes",
            ])

            assert exit_code == 2
