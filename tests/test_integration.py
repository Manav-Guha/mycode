"""Integration test (C6) — Session Manager + Project Ingester + Component Library.

Points all three components at myCode's own source code as the test project.
Verifies the full pipeline: environment creation, AST parsing, dependency
matching, function flow mapping, and coupling point identification.
"""

import sys
from pathlib import Path
from typing import Optional

import pytest

from mycode.ingester import (
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    IngestionResult,
    ProjectIngester,
)
from mycode.library import ComponentLibrary, ProfileMatch
from mycode.session import SessionManager

# ── Constants ──

# myCode's own source tree
MYCODE_SRC = Path(__file__).resolve().parent.parent / "src" / "mycode"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Fixtures ──


@pytest.fixture(scope="module")
def ingestion_result() -> IngestionResult:
    """Ingest myCode's own source code once for the entire module."""
    ingester = ProjectIngester(
        project_path=MYCODE_SRC,
        installed_packages=None,
        skip_pypi_check=True,
    )
    return ingester.ingest()


@pytest.fixture(scope="module")
def library() -> ComponentLibrary:
    """Create a ComponentLibrary pointing at the real profiles."""
    return ComponentLibrary()


# ── Tests: Session Manager creates venv for myCode ──


class TestSessionManagerWithMyCode:
    """Verify Session Manager can create a venv for myCode's own project."""

    @pytest.mark.slow
    def test_session_creates_venv(self, tmp_path):
        """Session Manager creates a venv and copies myCode source."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            assert session.workspace_dir is not None
            assert session.workspace_dir.exists()
            assert session.venv_dir.exists()
            assert session.project_copy_dir.exists()
            assert session.venv_python.exists()

    @pytest.mark.slow
    def test_session_detects_environment(self, tmp_path):
        """Session Manager detects Python version and installed packages."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            env = session.environment_info
            assert env.python_version
            assert env.python_executable
            assert env.platform_info

    @pytest.mark.slow
    def test_session_copies_source_files(self, tmp_path):
        """Session Manager copies myCode source files into workspace."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            copied_files = list(session.project_copy_dir.rglob("*.py"))
            # myCode has at least: __init__.py, session.py, ingester.py,
            # js_ingester.py, library/__init__.py, library/loader.py,
            # plus placeholder files
            assert len(copied_files) >= 6

            # Key source files should be present
            copied_names = {f.name for f in copied_files}
            assert "session.py" in copied_names
            assert "ingester.py" in copied_names
            assert "loader.py" in copied_names

    @pytest.mark.slow
    def test_session_can_run_python_in_venv(self, tmp_path):
        """Session Manager can execute Python commands inside the venv."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            result = session.run_in_session(
                [str(session.venv_python), "-c", "import sys; print(sys.version)"]
            )
            assert result.returncode == 0
            assert result.stdout.strip()

    @pytest.mark.slow
    def test_session_cleanup(self, tmp_path):
        """Session Manager cleans up workspace on exit."""
        workspace = None
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            workspace = session.workspace_dir
            assert workspace.exists()
        assert not workspace.exists()


# ── Tests: Project Ingester parses myCode's own source ──


class TestIngesterWithMyCode:
    """Verify Project Ingester correctly parses myCode's own Python source."""

    def test_ingestion_succeeds(self, ingestion_result):
        """Ingester produces a valid result with no fatal errors."""
        assert isinstance(ingestion_result, IngestionResult)
        assert ingestion_result.files_analyzed > 0

    def test_all_source_files_discovered(self, ingestion_result):
        """Ingester finds all myCode Python files."""
        analyzed_files = {
            a.file_path for a in ingestion_result.file_analyses
        }
        # Core source files that must be found
        expected_files = {
            "__init__.py",
            "session.py",
            "ingester.py",
            "js_ingester.py",
        }
        for expected in expected_files:
            assert any(
                f.endswith(expected) for f in analyzed_files
            ), f"Missing file: {expected}"

    def test_library_subpackage_discovered(self, ingestion_result):
        """Ingester finds files inside library/ subpackage."""
        analyzed_files = {
            a.file_path for a in ingestion_result.file_analyses
        }
        library_files = [f for f in analyzed_files if "library" in f]
        assert len(library_files) >= 2  # __init__.py + loader.py

    def test_no_parse_errors_on_real_source(self, ingestion_result):
        """myCode's own source should parse cleanly — zero parse errors."""
        assert ingestion_result.files_failed == 0, (
            f"Parse errors: {ingestion_result.parse_errors}"
        )

    def test_total_lines_reasonable(self, ingestion_result):
        """Sanity check: myCode source is a nontrivial codebase."""
        # session.py ~665 lines, ingester.py ~1031 lines, js_ingester ~1127 lines,
        # loader.py ~300 lines, plus others
        assert ingestion_result.total_lines > 2000

    def test_empty_placeholder_files_handled(self, ingestion_result):
        """Empty placeholder files (engine.py, etc.) should parse without error."""
        placeholders = ["engine.py", "interface.py", "recorder.py",
                        "report.py", "scenario.py"]
        for ph in placeholders:
            matches = [
                a for a in ingestion_result.file_analyses
                if a.file_path.endswith(ph)
            ]
            if matches:
                assert matches[0].parse_error is None, (
                    f"Placeholder {ph} should parse cleanly"
                )


# ── Tests: Ingester identifies myCode's internal structure ──


class TestIngesterFunctionFlows:
    """Verify the ingester correctly maps myCode's internal function flows."""

    def test_function_flows_detected(self, ingestion_result):
        """Ingester identifies function-to-function call relationships."""
        assert len(ingestion_result.function_flows) > 0

    def test_session_manager_internal_flows(self, ingestion_result):
        """SessionManager methods calling other SessionManager methods are detected."""
        session_flows = [
            f for f in ingestion_result.function_flows
            if "session" in f.caller.lower() or "session" in f.callee.lower()
        ]
        assert len(session_flows) > 0, "Should detect flows within session.py"

    def test_ingester_internal_flows(self, ingestion_result):
        """Ingester methods calling other Ingester methods are detected."""
        ingester_flows = [
            f for f in ingestion_result.function_flows
            if "ingester" in f.file_path and "ingester" in f.file_path
        ]
        assert len(ingester_flows) > 0, "Should detect flows within ingester.py"

    def test_cross_module_flows_via_imports(self, ingestion_result):
        """__init__.py imports from session, ingester, library — flows should reflect this."""
        init_flows = [
            f for f in ingestion_result.function_flows
            if f.file_path == "__init__.py"
        ]
        # __init__.py is mostly imports, not function calls, so flows may be
        # minimal there. But the import analysis should show cross-module refs.
        init_analysis = None
        for a in ingestion_result.file_analyses:
            if a.file_path == "__init__.py":
                init_analysis = a
                break
        assert init_analysis is not None
        import_modules = {imp.module for imp in init_analysis.imports}
        assert "mycode.session" in import_modules
        assert "mycode.ingester" in import_modules
        assert "mycode.library" in import_modules

    def test_function_flows_have_valid_structure(self, ingestion_result):
        """Every FunctionFlow has a non-empty caller, callee, and file_path."""
        for flow in ingestion_result.function_flows:
            assert flow.caller, f"Empty caller in flow: {flow}"
            assert flow.callee, f"Empty callee in flow: {flow}"
            assert flow.file_path, f"Empty file_path in flow: {flow}"
            assert flow.lineno > 0, f"Invalid lineno in flow: {flow}"


class TestIngesterCouplingPoints:
    """Verify the ingester identifies coupling points in myCode's architecture."""

    def test_coupling_points_detected(self, ingestion_result):
        """Ingester identifies at least one coupling point in myCode."""
        assert len(ingestion_result.coupling_points) > 0

    def test_coupling_point_types_valid(self, ingestion_result):
        """All coupling points have a recognized coupling type."""
        valid_types = {"high_fan_in", "cross_module_hub", "shared_state"}
        for cp in ingestion_result.coupling_points:
            assert cp.coupling_type in valid_types, (
                f"Unknown coupling type: {cp.coupling_type}"
            )

    def test_coupling_points_have_descriptions(self, ingestion_result):
        """Every coupling point has a human-readable description."""
        for cp in ingestion_result.coupling_points:
            assert cp.description, f"Empty description for coupling point: {cp.source}"
            assert cp.source, f"Empty source in coupling point"
            assert len(cp.targets) > 0, f"No targets for coupling point: {cp.source}"


# ── Tests: Ingester extracts classes and functions ──


class TestIngesterClassesAndFunctions:
    """Verify the ingester finds myCode's known classes and functions."""

    def test_session_manager_class_found(self, ingestion_result):
        """SessionManager class is detected in session.py."""
        session_analysis = _find_analysis(ingestion_result, "session.py")
        assert session_analysis is not None
        class_names = {c.name for c in session_analysis.classes}
        assert "SessionManager" in class_names

    def test_session_manager_methods_found(self, ingestion_result):
        """SessionManager's key methods are detected."""
        session_analysis = _find_analysis(ingestion_result, "session.py")
        assert session_analysis is not None
        method_names = {
            f.name for f in session_analysis.functions
            if f.class_name == "SessionManager"
        }
        expected_methods = {"setup", "teardown", "detect_environment",
                           "run_in_session"}
        missing = expected_methods - method_names
        assert not missing, f"Missing SessionManager methods: {missing}"

    def test_project_ingester_class_found(self, ingestion_result):
        """ProjectIngester class is detected in ingester.py."""
        ingester_analysis = _find_analysis(ingestion_result, "ingester.py")
        assert ingester_analysis is not None
        class_names = {c.name for c in ingester_analysis.classes}
        assert "ProjectIngester" in class_names

    def test_ingester_ingest_method_found(self, ingestion_result):
        """ProjectIngester.ingest() method is detected."""
        ingester_analysis = _find_analysis(ingestion_result, "ingester.py")
        assert ingester_analysis is not None
        method_names = {
            f.name for f in ingester_analysis.functions
            if f.class_name == "ProjectIngester"
        }
        assert "ingest" in method_names

    def test_component_library_class_found(self, ingestion_result):
        """ComponentLibrary class is detected in library/loader.py."""
        loader_analysis = _find_analysis(ingestion_result, "loader.py")
        assert loader_analysis is not None
        class_names = {c.name for c in loader_analysis.classes}
        assert "ComponentLibrary" in class_names

    def test_dataclasses_detected(self, ingestion_result):
        """Dataclasses (ResourceCaps, EnvironmentInfo, etc.) are found."""
        session_analysis = _find_analysis(ingestion_result, "session.py")
        assert session_analysis is not None
        class_names = {c.name for c in session_analysis.classes}
        expected_dataclasses = {"ResourceCaps", "EnvironmentInfo", "SessionResult"}
        missing = expected_dataclasses - class_names
        assert not missing, f"Missing dataclasses: {missing}"

    def test_ingester_dataclasses_detected(self, ingestion_result):
        """Ingester dataclasses are found."""
        ingester_analysis = _find_analysis(ingestion_result, "ingester.py")
        assert ingester_analysis is not None
        class_names = {c.name for c in ingester_analysis.classes}
        expected = {"FileAnalysis", "DependencyInfo", "FunctionFlow",
                    "CouplingPoint", "IngestionResult"}
        missing = expected - class_names
        assert not missing, f"Missing ingester dataclasses: {missing}"


# ── Tests: Ingester extracts imports correctly ──


class TestIngesterImports:
    """Verify import extraction from myCode's own source."""

    def test_session_stdlib_imports(self, ingestion_result):
        """session.py imports standard library modules."""
        session_analysis = _find_analysis(ingestion_result, "session.py")
        assert session_analysis is not None
        imported_modules = {imp.module for imp in session_analysis.imports}
        # session.py uses: json, logging, os, shutil, signal, subprocess, etc.
        expected_stdlib = {"json", "logging", "os", "shutil", "signal",
                          "subprocess", "sys", "tempfile", "venv"}
        found = expected_stdlib & imported_modules
        assert len(found) >= 5, (
            f"Expected at least 5 stdlib imports, found: {found}"
        )

    def test_ingester_imports_ast(self, ingestion_result):
        """ingester.py imports the ast module."""
        ingester_analysis = _find_analysis(ingestion_result, "ingester.py")
        assert ingester_analysis is not None
        imported_modules = {imp.module for imp in ingester_analysis.imports}
        assert "ast" in imported_modules

    def test_loader_imports(self, ingestion_result):
        """library/loader.py imports json and pathlib."""
        loader_analysis = _find_analysis(ingestion_result, "loader.py")
        assert loader_analysis is not None
        imported_modules = {imp.module for imp in loader_analysis.imports}
        assert "json" in imported_modules
        assert "pathlib" in imported_modules


# ── Tests: Component Library matches myCode's dependencies ──


class TestComponentLibraryMatching:
    """Verify Component Library matches dependencies found by the ingester."""

    def test_match_ingester_dependencies(self, ingestion_result, library):
        """Match ingester-extracted dependencies against the component library."""
        if not ingestion_result.dependencies:
            pytest.skip("No declared dependencies found (myCode has no deps in pyproject.toml)")

        dep_dicts = [
            {"name": d.name, "installed_version": d.installed_version}
            for d in ingestion_result.dependencies
        ]
        matches = library.match_dependencies("python", dep_dicts)
        assert isinstance(matches, list)
        # Every dependency should produce a match object (recognized or not)
        assert len(matches) == len(dep_dicts)

    def test_match_mycode_stdlib_imports(self, ingestion_result, library):
        """Test matching stdlib imports that myCode uses against profiles."""
        # Collect all stdlib-like imports across myCode's source
        all_imports = set()
        for analysis in ingestion_result.file_analyses:
            for imp in analysis.imports:
                top_level = imp.module.split(".")[0] if imp.module else ""
                if top_level:
                    all_imports.add(top_level)

        # myCode uses os, pathlib, json, etc. — some have profiles
        dep_dicts = [{"name": name} for name in sorted(all_imports)]
        matches = library.match_dependencies("python", dep_dicts)

        # At least some should match (e.g., "os" -> os_pathlib profile)
        recognized = library.get_recognized(matches)
        recognized_names = {m.profile.name for m in recognized}
        # os and pathlib should resolve to os_pathlib
        assert "os_pathlib" in recognized_names, (
            f"Expected os_pathlib profile to match. Recognized: {recognized_names}"
        )

    def test_match_hypothetical_project_deps(self, library):
        """Simulate what happens when myCode matches a typical user project."""
        typical_deps = [
            {"name": "flask", "installed_version": "3.1.0"},
            {"name": "sqlalchemy", "installed_version": "2.0.30"},
            {"name": "requests", "installed_version": "2.32.3"},
            {"name": "pandas", "installed_version": "2.2.3"},
            {"name": "some-unknown-dep"},
        ]
        matches = library.match_dependencies("python", typical_deps)
        recognized = library.get_recognized(matches)
        unrecognized = library.get_unrecognized(matches)

        assert len(recognized) == 4
        assert unrecognized == ["some-unknown-dep"]

        # Verify version checking works
        flask_match = next(m for m in recognized if m.profile.name == "flask")
        assert flask_match.version_match is True


# ── Tests: Full Pipeline C1 + C2 + C4 ──


class TestFullPipeline:
    """End-to-end integration: Session Manager -> Ingester -> Component Library."""

    @pytest.mark.slow
    def test_full_pipeline(self, tmp_path):
        """Chain all three components: create venv, ingest, match dependencies."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            # C1: Session provides environment info
            env = session.environment_info
            assert env.python_version

            # C2: Ingester parses the project copy inside the session workspace
            ingester = ProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=env.installed_packages,
                skip_pypi_check=True,
            )
            result = ingester.ingest()

            # Verify ingestion succeeded on the copied source
            assert result.files_analyzed > 0
            assert result.files_failed == 0

            # C4: Component Library matches dependencies
            lib = ComponentLibrary()
            dep_dicts = [
                {"name": d.name, "installed_version": d.installed_version}
                for d in result.dependencies
            ]
            if dep_dicts:
                matches = lib.match_dependencies("python", dep_dicts)
                assert len(matches) == len(dep_dicts)

            # Verify structural analysis still works on the copied files
            assert result.total_lines > 0
            assert len(result.function_flows) > 0

    @pytest.mark.slow
    def test_pipeline_ingester_uses_session_packages(self, tmp_path):
        """Ingester uses the Session Manager's installed_packages for version detection."""
        with SessionManager(MYCODE_SRC, temp_base=tmp_path) as session:
            env = session.environment_info
            # Feed session's package info to ingester
            ingester = ProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=env.installed_packages,
                skip_pypi_check=True,
            )
            result = ingester.ingest()

            # If there are declared deps, versions should be enriched from session env
            for dep in result.dependencies:
                if dep.installed_version:
                    # Version should be a real version string, not empty
                    assert "." in dep.installed_version, (
                        f"Unexpected version format for {dep.name}: {dep.installed_version}"
                    )


# ── Tests: Partial Parsing Edge Cases ──


class TestPartialParsing:
    """Verify myCode's own source doesn't trigger any partial parsing edge cases."""

    def test_no_warnings_on_clean_source(self, ingestion_result):
        """myCode's own clean source should produce minimal warnings."""
        # The only acceptable warning is about zero declared dependencies
        for w in ingestion_result.warnings:
            assert "couldn't be parsed" not in w.lower(), (
                f"Unexpected parse warning: {w}"
            )

    def test_all_files_have_line_counts(self, ingestion_result):
        """Every analyzed file should have a positive line count (except empty placeholders)."""
        for analysis in ingestion_result.file_analyses:
            if analysis.parse_error:
                continue
            # Empty placeholder files have 0 lines — that's ok
            assert analysis.lines_of_code >= 0, (
                f"{analysis.file_path} has negative line count"
            )

    def test_no_syntax_errors_in_own_source(self, ingestion_result):
        """myCode's own source should have zero syntax errors."""
        syntax_errors = [
            e for e in ingestion_result.parse_errors
            if "syntax" in e.get("error", "").lower()
        ]
        assert len(syntax_errors) == 0, (
            f"Syntax errors in myCode's own source: {syntax_errors}"
        )

    def test_file_analysis_complete_for_substantive_files(self, ingestion_result):
        """Non-empty source files should have functions, classes, or imports."""
        substantive_files = ["session.py", "ingester.py", "js_ingester.py",
                            "loader.py"]
        for filename in substantive_files:
            analysis = _find_analysis(ingestion_result, filename)
            assert analysis is not None, f"Missing analysis for {filename}"
            assert analysis.parse_error is None, (
                f"{filename} parse error: {analysis.parse_error}"
            )
            total_entities = (len(analysis.functions) + len(analysis.classes) +
                            len(analysis.imports))
            assert total_entities > 0, (
                f"{filename} has no functions, classes, or imports"
            )


# ── Helpers ──


def _find_analysis(result: IngestionResult, filename: str) -> Optional[FileAnalysis]:
    """Find a FileAnalysis by trailing filename."""
    for a in result.file_analyses:
        if a.file_path.endswith(filename):
            return a
    return None
