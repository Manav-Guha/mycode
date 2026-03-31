"""Tests for Session Manager (C1)."""

import json
import os
import platform
import signal
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

from mycode.session import (
    COPY_EXCLUDE_PATTERNS,
    COPY_EXCLUDE_SUFFIXES,
    DependencyInstallError,
    EnvironmentInfo,
    ResourceCaps,
    SessionError,
    SessionManager,
    SessionResult,
    VenvCreationError,
    _is_process_running,
    _IS_WINDOWS,
)


# ── Fixtures ──


@pytest.fixture
def sample_project(tmp_path):
    """Create a minimal sample project for testing."""
    project = tmp_path / "sample_project"
    project.mkdir()
    (project / "app.py").write_text(
        "def hello():\n    return 'Hello, World!'\n"
    )
    (project / "requirements.txt").write_text("# empty for fast tests\n")
    return project


@pytest.fixture
def project_with_exclusions(tmp_path):
    """Create a project with directories/files that should be excluded from copy."""
    project = tmp_path / "project_excl"
    project.mkdir()
    (project / "main.py").write_text("print('main')\n")
    (project / "util.py").write_text("print('util')\n")

    # Directories that should be excluded
    (project / ".git").mkdir()
    (project / "__pycache__").mkdir()
    (project / "node_modules").mkdir()
    (project / "node_modules" / "pkg").mkdir()
    (project / ".venv").mkdir()
    (project / ".env").write_text("SECRET=123")

    # Files that should be excluded by suffix
    (project / "old.pyc").write_bytes(b"\x00")

    # Subdirectory that should be kept
    (project / "src").mkdir()
    (project / "src" / "lib.py").write_text("x = 1\n")
    return project


@pytest.fixture(autouse=True)
def _reset_session_class_state():
    """Ensure class-level state is clean before and after each test."""
    SessionManager._active_sessions = []
    SessionManager._signal_handlers_installed = False
    SessionManager._original_sigint = None
    SessionManager._original_sigterm = None
    yield
    # Restore default signal handlers if tests left them modified
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if not _IS_WINDOWS:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    SessionManager._active_sessions = []
    SessionManager._signal_handlers_installed = False
    SessionManager._original_sigint = None
    SessionManager._original_sigterm = None


# ── Dataclass Tests ──


class TestResourceCaps:
    def test_defaults(self):
        caps = ResourceCaps()
        assert caps.memory_mb == 512
        assert caps.process_limit == 50
        assert caps.timeout_seconds == 300

    def test_custom_values(self):
        caps = ResourceCaps(memory_mb=1024, process_limit=100, timeout_seconds=600)
        assert caps.memory_mb == 1024
        assert caps.process_limit == 100
        assert caps.timeout_seconds == 600


class TestEnvironmentInfo:
    def test_defaults(self):
        info = EnvironmentInfo()
        assert info.python_version == ""
        assert info.python_executable == ""
        assert info.platform_info == ""
        assert info.installed_packages == {}
        assert info.dependency_files == {}


class TestSessionResult:
    def test_normal_result(self):
        result = SessionResult(returncode=0, stdout="ok", stderr="")
        assert result.returncode == 0
        assert result.timed_out is False

    def test_timeout_result(self):
        result = SessionResult(returncode=-1, stdout="", stderr="", timed_out=True)
        assert result.timed_out is True


# ── Initialization Tests ──


class TestSessionManagerInit:
    def test_valid_project_path(self, sample_project):
        sm = SessionManager(sample_project)
        assert sm.project_path == sample_project.resolve()
        assert sm._setup_complete is False
        assert sm._cleaned_up is False

    def test_nonexistent_path_raises(self):
        with pytest.raises(SessionError, match="does not exist"):
            SessionManager("/nonexistent/path/to/project")

    def test_file_path_raises(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(SessionError, match="not a directory"):
            SessionManager(f)

    def test_custom_resource_caps(self, sample_project):
        caps = ResourceCaps(memory_mb=1024)
        sm = SessionManager(sample_project, resource_caps=caps)
        assert sm.resource_caps.memory_mb == 1024

    def test_default_resource_caps(self, sample_project):
        sm = SessionManager(sample_project)
        assert sm.resource_caps.memory_mb == 512

    def test_custom_temp_base(self, sample_project, tmp_path):
        base = tmp_path / "my_temp"
        sm = SessionManager(sample_project, temp_base=base)
        assert sm.temp_base == base


# ── Environment Detection Tests ──


class TestDetectEnvironment:
    def test_python_version_detected(self, sample_project):
        sm = SessionManager(sample_project)
        info = sm.detect_environment()
        assert info.python_version == platform.python_version()

    def test_python_executable_detected(self, sample_project):
        sm = SessionManager(sample_project)
        info = sm.detect_environment()
        assert info.python_executable == sys.executable

    def test_platform_info_detected(self, sample_project):
        sm = SessionManager(sample_project)
        info = sm.detect_environment()
        assert info.platform_info == platform.platform()

    def test_finds_requirements_txt(self, sample_project):
        sm = SessionManager(sample_project)
        info = sm.detect_environment()
        assert "requirements.txt" in info.dependency_files

    def test_finds_no_absent_files(self, tmp_path):
        project = tmp_path / "bare"
        project.mkdir()
        sm = SessionManager(project)
        info = sm.detect_environment()
        assert info.dependency_files == {}

    def test_finds_multiple_dep_files(self, tmp_path):
        project = tmp_path / "multi"
        project.mkdir()
        (project / "requirements.txt").write_text("")
        (project / "pyproject.toml").write_text("")
        (project / "package.json").write_text("{}")
        sm = SessionManager(project)
        info = sm.detect_environment()
        assert "requirements.txt" in info.dependency_files
        assert "pyproject.toml" in info.dependency_files
        assert "package.json" in info.dependency_files

    def test_installed_packages_is_dict(self, sample_project):
        sm = SessionManager(sample_project)
        info = sm.detect_environment()
        assert isinstance(info.installed_packages, dict)

    def test_handles_pip_failure_gracefully(self, sample_project):
        sm = SessionManager(sample_project)
        with mock.patch("mycode.session.subprocess.run", side_effect=OSError("no pip")):
            packages = sm._get_installed_packages()
        assert packages == {}


# ── Project Copy Tests ──


class TestProjectCopy:
    def test_copies_source_files(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.environment_info = EnvironmentInfo()
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert (sm.project_copy_dir / "main.py").exists()
        assert (sm.project_copy_dir / "util.py").exists()
        assert (sm.project_copy_dir / "src" / "lib.py").exists()

    def test_excludes_git_directory(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / ".git").exists()

    def test_excludes_pycache(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / "__pycache__").exists()

    def test_excludes_node_modules(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / "node_modules").exists()

    def test_excludes_venv_dir(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / ".venv").exists()

    def test_excludes_env_file(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / ".env").exists()

    def test_excludes_pyc_files(self, project_with_exclusions, tmp_path):
        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        assert not (sm.project_copy_dir / "old.pyc").exists()

    def test_original_files_untouched(self, project_with_exclusions, tmp_path):
        original_content = (project_with_exclusions / "main.py").read_text()

        sm = SessionManager(project_with_exclusions, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.project_copy_dir = sm.workspace_dir / "project"

        sm._copy_project()

        # Modify the copy
        (sm.project_copy_dir / "main.py").write_text("modified!")

        # Original is unchanged
        assert (project_with_exclusions / "main.py").read_text() == original_content


# ── Python Dependency Installation Tests ──


class TestPyDependencyInstallation:
    """Test _install_dependencies for Python projects."""

    def _make_session(self, tmp_path, project):
        """Create a SessionManager with mocked venv_python and environment_info."""
        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project
        sm.venv_python = Path("/fake/venv/bin/python")
        sm.environment_info = EnvironmentInfo()
        return sm

    def test_requirement_singular_txt_installed(self, tmp_path):
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirement.txt").write_text("flask==3.0.0\n")

        sm = self._make_session(tmp_path, project)

        with mock.patch.object(sm, "_pip_install") as mock_pip:
            sm._install_dependencies()

        mock_pip.assert_called_once_with(
            ["-r", str(project / "requirement.txt")], deadline=mock.ANY,
        )

    def test_requirements_txt_installed(self, tmp_path):
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==3.0.0\n")

        sm = self._make_session(tmp_path, project)

        with mock.patch.object(sm, "_pip_install") as mock_pip:
            sm._install_dependencies()

        mock_pip.assert_called_once_with(
            ["-r", str(project / "requirements.txt")], deadline=mock.ANY,
        )

    def test_pip_failure_non_fatal(self, tmp_path):
        """pip install failure for requirements.txt should not kill the session."""
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("badpkg==0.0.0\n")

        sm = self._make_session(tmp_path, project)

        call_count = 0

        def mock_pip_install(args, deadline=None):
            nonlocal call_count
            call_count += 1
            if args[0] == "-r":
                raise DependencyInstallError("pip install failed")
            # Individual install also fails
            raise DependencyInstallError("pip install failed")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip_install):
            # Should NOT raise
            sm._install_dependencies()

        # Bulk install was attempted, then individual fallback cascade was attempted
        # (3 strategies per package for the fallback cascade)
        assert call_count >= 2

    def test_pip_uses_venv_python(self, tmp_path):
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==3.0.0\n")

        sm = self._make_session(tmp_path, project)
        sm.venv_python = Path("/test/venv/bin/python")

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_dependencies()

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/test/venv/bin/python"
        assert cmd[1:4] == ["-m", "pip", "install"]

    def test_no_dep_files_falls_back(self, tmp_path):
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "app.py").write_text("pass")

        sm = self._make_session(tmp_path, project)
        sm.environment_info = EnvironmentInfo(
            installed_packages={"requests": "2.31.0"}
        )

        with mock.patch.object(sm, "_install_from_package_list") as mock_fallback:
            sm._install_dependencies()

        mock_fallback.assert_called_once_with(
            {"requests": "2.31.0"}, deadline=mock.ANY,
        )

    def test_requirement_txt_in_find_dependency_files(self, tmp_path):
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirement.txt").write_text("flask==3.0.0\n")

        sm = SessionManager(project)
        info = sm.detect_environment()
        assert "requirement.txt" in info.dependency_files

    def test_individual_fallback_installs_each_package(self, tmp_path):
        """When bulk install fails, each package is tried via fallback cascade."""
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==3.0.0\nrequests==2.31.0\n")

        sm = self._make_session(tmp_path, project)

        calls = []

        def mock_pip_install(args, deadline=None):
            calls.append(args)
            if args[0] == "-r":
                raise DependencyInstallError("bulk failed")
            # Individual installs succeed on strategy 1

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip_install):
            sm._install_dependencies()

        # First call: bulk -r, then individual: flask, requests
        assert calls[0][0] == "-r"
        individual_args = [c[0] for c in calls[1:]]
        assert "flask==3.0.0" in individual_args
        assert "requests==2.31.0" in individual_args
        # Both should be recorded as "installed"
        assert sm.dep_install_results.get("flask==3.0.0") == "installed"
        assert sm.dep_install_results.get("requests==2.31.0") == "installed"

    def test_both_requirements_and_requirement_txt(self, tmp_path):
        """Both requirements.txt and requirement.txt present — both installed."""
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==3.0.0\n")
        (project / "requirement.txt").write_text("requests==2.31.0\n")

        sm = self._make_session(tmp_path, project)

        calls = []

        def mock_pip_install(args, deadline=None):
            calls.append(args)

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip_install):
            sm._install_dependencies()

        req_files_installed = [c for c in calls if c[0] == "-r"]
        assert len(req_files_installed) == 2


# ── Pyproject.toml Dependency Parsing Tests ──


class TestParsePyprojectDeps:
    """Test _parse_pyproject_deps static method."""

    def test_basic_deps(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text(
            '[project]\ndependencies = [\n'
            '  "flask>=3.0.0",\n'
            '  "requests==2.31.0",\n'
            ']\n'
        )
        result = SessionManager._parse_pyproject_deps(pp)
        assert [bare for bare, _ in result] == ["flask", "requests"]
        assert [spec for _, spec in result] == ["flask>=3.0.0", "requests==2.31.0"]

    def test_extras_and_complex_specifiers(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text(
            '[project]\ndependencies = [\n'
            '  "fastapi[standard]>=0.135.0,<0.136.0",\n'
            '  "pydantic[email]~=2.0",\n'
            '  "sqlmodel>=0.0.22,<0.1.0",\n'
            ']\n'
        )
        result = SessionManager._parse_pyproject_deps(pp)
        names = [bare for bare, _ in result]
        specs = [spec for _, spec in result]
        assert names == ["fastapi", "pydantic", "sqlmodel"]
        # Full specs preserved for pip
        assert specs[0] == "fastapi[standard]>=0.135.0,<0.136.0"
        assert specs[1] == "pydantic[email]~=2.0"

    def test_no_project_section(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text('[tool.setuptools]\npackages = ["myapp"]\n')
        assert SessionManager._parse_pyproject_deps(pp) == []

    def test_no_dependencies_key(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text('[project]\nname = "myapp"\n')
        assert SessionManager._parse_pyproject_deps(pp) == []

    def test_invalid_toml(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text("this is not valid toml {{{")
        assert SessionManager._parse_pyproject_deps(pp) == []

    def test_deduplicates(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text(
            '[project]\ndependencies = [\n'
            '  "Flask>=3.0",\n'
            '  "flask>=2.0",\n'
            ']\n'
        )
        result = SessionManager._parse_pyproject_deps(pp)
        assert len(result) == 1
        assert result[0][0] == "flask"


class TestPyprojectFallbackToParsingDeps:
    """Test that pyproject.toml pip-install failure triggers dep parsing."""

    def _make_session(self, tmp_path, project):
        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project
        sm.venv_python = Path("/fake/venv/bin/python")
        sm.environment_info = EnvironmentInfo()
        return sm

    def test_pyproject_failure_installs_parsed_deps(self, tmp_path):
        """When pip install . fails, parsed deps are installed individually."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "demo"\ndependencies = [\n'
            '  "fastapi[standard]>=0.135.0",\n'
            '  "sqlmodel>=0.0.22",\n'
            ']\n'
        )

        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            # pip install . fails (monorepo flat-layout error)
            if args[0] == str(project):
                raise DependencyInstallError("Multiple top-level packages discovered")
            # Individual installs succeed

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            sm._install_dependencies()

        # First call: pip install . (fails)
        assert calls[0] == [str(project)]
        # Then individual deps via fallback cascade (strategy 1)
        individual = [c[0] for c in calls[1:]]
        assert "fastapi[standard]>=0.135.0" in individual
        assert "sqlmodel>=0.0.22" in individual

    def test_pyproject_failure_does_not_fall_through(self, tmp_path):
        """When pip install . fails and deps are parsed, don't use env fallback."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "demo"\ndependencies = ["requests>=2.0"]\n'
        )

        sm = self._make_session(tmp_path, project)
        sm.environment_info = EnvironmentInfo(
            installed_packages={"stale-pkg": "1.0.0"}
        )

        def mock_pip(args, deadline=None):
            if args[0] == str(project):
                raise DependencyInstallError("flat-layout error")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            with mock.patch.object(sm, "_install_from_package_list") as mock_env:
                sm._install_dependencies()

        # Environment fallback should NOT be called
        mock_env.assert_not_called()

    def test_pyproject_no_deps_falls_through(self, tmp_path):
        """When pyproject.toml has no [project.dependencies], fall through."""
        project = tmp_path / "proj"
        project.mkdir()
        # Poetry-style: no [project.dependencies]
        (project / "pyproject.toml").write_text(
            '[tool.poetry]\nname = "demo"\n'
            '[tool.poetry.dependencies]\npython = "^3.11"\n'
        )

        sm = self._make_session(tmp_path, project)
        sm.environment_info = EnvironmentInfo(
            installed_packages={"fallback-pkg": "1.0.0"}
        )

        def mock_pip(args, deadline=None):
            if args[0] == str(project):
                raise DependencyInstallError("no setup.py or pyproject.toml")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            with mock.patch.object(sm, "_install_from_package_list") as mock_env:
                sm._install_dependencies()

        # No parsed deps → falls through to environment fallback
        mock_env.assert_called_once()


# ── Pip Install Fallback Cascade Tests ──


class TestPipInstallFallbackCascade:
    """Test _pip_install_with_fallback three-strategy cascade."""

    def _make_session(self, tmp_path, project):
        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project
        sm.venv_python = Path("/fake/venv/bin/python")
        sm.environment_info = EnvironmentInfo()
        return sm

    def test_strategy1_succeeds_no_fallback(self, tmp_path):
        """Normal install succeeds → no fallback attempted."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("flask==3.0.0")

        assert result == "installed"
        assert sm.dep_install_results["flask==3.0.0"] == "installed"
        assert len(calls) == 1
        assert calls[0] == ["flask==3.0.0"]

    def test_strategy2_after_strategy1_fails(self, tmp_path):
        """Strategy 1 fails → strategy 2 (binary-only) succeeds."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            if "--only-binary=:all:" not in args:
                raise DependencyInstallError("compilation failed")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("cryptography==42.0.0")

        assert result == "installed-binary-only"
        assert sm.dep_install_results["cryptography==42.0.0"] == "installed-binary-only"
        assert len(calls) == 2
        assert "--only-binary=:all:" in calls[1]

    def test_strategy3_after_strategies_1_2_fail(self, tmp_path):
        """Strategies 1+2 fail → strategy 3 (no-deps) succeeds."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            if "--no-deps" not in args:
                raise DependencyInstallError("install failed")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("obscure-pkg==1.0.0")

        assert result == "installed-no-deps"
        assert sm.dep_install_results["obscure-pkg==1.0.0"] == "installed-no-deps"
        assert len(calls) == 3
        assert "--no-deps" in calls[2]

    def test_all_strategies_fail(self, tmp_path):
        """All three strategies fail → recorded as 'failed'."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)

        def mock_pip(args, deadline=None):
            raise DependencyInstallError("always fails")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("broken-pkg==0.0.1")

        assert result == "failed"
        assert sm.dep_install_results["broken-pkg==0.0.1"] == "failed"

    def test_version_incompatible_skips_fallbacks(self, tmp_path):
        """Version incompatibility in strategy 1 → skip strategies 2+3."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            raise DependencyInstallError(
                "pip install failed (exit 1): "
                "ERROR: Package requires Python >=3.12 but you have 3.11"
            )

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("new-pkg==2.0.0")

        assert result == "failed"
        assert sm.dep_install_results["new-pkg==2.0.0"] == "failed"
        # Only strategy 1 attempted — no fallback
        assert len(calls) == 1

    def test_version_incompatible_python_requires(self, tmp_path):
        """python_requires marker detected → skip fallbacks."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            raise DependencyInstallError(
                "pip install failed (exit 1): "
                "python_requires='>=3.13' is not compatible"
            )

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("future-pkg==1.0.0")

        assert result == "failed"
        assert len(calls) == 1

    def test_low_budget_skips_strategy2(self, tmp_path):
        """Less than 10s remaining → skip strategies 2+3."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)

        # Deadline 5 seconds from now
        deadline = time.monotonic() + 5
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            raise DependencyInstallError("failed")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            result = sm._pip_install_with_fallback("slow-pkg==1.0.0", deadline=deadline)

        assert result == "failed"
        # Strategy 1 attempted, then budget guard skipped 2+3
        assert len(calls) == 1

    def test_low_budget_skips_strategy3_only(self, tmp_path):
        """Budget enough for strategy 2 but not 3."""
        project = tmp_path / "proj"
        project.mkdir()
        sm = self._make_session(tmp_path, project)

        calls = []
        # Use a fixed fake time that advances: starts at 100, then after
        # strategy 2 fails, returns deadline - 5 (< 10s guard)
        deadline = 200.0
        time_values = iter([
            185.0,   # strategy 2 budget check: 200 - 185 = 15s > 10s, proceed
            195.0,   # strategy 3 budget check: 200 - 195 = 5s < 10s, skip
        ])

        def mock_pip(args, deadline=None):
            calls.append(args)
            raise DependencyInstallError("failed")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            with mock.patch("mycode.session.time.monotonic", side_effect=time_values):
                result = sm._pip_install_with_fallback("tricky-pkg==1.0.0", deadline=deadline)

        assert result == "failed"
        # Strategy 1 + strategy 2 attempted, strategy 3 skipped
        assert len(calls) == 2

    def test_disable_pip_version_check_preserved(self, tmp_path):
        """--disable-pip-version-check flag present on all pip calls."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==3.0.0\n")

        sm = self._make_session(tmp_path, project)

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_dependencies()

        for call in mock_run.call_args_list:
            cmd = call[0][0]
            assert "--disable-pip-version-check" in cmd

    def test_failed_deps_warning_generated(self, tmp_path):
        """Failed deps produce a warning in dep_install_warnings."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("good==1.0\nbad==1.0\n")

        sm = self._make_session(tmp_path, project)

        def mock_pip(args, deadline=None):
            if args[0] == "-r":
                raise DependencyInstallError("bulk failed")
            if "bad==1.0" in args[0]:
                raise DependencyInstallError("always fails")
            # good==1.0 succeeds on strategy 1

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            sm._install_dependencies()

        assert sm.dep_install_results.get("good==1.0") == "installed"
        assert sm.dep_install_results.get("bad==1.0") == "failed"
        assert any("1 dependency could not be installed" in w for w in sm.dep_install_warnings)
        assert any("bad==1.0" in w for w in sm.dep_install_warnings)

    def test_multiple_failed_deps_warning(self, tmp_path):
        """Multiple failed deps listed in warning."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("bad1==1.0\nbad2==1.0\n")

        sm = self._make_session(tmp_path, project)

        def mock_pip(args, deadline=None):
            raise DependencyInstallError("always fails")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            sm._install_dependencies()

        assert sm.dep_install_results.get("bad1==1.0") == "failed"
        assert sm.dep_install_results.get("bad2==1.0") == "failed"
        assert any("2 dependencies could not be installed" in w for w in sm.dep_install_warnings)

    def test_install_from_package_list_uses_fallback(self, tmp_path):
        """_install_from_package_list individual fallback uses cascade."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "app.py").write_text("pass")

        sm = self._make_session(tmp_path, project)
        sm.environment_info = EnvironmentInfo(
            installed_packages={"bad-pkg": "2.0.0"}
        )

        def mock_pip(args, deadline=None):
            # Batch install fails
            if len(args) > 1 and "--only-binary=:all:" not in args and "--no-deps" not in args:
                raise DependencyInstallError("batch failed")
            # Strategy 1 (individual) fails for bad-pkg
            if args[0] == "bad-pkg==2.0.0" and "--only-binary=:all:" not in args and "--no-deps" not in args:
                raise DependencyInstallError("compilation failed")
            # Strategy 2 (binary-only) succeeds
            if "--only-binary=:all:" in args:
                return

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            sm._install_dependencies()

        assert sm.dep_install_results.get("bad-pkg==2.0.0") == "installed-binary-only"


class TestIsVersionIncompatible:
    """Test _is_version_incompatible static method."""

    def test_requires_python(self):
        assert SessionManager._is_version_incompatible(
            "ERROR: Package requires Python >=3.12"
        )

    def test_python_requires(self):
        assert SessionManager._is_version_incompatible(
            "python_requires='>=3.13' is not compatible"
        )

    def test_requires_python_mixed_case(self):
        assert SessionManager._is_version_incompatible(
            "Requires-Python >=3.12 not satisfied"
        )

    def test_normal_error_not_version(self):
        assert not SessionManager._is_version_incompatible(
            "Could not find a version that satisfies the requirement"
        )

    def test_compilation_error_not_version(self):
        assert not SessionManager._is_version_incompatible(
            "error: command 'gcc' failed with exit status 1"
        )


# ── Subdirectory Dep-File Discovery Tests ──


class TestFindDepFileDir:
    """Test find_dep_file_dir for root and subdirectory dep files."""

    def _make_session(self, tmp_path, project):
        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project
        sm.venv_python = Path("/fake/venv/bin/python")
        sm.environment_info = EnvironmentInfo()
        return sm

    def test_root_preferred(self, tmp_path):
        """Dep file at root → returns root, even if subdir also has one."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("flask\n")
        sub = project / "sub"
        sub.mkdir()
        (sub / "requirements.txt").write_text("streamlit\n")

        sm = self._make_session(tmp_path, project)
        assert sm.find_dep_file_dir() == project

    def test_subdir_found(self, tmp_path):
        """Dep file only in subdir → returns that subdir."""
        project = tmp_path / "proj"
        project.mkdir()
        sub = project / "My Projects"
        sub.mkdir()
        (sub / "requirements.txt").write_text("streamlit\n")
        (sub / "app.py").write_text("import streamlit\n")

        sm = self._make_session(tmp_path, project)
        assert sm.find_dep_file_dir() == sub

    def test_multiple_subdirs_picks_most_source_files(self, tmp_path):
        """Two subdirs have dep files → picks the one with more .py files."""
        project = tmp_path / "proj"
        project.mkdir()
        sub_a = project / "alpha"
        sub_a.mkdir()
        (sub_a / "requirements.txt").write_text("flask\n")
        (sub_a / "one.py").write_text("")

        sub_b = project / "beta"
        sub_b.mkdir()
        (sub_b / "requirements.txt").write_text("streamlit\n")
        (sub_b / "one.py").write_text("")
        (sub_b / "two.py").write_text("")
        (sub_b / "three.py").write_text("")

        sm = self._make_session(tmp_path, project)
        assert sm.find_dep_file_dir() == sub_b

    def test_no_dep_files_returns_root(self, tmp_path):
        """No dep files anywhere → returns root (fallback to env install)."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "app.py").write_text("print('hello')\n")

        sm = self._make_session(tmp_path, project)
        assert sm.find_dep_file_dir() == project

    def test_hidden_dirs_skipped(self, tmp_path):
        """Hidden directories (.git etc.) are not searched."""
        project = tmp_path / "proj"
        project.mkdir()
        git = project / ".git"
        git.mkdir()
        (git / "package.json").write_text("{}\n")

        sm = self._make_session(tmp_path, project)
        # Should return root (no visible dep files), not .git
        assert sm.find_dep_file_dir() == project

    def test_install_deps_from_subdir(self, tmp_path):
        """Full _install_dependencies() finds requirements.txt in subdir."""
        project = tmp_path / "proj"
        project.mkdir()
        sub = project / "app"
        sub.mkdir()
        (sub / "requirements.txt").write_text("streamlit==1.55.0\n")

        sm = self._make_session(tmp_path, project)

        with mock.patch.object(sm, "_pip_install") as mock_pip:
            sm._install_dependencies()

        mock_pip.assert_called_once_with(
            ["-r", str(sub / "requirements.txt")], deadline=mock.ANY,
        )

    def test_find_dep_dir_ignores_js_for_python(self, tmp_path):
        """Root package.json should not prevent finding Python deps in subdir."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "package.json").write_text('{"name":"root"}\n')
        backend = project / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").write_text('[project]\nname="app"\n')
        (backend / "app.py").write_text("pass\n")

        sm = self._make_session(tmp_path, project)
        # Default (all filenames) returns root because package.json is there
        assert sm.find_dep_file_dir() == project
        # Python-only filter finds backend/
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == backend

    def test_find_dep_dir_root_python_still_wins(self, tmp_path):
        """Root with both package.json and requirements.txt → returns root."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "package.json").write_text('{"name":"root"}\n')
        (project / "requirements.txt").write_text("flask\n")
        backend = project / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").write_text('[project]\nname="app"\n')

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        # Python dep at root → returns root even with filter
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == project

    def test_install_deps_monorepo_finds_subdir_pyproject(self, tmp_path):
        """_install_dependencies() finds backend/pyproject.toml in monorepo."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "package.json").write_text('{"name":"root"}\n')
        backend = project / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").write_text(
            '[project]\nname="app"\ndependencies=["fastapi>=0.135.0"]\n'
        )

        sm = self._make_session(tmp_path, project)
        calls = []

        def mock_pip(args, deadline=None):
            calls.append(args)
            # pip install backend/ fails (flat-layout)
            if args[0] == str(backend):
                raise DependencyInstallError("Multiple top-level packages")

        with mock.patch.object(sm, "_pip_install", side_effect=mock_pip):
            sm._install_dependencies()

        # Should have tried pip install backend/, then parsed deps
        assert calls[0] == [str(backend)]
        individual = [c[0] for c in calls[1:]]
        assert "fastapi>=0.135.0" in individual

    def test_root_pyproject_without_deps_skipped(self, tmp_path):
        """Root pyproject.toml without [project.dependencies] → search subdirs."""
        project = tmp_path / "proj"
        project.mkdir()
        # UV workspace config — no [project.dependencies]
        (project / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["backend"]\n'
        )
        backend = project / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").write_text(
            '[project]\nname="app"\ndependencies=["fastapi"]\n'
        )
        (backend / "app.py").write_text("pass\n")

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == backend

    def test_root_pyproject_with_deps_returns_root(self, tmp_path):
        """Root pyproject.toml WITH [project.dependencies] → returns root."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname="app"\ndependencies=["flask"]\n'
        )
        backend = project / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").write_text(
            '[project]\nname="api"\ndependencies=["fastapi"]\n'
        )

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == project

    def test_root_pyproject_empty_deps_skipped(self, tmp_path):
        """Root pyproject.toml with empty dependencies list → search subdirs."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname="meta"\ndependencies=[]\n'
        )
        backend = project / "backend"
        backend.mkdir()
        (backend / "requirements.txt").write_text("flask\n")
        (backend / "app.py").write_text("pass\n")

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == backend

    def test_root_pyproject_malformed_returns_root(self, tmp_path):
        """Unparseable root pyproject.toml → conservative, return root."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "pyproject.toml").write_text("this is not valid toml {{{")

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == project

    def test_root_requirements_txt_beats_empty_pyproject(self, tmp_path):
        """Root requirements.txt still short-circuits even with empty pyproject."""
        project = tmp_path / "proj"
        project.mkdir()
        (project / "requirements.txt").write_text("flask\n")
        # pyproject.toml with no deps — but shouldn't matter since
        # requirements.txt is checked first and short-circuits.
        (project / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["backend"]\n'
        )

        sm = self._make_session(tmp_path, project)
        from mycode.session import _PY_DEP_FILENAMES
        assert sm.find_dep_file_dir(filenames=_PY_DEP_FILENAMES) == project


# ── JS Dependency Installation Tests ──


class TestJsDependencyInstallation:
    """Test _install_js_dependencies for Node.js projects."""

    def test_npm_ci_when_lockfile_exists(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')
        (project / "package-lock.json").write_text("{}")

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_js_dependencies()

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["npm", "ci"]

    def test_npm_install_when_no_lockfile(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_js_dependencies()

        cmd = mock_run.call_args[0][0]
        assert cmd == ["npm", "install"]

    def test_skipped_when_no_package_json(self, tmp_path):
        project = tmp_path / "python_project"
        project.mkdir()
        (project / "app.py").write_text("pass")

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            sm._install_js_dependencies()

        mock_run.assert_not_called()
        assert sm.js_deps_installed is None  # not a JS project

    def test_skips_when_node_modules_exists(self, tmp_path):
        """If node_modules/ is already populated, skip npm install."""
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')
        nm = project / "node_modules"
        nm.mkdir()
        (nm / "express").mkdir()
        (nm / "express" / "index.js").write_text("module.exports = {};")

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            sm._install_js_dependencies()

        mock_run.assert_not_called()
        assert sm.js_deps_installed is True

    def test_npm_not_found_sets_failure(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run", side_effect=FileNotFoundError("npm not found")):
            sm._install_js_dependencies()  # should not raise

        assert sm.js_deps_installed is False
        assert "npm not found" in sm.js_deps_error

    def test_npm_failure_sets_error(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="ERR! missing dep")
            sm._install_js_dependencies()  # should not raise

        assert sm.js_deps_installed is False
        assert "ERR!" in sm.js_deps_error

    def test_npm_success_sets_installed(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_js_dependencies()

        assert sm.js_deps_installed is True
        assert sm.js_deps_error == ""

    def test_npm_eperm_retries_with_local_cache(self, tmp_path):
        """EPERM on default npm cache → retry with workspace-local cache."""
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project
        sm.workspace_dir = workspace

        eperm_result = mock.Mock(returncode=1, stdout="", stderr="npm error code EPERM\nnpm error syscall open")
        success_result = mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.side_effect = [eperm_result, success_result]
            sm._install_js_dependencies()

        assert sm.js_deps_installed is True
        assert mock_run.call_count == 2
        # Second call should have --cache flag
        retry_cmd = mock_run.call_args_list[1][0][0]
        assert "--cache" in retry_cmd

    def test_npm_timeout_sets_failure(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        import subprocess as sp
        with mock.patch("mycode.session.subprocess.run", side_effect=sp.TimeoutExpired("npm", 120)):
            sm._install_js_dependencies()

        assert sm.js_deps_installed is False
        assert "timed out" in sm.js_deps_error

    def test_node_env_development_set(self, tmp_path):
        """NODE_ENV=development should be passed so devDependencies install."""
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_js_dependencies()

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert env is not None
        assert env.get("NODE_ENV") == "development"

    def test_cwd_is_project_copy_dir(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            sm._install_js_dependencies()

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("cwd") == str(project) or call_kwargs[1].get("cwd") == str(project)

    def test_python_project_no_npm_attempted(self, tmp_path):
        """Python project (no package.json) should not attempt npm install."""
        project = tmp_path / "py_project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask==2.0.0\n")
        (project / "app.py").write_text("pass")

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            sm._install_js_dependencies()

        mock_run.assert_not_called()
        assert sm.js_deps_installed is None


# ── Orphan Cleanup Tests ──


class TestOrphanCleanup:
    def test_cleans_old_dead_process_orphan(self, tmp_path):
        orphan = tmp_path / f"{SessionManager.TEMP_PREFIX}old"
        orphan.mkdir()
        marker = orphan / SessionManager.MARKER_FILE
        marker.write_text(
            json.dumps({"pid": -1, "created": time.time() - 200000})
        )

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 1
        assert not orphan.exists()

    def test_cleans_orphan_with_dead_process(self, tmp_path):
        orphan = tmp_path / f"{SessionManager.TEMP_PREFIX}dead"
        orphan.mkdir()
        marker = orphan / SessionManager.MARKER_FILE
        # Recent but process is dead (pid -1 is never running)
        marker.write_text(
            json.dumps({"pid": -1, "created": time.time()})
        )

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 1
        assert not orphan.exists()

    def test_preserves_session_with_running_process(self, tmp_path):
        recent = tmp_path / f"{SessionManager.TEMP_PREFIX}alive"
        recent.mkdir()
        marker = recent / SessionManager.MARKER_FILE
        marker.write_text(
            json.dumps({"pid": os.getpid(), "created": time.time()})
        )

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 0
        assert recent.exists()

    def test_cleans_orphan_without_marker(self, tmp_path):
        orphan = tmp_path / f"{SessionManager.TEMP_PREFIX}nomarker"
        orphan.mkdir()
        # Make it look old by setting mtime far in the past
        old_time = time.time() - 200000
        os.utime(orphan, (old_time, old_time))

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 1
        assert not orphan.exists()

    def test_preserves_recent_orphan_without_marker(self, tmp_path):
        recent = tmp_path / f"{SessionManager.TEMP_PREFIX}recent_nomark"
        recent.mkdir()
        # mtime is now (just created), so it's recent

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 0
        assert recent.exists()

    def test_ignores_non_mycode_directories(self, tmp_path):
        unrelated = tmp_path / "some_other_dir"
        unrelated.mkdir()

        cleaned = SessionManager.cleanup_orphans(temp_base=tmp_path)
        assert cleaned == 0
        assert unrelated.exists()

    def test_handles_nonexistent_temp_base(self, tmp_path):
        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path / "does_not_exist"
        )
        assert cleaned == 0

    def test_handles_corrupt_marker(self, tmp_path):
        orphan = tmp_path / f"{SessionManager.TEMP_PREFIX}corrupt"
        orphan.mkdir()
        marker = orphan / SessionManager.MARKER_FILE
        marker.write_text("not valid json{{{")
        old_time = time.time() - 200000
        os.utime(orphan, (old_time, old_time))

        cleaned = SessionManager.cleanup_orphans(
            temp_base=tmp_path, max_age_hours=1
        )
        assert cleaned == 1


# ── Signal Handler Tests ──


class TestSignalHandling:
    def test_signal_handlers_installed_on_register(self, sample_project):
        sm = SessionManager(sample_project)
        sm._register_session()

        assert SessionManager._signal_handlers_installed is True
        assert signal.getsignal(signal.SIGINT) == SessionManager._signal_handler
        if not _IS_WINDOWS:
            assert signal.getsignal(signal.SIGTERM) == SessionManager._signal_handler

        sm._unregister_session()

    def test_signal_handlers_restored_on_unregister(self, sample_project):
        original_sigint = signal.getsignal(signal.SIGINT)
        if not _IS_WINDOWS:
            original_sigterm = signal.getsignal(signal.SIGTERM)

        sm = SessionManager(sample_project)
        sm._register_session()
        sm._unregister_session()

        assert SessionManager._signal_handlers_installed is False
        assert signal.getsignal(signal.SIGINT) == original_sigint
        if not _IS_WINDOWS:
            assert signal.getsignal(signal.SIGTERM) == original_sigterm

    def test_multiple_sessions_all_registered(self, sample_project):
        sm1 = SessionManager(sample_project)
        sm2 = SessionManager(sample_project)
        sm1._register_session()
        sm2._register_session()

        assert len(SessionManager._active_sessions) == 2

        sm1._unregister_session()
        # Handlers still installed because sm2 is active
        assert SessionManager._signal_handlers_installed is True

        sm2._unregister_session()
        assert SessionManager._signal_handlers_installed is False

    def test_signal_handler_cleans_all_sessions(self, sample_project, tmp_path):
        sm1 = SessionManager(sample_project, temp_base=tmp_path / "s1")
        sm2 = SessionManager(sample_project, temp_base=tmp_path / "s2")

        # Manually create workspace dirs to verify cleanup
        sm1.workspace_dir = tmp_path / "ws1"
        sm1.workspace_dir.mkdir()
        sm2.workspace_dir = tmp_path / "ws2"
        sm2.workspace_dir.mkdir()

        sm1._register_session()
        sm2._register_session()

        with pytest.raises(KeyboardInterrupt):
            SessionManager._signal_handler(signal.SIGINT, None)

        assert not sm1.workspace_dir  # set to None after teardown
        assert not sm2.workspace_dir


# ── Marker File Tests ──


class TestMarkerFile:
    def test_marker_written_with_correct_fields(self, sample_project, tmp_path):
        sm = SessionManager(sample_project)
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()

        sm._write_marker()

        marker = sm.workspace_dir / SessionManager.MARKER_FILE
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["pid"] == os.getpid()
        assert "created" in data
        assert data["project"] == str(sample_project.resolve())
        assert data["python_version"] == platform.python_version()


# ── Helper Tests ──


class TestIsProcessRunning:
    def test_current_process_is_running(self):
        assert _is_process_running(os.getpid()) is True

    def test_invalid_pid_not_running(self):
        assert _is_process_running(-1) is False
        assert _is_process_running(0) is False

    def test_nonexistent_pid(self):
        # PID 99999999 is very unlikely to exist
        result = _is_process_running(99999999)
        # Could be True (permission error) or False (not found)
        assert isinstance(result, bool)


# ── Teardown Tests ──


class TestTeardown:
    def test_teardown_removes_workspace(self, sample_project, tmp_path):
        sm = SessionManager(sample_project, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        (sm.workspace_dir / "file.txt").write_text("data")

        sm._register_session()
        sm.teardown()

        assert not (tmp_path / "ws").exists()

    def test_teardown_idempotent(self, sample_project, tmp_path):
        sm = SessionManager(sample_project, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm._register_session()

        sm.teardown()
        sm.teardown()  # should not raise

    def test_teardown_resets_state(self, sample_project, tmp_path):
        sm = SessionManager(sample_project, temp_base=tmp_path / "sess")
        sm.workspace_dir = tmp_path / "ws"
        sm.workspace_dir.mkdir()
        sm.venv_dir = tmp_path / "ws" / "venv"
        sm.project_copy_dir = tmp_path / "ws" / "project"
        sm.venv_python = tmp_path / "ws" / "venv" / "bin" / "python"
        sm._setup_complete = True
        sm._register_session()

        sm.teardown()

        assert sm.workspace_dir is None
        assert sm.venv_dir is None
        assert sm.project_copy_dir is None
        assert sm.venv_python is None
        assert sm._setup_complete is False


# ── Integration Tests (create real venvs — slow) ──


@pytest.mark.slow
class TestIntegrationContextManager:
    """Integration tests that create real virtual environments."""

    def test_full_lifecycle(self, sample_project, tmp_path):
        temp_base = tmp_path / "sessions"
        with SessionManager(sample_project, temp_base=temp_base) as session:
            assert session._setup_complete is True
            assert session.workspace_dir.exists()
            assert session.venv_dir.exists()
            assert session.project_copy_dir.exists()
            assert session.venv_python.exists()
            workspace = session.workspace_dir

        # After exit, everything is cleaned up
        assert not workspace.exists()

    def test_cleanup_on_exception(self, sample_project, tmp_path):
        workspace = None
        try:
            with SessionManager(
                sample_project, temp_base=tmp_path / "sessions"
            ) as session:
                workspace = session.workspace_dir
                assert workspace.exists()
                raise ValueError("deliberate test error")
        except ValueError:
            pass

        assert workspace is not None
        assert not workspace.exists()

    def test_project_copy_matches_source(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            copy_app = session.project_copy_dir / "app.py"
            assert copy_app.exists()
            assert copy_app.read_text() == (sample_project / "app.py").read_text()

    def test_original_files_never_modified(self, sample_project, tmp_path):
        original_content = (sample_project / "app.py").read_text()

        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            # Modify the copy
            (session.project_copy_dir / "app.py").write_text("MODIFIED")

        assert (sample_project / "app.py").read_text() == original_content


@pytest.mark.slow
class TestIntegrationRunInSession:
    """Integration tests for running commands inside the session."""

    def test_run_simple_python_command(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "print('hello from venv')"]
            )
            assert result.returncode == 0
            assert "hello from venv" in result.stdout

    def test_run_uses_venv_python(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import sys; print(sys.prefix)"]
            )
            assert result.returncode == 0
            assert "mycode_session_" in result.stdout

    def test_run_cwd_is_project_copy(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.getcwd())"]
            )
            assert result.returncode == 0
            assert str(session.project_copy_dir) in result.stdout

    def test_run_captures_stderr(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import sys; sys.stderr.write('err msg\\n')"]
            )
            assert "err msg" in result.stderr

    def test_run_captures_nonzero_exit(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(["python", "-c", "exit(42)"])
            assert result.returncode == 42

    def test_run_timeout(self, sample_project, tmp_path):
        caps = ResourceCaps(timeout_seconds=2)
        with SessionManager(
            sample_project, resource_caps=caps, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import time; time.sleep(30)"],
                timeout=1,
            )
            assert result.timed_out is True
            assert result.returncode == -1

    def test_run_not_setup_raises(self, sample_project):
        sm = SessionManager(sample_project)
        with pytest.raises(SessionError, match="not set up"):
            sm.run_in_session(["python", "-c", "pass"])

    def test_run_with_env_vars(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.environ['MY_VAR'])"],
                env_vars={"MY_VAR": "test_value"},
            )
            assert result.returncode == 0
            assert "test_value" in result.stdout

    def test_run_can_import_from_project(self, sample_project, tmp_path):
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "from app import hello; print(hello())"]
            )
            assert result.returncode == 0
            assert "Hello, World!" in result.stdout


@pytest.mark.slow
class TestIntegrationSetupTeardown:
    """Integration tests for setup failure handling."""

    def test_setup_idempotent(self, sample_project, tmp_path):
        sm = SessionManager(sample_project, temp_base=tmp_path / "sessions")
        sm.setup()
        workspace = sm.workspace_dir
        sm.setup()  # second call is a no-op
        assert sm.workspace_dir == workspace
        sm.teardown()

    def test_setup_cleans_up_on_venv_failure(self, sample_project, tmp_path):
        sm = SessionManager(sample_project, temp_base=tmp_path / "sessions")

        with mock.patch("mycode.session.venv.create", side_effect=OSError("boom")):
            with pytest.raises(VenvCreationError):
                sm.setup()

        # Workspace should be cleaned up after failed setup
        assert sm.workspace_dir is None
        assert sm._setup_complete is False


@pytest.mark.slow
class TestPythonPathInSession:
    """Tests for PYTHONPATH injection in run_in_session."""

    def test_pythonpath_includes_project_root(self, sample_project, tmp_path):
        """PYTHONPATH always contains the project copy directory."""
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"]
            )
            assert result.returncode == 0
            assert str(session.project_copy_dir) in result.stdout

    def test_pythonpath_includes_dep_subdir(self, tmp_path):
        """When dep files are in a subdirectory, PYTHONPATH includes it."""
        # Create project with code in a subdirectory
        project = tmp_path / "subdir_project"
        project.mkdir()
        (project / "README.md").write_text("top level\n")
        subdir = project / "myapp"
        subdir.mkdir()
        (subdir / "requirements.txt").write_text("# empty\n")
        (subdir / "helper.py").write_text("def greet():\n    return 'hi'\n")

        with SessionManager(
            project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"]
            )
            assert result.returncode == 0
            pypath = result.stdout.strip()
            # Both root and subdirectory should be on PYTHONPATH
            assert str(session.project_copy_dir) in pypath
            # The subdirectory name should appear
            assert "myapp" in pypath

    def test_pythonpath_no_duplicate_when_deps_at_root(self, sample_project, tmp_path):
        """When dep files are at root, PYTHONPATH has root only once."""
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"]
            )
            assert result.returncode == 0
            pypath = result.stdout.strip()
            root = str(session.project_copy_dir)
            # Root should appear exactly once
            assert pypath.count(root) == 1

    def test_import_from_subdirectory(self, tmp_path):
        """Code in a dep-file subdirectory can be imported by the harness."""
        project = tmp_path / "subdir_import_project"
        project.mkdir()
        (project / "README.md").write_text("top level\n")
        subdir = project / "src"
        subdir.mkdir()
        (subdir / "requirements.txt").write_text("# empty\n")
        (subdir / "mymod.py").write_text("VALUE = 42\n")

        with SessionManager(
            project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "from mymod import VALUE; print(VALUE)"]
            )
            assert result.returncode == 0
            assert "42" in result.stdout

    def test_env_vars_can_override_pythonpath(self, sample_project, tmp_path):
        """Caller-provided env_vars can override PYTHONPATH."""
        with SessionManager(
            sample_project, temp_base=tmp_path / "sessions"
        ) as session:
            result = session.run_in_session(
                ["python", "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"],
                env_vars={"PYTHONPATH": "/custom/path"},
            )
            assert result.returncode == 0
            assert "/custom/path" in result.stdout
