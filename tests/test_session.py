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
        assert cmd[:3] == ["npm", "ci", "--ignore-scripts"]

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
        assert cmd[:3] == ["npm", "install", "--ignore-scripts"]

    def test_skipped_when_no_package_json(self, tmp_path):
        project = tmp_path / "python_project"
        project.mkdir()
        (project / "app.py").write_text("pass")

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            sm._install_js_dependencies()

        mock_run.assert_not_called()

    def test_npm_not_found_handled_gracefully(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run", side_effect=FileNotFoundError("npm not found")):
            # Should not raise
            sm._install_js_dependencies()

    def test_npm_failure_logged_not_raised(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        with mock.patch("mycode.session.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="ERR!")
            # Should not raise
            sm._install_js_dependencies()

    def test_npm_timeout_handled_gracefully(self, tmp_path):
        project = tmp_path / "js_project"
        project.mkdir()
        (project / "package.json").write_text('{"name":"test"}')

        sm = SessionManager(project, temp_base=tmp_path / "sess")
        sm.project_copy_dir = project

        import subprocess as sp
        with mock.patch("mycode.session.subprocess.run", side_effect=sp.TimeoutExpired("npm", 120)):
            sm._install_js_dependencies()

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
        assert signal.getsignal(signal.SIGTERM) == SessionManager._signal_handler

        sm._unregister_session()

    def test_signal_handlers_restored_on_unregister(self, sample_project):
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        sm = SessionManager(sample_project)
        sm._register_session()
        sm._unregister_session()

        assert SessionManager._signal_handlers_installed is False
        assert signal.getsignal(signal.SIGINT) == original_sigint
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
