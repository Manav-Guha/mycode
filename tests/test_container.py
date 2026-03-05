"""Tests for Docker containerisation module and CLI integration.

Tests cover:
  - Docker availability detection
  - Dockerfile generation (from source and from PyPI)
  - Image tag construction
  - Image existence check
  - Repo root detection
  - CLI --containerised flag (Docker unavailable, passes through args)
  - CLI --python-version flag
  - CLI --yes flag and untrusted code warning
  - CLI passthrough arg collection
  - Container run command construction
  - Error handling (Docker not available, build failure, timeout)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mycode.container import (
    ContainerError,
    _find_repo_root,
    _generate_dockerfile,
    _image_exists,
    _image_tag,
    build_image,
    is_docker_available,
    run_containerised,
)


# ── Docker Detection ──


class TestDockerDetection:
    """Tests for is_docker_available()."""

    def test_docker_available(self):
        """Returns True when docker info succeeds."""
        mock_result = MagicMock(returncode=0)
        with patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run:
            assert is_docker_available() is True
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd == ["docker", "info"]

    def test_docker_not_installed(self):
        """Returns False when docker binary not found."""
        with patch("mycode.container.subprocess.run", side_effect=FileNotFoundError):
            assert is_docker_available() is False

    def test_docker_daemon_not_running(self):
        """Returns False when docker info fails (daemon not running)."""
        mock_result = MagicMock(returncode=1)
        with patch("mycode.container.subprocess.run", return_value=mock_result):
            assert is_docker_available() is False

    def test_docker_timeout(self):
        """Returns False when docker info times out."""
        import subprocess
        with patch(
            "mycode.container.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=10),
        ):
            assert is_docker_available() is False

    def test_docker_os_error(self):
        """Returns False on OSError."""
        with patch("mycode.container.subprocess.run", side_effect=OSError("fail")):
            assert is_docker_available() is False


# ── Image Tag ──


class TestImageTag:
    """Tests for _image_tag()."""

    def test_default_version(self):
        assert _image_tag("3.11") == "mycode:py3.11"

    def test_custom_version(self):
        assert _image_tag("3.12") == "mycode:py3.12"

    def test_minor_version(self):
        assert _image_tag("3.10") == "mycode:py3.10"


# ── Image Exists ──


class TestImageExists:
    """Tests for _image_exists()."""

    def test_image_exists_true(self):
        mock_result = MagicMock(returncode=0)
        with patch("mycode.container.subprocess.run", return_value=mock_result):
            assert _image_exists("mycode:py3.11") is True

    def test_image_not_found(self):
        mock_result = MagicMock(returncode=1)
        with patch("mycode.container.subprocess.run", return_value=mock_result):
            assert _image_exists("mycode:py3.11") is False

    def test_docker_not_available(self):
        with patch("mycode.container.subprocess.run", side_effect=FileNotFoundError):
            assert _image_exists("mycode:py3.11") is False


# ── Dockerfile Generation ──


class TestDockerfileGeneration:
    """Tests for _generate_dockerfile()."""

    def test_from_source(self):
        """Dockerfile from source copies local files."""
        content = _generate_dockerfile("3.11", from_source=True)
        assert "FROM python:3.11-slim" in content
        assert "COPY . /opt/mycode" in content
        assert "pip install --no-cache-dir /opt/mycode" in content
        assert "nodejs" in content
        assert 'ENTRYPOINT ["mycode"]' in content

    def test_from_pypi(self):
        """Dockerfile from PyPI installs via pip."""
        content = _generate_dockerfile("3.11", from_source=False)
        assert "FROM python:3.11-slim" in content
        assert "pip install --no-cache-dir mycode-ai" in content
        assert "COPY . /opt/mycode" not in content

    def test_custom_python_version(self):
        """Dockerfile uses specified Python version."""
        content = _generate_dockerfile("3.12", from_source=True)
        assert "FROM python:3.12-slim" in content

    def test_nodejs_installed(self):
        """Both variants install Node.js."""
        for from_source in [True, False]:
            content = _generate_dockerfile("3.11", from_source=from_source)
            assert "nodejs" in content
            assert "npm" in content

    def test_workspace_created(self):
        """Dockerfile creates /workspace directory."""
        content = _generate_dockerfile("3.11", from_source=True)
        assert "/workspace" in content

    def test_entrypoint_set(self):
        """Entrypoint is mycode."""
        content = _generate_dockerfile("3.11", from_source=True)
        assert 'ENTRYPOINT ["mycode"]' in content


# ── Repo Root Detection ──


class TestFindRepoRoot:
    """Tests for _find_repo_root()."""

    def test_finds_repo_root(self):
        """Should find the repo root from the container module location."""
        root = _find_repo_root()
        # When running from the repo, this should find it
        if root is not None:
            assert (root / "pyproject.toml").is_file()
            content = (root / "pyproject.toml").read_text()
            assert 'name = "mycode-ai"' in content

    def test_returns_path_or_none(self):
        """Return type is Path or None."""
        root = _find_repo_root()
        assert root is None or isinstance(root, Path)


# ── Build Image ──


class TestBuildImage:
    """Tests for build_image()."""

    def test_reuses_existing_image(self):
        """Skips build if image already exists."""
        with patch("mycode.container._image_exists", return_value=True):
            tag = build_image("3.11")
            assert tag == "mycode:py3.11"

    def test_force_rebuild(self):
        """Force flag triggers build even if image exists."""
        with (
            patch("mycode.container._image_exists", return_value=True),
            patch("mycode.container._find_repo_root", return_value=None),
            patch("mycode.container._docker_build") as mock_build,
        ):
            tag = build_image("3.11", force=True)
            assert tag == "mycode:py3.11"
            mock_build.assert_called_once()

    def test_build_from_pypi_when_no_repo(self):
        """Falls back to PyPI install when not in a repo."""
        with (
            patch("mycode.container._image_exists", return_value=False),
            patch("mycode.container._find_repo_root", return_value=None),
            patch("mycode.container._docker_build") as mock_build,
        ):
            tag = build_image("3.12")
            assert tag == "mycode:py3.12"
            mock_build.assert_called_once()
            # Check that the Dockerfile was written to a temp dir
            args = mock_build.call_args[0]
            assert "Dockerfile" in args[2]

    def test_build_from_source(self, tmp_path):
        """Builds from local source when repo root is found."""
        # Create a fake pyproject.toml
        (tmp_path / "pyproject.toml").write_text('name = "mycode-ai"\n')

        with (
            patch("mycode.container._image_exists", return_value=False),
            patch("mycode.container._find_repo_root", return_value=tmp_path),
            patch("mycode.container._docker_build") as mock_build,
        ):
            tag = build_image("3.11")
            assert tag == "mycode:py3.11"
            mock_build.assert_called_once()
            # Context path should be the repo root
            context_path = mock_build.call_args[0][1]
            assert context_path == str(tmp_path)

    def test_cleans_up_temp_dockerfile(self, tmp_path):
        """Temporary Dockerfile is cleaned up after build."""
        (tmp_path / "pyproject.toml").write_text('name = "mycode-ai"\n')
        temp_dockerfile = tmp_path / "Dockerfile.mycode.tmp"

        with (
            patch("mycode.container._image_exists", return_value=False),
            patch("mycode.container._find_repo_root", return_value=tmp_path),
            patch("mycode.container._docker_build"),
        ):
            build_image("3.11")
            # Temp dockerfile should be cleaned up
            assert not temp_dockerfile.exists()


# ── Container Run ──


class TestRunContainerised:
    """Tests for run_containerised()."""

    def test_basic_run(self, tmp_path):
        """Runs docker with correct arguments."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            exit_code = run_containerised(project, ["--offline"])
            assert exit_code == 0

            cmd = mock_run.call_args[0][0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "--rm" in cmd
            assert "--network=none" in cmd
            assert "/workspace/project" in cmd
            assert "--offline" in cmd

    def test_project_mounted_read_only(self, tmp_path):
        """Project directory is mounted as read-only."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            run_containerised(project, [])
            cmd = mock_run.call_args[0][0]
            # Find the -v flag
            volume_args = [
                cmd[i + 1] for i in range(len(cmd))
                if cmd[i] == "-v" and i + 1 < len(cmd)
            ]
            assert any(":ro" in v for v in volume_args)

    def test_non_interactive_added(self, tmp_path):
        """--non-interactive is added if not already present."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            run_containerised(project, [])
            cmd = mock_run.call_args[0][0]
            assert "--non-interactive" in cmd

    def test_non_interactive_not_duplicated(self, tmp_path):
        """--non-interactive is not added if already present."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            run_containerised(project, ["--non-interactive"])
            cmd = mock_run.call_args[0][0]
            assert cmd.count("--non-interactive") == 1

    def test_returns_container_exit_code(self, tmp_path):
        """Returns the container's exit code."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=42)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result),
        ):
            assert run_containerised(project, []) == 42

    def test_timeout_returns_1(self, tmp_path):
        """Returns 1 on timeout."""
        import subprocess
        project = tmp_path / "project"
        project.mkdir()

        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch(
                "mycode.container.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=1800),
            ),
        ):
            assert run_containerised(project, []) == 1

    def test_docker_not_found_raises(self, tmp_path):
        """Raises ContainerError when docker is not found during run."""
        project = tmp_path / "project"
        project.mkdir()

        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", side_effect=FileNotFoundError),
        ):
            with pytest.raises(ContainerError, match="Docker not found"):
                run_containerised(project, [])

    def test_resource_limits_set(self, tmp_path):
        """Docker run includes memory and CPU limits."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            run_containerised(project, [])
            cmd = mock_run.call_args[0][0]
            assert "--memory=2g" in cmd
            assert "--cpus=2" in cmd

    def test_containerised_env_var_set(self, tmp_path):
        """MYCODE_CONTAINERISED=1 env var is passed to the container."""
        project = tmp_path / "project"
        project.mkdir()

        mock_result = MagicMock(returncode=0)
        with (
            patch("mycode.container.build_image", return_value="mycode:py3.11"),
            patch("mycode.container.subprocess.run", return_value=mock_result) as mock_run,
        ):
            run_containerised(project, [])
            cmd = mock_run.call_args[0][0]
            # Check -e MYCODE_CONTAINERISED=1 is in the command
            env_args = [
                cmd[i + 1] for i in range(len(cmd))
                if cmd[i] == "-e" and i + 1 < len(cmd)
            ]
            assert "MYCODE_CONTAINERISED=1" in env_args


# ── Docker Build Errors ──


class TestDockerBuildErrors:
    """Tests for _docker_build error handling."""

    def test_build_failure_raises(self):
        from mycode.container import _docker_build
        mock_result = MagicMock(returncode=1, stderr="error: something went wrong")
        with patch("mycode.container.subprocess.run", return_value=mock_result):
            with pytest.raises(ContainerError, match="Docker build failed"):
                _docker_build("mycode:py3.11", "/tmp", "/tmp/Dockerfile")

    def test_build_timeout_raises(self):
        import subprocess
        from mycode.container import _docker_build
        with patch(
            "mycode.container.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=600),
        ):
            with pytest.raises(ContainerError, match="timed out"):
                _docker_build("mycode:py3.11", "/tmp", "/tmp/Dockerfile")

    def test_docker_not_found_raises(self):
        from mycode.container import _docker_build
        with patch("mycode.container.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ContainerError, match="Docker not found"):
                _docker_build("mycode:py3.11", "/tmp", "/tmp/Dockerfile")


# ── CLI Integration ──


class TestCLIContainerised:
    """Tests for --containerised CLI flag integration."""

    def test_parser_accepts_containerised(self):
        """Parser accepts --containerised flag."""
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--containerised"])
        assert args.containerised is True

    def test_parser_accepts_python_version(self):
        """Parser accepts --python-version flag."""
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--python-version", "3.12"])
        assert args.python_version == "3.12"

    def test_python_version_default(self):
        """Default python version is 3.11."""
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path"])
        assert args.python_version == "3.11"

    def test_parser_accepts_yes(self):
        """Parser accepts --yes / -y flag."""
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--yes"])
        assert args.yes is True
        args2 = parser.parse_args(["/some/path", "-y"])
        assert args2.yes is True

    def test_containerised_checks_docker(self, tmp_path):
        """--containerised returns error when Docker is not available."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with patch("mycode.container.is_docker_available", return_value=False):
            exit_code = main([str(proj), "--containerised"])
            assert exit_code == 1

    def test_containerised_calls_run(self, tmp_path):
        """--containerised calls run_containerised with correct args."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("mycode.container.is_docker_available", return_value=True),
            patch("mycode.container.run_containerised", return_value=0) as mock_run,
        ):
            exit_code = main([str(proj), "--containerised", "--offline"])
            assert exit_code == 0
            mock_run.assert_called_once()
            kwargs = mock_run.call_args[1]
            assert kwargs["python_version"] == "3.11"
            assert "--offline" in kwargs["cli_args"]

    def test_containerised_passes_python_version(self, tmp_path):
        """--python-version is forwarded to run_containerised."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("mycode.container.is_docker_available", return_value=True),
            patch("mycode.container.run_containerised", return_value=0) as mock_run,
        ):
            main([str(proj), "--containerised", "--python-version", "3.12"])
            kwargs = mock_run.call_args[1]
            assert kwargs["python_version"] == "3.12"

    def test_containerised_container_error(self, tmp_path):
        """ContainerError from run_containerised is caught and returns 1."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("mycode.container.is_docker_available", return_value=True),
            patch(
                "mycode.container.run_containerised",
                side_effect=ContainerError("build failed"),
            ),
        ):
            exit_code = main([str(proj), "--containerised"])
            assert exit_code == 1


# ── Untrusted Code Warning ──


class TestUntrustedCodeWarning:
    """Tests for the untrusted code warning in non-containerised mode."""

    def test_warning_skipped_with_yes(self, tmp_path):
        """--yes skips the untrusted code warning."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with patch("mycode.cli.run_pipeline") as mock_run:
            mock_run.return_value = MagicMock(
                success=True, report=None, execution=None,
                stages=[], warnings=[], recording_path=None,
                failed_stage=None,
            )
            exit_code = main([str(proj), "--yes", "--offline"])
            assert exit_code == 0
            mock_run.assert_called_once()

    def test_warning_skipped_with_non_interactive(self, tmp_path):
        """--non-interactive skips the untrusted code warning."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with patch("mycode.cli.run_pipeline") as mock_run:
            mock_run.return_value = MagicMock(
                success=True, report=None, execution=None,
                stages=[], warnings=[], recording_path=None,
                failed_stage=None,
            )
            exit_code = main([str(proj), "--non-interactive", "--offline"])
            assert exit_code == 0
            mock_run.assert_called_once()

    def test_warning_abort_on_no(self, tmp_path):
        """User entering 'n' at the warning prompt aborts."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("builtins.input", return_value="n"),
            patch("sys.stdin") as mock_stdin,
            patch("mycode.cli.run_pipeline") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            exit_code = main([str(proj), "--offline"])
            assert exit_code == 0
            mock_run.assert_not_called()

    def test_warning_proceed_on_yes(self, tmp_path):
        """User entering 'Y' at the warning prompt proceeds."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("builtins.input", return_value="Y"),
            patch("sys.stdin") as mock_stdin,
            patch("mycode.cli.run_pipeline") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            mock_run.return_value = MagicMock(
                success=True, report=None, execution=None,
                stages=[], warnings=[], recording_path=None,
                failed_stage=None,
            )
            exit_code = main([str(proj), "--offline"])
            assert exit_code == 0
            mock_run.assert_called_once()

    def test_warning_proceed_on_empty(self, tmp_path):
        """User pressing Enter (empty string) at the warning proceeds."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("builtins.input", return_value=""),
            patch("sys.stdin") as mock_stdin,
            patch("mycode.cli.run_pipeline") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            mock_run.return_value = MagicMock(
                success=True, report=None, execution=None,
                stages=[], warnings=[], recording_path=None,
                failed_stage=None,
            )
            exit_code = main([str(proj), "--offline"])
            assert exit_code == 0
            mock_run.assert_called_once()

    def test_warning_abort_on_eof(self, tmp_path):
        """EOFError at the warning prompt aborts."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("builtins.input", side_effect=EOFError),
            patch("sys.stdin") as mock_stdin,
            patch("mycode.cli.run_pipeline") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            exit_code = main([str(proj), "--offline"])
            assert exit_code == 0
            mock_run.assert_not_called()

    def test_warning_abort_on_keyboard_interrupt(self, tmp_path):
        """KeyboardInterrupt at the warning prompt aborts."""
        from mycode.cli import main
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "app.py").write_text("x = 1\n")

        with (
            patch("builtins.input", side_effect=KeyboardInterrupt),
            patch("sys.stdin") as mock_stdin,
            patch("mycode.cli.run_pipeline") as mock_run,
        ):
            mock_stdin.isatty.return_value = True
            exit_code = main([str(proj), "--offline"])
            assert exit_code == 0
            mock_run.assert_not_called()


# ── Passthrough Arg Collection ──


class TestPassthroughArgs:
    """Tests for _collect_passthrough_args()."""

    def test_empty_args(self):
        """No flags produces empty passthrough."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path"])
        result = _collect_passthrough_args(args)
        assert result == []

    def test_offline_passed_through(self):
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--offline"])
        result = _collect_passthrough_args(args)
        assert "--offline" in result

    def test_language_passed_through(self):
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--language", "python"])
        result = _collect_passthrough_args(args)
        assert "--language" in result
        assert "python" in result

    def test_api_key_passed_through(self):
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--api-key", "sk-123"])
        result = _collect_passthrough_args(args)
        assert "--api-key" in result
        assert "sk-123" in result

    def test_verbose_passed_through(self):
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--verbose"])
        result = _collect_passthrough_args(args)
        assert "--verbose" in result

    def test_containerised_not_passed_through(self):
        """--containerised is NOT forwarded (would cause recursion)."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--containerised", "--offline"])
        result = _collect_passthrough_args(args)
        assert "--containerised" not in result

    def test_yes_not_passed_through(self):
        """--yes is NOT forwarded."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--yes", "--offline"])
        result = _collect_passthrough_args(args)
        assert "--yes" not in result
        assert "-y" not in result

    def test_json_output_not_passed_through(self):
        """--json-output is NOT forwarded (read-only mount)."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--json-output"])
        result = _collect_passthrough_args(args)
        assert "--json-output" not in result

    def test_report_not_passed_through(self):
        """--report is NOT forwarded (read-only mount)."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--report"])
        result = _collect_passthrough_args(args)
        assert "--report" not in result

    def test_multiple_flags(self):
        """Multiple flags are all collected."""
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args([
            "/some/path", "--offline", "--verbose",
            "--skip-version-check", "--consent",
        ])
        result = _collect_passthrough_args(args)
        assert "--offline" in result
        assert "--verbose" in result
        assert "--skip-version-check" in result
        assert "--consent" in result

    def test_non_interactive_passed_through(self):
        from mycode.cli import _collect_passthrough_args, build_parser
        parser = build_parser()
        args = parser.parse_args(["/some/path", "--non-interactive"])
        result = _collect_passthrough_args(args)
        assert "--non-interactive" in result
