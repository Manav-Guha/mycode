"""Session Manager (C1) — Isolated environment management for stress testing.

Creates temporary virtual environments replicating the user's Python environment.
All stress tests run inside this venv with resource caps. Cleanup is guaranteed
on normal exit, exception, or signal (Ctrl+C / SIGTERM).

Pure Python. No LLM dependency.
"""

import atexit
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directories and file patterns excluded when copying user's project
COPY_EXCLUDE_PATTERNS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".DS_Store",
}

COPY_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class ResourceCaps:
    """Resource limits enforced during stress test execution."""

    memory_mb: int = 512
    process_limit: int = 50
    timeout_seconds: int = 300


@dataclass
class EnvironmentInfo:
    """Captured information about the user's Python environment."""

    python_version: str = ""
    python_executable: str = ""
    platform_info: str = ""
    installed_packages: dict = field(default_factory=dict)  # name -> version
    dependency_files: dict = field(default_factory=dict)  # filename -> absolute path


@dataclass
class SessionResult:
    """Result of running a command inside the session."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SessionError(Exception):
    """Base exception for session manager errors."""


class VenvCreationError(SessionError):
    """Failed to create virtual environment."""


class DependencyInstallError(SessionError):
    """Failed to install dependencies in virtual environment."""


class SessionManager:
    """Creates and manages isolated virtual environments for stress testing.

    - Creates a temporary venv replicating the user's Python environment
    - Copies the user's project into an isolated workspace
    - Enforces resource caps (memory, processes, timeouts)
    - Guarantees cleanup on normal exit, exceptions, and signals (Ctrl+C)
    - Never touches the user's original files
    - Never puts the user's host environment at risk

    Usage::

        with SessionManager("/path/to/project") as session:
            result = session.run_in_session(["python", "script.py"])
    """

    TEMP_PREFIX = "mycode_session_"
    MARKER_FILE = ".mycode_session"

    # Class-level registry for signal-based cleanup of all active sessions
    _active_sessions: list = []
    _signal_handlers_installed: bool = False
    _original_sigint = None
    _original_sigterm = None

    def __init__(
        self,
        project_path: str | Path,
        resource_caps: Optional[ResourceCaps] = None,
        temp_base: Optional[str | Path] = None,
    ):
        self.project_path = Path(project_path).resolve()
        if not self.project_path.is_dir():
            raise SessionError(
                f"Project path does not exist or is not a directory: {self.project_path}"
            )

        self.resource_caps = resource_caps or ResourceCaps()
        self.temp_base = Path(temp_base) if temp_base else None

        # Populated during setup
        self.workspace_dir: Optional[Path] = None
        self.venv_dir: Optional[Path] = None
        self.project_copy_dir: Optional[Path] = None
        self.venv_python: Optional[Path] = None
        self.environment_info: Optional[EnvironmentInfo] = None

        self._cleaned_up = False
        self._setup_complete = False

    # ── Context Manager ──

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False  # never suppress exceptions

    # ── Setup ──

    def setup(self):
        """Create isolated environment: detect env, copy project, create venv, install deps."""
        if self._setup_complete:
            return

        try:
            self._register_session()

            # 1. Detect user's environment
            self.environment_info = self.detect_environment()
            logger.info(
                "Detected Python %s with %d packages",
                self.environment_info.python_version,
                len(self.environment_info.installed_packages),
            )

            # 2. Create workspace temp directory
            mkdtemp_kwargs = {"prefix": self.TEMP_PREFIX}
            if self.temp_base:
                self.temp_base.mkdir(parents=True, exist_ok=True)
                mkdtemp_kwargs["dir"] = str(self.temp_base)
            self.workspace_dir = Path(tempfile.mkdtemp(**mkdtemp_kwargs))

            self._write_marker()

            # 3. Copy project into workspace (before pip install so originals are never touched)
            self.project_copy_dir = self.workspace_dir / "project"
            self._copy_project()

            # 4. Create venv
            self.venv_dir = self.workspace_dir / "venv"
            self._create_venv()

            # 5. Install dependencies from the project copy
            self._install_dependencies()

            # 6. Install JS dependencies if package.json exists
            self._install_js_dependencies()

            self._setup_complete = True
            logger.info("Session ready: %s", self.workspace_dir)

        except Exception:
            self.teardown()
            raise

    # ── Teardown ──

    def teardown(self):
        """Destroy the virtual environment and all temporary files."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        self._unregister_session()

        if self.workspace_dir and self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir)
                logger.info("Cleaned up session: %s", self.workspace_dir)
            except OSError as e:
                logger.warning(
                    "Failed to fully clean up %s: %s", self.workspace_dir, e
                )

        self.workspace_dir = None
        self.venv_dir = None
        self.project_copy_dir = None
        self.venv_python = None
        self._setup_complete = False

    # ── Environment Detection ──

    def detect_environment(self) -> EnvironmentInfo:
        """Read user's Python version, installed packages, and dependency files."""
        info = EnvironmentInfo()
        info.python_version = platform.python_version()
        info.python_executable = sys.executable
        info.platform_info = platform.platform()
        info.installed_packages = self._get_installed_packages()
        info.dependency_files = self._find_dependency_files()
        return info

    def _get_installed_packages(self) -> dict[str, str]:
        """Get installed packages and versions from the user's environment."""
        packages = {}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"]] = pkg["version"]
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to list installed packages: %s", e)
        return packages

    def _find_dependency_files(self) -> dict[str, str]:
        """Find dependency specification files in the project."""
        dep_files = {}
        candidates = [
            "requirements.txt",
            "requirement.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "Pipfile.lock",
            "poetry.lock",
            "package.json",
            "package-lock.json",
        ]
        for name in candidates:
            path = self.project_path / name
            if path.is_file():
                dep_files[name] = str(path)
        return dep_files

    # ── Venv Creation ──

    def _create_venv(self):
        """Create a virtual environment in the workspace."""
        try:
            venv.create(str(self.venv_dir), with_pip=True, clear=True)
        except Exception as e:
            raise VenvCreationError(
                f"Failed to create virtual environment: {e}"
            ) from e

        if sys.platform == "win32":
            self.venv_python = self.venv_dir / "Scripts" / "python.exe"
        else:
            self.venv_python = self.venv_dir / "bin" / "python"

        if not self.venv_python.exists():
            raise VenvCreationError(
                f"Venv created but Python executable not found at {self.venv_python}"
            )

    # ── Dependency Installation ──

    def _install_dependencies(self):
        """Install the user's dependencies into the venv from the project copy."""
        logger.debug("[PY-DEPS] _install_dependencies called, project_copy_dir=%s", self.project_copy_dir)
        if not self.project_copy_dir:
            logger.debug("[PY-DEPS] No project_copy_dir, skipping")
            return

        installed_any = False

        # Try requirements*.txt files (most common for vibe-coded projects)
        for req_file in [
            "requirements.txt",
            "requirement.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
        ]:
            req_path = self.project_copy_dir / req_file
            logger.debug("[PY-DEPS] Checking %s — exists=%s", req_file, req_path.is_file())
            if req_path.is_file():
                try:
                    logger.debug("[PY-DEPS] Installing from %s", req_file)
                    self._pip_install(["-r", str(req_path)])
                    installed_any = True
                    logger.debug("[PY-DEPS] Successfully installed from %s", req_file)
                except DependencyInstallError as e:
                    logger.warning("[PY-DEPS] Failed to install from %s: %s", req_file, e)
                    logger.debug("[PY-DEPS] Falling back to individual package install for %s", req_file)
                    count = self._pip_install_individually(req_path)
                    if count > 0:
                        installed_any = True
                    logger.debug("[PY-DEPS] Individual install from %s: %d packages succeeded", req_file, count)

        # Try pyproject.toml (install from copy dir so originals are untouched)
        pyproject = self.project_copy_dir / "pyproject.toml"
        logger.debug("[PY-DEPS] Checking pyproject.toml — exists=%s", pyproject.is_file())
        if pyproject.is_file() and not installed_any:
            try:
                logger.debug("[PY-DEPS] Installing from pyproject.toml")
                self._pip_install([str(self.project_copy_dir)])
                installed_any = True
                logger.debug("[PY-DEPS] Successfully installed from pyproject.toml")
            except DependencyInstallError as e:
                logger.warning(
                    "[PY-DEPS] Failed to install from pyproject.toml, falling back to package list: %s", e
                )

        # Try setup.py
        setup_py = self.project_copy_dir / "setup.py"
        logger.debug("[PY-DEPS] Checking setup.py — exists=%s", setup_py.is_file())
        if setup_py.is_file() and not installed_any:
            try:
                logger.debug("[PY-DEPS] Installing from setup.py")
                self._pip_install([str(self.project_copy_dir)])
                installed_any = True
                logger.debug("[PY-DEPS] Successfully installed from setup.py")
            except DependencyInstallError as e:
                logger.warning(
                    "[PY-DEPS] Failed to install from setup.py, falling back to package list: %s", e
                )

        # Fallback: install from detected environment package list
        if (
            not installed_any
            and self.environment_info
            and self.environment_info.installed_packages
        ):
            logger.info(
                "[PY-DEPS] No dependency files found, installing from detected environment"
            )
            self._install_from_package_list(self.environment_info.installed_packages)

        logger.debug("[PY-DEPS] Installation complete: installed_any=%s", installed_any)

    def _pip_install(self, args: list[str]):
        """Run pip install with given arguments inside the venv."""
        cmd = [str(self.venv_python), "-m", "pip", "install", "--quiet"] + args
        logger.info("[PY-DEPS] pip install: %s", " ".join(args)[:200])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            logger.debug("[PY-DEPS] Exit code: %d", result.returncode)
            logger.debug("[PY-DEPS] stdout:\n%s", result.stdout[:2000])
            logger.debug("[PY-DEPS] stderr:\n%s", result.stderr[:2000])
            if result.returncode != 0:
                raise DependencyInstallError(
                    f"pip install failed (exit {result.returncode}): "
                    f"{result.stderr[:500]}"
                )
        except subprocess.TimeoutExpired as e:
            logger.debug("[PY-DEPS] pip install timed out: %s", e)
            raise DependencyInstallError(f"pip install timed out: {e}") from e

    def _pip_install_individually(self, req_path: Path) -> int:
        """Parse a requirements file and install packages one at a time.

        Returns the number of successfully installed packages.
        """
        successful = 0
        try:
            lines = req_path.read_text().splitlines()
        except OSError as e:
            logger.warning("[PY-DEPS] Could not read %s: %s", req_path, e)
            return 0

        for line in lines:
            line = line.strip()
            # Skip blanks, comments, and pip options (e.g. -i, --index-url, -f)
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            try:
                logger.debug("[PY-DEPS] Installing individual package: %s", line)
                self._pip_install([line])
                successful += 1
            except DependencyInstallError:
                logger.warning("[PY-DEPS] Failed to install %s, skipping", line)
        return successful

    def _install_from_package_list(self, packages: dict[str, str]):
        """Install specific package versions into the venv."""
        skip = {"pip", "setuptools", "wheel", "pkg-resources", "distribute"}
        specs = [
            f"{name}=={version}"
            for name, version in packages.items()
            if name.lower() not in skip
        ]
        if not specs:
            return

        batch_size = 50
        for i in range(0, len(specs), batch_size):
            batch = specs[i : i + batch_size]
            try:
                self._pip_install(batch)
            except DependencyInstallError as e:
                logger.warning("Batch install failed, trying individually: %s", e)
                for spec in batch:
                    try:
                        self._pip_install([spec])
                    except DependencyInstallError:
                        logger.warning("Failed to install %s, skipping", spec)

    # ── JS Dependency Installation ──

    def _install_js_dependencies(self):
        """Install Node.js dependencies in the project copy if package.json exists."""
        logger.debug("[JS-DEPS] _install_js_dependencies called, project_copy_dir=%s", self.project_copy_dir)
        if not self.project_copy_dir:
            logger.debug("[JS-DEPS] No project_copy_dir, skipping")
            return

        pkg_json = self.project_copy_dir / "package.json"
        logger.debug("[JS-DEPS] Checking for package.json at %s — exists=%s", pkg_json, pkg_json.is_file())
        if not pkg_json.is_file():
            logger.debug("[JS-DEPS] No package.json found, skipping")
            return

        lock_file = self.project_copy_dir / "package-lock.json"
        if lock_file.is_file():
            cmd = ["npm", "ci", "--ignore-scripts"]
        else:
            cmd = ["npm", "install", "--ignore-scripts"]

        logger.debug("[JS-DEPS] Running command: %s in cwd=%s", cmd, self.project_copy_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_copy_dir),
            )
            logger.debug("[JS-DEPS] Exit code: %d", result.returncode)
            logger.debug("[JS-DEPS] stdout:\n%s", result.stdout[:2000])
            logger.debug("[JS-DEPS] stderr:\n%s", result.stderr[:2000])
            if result.returncode != 0:
                logger.warning(
                    "npm install failed (exit %d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
            else:
                logger.info("JS dependencies installed via %s", cmd[1])
        except FileNotFoundError:
            logger.warning("npm not found on PATH, skipping JS dependency installation")
        except subprocess.TimeoutExpired:
            logger.warning("npm install timed out after 120s")

    # ── Project Copy ──

    def _copy_project(self):
        """Copy user's project into the workspace. Original files are never touched."""

        def _ignore(directory, contents):
            ignored = set()
            for item in contents:
                if item in COPY_EXCLUDE_PATTERNS:
                    ignored.add(item)
                elif any(item.endswith(sfx) for sfx in COPY_EXCLUDE_SUFFIXES):
                    ignored.add(item)
            return ignored

        shutil.copytree(self.project_path, self.project_copy_dir, ignore=_ignore)

    # ── Command Execution ──

    def run_in_session(
        self,
        command: list[str],
        timeout: Optional[int] = None,
        env_vars: Optional[dict] = None,
    ) -> SessionResult:
        """Run a command inside the session's venv with resource caps.

        The command runs in the project copy directory using the venv's Python.
        Resource caps (memory, process limits) are enforced via preexec_fn.
        """
        if not self._setup_complete:
            raise SessionError(
                "Session not set up. Call setup() or use as context manager."
            )

        timeout = timeout or self.resource_caps.timeout_seconds

        # Build environment with venv activated
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(self.venv_dir)
        env["PATH"] = (
            str(self.venv_python.parent) + os.pathsep + env.get("PATH", "")
        )
        env.pop("PYTHONHOME", None)  # interferes with venv
        if env_vars:
            env.update(env_vars)

        # Resolve "python" to venv python
        resolved_command = list(command)
        if resolved_command and resolved_command[0] == "python":
            resolved_command[0] = str(self.venv_python)

        logger.info(
            "run_in_session: %s (timeout=%ss)",
            resolved_command[0].rsplit("/", 1)[-1] if resolved_command else "?",
            timeout,
        )
        sys.stderr.flush()

        try:
            result = subprocess.run(
                resolved_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_copy_dir),
                env=env,
                preexec_fn=self._make_preexec_fn(),
            )
            return SessionResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as e:
            stdout = ""
            stderr = ""
            if e.stdout:
                stdout = (
                    e.stdout
                    if isinstance(e.stdout, str)
                    else e.stdout.decode("utf-8", errors="replace")
                )
            if e.stderr:
                stderr = (
                    e.stderr
                    if isinstance(e.stderr, str)
                    else e.stderr.decode("utf-8", errors="replace")
                )
            return SessionResult(
                returncode=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
        except OSError as e:
            return SessionResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
            )

    def _make_preexec_fn(self):
        """Create a preexec function that sets resource limits for the child process."""
        caps = self.resource_caps

        def _set_limits():
            try:
                import resource as res

                memory_bytes = caps.memory_mb * 1024 * 1024

                # Memory limit — try platform-appropriate options
                for limit_name in ("RLIMIT_AS", "RLIMIT_RSS", "RLIMIT_DATA"):
                    limit_attr = getattr(res, limit_name, None)
                    if limit_attr is not None:
                        try:
                            res.setrlimit(limit_attr, (memory_bytes, memory_bytes))
                            break
                        except (ValueError, OSError):
                            continue

                # Process limit
                nproc = getattr(res, "RLIMIT_NPROC", None)
                if nproc is not None:
                    try:
                        res.setrlimit(
                            nproc, (caps.process_limit, caps.process_limit)
                        )
                    except (ValueError, OSError):
                        pass
            except ImportError:
                pass  # resource module not available (Windows)

        return _set_limits

    # ── Marker File ──

    def _write_marker(self):
        """Write a marker file for orphan detection."""
        marker = self.workspace_dir / self.MARKER_FILE
        marker.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "created": time.time(),
                    "project": str(self.project_path),
                    "python_version": platform.python_version(),
                }
            )
        )

    # ── Signal Handling & Session Registry ──

    def _register_session(self):
        """Register this session for signal-based cleanup."""
        SessionManager._active_sessions.append(self)
        if not SessionManager._signal_handlers_installed:
            SessionManager._install_signal_handlers()
        atexit.register(self.teardown)

    def _unregister_session(self):
        """Unregister this session from signal-based cleanup."""
        try:
            SessionManager._active_sessions.remove(self)
        except ValueError:
            pass
        try:
            atexit.unregister(self.teardown)
        except Exception:
            pass
        if not SessionManager._active_sessions:
            SessionManager._restore_signal_handlers()

    @classmethod
    def _install_signal_handlers(cls):
        """Install signal handlers that trigger cleanup on interrupt."""
        cls._original_sigint = signal.getsignal(signal.SIGINT)
        cls._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, cls._signal_handler)
        signal.signal(signal.SIGTERM, cls._signal_handler)
        cls._signal_handlers_installed = True

    @classmethod
    def _restore_signal_handlers(cls):
        """Restore original signal handlers."""
        if cls._signal_handlers_installed:
            if cls._original_sigint is not None:
                signal.signal(signal.SIGINT, cls._original_sigint)
            if cls._original_sigterm is not None:
                signal.signal(signal.SIGTERM, cls._original_sigterm)
            cls._signal_handlers_installed = False
            cls._original_sigint = None
            cls._original_sigterm = None

    @classmethod
    def _signal_handler(cls, signum, frame):
        """Handle interrupt signals by cleaning up all active sessions."""
        logger.warning("Received signal %s, cleaning up sessions...", signum)
        for session in list(cls._active_sessions):
            session.teardown()

        cls._restore_signal_handlers()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            sys.exit(128 + signum)

    # ── Orphan Cleanup ──

    @classmethod
    def cleanup_orphans(
        cls,
        temp_base: Optional[str | Path] = None,
        max_age_hours: float = 24,
    ) -> int:
        """Find and remove orphaned mycode session directories from previous runs.

        Called at startup to clean up after crashes or interrupted sessions.
        Only removes directories older than max_age_hours unless the creating
        process is no longer running.

        Returns the number of orphaned directories cleaned.
        """
        search_dir = Path(temp_base) if temp_base else Path(tempfile.gettempdir())
        cleaned = 0

        if not search_dir.exists():
            return cleaned

        now = time.time()
        max_age_seconds = max_age_hours * 3600

        try:
            for entry in search_dir.iterdir():
                if not entry.is_dir() or not entry.name.startswith(cls.TEMP_PREFIX):
                    continue

                marker = entry / cls.MARKER_FILE
                should_clean = False

                if marker.exists():
                    try:
                        data = json.loads(marker.read_text())
                        created = data.get("created", 0)
                        pid = data.get("pid", -1)

                        age = now - created
                        if age > max_age_seconds:
                            should_clean = True
                        elif not _is_process_running(pid):
                            should_clean = True
                    except (json.JSONDecodeError, OSError):
                        try:
                            dir_age = now - entry.stat().st_mtime
                            if dir_age > max_age_seconds:
                                should_clean = True
                        except OSError:
                            pass
                else:
                    # No marker — check directory age
                    try:
                        dir_age = now - entry.stat().st_mtime
                        if dir_age > max_age_seconds:
                            should_clean = True
                    except OSError:
                        pass

                if should_clean:
                    try:
                        shutil.rmtree(entry)
                        cleaned += 1
                        logger.info("Cleaned orphaned session: %s", entry)
                    except OSError as e:
                        logger.warning("Failed to clean orphan %s: %s", entry, e)
        except OSError as e:
            logger.warning("Failed to scan for orphans in %s: %s", search_dir, e)

        return cleaned


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it
