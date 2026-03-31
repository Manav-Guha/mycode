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
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.ingester import _read_text_safe

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

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
    ".claude",
}

# Directory name prefixes to exclude (matched with str.startswith)
COPY_EXCLUDE_PREFIXES = ("pytest-of-",)

COPY_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}

# Dependency file names to search for (in project root and one level of subdirs)
# Overall time budget for pip dependency installation (seconds).
# Generous for normal installs (10-30s) but catches native compilation
# hangs (llama-cpp-python, grpcio from source, etc.).
_DEP_INSTALL_TIMEOUT = 120

_PY_DEP_FILENAMES = (
    "requirements.txt", "requirement.txt",
    "requirements-dev.txt", "requirements_dev.txt",
    "pyproject.toml", "setup.py", "setup.cfg",
)

_JS_DEP_FILENAMES = (
    "package.json",
)

_DEP_FILENAMES = _PY_DEP_FILENAMES + _JS_DEP_FILENAMES


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
        self._active_process: Optional[subprocess.Popen] = None

        # JS dependency install tracking
        self.js_deps_installed: Optional[bool] = None  # None=not JS, True=success, False=failed
        self.js_deps_error: str = ""  # Error message if install failed

        # Dependency install warnings (surfaced to pipeline)
        self.dep_install_warnings: list[str] = []
        # Per-package install strategy tracking
        self.dep_install_results: dict[str, str] = {}
        # package → "installed" | "installed-binary-only" | "installed-no-deps" | "failed"

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
            # In containerised mode, deps are pre-installed during docker build
            if os.environ.get("MYCODE_CONTAINERISED") == "1":
                logger.info("Containerised mode — skipping dependency install "
                            "(pre-installed during docker build)")
            else:
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

        # Kill any in-flight child process tree before removing files
        process_confirmed_dead = True
        proc = self._active_process
        if proc is not None:
            process_confirmed_dead = self._kill_process_tree(proc)
            self._active_process = None

        self._unregister_session()

        if not process_confirmed_dead:
            # Process may still be writing — leave workspace intact for
            # orphan cleanup on next startup rather than risk deleting
            # files out from under a running process.
            logger.error(
                "Subprocess did not exit after forced termination; leaving "
                "workspace intact for orphan cleanup: %s", self.workspace_dir,
            )
        elif self.workspace_dir and self.workspace_dir.exists():
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

    def get_venv_packages(self) -> dict[str, str]:
        """Get installed packages from the session venv (not the host Python).

        Must be called after setup(). Returns {name: version} dict.
        """
        if not self.venv_python:
            return {}
        packages = {}
        try:
            result = subprocess.run(
                [str(self.venv_python), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"]] = pkg["version"]
            logger.debug("Venv has %d packages installed", len(packages))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to list venv packages: %s", e)
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
        """Create a virtual environment in the workspace.

        When running inside a Docker container (``MYCODE_CONTAINERISED=1``),
        enables ``system_site_packages`` so the venv inherits packages
        installed during ``docker build``.
        """
        containerised = os.environ.get("MYCODE_CONTAINERISED") == "1"
        try:
            venv.create(
                str(self.venv_dir),
                with_pip=True,
                clear=True,
                system_site_packages=containerised,
            )
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

    def find_dep_file_dir(
        self, filenames: tuple[str, ...] | None = None,
    ) -> Path:
        """Find the directory containing dependency files.

        Checks the project root first, then one level of subdirectories.
        If multiple subdirectories contain dep files, prefers the one with
        the most source files (.py, .js, .ts).  Returns the project root
        as fallback when no dep files are found anywhere.

        Args:
            filenames: Dep file names to search for.  Defaults to
                ``_DEP_FILENAMES`` (all languages).  Pass
                ``_PY_DEP_FILENAMES`` to avoid a root ``package.json``
                short-circuiting discovery of Python deps in subdirs.
        """
        names = filenames if filenames is not None else _DEP_FILENAMES
        root = self.project_copy_dir
        assert root is not None

        # 1. Check root — if any dep file exists, use root
        for name in names:
            if (root / name).is_file():
                return root

        # 2. Check one level of subdirectories
        candidates: list[tuple[Path, int]] = []
        try:
            children = sorted(root.iterdir())
        except OSError:
            return root

        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            has_dep_file = any(
                (child / name).is_file() for name in names
            )
            if has_dep_file:
                py_count = len(list(child.glob("*.py")))
                js_count = (
                    len(list(child.glob("*.js")))
                    + len(list(child.glob("*.ts")))
                )
                candidates.append((child, py_count + js_count))

        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            winner = candidates[0][0]
            logger.info(
                "[PY-DEPS] Dep files found in subdirectory: %s",
                winner.name,
            )
            return winner

        # 3. No dep files found — return root (fallback to env install)
        return root

    def _install_dependencies(self):
        """Install the user's dependencies into the venv from the project copy.

        Enforces an overall time budget of ``_DEP_INSTALL_TIMEOUT`` seconds.
        If the budget is exceeded, installation stops and a warning is
        recorded — the pipeline continues with whatever deps are available.
        """
        logger.debug("[PY-DEPS] _install_dependencies called, project_copy_dir=%s", self.project_copy_dir)
        if not self.project_copy_dir:
            logger.debug("[PY-DEPS] No project_copy_dir, skipping")
            return

        deadline = time.monotonic() + _DEP_INSTALL_TIMEOUT

        # Find the directory containing Python dep files (root or one subdir down).
        # Pass _PY_DEP_FILENAMES so a root package.json doesn't short-circuit
        # discovery of Python deps in subdirectories (monorepo case).
        dep_dir = self.find_dep_file_dir(filenames=_PY_DEP_FILENAMES)
        installed_any = False

        def _budget_exceeded() -> bool:
            if time.monotonic() > deadline:
                logger.warning(
                    "[PY-DEPS] Dependency install time budget (%ds) exceeded",
                    _DEP_INSTALL_TIMEOUT,
                )
                self.dep_install_warnings.append(
                    "Some dependencies could not be installed within the "
                    "time limit. Test results may be limited."
                )
                return True
            return False

        # Try requirements*.txt files (most common for vibe-coded projects)
        for req_file in [
            "requirements.txt",
            "requirement.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
        ]:
            if _budget_exceeded():
                break
            req_path = dep_dir / req_file
            logger.debug("[PY-DEPS] Checking %s — exists=%s", req_file, req_path.is_file())
            if req_path.is_file():
                try:
                    logger.debug("[PY-DEPS] Installing from %s", req_file)
                    self._pip_install(["-r", str(req_path)], deadline=deadline)
                    installed_any = True
                    logger.debug("[PY-DEPS] Successfully installed from %s", req_file)
                except DependencyInstallError as e:
                    logger.warning("[PY-DEPS] Failed to install from %s: %s", req_file, e)
                    if _budget_exceeded():
                        break
                    logger.debug("[PY-DEPS] Falling back to individual package install for %s", req_file)
                    count = self._pip_install_individually(req_path, deadline=deadline)
                    if count > 0:
                        installed_any = True
                    logger.debug("[PY-DEPS] Individual install from %s: %d packages succeeded", req_file, count)

        # Try pyproject.toml (install from dep dir so originals are untouched)
        pyproject = dep_dir / "pyproject.toml"
        logger.debug("[PY-DEPS] Checking pyproject.toml — exists=%s", pyproject.is_file())
        if pyproject.is_file() and not installed_any and not _budget_exceeded():
            try:
                logger.debug("[PY-DEPS] Installing from pyproject.toml")
                self._pip_install([str(dep_dir)], deadline=deadline)
                installed_any = True
                logger.debug("[PY-DEPS] Successfully installed from pyproject.toml")
            except DependencyInstallError as e:
                logger.warning(
                    "[PY-DEPS] pip install . failed for pyproject.toml: %s", e
                )
                parsed = self._parse_pyproject_deps(pyproject)
                if parsed:
                    logger.info(
                        "[PY-DEPS] Parsed %d dependencies from pyproject.toml, "
                        "installing individually", len(parsed),
                    )
                    for bare_name, full_spec in parsed:
                        if _budget_exceeded():
                            break
                        self._pip_install_with_fallback(full_spec, deadline=deadline)
                    installed_any = True

        # Try setup.py
        setup_py = dep_dir / "setup.py"
        logger.debug("[PY-DEPS] Checking setup.py — exists=%s", setup_py.is_file())
        if setup_py.is_file() and not installed_any and not _budget_exceeded():
            try:
                logger.debug("[PY-DEPS] Installing from setup.py")
                self._pip_install([str(dep_dir)], deadline=deadline)
                installed_any = True
                logger.debug("[PY-DEPS] Successfully installed from setup.py")
            except DependencyInstallError as e:
                logger.warning(
                    "[PY-DEPS] Failed to install from setup.py, falling back to package list: %s", e
                )

        # Fallback: install from detected environment package list
        if (
            not installed_any
            and not _budget_exceeded()
            and self.environment_info
            and self.environment_info.installed_packages
        ):
            logger.info(
                "[PY-DEPS] No dependency files found, installing from detected environment"
            )
            self._install_from_package_list(
                self.environment_info.installed_packages, deadline=deadline,
            )

        logger.debug("[PY-DEPS] Installation complete: installed_any=%s", installed_any)

        # Surface failed deps summary
        failed = [
            pkg for pkg, status in self.dep_install_results.items()
            if status == "failed"
        ]
        if failed:
            names = ", ".join(failed[:10])
            suffix = "..." if len(failed) > 10 else ""
            n = len(failed)
            self.dep_install_warnings.append(
                f"{n} dependenc{'y' if n == 1 else 'ies'} could not be "
                f"installed in the test environment: {names}{suffix}"
            )
            logger.warning("[PY-DEPS] %d package(s) failed all strategies: %s", n, names)

    @staticmethod
    def _parse_pyproject_deps(pyproject_path: Path) -> list[tuple[str, str]]:
        """Parse ``[project.dependencies]`` from a pyproject.toml.

        Returns a list of ``(bare_name, full_spec)`` tuples.
        ``full_spec`` is the original string (e.g.
        ``"fastapi[standard]>=0.135.0,<0.136.0"``), passed to pip as-is.
        ``bare_name`` is for logging/tracking only.
        """
        try:
            data = tomllib.loads(_read_text_safe(pyproject_path))
        except Exception as exc:
            logger.warning("[PY-DEPS] Could not parse pyproject.toml: %s", exc)
            return []

        deps = data.get("project", {}).get("dependencies", [])
        if not isinstance(deps, list):
            return []

        # PEP 508 name: letters, digits, hyphens, dots, underscores
        _NAME_RE = re.compile(r"^([A-Za-z0-9][-A-Za-z0-9_.]*)")
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for spec in deps:
            spec = spec.strip()
            if not spec:
                continue
            m = _NAME_RE.match(spec)
            if not m:
                continue
            bare = m.group(1).lower()
            if bare not in seen:
                seen.add(bare)
                result.append((bare, spec))
        return result

    def _pip_install(self, args: list[str], deadline: float | None = None):
        """Run pip install with given arguments inside the venv.

        When *deadline* is provided, the per-call timeout is reduced to
        fit within the remaining budget.
        """
        per_call_timeout = 60
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DependencyInstallError(
                    "Dependency install time budget exceeded"
                )
            per_call_timeout = min(per_call_timeout, max(5, int(remaining)))

        cmd = [str(self.venv_python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check"] + args
        logger.info("[PY-DEPS] pip install: %s", " ".join(args)[:200])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=per_call_timeout,
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

    @staticmethod
    def _is_version_incompatible(error_msg: str) -> bool:
        """Detect Python/Node version incompatibility from pip stderr."""
        markers = [
            "requires python",
            "python_requires",
            "requires-python",
            "not supported on this platform",
            "incompatible with this version",
            "python version",
        ]
        lower = error_msg.lower()
        return any(m in lower for m in markers)

    def _pip_install_with_fallback(
        self, package: str, deadline: float | None = None,
    ) -> str:
        """Try up to three install strategies for a single package.

        Returns the strategy that succeeded: ``"installed"``,
        ``"installed-binary-only"``, ``"installed-no-deps"``, or ``"failed"``.

        Annotation guards:
        - If remaining budget < 10s before strategy 2 or 3, skip and record
          as ``"failed"`` immediately.
        - If strategy 1 fails with a Python version incompatibility error,
          skip strategies 2 and 3 (they'll fail identically).
        """
        # Strategy 1: normal install
        try:
            self._pip_install([package], deadline=deadline)
            self.dep_install_results[package] = "installed"
            return "installed"
        except DependencyInstallError as exc:
            stderr = str(exc)
            if self._is_version_incompatible(stderr):
                logger.warning(
                    "[PY-DEPS] %s: version-incompatible, skipping fallbacks", package,
                )
                self.dep_install_results[package] = "failed"
                return "failed"

        # Strategy 2: pre-built wheel only (skip if <10s remaining)
        if deadline is not None and (deadline - time.monotonic()) < 10:
            logger.info("[PY-DEPS] %s: <10s remaining, skipping fallbacks", package)
            self.dep_install_results[package] = "failed"
            return "failed"
        try:
            self._pip_install([package, "--only-binary=:all:"], deadline=deadline)
            logger.info("[PY-DEPS] %s installed via binary-only fallback", package)
            self.dep_install_results[package] = "installed-binary-only"
            return "installed-binary-only"
        except DependencyInstallError:
            pass

        # Strategy 3: no transitive deps (skip if <10s remaining)
        if deadline is not None and (deadline - time.monotonic()) < 10:
            logger.info("[PY-DEPS] %s: <10s remaining, skipping no-deps fallback", package)
            self.dep_install_results[package] = "failed"
            return "failed"
        try:
            self._pip_install([package, "--no-deps"], deadline=deadline)
            logger.info("[PY-DEPS] %s installed via no-deps fallback", package)
            self.dep_install_results[package] = "installed-no-deps"
            return "installed-no-deps"
        except DependencyInstallError:
            logger.warning("[PY-DEPS] %s failed all install strategies", package)
            self.dep_install_results[package] = "failed"
            return "failed"

    def _pip_install_individually(
        self, req_path: Path, deadline: float | None = None,
    ) -> int:
        """Parse a requirements file and install packages one at a time.

        Returns the number of successfully installed packages.
        """
        successful = 0
        try:
            lines = _read_text_safe(req_path).splitlines()
        except OSError as e:
            logger.warning("[PY-DEPS] Could not read %s: %s", req_path, e)
            return 0

        for line in lines:
            line = line.strip()
            # Skip blanks, comments, and pip options (e.g. -i, --index-url, -f)
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if deadline is not None and time.monotonic() > deadline:
                logger.warning("[PY-DEPS] Budget exceeded, skipping remaining packages")
                break
            logger.debug("[PY-DEPS] Installing individual package: %s", line)
            result = self._pip_install_with_fallback(line, deadline=deadline)
            if result != "failed":
                successful += 1
        return successful

    def _install_from_package_list(
        self, packages: dict[str, str], deadline: float | None = None,
    ):
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
            if deadline is not None and time.monotonic() > deadline:
                logger.warning("[PY-DEPS] Budget exceeded, skipping remaining batches")
                break
            batch = specs[i : i + batch_size]
            try:
                self._pip_install(batch, deadline=deadline)
            except DependencyInstallError as e:
                logger.warning("Batch install failed, trying individually: %s", e)
                for spec in batch:
                    if deadline is not None and time.monotonic() > deadline:
                        logger.warning("[PY-DEPS] Budget exceeded, skipping remaining packages")
                        break
                    self._pip_install_with_fallback(spec, deadline=deadline)

    # ── JS Dependency Installation ──

    def _install_js_dependencies(self):
        """Install Node.js dependencies in the project copy if package.json exists.

        Sets self.js_deps_installed to True/False and self.js_deps_error on failure.
        Downstream stages (server_manager, HTTP testing) check these before attempting
        to start a server.
        """
        logger.debug("[JS-DEPS] _install_js_dependencies called, project_copy_dir=%s", self.project_copy_dir)
        if not self.project_copy_dir:
            logger.debug("[JS-DEPS] No project_copy_dir, skipping")
            return

        # Check root first, then dep-file subdirectory for package.json
        dep_dir = self.find_dep_file_dir()
        pkg_json = dep_dir / "package.json"
        logger.debug("[JS-DEPS] Checking for package.json at %s — exists=%s", pkg_json, pkg_json.is_file())
        if not pkg_json.is_file():
            logger.debug("[JS-DEPS] No package.json found, skipping")
            return

        # If node_modules already present (e.g. user pre-installed), skip
        node_modules = dep_dir / "node_modules"
        if node_modules.is_dir() and any(node_modules.iterdir()):
            logger.info("node_modules already exists, skipping npm install")
            self.js_deps_installed = True
            return

        lock_file = dep_dir / "package-lock.json"
        if lock_file.is_file():
            cmd = ["npm", "ci"]
        else:
            cmd = ["npm", "install"]

        # NODE_ENV=development ensures devDependencies are installed
        # (many frameworks like Next.js are devDependencies)
        env = os.environ.copy()
        env["NODE_ENV"] = "development"

        logger.debug("[JS-DEPS] Running command: %s in cwd=%s", cmd, dep_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(dep_dir),
                env=env,
            )
            logger.debug("[JS-DEPS] Exit code: %d", result.returncode)
            logger.debug("[JS-DEPS] stdout:\n%s", result.stdout[:2000])
            logger.debug("[JS-DEPS] stderr:\n%s", result.stderr[:2000])

            # Retry with workspace-local cache if default cache has permission issues
            if result.returncode != 0 and "EPERM" in (result.stderr or ""):
                logger.info("npm cache permission error — retrying with workspace-local cache")
                local_cache = self.workspace_dir / ".npm_cache"
                retry_cmd = cmd + ["--cache", str(local_cache)]
                result = subprocess.run(
                    retry_cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(dep_dir),
                    env=env,
                )

            if result.returncode != 0:
                self.js_deps_installed = False
                self.js_deps_error = result.stderr[:500] or result.stdout[:500]
                logger.warning(
                    "npm install failed (exit %d): %s",
                    result.returncode,
                    self.js_deps_error,
                )
            else:
                self.js_deps_installed = True
                logger.info("JS dependencies installed via %s", cmd[1])
        except FileNotFoundError:
            self.js_deps_installed = False
            self.js_deps_error = "npm not found on PATH"
            logger.warning("npm not found on PATH, skipping JS dependency installation")
        except subprocess.TimeoutExpired:
            self.js_deps_installed = False
            self.js_deps_error = "npm install timed out after 120 seconds"
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
                elif item.startswith(COPY_EXCLUDE_PREFIXES):
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

        # Add dep-file subdirectory to PYTHONPATH so imports work
        # when code lives in a subdirectory (e.g. "My Projects/")
        dep_dir = str(self.find_dep_file_dir())
        project_root = str(self.project_copy_dir)
        pythonpath_parts = [project_root]
        if dep_dir != project_root:
            pythonpath_parts.append(dep_dir)
        existing_pypath = env.get("PYTHONPATH", "")
        if existing_pypath:
            pythonpath_parts.append(existing_pypath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

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

        proc = None
        try:
            popen_kwargs: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "cwd": str(self.project_copy_dir),
                "env": env,
            }
            if _IS_WINDOWS:
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                popen_kwargs["preexec_fn"] = self._make_preexec_fn()
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(resolved_command, **popen_kwargs)
            self._active_process = proc
            stdout, stderr = proc.communicate(timeout=timeout)
            self._active_process = None
            return SessionResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            self._kill_process_tree(proc)
            self._active_process = None
            stdout, stderr = "", ""
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            return SessionResult(
                returncode=-1,
                stdout=stdout or "",
                stderr=stderr or "",
                timed_out=True,
            )
        except OSError as e:
            if proc is not None:
                self._kill_process_tree(proc)
                self._active_process = None
            return SessionResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
            )

    @staticmethod
    def _kill_process_tree(proc: subprocess.Popen) -> bool:
        """Kill the entire process tree rooted at *proc*.

        On POSIX: we launch children with ``start_new_session=True`` so every
        child and grandchild shares the same process-group ID (the child's
        PID).  ``os.killpg`` sends a signal to every process in that group.

        On Windows: we launch children with ``CREATE_NEW_PROCESS_GROUP``, then
        use ``taskkill /T /F /PID`` to kill the entire process tree.

        Returns True if the process was confirmed dead, False if it may
        still be running.
        """
        if proc is None or proc.poll() is not None:
            return True  # already exited

        if _IS_WINDOWS:
            return SessionManager._kill_process_tree_win32(proc)
        return SessionManager._kill_process_tree_posix(proc)

    @staticmethod
    def _kill_process_tree_posix(proc: subprocess.Popen) -> bool:
        """POSIX: SIGTERM the process group, then SIGKILL if needed."""
        pgid: Optional[int] = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pass

        # 1. SIGTERM the group (graceful)
        if pgid is not None and pgid > 0:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                pass
        else:
            try:
                proc.terminate()
            except OSError:
                pass

        # Give processes a brief window to exit cleanly
        try:
            proc.wait(timeout=3)
            return True
        except subprocess.TimeoutExpired:
            pass

        # 2. SIGKILL the group (forceful)
        if pgid is not None and pgid > 0:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        else:
            try:
                proc.kill()
            except OSError:
                pass

        try:
            proc.wait(timeout=3)
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Process %d did not exit after SIGKILL", proc.pid)
            return False

    @staticmethod
    def _kill_process_tree_win32(proc: subprocess.Popen) -> bool:
        """Windows: use taskkill /T /F to kill the entire process tree."""
        # 1. Graceful: terminate the root process
        try:
            proc.terminate()
        except OSError:
            pass

        try:
            proc.wait(timeout=3)
            return True
        except subprocess.TimeoutExpired:
            pass

        # 2. Forceful: taskkill /T (tree) /F (force) /PID
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

        try:
            proc.wait(timeout=3)
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                "Process %d did not exit after taskkill /T /F", proc.pid
            )
            return False

    def _make_preexec_fn(self):
        """Create a preexec function that sets resource limits for the child process."""
        caps = self.resource_caps

        def _set_limits():
            try:
                import resource as res

                memory_bytes = caps.memory_mb * 1024 * 1024

                # Memory limit — set every available rlimit so the
                # tightest one that the OS actually enforces wins.
                memory_limit_set = False
                for limit_name in ("RLIMIT_AS", "RLIMIT_RSS", "RLIMIT_DATA"):
                    limit_attr = getattr(res, limit_name, None)
                    if limit_attr is not None:
                        try:
                            res.setrlimit(limit_attr, (memory_bytes, memory_bytes))
                            memory_limit_set = True
                        except (ValueError, OSError):
                            continue

                if not memory_limit_set:
                    import sys as _sys
                    print(
                        "mycode: WARNING — could not set any memory rlimit "
                        f"(RLIMIT_AS/RSS/DATA) to {caps.memory_mb} MB; "
                        "memory cap will NOT be enforced by the OS",
                        file=_sys.stderr,
                    )

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
        """Install signal handlers that trigger cleanup on interrupt.

        On Windows only SIGINT (Ctrl+C) is reliably supported; SIGTERM
        cannot be caught via ``signal.signal``.  We install SIGTERM only
        on POSIX systems.

        signal.signal() can only be called from the main thread.  When
        running inside a web server (e.g. uvicorn worker threads), skip
        signal registration — atexit cleanup still works.
        """
        if threading.current_thread() is not threading.main_thread():
            logger.debug(
                "Skipping signal handler installation — not on main thread"
            )
            return
        cls._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, cls._signal_handler)
        if not _IS_WINDOWS:
            cls._original_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, cls._signal_handler)
        cls._signal_handlers_installed = True

    @classmethod
    def _restore_signal_handlers(cls):
        """Restore original signal handlers."""
        if cls._signal_handlers_installed:
            if cls._original_sigint is not None:
                signal.signal(signal.SIGINT, cls._original_sigint)
            if not _IS_WINDOWS and cls._original_sigterm is not None:
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
    """Check if a process with the given PID is still running.

    On POSIX: uses ``os.kill(pid, 0)`` (signal 0 is a no-op existence check).
    On Windows: uses ``ctypes`` to call ``OpenProcess`` + ``GetExitCodeProcess``.
    """
    if pid <= 0:
        return False
    if _IS_WINDOWS:
        return _is_process_running_win32(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but we can't signal it


def _is_process_running_win32(pid: int) -> bool:
    """Windows implementation: query the process via the Win32 API."""
    try:
        import ctypes
        import ctypes.wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        return False
