"""Server Manager — HTTP stress testing Phase 1: server lifecycle management.

Starts supported web framework servers inside the session sandbox, polls for
readiness, and tears them down cleanly using process-group signals.  This is
the foundation for HTTP-level stress testing (Phase 2: endpoint discovery,
Phase 3: load driving).

Supported frameworks:
  Python  — Streamlit, FastAPI, Flask
  Node.js — Express, Next.js, React (Create React App), generic npm start

Pure Python.  No LLM dependency.
"""

import ast
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.ingester import IngestionResult, _read_text_safe
from mycode.session import SessionManager

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# ── Framework Constants ──

# Default startup timeout — if the server isn't ready by this, it's a finding
STARTUP_TIMEOUT_SECONDS = 30

# Per-framework startup timeouts.  CRA/React and Next.js dev servers
# compile on cold start which can take 2-4 minutes.
_FRAMEWORK_STARTUP_TIMEOUTS: dict[str, int] = {
    "react-scripts": 120,
    "nextjs": 120,
    "npm-start": 30,
}

# Health-check poll interval
POLL_INTERVAL_SECONDS = 0.5

# Graceful shutdown grace period before SIGKILL
SHUTDOWN_GRACE_SECONDS = 5

# Preferred entry-file names, in priority order
_PREFERRED_ENTRY_FILES = ("app.py", "main.py", "server.py", "index.js", "app.js", "server.js")


# ── Data Classes ──


@dataclass
class ServerInfo:
    """Information about a running server process."""

    process: subprocess.Popen
    port: int
    framework: str
    entry_file: str  # relative to project root
    startup_time: float = 0.0  # seconds to reach readiness


@dataclass
class ServerStartResult:
    """Result of attempting to start a server."""

    success: bool
    server: Optional[ServerInfo] = None
    error: str = ""


@dataclass
class FrameworkDetection:
    """Detected framework and entry point from project analysis."""

    framework: str  # "streamlit", "fastapi", "flask", "express"
    entry_file: str  # relative path to entry file
    app_variable: str = ""  # e.g. "app" for FastAPI/Flask
    module_name: str = ""  # e.g. "app" (without .py) for uvicorn


# ── Port Selection ──


def _find_available_port() -> int:
    """Find a random available TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Entry File Detection ──


def detect_framework_entry(
    ingestion: IngestionResult,
    project_dir: Path,
    language: str = "python",
) -> Optional[FrameworkDetection]:
    """Detect the server framework and entry file from ingestion results.

    Scans file analyses for framework-specific patterns:
      - Streamlit: files with ``st.`` calls not imported by others
      - FastAPI: file containing ``app = FastAPI()``
      - Flask: file containing ``app = Flask(__name__)``
      - Express: file containing ``express()`` and ``app.listen``

    Returns None if no supported server framework is detected.
    """
    if language == "python":
        return _detect_python_framework(ingestion, project_dir)
    if language in ("javascript", "js"):
        return _detect_js_framework(ingestion, project_dir)
    return None


def _detect_python_framework(
    ingestion: IngestionResult,
    project_dir: Path,
) -> Optional[FrameworkDetection]:
    """Detect Python server frameworks from ingestion data and source files."""
    # Collect candidates per framework
    fastapi_candidates: list[tuple[str, str, str]] = []  # (file, var, module)
    flask_candidates: list[tuple[str, str]] = []  # (file, var)
    streamlit_candidates: list[str] = []

    # Track which files are imported by others (for Streamlit heuristic)
    imported_modules: set[str] = set()
    for fa in ingestion.file_analyses:
        for imp in fa.imports:
            if imp.module:
                imported_modules.add(imp.module)
            for name in imp.names:
                imported_modules.add(name)

    for fa in ingestion.file_analyses:
        rel_path = fa.file_path
        abs_path = project_dir / rel_path

        if not abs_path.is_file():
            continue

        try:
            source = _read_text_safe(abs_path)
        except OSError:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        # Check for FastAPI: app = FastAPI()
        result = _find_app_assignment(tree, "FastAPI")
        if result:
            var_name, _ = result
            module = _file_to_module(rel_path)
            fastapi_candidates.append((rel_path, var_name, module))

        # Check for Flask: app = Flask(__name__)
        result = _find_app_assignment(tree, "Flask")
        if result:
            var_name, _ = result
            flask_candidates.append((rel_path, var_name))

        # Check for Streamlit: st. calls
        has_st_calls = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "st":
                    has_st_calls = True
                    break

        if has_st_calls:
            # Check if this file's module name appears in other files' imports
            file_module = _file_to_module(rel_path)
            file_stem = Path(rel_path).stem
            is_imported = (
                file_module in imported_modules or file_stem in imported_modules
            )
            if not is_imported:
                streamlit_candidates.append(rel_path)

    # Priority: FastAPI > Flask > Streamlit (more specific first)
    if fastapi_candidates:
        entry = _pick_preferred(
            [c[0] for c in fastapi_candidates]
        )
        for f, var, mod in fastapi_candidates:
            if f == entry:
                return FrameworkDetection(
                    framework="fastapi",
                    entry_file=entry,
                    app_variable=var,
                    module_name=mod,
                )

    if flask_candidates:
        entry = _pick_preferred([c[0] for c in flask_candidates])
        for f, var in flask_candidates:
            if f == entry:
                return FrameworkDetection(
                    framework="flask",
                    entry_file=entry,
                    app_variable=var,
                )

    if streamlit_candidates:
        entry = _pick_preferred(streamlit_candidates)
        return FrameworkDetection(
            framework="streamlit",
            entry_file=entry,
        )

    # ── Dependency-based fallback for Streamlit ──
    # AST detection can miss files that use `import streamlit` without
    # the `st.` alias, or files not in file_analyses.  If streamlit is
    # declared as a dependency, scan for any file that imports it.
    dep_names = {d.name for d in ingestion.dependencies}
    if "streamlit" in dep_names:
        for fa in ingestion.file_analyses:
            for imp in fa.imports:
                if imp.module == "streamlit" or (
                    imp.module and imp.module.startswith("streamlit.")
                ):
                    return FrameworkDetection(
                        framework="streamlit",
                        entry_file=fa.file_path,
                    )
        # Last resort — streamlit in deps, pick preferred entry file
        for preferred in _PREFERRED_ENTRY_FILES:
            if (project_dir / preferred).is_file():
                return FrameworkDetection(
                    framework="streamlit",
                    entry_file=preferred,
                )

    return None


def _detect_js_framework(
    ingestion: IngestionResult,
    project_dir: Path,
) -> Optional[FrameworkDetection]:
    """Detect JS server frameworks from ingestion data.

    Priority: Express > Next.js > CRA/React > generic npm start.
    """
    all_dep_names = {d.name for d in ingestion.dependencies}
    non_dev_names = {d.name for d in ingestion.dependencies if not d.is_dev}

    # ── Express.js ──  (checked first — explicit server, most specific)
    express_candidates: list[str] = []

    for fa in ingestion.file_analyses:
        rel_path = fa.file_path
        abs_path = project_dir / rel_path

        if not abs_path.is_file():
            continue

        try:
            source = _read_text_safe(abs_path)
        except OSError:
            continue

        # Look for express() call and app.listen
        has_express = bool(re.search(
            r"""(?:require\s*\(\s*['"]express['"]\s*\)|from\s+['"]express['"])""",
            source,
        ))
        has_listen = bool(re.search(r"\bapp\.listen\s*\(", source))

        if has_express and has_listen:
            express_candidates.append(rel_path)

    if express_candidates:
        entry = _pick_preferred(express_candidates)
        return FrameworkDetection(
            framework="express",
            entry_file=entry,
        )

    # ── Next.js ──
    # Detection: 'next' in deps AND (next.config.* exists OR pages/ OR app/ dir)
    if "next" in all_dep_names:
        has_next_config = any(
            (project_dir / cfg).exists()
            for cfg in ("next.config.js", "next.config.mjs", "next.config.ts")
        )
        has_pages = (project_dir / "pages").is_dir()
        has_app_dir = (project_dir / "app").is_dir()
        has_src_pages = (project_dir / "src" / "pages").is_dir()
        has_src_app = (project_dir / "src" / "app").is_dir()

        if has_next_config or has_pages or has_app_dir or has_src_pages or has_src_app:
            # Pick a representative entry file for metadata
            entry = "next.config.js"
            for candidate in (
                "next.config.js", "next.config.mjs", "next.config.ts",
                "pages/index.js", "pages/index.tsx",
                "app/page.js", "app/page.tsx",
                "src/pages/index.js", "src/pages/index.tsx",
                "src/app/page.js", "src/app/page.tsx",
            ):
                if (project_dir / candidate).exists():
                    entry = candidate
                    break
            return FrameworkDetection(
                framework="nextjs",
                entry_file=entry,
            )

    # ── React / Create React App ──
    # Detection: react-scripts in dependencies + src/index.js or src/App.js
    if "react-scripts" in non_dev_names:
        for entry_candidate in ("src/index.js", "src/App.js", "src/index.tsx", "src/App.tsx"):
            if (project_dir / entry_candidate).is_file():
                return FrameworkDetection(
                    framework="react-scripts",
                    entry_file=entry_candidate,
                )

    # ── Generic npm start fallback ──
    # If package.json has a "start" script and none of the above matched,
    # try npm start with PORT env var.
    pkg_json_path = project_dir / "package.json"
    if pkg_json_path.is_file():
        try:
            import json as _json
            pkg = _json.loads(_read_text_safe(pkg_json_path))
            scripts = pkg.get("scripts", {})
            if "start" in scripts or "dev" in scripts:
                return FrameworkDetection(
                    framework="npm-start",
                    entry_file="package.json",
                )
        except Exception:
            pass

    return None


def _find_app_assignment(
    tree: ast.AST, class_name: str
) -> Optional[tuple[str, int]]:
    """Find ``var = ClassName(...)`` assignment in AST.

    Returns (variable_name, lineno) or None.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func_name = ""
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        if func_name != class_name:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                return (target.id, node.lineno)
    return None


def _file_to_module(rel_path: str) -> str:
    """Convert ``dir/app.py`` to ``dir.app`` (dotted module path)."""
    p = Path(rel_path)
    parts = list(p.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = p.stem
    return ".".join(parts) if parts else ""


def _pick_preferred(candidates: list[str]) -> str:
    """Pick the best entry file from candidates, preferring well-known names."""
    if len(candidates) == 1:
        return candidates[0]
    for preferred in _PREFERRED_ENTRY_FILES:
        for c in candidates:
            if Path(c).name == preferred:
                return c
    return candidates[0]


# ── Startup Command Builders ──


def build_startup_command(
    detection: FrameworkDetection,
    port: int,
) -> tuple[list[str], dict[str, str]]:
    """Build the startup command and extra env vars for a detected framework.

    Returns (command_list, extra_env_vars).
    """
    fw = detection.framework

    if fw == "streamlit":
        cmd = [
            "python", "-m", "streamlit", "run", detection.entry_file,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
        return cmd, {}

    if fw == "fastapi":
        module_app = f"{detection.module_name}:{detection.app_variable}"
        cmd = [
            "python", "-m", "uvicorn", module_app,
            "--port", str(port),
            "--host", "0.0.0.0",
        ]
        return cmd, {}

    if fw == "flask":
        cmd = [
            "python", "-m", "flask", "run",
            "--port", str(port),
            "--with-threads",
        ]
        env = {"FLASK_APP": detection.entry_file}
        return cmd, env

    if fw == "express":
        cmd = ["node", detection.entry_file]
        env = {"PORT": str(port)}
        return cmd, env

    if fw == "nextjs":
        cmd = ["npx", "next", "dev", "-p", str(port)]
        env = {"PORT": str(port), "NODE_ENV": "development"}
        return cmd, env

    if fw == "react-scripts":
        cmd = ["npx", "react-scripts", "start"]
        env = {"PORT": str(port), "BROWSER": "none"}
        return cmd, env

    if fw == "npm-start":
        cmd = ["npm", "start"]
        env = {"PORT": str(port)}
        return cmd, env

    raise ValueError(f"Unsupported framework: {fw}")


# ── Health Check URLs ──


def _health_check_url(framework: str, port: int) -> str:
    """Return the URL to poll for readiness."""
    if framework in ("fastapi", "flask", "express"):
        return f"http://localhost:{port}/health"
    # Streamlit, React, Next.js, and generic servers serve at root
    return f"http://localhost:{port}/"


def _check_health(url: str) -> bool:
    """Send a GET request and return True if the server responds (any status)."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return True
    except urllib.error.HTTPError:
        # Server responded with an error status — it's still alive
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _check_health_tcp(port: int) -> bool:
    """Fallback: check if a TCP connection can be established to the port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(("127.0.0.1", port))
            return True
    except (OSError, TimeoutError):
        return False


# ── Server Lifecycle ──


def start_server(
    session: SessionManager,
    detection: FrameworkDetection,
    timeout: Optional[int] = None,
) -> ServerStartResult:
    """Start a server process inside the session sandbox and poll for readiness.

    The server runs in a new process group (``start_new_session=True``) so the
    entire tree can be killed cleanly via ``os.killpg``.

    Args:
        session: An initialised SessionManager with setup() already called.
        detection: Framework detection result from ``detect_framework_entry``.
        timeout: Maximum seconds to wait for server readiness.  If None,
            uses a per-framework default (120s for CRA, 30s for others).

    Returns:
        ServerStartResult with success=True and a ServerInfo on success,
        or success=False with an error message on failure.
    """
    if timeout is None:
        timeout = _FRAMEWORK_STARTUP_TIMEOUTS.get(
            detection.framework, STARTUP_TIMEOUT_SECONDS
        )
    if not session._setup_complete:
        return ServerStartResult(
            success=False,
            error="Session not set up. Call setup() first.",
        )

    port = _find_available_port()
    cmd, extra_env = build_startup_command(detection, port)

    # Build environment with venv activated
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(session.venv_dir)
    env["PATH"] = (
        str(session.venv_python.parent) + os.pathsep + env.get("PATH", "")
    )
    env.pop("PYTHONHOME", None)
    env.update(extra_env)

    # Resolve commands that need the venv — Python frameworks use
    # "python -m <package>" so only the "python" prefix needs resolving.
    resolved_cmd = list(cmd)
    if resolved_cmd[0] == "python":
        resolved_cmd[0] = str(session.venv_python)

    logger.info(
        "Starting %s server on port %d: %s",
        detection.framework,
        port,
        " ".join(resolved_cmd[:4]),
    )

    try:
        popen_kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "cwd": str(session.project_copy_dir),
            "env": env,
        }
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(resolved_cmd, **popen_kwargs)
    except OSError as e:
        return ServerStartResult(
            success=False,
            error=f"Failed to start server process: {e}",
        )

    # Poll for readiness
    health_url = _health_check_url(detection.framework, port)
    start_time = time.monotonic()
    ready = False

    while time.monotonic() - start_time < timeout:
        # Check if process exited early (crash)
        if proc.poll() is not None:
            stderr = ""
            try:
                _, stderr_bytes = proc.communicate(timeout=2)
                stderr = stderr_bytes if isinstance(stderr_bytes, str) else stderr_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass
            reason = _diagnose_startup_failure(stderr, detection.framework)
            return ServerStartResult(
                success=False,
                error=f"Server could not start: {reason}",
            )

        # Try HTTP health check, then TCP fallback
        if _check_health(health_url) or _check_health_tcp(port):
            ready = True
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    if not ready:
        # Timeout — kill the process group
        kill_process_group(proc)
        stderr = ""
        try:
            _, stderr_bytes = proc.communicate(timeout=2)
            stderr = stderr_bytes if isinstance(stderr_bytes, str) else stderr_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass
        return ServerStartResult(
            success=False,
            error=(
                f"Server did not become ready within {timeout}s. "
                f"Last stderr: {stderr[:500]}"
            ),
        )

    startup_time = time.monotonic() - start_time
    logger.info(
        "%s server ready on port %d in %.1fs",
        detection.framework, port, startup_time,
    )

    return ServerStartResult(
        success=True,
        server=ServerInfo(
            process=proc,
            port=port,
            framework=detection.framework,
            entry_file=detection.entry_file,
            startup_time=startup_time,
        ),
    )


# Framework package prefixes — used to distinguish "missing dep" from
# "installed but crashes on import" when the ImportError is inside the
# framework's own package tree.
_FRAMEWORK_PACKAGES: dict[str, set[str]] = {
    "streamlit": {"streamlit"},
    "fastapi": {"fastapi", "starlette"},
    "flask": {"flask", "werkzeug", "jinja2"},
    "express": set(),
    "nextjs": set(),
}


def _diagnose_startup_failure(stderr: str, framework: str) -> str:
    """Translate raw stderr into a human-readable startup failure reason."""
    stderr_lower = stderr.lower()

    if "address already in use" in stderr_lower or "eaddrinuse" in stderr_lower:
        return "port conflict — another process is using the assigned port"

    # Import error INSIDE the framework package (installed but crashes).
    # Must come before the generic "no module named" check so that
    # e.g. "from streamlit.proto.X import Y" is not misclassified as
    # "missing dependency: streamlit".
    if "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
        # Look for "from <pkg>.submodule import" or "import <pkg>.submodule"
        m = re.search(
            r"(?:from|import)\s+(\w+)\.\S+",
            stderr,
        )
        if m:
            failing_pkg = m.group(1).lower()
            known = _FRAMEWORK_PACKAGES.get(framework, set())
            if failing_pkg == framework.lower() or failing_pkg in known:
                return (
                    f"{framework} is installed but crashes on import "
                    f"— likely incompatible with this Python version"
                )

    if "no module named" in stderr_lower or "modulenotfounderror" in stderr_lower:
        # Extract the module name
        m = re.search(r"no module named ['\"]?(\S+)['\"]?", stderr_lower)
        mod = m.group(1) if m else "unknown"
        return f"missing dependency: {mod}"

    if any(kw in stderr_lower for kw in ("api_key", "api key", "secret_key", "secret key")):
        return "missing API key or secret — the app requires environment variables"

    if any(kw in stderr_lower for kw in (
        "connection refused", "could not connect", "operational error",
        "connection error", "database",
    )):
        return "missing database or external service connection"

    if "permission denied" in stderr_lower:
        return "permission denied"

    if "syntaxerror" in stderr_lower:
        return "syntax error in application code"

    if stderr.strip():
        lines = stderr.strip().splitlines()
        # If stderr contains a traceback, the actual exception is the
        # last non-empty line (e.g. "ValueError: invalid literal...").
        if any(l.strip().startswith("Traceback") for l in lines):
            for line in reversed(lines):
                line = line.strip()
                if line:
                    return line[:200]
        # No traceback — return first non-empty line.
        for line in lines:
            line = line.strip()
            if line:
                return line[:200]

    return f"{framework} server process exited without output"


# ── Graceful Teardown ──


def kill_process_group(
    proc: subprocess.Popen,
    grace_seconds: int = SHUTDOWN_GRACE_SECONDS,
) -> bool:
    """Kill the server process group.  SIGTERM first, SIGKILL fallback.

    Uses the same process-group pattern as SessionManager._kill_process_tree.

    Returns True if the process was confirmed dead, False otherwise.
    """
    if proc is None or proc.poll() is not None:
        return True

    if _IS_WINDOWS:
        return _kill_group_win32(proc, grace_seconds)
    return _kill_group_posix(proc, grace_seconds)


def _kill_group_posix(proc: subprocess.Popen, grace: int) -> bool:
    """POSIX: SIGTERM the process group, then SIGKILL if needed."""
    pgid: Optional[int] = None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pass

    # SIGTERM
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

    try:
        proc.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        pass

    # SIGKILL
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
        logger.warning("Server process %d did not exit after SIGKILL", proc.pid)
        return False


def _kill_group_win32(proc: subprocess.Popen, grace: int) -> bool:
    """Windows: terminate, then taskkill /T /F."""
    try:
        proc.terminate()
    except OSError:
        pass

    try:
        proc.wait(timeout=grace)
        return True
    except subprocess.TimeoutExpired:
        pass

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
        logger.warning("Server process %d did not exit after taskkill", proc.pid)
        return False


def stop_server(server: ServerInfo) -> bool:
    """Stop a running server and ensure no orphan processes.

    Returns True if the process was confirmed dead.
    """
    logger.info(
        "Stopping %s server (pid=%d, port=%d)",
        server.framework, server.process.pid, server.port,
    )
    return kill_process_group(server.process)


# ── Integration Point ──


def can_start_server(
    ingestion: IngestionResult,
    project_dir: Path,
    language: str = "python",
) -> bool:
    """Return True if the project has a supported server framework and entry file.

    The execution engine can check this to decide whether to use HTTP-level
    testing or fall back to the callable harness.  Does not start anything.
    """
    detection = detect_framework_entry(ingestion, project_dir, language)
    return detection is not None
