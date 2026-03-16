"""js_module_loader — Python wrapper for the Node.js module loader.

Spawns a Node.js child process to load a JavaScript (or TypeScript) module,
discover its exported functions, and return structured results.

Part of Track B (Node.js callable harness). This is the foundation layer —
B3 wires it into the execution engine.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_LOADER_SCRIPT = Path(__file__).parent / "js_module_loader.js"

_TS_COMPILE_TIMEOUT = 30  # seconds
_DISCOVER_TIMEOUT = 10    # seconds


# ── Result types ──


@dataclass
class ExportedFunction:
    """A single exported function discovered from a JS/TS module."""
    name: str
    arity: int
    is_async: bool = False


@dataclass
class ModuleDiscoveryResult:
    """Result of loading a module and enumerating its exports."""
    exports: list[ExportedFunction] = field(default_factory=list)
    load_method: str | None = None  # "cjs" | "esm" | None on error
    error: str | None = None
    error_type: str | None = None


# ── TypeScript compilation ──


def _find_tsc(project_dir: str | None = None) -> str | None:
    """Find tsc binary: project-local first, then global PATH."""
    if project_dir:
        local = os.path.join(project_dir, "node_modules", ".bin", "tsc")
        if os.path.isfile(local) and os.access(local, os.X_OK):
            return local

    return shutil.which("tsc")


def _compile_typescript(
    file_path: str,
    project_dir: str | None = None,
    timeout: int = _TS_COMPILE_TIMEOUT,
) -> tuple[str | None, str | None]:
    """Compile a .ts/.tsx file to JS. Returns (compiled_js_path, error).

    The caller is responsible for cleaning up the temp directory
    (parent of the returned path).
    """
    tsc = _find_tsc(project_dir)
    if tsc is None:
        return None, "TypeScript file requires tsc — not found in project or PATH"

    tmp_dir = tempfile.mkdtemp(prefix="mycode_ts_")
    try:
        cmd = [
            tsc,
            "--outDir", tmp_dir,
            "--allowJs",
            "--esModuleInterop",
            "--skipLibCheck",
            "--declaration", "false",
            "--moduleResolution", "node",
            "--target", "ES2020",
            "--module", "commonjs",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()[:500]
            return None, f"tsc compilation failed: {stderr}"

        # Find the compiled .js file
        base = Path(file_path).stem + ".js"
        compiled = os.path.join(tmp_dir, base)
        if os.path.isfile(compiled):
            return compiled, None

        # tsc may preserve directory structure — search for it
        for root, _dirs, files in os.walk(tmp_dir):
            for f in files:
                if f == base:
                    return os.path.join(root, f), None

        return None, "tsc produced no output file"

    except subprocess.TimeoutExpired:
        return None, f"tsc compilation timed out after {timeout}s"
    except OSError as exc:
        return None, f"Failed to run tsc: {exc}"


# ── Module discovery ──


def _check_node_available(node_path: str = "node") -> bool:
    """Return True if Node.js is available."""
    try:
        result = subprocess.run(
            [node_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def discover_exports(
    file_path: str,
    node_path: str = "node",
    timeout: int = _DISCOVER_TIMEOUT,
    project_dir: str | None = None,
) -> ModuleDiscoveryResult:
    """Discover exported functions from a JS/TS module.

    Spawns a Node.js child process that loads the module and enumerates
    all exported functions (name, arity, async flag).

    For .ts/.tsx files, compiles to JS first using the project's tsc.
    """
    if not _check_node_available(node_path):
        return ModuleDiscoveryResult(
            error="Node.js is not available",
            error_type="NodeNotFound",
        )

    if not os.path.isfile(file_path):
        return ModuleDiscoveryResult(
            error=f"File not found: {file_path}",
            error_type="FileNotFoundError",
        )

    # TypeScript: compile first
    ts_tmp_dir = None
    actual_path = file_path
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".ts", ".tsx"):
        compiled, ts_error = _compile_typescript(file_path, project_dir)
        if ts_error:
            return ModuleDiscoveryResult(error=ts_error, error_type="TypeScriptError")
        actual_path = compiled
        ts_tmp_dir = os.path.dirname(compiled)

    try:
        return _run_loader(actual_path, node_path, timeout)
    finally:
        if ts_tmp_dir:
            shutil.rmtree(ts_tmp_dir, ignore_errors=True)


def _run_loader(
    file_path: str,
    node_path: str,
    timeout: int,
) -> ModuleDiscoveryResult:
    """Spawn the Node.js loader script and parse its output."""
    task = json.dumps({"command": "discover", "file_path": os.path.abspath(file_path)})

    try:
        result = subprocess.run(
            [node_path, str(_LOADER_SCRIPT)],
            input=task,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ModuleDiscoveryResult(
            error=f"Module loading timed out after {timeout}s",
            error_type="TimeoutError",
        )
    except OSError as exc:
        return ModuleDiscoveryResult(
            error=f"Failed to spawn Node.js: {exc}",
            error_type="OSError",
        )

    stdout = result.stdout.strip()
    if not stdout:
        stderr = (result.stderr or "").strip()[:500]
        return ModuleDiscoveryResult(
            error=f"No output from loader (exit {result.returncode}): {stderr}",
            error_type="LoaderError",
        )

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return ModuleDiscoveryResult(
            error=f"Invalid JSON from loader: {stdout[:200]}",
            error_type="ParseError",
        )

    if data.get("status") == "error":
        return ModuleDiscoveryResult(
            error=data.get("error", "Unknown error"),
            error_type=data.get("error_type", "Error"),
        )

    exports = [
        ExportedFunction(
            name=e["name"],
            arity=e["arity"],
            is_async=e.get("is_async", False),
        )
        for e in data.get("exports", [])
    ]

    return ModuleDiscoveryResult(
        exports=exports,
        load_method=data.get("load_method"),
    )
