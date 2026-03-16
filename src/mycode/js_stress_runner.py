"""js_stress_runner — Python wrapper for the Node.js instrumented caller.

Takes discovered exports from js_module_loader + scenario config, spawns
the Node.js stress runner, and returns raw harness output compatible with
engine.py's _parse_harness_output.

Part of Track B (Node.js callable harness). B3 wires this into the engine.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from mycode.js_module_loader import (
    ExportedFunction,
    ModuleDiscoveryResult,
    _check_node_available,
    _compile_typescript,
)

logger = logging.getLogger(__name__)

_RUNNER_SCRIPT = Path(__file__).parent / "js_stress_runner.js"

_RESULTS_START = "__MYCODE_RESULTS_START__"
_RESULTS_END = "__MYCODE_RESULTS_END__"

_DEFAULT_SCALE_LEVELS = [100, 1000, 10000]
_DEFAULT_STEP_TIMEOUT_MS = 60000
_DEFAULT_RUNNER_TIMEOUT = 300  # seconds (total process timeout)


# ── Result types ──


@dataclass
class StressRunResult:
    """Raw output from the Node.js stress runner.

    ``stdout`` contains the full process output (with markers).
    ``parsed`` contains the extracted JSON dict if markers were found.
    ``returncode`` is the process exit code.
    """
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    parsed: dict | None = None
    error: str | None = None


# ── Public API ──


def run_stress(
    file_path: str,
    exports: list[ExportedFunction],
    scale_levels: list[int] | None = None,
    step_timeout_ms: int = _DEFAULT_STEP_TIMEOUT_MS,
    param_names: dict[str, list[str]] | None = None,
    node_path: str = "node",
    timeout: int = _DEFAULT_RUNNER_TIMEOUT,
    project_dir: str | None = None,
) -> StressRunResult:
    """Run the Node.js stress runner against discovered exports.

    Returns a StressRunResult whose ``stdout`` field contains the raw
    process output compatible with engine.py's _parse_harness_output.
    The ``parsed`` field contains the extracted JSON dict if available.
    """
    if not _check_node_available(node_path):
        return StressRunResult(error="Node.js is not available", returncode=-1)

    if not os.path.isfile(file_path):
        return StressRunResult(error=f"File not found: {file_path}", returncode=-1)

    if not exports:
        return StressRunResult(error="No exported functions to test", returncode=-1)

    # TypeScript: compile first
    ts_tmp_dir = None
    actual_path = file_path
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".ts", ".tsx"):
        import shutil
        compiled, ts_error = _compile_typescript(file_path, project_dir)
        if ts_error:
            return StressRunResult(error=ts_error, returncode=-1)
        actual_path = compiled
        ts_tmp_dir = os.path.dirname(compiled)

    try:
        return _spawn_runner(
            actual_path, exports, scale_levels or _DEFAULT_SCALE_LEVELS,
            step_timeout_ms, param_names or {}, node_path, timeout,
        )
    finally:
        if ts_tmp_dir:
            import shutil
            shutil.rmtree(ts_tmp_dir, ignore_errors=True)


def _spawn_runner(
    file_path: str,
    exports: list[ExportedFunction],
    scale_levels: list[int],
    step_timeout_ms: int,
    param_names: dict[str, list[str]],
    node_path: str,
    timeout: int,
) -> StressRunResult:
    """Spawn the Node.js runner and collect output."""
    task = json.dumps({
        "command": "stress",
        "file_path": os.path.abspath(file_path),
        "functions": [
            {"name": e.name, "arity": e.arity, "is_async": e.is_async}
            for e in exports
        ],
        "scale_levels": scale_levels,
        "step_timeout_ms": step_timeout_ms,
        "param_names": param_names,
    })

    try:
        result = subprocess.run(
            [node_path, str(_RUNNER_SCRIPT)],
            input=task,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return StressRunResult(
            error=f"Stress runner timed out after {timeout}s",
            returncode=-1,
        )
    except OSError as exc:
        return StressRunResult(
            error=f"Failed to spawn Node.js: {exc}",
            returncode=-1,
        )

    stdout = result.stdout
    stderr = result.stderr

    # Extract JSON between markers
    parsed = _extract_results(stdout)

    return StressRunResult(
        stdout=stdout,
        stderr=stderr,
        returncode=result.returncode,
        parsed=parsed,
    )


def _extract_results(stdout: str) -> dict | None:
    """Extract the JSON dict between result markers, or None."""
    start = stdout.find(_RESULTS_START)
    end = stdout.find(_RESULTS_END)
    if start == -1 or end == -1:
        return None

    json_str = stdout[start + len(_RESULTS_START):end].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
