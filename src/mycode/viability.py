"""Baseline Viability Gate — checks sandbox health before running stress scenarios.

Sits between library matching (Stage 4) and conversation (Stage 5) in the
pipeline. If the sandbox environment is too broken to produce meaningful
stress test results, the gate fails and the pipeline produces a short
diagnostic report instead of running scenarios.

Three checks:
  1. Dependency install rate — >=50% of declared non-dev deps must be installed
  2. Import success rate — >=50% of installed deps must be importable in sandbox
  3. Syntax parse rate — >=25% of source files must parse successfully

Pure Python. No LLM dependency.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from mycode.ingester import DependencyInfo, IngestionResult
from mycode.session import SessionManager

logger = logging.getLogger(__name__)

# ── Thresholds ──

INSTALL_RATE_THRESHOLD = 0.50
IMPORT_RATE_THRESHOLD = 0.50
SYNTAX_RATE_THRESHOLD = 0.25

# ── Package Name → Import Name Mapping ──
# Common PyPI packages where the import name differs from the package name.
# The replace("-", "_") heuristic handles most cases; this dict covers
# the well-known exceptions.

_PYTHON_IMPORT_NAMES: dict[str, str] = {
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "python-dotenv": "dotenv",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "python-dateutil": "dateutil",
    "pymysql": "pymysql",
    "pyjwt": "jwt",
    "python-jose": "jose",
    "python-multipart": "multipart",
    "msgpack-python": "msgpack",
    "attrs": "attr",
    "google-cloud-storage": "google.cloud.storage",
    "google-cloud-bigquery": "google.cloud.bigquery",
}


# ── Data Classes ──


@dataclass
class ViabilityResult:
    """Outcome of the baseline viability gate.

    Always recorded on PipelineResult (even on pass) so the report
    generator can use the rates for confidence notes.

    Designed to be JSON-serialisable for future corpus integration.
    """

    viable: bool
    install_rate: float            # 0.0–1.0
    import_rate: float             # 0.0–1.0 (of installed deps)
    syntax_rate: float             # 0.0–1.0
    missing_deps: list[str] = field(default_factory=list)
    unimportable_deps: list[str] = field(default_factory=list)
    reason: str = ""               # human-readable if not viable
    suggest_docker: bool = False

    # Stack context — not used for gate decision, but makes the result
    # self-contained for corpus storage.
    language: str = ""
    framework: str = ""            # e.g. "flask", "express" — from classifier vertical
    declared_deps: list[dict] = field(default_factory=list)  # [{name, version}, ...]

    def as_dict(self) -> dict:
        """JSON-serialisable dictionary representation."""
        return {
            "viable": self.viable,
            "install_rate": round(self.install_rate, 4),
            "import_rate": round(self.import_rate, 4),
            "syntax_rate": round(self.syntax_rate, 4),
            "missing_deps": self.missing_deps,
            "unimportable_deps": self.unimportable_deps,
            "reason": self.reason,
            "suggest_docker": self.suggest_docker,
            "language": self.language,
            "framework": self.framework,
            "declared_deps": self.declared_deps,
        }


# ── Import Checking ──


def _python_import_name(package_name: str) -> str:
    """Map a PyPI package name to its Python import name."""
    lower = package_name.lower()
    if lower in _PYTHON_IMPORT_NAMES:
        return _PYTHON_IMPORT_NAMES[lower]
    return package_name.replace("-", "_").lower()


def check_python_imports(
    session: SessionManager,
    deps: list[DependencyInfo],
) -> tuple[list[str], list[str]]:
    """Try importing each Python dep in the sandbox venv.

    Returns (importable_names, failed_names).
    """
    if not deps:
        return [], []

    lines = ["import json", "results = {}"]
    for dep in deps:
        module = _python_import_name(dep.name)
        # Use repr to safely embed the name in the script
        lines.append(f"try:")
        lines.append(f"    __import__({module!r})")
        lines.append(f"    results[{dep.name!r}] = True")
        lines.append(f"except Exception:")
        lines.append(f"    results[{dep.name!r}] = False")
    lines.append("print(json.dumps(results))")
    script = "\n".join(lines)

    try:
        result = session.run_in_session(
            ["python", "-c", script], timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().splitlines()[-1])
            importable = [name for name, ok in data.items() if ok]
            failed = [name for name, ok in data.items() if not ok]
            return importable, failed
    except Exception as exc:
        logger.debug("Python import check failed: %s", exc)

    # Fail-open: if the check itself breaks, assume all are importable
    return [d.name for d in deps], []


def check_js_imports(
    session: SessionManager,
    deps: list[DependencyInfo],
) -> tuple[list[str], list[str]]:
    """Try requiring each JS dep in the sandbox node_modules.

    Returns (importable_names, failed_names).
    """
    if not deps:
        return [], []

    lines = ["const results = {};"]
    for dep in deps:
        safe_name = dep.name.replace("'", "\\'")
        lines.append(
            f"try {{ require('{safe_name}'); results['{safe_name}'] = true; }} "
            f"catch(e) {{ results['{safe_name}'] = false; }}"
        )
    lines.append("console.log(JSON.stringify(results));")
    script = "\n".join(lines)

    try:
        result = session.run_in_session(
            ["node", "-e", script], timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().splitlines()[-1])
            importable = [name for name, ok in data.items() if ok]
            failed = [name for name, ok in data.items() if not ok]
            return importable, failed
    except Exception as exc:
        logger.debug("JS import check failed: %s", exc)

    return [d.name for d in deps], []


def check_importability(
    session: SessionManager,
    installed_deps: list[DependencyInfo],
    language: str,
) -> tuple[list[str], list[str]]:
    """Test-import installed deps in the sandbox.

    Returns (importable_names, failed_names).
    """
    if not installed_deps:
        return [], []

    if language == "python":
        return check_python_imports(session, installed_deps)
    return check_js_imports(session, installed_deps)


# ── Gate Logic ──


def run_viability_gate(
    session: SessionManager,
    ingestion: IngestionResult,
    language: str,
    is_containerised: bool = False,
    framework: str = "",
) -> ViabilityResult:
    """Run the three viability checks and return a composite result.

    Args:
        session: Active session manager with sandbox venv.
        ingestion: Completed ingestion result.
        language: "python" or "javascript".
        is_containerised: Whether running inside Docker.
        framework: Detected framework (from classifier vertical).

    Returns:
        ViabilityResult with pass/fail and diagnostics.
    """
    non_dev = [d for d in ingestion.dependencies if not d.is_dev]
    missing = [d for d in non_dev if d.is_missing]
    installed = [d for d in non_dev if not d.is_missing]

    # Check 1: Install rate
    install_rate = (len(non_dev) - len(missing)) / len(non_dev) if non_dev else 1.0

    # Check 2: Import rate (only for installed deps)
    importable, unimportable = check_importability(session, installed, language)
    import_rate = len(importable) / len(installed) if installed else 1.0

    # Check 3: Syntax rate
    total_files = ingestion.files_analyzed + ingestion.files_failed
    syntax_rate = ingestion.files_analyzed / total_files if total_files > 0 else 1.0

    viable = (
        install_rate >= INSTALL_RATE_THRESHOLD
        and import_rate >= IMPORT_RATE_THRESHOLD
        and syntax_rate >= SYNTAX_RATE_THRESHOLD
    )

    # Build reason string
    reasons: list[str] = []
    if install_rate < INSTALL_RATE_THRESHOLD:
        reasons.append(
            f"Only {len(installed)} of {len(non_dev)} "
            f"dependencies installed ({install_rate:.0%})."
        )
    if import_rate < IMPORT_RATE_THRESHOLD:
        reasons.append(
            f"Only {len(importable)} of {len(installed)} installed "
            f"dependencies are importable ({import_rate:.0%})."
        )
    if syntax_rate < SYNTAX_RATE_THRESHOLD:
        reasons.append(
            f"Only {ingestion.files_analyzed} of {total_files} source "
            f"files could be parsed ({syntax_rate:.0%})."
        )

    # Stack context for corpus integration
    declared_deps = [
        {
            "name": d.name,
            "version": d.required_version or d.installed_version or "",
        }
        for d in non_dev
    ]

    result = ViabilityResult(
        viable=viable,
        install_rate=install_rate,
        import_rate=import_rate,
        syntax_rate=syntax_rate,
        missing_deps=[d.name for d in missing],
        unimportable_deps=unimportable,
        reason=" ".join(reasons),
        suggest_docker=not viable and not is_containerised,
        language=language,
        framework=framework,
        declared_deps=declared_deps,
    )

    if not viable:
        logger.warning("Viability gate FAILED: %s", result.reason)
    else:
        logger.info(
            "Viability gate passed: install=%.0f%% import=%.0f%% syntax=%.0f%%",
            install_rate * 100, import_rate * 100, syntax_rate * 100,
        )

    return result


# ── Baseline Failed Report ──


def build_baseline_failed_text(
    viability: ViabilityResult,
    ingestion: IngestionResult,
    project_name: str,
) -> str:
    """Build a plain-text baseline-failed report matching the normal report style.

    Uses ASCII dividers (not markdown headers) to match the format
    of DiagnosticReport.as_text().
    """
    sections: list[str] = []

    # Header — matches normal report style
    sections.append("=" * 60)
    sections.append("  myCode Baseline Viability Report")
    sections.append("=" * 60)

    sections.append(
        f"\nmyCode could not establish a healthy test environment for "
        f"{project_name}. Stress testing was not attempted because the "
        f"results would not be meaningful."
    )

    # Dependency installation
    non_dev = [d for d in ingestion.dependencies if not d.is_dev]
    installed_count = len(non_dev) - len(viability.missing_deps)

    if viability.install_rate < 1.0 and non_dev:
        sections.append("\n" + "-" * 40)
        sections.append("  Dependency Installation")
        sections.append("-" * 40)
        sections.append(
            f"\n{installed_count} of {len(non_dev)} dependencies "
            f"installed ({viability.install_rate:.0%})."
        )
        if viability.missing_deps:
            shown = viability.missing_deps[:10]
            missing_str = ", ".join(shown)
            if len(viability.missing_deps) > 10:
                missing_str += f" (+{len(viability.missing_deps) - 10} more)"
            sections.append(f"Could not install: {missing_str}")

    # Import failures
    if viability.unimportable_deps:
        sections.append("\n" + "-" * 40)
        sections.append("  Import Failures")
        sections.append("-" * 40)
        sections.append(
            "\nThese dependencies installed but could not be imported:"
        )
        for name in viability.unimportable_deps[:10]:
            sections.append(f"  - {name}")
        sections.append(
            "\nThis typically means a native library or system "
            "dependency is missing."
        )

    # Syntax parsing
    total_files = ingestion.files_analyzed + ingestion.files_failed
    if viability.syntax_rate < 1.0 and total_files > 0:
        sections.append("\n" + "-" * 40)
        sections.append("  Source File Parsing")
        sections.append("-" * 40)
        sections.append(
            f"\n{ingestion.files_analyzed} of {total_files} source files "
            f"parsed successfully ({viability.syntax_rate:.0%})."
        )

    # Docker suggestion
    if viability.suggest_docker:
        sections.append("\n" + "-" * 40)
        sections.append("  Suggestion")
        sections.append("-" * 40)
        sections.append(
            "\nTry running with --containerised for better dependency "
            "isolation. Docker can install system-level libraries (C "
            "extensions, native bindings) that may be missing from "
            "your local environment."
        )
        sections.append(
            "\n  mycode /path/to/project --containerised"
        )

    # Next steps
    sections.append("\n" + "-" * 40)
    sections.append("  Next Steps")
    sections.append("-" * 40)
    sections.append(
        "\n1. Check that your project's dependencies install correctly "
        "in a clean virtual environment."
    )
    sections.append(
        "2. Run `pip install -r requirements.txt` (or `npm install`) "
        "in a fresh environment to verify."
    )
    sections.append(
        "3. Fix any missing system libraries or build dependencies."
    )
    sections.append(
        "4. Re-run myCode once the environment is healthy."
    )

    sections.append("\n" + "=" * 60)

    return "\n".join(sections)
