"""Pipeline (D3/E4) — Orchestrates the full myCode stress-testing flow.

Wires Session Manager → Project Ingester → Component Library →
Conversational Interface → Scenario Generator → Execution Engine →
Report Generator into a single entry point.  Optionally records
anonymized session data via the Interaction Recorder.

Detects project language (Python / JavaScript) from project contents,
handles errors gracefully at each stage, and returns structured results.

Nine stages:
  1. Language Detection
  2. Session Setup
  3. Project Ingestion
  4. Component Library Matching
  5. Conversational Interface (or skip with operational_intent override)
  6. Scenario Generation
  7. Scenario Review (or auto-approve in headless mode)
  8. Execution
  9. Report Generation

Pure orchestration layer — no LLM dependency of its own.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.constraints import OperationalConstraints
from mycode.discovery import DiscoveryEngine
from mycode.engine import ExecutionEngine, ExecutionEngineResult
from mycode.http_load_driver import run_http_testing_phase
from mycode.hysteresis import load_prior_state
from mycode.ingester import IngestionResult, ProjectIngester
from mycode.interface import (
    ConversationalInterface,
    InterfaceResult,
    OperationalIntent,
    TerminalIO,
    UserIO,
)
from mycode.js_ingester import JsProjectIngester
from mycode.library import ComponentLibrary, ProfileMatch
from mycode.recorder import InteractionRecorder
from mycode.report import DiagnosticReport, ReportGenerator
from mycode.scenario import (
    LLMConfig,
    ScenarioGenerator,
    ScenarioGeneratorResult,
    StressTestScenario,
)
from mycode.session import ResourceCaps, SessionManager
from mycode.viability import ViabilityResult, build_baseline_failed_text, run_viability_gate

logger = logging.getLogger(__name__)

# ── Constants ──

_DEFAULT_INTENT = (
    "General-purpose application — test for common failure modes "
    "including data scaling, memory issues, edge cases, and "
    "concurrency problems."
)

# Files that indicate a Python project
_PYTHON_INDICATORS = frozenset({
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
})

# Files that indicate a JavaScript/Node.js project
_JS_INDICATORS = frozenset({
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "tsconfig.json",
})

# File extensions per language
_PYTHON_EXTENSIONS = frozenset({".py"})
_JS_EXTENSIONS = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})

# Minimum JS source files to classify as a real JS project (not just build tooling)
_MIN_JS_SOURCE_FILES = 3


# ── Exceptions ──


class PipelineError(Exception):
    """Base exception for pipeline errors."""


class LanguageDetectionError(PipelineError):
    """Could not determine project language."""


# ── Data Classes ──


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run.

    Attributes:
        project_path: Path to the user's project directory.
        operational_intent: Optional override — if non-empty, the
            conversational interface is skipped and this string is used
            as the intent for scenario generation.  Scenarios are
            auto-approved in this headless mode.
        language: Force language ("python" or "javascript"). Auto-detected
            if not provided.
        resource_caps: Resource limits for the session sandbox.
        llm_config: LLM backend configuration for scenario generation.
        offline: Use offline scenario generation (no LLM calls).
        skip_version_check: Skip PyPI/npm version lookups (faster, for testing).
        temp_base: Override temp directory base for session workspace.
        consent: Opt-in for anonymous interaction recording.
        data_dir: Override data directory for interaction recorder.
        io: Injectable I/O handler for conversational interface.
        auto_approve_scenarios: Skip interactive scenario review.
    """

    project_path: str | Path = ""
    operational_intent: str = ""
    language: Optional[str] = None
    resource_caps: Optional[ResourceCaps] = None
    llm_config: Optional[LLMConfig] = None
    offline: bool = True
    skip_version_check: bool = False
    temp_base: Optional[str | Path] = None
    consent: bool = False
    data_dir: Optional[Path] = None
    io: Optional[UserIO] = None
    auto_approve_scenarios: bool = False
    prebuilt_constraints: Optional[OperationalConstraints] = None


@dataclass
class StageResult:
    """Outcome of a single pipeline stage.

    Attributes:
        stage: Human-readable stage name.
        success: Whether the stage completed without fatal error.
        duration_ms: Wall-clock time for the stage.
        error: Error message if the stage failed.
    """

    stage: str
    success: bool = True
    duration_ms: float = 0.0
    error: str = ""


@dataclass
class PipelineResult:
    """Complete output of a pipeline run.

    Attributes:
        language: Detected or forced project language.
        stages: Per-stage outcome summaries.
        ingestion: Full ingestion result (None if ingestion failed).
        profile_matches: Component library matches (empty if matching failed).
        scenarios: Scenario generator result (None if generation failed).
        execution: Execution engine result (None if execution failed).
        interface_result: Conversational interface output (None if skipped).
        report: Diagnostic report (None if generation failed or skipped).
        recording_path: Path to saved recording (None if consent not given).
        total_duration_ms: Wall-clock time for the entire pipeline.
        warnings: Accumulated warnings from all stages.
    """

    language: str = ""
    stages: list[StageResult] = field(default_factory=list)
    ingestion: Optional[IngestionResult] = None
    profile_matches: list[ProfileMatch] = field(default_factory=list)
    scenarios: Optional[ScenarioGeneratorResult] = None
    execution: Optional[ExecutionEngineResult] = None
    interface_result: Optional[InterfaceResult] = None
    report: Optional[DiagnosticReport] = None
    recording_path: Optional[Path] = None
    viability: Optional[ViabilityResult] = None
    discovery_paths: list[Path] = field(default_factory=list)
    total_duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if all stages completed successfully."""
        return all(s.success for s in self.stages)

    @property
    def failed_stage(self) -> Optional[str]:
        """Name of the first stage that failed, or None."""
        for s in self.stages:
            if not s.success:
                return s.stage
        return None


# ── Language Detection ──


def detect_languages(project_path: Path) -> set[str]:
    """Detect ALL languages present in a project.

    Returns a set: ``{"python"}``, ``{"javascript"}``, or
    ``{"python", "javascript"}`` for multi-language projects.

    Scans indicator files in root AND subdirectories.  Only detects
    JavaScript if there are real source files (≥3 .js/.ts), not just
    node_modules or a lone package.json from a build dependency.

    Raises:
        LanguageDetectionError: If no language can be determined.
    """
    if not project_path.is_dir():
        raise LanguageDetectionError(
            f"Project path is not a directory: {project_path}"
        )

    detected: set[str] = set()

    # Scan for indicator files in root and immediate subdirectories
    has_python_indicator = False
    has_js_indicator = False
    dirs_to_check = [project_path]
    for child in project_path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            dirs_to_check.append(child)

    for d in dirs_to_check:
        files = {f.name for f in d.iterdir() if f.is_file()}
        if files & _PYTHON_INDICATORS:
            has_python_indicator = True
        if files & _JS_INDICATORS:
            has_js_indicator = True

    # Count source files to confirm indicators (avoid build-tool false positives)
    py_count = 0
    js_count = 0
    for f in project_path.rglob("*"):
        if not f.is_file():
            continue
        # Skip node_modules — these are installed deps, not source
        if "node_modules" in f.parts:
            continue
        if f.suffix in _PYTHON_EXTENSIONS:
            py_count += 1
        elif f.suffix in _JS_EXTENSIONS:
            js_count += 1
        if py_count + js_count > 500:
            break

    # Python: indicator OR source files
    if has_python_indicator or py_count > 0:
        detected.add("python")

    # JavaScript: indicator AND enough source files (not just build tooling)
    if has_js_indicator and js_count >= _MIN_JS_SOURCE_FILES:
        detected.add("javascript")
    elif js_count >= _MIN_JS_SOURCE_FILES and not has_python_indicator:
        # JS source files without indicator but no Python either
        detected.add("javascript")

    if not detected:
        raise LanguageDetectionError(
            "Could not determine project language — no Python or JavaScript "
            "files found."
        )

    return detected


def detect_language(project_path: Path) -> str:
    """Detect the primary language of a project.

    Returns ``"python"`` or ``"javascript"``.  For multi-language projects,
    uses the existing heuristic (Python wins ties).

    This is the backward-compatible wrapper around ``detect_languages()``.

    Raises:
        LanguageDetectionError: If neither language can be determined.
    """
    languages = detect_languages(project_path)
    if len(languages) == 1:
        return next(iter(languages))
    # Multi-language — pick primary using existing heuristic
    # Count source files for tie-breaking
    py_count = 0
    js_count = 0
    for f in project_path.rglob("*"):
        if not f.is_file() or "node_modules" in f.parts:
            continue
        if f.suffix in _PYTHON_EXTENSIONS:
            py_count += 1
        elif f.suffix in _JS_EXTENSIONS:
            js_count += 1
        if py_count + js_count > 200:
            break
    if js_count > py_count:
        return "javascript"
    return "python"  # Python wins ties


# ── Multi-Language Merge ──

_PYTHON_SERVER_FRAMEWORKS = frozenset({
    "flask", "fastapi", "django", "streamlit", "gradio",
})
_JS_SERVER_FRAMEWORKS = frozenset({
    "express", "next", "nextjs", "next.js", "nuxt",
})


def determine_primary_language(
    py_deps: list[str],
    js_deps: list[str],
    py_lines: int = 0,
    js_lines: int = 0,
) -> str:
    """Determine which language is the primary stress-testing target.

    Rules (in priority order):
    1. Python has server framework → Python primary
    2. JavaScript has server framework (and Python doesn't) → JS primary
    3. Neither has server framework → language with more source lines
    4. Tie → Python
    """
    py_dep_lower = {d.lower() for d in py_deps}
    js_dep_lower = {d.lower() for d in js_deps}

    py_has_server = bool(py_dep_lower & _PYTHON_SERVER_FRAMEWORKS)
    js_has_server = bool(js_dep_lower & _JS_SERVER_FRAMEWORKS)

    if py_has_server:
        return "python"
    if js_has_server:
        return "javascript"
    if js_lines > py_lines:
        return "javascript"
    return "python"


def merge_ingestion_results(
    primary: IngestionResult,
    secondary: IngestionResult,
    primary_language: str,
) -> IngestionResult:
    """Merge two single-language IngestionResults into one.

    The *primary* result's ``project_path`` is used for the merged output.
    Dependencies, files, functions, and coupling points are concatenated.
    """
    secondary_language = (
        "javascript" if primary_language == "python" else "python"
    )

    return IngestionResult(
        project_path=primary.project_path,
        files_analyzed=primary.files_analyzed + secondary.files_analyzed,
        files_failed=primary.files_failed + secondary.files_failed,
        total_lines=primary.total_lines + secondary.total_lines,
        file_analyses=list(primary.file_analyses) + list(secondary.file_analyses),
        dependencies=list(primary.dependencies) + list(secondary.dependencies),
        dependency_tree={**primary.dependency_tree, **secondary.dependency_tree},
        function_flows=list(primary.function_flows) + list(secondary.function_flows),
        coupling_points=list(primary.coupling_points) + list(secondary.coupling_points),
        parse_errors=list(primary.parse_errors) + list(secondary.parse_errors),
        warnings=list(primary.warnings) + list(secondary.warnings),
        language=primary_language,
        secondary_languages=[secondary_language],
    )


# ── Pipeline ──


# ── LLM Report Counter ──

_LLM_REPORTS_INITIAL = 3
_CONFIG_DIR = Path.home() / ".mycode"
_CONFIG_PATH = _CONFIG_DIR / "config.json"


def _read_config() -> dict:
    """Read ~/.mycode/config.json, returning empty dict if missing."""
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_config(data: dict) -> None:
    """Write ~/.mycode/config.json, creating directories if needed."""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write config: %s", exc)


def check_llm_report_allowance() -> int:
    """Return the number of free LLM reports remaining.

    Returns _LLM_REPORTS_INITIAL if the config file doesn't exist yet
    (first run hasn't happened).  Returns 0 if the user has exhausted
    their free reports.
    """
    data = _read_config()
    return data.get("llm_reports_remaining", _LLM_REPORTS_INITIAL)


def decrement_llm_report_counter() -> int:
    """Decrement the free LLM report counter and return the new value.

    Creates the config file on first call if it doesn't exist.
    """
    data = _read_config()
    remaining = data.get("llm_reports_remaining", _LLM_REPORTS_INITIAL)
    remaining = max(0, remaining - 1)
    data["llm_reports_remaining"] = remaining
    _write_config(data)
    return remaining


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full myCode stress-testing pipeline.

    Flow:
        1. Detect language (or use forced value)
        2. Create Session Manager sandbox (venv for Python)
        3. Run Project Ingester on copied project
        4. Match dependencies against Component Library
        5. Conversational Interface (or skip if operational_intent provided)
        6. Generate stress test scenarios (Scenario Generator)
        7. User reviews scenarios (or auto-approve in headless mode)
        8. Execute approved scenarios (Execution Engine)
        9. Generate diagnostic report (Report Generator)

    The Interaction Recorder captures anonymized data after relevant
    stages when consent is given.

    Each stage is wrapped in error handling. If a stage fails, the pipeline
    records the error and returns partial results — downstream stages are
    skipped.

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineResult with all available results and stage outcomes.
    """
    project_path = Path(config.project_path).resolve()
    result = PipelineResult()
    pipeline_start = time.monotonic()

    # Load prior run state for severity hysteresis
    prior_state = load_prior_state(project_path)

    # Initialize recorder (always created; consent checked internally)
    recorder = InteractionRecorder(
        consent=config.consent,
        data_dir=config.data_dir,
    )

    # ── Stage 1: Language Detection ──
    stage_start = time.monotonic()
    try:
        if config.language:
            language = config.language.lower()
            if language not in ("python", "javascript"):
                raise LanguageDetectionError(
                    f"Unsupported language: {config.language!r}. "
                    f"Supported: python, javascript."
                )
        else:
            language = detect_language(project_path)
        result.language = language
        result.stages.append(StageResult(
            stage="language_detection",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Language detected: %s", language)
    except Exception as exc:
        result.stages.append(StageResult(
            stage="language_detection",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=str(exc),
        ))
        result.total_duration_ms = _elapsed_ms(pipeline_start)
        return result

    # ── Stages 2–9 run inside the Session Manager context ──
    try:
        with SessionManager(
            project_path,
            resource_caps=config.resource_caps,
            temp_base=config.temp_base,
        ) as session:
            result.stages.append(StageResult(
                stage="session_setup",
                duration_ms=0.0,  # Measured by SessionManager internally
            ))
            logger.info("Session created: %s", session.workspace_dir)

            # Surface dependency install warnings
            if session.dep_install_warnings:
                result.warnings.extend(session.dep_install_warnings)

            # ── Stage 3: Project Ingestion ──
            ingestion = _run_ingestion(
                session, language, config.skip_version_check, result,
            )
            if ingestion is None:
                result.total_duration_ms = _elapsed_ms(pipeline_start)
                return result

            # ── Stage 4: Component Library Matching ──
            matches = _run_library_matching(
                ingestion, language, result,
            )

            # Record dependencies (after library matching)
            _safe_record(
                recorder.record_dependencies, ingestion, matches, language,
            )

            # ── Stage 4.5: Viability Gate ──
            viability = _run_viability_check(
                session, ingestion, language, result,
            )
            if not viability.viable:
                project_name = _infer_project_name(Path(config.project_path))
                report_text = build_baseline_failed_text(
                    viability, ingestion, project_name,
                )
                result.report = DiagnosticReport(
                    project_name=project_name,
                    summary=viability.reason,
                    baseline_failed=True,
                    _baseline_report_text=report_text,
                )
                result.total_duration_ms = _elapsed_ms(pipeline_start)
                _safe_save(recorder, result)
                return result

            # ── Stage 5: Conversational Interface ──
            intent, project_name = _run_conversation(
                ingestion, config, language, result,
            )

            # Record conversation
            if result.interface_result:
                _safe_record(
                    recorder.record_conversation,
                    result.interface_result.intent.as_intent_string(),
                    result.interface_result.intent.raw_responses,
                )

            # ── Stage 6: Scenario Generation ──
            constraints = (
                result.interface_result.constraints
                if result.interface_result else None
            )
            scenarios = _run_scenario_generation(
                ingestion, matches, config, language, result, intent,
                constraints=constraints,
            )
            if scenarios is None:
                result.total_duration_ms = _elapsed_ms(pipeline_start)
                _safe_save(recorder, result)
                return result

            # Record scenarios
            _safe_record(recorder.record_scenarios, scenarios)

            # ── Stage 7: Scenario Review ──
            approved = _run_scenario_review(scenarios, config, result)

            # ── Stage 8: Execution ──
            _run_execution(
                session, ingestion, approved, result, language,
                io=config.io, constraints=constraints,
            )

            # ── Stage 8.5: HTTP Load Testing ──
            if result.execution is not None:
                try:
                    result.execution = run_http_testing_phase(
                        session=session,
                        ingestion=ingestion,
                        execution=result.execution,
                        language=language,
                        constraints=constraints,
                        on_progress=lambda msg: (
                            config.io.display(msg)
                            if config.io else None
                        ),
                        prior_state=prior_state,
                    )
                except Exception as exc:
                    logger.warning("HTTP testing failed: %s", exc)
                    result.warnings.append(f"HTTP testing failed: {exc}")

            # Record execution
            if result.execution is not None:
                _safe_record(recorder.record_execution, result.execution)

            # ── Discovery Logging (no consent required) ──
            if result.execution is not None:
                _safe_record(
                    _run_discovery_logging,
                    result.execution, approved, matches, constraints,
                    language, result,
                )

            # ── Stage 9: Report Generation ──
            if result.execution is not None:
                _run_report_generation(
                    result.execution, ingestion, matches, intent,
                    project_name, config, result,
                    constraints=constraints,
                    prior_state=prior_state,
                )

                # Record report
                if result.report is not None:
                    _safe_record(recorder.record_report, result.report)

    except Exception as exc:
        # Session Manager setup/teardown failure
        if not any(s.stage == "session_setup" for s in result.stages):
            result.stages.append(StageResult(
                stage="session_setup",
                success=False,
                duration_ms=_elapsed_ms(stage_start),
                error=f"Session Manager failed: {exc}",
            ))
        else:
            result.warnings.append(f"Unexpected error during pipeline: {exc}")
        logger.exception("Pipeline error")

    # Save recorder
    _safe_save(recorder, result)

    result.total_duration_ms = _elapsed_ms(pipeline_start)
    return result


# ── Stage Implementations ──


def _run_ingestion(
    session: SessionManager,
    language: str,
    skip_version_check: bool,
    result: PipelineResult,
) -> Optional[IngestionResult]:
    """Stage 3: Run the appropriate ingester on the project copy."""
    stage_start = time.monotonic()
    try:
        # Use venv packages (where deps were actually installed), not host
        installed = session.get_venv_packages()
        dep_file_dir = session.find_dep_file_dir()

        if language == "python":
            ingester = ProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=installed,
                skip_pypi_check=skip_version_check,
                dep_file_dir=dep_file_dir,
            )
        else:
            ingester = JsProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=None,
                skip_npm_check=skip_version_check,
                dep_file_dir=dep_file_dir,
            )

        ingestion = ingester.ingest()
        result.ingestion = ingestion

        # Record partial parsing as warnings, not failures
        if ingestion.files_failed > 0:
            result.warnings.append(
                f"Analyzed {ingestion.files_analyzed} of "
                f"{ingestion.files_analyzed + ingestion.files_failed} files. "
                f"{ingestion.files_failed} couldn't be parsed."
            )
        if ingestion.warnings:
            result.warnings.extend(ingestion.warnings)

        result.stages.append(StageResult(
            stage="ingestion",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info(
            "Ingestion complete: %d files, %d dependencies",
            ingestion.files_analyzed,
            len(ingestion.dependencies),
        )
        return ingestion

    except Exception as exc:
        result.stages.append(StageResult(
            stage="ingestion",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Project analysis failed: {exc}",
        ))
        logger.exception("Ingestion failed")
        return None


def _run_library_matching(
    ingestion: IngestionResult,
    language: str,
    result: PipelineResult,
) -> list[ProfileMatch]:
    """Stage 4: Match project dependencies against Component Library."""
    stage_start = time.monotonic()
    try:
        library = ComponentLibrary()
        dep_dicts = [
            {"name": d.name, "installed_version": d.installed_version}
            for d in ingestion.dependencies
            if not d.is_dev
        ]

        if not dep_dicts:
            result.warnings.append(
                "No declared dependencies found. Stress tests will use "
                "generic scenarios based on code analysis."
            )
            result.profile_matches = []
            result.stages.append(StageResult(
                stage="library_matching",
                duration_ms=_elapsed_ms(stage_start),
            ))
            return []

        matches = library.match_dependencies(language, dep_dicts)
        result.profile_matches = matches

        recognized = library.get_recognized(matches)
        unrecognized = library.get_unrecognized(matches)

        if unrecognized:
            result.warnings.append(
                f"{len(unrecognized)} dependencies not in component library "
                f"(will use generic stress testing): "
                f"{', '.join(unrecognized[:10])}"
                + ("..." if len(unrecognized) > 10 else "")
            )

        result.stages.append(StageResult(
            stage="library_matching",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info(
            "Library matching: %d recognized, %d unrecognized",
            len(recognized), len(unrecognized),
        )
        return matches

    except Exception as exc:
        # Library matching failure is non-fatal — continue with empty matches
        result.stages.append(StageResult(
            stage="library_matching",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Component library matching failed: {exc}",
        ))
        result.warnings.append(
            "Component library matching failed. Proceeding with generic "
            "scenarios only."
        )
        logger.exception("Library matching failed")
        return []


def _run_viability_check(
    session: SessionManager,
    ingestion: IngestionResult,
    language: str,
    result: PipelineResult,
) -> ViabilityResult:
    """Stage 4.5: Check whether the sandbox can produce meaningful results."""
    stage_start = time.monotonic()

    is_containerised = os.environ.get("MYCODE_CONTAINERISED") == "1"

    viability = run_viability_gate(
        session=session,
        ingestion=ingestion,
        language=language,
        is_containerised=is_containerised,
    )

    result.viability = viability
    result.stages.append(StageResult(
        stage="viability_gate",
        success=viability.viable,
        duration_ms=_elapsed_ms(stage_start),
        error=viability.reason if not viability.viable else "",
    ))

    return viability


def _infer_project_name(project_path: Path) -> str:
    """Infer a human-readable project name from the filesystem.

    Checks pyproject.toml and package.json first, falls back to
    the directory name.
    """
    project_path = project_path.resolve()

    # Check pyproject.toml
    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            # Simple TOML parsing for name field under [project]
            import re
            m = re.search(
                r'^\[project\]\s*\n(?:.*\n)*?name\s*=\s*"([^"]+)"',
                text,
                re.MULTILINE,
            )
            if m:
                name = m.group(1).replace("-", " ").replace("_", " ")
                return name.title()
        except Exception:
            pass

    # Check package.json
    pkg_json = project_path / "package.json"
    if pkg_json.exists():
        try:
            import json
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            name = data.get("name", "")
            if name and not name.startswith("@"):
                name = name.replace("-", " ").replace("_", " ")
                return name.title()
        except Exception:
            pass

    # Fall back to directory name
    name = project_path.name
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def _run_conversation(
    ingestion: IngestionResult,
    config: PipelineConfig,
    language: str,
    result: PipelineResult,
) -> tuple[str, str]:
    """Stage 5: Run conversational interface to extract operational intent.

    Returns ``(intent_string, project_name)`` for downstream stages.
    If ``config.operational_intent`` is provided, the conversation is
    skipped and ``project_name`` is empty.
    """
    stage_start = time.monotonic()

    # Prebuilt constraints from host conversation (containerised mode)
    if config.operational_intent and config.prebuilt_constraints is not None:
        intent_obj = OperationalIntent(summary=config.operational_intent)
        result.interface_result = InterfaceResult(
            intent=intent_obj,
            constraints=config.prebuilt_constraints,
        )
        result.stages.append(StageResult(
            stage="conversation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Using prebuilt constraints from host conversation.")
        project_name = _infer_project_name(Path(config.project_path))
        return config.operational_intent, project_name

    # Headless mode — skip conversation
    if config.operational_intent:
        result.stages.append(StageResult(
            stage="conversation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Conversation skipped — operational intent provided.")
        project_name = _infer_project_name(Path(config.project_path))
        return config.operational_intent, project_name

    try:
        io = config.io or TerminalIO()
        interface = ConversationalInterface(
            llm_config=config.llm_config,
            offline=config.offline,
            io=io,
        )
        interface_result = interface.run(ingestion, language)
        result.interface_result = interface_result

        if interface_result.warnings:
            result.warnings.extend(interface_result.warnings)

        intent_string = interface_result.intent.as_intent_string()
        project_name = (
            interface_result.intent.project_name
            or _infer_project_name(Path(config.project_path))
        )
        if not intent_string:
            intent_string = _DEFAULT_INTENT
            result.warnings.append(
                "Empty operational intent from conversation — using default."
            )

        result.stages.append(StageResult(
            stage="conversation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Conversation complete, intent: %s", intent_string[:80])
        return intent_string, project_name

    except Exception as exc:
        # Conversation failure is non-fatal — fall back to default intent
        result.stages.append(StageResult(
            stage="conversation",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Conversational interface failed: {exc}",
        ))
        result.warnings.append(
            "Conversation failed. Using generic stress testing profile."
        )
        logger.exception("Conversation failed")
        return _DEFAULT_INTENT, ""


def _run_scenario_generation(
    ingestion: IngestionResult,
    matches: list[ProfileMatch],
    config: PipelineConfig,
    language: str,
    result: PipelineResult,
    intent: str,
    constraints: Optional[OperationalConstraints] = None,
) -> Optional[ScenarioGeneratorResult]:
    """Stage 6: Generate stress test scenarios."""
    stage_start = time.monotonic()
    try:
        generator = ScenarioGenerator(
            llm_config=config.llm_config,
            offline=config.offline,
        )

        scenarios = generator.generate(
            ingestion_result=ingestion,
            profile_matches=matches,
            operational_intent=intent,
            language=language,
            constraints=constraints,
        )
        result.scenarios = scenarios

        if scenarios.warnings:
            result.warnings.extend(scenarios.warnings)

        if not scenarios.scenarios:
            result.warnings.append(
                "Scenario generator produced no test scenarios."
            )

        result.stages.append(StageResult(
            stage="scenario_generation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info(
            "Generated %d scenarios (model: %s)",
            len(scenarios.scenarios), scenarios.model_used,
        )
        return scenarios

    except Exception as exc:
        result.stages.append(StageResult(
            stage="scenario_generation",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Scenario generation failed: {exc}",
        ))
        logger.exception("Scenario generation failed")
        return None


def _run_scenario_review(
    scenarios: ScenarioGeneratorResult,
    config: PipelineConfig,
    result: PipelineResult,
) -> list[StressTestScenario]:
    """Stage 7: Present scenarios for user review before execution.

    Returns the list of approved scenarios.  Auto-approves when
    ``operational_intent`` was provided or ``auto_approve_scenarios`` is set.
    """
    stage_start = time.monotonic()

    # Auto-approve in headless mode
    if config.auto_approve_scenarios or config.operational_intent:
        result.stages.append(StageResult(
            stage="scenario_review",
            duration_ms=_elapsed_ms(stage_start),
        ))
        return list(scenarios.scenarios)

    try:
        io = config.io or TerminalIO()
        interface = ConversationalInterface(
            llm_config=config.llm_config,
            offline=True,  # Review doesn't need LLM
            io=io,
        )
        review = interface.review_scenarios(scenarios.scenarios)

        if result.interface_result:
            result.interface_result.review = review

        if review.skipped:
            result.warnings.append(
                f"User skipped {len(review.skipped)} scenario(s): "
                f"{', '.join(review.skipped)}"
            )
        if review.user_notes:
            result.warnings.append(f"User feedback: {review.user_notes}")

        result.stages.append(StageResult(
            stage="scenario_review",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info(
            "Scenario review: %d approved, %d skipped",
            len(review.approved), len(review.skipped),
        )
        return review.approved

    except Exception as exc:
        # Review failure is non-fatal — approve all
        result.stages.append(StageResult(
            stage="scenario_review",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Scenario review failed: {exc}",
        ))
        result.warnings.append("Scenario review failed. Running all scenarios.")
        logger.exception("Scenario review failed")
        return list(scenarios.scenarios)


def _run_execution(
    session: SessionManager,
    ingestion: IngestionResult,
    approved_scenarios: list[StressTestScenario],
    result: PipelineResult,
    language: str = "python",
    io: Optional[UserIO] = None,
    constraints: Optional[OperationalConstraints] = None,
) -> None:
    """Stage 8: Execute approved scenarios."""
    stage_start = time.monotonic()

    if not approved_scenarios:
        result.stages.append(StageResult(
            stage="execution",
            duration_ms=_elapsed_ms(stage_start),
        ))
        result.warnings.append("No scenarios to execute.")
        return

    try:
        engine = ExecutionEngine(
            session=session, ingestion=ingestion, language=language,
            io=io, constraints=constraints,
        )
        execution = engine.execute(scenarios=approved_scenarios)
        result.execution = execution

        if execution.warnings:
            result.warnings.extend(execution.warnings)

        result.stages.append(StageResult(
            stage="execution",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info(
            "Execution complete: %d completed, %d failed, %d skipped",
            execution.scenarios_completed,
            execution.scenarios_failed,
            execution.scenarios_skipped,
        )

    except Exception as exc:
        result.stages.append(StageResult(
            stage="execution",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Execution failed: {exc}",
        ))
        logger.exception("Execution failed")


def _run_report_generation(
    execution: ExecutionEngineResult,
    ingestion: IngestionResult,
    matches: list[ProfileMatch],
    intent: str,
    project_name: str,
    config: PipelineConfig,
    result: PipelineResult,
    constraints: Optional[OperationalConstraints] = None,
    prior_state=None,
) -> None:
    """Stage 9: Generate the diagnostic report."""
    stage_start = time.monotonic()
    try:
        generator = ReportGenerator(
            llm_config=config.llm_config,
            offline=config.offline,
        )
        report = generator.generate(
            execution=execution,
            ingestion=ingestion,
            profile_matches=matches,
            operational_intent=intent,
            project_name=project_name,
            constraints=constraints,
            prior_state=prior_state,
        )
        result.report = report

        result.stages.append(StageResult(
            stage="report_generation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Report generated (model: %s)", report.model_used)

    except Exception as exc:
        # Report generation failure is non-fatal
        result.stages.append(StageResult(
            stage="report_generation",
            success=False,
            duration_ms=_elapsed_ms(stage_start),
            error=f"Report generation failed: {exc}",
        ))
        result.warnings.append(
            "Report generation failed. Raw execution results are still available."
        )
        logger.exception("Report generation failed")


# ── Helpers ──


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since *start* (monotonic)."""
    return (time.monotonic() - start) * 1000.0


def _safe_record(func: object, *args: object, **kwargs: object) -> None:
    """Call a recorder method, swallowing all errors."""
    try:
        func(*args, **kwargs)
    except Exception as exc:
        logger.debug("Recorder call failed (non-blocking): %s", exc)


def _safe_save(recorder: InteractionRecorder, result: PipelineResult) -> None:
    """Save recorder to disk if consent given. Never blocks pipeline."""
    if not recorder.consent:
        return
    if result.recording_path is not None:
        return  # Already saved
    try:
        result.recording_path = recorder.save()
    except Exception as exc:
        logger.debug("Recorder save failed (non-blocking): %s", exc)


def _run_discovery_logging(
    execution: ExecutionEngineResult,
    approved: list[StressTestScenario],
    matches: list[ProfileMatch],
    constraints: Optional[OperationalConstraints],
    language: str,
    result: PipelineResult,
) -> None:
    """Run discovery analysis and save any findings.

    Called via ``_safe_record`` — all exceptions are swallowed by the
    caller so this never blocks the pipeline.
    """
    engine = DiscoveryEngine()
    scenarios = ScenarioGeneratorResult(scenarios=approved)
    discoveries = engine.analyse(
        execution, scenarios, matches, constraints, language,
    )
    if discoveries:
        paths = engine.save(discoveries)
        result.discovery_paths = paths
        result.warnings.append(
            f"Discovery engine: {len(discoveries)} novel pattern(s) logged "
            f"to ~/.mycode/discoveries/"
        )
        logger.info("Discovery engine: %d candidate(s) saved", len(discoveries))
    else:
        logger.debug("Discovery engine: no novel patterns detected")
