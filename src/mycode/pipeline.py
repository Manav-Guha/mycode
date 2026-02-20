"""Pipeline (D3) — Orchestrates the full myCode stress-testing flow.

Wires Session Manager → Project Ingester → Component Library → Scenario
Generator → Execution Engine into a single entry point.  Detects project
language (Python / JavaScript) from project contents, handles errors
gracefully at each stage, and returns structured results.

Pure orchestration layer — no LLM dependency of its own.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.engine import ExecutionEngine, ExecutionEngineResult
from mycode.ingester import IngestionResult, ProjectIngester
from mycode.js_ingester import JsProjectIngester
from mycode.library import ComponentLibrary, ProfileMatch
from mycode.scenario import (
    LLMConfig,
    ScenarioGenerator,
    ScenarioGeneratorResult,
)
from mycode.session import ResourceCaps, SessionManager

logger = logging.getLogger(__name__)

# ── Constants ──

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
        operational_intent: Plain-language description of what the project
            does and how it will be used (from conversational interface).
        language: Force language ("python" or "javascript"). Auto-detected
            if not provided.
        resource_caps: Resource limits for the session sandbox.
        llm_config: LLM backend configuration for scenario generation.
        offline: Use offline scenario generation (no LLM calls).
        skip_version_check: Skip PyPI/npm version lookups (faster, for testing).
        temp_base: Override temp directory base for session workspace.
    """

    project_path: str | Path = ""
    operational_intent: str = ""
    language: Optional[str] = None
    resource_caps: Optional[ResourceCaps] = None
    llm_config: Optional[LLMConfig] = None
    offline: bool = True
    skip_version_check: bool = False
    temp_base: Optional[str | Path] = None


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
        total_duration_ms: Wall-clock time for the entire pipeline.
        warnings: Accumulated warnings from all stages.
    """

    language: str = ""
    stages: list[StageResult] = field(default_factory=list)
    ingestion: Optional[IngestionResult] = None
    profile_matches: list[ProfileMatch] = field(default_factory=list)
    scenarios: Optional[ScenarioGeneratorResult] = None
    execution: Optional[ExecutionEngineResult] = None
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


def detect_language(project_path: Path) -> str:
    """Detect whether a project is Python or JavaScript.

    Checks for indicator files first (requirements.txt, package.json, etc.),
    then falls back to counting source file extensions.

    Returns:
        "python" or "javascript"

    Raises:
        LanguageDetectionError: If neither language can be determined.
    """
    if not project_path.is_dir():
        raise LanguageDetectionError(
            f"Project path is not a directory: {project_path}"
        )

    top_level_files = {f.name for f in project_path.iterdir() if f.is_file()}

    has_python_indicators = bool(top_level_files & _PYTHON_INDICATORS)
    has_js_indicators = bool(top_level_files & _JS_INDICATORS)

    # Unambiguous cases
    if has_python_indicators and not has_js_indicators:
        return "python"
    if has_js_indicators and not has_python_indicators:
        return "javascript"

    # Both or neither — count source files
    py_count = 0
    js_count = 0
    for f in project_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix in _PYTHON_EXTENSIONS:
            py_count += 1
        elif f.suffix in _JS_EXTENSIONS:
            js_count += 1
        # Stop early once we have enough signal
        if py_count + js_count > 200:
            break

    if py_count > js_count:
        return "python"
    if js_count > py_count:
        return "javascript"
    if py_count > 0:
        return "python"  # Tie-break: Python (arbitrary but deterministic)

    raise LanguageDetectionError(
        "Could not determine project language — no Python or JavaScript "
        "files found."
    )


# ── Pipeline ──


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full myCode stress-testing pipeline.

    Flow:
        1. Detect language (or use forced value)
        2. Create Session Manager sandbox (venv for Python)
        3. Run Project Ingester on copied project
        4. Match dependencies against Component Library
        5. Generate stress test scenarios (Scenario Generator)
        6. Execute scenarios (Execution Engine)

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

    # ── Stages 2–6 run inside the Session Manager context ──
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

            # ── Stage 5: Scenario Generation ──
            scenarios = _run_scenario_generation(
                ingestion, matches, config, language, result,
            )
            if scenarios is None:
                result.total_duration_ms = _elapsed_ms(pipeline_start)
                return result

            # ── Stage 6: Execution ──
            _run_execution(session, ingestion, scenarios, result)

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
        env = session.environment_info
        installed = env.installed_packages if env else {}

        if language == "python":
            ingester = ProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=installed,
                skip_pypi_check=skip_version_check,
            )
        else:
            ingester = JsProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=installed,
                skip_npm_check=skip_version_check,
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


def _run_scenario_generation(
    ingestion: IngestionResult,
    matches: list[ProfileMatch],
    config: PipelineConfig,
    language: str,
    result: PipelineResult,
) -> Optional[ScenarioGeneratorResult]:
    """Stage 5: Generate stress test scenarios."""
    stage_start = time.monotonic()
    try:
        generator = ScenarioGenerator(
            llm_config=config.llm_config,
            offline=config.offline,
        )

        intent = config.operational_intent or (
            "General-purpose application — test for common failure modes "
            "including data scaling, memory issues, edge cases, and "
            "concurrency problems."
        )

        scenarios = generator.generate(
            ingestion_result=ingestion,
            profile_matches=matches,
            operational_intent=intent,
            language=language,
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


def _run_execution(
    session: SessionManager,
    ingestion: IngestionResult,
    scenarios: ScenarioGeneratorResult,
    result: PipelineResult,
) -> None:
    """Stage 6: Execute generated scenarios."""
    stage_start = time.monotonic()

    if not scenarios.scenarios:
        result.stages.append(StageResult(
            stage="execution",
            duration_ms=_elapsed_ms(stage_start),
        ))
        result.warnings.append("No scenarios to execute.")
        return

    try:
        engine = ExecutionEngine(session=session, ingestion=ingestion)
        execution = engine.execute(scenarios=scenarios.scenarios)
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


# ── Helpers ──


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since *start* (monotonic)."""
    return (time.monotonic() - start) * 1000.0
