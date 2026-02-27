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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.constraints import OperationalConstraints
from mycode.engine import ExecutionEngine, ExecutionEngineResult
from mycode.ingester import IngestionResult, ProjectIngester
from mycode.interface import (
    ConversationalInterface,
    InterfaceResult,
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
                io=config.io,
            )

            # Record execution
            if result.execution is not None:
                _safe_record(recorder.record_execution, result.execution)

            # ── Stage 9: Report Generation ──
            if result.execution is not None:
                _run_report_generation(
                    result.execution, ingestion, matches, intent,
                    project_name, config, result,
                    constraints=constraints,
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
                installed_packages=None,
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

    # Headless mode — skip conversation
    if config.operational_intent:
        result.stages.append(StageResult(
            stage="conversation",
            duration_ms=_elapsed_ms(stage_start),
        ))
        logger.info("Conversation skipped — operational intent provided.")
        project_name = Path(config.project_path).resolve().name
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
        project_name = interface_result.intent.project_name
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
            io=io,
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
