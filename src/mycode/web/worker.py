"""Background task runner for myCode web API.

Runs pipeline stages 6-9 in a thread pool so the API can return
immediately while tests execute.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from mycode.constraints import OperationalConstraints
from mycode.engine import ExecutionEngine
from mycode.http_load_driver import run_http_testing_phase
from mycode.ingester import IngestionResult
from mycode.interface import InterfaceResult, OperationalIntent, UserIO
from mycode.library import ProfileMatch
from mycode.pipeline import (
    PipelineConfig,
    PipelineResult,
    StageResult,
    _DEFAULT_INTENT,
    _elapsed_ms,
)
from mycode.report import ReportGenerator
from mycode.scenario import LLMConfig, ScenarioGenerator
from mycode.session import SessionManager
from mycode.web.analytics import log_job_completed
from mycode.web.jobs import Job

logger = logging.getLogger(__name__)


class NullIO:
    """No-op I/O for headless execution."""

    def display(self, message: str) -> None:
        pass

    def prompt(self, message: str) -> str:
        return ""


def run_analysis(job: Job) -> None:
    """Run stages 6-9 (scenario gen → execution → report) for a job.

    Updates job state in place. Called from a thread pool.
    """
    job.status = "running"
    job.progress_stage = "scenario_generation"
    job.progress_start_time = time.time()

    ingestion = job.ingestion
    matches = job.matches
    session = job.session
    language = job.language
    languages = getattr(job, "detected_languages", None) or {language}

    if ingestion is None or session is None:
        job.status = "failed"
        job.error = "Missing preflight data — run preflight first."
        return

    intent_string = job.intent_string or _DEFAULT_INTENT
    constraints = job.constraints
    offline = True  # Default to offline for free tier
    llm_config = job.llm_config

    try:
        # ── Stage 6: Scenario Generation ──
        generator = ScenarioGenerator(
            llm_config=llm_config,
            offline=offline if llm_config is None else False,
        )

        scenarios_result = generator.generate(
            ingestion_result=ingestion,
            profile_matches=matches,
            operational_intent=intent_string,
            language=language,
            languages=languages if len(languages) > 1 else None,
            constraints=constraints,
        )

        if not scenarios_result.scenarios:
            job.status = "failed"
            job.error = "Scenario generator produced no test scenarios."
            return

        approved = list(scenarios_result.scenarios)
        job.progress_scenarios_total = len(approved)
        job.progress_stage = "execution"

        # ── Stage 8: Execution ──
        engine = ExecutionEngine(
            session=session,
            ingestion=ingestion,
            language=language,
            io=NullIO(),
            constraints=constraints,
        )

        def _on_progress(completed: int, total: int, current: str) -> None:
            job.progress_scenarios_complete = completed
            job.progress_scenarios_total = total
            job.progress_current_scenario = current

        execution = engine.execute(
            scenarios=approved,
            on_progress=_on_progress,
        )

        callable_done = (
            execution.scenarios_completed
            + execution.scenarios_failed
            + execution.scenarios_skipped
        )
        job.progress_scenarios_complete = callable_done
        job.progress_current_scenario = ""

        # ── Stage 8.5: HTTP Load Testing ──
        # Add 1 to total for the HTTP testing phase so the progress bar
        # doesn't show 100% while HTTP tests are still running.
        job.progress_stage = "http_testing"
        job.progress_scenarios_total = callable_done + 1
        job.progress_current_scenario = "HTTP endpoint testing..."
        try:
            execution = run_http_testing_phase(
                session=session,
                ingestion=ingestion,
                execution=execution,
                language=language,
                constraints=constraints,
                on_progress=lambda msg: setattr(
                    job, "progress_current_scenario", msg
                ),
            )
        except Exception as exc:
            logger.warning("HTTP testing failed for job %s: %s", job.id, exc)
            execution.warnings.append(
                f"HTTP load testing could not run: {exc}"
            )
        job.progress_scenarios_complete = callable_done + 1

        job.progress_stage = "report_generation"

        # ── Stage 9: Report Generation ──
        report_gen = ReportGenerator(
            llm_config=llm_config,
            offline=offline if llm_config is None else False,
        )

        report = report_gen.generate(
            execution=execution,
            ingestion=ingestion,
            profile_matches=matches,
            operational_intent=intent_string,
            project_name=job.project_name,
            constraints=constraints,
        )

        # Build a PipelineResult for consistent output
        result = PipelineResult(
            language=language,
            ingestion=ingestion,
            profile_matches=matches,
            scenarios=scenarios_result,
            execution=execution,
            report=report,
            viability=job.viability,
        )

        if job.interface_result:
            result.interface_result = job.interface_result

        job.result = result
        job.status = "completed"
        job.progress_stage = "done"
        logger.info("Job %s completed successfully", job.id)

        # Log completion to analytics
        _log_completion(job, "completed")

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress_stage = "failed"
        logger.exception("Job %s failed: %s", job.id, exc)

        # Detect timeout vs generic failure
        error_lower = str(exc).lower()
        status = "timeout" if ("timeout" in error_lower or "timed out" in error_lower or "budget_exceeded" in error_lower) else "failed"
        _log_completion(job, status)


def _log_completion(job: Job, status: str) -> None:
    """Extract finding counts from job result and log to analytics."""
    critical = warning = info = scenarios_run = 0
    if job.result and job.result.report:
        report = job.result.report
        scenarios_run = report.scenarios_run or 0
        for f in getattr(report, "findings", []):
            sev = getattr(f, "severity", "").lower()
            if sev == "critical":
                critical += 1
            elif sev == "warning":
                warning += 1
            elif sev == "info":
                info += 1
    log_job_completed(job.id, status, critical, warning, info, scenarios_run)
