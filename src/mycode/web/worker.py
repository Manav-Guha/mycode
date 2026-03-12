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
        )

        def _on_progress(completed: int, total: int, current: str) -> None:
            job.progress_scenarios_complete = completed
            job.progress_scenarios_total = total
            job.progress_current_scenario = current

        execution = engine.execute(
            scenarios=approved,
            on_progress=_on_progress,
        )

        job.progress_scenarios_complete = (
            execution.scenarios_completed
            + execution.scenarios_failed
            + execution.scenarios_skipped
        )
        job.progress_current_scenario = ""
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

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.progress_stage = "failed"
        logger.exception("Job %s failed: %s", job.id, exc)
