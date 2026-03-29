"""API endpoint definitions for the myCode web backend."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional

from mycode.classifiers import classify_project
from mycode.ingester import IngestionResult, ProjectIngester
from mycode.interface import ConversationalInterface, InterfaceResult, OperationalIntent
from mycode.js_ingester import JsProjectIngester
from mycode.library import ComponentLibrary
from mycode.pipeline import detect_language, _infer_project_name
from mycode.scenario import LLMConfig
from mycode.session import SessionManager
from mycode.viability import ViabilityResult, run_viability_gate

from mycode.web.analytics import log_download, log_job_started
from mycode.web.jobs import Job, store, MAX_CONCURRENT_JOBS
from mycode.web.project_fetch import (
    FetchError,
    clone_github_repo,
    create_temp_dir,
    extract_zip,
)
from mycode.web.schemas import (
    AnalyzeResponse,
    ConverseResponse,
    DependencyStatus,
    HealthResponse,
    InferencePredictions,
    PreflightResponse,
    ProgressInfo,
    ReportResponse,
    StatusResponse,
    ViabilityStatus,
)
from mycode.web.worker import run_analysis

logger = logging.getLogger(__name__)


def _repo_name_from_url(url: str) -> str:
    """Extract a human-readable project name from a GitHub URL.

    E.g. "https://github.com/user/react-shopping-cart" → "React Shopping Cart"
    """
    try:
        parts = url.strip().rstrip("/").split("/")
        repo = parts[-1] if parts else ""
        if repo.endswith(".git"):
            repo = repo[:-4]
        name = repo.replace("-", " ").replace("_", " ")
        return name.title() if name else "Project"
    except Exception:
        return "Project"


# Thread pool for background analysis jobs
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)


# ── Preflight ──


def handle_preflight(
    github_url: str = "",
    file_obj: Optional[BytesIO] = None,
    filename: str = "",
    source: str = "public",
) -> PreflightResponse:
    """Run preflight diagnostics (stages 1-4.5).

    Accepts either a GitHub URL or an uploaded file object.
    Returns structured preflight data for the frontend.
    """
    job = store.create()
    job.source = source
    warnings: list[str] = []

    try:
        # Fetch project
        temp_dir = create_temp_dir()
        job.project_path = temp_dir

        if github_url:
            job.github_url = github_url
            repo_url = github_url.strip()
            project_path = clone_github_repo(github_url, temp_dir / "project")
        elif file_obj is not None:
            repo_url = "zip_upload"
            project_path = extract_zip(file_obj, temp_dir / "project")
        else:
            return PreflightResponse(
                error="Provide either a GitHub URL or upload a zip file."
            )

        # Log job start to analytics
        log_job_started(job.id, source, repo_url)

        # Stage 1: Language detection
        language = detect_language(project_path)
        job.language = language

        # Stage 2: Session setup
        session = SessionManager(
            project_path,
            temp_base=temp_dir / "session",
        )
        session.setup()
        job.session = session

        # Stage 3: Ingestion
        # Use venv packages (where deps were actually installed), not host
        installed = session.get_venv_packages()
        dep_file_dir = session.find_dep_file_dir()

        if language == "python":
            ingester = ProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=installed,
                skip_pypi_check=True,
                dep_file_dir=dep_file_dir,
            )
        else:
            ingester = JsProjectIngester(
                project_path=session.project_copy_dir,
                installed_packages=None,
                skip_npm_check=True,
                dep_file_dir=dep_file_dir,
            )

        ingestion = ingester.ingest()
        job.ingestion = ingestion

        # Stage 4: Library matching
        library = ComponentLibrary()
        dep_dicts = [
            {"name": d.name, "installed_version": d.installed_version}
            for d in ingestion.dependencies
            if not d.is_dev
        ]
        matches = library.match_dependencies(language, dep_dicts) if dep_dicts else []
        job.matches = matches

        recognized_names = {m.dependency_name for m in matches if m.profile is not None}

        # Stage 4.5: Viability gate
        viability = run_viability_gate(
            session=session,
            ingestion=ingestion,
            language=language,
            is_containerised=False,
        )
        job.viability = viability

        # Project name — prefer package.json/pyproject.toml, fall back to
        # GitHub repo name if the inferred name is generic (e.g. temp dir)
        project_name = _infer_project_name(project_path)
        if (
            project_name.lower() in ("project", "my project", "app")
            and github_url
        ):
            project_name = _repo_name_from_url(github_url)
        job.project_name = project_name

        # Build dependency list
        dep_statuses = [
            DependencyStatus(
                name=d.name,
                installed_version=d.installed_version,
                is_missing=d.is_missing,
                has_profile=d.name in recognized_names,
            )
            for d in ingestion.dependencies
            if not d.is_dev
        ]

        # Viability response
        viability_status = ViabilityStatus(
            viable=viability.viable,
            install_rate=round(viability.install_rate, 4),
            import_rate=round(viability.import_rate, 4),
            syntax_rate=round(viability.syntax_rate, 4),
            missing_deps=viability.missing_deps,
            unimportable_deps=viability.unimportable_deps,
            reason=viability.reason,
            suggest_docker=viability.suggest_docker,
        )

        # Inference predictions
        inference = _run_inference(ingestion, language)

        # Profile match names
        profile_names = [m.dependency_name for m in matches if m.profile is not None]

        job.status = "preflight_complete"

        return PreflightResponse(
            job_id=job.id,
            language=language,
            project_name=project_name,
            dependencies=dep_statuses,
            viability=viability_status,
            profile_matches=profile_names,
            inference=inference,
            warnings=warnings + list(ingestion.warnings),
        )

    except FetchError as exc:
        job.status = "preflight_failed"
        job.error = str(exc)
        return PreflightResponse(job_id=job.id, error=str(exc))

    except Exception as exc:
        job.status = "preflight_failed"
        job.error = str(exc)
        logger.exception("Preflight failed for job %s", job.id)
        return PreflightResponse(
            job_id=job.id,
            error=f"Preflight analysis failed: {exc}",
        )


def _run_inference(ingestion: IngestionResult, language: str) -> InferencePredictions:
    """Run Tier 1 inference engine for risk predictions."""
    try:
        from mycode.inference import CorpusIndex, InferenceEngine
        index = CorpusIndex()
        if not index._entries:
            deps = [d.name for d in ingestion.dependencies if not d.is_dev]
            files = [fa.file_path for fa in ingestion.file_analyses]
            cls = classify_project(
                dependencies=deps,
                file_structure=files,
                framework="",
                file_count=ingestion.files_analyzed,
                has_frontend=False,
                has_backend=False,
            )
            return InferencePredictions(
                vertical=cls.get("vertical", ""),
                architectural_pattern=cls.get("architectural_pattern", ""),
            )

        engine = InferenceEngine(index)
        dep_names = [d.name for d in ingestion.dependencies if not d.is_dev]
        if not dep_names:
            return InferencePredictions()

        result = engine.infer(dep_names)
        risk_areas = [
            {
                "domain": p.failure_domain,
                "description": p.description,
                "confidence": p.confidence,
            }
            for p in result.predictions[:5]
        ]
        deps = [d.name for d in ingestion.dependencies if not d.is_dev]
        files = [fa.file_path for fa in ingestion.file_analyses]
        cls = classify_project(
            dependencies=deps,
            file_structure=files,
            framework="",
            file_count=ingestion.files_analyzed,
            has_frontend=False,
            has_backend=False,
        )
        return InferencePredictions(
            vertical=cls.get("vertical", ""),
            architectural_pattern=cls.get("architectural_pattern", ""),
            risk_areas=risk_areas,
        )
    except Exception:
        return InferencePredictions()


# ── Conversation ──


def handle_converse(job_id: str, turn: int, user_response: str) -> ConverseResponse:
    """Handle one turn of the conversational interface.

    Turns 1-2 mirror the CLI's two open-ended questions.  Turn 3 extracts
    constraints from text, then checks for gaps.  Turns 4+ ask one explicit
    follow-up per missing constraint (user_scale, data_type, usage_pattern,
    max_payload_mb) — the same questions the CLI asks in
    ``_extract_constraints``.  The conversation is ``done`` once all
    follow-ups have been answered (or there were none to ask).
    """
    job = store.get(job_id)
    if job is None:
        return ConverseResponse(error=f"Job {job_id} not found.")

    if job.status not in ("preflight_complete", "conversing", "conversation_done"):
        return ConverseResponse(
            job_id=job_id,
            error=f"Cannot converse — job is in state '{job.status}'.",
        )

    if job.ingestion is None:
        return ConverseResponse(job_id=job_id, error="No ingestion data — run preflight first.")

    job.status = "conversing"

    interface = ConversationalInterface(offline=True)

    try:
        if turn == 1:
            # Prepare Turn 1 question
            question, summary = interface.prepare_turn_1(
                job.ingestion, job.language,
            )
            job.conversation_summary = summary

            return ConverseResponse(
                job_id=job_id,
                turn=1,
                question=question,
                project_summary=summary,
                done=False,
            )

        elif turn == 2:
            # Process Turn 1 response, get Turn 2 question
            job.turn1_response = user_response
            question = interface.process_turn_1(
                user_response, job.ingestion, job.language,
            )

            return ConverseResponse(
                job_id=job_id,
                turn=2,
                question=question,
                done=False,
            )

        elif turn == 3:
            # Process Turn 2 response, extract constraints from text
            job.turn2_response = user_response
            result = interface.process_turn_2(
                turn1_response=job.turn1_response,
                turn2_response=user_response,
                ingestion=job.ingestion,
                language=job.language,
            )

            job.interface_result = result
            job.constraints = result.constraints
            job.intent_string = result.intent.as_intent_string()

            # Check if there are missing constraints that need follow-up
            return _followup_or_done(job, turn)

        elif turn >= 4:
            # Follow-up answer for a specific constraint field
            if job.constraints is not None and job.pending_followup_field:
                constraints, parsed_ok, message = (
                    ConversationalInterface.apply_followup_answer(
                        job.constraints,
                        job.pending_followup_field,
                        user_response,
                        is_retry=job.followup_is_retry,
                    )
                )
                job.constraints = constraints

                if not parsed_ok:
                    # Re-ask with clarification (first failure only)
                    job.followup_is_retry = True
                    return ConverseResponse(
                        job_id=job_id,
                        turn=turn,
                        question=message,
                        done=False,
                    )

                # Parse succeeded (or defaulted) — reset retry state
                default_message = message  # non-empty when a default was applied
                job.pending_followup_field = ""
                job.followup_is_retry = False

                # If a default was applied, include the notification
                result = _followup_or_done(job, turn)
                if default_message and not result.message:
                    result.message = default_message
                return result

            return _followup_or_done(job, turn)

    except Exception as exc:
        logger.exception("Conversation turn %d failed for job %s", turn, job_id)
        return ConverseResponse(
            job_id=job_id,
            error="Could not process your response. Please try again.",
        )

    return ConverseResponse(job_id=job_id, error=f"Invalid turn number: {turn}")


def _followup_or_done(job: Job, current_turn: int) -> ConverseResponse:
    """Ask the next follow-up question, or finalise the conversation."""
    if job.constraints is not None:
        field_name, question = ConversationalInterface.get_followup_question(
            job.constraints,
        )
        if field_name is not None:
            job.pending_followup_field = field_name
            return ConverseResponse(
                job_id=job.id,
                turn=current_turn,
                question=question,
                done=False,
            )

    # All constraints filled (or as filled as they'll get) — done
    job.status = "conversation_done"

    constraints_dict = None
    if job.constraints:
        constraints_dict = {
            "user_scale": job.constraints.user_scale,
            "current_users": job.constraints.current_users,
            "max_users": job.constraints.max_users,
            "usage_pattern": job.constraints.usage_pattern,
            "max_payload_mb": job.constraints.max_payload_mb,
            "per_user_data": job.constraints.per_user_data,
            "max_total_data": job.constraints.max_total_data,
            "data_type": job.constraints.data_type,
            "deployment_context": job.constraints.deployment_context,
            "availability_requirement": job.constraints.availability_requirement,
            "data_sensitivity": job.constraints.data_sensitivity,
            "growth_expectation": job.constraints.growth_expectation,
            "timeout_per_scenario": job.constraints.timeout_per_scenario,
            "analysis_depth": job.constraints.analysis_depth,
        }

    return ConverseResponse(
        job_id=job.id,
        turn=current_turn,
        constraints=constraints_dict,
        operational_intent=job.intent_string,
        done=True,
    )


# ── Analyze ──


def handle_analyze(job_id: str, offline: bool = True) -> AnalyzeResponse:
    """Start the full test run (stages 6-9) in a background thread."""
    job = store.get(job_id)
    if job is None:
        return AnalyzeResponse(error=f"Job {job_id} not found.")

    if job.status not in ("conversation_done", "preflight_complete"):
        return AnalyzeResponse(
            job_id=job_id,
            error=f"Cannot analyze — job is in state '{job.status}'.",
        )

    _executor.submit(run_analysis, job)

    return AnalyzeResponse(
        job_id=job_id,
        status="running",
        message="Generating and executing stress tests...",
    )


# ── Status ──


def handle_status(job_id: str) -> StatusResponse:
    """Return current job status with progress information."""
    job = store.get(job_id)
    if job is None:
        return StatusResponse(error=f"Job {job_id} not found.")

    progress = None
    if job.status == "running":
        elapsed = time.time() - job.progress_start_time if job.progress_start_time else 0
        total = job.progress_scenarios_total or 1
        complete = job.progress_scenarios_complete
        pct = int((complete / total) * 100) if total > 0 else 0

        progress = ProgressInfo(
            scenarios_total=job.progress_scenarios_total,
            scenarios_complete=complete,
            current_scenario=job.progress_current_scenario,
            progress_pct=pct,
            elapsed_seconds=round(elapsed, 1),
        )

    return StatusResponse(
        job_id=job_id,
        status=job.status,
        stage=job.progress_stage,
        progress=progress,
        error=job.error,
    )


# ── Report ──


def handle_report(job_id: str) -> ReportResponse:
    """Return the full diagnostic report for a completed job."""
    job = store.get(job_id)
    if job is None:
        return ReportResponse(error=f"Job {job_id} not found.")

    if job.status != "completed":
        return ReportResponse(
            job_id=job_id,
            error=f"Report not ready — job is in state '{job.status}'.",
        )

    if job.result is None or job.result.report is None:
        return ReportResponse(
            job_id=job_id,
            error="No report generated.",
        )

    report_dict = job.result.report.as_dict()
    pipeline_summary = {
        "language": job.result.language,
        "total_duration_ms": job.result.total_duration_ms,
        "scenarios_run": job.result.report.scenarios_run,
        "scenarios_passed": job.result.report.scenarios_passed,
        "scenarios_failed": job.result.report.scenarios_failed,
        "scenarios_incomplete": job.result.report.scenarios_incomplete,
        "total_errors": job.result.report.total_errors,
    }

    # Render edition documents and cache PDF bytes on the job
    understanding_md = ""
    edition = 0
    has_pdf = False
    try:
        from mycode.documents import (
            _HAS_FPDF,
            get_next_edition,
            pdf_filename,
            render_understanding,
            render_understanding_pdf,
        )
        edition = get_next_edition(github_url=job.github_url or None)
        understanding_md = render_understanding(job.result.report, edition)

        if _HAS_FPDF:
            project_name = job.result.report.project_name or "Project"
            job._understanding_pdf = render_understanding_pdf(
                job.result.report, edition, project_name,
            )
            job._understanding_filename = pdf_filename(project_name, "Results")
            has_pdf = True
    except Exception as exc:
        logger.warning("Could not render edition documents: %s", exc)

    return ReportResponse(
        job_id=job_id,
        report=report_dict,
        pipeline_summary=pipeline_summary,
        understanding_md=understanding_md,
        edition=edition,
        has_pdf=has_pdf,
    )


# ── PDF Downloads ──


def handle_download_pdf(job_id: str, doc_type: str) -> tuple[bytes, str, str]:
    """Return PDF bytes, filename, and error for a completed job.

    doc_type: "understanding" or "fixes".
    Returns (pdf_bytes, filename, error).  On error, pdf_bytes is empty.
    """
    job = store.get(job_id)
    if job is None:
        return b"", "", f"Job {job_id} not found."

    if job.status != "completed":
        return b"", "", f"Report not ready — job is in state '{job.status}'."

    if doc_type == "understanding":
        pdf_bytes = getattr(job, "_understanding_pdf", b"")
        filename = getattr(job, "_understanding_filename", "")
    else:
        return b"", "", f"Unknown document type: {doc_type}"

    if not pdf_bytes:
        return b"", "", "PDF not available — fpdf2 may not be installed."

    log_download(job_id, "pdf")
    return pdf_bytes, filename, ""


# ── Health ──

# Cache Docker availability — subprocess call can take 10s if daemon is down.
# ── Submit Intent (web form) ──


def handle_submit_intent(job_id: str, answers: dict) -> dict:
    """Parse grouped form answers into OperationalConstraints and start analysis.

    Called by the web frontend when the user submits the structured intake
    form.  Unlike ``/api/converse``, this processes all answers in a single
    request.
    """
    from mycode.constraints import (
        OperationalConstraints,
        infer_availability,
        parse_analysis_depth,
        parse_data_type,
        parse_max_total_data,
        parse_per_user_data,
        parse_usage_pattern,
        parse_user_scale,
        depth_to_timeout,
    )

    job = store.get(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found."}
    if job.status not in ("preflight_complete", "conversing", "conversation_done"):
        return {"error": f"Cannot submit intent — job is in state '{job.status}'."}

    # Parse each field
    current_users = parse_user_scale(answers.get("current_users", ""))
    max_users = parse_user_scale(answers.get("max_users", ""))
    data_type = parse_data_type(answers.get("data_type", ""))
    usage_pattern = parse_usage_pattern(answers.get("usage_pattern", ""))
    per_user_data = parse_per_user_data(answers.get("per_user_data", ""))
    max_total_data = parse_max_total_data(answers.get("max_total_data", ""))
    analysis_depth = parse_analysis_depth(answers.get("analysis_depth", ""))

    constraints = OperationalConstraints(
        current_users=current_users,
        max_users=max_users,
        user_scale=max_users or current_users,
        data_type=data_type,
        usage_pattern=usage_pattern,
        per_user_data=per_user_data,
        max_total_data=max_total_data,
        analysis_depth=analysis_depth,
        timeout_per_scenario=depth_to_timeout(analysis_depth) if analysis_depth else None,
        availability_requirement=infer_availability(usage_pattern),
        project_description=answers.get("description", ""),
    )

    job.constraints = constraints
    job.intent_string = constraints.as_summary()
    job.status = "conversation_done"

    return {
        "job_id": job_id,
        "constraints": {
            "user_scale": constraints.user_scale,
            "current_users": constraints.current_users,
            "max_users": constraints.max_users,
            "usage_pattern": constraints.usage_pattern,
            "per_user_data": constraints.per_user_data,
            "max_total_data": constraints.max_total_data,
            "data_type": constraints.data_type,
            "analysis_depth": constraints.analysis_depth,
        },
        "done": True,
    }


# ── Predict ──


def handle_predict(job_id: str) -> dict:
    """Return corpus-based predictions for the job's dependency stack."""
    from mycode.prediction import predict_issues

    job = store.get(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found."}
    if job.ingestion is None:
        return {"error": "No ingestion data — run preflight first."}

    dep_names = [d.name for d in job.ingestion.dependencies]
    result = predict_issues(
        dependency_names=dep_names,
        constraints=job.constraints,
        ingestion=job.ingestion,
    )

    return {
        "total_similar_projects": result.total_similar_projects,
        "matching_deps": result.matching_deps,
        "predictions": [
            {
                "title": p.title,
                "probability_pct": p.probability_pct,
                "severity": p.severity,
                "confirmed_count": p.confirmed_count,
                "matching_deps": p.matching_deps,
                "scale_note": p.scale_note,
            }
            for p in result.predictions
        ],
    }


# Rechecked every 5 minutes.
_docker_cache: dict[str, object] = {}


def handle_health() -> HealthResponse:
    """Return server health status."""
    now = time.time()
    if "result" not in _docker_cache or now - _docker_cache.get("ts", 0) > 300:
        try:
            from mycode.container import is_docker_available
            _docker_cache["result"] = is_docker_available()
        except Exception:
            _docker_cache["result"] = False
        _docker_cache["ts"] = now

    docker_available = bool(_docker_cache.get("result", False))

    version = "0.1.2"
    try:
        import importlib.metadata
        version = importlib.metadata.version("mycode-ai")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        docker_available=docker_available,
        version=version,
        active_jobs=store.active_count(),
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
    )
