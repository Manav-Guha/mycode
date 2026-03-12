"""Pydantic request/response models for the myCode web API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Request Models ──


@dataclass
class PreflightRequest:
    """Request for the preflight endpoint.

    Exactly one of ``github_url`` or ``upload_filename`` should be set.
    For file uploads, the zip is sent as multipart form data alongside
    the JSON body — ``upload_filename`` is informational only.
    """

    github_url: str = ""
    upload_filename: str = ""


@dataclass
class ConverseRequest:
    """Request for the conversation endpoint."""

    job_id: str = ""
    turn: int = 1
    user_response: str = ""


@dataclass
class AnalyzeRequest:
    """Request to start the full test run."""

    job_id: str = ""
    auto_approve_scenarios: bool = True
    offline: bool = True


# ── Response Models ──


@dataclass
class DependencyStatus:
    """Dependency status for preflight response."""

    name: str
    installed_version: Optional[str] = None
    is_missing: bool = False
    has_profile: bool = False


@dataclass
class ViabilityStatus:
    """Viability gate result for preflight response."""

    viable: bool = True
    install_rate: float = 0.0
    import_rate: float = 0.0
    syntax_rate: float = 0.0
    missing_deps: list[str] = field(default_factory=list)
    unimportable_deps: list[str] = field(default_factory=list)
    reason: str = ""
    suggest_docker: bool = False


@dataclass
class InferencePredictions:
    """Tier 1 inference predictions for preflight response."""

    vertical: str = ""
    architectural_pattern: str = ""
    risk_areas: list[dict] = field(default_factory=list)


@dataclass
class PreflightResponse:
    """Response from the preflight endpoint."""

    job_id: str = ""
    language: str = ""
    project_name: str = ""
    dependencies: list[DependencyStatus] = field(default_factory=list)
    viability: Optional[ViabilityStatus] = None
    profile_matches: list[str] = field(default_factory=list)
    inference: Optional[InferencePredictions] = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ConverseResponse:
    """Response from the conversation endpoint."""

    job_id: str = ""
    turn: int = 1
    question: str = ""
    project_summary: str = ""
    constraints: Optional[dict] = None
    operational_intent: str = ""
    done: bool = False
    message: str = ""  # informational (e.g. "Defaulting to ...")
    error: str = ""


@dataclass
class AnalyzeResponse:
    """Response from the analyze endpoint (immediate acknowledgment)."""

    job_id: str = ""
    status: str = ""
    message: str = ""
    error: str = ""


@dataclass
class ProgressInfo:
    """Progress tracking for a running job."""

    scenarios_total: int = 0
    scenarios_complete: int = 0
    current_scenario: str = ""
    progress_pct: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class StatusResponse:
    """Response from the status endpoint."""

    job_id: str = ""
    status: str = ""
    stage: str = ""
    progress: Optional[ProgressInfo] = None
    error: str = ""


@dataclass
class ReportResponse:
    """Response from the report endpoint."""

    job_id: str = ""
    report: Optional[dict] = None
    pipeline_summary: Optional[dict] = None
    error: str = ""


@dataclass
class HealthResponse:
    """Response from the health endpoint."""

    status: str = "ok"
    docker_available: bool = False
    version: str = ""
    active_jobs: int = 0
    max_concurrent_jobs: int = 4
