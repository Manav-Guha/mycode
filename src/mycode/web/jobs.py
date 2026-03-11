"""Job state machine and in-memory store for myCode web API.

Jobs track the lifecycle of a single analysis request from preflight
through report generation. State is held in memory — swap this module
for a Redis-backed implementation when Kubernetes scaling is needed.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.constraints import OperationalConstraints
from mycode.ingester import IngestionResult
from mycode.interface import InterfaceResult
from mycode.library import ProfileMatch
from mycode.pipeline import PipelineResult
from mycode.scenario import LLMConfig
from mycode.session import SessionManager
from mycode.viability import ViabilityResult

logger = logging.getLogger(__name__)

# ── Constants ──

JOB_TTL_SECONDS = int(os.environ.get("MYCODE_JOB_TTL_SECONDS", "1800"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MYCODE_MAX_CONCURRENT_JOBS", "4"))

# Valid status transitions
VALID_STATUSES = frozenset({
    "preflight_running",
    "preflight_complete",
    "preflight_failed",
    "conversing",
    "conversation_done",
    "running",
    "completed",
    "failed",
})


# ── Job Data Class ──


@dataclass
class Job:
    """Represents a single analysis job and its accumulated state."""

    id: str
    status: str = "preflight_running"
    created_at: float = 0.0
    project_path: Optional[Path] = None
    language: str = ""
    project_name: str = ""

    # Stage results (accumulated during lifecycle)
    session: Optional[SessionManager] = None
    ingestion: Optional[IngestionResult] = None
    matches: list[ProfileMatch] = field(default_factory=list)
    viability: Optional[ViabilityResult] = None
    interface_result: Optional[InterfaceResult] = None
    intent_string: str = ""
    constraints: Optional[OperationalConstraints] = None
    llm_config: Optional[LLMConfig] = None

    # Final result
    result: Optional[PipelineResult] = None
    error: str = ""

    # Progress tracking
    progress_scenarios_total: int = 0
    progress_scenarios_complete: int = 0
    progress_current_scenario: str = ""
    progress_stage: str = ""
    progress_start_time: float = 0.0

    # Conversation state
    conversation_summary: str = ""
    turn1_response: str = ""


# ── Job Store ──


class JobStore:
    """Thread-safe in-memory job store with TTL-based cleanup."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        """Create a new job with a unique ID."""
        job_id = f"j_{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, created_at=time.time())
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def active_count(self) -> int:
        """Count jobs that are currently running."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status in ("preflight_running", "running", "conversing")
            )

    def cleanup_expired(self) -> int:
        """Remove jobs older than JOB_TTL_SECONDS. Returns count removed."""
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for jid, job in self._jobs.items():
                if now - job.created_at > JOB_TTL_SECONDS:
                    expired.append(jid)

        removed = 0
        for jid in expired:
            with self._lock:
                job = self._jobs.pop(jid, None)
            if job is not None:
                _cleanup_job(job)
                removed += 1

        if removed:
            logger.info("Cleaned up %d expired job(s)", removed)
        return removed

    def remove(self, job_id: str) -> None:
        """Remove a job and clean up its resources."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            _cleanup_job(job)


def _cleanup_job(job: Job) -> None:
    """Clean up a job's resources (session, temp files)."""
    if job.session is not None:
        try:
            job.session.teardown()
        except Exception as exc:
            logger.debug("Session teardown failed for job %s: %s", job.id, exc)

    if job.project_path is not None and job.project_path.exists():
        try:
            shutil.rmtree(job.project_path, ignore_errors=True)
        except Exception as exc:
            logger.debug("Project cleanup failed for job %s: %s", job.id, exc)


# Singleton store
store = JobStore()
