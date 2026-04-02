"""CI gate endpoint handlers for myCode.

Provides a single-submission + polling model for CI runners:

    POST /api/ci/check      → submit repo, get job_id
    GET  /api/ci/result/X    → poll until pass/fail
    POST /api/ci/override/X  → flip fail → pass (audit-trailed)

All handlers return plain dicts (serialised to JSON by FastAPI layer).
An ``_status`` key signals the HTTP status code to the app layer.
"""

from __future__ import annotations

import logging

from mycode.web.ci_keys import (
    create_api_key,
    get_override,
    list_api_keys,
    list_overrides,
    record_key_usage,
    record_override,
    validate_api_key,
)
from mycode.web.jobs import store
from mycode.web.routes import (
    handle_analyze,
    handle_report,
    handle_status,
    handle_submit_intent,
    handle_preflight,
)

logger = logging.getLogger(__name__)

_VALID_THRESHOLDS = frozenset({"report_only", "critical", "warning", "any"})
_TIER_TO_DEPTH = {1: "quick", 2: "standard", 3: "deep"}


# ── POST /api/ci/check ──


def handle_ci_check(
    repo_url: str,
    threshold: str,
    tier: int,
    api_key: str,
) -> dict:
    """Submit a CI check: preflight → non-interactive intent → analyze."""

    # Validate API key
    if not validate_api_key(api_key):
        return {"error": "Invalid or missing API key.", "_status": 403}

    # Validate threshold
    if threshold not in _VALID_THRESHOLDS:
        return {
            "error": f"Invalid threshold: {threshold!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_THRESHOLDS))}.",
            "_status": 400,
        }

    # Validate tier
    if tier not in _TIER_TO_DEPTH:
        return {
            "error": f"Invalid tier: {tier!r}. Must be 1, 2, or 3.",
            "_status": 400,
        }

    if not repo_url or not repo_url.strip():
        return {"error": "repo_url is required.", "_status": 400}

    # Stage 1-4.5: Preflight (reuse existing handler)
    preflight = handle_preflight(github_url=repo_url, source="ci")

    # Check for preflight error
    if preflight.error:
        return {"error": f"Preflight failed: {preflight.error}", "_status": 400}

    job_id = preflight.job_id
    job = store.get(job_id)
    if job is None:
        return {"error": "Job creation failed.", "_status": 500}

    # Store CI metadata on the job
    job.ci_threshold = threshold

    # Non-interactive intent submission (maps tier → analysis_depth)
    depth = _TIER_TO_DEPTH[tier]
    answers = {"analysis_depth": depth}
    intent_result = handle_submit_intent(job_id, answers)
    if "error" in intent_result:
        return {"error": intent_result["error"], "_status": 400}

    # Start analysis
    analyze_result = handle_analyze(job_id)
    if analyze_result.error:
        return {"error": analyze_result.error, "_status": 400}

    # Track key usage
    record_key_usage(api_key)

    return {"job_id": job_id, "status": "queued"}


# ── GET /api/ci/result/{job_id} ──


def handle_ci_result(job_id: str) -> dict:
    """Return CI pass/fail based on findings vs threshold."""
    job = store.get(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found.", "_status": 404}

    # Still running?
    if job.status in (
        "preflight_running", "preflight_complete",
        "conversing", "conversation_done", "running",
    ):
        return {"status": "running"}

    # Failed?
    if job.status == "failed":
        return {"status": "error", "error": job.error or "Analysis failed."}

    # Completed — count findings
    critical = 0
    warning = 0
    info = 0
    scenarios_run = 0

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

    findings_count = {"critical": critical, "warning": warning, "info": info}
    total_findings = critical + warning + info
    summary = (
        f"{critical} critical, {warning} warning, {info} info "
        f"across {scenarios_run} scenarios"
    )

    threshold = job.ci_threshold or "report_only"

    # Check for override
    override_record = get_override(job_id)
    if override_record:
        return {
            "status": "pass",
            "summary": summary,
            "findings_count": findings_count,
            "threshold_applied": threshold,
            "override": True,
            "override_reason": override_record.get("reason", ""),
            "report_url": f"/api/report/{job_id}",
        }

    # Evaluate threshold
    if threshold == "report_only":
        status = "report_only"
    elif threshold == "critical":
        status = "fail" if critical > 0 else "pass"
    elif threshold == "warning":
        status = "fail" if (critical + warning) > 0 else "pass"
    elif threshold == "any":
        status = "fail" if total_findings > 0 else "pass"
    else:
        status = "pass"

    return {
        "status": status,
        "summary": summary,
        "findings_count": findings_count,
        "threshold_applied": threshold,
        "override": False,
        "report_url": f"/api/report/{job_id}",
    }


# ── POST /api/ci/override/{job_id} ──


def handle_ci_override(job_id: str, api_key: str, reason: str) -> dict:
    """Override a failed CI check (audit-trailed)."""
    if not validate_api_key(api_key):
        return {"error": "Invalid or missing API key.", "_status": 403}

    job = store.get(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found.", "_status": 404}

    if job.status != "completed":
        return {
            "error": f"Cannot override — job is in state '{job.status}'.",
            "_status": 400,
        }

    record_override(job_id, api_key, reason)
    return {"ok": True, "job_id": job_id}


# ── Admin: key management ──


def handle_ci_keys_create() -> dict:
    """Generate a new CI API key."""
    key = create_api_key()
    return {"key": key}


def handle_ci_keys_list() -> dict:
    """List all CI API keys (prefix only)."""
    return {"keys": list_api_keys()}


def handle_admin_overrides() -> dict:
    """List all CI overrides (audit trail)."""
    return {"overrides": list_overrides()}
