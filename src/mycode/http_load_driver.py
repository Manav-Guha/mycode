"""HTTP Load Driver — HTTP stress testing Phase 3.

Drives concurrent load against discovered endpoints, generates degradation
curves and findings compatible with the existing report pipeline.

Uses urllib (stdlib) with ThreadPoolExecutor — no extra dependencies.

Pure Python.  No LLM dependency.
"""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mycode.constraints import OperationalConstraints
from mycode.endpoint_discovery import (
    DiscoveredEndpoint,
    DiscoveryResult,
    HttpTestRequest,
    ProbeResult,
    discover_endpoints,
    generate_requests,
    probe_endpoints,
)
from mycode.engine import (
    ExecutionEngineResult,
    ScenarioResult,
    StepResult,
)
from mycode.hysteresis import (
    PriorRunState,
    classify_with_hysteresis,
    finding_key,
)
from mycode.ingester import IngestionResult
from mycode.report import DegradationPoint, Finding
from mycode.server_manager import (
    FrameworkDetection,
    ServerInfo,
    detect_framework_entry,
    kill_process_group,
    start_server,
    stop_server,
)
from mycode.session import SessionManager

logger = logging.getLogger(__name__)


def _conns(n: int) -> str:
    """Format 'N concurrent connection(s)' with correct plural."""
    return f"{n} concurrent connection{'s' if n != 1 else ''}"


# ── Constants ──

# Default progressive load levels
_DEFAULT_LOAD_LEVELS = [1, 5, 10, 25, 50, 100]

# Per-level timeout (seconds) — if the entire level isn't done by this, stop
_LEVEL_TIMEOUT_SECONDS = 30

# Stop escalation thresholds
_MAX_ERROR_RATE = 0.50  # 50%
_MAX_MEDIAN_RESPONSE_MS = 10_000  # 10 seconds

# Number of rounds per concurrency level — median of rounds smooths variance
_ROUNDS_PER_LEVEL = 3

# If max/min ratio across rounds exceeds this, flag variance in findings
_HIGH_VARIANCE_RATIO = 1.5

# Per-request timeout
_REQUEST_TIMEOUT_SECONDS = 15

# Default time budget for the entire HTTP testing phase (seconds).
# Applied when no constraints are provided (e.g. batch/non-interactive mode).
_DEFAULT_HTTP_BUDGET = 120

# Skip endpoint if concurrency=1 response exceeds this (ms).
# Catches endpoints hanging on unconfigured external APIs (Stripe, DB, etc.).
_SLOW_BASELINE_MS = 10_000

# Framework → affected dependencies for findings
_FRAMEWORK_DEPS: dict[str, list[str]] = {
    "fastapi": ["fastapi", "uvicorn"],
    "flask": ["flask"],
    "streamlit": ["streamlit"],
    "express": ["express"],
    "react-scripts": ["react", "react-dom", "react-scripts"],
}


# ── Data Classes ──


@dataclass
class LoadLevelResult:
    """Result of testing at a single concurrency level."""

    concurrency: int
    total_requests: int = 0
    successful_requests: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    median_response_ms: float = 0.0
    p95_response_ms: float = 0.0
    memory_mb: float = 0.0
    response_times: list[float] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    server_crashed: bool = False
    high_variance: bool = False


@dataclass
class EndpointLoadResult:
    """Complete load test result for a single endpoint."""

    endpoint: HttpTestRequest
    levels: list[LoadLevelResult] = field(default_factory=list)
    breaking_point: int = 0  # concurrency level where it broke
    breaking_reason: str = ""  # "error_rate", "response_time", "crash"


@dataclass
class HttpLoadResult:
    """Complete result of HTTP load testing across all endpoints."""

    framework: str
    endpoint_results: list[EndpointLoadResult] = field(default_factory=list)
    server_startup_time: float = 0.0
    total_duration_ms: float = 0.0
    server_crash: bool = False
    server_crash_concurrency: int = 0
    startup_error: str = ""  # populated when server fails to start


# ── Load Levels ──


def _compute_load_levels(
    constraints: Optional[OperationalConstraints],
) -> list[int]:
    """Compute load levels based on user constraints.

    If user specified user_scale, test at user_scale and beyond to 3x,
    matching CLAUDE.md termination condition.
    """
    if constraints and constraints.user_scale:
        scale = constraints.user_scale
        levels = [1]
        # Add levels up to user_scale
        for lvl in [5, 10, 25, 50]:
            if lvl < scale:
                levels.append(lvl)
        levels.append(scale)
        # Test beyond: 1.5x, 2x, 3x
        for mult in (1.5, 2.0, 3.0):
            beyond = int(scale * mult)
            if beyond > levels[-1]:
                levels.append(beyond)
        return sorted(set(levels))
    return list(_DEFAULT_LOAD_LEVELS)


# ── Single Request ──


def _send_request(req: HttpTestRequest, timeout: int = _REQUEST_TIMEOUT_SECONDS) -> tuple[float, int, str]:
    """Send a single HTTP request and return (response_time_ms, status_code, error).

    Returns (elapsed_ms, status, error_str).
    """
    start = time.monotonic()
    try:
        data = req.body.encode("utf-8") if req.body else None
        http_req = urllib.request.Request(
            req.url,
            data=data,
            headers=req.headers,
            method=req.method,
        )
        with urllib.request.urlopen(http_req, timeout=timeout) as resp:
            resp.read()  # consume body
            elapsed = (time.monotonic() - start) * 1000
            return elapsed, resp.status, ""
    except urllib.error.HTTPError as e:
        elapsed = (time.monotonic() - start) * 1000
        return elapsed, e.code, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return elapsed, 0, str(e)[:200]


# ── Memory Monitoring ──


def _get_process_memory_mb(pid: int) -> float:
    """Get RSS memory of a process in MB.  Best-effort, returns 0 on failure."""
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    # Fallback: /proc on Linux
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (OSError, ValueError, IndexError):
        pass

    return 0.0


# ── Load Driver ──


def _drive_single_round(
    req: HttpTestRequest,
    concurrency: int,
    server_pid: int,
    timeout: int = _LEVEL_TIMEOUT_SECONDS,
) -> LoadLevelResult:
    """Run one round of concurrent requests at a single concurrency level."""
    response_times: list[float] = []
    errors: list[dict] = []
    successful = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_send_request, req, _REQUEST_TIMEOUT_SECONDS)
            for _ in range(concurrency)
        ]
        for future in as_completed(futures, timeout=timeout):
            try:
                elapsed_ms, status, error = future.result()
                response_times.append(elapsed_ms)
                if 200 <= status < 400:
                    successful += 1
                else:
                    errors.append({"status": status, "error": error})
            except Exception as e:
                errors.append({"status": 0, "error": str(e)[:200]})

    # Measure server memory after the load burst
    memory_mb = _get_process_memory_mb(server_pid) if server_pid > 0 else 0.0

    # response_times includes ALL completed requests (success + HTTP errors);
    # errors list only has non-2xx / exception entries.  Use response_times
    # plus exception-only errors (status==0) that never entered response_times.
    exception_only = sum(1 for e in errors if e["status"] == 0)
    total = len(response_times) + exception_only
    error_count = len(errors)
    error_rate = error_count / max(total, 1)

    # Compute percentiles
    sorted_times = sorted(response_times) if response_times else [0.0]
    median = sorted_times[len(sorted_times) // 2]
    p95_idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
    p95 = sorted_times[p95_idx]

    return LoadLevelResult(
        concurrency=concurrency,
        total_requests=total,
        successful_requests=successful,
        error_count=error_count,
        error_rate=error_rate,
        median_response_ms=round(median, 2),
        p95_response_ms=round(p95, 2),
        memory_mb=round(memory_mb, 2),
        response_times=sorted_times,
        errors=errors,
    )


def drive_load_level(
    req: HttpTestRequest,
    concurrency: int,
    server_pid: int,
    timeout: int = _LEVEL_TIMEOUT_SECONDS,
    rounds: int = _ROUNDS_PER_LEVEL,
) -> LoadLevelResult:
    """Drive concurrent load at a single concurrency level.

    Runs ``rounds`` independent bursts and returns the result whose median
    response time is the median across all rounds.  This smooths out
    infrastructure variance between runs.
    """
    round_results = []
    for _ in range(rounds):
        rr = _drive_single_round(req, concurrency, server_pid, timeout)
        round_results.append(rr)
        # If any round causes a server crash, stop immediately
        if rr.server_crashed:
            return rr

    # Pick the round with the median response time
    round_results.sort(key=lambda r: r.median_response_ms)
    median_idx = len(round_results) // 2
    chosen = round_results[median_idx]

    # Check for high variance between rounds
    if len(round_results) >= 2:
        lo = round_results[0].median_response_ms
        hi = round_results[-1].median_response_ms
        if lo > 0 and hi / lo > _HIGH_VARIANCE_RATIO:
            chosen.high_variance = True

    return chosen


def drive_endpoint(
    req: HttpTestRequest,
    levels: list[int],
    server: ServerInfo,
    deadline: Optional[float] = None,
) -> EndpointLoadResult:
    """Drive progressive load against a single endpoint.

    Escalates concurrency through ``levels`` until a stop condition is hit:
      - Error rate > 50%
      - Median response time > 10s
      - Server process crashed
      - Time budget exceeded (*deadline* is a ``time.monotonic()`` timestamp)
      - Baseline response > 10s (external dependency timeout)
    """
    result = EndpointLoadResult(endpoint=req)

    for concurrency in levels:
        # Check time budget
        if deadline is not None and time.monotonic() >= deadline:
            logger.info("HTTP budget exceeded — returning partial results for %s", req.description)
            break

        # Check if server is still alive
        if server.process.poll() is not None:
            result.breaking_point = concurrency
            result.breaking_reason = "crash"
            break

        level_result = drive_load_level(
            req, concurrency, server.process.pid,
        )
        result.levels.append(level_result)

        # Early abort: if concurrency=1 takes >10s, the endpoint is
        # hanging on an external dependency (Stripe, DB, third-party API).
        # Skip higher concurrency levels — they'll just waste budget.
        if (
            concurrency == levels[0]
            and level_result.median_response_ms > _SLOW_BASELINE_MS
        ):
            result.breaking_point = concurrency
            result.breaking_reason = "external_dependency_timeout"
            logger.info(
                "Endpoint %s took %.0fms at baseline — skipping (external dependency timeout)",
                req.description, level_result.median_response_ms,
            )
            break

        # Check stop conditions
        if level_result.error_rate > _MAX_ERROR_RATE:
            result.breaking_point = concurrency
            result.breaking_reason = "error_rate"
            break

        if level_result.median_response_ms > _MAX_MEDIAN_RESPONSE_MS:
            result.breaking_point = concurrency
            result.breaking_reason = "response_time"
            break

        # Check server crash after load
        if server.process.poll() is not None:
            level_result.server_crashed = True
            result.breaking_point = concurrency
            result.breaking_reason = "crash"
            break

    return result


def run_http_load_test(
    session: SessionManager,
    detection: FrameworkDetection,
    ingestion: IngestionResult,
    constraints: Optional[OperationalConstraints] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    timeout: Optional[int] = None,
) -> Optional[HttpLoadResult]:
    """Run the complete HTTP load testing pipeline.

    1. Start server (Phase 1)
    2. Discover endpoints (Phase 2)
    3. Generate and validate requests (Phase 2)
    4. Drive progressive load (Phase 3)
    5. Tear down server

    If *timeout* is set, the entire phase (server start through load
    driving) must complete within this budget.  Partial results are
    returned if the budget is exceeded.

    Returns None if the server couldn't start.
    """
    _progress = on_progress or (lambda msg: None)
    deadline = (time.monotonic() + timeout) if timeout else None

    _progress("Starting application server...")
    server_result = start_server(session, detection)
    if not server_result.success:
        logger.warning("Server failed to start: %s", server_result.error)
        return HttpLoadResult(
            framework=detection.framework,
            server_crash=True,
            startup_error=server_result.error,
        )

    server = server_result.server
    assert server is not None

    try:
        # Discover endpoints
        _progress("Discovering endpoints...")
        project_dir = session.project_copy_dir or Path(ingestion.project_path)
        discovery = discover_endpoints(detection, project_dir, ingestion)

        # Generate requests
        base_url = f"http://localhost:{server.port}"
        requests = generate_requests(discovery, base_url)
        if not requests:
            _progress("No endpoints discovered — testing root path only")
            requests = [HttpTestRequest(
                method="GET",
                url=f"{base_url}/",
                description="GET / — root path",
            )]

        # Validate endpoints
        _progress(f"Validating {len(requests)} endpoints...")
        probe_results = probe_endpoints(requests)

        # Filter to testable endpoints
        testable = [
            (req, probe)
            for req, probe in zip(requests, probe_results)
            if probe.testable
        ]

        if not testable:
            _progress("No testable endpoints found")
            # Still return results with probe info
            load_result = HttpLoadResult(
                framework=detection.framework,
                server_startup_time=server.startup_time,
            )
            return load_result

        # Compute load levels
        levels = _compute_load_levels(constraints)
        _progress(f"Testing {len(testable)} endpoints under load...")

        # Drive load on each testable endpoint
        endpoint_results: list[EndpointLoadResult] = []
        for i, (req, _) in enumerate(testable):
            # Check time budget before starting next endpoint
            if deadline is not None and time.monotonic() >= deadline:
                logger.info("HTTP budget exceeded — skipping remaining endpoints")
                break

            # Per-endpoint deadline: recalculate each iteration so fast
            # endpoints donate unused time to later ones.
            if deadline is not None:
                remaining = deadline - time.monotonic()
                endpoints_left = len(testable) - i
                per_ep = remaining / max(endpoints_left, 1)
                per_ep = max(15.0, min(60.0, per_ep))
                ep_deadline = min(time.monotonic() + per_ep, deadline)
            else:
                ep_deadline = None

            _progress(f"Testing endpoint {i + 1}/{len(testable)}: {req.description}")
            ep_result = drive_endpoint(req, levels, server, deadline=ep_deadline)
            endpoint_results.append(ep_result)

            # If server crashed, stop testing
            if ep_result.breaking_reason == "crash":
                break

        total_ms = (time.monotonic() - (time.monotonic() - server.startup_time)) * 1000

        load_result = HttpLoadResult(
            framework=detection.framework,
            endpoint_results=endpoint_results,
            server_startup_time=server.startup_time,
            total_duration_ms=round(total_ms, 2),
            server_crash=any(r.breaking_reason == "crash" for r in endpoint_results),
            server_crash_concurrency=next(
                (r.breaking_point for r in endpoint_results if r.breaking_reason == "crash"),
                0,
            ),
        )

        # Collect probe results for 500s (baseline findings)
        for req, probe in zip(requests, probe_results):
            if probe.is_finding:
                # Create a minimal endpoint result for baseline 500s
                endpoint_results.append(EndpointLoadResult(
                    endpoint=req,
                    breaking_point=0,
                    breaking_reason="baseline_error",
                ))

        return load_result

    finally:
        _progress("Shutting down server...")
        stop_server(server)


# ── Result Conversion: ScenarioResults ──


def http_results_to_scenario_results(
    load_result: HttpLoadResult,
) -> list[ScenarioResult]:
    """Convert HTTP load results to ScenarioResult objects.

    These are appended to the existing execution engine results so the
    report generator processes them without modification.
    """
    results: list[ScenarioResult] = []

    for ep_result in load_result.endpoint_results:
        req = ep_result.endpoint
        # Build scenario name from endpoint
        path_clean = req.url.split("//", 1)[-1].split("/", 1)[-1] if "//" in req.url else "root"
        path_clean = path_clean.replace("/", "_").strip("_") or "root"
        scenario_name = f"http_{req.method.lower()}_{path_clean}"

        steps: list[StepResult] = []
        peak_memory = 0.0
        total_errors = 0

        for lvl in ep_result.levels:
            step = StepResult(
                step_name=f"concurrent_{lvl.concurrency}",
                parameters={
                    "concurrency": lvl.concurrency,
                    "endpoint": req.description,
                },
                execution_time_ms=lvl.median_response_ms,
                memory_peak_mb=lvl.memory_mb,
                error_count=lvl.error_count,
                errors=lvl.errors[:5],  # cap stored errors
                measurements={
                    "median_response_ms": lvl.median_response_ms,
                    "p95_response_ms": lvl.p95_response_ms,
                    "error_rate": lvl.error_rate,
                    "successful_requests": lvl.successful_requests,
                    "total_requests": lvl.total_requests,
                },
            )
            steps.append(step)
            peak_memory = max(peak_memory, lvl.memory_mb)
            total_errors += lvl.error_count

        status = "completed"
        failure_reason = ""
        if ep_result.breaking_reason == "crash":
            status = "failed"
            failure_reason = "server_crash"
        elif ep_result.breaking_reason:
            status = "partial"

        results.append(ScenarioResult(
            scenario_name=scenario_name,
            scenario_category="http_load_testing",
            status=status,
            steps=steps,
            total_execution_time_ms=sum(s.execution_time_ms for s in steps),
            peak_memory_mb=peak_memory,
            total_errors=total_errors,
            failure_reason=failure_reason,
            summary=_build_http_summary(ep_result),
        ))

    return results


def _build_http_summary(ep_result: EndpointLoadResult) -> str:
    """Build a plain-language summary for an HTTP endpoint test."""
    req = ep_result.endpoint
    desc = req.description

    if not ep_result.levels:
        return f"Endpoint {desc} could not be tested."

    max_ok = 0
    for lvl in ep_result.levels:
        if lvl.error_rate <= _MAX_ERROR_RATE and lvl.median_response_ms <= _MAX_MEDIAN_RESPONSE_MS:
            max_ok = lvl.concurrency

    if ep_result.breaking_reason == "crash":
        return (
            f"Server crashed at {_conns(ep_result.breaking_point)} "
            f"while testing {desc}."
        )
    if ep_result.breaking_reason == "error_rate":
        return (
            f"Endpoint {desc} failed at {_conns(ep_result.breaking_point)} "
            f"(error rate exceeded 50%). Handled {max_ok} cleanly."
        )
    if ep_result.breaking_reason == "response_time":
        return (
            f"Endpoint {desc} became unresponsive at {_conns(ep_result.breaking_point)} "
            f"(>10s response time). Handled {max_ok} cleanly."
        )
    return f"Endpoint {desc} handled all tested load levels (up to {max_ok} concurrent)."


# ── Result Conversion: DegradationPoints ──


def http_results_to_degradation_points(
    load_result: HttpLoadResult,
) -> list[DegradationPoint]:
    """Convert HTTP load results to DegradationPoint objects for report curves."""
    points: list[DegradationPoint] = []

    for ep_result in load_result.endpoint_results:
        if not ep_result.levels:
            continue

        req = ep_result.endpoint
        path_clean = req.url.split("//", 1)[-1].split("/", 1)[-1] if "//" in req.url else "root"
        path_clean = path_clean.replace("/", "_").strip("_") or "root"
        scenario_name = f"http_{req.method.lower()}_{path_clean}"

        # Response time curve
        time_steps = [
            (f"{lvl.concurrency} concurrent", lvl.median_response_ms)
            for lvl in ep_result.levels
        ]
        breaking = (
            f"{ep_result.breaking_point} concurrent"
            if ep_result.breaking_point else ""
        )
        points.append(DegradationPoint(
            scenario_name=scenario_name,
            metric="response_time_ms",
            steps=time_steps,
            breaking_point=breaking,
            description=_describe_response_curve(ep_result),
        ))

        # Memory curve — always emit when measurements exist.
        # Flat-curve verdict logic (commit 25fa718) handles near-flat
        # curves by showing "Stable" instead of inheriting finding severity.
        mem_values = [lvl.memory_mb for lvl in ep_result.levels if lvl.memory_mb > 0]
        if mem_values:
            mem_steps = [
                (f"{lvl.concurrency} concurrent", lvl.memory_mb)
                for lvl in ep_result.levels
            ]
            # Explanatory description for flat curves
            first_mb = mem_values[0] if mem_values[0] > 0 else 1.0
            last_mb = mem_values[-1]
            avg_mb = sum(mem_values) / len(mem_values)
            if len(mem_values) >= 2 and first_mb > 0 and last_mb / first_mb < 1.15:
                mem_desc = (
                    f"Memory usage stays steady under load at {avg_mb:.0f}MB. "
                    "The capacity issue is the baseline memory per process, "
                    "not memory growth."
                )
            else:
                mem_desc = ""
            points.append(DegradationPoint(
                scenario_name=scenario_name,
                metric="memory_peak_mb",
                steps=mem_steps,
                breaking_point=breaking,
                description=mem_desc,
            ))

    return points


def _describe_response_curve(ep_result: EndpointLoadResult) -> str:
    """Describe the response time degradation curve."""
    if len(ep_result.levels) < 2:
        return ""
    baseline_ms = _warmed_up_baseline(ep_result.levels)
    last = ep_result.levels[-1]
    if baseline_ms > 0:
        ratio = last.median_response_ms / baseline_ms
        if ratio > 5:
            return (
                f"Response time increased {ratio:.0f}x from "
                f"{baseline_ms:.0f}ms to {last.median_response_ms:.0f}ms"
            )
        if ratio > 2:
            return (
                f"Response time doubled from {baseline_ms:.0f}ms "
                f"to {last.median_response_ms:.0f}ms"
            )
    return ""


# ── Result Conversion: Findings ──


def http_results_to_findings(
    load_result: HttpLoadResult,
    constraints: Optional[OperationalConstraints] = None,
    probe_results: Optional[list[ProbeResult]] = None,
    prior_state: Optional[PriorRunState] = None,
) -> list[Finding]:
    """Convert HTTP load results to Finding objects for the report."""
    findings: list[Finding] = []
    deps = _FRAMEWORK_DEPS.get(load_result.framework, [load_result.framework])
    user_scale = constraints.user_scale if constraints else None

    # Server crash during load
    if load_result.server_crash:
        crash_at = load_result.server_crash_concurrency
        desc = f"Your application server crashed under load at {_conns(crash_at)}."
        if user_scale:
            if crash_at < user_scale:
                desc += (
                    f" This is well below your expected {user_scale} concurrent users."
                )
            else:
                desc += (
                    f" This is above your expected {user_scale} concurrent users, "
                    f"but crash resilience is still important."
                )
        findings.append(Finding(
            title="Server crashed under load",
            severity="critical",
            category="http_load_testing",
            description=desc,
            affected_dependencies=list(deps),
            _finding_type="scenario_failed",
        ))

    has_high_variance = False
    for ep_result in load_result.endpoint_results:
        finding = _endpoint_to_finding(ep_result, deps, user_scale, prior_state)
        if finding:
            findings.append(finding)
        if any(lvl.high_variance for lvl in ep_result.levels):
            has_high_variance = True

    # Append variance note to all HTTP findings if any level showed instability
    if has_high_variance:
        _VARIANCE_NOTE = (
            " Results showed significant variance between test rounds — "
            "infrastructure conditions may affect these measurements."
        )
        for f in findings:
            if f.category == "http_load_testing":
                f.description += _VARIANCE_NOTE

    # Memory capacity finding — independent of per-endpoint response/error findings
    if user_scale:
        mem_finding = _memory_capacity_finding(
            load_result, deps, user_scale, prior_state,
        )
        if mem_finding:
            findings.append(mem_finding)

    # Baseline 500 errors from probe
    if probe_results:
        for probe in probe_results:
            if probe.is_finding:
                probe_label = _humanize_path(probe.endpoint.path)
                findings.append(Finding(
                    title=f"Server error on {probe_label} before load testing",
                    severity="critical",
                    category="http_load_testing",
                    description=(
                        f"Your {probe_label} returned "
                        f"HTTP {probe.status_code} even before load testing — this is a "
                        f"bug in your application, not a scaling issue."
                    ),
                    affected_dependencies=list(deps),
                    _finding_type="scenario_failed",
                ))

    return findings


def _humanize_path(raw_path: str) -> str:
    """Turn a raw URL path into a human-readable label for finding titles.

    '/' → 'your application'   (root is the homepage / single-page app)
    '/api/users' → '/api/users' (already meaningful)
    """
    if raw_path in ("/", ""):
        return "your application"
    return raw_path


def _endpoint_to_finding(
    ep_result: EndpointLoadResult,
    deps: list[str],
    user_scale: Optional[int],
    prior_state: Optional[PriorRunState] = None,
) -> Optional[Finding]:
    """Convert a single endpoint load result to a Finding, or None if it passed."""
    req = ep_result.endpoint
    path = req.url.split("//", 1)[-1].split("/", 1)[-1] if "//" in req.url else "/"
    if not path.startswith("/"):
        path = "/" + path
    label = _humanize_path(path)

    if ep_result.breaking_reason == "crash":
        return None  # handled at top level

    if ep_result.breaking_reason == "baseline_error":
        return None  # handled by probe findings

    if ep_result.breaking_reason == "external_dependency_timeout":
        path_desc = f"Your {label} endpoint" if path != "/" else "Your application"
        return Finding(
            title=f"Endpoint {label} skipped — slow response",
            severity="info",
            category="http_load_testing",
            description=(
                f"{path_desc} took over 10 seconds to respond even at "
                f"1 concurrent connection. This usually means it depends "
                f"on an external service (payment API, database, third-party "
                f"API) that isn't configured or is unreachable in this "
                f"environment. No further load testing was attempted for "
                f"this endpoint."
            ),
            affected_dependencies=list(deps),
        )

    if ep_result.breaking_reason in ("error_rate", "response_time"):
        bp = ep_result.breaking_point
        severity = "critical"
        verb = (
            "becomes unresponsive" if ep_result.breaking_reason == "response_time"
            else "fails with errors"
        )
        title = (
            "Application degrades under concurrent load" if path == "/"
            else f"Endpoint {label} degrades under concurrent load"
        )
        if path == "/":
            desc = f"Your application {verb} at {_conns(bp)}."
        else:
            desc = f"Your {label} endpoint {verb} at {_conns(bp)}."
        if user_scale:
            ratio = bp / user_scale if user_scale > 0 else float("inf")
            # Look up prior severity for hysteresis
            fkey = finding_key(title, "http_load_testing")
            prior_sev = (
                prior_state.finding_severities.get(fkey)
                if prior_state else None
            )
            severity = classify_with_hysteresis(
                measurement=ratio,
                thresholds=[(1.0, "critical"), (2.0, "warning")],
                default_severity="info",
                prior_severity=prior_sev,
            )
            if severity == "critical":
                desc += (
                    f" This is at or below your expected {user_scale} concurrent users — "
                    f"your users will experience failures."
                )
            elif severity == "warning":
                desc += (
                    f" This is {ratio:.1f}x your expected {user_scale} users — "
                    f"limited headroom for traffic spikes."
                )
            else:
                desc += (
                    f" This is {ratio:.1f}x your expected {user_scale} users — "
                    f"comfortable headroom."
                )
        return Finding(
            title=title,
            severity=severity,
            category="http_load_testing",
            description=desc,
            affected_dependencies=list(deps),
            _finding_type="scenario_failed" if severity == "critical" else "errors_during",
            _load_level=bp,
        )

    # Check for degradation without hard failure
    if len(ep_result.levels) >= 2:
        baseline_ms = _warmed_up_baseline(ep_result.levels)
        last = ep_result.levels[-1]

        # Response time degradation (>5x increase = WARNING)
        if baseline_ms > 0:
            ratio = last.median_response_ms / baseline_ms
            if ratio > 5:
                desc = (
                    f"Response time on your application increases {ratio:.0f}x "
                    f"under load ({baseline_ms:.0f}ms baseline → "
                    f"{last.median_response_ms:.0f}ms at "
                    f"{last.concurrency} concurrent users)."
                )
                # Find the concurrency where degradation became significant
                degrade_at = _find_degradation_onset(ep_result.levels)
                if degrade_at:
                    desc += (
                        f" At {_conns(degrade_at)}, response "
                        f"time begins degrading significantly."
                    )
                if user_scale:
                    if degrade_at and degrade_at <= user_scale:
                        desc += (
                            f" This is at or below your expected "
                            f"{user_scale} concurrent users."
                        )
                return Finding(
                    title=f"Response time degradation on {label}",
                    severity="warning",
                    category="http_load_testing",
                    description=desc,
                    affected_dependencies=list(deps),
                    _finding_type="errors_during",
                    _load_level=degrade_at or last.concurrency,
                )

        # Error rate above 10% at any level (but below 50% hard stop)
        worst_error_level = max(ep_result.levels, key=lambda l: l.error_rate)
        if worst_error_level.error_rate > 0.10:
            if path == "/":
                desc = (
                    f"Your application has a {worst_error_level.error_rate:.0%} "
                    f"error rate at {worst_error_level.concurrency} concurrent "
                    f"connections."
                )
            else:
                desc = (
                    f"Your {label} endpoint has a {worst_error_level.error_rate:.0%} "
                    f"error rate at {worst_error_level.concurrency} concurrent "
                    f"connections."
                )
            if user_scale:
                if worst_error_level.concurrency <= user_scale:
                    desc += (
                        f" This is within your expected {user_scale} "
                        f"concurrent users — some users will see errors."
                    )
            return Finding(
                title=f"Elevated error rate on {label}",
                severity="warning",
                category="http_load_testing",
                description=desc,
                affected_dependencies=list(deps),
                _finding_type="errors_during",
                _load_level=worst_error_level.concurrency,
            )

    return None  # Passed all levels


def _warmed_up_baseline(levels: list[LoadLevelResult]) -> float:
    """Return the warmed-up baseline response time, excluding cold-start outlier.

    The first request at concurrency=1 often hits server/JIT warm-up and is
    significantly slower than steady-state.  Use the minimum of the first 3
    data points (or fewer if there aren't 3) as the true baseline.
    """
    if not levels:
        return 0.0
    candidates = [lvl.median_response_ms for lvl in levels[:3] if lvl.median_response_ms > 0]
    return min(candidates) if candidates else 0.0


def _find_degradation_onset(levels: list[LoadLevelResult]) -> int:
    """Find the concurrency level where response time first doubles from baseline."""
    if not levels:
        return 0
    baseline = _warmed_up_baseline(levels)
    if baseline <= 0:
        return 0
    for lvl in levels[1:]:
        if lvl.median_response_ms > baseline * 2:
            return lvl.concurrency
    return 0


def _memory_capacity_finding(
    load_result: HttpLoadResult,
    deps: list[str],
    user_scale: int,
    prior_state: Optional[PriorRunState] = None,
) -> Optional[Finding]:
    """Generate a finding when per-process memory limits capacity below user_scale.

    Uses the baseline (first-level) memory measurement across all endpoints.
    Estimates capacity on a typical 2GB server (2048 - 512 MB OS overhead).

    Severity:
      - CRITICAL: estimated capacity < 25% of user_scale
      - WARNING:  estimated capacity < 75% of user_scale
      - None:     estimated capacity >= user_scale (no finding)
    """
    # Collect baseline memory from all endpoints
    baseline_mems = []
    for ep in load_result.endpoint_results:
        if ep.levels and ep.levels[0].memory_mb > 0:
            baseline_mems.append(ep.levels[0].memory_mb)
    if not baseline_mems:
        return None

    baseline_mem = max(baseline_mems)  # worst case across endpoints
    available_mb = 2048 - 512
    estimated_capacity = int(available_mb / baseline_mem)

    title = "Memory baseline limits concurrent capacity"
    ratio = estimated_capacity / user_scale if user_scale > 0 else float("inf")

    # Look up prior severity for hysteresis
    fkey = finding_key(title, "http_load_testing")
    prior_sev = (
        prior_state.finding_severities.get(fkey)
        if prior_state else None
    )
    severity = classify_with_hysteresis(
        measurement=ratio,
        thresholds=[(0.25, "critical"), (0.75, "warning")],
        default_severity=None,  # no finding when above 0.75
        prior_severity=prior_sev,
    )
    if severity is None:
        return None

    desc = (
        f"Your application uses {baseline_mem:.0f}MB per process. "
        f"On a 2GB server, this supports approximately "
        f"{estimated_capacity} concurrent sessions — "
        f"well below your expected {user_scale} users. "
        f"You will need to optimize memory usage or scale your "
        f"infrastructure."
    )

    return Finding(
        title="Memory baseline limits concurrent capacity",
        severity=severity,
        category="http_load_testing",
        description=desc,
        affected_dependencies=list(deps),
        _finding_type="scenario_failed" if severity == "critical" else "errors_during",
    )


# ── Pipeline Integration Helper ──


def run_http_testing_phase(
    session: SessionManager,
    ingestion: IngestionResult,
    execution: ExecutionEngineResult,
    language: str,
    constraints: Optional[OperationalConstraints] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    prior_state: Optional[PriorRunState] = None,
) -> ExecutionEngineResult:
    """Run HTTP testing and merge results into the existing execution result.

    Called after the callable harness phase. If the project has a server
    framework, starts the server, discovers endpoints, drives load, and
    appends findings to the existing result.

    Returns the (potentially augmented) ExecutionEngineResult.
    """
    _progress = on_progress or (lambda msg: None)

    project_dir = session.project_copy_dir or Path(ingestion.project_path)
    detection = detect_framework_entry(ingestion, project_dir, language)

    if detection is None:
        logger.debug("No server framework detected — skipping HTTP testing")
        return execution

    # Block server startup if JS dependency installation failed
    if session.js_deps_installed is False:
        error_msg = session.js_deps_error or "npm install failed"
        logger.warning("HTTP phase: skipping — JS deps not installed: %s", error_msg)
        _progress("Skipping HTTP testing — dependency installation failed")
        deps = _FRAMEWORK_DEPS.get(detection.framework, [detection.framework])
        execution.http_findings.append(Finding(
            title="Dependency installation failed",
            severity="critical",
            category="http_load_testing",
            description=(
                f"npm install failed for this project: {error_msg}. "
                f"Without installed dependencies, the {detection.framework} "
                f"server cannot start. No HTTP stress testing was attempted."
            ),
            affected_dependencies=list(deps),
            _finding_type="scenario_failed",
        ))
        return execution

    _progress(f"Detected {detection.framework} server — running HTTP stress tests...")
    logger.info("HTTP phase: detected %s, entry=%s", detection.framework, detection.entry_file)

    http_timeout = (
        constraints.timeout_per_scenario
        if constraints and constraints.timeout_per_scenario
        else _DEFAULT_HTTP_BUDGET
    )
    load_result = run_http_load_test(
        session=session,
        detection=detection,
        ingestion=ingestion,
        constraints=constraints,
        on_progress=on_progress,
        timeout=http_timeout,
    )

    if load_result is None:
        logger.warning("HTTP phase: run_http_load_test returned None")
        execution.warnings.append(
            "HTTP testing skipped: server could not start"
        )
        return execution

    # Server failed to start — generate CRITICAL finding
    if load_result.server_crash and not load_result.endpoint_results:
        deps = _FRAMEWORK_DEPS.get(
            detection.framework, [detection.framework]
        )
        error_detail = load_result.startup_error or "unknown error"
        execution.http_findings.append(Finding(
            title="Application server could not start",
            severity="critical",
            category="http_load_testing",
            description=(
                f"myCode detected a {detection.framework} application but "
                f"the server failed to start: {error_detail}. Your "
                f"application will not start on modern infrastructure. "
                f"No users can access your app until this is resolved."
            ),
            affected_dependencies=list(deps),
            _finding_type="scenario_failed",
        ))
        # http_ran stays False — server never ran, so runtime context
        # messaging for incomplete tests should NOT reference HTTP findings
        return execution

    # Log detailed results
    logger.info(
        "HTTP phase: server started in %.1fs, %d endpoints tested, crash=%s",
        load_result.server_startup_time,
        len(load_result.endpoint_results),
        load_result.server_crash,
    )
    for ep in load_result.endpoint_results:
        lvl_summary = ", ".join(
            f"{l.concurrency}c={l.median_response_ms:.0f}ms/{l.error_rate:.0%}err"
            for l in ep.levels
        )
        logger.info(
            "HTTP phase: endpoint %s — %d levels [%s] break=%s",
            ep.endpoint.description, len(ep.levels), lvl_summary,
            ep.breaking_reason or "none",
        )

    # Convert to scenario results and append
    http_scenarios = http_results_to_scenario_results(load_result)
    execution.scenario_results.extend(http_scenarios)

    # Generate HTTP-specific findings and degradation points
    execution.http_findings = http_results_to_findings(
        load_result, constraints, prior_state=prior_state,
    )
    execution.http_degradation_points = http_results_to_degradation_points(
        load_result,
    )

    # If no findings were generated but endpoints were tested, add an INFO
    # finding so the HTTP phase is visible in the report.
    if not execution.http_findings and load_result.endpoint_results:
        deps = _FRAMEWORK_DEPS.get(
            load_result.framework, [load_result.framework]
        )
        max_concurrency = 0
        for ep in load_result.endpoint_results:
            if ep.levels:
                max_concurrency = max(max_concurrency, ep.levels[-1].concurrency)
        execution.http_findings.append(Finding(
            title="Application handled HTTP load without issues",
            severity="info",
            category="http_load_testing",
            description=(
                f"myCode tested your {detection.framework} application "
                f"under load up to {max_concurrency} concurrent "
                f"connections. No degradation or errors were detected."
            ),
            affected_dependencies=list(deps),
            _finding_type="clean",
        ))

    logger.info(
        "HTTP phase: %d findings, %d degradation points, %d scenario results",
        len(execution.http_findings),
        len(execution.http_degradation_points),
        len(http_scenarios),
    )

    execution.http_ran = True

    # Update execution counts
    for sr in http_scenarios:
        if sr.status == "completed":
            execution.scenarios_completed += 1
        elif sr.status in ("failed", "partial"):
            execution.scenarios_failed += 1

    return execution
