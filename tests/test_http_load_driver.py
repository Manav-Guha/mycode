"""Tests for http_load_driver.py — HTTP stress testing Phase 3.

Tests load driving, degradation curve generation, finding generation,
engine integration, and Streamlit page load testing.
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mycode.constraints import OperationalConstraints
from mycode.endpoint_discovery import (
    DiscoveredEndpoint,
    DiscoveryResult,
    HttpTestRequest,
    ProbeResult,
    StreamlitPage,
)
from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.http_load_driver import (
    EndpointLoadResult,
    HttpLoadResult,
    LoadLevelResult,
    _build_http_summary,
    _compute_load_levels,
    _drive_single_round,
    _endpoint_to_finding,
    _describe_response_curve,
    _find_degradation_onset,
    _get_process_memory_mb,
    _memory_capacity_finding,
    _warmed_up_baseline,
    _send_request,
    _ROUNDS_PER_LEVEL,
    _HIGH_VARIANCE_RATIO,
    drive_endpoint,
    drive_load_level,
    http_results_to_degradation_points,
    http_results_to_findings,
    http_results_to_scenario_results,
    run_http_testing_phase,
)
from mycode.ingester import FileAnalysis, IngestionResult
from mycode.report import DegradationPoint, Finding
from mycode.server_manager import FrameworkDetection, ServerInfo


# ── Helpers ──


def _make_load_level(
    concurrency=10, median_ms=50.0, p95_ms=100.0, error_rate=0.0,
    error_count=0, memory_mb=50.0, total=10, successful=10,
    server_crashed=False,
):
    return LoadLevelResult(
        concurrency=concurrency,
        total_requests=total,
        successful_requests=successful,
        error_count=error_count,
        error_rate=error_rate,
        median_response_ms=median_ms,
        p95_response_ms=p95_ms,
        memory_mb=memory_mb,
        server_crashed=server_crashed,
    )


def _make_endpoint_result(
    path="/api/test", method="GET", levels=None,
    breaking_point=0, breaking_reason="",
):
    req = HttpTestRequest(
        method=method,
        url=f"http://localhost:8000{path}",
        description=f"{method} {path}",
    )
    return EndpointLoadResult(
        endpoint=req,
        levels=levels or [],
        breaking_point=breaking_point,
        breaking_reason=breaking_reason,
    )


# ═══════════════════════════════════════════════════════════════════════
# Load Level Computation
# ═══════════════════════════════════════════════════════════════════════


class TestComputeLoadLevels:
    def test_default_levels(self):
        levels = _compute_load_levels(None)
        assert levels == [1, 5, 10, 25, 50, 100]

    def test_user_scale_sets_upper_bound(self):
        c = OperationalConstraints(user_scale=20)
        levels = _compute_load_levels(c)
        assert 20 in levels
        assert 60 in levels  # 3x
        assert all(l <= 60 for l in levels)

    def test_user_scale_small(self):
        c = OperationalConstraints(user_scale=5)
        levels = _compute_load_levels(c)
        assert 5 in levels
        assert 15 in levels  # 3x
        assert 1 in levels

    def test_user_scale_large(self):
        c = OperationalConstraints(user_scale=100)
        levels = _compute_load_levels(c)
        assert 100 in levels
        assert 300 in levels  # 3x


# ═══════════════════════════════════════════════════════════════════════
# Single Request (mocked)
# ═══════════════════════════════════════════════════════════════════════


class TestSendRequest:
    @patch("mycode.http_load_driver.urllib.request.urlopen")
    def test_successful_request(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        elapsed, status, error = _send_request(req)
        assert status == 200
        assert error == ""
        assert elapsed >= 0

    @patch("mycode.http_load_driver.urllib.request.urlopen")
    def test_error_response(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "ISE", {}, None
        )

        req = HttpTestRequest(method="GET", url="http://localhost:8000/broken")
        elapsed, status, error = _send_request(req)
        assert status == 500
        assert "500" in error

    @patch("mycode.http_load_driver.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("timeout")

        req = HttpTestRequest(method="GET", url="http://localhost:8000/slow")
        elapsed, status, error = _send_request(req)
        assert status == 0
        assert "timeout" in error.lower()


# ═══════════════════════════════════════════════════════════════════════
# Load Level Driver (mocked)
# ═══════════════════════════════════════════════════════════════════════


class TestDriveLoadLevel:
    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=50.0)
    @patch("mycode.http_load_driver._send_request")
    def test_single_concurrency(self, mock_send, mock_mem):
        mock_send.return_value = (25.0, 200, "")

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=1, server_pid=123, rounds=1)

        assert result.concurrency == 1
        assert result.total_requests == 1
        assert result.successful_requests == 1
        assert result.error_count == 0
        assert result.error_rate == 0.0
        assert result.median_response_ms == 25.0
        assert result.memory_mb == 50.0

    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=100.0)
    @patch("mycode.http_load_driver._send_request")
    def test_multiple_concurrency(self, mock_send, mock_mem):
        mock_send.return_value = (50.0, 200, "")

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=5, server_pid=123, rounds=1)

        assert result.concurrency == 5
        assert result.total_requests == 5
        assert result.successful_requests == 5

    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=50.0)
    @patch("mycode.http_load_driver._send_request")
    def test_mixed_success_and_errors(self, mock_send, mock_mem):
        # All requests return 500 → 100% error rate
        mock_send.return_value = (100.0, 500, "HTTP 500")

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=4, server_pid=123, rounds=1)

        assert result.total_requests == 4
        assert result.error_count == 4
        assert result.error_rate == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Endpoint Driver
# ═══════════════════════════════════════════════════════════════════════


class TestDriveEndpoint:
    @patch("mycode.http_load_driver.drive_load_level")
    def test_stops_on_error_rate(self, mock_drive):
        mock_drive.side_effect = [
            _make_load_level(concurrency=1, error_rate=0.0),
            _make_load_level(concurrency=5, error_rate=0.0),
            _make_load_level(concurrency=10, error_rate=0.6),
        ]

        server = MagicMock()
        server.process.poll.return_value = None
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/api/test")
        result = drive_endpoint(req, [1, 5, 10, 25], server)

        assert result.breaking_point == 10
        assert result.breaking_reason == "error_rate"
        assert len(result.levels) == 3

    @patch("mycode.http_load_driver.drive_load_level")
    def test_stops_on_response_time(self, mock_drive):
        mock_drive.side_effect = [
            _make_load_level(concurrency=1, median_ms=100),
            _make_load_level(concurrency=5, median_ms=5000),
            _make_load_level(concurrency=10, median_ms=15000),
        ]

        server = MagicMock()
        server.process.poll.return_value = None
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_endpoint(req, [1, 5, 10], server)

        assert result.breaking_point == 10
        assert result.breaking_reason == "response_time"

    @patch("mycode.http_load_driver.drive_load_level")
    def test_stops_on_server_crash(self, mock_drive):
        mock_drive.return_value = _make_load_level(concurrency=1)

        server = MagicMock()
        # First poll: running, second poll: crashed
        server.process.poll.side_effect = [None, 1]
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_endpoint(req, [1, 5], server)

        assert result.breaking_reason == "crash"

    @patch("mycode.http_load_driver.drive_load_level")
    def test_passes_all_levels(self, mock_drive):
        mock_drive.side_effect = [
            _make_load_level(concurrency=c, median_ms=50, error_rate=0)
            for c in [1, 5, 10]
        ]

        server = MagicMock()
        server.process.poll.return_value = None
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_endpoint(req, [1, 5, 10], server)

        assert result.breaking_point == 0
        assert result.breaking_reason == ""
        assert len(result.levels) == 3


# ═══════════════════════════════════════════════════════════════════════
# Result Conversion: Scenario Results
# ═══════════════════════════════════════════════════════════════════════


class TestHttpResultsToScenarioResults:
    def test_basic_conversion(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=20),
                _make_load_level(concurrency=5, median_ms=50),
            ],
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )

        scenarios = http_results_to_scenario_results(load_result)
        assert len(scenarios) == 1
        assert scenarios[0].scenario_category == "http_load_testing"
        assert len(scenarios[0].steps) == 2
        assert scenarios[0].steps[0].step_name == "concurrent_1"
        assert scenarios[0].steps[1].step_name == "concurrent_5"

    def test_crash_status(self):
        ep = _make_endpoint_result(
            levels=[_make_load_level(concurrency=1)],
            breaking_point=5,
            breaking_reason="crash",
        )
        load_result = HttpLoadResult(
            framework="flask",
            endpoint_results=[ep],
        )

        scenarios = http_results_to_scenario_results(load_result)
        assert scenarios[0].status == "failed"
        assert scenarios[0].failure_reason == "server_crash"

    def test_partial_status(self):
        ep = _make_endpoint_result(
            levels=[_make_load_level(concurrency=1)],
            breaking_point=5,
            breaking_reason="error_rate",
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )

        scenarios = http_results_to_scenario_results(load_result)
        assert scenarios[0].status == "partial"

    def test_step_measurements(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=10, median_ms=100, p95_ms=200,
                                 error_rate=0.1, memory_mb=75),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])

        scenarios = http_results_to_scenario_results(load_result)
        step = scenarios[0].steps[0]
        assert step.measurements["median_response_ms"] == 100
        assert step.measurements["p95_response_ms"] == 200
        assert step.measurements["error_rate"] == 0.1
        assert step.memory_peak_mb == 75


# ═══════════════════════════════════════════════════════════════════════
# Result Conversion: Degradation Points
# ═══════════════════════════════════════════════════════════════════════


class TestHttpResultsToDegradationPoints:
    def test_response_time_curve(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=20),
                _make_load_level(concurrency=10, median_ms=200),
            ],
            breaking_point=10,
        )
        load_result = HttpLoadResult(framework="fastapi", endpoint_results=[ep])

        points = http_results_to_degradation_points(load_result)
        assert len(points) >= 1
        time_point = [p for p in points if p.metric == "response_time_ms"][0]
        assert len(time_point.steps) == 2
        assert time_point.steps[0] == ("1 concurrent", 20)
        assert time_point.steps[1] == ("10 concurrent", 200)

    def test_memory_curve_when_available(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=50),
                _make_load_level(concurrency=10, memory_mb=150),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])

        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert mem_points[0].steps[1] == ("10 concurrent", 150)

    def test_no_curve_for_empty_levels(self):
        ep = _make_endpoint_result(levels=[])
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])

        points = http_results_to_degradation_points(load_result)
        assert len(points) == 0

    def test_breaking_point_label(self):
        ep = _make_endpoint_result(
            levels=[_make_load_level(concurrency=1, median_ms=20)],
            breaking_point=5,
        )
        load_result = HttpLoadResult(framework="fastapi", endpoint_results=[ep])

        points = http_results_to_degradation_points(load_result)
        assert points[0].breaking_point == "5 concurrent"

    def test_memory_curve_65mb_app_with_2pct_growth(self):
        """A 65MB app with ~2% growth (1.5MB) — flat, produces curve with description."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=65.0),
                _make_load_level(concurrency=50, memory_mb=66.5),
            ],
        )
        load_result = HttpLoadResult(framework="streamlit", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert "stays steady" in mem_points[0].description

    def test_memory_curve_10mb_app_with_3pct_growth(self):
        """A 10MB app with 0.3MB range (3%) — flat, produces curve with description."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=10.0),
                _make_load_level(concurrency=50, memory_mb=10.3),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert "stays steady" in mem_points[0].description

    def test_flat_memory_always_emits_curve(self):
        """A 65MB app with 0.5MB range (0.8%) — previously suppressed, now emitted."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=65.0),
                _make_load_level(concurrency=50, memory_mb=65.5),
            ],
        )
        load_result = HttpLoadResult(framework="streamlit", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert "stays steady" in mem_points[0].description
        assert "65MB" in mem_points[0].description

    def test_memory_curve_large_app_flat(self):
        """A 100MB app with 2% range — flat, produces curve with description."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=100.0),
                _make_load_level(concurrency=50, memory_mb=101.5),
                _make_load_level(concurrency=100, memory_mb=102.0),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert "stays steady" in mem_points[0].description

    def test_growing_memory_curve_no_flat_description(self):
        """Significant memory growth (>15%) gets no flat-curve description."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=50.0),
                _make_load_level(concurrency=10, memory_mb=150.0),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert mem_points[0].description == ""

    def test_flat_memory_description_matches_r_app(self):
        """R's app scenario: 65MB → 70MB across 1–1500 concurrent."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, memory_mb=65.0),
                _make_load_level(concurrency=500, memory_mb=67.0),
                _make_load_level(concurrency=1500, memory_mb=70.0),
            ],
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        mem_points = [p for p in points if p.metric == "memory_peak_mb"]
        assert len(mem_points) == 1
        assert "stays steady" in mem_points[0].description
        assert "67MB" in mem_points[0].description  # avg of 65, 67, 70
        assert "baseline memory per process" in mem_points[0].description

    def test_response_time_no_memory_note_when_curve_present(self):
        """Response time description should not contain memory note when
        memory curve is emitted (the memory curve speaks for itself)."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=20.0, memory_mb=65.0),
                _make_load_level(concurrency=50, median_ms=200.0, memory_mb=65.5),
            ],
            breaking_point=10,
        )
        load_result = HttpLoadResult(framework="flask", endpoint_results=[ep])
        points = http_results_to_degradation_points(load_result)
        time_points = [p for p in points if p.metric == "response_time_ms"]
        assert len(time_points) == 1
        assert "Memory usage stable" not in (time_points[0].description or "")


# ═══════════════════════════════════════════════════════════════════════
# Finding Generation
# ═══════════════════════════════════════════════════════════════════════


class TestHttpResultsToFindings:
    def test_server_crash_critical(self):
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[],
            server_crash=True,
            server_crash_concurrency=25,
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert "crashed" in findings[0].title.lower()
        assert "25" in findings[0].description

    def test_crash_below_user_scale(self):
        c = OperationalConstraints(user_scale=100)
        load_result = HttpLoadResult(
            framework="flask",
            server_crash=True,
            server_crash_concurrency=25,
        )
        findings = http_results_to_findings(load_result, constraints=c)
        assert "well below" in findings[0].description

    def test_endpoint_error_rate_critical(self):
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[
                _make_load_level(concurrency=1, error_rate=0),
                _make_load_level(concurrency=25, error_rate=0.6),
            ],
            breaking_point=25,
            breaking_reason="error_rate",
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert "/api/data" in findings[0].description

    def test_endpoint_below_user_scale_critical(self):
        c = OperationalConstraints(user_scale=500)
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[_make_load_level(concurrency=25, error_rate=0.6)],
            breaking_point=25,
            breaking_reason="error_rate",
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result, constraints=c)
        assert findings[0].severity == "critical"
        assert "500" in findings[0].description

    def test_endpoint_above_user_scale_warning(self):
        c = OperationalConstraints(user_scale=10)
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[_make_load_level(concurrency=15, error_rate=0.6)],
            breaking_point=15,
            breaking_reason="error_rate",
        )
        load_result = HttpLoadResult(
            framework="flask",
            endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result, constraints=c)
        assert findings[0].severity == "warning"

    def test_degradation_warning(self):
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[
                _make_load_level(concurrency=1, median_ms=10),
                _make_load_level(concurrency=50, median_ms=100),
            ],
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        # 10x increase → warning
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "10x" in findings[0].description

    def test_clean_pass_no_finding(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=10),
                _make_load_level(concurrency=50, median_ms=20),
            ],
        )
        load_result = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 0

    def test_baseline_500_finding(self):
        load_result = HttpLoadResult(framework="flask")
        probe = ProbeResult(
            endpoint=DiscoveredEndpoint(method="GET", path="/broken"),
            status_code=500,
            testable=False,
            is_finding=True,
            skip_reason="server error at baseline",
        )
        findings = http_results_to_findings(load_result, probe_results=[probe])
        assert len(findings) == 1
        assert findings[0].severity == "critical"
        assert "500" in findings[0].description

    def test_affected_dependencies(self):
        load_result = HttpLoadResult(
            framework="fastapi",
            server_crash=True,
            server_crash_concurrency=10,
        )
        findings = http_results_to_findings(load_result)
        assert "fastapi" in findings[0].affected_dependencies
        assert "uvicorn" in findings[0].affected_dependencies

    def test_streamlit_deps(self):
        load_result = HttpLoadResult(
            framework="streamlit",
            server_crash=True,
            server_crash_concurrency=5,
        )
        findings = http_results_to_findings(load_result)
        assert "streamlit" in findings[0].affected_dependencies


# ═══════════════════════════════════════════════════════════════════════
# Summary and Curve Description
# ═══════════════════════════════════════════════════════════════════════


class TestBuildHttpSummary:
    def test_crash_summary(self):
        ep = _make_endpoint_result(
            breaking_point=25, breaking_reason="crash",
            levels=[_make_load_level(concurrency=1)],
        )
        s = _build_http_summary(ep)
        assert "crashed" in s.lower()
        assert "25" in s

    def test_error_rate_summary(self):
        ep = _make_endpoint_result(
            breaking_point=50, breaking_reason="error_rate",
            levels=[
                _make_load_level(concurrency=25, error_rate=0.0),
                _make_load_level(concurrency=50, error_rate=0.6),
            ],
        )
        s = _build_http_summary(ep)
        assert "50" in s
        assert "25" in s

    def test_clean_pass_summary(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, error_rate=0, median_ms=10),
                _make_load_level(concurrency=100, error_rate=0, median_ms=50),
            ],
        )
        s = _build_http_summary(ep)
        assert "100" in s
        assert "handled" in s.lower()


class TestDescribeResponseCurve:
    def test_large_increase(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=10),
                _make_load_level(concurrency=100, median_ms=1000),
            ],
        )
        desc = _describe_response_curve(ep)
        assert "100x" in desc

    def test_moderate_increase(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=50),
                _make_load_level(concurrency=10, median_ms=150),
            ],
        )
        desc = _describe_response_curve(ep)
        assert "doubled" in desc

    def test_stable(self):
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=50),
                _make_load_level(concurrency=10, median_ms=60),
            ],
        )
        desc = _describe_response_curve(ep)
        assert desc == ""


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════


class TestRunHttpTestingPhase:
    @patch("mycode.http_load_driver.detect_framework_entry", return_value=None)
    def test_no_framework_skips(self, mock_detect):
        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        # Should return same execution unchanged
        assert result is execution
        assert len(result.scenario_results) == 0

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_framework_detected_runs_test(self, mock_detect, mock_load):
        mock_detect.return_value = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )
        ep = _make_endpoint_result(
            levels=[_make_load_level(concurrency=1, median_ms=20)],
        )
        mock_load.return_value = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].scenario_category == "http_load_testing"

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_server_fail_generates_critical_finding(self, mock_detect, mock_load):
        mock_detect.return_value = FrameworkDetection(
            framework="flask", entry_file="app.py", app_variable="app",
        )
        mock_load.return_value = HttpLoadResult(
            framework="flask",
            server_crash=True,
            endpoint_results=[],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert result.http_ran is False
        assert len(result.http_findings) == 1
        assert result.http_findings[0].severity == "critical"
        assert "could not start" in result.http_findings[0].title.lower()

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_updates_execution_counts(self, mock_detect, mock_load):
        mock_detect.return_value = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )
        ep_ok = _make_endpoint_result(
            path="/ok",
            levels=[_make_load_level(concurrency=1, median_ms=20)],
        )
        ep_fail = _make_endpoint_result(
            path="/fail",
            levels=[_make_load_level(concurrency=1, error_rate=0.6)],
            breaking_point=1,
            breaking_reason="error_rate",
        )
        mock_load.return_value = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[ep_ok, ep_fail],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult(
            scenarios_completed=3, scenarios_failed=1,
        )

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        # ep_ok → completed (+1), ep_fail → partial (+1)
        assert result.scenarios_completed == 4
        assert result.scenarios_failed == 2


# ═══════════════════════════════════════════════════════════════════════
# Memory Monitoring
# ═══════════════════════════════════════════════════════════════════════


class TestGetProcessMemory:
    def test_returns_zero_for_invalid_pid(self):
        """Memory monitoring returns 0 for non-existent process."""
        result = _get_process_memory_mb(999999999)
        assert result == 0.0

    def test_returns_float(self):
        """Memory monitoring returns a float."""
        import os
        result = _get_process_memory_mb(os.getpid())
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════
# Degradation Onset Detection
# ═══════════════════════════════════════════════════════════════════════


class TestFindDegradationOnset:
    def test_finds_doubling_point(self):
        levels = [
            _make_load_level(concurrency=1, median_ms=1.0),
            _make_load_level(concurrency=10, median_ms=1.5),
            _make_load_level(concurrency=50, median_ms=3.0),
            _make_load_level(concurrency=100, median_ms=10.0),
        ]
        assert _find_degradation_onset(levels) == 50

    def test_no_degradation_returns_zero(self):
        levels = [
            _make_load_level(concurrency=1, median_ms=10.0),
            _make_load_level(concurrency=100, median_ms=15.0),
        ]
        assert _find_degradation_onset(levels) == 0

    def test_empty_returns_zero(self):
        assert _find_degradation_onset([]) == 0


# ═══════════════════════════════════════════════════════════════════════
# Enhanced Finding Generation
# ═══════════════════════════════════════════════════════════════════════


class TestEnhancedFindings:
    def test_degradation_finding_includes_onset(self):
        """Response time >5x increase produces WARNING with onset point."""
        ep = _make_endpoint_result(
            path="/",
            levels=[
                _make_load_level(concurrency=1, median_ms=0.77),
                _make_load_level(concurrency=50, median_ms=2.0),
                _make_load_level(concurrency=1250, median_ms=8.0),
                _make_load_level(concurrency=3750, median_ms=52.82),
            ],
        )
        load_result = HttpLoadResult(
            framework="streamlit", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == "warning"
        # 52.82/0.77 ≈ 68.6, rounds to 69x
        assert "69x" in f.description
        assert "concurrent connections, response time begins" in f.description

    def test_degradation_finding_with_user_scale(self):
        """Degradation finding references user's stated capacity."""
        c = OperationalConstraints(user_scale=100)
        ep = _make_endpoint_result(
            path="/",
            levels=[
                _make_load_level(concurrency=1, median_ms=1.0, memory_mb=2.0),
                _make_load_level(concurrency=50, median_ms=3.0, memory_mb=2.0),
                _make_load_level(concurrency=100, median_ms=10.0, memory_mb=2.0),
            ],
        )
        load_result = HttpLoadResult(
            framework="streamlit", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result, constraints=c)
        assert len(findings) == 1
        assert "100 concurrent users" in findings[0].description

    def test_elevated_error_rate_warning(self):
        """Error rate >10% (but <50%) produces WARNING."""
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[
                _make_load_level(concurrency=1, error_rate=0.0, median_ms=5),
                _make_load_level(concurrency=50, error_rate=0.15, median_ms=8),
            ],
        )
        load_result = HttpLoadResult(
            framework="fastapi", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "15%" in findings[0].description

    def test_low_error_rate_no_finding(self):
        """Error rate <10% produces no finding if response time stable."""
        ep = _make_endpoint_result(
            path="/api/data",
            levels=[
                _make_load_level(concurrency=1, error_rate=0.0, median_ms=5),
                _make_load_level(concurrency=50, error_rate=0.05, median_ms=8),
            ],
        )
        load_result = HttpLoadResult(
            framework="fastapi", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) == 0


class TestServerStartupFailureFinding:
    """Verify server startup failure generates a CRITICAL finding."""

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_startup_failure_critical_finding(self, mock_detect, mock_load):
        """Server crash with no endpoints → CRITICAL 'could not start' finding."""
        mock_detect.return_value = FrameworkDetection(
            framework="flask", entry_file="app.py",
        )
        mock_load.return_value = HttpLoadResult(
            framework="flask",
            endpoint_results=[],
            server_crash=True,
            startup_error="ModuleNotFoundError: No module named 'flask'",
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        # http_ran stays False — server never ran successfully
        assert result.http_ran is False
        assert len(result.http_findings) == 1
        finding = result.http_findings[0]
        assert finding.severity == "critical"
        assert "could not start" in finding.title.lower()
        assert "flask" in finding.description.lower()
        assert "ModuleNotFoundError" in finding.description
        assert "No users can access your app" in finding.description

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_startup_failure_unknown_error(self, mock_detect, mock_load):
        """Server crash with no startup_error → uses 'unknown error'."""
        mock_detect.return_value = FrameworkDetection(
            framework="streamlit", entry_file="app.py",
        )
        mock_load.return_value = HttpLoadResult(
            framework="streamlit",
            endpoint_results=[],
            server_crash=True,
            startup_error="",
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert len(result.http_findings) == 1
        assert "unknown error" in result.http_findings[0].description

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_startup_failure_no_extra_scenarios(self, mock_detect, mock_load):
        """Server crash → no scenario results added, only finding."""
        mock_detect.return_value = FrameworkDetection(
            framework="fastapi", entry_file="main.py",
        )
        mock_load.return_value = HttpLoadResult(
            framework="fastapi",
            endpoint_results=[],
            server_crash=True,
            startup_error="SyntaxError in main.py",
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert len(result.scenario_results) == 0
        assert len(result.http_degradation_points) == 0


class TestHttpRanIntegration:
    """Verify run_http_testing_phase sets http_ran and populates findings."""

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_sets_http_ran_and_findings(self, mock_detect, mock_load):
        mock_detect.return_value = FrameworkDetection(
            framework="streamlit", entry_file="app.py",
        )
        ep = _make_endpoint_result(
            path="/",
            levels=[
                _make_load_level(concurrency=1, median_ms=1),
                _make_load_level(concurrency=100, median_ms=100),
            ],
        )
        mock_load.return_value = HttpLoadResult(
            framework="streamlit",
            endpoint_results=[ep],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert result.http_ran is True
        # 100x degradation → should produce a finding
        assert len(result.http_findings) >= 1
        assert result.http_findings[0].severity == "warning"
        # Should have degradation points too
        assert len(result.http_degradation_points) >= 1

    @patch("mycode.http_load_driver.detect_framework_entry", return_value=None)
    def test_http_ran_false_when_no_framework(self, mock_detect):
        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        assert result.http_ran is False


class TestWarmedUpBaseline:
    """Test cold-start outlier exclusion from baseline calculation."""

    def test_cold_start_excluded(self):
        """First point at 45ms, next two at 5ms — baseline should be 5ms."""
        levels = [
            _make_load_level(concurrency=1, median_ms=45.0),
            _make_load_level(concurrency=5, median_ms=5.0),
            _make_load_level(concurrency=10, median_ms=6.0),
            _make_load_level(concurrency=50, median_ms=7.0),
        ]
        assert _warmed_up_baseline(levels) == 5.0

    def test_stable_baseline_unchanged(self):
        """When first point is already the lowest, it's used as-is."""
        levels = [
            _make_load_level(concurrency=1, median_ms=5.0),
            _make_load_level(concurrency=5, median_ms=6.0),
            _make_load_level(concurrency=10, median_ms=8.0),
        ]
        assert _warmed_up_baseline(levels) == 5.0

    def test_single_level(self):
        """Single data point is used as baseline."""
        levels = [_make_load_level(concurrency=1, median_ms=45.0)]
        assert _warmed_up_baseline(levels) == 45.0

    def test_empty_levels(self):
        assert _warmed_up_baseline([]) == 0.0


class TestColdStartDegradationFinding:
    """Test that cold-start outlier doesn't mask real degradation."""

    def test_degradation_detected_with_cold_start(self):
        """45ms cold start, 5ms warmed up, 127ms at load — 25x → WARNING."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=45.0),
                _make_load_level(concurrency=5, median_ms=5.0),
                _make_load_level(concurrency=10, median_ms=6.0),
                _make_load_level(concurrency=50, median_ms=20.0),
                _make_load_level(concurrency=100, median_ms=127.0),
            ],
        )
        finding = _endpoint_to_finding(ep, ["react"], user_scale=500)
        assert finding is not None
        assert finding.severity == "warning"
        assert "5ms" in finding.description or "baseline" in finding.description.lower()

    def test_no_degradation_without_cold_start_issue(self):
        """5ms → 10ms = 2x, below 5x threshold → no finding."""
        ep = _make_endpoint_result(
            levels=[
                _make_load_level(concurrency=1, median_ms=5.0, memory_mb=5.0),
                _make_load_level(concurrency=5, median_ms=6.0, memory_mb=5.0),
                _make_load_level(concurrency=50, median_ms=10.0, memory_mb=5.0),
            ],
        )
        finding = _endpoint_to_finding(ep, ["react"], user_scale=500)
        # Either None or a memory finding, but NOT a response time finding
        if finding is not None:
            assert "response time" not in finding.title.lower()


class TestMemoryCapacityFinding:
    """Test memory capacity finding generated independently of per-endpoint findings."""

    def _make_load_result(self, memory_mb=71.0):
        """Build an HttpLoadResult with one endpoint at given baseline memory."""
        return HttpLoadResult(
            framework="react-scripts",
            endpoint_results=[
                _make_endpoint_result(
                    levels=[
                        _make_load_level(concurrency=1, median_ms=5.0, memory_mb=memory_mb),
                        _make_load_level(concurrency=5, median_ms=6.0, memory_mb=memory_mb + 2),
                    ],
                ),
            ],
        )

    def test_critical_when_below_25_percent(self):
        """72MB baseline, 500 user_scale → ~21 capacity (<25%) → CRITICAL."""
        finding = _memory_capacity_finding(
            self._make_load_result(72.0), ["react"], user_scale=500,
        )
        assert finding is not None
        assert finding.severity == "critical"
        assert "memory" in finding.title.lower()
        assert "500" in finding.description
        assert "72MB" in finding.description

    def test_warning_when_between_25_and_75_percent(self):
        """10MB baseline, 200 user_scale → ~153 capacity (76%) — but let's
        use 20MB → ~76 capacity (38%) → WARNING."""
        finding = _memory_capacity_finding(
            self._make_load_result(20.0), ["react"], user_scale=200,
        )
        assert finding is not None
        assert finding.severity == "warning"

    def test_no_finding_when_capacity_exceeds_scale(self):
        """5MB baseline, 10 user_scale → ~307 capacity → no finding."""
        finding = _memory_capacity_finding(
            self._make_load_result(5.0), ["express"], user_scale=10,
        )
        assert finding is None

    def test_no_finding_without_user_scale(self):
        """_memory_capacity_finding requires user_scale — caller gates this."""
        # The function itself takes user_scale as required int, but
        # http_results_to_findings only calls it when user_scale is set.
        # Just verify there's no crash with a scale that's met.
        finding = _memory_capacity_finding(
            self._make_load_result(1.0), ["react"], user_scale=5,
        )
        assert finding is None  # 1536 capacity >> 5 scale

    def test_memory_finding_alongside_degradation(self):
        """Memory finding should appear even when response time degradation exists."""
        constraints = OperationalConstraints(user_scale=500)
        load_result = HttpLoadResult(
            framework="react-scripts",
            endpoint_results=[
                _make_endpoint_result(
                    levels=[
                        _make_load_level(concurrency=1, median_ms=45.0, memory_mb=72.0),
                        _make_load_level(concurrency=5, median_ms=5.0, memory_mb=72.0),
                        _make_load_level(concurrency=10, median_ms=6.0, memory_mb=73.0),
                        _make_load_level(concurrency=50, median_ms=20.0, memory_mb=75.0),
                        _make_load_level(concurrency=100, median_ms=127.0, memory_mb=80.0),
                    ],
                ),
            ],
        )
        findings = http_results_to_findings(load_result, constraints)
        titles = [f.title.lower() for f in findings]
        # Both response time AND memory findings should be present
        assert any("response time" in t for t in titles), f"Missing response time finding: {titles}"
        assert any("memory" in t for t in titles), f"Missing memory finding: {titles}"

    def test_no_memory_data_no_finding(self):
        """When memory measurements are all 0, no memory finding."""
        finding = _memory_capacity_finding(
            self._make_load_result(0.0), ["react"], user_scale=500,
        )
        assert finding is None


class TestMultiRoundMedian:
    """Test that drive_load_level runs multiple rounds and picks median."""

    def test_default_rounds(self):
        """Default rounds constant should be 3."""
        assert _ROUNDS_PER_LEVEL == 3

    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=50.0)
    @patch("mycode.http_load_driver._send_request")
    def test_median_of_three_rounds(self, mock_send, mock_mem):
        """With 3 rounds at varying speeds, result should be the median round."""
        # Round 1: 10ms, Round 2: 100ms, Round 3: 20ms → median = 20ms
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            # Each round gets 1 request (concurrency=1)
            # Rounds cycle: 10, 100, 20
            round_idx = (call_count[0] - 1) // 1  # concurrency=1
            times = [10.0, 100.0, 20.0]
            return (times[round_idx % 3], 200, "")

        mock_send.side_effect = side_effect
        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=1, server_pid=123, rounds=3)

        # Median of [10, 100, 20] sorted = [10, 20, 100] → median idx 1 = 20
        assert result.median_response_ms == 20.0

    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=50.0)
    @patch("mycode.http_load_driver._send_request")
    def test_high_variance_flagged(self, mock_send, mock_mem):
        """When rounds vary >50%, high_variance should be True."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            round_idx = (call_count[0] - 1) // 1
            # 10ms vs 100ms = 10x variance, well above 1.5x threshold
            times = [10.0, 100.0, 50.0]
            return (times[round_idx % 3], 200, "")

        mock_send.side_effect = side_effect
        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=1, server_pid=123, rounds=3)
        assert result.high_variance is True

    @patch("mycode.http_load_driver._get_process_memory_mb", return_value=50.0)
    @patch("mycode.http_load_driver._send_request")
    def test_low_variance_not_flagged(self, mock_send, mock_mem):
        """When rounds are consistent, high_variance should be False."""
        mock_send.return_value = (25.0, 200, "")
        req = HttpTestRequest(method="GET", url="http://localhost:8000/")
        result = drive_load_level(req, concurrency=1, server_pid=123, rounds=3)
        assert result.high_variance is False


class TestVarianceNote:
    """Test that high variance appends a note to findings."""

    def test_variance_note_appended(self):
        """When a level has high_variance, findings get the variance note."""
        ep = _make_endpoint_result(
            path="/",
            levels=[
                _make_load_level(concurrency=1, median_ms=5.0, memory_mb=2.0),
                _make_load_level(concurrency=50, median_ms=50.0, memory_mb=2.0),
            ],
        )
        # Mark one level as high variance
        ep.levels[1].high_variance = True
        load_result = HttpLoadResult(
            framework="react-scripts", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        assert len(findings) >= 1
        assert "variance" in findings[0].description.lower()

    def test_no_variance_note_when_stable(self):
        """When no level has high_variance, no note appended."""
        ep = _make_endpoint_result(
            path="/",
            levels=[
                _make_load_level(concurrency=1, median_ms=5.0, memory_mb=2.0),
                _make_load_level(concurrency=50, median_ms=50.0, memory_mb=2.0),
            ],
        )
        load_result = HttpLoadResult(
            framework="react-scripts", endpoint_results=[ep],
        )
        findings = http_results_to_findings(load_result)
        for f in findings:
            assert "variance" not in f.description.lower()


# ── Clean Endpoint INFO Finding ──


class TestCleanEndpointFinding:
    """Tests for INFO finding when HTTP testing produces no degradation."""

    def test_clean_endpoint_produces_info_finding(self):
        """run_http_testing_phase generates INFO finding when endpoint passes all levels."""
        ep = EndpointLoadResult(
            endpoint=HttpTestRequest(
                method="GET", url="http://localhost:8501/",
                description="GET / — Main Page",
            ),
            levels=[
                _make_load_level(concurrency=1, median_ms=50.0, memory_mb=2.0),
                _make_load_level(concurrency=10, median_ms=80.0, memory_mb=2.0),
                _make_load_level(concurrency=25, median_ms=100.0, memory_mb=2.0),
            ],
        )
        load_result = HttpLoadResult(
            framework="streamlit",
            endpoint_results=[ep],
            server_startup_time=2.0,
        )

        # No findings from threshold checks
        findings = http_results_to_findings(load_result)
        assert len(findings) == 0

        # But run_http_testing_phase should add INFO finding
        execution = ExecutionEngineResult()
        with patch("mycode.http_load_driver.detect_framework_entry") as mock_detect, \
             patch("mycode.http_load_driver.run_http_load_test") as mock_load:
            from mycode.server_manager import FrameworkDetection
            mock_detect.return_value = FrameworkDetection(
                framework="streamlit", entry_file="app.py",
            )
            mock_load.return_value = load_result

            from mycode.http_load_driver import run_http_testing_phase
            result = run_http_testing_phase(
                session=MagicMock(project_copy_dir=Path("/tmp")),
                ingestion=MagicMock(project_path="/tmp"),
                execution=execution,
                language="python",
            )

        assert result.http_ran is True
        assert len(result.http_findings) == 1
        assert result.http_findings[0].severity == "info"
        assert "without issues" in result.http_findings[0].title.lower()


# ── HTTP Timeout Budget ──


class TestHttpTimeoutBudget:
    """Tests for time budget in drive_endpoint and run_http_load_test."""

    def test_drive_endpoint_respects_deadline(self):
        """drive_endpoint stops testing when deadline is passed."""
        import time

        req = HttpTestRequest(
            method="GET", url="http://localhost:8000/",
            description="GET /",
        )
        server = MagicMock()
        server.process.poll.return_value = None  # still alive
        server.process.pid = 12345

        # Set deadline in the past so it triggers immediately
        deadline = time.monotonic() - 1

        with patch("mycode.http_load_driver.drive_load_level") as mock_drive:
            result = drive_endpoint(
                req, [1, 5, 10, 25], server, deadline=deadline,
            )
            # Should not have called drive_load_level at all
            mock_drive.assert_not_called()
            assert len(result.levels) == 0

    def test_run_http_load_test_passes_timeout(self):
        """run_http_load_test forwards timeout to drive_endpoint via deadline."""
        from mycode.http_load_driver import run_http_load_test

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        detection = MagicMock(framework="streamlit", entry_file="app.py")
        ingestion = MagicMock(project_path="/tmp/proj")

        with patch("mycode.http_load_driver.start_server") as mock_start, \
             patch("mycode.http_load_driver.discover_endpoints") as mock_disc, \
             patch("mycode.http_load_driver.generate_requests") as mock_gen, \
             patch("mycode.http_load_driver.probe_endpoints") as mock_probe, \
             patch("mycode.http_load_driver.drive_endpoint") as mock_drive, \
             patch("mycode.http_load_driver.stop_server"):

            mock_server = MagicMock()
            mock_server.port = 8501
            mock_server.startup_time = 1.0
            mock_server.process.poll.return_value = None
            mock_start.return_value = MagicMock(
                success=True, server=mock_server,
            )
            mock_disc.return_value = MagicMock()
            mock_gen.return_value = [
                HttpTestRequest(method="GET", url="http://localhost:8501/", description="GET /"),
            ]
            mock_probe.return_value = [MagicMock(testable=True, is_finding=False)]
            mock_drive.return_value = EndpointLoadResult(
                endpoint=mock_gen.return_value[0],
            )

            run_http_load_test(
                session=session,
                detection=detection,
                ingestion=ingestion,
                timeout=90,
            )

            # drive_endpoint should have been called with a deadline kwarg
            assert mock_drive.called
            call_kwargs = mock_drive.call_args
            assert call_kwargs.kwargs.get("deadline") is not None


class TestDefaultHttpBudget:
    """Tests for default HTTP budget and per-endpoint budget."""

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_default_budget_when_no_constraints(self, mock_detect, mock_load):
        """run_http_testing_phase applies 120s default budget when constraints is None."""
        mock_detect.return_value = FrameworkDetection(
            framework="flask", entry_file="app.py",
        )
        mock_load.return_value = HttpLoadResult(framework="flask")

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = None
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        run_http_testing_phase(session, ingestion, execution, "python")

        assert mock_load.called
        call_kwargs = mock_load.call_args
        assert call_kwargs.kwargs.get("timeout") == 120

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_constraint_budget_overrides_default(self, mock_detect, mock_load):
        """Explicit timeout_per_scenario overrides the 120s default."""
        mock_detect.return_value = FrameworkDetection(
            framework="flask", entry_file="app.py",
        )
        mock_load.return_value = HttpLoadResult(framework="flask")

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = None
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()
        constraints = OperationalConstraints(timeout_per_scenario=300)

        run_http_testing_phase(
            session, ingestion, execution, "python", constraints=constraints,
        )

        assert mock_load.called
        call_kwargs = mock_load.call_args
        assert call_kwargs.kwargs.get("timeout") == 300


class TestPerEndpointBudget:
    """Tests for per-endpoint budget distribution."""

    def test_per_endpoint_deadline_calculated(self):
        """Each endpoint gets a fair share of remaining budget, recalculated per iteration."""
        import time
        from mycode.http_load_driver import run_http_load_test

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        detection = MagicMock(framework="flask", entry_file="app.py")
        ingestion = MagicMock(project_path="/tmp/proj")

        with patch("mycode.http_load_driver.start_server") as mock_start, \
             patch("mycode.http_load_driver.discover_endpoints"), \
             patch("mycode.http_load_driver.generate_requests") as mock_gen, \
             patch("mycode.http_load_driver.probe_endpoints") as mock_probe, \
             patch("mycode.http_load_driver.drive_endpoint") as mock_drive, \
             patch("mycode.http_load_driver.stop_server"):

            mock_server = MagicMock()
            mock_server.port = 8000
            mock_server.startup_time = 1.0
            mock_server.process.poll.return_value = None
            mock_start.return_value = MagicMock(
                success=True, server=mock_server,
            )
            reqs = [
                HttpTestRequest(method="GET", url=f"http://localhost:8000/ep{i}", description=f"GET /ep{i}")
                for i in range(4)
            ]
            mock_gen.return_value = reqs
            mock_probe.return_value = [MagicMock(testable=True, is_finding=False)] * 4
            mock_drive.return_value = EndpointLoadResult(endpoint=reqs[0])

            run_http_load_test(
                session=session, detection=detection,
                ingestion=ingestion, timeout=60,
            )

            # Should have been called 4 times with per-endpoint deadlines
            assert mock_drive.call_count == 4
            # Each call should have a deadline kwarg
            for call in mock_drive.call_args_list:
                assert call.kwargs.get("deadline") is not None


class TestEarlyAbortSlowBaseline:
    """Tests for early abort of slow endpoints at concurrency=1."""

    @patch("mycode.http_load_driver.drive_load_level")
    def test_slow_baseline_aborts(self, mock_drive):
        """Endpoint taking >10s at concurrency=1 is skipped."""
        mock_drive.return_value = _make_load_level(
            concurrency=1, median_ms=12000,
        )

        server = MagicMock()
        server.process.poll.return_value = None
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/checkout")
        result = drive_endpoint(req, [1, 5, 10, 25], server)

        assert result.breaking_point == 1
        assert result.breaking_reason == "external_dependency_timeout"
        assert len(result.levels) == 1
        # Should NOT have tested higher concurrency levels
        assert mock_drive.call_count == 1

    @patch("mycode.http_load_driver.drive_load_level")
    def test_fast_baseline_not_aborted(self, mock_drive):
        """Endpoint responding in 50ms at concurrency=1 continues normally."""
        mock_drive.side_effect = [
            _make_load_level(concurrency=c, median_ms=50)
            for c in [1, 5, 10]
        ]

        server = MagicMock()
        server.process.poll.return_value = None
        server.process.pid = 123

        req = HttpTestRequest(method="GET", url="http://localhost:8000/api/test")
        result = drive_endpoint(req, [1, 5, 10], server)

        assert result.breaking_reason == ""
        assert len(result.levels) == 3

    def test_external_dep_timeout_generates_info_finding(self):
        """Skipped endpoint produces an INFO finding with explanatory description."""
        ep = EndpointLoadResult(
            endpoint=HttpTestRequest(
                method="GET", url="http://localhost:8000/checkout",
            ),
            levels=[_make_load_level(concurrency=1, median_ms=12000)],
            breaking_point=1,
            breaking_reason="external_dependency_timeout",
        )
        finding = _endpoint_to_finding(ep, ["flask"], user_scale=None)
        assert finding is not None
        assert finding.severity == "info"
        assert "10 seconds" in finding.description
        assert "external service" in finding.description
        assert "/checkout" in finding.title


# ── npm Install Failure Blocks HTTP Testing ──


class TestNpmInstallFailureBlocksHttp:
    """Verify HTTP testing is skipped with CRITICAL finding when npm install failed."""

    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_npm_failure_generates_critical_finding(self, mock_detect):
        """js_deps_installed=False → CRITICAL finding, no server start attempted."""
        mock_detect.return_value = FrameworkDetection(
            framework="nextjs", entry_file="next.config.js",
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = False
        session.js_deps_error = "ERR! missing peer dependency"
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "javascript"
        )
        assert len(result.http_findings) == 1
        finding = result.http_findings[0]
        assert finding.severity == "critical"
        assert "installation failed" in finding.title.lower()
        assert "ERR! missing peer dependency" in finding.description
        assert "nextjs" in finding.description

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_npm_success_proceeds_normally(self, mock_detect, mock_load):
        """js_deps_installed=True → server start attempted normally."""
        mock_detect.return_value = FrameworkDetection(
            framework="nextjs", entry_file="next.config.js",
        )
        ep = _make_endpoint_result(
            levels=[_make_load_level(concurrency=1, median_ms=20)],
        )
        mock_load.return_value = HttpLoadResult(
            framework="nextjs", endpoint_results=[ep],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = True
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "javascript"
        )
        mock_load.assert_called_once()
        assert result.http_ran is True

    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_npm_timeout_generates_finding(self, mock_detect):
        """npm timeout → CRITICAL finding with timeout message."""
        mock_detect.return_value = FrameworkDetection(
            framework="express", entry_file="server.js",
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = False
        session.js_deps_error = "npm install timed out after 120 seconds"
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "javascript"
        )
        assert len(result.http_findings) == 1
        assert "timed out" in result.http_findings[0].description

    @patch("mycode.http_load_driver.run_http_load_test")
    @patch("mycode.http_load_driver.detect_framework_entry")
    def test_python_project_unaffected(self, mock_detect, mock_load):
        """Python project (js_deps_installed=None) → proceeds normally."""
        mock_detect.return_value = FrameworkDetection(
            framework="flask", entry_file="app.py", app_variable="app",
        )
        mock_load.return_value = HttpLoadResult(
            framework="flask",
            server_crash=True,
            endpoint_results=[],
        )

        session = MagicMock()
        session.project_copy_dir = Path("/tmp/proj")
        session.js_deps_installed = None  # not a JS project
        ingestion = IngestionResult(project_path="/tmp/proj")
        execution = ExecutionEngineResult()

        result = run_http_testing_phase(
            session, ingestion, execution, "python"
        )
        # Should have proceeded to run_http_load_test (not blocked)
        mock_load.assert_called_once()
