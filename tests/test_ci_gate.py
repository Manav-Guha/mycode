"""Tests for the CI gate API: key store, check/result/override endpoints."""

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from mycode.web.analytics import close_connection


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary SQLite database for every test."""
    db_path = str(tmp_path / "test_ci.db")
    with patch.dict(os.environ, {"MYCODE_DB_PATH": db_path}):
        close_connection()
        yield db_path
        close_connection()


# ═══════════════════════════════════════════════════════════
# Key Store Tests
# ═══════════════════════════════════════════════════════════


class TestCIKeyStore:

    def test_create_api_key_format(self):
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        assert key.startswith("mck_")
        assert len(key) == 36  # "mck_" + 32 hex chars

    def test_validate_key_valid(self):
        from mycode.web.ci_keys import create_api_key, validate_api_key
        key = create_api_key()
        assert validate_api_key(key) is True

    def test_validate_key_invalid(self):
        from mycode.web.ci_keys import validate_api_key
        assert validate_api_key("mck_not_a_real_key_at_all_1234") is False
        assert validate_api_key("") is False
        assert validate_api_key("bad_prefix_abc") is False

    def test_key_usage_tracking(self):
        from mycode.web.analytics import get_db
        from mycode.web.ci_keys import create_api_key, record_key_usage, _hash_key
        key = create_api_key()
        record_key_usage(key)
        record_key_usage(key)
        db = get_db()
        row = db.execute(
            "SELECT total_runs, last_used FROM ci_api_keys WHERE key_hash = ?",
            (_hash_key(key),),
        ).fetchone()
        assert row[0] == 2
        assert row[1] is not None

    def test_list_keys_no_hash_exposed(self):
        from mycode.web.ci_keys import create_api_key, list_api_keys
        create_api_key()
        keys = list_api_keys()
        assert len(keys) == 1
        assert "key_hash" not in keys[0]
        assert keys[0]["key_prefix"].startswith("mck_")
        assert len(keys[0]["key_prefix"]) == 8

    def test_multiple_keys(self):
        from mycode.web.ci_keys import create_api_key, list_api_keys
        k1 = create_api_key()
        k2 = create_api_key()
        assert k1 != k2
        keys = list_api_keys()
        assert len(keys) == 2


# ═══════════════════════════════════════════════════════════
# Override Tests
# ═══════════════════════════════════════════════════════════


class TestCIOverrides:

    def test_record_and_get_override(self):
        from mycode.web.ci_keys import (
            create_api_key, record_override, get_override, _hash_key,
        )
        from mycode.web.analytics import get_db
        key = create_api_key()
        record_override("j_test123", key, "false positive")
        ov = get_override("j_test123")
        assert ov is not None
        assert ov["job_id"] == "j_test123"
        assert ov["reason"] == "false positive"
        assert ov["key_prefix"].startswith("mck_")
        # Check total_overrides incremented
        db = get_db()
        row = db.execute(
            "SELECT total_overrides FROM ci_api_keys WHERE key_hash = ?",
            (_hash_key(key),),
        ).fetchone()
        assert row[0] == 1

    def test_get_override_not_found(self):
        from mycode.web.ci_keys import get_override
        assert get_override("j_nonexistent") is None

    def test_list_overrides(self):
        from mycode.web.ci_keys import create_api_key, record_override, list_overrides
        key = create_api_key()
        record_override("j_a", key, "reason a")
        record_override("j_b", key, "reason b")
        overrides = list_overrides()
        assert len(overrides) == 2
        job_ids = {o["job_id"] for o in overrides}
        assert job_ids == {"j_a", "j_b"}


# ═══════════════════════════════════════════════════════════
# CI Check Endpoint Tests
# ═══════════════════════════════════════════════════════════


# Helpers for mocking preflight/analyze results

@dataclass
class _MockPreflightResponse:
    job_id: str = ""
    error: str = ""


@dataclass
class _MockAnalyzeResponse:
    job_id: str = ""
    status: str = ""
    message: str = ""
    error: str = ""


def _patch_ci_check():
    """Return a context manager that patches preflight + analyze for ci.py."""
    return patch.multiple(
        "mycode.web.ci",
        handle_preflight=MagicMock(
            return_value=_MockPreflightResponse(job_id="j_test001"),
        ),
        handle_submit_intent=MagicMock(return_value={"done": True}),
        handle_analyze=MagicMock(
            return_value=_MockAnalyzeResponse(job_id="j_test001", status="running"),
        ),
    )


class TestCICheck:

    def test_missing_api_key(self):
        from mycode.web.ci import handle_ci_check
        result = handle_ci_check("https://github.com/u/r", "report_only", 2, "")
        assert result["_status"] == 403

    def test_invalid_api_key(self):
        from mycode.web.ci import handle_ci_check
        result = handle_ci_check(
            "https://github.com/u/r", "report_only", 2, "mck_bad",
        )
        assert result["_status"] == 403

    def test_invalid_threshold(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        result = handle_ci_check(
            "https://github.com/u/r", "nope", 2, key,
        )
        assert result["_status"] == 400
        assert "threshold" in result["error"].lower()

    def test_invalid_tier(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        result = handle_ci_check(
            "https://github.com/u/r", "critical", 99, key,
        )
        assert result["_status"] == 400
        assert "tier" in result["error"].lower()

    def test_missing_repo_url(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        result = handle_ci_check("", "critical", 2, key)
        assert result["_status"] == 400
        assert "repo_url" in result["error"]

    def test_success(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        from mycode.web.jobs import store
        key = create_api_key()

        # Create a mock job in the store for submit-intent and analyze to find
        mock_job = store.create()
        mock_job.status = "preflight_complete"
        mock_job_id = mock_job.id

        with patch.multiple(
            "mycode.web.ci",
            handle_preflight=MagicMock(
                return_value=_MockPreflightResponse(job_id=mock_job_id),
            ),
            handle_submit_intent=MagicMock(return_value={"done": True}),
            handle_analyze=MagicMock(
                return_value=_MockAnalyzeResponse(job_id=mock_job_id, status="running"),
            ),
        ):
            result = handle_ci_check(
                "https://github.com/u/r", "critical", 2, key,
            )
        assert result["job_id"] == mock_job_id
        assert result["status"] == "queued"
        # CI threshold stored on the job
        assert mock_job.ci_threshold == "critical"

    def test_preflight_failure(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        with patch(
            "mycode.web.ci.handle_preflight",
            return_value=_MockPreflightResponse(error="Clone failed"),
        ):
            result = handle_ci_check(
                "https://github.com/u/r", "report_only", 1, key,
            )
        assert result["_status"] == 400
        assert "Clone failed" in result["error"]

    def test_source_tagged_ci(self):
        from mycode.web.ci import handle_ci_check
        from mycode.web.ci_keys import create_api_key
        from mycode.web.jobs import store
        key = create_api_key()
        mock_job = store.create()
        mock_job.status = "preflight_complete"

        mock_preflight = MagicMock(
            return_value=_MockPreflightResponse(job_id=mock_job.id),
        )
        with patch.multiple(
            "mycode.web.ci",
            handle_preflight=mock_preflight,
            handle_submit_intent=MagicMock(return_value={"done": True}),
            handle_analyze=MagicMock(
                return_value=_MockAnalyzeResponse(job_id=mock_job.id, status="running"),
            ),
        ):
            handle_ci_check("https://github.com/u/r", "report_only", 2, key)
            # Verify preflight was called with source="ci"
            mock_preflight.assert_called_once()
            call_kwargs = mock_preflight.call_args
            assert call_kwargs[1].get("source") == "ci" or call_kwargs[0][-1] == "ci"


# ═══════════════════════════════════════════════════════════
# CI Result Endpoint Tests
# ═══════════════════════════════════════════════════════════


@dataclass
class _MockFinding:
    severity: str = "info"
    title: str = "test finding"


@dataclass
class _MockReport:
    scenarios_run: int = 100
    findings: list = field(default_factory=list)


@dataclass
class _MockPipelineResult:
    report: _MockReport = field(default_factory=_MockReport)


def _make_completed_job(threshold="report_only", findings=None):
    """Create a completed job in the store with mock findings."""
    from mycode.web.jobs import store
    job = store.create()
    job.status = "completed"
    job.ci_threshold = threshold
    report = _MockReport(
        scenarios_run=100,
        findings=findings or [],
    )
    job.result = _MockPipelineResult(report=report)
    return job


class TestCIResult:

    def test_running(self):
        from mycode.web.ci import handle_ci_result
        from mycode.web.jobs import store
        job = store.create()
        job.status = "running"
        result = handle_ci_result(job.id)
        assert result["status"] == "running"

    def test_not_found(self):
        from mycode.web.ci import handle_ci_result
        result = handle_ci_result("j_nonexistent")
        assert result["_status"] == 404

    def test_job_failed(self):
        from mycode.web.ci import handle_ci_result
        from mycode.web.jobs import store
        job = store.create()
        job.status = "failed"
        job.error = "Out of memory"
        result = handle_ci_result(job.id)
        assert result["status"] == "error"
        assert "Out of memory" in result["error"]

    def test_report_only_always_passes(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="report_only",
            findings=[
                _MockFinding(severity="critical"),
                _MockFinding(severity="warning"),
            ],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "report_only"
        assert result["findings_count"]["critical"] == 1
        assert result["override"] is False

    def test_critical_pass(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="critical",
            findings=[_MockFinding(severity="warning")],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "pass"

    def test_critical_fail(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="critical",
            findings=[_MockFinding(severity="critical")],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "fail"
        assert result["findings_count"]["critical"] == 1

    def test_warning_fail(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="warning",
            findings=[_MockFinding(severity="warning")],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "fail"

    def test_warning_pass_info_only(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="warning",
            findings=[_MockFinding(severity="info")],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "pass"

    def test_any_fail_on_info(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="any",
            findings=[_MockFinding(severity="info")],
        )
        result = handle_ci_result(job.id)
        assert result["status"] == "fail"

    def test_any_pass_no_findings(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(threshold="any", findings=[])
        result = handle_ci_result(job.id)
        assert result["status"] == "pass"

    def test_with_override(self):
        from mycode.web.ci import handle_ci_result
        from mycode.web.ci_keys import create_api_key, record_override
        key = create_api_key()
        job = _make_completed_job(
            threshold="critical",
            findings=[_MockFinding(severity="critical")],
        )
        record_override(job.id, key, "false positive")
        result = handle_ci_result(job.id)
        assert result["status"] == "pass"
        assert result["override"] is True

    def test_summary_format(self):
        from mycode.web.ci import handle_ci_result
        job = _make_completed_job(
            threshold="report_only",
            findings=[
                _MockFinding(severity="critical"),
                _MockFinding(severity="critical"),
                _MockFinding(severity="warning"),
                _MockFinding(severity="info"),
                _MockFinding(severity="info"),
                _MockFinding(severity="info"),
            ],
        )
        result = handle_ci_result(job.id)
        assert "2 critical" in result["summary"]
        assert "1 warning" in result["summary"]
        assert "3 info" in result["summary"]
        assert "100 scenarios" in result["summary"]
        assert result["report_url"] == f"/api/report/{job.id}"


# ═══════════════════════════════════════════════════════════
# CI Override Endpoint Tests
# ═══════════════════════════════════════════════════════════


class TestCIOverrideEndpoint:

    def test_override_success(self):
        from mycode.web.ci import handle_ci_override
        from mycode.web.ci_keys import create_api_key, get_override
        key = create_api_key()
        job = _make_completed_job(threshold="critical")
        result = handle_ci_override(job.id, key, "known flaky dep")
        assert result["ok"] is True
        assert get_override(job.id) is not None

    def test_override_invalid_key(self):
        from mycode.web.ci import handle_ci_override
        job = _make_completed_job()
        result = handle_ci_override(job.id, "mck_bad", "reason")
        assert result["_status"] == 403

    def test_override_job_not_completed(self):
        from mycode.web.ci import handle_ci_override
        from mycode.web.ci_keys import create_api_key
        from mycode.web.jobs import store
        key = create_api_key()
        job = store.create()
        job.status = "running"
        result = handle_ci_override(job.id, key, "reason")
        assert result["_status"] == 400

    def test_override_job_not_found(self):
        from mycode.web.ci import handle_ci_override
        from mycode.web.ci_keys import create_api_key
        key = create_api_key()
        result = handle_ci_override("j_nope", key, "reason")
        assert result["_status"] == 404


# ═══════════════════════════════════════════════════════════
# Admin Endpoint Tests
# ═══════════════════════════════════════════════════════════


class TestCIAdminHandlers:

    def test_keys_create(self):
        from mycode.web.ci import handle_ci_keys_create
        result = handle_ci_keys_create()
        assert result["key"].startswith("mck_")

    def test_keys_list(self):
        from mycode.web.ci import handle_ci_keys_create, handle_ci_keys_list
        handle_ci_keys_create()
        handle_ci_keys_create()
        result = handle_ci_keys_list()
        assert len(result["keys"]) == 2

    def test_overrides_list(self):
        from mycode.web.ci import handle_admin_overrides
        from mycode.web.ci_keys import create_api_key, record_override
        key = create_api_key()
        record_override("j_x", key, "r1")
        result = handle_admin_overrides()
        assert len(result["overrides"]) == 1
        assert result["overrides"][0]["job_id"] == "j_x"


# ═══════════════════════════════════════════════════════════
# Source Validation
# ═══════════════════════════════════════════════════════════


class TestCISourceValidation:

    def test_ci_is_valid_source(self):
        from mycode.web.analytics import validate_source
        assert validate_source("ci") == "ci"
