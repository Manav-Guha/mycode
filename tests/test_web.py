"""Tests for myCode web interface backend.

Tests cover:
  - Job store lifecycle (create, get, cleanup, expiry)
  - Project fetch validation (GitHub URL, zip extraction)
  - Preflight handler (mocked session/ingestion)
  - Conversation handler (turn-based flow)
  - Analyze handler (job state checks)
  - Status and report handlers
  - Health endpoint
  - Schema dataclasses
  - Interface turn-based API (prepare_turn_1, process_turn_1, process_turn_2)
"""

import io
import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mycode.ingester import (
    DependencyInfo,
    FileAnalysis,
    FunctionInfo,
    IngestionResult,
)
from mycode.interface import (
    ConversationalInterface,
    InterfaceResult,
    OperationalIntent,
)
from mycode.constraints import OperationalConstraints
from mycode.viability import ViabilityResult


# ── Fixtures ──


@pytest.fixture
def simple_ingestion():
    """Minimal ingestion result for testing."""
    return IngestionResult(
        project_path="/tmp/test-project",
        files_analyzed=5,
        files_failed=0,
        total_lines=300,
        file_analyses=[
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(
                        name="index",
                        file_path="app.py",
                        lineno=10,
                        decorators=["app.route"],
                        is_method=False,
                    ),
                ],
                classes=[],
                lines_of_code=100,
            ),
        ],
        dependencies=[
            DependencyInfo(name="flask", installed_version="3.0.0"),
            DependencyInfo(name="sqlalchemy", installed_version="2.0.0"),
            DependencyInfo(name="requests", installed_version="2.31.0"),
        ],
    )


@pytest.fixture
def viable_result():
    """Passing viability result."""
    return ViabilityResult(
        viable=True,
        install_rate=0.9,
        import_rate=0.85,
        syntax_rate=1.0,
    )


@pytest.fixture
def failing_viability():
    """Failing viability result."""
    return ViabilityResult(
        viable=False,
        install_rate=0.3,
        import_rate=0.1,
        syntax_rate=1.0,
        reason="Only 30% of dependencies installed (need 50%)",
        suggest_docker=True,
    )


# ── Job Store Tests ──


class TestJobStore:
    """Tests for the in-memory job store."""

    def test_create_job(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        job = s.create()
        assert job.id.startswith("j_")
        assert job.status == "preflight_running"
        assert job.created_at > 0

    def test_get_job(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        job = s.create()
        retrieved = s.get(job.id)
        assert retrieved is job

    def test_get_nonexistent(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        assert s.get("j_nonexistent") is None

    def test_active_count(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        j1 = s.create()
        j2 = s.create()
        j1.status = "running"
        j2.status = "completed"
        assert s.active_count() == 1

    def test_active_count_includes_preflight(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        j1 = s.create()  # status = preflight_running
        assert s.active_count() == 1

    def test_cleanup_expired(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        job = s.create()
        job.created_at = time.time() - 99999  # way past TTL
        removed = s.cleanup_expired()
        assert removed == 1
        assert s.get(job.id) is None

    def test_cleanup_keeps_recent(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        job = s.create()
        removed = s.cleanup_expired()
        assert removed == 0
        assert s.get(job.id) is not None

    def test_remove_job(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        job = s.create()
        s.remove(job.id)
        assert s.get(job.id) is None

    def test_remove_nonexistent(self):
        from mycode.web.jobs import JobStore
        s = JobStore()
        s.remove("j_nonexistent")  # Should not raise


# ── Project Fetch Tests ──


class TestProjectFetch:
    """Tests for GitHub URL validation and zip extraction."""

    def test_valid_github_url(self):
        from mycode.web.project_fetch import validate_github_url
        url = validate_github_url("https://github.com/user/repo")
        assert url == "https://github.com/user/repo"

    def test_github_url_strips_git_suffix(self):
        from mycode.web.project_fetch import validate_github_url
        url = validate_github_url("https://github.com/user/repo.git")
        assert url == "https://github.com/user/repo"

    def test_github_url_strips_tree_path(self):
        from mycode.web.project_fetch import validate_github_url
        url = validate_github_url("https://github.com/user/repo/tree/main/src")
        assert url == "https://github.com/user/repo"

    def test_github_url_strips_trailing_slash(self):
        from mycode.web.project_fetch import validate_github_url
        url = validate_github_url("https://github.com/user/repo/")
        assert url == "https://github.com/user/repo"

    def test_invalid_github_url(self):
        from mycode.web.project_fetch import validate_github_url, FetchError
        with pytest.raises(FetchError, match="Invalid GitHub URL"):
            validate_github_url("https://gitlab.com/user/repo")

    def test_invalid_url_no_repo(self):
        from mycode.web.project_fetch import validate_github_url, FetchError
        with pytest.raises(FetchError, match="Invalid GitHub URL"):
            validate_github_url("https://github.com/user")

    def test_invalid_url_random_text(self):
        from mycode.web.project_fetch import validate_github_url, FetchError
        with pytest.raises(FetchError, match="Invalid GitHub URL"):
            validate_github_url("not a url at all")

    def test_extract_zip_valid(self):
        from mycode.web.project_fetch import extract_zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("app.py", "print('hello')")
            zf.writestr("requirements.txt", "flask")
        buf.seek(0)

        dest = Path(tempfile.mkdtemp())
        try:
            result = extract_zip(buf, dest)
            assert (result / "app.py").exists() or (dest / "app.py").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_extract_zip_single_dir(self):
        """Zip with single top-level directory should use that as root."""
        from mycode.web.project_fetch import extract_zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("myproject/app.py", "print('hello')")
            zf.writestr("myproject/requirements.txt", "flask")
        buf.seek(0)

        dest = Path(tempfile.mkdtemp())
        try:
            result = extract_zip(buf, dest)
            assert result.name == "myproject"
            assert (result / "app.py").exists()
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_extract_zip_too_large(self):
        from mycode.web.project_fetch import extract_zip, FetchError, MAX_PROJECT_SIZE_BYTES
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Write a file that would exceed the limit
            zf.writestr("big.bin", b"\0" * (MAX_PROJECT_SIZE_BYTES + 1))
        buf.seek(0)

        dest = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(FetchError, match="too large"):
                extract_zip(buf, dest)
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_extract_invalid_zip(self):
        from mycode.web.project_fetch import extract_zip, FetchError
        buf = io.BytesIO(b"not a zip file")
        dest = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(FetchError, match="Invalid zip"):
                extract_zip(buf, dest)
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def test_create_temp_dir(self):
        from mycode.web.project_fetch import create_temp_dir
        d = create_temp_dir()
        assert d.exists()
        assert d.is_dir()
        shutil.rmtree(d, ignore_errors=True)


# ── Schema Tests ──


class TestSchemas:
    """Tests for Pydantic request/response models."""

    def test_preflight_request_defaults(self):
        from mycode.web.schemas import PreflightRequest
        req = PreflightRequest()
        assert req.github_url == ""
        assert req.upload_filename == ""

    def test_preflight_response_defaults(self):
        from mycode.web.schemas import PreflightResponse
        resp = PreflightResponse()
        assert resp.job_id == ""
        assert resp.dependencies == []
        assert resp.error == ""

    def test_converse_response_defaults(self):
        from mycode.web.schemas import ConverseResponse
        resp = ConverseResponse()
        assert resp.turn == 1
        assert resp.done is False

    def test_progress_info(self):
        from mycode.web.schemas import ProgressInfo
        p = ProgressInfo(
            scenarios_total=10,
            scenarios_complete=5,
            current_scenario="memory_test",
            progress_pct=50,
        )
        assert p.progress_pct == 50

    def test_health_response_defaults(self):
        from mycode.web.schemas import HealthResponse
        h = HealthResponse()
        assert h.status == "ok"
        assert h.max_concurrent_jobs == 4

    def test_viability_status(self):
        from mycode.web.schemas import ViabilityStatus
        v = ViabilityStatus(
            viable=True, install_rate=0.9, import_rate=0.8, syntax_rate=1.0,
        )
        assert v.viable is True

    def test_report_response(self):
        from mycode.web.schemas import ReportResponse
        r = ReportResponse(job_id="j_123", report={"summary": "test"})
        assert r.report["summary"] == "test"


# ── Interface Turn-Based API Tests ──


class TestInterfaceTurnAPI:
    """Tests for the new turn-based conversation methods on ConversationalInterface."""

    def test_prepare_turn_1_returns_question_and_summary(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question, summary = interface.prepare_turn_1(simple_ingestion, "python")

        assert isinstance(question, str)
        assert len(question) > 20
        assert isinstance(summary, str)
        assert "flask" in summary.lower() or "Flask" in summary

    def test_prepare_turn_1_mentions_deps(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question, _ = interface.prepare_turn_1(simple_ingestion, "python")

        assert "flask" in question.lower()

    def test_prepare_turn_1_mentions_file_count(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question, _ = interface.prepare_turn_1(simple_ingestion, "python")

        assert "5 files" in question

    def test_prepare_turn_1_asks_confirmation(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question, _ = interface.prepare_turn_1(simple_ingestion, "python")

        assert "sound right" in question.lower() or "describe" in question.lower()

    def test_process_turn_1_returns_question(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question = interface.process_turn_1(
            "It's an inventory management app",
            simple_ingestion,
            "python",
        )

        assert isinstance(question, str)
        assert len(question) > 10

    def test_process_turn_1_skips_scale_if_mentioned(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question = interface.process_turn_1(
            "It's a warehouse app for 10 users",
            simple_ingestion,
            "python",
        )
        # Should not ask about user count since it was mentioned
        assert "how many" not in question.lower()

    def test_process_turn_1_asks_scale_if_not_mentioned(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        question = interface.process_turn_1(
            "It's a simple web app",
            simple_ingestion,
            "python",
        )
        assert "how many" in question.lower() or "people" in question.lower()

    def test_process_turn_2_returns_interface_result(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="It's an inventory management API for a warehouse",
            turn2_response="10 concurrent users, JSON payloads, deployed on a single server",
            ingestion=simple_ingestion,
            language="python",
        )

        assert isinstance(result, InterfaceResult)
        assert result.constraints is not None
        assert result.constraints.user_scale == 10
        assert result.intent is not None

    def test_process_turn_2_extracts_deployment(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="It's a web app",
            turn2_response="deployed on a single server, runs 24/7",
            ingestion=simple_ingestion,
            language="python",
        )

        assert result.constraints is not None
        assert result.constraints.deployment_context == "single_server"

    def test_process_turn_2_extracts_data_type(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="It processes CSV files",
            turn2_response="5 users, mostly tabular data",
            ingestion=simple_ingestion,
            language="python",
        )

        assert result.constraints is not None
        assert result.constraints.data_type == "tabular"

    def test_process_turn_2_handles_empty_responses(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="",
            turn2_response="",
            ingestion=simple_ingestion,
            language="python",
        )

        assert isinstance(result, InterfaceResult)
        assert result.constraints is not None
        # All constraint fields should be None
        assert result.constraints.user_scale is None

    def test_process_turn_2_has_project_summary(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="web app",
            turn2_response="10 users",
            ingestion=simple_ingestion,
            language="python",
        )

        assert result.project_summary
        assert "flask" in result.project_summary.lower() or "Flask" in result.project_summary

    def test_process_turn_2_has_intent(self, simple_ingestion):
        interface = ConversationalInterface(offline=True)
        result = interface.process_turn_2(
            turn1_response="inventory management system",
            turn2_response="10 users, worried about speed",
            ingestion=simple_ingestion,
            language="python",
        )

        intent_str = result.intent.as_intent_string()
        assert "inventory" in intent_str.lower() or "speed" in intent_str.lower()

    def test_existing_run_method_still_works(self, simple_ingestion):
        """Ensure the new methods don't break the existing run() flow."""

        class ScriptedIO:
            def __init__(self):
                self._responses = iter([
                    "It's a web app",  # Turn 1
                    "10 users, JSON data",  # Turn 2
                    "10",  # user_scale
                    "4",  # data_type (API responses)
                    "1",  # usage_pattern (steady)
                    "1",  # max_payload (small)
                ])
                self.displays = []

            def display(self, msg):
                self.displays.append(msg)

            def prompt(self, msg):
                return next(self._responses, "")

        interface = ConversationalInterface(offline=True, io=ScriptedIO())
        result = interface.run(simple_ingestion, "python")
        assert isinstance(result, InterfaceResult)
        assert result.intent is not None


# ── Route Handler Tests ──


class TestConverseHandler:
    """Tests for the conversation handler."""

    def test_converse_job_not_found(self):
        from mycode.web.routes import handle_converse
        resp = handle_converse("j_nonexistent", 1, "")
        assert resp.error
        assert "not found" in resp.error.lower()

    def test_converse_wrong_state(self):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "running"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 1, "")
            assert resp.error
            assert "running" in resp.error
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store

    def test_converse_no_ingestion(self):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 1, "")
            assert resp.error
            assert "ingestion" in resp.error.lower()
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store

    def test_converse_turn_1(self, simple_ingestion):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"
        job.ingestion = simple_ingestion
        job.language = "python"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 1, "")
            assert resp.question
            assert resp.project_summary
            assert not resp.done
            assert resp.turn == 1
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store

    def test_converse_turn_2(self, simple_ingestion):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"
        job.ingestion = simple_ingestion
        job.language = "python"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 2, "It's an inventory app for 10 users")
            assert resp.question
            assert not resp.done
            assert resp.turn == 2
            assert job.turn1_response == "It's an inventory app for 10 users"
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store

    def test_converse_turn_3_done(self, simple_ingestion):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"
        job.ingestion = simple_ingestion
        job.language = "python"
        job.turn1_response = "It's an inventory management API"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 3, "10 users, JSON payloads, single server")
            assert resp.done
            assert resp.turn == 3
            assert resp.constraints is not None
            assert resp.operational_intent
            assert job.status == "conversation_done"
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store

    def test_converse_invalid_turn(self, simple_ingestion):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"
        job.ingestion = simple_ingestion
        job.language = "python"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_converse(job.id, 99, "")
            assert resp.error
            assert "Invalid turn" in resp.error
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store


class TestAnalyzeHandler:
    """Tests for the analyze handler."""

    def test_analyze_job_not_found(self):
        from mycode.web.routes import handle_analyze
        resp = handle_analyze("j_nonexistent")
        assert resp.error
        assert "not found" in resp.error.lower()

    def test_analyze_wrong_state(self):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_analyze
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_running"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_analyze(job.id)
            assert resp.error
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store


class TestStatusHandler:
    """Tests for the status handler."""

    def test_status_job_not_found(self):
        from mycode.web.routes import handle_status
        resp = handle_status("j_nonexistent")
        assert resp.error

    def test_status_running_job(self):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_status
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "running"
        job.progress_scenarios_total = 10
        job.progress_scenarios_complete = 5
        job.progress_current_scenario = "memory_test"
        job.progress_start_time = time.time() - 30

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_status(job.id)
            assert resp.status == "running"
            assert resp.progress is not None
            assert resp.progress.scenarios_total == 10
            assert resp.progress.scenarios_complete == 5
            assert resp.progress.current_scenario == "memory_test"
            assert resp.progress.progress_pct == 50
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store


class TestReportHandler:
    """Tests for the report handler."""

    def test_report_job_not_found(self):
        from mycode.web.routes import handle_report
        resp = handle_report("j_nonexistent")
        assert resp.error

    def test_report_not_ready(self):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_report
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "running"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s
        try:
            resp = handle_report(job.id)
            assert resp.error
            assert "not ready" in resp.error.lower()
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store


class TestHealthHandler:
    """Tests for the health handler."""

    def test_health_response(self):
        from mycode.web.routes import handle_health
        resp = handle_health()
        assert resp.status == "ok"
        assert isinstance(resp.active_jobs, int)
        assert resp.max_concurrent_jobs > 0


# ── Worker Tests ──


class TestWorkerNullIO:
    """Tests for the NullIO class."""

    def test_null_io_display(self):
        from mycode.web.worker import NullIO
        io = NullIO()
        io.display("test")  # Should not raise

    def test_null_io_prompt(self):
        from mycode.web.worker import NullIO
        io = NullIO()
        result = io.prompt("test")
        assert result == ""


# ── App Module Tests ──


class TestDataclassToDict:
    """Tests for the dataclass-to-dict JSON serialiser."""

    def test_simple_dataclass(self):
        from mycode.web.app import _dataclass_to_dict
        from mycode.web.schemas import HealthResponse
        h = HealthResponse(status="ok", docker_available=True)
        d = _dataclass_to_dict(h)
        assert isinstance(d, dict)
        assert d["status"] == "ok"
        assert d["docker_available"] is True

    def test_nested_dataclass(self):
        from mycode.web.app import _dataclass_to_dict
        from mycode.web.schemas import StatusResponse, ProgressInfo
        s = StatusResponse(
            job_id="j_123",
            status="running",
            progress=ProgressInfo(
                scenarios_total=10,
                scenarios_complete=5,
                current_scenario="test",
                progress_pct=50,
            ),
        )
        d = _dataclass_to_dict(s)
        assert d["progress"]["scenarios_total"] == 10
        assert d["progress"]["current_scenario"] == "test"

    def test_list_of_dataclasses(self):
        from mycode.web.app import _dataclass_to_dict
        from mycode.web.schemas import DependencyStatus
        deps = [
            DependencyStatus(name="flask", installed_version="3.0.0"),
            DependencyStatus(name="requests", is_missing=True),
        ]
        result = _dataclass_to_dict(deps)
        assert len(result) == 2
        assert result[0]["name"] == "flask"
        assert result[1]["is_missing"] is True

    def test_non_dataclass_passthrough(self):
        from mycode.web.app import _dataclass_to_dict
        assert _dataclass_to_dict("hello") == "hello"
        assert _dataclass_to_dict(42) == 42
        assert _dataclass_to_dict(None) is None

    def test_dict_passthrough(self):
        from mycode.web.app import _dataclass_to_dict
        d = {"key": "value", "nested": {"a": 1}}
        result = _dataclass_to_dict(d)
        assert result == d


# ── Full Conversation Flow Test ──


class TestFullConversationFlow:
    """End-to-end test of the turn-based conversation through route handlers."""

    def test_three_turn_flow(self, simple_ingestion):
        from mycode.web.jobs import JobStore
        from mycode.web.routes import handle_converse
        import mycode.web.routes as routes_module
        import mycode.web.jobs as jobs_module

        s = JobStore()
        job = s.create()
        job.status = "preflight_complete"
        job.ingestion = simple_ingestion
        job.language = "python"

        old_store = jobs_module.store
        jobs_module.store = s
        routes_module.store = s

        try:
            # Turn 1: Get initial question
            r1 = handle_converse(job.id, 1, "")
            assert r1.question
            assert r1.project_summary
            assert not r1.done

            # Turn 2: Answer Turn 1, get Turn 2 question
            r2 = handle_converse(job.id, 2, "It manages warehouse inventory for a small team")
            assert r2.question
            assert not r2.done

            # Turn 3: Answer Turn 2, get constraints
            r3 = handle_converse(job.id, 3, "5 users, JSON data, deployed on AWS, runs all day")
            assert r3.done
            assert r3.constraints is not None
            assert r3.constraints.get("user_scale") == 5
            assert r3.operational_intent
            assert job.status == "conversation_done"
        finally:
            jobs_module.store = old_store
            routes_module.store = old_store
