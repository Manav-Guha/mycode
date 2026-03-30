"""Tests for GET /api/admin/jobs endpoint and get_admin_jobs() query."""

import json
import os
import time
from unittest.mock import patch

import pytest

from mycode.web.analytics import (
    close_connection,
    get_admin_jobs,
    get_db,
    log_download,
    log_job_completed,
    log_job_started,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary SQLite database for every test."""
    db_path = str(tmp_path / "test_admin_jobs.db")
    with patch.dict(os.environ, {"MYCODE_DB_PATH": db_path}):
        close_connection()
        yield db_path
        close_connection()


# ── get_admin_jobs() unit tests ──


def test_admin_jobs_empty_db():
    jobs, total = get_admin_jobs()
    assert jobs == []
    assert total == 0


def test_admin_jobs_returns_records():
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 2, 1, 0, 8)
    log_job_started("j_2", "hn", "https://github.com/b/repo2")
    log_job_completed("j_2", "completed", 0, 2, 1, 6)
    log_job_started("j_3", "test_group", "https://github.com/c/repo3")
    log_job_completed("j_3", "failed", 0, 0, 0, 0)

    jobs, total = get_admin_jobs()
    assert total == 3
    assert len(jobs) == 3
    # Check all expected fields present on first record
    required_keys = {
        "job_id", "source", "status", "created_at", "completed_at",
        "duration_seconds", "repo_identifier", "languages_detected",
        "deps_found", "findings_critical", "findings_warning",
        "findings_info", "finding_titles", "predictions_count",
        "pdf_downloaded", "json_downloaded",
    }
    assert required_keys <= set(jobs[0].keys())


def test_admin_jobs_filter_by_source():
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 1, 0, 0, 5)
    log_job_started("j_2", "test_group", "https://github.com/b/repo2")
    log_job_completed("j_2", "completed", 0, 1, 0, 3)
    log_job_started("j_3", "test_group", "https://github.com/c/repo3")
    log_job_completed("j_3", "failed", 0, 0, 0, 0)

    jobs, total = get_admin_jobs(source="test_group")
    assert total == 2
    assert len(jobs) == 2
    assert all(j["source"] == "test_group" for j in jobs)


def test_admin_jobs_filter_by_status():
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 1, 0, 0, 5)
    log_job_started("j_2", "public", "https://github.com/b/repo2")
    log_job_completed("j_2", "failed", 0, 0, 0, 0)

    jobs, total = get_admin_jobs(status="completed")
    assert total == 1
    assert jobs[0]["job_id"] == "j_1"


def test_admin_jobs_pagination():
    for i in range(5):
        log_job_started(f"j_{i}", "public", f"https://github.com/a/repo{i}")
        log_job_completed(f"j_{i}", "completed", 0, 0, 0, 1)

    jobs_p1, total = get_admin_jobs(limit=2, offset=0)
    assert total == 5
    assert len(jobs_p1) == 2

    jobs_p2, total = get_admin_jobs(limit=2, offset=2)
    assert total == 5
    assert len(jobs_p2) == 2

    # No overlap
    ids_p1 = {j["job_id"] for j in jobs_p1}
    ids_p2 = {j["job_id"] for j in jobs_p2}
    assert ids_p1.isdisjoint(ids_p2)


def test_admin_jobs_limit_cap():
    """Requesting limit > 200 is silently capped to 200."""
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 0, 0, 0, 1)
    # Should not raise — just capped internally
    jobs, total = get_admin_jobs(limit=500)
    assert total == 1
    assert len(jobs) == 1


def test_admin_jobs_newest_first():
    log_job_started("j_old", "public", "https://github.com/a/old")
    log_job_completed("j_old", "completed", 0, 0, 0, 1)
    # Small sleep to ensure different timestamps
    time.sleep(0.05)
    log_job_started("j_new", "public", "https://github.com/a/new")
    log_job_completed("j_new", "completed", 0, 0, 0, 1)

    jobs, _ = get_admin_jobs()
    assert jobs[0]["job_id"] == "j_new"
    assert jobs[1]["job_id"] == "j_old"


def test_admin_jobs_duration_computed():
    log_job_started("j_dur", "public", "https://github.com/a/repo")
    # Manually set timestamps for deterministic testing
    db = get_db()
    db.execute(
        "UPDATE job_log SET started_at='2026-03-30T10:00:00+00:00' WHERE job_id='j_dur'"
    )
    db.commit()
    log_job_completed("j_dur", "completed", 0, 0, 0, 1)
    # Override completed_at for deterministic result
    db.execute(
        "UPDATE job_log SET completed_at='2026-03-30T10:03:42+00:00' WHERE job_id='j_dur'"
    )
    db.commit()

    jobs, _ = get_admin_jobs()
    assert jobs[0]["duration_seconds"] == 222.0


def test_admin_jobs_null_fields_for_incomplete():
    """Started job with no completion has null duration and completion fields."""
    log_job_started("j_inc", "public", "https://github.com/a/repo")

    jobs, _ = get_admin_jobs()
    j = jobs[0]
    assert j["completed_at"] is None
    assert j["duration_seconds"] is None
    assert j["findings_critical"] is None
    assert j["finding_titles"] == []


def test_admin_jobs_finding_titles_parsed():
    """finding_titles is returned as a list, not a JSON string."""
    log_job_started("j_ft", "public", "https://github.com/a/repo")
    titles = ["Memory leak under load", "Response time degradation"]
    log_job_completed(
        "j_ft", "completed", 1, 1, 0, 5,
        finding_titles=json.dumps(titles),
    )

    jobs, _ = get_admin_jobs()
    assert jobs[0]["finding_titles"] == titles
    assert isinstance(jobs[0]["finding_titles"], list)


def test_admin_jobs_languages_returned_as_list():
    """languages_detected is returned as a JSON list, not a CSV string."""
    log_job_started("j_lang", "public", "https://github.com/a/repo")
    log_job_completed(
        "j_lang", "completed", 0, 0, 0, 1,
        languages_detected="javascript,python",
    )

    jobs, _ = get_admin_jobs()
    assert jobs[0]["languages_detected"] == ["javascript", "python"]
    assert isinstance(jobs[0]["languages_detected"], list)


def test_admin_jobs_new_fields_stored():
    """languages_detected, deps_found, finding_titles, predictions_count are stored."""
    log_job_started("j_nf", "public", "https://github.com/a/repo")
    log_job_completed(
        "j_nf", "completed", 1, 2, 3, 10,
        languages_detected="python",
        deps_found=15,
        finding_titles=json.dumps(["Memory issue", "Timeout"]),
        predictions_count=5,
    )

    jobs, _ = get_admin_jobs()
    j = jobs[0]
    assert j["languages_detected"] == ["python"]
    assert j["deps_found"] == 15
    assert j["finding_titles"] == ["Memory issue", "Timeout"]
    assert j["predictions_count"] == 5


def test_zip_filename_captured():
    """Zip uploads store the filename in repo_url."""
    log_job_started("j_zip", "public", "zip:myproject.zip")
    log_job_completed("j_zip", "completed", 0, 0, 0, 1)

    jobs, _ = get_admin_jobs()
    assert jobs[0]["repo_identifier"] == "zip:myproject.zip"


def test_admin_jobs_download_flags():
    log_job_started("j_dl", "public", "https://github.com/a/repo")
    log_job_completed("j_dl", "completed", 0, 0, 0, 1)
    log_download("j_dl", "pdf")

    jobs, _ = get_admin_jobs()
    assert jobs[0]["pdf_downloaded"] is True
    assert jobs[0]["json_downloaded"] is False


def test_admin_jobs_total_with_filters():
    """total reflects filtered count, not all jobs."""
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 0, 0, 0, 1)
    log_job_started("j_2", "hn", "https://github.com/b/repo2")
    log_job_completed("j_2", "completed", 0, 0, 0, 1)
    log_job_started("j_3", "hn", "https://github.com/c/repo3")
    log_job_completed("j_3", "failed", 0, 0, 0, 0)

    _, total = get_admin_jobs(source="hn")
    assert total == 2

    _, total = get_admin_jobs(source="hn", status="completed")
    assert total == 1


# ── FastAPI endpoint tests ──


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from mycode.web.app import app
    return TestClient(app)


def test_admin_jobs_endpoint_no_key(client):
    os.environ.pop("MYCODE_ADMIN_KEY", None)
    resp = client.get("/api/admin/jobs")
    assert resp.status_code == 403


def test_admin_jobs_endpoint_wrong_key(client):
    with patch.dict(os.environ, {"MYCODE_ADMIN_KEY": "secret123"}):
        resp = client.get("/api/admin/jobs?key=wrong")
        assert resp.status_code == 403


def test_admin_jobs_endpoint_valid(client):
    log_job_started("j_ep1", "test_group", "https://github.com/a/repo")
    log_job_completed("j_ep1", "completed", 1, 0, 0, 5)

    with patch.dict(os.environ, {"MYCODE_ADMIN_KEY": "secret123"}):
        resp = client.get("/api/admin/jobs?key=secret123&source=test_group")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "count" in data
        assert "total" in data
        assert data["total"] == 1
        assert data["count"] == 1
        assert data["jobs"][0]["job_id"] == "j_ep1"


def test_admin_jobs_endpoint_pagination(client):
    for i in range(5):
        log_job_started(f"j_pg{i}", "public", f"https://github.com/a/r{i}")
        log_job_completed(f"j_pg{i}", "completed", 0, 0, 0, 1)

    with patch.dict(os.environ, {"MYCODE_ADMIN_KEY": "s"}):
        resp = client.get("/api/admin/jobs?key=s&limit=2&offset=0")
        data = resp.json()
        assert data["count"] == 2
        assert data["total"] == 5
