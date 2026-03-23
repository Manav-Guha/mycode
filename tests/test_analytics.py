"""Tests for myCode web analytics — SQLite job logging and admin stats."""

import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest

from mycode.web.analytics import (
    close_connection,
    get_admin_stats,
    get_db,
    log_download,
    log_job_completed,
    log_job_started,
    validate_source,
    _local,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary SQLite database for every test."""
    db_path = str(tmp_path / "test_analytics.db")
    with patch.dict(os.environ, {"MYCODE_DB_PATH": db_path}):
        # Clear any cached connection from a prior test
        close_connection()
        yield db_path
        close_connection()


# ── Source validation ──


def test_validate_source_valid():
    assert validate_source("internal") == "internal"
    assert validate_source("hn") == "hn"
    assert validate_source("public") == "public"
    assert validate_source("cli") == "cli"
    assert validate_source("test_group") == "test_group"


def test_validate_source_invalid():
    assert validate_source("bogus") == "public"
    assert validate_source("") == "public"
    assert validate_source("INTERNAL") == "public"


# ── DB creation ──


def test_db_created_on_first_access(tmp_path):
    """DB file and table created if path doesn't exist."""
    nested = str(tmp_path / "sub" / "dir" / "analytics.db")
    with patch.dict(os.environ, {"MYCODE_DB_PATH": nested}):
        close_connection()
        conn = get_db()
        # Table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='job_log'"
        )
        assert cur.fetchone() is not None
        close_connection()


# ── log_job_started ──


def test_log_job_started():
    log_job_started("j_abc123", "hn", "https://github.com/user/repo")
    db = get_db()
    row = db.execute("SELECT * FROM job_log WHERE job_id='j_abc123'").fetchone()
    assert row is not None
    # Columns: id, job_id, source, repo_url, started_at, completed_at, status,
    #          findings_critical, findings_warning, findings_info, scenarios_run,
    #          pdf_downloaded, json_downloaded
    assert row[1] == "j_abc123"  # job_id
    assert row[2] == "hn"  # source
    assert row[3] == "https://github.com/user/repo"  # repo_url
    assert row[4] is not None  # started_at
    assert row[5] is None  # completed_at
    assert row[6] == "started"  # status


def test_log_job_started_invalid_source_defaults():
    log_job_started("j_bad", "hacker", "https://github.com/x/y")
    db = get_db()
    row = db.execute("SELECT source FROM job_log WHERE job_id='j_bad'").fetchone()
    assert row[0] == "public"


# ── log_job_completed ──


def test_log_job_completed():
    log_job_started("j_comp", "public", "https://github.com/a/b")
    log_job_completed("j_comp", "completed", 2, 3, 1, 10)
    db = get_db()
    row = db.execute("SELECT * FROM job_log WHERE job_id='j_comp'").fetchone()
    assert row[5] is not None  # completed_at
    assert row[6] == "completed"  # status
    assert row[7] == 2  # findings_critical
    assert row[8] == 3  # findings_warning
    assert row[9] == 1  # findings_info
    assert row[10] == 10  # scenarios_run


def test_log_job_completed_timeout():
    log_job_started("j_to", "cli", "https://github.com/a/b")
    log_job_completed("j_to", "timeout", 0, 0, 0, 5)
    db = get_db()
    row = db.execute("SELECT status FROM job_log WHERE job_id='j_to'").fetchone()
    assert row[0] == "timeout"


# ── log_download ──


def test_log_download_pdf():
    log_job_started("j_dl", "public", "https://github.com/a/b")
    log_download("j_dl", "pdf")
    db = get_db()
    row = db.execute(
        "SELECT pdf_downloaded, json_downloaded FROM job_log WHERE job_id='j_dl'"
    ).fetchone()
    assert row[0] == 1  # pdf_downloaded
    assert row[1] == 0  # json_downloaded


def test_log_download_json():
    log_job_started("j_dl2", "public", "https://github.com/a/b")
    log_download("j_dl2", "json")
    db = get_db()
    row = db.execute(
        "SELECT pdf_downloaded, json_downloaded FROM job_log WHERE job_id='j_dl2'"
    ).fetchone()
    assert row[0] == 0
    assert row[1] == 1


def test_log_download_invalid_type():
    """Invalid download type is silently ignored."""
    log_job_started("j_dl3", "public", "https://github.com/a/b")
    log_download("j_dl3", "csv")
    db = get_db()
    row = db.execute(
        "SELECT pdf_downloaded, json_downloaded FROM job_log WHERE job_id='j_dl3'"
    ).fetchone()
    assert row[0] == 0
    assert row[1] == 0


# ── Admin stats ──


def _seed_jobs():
    """Insert a realistic set of jobs for stats tests."""
    log_job_started("j_1", "public", "https://github.com/a/repo1")
    log_job_completed("j_1", "completed", 2, 1, 0, 8)
    log_download("j_1", "pdf")

    log_job_started("j_2", "hn", "https://github.com/b/repo2")
    log_job_completed("j_2", "completed", 0, 2, 1, 6)
    log_download("j_2", "json")

    log_job_started("j_3", "internal", "https://github.com/c/repo3")
    log_job_completed("j_3", "completed", 1, 0, 3, 12)
    log_download("j_3", "pdf")
    log_download("j_3", "json")

    log_job_started("j_4", "public", "https://github.com/a/repo1")  # return repo
    log_job_completed("j_4", "failed", 0, 0, 0, 0)

    log_job_started("j_5", "test_group", "https://github.com/d/repo4")
    log_job_completed("j_5", "timeout", 0, 0, 0, 3)

    log_job_started("j_6", "public", "zip_upload")
    log_job_completed("j_6", "completed", 1, 1, 1, 5)


def test_admin_stats_valid():
    _seed_jobs()
    stats = get_admin_stats()

    assert stats["total_jobs"] == 6
    assert stats["by_source"]["public"] == 3
    assert stats["by_source"]["hn"] == 1
    assert stats["by_source"]["internal"] == 1
    assert stats["by_source"]["test_group"] == 1
    assert stats["by_status"]["completed"] == 4
    assert stats["by_status"]["failed"] == 1
    assert stats["by_status"]["timeout"] == 1
    assert "started" not in stats["by_status"]


def test_admin_stats_external_excludes_internal():
    _seed_jobs()
    stats = get_admin_stats()
    # 6 total, minus 1 internal, minus 1 test_group = 4 external
    assert stats["external_jobs"] == 4


def test_admin_stats_return_repos():
    _seed_jobs()
    stats = get_admin_stats()
    # repo1 appears twice → 1 return repo
    assert stats["return_repos"] == 1
    # repo1, repo2, repo3, repo4 = 4 unique (zip_upload excluded)
    assert stats["unique_repos"] == 4


def test_admin_stats_download_rates():
    _seed_jobs()
    stats = get_admin_stats()
    # 4 completed. PDF: j_1 + j_3 = 2/4 = 0.5. JSON: j_2 + j_3 = 2/4 = 0.5
    assert stats["pdf_download_rate"] == 0.5
    assert stats["json_download_rate"] == 0.5


def test_admin_stats_avg_findings():
    _seed_jobs()
    stats = get_admin_stats()
    # Completed: j_1(2,1,0), j_2(0,2,1), j_3(1,0,3), j_6(1,1,1)
    # avg critical: (2+0+1+1)/4 = 1.0
    assert stats["avg_findings_critical"] == 1.0
    # avg warning: (1+2+0+1)/4 = 1.0
    assert stats["avg_findings_warning"] == 1.0


def test_admin_stats_empty_db():
    stats = get_admin_stats()
    assert stats["total_jobs"] == 0
    assert stats["external_jobs"] == 0
    assert stats["by_source"] == {}
    assert stats["by_status"] == {}
    assert stats["pdf_download_rate"] == 0.0
    assert stats["json_download_rate"] == 0.0
    assert stats["unique_repos"] == 0
    assert stats["return_repos"] == 0


# ── Admin stats endpoint (FastAPI TestClient) ──


@pytest.fixture
def client():
    """FastAPI TestClient for endpoint tests."""
    # Import inside fixture to avoid import errors when FastAPI not installed
    from fastapi.testclient import TestClient
    from mycode.web.app import app
    return TestClient(app)


def test_admin_stats_endpoint_no_key(client):
    """Returns 403 when MYCODE_ADMIN_KEY is not set."""
    with patch.dict(os.environ, {}, clear=False):
        # Ensure key is unset
        os.environ.pop("MYCODE_ADMIN_KEY", None)
        resp = client.get("/api/admin/stats?key=anything")
        assert resp.status_code == 403


def test_admin_stats_endpoint_wrong_key(client):
    with patch.dict(os.environ, {"MYCODE_ADMIN_KEY": "secret123"}):
        resp = client.get("/api/admin/stats?key=wrong")
        assert resp.status_code == 403


def test_admin_stats_endpoint_valid_key(client):
    _seed_jobs()
    with patch.dict(os.environ, {"MYCODE_ADMIN_KEY": "secret123"}):
        resp = client.get("/api/admin/stats?key=secret123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs"] == 6
        assert "by_source" in data
        assert "last_24h" in data


# ── Source propagation through preflight endpoint ──


def test_download_log_endpoint(client):
    """POST /api/report/{job_id}/download-log logs the download type."""
    log_job_started("j_dlep", "public", "https://github.com/x/y")
    resp = client.post(
        "/api/report/j_dlep/download-log",
        data={"type": "json"},
    )
    assert resp.status_code == 200
    db = get_db()
    row = db.execute(
        "SELECT json_downloaded FROM job_log WHERE job_id='j_dlep'"
    ).fetchone()
    assert row[0] == 1
