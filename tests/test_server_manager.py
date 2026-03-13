"""Tests for server_manager.py — HTTP stress testing Phase 1.

Tests framework command generation, entry file detection, health poll
timeout behaviour, process group teardown, and can_start_server logic.
"""

import ast
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mycode.server_manager import (
    FrameworkDetection,
    ServerInfo,
    ServerStartResult,
    build_startup_command,
    can_start_server,
    detect_framework_entry,
    kill_process_group,
    start_server,
    stop_server,
    _check_health,
    _check_health_tcp,
    _diagnose_startup_failure,
    _detect_js_framework,
    _detect_python_framework,
    _file_to_module,
    _find_app_assignment,
    _find_available_port,
    _health_check_url,
    _pick_preferred,
    STARTUP_TIMEOUT_SECONDS,
)
from mycode.ingester import (
    FileAnalysis,
    FunctionInfo,
    ImportInfo,
    IngestionResult,
    GlobalVarInfo,
)


# ── Helpers ──


def _make_ingestion(file_analyses=None, project_path="/tmp/proj"):
    """Build a minimal IngestionResult for testing."""
    return IngestionResult(
        project_path=project_path,
        file_analyses=file_analyses or [],
    )


def _make_file_analysis(file_path, imports=None):
    """Build a minimal FileAnalysis."""
    return FileAnalysis(
        file_path=file_path,
        imports=imports or [],
    )


# ── Framework Command Generation ──


class TestBuildStartupCommand:
    """Test startup command generation for each supported framework."""

    def test_streamlit_command(self):
        detection = FrameworkDetection(
            framework="streamlit", entry_file="app.py"
        )
        cmd, env = build_startup_command(detection, 8501)
        assert cmd == [
            "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
        assert env == {}

    def test_fastapi_command(self):
        detection = FrameworkDetection(
            framework="fastapi",
            entry_file="app.py",
            app_variable="app",
            module_name="app",
        )
        cmd, env = build_startup_command(detection, 8000)
        assert cmd == [
            "uvicorn", "app:app",
            "--port", "8000",
            "--host", "0.0.0.0",
        ]
        assert env == {}

    def test_fastapi_nested_module(self):
        detection = FrameworkDetection(
            framework="fastapi",
            entry_file="src/main.py",
            app_variable="application",
            module_name="src.main",
        )
        cmd, env = build_startup_command(detection, 9000)
        assert cmd[1] == "src.main:application"

    def test_flask_command(self):
        detection = FrameworkDetection(
            framework="flask",
            entry_file="server.py",
            app_variable="app",
        )
        cmd, env = build_startup_command(detection, 5000)
        assert cmd == ["flask", "run", "--port", "5000"]
        assert env == {"FLASK_APP": "server.py"}

    def test_express_command(self):
        detection = FrameworkDetection(
            framework="express",
            entry_file="index.js",
        )
        cmd, env = build_startup_command(detection, 3000)
        assert cmd == ["node", "index.js"]
        assert env == {"PORT": "3000"}

    def test_unsupported_framework_raises(self):
        detection = FrameworkDetection(
            framework="django", entry_file="manage.py"
        )
        with pytest.raises(ValueError, match="Unsupported framework"):
            build_startup_command(detection, 8000)


# ── Port Selection ──


class TestPortSelection:
    """Test dynamic port assignment."""

    def test_finds_available_port(self):
        try:
            port = _find_available_port()
        except PermissionError:
            pytest.skip("Socket binding not permitted in sandbox")
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_different_ports_on_successive_calls(self):
        try:
            ports = {_find_available_port() for _ in range(5)}
        except PermissionError:
            pytest.skip("Socket binding not permitted in sandbox")
        # Should get at least 2 different ports (not guaranteed but very likely)
        assert len(ports) >= 2


# ── Entry File Detection (Python) ──


class TestDetectPythonFramework:
    """Test Python framework detection from ingestion results."""

    def test_detect_fastapi(self, tmp_path):
        """Detect FastAPI app assignment."""
        app_py = tmp_path / "app.py"
        app_py.write_text(textwrap.dedent("""\
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/")
            def root():
                return {"message": "hello"}
        """))

        fa = _make_file_analysis("app.py", imports=[
            ImportInfo(module="fastapi", names=["FastAPI"], is_from_import=True),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "fastapi"
        assert result.entry_file == "app.py"
        assert result.app_variable == "app"
        assert result.module_name == "app"

    def test_detect_flask(self, tmp_path):
        """Detect Flask app assignment."""
        app_py = tmp_path / "server.py"
        app_py.write_text(textwrap.dedent("""\
            from flask import Flask

            app = Flask(__name__)

            @app.route("/")
            def hello():
                return "Hello World"
        """))

        fa = _make_file_analysis("server.py", imports=[
            ImportInfo(module="flask", names=["Flask"], is_from_import=True),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "flask"
        assert result.entry_file == "server.py"
        assert result.app_variable == "app"

    def test_detect_streamlit(self, tmp_path):
        """Detect Streamlit entry file with st. calls."""
        app_py = tmp_path / "app.py"
        app_py.write_text(textwrap.dedent("""\
            import streamlit as st

            st.title("Dashboard")
            st.write("Hello")
        """))

        fa = _make_file_analysis("app.py", imports=[
            ImportInfo(module="streamlit", alias="st"),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "streamlit"
        assert result.entry_file == "app.py"

    def test_streamlit_skips_imported_files(self, tmp_path):
        """Streamlit: files imported by others are not entry points."""
        (tmp_path / "components.py").write_text(textwrap.dedent("""\
            import streamlit as st

            def sidebar():
                st.sidebar.title("Nav")
        """))
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            import streamlit as st
            import components

            st.title("Main")
            components.sidebar()
        """))

        fa_comp = _make_file_analysis("components.py", imports=[
            ImportInfo(module="streamlit", alias="st"),
        ])
        fa_app = _make_file_analysis("app.py", imports=[
            ImportInfo(module="streamlit", alias="st"),
            ImportInfo(module="components"),
        ])
        ingestion = _make_ingestion([fa_comp, fa_app], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.entry_file == "app.py"

    def test_no_framework_detected(self, tmp_path):
        """Returns None for projects without server frameworks."""
        app_py = tmp_path / "main.py"
        app_py.write_text("print('hello')\n")

        fa = _make_file_analysis("main.py")
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is None

    def test_fastapi_takes_priority_over_flask(self, tmp_path):
        """When both FastAPI and Flask detected, FastAPI wins."""
        (tmp_path / "api.py").write_text(textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()
        """))
        (tmp_path / "web.py").write_text(textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)
        """))

        fa1 = _make_file_analysis("api.py", imports=[
            ImportInfo(module="fastapi", names=["FastAPI"], is_from_import=True),
        ])
        fa2 = _make_file_analysis("web.py", imports=[
            ImportInfo(module="flask", names=["Flask"], is_from_import=True),
        ])
        ingestion = _make_ingestion([fa1, fa2], str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "fastapi"

    def test_prefers_app_py_over_other_names(self, tmp_path):
        """When multiple candidates, prefer app.py."""
        for name in ("server.py", "app.py", "main.py"):
            (tmp_path / name).write_text(textwrap.dedent("""\
                from flask import Flask
                app = Flask(__name__)
            """))

        fas = []
        for name in ("server.py", "app.py", "main.py"):
            fas.append(_make_file_analysis(name, imports=[
                ImportInfo(module="flask", names=["Flask"], is_from_import=True),
            ]))
        ingestion = _make_ingestion(fas, str(tmp_path))

        result = _detect_python_framework(ingestion, tmp_path)
        assert result is not None
        assert result.entry_file == "app.py"


# ── Entry File Detection (JavaScript) ──


class TestDetectJsFramework:
    """Test Express.js detection."""

    def test_detect_express(self, tmp_path):
        """Detect Express app with require and app.listen."""
        index_js = tmp_path / "index.js"
        index_js.write_text(textwrap.dedent("""\
            const express = require('express');
            const app = express();

            app.get('/', (req, res) => res.send('Hello'));
            app.listen(3000);
        """))

        fa = _make_file_analysis("index.js", imports=[
            ImportInfo(module="express"),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_js_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "express"
        assert result.entry_file == "index.js"

    def test_detect_express_es_import(self, tmp_path):
        """Detect Express with ES module import syntax."""
        app_js = tmp_path / "app.js"
        app_js.write_text(textwrap.dedent("""\
            import express from 'express';
            const app = express();

            app.listen(3000);
        """))

        fa = _make_file_analysis("app.js", imports=[
            ImportInfo(module="express"),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_js_framework(ingestion, tmp_path)
        assert result is not None
        assert result.framework == "express"

    def test_no_express_without_listen(self, tmp_path):
        """Express without app.listen is not a server entry point."""
        router_js = tmp_path / "router.js"
        router_js.write_text(textwrap.dedent("""\
            const express = require('express');
            const router = express.Router();
            module.exports = router;
        """))

        fa = _make_file_analysis("router.js", imports=[
            ImportInfo(module="express"),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))

        result = _detect_js_framework(ingestion, tmp_path)
        assert result is None


# ── Health Check URLs ──


class TestHealthCheckUrl:
    def test_fastapi_url(self):
        assert _health_check_url("fastapi", 8000) == "http://localhost:8000/health"

    def test_flask_url(self):
        assert _health_check_url("flask", 5000) == "http://localhost:5000/health"

    def test_express_url(self):
        assert _health_check_url("express", 3000) == "http://localhost:3000/health"

    def test_streamlit_url(self):
        assert _health_check_url("streamlit", 8501) == "http://localhost:8501/"


# ── Health Check ──


class TestHealthCheck:
    def test_tcp_check_on_open_port(self):
        """TCP check returns True when a port is listening."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
        except PermissionError:
            pytest.skip("Socket binding not permitted in sandbox")
        try:
            assert _check_health_tcp(port) is True
        finally:
            s.close()

    def test_tcp_check_on_closed_port(self):
        """TCP check returns False when nothing is listening."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        except PermissionError:
            pytest.skip("Socket binding not permitted in sandbox")
        # Port is now closed
        assert _check_health_tcp(port) is False


# ── Startup Failure Diagnosis ──


class TestDiagnoseStartupFailure:
    def test_port_conflict(self):
        msg = _diagnose_startup_failure(
            "OSError: [Errno 48] Address already in use", "flask"
        )
        assert "port conflict" in msg

    def test_missing_module(self):
        msg = _diagnose_startup_failure(
            "ModuleNotFoundError: No module named 'psycopg2'", "fastapi"
        )
        assert "missing dependency" in msg
        assert "psycopg2" in msg

    def test_missing_api_key(self):
        msg = _diagnose_startup_failure(
            "ValueError: API_KEY environment variable not set", "flask"
        )
        assert "API key" in msg or "environment variable" in msg

    def test_database_connection(self):
        msg = _diagnose_startup_failure(
            "sqlalchemy.exc.OperationalError: could not connect to database",
            "fastapi",
        )
        assert "database" in msg or "connection" in msg

    def test_empty_stderr(self):
        msg = _diagnose_startup_failure("", "flask")
        assert "flask" in msg.lower()

    def test_syntax_error(self):
        msg = _diagnose_startup_failure(
            "SyntaxError: invalid syntax (app.py, line 5)", "streamlit"
        )
        assert "syntax error" in msg

    def test_eaddrinuse_node(self):
        msg = _diagnose_startup_failure(
            "Error: listen EADDRINUSE: address already in use :::3000", "express"
        )
        assert "port conflict" in msg


# ── Process Group Teardown ──


class TestKillProcessGroup:
    def test_already_exited_process(self):
        """kill_process_group on an already-exited process returns True."""
        proc = MagicMock()
        proc.poll.return_value = 0
        assert kill_process_group(proc) is True

    def test_none_process(self):
        """kill_process_group(None) returns True."""
        assert kill_process_group(None) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_kills_real_subprocess(self):
        """Start a real sleep process and verify kill_process_group terminates it."""
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.poll() is None  # still running
        result = kill_process_group(proc, grace_seconds=1)
        assert result is True
        assert proc.poll() is not None


# ── Stop Server ──


class TestStopServer:
    def test_stop_server_calls_kill(self):
        """stop_server delegates to kill_process_group."""
        proc = MagicMock()
        proc.poll.return_value = 0
        server = ServerInfo(
            process=proc, port=8000, framework="flask", entry_file="app.py"
        )
        result = stop_server(server)
        assert result is True


# ── can_start_server ──


class TestCanStartServer:
    def test_returns_true_for_fastapi(self, tmp_path):
        """can_start_server returns True for a FastAPI project."""
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()
        """))

        fa = _make_file_analysis("app.py", imports=[
            ImportInfo(module="fastapi", names=["FastAPI"], is_from_import=True),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))
        assert can_start_server(ingestion, tmp_path, "python") is True

    def test_returns_false_for_plain_python(self, tmp_path):
        """can_start_server returns False for non-server projects."""
        (tmp_path / "main.py").write_text("print('hello')\n")

        fa = _make_file_analysis("main.py")
        ingestion = _make_ingestion([fa], str(tmp_path))
        assert can_start_server(ingestion, tmp_path, "python") is False

    def test_returns_true_for_express(self, tmp_path):
        """can_start_server returns True for Express projects."""
        (tmp_path / "index.js").write_text(textwrap.dedent("""\
            const express = require('express');
            const app = express();
            app.listen(3000);
        """))

        fa = _make_file_analysis("index.js", imports=[
            ImportInfo(module="express"),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))
        assert can_start_server(ingestion, tmp_path, "javascript") is True

    def test_returns_false_for_unsupported_language(self, tmp_path):
        """can_start_server returns False for unsupported languages."""
        ingestion = _make_ingestion([], str(tmp_path))
        assert can_start_server(ingestion, tmp_path, "rust") is False


# ── detect_framework_entry dispatcher ──


class TestDetectFrameworkEntry:
    def test_dispatches_to_python(self, tmp_path):
        (tmp_path / "app.py").write_text("from flask import Flask\napp = Flask(__name__)\n")
        fa = _make_file_analysis("app.py", imports=[
            ImportInfo(module="flask", names=["Flask"], is_from_import=True),
        ])
        ingestion = _make_ingestion([fa], str(tmp_path))
        result = detect_framework_entry(ingestion, tmp_path, "python")
        assert result is not None
        assert result.framework == "flask"

    def test_dispatches_to_js(self, tmp_path):
        (tmp_path / "index.js").write_text(
            "const express = require('express');\nconst app = express();\napp.listen(3000);\n"
        )
        fa = _make_file_analysis("index.js", imports=[ImportInfo(module="express")])
        ingestion = _make_ingestion([fa], str(tmp_path))
        result = detect_framework_entry(ingestion, tmp_path, "javascript")
        assert result is not None
        assert result.framework == "express"

    def test_returns_none_for_unknown_language(self, tmp_path):
        ingestion = _make_ingestion([], str(tmp_path))
        assert detect_framework_entry(ingestion, tmp_path, "go") is None


# ── AST Helpers ──


class TestFindAppAssignment:
    def test_finds_fastapi_assignment(self):
        tree = ast.parse("app = FastAPI()")
        result = _find_app_assignment(tree, "FastAPI")
        assert result is not None
        assert result[0] == "app"

    def test_finds_custom_var_name(self):
        tree = ast.parse("server = Flask(__name__)")
        result = _find_app_assignment(tree, "Flask")
        assert result is not None
        assert result[0] == "server"

    def test_returns_none_when_not_found(self):
        tree = ast.parse("x = 42")
        assert _find_app_assignment(tree, "FastAPI") is None


class TestFileToModule:
    def test_simple(self):
        assert _file_to_module("app.py") == "app"

    def test_nested(self):
        assert _file_to_module("src/main.py") == "src.main"

    def test_init(self):
        assert _file_to_module("pkg/__init__.py") == "pkg"


class TestPickPreferred:
    def test_single_candidate(self):
        assert _pick_preferred(["foo.py"]) == "foo.py"

    def test_prefers_app_py(self):
        assert _pick_preferred(["server.py", "app.py", "main.py"]) == "app.py"

    def test_prefers_index_js(self):
        assert _pick_preferred(["router.js", "index.js"]) == "index.js"

    def test_falls_back_to_first(self):
        assert _pick_preferred(["foo.py", "bar.py"]) == "foo.py"


# ── Start Server (mocked) ──


class TestStartServer:
    """Test start_server with mocked subprocess and health checks."""

    def test_fails_when_session_not_setup(self):
        session = MagicMock()
        session._setup_complete = False
        detection = FrameworkDetection(framework="flask", entry_file="app.py")

        result = start_server(session, detection)
        assert result.success is False
        assert "not set up" in result.error

    @patch("mycode.server_manager._check_health_tcp", return_value=True)
    @patch("mycode.server_manager._check_health", return_value=False)
    @patch("mycode.server_manager._find_available_port", return_value=9999)
    @patch("mycode.server_manager.subprocess.Popen")
    def test_successful_startup(self, mock_popen, mock_port, mock_health, mock_tcp):
        """Server starts and TCP health check succeeds."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        mock_popen.return_value = proc

        session = MagicMock()
        session._setup_complete = True
        session.venv_dir = Path("/tmp/venv")
        session.venv_python = Path("/tmp/venv/bin/python")
        session.project_copy_dir = Path("/tmp/project")

        detection = FrameworkDetection(
            framework="flask", entry_file="app.py", app_variable="app"
        )

        result = start_server(session, detection, timeout=5)
        assert result.success is True
        assert result.server is not None
        assert result.server.port == 9999
        assert result.server.framework == "flask"

    @patch("mycode.server_manager._check_health_tcp", return_value=False)
    @patch("mycode.server_manager._check_health", return_value=False)
    @patch("mycode.server_manager._find_available_port", return_value=9999)
    @patch("mycode.server_manager.subprocess.Popen")
    def test_server_crashes_on_start(self, mock_popen, mock_port, mock_health, mock_tcp):
        """Server process exits immediately with error."""
        proc = MagicMock()
        proc.poll.return_value = 1  # exited
        proc.communicate.return_value = ("", "ModuleNotFoundError: No module named 'psycopg2'")
        mock_popen.return_value = proc

        session = MagicMock()
        session._setup_complete = True
        session.venv_dir = Path("/tmp/venv")
        session.venv_python = Path("/tmp/venv/bin/python")
        session.project_copy_dir = Path("/tmp/project")

        detection = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )

        result = start_server(session, detection, timeout=5)
        assert result.success is False
        assert "missing dependency" in result.error
        assert "psycopg2" in result.error

    @patch("mycode.server_manager.kill_process_group", return_value=True)
    @patch("mycode.server_manager.time.sleep")
    @patch("mycode.server_manager._check_health_tcp", return_value=False)
    @patch("mycode.server_manager._check_health", return_value=False)
    @patch("mycode.server_manager._find_available_port", return_value=9999)
    @patch("mycode.server_manager.subprocess.Popen")
    @patch("mycode.server_manager.time.monotonic")
    def test_startup_timeout(self, mock_mono, mock_popen, mock_port,
                             mock_health, mock_tcp, mock_sleep, mock_kill):
        """Server never becomes ready — timeout and clean kill."""
        # Simulate time passing past the timeout
        mock_mono.side_effect = [0.0, 0.0, 31.0]

        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.communicate.return_value = ("", "")
        mock_popen.return_value = proc

        session = MagicMock()
        session._setup_complete = True
        session.venv_dir = Path("/tmp/venv")
        session.venv_python = Path("/tmp/venv/bin/python")
        session.project_copy_dir = Path("/tmp/project")

        detection = FrameworkDetection(
            framework="streamlit", entry_file="app.py"
        )

        result = start_server(session, detection, timeout=30)
        assert result.success is False
        assert "did not become ready" in result.error
        mock_kill.assert_called_once()

    @patch("mycode.server_manager.subprocess.Popen")
    @patch("mycode.server_manager._find_available_port", return_value=9999)
    def test_popen_oserror(self, mock_port, mock_popen):
        """OSError from Popen is caught and returned as error."""
        mock_popen.side_effect = OSError("Permission denied")

        session = MagicMock()
        session._setup_complete = True
        session.venv_dir = Path("/tmp/venv")
        session.venv_python = Path("/tmp/venv/bin/python")
        session.project_copy_dir = Path("/tmp/project")

        detection = FrameworkDetection(
            framework="flask", entry_file="app.py", app_variable="app"
        )

        result = start_server(session, detection)
        assert result.success is False
        assert "Permission denied" in result.error
