"""Tests for endpoint_discovery.py — HTTP stress testing Phase 2.

Tests route extraction for FastAPI, Flask, Express; Streamlit page discovery;
request generation; and validation probe classification.
"""

import json
import textwrap
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mycode.endpoint_discovery import (
    DiscoveredEndpoint,
    DiscoveryResult,
    ProbeResult,
    StreamlitPage,
    HttpTestRequest,
    _classify_probe_response,
    _fill_path_params,
    _generate_body,
    discover_endpoints,
    discover_streamlit_pages,
    extract_express_routes,
    extract_fastapi_routes,
    extract_flask_routes,
    generate_requests,
    probe_endpoints,
)
from mycode.ingester import FileAnalysis, ImportInfo, IngestionResult
from mycode.server_manager import FrameworkDetection


# ── Helpers ──


def _make_ingestion(file_analyses=None, project_path="/tmp/proj"):
    return IngestionResult(
        project_path=project_path,
        file_analyses=file_analyses or [],
    )


def _make_file_analysis(file_path, imports=None):
    return FileAnalysis(file_path=file_path, imports=imports or [])


# ═══════════════════════════════════════════════════════════════════════
# FastAPI Route Extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractFastapiRoutes:
    def test_basic_get(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/")
            def root():
                return {"message": "hello"}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].method == "GET"
        assert eps[0].path == "/"
        assert eps[0].handler_name == "root"

    def test_multiple_methods(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/users")
            def list_users():
                return []

            @app.post("/users")
            def create_user(name: str):
                return {"name": name}

            @app.delete("/users/{user_id}")
            def delete_user(user_id: int):
                return {"deleted": user_id}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 3
        methods = {e.method for e in eps}
        assert methods == {"GET", "POST", "DELETE"}

    def test_path_parameters(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/users/{user_id}/posts/{post_id}")
            def get_post(user_id: int, post_id: int):
                return {}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].path_params == ["user_id", "post_id"]

    def test_api_router(self):
        source = textwrap.dedent("""\
            from fastapi import APIRouter
            router = APIRouter()

            @router.get("/items")
            def list_items():
                return []

            @router.post("/items")
            def create_item(name: str):
                return {"name": name}
        """)
        eps = extract_fastapi_routes(source, "routes.py", "app")
        assert len(eps) == 2

    def test_async_handlers(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/async")
            async def async_handler():
                return {"async": True}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].handler_name == "async_handler"

    def test_put_patch(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.put("/items/{id}")
            def update_item(id: int, name: str):
                return {}

            @app.patch("/items/{id}")
            def patch_item(id: int):
                return {}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 2
        assert {e.method for e in eps} == {"PUT", "PATCH"}

    def test_handler_args_exclude_framework_params(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI, Request
            app = FastAPI()

            @app.post("/data")
            def handle(request: Request, name: str, value: int):
                return {}
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert "request" not in eps[0].handler_args
        assert "name" in eps[0].handler_args
        assert "value" in eps[0].handler_args

    def test_syntax_error_returns_empty(self):
        eps = extract_fastapi_routes("def broken(:", "bad.py", "app")
        assert eps == []

    def test_non_http_decorators_ignored(self):
        source = textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.on_event("startup")
            async def startup():
                pass

            @app.middleware("http")
            async def add_header(request, call_next):
                pass
        """)
        eps = extract_fastapi_routes(source, "app.py", "app")
        assert len(eps) == 0


# ═══════════════════════════════════════════════════════════════════════
# Flask Route Extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractFlaskRoutes:
    def test_basic_route(self):
        source = textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)

            @app.route("/")
            def index():
                return "Hello"
        """)
        eps = extract_flask_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].method == "GET"
        assert eps[0].path == "/"
        assert eps[0].handler_name == "index"

    def test_explicit_methods(self):
        source = textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)

            @app.route("/submit", methods=["GET", "POST"])
            def submit():
                return "ok"
        """)
        eps = extract_flask_routes(source, "app.py", "app")
        assert len(eps) == 2
        methods = {e.method for e in eps}
        assert methods == {"GET", "POST"}

    def test_path_parameters(self):
        source = textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)

            @app.route("/users/<int:user_id>")
            def get_user(user_id):
                return str(user_id)
        """)
        eps = extract_flask_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].path_params == ["user_id"]
        assert eps[0].path == "/users/<int:user_id>"

    def test_plain_path_param(self):
        source = textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)

            @app.route("/users/<name>")
            def get_by_name(name):
                return name
        """)
        eps = extract_flask_routes(source, "app.py", "app")
        assert len(eps) == 1
        assert eps[0].path_params == ["name"]

    def test_blueprint(self):
        source = textwrap.dedent("""\
            from flask import Blueprint
            bp = Blueprint("api", __name__)

            @bp.route("/api/items")
            def items():
                return "[]"
        """)
        eps = extract_flask_routes(source, "routes.py", "app")
        assert len(eps) == 1
        assert eps[0].path == "/api/items"

    def test_syntax_error_returns_empty(self):
        eps = extract_flask_routes("def broken(:", "bad.py", "app")
        assert eps == []


# ═══════════════════════════════════════════════════════════════════════
# Express Route Extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractExpressRoutes:
    def test_basic_get(self):
        source = textwrap.dedent("""\
            const express = require('express');
            const app = express();

            app.get('/', (req, res) => {
                res.send('Hello');
            });
        """)
        eps = extract_express_routes(source, "index.js")
        assert len(eps) == 1
        assert eps[0].method == "GET"
        assert eps[0].path == "/"

    def test_multiple_methods(self):
        source = textwrap.dedent("""\
            app.get('/users', handler);
            app.post('/users', handler);
            app.put('/users/:id', handler);
            app.delete('/users/:id', handler);
        """)
        eps = extract_express_routes(source, "app.js")
        assert len(eps) == 4
        methods = {e.method for e in eps}
        assert methods == {"GET", "POST", "PUT", "DELETE"}

    def test_path_parameters(self):
        source = """app.get('/users/:userId/posts/:postId', handler);"""
        eps = extract_express_routes(source, "app.js")
        assert len(eps) == 1
        assert eps[0].path_params == ["userId", "postId"]

    def test_router(self):
        source = textwrap.dedent("""\
            const router = require('express').Router();
            router.get('/items', listItems);
            router.post('/items', createItem);
        """)
        eps = extract_express_routes(source, "routes.js")
        assert len(eps) == 2

    def test_double_quoted_paths(self):
        source = 'app.get("/api/health", handler);'
        eps = extract_express_routes(source, "app.js")
        assert len(eps) == 1
        assert eps[0].path == "/api/health"

    def test_patch_method(self):
        source = 'app.patch("/items/:id", handler);'
        eps = extract_express_routes(source, "app.js")
        assert len(eps) == 1
        assert eps[0].method == "PATCH"


# ═══════════════════════════════════════════════════════════════════════
# Streamlit Page Discovery
# ═══════════════════════════════════════════════════════════════════════


class TestDiscoverStreamlitPages:
    def test_root_page_always_included(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\nst.title('Hi')\n")
        pages = discover_streamlit_pages(tmp_path, "app.py")
        assert len(pages) >= 1
        assert pages[0].path == "/"

    def test_pages_directory(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\n")
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        (pages_dir / "1_Dashboard.py").write_text("import streamlit as st\n")
        (pages_dir / "2_Settings.py").write_text("import streamlit as st\n")
        (pages_dir / "__init__.py").write_text("")  # should be skipped

        pages = discover_streamlit_pages(tmp_path, "app.py")
        assert len(pages) == 3  # root + 2 pages
        paths = {p.path for p in pages}
        assert "/" in paths
        assert "/Dashboard" in paths
        assert "/Settings" in paths

    def test_pages_without_number_prefix(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\n")
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir()
        (pages_dir / "about.py").write_text("import streamlit as st\n")

        pages = discover_streamlit_pages(tmp_path, "app.py")
        assert len(pages) == 2
        assert pages[1].path == "/about"

    def test_st_page_navigation(self, tmp_path):
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            import streamlit as st

            pg = st.navigation([
                st.Page("home.py", title="Home", url_path="/"),
                st.Page("analytics.py", title="Analytics", url_path="/analytics"),
            ])
            pg.run()
        """))
        pages = discover_streamlit_pages(tmp_path, "app.py")
        paths = {p.path for p in pages}
        assert "/analytics" in paths

    def test_no_pages_dir(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\nst.write('hi')\n")
        pages = discover_streamlit_pages(tmp_path, "app.py")
        assert len(pages) == 1
        assert pages[0].path == "/"


# ═══════════════════════════════════════════════════════════════════════
# Unified Endpoint Discovery
# ═══════════════════════════════════════════════════════════════════════


class TestDiscoverEndpoints:
    def test_fastapi_discovery(self, tmp_path):
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/")
            def root():
                return {"msg": "hi"}

            @app.get("/items")
            def items():
                return []
        """))
        fa = _make_file_analysis("app.py")
        ingestion = _make_ingestion([fa], str(tmp_path))
        detection = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )

        result = discover_endpoints(detection, tmp_path, ingestion)
        assert result.framework == "fastapi"
        assert len(result.endpoints) == 2

    def test_flask_discovery(self, tmp_path):
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            from flask import Flask
            app = Flask(__name__)

            @app.route("/")
            def index():
                return "Hello"
        """))
        fa = _make_file_analysis("app.py")
        ingestion = _make_ingestion([fa], str(tmp_path))
        detection = FrameworkDetection(
            framework="flask", entry_file="app.py", app_variable="app",
        )

        result = discover_endpoints(detection, tmp_path, ingestion)
        assert result.framework == "flask"
        assert len(result.endpoints) == 1

    def test_express_discovery(self, tmp_path):
        (tmp_path / "index.js").write_text(textwrap.dedent("""\
            const express = require('express');
            const app = express();
            app.get('/', (req, res) => res.send('hi'));
            app.post('/data', (req, res) => res.json(req.body));
            app.listen(3000);
        """))
        fa = _make_file_analysis("index.js")
        ingestion = _make_ingestion([fa], str(tmp_path))
        detection = FrameworkDetection(
            framework="express", entry_file="index.js",
        )

        result = discover_endpoints(detection, tmp_path, ingestion)
        assert result.framework == "express"
        assert len(result.endpoints) == 2

    def test_streamlit_discovery(self, tmp_path):
        (tmp_path / "app.py").write_text("import streamlit as st\n")
        detection = FrameworkDetection(
            framework="streamlit", entry_file="app.py",
        )

        result = discover_endpoints(detection, tmp_path)
        assert result.framework == "streamlit"
        assert len(result.streamlit_pages) >= 1

    def test_no_routes_falls_back_to_root(self, tmp_path):
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()
            # No routes defined
        """))
        fa = _make_file_analysis("app.py")
        ingestion = _make_ingestion([fa], str(tmp_path))
        detection = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )

        result = discover_endpoints(detection, tmp_path, ingestion)
        assert len(result.endpoints) == 1
        assert result.endpoints[0].path == "/"

    def test_multi_file_scanning(self, tmp_path):
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            from fastapi import FastAPI
            app = FastAPI()

            @app.get("/")
            def root():
                return {}
        """))
        (tmp_path / "routes.py").write_text(textwrap.dedent("""\
            from fastapi import APIRouter
            router = APIRouter()

            @router.get("/extra")
            def extra():
                return {}
        """))
        fas = [_make_file_analysis("app.py"), _make_file_analysis("routes.py")]
        ingestion = _make_ingestion(fas, str(tmp_path))
        detection = FrameworkDetection(
            framework="fastapi", entry_file="app.py",
            app_variable="app", module_name="app",
        )

        result = discover_endpoints(detection, tmp_path, ingestion)
        assert len(result.endpoints) == 2


# ═══════════════════════════════════════════════════════════════════════
# Request Generation
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateRequests:
    def test_get_request(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(method="GET", path="/users", handler_name="list_users"),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert len(reqs) == 1
        assert reqs[0].method == "GET"
        assert reqs[0].url == "http://localhost:8000/users"
        assert reqs[0].body is None

    def test_post_request_has_body(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(
                    method="POST", path="/users",
                    handler_name="create_user",
                    handler_args=["name", "email"],
                ),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert len(reqs) == 1
        assert reqs[0].method == "POST"
        assert reqs[0].headers.get("Content-Type") == "application/json"
        body = json.loads(reqs[0].body)
        assert "name" in body
        assert "email" in body

    def test_path_params_filled(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(
                    method="GET", path="/users/{user_id}",
                    path_params=["user_id"],
                ),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert reqs[0].url == "http://localhost:8000/users/1"

    def test_streamlit_page_requests(self):
        discovery = DiscoveryResult(
            framework="streamlit",
            streamlit_pages=[
                StreamlitPage(path="/", file_path="app.py", title="Main"),
                StreamlitPage(path="/Dashboard", file_path="pages/1_Dashboard.py", title="Dashboard"),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8501")
        assert len(reqs) == 2
        assert all(r.method == "GET" for r in reqs)
        urls = {r.url for r in reqs}
        assert "http://localhost:8501/" in urls
        assert "http://localhost:8501/Dashboard" in urls

    def test_base_url_trailing_slash_stripped(self):
        discovery = DiscoveryResult(
            framework="flask",
            endpoints=[
                DiscoveredEndpoint(method="GET", path="/health"),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:5000/")
        assert reqs[0].url == "http://localhost:5000/health"

    def test_description_includes_handler_name(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(
                    method="POST", path="/items",
                    handler_name="create_item",
                ),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert "create_item" in reqs[0].description

    def test_delete_request_no_body(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(method="DELETE", path="/items/{id}", path_params=["id"]),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert reqs[0].body is None

    def test_put_request_has_body(self):
        discovery = DiscoveryResult(
            framework="fastapi",
            endpoints=[
                DiscoveredEndpoint(
                    method="PUT", path="/items/{id}",
                    handler_args=["id", "name", "price"],
                    path_params=["id"],
                ),
            ],
        )
        reqs = generate_requests(discovery, "http://localhost:8000")
        assert reqs[0].body is not None
        body = json.loads(reqs[0].body)
        assert "name" in body
        assert "price" in body
        # path param 'id' should not be in body
        assert "id" not in body


# ═══════════════════════════════════════════════════════════════════════
# Path Parameter Filling
# ═══════════════════════════════════════════════════════════════════════


class TestFillPathParams:
    def test_fastapi_style(self):
        assert _fill_path_params("/users/{user_id}", ["user_id"]) == "/users/1"

    def test_express_style(self):
        assert _fill_path_params("/users/:id", ["id"]) == "/users/1"

    def test_flask_style(self):
        assert _fill_path_params("/users/<int:user_id>", ["user_id"]) == "/users/1"

    def test_flask_plain(self):
        assert _fill_path_params("/users/<name>", ["name"]) == "/users/test"

    def test_multiple_params(self):
        result = _fill_path_params(
            "/users/{user_id}/posts/{post_id}", ["user_id", "post_id"]
        )
        assert result == "/users/1/posts/1"

    def test_unknown_param_defaults_to_1(self):
        result = _fill_path_params("/items/{xyz}", ["xyz"])
        assert result == "/items/1"

    def test_uuid_param(self):
        result = _fill_path_params("/items/{uuid}", ["uuid"])
        assert "00000000" in result

    def test_slug_param(self):
        result = _fill_path_params("/posts/{slug}", ["slug"])
        assert result == "/posts/test"


# ═══════════════════════════════════════════════════════════════════════
# Request Body Generation
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateBody:
    def test_default_body_when_no_args(self):
        ep = DiscoveredEndpoint(method="POST", path="/data")
        body = _generate_body(ep)
        assert "data" in body
        assert "id" in body

    def test_body_from_handler_args(self):
        ep = DiscoveredEndpoint(
            method="POST", path="/users",
            handler_args=["name", "email", "password"],
        )
        body = _generate_body(ep)
        assert body["name"] == "test"
        assert "example.com" in body["email"]
        assert "password" in body["password"]

    def test_skips_path_params_in_body(self):
        ep = DiscoveredEndpoint(
            method="PUT", path="/items/{id}",
            handler_args=["id", "title", "price"],
            path_params=["id"],
        )
        body = _generate_body(ep)
        assert "id" not in body
        assert "title" in body

    def test_numeric_fields(self):
        ep = DiscoveredEndpoint(
            method="POST", path="/order",
            handler_args=["item_id", "quantity", "amount"],
        )
        body = _generate_body(ep)
        assert isinstance(body["item_id"], int)
        assert isinstance(body["quantity"], int)
        assert isinstance(body["amount"], float)

    def test_boolean_fields(self):
        ep = DiscoveredEndpoint(
            method="POST", path="/settings",
            handler_args=["is_active", "enabled"],
        )
        body = _generate_body(ep)
        assert body["is_active"] is True
        assert body["enabled"] is True

    def test_list_fields(self):
        ep = DiscoveredEndpoint(
            method="POST", path="/data",
            handler_args=["tags", "items"],
        )
        body = _generate_body(ep)
        assert isinstance(body["tags"], list)
        assert isinstance(body["items"], list)


# ═══════════════════════════════════════════════════════════════════════
# Validation Probe Classification
# ═══════════════════════════════════════════════════════════════════════


class TestClassifyProbeResponse:
    def _ep(self):
        return DiscoveredEndpoint(method="GET", path="/test")

    def test_200_is_testable(self):
        result = _classify_probe_response(self._ep(), 200, 50.0)
        assert result.testable is True
        assert result.status_code == 200

    def test_201_is_testable(self):
        result = _classify_probe_response(self._ep(), 201, 30.0)
        assert result.testable is True

    def test_301_redirect_is_testable(self):
        result = _classify_probe_response(self._ep(), 301, 20.0)
        assert result.testable is True

    def test_401_requires_auth(self):
        result = _classify_probe_response(self._ep(), 401, 10.0)
        assert result.testable is False
        assert "authentication" in result.skip_reason

    def test_403_requires_auth(self):
        result = _classify_probe_response(self._ep(), 403, 10.0)
        assert result.testable is False
        assert "authentication" in result.skip_reason

    def test_404_not_reachable(self):
        result = _classify_probe_response(self._ep(), 404, 10.0)
        assert result.testable is False
        assert "not reachable" in result.skip_reason

    def test_500_is_finding(self):
        result = _classify_probe_response(self._ep(), 500, 100.0)
        assert result.testable is False
        assert result.is_finding is True
        assert "server error" in result.skip_reason

    def test_502_is_finding(self):
        result = _classify_probe_response(self._ep(), 502, 50.0)
        assert result.is_finding is True

    def test_422_is_skipped(self):
        result = _classify_probe_response(self._ep(), 422, 10.0)
        assert result.testable is False
        assert "422" in result.skip_reason

    def test_response_time_recorded(self):
        result = _classify_probe_response(self._ep(), 200, 123.4)
        assert result.response_time_ms == 123.4


# ═══════════════════════════════════════════════════════════════════════
# Probe Endpoints (mocked HTTP)
# ═══════════════════════════════════════════════════════════════════════


class TestProbeEndpoints:
    @patch("mycode.endpoint_discovery.urllib.request.urlopen")
    def test_probe_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        reqs = [HttpTestRequest(method="GET", url="http://localhost:8000/", description="GET /")]
        results = probe_endpoints(reqs)
        assert len(results) == 1
        assert results[0].testable is True
        assert results[0].status_code == 200

    @patch("mycode.endpoint_discovery.urllib.request.urlopen")
    def test_probe_401(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:8000/admin", 401, "Unauthorized", {}, None
        )

        reqs = [HttpTestRequest(method="GET", url="http://localhost:8000/admin", description="GET /admin")]
        results = probe_endpoints(reqs)
        assert len(results) == 1
        assert results[0].testable is False
        assert "authentication" in results[0].skip_reason

    @patch("mycode.endpoint_discovery.urllib.request.urlopen")
    def test_probe_500(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:8000/broken", 500, "ISE", {}, None
        )

        reqs = [HttpTestRequest(method="GET", url="http://localhost:8000/broken", description="GET /broken")]
        results = probe_endpoints(reqs)
        assert len(results) == 1
        assert results[0].is_finding is True

    @patch("mycode.endpoint_discovery.urllib.request.urlopen")
    def test_probe_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")

        reqs = [HttpTestRequest(method="GET", url="http://localhost:8000/slow", description="GET /slow")]
        results = probe_endpoints(reqs)
        assert len(results) == 1
        assert results[0].testable is False
        assert "unresponsive" in results[0].skip_reason

    @patch("mycode.endpoint_discovery.urllib.request.urlopen")
    def test_probe_multiple_endpoints(self, mock_urlopen):
        """Probe classifies each endpoint independently."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: 200
                mock_resp = MagicMock()
                mock_resp.status = 200
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                return mock_resp
            else:
                # Second call: 404
                raise urllib.error.HTTPError("url", 404, "Not Found", {}, None)

        mock_urlopen.side_effect = side_effect

        reqs = [
            HttpTestRequest(method="GET", url="http://localhost:8000/", description="root"),
            HttpTestRequest(method="GET", url="http://localhost:8000/missing", description="missing"),
        ]
        results = probe_endpoints(reqs)
        assert results[0].testable is True
        assert results[1].testable is False
        assert "not reachable" in results[1].skip_reason
