"""Endpoint Discovery & Request Generation — HTTP stress testing Phase 2.

Discovers testable endpoints from FastAPI, Flask, Express, and Streamlit
projects by parsing route decorators (AST for Python, regex for JS).
Generates synthetic test requests and validates them against a running server
before load testing.

Pure Python.  No LLM dependency.
"""

import ast
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.ingester import IngestionResult, _read_text_safe
from mycode.server_manager import FrameworkDetection, ServerInfo

logger = logging.getLogger(__name__)

# ── Constants ──

# Validation probe timeout per endpoint
_PROBE_TIMEOUT_SECONDS = 10

# HTTP methods we care about
_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})

# Synthetic values for path parameters
_SYNTHETIC_PATH_VALUES: dict[str, str] = {
    "id": "1",
    "user_id": "1",
    "item_id": "1",
    "post_id": "1",
    "order_id": "1",
    "product_id": "1",
    "slug": "test",
    "name": "test",
    "username": "test_user",
    "token": "test_token",
    "uuid": "00000000-0000-0000-0000-000000000001",
}

# Default synthetic request body
_DEFAULT_POST_BODY = {"data": "test", "id": 1, "name": "test_user"}


# ── Data Classes ──


@dataclass
class DiscoveredEndpoint:
    """A single endpoint discovered from source analysis."""

    method: str  # "GET", "POST", etc.
    path: str  # "/users/{id}" or "/api/items"
    handler_name: str = ""  # function name handling this route
    handler_args: list[str] = field(default_factory=list)  # function params
    file_path: str = ""  # source file where the route is defined
    path_params: list[str] = field(default_factory=list)  # ["id", "name"]


@dataclass
class StreamlitPage:
    """A Streamlit page discovered from multi-page structure."""

    path: str  # URL path: "/" or "/page_name"
    file_path: str  # source file
    title: str = ""  # page title if detectable


@dataclass
class HttpTestRequest:
    """A fully-formed test request ready for Phase 3's load driver."""

    method: str  # "GET", "POST", etc.
    url: str  # full URL with host:port and path params filled
    headers: dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None  # JSON string or None
    description: str = ""  # human-readable: "POST /users — create user"


@dataclass
class ProbeResult:
    """Result of validating a single endpoint at zero load."""

    endpoint: DiscoveredEndpoint
    status_code: int = 0
    testable: bool = False
    skip_reason: str = ""  # "requires authentication", "route not reachable", etc.
    is_finding: bool = False  # True if 500 at baseline (finding even before load)
    response_time_ms: float = 0.0


@dataclass
class DiscoveryResult:
    """Complete result of endpoint discovery for a project."""

    framework: str
    endpoints: list[DiscoveredEndpoint] = field(default_factory=list)
    streamlit_pages: list[StreamlitPage] = field(default_factory=list)
    requests: list[HttpTestRequest] = field(default_factory=list)
    probe_results: list[ProbeResult] = field(default_factory=list)


# ── FastAPI Route Extraction (AST) ──


def extract_fastapi_routes(
    source: str, file_path: str = "", app_var: str = "app"
) -> list[DiscoveredEndpoint]:
    """Extract routes from FastAPI source using AST parsing.

    Handles:
      - ``@app.get("/path")``, ``@app.post("/path")``, etc.
      - ``@router.get("/path")`` (APIRouter)
      - Path parameters: ``/users/{user_id}``
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Detect router variable names: router = APIRouter()
    router_vars = {app_var}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func_name = _ast_call_name(node.value)
            if func_name in ("APIRouter", "fastapi.APIRouter"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        router_vars.add(target.id)

    endpoints: list[DiscoveredEndpoint] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            ep = _parse_fastapi_decorator(dec, router_vars, node, file_path)
            if ep:
                endpoints.append(ep)

    return endpoints


def _parse_fastapi_decorator(
    dec: ast.expr,
    router_vars: set[str],
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
) -> Optional[DiscoveredEndpoint]:
    """Parse a single FastAPI route decorator."""
    if not isinstance(dec, ast.Call):
        return None

    # e.g. app.get, router.post
    if not isinstance(dec.func, ast.Attribute):
        return None

    method_name = dec.func.attr.upper()
    if method_name not in _HTTP_METHODS:
        return None

    # Check that the object is a known router/app variable
    obj_name = ""
    if isinstance(dec.func.value, ast.Name):
        obj_name = dec.func.value.id
    if obj_name not in router_vars:
        return None

    # Extract path from first positional argument
    path = "/"
    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
        path = dec.args[0].value

    # Extract handler args (skip self/cls and framework-injected)
    handler_args = [
        arg.arg for arg in func_node.args.args
        if arg.arg not in ("self", "cls", "request", "response", "db", "session")
    ]

    # Extract path parameters from {param} syntax
    path_params = re.findall(r"\{(\w+)\}", path)

    return DiscoveredEndpoint(
        method=method_name,
        path=path,
        handler_name=func_node.name,
        handler_args=handler_args,
        file_path=file_path,
        path_params=path_params,
    )


# ── Flask Route Extraction (AST) ──


def extract_flask_routes(
    source: str, file_path: str = "", app_var: str = "app"
) -> list[DiscoveredEndpoint]:
    """Extract routes from Flask source using AST parsing.

    Handles:
      - ``@app.route("/path", methods=["GET", "POST"])``
      - ``@blueprint.route("/path")``
      - Path parameters: ``/users/<int:user_id>`` or ``/users/<name>``
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Detect blueprint variable names
    blueprint_vars = {app_var}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func_name = _ast_call_name(node.value)
            if func_name in ("Blueprint", "flask.Blueprint"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        blueprint_vars.add(target.id)

    endpoints: list[DiscoveredEndpoint] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            eps = _parse_flask_decorator(dec, blueprint_vars, node, file_path)
            endpoints.extend(eps)

    return endpoints


def _parse_flask_decorator(
    dec: ast.expr,
    app_vars: set[str],
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
) -> list[DiscoveredEndpoint]:
    """Parse a single Flask route decorator.  May return multiple endpoints
    if methods=["GET", "POST"] is specified."""
    if not isinstance(dec, ast.Call):
        return []

    if not isinstance(dec.func, ast.Attribute):
        return []

    if dec.func.attr != "route":
        return []

    obj_name = ""
    if isinstance(dec.func.value, ast.Name):
        obj_name = dec.func.value.id
    if obj_name not in app_vars:
        return []

    # Extract path
    path = "/"
    if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
        path = dec.args[0].value

    # Extract methods from keyword argument
    methods = ["GET"]
    for kw in dec.keywords:
        if kw.arg == "methods" and isinstance(kw.value, ast.List):
            methods = []
            for elt in kw.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    methods.append(elt.value.upper())

    # Handler args
    handler_args = [
        arg.arg for arg in func_node.args.args
        if arg.arg not in ("self", "cls")
    ]

    # Path parameters: Flask uses <type:name> or <name>
    path_params = re.findall(r"<(?:\w+:)?(\w+)>", path)

    results = []
    for method in methods:
        if method in _HTTP_METHODS:
            results.append(DiscoveredEndpoint(
                method=method,
                path=path,
                handler_name=func_node.name,
                handler_args=handler_args,
                file_path=file_path,
                path_params=path_params,
            ))
    return results


# ── Express Route Extraction (Regex) ──


def extract_express_routes(
    source: str, file_path: str = ""
) -> list[DiscoveredEndpoint]:
    """Extract routes from Express.js source using regex.

    Handles:
      - ``app.get("/path", handler)``
      - ``app.post("/path", middleware, handler)``
      - ``router.get("/path", handler)``
      - Path parameters: ``/users/:id``
    """
    endpoints: list[DiscoveredEndpoint] = []

    # Match: app.get('/path' or router.post("/path"
    # Captures: (variable, method, path)
    pattern = re.compile(
        r"""(?:^|[;\s])(\w+)\.(get|post|put|patch|delete)\s*\(\s*"""
        r"""['"]([^'"]+)['"]""",
        re.IGNORECASE | re.MULTILINE,
    )

    for match in pattern.finditer(source):
        var_name = match.group(1)
        method = match.group(2).upper()
        path = match.group(3)

        if method not in _HTTP_METHODS:
            continue

        # Extract :param style path parameters
        path_params = re.findall(r":(\w+)", path)

        endpoints.append(DiscoveredEndpoint(
            method=method,
            path=path,
            handler_name="",
            file_path=file_path,
            path_params=path_params,
        ))

    return endpoints


# ── Streamlit Page Discovery ──


def discover_streamlit_pages(
    project_dir: Path,
    entry_file: str,
) -> list[StreamlitPage]:
    """Discover Streamlit pages from multi-page structure.

    Checks for:
      1. ``pages/`` directory (Streamlit convention)
      2. ``st.navigation`` calls in the entry file
      3. Falls back to just the root page
    """
    pages: list[StreamlitPage] = []

    # Always include the root page
    pages.append(StreamlitPage(
        path="/",
        file_path=entry_file,
        title="Main Page",
    ))

    # Check for pages/ directory (Streamlit multi-page convention)
    pages_dir = project_dir / "pages"
    if pages_dir.is_dir():
        for page_file in sorted(pages_dir.iterdir()):
            if page_file.suffix == ".py" and not page_file.name.startswith("_"):
                # Streamlit convention: filename → URL path
                # e.g. pages/1_Dashboard.py → /Dashboard
                page_name = page_file.stem
                # Strip leading number prefix: "1_Dashboard" → "Dashboard"
                clean_name = re.sub(r"^\d+[_\s]*", "", page_name)
                if clean_name:
                    pages.append(StreamlitPage(
                        path=f"/{clean_name}",
                        file_path=f"pages/{page_file.name}",
                        title=clean_name.replace("_", " "),
                    ))

    # Check entry file for st.navigation or st.Page references
    entry_path = project_dir / entry_file
    if entry_path.is_file():
        try:
            source = _read_text_safe(entry_path)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = _ast_call_name(node)
                if call_name in ("st.Page", "streamlit.Page"):
                    # st.Page("page_file.py", title="...", url_path="...")
                    _page = _parse_st_page_call(node, entry_file)
                    if _page and _page.path != "/" and not any(
                        p.path == _page.path for p in pages
                    ):
                        pages.append(_page)
        except (OSError, SyntaxError):
            pass

    return pages


def _parse_st_page_call(
    node: ast.Call, entry_file: str
) -> Optional[StreamlitPage]:
    """Extract page info from an ``st.Page(...)`` call."""
    file_path = ""
    title = ""
    url_path = ""

    # First positional arg is the page file or callable
    if node.args and isinstance(node.args[0], ast.Constant):
        val = node.args[0].value
        if isinstance(val, str):
            file_path = val

    for kw in node.keywords:
        if kw.arg == "title" and isinstance(kw.value, ast.Constant):
            title = str(kw.value.value)
        elif kw.arg == "url_path" and isinstance(kw.value, ast.Constant):
            url_path = str(kw.value.value)

    if not url_path and file_path:
        # Derive from filename
        stem = Path(file_path).stem
        url_path = f"/{stem}"

    if url_path:
        return StreamlitPage(
            path=url_path,
            file_path=file_path or entry_file,
            title=title or url_path.strip("/").replace("_", " "),
        )
    return None


# ── Unified Endpoint Discovery ──


def discover_endpoints(
    detection: FrameworkDetection,
    project_dir: Path,
    ingestion: Optional[IngestionResult] = None,
) -> DiscoveryResult:
    """Discover all testable endpoints for the detected framework.

    Scans source files for route definitions and returns a DiscoveryResult
    with all discovered endpoints or Streamlit pages.

    Args:
        detection: Framework detection result from Phase 1.
        project_dir: Absolute path to the project (or project copy).
        ingestion: Optional ingestion result for multi-file scanning.
    """
    framework = detection.framework

    if framework == "streamlit":
        pages = discover_streamlit_pages(project_dir, detection.entry_file)
        return DiscoveryResult(framework=framework, streamlit_pages=pages)

    # For API frameworks, scan all source files
    all_endpoints: list[DiscoveredEndpoint] = []

    files_to_scan: list[str] = []
    if ingestion:
        files_to_scan = [fa.file_path for fa in ingestion.file_analyses]
    else:
        # Fallback: just scan the entry file
        files_to_scan = [detection.entry_file]

    for rel_path in files_to_scan:
        abs_path = project_dir / rel_path
        if not abs_path.is_file():
            continue
        try:
            source = _read_text_safe(abs_path)
        except OSError:
            continue

        if framework == "fastapi":
            eps = extract_fastapi_routes(
                source, rel_path, detection.app_variable or "app"
            )
            all_endpoints.extend(eps)
        elif framework == "flask":
            eps = extract_flask_routes(
                source, rel_path, detection.app_variable or "app"
            )
            all_endpoints.extend(eps)
        elif framework == "express":
            eps = extract_express_routes(source, rel_path)
            all_endpoints.extend(eps)

    # If no endpoints found, fall back to root path
    if not all_endpoints:
        all_endpoints.append(DiscoveredEndpoint(
            method="GET",
            path="/",
            handler_name="root",
            file_path=detection.entry_file,
        ))

    return DiscoveryResult(framework=framework, endpoints=all_endpoints)


# ── Request Generation ──


def generate_requests(
    discovery: DiscoveryResult,
    base_url: str,
) -> list[HttpTestRequest]:
    """Generate test requests from discovered endpoints or Streamlit pages.

    For each endpoint: fills path parameters with synthetic values, generates
    POST bodies where needed, and builds fully-formed HttpTestRequest objects.

    Args:
        discovery: Result from discover_endpoints().
        base_url: e.g. "http://localhost:8501"
    """
    base = base_url.rstrip("/")
    requests: list[HttpTestRequest] = []

    if discovery.framework == "streamlit":
        for page in discovery.streamlit_pages:
            requests.append(HttpTestRequest(
                method="GET",
                url=f"{base}{page.path}",
                description=f"GET {page.path} — {page.title or 'page load'}",
            ))
        return requests

    for ep in discovery.endpoints:
        url_path = _fill_path_params(ep.path, ep.path_params)
        url = f"{base}{url_path}"

        headers: dict[str, str] = {}
        body: Optional[str] = None

        if ep.method in ("POST", "PUT", "PATCH"):
            headers["Content-Type"] = "application/json"
            body = json.dumps(_generate_body(ep))

        # Build human-readable description
        desc_parts = [f"{ep.method} {ep.path}"]
        if ep.handler_name:
            desc_parts.append(f"— {ep.handler_name}")
        description = " ".join(desc_parts)

        requests.append(HttpTestRequest(
            method=ep.method,
            url=url,
            headers=headers,
            body=body,
            description=description,
        ))

    return requests


def _fill_path_params(path: str, params: list[str]) -> str:
    """Replace path parameter placeholders with synthetic values.

    Handles both ``{param}`` (FastAPI) and ``:param`` (Express) styles.
    Flask ``<type:param>`` is already stripped to ``<param>`` during extraction,
    but we handle the raw form too for safety.
    """
    result = path
    for param in params:
        value = _SYNTHETIC_PATH_VALUES.get(param, "1")
        # FastAPI style: {param}
        result = result.replace(f"{{{param}}}", value)
        # Express style: :param
        result = re.sub(rf":({re.escape(param)})(?=/|$)", value, result)
        # Flask style: <type:param> or <param>
        result = re.sub(rf"<(?:\w+:)?{re.escape(param)}>", value, result)
    return result


def _generate_body(ep: DiscoveredEndpoint) -> dict:
    """Generate a synthetic JSON body based on handler arguments and param names."""
    # Try to infer fields from handler args that aren't path params or framework params
    skip_args = {"self", "cls", "request", "response", "db", "session"}
    skip_args.update(ep.path_params)

    body_args = [a for a in ep.handler_args if a not in skip_args]

    if not body_args:
        return dict(_DEFAULT_POST_BODY)

    body: dict = {}
    for arg in body_args:
        arg_lower = arg.lower()
        if "id" in arg_lower:
            body[arg] = 1
        elif "name" in arg_lower or "title" in arg_lower:
            body[arg] = "test"
        elif "email" in arg_lower:
            body[arg] = "test@example.com"
        elif "password" in arg_lower:
            body[arg] = "test_password_123"
        elif "price" in arg_lower or "amount" in arg_lower:
            body[arg] = 9.99
        elif "count" in arg_lower or "quantity" in arg_lower:
            body[arg] = 1
        elif "active" in arg_lower or "enabled" in arg_lower or "is_" in arg_lower:
            body[arg] = True
        elif "url" in arg_lower or "link" in arg_lower:
            body[arg] = "https://example.com"
        elif "date" in arg_lower:
            body[arg] = "2026-01-01"
        elif "description" in arg_lower or "text" in arg_lower or "content" in arg_lower:
            body[arg] = "test content"
        elif "items" in arg_lower or "list" in arg_lower or "tags" in arg_lower:
            body[arg] = ["test"]
        else:
            body[arg] = "test"
    return body


# ── Validation Probe ──


def probe_endpoints(
    requests: list[HttpTestRequest],
    timeout: int = _PROBE_TIMEOUT_SECONDS,
) -> list[ProbeResult]:
    """Send one request per endpoint at zero load to validate reachability.

    Classifies each endpoint:
      - 2xx/3xx → testable
      - 401/403 → requires authentication (skipped)
      - 404 → route not reachable (skipped)
      - 500 → server error at baseline (finding)
      - Timeout/error → unresponsive (skipped)
    """
    results: list[ProbeResult] = []

    for req in requests:
        result = _probe_single(req, timeout)
        results.append(result)

    return results


def _probe_single(req: HttpTestRequest, timeout: int) -> ProbeResult:
    """Probe a single endpoint and classify the response."""
    import time

    # Build a DiscoveredEndpoint from the request for the result
    ep = DiscoveredEndpoint(
        method=req.method,
        path=req.url.split("//", 1)[-1].split("/", 1)[-1] if "//" in req.url else req.url,
        handler_name=req.description,
    )
    # Ensure path starts with /
    if not ep.path.startswith("/"):
        ep.path = "/" + ep.path

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
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except (urllib.error.URLError, OSError, TimeoutError):
        elapsed = (time.monotonic() - start) * 1000
        return ProbeResult(
            endpoint=ep,
            status_code=0,
            testable=False,
            skip_reason="endpoint unresponsive at baseline — skipped",
            response_time_ms=elapsed,
        )

    elapsed = (time.monotonic() - start) * 1000

    return _classify_probe_response(ep, status, elapsed)


def _classify_probe_response(
    ep: DiscoveredEndpoint, status: int, elapsed_ms: float
) -> ProbeResult:
    """Classify a probe response by status code."""
    if 200 <= status < 400:
        return ProbeResult(
            endpoint=ep,
            status_code=status,
            testable=True,
            response_time_ms=elapsed_ms,
        )

    if status in (401, 403):
        return ProbeResult(
            endpoint=ep,
            status_code=status,
            testable=False,
            skip_reason="requires authentication — skipped",
            response_time_ms=elapsed_ms,
        )

    if status == 404:
        return ProbeResult(
            endpoint=ep,
            status_code=status,
            testable=False,
            skip_reason="route not reachable — skipped",
            response_time_ms=elapsed_ms,
        )

    if status >= 500:
        return ProbeResult(
            endpoint=ep,
            status_code=status,
            testable=False,
            skip_reason="server error at baseline",
            is_finding=True,
            response_time_ms=elapsed_ms,
        )

    # Other 4xx — treat as skipped
    return ProbeResult(
        endpoint=ep,
        status_code=status,
        testable=False,
        skip_reason=f"unexpected status {status} — skipped",
        response_time_ms=elapsed_ms,
    )


# ── AST Helpers ──


def _ast_call_name(node: ast.Call) -> str:
    """Get the full dotted name of a Call node's function."""
    return _ast_name(node.func)


def _ast_name(node: ast.expr) -> str:
    """Get the dotted name from a Name or Attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _ast_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    return ""
