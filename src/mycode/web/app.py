"""FastAPI application for the myCode web backend.

Usage:
    uvicorn mycode.web.app:app --reload --port 8000

All configuration via environment variables — see MYCODE_* vars.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Check for FastAPI availability before importing
try:
    from fastapi import FastAPI, File, Form, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the web interface. "
        "Install with: pip install 'mycode-ai[web]'"
    )

from mycode.web.jobs import store, MAX_CONCURRENT_JOBS
from mycode.web.analytics import get_admin_stats, log_download, validate_source
from mycode.web.routes import (
    handle_analyze,
    handle_converse,
    handle_download_pdf,
    handle_health,
    handle_preflight,
    handle_report,
    handle_status,
)


# ── JSON Encoder ──


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


# ── Thread Pool ──

# Dedicated thread pool for asyncio.to_thread() calls. The default pool
# has ~14 workers (min(32, cpu+4)) which saturates at 50 concurrent
# requests, causing 15s timeouts. Size this to handle peak HTTP traffic.
_WEB_POOL_SIZE = int(os.environ.get("MYCODE_WEB_POOL_SIZE", "50"))
_web_executor = ThreadPoolExecutor(max_workers=_WEB_POOL_SIZE)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — set executor, start cleanup, stop on shutdown."""
    loop = asyncio.get_event_loop()
    loop.set_default_executor(_web_executor)
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    _web_executor.shutdown(wait=False)


async def _periodic_cleanup():
    """Reap expired jobs every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        try:
            store.cleanup_expired()
        except Exception as exc:
            logger.debug("Cleanup error: %s", exc)


# ── App Setup ──

app = FastAPI(
    title="myCode Web API",
    description="Stress-testing tool for AI-generated code",
    version="0.1.2",
    lifespan=lifespan,
)

# CORS
cors_origins = os.environ.get("MYCODE_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──


@app.post("/api/preflight")
async def preflight(
    github_url: str = Form(default=""),
    source: str = Form(default="public"),
    file: UploadFile | None = File(default=None),
):
    """Run preflight diagnostics (stages 1-4.5)."""
    # Fast validation — return immediately without touching job store or executor
    has_url = bool(github_url and github_url.strip())
    has_file = file is not None and file.size and file.size > 0

    if not has_url and not has_file:
        return JSONResponse(
            content={"error": "Provide either a GitHub URL or upload a zip file."},
        )

    if store.active_count() >= MAX_CONCURRENT_JOBS:
        return JSONResponse(
            content={"error": "Server is at capacity. Please try again shortly."},
        )

    file_obj = None
    filename = ""
    if has_file:
        content = await file.read()
        file_obj = BytesIO(content)
        filename = file.filename or ""

    validated_source = validate_source(source)
    result = await asyncio.to_thread(
        handle_preflight, github_url, file_obj, filename, validated_source,
    )
    return JSONResponse(content=_dataclass_to_dict(result))


@app.post("/api/converse")
async def converse(
    job_id: str = Form(default=""),
    turn: int = Form(default=1),
    user_response: str = Form(default=""),
):
    """Handle one turn of the conversational interface."""
    result = await asyncio.to_thread(
        handle_converse, job_id, turn, user_response,
    )
    return JSONResponse(content=_dataclass_to_dict(result))


@app.post("/api/analyze")
async def analyze(
    job_id: str = Form(default=""),
    offline: bool = Form(default=True),
):
    """Start the full test run (stages 6-9)."""
    result = handle_analyze(job_id, offline)
    return JSONResponse(content=_dataclass_to_dict(result))


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Get current job status with progress."""
    result = handle_status(job_id)
    return JSONResponse(content=_dataclass_to_dict(result))


@app.get("/api/report/{job_id}")
async def report(job_id: str):
    """Get the full diagnostic report."""
    result = handle_report(job_id)
    return JSONResponse(content=_dataclass_to_dict(result))


@app.get("/api/report/{job_id}/understanding.pdf")
async def download_understanding(job_id: str):
    """Download the Understanding Your Results PDF."""
    from fastapi.responses import Response
    pdf_bytes, filename, error = handle_download_pdf(job_id, "understanding")
    if error:
        return JSONResponse(content={"error": error}, status_code=404 if "not found" in error.lower() else 400)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/report/{job_id}/download-log")
async def download_log(job_id: str, type: str = Form(default="")):
    """Log a PDF or JSON download event for analytics."""
    if type in ("pdf", "json"):
        log_download(job_id, type)
    return JSONResponse(content={"ok": True})


@app.get("/api/admin/stats")
async def admin_stats(key: str = Query(default="")):
    """Return aggregate analytics. Requires MYCODE_ADMIN_KEY."""
    admin_key = os.environ.get("MYCODE_ADMIN_KEY", "")
    if not admin_key or key != admin_key:
        return JSONResponse(content={"error": "Forbidden"}, status_code=403)
    stats = await asyncio.to_thread(get_admin_stats)
    return JSONResponse(content=stats)


@app.get("/api/health")
async def health():
    """Server health check."""
    result = handle_health()
    return JSONResponse(content=_dataclass_to_dict(result))
