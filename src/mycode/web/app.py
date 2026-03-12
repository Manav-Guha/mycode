"""FastAPI application for the myCode web backend.

Usage:
    uvicorn mycode.web.app:app --reload --port 8000

All configuration via environment variables — see MYCODE_* vars.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
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
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the web interface. "
        "Install with: pip install 'mycode-ai[web]'"
    )

from mycode.web.jobs import store
from mycode.web.routes import (
    handle_analyze,
    handle_converse,
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


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — start cleanup task, stop on shutdown."""
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


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
    file: UploadFile | None = File(default=None),
):
    """Run preflight diagnostics (stages 1-4.5)."""
    file_obj = None
    filename = ""
    if file is not None:
        content = await file.read()
        file_obj = BytesIO(content)
        filename = file.filename or ""

    # Run in thread pool (sync I/O)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, handle_preflight, github_url, file_obj, filename,
    )
    return JSONResponse(content=_dataclass_to_dict(result))


@app.post("/api/converse")
async def converse(
    job_id: str = Form(default=""),
    turn: int = Form(default=1),
    user_response: str = Form(default=""),
):
    """Handle one turn of the conversational interface."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, handle_converse, job_id, turn, user_response,
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


@app.get("/api/health")
async def health():
    """Server health check."""
    result = handle_health()
    return JSONResponse(content=_dataclass_to_dict(result))
