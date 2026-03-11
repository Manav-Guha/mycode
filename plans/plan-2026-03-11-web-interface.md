# Web Interface — Implementation Plan

## Problem

myCode is a CLI tool. The product vision is a web interface where any user can paste a GitHub URL or upload a zip, answer a few questions, and get a diagnostic report. The CLI remains for power users, but the web interface is the primary product surface.

---

## Architecture Overview

Two separate deployable units:

```
┌──────────────────────┐         ┌──────────────────────────────┐
│   FRONTEND (static)  │  HTTP   │   BACKEND (FastAPI + Docker) │
│   Single-page app    │ ──────► │   POST /api/analyze          │
│   Vanilla JS or      │         │   POST /api/preflight        │
│   lightweight frame  │         │   POST /api/converse          │
│                      │ ◄────── │   GET  /api/status/{job_id}  │
│   Configurable       │  JSON   │   GET  /api/report/{job_id}  │
│   backend URL        │         │                              │
└──────────────────────┘         │   myCode engine as library   │
                                 │   Docker for execution       │
                                 └──────────────────────────────┘
```

**No hardcoded hosting references.** Frontend reads `MYCODE_API_URL` from a config file or environment injection at build time. Backend reads all config from environment variables.

---

## User Flow

```
1. User lands on page
2. Pastes GitHub URL  ─or─  uploads .zip
3. Backend runs preflight:
   a. Clone/extract project
   b. Language detection
   c. Session setup (venv/node_modules in Docker)
   d. Ingestion (parse code, detect deps)
   e. Library matching
   f. Viability gate
4. Frontend shows preflight results:
   - Language detected, dependencies found
   - Viability status (install rate, import rate)
   - If viability fails: show diagnostic, stop here
5. Conversational interface (2 turns):
   a. Backend sends Turn 1 question (project analysis + inference)
   b. User answers
   c. Backend sends Turn 2 question (constraints)
   d. User answers
   e. Backend returns parsed constraints for confirmation
6. User clicks "Run Tests"
7. Backend runs scenario generation → execution → report (in Docker)
8. Frontend shows progress indicator (polling status endpoint)
9. Report displayed in browser (rendered from JSON)
```

---

## Backend: FastAPI Application

### File Structure

```
src/mycode/web/
├── __init__.py
├── app.py              # FastAPI app, CORS, lifespan
├── routes.py           # All endpoint definitions
├── schemas.py          # Pydantic request/response models
├── jobs.py             # Job state machine (in-memory dict)
├── project_fetch.py    # Clone GitHub URL / extract zip
└── worker.py           # Background task runner (asyncio)
```

**Why inside `src/mycode/web/`:** The backend imports myCode modules directly (`pipeline`, `viability`, `ingester`, `interface`, `container`). Co-locating it in the package avoids path hacks. The web module is an optional install extra (`pip install mycode-ai[web]`).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MYCODE_TEMP_DIR` | system temp | Where job working directories live |
| `MYCODE_MAX_CONCURRENT_JOBS` | `4` | Concurrent pipeline runs |
| `MYCODE_JOB_TTL_SECONDS` | `1800` | How long to keep finished job data |
| `MYCODE_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `MYCODE_DEFAULT_LLM_KEY` | (none) | MAS proxy key for free tier LLM |
| `MYCODE_DEFAULT_LLM_MODEL` | `gemini-2.5-flash` | Default model |
| `MYCODE_DEFAULT_LLM_BASE_URL` | Gemini endpoint | Default LLM endpoint |
| `MYCODE_DOCKER_ENABLED` | `true` | Run tests in Docker containers |
| `MYCODE_MAX_PROJECT_SIZE_MB` | `100` | Max upload / cloned repo size |
| `MYCODE_LOG_LEVEL` | `INFO` | Logging level |

### Endpoints

#### `POST /api/preflight`

Runs stages 1–4.5 (language detection through viability gate). Fast — no LLM calls, no test execution.

**Request:**
```json
{
  "github_url": "https://github.com/user/repo",   // either this
  "upload_id": "abc123"                             // or this (from upload)
}
```

**Or multipart upload:**
```
POST /api/preflight
Content-Type: multipart/form-data
file: project.zip
```

**Response:**
```json
{
  "job_id": "j_abc123def",
  "language": "python",
  "project_name": "my-flask-app",
  "dependencies": [
    {"name": "flask", "installed_version": "3.0.0", "is_missing": false},
    {"name": "sqlalchemy", "installed_version": null, "is_missing": true}
  ],
  "viability": {
    "viable": true,
    "install_rate": 0.85,
    "import_rate": 0.78,
    "syntax_rate": 1.0,
    "missing_deps": ["some-obscure-lib"],
    "reason": ""
  },
  "profile_matches": ["flask", "sqlalchemy", "requests"],
  "inference_predictions": {
    "vertical": "web_backend",
    "risk_areas": ["concurrent_execution", "memory_profiling"],
    "architectural_pattern": "mvc"
  },
  "warnings": []
}
```

If viability fails, `viability.viable` is `false` and `viability.reason` explains why. Frontend shows the diagnostic and stops — no conversation needed.

**Implementation:**
```python
# Pseudocode — routes.py
async def preflight(request: PreflightRequest):
    job_id = create_job()
    project_path = await fetch_project(request)  # clone or extract

    language = detect_language(project_path)

    # Run stages 2–4.5 in thread pool (they're sync/CPU-bound)
    session = SessionManager(project_path, temp_base=job_temp_dir(job_id))
    session.setup()

    ingestion = ingest(project_path, language)
    matches = library.match_dependencies(ingestion.dependencies)
    viability = run_viability_gate(session, ingestion, language, is_containerised=True)

    # Stash session + ingestion on job for later stages
    store_job_state(job_id, session=session, ingestion=ingestion,
                    matches=matches, viability=viability, language=language)

    return PreflightResponse(job_id=job_id, ...)
```

**myCode modules imported:** `pipeline.detect_language`, `session.SessionManager`, `ingester.ProjectIngester` / `js_ingester.JsProjectIngester`, `library.loader.ComponentLibrary`, `viability.run_viability_gate`, `inference.InferenceEngine`, `classifiers`.

---

#### `POST /api/converse`

Handles one turn of the conversational interface. Called twice (Turn 1 and Turn 2).

**Request:**
```json
{
  "job_id": "j_abc123def",
  "turn": 1,
  "user_response": ""           // empty for Turn 1 (backend generates first question)
}
```

Turn 1 response (backend asks the first question):
```json
{
  "job_id": "j_abc123def",
  "turn": 1,
  "question": "I've analysed your Flask web application built with SQLAlchemy and Redis...\n\nWhat does this application do, and how do people use it?",
  "project_summary": "Flask web app with 12 routes, SQLAlchemy ORM, Redis caching...",
  "done": false
}
```

Turn 2 request (user answers Turn 1, backend asks Turn 2):
```json
{
  "job_id": "j_abc123def",
  "turn": 2,
  "user_response": "It's an inventory management API for a small warehouse, about 10 concurrent users"
}
```

Turn 2 response:
```json
{
  "job_id": "j_abc123def",
  "turn": 2,
  "question": "Thanks. A few more details:\n\n1. What's the largest data payload...\n2. How is this deployed...",
  "done": false
}
```

Final submission (user answers Turn 2):
```json
{
  "job_id": "j_abc123def",
  "turn": 3,
  "user_response": "Payloads are small JSON, deployed on a single EC2 instance, runs 24/7"
}
```

Final response:
```json
{
  "job_id": "j_abc123def",
  "turn": 3,
  "constraints": {
    "user_scale": 10,
    "usage_pattern": "sustained",
    "deployment_context": "single_server",
    "availability_requirement": "always_on",
    "data_type": "api_responses",
    "max_payload_mb": null,
    "data_sensitivity": null,
    "growth_expectation": null
  },
  "operational_intent": "Inventory management API for warehouse operations, ~10 concurrent users, single server deployment",
  "done": true
}
```

**Implementation:**

The `ConversationalInterface` class currently uses the `UserIO` protocol (display/prompt). For the web, we split the conversation into discrete turns:

```python
# Approach: decompose ConversationalInterface.run() into turn-based methods
#
# Option A: Refactor interface.py to support turn-by-turn mode
# Option B: Build a WebConversationAdapter that simulates UserIO
#
# Choose Option A — cleaner, no hacks.

# New methods on ConversationalInterface:
#   prepare_turn_1(ingestion, language) -> (question_text, project_summary)
#   process_turn_1(user_response) -> (question_text_for_turn_2)
#   process_turn_2(user_response) -> InterfaceResult
#
# These are stateless — all state passed in request or stored in job dict.
```

The interface currently does:
1. `_summarize_ingestion()` → project summary (~500 tokens)
2. Turn 1: Display analysis + ask "what does this do?"
3. Turn 2: Parse Turn 1 answer → ask targeted follow-up questions
4. Parse Turn 2 answer → `InterfaceResult` with intent + constraints

For the web, we expose the same logic as three discrete calls. The `ConversationalInterface` needs a `prepare_turn_1()` / `process_turn_1()` / `process_turn_2()` API alongside the existing `run()` method (CLI still uses `run()`).

**myCode modules imported:** `interface.ConversationalInterface`, `constraints` (for parsing), `scenario.LLMConfig`.

---

#### `POST /api/analyze`

Triggers the full test run (scenarios 6–9). Long-running — returns immediately with job ID, client polls status.

**Request:**
```json
{
  "job_id": "j_abc123def",
  "auto_approve_scenarios": true,
  "tier": 2
}
```

`auto_approve_scenarios` defaults to `true` for web (no scenario review step — users don't understand scenario details). `tier` determines scope: 2 = targeted (8–12 tests), 3 = full suite.

**Response (immediate):**
```json
{
  "job_id": "j_abc123def",
  "status": "running",
  "message": "Generating and executing stress tests..."
}
```

**Implementation:**

```python
async def analyze(request: AnalyzeRequest):
    job = get_job(request.job_id)  # has session, ingestion, matches, constraints

    # Launch in background thread
    asyncio.get_event_loop().run_in_executor(
        thread_pool,
        run_remaining_pipeline,
        job
    )

    return {"job_id": job.id, "status": "running"}

def run_remaining_pipeline(job):
    """Runs stages 6-9 synchronously in a worker thread."""
    try:
        # Build PipelineConfig with prebuilt state
        config = PipelineConfig(
            project_path=job.project_path,
            operational_intent=job.intent_string,
            prebuilt_constraints=job.constraints,
            llm_config=job.llm_config,
            auto_approve_scenarios=True,
            offline=job.offline,
        )
        # We need a way to resume pipeline from stage 6.
        # Option: new function run_pipeline_from_scenarios(config, ingestion, matches, session)
        # This avoids re-running stages 1-5.
        result = run_pipeline_from_scenarios(
            config=config,
            session=job.session,
            ingestion=job.ingestion,
            profile_matches=job.matches,
            language=job.language,
        )
        job.status = "completed"
        job.result = result
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
```

**Pipeline change needed:** Extract a `run_pipeline_from_scenarios()` function from `pipeline.py` that accepts pre-computed stages 1–5 results and runs stages 6–9 only. This avoids duplicating work.

**myCode modules imported:** `pipeline.run_pipeline_from_scenarios` (new), `scenario.ScenarioGenerator`, `engine.ExecutionEngine`, `report.ReportGenerator`, `container.run_containerised`.

---

#### `GET /api/status/{job_id}`

Poll endpoint for frontend progress tracking.

**Response:**
```json
{
  "job_id": "j_abc123def",
  "status": "running",          // "preflight" | "conversing" | "running" | "completed" | "failed"
  "stage": "execution",         // current pipeline stage
  "progress": {
    "scenarios_total": 10,
    "scenarios_completed": 4,
    "current_scenario": "concurrent_db_queries",
    "elapsed_seconds": 45
  }
}
```

Progress data comes from the job state, updated by the worker thread as scenarios complete.

---

#### `GET /api/report/{job_id}`

Returns the full report as JSON once the job is completed.

**Response:**
```json
{
  "job_id": "j_abc123def",
  "report": { ... },           // DiagnosticReport.as_dict()
  "pipeline_result": {
    "language": "python",
    "total_duration_ms": 142000,
    "scenarios_run": 10,
    "scenarios_passed": 7,
    "scenarios_failed": 3
  }
}
```

The `report` field is exactly `DiagnosticReport.as_dict()` — the same JSON already used for `mycode-report.json` file output. No new serialisation needed.

---

#### `GET /api/health`

```json
{
  "status": "ok",
  "docker_available": true,
  "version": "0.1.2",
  "active_jobs": 2,
  "max_concurrent_jobs": 4
}
```

---

### Job State Machine

```
                    ┌──────────┐
         ┌────────►│ preflight │
         │         └─────┬─────┘
    POST /preflight      │
                         ▼
              ┌─────────────────────┐
              │ preflight_complete   │ ◄── viability passed
              │ (or preflight_failed)│ ◄── viability failed (terminal)
              └──────────┬──────────┘
                         │
          POST /converse │ (turns 1–3)
                         ▼
              ┌─────────────────┐
              │ conversation_done│
              └────────┬────────┘
                       │
          POST /analyze│
                       ▼
              ┌──────────────┐
              │   running     │
              └───────┬──────┘
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
     ┌───────────┐      ┌──────────┐
     │ completed  │      │  failed   │
     └───────────┘      └──────────┘
```

Jobs stored in an in-memory `dict[str, Job]`. A background task reaps expired jobs (older than `MYCODE_JOB_TTL_SECONDS`). For v1, this is sufficient — no database, no Redis. Kubernetes scaling will require moving to Redis or a shared store, but the `jobs.py` module is the only file that needs to change.

**Job dataclass:**
```python
@dataclass
class Job:
    id: str
    status: str                                     # see state machine
    created_at: float                               # time.time()
    project_path: Path                              # temp directory
    language: Optional[str] = None
    session: Optional[SessionManager] = None
    ingestion: Optional[IngestionResult] = None
    matches: Optional[list] = None
    viability: Optional[ViabilityResult] = None
    intent: Optional[OperationalIntent] = None
    constraints: Optional[OperationalConstraints] = None
    llm_config: Optional[LLMConfig] = None
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    progress: Optional[dict] = None                 # scenario progress
```

---

### Docker Execution Strategy

For the web backend, **every test run executes inside Docker**. This is non-negotiable — the web accepts untrusted code.

Two options for Docker integration:

**Option A: Use existing `container.run_containerised()`**
- Pros: Already works, handles two-phase build
- Cons: Designed for CLI subprocess invocation, returns exit code not structured data

**Option B: Build project image, run pipeline inside container, capture JSON output**
- Pros: Structured output, better error handling
- Cons: More plumbing

**Choose Option B with reuse of Option A internals:**

```python
# worker.py
def run_job_in_docker(job: Job):
    """
    1. Use container._build_project_image() to build image with deps
    2. Run myCode inside container with --json-output flag
    3. Capture stdout JSON → parse into PipelineResult
    4. Update job state
    """
    base_tag = container.build_image()
    project_tag = container._build_project_image(base_tag, job.project_path)

    # Serialize constraints to temp file, mount into container
    constraints_json = serialize_constraints(job.constraints)

    result = docker_run(
        image=project_tag,
        command=["python", "-m", "mycode",
                 "--json-output",           # NEW: output JSON to stdout
                 "--constraints-file", "/tmp/constraints.json",
                 "--skip-version-check",
                 "--auto-approve",
                 job.project_copy_path],
        mounts={constraints_json: "/tmp/constraints.json"},
        network="none",
        timeout=MYCODE_JOB_TIMEOUT,
    )

    report_json = json.loads(result.stdout)
    job.result = deserialize_pipeline_result(report_json)
```

**New CLI flag needed:** `--json-output` — writes the full `PipelineResult` JSON to stdout instead of the text report. The `as_dict()` methods already exist on all result types.

**Alternative (better for v1):** Import and call `run_pipeline()` directly inside the container process. The container runs a small Python script that imports myCode, runs the pipeline, and writes JSON to a mounted volume. This avoids parsing stdout.

---

### Project Fetch (`project_fetch.py`)

```python
async def fetch_from_github(url: str, dest: Path) -> Path:
    """
    Clone a GitHub repo to dest directory.
    - Shallow clone (--depth 1) for speed
    - Validate URL format before cloning
    - Enforce MYCODE_MAX_PROJECT_SIZE_MB limit
    - Strip .git directory after clone
    """

async def fetch_from_upload(upload: UploadFile, dest: Path) -> Path:
    """
    Extract uploaded zip to dest directory.
    - Validate zip integrity
    - Enforce MYCODE_MAX_PROJECT_SIZE_MB limit
    - Reject zip bombs (check uncompressed size)
    - Only accept .zip files
    """
```

**Security considerations:**
- GitHub URL validation: must match `https://github.com/<owner>/<repo>` pattern
- No private repos (would need auth — out of scope for free tier)
- Size limit enforced before extraction
- Zip bomb detection: check declared uncompressed size vs threshold
- No symlinks followed during extraction

---

## Frontend: Single-Page Application

### File Structure

```
web/
├── index.html              # Single page
├── style.css               # Styles
├── app.js                  # All application logic
├── config.js               # Backend URL configuration
└── assets/
    └── logo.svg            # myCode logo (if exists)
```

**No build step.** Vanilla HTML/CSS/JS. No React, no Vue, no bundler. This is a single page with a linear flow — a framework adds complexity without value. Can be served from any static file host.

If a framework is later needed (enterprise tier with dashboards), it can be introduced then without rewriting the backend.

### Configuration

```javascript
// config.js
const MYCODE_CONFIG = {
    API_URL: window.MYCODE_API_URL || "http://localhost:8000",
};
```

For deployment, inject `window.MYCODE_API_URL` via a `<script>` tag or environment variable at build/serve time. For local dev, defaults to localhost.

### Page Layout (Single Flow)

```
┌─────────────────────────────────────────────────┐
│  myCode — Stress test your AI-generated code    │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Paste GitHub URL:                        │  │
│  │  [https://github.com/user/repo    ] [Go]  │  │
│  │                                           │  │
│  │  ── or ──                                 │  │
│  │                                           │  │
│  │  [Upload .zip]                            │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  PREFLIGHT DIAGNOSTICS                    │  │
│  │  Language: Python 3.11                    │  │
│  │  Dependencies: 8/9 installed (89%)        │  │
│  │  Imports: 7/8 working (88%)              │  │
│  │  Code syntax: 100% valid                  │  │
│  │  Status: ✓ Ready for testing              │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  CONVERSATION                             │  │
│  │                                           │  │
│  │  myCode: I've analysed your Flask app...  │  │
│  │  What does this application do?           │  │
│  │                                           │  │
│  │  You: [It's an inventory management...]   │  │
│  │       [Send]                              │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  [Run Stress Tests]                       │  │
│  │                                           │  │
│  │  ████████░░░░░░░░░░ 4/10 scenarios        │  │
│  │  Running: concurrent_db_queries...        │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  DIAGNOSTIC REPORT                        │  │
│  │                                           │  │
│  │  Summary: ...                             │  │
│  │                                           │  │
│  │  Findings:                                │  │
│  │  [CRITICAL] Memory exhaustion at 500...   │  │
│  │  [WARNING] Response time degrades...      │  │
│  │                                           │  │
│  │  [Download JSON] [Download Markdown]      │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  myCode by Machine Adjacent Systems             │
│  Diagnostic tool — does not guarantee fitness   │
└─────────────────────────────────────────────────┘
```

Each section reveals progressively as the flow advances. Sections above remain visible (user can scroll up to see preflight while conversation is active).

### Frontend Logic (app.js)

```
State machine:
  idle → preflight_running → preflight_done → conversing → ready_to_run → running → report_ready

API calls:
  1. POST /api/preflight        → show diagnostics
  2. POST /api/converse (×3)    → Turn 1, Turn 2, final
  3. POST /api/analyze          → start tests
  4. GET  /api/status/{id}      → poll every 3s while running
  5. GET  /api/report/{id}      → fetch and render report
```

### Report Rendering

The report JSON (`DiagnosticReport.as_dict()`) is rendered client-side. Key sections:

- **Summary** — `report.summary` as paragraph text
- **Findings** — each finding as a card with severity badge (critical=red, warning=amber, info=blue), title, description, affected dependencies
- **Degradation curves** — if `degradation_points` exist, render as simple tables (charts can come later)
- **Incomplete tests** — collapsed section showing what couldn't run and why
- **Metadata** — scenarios run/passed/failed, duration, model used

Download buttons offer `report` as JSON (`DiagnosticReport.as_dict()`) and Markdown (`DiagnosticReport.as_markdown()`).

---

## Changes to Existing myCode Modules

### 1. `interface.py` — Turn-based conversation API

**New methods on `ConversationalInterface`:**

```python
def prepare_turn_1(
    self, ingestion: IngestionResult, language: str
) -> tuple[str, str]:
    """Returns (question_text, project_summary). No user input needed."""

def process_turn_1(
    self, user_response: str, ingestion: IngestionResult, language: str
) -> str:
    """Parses Turn 1 answer, returns Turn 2 question text."""

def process_turn_2(
    self, user_response: str, turn_1_response: str,
    ingestion: IngestionResult, language: str
) -> InterfaceResult:
    """Parses Turn 2 answer, returns final InterfaceResult."""
```

The existing `run()` method (used by CLI) is refactored to call these internally. No behaviour change for CLI users.

### 2. `pipeline.py` — Partial pipeline execution

**New function:**

```python
def run_pipeline_from_scenarios(
    config: PipelineConfig,
    session: SessionManager,
    ingestion: IngestionResult,
    profile_matches: list[ProfileMatch],
    language: str,
    interface_result: InterfaceResult,
) -> PipelineResult:
    """Run stages 6-9 with pre-computed results from stages 1-5."""
```

This extracts the latter half of `run_pipeline()` into a reusable function. The existing `run_pipeline()` calls it internally.

### 3. `cli.py` — JSON output flag

**New flag:** `--json-output`

When set, the CLI writes the full `PipelineResult` as JSON to stdout instead of the text report. Used by the Docker worker to capture structured output.

### 4. `container.py` — Expose image build

Make `_build_project_image()` public (rename to `build_project_image()`). The web worker needs to call it directly.

---

## Data Flow Diagram

```
                         Frontend                              Backend
                         ────────                              ───────

User pastes URL ──────► POST /api/preflight ──────────────► fetch_from_github()
                                                            detect_language()
                                                            SessionManager.setup()
                                                            ProjectIngester.ingest()
                                                            ComponentLibrary.match()
                                                            run_viability_gate()
                         ◄──── preflight response ◄────────  InferenceEngine.infer()

User reads question ◄── POST /api/converse (turn=1) ──────► ConversationalInterface
                                                              .prepare_turn_1()
                         ◄──── question text ◄─────────────

User answers ──────────► POST /api/converse (turn=2) ──────► .process_turn_1()
                         ◄──── question text ◄─────────────

User answers ──────────► POST /api/converse (turn=3) ──────► .process_turn_2()
                         ◄──── constraints + intent ◄──────   → InterfaceResult

User clicks Run ───────► POST /api/analyze ────────────────► worker thread:
                         ◄──── {status: running} ◄─────────   build_project_image()
                                                               docker run:
                                                                 ScenarioGenerator
                                                                 ExecutionEngine
                                                                 ReportGenerator
                                                               → PipelineResult

Poll every 3s ─────────► GET /api/status/{id} ─────────────► job.progress
                         ◄──── progress update ◄───────────

Final poll ────────────► GET /api/report/{id} ──────────────► job.result.report
                         ◄──── DiagnosticReport JSON ◄─────   .as_dict()

Render report in browser
```

---

## Statelessness and Scaling

The backend is stateless **per-request** but holds **per-job** state in memory during a job's lifecycle. This is acceptable because:

1. Jobs are short-lived (max ~30 minutes for Tier 3)
2. Job state is reconstructable (worst case: re-run from preflight)
3. No user accounts, no persistent state

**For Kubernetes scaling later:**
- Replace `jobs.py` in-memory dict with Redis
- Job assignment via queue (Redis or cloud queue)
- Worker pods pull jobs from queue
- Status/report endpoints read from Redis
- Only `jobs.py` changes — all other code is unaffected
- Docker-in-Docker or Kubernetes Jobs for container execution

This is why `jobs.py` is isolated as a separate module.

---

## Security Considerations

1. **Untrusted code execution:** All test execution happens inside Docker with `--network=none`. The web backend never runs user code on the host.
2. **GitHub URL validation:** Strict regex. No arbitrary URLs, no file:// protocol, no private repos.
3. **Upload validation:** Size limit, zip bomb detection, no symlinks.
4. **Resource limits:** Docker containers have memory and CPU limits. `MYCODE_MAX_CONCURRENT_JOBS` prevents resource exhaustion.
5. **Job cleanup:** Expired jobs are reaped. Temp directories are cleaned up by `SessionManager.teardown()`.
6. **No secrets in responses:** Error messages are sanitised. Stack traces are logged server-side only.
7. **CORS:** Configurable, defaults to `*` for development. Production should restrict to the frontend domain.
8. **Rate limiting:** Not in v1. Add via reverse proxy (nginx/Cloudflare) or middleware later.

---

## Dependency Additions

Backend (new pip extras in `pyproject.toml`):
```toml
[project.optional-dependencies]
web = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-multipart>=0.0.18",    # file upload support
]
```

Frontend: None. Vanilla HTML/CSS/JS.

---

## Local Development

### Backend
```bash
pip install -e ".[web]"
uvicorn mycode.web.app:app --reload --port 8000
```

### Frontend
```bash
cd web/
python -m http.server 3000
# or: npx serve .
```

Open `http://localhost:3000`. Frontend talks to `http://localhost:8000`.

### Full stack with Docker
```bash
# Terminal 1: Backend
MYCODE_DOCKER_ENABLED=true uvicorn mycode.web.app:app --reload --port 8000

# Terminal 2: Frontend
cd web/ && python -m http.server 3000

# Prerequisite: Docker daemon running
```

---

## Implementation Order

1. **`schemas.py`** — Pydantic models for all request/response types
2. **`jobs.py`** — Job state machine and in-memory store
3. **`project_fetch.py`** — GitHub clone + zip extraction
4. **`interface.py` changes** — Turn-based conversation methods
5. **`pipeline.py` changes** — `run_pipeline_from_scenarios()`
6. **`routes.py`** — Preflight endpoint (stages 1–4.5)
7. **`routes.py`** — Converse endpoint (turns 1–3)
8. **`worker.py`** — Background Docker execution
9. **`routes.py`** — Analyze, status, report endpoints
10. **`app.py`** — FastAPI app assembly, CORS, lifespan (job reaper)
11. **`cli.py`** — `--json-output` flag
12. **`container.py`** — Make `build_project_image()` public
13. **Frontend** — `index.html`, `style.css`, `app.js`, `config.js`
14. **`pyproject.toml`** — Add `[web]` extras
15. **Tests** — Backend endpoint tests, conversation turn tests, job lifecycle tests

---

## What This Plan Does NOT Cover

- Authentication / user accounts (freemium tier — post-funding)
- Rate limiting (add at reverse proxy layer)
- Persistent storage / database (not needed for free tier)
- CI/CD pipeline (deployment-specific)
- SSL/TLS termination (reverse proxy)
- Monitoring / alerting (ops concern)
- Stripe billing integration (freemium tier)
- "myCode tested" badge generation (freemium tier)
- Frontend framework migration (if needed for enterprise dashboard)
- WebSocket for real-time progress (polling is sufficient for v1; upgrade later if needed)
