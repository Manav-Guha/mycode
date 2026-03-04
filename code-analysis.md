# myCode — Complete Codebase Analysis

**Generated:** 2026-03-04
**Codebase LOC:** 14,564 lines of Python (source files only, excludes profiles/tests)
**Profile Count:** 41 JSON profiles (18 Python + 23 JavaScript)

---

## Part 1: Natural Language Description of Each Component

---

### 1. Session Manager (`session.py` — ~962 lines)

**What it does:**
The Session Manager creates and manages isolated execution environments so that stress tests never touch the user's actual project or host system. When activated (as a context manager), it:

- Creates a temporary directory under the system's temp folder
- For Python projects: builds a full virtual environment (venv) replicating the user's Python version, then installs the project's dependencies by checking (in order) requirements.txt, pyproject.toml, setup.py, and finally falls back to `pip list` from the user's environment
- For JavaScript projects: copies node_modules or runs `npm install`
- Copies the user's project into the temp workspace, excluding irrelevant directories (.git, __pycache__, node_modules, .venv, etc.)
- Provides `run_in_session()` which executes arbitrary commands inside the sandbox with resource caps enforced via POSIX `setrlimit` (memory ceiling via RLIMIT_AS/RSS/DATA, process limits via RLIMIT_NPROC, wall-clock timeout)
- Registers signal handlers for SIGINT and SIGTERM so that Ctrl+C triggers cleanup instead of leaving orphaned temp directories
- On startup, scans for and removes orphaned temp environments from previous crashed runs (identified by a `mycode_session_` prefix)
- On exit (whether success, failure, or interrupt), destroys the entire temp directory tree

**Inputs:** Project path, language identifier, resource cap configuration (ResourceCaps dataclass with defaults: 512MB memory, 50 processes, 300s timeout).

**Outputs:** A managed execution context providing `run_in_session(command, timeout, env)` which returns stdout, stderr, return code, and timing data.

**What it does NOT do:**
- Does not analyze code — it only provides the sandbox
- Does not decide what to run — it only executes what it's told
- Does not modify the user's original project directory in any way
- Does not handle Windows resource limiting (setrlimit is POSIX-only; Windows gets timeout-only enforcement)
- Cross-platform process tree killing exists (POSIX killpg vs Windows taskkill) but Windows path is minimally tested

---

### 2. Project Ingester — Python (`ingester.py` — ~999+ lines)

**What it does:**
The Python Ingester performs deep static analysis of Python projects using the `ast` module. It:

- Walks the project directory finding all `.py` files
- For each file, runs a `_FileAnalyzer` (an `ast.NodeVisitor` subclass) that extracts: all function/method definitions (with signatures, line numbers, decorators), all class definitions (with methods and base classes), all imports (standard lib, third-party, local), global variable assignments, and inter-function call relationships
- Handles encoding issues including UTF-16 BOM detection
- Extracts dependencies from multiple sources in priority order: requirements.txt (with version specifier parsing), pyproject.toml (using tomllib, parsing both `[project.dependencies]` and `[tool.poetry.dependencies]`), setup.py (AST-based extraction — parses the setup() call's install_requires argument without executing the file), and setup.cfg (configparser-based)
- Resolves transitive dependencies using `importlib.metadata` to build a full dependency tree, not just declared top-level packages
- Checks each dependency's installed version against PyPI's latest stable version using parallel HTTP requests (ThreadPoolExecutor) to the PyPI JSON API
- Builds a function flow map (call graph) showing how functions call each other across files
- Identifies coupling points — functions that are high fan-in (called by many), cross-module hubs (called from multiple files), or shared state accessors (read/write globals)
- Produces two outputs per the token optimization spec: a full `IngestionResult` (for Scenario Generator and Report Generator) and a ~500-token summary (for Conversational Interface)
- Reports partial success gracefully: "analyzed 12 of 15 files, 3 couldn't be parsed"

**Inputs:** Project path, optional flag to skip version checks.

**Outputs:** `IngestionResult` dataclass containing: file analyses (per-file function/class/import data), dependency list with version info, function flow map, coupling points, file counts, line counts. Plus a separate text summary (~500 tokens).

**What it does NOT do:**
- Does not execute any user code — purely static analysis via AST
- Does not analyze JavaScript files (that's the JS Ingester)
- Does not resolve dynamic imports or runtime-generated module paths
- Does not analyze code quality, style, or correctness — only structure and dependencies
- Does not handle Cython, C extensions, or non-Python files in the project

---

### 3. Project Ingester — JavaScript (`js_ingester.py` — ~999+ lines)

**What it does:**
The JavaScript Ingester performs structural analysis of JavaScript and TypeScript projects using regex-based parsing (no external parser dependency). It:

- Walks the project directory finding `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs` files (excluding node_modules, dist, build, .next, etc.)
- Strips comments (single-line and multi-line) while preserving line numbers for accurate position reporting
- Uses 12+ regex patterns to identify imports: ES6 default imports, named imports, namespace imports, side-effect imports, CommonJS `require()` calls, and dynamic `import()` expressions
- Extracts function definitions: function declarations, arrow function assignments, function expressions, class methods (including async variants) — using brace-depth tracking for scope awareness
- Parses `package.json` for dependencies, devDependencies, and peerDependencies, identifying the entry point (main/module fields) and scripts
- Checks dependency versions against the npm registry API (parallel requests)
- Resolves transitive dependencies from lockfiles (package-lock.json or yarn.lock) when available
- Builds a call graph and identifies coupling points using the same classification system as the Python ingester

**Inputs:** Project path, optional flag to skip version checks.

**Outputs:** `IngestionResult` (same dataclass as Python ingester — polymorphic output), plus ~500-token text summary.

**What it does NOT do:**
- Does not use a real AST parser (Babel, TypeScript compiler, etc.) — relies entirely on regex patterns, which means it can miss complex or unusual syntax patterns
- Does not resolve TypeScript type-only imports vs value imports
- Does not analyze JSX/TSX component trees or React component hierarchies
- Does not handle monorepo structures (workspaces) or path aliases (tsconfig paths)
- Does not execute any code

---

### 4. Component Library (`library/loader.py` — ~505 lines, `library/profiles/` — 41 JSON files)

**What it does:**
The Component Library maintains pre-built behavioral profiles for common dependencies used in "vibe-coded" projects. It:

- Loads JSON profile files from the `profiles/python/` and `profiles/javascript/` directories at initialization
- Each profile (`DependencyProfile` dataclass) contains: identity (name, category, current stable version), scaling characteristics, memory behavior predictions, known failure modes, edge case sensitivities, interaction patterns with other dependencies, and stress test templates
- Matches detected project dependencies against profiles using a normalization system with extensive alias maps — Python has 17 aliases (e.g., `PIL` → `pillow`, `sklearn` → `scikit-learn`, `cv2` → `opencv-python`), JavaScript has 30+ aliases (e.g., `@supabase/supabase-js` → `supabase_js`, `next/router` → `nextjs`)
- Flags matched dependencies with version discrepancies (installed version differs from profile's documented stable version)
- Flags unmatched dependencies as "unrecognized" for inclusion in the report
- Provides properties like `browser_only` and `server_framework` for JS profiles to help the scenario generator select appropriate test strategies

**Inputs:** Language identifier (to load correct profile set), list of detected dependencies from ingester.

**Outputs:** List of matched `DependencyProfile` objects, list of unrecognized dependency names.

**What it does NOT do:**
- Does not generate profiles dynamically — profiles are static JSON files created offline
- Does not fetch profile updates at runtime — uses whatever ships with the installed version
- Does not pass profiles to the Conversational Interface (token optimization: profiles go to Scenario Generator and Report Generator only)
- Does not analyze actual dependency behavior — profiles are predictions based on documented/known characteristics

**Profile inventory (41 total):**

*Python (18):* flask, fastapi, streamlit, gradio, pandas, numpy, sqlite3, sqlalchemy, supabase, langchain, llamaindex, chromadb, openai, anthropic, requests, httpx, pydantic, os_pathlib

*JavaScript (23):* react, nextjs, express, node_core, tailwindcss, threejs, svelte, openai_node, anthropic_node, langchainjs, supabase_js, prisma, axios, mongoose, stripe, dotenv, zod, socketio, chartjs, google_auth_library, plotlyjs, react_chartjs_2, react_plotlyjs

*Note:* JavaScript has 5 more profiles than the spec's 18 — the extras are: chartjs, google_auth_library, plotlyjs, react_chartjs_2, react_plotlyjs. These were likely added during the pre-launch repo testing phase.

---

### 5. Conversational Interface (`interface.py` — ~870 lines)

**What it does:**
The Conversational Interface is the bridge between the user (speaking in domain language) and the technical stress testing system. It genuinely processes user input — it does not just collect strings. Specifically:

- Receives the ~500-token ingester summary (NOT the full analysis, per token optimization spec)
- Receives NO component library profiles (those go directly to Scenario Generator)
- Runs a 2-turn conversational exchange:
  - **Turn 1:** Asks the user to describe their project, intended audience, and usage patterns. In LLM mode, generates context-aware questions from the ingester summary. In offline mode, asks structured questions.
  - **Turn 2:** Asks about stress testing priorities and specific concerns.
- After Turn 1, runs `_extract_constraints()` which:
  1. First attempts **context-only parsing** — runs all constraint parsers (from `constraints.py`) against the Turn 1 text to extract whatever it can without asking further questions
  2. For any constraint fields still `None` after context parsing, asks **explicit structured questions** — e.g., "How many users do you expect?" with a numbered menu
  3. Parsers are sophisticated: `parse_user_scale()` handles ranges ("20-50"), k/m suffixes ("5k"), word estimates ("a few hundred"); `parse_data_type()` does keyword scoring across categories; `parse_max_payload()` handles unit conversion (KB/MB/GB)
  4. Skip detection: responses like "not sure", "idk", "skip", "n/a" are recognized and the field is left as None
- Synthesizes everything into an `OperationalIntent` object containing: the operational description, the structured `OperationalConstraints`, and raw user answers
- Also handles `review_scenarios()`: presents generated scenarios to the user for approval before execution (Y=approve, S=skip individual, N=reject all), with an auto-approve path for non-interactive mode
- Has a `_ask_project_name()` helper for getting a short project name for report headers

**Inputs:** Ingester summary (~500 tokens), IO interface (TerminalIO or mock), optional LLM config.

**Outputs:** `InterfaceResult` containing `OperationalIntent` (with `OperationalConstraints` dataclass) and raw conversation turns. The constraint object has 8 structured fields plus raw_answers.

**What it does NOT do:**
- Does not receive full ingestion data or component library profiles (token optimization)
- Does not generate scenarios — it only collects and structures user intent
- Does not make decisions about what to test — it passes structured constraints downstream
- Does not interact with the execution engine or session manager

**Critical finding: User input genuinely influences downstream behavior.** The constraint object flows through the pipeline into scenario generation, shaping scale boundaries (user_scale → concurrency levels), template selection (data_type + usage_pattern → relevant test categories), and termination conditions (3x stated capacity or crash). This is not theater — the conversational interface is a functional data extraction pipeline.

---

### 6. Scenario Generator (`scenario.py` — ~1000+ lines)

**What it does:**
The Scenario Generator translates structured project analysis + user constraints into concrete stress test configurations. It operates in two modes:

**LLM mode (API key available):**
- Sends a structured prompt to the LLM containing: full ingester output, matched component library profiles, and the constraint object
- The prompt asks the LLM to generate stress scenarios that test dependency interaction chains (not individual components)
- Parses the LLM response into `StressScenario` objects with category, description, parameters, and expected behavior
- Applies constraint-driven bounds: scale boundaries from user_scale, relevant templates from data_type/usage_pattern, termination at 3x stated capacity

**Offline mode (no API key):**
- Template-based generation using component library profiles
- For each matched profile: generates scenarios from the profile's `stress_test_templates`
- For each matched profile: generates scenarios from `known_failure_modes`
- For coupling points identified by the ingester: classifies each into a `CouplingBehaviorType` (STATE_SETTER, API_CALLER, PURE_COMPUTATION, DOM_RENDER, ERROR_HANDLER) using `classify_coupling_point()` which resolves function names to behavior types based on imports and naming patterns
- Generates coupling-specific scenarios (e.g., state setters get concurrent mutation tests, API callers get timeout/retry tests)
- Constraint parameterization: if user_scale is 20, concurrency scenarios test at [20, 30, 50, 100, 200] not arbitrary [1, 10, 100, 1000, 10000]

**Scenario categories:**
- *Shared:* data_volume_scaling, memory_profiling, edge_case_input, concurrent_execution
- *Python-specific:* blocking_io, gil_contention
- *JavaScript-specific:* async_promise_chain, event_listener_accumulation, state_management_degradation

**Inputs:** Full `IngestionResult`, matched `DependencyProfile` list, `OperationalConstraints`, language, LLM config (optional).

**Outputs:** List of `StressScenario` objects, each with: category, title, description, parameters dict, expected behavior description, target dependencies.

**What it does NOT do:**
- Does not execute tests — only generates configurations
- Does not generate test harness code — that's the Execution Engine
- Does not have access to user conversation text — only the structured constraint object
- In offline mode: cannot generate novel scenarios beyond template combinations (this ceiling is acknowledged in the report)

---

### 7. Execution Engine (`engine.py` — ~1700+ lines, largest file)

**What it does:**
The Execution Engine is the workhorse that actually runs stress tests against the user's code. It:

- Takes each `StressScenario` and generates a **self-contained harness script** — a complete Python or JavaScript file that can be executed independently in the session sandbox
- Harness structure follows a strict pattern:
  - **Preamble:** Imports the user's modules, discovers callable functions/classes, sets up measurement helpers (memory tracking via `tracemalloc` for Python / `process.memoryUsage()` for Node.js, timing via `time.perf_counter` / `performance.now()`)
  - **Category body:** The actual test logic, specific to the scenario category. For example:
    - `data_volume_scaling`: generates progressively larger synthetic inputs, measures response time and memory at each level
    - `memory_profiling`: runs repeated operations, samples memory between iterations, detects accumulation patterns
    - `edge_case_input`: generates malformed, empty, oversized, and wrong-type inputs, records errors
    - `concurrent_execution`: uses threading (Python) or Promise.all (JS) to run parallel operations against shared resources
    - `blocking_io`: tests I/O operations under concurrent load (Python-specific)
    - `gil_contention`: tests CPU-bound operations with threading to expose GIL bottlenecks (Python-specific)
  - **Coupling-specific bodies:** For coupling point scenarios — tests state setters under concurrent mutation, API callers under timeout conditions, error handlers under cascading failure, DB connectors under connection pool exhaustion
  - **Postamble:** Outputs results as JSON between `===MYCODE_RESULTS_START===` and `===MYCODE_RESULTS_END===` markers
- Executes each harness via `SessionManager.run_in_session()` with a 30-second per-scenario timeout cap
- Parses results by extracting JSON between the markers from stdout
- Implements deadlock detection heuristic: if a process produces no output for an extended period and hasn't terminated, flags it
- Handles harness crashes gracefully — catches the error, records it as a finding, moves to the next scenario

**Inputs:** List of `StressScenario` objects, `SessionManager` instance, project metadata from ingester.

**Outputs:** `ExecutionResult` containing: per-scenario results (raw metrics, errors, timing), scenarios completed count, scenarios failed count, resource cap terminations.

**What it does NOT do:**
- Does not decide what to test — only executes scenarios it's given
- Does not modify the user's original code — works on the copy in the session sandbox
- Does not interpret results — raw data goes to the Report Generator
- Does not handle real network calls or external API testing (v1 limitation)
- Harness generation is template-based string construction, not AST manipulation — this means edge cases in module naming or import structure can cause harness syntax errors

---

### 8. Report Generator (`report.py` — ~999+ lines)

**What it does:**
The Report Generator transforms raw execution data into a plain-language diagnostic report that a non-engineer can understand. It:

- Takes raw execution results + constraint object + full ingester output
- **Finding extraction (`_analyze_execution()`):** Classifies each scenario result into findings by severity:
  - `critical`: crashes, errors, resource cap terminations within user's stated capacity
  - `warning`: degradation patterns, slow responses, high memory usage approaching limits
  - `informational`: failures beyond stated capacity, minor performance observations
  - Environment-only errors (ModuleNotFoundError, ImportError) are routed to a separate `incomplete_tests` category rather than reported as project failures — this is an important distinction that prevents false positives from sandbox setup issues
- **Degradation detection (`_detect_degradation()`):** Analyzes metric curves to find breaking points:
  - Looks for 2x+ ratios between consecutive measurements
  - Detects 3x+ spikes (sudden jumps)
  - Applies a flatness gate (coefficient of variation < 10% means "no meaningful variation, not a real degradation")
  - Applies minimum delta thresholds to avoid flagging tiny absolute changes as degradation
  - Produces `DegradationPoint` objects with the metric name, the load level where degradation begins, and the breaking point
- **Constraint contextualization (`_contextualise_findings()`):** This is where user intent meets results:
  - Findings within stated capacity are classified as critical — "You said 20 users. Your app crashes at 15."
  - Findings beyond stated capacity are classified as informational — "Your app breaks at 51 users, which is beyond your stated 20."
  - Where a constraint was None: "User scale not specified — tested at default range"
- **Grouping:** Similar findings and degradation points are grouped to avoid repetitive reporting
- **Output formats:**
  - `as_text()`: Plain terminal output with sections for summary, critical findings, warnings, degradation points, version flags, incomplete tests
  - `as_markdown()`: Structured markdown with headers, tables, and proper formatting for the `--report` flag
  - `as_dict()`: JSON-serializable dictionary for the `--json-output` flag
- **LLM-enhanced summary:** When an API key is available, sends findings to the LLM to generate a plain-English executive summary paragraph. Offline mode uses a template-based summary.

**Inputs:** `ExecutionResult`, `OperationalConstraints`, `IngestionResult`, matched profiles, LLM config (optional).

**Outputs:** `DiagnosticReport` containing: findings list, degradation points, version flags, unrecognized deps, incomplete tests, summary text. Renderable as text, markdown, or dict.

**What it does NOT do:**
- Does not prescribe fixes or generate patches — diagnosis only
- Does not rank findings by ease-of-fix — only by severity relative to user intent
- Does not generate code suggestions or link to documentation
- Does not compare results against previous runs (no historical tracking in v1)

---

### 9. Interaction Recorder (`recorder.py` — 514 lines)

**What it does:**
The Interaction Recorder captures anonymized session data for component library improvement, gated behind explicit user consent. It:

- Only activates if the user passes `--consent`
- Records: conversation turns (stripped of potential PII), dependency combinations encountered, scenario configurations, execution results summary, report summary
- Stores as date-partitioned JSONL files in `~/.mycode/recordings/` (e.g., `2026-03-04.jsonl`)
- Extracts failure patterns from execution results — identifies which dependency combinations produced failures and what category of failure
- Maintains an aggregated `unrecognized_deps.json` at `~/.mycode/` tracking frequency of unrecognized dependencies across all sessions (this file is updated regardless of consent, as it contains no PII)

**Inputs:** Session data from pipeline (conversation, scenarios, execution results, report).

**Outputs:** JSONL file written to disk. Failure pattern summary for potential library enrichment.

**What it does NOT do:**
- Does not activate without explicit consent
- Does not record user code content — only structural metadata
- Does not transmit data anywhere — everything stays local in v1
- Does not record if consent is not given (except unrecognized dep frequency, which contains no PII)

---

### 10. Discovery Engine (`discovery.py` — 762 lines)

**What it does:**
The Discovery Engine detects novel failure patterns that the component library didn't predict, creating a feedback loop from real-world usage into the library. It:

- Compares each execution result against the relevant component library profile's predictions
- Detects 5 types of discoveries:
  1. **crash_at_safe_level**: Code crashes at a load level the profile marks as safe
  2. **memory_growth_anomaly**: Memory grows >2x faster than the profile predicts
  3. **curve_shape_mismatch**: Degradation follows an unexpected curve (e.g., exponential where profile predicts linear)
  4. **interaction_failure**: A failure involving multiple dependencies that isn't documented in either dependency's profile
  5. **unrecognized_dep_failure**: An unrecognized dependency exhibits a catalogable failure pattern
- Each discovery is saved as a JSON file in `~/.mycode/discoveries/` with a UUID filename
- Discovery logging is local and requires NO consent — the files contain dependency behavior observations only, no user code or PII
- On subsequent runs, checks local discoveries against current results to identify "confirmed" patterns (reproduced across 2+ distinct projects)

**Inputs:** Execution results, matched component library profiles, ingestion metadata.

**Outputs:** List of `DiscoveryCandidate` objects written as JSON files to `~/.mycode/discoveries/`.

**What it does NOT do:**
- Does not use LLM — detection is deterministic comparison (actual vs expected)
- Does not automatically promote discoveries to the component library (requires manual review)
- Does not share discoveries externally in v1 (even with consent, sharing is not yet implemented)
- Does not detect user code bugs — only dependency behavior deviations from profile predictions

---

### 11. Pipeline Orchestrator (`pipeline.py` — 908 lines)

**What it does:**
The Pipeline orchestrates all 9 stages in sequence, handling errors at each stage and assembling the final result. The stages are:

1. **Language detection:** Checks for indicator files (requirements.txt, package.json, etc.) and counts file extensions (.py vs .js/.ts) to determine Python or JavaScript
2. **Session setup:** Creates the SessionManager with appropriate resource caps
3. **Ingestion:** Runs the appropriate ingester (Python or JS) to analyze the project
4. **Library matching:** Loads component library profiles and matches against detected dependencies
5. **Conversation:** Runs the Conversational Interface to extract user intent and constraints (skipped in non-interactive mode with defaults)
6. **Scenario generation:** Runs the Scenario Generator with full ingester output + profiles + constraints
7. **Scenario review:** Presents scenarios to user for approval (auto-approved in non-interactive mode)
8. **Execution:** Runs approved scenarios via the Execution Engine inside the session sandbox
9. **Report generation:** Produces the diagnostic report from execution results + constraints

Each stage is wrapped in error handling — if a stage fails, the pipeline records the failure, optionally continues to later stages where possible, and always runs cleanup. Discovery logging and interaction recording run as `_safe_record()` calls after the main pipeline completes (failures in recording/discovery never crash the pipeline).

**Inputs:** `PipelineConfig` containing: project path, language override, LLM config, offline flag, skip_version_check, consent, IO interface, operational intent (for non-interactive), auto_approve flag.

**Outputs:** `PipelineResult` containing: success boolean, per-stage results, ingestion data, interface result, scenarios, execution data, report, warnings, recording path, failed stage identifier.

---

### 12. CLI (`cli.py` — 276 lines)

**What it does:**
The CLI is the user-facing entry point. It parses command-line arguments, builds the pipeline configuration, runs the pipeline, and handles output. Flags: `--offline`, `--language`, `--consent`, `--api-key`, `--api-base`, `--model`, `--skip-version-check`, `--non-interactive`, `--json-output`, `--report`, `--verbose`. Auto-detects offline mode when no API key is available (checks `--api-key` flag then `GEMINI_API_KEY` env var). Auto-detects non-interactive mode when stdin is not a TTY.

---

### 13. Constraints Module (`constraints.py` — 479 lines)

**What it does:**
Provides the `OperationalConstraints` dataclass and all parsing functions used by the Conversational Interface. The parsers are genuinely sophisticated:

- `parse_user_scale()`: Handles "about 20", "20-50" (takes midpoint), "5k" (→5000), "a few hundred" (→300), "couple dozen" (→24)
- `parse_data_type()`: Keyword scoring system — scores each input against keyword sets for tabular/text/images/mixed/api_responses, returns highest-scoring category
- `parse_usage_pattern()`: Maps keywords to sustained/burst/periodic/growing
- `parse_max_payload()`: Handles "50MB", "2 GB", "500kb" with unit conversion to MB
- `parse_deployment_context()`: Maps to single_server/local_only/cloud/shared_hosting
- `parse_data_sensitivity()`: Maps to public/internal/customer_data/financial/medical
- `parse_growth_expectation()`: Maps to stable/slow_growth/rapid_growth
- Skip detection across all parsers: "not sure", "idk", "skip", "n/a", "don't know" → returns None

---

## Part 2: SWOT Analysis

---

### Strengths

**1. Architectural coherence and spec fidelity.**
The codebase closely follows the CLAUDE.md specification. The 9-component architecture is cleanly implemented with clear boundaries. Components communicate through well-defined dataclasses (`IngestionResult`, `OperationalConstraints`, `StressScenario`, `ExecutionResult`, `DiagnosticReport`). The pipeline orchestrator ties everything together without components reaching into each other's internals.

**2. The constraint extraction pipeline is real, not cosmetic.**
User input genuinely flows through the system: natural language → constraint parsers → structured `OperationalConstraints` → scenario parameterization → report contextualization. The parsers handle real-world input patterns (ranges, suffixes, colloquialisms). The scenario generator uses constraints to bound scale, filter templates, and set termination conditions. The report generator classifies findings relative to stated capacity. This is the product's core differentiator and it works end-to-end.

**3. Graceful degradation across all modes.**
Every LLM-dependent component (Interface, Scenario Generator, Report Generator) has a complete offline fallback. The offline path isn't a stub — it produces usable output through structured questions, template-based scenarios, and template-based reports. The system also degrades gracefully on partial ingestion failure, harness crashes, and resource cap terminations.

**4. Session isolation is thorough.**
The Session Manager creates genuine sandboxes with resource caps, signal handlers, orphan cleanup, and guaranteed teardown. User code is never touched. The venv replication logic (checking 4 dependency file types with fallback to pip list) is robust. This is critical for a tool that runs arbitrary user code.

**5. Token optimization is implemented, not just planned.**
The spec's token optimization (summarized ingester output to Conversational Interface, full output to Scenario Generator and Report Generator, no profiles to Interface) is actually wired in the code. The `summarize_ingestion()` function in `interface.py` produces a ~500-token summary. This keeps free-tier costs viable.

**6. Comprehensive dependency normalization.**
The alias maps in `loader.py` (17 Python aliases, 30+ JavaScript aliases including scoped npm packages) mean that real-world projects with varied import names will match profiles correctly. This is a practical detail that many tools get wrong.

**7. Component library exceeds spec.**
41 profiles vs the spec's 36. The 5 extra JS profiles (chartjs, google_auth_library, plotlyjs, react_chartjs_2, react_plotlyjs) show active growth from the pre-launch testing pipeline.

**8. Report differentiation: within-capacity vs beyond-capacity.**
The report's classification of findings as "critical" (within stated capacity) vs "informational" (beyond stated capacity) is genuinely useful. A non-engineer user learning that their 20-user app breaks at 15 users (critical) gets actionable information, while learning it also breaks at 200 users (informational) is useful context without inducing panic.

**9. Environment error routing.**
The report generator's handling of ModuleNotFoundError/ImportError — routing them to `incomplete_tests` rather than reporting them as project failures — prevents false positives from sandbox setup issues. This is a subtle but important quality signal.

**10. Discovery Engine creates a data flywheel.**
Every run generates potential discoveries. Local logging requires no consent. Confirmed patterns (reproduced across 2+ projects) are flagged for promotion. This creates a growth pipeline from usage into the library without requiring user opt-in for the local portion.

---

### Weaknesses

**1. JavaScript ingestion relies entirely on regex — no real parser.**
The JS ingester uses 12+ regex patterns instead of a proper AST parser (Babel, TypeScript compiler, Acorn). This means:
- Complex syntax will be missed: computed property names, decorators, nested destructuring in imports, template literal tag functions
- TypeScript-specific constructs (type-only imports, generics, enum declarations) are partially or wholly invisible
- JSX analysis is superficial — component trees, prop flows, and hook dependency arrays are not parsed
- Error recovery on malformed files is poor compared to a real parser
- This is the single largest technical gap between the Python path (which uses `ast` module, a real parser) and the JavaScript path

**2. Harness generation is string concatenation, not AST construction.**
The Execution Engine builds harness scripts by concatenating string templates. This means:
- Edge cases in module naming, unusual import structures, or special characters in function names can produce invalid harness code
- There's no validation that the generated harness is syntactically correct before execution
- Debugging harness failures requires reading the concatenated output, which is hard to trace back to the template source
- The harness preamble's "callable discovery" logic (finding functions to test in the imported module) uses `dir()` + `callable()` which misses some patterns (e.g., classes that aren't callable but have important methods)

**3. No real concurrency testing — threading in Python, Promise.all in JS.**
The concurrent execution harness uses Python `threading` and JS `Promise.all`. For Python, this means:
- True parallelism is limited by the GIL for CPU-bound work (which is tested separately in `gil_contention`, but the concurrency harness doesn't account for this)
- Process-based concurrency (`multiprocessing`) is not tested
- For web frameworks (Flask, FastAPI), the harness doesn't start an actual server and send HTTP requests — it calls functions directly, missing middleware, routing, request parsing, and response serialization overhead
- For JS, `Promise.all` tests async concurrency but not multi-process or cluster scenarios

**4. No actual HTTP/API stress testing.**
Despite profiling web frameworks (Flask, FastAPI, Express, Next.js), the execution engine never starts a server process and sends real HTTP requests. Stress testing a web app by calling its route handler functions directly misses: WSGI/ASGI overhead, middleware chains, request parsing, connection handling, keep-alive behavior, and response serialization. This is a significant gap for the most common vibe-coded project type (web apps).

**5. Windows support is incomplete.**
- `setrlimit` (resource caps) is POSIX-only — Windows gets timeout-only enforcement, meaning no memory ceiling or process limit enforcement
- Process tree killing uses `taskkill` on Windows but this path appears minimally tested
- Signal handlers (SIGINT, SIGTERM) behave differently on Windows
- Venv creation and activation paths differ on Windows and edge cases may not be covered

**6. No persistent state between runs.**
Each myCode run is independent — there's no mechanism to:
- Compare current results against previous runs on the same project
- Track whether issues found in a previous run have been fixed
- Show improvement or regression over time
- The Discovery Engine maintains local state, but it's for library enrichment, not for project-level tracking

**7. Degradation detection has limitations.**
The `_detect_degradation()` algorithm uses ratio-based detection (2x+ between consecutive measurements, 3x+ spikes). This means:
- Gradual degradation that never hits 2x between consecutive points is missed (e.g., 10% increase per step over 20 steps = 6.7x total but never 2x between steps)
- The flatness gate (CV < 10%) and minimum delta thresholds help prevent false positives but could also suppress real findings in edge cases
- No statistical significance testing — small sample sizes can produce spurious ratios

**8. Offline scenario generation is limited to templates.**
Without an LLM, the Scenario Generator can only produce scenarios from component library templates and known failure modes. It cannot:
- Reason about novel dependency interactions specific to this particular project
- Generate project-specific edge cases that aren't covered by generic templates
- Adapt scenarios based on the specific way dependencies are used in the code
- This ceiling is acknowledged in the report, but it means the free tier (offline) and the free tier (with LLM) have significantly different diagnostic quality

**9. Large file handling in the ingester.**
Both ingesters read entire files into memory for parsing. For projects with very large generated files (e.g., bundled output, large data files with .py/.js extensions accidentally included), this could cause memory issues in the analysis tool itself.

**10. 30-second per-scenario timeout is rigid.**
The execution engine caps each scenario at 30 seconds. For legitimate long-running operations (large data processing, complex model loading), this may terminate before meaningful results are captured. There's no user-configurable timeout.

---

### Opportunities

**1. Add a real JavaScript parser.**
Replacing the regex-based JS ingester with a proper parser (e.g., shell out to Node.js running Babel/Acorn/TypeScript compiler, or use a Python-based JS parser like `pyjsparser`) would dramatically improve JS analysis quality. This could be done incrementally — keep the regex fallback for when no Node.js is available, use the real parser when it is.

**2. HTTP-level stress testing for web frameworks.**
Starting an actual server process (via `subprocess`) and sending HTTP requests (via `httpx` or `requests`) would enable realistic web app stress testing. The framework profiles already identify entry points and routes — the infrastructure is there, the harness generation just needs a "server mode."

**3. Historical comparison ("regression mode").**
Storing previous run results (e.g., in `~/.mycode/history/<project-hash>/`) and showing deltas on subsequent runs would add significant value: "Last run: broke at 51 users. This run: broke at 38 users. Regression detected." This is low-hanging fruit with high user value.

**4. Component library auto-growth from batch mining.**
The batch mining infrastructure (described in the spec, partially implemented) can systematically discover new profiles. Running myCode against hundreds of public GitHub repos and analyzing discovery patterns could expand the library from 41 profiles to 100+ before public launch.

**5. Freemium tier implementation.**
The architecture cleanly supports the freemium tier — the LLM abstraction layer (`LLMBackend` in scenario.py) already supports configurable endpoints and API keys. Adding authentication, token metering, and Stripe integration is product/infrastructure work, not architectural work.

**6. GitHub Action integration.**
The `--non-interactive --json-output` flags already enable CI/CD usage. Wrapping this in a GitHub Action definition (action.yml) with status checks based on critical finding count is straightforward and opens a new distribution channel.

**7. Interactive report in terminal.**
The current terminal output is plain text. Using `rich` (Python library) for colored output, progress bars during execution, and collapsible finding sections would significantly improve the user experience without changing any underlying logic.

**8. Profile version tracking.**
When a dependency version changes between the profile's documented stable version and the user's installed version, the report flags this but doesn't adjust predictions. Profiles could include version-specific behavioral notes (e.g., "Streamlit 1.32+ changed caching behavior") for more accurate predictions.

**9. Monorepo and multi-project support.**
Currently requires pointing at a single project directory. Supporting monorepos (detecting workspace structure, testing individual packages) would expand the addressable market.

**10. MCP server integration.**
Exposing myCode's analysis capabilities as an MCP (Model Context Protocol) server would allow AI coding tools (Claude Code, Cursor, etc.) to directly invoke stress testing during development sessions. The pipeline's clean input/output contracts make this feasible.

---

### Threats

**1. LLM provider dependency and cost volatility.**
The free tier depends on Gemini Flash pricing remaining low (~$6-12 per 1,000 sessions). API pricing changes, rate limit reductions, or free tier eliminations by Google could make the free tier unviable. The BYOK architecture mitigates this partially, but the default "zero config" path depends on a specific provider.

**2. Accuracy of component library profiles.**
Profiles are LLM-generated and human-reviewed, but the volume of dependencies in the ecosystem means profiles can become outdated quickly. A profile that predicts "linear memory growth" for a dependency that changed behavior in a recent version will produce false discoveries and misleading reports. Profile maintenance is an ongoing cost that scales with the library size.

**3. False positives eroding trust.**
Several mechanisms can produce false positives:
- Harness generation failures (syntax errors in concatenated code) reported as execution failures
- Module import errors in the sandbox environment reported as project issues (mitigated by the incomplete_tests routing, but not perfectly)
- Degradation detection triggering on noise in small sample sizes
- Profile predictions being wrong (leading to false discoveries)
If a non-engineer user sees findings that don't match reality, trust in the tool is damaged. The target audience lacks the technical knowledge to distinguish real findings from artifacts.

**4. Sandbox escape from malicious code.**
The Session Manager creates venvs and runs arbitrary user code. While resource caps limit damage, the isolation is process-level, not container-level. A sophisticated malicious project could:
- Use `os.system()` or `subprocess` to escape the venv
- Access the host filesystem outside the temp directory
- Exfiltrate data via network calls (no network isolation in v1)
The "basic pre-execution scan" mentioned in the spec is not yet fully implemented. Docker containerization (freemium tier feature) would address this, but the free tier runs on the user's host.

**5. Competition from AI coding tools.**
Claude Code, Cursor, Copilot, and similar tools are rapidly adding code analysis and testing capabilities. If these tools add stress testing as a built-in feature, myCode's value proposition narrows. The differentiation is "stress testing from user intent in domain language" — this is defensible but requires continued innovation.

**6. JavaScript ecosystem churn.**
The JS ecosystem changes faster than Python. Frameworks like React, Next.js, and Svelte have frequent major version changes that alter behavior. Keeping 23 JS profiles current requires ongoing effort. The regex-based ingester compounds this — new syntax features in JS/TS may not be recognized.

**7. Scale of unrecognized dependencies.**
With only 41 profiles covering a universe of thousands of npm/PyPI packages, most real projects will have unrecognized dependencies. The "generic stress testing based on ingester analysis" fallback for unrecognized deps is less targeted. If users consistently see "X dependency is untested" warnings, the tool may feel incomplete.

**8. Single maintainer risk.**
The spec identifies Manav as sole merge authority and primary builder. At 14,564 lines of Python with complex interdependencies, maintaining this codebase, reviewing PRs, updating profiles, processing discoveries, and expanding the library is a significant workload for one person. Bus factor = 1.

**9. Open source sustainability.**
The free tier is fully open source and functional. If the free tier is "good enough" for most users, conversion to freemium may be low. The quality difference (Gemini Flash vs premium model for scenario generation) needs to be perceptible and meaningful to drive upgrades. If offline mode produces adequate results, even the LLM dependency becomes optional for many users.

**10. Regulatory and liability risk.**
Running arbitrary code, even in a sandbox, creates liability surface. If myCode's sandbox fails and user code damages the host system, or if a myCode report says "your app handles 50 users" and it doesn't, there's liability exposure. The disclaimer helps legally but doesn't prevent reputation damage. ADGM incorporation provides a regulatory framework, but this is untested territory for a diagnostic tool.

---

## Appendix: Key Metrics

| Metric | Value |
|---|---|
| Total Python source LOC | 14,564 |
| Number of source files | 16 (.py files in src/mycode/) |
| Python profiles | 18 |
| JavaScript profiles | 23 |
| Total profiles | 41 |
| LLM-dependent components | 3 (Interface, Scenario Gen, Report Gen) |
| Offline-capable components | All (3 LLM components have fallbacks) |
| Execution categories (shared) | 4 |
| Execution categories (Python-specific) | 2 |
| Execution categories (JS-specific) | 3 |
| Constraint fields extracted | 8 structured + raw_answers |
| Largest file | engine.py (~1700 lines) |
| Smallest file | __main__.py (7 lines) |

---

## Appendix: Data Flow Summary

```
User's project directory
    │
    ▼
[Language Detection] ──→ python | javascript
    │
    ▼
[Session Manager] ──→ isolated venv/sandbox with project copy
    │
    ▼
[Ingester (Py or JS)] ──→ IngestionResult (full) + Summary (~500 tokens)
    │                            │                      │
    │                            │                      ▼
    │                            │            [Conversational Interface]
    │                            │              receives: summary only
    │                            │              produces: OperationalConstraints
    │                            │                      │
    │                            ▼                      ▼
    │                    [Component Library] ──→ matched profiles
    │                            │                      │
    │                            ▼                      ▼
    │                    [Scenario Generator]
    │                      receives: full IngestionResult + profiles + constraints
    │                      produces: List[StressScenario]
    │                            │
    │                            ▼
    │                    [User Review] ──→ approved scenarios
    │                            │
    │                            ▼
    │                    [Execution Engine]
    │                      runs in: Session Manager sandbox
    │                      produces: ExecutionResult (raw metrics)
    │                            │
    │                            ▼
    │                    [Report Generator]
    │                      receives: ExecutionResult + constraints + full IngestionResult
    │                      produces: DiagnosticReport (text/markdown/JSON)
    │                            │
    │                            ├──→ [Discovery Engine] (compares actual vs profile predictions)
    │                            └──→ [Interaction Recorder] (if consent given)
    │
    ▼
[Session Manager cleanup] ──→ temp environment destroyed
```
