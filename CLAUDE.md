# myCode — Product Specification (CLAUDE.md)
# Version: 3.8 — May 2, 2026 (evening)
# This is the authoritative build specification. Do not override architectural decisions without explicit human approval.
# Current state (test counts, corpus numbers, infrastructure, completed tracks) is in MEMORY.md.
# Read all three at session start: CLAUDE.md, MEMORY.md, and known_issues.md (per discipline 3.11).
# Architectural decisions live in the canonical ADR document (myCode-Architecture-Decision-Records-v1_0-2026-05-01-with-ADR-012.md or its successor).
# When this document and an active ADR conflict, the ADR is authoritative and this document is reconciled to it.

---

## Parent Company
Machine Adjacent Systems (MAS)
Incorporation: ADGM (Abu Dhabi Global Market).

## Product Position within MAS
myCode is MAS's first product. MAS builds verification and analysis infrastructure for AI-generated software. myCode leads, MIAF is product 2. Funding is led by MAS (dual-product portfolio), not myCode alone.

## Tagline
"Built it with AI? Test it before it breaks."

---

## What myCode Is
A stress-testing tool that lets non-engineer builders verify their AI-generated code before deployment. The user points myCode at their project, describes what it does and how they intend to use it in a conversational exchange (domain language, not engineering language), and receives a plain-language degradation report showing where and how their code breaks under realistic future conditions.

## What myCode Is NOT
- Not a linter, code reviewer, or static analysis tool
- Not a security scanner (v1) — passive security assessment added in freemium tier
- Not a code generator or patch creator — myCode NEVER modifies user code
- Not a replacement for the user's coding tool — it complements Claude Code, Cursor, ChatGPT, Copilot
- Not a guarantee of code quality — diagnostic tool only, liability stays with user

---

## Three-Tier Product Architecture

### FREE TIER
- **Distribution:** GitHub (open source), pip install, community channels
- **LLM Backend:** Gemini 2.5 Flash via MAS proxy (anonymous, zero config, default) OR bring-your-own-key (any OpenAI-compatible endpoint including Gemini, Claude, local Ollama)
- **Registration:** None required. No account, no email, no payment.
- **Languages:** Python AND JavaScript/Node.js (full runtime testing on both, not dependency detection only — see ADR-011 for dependency-free project support)
- **Scope:** Local projects AND public GitHub URL ingestion. No private repo integration in v1. No external API stress testing in v1.
- **Note on tier boundaries (2026-05-02):** the public-GitHub-URL feature ships in the free tier in current build. The pricing/freemium boundary is under active review and not yet finalised. The free-tier feature set as documented here is a description of what ships, not a commitment that all listed features will remain free-tier post-pricing-decision. When the pricing architecture is decided, this section is reconciled.
- **Features:** Full stress testing across all five categories, conversational interface, diagnostic report, component library (stock profiles)
- **Data:** Interaction recorder with explicit consent, anonymized, feeds component library improvement
- **Cost to MAS:** Gemini Flash API cost absorbed, negligible at early volumes
- **Rate limiting:** Hashed machine ID to prevent abuse, no auth system

### FREEMIUM TIER
- **Distribution:** Account required. Stripe via ADGM.
- **LLM Backend:** Approved backends: Gemini 2.5 Pro, GPT-5, Claude Sonnet. BYOK supports any OpenAI-compatible endpoint. BYOK dual-pricing structure.
- **Pricing:** Monthly subscription with included runs + per-run overage. Exact pricing set post-build using real token consumption data. Tier-boundary decisions pending — see free tier note.
- **Languages:** Python AND JavaScript/Node.js
- **Additions over free tier (subject to tier-boundary review):**
  - Premium LLM-powered scenario generation and reporting (quality gate — same tool, better brain)
  - Passive security assessment layer (flags vulnerabilities, does NOT attack)
  - Richer diagnostic explanations (plain language fix categories, still no patches)
  - Structured config input alongside conversational interface
  - GitHub Action integration (stress test on push/PR) — note: CI gate endpoints currently ship in the open-source repo without a paywall guard; freemium boundary requires a guard layer to be added pre-billing-launch (tracked in known_issues.md)
  - "myCode tested" badge for repo READMEs
  - External API stress testing (real API calls under controlled escalating conditions)
  - Shared component library access (contribute data, get enriched library)
  - Docker containerization option for stronger isolation

### ENTERPRISE TIER
- **Distribution:** Organizational accounts. Custom pricing. Sales-led. Annual contracts.
- **LLM Backend:** Unmetered premium LLM. Dedicated MAS infrastructure.
- **Additions over freemium:**
  - Aggregate risk profiles across organizational repos (CISO/CTO dashboard)
  - Web interface (browser-based, no CLI required)
  - Team management with role-based access
  - Custom component library profiles (organization-specific dependencies)
  - Compliance and audit reporting
  - Orchestra integration (v3 roadmap — surfaces what the project could become)
  - Server-side execution in MAS-controlled containers (maximum isolation)
  - Priority support and SLA

---

## V1 Build Scope (Free Tier)

### Eight Core Components

NOTE 2026-05-02: this section reflects the original eight-component framing. The shipped product includes additional components not enumerated below — viability gate, prediction model (XGBoost), HTTP load driver, server manager, endpoint discovery, hysteresis, discovery logging, edition documents, web app. The package's `__init__.py` numbering scheme (C1–C9, D1–D3, E1–E3) extends this list. A full architecture rewrite of this section is tracked in known_issues.md as a post-HN follow-up. Until that rewrite, treat this section as descriptive of the original spec and consult the codebase for the complete component inventory.

#### 1. Session Manager
- Creates temporary virtual environment (venv) replicating the user's environment
- Reads user's Python version, installed packages, environment variables, dependency versions
- Creates venv, installs same dependencies, copies project working copy into venv workspace
- ALL stress tests run inside this venv with resource caps (memory ceiling, process limits, timeouts)
- Destroys venv and all temporary files on completion (including on crash, interrupt, or Ctrl+C)
- Signal handlers for interrupt signals
- Startup check that cleans orphaned temp environments from previous incomplete runs
- For JavaScript: equivalent isolation using temporary node_modules and sandboxed execution
- User's original files are NEVER touched
- User's host environment is NEVER at risk

#### 2. Project Ingester
- AST parsing for Python (Python ast module). JavaScript uses regex-based extraction (12+ patterns — weaker than Python path, known weakness)
- Dependency extraction from requirements.txt, package.json, package-lock.json
- Full dependency tree resolution (transitive dependencies, not just declared ones)
- Function flow mapping — how data and control flow between components
- Coupling point identification — where one component's failure cascades into another
- Version detection — reads installed versions, checks against latest stable via PyPI/npm registry
- Flags version discrepancies in report
- Handles partial parsing gracefully — "analyzed 12 of 15 files, 3 couldn't be parsed, here's why"
- Deterministic Python/JS code. No LLM dependency.

#### 3. Component Library
- Pre-built profiles for common vibe coding dependencies
- Each profile contains: identity, scaling characteristics, memory behavior, known failure modes, edge case sensitivities, interaction patterns, stress test templates
- Profiles are version-aware — document current stable version characteristics, flag when user's version differs

**Python profiles (18 files, 41 dependencies total across both languages):**
1. Flask
2. FastAPI
3. Streamlit
4. Gradio
5. Pandas
6. NumPy
7. SQLite3
8. SQLAlchemy
9. Supabase Python SDK
10. LangChain
11. LlamaIndex
12. ChromaDB
13. OpenAI SDK
14. Anthropic SDK
15. requests
16. httpx
17. Pydantic
18. os/pathlib

**JavaScript/Node.js profiles (23 files):**
1. React
2. Next.js
3. Express
4. Node.js core (fs, path, http)
5. Tailwind CSS
6. Three.js
7. Svelte
8. OpenAI Node SDK
9. Anthropic Node SDK
10. LangChain.js
11. Supabase JS SDK
12. Prisma
13. Axios/fetch
14. Mongoose/MongoDB driver
15. Stripe SDK
16. dotenv
17. Zod
18. Socket.io
19. Vue.js
20. Angular
21. D3.js
22. Chart.js
23. Puppeteer

**Profile generation and enrichment:**
- Source 1 (launch): LLM-generated from official documentation, known issues, Stack Overflow failure reports. Human-reviewed and corrected.
- Source 2 (pre-launch): Corpus mining — run myCode against public GitHub vibe-coded repos to discover real failure patterns and refine profiles. Profiles enriched with corpus_confirmed counts.
- Source 3 (post-launch): Interaction recorder anonymized data continuously enriches profiles.

**Unrecognized dependency handling:**
- Flag as untested in report
- Attempt generic stress testing based on ingester analysis of how the code uses it
- Log as candidate for future profile development

#### 4. Conversational Interface
- LLM-mediated (Gemini 2.5 Flash for free tier, premium LLM for freemium/enterprise)
- User describes in plain language: what the project does, who it's for, what conditions it operates under
- Two-turn structured exchange to extract operational intent (turn 1: present auto-detected analysis; turn 2: targeted questions about scale and data)
- User speaks in domain language, not engineering language
- User calibrates stress parameters in their own terms
- Presents generated stress scenarios for user review before execution
- User approves, calibrates, or adjusts before tests run

#### 5. Scenario Generator
- Core LLM layer. Takes ingester output + component library matches + operational intent.
- Tests dependency interaction chains as systems, not individual components in isolation
- Generates stress test configurations across categories:

**Shared (Python and JavaScript):**
  1. Data volume scaling — progressively larger inputs
  2. Memory profiling over time — repeated runs, track accumulation
  3. Edge case input generation — malformed, empty, unexpected type data
  4. Concurrent execution — multiple instances against shared resources

**Python-specific:**
  5. Blocking I/O under load
  6. GIL contention

**JavaScript-specific:**
  5. Async/promise chain failures under load
  6. Event listener accumulation (memory leaks)
  7. State management degradation in long-running apps

- LLM backend: Gemini 2.5 Flash for free tier (default). Freemium tier supports Gemini 2.5 Pro, GPT-5, Claude Sonnet per LLM API Architecture section below.
- Offline scenario generation path produces template-based scenarios when no LLM is available; see ADR-011 for dependency-free project handling in this path.

#### 6. Execution Engine
- Runs user's actual code inside the Session Manager's venv/sandbox
- Synthetic data generation based on scenario configurations
- Resource monitoring: memory, CPU, timing, process count
- Error capture: full traceback, error type, load level at failure
- Resource caps enforced: memory ceiling, process limit, timeout
- Controlled termination when caps exceeded — recorded as finding, not crash
- Handles user code crashes gracefully — catches, records, continues to next test
- Pure Python/Node.js. No LLM dependency.

#### 7. Report Generator
- LLM-powered (Gemini 2.5 Flash for free tier, premium LLM for freemium/enterprise)
- Takes raw execution data, produces plain-language diagnostic report
- Output: "Understanding Your Results" document (PDF via fpdf2, with markdown fallback if fpdf2 not installed). Optional JSON report (mycode-report.json) when --json-output flag is set.
- Per-finding fix prompts embedded in the document
- Degradation curves where relevant
- Identifies breaking points in terms the user understands based on their stated intent
- Flags version discrepancies found by ingester
- Flags unrecognized dependencies
- Reports dependency combination failures
- Does NOT prescribe fixes. Does NOT generate patches. Diagnoses only.
- Per ADR-012, baseline_status field will be added to the JSON report and a baseline framing line to the PDF/markdown header when the smoke-test stage ships.

#### 8. Interaction Recorder
- Explicit user consent required (opt-in, not opt-out)
- Stores anonymized: conversation, test configuration, results, dependency combinations encountered
- No personally identifiable information
- Feeds component library improvement
- Logs unrecognized dependencies for future profile development
- Logs failure patterns for scenario generator improvement

---

## Error Handling Philosophy

myCode NEVER shows raw errors to the user. Every failure is caught, translated into plain language, and either reported as a finding or reported as an operational issue with clear next steps.

1. **User's code crashes during stress test** — Expected finding. Catch, record, continue to next test.
2. **Venv creation fails** — Graceful message with explanation and fix steps.
3. **Ingester can't parse project** — Partial analysis, explain what couldn't be parsed, proceed.
4. **LLM API call fails** — Retry with backoff. Save work done so far even if report can't generate.
5. **Execution engine exceeds resource caps** — Controlled termination. Recorded as finding.
6. **myCode itself has a bug** — Cleanup routine runs regardless (finally block). Never leave environment worse.
7. **User interrupts (Ctrl+C)** — Signal handlers trigger cleanup. Startup check cleans orphans.
8. **Potentially malicious code detected** — Basic pre-execution scan, flag to user, proceed with restrictions.
9. **Project does not run cleanly at baseline (smoke test)** — Soft gate per ADR-012. Scenarios still run; report inherits baseline-failure framing.

---

## Security — Open Source Repository

- Sole merge authority (Manav only)
- Every PR reviewed line by line
- Branch protection: require review, signed commits, no force push to main
- Pin all dependency versions. No version ranges. Hash verification.
- pip-audit / npm audit for vulnerability scanning
- GitHub secret scanning and Dependabot alerts
- Signed releases
- MAS backend, premium LLM integration, freemium/enterprise logic NOT in public repo

---

## LLM API Architecture

### Architectural posture (provider-agnostic by design)
The backend abstraction (LLMBackend in scenario.py) is provider-agnostic by design. It speaks the OpenAI Chat Completions wire protocol via stdlib urllib.request and can call any HTTPS endpoint that follows that protocol — Gemini, OpenAI, Anthropic via OpenAI-compatible proxy, local Ollama, and so on. This openness is a feature: it is what makes BYOK work, what allows local SLM use, and what lets MAS swap backends per tier without code changes. The abstraction itself does not exclude any provider. Provider exclusion is enforced at the policy layer (see PROHIBITED below), not at the architectural layer.

### Components that call LLM:
- Conversational Interface (4) — YES
- Scenario Generator (5) — YES
- Report Generator (7) — YES
- All others — NO

### Free Tier Routing:
- Default (no key): myCode → MAS proxy → Gemini 2.5 Flash → return
- BYOK: User sets API key in config → myCode calls API directly via any OpenAI-compatible endpoint
- Auto-detection: check for local key at launch. Found → direct. Not found → MAS proxy.

### Freemium/Enterprise Routing:
- All calls through MAS backend → premium LLM API
- Token usage logged per user for billing

### Model Selection:
- Free: Gemini 2.5 Flash
- Freemium: TBD (quality testing across Gemini 2.5 Pro, GPT-5, Claude Sonnet)
- Enterprise: TBD (highest quality available)
- Local SLMs via BYOK (Ollama etc.) are supported by the architecture but unsuitable for Scenario Generator — requires genuine reasoning capability

### PROHIBITED:
- DeepSeek is prohibited as a chosen backend by MAS policy. The exclusion is a policy decision, not an architectural restriction — the LLMBackend abstraction can technically reach any OpenAI-compatible endpoint, including DeepSeek's. The policy is enforced via configuration defaults, documentation, and review discipline. The LLMBackend docstring will be updated to reflect this — tracked in known_issues.md.

---

## Key Architectural Decisions — Do Not Override

1. myCode creates a temporary venv/sandbox replicating the user's environment. ALL tests run inside this sandbox. User's host environment is NEVER at risk.
2. User's original files are NEVER touched. Working copy created by Session Manager.
3. myCode NEVER generates code patches or modifies user code.
4. v1 supports Python AND JavaScript/Node.js.
5. v1 supports Python AND JavaScript/Node.js across local projects and public GitHub URL ingestion. Private repo integration and external API stress testing are not in v1. Tier boundaries between free and freemium for these features are under review pending pricing-architecture decision.
6. The conversational interface is LLM-mediated — user speaks domain language, not engineering language.
7. Stress scenarios derived from intersection of user intent + parsed codebase + component library.
8. User calibrates stress parameters in their own terms.
9. Component library is LLM-generated then human-reviewed, enriched by corpus data.
10. Interaction data recorded ONLY with explicit consent, anonymized.
11. The report diagnoses — it does not prescribe.
12. Liability stays with the user at every point.
13. LLM calls confined to THREE components only. All others are pure Python/JS.
14. Human remains in directional role. AI executes and reports, does not steer.
15. Free tier is anonymous — no registration, no account, no email.
16. Cleanup runs ALWAYS — on success, failure, or interrupt.
17. Ingester checks dependency versions against latest stable, flags discrepancies.
18. Scenario generator tests dependency interaction chains as systems, not individual components.
19. Product integrity is non-negotiable — do not reduce stress parameters to fit infrastructure constraints.
20. A stress-testing tool cannot skip its own tests — fix first, then commit.
21. DeepSeek is prohibited as a chosen backend by MAS policy (architectural openness preserved per LLM API Architecture section).
22. Web interface is the product for target market. CLI is developer/power-user interface.
23. Active ADRs are authoritative on architectural decisions. CLAUDE.md and MEMORY.md reconcile to active ADRs, not the reverse.

---

## Liability Disclaimer
"myCode is a diagnostic tool. It does not guarantee code correctness, security, or fitness for purpose. All stress test results are informational. You are responsible for interpreting results and for all deployment decisions."

---

## Build Method
- Builder: Manav + Claude Code
- Languages: Python (tool + Python testing), JavaScript/Node.js (JS testing)
- Version control: GitHub with local backups
- Machine: MacBook Pro M4 Pro, 24GB RAM
- Claude Code: Opus 4.6 for development
- Safety: git commit before unsupervised Claude Code sessions; Time Machine backup before starting

---

## Current Build State
- Test baseline: 2,862 tests (2,851 passed, 10 skipped, 1 deselected — pre-existing test_empty_project_raises failure tracked in known_issues.md). Full suite wall-clock: ~52 minutes.
- E1-E4 constraint wiring: COMPLETE (commit 357c58c)
- prediction.py — corpus-backed failure prediction using XGBoost. Runs before stress tests to provide a predictive analysis panel in the web dashboard. Trained on corpus mining data. Model metrics (AUC, F1, recall) and target counts tracked in MEMORY.md.
- Web frontend: grouped intake form replacing sequential Q&A
- Endpoints: /api/submit-intent, /api/predict, /api/preflight (accepts public GitHub URLs), /api/analyze, /api/status/{job_id}, /api/report/{job_id}, plus admin and CI-gate endpoints
- OperationalConstraints fields: current_users, max_users, per_user_data, max_total_data, project_description

## Test Running Instructions

### Pre-commit verification (mandatory per discipline 3.10)
- **Full suite, no exclusions:** `pytest tests/ -x -q`
- NO `--ignore`, NO `-k`, NO `--deselect`, NO path arguments, NO `-m` exclusions
- ~52 minutes wall-clock acceptable
- If too slow for development iteration, REPORT and ASK before running a subset. Subset substitution without approval is a discipline 3.10 violation.

### Development iteration (subset use, with approval)
- Fast subset for in-development exploration: `pytest tests/ --ignore=tests/test_integration.py --ignore=tests/test_session.py --ignore=tests/test_pipeline.py -k "not (TestPipelineIntegration or TestCLIExitCode)"`
- Use `pytest -q --tb=short` to keep context window clean — only surface failures
- Slow tests (integration/session/pipeline) require venv creation, ~10min for that subset alone

---

## Working Disciplines (CC-binding)

These are the disciplines CC must observe at every session. They are summarised here; the full list and rationale are in MEMORY.md and the CC Session Workflow document.

- **3.1 — Strategic fidelity over scope-aggression.** "Ruthless" means strategic fidelity, not ruthlessness against scope creep.
- **3.2 — Session log discipline.** Single entry type, append-only, three lines per entry, file at `~/Desktop/mycode/session-logs/`, LOCAL-ONLY.
- **3.3 — Pre-commit scrutiny.** Explicit `git add` scope (never `git add .` or `git add -A`). Paste `git status` and `git diff --staged` for review before commit.
- **3.4 — CC instruction templates end with explicit commit-and-push instructions.**
- **3.5 — R is read-only on the repo until further notice.**
- **3.6 — Internal documents stay LOCAL.** ADRs (this document is a public reference; ADR drafts and working analysis are local-only), session logs, working notes, plan files.
- **3.7 — Document dating discipline.** `[topic]-[date]-[time].md` filename pattern, ISO date YYYY-MM-DD.
- **3.8 — No more handover documents from prior session.** Use continuity documents directly.
- **3.9 — User-facing path goes first.** Don't lead with methodological framings that delay user-experience problems.
- **3.10 — Full test suite required, no silent substitution.** `pytest -x -q` with NO exclusions for pre-commit verification. Subset substitution without explicit approval is a violation. If too slow, REPORT and ASK.
- **3.11 — CC reads CLAUDE.md, MEMORY.md, and known_issues.md at start of every action.** Both this self-reference and per-prompt reminders enforce.
- **3.12 — Branch operations require explicit announcement.** Any command that creates, deletes, or switches branches — local or remote — must be flagged explicitly before issuing, with: branch name, what HEAD it branches from, why it is being created or switched, and the lifecycle (when it merges, when it gets deleted). The `git checkout -b`, `git branch`, `git push --delete`, and `git branch -d/-D` commands are the visible markers. Burying branch operations in a checklist is a violation. Applies to Claude.ai issuing instructions, to CC writing or executing prompts, and to any future automation.
- **3.13 — File-system creation outside the repo root, or any new top-level directory under the user's home, requires explicit announcement and confirmation before the command is issued.** Complements 3.12. Operations covered include `mkdir` of a new parent directory, `mv` of a folder out of the repo, and any tool-side or CC-side directory creation. Applies symmetrically to Claude.ai-issued and CC-issued operations.
- **3.14 — Planning-folder location is `~/Desktop/mycode/myCode-plans/[YYYY-MM-DD]-[task-slug]/`.** PLAN.md and EXECUTION.md files for every CC session are written into this in-repo, gitignored directory. The dated subfolder is created per session; old subfolders are never deleted, accumulating as a genealogical record. CC must use this exact path. The earlier-drafted external `~/projects/myCode-plans/` location (from CC Session Workflow document v1) is superseded and must not be used.

The plan-review gate is non-negotiable: CC writes plan.md → Manav annotates → CC revises → Manav approves → CC implements. No implementation without plan review. See the CC Session Workflow document for the full session-orchestration protocol.

---

## Payment Infrastructure
- Stripe via ADGM
- Freemium tier only
- Account creation at freemium boundary only
- Token usage logged per user at MAS proxy
- Pricing set post-build from real token consumption data
- Tier boundaries currently under review (see free tier note above)

---

## License
- Business Source License 1.1
- Licensor: Manabrata Guha
- Change License: Apache 2.0 after four years
- No Additional Use Grant. All production use requires commercial license.

---

## Revision History

- v3.8 — 2026-05-02 (evening). Three disciplines added: 3.12 (branch operations require explicit announcement), 3.13 (filesystem creation outside repo root requires explicit announcement), 3.14 (planning-folder location is in-repo `~/Desktop/mycode/myCode-plans/[YYYY-MM-DD]-[task-slug]/`, gitignored). The 3.14 location supersedes the external `~/projects/myCode-plans/` location specified in CC Session Workflow v1; that document is revised. Trigger for these additions: Session 37 CC session for `test_empty_project_raises` fix wrote PLAN.md to the wrong external location (per the v1 workflow document), surfacing the gap.
- v3.7 — 2026-05-02. Sonnet/Opus model references corrected to Gemini 2.5 Flash. DeepSeek prohibition reframed as policy (not architectural) restriction. v1 LOCAL-ONLY claim updated to reflect shipped public GitHub URL ingestion; tier boundary deferred pending pricing decision. known_issues.md created and referenced. Disciplines 3.10 and 3.11 added explicitly. Test count updated from 2,570 to 2,862. ADR cross-reference added (decision 23). CI-gate-in-OSS-repo flagged as known issue requiring guard layer pre-billing-launch. JSON-for-Coding-Agent claim corrected to optional --json-output. Two-turn conversational interface clarified (was 3-5 minute exchange). ADR-011 (dependency-free project support) and ADR-012 (Stage 0 smoke test) referenced inline.
- v3.6 — 2026-04-10. Test counts and prediction.py module documented.
- Earlier versions in git history.
