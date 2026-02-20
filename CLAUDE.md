# myCode — Product Specification (CLAUDE.md)
# Version: 3.0 — February 20, 2026
# This is the authoritative build specification. Do not override architectural decisions without explicit human approval.
# Changes from v2.0: LLM backend updated based on token economics analysis. Architectural optimization applied to Conversational Interface. BYOK differential pricing added. DeepSeek replaced with Gemini Flash as free tier default.

---

## Parent Company
Machine Adjacent Systems (MAS)
Incorporation: ADGM (Abu Dhabi Global Market) — in progress, expected clearance within days.

## Product Position within MAS
myCode is MAS's consumer-facing product. It verifies AI-generated code behavior under real operational conditions. The underlying principle: take human intent, generate adversarial conditions, report the gap between intent and reality.

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
- **LLM Backend:** Gemini Flash via MAS proxy (anonymous, zero config, default) OR bring-your-own-key (Gemini/OpenAI/DeepSeek — any OpenAI-compatible API)
- **Registration:** None required. No account, no email, no payment.
- **Languages:** Python AND JavaScript/Node.js
- **Scope:** Local projects only. No remote repo integration. No API stress testing.
- **Features:** Full stress testing across all five categories, conversational interface, diagnostic report, component library (stock profiles)
- **Data:** Interaction recorder with explicit consent, anonymized, feeds component library improvement
- **Cost to MAS:** Gemini Flash API cost absorbed. Optimized architecture: ~$6-12 per 1,000 sessions. Negligible at early volumes. Gemini free tier (1,000 requests/day) may cover initial launch period.
- **Rate limiting:** Hashed machine ID to prevent abuse, no auth system

### FREEMIUM TIER
- **Distribution:** Account required. Stripe via ADGM.
- **LLM Backend:** Best available model via MAS backend, metered. Backend provider determined by quality testing (candidates: Gemini 2.5 Pro, GPT-5, Claude Sonnet). Quality of scenario generation is the deciding factor.
- **Pricing — two tracks:**
  - **MAS-provided model:** $25-30/month with 75-100 included runs + per-run overage ($0.15-0.30/run). MAS absorbs LLM cost.
  - **BYOK (Bring Your Own Key):** $10-15/month, user provides their own API key. No per-run LLM cost to MAS. User interface shows cost comparison transparently to encourage BYOK adoption.
- **Languages:** Python AND JavaScript/Node.js
- **Additions over free tier:**
  - Higher-quality scenario generation and reporting (quality gate — same tool, better brain)
  - Passive security assessment layer (flags vulnerabilities, does NOT attack)
  - Richer diagnostic explanations (plain language fix categories, still no patches)
  - Structured config input alongside conversational interface
  - GitHub Action integration (stress test on push/PR)
  - "myCode tested" badge for repo READMEs
  - External API stress testing (real API calls under controlled escalating conditions)
  - Shared component library access (contribute data, get enriched library)
  - Docker containerization option for stronger isolation

### ENTERPRISE TIER
- **Distribution:** Organizational accounts. Custom pricing. Sales-led. Annual contracts.
- **LLM Backend:** Unmetered premium model. Dedicated MAS infrastructure. Default: Claude Sonnet or Opus (premium positioning). BYOK supported for organizations with existing API contracts.
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
- AST parsing for Python (Python ast module) and JavaScript (Babel parser or TypeScript compiler API)
- Dependency extraction from requirements.txt, package.json, package-lock.json
- Full dependency tree resolution (transitive dependencies, not just declared ones)
- Function flow mapping — how data and control flow between components
- Coupling point identification — where one component's failure cascades into another
- Version detection — reads installed versions, checks against latest stable via PyPI/npm registry
- Flags version discrepancies in report
- Handles partial parsing gracefully — "analyzed 12 of 15 files, 3 couldn't be parsed, here's why"
- Deterministic Python/JS code. No LLM dependency.
- **Produces two outputs:** (a) full analysis (for Scenario Generator and Report Generator), (b) summarized analysis (~500 tokens, for Conversational Interface). Summary includes: project type, framework, key dependencies, entry points, identified risk areas. This separation is driven by token economics — see Token Optimization section.

#### 3. Component Library
- Pre-built profiles for common vibe coding dependencies
- Each profile contains: identity, scaling characteristics, memory behavior, known failure modes, edge case sensitivities, interaction patterns, stress test templates
- Profiles are version-aware — document current stable version characteristics, flag when user's version differs

**Python profiles (18):**
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

**JavaScript/Node.js profiles (18):**
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

**Profile generation method:**
- Source 1 (launch): LLM-generated from official documentation, known issues, Stack Overflow failure reports. Human-reviewed and corrected.
- Source 2 (pre-launch): Run myCode against public GitHub vibe-coded repos to discover real failure patterns and refine profiles.
- Source 3 (post-launch): Interaction recorder anonymized data continuously enriches profiles.

**Unrecognized dependency handling:**
- Flag as untested in report
- Attempt generic stress testing based on ingester analysis of how the code uses it
- Log as candidate for future profile development

#### 4. Conversational Interface
- LLM-mediated (Gemini Flash for free tier, best available model for freemium/enterprise)
- **Receives summarized ingester output (~500 tokens), NOT the full AST/dependency/flow analysis**
- **Receives NO component library profiles — those go to Scenario Generator only**
- User describes in plain language: what the project does, who it's for, what conditions it operates under
- **2-turn exchange to extract and confirm operational intent** (ingester already provides project structure; conversation confirms user context and priorities)
- User speaks in domain language, not engineering language
- User calibrates stress parameters in their own terms
- Presents generated stress scenarios for user review before execution
- User approves, calibrates, or adjusts before tests run

#### 5. Scenario Generator
- Core LLM layer. Takes **full** ingester output + component library matches + operational intent.
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

- Model selection: best available for the tier. Free tier uses Gemini Flash. Freemium/enterprise uses the highest-quality model available (determined by quality testing). Scenario generation is the most demanding LLM task — model quality matters most here.

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
- LLM-powered (Gemini Flash for free tier, best available model for freemium/enterprise)
- Takes raw execution data + **full** ingester output, produces plain-language diagnostic report
- Degradation curves where relevant
- Identifies breaking points in terms the user understands based on their stated intent
- Flags version discrepancies found by ingester
- Flags unrecognized dependencies
- Reports dependency combination failures
- Does NOT prescribe fixes. Does NOT generate patches. Diagnoses only.

#### 8. Interaction Recorder
- Explicit user consent required (opt-in, not opt-out)
- Stores anonymized: conversation, test configuration, results, dependency combinations encountered
- No personally identifiable information
- Feeds component library improvement
- Logs unrecognized dependencies for future profile development
- Logs failure patterns for scenario generator improvement

---

## Token Optimization (driven by economics analysis — see myCode-Pricing-and-Token-Economics.md)

The Conversational Interface (Call 1) is the largest token consumer because it historically received full ingester output and component library profiles, resent on every conversational turn.

**Optimization applied:**
1. Call 1 receives summarized ingester output (~500 tokens) instead of full analysis (~15,000+ tokens)
2. Call 1 receives NO component library profiles (saves ~8,000 tokens per turn)
3. Conversational exchange reduced from 4 turns to 2 turns (ingester provides structure; conversation confirms intent)
4. Full ingester output and component library profiles sent to Calls 2 (Scenario Generator) and 3 (Report Generator) only

**Result:** Total input tokens per session reduced from ~97,000 to ~38,400 (60% reduction). Cost per 1,000 sessions on Gemini Flash drops from ~$10.61 to ~$6.18.

**This does not change user experience.** The conversational interface still extracts the same operational intent — it just does so with a leaner context because the ingester has already provided the structural information.

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

---

## Security — Open Source Repository

- Sole merge authority (Manav only)
- Every PR reviewed line by line
- Branch protection: require review, signed commits, no force push to main
- Pin all dependency versions. No version ranges. Hash verification.
- pip-audit / npm audit for vulnerability scanning
- GitHub secret scanning and Dependabot alerts
- Signed releases
- MAS backend, model integrations, freemium/enterprise logic NOT in public repo

---

## LLM API Architecture

### Components that call LLM:
- Conversational Interface (4) — YES (receives summarized ingester output only)
- Scenario Generator (5) — YES (receives full ingester output + component library)
- Report Generator (7) — YES (receives full ingester output + execution results)
- All others — NO

### Free Tier Routing:
- Default (no key): myCode → MAS proxy → Gemini Flash API → return
- BYOK: User sets API key in config → myCode calls API directly (supports Gemini, OpenAI, DeepSeek, or any OpenAI-compatible endpoint)
- Auto-detection: check for local key at launch. Found → direct. Not found → MAS proxy.

### Freemium Routing:
- MAS-provided: All calls through MAS backend → best available model API. Token usage logged per user for billing.
- BYOK: User's key used directly. MAS charges platform fee only, no per-run LLM cost.

### Enterprise Routing:
- Default: All calls through MAS backend → premium model (Claude Sonnet/Opus or equivalent)
- BYOK: Supported for organizations with existing API contracts
- Token usage logged per organization for reporting

### Model Selection (v1):
- Free: Gemini Flash (default via MAS proxy)
- Freemium: Best available — determined by quality testing across Gemini 2.5 Pro, GPT-5, Claude Sonnet. Scenario generation quality is the deciding factor.
- Enterprise: Claude Sonnet or Opus (premium positioning), or customer's preferred model via BYOK

### Provider Strategy:
- No single-provider lock-in. Architecture supports multiple LLM backends via a common interface.
- Model selection per tier is a business decision, not an architectural constraint.
- BYOK ensures users are never locked to MAS's provider choice.

### Outstanding validation:
- Quality comparison needed: run identical scenario generation prompts across Gemini 2.5 Pro, GPT-5, and Claude Sonnet on real projects. Evaluate scenario quality, not just cost. This determines the freemium backend.

---

## Key Architectural Decisions — Do Not Override

1. myCode creates a temporary venv/sandbox replicating the user's environment. ALL tests run inside this sandbox. User's host environment is NEVER at risk.
2. User's original files are NEVER touched. Working copy created by Session Manager.
3. myCode NEVER generates code patches or modifies user code.
4. v1 supports Python AND JavaScript/Node.js.
5. v1 is LOCAL projects ONLY — no remote repo integration, no API stress testing.
6. The conversational interface is LLM-mediated — user speaks domain language, not engineering language.
7. Stress scenarios derived from intersection of user intent + parsed codebase + component library.
8. User calibrates stress parameters in their own terms.
9. Component library is LLM-generated then human-reviewed.
10. Interaction data recorded ONLY with explicit consent, anonymized.
11. The report diagnoses — it does not prescribe.
12. Liability stays with the user at every point.
13. LLM calls confined to THREE components only. All others are pure Python/JS.
14. Human remains in directional role. AI executes and reports, does not steer.
15. Free tier is anonymous — no registration, no account, no email.
16. Cleanup runs ALWAYS — on success, failure, or interrupt.
17. Ingester checks dependency versions against latest stable, flags discrepancies.
18. Scenario generator tests dependency interaction chains as systems, not individual components.
19. **Conversational Interface receives summarized ingester output only (~500 tokens). Full output goes to Scenario Generator and Report Generator. Component library profiles go to Scenario Generator only. This is a token optimization decision — do not revert without re-evaluating economics.**
20. **No single LLM provider lock-in. Architecture supports swappable backends. Provider choice per tier is a business decision.**
21. **BYOK is a first-class feature, not an afterthought. Users who bring their own key get a lower subscription price. Pricing transparency is a product value.**

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

## Payment Infrastructure
- Stripe via ADGM
- Freemium tier only (two tracks: MAS-provided model, BYOK)
- Account creation at freemium boundary only
- Token usage logged per user at MAS proxy (for MAS-provided model track)
- BYOK track: platform fee only, no token metering by MAS
- Pricing set post-build from real token consumption data and quality testing results

---

## Pre-Build Validation (completed)
Token consumption simulation conducted February 20, 2026. Results documented in myCode-Pricing-and-Token-Economics.md. Key findings:
- Unoptimized architecture: ~97,000 input tokens / ~11,100 output tokens per session
- Optimized architecture: ~38,400 input tokens / ~10,300 output tokens per session
- Free tier (Gemini Flash, optimized): ~$6-12 per 1,000 sessions — negligible, absorbable
- Freemium tier (Gemini Pro/GPT-5, optimized): ~$75-138 per 1,000 sessions — viable at $25-30/month subscription
- Freemium tier (Claude Sonnet, optimized): ~$270 per 1,000 sessions — viable only at $40+/month or with BYOK
- DeepSeek rejected as free tier default due to jurisdictional concerns (Chinese platform)
- Anthropic API payment issue unresolved — card accepted for Max subscription but rejected on Console
