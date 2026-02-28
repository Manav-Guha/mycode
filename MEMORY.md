# MEMORY.md — myCode Project Knowledge Base
# Read this file at the start of every Claude Code session.
# Last updated: February 28, 2026 (Session 8)
# Maintainer: Manav Guha (sole developer)

---

## Working Relationship

Manav is the sole developer of myCode. He is not a software engineer by background — he builds exclusively through Claude Code collaboration. He is 8+ sessions deep. He knows the project state, the architecture, and the codebase.

**How to work with Manav:**
- Skip orientation. Don't offer option menus or ask "what would you like to work on?" — he will tell you.
- Read source files before planning. Don't infer limitations from CLAUDE.md alone — check the filesystem.
- Challenge architectural assumptions directly when warranted. He values honest criticism over agreement.
- Be specific and direct. No sugar-coating, no hedging, no sycophancy.
- When he says "do X", do X. Don't propose alternatives unless there's a concrete reason.
- He tracks commits, test counts, and pipeline state closely. Be precise with numbers.

---

## Project Identity

- **Product:** myCode — stress-testing CLI tool for AI-generated code
- **Company:** Machine Adjacent Systems (MAS), incorporating in ADGM Abu Dhabi
- **Tagline:** "Built it with AI? Test it before it breaks."
- **Package:** `mycode-ai` on PyPI
- **Repo:** `https://github.com/Manav-Guha/mycode`
- **License:** Not yet assigned
- **Target funding:** Hub71 accelerator (September 2026), seed round $1.5–3M

---

## Architecture Summary

myCode is **static stress analysis with micro-execution**, not load testing. It reads code statically, generates stress scenarios from component library profiles, executes harnesses testing individual functions/dependencies in isolation, and measures memory/timing. It does NOT spin up the app, does NOT send real HTTP requests, does NOT create actual concurrent connections.

**9 Core Components:**
1. Session Manager — orchestrates pipeline
2. Project Ingester — Python + JavaScript file/dependency analysis
3. Component Library — 36+ dependency profiles with scaling characteristics and failure modes
4. Conversational Interface — collects user intent via natural language (offline: keyword parsing)
5. Scenario Generator — creates stress test scenarios parameterised by user constraints
6. Execution Engine — runs harnesses, measures timing/memory, detects failures
7. Report Generator — LLM-powered (online) or template-based (offline) diagnostic reports
8. Interaction Recorder — opt-in anonymised data collection
9. Scenario Discovery Engine — compares actual results vs profile predictions, logs novel patterns

**Key architectural principle:** Zero third-party runtime dependencies. Python standard library only. Gemini Flash as default LLM backend. BYOK support for any OpenAI-compatible API.

---

## Current Engineering State (as of Session 8)

### What's Built and Working
- Full CLI: `mycode <path> --offline --non-interactive --json-output`
- Python AND JavaScript project support
- Constraint extraction from conversational interface (E1–E4 complete)
- Constraint-driven severity: within-capacity = CRITICAL, 1–3x = WARNING, beyond 3x = INFO
- Report contextualisation: "You said N users. This issue occurs at M sessions."
- Simulated concurrency language (honest framing, no false load testing claims)
- Corpus-aware findings ("In myCode's test portfolio, streamlit showed failures in 80% of 5 tested projects")
- JSON structured output (E6)
- Non-interactive mode (E11)
- Scenario Discovery Engine with logging to ~/.mycode/discoveries/ (L1)
- UTF-16/encoding-safe file reading across ingester and session
- 36 component library profiles (18 Python + 18 JavaScript)

### Test Counts
- ~900+ tests passing across all test files
- 55 discovery engine tests
- 147 report tests
- 87 library tests
- Pipeline tests exist but are slow (~2h+ for full suite)

### Scripts (internal tools, not shipped)
- `scripts/repo_hunter.py` — searches GitHub API for vibe-coded repos, filters, outputs discovered_repos.json
- `scripts/batch_mine.py` — clones repos, runs myCode, collects reports + discoveries, generates aggregate summary

### Validated Pipeline
The full corpus mining pipeline works end-to-end: repo_hunter → discovered_repos.json → batch_mine → per-repo reports + discoveries + aggregate summary. Tested against 3 repos (crypto-streamlit-app, therapeutic-trigger-tracker, Streamlit-Webmap-app). streamlit_cache_memory_growth appeared in all 3.

---

## Codebase Layout

```
~/Desktop/mycode/
├── CLAUDE.md                    # Authoritative product spec (v3.1)
├── MEMORY.md                    # This file
├── src/mycode/
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point, --offline, --non-interactive, --json-output flags
│   ├── session.py               # Session manager, pipeline orchestration
│   ├── ingester.py              # Python project ingestion
│   ├── js_ingester.py           # JavaScript project ingestion
│   ├── interface.py             # Conversational constraint extraction
│   ├── scenario.py              # Scenario generator (constraint-parameterised)
│   ├── engine.py                # Execution engine
│   ├── report.py                # Report generator (DiagnosticReport, as_dict() for JSON)
│   ├── recorder.py              # Interaction recorder
│   ├── discovery.py             # Scenario Discovery Engine (L1)
│   ├── pipeline.py              # Pipeline orchestration, wires discovery logging
│   └── library/                 # Component library loader + profiles
├── profiles/
│   ├── python/                  # 18 Python dependency profiles
│   └── javascript/              # 18 JavaScript dependency profiles
├── scripts/
│   ├── repo_hunter.py           # GitHub repo discovery (L3)
│   └── batch_mine.py            # Batch mining pipeline (L4)
├── tests/
│   ├── test_report.py           # 147 tests
│   ├── test_library.py          # 87 tests
│   ├── test_discovery.py        # 55 tests
│   ├── test_pipeline.py         # Pipeline tests (slow)
│   └── ...
└── examples/
```

---

## Test Portfolio (16 repos)

Tested against 16 real-world vibe-coded projects across 12 verticals:
- 729 stress test scenarios, 219 with issues (30%)
- 80% of projects had issues
- 18 critical failures, 4,911 total errors
- Languages: Python (8), JavaScript (8)
- myCode passed all 287 of its own scenarios clean

Key findings: Streamlit 72MB/user memory, Socket.io 100% failure rate, LangChain 83% failure rate, pandas 9x memory growth with financial data.

---

## Corpus Mining (3-repo validation complete)

### Validated Results
| Repo | Findings | Discoveries |
|------|----------|-------------|
| crypto-streamlit-app | 18 | 13 |
| therapeutic-trigger-tracker | 13 | 1 |
| Streamlit-Webmap-app | 6 | 2 |

Cross-project pattern confirmed: streamlit_cache_memory_growth in 3/3 repos.

### Corpus Sources
- GPT-Engineer-App org: 3,313 Lovable-generated repos (React+Vite+Supabase)
- lovable-ai topic: 97+ repos
- GitHub search: "streamlit app", "built with cursor", "fastapi openai", etc.
- Full architecture spec: myCode-Corpus-Mining-Architecture.md (in project knowledge)

---

## Known Issues and Gaps

1. **Pipeline tests are slow** (~2h+ for full suite). Run unit tests per module instead.
2. **Measured vs projected findings not separated** — report mixes actual execution data with linear projections. Post-funding refinement.
3. **Phase 1 only:** Static analysis + micro-execution. No running app, no real HTTP, no Docker containers. Phase 2 (E13, Oct 2026) adds external API stress testing.
4. **Component library needs expansion** — 36 profiles, target 56+ before Hub71. Top priorities: scikit-learn, plotly, matplotlib, uvicorn, cors, react-router-dom.
5. **Anthropic API payment issue** — card rejected on Console. Unresolved.

---

## Roadmap Position

### Completed (Sessions 1–8)
E1–E4 (constraint extraction), E5 (report bugs), E6 (JSON output), E11 (non-interactive), L1 (discovery engine), L3 (repo hunter), L4 (batch mine), 16-repo test portfolio, 3-repo pipeline validation, corpus mining architecture spec.

### Immediate (L5 pending)
- L5: 100+ repo overnight batch run (prerequisites all done, awaiting execution)
- L6–L8: Component library expansion + discovery promotion (after L5 data)

### Near-term
- L9: Failure signature schema
- L10–L12: Automated enrichment pipeline
- L13–L15: Corpus segmentation + quadrant model + benchmark quartiles
- L16: First reliability report (August target)

### Critical Path
L5 (batch run) → L6–L8 (library expansion) → L9 (signatures) → L13–L15 (benchmarks) → L16 (reliability report)

---

## Market Evidence (18 data points collected)

Key highlights:
- InfoWorld: "True bottleneck is validation, not code generation"
- Escape.tech: 2,000+ vulnerabilities across 5,600 vibe-coded apps
- Lovable EdTech breach: 18,697 users exposed, inverted auth logic
- Claude Code CVEs: RCE and API exfiltration via config files
- Moltbook: 1.5M API keys exposed via misconfigured Supabase (vibe coded)
- Columbia University: "Coding agents optimize for making code run, not making code safe"
- Anthropic Claude Code Security: 500+ zero-days found by reasoning (security, not reliability — complementary to myCode)
- DOD solicitation: AI coding tools for tens of thousands of developers, no verification layer mentioned
- Karpathy: wants to retire "vibe coding" for "agentic engineering"
- Gene Kim: writing *Vibe Coding* book (DevOps pioneer, enterprise reach)

Full evidence file: myCode-Test-Portfolio-Evidence-Summary.docx

---

## Key People

- **Manav Guha** — sole developer, works at Rabdan Academy (government agency), building myCode through Claude Code
- **R** — committed partner, go-to-market and education partnerships, engineering background (MATLAB/C++), developer contacts for beta testing
- **Marketing colleague** — experienced vibe coder, will assist with marketing and Arabic-language pitching to Abu Dhabi schools

---

## Session Conventions

- All development via Claude Code. No manual coding.
- Git workflow: commit after each task, push to origin/main.
- Session records maintained in project knowledge for continuity.
- MEMORY.md updated at end of each session with new state.
- When in doubt, check the filesystem — don't assume from docs alone.

---

## What NOT to Do

- Don't offer orientation frameworks or "what would you like to work on?" prompts
- Don't re-derive architectural decisions — CLAUDE.md is authoritative
- Don't propose alternative architectures without being asked
- Don't run the full pipeline test suite unless specifically requested (it takes 2+ hours)
- Don't add third-party runtime dependencies
- Don't suggest features that modify user code — myCode diagnoses only
- Don't conflate myCode's micro-execution with actual load testing
