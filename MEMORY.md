# MEMORY.md — CC Knowledge Base
# Updated: March 6, 2026 (Session 12)
# Read this at session start. Update at session end.

---

## Session State

- **Test baseline:** 1,074 passing, 5 skipped, 0 failures. Do not break this.
- **PyPI:** Published as `mycode-ai` v0.1.1. Next publish: v0.1.2 after report rework + harness validation landed.
- **Profiles:** 18 Python + 23 JavaScript = 41 total. JS has 5 more than original spec (chartjs, google_auth_library, plotlyjs, react_chartjs_2, react_plotlyjs).
- **Corpus data:** 200 Python repos mined (168 successful, 84%). Lovable JS corpus run not yet started.
- **LOC:** 14,564 Python source lines (excludes profiles/tests).
- **Code analysis:** Full SWOT analysis completed March 4, saved at ~/Desktop/mycode/code-analysis.md

---

## Known Weaknesses (from SWOT — March 4, 2026)

These are confirmed by code analysis. Address in priority order:

1. **Report output is insufficient.** Summary is generic, parrots user input strings, no consequence analysis, no confidence indicators. Constraint contextualisation exists in code but report presentation doesn't use it well enough.
2. **No harness syntax validation.** Generated harnesses are string concatenation — edge cases produce invalid code reported as execution failures. Add pre-execution validation.
3. **No HTTP-level stress testing.** Web frameworks profiled but never tested with actual server + HTTP requests. Route handlers called directly, missing middleware, connection handling, serialisation overhead.
4. **JS ingester is regex-based.** 12+ regex patterns instead of real parser. Misses complex syntax, TypeScript constructs, JSX component trees. Python path uses real AST — JS path is substantially weaker.
5. **Offline scenario generation is template-only.** No novel scenarios without LLM. Significant quality gap between offline and LLM-powered runs.
6. **30-second per-scenario timeout is rigid.** No user-configurable timeout. Legitimate long-running operations may be terminated prematurely.
7. **No persistent state between runs.** No historical comparison, no regression detection.
8. **Windows resource limiting is POSIX-only.** Windows gets timeout-only enforcement — no memory ceiling or process limits.

---

## Confirmed Strengths (from SWOT — do not regress)

1. **Constraint extraction pipeline is real and functional.** User input flows through: natural language → constraint parsers → OperationalConstraints → scenario parameterisation → report contextualisation. This is NOT theatre — it works end-to-end.
2. **Graceful degradation across all modes.** Every LLM component has complete offline fallback.
3. **Session isolation is thorough.** Resource caps, signal handlers, orphan cleanup, guaranteed teardown.
4. **Token optimisation is implemented.** Summarised ingester output to Interface (~500 tokens), full output to Scenario Generator and Report Generator.
5. **Environment error routing.** ModuleNotFoundError/ImportError → incomplete_tests, not project failures.
6. **Discovery Engine creates data flywheel.** Local logging requires no consent. Confirmed patterns flagged for promotion.

---

## Critical Path — CC Session Sequence

### Session 11: COMPLETE
- Report quality rework: consequence analysis, constraint contextualisation, confidence indicators, auto-generated project descriptions, positive dependency coverage framing
- Harness syntax validation: Python via ast.parse, JS via balanced braces, failures reported as myCode limitations
- Three free LLM reports: counter in ~/.mycode/config.json, BYOK bypass, exhaustion forces offline mode
- Test baseline: 954 → 1,012 (+58 new tests). 0 failures.

### Session 12: COMPLETE
- Docker containerisation: `--containerised` flag runs myCode inside Docker container
- New module: src/mycode/container.py (Docker detection, image build, container run)
- Dockerfile in repo root (python:3.11-slim + Node.js, installs myCode from source)
- `--python-version` flag (default 3.11) for container Python version
- `--yes` / `-y` flag to skip confirmation prompts
- Untrusted code warning when running WITHOUT --containerised (skipped with --yes/--non-interactive)
- Container: read-only project mount, --network=none, --memory=2g, --cpus=2, auto-destroyed
- --json-output and --report not supported in containerised mode (read-only mount)
- Docker availability check with clear install instructions on failure
- Test baseline: 1,012 → 1,074 (+62 new tests). 0 failures.

### Session 13 (NEXT — Library taxonomy classifiers + L5 migration):
- Build auto-classifiers: vertical, architectural_pattern, failure_domain, failure_pattern, operational_trigger
- Classifier inputs: dependency list, framework detection, file structure, scenario category, error type
- Migrate 168 L5 repo JSON reports into new taxonomy schema
- Reference: myCode-Library-Taxonomy-Schema-v1.md in project knowledge

### Session 14:
- HTTP-level stress testing for web frameworks
- Server mode in harness generator: start server process, send real HTTP requests

### Session 15:
- Run 20-30 Lovable repos with current JS ingester (taxonomy applied from day one)
- Collect failure data on regex parser gaps
- Build real JS parser (Babel/Acorn via Node subprocess) targeting observed gaps

### Session 16:
- Web interface: Vercel frontend + Railway backend, Docker containers, tiered compute model
- Tier 1 (≤30s): static analysis + library pattern matching, no tests
- Tier 2 (3-7 min): targeted stress tests on highest-risk patterns
- Tier 3 (15-30 min): full suite, async delivery
- Non-interactive mode already exists — web backend wraps this

### Session 17:
- Report polish: collapse incomplete tests list, translate load levels to plain language
- Rich terminal output (rich library): coloured output, progress bars
- Conversational interface redesign: better questions, format-specific scenario generation

---

## Architectural Decisions That Affect Implementation

- **DeepSeek is NOT an option.** Do not add DeepSeek references anywhere. Free tier LLM is Gemini Flash.
- **Report does NOT prescribe fixes.** Diagnoses only. Users take findings to their coding tool.
- **Three free LLM reports** for free tier users before falling back to offline-only. Conversion trigger for freemium.
- **Web interface is the product** for target market (vibe coders). CLI becomes developer/power-user interface.
- **Unrecognised dependencies** should be framed positively: "tested 7/9 with targeted scenarios, 2 with usage-based analysis."
- **False positives:** Add confidence indicators per finding. Note sandbox limitations transparently. Let user calibrate trust.
- **Library taxonomy:** 8 failure domains, Level 2 patterns grow with data. Every entry classified on ingestion. See myCode-Library-Taxonomy-Schema-v1.md for full spec.
- **Auto-classifiers required:** vertical, architectural_pattern, failure_domain, failure_pattern, operational_trigger. Build these — don't defer.
- **Tiered compute model:** Tier 1 (≤30s, static + library lookup, free), Tier 2 (3-7min, targeted tests, 3 free then paid), Tier 3 (15-30min, full suite, async, freemium/enterprise). Each tier creates demand for the next.
- **Build principle:** Not a finished product, but a sophisticated and intelligent rendition. Don't defer obvious capability. If data exists to classify, classify. If logic is straightforward, build it.

---

## Product Context (not in source code)

- **Target users:** Non-technical "vibe coders" (Stage 3) who build with AI tools but lack operational mental models for testing.
- **Business entity:** Machine Adjacent Systems (MAS), incorporated in ADGM Abu Dhabi.
- **Funding target:** Hub71 accelerator → $1.5-3M seed. Hub71 provides credibility + network, actual raise from investors Hub71 connects to.
- **Collaborator R:** Handles go-to-market and education partnerships. Engineering background (MATLAB, C++). Confirmed Windows compatibility. Has developer contacts for beta testing, school networks in Abu Dhabi/Al Ain for hackathon distribution.
- **Adoption mechanism:** Only proven method is experiencing failure firsthand (hackathons, demos). Micro-equity incentives don't work.
- **Key investor terminology:** "Semantic conformance testing," "intent-behaviour divergence detection."
- **Market data:** 200 Python repos mined. Lovable: $300M ARR, 100K new projects/day. 46% of all new code is AI-generated. 92% of US devs use AI coding tools daily. AI coding market $81.4B → $127B by 2032.
- **Competitive intelligence:** Hacktron AI (security-focused, different market). No direct competitors in reliability testing for vibe-coded apps.

---

## File Locations

- CLAUDE.md: repo root (architectural constraints only)
- MEMORY.md: repo root (this file)
- Code analysis: ~/Desktop/mycode/code-analysis.md
- Corpus results: ~/Desktop/mycode/corpus_results/ (L5), corpus_results_retry/, corpus_results_timeout/
- Batch scripts: ~/Desktop/mycode/scripts/batch_mine.py, scripts/repo_hunter.py
- MIAF test project: ~/Desktop/miaf-tool/
- Session records: in Claude.ai project knowledge
- Market evidence: myCode-Market-Evidence-18-DataPoints.md in project knowledge

---

## Feature Creep Risk Areas

Do not expand beyond tasked scope in these areas:
- Scenario generation scope (don't add new categories without explicit approval)
- Report output expansion (don't add fix suggestions or code snippets)
- Profile enrichment (don't auto-generate profiles from discoveries without review)
- Ingester scope (don't add new file types or analysis methods beyond what's tasked)
