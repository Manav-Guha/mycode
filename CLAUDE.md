# myCode — Architectural Constraints & Product Boundaries (CLAUDE.md)
# Version: 3.3 — March 5, 2026
# This file contains ONLY what cannot be discovered from source code.
# Read source files directly for implementation details.
# Do not override architectural decisions without explicit human approval.

---

## Build Principle

myCode is not a finished product, but a sophisticated and intelligent rendition of a critical and growing problem. Do not defer obvious capability. If the data exists to make a classification, make it. If the logic is straightforward, build it. The distinction from feature creep: feature creep adds scope; this principle says execute defined scope to its intelligent conclusion.

---

## Session Protocol

1. Read MEMORY.md first
2. Read the latest session record in project knowledge
3. Check filesystem state before planning — do not infer from conversation history
4. Do not add functionality, abstractions, or interfaces beyond what is explicitly tasked
5. If a task requires supporting changes, list them before implementing
6. Maintain the passing test baseline (currently 1,686 passing, 0 failures)

---

## Parent Company

Machine Adjacent Systems (MAS)
Incorporation: ADGM (Abu Dhabi Global Market)

## Product Position

myCode is MAS's consumer-facing product. It verifies AI-generated code behaviour under real operational conditions. Take human intent, generate adversarial conditions, report the gap between intent and reality.

**Tagline:** "Built it with AI? Test it before it breaks."

---

## Architectural Decisions — Do Not Override

1. ALL tests run inside a temporary sandbox (venv/node_modules). User's host environment is NEVER at risk.
2. User's original files are NEVER touched. Working copy created by Session Manager.
3. myCode NEVER generates code patches or modifies user code.
4. v1 supports Python AND JavaScript/Node.js.
5. v1 is LOCAL projects ONLY — no remote repo integration, no API stress testing (yet — HTTP-level stress testing for web frameworks is on the roadmap).
6. The conversational interface is LLM-mediated — user speaks domain language, not engineering language.
7. Stress scenarios derived from intersection of user intent + parsed codebase + component library.
8. User calibrates stress parameters in their own terms.
9. Component library profiles are LLM-generated then human-reviewed.
10. Interaction data recorded ONLY with explicit consent, anonymised.
11. The report diagnoses — it does NOT prescribe. No fix suggestions. No patches.
12. Liability stays with the user at every point.
13. LLM calls confined to THREE components only: Conversational Interface, Scenario Generator, Report Generator. All others are deterministic.
14. Human remains in directional role. AI executes and reports, does not steer.
15. Free tier is anonymous — no registration, no account, no email.
16. Cleanup runs ALWAYS — on success, failure, or interrupt.
17. Ingester checks dependency versions against latest stable, flags discrepancies.
18. Scenario generator tests dependency interaction chains as systems, not individual components.

---

## LLM Backend Configuration

### Free Tier:
- Default (no key): Gemini Flash via MAS proxy (zero config)
- BYOK: User sets API key in config → myCode calls API directly
- Auto-detection: check for local key at launch. Found → direct. Not found → MAS proxy.
- Three free LLM-powered reports included before falling back to offline-only (implementation pending)
- **DeepSeek is NOT an option. Do not add DeepSeek references anywhere.**

### Freemium Tier (post-funding):
- Claude via MAS backend, metered via Stripe token billing
- Token usage logged per user

### Model Selection:
- Free: Gemini Flash
- Freemium: Claude Sonnet (Opus for complex Scenario Generator calls)
- Enterprise: Claude Opus

---

## Three-Tier Product Architecture

### FREE TIER
- Distribution: GitHub (open source), pip install, community channels
- Registration: None required
- Languages: Python AND JavaScript/Node.js
- Scope: Local projects only
- Features: Full stress testing, conversational interface, diagnostic report, component library
- Data: Interaction recorder with explicit consent, anonymised

### FREEMIUM TIER (post-funding)
- Account required. Stripe via ADGM.
- Claude-powered scenario generation and reporting
- Passive security assessment layer
- GitHub Action integration
- "myCode tested" badge
- External API stress testing
- Docker containerisation option

### ENTERPRISE TIER (post-funding)
- Organisational accounts. Custom pricing. Sales-led.
- Web interface (browser-based, no CLI required)
- Aggregate risk profiles across repos (CISO/CTO dashboard)
- Team management with role-based access
- Custom component library profiles
- Compliance and audit reporting

---

## Report Output Requirements

The report must:
- Answer "so what?" for every finding — consequences, not just facts
- Use user's constraint context: "you said 20 users, the app fails at 15" (critical) vs "the app fails at 200 users" (informational)
- Include confidence indicators per finding — note when sandbox limitations may affect accuracy
- Frame unrecognised dependencies positively: "tested 7/9 with targeted scenarios, 2 with usage-based analysis"
- Not parrot user input strings as project descriptions — interpret and summarise
- Distinguish environment errors (incomplete_tests) from project failures
- Not prescribe fixes

---

## Library Taxonomy & Classification

The component library uses a formal failure taxonomy defined in `myCode-Library-Taxonomy-Schema-v1.md` (project knowledge). Key points for CC:

- **8 failure domains:** Resource exhaustion, Concurrency failure, Scaling collapse, Input handling failure, Dependency failure, Integration failure, Configuration and environment failure, Unclassified
- **Every library entry must be classified** against this taxonomy on ingestion — not deferred
- **Auto-classifiers required:** vertical (from dependency + structure), architectural_pattern (from framework + files), failure_domain and failure_pattern (from scenario + error type), operational_trigger (from scenario category)
- **Unclassified entries** trigger automatic review at 3+ similar entries; human confirms promotion
- **The library is the moat.** The LLM is interchangeable. The library is not. Every design decision about the library is a decision about competitive durability.

---

## Tiered Compute Model (Web Interface)

Progressive disclosure — each tier creates demand for the next:

- **Tier 1 (≤30s):** Static analysis + library pattern matching. No tests executed. Always free.
- **Tier 2 (3-7 min):** 8-12 targeted stress tests on highest-risk patterns from Tier 1. Three free, then BYOK/subscription.
- **Tier 3 (15-30 min):** Full scenario suite. Async delivery ("we'll notify you"). Freemium/enterprise.

---

## Current Priorities (as of March 5, 2026)

### Critical Path (blocks beta distribution):
1. ~~Report quality rework~~ — DONE (Session 11)
2. ~~Harness syntax validation~~ — DONE (Session 11)
3. ~~Three free LLM reports~~ — DONE (Session 11)
4. Docker containerisation — `--containerised` flag for CLI + web backend. Non-negotiable before beta.
5. Library taxonomy classifiers — auto-classify vertical, architectural_pattern, failure_domain, failure_pattern, operational_trigger on every entry
6. L5 corpus migration — re-classify 168 repo JSON reports against taxonomy
7. HTTP-level stress testing — start actual server processes, send real HTTP requests
8. JS ingester upgrade — real parser (Babel/Acorn via Node subprocess)
9. Web interface — Vercel frontend + Railway backend + Docker + tiered compute
10. Report polish — collapse incomplete tests list, translate load levels to plain language
11. Rich terminal output — coloured output, progress bars
12. Untrusted code warning — CLI warning before dependency installation

### Secondary (after beta):
- Lovable corpus mining (200 JS repos — taxonomy applied from day one)
- Library expansion (scikit-learn, matplotlib, joblib, pillow profiles)
- Historical comparison / regression mode
- Freemium tier implementation (Stripe token billing)
- GitHub Action integration
- Conversational interface redesign (better questions, format-specific scenario generation)

---

## Product Vision — Post-Funding Architecture

### Discovery-Driven Decomposition (v2)

1. **Runtime discovery** — run code in a container, observe actual behaviour, map functional units and interactions
2. **Intelligent decomposition** — split into testable units along observed boundaries
3. **Dependency and invocation simulation** — simulate interfaces so each unit thinks it's in the complete system
4. **Static stress testing** — existing myCode categories against each unit independently
5. **Aggregation** — combine findings, flag cross-unit interaction risks

This enables:
- 100K+ LOC projects (never analyse whole codebase at once)
- Layer 2: intent-behaviour divergence detection
- General-purpose reliability testing beyond vibe coding (enterprise, legacy, migrations)

### Layer 2: Intent-Behaviour Divergence Detection

Compare runtime behaviour against user's stated intent. The gap is the divergence. This is "semantic conformance testing." The technical moat — requires solving runtime discovery, intelligent decomposition, and faithful interface simulation simultaneously.

---

## Security — Open Source Repository

- Sole merge authority (Manabrata only)
- Every PR reviewed line by line
- Branch protection: require review, signed commits, no force push to main
- Pin all dependency versions. No version ranges. Hash verification.
- MAS backend, Claude integration, freemium/enterprise logic NOT in public repo

---

## Error Handling Philosophy

myCode NEVER shows raw errors to the user. Every failure is caught, translated into plain language, and either reported as a finding or reported as an operational issue with clear next steps.

---

## Liability Disclaimer

"myCode is a diagnostic tool. It does not guarantee code correctness, security, or fitness for purpose. All stress test results are informational. You are responsible for interpreting results and for all deployment decisions."

---

## Build Method

- Builder: Manabrata + Claude Code
- Languages: Python (tool + Python testing), JavaScript/Node.js (JS testing)
- Machine: MacBook Pro M4 Pro, 24GB RAM
- PyPI: Published as `mycode-ai`, currently v0.1.2
- Codebase: ~/Desktop/mycode/
- LOC: 24,360 Python source lines, 41 profiles (18 Python + 23 JavaScript)
