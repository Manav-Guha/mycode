# MEMORY.md — CC Knowledge Base
# Updated: May 2, 2026 (evening)
# Read this at session start. Update at session end.
# Per discipline 3.11, read alongside CLAUDE.md and known_issues.md.

---

## Current State

- **Corpus:** 9,297 valid reports out of 9,449 directories (152 dirs have no report file, 0 corrupt). 15,552 total findings. Wave 7 complete, mining done. Dedup policy: always deduplicate before mining.
- **Model:** XGBoost, mean AUC 0.9307, mean F1 0.392, mean recall 0.462, 42 targets in production (after dedup + drops). Per-target thresholds calibrated via F1-optimized grid search. Class weighting applied. (CORRECTED 2026-05-02: prior values 0.931/40 targets were rounded/stale; the model_metadata.json file is the authoritative source.)
- **Tests:** 2,862 tests total. Last verified full-suite run: 2,851 passed, 10 skipped, 1 deselected (test_empty_project_raises pre-existing failure tracked in known_issues.md). Full-suite wall-clock: ~52 minutes.
- **Infrastructure:** Railway Pro ("rare-tranquility", 4 vCPU / 8GB RAM, single mycode service, empty volume, no DB). Vercel frontend. GitHub: github.com/Manav-Guha/mycode.git. PyPI: mycode-ai. Web: mycode-ai.vercel.app. Railway production URL: mycode-production-d3fa.up.railway.app.
- **Repo HEAD as of 2026-05-02 morning:** 4bc10bd (verify with `git log --oneline -5` if doubt; the world moves between sessions).

---

## Recent Commits

- `4bc10bd` — fix: preflight bug — _MIN_JS_SOURCE_FILES threshold lowered to 1 for projects without Python indicators (Saturday/Sunday late April)
- `2684f3a` — chore: CLAUDE.md reconciled with MEMORY.md, version bump to 3.6
- `c603eaf` — docs: corpus methodology, model diagnosis, model fixes
- `1484254` — fix: low-coverage warning (≤3 scenarios) + fast health endpoint
- `12db239` — fix: startup fix prompt, JS dep language detection, project name priority
- `7fce613` — fix: slow endpoint fix prompt + zip edition tracking

---

## Architecture Decision Records — full active set

The canonical ADR document is `myCode-Architecture-Decision-Records-v1_0-2026-05-01-with-ADR-012.md`. Twelve ADRs are Active as of 2026-05-01. ADRs are authoritative when they conflict with CLAUDE.md or MEMORY.md.

| ADR | Topic | Status | Date |
|---|---|---|---|
| 001 | `_pat_data_volume` threshold | Active | 2026-04-19 |
| 002 | `_pat_cascading_timeout` threshold | Active | 2026-04-19 |
| 003 | `_pat_unbounded_cache_growth` threshold | Active | 2026-04-19 |
| 004 | `_pat_input_handling_failure` threshold (renamed from `_pat_unvalidated_type_crash`) | Active | 2026-04-19 |
| 005 | `_pat_flask_concurrency` threshold (with 2026-04-19 amendment for low-load narrative branching) | Active | 2026-04-19 |
| 006 | `_pat_requests_concurrent` threshold | Active | 2026-04-19 |
| 007 | `_pat_fastapi_concurrency` | Active | 2026-04-19 |
| 008 | `_pat_streamlit_memory` (Awaiting Streamlit subset distribution from CC for final threshold) | Active | 2026-04-19 |
| 009 | `_pat_memory_baseline` (single-version with engine change) | Active | 2026-04-19 |
| 010 | `_pat_http_endpoint_blocking` | Active | 2026-04-19 |
| 011 | Dependency-free project support (Python and JavaScript) | Active | 2026-05-01 |
| 012 | Baseline correctness verification (Stage 0 smoke test) | Active | 2026-05-01 |

### Phase 4 implementation — gate summaries (ADRs 001–006)

| ADR | Pattern | Gate summary |
|---|---|---|
| 001 | _pat_data_volume | peak_memory_mb ≥ 47 OR error_count ≥ 11 |
| 002 | _pat_cascading_timeout | error_count ≥ 3, reframed narrative with cascade vs single-point branching |
| 003 | _pat_unbounded_cache_growth | has_cache_decorator AND peak_memory_mb ≥ 50; fires only on post-71fa8e4 enriched runs |
| 004 | _pat_input_handling_failure (renamed) | error_count ≥ 39 AND ≥1 of 7 exception markers; exception-type branching |
| 005 | _pat_flask_concurrency | load_level ≥ 2 (post-amendment); narrative branches at load_level=5 between low-load handler-blocking diagnosis and scale-level concurrency diagnosis |
| 006 | _pat_requests_concurrent | load_level ≥ 2 AND error_count ≥ 91 AND ≥1 I/O marker |

Three blanket rules apply before any pattern-specific gate:
- Rule 1: no pattern fires on severity == "info"
- Rule 2: no pattern fires on _finding_type == "clean"
- Rule 3: concurrency patterns (flask_concurrency, requests_concurrent, cascading_timeout, fastapi_concurrency) do not fire at _load_level < 2; null load_level also does not fire for these patterns

### ADR-007 through ADR-010 — Phase 4 extension

ADR-007 (`_pat_fastapi_concurrency`) introduces three-branch narrative on load_level and call_chain. ADR-008 (`_pat_streamlit_memory`) re-gates to `streamlit + unbounded_cache_growth + peak_memory_mb ≥ T` where T is `max(50, Streamlit-subset p25)` — pending CC's `phase4-streamlit-subset-distribution.md`. ADR-009 (`_pat_memory_baseline`) adds engine change to propagate flatness signal to Finding, plus three narrative branches (flat / not-flat / null). ADR-010 (`_pat_http_endpoint_blocking`) softens narrative from definitive "contains a blocking call" to correlational "likely contains a blocking operation."

### ADR-011 and ADR-012 — added 2026-05-01

**ADR-011 (Dependency-free project support):** Three coordinated changes to fix Python fatal error and JS viability rejection on dependency-free projects. (1) JS viability gate carve-out using `files_analyzed > 0` signal. (2) Generic scenario fallback in `_generate_offline` for `len(scenarios) == 0` case. (3) Top-level-only degenerate case via `_BODY_GENERIC` template. Three asserted premises pending PLAN.md verification before implementation. Scope confined to `viability.py` and `scenario.py`.

**ADR-012 (Baseline correctness verification, Stage 0 smoke test):** New pipeline stage between viability and scenario generation. Four-case taxonomy: script with `__main__`, web app import-and-assemble, library import, test suite (deferred to v2). Soft-gate failure semantics — scenarios still run on smoke failure, every finding inherits framing context. Standard 30s timeout, 10s for top-level-only sub-case. JSON `baseline_status` and `baseline_detail` top-level fields. Cross-references ADR-011 for top-level-only mechanism overlap.

### Phase 4 implementation queue (ADRs 005-amendment, 007, 008, 009, 010 plus 011, 012)

Seven items in implementation queue, none yet sent to CC:

1. ADR-005 amendment (low-load narrative branching for Flask)
2. ADR-007 (FastAPI concurrency three-branch narrative)
3. ADR-008 (Streamlit memory — requires CC's Streamlit subset distribution analysis first; threshold T finalisation blocks implementation)
4. ADR-009 (memory_baseline with engine change to http_load_driver.py and Finding dataclass)
5. ADR-010 (HTTP endpoint blocking narrative softening)
6. ADR-011 (dependency-free project support — three coordinated changes per the ADR; PLAN.md must verify three asserted premises before implementation)
7. ADR-012 (Stage 0 smoke test — new pipeline stage)

The CC Session Workflow document (separate file) governs the plan-review gate, dated-folder discipline, and post-session diff review for these implementations.

---

## Known Open Issues

The full known-issues list is in `known_issues.md`. The MEMORY.md summary below tracks the items most likely to affect a CC session's planning:

1. **`test_empty_project_raises` pre-existing failure** — sole non-skipped failure on Mac. Tracked.
2. **Lovable execution-environment failure** — surfaced during Saturday/Sunday testing; investigation pending.
3. **`_BODY_GENERIC` and `_BODY_DATA_VOLUME_SCALING` template behaviour for empty `_callables`** — asserted by CC during ADR-011 drafting; must be verified during PLAN.md before ADR-011 implementation.
4. **JS viability gate intent** — original `package.json contains no dependencies` rejection presumed designed for config-only-package case; ADR-011 implementation must verify this presumption via a code-history check before making the carve-out.
5. **`_BODY_GENERIC` `importlib.reload` path for top-level-only files** — presumed to handle the no-callables case; ADR-011 implementation must verify.
6. **`LLMBackend` docstring lists DeepSeek as supported provider** — contradicts MAS policy. CC task: edit docstring to remove DeepSeek from example list, add policy note.
7. **CI gate endpoints ship in OSS repo without paywall guard** — `/api/ci/check`, `/api/ci/result/{job_id}`, `/api/ci/override/{job_id}`, `/api/admin/ci-keys`. Pre-billing-launch task: add guard layer.
8. **README staleness** — corpus 6,000 (actual 9,297), profiles 30+ (actual 41), AUC 0.91 (actual 0.9307), targets 40 (actual 42), tests 2,600+ (actual 2,862). JS support described as "on the roadmap" — actual: full ingestion, execution, JS-specific scenarios shipped. README rewrite is a pre-HN-release task.
9. **Eight-component architecture in CLAUDE.md** is incomplete — viability gate, prediction model, HTTP load driver, server manager, endpoint discovery, hysteresis, discovery logging, edition documents, web app are shipped components not in the spec section. Full architecture rewrite tracked as post-HN follow-up.
10. **Two parallel prediction layers** (`inference.py` corpus lookup, `prediction.py` XGBoost) — unmanaged duplication or intentional tier separation. Decision pending.
11. **Repo-root cleanup** — `token_simulation.py`, `simulation_results.json`, `simulation_results.md` are unimported orphans. `/profiles/` at repo root may be vestigial alongside `/src/mycode/profiles/`. Pre-HN cleanup.
12. **Windows test failures at commit e6d7b08** — 8 tests failed on R's Windows run vs 1 on Mac. All 8 launch-blocking. Reactivate GitHub Actions Windows CI on a separate branch as priority. (Detail preserved from prior MEMORY.md.)
13. **Fix prompt pipeline enrichment** — ingestion data not yet flowing into prompts (installed_version, call graph, is_outdated flag).
14. **44.6% of corpus findings have failure_pattern=None** — fall through to generic fallback.
15. **Codex CLI broken on Darwin 25.3.0** — Rust panic in system-configuration.
16. **"Could not test:" label misleads on HTTP-deferred scenarios (2026-04-19)** — 6 of 7 "incomplete tests" in self-test carry failure_reason="http_tested" — they were deferred from the scenario engine to the HTTP phase, not failed. Clarity/labelling issue, not an engine bug. Log for Phase 8 narrative regeneration.
17. **Orphaned session directories blocking Phase 1 self-test verification** — original from Session 32. Status: unknown, may still be live.

---

## Current Task List (priority order — Friday May 2, 2026)

The Sunday priority list from Session 36 close is still active. None of those items has been closed. Work proceeds in the order below.

### Wednesday Priority 1 (carried over):
1. **CLAUDE.md and MEMORY.md updates** — IN PROGRESS today (2026-05-02). Includes disciplines 3.10/3.11, four factual corrections from clean-instance audit, ADR cross-reference, version bump.
2. **`.env` audit** — `cat ~/Desktop/mycode/.env` to verify credential exposure. 5 minutes.
3. **Session log flush** — T13–T74 still pending in v5 prompt at `CC-Prompt-Session35-Log-Flush-2026-04-26-v2.md` or similar. Run after .md updates.
4. **`known_issues.md` stub creation** — done as part of today's .md updates.

### Implementation queue (Sunday Priority 3-7):
5. **`test_empty_project_raises` hygiene fix** (Sunday Priority 3) — small commit, full-suite verification.
6. **Lovable execution-environment investigation** (Sunday Priority 4).
7. **A-prime cleanup elevation decision** (Sunday Priority 5).
8. **Item 76 continuation** (Sunday Priority 6) — Apps 2 and 3.
9. **Corpus-mining impact investigation** (Sunday Priority 7) — read-only.

### ADR implementation (Phase 4 extension + new):
10. ADRs 005-amendment, 007, 008, 009, 010, 011, 012 — seven CC-implementation items per the queue above.

### Operational / hardening:
11. SSD purchase initial backup — ordered, awaiting arrival.
12. 2FA audit across Railway/Vercel/GitHub/npm — postponed.
13. Railway API token audit — postponed.

### HN-prep (post-implementation):
14. README rewrite — staleness fix per known_issues.md item 8.
15. Eight-component architecture rewrite in CLAUDE.md (heavy-sync follow-up).
16. Hub71 deck updates — Anthropic COBOL/IBM, FT vibe coding, Boris Cherny interview.

### Strategic / longer arc:
17. R verification suite scoping document — drafted before R meeting.
18. ARR plan independent of HN outcomes.
19. Bangalore developer hire — post-funding.
20. ADGM company formation — post-HN.
21. MIAF/Rabdan Test Pilot — post-AVP conversation.

---

## HN Launch

Postponed since Session 32 pending fix completion. New target date not set as of 2026-05-02. Phases 2–9 of the original nine-phase fix plan have not started. README rewrite, eight-component architecture rewrite, and the seven-item ADR implementation queue are all pre-HN-release work.

---

## Workflow Disciplines (CC-binding)

The full discipline list is also in CLAUDE.md and the CC Session Workflow document.

- **Plan-review gate mandatory for all CC sessions.** CC writes plan.md → Manav annotates → CC revises → Manav approves → CC implements. Never skip.
- **3.10 — Full test suite required, no silent substitution.** `pytest -x -q` with NO exclusions for pre-commit verification. ~52 minutes wall-clock acceptable. Subset substitution without explicit approval is a violation. If too slow, REPORT and ASK.
- **3.11 — CC reads CLAUDE.md, MEMORY.md, and known_issues.md at start of every action.** Both CLAUDE.md self-references this AND per-prompt reminders enforce.
- **3.12 — Branch operations require explicit announcement.** Any command that creates, deletes, or switches branches — local or remote — must be flagged explicitly before issuing, with branch name, source HEAD, reason, and lifecycle. Burying branch operations in a checklist is a violation.
- **3.13 — File-system creation outside the repo root requires explicit announcement.** Any new top-level directory under user's home, any folder created outside `~/Desktop/mycode/`, requires announcement and confirmation before the command. Complements 3.12.
- **3.14 — Planning-folder location is `~/Desktop/mycode/myCode-plans/[YYYY-MM-DD]-[task-slug]/`.** PLAN.md and EXECUTION.md files for every CC session live there. In-repo, gitignored. Old dated subfolders accumulate as a genealogical record. The earlier external `~/projects/myCode-plans/` location is superseded.
- **3.3 — Pre-commit scrutiny.** Explicit `git add` scope (never `git add .` or `git add -A`). Paste `git status` and `git diff --staged` for review before commit.
- Earlier disciplines (3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9) are documented in CLAUDE.md.

---

## LLM Policy

- **DeepSeek prohibited as a chosen backend by MAS policy.** The exclusion is a policy decision, not an architectural restriction — the LLMBackend abstraction is provider-agnostic by design. The `LLMBackend` docstring will be updated to reflect this (tracked in known_issues.md).
- **Approved backends:** Free tier — Gemini 2.5 Flash. Freemium tier — Gemini 2.5 Pro, GPT-5, Claude Sonnet. BYOK supports any OpenAI-compatible endpoint.

---

## Mining Restart Command

```bash
cd ~/Desktop/mycode && caffeinate -s python3 scripts/batch_mine.py --input corpus/discovered/wave7_deduped.json --results-dir corpus/reports --timeout 300 --report
```

---

## Revision History

- 2026-05-02 (evening) — Disciplines 3.12 (branch announcement), 3.13 (filesystem-creation announcement), 3.14 (in-repo myCode-plans/ planning-folder location) added. CC Session Workflow document v1's external planning-folder location superseded by 3.14. Trigger: Session 37 CC session for test_empty_project_raises wrote PLAN.md to wrong external location, surfacing the gap.
- 2026-05-02 — Updated for Session 37: ADRs 011 and 012 added; ADR table extended to twelve entries; gate summaries refreshed; Phase 4 implementation queue documented; test count updated 2,570 → 2,862; AUC corrected 0.931 → 0.9307; targets corrected 40 → 42; known issues consolidated and pointer added to known_issues.md; current task list reset to Session 36+37 priorities; LLM policy reframed as policy-not-architecture per CLAUDE.md v3.7; disciplines 3.10/3.11 added.
- 2026-04-06 — Original.
