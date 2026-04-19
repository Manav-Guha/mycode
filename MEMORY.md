# MEMORY.md — CC Knowledge Base
# Updated: April 6, 2026
# Read this at session start. Update at session end.

---

## Current State

- **Corpus:** 9,297 valid reports out of 9,449 directories (152 dirs have no report file, 0 corrupt). 15,552 total findings. Wave 7 complete, mining done. Dedup policy: always deduplicate before mining.
- **Model:** XGBoost, mean AUC 0.931, mean F1 0.392, mean recall 0.462, 40 targets in production (after dedup + 2 drops). Per-target thresholds calibrated via F1-optimized grid search. Class weighting applied.
- **Tests:** 2,570 passing, 5 skipped.
- **Infrastructure:** Railway Pro ("rare-tranquility", 4 vCPU / 8GB RAM). Vercel frontend. GitHub: github.com/Manav-Guha/mycode.git. PyPI: mycode-ai. Web: mycode-ai.vercel.app. Railway production URL: mycode-production-d3fa.up.railway.app.

---

## Recent Commits

- `c603eaf` — docs: corpus methodology, model diagnosis, model fixes
- `1484254` — fix: low-coverage warning (≤3 scenarios) + fast health endpoint
- `12db239` — fix: startup fix prompt, JS dep language detection, project name priority
- `7fce613` — fix: slow endpoint fix prompt + zip edition tracking
- `a5fa187` — fix: Flask --with-threads + edition-aware footer
- `f84978f` — feat: enrich 11 remediation patterns + 4 new corpus-backed patterns

---

## Known Open Issues

1. **Fix prompt pipeline enrichment** — ingestion data not yet flowing into prompts (installed_version, call graph, is_outdated flag).
2. **44.6% of corpus findings have failure_pattern=None** — fall through to generic fallback.
3. **Codex CLI broken on Darwin 25.3.0** — Rust panic in system-configuration.
4. **"Could not test:" label misleads on HTTP-deferred scenarios (2026-04-19)** — 6 of 7 "incomplete tests" in self-test carry failure_reason="http_tested" — they were deferred from the scenario engine to the HTTP phase, not failed. User-facing label "Could not test:" misleads readers into thinking myCode failed to test them when the HTTP phase actually did the work and produced findings. Clarity/labelling issue, not an engine bug. Log for Phase 8 narrative regeneration. Surfaced during Phase 2 verification run.
5. **Windows test failures at commit e6d7b08 (2026-04-19)** — R ran the full test suite on Windows; 8 tests failed (vs 1 on Mac at the same commit). All 8 are launch-blocking given cross-platform commitment. Classification:

   | # | Failure | Test file | Platform scope |
   |---|---|---|---|
   | 1 | test_pip_uses_venv_python | tests/test_session.py | Windows-specific (literal path strings) |
   | 2 | test_empty_project_raises | tests/test_pipeline.py | Cross-platform (stale test regex) |
   | 3 | test_compile_and_discover_ts | tests/test_js_module_loader.py | Likely cross-platform (Mac skipped) |
   | 4 | test_relative_parent_dir | tests/test_js_ingester.py | Windows-specific (JS module path resolution) |
   | 5 | test_sorted_by_probability | tests/test_intent_wiring.py | Windows-specific (prediction output ordering) |
   | 6 | test_loads_model | tests/test_intent_wiring.py | Windows-specific or R-environment-specific |
   | 7 | test_component_library_class_found | tests/test_integration.py | Windows-specific (class discovery) |
   | 8 | test_loader_imports | tests/test_integration.py | Windows-specific (import discovery) |

   **Priority order:** (1) Reactivate GitHub Actions Windows CI on a separate branch. (2) Bug #2 — test-only patch. (3) Bug #3 — add `--ignoreDeprecations 6.0` to `_compile_typescript()`. (4) Bugs #7+#8 — investigate together, likely shared root cause. (5) Bugs #1+#4 — path handling. (6) Bug #6 — reproduce on clean Windows first. (7) Bug #5 — investigation needed.

   Phase 4 narrative-layer work continues in parallel; these are a separate workstream.
   Reference: `mac_baseline_e6d7b08.txt` (Manav's Desktop), `Codex-WindowResults.docx` + `mycode-main-full-validation-report-2026-04-18.md` (R's Windows validation).

---

## Architecture Decision Records — Phase 4 Pattern Gates

Six ADRs covering the weak-gate patterns from phase4-pattern-audit.md are now Active as of 2026-04-19. They are the authoritative specification for Phase 4 implementation.

| ADR | Pattern | Gate summary |
|---|---|---|
| 001 | _pat_data_volume | peak_memory_mb ≥ 47 OR error_count ≥ 11 |
| 002 | _pat_cascading_timeout | error_count ≥ 3, reframed narrative with cascade vs single-point branching |
| 003 | _pat_unbounded_cache_growth | has_cache_decorator AND peak_memory_mb ≥ 50; fires only on post-71fa8e4 enriched runs |
| 004 | _pat_unvalidated_type_crash → renamed _pat_input_handling_failure | error_count ≥ 39 AND ≥1 of 7 exception markers; exception-type branching |
| 005 | _pat_flask_concurrency | load_level ≥ 5 AND non-null; 4-question user-comprehension narrative |
| 006 | _pat_requests_concurrent | load_level ≥ 2 AND error_count ≥ 91 AND ≥1 I/O marker |

Three blanket rules apply before any pattern-specific gate:
- Rule 1: no pattern fires on severity == "info"
- Rule 2: no pattern fires on _finding_type == "clean"
- Rule 3: concurrency patterns (flask_concurrency, requests_concurrent, cascading_timeout) do not fire at _load_level < 2; null load_level also does not fire for these patterns

The ADRs are stored at myCode-Architecture-Decision-Record.md in project files. They are living documents; revisions are tracked in each ADR's revision history. The threshold calibration mechanism they use is documented in myCode-Product-Architecture.md Sections 1 and 2.

Reference artifacts (not committed, working artifacts in repo root):
- phase4-pattern-audit.md — the original audit of weak gates
- phase4-corpus-distributions.md — the corpus distribution analysis that informed Method B thresholds

### ADR-004 pattern rename

_pat_unvalidated_type_crash is renamed to _pat_input_handling_failure at the narrative layer. The failure_pattern classifier label stays as "unvalidated_type_crash" in classifiers.py for corpus continuity. The pattern-dispatch function maps the classifier label to the renamed narrative function.

### Phase 4 implementation scope (next task)

Phase 4 implementation covers:
1. Threshold provider abstraction (per Product Architecture Section 1.10)
2. Blanket rule application before any pattern-specific gate
3. Gate logic for each of the six patterns per their ADRs
4. Narrative template updates per ADR reference text
5. Fix prompt template updates per ADR reference text
6. Pattern rename for ADR-004

Phase 4 does not commit until:
- Full unit test suite passes (any test asserting on old narrative/gate shape is updated to match new correct shape)
- Self-test completes against the new gates
- Side-by-side comparison of pre/post report output confirms only the expected changes

---

## Current Task List (priority order)

1. **Phase 4 implementation** — implement measurement-based gates for six patterns per the ADR spec in myCode-Architecture-Decision-Record.md. Next immediate task.
2. Gemini integration — function-level code reading, line-specific fix suggestions, silent fallback
3. Canary automation — Railway cron job every 4 hours
4. Concurrent submission load test
5. Large repo test
6. Adversarial matrix — minimum 10 projects
7. Scope-gating — non-Python repo detection
8. README + HN post additions (edition tracking / no storage note)
9. Live model target count confirmation (40 after DROP)

---

## HN Launch

Postponed to approximately April 16, 2026. 2-day buffer.

---

## Workflow Rules

- **Plan-review gate mandatory for all CC sessions.** CC writes plan.md → Manav annotates → CC revises → CC implements. Never skip.

---

## LLM Policy

- **DeepSeek permanently prohibited.** Do not use in any architecture, plan, code, or document.
- **Approved backends:** Gemini 2.5 Pro, GPT-5, Claude Sonnet. BYOK supports any OpenAI-compatible endpoint.

---

## Mining Restart Command

```bash
cd ~/Desktop/mycode && caffeinate -s python3 scripts/batch_mine.py --input corpus/discovered/wave7_deduped.json --results-dir corpus/reports --timeout 300 --report
```
