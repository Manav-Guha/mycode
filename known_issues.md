# known_issues.md — myCode Repo
# Created: May 2, 2026
# Read this at every CC session start (per discipline 3.11).
# This file is the live tracking of bugs, drift, and pending fixes that affect product output, user experience, or operational hygiene.
# Items here are not parked. Each must trace to a resolution.

---

## Active issues (as of 2026-05-02)

### Test-suite issues

(No active test-suite issues.)

### Documentation drift (caught by clean-instance audit, 2026-05-01)

2. **`LLMBackend` docstring lists DeepSeek as supported provider**
   - File: `src/mycode/scenario.py`, class `LLMBackend` docstring.
   - Symptom: docstring lists DeepSeek as one of the supported providers; CLAUDE.md and MAS policy prohibit DeepSeek as a chosen backend.
   - Source: clean-instance code review, 2026-05-01.
   - Resolution: edit docstring to remove DeepSeek from example list and add a one-line policy note. Architectural openness preserved (the abstraction can technically reach any OpenAI-compatible endpoint); the exclusion is policy.
   - Next action: CC task, low-risk single-file edit.

3. **README staleness**
   - File: `README.md` at repo root.
   - Symptoms (numbers below verified against actual code/data on 2026-05-01):
     - "6,000+ repos in the corpus" — actual 9,297 valid reports.
     - "30+ dependency profiles" — actual 41.
     - "0.91 mean AUC across 40 prediction targets" — actual 0.9307 mean AUC across 42 targets.
     - "2,600+ tests" — actual 2,862.
     - "JavaScript/TypeScript: Dependency detection and basic analysis in place. Full runtime testing on the roadmap" — actual: full ingestion (`js_ingester.py`, ~52K), full execution path (`js_stress_runner.js`), JS-specific scenario categories, ~6 JS-focused test files. JS support ships, not on roadmap.
     - README does not mention untrusted-code warning or dependency-install confirmation prompt.
   - Resolution: README rewrite. Pre-HN-release task.
   - Next action: scheduled before HN launch.

4. **CLAUDE.md eight-component architecture section is incomplete**
   - File: `CLAUDE.md` v3.7, "V1 Build Scope > Eight Core Components" section.
   - Symptom: enumerates eight components from original spec; shipped product includes additional components (viability gate, prediction model, HTTP load driver, server manager, endpoint discovery, hysteresis, discovery logging, edition documents, web app). Inline note added in v3.7 acknowledging this; full rewrite still pending.
   - Resolution: rewrite the architecture section to match the package's `__init__.py` numbering scheme (C1–C9, D1–D3, E1–E3) and the actual code structure.
   - Next action: post-HN follow-up, "heavy-sync" task.

### Architectural / product decisions deferred

5. **CI gate endpoints ship in OSS repo without paywall guard**
   - Files: `web/app.py` exposes `/api/ci/check`, `/api/ci/result/{job_id}`, `/api/ci/override/{job_id}`, `/api/admin/ci-keys`.
   - Symptom: CLAUDE.md describes GitHub Action integration as a freemium feature; the endpoints are fully implemented in the open-source repo with no auth or tier guard.
   - Resolution path: pre-billing-launch, add a guard layer that gates these endpoints behind freemium auth. Or, alternatively, keep them free-tier and update CLAUDE.md tier boundary documentation. The pricing-architecture decision affects which path is correct.
   - Next action: blocked on pricing-architecture decision (currently under review).

6. **Two parallel prediction layers** (`inference.py` corpus lookup, `prediction.py` XGBoost)
   - Files: `src/mycode/inference.py` and `src/mycode/prediction.py`.
   - Symptom: overlap in API surfaces; CLI tier-1 uses inference; web `/api/predict` uses prediction; `cli.py:_build_json_report` also calls prediction. Unclear if intentional tier separation or unmanaged duplication.
   - Resolution: review and decide. If intentional, document the tier separation. If duplication, consolidate.
   - Next action: review pending; not blocking HN release.

### Operational hygiene

7. **Repo-root cleanup — orphaned files**
   - Files at repo root: `token_simulation.py`, `simulation_results.json`, `simulation_results.md`.
   - Symptom: not imported anywhere in `src/`. One-off scripts left in place.
   - Resolution: delete or move to `scratch/` directory.
   - Next action: pre-HN cleanup, low-priority but visible to anyone browsing the repo.

8. **Repo-root `/profiles/` directory may be vestigial**
   - Files: `/profiles/` at repo root coexists with `/src/mycode/profiles/`.
   - Symptom: `pyproject.toml` `[tool.setuptools.package-data]` line points at the package one. Root `/profiles/` may be unused.
   - Resolution: investigate what (if anything) reads the root directory; if nothing, delete.
   - Next action: pre-HN cleanup investigation.

9. **Orphaned session directories blocking Phase 1 self-test verification**
   - Source: original from Session 32 (April).
   - Symptom: orphaned session directories under `~/tmp/` interfered with Phase 1 self-test.
   - Resolution: clean orphans; verify session-manager startup-cleanup logic actually fires. Status as of 2026-05-02: unknown, may still be live; check before any self-test run.
   - Next action: verify state on next access.

### Cross-platform

10. **Windows test failures at commit e6d7b08 (2026-04-19)**
    - Source: R's Windows full-suite run.
    - Symptom: 8 tests failed on Windows vs 1 on Mac at the same commit. All 8 launch-blocking given cross-platform commitment.
    - Tests:

      | # | Failure | Test file | Platform scope |
      |---|---|---|---|
      | 1 | test_pip_uses_venv_python | tests/test_session.py | Windows-specific (literal path strings) |
      | 2 | test_empty_project_raises | tests/test_pipeline.py | Cross-platform (stale test regex) — also issue #1 above |
      | 3 | test_compile_and_discover_ts | tests/test_js_module_loader.py | Likely cross-platform (Mac skipped) |
      | 4 | test_relative_parent_dir | tests/test_js_ingester.py | Windows-specific (JS module path resolution) |
      | 5 | test_sorted_by_probability | tests/test_intent_wiring.py | Windows-specific (prediction output ordering) |
      | 6 | test_loads_model | tests/test_intent_wiring.py | Windows-specific or R-environment-specific |
      | 7 | test_component_library_class_found | tests/test_integration.py | Windows-specific (class discovery) |
      | 8 | test_loader_imports | tests/test_integration.py | Windows-specific (import discovery) |

    - Priority order:
      1. Reactivate GitHub Actions Windows CI on a separate branch.
      2. Bug #2 — test-only patch (also covers issue #1).
      3. Bug #3 — add `--ignoreDeprecations 6.0` to `_compile_typescript()`.
      4. Bugs #7+#8 — investigate together, likely shared root cause.
      5. Bugs #1+#4 — path handling.
      6. Bug #6 — reproduce on clean Windows first.
      7. Bug #5 — investigation needed.
    - Reference: `mac_baseline_e6d7b08.txt` (Manav's Desktop), `Codex-WindowResults.docx` + `mycode-main-full-validation-report-2026-04-18.md` (R's Windows validation).

### Pipeline / engine

11. **Fix prompt pipeline enrichment incomplete**
    - Symptom: ingestion data not yet flowing into prompts (installed_version, call graph, is_outdated flag).
    - Resolution: wire ingestion data into prompt construction.
    - Next action: scheduled.

12. **44.6% of corpus findings have failure_pattern=None**
    - Symptom: large fraction of corpus findings fall through to generic fallback because the classifier did not assign a specific pattern.
    - Resolution: classifier improvement; out of scope for Phase 4.
    - Next action: post-Phase-4 investigation.

13. **"Could not test:" label misleads on HTTP-deferred scenarios (2026-04-19)**
    - Symptom: 6 of 7 "incomplete tests" in self-test carry failure_reason="http_tested" — they were deferred from the scenario engine to the HTTP phase, not failed. User-facing label "Could not test:" misleads readers into thinking myCode failed to test them when the HTTP phase actually did the work and produced findings.
    - Source: surfaced during Phase 2 verification run.
    - Resolution: label change at narrative-rendering layer.
    - Next action: log for Phase 8 narrative regeneration.

### Tooling

14. **Codex CLI broken on Darwin 25.3.0**
    - Symptom: Rust panic in system-configuration crate.
    - Source: external (Codex CLI bug).
    - Resolution: external; track upstream fix.

### Pending verification (ADR-driven)

15. **ADR-011 asserted premises — pending PLAN.md verification before implementation**
    - 15.1: JS viability gate `package.json contains no dependencies` rejection at `viability.py:357` is presumed designed for empty/config-only `package.json`, not legitimate dependency-free source. Carve-out is a refinement of original intent — must be verified via code-history check.
    - 15.2: `_BODY_GENERIC` and `_BODY_DATA_VOLUME_SCALING` engine harness templates are presumed to handle execution when `_callables` is empty. Must be verified before fallback ships.
    - 15.3: `_BODY_GENERIC` is presumed to handle the top-level-only path via `importlib.reload`. Must be verified.
    - Next action: covered by PLAN.md for ADR-011 implementation; CC must verify all three before proceeding.

16. **Lovable execution-environment failure**
    - Symptom: surfaced during R's Saturday/Sunday testing of Lovable.dev outputs.
    - Resolution: investigation pending.
    - Next action: Sunday Priority 4 — investigation queued.

---

## How this file is maintained

- New issues are added when surfaced — by audit, by user report, by CC investigation, by R testing, by self-test failure.
- Each item has a Resolution and a Next action.
- Items are not removed; they are marked **RESOLVED** with date and commit hash, and moved to a Resolved section at the bottom.
- This file is read at every CC session start (discipline 3.11).
- This file is reviewed at every session-end; new issues from the session are added before the session is logged.

---

## Resolved (chronological)

1. **`test_empty_project_raises` pre-existing failure** — RESOLVED 2026-05-02
   - File: `tests/test_pipeline.py`
   - Fix: test regex updated from `"Could not determine"` to `"doesn't appear to use a supported language"` to match production error message in this commit.
   - Source: surfaced in Session 35–36 full-suite verification (April 25).

---

## Revision history

- 2026-05-02 — Created. Initial population from clean-instance audit (2026-05-01), Session 36 known issues, prior MEMORY.md issue list, and ADR-011 asserted premises. Aligned with CLAUDE.md v3.7 and MEMORY.md update of same date.
