# MEMORY.md — CC Knowledge Base
# Updated: April 6, 2026
# Read this at session start. Update at session end.

---

## Current State

- **Corpus:** 9,297 valid reports, wave 7 complete, mining done. Dedup policy: always deduplicate before mining.
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

---

## Current Task List (priority order)

1. Gemini integration — function-level code reading, line-specific fix suggestions, silent fallback
2. Canary automation — Railway cron job every 4 hours
3. Concurrent submission load test
4. Large repo test
5. Adversarial matrix — minimum 10 projects
6. Scope-gating — non-Python repo detection
7. README + HN post additions (edition tracking / no storage note)
8. Live model target count confirmation (40 after DROP)

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
