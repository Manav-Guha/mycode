# Corpus Collection Methodology — Reconstruction

**Date:** April 5, 2026
**Method:** Reconstructed from existing artifacts (scripts, manifests, logs, MEMORY.md, git history, corpus report contents). No prospective documentation existed.
**Corpus state at time of reconstruction:** 9,460 report directories, 9,297 parseable reports.

---

## 1. Source Artifacts Examined

### Mining Scripts
| File | Purpose | Git commits |
|------|---------|-------------|
| `scripts/repo_hunter.py` | GitHub search API repo discovery | 4 commits: c5a124a (initial), 6d067f9 (404 retry fix), dcba920 (auth header fix), 8e74f8c (gc.collect) |
| `scripts/batch_mine.py` | Clone + run myCode on discovered repos | 12 commits: b071fd3 (initial) through 47f16c4 (GitHub creation dates) |
| `scripts/lovable_scraper.py` | GPT-Engineer-App org scraper | Referenced in batch_mine commit d61b15f |
| `scripts/backfill_github_metadata.py` | Backfill GitHub metadata for repos missing it | Post-mining enrichment |
| `scripts/corpus_extract.py` | Extract and deduplicate failure patterns | Session 25 |
| `scripts/corpus_aggregator.py` | Aggregate reports across result directories | Session 23 |

### Wave Manifest Files (`corpus/discovered/`)
| File | Repos | Language breakdown |
|------|-------|-------------------|
| `discovered_repos_combined.json` | 2,062 | JS=1074, Python=667, Jupyter=132 |
| `discovered_repos_lovable.json` | 500 | JS=500 (all GPT-Engineer-App org) |
| `discovered_repos_lovable2.json` | 1,000 | JS=1000 (all GPT-Engineer-App org) |
| `discovered_repos_wave2.json` / `wave2_deduped.json` | 1,000 | Python=1000 |
| `discovered_repos_wave3_js.json` / `wave3_js_deduped.json` | 1,000 | TS=807, JS=193 |
| `discovered_repos_wave4_js.json` / `wave4_js_deduped.json` | 1,000 | TS=787, JS=213 |
| `discovered_repos_wave5_vibecoded.json` / `wave5_vibecoded_deduped.json` | 1,098 | TS=745, JS=153, Python=149 |
| `wave6_discovered.json` | 2,000 | Python=793, TS=738, JS=437 |
| `wave7_discovered.json` / `wave7_deduped.json` | 2,000 | Python=1287, Jupyter=188, HTML=188 |
| `discovered_repos_ai_ml.json` | 144 | Python=116, TS=10, HTML=9 |
| `discovered_repos_dashboards.json` | 148 | Python=112, Jupyter=21 |
| `discovered_repos_ecommerce.json` | 84 | Python=57, HTML=13, JS=10 |
| `discovered_repos_education.json` | 59 | Python=31, HTML=17 |
| `discovered_repos_fintech.json` | 141 | Python=47, TS=44, JS=33 |
| `discovered_repos_healthcare.json` | 131 | Python=97, Jupyter=30 |
| `discovered_repos_new.json` | 300 | Python=185, TS=45, Jupyter=37 |
| `discovered_repos_python2.json` | 62 | Jupyter=32, Python=27 |

### MEMORY.md Entries
- Session 25: Corpus extraction pipeline completed. 3,003 repos at that time.
- Session 26: Corpus state recorded as 3,003 repos, 414 unique patterns. MEMORY.md not updated after waves 3-7 expanded the corpus to 9,460.
- Critical policy: "Always deduplicate before mining" — documented after Lovable batches (1,500 repos, ~9 hours) were discovered to be entirely duplicates of already-processed repos.
- Wave 2 results explicitly recorded: 928 tested, 9 failed, 63 skipped.

### Log Files
- `batch_mine_log.txt` (root): 2,412 lines recording a run of 1,201 repos from `discovered_repos_combined.json`. 861 skipped as already-processed. 1,152 tested, 49 failed.
- `corpus_extraction/corpus_extraction_log.txt`: Scanned 6 legacy directories (results, corpus_results, corpus_results_retry, corpus_results_timeout, corpus_results_lovable, corpus_results_lovable2). 2,081 reports, 4,056 findings, 271 unique patterns.

### Legacy Results Directories
| Directory | Repo count | Status |
|-----------|-----------|--------|
| `results/` | 2,062 | Original combined run |
| `corpus_results/` | 200 | Early corpus run |
| `corpus_results_retry/` | 28 | Retries of failed repos |
| `corpus_results_timeout/` | 23 | Timeout retries |
| `corpus_results_lovable/` | 500 | Lovable wave 1 |
| `corpus_results_lovable2/` | 1,000 | Lovable wave 2 |
| `corpus_results_new/` | 8 | Small supplemental run |

Per MEMORY.md, these are redundant — `corpus/reports/` is the consolidated single source of truth.

---

## 2. Selection Logic By Wave

### Discovery Script Defaults (applies to all waves unless overridden)

From `repo_hunter.py`:
- **Default queries:** `"streamlit app"`, `"built with cursor"`, `"fastapi openai"`, `"crewai"`, `"langchain agent"`, `"flask chatbot"`, `"yfinance dashboard"`
- **Date filter:** `pushed:>={12 months ago}` appended to each query
- **Sort:** `updated` descending
- **Pagination:** 100/page, default 3 pages/query (max 10)
- **Basic filters:** not fork, not template, not archived, has detected language, min 0 stars (default)
- **Deep filters (per-repo API calls):** LOC >= 100, deps >= 3
- **Dedup:** by GitHub repo ID within a single run

From `batch_mine.py`:
- **Clone:** shallow (`--depth 1 --single-branch`), 120s timeout
- **Execution:** `python -m mycode <path> --offline --non-interactive --json-output --skip-version-check`, default 300s timeout
- **Cross-run dedup:** `~/.mycode/processed_repos.txt`
- **Output:** `{owner}__{repo}/mycode-report.json` + `github_metadata.json`

### Wave 1 — Initial Combined Run

**Manifest:** `discovered_repos_combined.json` (2,062 repos)
**Repos in corpus:** 1,660 unique
**Source:** `repo_hunter.py` with default queries. This is the earliest and largest single discovery run.
**Language mix:** JS=1074, Python=667, Jupyter=132, others scattered.
**Stars:** avg 75, indicating the default 0-star threshold was used.
**Result directories:** `results/` (2,062 dirs) + `corpus_results/` (200) + retry/timeout dirs (51).
**Log evidence:** `batch_mine_log.txt` records processing 1,201 repos from this manifest, 861 skipped as already-processed (indicating multiple batch_mine runs against this file).
**Selection logic not fully recoverable:** The specific `--queries` used for this run (if they differed from defaults) are not stored in the manifest file. The combined file may aggregate multiple repo_hunter runs with different query sets (it includes the vertical-specific files as subsets).

### Wave 1b — Vertical-Specific Discovery Files

**Manifests:** `discovered_repos_ai_ml.json` (144), `discovered_repos_dashboards.json` (148), `discovered_repos_ecommerce.json` (84), `discovered_repos_education.json` (59), `discovered_repos_fintech.json` (141), `discovered_repos_healthcare.json` (131)
**Total:** 707 repos across 6 verticals.
**Source:** Separate `repo_hunter.py` runs with vertical-specific `--queries` (queries not recorded in manifest files).
**Stars:** avg 0-11 for most verticals, 356 for ai_ml (suggesting different star thresholds or query nature).
**Relationship to combined:** These appear to be subsets included in `discovered_repos_combined.json`. Mapping confirms: all 707 repos from vertical files are contained in the combined file.
**Selection logic partially recoverable:** The filenames indicate the query themes (healthcare, fintech, etc.) but the exact `--queries` strings are not preserved.

### Wave 1c — Lovable Batches

**Manifests:** `discovered_repos_lovable.json` (500), `discovered_repos_lovable2.json` (1,000)
**Repos in corpus:** 429 + 868 = 1,297 total, but **0 unique** (all already in corpus from combined run).
**Source:** `lovable_scraper.py` targeting the `GPT-Engineer-App` GitHub organization.
**Language:** 100% JavaScript (Lovable generates JS/React apps).
**Stars:** all 0 (Lovable-generated repos are typically zero-star auto-generated repos).
**Critical note from MEMORY.md:** "The Lovable batches (1,500 repos, ~9 hours compute) were entirely duplicates." This led to the "always deduplicate before mining" policy. The Lovable repos were already in the corpus because they had been picked up by the general search queries or were already in the combined manifest.
**Selection logic fully recoverable:** `lovable_scraper.py` lists all repos in GPT-Engineer-App org, filters: not archived, not empty, not fork, has `package.json`, LOC >= 100, deps >= 3.

### Wave 2 — Python-Focused

**Manifest:** `discovered_repos_wave2.json` / `wave2_deduped.json` (1,000 repos)
**Repos in corpus:** 723 unique (not in prior waves)
**Language:** Python=1000 (100% Python).
**Stars:** avg 0 (zero-star threshold).
**MEMORY.md record:** "Wave 2 results: 928 tested, 9 failed, 63 skipped (already processed from crashed earlier run)."
**Selection logic partially recoverable:** The all-Python composition confirms a `language:python` search filter was used. The specific queries are not stored. The deduped variant exists, suggesting a dedup step was run against the existing corpus before mining.

### Wave 3 — JavaScript/TypeScript

**Manifest:** `discovered_repos_wave3_js.json` / `wave3_js_deduped.json` (1,000 repos)
**Repos in corpus:** 855 unique
**Language:** TypeScript=807, JavaScript=193.
**Stars:** avg 1.
**Selection logic partially recoverable:** The TS/JS composition confirms a JavaScript/TypeScript language filter. Aligns with MEMORY.md target: "Next: JS wave (React, Next.js, Express, Svelte, TypeScript)." Specific queries not stored.

### Wave 4 — JavaScript/TypeScript (continued)

**Manifest:** `discovered_repos_wave4_js.json` / `wave4_js_deduped.json` (1,000 repos)
**Repos in corpus:** 944 unique
**Language:** TypeScript=787, JavaScript=213.
**Stars:** avg 0.
**Selection logic partially recoverable:** Continuation of JS/TS wave. Different page offsets or query variants from Wave 3 (944 unique repos not in Wave 3 confirms different search results). Queries not stored.

### Wave 5 — "Vibe-Coded" Targeted

**Manifest:** `discovered_repos_wave5_vibecoded.json` / `wave5_vibecoded_deduped.json` (1,098 repos)
**Repos in corpus:** 855 unique
**Language:** TypeScript=745, JavaScript=153, Python=149.
**Stars:** avg 12.
**Selection logic partially recoverable:** The filename "vibecoded" and the mixed language composition suggest queries explicitly targeting vibe-coding markers. MEMORY.md growth strategy mentions: "platform-specific (.cursorrules, .replit)" and "Date-filtered queries (created:>2025-06-01)." The slightly higher average stars (12 vs 0-1 in other waves) suggests either a star threshold or naturally higher-star repos for vibe-coding keywords.

### Wave 6

**Manifest:** `wave6_discovered.json` (2,000 repos)
**Repos in corpus:** 1,260 unique
**Language:** Python=793, TypeScript=738, JavaScript=437.
**Stars:** avg 41 (highest of any wave).
**No deduped variant exists,** suggesting dedup was done differently or inline.
**Selection logic not recoverable:** No metadata in the manifest, no deduped file to indicate separate dedup step. The mixed-language composition and higher stars suggest broader queries or different star thresholds. Could be a combined run across multiple query sets.

### Wave 7 — Python-Focused

**Manifest:** `wave7_discovered.json` / `wave7_deduped.json` (2,000 repos)
**Repos in corpus:** 1,437 unique
**Language:** Python=1287, Jupyter=188, HTML=188.
**Stars:** avg 0.
**Selection logic partially recoverable:** Heavily Python. The Jupyter and HTML presence is atypical — GitHub's language detection classified these as Jupyter Notebook or HTML projects. The deep filter (deps >= 3) and LOC filter (>= 100) still apply. Specific queries not stored.

### Supplemental Files

**`discovered_repos_new.json`** (300 repos): Python=185, TS=45. Stars avg 169 (unusually high). Likely a targeted run with different query terms or a higher star threshold.
**`discovered_repos_python2.json`** (62 repos): Jupyter=32, Python=27. Stars avg 796 (very high). Likely a different search (e.g., starred Python data science repos). Small file, possibly manual curation.

### Unattributed Corpus Repos

**1,726 repos** in `corpus/reports/` are not present in any manifest file in `corpus/discovered/`. These likely came from:
1. Earlier batch_mine runs whose manifest files were not preserved
2. Legacy results directories (results/, corpus_results/, etc.) that were consolidated into corpus/reports/
3. Repos discovered via the processed_repos.txt dedup mechanism from runs whose manifests were later deleted

This represents 18.2% of the corpus with no recoverable provenance.

---

## 3. Corpus Composition (Observed)

All statistics computed from the 9,297 parseable reports in `corpus/reports/`.

### Language Distribution
| Language | Count | Percentage |
|----------|-------|------------|
| JavaScript | 5,012 | 53.9% |
| Python | 4,285 | 46.1% |

No other languages present. GitHub "TypeScript" repos are classified as `javascript` by myCode's ingester (TypeScript is processed as JS).

### Framework Distribution (Top 15)
| Framework | Count | % of corpus |
|-----------|-------|-------------|
| React | 3,364 | 36.2% |
| Next.js | 2,532 | 27.2% |
| Tailwind CSS | 2,215 | 23.8% |
| pandas | 1,841 | 19.8% |
| Streamlit | 1,705 | 18.3% |
| NumPy | 1,579 | 17.0% |
| requests | 1,220 | 13.1% |
| FastAPI | 1,009 | 10.9% |
| Pydantic | 986 | 10.6% |
| SQLAlchemy | 735 | 7.9% |
| Flask | 725 | 7.8% |
| Express | 707 | 7.6% |
| httpx | 631 | 6.8% |
| OpenAI SDK | 618 | 6.6% |
| Supabase | 571 | 6.1% |

### Top 20 Raw Dependencies (including UI component libraries)
| Dependency | Count | % |
|-----------|-------|---|
| react | 3,334 | 35.9% |
| react-dom | 3,323 | 35.7% |
| lucide-react | 2,386 | 25.7% |
| clsx | 2,048 | 22.0% |
| tailwind-merge | 1,978 | 21.3% |
| zod | 1,887 | 20.3% |
| class-variance-authority | 1,864 | 20.0% |
| pandas | 1,832 | 19.7% |
| streamlit | 1,702 | 18.3% |
| date-fns | 1,685 | 18.1% |
| @radix-ui/react-dialog | 1,650 | 17.7% |
| @radix-ui/react-slot | 1,649 | 17.7% |
| next-themes | 1,629 | 17.5% |
| @radix-ui/react-dropdown-menu | 1,592 | 17.1% |
| numpy | 1,579 | 17.0% |
| react-hook-form | 1,577 | 17.0% |
| @radix-ui/react-label | 1,546 | 16.6% |
| sonner | 1,531 | 16.5% |
| @radix-ui/react-separator | 1,530 | 16.5% |
| @hookform/resolvers | 1,526 | 16.4% |

**Note:** The Radix UI / shadcn / lucide-react cluster (>15% each) strongly indicates a large proportion of the JS corpus was generated by Lovable, Bolt, or v0, which use shadcn/ui as their default component library.

### Size Distribution (Lines of Code)
| Percentile | LOC |
|------------|-----|
| 10th | 221 |
| 25th | 509 |
| 50th (median) | 1,734 |
| 75th | 4,038 |
| 90th | 11,667 |
| Mean | 6,638 |

### Star Distribution (n=1,405 repos with star data)
| Percentile | Stars |
|------------|-------|
| 10th | 1 |
| 25th | 1 |
| 50th (median) | 1 |
| 75th | 3 |
| 90th | 21 |
| Mean | 151 |
| Max | 48,274 |

**Note:** 7,892 of 9,297 repos (84.9%) have no star data in github_metadata. Of those with data, 75% have ≤3 stars. The corpus heavily skews toward zero/low-star repos.

### Repo Age Distribution (n=3,962 repos with creation date)
| Percentile | Days | Approx. Months |
|------------|------|----------------|
| 10th | 11 | <1 |
| 25th | 18 | <1 |
| 50th (median) | 59 | ~2 |
| 75th | 619 | ~21 |
| 90th | 643 | ~21 |
| Mean | 245 | ~8 |

The bimodal distribution (50% under 2 months, then jump to 21 months at P75) suggests two populations: recently-created vibe-coded repos and older established repos that happened to match the search queries.

### Architectural Type Distribution
| Type | Count | % |
|------|-------|---|
| web_app | 5,793 | 62.3% |
| dashboard | 1,502 | 16.2% |
| api_service | 1,238 | 13.3% |
| utility | 295 | 3.2% |
| ml_model | 167 | 1.8% |
| data_pipeline | 102 | 1.1% |
| portfolio | 25 | 0.3% |
| chatbot | 16 | 0.2% |

### Business Domain Distribution (top 10)
| Domain | Count | % |
|--------|-------|---|
| general | 6,878 | 74.0% |
| data_science | 934 | 10.0% |
| ai_assistant | 599 | 6.4% |
| developer_tools | 459 | 4.9% |
| fintech | 192 | 2.1% |
| entertainment | 27 | 0.3% |
| education | 18 | 0.2% |
| healthcare | 16 | 0.2% |
| social_media | 8 | 0.1% |
| climate | 5 | 0.1% |

### Files Per Project
| Percentile | Files |
|------------|-------|
| 10th | 2 |
| 25th | 7 |
| 50th | 19 |
| 75th | 58 |
| 90th | 83 |

---

## 4. Known Limitations and Biases

### Selection Biases

1. **Query bias toward specific frameworks.** The default queries (`"streamlit app"`, `"fastapi openai"`, `"flask chatbot"`, `"langchain agent"`, `"yfinance dashboard"`, `"crewai"`) target specific Python frameworks. Projects using other frameworks (Django, Tornado, Bottle, aiohttp) are systematically underrepresented unless they appeared through vertical-specific or later wave queries.

2. **Lovable/shadcn UI monoculture in JS corpus.** The top 20 raw dependencies are dominated by shadcn/ui components (Radix UI, lucide-react, clsx, tailwind-merge, class-variance-authority). This indicates a large fraction of JS repos are Lovable/Bolt/v0-generated apps using the same template. This creates homogeneity: the JS corpus tests one UI stack repeatedly rather than diverse JS application patterns.

3. **Star threshold of 0 admits noise.** Most waves used min_stars=0 (default). 75% of repos with star data have ≤3 stars. Many zero-star repos are abandoned, broken, or tutorial forks. While this is arguably representative of vibe-coded output (most vibe-coded apps are low-star), it may include repos that were never functional.

4. **LOC minimum of 100 excludes micro-projects.** The deep filter requires 100+ LOC. Vibe-coded "hello world" or single-file scripts are excluded. These may represent a significant portion of AI-generated code that breaks in different ways than larger projects.

5. **Dependency minimum of 3 excludes simple projects.** The deep filter requires 3+ dependencies. Single-dependency projects (e.g., a pure pandas script, a pure Flask app) are excluded. These are common in vibe coding.

6. **Pushed-within-12-months filter creates recency bias.** The date filter (`pushed:>={12 months ago}`) excludes older repos. Combined with the low-star composition, this means the corpus captures recent, low-popularity, actively-pushed repos — heavily weighted toward 2025-2026 AI-generated code.

7. **GitHub search API ranking bias.** Results are sorted by `updated` descending, and pagination is limited to 3 pages (300 results) per query by default. This means only the most-recently-updated repos matching each query are discovered. Older-but-relevant repos are systematically missed.

### Populations Likely Underrepresented

| Population | Reason for underrepresentation |
|------------|-------------------------------|
| Django projects | No Django-specific search query in defaults |
| Mobile apps (React Native, Flutter) | No mobile-specific queries |
| CLI tools | Queries target web frameworks, not CLIs |
| Data pipelines (Airflow, Prefect, Luigi) | No pipeline-specific queries |
| Electron/desktop apps | No desktop-specific queries |
| Vue.js / Angular apps | No Vue/Angular queries; React/Next.js dominate JS results |
| Svelte apps | Present in framework list but only via late waves |
| Repos with <3 deps or <100 LOC | Excluded by deep filter |
| Private/enterprise repos | GitHub search only covers public repos |
| Non-GitHub repos (GitLab, Bitbucket) | repo_hunter.py uses GitHub API only |

### Failure Modes Likely Underrepresented

1. **Database-specific failures.** SQLAlchemy is at 7.9%, Prisma at ~3%, MongoDB (mongoose) at ~2%. Database-heavy apps are a minority, so deadlocks, connection pool exhaustion, and query scaling issues are underrepresented relative to their real-world frequency.

2. **Authentication/authorization failures.** The corpus has no auth-specific queries. Auth-related dependencies (passport, jwt, auth0) are not in the profiled dependency list. Auth failures under load are invisible.

3. **Multi-service/microservice failures.** The corpus is single-repo. Failures arising from service composition (cascading timeouts, circuit breaker failures) are not captured.

4. **Long-running process failures.** myCode runs with a 300-second timeout per repo. Failures that manifest only after sustained operation (memory leaks over hours, connection pool exhaustion over days) are truncated.

5. **Real API failures.** myCode runs `--offline`. Failures that only occur when real external APIs are called (rate limiting, auth expiry, payload size limits) are not captured.

---

## 5. Gaps in Reconstruction

1. **Wave 1 query terms not recoverable.** The `discovered_repos_combined.json` manifest does not store which queries generated which repos. If non-default `--queries` were used, they are lost. The vertical-specific files (ai_ml, dashboards, fintech, etc.) indicate themed queries were used, but the exact query strings are not preserved.

2. **Waves 2-5 query terms not recoverable.** The manifest files store only repo metadata (URL, stars, language, LOC, deps). The `--queries` argument used for each run is not logged in the output JSON.

3. **Wave 6 selection logic not recoverable.** No deduped variant exists. The mixed-language composition and higher average stars (41) differ from other waves but the cause (different queries, star threshold, or date filter) cannot be determined.

4. **1,726 unattributed repos (18.2% of corpus).** These exist in `corpus/reports/` but are not present in any manifest file in `corpus/discovered/`. They likely came from earlier mining runs whose manifests were not preserved, or from the consolidation of legacy results directories. Their selection logic is not recoverable.

5. **Run dates not systematically recorded.** The `batch_mine_log.txt` in the root directory records one run, but per-wave batch_mine logs are not preserved. The `mined_at` field in `github_metadata.json` records when each individual repo was processed, but the overall batch execution timeline is not recoverable.

6. **Dedup decisions not logged.** The `processed_repos.txt` dedup mechanism prevents re-mining, but does not record which repos were skipped in each run. The deduped manifest variants (wave2_deduped.json, etc.) indicate a dedup step occurred, but the repos removed are not logged.

7. **Lovable scraper query evolution.** The lovable_scraper.py scrapes the GPT-Engineer-App org, but it's not clear whether this org's repo count grew between the two Lovable waves. The MEMORY.md note that "Lovable batches were entirely duplicates" suggests the same repos were scraped twice.

8. **Star data incompleteness.** Only 1,405 of 9,297 repos (15.1%) have star counts in github_metadata. This may be because `backfill_github_metadata.py` was only run on a subset of the corpus, or because many repos returned 404 (deleted/private) when the backfill was attempted.

---

## Appendix: End-to-End Pipeline

```
repo_hunter.py / lovable_scraper.py
    ↓ discovered_repos_*.json (manifest)
    ↓
[manual dedup step against corpus/reports/]
    ↓ wave*_deduped.json
    ↓
batch_mine.py --input <manifest> --results-dir corpus/reports/ --timeout 300
    ↓ corpus/reports/{owner}__{repo}/mycode-report.json
    ↓ corpus/reports/{owner}__{repo}/github_metadata.json
    ↓
backfill_github_metadata.py  (fills gaps)
    ↓
corpus_extract.py  (dedup + rank failure patterns)
    ↓ corpus/extraction/corpus_patterns_ranked.json
    ↓
build_training_data.py  (feature extraction)
    ↓ src/mycode/data/training_data.csv
    ↓
train_prediction_model.py  (XGBoost training)
    ↓ src/mycode/data/prediction_model.joblib
    ↓ src/mycode/data/model_metadata.json
```
