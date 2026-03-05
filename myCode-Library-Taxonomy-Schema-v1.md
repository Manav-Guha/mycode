# myCode Library: Failure Taxonomy & Entry Schema
# Version 1.0 — March 5, 2026
# Companion to: myCode Library Design Note (March 2026)
# This document defines the classification system and data structure for all library entries.

---

## Build Principle

myCode is not a finished product, but a sophisticated and intelligent rendition of a critical and growing problem. Do not defer obvious capability. If the data exists to make a classification, make it. If the logic is straightforward, build it. The distinction from feature creep: feature creep adds scope; this principle says execute defined scope to its intelligent conclusion.

---

## Level 1: Failure Domains (8)

### 1. Resource Exhaustion
System runs out of a finite resource under sustained or increasing load.

**Level 2 patterns:**
- `unbounded_cache_growth` — cache never evicts, memory grows without limit
- `memory_accumulation_over_sessions` — repeated operations grow memory without release
- `large_payload_oom` — single large input exceeds available memory
- `connection_pool_depletion` — finite pool of connections/handles saturates
- `disk_write_accumulation` — temporary files or logs grow without cleanup
- `cpu_saturation` — compute-bound operation consumes all available CPU

### 2. Concurrency Failure
Code breaks when multiple operations execute simultaneously.

**Level 2 patterns:**
- `request_deadlock` — concurrent requests cause indefinite hang
- `shared_state_mutation` — concurrent access produces inconsistent results
- `race_condition` — outcome depends on unpredictable execution order
- `thread_pool_exhaustion` — all threads occupied, new requests blocked
- `gil_contention` — Python-specific: CPU-bound threads bottleneck on GIL

### 3. Scaling Collapse
Performance degrades non-linearly as load increases.

**Level 2 patterns:**
- `linear_to_exponential_transition` — acceptable until a threshold, then exponential degradation
- `response_time_cliff` — response time suddenly spikes at a specific load level
- `throughput_plateau` — system stops processing faster regardless of added resources
- `cascade_degradation` — one slow component makes the entire system slow

### 4. Input Handling Failure
Code crashes or produces incorrect results when receiving unexpected input.

**Level 2 patterns:**
- `unvalidated_type_crash` — wrong input type causes unhandled exception
- `empty_input_crash` — null/empty/missing input not handled
- `oversized_input_hang` — very large input causes indefinite processing or OOM
- `unsupported_input_format` — valid but untested format (e.g., handles PDF, crashes on DOCX)
- `format_boundary_failure` — closely related format fails (CSV works, TSV doesn't)
- `malformed_input_crash` — corrupted or partially valid input causes unhandled exception
- `encoding_failure` — unexpected character encoding crashes parser

### 5. Dependency Failure
Failure caused by the dependency itself — its version, API, or compatibility.

**Level 2 patterns:**
- `version_incompatibility` — code written for version A, running with version B
- `api_breaking_change` — dependency updated its API, existing code calls deprecated/removed methods
- `missing_transitive_dependency` — declared dependency installed but its own dependencies aren't
- `dependency_deprecation` — dependency no longer maintained, accumulating unpatched vulnerabilities
- `silent_behaviour_change` — dependency updated, same API but different behaviour

### 6. Integration Failure
Failure at the boundary between two components or dependencies interacting.

**Level 2 patterns:**
- `cascading_timeout` — one component timeout causes chain of failures downstream
- `state_desync` — two components disagree on shared state after concurrent operations
- `serialisation_mismatch` — data format between components doesn't match under stress
- `error_propagation_failure` — error in component A not properly caught/handled by component B
- `version_conflict_between_dependencies` — two dependencies require incompatible versions of a shared library

### 7. Configuration and Environment Failure
Failure in reproducing or operating within the project's stated environment.

**Level 2 patterns:**
- `dependency_install_failure` — declared dependency cannot be installed in test environment
- `transitive_dependency_conflict` — two dependencies require incompatible versions of a shared transitive dependency
- `platform_incompatibility` — dependency requires specific OS, architecture, or system library
- `runtime_service_unavailable` — dependency requires external service (database, API, message queue) not present
- `python_version_incompatibility` — dependency requires different Python/Node version than available

### 8. Unclassified
Failures that don't match the current taxonomy. Active holding pen with review mechanism.

**No predefined Level 2 patterns.** Entries sit here until reclassified.

**Review mechanism:**
- **Automatic flag:** 3+ entries sharing same dependency AND same error signature triggers review
- **Periodic scan:** After each corpus batch run, re-evaluate all unclassified entries against current taxonomy
- **Human-confirmed promotion:** Flagged entries reviewed by human, either assigned to existing domain/pattern or trigger creation of new Level 2 pattern
- **New domain evidence:** 10+ clustered entries outside all seven domains = evidence for a new Level 1 domain
- **Never auto-promote.** Human or verification team confirms all reclassifications.

---

## Level 2 Growth Protocol

Level 2 patterns are not fixed. New patterns are added when:
1. The discovery engine identifies a recurring failure not covered by existing patterns
2. A corpus batch reveals a cluster of similar failures with no matching pattern
3. The unclassified review mechanism identifies a new category

New patterns require:
- A clear name following the `snake_case_description` convention
- Assignment to exactly one Level 1 domain
- At least 3 observed instances across distinct projects
- A one-sentence description of the failure mechanism

---

## Entry Schema

### Mandatory (all entries):

| Field | Type | Description |
|-------|------|-------------|
| `entry_id` | UUID | Unique identifier |
| `source` | enum | `corpus_mining` or `runtime_discovery` |
| `source_batch` | string | Batch identifier (e.g., "L5_python_200", "lovable_js_200") |
| `mycode_version` | string | Version of myCode that produced the result |
| `timestamp` | datetime | When the entry was created |
| `language` | enum | `python` or `javascript` |
| `failure_domain` | enum | Level 1 domain (1-8) |
| `failure_pattern` | string | Level 2 pattern name (null for unclassified) |
| `scenario_name` | string | Specific scenario that triggered failure |
| `scenario_category` | string | data_volume_scaling, concurrent_execution, etc. |
| `operational_trigger` | enum | What usage pattern triggered this: `sustained_load`, `burst_traffic`, `long_session`, `large_input`, `concurrent_access`, `format_variation` |
| `affected_dependencies` | list | Dependencies involved |
| `severity_raw` | enum | `critical`, `warning`, `info` |
| `load_level_at_failure` | string | Step where failure manifested |
| `breaking_point` | string | Threshold label from degradation curve |
| `metric_name` | string | What was measured (execution_time_ms, memory_peak_mb) |
| `metric_start_value` | float | Value at lowest load |
| `metric_end_value` | float | Value at highest load |
| `multiplier` | float | Ratio of end to start |
| `codebase_origin` | enum | `github`, `lovable`, `bolt`, `cursor`, `unknown` |
| `vertical` | string | Auto-classified (V1-V8 per corpus mining architecture) |
| `architectural_pattern` | string | Auto-classified: `web_app`, `data_pipeline`, `chatbot`, `dashboard`, `api_service`, `portfolio`, `ml_model`, `utility` |

### Mandatory for corpus mining:

| Field | Type | Description |
|-------|------|-------------|
| `repo_url` | string | GitHub URL |
| `repo_last_commit_date` | date | Freshness of code when tested |
| `repo_stars` | int | Star count at time of mining |
| `repo_loc` | int | Lines of code |
| `repo_file_count` | int | Files analysed |
| `dependency_count` | int | Total declared dependencies |
| `profiled_dependency_count` | int | Dependencies with library profiles |
| `unrecognised_dependency_count` | int | Dependencies without profiles |

### Optional (runtime discovery context):

| Field | Type | Description |
|-------|------|-------------|
| `user_scale` | int | Stated concurrent users |
| `data_type` | string | Stated data type |
| `usage_pattern` | string | Stated usage pattern |
| `max_payload` | string | Stated maximum input size |
| `severity_contextualised` | enum | After applying user constraints |
| `within_stated_capacity` | boolean | Was the failure within stated operational limits |

### Optional (accumulates over time):

| Field | Type | Description |
|-------|------|-------------|
| `downstream_effects` | string | Cascading consequences observed |
| `resolution_pattern` | string | How the failure was addressed |
| `latency_cycles` | int | Test iterations before failure manifested |
| `confirmed_count` | int | Distinct projects reproducing this pattern |
| `first_seen` | datetime | First observation |
| `last_seen` | datetime | Most recent observation |

---

## Mapping from Existing JSON Reports

The existing `mycode-report.json` fields map to the schema as follows:

| JSON field | Schema field |
|------------|-------------|
| `project.language` | `language` |
| `findings[].severity` | `severity_raw` |
| `findings[].category` | `scenario_category` → derive `operational_trigger` |
| `findings[].affected_dependencies` | `affected_dependencies` |
| `findings[].load_level` | `load_level_at_failure` |
| `degradation_curves[].breaking_point` | `breaking_point` |
| `degradation_curves[].metric` | `metric_name` |
| `degradation_curves[].steps[0].value` | `metric_start_value` |
| `degradation_curves[].steps[-1].value` | `metric_end_value` |
| `project.files_analyzed` | `repo_file_count` |
| `project.total_lines` | `repo_loc` |
| `project.dependencies` | `dependency_count` |
| `unrecognized_dependencies` | `unrecognised_dependency_count` |
| `operational_context` | derive `vertical` and `architectural_pattern` |
| `model_used` | metadata, not in schema |

**Fields requiring new classifiers (to be built):**
- `failure_domain` — map from `scenario_category` + error type
- `failure_pattern` — map from scenario name + error signature
- `vertical` — classify from dependency list + file structure
- `architectural_pattern` — classify from framework + file structure
- `codebase_origin` — derive from batch metadata
- `operational_trigger` — derive from `scenario_category`

---

## Tiered Compute Model (Web Interface)

Progressive disclosure applied to compute time. Each tier creates demand for the next.

### Tier 1 — Instant (≤30 seconds)
- Static analysis only. No stress tests executed.
- Ingester runs, dependencies identified, library patterns matched.
- Report: "Based on your dependencies and code structure, here are known risk areas for projects like yours."
- Pure library inference — pattern matching against accumulated failure data.
- Always free. No LLM required.

### Tier 2 — Quick (3-7 minutes)
- Targeted stress tests on highest-risk patterns identified in Tier 1.
- 8-12 scenarios, priority-selected by library inference.
- Report: "We tested the areas most likely to cause problems. Here's what we found."
- Three free LLM-powered reports, then BYOK or subscription.

### Tier 3 — Comprehensive (15-30 minutes)
- Full scenario suite. Every profiled dependency, all coupling points, all categories.
- Report: complete diagnostic with all findings, degradation curves, confidence indicators.
- Async delivery for web interface — "We'll notify you when results are ready."
- Freemium and enterprise tier.

### Pricing alignment:
- Free tier: Tier 1 always + 3 free Tier 2 runs
- Freemium: Unlimited Tier 2 + Tier 3
- Enterprise: Priority compute + Tier 3 + historical comparison

---

## Future Direction (Post-Funding)

### Causal Modelling (CONDITIONS → FAILURE PATTERN → OUTCOMES)

The current schema captures outcomes (what failed) and operational triggers (what usage pattern was active). The full causal model adds:

- `architectural_vulnerability` — what code characteristic enabled the failure (no_eviction, shared_mutable_state, sync_in_async, no_validation, unbounded_allocation)
- `dependency_behaviour` — what the dependency did that contributed (caches_indefinitely, copies_on_operation, blocks_without_timeout, leaks_handles)
- `outcome_chain` — list of outcomes in causal order (e.g., memory_growth → oom_kill → session_loss → data_corruption)

This requires code analysis classifiers that identify architectural patterns in source code — not yet built. Design direction is documented here; implementation follows funding.

### Runtime Discovery Integration

The library receives data from two feeds:
1. **Corpus mining** — controlled, batch, systematic. Broad but shallow. Statistical authority.
2. **Runtime discovery** — uncontrolled, continuous, contextualised. Narrow but deep. Diagnostic authority.

Both feeds write to the same schema. Corpus mining provides pattern frequency. Runtime discovery validates which patterns matter in production. The two compound: corpus mining identifies what exists, runtime discovery confirms what's consequential.

### Inference Capability

A library structured for retrieval answers: "what does this failure historically mean?"
A library structured for inference answers: "given what we know, what failures does this codebase make likely that have not yet manifested?"

The taxonomy and schema are designed to support inference from the outset. The `operational_trigger`, `affected_dependencies`, `vertical`, and `architectural_pattern` fields enable matching new projects against historical outcome patterns. As the library grows from 400 to 1,000+ entries, inference queries become statistically meaningful.

---

*Machine Adjacent Systems (MAS) — ADGM, Abu Dhabi*
*This document is the authoritative classification system for the myCode component library.*
