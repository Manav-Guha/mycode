# XGBoost Prediction Model — Diagnosis

**Date:** April 5, 2026

## Finding 1: Massive Target Label Coupling

**Root cause:** `_extract_targets()` in `build_training_data.py:161-169` had a fallback match: when title matching failed, it matched by `category + failure_domain + failure_pattern`. But 844 of 959 corpus patterns have **empty `failure_pattern`**. This meant the fallback reduced to `category + failure_domain`, collapsing all targets sharing the same category+domain into a single label.

**Evidence (before fix):**
- 7 targets with identical label vectors (839 positives each): all share `concurrent_execution + concurrency_failure` — e.g., "Concurrent Request Load (requests)", "Rate Limit Stress (openai)", "Rate Limit Stress (anthropic)" were indistinguishable
- 5 targets identical (2187 positives): all share `data_volume_scaling + scaling_collapse`
- 4 targets identical (286 positives): all share `edge_case_input + input_handling_failure`
- 28 of 49 targets (57%) were duplicates in 9 groups
- Only 30 of 49 reported AUC values were distinct

**Impact on reported metrics:** The mean AUC of 0.908 was inflated by counting the same model performance 7 times for the concurrent group, 5 times for the scaling group, etc. The 49 "targets" were really ~30 distinct prediction tasks.

**Second coupling mechanism:** Title word-overlap matching (`≥3 shared words`) caused e.g. "Concurrent Request Load (requests)" to match the same findings as "Concurrent Request Load (axios)" — the qualifier in parentheses was ignored.

## Finding 2: Zero-Precision Targets

**Root cause:** Severe class imbalance + 0.5 threshold.

For `target_version_compatibility_react` (40 positives / 9257 negatives):
- Maximum CV probability for any positive sample: **0.1352**
- 39 of 40 positive samples had probability < 0.10
- At threshold 0.5, zero true positives (AUC 0.89 but P/R/F1 all 0.000)
- Best possible F1 at threshold 0.05: only 0.247

For `target_connection_pool_stress_prisma` (39 positives):
- Maximum probability for any positive: **0.137**
- Essentially, the model learned to rank these correctly (AUC 0.76-0.89) but never pushed probabilities above 0.5 due to the extreme class ratio (~1:230)

**Key insight:** High AUC with zero precision at 0.5 is expected when the positive class is <0.5% of samples. The model correctly ranks positives above negatives (hence AUC > 0.85) but assigns them low absolute probabilities because the prior is so low. This is not a bug — it's a threshold calibration problem.

## Finding 3: Missing Features

The current feature set (56 features: 41 dep flags, 5 scalar, 1 language, 8 architecture, 1 repo age) captures **what dependencies are present** but not **how they're used**. Targets with low recall tend to be those where the failure mode depends on usage patterns, not mere presence:

- `Connection Pool Stress (prisma)`: Depends on whether the code uses concurrent queries, not just whether prisma is installed
- `Version Compatibility (react)`: Depends on which React APIs are used, not just react being present
- `Scenario failed: requests_large_download_memory`: Depends on whether large responses are consumed, not just whether requests is present

These would benefit from code-structure features (function count per dep, async usage patterns, call graph characteristics) not currently in the feature set.
