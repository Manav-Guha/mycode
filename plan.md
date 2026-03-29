# Plan: L2 Prediction Model — Train XGBoost/LightGBM on Corpus Data

## Current State

**What exists:**
- `prediction.py` does corpus lookup — matches dependency names against `corpus_patterns_ranked.json`, scores by `overlap_count × confirmed_count`, returns top 5 as probability percentages.
- 3,990 corpus reports in `corpus/reports/`, each with `mycode-report.json` containing project metadata, findings, degradation curves.
- 3,677 reports have findings (92%), 313 have none.
- 271 patterns in `corpus_patterns_ranked.json`, 53 with ≥10 confirmed occurrences.
- 52 unique (category, failure_domain, failure_pattern) combinations.
- 41 profiled dependencies (18 Python + 23 JavaScript).
- Prediction box on the web frontend already renders probabilities — just needs better numbers.

**What's wrong with the current approach:**
- Probability = `confirmed_count / total_similar × 100` — this is a frequency ratio, not a true probability.
- No weighting for dependency *combinations* (Flask+SQLAlchemy together is different from Flask alone).
- No project complexity signal (LOC, file count, function count).
- No co-occurrence signal (if pattern A appears, pattern B is more likely).
- A trained model captures these interactions from the 3,990-project corpus.

---

## Architecture

```
corpus/reports/*/mycode-report.json
        │
        ▼
scripts/build_training_data.py  ──►  src/mycode/data/training_data.csv
        │
        ▼
scripts/train_prediction_model.py  ──►  src/mycode/data/prediction_model.joblib
                                        src/mycode/data/model_metadata.json
        │
        ▼
src/mycode/prediction.py  (loads model at import time, predicts on new projects)
```

The model file ships as package data alongside `corpus_patterns_ranked.json`. On Railway, the model is available at `src/mycode/data/prediction_model.joblib`.

---

## Phase 1: Feature Extraction Script

**File:** `scripts/build_training_data.py`

### Input
Read every `corpus/reports/*/mycode-report.json`.

### Feature Vector (per project)

**Dependency features (41 binary columns):**
One column per profiled dependency. Value = 1 if the project uses that dependency, 0 otherwise. Column names match profile filenames: `dep_flask`, `dep_pandas`, `dep_numpy`, etc.

Use a canonical mapping from `corpus_patterns_ranked.json` `affected_dependencies` and report `project.dependencies[].name` to the 41 profile names. Handle aliases (e.g. `react-dom` → `react`, `langchain-core` → `langchain`).

**Project complexity features (5 numeric columns):**
- `dep_count`: total number of dependencies (not just profiled)
- `loc`: total lines of code (`project.total_lines`)
- `file_count`: number of files analysed (`project.files_analyzed`)
- `files_failed`: number of files that couldn't be parsed
- `has_server_framework`: 1 if any of (flask, fastapi, express, streamlit, nextjs, gradio) is present

**Language feature (1 column):**
- `language`: 0 = python, 1 = javascript

### Target Labels (multi-label binary)

For each of the top ~20-30 failure patterns (those with ≥10 confirmed), a binary column: did this project exhibit this pattern?

Target column naming: `target_{sanitized_title}` where title is lowercased, spaces → underscores, special chars stripped.

Select target patterns by confirmed_count ≥ 10 AND not purely informational (skip "Application handled HTTP load without issues", "N unrecognized dependencies", "N missing dependencies").

**Filtering informational targets:**
- Skip patterns where 100% of severity_distribution is "info"
- Skip patterns whose title matches `r'^\d+ (missing|unrecognized) dependenc'`
- This leaves failure patterns that actually indicate problems

### Output
`src/mycode/data/training_data.csv` — one row per project, columns = features + targets.

Also output `src/mycode/data/target_columns.json` — ordered list of target column names, mapping each to its human-readable title, severity, and category. The model needs this at prediction time to label its outputs.

### Edge Cases
- Reports with no `project` key → skip.
- Reports with no `dependencies` → dep features all 0, keep the row (complexity features still informative).
- Reports with no findings → all target columns 0 (valid negative example).

---

## Phase 2: Model Training Script

**File:** `scripts/train_prediction_model.py`

### Approach: Binary Relevance with Gradient Boosting

Multi-label classification via one-vs-rest: train one binary classifier per target label, wrapped in `sklearn.multioutput.MultiOutputClassifier`. Each base estimator is a gradient boosting classifier.

**Model choice: scikit-learn `HistGradientBoostingClassifier`.**
- Native to scikit-learn (no extra install — already available in any Python 3.10+).
- Handles sparse features well (most dep columns are 0).
- Fast enough for 3,990 samples × 47 features × 20-30 targets.
- No need for XGBoost/LightGBM — HistGBT is scikit-learn's equivalent and avoids a dependency.

### Training Pipeline

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
import joblib

# Load training_data.csv
# Split into X (features) and Y (targets)
# Train
model = MultiOutputClassifier(
    HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42)
)
model.fit(X, Y)

# Evaluate: cross-validated predictions
Y_pred = cross_val_predict(model, X, Y, cv=5, method="predict_proba")
# Log per-target ROC-AUC, overall accuracy
```

### Evaluation Metrics
- Per-target ROC-AUC (skip targets with <5 positive samples — undefined AUC).
- Per-target precision, recall at 0.5 threshold.
- Overall mean AUC across viable targets.
- Log all metrics to stdout and to `src/mycode/data/model_metrics.json`.

### Acceptance Threshold
- Mean AUC > 0.60 across targets with ≥10 positive samples. This is a low bar — the corpus data is noisy (different projects, different dependency versions). Anything above random (0.50) is an improvement over the current frequency-ratio approach.

### Output Files
- `src/mycode/data/prediction_model.joblib` — serialised `MultiOutputClassifier`.
- `src/mycode/data/model_metadata.json`:
  ```json
  {
    "feature_columns": ["dep_flask", "dep_pandas", ..., "dep_count", "loc", ...],
    "target_columns": ["target_app_server_could_not_start", ...],
    "target_info": {
      "target_app_server_could_not_start": {
        "title": "Application server could not start",
        "severity": "critical",
        "category": "http_load_testing",
        "confirmed_count": 1218
      },
      ...
    },
    "training_samples": 3990,
    "mean_auc": 0.XX,
    "trained_at": "2026-03-29T..."
  }
  ```
- `src/mycode/data/training_data.csv` — retained for reproducibility (but NOT shipped as package data — too large).

### Dependency Column Alias Map

The dependency names in reports don't always match profile names exactly. Build an alias map:

```python
_DEP_ALIASES = {
    "react-dom": "react",
    "react-scripts": "react",
    "langchain-core": "langchain",
    "langchain-community": "langchain",
    "npm-start": None,  # not a real dep
    "uvicorn": "fastapi",  # co-occurs, map to framework
    ...
}
```

This map is used both in `build_training_data.py` and in `prediction.py` at prediction time. Define it once in `prediction.py` and import it from both scripts.

---

## Phase 3: Integration into prediction.py

### Changes to `predict_issues()`

```python
def predict_issues(dependency_names, corpus_path=None, constraints=None,
                   ingestion=None):
```

New optional parameter: `ingestion: Optional[IngestionResult]` — provides project complexity features (LOC, file count, etc.). When `None`, use defaults (median values from training data, stored in metadata).

### Flow

1. Try to load model + metadata from `src/mycode/data/`.
2. If model loaded:
   a. Extract feature vector from `dependency_names` + `ingestion`.
   b. Call `model.predict_proba(X)` → array of probabilities per target.
   c. Map target columns back to human-readable titles via metadata.
   d. Sort by probability descending, take top 5.
   e. Build `PredictionResult` with model-based probabilities.
3. If model NOT loaded (file missing, import error, deserialization error):
   a. Log warning once.
   b. Fall back to current corpus lookup (existing code, unchanged).
   c. Never crash.

### Feature Extraction Function

```python
def _extract_features(
    dependency_names: list[str],
    ingestion: Optional[IngestionResult] = None,
    feature_columns: list[str] = ...,
) -> list[float]:
    """Build feature vector matching the training schema."""
```

Uses the same alias map and profile list as the training script. Returns a list of floats in the same column order as `feature_columns` from metadata.

### Model Loading

Lazy-load on first call (module-level cache):

```python
_model_cache: dict = {}

def _load_model():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["metadata"]
    ...
    _model_cache["model"] = model
    _model_cache["metadata"] = metadata
    return model, metadata
```

### PredictionResult Changes

No structural changes. The `PredictionItem` fields stay the same:
- `title` — from `target_info` in metadata
- `probability_pct` — from `model.predict_proba()` × 100
- `severity` — from `target_info`
- `confirmed_count` — from `target_info`
- `matching_deps` — deps in the user's project that match the target's typical affected_dependencies
- `scale_note` — unchanged (computed from constraints, not model)

### `total_similar_projects`

Set to `metadata["training_samples"]` (3,990). This is the corpus size the model was trained on.

---

## Phase 4: Web Integration

No frontend changes needed. The prediction box already renders `PredictionResult`. The numbers just become more accurate.

One change: the prediction endpoint passes `ingestion` to `predict_issues`:

```python
# routes.py handle_predict
result = predict_issues(
    dependency_names=dep_names,
    constraints=job.constraints,
    ingestion=job.ingestion,  # NEW — provides complexity features
)
```

---

## Phase 5: Tests

### New tests in `tests/test_prediction_model.py`:

1. **Feature extraction:**
   - Known deps produce correct binary vector
   - Alias mapping works (react-dom → react)
   - Missing ingestion uses defaults
   - Ingestion data populates complexity features

2. **Model loading:**
   - Loads model from default path
   - Returns None gracefully when file missing
   - Caches on second call

3. **Fallback behavior:**
   - When model is missing, `predict_issues` returns corpus-lookup results (not empty)
   - When model is present, returns model-based results
   - Never raises exceptions

4. **Prediction output:**
   - Returns ≤5 predictions
   - Each has probability_pct between 0 and 100
   - Each has title, severity, confirmed_count
   - Results are sorted by probability descending

---

## Implementation Order

1. **Define alias map and profile list** in `prediction.py` (shared constants).
2. **Write `scripts/build_training_data.py`** — reads corpus, outputs CSV + target_columns.json.
3. **Run feature extraction** — verify CSV shape and target distribution.
4. **Write `scripts/train_prediction_model.py`** — trains model, outputs joblib + metadata.
5. **Run training** — verify AUC > 0.60, inspect metrics.
6. **Update `prediction.py`** — add `_extract_features`, `_load_model`, model-based prediction path.
7. **Update `routes.py`** — pass `ingestion` to `predict_issues`.
8. **Update `pyproject.toml`** — add `"data/*.joblib"` to package-data.
9. **Write tests** — feature extraction, loading, fallback, output validation.
10. **Run full test suite** — 2,373 existing + new tests.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/mycode/prediction.py` | Alias map, feature extraction, model loading, model-based predict path, fallback |
| `src/mycode/web/routes.py` | Pass `ingestion` to `predict_issues` in `handle_predict` |
| `pyproject.toml` | Add `"data/*.joblib"` to package-data |
| `scripts/build_training_data.py` | **NEW** — feature extraction from corpus |
| `scripts/train_prediction_model.py` | **NEW** — model training |
| `src/mycode/data/prediction_model.joblib` | **NEW** — trained model artifact |
| `src/mycode/data/model_metadata.json` | **NEW** — feature/target schema + metrics |
| `src/mycode/data/training_data.csv` | **NEW** — training dataset (not shipped) |
| `tests/test_prediction_model.py` | **NEW** — tests for model integration |

## Files NOT Modified

| File | Reason |
|------|--------|
| `web/app.js` | No frontend changes — prediction box already renders PredictionResult |
| `web/index.html` | No layout changes |
| `src/mycode/report.py` | No report framing changes |
| `src/mycode/constraints.py` | No constraint changes |
| `src/mycode/scenario.py` | No scenario changes |
| `corpus/` | Read-only — no corpus data modifications |

---

## Dependencies

**No new pip dependencies.** `scikit-learn` ships `HistGradientBoostingClassifier`, `MultiOutputClassifier`, and `joblib`. scikit-learn is a standard ML package available on any data science Python install.

However, scikit-learn is NOT in the current `pyproject.toml` dependencies. Two options:

**Option A (preferred):** Add `scikit-learn>=1.3.0` as an optional dependency group `[project.optional-dependencies] ml = ["scikit-learn>=1.3.0"]`. The training scripts require it. The `prediction.py` runtime import is guarded — if scikit-learn isn't installed, fall back to corpus lookup.

**Option B:** Add to core dependencies. This bloats the install for users who don't need ML predictions.

Go with Option A. The model file is pre-trained and shipped as package data. At prediction time, `prediction.py` needs `joblib` (to load the model) and `numpy` (for the feature array). Both are transitive deps of scikit-learn. If neither is installed, the corpus lookup fallback fires.

Guard the import:
```python
try:
    import joblib
    import numpy as np
    _HAS_ML = True
except ImportError:
    _HAS_ML = False
```

---

## Risk Areas

1. **Class imbalance:** Some targets have 1,000+ positives, others have 10. HistGBT handles this reasonably via its built-in loss function, but very rare targets (10-15 positives in 3,990 samples) will have poor AUC. Accept this — those targets fall back to the "worth knowing" tier anyway.

2. **Overfitting on small corpus:** 3,990 samples with 47 features is modest. Cross-validation (5-fold) mitigates this. `max_depth=4` and `max_iter=200` are conservative hyperparameters.

3. **Model file size:** HistGBT models are compact. ~20-30 targets × 200 trees × shallow depth = estimated 1-5 MB joblib file. Acceptable for package data.

4. **scikit-learn version skew:** A model trained with scikit-learn 1.5 may not load on scikit-learn 1.3. Pin a minimum version in the optional dep, and store the scikit-learn version in metadata.json for diagnostics.

5. **Alias map maintenance:** When new profiles are added, the alias map needs updating. Document this in a comment.
