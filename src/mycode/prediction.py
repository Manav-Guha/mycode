"""Predictive Analysis — corpus-based issue predictions for user projects.

Looks up ``corpus_patterns_ranked.json`` to find the most common failure
patterns for projects with a similar dependency stack.  Predictions appear
in the web UI *before* test results, giving the user early context.

No LLM dependency.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.constraints import OperationalConstraints, per_user_data_to_items

logger = logging.getLogger(__name__)

# ── Profiled Dependencies & Alias Map ──
# When new profiles are added, update both lists.

PROFILED_DEPS: list[str] = [
    "anthropic", "anthropic_node", "axios", "chartjs", "chromadb",
    "dotenv", "express", "fastapi", "flask", "google_auth_library",
    "gradio", "httpx", "langchain", "langchainjs", "llamaindex",
    "mongoose", "nextjs", "node_core", "numpy", "openai",
    "openai_node", "os_pathlib", "pandas", "plotlyjs", "prisma",
    "pydantic", "react", "react_chartjs_2", "react_plotlyjs",
    "requests", "socketio", "sqlalchemy", "sqlite3", "streamlit",
    "stripe", "supabase", "supabase_js", "svelte", "tailwindcss",
    "threejs", "zod",
]

# Map raw dependency names (from reports / ingestion) to profile names.
# None → ignore (not a real dep).  Missing key → keep original.
DEP_ALIASES: dict[str, Optional[str]] = {
    "react-dom": "react",
    "react-scripts": "react",
    "react-router-dom": "react",
    "langchain-core": "langchain",
    "langchain-community": "langchain",
    "langchain-openai": "langchain",
    "langchain-anthropic": "langchain",
    "@langchain/core": "langchainjs",
    "@langchain/openai": "langchainjs",
    "npm-start": None,
    "uvicorn": "fastapi",
    "gunicorn": "flask",
    "next": "nextjs",
    "next-themes": None,
    "python-dotenv": "dotenv",
    "@supabase/supabase-js": "supabase_js",
    "supabase": "supabase",
    "chart.js": "chartjs",
    "d3": None,
    "three": "threejs",
    "socket.io": "socketio",
    "socket.io-client": "socketio",
    "plotly": "plotlyjs",
    "react-plotly.js": "react_plotlyjs",
    "react-chartjs-2": "react_chartjs_2",
    "tailwindcss-animate": None,
    "autoprefixer": None,
    "postcss": None,
    "typescript": None,
    "eslint": None,
    "prettier": None,
    "http": "node_core",
    "fs": "node_core",
    "path": "node_core",
    "os": "os_pathlib",
    "pathlib": "os_pathlib",
    "scikit-learn": None,
    "matplotlib": None,
    "scipy": None,
    "pillow": None,
    "psycopg2-binary": "sqlalchemy",
    "psycopg2": "sqlalchemy",
    "openpyxl": None,
}

SERVER_FRAMEWORK_DEPS = frozenset({
    "flask", "fastapi", "express", "streamlit", "nextjs", "gradio",
})

_DATA_DIR = Path(__file__).parent / "data"
_MODEL_PATH = _DATA_DIR / "prediction_model.joblib"
_METADATA_PATH = _DATA_DIR / "model_metadata.json"

# Guarded ML imports — fall back to corpus lookup when unavailable.
try:
    import joblib
    import numpy as np
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

# Lazy model cache — loaded on first predict call.
_model_cache: dict = {}


def _load_model():
    """Load the trained model and metadata.  Returns (model, metadata) or
    (None, None) on any failure.  Caches after first successful load."""
    if "loaded" in _model_cache:
        return _model_cache.get("model"), _model_cache.get("metadata")

    _model_cache["loaded"] = True  # mark attempted even on failure

    if not _HAS_ML:
        logger.info("ML libraries not available — using corpus lookup")
        return None, None

    if not _MODEL_PATH.exists():
        logger.info("Prediction model not found at %s — using corpus lookup", _MODEL_PATH)
        return None, None

    try:
        model = joblib.load(_MODEL_PATH)
        with open(_METADATA_PATH) as f:
            metadata = json.load(f)
        _model_cache["model"] = model
        _model_cache["metadata"] = metadata
        logger.info(
            "Loaded prediction model (trained on %d samples, AUC %.3f)",
            metadata.get("training_samples", 0),
            metadata.get("mean_auc", 0),
        )
        return model, metadata
    except Exception as exc:
        logger.warning("Failed to load prediction model: %s", exc)
        return None, None


def _extract_features(
    dependency_names: list[str],
    feature_columns: list[str],
    ingestion=None,
) -> "np.ndarray":
    """Build a feature vector matching the training schema.

    Args:
        dependency_names: Raw dep names from the project.
        feature_columns: Ordered column names from model metadata.
        ingestion: Optional IngestionResult for complexity features.

    Returns:
        numpy array of shape (1, n_features).
    """
    # Normalize deps
    dep_set: set[str] = set()
    for raw in dependency_names:
        canonical = normalize_dep_name(raw)
        if canonical and canonical in PROFILED_DEPS:
            dep_set.add(canonical)

    # Complexity defaults (medians from a 4K corpus)
    loc = 0
    file_count = 0
    files_failed = 0
    language = 0  # python
    dep_count = len(dependency_names)

    if ingestion is not None:
        loc = getattr(ingestion, "total_lines", 0) or 0
        file_count = getattr(ingestion, "files_analyzed", 0) or 0
        files_failed = getattr(ingestion, "files_failed", 0) or 0
        if hasattr(ingestion, "dependencies"):
            dep_count = len(ingestion.dependencies)
        # Detect language from ingestion (check for JS indicators)
        lang_str = getattr(ingestion, "language", "python") or "python"
        language = 1 if lang_str.lower() == "javascript" else 0

    has_server = int(bool(dep_set & SERVER_FRAMEWORK_DEPS))

    # Build feature dict
    feat: dict[str, float] = {}
    for col in feature_columns:
        if col.startswith("dep_"):
            dep_name = col[4:]
            feat[col] = 1.0 if dep_name in dep_set else 0.0
        elif col == "dep_count":
            feat[col] = float(dep_count)
        elif col == "loc":
            feat[col] = float(loc)
        elif col == "file_count":
            feat[col] = float(file_count)
        elif col == "files_failed":
            feat[col] = float(files_failed)
        elif col == "has_server_framework":
            feat[col] = float(has_server)
        elif col == "language":
            feat[col] = float(language)
        else:
            feat[col] = 0.0

    values = [feat[col] for col in feature_columns]
    return np.array([values], dtype=np.float32)


def normalize_dep_name(raw: str) -> Optional[str]:
    """Map a raw dependency name to its canonical profile name.

    Returns ``None`` if the dep should be ignored (tooling, not a real dep).
    Returns the profile name if mapped, or the raw name (lowercased,
    hyphens → underscores) if no alias exists.
    """
    lower = raw.lower().strip()
    if lower in DEP_ALIASES:
        return DEP_ALIASES[lower]
    # Normalize: hyphens and dots to underscores
    normalized = lower.replace("-", "_").replace(".", "_")
    if normalized in DEP_ALIASES:
        return DEP_ALIASES[normalized]
    return normalized


# Default corpus paths, tried in order:
# 1. Bundled package data (works on Railway / pip install)
# 2. Repo-relative from source tree (works in development)
# 3. CWD-relative (fallback)
_CORPUS_PATHS = [
    Path(__file__).parent / "data" / "corpus_patterns_ranked.json",
    Path(__file__).parent.parent.parent / "corpus_extraction" / "corpus_patterns_ranked.json",
    Path("corpus_extraction/corpus_patterns_ranked.json"),
]

# Skip patterns that are purely informational or infrastructure noise.
_SKIP_TITLES = frozenset({
    "Application handled HTTP load without issues",
})

# Minimum overlap to consider a pattern relevant.
_MIN_DEP_OVERLAP = 1

# Maximum predictions to return.
_MAX_PREDICTIONS = 5

# Minimum confirmed_count to include in predictions.
_MIN_CONFIRMED = 3


@dataclass
class PredictionItem:
    """A single predicted issue."""

    title: str
    probability_pct: float
    severity: str
    confirmed_count: int
    matching_deps: list[str] = field(default_factory=list)
    scale_note: str = ""


@dataclass
class PredictionResult:
    """Complete prediction output."""

    predictions: list[PredictionItem] = field(default_factory=list)
    total_similar_projects: int = 0
    matching_deps: list[str] = field(default_factory=list)


def _load_corpus(corpus_path: Optional[str] = None) -> list[dict]:
    """Load corpus patterns from JSON file."""
    if corpus_path:
        paths = [Path(corpus_path)]
    else:
        paths = list(_CORPUS_PATHS)

    for p in paths:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load corpus from %s: %s", p, exc)
    return []


def _dominant_severity(dist: dict[str, int]) -> str:
    """Return the most common severity from a distribution dict."""
    if not dist:
        return "info"
    return max(dist, key=dist.get)  # type: ignore[arg-type]


def predict_issues(
    dependency_names: list[str],
    corpus_path: Optional[str] = None,
    constraints: Optional[OperationalConstraints] = None,
    ingestion=None,
) -> PredictionResult:
    """Predict likely failure patterns for a project.

    Uses a trained XGBoost model when available, falls back to corpus
    pattern lookup otherwise.

    Args:
        dependency_names: List of dependency names from ingestion.
        corpus_path: Override path to corpus_patterns_ranked.json.
        constraints: User's operational constraints for scale-relative
            framing.
        ingestion: Optional IngestionResult for project complexity features.
            Improves model accuracy when available.

    Returns:
        PredictionResult with top predictions and metadata.
    """
    # Try model-based prediction first
    model, metadata = _load_model()
    if model is not None and metadata is not None:
        try:
            return _predict_with_model(
                model, metadata, dependency_names, constraints, ingestion,
            )
        except Exception as exc:
            logger.warning("Model prediction failed, falling back to corpus: %s", exc)

    # Fallback: corpus lookup
    patterns = _load_corpus(corpus_path)
    if not patterns:
        return PredictionResult()

    dep_set = {d.lower() for d in dependency_names}
    if not dep_set:
        return PredictionResult()

    # Score each pattern by dependency overlap × confirmed_count
    scored: list[tuple[float, dict, list[str]]] = []
    total_confirmed = 0

    for pattern in patterns:
        title = pattern.get("title", "")
        if title in _SKIP_TITLES:
            continue

        confirmed = pattern.get("confirmed_count", 0)
        if confirmed < _MIN_CONFIRMED:
            continue

        affected = {d.lower() for d in pattern.get("affected_dependencies", [])}
        overlap = dep_set & affected
        if len(overlap) < _MIN_DEP_OVERLAP:
            continue

        # Score: overlap count × log-ish of confirmed count
        score = len(overlap) * confirmed
        matched_deps = sorted(overlap)
        scored.append((score, pattern, matched_deps))
        total_confirmed += confirmed

    if not scored:
        return PredictionResult()

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:_MAX_PREDICTIONS]

    # Compute total similar projects (sum of confirmed across all matching)
    total_similar = sum(p.get("confirmed_count", 0) for _, p, _ in scored)

    # Build all matching deps (union)
    all_matching = sorted({d for _, _, deps in scored for d in deps})

    # Build prediction items
    predictions: list[PredictionItem] = []
    for _score, pattern, matched_deps in top:
        confirmed = pattern.get("confirmed_count", 0)
        severity_dist = pattern.get("severity_distribution", {})
        severity = _dominant_severity(severity_dist)

        # Probability: what fraction of total matching patterns is this one
        prob_pct = (confirmed / total_similar * 100) if total_similar > 0 else 0

        scale_note = _build_scale_note(constraints)

        predictions.append(PredictionItem(
            title=pattern.get("title", "Unknown"),
            probability_pct=round(prob_pct, 1),
            severity=severity,
            confirmed_count=confirmed,
            matching_deps=matched_deps,
            scale_note=scale_note,
        ))

    return PredictionResult(
        predictions=predictions,
        total_similar_projects=total_similar,
        matching_deps=all_matching,
    )


def _build_scale_note(
    constraints: Optional[OperationalConstraints],
) -> str:
    """Build scale-relative framing note from constraints."""
    if not constraints:
        return ""
    max_u = constraints.max_users or constraints.user_scale
    if max_u is not None and constraints.per_user_data:
        pu_items = per_user_data_to_items(constraints.per_user_data)
        combined = max_u * pu_items
        return (
            f"At your stated {max_u:,} users with "
            f"{constraints.per_user_data} datasets "
            f"(~{pu_items:,} items each), combined data volume "
            f"of ~{combined:,} items may trigger this issue"
        )
    if max_u is not None:
        return (
            f"At your stated {max_u:,} users, "
            f"this is worth watching"
        )
    return ""


def _predict_with_model(
    model,
    metadata: dict,
    dependency_names: list[str],
    constraints: Optional[OperationalConstraints],
    ingestion,
) -> PredictionResult:
    """Generate predictions using the trained XGBoost model."""
    feature_columns = metadata["feature_columns"]
    target_columns = metadata["target_columns"]
    target_info = metadata.get("target_info", {})
    training_samples = metadata.get("training_samples", 0)

    X = _extract_features(dependency_names, feature_columns, ingestion)

    # predict_proba returns list of (1, 2) arrays for MultiOutputClassifier
    raw_proba = model.predict_proba(X)
    if isinstance(raw_proba, list):
        probas = [p[0, 1] for p in raw_proba]
    else:
        probas = raw_proba[0].tolist()

    # Pair with target info and sort by probability
    scored = []
    for i, col in enumerate(target_columns):
        info = target_info.get(col, {})
        prob = probas[i] if i < len(probas) else 0.0
        scored.append((prob, col, info))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:_MAX_PREDICTIONS]

    # Build matching deps for each prediction
    dep_set = {normalize_dep_name(d) for d in dependency_names}
    dep_set.discard(None)

    scale_note = _build_scale_note(constraints)

    predictions = []
    for prob, col, info in top:
        if prob < 0.01:
            continue
        # Find which user deps overlap with this pattern's typical deps
        pattern_deps = set()
        title = info.get("title", col)
        # Extract dep name from title like "Data Volume Scaling (pandas)"
        for dep in PROFILED_DEPS:
            if dep in title.lower() or dep in col:
                pattern_deps.add(dep)
        matching = sorted(dep_set & pattern_deps) if pattern_deps else []

        predictions.append(PredictionItem(
            title=title,
            probability_pct=round(float(prob) * 100, 1),
            severity=info.get("severity", "info"),
            confirmed_count=info.get("confirmed_count", 0),
            matching_deps=matching,
            scale_note=scale_note,
        ))

    # All matching deps from the project
    all_matching = sorted(d for d in dep_set if d in PROFILED_DEPS)

    return PredictionResult(
        predictions=predictions,
        total_similar_projects=training_samples,
        matching_deps=all_matching,
    )


def match_prediction_to_findings(
    prediction_title: str,
    finding_titles: list[str],
    finding_categories: list[str],
) -> bool:
    """Check whether a prediction was confirmed by test results.

    Uses keyword overlap between the prediction title and finding
    titles/categories.
    """
    # Extract meaningful words (3+ chars, lowercase)
    pred_words = {
        w.lower() for w in prediction_title.split()
        if len(w) >= 3 and w.isalpha()
    }
    # Remove common filler words
    pred_words -= {"the", "and", "for", "your", "with", "not", "was", "are"}

    if not pred_words:
        return False

    for title in finding_titles:
        title_words = {
            w.lower() for w in title.split()
            if len(w) >= 3 and w.isalpha()
        }
        overlap = pred_words & title_words
        if len(overlap) >= 2:
            return True

    # Also check category keywords
    for cat in finding_categories:
        cat_words = set(cat.lower().replace("_", " ").split())
        if pred_words & cat_words:
            return True

    return False
