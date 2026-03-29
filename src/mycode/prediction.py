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

from mycode.constraints import OperationalConstraints

logger = logging.getLogger(__name__)

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
) -> PredictionResult:
    """Look up corpus patterns for projects with similar dependency stacks.

    Args:
        dependency_names: List of dependency names from ingestion
            (e.g. ``["flask", "sqlalchemy", "pandas"]``).
        corpus_path: Override path to corpus_patterns_ranked.json.
        constraints: User's operational constraints for scale-relative
            framing.  ``None`` for non-interactive mode.

    Returns:
        PredictionResult with top predictions and metadata.
    """
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

        # Scale-relative framing
        scale_note = ""
        if constraints and constraints.max_users is not None:
            scale_note = (
                f"At your stated {constraints.max_users:,} users, "
                f"this is worth watching"
            )
        elif constraints and constraints.user_scale is not None:
            scale_note = (
                f"At your stated {constraints.user_scale:,} users, "
                f"this is worth watching"
            )

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
