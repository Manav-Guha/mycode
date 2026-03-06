"""Tier 1 Inference Engine — predict risk areas from historical patterns.

Pure pattern matching against accumulated corpus data. No test execution.
Runs in under 30 seconds.

Input: project dependency list + file structure (from ingester output)
Output: predicted risk areas with statistical backing

Example outputs:
  "In myCode's analysis of 168 projects, 100% of Streamlit projects
   exhibited resource exhaustion. Your project uses Streamlit."
  "Projects combining Flask + pandas showed scaling collapse in 41% of cases."
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mycode.classifiers import classify_project

logger = logging.getLogger(__name__)


# ── Data Classes ──


@dataclass
class RiskPrediction:
    """A single predicted risk area with statistical backing.

    Attributes:
        risk_summary: Plain-language description of the predicted risk.
        failure_domain: The taxonomy failure domain.
        failure_pattern: The Level 2 pattern name, if specific enough.
        affected_dependencies: Which project dependencies are involved.
        confidence: Statistical confidence ("high", "medium", "low").
        evidence_count: Number of historical findings supporting this.
        total_projects_with_dep: How many projects in corpus had this dep.
        occurrence_rate: Percentage of projects exhibiting this pattern.
        severity_distribution: Breakdown of severities in historical data.
    """

    risk_summary: str
    failure_domain: str
    failure_pattern: Optional[str] = None
    affected_dependencies: list[str] = field(default_factory=list)
    confidence: str = "medium"
    evidence_count: int = 0
    total_projects_with_dep: int = 0
    occurrence_rate: float = 0.0
    severity_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "risk_summary": self.risk_summary,
            "failure_domain": self.failure_domain,
            "failure_pattern": self.failure_pattern,
            "affected_dependencies": list(self.affected_dependencies),
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "total_projects_with_dep": self.total_projects_with_dep,
            "occurrence_rate": round(self.occurrence_rate, 1),
            "severity_distribution": dict(self.severity_distribution),
        }


@dataclass
class InferenceResult:
    """Complete output of the inference engine.

    Attributes:
        predictions: Risk predictions ordered by confidence.
        project_vertical: Classified vertical.
        project_architecture: Classified architectural pattern.
        corpus_size: Total projects in the corpus.
        matched_dependencies: Dependencies that matched corpus data.
        unmatched_dependencies: Dependencies with no corpus data.
    """

    predictions: list[RiskPrediction] = field(default_factory=list)
    project_vertical: str = ""
    project_architecture: str = ""
    corpus_size: int = 0
    matched_dependencies: list[str] = field(default_factory=list)
    unmatched_dependencies: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "predictions": [p.as_dict() for p in self.predictions],
            "project_vertical": self.project_vertical,
            "project_architecture": self.project_architecture,
            "corpus_size": self.corpus_size,
            "matched_dependencies": list(self.matched_dependencies),
            "unmatched_dependencies": list(self.unmatched_dependencies),
        }

    def as_text(self) -> str:
        """Render inference results as readable plain text."""
        sections: list[str] = []

        sections.append("=" * 60)
        sections.append("  myCode Tier 1 Analysis — Pattern Inference")
        sections.append("=" * 60)

        sections.append(
            f"\nProject type: {self.project_vertical} "
            f"({self.project_architecture})"
        )
        sections.append(
            f"Based on analysis of {self.corpus_size} projects in myCode's corpus."
        )

        if self.matched_dependencies:
            sections.append(
                f"\nDependencies with historical data: "
                f"{', '.join(self.matched_dependencies)}"
            )
        if self.unmatched_dependencies:
            sections.append(
                f"Dependencies without historical data: "
                f"{', '.join(self.unmatched_dependencies[:10])}"
                + ("..." if len(self.unmatched_dependencies) > 10 else "")
            )

        if not self.predictions:
            sections.append(
                "\nNo specific risk patterns found in historical data "
                "for this dependency combination."
            )
        else:
            # Group by confidence
            high = [p for p in self.predictions if p.confidence == "high"]
            medium = [p for p in self.predictions if p.confidence == "medium"]
            low = [p for p in self.predictions if p.confidence == "low"]

            if high:
                sections.append("\n" + "-" * 40)
                sections.append("  High Confidence Risks")
                sections.append("-" * 40)
                for p in high:
                    sections.append(f"\n[!!] {p.risk_summary}")
                    if p.severity_distribution:
                        sev_parts = []
                        for sev in ("critical", "warning", "info"):
                            if sev in p.severity_distribution:
                                sev_parts.append(
                                    f"{p.severity_distribution[sev]} {sev}"
                                )
                        if sev_parts:
                            sections.append(
                                f"     Historical severity: {', '.join(sev_parts)}"
                            )

            if medium:
                sections.append("\n" + "-" * 40)
                sections.append("  Medium Confidence Risks")
                sections.append("-" * 40)
                for p in medium:
                    sections.append(f"\n[! ] {p.risk_summary}")

            if low:
                sections.append("\n" + "-" * 40)
                sections.append("  Lower Confidence Risks")
                sections.append("-" * 40)
                for p in low:
                    sections.append(f"\n[  ] {p.risk_summary}")

        sections.append("\n" + "=" * 60)
        sections.append(
            "  This is a pattern-based prediction. Run a full analysis "
            "(Tier 2/3) for definitive results."
        )
        sections.append("=" * 60)

        return "\n".join(sections)


# ── Corpus Index ──


class CorpusIndex:
    """In-memory index of classified corpus data for fast lookups.

    Builds indexes on load for O(1) dependency and pattern lookups.
    """

    def __init__(self, corpus_path: Optional[str | Path] = None) -> None:
        self._entries: list[dict] = []
        self._total_repos: int = 0

        # Indexes
        self._dep_entries: dict[str, list[dict]] = {}  # dep_name → entries
        self._dep_repos: dict[str, set[str]] = {}  # dep_name → repo_urls
        self._dep_domains: dict[str, dict[str, int]] = {}  # dep → domain → count
        self._dep_patterns: dict[str, dict[str, int]] = {}  # dep → pattern → count
        self._dep_severities: dict[str, dict[str, int]] = {}  # dep → severity → count
        self._vertical_repos: dict[str, set[str]] = {}  # vertical → repo_urls
        self._combo_entries: dict[tuple[str, ...], list[dict]] = {}  # dep_combo → entries

        if corpus_path:
            self.load(corpus_path)

    def load(self, corpus_path: str | Path) -> None:
        """Load classified corpus data and build indexes."""
        path = Path(corpus_path)
        if not path.exists():
            logger.warning("Corpus file not found: %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load corpus: %s", exc)
            return

        self._entries = data.get("entries", [])

        # Count unique repos
        all_repos = set()
        for entry in self._entries:
            repo = entry.get("repo_url", "")
            if repo:
                all_repos.add(repo)
        self._total_repos = len(all_repos)

        # Build indexes
        for entry in self._entries:
            repo = entry.get("repo_url", "")
            domain = entry.get("failure_domain", "")
            pattern = entry.get("failure_pattern")
            severity = entry.get("severity_raw", "")
            vertical = entry.get("vertical", "")
            deps = entry.get("affected_dependencies", [])

            # Vertical index
            if vertical and repo:
                self._vertical_repos.setdefault(vertical, set()).add(repo)

            for dep in deps:
                dep_lower = dep.lower()

                # Per-dep entries
                self._dep_entries.setdefault(dep_lower, []).append(entry)

                # Per-dep repos
                if repo:
                    self._dep_repos.setdefault(dep_lower, set()).add(repo)

                # Per-dep domain counts
                if domain:
                    self._dep_domains.setdefault(dep_lower, {})
                    self._dep_domains[dep_lower][domain] = (
                        self._dep_domains[dep_lower].get(domain, 0) + 1
                    )

                # Per-dep pattern counts
                if pattern:
                    self._dep_patterns.setdefault(dep_lower, {})
                    self._dep_patterns[dep_lower][pattern] = (
                        self._dep_patterns[dep_lower].get(pattern, 0) + 1
                    )

                # Per-dep severity counts
                if severity:
                    self._dep_severities.setdefault(dep_lower, {})
                    self._dep_severities[dep_lower][severity] = (
                        self._dep_severities[dep_lower].get(severity, 0) + 1
                    )

            # Dep combo index (for pairs)
            if len(deps) >= 2:
                combo = tuple(sorted(d.lower() for d in deps))
                self._combo_entries.setdefault(combo, []).append(entry)

        logger.info(
            "Corpus loaded: %d entries, %d repos, %d unique deps",
            len(self._entries), self._total_repos, len(self._dep_entries),
        )

    @property
    def total_repos(self) -> int:
        return self._total_repos

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def dep_findings(self, dep: str) -> list[dict]:
        """Get all findings for a dependency."""
        return self._dep_entries.get(dep.lower(), [])

    def dep_repo_count(self, dep: str) -> int:
        """Count unique repos that had findings for this dep."""
        return len(self._dep_repos.get(dep.lower(), set()))

    def dep_domain_distribution(self, dep: str) -> dict[str, int]:
        """Get failure domain distribution for a dependency."""
        return dict(self._dep_domains.get(dep.lower(), {}))

    def dep_pattern_distribution(self, dep: str) -> dict[str, int]:
        """Get failure pattern distribution for a dependency."""
        return dict(self._dep_patterns.get(dep.lower(), {}))

    def dep_severity_distribution(self, dep: str) -> dict[str, int]:
        """Get severity distribution for a dependency."""
        return dict(self._dep_severities.get(dep.lower(), {}))

    def combo_findings(self, deps: list[str]) -> list[dict]:
        """Get findings for a specific dependency combination."""
        combo = tuple(sorted(d.lower() for d in deps))
        return self._combo_entries.get(combo, [])

    def vertical_repo_count(self, vertical: str) -> int:
        """Count repos in a specific vertical."""
        return len(self._vertical_repos.get(vertical, set()))


# ── Inference Engine ──


_DOMAIN_LABELS: dict[str, str] = {
    "resource_exhaustion": "resource exhaustion (memory leaks, cache overflow)",
    "concurrency_failure": "concurrency failures (deadlocks, race conditions)",
    "scaling_collapse": "scaling collapse (performance degradation under load)",
    "input_handling_failure": "input handling failures (crashes on unexpected data)",
    "dependency_failure": "dependency failures (version issues, API changes)",
    "integration_failure": "integration failures (timeouts, desync between components)",
    "configuration_environment_failure": "environment failures (install issues, platform problems)",
}

_PATTERN_LABELS: dict[str, str] = {
    "unbounded_cache_growth": "unbounded cache growth",
    "memory_accumulation_over_sessions": "memory accumulation over sessions",
    "large_payload_oom": "out-of-memory on large payloads",
    "connection_pool_depletion": "connection pool depletion",
    "request_deadlock": "request deadlocks",
    "race_condition": "race conditions",
    "linear_to_exponential_transition": "linear-to-exponential performance degradation",
    "response_time_cliff": "response time cliff under load",
    "throughput_plateau": "throughput plateau",
    "unvalidated_type_crash": "crashes on unvalidated input types",
    "empty_input_crash": "crashes on empty/null input",
    "format_boundary_failure": "format/boundary failures",
    "version_incompatibility": "version incompatibility issues",
    "api_breaking_change": "API breaking changes",
    "cascading_timeout": "cascading timeouts",
}

# Minimum evidence threshold for predictions
_MIN_EVIDENCE = 3
_HIGH_CONFIDENCE_RATE = 60.0  # 60%+ occurrence = high confidence
_MEDIUM_CONFIDENCE_RATE = 30.0  # 30%+ = medium
# Below 30% = low


def _confidence_level(rate: float) -> str:
    """Determine confidence level from occurrence rate."""
    if rate >= _HIGH_CONFIDENCE_RATE:
        return "high"
    if rate >= _MEDIUM_CONFIDENCE_RATE:
        return "medium"
    return "low"


def _default_corpus_path() -> Path:
    """Default path for corpus_classified.json."""
    # Check project root first (development), then package data
    candidates = [
        Path.cwd() / "corpus_classified.json",
        Path(__file__).parent.parent.parent / "corpus_classified.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Will fail gracefully on load


class InferenceEngine:
    """Tier 1 inference — predict risks from historical patterns.

    Usage::

        engine = InferenceEngine()
        result = engine.infer(dependencies=["flask", "pandas"], file_structure=["app.py"])

    Or with a custom corpus::

        engine = InferenceEngine(corpus_path="/path/to/corpus_classified.json")
    """

    def __init__(
        self,
        corpus_path: Optional[str | Path] = None,
    ) -> None:
        path = corpus_path or _default_corpus_path()
        self._index = CorpusIndex(path)

    def infer(
        self,
        dependencies: list[str],
        file_structure: list[str] | None = None,
        framework: str = "",
        file_count: int = 0,
        has_frontend: bool = False,
        has_backend: bool = False,
    ) -> InferenceResult:
        """Run inference on a project fingerprint.

        Args:
            dependencies: List of dependency names.
            file_structure: List of file paths.
            framework: Primary framework name.
            file_count: Total file count.
            has_frontend: Whether project has frontend code.
            has_backend: Whether project has backend code.

        Returns:
            InferenceResult with risk predictions.
        """
        result = InferenceResult(
            corpus_size=self._index.total_repos,
        )

        # Classify the project
        project_cls = classify_project(
            dependencies=dependencies,
            file_structure=file_structure,
            framework=framework,
            file_count=file_count,
            has_frontend=has_frontend,
            has_backend=has_backend,
        )
        result.project_vertical = project_cls["vertical"]
        result.project_architecture = project_cls["architectural_pattern"]

        # Match dependencies against corpus
        matched = []
        unmatched = []
        for dep in dependencies:
            if self._index.dep_findings(dep):
                matched.append(dep)
            else:
                unmatched.append(dep)
        result.matched_dependencies = matched
        result.unmatched_dependencies = unmatched

        # Generate per-dependency predictions
        predictions: list[RiskPrediction] = []

        for dep in matched:
            dep_preds = self._predict_for_dep(dep)
            predictions.extend(dep_preds)

        # Generate dep-combo predictions
        if len(matched) >= 2:
            combo_preds = self._predict_for_combos(matched)
            predictions.extend(combo_preds)

        # Sort by confidence then occurrence rate
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        predictions.sort(
            key=lambda p: (confidence_order.get(p.confidence, 9), -p.occurrence_rate),
        )

        # Deduplicate — keep the highest-confidence prediction per
        # domain+pattern+dep combo
        seen: set[tuple[str, str | None, str]] = set()
        deduped: list[RiskPrediction] = []
        for pred in predictions:
            key = (
                pred.failure_domain,
                pred.failure_pattern,
                ",".join(sorted(pred.affected_dependencies)),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(pred)
        result.predictions = deduped

        return result

    def _predict_for_dep(self, dep: str) -> list[RiskPrediction]:
        """Generate risk predictions for a single dependency."""
        predictions = []

        domain_dist = self._index.dep_domain_distribution(dep)
        if not domain_dist:
            return predictions

        total_findings = sum(domain_dist.values())
        repo_count = self._index.dep_repo_count(dep)

        if total_findings < _MIN_EVIDENCE:
            return predictions

        severity_dist = self._index.dep_severity_distribution(dep)

        # Top domain prediction
        top_domain = max(domain_dist, key=lambda d: domain_dist[d])
        if top_domain == "unclassified":
            # Skip if the top domain is unclassified
            sorted_domains = sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_domains) > 1:
                top_domain = sorted_domains[1][0]
            else:
                return predictions

        domain_count = domain_dist[top_domain]
        rate = domain_count / total_findings * 100

        domain_label = _DOMAIN_LABELS.get(top_domain, top_domain)

        summary = (
            f"In myCode's analysis of {self._index.total_repos} projects, "
            f"{rate:.0f}% of {dep} findings involved {domain_label}. "
            f"Your project uses {dep}."
        )

        predictions.append(RiskPrediction(
            risk_summary=summary,
            failure_domain=top_domain,
            affected_dependencies=[dep],
            confidence=_confidence_level(rate),
            evidence_count=domain_count,
            total_projects_with_dep=repo_count,
            occurrence_rate=rate,
            severity_distribution=severity_dist,
        ))

        # Pattern-level predictions (more specific)
        pattern_dist = self._index.dep_pattern_distribution(dep)
        if pattern_dist:
            top_pattern = max(pattern_dist, key=lambda p: pattern_dist[p])
            pattern_count = pattern_dist[top_pattern]
            if pattern_count >= _MIN_EVIDENCE:
                pattern_rate = pattern_count / total_findings * 100
                pattern_label = _PATTERN_LABELS.get(top_pattern, top_pattern)

                pattern_summary = (
                    f"{dep} projects showed {pattern_label} in "
                    f"{pattern_rate:.0f}% of stress test findings "
                    f"({pattern_count} occurrences across {repo_count} projects)."
                )

                predictions.append(RiskPrediction(
                    risk_summary=pattern_summary,
                    failure_domain=top_domain,
                    failure_pattern=top_pattern,
                    affected_dependencies=[dep],
                    confidence=_confidence_level(pattern_rate),
                    evidence_count=pattern_count,
                    total_projects_with_dep=repo_count,
                    occurrence_rate=pattern_rate,
                    severity_distribution=severity_dist,
                ))

        return predictions

    def _predict_for_combos(self, deps: list[str]) -> list[RiskPrediction]:
        """Generate risk predictions for dependency combinations."""
        predictions = []

        # Check all pairs
        for i in range(len(deps)):
            for j in range(i + 1, len(deps)):
                pair = [deps[i], deps[j]]
                combo_findings = self._index.combo_findings(pair)
                if len(combo_findings) < _MIN_EVIDENCE:
                    continue

                # Analyze combo findings
                domain_counts: dict[str, int] = {}
                severity_counts: dict[str, int] = {}
                for entry in combo_findings:
                    domain = entry.get("failure_domain", "")
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    severity = entry.get("severity_raw", "")
                    if severity:
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1

                if not domain_counts:
                    continue

                top_domain = max(domain_counts, key=lambda d: domain_counts[d])
                if top_domain == "unclassified":
                    continue

                domain_count = domain_counts[top_domain]
                total = sum(domain_counts.values())
                rate = domain_count / total * 100

                domain_label = _DOMAIN_LABELS.get(top_domain, top_domain)
                dep_str = f"{pair[0]} + {pair[1]}"

                summary = (
                    f"Projects combining {dep_str} showed {domain_label} "
                    f"in {rate:.0f}% of cases ({domain_count} findings)."
                )

                predictions.append(RiskPrediction(
                    risk_summary=summary,
                    failure_domain=top_domain,
                    affected_dependencies=pair,
                    confidence=_confidence_level(rate),
                    evidence_count=domain_count,
                    total_projects_with_dep=0,
                    occurrence_rate=rate,
                    severity_distribution=severity_counts,
                ))

        return predictions
