"""Scenario Discovery Engine (Component 9) — Discovery Logging.

Detects novel failure patterns by comparing execution results against
component library profile predictions.  Writes discovery candidate JSON
files to ``~/.mycode/discoveries/``.

No user consent required — only dependency behavior observations are
recorded, no PII, no user code, no conversation content.

No LLM dependency.  Detection is deterministic arithmetic (actual vs
expected).
"""

import importlib.metadata
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mycode.constraints import OperationalConstraints
from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.library.loader import DependencyProfile, ProfileMatch
from mycode.scenario import ScenarioGeneratorResult, StressTestScenario

logger = logging.getLogger(__name__)

# Deviation thresholds
_MEMORY_GROWTH_FACTOR = 2.0  # >2x faster than predicted
_SUPERLINEAR_RATIO_THRESHOLD = 1.3  # ratio increase between consecutive steps


def _get_version() -> str:
    """Read mycode-ai version from package metadata."""
    try:
        return importlib.metadata.version("mycode-ai")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ── Data Class ──


@dataclass
class DiscoveryCandidate:
    """A novel failure pattern worth cataloguing.

    Matches the spec JSON schema from CLAUDE.md §9.
    """

    discovery_id: str
    timestamp: str
    mycode_version: str
    language: str
    dependencies_involved: list[str]
    scenario_category: str
    expected_behavior: str
    actual_behavior: str
    deviation_factor: float
    load_level_at_discovery: str
    reproducible: bool
    constraint_context: dict
    raw_metrics: dict
    suggested_template: str

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON output."""
        return asdict(self)


# ── Helpers ──


def _parse_typical_limit(limit_str: object) -> Optional[float]:
    """Parse a scaling_limits ``typical_limit`` value to its upper bound.

    Handles:
      - ``"10-50"`` → ``50.0``
      - ``150000`` (int/float) → ``150000.0``
      - Invalid → ``None``
    """
    if isinstance(limit_str, (int, float)):
        return float(limit_str)
    if isinstance(limit_str, str):
        # Range form: "10-50"
        m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", limit_str)
        if m:
            return float(m.group(2))
        # Single number in string
        m = re.search(r"(\d+(?:\.\d+)?)", limit_str)
        if m:
            return float(m.group(1))
    return None


def _extract_load_level(
    scenario_result: ScenarioResult,
    step: StepResult,
) -> tuple[str, Optional[float]]:
    """Extract load level description and numeric value from a step.

    Looks in step parameters for common load-related keys, then falls
    back to parsing the step name.

    Returns ``(description_string, numeric_value_or_None)``.
    """
    # Check parameters for load-related keys
    _LOAD_KEYS = (
        "concurrent_users", "concurrency", "users", "load_level",
        "data_size", "num_rows", "file_size_mb", "payload_size_mb",
        "iterations", "connections", "sessions",
    )
    for key in _LOAD_KEYS:
        val = step.parameters.get(key)
        if val is not None:
            try:
                return f"{key}={val}", float(val)
            except (TypeError, ValueError):
                return f"{key}={val}", None

    # Fall back to step name pattern: "step_N", "load_100", "size_1000"
    m = re.search(r"(\d+(?:\.\d+)?)", step.step_name)
    if m:
        num = float(m.group(1))
        return step.step_name, num

    return step.step_name, None


def _build_dep_strings(
    scenario: StressTestScenario,
    profile_matches: list[ProfileMatch],
) -> list[str]:
    """Build ``"name==version"`` strings for dependencies targeted by a scenario."""
    match_by_name: dict[str, ProfileMatch] = {
        pm.dependency_name: pm for pm in profile_matches
    }
    result: list[str] = []
    for dep in scenario.target_dependencies:
        pm = match_by_name.get(dep)
        if pm and pm.installed_version:
            result.append(f"{dep}=={pm.installed_version}")
        else:
            result.append(dep)
    return result


def _constraints_to_dict(constraints: Optional[OperationalConstraints]) -> dict:
    """Convert OperationalConstraints to a plain dict for JSON."""
    if constraints is None:
        return {}
    return {
        "user_scale": constraints.user_scale,
        "usage_pattern": constraints.usage_pattern,
        "max_payload_mb": constraints.max_payload_mb,
        "data_type": constraints.data_type,
        "deployment_context": constraints.deployment_context,
        "availability_requirement": constraints.availability_requirement,
        "data_sensitivity": constraints.data_sensitivity,
        "growth_expectation": constraints.growth_expectation,
    }


def _is_known_failure(errors: list, profile: DependencyProfile) -> bool:
    """Check if errors match any of a profile's known_failure_modes.

    Compares error messages/types against ``detection_hint`` and
    ``name`` fields in the profile's known failure modes.
    """
    if not profile.known_failure_modes:
        return False

    error_text = " ".join(
        str(e.get("message", "")) + " " + str(e.get("type", ""))
        if isinstance(e, dict) else str(e)
        for e in errors
    ).lower()

    if not error_text.strip():
        return False

    for mode in profile.known_failure_modes:
        # Check detection_hint
        hint = mode.get("detection_hint", "")
        if hint and hint.lower() in error_text:
            return True
        # Check failure mode name (e.g., "memory_error_on_operations")
        name = mode.get("name", "")
        if name:
            # Convert snake_case to words for matching
            name_words = name.replace("_", " ").lower()
            if name_words in error_text:
                return True
    return False


def _detect_superlinear(values: list[float]) -> bool:
    """Check if a sequence of values shows accelerating (superlinear) growth.

    Computes ratios between consecutive values.  If ratios are
    *increasing* (each ratio > previous × threshold), growth is
    superlinear.

    Requires at least 3 values.
    """
    if len(values) < 3:
        return False

    # Filter out zeros/negatives to avoid division issues
    positive = [v for v in values if v > 0]
    if len(positive) < 3:
        return False

    ratios: list[float] = []
    for i in range(1, len(positive)):
        ratios.append(positive[i] / positive[i - 1])

    if len(ratios) < 2:
        return False

    # Check if ratios are consistently increasing
    increasing_count = sum(
        1 for i in range(1, len(ratios))
        if ratios[i] > ratios[i - 1] * _SUPERLINEAR_RATIO_THRESHOLD
    )
    # Majority of ratio transitions should be increasing
    return increasing_count >= (len(ratios) - 1) / 2


def _find_profile_for_scenario(
    scenario: StressTestScenario,
    profile_matches: list[ProfileMatch],
) -> Optional[DependencyProfile]:
    """Find the first matching profile for a scenario's target dependencies."""
    for dep in scenario.target_dependencies:
        for pm in profile_matches:
            if pm.dependency_name == dep and pm.profile is not None:
                return pm.profile
    return None


def _get_scenario_for_result(
    scenario_result: ScenarioResult,
    scenarios: list[StressTestScenario],
) -> Optional[StressTestScenario]:
    """Find the StressTestScenario matching a ScenarioResult by name."""
    for s in scenarios:
        if s.name == scenario_result.scenario_name:
            return s
    return None


def _step_has_errors(step: StepResult) -> bool:
    """Check if a step has errors."""
    return step.error_count > 0 or bool(step.errors)


def _scenario_has_errors(sr: ScenarioResult) -> bool:
    """Check if a scenario has errors (via steps or status)."""
    if sr.status in ("failed", "partial"):
        return True
    if sr.total_errors > 0:
        return True
    return any(_step_has_errors(s) for s in sr.steps)


# ── Discovery Engine ──


class DiscoveryEngine:
    """Detects novel failure patterns by comparing execution results
    against component library profile predictions.

    Usage::

        engine = DiscoveryEngine()
        discoveries = engine.analyse(execution, scenarios, matches, constraints, lang, ver)
        paths = engine.save(discoveries)
    """

    def __init__(self, discoveries_dir: Optional[Path] = None) -> None:
        self._discoveries_dir = discoveries_dir or (
            Path.home() / ".mycode" / "discoveries"
        )

    def analyse(
        self,
        execution: ExecutionEngineResult,
        scenarios: ScenarioGeneratorResult,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: Optional[str] = None,
    ) -> list[DiscoveryCandidate]:
        """Main entry point — analyse execution results for novel patterns.

        Returns a list of discovery candidates (may be empty).
        """
        version = version or _get_version()
        all_scenarios = scenarios.scenarios

        discoveries: list[DiscoveryCandidate] = []

        for sr in execution.scenario_results:
            # Skip skipped scenarios
            if sr.status == "skipped":
                continue

            scenario = _get_scenario_for_result(sr, all_scenarios)
            if scenario is None:
                continue

            discoveries.extend(
                self._detect_crash_at_safe_level(
                    sr, scenario, profile_matches, constraints, language, version,
                )
            )
            discoveries.extend(
                self._detect_memory_growth_anomaly(
                    sr, scenario, profile_matches, constraints, language, version,
                )
            )
            discoveries.extend(
                self._detect_curve_shape_mismatch(
                    sr, scenario, profile_matches, constraints, language, version,
                )
            )
            discoveries.extend(
                self._detect_interaction_failure(
                    sr, scenario, profile_matches, constraints, language, version,
                )
            )
            discoveries.extend(
                self._detect_unrecognized_dep_failure(
                    sr, scenario, profile_matches, constraints, language, version,
                )
            )

        return discoveries

    def save(self, discoveries: list[DiscoveryCandidate]) -> list[Path]:
        """Write each discovery as a JSON file to the discoveries directory.

        Creates the directory lazily on first write.  Returns list of
        file paths written.
        """
        if not discoveries:
            return []

        self._discoveries_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for d in discoveries:
            path = self._discoveries_dir / f"{d.discovery_id}.json"
            path.write_text(json.dumps(d.to_dict(), indent=2) + "\n")
            paths.append(path)

        return paths

    # ── Detection Methods ──

    def _detect_crash_at_safe_level(
        self,
        sr: ScenarioResult,
        scenario: StressTestScenario,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: str,
    ) -> list[DiscoveryCandidate]:
        """Detect crashes at load levels the profile marks as safe."""
        if not _scenario_has_errors(sr):
            return []

        profile = _find_profile_for_scenario(scenario, profile_matches)
        if profile is None:
            return []

        # Find the relevant scaling limit
        upper_safe: Optional[float] = None
        limit_metric = ""
        for limit in profile.scaling_characteristics.get("scaling_limits", []):
            parsed = _parse_typical_limit(limit.get("typical_limit"))
            if parsed is not None:
                upper_safe = parsed
                limit_metric = limit.get("metric", "")
                break

        if upper_safe is None:
            return []

        discoveries: list[DiscoveryCandidate] = []
        for step in sr.steps:
            if not _step_has_errors(step):
                continue

            # Check if errors match known failure modes (exclude)
            if _is_known_failure(step.errors, profile):
                continue

            load_desc, load_val = _extract_load_level(sr, step)

            if load_val is None:
                continue

            # Exclude extreme scale (>3x user_scale if set)
            if constraints and constraints.user_scale:
                if load_val > constraints.user_scale * 3:
                    continue

            # Crash within safe range → discovery
            if load_val <= upper_safe:
                dep_strings = _build_dep_strings(scenario, profile_matches)
                discoveries.append(DiscoveryCandidate(
                    discovery_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    mycode_version=version,
                    language=language,
                    dependencies_involved=dep_strings,
                    scenario_category=sr.scenario_category,
                    expected_behavior=(
                        f"Profile marks {limit_metric} safe up to "
                        f"{upper_safe} ({profile.name})"
                    ),
                    actual_behavior=(
                        f"Crash/error at {load_desc} "
                        f"(within safe range)"
                    ),
                    deviation_factor=round(
                        upper_safe / max(load_val, 0.01), 2,
                    ),
                    load_level_at_discovery=load_desc,
                    reproducible=True,
                    constraint_context=_constraints_to_dict(constraints),
                    raw_metrics={
                        "memory_peak_mb": step.memory_peak_mb,
                        "execution_time_ms": step.execution_time_ms,
                        "error_count": step.error_count,
                    },
                    suggested_template=(
                        f"Test {profile.name} at load levels within "
                        f"documented safe range ({limit_metric} <= "
                        f"{upper_safe}) to detect early failures."
                    ),
                ))

        return discoveries

    def _detect_memory_growth_anomaly(
        self,
        sr: ScenarioResult,
        scenario: StressTestScenario,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: str,
    ) -> list[DiscoveryCandidate]:
        """Detect memory growth >2x faster than profile predicts."""
        if sr.scenario_category != "memory_profiling":
            return []

        profile = _find_profile_for_scenario(scenario, profile_matches)
        if profile is None:
            return []

        baseline = profile.memory_behavior.get("baseline_footprint_mb")
        if baseline is None or baseline <= 0:
            return []

        # Collect memory values across steps
        mem_values = [s.memory_peak_mb for s in sr.steps if s.memory_peak_mb > 0]
        if len(mem_values) < 2:
            return []

        actual_growth = mem_values[-1] - mem_values[0]
        # Expected: baseline per step is a rough heuristic — each step
        # should add roughly baseline_footprint_mb / step_count.
        # A simpler model: total expected growth ≈ baseline (the profile's
        # baseline is the footprint; growth beyond that is usage).
        expected_growth = float(baseline)

        if expected_growth <= 0:
            return []

        deviation = actual_growth / expected_growth

        if deviation <= _MEMORY_GROWTH_FACTOR:
            return []

        dep_strings = _build_dep_strings(scenario, profile_matches)
        load_desc = f"{len(sr.steps)} steps"
        if sr.steps:
            _, load_val = _extract_load_level(sr, sr.steps[-1])
            if load_val is not None:
                load_desc = f"{len(sr.steps)} steps (last load: {load_val})"

        return [DiscoveryCandidate(
            discovery_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            mycode_version=version,
            language=language,
            dependencies_involved=dep_strings,
            scenario_category=sr.scenario_category,
            expected_behavior=(
                f"Memory growth ~{baseline}MB baseline "
                f"({profile.name} profile prediction)"
            ),
            actual_behavior=(
                f"Memory grew {actual_growth:.1f}MB over "
                f"{len(mem_values)} steps "
                f"({deviation:.1f}x expected)"
            ),
            deviation_factor=round(deviation, 2),
            load_level_at_discovery=load_desc,
            reproducible=True,
            constraint_context=_constraints_to_dict(constraints),
            raw_metrics={
                "memory_values_mb": mem_values,
                "actual_growth_mb": round(actual_growth, 2),
                "expected_growth_mb": round(expected_growth, 2),
                "baseline_footprint_mb": baseline,
            },
            suggested_template=(
                f"Profile memory across {len(mem_values)}+ steps for "
                f"{profile.name}; expected ~{baseline}MB growth, "
                f"check for {deviation:.1f}x anomaly."
            ),
        )]

    def _detect_curve_shape_mismatch(
        self,
        sr: ScenarioResult,
        scenario: StressTestScenario,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: str,
    ) -> list[DiscoveryCandidate]:
        """Detect superlinear growth when profile predicts linear."""
        if sr.scenario_category not in ("data_volume_scaling", "memory_profiling"):
            return []

        profile = _find_profile_for_scenario(scenario, profile_matches)
        if profile is None:
            return []

        # Check if profile expects linear growth
        growth_pattern = profile.memory_behavior.get("growth_pattern", "")
        expected_behavior = scenario.expected_behavior or ""
        combined = (growth_pattern + " " + expected_behavior).lower()

        expects_linear = "linear" in combined

        if not expects_linear:
            return []

        # Gather metric series — use memory for memory_profiling,
        # execution time for data_volume_scaling
        if sr.scenario_category == "memory_profiling":
            values = [s.memory_peak_mb for s in sr.steps if s.memory_peak_mb > 0]
        else:
            values = [s.execution_time_ms for s in sr.steps if s.execution_time_ms > 0]

        if not _detect_superlinear(values):
            return []

        dep_strings = _build_dep_strings(scenario, profile_matches)
        load_desc = f"{len(sr.steps)} steps"

        metric_name = (
            "memory" if sr.scenario_category == "memory_profiling"
            else "execution time"
        )

        return [DiscoveryCandidate(
            discovery_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            mycode_version=version,
            language=language,
            dependencies_involved=dep_strings,
            scenario_category=sr.scenario_category,
            expected_behavior=(
                f"Linear {metric_name} growth "
                f"({profile.name} profile prediction)"
            ),
            actual_behavior=(
                f"Superlinear (accelerating) {metric_name} growth "
                f"detected across {len(values)} data points"
            ),
            deviation_factor=round(values[-1] / max(values[0], 0.01), 2),
            load_level_at_discovery=load_desc,
            reproducible=True,
            constraint_context=_constraints_to_dict(constraints),
            raw_metrics={
                "metric_name": metric_name,
                "values": [round(v, 2) for v in values],
            },
            suggested_template=(
                f"Scale {profile.name} across {len(values)}+ load levels "
                f"and verify {metric_name} grows linearly, not "
                f"exponentially."
            ),
        )]

    def _detect_interaction_failure(
        self,
        sr: ScenarioResult,
        scenario: StressTestScenario,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: str,
    ) -> list[DiscoveryCandidate]:
        """Detect failures in multi-dependency scenarios not in any profile."""
        if len(scenario.target_dependencies) < 2:
            return []

        if not _scenario_has_errors(sr):
            return []

        # Collect all errors from steps
        all_errors: list = []
        for step in sr.steps:
            all_errors.extend(step.errors)

        if not all_errors:
            return []

        # Check if this failure pattern is documented in any involved profile
        for dep in scenario.target_dependencies:
            for pm in profile_matches:
                if pm.dependency_name == dep and pm.profile is not None:
                    # Check known_failure_modes
                    if _is_known_failure(all_errors, pm.profile):
                        return []
                    # Check known_conflicts
                    conflicts = pm.profile.interaction_patterns.get(
                        "known_conflicts", [],
                    )
                    for conflict in conflicts:
                        conflict_dep = conflict.get("dependency", "")
                        if conflict_dep in scenario.target_dependencies:
                            # Known conflict — not a discovery
                            return []

        dep_strings = _build_dep_strings(scenario, profile_matches)
        load_desc = "N/A"
        if sr.steps:
            load_desc, _ = _extract_load_level(sr, sr.steps[-1])

        error_summary = "; ".join(
            str(e.get("message", str(e))) if isinstance(e, dict) else str(e)
            for e in all_errors[:3]
        )

        return [DiscoveryCandidate(
            discovery_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            mycode_version=version,
            language=language,
            dependencies_involved=dep_strings,
            scenario_category=sr.scenario_category,
            expected_behavior=(
                f"No documented interaction failure between "
                f"{', '.join(scenario.target_dependencies)}"
            ),
            actual_behavior=(
                f"Interaction failure: {error_summary}"
            ),
            deviation_factor=0.0,
            load_level_at_discovery=load_desc,
            reproducible=True,
            constraint_context=_constraints_to_dict(constraints),
            raw_metrics={
                "total_errors": sr.total_errors,
                "error_types": list({
                    e.get("type", "unknown") if isinstance(e, dict) else "unknown"
                    for e in all_errors
                }),
            },
            suggested_template=(
                f"Test interaction between "
                f"{' and '.join(scenario.target_dependencies)} "
                f"under load to detect undocumented failure patterns."
            ),
        )]

    def _detect_unrecognized_dep_failure(
        self,
        sr: ScenarioResult,
        scenario: StressTestScenario,
        profile_matches: list[ProfileMatch],
        constraints: Optional[OperationalConstraints],
        language: str,
        version: str,
    ) -> list[DiscoveryCandidate]:
        """Detect failures in scenarios targeting unrecognized dependencies."""
        if not _scenario_has_errors(sr):
            return []

        # Find unrecognized deps (ProfileMatch.profile is None)
        match_by_name: dict[str, ProfileMatch] = {
            pm.dependency_name: pm for pm in profile_matches
        }

        unrecognized: list[str] = []
        for dep in scenario.target_dependencies:
            pm = match_by_name.get(dep)
            if pm is None or pm.profile is None:
                unrecognized.append(dep)

        if not unrecognized:
            return []

        all_errors: list = []
        for step in sr.steps:
            all_errors.extend(step.errors)

        if not all_errors:
            return []

        dep_strings = _build_dep_strings(scenario, profile_matches)
        load_desc = "N/A"
        if sr.steps:
            load_desc, _ = _extract_load_level(sr, sr.steps[-1])

        error_summary = "; ".join(
            str(e.get("message", str(e))) if isinstance(e, dict) else str(e)
            for e in all_errors[:3]
        )

        return [DiscoveryCandidate(
            discovery_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            mycode_version=version,
            language=language,
            dependencies_involved=dep_strings,
            scenario_category=sr.scenario_category,
            expected_behavior=(
                f"No profile for {', '.join(unrecognized)} — "
                f"behavior unknown"
            ),
            actual_behavior=(
                f"Failure with unrecognized dependency: {error_summary}"
            ),
            deviation_factor=0.0,
            load_level_at_discovery=load_desc,
            reproducible=True,
            constraint_context=_constraints_to_dict(constraints),
            raw_metrics={
                "total_errors": sr.total_errors,
                "unrecognized_dependencies": unrecognized,
            },
            suggested_template=(
                f"Create profile for {', '.join(unrecognized)} — "
                f"failure observed under {sr.scenario_category} testing."
            ),
        )]
