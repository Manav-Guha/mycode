"""Threshold Hysteresis for Severity Classification.

Prevents run-to-run severity flicker when measurements land near threshold
boundaries.  A 20 % hysteresis margin means:

  - To **escalate** (move to a higher severity), the measurement must cross
    ``threshold × (1 − margin)`` — i.e. overshoot the base threshold.
  - To **de-escalate** (move to a lower severity), the measurement must cross
    ``threshold × (1 + margin)`` — i.e. clearly retreat past the threshold.

First-run behaviour (no prior state): thresholds apply without hysteresis.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──

HYSTERESIS_MARGIN = 0.20  # 20 %


# ── Data ──

@dataclass
class PriorRunState:
    """Severity classifications from a previous myCode run.

    Built by parsing the ``mycode-report.json`` written by the CLI.
    """

    finding_severities: dict[str, str] = field(default_factory=dict)
    """finding_key → severity  (e.g. ``"memory_profiling::Memory growth" → "critical"``)"""

    degradation_keys: set[str] = field(default_factory=set)
    """Keys of degradation points that existed in the prior run."""


# ── Key helpers ──

def finding_key(title: str, category: str) -> str:
    """Stable identity key for a finding across runs."""
    return f"{category}::{title}"


def degradation_key(scenario_name: str, metric: str) -> str:
    """Stable identity key for a degradation point across runs."""
    return f"{scenario_name}::{metric}"


# ── Prior state loading ──

def load_prior_state(project_path: Path) -> Optional[PriorRunState]:
    """Load severity classifications from a previous ``mycode-report.json``.

    Returns ``None`` when no prior report exists or when the file is
    corrupt / unparseable — the caller should treat this as a first run.

    .. note::

       If myCode's severity threshold *values* change between versions,
       the prior severity recorded here may not align with current
       thresholds.  This is acceptable: the hysteresis logic will use
       the stale prior severity as a bias, which — in the worst case —
       adds one extra run of "sticky" classification before converging
       to the new threshold.  The next run's saved report will then
       reflect current thresholds and subsequent runs behave normally.
    """
    json_path = project_path / "mycode-report.json"
    if not json_path.is_file():
        return None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not read prior report %s: %s", json_path, exc)
        return None

    if not isinstance(data, dict):
        return None

    state = PriorRunState()

    # Parse findings
    for f in data.get("findings", []):
        if not isinstance(f, dict):
            continue
        title = f.get("title", "")
        category = f.get("category", "")
        severity = f.get("severity", "")
        if title and severity:
            state.finding_severities[finding_key(title, category)] = severity

    # Parse degradation curves
    for dp in data.get("degradation_curves", []):
        if not isinstance(dp, dict):
            continue
        scenario = dp.get("scenario_name", "")
        metric = dp.get("metric", "")
        if scenario and metric:
            state.degradation_keys.add(degradation_key(scenario, metric))

    return state


# ── Core hysteresis function ──

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


def classify_with_hysteresis(
    measurement: float,
    thresholds: list[tuple[float, str]],
    default_severity: Optional[str],
    prior_severity: Optional[str] = None,
) -> Optional[str]:
    """Classify *measurement* against *thresholds* with optional hysteresis.

    Parameters
    ----------
    measurement:
        The value to classify (e.g. a ratio like ``load_level / user_scale``).
    thresholds:
        Sorted **ascending** by threshold value.  Each ``(threshold, severity)``
        means *"if measurement ≤ threshold, assign this severity"*.  Lower
        thresholds correspond to **higher** severity (rank 0 = most severe).
    default_severity:
        Severity when *measurement* exceeds all thresholds.  ``None`` means
        "no finding" (used by memory-capacity classification).
    prior_severity:
        The severity assigned in the previous run for the same finding.
        ``None`` means first run — use base thresholds directly.

    Returns
    -------
    str | None
        The severity string, or ``None`` when the outcome is "no finding".
    """
    # ── First-run: plain threshold comparison ──
    if prior_severity is None:
        for threshold, severity in thresholds:
            if measurement <= threshold:
                return severity
        return default_severity

    # ── Subsequent run: apply hysteresis margins ──
    #
    # If prior severity is not a known rank, treat as first run.
    if prior_severity not in _SEVERITY_RANK:
        for threshold, severity in thresholds:
            if measurement <= threshold:
                return severity
        return default_severity

    # Determine base classification (what we'd assign without hysteresis).
    base_severity = default_severity
    for threshold, severity in thresholds:
        if measurement <= threshold:
            base_severity = severity
            break

    # If base and prior agree, no conflict.
    if base_severity == prior_severity:
        return base_severity

    # Determine severity ranks (lower number = more severe).
    prior_rank = _SEVERITY_RANK[prior_severity]
    base_rank = _SEVERITY_RANK.get(base_severity, 99) if base_severity else 99

    # Would this be an escalation (moving to higher severity)?
    escalating = base_rank < prior_rank

    if escalating:
        # To escalate, measurement must cross threshold * (1 - margin).
        # Find the threshold that separates prior_severity from base_severity.
        for threshold, severity in thresholds:
            sev_rank = _SEVERITY_RANK.get(severity, 99)
            if sev_rank <= base_rank:
                escalation_threshold = threshold * (1 - HYSTERESIS_MARGIN)
                if measurement <= escalation_threshold:
                    return severity  # escalation confirmed
        # Escalation not confirmed — keep prior severity.
        return prior_severity
    else:
        # De-escalating (moving to lower severity).
        # To de-escalate, measurement must cross threshold * (1 + margin).
        # Find the threshold that the measurement must exceed to leave
        # the prior (more severe) severity.
        for threshold, severity in thresholds:
            sev_rank = _SEVERITY_RANK.get(severity, 99)
            if sev_rank == prior_rank:
                # This is the threshold that defines the prior severity.
                # To de-escalate past it, measurement must exceed
                # threshold * (1 + margin).
                de_escalation_threshold = threshold * (1 + HYSTERESIS_MARGIN)
                if measurement > de_escalation_threshold:
                    # De-escalation confirmed — use base classification.
                    return base_severity
                else:
                    # Stays at prior severity.
                    return prior_severity

        # Prior severity not found in thresholds (e.g. prior was None/default).
        # Fall through to base classification.
        return base_severity


# ── Degradation detection hysteresis ──

def degradation_threshold_with_hysteresis(
    base_threshold: float,
    existed_in_prior: Optional[bool],
) -> float:
    """Adjust a degradation detection threshold based on prior run.

    Parameters
    ----------
    base_threshold:
        The normal detection threshold (e.g. 2.0 for overall ratio).
    existed_in_prior:
        ``True`` if the degradation point existed in the prior run,
        ``False`` if it did not, ``None`` if this is a first run.

    Returns
    -------
    float
        Adjusted threshold:
        - Prior existed → ``base × (1 − margin)`` (keep reporting)
        - Prior absent  → ``base × (1 + margin)`` (require stronger signal)
        - First run     → ``base`` (no adjustment)
    """
    if existed_in_prior is None:
        return base_threshold
    if existed_in_prior:
        return base_threshold * (1 - HYSTERESIS_MARGIN)
    return base_threshold * (1 + HYSTERESIS_MARGIN)
