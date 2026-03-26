"""Tests for threshold hysteresis in severity classification."""

import json

import pytest

from mycode.hysteresis import (
    HYSTERESIS_MARGIN,
    PriorRunState,
    classify_with_hysteresis,
    degradation_key,
    degradation_threshold_with_hysteresis,
    finding_key,
    load_prior_state,
)

# ── Thresholds used across tests ──
# Mirrors _contextualise_findings: [(1.0, "critical"), (3.0, "warning")], default="info"
CTX_THRESHOLDS = [(1.0, "critical"), (3.0, "warning")]
CTX_DEFAULT = "info"

# Mirrors _endpoint_to_finding: [(1.0, "critical"), (2.0, "warning")], default="info"
HTTP_THRESHOLDS = [(1.0, "critical"), (2.0, "warning")]
HTTP_DEFAULT = "info"

# Mirrors _memory_capacity_finding: [(0.25, "critical"), (0.75, "warning")], default=None
MEM_THRESHOLDS = [(0.25, "critical"), (0.75, "warning")]
MEM_DEFAULT = None


# ══════════════════════════════════════════════════════════════════
# 1. First run — no prior state → base thresholds
# ══════════════════════════════════════════════════════════════════


class TestFirstRunNoPrior:
    """When prior_severity is None, classify using base thresholds."""

    def test_ratio_below_critical_threshold(self):
        """Ratio 0.9 → critical (≤ 1.0)."""
        result = classify_with_hysteresis(0.9, CTX_THRESHOLDS, CTX_DEFAULT)
        assert result == "critical"

    def test_ratio_in_warning_range(self):
        """Ratio 2.0 → warning (> 1.0 and ≤ 3.0)."""
        result = classify_with_hysteresis(2.0, CTX_THRESHOLDS, CTX_DEFAULT)
        assert result == "warning"

    def test_ratio_above_all_thresholds(self):
        """Ratio 4.0 → info (> 3.0)."""
        result = classify_with_hysteresis(4.0, CTX_THRESHOLDS, CTX_DEFAULT)
        assert result == "info"

    def test_exactly_at_threshold(self):
        """Ratio exactly 1.0 → critical (≤ 1.0)."""
        result = classify_with_hysteresis(1.0, CTX_THRESHOLDS, CTX_DEFAULT)
        assert result == "critical"

    def test_exactly_at_warning_threshold(self):
        """Ratio exactly 3.0 → warning (≤ 3.0)."""
        result = classify_with_hysteresis(3.0, CTX_THRESHOLDS, CTX_DEFAULT)
        assert result == "warning"


# ══════════════════════════════════════════════════════════════════
# 2. Escalation — prior is lower severity, measurement wants higher
# ══════════════════════════════════════════════════════════════════


class TestEscalation:
    """Escalation requires crossing threshold × (1 − margin)."""

    def test_warning_stays_when_near_critical_boundary(self):
        """Prior=warning, ratio=0.95 (above 1.0 × 0.8 = 0.8) → stays warning."""
        result = classify_with_hysteresis(
            0.95, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="warning",
        )
        assert result == "warning"

    def test_warning_escalates_to_critical(self):
        """Prior=warning, ratio=0.75 (below 1.0 × 0.8 = 0.8) → escalates to critical."""
        result = classify_with_hysteresis(
            0.75, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="warning",
        )
        assert result == "critical"

    def test_info_escalates_to_warning(self):
        """Prior=info, ratio=2.5 (below 3.0 × 0.8 = 2.4? No, 2.5 > 2.4) → stays info.

        Actually: base classification at ratio 2.5 is "warning" (≤ 3.0).
        Prior is "info". To escalate info → warning, measurement must be
        ≤ 3.0 × 0.8 = 2.4. 2.5 > 2.4, so escalation NOT confirmed → stays info.
        """
        result = classify_with_hysteresis(
            2.5, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="info",
        )
        assert result == "info"

    def test_info_escalates_to_warning_confirmed(self):
        """Prior=info, ratio=2.3 (below 3.0 × 0.8 = 2.4) → escalates to warning."""
        result = classify_with_hysteresis(
            2.3, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="info",
        )
        assert result == "warning"


# ══════════════════════════════════════════════════════════════════
# 3. De-escalation — prior is higher severity, measurement wants lower
# ══════════════════════════════════════════════════════════════════


class TestDeescalation:
    """De-escalation requires crossing threshold × (1 + margin)."""

    def test_critical_stays_when_near_boundary(self):
        """Prior=critical, ratio=1.1 (below 1.0 × 1.2 = 1.2) → stays critical."""
        result = classify_with_hysteresis(
            1.1, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="critical",
        )
        assert result == "critical"

    def test_critical_deescalates_to_warning(self):
        """Prior=critical, ratio=1.25 (above 1.0 × 1.2 = 1.2) → de-escalates to warning."""
        result = classify_with_hysteresis(
            1.25, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="critical",
        )
        assert result == "warning"

    def test_warning_stays_when_near_info_boundary(self):
        """Prior=warning, ratio=3.1 (below 3.0 × 1.2 = 3.6) → stays warning."""
        result = classify_with_hysteresis(
            3.1, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="warning",
        )
        assert result == "warning"

    def test_warning_deescalates_to_info(self):
        """Prior=warning, ratio=3.7 (above 3.0 × 1.2 = 3.6) → de-escalates to info."""
        result = classify_with_hysteresis(
            3.7, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="info",
        )
        # ratio 3.7 > 3.0, base = info, prior = info → info (no change needed)
        assert result == "info"

    def test_warning_deescalates_to_info_from_warning(self):
        """Prior=warning, ratio=3.7 (above 3.0 × 1.2 = 3.6) → de-escalates to info."""
        result = classify_with_hysteresis(
            3.7, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="warning",
        )
        assert result == "info"


# ══════════════════════════════════════════════════════════════════
# 4. Memory capacity — default_severity=None (no finding)
# ══════════════════════════════════════════════════════════════════


class TestMemoryCapacityHysteresis:
    """Memory capacity uses None as default (no finding)."""

    def test_first_run_below_critical(self):
        """ratio=0.20 → critical."""
        result = classify_with_hysteresis(0.20, MEM_THRESHOLDS, MEM_DEFAULT)
        assert result == "critical"

    def test_first_run_in_warning_range(self):
        """ratio=0.50 → warning."""
        result = classify_with_hysteresis(0.50, MEM_THRESHOLDS, MEM_DEFAULT)
        assert result == "warning"

    def test_first_run_above_all(self):
        """ratio=0.80 → None (no finding)."""
        result = classify_with_hysteresis(0.80, MEM_THRESHOLDS, MEM_DEFAULT)
        assert result is None

    def test_prior_warning_stays_when_slightly_above(self):
        """Prior=warning, ratio=0.78 (above 0.75 but below 0.75 × 1.2 = 0.90) → stays warning."""
        result = classify_with_hysteresis(
            0.78, MEM_THRESHOLDS, MEM_DEFAULT, prior_severity="warning",
        )
        assert result == "warning"

    def test_prior_warning_becomes_none_when_clearly_above(self):
        """Prior=warning, ratio=0.95 (above 0.75 × 1.2 = 0.90) → None (no finding)."""
        result = classify_with_hysteresis(
            0.95, MEM_THRESHOLDS, MEM_DEFAULT, prior_severity="warning",
        )
        assert result is None

    def test_prior_critical_stays_near_boundary(self):
        """Prior=critical, ratio=0.27 (below 0.25 × 1.2 = 0.30) → stays critical."""
        result = classify_with_hysteresis(
            0.27, MEM_THRESHOLDS, MEM_DEFAULT, prior_severity="critical",
        )
        assert result == "critical"

    def test_prior_critical_deescalates(self):
        """Prior=critical, ratio=0.35 (above 0.25 × 1.2 = 0.30) → warning."""
        result = classify_with_hysteresis(
            0.35, MEM_THRESHOLDS, MEM_DEFAULT, prior_severity="critical",
        )
        assert result == "warning"


# ══════════════════════════════════════════════════════════════════
# 5. Degradation detection hysteresis
# ══════════════════════════════════════════════════════════════════


class TestDegradationThreshold:
    def test_first_run(self):
        """No prior → base threshold unchanged."""
        assert degradation_threshold_with_hysteresis(2.0, None) == 2.0

    def test_existed_in_prior(self):
        """Prior had this point → lower threshold (keep reporting)."""
        expected = 2.0 * (1 - HYSTERESIS_MARGIN)  # 1.6
        assert degradation_threshold_with_hysteresis(2.0, True) == expected

    def test_absent_in_prior(self):
        """Prior didn't have this point → raise threshold (require stronger signal)."""
        expected = 2.0 * (1 + HYSTERESIS_MARGIN)  # 2.4
        assert degradation_threshold_with_hysteresis(2.0, False) == expected

    def test_spike_threshold_existed(self):
        """Spike threshold also gets hysteresis."""
        expected = 3.0 * (1 - HYSTERESIS_MARGIN)  # 2.4
        assert degradation_threshold_with_hysteresis(3.0, True) == expected


# ══════════════════════════════════════════════════════════════════
# 6. Prior state loading
# ══════════════════════════════════════════════════════════════════


class TestLoadPriorState:
    def test_no_report_file(self, tmp_path):
        """Missing mycode-report.json → None."""
        result = load_prior_state(tmp_path)
        assert result is None

    def test_corrupt_json(self, tmp_path):
        """Invalid JSON → None, no crash."""
        (tmp_path / "mycode-report.json").write_text("not json{{{", encoding="utf-8")
        result = load_prior_state(tmp_path)
        assert result is None

    def test_valid_report(self, tmp_path):
        """Valid report → parsed PriorRunState."""
        report = {
            "findings": [
                {"title": "Memory growth", "category": "memory_profiling", "severity": "critical"},
                {"title": "Slow response", "category": "http_load_testing", "severity": "warning"},
            ],
            "degradation_curves": [
                {"scenario_name": "mem_test", "metric": "memory_peak_mb"},
            ],
        }
        (tmp_path / "mycode-report.json").write_text(
            json.dumps(report), encoding="utf-8",
        )
        state = load_prior_state(tmp_path)
        assert state is not None
        assert state.finding_severities[finding_key("Memory growth", "memory_profiling")] == "critical"
        assert state.finding_severities[finding_key("Slow response", "http_load_testing")] == "warning"
        assert degradation_key("mem_test", "memory_peak_mb") in state.degradation_keys

    def test_finding_not_in_prior(self, tmp_path):
        """New finding not in prior report → key not in dict."""
        report = {
            "findings": [
                {"title": "Old finding", "category": "x", "severity": "warning"},
            ],
            "degradation_curves": [],
        }
        (tmp_path / "mycode-report.json").write_text(
            json.dumps(report), encoding="utf-8",
        )
        state = load_prior_state(tmp_path)
        assert state is not None
        new_key = finding_key("Brand new finding", "y")
        assert new_key not in state.finding_severities

    def test_non_dict_json(self, tmp_path):
        """JSON that's a list, not a dict → None."""
        (tmp_path / "mycode-report.json").write_text("[1,2,3]", encoding="utf-8")
        result = load_prior_state(tmp_path)
        assert result is None


# ══════════════════════════════════════════════════════════════════
# 7. Key helpers
# ══════════════════════════════════════════════════════════════════


class TestKeyHelpers:
    def test_finding_key(self):
        assert finding_key("Memory growth", "memory_profiling") == "memory_profiling::Memory growth"

    def test_degradation_key(self):
        assert degradation_key("mem_test", "memory_peak_mb") == "mem_test::memory_peak_mb"


# ══════════════════════════════════════════════════════════════════
# 8. HTTP endpoint thresholds
# ══════════════════════════════════════════════════════════════════


class TestHttpEndpointHysteresis:
    """HTTP endpoint uses [(1.0, critical), (2.0, warning)], default=info."""

    def test_first_run_critical(self):
        result = classify_with_hysteresis(0.8, HTTP_THRESHOLDS, HTTP_DEFAULT)
        assert result == "critical"

    def test_prior_critical_stays_near_boundary(self):
        """bp/user_scale = 1.1 (below 1.0 × 1.2 = 1.2) → stays critical."""
        result = classify_with_hysteresis(
            1.1, HTTP_THRESHOLDS, HTTP_DEFAULT, prior_severity="critical",
        )
        assert result == "critical"

    def test_prior_critical_deescalates(self):
        """bp/user_scale = 1.3 (above 1.0 × 1.2 = 1.2) → warning."""
        result = classify_with_hysteresis(
            1.3, HTTP_THRESHOLDS, HTTP_DEFAULT, prior_severity="critical",
        )
        assert result == "warning"

    def test_prior_warning_escalates(self):
        """bp/user_scale = 0.7 (below 1.0 × 0.8 = 0.8) → critical."""
        result = classify_with_hysteresis(
            0.7, HTTP_THRESHOLDS, HTTP_DEFAULT, prior_severity="warning",
        )
        assert result == "critical"

    def test_prior_warning_stays_near_critical(self):
        """bp/user_scale = 0.9 (above 1.0 × 0.8 = 0.8) → stays warning."""
        result = classify_with_hysteresis(
            0.9, HTTP_THRESHOLDS, HTTP_DEFAULT, prior_severity="warning",
        )
        assert result == "warning"


# ══════════════════════════════════════════════════════════════════
# 9. Edge: prior severity not in threshold list
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_unknown_prior_severity_falls_through(self):
        """Prior severity not in _SEVERITY_RANK → treated like first run."""
        result = classify_with_hysteresis(
            0.9, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="unknown",
        )
        # base classification: 0.9 ≤ 1.0 → critical.  Prior rank=99, base rank=0.
        # escalating (0 < 99), but prior rank 99 means prior isn't meaningful.
        # Should still return critical.
        assert result == "critical"

    def test_prior_matches_base_no_change(self):
        """When prior and base agree, return immediately."""
        result = classify_with_hysteresis(
            0.5, CTX_THRESHOLDS, CTX_DEFAULT, prior_severity="critical",
        )
        assert result == "critical"
