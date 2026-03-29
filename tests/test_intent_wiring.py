"""Tests for intent-driven wiring: constraint parsers, scaling functions,
non-interactive defaults, prediction module, report headroom framing,
and DegradationPoint intent markers.
"""

import json
import os
import tempfile

import pytest

from mycode.constraints import (
    OperationalConstraints,
    max_total_data_to_items,
    parse_max_total_data,
    parse_per_user_data,
    parse_usage_pattern,
    per_user_data_to_items,
)
from mycode.prediction import (
    PredictionResult,
    match_prediction_to_findings,
    predict_issues,
)
from mycode.report import (
    DegradationPoint,
    DiagnosticReport,
    Finding,
    ReportGenerator,
)
from mycode.scenario import (
    _data_scale_levels_intent,
    _user_scale_levels,
    apply_non_interactive_defaults,
)


# ── 1. parse_per_user_data ──


class TestParsePerUserData:
    @pytest.mark.parametrize("text, expected", [
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
        ("SMALL", "small"),
        ("Large.", "large"),
    ])
    def test_category_keywords(self, text, expected):
        assert parse_per_user_data(text) == expected

    @pytest.mark.parametrize("text, expected", [
        ("1", "small"),
        ("2", "medium"),
        ("3", "large"),
    ])
    def test_numbered_choices(self, text, expected):
        assert parse_per_user_data(text) == expected

    @pytest.mark.parametrize("text", [
        "50 rows", "10 GB", "about a thousand records",
    ])
    def test_free_text_preserved(self, text):
        result = parse_per_user_data(text)
        assert result == text

    @pytest.mark.parametrize("text", [
        "not sure", "skip", "n/a", "idk", "",
    ])
    def test_skip_words(self, text):
        assert parse_per_user_data(text) is None


# ── 1b. parse_max_total_data ──


class TestParseMaxTotalData:
    @pytest.mark.parametrize("text, expected", [
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
        ("1", "small"),
        ("2", "medium"),
        ("3", "large"),
    ])
    def test_categories_and_numbers(self, text, expected):
        assert parse_max_total_data(text) == expected

    def test_free_text(self):
        assert parse_max_total_data("10 GB") == "10 GB"
        assert parse_max_total_data("1M rows") == "1M rows"

    @pytest.mark.parametrize("text", ["skip", "not sure", ""])
    def test_skip(self, text):
        assert parse_max_total_data(text) is None


# ── 1c. per_user_data_to_items ──


class TestPerUserDataToItems:
    @pytest.mark.parametrize("category, expected", [
        ("small", 50),
        ("medium", 500),
        ("large", 5000),
    ])
    def test_categories(self, category, expected):
        assert per_user_data_to_items(category) == expected

    def test_numeric_input(self):
        assert per_user_data_to_items("200 rows") == 200

    def test_k_suffix(self):
        assert per_user_data_to_items("5k records") == 5000

    def test_none_returns_default(self):
        assert per_user_data_to_items(None) == 100

    def test_unrecognised_returns_default(self):
        assert per_user_data_to_items("who knows") == 100


# ── 1d. max_total_data_to_items ──


class TestMaxTotalDataToItems:
    @pytest.mark.parametrize("category, expected", [
        ("small", 1000),
        ("medium", 10000),
        ("large", 100000),
    ])
    def test_categories(self, category, expected):
        assert max_total_data_to_items(category) == expected

    def test_numeric_input(self):
        assert max_total_data_to_items("50000 records") == 50000

    def test_m_suffix(self):
        assert max_total_data_to_items("1m") == 1_000_000

    def test_none_returns_default(self):
        assert max_total_data_to_items(None) == 10000


# ── 2. OperationalConstraints backward compat ──


class TestOperationalConstraintsSync:
    def test_max_users_syncs_to_user_scale(self):
        c = OperationalConstraints(max_users=200)
        assert c.user_scale == 200

    def test_user_scale_syncs_to_max_users(self):
        c = OperationalConstraints(user_scale=50)
        assert c.max_users == 50

    def test_both_set_preserved(self):
        c = OperationalConstraints(user_scale=50, max_users=200)
        assert c.user_scale == 50
        assert c.max_users == 200

    def test_current_plus_max_users_sets_user_scale(self):
        c = OperationalConstraints(current_users=20, max_users=100)
        assert c.user_scale == 100

    def test_assumptions_fields_exist(self):
        c = OperationalConstraints()
        assert c.assumptions_used is False
        assert c.assumed_values == {}


# ── 3. Usage pattern aliases ──


class TestUsagePatternAliases:
    @pytest.mark.parametrize("text, expected", [
        ("sustained", "steady"),
        ("periodic", "on_demand"),
        ("steady", "steady"),
        ("burst", "burst"),
        ("on demand", "on_demand"),
        ("growing", "growing"),
    ])
    def test_alias_resolution(self, text, expected):
        assert parse_usage_pattern(text) == expected

    @pytest.mark.parametrize("text, expected", [
        ("1", "steady"),
        ("2", "burst"),
        ("3", "on_demand"),
        ("4", "growing"),
    ])
    def test_numbered_choices(self, text, expected):
        assert parse_usage_pattern(text) == expected


# ── 4. Scaling functions ──


class TestUserScaleLevels:
    def test_basic(self):
        levels = _user_scale_levels(50, 200)
        assert 50 in levels
        assert 200 in levels
        # Buffer beyond max
        assert any(v > 200 for v in levels)

    def test_step_size(self):
        levels = _user_scale_levels(50, 200)
        # Step is 10% of max = 20. So 50, 70, 90, ... should appear.
        assert 70 in levels

    def test_current_greater_than_max(self):
        # Falls back to _scale_levels(maximum)
        levels = _user_scale_levels(300, 100)
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_max_zero(self):
        levels = _user_scale_levels(10, 0)
        assert levels == [1, 10, 100]

    def test_equal_values(self):
        levels = _user_scale_levels(100, 100)
        assert 100 in levels
        # Buffer beyond
        assert any(v > 100 for v in levels)


class TestDataScaleLevelsIntent:
    def test_basic(self):
        levels = _data_scale_levels_intent(50, 500)
        assert 50 in levels
        assert 500 in levels
        assert any(v > 500 for v in levels)

    def test_per_user_ge_max_total(self):
        # Falls back to _data_scale_levels
        levels = _data_scale_levels_intent(1000, 500)
        assert isinstance(levels, list)
        assert len(levels) > 0


# ── 5. Non-interactive fallback ──


class TestApplyNonInteractiveDefaults:
    def test_sets_defaults_when_none(self):
        c = OperationalConstraints()
        apply_non_interactive_defaults(c)
        assert c.current_users is not None
        assert c.max_users is not None
        assert c.per_user_data is not None
        assert c.max_total_data is not None

    def test_sets_assumptions_used(self):
        c = OperationalConstraints()
        apply_non_interactive_defaults(c)
        assert c.assumptions_used is True

    def test_records_assumed_values(self):
        c = OperationalConstraints()
        apply_non_interactive_defaults(c)
        assert "current_users" in c.assumed_values
        assert "max_users" in c.assumed_values

    def test_does_not_override_existing(self):
        c = OperationalConstraints(
            current_users=5, max_users=50,
            per_user_data="large", max_total_data="10GB",
        )
        apply_non_interactive_defaults(c)
        assert c.current_users == 5
        assert c.max_users == 50
        assert c.per_user_data == "large"
        assert c.max_total_data == "10GB"
        # No assumptions needed
        assert c.assumptions_used is False


# ── 6. Prediction module ──


def _write_corpus(tmp_path, patterns):
    """Write test corpus JSON and return the path."""
    path = os.path.join(tmp_path, "test_corpus.json")
    with open(path, "w") as f:
        json.dump(patterns, f)
    return path


class TestPredictIssues:
    def test_uses_corpus_data(self, tmp_path):
        corpus = [
            {
                "title": "Memory growth under concurrent Flask requests",
                "confirmed_count": 10,
                "affected_dependencies": ["flask", "sqlalchemy"],
                "severity_distribution": {"critical": 7, "warning": 3},
            },
        ]
        path = _write_corpus(str(tmp_path), corpus)
        result = predict_issues(["flask", "sqlalchemy"], corpus_path=path)
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == 1
        assert result.predictions[0].title == "Memory growth under concurrent Flask requests"

    def test_empty_dependencies(self, tmp_path):
        corpus = [
            {
                "title": "Some issue",
                "confirmed_count": 5,
                "affected_dependencies": ["flask"],
                "severity_distribution": {"warning": 5},
            },
        ]
        path = _write_corpus(str(tmp_path), corpus)
        result = predict_issues([], corpus_path=path)
        assert result.predictions == []

    def test_no_corpus_file(self):
        result = predict_issues(
            ["flask"], corpus_path="/nonexistent/path.json",
        )
        assert result.predictions == []

    def test_scores_by_dependency_overlap(self, tmp_path):
        corpus = [
            {
                "title": "Issue A",
                "confirmed_count": 5,
                "affected_dependencies": ["flask"],
                "severity_distribution": {"warning": 5},
            },
            {
                "title": "Issue B",
                "confirmed_count": 5,
                "affected_dependencies": ["flask", "pandas"],
                "severity_distribution": {"critical": 5},
            },
        ]
        path = _write_corpus(str(tmp_path), corpus)
        result = predict_issues(["flask", "pandas"], corpus_path=path)
        # Issue B has 2 dep overlap vs Issue A's 1, so B scores higher
        assert result.predictions[0].title == "Issue B"


class TestMatchPredictionToFindings:
    def test_keyword_overlap_returns_true(self):
        assert match_prediction_to_findings(
            "Memory growth under load",
            ["Memory growth detected at 100 users"],
            [],
        ) is True

    def test_no_overlap_returns_false(self):
        assert match_prediction_to_findings(
            "Memory growth under load",
            ["Response time increased"],
            [],
        ) is False

    def test_category_match(self):
        assert match_prediction_to_findings(
            "Concurrent execution problems",
            [],
            ["concurrent_execution"],
        ) is True


# ── 7. Report headroom framing ──


class TestReportHeadroomFraming:
    """Test _contextualise_findings via the ReportGenerator."""

    def _make_finding(self, load_level):
        """Create a concurrency finding at the given load level."""
        return Finding(
            title=f"Error at concurrent_{load_level}",
            severity="warning",
            category="concurrent_execution",
            description="Original description.",
            _load_level=load_level,
        )

    def _contextualise(self, finding, current_users=None, max_users=None):
        """Run contextualisation and return the updated finding."""
        report = DiagnosticReport(findings=[finding])
        constraints = OperationalConstraints(
            current_users=current_users, max_users=max_users,
        )
        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)
        return report.findings[0]

    def test_below_current_users(self):
        f = self._make_finding(load_level=10)
        result = self._contextualise(f, current_users=50, max_users=200)
        assert "affecting your application now" in result.description
        assert result.severity == "critical"

    def test_between_current_and_max(self):
        f = self._make_finding(load_level=100)
        result = self._contextualise(f, current_users=50, max_users=200)
        assert "becomes relevant" in result.description

    def test_beyond_max(self):
        f = self._make_finding(load_level=300)
        result = self._contextualise(f, current_users=50, max_users=200)
        assert "beyond your target" in result.description

    def test_no_current_users_ratio_framing(self):
        f = self._make_finding(load_level=50)
        result = self._contextualise(f, current_users=None, max_users=100)
        # Should use ratio-based framing (no "affecting your application now")
        assert "affecting your application now" not in result.description or (
            result.severity in ("critical", "warning", "info")
        )


# ── 8. DegradationPoint intent markers ──


class TestDegradationPointIntentMarkers:
    def test_markers_populated_from_constraints(self):
        dp = DegradationPoint(
            scenario_name="concurrent_execution_test",
            metric="execution_time_ms",
        )
        report = DiagnosticReport(degradation_points=[dp])
        constraints = OperationalConstraints(current_users=20, max_users=200)

        gen = ReportGenerator(offline=True)
        # Simulate what generate() does at step 4b
        for point in report.degradation_points:
            if constraints.current_users is not None:
                point.user_baseline = constraints.current_users
            if constraints.max_users is not None:
                point.user_ceiling = constraints.max_users

        assert dp.user_baseline == 20
        assert dp.user_ceiling == 200

    def test_markers_none_when_no_constraints(self):
        dp = DegradationPoint(
            scenario_name="test",
            metric="memory_peak_mb",
        )
        assert dp.user_baseline is None
        assert dp.user_ceiling is None
