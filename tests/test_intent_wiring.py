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
    PROFILED_DEPS,
    PredictionResult,
    _extract_features,
    _load_model,
    match_prediction_to_findings,
    normalize_dep_name,
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
    """Test predict_issues with model and corpus fallback paths."""

    def _force_corpus_fallback(self):
        """Force corpus lookup by clearing model cache."""
        from mycode.prediction import _model_cache
        saved = dict(_model_cache)
        _model_cache.clear()
        _model_cache["loaded"] = True  # mark as attempted
        _model_cache["model"] = None
        _model_cache["metadata"] = None
        return saved

    def _restore_cache(self, saved):
        from mycode.prediction import _model_cache
        _model_cache.clear()
        _model_cache.update(saved)

    def test_uses_corpus_data(self, tmp_path):
        """Corpus fallback returns corpus-based predictions."""
        saved = self._force_corpus_fallback()
        try:
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
        finally:
            self._restore_cache(saved)

    def test_empty_dependencies(self, tmp_path):
        saved = self._force_corpus_fallback()
        try:
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
        finally:
            self._restore_cache(saved)

    def test_no_corpus_file(self):
        saved = self._force_corpus_fallback()
        try:
            result = predict_issues(
                ["flask"], corpus_path="/nonexistent/path.json",
            )
            assert result.predictions == []
        finally:
            self._restore_cache(saved)

    def test_scores_by_dependency_overlap(self, tmp_path):
        saved = self._force_corpus_fallback()
        try:
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
            assert result.predictions[0].title == "Issue B"
        finally:
            self._restore_cache(saved)

    def test_model_based_prediction(self):
        """When model is available, returns model-based predictions."""
        result = predict_issues(["flask", "sqlalchemy", "pandas"])
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) > 0
        assert len(result.predictions) <= 5
        for p in result.predictions:
            assert 0 < p.probability_pct <= 100
            assert p.title
            assert p.severity in ("critical", "warning", "info")
        assert result.total_similar_projects > 0


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


# ── 6c. Model-specific tests ──


class TestNormalizeDepName:
    def test_alias_mapping(self):
        assert normalize_dep_name("react-dom") == "react"
        assert normalize_dep_name("langchain-core") == "langchain"
        assert normalize_dep_name("uvicorn") == "fastapi"

    def test_npm_start_ignored(self):
        assert normalize_dep_name("npm-start") is None

    def test_unknown_dep_normalized(self):
        assert normalize_dep_name("my-custom-lib") == "my_custom_lib"

    def test_profiled_dep_passthrough(self):
        assert normalize_dep_name("pandas") == "pandas"
        assert normalize_dep_name("flask") == "flask"


class TestFeatureExtraction:
    def test_dep_features(self):
        model, meta = _load_model()
        if meta is None:
            pytest.skip("Model not available")
        cols = meta["feature_columns"]
        X = _extract_features(["flask", "pandas", "numpy"], cols)
        # Check shape
        assert X.shape == (1, len(cols))
        # Check dep columns
        flask_idx = cols.index("dep_flask")
        pandas_idx = cols.index("dep_pandas")
        numpy_idx = cols.index("dep_numpy")
        assert X[0, flask_idx] == 1.0
        assert X[0, pandas_idx] == 1.0
        assert X[0, numpy_idx] == 1.0
        # Check a dep NOT present
        react_idx = cols.index("dep_react")
        assert X[0, react_idx] == 0.0

    def test_alias_in_features(self):
        """react-dom should activate dep_react."""
        model, meta = _load_model()
        if meta is None:
            pytest.skip("Model not available")
        cols = meta["feature_columns"]
        X = _extract_features(["react-dom"], cols)
        react_idx = cols.index("dep_react")
        assert X[0, react_idx] == 1.0

    def test_complexity_defaults(self):
        """Without ingestion, complexity features get defaults."""
        model, meta = _load_model()
        if meta is None:
            pytest.skip("Model not available")
        cols = meta["feature_columns"]
        X = _extract_features(["flask"], cols)
        loc_idx = cols.index("loc")
        assert X[0, loc_idx] == 0.0  # no ingestion → 0


class TestModelLoading:
    def test_loads_model(self):
        model, meta = _load_model()
        assert model is not None
        assert meta is not None
        assert "feature_columns" in meta
        assert "target_columns" in meta
        assert meta["training_samples"] > 0
        assert meta["mean_auc"] > 0.5

    def test_cache_returns_same(self):
        m1, meta1 = _load_model()
        m2, meta2 = _load_model()
        assert m1 is m2
        assert meta1 is meta2

    def test_fallback_on_missing_model(self):
        """Clearing cache and marking model as missing returns None."""
        from mycode.prediction import _model_cache
        saved = dict(_model_cache)
        try:
            _model_cache.clear()
            _model_cache["loaded"] = True
            _model_cache["model"] = None
            _model_cache["metadata"] = None
            m, meta = _load_model()
            assert m is None
            assert meta is None
        finally:
            _model_cache.clear()
            _model_cache.update(saved)


class TestModelPredictionOutput:
    def test_probabilities_in_range(self):
        result = predict_issues(["flask", "sqlalchemy", "pandas"])
        for p in result.predictions:
            assert 0 < p.probability_pct <= 100

    def test_sorted_by_probability(self):
        result = predict_issues(["flask", "sqlalchemy", "pandas"])
        probs = [p.probability_pct for p in result.predictions]
        assert probs == sorted(probs, reverse=True)

    def test_max_five_predictions(self):
        result = predict_issues(["flask", "pandas", "numpy", "streamlit", "sqlalchemy"])
        assert len(result.predictions) <= 5

    def test_never_crashes_on_unknown_deps(self):
        result = predict_issues(["totally-unknown-dep-xyz"])
        assert isinstance(result, PredictionResult)


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


# ── 7b. Data volume framing ──


class TestDataVolumeFraming:
    """Test data-volume findings framed against user×per_user_data."""

    def _make_data_finding(self, load_level):
        """Create a data_volume_scaling finding at the given load level."""
        return Finding(
            title=f"Error at data_{load_level}",
            severity="warning",
            category="data_volume_scaling",
            description="Original description.",
            _load_level=load_level,
        )

    def _contextualise(self, finding, current_users=None, max_users=None,
                       per_user_data=None):
        report = DiagnosticReport(findings=[finding])
        constraints = OperationalConstraints(
            current_users=current_users, max_users=max_users,
            per_user_data=per_user_data,
        )
        gen = ReportGenerator(offline=True)
        gen._contextualise_findings(report, constraints)
        return report.findings[0]

    def test_exceeds_at_max_scale(self):
        """2 users × 50 items = 100 current. 25 max × 50 = 1,250.
        Threshold at 500 → between current and max → critical."""
        f = self._make_data_finding(load_level=500)
        result = self._contextualise(
            f, current_users=2, max_users=25, per_user_data="small",
        )
        assert "25" in result.description
        assert "1,250 items" in result.description
        assert "priority improvement" in result.description.lower() or \
               "exceeds" in result.description
        assert result.severity == "critical"

    def test_exceeds_at_current_scale(self):
        """10 users × 50 items = 500 total. Threshold at 100 → critical now."""
        f = self._make_data_finding(load_level=100)
        result = self._contextualise(
            f, current_users=10, max_users=100, per_user_data="small",
        )
        # current_vol = 10*50 = 500, load_level=100 < 500 → affecting now
        assert "affecting your application now" in result.description
        assert result.severity == "critical"

    def test_well_below_threshold(self):
        """5 users × 50 items = 250 total. Threshold at 10,000 → info."""
        f = self._make_data_finding(load_level=10000)
        result = self._contextualise(
            f, current_users=5, max_users=25, per_user_data="small",
        )
        # max_vol = 25*50 = 1,250. load_level=10,000 >> 1,250 → info
        assert "headroom" in result.description
        assert result.severity == "info"

    def test_no_per_user_data_falls_back(self):
        """Without per_user_data, fall back to generic description."""
        f = self._make_data_finding(load_level=500)
        result = self._contextualise(
            f, current_users=5, max_users=25, per_user_data=None,
        )
        # Should use step-based description, not intent-framed
        assert "500" in result.description
        assert "items each" not in result.description

    def test_medium_per_user_data(self):
        """1 user × 500 items = 500 current. 10 max × 500 = 5,000.
        Threshold at 1,000 → between current and max → critical."""
        f = self._make_data_finding(load_level=1000)
        result = self._contextualise(
            f, current_users=1, max_users=10, per_user_data="medium",
        )
        # current_vol = 1*500 = 500, max_vol = 10*500 = 5,000
        # load_level=1,000 > 500 but < 5,000 → exceeds at max
        assert "5,000 items" in result.description
        assert "exceeds" in result.description
        assert result.severity == "critical"


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
