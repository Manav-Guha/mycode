"""Tests for the Tier 1 inference engine."""

import json
import tempfile
from pathlib import Path

import pytest

from mycode.inference import (
    CorpusIndex,
    InferenceEngine,
    InferenceResult,
    RiskPrediction,
)


# ── Test Fixtures ──


def _make_corpus(entries: list[dict], tmp_path: Path) -> Path:
    """Write a minimal corpus file and return its path."""
    corpus = {
        "migration_metadata": {
            "schema_version": "1.0",
            "statistics": {"total_entries": len(entries)},
        },
        "entries": entries,
    }
    path = tmp_path / "test_corpus.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")
    return path


def _make_entry(
    dep: str = "flask",
    domain: str = "resource_exhaustion",
    pattern: str | None = None,
    severity: str = "critical",
    scenario: str = "memory_test",
    category: str = "memory_profiling",
    repo: str = "user/repo",
    vertical: str = "web_app",
) -> dict:
    """Create a minimal corpus entry."""
    return {
        "entry_id": "test",
        "source": "corpus_mining",
        "source_batch": "test",
        "mycode_version": "0.1.2",
        "timestamp": "2026-01-01T00:00:00Z",
        "language": "python",
        "failure_domain": domain,
        "failure_pattern": pattern,
        "scenario_name": scenario,
        "scenario_category": category,
        "operational_trigger": "sustained_load",
        "affected_dependencies": [dep],
        "severity_raw": severity,
        "load_level_at_failure": "",
        "breaking_point": "",
        "metric_name": "",
        "metric_start_value": 0.0,
        "metric_end_value": 0.0,
        "multiplier": 0.0,
        "codebase_origin": "github",
        "vertical": vertical,
        "architectural_pattern": "web_app",
        "repo_url": f"https://github.com/{repo}",
    }


# ── CorpusIndex Tests ──


class TestCorpusIndex:
    """Tests for CorpusIndex."""

    def test_load_empty_corpus(self, tmp_path):
        path = _make_corpus([], tmp_path)
        index = CorpusIndex(path)
        assert index.total_repos == 0
        assert index.total_entries == 0

    def test_load_with_entries(self, tmp_path):
        entries = [_make_entry(repo="u/r1"), _make_entry(repo="u/r2")]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        assert index.total_entries == 2
        assert index.total_repos == 2

    def test_dep_findings(self, tmp_path):
        entries = [
            _make_entry(dep="flask"),
            _make_entry(dep="flask"),
            _make_entry(dep="django"),
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        assert len(index.dep_findings("flask")) == 2
        assert len(index.dep_findings("django")) == 1
        assert len(index.dep_findings("unknown")) == 0

    def test_dep_repo_count(self, tmp_path):
        entries = [
            _make_entry(dep="flask", repo="u/r1"),
            _make_entry(dep="flask", repo="u/r1"),
            _make_entry(dep="flask", repo="u/r2"),
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        assert index.dep_repo_count("flask") == 2

    def test_dep_domain_distribution(self, tmp_path):
        entries = [
            _make_entry(dep="flask", domain="resource_exhaustion"),
            _make_entry(dep="flask", domain="resource_exhaustion"),
            _make_entry(dep="flask", domain="concurrency_failure"),
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        dist = index.dep_domain_distribution("flask")
        assert dist["resource_exhaustion"] == 2
        assert dist["concurrency_failure"] == 1

    def test_dep_severity_distribution(self, tmp_path):
        entries = [
            _make_entry(dep="flask", severity="critical"),
            _make_entry(dep="flask", severity="critical"),
            _make_entry(dep="flask", severity="warning"),
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        dist = index.dep_severity_distribution("flask")
        assert dist["critical"] == 2
        assert dist["warning"] == 1

    def test_combo_findings(self, tmp_path):
        entries = [
            {
                **_make_entry(),
                "affected_dependencies": ["flask", "sqlalchemy"],
            },
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        assert len(index.combo_findings(["flask", "sqlalchemy"])) == 1
        assert len(index.combo_findings(["sqlalchemy", "flask"])) == 1  # order doesn't matter
        assert len(index.combo_findings(["flask", "django"])) == 0

    def test_missing_file_graceful(self, tmp_path):
        index = CorpusIndex(tmp_path / "nonexistent.json")
        assert index.total_repos == 0
        assert index.total_entries == 0

    def test_vertical_repo_count(self, tmp_path):
        entries = [
            _make_entry(repo="u/r1", vertical="web_app"),
            _make_entry(repo="u/r2", vertical="web_app"),
            _make_entry(repo="u/r3", vertical="dashboard"),
        ]
        path = _make_corpus(entries, tmp_path)
        index = CorpusIndex(path)
        assert index.vertical_repo_count("web_app") == 2
        assert index.vertical_repo_count("dashboard") == 1


# ── RiskPrediction Tests ──


class TestRiskPrediction:
    """Tests for RiskPrediction."""

    def test_as_dict(self):
        p = RiskPrediction(
            risk_summary="Test risk",
            failure_domain="resource_exhaustion",
            confidence="high",
            evidence_count=10,
            occurrence_rate=85.0,
        )
        d = p.as_dict()
        assert d["risk_summary"] == "Test risk"
        assert d["failure_domain"] == "resource_exhaustion"
        assert d["confidence"] == "high"
        assert d["occurrence_rate"] == 85.0


# ── InferenceResult Tests ──


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_as_dict(self):
        result = InferenceResult(
            project_vertical="web_app",
            project_architecture="web_app",
            corpus_size=100,
            matched_dependencies=["flask"],
            unmatched_dependencies=["custom_lib"],
        )
        d = result.as_dict()
        assert d["project_vertical"] == "web_app"
        assert d["corpus_size"] == 100

    def test_as_text_no_predictions(self):
        result = InferenceResult(
            project_vertical="utility",
            project_architecture="utility",
            corpus_size=168,
        )
        text = result.as_text()
        assert "Tier 1" in text
        assert "168" in text
        assert "utility" in text

    def test_as_text_with_predictions(self):
        result = InferenceResult(
            project_vertical="web_app",
            corpus_size=168,
            predictions=[
                RiskPrediction(
                    risk_summary="High risk test",
                    failure_domain="resource_exhaustion",
                    confidence="high",
                    evidence_count=50,
                    occurrence_rate=90.0,
                    severity_distribution={"critical": 40, "warning": 10},
                ),
            ],
        )
        text = result.as_text()
        assert "High Confidence" in text
        assert "High risk test" in text
        assert "40 critical" in text


# ── InferenceEngine Tests ──


class TestInferenceEngine:
    """Tests for InferenceEngine."""

    def test_empty_corpus(self, tmp_path):
        path = _make_corpus([], tmp_path)
        engine = InferenceEngine(corpus_path=path)
        result = engine.infer(dependencies=["flask"])
        assert result.corpus_size == 0
        assert len(result.predictions) == 0

    def test_single_dep_high_confidence(self, tmp_path):
        # Create 10 findings all resource_exhaustion for streamlit
        entries = [
            _make_entry(dep="streamlit", domain="resource_exhaustion", repo=f"u/r{i}")
            for i in range(10)
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["streamlit"])
        assert len(result.predictions) >= 1
        assert result.matched_dependencies == ["streamlit"]
        top = result.predictions[0]
        assert top.failure_domain == "resource_exhaustion"
        assert top.confidence == "high"
        assert top.occurrence_rate == 100.0

    def test_unmatched_deps(self, tmp_path):
        entries = [_make_entry(dep="flask")]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["flask", "custom_lib"])
        assert "custom_lib" in result.unmatched_dependencies

    def test_minimum_evidence_threshold(self, tmp_path):
        # Only 2 entries — below threshold of 3
        entries = [
            _make_entry(dep="flask", repo="u/r1"),
            _make_entry(dep="flask", repo="u/r2"),
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["flask"])
        # Should have 0 predictions (below minimum evidence)
        assert len(result.predictions) == 0

    def test_project_classification(self, tmp_path):
        path = _make_corpus([], tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(
            dependencies=["streamlit", "pandas"],
            file_structure=["app.py"],
        )
        assert result.project_vertical == "dashboard"

    def test_combo_prediction(self, tmp_path):
        entries = [
            {
                **_make_entry(domain="integration_failure", repo=f"u/r{i}"),
                "affected_dependencies": ["flask", "sqlalchemy"],
            }
            for i in range(5)
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["flask", "sqlalchemy"])
        # Should have combo prediction
        combo_preds = [
            p for p in result.predictions
            if len(p.affected_dependencies) == 2
        ]
        assert len(combo_preds) >= 1
        assert combo_preds[0].failure_domain == "integration_failure"

    def test_pattern_level_prediction(self, tmp_path):
        entries = [
            _make_entry(
                dep="streamlit",
                domain="resource_exhaustion",
                pattern="unbounded_cache_growth",
                repo=f"u/r{i}",
            )
            for i in range(5)
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["streamlit"])
        pattern_preds = [p for p in result.predictions if p.failure_pattern]
        assert len(pattern_preds) >= 1
        assert pattern_preds[0].failure_pattern == "unbounded_cache_growth"

    def test_deduplication(self, tmp_path):
        # All same domain for same dep — should deduplicate
        entries = [
            _make_entry(dep="pandas", domain="resource_exhaustion", repo=f"u/r{i}")
            for i in range(10)
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["pandas"])
        # Should not have multiple predictions for same domain+dep
        domain_dep_combos = [
            (p.failure_domain, tuple(sorted(p.affected_dependencies)))
            for p in result.predictions
        ]
        assert len(domain_dep_combos) == len(set(domain_dep_combos))

    def test_skips_unclassified_top_domain(self, tmp_path):
        entries = [
            _make_entry(dep="flask", domain="unclassified", repo=f"u/r{i}")
            for i in range(5)
        ]
        path = _make_corpus(entries, tmp_path)
        engine = InferenceEngine(corpus_path=path)

        result = engine.infer(dependencies=["flask"])
        # Should not predict "unclassified" as the top domain
        for p in result.predictions:
            assert p.failure_domain != "unclassified"


# ── CLI Integration Tests ──


class TestCLITierFlag:
    """Test that --tier flag is properly handled by CLI."""

    def test_parser_accepts_tier(self):
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([".", "--tier", "1"])
        assert args.tier == 1

    def test_parser_accepts_tier_2(self):
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([".", "--tier", "2"])
        assert args.tier == 2

    def test_parser_accepts_tier_3(self):
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([".", "--tier", "3"])
        assert args.tier == 3

    def test_parser_no_tier_default_none(self):
        from mycode.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["."])
        assert args.tier is None

    def test_parser_rejects_invalid_tier(self):
        from mycode.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([".", "--tier", "4"])
