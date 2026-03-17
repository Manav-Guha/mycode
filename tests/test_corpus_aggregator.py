"""Tests for corpus_aggregator — cross-directory report aggregation."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from corpus_aggregator import (
    _repo_key_from_dir,
    _dir_priority,
    _get_vertical,
    _get_arch_pattern,
    _get_actionable_findings,
    _finding_signature,
    _finding_deps,
    _all_deps,
    load_reports,
    aggregate,
    print_summary,
    main,
)


# ── Fixtures ──


def _report(
    vertical="web_app",
    architectural_pattern="fastapi",
    findings=None,
    deps=None,
):
    """Build a minimal mycode report dict."""
    return {
        "project": {
            "name": "test",
            "path": "/tmp/repo",
            "language": "python",
            "dependencies": [
                {"name": d, "installed_version": "1.0.0"}
                for d in (deps or [])
            ],
        },
        "vertical": vertical,
        "architectural_pattern": architectural_pattern,
        "findings": findings or [],
        "statistics": {"scenarios_run": 5, "scenarios_passed": 3},
    }


def _finding(severity="warning", category="memory", title="High memory", deps=None):
    return {
        "severity": severity,
        "category": category,
        "title": title,
        "affected_dependencies": deps or [],
    }


def _write_report(results_dir, repo_name, report):
    """Write a report JSON into results_dir/repo_name/mycode-report.json."""
    repo_dir = results_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "mycode-report.json").write_text(json.dumps(report))


# ── Unit Tests: Helpers ──


class TestRepoKey:
    def test_normalises_case(self):
        assert _repo_key_from_dir("Owner__Repo") == "owner__repo"

    def test_strips_whitespace(self):
        assert _repo_key_from_dir("  foo__bar  ") == "foo__bar"


class TestDirPriority:
    def test_results_highest(self):
        assert _dir_priority(Path("results")) == 3

    def test_corpus_results_lowest(self):
        assert _dir_priority(Path("corpus_results")) == 0

    def test_unknown_dir_defaults_zero(self):
        assert _dir_priority(Path("other")) == 0

    def test_priority_ordering(self):
        dirs = ["corpus_results", "corpus_results_timeout", "corpus_results_retry", "results"]
        priorities = [_dir_priority(Path(d)) for d in dirs]
        assert priorities == sorted(priorities)


class TestGetVertical:
    def test_returns_vertical(self):
        assert _get_vertical({"vertical": "ml_model"}) == "ml_model"

    def test_missing_defaults_unclassified(self):
        assert _get_vertical({}) == "unclassified"

    def test_none_defaults_unclassified(self):
        assert _get_vertical({"vertical": None}) == "unclassified"

    def test_empty_string_defaults_unclassified(self):
        assert _get_vertical({"vertical": ""}) == "unclassified"


class TestGetArchPattern:
    def test_returns_pattern(self):
        assert _get_arch_pattern({"architectural_pattern": "streamlit"}) == "streamlit"

    def test_missing_defaults_unclassified(self):
        assert _get_arch_pattern({}) == "unclassified"


class TestActionableFindings:
    def test_filters_critical_and_warning(self):
        findings = [
            _finding(severity="critical"),
            _finding(severity="warning"),
            _finding(severity="info"),
        ]
        result = _get_actionable_findings({"findings": findings})
        assert len(result) == 2
        assert all(f["severity"] in ("critical", "warning") for f in result)

    def test_empty_findings(self):
        assert _get_actionable_findings({"findings": []}) == []

    def test_no_findings_key(self):
        assert _get_actionable_findings({}) == []


class TestFindingSignature:
    def test_format(self):
        f = _finding(severity="critical", category="memory", title="OOM")
        assert _finding_signature(f) == "critical:memory:OOM"

    def test_truncates_at_120(self):
        f = _finding(title="x" * 200)
        assert len(_finding_signature(f)) == 120


class TestFindingDeps:
    def test_extracts_deps(self):
        f = _finding(deps=["pandas==1.0", "numpy"])
        assert _finding_deps(f) == ["pandas", "numpy"]

    def test_empty(self):
        assert _finding_deps(_finding()) == []


class TestAllDeps:
    def test_extracts_from_project(self):
        r = _report(deps=["flask", "sqlalchemy"])
        assert _all_deps(r) == ["flask", "sqlalchemy"]

    def test_empty_deps(self):
        assert _all_deps(_report()) == []


# ── Integration Tests: load_reports ──


class TestLoadReports:
    def test_loads_single_dir(self, tmp_path):
        results = tmp_path / "results"
        _write_report(results, "user__app", _report())

        reports = load_reports([results])
        assert len(reports) == 1
        assert "user__app" in reports

    def test_skips_missing_dir(self, tmp_path):
        reports = load_reports([tmp_path / "nonexistent"])
        assert reports == {}

    def test_skips_dir_without_report(self, tmp_path):
        results = tmp_path / "results"
        repo_dir = results / "user__app"
        repo_dir.mkdir(parents=True)
        # No mycode-report.json

        reports = load_reports([results])
        assert reports == {}

    def test_skips_invalid_json(self, tmp_path):
        results = tmp_path / "results"
        repo_dir = results / "user__app"
        repo_dir.mkdir(parents=True)
        (repo_dir / "mycode-report.json").write_text("not json{{{")

        reports = load_reports([results])
        assert reports == {}

    def test_dedup_higher_priority_wins(self, tmp_path):
        old_dir = tmp_path / "corpus_results"
        new_dir = tmp_path / "results"

        old_report = _report(vertical="old_vertical")
        new_report = _report(vertical="new_vertical")

        _write_report(old_dir, "user__app", old_report)
        _write_report(new_dir, "user__app", new_report)

        reports = load_reports([old_dir, new_dir])
        assert len(reports) == 1
        assert reports["user__app"]["vertical"] == "new_vertical"

    def test_dedup_order_independent(self, tmp_path):
        """Higher-priority dir wins regardless of scan order."""
        old_dir = tmp_path / "corpus_results"
        new_dir = tmp_path / "results"

        _write_report(old_dir, "user__app", _report(vertical="old"))
        _write_report(new_dir, "user__app", _report(vertical="new"))

        # Pass new_dir first, old_dir second — results/ should still win
        reports = load_reports([new_dir, old_dir])
        assert reports["user__app"]["vertical"] == "new"

    def test_dedup_case_insensitive(self, tmp_path):
        dir1 = tmp_path / "results"
        dir2 = tmp_path / "corpus_results"

        _write_report(dir1, "User__App", _report(vertical="v1"))
        _write_report(dir2, "user__app", _report(vertical="v2"))

        reports = load_reports([dir1, dir2])
        assert len(reports) == 1
        # results/ has higher priority
        assert reports["user__app"]["vertical"] == "v1"

    def test_multiple_repos(self, tmp_path):
        results = tmp_path / "results"
        _write_report(results, "user__app1", _report(vertical="web_app"))
        _write_report(results, "user__app2", _report(vertical="ml_model"))

        reports = load_reports([results])
        assert len(reports) == 2


# ── Integration Tests: aggregate ──


class TestAggregate:
    def test_basic_counts(self):
        reports = {
            "a": _report(vertical="web_app", architectural_pattern="fastapi"),
            "b": _report(vertical="web_app", architectural_pattern="flask"),
            "c": _report(vertical="ml_model", architectural_pattern="fastapi"),
        }
        result = aggregate(reports)

        assert result["total_repos"] == 3
        assert result["verticals"]["web_app"]["repo_count"] == 2
        assert result["verticals"]["ml_model"]["repo_count"] == 1
        assert result["architectural_patterns"]["fastapi"]["repo_count"] == 2
        assert result["architectural_patterns"]["flask"]["repo_count"] == 1

    def test_failure_rate(self):
        reports = {
            "clean": _report(vertical="web_app", findings=[]),
            "dirty": _report(vertical="web_app", findings=[_finding(severity="critical")]),
        }
        result = aggregate(reports)

        v = result["verticals"]["web_app"]
        assert v["repo_count"] == 2
        assert v["repos_with_issues"] == 1
        assert v["repos_clean"] == 1
        assert v["failure_rate"] == 0.5

    def test_info_findings_dont_count_as_issues(self):
        reports = {
            "a": _report(vertical="web_app", findings=[_finding(severity="info")]),
        }
        result = aggregate(reports)
        assert result["verticals"]["web_app"]["repos_with_issues"] == 0

    def test_top_signatures(self):
        findings = [
            _finding(severity="warning", category="mem", title="leak"),
            _finding(severity="warning", category="mem", title="leak"),
            _finding(severity="critical", category="cpu", title="spike"),
        ]
        reports = {"a": _report(vertical="v", findings=findings)}
        result = aggregate(reports)

        sigs = result["verticals"]["v"]["top_failure_signatures"]
        assert len(sigs) == 2
        assert sigs[0]["signature"] == "warning:mem:leak"
        assert sigs[0]["count"] == 2

    def test_top_signatures_capped_at_5(self):
        findings = [
            _finding(severity="warning", title=f"sig-{i}")
            for i in range(10)
        ]
        reports = {"a": _report(vertical="v", findings=findings)}
        result = aggregate(reports)
        assert len(result["verticals"]["v"]["top_failure_signatures"]) == 5

    def test_top_dependencies(self):
        findings = [
            _finding(severity="warning", deps=["pandas", "numpy"]),
            _finding(severity="critical", deps=["pandas"]),
        ]
        reports = {"a": _report(vertical="v", findings=findings, deps=["pandas", "numpy", "flask"])}
        result = aggregate(reports)

        deps = result["verticals"]["v"]["top_dependencies"]
        assert deps[0]["dependency"] == "pandas"
        assert deps[0]["failure_count"] == 2

    def test_dependency_failure_rate(self):
        findings = [_finding(severity="warning", deps=["pandas"])]
        reports = {
            "a": _report(vertical="v", findings=findings, deps=["pandas", "numpy", "flask"]),
        }
        result = aggregate(reports)

        v = result["verticals"]["v"]
        assert v["total_dependencies_seen"] == 3
        assert v["dependencies_with_failures"] == 1
        assert abs(v["dependency_failure_rate"] - 0.333) < 0.01

    def test_cross_tabulation(self):
        reports = {
            "a": _report(vertical="web_app", architectural_pattern="fastapi"),
            "b": _report(vertical="web_app", architectural_pattern="flask"),
            "c": _report(vertical="web_app", architectural_pattern="fastapi"),
            "d": _report(vertical="ml_model", architectural_pattern="fastapi"),
        }
        result = aggregate(reports)

        cross = result["cross_tabulation"]
        assert cross["web_app"]["fastapi"] == 2
        assert cross["web_app"]["flask"] == 1
        assert cross["ml_model"]["fastapi"] == 1
        assert "flask" not in cross["ml_model"]

    def test_old_format_defaults_to_unclassified(self):
        """Reports without vertical/architectural_pattern use 'unclassified'."""
        old_report = {
            "project": {"name": "old", "dependencies": []},
            "findings": [_finding(severity="warning")],
        }
        reports = {"a": old_report}
        result = aggregate(reports)

        assert "unclassified" in result["verticals"]
        assert result["verticals"]["unclassified"]["repo_count"] == 1

    def test_empty_reports(self):
        result = aggregate({})
        assert result["total_repos"] == 0
        assert result["verticals"] == {}
        assert result["architectural_patterns"] == {}
        assert result["cross_tabulation"] == {}


# ── Print Summary (smoke test) ──


class TestPrintSummary:
    def test_does_not_crash(self, capsys):
        reports = {
            "a": _report(vertical="web_app", findings=[_finding(severity="critical", deps=["flask"])], deps=["flask"]),
            "b": _report(vertical="ml_model"),
        }
        result = aggregate(reports)
        print_summary(result)

        captured = capsys.readouterr()
        assert "CORPUS AGGREGATE" in captured.out
        assert "web_app" in captured.out
        assert "ml_model" in captured.out

    def test_empty_corpus(self, capsys):
        result = aggregate({})
        print_summary(result)
        captured = capsys.readouterr()
        assert "0 repos" in captured.out


# ── CLI ──


class TestCLI:
    def test_main_writes_output(self, tmp_path):
        results = tmp_path / "results"
        _write_report(results, "user__app", _report())
        output = tmp_path / "out.json"

        main(["--dirs", str(results), "--output", str(output)])

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["total_repos"] == 1
        assert "verticals" in data

    def test_main_default_output(self, tmp_path, monkeypatch):
        results = tmp_path / "results"
        _write_report(results, "user__app", _report())
        monkeypatch.chdir(tmp_path)

        main(["--dirs", str(results)])

        default_output = tmp_path / "corpus_aggregate.json"
        assert default_output.exists()

    def test_main_no_reports(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--dirs", str(tmp_path / "empty")])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No reports found" in captured.out

    def test_main_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "results"
        dir2 = tmp_path / "corpus_results"
        _write_report(dir1, "user__app1", _report(vertical="v1"))
        _write_report(dir2, "user__app2", _report(vertical="v2"))
        output = tmp_path / "out.json"

        main(["--dirs", f"{dir1},{dir2}", "--output", str(output)])

        data = json.loads(output.read_text())
        assert data["total_repos"] == 2
