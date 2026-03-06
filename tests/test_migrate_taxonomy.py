"""Tests for the L5 corpus migration script."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from migrate_taxonomy import (
    _extract_error_type,
    _extract_scenario_name,
    _extract_breaking_point,
    _extract_metric_from_details,
    migrate_report,
    run_migration,
)


class TestExtractHelpers:
    """Test helper extraction functions."""

    def test_extract_error_type_memory(self):
        finding = {"details": "MemoryError: out of memory", "title": "test"}
        assert _extract_error_type(finding) == "MemoryError"

    def test_extract_error_type_timeout(self):
        finding = {"details": "TimeoutError: timed out", "title": "test"}
        assert _extract_error_type(finding) == "TimeoutError"

    def test_extract_error_type_from_title(self):
        finding = {"details": "", "title": "ImportError in module"}
        assert _extract_error_type(finding) == "ImportError"

    def test_extract_error_type_none(self):
        finding = {"details": "something happened", "title": "test"}
        assert _extract_error_type(finding) == ""

    def test_extract_scenario_name_with_colon(self):
        assert _extract_scenario_name("Scenario failed: flask_test") == "flask_test"

    def test_extract_scenario_name_without_colon(self):
        assert _extract_scenario_name("just a title") == "just a title"

    def test_extract_scenario_name_resource_limit(self):
        assert _extract_scenario_name("Resource limit hit: memory_test") == "memory_test"

    def test_extract_breaking_point(self):
        finding = {"details": "failed at concurrent_50 step"}
        assert _extract_breaking_point(finding) == "concurrent_50"

    def test_extract_breaking_point_none(self):
        finding = {"details": "no breaking point info"}
        assert _extract_breaking_point(finding) == ""

    def test_extract_metric_memory(self):
        metric, start, end, mult = _extract_metric_from_details("peak memory 135.7 MB")
        assert metric == "memory_peak_mb"
        assert end == 135.7

    def test_extract_metric_time(self):
        metric, start, end, mult = _extract_metric_from_details("took 250.5 ms")
        assert metric == "execution_time_ms"
        assert end == 250.5

    def test_extract_metric_errors(self):
        metric, start, end, mult = _extract_metric_from_details("19 errors recorded")
        assert metric == "error_count"
        assert end == 19.0

    def test_extract_metric_none(self):
        metric, start, end, mult = _extract_metric_from_details("no metrics here")
        assert metric == ""


class TestMigrateReport:
    """Test migrate_report function."""

    def _make_report(self, findings=None, language="python", deps=None):
        return {
            "project": {
                "name": "test_project",
                "language": language,
                "files_analyzed": 10,
                "total_lines": 500,
                "dependencies": [
                    {"name": d, "installed_version": "1.0.0"}
                    for d in (deps or ["flask"])
                ],
            },
            "constraints": None,
            "summary": "test summary",
            "statistics": {
                "scenarios_run": 5,
                "recognized_dependencies": 1,
                "unrecognized_dependencies": 0,
            },
            "findings": findings or [],
            "incomplete_tests": [],
            "degradation_curves": [],
            "version_discrepancies": [],
            "unrecognized_dependencies": [],
        }

    def test_empty_findings(self):
        report = self._make_report(findings=[])
        entries = migrate_report(report, "user__repo")
        assert entries == []

    def test_single_finding(self):
        report = self._make_report(findings=[{
            "title": "Scenario failed: flask_memory_test",
            "severity": "critical",
            "category": "memory_profiling",
            "description": "Memory leak detected",
            "details": "peak memory 200 MB",
            "affected_dependencies": ["flask"],
            "load_level": 100,
            "group_count": 1,
        }])
        entries = migrate_report(report, "user__repo")
        assert len(entries) == 1
        e = entries[0]
        assert e["failure_domain"] == "resource_exhaustion"
        assert e["scenario_name"] == "flask_memory_test"
        assert e["severity_raw"] == "critical"
        assert e["language"] == "python"
        assert e["source"] == "corpus_mining"
        assert e["vertical"] == "web_app"

    def test_skips_dependency_count_findings(self):
        report = self._make_report(findings=[
            {
                "title": "2 missing dependencies",
                "severity": "info",
                "category": "",
                "description": "deps missing",
                "details": "",
                "affected_dependencies": [],
                "load_level": None,
                "group_count": 1,
            },
            {
                "title": "Scenario failed: flask_test",
                "severity": "critical",
                "category": "memory_profiling",
                "description": "test",
                "details": "",
                "affected_dependencies": ["flask"],
                "load_level": None,
                "group_count": 1,
            },
        ])
        entries = migrate_report(report, "user__repo")
        assert len(entries) == 1
        assert entries[0]["scenario_name"] == "flask_test"

    def test_degradation_curves_migrated(self):
        report = self._make_report()
        report["degradation_curves"] = [{
            "scenario_name": "memory_scaling",
            "metric": "memory_peak_mb",
            "steps": [
                {"label": "step_1", "value": 10.0},
                {"label": "step_100", "value": 200.0},
            ],
            "breaking_point": "step_100",
            "description": "Memory grew 20x",
            "group_count": 1,
        }]
        entries = migrate_report(report, "user__repo")
        assert len(entries) == 1
        assert entries[0]["metric_name"] == "memory_peak_mb"
        assert entries[0]["multiplier"] == 20.0

    def test_entry_has_all_mandatory_fields(self):
        report = self._make_report(findings=[{
            "title": "Scenario failed: test",
            "severity": "critical",
            "category": "concurrent_execution",
            "description": "test",
            "details": "",
            "affected_dependencies": [],
            "load_level": None,
            "group_count": 1,
        }])
        entries = migrate_report(report, "user__repo")
        e = entries[0]
        mandatory = [
            "entry_id", "source", "source_batch", "mycode_version",
            "timestamp", "language", "failure_domain", "failure_pattern",
            "scenario_name", "scenario_category", "operational_trigger",
            "affected_dependencies", "severity_raw", "load_level_at_failure",
            "breaking_point", "metric_name", "metric_start_value",
            "metric_end_value", "multiplier", "codebase_origin",
            "vertical", "architectural_pattern",
        ]
        for field in mandatory:
            assert field in e, f"Missing mandatory field: {field}"

    def test_entry_has_corpus_mining_fields(self):
        report = self._make_report(findings=[{
            "title": "Scenario failed: test",
            "severity": "critical",
            "category": "",
            "description": "",
            "details": "",
            "affected_dependencies": [],
            "load_level": None,
            "group_count": 1,
        }])
        entries = migrate_report(report, "user__repo")
        e = entries[0]
        corpus_fields = [
            "repo_url", "repo_loc", "repo_file_count",
            "dependency_count", "profiled_dependency_count",
            "unrecognised_dependency_count",
        ]
        for field in corpus_fields:
            assert field in e, f"Missing corpus field: {field}"


class TestRunMigration:
    """Test the full migration pipeline."""

    def test_migration_with_temp_dir(self, tmp_path):
        # Create a minimal corpus
        repo_dir = tmp_path / "user__repo"
        repo_dir.mkdir()

        report = {
            "project": {
                "name": "repo",
                "language": "python",
                "files_analyzed": 5,
                "total_lines": 200,
                "dependencies": [{"name": "flask", "installed_version": "2.0.0"}],
            },
            "constraints": None,
            "statistics": {"scenarios_run": 1, "recognized_dependencies": 1, "unrecognized_dependencies": 0},
            "findings": [{
                "title": "Scenario failed: flask_memory_test",
                "severity": "critical",
                "category": "memory_profiling",
                "description": "Memory leak",
                "details": "peak memory 100 MB",
                "affected_dependencies": ["flask"],
                "load_level": None,
                "group_count": 1,
            }],
            "incomplete_tests": [],
            "degradation_curves": [],
            "version_discrepancies": [],
            "unrecognized_dependencies": [],
        }

        (repo_dir / "mycode-report.json").write_text(
            json.dumps(report), encoding="utf-8",
        )

        output_file = tmp_path / "output.json"
        stats = run_migration(str(tmp_path), str(output_file))

        assert stats["reports_processed"] == 1
        assert stats["total_entries"] == 1
        assert stats["classified"] == 1
        assert stats["unclassified"] == 0

        # Verify output file
        with open(output_file) as f:
            output = json.load(f)
        assert len(output["entries"]) == 1
        assert output["entries"][0]["failure_domain"] == "resource_exhaustion"

    def test_migration_skips_missing_reports(self, tmp_path):
        # Create repo dir without report
        (tmp_path / "user__repo").mkdir()
        output_file = tmp_path / "output.json"
        stats = run_migration(str(tmp_path), str(output_file))
        assert stats["reports_processed"] == 0
        assert stats["reports_skipped"] == 1
