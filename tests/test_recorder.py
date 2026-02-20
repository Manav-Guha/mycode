"""Tests for Interaction Recorder (E3).

Tests cover:
  - Consent enforcement (opt-in required, ConsentError on violations)
  - Conversation recording (intent + raw responses)
  - Dependency recording (combinations, recognized, unrecognized, versions)
  - Scenario config recording (anonymized, no file paths)
  - Execution result recording (aggregated, no stdout/stderr)
  - Report recording (finding counts, failure patterns)
  - Anonymization (no PII, no file paths, no stderr/stdout)
  - Persistence (save/load JSONL, date partitioning)
  - Aggregation helpers (unrecognized dep counts, failure patterns)
  - Edge cases (empty data, missing fields, corrupt files)
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.ingester import DependencyInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.recorder import (
    ConsentError,
    InteractionRecorder,
    RecorderError,
    SessionRecord,
    _anonymize_dependency,
    _anonymize_scenario,
    _anonymize_scenario_result,
    _count_severities,
    _extract_failure_patterns,
    _generate_session_id,
)
from mycode.report import DegradationPoint, DiagnosticReport, Finding
from mycode.scenario import ScenarioGeneratorResult, StressTestScenario


# ── Fixtures ──


@pytest.fixture
def ingestion() -> IngestionResult:
    return IngestionResult(
        project_path="/home/user/secret-project",
        files_analyzed=5,
        total_lines=500,
        dependencies=[
            DependencyInfo(
                name="flask",
                installed_version="3.1.0",
                latest_version="3.1.0",
                source_file="/home/user/secret-project/requirements.txt",
            ),
            DependencyInfo(
                name="old-lib",
                installed_version="1.0.0",
                latest_version="2.0.0",
                is_outdated=True,
            ),
            DependencyInfo(
                name="ghost-dep",
                is_missing=True,
            ),
        ],
    )


@pytest.fixture
def profile_matches() -> list[ProfileMatch]:
    return [
        ProfileMatch(
            dependency_name="flask",
            profile=MagicMock(),
            installed_version="3.1.0",
            version_match=True,
        ),
        ProfileMatch(
            dependency_name="old-lib",
            profile=MagicMock(),
            installed_version="1.0.0",
            version_match=False,
            version_notes="v1.0 has known memory leak",
        ),
        ProfileMatch(dependency_name="unknown-tool", profile=None),
        ProfileMatch(dependency_name="custom-lib", profile=None),
    ]


@pytest.fixture
def scenario_result() -> ScenarioGeneratorResult:
    return ScenarioGeneratorResult(
        scenarios=[
            StressTestScenario(
                name="flask_data_volume",
                category="data_volume_scaling",
                description="Test with large data",
                target_dependencies=["flask"],
                test_config={
                    "target_files": ["/secret/path/app.py"],
                    "parameters": {"data_sizes": [100, 1000]},
                },
                priority="high",
                source="offline",
            ),
            StressTestScenario(
                name="concurrent_queries",
                category="concurrent_execution",
                description="Test concurrency",
                target_dependencies=["flask", "old-lib"],
                priority="medium",
                source="llm",
            ),
        ],
    )


@pytest.fixture
def execution() -> ExecutionEngineResult:
    return ExecutionEngineResult(
        scenario_results=[
            ScenarioResult(
                scenario_name="flask_data_volume",
                scenario_category="data_volume_scaling",
                status="completed",
                steps=[
                    StepResult(
                        step_name="size_100",
                        execution_time_ms=50.0,
                        memory_peak_mb=10.0,
                        stdout_snippet="secret output here",
                        stderr_snippet="/home/user/path/error",
                    ),
                    StepResult(
                        step_name="size_1000",
                        execution_time_ms=500.0,
                        memory_peak_mb=100.0,
                        error_count=2,
                        errors=[
                            {"type": "MemoryError", "message": "/home/user/app.py line 42"},
                            {"type": "TimeoutError", "message": "too slow"},
                        ],
                    ),
                ],
                total_errors=2,
                total_execution_time_ms=550.0,
                peak_memory_mb=100.0,
            ),
            ScenarioResult(
                scenario_name="concurrent_queries",
                scenario_category="concurrent_execution",
                status="failed",
                summary="Connection pool exhausted at /home/user/db.py",
                total_errors=5,
                failure_indicators_triggered=["connection_pool_exhausted"],
                resource_cap_hit=True,
            ),
        ],
        total_execution_time_ms=600.0,
        scenarios_completed=1,
        scenarios_failed=1,
    )


@pytest.fixture
def report() -> DiagnosticReport:
    return DiagnosticReport(
        summary="Found issues under load",
        scenarios_run=2,
        scenarios_passed=0,
        scenarios_failed=2,
        total_errors=7,
        findings=[
            Finding(
                title="Memory leak",
                severity="critical",
                category="data_volume_scaling",
                affected_dependencies=["flask"],
            ),
            Finding(
                title="Slow queries",
                severity="warning",
                category="concurrent_execution",
                affected_dependencies=["old-lib"],
            ),
            Finding(
                title="Outdated dep",
                severity="info",
                affected_dependencies=["old-lib"],
            ),
        ],
        degradation_points=[
            DegradationPoint(
                scenario_name="flask_data_volume",
                metric="execution_time_ms",
                breaking_point="size_1000",
            ),
        ],
        version_flags=["old-lib: installed 1.0.0, latest 2.0.0"],
        unrecognized_deps=["unknown-tool", "custom-lib"],
    )


# ── Consent Tests ──


class TestConsent:
    """Tests for consent enforcement."""

    def test_no_consent_raises_on_record(self):
        recorder = InteractionRecorder(consent=False)
        with pytest.raises(ConsentError, match="explicit user consent"):
            recorder.record_conversation("test")

    def test_no_consent_raises_on_all_methods(self, ingestion, profile_matches):
        recorder = InteractionRecorder(consent=False)
        with pytest.raises(ConsentError):
            recorder.record_dependencies(ingestion, profile_matches)

    def test_no_consent_save_returns_none(self, tmp_path):
        recorder = InteractionRecorder(consent=False, data_dir=tmp_path)
        assert recorder.save() is None

    def test_consent_allows_recording(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_conversation("test intent")  # Should not raise

    def test_consent_property(self):
        assert InteractionRecorder(consent=True).consent is True
        assert InteractionRecorder(consent=False).consent is False


# ── Conversation Recording Tests ──


class TestRecordConversation:
    """Tests for conversation recording."""

    def test_records_intent_and_responses(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_conversation(
            intent_summary="Budget tracking app",
            raw_responses={"turn_1": "It tracks expenses", "turn_2": "Speed"},
        )

        conv = recorder.record.conversation
        assert conv["intent_summary"] == "Budget tracking app"
        assert conv["raw_responses"]["turn_1"] == "It tracks expenses"
        assert conv["raw_responses"]["turn_2"] == "Speed"

    def test_records_empty_responses(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_conversation("intent only")

        assert recorder.record.conversation["raw_responses"] == {}

    def test_no_pii_in_conversation(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_conversation(
            "intent",
            {"turn_1": "my app", "turn_2": "speed"},
        )
        # Conversation should not contain project_path or file paths
        conv_str = json.dumps(recorder.record.conversation)
        assert "/home/" not in conv_str
        assert "/Users/" not in conv_str


# ── Dependency Recording Tests ──


class TestRecordDependencies:
    """Tests for dependency combination recording."""

    def test_records_dependency_combinations(
        self, ingestion, profile_matches,
    ):
        recorder = InteractionRecorder(consent=True)
        recorder.record_dependencies(ingestion, profile_matches, "python")

        deps = recorder.record.dependency_combinations
        assert len(deps) == 3
        assert deps[0]["name"] == "flask"
        assert deps[0]["installed_version"] == "3.1.0"

    def test_records_recognized_vs_unrecognized(
        self, ingestion, profile_matches,
    ):
        recorder = InteractionRecorder(consent=True)
        recorder.record_dependencies(ingestion, profile_matches)

        assert "flask" in recorder.record.recognized_deps
        assert "old-lib" in recorder.record.recognized_deps
        assert "unknown-tool" in recorder.record.unrecognized_deps
        assert "custom-lib" in recorder.record.unrecognized_deps

    def test_records_version_discrepancies(
        self, ingestion, profile_matches,
    ):
        recorder = InteractionRecorder(consent=True)
        recorder.record_dependencies(ingestion, profile_matches)

        vd = recorder.record.version_discrepancies
        outdated = [v for v in vd if v["status"] == "outdated"]
        missing = [v for v in vd if v["status"] == "missing"]
        mismatch = [v for v in vd if v["status"] == "version_mismatch"]

        assert len(outdated) == 1
        assert outdated[0]["name"] == "old-lib"
        assert len(missing) == 1
        assert missing[0]["name"] == "ghost-dep"
        assert len(mismatch) == 1
        assert "memory leak" in mismatch[0]["notes"]

    def test_no_file_paths_in_deps(self, ingestion, profile_matches):
        recorder = InteractionRecorder(consent=True)
        recorder.record_dependencies(ingestion, profile_matches)

        deps_str = json.dumps(recorder.record.dependency_combinations)
        assert "/home/user" not in deps_str
        assert "requirements.txt" not in deps_str

    def test_records_language(self, ingestion, profile_matches):
        recorder = InteractionRecorder(consent=True)
        recorder.record_dependencies(ingestion, profile_matches, "javascript")
        assert recorder.record.language == "javascript"


# ── Scenario Recording Tests ──


class TestRecordScenarios:
    """Tests for scenario config recording."""

    def test_records_scenario_configs(self, scenario_result):
        recorder = InteractionRecorder(consent=True)
        recorder.record_scenarios(scenario_result)

        configs = recorder.record.scenario_configs
        assert len(configs) == 2
        assert configs[0]["name"] == "flask_data_volume"
        assert configs[0]["category"] == "data_volume_scaling"
        assert configs[0]["priority"] == "high"
        assert configs[0]["source"] == "offline"

    def test_no_file_paths_in_scenarios(self, scenario_result):
        recorder = InteractionRecorder(consent=True)
        recorder.record_scenarios(scenario_result)

        configs_str = json.dumps(recorder.record.scenario_configs)
        assert "/secret/path" not in configs_str
        assert "target_files" not in configs_str

    def test_target_dependencies_preserved(self, scenario_result):
        recorder = InteractionRecorder(consent=True)
        recorder.record_scenarios(scenario_result)

        assert recorder.record.scenario_configs[0]["target_dependencies"] == ["flask"]


# ── Execution Recording Tests ──


class TestRecordExecution:
    """Tests for execution result recording."""

    def test_records_execution_summary(self, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)

        summary = recorder.record.execution_summary
        assert summary["total_time_ms"] == 600.0
        assert summary["scenarios_completed"] == 1
        assert summary["scenarios_failed"] == 1

    def test_records_step_results(self, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)

        results = recorder.record.execution_summary["scenario_results"]
        assert len(results) == 2

        # First scenario should have steps
        steps = results[0]["steps"]
        assert len(steps) == 2
        assert steps[0]["step_name"] == "size_100"
        assert steps[1]["error_count"] == 2

    def test_no_stdout_stderr_in_execution(self, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)

        exec_str = json.dumps(recorder.record.execution_summary)
        assert "secret output" not in exec_str
        assert "stdout_snippet" not in exec_str
        assert "stderr_snippet" not in exec_str

    def test_no_error_messages_in_execution(self, execution):
        """Error messages may contain file paths — only types recorded."""
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)

        exec_str = json.dumps(recorder.record.execution_summary)
        assert "/home/user" not in exec_str
        assert "line 42" not in exec_str
        # But error types should be present
        assert "MemoryError" in exec_str
        assert "TimeoutError" in exec_str

    def test_failure_indicators_recorded(self, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)

        results = recorder.record.execution_summary["scenario_results"]
        failed = [r for r in results if r["status"] == "failed"]
        assert len(failed) == 1
        assert "connection_pool_exhausted" in failed[0]["failure_indicators_triggered"]


# ── Report Recording Tests ──


class TestRecordReport:
    """Tests for report data recording."""

    def test_records_report_summary(self, report):
        recorder = InteractionRecorder(consent=True)
        recorder.record_report(report)

        summary = recorder.record.report_summary
        assert summary["scenarios_run"] == 2
        assert summary["scenarios_failed"] == 2
        assert summary["total_errors"] == 7
        assert summary["finding_counts"]["critical"] == 1
        assert summary["finding_counts"]["warning"] == 1
        assert summary["finding_counts"]["info"] == 1
        assert summary["degradation_count"] == 1
        assert summary["unrecognized_dep_count"] == 2

    def test_extracts_failure_patterns(self, report, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)
        recorder.record_report(report)

        patterns = recorder.record.failure_patterns
        assert len(patterns) >= 1

        # Should have finding-based patterns
        finding_patterns = [p for p in patterns if p["type"] == "finding"]
        assert len(finding_patterns) >= 2  # critical + warning

        # Should have degradation patterns
        deg_patterns = [p for p in patterns if p["type"] == "degradation"]
        assert len(deg_patterns) == 1
        assert deg_patterns[0]["breaking_point"] == "size_1000"

    def test_failure_pattern_has_category(self, report, execution):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(execution)
        recorder.record_report(report)

        finding_patterns = [
            p for p in recorder.record.failure_patterns
            if p["type"] == "finding"
        ]
        categories = {p["category"] for p in finding_patterns}
        assert "data_volume_scaling" in categories

    def test_no_narrative_in_report(self, report):
        """LLM-generated narrative should not be recorded."""
        recorder = InteractionRecorder(consent=True)
        recorder.record_report(report)

        report_str = json.dumps(recorder.record.report_summary)
        assert "Found issues under load" not in report_str


# ── Persistence Tests ──


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_creates_jsonl_file(self, tmp_path):
        recorder = InteractionRecorder(consent=True, data_dir=tmp_path)
        recorder.record_conversation("test intent")
        filepath = recorder.save()

        assert filepath is not None
        assert filepath.exists()
        assert filepath.suffix == ".jsonl"
        assert "sessions_" in filepath.name

    def test_save_appends_to_existing(self, tmp_path):
        # First record
        r1 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r1.record_conversation("first")
        r1.save()

        # Second record
        r2 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r2.record_conversation("second")
        r2.save()

        # Both should be in the file
        records = InteractionRecorder.load_records(tmp_path)
        assert len(records) == 2
        intents = {r.conversation.get("intent_summary") for r in records}
        assert "first" in intents
        assert "second" in intents

    def test_load_records_empty_dir(self, tmp_path):
        records = InteractionRecorder.load_records(tmp_path)
        assert records == []

    def test_load_records_nonexistent_dir(self, tmp_path):
        records = InteractionRecorder.load_records(tmp_path / "nope")
        assert records == []

    def test_roundtrip_full_record(
        self, tmp_path, ingestion, profile_matches,
        scenario_result, execution, report,
    ):
        recorder = InteractionRecorder(consent=True, data_dir=tmp_path)
        recorder.record_conversation("intent", {"turn_1": "app", "turn_2": "speed"})
        recorder.record_dependencies(ingestion, profile_matches, "python")
        recorder.record_scenarios(scenario_result)
        recorder.record_execution(execution)
        recorder.record_report(report)
        recorder.save()

        records = InteractionRecorder.load_records(tmp_path)
        assert len(records) == 1
        rec = records[0]

        assert rec.session_id == recorder.session_id
        assert rec.language == "python"
        assert rec.consent_given is True
        assert rec.conversation["intent_summary"] == "intent"
        assert len(rec.dependency_combinations) == 3
        assert len(rec.recognized_deps) == 2
        assert len(rec.unrecognized_deps) == 2
        assert len(rec.scenario_configs) == 2
        assert rec.execution_summary["scenarios_completed"] == 1
        assert len(rec.failure_patterns) >= 1
        assert rec.report_summary["total_errors"] == 7

    def test_corrupt_jsonl_skipped(self, tmp_path):
        filepath = tmp_path / "sessions_2026-02-21.jsonl"
        filepath.write_text(
            '{"session_id":"abc","consent_given":true}\n'
            'THIS IS NOT JSON\n'
            '{"session_id":"def","consent_given":true}\n'
        )

        records = InteractionRecorder.load_records(tmp_path)
        # Should load 2 valid records, skip the corrupt line
        assert len(records) == 2

    def test_save_creates_data_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        recorder = InteractionRecorder(consent=True, data_dir=nested)
        recorder.record_conversation("test")
        filepath = recorder.save()

        assert filepath is not None
        assert nested.exists()


# ── Aggregation Helper Tests ──


class TestAggregationHelpers:
    """Tests for unrecognized dep counts and failure pattern aggregation."""

    def test_load_unrecognized_deps(self, tmp_path):
        # Record two sessions with overlapping unrecognized deps
        r1 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r1._record.unrecognized_deps = ["alpha", "beta"]
        r1.save()

        r2 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r2._record.unrecognized_deps = ["beta", "gamma"]
        r2.save()

        counts = InteractionRecorder.load_unrecognized_deps(tmp_path)
        assert counts["alpha"] == 1
        assert counts["beta"] == 2
        assert counts["gamma"] == 1

    def test_load_failure_patterns(self, tmp_path):
        r1 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r1._record.failure_patterns = [
            {"type": "finding", "severity": "critical"},
        ]
        r1.save()

        r2 = InteractionRecorder(consent=True, data_dir=tmp_path)
        r2._record.failure_patterns = [
            {"type": "degradation", "metric": "memory"},
            {"type": "finding", "severity": "warning"},
        ]
        r2.save()

        patterns = InteractionRecorder.load_failure_patterns(tmp_path)
        assert len(patterns) == 3

    def test_empty_aggregation(self, tmp_path):
        assert InteractionRecorder.load_unrecognized_deps(tmp_path) == {}
        assert InteractionRecorder.load_failure_patterns(tmp_path) == []


# ── Anonymization Tests ──


class TestAnonymization:
    """Tests for anonymization helpers."""

    def test_session_id_is_unique(self):
        ids = {_generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_session_id_length(self):
        sid = _generate_session_id()
        assert len(sid) == 16
        assert sid.isalnum()

    def test_anonymize_dependency_strips_source(self):
        dep = DependencyInfo(
            name="flask",
            installed_version="3.1.0",
            source_file="/home/user/requirements.txt",
        )
        anon = _anonymize_dependency(dep)
        assert "source_file" not in anon
        assert "/home/user" not in json.dumps(anon)
        assert anon["name"] == "flask"
        assert anon["installed_version"] == "3.1.0"

    def test_anonymize_scenario_strips_files(self):
        scenario = StressTestScenario(
            name="test",
            category="data_volume_scaling",
            description="Test desc",
            test_config={"target_files": ["/secret/app.py"], "parameters": {"x": 1}},
        )
        anon = _anonymize_scenario(scenario)
        assert "target_files" not in anon
        assert "/secret" not in json.dumps(anon)
        assert anon["name"] == "test"
        assert anon["category"] == "data_volume_scaling"

    def test_anonymize_scenario_result_strips_snippets(self):
        sr = ScenarioResult(
            scenario_name="test",
            scenario_category="data_volume_scaling",
            status="completed",
            steps=[
                StepResult(
                    step_name="step1",
                    execution_time_ms=100.0,
                    stdout_snippet="output with /home/user/path",
                    stderr_snippet="error at /secret/location",
                    errors=[{"type": "ValueError", "message": "/home/user/code.py:42"}],
                ),
            ],
        )
        anon = _anonymize_scenario_result(sr)
        anon_str = json.dumps(anon)
        assert "stdout_snippet" not in anon_str
        assert "stderr_snippet" not in anon_str
        assert "/home/user" not in anon_str
        assert "/secret" not in anon_str
        assert "ValueError" in anon_str  # Error types kept

    def test_count_severities(self):
        findings = [
            Finding(title="a", severity="critical"),
            Finding(title="b", severity="critical"),
            Finding(title="c", severity="warning"),
            Finding(title="d", severity="info"),
        ]
        counts = _count_severities(findings)
        assert counts == {"critical": 2, "warning": 1, "info": 1}


# ── SessionRecord Tests ──


class TestSessionRecord:
    """Tests for SessionRecord data class."""

    def test_default_values(self):
        rec = SessionRecord()
        assert rec.session_id == ""
        assert rec.consent_given is False
        assert rec.dependency_combinations == []

    def test_session_id_generated(self):
        recorder = InteractionRecorder(consent=True)
        assert len(recorder.session_id) == 16

    def test_timestamp_generated(self):
        recorder = InteractionRecorder(consent=True)
        assert recorder.record.timestamp
        assert "T" in recorder.record.timestamp


# ── Edge Cases ──


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_ingestion(self):
        recorder = InteractionRecorder(consent=True)
        ingestion = IngestionResult(project_path="/tmp/x")
        recorder.record_dependencies(ingestion, [])

        assert recorder.record.dependency_combinations == []
        assert recorder.record.recognized_deps == []
        assert recorder.record.unrecognized_deps == []

    def test_empty_execution(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_execution(ExecutionEngineResult())

        summary = recorder.record.execution_summary
        assert summary["scenarios_completed"] == 0
        assert summary["scenario_results"] == []

    def test_empty_report(self):
        recorder = InteractionRecorder(consent=True)
        recorder.record_report(DiagnosticReport())

        summary = recorder.record.report_summary
        assert summary["scenarios_run"] == 0
        assert summary["finding_counts"] == {"critical": 0, "warning": 0, "info": 0}

    def test_save_without_recording_anything(self, tmp_path):
        recorder = InteractionRecorder(consent=True, data_dir=tmp_path)
        filepath = recorder.save()

        assert filepath is not None
        records = InteractionRecorder.load_records(tmp_path)
        assert len(records) == 1
        assert records[0].consent_given is True

    def test_multiple_saves_same_recorder(self, tmp_path):
        recorder = InteractionRecorder(consent=True, data_dir=tmp_path)
        recorder.record_conversation("first")
        recorder.save()
        recorder.record_conversation("updated")
        recorder.save()

        records = InteractionRecorder.load_records(tmp_path)
        assert len(records) == 2
