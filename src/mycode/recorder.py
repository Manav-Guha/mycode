"""Interaction Recorder (E3) — Anonymized session data recording.

Records anonymized interaction data with explicit user consent.  Feeds
component library improvement, logs unrecognized dependencies for future
profile development, and logs failure patterns for scenario generator
improvement.

Per spec:
  - Explicit consent required (opt-in, not opt-out)
  - No personally identifiable information
  - Stores: conversation, test configuration, results, dependency combinations
  - Local JSON-lines storage (one record per session, append-only)

Pure Python. No LLM dependency.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from mycode.engine import ExecutionEngineResult, ScenarioResult
from mycode.ingester import DependencyInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.report import DiagnosticReport, Finding
from mycode.scenario import ScenarioGeneratorResult, StressTestScenario

logger = logging.getLogger(__name__)

# ── Constants ──

_DEFAULT_DATA_DIR = Path.home() / ".mycode" / "recordings"

# Fields stripped from recorded data to avoid PII
_PII_FIELDS = frozenset({
    "project_path", "file_path", "workspace_dir", "venv_dir",
    "project_copy_dir", "venv_python", "python_executable",
    "stdout_snippet", "stderr_snippet",
})


# ── Exceptions ──


class RecorderError(Exception):
    """Base exception for interaction recorder errors."""


class ConsentError(RecorderError):
    """Attempted to record without consent."""


# ── Data Classes ──


@dataclass
class SessionRecord:
    """A single anonymized session recording.

    Attributes:
        session_id: Random hash identifying this session (not traceable).
        timestamp: ISO 8601 timestamp of when the session was recorded.
        language: Project language ("python" or "javascript").
        consent_given: True if the user explicitly opted in.
        conversation: Anonymized conversational exchange data.
        dependency_combinations: Dependencies found in the project.
        recognized_deps: Dependencies matched by the component library.
        unrecognized_deps: Dependencies without profiles.
        scenario_configs: Stress test configurations used.
        execution_summary: Aggregated execution results.
        failure_patterns: Patterns extracted from failures.
        report_summary: High-level report findings.
        version_discrepancies: Flagged version issues.
    """

    session_id: str = ""
    timestamp: str = ""
    language: str = ""
    consent_given: bool = False
    conversation: dict = field(default_factory=dict)
    dependency_combinations: list[dict] = field(default_factory=list)
    recognized_deps: list[str] = field(default_factory=list)
    unrecognized_deps: list[str] = field(default_factory=list)
    scenario_configs: list[dict] = field(default_factory=list)
    execution_summary: dict = field(default_factory=dict)
    failure_patterns: list[dict] = field(default_factory=list)
    report_summary: dict = field(default_factory=dict)
    version_discrepancies: list[dict] = field(default_factory=list)


# ── Interaction Recorder ──


class InteractionRecorder:
    """Records anonymized session data for component library improvement.

    Usage::

        recorder = InteractionRecorder(consent=True)
        recorder.record_conversation(intent_summary, raw_responses)
        recorder.record_dependencies(ingestion, profile_matches)
        recorder.record_scenarios(scenario_result)
        recorder.record_execution(execution_result)
        recorder.record_report(diagnostic_report)
        recorder.save()

    Consent check: every mutating method raises ``ConsentError`` if consent
    was not given at construction time.  The recorder will never silently
    record data.
    """

    def __init__(
        self,
        consent: bool = False,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._consent = consent
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._record = SessionRecord(
            session_id=_generate_session_id(),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            consent_given=consent,
        )

    @property
    def consent(self) -> bool:
        return self._consent

    @property
    def session_id(self) -> str:
        return self._record.session_id

    @property
    def record(self) -> SessionRecord:
        return self._record

    def _require_consent(self) -> None:
        """Raise ConsentError if consent was not given."""
        if not self._consent:
            raise ConsentError(
                "Cannot record without explicit user consent. "
                "Pass consent=True to enable recording."
            )

    # ── Recording Methods ──

    def record_conversation(
        self,
        intent_summary: str = "",
        raw_responses: Optional[dict[str, str]] = None,
    ) -> None:
        """Record the anonymized conversational exchange.

        Stores the operational intent summary and the user's raw responses
        (turn 1 and turn 2).  No project paths or system info recorded.
        """
        self._require_consent()
        self._record.conversation = {
            "intent_summary": intent_summary,
            "raw_responses": dict(raw_responses) if raw_responses else {},
        }

    def record_dependencies(
        self,
        ingestion: IngestionResult,
        profile_matches: list[ProfileMatch],
        language: str = "",
    ) -> None:
        """Record dependency combinations and recognition status.

        Stores dependency name + version (no file paths).  Separates
        recognized from unrecognized for profile development.
        """
        self._require_consent()
        self._record.language = language

        # Dependency combinations (name + version only, no paths)
        self._record.dependency_combinations = [
            _anonymize_dependency(dep) for dep in ingestion.dependencies
        ]

        # Split recognized vs unrecognized
        self._record.recognized_deps = [
            m.dependency_name for m in profile_matches if m.profile is not None
        ]
        self._record.unrecognized_deps = [
            m.dependency_name for m in profile_matches if m.profile is None
        ]

        # Version discrepancies
        self._record.version_discrepancies = []
        for dep in ingestion.dependencies:
            if dep.is_outdated:
                self._record.version_discrepancies.append({
                    "name": dep.name,
                    "installed": dep.installed_version or "",
                    "latest": dep.latest_version or "",
                    "status": "outdated",
                })
            elif dep.is_missing:
                self._record.version_discrepancies.append({
                    "name": dep.name,
                    "status": "missing",
                })

        for match in profile_matches:
            if match.version_match is False and match.version_notes:
                self._record.version_discrepancies.append({
                    "name": match.dependency_name,
                    "installed": match.installed_version or "",
                    "notes": match.version_notes,
                    "status": "version_mismatch",
                })

    def record_scenarios(
        self,
        scenario_result: ScenarioGeneratorResult,
    ) -> None:
        """Record scenario configurations (anonymized).

        Stores scenario name, category, target dependencies, priority,
        and source.  No file paths or user code content.
        """
        self._require_consent()
        self._record.scenario_configs = [
            _anonymize_scenario(s) for s in scenario_result.scenarios
        ]

    def record_execution(
        self,
        execution: ExecutionEngineResult,
    ) -> None:
        """Record aggregated execution results.

        Stores per-scenario status, error counts, timing, and memory.
        No stdout/stderr snippets or file paths.
        """
        self._require_consent()
        self._record.execution_summary = {
            "total_time_ms": execution.total_execution_time_ms,
            "scenarios_completed": execution.scenarios_completed,
            "scenarios_failed": execution.scenarios_failed,
            "scenarios_skipped": execution.scenarios_skipped,
            "scenario_results": [
                _anonymize_scenario_result(sr)
                for sr in execution.scenario_results
            ],
        }

    def record_report(
        self,
        report: DiagnosticReport,
    ) -> None:
        """Record high-level report findings and failure patterns.

        Stores finding severities, categories, and failure patterns.
        No user code, file paths, or narrative text from the LLM.
        """
        self._require_consent()

        # Report summary
        self._record.report_summary = {
            "scenarios_run": report.scenarios_run,
            "scenarios_passed": report.scenarios_passed,
            "scenarios_failed": report.scenarios_failed,
            "total_errors": report.total_errors,
            "finding_counts": _count_severities(report.findings),
            "degradation_count": len(report.degradation_points),
            "version_flag_count": len(report.version_flags),
            "unrecognized_dep_count": len(report.unrecognized_deps),
        }

        # Failure patterns for scenario generator improvement
        self._record.failure_patterns = _extract_failure_patterns(
            report, self._record.execution_summary,
        )

    # ── Persistence ──

    def save(self) -> Optional[Path]:
        """Save the session record to disk.

        Returns the path to the saved file, or None if consent was not given.
        The record is appended as a single JSON line to a date-partitioned
        file.
        """
        if not self._consent:
            logger.debug("Recording skipped — no consent.")
            return None

        self._data_dir.mkdir(parents=True, exist_ok=True)
        date_str = self._record.timestamp[:10]  # YYYY-MM-DD
        filepath = self._data_dir / f"sessions_{date_str}.jsonl"

        record_dict = asdict(self._record)
        line = json.dumps(record_dict, separators=(",", ":")) + "\n"

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
            logger.info("Session recorded: %s", filepath)
            return filepath
        except OSError as exc:
            logger.warning("Failed to save recording: %s", exc)
            return None

    @staticmethod
    def load_records(data_dir: Optional[Path] = None) -> list[SessionRecord]:
        """Load all session records from disk.

        Returns a list of SessionRecord objects from all .jsonl files
        in the data directory.
        """
        directory = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        if not directory.exists():
            return []

        records: list[SessionRecord] = []
        for filepath in sorted(directory.glob("sessions_*.jsonl")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            records.append(_dict_to_record(data))
                        except json.JSONDecodeError as exc:
                            logger.warning("Skipping corrupt line in %s: %s", filepath, exc)
            except OSError as exc:
                logger.warning("Failed to load %s: %s", filepath, exc)

        return records

    @staticmethod
    def load_unrecognized_deps(
        data_dir: Optional[Path] = None,
    ) -> dict[str, int]:
        """Load aggregated unrecognized dependency counts.

        Returns a dict of {dependency_name: occurrence_count} across all
        recorded sessions, useful for prioritizing new profile development.
        """
        records = InteractionRecorder.load_records(data_dir)
        counts: dict[str, int] = {}
        for rec in records:
            for dep in rec.unrecognized_deps:
                counts[dep] = counts.get(dep, 0) + 1
        return counts

    @staticmethod
    def load_failure_patterns(
        data_dir: Optional[Path] = None,
    ) -> list[dict]:
        """Load all failure patterns across sessions.

        Returns a list of failure pattern dicts for scenario generator
        improvement.
        """
        records = InteractionRecorder.load_records(data_dir)
        patterns: list[dict] = []
        for rec in records:
            patterns.extend(rec.failure_patterns)
        return patterns


# ── Anonymization Helpers ──


_session_counter = 0


def _generate_session_id() -> str:
    """Generate a non-traceable session identifier."""
    global _session_counter
    _session_counter += 1
    raw = f"{time.time_ns()}-{id(object())}-{_session_counter}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _anonymize_dependency(dep: DependencyInfo) -> dict:
    """Extract dependency info without file paths or PII."""
    return {
        "name": dep.name,
        "installed_version": dep.installed_version or "",
        "latest_version": dep.latest_version or "",
        "required_version": dep.required_version or "",
        "is_outdated": dep.is_outdated or False,
        "is_missing": dep.is_missing,
    }


def _anonymize_scenario(s: StressTestScenario) -> dict:
    """Extract scenario config without file paths or user code."""
    config = dict(s.test_config)
    # Strip any file path references from test_config
    config.pop("target_files", None)
    return {
        "name": s.name,
        "category": s.category,
        "target_dependencies": list(s.target_dependencies),
        "priority": s.priority,
        "source": s.source,
    }


def _anonymize_scenario_result(sr: ScenarioResult) -> dict:
    """Extract scenario result without stdout/stderr or file paths."""
    steps = []
    for step in sr.steps:
        steps.append({
            "step_name": step.step_name,
            "execution_time_ms": step.execution_time_ms,
            "memory_peak_mb": step.memory_peak_mb,
            "error_count": step.error_count,
            "resource_cap_hit": step.resource_cap_hit,
            # Error types only, no messages (may contain paths)
            "error_types": list({
                e.get("type", "Unknown") if isinstance(e, dict) else "Unknown"
                for e in step.errors
            }),
        })
    return {
        "scenario_name": sr.scenario_name,
        "scenario_category": sr.scenario_category,
        "status": sr.status,
        "total_execution_time_ms": sr.total_execution_time_ms,
        "peak_memory_mb": sr.peak_memory_mb,
        "total_errors": sr.total_errors,
        "failure_indicators_triggered": list(sr.failure_indicators_triggered),
        "resource_cap_hit": sr.resource_cap_hit,
        "steps": steps,
    }


def _count_severities(findings: list[Finding]) -> dict[str, int]:
    """Count findings by severity."""
    counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts


def _extract_failure_patterns(
    report: DiagnosticReport,
    execution_summary: dict,
) -> list[dict]:
    """Extract failure patterns for scenario generator improvement.

    Captures: which dependency combinations failed, under what stress
    category, with what error types, and at what scale.
    """
    patterns: list[dict] = []

    # From findings
    for finding in report.findings:
        if finding.severity in ("critical", "warning"):
            patterns.append({
                "type": "finding",
                "severity": finding.severity,
                "category": finding.category,
                "affected_dependencies": list(finding.affected_dependencies),
                "title": finding.title,
            })

    # From degradation points
    for dp in report.degradation_points:
        patterns.append({
            "type": "degradation",
            "scenario": dp.scenario_name,
            "metric": dp.metric,
            "breaking_point": dp.breaking_point,
        })

    # From execution scenario results
    scenario_results = execution_summary.get("scenario_results", [])
    for sr in scenario_results:
        if sr.get("status") in ("failed", "partial"):
            error_types = set()
            for step in sr.get("steps", []):
                error_types.update(step.get("error_types", []))
            patterns.append({
                "type": "scenario_failure",
                "scenario": sr.get("scenario_name", ""),
                "category": sr.get("scenario_category", ""),
                "error_types": sorted(error_types),
                "resource_cap_hit": sr.get("resource_cap_hit", False),
            })

    return patterns


def _dict_to_record(data: dict) -> SessionRecord:
    """Convert a dict to a SessionRecord, handling missing fields."""
    return SessionRecord(
        session_id=data.get("session_id", ""),
        timestamp=data.get("timestamp", ""),
        language=data.get("language", ""),
        consent_given=data.get("consent_given", False),
        conversation=data.get("conversation", {}),
        dependency_combinations=data.get("dependency_combinations", []),
        recognized_deps=data.get("recognized_deps", []),
        unrecognized_deps=data.get("unrecognized_deps", []),
        scenario_configs=data.get("scenario_configs", []),
        execution_summary=data.get("execution_summary", {}),
        failure_patterns=data.get("failure_patterns", []),
        report_summary=data.get("report_summary", {}),
        version_discrepancies=data.get("version_discrepancies", []),
    )
