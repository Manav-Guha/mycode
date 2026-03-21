"""Report Generator (E2) — Plain-language diagnostic report from stress test results.

Takes raw execution data + FULL ingester output, produces a diagnostic report
that identifies breaking points in terms the user understands.

Per spec, this component:
  - Takes ExecutionEngineResult + full IngestionResult
  - Produces plain-language diagnostic report
  - Describes degradation curves where relevant
  - Identifies breaking points relative to the user's stated operational intent
  - Flags version discrepancies found by the ingester
  - Flags unrecognized dependencies
  - Reports dependency combination failures
  - Does NOT prescribe fixes. Does NOT generate patches. Diagnoses only.

LLM-dependent component (one of three).
"""

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from mycode.classifiers import classify_finding, classify_project
from mycode.constraints import OperationalConstraints
from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.ingester import DependencyInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.scenario import LLMBackend, LLMConfig, LLMError, LLMResponse

logger = logging.getLogger(__name__)

# Generic regex: extract the trailing number from any word_N step name.
# Matches the last ``word_DIGITS`` fragment so it works regardless of
# naming convention (concurrent_50, compute_10000, io_size_1048576, …).
_STEP_LEVEL_RE = re.compile(r"(\w+?)_(\d+)(?:\s|$|:)")

# Scenario categories whose load level represents a concurrency/user count
# and can meaningfully be compared against user_scale.
_CONCURRENCY_CATEGORIES = frozenset({
    "concurrent_execution",
    "blocking_io",
    "gil_contention",
    "async_failures",
})


# ── Exceptions ──


class ReportError(Exception):
    """Base exception for report generator errors."""


# ── Data Classes ──


@dataclass
class Finding:
    """A single diagnostic finding.

    Attributes:
        title: Short finding headline.
        severity: "critical", "warning", or "info".
        category: Which stress category produced this finding.
        description: Plain-language explanation of what happened.
        details: Supporting data (measurements, error messages, etc.).
        affected_dependencies: Dependencies involved in this finding.
        grouped_findings: Other findings collapsed into this representative.
        group_count: Total number of findings in this group (1 = ungrouped).
        _peak_memory_mb: Peak memory metric for grouping similarity.
        _execution_time_ms: Execution time metric for grouping similarity.
        _error_count: Error count metric for grouping similarity.
    """

    title: str
    severity: str  # "critical", "warning", "info"
    category: str = ""
    description: str = ""
    details: str = ""
    affected_dependencies: list[str] = field(default_factory=list)
    grouped_findings: list["Finding"] = field(default_factory=list)
    group_count: int = 1
    failure_domain: str = ""
    failure_pattern: Optional[str] = None
    operational_trigger: str = ""
    source_file: str = ""
    source_function: str = ""
    _peak_memory_mb: float = 0.0
    _execution_time_ms: float = 0.0
    _error_count: int = 0
    _load_level: Optional[int] = None
    _failure_reason: str = ""
    _finding_type: str = ""  # scenario_failed, resource_limit_hit, errors_during, failure_indicators


@dataclass
class DegradationPoint:
    """A point where performance degrades under increasing load.

    Attributes:
        scenario_name: Which scenario exhibited degradation.
        metric: What degraded (e.g. "execution_time_ms", "memory_peak_mb").
        steps: List of (label, value) pairs showing the progression.
        breaking_point: The step label where degradation became severe,
            or empty if performance stayed stable.
        description: Plain-language description of the degradation curve.
        grouped_points: Other degradation points collapsed into this one.
        group_count: Total in this group (1 = ungrouped).
    """

    scenario_name: str
    metric: str
    steps: list[tuple[str, float]] = field(default_factory=list)
    breaking_point: str = ""
    description: str = ""
    grouped_points: list["DegradationPoint"] = field(default_factory=list)
    group_count: int = 1


@dataclass
class DiagnosticReport:
    """Complete diagnostic report output.

    Attributes:
        summary: Executive summary in plain language.
        findings: Individual diagnostic findings, ordered by severity.
        degradation_points: Where performance degraded under load.
        version_flags: Version discrepancy warnings.
        unrecognized_deps: Dependencies without profiles.
        recognized_dep_count: Count of dependencies with targeted profiles.
        scenarios_run: Total scenarios executed.
        scenarios_passed: Scenarios with no critical findings.
        scenarios_failed: Scenarios with errors or resource cap hits.
        total_errors: Total error count across all scenarios.
        operational_context: The user's stated intent, for framing.
        project_description: Auto-generated description from ingester analysis.
        confidence_note: Note about sandbox limitations affecting accuracy.
        model_used: Which LLM produced the narrative (or "offline").
        token_usage: LLM token consumption.
    """

    summary: str = ""
    plain_summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    incomplete_tests: list[Finding] = field(default_factory=list)
    degradation_points: list[DegradationPoint] = field(default_factory=list)
    version_flags: list[str] = field(default_factory=list)
    unrecognized_deps: list[str] = field(default_factory=list)
    recognized_dep_count: int = 0
    recognized_dep_names: list[str] = field(default_factory=list)
    scenarios_run: int = 0
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    scenarios_incomplete: int = 0
    total_errors: int = 0
    operational_context: str = ""
    project_description: str = ""
    confidence_note: str = ""
    vertical: str = ""
    architectural_pattern: str = ""
    business_domain: str = ""
    has_user_constraints: bool = False
    user_scale: int | None = None
    _usage_pattern: str = ""
    _data_type: str = ""
    _data_type_detail: str = ""
    _data_type_note: str = ""
    _max_payload_mb: float | None = None
    _timeout_per_scenario: int | None = None
    _analysis_depth: str | None = None
    baseline_failed: bool = False
    _baseline_report_text: str = ""
    http_ran: bool = False
    project_name: str = ""
    model_used: str = "offline"
    token_usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )

    def as_text(self) -> str:
        """Render the report as readable plain text."""
        if self.baseline_failed and self._baseline_report_text:
            return self._baseline_report_text

        sections: list[str] = []

        # Header
        sections.append("=" * 60)
        sections.append("  myCode Diagnostic Report")
        sections.append("=" * 60)

        # Plain-language summary (for non-technical readers)
        if self.plain_summary:
            sections.append(f"\n{self.plain_summary}")

        # Summary
        if self.summary:
            sections.append(f"\n{self.summary}")

        # Stats bar
        stats_parts = [
            f"Scenarios: {self.scenarios_run} run",
            f"{self.scenarios_passed} clean",
            f"{self.scenarios_failed} with issues",
        ]
        if self.scenarios_incomplete:
            stats_parts.append(
                f"{self.scenarios_incomplete} could not test"
            )
        sections.append("\n" + ", ".join(stats_parts))
        if self.total_errors:
            sections.append(f"Total errors captured: {self.total_errors}")

        # Scenario coverage summary
        completed = self.scenarios_run - len(self.incomplete_tests)
        if self.incomplete_tests:
            sections.append(
                f"myCode fully tested {completed} of "
                f"{self.scenarios_run} scenarios. "
                f"{len(self.incomplete_tests)} could not be run "
                f"(see below)."
            )

        # Confidence note
        if self.confidence_note:
            sections.append(f"\n{self.confidence_note}")

        # Data-type methodology note
        if self._data_type_note:
            sections.append(f"\n{self._data_type_note}")

        # Dependency coverage
        total_deps = self.recognized_dep_count + len(self.unrecognized_deps)
        if total_deps > 0 and self.unrecognized_deps:
            rec_names = ""
            if self.recognized_dep_names:
                rec_names = f": {', '.join(self.recognized_dep_names[:10])}"
            sections.append(
                f"\nmyCode tested {self.recognized_dep_count} of "
                f"{total_deps} dependencies with targeted scenarios{rec_names}. "
                f"{len(self.unrecognized_deps)} "
                f"{'dependency was' if len(self.unrecognized_deps) == 1 else 'dependencies were'} "
                f"tested with general usage-based analysis."
            )

        # Findings — split by severity context
        criticals = [f for f in self.findings if f.severity == "critical"]
        warnings = [f for f in self.findings if f.severity == "warning"]
        infos = [f for f in self.findings if f.severity == "info"]

        # Header changes when no constraints were provided
        critical_header = (
            "  Fix Before Launch" if self.has_user_constraints
            else "  Findings at Default Test Range"
        )

        if not self.findings and self.scenarios_run:
            sections.append("\n" + "-" * 40)
            sections.append("  Findings")
            sections.append("-" * 40)
            sections.append(
                "\n  All stress test scenarios completed cleanly. "
                "No errors, resource limit hits, or degradation detected "
                "under the conditions tested."
            )
        else:
            if criticals:
                sections.append("\n" + "-" * 40)
                sections.append(critical_header)
                sections.append("-" * 40)
                if not self.has_user_constraints:
                    sections.append(
                        "  No usage context was provided, so myCode tested at "
                        "default ranges. For more targeted results, run with "
                        "the conversational interface."
                    )
                for f in criticals:
                    self._render_text_finding(sections, f)

            if warnings:
                sections.append("\n" + "-" * 40)
                sections.append(
                    "  Worth Investigating"
                )
                sections.append("-" * 40)
                for f in warnings:
                    self._render_text_finding(sections, f)

            if infos:
                sections.append("\n" + "-" * 40)
                sections.append(
                    "  Good to Know"
                )
                sections.append("-" * 40)
                for f in infos:
                    self._render_text_finding(sections, f)

        # Tests myCode could not run (harness/environment issues)
        if self.incomplete_tests:
            sections.append("\n" + "-" * 40)
            sections.append("  Tests myCode Could Not Run")
            sections.append("-" * 40)
            _render_incomplete_text(sections, self.incomplete_tests, self.http_ran)

        # Degradation
        if self.degradation_points:
            sections.append("\n" + "-" * 40)
            sections.append("  Degradation Curves")
            sections.append("-" * 40)
            for dp in self.degradation_points:
                header = _describe_scenario(dp.scenario_name) or dp.scenario_name
                metric_label = _metric_label(dp.metric)
                if metric_label:
                    header = f"{metric_label} — {header}"
                if dp.group_count > 1:
                    header += f" (and {dp.group_count - 1} similar)"
                sections.append(f"\n  {header}:")
                narrative = _build_degradation_narrative(dp)
                if narrative:
                    sections.append(f"    {narrative}")
                elif dp.description:
                    sections.append(f"    {dp.description}")
                if dp.breaking_point:
                    bp_label = _breaking_point_label(dp)
                    sections.append(
                        f"    >> Breaking point: {bp_label}"
                    )
                if dp.grouped_points:
                    names = [
                        _humanize_scenario_name(gp.scenario_name)
                        for gp in dp.grouped_points
                    ]
                    shown = names[:5]
                    extra = len(names) - 5
                    also_line = ", ".join(shown)
                    if extra > 0:
                        also_line += f" +{extra} more"
                    sections.append(f"    Also: {also_line}")

        # Version flags
        if self.version_flags:
            sections.append("\n" + "-" * 40)
            sections.append("  Version Discrepancies")
            sections.append("-" * 40)
            for vf in self.version_flags:
                sections.append(f"  - {vf}")

        # Footer
        sections.append("\n" + "=" * 60)
        sections.append(
            "  myCode diagnoses — it does not prescribe. "
            "Interpret results in your context."
        )
        sections.append("=" * 60)

        return "\n".join(sections)

    @staticmethod
    def _render_text_finding(sections: list[str], f: "Finding") -> None:
        """Render a single finding for plain text output."""
        marker = {
            "critical": "[!!]",
            "warning": "[! ]",
            "info": "[  ]",
        }.get(f.severity, "[  ]")
        title = f.title
        if f.group_count > 1:
            title += f" (and {f.group_count - 1} similar)"
        sections.append(f"\n{marker} {title}")
        if f.description:
            sections.append(f"    {f.description}")
        if f.details:
            sections.append(f"    Details: {f.details}")
        if f.grouped_findings:
            names = []
            for gf in f.grouped_findings:
                names.append(gf.title)
            shown = names[:5]
            extra = len(names) - 5
            also_line = ", ".join(shown)
            if extra > 0:
                also_line += f" +{extra} more"
            sections.append(f"    Also: {also_line}")

    def as_markdown(self, project_name: str = "", date: str = "") -> str:
        """Render the report as a clean markdown file for non-technical users.

        No terminal formatting codes, no raw log output. Designed to be
        readable by a vibe coder who wants to know what's wrong and what
        to do about it.
        """
        import datetime as _dt

        if not date:
            date = _dt.date.today().isoformat()

        # Use auto-generated description, then fall back to project_name
        display_name = self.project_description or project_name or "Your Project"

        lines: list[str] = []

        # Title
        lines.append(f"# myCode Report: {display_name}")
        lines.append("")
        lines.append(f"**Date:** {date}")
        lines.append("")

        # Summary score
        total = self.scenarios_run
        passed = self.scenarios_passed
        failed = self.scenarios_failed
        critical_count = sum(
            1 for f in self.findings if f.severity == "critical"
        )
        warning_count = sum(
            1 for f in self.findings if f.severity == "warning"
        )

        if total == 0:
            score_label = "No tests ran"
        elif critical_count > 0:
            score_label = "Needs attention"
        elif warning_count > 0:
            score_label = "Mostly healthy"
        else:
            score_label = "Looking good"

        lines.append(f"**Overall:** {score_label}")
        lines.append("")
        exec_summary = (
            f"We tested your {display_name} with **{total}** stress tests. "
            f"**{passed}** passed cleanly"
            + (f", **{failed}** found issues." if failed else ".")
        )
        if critical_count > 0 and self.findings:
            first_critical = next(
                (f for f in self.findings if f.severity == "critical"), None
            )
            if first_critical:
                exec_summary += f" **Key issue:** {first_critical.title}."
        lines.append(exec_summary)
        lines.append("")

        # Scenario coverage summary
        completed = total - len(self.incomplete_tests)
        if self.incomplete_tests:
            lines.append(
                f"myCode fully tested {completed} of "
                f"{total} scenarios. "
                f"{len(self.incomplete_tests)} could not be run "
                f"(see below)."
            )
            lines.append("")

        # Confidence note
        if self.confidence_note:
            lines.append(f"> {self.confidence_note}")
            lines.append("")

        # Dependency coverage
        total_deps = self.recognized_dep_count + len(self.unrecognized_deps)
        if total_deps > 0 and self.unrecognized_deps:
            rec_names = ""
            if self.recognized_dep_names:
                rec_names = f": {', '.join(self.recognized_dep_names[:10])}"
            lines.append(
                f"myCode tested {self.recognized_dep_count} of "
                f"{total_deps} dependencies with targeted scenarios{rec_names}. "
                f"{len(self.unrecognized_deps)} "
                f"{'dependency was' if len(self.unrecognized_deps) == 1 else 'dependencies were'} "
                f"tested with general usage-based analysis."
            )
            lines.append("")

        # Plain-language summary
        if self.plain_summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(self.plain_summary)
            lines.append("")
        elif self.summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(self.summary)
            lines.append("")

        # Findings grouped by severity with constraint context
        if self.findings:
            criticals = [f for f in self.findings if f.severity == "critical"]
            warnings = [f for f in self.findings if f.severity == "warning"]
            infos = [f for f in self.findings if f.severity == "info"]

            if criticals:
                if self.has_user_constraints:
                    lines.append("## Fix Before Launch")
                    lines.append("")
                    lines.append(
                        "These problems will affect your users under the "
                        "conditions you described. Address them before deploying."
                    )
                else:
                    lines.append("## Findings at Default Test Range")
                    lines.append("")
                    lines.append(
                        "No usage context was provided, so myCode tested at "
                        "default ranges. For more targeted results, run with "
                        "the conversational interface."
                    )
                lines.append("")
                for f in criticals:
                    _render_finding(lines, f)

            if warnings:
                lines.append("## Worth Investigating")
                lines.append("")
                lines.append(
                    "These aren't breaking under your stated conditions, "
                    "but they could become problems as usage grows."
                )
                lines.append("")
                for f in warnings:
                    _render_finding(lines, f)

            if infos:
                lines.append("## Good to Know")
                lines.append("")
                lines.append(
                    "Beyond your current needs, but useful context."
                )
                lines.append("")
                for f in infos:
                    _render_finding(lines, f)
        elif total > 0:
            lines.append("## Findings")
            lines.append("")
            lines.append(
                "All stress tests completed cleanly. No errors, resource "
                "limit hits, or degradation detected under the conditions "
                "tested."
            )
            lines.append("")

        # Tests myCode could not run
        if self.incomplete_tests:
            lines.append("## Tests myCode Could Not Run")
            lines.append("")
            _render_incomplete_markdown(lines, self.incomplete_tests, self.http_ran)

        # Degradation curves (simplified for non-technical readers)
        if self.degradation_points:
            lines.append("## Performance Under Load")
            lines.append("")
            for dp in self.degradation_points:
                name = _describe_scenario(dp.scenario_name) or dp.scenario_name
                metric_label = _metric_label(dp.metric)
                if metric_label:
                    name = f"{metric_label} — {name}"
                if dp.group_count > 1:
                    name += f" (and {dp.group_count - 1} similar)"
                lines.append(f"**{name}**")
                lines.append("")
                narrative = _build_degradation_narrative(dp)
                if narrative:
                    lines.append(narrative)
                    lines.append("")
                elif dp.description:
                    lines.append(dp.description)
                    lines.append("")
                if dp.breaking_point:
                    bp_label = _breaking_point_label(dp)
                    lines.append(
                        f"Breaking point: **{bp_label}**"
                    )
                    lines.append("")

        # Version discrepancies
        if self.version_flags:
            lines.append("## Outdated Dependencies")
            lines.append("")
            lines.append(
                "These dependencies have newer versions available. "
                "Updating may fix known bugs or security issues."
            )
            lines.append("")
            for vf in self.version_flags:
                lines.append(f"- {vf}")
            lines.append("")

        # What To Do Next
        if self.findings:
            lines.append("## What To Do Next")
            lines.append("")
            lines.append(
                "Copy the findings above and paste them into your coding "
                "agent (Claude Code, Cursor, Copilot). The specific issues "
                "and affected dependencies are formatted for an AI to "
                "understand and investigate."
            )
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(
            "*myCode diagnoses — it does not prescribe. "
            "Interpret these results in your own context.*"
        )
        lines.append("")

        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Serialize the report as a JSON-compatible dictionary.

        Returns a structured dict suitable for ``json.dumps()``.  Every
        ``Finding`` and ``DegradationPoint`` is fully expanded, including
        grouped sub-items.
        """

        def _finding_dict(f: Finding) -> dict:
            # Lazy import to avoid circular dependency (documents imports report)
            from mycode.documents import generate_finding_prompt
            d: dict = {
                "title": f.title,
                "severity": f.severity,
                "category": f.category,
                "description": f.description,
                "details": f.details,
                "load_level": f._load_level,
                "affected_dependencies": list(f.affected_dependencies),
                "group_count": f.group_count,
                "failure_domain": f.failure_domain,
                "failure_pattern": f.failure_pattern,
                "operational_trigger": f.operational_trigger,
                "failure_reason": f._failure_reason,
                "source_file": f.source_file,
                "source_function": f.source_function,
                "prompt": generate_finding_prompt(f),
            }
            if f.grouped_findings:
                d["grouped_findings"] = [
                    _finding_dict(gf) for gf in f.grouped_findings
                ]
            return d

        def _degradation_dict(dp: DegradationPoint) -> dict:
            d: dict = {
                "scenario_name": dp.scenario_name,
                "metric": dp.metric,
                "steps": [
                    {"label": label, "value": value}
                    for label, value in dp.steps
                ],
                "breaking_point": dp.breaking_point,
                "description": dp.description,
                "group_count": dp.group_count,
            }
            if dp.grouped_points:
                d["grouped_points"] = [
                    _degradation_dict(gp) for gp in dp.grouped_points
                ]
            return d

        return {
            "summary": self.summary,
            "plain_summary": self.plain_summary,
            "project_description": self.project_description,
            "confidence_note": self.confidence_note,
            "data_type_note": self._data_type_note,
            "statistics": {
                "scenarios_run": self.scenarios_run,
                "scenarios_passed": self.scenarios_passed,
                "scenarios_failed": self.scenarios_failed,
                "scenarios_incomplete": self.scenarios_incomplete,
                "total_errors": self.total_errors,
                "recognized_dependencies": self.recognized_dep_count,
                "recognized_dependency_names": list(self.recognized_dep_names),
                "unrecognized_dependencies": len(self.unrecognized_deps),
            },
            "findings": [_finding_dict(f) for f in self.findings],
            "incomplete_tests": [_finding_dict(f) for f in self.incomplete_tests],
            "degradation_curves": [
                _degradation_dict(dp) for dp in self.degradation_points
            ],
            "version_discrepancies": list(self.version_flags),
            "unrecognized_dependencies": list(self.unrecognized_deps),
            "operational_context": self.operational_context,
            "vertical": self.vertical,
            "architectural_pattern": self.architectural_pattern,
            "business_domain": self.business_domain,
            "model_used": self.model_used,
            "token_usage": dict(self.token_usage),
        }


# ── Plain-language explanations for harness failure reasons ──

_FAILURE_REASON_EXPLANATIONS: dict[str, str] = {
    "unsupported_framework": (
        "myCode does not yet support this framework. "
        "This is a myCode limitation, not a problem with your code."
    ),
    "dependency_unavailable": (
        "These tests could not run because dependencies were not installed "
        "in the test environment. Use --containerised for better dependency support."
    ),
    "harness_generation_error": (
        "myCode could not generate valid test scripts for these scenarios. "
        "This is a myCode limitation."
    ),
    "browser_framework": (
        "These scenarios target a frontend framework that requires a browser "
        "environment (DOM, JSX). myCode tests these projects via HTTP load "
        "testing instead of callable harnesses."
    ),
    "module_import_failure": (
        "myCode could not import your project's modules for these tests. "
        "This can happen with non-standard project structures or monorepos."
    ),
    "timeout": "These tests exceeded the time limit.",
    "user_timeout": (
        "These tests reached your time limit. Re-run with a longer "
        "limit for deeper testing."
    ),
    "runtime_context_required": (
        "These functions require runtime context (e.g. a running Streamlit "
        "server, database connection, or live API) that myCode cannot "
        "simulate in isolation."
    ),
    "http_tested": (
        "These scenarios target a server framework that myCode tested "
        "via HTTP load testing instead — see HTTP findings in this report."
    ),
    "unknown": "These tests could not run due to unexpected issues.",
}


def _runtime_ctx_explanation(http_ran: bool) -> str:
    """Return the appropriate runtime-context explanation based on whether HTTP testing ran."""
    if http_ran:
        return (
            "These functions could not be tested in isolation, but myCode "
            "tested your application under load via HTTP — see HTTP "
            "findings in this report."
        )
    return (
        "These functions require runtime context (e.g. a running Streamlit "
        "server, database connection, or live API) that myCode cannot "
        "simulate in isolation."
    )

_FAILURE_REASON_HEADERS: dict[str, str] = {
    "unsupported_framework": "Unsupported Framework",
    "dependency_unavailable": "Missing Dependencies",
    "harness_generation_error": "Test Script Generation Issue",
    "browser_framework": "Browser Framework",
    "module_import_failure": "Module Import Issue",
    "timeout": "Timed Out",
    "user_timeout": "Reached Your Time Limit",
    "runtime_context_required": "Requires Runtime Context",
    "http_tested": "Tested via HTTP",
    "unknown": "Other Issues",
    "": "Environment Issues",
}


def _group_by_failure_reason(
    findings: list["Finding"],
) -> dict[str, list["Finding"]]:
    """Group incomplete test findings by their failure_reason."""
    groups: dict[str, list["Finding"]] = {}
    for f in findings:
        reason = f._failure_reason or ""
        groups.setdefault(reason, []).append(f)
    return groups


def _render_incomplete_text(
    sections: list[str], findings: list["Finding"],
    http_ran: bool = False,
) -> None:
    """Render incomplete tests grouped by failure reason for plain text."""
    groups = _group_by_failure_reason(findings)
    for reason, group in groups.items():
        # http_tested: single summary line, never individual findings
        if reason == "http_tested":
            n = len(group)
            framework = ""
            for f in group:
                if f.affected_dependencies:
                    framework = f.affected_dependencies[0]
                    break
            label = f"{framework} " if framework else "framework "
            sections.append(
                f"\n    {n} {label}scenario{'s' if n != 1 else ''} "
                f"{'were' if n != 1 else 'was'} tested via HTTP load testing."
            )
            continue

        header = _FAILURE_REASON_HEADERS.get(reason, "Other Issues")
        explanation = _runtime_ctx_explanation(http_ran) if reason == "runtime_context_required" else _FAILURE_REASON_EXPLANATIONS.get(
            reason,
            "These tests could not run fully due to environment issues.",
        )
        n = len(group)
        sections.append(f"\n  {header} ({n} test{'s' if n != 1 else ''}):")
        sections.append(f"    {explanation}")
        if n <= 2:
            for f in group:
                sections.append(f"    - {f.title}")
        else:
            # Summarize — don't list individual failures for 3+
            names = [f.title.replace("Could not test: ", "") for f in group]
            shown = names[:3]
            extra = n - 3
            line = ", ".join(shown)
            if extra > 0:
                line += f", and {extra} more"
            sections.append(f"    Affected: {line}")


def _render_incomplete_markdown(
    lines: list[str], findings: list["Finding"],
    http_ran: bool = False,
) -> None:
    """Render incomplete tests grouped by failure reason for markdown."""
    groups = _group_by_failure_reason(findings)
    for reason, group in groups.items():
        # http_tested: single summary line, never individual findings
        if reason == "http_tested":
            n = len(group)
            framework = ""
            for f in group:
                if f.affected_dependencies:
                    framework = f.affected_dependencies[0]
                    break
            label = f"{framework} " if framework else "framework "
            lines.append(
                f"*{n} {label}scenario{'s' if n != 1 else ''} "
                f"{'were' if n != 1 else 'was'} tested via HTTP load testing.*"
            )
            lines.append("")
            continue

        header = _FAILURE_REASON_HEADERS.get(reason, "Other Issues")
        explanation = _runtime_ctx_explanation(http_ran) if reason == "runtime_context_required" else _FAILURE_REASON_EXPLANATIONS.get(
            reason,
            "These tests could not run fully due to environment issues.",
        )
        n = len(group)
        lines.append(f"### {header} ({n} test{'s' if n != 1 else ''})")
        lines.append("")
        lines.append(explanation)
        lines.append("")
        if n <= 2:
            for f in group:
                lines.append(f"- {f.title}")
            lines.append("")
        else:
            names = [f.title.replace("Could not test: ", "") for f in group]
            shown = names[:3]
            extra = n - 3
            line = ", ".join(shown)
            if extra > 0:
                line += f", and {extra} more"
            lines.append(f"Affected: {line}")
            lines.append("")


def _render_finding(lines: list[str], f: "Finding") -> None:
    """Render a single Finding as markdown bullet points."""
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    lines.append(f"### {title}")
    lines.append("")
    if f.description:
        lines.append(f.description)
        lines.append("")
    if f.details:
        lines.append(f"> {f.details}")
        lines.append("")
    if f.affected_dependencies:
        lines.append(
            f"**Related dependencies:** {', '.join(f.affected_dependencies)}"
        )
        lines.append("")


def _build_dep_file_map(ingestion: IngestionResult) -> dict[str, list[str]]:
    """Build a mapping from dependency name → list of files that import it.

    Files are ordered by usage count (most imports first) so that
    ``dep_map[dep][0]`` is the primary file for that dependency.
    """
    from collections import Counter

    dep_files: dict[str, Counter[str]] = {}
    for fa in ingestion.file_analyses:
        for imp in fa.imports:
            # Use top-level package name (e.g. "flask" from "flask.views")
            top = imp.module.split(".")[0] if imp.module else ""
            if not top:
                continue
            if top not in dep_files:
                dep_files[top] = Counter()
            # Count each from-import name as a usage, minimum 1 per import
            dep_files[top][fa.file_path] += max(len(imp.names), 1)

    # Convert to sorted lists (most usage first)
    return {
        dep: [f for f, _ in counter.most_common()]
        for dep, counter in dep_files.items()
    }


def _resolve_source_file(
    affected_deps: list[str],
    dep_file_map: dict[str, list[str]],
) -> str:
    """Find the primary source file for a set of affected dependencies.

    Returns the file with the most usage of the first matching dependency,
    or empty string if no match is found.
    """
    for dep in affected_deps:
        files = dep_file_map.get(dep)
        if files:
            return files[0]
    return ""


# ── Report Generator ──


class ReportGenerator:
    """Generates plain-language diagnostic reports from stress test results.

    Usage::

        generator = ReportGenerator(llm_config=LLMConfig(api_key="..."))
        report = generator.generate(execution, ingestion, matches, intent)

    For testing without API calls::

        generator = ReportGenerator(offline=True)
        report = generator.generate(execution, ingestion, matches, intent)
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        offline: bool = False,
    ) -> None:
        self._llm_config = llm_config or LLMConfig()
        self._offline = offline
        self._backend: Optional[LLMBackend] = None

        if not offline:
            try:
                self._backend = LLMBackend(self._llm_config)
            except LLMError as exc:
                logger.warning(
                    "LLM backend init failed, falling back to offline: %s", exc
                )
                self._offline = True

    def generate(
        self,
        execution: ExecutionEngineResult,
        ingestion: IngestionResult,
        profile_matches: list[ProfileMatch],
        operational_intent: str = "",
        project_name: str = "",
        constraints: Optional[OperationalConstraints] = None,
    ) -> DiagnosticReport:
        """Generate a diagnostic report.

        Args:
            execution: Raw results from the Execution Engine.
            ingestion: Full project ingestion result.
            profile_matches: Component library matches.
            operational_intent: User's stated intent from the
                Conversational Interface.
            project_name: Short project name from user.
            constraints: Structured constraints for intent-contextualised
                reporting.  When provided, findings are classified relative
                to user-stated capacity.

        Returns:
            DiagnosticReport with findings, degradation curves, and flags.
        """
        report = DiagnosticReport(
            operational_context=operational_intent,
            scenarios_run=len(execution.scenario_results),
        )

        # 0. Track whether HTTP testing ran (for runtime-context messaging)
        report.http_ran = execution.http_ran

        # 0a. Classify project first — description depends on it
        self._classify_project(report, ingestion)

        # 0b. Generate project description using classifier output
        report.project_description = _generate_project_description(
            ingestion, project_name,
            vertical=report.vertical,
            architectural_pattern=report.architectural_pattern,
        )

        # 0c. Build confidence note from incomplete tests
        report.confidence_note = _build_confidence_note(
            ingestion, profile_matches,
        )

        # 0d. Track dependency coverage
        report.recognized_dep_names = [
            pm.dependency_name for pm in profile_matches if pm.profile is not None
        ]
        report.recognized_dep_count = len(report.recognized_dep_names)

        # 0e. Set constraint flag and store parsed constraint fields
        report.has_user_constraints = (
            constraints is not None and constraints.user_scale is not None
        )
        if constraints is not None:
            if constraints.user_scale is not None:
                report.user_scale = constraints.user_scale
            if constraints.usage_pattern:
                report._usage_pattern = constraints.usage_pattern
            if constraints.data_type:
                report._data_type = constraints.data_type
            if constraints.data_type_detail:
                report._data_type_detail = constraints.data_type_detail
            if constraints.max_payload_mb is not None:
                report._max_payload_mb = constraints.max_payload_mb
            if constraints.timeout_per_scenario is not None:
                report._timeout_per_scenario = constraints.timeout_per_scenario
            if constraints.analysis_depth:
                report._analysis_depth = constraints.analysis_depth

            # Build data-type methodology note
            if constraints.data_type and constraints.data_type != "mixed":
                report._data_type_note = _build_data_type_note(
                    constraints.data_type,
                    constraints.data_type_detail or "",
                )
            elif constraints.data_type == "mixed":
                detail = constraints.data_type_detail or "mixed data types"
                report._data_type_note = (
                    f"You indicated {detail} — myCode ran all scenario "
                    f"categories at equal priority. Format-specific stress "
                    f"testing (malformed inputs, encoding edge cases) is "
                    f"planned for v2."
                )

        # 0f. Build dependency → file mapping for source_file fallback
        dep_file_map = _build_dep_file_map(ingestion)

        # 1. Analyze execution results → findings + degradation
        self._analyze_execution(execution, report, dep_file_map)

        # 1b. Merge HTTP-specific findings and degradation points
        if execution.http_findings:
            for hf in execution.http_findings:
                if not hf.source_file and hf.affected_dependencies:
                    hf.source_file = _resolve_source_file(
                        hf.affected_dependencies, dep_file_map,
                    )
            report.findings.extend(execution.http_findings)
        if execution.http_degradation_points:
            report.degradation_points.extend(execution.http_degradation_points)

        # 1c. Suppress browser-framework incomplete tests when HTTP testing
        #     produced findings — the skip entries are implementation noise
        has_http_findings = any(
            f.category == "http_load_testing" for f in report.findings
        )
        if has_http_findings:
            report.incomplete_tests = [
                f for f in report.incomplete_tests
                if f._failure_reason != "browser_framework"
            ]

        # 1a. Auto-classify every finding against taxonomy
        self._classify_all_findings(report)

        # 2. Flag version discrepancies from ingester
        self._flag_version_discrepancies(ingestion, profile_matches, report)

        # 2a. Enrich HTTP startup-failure findings with version discrepancy context
        self._enrich_startup_failure_with_version(ingestion, report)

        # 3. Flag unrecognized dependencies
        self._flag_unrecognized_deps(profile_matches, report)

        # 3a. Apply constraint-driven severity classification
        if constraints is not None:
            self._contextualise_findings(report, constraints, profile_matches)

        # 3b. Group similar findings and degradation points to reduce noise
        report.findings = self._group_similar_findings(report.findings)
        report.degradation_points = self._group_similar_degradation_points(
            report.degradation_points,
        )

        # 3c. Cross-type dedup: fold memory degradation points into findings
        #      for the same scenario to avoid reporting the same memory
        #      behavior twice (once as a finding, once as a degradation curve).
        report.degradation_points = self._fold_memory_degradation_into_findings(
            report.findings, report.degradation_points,
        )

        # 4. Sort findings by severity (critical first)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        report.findings.sort(key=lambda f: severity_order.get(f.severity, 9))

        # 4a. Sort degradation points by ratio (worst degradation first)
        report.degradation_points.sort(
            key=lambda dp: self._degradation_ratio(dp), reverse=True,
        )

        # 4b. Generate plain-language summary for non-technical readers
        report.plain_summary = self._generate_plain_summary(
            report, operational_intent, project_name,
        )

        # 4c. If online with API key, enhance plain summary via Gemini Flash
        if not self._offline and self._llm_config and self._llm_config.api_key:
            llm_plain = self._generate_llm_plain_summary(
                report, operational_intent, project_name,
            )
            if llm_plain:
                report.plain_summary = llm_plain

        # 5. Generate narrative summary
        if self._offline:
            report.summary = self._generate_offline_summary(report)
            report.model_used = "offline"
        else:
            report.summary = self._generate_llm_summary(
                execution, ingestion, report, operational_intent,
            )

        return report

    # ── Execution Analysis ──

    # Error types that indicate the test environment was incomplete,
    # not that the user's code failed under stress.
    _ENVIRONMENT_ERROR_TYPES = frozenset({
        "ModuleNotFoundError",
        "ImportError",
        "FileNotFoundError",
        "ConfigError",
    })

    def _is_environment_only(self, sr: ScenarioResult) -> bool:
        """True when every error in a scenario is an environment issue."""
        if sr.total_errors == 0:
            return False
        seen = 0
        for step in sr.steps:
            for err in step.errors:
                etype = err.get("type", "") if isinstance(err, dict) else ""
                if etype not in self._ENVIRONMENT_ERROR_TYPES:
                    return False
                seen += 1
        # Must have inspected at least one typed error dict
        return seen > 0

    def _analyze_execution(
        self,
        execution: ExecutionEngineResult,
        report: DiagnosticReport,
        dep_file_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Extract findings and degradation curves from execution results."""
        passed = 0
        failed = 0
        incomplete = 0
        total_errors = 0

        # Choose runtime-context message based on whether HTTP testing ran
        if execution.http_ran:
            _runtime_ctx_desc = (
                "This function could not be tested in isolation, but "
                "myCode tested your application under load via HTTP — "
                "see HTTP findings in this report."
            )
        else:
            _runtime_ctx_desc = (
                "This function requires runtime context that myCode "
                "cannot simulate in isolation."
            )

        for sr in execution.scenario_results:
            total_errors += sr.total_errors
            _findings_before = len(report.findings)

            # Extract source file/function for document generation
            sr_source_file = sr.source_files[0] if sr.source_files else ""
            sr_source_function = sr.source_functions[0] if sr.source_functions else ""

            # Extract metrics for grouping
            sr_peak_memory = max(
                (s.memory_peak_mb for s in sr.steps), default=0.0,
            )
            sr_exec_time = max(
                (s.execution_time_ms for s in sr.steps), default=0.0,
            )

            def _tag_source(f: Finding) -> Finding:
                """Attach source file/function from scenario result.

                Falls back to dep→file mapping when the scenario doesn't
                provide explicit source info (e.g. coupling scenarios).
                """
                f.source_file = sr_source_file
                f.source_function = sr_source_function
                if not f.source_file and f.affected_dependencies and dep_file_map:
                    f.source_file = _resolve_source_file(
                        f.affected_dependencies, dep_file_map,
                    )
                return f

            # ── User timeout → INFO finding with re-run suggestion ──
            if sr.hit_user_timeout:
                deps = self._deps_from_name(sr.scenario_name)
                desc = _build_timeout_description(
                    sr, execution.scenario_results, deps,
                    report._timeout_per_scenario,
                )
                f = Finding(
                    title=f"Reached time limit: {_humanize_title_name(sr.scenario_name)}",
                    severity="info",
                    category=sr.scenario_category,
                    description=desc,
                    affected_dependencies=deps,
                )
                f._failure_reason = "user_timeout"
                report.incomplete_tests.append(_tag_source(f))
                incomplete += 1
                continue

            # ── Budget exceeded → INFO finding ──
            if sr.failure_reason == "budget_exceeded":
                n_steps = len(sr.steps)
                depth_hint = (
                    " Choose 'deep analysis' for more thorough testing."
                    if report._analysis_depth != "deep" else ""
                )
                f = Finding(
                    title=f"Time budget reached: {_humanize_title_name(sr.scenario_name)}",
                    severity="info",
                    category=sr.scenario_category,
                    description=(
                        f"This scenario completed {n_steps} "
                        f"step{'s' if n_steps != 1 else ''} within its "
                        f"time allocation.{depth_hint}"
                    ),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._failure_reason = "budget_exceeded"
                report.incomplete_tests.append(_tag_source(f))
                incomplete += 1
                continue

            # ── Harness failures → incomplete tests (myCode limitation) ──
            if sr.failure_reason:
                if sr.failure_reason == "runtime_context_required":
                    desc = _runtime_ctx_desc
                else:
                    desc = sr.summary or "Test could not run."
                f = Finding(
                    title=f"Could not test: {_humanize_title_name(sr.scenario_name)}",
                    severity="info",
                    category=sr.scenario_category,
                    description=desc,
                    details=self._summarize_errors(sr),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._failure_reason = sr.failure_reason
                f._error_count = sr.total_errors
                report.incomplete_tests.append(_tag_source(f))
                incomplete += 1
                continue

            # ── Probe-skipped functions on a scenario that still ran ──
            if sr.probe_skipped:
                for ps in sr.probe_skipped:
                    pf = Finding(
                        title=f"Could not test: {ps.get('name', 'unknown')}",
                        severity="info",
                        category=sr.scenario_category,
                        description=_runtime_ctx_desc,
                        affected_dependencies=self._deps_from_name(
                            sr.scenario_name,
                        ),
                    )
                    pf._failure_reason = "runtime_context_required"
                    report.incomplete_tests.append(_tag_source(pf))

            # ── Environment-only failures → incomplete tests ──
            if self._is_environment_only(sr):
                f = Finding(
                    title=f"Could not test: {_humanize_title_name(sr.scenario_name)}",
                    severity="info",
                    category=sr.scenario_category,
                    description=(
                        "This test could not run fully due to missing "
                        "modules or files in the test environment."
                    ),
                    details=self._summarize_errors(sr),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._error_count = sr.total_errors
                report.incomplete_tests.append(_tag_source(f))
                incomplete += 1
                continue

            # Extract load level info from steps (for contextualisation)
            load_detail = self._load_level_detail(sr)
            failing_load = self._first_failing_load(sr)

            # Build structured description parts
            activity = _describe_scenario(sr.scenario_name) or sr.scenario_name
            consequence = _consequence_for_category(
                sr.scenario_category, sr.scenario_name,
            )

            # Check for failures
            if sr.status == "failed":
                what_happened = sr.summary or "The scenario failed during execution."
                desc = f"myCode tested {activity}. {what_happened}"
                if consequence:
                    desc = f"{desc} {consequence}"
                f = Finding(
                    title=_humanize_title_name(sr.scenario_name),
                    severity="critical",
                    category=sr.scenario_category,
                    description=desc,
                    details=load_detail.strip() if load_detail else "",
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._finding_type = "scenario_failed"
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                f._load_level = failing_load
                report.findings.append(_tag_source(f))

            # Check resource cap hits
            if sr.resource_cap_hit:
                cap_detail = self._summarize_cap_hits(sr)
                if load_detail:
                    cap_detail = f"{cap_detail}{load_detail}"
                base_desc = (
                    f"myCode tested {activity}. "
                    f"Your code exceeded safe operating limits during this test."
                )
                if consequence:
                    base_desc = f"{base_desc} {consequence}"
                f = Finding(
                    title=_humanize_title_name(sr.scenario_name),
                    severity="critical",
                    category=sr.scenario_category,
                    description=base_desc,
                    details=cap_detail,
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._finding_type = "resource_limit_hit"
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                f._load_level = failing_load
                report.findings.append(_tag_source(f))

            # Check for errors — skip if a CRITICAL finding already covers
            # this scenario (resource_cap_hit or failed status) to avoid
            # duplicate findings at different severity levels.
            if (
                sr.total_errors > 0
                and sr.status != "failed"
                and not sr.resource_cap_hit
            ):
                err_detail = self._summarize_errors(sr)
                if load_detail:
                    err_detail = f"{err_detail}{load_detail}"
                base_desc = (
                    f"myCode tested {activity}. "
                    f"{sr.total_errors} error(s) occurred during this test."
                )
                if consequence:
                    base_desc = f"{base_desc} {consequence}"
                f = Finding(
                    title=_humanize_title_name(sr.scenario_name),
                    severity="warning",
                    category=sr.scenario_category,
                    description=base_desc,
                    details=err_detail,
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._finding_type = "errors_during"
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                f._load_level = failing_load
                report.findings.append(_tag_source(f))

            # Check failure indicators — skip if a CRITICAL finding
            # already covers this scenario to avoid duplicate findings.
            if sr.failure_indicators_triggered and not (
                sr.status == "failed" or sr.resource_cap_hit
            ):
                human_triggers = [
                    _humanize_trigger(t)
                    for t in sr.failure_indicators_triggered
                ]
                f = Finding(
                    title=_humanize_title_name(sr.scenario_name),
                    severity="warning",
                    category=sr.scenario_category,
                    description=(
                        f"myCode detected: {', '.join(human_triggers)}."
                    ),
                    details=load_detail.strip() if load_detail else "",
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._finding_type = "failure_indicators"
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                f._load_level = failing_load
                report.findings.append(_tag_source(f))

            # Detect degradation curves
            self._detect_degradation(sr, report)

            # ── Count scenario outcome based on what was actually produced ──
            # Runtime context, HTTP-deferred, and user-timeout are excluded from counts
            if sr.failure_reason in ("runtime_context_required", "http_tested", "budget_exceeded") or sr.hit_user_timeout:
                incomplete += 1
            elif len(report.findings) > _findings_before:
                # Scenario produced at least one finding → failed
                failed += 1
            elif sr.failure_reason or self._is_environment_only(sr):
                # Routed entirely to incomplete_tests → incomplete
                incomplete += 1
            elif sr.status == "completed" and sr.total_errors == 0 and not sr.resource_cap_hit:
                passed += 1
            else:
                # Edge case: errors but no finding generated (e.g.
                # probe-skipped scenario with clean remaining execution).
                # Count as incomplete rather than silently failed.
                incomplete += 1

        report.scenarios_passed = passed
        report.scenarios_failed = failed
        report.scenarios_incomplete = incomplete
        report.total_errors = total_errors

    @staticmethod
    def _classify_all_findings(report: DiagnosticReport) -> None:
        """Apply taxonomy classifiers to every finding in the report."""
        all_findings = list(report.findings) + list(report.incomplete_tests)
        for finding in all_findings:
            # Extract primary error type from details
            error_type = ""
            if finding.details:
                # Look for known Python/JS error type names
                for etype in (
                    "MemoryError", "TimeoutError", "TypeError", "ValueError",
                    "KeyError", "IndexError", "ImportError", "ModuleNotFoundError",
                    "FileNotFoundError", "ConnectionError", "RuntimeError",
                    "AttributeError", "OSError", "PermissionError",
                    "UnicodeDecodeError", "JSONDecodeError",
                    "ConnectionRefusedError", "ConnectionResetError",
                    "BrokenPipeError",
                ):
                    if etype in finding.details:
                        error_type = etype
                        break

            classification = classify_finding(
                scenario_name=finding.title,
                scenario_category=finding.category,
                error_type=error_type,
                error_details=finding.details,
            )
            finding.failure_domain = classification["failure_domain"]
            finding.failure_pattern = classification["failure_pattern"]
            finding.operational_trigger = classification["operational_trigger"]

    @staticmethod
    def _classify_project(
        report: DiagnosticReport,
        ingestion: IngestionResult,
    ) -> None:
        """Classify project vertical and architectural pattern."""
        deps = [d.name for d in ingestion.dependencies if not d.is_dev]
        files = [fa.file_path for fa in ingestion.file_analyses]

        # Detect frontend/backend
        has_frontend = any(
            kw in f.lower()
            for f in files
            for kw in ("component", "page", "template", "view", "frontend")
        )
        has_backend = any(
            kw in f.lower()
            for f in files
            for kw in ("route", "api", "server", "backend", "endpoint")
        )

        project_cls = classify_project(
            dependencies=deps,
            file_structure=files,
            framework="",
            file_count=ingestion.files_analyzed,
            has_frontend=has_frontend,
            has_backend=has_backend,
            project_name=report.project_name,
        )
        report.vertical = project_cls["vertical"]
        report.architectural_pattern = project_cls["architectural_pattern"]
        report.business_domain = project_cls["business_domain"]

    def _detect_degradation(
        self,
        sr: ScenarioResult,
        report: DiagnosticReport,
    ) -> None:
        """Detect degradation curves in scenario step results."""
        if len(sr.steps) < 2:
            return

        # Check time degradation
        time_steps = [
            (s.step_name, s.execution_time_ms)
            for s in sr.steps
            if s.execution_time_ms > 0
        ]
        if len(time_steps) >= 2:
            dp = self._analyze_curve(
                sr.scenario_name, "execution_time_ms", time_steps,
            )
            if dp:
                self._annotate_non_monotonic(dp)
                report.degradation_points.append(dp)

        # Check memory degradation
        mem_steps = [
            (s.step_name, s.memory_peak_mb)
            for s in sr.steps
            if s.memory_peak_mb > 0
        ]
        if len(mem_steps) >= 2:
            dp = self._analyze_curve(
                sr.scenario_name, "memory_peak_mb", mem_steps,
            )
            if dp:
                self._annotate_non_monotonic(dp)
                report.degradation_points.append(dp)

        # Check error accumulation
        error_steps = [
            (s.step_name, float(s.error_count))
            for s in sr.steps
        ]
        if any(v > 0 for _, v in error_steps) and len(error_steps) >= 2:
            dp = self._analyze_curve(
                sr.scenario_name, "error_count", error_steps,
            )
            if dp:
                self._annotate_non_monotonic(dp)
                report.degradation_points.append(dp)

    # Minimum absolute change (last - first) below which a ratio-based
    # detection is suppressed.  Prevents reporting e.g. 0.05ms → 0.11ms
    # as "2.2x degradation" when the real delta is negligible noise.
    _MIN_MEANINGFUL_DELTA: dict[str, float] = {
        "execution_time_ms": 50.0,   # 50 ms
        "memory_peak_mb": 2.0,       # 2 MB
        "error_count": 1.0,          # 1 error
    }

    @staticmethod
    def _analyze_curve(
        scenario_name: str,
        metric: str,
        steps: list[tuple[str, float]],
    ) -> Optional[DegradationPoint]:
        """Analyze a sequence of measurements for degradation.

        Returns a DegradationPoint if degradation is detected, else None.
        Degradation is detected when the final value is >2x the first value,
        or any step shows a >3x jump from the previous step.

        Returns None (no degradation) when the data is essentially flat —
        i.e. the coefficient of variation is below 10 % — or when the
        absolute change is below the per-metric noise threshold.
        """
        if len(steps) < 2:
            return None

        values = [v for _, v in steps]
        first = values[0]
        last = values[-1]

        # ── Flatness gate ──
        # If all values cluster within ~10 % of the mean, the data is
        # measurement noise, not a degradation trend.
        mean_val = sum(values) / len(values)
        if mean_val > 0.001:
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            cv = (variance ** 0.5) / mean_val
            if cv < 0.10:
                return None

        # ── Minimum absolute delta gate ──
        min_delta = ReportGenerator._MIN_MEANINGFUL_DELTA.get(metric, 0.0)
        if abs(last - first) < min_delta:
            return None

        # Skip if first value is essentially zero
        if first < 0.001:
            # Check if later values are nonzero (degradation from zero baseline)
            if last > 0.1:
                return DegradationPoint(
                    scenario_name=scenario_name,
                    metric=metric,
                    steps=steps,
                    breaking_point=steps[-1][0],
                    description=(
                        f"{_human_metric(metric)} grew from near zero to "
                        f"{last:.2f} across the test steps."
                    ),
                )
            return None

        ratio = last / first

        # Find the breaking point — where the biggest jump happened
        breaking_point = ""
        max_jump_ratio = 1.0
        for i in range(1, len(values)):
            if values[i - 1] > 0.001:
                jump = values[i] / values[i - 1]
                if jump > max_jump_ratio:
                    max_jump_ratio = jump
                    breaking_point = steps[i][0]

        # Overall 2x+ degradation
        if ratio >= 2.0:
            return DegradationPoint(
                scenario_name=scenario_name,
                metric=metric,
                steps=steps,
                breaking_point=breaking_point,
                description=(
                    f"{_human_metric(metric)} increased {ratio:.1f}x from "
                    f"start ({first:.2f}) to end ({last:.2f})."
                ),
            )

        # Single step 3x+ spike
        if max_jump_ratio >= 3.0:
            return DegradationPoint(
                scenario_name=scenario_name,
                metric=metric,
                steps=steps,
                breaking_point=breaking_point,
                description=(
                    f"{_human_metric(metric)} spiked {max_jump_ratio:.1f}x "
                    f"at step '{breaking_point}'."
                ),
            )

        return None

    @staticmethod
    def _annotate_non_monotonic(dp: DegradationPoint) -> None:
        """Flag non-monotonic data points in a degradation curve.

        When a value drops by >40% from the previous step, append a
        variance note to the description so users know the data point
        may reflect infrastructure conditions rather than application
        behaviour.
        """
        if len(dp.steps) < 3:
            return

        anomalies: list[str] = []
        for i in range(1, len(dp.steps)):
            prev_val = dp.steps[i - 1][1]
            curr_val = dp.steps[i][1]
            if prev_val > 0.001 and curr_val < prev_val * 0.6:
                label = dp.steps[i][0]
                anomalies.append(label)

        if anomalies:
            note = (
                "Note: results showed significant variance at "
                + (
                    f"step '{anomalies[0]}'"
                    if len(anomalies) == 1
                    else f"steps {', '.join(repr(a) for a in anomalies)}"
                )
                + " — this data point may reflect infrastructure "
                "conditions rather than application behaviour."
            )
            if dp.description:
                dp.description = f"{dp.description} {note}"
            else:
                dp.description = note

    # ── Constraint-Driven Contextualisation ──

    @staticmethod
    def _contextualise_findings(
        report: DiagnosticReport,
        constraints: OperationalConstraints,
        profile_matches: Optional[list[ProfileMatch]] = None,
    ) -> None:
        """Classify finding severity relative to user-stated constraints.

        Per spec §7:
        - Within stated capacity → CRITICAL
        - Beyond stated capacity but ≤3x → WARNING
        - Far beyond (>3x) → INFORMATIONAL
        - None parameters → note "not specified — tested at default range"

        Also adds constraint context to finding descriptions so findings
        reference the user's stated intent (e.g. "You said 20 users").

        When *profile_matches* is provided, findings for dependencies with
        corpus_stats get an additional sentence referencing test-portfolio
        failure rates.
        """
        user_scale = constraints.user_scale
        max_payload = constraints.max_payload_mb

        # Build corpus_stats lookup from profile_matches
        corpus_lookup: dict[str, dict] = {}
        if profile_matches:
            for pm in profile_matches:
                if pm.profile and pm.profile.corpus_stats:
                    corpus_lookup[pm.dependency_name.lower()] = (
                        pm.profile.corpus_stats
                    )

        for finding in report.findings:
            # Skip version/dep info findings — not capacity-related
            if finding.severity == "info" and (
                "outdated" in finding.title.lower()
                or "missing" in finding.title.lower()
                or "unrecognized" in finding.title.lower()
            ):
                continue

            # Try to extract the load level from the scenario name or details
            load_level, is_concurrency = _extract_load_level(finding)

            if load_level is not None and is_concurrency and user_scale is not None:
                # Concurrency metric — compare against user-stated scale
                ratio = load_level / user_scale if user_scale > 0 else float("inf")

                if ratio <= 1.0:
                    # Within stated capacity → CRITICAL
                    finding.severity = "critical"
                    if load_level < user_scale:
                        finding.description = (
                            f"You said {user_scale:,} users. "
                            f"This breaks at just {load_level:,} concurrent "
                            f"users — well below your expected capacity. "
                            f"This is a problem you need to fix before "
                            f"launch. "
                            f"{finding.description}"
                        )
                    else:
                        # Breaks at exactly user's stated capacity
                        finding.description = (
                            f"You said {user_scale:,} users. "
                            f"This breaks at exactly {load_level:,} "
                            f"concurrent users — right at your expected "
                            f"capacity. You have no safety margin. "
                            f"{finding.description}"
                        )
                elif ratio <= 3.0:
                    # Beyond but ≤3x → WARNING
                    finding.severity = "warning"
                    finding.description = (
                        f"You said {user_scale:,} users. "
                        f"This breaks at {load_level:,} concurrent users "
                        f"({ratio:.1f}x your stated needs). "
                        f"Not a problem today, but won't take much growth "
                        f"to hit this. "
                        f"{finding.description}"
                    )
                else:
                    # Far beyond → INFORMATIONAL
                    finding.severity = "info"
                    finding.description = (
                        f"You said {user_scale:,} users. "
                        f"This breaks at {load_level:,} concurrent users "
                        f"— well beyond your stated needs ({ratio:.0f}x). "
                        f"Worth knowing, but not urgent. "
                        f"{finding.description}"
                    )
            elif load_level is not None and not is_concurrency:
                # Data-size metric — display the load level, no ratio
                combined = f"{finding.title} {finding.details}"
                m = _STEP_LEVEL_RE.search(combined)
                step_name = m.group(0).rstrip(": ") if m else ""
                step_desc = _describe_step(step_name) if step_name else ""
                if step_desc:
                    finding.description = (
                        f"This issue occurs at {step_desc}. "
                        f"{finding.description}"
                    )
                else:
                    finding.description = (
                        f"This issue occurs at {load_level:,} operations. "
                        f"{finding.description}"
                    )
            elif user_scale is None and finding.severity in ("critical", "warning"):
                # No user_scale → keep original severity but note it
                finding.description = (
                    f"User scale not specified — tested at default range. "
                    f"{finding.description}"
                )

        # Append corpus stats sentences when available (warning only —
        # critical findings already carry enough urgency)
        if corpus_lookup:
            for finding in report.findings:
                if finding.severity != "warning":
                    continue
                for dep_name in finding.affected_dependencies:
                    stats = corpus_lookup.get(dep_name.lower())
                    if stats and stats.get("tested_count", 0) >= 3:
                        rate = stats["failure_rate"]
                        count = stats["tested_count"]
                        finding.description += (
                            f" In myCode's test portfolio, {dep_name} "
                            f"showed failures in {rate:.0%} of "
                            f"{count} tested projects."
                        )
                        break  # one corpus sentence per finding

        # Add constraint summary to report operational_context
        if constraints.as_summary() != "No specific constraints provided":
            summary = constraints.as_summary()
            if report.operational_context:
                report.operational_context += f"\n\nConstraints: {summary}"
            else:
                report.operational_context = f"Constraints: {summary}"

    # ── Version Discrepancies ──

    def _flag_version_discrepancies(
        self,
        ingestion: IngestionResult,
        profile_matches: list[ProfileMatch],
        report: DiagnosticReport,
    ) -> None:
        """Flag version discrepancies from ingester and profile matches."""
        outdated: list[DependencyInfo] = []
        missing: list[DependencyInfo] = []

        # Detect JS/TS project — npm scoped packages start with @
        is_js = any(d.name.startswith("@") for d in ingestion.dependencies)

        for dep in ingestion.dependencies:
            if dep.is_dev:
                continue
            if dep.is_outdated:
                msg = f"{dep.name}: installed {dep.installed_version}"
                if dep.latest_version:
                    msg += f", latest is {dep.latest_version}"
                report.version_flags.append(msg)
                outdated.append(dep)
            elif dep.is_missing:
                if is_js:
                    # npm packages aren't "missing" — myCode just can't
                    # verify them via Python importlib.  Don't alarm.
                    report.version_flags.append(
                        f"{dep.name}: no stress profile available"
                    )
                else:
                    report.version_flags.append(
                        f"{dep.name}: declared but not installed"
                    )
                missing.append(dep)

        # Consolidated outdated finding
        if outdated:
            details = [
                f"{d.name} ({d.installed_version} \u2192 "
                f"{d.latest_version or 'unknown'})"
                for d in outdated[:5]
            ]
            n = len(outdated)
            desc = ", ".join(details)
            if n > 5:
                desc += f", and {n - 5} more"
            desc += (
                ". Outdated dependencies may have known issues "
                "that affect behavior under stress."
            )
            report.findings.append(Finding(
                title=(
                    f"{n} outdated "
                    f"{'dependency' if n == 1 else 'dependencies'}"
                ),
                severity="info",
                description=desc,
                affected_dependencies=[d.name for d in outdated[:10]],
            ))

        # Consolidated missing/unprofiled finding
        if missing:
            names = [d.name for d in missing[:5]]
            n = len(missing)
            desc = ", ".join(names)
            if n > 5:
                desc += f", and {n - 5} more"
            if is_js:
                desc += (
                    f" {'is' if n == 1 else 'are'} declared in your "
                    "project but don't have myCode stress profiles yet. "
                    "These dependencies were tested with generic patterns "
                    "rather than targeted scenarios."
                )
                report.findings.append(Finding(
                    title=(
                        f"{n} "
                        f"{'dependency' if n == 1 else 'dependencies'} "
                        f"without stress profiles"
                    ),
                    severity="info",
                    description=desc,
                    affected_dependencies=[d.name for d in missing[:10]],
                ))
            else:
                desc += (
                    f" {'is' if n == 1 else 'are'} declared in requirements "
                    "but not installed. This may cause import failures under "
                    "certain code paths."
                )
                report.findings.append(Finding(
                    title=(
                        f"{n} missing "
                        f"{'dependency' if n == 1 else 'dependencies'}"
                    ),
                    severity="warning",
                    description=desc,
                    affected_dependencies=[d.name for d in missing[:10]],
                ))

        # From profile version_match
        for match in profile_matches:
            if match.version_match is False and match.version_notes:
                msg = f"{match.dependency_name}: {match.version_notes}"
                if msg not in report.version_flags:
                    report.version_flags.append(msg)

    # ── HTTP Startup-Failure + Version Discrepancy Enrichment ──

    @staticmethod
    def _enrich_startup_failure_with_version(
        ingestion: IngestionResult,
        report: DiagnosticReport,
    ) -> None:
        """Connect 'server could not start' findings with version discrepancies.

        When an HTTP finding reports a server startup failure and the
        framework dependency has a version discrepancy, the finding is
        enriched with version context and reclassified from 'unclassified'
        to 'dependency_failure' / 'version_incompatibility'.
        """
        # Build lookup: dep name → DependencyInfo for outdated deps
        outdated_deps: dict[str, DependencyInfo] = {}
        for dep in ingestion.dependencies:
            if dep.is_outdated and dep.installed_version and dep.latest_version:
                outdated_deps[dep.name.lower()] = dep

        if not outdated_deps:
            return

        for finding in report.findings:
            if (
                finding.category != "http_load_testing"
                or "could not start" not in finding.title.lower()
            ):
                continue

            # Check affected_dependencies for a matching outdated dep
            for dep_name in finding.affected_dependencies:
                dep = outdated_deps.get(dep_name.lower())
                if dep is None:
                    continue

                # Enrich the finding description
                finding.description += (
                    f" Your {dep.name} version ({dep.installed_version})"
                    f" is significantly behind current"
                    f" ({dep.latest_version}). The startup failure"
                    f" is likely caused by this version gap."
                )
                finding.failure_domain = "dependency_failure"
                finding.failure_pattern = "version_incompatibility"
                break  # one enrichment per finding

    # ── Unrecognized Dependencies ──

    def _flag_unrecognized_deps(
        self,
        profile_matches: list[ProfileMatch],
        report: DiagnosticReport,
    ) -> None:
        """Track dependencies without component library profiles.

        These are noted in the dependency coverage line in the report
        output rather than as alarming "unrecognized" findings.
        """
        for match in profile_matches:
            if match.profile is None:
                report.unrecognized_deps.append(match.dependency_name)

    # ── Summary Generation ──

    def _generate_plain_summary(
        self,
        report: DiagnosticReport,
        operational_intent: str,
        project_name: str = "",
    ) -> str:
        """Generate a plain-language summary for non-technical readers.

        Uses template-based pattern matching on findings, degradation points,
        and the user's operational intent to produce a human-readable overview.
        """
        if not report.scenarios_run:
            return ""

        lines: list[str] = []

        # ── Project reference ──
        # Prefer the auto-generated description over raw user input
        if report.project_description and report.project_description != "Your Project":
            desc = report.project_description
            # Avoid "your Your ..." — description may already start with "Your"
            if desc.lower().startswith("your "):
                project_ref = desc[0].lower() + desc[1:]
            else:
                project_ref = f"your {desc}"
        elif project_name and len(project_name) <= 40:
            project_ref = f"your {project_name}"
        elif operational_intent:
            ref = _extract_project_ref(operational_intent)
            project_ref = f"your {ref}"
        else:
            project_ref = "your project"

        # ── Short reference for bullet points ──
        # Use full description in opening paragraph, short form in bullets
        short_ref = _short_project_ref(report.vertical)

        # ── Overall assessment ──
        critical = [f for f in report.findings if f.severity == "critical"]
        warnings = [f for f in report.findings if f.severity == "warning"]
        completed = report.scenarios_run - len(report.incomplete_tests)

        if completed == 0 and report.incomplete_tests:
            lines.append(
                f"myCode could not execute any stress tests for "
                f"{project_ref}. This is typically caused by missing "
                f"runtime dependencies or an unsupported execution "
                f"environment."
            )
        elif critical:
            lines.append(
                f"We found some problems that could affect "
                f"{project_ref} under real-world conditions."
            )
        elif warnings:
            lines.append(
                f"{project_ref[0].upper()}{project_ref[1:]} mostly "
                f"handles stress well, but there are a few areas to watch."
            )
        else:
            if report.incomplete_tests:
                lines.append(
                    f"{project_ref[0].upper()}{project_ref[1:]} looks "
                    f"solid under the conditions we tested, though "
                    f"{len(report.incomplete_tests)} scenario(s) could "
                    f"not be run."
                )
            else:
                lines.append(
                    f"{project_ref[0].upper()}{project_ref[1:]} looks "
                    f"solid under the conditions we tested."
                )

        # ── Top findings (up to 4, prioritized, one per scenario) ──
        #
        # SIGNIFICANCE FILTER: Only surface findings with real consequences.
        # Small memory (<50MB), stable fast timing (<500ms), zero-error
        # degradation curves are not worth a summary bullet.
        #
        # Priority order: resource cap hits first, then memory degradation,
        # then execution time degradation, then other findings.
        # Degradation points are preferred over findings for the same
        # scenario because they contain richer curve data (start→end
        # values, breaking points).

        _sev_order = {"critical": 0, "warning": 1, "info": 2}
        candidates: list[tuple[int, int, str, str]] = []

        # Degradation points first — they have the richest data
        degradation_scenarios: set[str] = set()
        for dp in report.degradation_points:
            # Significance filter: skip degradation that doesn't matter
            if not _is_significant_degradation(dp):
                continue
            translated = self._translate_degradation(
                dp, short_ref, user_scale=report.user_scale,
            )
            degradation_scenarios.add(dp.scenario_name)
            if dp.metric == "memory_peak_mb":
                prio = 1
            elif dp.metric == "error_count":
                prio = 1
            else:
                prio = 2
            candidates.append((prio, 0, dp.scenario_name, translated))

        # Critical/warning findings — only for scenarios without
        # degradation points (which already have better data)
        for f in report.findings:
            if f.severity == "info":
                continue
            # Significance filter: skip findings without real consequences
            if not _is_significant_finding(f):
                continue
            scenario_name = (
                f.title.split(": ", 1)[-1] if ": " in f.title else ""
            )
            if scenario_name in degradation_scenarios:
                continue
            translated = self._translate_finding(f, short_ref)
            if translated:
                sev_rank = _sev_order.get(f.severity, 9)
                if "resource" in f.title.lower() or f.category == "edge_case_input":
                    prio = 0
                elif f.severity == "critical":
                    prio = 1
                else:
                    prio = 2
                candidates.append((prio, sev_rank, scenario_name, translated))

        # Sort by priority, then severity rank; pick top 4 (one per scenario)
        candidates.sort(key=lambda c: (c[0], c[1]))
        items: list[str] = []
        seen_scenarios: set[str] = set()
        seen_text: set[str] = set()
        for _prio, _sev, scenario_name, translated in candidates:
            if len(items) >= 4:
                break
            if scenario_name in seen_scenarios:
                continue
            if translated in seen_text:
                continue
            seen_scenarios.add(scenario_name)
            seen_text.add(translated)
            items.append(translated)

        if items:
            lines.append("")
            for item in items:
                lines.append(f"- {item}")

        # ── Closing line (clean runs only) ──
        if not items:
            lines.append("")
            lines.append(
                "No issues were found under the conditions we tested."
            )

        return "\n".join(lines)

    # ── Gemini Flash Plain Summary ──

    _GEMINI_ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )

    def _generate_llm_plain_summary(
        self,
        report: DiagnosticReport,
        operational_intent: str,
        project_name: str = "",
    ) -> Optional[str]:
        """Enhance the plain summary via a single Gemini Flash call.

        Returns the LLM-generated summary text, or None on any failure
        (in which case the caller keeps the template summary).
        """
        if not self._llm_config or not self._llm_config.api_key:
            return None

        # ── Build the safe payload (no code, no paths, no traces) ──
        context_parts: list[str] = []

        if project_name:
            context_parts.append(f"Project: {project_name}")
        if operational_intent:
            context_parts.append(f"Description: {operational_intent}")

        # Top 5 findings — only safe fields
        if report.findings:
            context_parts.append("\nFindings:")
            for f in report.findings[:5]:
                context_parts.append(
                    f"- [{f.severity}] {f.category}: {f.description}"
                )

        # Top 5 degradation curves — start/end values only
        if report.degradation_points:
            context_parts.append("\nDegradation:")
            for dp in report.degradation_points[:5]:
                start = dp.steps[0][1] if dp.steps else 0
                end = dp.steps[-1][1] if dp.steps else 0
                bp = f", breaking at {dp.breaking_point}" if dp.breaking_point else ""
                context_parts.append(
                    f"- {dp.scenario_name} ({dp.metric}): "
                    f"{start:.2f} → {end:.2f}{bp}"
                )

        prompt = (
            "You are writing a summary for myCode, a stress-testing tool "
            "for AI-generated code. A non-technical user just ran stress "
            "tests on their project.\n\n"
            "Write 3-5 bullet points explaining what these findings mean "
            "for this specific project, in terms a non-technical user would "
            "understand.\n\n"
            "RULES:\n"
            "- Each bullet: state the problem, explain why it matters for "
            "the user, optionally note when it starts.\n"
            "- Connect numbers to user context: 'Memory reached 72MB. "
            "With your expected 2,983 users, your server will likely run "
            "out of memory.'\n"
            "- Use 'your app' in bullets, not the full project description.\n"
            "- Do not suggest fixes or code changes.\n"
            "- Do not use engineering jargon.\n"
            "- Each bullet should start with '- '.\n"
            "- Do not end with a closing line or boilerplate.\n\n"
            + "\n".join(context_parts)
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
        }

        url = f"{self._GEMINI_ENDPOINT}?key={self._llm_config.api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip()
            if not text:
                return None
            return text
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            logger.debug(
                "Gemini plain summary HTTP %s: %s", exc.code, error_body[:500],
            )
            return None
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
        ):
            logger.debug("Gemini plain summary call failed, using template", exc_info=True)
            return None

    def _translate_degradation(
        self, dp: DegradationPoint, project_ref: str,
        user_scale: int | None = None,
    ) -> str:
        """Translate a degradation point into a plain-language summary bullet.

        Every bullet answers three questions:
        1. What activity caused it (from scenario name)
        2. What happened (metric values)
        3. Why it matters (consequence for the user)

        Threshold details belong in degradation curves, not summary.
        """
        first_val = dp.steps[0][1] if dp.steps else 0.0
        last_val = dp.steps[-1][1] if dp.steps else 0.0
        activity = _describe_scenario(dp.scenario_name)
        impact = _describe_impact(
            dp.metric, first_val, last_val, user_scale, activity=activity,
        )
        return f"{impact[0].upper()}{impact[1:]}."

    def _translate_finding(
        self, f: Finding, project_ref: str,
    ) -> str:
        """Translate a finding into plain language using actual metric data.

        Every bullet must answer: what was tested, what happened, why it
        matters.  ``project_ref`` is woven into the description (e.g.
        "your incident matching system").
        """
        # Extract scenario name from finding title
        scenario_name = f.title.split(": ", 1)[-1] if ": " in f.title else ""
        activity = _describe_scenario(scenario_name)
        if activity:
            activity = f"{project_ref} is {activity}"

        if f.category == "concurrent_execution":
            ctx = activity or f"{project_ref} is handling multiple users"
            if "failed" in f.title.lower():
                return f"When {ctx}, the system fails."
            parts = []
            if f._execution_time_ms > 0:
                parts.append(
                    f"response time hit {_format_ms(f._execution_time_ms)}"
                )
            if f._peak_memory_mb > 0:
                parts.append(f"memory reached {f._peak_memory_mb:.0f}MB")
            if f._error_count > 0:
                parts.append(_describe_errors_with_context(f, scenario_name))
            if parts:
                return f"When {ctx}, {' and '.join(parts)}."
            return f"When {ctx}, the system struggled under load."

        if f.category == "edge_case_input":
            cap_type = _extract_cap_type(f)
            if cap_type:
                return (
                    f"When {project_ref} receives unexpected input, "
                    f"{cap_type}."
                )
            return (
                f"When {project_ref} receives unexpected or unusual "
                f"input, the code crashes instead of handling it gracefully."
            )

        if f.category == "data_volume_scaling":
            ctx = activity or f"{project_ref} is processing larger data"
            if "resource" in f.title.lower():
                cap_type = _extract_cap_type(f)
                if cap_type:
                    return f"When {ctx}, {cap_type}."
                return f"When {ctx}, the system hits safety limits."
            if f._error_count > 0:
                return (
                    f"When {ctx}, "
                    f"{_describe_errors_with_context(f, scenario_name)} "
                    f"at the highest load."
                )
            parts = []
            if f._execution_time_ms > 0:
                parts.append(
                    f"response time hit {_format_ms(f._execution_time_ms)}"
                )
            if f._peak_memory_mb > 0:
                parts.append(f"memory reached {f._peak_memory_mb:.0f}MB")
            if parts:
                return f"When {ctx}, {' and '.join(parts)}."
            return f"When {ctx}, performance degrades under load."

        if f.category == "memory_profiling":
            ctx = activity or "extended use"
            if f._peak_memory_mb > 0:
                sessions = max(1, int(2048 / f._peak_memory_mb))
                return (
                    f"During {ctx}, memory grew to "
                    f"{f._peak_memory_mb:.0f}MB and keeps growing. "
                    f"On a 2GB server, this limits you to approximately "
                    f"{sessions} concurrent sessions before running out "
                    f"of memory."
                )
            return (
                f"During {ctx}, memory grows without limit. "
                f"Your server will eventually run out of memory."
            )

        # Default: use actual metrics if available
        if activity:
            parts = []
            if f._execution_time_ms > 0:
                parts.append(
                    f"response time hit {_format_ms(f._execution_time_ms)}"
                )
            if f._peak_memory_mb > 0:
                parts.append(f"memory reached {f._peak_memory_mb:.0f}MB")
            if f._error_count > 0:
                parts.append(_describe_errors_with_context(f, scenario_name))
            if parts:
                return f"When {activity}, {' and '.join(parts)}."
            return f"When {activity}, issues were found."
        return f.description

    def _generate_offline_summary(self, report: DiagnosticReport) -> str:
        """Generate a plain-text summary without LLM."""
        parts: list[str] = []

        # Overall assessment
        critical = [f for f in report.findings if f.severity == "critical"]
        warnings = [f for f in report.findings if f.severity == "warning"]

        if not report.scenarios_run:
            return "No stress test scenarios were executed."

        # All scenarios failed to execute (crashed/incomplete)
        completed = report.scenarios_run - len(report.incomplete_tests)
        if completed == 0 and report.incomplete_tests:
            parts.append(
                f"myCode could not execute any of the "
                f"{report.scenarios_run} stress test scenarios for this "
                f"project. This is typically caused by missing runtime "
                f"dependencies or an unsupported execution environment."
            )
        elif not critical and not warnings:
            if report.incomplete_tests:
                parts.append(
                    f"{completed} of {report.scenarios_run} stress test "
                    f"scenarios completed without issues. "
                    f"{len(report.incomplete_tests)} could not be run."
                )
            else:
                parts.append(
                    f"All {report.scenarios_run} stress test scenarios "
                    f"completed without issues."
                )
        elif critical:
            parts.append(
                f"Found {len(critical)} critical issue(s) and "
                f"{len(warnings)} warning(s) across "
                f"{report.scenarios_run} scenarios."
            )
        else:
            parts.append(
                f"Found {len(warnings)} warning(s) across "
                f"{report.scenarios_run} scenarios. No critical issues."
            )

        # Degradation summary
        if report.degradation_points:
            breaking = [
                dp for dp in report.degradation_points if dp.breaking_point
            ]
            if breaking:
                parts.append(
                    f"Performance degradation detected in "
                    f"{len(breaking)} area(s)."
                )

        # Version flags
        if report.version_flags:
            parts.append(
                f"{len(report.version_flags)} version discrepancy(ies) noted."
            )

        # Context — prefer structured constraint fields over raw input
        ctx_parts: list[str] = []
        if report.user_scale is not None:
            ctx_parts.append(f"{report.user_scale:,} concurrent users")
        if report._data_type:
            if report._data_type_detail:
                # Use the user's original file-type phrasing
                ctx_parts.append(f"handling {report._data_type_detail}")
            else:
                _data_labels = {
                    "tabular": "tabular data",
                    "text": "text and documents",
                    "images": "images and media",
                    "api_responses": "API responses",
                    "mixed": "mixed data types",
                }
                ctx_parts.append(
                    f"handling {_data_labels.get(report._data_type, report._data_type)}"
                )
        if report._usage_pattern:
            _usage_labels = {
                "sustained": "steady usage",
                "burst": "burst/peak usage patterns",
                "periodic": "occasional use",
                "growing": "growing usage over time",
            }
            ctx_parts.append(
                _usage_labels.get(report._usage_pattern, report._usage_pattern)
            )
        if report._analysis_depth:
            _depth_labels = {
                "quick": "quick scan",
                "standard": "standard analysis",
                "deep": "deep analysis",
            }
            ctx_parts.append(
                _depth_labels.get(report._analysis_depth, report._analysis_depth)
            )
        elif report._timeout_per_scenario:
            ctx_parts.append(f"{report._timeout_per_scenario}s per test")
        if ctx_parts:
            parts.append(
                f"Results assessed relative to: {', '.join(ctx_parts)}."
            )
        elif report.operational_context:
            # Fall back to raw context (strip confirmation phrases)
            ctx = report.operational_context
            _CONFIRMATIONS = (
                "sounds right", "sounds good", "looks right", "looks good",
                "yes", "yeah", "yep", "correct", "that's right",
                "thats right",
            )
            cleaned = ctx
            for phrase in _CONFIRMATIONS:
                if cleaned.lower().startswith(phrase):
                    cleaned = cleaned[len(phrase):].lstrip(" .,;-—")
                    break
            if cleaned and len(cleaned) > 5:
                parts.append(
                    f"Results assessed relative to: {cleaned}"
                )

        return " ".join(parts)

    def _generate_llm_summary(
        self,
        execution: ExecutionEngineResult,
        ingestion: IngestionResult,
        report: DiagnosticReport,
        operational_intent: str,
    ) -> str:
        """Generate a narrative summary using the LLM."""
        system = (
            "You are the Report Generator for myCode, a stress-testing tool. "
            "Write a plain-language diagnostic summary for a non-technical user.\n\n"
            "RULES:\n"
            "1. Write in clear, simple language. No engineering jargon.\n"
            "2. DIAGNOSE ONLY. Never suggest fixes, patches, or code changes.\n"
            "3. Frame findings relative to the user's stated purpose.\n"
            "4. Be direct about what broke and where. If nothing broke, "
            "say so clearly and note what was tested.\n"
            "5. Describe degradation curves in terms the user understands "
            "(e.g. 'response time doubled when data exceeded 10,000 rows').\n"
            "6. Keep it under 500 words.\n\n"
            "Respond with JSON: {\"summary\": \"your summary here\"}"
        )

        user_parts: list[str] = []

        if operational_intent:
            user_parts.append(f"## User's Purpose\n{operational_intent}")

        incomplete_line = ""
        if report.scenarios_incomplete:
            incomplete_line = (
                f"\nCould not test: {report.scenarios_incomplete}"
            )
        user_parts.append(
            f"\n## Test Results\n"
            f"Scenarios run: {report.scenarios_run}\n"
            f"Passed: {report.scenarios_passed}\n"
            f"Failed: {report.scenarios_failed}"
            f"{incomplete_line}\n"
            f"Total errors: {report.total_errors}"
        )

        if report.findings:
            user_parts.append("\n## Key Findings")
            for f in report.findings[:15]:
                user_parts.append(
                    f"- [{f.severity}] {f.title}: {f.description}"
                )

        if report.degradation_points:
            user_parts.append("\n## Degradation Curves")
            for dp in report.degradation_points[:10]:
                user_parts.append(f"- {dp.description}")
                if dp.breaking_point:
                    user_parts.append(
                        f"  Breaking point: {dp.breaking_point}"
                    )

        if report.version_flags:
            user_parts.append(
                f"\n## Version Issues\n"
                + "\n".join(f"- {v}" for v in report.version_flags)
            )

        if report.unrecognized_deps:
            user_parts.append(
                f"\n## Dependencies Without Stress Profiles\n"
                + ", ".join(report.unrecognized_deps[:10])
            )

        # Include project structure context
        user_parts.append(
            f"\n## Project Context\n"
            f"Files: {ingestion.files_analyzed}, "
            f"Lines: {ingestion.total_lines}, "
            f"Dependencies: {len(ingestion.dependencies)}"
        )

        user = "\n".join(user_parts)

        # Call LLM
        fallback = self._generate_offline_summary(report)
        if self._backend is None:
            return fallback

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self._backend.generate(messages)
            report.token_usage = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
            report.model_used = response.model or self._llm_config.model

            # Extract summary from JSON response
            return self._extract_summary(response.content, fallback)
        except LLMError as exc:
            logger.warning("LLM summary failed, using offline: %s", exc)
            return fallback

    @staticmethod
    def _extract_summary(content: str, fallback: str) -> str:
        """Extract summary string from LLM JSON response."""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "summary" in data:
                return data["summary"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try brace extraction
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
            if isinstance(data, dict) and "summary" in data:
                return data["summary"]
        except (ValueError, json.JSONDecodeError):
            pass

        # If LLM returned plain text (no JSON), use it directly if reasonable
        stripped = content.strip()
        if 20 < len(stripped) < 3000 and "{" not in stripped[:5]:
            return stripped

        return fallback

    # ── Helpers ──

    @staticmethod
    def _deps_from_name(scenario_name: str) -> list[str]:
        """Infer dependency names from a scenario name.

        Scenario names from offline generation follow the pattern
        ``{dep}_{template_name}``.  This is best-effort.
        """
        # Common dependency names that might prefix scenario names
        parts = scenario_name.split("_", 1)
        if parts[0] and parts[0] not in (
            "coupling", "unrecognized", "generic", "edge", "version",
        ):
            return [parts[0]]
        return []

    @staticmethod
    def _max_load_from_steps(sr: ScenarioResult) -> Optional[int]:
        """Extract the highest load level reached across all steps.

        Parses the trailing ``_N`` from any step name (e.g.
        ``concurrent_50``, ``compute_10000``).  Returns the maximum N
        across all steps, or ``None`` if no numeric suffix found.
        """
        max_level: Optional[int] = None
        for step in sr.steps:
            level = _step_level(step.step_name)
            if level is not None and (max_level is None or level > max_level):
                max_level = level
        return max_level

    @staticmethod
    def _first_failing_load(sr: ScenarioResult) -> Optional[int]:
        """Find the load level of the first step that failed or errored.

        Returns the numeric suffix from the step name if the step had
        errors or hit a resource cap.  Returns ``None`` if no failing
        step has a parseable load level.
        """
        for step in sr.steps:
            if step.error_count > 0 or step.resource_cap_hit:
                level = _step_level(step.step_name)
                if level is not None:
                    return level
        return None

    @staticmethod
    def _load_level_detail(sr: ScenarioResult) -> str:
        """Build a details suffix with load level info from step names.

        Returns a string like ``" at api_concurrency_50 (50 concurrent API calls)"``
        or an empty string if no load level can be extracted.
        """
        def _fmt(step_name: str) -> str:
            desc = _describe_step(step_name)
            if desc:
                return f" at {desc}"
            level = _step_level(step_name)
            if level is not None:
                return f" at {level:,} operations"
            return ""

        # Prefer the first failing step's load level
        for step in sr.steps:
            if step.error_count > 0 or step.resource_cap_hit:
                detail = _fmt(step.step_name)
                if detail:
                    return detail

        # Fallback: highest load level reached
        max_level = None
        max_step = ""
        for step in sr.steps:
            level = _step_level(step.step_name)
            if level is not None and (max_level is None or level > max_level):
                max_level = level
                max_step = step.step_name
        if max_step:
            return _fmt(max_step)
        return ""

    @staticmethod
    def _summarize_errors(sr: ScenarioResult) -> str:
        """Summarize errors from a scenario result in plain language."""
        error_types: dict[str, int] = {}
        for step in sr.steps:
            for err in step.errors:
                etype = err.get("type", "Unknown") if isinstance(err, dict) else "Unknown"
                error_types[etype] = error_types.get(etype, 0) + 1

        if error_types:
            parts: list[str] = []
            for etype, count in list(error_types.items())[:5]:
                label = _human_error_type(etype)
                if count == 1:
                    parts.append(f"1 {label}")
                else:
                    parts.append(f"{count} {label}s")
            return ", ".join(parts)
        if sr.total_errors == 1:
            return "1 error"
        return f"{sr.total_errors} errors"

    @staticmethod
    def _summarize_cap_hits(sr: ScenarioResult) -> str:
        """Summarize which resource caps were hit."""
        caps = set()
        for step in sr.steps:
            if step.resource_cap_hit:
                caps.add(step.resource_cap_hit)
        return f"Caps hit: {', '.join(caps)}" if caps else "Resource cap hit"

    @staticmethod
    def _finding_pattern(finding: Finding) -> str:
        """Extract a groupable pattern from a finding's type."""
        if finding._finding_type:
            return finding._finding_type
        # Legacy fallback: parse from title prefix (for tests / older data)
        title = finding.title
        if title.startswith("Scenario failed:"):
            return "scenario_failed"
        if title.startswith("Resource limit hit:"):
            return "resource_limit_hit"
        if title.startswith("Errors during:"):
            return "errors_during"
        if title.startswith("Failure indicators triggered:"):
            return "failure_indicators"
        return title

    @staticmethod
    def _metrics_similar(a: Finding, b: Finding, tolerance: float = 0.10) -> bool:
        """Check if two findings have similar metrics (within ±tolerance).

        Compares _peak_memory_mb, _execution_time_ms, _error_count.
        All must be within tolerance. Zero pairs are always similar.
        """
        for va, vb in [
            (a._peak_memory_mb, b._peak_memory_mb),
            (a._execution_time_ms, b._execution_time_ms),
            (float(a._error_count), float(b._error_count)),
        ]:
            if va == 0.0 and vb == 0.0:
                continue
            denominator = max(abs(va), abs(vb))
            if denominator == 0:
                continue
            if abs(va - vb) / denominator > tolerance:
                return False
        return True

    @staticmethod
    def _group_similar_findings(findings: list[Finding]) -> list[Finding]:
        """Group findings with the same category/pattern and similar metrics.

        Uses anchor-based clustering: the first finding in each
        (category, pattern) group becomes the anchor. Subsequent findings
        that are metrically similar join the cluster. Non-matching ones
        start new clusters.

        Returns a flat list of representative findings (with grouped_findings
        and group_count set) plus any singletons.
        """
        from collections import defaultdict

        _sev_order = {"critical": 0, "warning": 1, "info": 2}

        # ── Pre-pass: deduplicate findings with the same title at
        # different severity levels ──
        # A single scenario can produce findings at different severity levels
        # (e.g. resource_cap_hit → CRITICAL and errors → WARNING). Keep only
        # the highest severity per title. Findings at the *same* severity
        # are left alone — they represent different aspects (errors vs.
        # failure indicators) and will be grouped later.
        by_title: dict[str, list[Finding]] = {}
        for f in findings:
            by_title.setdefault(f.title, []).append(f)
        deduped: list[Finding] = []
        for _title, group in by_title.items():
            if len(group) == 1:
                deduped.append(group[0])
                continue
            # Check if there are mixed severities
            severities = {f.severity for f in group}
            if len(severities) > 1:
                # Keep only the highest severity
                group.sort(key=lambda f: _sev_order.get(f.severity, 9))
                deduped.append(group[0])
            else:
                # Same severity — keep all (different finding types)
                deduped.extend(group)
        findings = deduped

        # Group by (category, pattern, primary_dep) so findings from
        # different dependency domains are never grouped together.
        buckets: dict[tuple[str, str, str], list[Finding]] = defaultdict(list)
        for f in findings:
            pattern = ReportGenerator._finding_pattern(f)
            primary_dep = f.affected_dependencies[0] if f.affected_dependencies else ""
            key = (f.category, pattern, primary_dep)
            buckets[key].append(f)

        result: list[Finding] = []
        for _key, bucket in buckets.items():
            if len(bucket) == 1:
                result.append(bucket[0])
                continue

            # Anchor-based clustering
            clusters: list[list[Finding]] = []
            for f in bucket:
                placed = False
                for cluster in clusters:
                    anchor = cluster[0]
                    if ReportGenerator._metrics_similar(anchor, f):
                        cluster.append(f)
                        placed = True
                        break
                if not placed:
                    clusters.append([f])

            for cluster in clusters:
                # Promote the highest-severity finding to representative
                # so that a critical finding is never hidden behind a
                # warning or info finding that happened to enter first.
                cluster.sort(key=lambda f: _sev_order.get(f.severity, 9))
                if len(cluster) == 1:
                    result.append(cluster[0])
                else:
                    representative = cluster[0]
                    representative.grouped_findings = cluster[1:]
                    representative.group_count = len(cluster)
                    result.append(representative)

        return result

    @staticmethod
    def _fold_memory_degradation_into_findings(
        findings: list[Finding],
        degradation_points: list[DegradationPoint],
    ) -> list[DegradationPoint]:
        """Remove memory degradation points that duplicate an existing finding.

        When a scenario produces both a Finding (error/resource cap) and a
        DegradationPoint for ``memory_peak_mb``, the report shows the same
        component's memory behavior twice.  This folds the degradation
        description into the finding and suppresses the duplicate point.

        Returns the filtered list of degradation points.
        """
        # Build a lookup: scenario_name → Finding for findings whose
        # category or title indicates memory involvement.
        finding_by_scenario: dict[str, Finding] = {}
        for f in findings:
            # Extract scenario name from title patterns like
            # "Errors during: <scenario>" or "Resource limit hit: <scenario>"
            parts = f.title.split(": ", 1)
            if len(parts) == 2:
                scenario_name = parts[1]
            else:
                scenario_name = f.title
            finding_by_scenario[scenario_name] = f

        kept: list[DegradationPoint] = []
        for dp in degradation_points:
            if dp.metric != "memory_peak_mb":
                kept.append(dp)
                continue
            matched = finding_by_scenario.get(dp.scenario_name)
            if matched is None:
                kept.append(dp)
                continue
            # Fold degradation description into the finding
            if dp.description:
                if matched.description:
                    matched.description += f" {dp.description}"
                else:
                    matched.description = dp.description
        return kept

    @staticmethod
    def _degradation_ratio(dp: DegradationPoint) -> float:
        """Compute first-to-last ratio from a degradation point's steps."""
        if len(dp.steps) < 2:
            return 1.0
        first = dp.steps[0][1]
        last = dp.steps[-1][1]
        if first < 0.001:
            return last  # treat near-zero baseline as the raw last value
        return last / first

    @staticmethod
    def _group_similar_degradation_points(
        points: list[DegradationPoint],
        tolerance: float = 0.10,
    ) -> list[DegradationPoint]:
        """Group degradation points with the same metric and similar ratio.

        Uses anchor-based clustering: within each metric group, the first
        point becomes the anchor. Subsequent points whose degradation ratio
        is within ±tolerance of the anchor join the cluster.

        Returns a flat list of representatives (with grouped_points and
        group_count set) plus singletons.
        """
        from collections import defaultdict

        # Group by metric
        buckets: dict[str, list[DegradationPoint]] = defaultdict(list)
        for dp in points:
            buckets[dp.metric].append(dp)

        result: list[DegradationPoint] = []
        for _metric, bucket in buckets.items():
            if len(bucket) == 1:
                result.append(bucket[0])
                continue

            # Anchor-based clustering by ratio similarity
            clusters: list[list[DegradationPoint]] = []
            for dp in bucket:
                ratio = ReportGenerator._degradation_ratio(dp)
                placed = False
                for cluster in clusters:
                    anchor_ratio = ReportGenerator._degradation_ratio(cluster[0])
                    # Both near-zero or both similar
                    if anchor_ratio == 0.0 and ratio == 0.0:
                        cluster.append(dp)
                        placed = True
                        break
                    denom = max(abs(anchor_ratio), abs(ratio))
                    if denom > 0 and abs(anchor_ratio - ratio) / denom <= tolerance:
                        cluster.append(dp)
                        placed = True
                        break
                if not placed:
                    clusters.append([dp])

            for cluster in clusters:
                if len(cluster) == 1:
                    result.append(cluster[0])
                else:
                    rep = cluster[0]
                    rep.grouped_points = cluster[1:]
                    rep.group_count = len(cluster)
                    result.append(rep)

        return result


# ── Project Description & Confidence ──


_VERTICAL_LABELS: dict[str, str] = {
    "web_app": "web application",
    "data_pipeline": "data pipeline",
    "chatbot": "AI chatbot",
    "dashboard": "dashboard",
    "api_service": "API service",
    "ml_model": "machine learning project",
    "portfolio": "portfolio site",
    "utility": "project",
    "cli_tool": "command-line tool",
    "automation": "automation script",
}


def _short_project_ref(vertical: str) -> str:
    """Return a brief project reference for repeated use in bullets.

    Uses just the vertical label (e.g. "your dashboard") instead of the
    full description ("your dashboard built with flask, fastapi, and uvicorn").
    """
    label = _VERTICAL_LABELS.get(vertical, "")
    if label and label != "project":
        return f"your {label}"
    return "your app"


_FRAMEWORK_LABELS: dict[str, str] = {
    "react": "React",
    "react-dom": "React",
    "vue": "Vue",
    "angular": "Angular",
    "@angular/core": "Angular",
    "svelte": "Svelte",
    "next": "Next.js",
    "nextjs": "Next.js",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "FastAPI",
    "express": "Express",
    "streamlit": "Streamlit",
    "gradio": "Gradio",
    "dash": "Dash",
    "langchain": "LangChain",
    "llama-index": "LlamaIndex",
    "tensorflow": "TensorFlow",
    "torch": "PyTorch",
    "pytorch": "PyTorch",
    "sklearn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "pandas": "pandas",
}


def _detect_primary_framework(deps: list[str]) -> str:
    """Detect the primary framework from dependency names.

    Returns a human-readable framework label (e.g. "React", "Flask")
    or empty string if no known framework is found.
    """
    for dep in deps:
        label = _FRAMEWORK_LABELS.get(dep.lower())
        if label:
            return label
    return ""


def _generate_project_description(
    ingestion: IngestionResult,
    project_name: str = "",
    vertical: str = "",
    architectural_pattern: str = "",
) -> str:
    """Generate a short project description from framework + project name.

    Preferred format: "Your React web application (react-shopping-cart)"
    Fallback chain:
      1. Framework + vertical + project name
      2. Framework + vertical (no project name)
      3. Vertical + project name
      4. Vertical only
      5. "Your project built with [deps]" (last resort)
    """
    human_label = _VERTICAL_LABELS.get(vertical, "")
    non_dev_deps = [d.name for d in ingestion.dependencies if not d.is_dev]
    framework = _detect_primary_framework(non_dev_deps)

    # Build the core description
    if framework and human_label:
        desc = f"Your {framework} {human_label}"
    elif framework:
        desc = f"Your {framework} project"
    elif human_label:
        desc = f"Your {human_label}"
    elif non_dev_deps:
        top = non_dev_deps[:3]
        if len(top) == 1:
            dep_str = top[0]
        elif len(top) == 2:
            dep_str = f"{top[0]} and {top[1]}"
        else:
            dep_str = f"{top[0]}, {top[1]}, and {top[2]}"
        desc = f"Your project built with {dep_str}"
    else:
        desc = "your project"

    # Append project name in parentheses if available and distinct
    if project_name:
        # Avoid redundancy: don't append if the project name is generic
        # or already contained in the description
        name_lower = project_name.lower().replace(" ", "-")
        desc_lower = desc.lower()
        generic_names = {"project", "my project", "app", "my app", "untitled"}
        if (
            name_lower not in generic_names
            and project_name.lower() not in desc_lower
        ):
            desc = f"{desc} ({project_name})"

    return desc


_DATA_TYPE_HUMAN_LABELS: dict[str, str] = {
    "text": "text and documents",
    "tabular": "tabular data",
    "images": "images and media",
    "api_responses": "API responses",
}

_DATA_TYPE_FOCUS_LABELS: dict[str, str] = {
    "text": "file I/O, encoding, and data volume",
    "tabular": "data volume and memory profiling",
    "images": "memory ceiling and concurrency",
    "api_responses": "concurrency and I/O",
}


def _build_data_type_note(data_type: str, detail: str) -> str:
    """Build a transparency note about file-type scenario prioritisation."""
    label = detail or _DATA_TYPE_HUMAN_LABELS.get(data_type, data_type)
    focus = _DATA_TYPE_FOCUS_LABELS.get(data_type, "")
    if focus:
        note = (
            f"Based on your data type ({label}), myCode prioritised "
            f"{focus} scenarios. Format-specific stress testing "
            f"(malformed inputs, encoding edge cases) is planned for v2."
        )
    else:
        note = (
            f"Based on your data type ({label}), myCode ran all scenario "
            f"categories. Format-specific stress testing is planned for v2."
        )
    return note


def _build_confidence_note(
    ingestion: IngestionResult,
    profile_matches: list["ProfileMatch"],
) -> str:
    """Build a confidence note about sandbox limitations.

    Returns a note like "2 of 7 dependencies could not be installed..."
    or empty string if everything looks good.
    """
    # Count dependencies that failed to install (missing in environment)
    missing_deps = [d for d in ingestion.dependencies if d.is_missing and not d.is_dev]
    total_deps = [d for d in ingestion.dependencies if not d.is_dev]

    if not missing_deps:
        return ""

    installed_names = [
        d.name for d in total_deps if not d.is_missing
    ]
    missing_names = [d.name for d in missing_deps]

    parts: list[str] = []
    parts.append(
        f"{len(missing_deps)} of {len(total_deps)} dependencies "
        f"could not be installed in the test environment."
    )
    if installed_names:
        shown = installed_names[:4]
        parts.append(
            f"Findings involving {', '.join(shown)} are fully tested."
        )
    if missing_names:
        shown = missing_names[:3]
        parts.append(
            f"Findings involving {', '.join(shown)} may be incomplete."
        )
    return " ".join(parts)


def _consequence_for_category(category: str, scenario_name: str = "") -> str:
    """Return a "so what?" consequence for a finding based on its category.

    Translates technical categories into user-meaningful impact statements.
    """
    name_lower = scenario_name.lower()
    _CONSEQUENCES: dict[str, str] = {
        "concurrent_execution": (
            "In practice, this means your app will become unusable "
            "when multiple people use it at the same time."
        ),
        "data_volume_scaling": (
            "This means your app will slow down or crash "
            "when handling larger amounts of data."
        ),
        "memory_profiling": (
            "This means your app will use increasingly more memory "
            "over time, eventually crashing or becoming unresponsive."
        ),
        "edge_case_input": (
            "This means unexpected input from users could crash your app "
            "instead of showing an error message."
        ),
        "blocking_io": (
            "This means your app will freeze or become very slow "
            "when waiting for external services while handling requests."
        ),
        "gil_contention": (
            "This means CPU-heavy operations will bottleneck "
            "when running alongside other tasks."
        ),
        "async_failures": (
            "This means async operations may fail silently or pile up "
            "under real-world load."
        ),
        "event_listener_accumulation": (
            "This means your app will slowly leak memory "
            "during long user sessions."
        ),
        "state_management_degradation": (
            "This means your app's state will become corrupted or slow "
            "during extended use."
        ),
    }
    consequence = _CONSEQUENCES.get(category, "")

    # Coupling-specific overrides
    if "coupling" in name_lower:
        if "api" in name_lower:
            return (
                "This means connected components may fail together "
                "when external services are slow or unavailable."
            )
        if "state" in name_lower or "render" in name_lower:
            return (
                "This means shared state between components could become "
                "inconsistent under load."
            )
    return consequence


# ── Module Helpers ──


def _human_error_type(etype: str) -> str:
    """Convert a Python/JS exception type name to plain language.

    ``MemoryError`` → ``out-of-memory error``,
    ``TimeoutError`` → ``timeout``,
    ``RuntimeError`` → ``runtime error``.
    """
    special = {
        "MemoryError": "out-of-memory error",
        "TimeoutError": "timeout",
        "Unknown": "error",
    }
    if etype in special:
        return special[etype]
    # CamelCase → space-separated lowercase: "RuntimeError" → "runtime error"
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", etype).lower()


def _humanize_trigger(trigger_name: str) -> str:
    """Translate a failure indicator trigger name to plain language.

    ``memory_growth_unbounded`` → ``memory usage grows without limit``.
    """
    _TRIGGER_MAP: dict[str, str] = {
        "memory_growth_unbounded": "memory usage grows without limit",
        "memory_growth_linear": "memory usage grows steadily with load",
        "memory_growth_exponential": "memory usage grows rapidly with load",
        "memory_spike": "sudden memory spike detected",
        "timeout_cascade": "timeouts cascade across operations",
        "error_rate_increasing": "error rate increases with load",
        "throughput_collapse": "throughput drops sharply under load",
        "latency_spike": "response time spikes suddenly",
        "connection_exhaustion": "connections are exhausted under load",
        "deadlock_detected": "potential deadlock detected",
        "resource_leak": "resources are not being released",
        "cpu_saturation": "CPU is fully saturated",
    }
    if trigger_name in _TRIGGER_MAP:
        return _TRIGGER_MAP[trigger_name]
    # Fallback: replace underscores with spaces
    return trigger_name.replace("_", " ")


def _human_metric(metric: str) -> str:
    """Convert a metric name to human-readable text."""
    return {
        "execution_time_ms": "Execution time",
        "memory_peak_mb": "Memory usage",
        "error_count": "Error count",
        "memory_mb": "Memory usage",
        "memory_growth_mb": "Memory growth",
        "throughput": "Throughput",
        "latency_p99_ms": "P99 latency",
    }.get(metric, metric.replace("_", " ").title())


# ── Plain Summary Helpers ──

# Maps profile template names to user-meaningful activity descriptions.
# Scenario names follow the pattern {dep}_{template_name}.
_TEMPLATE_DESCRIPTIONS: dict[str, str] = {
    "concurrent_request_load": "handling multiple users at once",
    "concurrent_session_load": "handling multiple users at once",
    "large_payload_response": "returning large results",
    "file_upload_scaling": "handling file uploads",
    "file_upload_memory_stress": "handling file uploads",
    "blocking_io_under_load": "handling requests while waiting for data",
    "repeated_request_memory_profile": "handling many requests over time",
    "script_rerun_cost": "re-running with larger data",
    "cache_memory_growth": "caching data over time",
    "repeated_interaction_memory_profile": "extended user sessions",
    "large_download_memory": "downloading large responses",
    "timeout_behavior": "calling external APIs that respond slowly",
    "error_handling_resilience": "dealing with unexpected API responses",
    "session_vs_individual_performance": "making many API calls",
    "data_volume_scaling": "processing larger amounts of data",
    "merge_memory_stress": "combining large datasets",
    "iterrows_vs_vectorized": "processing rows of data",
    "memory_profiling_over_time": "repeated data operations over time",
    "edge_case_dtypes": "handling unusual data formats",
    "concurrent_dataframe_access": "accessing data from multiple places at once",
    "session_write_concurrency": "multiple users writing at the same time",
    "array_size_scaling": "working with larger arrays",
    "matrix_operation_scaling": "heavy number crunching",
    "concurrent_array_access": "accessing data from multiple threads",
    "edge_case_inputs": "handling unusual or extreme inputs",
    "repeated_allocation_memory": "allocating memory repeatedly",
    "async_concurrent_load": "handling many requests at once",
    "sync_handler_thread_exhaustion": "running synchronous code under load",
    "pydantic_validation_stress": "validating request data at scale",
    "websocket_connection_scaling": "handling many open connections",
    "large_response_streaming": "streaming large responses",
    "middleware_chain_overhead": "processing middleware layers",
    "async_error_handling": "handling errors in async operations",
    "memory_under_load": "managing memory under heavy traffic",
    "large_payload_handling": "processing large request payloads",
}


def _human_time(ms: float) -> str:
    """Convert milliseconds to natural language time description.

    Uses a fine-grained scale so that values like 0.16ms and 78ms
    don't both collapse to "instant".
    """
    if ms < 1:
        return "instant"
    if ms < 100:
        return "fast"
    if ms < 500:
        return "noticeable delay"
    if ms < 2000:
        return "slow"
    if ms < 10000:
        return "very slow"
    if ms < 30000:
        seconds = round(ms / 5000) * 5
        return f"about {seconds} seconds"
    if ms < 60000:
        return "about 30 seconds"
    return "over a minute"


def _describe_scenario(scenario_name: str) -> str:
    """Map a scenario name to a user-meaningful activity description.

    Tries to match the template portion of the scenario name against
    ``_TEMPLATE_DESCRIPTIONS``. Falls back to keyword-based patterns
    for coupling scenarios and other conventions.
    """
    name = scenario_name.lower()

    # HTTP endpoint scenarios
    if name in ("http_get_root", "http_post_root"):
        return "loading your application's main page"
    if name.startswith("http_get_") or name.startswith("http_post_"):
        path = name.split("_", 2)[2] if name.count("_") >= 2 else ""
        if path and path != "root":
            return f"loading the /{path.replace('_', '/')} endpoint"

    # Try progressively shorter prefixes to find the template portion.
    # e.g. "flask_concurrent_request_load" → split into parts, try
    # joining from index 1, 2, ... until a match is found.
    parts = name.split("_")
    for start in range(1, len(parts)):
        template_key = "_".join(parts[start:])
        if template_key in _TEMPLATE_DESCRIPTIONS:
            return _TEMPLATE_DESCRIPTIONS[template_key]

    # Coupling scenario patterns
    if name.startswith("coupling_api_"):
        return "connecting components together"
    if name.startswith("coupling_compute_"):
        return "running calculations across components"
    if name.startswith("coupling_state_") or name.startswith("coupling_render_"):
        return "updating shared state"
    if name.startswith("coupling_errorhandler_"):
        return "handling errors between components"

    # Keyword fallbacks
    if "memory" in name or "leak" in name:
        return "managing memory over time"
    if "concurrent" in name:
        return "handling simultaneous operations"
    if "scaling" in name or "volume" in name:
        return "processing larger amounts of data"

    # Edge-case / check scenarios: strip dep prefix and _check suffix,
    # then humanize what's left so raw identifiers never reach the user.
    if name.endswith("_check"):
        remainder = name.rsplit("_check", 1)[0]
        # Strip leading dep prefix (first component)
        dep_parts = remainder.split("_", 1)
        if len(dep_parts) > 1:
            remainder = dep_parts[1]
        return f"{remainder.replace('_', ' ')} behavior"

    # Version discrepancy scenarios
    if name.endswith("_version_discrepancy"):
        dep = name.rsplit("_version_discrepancy", 1)[0]
        return f"{dep} version compatibility"

    # Generic stress for unrecognized deps
    if "generic_stress" in name:
        if name.startswith("unrecognized_deps"):
            return "general usage patterns for unrecognized dependencies"
        dep = name.split("_")[0] if "_" in name else name
        return f"general usage patterns for {dep}"

    # Last resort: strip known prefixes so raw identifiers don't leak.
    # coupling_compute_MyModule_foo → "running calculations (MyModule foo)"
    _COUPLING_PREFIXES = (
        "coupling_compute_", "coupling_api_", "coupling_render_",
        "coupling_state_", "coupling_errorhandler_", "coupling_behavior_",
        "coupling_",
    )
    for prefix in _COUPLING_PREFIXES:
        if name.startswith(prefix):
            remainder = scenario_name[len(prefix):]
            label = _humanize_identifier(remainder)
            if prefix.startswith("coupling_compute"):
                return f"running calculations ({label})"
            if prefix.startswith("coupling_api"):
                return f"connecting to external services ({label})"
            if prefix.startswith("coupling_render"):
                return f"rendering output ({label})"
            return f"component interaction ({label})"

    # Strip dep prefix and humanize
    dep_parts = name.split("_", 1)
    if len(dep_parts) > 1 and dep_parts[1]:
        return dep_parts[1].replace("_", " ")

    return ""


# Hand-curated display names for non-obvious scenario templates.
# Key is the template portion (after dep prefix), value is the
# consequence-oriented label.  The dep name is appended in parentheses
# by _humanize_title_name().
_CURATED_TITLE_MAP: dict[str, str] = {
    "apply_performance_cliff": "Sudden Slowdown on Large Operations",
    "settingwithcopy_warning_ignored": "Silent Data Corruption Risk",
    "memory_error_on_operations": "Memory Crash on Data Operations",
    "silent_dtype_coercion": "Silent Data Type Changes",
    "silent_overflow": "Silent Number Overflow",
    "broadcasting_shape_mismatch": "Array Shape Mismatch",
    "numpy_2_breaking_changes": "NumPy 2.0 Compatibility",
    "dtype_edge_cases": "Unusual Data Format Handling",
    "edge_case_dtypes": "Unusual Data Format Handling",
    "read_csv_encoding_crash": "CSV Encoding Crash",
    "empty_dataframe_operations": "Empty Data Edge Cases",
    "memory_error_on_allocation": "Memory Crash on Array Allocation",
}


def _build_timeout_description(
    sr: "ScenarioResult",
    all_results: list["ScenarioResult"],
    deps: list[str],
    current_timeout: int | None,
) -> str:
    """Build a detailed timeout description explaining why, what, and how.

    Uses timing data from the timed-out scenario's completed steps and
    from sibling scenarios that tested the same functions to estimate
    how long the full test would take.
    """
    import math

    parts: list[str] = []

    # ── 1. Why: extract avg execution time from completed steps ──
    completed_steps = [s for s in sr.steps if s.execution_time_ms > 0]
    avg_ms = 0.0
    func_name = ""
    if completed_steps:
        avg_ms = sum(s.execution_time_ms for s in completed_steps) / len(completed_steps)
        # Try to extract function name from step names or scenario name
        func_name = sr.source_functions[0] if sr.source_functions else ""
    else:
        # Fall back: look at sibling scenarios (coupling) that completed
        # and tested similar functions
        for other in all_results:
            if other is sr or not other.steps:
                continue
            if other.scenario_category != sr.scenario_category:
                continue
            other_steps = [s for s in other.steps if s.execution_time_ms > 0]
            if other_steps:
                avg_ms = (
                    sum(s.execution_time_ms for s in other_steps)
                    / len(other_steps)
                )
                func_name = other.source_functions[0] if other.source_functions else ""
                break

    if avg_ms > 0 and func_name:
        parts.append(
            f"Your application's {func_name} takes approximately "
            f"{avg_ms:.0f}ms per call."
        )
    elif avg_ms > 0:
        parts.append(
            f"Your application takes approximately {avg_ms:.0f}ms per "
            f"call at the tested scale."
        )

    # ── 2. What they're missing ──
    dep_str = deps[0] if deps else "application"
    cat = sr.scenario_category
    if cat == "data_volume_scaling":
        parts.append(
            f"This test checks whether your {dep_str} data processing "
            f"scales safely under increasing load."
        )
    elif cat == "memory_profiling":
        parts.append(
            f"This test checks whether your {dep_str} usage leaks "
            f"memory under sustained operation."
        )
    else:
        parts.append(
            f"This test checks how your {dep_str} behaves under "
            f"increasing stress."
        )

    # ── 3. What to do: recommend a timeout ──
    timeout_s = current_timeout or 90
    if avg_ms > 0 and completed_steps:
        # Estimate: typical data_volume_scaling has 5 tiers × N functions.
        # Use completed step count to estimate remaining work.
        total_steps_estimate = max(len(completed_steps) + 2, 5)
        estimated_total_s = (avg_ms * total_steps_estimate) / 1000
        recommended = max(
            timeout_s,
            math.ceil(estimated_total_s / 60) * 60,
        )
        if recommended > timeout_s:
            parts.append(
                f"Re-run with a {recommended}s limit (currently "
                f"{timeout_s}s) to complete this test."
            )
        else:
            parts.append(
                f"Re-run with a longer limit (currently {timeout_s}s) "
                f"to complete this test."
            )
    else:
        parts.append(
            f"Re-run with a longer limit (currently {timeout_s}s) "
            f"to complete this test."
        )

    return " ".join(parts)


def _humanize_title_name(scenario_name: str) -> str:
    """Convert a scenario name to a title-friendly label for finding headers.

    Checks ``_CURATED_TITLE_MAP`` first for hand-written labels, then
    falls back to mechanical conversion.

    ``pandas_apply_performance_cliff_check`` → ``Sudden Slowdown on Large Operations (pandas)``
    ``pandas_data_volume_scaling`` → ``Data Volume Scaling (pandas)``
    ``unrecognized_deps_generic_stress`` → ``General Stress Test``
    """
    name = scenario_name.lower()

    # Unrecognized deps — use affected_dependencies if available,
    # otherwise label as "unrecognized dependencies"
    if name.startswith("unrecognized_deps_"):
        return "General Stress Test (unrecognized dependencies)"

    # Coupling scenarios — differentiate by behavior type
    if name.startswith("coupling_api_"):
        fn = _humanize_identifier(scenario_name[len("coupling_api_"):])
        return f"API Coupling ({fn})"
    if name.startswith("coupling_compute_"):
        fn = _humanize_identifier(scenario_name[len("coupling_compute_"):])
        return f"Computation Coupling ({fn})"
    if name.startswith("coupling_render_"):
        fn = _humanize_identifier(scenario_name[len("coupling_render_"):])
        return f"Render Coupling ({fn})"
    if name.startswith("coupling_errorhandler_"):
        fn = _humanize_identifier(scenario_name[len("coupling_errorhandler_"):])
        return f"Error Handling ({fn})"
    if name.startswith("coupling_state_setters_group_"):
        return "Shared State Coupling"
    if name.startswith("coupling_"):
        return "Component Coupling"

    # Version discrepancy
    if name.endswith("_version_discrepancy"):
        dep = name.rsplit("_version_discrepancy", 1)[0]
        return f"Version Compatibility ({dep})"

    # Strip _check suffix if present
    base = name.rsplit("_check", 1)[0] if name.endswith("_check") else name

    # Try to find a (dep, template) split using _TEMPLATE_DESCRIPTIONS
    # or _CURATED_TITLE_MAP.  Try progressively shorter prefixes.
    parts = base.split("_")
    for start in range(1, len(parts)):
        template_key = "_".join(parts[start:])
        dep = "_".join(parts[:start])
        if template_key in _CURATED_TITLE_MAP:
            return f"{_CURATED_TITLE_MAP[template_key]} ({dep})"
        if template_key in _TEMPLATE_DESCRIPTIONS:
            label = template_key.replace("_", " ").title()
            return f"{label} ({dep})"

    # Fallback: first part as dep, rest as label
    dep = parts[0]
    template = "_".join(parts[1:]) if len(parts) > 1 else base
    label = template.replace("_", " ").title()
    if dep and label:
        return f"{label} ({dep})"
    return label or scenario_name


def _humanize_scenario_name(scenario_name: str) -> str:
    """Translate an internal scenario name to a readable label for "Also:" lines.

    Strips known prefixes (coupling_compute_, coupling_state_, etc.) and
    converts the remainder to readable text. Designed to give unique
    per-scenario labels, unlike ``_describe_scenario`` which returns
    category-level descriptions.
    """
    name_lower = scenario_name.lower()

    # Strip known coupling prefixes (preserve original case for remainder)
    _COUPLING_PREFIXES = (
        "coupling_compute_", "coupling_state_", "coupling_api_",
        "coupling_render_", "coupling_errorhandler_",
    )
    for prefix in _COUPLING_PREFIXES:
        if name_lower.startswith(prefix):
            remainder = scenario_name[len(prefix):]
            return _humanize_identifier(remainder)

    # Try template match for the suffix
    parts = name_lower.split("_")
    for start in range(1, len(parts)):
        template_key = "_".join(parts[start:])
        if template_key in _TEMPLATE_DESCRIPTIONS:
            dep = "_".join(parts[:start])
            return f"{dep} {_TEMPLATE_DESCRIPTIONS[template_key]}"

    # Fallback: humanize the full name
    return _humanize_identifier(scenario_name)


def _humanize_identifier(name: str) -> str:
    """Convert a snake_case or camelCase identifier to readable text.

    ``uuid_uuid4`` → ``UUID generation``,
    ``getattr`` → ``attribute access``,
    ``setError`` → ``set error``,
    ``setRawScores`` → ``set raw scores``.
    """
    _KNOWN_IDENTIFIERS: dict[str, str] = {
        "uuid_uuid4": "UUID generation",
        "uuid4": "UUID generation",
        "getattr": "attribute access",
        "setattr": "attribute assignment",
        "json_dumps": "JSON serialisation",
        "json_loads": "JSON parsing",
        "hashlib": "hashing operations",
        "re_compile": "regex compilation",
        "deepcopy": "deep copy operations",
        "pickle": "serialisation",
    }
    name_lower = name.lower()
    if name_lower in _KNOWN_IDENTIFIERS:
        return _KNOWN_IDENTIFIERS[name_lower]

    # Split camelCase: "setRawScores" → "set Raw Scores" → "set raw scores"
    result = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    # Replace underscores with spaces
    result = result.replace("_", " ").strip().lower()
    if result:
        return result[0].upper() + result[1:]
    return name


def _describe_step(step_name: str) -> str:
    """Translate a step name into user terms.

    Returns an empty string if the step name can't be meaningfully
    translated (e.g. edge cases, generic iterations).
    """
    m = re.match(r"data_size_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} items"

    m = re.match(r"concurrent_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} simultaneous users"

    m = re.match(r"api_concurrency_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} concurrent API calls"

    m = re.match(r"wsgi_concurrent_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} concurrent requests"

    m = re.match(r"async_handlers_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} async handlers"

    m = re.match(r"async_load_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} concurrent promises"

    m = re.match(r"sync_threadpool_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} threads in pool"

    m = re.match(r"sqlite_writers_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} concurrent writers"

    m = re.match(r"state_cycles_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} state mutation cycles"

    m = re.match(r"batch_(\d+)", step_name)
    if m:
        n = int(m.group(1))
        if n == 0:
            return "first iteration"
        return f"iteration {n:,}"

    m = re.match(r"io_size_(\d+)", step_name)
    if m:
        size = int(m.group(1))
        if size >= 1_000_000:
            return f"{size / 1_000_000:.0f}MB of data"
        if size >= 1000:
            return f"{size / 1000:.0f}KB of data"
        return f"{size:,} bytes of data"

    m = re.match(r"gil_threads_(\d+)", step_name)
    if m:
        return f"{int(m.group(1))} parallel threads"

    m = re.match(r"compute_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} items of data"

    m = re.match(r"rerun_rows_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} rows of data"

    m = re.match(r"cached_rows_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} rows of cached data"

    m = re.match(r"table_serialize_(\d+)kb", step_name, re.IGNORECASE)
    if m:
        kb = int(m.group(1))
        if kb >= 1000:
            return f"serialising a {kb // 1000}MB table"
        return f"serialising a {kb}KB table"

    m = re.match(r"session_reruns_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} page refreshes"

    m = re.match(r"render_nodes_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} UI elements"

    if step_name == "render_memory_growth":
        return "repeated rendering cycles"

    m = re.match(r"error_flood_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} simultaneous errors"

    m = re.match(r"session_writes_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} session write cycles"

    m = re.match(r"validation_fields_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} fields to validate"

    m = re.match(r"wsgi_payload_(\d+)kb", step_name, re.IGNORECASE)
    if m:
        kb = int(m.group(1))
        if kb >= 1000:
            return f"a {kb // 1000}MB request payload"
        return f"a {kb}KB request payload"

    if step_name == "api_timeout_handling":
        return "slow API responses"

    if step_name == "edge_no_callables":
        return "empty input"

    # HTTP load testing labels: "N concurrent"
    m = re.match(r"(\d+) concurrent", step_name)
    if m:
        n = int(m.group(1))
        return f"{n:,} concurrent connection{'s' if n != 1 else ''}"

    # Generic fallback: strip underscores, add spaces, capitalise
    # Only apply if the name has a recognisable word_number pattern
    # and is NOT an iteration counter (iteration_5, repeat_3, etc.)
    m = re.match(r"([a-z_]+?)_(\d+)$", step_name)
    if m:
        prefix = m.group(1)
        if prefix not in _ITERATION_PREFIXES and prefix != "edge":
            label = prefix.replace("_", " ")
            number = int(m.group(2))
            return f"{number:,} {label}"

    return ""


def _describe_impact(
    metric: str,
    first_val: float,
    last_val: float,
    user_scale: int | None = None,
    activity: str = "",
) -> str:
    """Describe the impact of a degradation in real terms.

    For timing: uses _human_time to convert both endpoints to natural
    language so users see "from fast to very slow" instead of "204x".

    For memory: projects multi-user impact with practical server sizing
    (e.g. "a 2GB server would handle approximately 27 sessions").

    For errors: names the count and relates to user impact.

    Args:
        metric: The degradation metric name.
        first_val: Value at the first measurement step.
        last_val: Value at the last measurement step.
        user_scale: User's stated concurrent user count.
        activity: What was being tested (e.g. "concurrent user handling").
    """
    during = f"during {activity}" if activity else "under stress testing"

    if metric == "execution_time_ms":
        first_desc = _human_time(first_val)
        last_desc = _human_time(last_val)
        if first_desc == last_desc:
            return (
                f"{during}, response time reached {_format_ms(last_val)}"
            )
        return (
            f"{during}, response time went from {first_desc} "
            f"to {last_desc}"
        )
    if metric == "memory_peak_mb":
        base = f"{during}, memory "
        if first_val > 0 and last_val > first_val * 1.5:
            base += f"grew from {first_val:.0f}MB to {last_val:.0f}MB"
        else:
            base += f"reached {last_val:.0f}MB"
        # Practical server sizing
        if last_val >= 50 and user_scale:
            sessions = max(1, int(2048 / last_val))
            base += (
                f". On a 2GB server, that leaves room for approximately "
                f"{sessions} concurrent sessions before running out of memory"
            )
        elif last_val >= 50:
            sessions = max(1, int(2048 / last_val))
            base += (
                f". On a typical 2GB server, that limits you to "
                f"approximately {sessions} concurrent sessions"
            )
        else:
            base += (
                ". This is moderate and unlikely to cause problems "
                "on a typical server"
            )
        return base
    if metric == "error_count":
        count = int(last_val) - int(first_val)
        if count <= 0:
            count = int(last_val)
        noun = "error" if count == 1 else "errors"
        return (
            f"{during}, {count} runtime {noun} occurred. "
            f"This means your app may crash or produce wrong "
            f"results under this load"
        )
    return f"{during}, {_human_metric(metric)} increases significantly"


def _format_ms(ms: float) -> str:
    """Format a millisecond value as a concrete human-readable string."""
    if ms < 1:
        return f"{ms:.2f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60000:.1f}min"


def _time_qualifier(ms: float) -> str:
    """Return a user-meaningful qualifier for a time value."""
    if ms < 10:
        return "still fast"
    if ms < 100:
        return "still fast"
    if ms < 500:
        return "your users will start to notice"
    if ms < 2000:
        return "your users will notice"
    if ms < 10000:
        return "the app feels slow"
    return "the app stops responding"


def _memory_qualifier(mb: float) -> str:
    """Return a user-meaningful qualifier for a memory value."""
    if mb < 50:
        return ""
    if mb < 200:
        return "getting heavy"
    if mb < 500:
        return "heavy — may crash on limited devices"
    return "very heavy — likely to crash"


def _metric_label(metric: str) -> str:
    """Return a human-readable label for a degradation metric."""
    if metric in ("execution_time_ms", "response_time_ms"):
        return "Response time under load"
    if metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb"):
        return "Memory usage under load"
    if metric == "error_count":
        return "Errors under load"
    return ""


def _metric_unit(metric: str) -> str:
    """Return the unit suffix for a metric value."""
    if metric in ("execution_time_ms", "response_time_ms"):
        return "ms"
    if metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb"):
        return "MB"
    if metric == "error_count":
        return " errors"
    return ""


def _breaking_point_label(dp: "DegradationPoint") -> str:
    """Build a breaking point label that includes what metric breaks and where.

    Returns e.g. "response time exceeds 500ms at 50,000 items of data"
    or "memory exceeds 120MB at 50 simultaneous users".
    """
    if not dp.breaking_point:
        return ""
    bp_desc = _describe_step(dp.breaking_point) or dp.breaking_point

    # Find the value at the breaking point step
    bp_value = None
    for label, value in dp.steps:
        if label == dp.breaking_point:
            bp_value = value
            break

    is_time = dp.metric in ("execution_time_ms", "response_time_ms")
    is_memory = dp.metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")
    is_errors = dp.metric == "error_count"

    if bp_value is not None:
        if is_time:
            val_str = _format_ms(bp_value)
            return f"response time exceeds {val_str} at {bp_desc}"
        if is_memory:
            val_str = f"{bp_value:.0f}MB" if bp_value >= 1 else f"{bp_value:.2f}MB"
            return f"memory exceeds {val_str} at {bp_desc}"
        if is_errors:
            return f"errors begin at {bp_desc}"

    return f"at {bp_desc}"


def _is_significant_finding(f: "Finding") -> bool:
    """Return True if a finding is worth surfacing in the summary.

    Critical findings are always significant. Warning findings are only
    significant if they involve resource caps, errors, high memory
    (≥50MB), or slow response times (≥500ms).
    """
    if f.severity == "critical":
        return True
    if f.category == "edge_case_input":
        return True
    if "resource" in (f.title or "").lower():
        return True
    if f._error_count > 0:
        return True
    if f._peak_memory_mb >= 50:
        return True
    if f._execution_time_ms >= 500:
        return True
    return False


def _is_significant_degradation(dp: "DegradationPoint") -> bool:
    """Return True if a degradation point is worth surfacing in the summary.

    Filters out degradation that has no real consequence for the user:
    - Memory peaking below 50MB (moderate, not actionable)
    - Execution time staying under 500ms (still fast for users)
    - Error count ending at 0
    """
    if not dp.steps:
        return False
    last_val = dp.steps[-1][1]
    first_val = dp.steps[0][1]
    if dp.metric == "memory_peak_mb":
        # Only significant if peak is ≥50MB or growth is ≥10x
        if last_val < 50 and (first_val <= 0 or last_val / max(first_val, 0.1) < 10):
            return False
    elif dp.metric == "execution_time_ms":
        # Only significant if it crosses into noticeable territory (>500ms)
        if last_val < 500:
            return False
    elif dp.metric == "error_count":
        if last_val <= 0:
            return False
    return True


def _build_degradation_narrative(dp: "DegradationPoint") -> str:
    """Build a concise narrative from degradation steps.

    Shows at most 3 key points: baseline (first), threshold (where
    degradation becomes noticeable), and peak (last).  This avoids
    dumping every data point as a sentence.
    """
    if not dp.steps:
        return dp.description or ""

    is_time = dp.metric in ("execution_time_ms", "response_time_ms")
    is_memory = dp.metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")
    is_errors = dp.metric == "error_count"

    # Flat memory defense-in-depth: if all values are within 10% of mean,
    # report as stable rather than showing a misleading degradation curve.
    if is_memory:
        values = [v for _, v in dp.steps]
        mean_val = sum(values) / len(values) if values else 0
        if mean_val > 0:
            max_dev = max(abs(v - mean_val) for v in values)
            if max_dev / mean_val < 0.10:
                return f"Memory usage stable at {mean_val:.0f}MB under load."

    # Select key points: baseline, threshold, peak
    key_points = _select_key_points(dp.steps, is_time, is_memory)

    parts: list[str] = []
    for idx, (label, value) in enumerate(key_points):
        step_desc = _describe_step(label) or label

        if is_time:
            val_str = _format_ms(value)
            if idx == 0:
                parts.append(f"With {step_desc}, response time is {val_str}")
            else:
                qualifier = _time_qualifier(value)
                parts.append(
                    f"With {step_desc}, it's {val_str} — {qualifier}"
                )
        elif is_memory:
            val_str = f"{value:.0f}MB" if value >= 1 else f"{value:.2f}MB"
            if idx == 0:
                parts.append(f"With {step_desc}, memory usage is {val_str}")
            else:
                qualifier = _memory_qualifier(value)
                if qualifier:
                    parts.append(
                        f"With {step_desc}, it's {val_str} — {qualifier}"
                    )
                else:
                    parts.append(f"With {step_desc}, it's {val_str}")
        elif is_errors:
            val_str = str(int(value))
            parts.append(
                f"With {step_desc}, {val_str} error{'s' if value != 1 else ''}"
            )
        else:
            unit = _metric_unit(dp.metric)
            parts.append(f"With {step_desc}, {value:.2f}{unit}")

    narrative = ". ".join(parts) + "."

    # Append consequence context — the "so what" for non-technical readers
    consequence = _consequence_line(dp.metric, dp.steps)
    if consequence:
        narrative += " " + consequence

    return narrative


def _consequence_line(metric: str, steps: list[tuple[str, float]]) -> str:
    """Return a plain-language consequence for the final degradation value."""
    if not steps:
        return ""
    last_val = steps[-1][1]

    if metric in ("execution_time_ms", "response_time_ms"):
        if last_val < 100:
            return "This is within acceptable range — users won't notice."
        if last_val >= 1000:
            return "The application will feel unresponsive."
        if last_val >= 500:
            return "Users will notice significant delays at this point."
        return ""

    if metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb"):
        if last_val > 50:
            sessions = int(2048 / last_val)
            return (
                f"At this memory usage, a typical 2GB server supports "
                f"approximately {sessions} concurrent sessions."
            )
        return ""

    return ""


def _select_key_points(
    steps: list[tuple[str, float]],
    is_time: bool,
    is_memory: bool,
) -> list[tuple[str, float]]:
    """Pick baseline, threshold, and peak from a degradation curve.

    Returns at most 3 points.  The threshold is the first step where
    the value becomes meaningfully worse than baseline.
    """
    if len(steps) <= 3:
        return list(steps)

    baseline = steps[0]
    peak = steps[-1]
    base_val = baseline[1]

    # Find threshold: first point where value >= 2x baseline (or qualifier changes)
    threshold = None
    for step in steps[1:-1]:
        val = step[1]
        if is_time:
            if base_val > 0 and val / base_val >= 2.0:
                threshold = step
                break
        elif is_memory:
            if base_val > 0 and val / base_val >= 1.5:
                threshold = step
                break
        else:
            if base_val > 0 and val / base_val >= 2.0:
                threshold = step
                break

    if threshold and threshold != peak:
        return [baseline, threshold, peak]
    return [baseline, peak]


def _extract_cap_type(f: "Finding") -> str:
    """Extract which resource cap was hit from a finding's details.

    Returns a plain-language fragment like "requests timed out" or
    "memory hit the safety limit", or empty string if unknown.
    """
    details_lower = f.details.lower() if f.details else ""
    title_lower = f.title.lower() if f.title else ""
    combined = f"{details_lower} {title_lower}"

    if "timeout" in combined:
        return "requests timed out under load"
    if "memory" in combined and ("cap" in combined or "limit" in combined or "oom" in combined):
        return "memory hit the safety limit"
    if "resource" in combined and "cap" in combined:
        return "the system hit its resource limits"
    return ""


def _describe_errors(f: "Finding") -> str:
    """Describe errors from a finding in plain language.

    Uses the finding's details/title to determine error type, and
    handles singular/plural correctly.
    """
    count = f._error_count
    details_lower = f.details.lower() if f.details else ""
    title_lower = f.title.lower() if f.title else ""
    combined = f"{details_lower} {title_lower}"

    if "timeout" in combined:
        if count == 1:
            return "1 request timed out"
        return f"{count} requests timed out"
    if "memory" in combined and ("oom" in combined or "limit" in combined or "cap" in combined):
        return "the system ran out of memory"
    if "connection" in combined:
        if count == 1:
            return "1 connection failed"
        return f"{count} connections failed"

    # Generic fallback with correct pluralization
    if count == 1:
        return "1 error occurred"
    return f"{count} errors occurred"


def _describe_errors_with_context(f: "Finding", scenario_name: str = "") -> str:
    """Describe errors with type and significance context.

    Extends ``_describe_errors`` by adding what the error means for
    the user, based on the error type and scenario context.
    """
    base = _describe_errors(f)
    count = f._error_count
    details_lower = f.details.lower() if f.details else ""
    title_lower = f.title.lower() if f.title else ""
    combined = f"{details_lower} {title_lower}"

    # Add significance for generic errors
    if "timed out" in base or "ran out of memory" in base or "connection" in base:
        return base  # Already specific enough

    # For generic "N errors occurred", add the error type if detectable
    if "runtime" in combined or "exception" in combined or "traceback" in combined:
        noun = "runtime error" if count == 1 else "runtime errors"
        return f"{count} {noun} occurred"
    if "assertion" in combined:
        noun = "assertion failure" if count == 1 else "assertion failures"
        return f"{count} {noun} occurred"

    return base


# Step name prefixes that are repetition counters, not load levels.
_ITERATION_PREFIXES = frozenset({"iteration", "repeat", "run", "attempt", "try", "batch"})


def _step_level(step_name: str) -> Optional[int]:
    """Extract the trailing numeric level from a step name.

    Matches any ``word_DIGITS`` pattern (e.g. ``concurrent_50``,
    ``compute_10000``, ``io_size_1048576``).  Returns ``None`` if:
    - the step name has no trailing number,
    - the prefix is a repetition counter (``iteration_N``, ``repeat_N``, …),
    - the level is 0.
    """
    m = re.match(r"(.+)_(\d+)$", step_name)
    if not m:
        return None
    prefix = m.group(1).rsplit("_", 1)[-1].lower()
    if prefix in _ITERATION_PREFIXES:
        return None
    val = int(m.group(2))
    return val if val > 0 else None


def _format_user_fraction(load_level: int, user_scale: int) -> str:
    """Format failure threshold relative to user scale without showing 0%.

    When the percentage is tiny (e.g. 20 out of 360,234 = 0.006%),
    shows the absolute number instead: "just 20 concurrent users".
    When percentage is meaningful (≥1%), shows both:
    "just 500 concurrent users (less than 1% of your expected 360,234)".

    Args:
        load_level: Concurrent user count that caused failure.
        user_scale: User's stated concurrent user count.

    Returns:
        A human-readable sentence fragment.
    """
    if user_scale <= 0:
        return ""
    pct = load_level / user_scale * 100
    if pct < 1:
        return (
            f"The app fails at just {load_level:,} concurrent users "
            f"— far below your expected {user_scale:,}."
        )
    return (
        f"Just {load_level:,} concurrent users "
        f"(only {pct:.0f}% of your expected {user_scale:,}) "
        f"is enough to break it."
    )


def _extract_load_level(finding: "Finding") -> tuple[Optional[int], bool]:
    """Extract the load level at which a finding occurred.

    Returns ``(level, is_concurrency)``.  *is_concurrency* is determined
    by the finding's **category** (not the step name prefix):
    categories in ``_CONCURRENCY_CATEGORIES`` are concurrency-type and
    comparable to ``user_scale``; everything else is data-size-type.

    Returns ``(None, False)`` if no load level can be determined.
    """
    is_concurrency = finding.category in _CONCURRENCY_CATEGORIES

    # Preferred: pre-extracted from ScenarioResult steps
    if finding._load_level is not None and finding._load_level > 0:
        return (finding._load_level, is_concurrency)

    combined = f"{finding.title} {finding.details}"

    # Look for any word_N pattern in the combined text
    m = _STEP_LEVEL_RE.search(combined)
    if m and int(m.group(2)) > 0:
        prefix = m.group(1).rsplit("_", 1)[-1].lower()
        if prefix not in _ITERATION_PREFIXES:
            return (int(m.group(2)), is_concurrency)

    # Prose fallback: "N concurrent" or "N sessions" or "N users"
    m = re.search(r"(\d+)\s+(?:concurrent|sessions?|users?)", combined, re.IGNORECASE)
    if m:
        return (int(m.group(1)), True)

    return (None, False)


def _extract_project_ref(operational_intent: str) -> str:
    """Extract a short noun phrase from the user's operational intent.

    Used to reference the project naturally (e.g. "your budget tracker"
    instead of "your project").

    Truncates at the first arrow, period, dash (surrounded by spaces),
    newline, comma, or semicolon and strips parenthetical framework
    references like "(Flask)".
    """
    text = operational_intent.strip()
    if not text:
        return "project"

    # Truncate at first arrow, newline, or clause boundary
    text = re.split(
        r"\u2192|->|\n|[.,;]| \u2014 | - | that | which ",
        text,
        maxsplit=1,
    )[0].strip()

    # Strip parenthetical framework/tech references e.g. "(Flask)"
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()

    # Strip leading filler phrases
    filler = [
        r"^I(?:'m| am) building\s+",
        r"^I(?:'ve| have) built\s+",
        r"^I built\s+",
        r"^I made\s+",
        r"^I have\s+",
        r"^It'?s\s+",
        r"^This is\s+",
        r"^We have\s+",
        r"^We built\s+",
    ]
    for pattern in filler:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Strip leading articles
    text = re.sub(r"^(?:a|an|the|my|our)\s+", "", text, flags=re.IGNORECASE).strip()

    # Cap at 50 chars, break at word boundary
    if len(text) > 50:
        cut = text[:50].rfind(" ")
        if cut > 10:
            text = text[:cut]
        else:
            text = text[:50]

    if len(text) < 3:
        return "project"

    return text
