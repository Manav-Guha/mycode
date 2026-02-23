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
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from mycode.engine import ExecutionEngineResult, ScenarioResult, StepResult
from mycode.ingester import DependencyInfo, IngestionResult
from mycode.library.loader import ProfileMatch
from mycode.scenario import LLMBackend, LLMConfig, LLMError, LLMResponse

logger = logging.getLogger(__name__)


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
    _peak_memory_mb: float = 0.0
    _execution_time_ms: float = 0.0
    _error_count: int = 0


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
        scenarios_run: Total scenarios executed.
        scenarios_passed: Scenarios with no critical findings.
        scenarios_failed: Scenarios with errors or resource cap hits.
        total_errors: Total error count across all scenarios.
        operational_context: The user's stated intent, for framing.
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
    scenarios_run: int = 0
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    total_errors: int = 0
    operational_context: str = ""
    model_used: str = "offline"
    token_usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )

    def as_text(self) -> str:
        """Render the report as readable plain text."""
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
        sections.append(
            f"\nScenarios: {self.scenarios_run} run, "
            f"{self.scenarios_passed} clean, "
            f"{self.scenarios_failed} with issues"
        )
        if self.total_errors:
            sections.append(f"Total errors captured: {self.total_errors}")

        # Findings
        if not self.findings and self.scenarios_run:
            sections.append("\n" + "-" * 40)
            sections.append("  Findings")
            sections.append("-" * 40)
            sections.append(
                "\n  All stress test scenarios completed cleanly. "
                "No errors, resource limit hits, or degradation detected "
                "under the conditions tested."
            )
        elif self.findings:
            sections.append("\n" + "-" * 40)
            sections.append("  Findings")
            sections.append("-" * 40)
            for f in self.findings:
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
                        # Extract the scenario name from the grouped finding title
                        parts = gf.title.split(": ", 1)
                        names.append(parts[1] if len(parts) > 1 else gf.title)
                    shown = names[:5]
                    extra = len(names) - 5
                    also_line = ", ".join(shown)
                    if extra > 0:
                        also_line += f" +{extra} more"
                    sections.append(f"    Also: {also_line}")

        # Incomplete tests (environment issues, not real failures)
        if self.incomplete_tests:
            sections.append("\n" + "-" * 40)
            sections.append("  Incomplete Tests")
            sections.append("-" * 40)
            sections.append(
                "  These tests could not run fully due to environment "
                "issues (missing modules, uninstalled dependencies, "
                "missing files). They are not code failures."
            )
            for f in self.incomplete_tests:
                sections.append(f"\n  - {f.title}")
                if f.details:
                    sections.append(f"    {f.details}")

        # Degradation
        if self.degradation_points:
            sections.append("\n" + "-" * 40)
            sections.append("  Degradation Curves")
            sections.append("-" * 40)
            for dp in self.degradation_points:
                header = f"{dp.scenario_name} ({dp.metric})"
                if dp.group_count > 1:
                    header += f" (and {dp.group_count - 1} similar)"
                sections.append(f"\n  {header}:")
                if dp.description:
                    sections.append(f"    {dp.description}")
                if dp.steps:
                    for label, value in dp.steps:
                        sections.append(f"      {label}: {value:.2f}")
                if dp.breaking_point:
                    sections.append(
                        f"    >> Breaking point: {dp.breaking_point}"
                    )
                if dp.grouped_points:
                    names = [gp.scenario_name for gp in dp.grouped_points]
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

        # Unrecognized deps
        if self.unrecognized_deps:
            sections.append("\n" + "-" * 40)
            sections.append("  Unrecognized Dependencies")
            sections.append("-" * 40)
            sections.append(
                "  The following dependencies have no profile in the component "
                "library. Generic stress testing was applied."
            )
            for dep in self.unrecognized_deps:
                sections.append(f"  - {dep}")

        # Footer
        sections.append("\n" + "=" * 60)
        sections.append(
            "  myCode diagnoses — it does not prescribe. "
            "Interpret results in your context."
        )
        sections.append("=" * 60)

        return "\n".join(sections)


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
    ) -> DiagnosticReport:
        """Generate a diagnostic report.

        Args:
            execution: Raw results from the Execution Engine.
            ingestion: Full project ingestion result.
            profile_matches: Component library matches.
            operational_intent: User's stated intent from the
                Conversational Interface.

        Returns:
            DiagnosticReport with findings, degradation curves, and flags.
        """
        report = DiagnosticReport(
            operational_context=operational_intent,
            scenarios_run=len(execution.scenario_results),
        )

        # 1. Analyze execution results → findings + degradation
        self._analyze_execution(execution, report)

        # 2. Flag version discrepancies from ingester
        self._flag_version_discrepancies(ingestion, profile_matches, report)

        # 3. Flag unrecognized dependencies
        self._flag_unrecognized_deps(profile_matches, report)

        # 3b. Group similar findings and degradation points to reduce noise
        report.findings = self._group_similar_findings(report.findings)
        report.degradation_points = self._group_similar_degradation_points(
            report.degradation_points,
        )

        # 4. Sort findings by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        report.findings.sort(key=lambda f: severity_order.get(f.severity, 9))

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
    ) -> None:
        """Extract findings and degradation curves from execution results."""
        passed = 0
        failed = 0
        total_errors = 0

        for sr in execution.scenario_results:
            total_errors += sr.total_errors

            if sr.status == "completed" and sr.total_errors == 0 and not sr.resource_cap_hit:
                passed += 1
            else:
                failed += 1

            # Extract metrics for grouping
            sr_peak_memory = max(
                (s.memory_peak_mb for s in sr.steps), default=0.0,
            )
            sr_exec_time = max(
                (s.execution_time_ms for s in sr.steps), default=0.0,
            )

            # ── Environment-only failures → incomplete tests ──
            if self._is_environment_only(sr):
                f = Finding(
                    title=f"Could not test: {sr.scenario_name}",
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
                report.incomplete_tests.append(f)
                continue

            # Check for failures
            if sr.status == "failed":
                f = Finding(
                    title=f"Scenario failed: {sr.scenario_name}",
                    severity="critical",
                    category=sr.scenario_category,
                    description=sr.summary or "Scenario failed during execution.",
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                report.findings.append(f)

            # Check resource cap hits
            if sr.resource_cap_hit:
                f = Finding(
                    title=f"Resource limit hit: {sr.scenario_name}",
                    severity="critical",
                    category=sr.scenario_category,
                    description=(
                        f"The test hit a resource cap during {sr.scenario_name}. "
                        f"This means the code exceeded safe operating limits "
                        f"under stress."
                    ),
                    details=self._summarize_cap_hits(sr),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                report.findings.append(f)

            # Check for errors
            if sr.total_errors > 0 and sr.status != "failed":
                f = Finding(
                    title=f"Errors during: {sr.scenario_name}",
                    severity="warning",
                    category=sr.scenario_category,
                    description=(
                        f"{sr.total_errors} error(s) occurred during this test."
                    ),
                    details=self._summarize_errors(sr),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                report.findings.append(f)

            # Check failure indicators
            if sr.failure_indicators_triggered:
                f = Finding(
                    title=f"Failure indicators triggered: {sr.scenario_name}",
                    severity="warning",
                    category=sr.scenario_category,
                    description=(
                        f"Triggered: {', '.join(sr.failure_indicators_triggered)}"
                    ),
                    affected_dependencies=self._deps_from_name(sr.scenario_name),
                )
                f._peak_memory_mb = sr_peak_memory
                f._execution_time_ms = sr_exec_time
                f._error_count = sr.total_errors
                report.findings.append(f)

            # Detect degradation curves
            self._detect_degradation(sr, report)

        report.scenarios_passed = passed
        report.scenarios_failed = failed
        report.total_errors = total_errors

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
                report.degradation_points.append(dp)

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
        """
        if len(steps) < 2:
            return None

        values = [v for _, v in steps]
        first = values[0]
        last = values[-1]

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

    # ── Version Discrepancies ──

    def _flag_version_discrepancies(
        self,
        ingestion: IngestionResult,
        profile_matches: list[ProfileMatch],
        report: DiagnosticReport,
    ) -> None:
        """Flag version discrepancies from ingester and profile matches."""
        # From ingester dependency info (skip devDependencies)
        for dep in ingestion.dependencies:
            if dep.is_dev:
                continue
            if dep.is_outdated:
                msg = f"{dep.name}: installed {dep.installed_version}"
                if dep.latest_version:
                    msg += f", latest is {dep.latest_version}"
                report.version_flags.append(msg)
                report.findings.append(Finding(
                    title=f"Outdated dependency: {dep.name}",
                    severity="info",
                    description=(
                        f"{dep.name} is outdated. Installed version "
                        f"{dep.installed_version} vs latest "
                        f"{dep.latest_version or 'unknown'}. "
                        f"Outdated dependencies may have known issues "
                        f"that affect behavior under stress."
                    ),
                    affected_dependencies=[dep.name],
                ))
            elif dep.is_missing:
                report.version_flags.append(
                    f"{dep.name}: declared but not installed"
                )
                report.findings.append(Finding(
                    title=f"Missing dependency: {dep.name}",
                    severity="warning",
                    description=(
                        f"{dep.name} is declared in requirements but not "
                        f"installed. This may cause import failures under "
                        f"certain code paths."
                    ),
                    affected_dependencies=[dep.name],
                ))

        # From profile version_match
        for match in profile_matches:
            if match.version_match is False and match.version_notes:
                msg = f"{match.dependency_name}: {match.version_notes}"
                if msg not in report.version_flags:
                    report.version_flags.append(msg)

    # ── Unrecognized Dependencies ──

    def _flag_unrecognized_deps(
        self,
        profile_matches: list[ProfileMatch],
        report: DiagnosticReport,
    ) -> None:
        """Flag dependencies without component library profiles."""
        for match in profile_matches:
            if match.profile is None:
                report.unrecognized_deps.append(match.dependency_name)

        if report.unrecognized_deps:
            report.findings.append(Finding(
                title=(
                    f"{len(report.unrecognized_deps)} unrecognized "
                    f"dependency(ies)"
                ),
                severity="info",
                description=(
                    f"These dependencies have no profile in the component "
                    f"library: {', '.join(report.unrecognized_deps[:10])}"
                    + ("..." if len(report.unrecognized_deps) > 10 else "")
                    + ". Generic stress testing was applied."
                ),
                affected_dependencies=report.unrecognized_deps[:10],
            ))

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
        if project_name:
            project_ref = f"your {project_name}"
        elif operational_intent:
            ref = _extract_project_ref(operational_intent)
            project_ref = f"your {ref}"
        else:
            project_ref = "your project"

        # ── Overall assessment ──
        critical = [f for f in report.findings if f.severity == "critical"]
        warnings = [f for f in report.findings if f.severity == "warning"]

        if critical:
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
            lines.append(
                f"{project_ref[0].upper()}{project_ref[1:]} looks solid "
                f"under the conditions we tested."
            )

        # ── Top findings (up to 3, prioritized, one per scenario) ──
        #
        # Priority order: resource cap hits first, then memory degradation,
        # then execution time degradation, then other findings.
        # Degradation points are preferred over findings for the same
        # scenario because they contain richer curve data (start→end
        # values, breaking points).

        # Build a list of (priority, scenario_name, translated_text)
        candidates: list[tuple[int, str, str]] = []

        # Degradation points first — they have the richest data
        degradation_scenarios: set[str] = set()
        for dp in report.degradation_points:
            translated = self._translate_degradation(dp, project_ref)
            degradation_scenarios.add(dp.scenario_name)
            # Memory degradation at priority 1, execution time at 2
            if dp.metric == "memory_peak_mb":
                prio = 1
            elif dp.metric == "error_count":
                prio = 1
            else:
                prio = 2
            candidates.append((prio, dp.scenario_name, translated))

        # Critical/warning findings — only for scenarios without
        # degradation points (which already have better data)
        for f in report.findings:
            if f.severity == "info":
                continue
            scenario_name = (
                f.title.split(": ", 1)[-1] if ": " in f.title else ""
            )
            if scenario_name in degradation_scenarios:
                continue
            translated = self._translate_finding(f, project_ref)
            if translated:
                # Resource cap hits get priority 0, other criticals 1, warnings 2
                if "resource" in f.title.lower() or f.category == "edge_case_input":
                    prio = 0
                elif f.severity == "critical":
                    prio = 1
                else:
                    prio = 2
                candidates.append((prio, scenario_name, translated))

        # Sort by priority, then pick top 3 (one per scenario, no duplicate text)
        candidates.sort(key=lambda c: c[0])
        items: list[str] = []
        seen_scenarios: set[str] = set()
        seen_text: set[str] = set()
        for _prio, scenario_name, translated in candidates:
            if len(items) >= 3:
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

        # ── Closing line ──
        lines.append("")
        if items:
            lines.append(
                "See detailed technical findings below — you can paste these "
                "into your coding tool for specific fixes."
            )
        else:
            lines.append(
                "No issues were found under the conditions we tested. "
                "See detailed results below."
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
            "- Do not suggest fixes or code changes.\n"
            "- Do not use engineering jargon.\n"
            "- Each bullet should start with '- '.\n"
            "- Be specific about what happened, using the project name.\n"
            "- End with a brief closing line (not a bullet) directing them "
            "to the detailed findings below.\n\n"
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
    ) -> str:
        """Translate a degradation point into plain language.

        Reads the actual curve data to describe what happens at the
        breaking point in user terms, not just name it.
        ``project_ref`` is woven into the activity phrase (e.g.
        "your incident matching system").
        """
        activity = _describe_scenario(dp.scenario_name)
        if activity:
            activity = f"{project_ref} is {activity}"

        first_val = dp.steps[0][1] if dp.steps else 0.0
        last_val = dp.steps[-1][1] if dp.steps else 0.0
        impact = _describe_impact(dp.metric, first_val, last_val)

        # Describe what happens at the breaking point
        breaking_desc = ""
        if dp.breaking_point and len(dp.steps) >= 2:
            step_desc = _describe_step(dp.breaking_point)
            # Find the value at the breaking point
            bp_val = None
            for label, val in dp.steps:
                if label == dp.breaking_point:
                    bp_val = val
                    break
            if step_desc and bp_val is not None:
                if dp.metric == "execution_time_ms":
                    breaking_desc = (
                        f"starts slowing down noticeably around "
                        f"{step_desc} ({_format_ms(bp_val)})"
                    )
                elif dp.metric == "memory_peak_mb":
                    breaking_desc = (
                        f"starts climbing around {step_desc} "
                        f"({bp_val:.0f}MB)"
                    )
                else:
                    breaking_desc = f"starts around {step_desc}"
            elif step_desc:
                breaking_desc = (
                    f"starts around {step_desc}"
                )

        if activity and breaking_desc:
            return f"When {activity}, {impact} — {breaking_desc}."
        if activity:
            return f"When {activity}, {impact}."
        if breaking_desc:
            return f"{impact[0].upper()}{impact[1:]} — {breaking_desc}."
        return f"{impact[0].upper()}{impact[1:]}."

    def _translate_finding(
        self, f: Finding, project_ref: str,
    ) -> str:
        """Translate a finding into plain language using actual metric data.

        ``project_ref`` is woven into the description (e.g.
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
            # Use actual metrics instead of vague "things slow down"
            parts = []
            if f._execution_time_ms > 0:
                parts.append(
                    f"response time hit {_format_ms(f._execution_time_ms)}"
                )
            if f._peak_memory_mb > 0:
                parts.append(f"memory reached {f._peak_memory_mb:.0f}MB")
            if f._error_count > 0:
                parts.append(_describe_errors(f))
            if parts:
                return f"When {ctx}, {' and '.join(parts)}."
            return f"When {ctx}, the system struggled under load."

        if f.category == "edge_case_input":
            # Identify what kind of cap/failure from details
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
                    f"When {ctx}, {_describe_errors(f)} "
                    f"at the highest load."
                )
            # Use actual metrics
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
            ctx = activity or f"{project_ref} is running over time"
            if f._peak_memory_mb > 0:
                return (
                    f"During {ctx}, memory grows to "
                    f"{f._peak_memory_mb:.0f}MB and keeps climbing."
                )
            return (
                f"During {ctx}, memory keeps growing and could "
                f"eventually run out."
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
                parts.append(_describe_errors(f))
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

        if not critical and not warnings:
            parts.append(
                f"All {report.scenarios_run} stress test scenarios completed "
                f"without issues."
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

        # Context
        if report.operational_context:
            parts.append(
                f"Results assessed relative to: {report.operational_context}"
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

        user_parts.append(
            f"\n## Test Results\n"
            f"Scenarios run: {report.scenarios_run}\n"
            f"Passed: {report.scenarios_passed}\n"
            f"Failed: {report.scenarios_failed}\n"
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
                f"\n## Untested Dependencies\n"
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
    def _summarize_errors(sr: ScenarioResult) -> str:
        """Summarize errors from a scenario result."""
        error_types: dict[str, int] = {}
        for step in sr.steps:
            for err in step.errors:
                etype = err.get("type", "Unknown") if isinstance(err, dict) else "Unknown"
                error_types[etype] = error_types.get(etype, 0) + 1

        if error_types:
            parts = [f"{t}: {c}" for t, c in error_types.items()]
            return "; ".join(parts[:5])
        return f"{sr.total_errors} error(s)"

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
        """Extract a groupable pattern from a finding's title prefix."""
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

        # Group by (category, pattern)
        buckets: dict[tuple[str, str], list[Finding]] = defaultdict(list)
        for f in findings:
            pattern = ReportGenerator._finding_pattern(f)
            key = (f.category, pattern)
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
                if len(cluster) == 1:
                    result.append(cluster[0])
                else:
                    representative = cluster[0]
                    representative.grouped_findings = cluster[1:]
                    representative.group_count = len(cluster)
                    result.append(representative)

        return result

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


# ── Module Helpers ──


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

    return ""


def _describe_step(step_name: str) -> str:
    """Translate a step name into user terms.

    Returns an empty string if the step name can't be meaningfully
    translated (e.g. edge cases, generic iterations).
    """
    import re

    m = re.match(r"data_size_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} items"

    m = re.match(r"concurrent_(\d+)", step_name)
    if m:
        return f"{int(m.group(1)):,} simultaneous users"

    m = re.match(r"batch_(\d+)", step_name)
    if m:
        return f"batch {int(m.group(1)):,}"

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

    return ""


def _describe_impact(metric: str, first_val: float, last_val: float) -> str:
    """Describe the impact of a degradation in real terms.

    For timing: uses _human_time to convert both endpoints to natural
    language so users see "from fast to very slow" instead of "204x".

    For memory: projects multi-user impact when peak is significant.
    """
    if metric == "execution_time_ms":
        first_desc = _human_time(first_val)
        last_desc = _human_time(last_val)
        if first_desc == last_desc:
            # Same band — fall back to concrete values
            return (
                f"response time goes from {_format_ms(first_val)} "
                f"to {_format_ms(last_val)}"
            )
        return (
            f"response time goes from {first_desc} "
            f"to {last_desc}"
        )
    if metric == "memory_peak_mb":
        base = (
            f"memory grows from {first_val:.0f}MB to {last_val:.0f}MB"
        )
        # Project multi-user impact for significant memory
        if last_val >= 50:
            projected = last_val * 10
            base += (
                f" — if 10 users are active, "
                f"that's {projected:.0f}MB on your server"
            )
        return base
    if metric == "error_count":
        return f"errors jump from {int(first_val)} to {int(last_val)}"
    return f"{_human_metric(metric)} increases significantly"


def _format_ms(ms: float) -> str:
    """Format a millisecond value as a concrete human-readable string."""
    if ms < 1:
        return f"{ms:.2f}ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60000:.1f}min"


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


def _extract_project_ref(operational_intent: str) -> str:
    """Extract a short noun phrase from the user's operational intent.

    Used to reference the project naturally (e.g. "your budget tracker"
    instead of "your project").

    Truncates at the first arrow, period, dash (surrounded by spaces),
    newline, comma, or semicolon and strips parenthetical framework
    references like "(Flask)".
    """
    import re

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
