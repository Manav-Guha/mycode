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
    """

    scenario_name: str
    metric: str
    steps: list[tuple[str, float]] = field(default_factory=list)
    breaking_point: str = ""
    description: str = ""


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
    findings: list[Finding] = field(default_factory=list)
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
        if self.findings:
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

        # Degradation
        if self.degradation_points:
            sections.append("\n" + "-" * 40)
            sections.append("  Degradation Curves")
            sections.append("-" * 40)
            for dp in self.degradation_points:
                sections.append(f"\n  {dp.scenario_name} ({dp.metric}):")
                if dp.description:
                    sections.append(f"    {dp.description}")
                if dp.steps:
                    for label, value in dp.steps:
                        sections.append(f"      {label}: {value:.2f}")
                if dp.breaking_point:
                    sections.append(
                        f"    >> Breaking point: {dp.breaking_point}"
                    )

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

        # 3b. Group similar findings to reduce noise
        report.findings = self._group_similar_findings(report.findings)

        # 4. Sort findings by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        report.findings.sort(key=lambda f: severity_order.get(f.severity, 9))

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
            "4. Be direct about what broke and where.\n"
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
