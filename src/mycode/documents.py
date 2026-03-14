"""Downloadable document generation and edition tracking.

Produces two markdown documents from a DiagnosticReport:
  1. "Understanding Your Results" — human-readable diagnostic report
  2. "Recommended Fixes" — agent-parseable investigation directives

Edition counter persists in ~/.mycode/editions/ keyed by project path hash.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from mycode.report import DiagnosticReport, Finding

logger = logging.getLogger(__name__)

# ── Edition Counter ──

_EDITIONS_DIR = Path.home() / ".mycode" / "editions"
_REPORTS_DIR = Path.home() / ".mycode" / "reports"


def _hash_key(raw: str) -> str:
    """SHA-256 hash of a normalized string, used as edition file key."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_github_url(url: str) -> str:
    """Normalize a GitHub URL for consistent hashing.

    Strips .git suffix, trailing slashes, lowercases, removes protocol.
    """
    url = url.strip().lower()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return f"{parsed.netloc}{path}"


def get_next_edition(project_path: Optional[Path] = None,
                     github_url: Optional[str] = None) -> int:
    """Increment and return the next edition number for a project.

    Args:
        project_path: Local project path (CLI mode).
        github_url: GitHub URL (web mode).

    Returns:
        The new edition number (starting from 1).
    """
    if github_url:
        key_input = _normalize_github_url(github_url)
    elif project_path:
        key_input = str(project_path.resolve())
    else:
        return 1

    file_hash = _hash_key(key_input)
    edition_file = _EDITIONS_DIR / f"{file_hash}.json"

    current = 0
    try:
        if edition_file.exists():
            data = json.loads(edition_file.read_text(encoding="utf-8"))
            current = data.get("edition", 0)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Could not read edition file: %s", exc)

    new_edition = current + 1

    try:
        _EDITIONS_DIR.mkdir(parents=True, exist_ok=True)
        edition_file.write_text(
            json.dumps({
                "key": key_input,
                "edition": new_edition,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write edition file: %s", exc)

    return new_edition


# ── Project Name Sanitization ──


def _sanitize_dirname(name: str) -> str:
    """Sanitize a project name for use as a directory name."""
    name = name.strip().lower()
    name = re.sub(r"[@/\\]", "-", name)
    name = re.sub(r"[^a-z0-9._-]", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name or "project"


# ── Document 1: Understanding Your Results ──


def render_understanding(report: DiagnosticReport, edition: int) -> str:
    """Render the human-readable diagnostic report document."""
    lines: list[str] = []

    # Header
    lines.append("# myCode")
    lines.append("")
    lines.append("*Stress test your AI-generated code before it breaks*")
    lines.append("")
    lines.append(f"**Edition {edition}**")
    lines.append("")

    # Project summary
    if report.project_description:
        lines.append("## Project Summary")
        lines.append("")
        lines.append(report.project_description)
        lines.append("")

    # Dependency stack
    _render_dependency_stack(lines, report)

    # Test overview
    lines.append("## Test Overview")
    lines.append("")
    lines.append(
        f"- Scenarios run: {report.scenarios_run}"
    )
    lines.append(f"- Passed: {report.scenarios_passed}")
    lines.append(f"- Failed: {report.scenarios_failed}")
    if report.scenarios_incomplete:
        lines.append(
            f"- Could not test: {report.scenarios_incomplete}"
        )
    if report.total_errors:
        lines.append(f"- Total errors: {report.total_errors}")
    lines.append("")

    # Findings grouped by severity
    all_findings = list(report.findings) + list(report.incomplete_tests)
    severity_groups = {"critical": [], "warning": [], "info": []}
    for f in all_findings:
        severity_groups.get(f.severity, severity_groups["info"]).append(f)

    for severity in ("critical", "warning", "info"):
        group = severity_groups[severity]
        if not group:
            continue
        label = severity.upper()
        lines.append(f"## {label} ({len(group)})")
        lines.append("")
        for f in group:
            _render_understanding_finding(lines, f)

    # Degradation curves
    if report.degradation_points:
        lines.append("## Performance Degradation")
        lines.append("")
        for dp in report.degradation_points:
            lines.append(f"### {dp.scenario_name}")
            lines.append("")
            if dp.description:
                lines.append(dp.description)
                lines.append("")
            if dp.breaking_point:
                lines.append(f"Breaking point: **{dp.breaking_point}**")
                lines.append("")

    # Confidence note
    if report.confidence_note:
        lines.append("---")
        lines.append("")
        lines.append(f"*{report.confidence_note}*")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("myCode by Machine Adjacent Systems")
    lines.append(
        "Diagnostic tool — does not guarantee code correctness, "
        "security, or fitness for purpose"
    )
    lines.append("")

    return "\n".join(lines)


def _render_dependency_stack(lines: list[str], report: DiagnosticReport) -> None:
    """Render the dependency stack section."""
    has_deps = report.recognized_dep_count > 0 or report.unrecognized_deps
    if not has_deps:
        return

    lines.append("## Dependency Stack")
    lines.append("")
    if report.recognized_dep_count:
        lines.append(
            f"- {report.recognized_dep_count} dependencies with targeted stress profiles"
        )
    if report.unrecognized_deps:
        lines.append(
            f"- {len(report.unrecognized_deps)} dependencies tested with "
            f"usage-based analysis: {', '.join(report.unrecognized_deps[:10])}"
            + ("..." if len(report.unrecognized_deps) > 10 else "")
        )
    lines.append("")


def _render_understanding_finding(lines: list[str], f: Finding) -> None:
    """Render a single finding for the understanding document."""
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


# ── Document 2: Recommended Fixes ──


def render_fixes(report: DiagnosticReport, edition: int) -> str:
    """Render the agent-parseable investigation document.

    Findings are grouped by source file, ordered by highest severity
    first. Within each file block, all findings appear with severity tags.
    Info findings are framed as context, not investigation targets.
    """
    lines: list[str] = []

    # Attribution in HTML comment (hidden from coding agents)
    lines.append(
        "<!-- Generated by myCode | Machine Adjacent Systems "
        f"| Edition {edition} -->"
    )
    lines.append("")

    # Collect all findings (regular + incomplete)
    all_findings = list(report.findings) + list(report.incomplete_tests)

    if not all_findings:
        lines.append("No findings to investigate.")
        lines.append("")
        return "\n".join(lines)

    # Group by source file
    file_groups: dict[str, list[Finding]] = {}
    for f in all_findings:
        key = f.source_file or "(no file identified)"
        file_groups.setdefault(key, []).append(f)

    # Sort files by highest severity finding (critical > warning > info)
    severity_rank = {"critical": 0, "warning": 1, "info": 2}

    def _file_priority(file_findings: list[Finding]) -> int:
        return min(
            severity_rank.get(f.severity, 9) for f in file_findings
        )

    sorted_files = sorted(
        file_groups.items(),
        key=lambda item: (_file_priority(item[1]), item[0]),
    )

    for file_path, findings in sorted_files:
        lines.append(f"## {file_path}")
        lines.append("")

        # Sort findings within file: critical > warning > info
        findings.sort(key=lambda f: severity_rank.get(f.severity, 9))

        for f in findings:
            _render_fix_finding(lines, f)

    return "\n".join(lines)


def _render_fix_finding(lines: list[str], f: Finding) -> None:
    """Render a single finding for the fixes document."""
    severity_tag = f"[{f.severity.upper()}]"
    lines.append(f"### {severity_tag} {f.title}")
    lines.append("")

    if f.source_function:
        lines.append(f"- **Function:** `{f.source_function}`")
    if f.category:
        lines.append(f"- **Category:** {f.category}")
    if f.affected_dependencies:
        lines.append(
            f"- **Dependencies:** {', '.join(f.affected_dependencies)}"
        )
    if f._load_level is not None:
        lines.append(f"- **Failed at load level:** {f._load_level}")
    lines.append("")

    if f.description:
        lines.append(f.description)
        lines.append("")
    if f.details:
        lines.append(f"```\n{f.details}\n```")
        lines.append("")

    # Investigation prompt — scoped directive for coding agents
    if f.severity == "info":
        # Info findings are context, not investigation targets
        lines.append(
            f"*Context: This finding is informational. "
            f"No investigation required unless related issues appear above.*"
        )
    else:
        prompt = _build_investigation_prompt(f)
        lines.append(f"**Investigate:** {prompt}")
    lines.append("")


def _build_investigation_prompt(f: Finding) -> str:
    """Build a scoped investigation prompt for a finding.

    Uses finding-specific data (metrics, thresholds, observed values)
    to generate targeted investigation guidance. Tells the agent what
    to look at and what behaviour to verify, NOT what code to write.
    """
    parts: list[str] = []

    # 1. Location
    location = ""
    if f.source_function and f.source_file:
        location = f"`{f.source_function}` in `{f.source_file}`"
    elif f.source_file:
        location = f"`{f.source_file}`"
    elif f.source_function:
        location = f"`{f.source_function}`"
    elif f.affected_dependencies:
        location = f"code using {', '.join(f.affected_dependencies[:3])}"

    if location:
        parts.append(f"Examine {location}.")

    # 2. Finding-specific guidance derived from actual data
    specific = _finding_specific_guidance(f)
    if specific:
        parts.append(specific)
    else:
        # 3. Category fallback (only when no specific guidance)
        fallback = _category_fallback_guidance(f.category)
        if fallback:
            parts.append(fallback)
        elif f.description:
            parts.append(
                "Verify that the observed behaviour does not occur "
                "under the conditions described above."
            )

    return " ".join(parts) if parts else (
        "Review the code path identified above and verify correct "
        "behaviour under the conditions described."
    )


def _finding_specific_guidance(f: Finding) -> str:
    """Generate investigation guidance from the finding's actual data.

    Returns empty string if no specific guidance can be derived.
    """
    parts: list[str] = []

    # Memory-related findings
    if f._peak_memory_mb and f._peak_memory_mb > 0:
        if f._finding_type == "resource_limit_hit":
            parts.append(
                f"Memory peaked at {f._peak_memory_mb:.0f} MB. "
                f"Check for unbounded data structures, large object "
                f"allocations, or caches that grow with each request."
            )
        elif f._peak_memory_mb > 100:
            parts.append(
                f"Memory reached {f._peak_memory_mb:.0f} MB. "
                f"Look for objects held in memory across requests — "
                f"module-level caches, connection pools, or accumulated "
                f"state that is never released."
            )

    # Execution time findings
    if f._execution_time_ms and f._execution_time_ms > 0:
        if f._execution_time_ms > 5000:
            parts.append(
                f"Response time reached {f._execution_time_ms:.0f} ms. "
                f"Investigate what causes the latency — synchronous "
                f"operations, unoptimized database queries, blocking I/O, "
                f"or missing connection pooling."
            )
        elif f._execution_time_ms > 1000:
            parts.append(
                f"Response time reached {f._execution_time_ms:.0f} ms. "
                f"Check for synchronous middleware, N+1 query patterns, "
                f"or operations that scale with request count."
            )

    # Error-heavy findings
    if f._error_count and f._error_count > 5 and not parts:
        parts.append(
            f"{f._error_count} errors occurred during testing. "
            f"Check error handling paths — unhandled exceptions, "
            f"missing error boundaries, or cascading failures."
        )

    # Load-level context
    if f._load_level is not None and f._load_level > 0:
        if f._finding_type == "scenario_failed":
            parts.append(
                f"Failed at load level {f._load_level}. "
                f"Verify behaviour at and below this threshold."
            )

    # HTTP-specific findings — use description to extract specifics
    if f.category == "http_load_testing" and not parts:
        desc_lower = (f.description or "").lower()
        if "memory" in desc_lower or "baseline" in desc_lower:
            parts.append(
                "Check what the application loads at startup — "
                "large bundles, unoptimized imports, in-memory data "
                "stores, or objects held per process."
            )
        elif "response time" in desc_lower or "latency" in desc_lower:
            parts.append(
                "Investigate what causes response time to increase "
                "under concurrent load — synchronous middleware, "
                "blocking operations, connection pool exhaustion, "
                "or per-request computation that doesn't scale."
            )
        elif "error" in desc_lower or "crash" in desc_lower:
            parts.append(
                "Check for unhandled exceptions under concurrent "
                "access — race conditions, shared state corruption, "
                "or resource limits (file descriptors, connections)."
            )

    return " ".join(parts)


def _category_fallback_guidance(category: str) -> str:
    """Return generic category guidance when no specific data is available."""
    fallbacks = {
        "data_volume_scaling": (
            "Verify that the code handles progressively larger inputs "
            "without unbounded memory growth or excessive processing time."
        ),
        "memory_profiling": (
            "Check for objects that accumulate across repeated calls "
            "and are never released (caches, listeners, connection pools)."
        ),
        "edge_case_inputs": (
            "Verify that the code validates and handles malformed, "
            "empty, or unexpected-type inputs without crashing."
        ),
        "concurrent_execution": (
            "Check for shared mutable state, missing locks, "
            "or race conditions under concurrent access."
        ),
        "blocking_io": (
            "Verify that blocking I/O operations do not stall "
            "the application under concurrent load."
        ),
        "gil_contention": (
            "Check whether CPU-bound work blocks the GIL and "
            "prevents concurrent request handling."
        ),
        "async_failures": (
            "Verify that async/promise chains handle rejections "
            "and don't leak unresolved promises."
        ),
        "http_load_testing": (
            "Verify that the endpoint handles concurrent requests "
            "without excessive response time degradation or errors."
        ),
    }
    return fallbacks.get(category, "")


# ── File Output ──


def write_edition_documents(
    report: DiagnosticReport,
    project_name: str,
    project_path: Optional[Path] = None,
    github_url: Optional[str] = None,
) -> tuple[Path, Path, int]:
    """Generate both documents and write them to ~/.mycode/reports/.

    Args:
        report: The diagnostic report to render.
        project_name: Human-readable project name.
        project_path: Local project path (CLI mode).
        github_url: GitHub URL (web mode).

    Returns:
        (path_to_understanding, path_to_fixes, edition_number)

    Raises:
        OSError: If the output directory cannot be created.
    """
    edition = get_next_edition(
        project_path=project_path,
        github_url=github_url,
    )

    safe_name = _sanitize_dirname(project_name)
    output_dir = _REPORTS_DIR / safe_name / f"edition-{edition}"
    output_dir.mkdir(parents=True, exist_ok=True)

    understanding = render_understanding(report, edition)
    fixes = render_fixes(report, edition)

    understanding_path = (
        output_dir / f"mycode-understanding-your-results-edition-{edition}.md"
    )
    fixes_path = (
        output_dir / f"mycode-recommended-fixes-edition-{edition}.md"
    )

    understanding_path.write_text(understanding, encoding="utf-8")
    fixes_path.write_text(fixes, encoding="utf-8")

    return understanding_path, fixes_path, edition
