"""Downloadable document generation and edition tracking.

Produces "Understanding Your Results" — a human-readable diagnostic report
with embedded coding agent prompts for each actionable finding.

Output format: PDF when fpdf2 is installed, markdown as fallback.
Edition counter persists in ~/.mycode/editions/ keyed by project path hash.
"""

import datetime as _dt
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from mycode.report import (
    DegradationPoint,
    DiagnosticReport,
    Finding,
    _breaking_point_label,
    _build_degradation_narrative,
    _describe_scenario,
    _describe_step,
    _format_ms,
    _humanize_scenario_name,
    _metric_label,
)

logger = logging.getLogger(__name__)


def _partition_http_tested(
    incomplete: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """Split incomplete tests into http_tested and other findings."""
    http_tested: list[Finding] = []
    other: list[Finding] = []
    for f in incomplete:
        if f._failure_reason == "http_tested":
            http_tested.append(f)
        else:
            other.append(f)
    return http_tested, other


def _http_tested_summary(http_tested: list[Finding]) -> str:
    """Build a single summary line for http_tested findings.

    Returns e.g. "6 Streamlit scenarios were tested via HTTP load testing."
    """
    n = len(http_tested)
    # Extract framework name from affected_dependencies
    framework = ""
    for f in http_tested:
        if f.affected_dependencies:
            framework = f.affected_dependencies[0]
            break
    if framework:
        return (
            f"{n} {framework} scenario{'s' if n != 1 else ''} "
            f"{'were' if n != 1 else 'was'} tested via HTTP load testing."
        )
    return (
        f"{n} framework scenario{'s' if n != 1 else ''} "
        f"{'were' if n != 1 else 'was'} tested via HTTP load testing."
    )


# ── Performance Summary Table ──


def _fmt_val(value: float, metric: str) -> str:
    """Format a degradation value with its unit."""
    is_time = metric in ("execution_time_ms", "response_time_ms")
    is_memory = metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")
    if is_time:
        return _format_ms(value)
    if is_memory:
        return f"{value:.0f}MB" if value >= 1 else f"{value:.2f}MB"
    if metric == "error_count":
        n = int(value)
        return f"{n} error{'s' if n != 1 else ''}"
    return f"{value:.1f}"


def _fmt_cell(value: float, label: str, metric: str) -> str:
    """Format a table cell: value (context)."""
    val = _fmt_val(value, metric)
    ctx = _describe_step(label) or label
    return f"{val} ({ctx})"


def _verdict(dp: DegradationPoint) -> str:
    """One-phrase verdict for a degradation curve."""
    if not dp.steps:
        return ""
    last_val = dp.steps[-1][1]
    is_time = dp.metric in ("execution_time_ms", "response_time_ms")
    is_memory = dp.metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")

    if is_time:
        if last_val < 100:
            return "No issues"
        bp_desc = _describe_step(dp.breaking_point) if dp.breaking_point else ""
        if last_val < 500:
            return f"Noticeable above {bp_desc}" if bp_desc else "Noticeable at peak"
        if last_val < 2000:
            return "Slow at peak load"
        return "Unresponsive at peak load"

    if is_memory:
        if last_val < 50:
            return "Fine at your scale"
        if last_val < 200:
            return "Heavy — limits concurrent users"
        return "Very heavy — risk of crashes"

    if dp.metric == "error_count":
        if last_val <= 0:
            return "No errors"
        return f"{int(last_val)} errors at peak"

    if "stable" in (dp.description or "").lower():
        return "Stable"

    return ""


def _perf_row_label(dp: DegradationPoint) -> str:
    """Human label for the 'What we tested' column."""
    name = _describe_scenario(dp.scenario_name) or dp.scenario_name
    is_memory = dp.metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")
    if is_memory:
        # Prefix with "Memory usage —" to distinguish from time curves
        # for the same scenario
        return f"Memory usage — {name}"
    return name.capitalize() if name and name[0].islower() else name


def _dedup_by_label(points: list[DegradationPoint]) -> list[DegradationPoint]:
    """Group degradation points by label, keep worst-case representative.

    When multiple curves produce the same "What we tested" label (e.g.,
    six coupling scenarios all labelled "Running calculations across
    components"), keep the one with the highest peak value.
    """
    if not points:
        return []

    groups: dict[str, DegradationPoint] = {}
    for dp in points:
        if not dp.steps:
            continue
        label = _perf_row_label(dp)
        existing = groups.get(label)
        if existing is None:
            groups[label] = dp
        else:
            # Keep the one with the higher peak value (worse case)
            existing_peak = existing.steps[-1][1] if existing.steps else 0
            new_peak = dp.steps[-1][1]
            if new_peak > existing_peak:
                groups[label] = dp

    return list(groups.values())


def _row_cells(dp: DegradationPoint) -> tuple[str, str, str, str, str]:
    """Extract (label, low, mid, high, verdict) for a table row."""
    label = _perf_row_label(dp)
    steps = dp.steps

    low_label, low_val = steps[0]
    low = _fmt_cell(low_val, low_label, dp.metric)

    if len(steps) >= 3:
        mid_idx = len(steps) // 2
        mid_label, mid_val = steps[mid_idx]
        mid = _fmt_cell(mid_val, mid_label, dp.metric)
    else:
        mid = "\u2014"

    if len(steps) >= 2:
        high_label, high_val = steps[-1]
        high = _fmt_cell(high_val, high_label, dp.metric)
    else:
        high = "\u2014"

    return label, low, mid, high, _verdict(dp)


def _render_perf_table_md(lines: list[str], points: list[DegradationPoint]) -> None:
    """Render degradation points as a markdown performance summary table."""
    rows = _dedup_by_label(points)
    if not rows:
        return

    lines.append("## Performance Under Load")
    lines.append("")
    lines.append(
        "| What we tested | At low load | At mid load | At peak load | Verdict |"
    )
    lines.append("|---|---|---|---|---|")

    for dp in rows:
        label, low, mid, high, v = _row_cells(dp)
        lines.append(f"| {label} | {low} | {mid} | {high} | {v} |")

    lines.append("")


def _verdict_color(verdict: str) -> tuple:
    """Return text colour for a verdict string."""
    vl = verdict.lower()
    if "no issues" in vl or "fine" in vl or "no errors" in vl or "stable" in vl:
        return _GREEN
    if "noticeable" in vl:
        return _AMBER_TEXT
    if "slow" in vl or "heavy" in vl or "unresponsive" in vl or "error" in vl:
        return _RED
    return _BODY


def _render_perf_table_pdf(pdf, points: list[DegradationPoint]) -> None:
    """Render degradation points as a styled PDF performance summary table."""
    rows = _dedup_by_label(points)
    if not rows:
        return

    pdf.section_heading("Performance Under Load")
    pdf.ln(2)

    # Column widths (mm) — total 170mm for A4 with 20mm margins
    col_w = [45, 30, 30, 30, 35]
    headers = ["What we tested", "At low load", "At mid load", "At peak load", "Verdict"]
    row_h = 7
    cell_pad_x = 2  # mm horizontal padding inside cells

    # Header row — dark blue background, white text
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*_WHITE)
    pdf.set_fill_color(*_BRAND)
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.18)
    for i, hdr in enumerate(headers):
        pdf.cell(col_w[i], row_h, f"  {_safe_text(hdr)}", border=1, fill=True)
    pdf.ln(row_h)

    # Data rows — alternating white/grey
    for row_idx, dp in enumerate(rows):
        label, low, mid, high, v = _row_cells(dp)
        is_alt = row_idx % 2 == 1

        if is_alt:
            pdf.set_fill_color(*_TABLE_ALT)
        else:
            pdf.set_fill_color(*_WHITE)

        pdf.set_draw_color(*_RULE)
        pdf.set_line_width(0.18)

        cells = [label, low, mid, high]
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_BODY)
        for i, cell_text in enumerate(cells):
            pdf.cell(
                col_w[i], row_h, f"  {_safe_text(cell_text)}",
                border=1, fill=True,
            )

        # Verdict column — bold, coloured
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_verdict_color(v))
        pdf.cell(
            col_w[4], row_h, f"  {_safe_text(v)}",
            border=1, fill=True,
        )
        pdf.ln(row_h)

    # Reset
    pdf.set_text_color(*_BODY)
    pdf.set_line_width(0.35)
    pdf.ln(6)


# ── PDF Library (optional) ──

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False

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


def _sanitize_filename(name: str) -> str:
    """Sanitize a project name for use in a PDF filename.

    Preserves case, converts spaces to hyphens, strips special chars.
    """
    name = name.strip()
    name = re.sub(r"[\s]+", "-", name)
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    return name or "Project"


def pdf_filename(project_name: str, doc_type: str, date: str = "") -> str:
    """Generate a PDF filename.

    Args:
        project_name: Human-readable project name.
        doc_type: "Results" or "Fixes".
        date: ISO date string (defaults to today).

    Returns:
        e.g. "myCode-Financial-Dashboard-Results-2026-03-19.pdf"
    """
    if not date:
        date = _dt.date.today().isoformat()
    safe = _sanitize_filename(project_name)
    return f"myCode-{safe}-{doc_type}-{date}.pdf"


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

    # Partition http_tested out of incomplete tests
    http_tested, other_incomplete = _partition_http_tested(
        report.incomplete_tests,
    )

    # Findings grouped by severity
    all_findings = list(report.findings) + other_incomplete
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

    # HTTP-tested summary (single line, not individual findings)
    if http_tested:
        lines.append(f"*{_http_tested_summary(http_tested)}*")
        lines.append("")

    # Performance summary table
    _render_perf_table_md(lines, report.degradation_points)

    # Confidence note
    if report.confidence_note:
        lines.append("---")
        lines.append("")
        lines.append(f"*{report.confidence_note}*")
        lines.append("")

    # JSON download callout
    has_actionable = any(
        f.severity in ("critical", "warning") for f in report.findings
    )
    if has_actionable:
        lines.append("---")
        lines.append("")
        lines.append(
            "**Download the JSON report and attach it when you paste the "
            "prompts above into your coding agent. The JSON contains the "
            "full diagnostic data your agent needs.**"
        )
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
        rec_names = ""
        if report.recognized_dep_names:
            rec_names = f": {', '.join(report.recognized_dep_names[:10])}"
        lines.append(
            f"- {report.recognized_dep_count} dependencies with targeted stress profiles{rec_names}"
        )
    if report.unrecognized_deps:
        lines.append(
            f"- {len(report.unrecognized_deps)} dependencies tested with "
            f"usage-based analysis: {', '.join(report.unrecognized_deps[:10])}"
            + ("..." if len(report.unrecognized_deps) > 10 else "")
        )
    lines.append("")


def _render_understanding_finding(lines: list[str], f: Finding) -> None:
    """Render a single finding for the understanding document.

    Critical/warning findings get the full 5-part layout:
      What we found / What this means / What to do / Prompt / After you fix it
    Info findings get only "What we found" with no action items.
    """
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    lines.append(f"### {title}")
    lines.append("")

    # What we found
    lines.append("**What we found:**")
    lines.append("")
    if f.description:
        lines.append(f.description)
        lines.append("")
    if f.details:
        lines.append(f"> {f.details}")
        lines.append("")

    # INFO findings: description only, no action items
    if f.severity == "info":
        if f.affected_dependencies:
            lines.append(
                f"**Related dependencies:** {', '.join(f.affected_dependencies)}"
            )
            lines.append("")
        return

    # What this means for you (critical/warning only)
    consequence = _consequence_for_user(f)
    if consequence:
        lines.append("**What this means for you:**")
        lines.append("")
        lines.append(consequence)
        lines.append("")

    # What to do
    lines.append("**What to do:**")
    lines.append("")
    lines.append(
        "Copy the prompt below and paste it into your coding agent "
        "(Claude Code, Cursor, Copilot) along with the JSON report."
    )
    lines.append("")

    # Prompt
    prompt = generate_finding_prompt(f)
    if prompt:
        lines.append("**Prompt:**")
        lines.append("")
        lines.append("```")
        lines.append(prompt)
        lines.append("```")
        lines.append("")

    # After you fix it
    lines.append("**After you fix it:**")
    lines.append("")
    lines.append("Run myCode again to verify the fix worked.")
    lines.append("")


def _consequence_for_user(f: Finding) -> str:
    """Generate a consequence line from the finding's data.

    Uses existing description which already contains constraint-aware
    language from the contextualise step.
    """
    desc = f.description or ""

    # The contextualise step already prepends constraint framing like
    # "This is below your expected 20 concurrent users" — if present,
    # the description IS the consequence.  Extract the consequence
    # portion for findings that have it.
    consequence_markers = (
        "this means", "in practice,", "your users will",
        "your app will", "this is below your expected",
        "this is at or below", "this is well below",
        "memory usage grows", "could exhaust",
    )
    for sentence in desc.split(". "):
        if any(m in sentence.lower() for m in consequence_markers):
            return sentence.strip().rstrip(".") + "."

    # Fallback: derive from metrics
    if f._peak_memory_mb and f._peak_memory_mb > 100:
        return (
            f"Memory usage of {f._peak_memory_mb:.0f}MB under load could "
            f"exhaust server memory, causing crashes or forced restarts."
        )
    if f._execution_time_ms and f._execution_time_ms > 2000:
        return (
            f"Response times of {f._execution_time_ms:.0f}ms mean your "
            f"users will experience significant delays."
        )
    if f._error_count and f._error_count > 5:
        return (
            f"{f._error_count} errors occurred during testing — under "
            f"real traffic, some users will see failures."
        )

    return ""


def generate_finding_prompt(f: Finding) -> str:
    """Generate a ready-to-paste prompt for a coding agent from a finding.

    Returns empty string for INFO findings (no action needed).
    """
    if f.severity == "info":
        return ""

    parts: list[str] = []
    parts.append(f"myCode found a {f.severity.upper()} issue.")

    if f.source_file:
        parts.append(f"File: {f.source_file}")
    if f.source_function:
        parts.append(f"Function: {f.source_function}")

    # Problem description
    problem = f.title
    if f.description:
        # Take first sentence only to keep it concise
        first_sentence = f.description.split(". ")[0].rstrip(".")
        problem += f" — {first_sentence}."
    if f._load_level is not None:
        problem += f" At load level {f._load_level}."
    parts.append(f"Problem: {problem}")

    if f.affected_dependencies:
        parts.append(f"Dependencies: {', '.join(f.affected_dependencies)}")

    # Investigation guidance
    guidance = _build_investigation_prompt(f)
    parts.append(f"Investigate: {guidance}")

    # Fix objective
    fix_goal = _fix_objective(f)
    parts.append(f"The fix should: {fix_goal}")

    parts.append("The attached JSON contains the full diagnostic data.")

    return "\n".join(parts)


_FIX_OBJECTIVES: dict[str, str] = {
    "memory_profiling": (
        "eliminate unbounded memory growth so the function stays "
        "within safe limits under repeated calls."
    ),
    "concurrent_execution": (
        "ensure thread safety so the function behaves correctly "
        "under concurrent access."
    ),
    "data_volume_scaling": (
        "handle larger inputs without excessive memory or time growth."
    ),
    "edge_case_inputs": (
        "validate and handle malformed or unexpected inputs without crashing."
    ),
    "blocking_io": (
        "avoid blocking the event loop or main thread under concurrent load."
    ),
    "gil_contention": (
        "reduce GIL contention so CPU-bound work doesn't block "
        "concurrent request handling."
    ),
    "async_failures": (
        "handle async rejections and avoid leaking unresolved promises."
    ),
    "http_load_testing": (
        "handle concurrent HTTP load without crashes, error spikes, "
        "or excessive response time."
    ),
}


def _fix_objective(f: Finding) -> str:
    """Return a one-sentence fix goal based on the finding's category."""
    return _FIX_OBJECTIVES.get(
        f.category,
        "resolve the issue described above so the function behaves "
        "correctly under the tested conditions.",
    )


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


# ── PDF Rendering ──

# Colour palette
_BRAND = (26, 58, 92)            # #1a3a5c — brand, H1
_HEADING = (51, 51, 51)          # #333333 — section headings, body
_BODY = (51, 51, 51)             # #333333
_SUBTLE = (136, 136, 136)        # #888888 — tagline, meta, footer
_RULE = (204, 204, 204)          # #cccccc — horizontal rules, table borders
_RED = (220, 53, 69)             # #dc3545 — critical
_AMBER = (255, 193, 7)           # #ffc107 — warning badge
_AMBER_TEXT = (133, 100, 4)      # #856404 — warning verdict text
_GREEN = (40, 167, 69)           # #28a745 — positive verdict
_BLUE = (13, 110, 253)           # #0d6efd — info badge
_CALLOUT_BG = (234, 244, 251)    # #eaf4fb
_CALLOUT_BORDER = (184, 218, 255)  # #b8daff
_CODE_BG = (245, 245, 245)       # #f5f5f5
_CODE_BORDER = (221, 221, 221)   # #dddddd
_TABLE_ALT = (249, 249, 249)     # #f9f9f9
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

# Keep old names as aliases for any external references
_DARK_BLUE = _BRAND
_BODY_GREY = _BODY
_LIGHT_GREY = _SUBTLE

_SEVERITY_COLORS: dict[str, tuple[tuple, tuple]] = {
    # (background, text)
    "critical": (_RED, _WHITE),
    "warning": (_AMBER, _BLACK),
    "info": (_BLUE, _WHITE),
}

_SEVERITY_BORDER: dict[str, tuple] = {
    "critical": _RED,
    "warning": _AMBER,
    "info": _BLUE,
}


def _safe_text(text: str) -> str:
    """Replace Unicode characters unsupported by built-in Helvetica."""
    return (
        text
        .replace("\u2014", "-")   # em-dash
        .replace("\u2013", "-")   # en-dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u2022", "-")   # bullet
        .replace("\u00b7", "|")   # middle dot
        .replace("\u2026", "...")  # ellipsis
    )


def _make_pdf_class():
    """Create the MyCodePDF class (requires fpdf2)."""
    if not _HAS_FPDF:
        return None

    class MyCodePDF(FPDF):
        """Styled PDF with myCode header and footer on every page.

        Design: generous whitespace, clear hierarchy, subtle containers.
        Professional but accessible — calm and authoritative.
        """

        def header(self):
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*_BRAND)
            self.cell(40, 10, "myCode", new_x="RIGHT")
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*_SUBTLE)
            self.cell(
                0, 10,
                "Stress test your AI-generated code before it breaks",
                align="R",
            )
            self.ln(4)
            # 1pt rule
            self.set_draw_color(*_RULE)
            self.set_line_width(0.35)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(8)

        def footer(self):
            self.set_y(-18)
            # 1pt rule
            self.set_draw_color(*_RULE)
            self.set_line_width(0.35)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(3)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_SUBTLE)
            self.cell(
                0, 4,
                _safe_text("myCode by Machine Adjacent Systems \u2014 Diagnostic tool"),
                new_x="LEFT",
            )
            self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="R")

        def section_heading(self, text: str, level: int = 2):
            """Render a section heading. H1=18pt, H2=14pt, H3=13pt."""
            sizes = {1: 18, 2: 14, 3: 13}
            size = sizes.get(level, 13)
            self.set_font("Helvetica", "B", size)
            self.set_text_color(*(_BRAND if level == 1 else _HEADING))
            lh = size * 0.55  # ~1.2x line height
            self.multi_cell(0, lh, _safe_text(text))
            self.ln(4 if level == 1 else 3)

        def body_text(self, text: str):
            """Render body text. 11pt, 1.4x line height, 4mm gap after."""
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*_BODY)
            self.multi_cell(0, 6, _safe_text(text))
            self.ln(4)

        def body_label(self, text: str):
            """Render a bold inline label (e.g. 'What we found:')."""
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*_HEADING)
            self.cell(0, 6, _safe_text(text))
            self.ln(4)

        def severity_badge(self, severity: str):
            """Render an inline severity badge with padding."""
            bg, fg = _SEVERITY_COLORS.get(severity, (_BLUE, _WHITE))
            label = severity.upper()
            self.set_font("Helvetica", "B", 9)
            w = self.get_string_width(label) + 6
            self.set_fill_color(*bg)
            self.set_text_color(*fg)
            self.cell(w, 6, f" {label} ", fill=True, new_x="RIGHT")
            self.set_text_color(*_BODY)
            self.cell(6, 6, "")  # 6mm spacer

        def code_block(self, text: str):
            """Render a prompt box with border, background, and padding."""
            safe = _safe_text(text[:1000])
            x = self.l_margin
            w = self.w - self.l_margin - self.r_margin
            pad = 8  # mm internal padding

            # Measure height needed
            self.set_font("Courier", "", 9)
            self.set_xy(x + pad, self.get_y())
            start_y = self.get_y()
            # Dry run: measure text height
            self.multi_cell(w - 2 * pad, 4.5, safe, dry_run=True, output="LINES")
            line_count = len(safe.split("\n"))
            # Estimate: each line ~4.5mm, plus wrapping
            est_h = max(line_count * 4.5, 10) + 2 * pad

            # Draw background + border
            box_y = self.get_y()
            self.set_fill_color(*_CODE_BG)
            self.set_draw_color(*_CODE_BORDER)
            self.set_line_width(0.35)
            self.rect(x, box_y, w, est_h, style="FD")

            # Render text inside
            self.set_font("Courier", "", 9)
            self.set_text_color(*_BODY)
            self.set_xy(x + pad, box_y + pad)
            self.multi_cell(w - 2 * pad, 4.5, safe)
            actual_end = self.get_y() + pad

            # If text was longer than estimate, extend
            if actual_end > box_y + est_h:
                real_h = actual_end - box_y
                self.set_fill_color(*_CODE_BG)
                self.set_draw_color(*_CODE_BORDER)
                self.rect(x, box_y, w, real_h, style="FD")
                # Re-render text
                self.set_font("Courier", "", 9)
                self.set_text_color(*_BODY)
                self.set_xy(x + pad, box_y + pad)
                self.multi_cell(w - 2 * pad, 4.5, safe)
                self.set_y(self.get_y() + pad)
            else:
                self.set_y(box_y + est_h)

            self.ln(4)

        def bullet(self, text: str):
            """Render a bullet point with 5mm indent."""
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*_BODY)
            x = self.get_x()
            self.set_x(x + 5)
            self.cell(4, 6, "-")
            self.multi_cell(0, 6, _safe_text(text))
            self.set_x(x)
            self.ln(1)

        def callout_box(self, text: str):
            """Render a callout box with light blue background."""
            safe = _safe_text(text)
            x = self.l_margin
            w = self.w - self.l_margin - self.r_margin
            pad = 8

            # Measure
            self.set_font("Helvetica", "", 10)
            start_y = self.get_y()
            # Estimate height
            lines_est = len(safe) / ((w - 2 * pad) / 2.0)  # rough char estimate
            est_h = max(lines_est * 5, 12) + 2 * pad

            # Draw box
            self.set_fill_color(*_CALLOUT_BG)
            self.set_draw_color(*_CALLOUT_BORDER)
            self.set_line_width(0.35)
            self.rect(x, start_y, w, est_h, style="FD")

            # Render text
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_BRAND)
            self.set_xy(x + pad, start_y + pad)
            self.multi_cell(w - 2 * pad, 5, safe)
            actual_end = self.get_y() + pad

            if actual_end > start_y + est_h:
                real_h = actual_end - start_y
                self.set_fill_color(*_CALLOUT_BG)
                self.set_draw_color(*_CALLOUT_BORDER)
                self.rect(x, start_y, w, real_h, style="FD")
                self.set_font("Helvetica", "", 10)
                self.set_text_color(*_BRAND)
                self.set_xy(x + pad, start_y + pad)
                self.multi_cell(w - 2 * pad, 5, safe)
                self.set_y(self.get_y() + pad)
            else:
                self.set_y(start_y + est_h)

            self.ln(6)

    return MyCodePDF


def render_understanding_pdf(
    report: DiagnosticReport, edition: int, project_name: str = "",
) -> bytes:
    """Render the understanding document as a styled PDF.

    Returns raw PDF bytes.  Requires fpdf2.
    """
    PDFClass = _make_pdf_class()
    if PDFClass is None:
        raise ImportError("fpdf2 is required for PDF generation")

    pdf = PDFClass(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 15, 20)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.section_heading("Understanding Your Results", level=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_SUBTLE)
    date = _dt.date.today().isoformat()
    pdf.cell(0, 5, f"Edition {edition}  |  {date}")
    pdf.ln(10)

    # Project summary
    if report.project_description:
        pdf.section_heading("Project Summary")
        pdf.body_text(report.project_description)
        pdf.ln(2)

    # Dependency stack
    has_deps = report.recognized_dep_count > 0 or report.unrecognized_deps
    if has_deps:
        pdf.section_heading("Dependency Stack")
        if report.recognized_dep_count:
            rec_names = ""
            if report.recognized_dep_names:
                rec_names = f": {', '.join(report.recognized_dep_names[:10])}"
            pdf.bullet(
                f"{report.recognized_dep_count} dependencies with "
                f"targeted stress profiles{rec_names}"
            )
        if report.unrecognized_deps:
            names = ", ".join(report.unrecognized_deps[:10])
            suffix = "..." if len(report.unrecognized_deps) > 10 else ""
            pdf.bullet(
                f"{len(report.unrecognized_deps)} dependencies tested "
                f"with usage-based analysis: {names}{suffix}"
            )
        pdf.ln(2)

    # Test overview
    pdf.section_heading("Test Overview")
    pdf.bullet(f"Scenarios run: {report.scenarios_run}")
    pdf.bullet(f"Passed: {report.scenarios_passed}")
    pdf.bullet(f"Failed: {report.scenarios_failed}")
    if report.scenarios_incomplete:
        pdf.bullet(f"Could not test: {report.scenarios_incomplete}")
    if report.total_errors:
        pdf.bullet(f"Total errors: {report.total_errors}")
    pdf.ln(6)

    # Partition http_tested out of incomplete tests
    http_tested, other_incomplete = _partition_http_tested(
        report.incomplete_tests,
    )

    # Findings grouped by severity
    all_findings = list(report.findings) + other_incomplete
    severity_groups: dict[str, list] = {"critical": [], "warning": [], "info": []}
    for f in all_findings:
        severity_groups.get(f.severity, severity_groups["info"]).append(f)

    for severity in ("critical", "warning", "info"):
        group = severity_groups[severity]
        if not group:
            continue
        pdf.section_heading(f"{severity.upper()} ({len(group)})", level=3)
        pdf.ln(2)
        for f in group:
            _render_pdf_finding(pdf, f)

    # HTTP-tested summary
    if http_tested:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 5, _safe_text(_http_tested_summary(http_tested)))
        pdf.ln(6)

    # Performance summary table
    _render_perf_table_pdf(pdf, report.degradation_points)

    # Confidence note
    if report.confidence_note:
        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 5, _safe_text(report.confidence_note))

    # JSON download callout
    has_actionable = any(
        f.severity in ("critical", "warning") for f in report.findings
    )
    if has_actionable:
        pdf.ln(10)
        pdf.callout_box(
            "Download the JSON report and attach it when you paste the "
            "prompts above into your coding agent. The JSON contains the "
            "full diagnostic data your agent needs."
        )

    return bytes(pdf.output())


def _integrate_details(description: str, details: str) -> str:
    """Merge details into the description as a natural sentence.

    Instead of showing details as a separate grey box, append them
    to the description text for a cleaner read.
    """
    if not details:
        return description
    if not description:
        return details
    # If details look like they're already in the description, skip
    if details.lower()[:20] in description.lower():
        return description
    # Append as additional context
    desc = description.rstrip(". ")
    return f"{desc}. {details.rstrip('. ')}."


def _render_pdf_finding(pdf, f: Finding) -> None:
    """Render a single finding for the understanding PDF.

    Each finding is a card with a coloured left border.
    Critical/warning: full 5-part layout with prompt box.
    Info: "What we found" only.
    """
    border_color = _SEVERITY_BORDER.get(f.severity, _BLUE)
    card_indent = 4  # mm from left margin to content (after border)
    original_margin = pdf.l_margin
    card_start_y = pdf.get_y()

    # Badge + title
    pdf.set_x(original_margin + card_indent)
    pdf.severity_badge(f.severity)
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*_HEADING)
    pdf.cell(0, 7, _safe_text(title))
    pdf.ln(8)

    # Temporarily indent all content inside the card
    pdf.set_left_margin(original_margin + card_indent)

    # What we found — description integrated with details
    pdf.body_label("What we found:")
    combined = _integrate_details(f.description or "", f.details or "")
    if combined:
        pdf.body_text(combined)
    pdf.ln(2)

    # INFO findings: description only
    if f.severity == "info":
        if f.affected_dependencies:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*_SUBTLE)
            pdf.cell(
                0, 4,
                f"Related dependencies: {', '.join(f.affected_dependencies)}",
            )
            pdf.ln(4)
        # Draw left border and restore margin
        card_end_y = pdf.get_y()
        pdf.set_left_margin(original_margin)
        pdf.set_draw_color(*border_color)
        pdf.set_line_width(1.0)
        pdf.line(
            original_margin + 0.5, card_start_y,
            original_margin + 0.5, card_end_y,
        )
        pdf.set_line_width(0.35)
        pdf.ln(10)
        return

    # What this means for you
    consequence = _consequence_for_user(f)
    if consequence:
        pdf.body_label("What this means for you:")
        pdf.body_text(consequence)
        pdf.ln(2)

    # What to do
    pdf.body_label("What to do:")
    pdf.body_text(
        "Copy the prompt below and paste it into your coding agent "
        "(Claude Code, Cursor, Copilot) along with the JSON report."
    )
    pdf.ln(2)

    # Prompt (boxed)
    prompt = generate_finding_prompt(f)
    if prompt:
        pdf.body_label("Prompt:")
        pdf.code_block(prompt)

    # After you fix it — outside the prompt box
    pdf.body_label("After you fix it:")
    pdf.body_text("Run myCode again to verify the fix worked.")

    # Draw left border
    card_end_y = pdf.get_y()
    pdf.set_left_margin(original_margin)
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(1.0)
    pdf.line(
        original_margin + 0.5, card_start_y,
        original_margin + 0.5, card_end_y,
    )
    pdf.set_line_width(0.35)
    pdf.ln(10)


# ── File Output ──


def write_edition_documents(
    report: DiagnosticReport,
    project_name: str,
    project_path: Optional[Path] = None,
    github_url: Optional[str] = None,
) -> tuple[Path, int]:
    """Generate the Understanding Your Results document and write it.

    Args:
        report: The diagnostic report to render.
        project_name: Human-readable project name.
        project_path: Local project path (CLI mode).
        github_url: GitHub URL (web mode).

    Returns:
        (path_to_understanding, edition_number)

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

    if _HAS_FPDF:
        date = _dt.date.today().isoformat()
        u_fname = pdf_filename(project_name, "Results", date)
        understanding_bytes = render_understanding_pdf(report, edition, project_name)
        understanding_path = output_dir / u_fname
        understanding_path.write_bytes(understanding_bytes)
    else:
        understanding = render_understanding(report, edition)
        understanding_path = (
            output_dir / f"mycode-understanding-your-results-edition-{edition}.md"
        )
        understanding_path.write_text(understanding, encoding="utf-8")

    # Clean up old editions — keep only the last 10
    _prune_old_editions(_REPORTS_DIR / safe_name)

    return understanding_path, edition


_MAX_EDITIONS = 10


def _prune_old_editions(project_dir: Path, max_keep: int = _MAX_EDITIONS) -> int:
    """Remove oldest edition directories if more than max_keep exist.

    Returns the number of editions removed.
    """
    if not project_dir.is_dir():
        return 0

    edition_dirs = sorted(
        (d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith("edition-")),
        key=lambda d: _edition_number(d.name),
    )

    if len(edition_dirs) <= max_keep:
        return 0

    to_remove = edition_dirs[: len(edition_dirs) - max_keep]
    removed = 0
    for d in to_remove:
        try:
            import shutil
            shutil.rmtree(d)
            removed += 1
        except OSError as exc:
            logger.debug("Could not remove old edition %s: %s", d, exc)

    return removed


def _edition_number(dirname: str) -> int:
    """Extract the edition number from a directory name like 'edition-3'."""
    try:
        return int(dirname.split("-", 1)[1])
    except (IndexError, ValueError):
        return 0
