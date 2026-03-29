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


def _short_step(label: str) -> str:
    """Abbreviated step label for tight table cells.

    '100,000 items' → '100K', '25,000 items of data' → '25K',
    '50 simultaneous users' → '50 users', '1 concurrent connection' → '1 conn'.
    """
    desc = _describe_step(label) or label

    # Shorten large numbers: "100,000 items" → "100K"
    m = re.match(r"([\d,]+)\s+(items?|rows?|items? of data|rows? of data)", desc)
    if m:
        n = int(m.group(1).replace(",", ""))
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)

    # Shorten user/connection counts
    m = re.match(r"([\d,]+)\s+simultaneous\s+users?", desc)
    if m:
        return f"{m.group(1)} users"
    m = re.match(r"([\d,]+)\s+concurrent\s+connections?", desc)
    if m:
        return f"{m.group(1)} conn"
    m = re.match(r"([\d,]+)\s+concurrent\s+", desc)
    if m:
        return desc  # already short enough

    # Generic: just return the step label cleaned
    return desc


def _fmt_cell(value: float, label: str, metric: str, short: bool = False) -> str:
    """Format a table cell: value (context).

    When *short* is True, uses abbreviated step labels for tight PDF cells.
    """
    val = _fmt_val(value, metric)
    ctx = _short_step(label) if short else (_describe_step(label) or label)
    return f"{val} ({ctx})"


def _verdict(
    dp: DegradationPoint,
    findings: list[Finding] | None = None,
) -> str:
    """One-phrase verdict for a degradation curve.

    When *findings* is provided, checks whether any finding covers the
    same scenario.  A finding-based verdict takes precedence over the
    threshold-based default — a 3ms → 29ms curve might look fine by
    absolute thresholds but the report may contain a WARNING for 9x
    degradation.

    However, if the curve itself is nearly flat (<15% growth from first
    to last step), the verdict is "Stable" regardless of matched findings.
    A finding about absolute levels (e.g. high baseline memory) is valid,
    but the curve shows the problem is static, not load-dependent.
    """
    if not dp.steps:
        return ""

    # ── Flat-curve override (need ≥2 steps to assess a trend) ──
    first_val = dp.steps[0][1]
    last_val = dp.steps[-1][1]
    if len(dp.steps) >= 2 and first_val > 0 and last_val / first_val < 1.15:
        return "Stable"

    # ── Finding-based verdict (takes precedence) ──
    if findings:
        severity = _finding_severity_for_dp(dp, findings)
        if severity == "critical":
            return "Critical -- see above"
        if severity == "warning":
            return "Warning -- see above"

    # ── Threshold-based verdict ──
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
            return "Monitor under load"
        return "Very heavy -- risk of crashes"

    if dp.metric == "error_count":
        if last_val <= 0:
            return "No errors"
        return f"{int(last_val)} errors at peak"

    if "stable" in (dp.description or "").lower():
        return "Stable"

    return ""


def _finding_severity_for_dp(
    dp: DegradationPoint,
    findings: list[Finding],
) -> str | None:
    """Return the highest severity finding that matches a degradation point.

    Match by category + metric overlap, not title substring (titles are
    humanised and don't reliably contain the raw scenario name).

    Rules:
    - HTTP findings (category ``http_load_testing``) match curves whose
      ``scenario_name`` starts with ``http_``.
    - Memory findings (category ``memory_profiling``) match curves whose
      ``metric`` is a memory metric.
    - Time/concurrency findings match curves whose ``metric`` is a time
      metric AND the finding category matches the curve's scenario
      category prefix (e.g. ``concurrent_execution`` matches
      ``flask_concurrent_*``).
    - Dependency overlap: if the finding's ``affected_dependencies``
      and the curve's ``scenario_name`` share a dependency token,
      that strengthens a category match.

    Returns ``"critical"``, ``"warning"``, or ``None``.
    """
    is_memory_metric = dp.metric in (
        "memory_peak_mb", "memory_mb", "memory_growth_mb",
    )
    is_time_metric = dp.metric in ("execution_time_ms", "response_time_ms")
    dp_name_lower = dp.scenario_name.lower()

    # Category prefixes extractable from scenario_name (e.g.
    # "flask_concurrent_request_load" → tokens include "concurrent")
    _CATEGORY_TOKENS: dict[str, list[str]] = {
        "concurrent_execution": ["concurrent", "gil_contention"],
        "data_volume_scaling": ["data", "volume", "scaling"],
        "memory_profiling": ["memory"],
        "blocking_io": ["blocking", "io"],
        "edge_case_input": ["edge_case"],
        "http_load_testing": ["http"],
    }

    # ── Metric-aware title keywords ──
    _RESPONSE_TIME_KEYWORDS = ("response time", "degradation", "latency")
    _MEMORY_KEYWORDS = ("memory",)

    def _metric_matches_finding(metric_is_time: bool, metric_is_memory: bool,
                                title_lower: str) -> bool:
        """Check whether a finding's title is compatible with the dp metric."""
        if metric_is_time:
            return any(kw in title_lower for kw in _RESPONSE_TIME_KEYWORDS)
        if metric_is_memory:
            return any(kw in title_lower for kw in _MEMORY_KEYWORDS)
        return True  # unknown metric — don't filter

    # First pass: category + metric match
    best: str | None = None
    fallback_best: str | None = None

    for f in findings:
        if f.severity not in ("critical", "warning"):
            continue

        matched = False
        title_lower = f.title.lower()

        # Rule 1: HTTP category ↔ http_ scenario prefix + metric filter
        if f.category == "http_load_testing" and dp_name_lower.startswith("http_"):
            if (is_time_metric or is_memory_metric):
                if _metric_matches_finding(is_time_metric, is_memory_metric,
                                           title_lower):
                    matched = True
                else:
                    # Category matches but metric doesn't — track as fallback
                    if f.severity == "critical":
                        fallback_best = "critical"
                    elif fallback_best is None:
                        fallback_best = "warning"
                    continue
            else:
                matched = True

        # Rule 2: Memory category ↔ memory metric
        elif f.category == "memory_profiling" and is_memory_metric:
            matched = True

        # Rule 3: Category token appears in scenario name
        elif not matched:
            tokens = _CATEGORY_TOKENS.get(f.category, [])
            if any(t in dp_name_lower for t in tokens):
                # Extra confidence: metric type should be compatible
                if is_time_metric and f.category in (
                    "concurrent_execution", "blocking_io",
                    "data_volume_scaling",
                ):
                    matched = True
                elif is_memory_metric and f.category == "memory_profiling":
                    matched = True
                elif f.category in ("edge_case_input", "http_load_testing"):
                    matched = True

        # Rule 4: Dependency overlap as tiebreaker — with metric awareness
        if not matched and f.affected_dependencies:
            for dep in f.affected_dependencies:
                if dep.lower().replace("-", "_") in dp_name_lower:
                    # Check metric compatibility: a memory finding should
                    # not match a time curve, and vice versa
                    if is_time_metric and any(
                        kw in title_lower for kw in _MEMORY_KEYWORDS
                    ):
                        continue
                    if is_memory_metric and any(
                        kw in title_lower for kw in _RESPONSE_TIME_KEYWORDS
                    ):
                        continue
                    matched = True
                    break

        if matched:
            if f.severity == "critical":
                return "critical"
            best = "warning"

    # If metric-filtered matching found results, use those; else fall back
    if best is not None:
        return best
    return fallback_best


def _perf_row_label(dp: DegradationPoint) -> str:
    """Human label for the 'What we tested' column."""
    name = _describe_scenario(dp.scenario_name) or dp.scenario_name
    is_memory = dp.metric in ("memory_peak_mb", "memory_mb", "memory_growth_mb")
    if is_memory:
        label = f"Memory — {name}"
    else:
        label = name.capitalize() if name and name[0].islower() else name
    return label


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


def _row_cells(
    dp: DegradationPoint,
    short: bool = False,
    findings: list[Finding] | None = None,
) -> tuple[str, str, str, str, str]:
    """Extract (label, low, mid, high, verdict) for a table row.

    If short=True, use abbreviated step labels for tight PDF cells.
    """
    label = _perf_row_label(dp)
    steps = dp.steps

    def fmt(value: float, step_label: str) -> str:
        return _fmt_cell(value, step_label, dp.metric, short=short)

    low_label, low_val = steps[0]
    low = fmt(low_val, low_label)

    if len(steps) >= 3:
        mid_idx = len(steps) // 2
        mid_label, mid_val = steps[mid_idx]
        mid = fmt(mid_val, mid_label)
    else:
        mid = "\u2014"

    if len(steps) >= 2:
        high_label, high_val = steps[-1]
        high = fmt(high_val, high_label)
    else:
        high = "\u2014"

    return label, low, mid, high, _verdict(dp, findings)


def _render_perf_table_md(
    lines: list[str],
    points: list[DegradationPoint],
    findings: list[Finding] | None = None,
) -> None:
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
        label, low, mid, high, v = _row_cells(dp, findings=findings)
        lines.append(f"| {label} | {low} | {mid} | {high} | {v} |")

    lines.append("")


def _verdict_display(verdict: str) -> str:
    """Convert internal verdict text to user-facing display text.

    Replaces 'Critical' with 'Priority' for PDF/display output while
    keeping the internal representation unchanged for tests.
    """
    return verdict.replace("Critical", "Priority")


def _verdict_color(verdict: str) -> tuple:
    """Return text colour for a verdict string."""
    vl = verdict.lower()
    if "critical" in vl or "priority" in vl:
        return _RED
    if "warning" in vl:
        return _AMBER_TEXT
    if "no issues" in vl or "fine" in vl or "no errors" in vl or "stable" in vl:
        return _GREEN
    if "noticeable" in vl:
        return _AMBER_TEXT
    if "slow" in vl or "heavy" in vl or "unresponsive" in vl or "error" in vl:
        return _RED
    return _BODY


def _render_perf_table_pdf(
    pdf,
    points: list[DegradationPoint],
    findings: list[Finding] | None = None,
) -> None:
    """Render degradation points as a styled PDF performance summary table."""
    rows = _dedup_by_label(points)
    if not rows:
        return

    pdf.section_heading("Scaling Roadmap")
    pdf.ln(2)

    # Column widths (mm) — total 161mm for A4 with 20mm margins
    col_w = [50, 27, 27, 27, 30]
    headers = ["What we tested", "At low load", "At mid load", "At peak load", "Verdict"]
    base_row_h = 6
    line_h = 3.5  # line height for multi_cell wrapping

    # Header row — dark blue background, white text
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_WHITE)
    pdf.set_fill_color(*_BRAND)
    pdf.set_draw_color(*_RULE)
    pdf.set_line_width(0.18)
    for i, hdr in enumerate(headers):
        pdf.cell(col_w[i], base_row_h, f"  {_safe_text(hdr)}", border=1, fill=True)
    pdf.ln(base_row_h)

    # Data rows — alternating white/grey, first column wraps
    for row_idx, dp in enumerate(rows):
        label, low, mid, high, v = _row_cells(dp, short=True, findings=findings)
        is_alt = row_idx % 2 == 1
        fill_color = _TABLE_ALT if is_alt else _WHITE

        # Calculate row height based on label text wrapping
        safe_label = _safe_text(label)
        # Estimate chars per line in 50mm col at 8pt Helvetica (~2.1mm/char)
        chars_per_line = max(1, int((col_w[0] - 2) / 2.1))
        wrapped_lines = max(1, -(-len(safe_label) // chars_per_line))  # ceil div
        row_h = max(base_row_h, wrapped_lines * line_h + 1)

        row_y = pdf.get_y()
        row_x = pdf.get_x()

        pdf.set_draw_color(*_RULE)
        pdf.set_line_width(0.18)

        # Draw background rectangles for all cells first
        pdf.set_fill_color(*fill_color)
        x_offset = row_x
        for w in col_w:
            pdf.rect(x_offset, row_y, w, row_h, style="FD")
            x_offset += w

        # First column: label with wrapping via multi_cell
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_BODY)
        pdf.set_xy(row_x + 1, row_y + 0.5)
        pdf.multi_cell(col_w[0] - 2, line_h, safe_label)

        # Remaining data columns: single-line cells (vertically centred)
        data_cells = [low, mid, high]
        x_pos = row_x + col_w[0]
        for i, cell_text in enumerate(data_cells):
            cell_y = row_y + (row_h - base_row_h) / 2
            pdf.set_xy(x_pos, cell_y)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*_BODY)
            pdf.cell(col_w[i + 1], base_row_h, f"  {_safe_text(cell_text)}")
            x_pos += col_w[i + 1]

        # Verdict column — bold, coloured, vertically centred
        cell_y = row_y + (row_h - base_row_h) / 2
        pdf.set_xy(x_pos, cell_y)
        pdf.set_font("Helvetica", "B", 8)
        v_display = _verdict_display(v)
        pdf.set_text_color(*_verdict_color(v_display))
        pdf.cell(col_w[4], base_row_h, f"  {_safe_text(v_display)}")

        pdf.set_xy(row_x, row_y + row_h)

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

    _severity_labels = {
        "critical": "PRIORITY IMPROVEMENTS",
        "warning": "IMPROVEMENT OPPORTUNITIES",
        "info": "GOOD TO KNOW",
    }
    for severity in ("critical", "warning", "info"):
        group = severity_groups[severity]
        if not group:
            continue
        label = _severity_labels.get(severity, severity.upper())
        lines.append(f"## {label} ({len(group)})")
        lines.append("")
        for f in group:
            _render_understanding_finding(lines, f)

    # HTTP-tested summary (single line, not individual findings)
    if http_tested:
        lines.append(f"*{_http_tested_summary(http_tested)}*")
        lines.append("")

    # Performance summary table
    _render_perf_table_md(lines, report.degradation_points, report.findings)

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

    # Architecture-aware diagnosis (why it matters)
    diagnosis = _build_diagnosis(f)
    if diagnosis:
        lines.append(f"**Why this matters:** {diagnosis}")
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
    """Generate a plain-language consequence from the finding's data.

    Produces text that explains what the finding means for real users,
    not just what happened during testing.  Uses the finding's category,
    metrics, load level, and description context to build a specific,
    actionable consequence.
    """
    desc = (f.description or "").lower()
    parts: list[str] = []

    # ── Extract user_scale from contextualised description ──
    user_scale = _extract_user_scale_from_desc(f.description or "")

    # ── Capacity-relative framing ──
    if user_scale and f._load_level:
        if f._load_level <= user_scale:
            parts.append(
                "This affects your users at the scale you described."
            )
        elif f._load_level > user_scale * 3:
            parts.append(
                f"This only becomes a problem if your usage grows "
                f"beyond {user_scale:,} users."
            )

    # ── Category-specific consequences ──
    if f.category == "data_volume_scaling":
        if f._load_level:
            parts.append(
                f"Users working with datasets larger than "
                f"{f._load_level:,} items will experience "
                f"crashes or slowdowns."
            )
        elif f._error_count and f._error_count > 0:
            parts.append(
                "Users with larger datasets will hit errors "
                "or slowdowns as data volume increases."
            )

    elif f.category == "memory_profiling":
        if f._peak_memory_mb and f._peak_memory_mb > 50:
            if user_scale:
                # Estimate total RAM: peak per session * user count
                estimated_gb = (f._peak_memory_mb * user_scale) / 1024
                if estimated_gb >= 1.0:
                    parts.append(
                        f"At your expected {user_scale:,} concurrent users, "
                        f"the server's memory will be exhausted. You'll need "
                        f"approximately {estimated_gb:.1f} GB of RAM, or "
                        f"reduce memory usage per session."
                    )
                else:
                    parts.append(
                        f"Memory usage of {f._peak_memory_mb:.0f} MB per "
                        f"session adds up across {user_scale:,} users. "
                        f"Monitor memory under real traffic."
                    )
            else:
                parts.append(
                    f"Memory usage of {f._peak_memory_mb:.0f} MB under load "
                    f"could exhaust server memory, causing crashes or "
                    f"forced restarts."
                )

    elif f.category in ("concurrent_execution", "gil_contention",
                         "async_failures", "blocking_io"):
        if f._execution_time_ms and f._execution_time_ms > 0:
            ms = f._execution_time_ms
            if f._load_level and f._load_level > 0:
                qualifier = _response_time_qualifier(ms)
                parts.append(
                    f"At {f._load_level:,} concurrent users, page loads "
                    f"will take {ms:,.0f} ms — {qualifier}."
                )
            elif ms > 2000:
                qualifier = _response_time_qualifier(ms)
                parts.append(
                    f"Response times of {ms:,.0f} ms mean your users "
                    f"will experience {qualifier} performance."
                )

    elif f.category == "http_load_testing":
        if f._execution_time_ms and f._load_level:
            ms = f._execution_time_ms
            qualifier = _response_time_qualifier(ms)
            parts.append(
                f"At {f._load_level:,} concurrent users, page loads "
                f"will take {ms:,.0f} ms — {qualifier}."
            )

    # ── Fallback: error count ──
    if not parts and f._error_count and f._error_count > 5:
        parts.append(
            f"{f._error_count} errors occurred during testing — under "
            f"real traffic, some users will see failures."
        )

    # ── Fallback: high memory without category match ──
    if not parts and f._peak_memory_mb and f._peak_memory_mb > 100:
        parts.append(
            f"Memory usage of {f._peak_memory_mb:.0f} MB under load could "
            f"exhaust server memory, causing crashes or forced restarts."
        )

    # ── Fallback: slow response without category match ──
    if not parts and f._execution_time_ms and f._execution_time_ms > 2000:
        qualifier = _response_time_qualifier(f._execution_time_ms)
        parts.append(
            f"Response times of {f._execution_time_ms:,.0f} ms mean your "
            f"users will experience {qualifier} performance."
        )

    return " ".join(parts)


def _response_time_qualifier(ms: float) -> str:
    """Return a human qualifier for a response time in milliseconds."""
    if ms < 500:
        return "noticeable but acceptable"
    if ms < 2000:
        return "noticeably slow"
    if ms < 5000:
        return "slow"
    return "unresponsive"


def _extract_user_scale_from_desc(desc: str) -> int | None:
    """Extract user_scale from contextualised description text.

    The contextualise step prepends "You said N users." or
    "You said N,NNN users." — extract N.
    """
    import re
    m = re.search(r"You said ([\d,]+) users", desc)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def generate_finding_prompt(f: Finding) -> str:
    """Generate a ready-to-paste prompt for a coding agent from a finding.

    Returns empty string for INFO findings (no action needed).
    """
    if f.severity == "info":
        return ""

    parts: list[str] = []
    parts.append(f"[{f.severity.upper()}] {f.title}")

    if f.source_file:
        loc = f.source_file
        if f.source_function:
            loc += f" -> {f.source_function}"
        parts.append(f"Location: {loc}")

    if f.affected_dependencies:
        parts.append(f"Deps: {', '.join(f.affected_dependencies)}")

    if f._load_level is not None:
        parts.append(f"Failed at load level: {f._load_level}")

    # Architecture-aware diagnosis + fix
    diagnosis = _build_diagnosis(f)
    if diagnosis:
        parts.append(f"Diagnosis: {diagnosis}")
    fix_goal = _build_fix(f)
    parts.append(f"Fix: {fix_goal}")

    parts.append("See attached JSON for full diagnostic data.")

    return "\n".join(parts)


_FIX_OBJECTIVES: dict[str, str] = {
    "memory_profiling": (
        "eliminate unbounded memory growth so the application stays "
        "within safe limits under repeated use."
    ),
    "concurrent_execution": (
        "ensure thread safety so the application behaves correctly "
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
        "resolve the issue described above so the application behaves "
        "correctly under the tested conditions.",
    )


# ── Architecture-aware remediation ──


def _detect_framework(deps: list[str]) -> str:
    """Derive the server framework from a finding's affected_dependencies."""
    for dep in deps:
        d = dep.lower()
        if d in ("fastapi", "flask", "streamlit", "django", "express"):
            return d
    # uvicorn implies FastAPI
    if any(d.lower() == "uvicorn" for d in deps):
        return "fastapi"
    return ""


def _extract_mb_from_text(text: str) -> str:
    """Extract a memory value like '54MB' or '65MB' from description text."""
    m = re.search(r"(\d+)\s*MB", text, re.IGNORECASE)
    return m.group(1) if m else ""


def _remediation_fields(f: Finding) -> dict[str, str]:
    """Extract interpolation fields from a finding for remediation templates."""
    title_lower = f.title.lower()

    # Load level: use field if set, else "high"
    load = str(f._load_level) if f._load_level else "high"

    # Memory: use field if set, else extract from description text
    if f._peak_memory_mb and f._peak_memory_mb > 0:
        mem = f"{f._peak_memory_mb:.0f}"
    else:
        mem = _extract_mb_from_text(f.description) or "high"

    # Endpoint label from title
    endpoint = ""
    if "endpoint" in title_lower:
        parts = f.title.split("Endpoint ", 1)
        if len(parts) == 2:
            endpoint = parts[1].split(" ")[0]
    if not endpoint:
        endpoint = f.source_function or "endpoint"

    return {"load": load, "mem": mem, "endpoint": endpoint}


# Each pattern returns (diagnosis, fix) or None if no match.
# Diagnosis = what's wrong (human-readable). Fix = what to do (actionable).
_REMEDIATION_PATTERNS: list[
    tuple[str, "Callable[[Finding, str, dict], tuple[str, str] | None]"]
] = []


def _register_pattern(fn):
    """Decorator to register a remediation pattern matcher."""
    _REMEDIATION_PATTERNS.append(fn)
    return fn


@_register_pattern
def _pat_fastapi_concurrency(f, framework, fields):
    if (
        framework == "fastapi"
        and f.category == "http_load_testing"
        and f.failure_domain == "concurrency_failure"
    ):
        return (
            f"Your FastAPI endpoint delegates blocking work to the default "
            f"thread pool. At {fields['load']} concurrent requests, the pool "
            f"saturates and requests queue — new requests wait for a thread, "
            f"causing cascading timeouts.",
            "Create a dedicated ThreadPoolExecutor sized for your expected "
            "concurrency, or convert blocking operations to native async "
            "(async def + await).",
        )
    return None


@_register_pattern
def _pat_fastapi_startup(f, framework, fields):
    if (
        framework == "fastapi"
        and f.category == "http_load_testing"
        and "could not start" in f.title.lower()
    ):
        return (
            "Your FastAPI app failed to start because a required dependency "
            "is missing.",
            "Check that all imports in your main app module are installed. "
            "Add missing packages to requirements.txt or pyproject.toml "
            "dependencies (not just dev/optional groups).",
        )
    return None


@_register_pattern
def _pat_streamlit_memory(f, framework, fields):
    if (
        framework == "streamlit"
        and f.failure_pattern == "memory_accumulation_over_sessions"
    ):
        return (
            f"Streamlit creates a new Python process per user session. Your "
            f"app uses {fields['mem']}MB per session. At {fields['load']} "
            f"concurrent users, you'll exhaust server memory.",
            "Cache shared data with @st.cache_data, move heavy computation "
            "to a background service, and avoid loading large "
            "models/datasets at module level.",
        )
    return None


@_register_pattern
def _pat_streamlit_response_time(f, framework, fields):
    if (
        framework == "streamlit"
        and f.category == "http_load_testing"
        and (
            f.failure_domain in ("concurrency_failure", "scaling_collapse")
            or "response time" in f.title.lower()
            or "degradation" in f.title.lower()
        )
    ):
        return (
            "Streamlit reruns the entire script on each user interaction. "
            "Under concurrent load, this compounds because each session "
            "triggers a full re-execution.",
            "Use @st.cache_data for expensive computations, "
            "@st.cache_resource for database connections and ML models, "
            "and move heavy initialization outside the main script flow.",
        )
    return None


@_register_pattern
def _pat_flask_concurrency(f, framework, fields):
    if (
        framework == "flask"
        and f.category in (
            "http_load_testing", "blocking_io", "concurrent_execution",
        )
    ):
        return (
            f"Flask handles requests synchronously — each request blocks a "
            f"worker thread. At {fields['load']} concurrent requests, all "
            f"threads are occupied and new requests queue.",
            "Use database connection pooling (SQLAlchemy pool_size), add "
            "gunicorn with multiple workers (gunicorn -w 4), or migrate "
            "to an async framework for I/O-heavy endpoints.",
        )
    return None


@_register_pattern
def _pat_memory_baseline(f, framework, fields):
    if "memory baseline" in f.title.lower():
        return (
            f"Your application uses {fields['mem']}MB per process. This is "
            f"a baseline issue, not a memory leak — memory stays flat "
            f"under load.",
            "Use lazy imports for heavy modules (import inside functions, "
            "not at module level), defer loading large models/data until "
            "first request, or increase server memory.",
        )
    return None


@_register_pattern
def _pat_external_timeout(f, framework, fields):
    title_lower = f.title.lower()
    if "skipped" in title_lower and "slow response" in title_lower:
        return (
            f"Your {fields['endpoint']} took over 10 seconds at 1 "
            f"concurrent connection — it's waiting on an external service "
            f"that isn't configured in the test environment.",
            "Ensure the service is available in production, add timeout "
            "handling (e.g. requests.get(url, timeout=5)), and return a "
            "graceful error when the service is down.",
        )
    return None


@_register_pattern
def _pat_pandas_silent_dtypes(f, framework, fields):
    if "pandas" in (f.affected_dependencies or []) and (
        f.failure_pattern == "silent_data_type_changes"
        or (
            ("dtype" in f.title.lower() or "type" in f.title.lower())
            and f.category == "edge_case_input"
        )
    ):
        return (
            "Pandas silently converts data types when input values don't match "
            "expected types. A single non-numeric value in an integer column "
            "converts the entire column to object dtype, increasing memory 10x "
            "and producing incorrect numeric operations without raising errors.",
            "Specify dtypes explicitly in read_csv(dtype={...}), use "
            "pd.to_numeric(errors='coerce') for controlled conversion, and "
            "validate column dtypes after loading with df.dtypes checks.",
        )
    return None


@_register_pattern
def _pat_requests_concurrent(f, framework, fields):
    if "requests" in (f.affected_dependencies or []) and (
        f.failure_domain == "concurrency_failure"
        or f.category == "concurrent_execution"
    ):
        return (
            f"The requests library is synchronous — each call blocks its "
            f"thread until the response arrives. At {fields['load']} concurrent "
            f"requests, all threads are occupied waiting on I/O and new "
            f"requests queue.",
            "Use requests.Session() for connection pooling, switch to "
            "httpx.AsyncClient or aiohttp for async I/O, or use "
            "concurrent.futures.ThreadPoolExecutor with a bounded pool size.",
        )
    return None


@_register_pattern
def _pat_data_volume(f, framework, fields):
    if f.category == "data_volume_scaling":
        return (
            "Your application's processing time grows with input size. At "
            "large inputs, this becomes the bottleneck.",
            "Use chunked or streaming processing instead of loading all "
            "data into memory, add pagination for large result sets, and "
            "consider caching intermediate results for repeated queries.",
        )
    return None


def _match_remediation(f: Finding) -> tuple[str, str] | None:
    """Try each remediation pattern against a finding.

    Returns (diagnosis, fix) or None if no pattern matches.
    """
    framework = _detect_framework(f.affected_dependencies)
    fields = _remediation_fields(f)
    for pattern_fn in _REMEDIATION_PATTERNS:
        result = pattern_fn(f, framework, fields)
        if result is not None:
            return result
    return None


def _build_diagnosis(f: Finding) -> str:
    """Return architecture-aware diagnosis, or empty string if unmatched."""
    match = _match_remediation(f)
    return match[0] if match else ""


def _build_fix(f: Finding) -> str:
    """Return architecture-aware fix instruction, or category-level fallback."""
    match = _match_remediation(f)
    if match:
        return match[1]
    return _FIX_OBJECTIVES.get(
        f.category,
        "resolve the issue described above so the application behaves "
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
    """Replace Unicode characters unsupported by built-in Helvetica.

    Helvetica in fpdf2 only supports Latin-1 (ISO 8859-1).  Known
    replacements are applied first; any remaining non-Latin-1 characters
    are stripped so the renderer never crashes.
    """
    out = (
        text
        .replace("\u2014", "--")  # em-dash —
        .replace("\u2013", "-")   # en-dash –
        .replace("\u2018", "'")   # left single quote '
        .replace("\u2019", "'")   # right single quote '
        .replace("\u201c", '"')   # left double quote "
        .replace("\u201d", '"')   # right double quote "
        .replace("\u2022", "*")   # bullet •
        .replace("\u00b7", "*")   # middle dot ·
        .replace("\u2026", "...")  # ellipsis …
        .replace("\u2192", "->")  # right arrow →
        .replace("\u2190", "<-")  # left arrow ←
        .replace("\u2264", "<=")  # less-than-or-equal ≤
        .replace("\u2265", ">=")  # greater-than-or-equal ≥
        .replace("\u00d7", "x")   # multiplication sign ×
        .replace("\u00f7", "/")   # division sign ÷
        .replace("\u00b1", "+/-") # plus-minus sign ±
    )
    # Strip any remaining non-Latin-1 characters
    return out.encode("latin-1", errors="ignore").decode("latin-1")


def _make_pdf_class():
    """Create the MyCodePDF class (requires fpdf2)."""
    if not _HAS_FPDF:
        return None

    class MyCodePDF(FPDF):
        """Styled PDF with myCode header and footer on every page."""

        def header(self):
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*_BRAND)
            self.cell(40, 8, "myCode", new_x="RIGHT")
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*_SUBTLE)
            self.cell(
                0, 8,
                "Stress test your AI-generated code before it breaks",
                align="R",
            )
            self.ln(10)  # clear below the text descenders
            self.set_draw_color(*_RULE)
            self.set_line_width(0.35)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(6)

        def footer(self):
            self.set_y(-15)
            self.set_draw_color(*_RULE)
            self.set_line_width(0.35)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(2)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_SUBTLE)
            self.cell(
                0, 4,
                _safe_text("myCode by Machine Adjacent Systems - Diagnostic tool"),
                new_x="LEFT",
            )
            self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="R")

        def section_heading(self, text: str, level: int = 2):
            """H1=16pt brand, H2=12pt heading bg strip, H3=11pt heading bg strip.

            Widow/orphan control: if the heading + ~25mm of following
            content won't fit on the current page, break first.  A
            section header must never appear alone at the bottom of a
            page.
            """
            sizes = {1: 16, 2: 12, 3: 11}
            size = sizes.get(level, 11)
            pad = 2  # mm padding around text

            # Background strip for H2 and H3
            if level in (2, 3):
                bg = (230, 240, 250) if level == 2 else (243, 243, 245)
                self.set_font("Helvetica", "B", size)
                safe = _safe_text(text)
                line_h = size * 0.55
                w = self.w - self.l_margin - self.r_margin
                chars_per_line = max(1, int(w / (size * 0.28)))
                n_lines = max(1, -(-len(safe) // chars_per_line))
                box_h = n_lines * line_h + 2 * pad

                # Widow/orphan: need heading + at least 80mm of content
                # so the first finding's title + "What we found" stays
                # with the section header.
                min_after = box_h + 80
                remaining = self.h - self.get_y() - 20
                if remaining < min_after:
                    self.add_page()

                x = self.l_margin
                y = self.get_y()
                self.set_fill_color(*bg)
                self.rect(x, y, w, box_h, style="F")
                self.set_xy(x + pad, y + pad)
                self.set_text_color(*(_BRAND if level == 1 else _HEADING))
                self.multi_cell(w - 2 * pad, line_h, safe)
                self.set_y(y + box_h)
                self.ln(2)
            else:
                # H1 — no widow control needed (always at top)
                self.set_font("Helvetica", "B", size)
                self.set_text_color(*_BRAND)
                self.multi_cell(0, size * 0.55, _safe_text(text))
                self.ln(2)

        def body_text(self, text: str):
            """10pt body text."""
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_BODY)
            self.multi_cell(0, 4.5, _safe_text(text))
            self.ln(1.5)

        def body_label(self, text: str):
            """Bold inline label."""
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(*_HEADING)
            self.multi_cell(0, 4.5, _safe_text(text))
            self.ln(1)

        def severity_badge(self, severity: str):
            """Inline severity badge with user-friendly labels."""
            bg, fg = _SEVERITY_COLORS.get(severity, (_BLUE, _WHITE))
            _badge_labels = {
                "critical": "PRIORITY",
                "warning": "OPPORTUNITY",
                "info": "INFO",
            }
            label = _badge_labels.get(severity, severity.upper())
            self.set_font("Helvetica", "B", 8)
            w = self.get_string_width(label) + 5
            self.set_fill_color(*bg)
            self.set_text_color(*fg)
            self.cell(w, 5, f" {label} ", fill=True, new_x="RIGHT")
            self.set_text_color(*_BODY)
            self.cell(4, 5, "")  # spacer

        def code_block(self, text: str):
            """Compact prompt box: grey bg, border, 7pt monospace.

            These are reference prompts for coding agents — compact is
            better.  If the box won't fit, breaks before it starts.
            """
            safe = _safe_text(text[:800])
            x = self.l_margin
            w = self.w - self.l_margin - self.r_margin
            pad = 3  # mm internal padding (compact)
            font_size = 7
            line_h = 3.0  # tight line spacing

            # Estimate box height
            self.set_font("Courier", "", font_size)
            chars_per_line = max(1, int((w - 2 * pad) / 1.7))
            wrapped_lines = sum(
                max(1, (len(ln) + chars_per_line - 1) // chars_per_line)
                for ln in safe.split("\n")
            )
            text_h = wrapped_lines * line_h
            box_h = text_h + 2 * pad

            # If box won't fit but would fit on a fresh page, break now
            remaining = self.h - self.get_y() - 20
            if box_h > remaining and box_h < self.h - 40:
                self.add_page()

            box_y = self.get_y()

            # Draw background — clip to available space on this page
            draw_h = min(box_h, self.h - box_y - 20)
            self.set_fill_color(*_CODE_BG)
            self.set_draw_color(*_CODE_BORDER)
            self.set_line_width(0.35)
            self.rect(x, box_y, w, draw_h, style="FD")

            # Render text inside (auto page break handles overflow)
            self.set_font("Courier", "", font_size)
            self.set_text_color(*_BODY)
            self.set_xy(x + pad, box_y + pad)
            self.multi_cell(w - 2 * pad, line_h, safe)

            # Ensure Y is past the box
            if self.get_y() < box_y + box_h:
                self.set_y(box_y + box_h)
            self.ln(2)

        def bullet(self, text: str):
            """Bullet point with indent."""
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*_BODY)
            x = self.get_x()
            self.set_x(x + 3)
            self.cell(4, 4.5, "-")
            self.multi_cell(0, 4.5, _safe_text(text))
            self.set_x(x)

        def callout_box(self, text: str):
            """Callout box with light blue background."""
            safe = _safe_text(text)
            x = self.l_margin
            w = self.w - self.l_margin - self.r_margin
            pad = 5

            # Measure
            self.set_font("Helvetica", "", 9)
            box_y = self.get_y()
            chars_per_line = max(1, int((w - 2 * pad) / 2.0))
            line_count = sum(
                max(1, (len(ln) + chars_per_line - 1) // chars_per_line)
                for ln in safe.split("\n")
            )
            text_h = line_count * 4.5
            box_h = text_h + 2 * pad

            # Draw box
            self.set_fill_color(*_CALLOUT_BG)
            self.set_draw_color(*_CALLOUT_BORDER)
            self.set_line_width(0.35)
            self.rect(x, box_y, w, box_h, style="FD")

            # Render text
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_BRAND)
            self.set_xy(x + pad, box_y + pad)
            self.multi_cell(w - 2 * pad, 4.5, safe)
            self.set_y(box_y + box_h)
            self.ln(4)

    return MyCodePDF


def _extract_memory_capacity(text: str) -> str:
    """Extract memory capacity info from plain_summary text.

    Looks for patterns like "X MB per process" or "N concurrent sessions".
    Returns the matched sentence/fragment, or empty string if not found.
    """
    if not text:
        return ""
    # Match sentences containing memory capacity indicators
    for pattern in (
        r"[^.]*\d+\s*MB\s+per\s+(?:process|session|user)[^.]*",
        r"[^.]*concurrent\s+sessions?[^.]*\d+\s*MB[^.]*",
        r"[^.]*\d+\s*MB[^.]*concurrent[^.]*",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


_PREDICTION_SEVERITY_COLORS: dict[str, tuple] = {
    "critical": _RED,
    "warning": _AMBER_TEXT,
    "info": _SUBTLE,
}


def _render_pred_bars(pdf, preds: list[dict]) -> None:
    """Render a list of prediction items as probability bars."""
    bar_max_w = 60
    bar_h = 4
    for pred in preds:
        title = pred.get("title", "")
        prob = pred.get("probability_pct", 0)
        severity = pred.get("severity", "info")
        bar_color = _PREDICTION_SEVERITY_COLORS.get(severity, _SUBTLE)
        y = pdf.get_y()
        x = pdf.l_margin
        bar_w = max(1, bar_max_w * prob / 100.0)
        pdf.set_fill_color(*bar_color)
        pdf.rect(x, y + 0.5, bar_w, bar_h, style="F")
        pdf.set_xy(x + bar_max_w + 3, y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*_BODY)
        pdf.cell(15, 5, f"{prob:.0f}%")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.cell(0, 5, _safe_text(title))
        pdf.ln(bar_h + 2)


def _render_predictive_analysis_pdf(pdf, predictions: dict) -> None:
    """Render the Predictive Analysis section in the PDF.

    Shows two layers when architecture-filtered: architecture-specific
    predictions first, then technology-wide predictions.
    """
    total = predictions.get("total_similar_projects", 0)
    deps = predictions.get("matching_deps", [])
    preds = predictions.get("predictions", [])
    if not preds:
        return

    pdf.section_heading("Predictive Analysis")
    deps_str = ", ".join(deps[:8])
    if len(deps) > 8:
        deps_str += "..."

    arch_type = predictions.get("architectural_type", "")
    arch_label = (
        arch_type.replace("_", " ")
        if arch_type and arch_type != "general" else ""
    )
    arch_filtered = predictions.get("arch_filtered", False)
    tw_preds = predictions.get("tech_wide_predictions", [])
    tw_total = predictions.get("tech_wide_total", 0)

    if arch_filtered and arch_label:
        # Section 1 — Architecture-specific
        intro = (
            f"For {arch_label} projects "
            f"({total} projects using {deps_str}):"
        )
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.multi_cell(0, 4.5, _safe_text(intro))
        pdf.ln(2)
        _render_pred_bars(pdf, preds)

        # Section 2 — Technology-wide
        if tw_preds:
            pdf.ln(2)
            tw_intro = (
                f"Across all project types "
                f"({tw_total} projects using {deps_str}):"
            )
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*_BODY)
            pdf.multi_cell(0, 4.5, _safe_text(tw_intro))
            pdf.ln(2)
            _render_pred_bars(pdf, tw_preds)
    elif arch_label and not arch_filtered:
        # Limited arch data
        intro = (
            f"Limited {arch_label}-specific data available. "
            f"Showing predictions across all project types "
            f"({total} projects using {deps_str}):"
        )
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.multi_cell(0, 4.5, _safe_text(intro))
        pdf.ln(2)
        _render_pred_bars(pdf, preds)
    else:
        # No architecture — tech-wide only
        intro = (
            f"Based on {total} projects with similar "
            f"technology stack ({deps_str}):"
        )
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.multi_cell(0, 4.5, _safe_text(intro))
        pdf.ln(2)
        _render_pred_bars(pdf, preds)

    # Scale note
    scale_notes = [
        p.get("scale_note", "") for p in preds if p.get("scale_note")
    ]
    if scale_notes:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 3.5, _safe_text(scale_notes[0]))
    pdf.ln(4)


def render_understanding_pdf(
    report: DiagnosticReport, edition: int, project_name: str = "",
    predictions: Optional[dict] = None,
    constraints: Optional["OperationalConstraints"] = None,
) -> bytes:
    """Render the understanding document as a styled PDF.

    Returns raw PDF bytes.  Requires fpdf2.

    Args:
        report: The diagnostic report to render.
        edition: Edition number.
        project_name: Human-readable project name.
        predictions: Optional predictive analysis data dict with keys
            ``total_similar_projects``, ``matching_deps``, and
            ``predictions`` (list of prediction dicts).
        constraints: Optional operational constraints from conversation.
    """
    PDFClass = _make_pdf_class()
    if PDFClass is None:
        raise ImportError("fpdf2 is required for PDF generation")

    pdf = PDFClass(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 15, 20)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Cover-page header block ──
    # Use user's description if available, fall back to project name
    display_title = (
        report.user_project_description
        or project_name
        or "Your Project"
    )
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_BRAND)
    pdf.multi_cell(0, 8, _safe_text(display_title))
    pdf.ln(1)

    # Date and edition in subtle text
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_SUBTLE)
    date = _dt.date.today().isoformat()
    pdf.cell(0, 4, f"Edition {edition}  |  {date}")
    pdf.ln(5)

    # Technology stack summary
    if report.recognized_dep_names:
        stack_text = "Stack: " + ", ".join(report.recognized_dep_names[:12])
        if len(report.recognized_dep_names) > 12:
            stack_text += "..."
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_SUBTLE)
        pdf.cell(0, 4, _safe_text(stack_text))
        pdf.ln(5)

    # User intent summary box (only if constraints provided)
    if constraints is not None:
        intent_parts: list[str] = []
        if getattr(constraints, "current_users", None) is not None:
            intent_parts.append(
                f"Current users: {constraints.current_users}"
            )
        if getattr(constraints, "max_users", None) is not None:
            intent_parts.append(f"Max users: {constraints.max_users}")
        if getattr(constraints, "data_type", None):
            intent_parts.append(f"Data profile: {constraints.data_type}")
        if getattr(constraints, "usage_pattern", None):
            intent_parts.append(f"Usage: {constraints.usage_pattern}")
        if intent_parts:
            pdf.callout_box(" | ".join(intent_parts))

    # ── Assessment Context section ──
    if constraints is not None:
        pdf.body_label("Assessment Context")
        # Project type from user description
        if report.user_project_description:
            pdf.bullet(
                f"Project type: {report.user_project_description}"
            )
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.multi_cell(0, 4.5, "Results assessed relative to:")
        pdf.ln(1)
        current = getattr(constraints, "current_users", None)
        maximum = getattr(constraints, "max_users", None)
        if current is not None or maximum is not None:
            c_str = f"{current:,}" if current is not None else "?"
            m_str = f"{maximum:,}" if maximum is not None else "?"
            pdf.bullet(f"{c_str} current users -> {m_str} maximum users")
        per_user = getattr(constraints, "per_user_data", None)
        d_type = getattr(constraints, "data_type", None)
        if per_user or d_type:
            data_desc = per_user or "unspecified"
            type_desc = d_type or "mixed"
            pdf.bullet(
                f"{data_desc} data per user ({type_desc} data types)"
            )
        usage = getattr(constraints, "usage_pattern", None)
        if usage:
            pdf.bullet(f"{usage} usage pattern")
        depth = getattr(constraints, "analysis_depth", None)
        _depth_labels = {
            "quick": "quick (~2 min)",
            "standard": "standard (~5 min)",
            "deep": "deep (~10 min)",
        }
        depth_label = _depth_labels.get(depth or "", depth or "standard")
        pdf.bullet(f"Analysis depth: {depth_label}")

        # Memory capacity assessment from plain_summary
        mem_cap = _extract_memory_capacity(report.plain_summary or "")
        if mem_cap:
            pdf.ln(1)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*_SUBTLE)
            pdf.multi_cell(0, 4, _safe_text(f"Memory note: {mem_cap}"))

        pdf.ln(3)

    # Project summary
    if report.project_description:
        pdf.section_heading("Project Summary")
        pdf.body_text(report.project_description)

    # Test overview — compact single-line stats
    stats_parts: list[str] = []
    # Count priority improvements for summary language
    n_critical = sum(
        1 for f_item in report.findings if f_item.severity == "critical"
    )
    n_warning = sum(
        1 for f_item in report.findings if f_item.severity == "warning"
    )
    stats_parts.append(
        f"Scenarios: {report.scenarios_run} run, "
        f"{report.scenarios_passed} passed, "
        f"{report.scenarios_failed} failed"
    )
    if report.scenarios_incomplete:
        stats_parts.append(
            f"{report.scenarios_incomplete} could not test"
        )
    if report.total_errors:
        stats_parts.append(f"{report.total_errors} errors")
    stats = ", ".join(stats_parts[:2])
    if len(stats_parts) > 2:
        stats += " | " + " | ".join(stats_parts[2:])
    # Severity summary line
    sev_notes: list[str] = []
    if n_critical:
        sev_notes.append(
            f"{n_critical} priority improvement"
            f"{'s' if n_critical != 1 else ''}"
        )
    if n_warning:
        sev_notes.append(
            f"{n_warning} opportunity improvement"
            f"{'s' if n_warning != 1 else ''}"
        )
    if sev_notes:
        stats += f".  Found {', '.join(sev_notes)}."
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_BODY)
    pdf.multi_cell(0, 4.5, _safe_text(stats))
    pdf.ln(1)

    # Dependency stack — compact
    has_deps = report.recognized_dep_count > 0 or report.unrecognized_deps
    if has_deps:
        dep_parts: list[str] = []
        if report.recognized_dep_count:
            rec_names = ""
            if report.recognized_dep_names:
                rec_names = f" ({', '.join(report.recognized_dep_names[:10])})"
            dep_parts.append(
                f"{report.recognized_dep_count} profiled{rec_names}"
            )
        if report.unrecognized_deps:
            dep_parts.append(
                f"{len(report.unrecognized_deps)} usage-tested"
            )
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 4, f"Dependencies: {', '.join(dep_parts)}")
    pdf.ln(2)

    # ── Predictive Analysis section ──
    if predictions:
        _render_predictive_analysis_pdf(pdf, predictions)

    # Partition http_tested out of incomplete tests
    http_tested, other_incomplete = _partition_http_tested(
        report.incomplete_tests,
    )

    # Findings grouped by severity
    all_findings = list(report.findings) + other_incomplete
    severity_groups: dict[str, list] = {"critical": [], "warning": [], "info": []}
    for f in all_findings:
        severity_groups.get(f.severity, severity_groups["info"]).append(f)

    _pdf_severity_labels = {
        "critical": "PRIORITY IMPROVEMENTS",
        "warning": "IMPROVEMENT OPPORTUNITIES",
        "info": "GOOD TO KNOW",
    }
    for severity in ("critical", "warning", "info"):
        group = severity_groups[severity]
        if not group:
            continue
        label = _pdf_severity_labels.get(severity, severity.upper())
        pdf.section_heading(f"{label} ({len(group)})", level=3)
        for f in group:
            _render_pdf_finding(pdf, f)

    # HTTP-tested summary
    if http_tested:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 5, _safe_text(_http_tested_summary(http_tested)))
        pdf.ln(6)

    # Performance summary table + confidence + callout
    rows = _dedup_by_label(report.degradation_points)
    has_actionable = any(
        f.severity in ("critical", "warning") for f in report.findings
    )

    if rows:
        # Give the table a fresh page if it won't fit comfortably
        # (heading + header row + at least 2 data rows = ~30mm minimum)
        table_min_h = 10 + 6 + min(len(rows), 2) * 6 + 4
        remaining = pdf.h - pdf.get_y() - 20
        if remaining < table_min_h:
            pdf.add_page()

    if rows:
        _render_perf_table_pdf(pdf, report.degradation_points, report.findings)

    # Confidence note
    if report.confidence_note:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 4, _safe_text(report.confidence_note))
        pdf.ln(2)

    # JSON download callout
    if has_actionable:
        pdf.callout_box(
            "Download the JSON report and attach it when you paste the "
            "prompts above into your coding agent. The JSON contains the "
            "full diagnostic data your agent needs."
        )

    return bytes(pdf.output())


_CARD_BG = (250, 250, 252)  # near-white card background


def _draw_card_border(pdf, color: tuple, margin: float, y_start: float, y_end: float) -> None:
    """Draw a coloured left border for a finding card."""
    pdf.set_draw_color(*color)
    pdf.set_line_width(1.5)
    pdf.line(margin + 0.75, y_start, margin + 0.75, y_end)
    pdf.set_line_width(0.35)


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
    # Threshold indicators like "at first iteration" read better as a sentence
    detail_clean = details.rstrip(". ")
    if detail_clean.lower().startswith("at "):
        desc = description.rstrip()
        if not desc.endswith("."):
            desc += "."
        return f"{desc} This issue begins {detail_clean}."
    # Append as additional context
    desc = description.rstrip(". ")
    return f"{desc}. {detail_clean}."


def _estimate_finding_height(f: Finding) -> float:
    """Estimate the total height (mm) of a finding card.

    Used to decide whether to page-break before rendering so the card
    stays together as a visual unit.
    """
    h = 8.0  # badge + title line + spacing
    combined = _integrate_details(f.description or "", f.details or "")
    if combined:
        h += 6 + len(combined) / 70 * 4.5  # label + wrapped text
    diagnosis = _build_diagnosis(f)
    if diagnosis:
        h += 6 + len(diagnosis) / 70 * 4.5
    if f.severity == "info":
        h += 8
        return h
    consequence = _consequence_for_user(f)
    if consequence:
        h += 6 + len(consequence) / 70 * 4.5
    prompt = generate_finding_prompt(f)
    if prompt:
        h += 6 + 8  # label + instruction
        # Code block: ~3.5mm per line, capped at 800 chars
        prompt_lines = _safe_text(prompt[:800]).count("\n") + 1
        h += 10 + prompt_lines * 3.5  # padding + lines
    h += 12  # "After you fix it" + spacing
    return h


def _render_pdf_finding(pdf, f: Finding) -> None:
    """Render a single finding for the understanding PDF.

    Each finding is a card with a coloured left border.
    Critical/warning: full layout with prompt box.
    Info: "What we found" only.

    The card is kept together on one page when possible — if the
    estimated height won't fit, a page break is inserted before
    the card starts (not in the middle).
    """
    # Pre-flight: keep the card together if it fits on a fresh page
    est_h = _estimate_finding_height(f)
    remaining = pdf.h - pdf.get_y() - 20  # bottom margin
    if est_h < pdf.h - 40 and remaining < est_h:
        # Card fits on a page but not on the current one — break first
        pdf.add_page()

    border_color = _SEVERITY_BORDER.get(f.severity, _BLUE)
    card_indent = 4  # mm from left margin to content (after border)
    card_pad = 2     # mm inner padding for the card area
    original_margin = pdf.l_margin
    card_start_y = pdf.get_y()

    # Badge + title on one line
    pdf.set_x(original_margin + card_indent + card_pad)
    pdf.severity_badge(f.severity)
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_HEADING)
    pdf.cell(0, 5, _safe_text(title))
    pdf.ln(6)

    # Indent card content
    pdf.set_left_margin(original_margin + card_indent + card_pad)

    # What we found — description integrated with details
    pdf.body_label("What we found")
    combined = _integrate_details(f.description or "", f.details or "")
    if combined:
        pdf.body_text(combined)

    # Architecture-aware diagnosis (why it matters)
    diagnosis = _build_diagnosis(f)
    if diagnosis:
        pdf.body_label("Why this matters")
        pdf.body_text(diagnosis)

    # INFO findings: description only
    if f.severity == "info":
        if f.affected_dependencies:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*_SUBTLE)
            pdf.cell(
                0, 4,
                f"Related dependencies: {', '.join(f.affected_dependencies)}",
            )
            pdf.ln(3)
        card_end_y = pdf.get_y() + card_pad
        pdf.set_left_margin(original_margin)
        _draw_card_border(pdf, border_color, original_margin, card_start_y, min(card_end_y, pdf.get_y()))
        pdf.set_y(card_end_y)
        pdf.ln(4)
        return

    # What this means for you
    consequence = _consequence_for_user(f)
    if consequence:
        pdf.body_label("What this means for you")
        pdf.body_text(consequence)

    # What to do + Prompt combined
    prompt = generate_finding_prompt(f)
    if prompt:
        pdf.body_label("What to do")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_BODY)
        pdf.multi_cell(
            0, 4.5,
            _safe_text(
                "Paste this into your coding agent with the JSON report."
            ),
        )
        pdf.ln(1)
        pdf.code_block(prompt)

    # After you fix it
    pdf.body_label("After you fix it")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_BODY)
    pdf.cell(0, 5, "Run myCode again to verify the fix worked.")
    pdf.ln(4)

    # Draw left border spanning the card
    card_end_y = pdf.get_y() + card_pad
    pdf.set_left_margin(original_margin)
    _draw_card_border(pdf, border_color, original_margin, card_start_y, min(card_end_y, pdf.get_y()))
    pdf.set_y(card_end_y)
    pdf.ln(4)


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
