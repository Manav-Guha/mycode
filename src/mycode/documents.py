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
    import re as _re
    desc = _describe_step(label) or label

    # Shorten large numbers: "100,000 items" → "100K"
    m = _re.match(r"([\d,]+)\s+(items?|rows?|items? of data|rows? of data)", desc)
    if m:
        n = int(m.group(1).replace(",", ""))
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)

    # Shorten user/connection counts
    m = _re.match(r"([\d,]+)\s+simultaneous\s+users?", desc)
    if m:
        return f"{m.group(1)} users"
    m = _re.match(r"([\d,]+)\s+concurrent\s+connections?", desc)
    if m:
        return f"{m.group(1)} conn"
    m = _re.match(r"([\d,]+)\s+concurrent\s+", desc)
    if m:
        return desc  # already short enough

    # Generic: just return the step label cleaned
    return desc


def _fmt_cell(value: float, label: str, metric: str) -> str:
    """Format a table cell: value (context)."""
    val = _fmt_val(value, metric)
    ctx = _describe_step(label) or label
    return f"{val} ({ctx})"


def _fmt_cell_short(value: float, label: str, metric: str) -> str:
    """Format a table cell with abbreviated context for PDF."""
    val = _fmt_val(value, metric)
    ctx = _short_step(label)
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
    """
    # ── Finding-based verdict (takes precedence) ──
    if findings:
        severity = _finding_severity_for_dp(dp, findings)
        if severity == "critical":
            return "Critical -- see above"
        if severity == "warning":
            return "Warning -- see above"

    # ── Threshold-based verdict ──
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
    fmt = _fmt_cell_short if short else _fmt_cell

    low_label, low_val = steps[0]
    low = fmt(low_val, low_label, dp.metric)

    if len(steps) >= 3:
        mid_idx = len(steps) // 2
        mid_label, mid_val = steps[mid_idx]
        mid = fmt(mid_val, mid_label, dp.metric)
    else:
        mid = "\u2014"

    if len(steps) >= 2:
        high_label, high_val = steps[-1]
        high = fmt(high_val, high_label, dp.metric)
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


def _verdict_color(verdict: str) -> tuple:
    """Return text colour for a verdict string."""
    vl = verdict.lower()
    if "critical" in vl:
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

    pdf.section_heading("Performance Under Load")
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
        pdf.set_text_color(*_verdict_color(v))
        pdf.cell(col_w[4], base_row_h, f"  {_safe_text(v)}")

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

    # Fix objective — single actionable sentence
    fix_goal = _fix_objective(f)
    parts.append(f"Fix: {fix_goal}")

    parts.append("See attached JSON for full diagnostic data.")

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
            """H1=16pt brand, H2=12pt heading, H3=11pt heading."""
            sizes = {1: 16, 2: 12, 3: 11}
            size = sizes.get(level, 11)
            self.set_font("Helvetica", "B", size)
            self.set_text_color(*(_BRAND if level == 1 else _HEADING))
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
            """Inline severity badge."""
            bg, fg = _SEVERITY_COLORS.get(severity, (_BLUE, _WHITE))
            label = severity.upper()
            self.set_font("Helvetica", "B", 8)
            w = self.get_string_width(label) + 5
            self.set_fill_color(*bg)
            self.set_text_color(*fg)
            self.cell(w, 5, f" {label} ", fill=True, new_x="RIGHT")
            self.set_text_color(*_BODY)
            self.cell(4, 5, "")  # spacer

        def code_block(self, text: str):
            """Prompt box: grey bg, border, monospace text."""
            safe = _safe_text(text[:800])
            x = self.l_margin
            w = self.w - self.l_margin - self.r_margin
            pad = 4  # mm internal padding

            # Two-pass: first measure, then draw box, then render
            self.set_font("Courier", "", 8)
            box_y = self.get_y()
            # Measure by rendering to nowhere
            self.set_xy(x + pad, box_y + pad)
            start_measure = self.get_y()
            self.multi_cell(w - 2 * pad, 3.8, safe, dry_run=True)
            text_h = self.get_y() - start_measure
            # dry_run doesn't move cursor in all fpdf2 versions,
            # so also estimate from line count
            line_count = safe.count("\n") + 1
            # Account for wrapping: estimate chars per line
            chars_per_line = max(1, int((w - 2 * pad) / 1.9))
            wrapped_lines = sum(
                max(1, (len(ln) + chars_per_line - 1) // chars_per_line)
                for ln in safe.split("\n")
            )
            text_h = max(text_h, wrapped_lines * 3.8)
            box_h = text_h + 2 * pad

            # Draw background + border
            self.set_fill_color(*_CODE_BG)
            self.set_draw_color(*_CODE_BORDER)
            self.set_line_width(0.35)
            self.rect(x, box_y, w, box_h, style="FD")

            # Render text inside
            self.set_font("Courier", "", 8)
            self.set_text_color(*_BODY)
            self.set_xy(x + pad, box_y + pad)
            self.multi_cell(w - 2 * pad, 3.8, safe)
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
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_SUBTLE)
    date = _dt.date.today().isoformat()
    pdf.cell(0, 4, f"Edition {edition}  |  {date}")
    pdf.ln(6)

    # Project summary
    if report.project_description:
        pdf.section_heading("Project Summary")
        pdf.body_text(report.project_description)

    # Test overview — compact single-line stats
    stats = (
        f"Scenarios: {report.scenarios_run} run, "
        f"{report.scenarios_passed} passed, "
        f"{report.scenarios_failed} failed"
    )
    if report.scenarios_incomplete:
        stats += f", {report.scenarios_incomplete} could not test"
    if report.total_errors:
        stats += f" | {report.total_errors} errors"
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*_BODY)
    pdf.multi_cell(0, 4.5, stats)
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
        for f in group:
            _render_pdf_finding(pdf, f)

    # HTTP-tested summary
    if http_tested:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_SUBTLE)
        pdf.multi_cell(0, 5, _safe_text(_http_tested_summary(http_tested)))
        pdf.ln(6)

    # Performance summary table + confidence + callout — keep together
    rows = _dedup_by_label(report.degradation_points)
    has_actionable = any(
        f.severity in ("critical", "warning") for f in report.findings
    )
    # Estimate total height of remaining content
    tail_h = 0
    if rows:
        tail_h += 10 + 6 + len(rows) * 6 + 4  # heading + header + rows + gap
    if report.confidence_note:
        tail_h += 10
    if has_actionable:
        tail_h += 25  # callout box
    remaining = pdf.h - pdf.get_y() - 18  # footer margin
    if tail_h > 0 and remaining < tail_h:
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


def _draw_card_border(pdf, color: tuple, margin: float, y_start: float, y_end: float) -> None:
    """Draw a coloured left border for a finding card."""
    pdf.set_draw_color(*color)
    pdf.set_line_width(1.0)
    pdf.line(margin + 0.5, y_start, margin + 0.5, y_end)
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

    # Badge + title on one line
    pdf.set_x(original_margin + card_indent)
    pdf.severity_badge(f.severity)
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_HEADING)
    pdf.cell(0, 5, _safe_text(title))
    pdf.ln(6)

    # Indent card content
    pdf.set_left_margin(original_margin + card_indent)

    # What we found — description integrated with details
    pdf.body_label("What we found:")
    combined = _integrate_details(f.description or "", f.details or "")
    if combined:
        pdf.body_text(combined)

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
        card_end_y = pdf.get_y()
        pdf.set_left_margin(original_margin)
        _draw_card_border(pdf, border_color, original_margin, card_start_y, card_end_y)
        pdf.ln(4)
        return

    # What this means for you
    consequence = _consequence_for_user(f)
    if consequence:
        pdf.body_label("What this means for you:")
        pdf.body_text(consequence)

    # What to do + Prompt combined
    prompt = generate_finding_prompt(f)
    if prompt:
        pdf.body_label("What to do: paste this into your coding agent with the JSON report.")
        pdf.code_block(prompt)

    # After you fix it
    pdf.body_label("After you fix it:")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_BODY)
    pdf.cell(0, 5, "Run myCode again to verify the fix worked.")
    pdf.ln(4)

    # Draw left border
    card_end_y = pdf.get_y()
    pdf.set_left_margin(original_margin)
    _draw_card_border(pdf, border_color, original_margin, card_start_y, card_end_y)
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
