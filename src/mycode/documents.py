"""Downloadable document generation and edition tracking.

Produces two documents from a DiagnosticReport:
  1. "Understanding Your Results" — human-readable diagnostic report
  2. "Recommended Fixes" — agent-parseable investigation directives

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

from mycode.report import DiagnosticReport, Finding

logger = logging.getLogger(__name__)

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


# ── PDF Rendering ──

# Colour constants
_DARK_BLUE = (26, 58, 92)     # #1a3a5c
_BODY_GREY = (51, 51, 51)     # #333333
_LIGHT_GREY = (128, 128, 128) # #808080
_RED = (220, 53, 69)          # #dc3545
_AMBER = (255, 193, 7)        # #ffc107
_BLUE = (13, 110, 253)        # #0d6efd
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

_SEVERITY_COLORS: dict[str, tuple[tuple, tuple]] = {
    # (background, text)
    "critical": (_RED, _WHITE),
    "warning": (_AMBER, _BLACK),
    "info": (_BLUE, _WHITE),
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
        """Styled PDF with myCode header and footer on every page."""

        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*_DARK_BLUE)
            self.cell(40, 10, "myCode", new_x="RIGHT")
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(*_LIGHT_GREY)
            self.cell(
                0, 10,
                "Stress test your AI-generated code before it breaks",
                align="R",
            )
            self.ln(6)
            self.set_draw_color(*_LIGHT_GREY)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(8)

        def footer(self):
            self.set_y(-20)
            self.set_draw_color(*_LIGHT_GREY)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(3)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_LIGHT_GREY)
            self.cell(0, 5, "myCode by Machine Adjacent Systems - Diagnostic tool", new_x="LEFT")
            self.cell(
                0, 5, f"Page {self.page_no()}/{{nb}}",
                align="R",
            )

        def section_heading(self, text: str, level: int = 2):
            """Render a section heading (H1=16pt, H2=14pt, H3=12pt)."""
            sizes = {1: 16, 2: 14, 3: 12}
            size = sizes.get(level, 12)
            self.set_font("Helvetica", "B", size)
            self.set_text_color(*_DARK_BLUE if level <= 2 else _BODY_GREY)
            self.multi_cell(0, size * 0.5, _safe_text(text))
            self.ln(3)

        def body_text(self, text: str):
            """Render body text in 11pt."""
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*_BODY_GREY)
            self.multi_cell(0, 6, _safe_text(text))
            self.ln(2)

        def severity_badge(self, severity: str):
            """Render an inline severity badge."""
            bg, fg = _SEVERITY_COLORS.get(severity, (_BLUE, _WHITE))
            label = severity.upper()
            self.set_font("Helvetica", "B", 9)
            w = self.get_string_width(label) + 6
            x, y = self.get_x(), self.get_y()
            self.set_fill_color(*bg)
            self.set_text_color(*fg)
            self.cell(w, 6, label, fill=True, new_x="RIGHT")
            self.set_text_color(*_BODY_GREY)
            self.cell(3, 6, " ")  # spacer

        def detail_block(self, text: str):
            """Render a quoted detail block."""
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(*_LIGHT_GREY)
            x = self.get_x()
            self.set_x(x + 5)
            self.multi_cell(0, 5, _safe_text(text[:500]))
            self.set_x(x)
            self.ln(2)

        def code_block(self, text: str):
            """Render a code/details block with grey background."""
            self.set_font("Courier", "", 9)
            self.set_text_color(*_BODY_GREY)
            self.set_fill_color(245, 245, 245)
            self.multi_cell(0, 4.5, _safe_text(text[:800]), fill=True)
            self.ln(2)

        def bullet(self, text: str):
            """Render a bullet point."""
            self.set_font("Helvetica", "", 11)
            self.set_text_color(*_BODY_GREY)
            self.cell(5, 6, "-")
            self.multi_cell(0, 6, _safe_text(text))
            self.ln(1)

        def callout_box(self, text: str):
            """Render a prominent callout box."""
            safe = _safe_text(text)
            self.set_fill_color(240, 248, 255)
            self.set_draw_color(*_DARK_BLUE)
            x = self.l_margin
            y = self.get_y()
            w = self.w - self.l_margin - self.r_margin
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*_DARK_BLUE)
            # Calculate height needed
            self.set_xy(x + 5, y + 5)
            self.multi_cell(w - 10, 6, safe)
            h = self.get_y() - y + 5
            # Draw box
            self.rect(x, y, w, h, style="D")
            self.set_fill_color(240, 248, 255)
            self.rect(x + 0.3, y + 0.3, w - 0.6, h - 0.6, style="F")
            # Re-render text on top of box
            self.set_xy(x + 5, y + 5)
            self.multi_cell(w - 10, 6, safe)
            self.ln(5)

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
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margins(25, 25, 25)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.section_heading("Understanding Your Results", level=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_LIGHT_GREY)
    date = _dt.date.today().isoformat()
    pdf.cell(0, 5, f"Edition {edition}  |  {date}")
    pdf.ln(8)

    # Project summary
    if report.project_description:
        pdf.section_heading("Project Summary")
        pdf.body_text(report.project_description)

    # Dependency stack
    has_deps = report.recognized_dep_count > 0 or report.unrecognized_deps
    if has_deps:
        pdf.section_heading("Dependency Stack")
        if report.recognized_dep_count:
            pdf.bullet(
                f"{report.recognized_dep_count} dependencies with "
                f"targeted stress profiles"
            )
        if report.unrecognized_deps:
            names = ", ".join(report.unrecognized_deps[:10])
            suffix = "..." if len(report.unrecognized_deps) > 10 else ""
            pdf.bullet(
                f"{len(report.unrecognized_deps)} dependencies tested "
                f"with usage-based analysis: {names}{suffix}"
            )

    # Test overview
    pdf.section_heading("Test Overview")
    pdf.bullet(f"Scenarios run: {report.scenarios_run}")
    pdf.bullet(f"Passed: {report.scenarios_passed}")
    pdf.bullet(f"Failed: {report.scenarios_failed}")
    if report.scenarios_incomplete:
        pdf.bullet(f"Could not test: {report.scenarios_incomplete}")
    if report.total_errors:
        pdf.bullet(f"Total errors: {report.total_errors}")

    # Findings grouped by severity
    all_findings = list(report.findings) + list(report.incomplete_tests)
    severity_groups: dict[str, list] = {"critical": [], "warning": [], "info": []}
    for f in all_findings:
        severity_groups.get(f.severity, severity_groups["info"]).append(f)

    for severity in ("critical", "warning", "info"):
        group = severity_groups[severity]
        if not group:
            continue
        pdf.section_heading(f"{severity.upper()} ({len(group)})")
        for f in group:
            _render_pdf_finding(pdf, f)

    # Degradation curves
    if report.degradation_points:
        pdf.section_heading("Performance Degradation")
        for dp in report.degradation_points:
            pdf.section_heading(dp.scenario_name, level=3)
            if dp.description:
                pdf.body_text(dp.description)
            if dp.breaking_point:
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(*_BODY_GREY)
                pdf.cell(0, 6, f"Breaking point: {dp.breaking_point}")
                pdf.ln(4)

    # Confidence note
    if report.confidence_note:
        pdf.ln(5)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_LIGHT_GREY)
        pdf.multi_cell(0, 5, _safe_text(report.confidence_note))

    return bytes(pdf.output())


def render_fixes_pdf(
    report: DiagnosticReport, edition: int, project_name: str = "",
) -> bytes:
    """Render the recommended fixes document as a styled PDF.

    Returns raw PDF bytes.  Requires fpdf2.
    """
    PDFClass = _make_pdf_class()
    if PDFClass is None:
        raise ImportError("fpdf2 is required for PDF generation")

    pdf = PDFClass(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margins(25, 25, 25)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.section_heading("Recommended Fixes", level=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*_LIGHT_GREY)
    date = _dt.date.today().isoformat()
    pdf.cell(0, 5, f"Edition {edition}  |  {date}")
    pdf.ln(8)

    # Callout box
    pdf.callout_box(
        "Copy the findings below and paste them into your coding agent "
        "(Claude Code, Cursor, Copilot, ChatGPT)."
    )

    # Collect all findings
    all_findings = list(report.findings) + list(report.incomplete_tests)
    if not all_findings:
        pdf.body_text("No findings to investigate.")
        return bytes(pdf.output())

    # Group by source file
    file_groups: dict[str, list] = {}
    for f in all_findings:
        key = f.source_file or "(no file identified)"
        file_groups.setdefault(key, []).append(f)

    severity_rank = {"critical": 0, "warning": 1, "info": 2}

    sorted_files = sorted(
        file_groups.items(),
        key=lambda item: (
            min(severity_rank.get(f.severity, 9) for f in item[1]),
            item[0],
        ),
    )

    for file_path, findings in sorted_files:
        pdf.section_heading(file_path)
        findings.sort(key=lambda f: severity_rank.get(f.severity, 9))
        for f in findings:
            _render_pdf_fix_finding(pdf, f)

    return bytes(pdf.output())


def _render_pdf_finding(pdf, f: Finding) -> None:
    """Render a single finding for the understanding PDF."""
    # Title with severity badge
    pdf.severity_badge(f.severity)
    title = f.title
    if f.group_count > 1:
        title += f" (and {f.group_count - 1} similar)"
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_BODY_GREY)
    pdf.cell(0, 6, _safe_text(title))
    pdf.ln(4)

    if f.description:
        pdf.body_text(f.description)
    if f.details:
        pdf.detail_block(f.details)
    if f.affected_dependencies:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*_BODY_GREY)
        pdf.cell(0, 5, f"Related dependencies: {', '.join(f.affected_dependencies)}")
        pdf.ln(5)


def _render_pdf_fix_finding(pdf, f: Finding) -> None:
    """Render a single finding for the fixes PDF."""
    # Title with severity tag
    pdf.severity_badge(f.severity)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_BODY_GREY)
    pdf.cell(0, 6, _safe_text(f.title))
    pdf.ln(4)

    # Metadata bullets
    if f.source_function:
        pdf.bullet(f"Function: {f.source_function}")
    if f.category:
        pdf.bullet(f"Category: {f.category}")
    if f.affected_dependencies:
        pdf.bullet(f"Dependencies: {', '.join(f.affected_dependencies)}")
    if f._load_level is not None:
        pdf.bullet(f"Failed at load level: {f._load_level}")

    if f.description:
        pdf.body_text(f.description)
    if f.details:
        pdf.code_block(f.details)

    # Investigation prompt
    if f.severity == "info":
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*_LIGHT_GREY)
        pdf.multi_cell(
            0, 5,
            "Context: This finding is informational. "
            "No investigation required unless related issues appear above.",
        )
        pdf.ln(3)
    else:
        prompt = _build_investigation_prompt(f)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*_BODY_GREY)
        pdf.cell(0, 5, "Investigate:")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _safe_text(prompt))
        pdf.ln(3)


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

    if _HAS_FPDF:
        # Generate PDFs
        date = _dt.date.today().isoformat()
        u_fname = pdf_filename(project_name, "Results", date)
        f_fname = pdf_filename(project_name, "Fixes", date)

        understanding_bytes = render_understanding_pdf(report, edition, project_name)
        fixes_bytes = render_fixes_pdf(report, edition, project_name)

        understanding_path = output_dir / u_fname
        fixes_path = output_dir / f_fname

        understanding_path.write_bytes(understanding_bytes)
        fixes_path.write_bytes(fixes_bytes)
    else:
        # Fallback to markdown
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

    # Clean up old editions — keep only the last 10
    _prune_old_editions(_REPORTS_DIR / safe_name)

    return understanding_path, fixes_path, edition


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
