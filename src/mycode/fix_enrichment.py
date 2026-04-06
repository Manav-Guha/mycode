"""LLM-enhanced fix suggestions for diagnostic findings.

Reads the function body from the ingester's line-range data and passes it
to Gemini (or any OpenAI-compatible endpoint) alongside the diagnostic
context.  Returns a concise, line-referenced fix suggestion.  Falls back
silently to empty string on any failure — the deterministic fix prompt is
always the baseline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycode.ingester import IngestionResult
    from mycode.report import Finding
    from mycode.scenario import LLMConfig

logger = logging.getLogger(__name__)

# Default model for fix-prompt enrichment.  Can be upgraded post-seed
# without a code change by setting MYCODE_FIX_LLM_MODEL in the environment.
LLM_FIX_MODEL = "gemini-2.0-flash"

_MAX_FUNCTION_LINES = 80
_MAX_RESPONSE_WORDS = 100
_TIMEOUT_SECONDS = 8.0
_MAX_TOKENS = 256
_TEMPERATURE = 0.2
_MAX_RETRIES = 1

_SYSTEM_PROMPT = (
    "You are a code diagnostic assistant for myCode, a stress-testing tool. "
    "Given a function's source code and a diagnostic finding from a stress "
    "test, provide a specific, line-referenced fix suggestion.\n\n"
    "Rules:\n"
    "- Reference specific line numbers or variable/function names from the code.\n"
    "- Be concise: 2-4 sentences maximum.\n"
    "- Be actionable: say exactly what to change and where.\n"
    "- Do not wrap output in markdown fences or formatting.\n"
    "- Do not repeat the diagnosis — only provide the fix.\n"
    "- If the code does not clearly relate to the finding, reply with "
    "exactly: NO_SUGGESTION"
)

_USER_TEMPLATE = (
    "Finding: [{severity}] {title}\n"
    "File: {source_file} → {source_function}()\n"
    "Diagnosis: {diagnosis}\n"
    "Dependencies involved: {deps}\n"
    "{load_line}"
    "\n"
    "Source code ({source_file}, lines {lineno}-{end_lineno}):\n"
    "```{language}\n"
    "{function_body}\n"
    "```\n"
    "\n"
    "What specific code change fixes this?"
)

_ERROR_INDICATORS = ("i cannot", "i'm sorry", "i am sorry", "as an ai")


# ── Public API ──


def extract_function_body(
    ingestion: "IngestionResult",
    source_file: str,
    source_function: str,
) -> tuple[str, int, int]:
    """Extract a function's source code using ingester line-range data.

    Returns:
        (body, lineno, end_lineno) on success.
        ("", 0, 0) on any failure.
    """
    if not source_file or not source_function:
        return ("", 0, 0)

    # Find the matching FileAnalysis
    file_analysis = None
    for fa in ingestion.file_analyses:
        if fa.file_path == source_file:
            file_analysis = fa
            break

    if file_analysis is None:
        logger.debug("fix_enrichment: file %s not in file_analyses", source_file)
        return ("", 0, 0)

    # Find the matching FunctionInfo
    func_info = None
    for fi in file_analysis.functions:
        if fi.name == source_function:
            func_info = fi
            break

    if func_info is None:
        logger.debug(
            "fix_enrichment: function %s not in %s", source_function, source_file,
        )
        return ("", 0, 0)

    if func_info.end_lineno == 0:
        logger.debug(
            "fix_enrichment: end_lineno=0 for %s in %s", source_function, source_file,
        )
        return ("", 0, 0)

    # Read the file from the project directory
    project_path = Path(ingestion.project_path)
    file_path = project_path / source_file
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        logger.debug("fix_enrichment: could not read %s", file_path)
        return ("", 0, 0)

    # Extract the function body (1-indexed → 0-indexed)
    start = func_info.lineno - 1
    end = func_info.end_lineno
    body_lines = lines[start:end]

    if not body_lines:
        logger.debug("fix_enrichment: empty body for %s", source_function)
        return ("", 0, 0)

    # Truncate long functions
    if len(body_lines) > _MAX_FUNCTION_LINES:
        body_lines = body_lines[:_MAX_FUNCTION_LINES]
        body_lines.append("    # ... truncated")

    body = "\n".join(body_lines)
    return (body, func_info.lineno, func_info.end_lineno)


def get_llm_fix_suggestion(
    finding: "Finding",
    function_body: str,
    lineno: int,
    end_lineno: int,
    llm_config: "LLMConfig",
    language: str = "python",
) -> str:
    """Call the LLM for a line-specific fix suggestion.

    Returns the suggestion text, or empty string on any failure.
    All failures are silent — logged at DEBUG only.
    """
    from mycode.documents import _build_diagnosis
    from mycode import scenario as _scenario_mod

    if not llm_config or not llm_config.api_key:
        logger.debug("fix_enrichment: no API key, skipping LLM fix")
        return ""

    if not function_body:
        return ""

    # Build the fix-specific LLM config
    model = os.environ.get("MYCODE_FIX_LLM_MODEL") or LLM_FIX_MODEL
    fix_config = _scenario_mod.LLMConfig(
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=_TEMPERATURE,
        timeout_seconds=_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )

    # Build the user message
    diagnosis = _build_diagnosis(finding) or finding.diagnosis or ""
    deps = ", ".join(finding.affected_dependencies) if finding.affected_dependencies else "none"
    load_line = (
        f"Failed at load level: {finding._load_level}\n"
        if finding._load_level is not None
        else ""
    )

    user_msg = _USER_TEMPLATE.format(
        severity=finding.severity.upper(),
        title=finding.title,
        source_file=finding.source_file,
        source_function=finding.source_function,
        diagnosis=diagnosis,
        deps=deps,
        load_line=load_line,
        lineno=lineno,
        end_lineno=end_lineno,
        language=language,
        function_body=function_body,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        backend = _scenario_mod.LLMBackend(fix_config)
        response = backend.generate(messages)
    except (_scenario_mod.LLMError, Exception) as exc:
        logger.debug("fix_enrichment: LLM call failed: %s", exc)
        return ""

    content = (response.content or "").strip()

    # Validate response
    if not content:
        logger.debug("fix_enrichment: empty response")
        return ""

    if content == "NO_SUGGESTION":
        logger.debug("fix_enrichment: model returned NO_SUGGESTION")
        return ""

    if len(content.split()) > _MAX_RESPONSE_WORDS:
        logger.debug("fix_enrichment: response too long (%d words)", len(content.split()))
        return ""

    content_lower = content.lower()
    for indicator in _ERROR_INDICATORS:
        if indicator in content_lower:
            logger.debug("fix_enrichment: error indicator found: %s", indicator)
            return ""

    return content


def enrich_finding(
    finding: "Finding",
    ingestion: "IngestionResult",
    llm_config: "LLMConfig",
) -> None:
    """Enrich a single finding with an LLM fix suggestion (in-place).

    Silent no-op on any failure.  Sets ``finding.llm_fix_suggestion``.
    """
    if finding.severity == "info":
        return

    if not finding.source_file or not finding.source_function:
        return

    body, lineno, end_lineno = extract_function_body(
        ingestion, finding.source_file, finding.source_function,
    )
    if not body:
        return

    language = ingestion.language or "python"
    suggestion = get_llm_fix_suggestion(
        finding, body, lineno, end_lineno, llm_config, language,
    )
    if suggestion:
        finding.llm_fix_suggestion = suggestion
