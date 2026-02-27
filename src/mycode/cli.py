"""myCode CLI — stress-test AI-generated code from the command line.

Usage::

    python -m mycode /path/to/project [options]

Options::

    --offline             Use offline mode (no LLM API calls)
    --language LANG       Force language (python or javascript)
    --consent             Opt in to anonymous interaction recording
    --api-key KEY         API key for LLM backend
    --api-base URL        Base URL for OpenAI-compatible API
    --model MODEL         Model identifier
    --skip-version-check  Skip PyPI/npm version lookups
    --non-interactive     Skip conversational interface
    --json-output         Write structured JSON report alongside terminal output
    --verbose / -v        Enable verbose logging
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mycode.interface import TerminalIO
from mycode.pipeline import PipelineConfig, PipelineResult, run_pipeline
from mycode.scenario import LLMConfig


def _build_json_report(result: PipelineResult, project_path: Path) -> dict:
    """Assemble the full structured JSON report from pipeline results.

    Combines project metadata (from ingestion), constraints (from
    conversational interface), and the diagnostic report into a single
    JSON-serializable dictionary.
    """
    report_dict = result.report.as_dict() if result.report else {}

    # Project metadata
    project_meta: dict = {
        "name": project_path.name,
        "path": str(project_path),
        "language": result.language,
    }
    if result.ingestion:
        ing = result.ingestion
        project_meta["files_analyzed"] = ing.files_analyzed
        project_meta["files_failed"] = ing.files_failed
        project_meta["total_lines"] = ing.total_lines
        project_meta["dependencies"] = [
            {
                "name": d.name,
                "installed_version": d.installed_version,
                "latest_version": d.latest_version,
                "is_outdated": d.is_outdated,
            }
            for d in ing.dependencies
            if not d.is_dev
        ]

    # Constraints from conversational interface
    constraints_dict: dict | None = None
    if result.interface_result and result.interface_result.constraints:
        c = result.interface_result.constraints
        constraints_dict = {
            "user_scale": c.user_scale,
            "usage_pattern": c.usage_pattern,
            "max_payload_mb": c.max_payload_mb,
            "data_type": c.data_type,
            "deployment_context": c.deployment_context,
            "availability_requirement": c.availability_requirement,
            "data_sensitivity": c.data_sensitivity,
            "growth_expectation": c.growth_expectation,
        }

    return {
        "project": project_meta,
        "constraints": constraints_dict,
        **report_dict,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mycode",
        description=(
            "myCode — Stress-test AI-generated code before it breaks.\n\n"
            "Point myCode at your project, describe what it does, "
            "and get a plain-language report showing where it breaks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project_path",
        type=Path,
        help="Path to the project directory to test",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Run without LLM API calls (uses offline scenario generation)",
    )
    parser.add_argument(
        "--language",
        choices=["python", "javascript"],
        default=None,
        help="Force project language (auto-detected if not specified)",
    )
    parser.add_argument(
        "--consent",
        action="store_true",
        default=False,
        help="Opt in to anonymous interaction recording",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM backend",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Base URL for OpenAI-compatible API endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model identifier",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        default=False,
        help="Skip PyPI/npm version lookups (faster)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Skip conversational interface (use default stress testing profile)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        default=False,
        help="Write structured JSON report to mycode-report.json in the project directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0=success, 1=failure)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Validate project path
    project = Path(args.project_path).resolve()
    if not project.exists():
        print(f"Error: Project path does not exist: {project}", file=sys.stderr)
        return 1
    if not project.is_dir():
        print(f"Error: Project path is not a directory: {project}", file=sys.stderr)
        return 1

    # Build LLM config — flag takes precedence, then env var
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    llm_config = None
    if api_key:
        kwargs: dict = {"api_key": api_key}
        if args.api_base:
            kwargs["base_url"] = args.api_base
        if args.model:
            kwargs["model"] = args.model
        llm_config = LLMConfig(**kwargs)

    # Offline mode: explicit flag or no API key available
    offline = args.offline or (api_key is None)

    # Build pipeline config
    non_interactive = args.non_interactive or not sys.stdin.isatty()
    config = PipelineConfig(
        project_path=project,
        language=args.language,
        llm_config=llm_config,
        offline=offline,
        skip_version_check=args.skip_version_check,
        consent=args.consent,
        io=TerminalIO(),
        operational_intent="General stress testing" if non_interactive else "",
        auto_approve_scenarios=non_interactive,
    )

    # Run pipeline
    result = run_pipeline(config)

    # Display report
    if result.report:
        print(result.report.as_text())
    elif result.execution:
        print("\n--- Execution Results (report generation failed) ---")
        print(f"Scenarios completed: {result.execution.scenarios_completed}")
        print(f"Scenarios failed: {result.execution.scenarios_failed}")
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")
    else:
        failed = result.failed_stage
        if failed:
            print(f"\nPipeline failed at stage: {failed}")
        for s in result.stages:
            if not s.success:
                print(f"  Error ({s.stage}): {s.error}")
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

    # JSON output
    if args.json_output and result.report:
        json_path = project / "mycode-report.json"
        try:
            json_report = _build_json_report(result, project)
            json_path.write_text(
                json.dumps(json_report, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"\nJSON report written: {json_path}")
        except Exception as exc:
            print(
                f"\nWarning: Could not write JSON report: {exc}",
                file=sys.stderr,
            )

    # Recording path
    if result.recording_path:
        print(f"\nSession recorded: {result.recording_path}")

    return 0 if result.success else 1
