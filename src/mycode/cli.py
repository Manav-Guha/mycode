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
    --verbose / -v        Enable verbose logging
"""

import argparse
import logging
import sys
from pathlib import Path

from mycode.interface import TerminalIO
from mycode.pipeline import PipelineConfig, run_pipeline
from mycode.scenario import LLMConfig


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

    # Build LLM config
    llm_config = None
    if args.api_key:
        kwargs: dict = {"api_key": args.api_key}
        if args.api_base:
            kwargs["base_url"] = args.api_base
        if args.model:
            kwargs["model"] = args.model
        llm_config = LLMConfig(**kwargs)

    # Offline mode: explicit flag or no API key
    offline = args.offline or (args.api_key is None)

    # Build pipeline config
    config = PipelineConfig(
        project_path=project,
        language=args.language,
        llm_config=llm_config,
        offline=offline,
        skip_version_check=args.skip_version_check,
        consent=args.consent,
        io=TerminalIO(),
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

    # Recording path
    if result.recording_path:
        print(f"\nSession recorded: {result.recording_path}")

    return 0 if result.success else 1
