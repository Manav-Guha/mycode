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
    --report              Write markdown report to the project directory
    --containerised       Run analysis inside a Docker container (full isolation)
    --python-version VER  Python version for the Docker container (default: 3.11)
    --yes / -y            Skip confirmation prompts
    --verbose / -v        Enable verbose logging
    --tier {1,2,3}        Analysis tier (1=inference only, 2=targeted, 3=full)
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

from mycode.interface import TerminalIO
from mycode.pipeline import (
    PipelineConfig,
    PipelineResult,
    check_llm_report_allowance,
    decrement_llm_report_counter,
    detect_language,
    run_pipeline,
)
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


_UNTRUSTED_CODE_WARNING = (
    "myCode will install this project's dependencies and execute test code "
    "in a sandboxed environment. Running untrusted code carries inherent "
    "risk. Use --containerised for full isolation. Proceed? [Y/n] "
)


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
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Write a clean markdown report (mycode-report.md) to the project directory",
    )
    parser.add_argument(
        "--containerised",
        action="store_true",
        default=False,
        help="Run analysis inside a Docker container for full isolation",
    )
    parser.add_argument(
        "--python-version",
        default="3.11",
        help="Python version for the Docker container (default: 3.11)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompts (e.g. untrusted code warning)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Analysis tier: 1=inference only (fast, no tests), "
            "2=targeted stress tests, 3=full suite (default)"
        ),
    )
    # Internal: used to pass host conversation constraints into a container
    parser.add_argument(
        "--constraints-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _collect_passthrough_args(args: argparse.Namespace) -> list[str]:
    """Build a list of CLI flags to pass through to the container.

    Excludes ``--containerised``, ``--python-version``, ``--yes``,
    ``project_path``, ``--json-output``, and ``--report`` (which cannot
    write to a read-only mount).
    """
    passthrough: list[str] = []
    if args.offline:
        passthrough.append("--offline")
    if args.language:
        passthrough.extend(["--language", args.language])
    if args.consent:
        passthrough.append("--consent")
    if args.api_key:
        passthrough.extend(["--api-key", args.api_key])
    if args.api_base:
        passthrough.extend(["--api-base", args.api_base])
    if args.model:
        passthrough.extend(["--model", args.model])
    if args.skip_version_check:
        passthrough.append("--skip-version-check")
    if args.verbose:
        passthrough.append("--verbose")
    if args.non_interactive:
        passthrough.append("--non-interactive")
    if args.tier:
        passthrough.extend(["--tier", str(args.tier)])
    if getattr(args, "constraints_file", None):
        passthrough.extend(["--constraints-file", args.constraints_file])
    return passthrough


def _run_host_conversation(project_path: Path, args: argparse.Namespace) -> str | None:
    """Run the conversational interface on the host before launching Docker.

    Performs lightweight local ingestion (no venv, no version checks) to
    gather project metadata, then runs the conversational interface to
    collect user constraints.  Serializes the result to a temp JSON file.

    Args:
        project_path: Resolved path to the user's project.
        args: Parsed CLI arguments.

    Returns:
        Path to the temp JSON constraints file, or None if conversation
        fails or produces no constraints.
    """
    from mycode.constraints import OperationalConstraints
    from mycode.interface import ConversationalInterface

    try:
        language = args.language or detect_language(project_path)
    except Exception as exc:
        logger.warning("Language detection failed for host conversation: %s", exc)
        return None

    # Lightweight ingestion — no venv, no version checks
    try:
        if language == "python":
            from mycode.ingester import ProjectIngester
            ingester = ProjectIngester(
                project_path=project_path,
                installed_packages={},
                skip_pypi_check=True,
            )
        else:
            from mycode.js_ingester import JsProjectIngester
            ingester = JsProjectIngester(
                project_path=project_path,
                installed_packages=None,
                skip_npm_check=True,
            )
        ingestion = ingester.ingest()
    except Exception as exc:
        logger.warning("Host ingestion failed: %s", exc)
        return None

    # Run conversation
    try:
        offline = args.offline or not (args.api_key or os.environ.get("GEMINI_API_KEY"))
        llm_config = None
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if api_key:
            kwargs: dict = {"api_key": api_key}
            if args.api_base:
                kwargs["base_url"] = args.api_base
            if args.model:
                kwargs["model"] = args.model
            llm_config = LLMConfig(**kwargs)

        interface = ConversationalInterface(
            llm_config=llm_config,
            offline=offline,
            io=TerminalIO(),
        )
        interface_result = interface.run(ingestion, language)
    except Exception as exc:
        logger.warning("Host conversation failed: %s", exc)
        return None

    # Serialize to temp JSON
    constraints_dict = None
    if interface_result.constraints:
        c = interface_result.constraints
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

    intent_string = interface_result.intent.as_intent_string() if interface_result.intent else ""

    data = {
        "operational_intent": intent_string,
        "constraints": constraints_dict,
    }

    try:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="mycode_constraints_")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        return path
    except Exception as exc:
        logger.warning("Could not write constraints file: %s", exc)
        return None


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

    # ── Containerised mode ──
    if args.containerised:
        from mycode.container import ContainerError, is_docker_available, run_containerised

        if not is_docker_available():
            print(
                "Error: Docker is not available. To use --containerised, "
                "install Docker and ensure the daemon is running.\n"
                "  macOS / Windows: https://www.docker.com/products/docker-desktop\n"
                "  Linux: https://docs.docker.com/engine/install/",
                file=sys.stderr,
            )
            return 1

        if args.json_output or args.report:
            print(
                "Note: --json-output and --report are not supported with "
                "--containerised (project is mounted read-only). "
                "The full report is displayed in the terminal.",
            )

        passthrough = _collect_passthrough_args(args)

        # Run conversation on host if interactive
        constraints_path = None
        is_interactive = sys.stdin.isatty()
        if not args.non_interactive and is_interactive:
            constraints_path = _run_host_conversation(project, args)
            # Container always runs non-interactive (conversation already done)
            if "--non-interactive" not in passthrough:
                passthrough.append("--non-interactive")

        try:
            return run_containerised(
                project_path=project,
                cli_args=passthrough,
                python_version=args.python_version,
                constraints_file=constraints_path,
            )
        except ContainerError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            # Clean up temp constraints file
            if constraints_path:
                try:
                    os.unlink(constraints_path)
                except OSError:
                    pass

    # ── Untrusted code warning (non-containerised mode) ──
    is_interactive = sys.stdin.isatty()
    skip_warning = args.yes or args.non_interactive or not is_interactive
    if not skip_warning:
        try:
            answer = input(_UNTRUSTED_CODE_WARNING)
            if answer.strip().lower() in ("n", "no"):
                print("Aborted. Use --containerised for full Docker isolation.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 0

    # ── Tier 1: inference only (no test execution) ──
    if args.tier == 1:
        from mycode.inference import InferenceEngine

        try:
            language = args.language or detect_language(project)
        except Exception as exc:
            print(f"Error detecting language: {exc}", file=sys.stderr)
            return 1

        # Run lightweight ingestion for dependency list
        if language == "python":
            from mycode.ingester import ProjectIngester
            ingester = ProjectIngester(
                project_path=project,
                installed_packages={},
                skip_pypi_check=True,
            )
        else:
            from mycode.js_ingester import JsProjectIngester
            ingester = JsProjectIngester(
                project_path=project,
                installed_packages=None,
                skip_npm_check=True,
            )
        try:
            ingestion = ingester.ingest()
        except Exception as exc:
            print(f"Error analyzing project: {exc}", file=sys.stderr)
            return 1

        deps = [d.name for d in ingestion.dependencies if not d.is_dev]
        files = [fa.file_path for fa in ingestion.file_analyses]

        engine = InferenceEngine()
        result = engine.infer(
            dependencies=deps,
            file_structure=files,
            file_count=ingestion.files_analyzed,
        )

        print(result.as_text())

        if args.json_output:
            json_path = project / "mycode-inference.json"
            try:
                json_path.write_text(
                    json.dumps(result.as_dict(), indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"\nJSON inference written: {json_path}")
            except Exception as exc:
                print(
                    f"\nWarning: Could not write JSON: {exc}",
                    file=sys.stderr,
                )
        return 0

    # Build LLM config — flag takes precedence, then env var
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    is_byok = bool(args.api_key)  # User explicitly provided --api-key
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

    # Three free LLM reports — only for non-BYOK users
    if not offline and not is_byok:
        remaining = check_llm_report_allowance()
        if remaining <= 0:
            print(
                "You've used your 3 free AI-powered analyses. "
                "Run with --offline for template-based analysis, "
                "or add your own API key with --api-key."
            )
            offline = True
        else:
            decrement_llm_report_counter()
            if remaining == 1:
                print("This is your last free AI-powered analysis.")

    # Load prebuilt constraints from host conversation (inside container)
    prebuilt_constraints = None
    prebuilt_intent = ""
    if args.constraints_file:
        try:
            from mycode.constraints import OperationalConstraints
            with open(args.constraints_file, encoding="utf-8") as f:
                cdata = json.load(f)
            prebuilt_intent = cdata.get("operational_intent", "")
            if cdata.get("constraints"):
                prebuilt_constraints = OperationalConstraints(**{
                    k: v for k, v in cdata["constraints"].items()
                    if v is not None
                })
        except Exception as exc:
            logger.warning("Could not load constraints file: %s", exc)

    # Build pipeline config
    non_interactive = args.non_interactive or not is_interactive
    intent = prebuilt_intent or ("General stress testing" if non_interactive else "")
    config = PipelineConfig(
        project_path=project,
        language=args.language,
        llm_config=llm_config,
        offline=offline,
        skip_version_check=args.skip_version_check,
        consent=args.consent,
        io=TerminalIO(),
        operational_intent=intent,
        auto_approve_scenarios=non_interactive,
        prebuilt_constraints=prebuilt_constraints,
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

    # Edition documents (always generated when report exists)
    if result.report:
        try:
            from mycode.documents import write_edition_documents
            understanding_path, edition = write_edition_documents(
                report=result.report,
                project_name=project.name,
                project_path=project,
            )
            print(f"\nEdition {edition} report written:")
            print(f"  {understanding_path}")
        except Exception as exc:
            print(
                f"\nWarning: Could not write edition documents: {exc}",
                file=sys.stderr,
            )

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

    # Markdown report
    if args.report and result.report:
        md_path = project / "mycode-report.md"
        try:
            md_content = result.report.as_markdown(
                project_name=project.name,
            )
            md_path.write_text(md_content, encoding="utf-8")
            print(f"\nMarkdown report written: {md_path}")
        except Exception as exc:
            print(
                f"\nWarning: Could not write markdown report: {exc}",
                file=sys.stderr,
            )

    # Recording path
    if result.recording_path:
        print(f"\nSession recorded: {result.recording_path}")

    # Exit code 2 = baseline viability failure (environment too broken)
    if result.report and getattr(result.report, "baseline_failed", False):
        return 2

    return 0 if result.success else 1
