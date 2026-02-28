#!/usr/bin/env python3
"""batch_mine — run myCode against repos discovered by repo_hunter.

Reads discovered_repos.json, clones each repo to a temp directory,
runs ``mycode --offline --non-interactive --json-output --skip-version-check``,
and collects the JSON report + discovery files.  Produces a per-repo
results directory and an aggregate summary (batch_results.json).

Internal pre-launch tool.  Not shipped to users.

Usage:
    python scripts/batch_mine.py                              # defaults
    python scripts/batch_mine.py --input repos.json           # custom input
    python scripts/batch_mine.py --max-repos 10               # limit
    python scripts/batch_mine.py --results-dir ./my_results   # custom output

No third-party dependencies — stdlib only.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("batch_mine")

# ── Constants ──

_DISCOVERIES_DIR = Path.home() / ".mycode" / "discoveries"
_CLONE_TIMEOUT = 120   # seconds
_MYCODE_TIMEOUT = 300  # seconds


# ── Helpers ──


def _clone_repo(clone_url: str, dest: Path) -> bool:
    """Shallow-clone a repo. Returns True on success."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", clone_url, str(dest)],
            capture_output=True,
            timeout=_CLONE_TIMEOUT,
            check=True,
        )
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Clone timed out: %s", clone_url)
        return False
    except subprocess.CalledProcessError as exc:
        logger.warning("Clone failed: %s — %s", clone_url, exc.stderr.decode(errors="replace")[:200])
        return False


def _snapshot_discoveries() -> set[str]:
    """Return the set of discovery file names currently in ~/.mycode/discoveries/."""
    if not _DISCOVERIES_DIR.is_dir():
        return set()
    return {f.name for f in _DISCOVERIES_DIR.iterdir() if f.suffix == ".json"}


def _collect_new_discoveries(before: set[str]) -> list[Path]:
    """Return paths to discovery files created after *before* snapshot."""
    if not _DISCOVERIES_DIR.is_dir():
        return []
    return [
        f for f in _DISCOVERIES_DIR.iterdir()
        if f.suffix == ".json" and f.name not in before
    ]


def _run_mycode(project_path: Path) -> tuple[int, str, str]:
    """Run mycode CLI against a project. Returns (returncode, stdout, stderr)."""
    cmd = [
        sys.executable, "-m", "mycode",
        str(project_path),
        "--offline",
        "--non-interactive",
        "--json-output",
        "--skip-version-check",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_MYCODE_TIMEOUT,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "mycode timed out"
    except Exception as exc:
        return -1, "", str(exc)


def _safe_name(repo_url: str) -> str:
    """Derive a filesystem-safe name from a repo URL.

    ``https://github.com/user/repo`` → ``user__repo``
    """
    parts = repo_url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    return parts[-1] if parts else "unknown"


def _extract_failure_signatures(report: dict) -> list[str]:
    """Pull short failure signature strings from a mycode JSON report."""
    sigs: list[str] = []
    for finding in report.get("findings", []):
        title = finding.get("title", "")
        category = finding.get("category", "")
        severity = finding.get("severity", "")
        if title:
            sigs.append(f"{severity}:{category}:{title}"[:120])
    return sigs


def _extract_unrecognized_deps(report: dict) -> list[str]:
    """Pull unrecognized dependency names from a mycode JSON report."""
    # Top-level list of strings: "unrecognized_dependencies": ["plotly", "yfinance", ...]
    return [d for d in report.get("unrecognized_dependencies", []) if isinstance(d, str)]


def _extract_dep_failures(report: dict) -> list[str]:
    """Pull dependency names associated with failures."""
    deps: list[str] = []
    for finding in report.get("findings", []):
        # Field is "affected_dependencies": ["pandas", "streamlit", ...]
        for dep in finding.get("affected_dependencies", []):
            if isinstance(dep, str):
                deps.append(dep.split("==")[0])
    return deps


# ── Main Logic ──


def mine(
    input_path: Path,
    results_dir: Path,
    max_repos: int,
) -> dict:
    """Run the full batch mining pipeline. Returns aggregate summary dict."""
    # Load repo manifest
    repos = json.loads(input_path.read_text())
    if max_repos > 0:
        repos = repos[:max_repos]

    logger.info("Loaded %d repos from %s", len(repos), input_path)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate counters
    total = len(repos)
    passed = 0
    failed = 0
    clone_errors = 0
    mycode_errors = 0
    all_failure_sigs: Counter = Counter()
    all_unrecognized_deps: Counter = Counter()
    all_dep_failures: Counter = Counter()
    repo_results: list[dict] = []

    for i, repo in enumerate(repos, 1):
        repo_url = repo.get("repo_url", "")
        clone_url = repo.get("clone_url", "")
        name = _safe_name(repo_url)

        logger.info("[%d/%d] Processing %s …", i, total, repo_url)
        repo_result_dir = results_dir / name
        repo_result_dir.mkdir(parents=True, exist_ok=True)

        entry: dict = {
            "repo_url": repo_url,
            "name": name,
            "status": "pending",
            "error": None,
            "findings_count": 0,
            "discovery_count": 0,
        }

        # Clone to temp directory
        tmp_dir = tempfile.mkdtemp(prefix="mycode_batch_")
        clone_dest = Path(tmp_dir) / "repo"

        try:
            if not _clone_repo(clone_url, clone_dest):
                entry["status"] = "clone_error"
                entry["error"] = "Failed to clone"
                clone_errors += 1
                failed += 1
                repo_results.append(entry)
                logger.warning("  SKIP — clone failed")
                continue

            # Snapshot discoveries before run
            disc_before = _snapshot_discoveries()

            # Run mycode
            returncode, stdout, stderr = _run_mycode(clone_dest)

            # Collect JSON report
            report_path = clone_dest / "mycode-report.json"
            report: dict = {}
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text())
                    # Copy to results dir
                    shutil.copy2(report_path, repo_result_dir / "mycode-report.json")
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("  Could not read report: %s", exc)

            # Collect new discoveries
            new_discoveries = _collect_new_discoveries(disc_before)
            if new_discoveries:
                disc_dest = repo_result_dir / "discoveries"
                disc_dest.mkdir(exist_ok=True)
                for dp in new_discoveries:
                    shutil.copy2(dp, disc_dest / dp.name)

            # Save stdout/stderr for debugging
            if stdout:
                (repo_result_dir / "stdout.txt").write_text(stdout)
            if stderr:
                (repo_result_dir / "stderr.txt").write_text(stderr)

            # Classify result
            if returncode == 0:
                entry["status"] = "success"
                passed += 1
            elif returncode == -1:
                entry["status"] = "mycode_error"
                entry["error"] = stderr[:200]
                mycode_errors += 1
                failed += 1
            else:
                # Non-zero exit but ran — partial results may exist
                entry["status"] = "completed_with_errors"
                passed += 1  # still counts as tested

            entry["findings_count"] = len(report.get("findings", []))
            entry["discovery_count"] = len(new_discoveries)

            # Aggregate stats from report
            sigs = _extract_failure_signatures(report)
            all_failure_sigs.update(sigs)

            unrec = _extract_unrecognized_deps(report)
            all_unrecognized_deps.update(unrec)

            dep_fails = _extract_dep_failures(report)
            all_dep_failures.update(dep_fails)

            repo_results.append(entry)

            logger.info(
                "  %s — %d findings, %d discoveries",
                entry["status"], entry["findings_count"], entry["discovery_count"],
            )

        finally:
            # Clean up temp clone
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # Build aggregate summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_repos": total,
        "repos_tested": passed,
        "repos_failed": failed,
        "clone_errors": clone_errors,
        "mycode_errors": mycode_errors,
        "repos": repo_results,
        "top_failure_signatures": [
            {"signature": sig, "count": count}
            for sig, count in all_failure_sigs.most_common(20)
        ],
        "top_unrecognized_dependencies": [
            {"dependency": dep, "count": count}
            for dep, count in all_unrecognized_deps.most_common(20)
        ],
        "failure_rate_by_dependency": [
            {"dependency": dep, "failure_count": count}
            for dep, count in all_dep_failures.most_common(20)
        ],
    }

    summary_path = results_dir / "batch_results.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote aggregate summary to %s", summary_path)

    return summary


# ── CLI ──


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run myCode against repos from repo_hunter and collect results.",
    )
    parser.add_argument(
        "--input", "-i",
        default="discovered_repos.json",
        help="Input JSON from repo_hunter.py (default: discovered_repos.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for per-repo results (default: results/)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=0,
        help="Max repos to process, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    summary = mine(
        input_path=input_path,
        results_dir=Path(args.results_dir),
        max_repos=args.max_repos,
    )

    tested = summary["repos_tested"]
    failed = summary["repos_failed"]
    total = summary["total_repos"]
    print(f"\nDone. {tested} tested, {failed} failed out of {total} repos.")
    print(f"Results in {args.results_dir}/")


if __name__ == "__main__":
    main()
