#!/usr/bin/env python3
"""L5 Corpus Migration — classify all corpus findings against the taxonomy.

Reads all mycode-report.json files from corpus_results/*/,
runs the taxonomy classifiers on each finding, and outputs
corpus_classified.json conforming to the Library Taxonomy Schema v1.

Usage:
    python scripts/migrate_taxonomy.py [--corpus-dir DIR] [--output FILE]

Defaults:
    --corpus-dir  corpus_results/
    --output      corpus_classified.json
"""

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mycode.classifiers import (
    classify_finding,
    classify_project,
    failure_domain_classifier,
    failure_pattern_classifier,
    operational_trigger_classifier,
    vertical_classifier,
    architectural_pattern_classifier,
)


def _extract_error_type(finding: dict) -> str:
    """Extract the primary error type from finding details."""
    details = finding.get("details", "")
    title = finding.get("title", "")
    combined = f"{title} {details}"

    error_types = [
        "MemoryError", "TimeoutError", "TypeError", "ValueError",
        "KeyError", "IndexError", "ImportError", "ModuleNotFoundError",
        "FileNotFoundError", "ConnectionError", "RuntimeError",
        "AttributeError", "OSError", "PermissionError",
        "UnicodeDecodeError", "JSONDecodeError",
        "ConnectionRefusedError", "ConnectionResetError",
        "BrokenPipeError", "OutOfMemoryError",
    ]
    for etype in error_types:
        if etype in combined:
            return etype
    return ""


def _extract_scenario_name(title: str) -> str:
    """Extract scenario name from finding title.

    Titles look like:
      'Scenario failed: flask_memory_under_load'
      'Resource limit hit: concurrent_test'
      'Errors during: some_test'
    """
    if ":" in title:
        return title.split(":", 1)[1].strip()
    return title


def _detect_codebase_origin(repo_name: str) -> str:
    """Detect codebase origin from repo metadata."""
    # All L5 corpus repos are from GitHub
    return "github"


def _extract_breaking_point(finding: dict) -> str:
    """Extract breaking point from finding details if available."""
    details = finding.get("details", "")
    # Look for patterns like "at concurrent_50" or "at step compute_100"
    import re
    match = re.search(r"at\s+(\w+_\d+)", details)
    if match:
        return match.group(1)
    return ""


def _extract_metric_from_details(details: str) -> tuple:
    """Extract metric info from finding details.

    Returns (metric_name, start_value, end_value, multiplier).
    """
    import re

    # Try to extract memory info: "peak memory X MB" or "memory: X MB"
    mem_match = re.search(r"(?:peak\s+)?memory\s+(\d+\.?\d*)\s*MB", details, re.IGNORECASE)
    if mem_match:
        val = float(mem_match.group(1))
        return ("memory_peak_mb", 0.0, val, 0.0)

    # Try to extract execution time
    time_match = re.search(r"(\d+\.?\d*)\s*ms", details)
    if time_match:
        val = float(time_match.group(1))
        return ("execution_time_ms", 0.0, val, 0.0)

    # Try to extract error count: "N errors"
    err_match = re.search(r"(\d+)\s+error", details)
    if err_match:
        val = float(err_match.group(1))
        return ("error_count", 0.0, val, 0.0)

    return ("", 0.0, 0.0, 0.0)


def migrate_report(
    report_data: dict,
    repo_dir_name: str,
    batch_name: str = "L5_corpus",
) -> list[dict]:
    """Migrate a single report's findings to taxonomy-classified entries.

    Returns a list of classified entry dicts conforming to the schema.
    """
    entries = []
    project = report_data.get("project", {})
    language = project.get("language", "python")
    dependencies = [d.get("name", "") for d in project.get("dependencies", [])]
    dep_names = [d for d in dependencies if d]

    # Project-level classification
    project_cls = classify_project(
        dependencies=dep_names,
        file_structure=None,
        framework="",
        file_count=project.get("files_analyzed", 0),
        has_frontend=False,
        has_backend=False,
    )

    findings = report_data.get("findings", [])
    for finding in findings:
        title = finding.get("title", "")
        category = finding.get("category", "")
        severity = finding.get("severity", "info")
        details = finding.get("details", "")
        description = finding.get("description", "")
        affected_deps = finding.get("affected_dependencies", [])
        load_level = finding.get("load_level")

        # Skip dependency-count findings (not stress test results)
        if any(kw in title for kw in ("missing dependencies", "unrecognized dependencies",
                                       "missing dependency", "unrecognized dependency")):
            continue

        scenario_name = _extract_scenario_name(title)
        error_type = _extract_error_type(finding)

        # Run classifiers
        classification = classify_finding(
            scenario_name=scenario_name,
            scenario_category=category,
            error_type=error_type,
            error_details=f"{details} {description}",
            severity=severity,
        )

        # Extract metrics
        metric_name, start_val, end_val, multiplier = _extract_metric_from_details(
            f"{description} {details}",
        )

        entry = {
            "entry_id": str(uuid.uuid4()),
            "source": "corpus_mining",
            "source_batch": batch_name,
            "mycode_version": "0.1.2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "failure_domain": classification["failure_domain"],
            "failure_pattern": classification["failure_pattern"],
            "scenario_name": scenario_name,
            "scenario_category": category,
            "operational_trigger": classification["operational_trigger"],
            "affected_dependencies": affected_deps,
            "severity_raw": severity,
            "load_level_at_failure": str(load_level) if load_level else "",
            "breaking_point": _extract_breaking_point(finding),
            "metric_name": metric_name,
            "metric_start_value": start_val,
            "metric_end_value": end_val,
            "multiplier": multiplier,
            "codebase_origin": _detect_codebase_origin(repo_dir_name),
            "vertical": project_cls["vertical"],
            "architectural_pattern": project_cls["architectural_pattern"],
            # Corpus mining mandatory fields
            "repo_url": f"https://github.com/{repo_dir_name.replace('__', '/')}",
            "repo_last_commit_date": None,
            "repo_stars": None,
            "repo_loc": project.get("total_lines", 0),
            "repo_file_count": project.get("files_analyzed", 0),
            "dependency_count": len(dep_names),
            "profiled_dependency_count": report_data.get("statistics", {}).get(
                "recognized_dependencies", 0,
            ),
            "unrecognised_dependency_count": report_data.get("statistics", {}).get(
                "unrecognized_dependencies", 0,
            ),
        }
        entries.append(entry)

    # Also process degradation curves as entries
    for curve in report_data.get("degradation_curves", []):
        scenario_name = curve.get("scenario_name", "")
        metric = curve.get("metric", "")
        breaking_point = curve.get("breaking_point", "")
        steps = curve.get("steps", [])

        start_val = steps[0].get("value", 0.0) if steps else 0.0
        end_val = steps[-1].get("value", 0.0) if steps else 0.0
        mult = end_val / start_val if start_val > 0.001 else 0.0

        classification = classify_finding(
            scenario_name=scenario_name,
            scenario_category="",
            error_type="",
            error_details=curve.get("description", ""),
        )

        entry = {
            "entry_id": str(uuid.uuid4()),
            "source": "corpus_mining",
            "source_batch": batch_name,
            "mycode_version": "0.1.2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "failure_domain": classification["failure_domain"],
            "failure_pattern": classification["failure_pattern"],
            "scenario_name": scenario_name,
            "scenario_category": "",
            "operational_trigger": classification["operational_trigger"],
            "affected_dependencies": [],
            "severity_raw": "warning",
            "load_level_at_failure": "",
            "breaking_point": breaking_point,
            "metric_name": metric,
            "metric_start_value": start_val,
            "metric_end_value": end_val,
            "multiplier": round(mult, 2),
            "codebase_origin": _detect_codebase_origin(repo_dir_name),
            "vertical": project_cls["vertical"],
            "architectural_pattern": project_cls["architectural_pattern"],
            "repo_url": f"https://github.com/{repo_dir_name.replace('__', '/')}",
            "repo_last_commit_date": None,
            "repo_stars": None,
            "repo_loc": project.get("total_lines", 0),
            "repo_file_count": project.get("files_analyzed", 0),
            "dependency_count": len(dep_names),
            "profiled_dependency_count": report_data.get("statistics", {}).get(
                "recognized_dependencies", 0,
            ),
            "unrecognised_dependency_count": report_data.get("statistics", {}).get(
                "unrecognized_dependencies", 0,
            ),
        }
        entries.append(entry)

    return entries


def run_migration(corpus_dir: str, output_file: str) -> dict:
    """Run the full corpus migration.

    Returns summary statistics.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        print(f"Error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    all_entries: list[dict] = []
    reports_processed = 0
    reports_skipped = 0

    for repo_dir in sorted(corpus_path.iterdir()):
        if not repo_dir.is_dir():
            continue

        report_file = repo_dir / "mycode-report.json"
        if not report_file.exists():
            reports_skipped += 1
            continue

        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: Could not read {report_file}: {exc}", file=sys.stderr)
            reports_skipped += 1
            continue

        entries = migrate_report(report_data, repo_dir.name)
        all_entries.extend(entries)
        reports_processed += 1

    # Compute statistics
    domain_counts = Counter(e["failure_domain"] for e in all_entries)
    pattern_counts = Counter(
        e["failure_pattern"] for e in all_entries if e["failure_pattern"]
    )
    trigger_counts = Counter(e["operational_trigger"] for e in all_entries)
    vertical_counts = Counter(e["vertical"] for e in all_entries)
    arch_counts = Counter(e["architectural_pattern"] for e in all_entries)
    classified = sum(1 for e in all_entries if e["failure_domain"] != "unclassified")
    unclassified = sum(1 for e in all_entries if e["failure_domain"] == "unclassified")

    stats = {
        "total_entries": len(all_entries),
        "reports_processed": reports_processed,
        "reports_skipped": reports_skipped,
        "classified": classified,
        "unclassified": unclassified,
        "classification_rate": f"{classified / len(all_entries) * 100:.1f}%" if all_entries else "0%",
        "failure_domain_distribution": dict(domain_counts.most_common()),
        "failure_pattern_distribution": dict(pattern_counts.most_common(20)),
        "operational_trigger_distribution": dict(trigger_counts.most_common()),
        "vertical_distribution": dict(vertical_counts.most_common()),
        "architectural_pattern_distribution": dict(arch_counts.most_common()),
    }

    # Write output
    output = {
        "migration_metadata": {
            "schema_version": "1.0",
            "source": "L5_corpus",
            "migration_date": datetime.now(timezone.utc).isoformat(),
            "mycode_version": "0.1.2",
            "statistics": stats,
        },
        "entries": all_entries,
    }

    output_path = Path(output_file)
    output_path.write_text(
        json.dumps(output, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate L5 corpus findings to taxonomy-classified entries.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="corpus_results",
        help="Path to corpus results directory (default: corpus_results/)",
    )
    parser.add_argument(
        "--output",
        default="corpus_classified.json",
        help="Output file path (default: corpus_classified.json)",
    )
    args = parser.parse_args()

    print(f"Migrating corpus from: {args.corpus_dir}")
    print(f"Output: {args.output}")
    print()

    stats = run_migration(args.corpus_dir, args.output)

    print(f"Migration complete!")
    print(f"  Reports processed: {stats['reports_processed']}")
    print(f"  Reports skipped: {stats['reports_skipped']}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Classified: {stats['classified']}")
    print(f"  Unclassified: {stats['unclassified']}")
    print(f"  Classification rate: {stats['classification_rate']}")
    print()
    print("Failure domain distribution:")
    for domain, count in sorted(
        stats["failure_domain_distribution"].items(),
        key=lambda x: x[1], reverse=True,
    ):
        print(f"  {domain}: {count}")
    print()
    print("Top failure patterns:")
    for pattern, count in list(
        sorted(stats["failure_pattern_distribution"].items(),
               key=lambda x: x[1], reverse=True)
    )[:10]:
        print(f"  {pattern}: {count}")
    print()
    print("Vertical distribution:")
    for vertical, count in sorted(
        stats["vertical_distribution"].items(),
        key=lambda x: x[1], reverse=True,
    ):
        print(f"  {vertical}: {count}")


if __name__ == "__main__":
    main()
