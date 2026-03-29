#!/usr/bin/env python3
"""Build training data from corpus reports for the prediction model.

Reads every corpus/reports/*/mycode-report.json, extracts feature vectors
and multi-label targets, outputs:
  - src/mycode/data/training_data.csv
  - src/mycode/data/target_columns.json
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# Add src to path so we can import prediction constants
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycode.prediction import (
    PROFILED_DEPS,
    SERVER_FRAMEWORK_DEPS,
    normalize_dep_name,
)

# ── Config ──

CORPUS_DIR = Path(__file__).parent.parent / "corpus" / "reports"
PATTERNS_FILE = (
    Path(__file__).parent.parent / "corpus" / "extraction"
    / "corpus_patterns_ranked.json"
)
OUTPUT_DIR = Path(__file__).parent.parent / "src" / "mycode" / "data"

# Minimum confirmed_count for a pattern to become a target column.
MIN_CONFIRMED = 10

# Patterns to skip as targets (informational, not failure indicators).
_SKIP_TARGET_RE = re.compile(
    r"^\d+ (missing|unrecognized) dependenc|"
    r"^Application handled HTTP load",
)


def _sanitize_col(title: str) -> str:
    """Sanitize a pattern title into a column name."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return f"target_{s}"[:80]


def _load_patterns() -> list[dict]:
    """Load corpus patterns and select those eligible as targets."""
    if not PATTERNS_FILE.exists():
        alt = Path(__file__).parent.parent / "corpus_extraction" / "corpus_patterns_ranked.json"
        if alt.exists():
            data = json.loads(alt.read_text())
        else:
            print(f"ERROR: Cannot find {PATTERNS_FILE}")
            sys.exit(1)
    else:
        data = json.loads(PATTERNS_FILE.read_text())

    targets = []
    seen_cols: set[str] = set()
    for p in data:
        title = p.get("title", "")
        confirmed = p.get("confirmed_count", 0)
        if confirmed < MIN_CONFIRMED:
            continue
        if _SKIP_TARGET_RE.search(title):
            continue
        # Skip purely informational patterns
        sev_dist = p.get("severity_distribution", {})
        if sev_dist and all(k == "info" for k in sev_dist):
            continue
        # Deduplicate by sanitized column name (keep highest confirmed)
        col = _sanitize_col(title)
        if col in seen_cols:
            continue
        seen_cols.add(col)
        targets.append(p)

    return targets


def _extract_project_features(report: dict) -> dict:
    """Extract feature dict from a single report."""
    proj = report.get("project", {})

    # Dependency features
    raw_deps = [d["name"] for d in proj.get("dependencies", [])]
    dep_set: set[str] = set()
    for raw in raw_deps:
        canonical = normalize_dep_name(raw)
        if canonical and canonical in PROFILED_DEPS:
            dep_set.add(canonical)

    features: dict = {}
    for dep in PROFILED_DEPS:
        features[f"dep_{dep}"] = 1 if dep in dep_set else 0

    # Complexity features
    features["dep_count"] = len(raw_deps)
    features["loc"] = proj.get("total_lines", 0)
    features["file_count"] = proj.get("files_analyzed", 0)
    features["files_failed"] = proj.get("files_failed", 0)
    features["has_server_framework"] = int(bool(dep_set & SERVER_FRAMEWORK_DEPS))

    # Language
    lang = proj.get("language", "python").lower()
    features["language"] = 1 if lang == "javascript" else 0

    return features


def _extract_targets(report: dict, target_patterns: list[dict]) -> dict:
    """Extract multi-label binary targets from a report's findings."""
    # Build set of finding titles for matching
    finding_titles = set()
    finding_categories = set()
    finding_domains = set()
    finding_patterns = set()
    for f in report.get("findings", []):
        finding_titles.add(f.get("title", "").lower().strip())
        finding_categories.add(f.get("category", ""))
        if f.get("failure_domain"):
            finding_domains.add(f["failure_domain"])
        if f.get("failure_pattern"):
            finding_patterns.add(f["failure_pattern"])

    targets: dict = {}
    for pattern in target_patterns:
        col = _sanitize_col(pattern["title"])
        title_lower = pattern["title"].lower().strip()

        # Match by exact title (normalized)
        matched = False
        for ft in finding_titles:
            # Check if the pattern title is a substring or vice versa
            if title_lower in ft or ft in title_lower:
                matched = True
                break
            # Check significant word overlap (≥3 shared words)
            p_words = set(title_lower.split())
            f_words = set(ft.split())
            if len(p_words & f_words) >= 3:
                matched = True
                break

        # Also match by category + failure_domain + failure_pattern combo
        if not matched:
            p_cat = pattern.get("category", "")
            p_domain = pattern.get("failure_domain", "")
            p_pattern = pattern.get("failure_pattern", "")
            if p_cat and p_domain:
                if p_cat in finding_categories and p_domain in finding_domains:
                    if not p_pattern or p_pattern in finding_patterns:
                        matched = True

        targets[col] = 1 if matched else 0

    return targets


def main():
    print("Loading target patterns...")
    target_patterns = _load_patterns()
    print(f"  {len(target_patterns)} target patterns selected")

    # Build target column info
    target_info = {}
    for p in target_patterns:
        col = _sanitize_col(p["title"])
        sev_dist = p.get("severity_distribution", {})
        dominant_sev = max(sev_dist, key=sev_dist.get) if sev_dist else "info"
        target_info[col] = {
            "title": p["title"],
            "severity": dominant_sev,
            "category": p.get("category", ""),
            "confirmed_count": p.get("confirmed_count", 0),
        }

    print(f"\nScanning {CORPUS_DIR}...")
    report_dirs = sorted(d for d in os.listdir(CORPUS_DIR)
                         if os.path.isdir(CORPUS_DIR / d))
    print(f"  {len(report_dirs)} report directories found")

    rows = []
    errors = 0
    for dirname in report_dirs:
        rp = CORPUS_DIR / dirname / "mycode-report.json"
        if not rp.is_file():
            continue
        try:
            report = json.loads(rp.read_text())
        except (json.JSONDecodeError, OSError):
            errors += 1
            continue

        if "project" not in report:
            continue

        features = _extract_project_features(report)
        targets = _extract_targets(report, target_patterns)
        row = {**features, **targets}
        rows.append(row)

    print(f"  {len(rows)} valid reports processed, {errors} errors")

    if not rows:
        print("ERROR: No valid data. Aborting.")
        sys.exit(1)

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "training_data.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Wrote {csv_path} ({len(rows)} rows × {len(fieldnames)} columns)")

    # Write target columns JSON
    target_cols_path = OUTPUT_DIR / "target_columns.json"
    # Ordered list of target column names
    target_col_names = [_sanitize_col(p["title"]) for p in target_patterns]
    with open(target_cols_path, "w") as f:
        json.dump({
            "target_columns": target_col_names,
            "target_info": target_info,
        }, f, indent=2)
    print(f"  Wrote {target_cols_path}")

    # Summary stats
    feature_cols = [c for c in fieldnames if not c.startswith("target_")]
    target_cols = [c for c in fieldnames if c.startswith("target_")]
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")

    # Show target distribution
    print("\n  Target distribution:")
    for col in target_cols:
        positives = sum(1 for r in rows if r.get(col, 0) == 1)
        pct = positives / len(rows) * 100
        title = target_info.get(col, {}).get("title", col)
        print(f"    {positives:5d} ({pct:5.1f}%) {title[:60]}")


if __name__ == "__main__":
    main()
