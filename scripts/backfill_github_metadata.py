#!/usr/bin/env python3
"""Backfill github_metadata.json for existing corpus repos.

Scans corpus/reports/ for folders missing github_metadata.json,
queries the GitHub API to retrieve metadata, and saves it.

Usage:
    python3 scripts/backfill_github_metadata.py \
        --token YOUR_GITHUB_PAT \
        --corpus-dir ~/Desktop/mycode/corpus/reports

No third-party dependencies — stdlib only.
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("backfill_github_metadata")

_REQUEST_DELAY = 0.8  # seconds between API calls (safe under 5k/hr)


def _parse_owner_repo(folder_name: str) -> tuple[str, str] | None:
    """Parse 'owner__repo' folder name into (owner, repo).

    Returns None if the name doesn't contain '__'.
    """
    if "__" not in folder_name:
        return None
    parts = folder_name.split("__", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _fetch_repo_metadata(owner: str, repo: str, token: str) -> dict | None:
    """Query GitHub API for repo metadata.

    Returns parsed JSON dict, or None on non-404 errors.
    Raises a special value for 404 (deleted/private repos).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "mycode-backfill",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"_status": "not_found"}
        if exc.code == 403:
            # Rate limit hit
            logger.error("Rate limited (403). Stopping.")
            sys.exit(1)
        logger.warning("HTTP %d for %s/%s: %s", exc.code, owner, repo, exc.reason)
        return None
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Request failed for %s/%s: %s", owner, repo, exc)
        return None


def backfill(corpus_dir: Path, token: str) -> None:
    """Scan corpus_dir and backfill missing github_metadata.json files."""
    if not corpus_dir.is_dir():
        logger.error("Corpus directory not found: %s", corpus_dir)
        sys.exit(1)

    # Find folders missing github_metadata.json
    all_dirs = sorted(
        d for d in corpus_dir.iterdir()
        if d.is_dir() and not (d / "github_metadata.json").exists()
    )
    total = len(all_dirs)
    if total == 0:
        logger.info("All repos already have github_metadata.json. Nothing to do.")
        return

    logger.info("Found %d repos missing github_metadata.json", total)

    backfilled = 0
    skipped = 0
    not_found = 0

    for i, repo_dir in enumerate(all_dirs, 1):
        parsed = _parse_owner_repo(repo_dir.name)
        if parsed is None:
            logger.warning("Cannot parse owner/repo from folder: %s", repo_dir.name)
            skipped += 1
            continue

        owner, repo = parsed

        data = _fetch_repo_metadata(owner, repo, token)
        if data is None:
            skipped += 1
            if i % 100 == 0:
                logger.info("Backfilled %d/%d repos...", backfilled, total)
            time.sleep(_REQUEST_DELAY)
            continue

        meta_path = repo_dir / "github_metadata.json"

        if data.get("_status") == "not_found":
            meta = {"full_name": f"{owner}/{repo}", "status": "not_found"}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            not_found += 1
        else:
            topics = data.get("topics", [])
            meta = {
                "full_name": data.get("full_name", f"{owner}/{repo}"),
                "created_at": data.get("created_at"),
                "stars": data.get("stargazers_count"),
                "language": data.get("language"),
                "description": (data.get("description") or "")[:200],
                "topics": topics if isinstance(topics, list) else [],
                "mined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        backfilled += 1

        if i % 100 == 0:
            logger.info("Backfilled %d/%d repos...", backfilled, total)

        time.sleep(_REQUEST_DELAY)

    logger.info(
        "Done. Backfilled: %d, Not found: %d, Skipped: %d",
        backfilled, not_found, skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill github_metadata.json for existing corpus repos.",
    )
    parser.add_argument(
        "--token", required=True,
        help="GitHub personal access token",
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(Path(__file__).parent.parent / "corpus" / "reports"),
        help="Path to corpus/reports/ directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    backfill(Path(args.corpus_dir), args.token)


if __name__ == "__main__":
    main()
