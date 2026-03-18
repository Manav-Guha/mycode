#!/usr/bin/env python3
"""lovable_scraper — collect Lovable-generated repos from GPT-Engineer-App org.

Paginates through all public repos in the GPT-Engineer-App GitHub org,
filters for TypeScript/React projects (has package.json, not archived,
not empty), and writes a JSON manifest compatible with batch_mine.py.

Internal pre-launch tool. Not shipped to users.

Usage:
    python scripts/lovable_scraper.py --token ghp_xxxx
    python scripts/lovable_scraper.py --token ghp_xxxx --max-repos 200
    python scripts/lovable_scraper.py --token ghp_xxxx --skip-deep-filter
    python scripts/lovable_scraper.py --token ghp_xxxx --output lovable_repos.json

No third-party dependencies — stdlib only (urllib, json, time).
"""

import argparse
import gc
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Import shared helpers from repo_hunter (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from repo_hunter import _api_get, _get_loc, _get_dep_count, _build_candidate

logger = logging.getLogger("lovable_scraper")

# ── Constants ──

_ORG = "GPT-Engineer-App"
_ORG_REPOS_URL = f"https://api.github.com/orgs/{_ORG}/repos"
_PER_PAGE = 100

# Rate limiting
_AUTH_DELAY = 2.5    # authenticated: 5,000 req/hr
_UNAUTH_DELAY = 6.5  # unauthenticated: 60 req/hr


# ── Org Pagination ──


def _list_org_repos(
    token: str | None,
    delay: float,
) -> list[dict]:
    """Paginate through all public repos in the GPT-Engineer-App org.

    Returns raw repo dicts from the API, sorted by most recently pushed.
    """
    all_repos: list[dict] = []
    page = 1

    while True:
        params = urllib.parse.urlencode({
            "type": "public",
            "sort": "pushed",
            "direction": "desc",
            "per_page": _PER_PAGE,
            "page": page,
        })
        url = f"{_ORG_REPOS_URL}?{params}"
        data = _api_get(url, token, delay)

        if data is None:
            logger.warning("Failed to fetch page %d, stopping pagination", page)
            break

        if not isinstance(data, list):
            logger.warning("Unexpected response type on page %d: %s", page, type(data).__name__)
            break

        if not data:
            break

        all_repos.extend(data)
        logger.info(
            "  Page %d — %d repos (total so far: %d)",
            page, len(data), len(all_repos),
        )

        if len(data) < _PER_PAGE:
            break

        page += 1

    return all_repos


# ── Filtering ──


def _passes_basic_filters(repo: dict) -> bool:
    """Quick filters that don't require extra API calls."""
    if repo.get("archived"):
        return False
    if repo.get("size", 0) == 0:
        return False
    if repo.get("fork"):
        return False
    return True


def _has_package_json(
    owner: str,
    repo_name: str,
    token: str | None,
    delay: float,
) -> bool:
    """Check if the repo has a package.json at root (single API call)."""
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/package.json"
    data = _api_get(url, token, delay)
    return data is not None


# ── Main Logic ──


def scrape(
    token: str | None = None,
    output: Path = Path("lovable_repos.json"),
    max_repos: int = 500,
    min_loc: int = 100,
    min_deps: int = 3,
    skip_deep_filter: bool = False,
) -> list[dict]:
    """Run the full scrape pipeline. Returns list of candidate dicts."""
    delay = _AUTH_DELAY if token else _UNAUTH_DELAY

    # Phase 1: List all org repos
    logger.info("Phase 1: Listing %s org repos …", _ORG)
    raw_repos = _list_org_repos(token, delay)
    logger.info("Phase 1 complete: %d repos listed", len(raw_repos))

    # Phase 2: Basic filters (no extra API calls)
    logger.info("Phase 2: Basic filtering …")
    basic_passed = [r for r in raw_repos if _passes_basic_filters(r)]
    logger.info(
        "Phase 2 complete: %d/%d passed (not archived, not empty, not fork)",
        len(basic_passed), len(raw_repos),
    )
    del raw_repos
    gc.collect()

    # Phase 3: Check for package.json (1 API call per repo)
    logger.info("Phase 3: Checking for package.json …")
    has_pkg: list[dict] = []
    for i, repo in enumerate(basic_passed):
        owner = repo["owner"]["login"]
        name = repo["name"]
        if _has_package_json(owner, name, token, delay):
            has_pkg.append(repo)
        if i % 50 == 0:
            logger.info(
                "  Checked %d/%d — %d with package.json",
                i + 1, len(basic_passed), len(has_pkg),
            )
            gc.collect()
    logger.info(
        "Phase 3 complete: %d/%d have package.json",
        len(has_pkg), len(basic_passed),
    )

    # Phase 4: Deep filters (LOC + dep count) or fast path
    if skip_deep_filter:
        logger.info("Phase 4: Skipping deep filter (--skip-deep-filter)")
        candidates = []
        for repo in has_pkg:
            if len(candidates) >= max_repos:
                break
            candidate = _build_candidate(repo)
            candidate["loc_estimate"] = 0
            candidate["dep_count"] = 0
            candidates.append(candidate)
    else:
        logger.info(
            "Phase 4: Deep filtering (LOC >= %d, deps >= %d) …",
            min_loc, min_deps,
        )
        candidates = []
        for i, repo in enumerate(has_pkg):
            if len(candidates) >= max_repos:
                logger.info("Reached max repos cap (%d). Stopping.", max_repos)
                break

            owner = repo["owner"]["login"]
            name = repo["name"]
            full = f"{owner}/{name}"

            loc = _get_loc(owner, name, repo.get("language"), token, delay)
            if loc < min_loc:
                logger.debug("  SKIP %s — LOC %d < %d", full, loc, min_loc)
                continue

            dep_count = _get_dep_count(owner, name, token, delay)
            if dep_count < min_deps:
                logger.debug("  SKIP %s — deps %d < %d", full, dep_count, min_deps)
                continue

            candidate = _build_candidate(repo)
            candidate["loc_estimate"] = loc
            candidate["dep_count"] = dep_count
            candidates.append(candidate)

            logger.info(
                "  [%d/%d] PASS %s (★%d, LOC ~%d, deps %d)",
                len(candidates), max_repos, full,
                repo.get("stargazers_count", 0), loc, dep_count,
            )

            if i % 50 == 0:
                gc.collect()

    logger.info("Phase 4 complete: %d candidates", len(candidates))

    # Write output
    output.write_text(json.dumps(candidates, indent=2) + "\n")
    logger.info("Wrote %d repos to %s", len(candidates), output)

    return candidates


# ── CLI ──


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Lovable-generated repos from GPT-Engineer-App GitHub org.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub personal access token (required for 3,300+ repos)",
    )
    parser.add_argument(
        "--output", "-o",
        default="lovable_repos.json",
        help="Output JSON file path (default: lovable_repos.json)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=500,
        help="Maximum repos to include in output (default: 500)",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=100,
        help="Minimum estimated source lines of code (default: 100)",
    )
    parser.add_argument(
        "--min-deps",
        type=int,
        default=3,
        help="Minimum runtime dependency count (default: 3)",
    )
    parser.add_argument(
        "--skip-deep-filter",
        action="store_true",
        help="Skip LOC/dep-count checks (faster, ~1hr vs ~2.5hr)",
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

    candidates = scrape(
        token=args.token,
        output=Path(args.output),
        max_repos=args.max_repos,
        min_loc=args.min_loc,
        min_deps=args.min_deps,
        skip_deep_filter=args.skip_deep_filter,
    )

    print(f"\nDone. {len(candidates)} repos written to {args.output}")


if __name__ == "__main__":
    main()
