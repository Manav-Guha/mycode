#!/usr/bin/env python3
"""repo_hunter — find vibe-coded GitHub repos for myCode batch testing.

Searches the GitHub Search API for repos matching common vibe-coding
patterns, filters by activity / size / dependency count, and writes a
JSON manifest of candidates for batch discovery runs.

Internal pre-launch tool. Not shipped to users.

Usage:
    python scripts/repo_hunter.py                          # unauthenticated
    python scripts/repo_hunter.py --token ghp_xxxx         # authenticated
    python scripts/repo_hunter.py --output repos.json      # custom output
    python scripts/repo_hunter.py --max-repos 200          # cap results
    python scripts/repo_hunter.py --min-stars 5            # filter stars

No third-party dependencies — stdlib only (urllib, json, time).
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("repo_hunter")

# ── Search Queries ──

DEFAULT_QUERIES = [
    "streamlit app",
    "built with cursor",
    "fastapi openai",
    "crewai",
    "langchain agent",
    "flask chatbot",
    "yfinance dashboard",
]

# ── Rate Limiting ──

_UNAUTH_DELAY = 6.5   # 10 req/min → 6s + margin
_AUTH_DELAY = 2.5      # 30 req/min → 2s + margin

# GitHub Search API returns max 1000 results per query, 100 per page.
_PER_PAGE = 100
_MAX_PAGES = 10  # 10 pages × 100 = 1000 (API ceiling)

# ── Dep-file Patterns ──
# Files that indicate runtime dependencies when present.
_DEP_FILES = {
    "requirements.txt",
    "Pipfile",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
}

# ── API Helpers ──


def _api_get(url: str, token: str | None, delay: float) -> dict | None:
    """GET a GitHub API URL. Returns parsed JSON or None on failure."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)

    try:
        time.sleep(delay)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            # Rate limit hit — read reset header and wait
            reset = exc.headers.get("X-RateLimit-Reset")
            if reset:
                wait = max(int(reset) - int(time.time()), 1)
                logger.warning("Rate limited. Waiting %ds …", wait)
                time.sleep(wait + 1)
                return _api_get(url, token, delay)  # retry once
            logger.warning("403 Forbidden (no reset header): %s", url)
        elif exc.code == 422:
            logger.warning("Unprocessable query: %s", url)
        else:
            logger.warning("HTTP %d: %s", exc.code, url)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Request failed: %s — %s", url, exc)
        return None


def _search_repos(
    query: str,
    token: str | None,
    delay: float,
    max_pages: int,
    cutoff_date: str,
) -> list[dict]:
    """Search GitHub for repos matching *query*, pushed after *cutoff_date*.

    Returns raw repo dicts from the API.
    """
    results: list[dict] = []

    for page in range(1, max_pages + 1):
        params = urllib.parse.urlencode({
            "q": f"{query} pushed:>={cutoff_date}",
            "sort": "updated",
            "order": "desc",
            "per_page": _PER_PAGE,
            "page": page,
        })
        url = f"https://api.github.com/search/repositories?{params}"
        data = _api_get(url, token, delay)

        if data is None:
            break

        items = data.get("items", [])
        if not items:
            break

        results.extend(items)
        logger.info(
            "  [%s] page %d — %d repos (total so far: %d)",
            query, page, len(items), len(results),
        )

        # Fewer than a full page means no more results
        if len(items) < _PER_PAGE:
            break

    return results


# ── Filtering ──


def _parse_dep_count(content: str, filename: str) -> int:
    """Rough count of runtime dependencies from a dep file's content."""
    if filename == "package.json":
        try:
            pkg = json.loads(content)
            return len(pkg.get("dependencies", {}))
        except (json.JSONDecodeError, TypeError):
            return 0

    if filename == "requirements.txt":
        count = 0
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                count += 1
        return count

    if filename == "pyproject.toml":
        # Rough: count lines in [project] dependencies array
        in_deps = False
        count = 0
        for line in content.splitlines():
            stripped = line.strip()
            if stripped == "dependencies = [" or stripped.startswith("dependencies"):
                in_deps = True
                continue
            if in_deps:
                if stripped.startswith("]"):
                    break
                if stripped.startswith('"') or stripped.startswith("'"):
                    count += 1
        return count

    if filename in ("setup.py", "setup.cfg", "Pipfile"):
        # Very rough: count install_requires / [packages] lines
        count = 0
        in_section = False
        for line in content.splitlines():
            stripped = line.strip()
            if "install_requires" in stripped or "[packages]" in stripped:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("]") or stripped.startswith("["):
                    break
                if stripped and not stripped.startswith("#"):
                    count += 1
        return count

    return 0


def _get_dep_count(
    owner: str,
    repo_name: str,
    token: str | None,
    delay: float,
) -> int:
    """Fetch dependency count by reading dep files from the repo."""
    # Try each dep file — stop at the first one found
    for filename in ("requirements.txt", "package.json", "pyproject.toml",
                     "Pipfile", "setup.py", "setup.cfg"):
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}"
            f"/contents/{filename}"
        )
        data = _api_get(url, token, delay)
        if data is None:
            continue

        # Content is base64-encoded
        import base64
        encoded = data.get("content", "")
        if not encoded:
            continue
        try:
            content = base64.b64decode(encoded).decode("utf-8", errors="replace")
        except Exception:
            continue

        count = _parse_dep_count(content, filename)
        if count > 0:
            return count

    return 0


def _get_loc(
    owner: str,
    repo_name: str,
    language: str | None,
    token: str | None,
    delay: float,
) -> int:
    """Estimate source LOC via the languages endpoint.

    GitHub's languages API returns bytes per language.  Rough conversion:
    ~40 bytes/line for Python, ~45 for JS/TS.
    """
    url = f"https://api.github.com/repos/{owner}/{repo_name}/languages"
    data = _api_get(url, token, delay)
    if data is None:
        return 0

    total_bytes = sum(data.values())
    # ~40 bytes per line is a reasonable average
    return total_bytes // 40


def _passes_basic_filters(
    repo: dict,
    min_stars: int,
) -> bool:
    """Quick filters that don't require extra API calls."""
    if repo.get("fork"):
        return False
    if repo.get("is_template"):
        return False
    if repo.get("archived"):
        return False
    if (repo.get("stargazers_count") or 0) < min_stars:
        return False
    # Must have a detected language
    if not repo.get("language"):
        return False
    return True


def _build_candidate(repo: dict) -> dict:
    """Extract the fields we care about from a raw repo dict."""
    return {
        "repo_url": repo["html_url"],
        "clone_url": repo["clone_url"],
        "stars": repo.get("stargazers_count", 0),
        "last_commit_date": repo.get("pushed_at", ""),
        "language": repo.get("language", ""),
        "description": (repo.get("description") or "")[:200],
    }


# ── Main Logic ──


def hunt(
    queries: list[str],
    token: str | None = None,
    output: Path = Path("discovered_repos.json"),
    max_repos: int = 500,
    min_stars: int = 0,
    min_loc: int = 100,
    min_deps: int = 3,
    max_pages_per_query: int = 3,
) -> list[dict]:
    """Run the full hunt pipeline. Returns list of candidate dicts."""
    delay = _AUTH_DELAY if token else _UNAUTH_DELAY

    # Cutoff: repos pushed within the last 12 months
    now = datetime.now(timezone.utc)
    cutoff = now.replace(year=now.year - 1).strftime("%Y-%m-%d")

    # Phase 1: Search
    logger.info("Phase 1: Searching GitHub (%d queries) …", len(queries))
    seen_ids: set[int] = set()
    raw_repos: list[dict] = []

    for query in queries:
        logger.info("Searching: %r", query)
        results = _search_repos(query, token, delay, max_pages_per_query, cutoff)
        for repo in results:
            rid = repo["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                raw_repos.append(repo)

    logger.info("Phase 1 complete: %d unique repos found", len(raw_repos))

    # Phase 2: Basic filters (no extra API calls)
    logger.info("Phase 2: Basic filtering …")
    basic_passed: list[dict] = [
        r for r in raw_repos if _passes_basic_filters(r, min_stars)
    ]
    logger.info(
        "Phase 2 complete: %d/%d passed (not fork, not template, not archived)",
        len(basic_passed), len(raw_repos),
    )

    # Phase 3: Deep filters (LOC + dep count — requires API calls)
    logger.info("Phase 3: Deep filtering (LOC >= %d, deps >= %d) …", min_loc, min_deps)
    candidates: list[dict] = []

    for i, repo in enumerate(basic_passed):
        if len(candidates) >= max_repos:
            logger.info("Reached max repos cap (%d). Stopping.", max_repos)
            break

        owner = repo["owner"]["login"]
        name = repo["name"]
        full = f"{owner}/{name}"

        # Check LOC
        loc = _get_loc(owner, name, repo.get("language"), token, delay)
        if loc < min_loc:
            logger.debug("  SKIP %s — LOC %d < %d", full, loc, min_loc)
            continue

        # Check dep count
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

    logger.info("Phase 3 complete: %d candidates", len(candidates))

    # Write output
    output.write_text(json.dumps(candidates, indent=2) + "\n")
    logger.info("Wrote %d repos to %s", len(candidates), output)

    return candidates


# ── CLI ──


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search GitHub for vibe-coded repos suitable for myCode batch testing.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub personal access token (raises rate limit from 10 to 30 req/min)",
    )
    parser.add_argument(
        "--output", "-o",
        default="discovered_repos.json",
        help="Output JSON file path (default: discovered_repos.json)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=500,
        help="Maximum repos to include in output (default: 500)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Max result pages per search query (default: 3, max 10)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=0,
        help="Minimum star count (default: 0)",
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
        "--queries",
        nargs="+",
        default=None,
        help="Custom search queries (overrides defaults)",
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

    queries = args.queries or DEFAULT_QUERIES

    candidates = hunt(
        queries=queries,
        token=args.token,
        output=Path(args.output),
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        min_loc=args.min_loc,
        min_deps=args.min_deps,
        max_pages_per_query=min(args.max_pages, _MAX_PAGES),
    )

    print(f"\nDone. {len(candidates)} repos written to {args.output}")


if __name__ == "__main__":
    main()
