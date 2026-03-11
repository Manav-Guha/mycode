"""Project fetching — clone GitHub repos or extract uploaded zips."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

# Max uncompressed size for uploaded zips (default 100 MB)
MAX_PROJECT_SIZE_BYTES = 100 * 1024 * 1024

# GitHub URL pattern — public repos only
_GITHUB_URL_RE = re.compile(
    r"^https://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
    r"(?:\.git)?(?:/.*)?$"
)


class FetchError(Exception):
    """Error fetching or extracting a project."""


def validate_github_url(url: str) -> str:
    """Validate and normalise a GitHub URL.

    Returns the normalised URL (strips trailing slashes, tree paths).
    Raises FetchError for invalid URLs.
    """
    url = url.strip().rstrip("/")

    if not _GITHUB_URL_RE.match(url):
        raise FetchError(
            "Invalid GitHub URL. Expected format: "
            "https://github.com/owner/repo"
        )

    # Strip /tree/branch or /blob/... paths — just need the repo root
    parts = url.split("/")
    if len(parts) > 5:
        # https://github.com/owner/repo/tree/main/... → keep first 5
        url = "/".join(parts[:5])

    # Strip .git suffix
    if url.endswith(".git"):
        url = url[:-4]

    return url


def clone_github_repo(url: str, dest: Path) -> Path:
    """Shallow-clone a GitHub repo to dest directory.

    Returns the path to the cloned project root.
    Raises FetchError on failure.
    """
    url = validate_github_url(url)
    clone_url = url + ".git"

    dest.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", clone_url, str(dest)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise FetchError("git is not installed. Cannot clone GitHub repos.")
    except subprocess.TimeoutExpired:
        raise FetchError("Clone timed out after 120 seconds.")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not found" in stderr.lower() or "404" in stderr:
            raise FetchError(
                "Repository not found. Check the URL and ensure "
                "the repository is public."
            )
        raise FetchError(f"Clone failed: {stderr[:200]}")

    # Remove .git directory — we don't need history
    git_dir = dest / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    return dest


def extract_zip(file_obj: BinaryIO, dest: Path) -> Path:
    """Extract an uploaded zip to dest directory.

    Returns the path to the extracted project root.
    Raises FetchError on invalid/oversized zips.
    """
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(file_obj) as zf:
            # Check for zip bombs
            total_size = sum(info.file_size for info in zf.infolist())
            if total_size > MAX_PROJECT_SIZE_BYTES:
                raise FetchError(
                    f"Zip contents too large: {total_size / 1024 / 1024:.0f} MB "
                    f"(max {MAX_PROJECT_SIZE_BYTES / 1024 / 1024:.0f} MB)"
                )

            # Check for path traversal
            for info in zf.infolist():
                target = (dest / info.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise FetchError(
                        "Zip contains path traversal attack — rejected."
                    )

            zf.extractall(dest)

    except zipfile.BadZipFile:
        raise FetchError("Invalid zip file.")

    # If the zip contains a single top-level directory, use that as root
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]

    return dest


def create_temp_dir() -> Path:
    """Create a temporary directory for a job's project files."""
    base = Path(tempfile.gettempdir()) / "mycode-web"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base))
