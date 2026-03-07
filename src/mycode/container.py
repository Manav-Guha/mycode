"""Docker containerisation for myCode — full isolation via Docker.

Builds and runs myCode inside a Docker container when ``--containerised``
is passed.  Two-phase build: installs project dependencies with network,
then runs tests without network.

1. Dependency install failures on projects with C extensions, system library
   requirements, or specific Python versions.
2. Security isolation for untrusted code.

Pure Python.  No LLM dependency.
"""

import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_IMAGE_PREFIX = "mycode"


class ContainerError(Exception):
    """Error related to Docker containerisation."""


# ── Docker Detection ──


def is_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running.

    Returns True only if ``docker info`` succeeds (daemon responsive).
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ── Image Management ──


def _image_tag(python_version: str) -> str:
    """Return the Docker image tag for a given Python version."""
    return f"{_IMAGE_PREFIX}:py{python_version}"


def _image_exists(tag: str) -> bool:
    """Check if a Docker image with the given tag exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _find_repo_root() -> Optional[Path]:
    """Find the myCode repo root by walking up from the package directory.

    Returns the repo root if found (contains pyproject.toml with
    ``name = "mycode-ai"``), or None if running from a pip install
    without the source tree.
    """
    pkg_dir = Path(__file__).resolve().parent  # src/mycode/
    for parent in [pkg_dir.parent.parent, pkg_dir.parent]:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            try:
                content = pyproject.read_text(encoding="utf-8")
                if 'name = "mycode-ai"' in content:
                    return parent
            except OSError:
                pass
    return None


def _generate_dockerfile(python_version: str, from_source: bool) -> str:
    """Generate a Dockerfile for building the myCode image.

    Args:
        python_version: Python version for the base image (e.g. "3.11").
        from_source: If True, copies local source into the image.
            If False, installs from PyPI.
    """
    lines = [
        f"FROM python:{python_version}-slim",
        "",
        "# Install Node.js for JavaScript project support",
        "RUN apt-get update \\",
        "    && apt-get install -y --no-install-recommends nodejs npm \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
    ]

    if from_source:
        lines.extend([
            "# Install myCode from local source",
            "COPY . /opt/mycode",
            "RUN pip install --no-cache-dir /opt/mycode",
        ])
    else:
        lines.extend([
            "# Install myCode from PyPI",
            "RUN pip install --no-cache-dir mycode-ai",
        ])

    lines.extend([
        "",
        "# Create workspace directory",
        "RUN mkdir -p /workspace",
        "",
        'ENTRYPOINT ["mycode"]',
        "",
    ])

    return "\n".join(lines)


def _docker_build(tag: str, context_path: str, dockerfile_path: str) -> None:
    """Run ``docker build`` with progress streamed to stderr."""
    cmd = [
        "docker", "build",
        "-t", tag,
        "-f", dockerfile_path,
        context_path,
    ]
    logger.info("Building Docker image: %s", tag)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        if result.returncode != 0:
            raise ContainerError(
                f"Docker build failed (exit {result.returncode}):\n"
                f"{result.stderr[:2000]}"
            )
        logger.info("Docker image built: %s", tag)
    except subprocess.TimeoutExpired:
        raise ContainerError("Docker build timed out after 10 minutes.")
    except FileNotFoundError:
        raise ContainerError(
            "Docker not found. Install Docker to use --containerised.\n"
            "See: https://docs.docker.com/get-docker/"
        )


def build_image(python_version: str = "3.11", force: bool = False) -> str:
    """Build the myCode Docker image.

    Reuses an existing image if one with the same tag exists (unless
    *force* is True).

    Args:
        python_version: Python version for the base image.
        force: Rebuild even if the image already exists.

    Returns:
        The Docker image tag (e.g. ``mycode:py3.11``).

    Raises:
        ContainerError: If the build fails or Docker is unavailable.
    """
    tag = _image_tag(python_version)

    if not force and _image_exists(tag):
        logger.info("Reusing existing Docker image: %s", tag)
        return tag

    repo_root = _find_repo_root()
    from_source = repo_root is not None

    if from_source:
        dockerfile_content = _generate_dockerfile(python_version, from_source=True)
        dockerfile_path = repo_root / "Dockerfile.mycode.tmp"
        try:
            dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
            print(f"Building Docker image ({tag}) from local source...")
            _docker_build(tag, str(repo_root), str(dockerfile_path))
        finally:
            try:
                dockerfile_path.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        with tempfile.TemporaryDirectory(prefix="mycode_docker_") as tmpdir:
            dockerfile_content = _generate_dockerfile(python_version, from_source=False)
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
            print(f"Building Docker image ({tag}) from PyPI...")
            _docker_build(tag, tmpdir, str(dockerfile_path))

    return tag


# ── Project Image (Two-Phase Build) ──


def _generate_project_dockerfile(base_tag: str) -> str:
    """Generate a Dockerfile that layers project deps onto the base image.

    Copies the project into the image and conditionally installs
    dependencies from requirements.txt, setup.py, pyproject.toml,
    or package.json.

    Args:
        base_tag: The cached myCode base image tag (e.g. ``mycode:py3.11``).

    Returns:
        Dockerfile content as a string.
    """
    lines = [
        f"FROM {base_tag}",
        "",
        "# Copy project files into image",
        "COPY . /workspace/project",
        "",
        "# Install Python dependencies (if present)",
        "RUN if [ -f /workspace/project/requirements.txt ]; then "
        "pip install --no-cache-dir -r /workspace/project/requirements.txt || true; fi",
        'RUN if [ -f /workspace/project/setup.py ]; then '
        'pip install --no-cache-dir /workspace/project || true; fi',
        'RUN if [ -f /workspace/project/pyproject.toml ] && '
        'grep -q "\\[project\\]" /workspace/project/pyproject.toml; then '
        'pip install --no-cache-dir /workspace/project || true; fi',
        "",
        "# Install JavaScript dependencies (if present)",
        "RUN if [ -f /workspace/project/package.json ]; then "
        "cd /workspace/project && npm install --ignore-scripts 2>/dev/null || true; fi",
        "",
    ]
    return "\n".join(lines)


def _build_project_image(base_tag: str, project_path: Path) -> str:
    """Build a project-specific Docker image with dependencies pre-installed.

    Uses the project directory as the build context so ``COPY .`` picks
    up all project files.  Dependencies are installed with network access
    during the build step.

    Args:
        base_tag: The cached myCode base image tag.
        project_path: Path to the user's project directory.

    Returns:
        The project image tag (e.g. ``mycode-project:a1b2c3d4``).

    Raises:
        ContainerError: If the build fails.
    """
    path_hash = hashlib.sha256(
        str(project_path.resolve()).encode()
    ).hexdigest()[:8]
    project_tag = f"mycode-project:{path_hash}"

    dockerfile_content = _generate_project_dockerfile(base_tag)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Dockerfile", delete=False, prefix="mycode_proj_",
    ) as f:
        f.write(dockerfile_content)
        dockerfile_path = f.name

    try:
        print("Installing project dependencies in container...")
        _docker_build(project_tag, str(project_path), dockerfile_path)
    finally:
        try:
            Path(dockerfile_path).unlink(missing_ok=True)
        except OSError:
            pass

    return project_tag


# ── Container Execution ──


def run_containerised(
    project_path: Path,
    cli_args: list[str],
    python_version: str = "3.11",
    constraints_file: Optional[str] = None,
) -> int:
    """Run myCode inside a Docker container.

    Two-phase approach:
      1. Build base image (cached) with myCode installed.
      2. Build project image (deps installed with network access).
      3. Run tests from project image with ``--network=none``.

    Args:
        project_path: Resolved path to the user's project directory.
        cli_args: Additional CLI arguments to pass through (e.g.
            ``["--offline", "--verbose"]``).  The project path is
            handled separately.
        python_version: Python version for the container image.
        constraints_file: Optional path to a JSON file with serialized
            conversational constraints (from host conversation).

    Returns:
        Exit code from the containerised myCode run.

    Raises:
        ContainerError: If Docker is unavailable or the build fails.
    """
    # Phase 1: Build base image (cached)
    base_tag = build_image(python_version)

    project_path = project_path.resolve()
    container_project = "/workspace/project"

    # Phase 2: Build project image (installs deps with network)
    project_tag = _build_project_image(base_tag, project_path)

    try:
        cmd = [
            "docker", "run",
            "--rm",                                        # Destroy after run
            "--network=none",                              # No network (security)
            "-e", "MYCODE_CONTAINERISED=1",                # Signal containerised mode
        ]

        # Mount constraints file if provided
        if constraints_file:
            cmd.extend([
                "-v", f"{constraints_file}:/workspace/constraints.json:ro",
            ])

        # Resource limits inside the container
        cmd.extend(["--memory=2g", "--cpus=2"])

        cmd.append(project_tag)

        # myCode args: project path (inside container) + pass-through flags
        cmd.append(container_project)
        cmd.extend(cli_args)

        # Add constraints-file arg if provided
        if constraints_file:
            cmd.extend(["--constraints-file", "/workspace/constraints.json"])

        # Skip PyPI/npm version lookups (no network in container)
        if "--skip-version-check" not in cli_args:
            cmd.append("--skip-version-check")

        # Force non-interactive if no constraints file (no conversation needed)
        if "--non-interactive" not in cli_args and not constraints_file:
            cmd.append("--non-interactive")

        print("Running myCode in Docker container...")
        logger.info("Container command: %s", " ".join(cmd))

        result = subprocess.run(cmd, timeout=1800)  # 30 minutes max
        return result.returncode
    except subprocess.TimeoutExpired:
        print(
            "Error: Containerised analysis timed out after 30 minutes.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError:
        raise ContainerError(
            "Docker not found. Install Docker to use --containerised.\n"
            "See: https://docs.docker.com/get-docker/"
        )
    finally:
        # Clean up project image
        try:
            subprocess.run(
                ["docker", "rmi", project_tag],
                capture_output=True,
                timeout=30,
            )
            logger.info("Cleaned up project image: %s", project_tag)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            logger.debug("Could not clean up project image: %s", project_tag)
