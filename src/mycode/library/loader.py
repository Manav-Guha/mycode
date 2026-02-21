"""Component Library loader — reads and matches dependency profiles.

Loads JSON profiles from the profiles/ directory, matches them against a
project's dependencies (from IngestionResult), and flags unrecognized
dependencies. Profiles are version-aware.

Pure Python. No LLM dependency.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──

# Default profiles root is profiles/ at the repository top level
_DEFAULT_PROFILES_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "profiles"

SUPPORTED_LANGUAGES = {"python", "javascript"}


# ── Exceptions ──


class LibraryError(Exception):
    """Base exception for component library errors."""


class ProfileNotFoundError(LibraryError):
    """Raised when a requested profile does not exist."""


# ── Data Classes ──


@dataclass
class DependencyProfile:
    """A loaded dependency profile with all sections.

    Attributes:
        identity: Name, version, category, and description.
        scaling_characteristics: How the dependency scales under load.
        memory_behavior: Baseline footprint, growth patterns, known leaks.
        known_failure_modes: Common ways this dependency fails.
        edge_case_sensitivities: Inputs and conditions that cause unexpected behavior.
        interaction_patterns: Common co-dependencies and known conflicts.
        stress_test_templates: Pre-built test configurations for this dependency.
        raw: The full parsed JSON dict for downstream consumers.
    """

    identity: dict
    scaling_characteristics: dict
    memory_behavior: dict
    known_failure_modes: list
    edge_case_sensitivities: list
    interaction_patterns: dict
    stress_test_templates: list
    raw: dict = field(repr=False)
    node_stress_test_templates: list = field(default_factory=list)

    @property
    def name(self) -> str:
        """Canonical lowercase name (e.g. 'flask', 'numpy')."""
        return self.identity["name"]

    @property
    def browser_only(self) -> bool:
        """Whether this dependency requires a browser environment (DOM/canvas/WebGL)."""
        return self.identity.get("browser_only", False)

    @property
    def pypi_name(self) -> Optional[str]:
        """PyPI package name, or None for stdlib modules."""
        return self.identity.get("pypi_name")

    @property
    def npm_name(self) -> Optional[str]:
        """npm package name, or None for non-JS modules."""
        return self.identity.get("npm_name")

    @property
    def category(self) -> str:
        """Profile category (e.g. 'web_framework', 'database')."""
        return self.identity["category"]

    @property
    def current_stable_version(self) -> str:
        """Current stable version tracked by this profile."""
        return self.identity["current_stable_version"]


@dataclass
class ProfileMatch:
    """Result of matching a project dependency against the component library.

    Attributes:
        dependency_name: The dependency name from the project.
        profile: The matched profile, or None if unrecognized.
        installed_version: The version installed in the user's project (if known).
        version_match: Whether installed version matches the profile's current stable.
        version_notes: Relevant notes for the installed version, if any.
    """

    dependency_name: str
    profile: Optional[DependencyProfile]
    installed_version: Optional[str] = None
    version_match: Optional[bool] = None
    version_notes: Optional[str] = None


# ── Component Library ──


class ComponentLibrary:
    """Loads and queries dependency profiles from the profiles/ directory.

    Usage::

        library = ComponentLibrary()
        flask_profile = library.get_profile("python", "flask")
        matches = library.match_dependencies("python", dependencies)

    Args:
        profiles_root: Path to the profiles/ directory. Defaults to the
            repository-level profiles/ directory.
    """

    def __init__(self, profiles_root: Optional[Path] = None) -> None:
        self._profiles_root = Path(profiles_root) if profiles_root else _DEFAULT_PROFILES_ROOT
        self._cache: dict[tuple[str, str], DependencyProfile] = {}

    @property
    def profiles_root(self) -> Path:
        """Root directory containing language-specific profile subdirectories."""
        return self._profiles_root

    # ── Loading ──

    def get_profile(self, language: str, name: str) -> DependencyProfile:
        """Load a single dependency profile by language and name.

        Args:
            language: 'python' or 'javascript'.
            name: Dependency name (lowercase), e.g. 'flask', 'react'.

        Returns:
            The loaded DependencyProfile.

        Raises:
            ProfileNotFoundError: If the profile file does not exist.
            LibraryError: If the profile file is invalid JSON or missing required fields.
        """
        cache_key = (language.lower(), name.lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        profile = self._load_profile(language.lower(), name.lower())
        self._cache[cache_key] = profile
        return profile

    def list_profiles(self, language: str) -> list[str]:
        """List all available profile names for a language.

        Args:
            language: 'python' or 'javascript'.

        Returns:
            Sorted list of profile names (without file extension).
        """
        lang_dir = self._profiles_root / language.lower()
        if not lang_dir.is_dir():
            return []
        return sorted(p.stem for p in lang_dir.glob("*.json"))

    def load_all(self, language: str) -> dict[str, DependencyProfile]:
        """Load all profiles for a language.

        Args:
            language: 'python' or 'javascript'.

        Returns:
            Dict mapping profile name to DependencyProfile.
        """
        profiles: dict[str, DependencyProfile] = {}
        for name in self.list_profiles(language):
            try:
                profiles[name] = self.get_profile(language, name)
            except LibraryError as exc:
                logger.warning("Skipping invalid profile %s/%s: %s", language, name, exc)
        return profiles

    # ── Matching ──

    def match_dependencies(
        self,
        language: str,
        dependencies: list[dict],
    ) -> list[ProfileMatch]:
        """Match project dependencies against the component library.

        Each dependency dict should have at minimum a ``name`` key. Optionally
        ``installed_version`` for version comparison.

        Compatible with ``DependencyInfo`` objects from the Project Ingester —
        pass ``[vars(d) for d in ingestion_result.dependencies]`` or any list
        of dicts with a ``name`` field.

        Args:
            language: 'python' or 'javascript'.
            dependencies: List of dependency dicts with at least a 'name' key.

        Returns:
            List of ProfileMatch objects, one per dependency. Unrecognized
            dependencies have profile=None.
        """
        available = set(self.list_profiles(language))
        matches: list[ProfileMatch] = []

        for dep in dependencies:
            dep_name = dep.get("name", "").lower()
            installed = dep.get("installed_version")

            if not dep_name:
                continue

            # Normalize common package name variants
            lookup_name = _normalize_dep_name(dep_name, language.lower())

            if lookup_name in available:
                try:
                    profile = self.get_profile(language, lookup_name)
                    version_match, version_note = _check_version(profile, installed)
                    matches.append(ProfileMatch(
                        dependency_name=dep_name,
                        profile=profile,
                        installed_version=installed,
                        version_match=version_match,
                        version_notes=version_note,
                    ))
                except LibraryError:
                    # Profile file exists but is invalid — treat as unrecognized
                    logger.warning("Profile %s/%s exists but failed to load", language, lookup_name)
                    matches.append(ProfileMatch(dependency_name=dep_name, profile=None, installed_version=installed))
            else:
                matches.append(ProfileMatch(
                    dependency_name=dep_name,
                    profile=None,
                    installed_version=installed,
                ))

        return matches

    def get_unrecognized(self, matches: list[ProfileMatch]) -> list[str]:
        """Extract dependency names that have no matching profile.

        Args:
            matches: Output from match_dependencies().

        Returns:
            List of unrecognized dependency names.
        """
        return [m.dependency_name for m in matches if m.profile is None]

    def get_recognized(self, matches: list[ProfileMatch]) -> list[ProfileMatch]:
        """Extract matches that have a loaded profile.

        Args:
            matches: Output from match_dependencies().

        Returns:
            List of ProfileMatch objects with non-None profiles.
        """
        return [m for m in matches if m.profile is not None]

    # ── Internal ──

    def _load_profile(self, language: str, name: str) -> DependencyProfile:
        """Load and validate a single profile from disk."""
        profile_path = self._profiles_root / language / f"{name}.json"

        if not profile_path.is_file():
            raise ProfileNotFoundError(
                f"No profile found for {language}/{name} "
                f"(expected {profile_path})"
            )

        try:
            raw = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LibraryError(f"Invalid JSON in {profile_path}: {exc}") from exc

        # Validate required top-level keys
        required_keys = {
            "identity",
            "scaling_characteristics",
            "memory_behavior",
            "known_failure_modes",
            "edge_case_sensitivities",
            "interaction_patterns",
            "stress_test_templates",
        }
        missing = required_keys - set(raw.keys())
        if missing:
            raise LibraryError(
                f"Profile {language}/{name} missing required fields: {', '.join(sorted(missing))}"
            )

        # Validate identity sub-keys
        identity = raw["identity"]
        for key in ("name", "category", "current_stable_version"):
            if key not in identity:
                raise LibraryError(
                    f"Profile {language}/{name} identity missing required field: {key}"
                )

        return DependencyProfile(
            identity=identity,
            scaling_characteristics=raw["scaling_characteristics"],
            memory_behavior=raw["memory_behavior"],
            known_failure_modes=raw["known_failure_modes"],
            edge_case_sensitivities=raw["edge_case_sensitivities"],
            interaction_patterns=raw["interaction_patterns"],
            stress_test_templates=raw["stress_test_templates"],
            node_stress_test_templates=raw.get("node_stress_test_templates", []),
            raw=raw,
        )


# ── Helpers ──


def _normalize_dep_name(name: str, language: str = "python") -> str:
    """Normalize a dependency name for profile lookup.

    Handles common variations:
    - PyPI/npm names use hyphens, profile names use underscores or bare names
    - Case insensitive
    - Language-specific aliases (Python vs JavaScript)
    - Scoped npm packages (@org/pkg)
    """
    normalized = name.lower().strip()

    # Handle scoped npm packages before general normalization
    # e.g. "@anthropic-ai/sdk" → "anthropic_node"
    scoped_aliases = {
        "@anthropic-ai/sdk": "anthropic_node",
        "@supabase/supabase-js": "supabase_js",
        "@prisma/client": "prisma",
        "@langchain/core": "langchainjs",
        "@langchain/community": "langchainjs",
        "@langchain/openai": "langchainjs",
    }
    if normalized in scoped_aliases:
        return scoped_aliases[normalized]

    normalized = normalized.replace("-", "_").replace(".", "_")

    # Language-specific alias maps
    _PYTHON_ALIASES = {
        "flask": "flask",
        "fastapi": "fastapi",
        "streamlit": "streamlit",
        "gradio": "gradio",
        "pandas": "pandas",
        "numpy": "numpy",
        "sqlite3": "sqlite3",
        "sqlalchemy": "sqlalchemy",
        "supabase": "supabase",
        "supabase_py": "supabase",
        "langchain": "langchain",
        "langchain_core": "langchain",
        "langchain_community": "langchain",
        "llama_index": "llamaindex",
        "llama_index_core": "llamaindex",
        "llamaindex": "llamaindex",
        "chromadb": "chromadb",
        "openai": "openai",
        "anthropic": "anthropic",
        "pydantic": "pydantic",
        "pydantic_core": "pydantic",
        "os": "os_pathlib",
        "pathlib": "os_pathlib",
        "requests": "requests",
        "httpx": "httpx",
    }

    _JAVASCRIPT_ALIASES = {
        "react": "react",
        "react_dom": "react",
        "next": "nextjs",
        "express": "express",
        "fs": "node_core",
        "path": "node_core",
        "http": "node_core",
        "https": "node_core",
        "url": "node_core",
        "crypto": "node_core",
        "child_process": "node_core",
        "os": "node_core",
        "util": "node_core",
        "tailwindcss": "tailwindcss",
        "three": "threejs",
        "svelte": "svelte",
        "openai": "openai_node",
        "anthropic": "anthropic_node",
        "langchain": "langchainjs",
        "langchain_core": "langchainjs",
        "langchain_community": "langchainjs",
        "supabase_js": "supabase_js",
        "prisma": "prisma",
        "axios": "axios",
        "node_fetch": "axios",
        "mongoose": "mongoose",
        "mongodb": "mongoose",
        "stripe": "stripe",
        "dotenv": "dotenv",
        "zod": "zod",
        "socket_io": "socketio",
        "socket_io_client": "socketio",
        "socketio": "socketio",
        "chart_js": "chartjs",
        "chartjs": "chartjs",
        "plotly_js": "plotlyjs",
        "plotly_js_dist_min": "plotlyjs",
        "plotly_js_basic_dist_min": "plotlyjs",
        "plotly_js_dist": "plotlyjs",
        "plotlyjs": "plotlyjs",
        "react_chartjs_2": "react_chartjs_2",
        "react_plotly_js": "react_plotlyjs",
        "react_plotlyjs": "react_plotlyjs",
        "google_auth_library": "google_auth_library",
    }

    aliases = _JAVASCRIPT_ALIASES if language == "javascript" else _PYTHON_ALIASES
    return aliases.get(normalized, normalized)


def _check_version(
    profile: DependencyProfile,
    installed_version: Optional[str],
) -> tuple[Optional[bool], Optional[str]]:
    """Check installed version against profile's tracked version.

    Returns:
        (version_match, version_notes) tuple.
        version_match is None if installed_version is unknown.
    """
    if not installed_version:
        return None, None

    stable = profile.current_stable_version
    if stable == "stdlib":
        # Stdlib modules — version tied to Python, not independently versioned
        return None, "Standard library module; version tied to Python interpreter"

    # Simple major.minor comparison
    installed_parts = installed_version.split(".")
    stable_parts = stable.split(".")

    try:
        installed_major = int(installed_parts[0])
        stable_major = int(stable_parts[0])
    except (ValueError, IndexError):
        return None, None

    # Check version notes for the installed version
    version_notes = profile.identity.get("version_notes", {})
    note = version_notes.get(installed_version)

    # Check major version match
    if installed_major < stable_major:
        msg = f"Major version behind: installed {installed_version}, current stable {stable}"
        if note:
            msg = f"{msg}. {note}"
        return False, msg

    # Check minor version if major matches
    if installed_major == stable_major:
        try:
            installed_minor = int(installed_parts[1]) if len(installed_parts) > 1 else 0
            stable_minor = int(stable_parts[1]) if len(stable_parts) > 1 else 0
            if installed_minor < stable_minor:
                msg = f"Minor version behind: installed {installed_version}, current stable {stable}"
                if note:
                    msg = f"{msg}. {note}"
                return False, msg
        except ValueError:
            pass

    # Version matches or is ahead
    if note:
        return True, note
    return True, None
