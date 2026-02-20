"""Component Library (C4) — Pre-built profiles for common dependencies.

Loads version-aware dependency profiles containing scaling characteristics,
memory behavior, known failure modes, edge case sensitivities, interaction
patterns, and stress test templates.

Pure Python. No LLM dependency.
"""

from mycode.library.loader import (
    ComponentLibrary,
    DependencyProfile,
    LibraryError,
    ProfileMatch,
    ProfileNotFoundError,
)

__all__ = [
    "ComponentLibrary",
    "DependencyProfile",
    "LibraryError",
    "ProfileMatch",
    "ProfileNotFoundError",
]
