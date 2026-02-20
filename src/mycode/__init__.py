"""myCode — Stress-testing tool for AI-generated code."""

from mycode.ingester import (
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    IngestionError,
    IngestionResult,
    ProjectIngester,
)
from mycode.session import (
    DependencyInstallError,
    EnvironmentInfo,
    ResourceCaps,
    SessionError,
    SessionManager,
    SessionResult,
    VenvCreationError,
)

__all__ = [
    # Session Manager (C1)
    "SessionManager",
    "ResourceCaps",
    "EnvironmentInfo",
    "SessionResult",
    "SessionError",
    "VenvCreationError",
    "DependencyInstallError",
    # Project Ingester (C2)
    "ProjectIngester",
    "IngestionResult",
    "IngestionError",
    "FileAnalysis",
    "DependencyInfo",
    "FunctionFlow",
    "CouplingPoint",
]
