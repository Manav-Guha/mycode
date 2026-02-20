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
from mycode.js_ingester import JsProjectIngester
from mycode.library import (
    ComponentLibrary,
    DependencyProfile,
    LibraryError,
    ProfileMatch,
    ProfileNotFoundError,
)
from mycode.scenario import (
    LLMConfig,
    LLMError,
    LLMResponseError,
    ScenarioError,
    ScenarioGenerator,
    ScenarioGeneratorResult,
    StressTestScenario,
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
    # JS Project Ingester (C3)
    "JsProjectIngester",
    # Component Library (C4)
    "ComponentLibrary",
    "DependencyProfile",
    "ProfileMatch",
    "LibraryError",
    "ProfileNotFoundError",
    # Scenario Generator (D1)
    "ScenarioGenerator",
    "ScenarioGeneratorResult",
    "StressTestScenario",
    "LLMConfig",
    "ScenarioError",
    "LLMError",
    "LLMResponseError",
]
