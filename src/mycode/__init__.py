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
from mycode.engine import (
    EngineError,
    ExecutionEngine,
    ExecutionEngineResult,
    ScenarioResult,
    StepResult,
)
from mycode.interface import (
    ConversationalInterface,
    InterfaceError,
    InterfaceResult,
    OperationalIntent,
    ScenarioReview,
    TerminalIO,
    UserIO,
    summarize_ingestion,
)
from mycode.report import (
    DegradationPoint,
    DiagnosticReport,
    Finding,
    ReportError,
    ReportGenerator,
)
from mycode.recorder import (
    ConsentError,
    InteractionRecorder,
    RecorderError,
    SessionRecord,
)
from mycode.pipeline import (
    LanguageDetectionError,
    PipelineConfig,
    PipelineError,
    PipelineResult,
    StageResult,
    check_llm_report_allowance,
    decrement_llm_report_counter,
    detect_language,
    run_pipeline,
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
from mycode.inference import (
    CorpusIndex,
    InferenceEngine,
    InferenceResult,
    RiskPrediction,
)
from mycode.classifiers import (
    architectural_pattern_classifier,
    business_domain_classifier,
    classify_finding,
    classify_project,
    failure_domain_classifier,
    failure_pattern_classifier,
    operational_trigger_classifier,
    vertical_classifier,
)
from mycode.container import (
    ContainerError,
    build_image,
    is_docker_available,
    run_containerised,
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
    # Execution Engine (D2)
    "ExecutionEngine",
    "ExecutionEngineResult",
    "ScenarioResult",
    "StepResult",
    "EngineError",
    # Pipeline (D3)
    "run_pipeline",
    "detect_language",
    "PipelineConfig",
    "PipelineResult",
    "StageResult",
    "PipelineError",
    "LanguageDetectionError",
    # Conversational Interface (E1)
    "ConversationalInterface",
    "InterfaceResult",
    "OperationalIntent",
    "ScenarioReview",
    "InterfaceError",
    "TerminalIO",
    "UserIO",
    "summarize_ingestion",
    # Report Generator (E2)
    "ReportGenerator",
    "DiagnosticReport",
    "Finding",
    "DegradationPoint",
    "ReportError",
    # Interaction Recorder (E3)
    "InteractionRecorder",
    "SessionRecord",
    "RecorderError",
    "ConsentError",
    # Docker Container (C5)
    "ContainerError",
    "build_image",
    "is_docker_available",
    "run_containerised",
    # Inference Engine (C7)
    "InferenceEngine",
    "InferenceResult",
    "RiskPrediction",
    "CorpusIndex",
    # Classifiers (C6)
    "classify_finding",
    "classify_project",
    "failure_domain_classifier",
    "failure_pattern_classifier",
    "operational_trigger_classifier",
    "vertical_classifier",
    "architectural_pattern_classifier",
    "business_domain_classifier",
]
