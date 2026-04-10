"""Library Taxonomy Auto-Classifiers — deterministic classification for every finding.

Classifies findings against the myCode Library Taxonomy (v1) automatically.
No deferred classification — every finding gets classified on creation.

Six classifiers:
  1. failure_domain_classifier → Level 1 failure domain (8 domains)
  2. failure_pattern_classifier → Level 2 pattern name within the domain
  3. operational_trigger_classifier → What triggered the failure
  4. vertical_classifier → Project vertical from dependencies/structure
  5. architectural_pattern_classifier → Architectural pattern from project shape
  6. business_domain_classifier → Industry/sector from deps + project metadata
"""

import re
from typing import Optional


# ── Level 1: Failure Domains ──

_FAILURE_DOMAINS = frozenset({
    "resource_exhaustion",
    "concurrency_failure",
    "scaling_collapse",
    "input_handling_failure",
    "dependency_failure",
    "integration_failure",
    "configuration_environment_failure",
    "unclassified",
})

# ── Level 2: Failure Patterns by Domain ──

_FAILURE_PATTERNS: dict[str, list[str]] = {
    "resource_exhaustion": [
        "unbounded_cache_growth",
        "memory_accumulation_over_sessions",
        "large_payload_oom",
        "connection_pool_depletion",
        "disk_write_accumulation",
        "cpu_saturation",
    ],
    "concurrency_failure": [
        "request_deadlock",
        "shared_state_mutation",
        "race_condition",
        "thread_pool_exhaustion",
        "gil_contention",
    ],
    "scaling_collapse": [
        "linear_to_exponential_transition",
        "response_time_cliff",
        "throughput_plateau",
        "cascade_degradation",
    ],
    "input_handling_failure": [
        "unvalidated_type_crash",
        "empty_input_crash",
        "oversized_input_hang",
        "unsupported_input_format",
        "format_boundary_failure",
        "malformed_input_crash",
        "encoding_failure",
    ],
    "dependency_failure": [
        "version_incompatibility",
        "api_breaking_change",
        "missing_transitive_dependency",
        "dependency_deprecation",
        "silent_behaviour_change",
    ],
    "integration_failure": [
        "cascading_timeout",
        "state_desync",
        "serialisation_mismatch",
        "error_propagation_failure",
        "version_conflict_between_dependencies",
    ],
    "configuration_environment_failure": [
        "dependency_install_failure",
        "transitive_dependency_conflict",
        "platform_incompatibility",
        "runtime_service_unavailable",
        "python_version_incompatibility",
    ],
}

# ── Operational Triggers ──

_OPERATIONAL_TRIGGERS = frozenset({
    "sustained_load",
    "burst_traffic",
    "long_session",
    "large_input",
    "concurrent_access",
    "format_variation",
})

# ── Verticals ──

_VERTICALS = frozenset({
    "web_app",
    "data_pipeline",
    "chatbot",
    "dashboard",
    "api_service",
    "portfolio",
    "ml_model",
    "utility",
})

# ── Architectural Patterns ──

_ARCHITECTURAL_PATTERNS = frozenset({
    "web_app",
    "data_pipeline",
    "chatbot",
    "dashboard",
    "api_service",
    "portfolio",
    "ml_model",
    "utility",
})


# ══════════════════════════════════════════════════════════════════
# 1. Failure Domain Classifier
# ══════════════════════════════════════════════════════════════════

# Category → domain mapping for scenario categories
_CATEGORY_DOMAIN_MAP: dict[str, str] = {
    "memory_profiling": "resource_exhaustion",
    "data_volume_scaling": "scaling_collapse",
    "concurrent_execution": "concurrency_failure",
    "blocking_io": "concurrency_failure",
    "gil_contention": "concurrency_failure",
    "edge_case_input": "input_handling_failure",
    "async_failures": "concurrency_failure",
    "event_listener_accumulation": "resource_exhaustion",
    "state_management_degradation": "resource_exhaustion",
    "http_load_testing": "concurrency_failure",
}

# Error type → domain overrides (these take priority)
_ERROR_DOMAIN_MAP: dict[str, str] = {
    "MemoryError": "resource_exhaustion",
    "OutOfMemoryError": "resource_exhaustion",
    "TimeoutError": "scaling_collapse",
    "ConnectionError": "integration_failure",
    "ConnectionRefusedError": "integration_failure",
    "ConnectionResetError": "integration_failure",
    "ModuleNotFoundError": "configuration_environment_failure",
    "ImportError": "configuration_environment_failure",
    "FileNotFoundError": "configuration_environment_failure",
    "ConfigError": "configuration_environment_failure",
    "PermissionError": "configuration_environment_failure",
    "OSError": "configuration_environment_failure",
    "TypeError": "input_handling_failure",
    "ValueError": "input_handling_failure",
    "KeyError": "input_handling_failure",
    "IndexError": "input_handling_failure",
    "UnicodeDecodeError": "input_handling_failure",
    "UnicodeEncodeError": "input_handling_failure",
    "JSONDecodeError": "input_handling_failure",
    "AttributeError": "dependency_failure",
    "DeprecationWarning": "dependency_failure",
    "RuntimeError": "concurrency_failure",
    "DeadlockError": "concurrency_failure",
    "BrokenPipeError": "integration_failure",
}

# Scenario name keywords → domain
_NAME_DOMAIN_KEYWORDS: list[tuple[str, str]] = [
    # Multi-word phrases first (more specific, checked before single keywords)
    ("could not start", "dependency_failure"),       # 1,218 repos — server start failures
    ("missing depend", "dependency_failure"),          # 190+102 repos — missing deps
    ("degradation", "scaling_collapse"),               # 82 repos — response time degradation
    # Single keywords
    ("memory", "resource_exhaustion"),
    ("cache", "resource_exhaustion"),
    ("oom", "resource_exhaustion"),
    ("leak", "resource_exhaustion"),
    ("pool", "resource_exhaustion"),
    ("concurrent", "concurrency_failure"),
    ("deadlock", "concurrency_failure"),
    ("race", "concurrency_failure"),
    ("thread", "concurrency_failure"),
    ("gil", "concurrency_failure"),
    ("async", "concurrency_failure"),
    ("parallel", "concurrency_failure"),
    ("scaling", "scaling_collapse"),
    ("throughput", "scaling_collapse"),
    ("load", "scaling_collapse"),
    ("volume", "scaling_collapse"),
    ("growth", "scaling_collapse"),
    ("edge_case", "input_handling_failure"),
    ("input", "input_handling_failure"),
    ("validation", "input_handling_failure"),
    ("parsing", "input_handling_failure"),
    ("format", "input_handling_failure"),
    ("encoding", "input_handling_failure"),
    ("payload", "input_handling_failure"),
    ("version", "dependency_failure"),
    ("deprecat", "dependency_failure"),
    ("migration", "dependency_failure"),
    ("breaking_change", "dependency_failure"),
    ("api_key", "dependency_failure"),
    ("timeout", "integration_failure"),
    ("connection", "integration_failure"),
    ("middleware", "integration_failure"),
    ("route", "integration_failure"),
    ("install", "configuration_environment_failure"),
    ("config", "configuration_environment_failure"),
    ("env", "configuration_environment_failure"),
    ("platform", "configuration_environment_failure"),
]


def failure_domain_classifier(
    scenario_name: str,
    scenario_category: str,
    error_type: str = "",
    error_details: str = "",
) -> str:
    """Classify a finding into one of 8 failure domains.

    Priority:
      1. Error type (most specific signal)
      2. Scenario name keywords
      3. Scenario category (broadest signal)
      4. "unclassified" fallback

    Returns one of the 8 failure domain strings.
    """
    # 1. Error type override
    if error_type and error_type in _ERROR_DOMAIN_MAP:
        return _ERROR_DOMAIN_MAP[error_type]

    # Also check error_details for error type names
    if error_details:
        for etype, domain in _ERROR_DOMAIN_MAP.items():
            if etype in error_details:
                return domain

    # 2. Scenario name keywords
    name_lower = scenario_name.lower()
    for keyword, domain in _NAME_DOMAIN_KEYWORDS:
        if keyword in name_lower:
            return domain

    # 3. Scenario category
    if scenario_category and scenario_category in _CATEGORY_DOMAIN_MAP:
        return _CATEGORY_DOMAIN_MAP[scenario_category]

    return "unclassified"


# ══════════════════════════════════════════════════════════════════
# 2. Failure Pattern Classifier
# ══════════════════════════════════════════════════════════════════

# Domain + keyword → Level 2 pattern
_PATTERN_RULES: list[tuple[str, list[str], str]] = [
    # Resource Exhaustion
    ("resource_exhaustion", ["cache", "lru", "memoiz"], "unbounded_cache_growth"),
    ("resource_exhaustion", ["session", "accumul", "long_running"], "memory_accumulation_over_sessions"),
    ("resource_exhaustion", ["payload", "oom", "large"], "large_payload_oom"),
    ("resource_exhaustion", ["pool", "connection"], "connection_pool_depletion"),
    ("resource_exhaustion", ["disk", "write", "log", "file_size"], "disk_write_accumulation"),
    ("resource_exhaustion", ["cpu", "saturat", "compute"], "cpu_saturation"),
    ("resource_exhaustion", ["memory", "leak", "growth"], "memory_accumulation_over_sessions"),

    # Concurrency Failure
    ("concurrency_failure", ["deadlock"], "request_deadlock"),
    ("concurrency_failure", ["shared_state", "global", "mutation"], "shared_state_mutation"),
    ("concurrency_failure", ["race"], "race_condition"),
    ("concurrency_failure", ["thread_pool", "worker"], "thread_pool_exhaustion"),
    ("concurrency_failure", ["gil"], "gil_contention"),
    ("concurrency_failure", ["concurrent load", "degrades", "saturate"], "thread_pool_exhaustion"),

    # Scaling Collapse
    ("scaling_collapse", ["exponential", "nonlinear", "n_squared"], "linear_to_exponential_transition"),
    ("scaling_collapse", ["response_time", "latency", "cliff", "degradation"], "response_time_cliff"),
    ("scaling_collapse", ["throughput", "plateau", "bottleneck"], "throughput_plateau"),
    ("scaling_collapse", ["cascade", "chain", "domino"], "cascade_degradation"),
    ("scaling_collapse", ["scaling", "volume", "growth"], "linear_to_exponential_transition"),

    # Input Handling Failure
    ("input_handling_failure", ["type", "TypeError"], "unvalidated_type_crash"),
    ("input_handling_failure", ["empty", "null", "none", "missing"], "empty_input_crash"),
    ("input_handling_failure", ["oversiz", "large_input", "huge"], "oversized_input_hang"),
    ("input_handling_failure", ["format", "unsupported"], "unsupported_input_format"),
    ("input_handling_failure", ["boundary", "edge"], "format_boundary_failure"),
    ("input_handling_failure", ["malform", "invalid", "corrupt"], "malformed_input_crash"),
    ("input_handling_failure", ["encod", "unicode", "utf"], "encoding_failure"),
    ("input_handling_failure", ["validation", "parsing"], "format_boundary_failure"),

    # Dependency Failure
    ("dependency_failure", ["could not start", "server", "startup"], "missing_server_dependency"),
    ("dependency_failure", ["missing depend", "unresolvable"], "unresolvable_dependency"),
    ("dependency_failure", ["version", "incompatib"], "version_incompatibility"),
    ("dependency_failure", ["breaking", "api_change", "migration"], "api_breaking_change"),
    ("dependency_failure", ["transitive", "missing_dep"], "missing_transitive_dependency"),
    ("dependency_failure", ["deprecat"], "dependency_deprecation"),
    ("dependency_failure", ["silent", "behaviour", "behavior"], "silent_behaviour_change"),

    # Integration Failure
    ("integration_failure", ["timeout", "cascad"], "cascading_timeout"),
    ("integration_failure", ["desync", "state", "sync"], "state_desync"),
    ("integration_failure", ["serial", "json", "marshal"], "serialisation_mismatch"),
    ("integration_failure", ["error_propag", "unhandled"], "error_propagation_failure"),
    ("integration_failure", ["conflict", "version"], "version_conflict_between_dependencies"),

    # Configuration & Environment
    ("configuration_environment_failure", ["install", "pip", "npm"], "dependency_install_failure"),
    ("configuration_environment_failure", ["transitive", "conflict"], "transitive_dependency_conflict"),
    ("configuration_environment_failure", ["platform", "os", "arch"], "platform_incompatibility"),
    ("configuration_environment_failure", ["service", "unavailable", "connection"], "runtime_service_unavailable"),
    ("configuration_environment_failure", ["python_version", "node_version"], "python_version_incompatibility"),
]


def failure_pattern_classifier(
    failure_domain: str,
    scenario_name: str,
    error_signature: str = "",
) -> Optional[str]:
    """Classify a finding into a Level 2 failure pattern.

    Returns the pattern name (e.g., 'unbounded_cache_growth') or None
    if no pattern matches (stays unclassified at Level 2).
    """
    if failure_domain == "unclassified":
        return None

    combined = f"{scenario_name} {error_signature}".lower()

    for domain, keywords, pattern in _PATTERN_RULES:
        if domain != failure_domain:
            continue
        if any(kw in combined for kw in keywords):
            return pattern

    return None


# ══════════════════════════════════════════════════════════════════
# 3. Operational Trigger Classifier
# ══════════════════════════════════════════════════════════════════

_CATEGORY_TRIGGER_MAP: dict[str, str] = {
    "data_volume_scaling": "large_input",
    "memory_profiling": "sustained_load",
    "concurrent_execution": "concurrent_access",
    "blocking_io": "concurrent_access",
    "gil_contention": "concurrent_access",
    "edge_case_input": "format_variation",
    "async_failures": "concurrent_access",
    "event_listener_accumulation": "long_session",
    "state_management_degradation": "long_session",
}

# Scenario name keywords → trigger overrides
_NAME_TRIGGER_KEYWORDS: list[tuple[str, str]] = [
    ("burst", "burst_traffic"),
    ("spike", "burst_traffic"),
    ("sustained", "sustained_load"),
    ("long_running", "long_session"),
    ("session", "long_session"),
    ("conversation_turn", "long_session"),
    ("concurrent", "concurrent_access"),
    ("parallel", "concurrent_access"),
    ("large_payload", "large_input"),
    ("large_response", "large_input"),
    ("volume", "large_input"),
    ("format", "format_variation"),
    ("edge_case", "format_variation"),
    ("encoding", "format_variation"),
    ("validation", "format_variation"),
]


def operational_trigger_classifier(
    scenario_category: str,
    scenario_name: str = "",
) -> str:
    """Classify the operational trigger for a finding.

    Returns one of: sustained_load, burst_traffic, long_session,
    large_input, concurrent_access, format_variation.
    """
    # Name-based override first (more specific)
    if scenario_name:
        name_lower = scenario_name.lower()
        for keyword, trigger in _NAME_TRIGGER_KEYWORDS:
            if keyword in name_lower:
                return trigger

    # Category-based mapping
    if scenario_category in _CATEGORY_TRIGGER_MAP:
        return _CATEGORY_TRIGGER_MAP[scenario_category]

    return "sustained_load"  # safe default


# ══════════════════════════════════════════════════════════════════
# 4. Vertical Classifier
# ══════════════════════════════════════════════════════════════════

# Dependency sets that indicate verticals
_VERTICAL_DEPS: dict[str, list[str]] = {
    "web_app": [
        "flask", "django", "fastapi", "express", "nextjs", "next",
        "react", "vue", "angular", "svelte", "sveltekit",
        "starlette", "tornado", "aiohttp", "koa", "hono",
    ],
    "data_pipeline": [
        "pandas", "numpy", "scipy", "polars", "dask", "pyspark",
        "airflow", "prefect", "dagster", "luigi", "celery",
        "spark", "flink",
    ],
    "chatbot": [
        "langchain", "llamaindex", "llama-index", "llama_index",
        "openai", "anthropic", "chromadb", "pinecone",
        "transformers", "huggingface", "groq", "cohere",
        "langserve", "langgraph",
    ],
    "dashboard": [
        "streamlit", "gradio", "dash", "plotly", "bokeh",
        "panel", "voila", "shiny",
    ],
    "api_service": [
        "fastapi", "flask-restful", "djangorestframework",
        "graphql", "grpc", "swagger", "connexion",
    ],
    "ml_model": [
        "scikit-learn", "sklearn", "tensorflow", "torch",
        "pytorch", "keras", "xgboost", "lightgbm",
        "catboost", "mlflow", "wandb", "optuna",
        "joblib", "pickle",
    ],
    "portfolio": [
        "tailwindcss", "framer-motion", "three",
        "gsap", "lottie",
    ],
}

# File patterns that indicate verticals
_VERTICAL_FILES: dict[str, list[str]] = {
    "web_app": ["routes", "views", "templates", "static", "middleware", "pages"],
    "data_pipeline": ["pipeline", "etl", "transform", "ingest", "dag"],
    "chatbot": ["chat", "agent", "prompt", "chain", "retriev"],
    "dashboard": ["dashboard", "app.py", "streamlit"],
    "api_service": ["api", "endpoint", "router", "schema"],
    "ml_model": ["model", "train", "predict", "feature", "dataset"],
    "portfolio": ["portfolio", "landing", "hero", "about"],
}


def _normalize_dep(name: str) -> str:
    """Normalize a dependency name for matching."""
    return name.lower().replace("-", "").replace("_", "")


def vertical_classifier(
    dependencies: list[str],
    file_structure: list[str] | None = None,
    framework: str = "",
) -> str:
    """Classify the project vertical from dependencies and file structure.

    Returns one of: web_app, data_pipeline, chatbot, dashboard,
    api_service, portfolio, ml_model, utility.
    """
    deps_norm = {_normalize_dep(d) for d in dependencies}
    dep_scores: dict[str, int] = {}

    for vertical, dep_list in _VERTICAL_DEPS.items():
        score = 0
        for dep in dep_list:
            dep_norm = _normalize_dep(dep)
            if dep_norm in deps_norm:
                score += 1
        if score > 0:
            dep_scores[vertical] = score

    # File structure scoring
    file_scores: dict[str, int] = {}
    if file_structure:
        files_lower = [f.lower() for f in file_structure]
        for vertical, patterns in _VERTICAL_FILES.items():
            score = 0
            for pattern in patterns:
                for f in files_lower:
                    if pattern in f:
                        score += 1
                        break
            if score > 0:
                file_scores[vertical] = score

    # Framework override
    if framework:
        fw_norm = _normalize_dep(framework)
        for vertical, dep_list in _VERTICAL_DEPS.items():
            for dep in dep_list:
                if _normalize_dep(dep) == fw_norm:
                    dep_scores[vertical] = dep_scores.get(vertical, 0) + 5
                    break

    # Combine scores (deps weighted 2x over files)
    combined: dict[str, int] = {}
    for v in _VERTICALS:
        combined[v] = dep_scores.get(v, 0) * 2 + file_scores.get(v, 0)

    # Priority resolution for more specific verticals:
    # dashboard > web_app, chatbot > web_app, api_service > web_app,
    # ml_model > data_pipeline
    _SPECIFIC_OVER_GENERAL = [
        ("dashboard", "web_app"),
        ("dashboard", "data_pipeline"),
        ("chatbot", "web_app"),
        ("api_service", "web_app"),
        ("ml_model", "data_pipeline"),
    ]
    for specific, general in _SPECIFIC_OVER_GENERAL:
        if combined.get(specific, 0) > 0 and combined.get(general, 0) > 0:
            if combined[specific] >= combined[general]:
                return specific

    best = max(combined, key=lambda v: combined[v])
    if combined[best] > 0:
        return best

    return "utility"


# ══════════════════════════════════════════════════════════════════
# 5. Architectural Pattern Classifier
# ══════════════════════════════════════════════════════════════════

_ARCH_FRAMEWORK_MAP: dict[str, str] = {
    "flask": "web_app",
    "django": "web_app",
    "fastapi": "api_service",
    "express": "web_app",
    "nextjs": "web_app",
    "next": "web_app",
    "react": "web_app",
    "vue": "web_app",
    "angular": "web_app",
    "streamlit": "dashboard",
    "gradio": "dashboard",
    "dash": "dashboard",
    "langchain": "chatbot",
    "llamaindex": "chatbot",
    "llama-index": "chatbot",
    "pandas": "data_pipeline",
    "airflow": "data_pipeline",
    "tensorflow": "ml_model",
    "torch": "ml_model",
    "pytorch": "ml_model",
    "sklearn": "ml_model",
    "scikit-learn": "ml_model",
}


def architectural_pattern_classifier(
    dependencies: list[str],
    framework: str = "",
    file_count: int = 0,
    has_frontend: bool = False,
    has_backend: bool = False,
) -> str:
    """Classify the architectural pattern of a project.

    Returns one of: web_app, data_pipeline, chatbot, dashboard,
    api_service, portfolio, ml_model, utility.
    """
    # Framework is the strongest signal
    if framework:
        fw_norm = _normalize_dep(framework)
        for key, pattern in _ARCH_FRAMEWORK_MAP.items():
            if _normalize_dep(key) == fw_norm:
                return pattern

    # Dependency-based scoring (exact match only)
    dep_scores: dict[str, int] = {}
    deps_norm = {_normalize_dep(d) for d in dependencies}

    for key, pattern in _ARCH_FRAMEWORK_MAP.items():
        key_norm = _normalize_dep(key)
        if key_norm in deps_norm:
            dep_scores[pattern] = dep_scores.get(pattern, 0) + 1

    if dep_scores:
        best = max(dep_scores, key=lambda p: dep_scores[p])
        return best

    # Structural heuristics
    if has_frontend and has_backend:
        return "web_app"
    if has_frontend and not has_backend:
        if file_count <= 10:
            return "portfolio"
        return "web_app"
    if has_backend and not has_frontend:
        return "api_service"

    return "utility"


# ══════════════════════════════════════════════════════════════════
# 6. Business Domain Classifier
# ══════════════════════════════════════════════════════════════════

_BUSINESS_DOMAINS = frozenset({
    "fintech",
    "healthcare",
    "education",
    "e_commerce",
    "real_estate",
    "climate",
    "entertainment",
    "social_media",
    "developer_tools",
    "data_science",
    "ai_assistant",
    "general",
})

# Dependency sets that indicate business domains
_BUSINESS_DOMAIN_DEPS: dict[str, list[str]] = {
    "fintech": [
        "yfinance", "alpaca-trade-api", "plaid", "polygon-api-client",
        "ccxt", "ta-lib", "quantlib", "robin-stocks", "finnhub",
        "pandas-ta", "backtrader", "zipline",
    ],
    "healthcare": [
        "pydicom", "hl7", "fhir", "fhir-resources", "fhirclient",
        "mne", "nibabel", "biopython", "medpy", "dicom",
        "nilearn", "lifelines", "scanpy",
    ],
    "education": [
        "edx", "moodle", "canvas", "nbgrader", "jupyterhub",
        "openedx", "edx-platform",
    ],
    "e_commerce": [
        "shopify", "woocommerce", "saleor", "medusa", "snipcart",
        "commercejs", "magento", "prestashop",
    ],
    "real_estate": [
        "zillow", "mls", "realtor", "rets",
    ],
    "climate": [
        "xarray", "netcdf4", "cftime", "cartopy", "iris",
        "climlab", "metpy", "cfgrib",
    ],
    "entertainment": [
        "pygame", "pyglet", "arcade", "ursina", "panda3d",
        "moviepy", "pyaudio", "librosa", "music21",
    ],
    "social_media": [
        "tweepy", "praw", "instaloader", "facebook-sdk", "mastodon",
        "python-twitter", "twython", "discord",
    ],
    "developer_tools": [
        "pytest", "black", "ruff", "mypy", "flake8",
        "eslint", "prettier", "webpack", "vite", "babel",
        "pylint", "isort", "bandit", "coverage",
    ],
    "data_science": [
        "scikit-learn", "sklearn", "tensorflow", "torch", "pytorch",
        "keras", "xgboost", "lightgbm", "catboost",
        "huggingface", "transformers", "datasets",
    ],
    "ai_assistant": [
        "openai", "anthropic", "langchain", "crewai", "autogen",
        "llamaindex", "llama-index", "llama_index",
        "langgraph", "langserve", "semantic-kernel",
    ],
}

# Keywords in project name/description that indicate business domains
_BUSINESS_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "fintech": [
        "finance", "trading", "stock", "crypto", "banking",
        "payment", "invoice", "ledger", "wallet", "forex",
    ],
    "healthcare": [
        "health", "medical", "patient", "clinical", "diagnosis",
        "hospital", "pharma", "drug", "ehr", "dicom",
    ],
    "education": [
        "learn", "course", "quiz", "student", "tutor",
        "classroom", "lms", "lesson", "curriculum",
    ],
    "e_commerce": [
        "shop", "store", "cart", "checkout", "product",
        "catalog", "inventory", "order", "ecommerce",
    ],
    "real_estate": [
        "property", "listing", "tenant", "mortgage", "rental",
        "realty", "housing", "landlord",
    ],
    "climate": [
        "weather", "climate", "carbon", "emission",
        "sustainability", "environmental",
    ],
    "entertainment": [
        "game", "music", "video", "stream", "media",
        "movie", "audio", "player",
    ],
    "social_media": [
        "social", "feed", "post", "follow", "chat",
        "message", "community", "forum",
    ],
    "developer_tools": [
        "cli", "lint", "debug", "deploy", "ci",
        "devtool", "sdk", "compiler", "formatter",
    ],
    "data_science": [
        "dataset", "prediction", "regression", "classification",
        "neural", "training", "inference",
    ],
    "ai_assistant": [
        "assistant", "chatbot", "agent", "copilot", "llm",
        "prompt", "rag", "retrieval",
    ],
}

# Stripe disambiguation: co-occurs with finance deps → fintech,
# co-occurs with commerce deps → e_commerce, alone → fintech.
_STRIPE_FINTECH_COOCCUR = frozenset(
    _normalize_dep(d) for d in _BUSINESS_DOMAIN_DEPS["fintech"]
)
_STRIPE_COMMERCE_COOCCUR = frozenset(
    _normalize_dep(d) for d in _BUSINESS_DOMAIN_DEPS["e_commerce"]
)


def business_domain_classifier(
    dependencies: list[str],
    project_name: str = "",
    project_description: str = "",
) -> str:
    """Classify the business domain from dependencies and project metadata.

    Returns one of the _BUSINESS_DOMAINS strings, defaulting to "general".
    """
    deps_norm = {_normalize_dep(d) for d in dependencies}
    dep_scores: dict[str, int] = {}

    # Handle Stripe disambiguation
    has_stripe = "stripe" in deps_norm
    if has_stripe:
        if deps_norm & _STRIPE_COMMERCE_COOCCUR:
            dep_scores["e_commerce"] = dep_scores.get("e_commerce", 0) + 1
        else:
            # Co-occurs with finance deps OR alone → fintech
            dep_scores["fintech"] = dep_scores.get("fintech", 0) + 1

    for domain, dep_list in _BUSINESS_DOMAIN_DEPS.items():
        score = 0
        for dep in dep_list:
            if _normalize_dep(dep) in deps_norm:
                score += 1
        if score > 0:
            dep_scores[domain] = dep_scores.get(domain, 0) + score

    # Keyword scoring from project name and description
    keyword_scores: dict[str, int] = {}
    text = f"{project_name} {project_description}".lower()
    if text.strip():
        for domain, keywords in _BUSINESS_DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                keyword_scores[domain] = score

    # Combine: deps weighted 2x over keywords
    combined: dict[str, int] = {}
    all_domains = set(dep_scores) | set(keyword_scores)
    for d in all_domains:
        combined[d] = dep_scores.get(d, 0) * 2 + keyword_scores.get(d, 0)

    if not combined:
        return "general"

    best = max(combined, key=lambda d: combined[d])
    if combined[best] > 0:
        return best

    return "general"


# ══════════════════════════════════════════════════════════════════
# Convenience: Classify a complete finding
# ══════════════════════════════════════════════════════════════════

def classify_finding(
    scenario_name: str,
    scenario_category: str,
    error_type: str = "",
    error_details: str = "",
    severity: str = "",
) -> dict:
    """Run all finding-level classifiers and return a classification dict.

    Returns:
        {
            "failure_domain": str,
            "failure_pattern": str | None,
            "operational_trigger": str,
        }
    """
    domain = failure_domain_classifier(
        scenario_name, scenario_category, error_type, error_details,
    )
    pattern = failure_pattern_classifier(
        domain, scenario_name, error_details,
    )
    trigger = operational_trigger_classifier(
        scenario_category, scenario_name,
    )
    return {
        "failure_domain": domain,
        "failure_pattern": pattern,
        "operational_trigger": trigger,
    }


def classify_project(
    dependencies: list[str],
    file_structure: list[str] | None = None,
    framework: str = "",
    file_count: int = 0,
    has_frontend: bool = False,
    has_backend: bool = False,
    project_name: str = "",
    project_description: str = "",
) -> dict:
    """Run all project-level classifiers and return a classification dict.

    Returns:
        {
            "vertical": str,
            "architectural_pattern": str,
            "business_domain": str,
        }
    """
    vert = vertical_classifier(dependencies, file_structure, framework)
    arch = architectural_pattern_classifier(
        dependencies, framework, file_count, has_frontend, has_backend,
    )
    domain = business_domain_classifier(
        dependencies, project_name, project_description,
    )
    return {
        "vertical": vert,
        "architectural_pattern": arch,
        "business_domain": domain,
    }
