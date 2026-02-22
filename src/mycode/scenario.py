"""Scenario Generator (D1) — LLM-powered stress test scenario generation.

Core LLM layer. Takes full ingester output + component library matches +
operational intent. Tests dependency interaction chains as systems, not
individual components in isolation.

Generates stress test configurations across categories:
  Shared: data_volume_scaling, memory_profiling, edge_case_input, concurrent_execution
  Python: blocking_io, gil_contention
  JavaScript: async_failures, event_listener_accumulation, state_management_degradation

Supports Gemini Flash (free tier default), BYOK for any OpenAI-compatible
endpoint, and offline mode for testing without API calls.

LLM-dependent component (one of three).
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from mycode.ingester import CouplingPoint, FunctionFlow, IngestionResult
from mycode.library.loader import DependencyProfile, ProfileMatch

logger = logging.getLogger(__name__)

# ── Constants ──

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.0-flash"

SHARED_CATEGORIES = frozenset({
    "data_volume_scaling",
    "memory_profiling",
    "edge_case_input",
    "concurrent_execution",
})

PYTHON_CATEGORIES = SHARED_CATEGORIES | frozenset({
    "blocking_io",
    "gil_contention",
})

JAVASCRIPT_CATEGORIES = SHARED_CATEGORIES | frozenset({
    "async_failures",
    "event_listener_accumulation",
    "state_management_degradation",
})

ALL_CATEGORIES = PYTHON_CATEGORIES | JAVASCRIPT_CATEGORIES


class CouplingBehaviorType(str, Enum):
    """Classification of coupling point functions by behavior."""

    STATE_SETTER = "state_setter"
    API_CALLER = "api_caller"
    PURE_COMPUTATION = "pure_computation"
    DOM_RENDER = "dom_render"
    ERROR_HANDLER = "error_handler"


# ── Exceptions ──


class ScenarioError(Exception):
    """Base exception for scenario generator errors."""


class LLMError(ScenarioError):
    """Failed to communicate with LLM backend."""


class LLMResponseError(ScenarioError):
    """LLM returned an unparseable or invalid response."""


# ── Data Classes ──


@dataclass
class LLMConfig:
    """Configuration for the LLM backend.

    Attributes:
        api_key: API key for the provider. Required unless offline.
        base_url: Base URL for the OpenAI-compatible API endpoint.
            Defaults to Gemini Flash endpoint.
        model: Model identifier. Defaults to gemini-2.0-flash.
        max_tokens: Maximum output tokens for the LLM response.
        temperature: Sampling temperature. Lower = more deterministic.
        timeout_seconds: HTTP request timeout.
        max_retries: Number of retry attempts on transient errors.
    """

    api_key: Optional[str] = None
    base_url: str = GEMINI_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout_seconds: float = 60.0
    max_retries: int = 2


@dataclass
class LLMResponse:
    """Raw response from the LLM backend."""

    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StressTestScenario:
    """A single stress test scenario for the Execution Engine.

    Attributes:
        name: Descriptive test name (e.g. "concurrent_db_queries").
        category: Stress category (e.g. "concurrent_execution").
        description: What this test does and why.
        target_dependencies: Which dependencies this scenario stresses.
        test_config: Parameters for the Execution Engine — target files,
            synthetic data spec, measurement config, resource limits.
        expected_behavior: What should happen under this stress.
        failure_indicators: Signals that indicate a problem.
        priority: high / medium / low.
        source: "llm" or "offline" — how this scenario was generated.
    """

    name: str
    category: str
    description: str
    target_dependencies: list[str] = field(default_factory=list)
    test_config: dict = field(default_factory=dict)
    expected_behavior: str = ""
    failure_indicators: list[str] = field(default_factory=list)
    priority: str = "medium"
    source: str = "offline"


@dataclass
class ScenarioGeneratorResult:
    """Complete output from the Scenario Generator.

    Attributes:
        scenarios: Generated stress test scenarios.
        reasoning: LLM's reasoning about what to test and why (empty in offline mode).
        warnings: Non-fatal issues encountered during generation.
        model_used: Which model produced the scenarios (or "offline").
        token_usage: Input/output token counts (zero in offline mode).
    """

    scenarios: list[StressTestScenario] = field(default_factory=list)
    reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    model_used: str = "offline"
    token_usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


# ── LLM Backend ──


class LLMBackend:
    """Sends requests to an OpenAI-compatible chat completions endpoint.

    Supports Gemini Flash (default), OpenAI, DeepSeek, or any OpenAI-compatible
    provider via base_url configuration. Uses urllib (no external dependencies).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        if not config.api_key:
            raise LLMError(
                "API key required. Set api_key in LLMConfig, or use offline mode."
            )

    def generate(self, messages: list[dict]) -> LLMResponse:
        """Send a chat completion request and return the response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            LLMResponse with the assistant's reply.

        Raises:
            LLMError: On network, auth, or provider errors after retries.
        """
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1 + self._config.max_retries):
            if attempt > 0:
                wait = 2 ** (attempt - 1)
                logger.info("Retry %d/%d after %ds", attempt, self._config.max_retries, wait)
                time.sleep(wait)

            try:
                return self._send_request(url, body, headers)
            except LLMError as exc:
                last_error = exc
                if not self._is_retryable(exc):
                    raise

        raise last_error  # type: ignore[misc]

    def _send_request(self, url: str, body: bytes, headers: dict) -> LLMResponse:
        """Execute a single HTTP request."""
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            raise LLMError(
                f"LLM API returned HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise LLMError(f"LLM API request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM API returned invalid JSON: {exc}") from exc

        # Extract response
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected LLM response structure: {exc}") from exc

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", self._config.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    @staticmethod
    def _is_retryable(exc: LLMError) -> bool:
        """Determine if an error is transient and worth retrying."""
        msg = str(exc)
        if "HTTP 429" in msg or "HTTP 5" in msg:
            return True
        if "request failed" in msg.lower():
            return True
        return False


# ── Coupling Point Classification ──


_ERROR_HANDLER_NAMES = frozenset({
    "error", "handle_error", "handleerror", "catch", "fallback", "exception",
    "on_error", "onerror", "error_handler", "errorhandler",
})

_ERROR_HANDLER_DECORATORS = frozenset({
    "errorhandler", "exception_handler",
})

_DOM_MODULES = frozenset({"react", "vue", "svelte", "preact"})
_DOM_CALLS = frozenset({
    "render", "createelement", "usestate", "useeffect",
    "useref", "usememo", "usecallback",
})

_HTTP_MODULES = frozenset({
    "requests", "httpx", "axios", "fetch", "openai", "anthropic",
    "urllib", "aiohttp", "supabase", "googleapis",
})
_HTTP_CALLS = frozenset({
    "get", "post", "put", "delete", "fetch", "request", "send",
    "query", "execute",
})

_STATE_CALLS = frozenset({"setstate", "dispatch", "commit"})


def _file_path_to_module(rel_path: str, language: str) -> str:
    """Convert a relative file path to a module-style identifier.

    Python: strip .py/__init__.py, join with '.'
    JS: strip .js/index.js, join with '/'
    """
    if language.lower() == "javascript":
        path = rel_path
        if path.endswith("/index.js"):
            path = path[: -len("/index.js")]
        elif path.endswith(".js"):
            path = path[: -len(".js")]
        elif path.endswith(".ts"):
            path = path[: -len(".ts")]
        elif path.endswith(".tsx"):
            path = path[: -len(".tsx")]
        elif path.endswith(".jsx"):
            path = path[: -len(".jsx")]
        return path.replace("\\", "/")
    else:
        path = rel_path
        if path.endswith("/__init__.py"):
            path = path[: -len("/__init__.py")]
        elif path.endswith(".py"):
            path = path[: -len(".py")]
        return path.replace("/", ".").replace("\\", ".")


def classify_coupling_point(
    cp: CouplingPoint,
    file_analyses: list,
    language: str,
) -> CouplingBehaviorType:
    """Classify a coupling point's behavior type.

    Resolves cp.source to its FunctionInfo and FileAnalysis, then applies
    classification rules in priority order (first match wins):
    1. Error handler (name or decorator)
    2. DOM/render (React/Vue/Svelte imports + render calls)
    3. API/network caller (HTTP module imports + HTTP calls)
    4. State setter (globals_accessed, shared_state type, or setState calls)
    5. Pure computation (fallback)
    """
    # Build module->FileAnalysis lookup
    module_to_analysis: dict[str, "FileAnalysis"] = {}
    for fa in file_analyses:
        mod = _file_path_to_module(fa.file_path, language)
        module_to_analysis[mod] = fa
        # Also index by file_path directly
        module_to_analysis[fa.file_path] = fa

    # Resolve source to FunctionInfo
    func_info = None
    file_analysis = None
    source = cp.source  # e.g. "app.create_app" or "models.User.save"

    # Try matching source prefix to a module
    parts = source.rsplit(".", 1)
    if len(parts) == 2:
        module_prefix, func_name = parts
    else:
        module_prefix, func_name = "", source

    # Look for the FileAnalysis
    fa = module_to_analysis.get(module_prefix)
    if fa is None:
        # Try further splitting for class methods: "models.User.save" -> "models"
        prefix_parts = module_prefix.rsplit(".", 1)
        if len(prefix_parts) == 2:
            fa = module_to_analysis.get(prefix_parts[0])

    if fa is not None:
        file_analysis = fa
        # Find the FunctionInfo
        for f in fa.functions:
            if f.name == func_name:
                func_info = f
                break
        if func_info is None:
            # Check class methods
            for cls in fa.classes:
                for f_list_item in fa.functions:
                    if f_list_item.name == func_name and f_list_item.is_method:
                        func_info = f_list_item
                        break

    # Apply classification rules in priority order
    func_name_lower = func_name.lower()

    # 0. React/Vue useState setter pattern: setXxx where Xxx is capitalized.
    #    Unambiguous — always a state setter. Checked first to prevent
    #    "setError" from matching the error-handler "error" pattern.
    if re.match(r"set[A-Z]", func_name):
        if file_analysis:
            file_imports_lower = {
                imp.module.lower().split(".")[0] for imp in file_analysis.imports
            }
            if file_imports_lower & _DOM_MODULES:
                return CouplingBehaviorType.STATE_SETTER
        if language.lower() == "javascript":
            return CouplingBehaviorType.STATE_SETTER

    # 1. Error handler — by name
    for pattern in _ERROR_HANDLER_NAMES:
        if pattern in func_name_lower:
            return CouplingBehaviorType.ERROR_HANDLER

    # 1b. Error handler — by decorator
    if func_info and func_info.decorators:
        for dec in func_info.decorators:
            dec_lower = dec.lower()
            for pattern in _ERROR_HANDLER_DECORATORS:
                if pattern in dec_lower:
                    return CouplingBehaviorType.ERROR_HANDLER

    # 2. DOM/render — file imports react/vue/svelte AND calls contain render funcs
    if file_analysis:
        file_imports_lower = {
            imp.module.lower().split(".")[0] for imp in file_analysis.imports
        }
        if file_imports_lower & _DOM_MODULES:
            if func_info and func_info.calls:
                calls_lower = {c.lower() for c in func_info.calls}
                if calls_lower & _DOM_CALLS:
                    return CouplingBehaviorType.DOM_RENDER

    # 3. API/network caller
    if file_analysis:
        file_imports_lower = {
            imp.module.lower().split(".")[0] for imp in file_analysis.imports
        }
        if file_imports_lower & _HTTP_MODULES:
            if func_info and func_info.calls:
                calls_lower = {c.lower() for c in func_info.calls}
                if calls_lower & _HTTP_CALLS:
                    return CouplingBehaviorType.API_CALLER
        # Also: async + HTTP-like call names
        if func_info and func_info.is_async and func_info.calls:
            calls_lower = {c.lower() for c in func_info.calls}
            if calls_lower & _HTTP_CALLS:
                return CouplingBehaviorType.API_CALLER

    # 4. State setter
    if func_info and func_info.globals_accessed:
        return CouplingBehaviorType.STATE_SETTER
    if cp.coupling_type == "shared_state":
        return CouplingBehaviorType.STATE_SETTER
    if func_info and func_info.calls:
        calls_lower = {c.lower() for c in func_info.calls}
        if calls_lower & _STATE_CALLS:
            return CouplingBehaviorType.STATE_SETTER

    # 5. Pure computation (fallback)
    return CouplingBehaviorType.PURE_COMPUTATION


def group_coupling_points_by_behavior(
    coupling_points: list[CouplingPoint],
    file_analyses: list,
    language: str,
) -> dict[CouplingBehaviorType, list[CouplingPoint]]:
    """Group coupling points by their classified behavior type."""
    groups: dict[CouplingBehaviorType, list[CouplingPoint]] = {}
    for cp in coupling_points:
        behavior = classify_coupling_point(cp, file_analyses, language)
        groups.setdefault(behavior, []).append(cp)
    return groups


# ── Scenario Generator ──


class ScenarioGenerator:
    """Generates stress test scenarios from project analysis and component profiles.

    Usage::

        generator = ScenarioGenerator(llm_config=LLMConfig(api_key="..."))
        result = generator.generate(ingestion_result, profile_matches, intent)

    For testing without API calls::

        generator = ScenarioGenerator(offline=True)
        result = generator.generate(ingestion_result, profile_matches, intent)

    Args:
        llm_config: LLM backend configuration. Ignored when offline=True.
        offline: If True, generate scenarios from component library templates
            only — no LLM calls. Useful for testing and fallback.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        offline: bool = False,
    ) -> None:
        self._llm_config = llm_config or LLMConfig()
        self._offline = offline
        self._backend: Optional[LLMBackend] = None
        if not offline:
            self._backend = LLMBackend(self._llm_config)

    def generate(
        self,
        ingestion_result: IngestionResult,
        profile_matches: list[ProfileMatch],
        operational_intent: str,
        language: str = "python",
    ) -> ScenarioGeneratorResult:
        """Generate stress test scenarios for the analyzed project.

        Args:
            ingestion_result: Full output from the Project Ingester (C2).
            profile_matches: Dependency matches from the Component Library (C4).
            operational_intent: User's description of what the project does,
                who it's for, and under what conditions it operates.
            language: Target language ('python' or 'javascript').

        Returns:
            ScenarioGeneratorResult with generated scenarios and metadata.
        """
        valid_categories = self._get_categories(language)
        recognized = [m for m in profile_matches if m.profile is not None]
        unrecognized = [m.dependency_name for m in profile_matches if m.profile is None]

        if self._offline:
            return self._generate_offline(
                ingestion_result, recognized, unrecognized,
                operational_intent, language, valid_categories,
            )

        return self._generate_with_llm(
            ingestion_result, recognized, unrecognized,
            operational_intent, language, valid_categories,
        )

    # ── LLM Generation ──

    def _generate_with_llm(
        self,
        ingestion: IngestionResult,
        recognized: list[ProfileMatch],
        unrecognized: list[str],
        intent: str,
        language: str,
        valid_categories: frozenset[str],
    ) -> ScenarioGeneratorResult:
        """Generate scenarios using the LLM backend."""
        messages = self._build_prompt(
            ingestion, recognized, unrecognized, intent, language, valid_categories,
        )

        warnings: list[str] = []
        try:
            assert self._backend is not None
            response = self._backend.generate(messages)
        except LLMError as exc:
            logger.warning("LLM call failed, falling back to offline: %s", exc)
            warnings.append(f"LLM unavailable ({exc}); used offline generation")
            result = self._generate_offline(
                ingestion, recognized, unrecognized,
                intent, language, valid_categories,
            )
            result.warnings = warnings + result.warnings
            return result

        # Parse LLM response
        try:
            scenarios, reasoning = self._parse_llm_response(
                response.content, valid_categories,
            )
        except LLMResponseError as exc:
            logger.warning("LLM response unparseable, falling back to offline: %s", exc)
            warnings.append(f"LLM response invalid ({exc}); used offline generation")
            result = self._generate_offline(
                ingestion, recognized, unrecognized,
                intent, language, valid_categories,
            )
            result.warnings = warnings + result.warnings
            return result

        for s in scenarios:
            s.source = "llm"

        return ScenarioGeneratorResult(
            scenarios=scenarios,
            reasoning=reasoning,
            warnings=warnings,
            model_used=response.model,
            token_usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

    # ── Prompt Construction ──

    def _build_prompt(
        self,
        ingestion: IngestionResult,
        recognized: list[ProfileMatch],
        unrecognized: list[str],
        intent: str,
        language: str,
        valid_categories: frozenset[str],
    ) -> list[dict]:
        """Construct the system and user messages for the LLM."""
        system = self._build_system_prompt(valid_categories)
        user = self._build_user_prompt(
            ingestion, recognized, unrecognized, intent, language,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_system_prompt(self, valid_categories: frozenset[str]) -> str:
        """Build the system prompt establishing the LLM's role and output format."""
        categories_str = ", ".join(sorted(valid_categories))
        return f"""\
You are the Scenario Generator for myCode, a stress-testing tool for AI-generated code.

Your task: analyze a project's structure, dependencies, and the user's operational intent, then generate targeted stress test scenarios.

CRITICAL RULES:
1. Test dependency INTERACTION CHAINS as systems — not individual components in isolation.
2. Focus on where components interact and where one failure cascades into another.
3. Prioritize scenarios that match the user's operational context.
4. Generate scenarios the user can understand in plain language.

OUTPUT FORMAT: Respond with a single JSON object:
{{
  "reasoning": "Your analysis of the project architecture, key risks, and testing strategy.",
  "scenarios": [
    {{
      "name": "descriptive_test_name",
      "category": "one of: {categories_str}",
      "description": "What this test does and why it matters for this project.",
      "target_dependencies": ["dep1", "dep2"],
      "test_config": {{
        "target_files": ["file.py"],
        "parameters": {{}},
        "synthetic_data": {{}},
        "measurements": ["memory_mb", "execution_time_ms", "error_count"],
        "resource_limits": {{"memory_mb": 512, "timeout_seconds": 60}}
      }},
      "expected_behavior": "What should happen under this stress.",
      "failure_indicators": ["indicator1", "indicator2"],
      "priority": "high|medium|low"
    }}
  ]
}}

Generate 5-15 scenarios covering different categories. Prioritize high-impact scenarios."""

    def _build_user_prompt(
        self,
        ingestion: IngestionResult,
        recognized: list[ProfileMatch],
        unrecognized: list[str],
        intent: str,
        language: str,
    ) -> str:
        """Build the user prompt with full project context."""
        sections = []

        # 1. Project overview
        sections.append(f"## Project Overview\n- Language: {language}")
        sections.append(f"- Files analyzed: {ingestion.files_analyzed}")
        sections.append(f"- Total lines: {ingestion.total_lines}")
        if ingestion.files_failed > 0:
            sections.append(f"- Files with parse errors: {ingestion.files_failed}")

        # 2. User's operational intent
        sections.append(f"\n## Operational Intent\n{intent}")

        # 3. Project structure (files, classes, key functions)
        sections.append("\n## Project Structure")
        sections.append(_serialize_file_analyses(ingestion.file_analyses))

        # 4. Dependencies
        sections.append("\n## Dependencies")
        if recognized:
            sections.append("### Recognized (have profiles):")
            for match in recognized:
                assert match.profile is not None
                version_str = f" v{match.installed_version}" if match.installed_version else ""
                status = ""
                if match.version_match is False:
                    status = f" [OUTDATED: {match.version_notes}]"
                sections.append(f"- {match.dependency_name}{version_str}{status}")
        if unrecognized:
            sections.append("### Unrecognized (no profile — generic testing):")
            for name in unrecognized:
                sections.append(f"- {name}")

        # 5. Component library profiles for recognized deps
        if recognized:
            sections.append("\n## Dependency Profiles")
            sections.append(_serialize_profiles(recognized))

        # 6. Function flows and coupling points
        if ingestion.function_flows:
            sections.append("\n## Function Call Graph (sampled)")
            sections.append(_serialize_function_flows(ingestion.function_flows))

        if ingestion.coupling_points:
            sections.append("\n## Coupling Points (failure cascade risks)")
            sections.append(_serialize_coupling_points(
                ingestion.coupling_points, ingestion.file_analyses, language,
            ))

        return "\n".join(sections)

    # ── Response Parsing ──

    def _parse_llm_response(
        self,
        content: str,
        valid_categories: frozenset[str],
    ) -> tuple[list[StressTestScenario], str]:
        """Parse the LLM's JSON response into validated scenarios.

        Returns:
            (scenarios, reasoning) tuple.

        Raises:
            LLMResponseError: If the response cannot be parsed as valid JSON.
        """
        data = _extract_json(content)
        reasoning = data.get("reasoning", "")

        scenarios: list[StressTestScenario] = []
        raw_scenarios = data.get("scenarios", [])
        if not isinstance(raw_scenarios, list):
            raise LLMResponseError("'scenarios' field is not a list")

        for i, raw in enumerate(raw_scenarios):
            if not isinstance(raw, dict):
                logger.warning("Scenario %d is not a dict, skipping", i)
                continue
            name = raw.get("name", "")
            category = raw.get("category", "")
            description = raw.get("description", "")
            if not name or not category or not description:
                logger.warning("Scenario %d missing required fields, skipping", i)
                continue
            if category not in valid_categories:
                logger.warning(
                    "Scenario '%s' has invalid category '%s', skipping", name, category,
                )
                continue

            scenarios.append(StressTestScenario(
                name=name,
                category=category,
                description=description,
                target_dependencies=raw.get("target_dependencies", []),
                test_config=raw.get("test_config", {}),
                expected_behavior=raw.get("expected_behavior", ""),
                failure_indicators=raw.get("failure_indicators", []),
                priority=raw.get("priority", "medium"),
                source="llm",
            ))

        if not scenarios:
            raise LLMResponseError("LLM produced zero valid scenarios")

        return scenarios, reasoning

    # ── Offline Generation ──

    def _generate_offline(
        self,
        ingestion: IngestionResult,
        recognized: list[ProfileMatch],
        unrecognized: list[str],
        intent: str,
        language: str,
        valid_categories: frozenset[str],
    ) -> ScenarioGeneratorResult:
        """Generate scenarios from component library templates without LLM.

        Uses stress_test_templates and known_failure_modes from matched profiles,
        coupling points from the ingester, and generic scenarios for coverage.
        """
        scenarios: list[StressTestScenario] = []
        warnings: list[str] = []

        # Collect browser-only deps — skip library-specific tests for these
        browser_only_deps: list[str] = []
        for match in recognized:
            if match.profile is not None and match.profile.browser_only:
                browser_only_deps.append(match.dependency_name)

        # Collect server-framework deps — use synthetic standalone bodies
        server_framework_deps: list[str] = []
        for match in recognized:
            if match.profile is not None and match.profile.server_framework:
                server_framework_deps.append(match.dependency_name)

        # Deduplication: when multiple packages share the same component library
        # profile (e.g. langchain, langchain-core, langchain-community all map to
        # "langchain"), generate one set of scenarios covering the family, not one
        # per package.  We track which profile names have been seen, and collect
        # all related dependency names into the first scenario's target_dependencies.
        seen_profiles: set[str] = set()
        # Map profile_name → list of dep names, for building target_dependencies
        profile_dep_names: dict[str, list[str]] = {}
        for match in recognized:
            profile = match.profile
            assert profile is not None
            pname = profile.name
            if pname not in profile_dep_names:
                profile_dep_names[pname] = []
            profile_dep_names[pname].append(match.dependency_name)

        # 1. Profile-based scenarios from stress_test_templates
        for match in recognized:
            profile = match.profile
            assert profile is not None

            # Browser-only deps: skip library-specific stress tests entirely.
            # Rendering stress tests require a browser environment.
            if profile.browser_only:
                continue

            # Deduplicate: only generate scenarios for the first match per profile
            if profile.name in seen_profiles:
                continue
            seen_profiles.add(profile.name)

            all_dep_names = profile_dep_names[profile.name]

            for template in profile.stress_test_templates:
                cat = template.get("category", "")
                if cat not in valid_categories:
                    continue
                raw_params = template.get("parameters", {})
                tc: dict = {
                    "parameters": _normalize_template_params(raw_params, cat),
                    "measurements": _infer_measurements(cat),
                    "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                }
                # Server frameworks: route to standalone body to avoid blocking
                if profile.server_framework:
                    tc["behavior"] = f"{profile.name}_server_stress"
                scenarios.append(StressTestScenario(
                    name=f"{profile.name}_{template['name']}",
                    category=cat,
                    description=template.get("description", ""),
                    target_dependencies=all_dep_names,
                    test_config=tc,
                    expected_behavior=template.get("expected_behavior", ""),
                    failure_indicators=template.get("failure_indicators", []),
                    priority=_infer_priority_from_template(template),
                    source="offline",
                ))

        # 2. Failure-mode scenarios for critical/high severity issues
        seen_fm_profiles: set[str] = set()
        for match in recognized:
            profile = match.profile
            assert profile is not None

            # Browser-only deps: skip failure mode tests too (they import browser libs)
            if profile.browser_only:
                continue

            # Server frameworks: skip failure mode tests (they describe patterns
            # best detected by static analysis, and the edge_case_input body has
            # nothing to call without importing user code)
            if profile.server_framework:
                continue

            # Deduplicate: only generate failure-mode scenarios once per profile
            if profile.name in seen_fm_profiles:
                continue
            seen_fm_profiles.add(profile.name)

            all_dep_names = profile_dep_names[profile.name]

            for mode in profile.known_failure_modes:
                if mode.get("severity") not in ("critical", "high"):
                    continue
                fm_config: dict = {
                    "detection_hint": mode.get("detection_hint", ""),
                    "severity": mode.get("severity", ""),
                    "versions_affected": mode.get("versions_affected", ""),
                    "measurements": ["error_count", "error_type"],
                    "resource_limits": {"memory_mb": 512, "timeout_seconds": 30},
                }
                scenarios.append(StressTestScenario(
                    name=f"{profile.name}_{mode['name']}_check",
                    category="edge_case_input",
                    description=(
                        f"Test for known failure mode: {mode.get('description', '')}. "
                        f"Trigger: {mode.get('trigger_conditions', 'N/A')}"
                    ),
                    target_dependencies=all_dep_names,
                    test_config=fm_config,
                    expected_behavior="Known failure mode should be detected or mitigated.",
                    failure_indicators=[mode["name"]],
                    priority="high" if mode.get("severity") == "critical" else "medium",
                    source="offline",
                ))

        # 3. Coupling point scenarios — classified by behavior
        if ingestion.coupling_points:
            groups = group_coupling_points_by_behavior(
                ingestion.coupling_points, ingestion.file_analyses, language,
            )
            group_counter = 0

            for behavior, cps in groups.items():
                if behavior == CouplingBehaviorType.STATE_SETTER:
                    # Group ALL state setters into ONE scenario
                    cat = "concurrent_execution"
                    if cat not in valid_categories:
                        continue
                    group_counter += 1
                    all_sources = [cp.source for cp in cps]
                    all_targets = []
                    for cp in cps:
                        all_targets.extend(cp.targets)
                    scenarios.append(StressTestScenario(
                        name=f"coupling_state_setters_group_{group_counter}",
                        category=cat,
                        description=(
                            f"Concurrent access test for {len(cps)} state-setting "
                            f"functions: {', '.join(all_sources[:5])}"
                            + (f" (+{len(all_sources) - 5} more)" if len(all_sources) > 5 else "")
                        ),
                        target_dependencies=[],
                        test_config={
                            "coupling_sources": all_sources,
                            "coupling_targets": list(set(all_targets)),
                            "coupling_type": "shared_state",
                            "behavior": "state_setter",
                            "measurements": _infer_measurements(cat),
                            "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                        },
                        expected_behavior=(
                            f"Concurrent access to {len(cps)} state setters should "
                            f"not cause race conditions or data corruption."
                        ),
                        failure_indicators=[
                            "race_condition", "data_corruption", "cascade_failure",
                        ],
                        priority="medium",
                        source="offline",
                    ))

                elif behavior == CouplingBehaviorType.API_CALLER:
                    # One scenario per API caller — latency/timeout focus
                    cat = "concurrent_execution"
                    if cat not in valid_categories:
                        continue
                    for cp in cps:
                        scenarios.append(StressTestScenario(
                            name=f"coupling_api_{_safe_name(cp.source)}",
                            category=cat,
                            description=(
                                f"Latency and timeout stress for API caller "
                                f"'{cp.source}': {cp.description}"
                            ),
                            target_dependencies=[],
                            test_config={
                                "coupling_source": cp.source,
                                "coupling_targets": cp.targets,
                                "coupling_type": cp.coupling_type,
                                "behavior": "api_caller",
                                "measurements": _infer_measurements(cat),
                                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                            },
                            expected_behavior=(
                                f"API caller '{cp.source}' should handle latency "
                                f"and timeouts without cascading failures to "
                                f"{len(cp.targets)} dependents."
                            ),
                            failure_indicators=[
                                "timeout", "cascade_failure", "connection_error",
                            ],
                            priority="medium",
                            source="offline",
                        ))

                elif behavior == CouplingBehaviorType.PURE_COMPUTATION:
                    # One scenario per function — data volume scaling
                    cat = "data_volume_scaling"
                    if cat not in valid_categories:
                        continue
                    for cp in cps:
                        scenarios.append(StressTestScenario(
                            name=f"coupling_compute_{_safe_name(cp.source)}",
                            category=cat,
                            description=(
                                f"Scaling stress for computation '{cp.source}': "
                                f"{cp.description}"
                            ),
                            target_dependencies=[],
                            test_config={
                                "coupling_source": cp.source,
                                "coupling_targets": cp.targets,
                                "coupling_type": cp.coupling_type,
                                "behavior": "pure_computation",
                                "measurements": _infer_measurements(cat),
                                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                            },
                            expected_behavior=(
                                f"Computation '{cp.source}' should scale without "
                                f"cascading failures to {len(cp.targets)} dependents."
                            ),
                            failure_indicators=[
                                "cascade_failure", "error_propagation", "timeout",
                            ],
                            priority="medium",
                            source="offline",
                        ))

                elif behavior == CouplingBehaviorType.DOM_RENDER:
                    # One scenario per function — state degradation (JS) or memory (Python)
                    if language.lower() == "javascript":
                        cat = "state_management_degradation"
                    else:
                        cat = "memory_profiling"
                    if cat not in valid_categories:
                        continue
                    for cp in cps:
                        dom_config: dict = {
                            "coupling_source": cp.source,
                            "coupling_targets": cp.targets,
                            "coupling_type": cp.coupling_type,
                            "behavior": "dom_render",
                            "measurements": _infer_measurements(cat),
                            "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                        }
                        if language.lower() == "javascript":
                            dom_config["skip_imports"] = True
                        scenarios.append(StressTestScenario(
                            name=f"coupling_render_{_safe_name(cp.source)}",
                            category=cat,
                            description=(
                                f"Render/UI stress for '{cp.source}': "
                                f"{cp.description}"
                            ),
                            target_dependencies=[],
                            test_config=dom_config,
                            expected_behavior=(
                                f"Render function '{cp.source}' should not degrade "
                                f"under repeated updates."
                            ),
                            failure_indicators=[
                                "memory_growth_unbounded", "render_stall", "cascade_failure",
                            ],
                            priority="medium",
                            source="offline",
                        ))

                elif behavior == CouplingBehaviorType.ERROR_HANDLER:
                    # One scenario per function — error flood focus
                    cat = "edge_case_input"
                    if cat not in valid_categories:
                        continue
                    for cp in cps:
                        scenarios.append(StressTestScenario(
                            name=f"coupling_errorhandler_{_safe_name(cp.source)}",
                            category=cat,
                            description=(
                                f"Error flood stress for handler '{cp.source}': "
                                f"{cp.description}"
                            ),
                            target_dependencies=[],
                            test_config={
                                "coupling_source": cp.source,
                                "coupling_targets": cp.targets,
                                "coupling_type": cp.coupling_type,
                                "behavior": "error_handler",
                                "measurements": _infer_measurements(cat),
                                "resource_limits": {"memory_mb": 512, "timeout_seconds": 60},
                            },
                            expected_behavior=(
                                f"Error handler '{cp.source}' should remain stable "
                                f"under a flood of errors."
                            ),
                            failure_indicators=[
                                "handler_crash", "cascade_failure", "resource_exhaustion",
                            ],
                            priority="medium",
                            source="offline",
                        ))

        # 4. Version discrepancy scenarios
        for match in recognized:
            if match.version_match is False and match.version_notes:
                scenarios.append(StressTestScenario(
                    name=f"{match.dependency_name}_version_discrepancy",
                    category="edge_case_input",
                    description=(
                        f"Dependency {match.dependency_name} is outdated: "
                        f"{match.version_notes}. Test for version-specific issues."
                    ),
                    target_dependencies=[match.dependency_name],
                    test_config={
                        "installed_version": match.installed_version,
                        "current_stable": (
                            match.profile.current_stable_version if match.profile else ""
                        ),
                        "measurements": ["error_count", "deprecation_warnings"],
                        "resource_limits": {"memory_mb": 512, "timeout_seconds": 30},
                    },
                    expected_behavior="Outdated version may exhibit known regressions.",
                    failure_indicators=["deprecation_warning", "version_specific_bug"],
                    priority="medium",
                    source="offline",
                ))

        # 5. Unrecognized dependency generic scenarios
        if unrecognized:
            scenarios.append(StressTestScenario(
                name="unrecognized_deps_generic_stress",
                category="data_volume_scaling",
                description=(
                    f"Generic stress test for unrecognized dependencies: "
                    f"{', '.join(unrecognized[:10])}. No profile available — "
                    f"test with progressively larger inputs."
                ),
                target_dependencies=unrecognized[:10],
                test_config={
                    "parameters": {
                        "data_sizes": [100, 1000, 10000],
                        "iterations": 50,
                    },
                    "measurements": ["memory_mb", "execution_time_ms", "error_count"],
                    "resource_limits": {"memory_mb": 512, "timeout_seconds": 120},
                },
                expected_behavior="Performance should degrade gracefully with data size.",
                failure_indicators=["crash", "memory_growth_unbounded", "timeout"],
                priority="low",
                source="offline",
            ))

        # Add warning for browser-only deps that were skipped
        if browser_only_deps:
            dep_list = ", ".join(sorted(set(browser_only_deps)))
            warnings.append(
                f"{dep_list} are browser-rendered libraries. "
                f"Rendering stress tests require a browser environment "
                f"(planned for future release). "
                f"Data flow and coupling tests were performed."
            )

        # Add warning for server-framework deps that use synthetic workloads
        if server_framework_deps:
            dep_list = ", ".join(sorted(set(server_framework_deps)))
            warnings.append(
                f"Server framework deps ({dep_list}): library-specific tests "
                f"use synthetic workloads instead of importing user code to "
                f"avoid blocking on server startup."
            )

        # Add warning for deduplicated dependency families
        deduped_families = {
            pname: deps for pname, deps in profile_dep_names.items()
            if len(deps) > 1
        }
        if deduped_families:
            parts = [
                f"{pname} ({', '.join(deps)})"
                for pname, deps in sorted(deduped_families.items())
            ]
            warnings.append(
                f"Related packages sharing a profile were deduplicated: "
                f"{'; '.join(parts)}. One set of scenarios covers each family."
            )

        return ScenarioGeneratorResult(
            scenarios=scenarios,
            reasoning="",
            warnings=warnings,
            model_used="offline",
        )

    # ── Helpers ──

    @staticmethod
    def _get_categories(language: str) -> frozenset[str]:
        """Return valid stress test categories for the given language."""
        if language.lower() == "javascript":
            return JAVASCRIPT_CATEGORIES
        return PYTHON_CATEGORIES


# ── Module-Level Helpers ──


def _extract_json(content: str) -> dict:
    """Extract a JSON object from LLM response content.

    Handles raw JSON, markdown code blocks, and leading/trailing noise.

    Raises:
        LLMResponseError: If no valid JSON object can be extracted.
    """
    content = content.strip()

    # Try direct parse
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Try finding the outermost { ... }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(content[first_brace : last_brace + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise LLMResponseError(
        f"Could not extract JSON from LLM response (length={len(content)})"
    )


def _serialize_file_analyses(file_analyses: list) -> str:
    """Summarize file analyses into a compact string for the prompt."""
    lines = []
    for analysis in file_analyses:
        if analysis.parse_error:
            lines.append(f"  {analysis.file_path}: [PARSE ERROR] {analysis.parse_error}")
            continue
        classes = [c.name for c in analysis.classes]
        funcs = [
            f.name for f in analysis.functions
            if not f.is_method
        ]
        loc = analysis.lines_of_code
        parts = [f"{analysis.file_path} ({loc} lines)"]
        if classes:
            parts.append(f"classes: {', '.join(classes)}")
        if funcs:
            parts.append(f"functions: {', '.join(funcs[:10])}")
            if len(funcs) > 10:
                parts.append(f"... +{len(funcs) - 10} more")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def _serialize_profiles(recognized: list[ProfileMatch]) -> str:
    """Serialize matched profiles' key info for the prompt."""
    lines = []
    for match in recognized:
        profile = match.profile
        assert profile is not None
        lines.append(f"\n### {profile.name} ({profile.category})")

        if profile.known_failure_modes:
            lines.append("Known failure modes:")
            for mode in profile.known_failure_modes:
                sev = mode.get("severity", "?")
                lines.append(
                    f"  - [{sev}] {mode.get('name', '?')}: "
                    f"{mode.get('description', '')}"
                )

        if profile.edge_case_sensitivities:
            lines.append("Edge case sensitivities:")
            for edge in profile.edge_case_sensitivities:
                lines.append(f"  - {edge.get('name', '?')}: {edge.get('description', '')}")

        if profile.stress_test_templates:
            template_names = [t.get("name", "?") for t in profile.stress_test_templates]
            lines.append(f"Stress templates: {', '.join(template_names)}")

        interactions = profile.interaction_patterns
        if interactions.get("known_conflicts"):
            lines.append("Known conflicts:")
            for conflict in interactions["known_conflicts"]:
                lines.append(
                    f"  - {conflict.get('dependency', '?')}: "
                    f"{conflict.get('description', '')}"
                )

    return "\n".join(lines)


def _serialize_function_flows(flows: list[FunctionFlow], limit: int = 30) -> str:
    """Serialize function flows for the prompt, sampled to limit."""
    lines = []
    shown = flows[:limit]
    for flow in shown:
        lines.append(f"  {flow.caller} -> {flow.callee}  [{flow.file_path}:{flow.lineno}]")
    if len(flows) > limit:
        lines.append(f"  ... +{len(flows) - limit} more flows")
    return "\n".join(lines)


def _serialize_coupling_points(
    coupling_points: list[CouplingPoint],
    file_analyses: Optional[list] = None,
    language: Optional[str] = None,
) -> str:
    """Serialize coupling points for the prompt.

    When file_analyses and language are provided, includes a [behavior: type]
    annotation for LLM-path prompts.
    """
    lines = []
    for cp in coupling_points:
        annotation = ""
        if file_analyses is not None and language is not None:
            behavior = classify_coupling_point(cp, file_analyses, language)
            annotation = f" [behavior: {behavior.value}]"
        lines.append(
            f"  [{cp.coupling_type}] {cp.source} -> "
            f"{len(cp.targets)} dependents: {cp.description}{annotation}"
        )
    return "\n".join(lines)


def _normalize_template_params(params: dict, category: str) -> dict:
    """Normalize profile template parameter keys to canonical harness keys.

    Profile templates use descriptive, dependency-specific parameter names
    (e.g. ``row_counts``, ``payload_sizes_kb``, ``initial_concurrent``).
    Harness bodies expect canonical keys (``data_sizes``, ``concurrent``,
    ``iterations``).  This function translates one to the other so
    profile-specific tuning actually takes effect at runtime.

    The original parameters are preserved alongside the canonical keys,
    so harness bodies that already read profile-specific keys (e.g.
    library-specific standalone bodies) continue to work.
    """
    out = dict(params)  # preserve originals

    if category in ("data_volume_scaling", "blocking_io"):
        # Canonical key: data_sizes (list[int])
        if "data_sizes" not in out:
            for key in (
                "row_counts", "array_sizes", "vector_counts",
                "matrix_sizes", "file_counts", "document_counts",
                "instance_counts", "chain_steps",
            ):
                if key in out and isinstance(out[key], list):
                    out["data_sizes"] = out[key]
                    break
            else:
                # Convert single-dimension size lists expressed in KB/MB
                for key in ("payload_sizes_kb", "file_sizes_mb", "response_sizes_mb"):
                    if key in out and isinstance(out[key], list):
                        # Convert to abstract item counts for the generic body
                        if "kb" in key:
                            out["data_sizes"] = [int(v * 10) for v in out[key]]
                        elif "mb" in key:
                            out["data_sizes"] = [int(v * 1000) for v in out[key]]
                        break

    elif category in ("concurrent_execution", "gil_contention", "async_failures"):
        # Canonical key: concurrent (list[int])
        if "concurrent" not in out:
            # Range-based: initial/max/step
            if "initial_concurrent" in out and "max_concurrent" in out:
                initial = out["initial_concurrent"]
                maximum = out["max_concurrent"]
                step = out.get("step", out.get("step_multiplier", 10))
                levels = []
                v = initial
                while v <= maximum:
                    levels.append(v)
                    if "step_multiplier" in out:
                        v = v * step
                    else:
                        v = v + step
                if not levels:
                    levels = [initial, maximum]
                out["concurrent"] = levels
            else:
                # Direct list keys
                for key in (
                    "thread_counts", "concurrent_requests",
                    "concurrent_threads", "concurrent_writers",
                    "concurrent_users", "concurrent_chains",
                    "concurrent_queries", "concurrent_sessions",
                    "concurrent_operations", "concurrent_connections",
                    "session_counts",
                ):
                    if key in out and isinstance(out[key], list):
                        out["concurrent"] = out[key]
                        break
                else:
                    # Scalar concurrency values → single-element list
                    for key in (
                        "concurrent_requests", "concurrent_writers",
                        "concurrent_threads", "thread_count",
                        "concurrent_users",
                    ):
                        if key in out and isinstance(out[key], (int, float)):
                            out["concurrent"] = [1, int(out[key])]
                            break

    elif category == "memory_profiling":
        # Canonical key: iterations (int)
        if "iterations" not in out:
            if "total_requests" in out:
                batch = out.get("batch_size", 100)
                out["iterations"] = max(1, out["total_requests"] // batch)
            elif "cycles" in out and isinstance(out["cycles"], int):
                out["iterations"] = out["cycles"]
            elif "sessions" in out and isinstance(out["sessions"], int):
                out["iterations"] = out["sessions"]

    return out


def _infer_measurements(category: str) -> list[str]:
    """Infer what to measure based on scenario category."""
    base = ["execution_time_ms", "error_count"]
    if category in ("memory_profiling", "event_listener_accumulation"):
        return base + ["memory_mb", "memory_growth_mb"]
    if category == "data_volume_scaling":
        return base + ["memory_mb", "throughput"]
    if category in ("concurrent_execution", "blocking_io", "gil_contention"):
        return base + ["memory_mb", "concurrent_active", "latency_p99_ms"]
    if category == "async_failures":
        return base + ["unhandled_rejections", "promise_chain_depth"]
    if category == "state_management_degradation":
        return base + ["memory_mb", "state_size_bytes"]
    return base + ["memory_mb"]


def _infer_priority_from_template(template: dict) -> str:
    """Infer scenario priority from a profile's stress test template."""
    indicators = template.get("failure_indicators", [])
    indicator_str = " ".join(str(i).lower() for i in indicators)
    if any(kw in indicator_str for kw in ("crash", "data_loss", "security", "corruption")):
        return "high"
    if any(kw in indicator_str for kw in ("timeout", "memory", "error", "blocked")):
        return "medium"
    return "low"


def _safe_name(source: str) -> str:
    """Convert a qualified name to a safe identifier for scenario naming."""
    return re.sub(r"[^a-zA-Z0-9]", "_", source)[:60].strip("_")
