"""Tests for Scenario Generator (D1) — stress test scenario generation.

Uses mocked LLM responses for unit tests. Tests offline mode, prompt
construction, response parsing, category filtering, BYOK configuration,
and integration with real ingester output.
"""

import json
import urllib.error
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from mycode.ingester import (
    ClassInfo,
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    FunctionInfo,
    ImportInfo,
    IngestionResult,
    ProjectIngester,
)
from mycode.library import ComponentLibrary, ProfileMatch
from mycode.library.loader import DependencyProfile
from mycode.scenario import (
    ALL_CATEGORIES,
    JAVASCRIPT_CATEGORIES,
    PYTHON_CATEGORIES,
    SHARED_CATEGORIES,
    LLMBackend,
    LLMConfig,
    LLMError,
    LLMResponse,
    LLMResponseError,
    ScenarioError,
    ScenarioGenerator,
    ScenarioGeneratorResult,
    StressTestScenario,
    _extract_json,
    _infer_measurements,
    _infer_priority_from_template,
    _safe_name,
    _serialize_coupling_points,
    _serialize_file_analyses,
    _serialize_function_flows,
    _serialize_profiles,
)


# ── Fixtures ──


def _make_profile(name: str, category: str = "web_framework") -> DependencyProfile:
    """Create a minimal DependencyProfile for testing."""
    raw = {
        "identity": {
            "name": name,
            "pypi_name": name,
            "category": category,
            "description": f"Test profile for {name}",
            "current_stable_version": "1.0.0",
            "min_supported_version": "1.0.0",
            "version_notes": {},
        },
        "scaling_characteristics": {"description": "test", "concurrency_model": "test",
                                    "bottlenecks": [], "scaling_limits": []},
        "memory_behavior": {"baseline_footprint_mb": 10, "growth_pattern": "test",
                           "known_leaks": [], "gc_behavior": "test"},
        "known_failure_modes": [
            {
                "name": "test_failure",
                "description": "A test failure mode",
                "trigger_conditions": "Under load",
                "severity": "high",
                "versions_affected": "all",
                "detection_hint": "Look for errors",
            },
        ],
        "edge_case_sensitivities": [
            {"name": "test_edge", "description": "A test edge case", "test_approach": "test"},
        ],
        "interaction_patterns": {
            "commonly_used_with": [],
            "known_conflicts": [],
            "dependency_chain_risks": [],
        },
        "stress_test_templates": [
            {
                "name": "load_test",
                "category": "concurrent_execution",
                "description": "Test under concurrent load",
                "parameters": {"concurrent": [1, 10, 100]},
                "expected_behavior": "Degrades gracefully",
                "failure_indicators": ["timeout", "crash"],
            },
            {
                "name": "memory_test",
                "category": "memory_profiling",
                "description": "Test memory growth over time",
                "parameters": {"iterations": 1000},
                "expected_behavior": "Memory stable",
                "failure_indicators": ["memory_growth_unbounded"],
            },
        ],
    }
    return DependencyProfile(
        identity=raw["identity"],
        scaling_characteristics=raw["scaling_characteristics"],
        memory_behavior=raw["memory_behavior"],
        known_failure_modes=raw["known_failure_modes"],
        edge_case_sensitivities=raw["edge_case_sensitivities"],
        interaction_patterns=raw["interaction_patterns"],
        stress_test_templates=raw["stress_test_templates"],
        raw=raw,
    )


@pytest.fixture
def sample_ingestion() -> IngestionResult:
    """Create a realistic IngestionResult for testing."""
    return IngestionResult(
        project_path="/tmp/test_project",
        files_analyzed=3,
        files_failed=0,
        total_lines=500,
        file_analyses=[
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(name="main", file_path="app.py", lineno=1, calls=["create_app"]),
                    FunctionInfo(name="create_app", file_path="app.py", lineno=10, calls=["db.init"]),
                ],
                classes=[],
                imports=[
                    ImportInfo(module="flask", names=["Flask"], is_from_import=True, lineno=1),
                    ImportInfo(module="sqlalchemy", names=[], lineno=2),
                ],
                lines_of_code=200,
            ),
            FileAnalysis(
                file_path="models.py",
                functions=[],
                classes=[ClassInfo(name="User", file_path="models.py", lineno=5,
                                  methods=["save", "delete"], bases=["Model"])],
                imports=[
                    ImportInfo(module="sqlalchemy", names=["Column"], is_from_import=True, lineno=1),
                ],
                lines_of_code=150,
            ),
            FileAnalysis(
                file_path="utils.py",
                functions=[
                    FunctionInfo(name="validate_input", file_path="utils.py", lineno=1, calls=[]),
                ],
                classes=[],
                imports=[],
                lines_of_code=150,
            ),
        ],
        dependencies=[
            DependencyInfo(name="flask", installed_version="3.1.0"),
            DependencyInfo(name="sqlalchemy", installed_version="2.0.30"),
            DependencyInfo(name="gunicorn", installed_version="21.2.0"),
        ],
        function_flows=[
            FunctionFlow(caller="app.main", callee="app.create_app", file_path="app.py", lineno=5),
            FunctionFlow(caller="app.create_app", callee="sqlalchemy.init", file_path="app.py", lineno=15),
        ],
        coupling_points=[
            CouplingPoint(
                source="app.create_app",
                targets=["app.main", "models.User.save", "models.User.delete"],
                coupling_type="high_fan_in",
                description="'app.create_app' is called by 3 functions",
            ),
        ],
    )


@pytest.fixture
def sample_matches() -> list[ProfileMatch]:
    """Create ProfileMatch objects for testing."""
    flask_profile = _make_profile("flask", "web_framework")
    sqla_profile = _make_profile("sqlalchemy", "database")
    return [
        ProfileMatch(
            dependency_name="flask",
            profile=flask_profile,
            installed_version="3.1.0",
            version_match=True,
        ),
        ProfileMatch(
            dependency_name="sqlalchemy",
            profile=sqla_profile,
            installed_version="2.0.30",
            version_match=True,
        ),
        ProfileMatch(
            dependency_name="gunicorn",
            profile=None,
            installed_version="21.2.0",
        ),
    ]


def _make_llm_response_json(
    scenarios: Optional[list[dict]] = None,
    reasoning: str = "Test reasoning",
) -> str:
    """Build a valid LLM response JSON string."""
    if scenarios is None:
        scenarios = [
            {
                "name": "concurrent_requests",
                "category": "concurrent_execution",
                "description": "Test concurrent HTTP requests",
                "target_dependencies": ["flask"],
                "test_config": {
                    "parameters": {"concurrent": [1, 10, 50]},
                    "measurements": ["response_time_ms"],
                    "resource_limits": {"memory_mb": 512},
                },
                "expected_behavior": "Response time increases linearly",
                "failure_indicators": ["timeout", "error_rate > 10%"],
                "priority": "high",
            },
            {
                "name": "data_volume_db",
                "category": "data_volume_scaling",
                "description": "Scale database queries with increasing data",
                "target_dependencies": ["sqlalchemy"],
                "test_config": {
                    "parameters": {"row_counts": [100, 1000, 10000]},
                    "measurements": ["query_time_ms", "memory_mb"],
                    "resource_limits": {"memory_mb": 512},
                },
                "expected_behavior": "Query time scales with index usage",
                "failure_indicators": ["full_table_scan", "timeout"],
                "priority": "medium",
            },
        ]
    return json.dumps({"reasoning": reasoning, "scenarios": scenarios})


# ── Tests: Category Constants ──


class TestCategories:
    """Verify stress test category definitions."""

    def test_shared_categories(self):
        expected = {"data_volume_scaling", "memory_profiling",
                    "edge_case_input", "concurrent_execution"}
        assert SHARED_CATEGORIES == frozenset(expected)

    def test_python_categories_include_shared(self):
        assert SHARED_CATEGORIES.issubset(PYTHON_CATEGORIES)

    def test_python_specific_categories(self):
        python_only = PYTHON_CATEGORIES - SHARED_CATEGORIES
        assert python_only == frozenset({"blocking_io", "gil_contention"})

    def test_javascript_categories_include_shared(self):
        assert SHARED_CATEGORIES.issubset(JAVASCRIPT_CATEGORIES)

    def test_javascript_specific_categories(self):
        js_only = JAVASCRIPT_CATEGORIES - SHARED_CATEGORIES
        assert js_only == frozenset({
            "async_failures", "event_listener_accumulation",
            "state_management_degradation",
        })

    def test_all_categories(self):
        assert ALL_CATEGORIES == PYTHON_CATEGORIES | JAVASCRIPT_CATEGORIES


class TestGetCategories:
    """Test language-specific category selection."""

    def test_python_categories(self):
        gen = ScenarioGenerator(offline=True)
        cats = gen._get_categories("python")
        assert "blocking_io" in cats
        assert "gil_contention" in cats
        assert "async_failures" not in cats

    def test_javascript_categories(self):
        gen = ScenarioGenerator(offline=True)
        cats = gen._get_categories("javascript")
        assert "async_failures" in cats
        assert "event_listener_accumulation" in cats
        assert "blocking_io" not in cats

    def test_case_insensitive(self):
        gen = ScenarioGenerator(offline=True)
        assert gen._get_categories("Python") == PYTHON_CATEGORIES
        assert gen._get_categories("JavaScript") == JAVASCRIPT_CATEGORIES


# ── Tests: LLM Config ──


class TestLLMConfig:
    """Test LLM configuration defaults and BYOK."""

    def test_default_config(self):
        config = LLMConfig()
        assert config.model == "gemini-2.0-flash"
        assert "googleapis.com" in config.base_url
        assert config.api_key is None
        assert config.max_retries == 2

    def test_byok_openai(self):
        config = LLMConfig(
            api_key="sk-test-key",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        assert config.api_key == "sk-test-key"
        assert "openai.com" in config.base_url
        assert config.model == "gpt-4o"

    def test_byok_deepseek(self):
        config = LLMConfig(
            api_key="ds-test-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        assert config.model == "deepseek-chat"

    def test_byok_custom_endpoint(self):
        config = LLMConfig(
            api_key="custom-key",
            base_url="http://localhost:8080/v1",
            model="local-model",
        )
        assert config.base_url == "http://localhost:8080/v1"


# ── Tests: LLM Backend ──


class TestLLMBackend:
    """Test LLM backend initialization and error handling."""

    def test_requires_api_key(self):
        with pytest.raises(LLMError, match="API key required"):
            LLMBackend(LLMConfig())

    def test_creates_with_key(self):
        backend = LLMBackend(LLMConfig(api_key="test-key"))
        assert backend is not None

    def test_retryable_429(self):
        assert LLMBackend._is_retryable(LLMError("LLM API returned HTTP 429: rate limited"))

    def test_retryable_500(self):
        assert LLMBackend._is_retryable(LLMError("LLM API returned HTTP 500: server error"))

    def test_not_retryable_401(self):
        assert not LLMBackend._is_retryable(LLMError("LLM API returned HTTP 401: unauthorized"))

    def test_not_retryable_400(self):
        assert not LLMBackend._is_retryable(LLMError("LLM API returned HTTP 400: bad request"))

    def test_retryable_network_error(self):
        assert LLMBackend._is_retryable(LLMError("LLM API request failed: connection refused"))


# ── Tests: JSON Extraction ──


class TestExtractJson:
    """Test JSON extraction from various LLM response formats."""

    def test_raw_json(self):
        data = _extract_json('{"reasoning": "test", "scenarios": []}')
        assert data["reasoning"] == "test"

    def test_json_in_code_block(self):
        content = '```json\n{"reasoning": "test", "scenarios": []}\n```'
        data = _extract_json(content)
        assert data["reasoning"] == "test"

    def test_json_in_bare_code_block(self):
        content = '```\n{"reasoning": "test", "scenarios": []}\n```'
        data = _extract_json(content)
        assert data["reasoning"] == "test"

    def test_json_with_leading_text(self):
        content = 'Here is the result:\n{"reasoning": "test", "scenarios": []}'
        data = _extract_json(content)
        assert data["reasoning"] == "test"

    def test_json_with_surrounding_text(self):
        content = 'Preamble text\n{"reasoning": "test", "scenarios": []}\nTrailing text'
        data = _extract_json(content)
        assert data["reasoning"] == "test"

    def test_invalid_json_raises(self):
        with pytest.raises(LLMResponseError, match="Could not extract JSON"):
            _extract_json("This is not JSON at all")

    def test_empty_string_raises(self):
        with pytest.raises(LLMResponseError):
            _extract_json("")

    def test_non_object_raises(self):
        with pytest.raises(LLMResponseError):
            _extract_json('"just a string"')


# ── Tests: Offline Generation ──


class TestOfflineGeneration:
    """Test scenario generation without LLM calls."""

    def test_offline_init_no_api_key_needed(self):
        gen = ScenarioGenerator(offline=True)
        assert gen._backend is None

    def test_offline_generates_scenarios(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app for managing users", "python")
        assert isinstance(result, ScenarioGeneratorResult)
        assert len(result.scenarios) > 0
        assert result.model_used == "offline"
        assert result.reasoning == ""

    def test_offline_all_sources_marked(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        for s in result.scenarios:
            assert s.source == "offline"

    def test_offline_generates_profile_template_scenarios(self, sample_ingestion, sample_matches):
        """Recognized deps should produce scenarios from their stress_test_templates."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        template_scenarios = [s for s in result.scenarios
                              if s.name.startswith("flask_") or s.name.startswith("sqlalchemy_")]
        assert len(template_scenarios) >= 4  # 2 templates * 2 recognized deps

    def test_offline_generates_failure_mode_scenarios(self, sample_ingestion, sample_matches):
        """Critical/high failure modes should produce edge_case_input scenarios."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        failure_scenarios = [s for s in result.scenarios if s.name.endswith("_check")]
        assert len(failure_scenarios) >= 2  # 1 high-severity mode per recognized dep

    def test_offline_generates_coupling_scenarios(self, sample_ingestion, sample_matches):
        """Coupling points should produce scenarios."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        coupling_scenarios = [s for s in result.scenarios if s.name.startswith("coupling_")]
        assert len(coupling_scenarios) >= 1

    def test_offline_generates_unrecognized_scenario(self, sample_ingestion, sample_matches):
        """Unrecognized deps should produce a generic stress scenario."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        unrec = [s for s in result.scenarios if s.name == "unrecognized_deps_generic_stress"]
        assert len(unrec) == 1
        assert "gunicorn" in unrec[0].description

    def test_offline_no_unrecognized_when_all_matched(self, sample_ingestion):
        """No unrecognized scenario when all deps have profiles."""
        matches = [
            ProfileMatch(dependency_name="flask", profile=_make_profile("flask"),
                         installed_version="3.1.0", version_match=True),
        ]
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, matches, "A web app", "python")
        unrec = [s for s in result.scenarios if "unrecognized" in s.name]
        assert len(unrec) == 0

    def test_offline_version_discrepancy_scenario(self, sample_ingestion):
        """Outdated deps should produce a version discrepancy scenario."""
        matches = [
            ProfileMatch(
                dependency_name="flask",
                profile=_make_profile("flask"),
                installed_version="2.0.0",
                version_match=False,
                version_notes="Major version behind: installed 2.0.0, current stable 1.0.0",
            ),
        ]
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, matches, "A web app", "python")
        version_scenarios = [s for s in result.scenarios if "version_discrepancy" in s.name]
        assert len(version_scenarios) == 1

    def test_offline_categories_filtered_python(self, sample_ingestion, sample_matches):
        """Python offline scenarios should only use Python-valid categories."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        for s in result.scenarios:
            assert s.category in PYTHON_CATEGORIES, (
                f"Scenario '{s.name}' has non-Python category: {s.category}"
            )

    def test_offline_categories_filtered_javascript(self, sample_ingestion, sample_matches):
        """JavaScript offline scenarios should only use JS-valid categories."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "javascript")
        for s in result.scenarios:
            assert s.category in JAVASCRIPT_CATEGORIES, (
                f"Scenario '{s.name}' has non-JS category: {s.category}"
            )

    def test_offline_empty_matches(self, sample_ingestion):
        """Should produce minimal scenarios with no profile matches."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, [], "A web app", "python")
        # Should still produce coupling point scenarios
        assert len(result.scenarios) >= 1

    def test_offline_empty_ingestion(self):
        """Should handle a near-empty ingestion result gracefully."""
        ingestion = IngestionResult(project_path="/tmp/empty")
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, [], "A web app", "python")
        assert isinstance(result, ScenarioGeneratorResult)
        assert len(result.scenarios) == 0

    def test_offline_scenario_has_test_config(self, sample_ingestion, sample_matches):
        """Every offline scenario should have a populated test_config."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        for s in result.scenarios:
            assert isinstance(s.test_config, dict), f"{s.name}: test_config not dict"
            assert len(s.test_config) > 0, f"{s.name}: empty test_config"

    def test_offline_scenario_has_failure_indicators(self, sample_ingestion, sample_matches):
        """Every offline scenario should have failure indicators."""
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")
        for s in result.scenarios:
            assert isinstance(s.failure_indicators, list), f"{s.name}: indicators not list"
            assert len(s.failure_indicators) > 0, f"{s.name}: no failure indicators"


# ── Tests: LLM Response Parsing ──


class TestLLMResponseParsing:
    """Test parsing of LLM JSON responses into scenarios."""

    def test_parse_valid_response(self):
        gen = ScenarioGenerator(offline=True)
        content = _make_llm_response_json()
        scenarios, reasoning = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        assert len(scenarios) == 2
        assert reasoning == "Test reasoning"

    def test_parse_sets_source_llm(self):
        gen = ScenarioGenerator(offline=True)
        content = _make_llm_response_json()
        scenarios, _ = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        for s in scenarios:
            assert s.source == "llm"

    def test_parse_filters_invalid_categories(self):
        """Scenarios with categories not in valid set should be skipped."""
        gen = ScenarioGenerator(offline=True)
        scenarios_data = [
            {"name": "valid", "category": "concurrent_execution",
             "description": "A valid scenario"},
            {"name": "invalid_cat", "category": "nonexistent_category",
             "description": "Should be skipped"},
        ]
        content = _make_llm_response_json(scenarios=scenarios_data)
        scenarios, _ = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        assert len(scenarios) == 1
        assert scenarios[0].name == "valid"

    def test_parse_skips_incomplete_scenarios(self):
        """Scenarios missing required fields should be skipped."""
        gen = ScenarioGenerator(offline=True)
        scenarios_data = [
            {"name": "valid", "category": "concurrent_execution",
             "description": "Complete"},
            {"name": "no_desc", "category": "concurrent_execution"},
            {"category": "concurrent_execution", "description": "No name"},
        ]
        content = _make_llm_response_json(scenarios=scenarios_data)
        scenarios, _ = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        assert len(scenarios) == 1

    def test_parse_empty_scenarios_raises(self):
        """Zero valid scenarios should raise LLMResponseError."""
        gen = ScenarioGenerator(offline=True)
        content = _make_llm_response_json(scenarios=[])
        with pytest.raises(LLMResponseError, match="zero valid scenarios"):
            gen._parse_llm_response(content, PYTHON_CATEGORIES)

    def test_parse_default_priority(self):
        """Missing priority should default to 'medium'."""
        gen = ScenarioGenerator(offline=True)
        scenarios_data = [
            {"name": "no_priority", "category": "concurrent_execution",
             "description": "No priority set"},
        ]
        content = _make_llm_response_json(scenarios=scenarios_data)
        scenarios, _ = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        assert scenarios[0].priority == "medium"

    def test_parse_preserves_test_config(self):
        gen = ScenarioGenerator(offline=True)
        content = _make_llm_response_json()
        scenarios, _ = gen._parse_llm_response(content, PYTHON_CATEGORIES)
        assert "parameters" in scenarios[0].test_config

    def test_parse_js_categories(self):
        """JavaScript-specific categories should be accepted for JS."""
        gen = ScenarioGenerator(offline=True)
        scenarios_data = [
            {"name": "async_test", "category": "async_failures",
             "description": "Test async failures"},
            {"name": "listener_test", "category": "event_listener_accumulation",
             "description": "Test listener leaks"},
        ]
        content = _make_llm_response_json(scenarios=scenarios_data)
        scenarios, _ = gen._parse_llm_response(content, JAVASCRIPT_CATEGORIES)
        assert len(scenarios) == 2


# ── Tests: Prompt Construction ──


class TestPromptConstruction:
    """Test that prompts include all required context."""

    def test_prompt_has_system_and_user(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        messages = gen._build_prompt(
            sample_ingestion,
            [m for m in sample_matches if m.profile],
            ["gunicorn"],
            "A web app for managing users",
            "python",
            PYTHON_CATEGORIES,
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_contains_categories(self):
        gen = ScenarioGenerator(offline=True)
        system = gen._build_system_prompt(PYTHON_CATEGORIES)
        assert "concurrent_execution" in system
        assert "blocking_io" in system
        assert "gil_contention" in system
        assert "JSON" in system

    def test_user_prompt_contains_project_info(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, ["gunicorn"],
            "A web app for managing users", "python",
        )
        assert "python" in user.lower()
        assert "500" in user  # total_lines
        assert "A web app for managing users" in user

    def test_user_prompt_contains_dependencies(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, ["gunicorn"],
            "intent", "python",
        )
        assert "flask" in user
        assert "sqlalchemy" in user
        assert "gunicorn" in user
        assert "Unrecognized" in user

    def test_user_prompt_contains_coupling_points(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, [], "intent", "python",
        )
        assert "Coupling Points" in user
        assert "high_fan_in" in user

    def test_user_prompt_contains_function_flows(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, [], "intent", "python",
        )
        assert "Function Call Graph" in user
        assert "app.main" in user

    def test_user_prompt_contains_failure_modes(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, [], "intent", "python",
        )
        assert "failure mode" in user.lower()

    def test_user_prompt_contains_file_structure(self, sample_ingestion, sample_matches):
        gen = ScenarioGenerator(offline=True)
        recognized = [m for m in sample_matches if m.profile]
        user = gen._build_user_prompt(
            sample_ingestion, recognized, [], "intent", "python",
        )
        assert "app.py" in user
        assert "models.py" in user


# ── Tests: LLM Mode with Mocked Backend ──


class TestLLMGeneration:
    """Test LLM-based generation with mocked HTTP calls."""

    def _mock_urlopen(self, response_json: str):
        """Create a mock for urllib.request.urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": response_json}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "model": "gemini-2.0-flash",
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("mycode.scenario.urllib.request.urlopen")
    def test_llm_generation_success(self, mock_urlopen, sample_ingestion, sample_matches):
        response_json = _make_llm_response_json()
        mock_urlopen.return_value = self._mock_urlopen(response_json)

        gen = ScenarioGenerator(llm_config=LLMConfig(api_key="test-key"))
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")

        assert len(result.scenarios) == 2
        assert result.model_used == "gemini-2.0-flash"
        assert result.token_usage["input_tokens"] == 1000
        assert result.token_usage["output_tokens"] == 500
        for s in result.scenarios:
            assert s.source == "llm"

    @patch("mycode.scenario.urllib.request.urlopen")
    def test_llm_fallback_on_network_error(self, mock_urlopen, sample_ingestion, sample_matches):
        """LLM failure should fall back to offline generation."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        gen = ScenarioGenerator(
            llm_config=LLMConfig(api_key="test-key", max_retries=0),
        )
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")

        # Should have offline scenarios
        assert len(result.scenarios) > 0
        assert result.model_used == "offline"
        assert len(result.warnings) > 0
        assert "unavailable" in result.warnings[0].lower()

    @patch("mycode.scenario.urllib.request.urlopen")
    def test_llm_fallback_on_bad_response(self, mock_urlopen, sample_ingestion, sample_matches):
        """Unparseable LLM response should fall back to offline."""
        mock_resp = self._mock_urlopen("not json at all {{{")
        # Override the content in the mock response
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "not json at all {{{"}}],
            "usage": {},
            "model": "test",
        }).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        gen = ScenarioGenerator(
            llm_config=LLMConfig(api_key="test-key", max_retries=0),
        )
        result = gen.generate(sample_ingestion, sample_matches,
                              "A web app", "python")

        assert len(result.scenarios) > 0
        assert len(result.warnings) > 0
        assert "invalid" in result.warnings[0].lower()

    @patch("mycode.scenario.urllib.request.urlopen")
    def test_llm_sends_correct_request(self, mock_urlopen, sample_ingestion, sample_matches):
        """Verify the HTTP request has correct structure."""
        response_json = _make_llm_response_json()
        mock_urlopen.return_value = self._mock_urlopen(response_json)

        gen = ScenarioGenerator(
            llm_config=LLMConfig(api_key="test-key", model="gpt-4o",
                                 base_url="https://api.openai.com/v1"),
        )
        gen.generate(sample_ingestion, sample_matches, "A web app", "python")

        # Verify the request was made
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert "openai.com" in req.full_url
        assert req.get_header("Authorization") == "Bearer test-key"
        assert req.get_header("Content-type") == "application/json"

        body = json.loads(req.data)
        assert body["model"] == "gpt-4o"
        assert body["response_format"] == {"type": "json_object"}
        assert len(body["messages"]) == 2


# ── Tests: Serialization Helpers ──


class TestSerializationHelpers:
    """Test prompt serialization functions."""

    def test_serialize_file_analyses(self):
        analyses = [
            FileAnalysis(
                file_path="app.py",
                functions=[FunctionInfo(name="main", file_path="app.py", lineno=1)],
                classes=[ClassInfo(name="App", file_path="app.py", lineno=10)],
                imports=[],
                lines_of_code=100,
            ),
        ]
        result = _serialize_file_analyses(analyses)
        assert "app.py" in result
        assert "100 lines" in result
        assert "App" in result
        assert "main" in result

    def test_serialize_file_analyses_with_parse_error(self):
        analyses = [
            FileAnalysis(file_path="broken.py", parse_error="Syntax error at line 5"),
        ]
        result = _serialize_file_analyses(analyses)
        assert "PARSE ERROR" in result
        assert "broken.py" in result

    def test_serialize_profiles(self):
        profile = _make_profile("flask")
        matches = [ProfileMatch(dependency_name="flask", profile=profile)]
        result = _serialize_profiles(matches)
        assert "flask" in result
        assert "failure mode" in result.lower()
        assert "test_failure" in result

    def test_serialize_function_flows(self):
        flows = [
            FunctionFlow(caller="a.func1", callee="b.func2", file_path="a.py", lineno=10),
        ]
        result = _serialize_function_flows(flows)
        assert "a.func1" in result
        assert "b.func2" in result

    def test_serialize_function_flows_with_limit(self):
        flows = [
            FunctionFlow(caller=f"f{i}", callee=f"g{i}", file_path="a.py", lineno=i)
            for i in range(50)
        ]
        result = _serialize_function_flows(flows, limit=5)
        assert "+45 more flows" in result

    def test_serialize_coupling_points(self):
        points = [
            CouplingPoint(source="mod.func", targets=["a", "b", "c"],
                         coupling_type="high_fan_in",
                         description="Called by 3 functions"),
        ]
        result = _serialize_coupling_points(points)
        assert "high_fan_in" in result
        assert "3 dependents" in result


# ── Tests: Utility Functions ──


class TestUtilityFunctions:
    """Test helper functions."""

    def test_infer_measurements_memory(self):
        m = _infer_measurements("memory_profiling")
        assert "memory_mb" in m
        assert "memory_growth_mb" in m

    def test_infer_measurements_concurrent(self):
        m = _infer_measurements("concurrent_execution")
        assert "concurrent_active" in m
        assert "latency_p99_ms" in m

    def test_infer_measurements_data_volume(self):
        m = _infer_measurements("data_volume_scaling")
        assert "throughput" in m

    def test_infer_measurements_async(self):
        m = _infer_measurements("async_failures")
        assert "unhandled_rejections" in m

    def test_infer_measurements_state(self):
        m = _infer_measurements("state_management_degradation")
        assert "state_size_bytes" in m

    def test_infer_priority_high(self):
        template = {"failure_indicators": ["crash", "data_loss"]}
        assert _infer_priority_from_template(template) == "high"

    def test_infer_priority_medium(self):
        template = {"failure_indicators": ["timeout", "memory_growth"]}
        assert _infer_priority_from_template(template) == "medium"

    def test_infer_priority_low(self):
        template = {"failure_indicators": ["slow_response"]}
        assert _infer_priority_from_template(template) == "low"

    def test_safe_name(self):
        assert _safe_name("module.Class.method") == "module_Class_method"

    def test_safe_name_truncates(self):
        long_name = "a" * 100
        assert len(_safe_name(long_name)) <= 60


# ── Tests: Integration with Real Profiles ──


class TestRealProfileIntegration:
    """Integration tests using real component library profiles."""

    MYCODE_SRC = Path(__file__).resolve().parent.parent / "src" / "mycode"

    @pytest.fixture
    def real_library(self):
        return ComponentLibrary()

    def test_offline_with_real_flask_profile(self, real_library):
        """Generate scenarios using real Flask profile."""
        ingestion = IngestionResult(
            project_path="/tmp/test",
            files_analyzed=1,
            total_lines=100,
            file_analyses=[
                FileAnalysis(file_path="app.py", lines_of_code=100,
                             imports=[ImportInfo(module="flask", lineno=1)]),
            ],
            dependencies=[DependencyInfo(name="flask", installed_version="3.1.0")],
        )
        flask_match = ProfileMatch(
            dependency_name="flask",
            profile=real_library.get_profile("python", "flask"),
            installed_version="3.1.0",
            version_match=True,
        )
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, [flask_match],
                              "A small web API for internal use", "python")

        assert len(result.scenarios) > 0
        # Flask profile has multiple stress templates and failure modes
        flask_scenarios = [s for s in result.scenarios if "flask" in s.name]
        assert len(flask_scenarios) >= 3

        # Should have concurrent execution scenarios (Flask template)
        categories = {s.category for s in result.scenarios}
        assert "concurrent_execution" in categories

    def test_offline_with_real_express_profile(self, real_library):
        """Generate scenarios using real Express profile for JavaScript."""
        ingestion = IngestionResult(
            project_path="/tmp/test",
            files_analyzed=1,
            total_lines=80,
            file_analyses=[],
            dependencies=[DependencyInfo(name="express", installed_version="4.21.0")],
        )
        express_match = ProfileMatch(
            dependency_name="express",
            profile=real_library.get_profile("javascript", "express"),
            installed_version="4.21.0",
            version_match=True,
        )
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, [express_match],
                              "A REST API for a mobile app", "javascript")

        assert len(result.scenarios) > 0
        # Should only have JS-valid categories
        for s in result.scenarios:
            assert s.category in JAVASCRIPT_CATEGORIES

    def test_offline_with_multiple_real_profiles(self, real_library):
        """Generate scenarios for a multi-dependency project."""
        ingestion = IngestionResult(
            project_path="/tmp/test",
            files_analyzed=5,
            total_lines=1000,
            file_analyses=[],
            dependencies=[
                DependencyInfo(name="flask", installed_version="3.1.0"),
                DependencyInfo(name="sqlalchemy", installed_version="2.0.30"),
                DependencyInfo(name="requests", installed_version="2.32.3"),
            ],
            coupling_points=[
                CouplingPoint(
                    source="db.session", targets=["routes.get", "routes.post", "routes.delete"],
                    coupling_type="high_fan_in",
                    description="Database session used by 3 route handlers",
                ),
            ],
        )
        matches = [
            ProfileMatch(dependency_name="flask",
                         profile=real_library.get_profile("python", "flask"),
                         installed_version="3.1.0", version_match=True),
            ProfileMatch(dependency_name="sqlalchemy",
                         profile=real_library.get_profile("python", "sqlalchemy"),
                         installed_version="2.0.30", version_match=True),
            ProfileMatch(dependency_name="requests",
                         profile=real_library.get_profile("python", "requests"),
                         installed_version="2.32.3", version_match=True),
        ]
        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, matches,
                              "Internal tool that queries external APIs and stores results in a database",
                              "python")

        assert len(result.scenarios) >= 10  # Multiple templates + failure modes + coupling
        # Should cover multiple categories
        categories = {s.category for s in result.scenarios}
        assert len(categories) >= 3

    def test_offline_with_mycode_own_source(self, real_library):
        """Generate scenarios for myCode's own codebase using real ingester."""
        ingester = ProjectIngester(
            project_path=self.MYCODE_SRC,
            skip_pypi_check=True,
        )
        ingestion = ingester.ingest()

        # Match stdlib imports against profiles
        all_imports = set()
        for analysis in ingestion.file_analyses:
            for imp in analysis.imports:
                top = imp.module.split(".")[0] if imp.module else ""
                if top:
                    all_imports.add(top)

        dep_dicts = [{"name": n} for n in sorted(all_imports)]
        matches = real_library.match_dependencies("python", dep_dicts)

        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, matches,
                              "A stress-testing CLI tool for AI-generated code", "python")

        assert isinstance(result, ScenarioGeneratorResult)
        # myCode has coupling points, so should generate some scenarios
        assert len(result.scenarios) >= 1
