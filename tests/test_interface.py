"""Tests for Conversational Interface (E1).

Tests cover:
  - Ingester summary generation (~500 tokens, correct content)
  - 2-turn conversation flow (offline and LLM-backed)
  - OperationalIntent structure and as_intent_string()
  - Scenario review and approval (approve all, skip some, notes)
  - Offline mode (no LLM calls)
  - LLM error fallback to offline questions
  - Empty user responses
  - I/O injection for all user interactions
"""

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from mycode.ingester import (
    ClassInfo,
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionInfo,
    IngestionResult,
)
from mycode.interface import (
    ConversationalInterface,
    InterfaceError,
    InterfaceResult,
    OperationalIntent,
    ScenarioReview,
    TerminalIO,
    _human_category,
    summarize_ingestion,
)
from mycode.scenario import (
    LLMBackend,
    LLMConfig,
    LLMError,
    LLMResponse,
    StressTestScenario,
)


# ── Fixtures ──


class MockIO:
    """Injectable I/O that records displays and returns scripted responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses) if responses else []
        self.displays: list[str] = []
        self.prompts: list[str] = []
        self._response_idx = 0

    def display(self, message: str) -> None:
        self.displays.append(message)

    def prompt(self, message: str) -> str:
        self.prompts.append(message)
        if self._response_idx < len(self.responses):
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            return resp
        return ""


@pytest.fixture
def simple_ingestion() -> IngestionResult:
    """Minimal ingestion result for a Flask app."""
    return IngestionResult(
        project_path="/tmp/myapp",
        files_analyzed=3,
        files_failed=0,
        total_lines=250,
        file_analyses=[
            FileAnalysis(
                file_path="app.py",
                functions=[
                    FunctionInfo(
                        name="index",
                        file_path="app.py",
                        lineno=10,
                        decorators=["app.route"],
                        is_method=False,
                    ),
                    FunctionInfo(
                        name="get_items",
                        file_path="app.py",
                        lineno=20,
                        is_method=False,
                    ),
                ],
                classes=[],
                lines_of_code=100,
            ),
            FileAnalysis(
                file_path="models.py",
                functions=[],
                classes=[
                    ClassInfo(name="Item", file_path="models.py", lineno=1),
                    ClassInfo(name="User", file_path="models.py", lineno=20),
                ],
                lines_of_code=80,
            ),
            FileAnalysis(
                file_path="utils.py",
                functions=[
                    FunctionInfo(
                        name="format_date",
                        file_path="utils.py",
                        lineno=5,
                        is_method=False,
                    ),
                ],
                classes=[],
                lines_of_code=70,
            ),
        ],
        dependencies=[
            DependencyInfo(
                name="flask",
                installed_version="3.1.0",
                is_outdated=False,
            ),
            DependencyInfo(
                name="sqlalchemy",
                installed_version="2.0.30",
                is_outdated=False,
            ),
        ],
        coupling_points=[
            CouplingPoint(
                source="app.get_items",
                targets=["models.Item", "utils.format_date"],
                coupling_type="high_fan_in",
                description="get_items is called from multiple routes",
            ),
        ],
    )


@pytest.fixture
def large_ingestion() -> IngestionResult:
    """Larger ingestion result to test summary truncation."""
    deps = [
        DependencyInfo(name=f"dep_{i}", installed_version=f"1.{i}.0")
        for i in range(25)
    ]
    deps[0].is_outdated = True
    deps[1].is_missing = True
    return IngestionResult(
        project_path="/tmp/bigapp",
        files_analyzed=50,
        files_failed=3,
        total_lines=12000,
        file_analyses=[],
        dependencies=deps,
        coupling_points=[
            CouplingPoint(
                source="core.engine",
                targets=["a", "b", "c"],
                coupling_type="high_fan_in",
                description="Central engine",
            ),
            CouplingPoint(
                source="state.store",
                targets=["x", "y"],
                coupling_type="shared_state",
                description="Global mutable state",
            ),
        ],
        warnings=["No entry point detected", "Circular import risk"],
    )


@pytest.fixture
def sample_scenarios() -> list[StressTestScenario]:
    """Sample scenarios for review testing."""
    return [
        StressTestScenario(
            name="flask_data_volume",
            category="data_volume_scaling",
            description="Test Flask routes with increasing data sizes",
            target_dependencies=["flask"],
            priority="high",
        ),
        StressTestScenario(
            name="sqlalchemy_concurrent_queries",
            category="concurrent_execution",
            description="Test concurrent database queries under load",
            target_dependencies=["sqlalchemy"],
            priority="medium",
        ),
        StressTestScenario(
            name="edge_case_inputs",
            category="edge_case_input",
            description="Test with malformed and empty inputs",
            target_dependencies=["flask"],
            priority="low",
        ),
    ]


# ── Summarization Tests ──


class TestSummarizeIngestion:
    """Tests for summarize_ingestion()."""

    def test_basic_summary_contains_language(self, simple_ingestion):
        summary = summarize_ingestion(simple_ingestion, "python")
        assert "python" in summary.lower()

    def test_summary_contains_file_count(self, simple_ingestion):
        summary = summarize_ingestion(simple_ingestion)
        assert "3" in summary

    def test_summary_contains_dependencies(self, simple_ingestion):
        summary = summarize_ingestion(simple_ingestion)
        assert "flask" in summary.lower()
        assert "sqlalchemy" in summary.lower()

    def test_summary_contains_classes(self, simple_ingestion):
        summary = summarize_ingestion(simple_ingestion)
        assert "Item" in summary
        assert "User" in summary

    def test_summary_contains_risk_areas(self, simple_ingestion):
        summary = summarize_ingestion(simple_ingestion)
        assert "fan-in" in summary.lower() or "risk" in summary.lower()

    def test_summary_token_budget(self, simple_ingestion):
        """Summary should be roughly ~500 tokens (~2000 chars)."""
        summary = summarize_ingestion(simple_ingestion)
        # 500 tokens * 4 chars/token = 2000 chars, generous ceiling
        assert len(summary) < 4000

    def test_large_project_summary(self, large_ingestion):
        summary = summarize_ingestion(large_ingestion)
        assert "50" in summary  # files
        assert "3" in summary   # failed
        assert "12000" in summary or "12,000" in summary  # lines
        # Should truncate deps at 15
        assert "dep_0" in summary
        assert "+10 more" in summary or "+1" in summary

    def test_outdated_deps_flagged(self, large_ingestion):
        summary = summarize_ingestion(large_ingestion)
        assert "outdated" in summary.lower()

    def test_missing_deps_flagged(self, large_ingestion):
        summary = summarize_ingestion(large_ingestion)
        assert "not installed" in summary.lower()

    def test_warnings_included(self, large_ingestion):
        summary = summarize_ingestion(large_ingestion)
        assert "entry point" in summary.lower() or "circular" in summary.lower()

    def test_empty_ingestion(self):
        empty = IngestionResult(project_path="/tmp/empty")
        summary = summarize_ingestion(empty)
        assert "0" in summary  # 0 files

    def test_summary_contains_coupling_types(self, large_ingestion):
        summary = summarize_ingestion(large_ingestion)
        assert "fan-in" in summary.lower() or "shared" in summary.lower()


# ── OperationalIntent Tests ──


class TestOperationalIntent:
    """Tests for OperationalIntent data class."""

    def test_as_intent_string_full(self):
        intent = OperationalIntent(
            project_description="A budget tracking app",
            audience="small business owners",
            operating_conditions="100 users daily",
            stress_priorities="data integrity and speed",
        )
        result = intent.as_intent_string()
        assert "budget tracking" in result
        assert "small business" in result
        assert "100 users" in result
        assert "data integrity" in result

    def test_as_intent_string_summary_preferred(self):
        intent = OperationalIntent(
            summary="Custom summary text",
            project_description="desc",
        )
        assert intent.as_intent_string() == "Custom summary text"

    def test_as_intent_string_empty(self):
        intent = OperationalIntent()
        assert intent.as_intent_string() == ""

    def test_as_intent_string_partial(self):
        intent = OperationalIntent(
            project_description="A CLI tool",
            stress_priorities="memory",
        )
        result = intent.as_intent_string()
        assert "CLI tool" in result
        assert "memory" in result


# ── Offline Conversation Tests ──


class TestConversationalInterfaceOffline:
    """Tests for offline mode (no LLM calls)."""

    def test_offline_2turn_conversation(self, simple_ingestion):
        io = MockIO(responses=[
            "It's a web app for tracking expenses",
            "I care most about handling lots of data",
            # Constraint questions (user_scale, data_type, usage_pattern, max_payload, timeout)
            "about 20",
            "1",
            "1",
            "2",
            "1",
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        # 7 prompts: turn 1, turn 2, 5 constraint questions
        assert len(io.prompts) == 7
        # First prompt asks user to confirm analysis
        assert "sound right" in io.prompts[0].lower() or "project" in io.prompts[0].lower()
        # Second prompt asks targeted questions
        assert "people" in io.prompts[1].lower() or "users" in io.prompts[1].lower() or "worried" in io.prompts[1].lower()
        # Constraint questions asked
        assert "how many users" in io.prompts[2].lower()

        # Intent should capture responses
        assert "expenses" in result.intent.project_description.lower()
        assert "data" in result.intent.stress_priorities.lower()

        # Constraints should be extracted
        assert result.constraints is not None
        assert result.constraints.user_scale == 20
        assert result.constraints.data_type == "tabular"

    def test_offline_summary_displayed(self, simple_ingestion):
        io = MockIO(responses=["app", "speed", ""])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        # Summary should be displayed
        assert any("flask" in d.lower() for d in io.displays)
        assert result.project_summary

    def test_offline_no_llm_calls(self, simple_ingestion):
        io = MockIO(responses=["my app", "speed", ""])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        # Token usage should be zero
        assert result.token_usage["input_tokens"] == 0
        assert result.token_usage["output_tokens"] == 0

    def test_offline_empty_responses(self, simple_ingestion):
        io = MockIO(responses=["", "", ""])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        # Should produce warnings about empty responses
        assert any("no response" in w.lower() for w in result.warnings)
        # Intent should still be valid
        assert isinstance(result.intent, OperationalIntent)

    def test_offline_intent_raw_responses_stored(self, simple_ingestion):
        io = MockIO(responses=["desc here", "priorities here", ""])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        assert result.intent.raw_responses["turn_1"] == "desc here"
        assert result.intent.raw_responses["turn_2"] == "priorities here"

    def test_project_name_not_asked_by_interface(self, simple_ingestion):
        """Project name is now inferred by pipeline, not asked by interface."""
        io = MockIO(responses=["desc", "priorities"])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)
        # project_name should be empty — pipeline sets it
        assert result.intent.project_name == ""
        # No prompt should ask "call this project"
        assert not any("call this project" in p.lower() for p in io.prompts)


# ── LLM-Backed Conversation Tests ──


class TestConversationalInterfaceLLM:
    """Tests with mocked LLM backend."""

    def _make_interface_with_mock_backend(
        self,
        responses: list[str],
        llm_responses: list[str],
    ) -> tuple[ConversationalInterface, MockIO]:
        """Create interface with mocked LLM and scripted user I/O."""
        io = MockIO(responses=responses)
        config = LLMConfig(api_key="test-key")
        interface = ConversationalInterface(
            llm_config=config, offline=False, io=io,
        )

        # Replace backend with mock
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.generate = MagicMock(
            side_effect=[
                LLMResponse(content=r, input_tokens=100, output_tokens=50)
                for r in llm_responses
            ]
        )
        interface._backend = mock_backend

        return interface, io

    def test_llm_generates_turn1_question(self, simple_ingestion):
        interface, io = self._make_interface_with_mock_backend(
            responses=["It tracks expenses", "speed and reliability", "my app"],
            llm_responses=[
                '{"question": "I see you built a Flask app! What does it do and who will use it?"}',
                '{"question": "Got it — how many users do you expect, and what worries you most?"}',
                '{"summary": "Expense tracker for personal use", '
                '"project_description": "tracks expenses", '
                '"audience": "individual users", '
                '"operating_conditions": "light load", '
                '"stress_priorities": "speed and reliability"}',
            ],
        )
        result = interface.run(simple_ingestion)

        # LLM question should have been used
        assert "Flask" in io.prompts[0] or "expenses" in io.prompts[0].lower()
        assert result.intent.project_description

    def test_llm_token_usage_tracked(self, simple_ingestion):
        interface, _ = self._make_interface_with_mock_backend(
            responses=["app desc", "priorities", ""],
            llm_responses=[
                '{"question": "What does your app do?"}',
                '{"question": "What matters most?"}',
                '{"summary": "An app", "project_description": "app desc", '
                '"audience": "", "operating_conditions": "", '
                '"stress_priorities": "priorities"}',
            ],
        )
        result = interface.run(simple_ingestion)

        # 3 LLM calls * 100 input + 3 * 50 output
        assert result.token_usage["input_tokens"] == 300
        assert result.token_usage["output_tokens"] == 150

    def test_llm_synthesizes_structured_intent(self, simple_ingestion):
        interface, _ = self._make_interface_with_mock_backend(
            responses=[
                "It's a budgeting app for freelancers",
                "Handling large CSV imports is critical",
                "Budget App",
            ],
            llm_responses=[
                '{"question": "Tell me about your project"}',
                '{"question": "What should we focus on?"}',
                json.dumps({
                    "summary": "Budgeting app for freelancers that imports CSV data",
                    "project_description": "budgeting app",
                    "audience": "freelancers",
                    "operating_conditions": "CSV imports of varying size",
                    "stress_priorities": "large CSV import handling",
                }),
            ],
        )
        result = interface.run(simple_ingestion)

        assert "freelancer" in result.intent.audience.lower()
        assert "csv" in result.intent.stress_priorities.lower()
        assert "budget" in result.intent.summary.lower()

    def test_llm_failure_falls_back_to_offline(self, simple_ingestion):
        io = MockIO(responses=[
            "my app", "speed",
            # Constraint questions (after LLM fallback triggers offline)
            "not sure", "not sure", "not sure", "not sure", "not sure",
        ])
        config = LLMConfig(api_key="test-key")
        interface = ConversationalInterface(
            llm_config=config, offline=False, io=io,
        )

        # Mock backend that always fails
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.generate = MagicMock(
            side_effect=LLMError("connection refused"),
        )
        interface._backend = mock_backend

        result = interface.run(simple_ingestion)

        # 7 prompts: turn 1, turn 2, 5 constraint questions
        assert len(io.prompts) == 7
        assert "project" in io.prompts[0].lower() or "tell me" in io.prompts[0].lower()
        # Warnings should mention LLM failure
        assert any("llm" in w.lower() or "unavailable" in w.lower()
                    for w in result.warnings)
        # Intent should still capture user responses
        assert result.intent.project_description == "my app"
        # Should have switched to offline after first failure — backend
        # called only once (turn 1), not again for turn 2 or synthesis
        assert interface._offline is True
        assert interface._backend is None
        assert mock_backend.generate.call_count == 1

    def test_llm_bad_json_falls_back(self, simple_ingestion):
        interface, io = self._make_interface_with_mock_backend(
            responses=["my app", "speed"],
            llm_responses=[
                "This is not JSON at all",
                "Also not JSON",
                "Still not JSON",
            ],
        )
        result = interface.run(simple_ingestion)

        # Should fall back to offline questions and still work
        # 2 prompts: turn 1, turn 2 (no project name question)
        assert len(io.prompts) == 2
        assert result.intent.project_description == "my app"


# ── Scenario Review Tests ──


class TestScenarioReview:
    """Tests for scenario review and approval."""

    def test_approve_all(self, sample_scenarios):
        io = MockIO(responses=["y"])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        assert review.approved_all
        assert len(review.approved) == 3
        assert len(review.skipped) == 0

    def test_approve_all_default(self, sample_scenarios):
        """Empty response defaults to approve all."""
        io = MockIO(responses=[""])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        assert review.approved_all
        assert len(review.approved) == 3

    def test_skip_some(self, sample_scenarios):
        io = MockIO(responses=["s", "2,3"])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        assert len(review.approved) == 1
        assert review.approved[0].name == "flask_data_volume"
        assert len(review.skipped) == 2
        assert "sqlalchemy_concurrent_queries" in review.skipped
        assert "edge_case_inputs" in review.skipped

    def test_skip_invalid_numbers(self, sample_scenarios):
        io = MockIO(responses=["s", "99,abc,0"])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        # No valid selections — run all
        assert len(review.approved) == 3

    def test_add_notes(self, sample_scenarios):
        io = MockIO(responses=["n", "Focus on database queries"])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        assert review.user_notes == "Focus on database queries"
        assert len(review.approved) == 3
        assert not review.approved_all

    def test_empty_scenarios(self):
        io = MockIO(responses=[])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios([])

        assert len(review.approved) == 0
        assert any("no" in d.lower() for d in io.displays)

    def test_scenarios_displayed_with_details(self, sample_scenarios):
        io = MockIO(responses=["y"])
        interface = ConversationalInterface(offline=True, io=io)
        interface.review_scenarios(sample_scenarios)

        all_displayed = "\n".join(io.displays)
        assert "flask_data_volume" in all_displayed
        assert "Data Volume Scaling" in all_displayed
        assert "[!]" in all_displayed   # high priority
        assert "[~]" in all_displayed   # medium priority
        assert "[ ]" in all_displayed   # low priority

    def test_skip_single_scenario(self, sample_scenarios):
        io = MockIO(responses=["s", "1"])
        interface = ConversationalInterface(offline=True, io=io)
        review = interface.review_scenarios(sample_scenarios)

        assert len(review.approved) == 2
        assert len(review.skipped) == 1
        assert review.skipped[0] == "flask_data_volume"


# ── InterfaceResult Tests ──


class TestInterfaceResult:
    """Tests for InterfaceResult data class."""

    def test_default_values(self):
        result = InterfaceResult()
        assert isinstance(result.intent, OperationalIntent)
        assert result.review is None
        assert result.project_summary == ""
        assert result.token_usage == {"input_tokens": 0, "output_tokens": 0}

    def test_warnings_list(self):
        result = InterfaceResult(warnings=["warn1", "warn2"])
        assert len(result.warnings) == 2


# ── Helper Tests ──


class TestHumanCategory:
    """Tests for _human_category()."""

    def test_known_categories(self):
        assert _human_category("data_volume_scaling") == "Data Volume Scaling"
        assert _human_category("memory_profiling") == "Memory Profiling"
        assert _human_category("edge_case_input") == "Edge Case Input"
        assert _human_category("concurrent_execution") == "Concurrent Execution"
        assert _human_category("blocking_io") == "Blocking I/O"
        assert _human_category("gil_contention") == "GIL Contention"
        assert _human_category("async_failures") == "Async/Promise Failures"

    def test_unknown_category(self):
        assert _human_category("some_new_thing") == "Some New Thing"


# ── TerminalIO Tests ──


class TestTerminalIO:
    """Tests for the default TerminalIO class."""

    def test_display_calls_print(self, capsys):
        io = TerminalIO()
        io.display("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_prompt_calls_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "user response")
        io = TerminalIO()
        result = io.prompt("question?")
        assert result == "user response"


# ── Constraint Extraction Tests ──


class TestConstraintExtraction:
    """Tests for offline constraint extraction wiring in ConversationalInterface."""

    def test_constraints_extracted_from_explicit_answers(self, simple_ingestion):
        """Explicit numbered answers produce correct constraints."""
        io = MockIO(responses=[
            "It's a Flask app for tracking budgets",
            "I want to test data handling and speed",
            "50",       # user_scale
            "1",        # data_type → tabular
            "2",        # usage_pattern → burst
            "2",        # max_payload → medium (50 MB)
            "2",        # analysis_depth → standard
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        assert c.user_scale == 50
        assert c.data_type == "tabular"
        assert c.usage_pattern == "burst"
        assert c.max_payload_mb == 50.0
        assert c.availability_requirement == "business_hours"  # derived from burst
        assert c.analysis_depth == "standard"
        assert c.timeout_per_scenario == 300  # derived from standard

    def test_constraints_inferred_from_turn_text(self, simple_ingestion):
        """Context-only fields extracted from turn 1 text."""
        io = MockIO(responses=[
            "It's a medical records app deployed on AWS cloud for hospital staff",
            "About 200 users uploading CSV files, steady use all day",
            # user_scale, data_type, usage_pattern inferred from turn text
            # max_payload always asked explicitly
            "3",        # max_payload (large = 100 MB)
            "1",        # analysis_depth → quick
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        assert c.deployment_context == "cloud"
        assert c.data_sensitivity == "medical"
        assert c.user_scale == 200
        assert c.data_type == "tabular"  # "CSV" matches tabular
        assert c.usage_pattern == "sustained"  # "steady" + "all day"
        assert c.max_payload_mb == 100.0  # choice 3 = large
        assert c.availability_requirement == "always_on"  # derived from sustained
        assert c.analysis_depth == "quick"
        assert c.timeout_per_scenario == 120  # derived from quick

    def test_constraints_none_when_skipped(self, simple_ingestion):
        """Skipping all constraint questions leaves parameters as None."""
        io = MockIO(responses=[
            "just a side project",
            "nothing specific",
            "not sure",
            "skip",
            "n/a",
            "pass",
            "",
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        assert c.user_scale is None
        assert c.data_type is None
        assert c.usage_pattern is None
        assert c.max_payload_mb is None
        assert c.availability_requirement is None

    def test_constraints_raw_answers_preserved(self, simple_ingestion):
        """Raw answers include all turns and explicit questions."""
        io = MockIO(responses=[
            "turn one",
            "turn two",
            "10",
            "1",
            "3",
            "1",
            "1",        # analysis_depth → quick
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        # First two entries are turn 1 and turn 2
        assert c.raw_answers[0] == "turn one"
        assert c.raw_answers[1] == "turn two"
        # Additional entries for explicit constraint questions
        assert len(c.raw_answers) >= 2

    def test_constraints_keyword_parsing_for_data_type(self, simple_ingestion):
        """Free-text keywords in turn text match data_type."""
        io = MockIO(responses=[
            "It processes images and photos uploaded by users",
            "Speed matters most",
            "5",
            # data_type inferred from "images" + "photos" → no question asked
            "1",
            "1",
            "",
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        assert c.data_type == "images"

    def test_constraints_word_based_user_scale(self, simple_ingestion):
        """Word-based user scale like 'a few hundred' is parsed."""
        io = MockIO(responses=[
            "A dashboard for our team",
            "We need to handle a few hundred people at once",
            # user_scale inferred from "a few hundred" → no question asked
            "1",
            "1",
            "1",
            "",
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)

        c = result.constraints
        assert c is not None
        assert c.user_scale == 300  # "a few hundred" → 300

    def test_different_constraints_produce_different_results(self, simple_ingestion):
        """Same project with different constraint inputs yields different constraints."""
        # Run 1: small scale, tabular data
        # "scaling" in turn 2 infers usage_pattern="growing", so that Q is skipped
        io1 = MockIO(responses=[
            "A data processing app",
            "I want to test scaling",
            "5",        # user_scale
            "1",        # tabular
            # usage_pattern skipped — inferred "growing" from "scaling"
            "1",        # small payload
            "",          # project name
        ])
        interface1 = ConversationalInterface(offline=True, io=io1)
        result1 = interface1.run(simple_ingestion)

        # Run 2: large scale, images
        io2 = MockIO(responses=[
            "A data processing app",
            "I want to test scaling",
            "5000",     # user_scale
            "4",        # images (was choice 3 before documents added)
            # usage_pattern skipped — inferred "growing" from "scaling"
            "3",        # large payload
            "",          # project name
        ])
        interface2 = ConversationalInterface(offline=True, io=io2)
        result2 = interface2.run(simple_ingestion)

        assert result1.constraints.user_scale == 5
        assert result2.constraints.user_scale == 5000
        assert result1.constraints.data_type == "tabular"
        assert result2.constraints.data_type == "images"
        assert result1.constraints.max_payload_mb == 1.0
        assert result2.constraints.max_payload_mb == 100.0

    def test_online_mode_skips_explicit_questions(self, simple_ingestion):
        """In online mode (non-offline), no structured questions are asked."""
        io = MockIO(responses=[
            "A web app for 50 users handling CSV data",
            "Steady use, up to 10MB files",
            "",  # project name
        ])
        config = LLMConfig(api_key="test-key")
        interface = ConversationalInterface(
            llm_config=config, offline=False, io=io,
        )
        # Force offline=False but mock the backend to avoid real calls
        mock_backend = MagicMock(spec=LLMBackend)
        mock_backend.generate = MagicMock(
            side_effect=[
                LLMResponse(
                    content='{"question": "Tell me about your project"}',
                    input_tokens=100, output_tokens=50,
                ),
                LLMResponse(
                    content='{"question": "What matters most?"}',
                    input_tokens=100, output_tokens=50,
                ),
                LLMResponse(
                    content='{"summary": "A web app", "project_description": "web app", '
                            '"audience": "users", "operating_conditions": "50 users", '
                            '"stress_priorities": "CSV handling"}',
                    input_tokens=100, output_tokens=50,
                ),
            ],
        )
        interface._backend = mock_backend
        interface._offline = False

        result = interface.run(simple_ingestion)

        # Only 2 prompts: turn 1, turn 2 (no project name, no constraint questions)
        assert len(io.prompts) == 2
        # Constraints still extracted from turn text (without explicit Qs)
        assert result.constraints is not None
        assert result.constraints.user_scale == 50
        assert result.constraints.data_type == "tabular"  # "CSV" matches tabular


# ── LLM Config Fallback Tests ──


class TestInterfaceLLMInit:
    """Tests for LLM initialization edge cases."""

    def test_no_api_key_falls_back_to_offline(self):
        io = MockIO(responses=["app", "speed", ""])
        # No API key — should fall back to offline
        interface = ConversationalInterface(
            llm_config=LLMConfig(), offline=False, io=io,
        )
        assert interface._offline is True

    def test_explicit_offline(self):
        io = MockIO(responses=[])
        interface = ConversationalInterface(offline=True, io=io)
        assert interface._offline is True
        assert interface._backend is None


# ── Input Validation Tests ──


class TestInputValidation:
    """Tests for re-ask and default behavior on bad input."""

    def test_valid_input_accepted_first_try(self, simple_ingestion):
        """Valid answers are accepted without re-asking."""
        io = MockIO(responses=[
            "It's a web app",
            "speed matters",
            "20",           # user_scale — valid
            "1",            # data_type — valid (tabular)
            "2",            # usage_pattern — valid (burst)
            "1",            # max_payload — valid (small)
            "1",            # analysis_depth — valid (quick)
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.user_scale == 20
        assert result.constraints.data_type == "tabular"
        assert result.constraints.usage_pattern == "burst"
        assert result.constraints.max_payload_mb == 1.0
        assert result.constraints.analysis_depth == "quick"
        # No re-ask prompts — exactly 7 prompts total
        assert len(io.prompts) == 7

    def test_bad_input_re_asked_once(self, simple_ingestion):
        """Invalid input triggers one re-ask with clarification."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "banana",       # user_scale — invalid (first attempt)
            "50",           # user_scale — valid (second attempt)
            "1",            # data_type
            "1",            # usage_pattern
            "2",            # max_payload
            "1",            # analysis_depth → quick
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.user_scale == 50
        # 8 prompts: 2 turns + 5 constraints + 1 re-ask
        assert len(io.prompts) == 8
        # The re-ask prompt should mention "didn't catch that"
        assert any("didn't catch that" in p for p in io.prompts)

    def test_two_bad_inputs_defaults_with_notification(self, simple_ingestion):
        """Two invalid inputs → default applied and user notified."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "banana",       # user_scale — invalid (first)
            "also banana",  # user_scale — invalid (second) → defaults to 10
            "1",            # data_type
            "1",            # usage_pattern
            "2",            # max_payload
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.user_scale == 10  # default
        # User should be notified about the default
        assert any("defaulting to" in d.lower() for d in io.displays)

    def test_skip_word_accepted_without_reask(self, simple_ingestion):
        """'not sure' is accepted as opt-out, no re-ask."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "not sure",     # user_scale — explicit skip
            "1",            # data_type
            "1",            # usage_pattern
            "not sure",     # max_payload — explicit skip
            "not sure",     # analysis_depth — explicit skip
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.user_scale is None
        assert result.constraints.max_payload_mb is None
        assert result.constraints.analysis_depth is None
        # No re-asks — exactly 7 prompts
        assert len(io.prompts) == 7

    def test_data_type_reask_then_default(self, simple_ingestion):
        """data_type defaults to 'mixed' after two failures."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "10",           # user_scale — valid
            "banana",       # data_type — invalid
            "still banana", # data_type — invalid → defaults to mixed
            "1",            # usage_pattern
            "2",            # max_payload
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.data_type == "mixed"
        assert any("mixed data types" in d.lower() for d in io.displays)

    def test_usage_pattern_reask_then_valid(self, simple_ingestion):
        """usage_pattern re-asked once, valid on second try."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "10",           # user_scale
            "1",            # data_type
            "xyz",          # usage_pattern — invalid
            "steady",       # usage_pattern — valid on retry
            "2",            # max_payload
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.usage_pattern == "sustained"

    def test_max_payload_defaults_to_medium(self, simple_ingestion):
        """max_payload_mb defaults to 50.0 after two failures."""
        io = MockIO(responses=[
            "web app",
            "speed",
            "10",           # user_scale
            "1",            # data_type
            "1",            # usage_pattern
            "bananas",      # max_payload — invalid
            "more bananas", # max_payload — invalid → defaults to 50
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        assert result.constraints.max_payload_mb == 50.0
        assert any("50 MB" in d for d in io.displays)


class TestWebFollowupValidation:
    """Tests for apply_followup_answer retry logic (web path)."""

    def test_valid_answer_returns_parsed_ok(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "user_scale", "50",
        )
        assert ok is True
        assert constraints.user_scale == 50
        assert msg == ""

    def test_invalid_first_attempt_returns_clarification(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "user_scale", "banana",
        )
        assert ok is False
        assert "didn't catch that" in msg
        assert constraints.user_scale is None

    def test_invalid_retry_applies_default(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "user_scale", "banana", is_retry=True,
        )
        assert ok is True
        assert constraints.user_scale == 10  # default
        assert "defaulting to" in msg.lower()

    def test_skip_word_returns_parsed_ok(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "data_type", "not sure",
        )
        assert ok is True
        assert constraints.data_type is None  # skipped

    def test_data_type_retry_defaults_to_mixed(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "data_type", "xyz", is_retry=True,
        )
        assert ok is True
        assert constraints.data_type == "mixed"

    def test_usage_pattern_retry_defaults_to_sustained(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "usage_pattern", "xyz", is_retry=True,
        )
        assert ok is True
        assert constraints.usage_pattern == "sustained"

    def test_max_payload_retry_defaults_to_medium(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "max_payload_mb", "xyz", is_retry=True,
        )
        assert ok is True
        assert constraints.max_payload_mb == 50.0

    def test_analysis_depth_valid(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "analysis_depth", "2",
        )
        assert ok is True
        assert constraints.analysis_depth == "standard"
        assert constraints.timeout_per_scenario == 300

    def test_analysis_depth_keyword(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "analysis_depth", "deep",
        )
        assert ok is True
        assert constraints.analysis_depth == "deep"
        assert constraints.timeout_per_scenario == 600

    def test_analysis_depth_skip(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "analysis_depth", "not sure",
        )
        assert ok is True
        assert constraints.analysis_depth is None

    def test_analysis_depth_defaults_to_standard(self):
        from mycode.constraints import OperationalConstraints
        constraints = OperationalConstraints()
        constraints, ok, msg = ConversationalInterface.apply_followup_answer(
            constraints, "analysis_depth", "banana", is_retry=True,
        )
        assert ok is True
        assert constraints.analysis_depth == "standard"
        assert constraints.timeout_per_scenario == 300


# ── Timeout Parser Unit Tests ──


class TestTimeoutParser:
    """Tests for parse_timeout_per_scenario."""

    def test_numbered_choices(self):
        from mycode.constraints import parse_timeout_per_scenario
        assert parse_timeout_per_scenario("1") == 90
        assert parse_timeout_per_scenario("2") == 180
        assert parse_timeout_per_scenario("3") == 300

    def test_duration_with_units(self):
        from mycode.constraints import parse_timeout_per_scenario
        assert parse_timeout_per_scenario("2 min") == 120
        assert parse_timeout_per_scenario("120s") == 120
        assert parse_timeout_per_scenario("5 minutes") == 300
        assert parse_timeout_per_scenario("3m") == 180

    def test_plain_seconds(self):
        from mycode.constraints import parse_timeout_per_scenario
        assert parse_timeout_per_scenario("120") == 120
        assert parse_timeout_per_scenario("300") == 300

    def test_floor_enforcement(self):
        from mycode.constraints import parse_timeout_per_scenario
        assert parse_timeout_per_scenario("10") == 60
        assert parse_timeout_per_scenario("30s") == 60
        assert parse_timeout_per_scenario("0.5 min") == 60

    def test_skip_words(self):
        from mycode.constraints import parse_timeout_per_scenario
        assert parse_timeout_per_scenario("not sure") is None
        assert parse_timeout_per_scenario("skip") is None
        assert parse_timeout_per_scenario("") is None


class TestAnalysisDepthParser:
    """Tests for parse_analysis_depth."""

    def test_numbered_choices(self):
        from mycode.constraints import parse_analysis_depth
        assert parse_analysis_depth("1") == "quick"
        assert parse_analysis_depth("2") == "standard"
        assert parse_analysis_depth("3") == "deep"

    def test_keyword_matching(self):
        from mycode.constraints import parse_analysis_depth
        assert parse_analysis_depth("quick") == "quick"
        assert parse_analysis_depth("fast") == "quick"
        assert parse_analysis_depth("standard") == "standard"
        assert parse_analysis_depth("normal") == "standard"
        assert parse_analysis_depth("deep") == "deep"
        assert parse_analysis_depth("thorough") == "deep"
        assert parse_analysis_depth("comprehensive") == "deep"
        assert parse_analysis_depth("full") == "deep"

    def test_skip_words(self):
        from mycode.constraints import parse_analysis_depth
        assert parse_analysis_depth("not sure") is None
        assert parse_analysis_depth("skip") is None

    def test_garbage_returns_none(self):
        from mycode.constraints import parse_analysis_depth
        assert parse_analysis_depth("banana") is None
        assert parse_analysis_depth("42") is None

    def test_case_insensitive(self):
        from mycode.constraints import parse_analysis_depth
        assert parse_analysis_depth("QUICK") == "quick"
        assert parse_analysis_depth("Deep Analysis") == "deep"


class TestDepthMappings:
    """Tests for depth_to_timeout and depth_to_coupling_cap."""

    def test_depth_to_timeout(self):
        from mycode.constraints import depth_to_timeout
        assert depth_to_timeout("quick") == 120
        assert depth_to_timeout("standard") == 300
        assert depth_to_timeout("deep") == 600

    def test_depth_to_coupling_cap(self):
        from mycode.constraints import depth_to_coupling_cap
        assert depth_to_coupling_cap("quick") == 5
        assert depth_to_coupling_cap("standard") == 10
        assert depth_to_coupling_cap("deep") is None


class TestDocumentsDataType:
    """Tests for the 'documents' data_type category (Issue 3)."""

    def test_pdf_maps_to_documents(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("PDF") == "documents"

    def test_pdfs_maps_to_documents(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("PDFs") == "documents"

    def test_word_maps_to_documents(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("Word files") == "documents"

    def test_docx_maps_to_documents(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("docx") == "documents"

    def test_plain_text_still_maps_to_text(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("text files") == "text"

    def test_logs_still_maps_to_text(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("log files and emails") == "text"

    def test_choice_3_maps_to_documents(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("3") == "documents"

    def test_choice_6_maps_to_mixed(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("6") == "mixed"

    def test_pdf_and_csv_maps_to_mixed(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("PDFs and CSV files") == "mixed"

    def test_images_choice_now_4(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("4") == "images"

    def test_api_responses_choice_now_5(self):
        from mycode.constraints import parse_data_type
        assert parse_data_type("5") == "api_responses"


class TestTurn2QuestionNoDataType:
    """Tests for Issue 1: Turn 2 question should not ask about data type."""

    def test_turn2_cli_no_data_type_question(self, simple_ingestion):
        """CLI Turn 2 question does not ask about uploads/data type."""
        io = MockIO(responses=[
            "A web app",
            "10 users",  # answer to Turn 2 (user_scale question)
            "3",         # data_type (documents)
            "1",         # usage_pattern
            "1",         # max_payload
            "1",         # analysis_depth → quick
            "",          # project name
        ])
        interface = ConversationalInterface(offline=True, io=io)
        interface.run(simple_ingestion)

        # Turn 2 question is the second prompt (index 1)
        turn2_q = io.prompts[1]
        assert "upload" not in turn2_q.lower()
        assert "PDFs, images" not in turn2_q

    def test_turn2_web_no_data_type_question(self, simple_ingestion):
        """Web path process_turn_1 does not ask about uploads/data type."""
        interface = ConversationalInterface(offline=True)
        question = interface.process_turn_1("A web app", simple_ingestion)
        assert "upload" not in question.lower()
        assert "PDFs, images" not in question
