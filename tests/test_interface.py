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
            # Constraint questions (user_scale, data_type, usage_pattern, max_payload)
            "about 20",
            "1",
            "1",
            "2",
            # Project name
            "Expense Tracker",
        ])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion, language="python")

        # 7 prompts: turn 1, turn 2, 4 constraint questions, project name
        assert len(io.prompts) == 7
        # First prompt asks about the project
        assert "project" in io.prompts[0].lower() or "tell me" in io.prompts[0].lower()
        # Second prompt asks about stress priorities
        assert "stress" in io.prompts[1].lower() or "matters" in io.prompts[1].lower()
        # Constraint questions asked
        assert "how many users" in io.prompts[2].lower()
        # Last prompt asks for a short project name
        assert "call this project" in io.prompts[6].lower()

        # Intent should capture responses
        assert "expenses" in result.intent.project_description.lower()
        assert "data" in result.intent.stress_priorities.lower()
        assert result.intent.project_name == "Expense Tracker"

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

    def test_project_name_empty_fallback(self, simple_ingestion):
        io = MockIO(responses=["desc", "priorities", ""])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)
        assert result.intent.project_name == ""

    def test_project_name_capped_at_50_chars(self, simple_ingestion):
        io = MockIO(responses=["desc", "priorities", "a" * 80])
        interface = ConversationalInterface(offline=True, io=io)
        result = interface.run(simple_ingestion)
        assert len(result.intent.project_name) <= 50


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
            "not sure", "not sure", "not sure", "not sure",
            "",  # project name
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

        # 7 prompts: turn 1, turn 2, 4 constraint questions, project name
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
            responses=["my app", "speed", ""],
            llm_responses=[
                "This is not JSON at all",
                "Also not JSON",
                "Still not JSON",
            ],
        )
        result = interface.run(simple_ingestion)

        # Should fall back to offline questions and still work
        assert len(io.prompts) == 3
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
            "Budget App",
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

    def test_constraints_inferred_from_turn_text(self, simple_ingestion):
        """Context-only fields extracted from turn 1 text."""
        io = MockIO(responses=[
            "It's a medical records app deployed on AWS cloud for hospital staff",
            "About 200 users uploading CSV files, steady use all day",
            # user_scale, data_type, usage_pattern inferred from turn text
            # max_payload always asked explicitly
            "3",        # max_payload (large = 100 MB)
            "Medical App",
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
            "myproject",
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
            "3",        # images
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

        # Only 3 prompts: turn 1, turn 2, project name (no constraint questions)
        assert len(io.prompts) == 3
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
