"""Conversational Interface (E1) — LLM-mediated user interaction layer.

Extracts operational intent from the user through a 2-turn plain-language
exchange.  The user describes what their project does, who it's for, and
what conditions it operates under.  The interface then presents generated
stress scenarios for review before execution.

Per spec, this component:
  - Receives SUMMARIZED ingester output only (~500 tokens)
  - Receives NO component library profiles
  - Conducts a 2-turn exchange (ingester provides structure; conversation
    confirms user context and priorities)
  - User speaks domain language, not engineering language
  - Presents scenarios for approval before execution

LLM-dependent component (one of three).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from mycode.constraints import (
    OperationalConstraints,
    infer_availability,
    parse_data_sensitivity,
    parse_data_type,
    parse_deployment_context,
    parse_growth_expectation,
    parse_max_payload,
    parse_usage_pattern,
    parse_user_scale,
)
from mycode.ingester import IngestionResult
from mycode.scenario import (
    LLMBackend,
    LLMConfig,
    LLMError,
    LLMResponse,
    StressTestScenario,
)

logger = logging.getLogger(__name__)


# ── Constants ──

# Target token count for the ingester summary
_SUMMARY_TARGET_TOKENS = 500

# Approximate chars-per-token for estimation (conservative for English)
_CHARS_PER_TOKEN = 4


# ── Exceptions ──


class InterfaceError(Exception):
    """Base exception for conversational interface errors."""


# ── I/O Protocol ──


class UserIO(Protocol):
    """Protocol for user input/output — injectable for testing."""

    def display(self, message: str) -> None:
        """Show a message to the user."""
        ...

    def prompt(self, message: str) -> str:
        """Prompt the user for input and return their response."""
        ...


class TerminalIO:
    """Default terminal-based I/O using print/input."""

    def display(self, message: str) -> None:
        print(message)

    def prompt(self, message: str) -> str:
        print(message)
        return input("> ").strip()


# ── Data Classes ──


@dataclass
class OperationalIntent:
    """Structured output from the conversational exchange.

    This is what feeds into the Scenario Generator as ``operational_intent``.

    Attributes:
        summary: The combined plain-language intent from the user's responses.
        project_description: What the project does (from turn 1).
        audience: Who uses the project (from turn 1).
        operating_conditions: Expected load, environment, usage patterns
            (from turn 2).
        stress_priorities: What the user cares about most — performance,
            reliability, data handling, etc. (from turn 2).
        raw_responses: The user's exact text responses, keyed by turn.
    """

    summary: str = ""
    project_description: str = ""
    project_name: str = ""
    audience: str = ""
    operating_conditions: str = ""
    stress_priorities: str = ""
    raw_responses: dict[str, str] = field(default_factory=dict)

    def as_intent_string(self) -> str:
        """Format as a single string for the Scenario Generator."""
        parts = []
        if self.project_description:
            parts.append(f"Project: {self.project_description}")
        if self.audience:
            parts.append(f"Users: {self.audience}")
        if self.operating_conditions:
            parts.append(f"Conditions: {self.operating_conditions}")
        if self.stress_priorities:
            parts.append(f"Priorities: {self.stress_priorities}")
        return self.summary or ". ".join(parts)


@dataclass
class ScenarioReview:
    """Result of user reviewing generated scenarios.

    Attributes:
        approved: Scenarios the user approved for execution.
        skipped: Scenario names the user chose to skip.
        user_notes: Any freeform feedback the user gave.
        approved_all: True if user approved all without changes.
    """

    approved: list[StressTestScenario] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    user_notes: str = ""
    approved_all: bool = False


@dataclass
class InterfaceResult:
    """Complete output from the conversational interface.

    Attributes:
        intent: Structured operational intent for the Scenario Generator.
        review: Scenario review result (None if scenarios not yet generated).
        project_summary: The ~500-token summary shown to the user.
        token_usage: Cumulative LLM token usage across all interface calls.
        warnings: Non-fatal issues.
    """

    intent: OperationalIntent = field(default_factory=OperationalIntent)
    constraints: Optional[OperationalConstraints] = None
    review: Optional[ScenarioReview] = None
    project_summary: str = ""
    token_usage: dict = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )
    warnings: list[str] = field(default_factory=list)


# ── Ingester Summary ──


def summarize_ingestion(ingestion: IngestionResult, language: str = "python") -> str:
    """Produce a ~500-token summary of the ingestion result.

    This summary is what the Conversational Interface receives — NOT the
    full AST/dependency/flow analysis.  The full output goes only to the
    Scenario Generator and Report Generator.

    Includes: project type, framework, key dependencies, entry points,
    identified risk areas.
    """
    sections: list[str] = []

    # Project overview
    sections.append(f"Language: {language}")
    sections.append(f"Files: {ingestion.files_analyzed} analyzed")
    if ingestion.files_failed:
        sections.append(f"  ({ingestion.files_failed} could not be parsed)")
    sections.append(f"Total lines: {ingestion.total_lines}")

    # Key dependencies
    if ingestion.dependencies:
        dep_strs = []
        for dep in ingestion.dependencies[:15]:
            v = f" v{dep.installed_version}" if dep.installed_version else ""
            flag = ""
            if dep.is_outdated:
                flag = " [outdated]"
            elif dep.is_missing:
                flag = " [not installed]"
            dep_strs.append(f"{dep.name}{v}{flag}")
        sections.append(f"Dependencies: {', '.join(dep_strs)}")
        if len(ingestion.dependencies) > 15:
            sections.append(
                f"  (+{len(ingestion.dependencies) - 15} more)"
            )

    # Entry points and key structures
    entry_points = []
    key_classes = []
    for analysis in ingestion.file_analyses:
        if analysis.parse_error:
            continue
        # Files with __main__ or app patterns are entry points
        for func in analysis.functions:
            if func.name in ("main", "__main__") or not func.is_method:
                if any(d in ("route", "app.route", "get", "post")
                       for d in func.decorators):
                    entry_points.append(
                        f"{analysis.file_path}:{func.name} (route handler)"
                    )
        for cls in analysis.classes:
            key_classes.append(f"{analysis.file_path}:{cls.name}")

    if entry_points:
        sections.append(
            f"Entry points: {', '.join(entry_points[:8])}"
        )
    if key_classes:
        sections.append(
            f"Key classes: {', '.join(key_classes[:8])}"
        )

    # Risk areas
    risks = []
    if ingestion.coupling_points:
        high_fan = [
            cp for cp in ingestion.coupling_points
            if cp.coupling_type == "high_fan_in"
        ]
        shared = [
            cp for cp in ingestion.coupling_points
            if cp.coupling_type == "shared_state"
        ]
        if high_fan:
            risks.append(
                f"{len(high_fan)} high-fan-in coupling point(s)"
            )
        if shared:
            risks.append(
                f"{len(shared)} shared mutable state risk(s)"
            )
    if ingestion.files_failed:
        risks.append(f"{ingestion.files_failed} unparseable file(s)")

    outdated = [d for d in ingestion.dependencies if d.is_outdated]
    if outdated:
        risks.append(
            f"{len(outdated)} outdated dependency(ies)"
        )

    if risks:
        sections.append(f"Risk areas: {'; '.join(risks)}")

    # Warnings from ingester
    if ingestion.warnings:
        sections.append(
            f"Notes: {'; '.join(ingestion.warnings[:3])}"
        )

    summary = "\n".join(sections)

    # Truncate if significantly over budget
    max_chars = _SUMMARY_TARGET_TOKENS * _CHARS_PER_TOKEN * 2  # generous ceiling
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... (truncated)"

    return summary


# ── Conversational Interface ──


class ConversationalInterface:
    """LLM-mediated 2-turn exchange to extract operational intent.

    Usage::

        interface = ConversationalInterface(llm_config=LLMConfig(api_key="..."))
        result = interface.run(ingestion_result, language="python")

    For testing without API calls::

        interface = ConversationalInterface(offline=True)
        result = interface.run(ingestion_result, language="python")

    The ``io`` parameter allows injecting custom I/O for testing::

        interface = ConversationalInterface(offline=True, io=mock_io)
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        offline: bool = False,
        io: Optional[UserIO] = None,
    ) -> None:
        self._llm_config = llm_config or LLMConfig()
        self._offline = offline
        self._io: UserIO = io or TerminalIO()
        self._backend: Optional[LLMBackend] = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        if not offline:
            try:
                self._backend = LLMBackend(self._llm_config)
            except LLMError as exc:
                logger.warning(
                    "LLM backend init failed, falling back to offline: %s", exc
                )
                self._offline = True

    def run(
        self,
        ingestion: IngestionResult,
        language: str = "python",
    ) -> InterfaceResult:
        """Run the 2-turn conversational exchange.

        Args:
            ingestion: Project ingestion result (will be summarized internally).
            language: Detected project language.

        Returns:
            InterfaceResult with operational intent and token usage.
        """
        result = InterfaceResult()
        warnings: list[str] = []

        # Generate and display project summary
        summary = summarize_ingestion(ingestion, language)
        result.project_summary = summary

        self._io.display(
            "\n--- myCode Project Analysis ---\n"
            f"{summary}\n"
            "-------------------------------\n"
        )

        # Turn 1: Ask about project purpose, audience, conditions
        turn1_response = self._run_turn_1(summary, warnings)

        # Turn 2: Confirm understanding and ask about stress priorities
        turn2_response = self._run_turn_2(summary, turn1_response, warnings)

        # Extract structured constraints from conversation
        constraints = self._extract_constraints(
            turn1_response, turn2_response, warnings,
        )
        result.constraints = constraints

        # Ask for a short project name
        project_name = self._ask_project_name()

        # Synthesize into structured intent
        intent = self._synthesize_intent(
            summary, turn1_response, turn2_response, warnings,
        )
        intent.project_name = project_name
        result.intent = intent
        result.warnings = warnings
        result.token_usage = {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }

        return result

    def review_scenarios(
        self,
        scenarios: list[StressTestScenario],
    ) -> ScenarioReview:
        """Present generated scenarios for user review and approval.

        Shows each scenario with its description and priority, then asks
        the user to approve, skip specific scenarios, or approve all.

        Args:
            scenarios: Stress test scenarios from the Scenario Generator.

        Returns:
            ScenarioReview with approved/skipped scenarios.
        """
        if not scenarios:
            self._io.display("\nNo stress test scenarios were generated.")
            return ScenarioReview()

        # Display scenarios
        self._io.display(
            f"\n--- Generated Stress Test Scenarios ({len(scenarios)}) ---"
        )
        for i, s in enumerate(scenarios, 1):
            priority_marker = {
                "high": "[!]",
                "medium": "[~]",
                "low": "[ ]",
            }.get(s.priority, "[ ]")
            deps = ", ".join(s.target_dependencies) if s.target_dependencies else "general"
            self._io.display(
                f"\n  {i}. {priority_marker} {s.name}\n"
                f"     Category: {_human_category(s.category)}\n"
                f"     Targets: {deps}\n"
                f"     {s.description}"
            )
        self._io.display("\n" + "-" * 45)

        # Ask for approval
        response = self._io.prompt(
            "\nApprove all scenarios? [Y]es / [S]kip some / [N]ote feedback"
        ).strip().lower()

        if not response or response.startswith("y"):
            return ScenarioReview(
                approved=list(scenarios),
                approved_all=True,
            )

        if response.startswith("s"):
            return self._handle_skip_selection(scenarios)

        if response.startswith("n"):
            notes = self._io.prompt("Enter your feedback:")
            return ScenarioReview(
                approved=list(scenarios),
                user_notes=notes,
                approved_all=False,
            )

        # Unrecognized — approve all as default
        return ScenarioReview(
            approved=list(scenarios),
            approved_all=True,
        )

    # ── Turn Implementations ──

    def _run_turn_1(
        self,
        summary: str,
        warnings: list[str],
    ) -> str:
        """Turn 1: Ask the user to describe their project."""
        if self._offline:
            question = (
                "Tell me about your project: What does it do, who is it for, "
                "and how will people use it?"
            )
        else:
            question = self._llm_generate_question_turn1(summary, warnings)

        response = self._io.prompt(question)

        if not response:
            warnings.append(
                "No response in turn 1 — using project analysis only."
            )

        return response

    def _run_turn_2(
        self,
        summary: str,
        turn1_response: str,
        warnings: list[str],
    ) -> str:
        """Turn 2: Confirm understanding and ask about stress priorities."""
        if self._offline:
            question = (
                "What matters most when stress-testing your project? "
                "For example: handling lots of data, many users at once, "
                "bad input, speed, memory usage, or something else?"
            )
        else:
            question = self._llm_generate_question_turn2(
                summary, turn1_response, warnings,
            )

        response = self._io.prompt(question)

        if not response:
            warnings.append(
                "No response in turn 2 — using defaults for stress priorities."
            )

        return response

    def _ask_project_name(self) -> str:
        """Ask the user for a short project name."""
        response = self._io.prompt(
            "What do you call this project? "
            "(a short name like 'my budget app' or 'incident tracker')"
        )
        name = response.strip()
        if len(name) < 2:
            return ""
        # Cap at 50 chars, break at word boundary
        if len(name) > 50:
            cut = name[:50].rfind(" ")
            name = name[:cut] if cut > 10 else name[:50]
        return name

    # ── Constraint Extraction (Offline) ──

    def _extract_constraints(
        self,
        turn1: str,
        turn2: str,
        warnings: list[str],
    ) -> OperationalConstraints:
        """Extract structured constraints from conversation turns.

        In offline mode, runs context-only parsers on turn 1 text, then
        asks explicit structured questions for parameters that couldn't
        be inferred.  In online mode (future E7), the LLM extracts
        parameters — this method only handles offline extraction.

        Any parameter that cannot be extracted stays ``None``.
        """
        combined = f"{turn1} {turn2}"
        raw_answers = [turn1, turn2]

        # ── Context-only extraction from Turn 1 ──
        deployment_context = parse_deployment_context(combined)
        data_sensitivity = parse_data_sensitivity(combined)
        growth_expectation = parse_growth_expectation(combined)

        # ── Try to infer structured parameters from existing text ──
        # user_scale, data_type, usage_pattern use keyword matching that's
        # specific enough for turn text.  max_payload_mb is always asked
        # explicitly — bare numbers in turn text are too ambiguous (e.g.
        # "200 users" would be misread as 200 MB).
        user_scale = parse_user_scale(combined)
        data_type = parse_data_type(combined)
        usage_pattern = parse_usage_pattern(combined)
        max_payload_mb: float | None = None

        # ── Ask explicit questions for parameters still None ──
        if self._offline:
            if user_scale is None:
                answer = self._io.prompt(
                    "How many users do you expect at the same time? "
                    "(a number, or 'not sure')"
                )
                raw_answers.append(answer)
                user_scale = parse_user_scale(answer)

            if data_type is None:
                answer = self._io.prompt(
                    "What kind of data does your project handle?\n"
                    "  1. Tabular data (CSV, spreadsheets, databases)\n"
                    "  2. Text / documents\n"
                    "  3. Images / media / files\n"
                    "  4. API responses / JSON\n"
                    "  5. Mixed / various\n"
                    "(enter a number or describe it)"
                )
                raw_answers.append(answer)
                data_type = parse_data_type(answer)

            if usage_pattern is None:
                answer = self._io.prompt(
                    "How will people use it?\n"
                    "  1. Steady, continuous use throughout the day\n"
                    "  2. Bursts of heavy use at peak times\n"
                    "  3. Occasional, on-demand use\n"
                    "  4. Growing usage over time\n"
                    "(enter a number or describe it)"
                )
                raw_answers.append(answer)
                usage_pattern = parse_usage_pattern(answer)

            if max_payload_mb is None:
                answer = self._io.prompt(
                    "What's the largest input your project handles?\n"
                    "  1. Small (under 1 MB)\n"
                    "  2. Medium (1–50 MB)\n"
                    "  3. Large (over 50 MB)\n"
                    "(enter a number, a size like '50 MB', or 'not sure')"
                )
                raw_answers.append(answer)
                max_payload_mb = parse_max_payload(answer)

        # ── Derive availability from usage pattern ──
        availability_requirement = infer_availability(usage_pattern)

        return OperationalConstraints(
            user_scale=user_scale,
            usage_pattern=usage_pattern,
            max_payload_mb=max_payload_mb,
            data_type=data_type,
            deployment_context=deployment_context,
            availability_requirement=availability_requirement,
            data_sensitivity=data_sensitivity,
            growth_expectation=growth_expectation,
            raw_answers=raw_answers,
        )

    # ── LLM-Powered Question Generation ──

    def _llm_generate_question_turn1(
        self,
        summary: str,
        warnings: list[str],
    ) -> str:
        """Use LLM to generate a natural Turn 1 question."""
        system = (
            "You are the conversational interface for myCode, a stress-testing "
            "tool. Your job is to ask the user about their project in plain, "
            "friendly language. The user is NOT an engineer — they built this "
            "with AI tools.\n\n"
            "You've just analyzed their project. Generate ONE welcoming question "
            "that asks them:\n"
            "1. What the project does (in their own words)\n"
            "2. Who will use it\n"
            "3. How it will be used (web app, script, API, etc.)\n\n"
            "Keep it under 3 sentences. Be warm and approachable. "
            "Do NOT use technical jargon.\n\n"
            "Respond with JSON: {\"question\": \"your question here\"}"
        )
        user = f"Project summary:\n{summary}"

        fallback = (
            "Tell me about your project: What does it do, who is it for, "
            "and how will people use it?"
        )
        return self._llm_ask(system, user, "question", fallback, warnings)

    def _llm_generate_question_turn2(
        self,
        summary: str,
        turn1_response: str,
        warnings: list[str],
    ) -> str:
        """Use LLM to generate a natural Turn 2 question."""
        system = (
            "You are the conversational interface for myCode, a stress-testing "
            "tool. You've analyzed the user's project and heard their "
            "description. Now ask them about stress testing priorities.\n\n"
            "Generate ONE follow-up question that asks:\n"
            "1. What conditions their project will run under "
            "(how many users, how much data, how often)\n"
            "2. What they're most worried about "
            "(speed, crashes, data issues, memory)\n\n"
            "Reflect their language — mirror the terms they used.\n"
            "Keep it under 3 sentences. Stay friendly, no jargon.\n\n"
            "Respond with JSON: {\"question\": \"your question here\"}"
        )
        user = (
            f"Project summary:\n{summary}\n\n"
            f"User's project description:\n{turn1_response}"
        )

        fallback = (
            "What matters most when stress-testing your project? "
            "For example: handling lots of data, many users at once, "
            "bad input, speed, memory usage, or something else?"
        )
        return self._llm_ask(system, user, "question", fallback, warnings)

    def _synthesize_intent(
        self,
        summary: str,
        turn1: str,
        turn2: str,
        warnings: list[str],
    ) -> OperationalIntent:
        """Combine conversation into structured intent."""
        raw = {"turn_1": turn1, "turn_2": turn2}

        if self._offline or not turn1:
            # Offline: assemble from raw responses directly
            return OperationalIntent(
                summary=f"{turn1} {turn2}".strip() or "General-purpose project",
                project_description=turn1,
                stress_priorities=turn2,
                raw_responses=raw,
            )

        # LLM-powered synthesis
        system = (
            "You are extracting structured information from a user "
            "conversation about their project. Parse their responses into "
            "these fields.\n\n"
            "Respond with JSON:\n"
            "{\n"
            '  "summary": "1-2 sentence overall description",\n'
            '  "project_description": "what the project does",\n'
            '  "audience": "who uses it",\n'
            '  "operating_conditions": "expected load, environment, patterns",\n'
            '  "stress_priorities": "what matters most for testing"\n'
            "}"
        )
        user = (
            f"Project analysis:\n{summary}\n\n"
            f"User's description (turn 1):\n{turn1}\n\n"
            f"User's stress priorities (turn 2):\n{turn2}"
        )

        fallback_intent = OperationalIntent(
            summary=f"{turn1} {turn2}".strip(),
            project_description=turn1,
            stress_priorities=turn2,
            raw_responses=raw,
        )

        result = self._llm_ask_raw(system, user, warnings)
        if result is None:
            return fallback_intent

        try:
            data = json.loads(result)
            if not isinstance(data, dict):
                return fallback_intent
        except (json.JSONDecodeError, ValueError):
            # Try extracting JSON from response
            try:
                start = result.index("{")
                end = result.rindex("}") + 1
                data = json.loads(result[start:end])
            except (ValueError, json.JSONDecodeError):
                return fallback_intent

        return OperationalIntent(
            summary=data.get("summary", f"{turn1} {turn2}".strip()),
            project_description=data.get("project_description", turn1),
            audience=data.get("audience", ""),
            operating_conditions=data.get("operating_conditions", ""),
            stress_priorities=data.get("stress_priorities", turn2),
            raw_responses=raw,
        )

    # ── LLM Helpers ──

    def _llm_ask(
        self,
        system: str,
        user: str,
        json_key: str,
        fallback: str,
        warnings: list[str],
    ) -> str:
        """Call LLM, extract a single key from JSON response, fall back on error."""
        raw = self._llm_ask_raw(system, user, warnings)
        if raw is None:
            return fallback

        try:
            data = json.loads(raw)
            if isinstance(data, dict) and json_key in data:
                value = data[json_key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (json.JSONDecodeError, ValueError):
            pass

        # Try brace extraction
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            if isinstance(data, dict) and json_key in data:
                value = data[json_key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (ValueError, json.JSONDecodeError):
            pass

        warnings.append(f"LLM response missing '{json_key}' key — using default.")
        return fallback

    def _llm_ask_raw(
        self,
        system: str,
        user: str,
        warnings: list[str],
    ) -> Optional[str]:
        """Call LLM and return raw content string, or None on failure."""
        if self._backend is None:
            return None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self._backend.generate(messages)
            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens
            return response.content
        except LLMError as exc:
            logger.warning("LLM call failed in interface: %s", exc)
            warnings.append(f"LLM unavailable ({exc}) — using defaults.")
            # Switch to offline mode so remaining calls skip the LLM
            self._offline = True
            self._backend = None
            return None

    # ── Scenario Review Helpers ──

    def _handle_skip_selection(
        self,
        scenarios: list[StressTestScenario],
    ) -> ScenarioReview:
        """Let user select which scenarios to skip."""
        skip_input = self._io.prompt(
            "Enter scenario numbers to skip (comma-separated), "
            "e.g. '2,5,7':"
        )

        skipped_indices: set[int] = set()
        for part in skip_input.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(scenarios):
                    skipped_indices.add(idx - 1)

        approved = []
        skipped_names = []
        for i, s in enumerate(scenarios):
            if i in skipped_indices:
                skipped_names.append(s.name)
            else:
                approved.append(s)

        if skipped_names:
            self._io.display(
                f"Skipping {len(skipped_names)} scenario(s). "
                f"Running {len(approved)}."
            )
        else:
            self._io.display("No valid selections — running all scenarios.")
            approved = list(scenarios)

        return ScenarioReview(
            approved=approved,
            skipped=skipped_names,
        )


# ── Module Helpers ──


def _human_category(category: str) -> str:
    """Convert a category slug to human-readable text."""
    return {
        "data_volume_scaling": "Data Volume Scaling",
        "memory_profiling": "Memory Profiling",
        "edge_case_input": "Edge Case Input",
        "concurrent_execution": "Concurrent Execution",
        "blocking_io": "Blocking I/O",
        "gil_contention": "GIL Contention",
        "async_failures": "Async/Promise Failures",
        "event_listener_accumulation": "Event Listener Accumulation",
        "state_management_degradation": "State Management Degradation",
    }.get(category, category.replace("_", " ").title())
