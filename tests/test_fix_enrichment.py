"""Tests for LLM-enhanced fix suggestions (fix_enrichment module)."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mycode.fix_enrichment import (
    LLM_FIX_MODEL,
    _MAX_FUNCTION_LINES,
    _MAX_RESPONSE_WORDS,
    enrich_finding,
    extract_function_body,
    get_llm_fix_suggestion,
)
from mycode.ingester import FileAnalysis, FunctionInfo, IngestionResult
from mycode.report import Finding
from mycode.scenario import LLMConfig, LLMError, LLMResponse


# ── Helpers ──


def _make_ingestion(project_path: str, file_analyses: list | None = None) -> IngestionResult:
    return IngestionResult(
        project_path=project_path,
        file_analyses=file_analyses or [],
    )


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "title": "Memory leak in handler",
        "severity": "critical",
        "category": "memory_profiling",
        "source_file": "app.py",
        "source_function": "handle_request",
        "affected_dependencies": ["flask"],
    }
    defaults.update(kwargs)
    return Finding(**defaults)


def _make_llm_config(**kwargs) -> LLMConfig:
    defaults = {
        "api_key": "test-key-123",
    }
    defaults.update(kwargs)
    return LLMConfig(**defaults)


# ── extract_function_body ──


class TestExtractFunctionBody:
    def test_basic_extraction(self, tmp_path):
        source = (
            "import os\n"
            "\n"
            "def handle_request(data):\n"
            "    result = process(data)\n"
            "    return result\n"
            "\n"
            "def other():\n"
            "    pass\n"
        )
        (tmp_path / "app.py").write_text(source)

        fi = FunctionInfo(
            name="handle_request", file_path="app.py", lineno=3, end_lineno=5,
        )
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])

        body, start, end = extract_function_body(ingestion, "app.py", "handle_request")
        assert start == 3
        assert end == 5
        assert "def handle_request" in body
        assert "return result" in body

    def test_empty_source_file(self):
        ingestion = _make_ingestion("/tmp/proj")
        body, start, end = extract_function_body(ingestion, "", "handle_request")
        assert body == ""
        assert start == 0

    def test_empty_source_function(self):
        ingestion = _make_ingestion("/tmp/proj")
        body, start, end = extract_function_body(ingestion, "app.py", "")
        assert body == ""

    def test_file_not_in_analyses(self, tmp_path):
        fa = FileAnalysis(file_path="other.py", functions=[])
        ingestion = _make_ingestion(str(tmp_path), [fa])
        body, start, end = extract_function_body(ingestion, "app.py", "handle_request")
        assert body == ""

    def test_function_not_found(self, tmp_path):
        fi = FunctionInfo(name="other_func", file_path="app.py", lineno=1, end_lineno=3)
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])
        body, start, end = extract_function_body(ingestion, "app.py", "handle_request")
        assert body == ""

    def test_end_lineno_zero(self, tmp_path):
        fi = FunctionInfo(name="handle_request", file_path="app.py", lineno=1, end_lineno=0)
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])
        body, start, end = extract_function_body(ingestion, "app.py", "handle_request")
        assert body == ""

    def test_file_not_readable(self, tmp_path):
        fi = FunctionInfo(name="handle_request", file_path="app.py", lineno=1, end_lineno=3)
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])
        # File does not exist on disk
        body, start, end = extract_function_body(ingestion, "app.py", "handle_request")
        assert body == ""

    def test_truncation_long_function(self, tmp_path):
        lines = ["def big_func():"] + [f"    x = {i}" for i in range(100)]
        (tmp_path / "big.py").write_text("\n".join(lines))

        fi = FunctionInfo(name="big_func", file_path="big.py", lineno=1, end_lineno=101)
        fa = FileAnalysis(file_path="big.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])

        body, start, end = extract_function_body(ingestion, "big.py", "big_func")
        body_lines = body.splitlines()
        assert len(body_lines) == _MAX_FUNCTION_LINES + 1  # +1 for truncation comment
        assert "truncated" in body_lines[-1]

    def test_empty_function_body(self, tmp_path):
        # File exists but extracted range yields nothing
        (tmp_path / "empty.py").write_text("")
        fi = FunctionInfo(name="f", file_path="empty.py", lineno=5, end_lineno=10)
        fa = FileAnalysis(file_path="empty.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])
        body, start, end = extract_function_body(ingestion, "empty.py", "f")
        assert body == ""


# ── get_llm_fix_suggestion ──


class TestGetLlmFixSuggestion:
    def _finding(self):
        return _make_finding()

    @patch("mycode.scenario.LLMBackend")
    def test_successful_suggestion(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            content="On line 4, add a lock around the shared dict access to prevent race conditions.",
        )
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def handle_request():\n    pass", 3, 5,
            _make_llm_config(),
        )
        assert "line 4" in result
        assert "lock" in result

    def test_no_api_key(self):
        config = LLMConfig(api_key=None)
        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, config,
        )
        assert result == ""

    def test_no_config(self):
        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, None,
        )
        assert result == ""

    def test_empty_function_body(self):
        result = get_llm_fix_suggestion(
            self._finding(), "", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_empty_response(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(content="")
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_no_suggestion_response(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(content="NO_SUGGESTION")
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_response_too_long(self, mock_backend_cls):
        mock_backend = MagicMock()
        long_response = " ".join(["word"] * (_MAX_RESPONSE_WORDS + 10))
        mock_backend.generate.return_value = LLMResponse(content=long_response)
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_error_indicator_i_cannot(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            content="I cannot determine the fix from this code.",
        )
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_error_indicator_sorry(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            content="I'm sorry, I cannot help with that.",
        )
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_error_indicator_as_an_ai(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            content="As an AI language model, I need more context.",
        )
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_llm_error_fallback(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = LLMError("timeout")
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_unexpected_exception_fallback(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = RuntimeError("unexpected")
        mock_backend_cls.return_value = mock_backend

        result = get_llm_fix_suggestion(
            self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
        )
        assert result == ""

    @patch("mycode.scenario.LLMBackend")
    def test_model_override_env_var(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(content="Fix line 5.")
        mock_backend_cls.return_value = mock_backend

        with patch.dict(os.environ, {"MYCODE_FIX_LLM_MODEL": "gemini-2.5-pro"}):
            get_llm_fix_suggestion(
                self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
            )

        # Verify the config passed to LLMBackend uses the override
        call_args = mock_backend_cls.call_args[0][0]
        assert call_args.model == "gemini-2.5-pro"

    @patch("mycode.scenario.LLMBackend")
    def test_default_model_constant(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(content="Fix line 5.")
        mock_backend_cls.return_value = mock_backend

        with patch.dict(os.environ, {}, clear=True):
            # Ensure MYCODE_FIX_LLM_MODEL is not set
            os.environ.pop("MYCODE_FIX_LLM_MODEL", None)
            get_llm_fix_suggestion(
                self._finding(), "def f(): pass", 1, 1, _make_llm_config(),
            )

        call_args = mock_backend_cls.call_args[0][0]
        assert call_args.model == LLM_FIX_MODEL

    @patch("mycode.scenario.LLMBackend")
    def test_prompt_contains_finding_context(self, mock_backend_cls):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(content="Fix it.")
        mock_backend_cls.return_value = mock_backend

        finding = _make_finding(
            title="Memory leak",
            source_file="app.py",
            source_function="handler",
            affected_dependencies=["flask", "pandas"],
            _load_level=500,
        )

        get_llm_fix_suggestion(
            finding, "def handler():\n    pass", 10, 12, _make_llm_config(),
        )

        call_args = mock_backend.generate.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "Memory leak" in user_msg
        assert "app.py" in user_msg
        assert "handler()" in user_msg
        assert "flask, pandas" in user_msg
        assert "500" in user_msg
        assert "lines 10-12" in user_msg


# ── enrich_finding (integration) ──


class TestEnrichFinding:
    @patch("mycode.scenario.LLMBackend")
    def test_enriches_finding_with_suggestion(self, mock_backend_cls, tmp_path):
        source = "def handle_request(data):\n    return process(data)\n"
        (tmp_path / "app.py").write_text(source)

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            content="Add error handling around process() call on line 2.",
        )
        mock_backend_cls.return_value = mock_backend

        fi = FunctionInfo(name="handle_request", file_path="app.py", lineno=1, end_lineno=2)
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = _make_ingestion(str(tmp_path), [fa])

        finding = _make_finding()
        config = _make_llm_config()

        enrich_finding(finding, ingestion, config)
        assert "error handling" in finding.llm_fix_suggestion

    def test_skips_info_findings(self, tmp_path):
        finding = _make_finding(severity="info")
        ingestion = _make_ingestion(str(tmp_path))
        config = _make_llm_config()

        enrich_finding(finding, ingestion, config)
        assert finding.llm_fix_suggestion == ""

    def test_skips_missing_source_file(self, tmp_path):
        finding = _make_finding(source_file="")
        ingestion = _make_ingestion(str(tmp_path))
        config = _make_llm_config()

        enrich_finding(finding, ingestion, config)
        assert finding.llm_fix_suggestion == ""

    def test_skips_missing_source_function(self, tmp_path):
        finding = _make_finding(source_function="")
        ingestion = _make_ingestion(str(tmp_path))
        config = _make_llm_config()

        enrich_finding(finding, ingestion, config)
        assert finding.llm_fix_suggestion == ""

    @patch("mycode.scenario.LLMBackend")
    def test_no_enrichment_when_body_not_found(self, mock_backend_cls, tmp_path):
        fa = FileAnalysis(file_path="other.py", functions=[])
        ingestion = _make_ingestion(str(tmp_path), [fa])

        finding = _make_finding()
        config = _make_llm_config()

        enrich_finding(finding, ingestion, config)
        assert finding.llm_fix_suggestion == ""
        mock_backend_cls.assert_not_called()


# ── Constants ──


class TestConstants:
    def test_llm_fix_model_is_string(self):
        assert isinstance(LLM_FIX_MODEL, str)
        assert "gemini" in LLM_FIX_MODEL

    def test_max_function_lines(self):
        assert _MAX_FUNCTION_LINES == 80

    def test_max_response_words(self):
        assert _MAX_RESPONSE_WORDS == 100


# ── _enrich_findings_from_ingestion ──

from mycode.ingester import DependencyInfo, FunctionFlow
from mycode.report import _enrich_findings_from_ingestion, _build_call_chain


class TestEnrichFindingsFromIngestion:
    """Tests for deterministic ingestion-level enrichment."""

    def _ingestion(self, tmp_path, **kwargs):
        ing = IngestionResult(project_path=str(tmp_path))
        for key, val in kwargs.items():
            setattr(ing, key, val)
        # Ensure defaults for fields used by enrichment
        if not hasattr(ing, "dependencies") or ing.dependencies is None:
            ing.dependencies = []
        if not hasattr(ing, "function_flows") or ing.function_flows is None:
            ing.function_flows = []
        return ing

    # ── Version enrichment ──

    def test_version_enrichment_basic(self, tmp_path):
        deps = [
            DependencyInfo(name="flask", installed_version="2.0.1",
                           latest_version="3.1.0", is_outdated=True),
            DependencyInfo(name="pandas", installed_version="2.2.0",
                           latest_version="2.2.0", is_outdated=False),
        ]
        ingestion = self._ingestion(tmp_path, dependencies=deps)
        finding = _make_finding(affected_dependencies=["flask", "pandas"])
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.dep_versions == {"flask": "2.0.1", "pandas": "2.2.0"}
        assert finding.dep_latest_versions == {"flask": "3.1.0", "pandas": "2.2.0"}
        assert finding.dep_outdated == ["flask"]

    def test_version_enrichment_missing_dep(self, tmp_path):
        deps = [DependencyInfo(name="flask", installed_version="2.0.1")]
        ingestion = self._ingestion(tmp_path, dependencies=deps)
        finding = _make_finding(affected_dependencies=["flask", "unknown_lib"])
        _enrich_findings_from_ingestion([finding], ingestion)

        assert "flask" in finding.dep_versions
        assert "unknown_lib" not in finding.dep_versions

    def test_version_enrichment_no_installed_version(self, tmp_path):
        deps = [DependencyInfo(name="flask", installed_version=None)]
        ingestion = self._ingestion(tmp_path, dependencies=deps)
        finding = _make_finding(affected_dependencies=["flask"])
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.dep_versions == {}

    # ── Call chain enrichment ──

    def test_call_chain_basic(self, tmp_path):
        flows = [
            FunctionFlow(caller="app.index", callee="app.get_data", file_path="app.py", lineno=10),
            FunctionFlow(caller="app.get_data", callee="app.fetch_api", file_path="app.py", lineno=20),
        ]
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="index")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.call_chain == ["index", "get_data", "fetch_api"]

    def test_call_chain_cycle_detection(self, tmp_path):
        """Circular call graph must not cause infinite loop."""
        flows = [
            FunctionFlow(caller="app.a", callee="app.b", file_path="app.py", lineno=1),
            FunctionFlow(caller="app.b", callee="app.c", file_path="app.py", lineno=2),
            FunctionFlow(caller="app.c", callee="app.a", file_path="app.py", lineno=3),
        ]
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="a")
        _enrich_findings_from_ingestion([finding], ingestion)

        # Should not loop forever; chain should stop before revisiting 'a'
        assert len(finding.call_chain) <= 4
        assert finding.call_chain[0] == "a"
        # 'a' should not appear again in the chain
        assert finding.call_chain.count("a") == 1

    def test_call_chain_depth_limit(self, tmp_path):
        flows = [
            FunctionFlow(caller="app.f1", callee="app.f2", file_path="app.py", lineno=1),
            FunctionFlow(caller="app.f2", callee="app.f3", file_path="app.py", lineno=2),
            FunctionFlow(caller="app.f3", callee="app.f4", file_path="app.py", lineno=3),
            FunctionFlow(caller="app.f4", callee="app.f5", file_path="app.py", lineno=4),
            FunctionFlow(caller="app.f5", callee="app.f6", file_path="app.py", lineno=5),
        ]
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="f1")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert len(finding.call_chain) <= 4

    def test_call_chain_no_match(self, tmp_path):
        flows = [
            FunctionFlow(caller="app.other", callee="app.thing", file_path="app.py", lineno=1),
        ]
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="handle_request")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.call_chain == []

    def test_call_chain_single_callee_no_chain(self, tmp_path):
        """A function with no callees should not get a call chain (length 1 = not useful)."""
        flows = [
            FunctionFlow(caller="app.index", callee="app.helper", file_path="app.py", lineno=1),
        ]
        # helper has no outgoing edges, so index → helper (length 2) is useful
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="index")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.call_chain == ["index", "helper"]

    def test_call_chain_disambiguates_by_source_file(self, tmp_path):
        """When same function name in multiple files, prefer matching source_file."""
        flows = [
            FunctionFlow(caller="app.handler", callee="app.do_work", file_path="app.py", lineno=1),
            FunctionFlow(caller="other.handler", callee="other.other_work", file_path="other.py", lineno=1),
        ]
        ingestion = self._ingestion(tmp_path, function_flows=flows)
        finding = _make_finding(source_file="app.py", source_function="handler")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert "do_work" in finding.call_chain

    # ── Decorator enrichment ──

    def test_decorator_enrichment_basic(self, tmp_path):
        fi = FunctionInfo(
            name="handle_request", file_path="app.py",
            lineno=1, end_lineno=5,
            decorators=["app.route", "st.cache_data"],
        )
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = self._ingestion(tmp_path, file_analyses=[fa])
        finding = _make_finding(source_file="app.py", source_function="handle_request")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.source_decorators == ["app.route", "st.cache_data"]

    def test_decorator_enrichment_no_match(self, tmp_path):
        fi = FunctionInfo(name="other_func", file_path="app.py", lineno=1, end_lineno=5)
        fa = FileAnalysis(file_path="app.py", functions=[fi])
        ingestion = self._ingestion(tmp_path, file_analyses=[fa])
        finding = _make_finding(source_file="app.py", source_function="handle_request")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.source_decorators == []

    def test_decorator_enrichment_no_source_file(self, tmp_path):
        ingestion = self._ingestion(tmp_path)
        finding = _make_finding(source_file="", source_function="handle_request")
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.source_decorators == []

    # ── Combined enrichment ──

    def test_all_enrichments_together(self, tmp_path):
        deps = [
            DependencyInfo(name="flask", installed_version="2.0.1",
                           latest_version="3.1.0", is_outdated=True),
        ]
        flows = [
            FunctionFlow(caller="app.handle_request", callee="app.query_db",
                         file_path="app.py", lineno=10),
        ]
        fi = FunctionInfo(
            name="handle_request", file_path="app.py",
            lineno=1, end_lineno=5,
            decorators=["app.route"],
        )
        fa = FileAnalysis(file_path="app.py", functions=[fi])

        ingestion = self._ingestion(
            tmp_path, dependencies=deps, function_flows=flows, file_analyses=[fa],
        )
        finding = _make_finding(
            source_file="app.py", source_function="handle_request",
            affected_dependencies=["flask"],
        )
        _enrich_findings_from_ingestion([finding], ingestion)

        assert finding.dep_versions == {"flask": "2.0.1"}
        assert finding.dep_outdated == ["flask"]
        assert finding.call_chain == ["handle_request", "query_db"]
        assert finding.source_decorators == ["app.route"]


class TestBuildCallChain:
    """Direct tests for _build_call_chain with visited set."""

    def test_simple_chain(self):
        graph = {
            "app.a": ["app.b"],
            "app.b": ["app.c"],
        }
        chain = _build_call_chain("app.a", graph, max_depth=4)
        assert chain == ["a", "b", "c"]

    def test_cycle_terminates(self):
        graph = {
            "app.a": ["app.b"],
            "app.b": ["app.a"],
        }
        chain = _build_call_chain("app.a", graph, max_depth=4)
        assert len(chain) <= 4
        assert chain[0] == "a"
        assert chain.count("a") == 1

    def test_depth_limit(self):
        graph = {
            "a": ["b"], "b": ["c"], "c": ["d"],
            "d": ["e"], "e": ["f"],
        }
        chain = _build_call_chain("a", graph, max_depth=3)
        assert len(chain) <= 3

    def test_branching_picks_longest(self):
        graph = {
            "a": ["b", "c"],
            "b": ["d"],
            # c has no outgoing edges
        }
        chain = _build_call_chain("a", graph, max_depth=4)
        assert chain == ["a", "b", "d"]

    def test_no_edges(self):
        chain = _build_call_chain("a", {}, max_depth=4)
        assert chain == ["a"]

    def test_self_loop(self):
        graph = {"a": ["a"]}
        chain = _build_call_chain("a", graph, max_depth=4)
        # 'a' is in visited from the start, so the self-edge is skipped
        assert chain == ["a"]
