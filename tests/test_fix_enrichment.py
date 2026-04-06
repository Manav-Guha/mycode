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
