"""Tests for Pipeline (D3/E4) — orchestrates the full stress-testing flow.

Tests cover:
  - Language detection (Python, JavaScript, ambiguous, empty)
  - PipelineConfig / PipelineResult data classes
  - Stage-level error handling (each stage can fail independently)
  - Conversational interface integration (E1)
  - Report generation integration (E2)
  - Interaction recorder integration (E3)
  - Full 9-stage end-to-end pipeline on the expense_tracker example project
  - CLI entry point
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mycode.pipeline import (
    LanguageDetectionError,
    PipelineConfig,
    PipelineError,
    PipelineResult,
    StageResult,
    detect_language,
    run_pipeline,
)

# ── Constants ──

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPENSE_TRACKER = REPO_ROOT / "examples" / "expense_tracker"

# All 9 stage names in the full pipeline
_ALL_STAGES = {
    "language_detection",
    "session_setup",
    "ingestion",
    "library_matching",
    "conversation",
    "scenario_generation",
    "scenario_review",
    "execution",
    "report_generation",
}


# ── Test Helpers ──


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


# ── Fixtures ──


@pytest.fixture
def py_project(tmp_path):
    """Minimal Python project with no external deps (avoids pip install)."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        textwrap.dedent("""\
            import json
            import os
            import sqlite3
            from pathlib import Path

            DB_PATH = Path(__file__).parent / "data.db"

            def init_db():
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS items "
                    "(id INTEGER PRIMARY KEY, name TEXT, value REAL)"
                )
                conn.commit()
                conn.close()

            def add_item(name, value):
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute("INSERT INTO items (name, value) VALUES (?, ?)", (name, value))
                conn.commit()
                conn.close()

            def get_items():
                conn = sqlite3.connect(str(DB_PATH))
                rows = conn.execute("SELECT * FROM items").fetchall()
                conn.close()
                return rows

            def summarize():
                items = get_items()
                total = sum(row[2] for row in items)
                return {"count": len(items), "total": total}

            if __name__ == "__main__":
                init_db()
                add_item("test", 9.99)
                print(json.dumps(summarize()))
        """)
    )
    return project


@pytest.fixture
def js_project(tmp_path):
    """Minimal JavaScript project directory."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text('{"name":"demo","dependencies":{"express":"4.18.0"}}\n')
    (project / "index.js").write_text('const express = require("express");\n')
    return project


@pytest.fixture
def empty_project(tmp_path):
    """Project directory with no recognizable files."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("nothing here\n")
    return project


@pytest.fixture
def mixed_project(tmp_path):
    """Project with both Python and JS indicators but more Python files."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements.txt").write_text("flask\n")
    (project / "package.json").write_text('{"name":"mixed"}\n')
    (project / "app.py").write_text("print('hello')\n")
    (project / "utils.py").write_text("x = 1\n")
    (project / "index.js").write_text("console.log('hi');\n")
    return project


# ── Language Detection ──


class TestDetectLanguage:
    """Tests for detect_language()."""

    def test_python_by_indicator(self, py_project):
        assert detect_language(py_project) == "python"

    def test_javascript_by_indicator(self, js_project):
        assert detect_language(js_project) == "javascript"

    def test_empty_project_raises(self, empty_project):
        with pytest.raises(LanguageDetectionError, match="Could not determine"):
            detect_language(empty_project)

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(LanguageDetectionError, match="not a directory"):
            detect_language(f)

    def test_mixed_project_uses_file_count(self, mixed_project):
        # 2 .py files vs 1 .js file → Python wins
        assert detect_language(mixed_project) == "python"

    def test_pyproject_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname="x"\n')
        (tmp_path / "main.py").write_text("x = 1\n")
        assert detect_language(tmp_path) == "python"

    def test_tsconfig(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}\n")
        (tmp_path / "app.ts").write_text("const x: number = 1;\n")
        assert detect_language(tmp_path) == "javascript"


# ── PipelineResult ──


class TestPipelineResult:
    """Tests for PipelineResult properties."""

    def test_success_all_stages_pass(self):
        r = PipelineResult(stages=[
            StageResult(stage="a"),
            StageResult(stage="b"),
        ])
        assert r.success is True
        assert r.failed_stage is None

    def test_failure_detected(self):
        r = PipelineResult(stages=[
            StageResult(stage="a"),
            StageResult(stage="b", success=False, error="boom"),
        ])
        assert r.success is False
        assert r.failed_stage == "b"

    def test_first_failure_reported(self):
        r = PipelineResult(stages=[
            StageResult(stage="a", success=False, error="first"),
            StageResult(stage="b", success=False, error="second"),
        ])
        assert r.failed_stage == "a"

    def test_new_result_fields_default_none(self):
        r = PipelineResult()
        assert r.interface_result is None
        assert r.report is None
        assert r.recording_path is None


# ── Pipeline with forced language (skips detection) ──


class TestPipelineConfig:
    """Tests for pipeline configuration edge cases."""

    def test_invalid_language_returns_error(self, py_project, tmp_path):
        config = PipelineConfig(
            project_path=py_project,
            language="rust",
            offline=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)
        assert not result.success
        assert result.failed_stage == "language_detection"
        assert "Unsupported language" in result.stages[0].error

    def test_nonexistent_path_returns_error(self, tmp_path):
        config = PipelineConfig(
            project_path=tmp_path / "does_not_exist",
            offline=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)
        assert not result.success


# ── Pipeline runs on a minimal Python project ──


class TestPipelineMinimalPython:
    """Run the pipeline against a tiny Python project (no Flask install)."""

    @pytest.mark.slow
    def test_pipeline_runs_all_stages(self, py_project, tmp_path):
        """Pipeline completes all 9 stages."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Simple web server handling GET requests",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        # Check all 9 stages were attempted
        stage_names = {s.stage for s in result.stages}
        assert _ALL_STAGES == stage_names

        # Language should be set
        assert result.language == "python"

        # Ingestion should succeed
        assert result.ingestion is not None
        assert result.ingestion.files_analyzed >= 1

        # Scenarios should be generated
        assert result.scenarios is not None
        assert len(result.scenarios.scenarios) > 0

        # Execution should produce results (even if some fail)
        assert result.execution is not None
        assert len(result.execution.scenario_results) > 0

        # Report should be generated
        assert result.report is not None
        assert result.report.scenarios_run > 0

        # Total duration should be positive
        assert result.total_duration_ms > 0


# ── Full end-to-end on expense_tracker demo ──


class TestPipelineExpenseTracker:
    """Run the full pipeline against the expense_tracker example project."""

    @pytest.mark.slow
    def test_expense_tracker_end_to_end(self, tmp_path):
        """Full pipeline on expense_tracker produces real stress test results."""
        if not EXPENSE_TRACKER.exists():
            pytest.skip("expense_tracker example not found")

        config = PipelineConfig(
            project_path=EXPENSE_TRACKER,
            operational_intent=(
                "Personal expense tracking web app used by a single user. "
                "Records daily expenses with categories. Expected to handle "
                "a few hundred entries. Runs locally on the user's machine."
            ),
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        # Language auto-detected as Python
        assert result.language == "python"

        # All 9 stages should be attempted
        stage_names = {s.stage for s in result.stages}
        assert _ALL_STAGES == stage_names

        # Ingestion finds app.py
        assert result.ingestion is not None
        analyzed_files = {
            a.file_path for a in result.ingestion.file_analyses
        }
        assert any("app.py" in f for f in analyzed_files)

        # Flask dependency detected
        dep_names = {d.name for d in result.ingestion.dependencies}
        assert "flask" in dep_names

        # Flask profile matched in component library
        matched_profiles = {
            m.profile.name
            for m in result.profile_matches
            if m.profile is not None
        }
        assert "flask" in matched_profiles

        # Scenarios generated
        assert result.scenarios is not None
        assert len(result.scenarios.scenarios) > 0

        # Execution produced results
        assert result.execution is not None
        assert len(result.execution.scenario_results) > 0

        # Report generated
        assert result.report is not None
        report_text = result.report.as_text()
        assert "myCode Diagnostic Report" in report_text

        # At least one scenario should have steps with real measurements
        has_measurements = False
        for sr in result.execution.scenario_results:
            for step in sr.steps:
                if step.execution_time_ms > 0 or step.measurements:
                    has_measurements = True
                    break
        assert has_measurements, (
            "Expected at least one step with real timing/measurements"
        )

    @pytest.mark.slow
    def test_expense_tracker_detects_flask_scenarios(self, tmp_path):
        """Scenario generator creates Flask-specific stress tests."""
        if not EXPENSE_TRACKER.exists():
            pytest.skip("expense_tracker example not found")

        config = PipelineConfig(
            project_path=EXPENSE_TRACKER,
            operational_intent="Expense tracking API",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)
        assert result.scenarios is not None

        # Should have scenarios targeting flask
        flask_scenarios = [
            s for s in result.scenarios.scenarios
            if "flask" in s.name.lower()
            or "flask" in [t.lower() for t in s.target_dependencies]
        ]
        assert len(flask_scenarios) > 0, (
            f"Expected Flask-targeted scenarios. Got: "
            f"{[s.name for s in result.scenarios.scenarios]}"
        )

    @pytest.mark.slow
    def test_expense_tracker_categories_covered(self, tmp_path):
        """Scenarios span multiple stress test categories."""
        if not EXPENSE_TRACKER.exists():
            pytest.skip("expense_tracker example not found")

        config = PipelineConfig(
            project_path=EXPENSE_TRACKER,
            operational_intent="Expense tracker",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)
        assert result.scenarios is not None

        categories = {s.category for s in result.scenarios.scenarios}
        # Offline mode should generate at least data_volume and edge_case
        assert len(categories) >= 2, (
            f"Expected multiple categories. Got: {categories}"
        )


# ── Error handling at each stage ──


class TestPipelineErrorHandling:
    """Verify graceful degradation when stages fail."""

    @pytest.mark.slow
    def test_ingestion_failure_stops_pipeline(self, tmp_path):
        """If ingestion fails fatally, pipeline returns partial result."""
        # Create a project with a valid indicator but corrupt Python
        project = tmp_path / "project"
        project.mkdir()
        (project / "requirements.txt").write_text("flask\n")
        # No .py files at all — ingester may succeed with 0 files

        config = PipelineConfig(
            project_path=project,
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path / "work",
        )
        result = run_pipeline(config)

        # Pipeline should still run (ingester handles 0 files gracefully)
        assert result.language == "python"
        assert result.ingestion is not None

    @pytest.mark.slow
    def test_library_failure_is_nonfatal(self, py_project, tmp_path):
        """Library matching failure doesn't kill the pipeline."""
        config = PipelineConfig(
            project_path=py_project,
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )

        with patch(
            "mycode.pipeline.ComponentLibrary",
            side_effect=RuntimeError("library exploded"),
        ):
            result = run_pipeline(config)

        # Pipeline should continue past library matching
        stage_names = [s.stage for s in result.stages]
        assert "library_matching" in stage_names
        lib_stage = next(s for s in result.stages if s.stage == "library_matching")
        assert not lib_stage.success

        # Scenario generation should still be attempted
        assert "scenario_generation" in stage_names

    @pytest.mark.slow
    def test_default_operational_intent(self, py_project, tmp_path):
        """Pipeline uses a sensible default when no intent is provided."""
        config = PipelineConfig(
            project_path=py_project,
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            auto_approve_scenarios=True,
            io=MockIO(responses=["A data app", "Speed"]),
        )
        result = run_pipeline(config)
        assert result.scenarios is not None
        assert len(result.scenarios.scenarios) > 0


# ── JavaScript detection (unit-level, no execution) ──


class TestJavaScriptDetection:
    """Verify JS projects are correctly identified."""

    def test_package_json_detected(self, js_project):
        assert detect_language(js_project) == "javascript"

    def test_yarn_lock(self, tmp_path):
        (tmp_path / "yarn.lock").write_text("# yarn\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.tsx").write_text("export default function App() {}\n")
        assert detect_language(tmp_path) == "javascript"

    def test_js_files_only_no_indicator(self, tmp_path):
        (tmp_path / "main.js").write_text("console.log('x');\n")
        (tmp_path / "utils.mjs").write_text("export const x = 1;\n")
        assert detect_language(tmp_path) == "javascript"


# ── Conversational Interface Integration (E1) ──


class TestPipelineConversation:
    """Tests for the conversational interface stage."""

    @pytest.mark.slow
    def test_conversation_produces_intent(self, py_project, tmp_path):
        """Pipeline runs conversation when no operational_intent given."""
        io = MockIO(responses=[
            "A simple data tracker app",
            "Speed and data handling",
            "y",  # approve all scenarios
        ])
        config = PipelineConfig(
            project_path=py_project,
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            io=io,
        )
        result = run_pipeline(config)

        assert result.interface_result is not None
        assert result.interface_result.intent.summary
        stage_names = [s.stage for s in result.stages]
        assert "conversation" in stage_names
        conv_stage = next(s for s in result.stages if s.stage == "conversation")
        assert conv_stage.success

    @pytest.mark.slow
    def test_operational_intent_override_skips_conversation(
        self, py_project, tmp_path,
    ):
        """Providing operational_intent skips conversation."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Test API under load",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        assert result.interface_result is None  # Conversation was skipped
        conv_stage = next(s for s in result.stages if s.stage == "conversation")
        assert conv_stage.success

    @pytest.mark.slow
    def test_conversation_failure_nonfatal(self, py_project, tmp_path):
        """Conversation failure falls back to default intent."""
        with patch(
            "mycode.pipeline.ConversationalInterface",
            side_effect=RuntimeError("conversation exploded"),
        ):
            config = PipelineConfig(
                project_path=py_project,
                language="python",
                offline=True,
                skip_version_check=True,
                temp_base=tmp_path,
                auto_approve_scenarios=True,
            )
            result = run_pipeline(config)

        # Pipeline should continue despite conversation failure
        conv_stage = next(s for s in result.stages if s.stage == "conversation")
        assert not conv_stage.success
        # Scenarios should still be generated with default intent
        assert result.scenarios is not None
        assert len(result.scenarios.scenarios) > 0


# ── Scenario Review Integration ──


class TestPipelineScenarioReview:
    """Tests for the scenario review stage."""

    @pytest.mark.slow
    def test_auto_approve_with_operational_intent(self, py_project, tmp_path):
        """Scenarios auto-approved when operational_intent is provided."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Test app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        review_stage = next(s for s in result.stages if s.stage == "scenario_review")
        assert review_stage.success
        # All scenarios should have been run (auto-approved)
        assert result.execution is not None

    @pytest.mark.slow
    def test_auto_approve_flag(self, py_project, tmp_path):
        """auto_approve_scenarios=True skips review."""
        io = MockIO(responses=["A data app", "Speed"])
        config = PipelineConfig(
            project_path=py_project,
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            io=io,
            auto_approve_scenarios=True,
        )
        result = run_pipeline(config)

        review_stage = next(s for s in result.stages if s.stage == "scenario_review")
        assert review_stage.success
        assert result.execution is not None


# ── Report Generation Integration (E2) ──


class TestPipelineReport:
    """Tests for the report generation stage."""

    @pytest.mark.slow
    def test_report_generated(self, py_project, tmp_path):
        """Pipeline generates a diagnostic report."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Simple app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        assert result.report is not None
        assert result.report.scenarios_run > 0
        report_text = result.report.as_text()
        assert "myCode Diagnostic Report" in report_text

        report_stage = next(
            s for s in result.stages if s.stage == "report_generation"
        )
        assert report_stage.success

    @pytest.mark.slow
    def test_report_failure_nonfatal(self, py_project, tmp_path):
        """Report failure doesn't crash the pipeline."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Simple app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        with patch(
            "mycode.pipeline.ReportGenerator",
            side_effect=RuntimeError("report exploded"),
        ):
            result = run_pipeline(config)

        # Execution should still succeed
        assert result.execution is not None
        # Report should be None
        assert result.report is None
        report_stage = next(
            s for s in result.stages if s.stage == "report_generation"
        )
        assert not report_stage.success


# ── Interaction Recorder Integration (E3) ──


class TestPipelineRecorder:
    """Tests for the interaction recorder integration."""

    @pytest.mark.slow
    def test_recorder_with_consent(self, py_project, tmp_path):
        """Pipeline records session data when consent is given."""
        data_dir = tmp_path / "recordings"
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Test app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            consent=True,
            data_dir=data_dir,
        )
        result = run_pipeline(config)

        assert result.recording_path is not None
        assert result.recording_path.exists()

    @pytest.mark.slow
    def test_recorder_without_consent(self, py_project, tmp_path):
        """Pipeline does not record when consent is not given."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Test app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            consent=False,
        )
        result = run_pipeline(config)

        assert result.recording_path is None

    @pytest.mark.slow
    def test_recorder_failure_nonfatal(self, py_project, tmp_path):
        """Recorder failure doesn't block pipeline."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Test app",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            consent=True,
            data_dir=Path("/nonexistent/readonly/path"),
        )
        # Even if recorder can't save, pipeline should complete
        result = run_pipeline(config)
        assert result.execution is not None


# ── Full 9-stage end-to-end with conversation ──


class TestPipelineFullFlow:
    """Full pipeline with conversation, report, and recorder."""

    @pytest.mark.slow
    def test_full_flow_with_conversation_and_report(self, tmp_path):
        """Full 9-stage pipeline on expense_tracker with MockIO."""
        if not EXPENSE_TRACKER.exists():
            pytest.skip("expense_tracker example not found")

        io = MockIO(responses=[
            "It's a personal expense tracker web app",
            "I care about handling lots of entries and speed",
            "y",  # approve all scenarios
        ])
        data_dir = tmp_path / "recordings"

        config = PipelineConfig(
            project_path=EXPENSE_TRACKER,
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
            io=io,
            consent=True,
            data_dir=data_dir,
        )
        result = run_pipeline(config)

        # All 9 stages attempted
        stage_names = {s.stage for s in result.stages}
        assert _ALL_STAGES == stage_names

        # Conversation produced intent
        assert result.interface_result is not None
        assert result.interface_result.intent.summary

        # Report generated
        assert result.report is not None
        assert result.report.scenarios_run > 0
        assert "myCode Diagnostic Report" in result.report.as_text()

        # Recording saved
        assert result.recording_path is not None
        assert result.recording_path.exists()


# ── CLI entry point ──


class TestCLI:
    """Tests for the CLI entry point."""

    def test_cli_help(self):
        """CLI parser builds without error."""
        from mycode.cli import build_parser
        parser = build_parser()
        assert parser.prog == "mycode"

    def test_cli_nonexistent_path(self):
        """CLI returns error for nonexistent path."""
        from mycode.cli import main
        exit_code = main(["/nonexistent/path"])
        assert exit_code == 1

    def test_cli_not_a_directory(self, tmp_path):
        """CLI returns error for a file path."""
        from mycode.cli import main
        f = tmp_path / "file.txt"
        f.write_text("x")
        exit_code = main([str(f)])
        assert exit_code == 1
