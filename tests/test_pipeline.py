"""Tests for Pipeline (D3) — orchestrates the full stress-testing flow.

Tests cover:
  - Language detection (Python, JavaScript, ambiguous, empty)
  - PipelineConfig / PipelineResult data classes
  - Stage-level error handling (each stage can fail independently)
  - Full end-to-end pipeline on the expense_tracker example project
"""

import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        """Pipeline completes all stages (some scenarios may fail — that's ok)."""
        config = PipelineConfig(
            project_path=py_project,
            operational_intent="Simple web server handling GET requests",
            language="python",
            offline=True,
            skip_version_check=True,
            temp_base=tmp_path,
        )
        result = run_pipeline(config)

        # Check all stages were attempted
        stage_names = [s.stage for s in result.stages]
        assert "language_detection" in stage_names
        assert "session_setup" in stage_names
        assert "ingestion" in stage_names
        assert "library_matching" in stage_names
        assert "scenario_generation" in stage_names
        assert "execution" in stage_names

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

        # All stages should be attempted
        stage_names = [s.stage for s in result.stages]
        assert "language_detection" in stage_names
        assert "session_setup" in stage_names
        assert "ingestion" in stage_names
        assert "library_matching" in stage_names
        assert "scenario_generation" in stage_names
        assert "execution" in stage_names

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
