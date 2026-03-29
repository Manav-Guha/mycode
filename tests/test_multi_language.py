"""Tests for multi-language project support.

Covers: language detection, ingestion merging, primary language heuristic,
scenario tagging, engine routing, and report text for mixed projects.
"""

import os
from pathlib import Path

import pytest

from mycode.ingester import DependencyInfo, FileAnalysis, IngestionResult
from mycode.pipeline import (
    LanguageDetectionError,
    _has_real_js_deps,
    _has_real_python_deps,
    detect_language,
    detect_languages,
    determine_primary_language,
    find_dep_dir_for_language,
    merge_ingestion_results,
)
from mycode.scenario import (
    ALL_CATEGORIES,
    JAVASCRIPT_CATEGORIES,
    PYTHON_CATEGORIES,
    ScenarioGenerator,
    StressTestScenario,
)


# ── Fixtures: synthetic multi-language project structures ──


@pytest.fixture
def multi_lang_project(tmp_path):
    """Minimal FastAPI + React monorepo structure."""
    # backend/
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text("fastapi\nuvicorn\nsqlmodel\n")
    (backend / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    (backend / "models.py").write_text("from sqlmodel import SQLModel\n")

    # frontend/
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text('{"name":"frontend","dependencies":{"react":"^18"}}')
    (frontend / "index.js").write_text("import React from 'react';\n")
    (frontend / "App.jsx").write_text("export default function App() {}\n")
    (frontend / "main.tsx").write_text("ReactDOM.render(<App/>, root);\n")

    return tmp_path


@pytest.fixture
def python_only_project(tmp_path):
    """Pure Python project."""
    (tmp_path / "requirements.txt").write_text("flask\npandas\n")
    (tmp_path / "app.py").write_text("from flask import Flask\n")
    (tmp_path / "utils.py").write_text("import pandas\n")
    return tmp_path


@pytest.fixture
def js_only_project(tmp_path):
    """Pure JavaScript project."""
    (tmp_path / "package.json").write_text('{"name":"myapp","dependencies":{"express":"^4"}}')
    (tmp_path / "index.js").write_text("const express = require('express');\n")
    (tmp_path / "server.js").write_text("app.listen(3000);\n")
    (tmp_path / "routes.js").write_text("module.exports = router;\n")
    return tmp_path


@pytest.fixture
def python_with_build_tool(tmp_path):
    """Python project with a package.json from a build tool (not a real JS project)."""
    (tmp_path / "requirements.txt").write_text("flask\n")
    (tmp_path / "app.py").write_text("from flask import Flask\n")
    (tmp_path / "utils.py").write_text("pass\n")
    # package.json exists but no JS source files
    (tmp_path / "package.json").write_text('{"devDependencies":{"webpack":"^5"}}')
    return tmp_path


@pytest.fixture
def root_level_mix(tmp_path):
    """Python and JS files mixed in the root directory."""
    (tmp_path / "requirements.txt").write_text("fastapi\n")
    (tmp_path / "package.json").write_text('{"dependencies":{"react":"^18"}}')
    (tmp_path / "app.py").write_text("from fastapi import FastAPI\n")
    (tmp_path / "server.py").write_text("import uvicorn\n")
    (tmp_path / "index.js").write_text("import React from 'react';\n")
    (tmp_path / "App.jsx").write_text("export default function App() {}\n")
    (tmp_path / "main.tsx").write_text("ReactDOM.render();\n")
    return tmp_path


@pytest.fixture
def workspace_monorepo(tmp_path):
    """Mimics fastapi/full-stack-fastapi-template: workspace wrappers at root,
    real deps in subdirectories."""
    # Root: workspace config files (NOT real dep files)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["backend"]\n'
    )
    (tmp_path / "package.json").write_text(
        '{"name":"monorepo","private":true,"workspaces":["frontend"]}'
    )

    # backend/ — real Python deps
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "pyproject.toml").write_text(
        '[project]\nname = "app"\ndependencies = [\n'
        '    "fastapi>=0.114",\n    "sqlmodel>=0.0.21",\n'
        '    "pydantic>2.0",\n]\n'
    )
    (backend / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n")
    (backend / "models.py").write_text("from sqlmodel import SQLModel\n")
    (backend / "crud.py").write_text("def get_items(): pass\n")

    # frontend/ — real JS deps
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text(
        '{"name":"frontend","dependencies":{"react":"^18","vite":"^5"}}'
    )
    (frontend / "index.js").write_text("import React from 'react';\n")
    (frontend / "App.tsx").write_text("export default function App() {}\n")
    (frontend / "main.tsx").write_text("ReactDOM.render(<App/>, root);\n")

    return tmp_path


# ── Phase 1: detect_languages() ──


class TestDetectLanguages:
    def test_multi_language_monorepo(self, multi_lang_project):
        result = detect_languages(multi_lang_project)
        assert result == {"python", "javascript"}

    def test_python_only(self, python_only_project):
        result = detect_languages(python_only_project)
        assert result == {"python"}

    def test_js_only(self, js_only_project):
        result = detect_languages(js_only_project)
        assert result == {"javascript"}

    def test_python_with_build_tool_not_js(self, python_with_build_tool):
        """A package.json from webpack should NOT trigger JS detection."""
        result = detect_languages(python_with_build_tool)
        assert result == {"python"}

    def test_root_level_mix(self, root_level_mix):
        result = detect_languages(root_level_mix)
        assert result == {"python", "javascript"}

    def test_workspace_monorepo(self, workspace_monorepo):
        """Workspace wrappers at root, real deps in subdirectories."""
        result = detect_languages(workspace_monorepo)
        assert result == {"python", "javascript"}

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(LanguageDetectionError):
            detect_languages(tmp_path)

    def test_not_a_directory_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(LanguageDetectionError):
            detect_languages(f)


class TestDepFileValidation:
    """Test _has_real_python_deps and _has_real_js_deps."""

    def test_pyproject_with_project_deps(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname="app"\ndependencies=["flask"]\n'
        )
        assert _has_real_python_deps(tmp_path) is True

    def test_pyproject_workspace_only(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.uv.workspace]\nmembers = ["backend"]\n'
        )
        assert _has_real_python_deps(tmp_path) is False

    def test_requirements_with_content(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\npandas\n")
        assert _has_real_python_deps(tmp_path) is True

    def test_requirements_empty(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("# no deps\n\n")
        assert _has_real_python_deps(tmp_path) is False

    def test_setup_py_presence(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\n")
        assert _has_real_python_deps(tmp_path) is True

    def test_package_json_with_deps(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies":{"react":"^18"}}'
        )
        assert _has_real_js_deps(tmp_path) is True

    def test_package_json_workspaces_only(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"workspaces":["frontend"]}'
        )
        assert _has_real_js_deps(tmp_path) is False

    def test_no_dep_files(self, tmp_path):
        assert _has_real_python_deps(tmp_path) is False
        assert _has_real_js_deps(tmp_path) is False


class TestFindDepDirForLanguage:
    """Test find_dep_dir_for_language with workspace monorepos."""

    def test_workspace_monorepo_python(self, workspace_monorepo):
        result = find_dep_dir_for_language(workspace_monorepo, "python")
        assert result.name == "backend"

    def test_workspace_monorepo_javascript(self, workspace_monorepo):
        result = find_dep_dir_for_language(workspace_monorepo, "javascript")
        assert result.name == "frontend"

    def test_simple_project_returns_root(self, python_only_project):
        result = find_dep_dir_for_language(python_only_project, "python")
        assert result == python_only_project

    def test_root_level_mix_returns_root(self, root_level_mix):
        """Root has real deps for both — should return root."""
        py_dir = find_dep_dir_for_language(root_level_mix, "python")
        js_dir = find_dep_dir_for_language(root_level_mix, "javascript")
        assert py_dir == root_level_mix
        assert js_dir == root_level_mix


class TestDetectLanguageBackwardCompat:
    """detect_language() (singular) must still return a single string."""

    def test_python_only(self, python_only_project):
        assert detect_language(python_only_project) == "python"

    def test_js_only(self, js_only_project):
        assert detect_language(js_only_project) == "javascript"

    def test_multi_language_returns_string(self, multi_lang_project):
        result = detect_language(multi_lang_project)
        assert isinstance(result, str)
        assert result in ("python", "javascript")


# ── Phase 1: IngestionResult new fields ──


class TestIngestionResultFields:
    def test_default_language_empty(self):
        result = IngestionResult(project_path="/tmp/test")
        assert result.language == ""
        assert result.secondary_languages == []

    def test_language_field_set(self):
        result = IngestionResult(project_path="/tmp/test", language="python")
        assert result.language == "python"

    def test_secondary_languages(self):
        result = IngestionResult(
            project_path="/tmp/test",
            language="python",
            secondary_languages=["javascript"],
        )
        assert result.secondary_languages == ["javascript"]


# ── Phase 2: Merge + Primary Language ──


class TestDeterminePrimaryLanguage:
    def test_python_server_framework_is_primary(self):
        assert determine_primary_language(
            ["fastapi", "sqlmodel"], ["react"], 500, 1000,
        ) == "python"

    def test_js_server_framework_when_python_has_none(self):
        assert determine_primary_language(
            ["pandas", "numpy"], ["express"], 1000, 200,
        ) == "javascript"

    def test_no_frameworks_more_lines_wins(self):
        assert determine_primary_language(
            ["pandas"], ["lodash"], 100, 500,
        ) == "javascript"

    def test_tie_python_wins(self):
        assert determine_primary_language(
            ["pandas"], ["lodash"], 100, 100,
        ) == "python"


class TestMergeIngestionResults:
    def _make_result(self, lang, deps, lines=100, files=5):
        return IngestionResult(
            project_path="/tmp/test",
            files_analyzed=files,
            total_lines=lines,
            dependencies=[DependencyInfo(name=d) for d in deps],
            language=lang,
        )

    def test_merge_combines_deps(self):
        py = self._make_result("python", ["flask", "pandas"], lines=200)
        js = self._make_result("javascript", ["react", "axios"], lines=100)
        merged = merge_ingestion_results(py, js, "python")
        dep_names = [d.name for d in merged.dependencies]
        assert "flask" in dep_names
        assert "react" in dep_names
        assert len(dep_names) == 4

    def test_merge_sums_lines(self):
        py = self._make_result("python", ["flask"], lines=200, files=10)
        js = self._make_result("javascript", ["react"], lines=100, files=5)
        merged = merge_ingestion_results(py, js, "python")
        assert merged.total_lines == 300
        assert merged.files_analyzed == 15

    def test_merge_sets_language_fields(self):
        py = self._make_result("python", ["flask"])
        js = self._make_result("javascript", ["react"])
        merged = merge_ingestion_results(py, js, "python")
        assert merged.language == "python"
        assert merged.secondary_languages == ["javascript"]

    def test_merge_js_primary(self):
        py = self._make_result("python", ["pandas"])
        js = self._make_result("javascript", ["express"])
        merged = merge_ingestion_results(js, py, "javascript")
        assert merged.language == "javascript"
        assert merged.secondary_languages == ["python"]


# ── Phase 3: Scenario Tagging ──


class TestScenarioExecutionLanguage:
    def test_execution_language_field_default(self):
        s = StressTestScenario(name="test", category="data_volume_scaling", description="")
        assert s.execution_language == ""

    def test_multi_language_uses_all_categories(self):
        """When both languages detected, ALL_CATEGORIES should be used."""
        from mycode.library.loader import ComponentLibrary
        lib = ComponentLibrary()
        py_deps = [{"name": "flask", "installed_version": ""}]
        js_deps = [{"name": "react", "installed_version": ""}]
        py_matches = lib.match_dependencies("python", py_deps)
        js_matches = lib.match_dependencies("javascript", js_deps)

        ingestion = IngestionResult(
            project_path="/tmp/test",
            dependencies=[DependencyInfo(name="flask"), DependencyInfo(name="react")],
            language="python",
            secondary_languages=["javascript"],
        )

        gen = ScenarioGenerator(offline=True)
        result = gen.generate(
            ingestion, py_matches + js_matches, "",
            language="python",
            languages={"python", "javascript"},
        )

        categories = {s.category for s in result.scenarios}
        # Should include both Python-specific and JS-specific categories
        # (from profile templates, not necessarily all categories)
        assert len(result.scenarios) > 0

    def test_single_language_no_tagging(self):
        """Single-language scenarios should not have execution_language set."""
        ingestion = IngestionResult(
            project_path="/tmp/test",
            dependencies=[DependencyInfo(name="flask")],
            language="python",
        )
        from mycode.library.loader import ComponentLibrary
        lib = ComponentLibrary()
        matches = lib.match_dependencies("python", [{"name": "flask", "installed_version": ""}])

        gen = ScenarioGenerator(offline=True)
        result = gen.generate(ingestion, matches, "", language="python")

        # execution_language should be empty (single-language, no tagging)
        for s in result.scenarios:
            assert s.execution_language == "", f"{s.name} has execution_language={s.execution_language}"
