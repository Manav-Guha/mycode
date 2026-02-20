"""Tests for Component Library (C4) — profile loading, matching, and validation."""

import json
import os
from pathlib import Path

import pytest

from mycode.library import (
    ComponentLibrary,
    DependencyProfile,
    LibraryError,
    ProfileMatch,
    ProfileNotFoundError,
)
from mycode.library.loader import _check_version, _normalize_dep_name


# ── Fixtures ──


@pytest.fixture
def profiles_dir(tmp_path):
    """Create a temporary profiles directory with sample profiles."""
    python_dir = tmp_path / "python"
    python_dir.mkdir()
    js_dir = tmp_path / "javascript"
    js_dir.mkdir()

    # Minimal valid profile
    flask_profile = {
        "identity": {
            "name": "flask",
            "pypi_name": "Flask",
            "category": "web_framework",
            "description": "Lightweight WSGI web framework",
            "current_stable_version": "3.1.0",
            "min_supported_version": "2.0.0",
            "version_notes": {
                "3.0.0": "Dropped Python 3.7 support",
                "2.0.0": "Major breaking changes",
            },
        },
        "scaling_characteristics": {
            "description": "Single-threaded WSGI",
            "concurrency_model": "synchronous_wsgi",
            "bottlenecks": [],
            "scaling_limits": [],
        },
        "memory_behavior": {
            "baseline_footprint_mb": 25,
            "growth_pattern": "Lean framework",
            "known_leaks": [],
            "gc_behavior": "Standard CPython GC",
        },
        "known_failure_modes": [
            {
                "name": "dev_server_in_production",
                "description": "Using app.run() in production",
                "trigger_conditions": "Deploying with app.run()",
                "severity": "critical",
                "versions_affected": "all",
            }
        ],
        "edge_case_sensitivities": [
            {
                "name": "trailing_slash_redirects",
                "description": "Auto-redirects /path to /path/",
                "test_approach": "Send requests with and without trailing slashes",
            }
        ],
        "interaction_patterns": {
            "commonly_used_with": ["sqlalchemy", "requests"],
            "known_conflicts": [],
            "dependency_chain_risks": [],
        },
        "stress_test_templates": [
            {
                "name": "concurrent_request_load",
                "category": "concurrent_execution",
                "description": "Increase concurrent requests",
                "parameters": {"max_concurrent": 100},
                "expected_behavior": "Response time degrades",
                "failure_indicators": ["timeout"],
            }
        ],
    }
    (python_dir / "flask.json").write_text(json.dumps(flask_profile), encoding="utf-8")

    # Second profile for multi-profile tests
    pandas_profile = dict(flask_profile)
    pandas_profile["identity"] = {
        "name": "pandas",
        "pypi_name": "pandas",
        "category": "data_processing",
        "description": "Data analysis library",
        "current_stable_version": "2.2.3",
        "min_supported_version": "2.0.0",
        "version_notes": {
            "2.0.0": "Major breaking changes: Copy-on-Write opt-in",
        },
    }
    (python_dir / "pandas.json").write_text(json.dumps(pandas_profile), encoding="utf-8")

    # Stdlib profile (no pypi_name)
    sqlite_profile = dict(flask_profile)
    sqlite_profile["identity"] = {
        "name": "sqlite3",
        "pypi_name": None,
        "category": "database",
        "description": "Python standard library SQLite interface",
        "current_stable_version": "stdlib",
        "min_supported_version": "stdlib",
        "version_notes": {},
    }
    (python_dir / "sqlite3.json").write_text(json.dumps(sqlite_profile), encoding="utf-8")

    # JavaScript profile
    react_profile = dict(flask_profile)
    react_profile["identity"] = {
        "name": "react",
        "pypi_name": None,
        "category": "ui_framework",
        "description": "UI component library",
        "current_stable_version": "19.0.0",
        "min_supported_version": "18.0.0",
        "version_notes": {},
    }
    (js_dir / "react.json").write_text(json.dumps(react_profile), encoding="utf-8")

    return tmp_path


@pytest.fixture
def library(profiles_dir):
    """Create a ComponentLibrary pointing at the temp profiles directory."""
    return ComponentLibrary(profiles_root=profiles_dir)


@pytest.fixture
def invalid_profiles_dir(tmp_path):
    """Create a profiles directory with various invalid files."""
    python_dir = tmp_path / "python"
    python_dir.mkdir()

    # Invalid JSON
    (python_dir / "broken.json").write_text("{not valid json", encoding="utf-8")

    # Valid JSON but missing required fields
    (python_dir / "incomplete.json").write_text(
        json.dumps({"identity": {"name": "incomplete"}}),
        encoding="utf-8",
    )

    # Valid JSON, has all top-level keys but identity missing required sub-keys
    minimal = {
        "identity": {"name": "noversion"},
        "scaling_characteristics": {},
        "memory_behavior": {},
        "known_failure_modes": [],
        "edge_case_sensitivities": [],
        "interaction_patterns": {},
        "stress_test_templates": [],
    }
    (python_dir / "noversion.json").write_text(json.dumps(minimal), encoding="utf-8")

    return tmp_path


# ── Tests: Profile Loading ──


class TestGetProfile:
    """Tests for loading individual profiles."""

    def test_load_valid_profile(self, library):
        profile = library.get_profile("python", "flask")
        assert isinstance(profile, DependencyProfile)
        assert profile.name == "flask"
        assert profile.pypi_name == "Flask"
        assert profile.category == "web_framework"
        assert profile.current_stable_version == "3.1.0"

    def test_load_profile_sections(self, library):
        profile = library.get_profile("python", "flask")
        assert isinstance(profile.scaling_characteristics, dict)
        assert isinstance(profile.memory_behavior, dict)
        assert isinstance(profile.known_failure_modes, list)
        assert isinstance(profile.edge_case_sensitivities, list)
        assert isinstance(profile.interaction_patterns, dict)
        assert isinstance(profile.stress_test_templates, list)
        assert isinstance(profile.raw, dict)

    def test_profile_content(self, library):
        profile = library.get_profile("python", "flask")
        assert len(profile.known_failure_modes) == 1
        assert profile.known_failure_modes[0]["name"] == "dev_server_in_production"
        assert len(profile.stress_test_templates) == 1
        assert profile.stress_test_templates[0]["category"] == "concurrent_execution"

    def test_load_stdlib_profile(self, library):
        profile = library.get_profile("python", "sqlite3")
        assert profile.name == "sqlite3"
        assert profile.pypi_name is None
        assert profile.current_stable_version == "stdlib"

    def test_load_javascript_profile(self, library):
        profile = library.get_profile("javascript", "react")
        assert profile.name == "react"
        assert profile.category == "ui_framework"

    def test_case_insensitive_language(self, library):
        profile = library.get_profile("Python", "flask")
        assert profile.name == "flask"

    def test_case_insensitive_name(self, library):
        profile = library.get_profile("python", "Flask")
        assert profile.name == "flask"

    def test_profile_caching(self, library):
        profile1 = library.get_profile("python", "flask")
        profile2 = library.get_profile("python", "flask")
        assert profile1 is profile2  # Same object, not just equal

    def test_profile_not_found(self, library):
        with pytest.raises(ProfileNotFoundError, match="nonexistent"):
            library.get_profile("python", "nonexistent")

    def test_profile_not_found_wrong_language(self, library):
        with pytest.raises(ProfileNotFoundError):
            library.get_profile("python", "react")  # react is in javascript/

    def test_raw_dict_preserved(self, library):
        profile = library.get_profile("python", "flask")
        assert profile.raw["identity"]["name"] == "flask"
        assert "scaling_characteristics" in profile.raw


class TestInvalidProfiles:
    """Tests for handling invalid profile files."""

    def test_invalid_json(self, invalid_profiles_dir):
        lib = ComponentLibrary(profiles_root=invalid_profiles_dir)
        with pytest.raises(LibraryError, match="Invalid JSON"):
            lib.get_profile("python", "broken")

    def test_missing_required_fields(self, invalid_profiles_dir):
        lib = ComponentLibrary(profiles_root=invalid_profiles_dir)
        with pytest.raises(LibraryError, match="missing required fields"):
            lib.get_profile("python", "incomplete")

    def test_missing_identity_subfields(self, invalid_profiles_dir):
        lib = ComponentLibrary(profiles_root=invalid_profiles_dir)
        with pytest.raises(LibraryError, match="identity missing required field"):
            lib.get_profile("python", "noversion")


# ── Tests: Listing Profiles ──


class TestListProfiles:
    """Tests for listing available profiles."""

    def test_list_python_profiles(self, library):
        names = library.list_profiles("python")
        assert names == ["flask", "pandas", "sqlite3"]

    def test_list_javascript_profiles(self, library):
        names = library.list_profiles("javascript")
        assert names == ["react"]

    def test_list_empty_language(self, library):
        names = library.list_profiles("rust")
        assert names == []

    def test_list_case_insensitive(self, library):
        names = library.list_profiles("Python")
        assert "flask" in names


# ── Tests: Load All ──


class TestLoadAll:
    """Tests for loading all profiles at once."""

    def test_load_all_python(self, library):
        profiles = library.load_all("python")
        assert len(profiles) == 3
        assert "flask" in profiles
        assert "pandas" in profiles
        assert "sqlite3" in profiles

    def test_load_all_javascript(self, library):
        profiles = library.load_all("javascript")
        assert len(profiles) == 1
        assert "react" in profiles

    def test_load_all_empty(self, library):
        profiles = library.load_all("rust")
        assert profiles == {}

    def test_load_all_skips_invalid(self, invalid_profiles_dir):
        lib = ComponentLibrary(profiles_root=invalid_profiles_dir)
        profiles = lib.load_all("python")
        # All three files are invalid, so none should load
        assert len(profiles) == 0


# ── Tests: Dependency Matching ──


class TestMatchDependencies:
    """Tests for matching project dependencies against profiles."""

    def test_match_recognized_dependency(self, library):
        deps = [{"name": "flask", "installed_version": "3.1.0"}]
        matches = library.match_dependencies("python", deps)
        assert len(matches) == 1
        assert matches[0].profile is not None
        assert matches[0].profile.name == "flask"
        assert matches[0].version_match is True

    def test_match_unrecognized_dependency(self, library):
        deps = [{"name": "obscure-lib"}]
        matches = library.match_dependencies("python", deps)
        assert len(matches) == 1
        assert matches[0].profile is None
        assert matches[0].dependency_name == "obscure-lib"

    def test_match_mixed_dependencies(self, library):
        deps = [
            {"name": "flask", "installed_version": "3.1.0"},
            {"name": "pandas", "installed_version": "2.2.3"},
            {"name": "unknown-pkg"},
        ]
        matches = library.match_dependencies("python", deps)
        assert len(matches) == 3

        recognized = library.get_recognized(matches)
        unrecognized = library.get_unrecognized(matches)
        assert len(recognized) == 2
        assert len(unrecognized) == 1
        assert unrecognized[0] == "unknown-pkg"

    def test_match_with_version_behind(self, library):
        deps = [{"name": "flask", "installed_version": "2.0.0"}]
        matches = library.match_dependencies("python", deps)
        assert matches[0].version_match is False
        assert "behind" in matches[0].version_notes.lower()

    def test_match_without_version(self, library):
        deps = [{"name": "flask"}]
        matches = library.match_dependencies("python", deps)
        assert matches[0].profile is not None
        assert matches[0].version_match is None
        assert matches[0].installed_version is None

    def test_match_stdlib_version(self, library):
        deps = [{"name": "sqlite3", "installed_version": "3.12.0"}]
        matches = library.match_dependencies("python", deps)
        assert matches[0].profile is not None
        assert matches[0].version_match is None  # Stdlib — no version comparison
        assert "standard library" in matches[0].version_notes.lower()

    def test_match_empty_dependencies(self, library):
        matches = library.match_dependencies("python", [])
        assert matches == []

    def test_match_skips_empty_names(self, library):
        deps = [{"name": ""}, {"name": "flask"}]
        matches = library.match_dependencies("python", deps)
        assert len(matches) == 1
        assert matches[0].dependency_name == "flask"

    def test_match_case_insensitive(self, library):
        deps = [{"name": "Flask"}]
        matches = library.match_dependencies("python", deps)
        assert len(matches) == 1
        assert matches[0].profile is not None

    def test_get_unrecognized(self, library):
        deps = [
            {"name": "flask"},
            {"name": "unknown1"},
            {"name": "unknown2"},
        ]
        matches = library.match_dependencies("python", deps)
        unrecognized = library.get_unrecognized(matches)
        assert unrecognized == ["unknown1", "unknown2"]

    def test_get_recognized(self, library):
        deps = [
            {"name": "flask"},
            {"name": "unknown"},
            {"name": "pandas"},
        ]
        matches = library.match_dependencies("python", deps)
        recognized = library.get_recognized(matches)
        assert len(recognized) == 2
        assert {m.profile.name for m in recognized} == {"flask", "pandas"}


# ── Tests: Name Normalization ──


class TestNormalizeName:
    """Tests for dependency name normalization."""

    # -- Python aliases --

    def test_lowercase(self):
        assert _normalize_dep_name("Flask", "python") == "flask"

    def test_hyphen_to_underscore(self):
        assert _normalize_dep_name("supabase-py", "python") == "supabase"

    def test_dot_to_underscore(self):
        assert _normalize_dep_name("some.pkg", "python") == "some_pkg"

    def test_known_alias_supabase(self):
        assert _normalize_dep_name("supabase_py", "python") == "supabase"

    def test_known_alias_llama_index(self):
        assert _normalize_dep_name("llama-index", "python") == "llamaindex"
        assert _normalize_dep_name("llama_index_core", "python") == "llamaindex"

    def test_known_alias_langchain(self):
        assert _normalize_dep_name("langchain-core", "python") == "langchain"
        assert _normalize_dep_name("langchain-community", "python") == "langchain"

    def test_known_alias_os_pathlib(self):
        assert _normalize_dep_name("os", "python") == "os_pathlib"
        assert _normalize_dep_name("pathlib", "python") == "os_pathlib"

    def test_known_alias_pydantic_core(self):
        assert _normalize_dep_name("pydantic-core", "python") == "pydantic"

    def test_unknown_passthrough(self):
        assert _normalize_dep_name("my-custom-lib", "python") == "my_custom_lib"

    # -- JavaScript aliases --

    def test_js_next_to_nextjs(self):
        assert _normalize_dep_name("next", "javascript") == "nextjs"

    def test_js_three_to_threejs(self):
        assert _normalize_dep_name("three", "javascript") == "threejs"

    def test_js_openai_to_openai_node(self):
        assert _normalize_dep_name("openai", "javascript") == "openai_node"

    def test_js_anthropic_scoped(self):
        assert _normalize_dep_name("@anthropic-ai/sdk", "javascript") == "anthropic_node"

    def test_js_supabase_scoped(self):
        assert _normalize_dep_name("@supabase/supabase-js", "javascript") == "supabase_js"

    def test_js_prisma_scoped(self):
        assert _normalize_dep_name("@prisma/client", "javascript") == "prisma"

    def test_js_langchain_scoped(self):
        assert _normalize_dep_name("@langchain/core", "javascript") == "langchainjs"
        assert _normalize_dep_name("@langchain/community", "javascript") == "langchainjs"

    def test_js_socket_io(self):
        assert _normalize_dep_name("socket.io", "javascript") == "socketio"
        assert _normalize_dep_name("socket.io-client", "javascript") == "socketio"

    def test_js_node_core_modules(self):
        for mod in ("fs", "path", "http", "https", "crypto", "child_process"):
            assert _normalize_dep_name(mod, "javascript") == "node_core"

    def test_js_react_dom_alias(self):
        assert _normalize_dep_name("react-dom", "javascript") == "react"

    def test_js_node_fetch_to_axios(self):
        assert _normalize_dep_name("node-fetch", "javascript") == "axios"

    def test_js_mongodb_to_mongoose(self):
        assert _normalize_dep_name("mongodb", "javascript") == "mongoose"

    def test_js_default_language_is_python(self):
        """Without explicit language, should use Python aliases."""
        assert _normalize_dep_name("os") == "os_pathlib"


# ── Tests: Version Checking ──


class TestCheckVersion:
    """Tests for version comparison logic."""

    def test_exact_match(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, "3.1.0")
        assert match is True

    def test_major_behind(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, "2.3.0")
        assert match is False
        assert "Major version behind" in note

    def test_minor_behind(self, library):
        profile = library.get_profile("python", "pandas")
        match, note = _check_version(profile, "2.1.0")
        assert match is False
        assert "Minor version behind" in note

    def test_no_installed_version(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, None)
        assert match is None
        assert note is None

    def test_stdlib_version(self, library):
        profile = library.get_profile("python", "sqlite3")
        match, note = _check_version(profile, "3.12.0")
        assert match is None
        assert "Standard library" in note

    def test_version_with_notes(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, "3.0.0")
        assert match is False  # 3.0.0 is behind 3.1.0 (minor version behind)
        assert "Dropped Python 3.7" in note

    def test_version_ahead_of_stable(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, "4.0.0")
        assert match is True  # Ahead is fine

    def test_malformed_version(self, library):
        profile = library.get_profile("python", "flask")
        match, note = _check_version(profile, "unknown")
        assert match is None


# ── Tests: Default Profiles Root ──


class TestDefaultProfilesRoot:
    """Tests for the default profiles root path resolution."""

    def test_default_root_points_to_profiles_dir(self):
        lib = ComponentLibrary()
        expected = Path(__file__).resolve().parent.parent / "profiles"
        assert lib.profiles_root == expected

    def test_custom_root(self, profiles_dir):
        lib = ComponentLibrary(profiles_root=profiles_dir)
        assert lib.profiles_root == profiles_dir


# ── Tests: Real Profiles ──


class TestRealProfiles:
    """Integration tests that load the actual shipped profiles."""

    REAL_PROFILES_ROOT = Path(__file__).resolve().parent.parent / "profiles"

    @pytest.fixture
    def real_library(self):
        return ComponentLibrary(profiles_root=self.REAL_PROFILES_ROOT)

    def test_all_python_profiles_load(self, real_library):
        """Every JSON file in profiles/python/ should load without errors."""
        profiles = real_library.load_all("python")
        assert len(profiles) >= 18  # All 18 Python profiles

    def test_all_profiles_have_required_sections(self, real_library):
        """Every loaded profile must have all required sections populated."""
        for name, profile in real_library.load_all("python").items():
            assert profile.identity, f"{name}: empty identity"
            assert profile.name, f"{name}: empty name"
            assert profile.category, f"{name}: empty category"
            assert profile.current_stable_version, f"{name}: empty version"
            assert isinstance(profile.scaling_characteristics, dict), f"{name}: scaling not dict"
            assert isinstance(profile.memory_behavior, dict), f"{name}: memory not dict"
            assert isinstance(profile.known_failure_modes, list), f"{name}: failure modes not list"
            assert len(profile.known_failure_modes) > 0, f"{name}: no failure modes"
            assert isinstance(profile.edge_case_sensitivities, list), f"{name}: edge cases not list"
            assert len(profile.edge_case_sensitivities) > 0, f"{name}: no edge cases"
            assert isinstance(profile.interaction_patterns, dict), f"{name}: interactions not dict"
            assert isinstance(profile.stress_test_templates, list), f"{name}: templates not list"
            assert len(profile.stress_test_templates) > 0, f"{name}: no stress test templates"

    def test_stress_templates_have_required_fields(self, real_library):
        """Every stress test template must have name, category, description, parameters."""
        required = {"name", "category", "description", "parameters", "expected_behavior", "failure_indicators"}
        for name, profile in real_library.load_all("python").items():
            for i, template in enumerate(profile.stress_test_templates):
                missing = required - set(template.keys())
                assert not missing, (
                    f"{name} template[{i}] ({template.get('name', '?')}) "
                    f"missing: {missing}"
                )

    def test_failure_modes_have_required_fields(self, real_library):
        """Every failure mode must have name, description, severity."""
        required = {"name", "description", "trigger_conditions", "severity", "versions_affected"}
        for name, profile in real_library.load_all("python").items():
            for i, mode in enumerate(profile.known_failure_modes):
                missing = required - set(mode.keys())
                assert not missing, (
                    f"{name} failure_mode[{i}] ({mode.get('name', '?')}) "
                    f"missing: {missing}"
                )

    def test_expected_profiles_present(self, real_library):
        """All 18 specified Python profiles should be present."""
        expected = {
            "flask", "fastapi", "streamlit", "gradio",
            "pandas", "numpy",
            "sqlite3", "sqlalchemy", "supabase",
            "langchain", "llamaindex", "chromadb",
            "openai", "anthropic",
            "requests", "httpx",
            "pydantic", "os_pathlib",
        }
        available = set(real_library.list_profiles("python"))
        missing = expected - available
        assert not missing, f"Missing profiles: {missing}"

    def test_match_against_sample_project(self, real_library):
        """Simulate matching a typical vibe-coded project's dependencies."""
        project_deps = [
            {"name": "flask", "installed_version": "3.0.0"},
            {"name": "pandas", "installed_version": "2.2.3"},
            {"name": "requests", "installed_version": "2.32.3"},
            {"name": "python-dotenv", "installed_version": "1.0.0"},
            {"name": "gunicorn", "installed_version": "21.2.0"},
        ]
        matches = real_library.match_dependencies("python", project_deps)
        assert len(matches) == 5

        recognized = real_library.get_recognized(matches)
        unrecognized = real_library.get_unrecognized(matches)

        assert len(recognized) == 3  # flask, pandas, requests
        assert set(unrecognized) == {"python-dotenv", "gunicorn"}

    def test_profile_categories_reasonable(self, real_library):
        """Each profile should have a meaningful category."""
        valid_categories = {
            "web_framework", "data_processing", "numerical_computing",
            "database", "backend_service", "http_client",
            "ai_framework", "filesystem",
        }
        for name, profile in real_library.load_all("python").items():
            assert profile.category in valid_categories, (
                f"{name} has unexpected category: {profile.category}"
            )

    def test_alias_resolution_llama_index(self, real_library):
        """llama-index and llama_index should resolve to llamaindex profile."""
        deps = [{"name": "llama-index"}, {"name": "llama_index_core"}]
        matches = real_library.match_dependencies("python", deps)
        for m in matches:
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == "llamaindex"

    def test_alias_resolution_langchain_variants(self, real_library):
        """langchain-core and langchain-community should resolve to langchain profile."""
        deps = [{"name": "langchain-core"}, {"name": "langchain-community"}]
        matches = real_library.match_dependencies("python", deps)
        for m in matches:
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == "langchain"

    def test_alias_resolution_os_pathlib(self, real_library):
        """os and pathlib should resolve to os_pathlib profile."""
        deps = [{"name": "os"}, {"name": "pathlib"}]
        matches = real_library.match_dependencies("python", deps)
        for m in matches:
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == "os_pathlib"


# ── Tests: Real JavaScript Profiles ──


class TestRealJavaScriptProfiles:
    """Integration tests that load the actual shipped JavaScript profiles."""

    REAL_PROFILES_ROOT = Path(__file__).resolve().parent.parent / "profiles"

    @pytest.fixture
    def real_library(self):
        return ComponentLibrary(profiles_root=self.REAL_PROFILES_ROOT)

    def test_all_js_profiles_load(self, real_library):
        """Every JSON file in profiles/javascript/ should load without errors."""
        profiles = real_library.load_all("javascript")
        assert len(profiles) >= 18  # All 18 JavaScript profiles

    def test_all_js_profiles_have_required_sections(self, real_library):
        """Every loaded JS profile must have all required sections populated."""
        for name, profile in real_library.load_all("javascript").items():
            assert profile.identity, f"{name}: empty identity"
            assert profile.name, f"{name}: empty name"
            assert profile.category, f"{name}: empty category"
            assert profile.current_stable_version, f"{name}: empty version"
            assert isinstance(profile.scaling_characteristics, dict), f"{name}: scaling not dict"
            assert isinstance(profile.memory_behavior, dict), f"{name}: memory not dict"
            assert isinstance(profile.known_failure_modes, list), f"{name}: failure modes not list"
            assert len(profile.known_failure_modes) > 0, f"{name}: no failure modes"
            assert isinstance(profile.edge_case_sensitivities, list), f"{name}: edge cases not list"
            assert len(profile.edge_case_sensitivities) > 0, f"{name}: no edge cases"
            assert isinstance(profile.interaction_patterns, dict), f"{name}: interactions not dict"
            assert isinstance(profile.stress_test_templates, list), f"{name}: templates not list"
            assert len(profile.stress_test_templates) > 0, f"{name}: no stress test templates"

    def test_js_stress_templates_have_required_fields(self, real_library):
        """Every stress test template must have name, category, description, parameters."""
        required = {"name", "category", "description", "parameters", "expected_behavior", "failure_indicators"}
        for name, profile in real_library.load_all("javascript").items():
            for i, template in enumerate(profile.stress_test_templates):
                missing = required - set(template.keys())
                assert not missing, (
                    f"{name} template[{i}] ({template.get('name', '?')}) "
                    f"missing: {missing}"
                )

    def test_js_failure_modes_have_required_fields(self, real_library):
        """Every failure mode must have name, description, severity."""
        required = {"name", "description", "trigger_conditions", "severity", "versions_affected"}
        for name, profile in real_library.load_all("javascript").items():
            for i, mode in enumerate(profile.known_failure_modes):
                missing = required - set(mode.keys())
                assert not missing, (
                    f"{name} failure_mode[{i}] ({mode.get('name', '?')}) "
                    f"missing: {missing}"
                )

    def test_expected_js_profiles_present(self, real_library):
        """All 18 specified JavaScript profiles should be present."""
        expected = {
            "react", "nextjs", "express", "node_core",
            "tailwindcss", "threejs", "svelte",
            "openai_node", "anthropic_node", "langchainjs",
            "supabase_js", "prisma", "axios",
            "mongoose", "stripe", "dotenv", "zod", "socketio",
        }
        available = set(real_library.list_profiles("javascript"))
        missing = expected - available
        assert not missing, f"Missing JS profiles: {missing}"

    def test_js_profiles_have_npm_name(self, real_library):
        """All JS profiles should have npm_name in identity."""
        for name, profile in real_library.load_all("javascript").items():
            if name != "node_core":  # node_core is stdlib, no npm_name
                assert profile.npm_name is not None, f"{name}: missing npm_name"

    def test_js_profile_categories_reasonable(self, real_library):
        """Each JS profile should have a meaningful category."""
        valid_categories = {
            "ui_framework", "web_framework", "server_framework",
            "runtime_core", "runtime", "css_framework", "3d_graphics", "graphics",
            "ai_sdk", "ai_framework", "backend_service",
            "orm", "http_client", "database",
            "payment", "configuration", "validation", "realtime",
        }
        for name, profile in real_library.load_all("javascript").items():
            assert profile.category in valid_categories, (
                f"{name} has unexpected category: {profile.category}"
            )

    def test_js_match_against_sample_project(self, real_library):
        """Simulate matching a typical Next.js project's dependencies."""
        project_deps = [
            {"name": "next", "installed_version": "15.1.0"},
            {"name": "react", "installed_version": "19.0.0"},
            {"name": "@prisma/client", "installed_version": "6.2.0"},
            {"name": "zod", "installed_version": "3.24.1"},
            {"name": "stripe", "installed_version": "17.5.0"},
            {"name": "tailwindcss", "installed_version": "4.0.0"},
            {"name": "some-unknown-lib"},
        ]
        matches = real_library.match_dependencies("javascript", project_deps)
        assert len(matches) == 7

        recognized = real_library.get_recognized(matches)
        unrecognized = real_library.get_unrecognized(matches)

        assert len(recognized) == 6  # next, react, prisma, zod, stripe, tailwindcss
        assert unrecognized == ["some-unknown-lib"]

    def test_js_alias_resolution_scoped_packages(self, real_library):
        """Scoped npm packages should resolve to correct profiles."""
        deps = [
            {"name": "@anthropic-ai/sdk"},
            {"name": "@supabase/supabase-js"},
            {"name": "@prisma/client"},
            {"name": "@langchain/core"},
        ]
        matches = real_library.match_dependencies("javascript", deps)
        expected_names = ["anthropic_node", "supabase_js", "prisma", "langchainjs"]
        for m, expected in zip(matches, expected_names):
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == expected, (
                f"{m.dependency_name} should resolve to {expected}, got {m.profile.name}"
            )

    def test_js_alias_resolution_node_core_modules(self, real_library):
        """Node.js core modules should resolve to node_core profile."""
        deps = [{"name": "fs"}, {"name": "path"}, {"name": "http"}, {"name": "crypto"}]
        matches = real_library.match_dependencies("javascript", deps)
        for m in matches:
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == "node_core"

    def test_js_alias_resolution_socket_io(self, real_library):
        """socket.io variants should resolve to socketio profile."""
        deps = [{"name": "socket.io"}, {"name": "socket.io-client"}]
        matches = real_library.match_dependencies("javascript", deps)
        for m in matches:
            assert m.profile is not None, f"{m.dependency_name} should resolve"
            assert m.profile.name == "socketio"

    def test_js_alias_next_and_three(self, real_library):
        """Short npm names should resolve to full profile names."""
        deps = [{"name": "next"}, {"name": "three"}]
        matches = real_library.match_dependencies("javascript", deps)
        assert matches[0].profile is not None
        assert matches[0].profile.name == "nextjs"
        assert matches[1].profile is not None
        assert matches[1].profile.name == "threejs"
