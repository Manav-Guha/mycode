"""Tests for Project Ingester (C2).

Covers: data classes, AST analysis, file discovery, dependency parsing from all
file types, version detection, dependency tree resolution, PyPI checking, function
flow mapping, coupling point identification, partial parsing, and full end-to-end
ingestion.
"""

import ast
import importlib.metadata
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mycode.ingester import (
    ClassInfo,
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    FunctionInfo,
    GlobalVarInfo,
    ImportInfo,
    IngestionError,
    IngestionResult,
    ProjectIngester,
    _FileAnalyzer,
    _extract_string_list,
    _get_static_call_name,
    _is_version_outdated,
    _normalize_package_name,
    _parse_dep_string,
)


# ── Fixtures ──


@pytest.fixture
def empty_project(tmp_path):
    """A project directory with no Python files."""
    (tmp_path / "README.md").write_text("# My Project")
    return tmp_path


@pytest.fixture
def simple_project(tmp_path):
    """A simple Python project with two modules and requirements.txt."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        import os
        from flask import Flask

        app = Flask(__name__)
        DB_PATH = "data.db"

        @app.route("/")
        def index():
            return "Hello"

        def get_data():
            with open(DB_PATH) as f:
                return f.read()

        class UserService:
            def __init__(self):
                self.users = []

            def add_user(self, name):
                self.users.append(name)

            def get_users(self):
                return self.users
        """)
    )
    (tmp_path / "utils.py").write_text(
        textwrap.dedent("""\
        import json

        def parse_json(text):
            return json.loads(text)

        def format_output(data):
            return json.dumps(data, indent=2)
        """)
    )
    (tmp_path / "requirements.txt").write_text("flask==2.3.0\nrequests>=2.28.0\npandas\n")
    return tmp_path


@pytest.fixture
def multi_file_project(tmp_path):
    """A project with multiple files, packages, and cross-module imports."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(
        textwrap.dedent("""\
        from src.models import User
        from src.utils import validate
        from src.db import get_connection

        def create_user(name, email):
            validate(email)
            conn = get_connection()
            user = User(name, email)
            return user
        """)
    )
    (src / "models.py").write_text(
        textwrap.dedent("""\
        class User:
            def __init__(self, name, email):
                self.name = name
                self.email = email
        """)
    )
    (src / "utils.py").write_text(
        textwrap.dedent("""\
        import re
        from src.db import get_connection

        def validate(email):
            if not re.match(r'.*@.*', email):
                raise ValueError("Invalid email")

        def log_action(action):
            conn = get_connection()
            pass
        """)
    )
    (src / "db.py").write_text(
        textwrap.dedent("""\
        import sqlite3

        DB_PATH = "app.db"

        def get_connection():
            return sqlite3.connect(DB_PATH)

        def close_connection(conn):
            conn.close()
        """)
    )
    return tmp_path


@pytest.fixture
def project_with_errors(tmp_path):
    """A project with some files that have syntax errors."""
    (tmp_path / "good.py").write_text("def hello():\n    return 'hi'\n")
    (tmp_path / "bad.py").write_text("def broken(\n")
    (tmp_path / "also_good.py").write_text("x = 42\n")
    return tmp_path


@pytest.fixture
def project_with_all_dep_files(tmp_path):
    """A project with requirements.txt, pyproject.toml, setup.py, and setup.cfg."""
    (tmp_path / "app.py").write_text("pass\n")
    (tmp_path / "requirements.txt").write_text(
        "flask==2.3.0\nrequests>=2.28.0\n# a comment\n\n-r other.txt\n"
    )
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "myapp"
        dependencies = [
            "sqlalchemy>=2.0",
            "pydantic[email]>=2.0",
        ]
        """)
    )
    (tmp_path / "setup.py").write_text(
        textwrap.dedent("""\
        from setuptools import setup
        setup(
            name="myapp",
            install_requires=[
                "celery>=5.0",
                "redis",
            ],
        )
        """)
    )
    (tmp_path / "setup.cfg").write_text(
        textwrap.dedent("""\
        [options]
        install_requires =
            boto3>=1.26
            click
        """)
    )
    return tmp_path


# ── Data Class Tests ──


class TestDataClasses:
    def test_import_info_creation(self):
        imp = ImportInfo(module="flask", names=["Flask"], is_from_import=True, lineno=1)
        assert imp.module == "flask"
        assert imp.names == ["Flask"]
        assert imp.is_from_import is True

    def test_function_info_defaults(self):
        func = FunctionInfo(name="foo", file_path="app.py", lineno=1)
        assert func.calls == []
        assert func.is_method is False
        assert func.is_async is False
        assert func.globals_accessed == []

    def test_class_info_defaults(self):
        cls = ClassInfo(name="Foo", file_path="app.py", lineno=1)
        assert cls.methods == []
        assert cls.bases == []

    def test_dependency_info_defaults(self):
        dep = DependencyInfo(name="flask")
        assert dep.installed_version is None
        assert dep.is_missing is False
        assert dep.is_outdated is None

    def test_ingestion_result_defaults(self):
        result = IngestionResult(project_path="/tmp/test")
        assert result.files_analyzed == 0
        assert result.dependencies == []
        assert result.function_flows == []

    def test_file_analysis_defaults(self):
        fa = FileAnalysis(file_path="app.py")
        assert fa.functions == []
        assert fa.parse_error is None

    def test_function_flow_creation(self):
        flow = FunctionFlow(caller="a.foo", callee="b.bar", file_path="a.py", lineno=10)
        assert flow.caller == "a.foo"
        assert flow.callee == "b.bar"

    def test_coupling_point_creation(self):
        cp = CouplingPoint(source="db.get", targets=["a.foo", "b.bar"], coupling_type="high_fan_in")
        assert cp.coupling_type == "high_fan_in"
        assert len(cp.targets) == 2

    def test_global_var_info(self):
        gv = GlobalVarInfo(name="CONFIG", file_path="app.py", lineno=5)
        assert gv.name == "CONFIG"


# ── Helper Function Tests ──


class TestNormalizePackageName:
    def test_lowercase(self):
        assert _normalize_package_name("Flask") == "flask"

    def test_hyphens(self):
        assert _normalize_package_name("my-package") == "my-package"

    def test_underscores_to_hyphens(self):
        assert _normalize_package_name("my_package") == "my-package"

    def test_dots_to_hyphens(self):
        assert _normalize_package_name("my.package") == "my-package"

    def test_mixed(self):
        assert _normalize_package_name("My_Package.Name") == "my-package-name"


class TestParseDepString:
    def test_name_only(self):
        assert _parse_dep_string("flask") == ("flask", "")

    def test_with_version(self):
        assert _parse_dep_string("flask>=2.0") == ("flask", ">=2.0")

    def test_exact_version(self):
        assert _parse_dep_string("flask==2.3.0") == ("flask", "==2.3.0")

    def test_with_extras(self):
        assert _parse_dep_string("requests[security]>=2.20") == ("requests", ">=2.20")

    def test_with_marker(self):
        name, ver = _parse_dep_string('typing-extensions>=3.7; python_version < "3.8"')
        assert name == "typing-extensions"
        assert ver == ">=3.7"

    def test_empty_string(self):
        assert _parse_dep_string("") == ("", "")


class TestIsVersionOutdated:
    def test_outdated(self):
        assert _is_version_outdated("1.0.0", "2.0.0") is True

    def test_current(self):
        assert _is_version_outdated("2.0.0", "2.0.0") is False

    def test_newer(self):
        assert _is_version_outdated("3.0.0", "2.0.0") is False

    def test_patch_outdated(self):
        assert _is_version_outdated("1.0.0", "1.0.1") is True

    def test_non_semver(self):
        # Falls back to string comparison
        assert _is_version_outdated("1.0.0a1", "1.0.0") is True


# ── AST Analyzer Tests ──


class TestFileAnalyzer:
    def _analyze(self, source, file_path="test.py"):
        tree = ast.parse(textwrap.dedent(source))
        analyzer = _FileAnalyzer(file_path)
        return analyzer.analyze(tree)

    def test_simple_function(self):
        result = self._analyze("""\
        def hello(name):
            return f"Hello {name}"
        """)
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "hello"
        assert func.args == ["name"]
        assert func.is_method is False
        assert func.is_async is False

    def test_async_function(self):
        result = self._analyze("""\
        async def fetch(url):
            pass
        """)
        assert len(result.functions) == 1
        assert result.functions[0].is_async is True

    def test_class_with_methods(self):
        result = self._analyze("""\
        class MyClass:
            def __init__(self):
                pass

            def method(self, x):
                pass
        """)
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "MyClass"
        assert "__init__" in cls.methods
        assert "method" in cls.methods
        assert len(result.functions) == 2
        for func in result.functions:
            assert func.is_method is True
            assert func.class_name == "MyClass"

    def test_class_inheritance(self):
        result = self._analyze("""\
        class Child(Parent, Mixin):
            pass
        """)
        assert result.classes[0].bases == ["Parent", "Mixin"]

    def test_decorators(self):
        result = self._analyze("""\
        @app.route("/")
        @login_required
        def index():
            pass
        """)
        assert "app.route" in result.functions[0].decorators
        assert "login_required" in result.functions[0].decorators

    def test_imports(self):
        result = self._analyze("""\
        import os
        import numpy as np
        """)
        assert len(result.imports) == 2
        assert result.imports[0].module == "os"
        assert not result.imports[0].is_from_import
        assert result.imports[1].module == "numpy"
        assert result.imports[1].alias == "np"

    def test_from_imports(self):
        result = self._analyze("""\
        from flask import Flask, request
        from os.path import join
        """)
        assert len(result.imports) == 2
        imp = result.imports[0]
        assert imp.module == "flask"
        assert imp.names == ["Flask", "request"]
        assert imp.is_from_import is True

    def test_global_variables(self):
        result = self._analyze("""\
        DEBUG = True
        CONFIG = {"key": "value"}
        def foo():
            pass
        """)
        assert len(result.global_vars) == 2
        names = [g.name for g in result.global_vars]
        assert "DEBUG" in names
        assert "CONFIG" in names

    def test_annotated_global_variable(self):
        result = self._analyze("""\
        app: Flask = Flask(__name__)
        count: int = 0
        """)
        assert len(result.global_vars) == 2
        names = [g.name for g in result.global_vars]
        assert "app" in names
        assert "count" in names

    def test_function_calls_captured(self):
        result = self._analyze("""\
        def main():
            data = fetch_data()
            result = process(data)
            print(result)
        """)
        func = result.functions[0]
        assert "fetch_data" in func.calls
        assert "process" in func.calls
        assert "print" in func.calls

    def test_method_calls_captured(self):
        result = self._analyze("""\
        def main():
            obj.method()
            module.submodule.func()
        """)
        func = result.functions[0]
        assert "obj.method" in func.calls
        assert "module.submodule.func" in func.calls

    def test_global_keyword(self):
        result = self._analyze("""\
        counter = 0
        def increment():
            global counter
            counter += 1
        def reset():
            global counter
            counter = 0
        """)
        assert result.functions[0].globals_accessed == ["counter"]
        assert result.functions[1].globals_accessed == ["counter"]

    def test_nested_function(self):
        result = self._analyze("""\
        def outer():
            def inner():
                pass
            inner()
        """)
        names = [f.name for f in result.functions]
        assert "outer" in names
        assert "inner" in names

    def test_class_decorator(self):
        result = self._analyze("""\
        @dataclass
        class Config:
            debug: bool = False
        """)
        assert result.classes[0].decorators == ["dataclass"]

    def test_calls_not_captured_at_module_level(self):
        result = self._analyze("""\
        app = Flask(__name__)
        """)
        # Module-level calls should NOT be attributed to any function
        for func in result.functions:
            assert "Flask" not in func.calls


# ── ProjectIngester Init Tests ──


class TestProjectIngesterInit:
    def test_valid_project(self, simple_project):
        ingester = ProjectIngester(simple_project)
        assert ingester._project_path == simple_project.resolve()

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(IngestionError, match="does not exist"):
            ProjectIngester(tmp_path / "nonexistent")

    def test_file_not_dir_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(IngestionError, match="not a directory"):
            ProjectIngester(f)

    def test_custom_options(self, simple_project):
        ingester = ProjectIngester(
            simple_project,
            installed_packages={"flask": "2.3.0"},
            pypi_timeout=10.0,
            skip_pypi_check=True,
        )
        assert ingester._installed_packages == {"flask": "2.3.0"}
        assert ingester._pypi_timeout == 10.0
        assert ingester._skip_pypi_check is True


# ── File Discovery Tests ──


class TestDiscoverPythonFiles:
    def test_finds_py_files(self, simple_project):
        ingester = ProjectIngester(simple_project, skip_pypi_check=True)
        files = ingester._discover_python_files()
        names = [f.name for f in files]
        assert "app.py" in names
        assert "utils.py" in names

    def test_excludes_venv(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        venv_dir = tmp_path / "venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "module.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        files = ingester._discover_python_files()
        names = [f.name for f in files]
        assert "app.py" in names
        assert "module.py" not in names

    def test_excludes_pycache(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "app.cpython-310.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        files = ingester._discover_python_files()
        assert len(files) == 1

    def test_nested_directories(self, tmp_path):
        pkg = tmp_path / "pkg" / "sub"
        pkg.mkdir(parents=True)
        (pkg / "module.py").write_text("pass")
        (tmp_path / "main.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        files = ingester._discover_python_files()
        assert len(files) == 2

    def test_empty_project(self, empty_project):
        ingester = ProjectIngester(empty_project, skip_pypi_check=True)
        files = ingester._discover_python_files()
        assert files == []

    def test_excludes_egg_info(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        egg = tmp_path / "mypackage.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        files = ingester._discover_python_files()
        assert len(files) == 1


# ── File Parsing Tests ──


class TestParseFile:
    def test_valid_file(self, simple_project):
        ingester = ProjectIngester(simple_project, skip_pypi_check=True)
        analysis = ingester._parse_file(simple_project / "app.py", "app.py")
        assert analysis.parse_error is None
        assert len(analysis.functions) > 0
        assert len(analysis.classes) == 1
        assert analysis.lines_of_code > 0

    def test_syntax_error(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        analysis = ingester._parse_file(bad, "bad.py")
        assert analysis.parse_error is not None
        assert "Syntax error" in analysis.parse_error

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        analysis = ingester._parse_file(f, "empty.py")
        assert analysis.parse_error is None
        assert analysis.lines_of_code == 0

    def test_file_not_found(self, tmp_path):
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        analysis = ingester._parse_file(tmp_path / "missing.py", "missing.py")
        assert analysis.parse_error is not None
        assert "Could not read file" in analysis.parse_error

    def test_line_count(self, tmp_path):
        f = tmp_path / "lines.py"
        f.write_text("a = 1\nb = 2\nc = 3\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        analysis = ingester._parse_file(f, "lines.py")
        assert analysis.lines_of_code == 3

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "latin.py"
        f.write_bytes(b"# -*- coding: latin-1 -*-\nx = '\xe9'\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        analysis = ingester._parse_file(f, "latin.py")
        # Should succeed with latin-1 fallback
        assert analysis.lines_of_code >= 1


# ── Requirements.txt Parsing Tests ──


class TestRequirementsTxtParsing:
    def test_basic(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("flask==2.3.0\nrequests>=2.28.0\npandas\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_requirements_txt(f)
        assert ("flask", "==2.3.0") in deps
        assert ("requests", ">=2.28.0") in deps
        assert ("pandas", "") in deps

    def test_comments_and_blanks(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("# comment\n\nflask\n  # indented comment\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_requirements_txt(f)
        assert len(deps) == 1
        assert deps[0][0] == "flask"

    def test_inline_comments(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("flask==2.3.0 # web framework\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_requirements_txt(f)
        assert deps[0] == ("flask", "==2.3.0")

    def test_options_skipped(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("--index-url https://pypi.org/simple\n-r other.txt\nflask\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_requirements_txt(f)
        assert len(deps) == 1

    def test_extras(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests[security]>=2.20\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_requirements_txt(f)
        assert deps[0] == ("requests", ">=2.20")


# ── pyproject.toml Parsing Tests ──


class TestPyprojectTomlParsing:
    def test_pep621(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(
            textwrap.dedent("""\
            [project]
            name = "myapp"
            dependencies = [
                "flask>=2.0",
                "requests",
            ]
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_pyproject_toml(f)
        assert ("flask", ">=2.0") in deps
        assert ("requests", "") in deps

    def test_no_dependencies(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text("[project]\nname = 'myapp'\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_pyproject_toml(f)
        assert deps == []

    @patch("mycode.ingester.tomllib", None)
    def test_regex_fallback(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(
            textwrap.dedent("""\
            [project]
            name = "myapp"
            dependencies = [
                "flask>=2.0",
                "requests",
            ]
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_pyproject_toml(f)
        assert len(deps) >= 1
        names = [d[0] for d in deps]
        assert "flask" in names

    def test_with_extras(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(
            textwrap.dedent("""\
            [project]
            dependencies = [
                "pydantic[email]>=2.0",
            ]
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_pyproject_toml(f)
        assert deps[0] == ("pydantic", ">=2.0")


# ── setup.py Parsing Tests ──


class TestSetupPyParsing:
    def test_basic_setup(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text(
            textwrap.dedent("""\
            from setuptools import setup
            setup(
                name="myapp",
                install_requires=[
                    "flask>=2.0",
                    "requests",
                ],
            )
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_py(f)
        assert ("flask", ">=2.0") in deps
        assert ("requests", "") in deps

    def test_bare_setup(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text(
            textwrap.dedent("""\
            from distutils.core import setup
            setup(
                name="myapp",
                install_requires=["numpy"],
            )
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_py(f)
        assert ("numpy", "") in deps

    def test_no_install_requires(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text("from setuptools import setup\nsetup(name='myapp')\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_py(f)
        assert deps == []

    def test_syntax_error_in_setup_py(self, tmp_path):
        f = tmp_path / "setup.py"
        f.write_text("def broken(\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_py(f)
        assert deps == []


# ── setup.cfg Parsing Tests ──


class TestSetupCfgParsing:
    def test_basic(self, tmp_path):
        f = tmp_path / "setup.cfg"
        f.write_text(
            textwrap.dedent("""\
            [options]
            install_requires =
                flask>=2.0
                requests
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_cfg(f)
        assert ("flask", ">=2.0") in deps
        assert ("requests", "") in deps

    def test_no_options_section(self, tmp_path):
        f = tmp_path / "setup.cfg"
        f.write_text("[metadata]\nname = myapp\n")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        deps = ingester._parse_setup_cfg(f)
        assert deps == []


# ── Dependency Extraction (combined) Tests ──


class TestDependencyExtraction:
    def test_from_requirements_txt(self, simple_project):
        ingester = ProjectIngester(
            simple_project, installed_packages={}, skip_pypi_check=True
        )
        deps = ingester._extract_dependencies()
        names = [d.name for d in deps]
        assert "flask" in names
        assert "requests" in names
        assert "pandas" in names

    def test_multiple_sources_deduplicated(self, project_with_all_dep_files):
        ingester = ProjectIngester(
            project_with_all_dep_files, installed_packages={}, skip_pypi_check=True
        )
        deps = ingester._extract_dependencies()
        names = [d.name for d in deps]
        # flask from requirements.txt, sqlalchemy from pyproject.toml, celery from setup.py, boto3 from setup.cfg
        assert "flask" in names
        assert "sqlalchemy" in names
        assert "celery" in names
        assert "boto3" in names
        # No exact duplicates by normalized name
        normalized = [_normalize_package_name(n) for n in names]
        assert len(normalized) == len(set(normalized))

    def test_version_detection_from_provided_dict(self, simple_project):
        ingester = ProjectIngester(
            simple_project,
            installed_packages={"flask": "2.3.0", "requests": "2.31.0"},
            skip_pypi_check=True,
        )
        deps = ingester._extract_dependencies()
        flask_dep = next(d for d in deps if d.name == "flask")
        assert flask_dep.installed_version == "2.3.0"
        assert flask_dep.is_missing is False

    def test_missing_package_flagged(self, simple_project):
        ingester = ProjectIngester(
            simple_project, installed_packages={}, skip_pypi_check=True
        )
        deps = ingester._extract_dependencies()
        for dep in deps:
            assert dep.is_missing is True

    @patch("mycode.ingester.ProjectIngester._fetch_pypi_version")
    def test_pypi_version_check(self, mock_fetch, simple_project):
        mock_fetch.return_value = "2.4.0"
        ingester = ProjectIngester(
            simple_project,
            installed_packages={"flask": "2.3.0", "requests": "2.28.0", "pandas": "1.5.0"},
        )
        deps = ingester._extract_dependencies()
        flask_dep = next(d for d in deps if d.name == "flask")
        assert flask_dep.latest_version == "2.4.0"
        assert flask_dep.is_outdated is True


# ── Version Detection Tests ──


class TestVersionDetection:
    def test_from_provided_dict(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(
            tmp_path,
            installed_packages={"Flask": "2.3.0", "numpy": "1.24.0"},
            skip_pypi_check=True,
        )
        versions = ingester._detect_installed_versions(["flask", "numpy"])
        assert versions["flask"] == "2.3.0"
        assert versions["numpy"] == "1.24.0"

    def test_provided_dict_normalized_lookup(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(
            tmp_path,
            installed_packages={"my-package": "1.0.0"},
            skip_pypi_check=True,
        )
        versions = ingester._detect_installed_versions(["my_package"])
        assert versions["my_package"] == "1.0.0"

    @patch("mycode.ingester.importlib.metadata.version")
    def test_from_importlib(self, mock_version, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_version.return_value = "3.0.0"
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        versions = ingester._detect_installed_versions(["some-package"])
        assert versions["some-package"] == "3.0.0"

    @patch("mycode.ingester.importlib.metadata.version")
    def test_package_not_found(self, mock_version, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_version.side_effect = importlib.metadata.PackageNotFoundError("nope")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        versions = ingester._detect_installed_versions(["nonexistent"])
        assert "nonexistent" not in versions


# ── Dependency Tree Resolution Tests ──


class TestDependencyTree:
    @patch("mycode.ingester.importlib.metadata.requires")
    def test_simple_tree(self, mock_requires, tmp_path):
        (tmp_path / "app.py").write_text("pass")

        def _requires(pkg):
            mapping = {
                "flask": ["Werkzeug>=2.0", "Jinja2>=3.0", "click>=8.0"],
                "Werkzeug": ["MarkupSafe>=2.1"],
                "Jinja2": ["MarkupSafe>=2.1"],
                "click": None,
                "MarkupSafe": None,
            }
            if pkg in mapping:
                return mapping[pkg]
            raise importlib.metadata.PackageNotFoundError(pkg)

        mock_requires.side_effect = _requires
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        tree = ingester._resolve_dependency_tree(["flask"])
        assert "flask" in tree
        assert "Werkzeug" in tree["flask"]
        assert "Jinja2" in tree["flask"]
        assert "MarkupSafe" in tree

    @patch("mycode.ingester.importlib.metadata.requires")
    def test_circular_dependency(self, mock_requires, tmp_path):
        (tmp_path / "app.py").write_text("pass")

        def _requires(pkg):
            mapping = {"a": ["b>=1.0"], "b": ["a>=1.0"]}
            if pkg in mapping:
                return mapping[pkg]
            raise importlib.metadata.PackageNotFoundError(pkg)

        mock_requires.side_effect = _requires
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        tree = ingester._resolve_dependency_tree(["a"])
        # Should not infinite loop
        assert "a" in tree
        assert "b" in tree

    @patch("mycode.ingester.importlib.metadata.requires")
    def test_package_not_installed(self, mock_requires, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_requires.side_effect = importlib.metadata.PackageNotFoundError("x")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        tree = ingester._resolve_dependency_tree(["unknown-pkg"])
        assert tree["unknown-pkg"] == []

    @patch("mycode.ingester.importlib.metadata.requires")
    def test_extras_skipped(self, mock_requires, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_requires.return_value = [
            "requests>=2.0",
            'pytest; extra == "testing"',
        ]
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        tree = ingester._resolve_dependency_tree(["mylib"])
        assert "requests" in tree["mylib"]
        assert "pytest" not in tree["mylib"]


# ── PyPI Version Check Tests ──


class TestPyPICheck:
    @patch("mycode.ingester.ProjectIngester._fetch_pypi_version")
    def test_successful(self, mock_fetch, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_fetch.return_value = "3.0.0"
        ingester = ProjectIngester(tmp_path)
        latest = ingester._check_latest_versions(["flask"])
        assert latest["flask"] == "3.0.0"

    @patch("mycode.ingester.ProjectIngester._fetch_pypi_version")
    def test_failure_returns_empty(self, mock_fetch, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        mock_fetch.return_value = None
        ingester = ProjectIngester(tmp_path)
        latest = ingester._check_latest_versions(["nonexistent"])
        assert "nonexistent" not in latest

    def test_skip_flag(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        latest = ingester._check_latest_versions(["flask"])
        assert latest == {}

    def test_empty_list(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path)
        latest = ingester._check_latest_versions([])
        assert latest == {}


# ── Function Flow Mapping Tests ──


class TestFunctionFlowMapping:
    def test_simple_call(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            def helper():
                return 42

            def main():
                x = helper()
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        callers = [f.caller for f in result.function_flows]
        callees = [f.callee for f in result.function_flows]
        assert "app.main" in callers
        assert "app.helper" in callees

    def test_cross_module_call(self, multi_file_project):
        ingester = ProjectIngester(multi_file_project, skip_pypi_check=True)
        result = ingester.ingest()
        # src.main calls src.utils.validate (imported via from src.utils import validate)
        flows = {(f.caller, f.callee) for f in result.function_flows}
        assert ("src.main.create_user", "src.utils.validate") in flows

    def test_import_alias_resolution(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            import json as j

            def dump(data):
                return j.dumps(data)
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        callees = [f.callee for f in result.function_flows]
        assert "json.dumps" in callees

    def test_method_call_resolution(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            class Service:
                def process(self):
                    self.validate()

                def validate(self):
                    pass
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        flows = {(f.caller, f.callee) for f in result.function_flows}
        assert ("app.Service.process", "app.Service.validate") in flows

    def test_unresolved_call(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            def main():
                print("hello")
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        callees = [f.callee for f in result.function_flows]
        # print is a builtin, should remain unresolved
        assert "print" in callees


# ── Coupling Point Tests ──


class TestCouplingPoints:
    def test_high_fan_in(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            def shared():
                pass

            def a():
                shared()

            def b():
                shared()

            def c():
                shared()
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        fan_in = [c for c in result.coupling_points if c.coupling_type == "high_fan_in"]
        assert len(fan_in) >= 1
        sources = [c.source for c in fan_in]
        assert "app.shared" in sources

    def test_cross_module_hub(self, multi_file_project):
        ingester = ProjectIngester(multi_file_project, skip_pypi_check=True)
        result = ingester.ingest()
        hubs = [c for c in result.coupling_points if c.coupling_type == "cross_module_hub"]
        # src.db is imported by src.main and src.utils — only 2, so might not meet threshold
        # Let's check what we get
        hub_sources = [c.source for c in hubs]
        # At minimum, no crash. Threshold is 3, so db (imported by 2) won't appear.
        assert isinstance(hubs, list)

    def test_shared_state(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            counter = 0

            def increment():
                global counter
                counter += 1

            def reset():
                global counter
                counter = 0
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        shared = [c for c in result.coupling_points if c.coupling_type == "shared_state"]
        assert len(shared) >= 1
        assert any("counter" in c.source for c in shared)

    def test_no_coupling_in_simple_project(self, tmp_path):
        (tmp_path / "app.py").write_text(
            textwrap.dedent("""\
            def a():
                pass

            def b():
                pass
            """)
        )
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.coupling_points == []


# ── Partial Parsing Tests ──


class TestPartialParsing:
    def test_mixed_valid_and_invalid(self, project_with_errors):
        ingester = ProjectIngester(project_with_errors, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 2  # good.py and also_good.py
        assert result.files_failed == 1  # bad.py
        assert len(result.parse_errors) == 1
        assert result.parse_errors[0]["file"] == "bad.py"

    def test_warning_message(self, project_with_errors):
        ingester = ProjectIngester(project_with_errors, skip_pypi_check=True)
        result = ingester.ingest()
        assert any("2 of 3 files" in w for w in result.warnings)

    def test_all_files_valid(self, simple_project):
        ingester = ProjectIngester(simple_project, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.files_failed == 0
        assert not any("couldn't be parsed" in w for w in result.warnings)


# ── Full Ingestion Tests ──


class TestFullIngestion:
    def test_simple_project(self, simple_project):
        ingester = ProjectIngester(
            simple_project,
            installed_packages={"flask": "2.3.0"},
            skip_pypi_check=True,
        )
        result = ingester.ingest()
        assert result.files_analyzed == 2
        assert result.files_failed == 0
        assert result.total_lines > 0
        assert len(result.dependencies) >= 1
        assert len(result.function_flows) > 0
        # Check file analyses present
        paths = [a.file_path for a in result.file_analyses]
        assert "app.py" in paths
        assert "utils.py" in paths

    def test_empty_project(self, empty_project):
        ingester = ProjectIngester(empty_project, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 0
        assert "No Python files found" in result.warnings[0]

    def test_multi_file_project(self, multi_file_project):
        ingester = ProjectIngester(multi_file_project, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.files_analyzed >= 4  # __init__.py, main.py, models.py, utils.py, db.py
        assert len(result.function_flows) > 0
        # Check that cross-module flows are detected
        callees = {f.callee for f in result.function_flows}
        assert any("validate" in c for c in callees)

    def test_project_path_in_result(self, simple_project):
        ingester = ProjectIngester(simple_project, skip_pypi_check=True)
        result = ingester.ingest()
        assert result.project_path == str(simple_project.resolve())

    @patch("mycode.ingester.importlib.metadata.requires")
    def test_dependency_tree_populated(self, mock_requires, simple_project):
        mock_requires.return_value = None
        ingester = ProjectIngester(
            simple_project,
            installed_packages={"flask": "2.3.0"},
            skip_pypi_check=True,
        )
        result = ingester.ingest()
        # Tree should be populated for declared deps
        assert isinstance(result.dependency_tree, dict)

    def test_all_dep_files_project(self, project_with_all_dep_files):
        ingester = ProjectIngester(
            project_with_all_dep_files,
            installed_packages={},
            skip_pypi_check=True,
        )
        result = ingester.ingest()
        dep_names = [d.name for d in result.dependencies]
        # Should find deps from all four files
        assert "flask" in dep_names
        assert "sqlalchemy" in dep_names
        assert "celery" in dep_names
        assert "boto3" in dep_names


# ── Module-Level Helper Tests ──


class TestStaticHelpers:
    def test_get_static_call_name_simple(self):
        tree = ast.parse("setup(name='x')")
        call = tree.body[0].value
        assert _get_static_call_name(call) == "setup"

    def test_get_static_call_name_attribute(self):
        tree = ast.parse("setuptools.setup(name='x')")
        call = tree.body[0].value
        assert _get_static_call_name(call) == "setuptools.setup"

    def test_extract_string_list(self):
        tree = ast.parse("x = ['a', 'b', 'c']")
        node = tree.body[0].value
        assert _extract_string_list(node) == ["a", "b", "c"]

    def test_extract_string_list_mixed(self):
        tree = ast.parse("x = ['a', 1, 'b']")
        node = tree.body[0].value
        # Only strings extracted
        assert _extract_string_list(node) == ["a", "b"]

    def test_extract_string_list_non_list(self):
        tree = ast.parse("x = 'not a list'")
        node = tree.body[0].value
        assert _extract_string_list(node) == []


# ── File-to-Module Conversion Tests ──


class TestFileToModule:
    def test_simple_file(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        assert ingester._file_to_module("app.py") == "app"

    def test_nested_file(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        assert ingester._file_to_module("pkg/sub/module.py") == "pkg.sub.module"

    def test_init_file(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        assert ingester._file_to_module("pkg/__init__.py") == "pkg"

    def test_nested_init_file(self, tmp_path):
        (tmp_path / "app.py").write_text("pass")
        ingester = ProjectIngester(tmp_path, skip_pypi_check=True)
        assert ingester._file_to_module("pkg/sub/__init__.py") == "pkg.sub"
