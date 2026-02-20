"""Tests for JavaScript/Node.js Project Ingester (C3).

Covers: comment stripping, import extraction (ES6 + CommonJS), function
extraction (declarations, arrows, expressions, methods), class extraction,
file discovery, package.json parsing, dependency tree resolution, npm version
checking, function flow mapping, coupling points, partial parsing, and full
end-to-end ingestion.
"""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mycode.js_ingester import (
    JS_EXTENSIONS,
    JsProjectIngester,
    _JsFileAnalyzer,
    _compute_brace_depths,
    _find_brace_range,
    _parse_js_args,
    _strip_js_comments,
)
from mycode.ingester import (
    CouplingPoint,
    DependencyInfo,
    FileAnalysis,
    FunctionFlow,
    FunctionInfo,
    IngestionError,
    IngestionResult,
)


# ── Fixtures ──


@pytest.fixture
def empty_project(tmp_path):
    (tmp_path / "README.md").write_text("# My Project")
    return tmp_path


@pytest.fixture
def simple_js_project(tmp_path):
    """A simple Node.js project with two modules and package.json."""
    (tmp_path / "index.js").write_text(
        textwrap.dedent("""\
        const express = require('express');
        const { helper } = require('./utils');

        const app = express();
        const PORT = 3000;

        function startServer() {
            app.listen(PORT);
        }

        app.get('/', (req, res) => {
            const data = helper();
            res.json(data);
        });

        module.exports = { startServer };
        """)
    )
    (tmp_path / "utils.js").write_text(
        textwrap.dedent("""\
        function helper() {
            return { status: 'ok' };
        }

        function formatData(data) {
            return JSON.stringify(data);
        }

        module.exports = { helper, formatData };
        """)
    )
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "test-project",
                "version": "1.0.0",
                "dependencies": {"express": "^4.18.0", "lodash": "^4.17.21"},
                "devDependencies": {"jest": "^29.0.0"},
            }
        )
    )
    return tmp_path


@pytest.fixture
def es6_project(tmp_path):
    """A project using ES6 module syntax."""
    (tmp_path / "app.js").write_text(
        textwrap.dedent("""\
        import React from 'react';
        import { useState, useEffect } from 'react';
        import * as utils from './utils';

        export function App() {
            const [data, setData] = useState(null);
            useEffect(() => {
                const result = utils.fetchData();
                setData(result);
            }, []);
            return React.createElement('div', null, data);
        }

        export default App;
        """)
    )
    (tmp_path / "utils.js").write_text(
        textwrap.dedent("""\
        export function fetchData() {
            return fetch('/api/data');
        }

        export const processData = (data) => {
            return data.map(item => item.name);
        };
        """)
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "es6-project", "dependencies": {"react": "^18.0.0"}})
    )
    return tmp_path


@pytest.fixture
def ts_project(tmp_path):
    """A TypeScript project."""
    (tmp_path / "app.ts").write_text(
        textwrap.dedent("""\
        import { Request, Response } from 'express';

        interface User {
            name: string;
            email: string;
        }

        export async function getUser(req: Request, res: Response): Promise<User> {
            const id: number = parseInt(req.params.id);
            const user = await findUser(id);
            return user;
        }

        function findUser(id: number): Promise<User> {
            return db.query('SELECT * FROM users WHERE id = ?', [id]);
        }
        """)
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"express": "^4.18.0", "typescript": "^5.0.0"}})
    )
    return tmp_path


@pytest.fixture
def class_project(tmp_path):
    """A project with ES6 classes."""
    (tmp_path / "service.js").write_text(
        textwrap.dedent("""\
        class UserService {
            constructor(db) {
                this.db = db;
            }

            async getUser(id) {
                return this.db.find(id);
            }

            async createUser(data) {
                this.validate(data);
                return this.db.insert(data);
            }

            validate(data) {
                if (!data.name) throw new Error('Name required');
            }
        }

        class AdminService extends UserService {
            async deleteUser(id) {
                return this.db.delete(id);
            }
        }

        module.exports = { UserService, AdminService };
        """)
    )
    (tmp_path / "package.json").write_text(json.dumps({"name": "class-project"}))
    return tmp_path


@pytest.fixture
def project_with_lockfile(tmp_path):
    """A project with package-lock.json."""
    (tmp_path / "index.js").write_text("const express = require('express');\n")
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"express": "^4.18.0"}})
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"dependencies": {"express": "^4.18.0"}},
                    "node_modules/express": {
                        "version": "4.18.2",
                        "dependencies": {
                            "accepts": "~1.3.8",
                            "body-parser": "1.20.1",
                        },
                    },
                    "node_modules/accepts": {"version": "1.3.8"},
                    "node_modules/body-parser": {"version": "1.20.1"},
                },
            }
        )
    )
    return tmp_path


@pytest.fixture
def project_with_node_modules(tmp_path):
    """A project with node_modules installed."""
    (tmp_path / "index.js").write_text("const express = require('express');\n")
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"express": "^4.18.0", "lodash": "^4.17.21"}})
    )
    nm = tmp_path / "node_modules"
    nm.mkdir()
    express_dir = nm / "express"
    express_dir.mkdir()
    (express_dir / "package.json").write_text(
        json.dumps({"name": "express", "version": "4.18.2", "dependencies": {"accepts": "~1.3.8"}})
    )
    lodash_dir = nm / "lodash"
    lodash_dir.mkdir()
    (lodash_dir / "package.json").write_text(
        json.dumps({"name": "lodash", "version": "4.17.21"})
    )
    return tmp_path


@pytest.fixture
def project_with_errors(tmp_path):
    """A project with some files that cause parse issues."""
    (tmp_path / "good.js").write_text("function hello() { return 'hi'; }\n")
    (tmp_path / "binary.js").write_bytes(b"\x00\x01\x02\xff\xfe")
    (tmp_path / "also_good.js").write_text("const x = 42;\n")
    (tmp_path / "package.json").write_text(json.dumps({"name": "error-project"}))
    return tmp_path


# ── Comment Stripping Tests ──


class TestStripJsComments:
    def test_single_line_comment(self):
        assert _strip_js_comments("let x = 1; // comment\n").strip() == "let x = 1;"

    def test_multi_line_comment(self):
        result = _strip_js_comments("let x = 1; /* block\ncomment */ let y = 2;")
        assert "let x = 1;" in result
        assert "let y = 2;" in result
        assert "block" not in result

    def test_preserves_line_numbers(self):
        result = _strip_js_comments("a;\n/* line2\nline3\n*/\nb;")
        lines = result.splitlines()
        # b should still be on line 5 (approx)
        assert lines[0].strip() == "a;"

    def test_string_with_slashes(self):
        result = _strip_js_comments("""const x = "http://example.com";""")
        assert "http://example.com" in result

    def test_single_quoted_string(self):
        result = _strip_js_comments("const x = '// not a comment';")
        assert "// not a comment" in result

    def test_template_literal(self):
        result = _strip_js_comments("const x = `template // not stripped`;")
        assert "template // not stripped" in result

    def test_template_expression(self):
        result = _strip_js_comments("const x = `${a + b}`;")
        assert "${a + b}" in result

    def test_empty_source(self):
        assert _strip_js_comments("") == ""

    def test_only_comments(self):
        result = _strip_js_comments("// all comment\n/* also comment */")
        assert result.strip() == ""


# ── Brace Matching Tests ──


class TestBraceMatching:
    def test_simple_braces(self):
        lines = ["function foo() {", "  return 1;", "}"]
        assert _find_brace_range(lines, 0) == (0, 3)

    def test_nested_braces(self):
        lines = ["function foo() {", "  if (x) {", "    return 1;", "  }", "}"]
        assert _find_brace_range(lines, 0) == (0, 5)

    def test_no_braces(self):
        lines = ["const x = 1;"]
        assert _find_brace_range(lines, 0) == (0, 1)

    def test_brace_depths(self):
        lines = ["function foo() {", "  const x = 1;", "}", "const y = 2;"]
        depths = _compute_brace_depths(lines)
        assert depths[0] == 0  # before function
        assert depths[1] == 1  # inside function
        assert depths[2] == 1  # closing brace line (depth at start)
        assert depths[3] == 0  # after function


# ── Argument Parsing Tests ──


class TestParseJsArgs:
    def test_simple_args(self):
        assert _parse_js_args("a, b, c") == ["a", "b", "c"]

    def test_default_values(self):
        assert _parse_js_args("a, b = 10") == ["a", "b"]

    def test_ts_type_annotations(self):
        assert _parse_js_args("x: number, y: string") == ["x", "y"]

    def test_rest_params(self):
        assert _parse_js_args("...args") == ["args"]

    def test_empty(self):
        assert _parse_js_args("") == []
        assert _parse_js_args("  ") == []

    def test_destructured(self):
        result = _parse_js_args("{ name, age }")
        assert "name" in result


# ── File Analyzer Tests ──


class TestJsFileAnalyzer:
    def _analyze(self, source, file_path="test.js"):
        analyzer = _JsFileAnalyzer(file_path)
        return analyzer.analyze(textwrap.dedent(source))

    # ── Imports ──

    def test_es6_default_import(self):
        result = self._analyze("import React from 'react';\n")
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "react"
        assert "React" in imp.names
        assert imp.is_from_import is True

    def test_es6_named_import(self):
        result = self._analyze("import { useState, useEffect } from 'react';\n")
        assert len(result.imports) == 1
        assert "useState" in result.imports[0].names
        assert "useEffect" in result.imports[0].names

    def test_es6_namespace_import(self):
        result = self._analyze("import * as utils from './utils';\n")
        assert len(result.imports) == 1
        assert result.imports[0].alias == "utils"
        assert result.imports[0].module == "./utils"

    def test_es6_default_and_named_import(self):
        result = self._analyze("import React, { useState } from 'react';\n")
        assert len(result.imports) == 1
        assert "React" in result.imports[0].names
        assert "useState" in result.imports[0].names

    def test_es6_side_effect_import(self):
        result = self._analyze("import './styles.css';\n")
        assert len(result.imports) == 1
        assert result.imports[0].module == "./styles.css"

    def test_commonjs_require(self):
        result = self._analyze("const express = require('express');\n")
        assert len(result.imports) == 1
        imp = result.imports[0]
        assert imp.module == "express"
        assert "express" in imp.names
        assert imp.is_from_import is False

    def test_commonjs_destructured_require(self):
        result = self._analyze("const { Router, json } = require('express');\n")
        assert len(result.imports) == 1
        assert "Router" in result.imports[0].names
        assert "json" in result.imports[0].names

    def test_multiline_named_import(self):
        result = self._analyze("""\
        import {
            useState,
            useEffect,
            useCallback
        } from 'react';
        """)
        assert len(result.imports) == 1
        names = result.imports[0].names
        assert "useState" in names
        assert "useEffect" in names
        assert "useCallback" in names

    # ── Functions ──

    def test_function_declaration(self):
        result = self._analyze("""\
        function hello(name) {
            return 'Hello ' + name;
        }
        """)
        assert len(result.functions) >= 1
        func = next(f for f in result.functions if f.name == "hello")
        assert func.args == ["name"]
        assert func.is_async is False

    def test_async_function(self):
        result = self._analyze("""\
        async function fetchData(url) {
            const res = await fetch(url);
            return res.json();
        }
        """)
        func = next(f for f in result.functions if f.name == "fetchData")
        assert func.is_async is True

    def test_arrow_function(self):
        result = self._analyze("""\
        const greet = (name) => {
            return 'Hello ' + name;
        };
        """)
        func = next((f for f in result.functions if f.name == "greet"), None)
        assert func is not None

    def test_function_expression(self):
        result = self._analyze("""\
        const handler = function processRequest(req, res) {
            res.send('ok');
        };
        """)
        func = next((f for f in result.functions if f.name == "handler"), None)
        assert func is not None

    def test_exported_function(self):
        result = self._analyze("""\
        export function getData() {
            return [];
        }
        """)
        func = next(f for f in result.functions if f.name == "getData")
        assert func is not None

    def test_function_calls_captured(self):
        result = self._analyze("""\
        function main() {
            const data = fetchData();
            const result = process(data);
            console.log(result);
        }
        """)
        func = next(f for f in result.functions if f.name == "main")
        assert "fetchData" in func.calls
        assert "process" in func.calls
        assert "console.log" in func.calls

    # ── Classes ──

    def test_class_declaration(self):
        result = self._analyze("""\
        class UserService {
            constructor(db) {
                this.db = db;
            }
            getUser(id) {
                return this.db.find(id);
            }
        }
        """)
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "UserService"
        assert "constructor" in cls.methods
        assert "getUser" in cls.methods

    def test_class_extends(self):
        result = self._analyze("""\
        class Admin extends User {
            delete() {}
        }
        """)
        cls = result.classes[0]
        assert cls.bases == ["User"]

    def test_methods_are_flagged(self):
        result = self._analyze("""\
        class Foo {
            bar() {}
        }
        """)
        method = next(f for f in result.functions if f.name == "bar")
        assert method.is_method is True
        assert method.class_name == "Foo"

    # ── Global Variables ──

    def test_module_level_const(self):
        result = self._analyze("""\
        const PORT = 3000;
        const DB_URL = 'postgres://localhost';
        function foo() { const x = 1; }
        """)
        names = [g.name for g in result.global_vars]
        assert "PORT" in names
        assert "DB_URL" in names

    def test_no_inner_vars_as_global(self):
        result = self._analyze("""\
        function foo() {
            const inner = 1;
        }
        """)
        global_names = [g.name for g in result.global_vars]
        assert "inner" not in global_names


# ── Ingester Init Tests ──


class TestJsProjectIngesterInit:
    def test_valid_project(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project)
        assert ingester._project_path == simple_js_project.resolve()

    def test_invalid_path(self, tmp_path):
        with pytest.raises(IngestionError, match="does not exist"):
            JsProjectIngester(tmp_path / "nonexistent")

    def test_custom_options(self, simple_js_project):
        ingester = JsProjectIngester(
            simple_js_project,
            installed_packages={"express": "4.18.2"},
            npm_timeout=10.0,
            skip_npm_check=True,
        )
        assert ingester._installed_packages == {"express": "4.18.2"}
        assert ingester._npm_timeout == 10.0
        assert ingester._skip_npm_check is True


# ── File Discovery Tests ──


class TestDiscoverJsFiles:
    def test_finds_js_files(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        files = ingester._discover_js_files()
        names = [f.name for f in files]
        assert "index.js" in names
        assert "utils.js" in names

    def test_excludes_node_modules(self, tmp_path):
        (tmp_path / "app.js").write_text("pass")
        nm = tmp_path / "node_modules" / "express"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("pass")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        files = ingester._discover_js_files()
        assert len(files) == 1

    def test_finds_ts_files(self, tmp_path):
        (tmp_path / "app.ts").write_text("pass")
        (tmp_path / "component.tsx").write_text("pass")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        files = ingester._discover_js_files()
        names = [f.name for f in files]
        assert "app.ts" in names
        assert "component.tsx" in names

    def test_excludes_build_dirs(self, tmp_path):
        (tmp_path / "app.js").write_text("pass")
        build = tmp_path / ".next" / "static"
        build.mkdir(parents=True)
        (build / "chunk.js").write_text("pass")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        files = ingester._discover_js_files()
        assert len(files) == 1

    def test_empty_project(self, empty_project):
        ingester = JsProjectIngester(empty_project, skip_npm_check=True)
        files = ingester._discover_js_files()
        assert files == []

    def test_nested_directories(self, tmp_path):
        src = tmp_path / "src" / "components"
        src.mkdir(parents=True)
        (src / "Button.jsx").write_text("pass")
        (tmp_path / "index.js").write_text("pass")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        files = ingester._discover_js_files()
        assert len(files) == 2


# ── File Parsing Tests ──


class TestParseFile:
    def test_valid_file(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        analysis = ingester._parse_file(simple_js_project / "index.js", "index.js")
        assert analysis.parse_error is None
        assert analysis.lines_of_code > 0
        assert len(analysis.imports) >= 1

    def test_file_not_found(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        analysis = ingester._parse_file(tmp_path / "missing.js", "missing.js")
        assert analysis.parse_error is not None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.js"
        f.write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        analysis = ingester._parse_file(f, "empty.js")
        assert analysis.parse_error is None
        assert analysis.lines_of_code == 0


# ── Package.json Parsing Tests ──


class TestPackageJsonParsing:
    def test_basic(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        deps = ingester._parse_package_json(simple_js_project / "package.json")
        names = [d[0] for d in deps]
        assert "express" in names
        assert "lodash" in names
        assert "jest" in names  # devDependencies

    def test_version_specs(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        deps = ingester._parse_package_json(simple_js_project / "package.json")
        express = next(d for d in deps if d[0] == "express")
        assert express[1] == "^4.18.0"

    def test_no_dependencies(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text(json.dumps({"name": "empty"}))
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        deps = ingester._parse_package_json(f)
        assert deps == []

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "package.json"
        f.write_text("{bad json")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        deps = ingester._parse_package_json(f)
        assert deps == []


# ── Version Detection Tests ──


class TestVersionDetection:
    def test_from_provided_dict(self, simple_js_project):
        ingester = JsProjectIngester(
            simple_js_project,
            installed_packages={"express": "4.18.2"},
            skip_npm_check=True,
        )
        versions = ingester._detect_installed_versions(["express", "lodash"])
        assert versions["express"] == "4.18.2"
        assert "lodash" not in versions

    def test_from_node_modules(self, project_with_node_modules):
        ingester = JsProjectIngester(
            project_with_node_modules, skip_npm_check=True
        )
        versions = ingester._detect_installed_versions(["express", "lodash"])
        assert versions["express"] == "4.18.2"
        assert versions["lodash"] == "4.17.21"

    def test_no_node_modules(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        versions = ingester._detect_installed_versions(["express"])
        assert versions == {}


# ── Dependency Tree Tests ──


class TestDependencyTree:
    def test_from_lockfile_v3(self, project_with_lockfile):
        ingester = JsProjectIngester(project_with_lockfile, skip_npm_check=True)
        tree = ingester._resolve_dependency_tree(["express"])
        assert "express" in tree
        assert "accepts" in tree["express"]
        assert "body-parser" in tree["express"]

    def test_from_lockfile_v1(self, tmp_path):
        (tmp_path / "index.js").write_text("pass\n")
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"express": "^4.18.0"}})
        )
        (tmp_path / "package-lock.json").write_text(
            json.dumps(
                {
                    "lockfileVersion": 1,
                    "dependencies": {
                        "express": {
                            "version": "4.18.2",
                            "requires": {"accepts": "~1.3.8"},
                        }
                    },
                }
            )
        )
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        tree = ingester._resolve_dependency_tree(["express"])
        assert tree["express"] == ["accepts"]

    def test_from_node_modules_fallback(self, project_with_node_modules):
        ingester = JsProjectIngester(
            project_with_node_modules, skip_npm_check=True
        )
        tree = ingester._resolve_dependency_tree(["express", "lodash"])
        assert "express" in tree
        assert "accepts" in tree["express"]
        assert tree["lodash"] == []

    def test_no_lockfile_no_modules(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        tree = ingester._resolve_dependency_tree(["express"])
        assert tree["express"] == []


# ── npm Version Check Tests ──


class TestNpmCheck:
    @patch("mycode.js_ingester.JsProjectIngester._fetch_npm_version")
    def test_successful(self, mock_fetch, simple_js_project):
        mock_fetch.return_value = "4.19.0"
        ingester = JsProjectIngester(simple_js_project)
        latest = ingester._check_latest_versions(["express"])
        assert latest["express"] == "4.19.0"

    @patch("mycode.js_ingester.JsProjectIngester._fetch_npm_version")
    def test_failure_returns_empty(self, mock_fetch, simple_js_project):
        mock_fetch.return_value = None
        ingester = JsProjectIngester(simple_js_project)
        latest = ingester._check_latest_versions(["nonexistent"])
        assert "nonexistent" not in latest

    def test_skip_flag(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        latest = ingester._check_latest_versions(["express"])
        assert latest == {}

    def test_empty_list(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project)
        latest = ingester._check_latest_versions([])
        assert latest == {}


# ── Dependency Extraction (combined) Tests ──


class TestDependencyExtraction:
    def test_extracts_from_package_json(self, simple_js_project):
        ingester = JsProjectIngester(
            simple_js_project, installed_packages={}, skip_npm_check=True
        )
        deps = ingester._extract_dependencies()
        names = [d.name for d in deps]
        assert "express" in names
        assert "lodash" in names

    def test_version_detection(self, project_with_node_modules):
        ingester = JsProjectIngester(
            project_with_node_modules, skip_npm_check=True
        )
        deps = ingester._extract_dependencies()
        express = next(d for d in deps if d.name == "express")
        assert express.installed_version == "4.18.2"
        assert express.is_missing is False

    def test_missing_package_flagged(self, simple_js_project):
        ingester = JsProjectIngester(
            simple_js_project, installed_packages={}, skip_npm_check=True
        )
        deps = ingester._extract_dependencies()
        for dep in deps:
            assert dep.is_missing is True

    @patch("mycode.js_ingester.JsProjectIngester._fetch_npm_version")
    def test_outdated_detection(self, mock_fetch, project_with_node_modules):
        mock_fetch.return_value = "5.0.0"
        ingester = JsProjectIngester(project_with_node_modules)
        deps = ingester._extract_dependencies()
        express = next(d for d in deps if d.name == "express")
        assert express.is_outdated is True


# ── Function Flow Tests ──


class TestFunctionFlowMapping:
    def test_simple_call(self, tmp_path):
        (tmp_path / "app.js").write_text(
            textwrap.dedent("""\
            function helper() { return 42; }
            function main() { const x = helper(); }
            """)
        )
        (tmp_path / "package.json").write_text("{}")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        result = ingester.ingest()
        callers = [f.caller for f in result.function_flows]
        callees = [f.callee for f in result.function_flows]
        assert "app.main" in callers
        assert "app.helper" in callees

    def test_import_resolution(self, es6_project):
        ingester = JsProjectIngester(es6_project, skip_npm_check=True)
        result = ingester.ingest()
        callees = {f.callee for f in result.function_flows}
        # utils.fetchData should be resolved via import * as utils
        assert any("utils" in c and "fetchData" in c for c in callees)

    def test_method_call_resolution(self, class_project):
        ingester = JsProjectIngester(class_project, skip_npm_check=True)
        result = ingester.ingest()
        flows = {(f.caller, f.callee) for f in result.function_flows}
        # createUser calls this.validate which should resolve
        assert any("createUser" in c and "validate" in d for c, d in flows)


# ── Coupling Point Tests ──


class TestCouplingPoints:
    def test_high_fan_in(self, tmp_path):
        (tmp_path / "app.js").write_text(
            textwrap.dedent("""\
            function shared() { return 1; }
            function a() { shared(); }
            function b() { shared(); }
            function c() { shared(); }
            """)
        )
        (tmp_path / "package.json").write_text("{}")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        result = ingester.ingest()
        fan_in = [c for c in result.coupling_points if c.coupling_type == "high_fan_in"]
        assert len(fan_in) >= 1
        assert any("shared" in c.source for c in fan_in)

    def test_no_coupling_simple(self, tmp_path):
        (tmp_path / "app.js").write_text(
            textwrap.dedent("""\
            function a() { return 1; }
            function b() { return 2; }
            """)
        )
        (tmp_path / "package.json").write_text("{}")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        result = ingester.ingest()
        assert result.coupling_points == []


# ── Partial Parsing Tests ──


class TestPartialParsing:
    def test_mixed_valid_and_error(self, project_with_errors):
        ingester = JsProjectIngester(project_with_errors, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_analyzed >= 2  # good.js and also_good.js
        # binary.js should have a parse error or analysis failure
        total = result.files_analyzed + result.files_failed
        assert total >= 2

    def test_all_files_valid(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_failed == 0


# ── Full Ingestion Tests ──


class TestFullIngestion:
    def test_simple_project(self, simple_js_project):
        ingester = JsProjectIngester(
            simple_js_project,
            installed_packages={"express": "4.18.2"},
            skip_npm_check=True,
        )
        result = ingester.ingest()
        assert result.files_analyzed == 2
        assert result.total_lines > 0
        assert len(result.dependencies) >= 1
        paths = [a.file_path for a in result.file_analyses]
        assert "index.js" in paths
        assert "utils.js" in paths

    def test_es6_project(self, es6_project):
        ingester = JsProjectIngester(es6_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 2
        assert len(result.function_flows) > 0

    def test_ts_project(self, ts_project):
        ingester = JsProjectIngester(ts_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 1
        funcs = result.file_analyses[0].functions
        func_names = [f.name for f in funcs]
        assert "getUser" in func_names
        assert "findUser" in func_names

    def test_class_project(self, class_project):
        ingester = JsProjectIngester(class_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 1
        classes = result.file_analyses[0].classes
        class_names = [c.name for c in classes]
        assert "UserService" in class_names
        assert "AdminService" in class_names
        admin = next(c for c in classes if c.name == "AdminService")
        assert admin.bases == ["UserService"]

    def test_empty_project(self, empty_project):
        ingester = JsProjectIngester(empty_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.files_analyzed == 0
        assert "No JavaScript/TypeScript files found" in result.warnings[0]

    def test_project_path_in_result(self, simple_js_project):
        ingester = JsProjectIngester(simple_js_project, skip_npm_check=True)
        result = ingester.ingest()
        assert result.project_path == str(simple_js_project.resolve())

    def test_lockfile_dependency_tree(self, project_with_lockfile):
        ingester = JsProjectIngester(
            project_with_lockfile,
            installed_packages={},
            skip_npm_check=True,
        )
        result = ingester.ingest()
        assert "express" in result.dependency_tree
        assert "accepts" in result.dependency_tree["express"]


# ── File-to-Module Tests ──


class TestFileToModule:
    def test_simple_file(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._file_to_module("app.js") == "app"

    def test_nested_file(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._file_to_module("src/components/Button.jsx") == "src/components/Button"

    def test_index_file(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._file_to_module("src/index.js") == "src"

    def test_root_index(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._file_to_module("index.js") == ""


# ── Module Path Resolution Tests ──


class TestResolveModulePath:
    def test_relative_same_dir(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._resolve_module_path("./utils", "src/app.js") == "src/utils"

    def test_relative_parent_dir(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._resolve_module_path("../utils", "src/lib/app.js") == "src/utils"

    def test_absolute_module(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._resolve_module_path("express", "app.js") == "express"

    def test_strips_extension(self, tmp_path):
        (tmp_path / "x.js").write_text("")
        ingester = JsProjectIngester(tmp_path, skip_npm_check=True)
        assert ingester._resolve_module_path("./utils.js", "app.js") == "utils"
