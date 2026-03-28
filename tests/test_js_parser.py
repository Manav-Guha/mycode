"""Tests for the TypeScript Compiler API parser (js_parser.js).

Covers: plain JS, TypeScript, JSX, TSX, ES6 imports, CommonJS require,
class declarations, nested functions, empty files, syntax errors,
dynamic imports, export detection, global variables, async functions,
and batch mode.
"""

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mycode.js_ingester import (
    _ast_result_to_file_analysis,
    _check_node_available,
    _check_typescript_available,
    _parse_files_with_ts_ast,
    _JS_PARSER_SCRIPT,
)
from mycode.ingester import (
    ClassInfo,
    FileAnalysis,
    FunctionInfo,
    GlobalVarInfo,
    ImportInfo,
)


# ── Helpers ──


def _parse_source(source: str, file_path: str = "test.js") -> dict:
    """Parse a source string via the AST parser and return the raw result dict."""
    results = _parse_files_with_ts_ast([{
        "file_path": file_path,
        "source": source,
    }])
    assert results is not None, "AST parser returned None"
    assert len(results) == 1
    return results[0]


def _parse_to_analysis(source: str, file_path: str = "test.js", rel_path: str = "test.js") -> FileAnalysis:
    """Parse source and convert to FileAnalysis."""
    raw = _parse_source(source, file_path)
    assert raw["status"] == "ok", f"Parser error: {raw.get('error')}"
    analysis = _ast_result_to_file_analysis(raw, rel_path)
    assert analysis is not None
    return analysis


# ── Prerequisite Check ──


@pytest.fixture(autouse=True, scope="module")
def check_node_and_ts():
    """Skip all tests in this module if Node.js or typescript is not available."""
    if not _check_node_available():
        pytest.skip("Node.js not available")
    if not _check_typescript_available():
        pytest.skip("typescript package not available")


# ── Test: Plain JS Function Declarations ──


class TestPlainJSFunctions:
    def test_function_declaration(self):
        source = "function greet(name) { return 'hello ' + name; }"
        analysis = _parse_to_analysis(source)
        assert len(analysis.functions) == 1
        fn = analysis.functions[0]
        assert fn.name == "greet"
        assert fn.args == ["name"]
        assert fn.lineno == 1
        assert not fn.is_async
        assert not fn.is_method

    def test_arrow_function(self):
        source = "const add = (a, b) => a + b;"
        analysis = _parse_to_analysis(source)
        assert len(analysis.functions) == 1
        fn = analysis.functions[0]
        assert fn.name == "add"
        assert fn.args == ["a", "b"]

    def test_function_expression(self):
        source = "const multiply = function(x, y) { return x * y; };"
        analysis = _parse_to_analysis(source)
        assert len(analysis.functions) == 1
        assert analysis.functions[0].name == "multiply"
        assert analysis.functions[0].args == ["x", "y"]

    def test_async_function(self):
        source = "async function fetchData(url) { return await fetch(url); }"
        analysis = _parse_to_analysis(source)
        fn = analysis.functions[0]
        assert fn.name == "fetchData"
        assert fn.is_async
        assert "fetch" in fn.calls

    def test_async_arrow(self):
        source = "const getData = async (id) => { return await db.find(id); };"
        analysis = _parse_to_analysis(source)
        fn = analysis.functions[0]
        assert fn.name == "getData"
        assert fn.is_async
        assert "db.find" in fn.calls

    def test_call_extraction(self):
        source = textwrap.dedent("""\
            function process(data) {
                const result = transform(data);
                console.log(result);
                return helper.format(result);
            }
        """)
        analysis = _parse_to_analysis(source)
        fn = analysis.functions[0]
        assert "transform" in fn.calls
        assert "console.log" in fn.calls
        assert "helper.format" in fn.calls


# ── Test: TypeScript ──


class TestTypeScript:
    def test_typed_function(self):
        source = textwrap.dedent("""\
            function add(a: number, b: number): number {
                return a + b;
            }
        """)
        analysis = _parse_to_analysis(source, "test.ts")
        fn = analysis.functions[0]
        assert fn.name == "add"
        assert fn.args == ["a", "b"]

    def test_generics(self):
        source = textwrap.dedent("""\
            function identity<T>(value: T): T {
                return value;
            }
        """)
        analysis = _parse_to_analysis(source, "test.ts")
        assert analysis.functions[0].name == "identity"
        assert analysis.functions[0].args == ["value"]

    def test_interface_ignored(self):
        source = textwrap.dedent("""\
            interface User {
                name: string;
                age: number;
            }
            function getUser(id: string): User {
                return db.find(id);
            }
        """)
        analysis = _parse_to_analysis(source, "test.ts")
        assert len(analysis.functions) == 1
        assert analysis.functions[0].name == "getUser"

    def test_complex_type_annotations(self):
        source = textwrap.dedent("""\
            const handler: (req: Request, res: Response) => Promise<void> = async (req, res) => {
                const data = await process(req.body);
                res.json(data);
            };
        """)
        analysis = _parse_to_analysis(source, "test.ts")
        assert len(analysis.functions) == 1
        assert analysis.functions[0].name == "handler"
        assert analysis.functions[0].is_async


# ── Test: JSX ──


class TestJSX:
    def test_jsx_component(self):
        source = textwrap.dedent("""\
            import React from 'react';

            function App({ title }) {
                return <div className="app"><h1>{title}</h1></div>;
            }
        """)
        analysis = _parse_to_analysis(source, "App.jsx")
        assert len(analysis.functions) == 1
        assert analysis.functions[0].name == "App"
        assert analysis.functions[0].args == ["title"]


# ── Test: TSX ──


class TestTSX:
    def test_tsx_component_with_types(self):
        source = textwrap.dedent("""\
            import React, { useState } from 'react';

            interface Props {
                name: string;
                count?: number;
            }

            export const Counter: React.FC<Props> = ({ name, count = 0 }) => {
                const [value, setValue] = useState(count);
                return <button onClick={() => setValue(value + 1)}>{name}: {value}</button>;
            };
        """)
        analysis = _parse_to_analysis(source, "Counter.tsx")
        fns = {f.name for f in analysis.functions}
        assert "Counter" in fns
        counter = [f for f in analysis.functions if f.name == "Counter"][0]
        assert "useState" in counter.calls
        assert "setValue" in counter.calls


# ── Test: ES6 Imports ──


class TestES6Imports:
    def test_default_import(self):
        source = "import React from 'react';"
        analysis = _parse_to_analysis(source)
        assert len(analysis.imports) == 1
        imp = analysis.imports[0]
        assert imp.module == "react"
        assert imp.names == ["React"]
        assert imp.is_from_import

    def test_named_imports(self):
        source = "import { useState, useEffect } from 'react';"
        analysis = _parse_to_analysis(source)
        imp = analysis.imports[0]
        assert "useState" in imp.names
        assert "useEffect" in imp.names

    def test_namespace_import(self):
        source = "import * as utils from './utils';"
        analysis = _parse_to_analysis(source)
        imp = analysis.imports[0]
        assert imp.module == "./utils"
        assert imp.alias == "utils"

    def test_side_effect_import(self):
        source = "import './styles.css';"
        analysis = _parse_to_analysis(source)
        assert len(analysis.imports) == 1
        assert analysis.imports[0].module == "./styles.css"

    def test_default_and_named(self):
        source = "import React, { useState } from 'react';"
        analysis = _parse_to_analysis(source)
        imp = analysis.imports[0]
        assert "React" in imp.names
        assert "useState" in imp.names

    def test_reexport(self):
        source = "export { foo, bar } from './helpers';"
        analysis = _parse_to_analysis(source)
        assert len(analysis.imports) == 1
        assert analysis.imports[0].module == "./helpers"


# ── Test: CommonJS Require ──


class TestCommonJSRequire:
    def test_const_require(self):
        source = "const express = require('express');"
        analysis = _parse_to_analysis(source)
        assert len(analysis.imports) == 1
        imp = analysis.imports[0]
        assert imp.module == "express"
        assert imp.names == ["express"]
        assert not imp.is_from_import

    def test_destructured_require(self):
        source = "const { readFile, writeFile } = require('fs');"
        analysis = _parse_to_analysis(source)
        imp = analysis.imports[0]
        assert imp.module == "fs"
        assert "readFile" in imp.names
        assert "writeFile" in imp.names

    def test_dynamic_require(self):
        source = "const mod = require(dynamicPath);"
        analysis = _parse_to_analysis(source)
        assert len(analysis.imports) == 1
        assert analysis.imports[0].module.startswith("<dynamic:")


# ── Test: Class Declarations ──


class TestClassDeclarations:
    def test_basic_class(self):
        source = textwrap.dedent("""\
            class UserService {
                getUser(id) {
                    return this.db.find(id);
                }
                async saveUser(user) {
                    return this.db.save(user);
                }
            }
        """)
        analysis = _parse_to_analysis(source)
        assert len(analysis.classes) == 1
        cls = analysis.classes[0]
        assert cls.name == "UserService"
        assert "getUser" in cls.methods
        assert "saveUser" in cls.methods

        methods = [f for f in analysis.functions if f.is_method]
        assert len(methods) == 2
        get_user = [f for f in methods if f.name == "getUser"][0]
        assert get_user.class_name == "UserService"
        assert "this.db.find" in get_user.calls

        save_user = [f for f in methods if f.name == "saveUser"][0]
        assert save_user.is_async

    def test_class_extends(self):
        source = textwrap.dedent("""\
            class Admin extends User {
                getRole() { return 'admin'; }
            }
        """)
        analysis = _parse_to_analysis(source)
        cls = analysis.classes[0]
        assert cls.bases == ["User"]

    def test_constructor(self):
        source = textwrap.dedent("""\
            class App {
                constructor(config) {
                    this.config = config;
                    this.init();
                }
            }
        """)
        analysis = _parse_to_analysis(source)
        assert "constructor" in analysis.classes[0].methods
        ctor = [f for f in analysis.functions if f.name == "constructor"][0]
        assert ctor.args == ["config"]
        assert "this.init" in ctor.calls


# ── Test: Nested Functions ──


class TestNestedFunctions:
    def test_calls_in_nested_callbacks(self):
        """Nested calls inside callbacks should be attributed to the outer function."""
        source = textwrap.dedent("""\
            function processItems(items) {
                items.forEach((item) => {
                    transform(item);
                    item.parts.map((part) => {
                        validate(part);
                    });
                });
            }
        """)
        analysis = _parse_to_analysis(source)
        fn = analysis.functions[0]
        assert fn.name == "processItems"
        assert "transform" in fn.calls
        assert "validate" in fn.calls
        assert "items.forEach" in fn.calls


# ── Test: Empty File ──


class TestEmptyFile:
    def test_empty_source(self):
        analysis = _parse_to_analysis("", "empty.js")
        assert analysis.functions == []
        assert analysis.classes == []
        assert analysis.imports == []
        assert analysis.global_vars == []

    def test_whitespace_only(self):
        analysis = _parse_to_analysis("   \n\n  \n", "blank.js")
        assert analysis.functions == []


# ── Test: Syntax Errors ──


class TestSyntaxErrors:
    def test_partial_parse_on_error(self):
        """TS parser is error-recovering — should produce partial results."""
        source = textwrap.dedent("""\
            function valid(x) { return x; }
            function broken( { return;
            const y = 42;
        """)
        result = _parse_source(source)
        assert result["status"] == "ok"
        # Should still find at least the valid function
        names = [f["name"] for f in result["functions"]]
        assert "valid" in names


# ── Test: Dynamic Imports ──


class TestDynamicImports:
    def test_static_dynamic_import(self):
        source = "const mod = await import('lodash');"
        analysis = _parse_to_analysis(source)
        modules = [i.module for i in analysis.imports]
        assert "lodash" in modules

    def test_variable_dynamic_import(self):
        source = "const mod = await import(modulePath);"
        analysis = _parse_to_analysis(source)
        dynamic = [i for i in analysis.imports if i.module.startswith("<dynamic:")]
        assert len(dynamic) == 1


# ── Test: Export Detection ──


class TestExportDetection:
    def test_exported_function_in_raw_output(self):
        source = textwrap.dedent("""\
            export function publicFn() {}
            function privateFn() {}
        """)
        result = _parse_source(source)
        assert result["status"] == "ok"
        exports = result["exports"]
        assert "publicFn" in exports
        assert "privateFn" not in exports

    def test_module_exports(self):
        source = textwrap.dedent("""\
            function a() {}
            function b() {}
            module.exports = { a };
        """)
        result = _parse_source(source)
        assert "a" in result["exports"]


# ── Test: Global Variables ──


class TestGlobalVariables:
    def test_module_level_vars(self):
        source = textwrap.dedent("""\
            const API_URL = 'https://api.example.com';
            let counter = 0;

            function increment() {
                const local = counter + 1;
                counter = local;
            }
        """)
        analysis = _parse_to_analysis(source)
        gvar_names = [g.name for g in analysis.global_vars]
        assert "API_URL" in gvar_names
        assert "counter" in gvar_names
        # 'local' is inside a function, should NOT be a global
        assert "local" not in gvar_names


# ── Test: Batch Mode ──


class TestBatchMode:
    def test_multiple_files(self):
        entries = [
            {"file_path": "a.js", "source": "function foo() {}"},
            {"file_path": "b.ts", "source": "const bar = (x: number) => x;"},
            {"file_path": "c.jsx", "source": "import React from 'react';\nfunction App() { return <div/>; }"},
        ]
        results = _parse_files_with_ts_ast(entries)
        assert results is not None
        assert len(results) == 3
        assert all(r["status"] == "ok" for r in results)
        assert results[0]["functions"][0]["name"] == "foo"
        assert results[1]["functions"][0]["name"] == "bar"
        assert results[2]["functions"][0]["name"] == "App"

    def test_empty_batch(self):
        results = _parse_files_with_ts_ast([])
        assert results == []


# ── Test: ast_result_to_file_analysis conversion ──


class TestResultConversion:
    def test_error_result_returns_none(self):
        result = {"status": "error", "error": "something broke"}
        assert _ast_result_to_file_analysis(result, "test.js") is None

    def test_ok_result_converts(self):
        result = {
            "status": "ok",
            "functions": [{"name": "fn", "lineno": 1, "end_lineno": 3, "args": ["x"],
                           "is_async": True, "is_method": False, "class_name": None,
                           "decorators": ["Log"], "calls": ["process"], "exported": True}],
            "classes": [{"name": "Svc", "lineno": 5, "end_lineno": 10,
                         "bases": ["Base"], "methods": ["run"], "decorators": [],
                         "exported": False}],
            "imports": [{"module": "fs", "names": ["readFile"], "alias": None,
                         "is_from_import": True, "lineno": 1}],
            "global_vars": [{"name": "VERSION", "lineno": 2}],
            "exports": ["fn"],
            "lines_of_code": 12,
        }
        analysis = _ast_result_to_file_analysis(result, "app.js")
        assert analysis is not None
        assert analysis.file_path == "app.js"
        assert analysis.lines_of_code == 12
        assert len(analysis.functions) == 1
        assert analysis.functions[0].is_async
        assert analysis.functions[0].decorators == ["Log"]
        assert len(analysis.classes) == 1
        assert analysis.classes[0].bases == ["Base"]
        assert len(analysis.imports) == 1
        assert analysis.imports[0].module == "fs"
        assert len(analysis.global_vars) == 1
