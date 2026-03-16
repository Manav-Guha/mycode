"""Tests for js_module_loader — Node.js module loading and export discovery."""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from mycode.js_module_loader import (
    ExportedFunction,
    ModuleDiscoveryResult,
    _check_node_available,
    _compile_typescript,
    _find_tsc,
    _run_loader,
    discover_exports,
)

# Skip all tests if Node.js is not available
pytestmark = pytest.mark.skipif(
    not _check_node_available(), reason="Node.js not available"
)


# ── Fixtures ──


@pytest.fixture
def js_dir(tmp_path):
    """Create a temp directory for JS fixture files."""
    return tmp_path


def _write_js(directory, filename, content):
    """Write a JS file and return its absolute path."""
    p = directory / filename
    p.write_text(textwrap.dedent(content))
    return str(p)


# ── ExportedFunction / ModuleDiscoveryResult dataclasses ──


class TestDataclasses:
    def test_exported_function_defaults(self):
        ef = ExportedFunction(name="foo", arity=2)
        assert ef.name == "foo"
        assert ef.arity == 2
        assert ef.is_async is False

    def test_exported_function_async(self):
        ef = ExportedFunction(name="bar", arity=0, is_async=True)
        assert ef.is_async is True

    def test_discovery_result_defaults(self):
        r = ModuleDiscoveryResult()
        assert r.exports == []
        assert r.load_method is None
        assert r.error is None
        assert r.error_type is None

    def test_discovery_result_error(self):
        r = ModuleDiscoveryResult(error="boom", error_type="SyntaxError")
        assert r.error == "boom"
        assert r.error_type == "SyntaxError"


# ── CJS module loading ──


class TestCJSLoading:
    def test_simple_exports(self, js_dir):
        path = _write_js(js_dir, "math_utils.js", """\
            function add(a, b) { return a + b; }
            function multiply(a, b) { return a * b; }
            module.exports = { add, multiply };
        """)
        result = discover_exports(path)
        assert result.error is None
        assert result.load_method == "cjs"
        names = {e.name for e in result.exports}
        assert "add" in names
        assert "multiply" in names

    def test_function_arity(self, js_dir):
        path = _write_js(js_dir, "arity.js", """\
            function zero() {}
            function one(a) {}
            function three(a, b, c) {}
            module.exports = { zero, one, three };
        """)
        result = discover_exports(path)
        assert result.error is None
        by_name = {e.name: e for e in result.exports}
        assert by_name["zero"].arity == 0
        assert by_name["one"].arity == 1
        assert by_name["three"].arity == 3

    def test_async_detection(self, js_dir):
        path = _write_js(js_dir, "async_mod.js", """\
            async function fetchData(url) { return url; }
            function processSync(data) { return data; }
            module.exports = { fetchData, processSync };
        """)
        result = discover_exports(path)
        assert result.error is None
        by_name = {e.name: e for e in result.exports}
        assert by_name["fetchData"].is_async is True
        assert by_name["processSync"].is_async is False

    def test_single_function_export(self, js_dir):
        path = _write_js(js_dir, "single.js", """\
            module.exports = function handler(req, res) { return res; };
        """)
        result = discover_exports(path)
        assert result.error is None
        assert len(result.exports) == 1
        assert result.exports[0].name == "handler"
        assert result.exports[0].arity == 2

    def test_arrow_functions(self, js_dir):
        path = _write_js(js_dir, "arrows.js", """\
            const greet = (name) => `Hello ${name}`;
            const add = (a, b) => a + b;
            module.exports = { greet, add };
        """)
        result = discover_exports(path)
        assert result.error is None
        assert len(result.exports) == 2

    def test_non_function_exports_filtered(self, js_dir):
        path = _write_js(js_dir, "mixed.js", """\
            const VERSION = "1.0";
            const CONFIG = { debug: true };
            function run() {}
            module.exports = { VERSION, CONFIG, run };
        """)
        result = discover_exports(path)
        assert result.error is None
        assert len(result.exports) == 1
        assert result.exports[0].name == "run"

    def test_class_methods_not_enumerated(self, js_dir):
        """Class constructors appear as functions; methods do not."""
        path = _write_js(js_dir, "cls.js", """\
            class MyClass {
                constructor(x) { this.x = x; }
                doWork() { return this.x; }
            }
            module.exports = { MyClass };
        """)
        result = discover_exports(path)
        assert result.error is None
        assert len(result.exports) == 1
        assert result.exports[0].name == "MyClass"

    def test_empty_module(self, js_dir):
        path = _write_js(js_dir, "empty.js", """\
            // nothing exported
        """)
        result = discover_exports(path)
        assert result.error is None
        assert result.exports == []
        assert result.load_method == "cjs"


# ── ESM module loading ──


class TestESMLoading:
    def test_esm_named_exports(self, js_dir):
        path = _write_js(js_dir, "esm_mod.mjs", """\
            export function hello(name) { return `Hello ${name}`; }
            export async function fetchItem(id) { return id; }
        """)
        result = discover_exports(path)
        assert result.error is None
        # Node 24+ can require() ESM, so load_method may be "cjs" or "esm"
        assert result.load_method in ("cjs", "esm")
        by_name = {e.name: e for e in result.exports}
        assert "hello" in by_name
        assert "fetchItem" in by_name
        assert by_name["fetchItem"].is_async is True

    def test_esm_default_export_function(self, js_dir):
        path = _write_js(js_dir, "esm_default.mjs", """\
            export default function main() { return 42; }
        """)
        result = discover_exports(path)
        assert result.error is None
        # The default export should be discoverable
        assert len(result.exports) >= 1
        assert result.exports[0].name == "main"

    def test_esm_with_type_module(self, js_dir):
        """ESM via package.json type:module (common in modern projects)."""
        (js_dir / "package.json").write_text('{"type": "module"}')
        path = _write_js(js_dir, "mod.js", """\
            export function process(data) { return data; }
        """)
        result = discover_exports(path)
        assert result.error is None
        assert len(result.exports) == 1
        assert result.exports[0].name == "process"


# ── Error handling ──


class TestErrorHandling:
    def test_file_not_found(self, js_dir):
        result = discover_exports(str(js_dir / "nonexistent.js"))
        assert result.error is not None
        assert "not found" in result.error.lower() or "FileNotFoundError" == result.error_type

    def test_syntax_error(self, js_dir):
        path = _write_js(js_dir, "bad_syntax.js", """\
            function broken( { return; }
        """)
        result = discover_exports(path)
        assert result.error is not None
        assert result.error_type is not None

    def test_missing_dependency(self, js_dir):
        path = _write_js(js_dir, "needs_dep.js", """\
            const express = require("express");
            module.exports = { start: () => express() };
        """)
        result = discover_exports(path)
        # Should report error cleanly, not crash
        assert result.error is not None
        assert "express" in result.error.lower() or "MODULE_NOT_FOUND" in result.error

    def test_node_not_available(self, js_dir):
        path = _write_js(js_dir, "ok.js", "module.exports = {};")
        result = discover_exports(path, node_path="/nonexistent/node")
        assert result.error is not None
        assert result.error_type == "NodeNotFound"

    def test_timeout(self, js_dir):
        path = _write_js(js_dir, "slow.js", """\
            // Simulate slow module init
            const start = Date.now();
            while (Date.now() - start < 15000) {}
            module.exports = {};
        """)
        result = discover_exports(path, timeout=2)
        assert result.error is not None
        assert "timed out" in result.error.lower()

    def test_circular_require(self, js_dir):
        """Circular requires should not crash — Node handles them gracefully."""
        _write_js(js_dir, "circ_a.js", """\
            const b = require("./circ_b.js");
            function fromA() { return "a"; }
            module.exports = { fromA, bVal: b };
        """)
        _write_js(js_dir, "circ_b.js", """\
            const a = require("./circ_a.js");
            function fromB() { return "b"; }
            module.exports = { fromB, aVal: a };
        """)
        result = discover_exports(str(js_dir / "circ_a.js"))
        # Should succeed (Node resolves circular deps with partial exports)
        assert result.error is None
        names = {e.name for e in result.exports}
        assert "fromA" in names


# ── TypeScript ──


class TestTypeScript:
    def test_ts_without_tsc_returns_error(self, js_dir):
        path = _write_js(js_dir, "app.ts", """\
            export function greet(name: string): string {
                return `Hello ${name}`;
            }
        """)
        with mock.patch("mycode.js_module_loader._find_tsc", return_value=None):
            result = discover_exports(path, project_dir=str(js_dir))
        assert result.error is not None
        assert "tsc" in result.error.lower()
        assert result.error_type == "TypeScriptError"

    def test_find_tsc_project_local(self, js_dir):
        bin_dir = js_dir / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        tsc_path = bin_dir / "tsc"
        tsc_path.write_text("#!/bin/sh\necho fake")
        tsc_path.chmod(0o755)
        assert _find_tsc(str(js_dir)) == str(tsc_path)

    def test_find_tsc_not_found(self, js_dir):
        with mock.patch("shutil.which", return_value=None):
            assert _find_tsc(str(js_dir)) is None

    def test_compile_typescript_no_tsc(self, js_dir):
        path = _write_js(js_dir, "mod.ts", "export const x = 1;")
        with mock.patch("mycode.js_module_loader._find_tsc", return_value=None):
            compiled, error = _compile_typescript(str(js_dir / "mod.ts"))
        assert compiled is None
        assert "tsc" in error.lower()

    @pytest.mark.skipif(
        shutil.which("tsc") is None,
        reason="tsc not available globally",
    )
    def test_compile_and_discover_ts(self, js_dir):
        path = _write_js(js_dir, "utils.ts", """\
            export function double(n: number): number {
                return n * 2;
            }
            export async function fetchName(id: number): Promise<string> {
                return `name_${id}`;
            }
        """)
        result = discover_exports(path, project_dir=str(js_dir))
        assert result.error is None
        by_name = {e.name: e for e in result.exports}
        assert "double" in by_name
        assert by_name["double"].arity == 1
        assert "fetchName" in by_name
        assert by_name["fetchName"].is_async is True

    def test_compile_timeout(self, js_dir):
        path = _write_js(js_dir, "slow.ts", "export const x = 1;")
        with mock.patch("mycode.js_module_loader._find_tsc", return_value="/usr/bin/sleep"):
            compiled, error = _compile_typescript(str(js_dir / "slow.ts"), timeout=1)
        # sleep won't produce output, so either timeout or failure
        assert compiled is None
        assert error is not None


# ── _run_loader edge cases ──


class TestRunLoader:
    def test_invalid_json_from_node(self, js_dir):
        """If the Node script somehow produces non-JSON, handle gracefully."""
        path = _write_js(js_dir, "prints_garbage.js", """\
            console.log("not json");
            module.exports = {};
        """)
        # We can't easily make our loader script produce bad JSON,
        # so test _run_loader with a script that prints garbage
        result = _run_loader(str(js_dir / "prints_garbage.js"), "node", 10)
        # The loader script loads the module first, then outputs JSON.
        # The console.log in the module will pollute stdout.
        # Our loader should still work because it writes its own JSON last.
        # But if it doesn't, we handle the parse error.
        assert result is not None  # Should not crash

    def test_spawn_failure(self, js_dir):
        path = _write_js(js_dir, "ok.js", "module.exports = {};")
        result = _run_loader(str(js_dir / "ok.js"), "/nonexistent/node", 10)
        assert result.error is not None
        assert result.error_type == "OSError"


# ── _check_node_available ──


class TestCheckNodeAvailable:
    def test_real_node(self):
        assert _check_node_available("node") is True

    def test_missing_node(self):
        assert _check_node_available("/nonexistent/node") is False
