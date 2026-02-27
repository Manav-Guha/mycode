"""JavaScript/Node.js Project Ingester (C3) — Static analysis and dependency resolution.

Regex-based JS/TS parsing, dependency extraction from package.json and
package-lock.json, function flow mapping, and coupling point identification.
Handles partial parsing gracefully.

Pure Python. No LLM dependency. No Node.js dependency for analysis.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

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
    _is_version_outdated,
    _normalize_package_name,
    _read_text_safe,
)

logger = logging.getLogger(__name__)

# ── Constants ──

JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

JS_DISCOVER_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".netlify",
    ".cache",
    ".vite",
    "coverage",
    ".DS_Store",
    ".claude",
}

# Directory name prefixes to exclude (matched with str.startswith)
JS_DISCOVER_EXCLUDE_PREFIXES = ("pytest-of-",)

# JS keywords that look like function calls but aren't
_JS_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "typeof",
    "instanceof",
    "throw",
    "await",
    "yield",
    "void",
    "delete",
    "class",
    "function",
    "import",
    "export",
    "from",
    "try",
    "finally",
    "else",
    "do",
    "case",
    "super",
}

# ── Import Patterns ──

# import default, { named } from 'module'
_ES6_IMPORT_DEFAULT_NAMED_RE = re.compile(
    r"""import\s+(\w+)\s*,\s*\{([^}]*)\}\s+from\s+['"]([^'"]+)['"]""",
    re.DOTALL,
)

# import default from 'module'
_ES6_IMPORT_DEFAULT_RE = re.compile(
    r"""import\s+(\w+)\s+from\s+['"]([^'"]+)['"]"""
)

# import { named } from 'module'
_ES6_IMPORT_NAMED_RE = re.compile(
    r"""import\s+\{([^}]*)\}\s+from\s+['"]([^'"]+)['"]""",
    re.DOTALL,
)

# import * as name from 'module'
_ES6_IMPORT_NAMESPACE_RE = re.compile(
    r"""import\s+\*\s+as\s+(\w+)\s+from\s+['"]([^'"]+)['"]"""
)

# import 'module'  (side-effect)
_ES6_IMPORT_SIDE_EFFECT_RE = re.compile(
    r"""^\s*import\s+['"]([^'"]+)['"]""", re.MULTILINE
)

# export { ... } from 'module'  (re-export)
_ES6_REEXPORT_RE = re.compile(
    r"""export\s+\{[^}]*\}\s+from\s+['"]([^'"]+)['"]""", re.DOTALL
)

# const x = require('module')
_REQUIRE_CONST_RE = re.compile(
    r"""(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)"""
)

# const { x, y } = require('module')
_REQUIRE_DESTRUCTURE_RE = re.compile(
    r"""(?:const|let|var)\s+\{([^}]*)\}\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    re.DOTALL,
)

# Dynamic import('module') with string literal
_DYNAMIC_IMPORT_LITERAL_RE = re.compile(
    r"""\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)"""
)

# Dynamic import(expr) with non-literal (variable, template, concatenation)
_DYNAMIC_IMPORT_VARIABLE_RE = re.compile(
    r"""\bimport\s*\(\s*(?!['"])"""
)

# require('module') — all forms including bare (side-effect) requires.
# Span dedup skips those already matched by _REQUIRE_CONST_RE / _REQUIRE_DESTRUCTURE_RE.
_REQUIRE_BARE_RE = re.compile(
    r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)"""
)

# require(expr) with non-literal (variable, template, concatenation)
_REQUIRE_VARIABLE_RE = re.compile(
    r"""\brequire\s*\(\s*(?!['"])"""
)

# ── Function Patterns ──

# [export [default]] [async] function name(args)
_JS_FUNC_DECL_RE = re.compile(
    r"(?:export\s+(?:default\s+)?)?"
    r"(async\s+)?function\s+"
    r"(\w+)\s*"
    r"(?:<[^>]*>)?\s*"  # optional TS generics
    r"\(([^)]*)\)",
)

# [export] const/let/var name = [async] (args) =>
_JS_ARROW_FUNC_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+"
    r"(\w+)\s*"
    r"(?::\s*[^=]*?)?\s*"  # optional TS type annotation
    r"=\s*(async\s+)?"
    r"\(([^)]*)\)\s*"
    r"(?::\s*\S[^=]*?)?\s*"  # optional TS return type
    r"=>",
)

# [export] const/let/var name = [async] function(args)
_JS_FUNC_EXPR_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+"
    r"(\w+)\s*=\s*(async\s+)?"
    r"function\s*(?:\w+)?\s*"
    r"\(([^)]*)\)",
)

# Class method: [async] [static] [get|set] name(args) {
_JS_METHOD_RE = re.compile(
    r"^\s+"
    r"(async\s+)?(?:static\s+)?(?:get\s+|set\s+)?"
    r"(?:#)?(\w+)\s*"
    r"(?:<[^>]*>)?\s*"  # optional TS generics
    r"\(([^)]*)\)\s*"
    r"(?::\s*\S[^{]*?)?\s*"  # optional TS return type
    r"\{",
    re.MULTILINE,
)

# ── Class Pattern ──

_JS_CLASS_RE = re.compile(
    r"(?:export\s+(?:default\s+)?)?"
    r"class\s+(\w+)"
    r"(?:\s+extends\s+([\w.]+))?",
)

# ── Variable Pattern (module-level) ──

_JS_VAR_DECL_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]*?)?\s*=",
)

# ── Call Pattern ──

_JS_CALL_RE = re.compile(r"\b([a-zA-Z_$]\w*(?:\.[a-zA-Z_$]\w*)*)\s*\(")


# ── Comment Stripping ──


def _strip_js_comments(source: str) -> str:
    """Remove JS/TS comments while preserving line numbers and string contents."""
    result: list[str] = []
    i = 0
    n = len(source)

    while i < n:
        c = source[i]

        # Single-quoted string
        if c == "'":
            result.append(c)
            i += 1
            while i < n and source[i] != "'":
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    if i < n:
                        result.append(source[i])
                        i += 1
                    continue
                result.append(source[i])
                i += 1
            if i < n:
                result.append(source[i])
                i += 1
            continue

        # Double-quoted string
        if c == '"':
            result.append(c)
            i += 1
            while i < n and source[i] != '"':
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    if i < n:
                        result.append(source[i])
                        i += 1
                    continue
                result.append(source[i])
                i += 1
            if i < n:
                result.append(source[i])
                i += 1
            continue

        # Template literal
        if c == "`":
            result.append(c)
            i += 1
            while i < n and source[i] != "`":
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    if i < n:
                        result.append(source[i])
                        i += 1
                    continue
                # ${...} expression — pass through
                if source[i] == "$" and i + 1 < n and source[i + 1] == "{":
                    result.append(source[i])
                    result.append(source[i + 1])
                    i += 2
                    depth = 1
                    while i < n and depth > 0:
                        if source[i] == "{":
                            depth += 1
                        elif source[i] == "}":
                            depth -= 1
                        result.append(source[i])
                        i += 1
                    continue
                result.append(source[i])
                i += 1
            if i < n:
                result.append(source[i])
                i += 1
            continue

        # Single-line comment
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Multi-line comment
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1:
                if source[i] == "\n":
                    result.append("\n")  # preserve line numbers
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                if i < n and source[i] == "\n":
                    result.append("\n")
                i = n
            continue

        result.append(c)
        i += 1

    return "".join(result)


# ── Brace Matching ──


def _find_brace_range(lines: list[str], start_line: int) -> tuple[int, int]:
    """Find the matching closing brace from start_line (0-indexed). Returns (start, end)."""
    depth = 0
    started = False
    for i in range(start_line, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return (start_line, i + 1)
    return (start_line, min(start_line + 1, len(lines)))


def _compute_brace_depths(lines: list[str]) -> list[int]:
    """Compute brace nesting depth at the start of each line."""
    depths: list[int] = []
    depth = 0
    for line in lines:
        depths.append(depth)
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depths


# ── Argument Parsing ──


def _parse_js_args(args_str: str) -> list[str]:
    """Parse a JS/TS function argument string into argument names."""
    if not args_str or not args_str.strip():
        return []
    args: list[str] = []
    for arg in args_str.split(","):
        arg = arg.strip()
        # Strip default values
        if "=" in arg:
            arg = arg[: arg.index("=")].strip()
        # Strip TS type annotations
        if ":" in arg:
            arg = arg[: arg.index(":")].strip()
        # Handle rest params
        if arg.startswith("..."):
            arg = arg[3:]
        # Handle destructuring — just note the pattern
        if arg.startswith("{") or arg.startswith("["):
            arg = arg.strip("{}[] ")
        arg = arg.strip()
        if arg and arg.isidentifier():
            args.append(arg)
    return args


# ── File Analyzer ──


class _JsFileAnalyzer:
    """Regex-based JavaScript/TypeScript file analyzer."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def analyze(self, source: str) -> FileAnalysis:
        """Analyze a JS/TS source string and return a FileAnalysis."""
        clean = _strip_js_comments(source)
        lines = clean.splitlines()
        brace_depths = _compute_brace_depths(lines)

        # Extract classes first (needed to identify methods)
        classes, class_ranges = self._extract_classes(clean, lines)

        # Extract functions and methods
        functions = self._extract_functions(clean, lines, brace_depths, class_ranges)

        # Assign methods to classes
        for cls in classes:
            cls.methods = [
                f.name for f in functions if f.class_name == cls.name
            ]

        imports = self._extract_imports(clean)
        global_vars = self._extract_global_vars(clean, lines, brace_depths)

        return FileAnalysis(
            file_path=self.file_path,
            functions=functions,
            classes=classes,
            imports=imports,
            global_vars=global_vars,
            lines_of_code=len(source.splitlines()),
        )

    # ── Imports ──

    def _extract_imports(self, source: str) -> list[ImportInfo]:
        imports: list[ImportInfo] = []
        seen_spans: list[tuple[int, int]] = []  # track matched spans to avoid dupes

        def _add(span, imp):
            for s in seen_spans:
                if span[0] >= s[0] and span[0] < s[1]:
                    return
            seen_spans.append(span)
            imports.append(imp)

        # import default, { named } from 'module'
        for m in _ES6_IMPORT_DEFAULT_NAMED_RE.finditer(source):
            default_name = m.group(1)
            named_str = m.group(2)
            module = m.group(3)
            names = [n.strip().split(" as ")[0].strip() for n in named_str.split(",") if n.strip()]
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=module, names=[default_name] + names,
                is_from_import=True, lineno=lineno,
            ))

        # import default from 'module'
        for m in _ES6_IMPORT_DEFAULT_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(2), names=[m.group(1)],
                is_from_import=True, lineno=lineno,
            ))

        # import { named } from 'module'
        for m in _ES6_IMPORT_NAMED_RE.finditer(source):
            named_str = m.group(1)
            names = [n.strip().split(" as ")[0].strip() for n in named_str.split(",") if n.strip()]
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(2), names=names,
                is_from_import=True, lineno=lineno,
            ))

        # import * as name from 'module'
        for m in _ES6_IMPORT_NAMESPACE_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(2), alias=m.group(1),
                is_from_import=True, lineno=lineno,
            ))

        # import 'module'
        for m in _ES6_IMPORT_SIDE_EFFECT_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(1), is_from_import=True, lineno=lineno,
            ))

        # const x = require('module')
        for m in _REQUIRE_CONST_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(2), names=[m.group(1)],
                is_from_import=False, lineno=lineno,
            ))

        # const { x, y } = require('module')
        for m in _REQUIRE_DESTRUCTURE_RE.finditer(source):
            names_str = m.group(1)
            names = [n.strip().split(":")[0].strip() for n in names_str.split(",") if n.strip()]
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(2), names=names,
                is_from_import=False, lineno=lineno,
            ))

        # Dynamic import('module') with string literal
        for m in _DYNAMIC_IMPORT_LITERAL_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(1), is_from_import=True, lineno=lineno,
            ))

        # Bare require('module') — side-effect form without assignment.
        # Span dedup skips those already matched by _REQUIRE_CONST_RE / _REQUIRE_DESTRUCTURE_RE.
        for m in _REQUIRE_BARE_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            _add(m.span(), ImportInfo(
                module=m.group(1), is_from_import=False, lineno=lineno,
            ))

        # Dynamic import(variable) — unresolvable, record for warning
        for m in _DYNAMIC_IMPORT_VARIABLE_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            rest = source[m.end():m.end() + 80]
            expr = rest.split(")")[0].strip() if ")" in rest else rest.split("\n")[0].strip()
            _add(m.span(), ImportInfo(
                module=f"<dynamic: {expr[:60]}>",
                is_from_import=True, lineno=lineno,
            ))

        # require(variable) — unresolvable, record for warning
        for m in _REQUIRE_VARIABLE_RE.finditer(source):
            lineno = source[: m.start()].count("\n") + 1
            rest = source[m.end():m.end() + 80]
            expr = rest.split(")")[0].strip() if ")" in rest else rest.split("\n")[0].strip()
            _add(m.span(), ImportInfo(
                module=f"<dynamic: {expr[:60]}>",
                is_from_import=False, lineno=lineno,
            ))

        return imports

    # ── Classes ──

    def _extract_classes(
        self, source: str, lines: list[str]
    ) -> tuple[list[ClassInfo], dict[str, tuple[int, int]]]:
        """Extract class declarations and their body ranges."""
        classes: list[ClassInfo] = []
        ranges: dict[str, tuple[int, int]] = {}

        for m in _JS_CLASS_RE.finditer(source):
            name = m.group(1)
            base = m.group(2) or ""
            lineno = source[: m.start()].count("\n") + 1
            body_start, body_end = _find_brace_range(lines, lineno - 1)
            classes.append(
                ClassInfo(
                    name=name,
                    file_path=self.file_path,
                    lineno=lineno,
                    end_lineno=body_end,
                    bases=[base] if base else [],
                )
            )
            ranges[name] = (lineno, body_end)

        return classes, ranges

    # ── Functions ──

    def _extract_functions(
        self,
        source: str,
        lines: list[str],
        brace_depths: list[int],
        class_ranges: dict[str, tuple[int, int]],
    ) -> list[FunctionInfo]:
        functions: list[FunctionInfo] = []
        seen_names_at_line: set[tuple[str, int]] = set()

        def _get_class_at(lineno: int) -> Optional[str]:
            for cls_name, (start, end) in class_ranges.items():
                if start <= lineno <= end:
                    return cls_name
            return None

        def _add_func(name, lineno, args_str, is_async, source_match_line):
            key = (name, lineno)
            if key in seen_names_at_line:
                return
            seen_names_at_line.add(key)

            class_name = _get_class_at(lineno)
            is_method = class_name is not None

            body_start, body_end = _find_brace_range(lines, source_match_line)
            calls = self._extract_calls(lines, body_start, body_end)

            functions.append(
                FunctionInfo(
                    name=name,
                    file_path=self.file_path,
                    lineno=lineno,
                    end_lineno=body_end,
                    args=_parse_js_args(args_str),
                    calls=calls,
                    is_method=is_method,
                    class_name=class_name,
                    is_async=is_async,
                )
            )

        # Function declarations
        for m in _JS_FUNC_DECL_RE.finditer(source):
            is_async = m.group(1) is not None
            name = m.group(2)
            args_str = m.group(3)
            lineno = source[: m.start()].count("\n") + 1
            _add_func(name, lineno, args_str, is_async, lineno - 1)

        # Arrow functions
        for m in _JS_ARROW_FUNC_RE.finditer(source):
            name = m.group(1)
            is_async = m.group(2) is not None
            args_str = m.group(3)
            lineno = source[: m.start()].count("\n") + 1
            # Arrow might have block body or expression body
            match_line = source[: m.end()].count("\n")
            _add_func(name, lineno, args_str, is_async, match_line)

        # Function expressions
        for m in _JS_FUNC_EXPR_RE.finditer(source):
            name = m.group(1)
            is_async = m.group(2) is not None
            args_str = m.group(3)
            lineno = source[: m.start()].count("\n") + 1
            _add_func(name, lineno, args_str, is_async, lineno - 1)

        # Class methods
        for m in _JS_METHOD_RE.finditer(source):
            is_async = m.group(1) is not None
            name = m.group(2)
            args_str = m.group(3)
            lineno = source[: m.start()].count("\n") + 1
            class_name = _get_class_at(lineno)
            if class_name is not None:
                _add_func(name, lineno, args_str, is_async, lineno - 1)

        return functions

    # ── Call Extraction ──

    def _extract_calls(
        self, lines: list[str], start: int, end: int
    ) -> list[str]:
        """Extract function call names from a range of lines."""
        calls: list[str] = []
        body = "\n".join(lines[start:end])
        for m in _JS_CALL_RE.finditer(body):
            name = m.group(1)
            # Filter out keywords and common non-function patterns
            root = name.split(".")[0]
            if root not in _JS_KEYWORDS:
                calls.append(name)
        return calls

    # ── Global Variables ──

    def _extract_global_vars(
        self, source: str, lines: list[str], brace_depths: list[int]
    ) -> list[GlobalVarInfo]:
        """Extract module-level variable declarations (brace depth 0)."""
        gvars: list[GlobalVarInfo] = []
        for m in _JS_VAR_DECL_RE.finditer(source):
            lineno_0 = source[: m.start()].count("\n")
            if lineno_0 < len(brace_depths) and brace_depths[lineno_0] == 0:
                name = m.group(1)
                # Skip function/class names already captured
                gvars.append(
                    GlobalVarInfo(
                        name=name,
                        file_path=self.file_path,
                        lineno=lineno_0 + 1,
                    )
                )
        return gvars


# ── Main Ingester ──


class JsProjectIngester:
    """Analyzes a JavaScript/Node.js project: parses code, extracts dependencies, maps data flow.

    Works standalone or with a SessionManager. When used with SessionManager,
    pass ``session.project_copy_dir`` as ``project_path``.

    Usage::

        ingester = JsProjectIngester("/path/to/project")
        result = ingester.ingest()
        print(f"Analyzed {result.files_analyzed} files, found {len(result.dependencies)} deps")
    """

    def __init__(
        self,
        project_path: str | Path,
        installed_packages: Optional[dict[str, str]] = None,
        npm_timeout: float = 5.0,
        skip_npm_check: bool = False,
    ):
        self._project_path = Path(project_path).resolve()
        if not self._project_path.is_dir():
            raise IngestionError(
                f"Project path does not exist or is not a directory: {self._project_path}"
            )

        self._installed_packages = installed_packages
        self._npm_timeout = npm_timeout
        self._skip_npm_check = skip_npm_check

    # ── Public API ──

    def ingest(self) -> IngestionResult:
        """Analyze the project and return complete ingestion results."""
        result = IngestionResult(project_path=str(self._project_path))

        # 1. Discover JS/TS files
        js_files = self._discover_js_files()
        if not js_files:
            result.warnings.append("No JavaScript/TypeScript files found in project")
            return result

        # 2. Parse each file
        for file_path in js_files:
            rel_path = str(file_path.relative_to(self._project_path))
            analysis = self._parse_file(file_path, rel_path)
            result.file_analyses.append(analysis)
            if analysis.parse_error:
                result.files_failed += 1
                result.parse_errors.append(
                    {"file": rel_path, "error": analysis.parse_error}
                )
            else:
                result.files_analyzed += 1
            result.total_lines += analysis.lines_of_code

        # 3. Extract and enrich dependencies
        result.dependencies = self._extract_dependencies()

        # 4. Resolve transitive dependency tree
        dep_names = [d.name for d in result.dependencies]
        result.dependency_tree = self._resolve_dependency_tree(dep_names)

        # 5. Build function flow graph
        successful = [a for a in result.file_analyses if not a.parse_error]
        result.function_flows = self._build_function_flows(successful)

        # 6. Identify coupling points
        result.coupling_points = self._identify_coupling_points(
            successful, result.function_flows
        )

        # 7. Warn about unresolvable dynamic imports
        unresolved_files = []
        for analysis in result.file_analyses:
            if any(imp.module.startswith("<dynamic:") for imp in analysis.imports):
                unresolved_files.append(analysis.file_path)
        if unresolved_files:
            listing = ", ".join(unresolved_files[:5])
            if len(unresolved_files) > 5:
                listing += f" (and {len(unresolved_files) - 5} more)"
            result.warnings.append(
                f"Dynamic imports with variable paths in {len(unresolved_files)} "
                f"file(s) could not be statically resolved: {listing}"
            )

        # 8. Summary
        total = result.files_analyzed + result.files_failed
        if result.files_failed > 0:
            result.warnings.append(
                f"Analyzed {result.files_analyzed} of {total} files; "
                f"{result.files_failed} couldn't be parsed"
            )

        return result

    # ── File Discovery ──

    def _discover_js_files(self) -> list[Path]:
        """Find all JS/TS files in the project, excluding non-project directories."""
        js_files: list[Path] = []
        for ext in sorted(JS_EXTENSIONS):
            pattern = f"*{ext}"
            for path in self._project_path.rglob(pattern):
                parts = path.relative_to(self._project_path).parts
                skip = False
                for part in parts[:-1]:
                    if part in JS_DISCOVER_EXCLUDE_DIRS or part.startswith(JS_DISCOVER_EXCLUDE_PREFIXES):
                        skip = True
                        break
                if not skip:
                    js_files.append(path)
        # Deduplicate (if rglob patterns overlap) and sort
        return sorted(set(js_files))

    # ── File Parsing ──

    def _parse_file(self, file_path: Path, rel_path: str) -> FileAnalysis:
        """Parse a single JS/TS file and extract its structure."""
        analysis = FileAnalysis(file_path=rel_path)

        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                source = file_path.read_text(encoding="latin-1")
            except Exception as e:
                analysis.parse_error = f"Could not read file: {e}"
                return analysis
        except OSError as e:
            analysis.parse_error = f"Could not read file: {e}"
            return analysis

        analysis.lines_of_code = len(source.splitlines())

        try:
            analyzer = _JsFileAnalyzer(rel_path)
            result = analyzer.analyze(source)
            analysis.functions = result.functions
            analysis.classes = result.classes
            analysis.imports = result.imports
            analysis.global_vars = result.global_vars
        except Exception as e:
            analysis.parse_error = f"Analysis failed: {e}"

        return analysis

    # ── Dependency Extraction ──

    def _extract_dependencies(self) -> list[DependencyInfo]:
        """Extract declared dependencies from package.json."""
        seen: dict[str, DependencyInfo] = {}

        pkg_json = self._project_path / "package.json"
        if pkg_json.is_file():
            for name, version_spec, is_dev in self._parse_package_json(pkg_json):
                if name not in seen or (version_spec and not seen[name].required_version):
                    seen[name] = DependencyInfo(
                        name=name,
                        required_version=version_spec or None,
                        source_file="package.json",
                        is_dev=is_dev,
                    )

        deps = list(seen.values())

        # Enrich with installed versions
        installed = self._detect_installed_versions([d.name for d in deps])
        for dep in deps:
            if dep.name in installed:
                dep.installed_version = installed[dep.name]
            else:
                dep.is_missing = True

        # Enrich with latest versions from npm registry
        latest = self._check_latest_versions([d.name for d in deps])
        for dep in deps:
            if dep.name in latest:
                dep.latest_version = latest[dep.name]

            if dep.installed_version and dep.latest_version:
                dep.is_outdated = _is_version_outdated(
                    dep.installed_version, dep.latest_version
                )

        return deps

    def _parse_package_json(self, path: Path) -> list[tuple[str, str, bool]]:
        """Parse package.json for dependencies.

        Returns:
            List of (name, version_spec, is_dev) tuples.  ``is_dev`` is True
            for packages listed under ``devDependencies``.
        """
        deps: list[tuple[str, str, bool]] = []
        try:
            data = json.loads(_read_text_safe(path))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return deps

        for section in ("dependencies", "devDependencies", "peerDependencies"):
            is_dev = section == "devDependencies"
            for name, version in data.get(section, {}).items():
                deps.append((name, version if isinstance(version, str) else "", is_dev))

        return deps

    # ── Version Detection ──

    def _detect_installed_versions(self, dep_names: list[str]) -> dict[str, str]:
        """Check node_modules/<name>/package.json for installed versions."""
        if self._installed_packages is not None:
            return {
                name: self._installed_packages[name]
                for name in dep_names
                if name in self._installed_packages
            }

        versions: dict[str, str] = {}
        node_modules = self._project_path / "node_modules"
        if not node_modules.is_dir():
            return versions

        for name in dep_names:
            pkg_json = node_modules / name / "package.json"
            if pkg_json.is_file():
                try:
                    data = json.loads(_read_text_safe(pkg_json))
                    version = data.get("version", "")
                    if version:
                        versions[name] = version
                except (json.JSONDecodeError, OSError):
                    pass
        return versions

    # ── Transitive Dependency Resolution ──

    def _resolve_dependency_tree(self, root_deps: list[str]) -> dict[str, list[str]]:
        """Build transitive dependency tree from package-lock.json or node_modules."""
        # Try package-lock.json first (most complete)
        lock_path = self._project_path / "package-lock.json"
        if lock_path.is_file():
            return self._resolve_from_lockfile(lock_path, root_deps)

        # Fallback: walk node_modules
        return self._resolve_from_node_modules(root_deps)

    def _resolve_from_lockfile(
        self, lock_path: Path, root_deps: list[str]
    ) -> dict[str, list[str]]:
        """Resolve dependency tree from package-lock.json."""
        tree: dict[str, list[str]] = {}
        try:
            data = json.loads(_read_text_safe(lock_path))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse %s: %s", lock_path, e)
            return tree

        lockfile_version = data.get("lockfileVersion", 1)

        if lockfile_version >= 2:
            # v2/v3: use "packages" key
            packages = data.get("packages", {})
            for dep_name in root_deps:
                key = f"node_modules/{dep_name}"
                pkg_data = packages.get(key, {})
                dep_deps = list(pkg_data.get("dependencies", {}).keys())
                tree[dep_name] = dep_deps
        else:
            # v1: use "dependencies" key
            lock_deps = data.get("dependencies", {})
            for dep_name in root_deps:
                pkg_data = lock_deps.get(dep_name, {})
                dep_deps = list(pkg_data.get("requires", {}).keys())
                tree[dep_name] = dep_deps

        return tree

    def _resolve_from_node_modules(self, root_deps: list[str]) -> dict[str, list[str]]:
        """Resolve dependency tree by reading node_modules package.json files."""
        tree: dict[str, list[str]] = {}
        node_modules = self._project_path / "node_modules"
        if not node_modules.is_dir():
            return {dep: [] for dep in root_deps}

        for dep_name in root_deps:
            pkg_json = node_modules / dep_name / "package.json"
            if pkg_json.is_file():
                try:
                    data = json.loads(_read_text_safe(pkg_json))
                    tree[dep_name] = list(data.get("dependencies", {}).keys())
                except (json.JSONDecodeError, OSError):
                    tree[dep_name] = []
            else:
                tree[dep_name] = []

        return tree

    # ── npm Registry Version Checking ──

    def _check_latest_versions(self, dep_names: list[str]) -> dict[str, str]:
        """Query npm registry for latest stable versions. Parallelized."""
        if self._skip_npm_check or not dep_names:
            return {}

        latest: dict[str, str] = {}
        workers = min(8, len(dep_names))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_name = {
                executor.submit(self._fetch_npm_version, name): name
                for name in dep_names
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    version = future.result()
                    if version:
                        latest[name] = version
                except Exception:
                    pass

        return latest

    def _fetch_npm_version(self, package_name: str) -> Optional[str]:
        """Fetch the latest version of an npm package from the registry."""
        encoded = urllib.parse.quote(package_name, safe="@")
        url = f"https://registry.npmjs.org/{encoded}/latest"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._npm_timeout) as resp:
                data = json.loads(resp.read())
                return data.get("version")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            OSError,
            TimeoutError,
        ) as e:
            logger.debug("Failed to fetch npm version for %s: %s", package_name, e)
            return None

    # ── Function Flow Mapping ──

    def _build_function_flows(
        self, file_analyses: list[FileAnalysis]
    ) -> list[FunctionFlow]:
        """Build call graph edges between functions across the project."""
        flows: list[FunctionFlow] = []

        # For each file, resolve calls using imports and local definitions
        for analysis in file_analyses:
            module = self._file_to_module(analysis.file_path)

            # Build import resolution map
            import_map: dict[str, str] = {}
            for imp in analysis.imports:
                if imp.alias:
                    import_map[imp.alias] = imp.module
                for name in imp.names:
                    import_map[name] = f"{imp.module}.{name}"

            # Local function lookup
            local_funcs: dict[str, str] = {}
            for func in analysis.functions:
                if func.class_name:
                    local_funcs[f"this.{func.name}"] = (
                        f"{module}.{func.class_name}.{func.name}"
                    )
                    local_funcs[f"{func.class_name}.{func.name}"] = (
                        f"{module}.{func.class_name}.{func.name}"
                    )
                else:
                    local_funcs[func.name] = f"{module}.{func.name}"

            # Resolve calls
            for func in analysis.functions:
                if func.class_name:
                    caller = f"{module}.{func.class_name}.{func.name}"
                else:
                    caller = f"{module}.{func.name}"

                for call_name in func.calls:
                    resolved = self._resolve_call(call_name, local_funcs, import_map)
                    flows.append(
                        FunctionFlow(
                            caller=caller,
                            callee=resolved,
                            file_path=analysis.file_path,
                            lineno=func.lineno,
                        )
                    )

        return flows

    def _resolve_call(
        self, call_name: str, local_funcs: dict[str, str], import_map: dict[str, str]
    ) -> str:
        """Resolve a call name using local definitions and imports."""
        if call_name in local_funcs:
            return local_funcs[call_name]

        parts = call_name.split(".")
        if parts[0] in import_map:
            resolved_base = import_map[parts[0]]
            if len(parts) > 1:
                return f"{resolved_base}.{'.'.join(parts[1:])}"
            return resolved_base

        return call_name

    # ── Coupling Point Identification ──

    def _identify_coupling_points(
        self, file_analyses: list[FileAnalysis], flows: list[FunctionFlow]
    ) -> list[CouplingPoint]:
        """Identify points where one component's failure cascades into another."""
        points: list[CouplingPoint] = []

        # 1. High fan-in: functions called by many others
        callee_callers: dict[str, set[str]] = defaultdict(set)
        for flow in flows:
            callee_callers[flow.callee].add(flow.caller)

        for callee, callers in callee_callers.items():
            if len(callers) >= 3:
                points.append(
                    CouplingPoint(
                        source=callee,
                        targets=sorted(callers),
                        coupling_type="high_fan_in",
                        description=(
                            f"'{callee}' is called by {len(callers)} functions — "
                            f"failure here cascades to all callers"
                        ),
                    )
                )

        # 2. Cross-module dependency hubs
        project_modules = {
            self._file_to_module(a.file_path) for a in file_analyses
        }
        module_importers: dict[str, set[str]] = defaultdict(set)
        for analysis in file_analyses:
            src_module = self._file_to_module(analysis.file_path)
            for imp in analysis.imports:
                # Resolve relative imports like ./utils to module names
                target = self._resolve_module_path(imp.module, analysis.file_path)
                if target in project_modules and target != src_module:
                    module_importers[target].add(src_module)

        for target, importers in module_importers.items():
            if len(importers) >= 3:
                points.append(
                    CouplingPoint(
                        source=target,
                        targets=sorted(importers),
                        coupling_type="cross_module_hub",
                        description=(
                            f"Module '{target}' is imported by {len(importers)} "
                            f"project modules — central point of failure"
                        ),
                    )
                )

        return points

    # ── Helpers ──

    def _file_to_module(self, rel_path: str) -> str:
        """Convert a relative file path to a JS module name."""
        path = Path(rel_path)
        parts = list(path.parts)
        stem = path.stem
        if stem == "index":
            parts = parts[:-1]
        else:
            parts[-1] = stem
        return "/".join(parts) if parts else ""

    def _resolve_module_path(self, import_path: str, from_file: str) -> str:
        """Resolve a relative import path to a module name."""
        if not import_path.startswith("."):
            return import_path

        from_dir = str(Path(from_file).parent)
        if from_dir == ".":
            from_dir = ""

        # Resolve relative path
        if import_path.startswith("./"):
            resolved = f"{from_dir}/{import_path[2:]}" if from_dir else import_path[2:]
        elif import_path.startswith("../"):
            parts = from_dir.split("/") if from_dir else []
            import_parts = import_path.split("/")
            while import_parts and import_parts[0] == "..":
                import_parts.pop(0)
                if parts:
                    parts.pop()
            resolved = "/".join(parts + import_parts)
        else:
            resolved = import_path

        # Strip extension if present
        for ext in JS_EXTENSIONS:
            if resolved.endswith(ext):
                resolved = resolved[: -len(ext)]
                break

        return resolved
